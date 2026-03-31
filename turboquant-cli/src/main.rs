//! TurboQuant CLI — standalone demo and benchmark tool.
//!
//! Usage:
//!   turboquant demo          — Run a quick compression demo
//!   turboquant bench          — Run performance benchmarks
//!   turboquant info           — Print compression ratio table

use clap::{Parser, Subcommand};
use std::time::Instant;

use turboquant_core::turboquant::TurboQuant;

#[derive(Parser)]
#[command(
    name = "turboquant",
    version,
    about = "TurboQuant KV cache compression CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a quick compression demo showing quality metrics
    Demo {
        /// Vector dimension
        #[arg(short, long, default_value = "128")]
        dim: usize,

        /// Bit width (2=turbo2, 3=turbo3, 4=turbo4)
        #[arg(short, long, default_value = "3")]
        bits: u32,

        /// Number of test vectors
        #[arg(short, long, default_value = "100")]
        count: usize,
    },

    /// Run performance benchmarks
    Bench {
        /// Vector dimension
        #[arg(short, long, default_value = "128")]
        dim: usize,

        /// Number of vectors to benchmark
        #[arg(short, long, default_value = "10000")]
        count: usize,
    },

    /// Output JSON benchmark results (for programmatic comparison)
    BenchJson {
        /// Vector dimension
        #[arg(short, long, default_value = "128")]
        dim: usize,

        /// Number of vectors
        #[arg(short, long, default_value = "1000")]
        count: usize,

        /// Number of primitive iterations (FWHT, centroid, pack_bits)
        #[arg(short, long, default_value = "5000")]
        iter: usize,

        /// Random seed for synthetic benchmark inputs
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Print compression ratio table
    Info {
        /// Vector dimension
        #[arg(short, long, default_value = "128")]
        dim: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Demo { dim, bits, count } => run_demo(dim, bits, count),
        Commands::Bench { dim, count } => run_bench(dim, count),
        Commands::BenchJson {
            dim,
            count,
            iter: n_iter,
            seed,
        } => run_bench_json(dim, count, n_iter, seed),
        Commands::Info { dim } => print_info(dim),
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let len = values.len();
    if len % 2 == 0 {
        (values[len / 2 - 1] + values[len / 2]) / 2.0
    } else {
        values[len / 2]
    }
}

fn run_demo(dim: usize, bits: u32, count: usize) {
    println!("TurboQuant Demo");
    println!("===============");
    println!("Dimension: {dim}, Bits: {bits}, Vectors: {count}\n");

    let tq = TurboQuant::new(dim, bits, 42, true).unwrap();

    // Generate random-ish test vectors
    let mut vectors: Vec<f64> = Vec::with_capacity(count * dim);
    for i in 0..count {
        for j in 0..dim {
            let val = ((i * dim + j) as f64 * 0.01).sin() * (1.0 + (i as f64 * 0.1).cos());
            vectors.push(val);
        }
    }

    // Compress
    let start = Instant::now();
    let compressed = tq.quantize(&vectors, count).unwrap();
    let compress_time = start.elapsed();

    // Decompress
    let start = Instant::now();
    let reconstructed = tq.dequantize(&compressed).unwrap();
    let decompress_time = start.elapsed();

    // Compute quality metrics
    let mut total_mse = 0.0;
    let mut total_ip_error = 0.0;
    let mut n_pairs = 0;

    for i in 0..count {
        let x = &vectors[i * dim..(i + 1) * dim];
        let x_hat = &reconstructed[i * dim..(i + 1) * dim];

        // MSE
        let mse: f64 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / dim as f64;
        total_mse += mse;

        // Inner product error (compare with next vector if available)
        if i + 1 < count {
            let y = &vectors[(i + 1) * dim..(i + 2) * dim];
            let y_hat = &reconstructed[(i + 1) * dim..(i + 2) * dim];

            let ip_orig: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            let ip_recon: f64 = x_hat.iter().zip(y_hat.iter()).map(|(a, b)| a * b).sum();

            if ip_orig.abs() > 1e-10 {
                total_ip_error += ((ip_recon - ip_orig) / ip_orig).abs();
                n_pairs += 1;
            }
        }
    }

    let avg_mse = total_mse / count as f64;
    let avg_ip_error = if n_pairs > 0 {
        total_ip_error / n_pairs as f64
    } else {
        0.0
    };
    let ratio = tq.compression_ratio(16);

    println!("Results:");
    println!("  Compression ratio:    {ratio:.2}x (vs fp16)");
    println!("  Average MSE:          {avg_mse:.6}");
    println!(
        "  Avg IP relative error: {avg_ip_error:.4} ({:.1}%)",
        avg_ip_error * 100.0
    );
    println!(
        "  Compress time:        {compress_time:.2?} ({:.0} vec/s)",
        count as f64 / compress_time.as_secs_f64()
    );
    println!(
        "  Decompress time:      {decompress_time:.2?} ({:.0} vec/s)",
        count as f64 / decompress_time.as_secs_f64()
    );
}

fn run_bench(dim: usize, count: usize) {
    println!("TurboQuant Benchmark");
    println!("====================");
    println!("Dimension: {dim}, Vectors: {count}\n");

    // Generate test data
    let vectors: Vec<f64> = (0..count * dim)
        .map(|i| ((i as f64) * 0.0031415).sin())
        .collect();

    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>10}",
        "Format", "Compress", "Decompress", "Roundtrip", "Ratio"
    );
    println!("{}", "-".repeat(60));

    const LATENCY_WARMUP: usize = 2;
    const LATENCY_SAMPLES: usize = 9;

    for bits in [2u32, 3, 4] {
        let tq = TurboQuant::new(dim, bits, 42, true).unwrap();

        for _ in 0..LATENCY_WARMUP {
            let compressed = tq.quantize(&vectors, count).unwrap();
            let _ = tq.dequantize(&compressed).unwrap();
        }

        let mut compress_samples_us = Vec::with_capacity(LATENCY_SAMPLES);
        let mut decompress_samples_us = Vec::with_capacity(LATENCY_SAMPLES);
        let mut roundtrip_samples_us = Vec::with_capacity(LATENCY_SAMPLES);
        for _ in 0..LATENCY_SAMPLES {
            let start = Instant::now();
            let compressed = tq.quantize(&vectors, count).unwrap();
            let compress_us = start.elapsed().as_secs_f64() * 1e6;

            let start = Instant::now();
            let _ = tq.dequantize(&compressed).unwrap();
            let decompress_us = start.elapsed().as_secs_f64() * 1e6;

            compress_samples_us.push(compress_us);
            decompress_samples_us.push(decompress_us);
            roundtrip_samples_us.push(compress_us + decompress_us);
        }

        let compress_us = median(compress_samples_us);
        let decompress_us = median(decompress_samples_us);
        let total_us = median(roundtrip_samples_us);
        let ratio = tq.compression_ratio(16);

        println!(
            "turbo{bits:<5} {:>9.1} us {:>9.1} us {:>9.1} us {:>8.2}x",
            compress_us, decompress_us, total_us, ratio
        );
    }

    println!("\nThroughput (vectors/second):");
    const THROUGHPUT_WARMUP: usize = 3;
    const THROUGHPUT_SAMPLES: usize = 9;
    for bits in [2u32, 3, 4] {
        let tq = TurboQuant::new(dim, bits, 42, true).unwrap();

        // Warmup full roundtrips to stabilize cache and branch predictor state.
        for _ in 0..THROUGHPUT_WARMUP {
            let c = tq.quantize(&vectors, count).unwrap();
            let _ = tq.dequantize(&c).unwrap();
        }

        // Measure: median of multiple runs, more stable than best-of-N.
        let mut elapsed_samples = Vec::with_capacity(THROUGHPUT_SAMPLES);
        for _ in 0..THROUGHPUT_SAMPLES {
            let start = Instant::now();
            let compressed = tq.quantize(&vectors, count).unwrap();
            let _ = tq.dequantize(&compressed).unwrap();
            elapsed_samples.push(start.elapsed().as_secs_f64());
        }
        let median_elapsed = median(elapsed_samples);
        println!(
            "  turbo{bits}: {:.0} vec/s (encode+decode, median of {THROUGHPUT_SAMPLES})",
            count as f64 / median_elapsed
        );
    }
}

/// JSON benchmark aligned with Python+Rust benchmark path.
/// Uses seeded random-normal inputs and the packed wire-format
/// `quantize` / `dequantize` roundtrip.
fn run_bench_json(dim: usize, count: usize, n_iter: usize, seed: u64) {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    use turboquant_core::codebook;
    use turboquant_core::rotation;
    use turboquant_core::utils;

    fn draw_normal_vec(rng: &mut StdRng, len: usize) -> Vec<f64> {
        let normal = StandardNormal;
        (0..len).map(|_| normal.sample(rng)).collect()
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Collect all results
    print!("{{");

    // --- Primitives ---
    {
        let x_input = draw_normal_vec(&mut rng, dim);
        let mut x = x_input.clone();
        let start = Instant::now();
        for _ in 0..n_iter {
            x.copy_from_slice(&x_input);
            rotation::fast_walsh_hadamard_transform(&mut x).unwrap();
        }
        let fwht_us = start.elapsed().as_secs_f64() / n_iter as f64 * 1e6;
        print!("\"fwht_us\":{fwht_us:.2}");
    }

    {
        let centroids = codebook::optimal_centroids(3, dim).unwrap();
        let values = draw_normal_vec(&mut rng, dim);
        let start = Instant::now();
        for _ in 0..n_iter {
            let _ = codebook::nearest_centroid_indices(&values, &centroids).unwrap();
        }
        let us = start.elapsed().as_secs_f64() / n_iter as f64 * 1e6;
        print!(",\"centroid_us\":{us:.2}");
    }

    {
        let normal = StandardNormal;
        let signs: Vec<i8> = (0..dim)
            .map(|_| {
                let v: f64 = normal.sample(&mut rng);
                if v >= 0.0 {
                    1
                } else {
                    -1
                }
            })
            .collect();
        let start = Instant::now();
        for _ in 0..n_iter {
            let _ = utils::pack_bits(&signs);
        }
        let us = start.elapsed().as_secs_f64() / n_iter as f64 * 1e6;
        print!(",\"pack_bits_us\":{us:.2}");
    }

    // --- Per bit-width: single-vector + batch ---
    let single_vectors: Vec<Vec<f64>> = (0..3).map(|_| draw_normal_vec(&mut rng, dim)).collect();
    let batch_vectors: Vec<Vec<f64>> = (0..3)
        .map(|_| draw_normal_vec(&mut rng, count * dim))
        .collect();

    for (idx, bits) in [2u32, 3, 4].iter().copied().enumerate() {
        let tq = TurboQuant::new(dim, bits, seed, true).unwrap();
        const SINGLE_WARMUP: usize = 1;
        const SINGLE_SAMPLES: usize = 5;
        const BATCH_WARMUP: usize = 2;
        const BATCH_SAMPLES: usize = 7;

        // Single-vector roundtrip — unpacked (matches pure Python: no bit packing)
        {
            let x = &single_vectors[idx];

            for _ in 0..SINGLE_WARMUP {
                for _ in 0..count {
                    let c = tq.quantize_unpacked(x, 1).unwrap();
                    let _ = tq.dequantize_unpacked(&c).unwrap();
                }
            }

            let mut single_samples_us = Vec::with_capacity(SINGLE_SAMPLES);
            for _ in 0..SINGLE_SAMPLES {
                let start = Instant::now();
                for _ in 0..count {
                    let c = tq.quantize_unpacked(x, 1).unwrap();
                    let _ = tq.dequantize_unpacked(&c).unwrap();
                }
                single_samples_us.push(start.elapsed().as_secs_f64() / count as f64 * 1e6);
            }
            let us = median(single_samples_us);
            print!(",\"turbo{bits}_single_us\":{us:.1}");
        }

        // Batch roundtrip — unpacked (matches pure Python: no bit packing)
        {
            let x_batch = &batch_vectors[idx];
            for _ in 0..BATCH_WARMUP {
                let c = tq.quantize_unpacked(x_batch, count).unwrap();
                let _ = tq.dequantize_unpacked(&c).unwrap();
            }

            let mut elapsed_samples = Vec::with_capacity(BATCH_SAMPLES);
            for _ in 0..BATCH_SAMPLES {
                let start = Instant::now();
                let c = tq.quantize_unpacked(x_batch, count).unwrap();
                let _ = tq.dequantize_unpacked(&c).unwrap();
                elapsed_samples.push(start.elapsed().as_secs_f64());
            }
            let vps = count as f64 / median(elapsed_samples);
            print!(",\"turbo{bits}_batch_vps\":{vps:.0}");
        }
    }

    println!("}}");
}

fn print_info(dim: usize) {
    println!("TurboQuant Compression Info");
    println!("==========================");
    println!("Head dimension: {dim}\n");

    println!(
        "{:<10} {:>10} {:>12} {:>15}",
        "Format", "Bits/val", "Ratio vs fp16", "KV MB @32K/32H/32L"
    );
    println!("{}", "-".repeat(54));

    let seq_len = 32768;
    let num_layers = 32;
    let num_heads = 32;
    let n_vecs = num_layers * num_heads * seq_len;
    let original_kv_bytes = n_vecs * dim * 2 * 2; // fp16 K + fp16 V

    // Baselines
    let f16_mb = original_kv_bytes as f64 / 1024.0 / 1024.0;
    println!(
        "{:<10} {:>10} {:>12.2}x {:>12.1} MB",
        "f16", 16, 1.0, f16_mb
    );
    let q8_bytes = n_vecs * dim * 2; // q8 K + q8 V
    println!(
        "{:<10} {:>10} {:>12.2}x {:>12.1} MB",
        "q8_0",
        8,
        original_kv_bytes as f64 / q8_bytes as f64,
        q8_bytes as f64 / 1024.0 / 1024.0
    );

    // TurboQuant formats (wire-size metrics)
    for bits in [4u32, 3, 2] {
        let compressor =
            turboquant_core::KvCacheCompressor::new(dim, bits, bits, 42, true).unwrap();
        let stats = compressor.memory_stats(seq_len, num_layers, num_heads);
        let name = format!("turbo{bits}");
        println!(
            "{name:<10} {:>10} {:>12.2}x {:>12.1} MB",
            bits, stats.wire_compression_ratio, stats.wire_compressed_mb
        );
    }
}
