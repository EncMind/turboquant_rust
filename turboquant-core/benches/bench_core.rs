//! Benchmarks for TurboQuant core operations.
//!
//! Run with: cargo bench -p turboquant-core
//!
//! Uses seeded synthetic data for reproducible runs.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use turboquant_core::codebook;
use turboquant_core::kv_cache::KvCacheCompressor;
use turboquant_core::outlier::OutlierTurboQuant;
use turboquant_core::polar_quant::PolarQuant;
use turboquant_core::qjl::Qjl;
use turboquant_core::rotation;
use turboquant_core::turboquant::{TurboQuant, TurboQuantMse};
use turboquant_core::utils;

fn random_vectors(d: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..d * n)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect()
}

fn normalize_rows_in_place(data: &mut [f64], d: usize) {
    for row in data.chunks_exact_mut(d) {
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
}

fn mse(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / a.len() as f64
}

// ============================================================
// Walsh-Hadamard Transform
// ============================================================

fn bench_fwht(c: &mut Criterion) {
    let mut group = c.benchmark_group("fwht");
    for n in [64, 128, 256, 512, 1024] {
        let data_input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let mut data = data_input.clone();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                data.copy_from_slice(&data_input);
                rotation::fast_walsh_hadamard_transform(black_box(&mut data)).unwrap();
            });
        });
    }
    group.finish();
}

// ============================================================
// Codebook: nearest centroid lookup
// ============================================================

fn bench_nearest_centroid(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_centroid");
    for (bit_width, d) in [(2, 128), (3, 128), (4, 128), (3, 256)] {
        let centroids = codebook::optimal_centroids(bit_width, d).unwrap();
        let values = random_vectors(d, 1, 42);
        let label = format!("b{bit_width}_d{d}");
        group.throughput(Throughput::Elements(d as u64));
        group.bench_function(&label, |b| {
            b.iter(|| codebook::nearest_centroid_indices(black_box(&values), &centroids).unwrap());
        });
    }
    group.finish();
}

// ============================================================
// Bit packing
// ============================================================

fn bench_pack_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack_bits");
    for d in [128, 512, 2048] {
        let signs: Vec<i8> = (0..d).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();
        group.throughput(Throughput::Elements(d as u64));
        group.bench_with_input(BenchmarkId::from_parameter(d), &signs, |b, s| {
            b.iter(|| utils::pack_bits(black_box(s)));
        });
    }
    group.finish();
}

// ============================================================
// PolarQuant: quantize + dequantize
// ============================================================

fn bench_polar_quant(c: &mut Criterion) {
    let mut group = c.benchmark_group("polar_quant");

    for (bit_width, d) in [(2, 128), (3, 128), (4, 128)] {
        let pq = PolarQuant::new(d, bit_width, 42, true).unwrap();
        let x = random_vectors(d, 1, 99);
        let label = format!("quantize_b{bit_width}_d{d}");
        group.bench_function(&label, |b| {
            b.iter(|| pq.quantize_single(black_box(&x)).unwrap());
        });
    }

    for (bit_width, d) in [(2, 128), (3, 128), (4, 128)] {
        let pq = PolarQuant::new(d, bit_width, 42, true).unwrap();
        let x = random_vectors(d, 1, 99);
        let (indices, norm) = pq.quantize_single(&x).unwrap();
        let label = format!("dequantize_b{bit_width}_d{d}");
        group.bench_function(&label, |b| {
            b.iter(|| pq.dequantize_single(black_box(&indices), norm).unwrap());
        });
    }

    group.finish();
}

// ============================================================
// PolarQuant: batch quantize
// ============================================================

fn bench_polar_quant_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("polar_quant_batch");

    for n in [10, 100, 1000] {
        let d = 128;
        let pq = PolarQuant::new(d, 3, 42, true).unwrap();
        let data = random_vectors(d, n, 99);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("quantize_n", n), &data, |b, data| {
            b.iter(|| pq.quantize_batch(black_box(data), n).unwrap());
        });
    }

    group.finish();
}

// ============================================================
// QJL: quantize + dequantize
// ============================================================

fn bench_qjl(c: &mut Criterion) {
    let mut group = c.benchmark_group("qjl");

    for d in [64, 128, 256] {
        let qjl = Qjl::new(d, 42).unwrap();
        let r = random_vectors(d, 1, 99);
        let label = format!("quantize_d{d}");
        group.bench_function(&label, |b| {
            b.iter(|| qjl.quantize_single(black_box(&r)).unwrap());
        });
    }

    for d in [64, 128, 256] {
        let qjl = Qjl::new(d, 42).unwrap();
        let r = random_vectors(d, 1, 99);
        let (signs, norm) = qjl.quantize_single(&r).unwrap();
        let label = format!("dequantize_d{d}");
        group.bench_function(&label, |b| {
            b.iter(|| qjl.dequantize_single(black_box(&signs), norm).unwrap());
        });
    }

    group.finish();
}

// ============================================================
// Full TurboQuant: single-vector roundtrip
// ============================================================

fn bench_turboquant_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("turboquant_single");

    for bit_width in [2u32, 3, 4] {
        let d = 128;
        let tq = TurboQuant::new(d, bit_width, 42, true).unwrap();
        let x = random_vectors(d, 1, 99);

        let label_q = format!("turbo{bit_width}_quantize");
        group.bench_function(&label_q, |b| {
            b.iter(|| tq.quantize(black_box(&x), 1).unwrap());
        });

        let compressed = tq.quantize(&x, 1).unwrap();
        let label_d = format!("turbo{bit_width}_dequantize");
        group.bench_function(&label_d, |b| {
            b.iter(|| tq.dequantize(black_box(&compressed)).unwrap());
        });

        let label_rt = format!("turbo{bit_width}_roundtrip");
        group.bench_function(&label_rt, |b| {
            b.iter(|| {
                let c = tq.quantize(black_box(&x), 1);
                tq.dequantize(&c.unwrap()).unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================
// Full TurboQuant: batch throughput
// ============================================================

fn bench_turboquant_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("turboquant_batch");
    group.sample_size(10); // fewer samples for expensive benchmarks

    for (bit_width, n) in [(3, 100), (3, 1000), (4, 100), (4, 1000)] {
        let d = 128;
        let tq = TurboQuant::new(d, bit_width, 42, true).unwrap();
        let data = random_vectors(d, n, 99);

        let label = format!("turbo{bit_width}_n{n}_roundtrip");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(&label, |b| {
            b.iter(|| {
                let c = tq.quantize(black_box(&data), n);
                tq.dequantize(&c.unwrap()).unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================
// TurboQuantMSE (V cache path)
// ============================================================

fn bench_turboquant_mse(c: &mut Criterion) {
    let mut group = c.benchmark_group("turboquant_mse");

    for bit_width in [2u32, 3, 4] {
        let d = 128;
        let tqm = TurboQuantMse::new(d, bit_width, 42, true).unwrap();
        let data = random_vectors(d, 100, 99);

        let label = format!("mse_b{bit_width}_n100_roundtrip");
        group.throughput(Throughput::Elements(100));
        group.bench_function(&label, |b| {
            b.iter(|| {
                let result = tqm.quantize(black_box(&data), 100);
                tqm.dequantize(&result.unwrap()).unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================
// Outlier quantizer
// ============================================================

fn bench_outlier(c: &mut Criterion) {
    let mut group = c.benchmark_group("outlier");

    for target_bits in [2.5, 3.5] {
        let d = 128;
        let oq = OutlierTurboQuant::new(d, target_bits, 42).unwrap();
        let data = random_vectors(d, 100, 99);

        let label = format!("outlier_{target_bits:.1}bit_n100_roundtrip");
        group.throughput(Throughput::Elements(100));
        group.bench_function(&label, |b| {
            b.iter(|| {
                let c = oq.quantize(black_box(&data), 100);
                oq.dequantize(&c.unwrap()).unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================
// KV Cache compressor (simulated small model)
// ============================================================

fn bench_kv_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache");
    group.sample_size(10);

    let head_dim = 128;
    let num_layers = 2;
    let num_heads = 4;
    let seq_len = 64;
    let total = num_layers * num_heads * seq_len * head_dim;

    let k_cache = random_vectors(1, total, 42);
    let v_cache = random_vectors(1, total, 99);

    for (k_bits, v_bits) in [(3, 3), (4, 4), (4, 3)] {
        let compressor = KvCacheCompressor::new(head_dim, k_bits, v_bits, 42, true).unwrap();
        let label = format!("compress_k{k_bits}v{v_bits}_L{num_layers}H{num_heads}S{seq_len}");
        group.bench_function(&label, |b| {
            b.iter(|| {
                compressor
                    .compress(
                        black_box(&k_cache),
                        black_box(&v_cache),
                        num_layers,
                        num_heads,
                        seq_len,
                    )
                    .unwrap()
            });
        });

        let compressed = compressor
            .compress(&k_cache, &v_cache, num_layers, num_heads, seq_len)
            .unwrap();
        let label_d = format!("decompress_k{k_bits}v{v_bits}_L{num_layers}H{num_heads}S{seq_len}");
        group.bench_function(&label_d, |b| {
            b.iter(|| compressor.decompress(black_box(&compressed)).unwrap());
        });
    }

    group.finish();
}

// ============================================================
// Quality roundtrip benchmark + one-time MSE snapshot (outside timed path)
// ============================================================

fn bench_quality_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_roundtrip");
    group.sample_size(10);

    let d = 128;
    let n = 1000;
    let mut unit_data = random_vectors(d, n, 42);
    normalize_rows_in_place(&mut unit_data, d);

    for bit_width in [2u32, 3, 4] {
        let tq = TurboQuant::new(d, bit_width, 42, true).unwrap();
        let label = format!("turbo{bit_width}_n{n}_roundtrip");

        // One-shot quality signal, intentionally outside benchmark timing.
        let compressed = tq.quantize(&unit_data, n).unwrap();
        let recon = tq.dequantize(&compressed).unwrap();
        black_box(mse(&unit_data, &recon));

        group.bench_function(&label, |b| {
            b.iter(|| {
                let compressed = tq.quantize(black_box(&unit_data), n).unwrap();
                tq.dequantize(&compressed).unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fwht,
    bench_nearest_centroid,
    bench_pack_bits,
    bench_polar_quant,
    bench_polar_quant_batch,
    bench_qjl,
    bench_turboquant_single,
    bench_turboquant_batch,
    bench_turboquant_mse,
    bench_outlier,
    bench_kv_cache,
    bench_quality_roundtrip,
);
criterion_main!(benches);
