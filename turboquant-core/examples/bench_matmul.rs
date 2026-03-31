/// Benchmark raw matrix-matrix multiply to compare nalgebra vs BLAS.
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::time::Instant;

fn main() {
    let d = 128;
    let n = 1000;
    let mut rng = StdRng::seed_from_u64(42);
    let normal = StandardNormal;

    // Seeded random matrices for reproducible benchmark runs.
    let q = DMatrix::<f64>::from_fn(d, d, |_, _| normal.sample(&mut rng));
    let x = DMatrix::<f64>::from_fn(n, d, |_, _| normal.sample(&mut rng));
    let xt = x.transpose(); // (d x n)

    // Warmup
    let _ = &q * &xt;

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = &q * &xt;
    }
    let elapsed = start.elapsed();
    let us_per_call = elapsed.as_micros() as f64 / iters as f64;
    let vecs_per_sec = n as f64 / (us_per_call / 1e6);

    println!(
        "nalgebra (matrixmultiply): {}x{} @ {}x{} = {:.0} us/call",
        d, d, d, n, us_per_call
    );
    println!("  = {:.0} vectors/sec of rotation", vecs_per_sec);
}
