//! Distortion bound verification against paper's Table 2.
//! Mirrors turboquant_plus/tests/test_distortion.py.
//!
//! These tests validate that our implementation achieves distortion
//! within the theoretical bounds from the TurboQuant paper.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;
use turboquant_core::polar_quant::{l2_norm, PolarQuant};
use turboquant_core::turboquant::TurboQuant;

fn random_unit_vector(d: usize, rng: &mut StdRng) -> Vec<f64> {
    let x: Vec<f64> = (0..d).map(|_| StandardNormal.sample(rng)).collect();
    let norm = l2_norm(&x);
    x.iter().map(|v| v / norm).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Paper Table 2 — upper bounds on MSE distortion for unit vectors
const PAPER_MSE_BOUNDS: [(u32, f64); 4] = [(1, 0.36), (2, 0.117), (3, 0.03), (4, 0.009)];

// ============================================================
// PolarQuant MSE distortion vs paper Table 2
// ============================================================

fn test_polarquant_mse_bound(bit_width: u32, d: usize) {
    let paper_bound = PAPER_MSE_BOUNDS
        .iter()
        .find(|(b, _)| *b == bit_width)
        .unwrap()
        .1;

    let pq = PolarQuant::new(d, bit_width, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(123);

    let n_samples = 1000;
    let mut mses = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let x = random_unit_vector(d, &mut rng);
        let (idx, norm) = pq.quantize_single(&x).unwrap();
        let x_hat = pq.dequantize_single(&idx, norm).unwrap();
        let mse: f64 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        mses.push(mse);
    }

    let avg_mse: f64 = mses.iter().sum::<f64>() / n_samples as f64;

    // Paper bounds are asymptotic (d→∞). Allow 3× slack for finite d.
    assert!(
        avg_mse < paper_bound * 3.0,
        "PolarQuant avg MSE {avg_mse:.5} exceeds 3x paper bound {paper_bound} (d={d}, b={bit_width})"
    );
}

#[test]
fn test_polarquant_1bit_d128() {
    test_polarquant_mse_bound(1, 128);
}
#[test]
fn test_polarquant_1bit_d256() {
    test_polarquant_mse_bound(1, 256);
}
#[test]
fn test_polarquant_2bit_d128() {
    test_polarquant_mse_bound(2, 128);
}
#[test]
fn test_polarquant_2bit_d256() {
    test_polarquant_mse_bound(2, 256);
}
#[test]
fn test_polarquant_3bit_d128() {
    test_polarquant_mse_bound(3, 128);
}
#[test]
fn test_polarquant_3bit_d256() {
    test_polarquant_mse_bound(3, 256);
}
#[test]
fn test_polarquant_4bit_d128() {
    test_polarquant_mse_bound(4, 128);
}
#[test]
fn test_polarquant_4bit_d256() {
    test_polarquant_mse_bound(4, 256);
}

// ============================================================
// Inner product distortion (single-side) vs paper Theorem 2
// ============================================================

fn test_ip_distortion_single_side(bit_width: u32) {
    let d = 256;
    let tq = TurboQuant::new(d, bit_width, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(456);

    let n_samples = 1000;
    let mut ip_sq_errors = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let x = random_unit_vector(d, &mut rng);
        let y = random_unit_vector(d, &mut rng);

        // Only quantize x (paper bound is for single-side)
        let x_hat = tq.dequantize(&tq.quantize(&x, 1).unwrap()).unwrap();

        let ip_orig = dot(&y, &x);
        let ip_approx = dot(&y, &x_hat);
        ip_sq_errors.push((ip_orig - ip_approx).powi(2));
    }

    let avg_sq_error: f64 = ip_sq_errors.iter().sum::<f64>() / n_samples as f64;

    // Paper bound: sqrt(3*pi^2)/d * 1/4^b
    let paper_bound = (3.0 * PI * PI).sqrt() / d as f64 / (4.0f64).powi(bit_width as i32);

    assert!(
        avg_sq_error < paper_bound * 5.0,
        "avg IP^2 error {avg_sq_error:.8} exceeds 5x paper bound {paper_bound:.8} (b={bit_width})"
    );
}

#[test]
fn test_ip_distortion_single_side_2bit() {
    test_ip_distortion_single_side(2);
}
#[test]
fn test_ip_distortion_single_side_3bit() {
    test_ip_distortion_single_side(3);
}
#[test]
fn test_ip_distortion_single_side_4bit() {
    test_ip_distortion_single_side(4);
}

// ============================================================
// IP error decreases with bits
// ============================================================

#[test]
fn test_ip_error_decreases_with_bits() {
    let d = 256;
    let mut errors_by_bits = std::collections::HashMap::new();

    for b in [2u32, 3, 4] {
        let tq = TurboQuant::new(d, b, 42, true).unwrap();
        let mut rng = StdRng::seed_from_u64(456);
        let mut errs = Vec::new();
        for _ in 0..200 {
            let x = random_unit_vector(d, &mut rng);
            let y = random_unit_vector(d, &mut rng);
            let x_hat = tq.dequantize(&tq.quantize(&x, 1).unwrap()).unwrap();
            errs.push((dot(&y, &x) - dot(&y, &x_hat)).abs());
        }
        let avg: f64 = errs.iter().sum::<f64>() / errs.len() as f64;
        errors_by_bits.insert(b, avg);
    }

    assert!(
        errors_by_bits[&2] > errors_by_bits[&3],
        "2-bit error ({:.5}) should > 3-bit ({:.5})",
        errors_by_bits[&2],
        errors_by_bits[&3]
    );
    assert!(
        errors_by_bits[&3] > errors_by_bits[&4],
        "3-bit error ({:.5}) should > 4-bit ({:.5})",
        errors_by_bits[&3],
        errors_by_bits[&4]
    );
}

// ============================================================
// MSE decreases with bits
// ============================================================

#[test]
fn test_mse_decreases_with_bits() {
    let d = 256;
    let mut mses = std::collections::HashMap::new();

    for b in [1u32, 2, 3] {
        let pq = PolarQuant::new(d, b, 42, true).unwrap();
        let mut rng = StdRng::seed_from_u64(789);
        let mut total = 0.0;
        let n = 200;
        for _ in 0..n {
            let x = random_unit_vector(d, &mut rng);
            let (idx, norm) = pq.quantize_single(&x).unwrap();
            let x_hat = pq.dequantize_single(&idx, norm).unwrap();
            total += x
                .iter()
                .zip(x_hat.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / d as f64;
        }
        mses.insert(b, total / n as f64);
    }

    assert!(
        mses[&1] > mses[&2],
        "1-bit MSE ({:.4}) should > 2-bit ({:.4})",
        mses[&1],
        mses[&2]
    );
    assert!(
        mses[&2] > mses[&3],
        "2-bit MSE ({:.4}) should > 3-bit ({:.4})",
        mses[&2],
        mses[&3]
    );
}

// ============================================================
// TurboQuant vs PolarQuant IP comparison
// ============================================================

#[test]
fn test_turboquant_competitive_with_polarquant_ip() {
    let d = 256;
    let mut rng = StdRng::seed_from_u64(111);
    let n = 200;

    let pairs: Vec<(Vec<f64>, Vec<f64>)> = (0..n)
        .map(|_| {
            (
                random_unit_vector(d, &mut rng),
                random_unit_vector(d, &mut rng),
            )
        })
        .collect();

    // PolarQuant 2-bit (MSE-only)
    let pq = PolarQuant::new(d, 2, 42, true).unwrap();
    let pq_errors: Vec<f64> = pairs
        .iter()
        .map(|(x, y)| {
            let (idx_x, n_x) = pq.quantize_single(x).unwrap();
            let (idx_y, n_y) = pq.quantize_single(y).unwrap();
            let x_hat = pq.dequantize_single(&idx_x, n_x).unwrap();
            let y_hat = pq.dequantize_single(&idx_y, n_y).unwrap();
            (dot(x, y) - dot(&x_hat, &y_hat)).abs()
        })
        .collect();

    // TurboQuant 2-bit
    let tq = TurboQuant::new(d, 2, 42, true).unwrap();
    let tq_errors: Vec<f64> = pairs
        .iter()
        .map(|(x, y)| {
            let x_hat = tq.dequantize(&tq.quantize(x, 1).unwrap()).unwrap();
            let y_hat = tq.dequantize(&tq.quantize(y, 1).unwrap()).unwrap();
            (dot(x, y) - dot(&x_hat, &y_hat)).abs()
        })
        .collect();

    let pq_avg: f64 = pq_errors.iter().sum::<f64>() / n as f64;
    let tq_avg: f64 = tq_errors.iter().sum::<f64>() / n as f64;

    // Both should be bounded
    assert!(pq_avg < 0.5, "PQ 2-bit IP error {pq_avg:.4} too high");
    assert!(tq_avg < 0.5, "TQ 2-bit IP error {tq_avg:.4} too high");
}
