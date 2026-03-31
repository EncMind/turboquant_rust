//! Tests for PolarQuant (Algorithm 1).
//! Mirrors turboquant_plus/tests/test_polar_quant.py.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turboquant_core::polar_quant::*;

fn random_unit_vector(d: usize, rng: &mut StdRng) -> Vec<f64> {
    let x: Vec<f64> = (0..d).map(|_| StandardNormal.sample(rng)).collect();
    let norm = l2_norm(&x);
    x.iter().map(|v| v / norm).collect()
}

fn random_vector(d: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..d).map(|_| StandardNormal.sample(rng)).collect()
}

// ============================================================
// Round-trip MSE bounds (Paper Table 2)
// ============================================================

#[test]
fn test_1bit_mse_within_bounds() {
    test_mse_bounds(1, 128, 0.36);
}

#[test]
fn test_2bit_mse_within_bounds() {
    test_mse_bounds(2, 128, 0.117);
}

#[test]
fn test_3bit_mse_within_bounds() {
    test_mse_bounds(3, 128, 0.03);
}

#[test]
fn test_2bit_mse_d64() {
    test_mse_bounds(2, 64, 0.117);
}

#[test]
fn test_2bit_mse_d256() {
    test_mse_bounds(2, 256, 0.117);
}

fn test_mse_bounds(bit_width: u32, d: usize, paper_bound: f64) {
    let pq = PolarQuant::new(d, bit_width, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(99);

    let n_samples = 500;
    let mut mse_total = 0.0;
    for _ in 0..n_samples {
        let x = random_unit_vector(d, &mut rng);
        let (indices, norm) = pq.quantize_single(&x).unwrap();
        let x_hat = pq.dequantize_single(&indices, norm).unwrap();
        let mse: f64 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        mse_total += mse;
    }

    let avg_mse = mse_total / n_samples as f64;
    assert!(
        avg_mse < paper_bound * 2.0,
        "MSE {avg_mse:.5} exceeds 2x paper bound {paper_bound} at d={d}, b={bit_width}"
    );
}

// ============================================================
// Zero vector
// ============================================================

#[test]
fn test_zero_vector() {
    let pq = PolarQuant::new(128, 2, 42, true).unwrap();
    let x = vec![0.0; 128];
    let (indices, norm) = pq.quantize_single(&x).unwrap();
    assert!(norm.abs() < 1e-15);
    let x_hat = pq.dequantize_single(&indices, norm).unwrap();
    for v in &x_hat {
        assert!(v.abs() < 1e-15);
    }
}

// ============================================================
// Non-unit-norm vectors (real KV cache)
// ============================================================

#[test]
fn test_non_unit_norm_vectors() {
    let d = 128;
    let pq = PolarQuant::new(d, 3, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    for scale in [0.01, 1.0, 10.0, 50.0, 100.0] {
        let x: Vec<f64> = random_vector(d, &mut rng)
            .iter()
            .map(|v| v * scale)
            .collect();
        let (indices, norm) = pq.quantize_single(&x).unwrap();
        let x_hat = pq.dequantize_single(&indices, norm).unwrap();

        let norm_sq_per_d = l2_norm(&x).powi(2) / d as f64;
        if norm_sq_per_d > 1e-10 {
            let mse: f64 = x
                .iter()
                .zip(x_hat.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / d as f64;
            let relative_mse = mse / norm_sq_per_d;
            assert!(
                relative_mse < 0.5,
                "relative MSE {relative_mse:.4} too high at scale={scale}"
            );
        }
    }
}

// ============================================================
// Deterministic
// ============================================================

#[test]
fn test_deterministic() {
    let d = 128;
    let x: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) * 0.01).collect();

    let pq1 = PolarQuant::new(d, 2, 42, true).unwrap();
    let pq2 = PolarQuant::new(d, 2, 42, true).unwrap();

    let (idx1, n1) = pq1.quantize_single(&x).unwrap();
    let (idx2, n2) = pq2.quantize_single(&x).unwrap();
    assert_eq!(idx1, idx2);
    assert!((n1 - n2).abs() < 1e-15);
}

// ============================================================
// Batch matches single
// ============================================================

#[test]
fn test_batch_matches_single() {
    let d = 128;
    let pq = PolarQuant::new(d, 2, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let n = 10;

    let vectors: Vec<Vec<f64>> = (0..n).map(|_| random_vector(d, &mut rng)).collect();
    let flat: Vec<f64> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

    let batch = pq.quantize_batch(&flat, n).unwrap();
    let batch_recon = pq.dequantize_batch(&batch).unwrap();

    for i in 0..n {
        let (single_idx, single_norm) = pq.quantize_single(&vectors[i]).unwrap();
        let single_recon = pq.dequantize_single(&single_idx, single_norm).unwrap();

        assert_eq!(
            &batch.indices[i * d..(i + 1) * d],
            &single_idx[..],
            "indices mismatch at row {i}"
        );
        for j in 0..d {
            assert!(
                (batch_recon[i * d + j] - single_recon[j]).abs() < 1e-12,
                "recon mismatch at [{i},{j}]"
            );
        }
    }
}

// ============================================================
// Indices in range
// ============================================================

#[test]
fn test_indices_in_range() {
    let d = 256;
    let pq = PolarQuant::new(d, 3, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(5);
    let x = random_vector(d, &mut rng);
    let (indices, _) = pq.quantize_single(&x).unwrap();
    for &idx in &indices {
        assert!(idx < (1 << 3), "index {idx} out of range for 3-bit");
    }
}

// ============================================================
// Norm correction
// ============================================================

#[test]
fn test_norm_correction_preserves_unit_norm_better() {
    let d = 128;
    let mut rng = StdRng::seed_from_u64(11);
    let x = random_unit_vector(d, &mut rng);

    let pq_uncorrected = PolarQuant::new(d, 3, 42, false).unwrap();
    let pq_corrected = PolarQuant::new(d, 3, 42, true).unwrap();

    let (idx_u, n_u) = pq_uncorrected.quantize_single(&x).unwrap();
    let (idx_c, n_c) = pq_corrected.quantize_single(&x).unwrap();

    let x_hat_u = pq_uncorrected.dequantize_single(&idx_u, n_u).unwrap();
    let x_hat_c = pq_corrected.dequantize_single(&idx_c, n_c).unwrap();

    let err_u = (l2_norm(&x_hat_u) - 1.0).abs();
    let err_c = (l2_norm(&x_hat_c) - 1.0).abs();
    assert!(
        err_c < err_u,
        "norm correction should help: err_c={err_c:.4} >= err_u={err_u:.4}"
    );
}

// ============================================================
// Residual identity
// ============================================================

#[test]
fn test_residual_identity() {
    let d = 128;
    let pq = PolarQuant::new(d, 2, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let x = random_vector(d, &mut rng);

    let (result, residual) = pq.quantize_and_residual(&x, 1).unwrap();
    let x_hat = pq.dequantize_batch(&result).unwrap();

    for i in 0..d {
        assert!(
            (residual[i] - (x[i] - x_hat[i])).abs() < 1e-12,
            "residual mismatch at {i}"
        );
    }
}
