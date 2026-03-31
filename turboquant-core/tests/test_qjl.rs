//! Tests for QJL (Quantized Johnson-Lindenstrauss).
//! Mirrors turboquant_plus/tests/test_qjl.py.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turboquant_core::polar_quant::l2_norm;
use turboquant_core::qjl::*;

fn random_vector(d: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..d).map(|_| StandardNormal.sample(rng)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================
// Norm preservation
// ============================================================

#[test]
fn test_dequantized_has_correct_scale() {
    // Dequantized vectors should have roughly the same norm as originals.
    for d in [64, 128, 256] {
        let qjl = Qjl::new(d, 42).unwrap();
        let mut rng = StdRng::seed_from_u64(99);

        let mut norm_ratios = Vec::new();
        for _ in 0..200 {
            let x = random_vector(d, &mut rng);
            let (signs, norm) = qjl.quantize_single(&x).unwrap();
            let x_hat = qjl.dequantize_single(&signs, norm).unwrap();

            let norm_x = l2_norm(&x);
            if norm_x > 1e-10 {
                norm_ratios.push(l2_norm(&x_hat) / norm_x);
            }
        }

        let avg_ratio: f64 = norm_ratios.iter().sum::<f64>() / norm_ratios.len() as f64;
        assert!(
            avg_ratio > 0.5 && avg_ratio < 2.0,
            "d={d}: avg norm ratio {avg_ratio:.3} out of range"
        );
    }
}

// ============================================================
// Inner product unbiasedness (single-side)
// ============================================================

#[test]
fn test_inner_product_unbiased_single_side() {
    // E[<y, Q^-1(Q(x))>] = <y, x> when only x is quantized (Theorem 2).
    let d = 256;
    let qjl = Qjl::new(d, 42).unwrap();
    let mut rng = StdRng::seed_from_u64(77);

    let mut errors = Vec::new();
    for _ in 0..500 {
        let x = random_vector(d, &mut rng);
        let y = random_vector(d, &mut rng);

        let (signs_x, norm_x) = qjl.quantize_single(&x).unwrap();
        let x_hat = qjl.dequantize_single(&signs_x, norm_x).unwrap();

        let ip_original = dot(&x, &y);
        let ip_approx = dot(&x_hat, &y); // y NOT quantized
        errors.push(ip_approx - ip_original);
    }

    let mean_error: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
    let variance: f64 =
        errors.iter().map(|e| (e - mean_error).powi(2)).sum::<f64>() / errors.len() as f64;
    let std_error = variance.sqrt() / (errors.len() as f64).sqrt();

    assert!(
        mean_error.abs() < 3.0 * std_error + 0.1,
        "mean IP error {mean_error:.4} +/- {std_error:.4} — should be unbiased"
    );
}

// ============================================================
// Signs are binary
// ============================================================

#[test]
fn test_signs_are_binary() {
    let d = 128;
    let qjl = Qjl::new(d, 42).unwrap();
    let mut rng = StdRng::seed_from_u64(1);
    let x = random_vector(d, &mut rng);
    let (signs, _) = qjl.quantize_single(&x).unwrap();
    for &s in &signs {
        assert!(s == 1 || s == -1, "unexpected sign: {s}");
    }
}

// ============================================================
// Zero vector
// ============================================================

#[test]
fn test_zero_vector() {
    let d = 128;
    let qjl = Qjl::new(d, 42).unwrap();
    let x = vec![0.0; d];
    let (_, norm) = qjl.quantize_single(&x).unwrap();
    assert!(norm < 1e-15);
    let x_hat = qjl.dequantize_single(&[1i8; 128], norm).unwrap(); // signs don't matter at norm=0
    for v in &x_hat {
        assert!(v.abs() < 1e-15, "zero input → zero output");
    }
}

// ============================================================
// Batch matches single
// ============================================================

#[test]
fn test_batch_matches_single() {
    let d = 128;
    let qjl = Qjl::new(d, 42).unwrap();
    let mut rng = StdRng::seed_from_u64(7);

    let vectors: Vec<Vec<f64>> = (0..10).map(|_| random_vector(d, &mut rng)).collect();
    let flat: Vec<f64> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

    let batch = qjl.quantize_batch(&flat, 10).unwrap();

    for (i, vec) in vectors.iter().enumerate() {
        let (single_signs, single_norm) = qjl.quantize_single(vec).unwrap();
        assert_eq!(
            &batch.signs[i * d..(i + 1) * d],
            &single_signs[..],
            "signs mismatch at row {i}"
        );
        assert!(
            (batch.norms[i] - single_norm).abs() < 1e-12,
            "norm mismatch at row {i}"
        );
    }
}

// ============================================================
// Deterministic
// ============================================================

#[test]
fn test_deterministic() {
    let d = 128;
    let x: Vec<f64> = (0..d).map(|i| (i as f64 + 0.5) * 0.01).collect();

    let qjl1 = Qjl::new(d, 42).unwrap();
    let qjl2 = Qjl::new(d, 42).unwrap();

    let (signs1, norm1) = qjl1.quantize_single(&x).unwrap();
    let (signs2, norm2) = qjl2.quantize_single(&x).unwrap();
    assert_eq!(signs1, signs2);
    assert!((norm1 - norm2).abs() < 1e-15);
}
