//! Tests for random rotation matrix generation.
//! Mirrors turboquant_plus/tests/test_rotation.py.

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use turboquant_core::rotation::*;

fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================
// Dense Rotation
// ============================================================

#[test]
fn test_orthogonality() {
    // Q @ Q^T should equal identity.
    let mut rng = StdRng::seed_from_u64(42);
    let d = 128;
    let q = random_rotation_dense(d, &mut rng).unwrap();
    let qtq = q.transpose() * &q;
    let eye = DMatrix::<f64>::identity(d, d);
    assert!((qtq - &eye).norm() < 1e-10);
}

#[test]
fn test_transpose_is_inverse() {
    let mut rng = StdRng::seed_from_u64(42);
    let d = 64;
    let q = random_rotation_dense(d, &mut rng).unwrap();
    let qqt = &q * q.transpose();
    let eye = DMatrix::<f64>::identity(d, d);
    assert!((qqt - eye).norm() < 1e-10);
}

#[test]
fn test_determinant_positive_one() {
    // det(Q) should be +1 (proper rotation, not reflection).
    for seed in [42u64, 99, 7, 123, 0] {
        let mut rng = StdRng::seed_from_u64(seed);
        let q = random_rotation_dense(64, &mut rng).unwrap();
        let det = q.determinant();
        assert!(det > 0.0, "det = {det} for seed={seed}, expected positive");
        assert!(
            (det - 1.0).abs() < 1e-6,
            "det = {det} for seed={seed}, expected ~1.0"
        );
    }
}

#[test]
fn test_deterministic_same_seed() {
    let d = 128;
    let q1 = random_rotation_dense(d, &mut StdRng::seed_from_u64(42)).unwrap();
    let q2 = random_rotation_dense(d, &mut StdRng::seed_from_u64(42)).unwrap();
    assert!((q1 - q2).norm() < 1e-15);
}

#[test]
fn test_different_seeds_differ() {
    let d = 64;
    let q1 = random_rotation_dense(d, &mut StdRng::seed_from_u64(42)).unwrap();
    let q2 = random_rotation_dense(d, &mut StdRng::seed_from_u64(99)).unwrap();
    assert!((q1 - q2).norm() > 0.1);
}

#[test]
fn test_preserves_norms() {
    // ||Q @ x|| should equal ||x|| for any x.
    let mut rng_rot = StdRng::seed_from_u64(42);
    let d = 128;
    let q = random_rotation_dense(d, &mut rng_rot).unwrap();

    let mut rng_vec = StdRng::seed_from_u64(99);
    for _ in 0..100 {
        let x: Vec<f64> = (0..d)
            .map(|_| StandardNormal.sample(&mut rng_vec))
            .collect();
        let y = apply_dense_rotation(&q, &x).unwrap();
        let norm_x = l2_norm(&x);
        let norm_y = l2_norm(&y);
        assert!(
            (norm_x - norm_y).abs() / norm_x < 1e-10,
            "norms differ: {norm_x} vs {norm_y}"
        );
    }
}

#[test]
fn test_preserves_inner_products() {
    // <Q@x, Q@y> should equal <x, y>.
    let mut rng_rot = StdRng::seed_from_u64(42);
    let d = 128;
    let q = random_rotation_dense(d, &mut rng_rot).unwrap();

    let mut rng_vec = StdRng::seed_from_u64(77);
    for _ in 0..100 {
        let x: Vec<f64> = (0..d)
            .map(|_| StandardNormal.sample(&mut rng_vec))
            .collect();
        let y: Vec<f64> = (0..d)
            .map(|_| StandardNormal.sample(&mut rng_vec))
            .collect();
        let ip_original = dot(&x, &y);
        let qx = apply_dense_rotation(&q, &x).unwrap();
        let qy = apply_dense_rotation(&q, &y).unwrap();
        let ip_rotated = dot(&qx, &qy);
        assert!(
            (ip_original - ip_rotated).abs() < 1e-8,
            "IP mismatch: {ip_original} vs {ip_rotated}"
        );
    }
}

#[test]
fn test_post_rotation_distribution() {
    // After rotating a FIXED unit vector by many random Q,
    // each coordinate should have mean ~0 and variance ~1/d.
    let d = 64;
    let mut e1 = vec![0.0; d];
    e1[0] = 1.0;

    let n_samples = 400;
    let mut rotated = vec![vec![0.0; d]; n_samples];
    for (i, row) in rotated.iter_mut().enumerate().take(n_samples) {
        let mut rng = StdRng::seed_from_u64(i as u64);
        let q = random_rotation_dense(d, &mut rng).unwrap();
        *row = apply_dense_rotation(&q, &e1).unwrap();
    }

    // Coordinate means should be ~0
    for j in 0..d {
        let mean: f64 = rotated.iter().map(|r| r[j]).sum::<f64>() / n_samples as f64;
        let bound = 4.0 * (1.0 / d as f64 / n_samples as f64).sqrt();
        let bound = bound.max(0.05);
        assert!(
            mean.abs() < bound,
            "coord {j} mean {mean:.4} exceeds bound {bound:.4}"
        );
    }

    // Coordinate variances should be ~1/d
    let expected_var = 1.0 / d as f64;
    for j in 0..d {
        let mean: f64 = rotated.iter().map(|r| r[j]).sum::<f64>() / n_samples as f64;
        let var: f64 =
            rotated.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n_samples as f64;
        assert!(
            var < expected_var * 1.8,
            "coord {j} var {var:.6} exceeds {:.6}",
            expected_var * 1.8
        );
        assert!(
            var > expected_var * 0.5,
            "coord {j} var {var:.6} below {:.6}",
            expected_var * 0.5
        );
    }
}

#[test]
fn test_small_dimensions() {
    for d in [1, 2, 4, 8] {
        let mut rng = StdRng::seed_from_u64(42);
        let q = random_rotation_dense(d, &mut rng).unwrap();
        assert_eq!(q.nrows(), d);
        assert_eq!(q.ncols(), d);
        let eye = DMatrix::<f64>::identity(d, d);
        assert!((q.transpose() * &q - eye).norm() < 1e-10);
    }
}

#[test]
fn test_invalid_dimension_raises() {
    let err = random_rotation_dense(0, &mut StdRng::seed_from_u64(42)).unwrap_err();
    assert!(format!("{err}").contains("d must be"));
}

// ============================================================
// Fast Walsh-Hadamard Transform
// ============================================================

#[test]
fn test_fwht_involutory() {
    // Applying FWHT twice should return the original.
    let mut rng = StdRng::seed_from_u64(42);
    let n = 32;
    let original: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();

    let mut x = original.clone();
    fast_walsh_hadamard_transform(&mut x).unwrap();
    fast_walsh_hadamard_transform(&mut x).unwrap();

    for (a, b) in x.iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
}

#[test]
fn test_fwht_preserves_norm() {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 64;
    let original: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    let norm_orig = l2_norm(&original);

    let mut x = original;
    fast_walsh_hadamard_transform(&mut x).unwrap();
    let norm_after = l2_norm(&x);

    assert!(
        (norm_orig - norm_after).abs() / norm_orig < 1e-10,
        "norms differ: {norm_orig} vs {norm_after}"
    );
}

#[test]
fn test_fwht_various_sizes() {
    for n in [2, 4, 8, 16, 32, 64, 128] {
        let mut rng = StdRng::seed_from_u64(42);
        let original: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let norm_orig = l2_norm(&original);
        let mut x = original;
        fast_walsh_hadamard_transform(&mut x).unwrap();
        assert_eq!(x.len(), n);
        let norm_after = l2_norm(&x);
        assert!(
            (norm_orig - norm_after).abs() / norm_orig < 1e-10,
            "norm not preserved at n={n}"
        );
    }
}

#[test]
fn test_fwht_non_pow2_panics() {
    let mut x = vec![1.0, 2.0, 3.0];
    let err = fast_walsh_hadamard_transform(&mut x).unwrap_err();
    assert!(format!("{err}").contains("power of 2"));
}

// ============================================================
// Fast Rotation (Hadamard + random signs)
// ============================================================

#[test]
fn test_fast_rotation_preserves_norms_pow2() {
    let mut rng = StdRng::seed_from_u64(42);
    let d = 64;
    let rot = FastRotation::new(d, &mut rng).unwrap();

    let mut rng_vec = StdRng::seed_from_u64(99);
    for _ in 0..50 {
        let x: Vec<f64> = (0..d)
            .map(|_| StandardNormal.sample(&mut rng_vec))
            .collect();
        let y = rot.apply(&x).unwrap();
        let norm_x = l2_norm(&x);
        let norm_y = l2_norm(&y);
        assert!(
            (norm_x - norm_y).abs() / norm_x < 1e-8,
            "norms differ: {norm_x} vs {norm_y}"
        );
    }
}

#[test]
fn test_fast_rotation_non_pow2_length() {
    // For non-power-of-2, output should still be d elements.
    let d = 100;
    let mut rng = StdRng::seed_from_u64(42);
    let rot = FastRotation::new(d, &mut rng).unwrap();
    assert_eq!(rot.padded_d, 128);

    let x: Vec<f64> = (0..d).map(|i| i as f64).collect();
    let y = rot.apply(&x).unwrap();
    assert_eq!(y.len(), d);
}

#[test]
fn test_fast_rotation_transpose_inverts() {
    let mut rng = StdRng::seed_from_u64(42);
    let d = 64;
    let rot = FastRotation::new(d, &mut rng).unwrap();

    let mut rng_vec = StdRng::seed_from_u64(99);
    let x: Vec<f64> = (0..d)
        .map(|_| StandardNormal.sample(&mut rng_vec))
        .collect();
    let y = rot.apply(&x).unwrap();
    let x_back = rot.apply_transpose(&y).unwrap();

    for (a, b) in x.iter().zip(x_back.iter()) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
}

#[test]
fn test_fast_rotation_deterministic() {
    let d = 64;
    let x: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();

    let rot1 = FastRotation::new(d, &mut StdRng::seed_from_u64(42)).unwrap();
    let rot2 = FastRotation::new(d, &mut StdRng::seed_from_u64(42)).unwrap();

    let y1 = rot1.apply(&x).unwrap();
    let y2 = rot2.apply(&x).unwrap();
    assert_eq!(y1, y2);
}

#[test]
fn test_fast_rotation_batch_matches_single() {
    let mut rng = StdRng::seed_from_u64(42);
    let d = 64;
    let rot = FastRotation::new(d, &mut rng).unwrap();

    let mut rng_vec = StdRng::seed_from_u64(99);
    let batch: Vec<Vec<f64>> = (0..10)
        .map(|_| {
            (0..d)
                .map(|_| StandardNormal.sample(&mut rng_vec))
                .collect()
        })
        .collect();

    let batch_result = rot.apply_batch(&batch).unwrap();

    for (i, row) in batch.iter().enumerate() {
        let single_result = rot.apply(row).unwrap();
        for (a, b) in batch_result[i].iter().zip(single_result.iter()) {
            assert!((a - b).abs() < 1e-10, "batch[{i}] mismatch");
        }
    }
}
