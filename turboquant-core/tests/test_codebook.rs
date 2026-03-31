//! Tests for codebook construction.
//! Mirrors turboquant_plus/tests/test_codebook.py.

use std::f64::consts::PI;
use turboquant_core::codebook::*;

// ============================================================
// Optimal Centroids
// ============================================================

#[test]
fn test_1bit_centroids_match_paper() {
    // 1-bit centroids should be ±sqrt(2/pi*d) per paper Theorem 3.1.
    for d in [64, 128, 256, 1024] {
        let c = optimal_centroids(1, d).unwrap();
        let expected = (2.0 / (PI * d as f64)).sqrt();
        assert_eq!(c.len(), 2);
        assert!(
            (c[0] + expected).abs() < 1e-10,
            "d={d}: c[0]={}, expected -{}",
            c[0],
            expected
        );
        assert!(
            (c[1] - expected).abs() < 1e-10,
            "d={d}: c[1]={}, expected {}",
            c[1],
            expected
        );
    }
}

#[test]
fn test_2bit_centroids_match_paper() {
    // 2-bit centroids: {±0.453/sqrt(d), ±1.51/sqrt(d)} per paper Table 1.
    for d in [64, 128, 256] {
        let c = optimal_centroids(2, d).unwrap();
        let scale = 1.0 / (d as f64).sqrt();
        let expected = [-1.51 * scale, -0.453 * scale, 0.453 * scale, 1.51 * scale];
        assert_eq!(c.len(), 4);
        for (a, b) in c.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "d={d}: {a} != {b}");
        }
    }
}

#[test]
fn test_centroids_sorted() {
    for b in 1..=4 {
        for d in [64, 128, 256] {
            let c = optimal_centroids(b, d).unwrap();
            for w in c.windows(2) {
                assert!(w[0] < w[1], "not sorted for b={b}, d={d}: {:?}", c);
            }
        }
    }
}

#[test]
fn test_correct_count() {
    for b in 1..=4 {
        let c = optimal_centroids(b, 128).unwrap();
        assert_eq!(c.len(), 1 << b, "b={b}: expected {} centroids", 1 << b);
    }
}

#[test]
fn test_centroids_symmetric() {
    // Centroids should be symmetric around 0 (Gaussian is symmetric).
    for b in 1..=4 {
        let c = optimal_centroids(b, 128).unwrap();
        let n = c.len();
        for i in 0..n {
            assert!(
                (c[i] + c[n - 1 - i]).abs() < 1e-10,
                "not symmetric for b={b}: c[{i}]={}, c[{}]={}",
                c[i],
                n - 1 - i,
                c[n - 1 - i]
            );
        }
    }
}

#[test]
fn test_lloyd_converges_3bit() {
    let d = 128;
    let c = optimal_centroids(3, d).unwrap();
    let sigma = 1.0 / (d as f64).sqrt();
    assert_eq!(c.len(), 8);
    for &v in &c {
        assert!(
            v.abs() < 4.0 * sigma,
            "centroid {v} outside 4sigma={:.4}",
            4.0 * sigma
        );
    }
}

#[test]
fn test_lloyd_converges_4bit() {
    let d = 256;
    let c = optimal_centroids(4, d).unwrap();
    let sigma = 1.0 / (d as f64).sqrt();
    assert_eq!(c.len(), 16);
    for &v in &c {
        assert!(v.abs() < 4.0 * sigma);
    }
}

#[test]
fn test_centroids_scale_with_dimension() {
    // Centroids should shrink as d increases (scale is 1/sqrt(d)).
    for b in 1..=3 {
        let c_small = optimal_centroids(b, 64).unwrap();
        let c_large = optimal_centroids(b, 256).unwrap();
        let max_small = c_small.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let max_large = c_large.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let ratio = max_small / max_large;
        assert!(
            ratio > 1.5 && ratio < 2.5,
            "b={b}: scale ratio {ratio:.2} unexpected"
        );
    }
}

// ============================================================
// Nearest Centroid Indices
// ============================================================

#[test]
fn test_exact_centroids_map_to_themselves() {
    let centroids = vec![-1.0, 0.0, 1.0];
    let values = vec![-1.0, 0.0, 1.0];
    let indices = nearest_centroid_indices(&values, &centroids).unwrap();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn test_values_outside_range_clamp() {
    let centroids = vec![-1.0, 0.0, 1.0];
    let values = vec![-100.0, 100.0];
    let indices = nearest_centroid_indices(&values, &centroids).unwrap();
    assert_eq!(indices[0], 0); // far left → first
    assert_eq!(indices[1], 2); // far right → last
}

#[test]
fn test_matches_brute_force() {
    // searchsorted result should match argmin brute force.
    let centroids = vec![-1.5, -0.5, 0.5, 1.5];
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = StdRng::seed_from_u64(42);
    let values: Vec<f64> = (0..500).map(|_| StandardNormal.sample(&mut rng)).collect();

    let fast_indices = nearest_centroid_indices(&values, &centroids).unwrap();

    // Brute force
    for (i, &v) in values.iter().enumerate() {
        let brute = centroids
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (v - *a).abs().partial_cmp(&(v - *b).abs()).unwrap())
            .unwrap()
            .0 as u8;
        assert_eq!(
            fast_indices[i], brute,
            "mismatch at i={i}, v={v}: fast={}, brute={}",
            fast_indices[i], brute
        );
    }
}

#[test]
fn test_all_indices_valid() {
    let centroids = vec![-1.5, -0.5, 0.5, 1.5];
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = StdRng::seed_from_u64(42);
    let values: Vec<f64> = (0..1000).map(|_| StandardNormal.sample(&mut rng)).collect();
    let indices = nearest_centroid_indices(&values, &centroids).unwrap();
    for &idx in &indices {
        assert!(idx < centroids.len() as u8);
    }
}
