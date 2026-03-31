//! Tests for outlier channel strategy (non-integer bit precision).
//! Mirrors turboquant_plus/tests/test_outlier.py.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turboquant_core::outlier::*;
use turboquant_core::polar_quant::l2_norm;

fn random_unit_vector(d: usize, rng: &mut StdRng) -> Vec<f64> {
    let x: Vec<f64> = (0..d).map(|_| StandardNormal.sample(rng)).collect();
    let norm = l2_norm(&x);
    x.iter().map(|v| v / norm).collect()
}

fn random_vector(d: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..d).map(|_| StandardNormal.sample(rng)).collect()
}

#[test]
fn test_2_5_bit_effective_rate() {
    let oq = OutlierTurboQuant::new(128, 2.5, 42).unwrap();
    assert!(
        (oq.effective_bits - 2.5).abs() < 0.01,
        "effective_bits={}, expected 2.5",
        oq.effective_bits
    );
}

#[test]
fn test_3_5_bit_effective_rate() {
    let oq = OutlierTurboQuant::new(128, 3.5, 42).unwrap();
    assert!(
        (oq.effective_bits - 3.5).abs() < 0.01,
        "effective_bits={}, expected 3.5",
        oq.effective_bits
    );
}

#[test]
fn test_round_trip_quality() {
    let d = 128;
    let oq = OutlierTurboQuant::new(d, 3.5, 42).unwrap();
    let mut rng = StdRng::seed_from_u64(99);

    let mut mses = Vec::new();
    for _ in 0..200 {
        let x = random_unit_vector(d, &mut rng);
        let compressed = oq.quantize(&x, 1).unwrap();
        let x_hat = oq.dequantize(&compressed).unwrap();
        let mse: f64 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        mses.push(mse);
    }

    let avg_mse: f64 = mses.iter().sum::<f64>() / mses.len() as f64;
    assert!(avg_mse < 0.03 * 2.0, "3.5-bit MSE {avg_mse:.5} too high");
}

#[test]
fn test_compression_ratio_2_5bit() {
    let oq = OutlierTurboQuant::new(128, 2.5, 42).unwrap();
    let ratio = oq.compression_ratio(16);
    assert!(ratio > 4.5, "2.5-bit ratio {ratio:.1} too low");
}

#[test]
fn test_compression_ratio_3_5bit() {
    let oq = OutlierTurboQuant::new(128, 3.5, 42).unwrap();
    let ratio = oq.compression_ratio(16);
    assert!(ratio > 3.5, "3.5-bit ratio {ratio:.1} too low");
}

#[test]
fn test_outlier_channels_identified() {
    let d = 128;
    let oq = OutlierTurboQuant::new(d, 2.5, 42).unwrap();
    assert!(oq.n_outlier > 0);
    assert!(oq.n_outlier < d);
    assert_eq!(oq.n_outlier + oq.n_normal, d);
}

#[test]
fn test_batch_matches_single() {
    let d = 128;
    let oq = OutlierTurboQuant::new(d, 3.5, 42).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let n = 5;

    let vectors: Vec<Vec<f64>> = (0..n).map(|_| random_vector(d, &mut rng)).collect();

    // Batch
    let flat: Vec<f64> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    let batch_comp = oq.quantize(&flat, n).unwrap();
    let batch_recon = oq.dequantize(&batch_comp).unwrap();

    // Single path can choose different outlier channels than the batch path.
    // Compare reconstruction quality rather than elementwise identity.
    let mut mse_batch_total = 0.0;
    let mut mse_single_total = 0.0;
    for i in 0..n {
        let x = &vectors[i];
        let batch_vec = &batch_recon[i * d..(i + 1) * d];
        let single_comp = oq.quantize(&vectors[i], 1).unwrap();
        let single_recon = oq.dequantize(&single_comp).unwrap();
        let mse_batch: f64 = x
            .iter()
            .zip(batch_vec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        let mse_single: f64 = x
            .iter()
            .zip(single_recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        mse_batch_total += mse_batch;
        mse_single_total += mse_single;
    }

    let avg_batch = mse_batch_total / n as f64;
    let avg_single = mse_single_total / n as f64;
    assert!(avg_batch.is_finite() && avg_single.is_finite());
    assert!(
        avg_batch < avg_single * 2.0 && avg_single < avg_batch * 2.0,
        "batch/single quality diverged too much: avg_batch={avg_batch:.5}, avg_single={avg_single:.5}"
    );
}

#[test]
fn test_deterministic() {
    let d = 128;
    let x: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) * 0.01).collect();

    let oq1 = OutlierTurboQuant::new(d, 3.5, 42).unwrap();
    let oq2 = OutlierTurboQuant::new(d, 3.5, 42).unwrap();

    let r1 = oq1.dequantize(&oq1.quantize(&x, 1).unwrap()).unwrap();
    let r2 = oq2.dequantize(&oq2.quantize(&x, 1).unwrap()).unwrap();

    for (a, b) in r1.iter().zip(r2.iter()) {
        assert!((a - b).abs() < 1e-15);
    }
}
