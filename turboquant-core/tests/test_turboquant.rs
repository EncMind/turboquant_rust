//! Tests for full TurboQuant (Algorithm 2 — PolarQuant + QJL).
//! Mirrors turboquant_plus/tests/test_turboquant.py and test_turbo4.py.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turboquant_core::polar_quant::l2_norm;
use turboquant_core::turboquant::*;

fn random_unit_vector(d: usize, rng: &mut StdRng) -> Vec<f64> {
    let x: Vec<f64> = (0..d).map(|_| StandardNormal.sample(rng)).collect();
    let norm = l2_norm(&x);
    x.iter().map(|v| v / norm).collect()
}

fn random_vector(d: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..d).map(|_| StandardNormal.sample(rng)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================
// MSE within paper bounds (Table 2)
// ============================================================

fn test_mse_within_paper_bounds(bit_width: u32, d: usize, paper_bound: f64) {
    let tq = TurboQuant::new(d, bit_width, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(99);

    let n_samples = 500;
    let mut mse_total = 0.0;
    for _ in 0..n_samples {
        let x = random_unit_vector(d, &mut rng);
        let compressed = tq.quantize(&x, 1).unwrap();
        let x_hat = tq.dequantize(&compressed).unwrap();
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
        avg_mse < paper_bound * 3.0,
        "MSE {avg_mse:.5} exceeds 3x paper bound {paper_bound} at d={d}, b={bit_width}"
    );
}

#[test]
fn test_mse_2bit_d64() {
    test_mse_within_paper_bounds(2, 64, 0.117);
}
#[test]
fn test_mse_2bit_d128() {
    test_mse_within_paper_bounds(2, 128, 0.117);
}
#[test]
fn test_mse_2bit_d256() {
    test_mse_within_paper_bounds(2, 256, 0.117);
}
#[test]
fn test_mse_3bit_d64() {
    test_mse_within_paper_bounds(3, 64, 0.03);
}
#[test]
fn test_mse_3bit_d128() {
    test_mse_within_paper_bounds(3, 128, 0.03);
}
#[test]
fn test_mse_3bit_d256() {
    test_mse_within_paper_bounds(3, 256, 0.03);
}
#[test]
fn test_mse_4bit_d64() {
    test_mse_within_paper_bounds(4, 64, 0.009);
}
#[test]
fn test_mse_4bit_d128() {
    test_mse_within_paper_bounds(4, 128, 0.009);
}
#[test]
fn test_mse_4bit_d256() {
    test_mse_within_paper_bounds(4, 256, 0.009);
}

// ============================================================
// Inner product preservation
// ============================================================

fn test_ip_preservation(bit_width: u32) {
    let d = 256;
    let tq = TurboQuant::new(d, bit_width, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(77);

    let mut ip_errors = Vec::new();
    for _ in 0..500 {
        let x = random_unit_vector(d, &mut rng);
        let y = random_unit_vector(d, &mut rng);

        let x_hat = tq.dequantize(&tq.quantize(&x, 1).unwrap()).unwrap();
        let y_hat = tq.dequantize(&tq.quantize(&y, 1).unwrap()).unwrap();

        let ip_original = dot(&x, &y);
        let ip_approx = dot(&x_hat, &y_hat);
        ip_errors.push((ip_approx - ip_original).abs());
    }

    let avg_ip_error: f64 = ip_errors.iter().sum::<f64>() / ip_errors.len() as f64;
    assert!(
        avg_ip_error < 0.5,
        "avg IP error {avg_ip_error:.6} unreasonably high at b={bit_width}"
    );
}

#[test]
fn test_ip_preservation_2bit() {
    test_ip_preservation(2);
}
#[test]
fn test_ip_preservation_3bit() {
    test_ip_preservation(3);
}
#[test]
fn test_ip_preservation_4bit() {
    test_ip_preservation(4);
}

// ============================================================
// bit_width=1 panics
// ============================================================

#[test]
fn test_bit_width_1_raises() {
    let err = TurboQuant::new(128, 1, 42, true).err().unwrap();
    assert!(
        matches!(
            err,
            turboquant_core::TurboQuantError::InvalidBitWidth {
                param: "bit_width",
                ..
            }
        ),
        "unexpected error: {err}"
    );
}

// ============================================================
// Zero vector
// ============================================================

#[test]
fn test_zero_vector() {
    let tq = TurboQuant::new(128, 3, 42, true).unwrap();
    let x = vec![0.0; 128];
    let compressed = tq.quantize(&x, 1).unwrap();
    let x_hat = tq.dequantize(&compressed).unwrap();
    assert!(l2_norm(&x_hat) < 1.0);
}

// ============================================================
// Deterministic
// ============================================================

#[test]
fn test_deterministic() {
    let d = 128;
    let x: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) * 0.01).collect();

    let tq1 = TurboQuant::new(d, 3, 42, true).unwrap();
    let tq2 = TurboQuant::new(d, 3, 42, true).unwrap();

    let c1 = tq1.quantize(&x, 1).unwrap();
    let c2 = tq2.quantize(&x, 1).unwrap();

    assert_eq!(
        c1.mse.unpack_indices().unwrap(),
        c2.mse.unpack_indices().unwrap()
    );
    assert_eq!(
        c1.qjl.unpack_signs().unwrap(),
        c2.qjl.unpack_signs().unwrap()
    );
}

// ============================================================
// Batch quantization
// ============================================================

#[test]
fn test_batch_matches_single() {
    let d = 128;
    let tq = TurboQuant::new(d, 3, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let n = 5;

    let vectors: Vec<Vec<f64>> = (0..n).map(|_| random_vector(d, &mut rng)).collect();
    let flat: Vec<f64> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

    let batch_compressed = tq.quantize(&flat, n).unwrap();
    let batch_recon = tq.dequantize(&batch_compressed).unwrap();

    for i in 0..n {
        let single_compressed = tq.quantize(&vectors[i], 1).unwrap();
        let single_recon = tq.dequantize(&single_compressed).unwrap();
        for j in 0..d {
            assert!(
                (batch_recon[i * d + j] - single_recon[j]).abs() < 1e-10,
                "mismatch at [{i},{j}]"
            );
        }
    }
}

// ============================================================
// turbo4-specific tests (from test_turbo4.py)
// ============================================================

#[test]
fn test_turbo4_non_128_head_dim_192() {
    // d=192 exercises non-standard head dimensions.
    let d = 192;
    let tq = TurboQuant::new(d, 4, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let x = random_vector(d, &mut rng);
    let compressed = tq.quantize(&x, 1).unwrap();
    let x_hat = tq.dequantize(&compressed).unwrap();

    assert_eq!(x_hat.len(), d);
    assert!(x_hat.iter().all(|v| v.is_finite()));

    let rel_mse = x
        .iter()
        .zip(x_hat.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / l2_norm(&x).powi(2);
    assert!(rel_mse < 0.7, "rel MSE {rel_mse:.4} too high at d=192");
}

#[test]
fn test_turbo4_non_128_aligned_dims() {
    for d in [96, 160, 192, 320] {
        let tq = TurboQuant::new(d, 4, 42, true).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let x = random_vector(d, &mut rng);
        let compressed = tq.quantize(&x, 1).unwrap();
        let x_hat = tq.dequantize(&compressed).unwrap();

        assert_eq!(x_hat.len(), d);
        assert!(x_hat.iter().all(|v| v.is_finite()));
        assert!(
            l2_norm(
                &x.iter()
                    .zip(x_hat.iter())
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>()
            ) < l2_norm(&x),
            "reconstruction worse than zero at d={d}"
        );
    }
}

#[test]
fn test_turbo4_various_norms() {
    let d = 128;
    let tq = TurboQuant::new(d, 4, 42, true).unwrap();
    for scale in [0.001, 1.0, 100.0, 10000.0] {
        let mut rng = StdRng::seed_from_u64(42);
        let x: Vec<f64> = random_vector(d, &mut rng)
            .iter()
            .map(|v| v * scale)
            .collect();
        let compressed = tq.quantize(&x, 1).unwrap();
        let x_hat = tq.dequantize(&compressed).unwrap();

        let norm_x = l2_norm(&x);
        if norm_x > 1e-10 {
            let err: Vec<f64> = x.iter().zip(x_hat.iter()).map(|(a, b)| a - b).collect();
            let rel_err = l2_norm(&err) / norm_x;
            assert!(
                rel_err < 0.6,
                "rel error {rel_err:.4} too high at scale={scale}"
            );
        }
    }
}

#[test]
fn test_turbo4_vs_turbo3_quality() {
    // turbo4 (4-bit) should have lower MSE than turbo3 (3-bit).
    let d = 128;
    let tq3 = TurboQuant::new(d, 3, 42, true).unwrap();
    let tq4 = TurboQuant::new(d, 4, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(99);

    let n = 200;
    let mut mse3 = 0.0;
    let mut mse4 = 0.0;
    for _ in 0..n {
        let x = random_unit_vector(d, &mut rng);
        let x3 = tq3.dequantize(&tq3.quantize(&x, 1).unwrap()).unwrap();
        let x4 = tq4.dequantize(&tq4.quantize(&x, 1).unwrap()).unwrap();
        mse3 += x
            .iter()
            .zip(x3.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        mse4 += x
            .iter()
            .zip(x4.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
    }
    let avg4 = mse4 / n as f64;
    let avg3 = mse3 / n as f64;
    assert!(
        avg4 < avg3,
        "turbo4 ({avg4:.5}) should be lower than turbo3 ({avg3:.5})"
    );
}

// ============================================================
// TurboQuantMSE
// ============================================================

#[test]
fn test_mse_only_round_trip() {
    let d = 128;
    let tqm = TurboQuantMse::new(d, 3, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(1);
    let x = random_unit_vector(d, &mut rng);
    let result = tqm.quantize(&x, 1).unwrap();
    let x_hat = tqm.dequantize(&result).unwrap();
    let mse: f64 = x
        .iter()
        .zip(x_hat.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / d as f64;
    assert!(mse < 0.1, "MSE-only 3-bit MSE {mse:.4} too high");
}

#[test]
fn test_norm_correction_can_be_disabled() {
    let d = 64;
    let x: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) * 0.1).collect();

    let tq_old = TurboQuantMse::new(d, 3, 42, false).unwrap();
    let tq_new = TurboQuantMse::new(d, 3, 42, true).unwrap();

    let res_old = tq_old.quantize(&x, 1).unwrap();
    let res_new = tq_new.quantize(&x, 1).unwrap();

    assert_eq!(res_old.indices, res_new.indices);

    let x_hat_old = tq_old.dequantize(&res_old).unwrap();
    let x_hat_new = tq_new.dequantize(&res_new).unwrap();

    // They should differ (norm correction changes the output)
    let diff: f64 = x_hat_old
        .iter()
        .zip(x_hat_new.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-6, "norm correction should change output");
}

// ============================================================
// Compressed size and ratio
// ============================================================

#[test]
fn test_size_calculation() {
    let tq = TurboQuant::new(128, 3, 42, true).unwrap();
    let bits = tq.compressed_size_bits(100);
    assert_eq!(bits, 100 * (128 * 3 + 32));
}

#[test]
fn test_size_scales_with_vectors() {
    let tq = TurboQuant::new(64, 4, 42, true).unwrap();
    let bits_10 = tq.compressed_size_bits(10);
    let bits_100 = tq.compressed_size_bits(100);
    assert_eq!(bits_100, bits_10 * 10);
}

#[test]
fn test_3bit_compression_ratio() {
    let tq = TurboQuant::new(128, 3, 42, true).unwrap();
    let ratio = tq.compression_ratio(16);
    assert!(
        ratio > 4.0 && ratio < 6.0,
        "3-bit ratio {ratio:.2} unexpected"
    );
}

#[test]
fn test_4bit_compression_ratio() {
    let tq = TurboQuant::new(128, 4, 42, true).unwrap();
    let ratio = tq.compression_ratio(16);
    assert!(
        ratio > 3.0 && ratio < 5.0,
        "4-bit ratio {ratio:.2} unexpected"
    );
}
