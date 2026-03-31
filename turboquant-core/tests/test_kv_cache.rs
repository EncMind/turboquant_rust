//! Tests for KV cache integration layer.
//! Mirrors turboquant_plus/tests/test_kv_cache.py.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turboquant_core::kv_cache::*;
use turboquant_core::polar_quant::l2_norm;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

// ============================================================
// Round-trip shape
// ============================================================

#[test]
fn test_round_trip_shape() {
    let head_dim = 64;
    let num_layers = 2;
    let num_heads = 4;
    let seq_len = 16;

    let compressor = KvCacheCompressor::new(head_dim, 3, 3, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let total = num_layers * num_heads * seq_len * head_dim;
    let k: Vec<f64> = (0..total)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let v: Vec<f64> = (0..total)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();

    let compressed = compressor.compress(&k, &v, num_layers, num_heads, seq_len).unwrap();
    let (k_hat, v_hat) = compressor.decompress(&compressed).unwrap();

    assert_eq!(k_hat.len(), total);
    assert_eq!(v_hat.len(), total);
}

// ============================================================
// Round-trip quality
// ============================================================

#[test]
fn test_round_trip_quality() {
    let head_dim = 128;
    let num_layers = 2;
    let num_heads = 4;
    let seq_len = 32;
    let total = num_layers * num_heads * seq_len * head_dim;

    let compressor = KvCacheCompressor::new(head_dim, 3, 3, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    // Generate and normalize to unit vectors per head_dim chunk
    let mut k: Vec<f64> = (0..total)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let mut v: Vec<f64> = (0..total)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    for i in 0..(total / head_dim) {
        let start = i * head_dim;
        let end = start + head_dim;
        let norm_k = l2_norm(&k[start..end]);
        let norm_v = l2_norm(&v[start..end]);
        for j in start..end {
            k[j] /= norm_k;
            v[j] /= norm_v;
        }
    }

    let compressed = compressor.compress(&k, &v, num_layers, num_heads, seq_len).unwrap();
    let (k_hat, v_hat) = compressor.decompress(&compressed).unwrap();

    let k_mse: f64 = k
        .iter()
        .zip(k_hat.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / total as f64;
    let v_mse: f64 = v
        .iter()
        .zip(v_hat.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / total as f64;

    assert!(k_mse < 0.09, "K cache MSE {k_mse:.4} too high");
    assert!(v_mse < 0.09, "V cache MSE {v_mse:.4} too high");
}

// ============================================================
// Attention score preservation
// ============================================================

#[test]
fn test_attention_score_preservation() {
    let head_dim = 64;
    let seq_len = 16;

    let compressor = KvCacheCompressor::new(head_dim, 3, 3, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    // Query, Keys, Values
    let q: Vec<f64> = (0..head_dim)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let k: Vec<f64> = (0..seq_len * head_dim)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let v: Vec<f64> = (0..seq_len * head_dim)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();

    // Original attention
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut scores_orig = vec![0.0; seq_len];
    for s in 0..seq_len {
        scores_orig[s] = dot(&q, &k[s * head_dim..(s + 1) * head_dim]) * scale;
    }
    let attn_orig = softmax(&scores_orig);
    let mut out_orig = vec![0.0; head_dim];
    for s in 0..seq_len {
        for d in 0..head_dim {
            out_orig[d] += attn_orig[s] * v[s * head_dim + d];
        }
    }

    // Compressed: wrap as 1 layer, 1 head
    let compressed = compressor.compress(&k, &v, 1, 1, seq_len).unwrap();
    let (k_hat, v_hat) = compressor.decompress(&compressed).unwrap();

    let mut scores_comp = vec![0.0; seq_len];
    for s in 0..seq_len {
        scores_comp[s] = dot(&q, &k_hat[s * head_dim..(s + 1) * head_dim]) * scale;
    }
    let attn_comp = softmax(&scores_comp);
    let mut out_comp = vec![0.0; head_dim];
    for s in 0..seq_len {
        for d in 0..head_dim {
            out_comp[d] += attn_comp[s] * v_hat[s * head_dim + d];
        }
    }

    // Cosine similarity
    let cosine_sim = dot(&out_orig, &out_comp) / (l2_norm(&out_orig) * l2_norm(&out_comp));
    assert!(
        cosine_sim > 0.5,
        "attention output cosine similarity {cosine_sim:.3} too low"
    );
}

// ============================================================
// Memory stats
// ============================================================

#[test]
fn test_memory_stats() {
    let compressor = KvCacheCompressor::new(128, 3, 3, 42, true).unwrap();
    let stats = compressor.memory_stats(1024, 32, 32);
    assert!(stats.wire_compression_ratio > 2.0);
    assert!(stats.in_memory_compression_ratio > 1.0);
    assert!(stats.wire_compressed_mb < stats.original_mb);
}

// ============================================================
// Metadata stored
// ============================================================

#[test]
fn test_metadata_stored() {
    let compressor = KvCacheCompressor::new(64, 3, 3, 42, true).unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let total = 2 * 4 * 8 * 64;
    let k: Vec<f64> = (0..total)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let v: Vec<f64> = (0..total)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();

    let compressed = compressor.compress(&k, &v, 2, 4, 8).unwrap();
    assert_eq!(compressed.num_layers, 2);
    assert_eq!(compressed.num_heads, 4);
    assert_eq!(compressed.seq_len, 8);
    assert_eq!(compressed.head_dim, 64);
    assert_eq!(compressed.k_bit_width, 3);
    assert_eq!(compressed.v_bit_width, 3);
}
