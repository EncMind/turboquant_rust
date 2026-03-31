//! KV Cache integration layer for TurboQuant.
//!
//! Compresses transformer KV cache tensors using:
//! - TurboQuant (Algorithm 2) for K cache — inner product preservation
//! - TurboQuantMSE (Algorithm 1) for V cache — MSE preservation
//!
//! KV cache shape: (num_layers, num_heads, seq_len, head_dim)
//! Quantization is along head_dim — each (head_dim,) vector is quantized independently.
//!
//! Mirrors `turboquant/kv_cache.py`.

use rayon::prelude::*;
use std::mem::size_of;

use crate::error::{Result, TurboQuantError};
use crate::polar_quant::PackedPolarQuantResult;
use crate::turboquant::{CompressedVector, TurboQuant, TurboQuantMse};

/// Compressed KV cache container.
#[derive(Debug)]
pub struct CompressedKvCache {
    /// Per-layer, per-head compressed K vectors.
    pub k_compressed: Vec<Vec<CompressedVector>>,
    /// Per-layer, per-head compressed V (packed indices + norms).
    pub v_compressed: Vec<Vec<PackedPolarQuantResult>>,

    pub num_layers: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
    pub k_bit_width: u32,
    pub v_bit_width: u32,
}

/// KV cache memory statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub original_mb: f64,
    pub wire_compressed_mb: f64,
    pub in_memory_mb: f64,
    pub wire_compression_ratio: f64,
    pub in_memory_compression_ratio: f64,
    // Backward-compatible aliases for existing callers:
    pub compressed_mb: f64,
    pub compression_ratio: f64,
    pub k_bits_per_value: u32,
    pub v_bits_per_value: u32,
}

/// Compress and decompress transformer KV cache tensors.
///
/// Uses:
/// - TurboQuant (Algorithm 2) for K cache — inner product preservation matters
///   for attention score computation (Q @ K^T)
/// - TurboQuantMSE (Algorithm 1) for V cache — MSE preservation matters
///   for value reconstruction (attn_weights @ V)
pub struct KvCacheCompressor {
    pub head_dim: usize,
    pub k_bits: u32,
    pub v_bits: u32,
    k_quantizer: TurboQuant,
    v_quantizer: TurboQuantMse,
}

impl KvCacheCompressor {
    /// Create a new KV cache compressor.
    ///
    /// # Arguments
    /// * `head_dim` — dimension of each attention head vector.
    /// * `k_bits` — bit-width for K cache (TurboQuant, inner product).
    /// * `v_bits` — bit-width for V cache (PolarQuant MSE-only).
    /// * `seed` — random seed.
    /// * `norm_correction` — whether PolarQuant re-normalizes reconstructed vectors.
    pub fn new(
        head_dim: usize,
        k_bits: u32,
        v_bits: u32,
        seed: u64,
        norm_correction: bool,
    ) -> Result<Self> {
        if head_dim < 1 {
            return Err(TurboQuantError::InvalidDimension {
                param: "head_dim",
                got: head_dim,
                min: 1,
            });
        }
        if !(2..=9).contains(&k_bits) {
            return Err(TurboQuantError::InvalidBitWidth {
                param: "k_bits",
                got: k_bits,
                min: 2,
                max: 9,
            });
        }
        if !(1..=8).contains(&v_bits) {
            return Err(TurboQuantError::InvalidBitWidth {
                param: "v_bits",
                got: v_bits,
                min: 1,
                max: 8,
            });
        }
        let k_quantizer = TurboQuant::new(head_dim, k_bits, seed, norm_correction)?;
        let v_quantizer = TurboQuantMse::new(head_dim, v_bits, seed + 500, norm_correction)?;
        Ok(Self {
            head_dim,
            k_bits,
            v_bits,
            k_quantizer,
            v_quantizer,
        })
    }

    /// Compress full KV cache tensors.
    ///
    /// `k_cache` and `v_cache` are flat arrays of shape
    /// `(num_layers, num_heads, seq_len, head_dim)`, row-major.
    ///
    /// Uses rayon to parallelize across (layer, head) pairs.
    pub fn compress(
        &self,
        k_cache: &[f64],
        v_cache: &[f64],
        num_layers: usize,
        num_heads: usize,
        seq_len: usize,
    ) -> Result<CompressedKvCache> {
        let expected_len = num_layers * num_heads * seq_len * self.head_dim;
        if k_cache.len() != expected_len {
            return Err(TurboQuantError::LengthMismatch {
                param: "k_cache",
                expected: expected_len,
                got: k_cache.len(),
            });
        }
        if v_cache.len() != expected_len {
            return Err(TurboQuantError::LengthMismatch {
                param: "v_cache",
                expected: expected_len,
                got: v_cache.len(),
            });
        }

        let head_stride = seq_len * self.head_dim;
        let layer_stride = num_heads * head_stride;

        // Parallel over layers, each layer parallel over heads
        let results: Result<Vec<Vec<(CompressedVector, PackedPolarQuantResult)>>> = (0..num_layers)
            .into_par_iter()
            .map(|layer| -> Result<Vec<(CompressedVector, PackedPolarQuantResult)>> {
                (0..num_heads)
                    .into_par_iter()
                    .map(|head| -> Result<(CompressedVector, PackedPolarQuantResult)> {
                        let offset = layer * layer_stride + head * head_stride;
                        let k_vecs = &k_cache[offset..offset + head_stride];
                        let v_vecs = &v_cache[offset..offset + head_stride];
                        let k_comp = self.k_quantizer.quantize(k_vecs, seq_len)?;
                        let v_comp = self.v_quantizer.quantize_packed(v_vecs, seq_len)?;
                        Ok((k_comp, v_comp))
                    })
                    .collect()
            })
            .collect();
        let results = results?;

        // Unpack into separate k/v structures
        let mut k_compressed = Vec::with_capacity(num_layers);
        let mut v_compressed = Vec::with_capacity(num_layers);
        for layer_results in results {
            let mut k_layer = Vec::with_capacity(num_heads);
            let mut v_layer = Vec::with_capacity(num_heads);
            for (k, v) in layer_results {
                k_layer.push(k);
                v_layer.push(v);
            }
            k_compressed.push(k_layer);
            v_compressed.push(v_layer);
        }

        Ok(CompressedKvCache {
            k_compressed,
            v_compressed,
            num_layers,
            num_heads,
            seq_len,
            head_dim: self.head_dim,
            k_bit_width: self.k_bits,
            v_bit_width: self.v_bits,
        })
    }

    /// Decompress back to full KV cache tensors.
    ///
    /// Returns `(k_cache, v_cache)` each as flat `Vec<f64>`.
    /// Uses rayon to parallelize across (layer, head) pairs.
    pub fn decompress(&self, compressed: &CompressedKvCache) -> Result<(Vec<f64>, Vec<f64>)> {
        if compressed.head_dim != self.head_dim {
            return Err(TurboQuantError::InvalidPackedMetadata {
                param: "head_dim",
                expected: self.head_dim,
                got: compressed.head_dim,
            });
        }
        if compressed.k_bit_width != self.k_bits {
            return Err(TurboQuantError::InvalidPackedMetadata {
                param: "k_bit_width",
                expected: self.k_bits as usize,
                got: compressed.k_bit_width as usize,
            });
        }
        if compressed.v_bit_width != self.v_bits {
            return Err(TurboQuantError::InvalidPackedMetadata {
                param: "v_bit_width",
                expected: self.v_bits as usize,
                got: compressed.v_bit_width as usize,
            });
        }
        if compressed.k_compressed.len() != compressed.num_layers {
            return Err(TurboQuantError::LengthMismatch {
                param: "k_compressed layers",
                expected: compressed.num_layers,
                got: compressed.k_compressed.len(),
            });
        }
        if compressed.v_compressed.len() != compressed.num_layers {
            return Err(TurboQuantError::LengthMismatch {
                param: "v_compressed layers",
                expected: compressed.num_layers,
                got: compressed.v_compressed.len(),
            });
        }
        let total_len =
            compressed.num_layers * compressed.num_heads * compressed.seq_len * self.head_dim;
        let head_stride = compressed.seq_len * self.head_dim;
        let layer_stride = compressed.num_heads * head_stride;

        // Parallel decompression: each (layer, head) produces a (offset, k_data, v_data) tuple
        let chunks: Result<Vec<(usize, Vec<f64>, Vec<f64>)>> = (0..compressed.num_layers)
            .into_par_iter()
            .map(|layer| -> Result<Vec<(usize, Vec<f64>, Vec<f64>)>> {
                if compressed.k_compressed[layer].len() != compressed.num_heads {
                    return Err(TurboQuantError::LengthMismatch {
                        param: "k_compressed heads",
                        expected: compressed.num_heads,
                        got: compressed.k_compressed[layer].len(),
                    });
                }
                if compressed.v_compressed[layer].len() != compressed.num_heads {
                    return Err(TurboQuantError::LengthMismatch {
                        param: "v_compressed heads",
                        expected: compressed.num_heads,
                        got: compressed.v_compressed[layer].len(),
                    });
                }
                (0..compressed.num_heads)
                    .into_par_iter()
                    .map(move |head| -> Result<(usize, Vec<f64>, Vec<f64>)> {
                        let offset = layer * layer_stride + head * head_stride;
                        let k_recon = self
                            .k_quantizer
                            .dequantize(&compressed.k_compressed[layer][head])?;
                        let v_recon = self
                            .v_quantizer
                            .dequantize_packed(&compressed.v_compressed[layer][head])?;
                        Ok((offset, k_recon, v_recon))
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<(usize, Vec<f64>, Vec<f64>)>>>>()
            .map(|vv| vv.into_iter().flatten().collect());
        let chunks = chunks?;

        // Assemble into flat output
        let mut k_cache = vec![0.0; total_len];
        let mut v_cache = vec![0.0; total_len];
        for (offset, k_data, v_data) in chunks {
            k_cache[offset..offset + head_stride].copy_from_slice(&k_data);
            v_cache[offset..offset + head_stride].copy_from_slice(&v_data);
        }

        Ok((k_cache, v_cache))
    }

    /// Compute memory usage statistics.
    pub fn memory_stats(&self, seq_len: usize, num_layers: usize, num_heads: usize) -> MemoryStats {
        let n_vectors = num_layers * num_heads * seq_len;
        let n_heads_total = num_layers * num_heads;
        let original_bytes = n_vectors * self.head_dim * 2 * 2; // fp16 K + fp16 V

        let k_mse_bytes_per_head =
            (seq_len * self.head_dim * (self.k_bits as usize - 1)).div_ceil(8);
        let k_qjl_bytes_per_head = (seq_len * self.head_dim).div_ceil(8);
        let k_norm_bytes_per_head = seq_len * 2 * size_of::<f32>(); // mse norm + qjl norm
        let v_indices_bytes_per_head = (seq_len * self.head_dim * self.v_bits as usize).div_ceil(8);
        let v_norm_bytes_per_head = seq_len * size_of::<f32>(); // mse norm

        let wire_bytes = n_heads_total
            * (k_mse_bytes_per_head
                + k_qjl_bytes_per_head
                + k_norm_bytes_per_head
                + v_indices_bytes_per_head
                + v_norm_bytes_per_head);

        // Payload + container metadata assuming Vec capacity ~= length.
        let in_memory_bytes = wire_bytes
            + size_of::<CompressedKvCache>()
            + num_layers * size_of::<Vec<CompressedVector>>()
            + num_layers * size_of::<Vec<PackedPolarQuantResult>>()
            + n_heads_total * size_of::<CompressedVector>()
            + n_heads_total * size_of::<PackedPolarQuantResult>();

        MemoryStats {
            original_mb: original_bytes as f64 / 1024.0 / 1024.0,
            wire_compressed_mb: wire_bytes as f64 / 1024.0 / 1024.0,
            in_memory_mb: in_memory_bytes as f64 / 1024.0 / 1024.0,
            wire_compression_ratio: original_bytes as f64 / wire_bytes as f64,
            in_memory_compression_ratio: original_bytes as f64 / in_memory_bytes as f64,
            compressed_mb: wire_bytes as f64 / 1024.0 / 1024.0,
            compression_ratio: original_bytes as f64 / wire_bytes as f64,
            k_bits_per_value: self.k_bits,
            v_bits_per_value: self.v_bits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_roundtrip() {
        let head_dim = 8;
        let num_layers = 1;
        let num_heads = 2;
        let seq_len = 4;

        let compressor = KvCacheCompressor::new(head_dim, 3, 3, 42, true).unwrap();

        // Random-ish data
        let total = num_layers * num_heads * seq_len * head_dim;
        let k_cache: Vec<f64> = (0..total).map(|i| (i as f64) * 0.01).collect();
        let v_cache: Vec<f64> = (0..total).map(|i| (i as f64) * 0.02 - 0.5).collect();

        let compressed = compressor
            .compress(&k_cache, &v_cache, num_layers, num_heads, seq_len)
            .unwrap();
        assert_eq!(compressed.num_layers, num_layers);
        assert_eq!(compressed.num_heads, num_heads);
        assert_eq!(compressed.seq_len, seq_len);

        let (k_hat, v_hat) = compressor.decompress(&compressed).unwrap();
        assert_eq!(k_hat.len(), total);
        assert_eq!(v_hat.len(), total);
    }

    #[test]
    fn test_memory_stats() {
        let compressor = KvCacheCompressor::new(128, 3, 3, 42, true).unwrap();
        let stats = compressor.memory_stats(1024, 32, 32);
        assert!(stats.wire_compression_ratio > 1.0);
        assert!(stats.original_mb > stats.wire_compressed_mb);
    }
}
