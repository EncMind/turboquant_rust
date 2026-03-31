//! PolarQuant: Random rotation + optimal scalar quantization.
//!
//! Algorithm 1 from the TurboQuant paper (AISTATS 2026).
//!
//! After random rotation, coordinates follow a known Beta distribution (Gaussian in
//! high d), enabling optimal scalar quantization per coordinate independently.
//!
//! Important: codebook is calibrated for unit-norm vectors. For non-unit-norm inputs,
//! we extract norms, normalize, quantize, then rescale on dequantization.
//!
//! Mirrors `turboquant/polar_quant.py`.

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::mem::size_of;

use crate::codebook::{
    centroid_lookup, centroid_lookup_unchecked, nearest_centroid_indices, optimal_centroids,
};
use crate::error::{Result, TurboQuantError};
use crate::rotation::random_rotation_dense;
use crate::utils::{pack_indices, unpack_indices};

/// MSE-optimized vector quantizer via random rotation + scalar quantization.
///
/// Handles arbitrary-norm vectors by extracting norms before quantization
/// and rescaling after dequantization.
#[derive(Clone)]
pub struct PolarQuant {
    pub d: usize,
    pub bit_width: u32,
    pub n_centroids: usize,
    pub norm_correction: bool,
    pub rotation: DMatrix<f64>,
    pub centroids: Vec<f64>,
}

/// Result of PolarQuant quantization.
#[derive(Debug, Clone)]
pub struct PolarQuantResult {
    /// Centroid indices, shape `(batch, d)` flattened row-major.
    pub indices: Vec<u8>,
    /// L2 norms of the original vectors, one per vector.
    pub norms: Vec<f64>,
    /// Number of vectors in the batch.
    pub batch_size: usize,
}

/// Packed PolarQuant payload suitable for wire/storage format.
#[derive(Debug, Clone)]
pub struct PackedPolarQuantResult {
    /// Packed centroid indices (`bit_width` bits each).
    pub packed_indices: Vec<u8>,
    /// Original vector norms (stored as f32 for compactness).
    pub norms: Vec<f32>,
    pub batch_size: usize,
    pub d: usize,
    pub bit_width: u8,
}

impl PolarQuantResult {
    /// Pack raw centroid indices/norms into a compact payload.
    pub fn pack(&self, d: usize, bit_width: u8) -> Result<PackedPolarQuantResult> {
        if !(1..=8).contains(&bit_width) {
            return Err(TurboQuantError::InvalidBitWidthU8 {
                param: "bit_width",
                got: bit_width,
                min: 1,
                max: 8,
            });
        }
        let expected_indices = self.batch_size * d;
        if self.indices.len() != expected_indices {
            return Err(TurboQuantError::LengthMismatch {
                param: "indices",
                expected: expected_indices,
                got: self.indices.len(),
            });
        }
        if self.norms.len() != self.batch_size {
            return Err(TurboQuantError::LengthMismatch {
                param: "norms",
                expected: self.batch_size,
                got: self.norms.len(),
            });
        }

        Ok(PackedPolarQuantResult {
            packed_indices: pack_indices(&self.indices, bit_width)?,
            norms: self.norms.iter().map(|&n| n as f32).collect(),
            batch_size: self.batch_size,
            d,
            bit_width,
        })
    }
}

impl PackedPolarQuantResult {
    /// Unpack to raw per-coordinate indices and f64 norms.
    pub fn unpack(&self) -> Result<PolarQuantResult> {
        Ok(PolarQuantResult {
            indices: unpack_indices(&self.packed_indices, self.batch_size * self.d, self.bit_width)?,
            norms: self.norms.iter().map(|&n| n as f64).collect(),
            batch_size: self.batch_size,
        })
    }

    /// Unpack only centroid indices.
    pub fn unpack_indices(&self) -> Result<Vec<u8>> {
        unpack_indices(&self.packed_indices, self.batch_size * self.d, self.bit_width)
    }

    /// Payload size for packed wire format.
    pub fn wire_size_bytes(&self) -> usize {
        self.packed_indices.len() + self.norms.len() * size_of::<f32>()
    }

    /// Approximate in-memory size (payload + Vec metadata).
    pub fn in_memory_size_bytes(&self) -> usize {
        size_of::<Self>() + self.packed_indices.len() + self.norms.len() * size_of::<f32>()
    }
}

impl PolarQuant {
    /// Create a new PolarQuant quantizer.
    ///
    /// # Arguments
    /// * `d` — vector dimension.
    /// * `bit_width` — bits per coordinate for PolarQuant stage.
    /// * `seed` — random seed for rotation matrix and codebook.
    /// * `norm_correction` — whether to re-normalize reconstructed rotated vectors.
    pub fn new(d: usize, bit_width: u32, seed: u64, norm_correction: bool) -> Result<Self> {
        if d < 1 {
            return Err(TurboQuantError::InvalidDimension {
                param: "d",
                got: d,
                min: 1,
            });
        }
        if !(1..=8).contains(&bit_width) {
            return Err(TurboQuantError::InvalidBitWidth {
                param: "bit_width",
                got: bit_width,
                min: 1,
                max: 8,
            });
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let rotation = random_rotation_dense(d, &mut rng)?;
        let centroids = optimal_centroids(bit_width, d)?;
        let n_centroids = 1usize << bit_width;
        Ok(Self {
            d,
            bit_width,
            n_centroids,
            norm_correction,
            rotation,
            centroids,
        })
    }

    /// Quantize a single vector.
    ///
    /// Returns `(indices, norm)`.
    pub fn quantize_single(&self, x: &[f64]) -> Result<(Vec<u8>, f64)> {
        if x.len() != self.d {
            return Err(TurboQuantError::LengthMismatch {
                param: "x",
                expected: self.d,
                got: x.len(),
            });
        }
        let norm = l2_norm(x);
        let safe_norm = if norm > 0.0 { norm } else { 1.0 };

        // Normalize
        let x_normalized: Vec<f64> = x.iter().map(|&v| v / safe_norm).collect();

        // Rotate: y = Q @ x_normalized
        let y = crate::rotation::apply_dense_rotation(&self.rotation, &x_normalized)?;

        // Nearest centroid per coordinate
        let indices = nearest_centroid_indices(&y, &self.centroids)?;

        Ok((indices, norm))
    }

    /// Quantize a batch of vectors using batch BLAS (Level 3).
    ///
    /// `batch` is a flat slice of `batch_size * d` f64 values, row-major.
    pub fn quantize_batch(&self, batch: &[f64], batch_size: usize) -> Result<PolarQuantResult> {
        let expected_len = batch_size * self.d;
        if batch.len() != expected_len {
            return Err(TurboQuantError::LengthMismatch {
                param: "batch",
                expected: expected_len,
                got: batch.len(),
            });
        }

        if batch_size <= 1 {
            // Fall back to single-vector path for trivial batch
            if batch_size == 1 {
                let (indices, norm) = self.quantize_single(batch)?;
                return Ok(PolarQuantResult {
                    indices,
                    norms: vec![norm],
                    batch_size: 1,
                });
            }
            return Ok(PolarQuantResult {
                indices: vec![],
                norms: vec![],
                batch_size: 0,
            });
        }

        // 1) Compute norms and normalized vectors in one pass.
        let mut norms = vec![0.0; batch_size];
        let mut normalized = vec![0.0; batch_size * self.d];
        for (i, (src, dst)) in batch
            .chunks_exact(self.d)
            .zip(normalized.chunks_exact_mut(self.d))
            .enumerate()
        {
            let norm = l2_norm(src);
            norms[i] = norm;
            let inv = if norm > 0.0 { 1.0 / norm } else { 1.0 };
            for (out, &v) in dst.iter_mut().zip(src.iter()) {
                *out = v * inv;
            }
        }

        // 3. Batch rotation: Q @ X^T — single matrix-matrix multiply
        let rotated =
            crate::rotation::apply_dense_rotation_batch(&self.rotation, &normalized, self.d)?;

        // 4. Nearest centroid for all coordinates at once
        let all_indices = nearest_centroid_indices(&rotated, &self.centroids)?;

        Ok(PolarQuantResult {
            indices: all_indices,
            norms,
            batch_size,
        })
    }

    /// Quantize and return packed payload.
    pub fn quantize_batch_packed(
        &self,
        batch: &[f64],
        batch_size: usize,
    ) -> Result<PackedPolarQuantResult> {
        self.quantize_batch(batch, batch_size)
            ?.pack(self.d, self.bit_width as u8)
    }

    /// Dequantize a single vector from indices and norm.
    pub fn dequantize_single(&self, indices: &[u8], norm: f64) -> Result<Vec<f64>> {
        if indices.len() != self.d {
            return Err(TurboQuantError::LengthMismatch {
                param: "indices",
                expected: self.d,
                got: indices.len(),
            });
        }

        // Look up centroids
        let mut y_hat = centroid_lookup(indices, &self.centroids)?;

        // Norm correction: re-normalize y_hat to unit length in rotated space
        if self.norm_correction {
            let y_norm = l2_norm(&y_hat);
            if y_norm > 1e-10 {
                for v in y_hat.iter_mut() {
                    *v /= y_norm;
                }
            }
        }

        // Inverse rotation: x_hat_unit = Q^T @ y_hat
        let x_hat_unit = crate::rotation::apply_dense_rotation_transpose(&self.rotation, &y_hat)?;

        // Rescale by original norm
        Ok(x_hat_unit.iter().map(|&v| v * norm).collect())
    }

    /// Dequantize a batch using batch BLAS (Level 3).
    pub fn dequantize_batch(&self, result: &PolarQuantResult) -> Result<Vec<f64>> {
        if result.norms.len() != result.batch_size {
            return Err(TurboQuantError::LengthMismatch {
                param: "result.norms",
                expected: result.batch_size,
                got: result.norms.len(),
            });
        }
        let expected_indices = result.batch_size * self.d;
        if result.indices.len() != expected_indices {
            return Err(TurboQuantError::LengthMismatch {
                param: "result.indices",
                expected: expected_indices,
                got: result.indices.len(),
            });
        }
        self.dequantize_batch_impl(&result.indices, &result.norms, result.batch_size, false)
    }

    /// Dequantize from a packed payload.
    pub fn dequantize_batch_packed(&self, packed: &PackedPolarQuantResult) -> Result<Vec<f64>> {
        if packed.d != self.d {
            return Err(TurboQuantError::InvalidPackedMetadata {
                param: "d",
                expected: self.d,
                got: packed.d,
            });
        }
        if packed.bit_width as u32 != self.bit_width {
            return Err(TurboQuantError::InvalidPackedMetadata {
                param: "bit_width",
                expected: self.bit_width as usize,
                got: packed.bit_width as usize,
            });
        }
        self.dequantize_batch(&packed.unpack()?)
    }

    /// Quantize and return indices, norms, and residual error.
    ///
    /// Used by TurboQuant's second stage (QJL on residual).
    /// Returns `(indices, norms, residual)` where residual is flat `batch_size * d`.
    pub fn quantize_and_residual(
        &self,
        batch: &[f64],
        batch_size: usize,
    ) -> Result<(PolarQuantResult, Vec<f64>)> {
        let result = self.quantize_batch(batch, batch_size)?;
        // Internal fast path: indices come directly from this quantizer, so
        // bounds/sign validation work can be skipped safely.
        let reconstructed =
            self.dequantize_batch_impl(&result.indices, &result.norms, result.batch_size, true)?;

        // residual = x - x_hat
        let mut residual = vec![0.0; batch.len()];
        for ((out, &x), &xh) in residual.iter_mut().zip(batch.iter()).zip(reconstructed.iter()) {
            *out = x - xh;
        }

        Ok((result, residual))
    }

    fn dequantize_batch_impl(
        &self,
        indices: &[u8],
        norms: &[f64],
        batch_size: usize,
        trusted_indices: bool,
    ) -> Result<Vec<f64>> {
        if batch_size <= 1 {
            if batch_size == 1 {
                return self.dequantize_single(indices, norms[0]);
            }
            return Ok(vec![]);
        }

        // 1. Centroid lookup for all vectors at once.
        let mut y_hat: Vec<f64> = if trusted_indices {
            centroid_lookup_unchecked(indices, &self.centroids)
        } else {
            centroid_lookup(indices, &self.centroids)?
        };

        // 2. Norm correction per vector.
        if self.norm_correction {
            for row in y_hat.chunks_exact_mut(self.d) {
                let y_norm = l2_norm(row);
                if y_norm > 1e-10 {
                    let inv = 1.0 / y_norm;
                    for v in row {
                        *v *= inv;
                    }
                }
            }
        }

        // 3. Batch inverse rotation: Q^T @ Y_hat^T — single matrix-matrix multiply.
        let mut x_hat_unit =
            crate::rotation::apply_dense_rotation_transpose_batch(&self.rotation, &y_hat, self.d)?;

        // 4. Rescale each vector by its original norm.
        for (row, &norm) in x_hat_unit.chunks_exact_mut(self.d).zip(norms.iter()) {
            for v in row {
                *v *= norm;
            }
        }

        Ok(x_hat_unit)
    }
}

/// Compute L2 norm of a vector.
#[inline]
pub fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let pq = PolarQuant::new(16, 2, 42, true).unwrap();
        let x: Vec<f64> = (0..16).map(|i| (i as f64) * 0.1 + 0.1).collect();
        let (indices, norm) = pq.quantize_single(&x).unwrap();
        let x_hat = pq.dequantize_single(&indices, norm).unwrap();

        assert_eq!(x.len(), x_hat.len());
        // Check that reconstruction is in the right ballpark (not exact due to quantization)
        let error: f64 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let original_norm = l2_norm(&x);
        // Relative error should be reasonable for 2-bit
        assert!(
            error / original_norm < 0.5,
            "relative error too high: {:.4}",
            error / original_norm
        );
    }

    #[test]
    fn test_batch_matches_single() {
        let pq = PolarQuant::new(8, 2, 42, true).unwrap();
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v2: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        // Single
        let (idx1, n1) = pq.quantize_single(&v1).unwrap();
        let (idx2, n2) = pq.quantize_single(&v2).unwrap();

        // Batch
        let mut batch = v1.clone();
        batch.extend(&v2);
        let result = pq.quantize_batch(&batch, 2).unwrap();

        assert_eq!(result.indices[..8], idx1[..]);
        assert_eq!(result.indices[8..], idx2[..]);
        assert!((result.norms[0] - n1).abs() < 1e-12);
        assert!((result.norms[1] - n2).abs() < 1e-12);
    }

    #[test]
    fn test_zero_vector() {
        let pq = PolarQuant::new(8, 2, 42, true).unwrap();
        let x = vec![0.0; 8];
        let (indices, norm) = pq.quantize_single(&x).unwrap();
        assert!(norm.abs() < 1e-15);
        let x_hat = pq.dequantize_single(&indices, norm).unwrap();
        // Dequantized zero vector should be zero
        for v in &x_hat {
            assert!(v.abs() < 1e-15);
        }
    }

    #[test]
    fn test_quantize_and_residual() {
        let pq = PolarQuant::new(8, 2, 42, true).unwrap();
        let x: Vec<f64> = (0..8).map(|i| i as f64 + 1.0).collect();
        let (result, residual) = pq.quantize_and_residual(&x, 1).unwrap();

        // residual = x - dequant(quant(x))
        let x_hat = pq.dequantize_batch(&result).unwrap();
        for i in 0..8 {
            assert!((residual[i] - (x[i] - x_hat[i])).abs() < 1e-12);
        }
    }
}
