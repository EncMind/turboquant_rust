//! TurboQuant: Full algorithm combining PolarQuant + QJL.
//!
//! Algorithm 2 from the paper — Inner Product TurboQuant.
//!
//! Two-stage process:
//! 1. PolarQuant at (b-1) bits for MSE-optimal compression
//! 2. QJL at 1 bit on the residual for bias elimination
//!
//! Total: b bits per coordinate with near-optimal inner product distortion.
//!
//! Mirrors `turboquant/turboquant.py`.

use crate::error::{Result, TurboQuantError};
use crate::polar_quant::{PackedPolarQuantResult, PolarQuant, PolarQuantResult};
use crate::qjl::{PackedQjlResult, Qjl, QjlResult};
use std::mem::size_of;

/// Container for a TurboQuant-compressed vector (or batch).
#[derive(Debug, Clone)]
pub struct CompressedVector {
    /// PolarQuant packed result: bit-packed indices + vector norms.
    pub mse: PackedPolarQuantResult,
    /// QJL packed result: bit-packed sign bits + residual norms.
    pub qjl: PackedQjlResult,
    /// Total bits per coordinate.
    pub bit_width: u32,
}

/// In-memory compressed representation optimized for immediate dequantization.
///
/// This skips bit packing/unpacking overhead and is useful for latency-sensitive
/// roundtrip workloads where a wire format is not required.
#[derive(Debug, Clone)]
pub struct CompressedVectorUnpacked {
    /// PolarQuant raw result: per-coordinate centroid indices + vector norms.
    pub mse: PolarQuantResult,
    /// QJL raw result: per-coordinate signs + residual norms.
    pub qjl: QjlResult,
    /// Total bits per coordinate.
    pub bit_width: u32,
}

impl CompressedVector {
    /// Packed payload bytes for wire/storage format.
    pub fn wire_size_bytes(&self) -> usize {
        self.mse.wire_size_bytes() + self.qjl.wire_size_bytes()
    }

    /// Approximate in-memory size (payload + Vec metadata).
    pub fn in_memory_size_bytes(&self) -> usize {
        size_of::<Self>() + self.wire_size_bytes()
    }
}

/// Full TurboQuant quantizer: PolarQuant (b-1 bits) + QJL (1 bit).
///
/// # Example
/// ```
/// use turboquant_core::turboquant::TurboQuant;
///
/// let tq = TurboQuant::new(16, 3, 42, true).unwrap();
/// let x: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
/// let compressed = tq.quantize(&x, 1).unwrap();
/// let x_hat = tq.dequantize(&compressed).unwrap();
/// assert_eq!(x_hat.len(), 16);
/// ```
#[derive(Clone)]
pub struct TurboQuant {
    pub d: usize,
    pub bit_width: u32,
    pub polar_quant: PolarQuant,
    pub qjl: Qjl,
}

impl TurboQuant {
    /// Create a new TurboQuant quantizer.
    ///
    /// # Arguments
    /// * `d` — vector dimension.
    /// * `bit_width` — total bits per coordinate. PolarQuant uses `bit_width - 1`, QJL uses 1.
    /// * `seed` — random seed.
    /// * `norm_correction` — whether PolarQuant re-normalizes reconstructed vectors.
    ///
    pub fn new(d: usize, bit_width: u32, seed: u64, norm_correction: bool) -> Result<Self> {
        if !(2..=9).contains(&bit_width) {
            return Err(TurboQuantError::InvalidBitWidth {
                param: "bit_width",
                got: bit_width,
                min: 2,
                max: 9,
            });
        }

        let polar_quant = PolarQuant::new(d, bit_width - 1, seed, norm_correction)?;
        let qjl = Qjl::new(d, seed + 1000)?;

        Ok(Self {
            d,
            bit_width,
            polar_quant,
            qjl,
        })
    }

    /// Quantize a flat batch of vectors.
    ///
    /// `batch` is `batch_size * d` f64 values, row-major.
    pub fn quantize(&self, batch: &[f64], batch_size: usize) -> Result<CompressedVector> {
        let unpacked = self.quantize_unpacked(batch, batch_size)?;
        Ok(CompressedVector {
            mse: unpacked.mse.pack(self.d, (self.bit_width - 1) as u8)?,
            qjl: unpacked.qjl.pack(self.d)?,
            bit_width: self.bit_width,
        })
    }

    /// Quantize to in-memory raw representation (no bit packing).
    pub fn quantize_unpacked(
        &self,
        batch: &[f64],
        batch_size: usize,
    ) -> Result<CompressedVectorUnpacked> {
        // Stage 1: PolarQuant (with norm extraction and residual)
        let (mse_result, residual) = self.polar_quant.quantize_and_residual(batch, batch_size)?;

        // Stage 2: QJL on residual
        let qjl_result = self.qjl.quantize_batch(&residual, batch_size)?;

        Ok(CompressedVectorUnpacked {
            mse: mse_result,
            qjl: qjl_result,
            bit_width: self.bit_width,
        })
    }

    /// Dequantize back to approximate vectors.
    ///
    /// Returns flat `batch_size * d` f64 values.
    pub fn dequantize(&self, compressed: &CompressedVector) -> Result<Vec<f64>> {
        let mut x_mse = self.polar_quant.dequantize_batch_packed(&compressed.mse)?;
        let x_qjl = self.qjl.dequantize_batch_packed(&compressed.qjl)?;

        for (a, b) in x_mse.iter_mut().zip(x_qjl.iter()) {
            *a += *b;
        }
        Ok(x_mse)
    }

    /// Dequantize from in-memory raw representation (no bit unpacking).
    pub fn dequantize_unpacked(&self, compressed: &CompressedVectorUnpacked) -> Result<Vec<f64>> {
        if compressed.bit_width != self.bit_width {
            return Err(TurboQuantError::InvalidPackedMetadata {
                param: "bit_width",
                expected: self.bit_width as usize,
                got: compressed.bit_width as usize,
            });
        }
        let mut x_mse = self.polar_quant.dequantize_batch(&compressed.mse)?;
        let x_qjl = self.qjl.dequantize_batch(&compressed.qjl)?;
        for (a, b) in x_mse.iter_mut().zip(x_qjl.iter()) {
            *a += *b;
        }
        Ok(x_mse)
    }

    /// Compute total storage in bits for `n_vectors` compressed vectors.
    pub fn compressed_size_bits(&self, n_vectors: usize) -> usize {
        let per_vector_bits = self.d * self.bit_width as usize; // (b-1) + 1 bits per coordinate
        let norms_bits = size_of::<f32>() * 8; // Residual norm only (pure-Python semantics)
        n_vectors * (per_vector_bits + norms_bits)
    }

    /// Compression ratio vs original precision.
    pub fn compression_ratio(&self, original_bits_per_value: usize) -> f64 {
        let original = self.d * original_bits_per_value;
        let compressed = self.d * self.bit_width as usize + 32;
        original as f64 / compressed as f64
    }
}

/// MSE-only TurboQuant (Algorithm 1) — no QJL stage.
///
/// Use for V cache compression where MSE matters more than inner product.
#[derive(Clone)]
pub struct TurboQuantMse {
    pub d: usize,
    pub bit_width: u32,
    pub polar_quant: PolarQuant,
}

impl TurboQuantMse {
    pub fn new(d: usize, bit_width: u32, seed: u64, norm_correction: bool) -> Result<Self> {
        if !(1..=8).contains(&bit_width) {
            return Err(TurboQuantError::InvalidBitWidth {
                param: "bit_width",
                got: bit_width,
                min: 1,
                max: 8,
            });
        }
        let polar_quant = PolarQuant::new(d, bit_width, seed, norm_correction)?;
        Ok(Self {
            d,
            bit_width,
            polar_quant,
        })
    }

    /// Quantize a batch. Returns `PolarQuantResult`.
    pub fn quantize(&self, batch: &[f64], batch_size: usize) -> Result<PolarQuantResult> {
        self.polar_quant.quantize_batch(batch, batch_size)
    }

    /// Quantize a batch to packed payload format.
    pub fn quantize_packed(
        &self,
        batch: &[f64],
        batch_size: usize,
    ) -> Result<PackedPolarQuantResult> {
        self.polar_quant.quantize_batch_packed(batch, batch_size)
    }

    /// Dequantize a batch.
    pub fn dequantize(&self, result: &PolarQuantResult) -> Result<Vec<f64>> {
        self.polar_quant.dequantize_batch(result)
    }

    /// Dequantize from packed payload.
    pub fn dequantize_packed(&self, packed: &PackedPolarQuantResult) -> Result<Vec<f64>> {
        self.polar_quant.dequantize_batch_packed(packed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polar_quant::l2_norm;

    #[test]
    fn test_turboquant_roundtrip() {
        let d = 32;
        let tq = TurboQuant::new(d, 3, 42, true).unwrap();
        let x: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) / d as f64).collect();

        let compressed = tq.quantize(&x, 1).unwrap();
        let x_hat = tq.dequantize(&compressed).unwrap();
        assert_eq!(x_hat.len(), d);

        let error = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = l2_norm(&x);
        assert!(
            error / norm < 1.0,
            "relative error too high: {:.4}",
            error / norm
        );
    }

    #[test]
    fn test_turboquant_batch() {
        let d = 16;
        let tq = TurboQuant::new(d, 3, 42, true).unwrap();
        let batch: Vec<f64> = (0..d * 4).map(|i| (i as f64) * 0.01).collect();

        let compressed = tq.quantize(&batch, 4).unwrap();
        assert_eq!(compressed.mse.batch_size, 4);
        assert_eq!(compressed.qjl.batch_size, 4);

        let x_hat = tq.dequantize(&compressed).unwrap();
        assert_eq!(x_hat.len(), d * 4);
    }

    #[test]
    fn test_turboquant_compression_ratio() {
        let tq = TurboQuant::new(128, 3, 42, true).unwrap();
        let ratio = tq.compression_ratio(16);
        // 128*16 / (128*3 + 32) = 2048/416 ≈ 4.92
        assert!(ratio > 4.0 && ratio < 6.0, "ratio = {ratio}");
    }

    #[test]
    fn test_mse_only() {
        let d = 16;
        let tqm = TurboQuantMse::new(d, 3, 42, true).unwrap();
        let x: Vec<f64> = (0..d).map(|i| i as f64 + 1.0).collect();
        let result = tqm.quantize(&x, 1).unwrap();
        let x_hat = tqm.dequantize(&result).unwrap();
        assert_eq!(x_hat.len(), d);
    }

    #[test]
    fn test_unpacked_roundtrip_matches_packed() {
        let d = 32;
        let batch_size = 4;
        let tq = TurboQuant::new(d, 3, 42, true).unwrap();
        let batch: Vec<f64> = (0..d * batch_size)
            .map(|i| ((i as f64) * 0.013).sin())
            .collect();

        let packed = tq.quantize(&batch, batch_size).unwrap();
        let x_packed = tq.dequantize(&packed).unwrap();

        let unpacked = tq.quantize_unpacked(&batch, batch_size).unwrap();
        let x_unpacked = tq.dequantize_unpacked(&unpacked).unwrap();

        assert_eq!(x_packed.len(), x_unpacked.len());
        for (a, b) in x_packed.iter().zip(x_unpacked.iter()) {
            // Packed norms use f32, so allow tiny numeric drift.
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_turboquant_bit_width_1_errors() {
        let err = TurboQuant::new(16, 1, 42, true).err().unwrap();
        assert!(
            matches!(
                err,
                TurboQuantError::InvalidBitWidth {
                    param: "bit_width",
                    ..
                }
            ),
            "unexpected error: {err}"
        );
    }
}
