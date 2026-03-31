//! Outlier channel strategy for non-integer bit precision.
//!
//! Split channels into outlier (higher bits) and non-outlier (lower bits) to achieve
//! fractional average bit rates like 2.5 or 3.5 bits per channel.
//!
//! Outlier channels are fixed-position (prefix/suffix split) to mirror the
//! pure-Python reference implementation.
//!
//! Examples:
//! - 2.5-bit: 50% channels at 3 bits + 50% at 2 bits
//! - 3.5-bit: 50% channels at 4 bits + 50% at 3 bits
//!
//! Mirrors `turboquant/outlier.py`.

use crate::error::{Result, TurboQuantError};
use crate::polar_quant::{PackedPolarQuantResult, PolarQuant};
use crate::qjl::{PackedQjlResult, Qjl};

/// Channel split configuration.
struct ChannelSplit {
    n_outlier: usize,
    high_bits: u32,
    n_normal: usize,
    low_bits: u32,
}

/// Compute how many channels get higher vs lower bit-width.
fn compute_channel_split(d: usize, target_bits: f64) -> ChannelSplit {
    let low_bits = target_bits.floor() as u32;
    let high_bits = low_bits + 1;
    let frac = target_bits - low_bits as f64;

    let n_outlier = (d as f64 * frac).round() as usize;
    let n_normal = d - n_outlier;

    ChannelSplit {
        n_outlier,
        high_bits,
        n_normal,
        low_bits,
    }
}

/// Result of outlier-strategy compression.
#[derive(Debug, Clone)]
pub struct OutlierCompressed {
    pub outlier: Option<PackedPolarQuantResult>,
    pub normal: Option<PackedPolarQuantResult>,
    pub qjl: PackedQjlResult,
    pub effective_bits: f64,
    pub batch_size: usize,
}

/// TurboQuant with outlier channel strategy for non-integer bit rates.
#[derive(Clone)]
pub struct OutlierTurboQuant {
    pub d: usize,
    pub target_bits: f64,
    pub n_outlier: usize,
    pub n_normal: usize,
    pub high_bits: u32,
    pub low_bits: u32,
    pub effective_bits: f64,
    outlier_idx: Vec<usize>,
    normal_idx: Vec<usize>,
    pq_outlier: Option<PolarQuant>,
    pq_normal: Option<PolarQuant>,
    qjl: Qjl,
}

impl OutlierTurboQuant {
    /// Create a new outlier-strategy quantizer.
    ///
    /// # Arguments
    /// * `d` — total vector dimension.
    /// * `target_bits` — target average bits per channel (e.g., 2.5, 3.5).
    /// * `seed` — random seed.
    pub fn new(d: usize, target_bits: f64, seed: u64) -> Result<Self> {
        if d < 1 {
            return Err(TurboQuantError::InvalidDimension {
                param: "d",
                got: d,
                min: 1,
            });
        }
        if !target_bits.is_finite() {
            return Err(TurboQuantError::InvalidTargetBits {
                got: target_bits,
                reason: "must be finite",
            });
        }
        let low = target_bits.floor();
        let high = target_bits.ceil();
        if !(low >= 2.0 && high <= 9.0) {
            return Err(TurboQuantError::InvalidTargetBits {
                got: target_bits,
                reason: "floor must be >=2 and ceil must be <=9",
            });
        }

        let split = compute_channel_split(d, target_bits);
        let effective_bits = (split.n_outlier as f64 * split.high_bits as f64
            + split.n_normal as f64 * split.low_bits as f64)
            / d as f64;

        let outlier_idx: Vec<usize> = (0..split.n_outlier).collect();
        let normal_idx: Vec<usize> = (split.n_outlier..d).collect();

        // PolarQuant bit-width is (total - 1) since QJL adds 1 bit
        let pq_outlier = if split.n_outlier > 0 {
            Some(PolarQuant::new(
                split.n_outlier,
                split.high_bits - 1,
                seed,
                true,
            )?)
        } else {
            None
        };
        let pq_normal = if split.n_normal > 0 {
            Some(PolarQuant::new(
                split.n_normal,
                split.low_bits - 1,
                seed + 500,
                true,
            )?)
        } else {
            None
        };

        let qjl = Qjl::new(d, seed + 1000)?;

        Ok(Self {
            d,
            target_bits,
            n_outlier: split.n_outlier,
            n_normal: split.n_normal,
            high_bits: split.high_bits,
            low_bits: split.low_bits,
            effective_bits,
            outlier_idx,
            normal_idx,
            pq_outlier,
            pq_normal,
            qjl,
        })
    }

    /// Fixed outlier channel indices (prefix split), matching pure-Python behavior.
    pub fn outlier_indices(&self) -> &[usize] {
        &self.outlier_idx
    }

    /// Fixed normal channel indices (suffix split), matching pure-Python behavior.
    pub fn normal_indices(&self) -> &[usize] {
        &self.normal_idx
    }

    /// Outlier-stream PolarQuant instance (high-bit channels), if present.
    pub fn outlier_quantizer(&self) -> Option<&PolarQuant> {
        self.pq_outlier.as_ref()
    }

    /// Normal-stream PolarQuant instance (low-bit channels), if present.
    pub fn normal_quantizer(&self) -> Option<&PolarQuant> {
        self.pq_normal.as_ref()
    }

    /// QJL instance used for residual correction.
    pub fn qjl_quantizer(&self) -> &Qjl {
        &self.qjl
    }

    /// Quantize a batch of vectors with outlier channel split.
    ///
    /// `batch` is flat `batch_size * d`, row-major.
    pub fn quantize(&self, batch: &[f64], batch_size: usize) -> Result<OutlierCompressed> {
        let expected_len = batch_size * self.d;
        if batch.len() != expected_len {
            return Err(TurboQuantError::LengthMismatch {
                param: "batch",
                expected: expected_len,
                got: batch.len(),
            });
        }

        // Split channels by fixed prefix/suffix indices.
        let mut outlier_data = Vec::with_capacity(batch_size * self.n_outlier);
        let mut normal_data = Vec::with_capacity(batch_size * self.n_normal);
        for i in 0..batch_size {
            let row = &batch[i * self.d..(i + 1) * self.d];
            for &idx in &self.outlier_idx {
                outlier_data.push(row[idx]);
            }
            for &idx in &self.normal_idx {
                normal_data.push(row[idx]);
            }
        }

        // Quantize outlier channels
        let (outlier_packed, out_residual) = if let Some(ref pq) = self.pq_outlier {
            let (result, residual) = pq.quantize_and_residual(&outlier_data, batch_size)?;
            (
                Some(result.pack(self.n_outlier, (self.high_bits - 1) as u8)?),
                residual,
            )
        } else {
            (None, vec![0.0; batch_size * self.n_outlier])
        };

        // Quantize normal channels
        let (normal_packed, norm_residual) = if let Some(ref pq) = self.pq_normal {
            let (result, residual) = pq.quantize_and_residual(&normal_data, batch_size)?;
            (
                Some(result.pack(self.n_normal, (self.low_bits - 1) as u8)?),
                residual,
            )
        } else {
            (None, vec![0.0; batch_size * self.n_normal])
        };

        // Reconstruct full residual
        let mut full_residual = vec![0.0; batch_size * self.d];
        for i in 0..batch_size {
            for (k, &idx) in self.outlier_idx.iter().enumerate() {
                full_residual[i * self.d + idx] = out_residual[i * self.n_outlier + k];
            }
            for (k, &idx) in self.normal_idx.iter().enumerate() {
                full_residual[i * self.d + idx] = norm_residual[i * self.n_normal + k];
            }
        }

        // QJL on full residual
        let qjl_result = self.qjl.quantize_batch_packed(&full_residual, batch_size)?;

        Ok(OutlierCompressed {
            outlier: outlier_packed,
            normal: normal_packed,
            qjl: qjl_result,
            effective_bits: self.effective_bits,
            batch_size,
        })
    }

    /// Dequantize outlier-strategy compressed vectors.
    pub fn dequantize(&self, compressed: &OutlierCompressed) -> Result<Vec<f64>> {
        let batch_size = compressed.batch_size;

        // Reconstruct outlier channels
        let outlier_recon = if let Some(ref pq) = self.pq_outlier {
            let packed = compressed
                .outlier
                .as_ref()
                .ok_or(TurboQuantError::MissingPayload { field: "outlier" })?;
            pq.dequantize_batch_packed(packed)?
        } else {
            vec![0.0; batch_size * self.n_outlier]
        };

        // Reconstruct normal channels
        let normal_recon = if let Some(ref pq) = self.pq_normal {
            let packed = compressed
                .normal
                .as_ref()
                .ok_or(TurboQuantError::MissingPayload { field: "normal" })?;
            pq.dequantize_batch_packed(packed)?
        } else {
            vec![0.0; batch_size * self.n_normal]
        };

        // QJL residual
        let qjl_recon = self.qjl.dequantize_batch_packed(&compressed.qjl)?;

        // Combine
        let mut output = vec![0.0; batch_size * self.d];
        for i in 0..batch_size {
            // Outlier channels
            for (k, &idx) in self.outlier_idx.iter().enumerate() {
                output[i * self.d + idx] = outlier_recon[i * self.n_outlier + k];
            }
            // Normal channels
            for (k, &idx) in self.normal_idx.iter().enumerate() {
                output[i * self.d + idx] = normal_recon[i * self.n_normal + k];
            }
            // Add QJL residual
            for j in 0..self.d {
                output[i * self.d + j] += qjl_recon[i * self.d + j];
            }
        }

        Ok(output)
    }

    /// Compression ratio vs original precision.
    pub fn compression_ratio(&self, original_bits: usize) -> f64 {
        // Pure-Python formula: effective bits + 32-bit QJL norm + two 32-bit PQ norms.
        let per_vector_bits = self.d as f64 * self.effective_bits + 96.0;
        let original = (self.d * original_bits) as f64;
        original / per_vector_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polar_quant::l2_norm;

    #[test]
    fn test_channel_split_2_5() {
        let split = compute_channel_split(128, 2.5);
        assert_eq!(split.low_bits, 2);
        assert_eq!(split.high_bits, 3);
        assert_eq!(split.n_outlier + split.n_normal, 128);
        let effective = (split.n_outlier as f64 * 3.0 + split.n_normal as f64 * 2.0) / 128.0;
        assert!((effective - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_outlier_roundtrip() {
        let d = 32;
        let oq = OutlierTurboQuant::new(d, 2.5, 42).unwrap();
        let x: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) / d as f64).collect();

        let compressed = oq.quantize(&x, 1).unwrap();
        let x_hat = oq.dequantize(&compressed).unwrap();
        assert_eq!(x_hat.len(), d);

        let error: f64 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = l2_norm(&x);
        assert!(
            error / norm < 1.5,
            "relative error too high: {:.4}",
            error / norm
        );
    }

    #[test]
    fn test_effective_bits() {
        let oq = OutlierTurboQuant::new(128, 3.5, 42).unwrap();
        assert!((oq.effective_bits - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_compression_ratio() {
        let oq = OutlierTurboQuant::new(128, 2.5, 42).unwrap();
        let ratio = oq.compression_ratio(16);
        assert!(ratio > 1.0, "should compress");
    }
}
