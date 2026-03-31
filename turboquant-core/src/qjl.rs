//! QJL: Quantized Johnson-Lindenstrauss Transform.
//!
//! 1-bit quantization via random projection -> sign to compress vectors while
//! preserving inner products.
//!
//! Key property: unbiased and optimal at 1-bit.
//!     Q_qjl(x) = sign(S * x) where S ~ N(0,1)^(d x d)
//!     Q_qjl_inv(z) = sqrt(pi/2) / d * S^T * z
//!
//! Mirrors `turboquant/qjl.py`.

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::mem::size_of;

use crate::error::{Result, TurboQuantError};
use crate::polar_quant::l2_norm;
use crate::utils::{pack_bits, unpack_bits};

/// sqrt(pi / 2) constant used in QJL dequantization.
const QJL_CONST: f64 = 1.2533141373155003; // sqrt(pi/2)

/// QJL result from quantization.
#[derive(Debug, Clone)]
pub struct QjlResult {
    /// Sign bits: flat `batch_size * d`, each `+1` or `-1`.
    pub signs: Vec<i8>,
    /// L2 norms of the original residual vectors.
    pub norms: Vec<f64>,
    pub batch_size: usize,
}

/// Packed QJL payload suitable for wire/storage format.
#[derive(Debug, Clone)]
pub struct PackedQjlResult {
    /// Packed sign bits (`+1/-1` mapped to 1/0).
    pub packed_signs: Vec<u8>,
    /// L2 norms for each residual vector (f32 for compactness).
    pub norms: Vec<f32>,
    pub batch_size: usize,
    pub d: usize,
}

impl QjlResult {
    /// Pack raw signs/norms into a compact payload.
    pub fn pack(&self, d: usize) -> Result<PackedQjlResult> {
        let expected_signs = self.batch_size * d;
        if self.signs.len() != expected_signs {
            return Err(TurboQuantError::LengthMismatch {
                param: "signs",
                expected: expected_signs,
                got: self.signs.len(),
            });
        }
        if self.norms.len() != self.batch_size {
            return Err(TurboQuantError::LengthMismatch {
                param: "norms",
                expected: self.batch_size,
                got: self.norms.len(),
            });
        }

        Ok(PackedQjlResult {
            packed_signs: pack_bits(&self.signs),
            norms: self.norms.iter().map(|&n| n as f32).collect(),
            batch_size: self.batch_size,
            d,
        })
    }
}

impl PackedQjlResult {
    /// Unpack to raw signs and f64 norms.
    pub fn unpack(&self) -> Result<QjlResult> {
        Ok(QjlResult {
            signs: unpack_bits(&self.packed_signs, self.batch_size * self.d)?,
            norms: self.norms.iter().map(|&n| n as f64).collect(),
            batch_size: self.batch_size,
        })
    }

    /// Unpack only sign values.
    pub fn unpack_signs(&self) -> Result<Vec<i8>> {
        unpack_bits(&self.packed_signs, self.batch_size * self.d)
    }

    /// Payload size for packed wire format.
    pub fn wire_size_bytes(&self) -> usize {
        self.packed_signs.len() + self.norms.len() * size_of::<f32>()
    }

    /// Approximate in-memory size (payload + Vec metadata).
    pub fn in_memory_size_bytes(&self) -> usize {
        size_of::<Self>() + self.packed_signs.len() + self.norms.len() * size_of::<f32>()
    }
}

/// Quantized Johnson-Lindenstrauss 1-bit quantizer.
///
/// Stores a full `d x d` random projection matrix. For production use at large `d`,
/// a structured/seeded approach would reduce memory.
#[derive(Clone)]
pub struct Qjl {
    pub d: usize,
    /// Random projection matrix S, shape `(d, d)`, entries ~ N(0, 1).
    s: DMatrix<f64>,
}

impl Qjl {
    /// Create a new QJL quantizer.
    ///
    /// # Arguments
    /// * `d` — vector dimension.
    /// * `seed` — random seed for the projection matrix.
    pub fn new(d: usize, seed: u64) -> Result<Self> {
        if d < 1 {
            return Err(TurboQuantError::InvalidDimension {
                param: "d",
                got: d,
                min: 1,
            });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let s = DMatrix::from_fn(d, d, |_, _| StandardNormal.sample(&mut rng));
        Ok(Self { d, s })
    }

    /// Return the projection matrix as a row-major flat buffer.
    ///
    /// This is exposed for Python API parity (`QJL.S`).
    pub fn projection_matrix_row_major(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.d * self.d);
        for i in 0..self.d {
            for j in 0..self.d {
                out.push(self.s[(i, j)]);
            }
        }
        out
    }

    /// Quantize a single residual vector to sign bits.
    ///
    /// Returns `(signs, norm)`.
    pub fn quantize_single(&self, r: &[f64]) -> Result<(Vec<i8>, f64)> {
        if r.len() != self.d {
            return Err(TurboQuantError::LengthMismatch {
                param: "r",
                expected: self.d,
                got: r.len(),
            });
        }
        let norm = l2_norm(r);

        // Project: S @ r
        let rv = nalgebra::DVector::from_column_slice(r);
        let projected = &self.s * &rv;

        // Sign quantization
        let signs: Vec<i8> = projected
            .iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect();

        Ok((signs, norm))
    }

    /// Quantize a batch of residual vectors using batch BLAS (Level 3).
    ///
    /// `batch` is flat `batch_size * d`, row-major.
    pub fn quantize_batch(&self, batch: &[f64], batch_size: usize) -> Result<QjlResult> {
        let expected_len = batch_size * self.d;
        if batch.len() != expected_len {
            return Err(TurboQuantError::LengthMismatch {
                param: "batch",
                expected: expected_len,
                got: batch.len(),
            });
        }

        if batch_size <= 1 {
            if batch_size == 1 {
                let (signs, norm) = self.quantize_single(batch)?;
                return Ok(QjlResult {
                    signs,
                    norms: vec![norm],
                    batch_size: 1,
                });
            }
            return Ok(QjlResult {
                signs: vec![],
                norms: vec![],
                batch_size: 0,
            });
        }

        // 1. Compute all norms
        let mut norms = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            norms.push(l2_norm(&batch[i * self.d..(i + 1) * self.d]));
        }

        // 2. Batch projection: S @ R^T = (R @ S^T) transposed
        //    In row-major: result(n,d) = R(n,d) @ S^T(d,d)
        //    S is column-major in nalgebra, so S.as_slice() is col-major = S^T in row-major
        let projected_row_major = crate::rotation::apply_dense_rotation_batch_raw(
            self.s.as_slice(),
            batch,
            self.d,
            batch_size,
            true, // batch @ S^T (= S @ batch^T transposed)
        )?;

        // 3. Sign quantization on the row-major projected result
        let all_signs: Vec<i8> = projected_row_major
            .iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect();

        Ok(QjlResult {
            signs: all_signs,
            norms,
            batch_size,
        })
    }

    /// Quantize and return packed payload.
    pub fn quantize_batch_packed(
        &self,
        batch: &[f64],
        batch_size: usize,
    ) -> Result<PackedQjlResult> {
        self.quantize_batch(batch, batch_size)?.pack(self.d)
    }

    /// Dequantize sign bits back to approximate residual.
    pub fn dequantize_single(&self, signs: &[i8], norm: f64) -> Result<Vec<f64>> {
        if signs.len() != self.d {
            return Err(TurboQuantError::LengthMismatch {
                param: "signs",
                expected: self.d,
                got: signs.len(),
            });
        }
        if let Some((index, &value)) = signs.iter().enumerate().find(|(_, &s)| s != -1 && s != 1) {
            return Err(TurboQuantError::InvalidSignValue {
                param: "signs",
                index,
                got: value,
            });
        }

        // Convert signs to f64
        let sv = nalgebra::DVector::from_iterator(self.d, signs.iter().map(|&s| s as f64));

        // S^T @ signs
        let reconstructed = self.s.transpose() * &sv;

        // Scale: sqrt(pi/2) / d * norm
        let scale = QJL_CONST / self.d as f64 * norm;

        Ok(reconstructed.iter().map(|&v| v * scale).collect())
    }

    /// Dequantize a batch using batch BLAS (Level 3).
    pub fn dequantize_batch(&self, result: &QjlResult) -> Result<Vec<f64>> {
        if result.norms.len() != result.batch_size {
            return Err(TurboQuantError::LengthMismatch {
                param: "result.norms",
                expected: result.batch_size,
                got: result.norms.len(),
            });
        }
        let expected_signs = result.batch_size * self.d;
        if result.signs.len() != expected_signs {
            return Err(TurboQuantError::LengthMismatch {
                param: "result.signs",
                expected: expected_signs,
                got: result.signs.len(),
            });
        }
        if let Some((index, &value)) = result
            .signs
            .iter()
            .enumerate()
            .find(|(_, &s)| s != -1 && s != 1)
        {
            return Err(TurboQuantError::InvalidSignValue {
                param: "result.signs",
                index,
                got: value,
            });
        }
        if result.batch_size <= 1 {
            if result.batch_size == 1 {
                return self.dequantize_single(&result.signs, result.norms[0]);
            }
            return Ok(vec![]);
        }

        // S^T @ Signs^T = (Signs @ S) in row-major
        // Signs is row-major (batch_size x d), S col-major
        let signs_f64: Vec<f64> = result.signs.iter().map(|&s| s as f64).collect();
        let mut reconstructed = crate::rotation::apply_dense_rotation_batch_raw(
            self.s.as_slice(),
            &signs_f64,
            self.d,
            result.batch_size,
            false, // signs @ S (= S^T @ signs^T transposed)
        )?;

        // Scale each vector by sqrt(pi/2) / d * norm
        for i in 0..result.batch_size {
            let scale = QJL_CONST / self.d as f64 * result.norms[i];
            for j in 0..self.d {
                reconstructed[i * self.d + j] *= scale;
            }
        }

        Ok(reconstructed)
    }

    /// Dequantize from a packed payload.
    pub fn dequantize_batch_packed(&self, packed: &PackedQjlResult) -> Result<Vec<f64>> {
        if packed.d != self.d {
            return Err(TurboQuantError::InvalidPackedMetadata {
                param: "d",
                expected: self.d,
                got: packed.d,
            });
        }
        self.dequantize_batch(&packed.unpack()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_preserves_direction() {
        let d = 32;
        let qjl = Qjl::new(d, 42).unwrap();
        let r: Vec<f64> = (0..d).map(|i| (i as f64) * 0.1 - 1.6).collect();
        let norm_r = l2_norm(&r);

        let (signs, norm) = qjl.quantize_single(&r).unwrap();
        assert!((norm - norm_r).abs() < 1e-12);
        assert_eq!(signs.len(), d);
        // All signs should be +1 or -1
        for &s in &signs {
            assert!(s == 1 || s == -1);
        }

        let r_hat = qjl.dequantize_single(&signs, norm).unwrap();
        assert_eq!(r_hat.len(), d);
    }

    #[test]
    fn test_zero_vector() {
        let d = 16;
        let qjl = Qjl::new(d, 42).unwrap();
        let r = vec![0.0; d];
        let (signs, norm) = qjl.quantize_single(&r).unwrap();
        assert!(norm < 1e-15);

        let r_hat = qjl.dequantize_single(&signs, norm).unwrap();
        for v in &r_hat {
            assert!(v.abs() < 1e-15, "zero input should give zero output");
        }
    }

    #[test]
    fn test_inner_product_preservation() {
        // QJL should approximately preserve inner products on average
        let d = 64;
        let qjl = Qjl::new(d, 42).unwrap();

        let x: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) / d as f64).collect();
        let y: Vec<f64> = (0..d).map(|i| (d as f64 - i as f64) / d as f64).collect();

        let ip_original: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        let (sx, nx) = qjl.quantize_single(&x).unwrap();
        let (sy, ny) = qjl.quantize_single(&y).unwrap();
        let x_hat = qjl.dequantize_single(&sx, nx).unwrap();
        let y_hat = qjl.dequantize_single(&sy, ny).unwrap();
        let ip_approx: f64 = x_hat.iter().zip(y_hat.iter()).map(|(a, b)| a * b).sum();

        // Should be in the right order of magnitude
        let ratio = ip_approx / ip_original;
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "inner product ratio way off: {ratio:.3} (orig={ip_original:.4}, approx={ip_approx:.4})"
        );
    }

    #[test]
    fn test_batch_consistency() {
        let d = 8;
        let qjl = Qjl::new(d, 42).unwrap();
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v2: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

        let (s1, n1) = qjl.quantize_single(&v1).unwrap();
        let (s2, n2) = qjl.quantize_single(&v2).unwrap();

        let mut batch = v1.clone();
        batch.extend(&v2);
        let result = qjl.quantize_batch(&batch, 2).unwrap();

        assert_eq!(&result.signs[..d], &s1[..]);
        assert_eq!(&result.signs[d..], &s2[..]);
        assert!((result.norms[0] - n1).abs() < 1e-12);
        assert!((result.norms[1] - n2).abs() < 1e-12);
    }
}
