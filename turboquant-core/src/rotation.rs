//! Random rotation matrix generation for PolarQuant.
//!
//! Two implementations:
//! 1. **Dense Haar-distributed rotation** via QR decomposition — O(d^2) multiply, exact.
//! 2. **Fast structured rotation** via Hadamard + random sign flips — O(d log d), approximate.
//!
//! Mirrors `turboquant/rotation.py`.

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;
use crate::error::{Result, TurboQuantError};

/// Generate a Haar-distributed random rotation matrix via QR decomposition.
///
/// Returns an orthogonal matrix `Q` of shape `(d, d)` with `det(Q) = +1`.
///
pub fn random_rotation_dense(d: usize, rng: &mut impl Rng) -> Result<DMatrix<f64>> {
    if d < 1 {
        return Err(TurboQuantError::InvalidDimension {
            param: "d",
            got: d,
            min: 1,
        });
    }

    // Random Gaussian matrix
    let g = DMatrix::from_fn(d, d, |_, _| rng.sample::<f64, _>(StandardNormal));

    // QR decomposition
    let qr = g.qr();
    let mut q = qr.q();
    let r = qr.r();

    // Make Q Haar-distributed: fix signs via diagonal of R
    for j in 0..d {
        let sign = if r[(j, j)] >= 0.0 { 1.0 } else { -1.0 };
        // Multiply column j by sign
        for i in 0..d {
            q[(i, j)] *= sign;
        }
    }

    // Ensure proper rotation (det = +1). Flip first column if det = -1.
    let det = q.determinant();
    if det < 0.0 {
        for i in 0..d {
            q[(i, 0)] = -q[(i, 0)];
        }
    }

    Ok(q)
}

/// Return the smallest power of 2 >= n.
pub fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place Fast Walsh-Hadamard Transform, O(n log n).
///
/// Input must have length that is a power of 2. Result is normalized by `1/sqrt(n)`.
///
pub fn fast_walsh_hadamard_transform(x: &mut [f64]) -> Result<()> {
    let n = x.len();
    if !(n >= 1 && (n & (n - 1)) == 0) {
        return Err(TurboQuantError::InvalidDimension {
            param: "x.len() (power of 2 required)",
            got: n,
            min: 1,
        });
    }

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..(i + h) {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }

    let norm = (n as f64).sqrt();
    for v in x.iter_mut() {
        *v /= norm;
    }
    Ok(())
}

/// Components for a fast structured random rotation: D2 @ H @ D1.
///
/// The rotation is applied as:
/// 1. Pad `x` to `padded_d` (next power of 2)
/// 2. `x *= signs1`
/// 3. `x = WHT(x) / sqrt(padded_d)`
/// 4. `x *= signs2`
/// 5. Truncate back to `d`
#[derive(Debug, Clone)]
pub struct FastRotation {
    pub signs1: Vec<f64>,
    pub signs2: Vec<f64>,
    pub padded_d: usize,
    pub d: usize,
}

impl FastRotation {
    /// Generate a fast structured random rotation for dimension `d`.
    pub fn new(d: usize, rng: &mut impl Rng) -> Result<Self> {
        if d < 1 {
            return Err(TurboQuantError::InvalidDimension {
                param: "d",
                got: d,
                min: 1,
            });
        }
        let padded_d = next_power_of_2(d);
        let signs1: Vec<f64> = (0..padded_d)
            .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        let signs2: Vec<f64> = (0..padded_d)
            .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        Ok(Self {
            signs1,
            signs2,
            padded_d,
            d,
        })
    }

    /// Apply the structured rotation to a single vector.
    pub fn apply(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.d {
            return Err(TurboQuantError::LengthMismatch {
                param: "x",
                expected: self.d,
                got: x.len(),
            });
        }
        let mut padded = vec![0.0; self.padded_d];
        padded[..self.d].copy_from_slice(x);

        // D1 @ x
        padded
            .iter_mut()
            .zip(self.signs1.iter())
            .for_each(|(v, &s)| *v *= s);
        // H @ D1 @ x (normalized)
        fast_walsh_hadamard_transform(&mut padded)?;
        // D2 @ H @ D1 @ x
        padded
            .iter_mut()
            .zip(self.signs2.iter())
            .for_each(|(v, &s)| *v *= s);

        padded.truncate(self.d);
        Ok(padded)
    }

    /// Apply the transpose rotation (D1 @ H @ D2 — reversed order, since D and H are symmetric).
    pub fn apply_transpose(&self, y: &[f64]) -> Result<Vec<f64>> {
        if y.len() != self.d {
            return Err(TurboQuantError::LengthMismatch {
                param: "y",
                expected: self.d,
                got: y.len(),
            });
        }
        let mut padded = vec![0.0; self.padded_d];
        padded[..self.d].copy_from_slice(y);

        padded
            .iter_mut()
            .zip(self.signs2.iter())
            .for_each(|(v, &s)| *v *= s);
        fast_walsh_hadamard_transform(&mut padded)?;
        padded
            .iter_mut()
            .zip(self.signs1.iter())
            .for_each(|(v, &s)| *v *= s);

        padded.truncate(self.d);
        Ok(padded)
    }

    /// Apply the structured rotation to a batch of vectors (row-major).
    ///
    /// Each row of `batch` is a vector of dimension `d`.
    pub fn apply_batch(&self, batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        batch.iter().map(|row| self.apply(row)).collect()
    }

    /// Apply the structured rotation to a batch using rayon for parallelism.
    #[cfg(feature = "parallel")]
    pub fn apply_batch_parallel(&self, batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        use rayon::prelude::*;
        batch.par_iter().map(|row| self.apply(row)).collect()
    }
}

/// Vectorized batch Walsh-Hadamard transform.
///
/// Processes all rows of a `(batch, padded_d)` matrix using the butterfly pattern.
/// More cache-friendly than calling [`fast_walsh_hadamard_transform`] per row.
pub fn fast_walsh_hadamard_batch(data: &mut [Vec<f64>], padded_d: usize) -> Result<()> {
    let n = padded_d;
    if !(n >= 1 && (n & (n - 1)) == 0) {
        return Err(TurboQuantError::InvalidDimension {
            param: "padded_d (power of 2 required)",
            got: n,
            min: 1,
        });
    }
    if let Some((_idx, row)) = data.iter().enumerate().find(|(_, row)| row.len() != n) {
        return Err(TurboQuantError::LengthMismatch {
            param: "data[row].len()",
            expected: n,
            got: row.len(),
        });
    }

    let mut h = 1;
    while h < n {
        for row in data.iter_mut() {
            let mut i = 0;
            while i < n {
                for j in i..(i + h) {
                    let a = row[j];
                    let b = row[j + h];
                    row[j] = a + b;
                    row[j + h] = a - b;
                }
                i += h * 2;
            }
        }
        h *= 2;
    }

    let norm = (n as f64).sqrt();
    for row in data.iter_mut() {
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    Ok(())
}

/// Apply dense rotation to a single vector: `Q @ x`.
pub fn apply_dense_rotation(q: &DMatrix<f64>, x: &[f64]) -> Result<Vec<f64>> {
    if q.nrows() != q.ncols() {
        return Err(TurboQuantError::LengthMismatch {
            param: "q (square matrix)",
            expected: q.nrows(),
            got: q.ncols(),
        });
    }
    if x.len() != q.ncols() {
        return Err(TurboQuantError::LengthMismatch {
            param: "x",
            expected: q.ncols(),
            got: x.len(),
        });
    }
    let xv = DVector::from_column_slice(x);
    let result = q * &xv;
    Ok(result.as_slice().to_vec())
}

/// Apply dense rotation transpose to a single vector: `Q^T @ y`.
pub fn apply_dense_rotation_transpose(q: &DMatrix<f64>, y: &[f64]) -> Result<Vec<f64>> {
    if q.nrows() != q.ncols() {
        return Err(TurboQuantError::LengthMismatch {
            param: "q (square matrix)",
            expected: q.nrows(),
            got: q.ncols(),
        });
    }
    if y.len() != q.nrows() {
        return Err(TurboQuantError::LengthMismatch {
            param: "y",
            expected: q.nrows(),
            got: y.len(),
        });
    }
    let yv = DVector::from_column_slice(y);
    let result = q.transpose() * &yv;
    Ok(result.as_slice().to_vec())
}

/// Apply dense rotation to a batch of row vectors using Level 3 BLAS (matrix-matrix multiply).
///
/// Computes `(Q @ X^T)^T` where X is `(n x d)` row-major → result is `(n x d)` row-major.
/// When the `accelerate` feature is enabled, uses Apple Accelerate CBLAS dgemm directly.
/// Otherwise falls back to nalgebra's `matrixmultiply` crate.
pub fn apply_dense_rotation_batch(q: &DMatrix<f64>, batch: &[f64], d: usize) -> Result<Vec<f64>> {
    if d == 0 {
        return Err(TurboQuantError::InvalidDimension {
            param: "d",
            got: d,
            min: 1,
        });
    }
    if q.nrows() != d || q.ncols() != d {
        return Err(TurboQuantError::LengthMismatch {
            param: "q dimensions",
            expected: d,
            got: q.nrows().max(q.ncols()),
        });
    }
    if batch.len() % d != 0 {
        return Err(TurboQuantError::LengthMismatch {
            param: "batch",
            expected: ((batch.len() / d) + 1) * d,
            got: batch.len(),
        });
    }
    let n = batch.len() / d;
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return apply_dense_rotation(q, batch);
    }
    // (Q @ X^T)^T = X @ Q^T. With col-major Q data, transpose_m=false gives batch @ Q.
    // Wait — we want batch @ Q^T (since (Q @ X^T)^T = X @ Q^T).
    // Q stored col-major: data = Q. transpose_m=true → batch @ Q^T. Correct.
    // Actually let's think again: apply_dense_rotation_batch_raw(m, batch, d, n, transpose_m):
    //   transpose_m=false → batch @ M
    //   transpose_m=true  → batch @ M^T
    // We want X @ Q^T, so M=Q, transpose_m=true.
    apply_dense_rotation_batch_raw(q.as_slice(), batch, d, n, true)
}

/// Apply dense rotation transpose to a batch of row vectors using Level 3 BLAS.
///
/// Computes `(Q^T @ Y^T)^T` where Y is `(n x d)` row-major.
pub fn apply_dense_rotation_transpose_batch(
    q: &DMatrix<f64>,
    batch: &[f64],
    d: usize,
) -> Result<Vec<f64>> {
    if d == 0 {
        return Err(TurboQuantError::InvalidDimension {
            param: "d",
            got: d,
            min: 1,
        });
    }
    if q.nrows() != d || q.ncols() != d {
        return Err(TurboQuantError::LengthMismatch {
            param: "q dimensions",
            expected: d,
            got: q.nrows().max(q.ncols()),
        });
    }
    if batch.len() % d != 0 {
        return Err(TurboQuantError::LengthMismatch {
            param: "batch",
            expected: ((batch.len() / d) + 1) * d,
            got: batch.len(),
        });
    }
    let n = batch.len() / d;
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return apply_dense_rotation_transpose(q, batch);
    }
    // (Q^T @ Y^T)^T = Y @ Q. M=Q, transpose_m=false → batch @ Q. Correct.
    apply_dense_rotation_batch_raw(q.as_slice(), batch, d, n, false)
}

/// Batch matrix multiply: `result(n,d) = batch(n,d) @ op(M)^T`
///
/// Where M is stored column-major (d x d) as nalgebra does.
/// - `transpose_m = false`: computes `batch @ M` (i.e., `(M^T @ batch^T)^T` = rotation by M^T)
/// - `transpose_m = true`: computes `batch @ M^T` (i.e., `(M @ batch^T)^T` — but M^T)
///
/// Actually — let's be precise about what callers need:
/// - Rotation: `(Q @ X^T)^T = X @ Q^T`. Q is col-major → Q^T is row-major. So: `batch @ Q_as_row_major`.
///   With `transpose_q = false`, transB = Transpose (B^T = Q in math).
/// - Rotation transpose: `(Q^T @ Y^T)^T = Y @ Q`. So: `batch @ Q^T_as_row_major = batch @ Q_col_major_direct`.
///   With `transpose_q = true`, transB = None (B as-is = Q^T in math).
///
/// When `accelerate` feature is off, falls back to nalgebra.
pub fn apply_dense_rotation_batch_raw(
    m_col_major: &[f64],
    batch_row_major: &[f64],
    d: usize,
    n: usize,
    transpose_m: bool,
) -> Result<Vec<f64>> {
    if d < 1 {
        return Err(TurboQuantError::InvalidDimension {
            param: "d",
            got: d,
            min: 1,
        });
    }
    if m_col_major.len() != d * d {
        return Err(TurboQuantError::LengthMismatch {
            param: "m_col_major",
            expected: d * d,
            got: m_col_major.len(),
        });
    }
    if batch_row_major.len() != n * d {
        return Err(TurboQuantError::LengthMismatch {
            param: "batch_row_major",
            expected: n * d,
            got: batch_row_major.len(),
        });
    }
    #[cfg(feature = "accelerate")]
    {
        extern crate accelerate_src;
        use cblas::{dgemm, Layout, Transpose};

        let mut out = vec![0.0; n * d];

        // m_col_major is column-major (d x d).
        // In row-major interpretation, this data is M^T.
        // We want:
        //   transpose_m=false → C = batch @ M   → C = A @ B where B=M → transB=Trans (since data is M^T)
        //   transpose_m=true  → C = batch @ M^T → C = A @ B where B=M^T → transB=None (data is already M^T)
        let trans_b = if transpose_m {
            Transpose::None
        } else {
            Transpose::Ordinary
        };

        unsafe {
            dgemm(
                Layout::RowMajor,
                Transpose::None, // A = batch, no transpose
                trans_b,         // B = M data
                n as i32,        // M (rows of C)
                d as i32,        // N (cols of C)
                d as i32,        // K (shared dimension)
                1.0,             // alpha
                batch_row_major, // A
                d as i32,        // lda
                m_col_major,     // B
                d as i32,        // ldb
                0.0,             // beta
                &mut out,        // C
                d as i32,        // ldc
            );
        }
        Ok(out)
    }

    #[cfg(not(feature = "accelerate"))]
    {
        let m = DMatrix::from_column_slice(d, d, m_col_major);
        let x_mat = DMatrix::from_row_slice(n, d, batch_row_major);
        let result = if transpose_m {
            // batch @ M^T = (M @ batch^T)^T
            (m * x_mat.transpose()).transpose()
        } else {
            // batch @ M = (M^T @ batch^T)^T
            (m.transpose() * x_mat.transpose()).transpose()
        };
        let mut out = Vec::with_capacity(n * d);
        for i in 0..n {
            out.extend_from_slice(result.row(i).transpose().as_slice());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_fwht_identity() {
        // WHT of [1, 0, 0, 0] should be [0.5, 0.5, 0.5, 0.5] (normalized)
        let mut x = vec![1.0, 0.0, 0.0, 0.0];
        fast_walsh_hadamard_transform(&mut x).unwrap();
        let expected = 1.0 / 2.0; // 1/sqrt(4) * 1 (only first element contributes)
        for &v in &x {
            assert!((v - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_fwht_involution() {
        // WHT applied twice (with normalization) gives back the original.
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = original.clone();
        fast_walsh_hadamard_transform(&mut x).unwrap();
        fast_walsh_hadamard_transform(&mut x).unwrap();
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-10, "{a} != {b}");
        }
    }

    #[test]
    fn test_dense_rotation_orthogonal() {
        let mut rng = StdRng::seed_from_u64(42);
        let d = 16;
        let q = random_rotation_dense(d, &mut rng).unwrap();
        let qtq = q.transpose() * &q;
        let eye = DMatrix::<f64>::identity(d, d);
        assert!((qtq - eye).norm() < 1e-10);
    }

    #[test]
    fn test_dense_rotation_det_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        let q = random_rotation_dense(32, &mut rng).unwrap();
        assert!(q.determinant() > 0.0);
    }

    #[test]
    fn test_fast_rotation_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let d = 16;
        let rot = FastRotation::new(d, &mut rng).unwrap();
        let x: Vec<f64> = (0..d).map(|i| i as f64).collect();
        let y = rot.apply(&x).unwrap();
        let x_back = rot.apply_transpose(&y).unwrap();
        for (a, b) in x.iter().zip(x_back.iter()) {
            assert!((a - b).abs() < 1e-10, "{a} != {b}");
        }
    }

    #[test]
    fn test_fast_rotation_preserves_norm() {
        let mut rng = StdRng::seed_from_u64(99);
        let d = 64;
        let rot = FastRotation::new(d, &mut rng).unwrap();
        let x: Vec<f64> = (0..d).map(|i| (i as f64) * 0.1).collect();
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let y = rot.apply(&x).unwrap();
        let norm_y: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm_x - norm_y).abs() < 1e-8,
            "norms differ: {norm_x} vs {norm_y}"
        );
    }
}
