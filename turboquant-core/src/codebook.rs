//! Codebook construction for PolarQuant.
//!
//! After random rotation, each coordinate follows Beta(d/2, d/2) on [-1/sqrt(d), 1/sqrt(d)],
//! which converges to N(0, 1/d) for large d. We compute optimal scalar quantizers for this
//! distribution.
//!
//! Paper provides closed-form centroids for 1-bit and 2-bit. For higher bit-widths,
//! we use Lloyd's algorithm on the Gaussian approximation.
//!
//! Mirrors `turboquant/codebook.py`.

use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::PI;
use crate::error::{Result, TurboQuantError};

/// Compute optimal MSE centroids for the post-rotation coordinate distribution.
///
/// # Arguments
/// * `bit_width` — bits per coordinate (1, 2, 3, 4, ...).
/// * `d` — vector dimension (affects centroid scale via `1/sqrt(d)`).
///
/// # Returns
/// Sorted `Vec<f64>` of `2^bit_width` centroids.
pub fn optimal_centroids(bit_width: u32, d: usize) -> Result<Vec<f64>> {
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
    let n_centroids = 1usize << bit_width;

    if bit_width == 1 {
        let c = (2.0 / (PI * d as f64)).sqrt();
        return Ok(vec![-c, c]);
    }

    if bit_width == 2 {
        let scale = 1.0 / (d as f64).sqrt();
        return Ok(vec![-1.51 * scale, -0.453 * scale, 0.453 * scale, 1.51 * scale]);
    }

    // For b >= 3, use Lloyd's algorithm on N(0, 1/d)
    let sigma = 1.0 / (d as f64).sqrt();
    lloyds_gaussian(n_centroids, sigma, 100)
}

/// Lloyd's algorithm for optimal scalar quantization of N(0, sigma^2).
///
/// Iteratively refines centroid positions to minimize MSE distortion.
///
/// # Arguments
/// * `n_centroids` — number of quantization levels.
/// * `sigma` — standard deviation of the Gaussian.
/// * `n_iter` — number of Lloyd iterations.
fn lloyds_gaussian(n_centroids: usize, sigma: f64, n_iter: usize) -> Result<Vec<f64>> {
    let normal = Normal::new(0.0, sigma).map_err(|e| TurboQuantError::Internal {
        context: "failed to initialize Gaussian for Lloyd's algorithm",
        message: e.to_string(),
    })?;

    // Initialize boundaries from uniform quantiles
    let mut boundaries: Vec<f64> = (1..n_centroids)
        .map(|i| {
            let p = i as f64 / n_centroids as f64;
            normal.inverse_cdf(p)
        })
        .collect();

    let mut centroids = vec![0.0; n_centroids];

    // Initial centroids: conditional expectations within each region
    centroids[0] = gaussian_conditional_expectation(sigma, f64::NEG_INFINITY, boundaries[0])?;
    for i in 1..n_centroids - 1 {
        centroids[i] = gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])?;
    }
    centroids[n_centroids - 1] = gaussian_conditional_expectation(
        sigma,
        boundaries[n_centroids - 2],
        f64::INFINITY,
    )?;

    for _ in 0..n_iter {
        // Update boundaries (midpoints between consecutive centroids)
        for i in 0..n_centroids - 1 {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        // Update centroids
        centroids[0] = gaussian_conditional_expectation(sigma, f64::NEG_INFINITY, boundaries[0])?;
        for i in 1..n_centroids - 1 {
            centroids[i] =
                gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])?;
        }
        centroids[n_centroids - 1] = gaussian_conditional_expectation(
            sigma,
            boundaries[n_centroids - 2],
            f64::INFINITY,
        )?;
    }

    centroids.sort_by(|a, b| a.total_cmp(b));
    Ok(centroids)
}

/// E[X | a < X < b] where X ~ N(0, sigma^2).
///
/// Uses: `E[X | a < X < b] = sigma^2 * (phi(a/s) - phi(b/s)) / (Phi(b/s) - Phi(a/s))`
/// where phi is PDF and Phi is CDF of standard normal.
fn gaussian_conditional_expectation(sigma: f64, a: f64, b: f64) -> Result<f64> {
    let std_normal = Normal::new(0.0, 1.0).map_err(|e| TurboQuantError::Internal {
        context: "failed to initialize standard normal",
        message: e.to_string(),
    })?;

    let a_std = if a.is_finite() { a / sigma } else { a };
    let b_std = if b.is_finite() { b / sigma } else { b };

    // P(a/s < Z < b/s) where Z ~ N(0,1)
    let prob = if !a_std.is_finite() {
        std_normal.cdf(b_std)
    } else if !b_std.is_finite() {
        1.0 - std_normal.cdf(a_std) // sf
    } else {
        std_normal.cdf(b_std) - std_normal.cdf(a_std)
    };

    if prob < 1e-15 {
        if a.is_finite() && !b.is_finite() {
            return Ok(a + sigma);
        } else if !a.is_finite() && b.is_finite() {
            return Ok(b - sigma);
        } else if a.is_finite() && b.is_finite() {
            return Ok((a + b) / 2.0);
        } else {
            return Ok(0.0);
        }
    }

    let pdf_a = if a_std.is_finite() {
        standard_normal_pdf(a_std)
    } else {
        0.0
    };
    let pdf_b = if b_std.is_finite() {
        standard_normal_pdf(b_std)
    } else {
        0.0
    };

    Ok(sigma * (pdf_a - pdf_b) / prob)
}

/// Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi).
#[inline]
fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Find nearest centroid index for each value (vectorized).
///
/// Uses binary search on midpoint boundaries — O(n log k).
///
/// # Arguments
/// * `values` — values to quantize.
/// * `centroids` — sorted centroid array of length `k`.
///
/// # Returns
/// `Vec<u8>` of indices into `centroids`, same length as `values`.
pub fn nearest_centroid_indices(values: &[f64], centroids: &[f64]) -> Result<Vec<u8>> {
    if centroids.is_empty() {
        return Err(TurboQuantError::EmptyInput { param: "centroids" });
    }
    if centroids.len() > (u8::MAX as usize + 1) {
        return Err(TurboQuantError::TooManyLevels {
            param: "centroids",
            got: centroids.len(),
            max: u8::MAX as usize + 1,
        });
    }
    if let Some((idx, &value)) = centroids.iter().enumerate().find(|(_, c)| !c.is_finite()) {
        return Err(TurboQuantError::NonFiniteValue {
            param: "centroids",
            index: Some(idx),
            value,
        });
    }
    if !centroids.windows(2).all(|w| w[0] <= w[1]) {
        return Err(TurboQuantError::UnsortedInput { param: "centroids" });
    }

    // Precompute boundaries (midpoints between consecutive centroids)
    let boundaries: Vec<f64> = centroids.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

    let mut indices = Vec::with_capacity(values.len());
    for (idx, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(TurboQuantError::NonFiniteValue {
                param: "values",
                index: Some(idx),
                value: v,
            });
        }
        // Binary search for insertion point in boundaries
        let centroid_idx = match boundaries.binary_search_by(|b| b.total_cmp(&v)) {
            Ok(pos) => (pos + 1) as u8, // exactly on boundary → right centroid
            Err(pos) => pos as u8,
        };
        indices.push(centroid_idx);
    }
    Ok(indices)
}

/// Look up centroid values from indices.
#[inline]
pub fn centroid_lookup(indices: &[u8], centroids: &[f64]) -> Result<Vec<f64>> {
    let mut values = Vec::with_capacity(indices.len());
    for (pos, &i) in indices.iter().enumerate() {
        let idx = i as usize;
        if idx >= centroids.len() {
            return Err(TurboQuantError::ValueOutOfRange {
                param: "indices",
                index: pos,
                got: i,
                max: centroids.len().saturating_sub(1) as u8,
            });
        }
        values.push(centroids[idx]);
    }
    Ok(values)
}

/// Fast centroid lookup without per-index bounds checks.
///
/// Caller must guarantee all indices are in-range for `centroids`.
#[inline]
pub(crate) fn centroid_lookup_unchecked(indices: &[u8], centroids: &[f64]) -> Vec<f64> {
    let mut values = Vec::with_capacity(indices.len());
    for &i in indices {
        // SAFETY: caller guarantees `i as usize < centroids.len()`.
        values.push(unsafe { *centroids.get_unchecked(i as usize) });
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1bit_centroids_symmetric() {
        let c = optimal_centroids(1, 128).unwrap();
        assert_eq!(c.len(), 2);
        assert!(
            (c[0] + c[1]).abs() < 1e-12,
            "1-bit centroids must be symmetric"
        );
    }

    #[test]
    fn test_2bit_centroids() {
        let c = optimal_centroids(2, 128).unwrap();
        assert_eq!(c.len(), 4);
        // Should be symmetric: c[0] = -c[3], c[1] = -c[2]
        assert!((c[0] + c[3]).abs() < 1e-10);
        assert!((c[1] + c[2]).abs() < 1e-10);
    }

    #[test]
    fn test_3bit_centroids_sorted() {
        let c = optimal_centroids(3, 128).unwrap();
        assert_eq!(c.len(), 8);
        for w in c.windows(2) {
            assert!(w[0] <= w[1], "centroids must be sorted");
        }
    }

    #[test]
    fn test_nearest_centroid_basic() {
        let centroids = vec![-1.0, 0.0, 1.0];
        let values = vec![-0.8, 0.1, 0.6, -0.3];
        let indices = nearest_centroid_indices(&values, &centroids).unwrap();
        assert_eq!(indices, vec![0, 1, 2, 1]);
    }

    #[test]
    fn test_centroid_lookup() {
        let centroids = vec![0.1, 0.5, 0.9];
        let indices = vec![0, 2, 1, 0];
        let values = centroid_lookup(&indices, &centroids).unwrap();
        assert_eq!(values, vec![0.1, 0.9, 0.5, 0.1]);
    }

    #[test]
    fn test_lloyds_convergence() {
        // For N(0,1), 2-level Lloyd's should converge to ±0.7979 (1/sqrt(pi/2))
        let c = lloyds_gaussian(2, 1.0, 200).unwrap();
        assert_eq!(c.len(), 2);
        let expected = (2.0 / PI).sqrt(); // ≈ 0.7979
        assert!(
            (c[1] - expected).abs() < 0.01,
            "expected ~{expected}, got {}",
            c[1]
        );
    }
}
