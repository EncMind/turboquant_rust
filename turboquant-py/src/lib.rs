//! Python bindings for TurboQuant via PyO3.
//!
//! Exposes the Rust core as a Python module `turboquant_rs` that can be used
//! as a drop-in accelerator for the Python prototype.
//!
//! Install with: `cd rust/turboquant-py && maturin develop --release`

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use turboquant_core::codebook;
use turboquant_core::error::TurboQuantError;
use turboquant_core::kv_cache::KvCacheCompressor as RustKvCacheCompressor;
use turboquant_core::polar_quant::PolarQuant as RustPolarQuant;
use turboquant_core::qjl::Qjl as RustQjl;
use turboquant_core::rotation;
use turboquant_core::turboquant::TurboQuant as RustTurboQuant;
use turboquant_core::utils;

// ============================================================
// Utility functions
// ============================================================

/// Pack {+1, -1} sign array into uint8 bitfield.
#[pyfunction]
fn pack_bits<'py>(
    py: Python<'py>,
    signs: PyReadonlyArray1<i8>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let s = signs
        .as_slice()
        .map_err(|_| PyValueError::new_err("signs must be a contiguous 1D NumPy array of int8"))?;
    let packed = utils::pack_bits(s);
    Ok(Array1::from_vec(packed).into_pyarray(py))
}

/// Unpack uint8 bitfield back to {+1, -1} signs.
#[pyfunction]
fn unpack_bits<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray1<u8>,
    d: usize,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let p = packed.as_slice().map_err(|_| {
        PyValueError::new_err("packed must be a contiguous 1D NumPy array of uint8")
    })?;
    let signs = utils::unpack_bits(p, d).map_err(to_py_value_error)?;
    Ok(Array1::from_vec(signs).into_pyarray(py))
}

/// Pack b-bit indices into compact byte array.
#[pyfunction]
fn pack_indices<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<u8>,
    bit_width: u8,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let idx = indices.as_slice().map_err(|_| {
        PyValueError::new_err("indices must be a contiguous 1D NumPy array of uint8")
    })?;
    let packed = utils::pack_indices(idx, bit_width).map_err(to_py_value_error)?;
    Ok(Array1::from_vec(packed).into_pyarray(py))
}

/// Fast Walsh-Hadamard Transform (in-place, returns new array).
#[pyfunction]
fn fast_walsh_hadamard_transform<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut data = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be a contiguous 1D NumPy array of float64"))?
        .to_vec();
    rotation::fast_walsh_hadamard_transform(&mut data).map_err(to_py_value_error)?;
    Ok(Array1::from_vec(data).into_pyarray(py))
}

/// Compute optimal centroids for given bit-width and dimension.
#[pyfunction]
fn optimal_centroids<'py>(
    py: Python<'py>,
    bit_width: u32,
    d: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = codebook::optimal_centroids(bit_width, d).map_err(to_py_value_error)?;
    Ok(Array1::from_vec(c).into_pyarray(py))
}

/// Find nearest centroid indices for each value.
#[pyfunction]
fn nearest_centroid_indices<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    centroids: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let v = values.as_slice().map_err(|_| {
        PyValueError::new_err("values must be a contiguous 1D NumPy array of float64")
    })?;
    let c = centroids.as_slice().map_err(|_| {
        PyValueError::new_err("centroids must be a contiguous 1D NumPy array of float64")
    })?;
    let indices = codebook::nearest_centroid_indices(v, c).map_err(to_py_value_error)?;
    Ok(Array1::from_vec(indices).into_pyarray(py))
}

/// Memory footprint calculation.
#[pyfunction]
fn memory_footprint_bytes<'py>(
    py: Python<'py>,
    n_vectors: usize,
    d: usize,
    bit_width: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mf = utils::memory_footprint_bytes(n_vectors, d, bit_width).map_err(to_py_value_error)?;
    let dict = PyDict::new(py);
    dict.set_item("mse_indices_bytes", mf.mse_indices_bytes)?;
    dict.set_item("qjl_signs_bytes", mf.qjl_signs_bytes)?;
    dict.set_item("norms_bytes", mf.norms_bytes)?;
    dict.set_item("total_bytes", mf.total_bytes)?;
    dict.set_item("original_fp16_bytes", mf.original_fp16_bytes)?;
    dict.set_item("compression_ratio", mf.compression_ratio)?;
    Ok(dict)
}

// ============================================================
// PolarQuant wrapper
// ============================================================

#[pyclass]
struct PolarQuant {
    inner: RustPolarQuant,
}

#[pymethods]
impl PolarQuant {
    #[new]
    #[pyo3(signature = (d, bit_width, seed=42, norm_correction=true))]
    fn new(d: usize, bit_width: u32, seed: u64, norm_correction: bool) -> PyResult<Self> {
        Ok(Self {
            inner: RustPolarQuant::new(d, bit_width, seed, norm_correction)
                .map_err(to_py_value_error)?,
        })
    }

    /// Quantize a single vector. Returns (indices, norm).
    fn quantize_single<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, f64)> {
        let xv = x.as_slice().map_err(|_| {
            PyValueError::new_err("x must be a contiguous 1D NumPy array of float64")
        })?;
        let (indices, norm) = self.inner.quantize_single(xv).map_err(to_py_value_error)?;
        Ok((Array1::from_vec(indices).into_pyarray(py), norm))
    }

    /// Dequantize a single vector.
    fn dequantize_single<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<u8>,
        norm: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let idx = indices.as_slice().map_err(|_| {
            PyValueError::new_err("indices must be a contiguous 1D NumPy array of uint8")
        })?;
        let result = self
            .inner
            .dequantize_single(idx, norm)
            .map_err(to_py_value_error)?;
        Ok(Array1::from_vec(result).into_pyarray(py))
    }
}

// ============================================================
// QJL wrapper
// ============================================================

#[pyclass]
struct Qjl {
    inner: RustQjl,
}

#[pymethods]
impl Qjl {
    #[new]
    #[pyo3(signature = (d, seed=123))]
    fn new(d: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: RustQjl::new(d, seed).map_err(to_py_value_error)?,
        })
    }

    /// Quantize a single residual. Returns (signs, norm).
    fn quantize_single<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray1<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<i8>>, f64)> {
        let rv = r.as_slice().map_err(|_| {
            PyValueError::new_err("r must be a contiguous 1D NumPy array of float64")
        })?;
        let (signs, norm) = self.inner.quantize_single(rv).map_err(to_py_value_error)?;
        Ok((Array1::from_vec(signs).into_pyarray(py), norm))
    }

    /// Dequantize signs back to approximate residual.
    fn dequantize_single<'py>(
        &self,
        py: Python<'py>,
        signs: PyReadonlyArray1<i8>,
        norm: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let sv = signs.as_slice().map_err(|_| {
            PyValueError::new_err("signs must be a contiguous 1D NumPy array of int8")
        })?;
        let result = self
            .inner
            .dequantize_single(sv, norm)
            .map_err(to_py_value_error)?;
        Ok(Array1::from_vec(result).into_pyarray(py))
    }
}

// ============================================================
// TurboQuant wrapper
// ============================================================

/// Compressed vector container returned from TurboQuant.quantize() (packed wire format).
#[pyclass]
#[derive(Clone)]
struct CompressedVector {
    inner: turboquant_core::turboquant::CompressedVector,
}

#[pymethods]
impl CompressedVector {
    #[getter]
    fn bit_width(&self) -> u32 {
        self.inner.bit_width
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.mse.batch_size
    }
}

/// In-memory compressed vector (no bit packing — faster for roundtrip benchmarks).
#[pyclass]
#[derive(Clone)]
struct CompressedVectorUnpacked {
    inner: turboquant_core::turboquant::CompressedVectorUnpacked,
}

#[pymethods]
impl CompressedVectorUnpacked {
    #[getter]
    fn bit_width(&self) -> u32 {
        self.inner.bit_width
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.mse.batch_size
    }
}

#[pyclass]
struct TurboQuant {
    inner: RustTurboQuant,
}

#[pymethods]
impl TurboQuant {
    #[new]
    #[pyo3(signature = (d, bit_width, seed=42, norm_correction=true))]
    fn new(d: usize, bit_width: u32, seed: u64, norm_correction: bool) -> PyResult<Self> {
        Ok(Self {
            inner: RustTurboQuant::new(d, bit_width, seed, norm_correction)
                .map_err(to_py_value_error)?,
        })
    }

    /// Quantize a batch of vectors to packed wire format (flat array, row-major).
    fn quantize(
        &self,
        batch: PyReadonlyArray1<f64>,
        batch_size: usize,
    ) -> PyResult<CompressedVector> {
        let data = batch.as_slice().map_err(|_| {
            PyValueError::new_err("batch must be a contiguous 1D NumPy array of float64")
        })?;
        Ok(CompressedVector {
            inner: self
                .inner
                .quantize(data, batch_size)
                .map_err(to_py_value_error)?,
        })
    }

    /// Quantize without bit packing (faster for in-memory roundtrips).
    fn quantize_unpacked(
        &self,
        batch: PyReadonlyArray1<f64>,
        batch_size: usize,
    ) -> PyResult<CompressedVectorUnpacked> {
        let data = batch.as_slice().map_err(|_| {
            PyValueError::new_err("batch must be a contiguous 1D NumPy array of float64")
        })?;
        Ok(CompressedVectorUnpacked {
            inner: self
                .inner
                .quantize_unpacked(data, batch_size)
                .map_err(to_py_value_error)?,
        })
    }

    /// Dequantize from packed wire format.
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        compressed: &CompressedVector,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let result = self
            .inner
            .dequantize(&compressed.inner)
            .map_err(to_py_value_error)?;
        Ok(Array1::from_vec(result).into_pyarray(py))
    }

    /// Dequantize from unpacked in-memory format (no bit unpacking overhead).
    fn dequantize_unpacked<'py>(
        &self,
        py: Python<'py>,
        compressed: &CompressedVectorUnpacked,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let result = self
            .inner
            .dequantize_unpacked(&compressed.inner)
            .map_err(to_py_value_error)?;
        Ok(Array1::from_vec(result).into_pyarray(py))
    }

    fn compression_ratio(&self, original_bits: usize) -> f64 {
        self.inner.compression_ratio(original_bits)
    }
}

// ============================================================
// KV Cache Compressor wrapper
// ============================================================

#[pyclass]
struct KvCacheCompressor {
    inner: RustKvCacheCompressor,
}

#[pymethods]
impl KvCacheCompressor {
    #[new]
    #[pyo3(signature = (head_dim, k_bits=3, v_bits=3, seed=42, norm_correction=true))]
    fn new(
        head_dim: usize,
        k_bits: u32,
        v_bits: u32,
        seed: u64,
        norm_correction: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RustKvCacheCompressor::new(head_dim, k_bits, v_bits, seed, norm_correction)
                .map_err(to_py_value_error)?,
        })
    }

    /// Compute memory usage statistics.
    fn memory_stats<'py>(
        &self,
        py: Python<'py>,
        seq_len: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.memory_stats(seq_len, num_layers, num_heads);
        let dict = PyDict::new(py);
        dict.set_item("original_mb", stats.original_mb)?;
        dict.set_item("wire_compressed_mb", stats.wire_compressed_mb)?;
        dict.set_item("in_memory_mb", stats.in_memory_mb)?;
        dict.set_item("wire_compression_ratio", stats.wire_compression_ratio)?;
        dict.set_item(
            "in_memory_compression_ratio",
            stats.in_memory_compression_ratio,
        )?;
        dict.set_item("compressed_mb", stats.compressed_mb)?;
        dict.set_item("compression_ratio", stats.compression_ratio)?;
        dict.set_item("k_bits_per_value", stats.k_bits_per_value)?;
        dict.set_item("v_bits_per_value", stats.v_bits_per_value)?;
        Ok(dict)
    }
}

// ============================================================
// Module definition
// ============================================================

/// TurboQuant Rust-accelerated Python bindings.
#[pymodule]
fn turboquant_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Utility functions
    m.add_function(wrap_pyfunction!(pack_bits, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bits, m)?)?;
    m.add_function(wrap_pyfunction!(pack_indices, m)?)?;
    m.add_function(wrap_pyfunction!(fast_walsh_hadamard_transform, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_centroids, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_centroid_indices, m)?)?;
    m.add_function(wrap_pyfunction!(memory_footprint_bytes, m)?)?;

    // Classes
    m.add_class::<PolarQuant>()?;
    m.add_class::<Qjl>()?;
    m.add_class::<TurboQuant>()?;
    m.add_class::<CompressedVector>()?;
    m.add_class::<CompressedVectorUnpacked>()?;
    m.add_class::<KvCacheCompressor>()?;

    Ok(())
}
fn to_py_value_error(err: TurboQuantError) -> PyErr {
    PyValueError::new_err(err.to_string())
}
