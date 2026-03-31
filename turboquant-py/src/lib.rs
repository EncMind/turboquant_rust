//! Python bindings for TurboQuant via PyO3.
//!
//! Exposes the Rust core as a Python module `turboquant_rs` that can be used
//! as a drop-in accelerator for the Python prototype.
//!
//! Install with: `cd rust/turboquant-py && maturin develop --release`

use numpy::ndarray::{Array1, Array2, Array4};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArray1, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use turboquant_core::codebook;
use turboquant_core::error::TurboQuantError;
use turboquant_core::kv_cache::{
    CompressedKvCache as RustCompressedKvCache, KvCacheCompressor as RustKvCacheCompressor,
};
use turboquant_core::outlier::{
    OutlierCompressed as RustOutlierCompressed, OutlierTurboQuant as RustOutlierTurboQuant,
};
use turboquant_core::polar_quant::{PolarQuant as RustPolarQuant, PolarQuantResult};
use turboquant_core::qjl::{Qjl as RustQjl, QjlResult};
use turboquant_core::rotation;
use turboquant_core::turboquant::{
    TurboQuant as RustTurboQuant, TurboQuantMse as RustTurboQuantMse,
};
use turboquant_core::utils;

const SHAPE_VECTOR_1D: u8 = 0;
const SHAPE_MATRIX_2D: u8 = 1;
const SHAPE_FLAT_1D: u8 = 2;

struct ParsedF64Batch {
    data: Vec<f64>,
    batch_size: usize,
    shape_mode: u8,
}

struct ParsedU8Batch {
    data: Vec<u8>,
    batch_size: usize,
    shape_mode: u8,
}

struct ParsedI8Batch {
    data: Vec<i8>,
    batch_size: usize,
    shape_mode: u8,
}

fn parse_f64_batch(
    batch: PyReadonlyArrayDyn<f64>,
    d: usize,
    batch_size: Option<usize>,
    param: &'static str,
) -> PyResult<ParsedF64Batch> {
    let shape = batch.shape();
    match shape.len() {
        1 => {
            let data = batch.as_slice().map_err(|_| {
                PyValueError::new_err(format!(
                    "{param} must be a contiguous NumPy array of float64"
                ))
            })?;
            if let Some(bs) = batch_size {
                let expected = bs * d;
                if data.len() != expected {
                    return Err(PyValueError::new_err(format!(
                        "{param} length mismatch: expected {expected}, got {}",
                        data.len()
                    )));
                }
                return Ok(ParsedF64Batch {
                    data: data.to_vec(),
                    batch_size: bs,
                    shape_mode: SHAPE_FLAT_1D,
                });
            }
            if data.len() != d {
                return Err(PyValueError::new_err(format!(
                    "{param} must have length {d} for single-vector input, or pass batch_size for flat batches"
                )));
            }
            Ok(ParsedF64Batch {
                data: data.to_vec(),
                batch_size: 1,
                shape_mode: SHAPE_VECTOR_1D,
            })
        }
        2 => {
            let bs = shape[0];
            let dim = shape[1];
            if dim != d {
                return Err(PyValueError::new_err(format!(
                    "{param} shape mismatch: expected second dim {d}, got {dim}"
                )));
            }
            if let Some(expected_bs) = batch_size {
                if expected_bs != bs {
                    return Err(PyValueError::new_err(format!(
                        "{param} batch_size mismatch: expected {expected_bs}, got {bs}"
                    )));
                }
            }
            let data = batch.as_slice().map_err(|_| {
                PyValueError::new_err(format!(
                    "{param} must be a contiguous NumPy array of float64"
                ))
            })?;
            Ok(ParsedF64Batch {
                data: data.to_vec(),
                batch_size: bs,
                shape_mode: SHAPE_MATRIX_2D,
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "{param} must be a 1D or 2D NumPy array"
        ))),
    }
}

fn parse_u8_batch(
    values: PyReadonlyArrayDyn<u8>,
    d: usize,
    batch_size: Option<usize>,
    param: &'static str,
) -> PyResult<ParsedU8Batch> {
    let shape = values.shape();
    match shape.len() {
        1 => {
            let data = values.as_slice().map_err(|_| {
                PyValueError::new_err(format!("{param} must be a contiguous NumPy array of uint8"))
            })?;
            if let Some(bs) = batch_size {
                let expected = bs * d;
                if data.len() != expected {
                    return Err(PyValueError::new_err(format!(
                        "{param} length mismatch: expected {expected}, got {}",
                        data.len()
                    )));
                }
                return Ok(ParsedU8Batch {
                    data: data.to_vec(),
                    batch_size: bs,
                    shape_mode: SHAPE_FLAT_1D,
                });
            }
            if data.len() != d {
                return Err(PyValueError::new_err(format!(
                    "{param} must have length {d} for single-vector input, or pass batch_size for flat batches"
                )));
            }
            Ok(ParsedU8Batch {
                data: data.to_vec(),
                batch_size: 1,
                shape_mode: SHAPE_VECTOR_1D,
            })
        }
        2 => {
            let bs = shape[0];
            let dim = shape[1];
            if dim != d {
                return Err(PyValueError::new_err(format!(
                    "{param} shape mismatch: expected second dim {d}, got {dim}"
                )));
            }
            if let Some(expected_bs) = batch_size {
                if expected_bs != bs {
                    return Err(PyValueError::new_err(format!(
                        "{param} batch_size mismatch: expected {expected_bs}, got {bs}"
                    )));
                }
            }
            let data = values.as_slice().map_err(|_| {
                PyValueError::new_err(format!("{param} must be a contiguous NumPy array of uint8"))
            })?;
            Ok(ParsedU8Batch {
                data: data.to_vec(),
                batch_size: bs,
                shape_mode: SHAPE_MATRIX_2D,
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "{param} must be a 1D or 2D NumPy array"
        ))),
    }
}

fn parse_i8_batch(
    values: PyReadonlyArrayDyn<i8>,
    d: usize,
    batch_size: Option<usize>,
    param: &'static str,
) -> PyResult<ParsedI8Batch> {
    let shape = values.shape();
    match shape.len() {
        1 => {
            let data = values.as_slice().map_err(|_| {
                PyValueError::new_err(format!("{param} must be a contiguous NumPy array of int8"))
            })?;
            if let Some(bs) = batch_size {
                let expected = bs * d;
                if data.len() != expected {
                    return Err(PyValueError::new_err(format!(
                        "{param} length mismatch: expected {expected}, got {}",
                        data.len()
                    )));
                }
                return Ok(ParsedI8Batch {
                    data: data.to_vec(),
                    batch_size: bs,
                    shape_mode: SHAPE_FLAT_1D,
                });
            }
            if data.len() != d {
                return Err(PyValueError::new_err(format!(
                    "{param} must have length {d} for single-vector input, or pass batch_size for flat batches"
                )));
            }
            Ok(ParsedI8Batch {
                data: data.to_vec(),
                batch_size: 1,
                shape_mode: SHAPE_VECTOR_1D,
            })
        }
        2 => {
            let bs = shape[0];
            let dim = shape[1];
            if dim != d {
                return Err(PyValueError::new_err(format!(
                    "{param} shape mismatch: expected second dim {d}, got {dim}"
                )));
            }
            if let Some(expected_bs) = batch_size {
                if expected_bs != bs {
                    return Err(PyValueError::new_err(format!(
                        "{param} batch_size mismatch: expected {expected_bs}, got {bs}"
                    )));
                }
            }
            let data = values.as_slice().map_err(|_| {
                PyValueError::new_err(format!("{param} must be a contiguous NumPy array of int8"))
            })?;
            Ok(ParsedI8Batch {
                data: data.to_vec(),
                batch_size: bs,
                shape_mode: SHAPE_MATRIX_2D,
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "{param} must be a 1D or 2D NumPy array"
        ))),
    }
}

fn parse_norms(
    norms: &Bound<'_, PyAny>,
    batch_size: usize,
    param: &'static str,
) -> PyResult<Vec<f64>> {
    if batch_size == 1 {
        if let Ok(v) = norms.extract::<f64>() {
            return Ok(vec![v]);
        }
    }

    let arr: PyReadonlyArrayDyn<f64> = norms.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{param} must be a float or contiguous NumPy array of float64"
        ))
    })?;

    match arr.ndim() {
        0 => {
            if batch_size != 1 {
                return Err(PyValueError::new_err(format!(
                    "{param} scalar only valid for batch_size=1"
                )));
            }
            let v =
                *arr.as_array().iter().next().ok_or_else(|| {
                    PyValueError::new_err(format!("{param} scalar value missing"))
                })?;
            Ok(vec![v])
        }
        1 => {
            let data = arr.as_slice().map_err(|_| {
                PyValueError::new_err(format!("{param} must be contiguous in memory"))
            })?;
            if data.len() != batch_size {
                return Err(PyValueError::new_err(format!(
                    "{param} length mismatch: expected {batch_size}, got {}",
                    data.len()
                )));
            }
            Ok(data.to_vec())
        }
        _ => Err(PyValueError::new_err(format!(
            "{param} must be scalar or 1D"
        ))),
    }
}

fn reshape_f64<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    d: usize,
    batch_size: usize,
    shape_mode: u8,
) -> PyResult<Py<PyAny>> {
    match shape_mode {
        SHAPE_VECTOR_1D => {
            if batch_size != 1 || values.len() != d {
                return Err(PyValueError::new_err(
                    "invalid vector output shape metadata",
                ));
            }
            Ok(Array1::from_vec(values)
                .into_pyarray(py)
                .into_any()
                .unbind())
        }
        SHAPE_MATRIX_2D => {
            let arr = Array2::from_shape_vec((batch_size, d), values).map_err(|e| {
                PyValueError::new_err(format!("failed to reshape output to (batch, d): {e}"))
            })?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        SHAPE_FLAT_1D => Ok(Array1::from_vec(values)
            .into_pyarray(py)
            .into_any()
            .unbind()),
        _ => Err(PyValueError::new_err("invalid output shape metadata")),
    }
}

fn reshape_u8<'py>(
    py: Python<'py>,
    values: Vec<u8>,
    d: usize,
    batch_size: usize,
    shape_mode: u8,
) -> PyResult<Py<PyAny>> {
    match shape_mode {
        SHAPE_VECTOR_1D => {
            if batch_size != 1 || values.len() != d {
                return Err(PyValueError::new_err(
                    "invalid vector output shape metadata",
                ));
            }
            Ok(Array1::from_vec(values)
                .into_pyarray(py)
                .into_any()
                .unbind())
        }
        SHAPE_MATRIX_2D => {
            let arr = Array2::from_shape_vec((batch_size, d), values).map_err(|e| {
                PyValueError::new_err(format!("failed to reshape output to (batch, d): {e}"))
            })?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        SHAPE_FLAT_1D => Ok(Array1::from_vec(values)
            .into_pyarray(py)
            .into_any()
            .unbind()),
        _ => Err(PyValueError::new_err("invalid output shape metadata")),
    }
}

fn reshape_i8<'py>(
    py: Python<'py>,
    values: Vec<i8>,
    d: usize,
    batch_size: usize,
    shape_mode: u8,
) -> PyResult<Py<PyAny>> {
    match shape_mode {
        SHAPE_VECTOR_1D => {
            if batch_size != 1 || values.len() != d {
                return Err(PyValueError::new_err(
                    "invalid vector output shape metadata",
                ));
            }
            Ok(Array1::from_vec(values)
                .into_pyarray(py)
                .into_any()
                .unbind())
        }
        SHAPE_MATRIX_2D => {
            let arr = Array2::from_shape_vec((batch_size, d), values).map_err(|e| {
                PyValueError::new_err(format!("failed to reshape output to (batch, d): {e}"))
            })?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        SHAPE_FLAT_1D => Ok(Array1::from_vec(values)
            .into_pyarray(py)
            .into_any()
            .unbind()),
        _ => Err(PyValueError::new_err("invalid output shape metadata")),
    }
}

fn row_major_matrix_to_pyarray2<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    name: &'static str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = Array2::from_shape_vec((rows, cols), values).map_err(|e| {
        PyValueError::new_err(format!("failed to reshape {name} to ({rows}, {cols}): {e}"))
    })?;
    Ok(arr.into_pyarray(py))
}

fn norms_to_py<'py>(py: Python<'py>, norms: Vec<f64>, shape_mode: u8) -> PyResult<Py<PyAny>> {
    match shape_mode {
        SHAPE_VECTOR_1D => {
            if norms.len() != 1 {
                return Err(PyValueError::new_err("invalid scalar norms shape metadata"));
            }
            Ok(norms[0].into_pyobject(py)?.into_any().unbind())
        }
        SHAPE_MATRIX_2D | SHAPE_FLAT_1D => {
            Ok(Array1::from_vec(norms).into_pyarray(py).into_any().unbind())
        }
        _ => Err(PyValueError::new_err("invalid norms output shape metadata")),
    }
}

fn maybe_norms_to_py<'py>(
    py: Python<'py>,
    norms: Vec<f64>,
    shape_mode: u8,
    has_stream: bool,
) -> PyResult<Py<PyAny>> {
    if !has_stream {
        return Ok(Array1::from_vec(Vec::<f64>::new())
            .into_pyarray(py)
            .into_any()
            .unbind());
    }
    norms_to_py(py, norms, shape_mode)
}

// ============================================================
// Utility functions
// ============================================================

/// Pack {+1, -1} sign array into uint8 bitfield.
///
/// Accepts 1D `(d,)` or 2D `(batch, d)` input. Returns matching shape.
#[pyfunction]
fn pack_bits<'py>(py: Python<'py>, signs: PyReadonlyArrayDyn<i8>) -> PyResult<PyObject> {
    let shape = signs.shape();
    match shape.len() {
        1 => {
            let s = signs.as_slice().map_err(|_| {
                PyValueError::new_err("signs must be a contiguous NumPy array of int8")
            })?;
            let packed = utils::pack_bits(s);
            Ok(Array1::from_vec(packed).into_pyarray(py).into_any().unbind())
        }
        2 => {
            let (batch, d) = (shape[0], shape[1]);
            let s = signs.as_slice().map_err(|_| {
                PyValueError::new_err("signs must be a contiguous NumPy array of int8")
            })?;
            let packed_d = (d + 7) / 8;
            let mut result = Vec::with_capacity(batch * packed_d);
            for i in 0..batch {
                result.extend_from_slice(&utils::pack_bits(&s[i * d..(i + 1) * d]));
            }
            let arr = Array2::from_shape_vec((batch, packed_d), result)
                .map_err(|e| PyValueError::new_err(format!("shape error: {e}")))?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        _ => Err(PyValueError::new_err("signs must be 1D or 2D")),
    }
}

/// Unpack uint8 bitfield back to {+1, -1} signs.
///
/// Accepts 1D `(packed_d,)` or 2D `(batch, packed_d)` input. Returns matching shape.
#[pyfunction]
fn unpack_bits<'py>(py: Python<'py>, packed: PyReadonlyArrayDyn<u8>, d: usize) -> PyResult<PyObject> {
    let shape = packed.shape();
    match shape.len() {
        1 => {
            let p = packed.as_slice().map_err(|_| {
                PyValueError::new_err("packed must be a contiguous NumPy array of uint8")
            })?;
            let signs = utils::unpack_bits(p, d).map_err(to_py_value_error)?;
            Ok(Array1::from_vec(signs).into_pyarray(py).into_any().unbind())
        }
        2 => {
            let (batch, packed_d) = (shape[0], shape[1]);
            let p = packed.as_slice().map_err(|_| {
                PyValueError::new_err("packed must be a contiguous NumPy array of uint8")
            })?;
            let mut result = Vec::with_capacity(batch * d);
            for i in 0..batch {
                let row = utils::unpack_bits(&p[i * packed_d..(i + 1) * packed_d], d)
                    .map_err(to_py_value_error)?;
                result.extend_from_slice(&row);
            }
            let arr = Array2::from_shape_vec((batch, d), result)
                .map_err(|e| PyValueError::new_err(format!("shape error: {e}")))?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        _ => Err(PyValueError::new_err("packed must be 1D or 2D")),
    }
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
/// Find nearest centroid index for each value.
///
/// Accepts 1D or 2D values array. Output shape matches input shape.
/// Centroids must be a sorted 1D array.
#[pyfunction]
fn nearest_centroid_indices<'py>(
    py: Python<'py>,
    values: PyReadonlyArrayDyn<f64>,
    centroids: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let c = centroids.as_slice().map_err(|_| {
        PyValueError::new_err("centroids must be a contiguous 1D NumPy array of float64")
    })?;
    let shape = values.shape();
    let v = values.as_slice().map_err(|_| {
        PyValueError::new_err("values must be a contiguous NumPy array of float64")
    })?;

    // Compute indices on the flat data
    let indices = codebook::nearest_centroid_indices(v, c).map_err(to_py_value_error)?;

    // Reshape output to match input shape
    match shape.len() {
        1 => Ok(Array1::from_vec(indices).into_pyarray(py).into_any().unbind()),
        2 => {
            let (rows, cols) = (shape[0], shape[1]);
            let arr = Array2::from_shape_vec((rows, cols), indices)
                .map_err(|e| PyValueError::new_err(format!("shape error: {e}")))?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        _ => Err(PyValueError::new_err("values must be 1D or 2D")),
    }
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

#[pyclass(name = "PolarQuant")]
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

    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }

    #[getter]
    fn bit_width(&self) -> u32 {
        self.inner.bit_width
    }

    #[getter]
    fn n_centroids(&self) -> usize {
        self.inner.n_centroids
    }

    #[getter]
    fn norm_correction(&self) -> bool {
        self.inner.norm_correction
    }

    #[getter]
    fn centroids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.centroids.clone()).into_pyarray(py)
    }

    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.inner.d;
        let mut values = Vec::with_capacity(d * d);
        for i in 0..d {
            for j in 0..d {
                values.push(self.inner.rotation[(i, j)]);
            }
        }
        row_major_matrix_to_pyarray2(py, values, d, d, "rotation")
    }

    #[pyo3(signature = (x, batch_size=None))]
    fn quantize<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        batch_size: Option<usize>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let parsed = parse_f64_batch(x, self.inner.d, batch_size, "x")?;
        let result = self
            .inner
            .quantize_batch(&parsed.data, parsed.batch_size)
            .map_err(to_py_value_error)?;

        Ok((
            reshape_u8(
                py,
                result.indices,
                self.inner.d,
                result.batch_size,
                parsed.shape_mode,
            )?,
            norms_to_py(py, result.norms, parsed.shape_mode)?,
        ))
    }

    #[pyo3(signature = (x, batch_size=None))]
    fn quantize_and_residual<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        batch_size: Option<usize>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let parsed = parse_f64_batch(x, self.inner.d, batch_size, "x")?;
        let (result, residual) = self
            .inner
            .quantize_and_residual(&parsed.data, parsed.batch_size)
            .map_err(to_py_value_error)?;

        Ok((
            reshape_u8(
                py,
                result.indices,
                self.inner.d,
                result.batch_size,
                parsed.shape_mode,
            )?,
            norms_to_py(py, result.norms, parsed.shape_mode)?,
            reshape_f64(
                py,
                residual,
                self.inner.d,
                result.batch_size,
                parsed.shape_mode,
            )?,
        ))
    }

    #[pyo3(signature = (indices, norms, batch_size=None))]
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArrayDyn<u8>,
        norms: &Bound<'py, PyAny>,
        batch_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let parsed = parse_u8_batch(indices, self.inner.d, batch_size, "indices")?;
        let parsed_norms = parse_norms(norms, parsed.batch_size, "norms")?;
        let result = self
            .inner
            .dequantize_batch(&PolarQuantResult {
                indices: parsed.data,
                norms: parsed_norms,
                batch_size: parsed.batch_size,
            })
            .map_err(to_py_value_error)?;

        reshape_f64(
            py,
            result,
            self.inner.d,
            parsed.batch_size,
            parsed.shape_mode,
        )
    }

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

#[pyclass(name = "QJL")]
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

    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }

    #[getter]
    #[allow(non_snake_case)]
    fn S<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.inner.d;
        let values = self.inner.projection_matrix_row_major();
        row_major_matrix_to_pyarray2(py, values, d, d, "S")
    }

    #[pyo3(signature = (r, batch_size=None))]
    fn quantize<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArrayDyn<f64>,
        batch_size: Option<usize>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let parsed = parse_f64_batch(r, self.inner.d, batch_size, "r")?;
        let result = self
            .inner
            .quantize_batch(&parsed.data, parsed.batch_size)
            .map_err(to_py_value_error)?;

        Ok((
            reshape_i8(
                py,
                result.signs,
                self.inner.d,
                result.batch_size,
                parsed.shape_mode,
            )?,
            norms_to_py(py, result.norms, parsed.shape_mode)?,
        ))
    }

    #[pyo3(signature = (signs, norms, batch_size=None))]
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        signs: PyReadonlyArrayDyn<i8>,
        norms: &Bound<'py, PyAny>,
        batch_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let parsed = parse_i8_batch(signs, self.inner.d, batch_size, "signs")?;
        let parsed_norms = parse_norms(norms, parsed.batch_size, "norms")?;
        let result = self
            .inner
            .dequantize_batch(&QjlResult {
                signs: parsed.data,
                norms: parsed_norms,
                batch_size: parsed.batch_size,
            })
            .map_err(to_py_value_error)?;

        reshape_f64(
            py,
            result,
            self.inner.d,
            parsed.batch_size,
            parsed.shape_mode,
        )
    }

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

#[pyclass(name = "CompressedVector")]
#[derive(Clone)]
struct CompressedVector {
    inner: turboquant_core::turboquant::CompressedVectorUnpacked,
    shape_mode: u8,
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

    #[getter]
    fn mse_indices<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let indices = self.inner.mse.indices.clone();
        let d = indices.len() / self.inner.mse.batch_size.max(1);
        reshape_u8(py, indices, d, self.inner.mse.batch_size, self.shape_mode)
    }

    #[getter]
    fn vector_norms<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        norms_to_py(py, self.inner.mse.norms.clone(), self.shape_mode)
    }

    #[getter]
    fn qjl_signs<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let signs = self.inner.qjl.signs.clone();
        let d = signs.len() / self.inner.qjl.batch_size.max(1);
        reshape_i8(py, signs, d, self.inner.qjl.batch_size, self.shape_mode)
    }

    #[getter]
    fn residual_norms<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        norms_to_py(py, self.inner.qjl.norms.clone(), self.shape_mode)
    }
}

#[pyclass(name = "TurboQuant")]
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

    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }

    #[getter]
    fn bit_width(&self) -> u32 {
        self.inner.bit_width
    }

    #[getter]
    fn polar_quant(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let obj = Py::new(
            py,
            PolarQuant {
                inner: self.inner.polar_quant.clone(),
            },
        )?;
        Ok(obj.into_any())
    }

    #[getter]
    fn qjl(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let obj = Py::new(
            py,
            Qjl {
                inner: self.inner.qjl.clone(),
            },
        )?;
        Ok(obj.into_any())
    }

    #[pyo3(signature = (x, batch_size=None))]
    fn quantize(
        &self,
        x: PyReadonlyArrayDyn<f64>,
        batch_size: Option<usize>,
    ) -> PyResult<CompressedVector> {
        let parsed = parse_f64_batch(x, self.inner.d, batch_size, "x")?;
        Ok(CompressedVector {
            inner: self
                .inner
                .quantize_unpacked(&parsed.data, parsed.batch_size)
                .map_err(to_py_value_error)?,
            shape_mode: parsed.shape_mode,
        })
    }

    #[pyo3(signature = (x, batch_size=None))]
    fn quantize_unpacked(
        &self,
        x: PyReadonlyArrayDyn<f64>,
        batch_size: Option<usize>,
    ) -> PyResult<CompressedVector> {
        self.quantize(x, batch_size)
    }

    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        compressed: &CompressedVector,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .dequantize_unpacked(&compressed.inner)
            .map_err(to_py_value_error)?;
        reshape_f64(
            py,
            result,
            self.inner.d,
            compressed.inner.mse.batch_size,
            compressed.shape_mode,
        )
    }

    fn dequantize_unpacked<'py>(
        &self,
        py: Python<'py>,
        compressed: &CompressedVector,
    ) -> PyResult<Py<PyAny>> {
        self.dequantize(py, compressed)
    }

    fn compressed_size_bits(&self, n_vectors: usize) -> usize {
        self.inner.compressed_size_bits(n_vectors)
    }

    #[pyo3(signature = (original_bits_per_value=16))]
    fn compression_ratio(&self, original_bits_per_value: usize) -> f64 {
        self.inner.compression_ratio(original_bits_per_value)
    }
}

// ============================================================
// TurboQuantMSE wrapper
// ============================================================

#[pyclass(name = "TurboQuantMSE")]
struct TurboQuantMse {
    inner: RustTurboQuantMse,
}

#[pymethods]
impl TurboQuantMse {
    #[new]
    #[pyo3(signature = (d, bit_width, seed=42, norm_correction=true))]
    fn new(d: usize, bit_width: u32, seed: u64, norm_correction: bool) -> PyResult<Self> {
        Ok(Self {
            inner: RustTurboQuantMse::new(d, bit_width, seed, norm_correction)
                .map_err(to_py_value_error)?,
        })
    }

    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }

    #[getter]
    fn bit_width(&self) -> u32 {
        self.inner.bit_width
    }

    #[getter]
    fn polar_quant(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let obj = Py::new(
            py,
            PolarQuant {
                inner: self.inner.polar_quant.clone(),
            },
        )?;
        Ok(obj.into_any())
    }

    #[pyo3(signature = (x, batch_size=None))]
    fn quantize<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        batch_size: Option<usize>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let parsed = parse_f64_batch(x, self.inner.d, batch_size, "x")?;
        let result = self
            .inner
            .quantize(&parsed.data, parsed.batch_size)
            .map_err(to_py_value_error)?;

        Ok((
            reshape_u8(
                py,
                result.indices,
                self.inner.d,
                result.batch_size,
                parsed.shape_mode,
            )?,
            norms_to_py(py, result.norms, parsed.shape_mode)?,
        ))
    }

    #[pyo3(signature = (indices, norms, batch_size=None))]
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArrayDyn<u8>,
        norms: &Bound<'py, PyAny>,
        batch_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let parsed = parse_u8_batch(indices, self.inner.d, batch_size, "indices")?;
        let parsed_norms = parse_norms(norms, parsed.batch_size, "norms")?;
        let result = self
            .inner
            .dequantize(&PolarQuantResult {
                indices: parsed.data,
                norms: parsed_norms,
                batch_size: parsed.batch_size,
            })
            .map_err(to_py_value_error)?;

        reshape_f64(
            py,
            result,
            self.inner.d,
            parsed.batch_size,
            parsed.shape_mode,
        )
    }
}

// ============================================================
// Outlier wrapper
// ============================================================

#[pyclass(name = "OutlierCompressedVector")]
#[derive(Clone)]
struct OutlierCompressedVector {
    inner: RustOutlierCompressed,
    shape_mode: u8,
    d: usize,
    n_outlier: usize,
    n_normal: usize,
}

#[pymethods]
impl OutlierCompressedVector {
    #[getter]
    fn effective_bits(&self) -> f64 {
        self.inner.effective_bits
    }

    #[getter]
    fn outlier_indices<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let indices = if let Some(out) = &self.inner.outlier {
            out.unpack().map_err(to_py_value_error)?.indices
        } else {
            vec![]
        };
        reshape_u8(
            py,
            indices,
            self.n_outlier,
            self.inner.batch_size,
            self.shape_mode,
        )
    }

    #[getter]
    fn outlier_norms<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let (norms, has_stream) = if let Some(out) = &self.inner.outlier {
            (
                out.unpack()
                    .map_err(to_py_value_error)?
                    .norms
                    .into_iter()
                    .collect(),
                true,
            )
        } else {
            (Vec::<f64>::new(), false)
        };
        maybe_norms_to_py(py, norms, self.shape_mode, has_stream)
    }

    #[getter]
    fn normal_indices<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let indices = if let Some(out) = &self.inner.normal {
            out.unpack().map_err(to_py_value_error)?.indices
        } else {
            vec![]
        };
        reshape_u8(
            py,
            indices,
            self.n_normal,
            self.inner.batch_size,
            self.shape_mode,
        )
    }

    #[getter]
    fn normal_norms<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let (norms, has_stream) = if let Some(out) = &self.inner.normal {
            (
                out.unpack()
                    .map_err(to_py_value_error)?
                    .norms
                    .into_iter()
                    .collect(),
                true,
            )
        } else {
            (Vec::<f64>::new(), false)
        };
        maybe_norms_to_py(py, norms, self.shape_mode, has_stream)
    }

    #[getter]
    fn qjl_signs<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let signs = self.inner.qjl.unpack_signs().map_err(to_py_value_error)?;
        reshape_i8(py, signs, self.d, self.inner.batch_size, self.shape_mode)
    }

    #[getter]
    fn residual_norms<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let norms: Vec<f64> = self.inner.qjl.norms.iter().map(|&v| v as f64).collect();
        norms_to_py(py, norms, self.shape_mode)
    }
}

#[pyclass(name = "OutlierTurboQuant")]
struct OutlierTurboQuant {
    inner: RustOutlierTurboQuant,
}

#[pymethods]
impl OutlierTurboQuant {
    #[new]
    #[pyo3(signature = (d, target_bits, seed=42))]
    fn new(d: usize, target_bits: f64, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: RustOutlierTurboQuant::new(d, target_bits, seed).map_err(to_py_value_error)?,
        })
    }

    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }

    #[getter]
    fn target_bits(&self) -> f64 {
        self.inner.target_bits
    }

    #[getter]
    fn n_outlier(&self) -> usize {
        self.inner.n_outlier
    }

    #[getter]
    fn n_normal(&self) -> usize {
        self.inner.n_normal
    }

    #[getter]
    fn high_bits(&self) -> u32 {
        self.inner.high_bits
    }

    #[getter]
    fn low_bits(&self) -> u32 {
        self.inner.low_bits
    }

    #[getter]
    fn effective_bits(&self) -> f64 {
        self.inner.effective_bits
    }

    #[getter]
    fn outlier_idx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        Array1::from_vec(self.inner.outlier_indices().to_vec()).into_pyarray(py)
    }

    #[getter]
    fn normal_idx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        Array1::from_vec(self.inner.normal_indices().to_vec()).into_pyarray(py)
    }

    #[getter]
    fn pq_outlier(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.inner.outlier_quantizer() {
            Some(pq) => Ok(Py::new(py, PolarQuant { inner: pq.clone() })?.into_any()),
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn pq_normal(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.inner.normal_quantizer() {
            Some(pq) => Ok(Py::new(py, PolarQuant { inner: pq.clone() })?.into_any()),
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn qjl(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let obj = Py::new(
            py,
            Qjl {
                inner: self.inner.qjl_quantizer().clone(),
            },
        )?;
        Ok(obj.into_any())
    }

    #[pyo3(signature = (x, batch_size=None))]
    fn quantize(
        &self,
        x: PyReadonlyArrayDyn<f64>,
        batch_size: Option<usize>,
    ) -> PyResult<OutlierCompressedVector> {
        let parsed = parse_f64_batch(x, self.inner.d, batch_size, "x")?;
        Ok(OutlierCompressedVector {
            inner: self
                .inner
                .quantize(&parsed.data, parsed.batch_size)
                .map_err(to_py_value_error)?,
            shape_mode: parsed.shape_mode,
            d: self.inner.d,
            n_outlier: self.inner.n_outlier,
            n_normal: self.inner.n_normal,
        })
    }

    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        compressed: &OutlierCompressedVector,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .dequantize(&compressed.inner)
            .map_err(to_py_value_error)?;
        reshape_f64(
            py,
            result,
            self.inner.d,
            compressed.inner.batch_size,
            compressed.shape_mode,
        )
    }

    #[pyo3(signature = (original_bits=16))]
    fn compression_ratio(&self, original_bits: usize) -> f64 {
        self.inner.compression_ratio(original_bits)
    }
}

// ============================================================
// KV Cache Compressor wrapper
// ============================================================

#[pyclass(name = "CompressedKVCache")]
struct CompressedKVCache {
    inner: RustCompressedKvCache,
}

#[pymethods]
impl CompressedKVCache {
    #[getter]
    fn num_layers(&self) -> usize {
        self.inner.num_layers
    }

    #[getter]
    fn num_heads(&self) -> usize {
        self.inner.num_heads
    }

    #[getter]
    fn seq_len(&self) -> usize {
        self.inner.seq_len
    }

    #[getter]
    fn head_dim(&self) -> usize {
        self.inner.head_dim
    }

    #[getter]
    fn k_bit_width(&self) -> u32 {
        self.inner.k_bit_width
    }

    #[getter]
    fn v_bit_width(&self) -> u32 {
        self.inner.v_bit_width
    }

    #[getter]
    fn k_compressed<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let layers = PyList::empty(py);
        for layer in &self.inner.k_compressed {
            let heads = PyList::empty(py);
            for head in layer {
                let unpacked = turboquant_core::turboquant::CompressedVectorUnpacked {
                    mse: head.mse.unpack().map_err(to_py_value_error)?,
                    qjl: head.qjl.unpack().map_err(to_py_value_error)?,
                    bit_width: head.bit_width,
                };
                let py_obj = Py::new(
                    py,
                    CompressedVector {
                        inner: unpacked,
                        shape_mode: SHAPE_MATRIX_2D,
                    },
                )?;
                heads.append(py_obj)?;
            }
            layers.append(heads)?;
        }
        Ok(layers.into_any().unbind())
    }

    #[getter]
    fn v_indices<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let layers = PyList::empty(py);
        for layer in &self.inner.v_compressed {
            let heads = PyList::empty(py);
            for head in layer {
                let unpacked = head.unpack().map_err(to_py_value_error)?;
                let arr = Array2::from_shape_vec((unpacked.batch_size, head.d), unpacked.indices)
                    .map_err(|e| {
                    PyValueError::new_err(format!("failed to reshape v_indices: {e}"))
                })?;
                heads.append(arr.into_pyarray(py))?;
            }
            layers.append(heads)?;
        }
        Ok(layers.into_any().unbind())
    }

    #[getter]
    fn v_norms<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let layers = PyList::empty(py);
        for layer in &self.inner.v_compressed {
            let heads = PyList::empty(py);
            for head in layer {
                let unpacked = head.unpack().map_err(to_py_value_error)?;
                heads.append(Array1::from_vec(unpacked.norms).into_pyarray(py))?;
            }
            layers.append(heads)?;
        }
        Ok(layers.into_any().unbind())
    }
}

#[pyclass(name = "KVCacheCompressor")]
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

    #[getter]
    fn head_dim(&self) -> usize {
        self.inner.head_dim
    }

    #[getter]
    fn k_bits(&self) -> u32 {
        self.inner.k_bits
    }

    #[getter]
    fn v_bits(&self) -> u32 {
        self.inner.v_bits
    }

    #[getter]
    fn k_quantizer(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let obj = Py::new(
            py,
            TurboQuant {
                inner: self.inner.k_quantizer().clone(),
            },
        )?;
        Ok(obj.into_any())
    }

    #[getter]
    fn v_quantizer(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let obj = Py::new(
            py,
            TurboQuantMse {
                inner: self.inner.v_quantizer().clone(),
            },
        )?;
        Ok(obj.into_any())
    }

    fn compress(
        &self,
        k_cache: PyReadonlyArrayDyn<f64>,
        v_cache: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<CompressedKVCache> {
        if k_cache.ndim() != 4 {
            return Err(PyValueError::new_err(
                "k_cache must be a contiguous 4D NumPy array of float64",
            ));
        }
        if v_cache.ndim() != 4 {
            return Err(PyValueError::new_err(
                "v_cache must be a contiguous 4D NumPy array of float64",
            ));
        }

        let ks = k_cache.shape();
        let vs = v_cache.shape();
        if ks != vs {
            return Err(PyValueError::new_err(format!(
                "shape mismatch: k_cache={ks:?}, v_cache={vs:?}"
            )));
        }
        if ks[3] != self.inner.head_dim {
            return Err(PyValueError::new_err(format!(
                "head_dim mismatch: compressor head_dim={}, input head_dim={}",
                self.inner.head_dim, ks[3]
            )));
        }

        let k = k_cache
            .as_slice()
            .map_err(|_| PyValueError::new_err("k_cache must be contiguous in row-major order"))?;
        let v = v_cache
            .as_slice()
            .map_err(|_| PyValueError::new_err("v_cache must be contiguous in row-major order"))?;

        Ok(CompressedKVCache {
            inner: self
                .inner
                .compress(k, v, ks[0], ks[1], ks[2])
                .map_err(to_py_value_error)?,
        })
    }

    fn decompress<'py>(
        &self,
        py: Python<'py>,
        compressed: &CompressedKVCache,
    ) -> PyResult<(Bound<'py, PyArray4<f64>>, Bound<'py, PyArray4<f64>>)> {
        let (k, v) = self
            .inner
            .decompress(&compressed.inner)
            .map_err(to_py_value_error)?;

        let shape = (
            compressed.inner.num_layers,
            compressed.inner.num_heads,
            compressed.inner.seq_len,
            compressed.inner.head_dim,
        );

        let k_arr = Array4::from_shape_vec(shape, k)
            .map_err(|e| PyValueError::new_err(format!("failed to reshape k_cache output: {e}")))?;
        let v_arr = Array4::from_shape_vec(shape, v)
            .map_err(|e| PyValueError::new_err(format!("failed to reshape v_cache output: {e}")))?;

        Ok((k_arr.into_pyarray(py), v_arr.into_pyarray(py)))
    }

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
    m.add_class::<TurboQuantMse>()?;
    m.add_class::<OutlierTurboQuant>()?;
    m.add_class::<OutlierCompressedVector>()?;
    m.add_class::<KvCacheCompressor>()?;
    m.add_class::<CompressedKVCache>()?;

    // Backward-compatible aliases
    m.add("Qjl", m.getattr("QJL")?)?;
    m.add("KvCacheCompressor", m.getattr("KVCacheCompressor")?)?;
    m.add("TurboQuantMse", m.getattr("TurboQuantMSE")?)?;

    Ok(())
}

fn to_py_value_error(err: TurboQuantError) -> PyErr {
    PyValueError::new_err(err.to_string())
}
