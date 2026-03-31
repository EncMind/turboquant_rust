//! TurboQuant: KV cache compression via PolarQuant + QJL.
//!
//! A Rust implementation of the TurboQuant algorithm (Google Research, ICLR 2026)
//! for compressing transformer KV caches by 3.8–6.4x with minimal quality loss.
//!
//! # Architecture
//!
//! The compression pipeline has two stages:
//!
//! 1. **PolarQuant** (b-1 bits): Random rotation + optimal scalar quantization.
//!    After Haar-distributed rotation, coordinates follow a known distribution,
//!    enabling optimal per-coordinate quantization.
//!
//! 2. **QJL** (1 bit): Quantized Johnson-Lindenstrauss on the residual.
//!    Compresses the PolarQuant residual to 1-bit sign representations,
//!    eliminating quantization bias for inner product preservation.
//!
//! # Modules
//!
//! - [`utils`] — Bit packing/unpacking and memory footprint calculation.
//! - [`rotation`] — Random rotation matrices (dense Haar + fast Walsh-Hadamard).
//! - [`codebook`] — Optimal centroid computation via Lloyd's algorithm.
//! - [`polar_quant`] — PolarQuant quantizer (Algorithm 1).
//! - [`qjl`] — QJL 1-bit quantizer.
//! - [`turboquant`] — Full TurboQuant algorithm (Algorithm 2).
//! - [`outlier`] — Non-integer bit precision via outlier channel splitting.
//! - [`kv_cache`] — Transformer KV cache compression integration.

pub mod codebook;
pub mod error;
pub mod kv_cache;
pub mod outlier;
pub mod polar_quant;
pub mod qjl;
pub mod rotation;
pub mod turboquant;
pub mod utils;

// Re-export main types at crate root.
pub use codebook::{nearest_centroid_indices, optimal_centroids};
pub use error::{Result, TurboQuantError};
pub use kv_cache::{CompressedKvCache, KvCacheCompressor, MemoryStats};
pub use outlier::OutlierTurboQuant;
pub use polar_quant::{PackedPolarQuantResult, PolarQuant};
pub use qjl::{PackedQjlResult, Qjl};
pub use rotation::{fast_walsh_hadamard_transform, random_rotation_dense, FastRotation};
pub use turboquant::{CompressedVector, CompressedVectorUnpacked, TurboQuant, TurboQuantMse};
pub use utils::{memory_footprint_bytes, pack_bits, pack_indices, unpack_bits, unpack_indices};
