# TurboQuant — Rust Implementation

Rust implementation of the TurboQuant algorithm (Google Research, ICLR 2026) for
KV cache compression in large language models.

## Crate Structure

```
rust/
├── turboquant-core/    Pure Rust core library (all algorithms, 153 tests)
├── turboquant-py/      PyO3 Python bindings (drop-in accelerator)
└── turboquant-cli/     CLI tool for demos and benchmarks
```

### turboquant-core

Complete implementation of all TurboQuant algorithms:

| Module | Description |
|--------|-------------|
| `utils` | Bit packing/unpacking, memory footprint |
| `rotation` | Dense Haar rotation (QR), Fast Walsh-Hadamard Transform |
| `codebook` | Optimal centroids via Lloyd's algorithm on Gaussian |
| `polar_quant` | PolarQuant: rotation + scalar quantization (Algorithm 1) |
| `qjl` | QJL: 1-bit sign quantization via random projection |
| `turboquant` | Full TurboQuant: PolarQuant + QJL (Algorithm 2) |
| `outlier` | Non-integer bit precision (2.5-bit, 3.5-bit) |
| `kv_cache` | Transformer KV cache compression integration |

### turboquant-py

Python bindings via [PyO3](https://pyo3.rs/) + [maturin](https://maturin.rs/).
Drop-in replacement for performance-critical functions in the Python prototype.

### turboquant-cli

Standalone CLI for compression demos and benchmarks.

---

## Benchmark Results

All three backends use seeded random-normal vectors (seed=42), d=128, 1000 vectors,
and the unpacked (no bit-packing) roundtrip to ensure an apples-to-apples comparison.
Rust built with `--features accelerate` (Apple Accelerate CBLAS).

```bash
# 3-way comparison
python benchmarks/bench_comparison.py --dim 128 --count 1000 --iter 5000 --seed 42

# Pure Rust standalone
cargo run --release -p turboquant-cli --features accelerate -- bench-json --dim 128 --count 1000 --iter 5000 --seed 42
cargo run --release -p turboquant-cli --features accelerate -- bench --dim 128 --count 10000
```

### Low-level Primitives

| Operation | Pure Python | Python+Rust (PyO3) | Pure Rust (CLI) | Speedup |
|---|---:|---:|---:|---:|
| Fast Walsh-Hadamard (d=128) | 282 us | 2.3 us | **1.2 us** | **235x** |
| Nearest centroid (3-bit) | 6.7 us | 2.0 us | **1.0 us** | **6.7x** |
| Pack bits (d=128) | 5.5 us | 1.0 us | **0.2 us** | **28x** |

### Single-Vector Roundtrip (quantize + dequantize, d=128)

| Format | Pure Python | Python+Rust (PyO3) | Pure Rust (CLI) | Speedup |
|---|---:|---:|---:|---:|
| turbo2 (2-bit) | 215 us | 92 us | **77 us** | **2.8x** |
| turbo3 (3-bit) | 218 us | 79 us | **80 us** | **2.8x** |
| turbo4 (4-bit) | 197 us | 78 us | **79 us** | **2.5x** |

Python+Rust and Pure Rust single-vector latencies are within ~15% of each other,
confirming they run the same compiled Rust code.

### Batch Throughput (1000 vectors, d=128)

| Format | Pure Python (vec/s) | Python+Rust (vec/s) | Pure Rust (vec/s) | Speedup |
|---|---:|---:|---:|---:|
| turbo2 (2-bit) | 139,000 | 243,000 | **204,000** | **1.7x** |
| turbo3 (3-bit) | 123,000 | 227,000 | **224,000** | **1.8x** |
| turbo4 (4-bit) | 125,000 | 209,000 | **190,000** | **1.7x** |

### CLI End-to-End (Pure Rust, 10,000 vectors, packed wire format)

Latency (median of 9 samples per stage):

| Format | Compress | Decompress | Roundtrip | Ratio |
|---|---:|---:|---:|---:|
| turbo2 | 62,594 us | 42,088 us | 101,517 us | 7.11x |
| turbo3 | 69,707 us | 48,726 us | 123,224 us | 4.92x |
| turbo4 | 75,744 us | 52,473 us | 130,113 us | 3.76x |

Throughput (median of 9 roundtrips):

| Format | Throughput (vec/s) |
|---|---:|
| turbo2 | **105,293** |
| turbo3 | **86,451** |
| turbo4 | **85,669** |

Note: `bench` uses packed `quantize`/`dequantize` (includes wire-format
packing overhead), so throughput is lower than the unpacked 3-way comparison
above.

---

## Key Design Decisions

- **Batch BLAS (Level 3)**: All batch operations use matrix-matrix multiply
  (`DGEMM`) instead of per-vector matrix-vector multiply. On macOS, this calls
  Apple Accelerate directly via CBLAS. Without the `accelerate` feature, falls
  back to nalgebra's `matrixmultiply` crate.

- **Rayon parallelism**: KV cache compression parallelizes across all
  (layer, head) pairs via `par_iter()`. Scales linearly with core count.

- **Feature flags**:
  - `accelerate` (optional, macOS): Links Apple Accelerate CBLAS for faster matrix multiply
  - `parallel`: Enables rayon-based batch rotation (experimental)

---

## Quick Start

### Build & Test (Rust only — no Python needed)

```bash
cd rust
cargo test                  # Run all 153 Rust tests
cargo build --release       # Build optimized binaries
```

### Build & Test (Python bindings)

Requires: Python 3.9+, [maturin](https://maturin.rs/) (`pip install maturin`), numpy

```bash
cd rust/turboquant-py

# Build and install the Rust extension into your Python environment
pip install maturin
maturin develop --release --features accelerate   # macOS (Apple Accelerate BLAS)
maturin develop --release                          # Linux/other (pure Rust BLAS)

# Run PyO3 binding tests (40 tests covering 1D/2D/batch roundtrips)
pip install pytest
pytest tests/ -v
```

### Run Benchmarks

```bash
# Criterion micro-benchmarks (Rust)
cargo bench -p turboquant-core --features accelerate

# CLI benchmarks
cargo run --release -p turboquant-cli --features accelerate -- bench --count 10000
cargo run --release -p turboquant-cli --features accelerate -- demo --count 1000
cargo run --release -p turboquant-cli -- info

# Machine-readable benchmark output (for regression tracking)
cargo run --release -p turboquant-cli --features accelerate -- bench-json --dim 128 --count 1000 --iter 5000 --seed 42
```

### Python Usage

```python
import turboquant_rs

# Drop-in accelerated primitives
result = turboquant_rs.fast_walsh_hadamard_transform(x)   # ~235x faster
indices = turboquant_rs.nearest_centroid_indices(values, centroids)
packed = turboquant_rs.pack_bits(signs)

# Full quantizer objects — accepts 1D (single vector) or 2D (batch, d)
tq = turboquant_rs.TurboQuant(d=128, bit_width=3, seed=42)
compressed = tq.quantize(x)           # x: (128,) or (batch, 128)
reconstructed = tq.dequantize(compressed)

# KV cache compression
compressor = turboquant_rs.KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)
compressed = compressor.compress(k_cache, v_cache)  # 4D numpy arrays
k_hat, v_hat = compressor.decompress(compressed)
```

### Using as a Rust Library

```toml
[dependencies]
turboquant-core = { path = "path/to/rust/turboquant-core" }
```

```rust
use turboquant_core::{TurboQuant, KvCacheCompressor};

// Compress a single vector
let tq = TurboQuant::new(128, 3, 42, true).unwrap();
let x: Vec<f64> = vec![0.1; 128];
let compressed = tq.quantize(&x, 1).unwrap();
let x_hat = tq.dequantize(&compressed).unwrap();

// Compress KV cache
let compressor = KvCacheCompressor::new(128, 3, 3, 42, true).unwrap();
let stats = compressor.memory_stats(32768, 32, 32);
println!("Wire compression: {:.2}x", stats.wire_compression_ratio);
println!("In-memory compression: {:.2}x", stats.in_memory_compression_ratio);
```

---

## Test Coverage

153 tests across 8 integration test files + inline unit tests:

| Test File | Count | What It Tests |
|---|---:|---|
| Unit tests (inline) | 37 | Per-module correctness |
| test_utils.rs | 13 | Bit packing roundtrips, memory footprint |
| test_rotation.rs | 19 | Orthogonality, norm/IP preservation, distribution, FWHT |
| test_codebook.rs | 12 | Paper centroids, Lloyd's convergence, brute-force matching |
| test_polar_quant.rs | 12 | MSE bounds (Table 2), non-unit norms, norm correction |
| test_qjl.rs | 6 | Unbiasedness (Theorem 2), binary signs, norm scale |
| test_turboquant.rs | 26 | MSE/IP bounds at 2/3/4-bit x d=64/128/256, turbo4 edge cases |
| test_outlier.rs | 8 | 2.5/3.5-bit effective rates, batch consistency |
| test_kv_cache.rs | 5 | Attention score preservation, metadata, memory stats |
| test_distortion.rs | 14 | Paper Table 2 bounds (1000 samples), IP monotonicity |
| Doc-tests | 1 | Usage example |

---

## Release Notes

See [`CHANGELOG.md`](CHANGELOG.md) for unreleased and versioned changes.

---

## Compression Formats

| Format | Bits/val | Compression | Quality (PPL vs q8_0) |
|--------|----------|-------------|----------------------|
| turbo4 | 4 | 3.8x | +0.23% |
| turbo3 | 3 | 4.6-5.1x | +1.06% |
| turbo2 | 2 | 6.4x | +6.48% |
