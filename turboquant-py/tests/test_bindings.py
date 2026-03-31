"""Tests for turboquant_rs PyO3 bindings — 1D and 2D roundtrip coverage.

Run with:
    cd rust/turboquant-py
    maturin develop --release --features accelerate
    pytest tests/ -v
"""

import numpy as np
import pytest

import turboquant_rs as rs


# ============================================================
# Utility functions: pack_bits / unpack_bits
# ============================================================

class TestPackBits:
    def test_1d_roundtrip(self):
        signs = np.array([1, -1, 1, -1, 1, 1, -1, 1], dtype=np.int8)
        packed = rs.pack_bits(signs)
        assert packed.ndim == 1
        unpacked = rs.unpack_bits(packed, 8)
        np.testing.assert_array_equal(signs, unpacked)

    def test_1d_non_multiple_of_8(self):
        signs = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        packed = rs.pack_bits(signs)
        unpacked = rs.unpack_bits(packed, 5)
        np.testing.assert_array_equal(signs, unpacked)

    def test_2d_roundtrip(self):
        rng = np.random.default_rng(42)
        signs = rng.choice([-1, 1], size=(5, 64)).astype(np.int8)
        packed = rs.pack_bits(signs)
        assert packed.shape == (5, 8)
        unpacked = rs.unpack_bits(packed, 64)
        assert unpacked.shape == (5, 64)
        np.testing.assert_array_equal(signs, unpacked)

    def test_2d_non_multiple_of_8(self):
        signs = np.array([[1, -1, 1], [-1, 1, -1]], dtype=np.int8)
        packed = rs.pack_bits(signs)
        assert packed.shape[0] == 2
        unpacked = rs.unpack_bits(packed, 3)
        assert unpacked.shape == (2, 3)
        np.testing.assert_array_equal(signs, unpacked)


# ============================================================
# Utility functions: pack_indices
# ============================================================

class TestPackIndices:
    def test_1d_roundtrip_2bit(self):
        indices = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
        packed = rs.pack_indices(indices, 2)
        assert len(packed) < len(indices)

    def test_1d_8bit_passthrough(self):
        indices = np.array([0, 127, 255], dtype=np.uint8)
        packed = rs.pack_indices(indices, 8)
        np.testing.assert_array_equal(packed, indices)


# ============================================================
# Codebook: nearest_centroid_indices
# ============================================================

class TestNearestCentroidIndices:
    def test_1d(self):
        centroids = np.array([-1.0, 0.0, 1.0])
        values = np.array([-0.8, 0.1, 0.6, -0.3])
        indices = rs.nearest_centroid_indices(values, centroids)
        assert indices.ndim == 1
        assert indices.shape == (4,)
        assert indices[0] == 0  # -0.8 → -1.0
        assert indices[2] == 2  # 0.6 → 1.0

    def test_2d_shape_preserved(self):
        centroids = np.array([-1.0, 0.0, 1.0])
        values = np.random.default_rng(42).standard_normal((5, 10))
        indices = rs.nearest_centroid_indices(values, centroids)
        assert indices.shape == (5, 10)

    def test_2d_matches_1d(self):
        centroids = np.array([-1.5, -0.5, 0.5, 1.5])
        rng = np.random.default_rng(42)
        values_2d = rng.standard_normal((3, 20))
        indices_2d = rs.nearest_centroid_indices(values_2d, centroids)
        for i in range(3):
            indices_1d = rs.nearest_centroid_indices(values_2d[i], centroids)
            np.testing.assert_array_equal(indices_2d[i], indices_1d)


# ============================================================
# PolarQuant: 1D, 2D, flat+batch_size roundtrips
# ============================================================

class TestPolarQuant:
    def test_1d_single_vector_roundtrip(self):
        pq = rs.PolarQuant(d=16, bit_width=2, seed=42)
        x = np.random.default_rng(1).standard_normal(16)
        indices, norm = pq.quantize(x)
        assert indices.shape == (16,)
        assert isinstance(norm, float)
        x_hat = pq.dequantize(indices, norm)
        assert x_hat.shape == (16,)

    def test_2d_batch_roundtrip(self):
        pq = rs.PolarQuant(d=16, bit_width=2, seed=42)
        X = np.random.default_rng(1).standard_normal((5, 16))
        indices, norms = pq.quantize(X)
        assert indices.shape == (5, 16)
        assert norms.shape == (5,)
        X_hat = pq.dequantize(indices, norms)
        assert X_hat.shape == (5, 16)

    def test_flat_batch_size_roundtrip(self):
        pq = rs.PolarQuant(d=8, bit_width=2, seed=42)
        X = np.random.default_rng(1).standard_normal(24)  # 3 * 8
        indices, norms = pq.quantize(X, batch_size=3)
        assert indices.shape == (24,)
        assert norms.shape == (3,)
        X_hat = pq.dequantize(indices, norms, batch_size=3)
        assert X_hat.shape == (24,)

    def test_2d_batch_matches_single(self):
        pq = rs.PolarQuant(d=16, bit_width=2, seed=42)
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, 16))
        batch_idx, batch_norms = pq.quantize(X)
        batch_hat = pq.dequantize(batch_idx, batch_norms)
        for i in range(5):
            single_idx, single_norm = pq.quantize(X[i])
            single_hat = pq.dequantize(single_idx, single_norm)
            np.testing.assert_array_equal(batch_idx[i], single_idx)
            np.testing.assert_allclose(batch_hat[i], single_hat, atol=1e-12)

    def test_batch_size_mismatch_rejected(self):
        pq = rs.PolarQuant(d=8, bit_width=2, seed=42)
        X = np.random.default_rng(1).standard_normal((3, 8))
        indices, norms = pq.quantize(X)
        with pytest.raises(ValueError, match="batch_size mismatch"):
            pq.dequantize(indices, norms, batch_size=999)

    def test_quantize_and_residual(self):
        pq = rs.PolarQuant(d=16, bit_width=2, seed=42)
        x = np.random.default_rng(1).standard_normal(16)
        indices, norms, residual = pq.quantize_and_residual(x)
        x_hat = pq.dequantize(indices, norms)
        np.testing.assert_allclose(residual, x - x_hat, atol=1e-12)


# ============================================================
# QJL: 1D, 2D, flat+batch_size roundtrips
# ============================================================

class TestQJL:
    def test_1d_single_vector_roundtrip(self):
        qjl = rs.QJL(d=16, seed=42)
        r = np.random.default_rng(1).standard_normal(16)
        signs, norm = qjl.quantize(r)
        assert signs.shape == (16,)
        assert isinstance(norm, float)
        assert set(signs.tolist()).issubset({1, -1})
        r_hat = qjl.dequantize(signs, norm)
        assert r_hat.shape == (16,)

    def test_2d_batch_roundtrip(self):
        qjl = rs.QJL(d=16, seed=42)
        R = np.random.default_rng(1).standard_normal((5, 16))
        signs, norms = qjl.quantize(R)
        assert signs.shape == (5, 16)
        assert norms.shape == (5,)
        R_hat = qjl.dequantize(signs, norms)
        assert R_hat.shape == (5, 16)

    def test_flat_batch_size_roundtrip(self):
        qjl = rs.QJL(d=8, seed=42)
        R = np.random.default_rng(1).standard_normal(24)  # 3 * 8
        signs, norms = qjl.quantize(R, batch_size=3)
        assert signs.shape == (24,)
        assert norms.shape == (3,)
        R_hat = qjl.dequantize(signs, norms, batch_size=3)
        assert R_hat.shape == (24,)

    def test_batch_size_mismatch_rejected(self):
        qjl = rs.QJL(d=8, seed=42)
        R = np.random.default_rng(1).standard_normal((3, 8))
        signs, norms = qjl.quantize(R)
        with pytest.raises(ValueError, match="batch_size mismatch"):
            qjl.dequantize(signs, norms, batch_size=999)

    def test_zero_vector(self):
        qjl = rs.QJL(d=16, seed=42)
        r = np.zeros(16)
        signs, norm = qjl.quantize(r)
        assert norm < 1e-15
        r_hat = qjl.dequantize(signs, norm)
        for v in r_hat:
            assert abs(v) < 1e-15


# ============================================================
# TurboQuant: 1D, 2D roundtrips
# ============================================================

class TestTurboQuant:
    def test_1d_single_vector_roundtrip(self):
        tq = rs.TurboQuant(d=16, bit_width=3, seed=42)
        x = np.random.default_rng(1).standard_normal(16)
        compressed = tq.quantize(x)
        assert compressed.bit_width == 3
        x_hat = tq.dequantize(compressed)
        assert x_hat.shape == (16,)

    def test_2d_batch_roundtrip(self):
        tq = rs.TurboQuant(d=16, bit_width=3, seed=42)
        X = np.random.default_rng(1).standard_normal((5, 16))
        compressed = tq.quantize(X)
        assert compressed.batch_size == 5
        X_hat = tq.dequantize(compressed)
        assert X_hat.shape == (5, 16)

    def test_2d_batch_matches_single(self):
        tq = rs.TurboQuant(d=16, bit_width=3, seed=42)
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, 16))
        batch_hat = tq.dequantize(tq.quantize(X))
        for i in range(5):
            single_hat = tq.dequantize(tq.quantize(X[i]))
            np.testing.assert_allclose(batch_hat[i], single_hat, atol=1e-10)

    def test_compressed_vector_fields(self):
        tq = rs.TurboQuant(d=16, bit_width=3, seed=42)
        x = np.random.default_rng(1).standard_normal(16)
        c = tq.quantize(x)
        assert c.bit_width == 3
        assert c.mse_indices.shape == (16,)
        assert c.qjl_signs.shape == (16,)
        assert isinstance(c.vector_norms, float)
        assert isinstance(c.residual_norms, float)

    def test_attributes(self):
        tq = rs.TurboQuant(d=128, bit_width=4, seed=42)
        assert tq.d == 128
        assert tq.bit_width == 4
        assert tq.polar_quant.d == 128
        assert tq.qjl.d == 128


# ============================================================
# TurboQuantMSE: 1D, 2D, flat+batch_size roundtrips
# ============================================================

class TestTurboQuantMSE:
    def test_1d_single_vector_roundtrip(self):
        tqm = rs.TurboQuantMSE(d=16, bit_width=3, seed=42)
        x = np.random.default_rng(1).standard_normal(16)
        indices, norms = tqm.quantize(x)
        assert indices.shape == (16,)
        x_hat = tqm.dequantize(indices, norms)
        assert x_hat.shape == (16,)

    def test_2d_batch_roundtrip(self):
        tqm = rs.TurboQuantMSE(d=16, bit_width=3, seed=42)
        X = np.random.default_rng(1).standard_normal((5, 16))
        indices, norms = tqm.quantize(X)
        assert indices.shape == (5, 16)
        X_hat = tqm.dequantize(indices, norms)
        assert X_hat.shape == (5, 16)

    def test_flat_batch_size_roundtrip(self):
        tqm = rs.TurboQuantMSE(d=8, bit_width=3, seed=42)
        X = np.random.default_rng(1).standard_normal(24)
        indices, norms = tqm.quantize(X, batch_size=3)
        assert indices.shape == (24,)
        X_hat = tqm.dequantize(indices, norms, batch_size=3)
        assert X_hat.shape == (24,)

    def test_batch_size_mismatch_rejected(self):
        tqm = rs.TurboQuantMSE(d=8, bit_width=3, seed=42)
        X = np.random.default_rng(1).standard_normal((3, 8))
        indices, norms = tqm.quantize(X)
        with pytest.raises(ValueError, match="batch_size mismatch"):
            tqm.dequantize(indices, norms, batch_size=999)


# ============================================================
# OutlierTurboQuant: 1D, 2D roundtrips
# ============================================================

class TestOutlierTurboQuant:
    def test_1d_roundtrip(self):
        oq = rs.OutlierTurboQuant(d=32, target_bits=2.5, seed=42)
        x = np.random.default_rng(1).standard_normal(32)
        compressed = oq.quantize(x)
        x_hat = oq.dequantize(compressed)
        assert x_hat.shape == (32,)

    def test_2d_roundtrip(self):
        oq = rs.OutlierTurboQuant(d=32, target_bits=3.5, seed=42)
        X = np.random.default_rng(1).standard_normal((5, 32))
        compressed = oq.quantize(X)
        X_hat = oq.dequantize(compressed)
        assert X_hat.shape == (5, 32)

    def test_effective_bits(self):
        oq = rs.OutlierTurboQuant(d=128, target_bits=2.5, seed=42)
        assert abs(oq.effective_bits - 2.5) < 0.01

    def test_attributes(self):
        oq = rs.OutlierTurboQuant(d=128, target_bits=3.5, seed=42)
        assert oq.d == 128
        assert oq.n_outlier + oq.n_normal == 128
        assert len(oq.outlier_idx) == oq.n_outlier
        assert len(oq.normal_idx) == oq.n_normal


# ============================================================
# KVCacheCompressor: compress/decompress roundtrip
# ============================================================

class TestKVCacheCompressor:
    def test_compress_decompress_roundtrip(self):
        compressor = rs.KVCacheCompressor(head_dim=16, k_bits=3, v_bits=3, seed=42)
        rng = np.random.default_rng(42)
        k = rng.standard_normal((2, 4, 8, 16))
        v = rng.standard_normal((2, 4, 8, 16))
        compressed = compressor.compress(k, v)
        assert compressed.num_layers == 2
        assert compressed.num_heads == 4
        assert compressed.seq_len == 8
        assert compressed.head_dim == 16
        k_hat, v_hat = compressor.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_memory_stats(self):
        compressor = rs.KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3, seed=42)
        stats = compressor.memory_stats(1024, 32, 32)
        assert stats["compression_ratio"] > 2.0

    def test_attributes(self):
        compressor = rs.KVCacheCompressor(head_dim=64, k_bits=4, v_bits=3, seed=42)
        assert compressor.head_dim == 64
        assert compressor.k_bits == 4
        assert compressor.v_bits == 3


# ============================================================
# FWHT
# ============================================================

class TestFWHT:
    def test_involutory(self):
        x = np.random.default_rng(42).standard_normal(32)
        y = rs.fast_walsh_hadamard_transform(x)
        x_back = rs.fast_walsh_hadamard_transform(y)
        np.testing.assert_allclose(x_back, x, atol=1e-10)

    def test_preserves_norm(self):
        x = np.random.default_rng(42).standard_normal(64)
        y = rs.fast_walsh_hadamard_transform(x)
        np.testing.assert_allclose(np.linalg.norm(y), np.linalg.norm(x), rtol=1e-10)


# ============================================================
# Codebook
# ============================================================

class TestCodebook:
    def test_optimal_centroids_1bit(self):
        c = rs.optimal_centroids(1, 128)
        assert len(c) == 2
        assert abs(c[0] + c[1]) < 1e-10  # symmetric

    def test_optimal_centroids_3bit(self):
        c = rs.optimal_centroids(3, 128)
        assert len(c) == 8
        for w in zip(c[:-1], c[1:]):
            assert w[0] < w[1]  # sorted
