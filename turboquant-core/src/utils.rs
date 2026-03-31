//! Bit packing/unpacking utilities and memory footprint calculation.
//!
//! Mirrors `turboquant/utils.py` — packs {+1, -1} sign arrays into compact
//! bitfields and packs sub-byte index arrays into packed byte buffers.

/// Pack `{+1, -1}` sign values into a compact bitfield.
///
/// 8 signs per byte. `+1` maps to bit 1, `-1` maps to bit 0.
/// MSB-first within each byte (matches NumPy `packbits` convention).
///
/// # Arguments
/// * `signs` — slice of `i8` values, each `+1` or `-1`.
///
/// # Returns
/// `Vec<u8>` of length `ceil(signs.len() / 8)`.
pub fn pack_bits(signs: &[i8]) -> Vec<u8> {
    let n_bytes = signs.len().div_ceil(8);
    let mut packed = vec![0u8; n_bytes];
    for (i, &s) in signs.iter().enumerate() {
        if s > 0 {
            packed[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    packed
}

/// Unpack a bitfield back to `{+1, -1}` sign values.
///
/// # Arguments
/// * `packed` — byte slice from [`pack_bits`].
/// * `d` — original number of signs (to trim padding).
pub fn unpack_bits(packed: &[u8], d: usize) -> Result<Vec<i8>> {
    let required = d.div_ceil(8);
    if packed.len() < required {
        return Err(TurboQuantError::BufferTooShort {
            param: "packed",
            expected_at_least: required,
            got: packed.len(),
        });
    }
    let mut signs = Vec::with_capacity(d);
    for i in 0..d {
        let bit = (packed[i / 8] >> (7 - (i % 8))) & 1;
        signs.push(if bit == 1 { 1 } else { -1 });
    }
    Ok(signs)
}

/// Pack `b`-bit unsigned indices into a compact byte buffer.
///
/// For `bit_width` 1–4, multiple indices are packed per byte (MSB-first).
/// For `bit_width` 5–8, each index occupies one `u8`.
///
/// Returns an error if `bit_width` is outside `1..=8` or if values do not fit.
pub fn pack_indices(indices: &[u8], bit_width: u8) -> Result<Vec<u8>> {
    if !(1..=8).contains(&bit_width) {
        return Err(TurboQuantError::InvalidBitWidthU8 {
            param: "bit_width",
            got: bit_width,
            min: 1,
            max: 8,
        });
    }
    if bit_width > 4 {
        let max_val = if bit_width == 8 {
            u8::MAX
        } else {
            (1u8 << bit_width) - 1
        };
        if let Some((index, &value)) = indices.iter().enumerate().find(|(_, v)| **v > max_val) {
            return Err(TurboQuantError::ValueOutOfRange {
                param: "indices",
                index,
                got: value,
                max: max_val,
            });
        }
        return Ok(indices.to_vec());
    }
    // Convert each index to `bit_width` binary digits, then pack 8 bits per byte.
    let total_bits = indices.len() * bit_width as usize;
    let n_bytes = total_bits.div_ceil(8);
    let mut packed = vec![0u8; n_bytes];
    let bw = bit_width as usize;
    let max_val = (1u8 << bit_width) - 1;
    for (idx_pos, &val) in indices.iter().enumerate() {
        if val > max_val {
            return Err(TurboQuantError::ValueOutOfRange {
                param: "indices",
                index: idx_pos,
                got: val,
                max: max_val,
            });
        }
        for b in 0..bw {
            let bit = (val >> (bw - 1 - b)) & 1;
            let flat_pos = idx_pos * bw + b;
            if bit == 1 {
                packed[flat_pos / 8] |= 1 << (7 - (flat_pos % 8));
            }
        }
    }
    Ok(packed)
}

/// Unpack a packed index buffer back to individual indices.
///
/// # Arguments
/// * `packed` — byte buffer from [`pack_indices`].
/// * `n_indices` — number of original indices.
/// * `bit_width` — bits per index (1–8).
pub fn unpack_indices(packed: &[u8], n_indices: usize, bit_width: u8) -> Result<Vec<u8>> {
    if !(1..=8).contains(&bit_width) {
        return Err(TurboQuantError::InvalidBitWidthU8 {
            param: "bit_width",
            got: bit_width,
            min: 1,
            max: 8,
        });
    }
    if bit_width > 4 {
        if packed.len() < n_indices {
            return Err(TurboQuantError::BufferTooShort {
                param: "packed",
                expected_at_least: n_indices,
                got: packed.len(),
            });
        }
        return Ok(packed[..n_indices].to_vec());
    }
    let bw = bit_width as usize;
    let required_bits = n_indices * bw;
    let required_bytes = required_bits.div_ceil(8);
    if packed.len() < required_bytes {
        return Err(TurboQuantError::BufferTooShort {
            param: "packed",
            expected_at_least: required_bytes,
            got: packed.len(),
        });
    }
    let mut indices = Vec::with_capacity(n_indices);
    for idx_pos in 0..n_indices {
        let mut val = 0u8;
        for b in 0..bw {
            let flat_pos = idx_pos * bw + b;
            let bit = (packed[flat_pos / 8] >> (7 - (flat_pos % 8))) & 1;
            val |= bit << (bw - 1 - b);
        }
        indices.push(val);
    }
    Ok(indices)
}

/// Memory footprint breakdown for compressed KV cache vectors.
#[derive(Debug, Clone)]
pub struct MemoryFootprint {
    pub mse_indices_bytes: usize,
    pub qjl_signs_bytes: usize,
    pub norms_bytes: usize,
    pub total_bytes: usize,
    pub original_fp16_bytes: usize,
    pub compression_ratio: f64,
}

/// Calculate memory footprint of compressed KV cache.
///
/// # Arguments
/// * `n_vectors` — number of vectors stored.
/// * `d` — vector dimension.
/// * `bit_width` — total bits per coordinate (PolarQuant uses `bit_width - 1`, QJL uses 1).
pub fn memory_footprint_bytes(
    n_vectors: usize,
    d: usize,
    bit_width: usize,
) -> Result<MemoryFootprint> {
    if bit_width < 2 {
        return Err(TurboQuantError::InvalidBitWidth {
            param: "bit_width",
            got: bit_width as u32,
            min: 2,
            max: u32::MAX,
        });
    }
    let mse_bits = bit_width - 1;
    let mse_bytes = (n_vectors * d * mse_bits).div_ceil(8);
    let qjl_bytes = (n_vectors * d).div_ceil(8);
    let norm_bytes = n_vectors * 4; // f32 per vector
    let total = mse_bytes + qjl_bytes + norm_bytes;
    let original = n_vectors * d * 2; // fp16

    Ok(MemoryFootprint {
        mse_indices_bytes: mse_bytes,
        qjl_signs_bytes: qjl_bytes,
        norms_bytes: norm_bytes,
        total_bytes: total,
        original_fp16_bytes: original,
        compression_ratio: if total > 0 {
            original as f64 / total as f64
        } else {
            f64::INFINITY
        },
    })
}

/// Pack a 2D batch of signs. Each row is packed independently.
pub fn pack_bits_batch(signs: &[Vec<i8>]) -> Vec<Vec<u8>> {
    signs.iter().map(|row| pack_bits(row)).collect()
}

/// Unpack a 2D batch of packed signs.
pub fn unpack_bits_batch(packed: &[Vec<u8>], d: usize) -> Result<Vec<Vec<i8>>> {
    packed.iter().map(|row| unpack_bits(row, d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_bits_roundtrip() {
        let signs: Vec<i8> = vec![1, -1, 1, 1, -1, -1, 1, -1, 1, -1];
        let packed = pack_bits(&signs);
        let unpacked = unpack_bits(&packed, signs.len()).unwrap();
        assert_eq!(signs, unpacked);
    }

    #[test]
    fn test_pack_bits_values() {
        // [+1, -1, +1, +1, -1, -1, +1, -1] → bits: 10110010 = 0xB2
        let signs: Vec<i8> = vec![1, -1, 1, 1, -1, -1, 1, -1];
        let packed = pack_bits(&signs);
        assert_eq!(packed, vec![0xB2]);
    }

    #[test]
    fn test_pack_unpack_indices_roundtrip() {
        for bw in 1..=4u8 {
            let max_val = (1u8 << bw) - 1;
            let indices: Vec<u8> = (0..20).map(|i| i % (max_val + 1)).collect();
            let packed = pack_indices(&indices, bw).unwrap();
            let unpacked = unpack_indices(&packed, indices.len(), bw).unwrap();
            assert_eq!(indices, unpacked, "bit_width={bw}");
        }
    }

    #[test]
    fn test_pack_indices_passthrough_high_bits() {
        let indices: Vec<u8> = vec![0, 100, 200, 255];
        let packed = pack_indices(&indices, 8).unwrap();
        assert_eq!(packed, indices);
    }

    #[test]
    fn test_memory_footprint() {
        let mf = memory_footprint_bytes(100, 128, 3).unwrap();
        assert!(mf.compression_ratio > 1.0);
        assert_eq!(mf.original_fp16_bytes, 100 * 128 * 2);
    }
}
use crate::error::{Result, TurboQuantError};
