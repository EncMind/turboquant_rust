//! Tests for bit packing and memory utilities.
//! Mirrors turboquant_plus/tests/test_utils.py.

use turboquant_core::utils::*;

// ============================================================
// Bit Packing
// ============================================================

#[test]
fn test_pack_unpack_round_trip() {
    let signs: Vec<i8> = vec![1, -1, 1, -1, 1, 1, -1, 1];
    let packed = pack_bits(&signs);
    let unpacked = unpack_bits(&packed, 8).unwrap();
    assert_eq!(signs, unpacked);
}

#[test]
fn test_pack_unpack_non_multiple_of_8() {
    let signs: Vec<i8> = vec![1, -1, 1, -1, 1];
    let packed = pack_bits(&signs);
    let unpacked = unpack_bits(&packed, 5).unwrap();
    assert_eq!(signs, unpacked);
}

#[test]
fn test_pack_correct_size() {
    // 128 signs → 16 bytes
    let signs: Vec<i8> = (0..128).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();
    let packed = pack_bits(&signs);
    assert_eq!(packed.len(), 16);
}

#[test]
fn test_batch_pack_unpack() {
    // 5 rows × 64 columns
    let batch: Vec<Vec<i8>> = (0..5)
        .map(|row| {
            (0..64)
                .map(|col| if (row + col) % 2 == 0 { 1 } else { -1 })
                .collect()
        })
        .collect();
    let packed = pack_bits_batch(&batch);
    assert_eq!(packed.len(), 5);
    assert_eq!(packed[0].len(), 8); // 64/8
    let unpacked = unpack_bits_batch(&packed, 64).unwrap();
    assert_eq!(batch, unpacked);
}

#[test]
fn test_pack_indices_round_trip_2bit() {
    let indices: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let packed = pack_indices(&indices, 2).unwrap();
    assert!(packed.len() < indices.len()); // compressed
    let unpacked = unpack_indices(&packed, indices.len(), 2).unwrap();
    assert_eq!(indices, unpacked);
}

#[test]
fn test_pack_indices_round_trip_all_widths() {
    for bw in 1..=4u8 {
        let max_val = (1u8 << bw) - 1;
        let indices: Vec<u8> = (0..20).map(|i| i % (max_val + 1)).collect();
        let packed = pack_indices(&indices, bw).unwrap();
        let unpacked = unpack_indices(&packed, indices.len(), bw).unwrap();
        assert_eq!(indices, unpacked, "bit_width={bw}");
    }
}

#[test]
fn test_pack_indices_invalid_bit_width_0() {
    let err = pack_indices(&[0, 1], 0).err().unwrap();
    assert!(
        matches!(
            err,
            turboquant_core::TurboQuantError::InvalidBitWidthU8 {
                param: "bit_width",
                got: 0,
                min: 1,
                max: 8
            }
        ),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pack_indices_invalid_bit_width_9() {
    let err = pack_indices(&[0, 1], 9).err().unwrap();
    assert!(
        matches!(
            err,
            turboquant_core::TurboQuantError::InvalidBitWidthU8 {
                param: "bit_width",
                got: 9,
                min: 1,
                max: 8
            }
        ),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pack_indices_8bit_passthrough() {
    let indices: Vec<u8> = vec![0, 127, 255];
    let packed = pack_indices(&indices, 8).unwrap();
    assert_eq!(packed, indices);
}

// ============================================================
// Memory Footprint
// ============================================================

#[test]
fn test_compression_ratio_3bit() {
    let result = memory_footprint_bytes(1000, 128, 3).unwrap();
    assert!(result.compression_ratio > 4.0);
    assert!(result.total_bytes < result.original_fp16_bytes);
}

#[test]
fn test_compression_ratio_4bit() {
    let result = memory_footprint_bytes(1000, 128, 4).unwrap();
    assert!(result.compression_ratio > 3.0);
}

#[test]
fn test_components_add_up() {
    let result = memory_footprint_bytes(100, 64, 3).unwrap();
    let expected_total = result.mse_indices_bytes + result.qjl_signs_bytes + result.norms_bytes;
    assert_eq!(result.total_bytes, expected_total);
}

#[test]
fn test_large_scale_footprint() {
    // Realistic: 32 layers × 32 heads × 32K seq = 33M vectors, d=128, 3-bit
    let result = memory_footprint_bytes(32 * 32 * 32768, 128, 3).unwrap();
    assert!(result.compression_ratio > 4.0);
    assert!(result.compression_ratio < 8.0);
}
