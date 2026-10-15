// SPDX-License-Identifier: MIT OR Apache-2.0

// Q1_0_G128 block: 128 elements, 18 bytes total
//   [0..2)  f16 scale (little-endian)
//   [2..18) 16 bytes packed sign bits (bit=0 → -1, bit=1 → +1)
const Q1_0G128_BLOCK_SIZE: u32 = 128u;
const Q1_0G128_BLOCK_BYTES: u32 = 18u;

// Q2_0 block: 128 elements (32 packed 2-bit values per byte × 4 bytes + header)
//   [0..2)  f16 scale (little-endian)
//   [2..34) 32 bytes packed 2-bit values (only used if d % 128 == 0)
const Q2_0_BLOCK_SIZE: u32 = 128u;
const Q2_0_BLOCK_BYTES: u32 = 34u;

// Q8_0 block: 32 elements, 34 bytes total
//   [0..2)  f16 scale (little-endian)
//   [2..34) 32 bytes i8 values
const Q8_0_BLOCK_SIZE: u32 = 32u;
const Q8_0_BLOCK_BYTES: u32 = 34u;

/// Extract sign (+1/-1) from a Q1_0_G128 packed byte for a specific element within that byte.
fn q1_0_sign(packed_byte: u32, bit_idx: u32) -> f32 {
    // bit=0 → -1, bit=1 → +1
    let bit = (packed_byte >> bit_idx) & 1u;
    return select(-1.0, 1.0, bit == 1u);
}

/// Dequantize one element from a Q1_0_G128 block.
/// block_start: byte offset of the Q1_0 block (18-byte aligned).
/// elem_idx: element index within the block (0..128).
fn q1_0_dequant_elem(weight_buf: ptr<function, array<u32>>, block_start: u32, elem_idx: u32) -> f32 {
    let scale_lo = weight_buf[block_start / 4u];     // contains the f16 scale bytes
    // Extract f16 scale (little-endian) — the 2 bytes are at block_start
    let lo_byte = (scale_lo >> ((block_start % 4u) * 8u)) & 0xFFu;
    let hi_byte = (scale_lo >> ((block_start % 4u) * 8u + 8u)) & 0xFFu;
    // f16 bits: 1 sign, 5 exponent, 10 mantissa
    let sign_f16 = f32(bitcast<u32>((lo_byte >> 7u) & 1u));
    let exp_f16 = (hi_byte >> 2u) & 0x1Fu;
    let mant_f16 = ((u32(hi_byte & 0x3u) << 8u) | u32(lo_byte & 0xFFu)) & 0x3FFu;
    let scale = unpack_f16_mant(sign_f16, exp_f16, mant_f16);

    let byte_idx = elem_idx / 8u;   // which of the 16 sign bytes
    let bit_idx = elem_idx % 8u;    // which bit within that byte
    let bits_byte_offset = block_start + 2u + byte_idx;
    let bits_word = weight_buf[bits_byte_offset / 4u];
    let byte_val = (bits_word >> ((bits_byte_offset % 4u) * 8u)) & 0xFFu;
    let sign = q1_0_sign(byte_val, bit_idx);
    return sign * scale;
}

/// Unpack f16 to f32 (software fallback when hardware doesn't support f16).
fn unpack_f16_mant(sign: f32, exp: u32, mant: u32) -> f32 {
    // f16: 1 sign, 5 exponent, 10 mantissa
    // f32: 1 sign, 8 exponent, 23 mantissa
    if exp == 0u {
        // Zero or subnormal
        if mant == 0u { return 0.0; }
        // Subnormal: normalize
        let norm_exp = 1u - 15i - count_leading_zeros(mant) + 10i; // approximate
        return 0.0; // simplified — subnormals rare in practice
    }
    if exp == 31u {
        // Infinity or NaN
        if mant == 0u { return sign * 1.0 / 0.0; }
        return 0.0 / 0.0; // NaN
    }
    // Normal f16: exponent - 15 + 127 = exponent + 112
    let f32_bits = (u32(sign) << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    return bitcast<f32>(f32_bits);
}

/// Workgroup reduction: sum all values in shared memory.
/// Assumes `shared` contains partial sums from each invocation.
/// Returns the total sum in invocation 0.
fn wg_reduce_sum(shared: ptr<workgroup, array<f32>>, lid: u32) -> f32 {
    // Tree reduction: each step halves the active invocations
    var offset = 64u;
    while offset > 0u {
        if lid < offset {
            shared[lid] = shared[lid] + shared[lid + offset];
        }
        workgroupBarrier();
        offset = offset / 2u;
    }
    return shared[0];
}
