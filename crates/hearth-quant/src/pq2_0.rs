// PQ2_0: Prism 2-bit ternary packing, GGML type 142.
// 128-element blocks, 34 bytes: fp16 scale d FIRST, then qs[32] (2 bits/element).
// 2-bit codes: 00 = -1, 01 = 0, 10 = +1, 11 = +2 (11 unused for ternary weights).
// Reference: PrismML-Eng/llama.cpp (prism branch) ggml-common.h / ggml-quants.c.
// NOTE: identical per-block layout to Hearth's Q2_0 (type 42), but PQ2_0 files carry
// a Hadamard-rotated basis and must not be run through this kernel interchangeably
// with un-rotated Q2_0 weights.

use half::f16;

const BLOCK_SIZE: usize = 128;
const BLOCK_BYTES: usize = 34;

/// PQ2V[b][j] = 2-bit code j of byte b mapped through {0:-1, 1:0, 2:+1, 3:+2}.
const PQ2V: [[i8; 4]; 256] = {
    let mut table = [[0i8; 4]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut j = 0;
        while j < 4 {
            let q = (b >> (j * 2)) & 3;
            table[b][j] = q as i8 - 1;
            j += 1;
        }
        b += 1;
    }
    table
};

/// Dequantize PQ2_0 (type 142) to f32, exactly matching dequantize_row_pq2_0:
/// within a byte, element j uses bit offset (j % 4) * 2, LSB first.
pub fn dequantize(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([data[bo], data[bo + 1]]).to_f32();
        let qs = &data[bo + 2..bo + 2 + 32];
        let out_base = b * BLOCK_SIZE;
        for byte_i in 0..32 {
            let vals = PQ2V[qs[byte_i] as usize];
            let elem_base = out_base + byte_i * 4;
            for (i, &v) in vals.iter().enumerate() {
                let idx = elem_base + i;
                if idx < out.len() {
                    out[idx] = (v as f32) * scale;
                }
            }
        }
    }
}

/// Dot product of one PQ2_0 row with an f32 vector.
pub fn dot_pq2_0_f32(weight_row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([weight_row[bo], weight_row[bo + 1]]).to_f32();
        let qs = &weight_row[bo + 2..bo + 2 + 32];
        let base = b * BLOCK_SIZE;
        let mut block_acc = 0.0f32;
        for byte_i in 0..32 {
            let vals = PQ2V[qs[byte_i] as usize];
            let elem_base = base + byte_i * 4;
            block_acc += vals[0] as f32 * vec[elem_base]
                + vals[1] as f32 * vec[elem_base + 1]
                + vals[2] as f32 * vec[elem_base + 2]
                + vals[3] as f32 * vec[elem_base + 3];
        }
        sum += block_acc * scale;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_mapping() {
        // Byte 0b00100100 = codes 0,1,2,0 (LSB first) -> values -1, 0, +1, -1.
        let mut data = vec![0u8; BLOCK_BYTES * 2];
        let s = f16::from_f32(2.0).to_le_bytes();
        data[0] = s[0];
        data[1] = s[1];
        data[2] = 0b00100100;
        let mut out = vec![0.0f32; BLOCK_SIZE * 2];
        dequantize(&data, &mut out);
        assert_eq!(out[0], -2.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], -2.0);
    }

    #[test]
    fn test_dot_matches_dequant() {
        let mut w = vec![0u8; BLOCK_BYTES * 2];
        for b in 0..2 {
            let bo = b * BLOCK_BYTES;
            let scale = if b == 0 { 1.5f32 } else { 2.0f32 };
            let ws = f16::from_f32(scale).to_le_bytes();
            w[bo] = ws[0];
            w[bo + 1] = ws[1];
            for j in 0..32 {
                w[bo + 2 + j] = (j as u8).wrapping_mul(37) ^ 0xAB;
            }
        }
        let vec: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.13).sin()).collect();
        let mut deq = vec![0.0f32; 256];
        dequantize(&w, &mut deq);
        let expected: f32 = deq.iter().zip(vec.iter()).map(|(&wv, &x)| wv * x).sum();
        let got = dot_pq2_0_f32(&w, &vec, 256);
        assert!(
            (got - expected).abs() < 1e-3 * expected.abs().max(1.0),
            "dot={} expected={}",
            got,
            expected
        );
    }

    #[test]
    fn test_all_codes_within_range() {
        // Any byte pattern must dequantize to {-1,0,+1,+2}*scale.
        let mut data = vec![0u8; BLOCK_BYTES];
        for i in 0..32 {
            data[2 + i] = (i * 73 + 5) as u8;
        }
        let s = f16::from_f32(1.0).to_le_bytes();
        data[0] = s[0];
        data[1] = s[1];
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequantize(&data, &mut out);
        assert!(out.iter().all(|&v| (-2.0..=2.0).contains(&v)));
        assert!(out.iter().all(|&v| v.is_finite()));
    }
}
