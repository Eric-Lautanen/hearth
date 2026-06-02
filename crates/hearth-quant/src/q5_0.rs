use half::f16;
use wide::f32x8;

const BLOCK_SIZE: usize = 32;
const BLOCK_BYTES: usize = 22;

pub fn dequantize(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let bp = b * BLOCK_BYTES;
        let d_f32: f32 = f16::from_le_bytes([data[bp], data[bp + 1]]).to_f32();
        let qh_off = bp + 2;
        let qs_off = bp + 6;

        for i in 0..BLOCK_SIZE {
            let idx = b * BLOCK_SIZE + i;
            if idx >= out.len() {
                break;
            }
            let qs_byte = data[qs_off + i / 2];
            let low4 = if i % 2 == 0 {
                qs_byte & 0x0F
            } else {
                (qs_byte >> 4) & 0x0F
            };
            let qh_bit = (data[qh_off + i / 8] >> (i % 8)) & 1;
            let q = (low4 as i32) | ((qh_bit as i32) << 4);
            out[idx] = (q as f32 - 16.0) * d_f32;
        }
    }
}

pub fn dot_q5_0_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let bp = b * BLOCK_BYTES;
        let d_f32 = f16::from_le_bytes([row[bp], row[bp + 1]]).to_f32();
        let qh_off = bp + 2;
        let qs_off = bp + 6;
        let vec_base = b * BLOCK_SIZE;

        let mut vals = [0.0f32; 32];
        for i in 0..32 {
            let qs_byte = row[qs_off + i / 2];
            let low4 = if i % 2 == 0 {
                qs_byte & 0x0F
            } else {
                (qs_byte >> 4) & 0x0F
            };
            let qh_bit = (row[qh_off + i / 8] >> (i % 8)) & 1;
            let q = (low4 as i32) | ((qh_bit as i32) << 4);
            vals[i] = (q as f32 - 16.0) * d_f32;
        }

        let mut vsum = f32x8::ZERO;
        for i in (0..32).step_by(8) {
            let vw = f32x8::new(vals[i..i + 8].try_into().unwrap());
            let vv = f32x8::from(&vec[vec_base + i..vec_base + i + 8]);
            vsum = vw.mul_add(vv, vsum);
        }
        sum += vsum.reduce_add();
    }
    let remainder_start = blocks * BLOCK_SIZE;
    let rem = n - remainder_start;
    if rem > 0 {
        let bp = blocks * BLOCK_BYTES;
        let d_f32 = f16::from_le_bytes([row[bp], row[bp + 1]]).to_f32();
        let qh_off = bp + 2;
        let qs_off = bp + 6;
        let mut block_sum = 0.0f32;
        for i in 0..rem {
            let qs_byte = row[qs_off + i / 2];
            let low4 = if i % 2 == 0 {
                qs_byte & 0x0F
            } else {
                (qs_byte >> 4) & 0x0F
            };
            let qh_bit = (row[qh_off + i / 8] >> (i % 8)) & 1;
            let q = (low4 as i32) | ((qh_bit as i32) << 4);
            block_sum += (q as f32 - 16.0) * d_f32 * vec[remainder_start + i];
        }
        sum += block_sum;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantize_q5_0(src: &[f32]) -> Vec<u8> {
        let n = src.len();
        let blocks = n.div_ceil(32);
        let mut dst = Vec::with_capacity(blocks * 22);
        for b in 0..blocks {
            let start = b * 32;
            let mut max_abs = 0.0f32;
            for i in 0..32 {
                let idx = start + i;
                if idx < n {
                    max_abs = max_abs.max(src[idx].abs());
                }
            }
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 15.0 };
            let scale_f16 = f16::from_f32(scale);
            dst.extend_from_slice(&scale_f16.to_le_bytes());
            let mut qh = [0u8; 4];
            let mut qs_bytes = [0u8; 16];
            for i in 0..16 {
                let idx0 = start + i * 2;
                let idx1 = start + i * 2 + 1;
                let q0 = if idx0 < n {
                    ((src[idx0] / scale + 16.0).round().clamp(0.0, 31.0)) as u8
                } else {
                    16u8
                };
                let q1 = if idx1 < n {
                    ((src[idx1] / scale + 16.0).round().clamp(0.0, 31.0)) as u8
                } else {
                    16u8
                };
                if q0 & 0x10 != 0 {
                    qh[(i * 2) / 8] |= 1 << ((i * 2) % 8);
                }
                if q1 & 0x10 != 0 {
                    qh[(i * 2 + 1) / 8] |= 1 << ((i * 2 + 1) % 8);
                }
                qs_bytes[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
            dst.extend_from_slice(&qh);
            dst.extend_from_slice(&qs_bytes);
        }
        dst
    }

    #[test]
    fn test_q5_0_dequantize() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let quant = quantize_q5_0(&src);
        let mut dequant = vec![0.0f32; 64];
        dequantize(&quant, &mut dequant);
        for i in 0..64 {
            let err = (src[i] - dequant[i]).abs();
            assert!(
                err < 0.2,
                "Error at {}: src={} deq={}",
                i,
                src[i],
                dequant[i]
            );
        }
    }

    #[test]
    fn test_dot_q5_0() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let vec: Vec<f32> = (0..64).map(|i| i as f32 * 0.05).collect();
        let quant = quantize_q5_0(&src);
        let dot = dot_q5_0_f32(&quant, &vec, 64);
        let expected: f32 = src.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
        let err = (dot - expected).abs();
        assert!(
            err < expected.abs() * 0.1 + 1.0,
            "dot error: {} vs {}, diff={}",
            dot,
            expected,
            err
        );
    }
}
