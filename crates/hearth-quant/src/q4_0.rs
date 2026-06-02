use half::f16;
use wide::f32x8;

const BLOCK_SIZE: usize = 32;
const BLOCK_BYTES: usize = 18;

pub fn dequantize(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let block_start = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let d_f32: f32 = d.to_f32();

        let qs_start = block_start + 2;
        for i in 0..BLOCK_SIZE {
            let byte_idx = qs_start + i / 2;
            let nibble = if i % 2 == 0 {
                data[byte_idx] & 0x0F
            } else {
                (data[byte_idx] >> 4) & 0x0F
            };
            let idx = b * BLOCK_SIZE + i;
            if idx < out.len() {
                out[idx] = (nibble as f32 - 8.0) * d_f32;
            }
        }
    }
}

pub fn dot_q4_0_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let block_start = b * BLOCK_BYTES;
        let d_f32 = f16::from_le_bytes([row[block_start], row[block_start + 1]]).to_f32();
        let qs_start = block_start + 2;
        let vec_base = b * BLOCK_SIZE;
        let mut nibs = [0.0f32; 32];
        for (i, nib_out) in nibs.iter_mut().enumerate() {
            let byte_idx = qs_start + i / 2;
            let nibble = if i % 2 == 0 {
                row[byte_idx] & 0x0F
            } else {
                (row[byte_idx] >> 4) & 0x0F
            };
            *nib_out = (nibble as f32 - 8.0) * d_f32;
        }
        let mut vsum = f32x8::ZERO;
        for i in (0..32).step_by(8) {
            let vw = f32x8::new(nibs[i..i + 8].try_into().unwrap());
            let vv = f32x8::from(&vec[vec_base + i..vec_base + i + 8]);
            vsum = vw.mul_add(vv, vsum);
        }
        sum += vsum.reduce_add();
    }
    let remainder_start = blocks * BLOCK_SIZE;
    let rem = n - remainder_start;
    if rem > 0 {
        let b = blocks;
        let block_start = b * BLOCK_BYTES;
        let d_f32 = f16::from_le_bytes([row[block_start], row[block_start + 1]]).to_f32();
        let qs_start = block_start + 2;
        let mut block_sum = 0.0f32;
        for i in 0..rem {
            let byte_idx = qs_start + i / 2;
            let nibble = if i % 2 == 0 {
                row[byte_idx] & 0x0F
            } else {
                (row[byte_idx] >> 4) & 0x0F
            };
            block_sum += (nibble as f32 - 8.0) * d_f32 * vec[remainder_start + i];
        }
        sum += block_sum;
    }
    sum
}

pub fn dot_q4_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sumf = 0.0f32;
    for b in 0..blocks {
        let bp = b * BLOCK_BYTES;
        let d0 = f16::from_le_bytes([row[bp], row[bp + 1]]).to_f32();
        let qs = &row[bp + 2..bp + BLOCK_BYTES];
        let ap = b * 34;
        let d1 = f16::from_le_bytes([act[ap], act[ap + 1]]).to_f32();
        let qy = &act[ap + 2..ap + 34];
        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;
        for j in 0..16 {
            let v0 = (qs[j] & 0x0F) as i32 - 8;
            let v1 = (qs[j] as i32 >> 4) - 8;
            sumi0 += v0 * (qy[j] as i8 as i32);
            sumi1 += v1 * (qy[j + 16] as i8 as i32);
        }
        sumf += (sumi0 + sumi1) as f32 * d0 * d1;
    }
    sumf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantize_q4_0(src: &[f32]) -> Vec<u8> {
        let n = src.len();
        let blocks = n.div_ceil(32);
        let mut dst = Vec::with_capacity(blocks * 18);
        for b in 0..blocks {
            let start = b * 32;
            let mut max_abs = 0.0f32;
            for i in 0..32 {
                let idx = start + i;
                if idx < n {
                    max_abs = max_abs.max(src[idx].abs());
                }
            }
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            let scale_f16 = f16::from_f32(scale);
            dst.extend_from_slice(&scale_f16.to_le_bytes());
            for i in 0..16 {
                let idx0 = start + i * 2;
                let idx1 = start + i * 2 + 1;
                let q0 = if idx0 < n {
                    ((src[idx0] / scale + 8.0).round().clamp(0.0, 15.0)) as u8
                } else {
                    8u8
                };
                let q1 = if idx1 < n {
                    ((src[idx1] / scale + 8.0).round().clamp(0.0, 15.0)) as u8
                } else {
                    8u8
                };
                dst.push(q0 | (q1 << 4));
            }
        }
        dst
    }

    #[test]
    fn test_q4_0_dequantize() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let quant = quantize_q4_0(&src);
        let mut dequant = vec![0.0f32; 64];
        dequantize(&quant, &mut dequant);
        for i in 0..64 {
            let err = (src[i] - dequant[i]).abs();
            assert!(
                err < 0.3,
                "Error at {}: src={} deq={}",
                i,
                src[i],
                dequant[i]
            );
        }
    }

    #[test]
    fn test_dot_q4_0() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let vec: Vec<f32> = (0..64).map(|i| i as f32 * 0.05).collect();
        let quant = quantize_q4_0(&src);
        let dot = dot_q4_0_f32(&quant, &vec, 64);
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
