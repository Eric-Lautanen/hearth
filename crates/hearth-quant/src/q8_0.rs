use half::f16;
use wide::f32x8;

pub fn dequantize(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / 34;
    for b in 0..blocks {
        let block_start = b * 34;
        let scale = f16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale_f32: f32 = scale.to_f32();
        let vals_start = block_start + 2;
        for i in 0..32 {
            let idx = b * 32 + i;
            if idx < out.len() {
                out[idx] = scale_f32 * (data[vals_start + i] as i8) as f32;
            }
        }
    }
}

pub fn quantize_into(src: &[f32], dst: &mut [u8]) {
    let n = src.len();
    let blocks = n.div_ceil(32);
    for b in 0..blocks {
        let start = b * 32;
        let mut max_abs: f32 = 0.0;
        for i in 0..32 {
            let idx = start + i;
            if idx < n {
                let v = src[idx].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let scale_f16 = f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        let off = b * 34;
        dst[off] = scale_bytes[0];
        dst[off + 1] = scale_bytes[1];
        for i in 0..32 {
            let idx = start + i;
            let q = if idx < n {
                (src[idx] / scale).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
            dst[off + 2 + i] = q as u8;
        }
    }
}

pub fn quantize(src: &[f32], dst: &mut Vec<u8>) {
    let n = src.len();
    let blocks = n.div_ceil(32);
    dst.reserve(blocks * 34);
    for b in 0..blocks {
        let start = b * 32;
        let mut max_abs: f32 = 0.0;
        for i in 0..32 {
            let idx = start + i;
            if idx < n {
                let v = src[idx].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let scale_f16 = f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        dst.push(scale_bytes[0]);
        dst.push(scale_bytes[1]);
        for i in 0..32 {
            let idx = start + i;
            let q = if idx < n {
                let v = (src[idx] / scale).round().clamp(-128.0, 127.0);
                v as i8
            } else {
                0
            };
            dst.push(q as u8);
        }
    }
}

/// Fused Q8_0 dot product: dequantize and dot in one pass.
/// Returns sum(row[i] * vec[i]) for i in 0..n.
pub fn dot_q8_0_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / 32;
    let mut sum = 0.0_f32;
    for b in 0..blocks {
        let block_start = b * 34;
        let scale = f16::from_le_bytes([row[block_start], row[block_start + 1]]);
        let scale_f32: f32 = scale.to_f32();
        let vals_start = block_start + 2;
        let vec_base = b * 32;
        let mut vsum = f32x8::ZERO;
        for i in (0..32).step_by(8) {
            let w0 = row[vals_start + i] as i8 as f32;
            let w1 = row[vals_start + i + 1] as i8 as f32;
            let w2 = row[vals_start + i + 2] as i8 as f32;
            let w3 = row[vals_start + i + 3] as i8 as f32;
            let w4 = row[vals_start + i + 4] as i8 as f32;
            let w5 = row[vals_start + i + 5] as i8 as f32;
            let w6 = row[vals_start + i + 6] as i8 as f32;
            let w7 = row[vals_start + i + 7] as i8 as f32;
            let vw = f32x8::new([w0, w1, w2, w3, w4, w5, w6, w7]);
            let vv = f32x8::from(&vec[vec_base + i..vec_base + i + 8]);
            vsum = vw.mul_add(vv, vsum);
        }
        sum += vsum.reduce_add() * scale_f32;
    }
    let remainder_start = blocks * 32;
    let rem = n - remainder_start;
    if rem > 0 {
        let b = blocks;
        let block_start = b * 34;
        let scale = f16::from_le_bytes([row[block_start], row[block_start + 1]]);
        let scale_f32: f32 = scale.to_f32();
        let vals_start = block_start + 2;
        let mut block_sum = 0.0_f32;
        for i in 0..rem {
            let w = (row[vals_start + i] as i8) as f32;
            block_sum += w * vec[remainder_start + i];
        }
        sum += block_sum * scale_f32;
    }
    sum
}

pub fn dot_q8_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    let blocks = n / 32;
    let mut sumf = 0.0f32;
    for b in 0..blocks {
        let wp = b * 34;
        let d0 = f16::from_le_bytes([row[wp], row[wp + 1]]).to_f32();
        let w_vals = &row[wp + 2..wp + 34];
        let ap = b * 34;
        let d1 = f16::from_le_bytes([act[ap], act[ap + 1]]).to_f32();
        let a_vals = &act[ap + 2..ap + 34];
        let mut sumi: i32 = 0;
        for i in 0..32 {
            sumi += (w_vals[i] as i8 as i32) * (a_vals[i] as i8 as i32);
        }
        sumf += sumi as f32 * d0 * d1;
    }
    sumf
}

/// Raw-pointer variant of dot_q8_0_q8_0.
/// # Safety
/// w_ptr valid for n/32*34 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q8_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let blocks = n / 32;
    let mut sumf = 0.0f32;
    for b in 0..blocks {
        let wp = b * 34;
        let d0 = f16::from_le_bytes([*w_ptr.add(wp), *w_ptr.add(wp + 1)]).to_f32();
        let w_vals = w_ptr.add(wp + 2);
        let ap = b * 34;
        let d1 = f16::from_le_bytes([*a_ptr.add(ap), *a_ptr.add(ap + 1)]).to_f32();
        let a_vals = a_ptr.add(ap + 2);
        let mut sumi: i32 = 0;
        for i in 0..32 {
            sumi += (*w_vals.add(i) as i8 as i32) * (*a_vals.add(i) as i8 as i32);
        }
        sumf += sumi as f32 * d0 * d1;
    }
    sumf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_q8_0() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let mut quant = Vec::new();
        quantize(&src, &mut quant);
        let mut dequant = vec![0.0_f32; 64];
        dequantize(&quant, &mut dequant);
        for i in 0..64 {
            let err = (src[i] - dequant[i]).abs();
            assert!(
                err < 1.0,
                "Error at {}: src={} deq={}",
                i,
                src[i],
                dequant[i]
            );
        }
    }

    #[test]
    fn test_dot_q8_0() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let vec: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let mut quant = Vec::new();
        quantize(&src, &mut quant);
        let dot = dot_q8_0_f32(&quant, &vec, 64);
        let expected: f32 = src.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
        let err = (dot - expected).abs();
        assert!(
            err < expected.abs() * 0.02 + 1.0,
            "dot error: {} vs {}, diff={}",
            dot,
            expected,
            err
        );
    }
}
