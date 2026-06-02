use half::f16;
use wide::f32x8;

const SUPER_BLOCK_SIZE: usize = 256;
const SUB_BLOCK_SIZE: usize = 32;
const SUB_BLOCKS: usize = 8;
const BLOCK_BYTES: usize = 144;

pub fn dequantize(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let block_start = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let dmin = f16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
        let d_f32: f32 = d.to_f32();
        let dmin_f32: f32 = dmin.to_f32();

        let scales_data = &data[block_start + 4..block_start + 16];
        let mut sub_scale = [0u8; 8];
        let mut sub_min = [0u8; 8];
        for i in 0..4 {
            let s0 = scales_data[i * 3];
            let s1 = scales_data[i * 3 + 1];
            let s2 = scales_data[i * 3 + 2];
            sub_scale[i * 2] = s0 & 0x3F;
            sub_min[i * 2] = (s0 >> 6) | ((s1 & 0x0F) << 2);
            sub_scale[i * 2 + 1] = (s1 >> 4) | ((s2 & 0x03) << 4);
            sub_min[i * 2 + 1] = (s2 >> 2) & 0x3F;
        }

        let qs_start = block_start + 16;
        for sb in 0..SUB_BLOCKS {
            let scale = (sub_scale[sb] as f32) * d_f32;
            let min = (sub_min[sb] as f32) * dmin_f32;
            for i in 0..SUB_BLOCK_SIZE {
                let byte_idx = (sb * SUB_BLOCK_SIZE + i) / 2;
                let nibble = if i % 2 == 0 {
                    data[qs_start + byte_idx] & 0x0F
                } else {
                    (data[qs_start + byte_idx] >> 4) & 0x0F
                };
                let idx = b * SUPER_BLOCK_SIZE + sb * SUB_BLOCK_SIZE + i;
                if idx < out.len() {
                    out[idx] = (nibble as f32) * scale - min;
                }
            }
        }
    }
}

pub fn dot_q4_k_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / SUPER_BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let bp = b * BLOCK_BYTES;
        let d_f32 = f16::from_le_bytes([row[bp], row[bp + 1]]).to_f32();
        let dmin_f32 = f16::from_le_bytes([row[bp + 2], row[bp + 3]]).to_f32();
        let scales_data = &row[bp + 4..bp + 16];
        let mut sub_scale = [0u8; 8];
        let mut sub_min = [0u8; 8];
        for i in 0..4 {
            let s0 = scales_data[i * 3];
            let s1 = scales_data[i * 3 + 1];
            let s2 = scales_data[i * 3 + 2];
            sub_scale[i * 2] = s0 & 0x3F;
            sub_min[i * 2] = (s0 >> 6) | ((s1 & 0x0F) << 2);
            sub_scale[i * 2 + 1] = (s1 >> 4) | ((s2 & 0x03) << 4);
            sub_min[i * 2 + 1] = (s2 >> 2) & 0x3F;
        }
        let qs = &row[bp + 16..bp + BLOCK_BYTES];
        for sb in 0..SUB_BLOCKS {
            let sc = (sub_scale[sb] as f32) * d_f32;
            let mn = (sub_min[sb] as f32) * dmin_f32;
            let vec_base = b * SUPER_BLOCK_SIZE + sb * SUB_BLOCK_SIZE;
            // Unpack nibbles into f32, then SIMD dot with vec
            let mut nibs = [0.0f32; 32];
            for (i, nib_out) in nibs.iter_mut().enumerate() {
                let byte_idx = (sb * SUB_BLOCK_SIZE + i) / 2;
                let nib = if i % 2 == 0 {
                    qs[byte_idx] & 0x0F
                } else {
                    (qs[byte_idx] >> 4) & 0x0F
                };
                *nib_out = nib as f32 * sc - mn;
            }
            let mut vsum = f32x8::ZERO;
            for i in (0..32).step_by(8) {
                let vw = f32x8::new(nibs[i..i + 8].try_into().unwrap());
                let vv = f32x8::from(&vec[vec_base + i..vec_base + i + 8]);
                vsum = vw.mul_add(vv, vsum);
            }
            sum += vsum.reduce_add();
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_k_dequantize() {
        let data = vec![0u8; 144 * 2]; // 2 blocks of zeros
        let mut out = vec![0.0f32; 512];
        dequantize(&data, &mut out);
        for v in &out {
            assert!((*v).abs() < 1e-6);
        }
    }
}
