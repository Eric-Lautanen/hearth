use half::f16;

/// Q2_K dequant: 256 elements → f32, super-block structure
/// Block size: 84 bytes per 256 elements
pub fn dequantize_q2_k(data: &[u8], out: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 84;
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let bp = b * BLOCK_BYTES;
        let dp = b * BLOCK_SIZE;
        // scales: 16 uint8 values, each encodes scale_low (lower 4 bits) and scale_high (upper 4 bits)
        // dmin: 2 half-precision floats
        let d = f16::from_le_bytes([data[bp], data[bp + 1]]).to_f32();
        let dmin = f16::from_le_bytes([data[bp + 2], data[bp + 3]]).to_f32();
        // 16 byte-aligned scales (low nibble = scale, high nibble = min)
        let mut scales = [0u8; 16];
        let sc_off = bp + 4;
        scales.copy_from_slice(&data[sc_off..sc_off + 16]);
        // 64 bytes of quantized data (4 bits each → 256 values)
        let q_off = bp + 20;
        for j in 0..64 {
            let bv = data[q_off + j];
            // Each byte encodes 2 values
            let idx0 = j * 4;
            let idx1 = j * 4 + 1;
            let idx2 = j * 4 + 2;
            let idx3 = j * 4 + 3;
            // Scale index for each position
            let si0 = idx0 / 16;
            let si1 = idx1 / 16;
            let si2 = idx2 / 16;
            let si3 = idx3 / 16;
            let dl0 = d * (scales[si0] & 0xF) as f32;
            let ml0 = dmin * (scales[si0] >> 4) as f32;
            let dl1 = d * (scales[si1] & 0xF) as f32;
            let ml1 = dmin * (scales[si1] >> 4) as f32;
            let dl2 = d * (scales[si2] & 0xF) as f32;
            let ml2 = dmin * (scales[si2] >> 4) as f32;
            let dl3 = d * (scales[si3] & 0xF) as f32;
            let ml3 = dmin * (scales[si3] >> 4) as f32;
            let v0 = (bv & 0x03) as f32;
            let v1 = ((bv >> 2) & 0x03) as f32;
            let v2 = ((bv >> 4) & 0x03) as f32;
            let v3 = ((bv >> 6) & 0x03) as f32;
            let off = dp + j * 4;
            if off < out.len() {
                out[off] = v0 * dl0 - ml0;
            }
            if off + 1 < out.len() {
                out[off + 1] = v1 * dl1 - ml1;
            }
            if off + 2 < out.len() {
                out[off + 2] = v2 * dl2 - ml2;
            }
            if off + 3 < out.len() {
                out[off + 3] = v3 * dl3 - ml3;
            }
        }
    }
}

/// Q5_K dequant: 256 elements → f32
/// Block size: 176 bytes per 256 elements
pub fn dequantize_q5_k(data: &[u8], out: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 176;
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let bp = b * BLOCK_BYTES;
        let dp = b * BLOCK_SIZE;
        let d = f16::from_le_bytes([data[bp], data[bp + 1]]).to_f32();
        let dmin = f16::from_le_bytes([data[bp + 2], data[bp + 3]]).to_f32();
        let mut scales = [0u8; 16];
        let sc_off = bp + 4;
        scales.copy_from_slice(&data[sc_off..sc_off + 16]);
        // High bits for the 5th bit (32 bytes)
        let hi_off = bp + 20;
        // 5-bit quantized data: Q5_K packs as 4-bit low (64 bytes) + 1-bit high (32 bytes)
        let ql_off = bp + 52;
        for j in 0..64 {
            let ql = data[ql_off + j];
            let qh = data[hi_off + j / 2];
            for k in 0..4 {
                let idx = j * 4 + k;
                if dp + idx >= out.len() {
                    break;
                }
                let si = idx / 16;
                // Low 4 bits from ql, high bit from qh (2 values per byte)
                let nib = (ql >> (k * 2)) & 0x03;
                let hb = (qh >> (if k < 2 { k * 4 + 2 } else { (k - 2) * 4 })) & 0x01;
                let v = nib as i32 | ((hb as i32) << 2);
                let dl = d * ((scales[si] & 0x0F) as f32);
                let ml = dmin * ((scales[si] >> 4) as f32);
                out[dp + idx] = (v as f32) * dl - ml;
            }
        }
    }
}

/// Q6_K dequant: 256 elements per block
/// Block layout (210 bytes total):
///   ql[128]    lower 4 bits of each 6-bit quant (2 values per byte)
///   qh[64]     upper 2 bits of each 6-bit quant (4 values per byte)
///   scales[16] int8 scales, one per 16-element sub-block
///   d[2]       fp16 super-block scale
pub fn dequantize_q6_k(data: &[u8], out: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let bp = b * BLOCK_BYTES;
        let dp = b * BLOCK_SIZE;
        let ql_off = bp;
        let qh_off = bp + 128;
        let sc_off = bp + 192;
        let d = f16::from_le_bytes([data[bp + 208], data[bp + 209]]).to_f32();
        for i in 0..256usize {
            let out_idx = dp + i;
            if out_idx >= out.len() {
                break;
            }
            let ql_byte = data[ql_off + i / 2];
            let low4 = if i % 2 == 0 {
                ql_byte & 0x0F
            } else {
                ql_byte >> 4
            };
            let qh_byte = data[qh_off + i / 4];
            let high2 = (qh_byte >> ((i % 4) * 2)) & 0x03;
            let q = ((low4 as i32) | ((high2 as i32) << 4)) - 32;
            let scale = data[sc_off + i / 16] as i8 as f32;
            out[out_idx] = (q as f32) * scale * d;
        }
    }
}

/// Q6_K fused dot product: compute q·vec without full dequant.
/// Block: 256 elements packed into 210 bytes (ql[128], qh[64], scales[16], d[2]).
pub fn dot_q6_k_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    use half::f16;
    use wide::f32x8;
    let blocks = n / 256;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let bp = b * 210;
        let d = f16::from_le_bytes([row[bp + 208], row[bp + 209]]).to_f32();
        let vec_base = b * 256;
        for sb in 0..16 {
            let sc = row[bp + 192 + sb] as i8 as f32;
            let sd = sc * d;
            let elem_base = sb * 16;
            for i in (0..16).step_by(8) {
                let mut q = [0.0f32; 8];
                for (j, qj) in q.iter_mut().enumerate() {
                    let elem = elem_base + i + j;
                    let ql_byte = row[bp + (elem >> 1)];
                    let low4 = if (elem & 1) == 0 {
                        ql_byte & 0x0F
                    } else {
                        ql_byte >> 4
                    };
                    let qh_byte = row[bp + 128 + (elem >> 2)];
                    let high2 = (qh_byte >> ((elem & 3) << 1)) & 0x03;
                    let quant = (low4 | (high2 << 4)) as i32 - 32;
                    *qj = (quant as f32) * sd;
                }
                let vq = f32x8::new(q);
                let vv = f32x8::from(&vec[vec_base + elem_base + i..vec_base + elem_base + i + 8]);
                sum += vq.mul_add(vv, f32x8::ZERO).reduce_add();
            }
        }
    }
    sum
}

/// Q3_K dequant: 256 elements → f32
/// Block size: 110 bytes per 256 elements
pub fn dequantize_q3_k(data: &[u8], out: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 110;
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let bp = b * BLOCK_BYTES;
        let dp = b * BLOCK_SIZE;
        let d = f16::from_le_bytes([data[bp], data[bp + 1]]).to_f32();
        // 16 scales as uint8
        let mut scales = [0u8; 16];
        for i in 0..16 {
            scales[i] = data[bp + 2 + i];
        }
        // High bit mask (32 bytes)
        let hm_off = bp + 18;
        // Q3_K: 3-bit values packed as 2-bit low (64 bytes) + 1-bit high (32 bytes)
        let ql_off = bp + 50;
        for j in 0..64 {
            let ql = data[ql_off + j];
            let qh = data[hm_off + j / 2];
            for k in 0..4 {
                let idx = j * 4 + k;
                if dp + idx >= out.len() {
                    break;
                }
                let si = idx / 16;
                let shift = k * 2;
                let lv = (ql >> shift) & 0x03;
                let hv = (qh >> (if k < 2 { k * 4 + 2 } else { (k - 2) * 4 })) & 0x01;
                let mut v = (lv as i32) | ((hv as i32) << 2);
                // Sign-extend from 3 bits
                if v & 0x04 != 0 {
                    v |= !0x07;
                }
                let dl = d * (scales[si] as f32);
                out[dp + idx] = (v as f32) * dl;
            }
        }
    }
}
