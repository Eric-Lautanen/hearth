use half::f16;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const BLOCK_SIZE: usize = 128;
const BLOCK_BYTES: usize = 34;

/// Precomputed lookup: for each byte, extract 4 two-bit values → {-1, 0, 1, 2} as i8
const Q2V: [[i8; 4]; 256] = {
    let mut table = [[0i8; 4]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut j = 0;
        while j < 4 {
            let q = (b >> (j * 2)) & 3;
            table[b][j] = (q as i8) - 1;
            j += 1;
        }
        b += 1;
    }
    table
};

/// Pre-computed i16 LUT: same values as Q2V but sign-extended to i16.
/// Eliminates `_mm256_cvtepi8_epi16` in the AVX2 dot kernel.
/// 256 entries × 4 i16 × 2 bytes = 2 KB — stays hot in L1.
const Q2V_I16: [[i16; 4]; 256] = {
    let mut table = [[0i16; 4]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut j = 0;
        while j < 4 {
            let q = (b >> (j * 2)) & 3;
            table[b][j] = (q as i8 as i16) - 1;
            j += 1;
        }
        b += 1;
    }
    table
};

/// Q2V_U8: 2-bit packed values as u8 {0, 1, 2, 3} for vpdpbusd.
/// vpdpbusd computes Σ u8 * i8; true Q2V = u8 - 1, so correction subtracts Σ act.
/// 256 entries × 4 u8 = 1 KB.
const Q2V_U8: [[u8; 4]; 256] = {
    let mut table = [[0u8; 4]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut j = 0;
        while j < 4 {
            let q = (b >> (j * 2)) & 3;
            table[b][j] = q as u8;
            j += 1;
        }
        b += 1;
    }
    table
};

/// Dequantize Q2_0 (type 42): 128-element 2-bit ternary blocks
/// Format: 2 bytes FP16 scale + 32 bytes packed 2-bit codes (4 per byte)
/// 2-bit values: 0→-1, 1→0, 2→+1, 3→+2 (unused for ternary)
pub fn dequantize(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale: f32 = f16::from_le_bytes([data[bo], data[bo + 1]]).to_f32();
        let qs = &data[bo + 2..bo + 2 + 32];
        let out_base = b * BLOCK_SIZE;
        for byte_i in 0..32 {
            let vals = Q2V[qs[byte_i] as usize];
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

/// Dot product Q2_0 weight row with f32 activation vector.
pub fn dot_q2_0_f32(weight_row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale: f32 = f16::from_le_bytes([weight_row[bo], weight_row[bo + 1]]).to_f32();
        let qs = &weight_row[bo + 2..bo + 2 + 32];
        let base = b * BLOCK_SIZE;
        let mut block_acc = 0.0f32;
        for byte_i in 0..32 {
            let vals = Q2V[qs[byte_i] as usize];
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

/// Horizontal sum of 2×2 i32 in __m128i → i32
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_128i(v: __m128i) -> i32 {
    let sum = _mm_hadd_epi32(v, v);
    let sum = _mm_hadd_epi32(sum, sum);
    _mm_extract_epi32(sum, 0)
}

/// Horizontal sum of 8 × f32 in __m256 → f32
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_float_8(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let mut res = _mm_add_ps(lo, hi);
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

/// Dot product Q2_0 weight row with Q8_0 quantized activation.
/// Q2_0 has 128-element blocks (34 bytes), Q8_0 has 32-element blocks (34 bytes).
/// One Q2_0 block spans 4 Q8_0 blocks.
pub fn dot_q2_0_q8_0(weight_row: &[u8], act_data: &[u8], n: usize) -> f32 {
    unsafe { dot_q2_0_q8_0_ptr(weight_row.as_ptr(), act_data.as_ptr(), n) }
}

/// AVX-512 VNNI kernel: uses vpdpbusd (dot product of u8 × i8 → i32)
/// for the core dot product. Q2_0 values as u8 {0,1,2,3} minus the
/// activation sum correction: true_dot = vpdpbusd(w_u8, act) - Σact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
unsafe fn dot_q2_0_q8_0_vnni_avx512(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    use core::arch::x86_64::*;
    let blocks = n / BLOCK_SIZE;
    let mut acc_global = _mm256_setzero_ps();
    let ones = _mm256_set1_epi8(1);

    for b in 0..blocks {
        let w_off = b * BLOCK_BYTES;
        let w_scale = f16::from_le_bytes([*w_ptr.add(w_off), *w_ptr.add(w_off + 1)]).to_f32();
        let qs = w_ptr.add(w_off + 2);

        let mut acc_block = _mm256_setzero_ps();

        for sub in 0..4 {
            let a_off = (b * 4 + sub) * 34;
            let a_scale = f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let a_vals = a_ptr.add(a_off + 2);
            let qs_sub = qs.add(sub * 8);

            let mut w_buf: [u8; 32] = [0u8; 32];
            for j in 0..8usize {
                core::ptr::copy_nonoverlapping(
                    Q2V_U8[*qs_sub.add(j) as usize].as_ptr(),
                    w_buf.as_mut_ptr().add(j * 4),
                    4,
                );
            }
            let w = _mm256_loadu_si256(w_buf.as_ptr() as *const __m256i);
            let a = _mm256_loadu_si256(a_vals as *const __m256i);

            let zero = _mm256_setzero_si256();
            let prod = _mm256_dpbusd_epi32(zero, w, a);
            let sum_a = _mm256_dpbusd_epi32(zero, ones, a);
            let diff = _mm256_sub_epi32(prod, sum_a);

            let a_scale_ps = _mm256_set1_ps(a_scale);
            acc_block = _mm256_fmadd_ps(a_scale_ps, _mm256_cvtepi32_ps(diff), acc_block);
        }

        let w_scale_ps = _mm256_set1_ps(w_scale);
        acc_global = _mm256_fmadd_ps(w_scale_ps, acc_block, acc_global);
    }

    hsum_float_8(acc_global)
}

/// AVX2 LUT kernel: 16 elements per batch, FMA accumulation across all blocks, single hsum per row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q2_0_q8_0_lut_avx2(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut acc_global = _mm256_setzero_ps();
    for b in 0..blocks {
        let w_off = b * BLOCK_BYTES;
        let w_scale =
            _mm256_set1_ps(f16::from_le_bytes([*w_ptr.add(w_off), *w_ptr.add(w_off + 1)]).to_f32());
        let qs = w_ptr.add(w_off + 2);

        let mut acc_block = _mm256_setzero_ps();
        for sub in 0..4 {
            let a_off = (b * 4 + sub) * 34;
            let a_scale = _mm256_set1_ps(
                f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32(),
            );
            let a_vals = a_ptr.add(a_off + 2);
            let qs_sub = qs.add(sub * 8);

            let mut acc = _mm256_setzero_si256();

            {
                let a0 = _mm_loadu_si128(a_vals as *const __m128i);
                let a16_0 = _mm256_cvtepi8_epi16(a0);
                let idx0 = *qs_sub as usize;
                let idx1 = *qs_sub.add(1) as usize;
                let idx2 = *qs_sub.add(2) as usize;
                let idx3 = *qs_sub.add(3) as usize;
                let mut w_buf: [i16; 16] = [0i16; 16];
                core::ptr::copy_nonoverlapping(Q2V_I16[idx0].as_ptr(), w_buf.as_mut_ptr(), 4);
                core::ptr::copy_nonoverlapping(
                    Q2V_I16[idx1].as_ptr(),
                    w_buf.as_mut_ptr().add(4),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    Q2V_I16[idx2].as_ptr(),
                    w_buf.as_mut_ptr().add(8),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    Q2V_I16[idx3].as_ptr(),
                    w_buf.as_mut_ptr().add(12),
                    4,
                );
                let s0 = _mm256_loadu_si256(w_buf.as_ptr() as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16_0, s0));
            }

            {
                let a1 = _mm_loadu_si128(a_vals.add(16) as *const __m128i);
                let a16_1 = _mm256_cvtepi8_epi16(a1);
                let idx4 = *qs_sub.add(4) as usize;
                let idx5 = *qs_sub.add(5) as usize;
                let idx6 = *qs_sub.add(6) as usize;
                let idx7 = *qs_sub.add(7) as usize;
                let mut w_buf: [i16; 16] = [0i16; 16];
                core::ptr::copy_nonoverlapping(Q2V_I16[idx4].as_ptr(), w_buf.as_mut_ptr(), 4);
                core::ptr::copy_nonoverlapping(
                    Q2V_I16[idx5].as_ptr(),
                    w_buf.as_mut_ptr().add(4),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    Q2V_I16[idx6].as_ptr(),
                    w_buf.as_mut_ptr().add(8),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    Q2V_I16[idx7].as_ptr(),
                    w_buf.as_mut_ptr().add(12),
                    4,
                );
                let s1 = _mm256_loadu_si256(w_buf.as_ptr() as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16_1, s1));
            }

            acc_block = _mm256_fmadd_ps(a_scale, _mm256_cvtepi32_ps(acc), acc_block);
        }
        acc_global = _mm256_fmadd_ps(w_scale, acc_block, acc_global);
    }
    hsum_float_8(acc_global)
}

/// SSE4.1 LUT kernel: 8 elements per batch, 4 batches per sub-block.
#[cfg(target_arch = "x86_64")]
unsafe fn dot_q2_0_q8_0_ptr_sse41(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let w_off = b * BLOCK_BYTES;
        let w_scale: f32 = f16::from_le_bytes([*w_ptr.add(w_off), *w_ptr.add(w_off + 1)]).to_f32();
        let qs = w_ptr.add(w_off + 2);

        let mut block_acc = 0.0f32;
        for sub in 0..4 {
            let a_off = (b * 4 + sub) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let a_vals = a_ptr.add(a_off + 2);
            let qs_sub = qs.add(sub * 8);

            let mut dot: i32 = 0;
            for g in 0..4 {
                let b0 = *qs_sub.add(g * 2) as usize;
                let b1 = *qs_sub.add(g * 2 + 1) as usize;
                let a_off_inner = g * 8;
                let a8 = _mm_loadl_epi64(a_vals.add(a_off_inner) as *const __m128i);
                let a16 = _mm_cvtepi8_epi16(a8);
                let w_packed = _mm_set_epi32(
                    0,
                    0,
                    *(Q2V.as_ptr().add(b1) as *const u32) as i32,
                    *(Q2V.as_ptr().add(b0) as *const u32) as i32,
                );
                let w16 = _mm_cvtepi8_epi16(w_packed);
                let prod = _mm_madd_epi16(a16, w16);
                dot += hsum_128i(prod);
            }
            block_acc += dot as f32 * a_scale;
        }
        sum += block_acc * w_scale;
    }
    sum
}

/// Scalar fallback for Q2_0 × Q8_0 dot product.
unsafe fn dot_q2_0_q8_0_ptr_scalar(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let w_off = b * BLOCK_BYTES;
        let w_scale: f32 = f16::from_le_bytes([*w_ptr.add(w_off), *w_ptr.add(w_off + 1)]).to_f32();
        let qs = w_ptr.add(w_off + 2);

        let mut block_acc = 0.0f32;
        for sub in 0..4 {
            let a_off = (b * 4 + sub) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let a_vals = a_ptr.add(a_off + 2);
            let qs_sub = qs.add(sub * 8);

            let mut dot: i32 = 0;
            for g in 0..4 {
                let b0 = *qs_sub.add(g * 2) as usize;
                let b1 = *qs_sub.add(g * 2 + 1) as usize;
                let w0 = Q2V[b0];
                let w1 = Q2V[b1];
                let a_off_inner = g * 8;
                dot += w0[0] as i32 * (*a_vals.add(a_off_inner) as i8 as i32);
                dot += w0[1] as i32 * (*a_vals.add(a_off_inner + 1) as i8 as i32);
                dot += w0[2] as i32 * (*a_vals.add(a_off_inner + 2) as i8 as i32);
                dot += w0[3] as i32 * (*a_vals.add(a_off_inner + 3) as i8 as i32);
                dot += w1[0] as i32 * (*a_vals.add(a_off_inner + 4) as i8 as i32);
                dot += w1[1] as i32 * (*a_vals.add(a_off_inner + 5) as i8 as i32);
                dot += w1[2] as i32 * (*a_vals.add(a_off_inner + 6) as i8 as i32);
                dot += w1[3] as i32 * (*a_vals.add(a_off_inner + 7) as i8 as i32);
            }
            block_acc += dot as f32 * a_scale;
        }
        sum += block_acc * w_scale;
    }
    sum
}

/// Raw-pointer dispatch: AVX-512 VNNI > AVX2 LUT > SSE4.1 LUT > scalar.
/// # Safety
/// w_ptr valid for n/128*34 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q2_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 VNNI: vpdpbusd replaces cvtepi8+madd pair, check both avx512f and vnni
        #[cfg(target_feature = "avx512f")]
        #[cfg(target_feature = "avx512vnni")]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vnni")
        {
            return dot_q2_0_q8_0_vnni_avx512(w_ptr, a_ptr, n);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            dot_q2_0_q8_0_lut_avx2(w_ptr, a_ptr, n)
        } else if std::arch::is_x86_feature_detected!("sse4.1") {
            dot_q2_0_q8_0_ptr_sse41(w_ptr, a_ptr, n)
        } else {
            dot_q2_0_q8_0_ptr_scalar(w_ptr, a_ptr, n)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        dot_q2_0_q8_0_ptr_scalar(w_ptr, a_ptr, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q2_0_dequantize() {
        let mut data = vec![0u8; BLOCK_BYTES * 2];
        let s = f16::from_f32(2.0).to_le_bytes();
        data[0] = s[0];
        data[1] = s[1];
        // Pack 4 values per byte: each 2-bit pair
        // byte 0: q[0..3] = 0,1,2,0 → values: -1, 0, +1, -1
        // 0→00, 1→01, 2→10, 0→00 = 00_10_01_00 = 0b00100100 = 0x24
        data[2] = 0b00100100;
        let mut out = vec![0.0f32; 256];
        dequantize(&data, &mut out);
        assert!(
            (out[0] - (-2.0)).abs() < 1e-6,
            "out[0]={} should be -2",
            out[0]
        );
        assert!((out[1] - 0.0).abs() < 1e-6, "out[1]={} should be 0", out[1]);
        assert!(
            (out[2] - 2.0).abs() < 1e-6,
            "out[2]={} should be +2",
            out[2]
        );
        assert!(
            (out[3] - (-2.0)).abs() < 1e-6,
            "out[3]={} should be -2",
            out[3]
        );
    }

    #[test]
    fn test_dot_q2_0_q8_0_basic() {
        use crate::q8_0;
        // Weight: all values = +1 (q=2, bits 10)
        let mut w = vec![0u8; BLOCK_BYTES];
        let ws = f16::from_f32(2.0).to_le_bytes();
        w[0] = ws[0];
        w[1] = ws[1];
        for j in 0..32 {
            w[2 + j] = 0xAA;
        }
        // Activation: known values
        let src: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let mut act_data = Vec::new();
        q8_0::quantize(&src, &mut act_data);
        let dot = dot_q2_0_q8_0(&w, &act_data, 128);
        assert!(dot.is_finite(), "dot should be finite, got {}", dot);
    }

    #[test]
    fn test_dot_q2_0_q8_0_vs_f32() {
        use crate::q8_0;
        // Compare Q2_0×Q8_0 dot product against Q2_0×dequant(Q8_0) dot
        let mut w = vec![0u8; BLOCK_BYTES * 2];
        for b in 0..2 {
            let bo = b * BLOCK_BYTES;
            let scale = if b == 0 { 1.5f32 } else { 2.0f32 };
            let ws = f16::from_f32(scale).to_le_bytes();
            w[bo] = ws[0];
            w[bo + 1] = ws[1];
            // Mix of values: 0->-1, 1->0, 2->+1, 3->+2
            for j in 0..32 {
                w[bo + 2 + j] = (j as u8).wrapping_mul(37) ^ 0xAB;
            }
        }
        let src: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.13).sin()).collect();
        let mut act_data = Vec::new();
        q8_0::quantize(&src, &mut act_data);
        let dot_q8 = dot_q2_0_q8_0(&w, &act_data, 256);
        let dot_f32 = dot_q2_0_f32(&w, &src, 256);
        let err = (dot_q8 - dot_f32).abs();
        let max_err = dot_f32.abs().max(1.0) * 0.01 + 0.5;
        assert!(
            err < max_err,
            "dot_q8_0={} vs dot_f32={}, diff={}, max_err={}",
            dot_q8,
            dot_f32,
            err,
            max_err
        );
    }

    #[test]
    fn test_q2_0_dot_f32() {
        let mut w = vec![0u8; BLOCK_BYTES];
        let ws = f16::from_f32(1.0).to_le_bytes();
        w[0] = ws[0];
        w[1] = ws[1];
        // All values = +1 (q=2, bits 10)
        for j in 0..32 {
            w[2 + j] = 0xAA; // 10101010 → each 2-bit pair = 10 = 2
        }
        let vec: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let dot = dot_q2_0_f32(&w, &vec, 128);
        let expected: f32 = vec.iter().sum::<f32>() * 1.0; // all weights = +1
        let err = (dot - expected).abs();
        assert!(
            err < 0.01,
            "dot error: {} vs {}, diff={}",
            dot,
            expected,
            err
        );
    }
}
