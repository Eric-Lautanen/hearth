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
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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
///
/// The attribute states what the body already needs — 256-bit code — instead of inheriting it from
/// the build's baseline. Every caller enables `avx2` or `avx512f`. It did not change the measured
/// `x86-64-v2` numbers; the baseline's real cost is measured in `docs/BUG_TRACKER.md`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
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
///
/// # Safety
/// `w_ptr` must be valid for `n/128*34` bytes and `a_ptr` for `n/32*34` bytes;
/// the caller must confirm AVX-512F, AVX-512VNNI and AVX-512VL are present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
// The 256-bit EVEX VNNI intrinsic is `avx512vnni` *and* `avx512vl`; without VL
// declared, LLVM cannot inline it and every `vpdpbusd` becomes a call.
#[target_feature(enable = "avx512vl")]
pub unsafe fn dot_q2_0_q8_0_vnni_avx512(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    use core::arch::x86_64::*;
    let blocks = n / BLOCK_SIZE;
    let mut acc_global = _mm256_setzero_ps();
    let ones = _mm256_set1_epi8(1);

    for b in 0..blocks {
        let w_off = b * BLOCK_BYTES;
        let w_scale = crate::f16_bits_to_f32(u16::from_le_bytes([
            *w_ptr.add(w_off),
            *w_ptr.add(w_off + 1),
        ]));
        let qs = w_ptr.add(w_off + 2);

        let mut acc_block = _mm256_setzero_ps();

        for sub in 0..4 {
            let a_off = (b * 4 + sub) * 34;
            let a_scale = crate::f16_bits_to_f32(u16::from_le_bytes([
                *a_ptr.add(a_off),
                *a_ptr.add(a_off + 1),
            ]));
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

/// AVX-512 512-bit VNNI kernel: processes 2 Q8_0 sub-blocks (64 elements) per iteration.
/// Uses _mm512_dpbusd_epi32 to compute the u8×i8→i32 dot product for 64 elements at once.
/// The 512-bit result is split into two 256-bit halves with independent Q8_0 scale factors.
/// Available on Zen 4 (double-pumped 256-bit → front-end μop reduction).
///
/// # Safety
/// `w_ptr` must be valid for `n/128*34` bytes and `a_ptr` for `n/32*34` bytes;
/// the caller must confirm AVX-512F, AVX-512VNNI and AVX-512DQ are present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512dq")]
pub unsafe fn dot_q2_0_q8_0_vnni_avx512_2sub(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    use core::arch::x86_64::*;
    let blocks = n / BLOCK_SIZE;
    let mut acc_global = _mm256_setzero_ps();
    let ones_8 = _mm512_set1_epi8(1);

    for b in 0..blocks {
        let w_off = b * BLOCK_BYTES;
        let w_scale = crate::f16_bits_to_f32(u16::from_le_bytes([
            *w_ptr.add(w_off),
            *w_ptr.add(w_off + 1),
        ]));
        let qs = w_ptr.add(w_off + 2);

        let mut acc_block_lo = _mm256_setzero_ps();
        let mut acc_block_hi = _mm256_setzero_ps();

        for pair in 0..2 {
            let sub0 = pair * 2;
            let sub1 = pair * 2 + 1;

            let a_off_0 = (b * 4 + sub0) * 34;
            let a_off_1 = (b * 4 + sub1) * 34;
            let a_scale_0 = crate::f16_bits_to_f32(u16::from_le_bytes([
                *a_ptr.add(a_off_0),
                *a_ptr.add(a_off_0 + 1),
            ]));
            let a_scale_1 = crate::f16_bits_to_f32(u16::from_le_bytes([
                *a_ptr.add(a_off_1),
                *a_ptr.add(a_off_1 + 1),
            ]));
            let a_vals_0 = a_ptr.add(a_off_0 + 2);
            let a_vals_1 = a_ptr.add(a_off_1 + 2);

            let qs_0 = qs.add(sub0 * 8);
            let qs_1 = qs.add(sub1 * 8);

            let mut w_buf: [u8; 64] = [0u8; 64];
            for j in 0..8usize {
                core::ptr::copy_nonoverlapping(
                    Q2V_U8[*qs_0.add(j) as usize].as_ptr(),
                    w_buf.as_mut_ptr().add(j * 4),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    Q2V_U8[*qs_1.add(j) as usize].as_ptr(),
                    w_buf.as_mut_ptr().add(32 + j * 4),
                    4,
                );
            }

            let mut a_buf: [u8; 64] = [0u8; 64];
            core::ptr::copy_nonoverlapping(a_vals_0, a_buf.as_mut_ptr(), 32);
            core::ptr::copy_nonoverlapping(a_vals_1, a_buf.as_mut_ptr().add(32), 32);

            let w = _mm512_loadu_si512(w_buf.as_ptr() as *const __m512i);
            let a = _mm512_loadu_si512(a_buf.as_ptr() as *const __m512i);

            let zero = _mm512_setzero_si512();
            let prod = _mm512_dpbusd_epi32(zero, w, a);
            let sum_a = _mm512_dpbusd_epi32(zero, ones_8, a);
            let diff = _mm512_sub_epi32(prod, sum_a);

            let diff_lo_ps = _mm256_cvtepi32_ps(_mm512_castsi512_si256(diff));
            let diff_hi_ps = _mm256_cvtepi32_ps(_mm512_extracti64x4_epi64::<1>(diff));

            let a_scale_0_ps = _mm256_set1_ps(a_scale_0);
            let a_scale_1_ps = _mm256_set1_ps(a_scale_1);
            acc_block_lo = _mm256_fmadd_ps(a_scale_0_ps, diff_lo_ps, acc_block_lo);
            acc_block_hi = _mm256_fmadd_ps(a_scale_1_ps, diff_hi_ps, acc_block_hi);
        }

        let acc_block = _mm256_add_ps(acc_block_lo, acc_block_hi);
        let w_scale_ps = _mm256_set1_ps(w_scale);
        acc_global = _mm256_fmadd_ps(w_scale_ps, acc_block, acc_global);
    }

    hsum_float_8(acc_global)
}

/// AVX2 LUT kernel: 16 elements per batch, FMA accumulation across all blocks, single hsum per row.
///
/// # Safety
/// `w_ptr` must be valid for `n/128*34` bytes and `a_ptr` for `n/32*34` bytes;
/// the caller must confirm AVX2, FMA and F16C are present — see [`avx2_lut_available`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
pub unsafe fn dot_q2_0_q8_0_lut_avx2(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut acc_global = _mm256_setzero_ps();
    for b in 0..blocks {
        let w_off = b * BLOCK_BYTES;
        let w_scale = _mm256_set1_ps(crate::f16_bits_to_f32(u16::from_le_bytes([
            *w_ptr.add(w_off),
            *w_ptr.add(w_off + 1),
        ])));
        let qs = w_ptr.add(w_off + 2);

        let mut acc_block = _mm256_setzero_ps();
        for sub in 0..4 {
            let a_off = (b * 4 + sub) * 34;
            let a_scale = _mm256_set1_ps(crate::f16_bits_to_f32(u16::from_le_bytes([
                *a_ptr.add(a_off),
                *a_ptr.add(a_off + 1),
            ])));
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

/// Paired-token AVX2 LUT kernel: one Q2_0 weight row against two Q8_0 activation
/// rows at once.
///
/// Prefill re-runs the Q2V_I16 gather (8 table entries, 16 elements each) for every
/// prompt token, though it depends only on the weights. Expanding it once and
/// applying it to both activations removes that work from the per-token path and
/// halves the weight bytes pulled per token.
///
/// Accumulation order matches [`dot_q2_0_q8_0_lut_avx2`], so each token is bit
/// identical to the single-token kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
unsafe fn dot_q2_0_q8_0x2_lut_avx2(
    w_ptr: *const u8,
    a0_ptr: *const u8,
    a1_ptr: *const u8,
    n: usize,
) -> (f32, f32) {
    let blocks = n / BLOCK_SIZE;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    for b in 0..blocks {
        let w_off = b * BLOCK_BYTES;
        let w_scale = _mm256_set1_ps(crate::f16_bits_to_f32(u16::from_le_bytes([
            *w_ptr.add(w_off),
            *w_ptr.add(w_off + 1),
        ])));
        let qs = w_ptr.add(w_off + 2);

        let mut acc_block0 = _mm256_setzero_ps();
        let mut acc_block1 = _mm256_setzero_ps();
        for sub in 0..4 {
            let a_off = (b * 4 + sub) * 34;
            let a0_scale = _mm256_set1_ps(crate::f16_bits_to_f32(u16::from_le_bytes([
                *a0_ptr.add(a_off),
                *a0_ptr.add(a_off + 1),
            ])));
            let a1_scale = _mm256_set1_ps(crate::f16_bits_to_f32(u16::from_le_bytes([
                *a1_ptr.add(a_off),
                *a1_ptr.add(a_off + 1),
            ])));
            let a0_vals = a0_ptr.add(a_off + 2);
            let a1_vals = a1_ptr.add(a_off + 2);
            let qs_sub = qs.add(sub * 8);

            let mut w_buf: [i16; 32] = [0i16; 32];
            for g in 0..8 {
                let idx = *qs_sub.add(g) as usize;
                core::ptr::copy_nonoverlapping(
                    Q2V_I16[idx].as_ptr(),
                    w_buf.as_mut_ptr().add(g * 4),
                    4,
                );
            }
            let s0 = _mm256_loadu_si256(w_buf.as_ptr() as *const __m256i);
            let s1 = _mm256_loadu_si256(w_buf.as_ptr().add(16) as *const __m256i);

            let x0_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(a0_vals as *const __m128i));
            let x0_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(a0_vals.add(16) as *const __m128i));
            let x1_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(a1_vals as *const __m128i));
            let x1_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(a1_vals.add(16) as *const __m128i));

            let mut acc =
                _mm256_add_epi32(_mm256_madd_epi16(x0_lo, s0), _mm256_madd_epi16(x0_hi, s1));
            acc_block0 = _mm256_fmadd_ps(a0_scale, _mm256_cvtepi32_ps(acc), acc_block0);

            acc = _mm256_add_epi32(_mm256_madd_epi16(x1_lo, s0), _mm256_madd_epi16(x1_hi, s1));
            acc_block1 = _mm256_fmadd_ps(a1_scale, _mm256_cvtepi32_ps(acc), acc_block1);
        }
        acc0 = _mm256_fmadd_ps(w_scale, acc_block0, acc0);
        acc1 = _mm256_fmadd_ps(w_scale, acc_block1, acc1);
    }
    (hsum_float_8(acc0), hsum_float_8(acc1))
}

/// Paired-token dot product: one Q2_0 weight row against two Q8_0 activation rows.
/// Falls back to two single-token dots on CPUs without AVX2.
/// # Safety
/// w_ptr valid for n/128*34 bytes; a0_ptr and a1_ptr each valid for n/32*34 bytes.
/// n must be a multiple of 128.
pub unsafe fn dot_q2_0_q8_0x2_ptr(
    w_ptr: *const u8,
    a0_ptr: *const u8,
    a1_ptr: *const u8,
    n: usize,
) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_lut_available() {
            return dot_q2_0_q8_0x2_lut_avx2(w_ptr, a0_ptr, a1_ptr, n);
        }
    }
    (
        dot_q2_0_q8_0_ptr(w_ptr, a0_ptr, n),
        dot_q2_0_q8_0_ptr(w_ptr, a1_ptr, n),
    )
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

/// Cached probe for the two AVX2 LUT kernels, whose bodies use FMA and F16C intrinsics.
///
/// All three features are named because all three are executed by the bodies, and a
/// `#[target_feature]` declaration is what decides whether an intrinsic can be inlined. At
/// `x86-64-v2` the same source called `_mm256_fmadd_ps` and `f16_to_f32_x86_f16c` out of line
/// (`dot_q2_0_q8_0_lut_avx2`: 256 instructions against 167) and read 0.87 GB/s against 2.64
/// at `x86-64-v3`. Dispatch *and* [`kernel_name`] read this one function so the two cannot
/// drift. See `docs/BUG_TRACKER.md`.
#[cfg(target_arch = "x86_64")]
fn avx2_lut_available() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(2); // 2 = unknown
    let v = STATE.load(Ordering::Relaxed);
    if v < 2 {
        return v == 1;
    }
    let ok = std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c");
    STATE.store(if ok { 1 } else { 0 }, Ordering::Relaxed);
    ok
}

/// Cached AVX-512 F + VNNI + VL probe, the features `vpdpbusd` needs.
///
/// The probe is a function rather than an inline condition precisely so that the dispatch
/// and [`kernel_name`] cannot drift: both read this. The crate's own history is the
/// argument — a gate that answered "no" on every machine here cost 34% on PTQ1_0 and was
/// invisible in everything a user or a benchmark printed.
#[cfg(target_arch = "x86_64")]
fn avx512_vnni_available() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(2); // 2 = unknown
    let v = STATE.load(Ordering::Relaxed);
    if v < 2 {
        return v == 1;
    }
    // VL is not optional here: the kernel uses the 256-bit EVEX form, and naming only
    // F + VNNI let the intrinsic fall out of line (see the kernel's own attribute).
    let ok = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512vnni")
        && std::arch::is_x86_feature_detected!("avx512vl");
    STATE.store(if ok { 1 } else { 0 }, Ordering::Relaxed);
    ok
}

/// Cached probe for the 512-bit two-sub-block kernel, which additionally needs AVX-512DQ
/// for `vextracti64x4`. Checked first by both callers, so the wider kernel wins where it
/// can run at all.
#[cfg(target_arch = "x86_64")]
fn avx512_vnni_2sub_available() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(2);
    let v = STATE.load(Ordering::Relaxed);
    if v < 2 {
        return v == 1;
    }
    let ok = avx512_vnni_available() && std::arch::is_x86_feature_detected!("avx512dq");
    STATE.store(if ok { 1 } else { 0 }, Ordering::Relaxed);
    ok
}

/// Raw-pointer dispatch: AVX-512 VNNI > AVX2 LUT > SSE4.1 LUT > scalar.
///
/// Every gate here is a **runtime** probe. There were `#[cfg(target_feature = …)]`
/// attributes on the AVX-512 arms as well, and they were worse than redundant: they made a
/// kernel's *existence* a property of the machine the crate was built on rather than the
/// machine it runs on, so a binary built anywhere without AVX-512 VNNI could not select the
/// fastest Q2_0 kernel even on a CPU that has it. The kernels themselves carry
/// `#[target_feature]`, which is what lets them be compiled unconditionally and called
/// behind this check — the arrangement `ptq1_0` has always used.
/// # Safety
/// w_ptr valid for n/128*34 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q2_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 512-bit VNNI: two sub-blocks (64 elements) per iteration, which needs
        // `avx512dq` for `vextracti64x4` on top of AVX-512F + VNNI. Allowed on Zen 4:
        // `vpdpbusd` cuts the front-end work enough to pay for the half-width datapath —
        // over a 4B PQ2_0 `blk.0` row set (`benchk <model> pq2_0`) the aggregate led the
        // 256-bit form by 3.10 vs 3.09 GB/s and the AVX2 LUT kernel by 3.09 vs 2.91 on a
        // warm machine (2026-06). Re-measured cold on the same box it leads the same way
        // and by more (1.59 vs 1.30 GB/s vs `avx2_lut`, 1.62 dispatched), which is why the
        // ordering here is not in doubt even though the absolute figures move with clock
        // ramp-up.
        if avx512_vnni_2sub_available() {
            return dot_q2_0_q8_0_vnni_avx512_2sub(w_ptr, a_ptr, n);
        }
        // AVX-512 256-bit VNNI: `vpdpbusd` replaces the cvtepi8+madd pair. Gated on the
        // VNNI features only, for the reason above: on Zen 4 it beats the AVX2 LUT kernel
        // despite the same double-pumped datapath.
        if avx512_vnni_available() {
            return dot_q2_0_q8_0_vnni_avx512(w_ptr, a_ptr, n);
        }
        if avx2_lut_available() {
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

/// Name of the kernel [`dot_q2_0_q8_0_ptr`] selects on this machine.
///
/// Reports the choice by *calling the same probes in the same order*, so it cannot
/// disagree with the dispatch — a restated predicate is a predicate that can drift, and the
/// previous version went further and wrapped the restatement in `cfg(target_feature)`,
/// which made the report a function of the build machine too. Note the VNNI entries are
/// deliberately *not* gated on [`crate::avx512_full_width`]: `vpdpbusd` cuts enough
/// front-end work to beat the AVX2 LUT kernel even on a double-pumped Zen 4 datapath. See
/// [`crate::selected_kernel`].
pub fn kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_vnni_2sub_available() {
            return "vnni512x2";
        }
        if avx512_vnni_available() {
            return "vnni256";
        }
        if avx2_lut_available() {
            return "avx2_lut";
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return "sse41_lut";
        }
    }
    "scalar"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The report must name the arm the raw features imply, on whatever machine runs this.
    ///
    /// The predicates are restated here on purpose: this *is* the oracle the dispatch is
    /// compared against, and the version of it that shipped before sat behind
    /// `cfg(target_feature)`, so it could answer `avx2_lut` on a machine with AVX-512 VNNI
    /// whenever the crate had been built somewhere that lacked it. Restating the rule in a
    /// test catches that; restating it in the code is what caused it.
    #[test]
    fn the_reported_kernel_is_the_one_the_features_imply() {
        #[cfg(target_arch = "x86_64")]
        {
            let f = std::arch::is_x86_feature_detected!("avx512f");
            let vnni = std::arch::is_x86_feature_detected!("avx512vnni");
            let dq = std::arch::is_x86_feature_detected!("avx512dq");
            let expected = if f && vnni && dq {
                "vnni512x2"
            } else if f && vnni {
                "vnni256"
            } else if avx2_lut_available() {
                "avx2_lut"
            } else if std::arch::is_x86_feature_detected!("sse4.1") {
                "sse41_lut"
            } else {
                "scalar"
            };
            assert_eq!(kernel_name(), expected);
        }
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(kernel_name(), "scalar");
    }

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

    /// Microbenchmark: compare 256-bit vs 512-bit AVX-512 VNNI kernels
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn bench_vnni_256_vs_512() {
        if !std::arch::is_x86_feature_detected!("avx512f")
            || !std::arch::is_x86_feature_detected!("avx512vnni")
        {
            eprintln!("Skipping VNNI benchmark (no AVX-512 VNNI)");
            return;
        }
        let has_dq = std::arch::is_x86_feature_detected!("avx512dq");
        let dims = [2048usize, 4096, 6144, 9728];
        let iters = 10_000;

        for &n in &dims {
            let n_blocks = n / BLOCK_SIZE;
            let mut w = vec![0u8; n_blocks * BLOCK_BYTES];
            let mut a = vec![0u8; n_blocks * 4 * 34];

            for b in 0..n_blocks {
                let bo = b * BLOCK_BYTES;
                let scale = ((b as f32 + 1.0) * 0.1).sin().abs() * 0.5 + 0.01;
                let s = half::f16::from_f32(scale).to_le_bytes();
                w[bo] = s[0];
                w[bo + 1] = s[1];
                for j in 0..32 {
                    w[bo + 2 + j] = ((b * 32 + j).wrapping_mul(37) ^ 0xAB) as u8;
                }
            }
            for b in 0..n_blocks * 4 {
                let bo = b * 34;
                let scale = ((b as f32 + 1.0) * 0.05).cos().abs() * 0.5 + 0.01;
                let s = half::f16::from_f32(scale).to_le_bytes();
                a[bo] = s[0];
                a[bo + 1] = s[1];
                for j in 0..32 {
                    a[bo + 2 + j] = ((b * 32 + j).wrapping_mul(53) ^ 0xCD) as u8;
                }
            }

            // Warmup
            unsafe {
                let _ = dot_q2_0_q8_0_vnni_avx512(w.as_ptr(), a.as_ptr(), n);
            }

            let t0 = std::time::Instant::now();
            let mut sum = 0.0f32;
            for _ in 0..iters {
                sum += unsafe { dot_q2_0_q8_0_vnni_avx512(w.as_ptr(), a.as_ptr(), n) };
            }
            let ns_256 = t0.elapsed().as_nanos() as f64 / iters as f64;

            if has_dq {
                unsafe {
                    let _ = dot_q2_0_q8_0_vnni_avx512_2sub(w.as_ptr(), a.as_ptr(), n);
                }
                let t0 = std::time::Instant::now();
                let mut sum512 = 0.0f32;
                for _ in 0..iters {
                    sum512 += unsafe { dot_q2_0_q8_0_vnni_avx512_2sub(w.as_ptr(), a.as_ptr(), n) };
                }
                let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
                eprintln!(
                    "n={:5}: 256VNNI={:7.1}ns  512VNNI={:7.1}ns  {:+.1}% (sum256={}, sum512={})",
                    n,
                    ns_256,
                    ns,
                    (ns / ns_256 - 1.0) * 100.0,
                    sum,
                    sum512
                );
            } else {
                eprintln!(
                    "n={:5}: 256VNNI={:7.1}ns  512VNNI=N/A (no avx512dq)",
                    n, ns_256
                );
            }
            // Compare per-iteration results
            if has_dq {
                let ref_256 = unsafe { dot_q2_0_q8_0_vnni_avx512(w.as_ptr(), a.as_ptr(), n) };
                let ref_512 = unsafe { dot_q2_0_q8_0_vnni_avx512_2sub(w.as_ptr(), a.as_ptr(), n) };
                let err = (ref_256 - ref_512).abs();
                assert!(
                    err < 0.01 || err / ref_256.abs().max(1.0) < 0.01,
                    "Results differ: 256={} 512={} err={}",
                    ref_256,
                    ref_512,
                    err
                );
            }
        }
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

    /// The paired prefill kernel must reproduce the single-token kernel exactly —
    /// it only shares weight-side work, so any difference means the sharing leaked
    /// into the per-token math.
    #[test]
    fn test_dot_q2_0_q8_0x2_matches_single() {
        for &n in &[128usize, 256, 2048, 6144] {
            let n_blocks = n / BLOCK_SIZE;
            let mut w = vec![0u8; n_blocks * BLOCK_BYTES];
            let mut a0 = vec![0u8; n_blocks * 4 * 34];
            let mut a1 = vec![0u8; n_blocks * 4 * 34];

            for b in 0..n_blocks {
                let bo = b * BLOCK_BYTES;
                let s = f16::from_f32(((b % 7) as f32 + 0.5) * 0.25).to_le_bytes();
                w[bo] = s[0];
                w[bo + 1] = s[1];
                for j in 0..32 {
                    w[bo + 2 + j] = ((b * 32 + j).wrapping_mul(37) ^ 0xAB) as u8;
                }
            }
            for (act, salt) in [(a0.as_mut_slice(), 53u32), (a1.as_mut_slice(), 97u32)] {
                for b in 0..n_blocks * 4 {
                    let bo = b * 34;
                    let s = f16::from_f32(((b % 5) as f32 + 0.5) * 0.125).to_le_bytes();
                    act[bo] = s[0];
                    act[bo + 1] = s[1];
                    for j in 0..32 {
                        act[bo + 2 + j] = (b as u32 * salt).wrapping_mul(2654435761) as u8;
                    }
                }
            }

            let (got0, got1) =
                unsafe { dot_q2_0_q8_0x2_ptr(w.as_ptr(), a0.as_ptr(), a1.as_ptr(), n) };
            let want0 = unsafe { dot_q2_0_q8_0_ptr(w.as_ptr(), a0.as_ptr(), n) };
            let want1 = unsafe { dot_q2_0_q8_0_ptr(w.as_ptr(), a1.as_ptr(), n) };
            assert_eq!(
                got0.to_bits(),
                want0.to_bits(),
                "token0 mismatch at n={}",
                n
            );
            assert_eq!(
                got1.to_bits(),
                want1.to_bits(),
                "token1 mismatch at n={}",
                n
            );
            assert!(want0.abs() > 1e-6 && want1.abs() > 1e-6, "n={} fixture", n);
        }
    }
}
