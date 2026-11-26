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
    // `full` belongs to the vector path alone: on a target without one the whole
    // row goes to the scalar reference, so binding it outside the `cfg` block is
    // an unused variable there (and a warning the gate would refuse).
    #[cfg(target_arch = "x86_64")]
    {
        let full = src.len() / 32;
        if !crate::scalar_only() && crate::avx2_available() {
            // SAFETY: the vector path is entered only when the CPU reports
            // AVX2, and it reads and writes exactly the blocks it is given.
            unsafe { avx2::quantize_blocks_avx2(src, dst, full) };
            // The tail is a partial block. The scalar path is the reference
            // for it, and the vector path would read past the end of `src`.
            let rest = &src[full * 32..];
            if !rest.is_empty() {
                quantize_block_scalar(rest, &mut dst[full * 34..]);
            }
            return;
        }
    }
    quantize_into_scalar(src, dst);
}

pub fn quantize(src: &[f32], dst: &mut Vec<u8>) {
    let blocks = src.len().div_ceil(32);
    dst.reserve(blocks * 34);
    let start = dst.len();
    // Write into the spare capacity instead of growing the vector with zeros
    // first: this is the dense model's activation path, and the memset would
    // be a second pass over bytes that are overwritten a moment later.
    //
    // SAFETY: `reserve` guarantees room for `blocks * 34` more bytes, and
    // `quantize_into` writes every one of them (it is given exactly that
    // many) before `set_len` grows the vector over them. `u8` has no invalid
    // bit patterns, so the bytes are fully initialized once written.
    unsafe {
        let spare = dst.spare_capacity_mut();
        let out = std::slice::from_raw_parts_mut(spare.as_mut_ptr() as *mut u8, blocks * 34);
        quantize_into(src, out);
        dst.set_len(start + blocks * 34);
    }
}

/// The reference quantization: one block at a time, on the CPU.
///
/// The block math is: `max_abs` over the block, `scale = max_abs / 127` (or
/// `1.0` for an all-zero block, so the stored fp16 is never a zero scale),
/// and `q = (v / scale).round().clamp(-128, 127)` per value, with the
/// division against the **f32** scale - the fp16 is only what gets stored.
fn quantize_into_scalar(src: &[f32], dst: &mut [u8]) {
    let blocks = src.len().div_ceil(32);
    for b in 0..blocks {
        let start = b * 32;
        let end = (start + 32).min(src.len());
        quantize_block_scalar(&src[start..end], &mut dst[b * 34..b * 34 + 34]);
    }
}

/// The reference quantization of one block of at most 32 values.
///
/// The vector path has to reproduce this byte for byte, so the details that
/// look incidental are the contract:
///
/// * `max_abs` ignores a NaN (`v > max_abs` is false for it) and the scale is
///   `1.0` when the block is all zeros, so the stored fp16 is `1.0` and the
///   division never sees a zero scale.
/// * The value is rounded **half away from zero** (`f32::round`), then
///   clamped to `[-128, 127]`, then truncated to `i8`. A NaN casts to `0`.
fn quantize_block_scalar(src: &[f32], dst: &mut [u8]) {
    debug_assert!(src.len() <= 32);
    debug_assert!(dst.len() >= 34);
    let mut max_abs: f32 = 0.0;
    for &v in src {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let scale_bytes = f16::from_f32(scale).to_le_bytes();
    dst[0] = scale_bytes[0];
    dst[1] = scale_bytes[1];
    for (i, &v) in src.iter().enumerate() {
        let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
        dst[2 + i] = q as u8;
    }
    for i in src.len()..32 {
        dst[2 + i] = 0;
    }
}

/// The x86_64 quantization of whole 32-element blocks.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::*;

    use half::f16;

    use super::quantize_block_scalar;

    /// The horizontal maximum of eight lanes.
    ///
    /// The max never sees a NaN (the block loop keeps NaNs out of it), so the
    /// operand order does not matter here.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hmax_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let m = _mm_max_ps(lo, hi);
        let m = _mm_max_ps(m, _mm_movehdup_ps(m));
        let m = _mm_max_ss(m, _mm_movehl_ps(m, m));
        _mm_cvtss_f32(m)
    }

    /// Quantize eight values against one scale, to eight i32.
    ///
    /// * The division is the same correctly-rounded f32 division the scalar
    ///   path does.
    /// * Clamping *before* rounding gives the same integer as rounding and
    ///   then clamping (the bounds are integers, so a value outside them
    ///   lands on the same bound either way), and it keeps `|x| <= 128`.
    /// * Rounding half away from zero, the way `f32::round` does, is
    ///   `x + copysign(0.5, x)` truncated toward zero. That step cannot cross
    ///   an integer: 0.5 is a multiple of x's ulp, so `x + 0.5` is exact
    ///   while it stays in x's own binade, and when it leaves that binade the
    ///   only integer left in range is the boundary, which the sum cannot
    ///   pass. So the truncation lands on the same integer `f32::round` does.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_vec(v: __m256, s: __m256, sign_bit: __m256) -> __m256i {
        let x = _mm256_div_ps(v, s);
        let x = _mm256_min_ps(
            _mm256_max_ps(x, _mm256_set1_ps(-128.0)),
            _mm256_set1_ps(127.0),
        );
        let half = _mm256_or_ps(_mm256_set1_ps(0.5), _mm256_and_ps(x, sign_bit));
        let y = _mm256_add_ps(x, half);
        _mm256_cvttps_epi32(y)
    }

    /// Quantize `blocks` whole 32-element blocks of `src` into `dst`.
    ///
    /// # Safety
    /// The CPU must support AVX2 (see [`crate::avx2_available`]).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn quantize_blocks_avx2(src: &[f32], dst: &mut [u8], blocks: usize) {
        let sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
        let big = _mm256_set1_ps(f32::MAX);

        for b in 0..blocks {
            let base = b * 32;
            let sp = src.as_ptr().add(base);
            let v0 = _mm256_loadu_ps(sp);
            let v1 = _mm256_loadu_ps(sp.add(8));
            let v2 = _mm256_loadu_ps(sp.add(16));
            let v3 = _mm256_loadu_ps(sp.add(24));

            let a0 = _mm256_andnot_ps(sign_bit, v0);
            let a1 = _mm256_andnot_ps(sign_bit, v1);
            let a2 = _mm256_andnot_ps(sign_bit, v2);
            let a3 = _mm256_andnot_ps(sign_bit, v3);

            // The block maximum. MAXPS returns its second operand when the
            // first is a NaN, so the data is always the first operand: a NaN
            // in the row is ignored by the max, exactly as the scalar
            // `if v > max_abs` ignores it.
            let mut m = _mm256_setzero_ps();
            m = _mm256_max_ps(a0, m);
            m = _mm256_max_ps(a1, m);
            m = _mm256_max_ps(a2, m);
            m = _mm256_max_ps(a3, m);
            let max_abs = hmax_ps(m);

            // A non-finite value has no vector answer that matches the
            // scalar one (a NaN casts to 0, an infinite scale divides
            // everything to 0), so such a block goes back to the scalar path.
            let finite = _mm256_and_ps(
                _mm256_and_ps(
                    _mm256_cmp_ps(a0, big, _CMP_LE_OQ),
                    _mm256_cmp_ps(a1, big, _CMP_LE_OQ),
                ),
                _mm256_and_ps(
                    _mm256_cmp_ps(a2, big, _CMP_LE_OQ),
                    _mm256_cmp_ps(a3, big, _CMP_LE_OQ),
                ),
            );
            if _mm256_movemask_ps(finite) != 0xFF {
                quantize_block_scalar(&src[base..base + 32], &mut dst[b * 34..b * 34 + 34]);
                continue;
            }

            // `scale = max_abs / 127`, or 1.0 for an all-zero block. The
            // blend keeps the scalar's `max_abs == 0.0 -> 1.0` rule, which
            // is also what keeps the stored fp16 out of being a zero scale.
            let m1 = _mm_set_ss(max_abs);
            let div = _mm_div_ss(m1, _mm_set_ss(127.0));
            let zero = _mm_cmpeq_ss(m1, _mm_setzero_ps());
            let scale = _mm_cvtss_f32(_mm_blendv_ps(div, _mm_set_ss(1.0), zero));
            // A scale that underflowed to zero would make 0/0 a NaN, which
            // the scalar casts to 0 and the vector clamp cannot reproduce.
            if scale == 0.0 {
                quantize_block_scalar(&src[base..base + 32], &mut dst[b * 34..b * 34 + 34]);
                continue;
            }

            let s = _mm256_set1_ps(scale);
            let q0 = quantize_vec(v0, s, sign_bit);
            let q1 = quantize_vec(v1, s, sign_bit);
            let q2 = quantize_vec(v2, s, sign_bit);
            let q3 = quantize_vec(v3, s, sign_bit);

            // Narrow the four vectors of eight i32 into 32 bytes. The
            // packs/unpack pair keeps the element order: the first 16 bytes
            // are elements 0..16, the next 16 are elements 16..32.
            let t0 = _mm256_packs_epi32(q0, q1);
            let t1 = _mm256_packs_epi32(q2, q3);
            let a = _mm256_castsi256_si128(t0);
            let b2 = _mm256_extracti128_si256(t0, 1);
            let c = _mm256_castsi256_si128(t1);
            let d = _mm256_extracti128_si256(t1, 1);
            let p01 = _mm_packs_epi16(_mm_unpacklo_epi64(a, b2), _mm_unpackhi_epi64(a, b2));
            let p23 = _mm_packs_epi16(_mm_unpacklo_epi64(c, d), _mm_unpackhi_epi64(c, d));
            let dp = dst.as_mut_ptr().add(b * 34 + 2);
            _mm_storeu_si128(dp as *mut __m128i, p01);
            _mm_storeu_si128(dp.add(16) as *mut __m128i, p23);

            let scale_bytes = f16::from_f32(scale).to_le_bytes();
            let ds = dst.as_mut_ptr().add(b * 34);
            *ds = scale_bytes[0];
            *ds.add(1) = scale_bytes[1];
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

    /// A deterministic pseudo-random row, so the tests do not depend on a
    /// random number generator's implementation.
    ///
    /// Only the vector paths' equality tests use it, and those exist on x86_64
    /// alone; on any other target it would be dead code.
    #[cfg(target_arch = "x86_64")]
    fn pseudo_random(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    /// The vector path must produce the same bytes as the scalar reference,
    /// for every shape of row the callers use - including the rows that are
    /// not a whole number of blocks.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_bit_for_bit() {
        if !crate::avx2_available() {
            return;
        }
        for n in [1usize, 31, 32, 33, 64, 100, 1024, 2048, 4097] {
            let src = pseudo_random(n, 0x9e37_79b9 ^ n as u32);
            let blocks = src.len().div_ceil(32);
            let mut scalar = vec![0u8; blocks * 34];
            quantize_into_scalar(&src, &mut scalar);
            let mut vector = vec![0u8; blocks * 34];
            quantize_into(&src, &mut vector);
            assert_eq!(scalar, vector, "n = {n}");
        }
    }

    /// The cases where the vector path could disagree with the scalar one:
    /// an all-zero block, exact ties, the max/127 boundary, a scale that
    /// underflows, a denormal scale, a NaN and an infinity.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_the_hard_cases() {
        if !crate::avx2_available() {
            return;
        }
        let tiny = f32::from_bits(1); // the smallest denormal
        let small = f32::MIN_POSITIVE; // the smallest normal
        let cases: Vec<Vec<f32>> = vec![
            // all-zero block
            vec![0.0; 32],
            // exact ties, and the ends of the clamp
            vec![
                0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127.0, -127.0, 128.0, -128.0,
            ],
            // the max/127 boundary: 1.0 / (1.0 / 127) is a hair over 127
            vec![1.0; 32],
            // a scale that underflows to zero
            vec![tiny, 0.0, tiny, -tiny],
            // a denormal, but non-zero, scale
            vec![small, 0.0, small, -small],
            // a NaN in the block
            vec![f32::NAN, 1.0, -1.0, 2.0],
            // an infinity in the block
            vec![f32::INFINITY, 1.0, -1.0, 2.0],
            // signed zeros
            vec![-0.0, 0.0, 1e-45, -1e-45],
        ];
        for case in cases {
            let src = case.clone();
            let blocks = src.len().div_ceil(32);
            let mut scalar = vec![0u8; blocks * 34];
            quantize_into_scalar(&src, &mut scalar);
            let mut vector = vec![0u8; blocks * 34];
            quantize_into(&src, &mut vector);
            assert_eq!(scalar, vector, "case {case:?}");
        }
    }

    /// An all-zero block stores the scale 1.0, not a zero scale: the stored
    /// fp16 is what the dot kernel multiplies by, so it is part of the
    /// contract and not an internal detail.
    #[test]
    fn zero_block_stores_a_unit_scale() {
        let src = vec![0.0f32; 32];
        let mut dst = vec![0u8; 34];
        quantize_into(&src, &mut dst);
        let scale = f16::from_le_bytes([dst[0], dst[1]]);
        assert_eq!(scale.to_f32(), 1.0);
        assert!(dst[2..].iter().all(|&b| b == 0));
    }

    /// A partial last block is padded with zeros and quantized as if the
    /// missing values were zero.
    #[test]
    fn partial_block_is_padded_with_zeros() {
        let src = vec![1.0f32; 20];
        let mut dst = vec![0u8; 34];
        quantize_into(&src, &mut dst);
        let scale = f16::from_le_bytes([dst[0], dst[1]]);
        assert_eq!(scale.to_f32(), f16::from_f32(1.0 / 127.0).to_f32());
        assert!(dst[2..22].iter().all(|&b| b == 127));
        assert!(dst[22..34].iter().all(|&b| b == 0));
    }
}
