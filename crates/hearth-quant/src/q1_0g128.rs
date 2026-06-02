use half::f16;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Precomputed lookup: for each byte, extract 8 sign values {1, -1} based on bits
const Q1V: [[i8; 8]; 256] = {
    let mut table = [[0i8; 8]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut j = 0;
        while j < 8 {
            let bit = (b >> j) & 1;
            table[b][j] = if bit != 0 { 1 } else { -1 };
            j += 1;
        }
        b += 1;
    }
    table
};

/// Dequantize PRISM Q1_0 (type 42): 32-element ternary {-1,+1} blocks
/// Custom format used by Ternary-Bonsai GGUF files: 2 bytes FP16 scale + 4 bytes packed 1-bit codes
/// Each bit: 0→-1, 1→+1 (32 ternary values packed into 4 bytes)
pub fn dequantize_q1_0(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / 6;
    for b in 0..blocks {
        let bo = b * 6;
        let scale: f32 = f16::from_le_bytes([data[bo], data[bo + 1]]).to_f32();
        let codes_start = bo + 2;
        let out_base = b * 32;
        let out_end = (out_base + 32).min(out.len());
        if out_end <= out_base {
            break;
        }
        for byte_i in 0..4 {
            let signs = Q1V[data[codes_start + byte_i] as usize];
            let elem_base = out_base + byte_i * 8;
            for (i, &s) in signs.iter().enumerate() {
                let idx = elem_base + i;
                if idx < out_end {
                    out[idx] = (s as f32) * scale;
                }
            }
        }
    }
}

/// Dequantize PRISM Q1_0_G128 (type 43): 128-element binary {-1,+1} blocks
pub fn dequantize(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / 18;
    for b in 0..blocks {
        let block_start = b * 18;
        let scale_f32: f32 =
            f16::from_le_bytes([data[block_start], data[block_start + 1]]).to_f32();
        let bits_start = block_start + 2;
        let out_base = b * 128;
        let out_end = (out_base + 128).min(out.len());
        if out_end <= out_base {
            break;
        }
        for byte_i in 0..16usize {
            let signs = Q1V[data[bits_start + byte_i] as usize];
            let elem_base = out_base + byte_i * 8;
            for (i, &s) in signs.iter().enumerate() {
                let idx = elem_base + i;
                if idx < out_end {
                    out[idx] = (s as f32) * scale_f32;
                }
            }
        }
    }
}

/// Fused dot product for Q1_0 (type 42): 32-element ternary {-1,+1} blocks
/// Format: 2 bytes FP16 scale + 4 bytes packed 1-bit codes
pub fn dot_q1_0(weight_row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / 32;
    let mut sum = 0.0_f32;
    for b in 0..blocks {
        let bo = b * 6;
        let scale: f32 = f16::from_le_bytes([weight_row[bo], weight_row[bo + 1]]).to_f32();
        let codes = &weight_row[bo + 2..bo + 6];
        let xs = b * 32;
        let mut bsum = 0.0_f32;
        for byte_i in 0..4 {
            let signs = Q1V[codes[byte_i] as usize];
            let elem_base = xs + byte_i * 8;
            bsum += signs[0] as f32 * vec[elem_base]
                + signs[1] as f32 * vec[elem_base + 1]
                + signs[2] as f32 * vec[elem_base + 2]
                + signs[3] as f32 * vec[elem_base + 3]
                + signs[4] as f32 * vec[elem_base + 4]
                + signs[5] as f32 * vec[elem_base + 5]
                + signs[6] as f32 * vec[elem_base + 6]
                + signs[7] as f32 * vec[elem_base + 7];
        }
        sum += bsum * scale;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q1_0_q8_0_avx2(weight_row: &[u8], act_data: &[u8], n: usize) -> f32 {
    let q1_blocks = n / 32;
    let mut sum = 0.0f32;
    for b in 0..q1_blocks {
        let bo = b * 6;
        let w_scale: f32 = f16::from_le_bytes([weight_row[bo], weight_row[bo + 1]]).to_f32();
        let codes = &weight_row[bo + 2..bo + 6];
        let a_start = b * 34;
        let a_scale: f32 = f16::from_le_bytes([act_data[a_start], act_data[a_start + 1]]).to_f32();
        let a_vals = &act_data[a_start + 2..a_start + 34];

        let mut acc = _mm256_setzero_si256();
        // Batch 0: elements 0..15 (mask bytes 0,1)
        let a0 = _mm_loadu_si128(a_vals.as_ptr() as *const __m128i);
        let a16_0 = _mm256_cvtepi8_epi16(a0);
        let s0 = {
            let s0_arr = Q1V[*codes.get_unchecked(0) as usize];
            let s1_arr = Q1V[*codes.get_unchecked(1) as usize];
            let lo = _mm_loadl_epi64(s0_arr.as_ptr() as *const __m128i);
            let hi = _mm_loadl_epi64(s1_arr.as_ptr() as *const __m128i);
            _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(lo, hi))
        };
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16_0, s0));
        // Batch 1: elements 16..31 (mask bytes 2,3)
        let a1 = _mm_loadu_si128(a_vals.as_ptr().add(16) as *const __m128i);
        let a16_1 = _mm256_cvtepi8_epi16(a1);
        let s1 = {
            let s2_arr = Q1V[*codes.get_unchecked(2) as usize];
            let s3_arr = Q1V[*codes.get_unchecked(3) as usize];
            let lo = _mm_loadl_epi64(s2_arr.as_ptr() as *const __m128i);
            let hi = _mm_loadl_epi64(s3_arr.as_ptr() as *const __m128i);
            _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(lo, hi))
        };
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16_1, s1));

        let dot = hsum_256i_avx2(acc);
        sum += dot as f32 * w_scale * a_scale;
    }
    sum
}

pub fn dot_q1_0_q8_0(weight_row: &[u8], act_data: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dot_q1_0_q8_0_avx2(weight_row, act_data, n) };
        }
    }
    dot_q1_0_q8_0_scalar(weight_row, act_data, n)
}

fn dot_q1_0_q8_0_scalar(weight_row: &[u8], act_data: &[u8], n: usize) -> f32 {
    let q1_blocks = n / 32;
    let mut sum = 0.0f32;
    for b in 0..q1_blocks {
        let bo = b * 6;
        let w_scale: f32 = f16::from_le_bytes([weight_row[bo], weight_row[bo + 1]]).to_f32();
        let codes = &weight_row[bo + 2..bo + 6];
        let a_start = b * 34;
        let a_scale: f32 = f16::from_le_bytes([act_data[a_start], act_data[a_start + 1]]).to_f32();
        let a_vals = &act_data[a_start + 2..a_start + 34];

        let mut dot: i32 = 0;
        for (&mask, a_off) in codes.iter().zip([0usize, 8, 16, 24]) {
            let s = Q1V[mask as usize];
            dot += s[0] as i32 * (a_vals[a_off] as i8 as i32);
            dot += s[1] as i32 * (a_vals[a_off + 1] as i8 as i32);
            dot += s[2] as i32 * (a_vals[a_off + 2] as i8 as i32);
            dot += s[3] as i32 * (a_vals[a_off + 3] as i8 as i32);
            dot += s[4] as i32 * (a_vals[a_off + 4] as i8 as i32);
            dot += s[5] as i32 * (a_vals[a_off + 5] as i8 as i32);
            dot += s[6] as i32 * (a_vals[a_off + 6] as i8 as i32);
            dot += s[7] as i32 * (a_vals[a_off + 7] as i8 as i32);
        }
        sum += dot as f32 * w_scale * a_scale;
    }
    sum
}

/// Fused dot product for Q1_0_G128 (model weights) with f32 activation vector.
/// Each block: 2 bytes f16 scale + 16 bytes packed 1-bit codes (128 elements).
pub fn dot_q1_0g128_f32(weight_row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / 128;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let w_start = b * 18;
        let scale: f32 =
            f16::from_le_bytes([weight_row[w_start], weight_row[w_start + 1]]).to_f32();
        let w_bits = &weight_row[w_start + 2..w_start + 18];
        let base = b * 128;
        let mut block_acc = 0.0f32;
        for (byte_i, &bits) in w_bits[..16].iter().enumerate() {
            let signs = Q1V[bits as usize];
            let elem_base = base + byte_i * 8;
            block_acc += signs[0] as f32 * vec[elem_base]
                + signs[1] as f32 * vec[elem_base + 1]
                + signs[2] as f32 * vec[elem_base + 2]
                + signs[3] as f32 * vec[elem_base + 3]
                + signs[4] as f32 * vec[elem_base + 4]
                + signs[5] as f32 * vec[elem_base + 5]
                + signs[6] as f32 * vec[elem_base + 6]
                + signs[7] as f32 * vec[elem_base + 7];
        }
        sum += block_acc * scale;
    }
    sum
}

/// Reference-style AVX2 dot: Q1_0_G128 (128-el) × Q8_0 (32-el).
/// Uses _mm256_shuffle_epi8 + bit masks for sign expansion (no LUT),
/// _mm256_maddubs_epi16 for 32-element dot product, and FMA accumulation.
/// Mirrors ggml_vec_dot_q1_0_q8_0 AVX2 path from llama.cpp.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q1_0g128_q8_0_ptr_avx2(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    use core::ptr::read_unaligned;
    let q1_blocks = n / 128;

    let ones_8 = _mm256_set1_epi8(1);
    let ones_16 = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_si256();
    let byte_shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    );
    let bit_masks: __m256i = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128,
    );

    let mut acc = _mm256_setzero_ps();

    for b in 0..q1_blocks {
        let w_off = b * 18;
        let w_scale: f32 = f16::from_le_bytes([*w_ptr.add(w_off), *w_ptr.add(w_off + 1)]).to_f32();
        let w_bits = w_ptr.add(w_off + 2);

        let qs0 = read_unaligned(w_bits as *const u32);
        let qs1 = read_unaligned(w_bits.add(4) as *const u32);
        let qs2 = read_unaligned(w_bits.add(8) as *const u32);
        let qs3 = read_unaligned(w_bits.add(12) as *const u32);

        let mut acc_block = _mm256_setzero_ps();

        // K=0
        {
            let a_off = (b * 4) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let a_vals = a_ptr.add(a_off + 2);
            let qy = _mm256_loadu_si256(a_vals as *const __m256i);
            let sm = _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_set1_epi32(qs0 as i32), byte_shuf),
                    bit_masks,
                ),
                zero,
            );
            let sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            let s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block =
                _mm256_fmadd_ps(_mm256_set1_ps(a_scale), _mm256_cvtepi32_ps(s32), acc_block);
        }
        // K=1
        {
            let a_off = (b * 4 + 1) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let a_vals = a_ptr.add(a_off + 2);
            let qy = _mm256_loadu_si256(a_vals as *const __m256i);
            let sm = _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_set1_epi32(qs1 as i32), byte_shuf),
                    bit_masks,
                ),
                zero,
            );
            let sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            let s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block =
                _mm256_fmadd_ps(_mm256_set1_ps(a_scale), _mm256_cvtepi32_ps(s32), acc_block);
        }
        // K=2
        {
            let a_off = (b * 4 + 2) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let a_vals = a_ptr.add(a_off + 2);
            let qy = _mm256_loadu_si256(a_vals as *const __m256i);
            let sm = _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_set1_epi32(qs2 as i32), byte_shuf),
                    bit_masks,
                ),
                zero,
            );
            let sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            let s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block =
                _mm256_fmadd_ps(_mm256_set1_ps(a_scale), _mm256_cvtepi32_ps(s32), acc_block);
        }
        // K=3
        {
            let a_off = (b * 4 + 3) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let a_vals = a_ptr.add(a_off + 2);
            let qy = _mm256_loadu_si256(a_vals as *const __m256i);
            let sm = _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_set1_epi32(qs3 as i32), byte_shuf),
                    bit_masks,
                ),
                zero,
            );
            let sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            let s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block =
                _mm256_fmadd_ps(_mm256_set1_ps(a_scale), _mm256_cvtepi32_ps(s32), acc_block);
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(w_scale), acc_block, acc);
    }

    hsum_float_8(acc)
}

/// SSE4.1/AVX2 horizontal sum of 4 × i32 in __m128i
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_128i(v: __m128i) -> i32 {
    let sum = _mm_hadd_epi32(v, v);
    let sum = _mm_hadd_epi32(sum, sum);
    _mm_extract_epi32(sum, 0)
}

/// AVX2 horizontal sum of 8 × i32 in __m256i
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_256i_avx2(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extractf128_si256(v, 1);
    hsum_128i(_mm_add_epi32(lo, hi))
}

/// Horizontal sum of 8 × f32 in __m256
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

/// AVX2-optimized dot product: Q1_0_G128 weight × Q8_0 activation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q1_0g128_q8_0_avx2(weight_row: &[u8], act_data: &[u8], n: usize) -> f32 {
    let q1_blocks = n / 128;
    let mut sum = 0.0f32;
    for b in 0..q1_blocks {
        let w_start = b * 18;
        let w_scale: f32 =
            f16::from_le_bytes([weight_row[w_start], weight_row[w_start + 1]]).to_f32();
        let w_bits = &weight_row[w_start + 2..w_start + 18];

        let mut block_acc = 0.0f32;
        for q8_sub in 0..4 {
            let a_start = (b * 4 + q8_sub) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([act_data[a_start], act_data[a_start + 1]]).to_f32();
            let a_vals = &act_data[a_start + 2..a_start + 34];
            let bits_sub = &w_bits[q8_sub * 4..q8_sub * 4 + 4];

            // Process 32 elements in two 16-element AVX2 batches
            // Each batch: vpmovsxbw (i8→i16) + vpmaddwd (i16×i16→i32)
            let mut acc = _mm256_setzero_si256();

            // Batch 0: elements 0..15 (mask bytes 0,1)
            let a0 = _mm_loadu_si128(a_vals.as_ptr() as *const __m128i);
            let a16_0 = _mm256_cvtepi8_epi16(a0);
            let s0 = {
                let s0_arr = Q1V[*bits_sub.get_unchecked(0) as usize];
                let s1_arr = Q1V[*bits_sub.get_unchecked(1) as usize];
                let lo = _mm_loadl_epi64(s0_arr.as_ptr() as *const __m128i);
                let hi = _mm_loadl_epi64(s1_arr.as_ptr() as *const __m128i);
                _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(lo, hi))
            };
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16_0, s0));

            // Batch 1: elements 16..31 (mask bytes 2,3)
            let a1 = _mm_loadu_si128(a_vals.as_ptr().add(16) as *const __m128i);
            let a16_1 = _mm256_cvtepi8_epi16(a1);
            let s1 = {
                let s2_arr = Q1V[*bits_sub.get_unchecked(2) as usize];
                let s3_arr = Q1V[*bits_sub.get_unchecked(3) as usize];
                let lo = _mm_loadl_epi64(s2_arr.as_ptr() as *const __m128i);
                let hi = _mm_loadl_epi64(s3_arr.as_ptr() as *const __m128i);
                _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(lo, hi))
            };
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16_1, s1));

            block_acc += hsum_256i_avx2(acc) as f32 * a_scale;
        }
        sum += block_acc * w_scale;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
unsafe fn dot_q1_0g128_q8_0_ptr_sse41(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let q1_blocks = n / 128;
    let mut sum = 0.0f32;
    for b in 0..q1_blocks {
        let w_off = b * 18;
        let w_scale: f32 = f16::from_le_bytes([*w_ptr.add(w_off), *w_ptr.add(w_off + 1)]).to_f32();
        let w_bits = w_ptr.add(w_off + 2);

        let mut block_acc = 0.0f32;
        for q8_sub in 0..4 {
            let a_off = (b * 4 + q8_sub) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let a_vals = a_ptr.add(a_off + 2);
            let bits_sub = w_bits.add(q8_sub * 4);

            let mut dot: i32 = 0;
            for g in 0..4 {
                let mask = *bits_sub.add(g);
                let a_off_inner = g * 8;
                let a8 = _mm_loadl_epi64(a_vals.add(a_off_inner) as *const __m128i);
                let a16 = _mm_cvtepi8_epi16(a8);
                let s_arr = Q1V[mask as usize];
                let s8 = _mm_loadl_epi64(s_arr.as_ptr() as *const __m128i);
                let s16 = _mm_cvtepi8_epi16(s8);
                let prod = _mm_madd_epi16(a16, s16);
                dot += hsum_128i(prod);
            }
            block_acc += dot as f32 * a_scale;
        }
        sum += block_acc * w_scale;
    }
    sum
}

/// Raw-pointer AVX2 for Q1_0 (32-element blocks, 6 bytes/block)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q1_0_q8_0_ptr_avx2(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let q1_blocks = n / 32;
    let mut sum = 0.0f32;
    for b in 0..q1_blocks {
        let bo = b * 6;
        let w_scale: f32 = f16::from_le_bytes([*w_ptr.add(bo), *w_ptr.add(bo + 1)]).to_f32();
        let codes = w_ptr.add(bo + 2);
        let a_start = b * 34;
        let a_scale: f32 =
            f16::from_le_bytes([*a_ptr.add(a_start), *a_ptr.add(a_start + 1)]).to_f32();
        let a_vals = a_ptr.add(a_start + 2);

        let mut acc = _mm256_setzero_si256();
        let a0 = _mm_loadu_si128(a_vals as *const __m128i);
        let a16_0 = _mm256_cvtepi8_epi16(a0);
        let s0 = {
            let s0_arr = Q1V[*codes.add(0) as usize];
            let s1_arr = Q1V[*codes.add(1) as usize];
            let lo = _mm_loadl_epi64(s0_arr.as_ptr() as *const __m128i);
            let hi = _mm_loadl_epi64(s1_arr.as_ptr() as *const __m128i);
            _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(lo, hi))
        };
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16_0, s0));

        let a1 = _mm_loadu_si128(a_vals.add(16) as *const __m128i);
        let a16_1 = _mm256_cvtepi8_epi16(a1);
        let s1 = {
            let s2_arr = Q1V[*codes.add(2) as usize];
            let s3_arr = Q1V[*codes.add(3) as usize];
            let lo = _mm_loadl_epi64(s2_arr.as_ptr() as *const __m128i);
            let hi = _mm_loadl_epi64(s3_arr.as_ptr() as *const __m128i);
            _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(lo, hi))
        };
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16_1, s1));

        let dot = hsum_256i_avx2(acc);
        sum += dot as f32 * w_scale * a_scale;
    }
    sum
}

/// Raw-pointer SSE4.1 for Q1_0 (32-element blocks)
#[cfg(target_arch = "x86_64")]
unsafe fn dot_q1_0_q8_0_ptr_sse41(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let q1_blocks = n / 32;
    let mut sum = 0.0f32;
    for b in 0..q1_blocks {
        let bo = b * 6;
        let w_scale: f32 = f16::from_le_bytes([*w_ptr.add(bo), *w_ptr.add(bo + 1)]).to_f32();
        let codes = w_ptr.add(bo + 2);
        let a_start = b * 34;
        let a_scale: f32 =
            f16::from_le_bytes([*a_ptr.add(a_start), *a_ptr.add(a_start + 1)]).to_f32();
        let a_vals = a_ptr.add(a_start + 2);

        let mut dot: i32 = 0;
        for g in 0..4 {
            let mask = *codes.add(g);
            let s = Q1V[mask as usize];
            let a_off_inner = g * 8;
            let a8 = _mm_loadl_epi64(a_vals.add(a_off_inner) as *const __m128i);
            let a16 = _mm_cvtepi8_epi16(a8);
            let s8 = _mm_loadl_epi64(s.as_ptr() as *const __m128i);
            let s16 = _mm_cvtepi8_epi16(s8);
            let prod = _mm_madd_epi16(a16, s16);
            dot += hsum_128i(prod);
        }
        sum += dot as f32 * w_scale * a_scale;
    }
    sum
}

/// Raw-pointer dispatch for Q1_0 (32-element blocks) dot with Q8_0
/// # Safety
/// w_ptr and a_ptr must be valid for the full accessed range.
pub unsafe fn dot_q1_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            dot_q1_0_q8_0_ptr_avx2(w_ptr, a_ptr, n)
        } else {
            dot_q1_0_q8_0_ptr_sse41(w_ptr, a_ptr, n)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let q1_blocks = n / 32;
        let mut sum = 0.0f32;
        for b in 0..q1_blocks {
            let bo = b * 6;
            let w_scale: f32 = f16::from_le_bytes([*w_ptr.add(bo), *w_ptr.add(bo + 1)]).to_f32();
            let codes = w_ptr.add(bo + 2);
            let a_start = b * 34;
            let a_scale: f32 =
                f16::from_le_bytes([*a_ptr.add(a_start), *a_ptr.add(a_start + 1)]).to_f32();
            let a_vals = a_ptr.add(a_start + 2);
            let mut dot: i32 = 0;
            for g in 0..4 {
                let mask = *codes.add(g);
                let s = Q1V[mask as usize];
                let a_off_inner = g * 8;
                dot += s[0] as i32 * (*a_vals.add(a_off_inner) as i8 as i32);
                dot += s[1] as i32 * (*a_vals.add(a_off_inner + 1) as i8 as i32);
                dot += s[2] as i32 * (*a_vals.add(a_off_inner + 2) as i8 as i32);
                dot += s[3] as i32 * (*a_vals.add(a_off_inner + 3) as i8 as i32);
                dot += s[4] as i32 * (*a_vals.add(a_off_inner + 4) as i8 as i32);
                dot += s[5] as i32 * (*a_vals.add(a_off_inner + 5) as i8 as i32);
                dot += s[6] as i32 * (*a_vals.add(a_off_inner + 6) as i8 as i32);
                dot += s[7] as i32 * (*a_vals.add(a_off_inner + 7) as i8 as i32);
            }
            sum += dot as f32 * w_scale * a_scale;
        }
        sum
    }
}

/// Dispatch: dot_q1_0g128_q8_0 using raw pointers (no slice bounds checks).
/// Uses shuffle-based AVX2 kernel (no LUT, avoids L1 cache pressure).
/// # Safety
/// Both pointers must be valid for n/128*18 (w_ptr) and n/32*34 (a_ptr) bytes.
/// n must be a multiple of 128.
#[inline]
pub unsafe fn dot_q1_0g128_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            dot_q1_0g128_q8_0_ptr_avx2(w_ptr, a_ptr, n)
        } else {
            dot_q1_0g128_q8_0_ptr_sse41(w_ptr, a_ptr, n)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Scalar fallback with pointer arithmetic
        let q1_blocks = n / 128;
        let mut sum = 0.0f32;
        for b in 0..q1_blocks {
            let w_off = b * 18;
            let w_scale: f32 =
                f16::from_le_bytes([*w_ptr.add(w_off), *w_ptr.add(w_off + 1)]).to_f32();
            let w_bits = w_ptr.add(w_off + 2);
            let mut block_acc = 0.0f32;
            for q8_sub in 0..4 {
                let a_off = (b * 4 + q8_sub) * 34;
                let a_scale: f32 =
                    f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
                let a_vals = a_ptr.add(a_off + 2);
                let bits_sub = w_bits.add(q8_sub * 4);
                let mut dot: i32 = 0;
                for g in 0..4 {
                    let mask = *bits_sub.add(g);
                    let s = Q1V[mask as usize];
                    let a_off_inner = g * 8;
                    dot += s[0] as i32 * (*a_vals.add(a_off_inner) as i8 as i32);
                    dot += s[1] as i32 * (*a_vals.add(a_off_inner + 1) as i8 as i32);
                    dot += s[2] as i32 * (*a_vals.add(a_off_inner + 2) as i8 as i32);
                    dot += s[3] as i32 * (*a_vals.add(a_off_inner + 3) as i8 as i32);
                    dot += s[4] as i32 * (*a_vals.add(a_off_inner + 4) as i8 as i32);
                    dot += s[5] as i32 * (*a_vals.add(a_off_inner + 5) as i8 as i32);
                    dot += s[6] as i32 * (*a_vals.add(a_off_inner + 6) as i8 as i32);
                    dot += s[7] as i32 * (*a_vals.add(a_off_inner + 7) as i8 as i32);
                }
                block_acc += dot as f32 * a_scale;
            }
            sum += block_acc * w_scale;
        }
        sum
    }
}

/// SSE4.1-optimized dot: processes 8 elements per batch using 128-bit SIMD.
#[cfg(target_arch = "x86_64")]
fn dot_q1_0g128_q8_0_sse41(weight_row: &[u8], act_data: &[u8], n: usize) -> f32 {
    let q1_blocks = n / 128;
    let mut sum = 0.0f32;
    for b in 0..q1_blocks {
        let w_start = b * 18;
        let w_scale: f32 =
            f16::from_le_bytes([weight_row[w_start], weight_row[w_start + 1]]).to_f32();
        let w_bits = &weight_row[w_start + 2..w_start + 18];

        let mut block_acc = 0.0f32;
        for q8_sub in 0..4 {
            let a_start = (b * 4 + q8_sub) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([act_data[a_start], act_data[a_start + 1]]).to_f32();
            let a_vals = &act_data[a_start + 2..a_start + 34];
            let bits_sub = &w_bits[q8_sub * 4..q8_sub * 4 + 4];

            let mut dot: i32 = 0;
            unsafe {
                for (&mask, a_off) in bits_sub.iter().zip([0usize, 8, 16, 24]) {
                    // 8-element batch: load i8, sign-extend to i16, pmaddwd
                    let a8 = _mm_loadl_epi64(a_vals.as_ptr().add(a_off) as *const __m128i);
                    let a16 = _mm_cvtepi8_epi16(a8);
                    let s_arr = Q1V[mask as usize];
                    let s8 = _mm_loadl_epi64(s_arr.as_ptr() as *const __m128i);
                    let s16 = _mm_cvtepi8_epi16(s8);
                    let prod = _mm_madd_epi16(a16, s16);
                    dot += hsum_128i(prod);
                }
            }
            block_acc += dot as f32 * a_scale;
        }
        sum += block_acc * w_scale;
    }
    sum
}

pub fn dot_q1_0g128_q8_0(weight_row: &[u8], act_data: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dot_q1_0g128_q8_0_avx2(weight_row, act_data, n) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        dot_q1_0g128_q8_0_sse41(weight_row, act_data, n)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        dot_q1_0g128_q8_0_scalar(weight_row, act_data, n)
    }
}

/// Scalar fallback (same as original implementation)
#[cfg(not(target_arch = "x86_64"))]
fn dot_q1_0g128_q8_0_scalar(weight_row: &[u8], act_data: &[u8], n: usize) -> f32 {
    let q1_blocks = n / 128;
    let mut sum = 0.0f32;
    for b in 0..q1_blocks {
        let w_start = b * 18;
        let w_scale: f32 =
            f16::from_le_bytes([weight_row[w_start], weight_row[w_start + 1]]).to_f32();
        let w_bits = &weight_row[w_start + 2..w_start + 18];

        let mut block_acc = 0.0f32;
        for q8_sub in 0..4 {
            let a_start = (b * 4 + q8_sub) * 34;
            let a_scale: f32 =
                f16::from_le_bytes([act_data[a_start], act_data[a_start + 1]]).to_f32();
            let a_vals = &act_data[a_start + 2..a_start + 34];
            let bits_sub = &w_bits[q8_sub * 4..q8_sub * 4 + 4];

            let mut dot: i32 = 0;
            for (&mask, a_off) in bits_sub.iter().zip([0usize, 8, 16, 24]) {
                let s = Q1V[mask as usize];
                dot += s[0] as i32 * (a_vals[a_off] as i8 as i32);
                dot += s[1] as i32 * (a_vals[a_off + 1] as i8 as i32);
                dot += s[2] as i32 * (a_vals[a_off + 2] as i8 as i32);
                dot += s[3] as i32 * (a_vals[a_off + 3] as i8 as i32);
                dot += s[4] as i32 * (a_vals[a_off + 4] as i8 as i32);
                dot += s[5] as i32 * (a_vals[a_off + 5] as i8 as i32);
                dot += s[6] as i32 * (a_vals[a_off + 6] as i8 as i32);
                dot += s[7] as i32 * (a_vals[a_off + 7] as i8 as i32);
            }
            block_acc += dot as f32 * a_scale;
        }
        sum += block_acc * w_scale;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q8_0;
    use std::time::Instant;

    /// Microbenchmark the active shuffle AVX2 kernel across model dimensions
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn bench_shuffle_kernel() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 benchmark (no AVX2)");
            return;
        }
        let dims = [2048usize, 4096, 2560, 9728];
        let iters = 5_000;

        for &n in &dims {
            let n_blocks = n / 128;
            let mut w = vec![0u8; n_blocks * 18];
            let mut a = vec![0u8; n_blocks * 4 * 34];

            for b in 0..n_blocks {
                let bo = b * 18;
                let scale = ((b as f32 + 1.0) * 0.1).sin().abs() * 0.5 + 0.01;
                let s = half::f16::from_f32(scale).to_le_bytes();
                w[bo] = s[0];
                w[bo + 1] = s[1];
                for j in 0..16 {
                    w[bo + 2 + j] = ((b * 16 + j).wrapping_mul(37) ^ 0xAB) as u8;
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

            unsafe {
                let _ = dot_q1_0g128_q8_0_ptr_avx2(w.as_ptr(), a.as_ptr(), n);
            }

            let t0 = Instant::now();
            let mut sum = 0.0f32;
            for _ in 0..iters {
                sum += unsafe { dot_q1_0g128_q8_0_ptr_avx2(w.as_ptr(), a.as_ptr(), n) };
            }
            let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
            eprintln!("n={:5}: Shuffle={:7.1}ns  (sum={})", n, ns, sum);
        }
    }

    #[test]
    fn test_q1_0_dequantize() {
        let mut data = vec![0u8; 6 * 2];
        let s = f16::from_f32(1.0).to_le_bytes();
        data[0] = s[0];
        data[1] = s[1];
        // Pack 1-bit codes: 0b00001011 = bits [1,1,0,1,0,0,0,0] = [+1,+1,-1,+1,-1,-1,-1,-1]
        data[2] = 0b00001011;
        let mut out = vec![0.0f32; 64];
        dequantize_q1_0(&data, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6, "out[0] should be +1");
        assert!((out[1] - 1.0).abs() < 1e-6, "out[1] should be +1");
        assert!((out[2] - (-1.0)).abs() < 1e-6, "out[2] should be -1");
        assert!((out[3] - 1.0).abs() < 1e-6, "out[3] should be +1");
    }

    #[test]
    fn test_q1_0_dot() {
        let mut w_data = vec![0u8; 6];
        let ws = f16::from_f32(2.0).to_le_bytes();
        w_data[0] = ws[0];
        w_data[1] = ws[1];
        // All bits = 1 → weight = +scale for all
        for i in 0..4 {
            w_data[2 + i] = 0xFF;
        }
        let vec: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.01).collect();
        let dot = dot_q1_0(&w_data, &vec, 32);
        assert!(dot.is_finite(), "dot should be finite, got {}", dot);
    }

    #[test]
    fn test_dot_q1_0g128_q8_0() {
        // Create a simple Q1_0_G128 weight row
        let mut w_data = vec![0u8; 18];
        let ws = f16::from_f32(2.0).to_le_bytes();
        w_data[0] = ws[0];
        w_data[1] = ws[1];
        for i in 0..16 {
            w_data[2 + i] = 0xFF; // all 1 bits -> +scale
        }

        // Q8_0 activation: all ones scaled by 1.0
        let src: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let mut act_data = Vec::new();
        q8_0::quantize(&src, &mut act_data);

        let dot = dot_q1_0g128_q8_0(&w_data, &act_data, 128);
        assert!(dot.is_finite(), "dot should be finite, got {}", dot);
    }

    /// Fused dot product for Q1_0_G128: inlines both dequantize and dot into one pass.
    /// This is what the matmul uses. We compare it against dequantize-then-f32-dot.
    fn fused_dot_q1_0g128(w: &[u8], x: &[f32], n: usize) -> f32 {
        let blocks = n / 128;
        let mut total = 0.0f32;
        for b in 0..blocks {
            let bo = b * 18;
            let sf: f32 = f16::from_le_bytes([w[bo], w[bo + 1]]).to_f32();
            let bb = bo + 2;
            let xs = b * 128;
            let mut bsum = 0.0f32;
            for byte_i in 0..16usize {
                let mut bits = w[bb + byte_i];
                let xb = xs + byte_i * 8;
                for k in 0..8usize {
                    bsum += if bits & 1 != 0 { x[xb + k] } else { -x[xb + k] };
                    bits >>= 1;
                }
            }
            total += bsum * sf;
        }
        total
    }

    /// Reference: dequantize then f32 dot product.
    fn ref_dot_q1_0g128(w: &[u8], x: &[f32], n: usize) -> f32 {
        // Dequantize
        let mut dq = vec![0.0f32; n];
        for b in 0..n / 128 {
            let bo = b * 18;
            let scale = f16::from_le_bytes([w[bo], w[bo + 1]]).to_f32();
            let bits_start = bo + 2;
            let out_base = b * 128;
            for byte_i in 0..16 {
                let byte_val = w[bits_start + byte_i];
                let elem_base = out_base + byte_i * 8;
                for bit in 0..8 {
                    let idx = elem_base + bit;
                    if idx < n {
                        dq[idx] = if (byte_val >> bit) & 1 != 0 {
                            scale
                        } else {
                            -scale
                        };
                    }
                }
            }
        }
        // F32 dot
        x.iter().zip(dq.iter()).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn test_fused_vs_ref_dot_q1_0g128_random() {
        // Create a random Q1_0_G128 weight row (2 blocks = 256 elements)
        let mut w = vec![0u8; 2 * 18];
        for b in 0..2 {
            let bo = b * 18;
            // Random scale in [0.01, 1.0]
            let scale = 0.01 + (b as f32 * 0.3).sin().abs() * 0.99;
            let s = f16::from_f32(scale).to_le_bytes();
            w[bo] = s[0];
            w[bo + 1] = s[1];
            // Random bits
            for j in 0..16 {
                w[bo + 2 + j] = (j.wrapping_mul(37) ^ 0xAB) as u8;
            }
        }

        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.13).sin()).collect();

        let fused = fused_dot_q1_0g128(&w, &x, 256);
        let reference = ref_dot_q1_0g128(&w, &x, 256);

        assert!(
            (fused - reference).abs() < 1e-4,
            "fused={} != ref={}, diff={}",
            fused,
            reference,
            (fused - reference).abs()
        );
    }

    #[test]
    fn test_fused_vs_ref_dot_q1_0g128_zeros() {
        // Edge case: zero input
        let mut w = vec![0u8; 18];
        let s = f16::from_f32(0.5).to_le_bytes();
        w[0] = s[0];
        w[1] = s[1];
        for j in 0..16 {
            w[2 + j] = 0xFF;
        }
        let x = vec![0.0f32; 128];
        let fused = fused_dot_q1_0g128(&w, &x, 128);
        let reference = ref_dot_q1_0g128(&w, &x, 128);
        assert!(
            (fused - reference).abs() < 1e-6,
            "zeros: fused={} != ref={}",
            fused,
            reference
        );
    }

    #[test]
    fn test_fused_vs_ref_dot_q1_0g128_negative() {
        // Edge case: all negative bits
        let mut w = vec![0u8; 18];
        let s = f16::from_f32(1.5).to_le_bytes();
        w[0] = s[0];
        w[1] = s[1];
        for j in 0..16 {
            w[2 + j] = 0x00; // all bits 0 -> -scale
        }
        let x: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let fused = fused_dot_q1_0g128(&w, &x, 128);
        let reference = ref_dot_q1_0g128(&w, &x, 128);
        assert!(
            (fused - reference).abs() < 1e-5,
            "neg: fused={} != ref={}",
            fused,
            reference
        );
    }
}
