// Q2_0_G64: stock llama.cpp Q2_0 ternary, 64-element groups.
// 18 bytes per 64 weights: fp16 scale d FIRST, then qs[16] (2 bits/element).
// 2-bit codes: 00 = -1, 01 = 0, 10 = +1, 11 = +2 (11 unused for ternary weights).
// Code order matches Q2_0: element j of a byte uses bit offset (j % 4) * 2, LSB first.
// Reference: ggml-org/llama.cpp `ggml-quants.c` (`dequantize_row_q2_0`, QK=64).
//
// This is not a format of Hearth's invention. GGUF type 42 names two incompatible layouts:
// the Prism fork packs 128 weights into a 34-byte block (`q2_0.rs`), stock llama.cpp packs 64
// into 18 bytes, and the Ternary-Bonsai 2.5 `Q2_0` file is the stock one — 36 bytes per 128
// weights, i.e. two of these groups. `hearth-gguf` tells them apart from the tensor footprint
// before any weight is decoded and reports this type only for the 64-element one.
//
// Verified against the working `PQ2_0` pack on the same model: decoding the 27B embedding row
// with this layout gives the identical values (cosine 1.00000, 512/512 signs, same mean|w|).

use half::f16;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const BLOCK_SIZE: usize = 64;
const BLOCK_BYTES: usize = 18;
/// Code bytes per block: 64 elements at 2 bits each.
const QS_BYTES: usize = 16;
/// Q8_0 activation block: 32 elements, fp16 scale + 32 i8 — one per half of a Q2_0_G64 block.
const Q8_BLOCK_BYTES: usize = 34;

/// Q2V_I16[b][j] = 2-bit code j of byte b as i16, mapped through {0:-1, 1:0, 2:+1, 3:+2}.
/// 256 entries × 4 i16 × 2 bytes = 2 KB, kept hot in L1 like the Q2_0 table.
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

/// Dequantize Q2_0_G64 to f32.
pub fn dequantize(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([data[bo], data[bo + 1]]).to_f32();
        let out_base = b * BLOCK_SIZE;
        for byte_i in 0..QS_BYTES {
            let vals = Q2V_I16[data[bo + 2 + byte_i] as usize];
            let elem_base = out_base + byte_i * 4;
            for (i, &v) in vals.iter().enumerate() {
                let idx = elem_base + i;
                if idx < out.len() {
                    out[idx] = v as f32 * scale;
                }
            }
        }
    }
}

/// Dot product of one Q2_0_G64 weight row with an f32 vector. Reference path.
pub fn dot_q2_0_g64_f32(weight_row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([weight_row[bo], weight_row[bo + 1]]).to_f32();
        let mut block_acc = 0.0f32;
        for byte_i in 0..QS_BYTES {
            let vals = Q2V_I16[weight_row[bo + 2 + byte_i] as usize];
            let elem_base = b * BLOCK_SIZE + byte_i * 4;
            for (i, &v) in vals.iter().enumerate() {
                block_acc += v as f32 * vec[elem_base + i];
            }
        }
        sum += block_acc * scale;
    }
    sum
}

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

/// AVX2 LUT kernel: 16 elements per batch, FMA accumulation across all blocks, one hsum.
///
/// # Safety
/// `w_ptr` must be valid for `n/64*18` bytes and `a_ptr` for `n/32*34` bytes; the caller must
/// confirm AVX2, FMA and F16C are present — see [`avx2_lut_available`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
pub unsafe fn dot_q2_0_g64_q8_0_lut_avx2(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
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
        // One Q2_0_G64 block covers 64 weights: two Q8_0 activation blocks of 32.
        for sub in 0..2 {
            let a_off = (b * 2 + sub) * Q8_BLOCK_BYTES;
            let a_scale = _mm256_set1_ps(crate::f16_bits_to_f32(u16::from_le_bytes([
                *a_ptr.add(a_off),
                *a_ptr.add(a_off + 1),
            ])));
            let a_vals = a_ptr.add(a_off + 2);
            let qs_sub = qs.add(sub * 8);

            let mut acc = _mm256_setzero_si256();
            for half in 0..2usize {
                let a = _mm_loadu_si128(a_vals.add(half * 16) as *const __m128i);
                let a16 = _mm256_cvtepi8_epi16(a);
                let mut w_buf: [i16; 16] = [0i16; 16];
                for j in 0..4usize {
                    core::ptr::copy_nonoverlapping(
                        Q2V_I16[*qs_sub.add(half * 4 + j) as usize].as_ptr(),
                        w_buf.as_mut_ptr().add(j * 4),
                        4,
                    );
                }
                let s = _mm256_loadu_si256(w_buf.as_ptr() as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, s));
            }

            acc_block = _mm256_fmadd_ps(a_scale, _mm256_cvtepi32_ps(acc), acc_block);
        }
        acc_global = _mm256_fmadd_ps(w_scale, acc_block, acc_global);
    }
    hsum_float_8(acc_global)
}

/// Scalar fallback: portable and dependency-free, used when AVX2/FMA/F16C are absent and on
/// every non-x86_64 target. It converts the fp16 scales through `half` rather than the f16c
/// intrinsic, which is x86-only.
///
/// # Safety
/// `w_ptr` must be valid for `n/64*18` bytes and `a_ptr` for `n/32*34` bytes.
pub unsafe fn dot_q2_0_g64_q8_0_scalar(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let w_off = b * BLOCK_BYTES;
        let w_scale = f16::from_le_bytes([*w_ptr.add(w_off), *w_ptr.add(w_off + 1)]).to_f32();
        let mut block_acc = 0.0f32;
        for sub in 0..2 {
            let a_off = (b * 2 + sub) * Q8_BLOCK_BYTES;
            let a_scale = f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let a_vals = a_ptr.add(a_off + 2);
            let qs = w_ptr.add(w_off + 2 + sub * 8);
            let mut int_acc = 0i32;
            for j in 0..8usize {
                let vals = Q2V_I16[*qs.add(j) as usize];
                for (k, &v) in vals.iter().enumerate() {
                    int_acc += v as i32 * (*a_vals.add(j * 4 + k) as i8) as i32;
                }
            }
            block_acc += int_acc as f32 * a_scale;
        }
        sum += block_acc * w_scale;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
fn avx2_lut_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
}

/// The Q8 activation dot kernel this machine runs for a `Q2_0G64` weight row.
///
/// # Safety
/// `w_ptr` must be valid for `n/64*18` bytes and `a_ptr` for `n/32*34` bytes.
pub unsafe fn dot_q2_0_g64_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_lut_available() {
            return dot_q2_0_g64_q8_0_lut_avx2(w_ptr, a_ptr, n);
        }
    }
    dot_q2_0_g64_q8_0_scalar(w_ptr, a_ptr, n)
}

/// Name of the kernel [`dot_q2_0_g64_q8_0_ptr`] selects on this machine.
pub fn kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_lut_available() {
            return "avx2_lut_g64";
        }
    }
    "scalar_g64"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        f16::from_f32(v).to_le_bytes()
    }

    /// Codes 0,1,2,3 in one byte must map to -1, 0, +1, +2 times the block scale.
    #[test]
    fn dequantize_maps_codes_through_the_group_scale() {
        let mut data = vec![0u8; BLOCK_BYTES];
        let s = f16_bytes(2.0);
        data[0] = s[0];
        data[1] = s[1];
        data[2] = 0b1110_0100; // codes (LSB first) 0, 1, 2, 3
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequantize(&data, &mut out);
        assert_eq!(&out[..4], &[-2.0, 0.0, 2.0, 4.0]);
        assert_eq!(out[4], -2.0, "second byte starts the next four elements");
    }

    /// A second 18-byte block must be read from offset 18, not from a 128-element stride.
    #[test]
    fn dequantize_reads_eighteen_byte_blocks() {
        let mut data = vec![0u8; BLOCK_BYTES * 2];
        let s0 = f16_bytes(1.0);
        let s1 = f16_bytes(4.0);
        data[0] = s0[0];
        data[1] = s0[1];
        data[2] = 0b0000_0010; // first code +1
        data[BLOCK_BYTES] = s1[0];
        data[BLOCK_BYTES + 1] = s1[1];
        data[BLOCK_BYTES + 2] = 0b0000_0010;
        let mut out = vec![0.0f32; BLOCK_SIZE * 2];
        dequantize(&data, &mut out);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[BLOCK_SIZE], 4.0);
    }

    /// The dispatch must agree with the reference dot, bit-comparable within f32 tolerance.
    #[test]
    fn q8_kernel_matches_reference_dot() {
        // Deterministic pseudo-random weights and activations, four blocks' worth.
        let n = BLOCK_SIZE * 4;
        let mut w = vec![0u8; n / BLOCK_SIZE * BLOCK_BYTES];
        let mut a = vec![0u8; n / 32 * Q8_BLOCK_BYTES];
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        };
        for (i, b) in w.iter_mut().enumerate() {
            *b = if i % BLOCK_BYTES < 2 { 0x3c } else { next() };
        }
        for blk in 0..n / 32 {
            let off = blk * Q8_BLOCK_BYTES;
            a[off] = 0x30;
            a[off + 1] = 0x3c;
            for j in 0..32 {
                a[off + 2 + j] = next();
            }
        }

        let mut acts = vec![0i8; n];
        let mut act_scale = vec![0f32; n / 32];
        for blk in 0..n / 32 {
            let off = blk * Q8_BLOCK_BYTES;
            act_scale[blk] = f16::from_le_bytes([a[off], a[off + 1]]).to_f32();
            for j in 0..32 {
                acts[blk * 32 + j] = a[off + 2 + j] as i8;
            }
        }
        let mut f = vec![0f32; n];
        for i in 0..n {
            f[i] = acts[i] as f32 * act_scale[i / 32];
        }
        let reference = dot_q2_0_g64_f32(&w, &f, n);
        let scalar = unsafe { dot_q2_0_g64_q8_0_scalar(w.as_ptr(), a.as_ptr(), n) };
        let dispatched = unsafe { dot_q2_0_g64_q8_0_ptr(w.as_ptr(), a.as_ptr(), n) };
        assert!(
            (scalar - reference).abs() <= reference.abs() * 1e-4 + 1e-3,
            "scalar {scalar} vs reference {reference}"
        );
        assert!(
            (dispatched - reference).abs() <= reference.abs() * 1e-4 + 1e-3,
            "{} {dispatched} vs reference {reference}",
            kernel_name()
        );
    }
}
