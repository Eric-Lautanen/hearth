// PTQ1_0: Prism ternary packing, GGML type 143.
// 128-element blocks, 28 bytes: qs[24] (5 trits/byte, stages 16-then-8) +
// qh[2] (4 trits/byte, stride 2) + fp16 scale d (last field).
// Reference: PrismML-Eng/llama.cpp (prism branch) ggml-common.h / ggml-quants.c.
//
// Packing scheme: 5 trits t0..t4 (each 0,1,2 representing weight -1,0,+1) are
// combined into Q = 81*t0 + 27*t1 + 9*t2 + 3*t3 + t4 (0..242) and stored as
// b = ceil-round(256*Q/243). Extraction uses modular arithmetic:
//   b*3^n (mod 256) = Q*3^n (mod 256)   (Q*3^n <= 242*243 = 6*3^9 < 65536)
//   Q = (b*243 (mod 256)) * inv3 (mod 256)   where inv3 = 171 (3*171 = 513 = 2 mod 256)
//   trit_n = (Q * 3^n) / 243   via ((Q*3^n mod 256) * 3) >> 8
// NOTE: extraction does NOT need the ceil-round to be inverted first — b*243 mod 256
// recovers Q mod 256 exactly because 242*243 < 65536 and 256 | 243*256.

use half::f16;

const BLOCK_SIZE: usize = 128;
const BLOCK_BYTES: usize = 28;
const QS_BYTES: usize = 24;
const QH_BYTES: usize = 2;

const POW3: [u16; 5] = [1, 3, 9, 27, 81];

/// TRIT_N[b][n] = trit n (0..5) of byte b, already mapped to weight space {-1, 0, +1}.
/// trit_n = (((b*3^n) mod 256) * 3) >> 8, value - 1.
const TRIT_N: [[i8; 5]; 256] = {
    let mut table = [[0i8; 5]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut n = 0usize;
        while n < 5 {
            let prod = ((b as u16 * POW3[n]) & 0xFF) as u8; // b*3^n mod 256
            let v = (prod as u16 * 3) >> 8; // 0..=2
            table[b][n] = v as i8 - 1;
            n += 1;
        }
        b += 1;
    }
    table
};

/// QH_TRIT[b][n] = trit n (0..4) of a qh byte (4 trits/byte), mapped to {-1, 0, +1}.
const QH_TRIT: [[i8; 4]; 256] = {
    let mut table = [[0i8; 4]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut n = 0usize;
        while n < 4 {
            let prod = ((b as u16 * POW3[n]) & 0xFF) as u8;
            let v = (prod as u16 * 3) >> 8;
            table[b][n] = v as i8 - 1;
            n += 1;
        }
        b += 1;
    }
    table
};

/// Dequantize PTQ1_0 (type 143) to f32, exactly matching dequantize_row_ptq1_0.
/// Output order per block: qs bytes are consumed in stages {32,16,8} covering 24 bytes
/// (16-byte stage then 8-byte stage); within each stage, for each of 5 trit slots,
/// all `c` consecutive bytes emit c consecutive values; then qh emits 2*4 values
/// with byte-major order (trit slot n, then byte h).
pub fn dequantize(data: &[u8], out: &mut [f32]) {
    let blocks = data.len() / BLOCK_BYTES;
    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([data[bo + 26], data[bo + 27]]).to_f32();
        let qs = &data[bo..bo + QS_BYTES];
        let qh = &data[bo + QS_BYTES..bo + QS_BYTES + QH_BYTES];
        let out_base = b * BLOCK_SIZE;
        let out_end = (out_base + BLOCK_SIZE).min(out.len());
        let y = &mut out[out_base..out_end];

        // Stages 32/16/8 over the 24 qs bytes — for this block that is a 16-byte
        // window then an 8-byte one. Emission is slot-major, so the output index
        // of (slot n, byte m) is `out_off + n * c + m`.
        let mut out_off = 0usize;
        let mut j = 0usize;
        for c in [32usize, 16, 8] {
            while j + c <= QS_BYTES {
                for (m, &byte) in qs[j..j + c].iter().enumerate() {
                    for (n, &trit) in TRIT_N[byte as usize].iter().enumerate() {
                        let idx = out_off + n * c + m;
                        if idx < y.len() {
                            y[idx] = trit as f32 * scale;
                        }
                    }
                }
                out_off += 5 * c;
                j += c;
            }
        }
        // qh: QH_BYTES bytes x 4 trits, same slot-major layout.
        for (h, &byte) in qh.iter().enumerate() {
            for (n, &trit) in QH_TRIT[byte as usize].iter().enumerate() {
                let idx = out_off + n * QH_BYTES + h;
                if idx < y.len() {
                    y[idx] = trit as f32 * scale;
                }
            }
        }
        debug_assert_eq!(out_off + 4 * QH_BYTES, BLOCK_SIZE);
    }
}

/// Dot product of one PTQ1_0 row with an f32 vector. Weights are {-1,0,+1}*scale,
/// so the dot is a signed sum of activations scaled per 128-group.
pub fn dot_ptq1_0_f32(weight_row: &[u8], vec: &[f32], n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([weight_row[bo + 26], weight_row[bo + 27]]).to_f32();
        let qs = &weight_row[bo..bo + QS_BYTES];
        let qh = &weight_row[bo + QS_BYTES..bo + QS_BYTES + QH_BYTES];
        let base = b * BLOCK_SIZE;
        let mut acc = 0.0f32;

        // Slot-major, matching the SIMD kernels' summation order.
        let mut out_off = 0usize;
        let mut j = 0usize;
        for c in [32usize, 16, 8] {
            while j + c <= QS_BYTES {
                for n in [0usize, 1, 2, 3, 4] {
                    for (m, &byte) in qs[j..j + c].iter().enumerate() {
                        acc += TRIT_N[byte as usize][n] as f32 * vec[base + out_off + n * c + m];
                    }
                }
                out_off += 5 * c;
                j += c;
            }
        }
        for n in [0usize, 1, 2, 3] {
            for (h, &byte) in qh.iter().enumerate() {
                acc += QH_TRIT[byte as usize][n] as f32 * vec[base + out_off + n * QH_BYTES + h];
            }
        }
        sum += acc * scale;
    }
    sum
}

/// Dot product of one PTQ1_0 row with a Q8_0-quantized activation vector.
/// One PTQ1_0 block (128 elements) spans 4 Q8_0 sub-blocks (32 elements, 34 bytes each).
/// Weights are dequantized to a stack buffer per block, then dotted with per-sub-block
/// activation scales applied. Scalar reference implementation — SIMD is future work.
pub fn dot_ptq1_0_q8_0(weight_row: &[u8], act_data: &[u8], n: usize) -> f32 {
    unsafe { dot_ptq1_0_q8_0_ptr(weight_row.as_ptr(), act_data.as_ptr(), n) }
}

/// # Safety
/// w_ptr valid for n/128*28 bytes, a_ptr valid for n/32*34 bytes.
///
/// Dispatches to the AVX2 kernel when the CPU supports it (the scalar path is
/// kept for other architectures and as the reference implementation).
pub unsafe fn dot_ptq1_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_available() {
            return dot_ptq1_0_q8_0_avx512(w_ptr, a_ptr, n);
        }
        if avx2_available() {
            return dot_ptq1_0_q8_0_avx2(w_ptr, a_ptr, n);
        }
    }
    dot_ptq1_0_q8_0_ptr_scalar(w_ptr, a_ptr, n)
}

#[cfg(target_arch = "x86_64")]
static AVX512_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// Cached AVX-512 (F + BW + VL) probe; also requires a full-width 512-bit
/// datapath (see [`crate::avx512_full_width`]).
#[cfg(target_arch = "x86_64")]
#[inline]
fn avx512_available() -> bool {
    use std::sync::atomic::Ordering;
    let v = AVX512_STATE.load(Ordering::Relaxed);
    if v < 2 {
        return v == 1;
    }
    let ok = crate::avx512_full_width() && std::arch::is_x86_feature_detected!("avx512vl");
    AVX512_STATE.store(if ok { 1 } else { 0 }, Ordering::Relaxed);
    ok
}

/// Name of the kernel [`dot_ptq1_0_q8_0_ptr`] selects on this machine.
///
/// Reports the choice without duplicating the predicates, so a benchmark can
/// verify dispatch instead of inferring it from timing.
pub fn kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_available() {
            return "avx512";
        }
        if avx2_available() {
            return "avx2";
        }
    }
    "scalar"
}

#[cfg(target_arch = "x86_64")]
static AVX2_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// Cached AVX2 capability probe (2 = unknown, 1 = yes, 0 = no).
#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_available() -> bool {
    use std::sync::atomic::Ordering;
    let v = AVX2_STATE.load(Ordering::Relaxed);
    if v < 2 {
        return v == 1;
    }
    let ok = std::arch::is_x86_feature_detected!("avx2");
    AVX2_STATE.store(if ok { 1 } else { 0 }, Ordering::Relaxed);
    ok
}

/// AVX-512 PTQ1_0 x Q8_0 dot (AVX512F + AVX512BW).
///
/// Same algorithm as the AVX2 kernel but 32 lanes wide: all 24 packed `qs` bytes
/// are decoded in a single 512-bit register per trit slot, and each 32-element
/// Q8_0 sub-block is widened, multiplied and accumulated with one vector each.
/// This cuts roughly a third of the instructions per 128 weights.
///
/// # Safety
/// Same contract as the scalar version.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn dot_ptq1_0_q8_0_avx512(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let blocks = n / BLOCK_SIZE;
    if blocks == 0 {
        return 0.0;
    }
    let mask_ff = _mm512_set1_epi16(0xFF);
    let three = _mm512_set1_epi16(3);
    let one = _mm512_set1_epi16(1);
    let mut wvals = [0i8; BLOCK_SIZE];
    let wp = wvals.as_mut_ptr();
    let mut sum = 0.0f32;

    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([*w_ptr.add(bo + 26), *w_ptr.add(bo + 27)]).to_f32();

        // 24 qs bytes, zero-extended into 32 u16 lanes.
        let raw = _mm256_maskz_loadu_epi8(0x00FF_FFFFu32 as __mmask32, w_ptr.add(bo) as *const i8);
        let bytes16 = _mm512_cvtepu8_epi16(raw);
        for (nn, &p3) in POW3.iter().enumerate() {
            let k = _mm512_set1_epi16(p3 as i16);
            let t = _mm512_and_si512(_mm512_mullo_epi16(bytes16, k), mask_ff);
            let v = _mm512_sub_epi16(_mm512_srli_epi16(_mm512_mullo_epi16(t, three), 8), one);
            // lanes 0..16 -> elements nn*16..nn*16+16, lanes 16..24 -> 80+nn*8..
            let packed = _mm512_cvtepi16_epi8(v);
            _mm_storeu_si128(
                wp.add(nn * 16) as *mut __m128i,
                _mm256_castsi256_si128(packed),
            );
            _mm_storel_epi64(
                wp.add(80 + nn * 8) as *mut __m128i,
                _mm256_extracti128_si256(packed, 1),
            );
        }
        // qh: 2 bytes x 4 trits, interleaved as [n][byte] at elements 120..128.
        let qh0 = *w_ptr.add(bo + QS_BYTES) as usize;
        let qh1 = *w_ptr.add(bo + QS_BYTES + 1) as usize;
        for (nn, (&t0, &t1)) in QH_TRIT[qh0].iter().zip(QH_TRIT[qh1].iter()).enumerate() {
            *wp.add(120 + 2 * nn) = t0;
            *wp.add(121 + 2 * nn) = t1;
        }

        let mut acc = _mm512_setzero_ps();
        for s in 0..4 {
            let a_off = (b * 4 + s) * 34;
            let a_scale = f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let w16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(wp.add(s * 32) as *const __m256i));
            let a16 =
                _mm512_cvtepi8_epi16(_mm256_loadu_si256(a_ptr.add(a_off + 2) as *const __m256i));
            let p = _mm512_madd_epi16(w16, a16); // 16 x i32 pairwise sums
            acc = _mm512_add_ps(
                acc,
                _mm512_mul_ps(_mm512_cvtepi32_ps(p), _mm512_set1_ps(a_scale)),
            );
        }
        let mut lanes = [0.0f32; 16];
        _mm512_storeu_ps(lanes.as_mut_ptr(), acc);
        let block_acc: f32 = lanes.iter().sum();
        sum += block_acc * scale;
    }
    sum
}

/// AVX2 PTQ1_0 x Q8_0 dot.
///
/// Two hot spots are vectorized: the base-3 trit decode (5 trits per packed
/// byte, computed with multiply/mask/shift instead of per-element table loads)
/// and the ternary dot itself (trits are widened to i16 and multiplied with
/// `vpmaddwd`, which avoids `vpsignb`'s negation overflow on q8 = -128).
///
/// # Safety
/// Same contract as the scalar version.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_ptq1_0_q8_0_avx2(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let blocks = n / BLOCK_SIZE;
    if blocks == 0 {
        return 0.0;
    }
    let mask_ff = _mm_set1_epi16(0xFF);
    let three = _mm_set1_epi16(3);
    let one = _mm_set1_epi16(1);
    let mut wvals = [0i8; BLOCK_SIZE];
    let wp = wvals.as_mut_ptr();
    let mut sum = 0.0f32;

    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([*w_ptr.add(bo + 26), *w_ptr.add(bo + 27)]).to_f32();

        // Decode the 24 qs bytes: 16-byte stage (elements 0..80, stride 16 per
        // trit slot) and 8-byte stage (elements 80..120, stride 8).
        let qs = _mm_loadu_si128(w_ptr.add(bo) as *const __m128i);
        let qh_bytes = _mm_loadl_epi64(w_ptr.add(bo + 16) as *const __m128i);
        let zv = _mm_setzero_si128();
        let a_lo = _mm_unpacklo_epi8(qs, zv);
        let a_hi = _mm_unpackhi_epi8(qs, zv);
        let b_lo = _mm_unpacklo_epi8(qh_bytes, zv);
        let b_hi = _mm_unpackhi_epi8(qh_bytes, zv);
        for (nn, &p3) in POW3.iter().enumerate() {
            let k = _mm_set1_epi16(p3 as i16);
            let t_lo = _mm_and_si128(_mm_mullo_epi16(a_lo, k), mask_ff);
            let t_hi = _mm_and_si128(_mm_mullo_epi16(a_hi, k), mask_ff);
            let v_lo = _mm_sub_epi16(_mm_srli_epi16(_mm_mullo_epi16(t_lo, three), 8), one);
            let v_hi = _mm_sub_epi16(_mm_srli_epi16(_mm_mullo_epi16(t_hi, three), 8), one);
            _mm_storeu_si128(wp.add(nn * 16) as *mut __m128i, _mm_packs_epi16(v_lo, v_hi));

            // stage B: 8 bytes cover elements 80..120 with stride 8
            let u_lo = _mm_and_si128(_mm_mullo_epi16(b_lo, k), mask_ff);
            let u_hi = _mm_and_si128(_mm_mullo_epi16(b_hi, k), mask_ff);
            let w_lo = _mm_sub_epi16(_mm_srli_epi16(_mm_mullo_epi16(u_lo, three), 8), one);
            let w_hi = _mm_sub_epi16(_mm_srli_epi16(_mm_mullo_epi16(u_hi, three), 8), one);
            _mm_storel_epi64(
                wp.add(80 + nn * 8) as *mut __m128i,
                _mm_packs_epi16(w_lo, w_hi),
            );
        }
        // qh: 2 bytes x 4 trits, interleaved as [n][byte] at elements 120..128.
        let qh0 = *w_ptr.add(bo + QS_BYTES) as usize;
        let qh1 = *w_ptr.add(bo + QS_BYTES + 1) as usize;
        for (nn, (&t0, &t1)) in QH_TRIT[qh0].iter().zip(QH_TRIT[qh1].iter()).enumerate() {
            *wp.add(120 + 2 * nn) = t0;
            *wp.add(121 + 2 * nn) = t1;
        }

        // Ternary dot, 4 Q8_0 sub-blocks of 32 elements.
        let mut acc = _mm256_setzero_ps();
        for s in 0..4 {
            let a_off = (b * 4 + s) * 34;
            let a_scale = f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let w32 = _mm256_loadu_si256(wp.add(s * 32) as *const __m256i);
            let a32 = _mm256_loadu_si256(a_ptr.add(a_off + 2) as *const __m256i);
            let wl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w32));
            let wh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w32, 1));
            let al = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a32));
            let ah = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a32, 1));
            let p = _mm256_add_epi32(_mm256_madd_epi16(wl, al), _mm256_madd_epi16(wh, ah));
            acc = _mm256_add_ps(
                acc,
                _mm256_mul_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(a_scale)),
            );
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let block_acc: f32 = lanes.iter().sum();
        sum += block_acc * scale;
    }
    sum
}

/// Scalar reference implementation (non-x86 targets and verification).
///
/// # Safety
/// w_ptr valid for n/128*28 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_ptq1_0_q8_0_ptr_scalar(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    let blocks = n / BLOCK_SIZE;
    let mut wvals = [0i8; BLOCK_SIZE];
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let bo = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([*w_ptr.add(bo + 26), *w_ptr.add(bo + 27)]).to_f32();

        // Walk the qs stages (32,16,8 over 24 bytes) then qh, filling wvals in order.
        let qs = std::slice::from_raw_parts(w_ptr.add(bo), QS_BYTES);
        let qh0 = *w_ptr.add(bo + QS_BYTES) as usize;
        let qh1 = *w_ptr.add(bo + QS_BYTES + 1) as usize;
        let mut out_off = 0usize;
        let mut j = 0usize;
        for c in [32usize, 16, 8] {
            while j + c <= QS_BYTES {
                for (m, &byte) in qs[j..j + c].iter().enumerate() {
                    for (nn, &trit) in TRIT_N[byte as usize].iter().enumerate() {
                        wvals[out_off + nn * c + m] = trit;
                    }
                }
                out_off += 5 * c;
                j += c;
            }
        }
        for (nn, (&t0, &t1)) in QH_TRIT[qh0].iter().zip(QH_TRIT[qh1].iter()).enumerate() {
            let base = out_off + nn * QH_BYTES;
            wvals[base] = t0;
            wvals[base + 1] = t1;
        }
        debug_assert_eq!(out_off + 4 * QH_BYTES, BLOCK_SIZE);

        let base = b * BLOCK_SIZE;
        let mut block_acc = 0.0f32;
        for sub in 0..4 {
            let a_off = (b * 4 + sub) * 34;
            let a_scale = f16::from_le_bytes([*a_ptr.add(a_off), *a_ptr.add(a_off + 1)]).to_f32();
            let mut dot: i32 = 0;
            for i in 0..32 {
                dot += wvals[sub * 32 + i] as i32 * (*a_ptr.add(a_off + 2 + i) as i8 as i32);
            }
            let _ = base;
            block_acc += dot as f32 * a_scale;
        }
        sum += block_acc * scale;
    }
    sum
}

/// Quantize f32 to PTQ1_0 (reference quantize_row_ptq1_0_ref). Used by round-trip tests.
pub fn quantize(src: &[f32], dst: &mut Vec<u8>) {
    let blocks = src.len() / BLOCK_SIZE;
    dst.reserve(blocks * BLOCK_BYTES);
    for b in 0..blocks {
        let x = &src[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut block = [0u8; BLOCK_BYTES];
        let d16 = f16::from_f32(d).to_le_bytes();
        block[26] = d16[0];
        block[27] = d16[1];

        let mut x_off = 0usize;
        let mut j = 0usize;
        for c in [32usize, 16, 8] {
            while j + c <= QS_BYTES {
                for m in 0..c {
                    let mut q: u8 = 0;
                    for n in 0..5 {
                        let xi = (x[x_off + m + n * c] * id).round() as i32 + 1; // 0..2
                        q = q.wrapping_mul(3).wrapping_add(xi as u8);
                    }
                    // ceil-round to 256/243 grid
                    block[j + m] = (q as u16 * 256).div_ceil(243) as u8;
                }
                x_off += 5 * c;
                j += c;
            }
        }
        for h in 0..QH_BYTES {
            let mut q: u8 = 0;
            for m in 0..4 {
                let xi = (x[x_off + h + m * QH_BYTES] * id).round() as i32 + 1;
                q = q.wrapping_mul(3).wrapping_add(xi as u8);
            }
            q = q.wrapping_mul(3); // shift first trit to most significant position
            block[QS_BYTES + h] = (q as u16 * 256).div_ceil(243) as u8;
        }
        dst.extend_from_slice(&block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_ternary_values() {
        // Build one block via quantize, then check dequant round-trip.
        let mut src = vec![0.0f32; BLOCK_SIZE];
        src[0] = -1.0;
        src[1] = 1.0;
        src[2] = 0.4; // rounds to 0 (round-half-away-from-zero, same as C llroundf)
        src[3] = -0.4; // rounds to 0
        src[127] = 1.0;
        let mut data = Vec::new();
        quantize(&src, &mut data);
        assert_eq!(data.len(), BLOCK_BYTES);

        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequantize(&data, &mut out);
        assert!((out[0] - (-1.0)).abs() < 1e-6, "out[0]={}", out[0]);
        assert!((out[1] - 1.0).abs() < 1e-6, "out[1]={}", out[1]);
        assert!(out[2].abs() < 1e-6, "out[2]={}", out[2]);
        assert!(out[3].abs() < 1e-6, "out[3]={}", out[3]);
        assert!(out[4].abs() < 1e-6, "out[4]={}", out[4]);
        assert!((out[127] - 1.0).abs() < 1e-6, "out[127]={}", out[127]);
    }

    #[test]
    fn test_roundtrip_all_patterns() {
        // Every byte pattern for qs/qh through dequant must land in {-1,0,+1}*scale.
        let mut data = vec![0u8; BLOCK_BYTES];
        for (i, b) in data.iter_mut().enumerate().take(QS_BYTES) {
            *b = (i * 37 + 11) as u8;
        }
        data[QS_BYTES] = 0xAB;
        data[QS_BYTES + 1] = 0xC3;
        let d = f16::from_f32(0.25);
        let d16 = d.to_le_bytes();
        data[26] = d16[0];
        data[27] = d16[1];

        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequantize(&data, &mut out);
        for (i, &v) in out.iter().enumerate() {
            let q = (v / d.to_f32()).round();
            assert!(
                (-1.0..=1.0).contains(&q) && (q - v / d.to_f32()).abs() < 1e-6,
                "out[{}]={} not ternary",
                i,
                v
            );
        }
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_dot_matches_dequant() {
        // Random ternary weights vs f32 activations: dot_ptq1_0_f32 == dot of dequantized.
        let mut data = vec![0u8; BLOCK_BYTES * 4];
        for (i, b) in data.iter_mut().enumerate() {
            *b = if i % 28 == 26 || i % 28 == 27 {
                // pseudo-random small positive scales
                ((i / 28) as u16 * 0x1234 % 0x3800 + 0x0400).to_le_bytes()[0]
            } else {
                (i * 91 + 7) as u8
            };
        }
        // Fix scales to known f16 values
        for b in 0..4 {
            let s = f16::from_f32(0.5 + b as f32 * 0.25).to_le_bytes();
            data[b * BLOCK_BYTES + 26] = s[0];
            data[b * BLOCK_BYTES + 27] = s[1];
        }
        let vec: Vec<f32> = (0..4 * BLOCK_SIZE)
            .map(|i| ((i as f32) * 0.173).sin())
            .collect();

        let mut deq = vec![0.0f32; 4 * BLOCK_SIZE];
        dequantize(&data, &mut deq);
        let expected: f32 = deq.iter().zip(vec.iter()).map(|(&w, &x)| w * x).sum();
        let got = dot_ptq1_0_f32(&data, &vec, 4 * BLOCK_SIZE);
        assert!(
            (got - expected).abs() < 1e-3 * expected.abs().max(1.0),
            "dot={} expected={}",
            got,
            expected
        );
    }

    #[test]
    fn test_ternary_weights_sum() {
        // All weights = +1 (amax=1.0 so d=1.0): dot with all-ones activations = 128.
        let vec = vec![1.0f32; BLOCK_SIZE];
        let mut qdata = Vec::new();
        quantize(&vec, &mut qdata);
        let mut deq = vec![0.0f32; BLOCK_SIZE];
        dequantize(&qdata, &mut deq);
        assert!(
            deq.iter().all(|&v| (v - 1.0).abs() < 1e-6),
            "deq[0]={}",
            deq[0]
        );
        let dot = dot_ptq1_0_f32(&qdata, &vec, BLOCK_SIZE);
        assert!((dot - 128.0).abs() < 0.1, "dot={}", dot);
    }
}
