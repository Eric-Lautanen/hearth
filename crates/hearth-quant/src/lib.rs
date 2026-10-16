pub mod pq2_0;
pub mod ptq1_0;
pub mod q1_0g128;
pub mod q2_0;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q8_0;
pub mod qk_quant;

use hearth_gguf::GgmlDType;

/// Whether 512-bit vector kernels are expected to beat their 256-bit versions
/// on this CPU.
///
/// AMD family 19h (Zen 3 and Zen 4) splits every 512-bit operation into two
/// 256-bit micro-ops, so wider registers buy no throughput while wider-kernel
/// overhead (packing, masked loads) is real: measured on a Ryzen 7 8840HS, the
/// AVX-512 PTQ1_0 dot ran at 1.28 GB/s against 1.76 GB/s for the AVX2 one.
/// Full-width implementations (Intel server parts, Zen 5+) do benefit.
///
/// This is not a blanket "no AVX-512 on Zen 4": the VNNI kernels reduce the
/// front-end work enough to stay ahead there and are gated on the VNNI feature
/// alone (see `q2_0::dot_q2_0_q8_0_ptr`).
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx512_full_width() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(2); // 2 = unknown
    let v = STATE.load(Ordering::Relaxed);
    if v < 2 {
        return v == 1;
    }
    let mut ok = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw");
    if ok && cpu_is_amd_zen4() {
        ok = false;
    }
    STATE.store(if ok { 1 } else { 0 }, Ordering::Relaxed);
    ok
}

/// AMD family 19h (Zen 3 / Zen 4): the only AVX-512 part in it is Zen 4, whose
/// 512-bit datapath is two 256-bit halves.
///
/// `__cpuid` is safe or unsafe depending on the target features the crate is
/// built with, so the lint is allowed rather than the call being rewritten per
/// build configuration.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
fn cpu_is_amd_zen4() -> bool {
    let vendor = unsafe { std::arch::x86_64::__cpuid(0) };
    let is_amd = vendor.ebx == u32::from_le_bytes(*b"Auth")
        && vendor.edx == u32::from_le_bytes(*b"enti")
        && vendor.ecx == u32::from_le_bytes(*b"cAMD");
    if !is_amd {
        return false;
    }
    // CPUID reports the family in a 4-bit field, so everything from 15h up is
    // encoded as base 0xF plus an extended family; Zen 4's 19h arrives as
    // base 15 with extended 0xA and must be reconstructed before comparing.
    let id = unsafe { std::arch::x86_64::__cpuid(1) };
    let base = (id.eax >> 8) & 0xF;
    let ext = (id.eax >> 20) & 0xFF;
    let family = if base == 0xF { base + ext } else { base };
    family == 0x19
}

pub fn dequantize(dtype: GgmlDType, data: &[u8], out: &mut [f32]) -> Result<(), String> {
    match dtype {
        GgmlDType::F32 => {
            let src = bytemuck::cast_slice::<u8, f32>(data);
            let n = out.len().min(src.len());
            out[..n].copy_from_slice(&src[..n]);
        }
        GgmlDType::F16 => {
            let src = bytemuck::cast_slice::<u8, half::f16>(data);
            for (i, &v) in src.iter().enumerate() {
                if i >= out.len() {
                    break;
                }
                out[i] = v.to_f32();
            }
        }
        GgmlDType::BF16 => {
            let chunks = data.chunks_exact(2);
            for (i, chunk) in chunks.enumerate() {
                if i >= out.len() {
                    break;
                }
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = f32::from_bits((bits as u32) << 16);
            }
        }
        GgmlDType::F64 => {
            let src = bytemuck::cast_slice::<u8, f64>(data);
            for (i, &v) in src.iter().enumerate() {
                if i >= out.len() {
                    break;
                }
                out[i] = v as f32;
            }
        }
        GgmlDType::Q8_0 => q8_0::dequantize(data, out),
        GgmlDType::Q4_0 => q4_0::dequantize(data, out),
        GgmlDType::Q4_1 => q4_1::dequantize(data, out),
        GgmlDType::Q5_0 => q5_0::dequantize(data, out),
        GgmlDType::Q5_1 => q5_1::dequantize(data, out),
        GgmlDType::Q4_K => q4_k::dequantize(data, out),
        GgmlDType::Q2_0 => q2_0::dequantize(data, out),
        GgmlDType::PQ2_0 => pq2_0::dequantize(data, out),
        GgmlDType::PTQ1_0 => ptq1_0::dequantize(data, out),
        GgmlDType::Q1_0 | GgmlDType::Q1_0_G128 => q1_0g128::dequantize(data, out),
        GgmlDType::Q2_K => qk_quant::dequantize_q2_k(data, out),
        GgmlDType::Q3_K => qk_quant::dequantize_q3_k(data, out),
        GgmlDType::Q5_K => qk_quant::dequantize_q5_k(data, out),
        GgmlDType::Q6_K => qk_quant::dequantize_q6_k(data, out),
        GgmlDType::Q8_1 => {
            return Err(format!(
                "Dequant for {} not yet implemented - try a Q8_0 or Q4_K model instead",
                dtype.name()
            ));
        }
        GgmlDType::Q8_K => {
            return Err(format!(
                "Dequant for {} not yet implemented - try a Q8_0 or Q4_K model instead",
                dtype.name()
            ));
        }
        _ => {
            return Err(format!(
                "Unsupported quantization type: {} (requires a dequant kernel)",
                dtype.name()
            ));
        }
    }
    Ok(())
}

pub fn dot_q8_0(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q8_0::dot_q8_0_f32(row, vec, n)
}

pub fn dot_q4_0(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q4_0::dot_q4_0_f32(row, vec, n)
}

pub fn dot_q4_1(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q4_1::dot_q4_1_f32(row, vec, n)
}

pub fn dot_q5_0(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q5_0::dot_q5_0_f32(row, vec, n)
}

pub fn dot_q5_1(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q5_1::dot_q5_1_f32(row, vec, n)
}

pub fn dot_q4_k(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q4_k::dot_q4_k_f32(row, vec, n)
}

pub fn dot_q6_k(row: &[u8], vec: &[f32], n: usize) -> f32 {
    qk_quant::dot_q6_k_f32(row, vec, n)
}

pub fn dot_q2_0_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q2_0::dot_q2_0_f32(row, vec, n)
}

pub fn dot_q2_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q2_0::dot_q2_0_q8_0(row, act, n)
}

pub fn dot_pq2_0_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    pq2_0::dot_pq2_0_f32(row, vec, n)
}

pub fn dot_ptq1_0_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    ptq1_0::dot_ptq1_0_f32(row, vec, n)
}

/// # Safety
/// w_ptr valid for n/128*28 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_ptq1_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    ptq1_0::dot_ptq1_0_q8_0_ptr(w_ptr, a_ptr, n)
}

/// # Safety
/// w_ptr valid for n/128*34 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q2_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    q2_0::dot_q2_0_q8_0_ptr(w_ptr, a_ptr, n)
}

/// Paired-token Q2_0 dot: one weight row against two Q8_0 activation rows,
/// sharing the Q2V_I16 weight expansion. Output is bit-identical to two calls to
/// [`dot_q2_0_q8_0_ptr`].
/// # Safety
/// w_ptr valid for n/128*34 bytes, a0_ptr and a1_ptr each valid for n/32*34 bytes.
pub unsafe fn dot_q2_0_q8_0x2_ptr(
    w_ptr: *const u8,
    a0_ptr: *const u8,
    a1_ptr: *const u8,
    n: usize,
) -> (f32, f32) {
    q2_0::dot_q2_0_q8_0x2_ptr(w_ptr, a0_ptr, a1_ptr, n)
}

pub fn dot_q1_0(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_f32(row, vec, n)
}

pub fn dot_q1_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_q8_0(row, act, n)
}

pub fn dot_q1_0g128_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_f32(row, vec, n)
}

pub fn dot_q1_0g128_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_q8_0(row, act, n)
}

/// # Safety
/// w_ptr valid for n/32*6 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q1_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    q1_0g128::dot_q1_0_q8_0_ptr(w_ptr, a_ptr, n)
}

/// # Safety
/// w_ptr valid for n/128*18 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q1_0g128_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_q8_0_ptr(w_ptr, a_ptr, n)
}

/// Paired-token Q1_0_G128 dot: one weight row against two Q8_0 activation rows,
/// sharing the weight bit expansion. Output is bit-identical to two calls to
/// [`dot_q1_0g128_q8_0_ptr`].
/// # Safety
/// w_ptr valid for n/128*18 bytes, a0_ptr and a1_ptr each valid for n/32*34 bytes.
pub unsafe fn dot_q1_0g128_q8_0x2_ptr(
    w_ptr: *const u8,
    a0_ptr: *const u8,
    a1_ptr: *const u8,
    n: usize,
) -> (f32, f32) {
    q1_0g128::dot_q1_0g128_q8_0x2_ptr(w_ptr, a0_ptr, a1_ptr, n)
}

pub fn dot_q4_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q4_0::dot_q4_0_q8_0(row, act, n)
}

pub fn dot_q8_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q8_0::dot_q8_0_q8_0(row, act, n)
}

/// # Safety
/// Both pointers valid for n/32*34 bytes.
pub unsafe fn dot_q8_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    q8_0::dot_q8_0_q8_0_ptr(w_ptr, a_ptr, n)
}

pub fn quantize_q8_0(src: &[f32], dst: &mut Vec<u8>) {
    q8_0::quantize(src, dst)
}

pub fn quantize_q8_0_into(src: &[f32], dst: &mut [u8]) {
    q8_0::quantize_into(src, dst);
}

pub fn quantize_f32(dtype: GgmlDType, src: &[f32], dst: &mut Vec<u8>) {
    match dtype {
        GgmlDType::Q8_0 => q8_0::quantize(src, dst),
        _ => {
            if dtype == GgmlDType::F32 {
                let bytes = bytemuck::cast_slice::<f32, u8>(src);
                dst.extend_from_slice(bytes);
            }
        }
    }
}
