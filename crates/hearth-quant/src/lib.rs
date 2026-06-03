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

/// # Safety
/// w_ptr valid for n/128*34 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q2_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    q2_0::dot_q2_0_q8_0_ptr(w_ptr, a_ptr, n)
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

/// MSVC-compiled scalar kernel (FFI). This is the reference's ggml_vec_dot_q1_0_q8_0_generic,
/// compiled by MSVC which auto-vectorizes it to efficient 128-bit XMM code.
/// Falls back to the LLVM kernel if not available.
///
/// # Safety
/// w_ptr valid for n/128*18 bytes, a_ptr valid for n/32*34 bytes.
#[cfg(msvc_kernel)]
pub unsafe fn dot_q1_0g128_q8_0_ptr_msvc(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    extern "C" {
        fn dot_q1_0_q8_0_msvc(
            w_ptr: *const core::ffi::c_void,
            a_ptr: *const core::ffi::c_void,
            n: core::ffi::c_int,
        ) -> f32;
    }
    dot_q1_0_q8_0_msvc(
        w_ptr as *const core::ffi::c_void,
        a_ptr as *const core::ffi::c_void,
        n as core::ffi::c_int,
    )
}

#[cfg(test)]
mod msvc_tests {
    #[cfg(msvc_kernel)]
    #[test]
    fn msvc_kernel_correctness() {
        use half::f16;
        // Create test data: 1 Q1_0 block (128 elements, 18 bytes) + 4 Q8_0 sub-blocks (34 bytes each)
        let n = 128;
        let mut w = vec![0u8; 18];
        let mut a = vec![0u8; 4 * 34];

        // Set weight scale
        w[0] = 0x00;
        w[1] = 0x3C; // f16 value 1.0

        // Set weight bits: all 1s (value +1 for each element)
        for i in 2..18 {
            w[i] = 0xFF;
        }

        // Set activation scale for all 4 sub-blocks
        for sub in 0..4 {
            let off = sub * 34;
            a[off] = 0x00;
            a[off + 1] = 0x3C; // f16 value 1.0
                               // Set activation values to 1
            for j in 2..34 {
                a[off + j] = 1i8 as u8;
            }
        }

        let llvm_result = unsafe { crate::dot_q1_0g128_q8_0_ptr(w.as_ptr(), a.as_ptr(), n) };
        let msvc_result = unsafe { crate::dot_q1_0g128_q8_0_ptr_msvc(w.as_ptr(), a.as_ptr(), n) };

        eprintln!("LLVM: {:.6}, MSVC: {:.6}", llvm_result, msvc_result);
        // Both should be 128.0 (128 elements × 1.0 weight × 1.0 activation, scale=1.0)
        assert!(
            (llvm_result - 128.0).abs() < 1.0,
            "LLVM kernel wrong: {}",
            llvm_result
        );
        assert!(
            (msvc_result - 128.0).abs() < 1.0,
            "MSVC kernel wrong: {}",
            msvc_result
        );
    }

    #[cfg(msvc_kernel)]
    #[test]
    fn msvc_kernel_bench() {
        use std::time::Instant;
        let n = 2048; // 16 blocks = one matmul row
        let mut w = vec![0u8; n / 128 * 18];
        let mut a = vec![0u8; n / 32 * 34];
        for b in 0..w.len() {
            w[b] = (b % 255) as u8;
        }
        for b in 0..a.len() {
            a[b] = (b % 255) as u8;
        }

        let iters = 100_000;
        let t0 = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..iters {
            sum += unsafe { crate::dot_q1_0g128_q8_0_ptr_msvc(w.as_ptr(), a.as_ptr(), n) };
        }
        let msvc_elapsed = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        for _ in 0..iters {
            sum += unsafe { crate::dot_q1_0g128_q8_0_ptr(w.as_ptr(), a.as_ptr(), n) };
        }
        let llvm_elapsed = t0.elapsed().as_secs_f64();

        eprintln!("sum={:.6} (avoid optimize-away)", sum);
        eprintln!(
            "MSVC: {:.3}µs/call, LLVM: {:.3}µs/call",
            msvc_elapsed * 1_000_000.0 / iters as f64,
            llvm_elapsed * 1_000_000.0 / iters as f64
        );
        eprintln!("Ratio: {:.2}x", msvc_elapsed / llvm_elapsed);
    }
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
