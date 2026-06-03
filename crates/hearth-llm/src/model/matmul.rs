use hearth_compute::QuantKind;
use hearth_gguf::GgmlDType;
use hearth_quant;
use rayon::prelude::*;

use super::LlamaModel;

/// Select the fastest Q1_0 dot kernel for the current build configuration.
/// MSVC-compiled scalar kernel (SSE2 auto-vectorized) beats LLVM AVX2 by ~1.3-1.5×.
#[cfg(feature = "msvc-kernel")]
unsafe fn dot_q1_0g128_fast(w: *const u8, a: *const u8, n: usize) -> f32 {
    hearth_quant::dot_q1_0g128_q8_0_ptr_msvc(w, a, n)
}

#[cfg(not(feature = "msvc-kernel"))]
unsafe fn dot_q1_0g128_fast(w: *const u8, a: *const u8, n: usize) -> f32 {
    hearth_quant::dot_q1_0g128_q8_0_ptr(w, a, n)
}

impl LlamaModel {
    pub(crate) fn matmul(
        &self,
        weight_name: &str,
        x: &[f32],
        out: &mut [f32],
        row_buf: &mut Vec<f32>,
        layer: usize,
        x_q8: Option<&[u8]>,
    ) -> Result<(), String> {
        let entry = self
            .tensors
            .get(weight_name)
            .ok_or_else(|| format!("Tensor not found: {}", weight_name))?;

        let n_cols = entry.n_cols();
        let n_rows = entry.n_rows();
        if n_cols == 0 || x.len() != n_cols {
            return Err(format!(
                "Matmul dim mismatch: {} cols={} x.len={}",
                weight_name,
                n_cols,
                x.len()
            ));
        }

        let use_gpu = layer < self.gpu_layers;

        if use_gpu {
            if let Some(ref gpu) = self.gpu {
                let w_buf = {
                    let pool = gpu.pool.lock().unwrap();
                    pool.get(weight_name).cloned()
                };
                if let Some(w_buf) = w_buf {
                    let quant_kind = QuantKind::from_gguf_dtype(entry.dtype);
                    if let Some(kind) = quant_kind {
                        if gpu.has_dequant(&kind) {
                            let t0 = std::time::Instant::now();
                            let x_buf = gpu.upload_f32(x, "x_vec");
                            let t1 = std::time::Instant::now();
                            if let Some(result) = gpu.dequant_matmul_fused(
                                &w_buf,
                                &kind,
                                &x_buf,
                                n_rows as u32,
                                1,
                                n_cols as u32,
                            ) {
                                let t2 = std::time::Instant::now();
                                let out_f32 = gpu.readback_f32(&result, n_rows);
                                let t3 = std::time::Instant::now();
                                let upload_us = t1.duration_since(t0).as_micros();
                                let compute_us = t2.duration_since(t1).as_micros();
                                let readback_us = t3.duration_since(t2).as_micros();
                                eprintln!("  [GPU {}] upload={}us compute={}us readback={}us (n_rows={}, n_cols={})",
                                    weight_name, upload_us, compute_us, readback_us, n_rows, n_cols);
                                out[..n_rows].copy_from_slice(&out_f32[..n_rows]);
                                return Ok(());
                            }
                        }
                    } else if entry.dtype == GgmlDType::F16 && gpu.has_matmul_f16() {
                        let t0 = std::time::Instant::now();
                        let x_buf = gpu.upload_f32(x, "x_vec");
                        let t1 = std::time::Instant::now();
                        if let Some(result) =
                            gpu.matmul_f16(&w_buf, &x_buf, n_rows as u32, 1, n_cols as u32)
                        {
                            let t2 = std::time::Instant::now();
                            let out_f32 = gpu.readback_f32(&result, n_rows);
                            let t3 = std::time::Instant::now();
                            eprintln!(
                                "  [GPU {}] upload={}us compute={}us readback={}us",
                                weight_name,
                                t1.duration_since(t0).as_micros(),
                                t2.duration_since(t1).as_micros(),
                                t3.duration_since(t2).as_micros()
                            );
                            out[..n_rows].copy_from_slice(&out_f32[..n_rows]);
                            return Ok(());
                        }
                    } else if entry.dtype == GgmlDType::F32 {
                        let t0 = std::time::Instant::now();
                        let x_buf = gpu.upload_f32(x, "x_vec");
                        let t1 = std::time::Instant::now();
                        if let Some(result) =
                            gpu.mat_vec(&w_buf, &x_buf, n_rows as u32, n_cols as u32)
                        {
                            let t2 = std::time::Instant::now();
                            eprintln!(
                                "  [GPU {}] upload={}us compute+readback={}us",
                                weight_name,
                                t1.duration_since(t0).as_micros(),
                                t2.duration_since(t1).as_micros()
                            );
                            out[..n_rows].copy_from_slice(&result[..n_rows]);
                            return Ok(());
                        }
                    }
                }
            }
        }

        match entry.dtype {
            GgmlDType::Q2_0 => {
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let w_base = entry.data.as_ptr() as usize;
                let row_bytes = entry.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                self.pool.par_dot_rows(
                    n_rows,
                    w_base,
                    a_ptr,
                    out_ptr,
                    row_bytes,
                    n_cols,
                    hearth_quant::dot_q2_0_q8_0_ptr,
                );
            }
            GgmlDType::Q1_0_G128 => {
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let w_base = entry.data.as_ptr() as usize;
                let row_bytes = entry.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                self.pool.par_dot_rows(
                    n_rows,
                    w_base,
                    a_ptr,
                    out_ptr,
                    row_bytes,
                    n_cols,
                    dot_q1_0g128_fast,
                );
            }
            GgmlDType::Q1_0 => {
                // Prism Q1_0 (type 41) uses 128-element, 18-byte blocks (same as Q1_0_G128)
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let w_base = entry.data.as_ptr() as usize;
                let row_bytes = entry.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                self.pool.par_dot_rows(
                    n_rows,
                    w_base,
                    a_ptr,
                    out_ptr,
                    row_bytes,
                    n_cols,
                    dot_q1_0g128_fast,
                );
            }
            GgmlDType::Q8_0 => {
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let w_base = entry.data.as_ptr() as usize;
                let row_bytes = entry.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                self.pool.par_dot_rows(
                    n_rows,
                    w_base,
                    a_ptr,
                    out_ptr,
                    row_bytes,
                    n_cols,
                    hearth_quant::dot_q8_0_q8_0_ptr,
                );
            }
            GgmlDType::Q4_K => {
                let rows: Vec<usize> = (0..n_rows).collect();
                let result: Vec<f32> = rows
                    .par_iter()
                    .map(|&row| {
                        let w = entry.row_data(row);
                        hearth_quant::dot_q4_k(w, x, n_cols)
                    })
                    .collect();
                out[..n_rows].copy_from_slice(&result);
            }
            GgmlDType::Q4_0 => {
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let rows: Vec<usize> = (0..n_rows).collect();
                let result: Vec<f32> = rows
                    .par_iter()
                    .map(|&row| {
                        let w = entry.row_data(row);
                        hearth_quant::dot_q4_0_q8_0(w, q8, n_cols)
                    })
                    .collect();
                out[..n_rows].copy_from_slice(&result);
            }
            GgmlDType::Q4_1 => {
                let rows: Vec<usize> = (0..n_rows).collect();
                let result: Vec<f32> = rows
                    .par_iter()
                    .map(|&row| {
                        let w = entry.row_data(row);
                        hearth_quant::dot_q4_1(w, x, n_cols)
                    })
                    .collect();
                out[..n_rows].copy_from_slice(&result);
            }
            GgmlDType::Q5_0 => {
                let rows: Vec<usize> = (0..n_rows).collect();
                let result: Vec<f32> = rows
                    .par_iter()
                    .map(|&row| {
                        let w = entry.row_data(row);
                        hearth_quant::dot_q5_0(w, x, n_cols)
                    })
                    .collect();
                out[..n_rows].copy_from_slice(&result);
            }
            GgmlDType::Q5_1 => {
                let rows: Vec<usize> = (0..n_rows).collect();
                let result: Vec<f32> = rows
                    .par_iter()
                    .map(|&row| {
                        let w = entry.row_data(row);
                        hearth_quant::dot_q5_1(w, x, n_cols)
                    })
                    .collect();
                out[..n_rows].copy_from_slice(&result);
            }
            GgmlDType::Q6_K => {
                let rows: Vec<usize> = (0..n_rows).collect();
                let result: Vec<f32> = rows
                    .par_iter()
                    .map(|&row| {
                        let w = entry.row_data(row);
                        hearth_quant::dot_q6_k(w, x, n_cols)
                    })
                    .collect();
                out[..n_rows].copy_from_slice(&result);
            }
            _ => {
                if row_buf.len() < n_cols * n_rows {
                    row_buf.resize(n_cols * n_rows, 0.0f32);
                }

                hearth_quant::dequantize(entry.dtype, &entry.data, &mut row_buf[..n_cols * n_rows])
                    .map_err(|e| format!("Dequant {}: {}", weight_name, e))?;

                unsafe {
                    matrixmultiply::sgemm(
                        n_rows,
                        1,
                        n_cols,
                        1.0,
                        row_buf.as_ptr(),
                        n_cols as isize,
                        1,
                        x.as_ptr(),
                        1,
                        1,
                        0.0,
                        out.as_mut_ptr(),
                        1,
                        1,
                    );
                }
            }
        }

        Ok(())
    }

    /// Fused matmul for two weight tensors sharing the same input and Q8 activation.
    /// Processes both weight matrices in a single parallel pass over combined rows.
    /// Only Q1_0_G128 path is fused; other dtypes fall back to two separate matmul calls.
    pub(crate) fn matmul_fused2(
        &self,
        weight_a: &str,
        weight_b: &str,
        x: &[f32],
        out_a: &mut [f32],
        out_b: &mut [f32],
        row_buf: &mut Vec<f32>,
        layer: usize,
        x_q8: Option<&[u8]>,
    ) -> Result<(), String> {
        let entry_a = self
            .tensors
            .get(weight_a)
            .ok_or_else(|| format!("Tensor not found: {}", weight_a))?;
        let entry_b = self
            .tensors
            .get(weight_b)
            .ok_or_else(|| format!("Tensor not found: {}", weight_b))?;

        let n_cols = entry_a.n_cols();
        if n_cols != entry_b.n_cols() || n_cols == 0 || x.len() != n_cols {
            return Err(format!(
                "Matmul fused2 dim mismatch: {} cols={} {} cols={} x.len={}",
                weight_a,
                n_cols,
                weight_b,
                entry_b.n_cols(),
                x.len()
            ));
        }
        let n_rows_a = entry_a.n_rows();
        let n_rows_b = entry_b.n_rows();

        match entry_a.dtype {
            GgmlDType::Q1_0_G128 | GgmlDType::Q1_0 => {
                if !matches!(entry_b.dtype, GgmlDType::Q1_0_G128 | GgmlDType::Q1_0) {
                    return Err(format!(
                        "Fused matmul dtype mismatch: {} {:?} vs {} {:?}",
                        weight_a, entry_a.dtype, weight_b, entry_b.dtype
                    ));
                }
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let w_base_a = entry_a.data.as_ptr() as usize;
                let w_base_b = entry_b.data.as_ptr() as usize;
                let row_bytes = entry_a.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_a_ptr = out_a.as_mut_ptr() as usize;
                let out_b_ptr = out_b.as_mut_ptr() as usize;
                self.pool.par_dot_rows(
                    n_rows_a,
                    w_base_a,
                    a_ptr,
                    out_a_ptr,
                    row_bytes,
                    n_cols,
                    dot_q1_0g128_fast,
                );
                self.pool.par_dot_rows(
                    n_rows_b,
                    w_base_b,
                    a_ptr,
                    out_b_ptr,
                    row_bytes,
                    n_cols,
                    dot_q1_0g128_fast,
                );
            }
            _ => {
                self.matmul(weight_a, x, out_a, row_buf, layer, x_q8)?;
                self.matmul(weight_b, x, out_b, row_buf, layer, x_q8)?;
            }
        }
        Ok(())
    }

    /// Fused QKV matmul: processes Q, K, V weight matrices in a single parallel pass.
    /// Reduces Rayon dispatch overhead and improves activation cache reuse.
    /// All three must share the same n_cols and quantized dtype (Q1_0 or Q1_0_G128).
    pub(crate) fn matmul_fused_qkv(
        &self,
        w_q: &str,
        w_k: &str,
        w_v: &str,
        x: &[f32],
        out_q: &mut [f32],
        out_k: &mut [f32],
        out_v: &mut [f32],
        row_buf: &mut Vec<f32>,
        layer: usize,
        x_q8: Option<&[u8]>,
    ) -> Result<(), String> {
        let eq = self
            .tensors
            .get(w_q)
            .ok_or_else(|| format!("Tensor not found: {}", w_q))?;
        let ek = self
            .tensors
            .get(w_k)
            .ok_or_else(|| format!("Tensor not found: {}", w_k))?;
        let ev = self
            .tensors
            .get(w_v)
            .ok_or_else(|| format!("Tensor not found: {}", w_v))?;

        let n_cols = eq.n_cols();
        if n_cols != ek.n_cols() || n_cols != ev.n_cols() || n_cols == 0 || x.len() != n_cols {
            return Err(format!(
                "QKV fused dim mismatch: cols {} {} {} x.len={}",
                n_cols,
                ek.n_cols(),
                ev.n_cols(),
                x.len()
            ));
        }
        let nr_q = eq.n_rows();
        let nr_k = ek.n_rows();
        let nr_v = ev.n_rows();

        match eq.dtype {
            GgmlDType::Q1_0_G128 | GgmlDType::Q1_0 => {
                if !matches!(ek.dtype, GgmlDType::Q1_0_G128 | GgmlDType::Q1_0)
                    || !matches!(ev.dtype, GgmlDType::Q1_0_G128 | GgmlDType::Q1_0)
                {
                    return Err("QKV fused dtype mismatch".into());
                }
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let wq = eq.data.as_ptr() as usize;
                let wk = ek.data.as_ptr() as usize;
                let wv = ev.data.as_ptr() as usize;
                let row_bytes = eq.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let oq = out_q.as_mut_ptr() as usize;
                let ok = out_k.as_mut_ptr() as usize;
                let ov = out_v.as_mut_ptr() as usize;
                self.pool
                    .par_dot_rows(nr_q, wq, a_ptr, oq, row_bytes, n_cols, dot_q1_0g128_fast);
                self.pool
                    .par_dot_rows(nr_k, wk, a_ptr, ok, row_bytes, n_cols, dot_q1_0g128_fast);
                self.pool
                    .par_dot_rows(nr_v, wv, a_ptr, ov, row_bytes, n_cols, dot_q1_0g128_fast);
            }
            _ => {
                self.matmul(w_q, x, out_q, row_buf, layer, x_q8)?;
                self.matmul(w_k, x, out_k, row_buf, layer, x_q8)?;
                self.matmul(w_v, x, out_v, row_buf, layer, x_q8)?;
            }
        }
        Ok(())
    }

    fn quantize_act(&self, x: &[f32]) -> Vec<u8> {
        let mut q8 = Vec::with_capacity(x.len().div_ceil(32) * 34);
        hearth_quant::quantize_q8_0(x, &mut q8);
        q8
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_batch(
        &self,
        weight_name: &str,
        input: &[f32],
        out: &mut [f32],
        row_buf: &mut Vec<f32>,
        _n_rows: usize,
        n_cols: usize,
        seq_len: usize,
    ) -> Result<(), String> {
        let entry = self
            .tensors
            .get(weight_name)
            .ok_or_else(|| format!("Tensor not found: {}", weight_name))?;

        let actual_rows = entry.n_rows();
        let actual_cols = entry.n_cols();
        if actual_cols == 0 || n_cols != actual_cols {
            return Err(format!(
                "Matmul_batch dim mismatch: {} cols={} expected_cols={}",
                weight_name, actual_cols, n_cols
            ));
        }
        let m = actual_rows;
        let k = actual_cols;

        if matches!(
            entry.dtype,
            GgmlDType::Q2_0 | GgmlDType::Q1_0 | GgmlDType::Q1_0_G128
        ) {
            let q8_size = k.div_ceil(32) * 34;
            let dot_fn = match entry.dtype {
                GgmlDType::Q2_0 => hearth_quant::dot_q2_0_q8_0_ptr,
                _ => dot_q1_0g128_fast,
            };
            let mut q8_buf = Vec::with_capacity(seq_len * q8_size);
            for s in 0..seq_len {
                hearth_quant::quantize_q8_0(&input[s * k..(s + 1) * k], &mut q8_buf);
            }
            self.pool.par_dot_rows_batched(
                m,
                seq_len,
                q8_size,
                entry.data.as_ptr() as usize,
                q8_buf.as_ptr() as usize,
                out.as_mut_ptr() as usize,
                entry.dtype.byte_size(k),
                k,
                dot_fn,
            );
        } else {
            for s in 0..seq_len {
                self.matmul(
                    weight_name,
                    &input[s * k..(s + 1) * k],
                    &mut out[s * m..(s + 1) * m],
                    row_buf,
                    0,
                    None,
                )?;
            }
        }
        Ok(())
    }
}
