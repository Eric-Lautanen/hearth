use hearth_compute::QuantKind;
use hearth_gguf::GgmlDType;
use hearth_quant;
use rayon::prelude::*;

use super::LlamaModel;

/// The shape of a batched matmul: how wide the weight's rows are, and how many tokens are being
/// pushed through it together.
///
/// `n_cols` is a caller-side expectation that the body checks against the tensor's own column
/// count — that check is what catches a caller that reached for the wrong weight — and `seq_len`
/// is the token count the output rows are sliced by. The weight's row count is *not* here: the
/// body derives it from the tensor, and it used to arrive as an argument nobody read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchShape {
    pub n_cols: usize,
    pub seq_len: usize,
}

/// Parameters for the fused gate+up matmul. Grouped so the call stays
/// under the argument-count lint; field order matches the old signature.
pub(crate) struct FusedMatmul2Params<'a> {
    pub weight_a: &'a str,
    pub weight_b: &'a str,
    pub x: &'a [f32],
    pub out_a: &'a mut [f32],
    pub out_b: &'a mut [f32],
    pub row_buf: &'a mut Vec<f32>,
    pub layer: usize,
    pub x_q8: Option<&'a [u8]>,
}

/// Parameters for the fused QKV matmul. Grouped so the call stays under
/// the argument-count lint; field order matches the old signature.
pub(crate) struct FusedQkvParams<'a> {
    pub w_q: &'a str,
    pub w_k: &'a str,
    pub w_v: &'a str,
    pub x: &'a [f32],
    pub out_q: &'a mut [f32],
    pub out_k: &'a mut [f32],
    pub out_v: &'a mut [f32],
    pub row_buf: &'a mut Vec<f32>,
    pub layer: usize,
    pub x_q8: Option<&'a [u8]>,
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
            GgmlDType::Q2_0 | GgmlDType::PQ2_0 => {
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let w_base = entry.data().as_ptr() as usize;
                let row_bytes = entry.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: n_rows,
                    w_base,
                    a_ptr,
                    out_ptr,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_q2_0_q8_0_ptr,
                });
            }
            GgmlDType::PTQ1_0 => {
                // Prism PTQ1_0 (type 143): dense-trit ternary, 128-el blocks, 28 bytes.
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let w_base = entry.data().as_ptr() as usize;
                let row_bytes = entry.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: n_rows,
                    w_base,
                    a_ptr,
                    out_ptr,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_ptq1_0_q8_0_ptr,
                });
            }
            GgmlDType::Q10G128 => {
                let x_q8_data;
                let q8 = match x_q8 {
                    Some(d) => d,
                    None => {
                        x_q8_data = self.quantize_act(x);
                        &x_q8_data[..]
                    }
                };
                let w_base = entry.data().as_ptr() as usize;
                let row_bytes = entry.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: n_rows,
                    w_base,
                    a_ptr,
                    out_ptr,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_q1_0g128_q8_0_ptr,
                });
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
                let w_base = entry.data().as_ptr() as usize;
                let row_bytes = entry.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: n_rows,
                    w_base,
                    a_ptr,
                    out_ptr,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_q1_0g128_q8_0_ptr,
                });
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
                let w_base = entry.data().as_ptr() as usize;
                let row_bytes = entry.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: n_rows,
                    w_base,
                    a_ptr,
                    out_ptr,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_q8_0_q8_0_ptr,
                });
            }
            GgmlDType::Q4K => {
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
            GgmlDType::Q6K => {
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

                hearth_quant::dequantize(
                    entry.dtype,
                    entry.data(),
                    &mut row_buf[..n_cols * n_rows],
                )
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
    pub(crate) fn matmul_fused2(&self, p: FusedMatmul2Params<'_>) -> Result<(), String> {
        let FusedMatmul2Params {
            weight_a,
            weight_b,
            x,
            out_a,
            out_b,
            row_buf,
            layer,
            x_q8,
        } = p;
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
            GgmlDType::Q10G128 | GgmlDType::Q1_0 => {
                if !matches!(entry_b.dtype, GgmlDType::Q10G128 | GgmlDType::Q1_0) {
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
                let w_base_a = entry_a.data().as_ptr() as usize;
                let w_base_b = entry_b.data().as_ptr() as usize;
                let row_bytes = entry_a.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let out_a_ptr = out_a.as_mut_ptr() as usize;
                let out_b_ptr = out_b.as_mut_ptr() as usize;
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: n_rows_a,
                    w_base: w_base_a,
                    a_ptr,
                    out_ptr: out_a_ptr,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_q1_0g128_q8_0_ptr,
                });
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: n_rows_b,
                    w_base: w_base_b,
                    a_ptr,
                    out_ptr: out_b_ptr,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_q1_0g128_q8_0_ptr,
                });
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
    pub(crate) fn matmul_fused_qkv(&self, p: FusedQkvParams<'_>) -> Result<(), String> {
        let FusedQkvParams {
            w_q,
            w_k,
            w_v,
            x,
            out_q,
            out_k,
            out_v,
            row_buf,
            layer,
            x_q8,
        } = p;
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
            GgmlDType::Q10G128 | GgmlDType::Q1_0 => {
                if !matches!(ek.dtype, GgmlDType::Q10G128 | GgmlDType::Q1_0)
                    || !matches!(ev.dtype, GgmlDType::Q10G128 | GgmlDType::Q1_0)
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
                let wq = eq.data().as_ptr() as usize;
                let wk = ek.data().as_ptr() as usize;
                let wv = ev.data().as_ptr() as usize;
                let row_bytes = eq.dtype.byte_size(n_cols);
                let a_ptr = q8.as_ptr() as usize;
                let oq = out_q.as_mut_ptr() as usize;
                let ok = out_k.as_mut_ptr() as usize;
                let ov = out_v.as_mut_ptr() as usize;
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: nr_q,
                    w_base: wq,
                    a_ptr,
                    out_ptr: oq,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_q1_0g128_q8_0_ptr,
                });
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: nr_k,
                    w_base: wk,
                    a_ptr,
                    out_ptr: ok,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_q1_0g128_q8_0_ptr,
                });
                self.pool.par_dot_rows(crate::pool::DotRowsParams {
                    n: nr_v,
                    w_base: wv,
                    a_ptr,
                    out_ptr: ov,
                    row_bytes,
                    n_cols,
                    dot_fn: hearth_quant::dot_q1_0g128_q8_0_ptr,
                });
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

    /// Batched (prefill) matmul against a quantized weight.
    ///
    /// The weight's row count comes from the tensor, not from the caller: it used to arrive as an
    /// argument that the body never read. `n_cols` *is* checked against the tensor below, which is
    /// what catches a caller that mixed up two weight names.
    pub(crate) fn matmul_batch(
        &self,
        weight_name: &str,
        input: &[f32],
        out: &mut [f32],
        row_buf: &mut Vec<f32>,
        shape: BatchShape,
        x_q8: Option<&[u8]>,
    ) -> Result<(), String> {
        let BatchShape { n_cols, seq_len } = shape;
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
            GgmlDType::Q2_0
                | GgmlDType::PQ2_0
                | GgmlDType::PTQ1_0
                | GgmlDType::Q1_0
                | GgmlDType::Q10G128
        ) {
            let q8_size = k.div_ceil(32) * 34;
            let dot_fn = match entry.dtype {
                GgmlDType::Q2_0 | GgmlDType::PQ2_0 => hearth_quant::dot_q2_0_q8_0_ptr,
                GgmlDType::PTQ1_0 => hearth_quant::dot_ptq1_0_q8_0_ptr,
                _ => hearth_quant::dot_q1_0g128_q8_0_ptr,
            };
            // Paired kernel: expands each weight row once for two prompt tokens.
            // Bit-identical to two `dot_fn` calls, so only prefill is affected.
            let dot_fn2: Option<crate::pool::DotFn2> = match entry.dtype {
                GgmlDType::Q2_0 | GgmlDType::PQ2_0 => Some(hearth_quant::dot_q2_0_q8_0x2_ptr),
                GgmlDType::Q1_0 | GgmlDType::Q10G128 => Some(hearth_quant::dot_q1_0g128_q8_0x2_ptr),
                // PTQ1_0 deliberately stays None here. A paired PTQ1_0 kernel
                // exists (`hearth_quant::dot_ptq1_0_q8_0x2_ptr`, bit-identical to
                // two single calls), but this is the *dense* batch path and every
                // PTQ1_0 model in the catalog is a qwen35 hybrid that never calls
                // `matmul_batch` — wiring it here would be a no-op that reads like
                // a prefill speedup. The kernel belongs in the hybrid prefill,
                // which is still token-by-token; see docs/RESUME.md.
                _ => None,
            };
            let _q8_hold;
            let q8_buf: &[u8] = if let Some(data) = x_q8 {
                if data.len() < seq_len * q8_size {
                    return Err("matmul_batch: x_q8 too small".into());
                }
                data
            } else {
                let mut buf = Vec::with_capacity(seq_len * q8_size);
                for s in 0..seq_len {
                    hearth_quant::quantize_q8_0(&input[s * k..(s + 1) * k], &mut buf);
                }
                _q8_hold = buf;
                &_q8_hold[..]
            };
            self.pool
                .par_dot_rows_batched(crate::pool::BatchedDotRowsParams {
                    n: m,
                    seq_len,
                    q8_stride: q8_size,
                    w_base: entry.data().as_ptr() as usize,
                    a_ptr: q8_buf.as_ptr() as usize,
                    out_ptr: out.as_mut_ptr() as usize,
                    row_bytes: entry.dtype.byte_size(k),
                    n_cols: k,
                    dot_fn,
                    dot_fn2,
                });
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
