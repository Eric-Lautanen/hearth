use half::f16;
use wide::f32x8;

use crate::pool::{ParForFn, ThreadPool};

/// SIMD-accelerated RMS normalization using 8-wide f32 lanes.
/// out[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps)
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    let chunks = n / 8;
    let rem = n % 8;
    let mut vsum = f32x8::ZERO;
    for i in 0..chunks {
        let vx = f32x8::from(&x[i * 8..(i + 1) * 8]);
        vsum = vx.mul_add(vx, vsum);
    }
    sum_sq += vsum.reduce_add();
    for &xi in &x[n - rem..n] {
        sum_sq += xi * xi;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let vinv = f32x8::splat(inv_rms);
    for i in 0..chunks {
        let vx = f32x8::from(&x[i * 8..(i + 1) * 8]);
        let vw = f32x8::from(&weight[i * 8..(i + 1) * 8]);
        let vo = vx * vinv * vw;
        let slice = &mut out[i * 8..(i + 1) * 8];
        slice.copy_from_slice(&vo.to_array());
    }
    for i in n - rem..n {
        out[i] = weight[i] * x[i] * inv_rms;
    }
}

/// Gemma-style post-norm: out[i] = (1 + weight[i]) * x[i] / sqrt(mean(x^2) + eps)
/// Gemma initializes norm weights to zero, so at init this is just the identity.
pub fn rms_norm_gemma(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    let chunks = n / 8;
    let rem = n % 8;
    let mut vsum = f32x8::ZERO;
    for i in 0..chunks {
        let vx = f32x8::from(&x[i * 8..(i + 1) * 8]);
        vsum = vx.mul_add(vx, vsum);
    }
    sum_sq += vsum.reduce_add();
    for &xi in &x[n - rem..n] {
        sum_sq += xi * xi;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let vinv = f32x8::splat(inv_rms);
    let one = f32x8::splat(1.0);
    for i in 0..chunks {
        let vx = f32x8::from(&x[i * 8..(i + 1) * 8]);
        let vw = f32x8::from(&weight[i * 8..(i + 1) * 8]);
        let vo = vx * vinv * (one + vw);
        let slice = &mut out[i * 8..(i + 1) * 8];
        slice.copy_from_slice(&vo.to_array());
    }
    for i in n - rem..n {
        out[i] = (1.0 + weight[i]) * x[i] * inv_rms;
    }
}

/// Pre-computed sin/cos table for Rotary Position Embedding.
/// Eliminates trig (sin_cos) and exponent (theta.powf) calls from the hot path.
pub struct RopeCache {
    data: Vec<f32>,
    half: usize,
    max_seq_len: usize,
}

impl RopeCache {
    pub fn new(
        max_seq_len: usize,
        rope_dim: usize,
        theta: f32,
        scaling_type: Option<&str>,
        scaling_factor: Option<f32>,
        original_ctx_len: Option<u32>,
    ) -> Self {
        let half = rope_dim / 2;
        let factor = scaling_factor.unwrap_or(1.0);
        let orig_ctx = original_ctx_len.unwrap_or(2048) as f32;

        let mut eff_rate = vec![1.0f32; half];
        let mut mscale = 1.0f32;

        match scaling_type {
            Some("yarn") => {
                let freq_scale = 1.0 / factor;
                let n_dims = rope_dim as f32;
                let corr_dim = |n_rot: f32| -> f32 {
                    n_dims * (orig_ctx / (n_rot * 2.0 * std::f32::consts::PI)).ln()
                        / (2.0 * theta.ln())
                };
                let corr_low = (corr_dim(32.0)).floor().max(0.0);
                let corr_high = (corr_dim(1.0)).ceil().min(n_dims - 1.0);
                for i in 0..half {
                    let ramp_y = (i as f32 - corr_low) / (corr_high - corr_low).max(0.001);
                    let ramp_mix = 1.0 - ramp_y.clamp(0.0, 1.0);
                    eff_rate[i] = freq_scale * (1.0 - ramp_mix) + ramp_mix;
                }
                mscale = 1.0 + 0.1 * factor.ln();
            }
            Some("linear") if factor > 1.0 => {
                let scale = 1.0 / factor;
                for i in 0..half {
                    eff_rate[i] = scale;
                }
            }
            _ => {}
        }

        let mut inv_freq = vec![0.0f32; half];
        for i in 0..half {
            inv_freq[i] = 1.0 / theta.powf(2.0 * (i as f32) / rope_dim as f32);
        }

        let mut data = vec![0.0f32; max_seq_len * half * 2];
        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = pos as f32 * eff_rate[i] * inv_freq[i];
                let (sin, cos) = freq.sin_cos();
                let base = pos * half * 2;
                data[base + i] = sin * mscale;
                data[base + half + i] = cos * mscale;
            }
        }

        RopeCache {
            data,
            half,
            max_seq_len,
        }
    }

    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let half = self.half;
        let base = (pos % self.max_seq_len) * half * 2;
        let chunks = half / 8;
        let rem = half % 8;
        for i in 0..chunks {
            let off = i * 8;
            let vs = f32x8::from(&self.data[base + off..base + off + 8]);
            let vc = f32x8::from(&self.data[base + half + off..base + half + off + 8]);
            let vx0 = f32x8::from(&x[off..off + 8]);
            let vx1 = f32x8::from(&x[off + half..off + half + 8]);
            let r0 = vx0 * vc - vx1 * vs;
            let r1 = vx0 * vs + vx1 * vc;
            x[off..off + 8].copy_from_slice(&r0.to_array());
            x[off + half..off + half + 8].copy_from_slice(&r1.to_array());
        }
        for i in half - rem..half {
            let sin = self.data[base + i];
            let cos = self.data[base + half + i];
            let x0 = x[i];
            let x1 = x[i + half];
            x[i] = x0 * cos - x1 * sin;
            x[i + half] = x0 * sin + x1 * cos;
        }
    }
}

pub fn rope(
    x: &mut [f32],
    pos: usize,
    head_dim: usize,
    theta: f32,
    scaling_type: Option<&str>,
    scaling_factor: Option<f32>,
    original_ctx_len: Option<u32>,
) {
    let half = head_dim / 2;
    match scaling_type {
        Some("yarn") => {
            let factor = scaling_factor.unwrap_or(1.0f32);
            let orig_ctx = original_ctx_len.unwrap_or(2048) as f32;
            let freq_scale = 1.0 / factor;
            let beta_fast = 32.0f32;
            let beta_slow = 1.0f32;
            let n_dims = head_dim as f32;
            let corr_dim = |n_rot: f32| -> f32 {
                n_dims * (orig_ctx / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * theta.ln())
            };
            let corr_low = (corr_dim(beta_fast)).floor().max(0.0);
            let corr_high = (corr_dim(beta_slow)).ceil().min(n_dims - 1.0);
            let mscale = 1.0f32 + 0.1f32 * factor.ln();
            for i in 0..half {
                let theta_extrap = pos as f32 * theta.powf(-2.0 * (i as f32) / head_dim as f32);
                let theta_interp = freq_scale * theta_extrap;
                let ramp_y = (i as f32 - corr_low) / (corr_high - corr_low).max(0.001);
                let ramp_mix = 1.0 - ramp_y.clamp(0.0, 1.0);
                let t = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                let (sin, cos) = t.sin_cos();
                let x0 = x[i];
                let x1 = x[i + half];
                x[i] = (x0 * cos - x1 * sin) * mscale;
                x[i + half] = (x0 * sin + x1 * cos) * mscale;
            }
        }
        Some("dynamic") => {
            let factor = scaling_factor.unwrap_or(1.0);
            let orig_ctx = original_ctx_len.unwrap_or(2048) as usize;
            let base_theta = if pos > orig_ctx {
                theta * (factor * factor)
            } else {
                theta
            };
            for i in 0..half {
                let freq = pos as f32 / (base_theta.powf(2.0 * (i as f32) / head_dim as f32));
                let (sin, cos) = freq.sin_cos();
                let x0 = x[i];
                let x1 = x[i + half];
                x[i] = x0 * cos - x1 * sin;
                x[i + half] = x0 * sin + x1 * cos;
            }
        }
        _ => {
            let scale = match (scaling_type, scaling_factor) {
                (Some("linear"), Some(factor)) if factor > 1.0 => 1.0 / factor,
                _ => 1.0,
            };
            for i in 0..half {
                let freq = (pos as f32 * scale) / (theta.powf(2.0 * (i as f32) / head_dim as f32));
                let (sin, cos) = freq.sin_cos();
                let x0 = x[i];
                let x1 = x[i + half];
                x[i] = x0 * cos - x1 * sin;
                x[i + half] = x0 * sin + x1 * cos;
            }
        }
    }
}

/// SIMD-accelerated SiLU (sigmoid linear unit).
/// silu(x) = x / (1 + exp(-x))
pub fn silu(x: &mut [f32]) {
    let n = x.len();
    let chunks = n / 8;
    let rem = n % 8;
    for i in 0..chunks {
        let mut v = f32x8::from(&x[i * 8..(i + 1) * 8]);
        v = v / (f32x8::splat(1.0) + (-v).exp());
        let slice = &mut x[i * 8..(i + 1) * 8];
        slice.copy_from_slice(&v.to_array());
    }
    for v in x[n - rem..].iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

pub fn mul_elem(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len().min(a.len()).min(b.len());
    let chunks = n / 8;
    let rem = n % 8;
    for i in 0..chunks {
        let va = f32x8::from(&a[i * 8..(i + 1) * 8]);
        let vb = f32x8::from(&b[i * 8..(i + 1) * 8]);
        let vo = va * vb;
        out[i * 8..(i + 1) * 8].copy_from_slice(&vo.to_array());
    }
    for i in n - rem..n {
        out[i] = a[i] * b[i];
    }
}

/// Fused silu(x) * y + Q8_0 quantize. Computes silu(gate[i])*up[i] per element,
/// then quantizes to Q8_0 blocks in one pass (skipping ffn_tmp f32 intermediate).
/// dim must be multiple of 32.
pub fn silu_mul_q8(gate: &[f32], up: &[f32], q8_out: &mut [u8], dim: usize) {
    let blocks = dim / 32;
    for b in 0..blocks {
        let start = b * 32;
        let off = b * 34;
        let mut max_abs = 0.0f32;
        let mut tmp = [0.0f32; 32];
        for i in (0..32).step_by(8) {
            let idx = start + i;
            let vg = f32x8::from(&gate[idx..idx + 8]);
            let vu = f32x8::from(&up[idx..idx + 8]);
            let vs = vg / (f32x8::splat(1.0) + (-vg).exp());
            let vr = vs * vu;
            let arr = vr.to_array();
            for j in 0..8 {
                let v = arr[j];
                tmp[i + j] = v;
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let scale_f16 = f16::from_f32(scale);
        let sb = scale_f16.to_le_bytes();
        q8_out[off] = sb[0];
        q8_out[off + 1] = sb[1];
        for i in 0..32 {
            let q = (tmp[i] / scale).round().clamp(-128.0, 127.0) as i8;
            q8_out[off + 2 + i] = q as u8;
        }
    }
}

pub fn softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let chunks = n / 8;
    let rem = n % 8;
    let mut vmax = f32x8::splat(f32::NEG_INFINITY);
    for i in 0..chunks {
        let vx = f32x8::from(&x[i * 8..(i + 1) * 8]);
        vmax = vmax.fast_max(vx);
    }
    let arr = vmax.to_array();
    let mut max_val = arr[0];
    for &v in &arr[1..] {
        max_val = max_val.max(v);
    }
    for &v in &x[n - rem..n] {
        max_val = max_val.max(v);
    }
    let mut vsum = f32x8::ZERO;
    for i in 0..chunks {
        let mut vx = f32x8::from(&x[i * 8..(i + 1) * 8]);
        vx = (vx - f32x8::splat(max_val)).exp();
        vsum = vx.mul_add(f32x8::splat(1.0), vsum);
        x[i * 8..(i + 1) * 8].copy_from_slice(&vx.to_array());
    }
    let mut sum = vsum.reduce_add();
    for v in x[n - rem..n].iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = f32x8::splat(1.0 / sum);
        for i in 0..chunks {
            let vx = f32x8::from(&x[i * 8..(i + 1) * 8]);
            let vr = vx * inv;
            x[i * 8..(i + 1) * 8].copy_from_slice(&vr.to_array());
        }
        for v in x[n - rem..n].iter_mut() {
            *v /= sum;
        }
    }
}

#[allow(clippy::needless_range_loop)]
pub fn attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    head_dim: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    // Compute attention scores: q · k for all positions
    // Use SIMD dot product for each position
    let chunks = head_dim / 8;
    let rem = head_dim % 8;
    let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();
    for pos in 0..seq_len {
        let k_start = pos * head_dim;
        let mut vsum = f32x8::ZERO;
        for i in 0..chunks {
            let vq = f32x8::from(&q[i * 8..(i + 1) * 8]);
            let vk = f32x8::from(&k_cache[k_start + i * 8..k_start + (i + 1) * 8]);
            vsum = vq.mul_add(vk, vsum);
        }
        let mut score = vsum.reduce_add();
        for i in head_dim - rem..head_dim {
            score += q[i] * k_cache[k_start + i];
        }
        scratch[pos] = score * inv_sqrt_hd;
    }

    // Softmax over scores
    softmax(&mut scratch[..seq_len]);

    // Weighted sum of values — chunk-outer, position-inner
    let chunks = head_dim / 8;
    let rem = head_dim % 8;
    for i in 0..chunks {
        let start = i * 8;
        let mut vacc = f32x8::ZERO;
        for pos in 0..seq_len {
            let att = scratch[pos];
            let v_start = pos * head_dim;
            let vatt = f32x8::splat(att);
            let vv = f32x8::from(&v_cache[v_start + start..v_start + start + 8]);
            vacc = vv.mul_add(vatt, vacc);
        }
        out[start..start + 8].copy_from_slice(&vacc.to_array());
    }
    for i in head_dim - rem..head_dim {
        let mut sum = 0.0f32;
        for pos in 0..seq_len {
            sum += scratch[pos] * v_cache[pos * head_dim + i];
        }
        out[i] = sum;
    }
}

/// Batched RMS norm: apply rms_norm independently to each row of a [seq_len × dim] matrix.
/// `x` and `out` are laid out as [seq_len × dim] in row-major order.
pub fn rms_norm_batch(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
    seq_len: usize,
    dim: usize,
) {
    for s in 0..seq_len {
        let row_in = &x[s * dim..(s + 1) * dim];
        let row_out = &mut out[s * dim..(s + 1) * dim];
        rms_norm(row_in, weight, eps, row_out);
    }
}

pub fn rms_norm_gemma_batch(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
    seq_len: usize,
    dim: usize,
) {
    for s in 0..seq_len {
        let row_in = &x[s * dim..(s + 1) * dim];
        let row_out = &mut out[s * dim..(s + 1) * dim];
        rms_norm_gemma(row_in, weight, eps, row_out);
    }
}

/// Fused RMS norm + Q8_0 quantize: computes normed values and quantizes directly
/// to Q8_0 blocks, skipping the intermediate f32 write (saves 1 read+1 write pass).
pub fn rms_norm_q8(x: &[f32], weight: &[f32], eps: f32, q8_out: &mut [u8], dim: usize) {
    let n = dim;
    let mut sum_sq = 0.0f32;
    let chunks = n / 8;
    let rem = n % 8;
    let mut vsum = f32x8::ZERO;
    for i in 0..chunks {
        let vx = f32x8::from(&x[i * 8..(i + 1) * 8]);
        vsum = vx.mul_add(vx, vsum);
    }
    sum_sq += vsum.reduce_add();
    for &xi in &x[n - rem..n] {
        sum_sq += xi * xi;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    let blocks = n.div_ceil(32);
    for b in 0..blocks {
        let start = b * 32;
        let mut max_abs = 0.0f32;
        for i in 0..32.min(n - start) {
            let idx = start + i;
            let normed = weight[idx] * x[idx] * inv_rms;
            let a = normed.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let scale_f16 = f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        let off = b * 34;
        q8_out[off] = scale_bytes[0];
        q8_out[off + 1] = scale_bytes[1];
        for i in 0..32.min(n - start) {
            let idx = start + i;
            let normed = weight[idx] * x[idx] * inv_rms;
            let q = (normed / scale).round().clamp(-128.0, 127.0) as i8;
            q8_out[off + 2 + i] = q as u8;
        }
        for i in 32.min(n - start)..32 {
            q8_out[off + 2 + i] = 0;
        }
    }
}

/// Batched version of rms_norm_q8 — processes seq_len rows independently.
pub fn rms_norm_q8_batch(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    q8_out: &mut [u8],
    seq_len: usize,
    dim: usize,
) {
    let q8_row = dim.div_ceil(32) * 34;
    for s in 0..seq_len {
        let row_in = &x[s * dim..(s + 1) * dim];
        let row_out = &mut q8_out[s * q8_row..(s + 1) * q8_row];
        rms_norm_q8(row_in, weight, eps, row_out, dim);
    }
}

/// Batched attention with causal mask for prefill.
/// q_heads: [seq_len × n_q_heads × head_dim] row-major
/// k_cache/v_cache: per kv-head, [max_seq_len × head_dim] (standard KV cache layout)
/// out: [seq_len × d_model] row-major
/// scratch: temporary buffer of size [seq_len] per call
///
/// Each query position `s` attends to positions `0..=s` (causal mask).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn attention_batch(
    q_heads: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let kv_repeat = n_q_heads / n_kv_heads;
    let d = n_q_heads * head_dim;
    out[..seq_len * d].fill(0.0f32);

    for s in 0..seq_len {
        let attended_len = s + 1;
        for qh in 0..n_q_heads {
            let kvh = qh / kv_repeat;
            let q_row = &q_heads[s * n_q_heads * head_dim + qh * head_dim..][..head_dim];
            let head_offset = kvh * head_dim * max_seq_len;
            let ks = &k_cache[head_offset..head_offset + attended_len * head_dim];
            let vs = &v_cache[head_offset..head_offset + attended_len * head_dim];
            let out_row = &mut out[s * d + qh * head_dim..][..head_dim];

            // Dot product Q·K with scaling, causal mask implicit since attended_len = s+1
            let chunks = head_dim / 8;
            let rem = head_dim % 8;
            let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();
            for pos in 0..attended_len {
                let k_start = pos * head_dim;
                let mut vsum = f32x8::ZERO;
                for i in 0..chunks {
                    let vq = f32x8::from(&q_row[i * 8..(i + 1) * 8]);
                    let vk = f32x8::from(&ks[k_start + i * 8..k_start + (i + 1) * 8]);
                    vsum = vq.mul_add(vk, vsum);
                }
                let mut score = vsum.reduce_add();
                for i in head_dim - rem..head_dim {
                    score += q_row[i] * ks[k_start + i];
                }
                scratch[pos] = score * inv_sqrt_hd;
            }

            softmax(&mut scratch[..attended_len]);

            // Weighted sum of values — chunk-outer, position-inner
            for i in 0..chunks {
                let start = i * 8;
                let mut vacc = f32x8::ZERO;
                for pos in 0..attended_len {
                    let vv = f32x8::from(&vs[pos * head_dim + start..pos * head_dim + start + 8]);
                    vacc = vv.mul_add(f32x8::splat(scratch[pos]), vacc);
                }
                out_row[start..start + 8].copy_from_slice(&vacc.to_array());
            }
            for i in head_dim - rem..head_dim {
                let mut sum = 0.0f32;
                for pos in 0..attended_len {
                    sum += scratch[pos] * vs[pos * head_dim + i];
                }
                out_row[i] = sum;
            }
        }
    }
}

struct AttnBatchCtx {
    q_heads: *const f32,
    k_cache: *const f32,
    v_cache: *const f32,
    seq_len: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    out: *mut f32,
    scratch: *mut f32,
    kv_repeat: usize,
    d: usize,
}

unsafe fn attn_batch_worker(worker_id: usize, kvh_begin: usize, kvh_end: usize, ctx_ptr: usize) {
    let ctx = &*(ctx_ptr as *const AttnBatchCtx);
    let kv_repeat = ctx.kv_repeat;
    let head_dim = ctx.head_dim;
    let d = ctx.d;
    let n_q_heads = ctx.n_q_heads;
    let n_kv_heads = ctx.n_kv_heads;
    let max_seq_len = ctx.max_seq_len;
    let seq_len = ctx.seq_len;

    let q_heads = std::slice::from_raw_parts(ctx.q_heads, seq_len * d);
    let k_cache = std::slice::from_raw_parts(ctx.k_cache, n_kv_heads * max_seq_len * head_dim);
    let v_cache = std::slice::from_raw_parts(ctx.v_cache, n_kv_heads * max_seq_len * head_dim);
    let out = std::slice::from_raw_parts_mut(ctx.out, seq_len * d);
    let scratch =
        std::slice::from_raw_parts_mut(ctx.scratch.add(worker_id * max_seq_len), max_seq_len);

    let chunks = head_dim / 8;
    let rem = head_dim % 8;
    let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();

    for s in 0..seq_len {
        let attended_len = s + 1;
        for kvh in kvh_begin..kvh_end {
            for r in 0..kv_repeat {
                let qh = kvh * kv_repeat + r;
                let q_row = &q_heads[s * n_q_heads * head_dim + qh * head_dim..][..head_dim];
                let head_offset = kvh * head_dim * max_seq_len;
                let ks = &k_cache[head_offset..head_offset + attended_len * head_dim];
                let vs = &v_cache[head_offset..head_offset + attended_len * head_dim];
                let out_row = &mut out[s * d + qh * head_dim..][..head_dim];

                for pos in 0..attended_len {
                    let k_start = pos * head_dim;
                    let mut vsum = f32x8::ZERO;
                    for i in 0..chunks {
                        let vq = f32x8::from(&q_row[i * 8..(i + 1) * 8]);
                        let vk = f32x8::from(&ks[k_start + i * 8..k_start + (i + 1) * 8]);
                        vsum = vq.mul_add(vk, vsum);
                    }
                    let mut score = vsum.reduce_add();
                    for i in head_dim - rem..head_dim {
                        score += q_row[i] * ks[k_start + i];
                    }
                    scratch[pos] = score * inv_sqrt_hd;
                }

                softmax(&mut scratch[..attended_len]);

                for i in 0..chunks {
                    let start = i * 8;
                    let mut vacc = f32x8::ZERO;
                    for pos in 0..attended_len {
                        let vv =
                            f32x8::from(&vs[pos * head_dim + start..pos * head_dim + start + 8]);
                        vacc = vv.mul_add(f32x8::splat(scratch[pos]), vacc);
                    }
                    out_row[start..start + 8].copy_from_slice(&vacc.to_array());
                }
                for i in head_dim - rem..head_dim {
                    let mut sum = 0.0f32;
                    for pos in 0..attended_len {
                        sum += scratch[pos] * vs[pos * head_dim + i];
                    }
                    out_row[i] = sum;
                }
            }
        }
    }
}

/// Parallel batched attention: dispatches query positions across thread pool workers.
/// scratch must be sized >= pool.num_threads * max_seq_len.
pub fn attention_batch_parallel(
    q_heads: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    out: &mut [f32],
    scratch: &mut [f32],
    pool: &ThreadPool,
) {
    let kv_repeat = n_q_heads / n_kv_heads;
    let d = n_q_heads * head_dim;
    out[..seq_len * d].fill(0.0f32);

    let ctx = AttnBatchCtx {
        q_heads: q_heads.as_ptr(),
        k_cache: k_cache.as_ptr(),
        v_cache: v_cache.as_ptr(),
        seq_len,
        n_q_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        out: out.as_mut_ptr(),
        scratch: scratch.as_mut_ptr(),
        kv_repeat,
        d,
    };
    let func: ParForFn = attn_batch_worker;
    pool.par_for(n_kv_heads, func, &ctx as *const AttnBatchCtx as usize);
}

struct AttnBatchQ8Ctx {
    q_heads: *const f32,
    k_q8: *const u8,
    v_q8: *const u8,
    seq_len: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    out: *mut f32,
    scratch: *mut f32,
    kv_repeat: usize,
    d: usize,
    q8_row: usize,
    q8_stride: usize,
}

unsafe fn attn_batch_q8_worker(worker_id: usize, begin: usize, end: usize, ctx_ptr: usize) {
    let ctx = &*(ctx_ptr as *const AttnBatchQ8Ctx);
    let kv_repeat = ctx.kv_repeat;
    let head_dim = ctx.head_dim;
    let d = ctx.d;
    let n_q_heads = ctx.n_q_heads;
    let n_kv_heads = ctx.n_kv_heads;
    let max_seq_len = ctx.max_seq_len;
    let seq_len = ctx.seq_len;
    let blocks = head_dim.div_ceil(32);
    let q8_row = ctx.q8_row;
    let q8_stride = ctx.q8_stride;
    let rem = head_dim % 8;
    let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();

    let q_heads = std::slice::from_raw_parts(ctx.q_heads, seq_len * d);
    let k_q8 = std::slice::from_raw_parts(ctx.k_q8, n_kv_heads * q8_stride);
    let v_q8 = std::slice::from_raw_parts(ctx.v_q8, n_kv_heads * q8_stride);
    let out = std::slice::from_raw_parts_mut(ctx.out, seq_len * d);
    let scratch =
        std::slice::from_raw_parts_mut(ctx.scratch.add(worker_id * max_seq_len), max_seq_len);

    let kvh_begin = begin;
    let kvh_end = end;
    for s in 0..seq_len {
        let attended_len = s + 1;
        let attend_bytes = attended_len * q8_row;
        for kvh in kvh_begin..kvh_end {
            for r in 0..kv_repeat {
                let qh = kvh * kv_repeat + r;
                let q_row = &q_heads[s * n_q_heads * head_dim + qh * head_dim..][..head_dim];
                let head_off = kvh * q8_stride;
                let ks = &k_q8[head_off..head_off + attend_bytes];
                let vs = &v_q8[head_off..head_off + attend_bytes];
                let out_row = &mut out[s * d + qh * head_dim..][..head_dim];
                for pos in 0..attended_len {
                    let bp = pos * q8_row;
                    let mut vsum = f32x8::ZERO;
                    for b in 0..blocks {
                        let bo = bp + b * 34;
                        let d_scale = f16::from_le_bytes([ks[bo], ks[bo + 1]]).to_f32();
                        let vd = f32x8::splat(d_scale);
                        let vs_off = bo + 2;
                        for i in 0..4 {
                            let qb = b * 32 + i * 8;
                            let vq = f32x8::from(&q_row[qb..qb + 8]);
                            let vk = f32x8::new([
                                ks[vs_off + i * 8] as i8 as f32,
                                ks[vs_off + i * 8 + 1] as i8 as f32,
                                ks[vs_off + i * 8 + 2] as i8 as f32,
                                ks[vs_off + i * 8 + 3] as i8 as f32,
                                ks[vs_off + i * 8 + 4] as i8 as f32,
                                ks[vs_off + i * 8 + 5] as i8 as f32,
                                ks[vs_off + i * 8 + 6] as i8 as f32,
                                ks[vs_off + i * 8 + 7] as i8 as f32,
                            ]);
                            vsum = vq.mul_add(vk * vd, vsum);
                        }
                    }
                    let mut score = vsum.reduce_add();
                    if rem > 0 {
                        for i in head_dim - rem..head_dim {
                            let blk = i / 32;
                            let in_blk = i % 32;
                            let bo = bp + blk * 34;
                            let d_scale = f16::from_le_bytes([ks[bo], ks[bo + 1]]).to_f32();
                            score += q_row[i] * (ks[bo + 2 + in_blk] as i8 as f32) * d_scale;
                        }
                    }
                    scratch[pos] = score * inv_sqrt_hd;
                }

                softmax(&mut scratch[..attended_len]);

                // Weighted sum — chunk-outer, position-inner
                for i in 0..(head_dim / 8) {
                    let start = i * 8;
                    let mut vacc = f32x8::ZERO;
                    for pos in 0..attended_len {
                        let bp = pos * q8_row;
                        let blk = i / 4;
                        let off_in_blk = (i % 4) * 8;
                        let bo = bp + blk * 34;
                        let d_scale = f16::from_le_bytes([vs[bo], vs[bo + 1]]).to_f32();
                        let vd = f32x8::splat(d_scale);
                        let vs_off = bo + 2 + off_in_blk;
                        let vv = f32x8::new([
                            vs[vs_off] as i8 as f32,
                            vs[vs_off + 1] as i8 as f32,
                            vs[vs_off + 2] as i8 as f32,
                            vs[vs_off + 3] as i8 as f32,
                            vs[vs_off + 4] as i8 as f32,
                            vs[vs_off + 5] as i8 as f32,
                            vs[vs_off + 6] as i8 as f32,
                            vs[vs_off + 7] as i8 as f32,
                        ]);
                        vacc = vv.mul_add(f32x8::splat(scratch[pos]) * vd, vacc);
                    }
                    out_row[start..start + 8].copy_from_slice(&vacc.to_array());
                }
                for i in head_dim - rem..head_dim {
                    let mut sum = 0.0f32;
                    for pos in 0..attended_len {
                        let bp = pos * q8_row;
                        let blk = i / 32;
                        let in_blk = i % 32;
                        let bo = bp + blk * 34;
                        let d_scale = f16::from_le_bytes([vs[bo], vs[bo + 1]]).to_f32();
                        sum += scratch[pos] * (vs[bo + 2 + in_blk] as i8 as f32) * d_scale;
                    }
                    out_row[i] = sum;
                }
            }
        }
    }
}

/// Parallel batched attention with Q8_0 K/V cache.
/// scratch must be sized >= pool.num_threads * max_seq_len.
pub fn attention_batch_q8_0_parallel(
    q_heads: &[f32],
    k_q8: &[u8],
    v_q8: &[u8],
    seq_len: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    out: &mut [f32],
    scratch: &mut [f32],
    pool: &ThreadPool,
) {
    let kv_repeat = n_q_heads / n_kv_heads;
    let d = n_q_heads * head_dim;
    let blocks = head_dim.div_ceil(32);
    let q8_row = blocks * 34;
    let q8_stride = max_seq_len * q8_row;
    out[..seq_len * d].fill(0.0f32);

    let ctx = AttnBatchQ8Ctx {
        q_heads: q_heads.as_ptr(),
        k_q8: k_q8.as_ptr(),
        v_q8: v_q8.as_ptr(),
        seq_len,
        n_q_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        out: out.as_mut_ptr(),
        scratch: scratch.as_mut_ptr(),
        kv_repeat,
        d,
        q8_row,
        q8_stride,
    };
    let func: ParForFn = attn_batch_q8_worker;
    pool.par_for(n_kv_heads, func, &ctx as *const AttnBatchQ8Ctx as usize);
}

/// Single-query attention with Q8_0-encoded K/V cache.
/// k_q8, v_q8 hold seq_len positions of Q8_0 blocks for ONE kv_head.
/// Layout: [pos0_block0_34bytes, pos0_block1_34bytes, ..., posN_blockM_34bytes]
pub fn attention_q8_0(
    q: &[f32],
    k_q8: &[u8],
    v_q8: &[u8],
    seq_len: usize,
    head_dim: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let blocks = head_dim.div_ceil(32);
    let q8_row = blocks * 34;
    let rem = head_dim % 8;
    let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();

    // Score computation with on-the-fly K dequant
    for pos in 0..seq_len {
        let bp = pos * q8_row;
        let mut vsum = f32x8::ZERO;
        for b in 0..blocks {
            let bo = bp + b * 34;
            let d = f16::from_le_bytes([k_q8[bo], k_q8[bo + 1]]).to_f32();
            let vd = f32x8::splat(d);
            let vs = bo + 2;
            for i in 0..4 {
                let qb = b * 32 + i * 8;
                let vq = f32x8::from(&q[qb..qb + 8]);
                let vk = f32x8::new([
                    k_q8[vs + i * 8] as i8 as f32,
                    k_q8[vs + i * 8 + 1] as i8 as f32,
                    k_q8[vs + i * 8 + 2] as i8 as f32,
                    k_q8[vs + i * 8 + 3] as i8 as f32,
                    k_q8[vs + i * 8 + 4] as i8 as f32,
                    k_q8[vs + i * 8 + 5] as i8 as f32,
                    k_q8[vs + i * 8 + 6] as i8 as f32,
                    k_q8[vs + i * 8 + 7] as i8 as f32,
                ]);
                vsum = vq.mul_add(vk * vd, vsum);
            }
        }
        let mut score = vsum.reduce_add();
        if rem > 0 {
            for i in head_dim - rem..head_dim {
                let blk = i / 32;
                let in_blk = i % 32;
                let bo = bp + blk * 34;
                let d = f16::from_le_bytes([k_q8[bo], k_q8[bo + 1]]).to_f32();
                score += q[i] * (k_q8[bo + 2 + in_blk] as i8 as f32) * d;
            }
        }
        scratch[pos] = score * inv_sqrt_hd;
    }

    softmax(&mut scratch[..seq_len]);

    // Weighted sum with on-the-fly V dequant
    for i in 0..head_dim {
        out[i] = 0.0;
    }
    for pos in 0..seq_len {
        let att = scratch[pos];
        let bp = pos * q8_row;
        let vatt = f32x8::splat(att);
        for b in 0..blocks {
            let bo = bp + b * 34;
            let d = f16::from_le_bytes([v_q8[bo], v_q8[bo + 1]]).to_f32();
            let vd = f32x8::splat(d);
            let vs = bo + 2;
            for i in 0..4 {
                let start = b * 32 + i * 8;
                let mut vacc = f32x8::from(&out[start..start + 8]);
                let vv = f32x8::new([
                    v_q8[vs + i * 8] as i8 as f32,
                    v_q8[vs + i * 8 + 1] as i8 as f32,
                    v_q8[vs + i * 8 + 2] as i8 as f32,
                    v_q8[vs + i * 8 + 3] as i8 as f32,
                    v_q8[vs + i * 8 + 4] as i8 as f32,
                    v_q8[vs + i * 8 + 5] as i8 as f32,
                    v_q8[vs + i * 8 + 6] as i8 as f32,
                    v_q8[vs + i * 8 + 7] as i8 as f32,
                ]);
                vacc = vv.mul_add(vatt * vd, vacc);
                out[start..start + 8].copy_from_slice(&vacc.to_array());
            }
        }
        if rem > 0 {
            for i in head_dim - rem..head_dim {
                let blk = i / 32;
                let in_blk = i % 32;
                let bo = bp + blk * 34;
                let d = f16::from_le_bytes([v_q8[bo], v_q8[bo + 1]]).to_f32();
                out[i] += att * (v_q8[bo + 2 + in_blk] as i8 as f32) * d;
            }
        }
    }
}

/// Batched attention with Q8_0-encoded K/V cache for prefill.
/// k_q8, v_q8 hold all KV heads' data (same layout as k_q8/v_q8 in KVCache).
/// max_seq_len is the stride between heads (same as KVCache).
pub fn attention_batch_q8_0(
    q_heads: &[f32],
    k_q8: &[u8],
    v_q8: &[u8],
    seq_len: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let blocks = head_dim.div_ceil(32);
    let q8_row = blocks * 34;
    let q8_stride = max_seq_len * q8_row;
    let kv_repeat = n_q_heads / n_kv_heads;
    let d = n_q_heads * head_dim;
    let rem = head_dim % 8;
    let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();

    out[..seq_len * d].fill(0.0f32);

    for s in 0..seq_len {
        let attended_len = s + 1;
        let attend_bytes = attended_len * q8_row;
        for qh in 0..n_q_heads {
            let kvh = qh / kv_repeat;
            let q_row = &q_heads[s * n_q_heads * head_dim + qh * head_dim..][..head_dim];
            let head_off = kvh * q8_stride;
            let ks = &k_q8[head_off..head_off + attend_bytes];
            let vs = &v_q8[head_off..head_off + attend_bytes];
            let out_row = &mut out[s * d + qh * head_dim..][..head_dim];

            // Score computation with on-the-fly K dequant
            for pos in 0..attended_len {
                let bp = pos * q8_row;
                let mut vsum = f32x8::ZERO;
                for b in 0..blocks {
                    let bo = bp + b * 34;
                    let d_scale = f16::from_le_bytes([ks[bo], ks[bo + 1]]).to_f32();
                    let vd = f32x8::splat(d_scale);
                    let vs_off = bo + 2;
                    for i in 0..4 {
                        let qb = b * 32 + i * 8;
                        let vq = f32x8::from(&q_row[qb..qb + 8]);
                        let vk = f32x8::new([
                            ks[vs_off + i * 8] as i8 as f32,
                            ks[vs_off + i * 8 + 1] as i8 as f32,
                            ks[vs_off + i * 8 + 2] as i8 as f32,
                            ks[vs_off + i * 8 + 3] as i8 as f32,
                            ks[vs_off + i * 8 + 4] as i8 as f32,
                            ks[vs_off + i * 8 + 5] as i8 as f32,
                            ks[vs_off + i * 8 + 6] as i8 as f32,
                            ks[vs_off + i * 8 + 7] as i8 as f32,
                        ]);
                        vsum = vq.mul_add(vk * vd, vsum);
                    }
                }
                let mut score = vsum.reduce_add();
                if rem > 0 {
                    for i in head_dim - rem..head_dim {
                        let blk = i / 32;
                        let in_blk = i % 32;
                        let bo = bp + blk * 34;
                        let d_scale = f16::from_le_bytes([ks[bo], ks[bo + 1]]).to_f32();
                        score += q_row[i] * (ks[bo + 2 + in_blk] as i8 as f32) * d_scale;
                    }
                }
                scratch[pos] = score * inv_sqrt_hd;
            }

            softmax(&mut scratch[..attended_len]);

            // Weighted sum with on-the-fly V dequant
            for pos in 0..attended_len {
                let att = scratch[pos];
                let bp = pos * q8_row;
                let vatt = f32x8::splat(att);
                for b in 0..blocks {
                    let bo = bp + b * 34;
                    let d_scale = f16::from_le_bytes([vs[bo], vs[bo + 1]]).to_f32();
                    let vd = f32x8::splat(d_scale);
                    let vs_off = bo + 2;
                    for i in 0..4 {
                        let start = b * 32 + i * 8;
                        let mut vacc = f32x8::from(&out_row[start..start + 8]);
                        let vv = f32x8::new([
                            vs[vs_off + i * 8] as i8 as f32,
                            vs[vs_off + i * 8 + 1] as i8 as f32,
                            vs[vs_off + i * 8 + 2] as i8 as f32,
                            vs[vs_off + i * 8 + 3] as i8 as f32,
                            vs[vs_off + i * 8 + 4] as i8 as f32,
                            vs[vs_off + i * 8 + 5] as i8 as f32,
                            vs[vs_off + i * 8 + 6] as i8 as f32,
                            vs[vs_off + i * 8 + 7] as i8 as f32,
                        ]);
                        vacc = vv.mul_add(vatt * vd, vacc);
                        out_row[start..start + 8].copy_from_slice(&vacc.to_array());
                    }
                }
                if rem > 0 {
                    for i in head_dim - rem..head_dim {
                        let blk = i / 32;
                        let in_blk = i % 32;
                        let bo = bp + blk * 34;
                        let d_scale = f16::from_le_bytes([vs[bo], vs[bo + 1]]).to_f32();
                        out_row[i] += att * (vs[bo + 2 + in_blk] as i8 as f32) * d_scale;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        rms_norm(&x, &w, 1e-5, &mut out);
        let rms: f32 = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.01,
            "RMS norm output should have unit RMS, got {}",
            rms
        );
    }

    #[test]
    fn test_rope_shape() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0];
        rope(&mut x, 0, 4, 10000.0, None, None, None);
        for v in &x {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_silu() {
        let mut x = vec![0.0, 1.0, -1.0];
        silu(&mut x);
        assert!((x[0] - 0.0).abs() < 0.01);
        assert!((x[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_attention() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 2];
        let mut scratch = vec![0.0; 2];
        attention(&q, &k, &v, 2, 2, &mut out, &mut scratch);
        assert!(out[0].is_finite());
        assert!(out[1].is_finite());
    }
}
