use wide::f32x8;

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

    // Weighted sum of values — SIMD accelerated
    for i in 0..head_dim {
        out[i] = 0.0;
    }
    let chunks = head_dim / 8;
    let rem = head_dim % 8;
    for pos in 0..seq_len {
        let att = scratch[pos];
        let v_start = pos * head_dim;
        let vatt = f32x8::splat(att);
        for i in 0..chunks {
            let start = i * 8;
            let mut vacc = f32x8::from(&out[start..start + 8]);
            let vv = f32x8::from(&v_cache[v_start + start..v_start + start + 8]);
            vacc = vv.mul_add(vatt, vacc);
            out[start..start + 8].copy_from_slice(&vacc.to_array());
        }
        for i in head_dim - rem..head_dim {
            out[i] += att * v_cache[v_start + i];
        }
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

            // Weighted sum of values
            for pos in 0..attended_len {
                let att = scratch[pos];
                let v_start = pos * head_dim;
                let vatt = f32x8::splat(att);
                for i in 0..chunks {
                    let start = i * 8;
                    let mut vacc = f32x8::from(&out_row[start..start + 8]);
                    let vv = f32x8::from(&vs[v_start + start..v_start + start + 8]);
                    vacc = vv.mul_add(vatt, vacc);
                    out_row[start..start + 8].copy_from_slice(&vacc.to_array());
                }
                for i in head_dim - rem..head_dim {
                    out_row[i] += att * vs[v_start + i];
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
