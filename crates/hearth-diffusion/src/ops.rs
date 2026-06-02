use hearth_quant;
use rayon::prelude::*;

pub fn matmul_q2(q2: &super::weights::Q2Tensor, x: &[f32], out: &mut [f32]) {
    let rows = q2.rows;
    let cols = q2.cols;
    let block_size = 34;
    let row_bytes = (cols / 128) * block_size;
    let w_base = q2.data.as_ptr() as usize;
    let mut x_q8 = Vec::new();
    hearth_quant::quantize_q8_0(x, &mut x_q8);
    let a_ptr = x_q8.as_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    (0..rows).into_par_iter().for_each(|row| unsafe {
        *((out_ptr + row * 4) as *mut f32) = hearth_quant::dot_q2_0_q8_0_ptr(
            (w_base + row * row_bytes) as *const u8,
            a_ptr as *const u8,
            cols,
        );
    });
}

pub fn matmul_q2_batched(q2: &super::weights::Q2Tensor, x: &[f32], out: &mut [f32], batch: usize) {
    let rows = q2.rows;
    let cols = q2.cols;
    let block_size = 34;
    let row_bytes = (cols / 128) * block_size;
    let w_base = q2.data.as_ptr() as usize;
    let x_ptr = x.as_ptr() as usize;
    let out_ptr_base = out.as_mut_ptr() as usize;
    (0..batch).into_par_iter().for_each(|b| {
        let x_off = b * cols;
        let out_off = b * rows;
        let mut x_q8 = Vec::new();
        unsafe {
            let x_slice = std::slice::from_raw_parts((x_ptr + x_off) as *const f32, cols);
            hearth_quant::quantize_q8_0(x_slice, &mut x_q8);
        }
        let a_ptr = x_q8.as_ptr() as usize;
        for row in 0..rows {
            unsafe {
                *((out_ptr_base + (out_off + row) * 4) as *mut f32) =
                    hearth_quant::dot_q2_0_q8_0_ptr(
                        (w_base + row * row_bytes) as *const u8,
                        a_ptr as *const u8,
                        cols,
                    );
            }
        }
    });
}

pub fn matmul_bf16(weight: &[f32], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    for i in 0..m {
        let mut sum = 0.0f32;
        let w_row = &weight[i * k..(i + 1) * k];
        for j in 0..k {
            sum += w_row[j] * x[j];
        }
        out[i] = sum;
    }
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
        *ai += bi;
    }
}

pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
        *ai *= bi;
    }
}

pub fn scale(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|&v| v * s).collect()
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn rope_2d(
    q: &mut [f32],
    k: &mut [f32],
    pos_h: usize,
    pos_w: usize,
    head_dim: usize,
    theta: f32,
    axes_dims: &[usize],
) {
    let n_heads = q.len() / head_dim;
    for h in 0..n_heads {
        let q_off = h * head_dim;
        let k_off = h * head_dim;
        let mut dim_offset = 0;
        for &d in axes_dims {
            let half_d = d / 2;
            for i in 0..half_d {
                let freq = 1.0 / theta.powf((i * 2) as f32 / d as f32);
                let pos_val = if dim_offset < head_dim / 2 {
                    pos_h as f32
                } else {
                    pos_w as f32
                };
                let angle = pos_val * freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let qi = q_off + dim_offset + i * 2;
                let ki = k_off + dim_offset + i * 2;
                let q0 = q[qi];
                let q1 = q[qi + 1];
                q[qi] = q0 * cos_a - q1 * sin_a;
                q[qi + 1] = q0 * sin_a + q1 * cos_a;
                let k0 = k[ki];
                let k1 = k[ki + 1];
                k[ki] = k0 * cos_a - k1 * sin_a;
                k[ki + 1] = k0 * sin_a + k1 * cos_a;
            }
            dim_offset += d;
        }
    }
}

pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    n_heads: usize,
    out: &mut [f32],
    scores: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_heads {
        let q_off = h * head_dim;
        for s in 0..seq_len {
            let mut dot = 0.0f32;
            let k_off = s * n_heads * head_dim + h * head_dim;
            for d in 0..head_dim {
                dot += q[q_off + d] * k[k_off + d];
            }
            scores[s] = dot * scale;
        }
        softmax(&mut scores[..seq_len]);
        let o_off = h * head_dim;
        for d in 0..head_dim {
            out[o_off + d] = 0.0;
        }
        for s in 0..seq_len {
            let v_off = s * n_heads * head_dim + h * head_dim;
            for d in 0..head_dim {
                out[o_off + d] += scores[s] * v[v_off + d];
            }
        }
    }
}

pub fn conv2d_3x3(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * c_out];
    for oh in 0..h {
        for ow in 0..w {
            for oc in 0..c_out {
                let mut sum = bias[oc];
                for ic in 0..c_in {
                    for kh in 0..3usize {
                        for kw in 0..3usize {
                            let ih = oh as isize + kh as isize - 1;
                            let iw = ow as isize + kw as isize - 1;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let w_idx = ((oc * c_in + ic) * 3 + kh) * 3 + kw;
                                let i_idx = (ih as usize * w + iw as usize) * c_in + ic;
                                sum += weight[w_idx] * input[i_idx];
                            }
                        }
                    }
                }
                out[(oh * w + ow) * c_out + oc] = sum;
            }
        }
    }
    out
}

pub fn upsample_nearest_2x(input: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let h2 = h * 2;
    let w2 = w * 2;
    let mut out = vec![0.0f32; h2 * w2 * c];
    for ih in 0..h {
        for iw in 0..w {
            for ic in 0..c {
                let val = input[(ih * w + iw) * c + ic];
                for dy in 0..2 {
                    for dx in 0..2 {
                        out[((ih * 2 + dy) * w2 + (iw * 2 + dx)) * c + ic] = val;
                    }
                }
            }
        }
    }
    out
}

pub fn group_norm(
    x: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    n_groups: usize,
    c: usize,
    hw: usize,
    eps: f32,
) {
    let c_per_group = c / n_groups;
    for g in 0..n_groups {
        let c_start = g * c_per_group;
        let c_end = c_start + c_per_group;
        let n_elem = c_per_group * hw;
        let mean: f32 = x[c_start * hw..c_end * hw].iter().sum::<f32>() / n_elem as f32;
        let var: f32 = x[c_start * hw..c_end * hw]
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>()
            / n_elem as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ic in c_start..c_end {
            for i in 0..hw {
                let idx = ic * hw + i;
                x[idx] = (x[idx] - mean) * inv_std * weight[ic] + bias[ic];
            }
        }
    }
}

pub fn conv2d_1x1(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * c_out];
    for oh in 0..h {
        for ow in 0..w {
            for oc in 0..c_out {
                let mut sum = if oc < bias.len() { bias[oc] } else { 0.0 };
                for ic in 0..c_in {
                    let w_idx = oc * c_in + ic;
                    let i_idx = (oh * w + ow) * c_in + ic;
                    sum += weight[w_idx] * input[i_idx];
                }
                out[(oh * w + ow) * c_out + oc] = sum;
            }
        }
    }
    out
}
