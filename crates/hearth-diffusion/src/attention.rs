use rayon::prelude::*;

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
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
        softmax_inplace(&mut scores[..seq_len]);
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

pub fn attention_batched(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    n_heads: usize,
    out: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_qkv = n_heads * head_dim;
    let batch = q.len() / n_qkv;

    out[..batch * n_qkv].fill(0.0);

    out.par_chunks_mut(n_qkv)
        .zip(q.par_chunks(n_qkv))
        .for_each(|(o_chunk, q_chunk)| {
            let mut scores = vec![0.0f32; seq_len];

            for h in 0..n_heads {
                let q_off = h * head_dim;

                let mut max_logit = f32::NEG_INFINITY;
                for s in 0..seq_len {
                    let mut dot = 0.0f32;
                    let k_off = s * n_qkv + h * head_dim;
                    for d in 0..head_dim {
                        dot += q_chunk[q_off + d] * k[k_off + d];
                    }
                    scores[s] = dot * scale;
                    if scores[s] > max_logit {
                        max_logit = scores[s];
                    }
                }

                let mut sum = 0.0f32;
                for s in 0..seq_len {
                    scores[s] = (scores[s] - max_logit).exp();
                    sum += scores[s];
                }
                let inv = 1.0 / sum;
                for s in 0..seq_len {
                    scores[s] *= inv;
                }

                let o_off = h * head_dim;
                for d in 0..head_dim {
                    o_chunk[o_off + d] = 0.0;
                }
                for s in 0..seq_len {
                    let v_off = s * n_qkv + h * head_dim;
                    for d in 0..head_dim {
                        o_chunk[o_off + d] += scores[s] * v[v_off + d];
                    }
                }
            }
        });
}

pub fn attention_batched_matmul(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    n_heads: usize,
    out: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_qkv = n_heads * head_dim;
    let batch = q.len() / n_qkv;

    out[..batch * n_qkv].fill(0.0);

    (0..batch).into_par_iter().for_each(|b| {
        let q_chunk = &q[b * n_qkv..(b + 1) * n_qkv];
        let o_chunk_ptr = out.as_ptr() as usize + b * n_qkv * 4;

        let mut qk = vec![0.0f32; n_heads * seq_len];
        for h in 0..n_heads {
            for s in 0..seq_len {
                let mut dot = 0.0f32;
                let qo = h * head_dim;
                let ko = s * n_qkv + h * head_dim;
                for d in 0..head_dim {
                    dot += q_chunk[qo + d] * k[ko + d];
                }
                qk[h * seq_len + s] = dot * scale;
            }
        }

        for h in 0..n_heads {
            let qk_slice = &mut qk[h * seq_len..(h + 1) * seq_len];
            let max_val = qk_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in 0..seq_len {
                let val = (qk_slice[s] - max_val).exp();
                qk_slice[s] = val;
                sum += val;
            }
            let inv = 1.0 / sum;
            for s in 0..seq_len {
                qk_slice[s] *= inv;
            }
        }

        for h in 0..n_heads {
            let o_off = h * head_dim;
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for s in 0..seq_len {
                    sum += qk[h * seq_len + s] * v[s * n_qkv + h * head_dim + d];
                }
                unsafe {
                    *((o_chunk_ptr + (o_off + d) * 4) as *mut f32) = sum;
                }
            }
        }
    });
}
