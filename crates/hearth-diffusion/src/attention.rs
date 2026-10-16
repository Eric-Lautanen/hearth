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

/// Parameters for [`attention`]. Grouped so the call stays under the
/// argument-count lint; field order matches the old signature.
pub struct AttentionParams<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub seq_len: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub out: &'a mut [f32],
    pub scores: &'a mut [f32],
}

pub fn attention(p: AttentionParams<'_>) {
    let AttentionParams {
        q,
        k,
        v,
        seq_len,
        head_dim,
        n_heads,
        out,
        scores,
    } = p;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_heads {
        let q_off = h * head_dim;
        for (s, slot) in scores.iter_mut().enumerate().take(seq_len) {
            let mut dot = 0.0f32;
            let k_off = s * n_heads * head_dim + h * head_dim;
            for d in 0..head_dim {
                dot += q[q_off + d] * k[k_off + d];
            }
            *slot = dot * scale;
        }
        softmax_inplace(&mut scores[..seq_len]);
        let o_off = h * head_dim;
        for d in 0..head_dim {
            out[o_off + d] = 0.0;
        }
        for (s, &sc) in scores.iter().enumerate().take(seq_len) {
            let v_off = s * n_heads * head_dim + h * head_dim;
            for d in 0..head_dim {
                out[o_off + d] += sc * v[v_off + d];
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
                for (s, slot) in scores.iter_mut().enumerate().take(seq_len) {
                    let mut dot = 0.0f32;
                    let k_off = s * n_qkv + h * head_dim;
                    for d in 0..head_dim {
                        dot += q_chunk[q_off + d] * k[k_off + d];
                    }
                    *slot = dot * scale;
                    if *slot > max_logit {
                        max_logit = *slot;
                    }
                }

                let mut sum = 0.0f32;
                for slot in scores.iter_mut().take(seq_len) {
                    *slot = (*slot - max_logit).exp();
                    sum += *slot;
                }
                let inv = 1.0 / sum;
                for slot in scores.iter_mut().take(seq_len) {
                    *slot *= inv;
                }

                let o_off = h * head_dim;
                for d in 0..head_dim {
                    o_chunk[o_off + d] = 0.0;
                }
                for (s, &sc) in scores.iter().enumerate().take(seq_len) {
                    let v_off = s * n_qkv + h * head_dim;
                    for d in 0..head_dim {
                        o_chunk[o_off + d] += sc * v[v_off + d];
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
            for (s, slot) in qk[h * seq_len..(h + 1) * seq_len].iter_mut().enumerate() {
                let mut dot = 0.0f32;
                let qo = h * head_dim;
                let ko = s * n_qkv + h * head_dim;
                for d in 0..head_dim {
                    dot += q_chunk[qo + d] * k[ko + d];
                }
                *slot = dot * scale;
            }
        }

        for h in 0..n_heads {
            let qk_slice = &mut qk[h * seq_len..(h + 1) * seq_len];
            let max_val = qk_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for slot in qk_slice.iter_mut() {
                let val = (*slot - max_val).exp();
                *slot = val;
                sum += val;
            }
            let inv = 1.0 / sum;
            for slot in qk_slice.iter_mut() {
                *slot *= inv;
            }
        }

        for h in 0..n_heads {
            let o_off = h * head_dim;
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for (s, &qv) in qk[h * seq_len..(h + 1) * seq_len].iter().enumerate() {
                    sum += qv * v[s * n_qkv + h * head_dim + d];
                }
                unsafe {
                    *((o_chunk_ptr + (o_off + d) * 4) as *mut f32) = sum;
                }
            }
        }
    });
}
