// SPDX-License-Identifier: MIT OR Apache-2.0

// Flash attention for batch prefill (seq_len_q > 1) with causal masking.
//
// Buffer layout:
//   q:        [seq_len_q × n_heads × head_dim] f32 row-major
//   kv_cache: [K: n_kv_heads × cache_stride × head_dim]
//             [V: n_kv_heads × cache_stride × head_dim]  (f32)
//   output:   [seq_len_q × n_heads × head_dim] f32 row-major
//
// Dispatch: (n_heads, seq_len_q, 1) workgroups, workgroup_size=128.
// Each workgroup (qh, s) computes attention for query head qh at sequence position s.
// Causal: position s attends to positions 0..=s.
//
// params: [n_heads, n_kv_heads, head_dim, seq_len_q, cache_stride, kv_repeat, num_kv_dummy]

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kv_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32, 7>;

const MAX_SEQ: u32 = 512u;
var<workgroup> scores: array<f32, MAX_SEQ>;
var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n_heads = params[0];
    let qh = wg_id.x;
    let s = wg_id.y;
    if (qh >= n_heads || s >= params[3]) {
        return;
    }
    let n_kv_heads = params[1];
    let head_dim = params[2];
    let cache_stride = params[4];
    let kv_repeat = params[5];

    let num_kv = s + 1; // causal: attend to positions 0..=s
    let kvh = qh / kv_repeat;
    let inv_sqrt_d = 1.0 / sqrt(f32(head_dim));

    // Q offset: [s * n_heads * head_dim + qh * head_dim]
    let q_offset = s * n_heads * head_dim + qh * head_dim;

    // KV cache offsets (in f32 elements)
    let k_head_offset = kvh * cache_stride * head_dim;
    let v_base_offset = n_kv_heads * cache_stride * head_dim;
    let v_head_offset = v_base_offset + kvh * cache_stride * head_dim;

    // --- Step 1: Compute attention scores ---
    var local_max: f32 = -3.402823e+38;
    for (var i = lid.x; i < num_kv; i += 128u) {
        let k_off = k_head_offset + i * head_dim;
        var dot: f32 = 0.0;
        for (var hd = 0u; hd < head_dim; hd++) {
            dot = dot + q[q_offset + hd] * kv_cache[k_off + hd];
        }
        let score = dot * inv_sqrt_d;
        scores[i] = score;
        local_max = max(local_max, score);
    }

    // Tree-reduce max across all invocations
    partial[lid.x] = local_max;
    workgroupBarrier();
    var offset: u32 = 64u;
    while (offset > 0u) {
        if (lid.x < offset) {
            partial[lid.x] = max(partial[lid.x], partial[lid.x + offset]);
        }
        workgroupBarrier();
        offset = offset >> 1u;
    }
    let global_max = partial[0];
    workgroupBarrier();

    // --- Step 2: Softmax (exp, sum, normalize) ---
    var local_sum: f32 = 0.0;
    for (var i = lid.x; i < num_kv; i += 128u) {
        let e = exp(scores[i] - global_max);
        scores[i] = e;
        local_sum = local_sum + e;
    }

    partial[lid.x] = local_sum;
    workgroupBarrier();
    offset = 64u;
    while (offset > 0u) {
        if (lid.x < offset) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + offset];
        }
        workgroupBarrier();
        offset = offset >> 1u;
    }
    let total_sum = partial[0];
    workgroupBarrier();

    // Normalize
    for (var i = lid.x; i < num_kv; i += 128u) {
        scores[i] = scores[i] / total_sum;
    }
    workgroupBarrier();

    // --- Step 3: Weighted sum of V values ---
    for (var hd = lid.x; hd < head_dim; hd += 128u) {
        var acc: f32 = 0.0;
        for (var i = 0u; i < num_kv; i++) {
            let v_off = v_head_offset + i * head_dim + hd;
            acc = acc + scores[i] * kv_cache[v_off];
        }
        output[q_offset + hd] = acc;
    }
}
