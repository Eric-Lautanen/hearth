// SPDX-License-Identifier: MIT OR Apache-2.0

// Flash attention for bidirectional (non-causal) batch, used by diffusion.
// All positions attend to all other positions (no causal mask).
//
// Buffer layout:
//   q:        [batch × n_heads × head_dim] f32 row-major
//   k:        [seq_len_kv × n_heads × head_dim] f32 row-major
//   v:        [seq_len_kv × n_heads × head_dim] f32 row-major
//   output:   [batch × n_heads × head_dim] f32 row-major
//
// Dispatch: (n_heads, batch, 1) workgroups, workgroup_size=128.
// params: [n_heads, head_dim, seq_len_kv, seq_len_q_dummy]

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 4>;

const MAX_SEQ: u32 = 512u;
var<workgroup> scores: array<f32, MAX_SEQ>;
var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n_heads = params[0];
    let qh = wg_id.x;
    let batch_idx = wg_id.y;
    if (qh >= n_heads) {
        return;
    }
    let head_dim = params[1];
    let seq_len = params[2];

    let num_kv = seq_len; // full bidirectional attention
    let inv_sqrt_d = 1.0 / sqrt(f32(head_dim));

    // Q offset: [batch_idx * n_heads * head_dim + qh * head_dim]
    let q_offset = batch_idx * n_heads * head_dim + qh * head_dim;

    // K/V offsets: K and V are [seq_len × n_heads × head_dim] row-major
    let k_head_offset = qh * head_dim; // within each seq position
    let v_head_offset = qh * head_dim;

    // --- Step 1: Compute attention scores ---
    var local_max: f32 = -3.402823e+38;
    for (var i = lid.x; i < num_kv; i += 128u) {
        let k_off = i * n_heads * head_dim + k_head_offset;
        var dot: f32 = 0.0;
        for (var hd = 0u; hd < head_dim; hd++) {
            dot = dot + q[q_offset + hd] * k[k_off + hd];
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

    for (var i = lid.x; i < num_kv; i += 128u) {
        scores[i] = scores[i] / total_sum;
    }
    workgroupBarrier();

    // --- Step 3: Weighted sum of V values ---
    for (var hd = lid.x; hd < head_dim; hd += 128u) {
        var acc: f32 = 0.0;
        for (var i = 0u; i < num_kv; i++) {
            let v_off = i * n_heads * head_dim + v_head_offset + hd;
            acc = acc + scores[i] * v[v_off];
        }
        output[q_offset + hd] = acc;
    }
}
