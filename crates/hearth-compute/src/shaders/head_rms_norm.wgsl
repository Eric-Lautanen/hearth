// SPDX-License-Identifier: MIT OR Apache-2.0

// Per-head RMS norm — in-place on x_buf.
// x_buf layout: [head_0: head_dim, head_1: head_dim, ...]
// w_buf: single set of weights (size head_dim), shared across all heads.
//
// Workgroup size: 128, dispatch: n_heads workgroups.
// Each workgroup reduces head_dim elements.

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<f32>;

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wg_id.x;
    let head_dim = u32(params.x);
    let eps = params.y;
    let post_norm = params.z;

    let base = head * head_dim;
    let stride = 128u;
    var sum_sq = 0.0;
    for (var i = lid.x; i < head_dim; i += stride) {
        let val = x[base + i];
        sum_sq = sum_sq + val * val;
    }

    partial[lid.x] = sum_sq;
    workgroupBarrier();

    var offset = 64u;
    while (offset > 0u) {
        if (lid.x < offset) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + offset];
        }
        workgroupBarrier();
        offset = offset >> 1u;
    }

    let total_sum_sq = partial[0];
    let inv_rms = 1.0 / sqrt(total_sum_sq / f32(head_dim) + eps);

    for (var i = lid.x; i < head_dim; i += stride) {
        if (post_norm == 0.0) {
            x[base + i] = w[i] * x[base + i] * inv_rms;
        } else {
            x[base + i] = (1.0 + w[i]) * x[base + i] * inv_rms;
        }
    }
}
