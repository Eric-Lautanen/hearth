// SPDX-License-Identifier: MIT OR Apache-2.0

// Batch RMS norm: processes seq_len independent rows.
// Each workgroup handles one row.
//
// x_buf: [seq_len × d] f32 row-major
// w_buf: [d] f32 (shared weight for all rows)
// out_buf: [seq_len × d] f32 row-major
//
// Dispatch: seq_len workgroups, workgroup_size=128.
// params: (d, eps, post_norm) — same as rms_norm

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>;

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg_id.x;
    let n = u32(params.x);
    let eps = params.y;
    let post_norm = params.z;

    let base = row * n;
    let stride = 128u;
    var sum_sq = 0.0;
    for (var i = lid.x; i < n; i += stride) {
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
    let inv_rms = 1.0 / sqrt(total_sum_sq / f32(n) + eps);

    for (var i = lid.x; i < n; i += stride) {
        if (post_norm == 0.0) {
            out[base + i] = w[i] * x[base + i] * inv_rms;
        } else {
            out[base + i] = (1.0 + w[i]) * x[base + i] * inv_rms;
        }
    }
}
