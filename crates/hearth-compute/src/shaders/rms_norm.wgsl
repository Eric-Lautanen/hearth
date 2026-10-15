// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused RMS norm: sum-of-squares reduction + element-wise scaling in one pass.
//
// Workgroup size: 128, dispatches 1 workgroup.
// Each invocation strides through d elements, accumulates partial sum_sq in shared memory,
// tree-reduces to get total, then applies scaling.
//
// post_norm=0 (standard):  out[i] = w[i] * x[i] / sqrt(mean(x^2) + eps)
// post_norm=1 (Gemma):      out[i] = (1 + w[i]) * x[i] / sqrt(mean(x^2) + eps)

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>;

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = u32(params.x);
    let eps = params.y;
    let post_norm = params.z; // 0.0 or 1.0

    let stride = 128u;
    var sum_sq = 0.0;
    for (var i = lid.x; i < n; i += stride) {
        let val = x[i];
        sum_sq = sum_sq + val * val;
    }

    partial[lid.x] = sum_sq;
    workgroupBarrier();

    // Tree reduction
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
            out[i] = w[i] * x[i] * inv_rms;
        } else {
            out[i] = (1.0 + w[i]) * x[i] * inv_rms;
        }
    }
}
