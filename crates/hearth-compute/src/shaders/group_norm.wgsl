// SPDX-License-Identifier: MIT OR Apache-2.0

// Group Normalization.
// Input/Output: [c × hw] f32 (channel-first, spatial dimensions flattened)
// Weight: [c] f32 (per-channel scale)
// Bias:   [c] f32 (per-channel bias)
//
// Groups channels into n_groups, each group normalized independently.
//
// Dispatch: n_groups workgroups, workgroup_size=128.
// params: (c, hw, n_groups, eps_f32_bits) packed as u32

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>;

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wg_id.x;
    let c = u32(params.x);
    let hw = u32(params.y);
    let n_groups = u32(params.z);
    let eps = params.w;

    if (g >= n_groups) {
        return;
    }

    let c_per_group = c / n_groups;
    let group_start = g * c_per_group;
    let n_elem = c_per_group * hw;
    let stride = 128u;

    // --- Step 1: Compute mean ---
    var sum: f32 = 0.0;
    for (var i = lid.x; i < n_elem; i += stride) {
        let ic = i / hw;
        let ihw = i % hw;
        let idx = (group_start + ic) * hw + ihw;
        sum = sum + x[idx];
    }

    partial[lid.x] = sum;
    workgroupBarrier();

    var offset = 64u;
    while (offset > 0u) {
        if (lid.x < offset) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + offset];
        }
        workgroupBarrier();
        offset = offset >> 1u;
    }

    let mean = partial[0] / f32(n_elem);
    workgroupBarrier();

    // --- Step 2: Compute variance ---
    var var_sum: f32 = 0.0;
    for (var i = lid.x; i < n_elem; i += stride) {
        let ic = i / hw;
        let ihw = i % hw;
        let idx = (group_start + ic) * hw + ihw;
        let diff = x[idx] - mean;
        var_sum = var_sum + diff * diff;
    }

    partial[lid.x] = var_sum;
    workgroupBarrier();

    offset = 64u;
    while (offset > 0u) {
        if (lid.x < offset) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + offset];
        }
        workgroupBarrier();
        offset = offset >> 1u;
    }

    let variance = partial[0] / f32(n_elem);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // --- Step 3: Normalize ---
    for (var i = lid.x; i < n_elem; i += stride) {
        let ic = i / hw;
        let ihw = i % hw;
        let abs_c = group_start + ic;
        let idx = abs_c * hw + ihw;
        x[idx] = (x[idx] - mean) * inv_std * weight[abs_c] + bias[abs_c];
    }
}
