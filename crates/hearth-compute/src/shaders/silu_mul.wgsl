// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused SiLU(gate) * up element-wise operation.
// out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//
// Dispatch: ceil(n / 128) workgroups, workgroup_size=128.

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> n: u32;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= n) {
        return;
    }
    let g = gate[i];
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    let silu_val = g / (1.0 + exp(-g));
    out[i] = silu_val * up[i];
}
