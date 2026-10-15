// SPDX-License-Identifier: MIT OR Apache-2.0

// Element-wise: a[i] = a[i] + b[i]
// Writes result back to a (in-place).
//
// Dispatch: ceil(n / 128) workgroups, workgroup_size=128.

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= n) {
        return;
    }
    a[i] = a[i] + b[i];
}
