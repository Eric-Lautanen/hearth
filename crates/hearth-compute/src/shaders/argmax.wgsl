// SPDX-License-Identifier: MIT OR Apache-2.0

// Argmax reduction: find the index of the maximum value in an f32 array.
//
// Two-pass approach:
//   Pass 1: Dispatch ceil(N / (WORKGROUP_SIZE * ELEMS_PER_THREAD)) workgroups.
//           Each workgroup finds (max_value, max_index) within its chunk.
//           Outputs partial_max[workgroup_id] and partial_idx[workgroup_id].
//   Pass 2: CPU reads the partial arrays (small, ~hundreds of bytes)
//           and finds the global argmax.
//
// Workgroup size: 128
// Elements per thread: 16 (so each workgroup covers 2048 elements)
//
// Dispatch: ceil(N / 2048) workgroups. For 151699 -> 75 workgroups.
// Partial arrays: 75 f32 + 75 u32 = 600 bytes.

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_max: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_idx: array<u32>;
@group(0) @binding(3) var<uniform> n: u32;
@group(0) @binding(4) var<uniform> elems_per_thread: u32;

var<workgroup> wg_max: array<f32, 128>;
var<workgroup> wg_idx: array<u32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // Each workgroup covers WORKGROUP_SIZE * elems_per_thread elements
    let elems_per_wg = 128u * elems_per_thread;
    let wg_offset = wg_id.x * elems_per_wg;

    // Each thread processes elems_per_thread consecutive elements
    let thread_start = wg_offset + lid.x * elems_per_thread;
    let thread_end = min(thread_start + elems_per_thread, n);

    var my_max: f32 = -3.4028235e+38;
    var my_idx: u32 = 0u;

    for (var i: u32 = thread_start; i < thread_end; i = i + 1u) {
        let v = data[i];
        if (v > my_max) {
            my_max = v;
            my_idx = i;
        }
    }

    // Write to workgroup shared memory
    wg_max[lid.x] = my_max;
    wg_idx[lid.x] = my_idx;
    workgroupBarrier();

    // Tree reduction (64, 32, 16, 8, 4, 2, 1)
    var offset: u32 = 64u;
    while (offset > 0u) {
        if (lid.x < offset) {
            if (wg_max[lid.x + offset] > wg_max[lid.x]) {
                wg_max[lid.x] = wg_max[lid.x + offset];
                wg_idx[lid.x] = wg_idx[lid.x + offset];
            }
        }
        workgroupBarrier();
        offset = offset >> 1u;
    }

    // Write this workgroup's result
    if (lid.x == 0u) {
        partial_max[wg_id.x] = wg_max[0];
        partial_idx[wg_id.x] = wg_idx[0];
    }
}
