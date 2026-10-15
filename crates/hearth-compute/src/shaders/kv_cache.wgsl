// SPDX-License-Identifier: MIT OR Apache-2.0

// Write K and V head data to the layer's KV cache at position `pos`.
//
// k_buf:   [n_kv_heads * head_dim] f32 (from the current token)
// v_buf:   [n_kv_heads * head_dim] f32
// cache:   [K part: n_kv_heads * cache_stride * head_dim]
//          [V part: n_kv_heads * cache_stride * head_dim]  (f32)
//
// Dispatch: n_kv_heads * head_dim threads (1D), workgroup_size=128.

@group(0) @binding(0) var<storage, read> k_buf: array<f32>;
@group(0) @binding(1) var<storage, read> v_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

// params: (pos, n_kv_heads, head_dim, cache_stride)

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pos = params.x;
    let n_kv_heads = params.y;
    let head_dim = params.z;
    let cache_stride = params.w;

    let total_elements = n_kv_heads * head_dim;
    let idx = id.x;
    if (idx >= total_elements) {
        return;
    }

    let head = idx / head_dim;
    let elem_in_head = idx % head_dim;

    // K part: head * cache_stride * head_dim + pos * head_dim + elem_in_head
    let k_cache_idx = head * cache_stride * head_dim + pos * head_dim + elem_in_head;
    cache[k_cache_idx] = k_buf[idx];

    // V part: same layout but after K data
    let v_offset = n_kv_heads * cache_stride * head_dim;
    let v_cache_idx = v_offset + head * cache_stride * head_dim + pos * head_dim + elem_in_head;
    cache[v_cache_idx] = v_buf[idx];
}
