// SPDX-License-Identifier: MIT OR Apache-2.0

// Batch KV cache write: writes all positions from batched K/V into the layer cache.
//
// k_buf:   [seq_len × n_kv_heads × head_dim] f32 (from batch Q/K/V matmul)
// v_buf:   [seq_len × n_kv_heads × head_dim] f32
// cache:   [K: n_kv_heads × cache_stride × head_dim]
//          [V: n_kv_heads × cache_stride × head_dim]  (f32)
//
// Dispatch: total_elements = seq_len * n_kv_heads * head_dim (1D).

@group(0) @binding(0) var<storage, read> k_buf: array<f32>;
@group(0) @binding(1) var<storage, read> v_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

// params: (seq_len, n_kv_heads, head_dim, cache_stride)

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let seq_len = params.x;
    let n_kv_heads = params.y;
    let head_dim = params.z;
    let cache_stride = params.w;

    let total_elements = seq_len * n_kv_heads * head_dim;
    let idx = id.x;
    if (idx >= total_elements) {
        return;
    }

    // idx encodes: position = idx / (n_kv_heads * head_dim)
    //              head = (idx / head_dim) % n_kv_heads
    //              elem = idx % head_dim
    let pos = idx / (n_kv_heads * head_dim);
    let head_elem = idx % (n_kv_heads * head_dim);
    let head = head_elem / head_dim;
    let elem_in_head = head_elem % head_dim;

    // K part
    let k_cache_idx = head * cache_stride * head_dim + pos * head_dim + elem_in_head;
    cache[k_cache_idx] = k_buf[idx];

    // V part: after K data
    let v_offset = n_kv_heads * cache_stride * head_dim;
    let v_cache_idx = v_offset + head * cache_stride * head_dim + pos * head_dim + elem_in_head;
    cache[v_cache_idx] = v_buf[idx];
}
