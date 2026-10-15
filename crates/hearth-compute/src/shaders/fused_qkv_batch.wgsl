// SPDX-License-Identifier: MIT OR Apache-2.0

// Batch fused Q/KV projection: three Q1_0_G128 × f32 matmuls sharing the same input.
// Processes seq_len positions in one dispatch.
//
// Computes:
//   q_out[seq_pos * nq + i] = dot(q_weight[i], activation[seq_pos * k + ..])
//   k_out[seq_pos * nkv + i] = dot(k_weight[i], activation[seq_pos * k + ..])   if i < n_kv_rows
//   v_out[seq_pos * nkv + i] = dot(v_weight[i], activation[seq_pos * k + ..])   if i < n_kv_rows
//
// Dispatch: (nq, seq_len, 1) workgroups.
// Workgroup size: 128 (maps to Q1_0 block size)
//
// params: (n_kv_rows, k)  where k = d_model = input dimension

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> q_weight: array<u32>;
@group(0) @binding(2) var<storage, read> k_weight: array<u32>;
@group(0) @binding(3) var<storage, read> v_weight: array<u32>;
@group(0) @binding(4) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> k_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_out: array<f32>;
@group(0) @binding(7) var<storage, read> params: array<u32, 3>;

const BLOCK_SIZE: u32 = 128u;
const BLOCK_BYTES: u32 = 18u;
const SCALE_BYTES: u32 = 2u;

var<workgroup> partial: array<f32, 128>;

fn f16_to_f32(lo: u32, hi: u32) -> f32 {
    let sign = f32((hi >> 7u) & 1u);
    let exp = (hi >> 2u) & 0x1Fu;
    let mant = ((hi & 0x3u) << 8u) | (lo & 0xFFu);
    let bits = (u32(sign) << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    return bitcast<f32>(bits);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg_id.x;
    let seq_pos = wg_id.y;
    let k = params[1];
    let nq = params[2];
    let n_kv_rows = params[0];
    let blocks = k / BLOCK_SIZE;
    let row_byte_off = row * blocks * BLOCK_BYTES;
    let act_base = seq_pos * k;

    // Q: always computed
    var q_sum: f32 = 0.0;
    for (var b: u32 = 0u; b < blocks; b = b + 1u) {
        let block_byte_off = row_byte_off + b * BLOCK_BYTES;

        // Read f16 scale
        var w_scale: f32 = 1.0;
        if (lid.x == 0u) {
            let word0 = q_weight[block_byte_off >> 2u];
            let word1 = q_weight[(block_byte_off + 1u) >> 2u];
            let shift0 = (block_byte_off & 3u) << 3u;
            let shift1 = ((block_byte_off + 1u) & 3u) << 3u;
            let scale_lo = (word0 >> shift0) & 0xFFu;
            let scale_hi = (word1 >> shift1) & 0xFFu;
            w_scale = f16_to_f32(scale_lo, scale_hi);
        }
        partial[lid.x] = w_scale;
        workgroupBarrier();
        w_scale = partial[0];
        workgroupBarrier();

        let elem_idx = lid.x;
        let global_elem = act_base + b * BLOCK_SIZE + elem_idx;
        let act_val = activation[global_elem];

        // Read sign bit
        let sign_byte_off = block_byte_off + SCALE_BYTES + (elem_idx >> 3u);
        let word = q_weight[sign_byte_off >> 2u];
        let shift = (sign_byte_off & 3u) << 3u;
        let sign_byte = (word >> shift) & 0xFFu;
        let bit = (sign_byte >> (elem_idx & 7u)) & 1u;

        var w_val: f32 = -1.0;
        if (bit == 1u) {
            w_val = 1.0;
        }

        partial[lid.x] = w_val * act_val;
        workgroupBarrier();

        var offset: u32 = 64u;
        while (offset > 0u) {
            if (lid.x < offset) {
                partial[lid.x] = partial[lid.x] + partial[lid.x + offset];
            }
            workgroupBarrier();
            offset = offset >> 1u;
        }

        if (lid.x == 0u) {
            q_sum = q_sum + partial[0] * w_scale;
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        q_out[seq_pos * nq + row] = q_sum;
    }

    // K and V: only for rows < n_kv_rows
    var k_sum: f32 = 0.0;
    var v_sum: f32 = 0.0;
    if (row < n_kv_rows) {
        for (var b: u32 = 0u; b < blocks; b = b + 1u) {
            let block_byte_off = row_byte_off + b * BLOCK_BYTES;

            // --- K block ---
            var k_scale: f32 = 1.0;
            if (lid.x == 0u) {
                let word0 = k_weight[block_byte_off >> 2u];
                let word1 = k_weight[(block_byte_off + 1u) >> 2u];
                let shift0 = (block_byte_off & 3u) << 3u;
                let shift1 = ((block_byte_off + 1u) & 3u) << 3u;
                let scale_lo = (word0 >> shift0) & 0xFFu;
                let scale_hi = (word1 >> shift1) & 0xFFu;
                k_scale = f16_to_f32(scale_lo, scale_hi);
            }
            partial[lid.x] = k_scale;
            workgroupBarrier();
            k_scale = partial[0];
            workgroupBarrier();

            let elem_idx = lid.x;
            let global_elem = act_base + b * BLOCK_SIZE + elem_idx;
            let act_val = activation[global_elem];

            let k_sign_byte_off = block_byte_off + SCALE_BYTES + (elem_idx >> 3u);
            let k_word = k_weight[k_sign_byte_off >> 2u];
            let k_shift = (k_sign_byte_off & 3u) << 3u;
            let k_sign_byte = (k_word >> k_shift) & 0xFFu;
            let k_bit = (k_sign_byte >> (elem_idx & 7u)) & 1u;

            var k_w_val: f32 = -1.0;
            if (k_bit == 1u) {
                k_w_val = 1.0;
            }

            partial[lid.x] = k_w_val * act_val;
            workgroupBarrier();

            var k_offset: u32 = 64u;
            while (k_offset > 0u) {
                if (lid.x < k_offset) {
                    partial[lid.x] = partial[lid.x] + partial[lid.x + k_offset];
                }
                workgroupBarrier();
                k_offset = k_offset >> 1u;
            }

            if (lid.x == 0u) {
                k_sum = k_sum + partial[0] * k_scale;
            }
            workgroupBarrier();

            // --- V block ---
            var v_scale: f32 = 1.0;
            if (lid.x == 0u) {
                let word0 = v_weight[block_byte_off >> 2u];
                let word1 = v_weight[(block_byte_off + 1u) >> 2u];
                let shift0 = (block_byte_off & 3u) << 3u;
                let shift1 = ((block_byte_off + 1u) & 3u) << 3u;
                let scale_lo = (word0 >> shift0) & 0xFFu;
                let scale_hi = (word1 >> shift1) & 0xFFu;
                v_scale = f16_to_f32(scale_lo, scale_hi);
            }
            partial[lid.x] = v_scale;
            workgroupBarrier();
            v_scale = partial[0];
            workgroupBarrier();

            let v_sign_byte_off = block_byte_off + SCALE_BYTES + (elem_idx >> 3u);
            let v_word = v_weight[v_sign_byte_off >> 2u];
            let v_shift = (v_sign_byte_off & 3u) << 3u;
            let v_sign_byte = (v_word >> v_shift) & 0xFFu;
            let v_bit = (v_sign_byte >> (elem_idx & 7u)) & 1u;

            var v_w_val: f32 = -1.0;
            if (v_bit == 1u) {
                v_w_val = 1.0;
            }

            partial[lid.x] = v_w_val * act_val;
            workgroupBarrier();

            var v_offset: u32 = 64u;
            while (v_offset > 0u) {
                if (lid.x < v_offset) {
                    partial[lid.x] = partial[lid.x] + partial[lid.x + v_offset];
                }
                workgroupBarrier();
                v_offset = v_offset >> 1u;
            }

            if (lid.x == 0u) {
                v_sum = v_sum + partial[0] * v_scale;
            }
            workgroupBarrier();
        }
    }

    if (lid.x == 0u && row < n_kv_rows) {
        let k_out_idx = seq_pos * n_kv_rows + row;
        k_out[k_out_idx] = k_sum;
        v_out[k_out_idx] = v_sum;
    }
}
