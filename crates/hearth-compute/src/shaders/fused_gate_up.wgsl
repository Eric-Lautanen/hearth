// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused gate+up projection: two Q1_0_G128 × f32 matmuls sharing the same input.
//
// Computes:
//   gate_out[i] = dot(gate_weight[i], activation)
//   up_out[i]   = dot(up_weight[i],   activation)
//
// Both weight matrices share the same k dimension and are quantized Q1_0_G128.
// Processing is sequential per block (gate first, then up), reusing shared memory.
//
// Workgroup size: 128 (maps to Q1_0 block size)
// Dispatch: m workgroups (one per row, where m = ffn_dim)

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> gate_weight: array<u32>;
@group(0) @binding(2) var<storage, read> up_weight: array<u32>;
@group(0) @binding(3) var<storage, read_write> gate_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> up_out: array<f32>;

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
    let k = arrayLength(&activation);
    let blocks = k / BLOCK_SIZE;
    let row_byte_off = row * blocks * BLOCK_BYTES;
    var gate_sum: f32 = 0.0;
    var up_sum: f32 = 0.0;

    for (var b: u32 = 0u; b < blocks; b = b + 1u) {
        let block_byte_off = row_byte_off + b * BLOCK_BYTES;

        // --- GATE block ---
        // Read f16 scale via invocation 0
        var g_scale: f32 = 1.0;
        if (lid.x == 0u) {
            let word0 = gate_weight[block_byte_off >> 2u];
            let word1 = gate_weight[(block_byte_off + 1u) >> 2u];
            let shift0 = (block_byte_off & 3u) << 3u;
            let shift1 = ((block_byte_off + 1u) & 3u) << 3u;
            let scale_lo = (word0 >> shift0) & 0xFFu;
            let scale_hi = (word1 >> shift1) & 0xFFu;
            g_scale = f16_to_f32(scale_lo, scale_hi);
        }
        partial[lid.x] = g_scale;
        workgroupBarrier();
        g_scale = partial[0];
        workgroupBarrier();

        let elem_idx = lid.x;
        let global_elem = b * BLOCK_SIZE + elem_idx;
        let act_val = activation[global_elem];

        // Read sign bit for gate
        let g_sign_byte_off = block_byte_off + SCALE_BYTES + (elem_idx >> 3u);
        let g_word = gate_weight[g_sign_byte_off >> 2u];
        let g_shift = (g_sign_byte_off & 3u) << 3u;
        let g_sign_byte = (g_word >> g_shift) & 0xFFu;
        let g_bit = (g_sign_byte >> (elem_idx & 7u)) & 1u;

        var g_w_val: f32 = -1.0;
        if (g_bit == 1u) {
            g_w_val = 1.0;
        }

        partial[lid.x] = g_w_val * act_val;
        workgroupBarrier();

        // Tree reduction for gate
        var offset: u32 = 64u;
        while (offset > 0u) {
            if (lid.x < offset) {
                partial[lid.x] = partial[lid.x] + partial[lid.x + offset];
            }
            workgroupBarrier();
            offset = offset >> 1u;
        }

        if (lid.x == 0u) {
            gate_sum = gate_sum + partial[0] * g_scale;
        }
        workgroupBarrier();

        // --- UP block ---
        // Read f16 scale for up
        var u_scale: f32 = 1.0;
        if (lid.x == 0u) {
            let word0 = up_weight[block_byte_off >> 2u];
            let word1 = up_weight[(block_byte_off + 1u) >> 2u];
            let shift0 = (block_byte_off & 3u) << 3u;
            let shift1 = ((block_byte_off + 1u) & 3u) << 3u;
            let scale_lo = (word0 >> shift0) & 0xFFu;
            let scale_hi = (word1 >> shift1) & 0xFFu;
            u_scale = f16_to_f32(scale_lo, scale_hi);
        }
        partial[lid.x] = u_scale;
        workgroupBarrier();
        u_scale = partial[0];
        workgroupBarrier();

        // Read sign bit for up
        let u_sign_byte_off = block_byte_off + SCALE_BYTES + (elem_idx >> 3u);
        let u_word = up_weight[u_sign_byte_off >> 2u];
        let u_shift = (u_sign_byte_off & 3u) << 3u;
        let u_sign_byte = (u_word >> u_shift) & 0xFFu;
        let u_bit = (u_sign_byte >> (elem_idx & 7u)) & 1u;

        var u_w_val: f32 = -1.0;
        if (u_bit == 1u) {
            u_w_val = 1.0;
        }

        partial[lid.x] = u_w_val * act_val;
        workgroupBarrier();

        // Tree reduction for up
        offset = 64u;
        while (offset > 0u) {
            if (lid.x < offset) {
                partial[lid.x] = partial[lid.x] + partial[lid.x + offset];
            }
            workgroupBarrier();
            offset = offset >> 1u;
        }

        if (lid.x == 0u) {
            up_sum = up_sum + partial[0] * u_scale;
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        gate_out[row] = gate_sum;
        up_out[row] = up_sum;
    }
}
