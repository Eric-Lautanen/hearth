// SPDX-License-Identifier: MIT OR Apache-2.0

// Q1_0_G128 × f32 batch matmul (dequantize on-the-fly)
//
// Weight layout: Q1_0_G128 blocks (18 bytes per 128 elements)
// Activation: [batch × k] f32 row-major
// Output: [batch × m] f32 row-major
//
// Dispatch: (m, batch, 1) workgroups, workgroup_size=128.
// Each workgroup computes one output element:
//   output[batch_idx * m + row] = dot(weight[row], activation[batch_idx * k + ..])
//
// params: [m, k]

@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> activation: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32, 2>;

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

fn read_u8_weight(byte_off: u32) -> u32 {
    let word_idx = byte_off >> 2u;
    let word = weight[word_idx];
    let shift = (byte_off & 3u) << 3u;
    return (word >> shift) & 0xFFu;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg_id.x;
    let batch_idx = wg_id.y;
    let m = params[0];
    let k = params[1];
    let blocks = k / BLOCK_SIZE;
    let row_byte_off = row * blocks * BLOCK_BYTES;
    let act_base = batch_idx * k;
    var sum: f32 = 0.0;

    for (var b: u32 = 0u; b < blocks; b = b + 1u) {
        let block_byte_off = row_byte_off + b * BLOCK_BYTES;

        // Read f16 scale via invocation 0
        var w_scale: f32 = 1.0;
        if (lid.x == 0u) {
            let scale_lo = read_u8_weight(block_byte_off);
            let scale_hi = read_u8_weight(block_byte_off + 1u);
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
        let sign_byte = read_u8_weight(sign_byte_off);
        let bit = (sign_byte >> (elem_idx & 7u)) & 1u;

        var w_val: f32 = -1.0;
        if (bit == 1u) {
            w_val = 1.0;
        }

        partial[lid.x] = w_val * act_val;
        workgroupBarrier();

        // Tree reduction
        var offset: u32 = 64u;
        while (offset > 0u) {
            if (lid.x < offset) {
                partial[lid.x] = partial[lid.x] + partial[lid.x + offset];
            }
            workgroupBarrier();
            offset = offset >> 1u;
        }

        if (lid.x == 0u) {
            sum = sum + partial[0] * w_scale;
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[batch_idx * m + row] = sum;
    }
}
