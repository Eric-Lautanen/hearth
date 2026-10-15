// SPDX-License-Identifier: MIT OR Apache-2.0

// Q2_0 × f32 matmul (dequantize on-the-fly) with fused add-inplace.
//
// Computes: output[i] = output[i] + dot(weight[i], activation)
// where output is pre-initialized with the accumulator values.
//
// Weight layout: Q2_0 blocks (34 bytes per 128 elements)
//   [0..2)  f16 scale
//   [2..34) 32 bytes packed 2-bit values (4 values per byte)
//     Each 2-bit code: 0→-1, 1→0, 2→+1, 3→+2
// Activation: f32 vector of length k
// Output: f32 vector of length m, read-modify-write
//
// Workgroup size: 128 (maps to Q2_0 block size)
// Dispatch: m workgroups (one per row)

@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> activation: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const BLOCK_SIZE: u32 = 128u;
const BLOCK_BYTES: u32 = 34u;

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

fn q2_code_to_f32(code: u32) -> f32 {
    var result: f32 = -1.0;
    if (code == 1u) { result = 0.0; }
    if (code == 2u) { result = 1.0; }
    if (code == 3u) { result = 2.0; }
    return result;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg_id.x;
    let k = arrayLength(&activation);
    let blocks = k / BLOCK_SIZE;
    let row_byte_off = row * blocks * BLOCK_BYTES;
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
        let global_elem = b * BLOCK_SIZE + elem_idx;
        let act_val = activation[global_elem];

        // Extract 2-bit code: 4 codes per byte
        let byte_idx = elem_idx >> 2u;
        let code_shift = (elem_idx & 3u) << 1u;
        let code_byte_off = block_byte_off + 2u + byte_idx;
        let code_byte = read_u8_weight(code_byte_off);
        let code = (code_byte >> code_shift) & 3u;

        let w_val = q2_code_to_f32(code);

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
        output[row] = output[row] + sum;
    }
}
