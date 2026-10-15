// SPDX-License-Identifier: MIT OR Apache-2.0

// Batch Rotary Position Embedding — applies on-the-fly sin/cos rotation to Q/K heads.
// Processes seq_len rows of heads independently.
//
// x_buf layout: [seq_len × n_heads × head_dim] row-major
// Output is written in-place to x_buf.
//
// Dispatch: (n_heads, seq_len, 1) workgroups, workgroup_size=128.

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32, 8>;
@group(0) @binding(2) var<storage, read> yarn: array<f32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wg_id.x;
    let row = wg_id.y;
    let n_heads = u32(params[0]);
    let head_dim = u32(params[1]);
    let rope_half = u32(params[2]);
    let theta = params[4];
    let freq_scale = params[5];
    let n_dims = params[7];

    let corr_low = yarn[0];
    let corr_high = yarn[1];

    if (head >= n_heads) {
        return;
    }

    // Position `row` = seq_len position (matches batch index)
    let pos = row;
    let base = row * n_heads * head_dim + head * head_dim;

    for (var i = lid.x; i < rope_half; i += 128u) {
        let idx0 = base + i;
        let idx1 = base + i + rope_half;

        let x0 = x[idx0];
        let x1 = x[idx1];

        var freq: f32;

        if (n_dims == 0.0) {
            // Standard RoPE
            let fi = f32(i);
            let inv_freq = 1.0 / pow(theta, (fi + fi) / f32(head_dim));
            freq = f32(pos) * freq_scale * inv_freq;
        } else {
            // YaRN
            let fi = f32(i);
            let inv_freq = 1.0 / pow(theta, (fi + fi) / n_dims);
            let theta_extrap = f32(pos) * inv_freq;
            let theta_interp = freq_scale * theta_extrap;
            let ramp_y = (fi - corr_low) / (corr_high - corr_low + 0.001);
            let ramp_mix = 1.0 - clamp(ramp_y, 0.0, 1.0);
            freq = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        }

        let sin_val = sin(freq);
        let cos_val = cos(freq);
        let mscale = yarn[2];
        let sin_scaled = sin_val * mscale;
        let cos_scaled = cos_val * mscale;

        x[idx0] = x0 * cos_scaled - x1 * sin_scaled;
        x[idx1] = x0 * sin_scaled + x1 * cos_scaled;
    }
}
