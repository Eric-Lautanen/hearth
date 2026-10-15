// SPDX-License-Identifier: MIT OR Apache-2.0

// Conv2D 3x3 with padding=1, stride=1.
// Input:  [c_in × h × w] f32 (channel-first)
// Weight: [c_out × c_in × 3 × 3] f32
// Bias:   [c_out] f32
// Output: [c_out × h × w] f32
//
// Dispatch: ceil(c_out * h * w / 128) workgroups, workgroup_size=128.
// Each thread computes one output element.
// params: (c_in, c_out, h, w)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c_in = params.x;
    let c_out = params.y;
    let h = params.z;
    let w = params.w;

    let total = c_out * h * w;
    let idx = id.x;
    if (idx >= total) {
        return;
    }

    let oc = idx / (h * w);
    let remainder = idx % (h * w);
    let oh = remainder / w;
    let ow = remainder % w;

    var sum = bias[oc];
    for (var ic = 0u; ic < c_in; ic++) {
        for (var kh = 0u; kh < 3u; kh++) {
            let ih = i32(oh) + i32(kh) - 1;
            if (ih < 0 || ih >= i32(h)) { continue; }
            for (var kw = 0u; kw < 3u; kw++) {
                let iw = i32(ow) + i32(kw) - 1;
                if (iw < 0 || iw >= i32(w)) { continue; }
                let w_idx = ((oc * c_in + ic) * 3u + kh) * 3u + kw;
                let i_idx = ic * (h * w) + u32(ih) * w + u32(iw);
                sum += weight[w_idx] * input[i_idx];
            }
        }
    }
    output[idx] = sum;
}
