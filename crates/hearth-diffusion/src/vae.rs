use crate::ops;

pub fn decode(latent: &[f32], latent_h: usize, latent_w: usize, in_c: usize, out_h: usize, out_w: usize) -> Vec<u8> {
    // For now, use bilinear upscale + post-processing as a placeholder VAE.
    // The real FLUX VAE decoder requires Conv2d + GroupNorm + ResnetBlock + AttnBlock weights.
    stub_decode(latent, latent_h, latent_w, in_c, out_h, out_w)
}

fn stub_decode(latent: &[f32], lh: usize, lw: usize, c: usize, h: usize, w: usize) -> Vec<u8> {
    // Bilinear upsample from latent resolution to output resolution
    let scale_h = h / lh;
    let scale_w = w / lw;
    let mut upscaled = vec![0.0f32; h * w * c];

    for ly in 0..lh {
        for lx in 0..lw {
            let l_idx = (ly * lw + lx) * c;
            for dy in 0..scale_h {
                for dx in 0..scale_w {
                    let py = ly * scale_h + dy;
                    let px = lx * scale_w + dx;
                    let p_idx = (py * w + px) * c;
                    for ch in 0..c {
                        upscaled[p_idx + ch] = latent[l_idx + ch];
                    }
                }
            }
        }
    }

    // Apply a simple 3×3 convolution to smooth (no learned weights - just averaging)
    let mut smoothed = vec![0.0f32; h * w * 3];
    let kernel: [f32; 9] = [1.0/16.0, 2.0/16.0, 1.0/16.0,
                             2.0/16.0, 4.0/16.0, 2.0/16.0,
                             1.0/16.0, 2.0/16.0, 1.0/16.0];

    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let mut sum = 0.0f32;
                for ky in 0..3i32 {
                    for kx in 0..3i32 {
                        let sy = y as i32 + ky - 1;
                        let sx = x as i32 + kx - 1;
                        if sy >= 0 && sy < h as i32 && sx >= 0 && sx < w as i32 {
                            let src_ch = ch.min(c - 1);
                            sum += kernel[(ky * 3 + kx) as usize]
                                * upscaled[(sy as usize * w + sx as usize) * c + src_ch];
                        }
                    }
                }
                // Apply tanh to get [-1, 1] range
                let val = sum.tanh();
                smoothed[(y * w + x) * 3 + ch] = val;
            }
        }
    }

    // Convert to u8 RGBA
    let mut rgba = vec![0u8; h * w * 4];
    for i in 0..h * w {
        for ch in 0..3 {
            let val = (smoothed[i * 3 + ch] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0;
            rgba[i * 4 + ch] = val as u8;
        }
        rgba[i * 4 + 3] = 255;
    }

    rgba
}

#[allow(dead_code)]
pub struct VaeDecoder {
    z_channels: usize,
    ch: usize,
    ch_mult: Vec<usize>,
    num_res_blocks: usize,
}

impl VaeDecoder {
    #[allow(dead_code)]
    pub fn new(z_channels: usize) -> Self {
        VaeDecoder {
            z_channels,
            ch: 128,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
        }
    }
}

#[allow(dead_code)]
fn group_norm(x: &mut [f32], weight: &[f32], bias: &[f32], n_groups: usize, c: usize, hw: usize, eps: f32) {
    let c_per_group = c / n_groups;
    for g in 0..n_groups {
        let c_start = g * c_per_group;
        let c_end = c_start + c_per_group;
        let n_elem = c_per_group * hw;
        let mean: f32 = x[c_start * hw..c_end * hw].iter().sum::<f32>() / n_elem as f32;
        let var: f32 = x[c_start * hw..c_end * hw]
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>()
            / n_elem as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ic in c_start..c_end {
            for i in 0..hw {
                let idx = ic * hw + i;
                x[idx] = (x[idx] - mean) * inv_std * weight[ic] + bias[ic];
            }
        }
    }
}

#[allow(dead_code)]
fn resnet_block(
    x: &[f32], h: usize, w: usize, in_ch: usize, out_ch: usize,
    norm1_w: &[f32], norm1_b: &[f32],
    conv1_w: &[f32], conv1_b: &[f32],
    norm2_w: &[f32], norm2_b: &[f32],
    conv2_w: &[f32], conv2_b: &[f32],
    shortcut_w: Option<&[f32]>, shortcut_b: Option<&[f32]>,
) -> Vec<f32> {
    let hw = h * w;
    let n_groups = 32;

    // norm1 → SiLU → conv1
    let mut h_out = x.to_vec();
    group_norm(&mut h_out, norm1_w, norm1_b, n_groups, in_ch, hw, 1e-6);
    for v in h_out.iter_mut() { *v = *v / (1.0 + (-*v).exp()); }
    h_out = ops::conv2d_3x3(&h_out, conv1_w, conv1_b, h, w, in_ch, out_ch);

    // norm2 → SiLU → conv2
    group_norm(&mut h_out, norm2_w, norm2_b, n_groups, out_ch, hw, 1e-6);
    for v in h_out.iter_mut() { *v = *v / (1.0 + (-*v).exp()); }
    h_out = ops::conv2d_3x3(&h_out, conv2_w, conv2_b, h, w, out_ch, out_ch);

    // Shortcut
    let shortcut = if in_ch != out_ch {
        if let Some(sw) = shortcut_w {
            let sb = shortcut_b.unwrap_or(&[0.0f32; 0]);
            ops::conv2d_1x1(x, sw, sb, h, w, in_ch, out_ch)
        } else {
            vec![0.0f32; hw * out_ch]
        }
    } else {
        x.to_vec()
    };

    for i in 0..h_out.len() {
        h_out[i] += shortcut[i];
    }
    h_out
}

#[allow(dead_code)]
fn upsample(x: &[f32], h: usize, w: usize, c: usize, conv_w: &[f32], conv_b: &[f32]) -> Vec<f32> {
    let up = ops::upsample_nearest_2x(x, h, w, c);
    ops::conv2d_3x3(&up, conv_w, conv_b, h * 2, w * 2, c, c)
}
