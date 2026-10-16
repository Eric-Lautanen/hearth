//! Latent → image decoding.
//!
//! Only the placeholder path below is wired up. A real FLUX VAE decoder needs Conv2d +
//! GroupNorm + ResnetBlock + AttnBlock weights from the VAE GGUF, which this crate does not
//! load yet; the half-written scaffolding for it (a `VaeDecoder` config stub and local copies of
//! `group_norm`/`resnet_block`/`upsample`) was removed rather than left behind an allow attribute
//! — it duplicated `crate::ops`, nothing called it, and it is in git history if it is wanted back.

pub fn decode(
    latent: &[f32],
    latent_h: usize,
    latent_w: usize,
    in_c: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<u8> {
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
                    upscaled[p_idx..p_idx + c].copy_from_slice(&latent[l_idx..l_idx + c]);
                }
            }
        }
    }

    // Apply a simple 3×3 convolution to smooth (no learned weights - just averaging)
    let mut smoothed = vec![0.0f32; h * w * 3];
    let kernel: [f32; 9] = [
        1.0 / 16.0,
        2.0 / 16.0,
        1.0 / 16.0,
        2.0 / 16.0,
        4.0 / 16.0,
        2.0 / 16.0,
        1.0 / 16.0,
        2.0 / 16.0,
        1.0 / 16.0,
    ];

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
