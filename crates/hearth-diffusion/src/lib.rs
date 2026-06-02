pub mod attention;
pub mod ops;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod weights;

use std::path::Path;
use transformer::{FluxConfig, FluxTransformer};

pub use text_encoder::TextEncoder;

pub fn generate(
    model_path: &Path,
    prompt_embeds: &[f32],
    pooled_embeds: &[f32],
    height: usize,
    width: usize,
    num_steps: usize,
    _guidance: f32,
    seed: u64,
) -> Result<Vec<u8>, String> {
    let w = weights::load_weights(model_path).map_err(|e| format!("load: {}", e))?;
    let cfg = FluxConfig::default();
    let transformer = FluxTransformer::new(cfg, w);

    let latent_h = height / 16;
    let latent_w = width / 16;
    let in_c = 128;
    let d = 3072;

    let seq_img = latent_h * latent_w;
    let mut latent = vec![0.0f32; seq_img * in_c];

    let mut rng_state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(12345);
    for v in latent.iter_mut() {
        rng_state = rng_state.wrapping_mul(0x9E3779B97F4A7C15);
        let u = (rng_state >> 32) as u32;
        *v = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *v *= 0.02;
    }

    let mut img_seq = vec![0.0f32; seq_img * d];
    let x_embedder = transformer.weights.bf16_as_f32("x_embedder")
        .unwrap_or_else(|| vec![0.0f32; d * in_c]);
    for s in 0..seq_img {
        let off_in = s * in_c;
        let off_out = s * d;
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..in_c {
                sum += x_embedder[i * in_c + j] * latent[off_in + j];
            }
            img_seq[off_out + i] = sum;
        }
    }

    let shift = 3.0f32;

    for step in 0..num_steps {
        let t = 1.0 - step as f32 / num_steps as f32;
        let t_next = 1.0 - (step + 1) as f32 / num_steps as f32;

        let t_s = (shift * t) / (1.0 + (shift - 1.0) * t);
        let t_next_s = (shift * t_next) / (1.0 + (shift - 1.0) * t_next);

        let dt = t_next_s - t_s;

        let t_emb = timestep_embedding(t_s, 256);

        let pred = transformer.forward(
            &img_seq,
            prompt_embeds,
            pooled_embeds,
            &t_emb,
            latent_h,
            latent_w,
        );

        for s in 0..seq_img {
            let off = s * in_c;
            for c in 0..in_c {
                latent[off + c] += dt * pred[off + c];
            }
        }
    }

    let pixels = vae::decode(&latent, latent_h, latent_w, in_c, height, width);

    encode_png(&pixels, height, width)
}

fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let mut emb = vec![0.0f32; dim];
    let half = dim / 2;
    for i in 0..half {
        let freq = 1.0 / (10000.0f32.powf(i as f32 / half as f32));
        emb[i] = (t * freq).sin();
        emb[half + i] = (t * freq).cos();
    }
    emb
}

fn encode_png(pixels: &[u8], h: usize, w: usize) -> Result<Vec<u8>, String> {
    use std::io::Write;

    let mut out = Vec::new();
    out.write_all(&[137, 80, 78, 71, 13, 10, 26, 10]).unwrap();

    let mut ihdr_data = Vec::new();
    ihdr_data.write_all(&(w as u32).to_be_bytes()).unwrap();
    ihdr_data.write_all(&(h as u32).to_be_bytes()).unwrap();
    ihdr_data.write_all(&[8, 6, 0, 0, 0]).unwrap();
    write_png_chunk(&mut out, b"IHDR", &ihdr_data);

    let mut raw = Vec::new();
    let row_bytes = w * 4;
    for y in 0..h {
        raw.push(0);
        raw.extend_from_slice(&pixels[y * row_bytes..(y + 1) * row_bytes]);
    }

    let compressed = deflate_raw(&raw);
    write_png_chunk(&mut out, b"IDAT", &compressed);
    write_png_chunk(&mut out, b"IEND", &[]);

    Ok(out)
}

fn deflate_raw(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let chunk_size = 65535usize;
    let mut pos = 0;
    let mut is_final = false;

    while pos < data.len() {
        let remaining = data.len() - pos;
        if remaining <= chunk_size {
            is_final = true;
        }
        let block_len = remaining.min(chunk_size);
        let block_nlen = !block_len as u16;

        out.push(if is_final { 1 } else { 0 });
        out.extend_from_slice(&(block_len as u16).to_le_bytes());
        out.extend_from_slice(&block_nlen.to_le_bytes());
        out.extend_from_slice(&data[pos..pos + block_len]);
        pos += block_len;
    }

    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    for &b in data {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    let adler = (s2 << 16) | s1;
    out.extend_from_slice(&adler.to_be_bytes());

    out
}

fn write_png_chunk(out: &mut Vec<u8>, name: &[u8; 4], data: &[u8]) {
    use std::io::Write;
    out.write_all(&(data.len() as u32).to_be_bytes()).unwrap();
    out.write_all(name).unwrap();
    out.write_all(data).unwrap();

    let mut crc = crc32(name);
    crc = crc32_update(crc, data);
    out.write_all(&crc.to_be_bytes()).unwrap();
}

fn crc32(data: &[u8]) -> u32 {
    crc32_update(0xFFFFFFFF, data) ^ 0xFFFFFFFF
}

fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}
