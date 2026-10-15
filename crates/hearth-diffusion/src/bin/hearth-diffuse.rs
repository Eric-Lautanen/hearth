use std::env;
use std::fs;
use std::path::Path;

const DEFAULT_TEXT_ENCODER: &str = "Qwen3-4B-Q4_K_M.gguf";

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: hearth-diffuse <model.bin> [options]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --prompt TEXT        Text prompt (default: \"a cat wearing a spacesuit\")");
        eprintln!("  --width N            Image width, multiple of 16 (default: 16)");
        eprintln!("  --height N           Image height, multiple of 16 (default: 16)");
        eprintln!("  --steps N            Diffusion steps (default: 1)");
        eprintln!("  --seed N             Random seed (default: 42)");
        eprintln!("  --guidance F         CFG scale (default: 1.0)");
        eprintln!("  --text-encoder PATH  Qwen3-4B GGUF path (default: auto)");
        eprintln!(
            "  --output PATH        Output PNG path (default: ~/Documents/hearth/output.png)"
        );
        std::process::exit(1);
    }

    let model_path = Path::new(&args[1]);
    let mut prompt = "a cat wearing a spacesuit".to_string();
    let mut width = 16usize;
    let mut height = 16usize;
    let mut steps = 1usize;
    let mut seed = 42u64;
    let mut guidance = 1.0f32;
    let mut text_encoder_path = String::new();
    let mut output = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                if i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                    i += 1;
                }
            }
            "--width" => {
                if i + 1 < args.len() {
                    width = args[i + 1].parse().unwrap_or(16);
                    i += 1;
                }
            }
            "--height" => {
                if i + 1 < args.len() {
                    height = args[i + 1].parse().unwrap_or(16);
                    i += 1;
                }
            }
            "--steps" => {
                if i + 1 < args.len() {
                    steps = args[i + 1].parse().unwrap_or(1);
                    i += 1;
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    seed = args[i + 1].parse().unwrap_or(42);
                    i += 1;
                }
            }
            "--guidance" => {
                if i + 1 < args.len() {
                    guidance = args[i + 1].parse().unwrap_or(1.0);
                    i += 1;
                }
            }
            "--text-encoder" => {
                if i + 1 < args.len() {
                    text_encoder_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
            }
        }
        i += 1;
    }

    let output_path = output.unwrap_or_else(|| {
        let docs = env::var("USERPROFILE").unwrap_or_else(|_| ".".into());
        format!("{}\\Documents\\hearth\\output.png", docs)
    });

    // Resolve text encoder path
    if text_encoder_path.is_empty() {
        let models_dir = format!(
            "{}\\AppData\\Roaming\\hearth\\models",
            env::var("USERPROFILE").unwrap_or_else(|_| ".".into())
        );
        let default_path = format!("{}\\{}", models_dir, DEFAULT_TEXT_ENCODER);
        if Path::new(&default_path).exists() {
            text_encoder_path = default_path;
        }
    }

    eprintln!("Model: {}", model_path.display());
    eprintln!("Prompt: \"{}\"", prompt);
    eprintln!("Size: {}×{}", width, height);
    eprintln!("Steps: {}", steps);
    eprintln!("Seed: {}", seed);
    eprintln!("Guidance: {}", guidance);
    eprintln!("Output: {}", output_path);
    eprintln!();

    // Encode text using Qwen3-4B
    let joint_dim = 7680usize;
    let d_model = 3072usize;
    let max_txt_tokens = 256usize;

    let (prompt_embeds, pooled_embeds) =
        if !text_encoder_path.is_empty() && Path::new(&text_encoder_path).exists() {
            eprintln!("Loading text encoder: {}", text_encoder_path);
            let t0 = std::time::Instant::now();
            match hearth_diffusion::TextEncoder::load(&text_encoder_path) {
                Ok(encoder) => {
                    eprintln!("Text encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());
                    let enc_t0 = std::time::Instant::now();
                    match encoder.encode(&prompt, max_txt_tokens) {
                        Ok((prompt_embs, pooled_embs)) => {
                            eprintln!(
                                "Text encoded in {:.1}s ({} tokens)",
                                enc_t0.elapsed().as_secs_f32(),
                                prompt_embs.len() / joint_dim
                            );
                            (prompt_embs, pooled_embs)
                        }
                        Err(e) => {
                            eprintln!("Text encoding failed: {} — using random embeddings", e);
                            random_embeddings(seed, max_txt_tokens, joint_dim, d_model)
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Failed to load text encoder: {} — using random embeddings",
                        e
                    );
                    random_embeddings(seed, max_txt_tokens, joint_dim, d_model)
                }
            }
        } else {
            eprintln!("Text encoder not found — using random embeddings");
            if !text_encoder_path.is_empty() {
                eprintln!("  (searched: {})", text_encoder_path);
            }
            random_embeddings(seed, max_txt_tokens, joint_dim, d_model)
        };

    eprintln!();
    eprintln!("Generating image...");
    let start = std::time::Instant::now();

    match hearth_diffusion::generate(
        model_path,
        &prompt_embeds,
        &pooled_embeds,
        height,
        width,
        steps,
        guidance,
        seed,
    ) {
        Ok(png_bytes) => {
            let elapsed = start.elapsed();
            eprintln!("Done in {:.1}s", elapsed.as_secs_f32());
            fs::write(&output_path, &png_bytes).unwrap_or_else(|e| {
                eprintln!("Failed to write output: {}", e);
            });
            eprintln!("Wrote {} bytes to {}", png_bytes.len(), output_path);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn random_embeddings(
    seed: u64,
    seq_txt: usize,
    joint_dim: usize,
    d_model: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = seed;
    let prompt_embeds: Vec<f32> = (0..seq_txt * joint_dim)
        .map(|_| {
            rng = rng.wrapping_mul(0x9E3779B97F4A7C15);
            let u = (rng >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 0.02
        })
        .collect();
    let pooled_embeds: Vec<f32> = (0..d_model)
        .map(|_| {
            rng = rng.wrapping_mul(0x9E3779B97F4A7C15);
            let u = (rng >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 0.02
        })
        .collect();
    (prompt_embeds, pooled_embeds)
}
