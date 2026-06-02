# BUG: Bonsai Image 4B — Pure Rust Diffusion Inference

**Target**: Generate images from text using the Bonsai Image Ternary 4B model — entirely in Rust on consumer AMD hardware (no CUDA, no MLX).

## Model

- **Architecture**: FLUX.2 Klein 4B MMDiT (Flux2Transformer2DModel)
- **Weights**: Ternary {-1, 0, +1} = same Q2_0 format as text model
- **Layers**: 5 double-stream + 20 single-stream MMDiT blocks
- **Dimensions**: d=3072 per stream, 24 heads × 128, joint_dim=7680, ffn=9216
- **Sampler**: FlowMatchEuler, 4 steps, guidance=1.0, shift=3.0
- **Text encoder**: Qwen3-4B Q4_K_M GGUF (2.33 GB) — **integrated**, zero-pad 2560→7680
- **VAE**: FLUX 32-channel latent, tiled 128px decode (stub for now)
- **Resolution**: 1024×1024 (also 512²)

## Current state (2026-06-02 session 2)

### Done

| Component | Status | Notes |
|-----------|--------|-------|
| Weight download | Done | 1.54 GB Gemlite → converted to 1.37 GB Q2_0 binary |
| Weight conversion | Done | `scripts/convert_bonsai_image.py` — 100 Q2_0 + 9 BF16 tensors |
| MMDiT transformer | Done | 5 double-stream + 20 single-stream blocks, 100 matmuls |
| FlowMatchEuler | Done | 4-step, shift=3.0 |
| End-to-end pipeline | Done | Load → encode → denoise → PNG |
| Parallel matmuls | Done | Rayon `par_iter` on Q2_0 batched matmuls |
| Text encoder integration | ✅ | `LlamaModel::encode_text()`, zero-pad 2560→7680 |
| Batched attention | ✅ | `attention::attention_batched()` - Rayon par_chunks_mut |
| PNG encoder | Done | Minimal deflate PNG writer |

### Not yet working

| Component | Status | Notes |
|-----------|--------|-------|
| Real VAE decoder | ❌ | Stub uses bilinear + tanh. Architecture scaffold ready, need AutoencoderKLFlux2 weights |
| Image quality | ❌ | Stub VAE + no real text conditioning = noisy output |
| 512×512+ speeds | ❌ | 64×64 at 50s; 512×512 would be ~50min |

### Performance (16×16, 4 steps, Ryzen 7 8840HS, release build)

| Session | Time | Improvement |
|---------|------|-------------|
| Session 1 (naive) | 117s | — |
| Session 2 (text encoder + batched attention) | 28.8s | 4.1× |

## Files

```
crates/hearth-diffusion/
  Cargo.toml
  src/
    lib.rs            — pipeline, PNG encoder
    transformer.rs    — MMDiT forward pass (double + single stream)
    ops.rs            — Q2_0 matmul, RMS norm, SiLU, Conv2d
    attention.rs      — batched multi-head attention (new)
    text_encoder.rs   — Qwen3-4B GGUF wrapper with zero-padding (new)
    vae.rs            — VAE decoder stub + scaffold (new)
    weights.rs        — Q2_0 binary loader
    bin/
      hearth-diffuse.rs  — CLI with --text-encoder

crates/hearth-llm/
  src/model/mod.rs    — added encode_text(), tokenizer(), fixed attn_out sizing

scripts/
  convert_bonsai_image.py  — Gemlite state_dict.pt → Q2_0 binary

models/
  Bonsai-Image-4B-Q2_0.bin           — converted Q2_0 binary (1.37 GB)
  Qwen3-4B-Q4_K_M.gguf               — text encoder (2.33 GB)
```

## Key design decisions

- **Pure Rust, zero bridging** — no Python at runtime, no FFI to C/CUDA/MLX
- **Separate crate** — `hearth-diffusion` alongside `hearth-llm`, depends on hearth-llm for text encoding
- **Reuse quant kernels** — `hearth-quant` Q2_0 dot products
- **CPU-only** — iGPU later via `hearth-compute`
- **No new deps** — `half`, `bytemuck`, `rayon` only
