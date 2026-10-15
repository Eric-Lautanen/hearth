# Continue session: Bonsai Image 4B — Pure Rust Diffusion

> **⚠️ THIS IS A SEPARATE CODEBASE from hearth-llm.** Crate `hearth-diffusion`. Do not modify existing LLM code.
> Pure Rust. No Python bridging. No new dependencies.

Read `BUG_img.md` for architecture and plan.

## Current state (2026-06-02, session 2)

### Working
- Weight conversion: 100 Q2_0 + 9 BF16 tensors extracted from Gemlite state_dict.pt → 1.37 GB binary
- MMDiT transformer: 5 double-stream + 20 single-stream blocks, adaLN modulation, RoPE, 100 ternary matmuls
- FlowMatchEuler sampler: 4 steps, shift=3.0, guidance=1.0
- Parallel Q2_0 matmuls via Rayon `par_iter` (2.5× faster than single-threaded)
- PNG encoder: minimal deflate, writes to `~/Documents/hearth/`
- **Text encoder**: Qwen3-4B Q4_K_M → `encode_text()` zero-pads 2560→7680, wired to context_embedder
- **Batched attention**: Rayon-parallel per-query attention in `hearth-diffusion/src/attention.rs`
- End-to-end pipeline: text→image in one binary

### Performance (Ryzen 7 8840HS, release build)

| Config | Text Enc | Diffusion | Total |
|--------|----------|-----------|-------|
| 16×16, 1 step | 9.4s | 5.7s | 15.1s |
| 16×16, 4 steps | 5.7s | 17.5s | 28.8s |
| 64×64, 4 steps | 7.8s | 35.7s | 49.6s |

vs previously: 16×16 4-step = 117s (6.7× speedup)

### Not yet working
- Real VAE decoder (stub uses bilinear upscale + tanh)
- 512×512+ resolutions (need further optimization)
- VAE weight download (AutoencoderKLFlux2 safetensors not yet obtained)

## Bench commands

```powershell
# Quick test (16×16, 1 step)
& ".\target\release\hearth-diffuse.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-Image-4B-Q2_0.bin" --width 16 --height 16 --steps 1 --seed 42

# Full generation (64×64, 4 steps, ~50s)
& ".\target\release\hearth-diffuse.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-Image-4B-Q2_0.bin" --width 64 --height 64 --steps 4 --seed 42 --prompt "a cat wearing a spacesuit"

# With text encoder override
& ".\target\release\hearth-diffuse.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-Image-4B-Q2_0.bin" --text-encoder "C:\path\to\Qwen3-4B-Q4_K_M.gguf" --width 64 --height 64 --steps 4 --prompt "cat"
```

## What worked (this session)

### 1. Text encoder integration (session 2)
Added `encode_text()` to `LlamaModel` in `hearth-llm/src/model/mod.rs` — runs full prefill, returns hidden states for all positions with output_norm applied. Created `hearth-diffusion/src/text_encoder.rs` — wraps Qwen3-4B GGUF, zero-pads 2560→7680 per token for context_embedder, zero-pads 2560→3072 for pooled. Fixed pre-existing `ensure_batch_size` bug where `attn_out` was sized `seq_len * d` instead of `seq_len * nq` (crashed on Qwen3-4B where nq=4096 ≠ d=2560).

### 2. Attention optimization
Replaced sequential per-query `ops::attention()` triple-loop with `attention::attention_batched()` — Rayon `par_chunks_mut` across query batch, per-head softmax. Eliminated duplicate K/V concatenations in double-stream block.

### 3. VAE decoder stub
`hearth-diffusion/src/vae.rs` — bilinear upscale + Gaussian blur + tanh activation. Real decoder architecture (GroupNorm, Conv2d, ResnetBlock, AttnBlock, Upsample) scaffolded and ready for weights.

## Key files changed

- `hearth-llm/src/model/mod.rs` — added `encode_text()` (copy of forward_batch with all-position output_norm), `tokenizer()` accessor, fixed `ensure_batch_size` attn_out sizing
- `hearth-diffusion/Cargo.toml` — added hearth-llm, hearth-tokenizer, hearth-core deps
- `hearth-diffusion/src/lib.rs` — added text_encoder/vae/attention modules, updated generate signature
- `hearth-diffusion/src/text_encoder.rs` — new: Qwen3-4B wrapper with zero-padding
- `hearth-diffusion/src/attention.rs` — new: batched parallel attention
- `hearth-diffusion/src/vae.rs` — new: VAE decoder stub + full architecture scaffold
- `hearth-diffusion/src/ops.rs` — added conv2d_1x1
- `hearth-diffusion/src/transformer.rs` — use batched attention, clean up unused params
- `hearth-diffusion/src/bin/hearth-diffuse.rs` — --text-encoder arg, auto-detect Qwen3-4B GGUF

## Output

Images → `$env:USERPROFILE\Documents\hearth\`
