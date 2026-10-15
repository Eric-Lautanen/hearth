# Hearth vs llama.cpp (Prism fork) — Bonsai/Inference Benchmarks

**Date**: 2026-06-02 (Session 11)
**Commit**: Current HEAD — no kernel changes survived review (S11 was exploration only)

## System

| Component | Detail |
|-----------|--------|
| CPU | AMD Ryzen 7 8840HS (8C/16T, Zen 4) |
| Base / Boost | 3.3 / 4.3 GHz |
| RAM | 16 GB DDR5-5600 |
| OS | Windows 11 Home |
| Rust | 1.95.0 |
| Hearth build | `--release`, `target-cpu=native`, `lto=fat`, `codegen-units=1` |
| llama.cpp build | Prism fork `b1-747eb36`, MSVC Release, `/O2 /arch:AVX2` |
| Hearth workers | 8 (n/2, spin-loop gen-counter pool) |
| llama.cpp threads | 16 (default) |

## Models

All 6 models: Qwen3 architecture, head_dim=128, vocab=151669, YaRN rope scaling factor 4.0, Q/K head norms.

| Model | Format | Size | d_model | ffn_dim | Layers | Heads | KV Heads |
|-------|--------|------|---------|---------|--------|-------|----------|
| Bonsai-1.7B-Q1_0 | Q1_0 128/18 | 293 MB | 2048 | 6144 | 28 | 16 | 8 |
| Ternary-Bonsai-1.7B-Q2_0 | Q2_0 128/34 | 554 MB | 2048 | 6144 | 28 | 16 | 8 |
| Bonsai-4B | Q1_0 128/18 | 546 MB | 2560 | 9728 | 36 | 32 | 8 |
| Ternary-Bonsai-4B-Q2_0 | Q2_0 128/34 | 1025 MB | 2560 | 9728 | 36 | 32 | 8 |
| Bonsai-8B | Q1_0 128/18 | 1105 MB | 4096 | 12288 | 36 | 32 | 8 |
| Ternary-Bonsai-8B-Q2_0 | Q2_0 128/34 | 2081 MB | 4096 | 12288 | 36 | 32 | 8 |

## Methodology

- **50 generation tokens** for both Hearth (`--max-tokens 50`) and llama.cpp (`-n 50`)
- Prompt: `"Hello"` (Hearth: `--prompt-raw`, llama.cpp: `-p "Hello"`)
- `--temp 0` for deterministic greedy sampling
- All runs with system warm (multiple runs completed before data collection)
- Hearth metric: `avg_cpu_overhead` (excludes prompt prefill and sampling)
- llama.cpp metric: `Generation: X.X t/s` (generation speed excluding prompt processing)

## Results

### Multi-threaded Generation (tok/s)

| Model | Hearth (tok/s) | llama.cpp (tok/s) | Speedup |
|-------|:-------------:|:-----------------:|:-------:|
| 1.7B Q1_0 | **43.9** | 33.2 | **1.32×** |
| 1.7B Q2_0 | **27.6** | 5.5 | **5.02×** |
| 4B Q1_0 | **22.6** | 15.3 | **1.48×** |
| 4B Q2_0 | **13.1** | 2.5 | **5.24×** |
| 8B Q1_0 | **13.1** | 9.4 | **1.39×** |
| 8B Q2_0 | **7.6** | 1.4 | **5.43×** |

### Per-token Forward Pass (Hearth timing breakdown)

| Model | Total | qkv | ffn_gate_up | ffn_down | attn_out | lm_head | attn | rope | rest |
|-------|:----:|:---:|:-----------:|:--------:|:--------:|:-------:|:----:|:----:|:----:|
| 1.7B Q1_0 | 18.1ms | 12% | 35% | 21% | 7% | 15% | 4% | 0.02% | 6% |
| 1.7B Q2_0 | 31.1ms | 13% | 37% | 20% | 7% | 16% | 3% | 0.01% | 4% |
| 4B Q1_0 | 41.1ms | 13% | 39% | 22% | 10% | 8% | 4% | 0.02% | 4% |
| 4B Q2_0 | 72.5ms | 14% | 41% | 22% | 10% | 9% | 2% | 0.03% | 2% |
| 8B Q1_0 | 71.9ms | 11% | 44% | 24% | 8% | 7% | 2% | 0.03% | 4% |
| 8B Q2_0 | 127.6ms | 12% | 45% | 24% | 8% | 8% | 1% | 0.02% | 2% |

## Analysis

### Q1_0 models (1-bit {-1,+1})
Hearth beats reference by **1.32–1.48×**. The Q1_0 shuffle kernel (AVX2, no LUT) avoids L1 cache pressure that plagues the LUT-based reference kernel at d=4096.

### Q2_0 models (2-bit {-1,0,1,2})
Hearth beats reference by **5.0–5.4×**. The Prism reference uses a purely scalar kernel (bit shifts and masks, no SIMD). Hearth uses AVX2 LUT with Q2V_I16 pre-expanded table (or AVX-512 VNNI with vpdpbusd on Zen 4).

### Why Q2_0 ref is so slow
The reference `ggml_vec_dot_q2_0_q8_0_generic` in `quants.c:177-222` is entirely scalar:
```c
for (int b = 0; b < 8; ++b) {
    const uint8_t byte = qs[b];
    sumi_block += ((int)((byte >> 0) & 3) - 1) * qy[b*4 + 0];
    sumi_block += ((int)((byte >> 2) & 3) - 1) * qy[b*4 + 1];
    sumi_block += ((int)((byte >> 4) & 3) - 1) * qy[b*4 + 2];
    sumi_block += ((int)((byte >> 6) & 3) - 1) * qy[b*4 + 3];
}
```
No SIMD kernels exist for Q2_0 in the Prism fork. Hearth's AVX2 LUT kernel processes 16 elements per batch with `vpmaddwd`.

### Key insights
- **1.7B models are CPU-bound**: d=2048 is small enough that compute dominates, not memory bandwidth
- **8B models are memory-bandwidth-bound**: d=4096 with larger weight matrices saturate DDR5 bandwidth
- **Q2_0 vs Q1_0 ratio**: 1.7B: 43.9/27.6 = 1.59× Q1_0 is faster (expected: ~1.4× from bits alone)
- **Ref Q2_0 is extreme outlier**: 1.4 tok/s on 8B Q2_0 shows the scalar kernel completely fails on this hardware

## Architecture

Hearth is **pure Rust** — no C dependencies, no FFI, no `unsafe` except for SIMD intrinsics and raw-pointer matmul dispatch. The entire stack (GGUF parser, tokenizer, sampler, quantized dot-product kernels, KV cache, parallel thread pool) is Rust from `gguf` file to generated text.

Key components:
- `hearth-gguf` — zero-copy mmap GGUF parser
- `hearth-quant` — AVX2 Q1_0/Q2_0/Q8_0 dot product kernels
- `hearth-tokenizer` — BPE tokenizer with GGUF template support
- `hearth-sampler` — temperature, top-k, top-p, min-p, repetition penalty
- `hearth-llm` — CPU inference engine with custom `ThreadPool` (spin-loop gen-counter)
- `hearth-compute` — GPU stubs (wgpu, type-compat only)
