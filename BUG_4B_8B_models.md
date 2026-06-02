# Bonsai 4B + 8B: Q1_0 & Q2_0 — Load & Perf

Target: **Both Q1_0 and Q2_0 variants of 4B and 8B Bonsai models must load and generate correctly, with competitive performance vs llama.cpp reference.**

> **⚠️ ALL 6 MODELS ARE TARGETS.** Both Q1_0 and Q2_0 at all sizes. Every change must be benchmarked against all 6.

## Models

| Model | Format | File | Size | d_model | ffn_dim | Layers | Heads | KV Heads |
|-------|--------|------|------|---------|---------|--------|-------|----------|
| Bonsai-4B-Q1_0 | Q1_0 (128/18) | `Bonsai-4B.gguf` | 546 MB | 2560 | 9728 | 36 | 32 | 8 |
| Ternary-Bonsai-4B-Q2_0 | Q2_0 (128/34) | `Ternary-Bonsai-4B-Q2_0.gguf` | 1025 MB | 2560 | 9728 | 36 | 32 | 8 |
| Bonsai-8B-Q1_0 | Q1_0 (128/18) | `Bonsai-8B.gguf` | 1105 MB | 4096 | 12288 | 36 | 32 | 8 |
| Ternary-Bonsai-8B-Q2_0 | Q2_0 (128/34) | `Ternary-Bonsai-8B-Q2_0.gguf` | 2081 MB | 4096 | 12288 | 36 | 32 | 8 |

All models: Qwen3 architecture, head_dim=128, vocab=151669, YaRN rope scaling (factor 4.0), Q/K head norms active.

## Current status (2026-06-02)

| Model | Hearth | Reference | Delta | Notes |
|-------|--------|-----------|-------|-------|
| 4B Q1_0 | **~14** tok/s | 14.1 tok/s | ~0% | On par with reference |
| 4B Q2_0 | **~2.9** tok/s | 1.8 tok/s | **+56%** | Hearth faster! |
| 8B Q1_0 | **~2.5** tok/s | 8.4 tok/s | **-70%** | ⚠️ Major regression |
| 8B Q2_0 | **~1.6** tok/s | 1.6 tok/s | ~0% | On par with reference |

### Key finding: 8B Q1_0 is 3.4× slower than reference

The 8B Q1_0 model at 393ms/token vs reference's 119ms/token. This is the primary perf bug. The 1.7B Q1_0, 4B Q1_0, and all Q2_0 models don't show this regression.

### Q2_0 vs Q1_0 scaling

Q2_0 forward pass is ~5× slower than Q1_0 at same model size (34-byte blocks vs 18-byte blocks for 128 elements). This is consistent across all model sizes and is a fundamental block-size bottleneck.

## Per-token forward pass (warm, non-prefill)

### 4B Q1_0 (~72ms)

| Section | μs/token | % total |
|---|---|---|
| ffn_gate_up_matmul | 27,979 | 39% |
| ffn_down_matmul | 16,112 | 22% |
| kv_cache_write | 107 | 0% |
| qkv_matmul | 11,282 | 16% |
| attn_output_matmul | 7,124 | 10% |
| lm_head_matmul | 5,798 | 8% |
| Other | 3,415 | 5% |
| **TOTAL** | **71,817** | **100%** |

### 4B Q2_0 (~355ms)

| Section | μs/token | % total |
|---|---|---|
| ffn_gate_up_matmul | 139,647 | 40% |
| ffn_down_matmul | 67,416 | 19% |
| qkv_matmul | 55,758 | 16% |
| kv_cache_write | 40,914 | 12% |
| attn_output_matmul | 33,765 | 10% |
| lm_head_matmul | 8,299 | 2% |
| Other | 10,221 | 3% |
| **TOTAL** | **355,043** | **100%** |

### 8B Q1_0 (~393ms)

| Section | μs/token | % total |
|---|---|---|
| ffn_gate_up_matmul | ~152,000 | ~39% |
| ffn_down_matmul | ~95,000 | ~24% |
| qkv_matmul | ~60,000 | ~15% |
| attn_output_matmul | ~36,000 | ~9% |
| lm_head_matmul | ~20,000 | ~5% |
| Other | ~30,000 | ~8% |
| **TOTAL** | **~393,000** | **100%** |

### 8B Q2_0 (~634ms)

| Section | μs/token | % total |
|---|---|---|
| ffn_gate_up_matmul | 291,831 | 46% |
| ffn_down_matmul | 158,658 | 25% |
| qkv_matmul | 102,458 | 16% |
| attn_output_matmul | 57,377 | 9% |
| lm_head_matmul | 12,243 | 2% |
| Other | 11,713 | 2% |
| **TOTAL** | **633,560** | **100%** |

## Root causes (hypotheses)

### 1. 🔴 8B Q1_0 regression: 3.4× slower than reference

Possible causes:
- **Memory bandwidth saturation**: 8B model is 1.1 GB — may exceed effective L3 cache (16MB Zen 4), causing constant DRAM fetches. Reference llama.cpp (MSVC) may have better prefetch or different loop ordering.
- **Thread pool scaling**: Custom pool with `available_parallelism-1 = 15` workers may have contention on the larger model's tensors. 8B Q1_0 has 36 layers × larger matrices = more work items, but each work item is also bigger.
- **Q8_0 KV cache**: 8B uses Q8_0 for KV cache at d_model=4096 vs 4B's 2560. The quant/dequant overhead is higher.
- **Row count**: The 8B's matmul rows (d_model=4096) are 1.6× the 4B's (2560). Par_dot_rows static partitioning may be less balanced.
- **First-token cold start?** The measurement was a single generated token — need multi-token warm-up runs.

### 2. Q2_0 block size: 5× slower than Q1_0

Q2_0 blocks are 34 bytes vs Q1_0's 18 bytes for the same 128 elements. More bytes to fetch from memory per dot product. This is a fundamental format characteristic. Pre-computing Q2V as i16 (2KB LUT) could help (see 1.7B bug report).

### 3. KV cache write overhead (4B Q2_0: 12% != 0%)

4B Q2_0 shows kv_cache_write at 12% of forward pass vs 0-3% for other models. Q8_0 quant/dequant in KV cache path may be misbehaving.

## To try

### 1. 🔴 Debug 8B Q1_0 regression
- Run multi-token warm-up benchmark (20 tokens) to get stable per-token timing
- Test with reduced thread count (--threads 8, 4, 1) to isolate scaling
- Add per-layer timing to find if specific layers bottleneck
- Compare single-thread performance vs reference single-thread

### 2. Q2_0 kernel: pre-computed i16 Q2V LUT
Expand Q2V from `[i8;4]` to `[i16;4]` (2KB). Eliminates `pmovsxbw` sign-extension per Q2_0 batch. See BUG_perf_1.7B_models.md for details.

### 3. Q8_0 KV cache optimization
4B Q2_0 shows 12% KV cache write overhead. Investigate if Q8_0 quant path is falling to scalar.

### 4. Matmul row partitioning for large models
For 8B models with d_model=4096, static row partitioning via `par_dot_rows` may need tuning — the row count per thread may be too large, causing worse L1/L2 cache utilization.

## Source models
- Q1_0: [prism-ml/Bonsai-4B-gguf](https://huggingface.co/prism-ml/Bonsai-4B-gguf), [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf)
- Q2_0: [prism-ml/Ternary-Bonsai-4B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf), [prism-ml/Ternary-Bonsai-8B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf)
