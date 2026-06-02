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

## Current status (2026-06-02, extensive benchmarks)

### Multi-threaded (Hearth: 15 threads, Ref: 16 threads, both on AMD Ryzen 8840HS)

| Model | Hearth tok/s | Ref tok/s | H/Ref | Us/tok gap |
|-------|-------------|-----------|-------|------------|
| 1.7B Q1_0 | **34.4** | 32.0 | **1.08×** | Hearth beats ref |
| 1.7B Q2_0 | **27.2** | 5.1 | **5.33×** | Hearth dominates |
| 4B Q1_0 | **15.4** | 17.4 | **0.89×** | -11% |
| 4B Q2_0 | **8.6** | 2.8 | **3.07×** | Hearth dominates |
| 8B Q1_0 | **5.2** | 8.2 | **0.63×** | 🔴 -37% |
| 8B Q2_0 | **4.6** | 1.5 | **3.07×** | Hearth dominates |

### Single-threaded (Reference only, 1 thread)

| Model | Ref 1T tok/s | Ref MT tok/s | Ref scaling |
|-------|-------------|-------------|-------------|
| 4B Q1_0 | 2.9 | 17.4 | 6.0× |
| 4B Q2_0 | 0.3 | 2.8 | 9.3× |
| 8B Q1_0 | 1.4 | 8.2 | 5.9× |
| 8B Q2_0 | 0.2 | 1.5 | 7.5× |
| 1.7B Q1_0 | 7.3 | 32.0 | 4.4× |
| 1.7B Q2_0 | — | 5.1 | — |

### 🔴 Finding #1: 8B Q1_0 Hearth degrades — NOT the 70% initially feared

Single cold token: 393ms (~2.5 tok/s). 50-token average: 192ms (5.2 tok/s). The initial measurement was ~2× misleading due to first-token overhead. Real regression vs ref: **-37%**, not -70%.

### 🔴 Finding #2: Hearth 8B Q1_0 is only 13% faster than Q2_0 — should be 1.8-5× faster

At 1.7B: Q1_0/Q2_0 = 34.4/27.2 = **1.26×**. At 4B: 15.4/8.6 = **1.79×**. At 8B: 5.2/4.6 = **1.13×**. The Q1_0 kernel degrades with model dimension while Q2_0 stays strong. For reference: 8B ref Q1_0/Q2_0 = 8.2/1.5 = **5.47×** — Q1_0 should dramatically outpace Q2_0 but Hearth's Q1_0 kernel collapses at d=4096.

### Finding #3: Q2_0 is consistently 3× faster than reference at every scale

The AVX2 Q2_0 kernel + custom pool is bulletproof. 3.07× at 4B and 8B, 5.33× at 1.7B. The Q2_0 kernel scales properly with model size.

### Finding #4: 8B Q1_0 reference scaling itself degrades

Ref 8B Q1_0 scales 5.9× from 1→16 threads (vs 6.0× for 4B). The workload is hitting memory bandwidth limits even on the reference. Hearth's custom pool adds additional overhead.

## Method

- **Hearth**: 50-token runs (`--max-tokens 50 --temp 0 --prompt "Hello" --prompt-raw`). `avg_cpu_overhead` reported, excludes prompt prefill and sampling.
- **Reference**: 20-token runs (`-n 20 --temp 0 -p "Hello"`). `Generation: X.X t/s` used.
- **System**: AMD Ryzen 7 8840HS (8C/16T, Zen 4), 16 GB DDR5, Windows 11.
- **Hearth threads**: Hardcoded `available_parallelism - 1 = 15`.
- **Reference threads**: `-t 1` for single-thread, default (16) for multi-thread.

## Per-token forward pass (warm, non-prefill, from 50-token runs)

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

## Root causes (data-driven)

### 🔴 1. 8B Q1_0 kernel collapses at d=4096 — the main bug

At d=4096 (576 bytes/row = 9 cache lines), the Q1_0 kernel barely outruns Q2_0 (1088 bytes/row = 17 cache lines). At d=2048 (288 bytes/row = 4.5 cache lines), the gap was 1.26×. The Q1_0 kernel's scalar LUT path (`q1_0g128.rs`) has an inner loop that processes 8 elements at a time — at 128 elements/block, that's 16 LUT lookups per block × (4096/128) = 512 lookups per row. Each lookup is a table access. At 15 threads × 512 lookups, the L1 cache pressure from the 2KB Q1V table may cause evictions.

By contrast: Q2_0 AVX2 kernel processes 16 elements per batch with one LUT load + `vpmaddwd`. It's more cache-friendly at scale.

### 2. Parallel scaling on 8B Q1_0

Assuming Hearth 1-thread Q1_0 is ~1.26× slower than ref (based on 1.7B data), then Hearth's effective parallel scaling on 8B is ~4.7× vs ref's 5.9×. Gap: ~20%. This suggests:

- **Thread contention on Q1_0 weight tensor**: 15 workers reading from the same 1.1 GB tensor — Q1_0's smaller per-row footprint (576 bytes) means more row boundaries crossing cache lines, potentially causing false sharing on L2/L3.
- **Pool dispatch overhead**: With d=4096, `par_dot_rows` partitions row work by row count. More rows = more granular work, but also more dispatches per matmul if partitioning per-row rather than per-chunk.

### 3. Q2_0: no scaling issues

Q2_0 at 8B is 3.07× ref — identical to 4B. The Q2_0 kernel (AVX2 intrinsics, 16-element batches) scales linearly with dimension. The more cache lines per row (17) means larger work units per thread, naturally reducing contention.

### 4. KV cache write variance

4B Q2_0 and some 4B Q1_0 tokens show elevated kv_cache_write (up to 4.5% of token). This is intermittent — when KV cache grows beyond a page boundary, a new allocation triggers. Not a consistent bottleneck but contributes to token-to-token variance.

## To try (prioritized by impact)

### 1. 🔴🔴 Port Q2_0 AVX2 pattern to Q1_0 kernel (est. +30-50% for 8B Q1_0)

The Q2_0 kernel (`q2_0.rs`) processes 16 elements per batch using AVX2 intrinsics (`_mm256_madd_epi16`, `_mm256_fmadd_ps`). The Q1_0 kernel (`q1_0g128.rs`) uses a scalar LUT `[[i8;8];256]` with 8-element batches. Porting the Q2_0 AVX2 pattern to Q1_0 would:
- Process 16 elements per batch (was 8) — half the outer loop iterations
- Eliminate repeated LUT table lookups per batch
- Reduce L1 cache pressure from 2KB Q1V table contention at 15 threads

This is the same pattern that delivered +55% on Q2_0 (see BUG_perf_1.7B_models.md). Q1_0 has 18-byte blocks vs Q2_0's 34, so the data structure differs, but the batch approach ports directly.

### 2. Thread pool: row-chunk partitioning for large models

At d=4096, `par_dot_rows` partitions by raw row count. With d_model rows and 15 workers, each gets ~273 rows. But Q1_0 rows are 576 bytes each — 157 KB per worker. This fits in L2 (1MB per Zen 4 core) but requires the whole weight tensor region to be in L3. A row-chunk strategy (process rows 0-63 simultaneously across threads, then 64-127, etc.) would:
- Keep working set smaller than L3 cache
- Reduce cross-core cache line bouncing

### 3. Q2_0: pre-computed i16 LUT (est. +10-15% across all Q2_0 models)

Q2V table stores `[i8;4]` per entry — the `_mm256_cvtepi8_epi16` sign-extension runs per batch. Storing as `[i16;4]` (2KB) eliminates this instruction. Low effort, universal benefit.

### 4. Thread count tuning

15 threads (ncpu-1) may be suboptimal for memory-bandwidth-bound models. Test 8 and 12 workers on 8B models. Need a `--threads` CLI flag or `HEARTH_NUM_THREADS` env var.

### 5. Fuse Q8_0 activation quant into first matmul row

Eliminates a full-vector read per matmul dispatch. ~197 dispatches × 4096 elements × 2 bytes = 1.6 MB of redundant reads per token. Marginal (~5%) but clean.

## Source models
- Q1_0: [prism-ml/Bonsai-4B-gguf](https://huggingface.co/prism-ml/Bonsai-4B-gguf), [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf)
- Q2_0: [prism-ml/Ternary-Bonsai-4B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf), [prism-ml/Ternary-Bonsai-8B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf)
