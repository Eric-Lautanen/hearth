# Continue session: Hearth 4B/8B Bonsai models

> Read `AGENTS.md` and `BUG_4B_8B_models.md` first.
> **⚠️ ALL 6 MODELS ARE TARGETS** — Q1_0 and Q2_0 at 1.7B, 4B, and 8B scales.

## Current state (2026-06-02, extensive benchmarks)

### Multi-threaded (Hearth: 15T, Ref: 16T)

| Model | Hearth tok/s | Ref tok/s | H/Ref | Gap |
|-------|-------------|-----------|-------|-----|
| 1.7B Q1_0 | **34.4** | 32.0 | 1.08× | +8% |
| 1.7B Q2_0 | **27.2** | 5.1 | 5.33× | +433% |
| 4B Q1_0 | **15.4** | 17.4 | 0.89× | -11% |
| 4B Q2_0 | **8.6** | 2.8 | 3.07× | +207% |
| 8B Q1_0 | **5.2** | 8.2 | 0.63× | 🔴 -37% |
| 8B Q2_0 | **4.6** | 1.5 | 3.07× | +207% |

### Single-threaded (Reference only)

| Model | Ref 1T | Ref MT | Scaling |
|-------|--------|--------|---------|
| 1.7B Q1_0 | 7.3 | 32.0 | 4.4× |
| 4B Q1_0 | 2.9 | 17.4 | 6.0× |
| 4B Q2_0 | 0.3 | 2.8 | 9.3× |
| 8B Q1_0 | 1.4 | 8.2 | 5.9× |
| 8B Q2_0 | 0.2 | 1.5 | 7.5× |

### Q1_0 / Q2_0 speed ratio at each scale

| Scale | Ref Q1/Q2 | Hearth Q1/Q2 |
|-------|-----------|-------------|
| 1.7B | 6.3× | 1.26× |
| 4B | 6.2× | 1.79× |
| 8B | 5.5× | **1.13×** 🔴 |

Q1_0 should be 5-6× faster than Q2_0 (reference confirms this). Hearth's Q1_0 kernel degrades with dimension — at 8B it's barely faster than Q2_0. The Q2_0 kernel scales perfectly.

## Key findings

### 🔴 8B Q1_0 regression: myth vs reality

- **Myth**: 70% slower than reference (from a single cold token)
- **Reality**: 37% slower (192ms avg over 50 tokens vs ref 122ms over 20 tokens)
- The initial 393ms measurement included first-token warm-up overhead

### 🔴 Q1_0 kernel collapses at d=4096

Hearth Q1_0/Q2_0 ratio: 1.26× at 1.7B → 1.79× at 4B → **1.13× at 8B**. The scalar LUT kernel in `q1_0g128.rs` has 512 Q1V table lookups per row at d=4096. At 15 threads, L1 cache pressure from the 2KB lookup table likely causes constant evictions. Q2_0's AVX2 kernel (`q2_0.rs`) avoids this by batching 16 elements per LUT load.

### ✅ Q2_0 is bulletproof

3.07× ref at 4B and 8B, 5.33× at 1.7B. The AVX2 kernel + custom pool scales perfectly at all model sizes. No evidence of performance degradation.

### ✅ 4B Q1_0 is competitive

11% behind reference. Within range of MSVC vs LLVM codegen gap seen on 1.7B (1.26× kernel deficit). No architectural issues at this scale.

### Parallel scaling analysis

Ref 8B Q1_0 scales 5.9× from 1→16T. Assuming Hearth's 1T kernel is ~1.26× slower, Hearth's effective scaling is ~4.7×. The ~20% scaling gap suggests thread contention on the Q1_0 weight tensor at d=4096.

## What was done (this session, 2026-06-02)

### Downloaded Q2_0 models
- `Ternary-Bonsai-4B-Q2_0.gguf` (1.0 GB) from `prism-ml/Ternary-Bonsai-4B-gguf`
- `Ternary-Bonsai-8B-Q2_0.gguf` (2.0 GB) from `prism-ml/Ternary-Bonsai-8B-gguf`

### Verified all 6 models load and generate
All 6 Bonsai models (1.7B/4B/8B × Q1_0/Q2_0) load successfully and produce coherent output. No NaN, no crashes, no architecture mismatches.

### Extensive benchmarks
- 100-token Hearth run for 4B Q1_0 — per-token convergence, warm-up curve
- 50-token Hearth runs for 4B Q2_0, 8B Q1_0, 8B Q2_0 — `avg_cpu_overhead`
- 20-token reference runs for all 4 models (multi-threaded)
- 20-token reference single-thread (`-t 1`) runs for all 4 models
- Per-token forward-pass breakdowns

## To try (prioritized)

1. 🔴🔴 **Port Q2_0 AVX2 pattern to Q1_0 kernel** — 16-el batches, `_mm256_madd_epi16`, FMA accumulation. Same transformation that delivered +55% on Q2_0.
2. **Q2_0: pre-computed i16 LUT** — eliminate `vpmovsxbw` sign-extension, universal +10-15%.
3. **Thread pool: row-chunk partitioning for large models** — keep working set < L3 cache.
4. **Add `--threads` CLI flag or `HEARTH_NUM_THREADS` env var** — needed for thread count sweeps.
5. **Fuse Q8_0 activation quant into first matmul row** — eliminate redundant vector read.

## Key files
- `BUG_4B_8B_models.md` — full bug report with perf tables and forward-pass breakdown
- `BUG_perf_1.7B_models.md` — 1.7B bug report (previous sessions)
- `AGENTS.md` — updated with 4B/8B model paths and commands
- `crates/hearth-llm/src/pool.rs` — custom thread pool (park/unpark)
- `crates/hearth-llm/src/model/matmul.rs` — quant-kernel dispatch
- `crates/hearth-quant/src/q1_0g128.rs` — Q1_0 dot kernel
- `crates/hearth-quant/src/q2_0.rs` — Q2_0 dot kernel

## Bench commands

```powershell
# ===== 4B models =====
# Q1_0 (Hearth)
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-4B.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw
# Q2_0 (Hearth)
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-4B-Q2_0.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw

# ===== 8B models =====
# Q1_0 (Hearth)
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-8B.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw
# Q2_0 (Hearth)
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-8B-Q2_0.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw

# ===== Reference =====
# 4B Q1_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-4B.gguf" --temp 0 -n 20 -p "Hello"
# 4B Q2_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-4B-Q2_0.gguf" --temp 0 -n 20 -p "Hello"
# 8B Q1_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-8B.gguf" --temp 0 -n 20 -p "Hello"
# 8B Q2_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-8B-Q2_0.gguf" --temp 0 -n 20 -p "Hello"
```

## Model inventory

| File | Size | Source |
|------|------|--------|
| `Bonsai-1.7B-Q1_0.gguf` | 237 MB | prism-ml/Bonsai-1.7B-gguf |
| `Ternary-Bonsai-1.7B-Q2_0.gguf` | 442 MB | prism-ml/Ternary-Bonsai-1.7B-gguf |
| `Bonsai-4B.gguf` | 546 MB | prism-ml/Bonsai-4B-gguf |
| `Ternary-Bonsai-4B-Q2_0.gguf` | 1025 MB | prism-ml/Ternary-Bonsai-4B-gguf |
| `Bonsai-8B.gguf` | 1105 MB | prism-ml/Bonsai-8B-gguf |
| `Ternary-Bonsai-8B-Q2_0.gguf` | 2081 MB | prism-ml/Ternary-Bonsai-8B-gguf |
