# Continue session: Hearth 4B/8B Bonsai models

> Read `AGENTS.md` and `BUG_4B_8B_models.md` first.
> **⚠️ ALL 6 MODELS ARE TARGETS** — Q1_0 and Q2_0 at 1.7B, 4B, and 8B scales.

## Current state (2026-06-02, init)

### 1.7B (baseline — previously optimized)
| Model | Hearth | Reference | Delta |
|-------|--------|-----------|-------|
| Q1_0 | **34.4** | 32.0 | +7.5% |
| Q2_0 | **27.2** | 5.1 | +433% |

### 4B
| Model | Hearth | Reference | Delta |
|-------|--------|-----------|-------|
| Q1_0 | **~14** | 14.1 | ~0% |
| Q2_0 | **~2.9** | 1.8 | +56% |

### 8B
| Model | Hearth | Reference | Delta |
|-------|--------|-----------|-------|
| Q1_0 | **~2.5** | 8.4 | **-70% 🔴** |
| Q2_0 | **~1.6** | 1.6 | ~0% |

## What was done (this session)

### Downloaded ternary (Q2_0) models from HuggingFace
- `Ternary-Bonsai-4B-Q2_0.gguf` (1.0 GB) from `prism-ml/Ternary-Bonsai-4B-gguf`
- `Ternary-Bonsai-8B-Q2_0.gguf` (2.0 GB) from `prism-ml/Ternary-Bonsai-8B-gguf`

### Verified all 6 models load and generate
All 6 Bonsai models (1.7B/4B/8B × Q1_0/Q2_0) load successfully and produce coherent output. No NaN, no crashes, no architecture mismatches.

### Initial benchmarks against llama.cpp reference
Ran initial 5-token benchmarks for all 4 new models. The 8B Q1_0 shows a major ~70% regression vs reference.

## 🔴 Critical: 8B Q1_0 perf regression

The 8B Q1_0 model at ~393ms/token vs reference's ~119ms/token (8.4 tok/s) is a 3.4× regression. This is the primary focus for next session.

Hypotheses:
1. **Memory bandwidth saturation** — 1.1GB model exceeds L3 cache, constant DRAM fetches hurt more with Hearth's parallel dispatch pattern
2. **Thread pool contention** — 15 workers on 1.6× larger rows (4096 vs 2560) may cause more contention
3. **Single-token measurement noise** — need multi-token warm-up runs
4. **Q8_0 KV cache path** — 8B model has d_model=4096, the Q8_0 KV cache quant/dequant might be a bottleneck

## To try (next sessions)

1. 🔴 **Debug 8B Q1_0**: multi-token warm-up benchmark, per-layer timing, thread count sweep
2. **Q2_0 i16 LUT**: pre-computed Q2V lookup table to eliminate sign-extension
3. **KV cache optimization**: 4B Q2_0 shows 12% time in KV cache writes
4. **Row partitioning tuning**: larger d_model may need different static partitioning strategy

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
