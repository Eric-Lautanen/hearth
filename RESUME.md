# Continue session: Hearth perf optimization

> **⚠️ BOTH Q1_0 AND Q2_0 ARE CRITICAL TARGETS.** Work on both models.
> Every change must be benchmarked against both. Do NOT optimize for one at the expense of the other.

Read `AGENTS.md` and `BUG_perf_1.7B_models.md` first.

## Current state (2026-06-02, session 2 final)

| Model  | Hearth | Reference | Delta  | Forward pass |
|--------|--------|-----------|--------|-------------|
| Q1_0   | **34.4** | 32.0    | **+7.5%** | 29.1ms |
| Q2_0   | **27.2** | 5.1     | **+433%** | 36.8ms |

**Hearth now BEATS the reference on Q1_0! Q1_0: 18.7→34.4 tok/s (+84%). Q2_0: 17.4→27.2 tok/s (+56%).**

1-thread: Hearth Q1_0 5.8 tok/s vs Ref 7.3 tok/s → kernel codegen gap persists at 1.26×.
But custom thread pool parallel scaling: 19.0→34.4 tok/s — 5.93× scaling (was 3.26× with Rayon).

## What worked (this session, 2026-06-02)

### 7. 🔴🔴 Custom thread pool via `thread::park`/`unpark` — GAME CHANGER (+55% Q1_0, +49% Q2_0)
Replaced all Rayon parallel dispatch with a custom `ThreadPool` using `thread::park`/`unpark` for worker signaling.
- Workers sleep via `thread::park()` between matmuls, consuming 0% CPU idle
- Main thread writes `WorkParams` to shared memory, signals workers via `unpark()`, spin-waits on done flags
- Zero allocation per dispatch — `WorkParams` is just 7× `usize` + function pointer written to a pre-allocated box
- All matmul closures decomposed into uniform `par_dot_rows(w_base, a_ptr, out_ptr, ...)` calls

Key files: `crates/hearth-llm/src/pool.rs` (new), `crates/hearth-llm/src/model/matmul.rs` (converted all Q1_0/Q2_0/Q8_0 paths)

Root cause of Rayon's scaling gap: `rayon::broadcast` uses internal barriers that require thread wake-up synchronization. Custom pool keeps threads parked (0% CPU) and unpark has sub-µs latency. With 197 dispatches per forward pass, the cumulative savings are ~15ms/token.

### 6. Q1_0/Q2_0 kernel hsum optimization (marginal)
Changed inner kernels to accumulate f32 across all Q1_0/Q2_0 blocks using `_mm256_fmadd_ps` instead of per-block scalar `hsum_float_8` + multiply + accumulate. Single hsum at end of row. Within noise range but cleaner code.

### What didn't work (this session)

### 💀 Raw `std::thread::scope` — 5.2 tok/s (Q1_0), 2.8 tok/s (Q2_0)
Creating/joining OS threads per matmul call is catastrophic on Windows. ~100µs thread creation × 113 calls × 16 threads = massive overhead.
Also tried spin-wait thread pool (consumed 100% CPU, starved main thread) — fixed by using `thread::park`/`unpark`.

### 💀 LM head F32 path
Both models have lm_head as quantized dtype (Q1_0 for Bonsai, Q2_0 for Ternary). No F32 weight.

### 💀 Q8_0 activation quant fusion / LLVM codegen flags
Quant pass is ~0.5% of forward pass. `+-slow-unaligned-mem-256` not recognized by Rust LLVM.

## Remaining gap: NONE! Hearth beats reference on Q1_0, dominates Q2_0

### ~26% kernel codegen gap at 1-thread persists (LLVM 5.8 tok/s vs MSVC 7.3 tok/s)
But custom pool parallelism compensates: 5.93× scaling vs reference OpenMP 4.38×.
### Q2_0 kernel: 27.2 tok/s vs 34.4 for Q1_0 (gap due to larger block size: 34 vs 18 bytes)

## To try (next sessions) — see BUG_perf_1.7B_models.md for details

1. **Pre-computed Q2V i16 LUT** — expand `Q2V` from `[i8;4]` to `[i16;4]` (2KB). Eliminates sign-extension per Q2_0 batch. Could close Q2_0→Q1_0 gap.
2. **Kernel micro-optimizations** — low priority given current results.

## Key files changed (all sessions)
- `.cargo/config.toml` — added `target-cpu=native`
- `crates/hearth-llm/src/pool.rs` — **NEW**: custom thread pool with `park`/`unpark`, static row partitioning
- `crates/hearth-llm/src/model/matmul.rs` — converted Q1_0/Q2_0/Q8_0/fused paths to use pool
- `crates/hearth-llm/src/model/mod.rs` — added `pool: ThreadPool` field, threads = ncpu-1
- `crates/hearth-llm/src/lib.rs` — added `mod pool`
- `crates/hearth-quant/src/q1_0g128.rs` — hsum→FMA accumulation across blocks
- `crates/hearth-quant/src/q2_0.rs` — hsum→FMA accumulation across blocks
- `crates/hearth-llm/src/parallel.rs` — now unused (kept for reference)

## Bench commands
```powershell
# Q1_0 — default (best)
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-1.7B-Q1_0.gguf" --temp 0 --max-tokens 100 --prompt "Hello" --prompt-raw

# Q2_0 — ternary model (CRITICAL, work on this too)
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-1.7B-Q2_0.gguf" --temp 0 --max-tokens 100 --prompt "Hello" --prompt-raw

# 1-thread (Q1_0)
$env:RAYON_NUM_THREADS="1"; & ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-1.7B-Q1_0.gguf" --temp 0 --max-tokens 60 --prompt "Hello" --prompt-raw

# Reference Q1_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-1.7B-Q1_0.gguf" --temp 0 -n 20 -p "Hello"

# Reference Q2_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-1.7B-Q2_0.gguf" --temp 0 -n 20 -p "Hello"
```
