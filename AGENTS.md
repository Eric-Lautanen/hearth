# Hearth Agent Guidelines

> **Date**: 2026-05 — my training data snapshot may be stale. Use web search tools (`rust-search_*`) liberally for latest papers, code improvements, quantization formats, and LLM inference optimizations. Don't guess — search.

***ALWAYS*** keep bug reports updated and compilation errors and warnings clean.  NEVER USE allow blocks!  NEVER git commit/push unless explicitly asked — this is a working copy, not a shared repo.

## ⚠️ ABSOLUTELY NO PARALLEL EXECUTION ⚠️

**NEVER** launch multiple commands in parallel. This includes:
- **DO NOT** send multiple `bash` tool calls in a single message
- **DO NOT** run benchmarks concurrently — each model must run alone with no other processes competing for CPU
- **DO NOT** parallelize `cargo test` — always use `--test-threads=1`
- **DO NOT** run builds in parallel with other work
- **ALWAYS** send one command at a time, wait for it to complete, then send the next

Running anything in parallel on this 8C/16T CPU causes thermal throttling and SMT contention that invalidates benchmark results and corrupts test output. One command per message. Period.

**IMPORTANT: Never push code to github.  Only use git for diffs if noticable performance degredation.

**IMPORTANT: ALWAYS update BUG_TRACKER.md and at the end of each session completely rewrite RESUME.MD with things to try during the next session!!!

**File purposes:** `RESUME.md` = detailed implementation instructions for the NEXT session (what to do). `BUG_TRACKER.md` = historical record of what was done (perf table, change log, what didn't work). Keep both concise — token efficiency matters for context limits.

**BENCHMARK MANDATE: Always benchmark before-and-after on every session.** Run all 6 models one at a time — NEVER launch multiple benchmarks in the same message. Each model must run alone with no other processes competing for CPU. Compare tok/s to session baseline in BUG_TRACKER.md. If any model degrades, revert the change — do not merge regressions. Benchmark command: `& ".\target\release\hearth-chat-cli.exe" "$model" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw`

## Key paths
- Repo: `C:\Users\ericl\Documents\hearth`
- Ref (Prism fork): `$env:TEMP\llama.cpp-prism`
- Ref binary: `$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe`
- Models: `$env:USERPROFILE\AppData\Roaming\hearth\models\`
  - `Bonsai-1.7B-Q1_0.gguf` (Q1_0_G128, 128/18 blocks) — 28 layers, d=2048, ffn=6144
  - `Ternary-Bonsai-1.7B-Q2_0.gguf` (Q2_0, 128/34 blocks) — 28 layers, d=2048, ffn=6144
  - `Bonsai-4B.gguf` (Q1_0_G128, 128/18 blocks) — 36 layers, d=2560, ffn=9728
  - `Ternary-Bonsai-4B-Q2_0.gguf` (Q2_0, 128/34 blocks) — 36 layers, d=2560, ffn=9728
  - `Bonsai-8B.gguf` (Q1_0_G128, 128/18 blocks) — 36 layers, d=4096, ffn=12288
  - `Ternary-Bonsai-8B-Q2_0.gguf` (Q2_0, 128/34 blocks) — 36 layers, d=4096, ffn=12288

## Ref commands
```powershell
# 1.7B Q1_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-1.7B-Q1_0.gguf" --temp 0 -n 20 -p "Hello"
# 1.7B Q2_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-1.7B-Q2_0.gguf" --temp 0 -n 20 -p "Hello"
# 4B Q1_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-4B.gguf" --temp 0 -n 20 -p "Hello"
# 4B Q2_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-4B-Q2_0.gguf" --temp 0 -n 20 -p "Hello"
# 8B Q1_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-8B.gguf" --temp 0 -n 20 -p "Hello"
# 8B Q2_0
& "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe" -m "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-8B-Q2_0.gguf" --temp 0 -n 20 -p "Hello"
```

## Hearth commands
```powershell
# Generic
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\model.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw
# 1.7B Q1_0
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-1.7B-Q1_0.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw
# 1.7B Q2_0
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-1.7B-Q2_0.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw
# 4B Q1_0
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-4B.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw
# 4B Q2_0
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-4B-Q2_0.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw
# 8B Q1_0
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-8B.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw
# 8B Q2_0
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Ternary-Bonsai-8B-Q2_0.gguf" --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw
```

## Build & verify
```powershell
cargo clean if need be
cargo check/build --release
cargo test --test-threads=1 -p hearth-quant
cargo fmt
```



## Key ref implementation files (Prism fork)
- `ggml/src/ggml-common.h` — block structs (Q1_0, Q2_0, Q8_0, etc.)
- `ggml/src/ggml-cpu/quants.c` — `ggml_vec_dot_q1_0_q8_0_generic`, `ggml_vec_dot_q2_0_q8_0_generic`
- `ggml/include/ggml.h` — enum values for type IDs
- `src/models/qwen3.cpp` — Qwen3 architecture

## Critical format IDs
- 41 = Q1_0 (128-el, 18-byte blocks, 1-bit {-1,+1}) — Prism fork uses 128-el blocks (same as Q1_0_G128!)
- 42 = Q2_0 (128-el, 34-byte blocks, 2-bit values {-1,0,1,2})
- 43 = Q1_0_G128 (128-el, 18-byte blocks, 1-bit {-1,+1})

## Workspace crates
- `hearth-core` — types: Model, PipelineRequest
- `hearth-gguf` — GGUF binary parser (zero-copy mmap)
- `hearth-quant` — CPU quant kernels (Q2V/Q1V lookup tables + SIMD)
- `hearth-tokenizer` — BPE tokenizer
- `hearth-sampler` — temp/top-k/top-p/min-p/rep-penalty
- `hearth-llm` — Llama CPU inference, CLI binary
- `hearth-compute` — GPU stubs (type compat only)

## Quant kernel rules
- Q2_0: `Q2V[256]` → `[[i8;4];256]` (1KB L1), `i32x8` SIMD. See `hearth-quant/src/q2_0.rs`
- Q1_0: `Q1V[256]` → `[[i8;8];256]` (2KB L1), scalar i8×i8→i32. See `hearth-quant/src/q1_0g128.rs`
- DON'T use conditional branches in hot paths (LLVM generates bad code)
- DON'T use `i32x8::new([...])` for <8 elements (construction overhead > SIMD benefit)
- DO use `From<&[f32]>` / `f32x8` for contiguous data (rms_norm, silu, softmax in ops.rs)

## Rust version
Workspace uses `half = "2.6"`, `wide = "1.4"`, `bytemuck = "1"`. `const fn` with `while` loops OK.

## Verifications
After quant kernel changes:
```powershell
cargo check -p hearth-quant
cargo clippy -p hearth-quant -- -D warnings
cargo fmt --check
cargo test --test-threads=1 -p hearth-quant
cargo build --release
```
After any LLM change also check `cargo check -p hearth-llm` (pre-existing clippy warnings in GPU stubs are OK).
