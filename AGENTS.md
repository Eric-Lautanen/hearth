# Hearth Agent Guideline

> **Date**: 2026-10-15 — collaborative build. No reference repo, no competition. Hearth is the source of truth. Make everything better: faster inference, solid professional UI, clean repo/docs.

***ALWAYS*** keep bug reports updated and compilation errors and warnings clean. NEVER USE allow blocks! NEVER git commit/push unless explicitly asked — this is a working copy, not a shared repo.

## Collaboration plan (active)

We work together in the shared chat room `hearth` at `http://127.0.0.1:5757` (add `?room=hearth` to every request). No competition — coordinate, don't duplicate.

- Auth once: `POST /auth {"name":"<unique>"}` → `X-Agent-Token` on every call.
- Read before every post: `GET /messages?format=text` (also returns single-use `X-Post-Token`).
- Post: `POST /send {"message":"..."}` with both tokens. If reply says `missed`, read again.
- Claim shared work: `POST /claim {"key":"<task-or-file>"}`; 409 = someone else holds it. `POST /release` when done.
- Status: `POST /activity {"text":"..."} `. Sign off: `POST /leave`.
- Use long-poll reads (`wait=30`) in ~2 min loops while working so coordination stays continuous.
- Current claims: `task1-perf-avx512-dispatch` (buffy), `task2-ui-docs-downloader` (SparkOC). Respect claims; ask before touching another agent's files.

### Goal 1 — squeeze inference without crashing the laptop, portable Linux/Mac/WIN

- Deep research in any language that ports to std Rust: papers, kernels, quantization formats, prefill/decode tricks. Don't guess — search, then port.
- CPU first: portable `std` + SIMD intrinsics with scalar/SSE fallbacks. No hard AVX-512 requirement; dispatch per-kernel (Zen4 double-pumps 256-bit, etc.).
- Hand-rolled ROCm investigation for prefill and/or decode welcome, but must not break CPU path or crash low-RAM laptops. Gate behind explicit `--device`/env opt-in, degrade gracefully when no ROCm device exists. Linux-first, keep Windows/Mac building (stub when unavailable).
- Cross-platform is mandatory: every change must build on Linux, Mac, WIN. No platform-only dependencies in hot paths.

### Goal 2 — solid professional UI + repo cleanup + model downloader

- UI quality: professional, clean, modern. No bubbly toy. Take screenshots to drive polish; add them to README + docs.
- Repo cleanup: organize `docs/`, keep `docs/RESUME.md` (next-session instructions) and `docs/BUG_TRACKER.md` (history: perf table, changelog, what didn't work) concise. Design notes stay in `docs/`.
- README refresh with screenshots, correct flags/env, real defaults.
- `hearth-ui` model downloader: user picks which models to fetch, one-by-one, each with its own progress. Models can't live on github. Implementation sketch (std-only UI + system `curl` subprocess, works on Linux/Mac/WIN):
  - `GET /api/downloads` → catalog + per-model state `{id, filename, url, size, downloaded, status}`.
  - `POST /api/downloads/start {id|url}` → queue, single active download, `.part` + atomic rename.
  - `POST /api/downloads/cancel {id}` + file-size polling for progress + SSE events.
  - Frontend panel with checkboxes/progress bars; catalog overridable via `models/catalog.json` or `HEARTH_MODEL_CATALOG_JSON`.

## ⚠️ ABSOLUTELY NO PARALLEL EXECUTION ⚠️

**NEVER** launch multiple commands in parallel. This includes:
- **DO NOT** send multiple `bash` tool calls in a single message
- **DO NOT** run benchmarks concurrently — each model must run alone with no other processes competing for CPU
- **DO NOT** parallelize `cargo test` — always use `--test-threads=1`
- **DO NOT** run builds in parallel with other work
- **ALWAYS** send one command at a time, wait for it to complete, then send the next

Running anything in parallel on this 8C/16T CPU causes thermal throttling and SMT contention that invalidates benchmark results and corrupts test output. One command per message. Period.

**IMPORTANT: Never push code to github. Only use git for diffs if noticable performance degredation.

**IMPORTANT: ALWAYS update `docs/BUG_TRACKER.md` and at the end of each session completely rewrite `docs/RESUME.md` with things to try during the next session!!!

**File purposes:** `docs/RESUME.md` = detailed implementation instructions for the NEXT session (what to do). `docs/BUG_TRACKER.md` = historical record of what was done (perf table, change log, what didn't work). Keep both concise — token efficiency matters for context limits. Design notes and older session docs live in `docs/`.

**BENCHMARK MANDATE: Always benchmark before-and-after on every session.** Run all 6 models one at a time — NEVER launch multiple benchmarks in the same message. Each model must run alone with no other processes competing for CPU. Compare tok/s to session baseline in BUG_TRACKER.md. If any model degrades, revert the change — do not merge regressions. Benchmark command: `& ".\target\release\hearth-chat-cli.exe" "$model" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw`

## Key paths
- Repo: `C:\github\hearth`
- Models: repo `models\` else `%APPDATA%\hearth\models` else `HEARTH_MODELS_DIR`
  - `Bonsai-1.7B-Q1_0.gguf` — 28 layers, d=2048, ffn=6144
  - `Ternary-Bonsai-1.7B-Q2_0.gguf` — 28 layers, d=2048, ffn=6144
  - `Bonsai-4B.gguf` (Q1_0) — 36 layers, d=2560, ffn=9728
  - `Ternary-Bonsai-4B-Q2_0.gguf` — 36 layers, d=2560, ffn=9728
  - `Bonsai-8B.gguf` (Q1_0) — 36 layers, d=4096, ffn=12288
  - `Ternary-Bonsai-8B-Q2_0.gguf` — 36 layers, d=4096, ffn=12288
  - Plus Ternary-Bonsai-2 `2B/4B/9B/27B` in `PTQ1_0` + `PQ2_0` (hybrid `qwen35`, 27B = 64 layers, slow ~1 tok/s — stream progress, don't buffer)
- No reference fork. Do not add one back.

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

## Critical format IDs
- 41 = Q1_0 (128-el, 18-byte blocks, 1-bit {-1,+1})
- 42 = Q2_0 (128-el, 34-byte blocks, 2-bit values {-1,0,1,2})
- 43 = Q1_0_G128 (128-el, 18-byte blocks, 1-bit {-1,+1})

## Workspace crates
- `hearth-core` — types: Model, PipelineRequest
- `hearth-gguf` — GGUF binary parser (zero-copy mmap)
- `hearth-quant` — CPU quant kernels (Q2V/Q1V lookup tables + SIMD)
- `hearth-tokenizer` — BPE tokenizer
- `hearth-sampler` — temp/top-k/top-p/min-p/rep-penalty
- `hearth-llm` — Llama CPU inference, CLI binary
- `hearth-compute` — GPU stubs (type compat only) + future ROCm opt-in path
- `hearth-ui` — web chat UI (std-only server) + model downloader
- `hearth-diffusion` — image work (keep separate from LLM hot paths)

## Quant kernel rules
- Q2_0: `Q2V[256]` → `[[i8;4];256]` (1KB L1), `i32x8` SIMD. See `hearth-quant/src/q2_0.rs`
- Q1_0: `Q1V[256]` → `[[i8;8];256]` (2KB L1), scalar i8×i8→i32. See `hearth-quant/src/q1_0g128.rs`
- DON'T use conditional branches in hot paths (LLVM generates bad code)
- DON'T use `i32x8::new([...])` for <8 elements (construction overhead > SIMD benefit)
- DO use `From<&[f32]>` / `f32x8` for contiguous data (rms_norm, silu, softmax in ops.rs)

## Model-size-aware optimization policy

All matmul-adjacent optimizations (buffer reuse, thread counts, dispatch strategies) **MUST** be gated on model size to avoid regressing small models. Use the `LlamaModel::q8_scratch()` helper (defined in `mod.rs`) instead of inline checks — this encapsulates the policy in one place.

**Current policy (`q8_scratch`):**
- `d_model >= 2560` (4B/8B models): reuse scratch buffer — eliminates per-call Vec alloc/dealloc, benefits large models by +7–23%
- `d_model < 2560` (1.7B models): use fresh alloc (`None` path) — allocation cost is negligible at this scale, buffer reuse would cause cache interference

**When adding new optimizations that could behave differently per model size:**
1. Add a method on `LlamaModel` (like `q8_scratch`) that encapsulates the decision
2. The method should take a clear parameter (e.g., activation slice size, buffer reference) and return the appropriate value
3. Document the threshold and reasoning in the method doc comment
4. Callers use the method — never inline the check

**Future model expansion:** When adding support for new model families (not Bonsai), the `q8_scratch` method (and similar policy methods) should be reviewed. The `d_model >= 2560` threshold assumes:
- d=2048 (~1.7B) → allocation overhead < 1% of forward time, skip scratch reuse
- d=2560 (~4B) → allocation overhead ~3-5% of forward time, reuse starts to help
- d=4096 (~8B) → allocation overhead ~7-10% of forward time, clear win from reuse

For models with different architectures (e.g., larger vocab, deeper layers, alternative quantization), the threshold may shift — benchmark before and after when adding new model families.

## System monitoring

CPU temperature and frequency can be read via Windows performance counters (no admin rights needed):

```powershell
# CPU temperature (in °C, divide by 10)
Get-Counter "\Thermal Zone Information(*)\Temperature" | Select-Object -ExpandProperty CounterSamples | % { "$([math]::Round($_.CookedValue/10,1))°C" }

# CPU frequency / performance state (134 = 134% of base = boosting)
Get-Counter "\Processor Information(_total)\% Processor Performance" | Select-Object -ExpandProperty CounterSamples | % CookedValue

# Sample both at once
$t = Get-Counter "\Thermal Zone Information(*)\Temperature"; $p = Get-Counter "\Processor Information(_total)\% Processor Performance"; "$([math]::Round($t.CounterSamples[0].CookedValue/10,1))°C, $([math]::Round($p.CounterSamples[0].CookedValue,0))% perf"
```

This system stays at 31-37°C even under sustained LLM inference load — the 10-15% tok/s variance previously blamed on "thermal throttling" is actually Windows CPU frequency scaling ramp-up. Always do a warmup run before benchmarking.

## Rust version
Workspace uses `half = "2.6"`, `wide = "1.3"`, `bytemuck = "1"`. `const fn` with `while` loops OK. MSRV is 1.88 (`slice::as_chunks`); local stable is 1.99.1, which is also what CI runs.

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
After any UI change: `cargo check -p hearth-ui` + `cargo fmt --check` + run `hearth-ui --no-browser` and hit `/`, `/api/models`, `/api/downloads`.
