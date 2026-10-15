# Hearth

**Pure Rust LLM inference engine — from GGUF file to generated token with zero C dependencies.**

Hearth is a local LLM runtime that loads quantized GGUF models and runs inference entirely on CPU using hand-tuned AVX2 kernels and a custom parallel thread pool. It matches or beats llama.cpp on the same hardware while being a single `cargo build` away.

```
$ hearth-chat-cli model.gguf --temp 0 --max-tokens 100 --prompt "Explain quaternions"

Quaternions are a number system that extends complex numbers. A quaternion
has one real part and three imaginary parts: q = a + bi + cj + dk where
i² = j² = k² = ijk = -1. They're used extensively in 3D computer graphics
for rotation representation, avoiding gimbal lock...
```

## Performance

All 6 Bonsai models (Qwen3, head_dim=128, vocab=151669, YaRN rope) on AMD Ryzen 7 8840HS (Zen 4, 8C/16T, DDR5). Benchmarked at 50 generated tokens, `--temp 0`, one-word prompt, warm average over 10 runs. Hearth uses 8 workers (n/2). Ref uses llama.cpp-prism 16 threads.

| Size | Format               | d_model | Layers | Hearth         | Ref  | H/Ref |
| ---- | -------------------- | ------- | ------ | -------------- | ---- | ----- |
| 1.7B | Q1_0 (1-bit, 128-el) | 2048    | 28     | **49.5 tok/s** | 32.0 | 1.55× |
| 1.7B | Q2_0 (2-bit ternary) | 2048    | 28     | **28.7 tok/s** | 5.1  | 5.63× |
| 4B   | Q1_0 (1-bit, 128-el) | 2560    | 36     | **23.5 tok/s** | 17.4 | 1.35× |
| 4B   | Q2_0 (2-bit ternary) | 2560    | 36     | **13.0 tok/s** | 2.8  | 4.64× |
| 8B   | Q1_0 (1-bit, 128-el) | 4096    | 36     | **13.4 tok/s** | 8.2  | 1.63× |
| 8B   | Q2_0 (2-bit ternary) | 4096    | 36     | **7.1 tok/s**  | 1.5  | 4.73× |

The two **Ternary Bonsai 2 27B** models (PrismML `qwen35` hybrid architecture, 64 layers) also run,  
at **~1.3 tok/s** (PTQ1_0) and **~0.8 tok/s** (PQ2_0) — ~3 min for 256 tokens, so raise  
`--max-tokens` deliberately. They are the reason the UI shows live progress and streams reasoning  
as it is produced rather than buffering a reply.

Full benchmarks with raw data: [`docs/BUG_TRACKER.md`](docs/BUG_TRACKER.md)

## Features

- **Pure Rust stack** — GGUF parser, BPE tokenizer, samplers, quant kernels, KV cache, thread pool. No C, no CMake, no `libllama`.
- **Hand-tuned AVX2 kernels** — Q1_0, Q2_0, Q8_0 dot products with lookup-table acceleration and FMA accumulation. SSE4.1 + scalar fallbacks for non-AVX2 CPUs.
- **Custom thread pool** — replaces Rayon with a generation-counter spin-loop pool. Zero syscalls per dispatch, <100ns wake latency. Static work partitioning with L2-sized tile iteration — no work-stealing overhead. Scales 5.93× on 8 cores (OpenMP: 4.38×).
- **KV cache** — F32 key/value cache sized to the loaded model's own context window (capped by `HEARTH_CTX_CAP`). A Q8_0 backend exists in the cache type but is not wired to a request yet.
- **Chat with templates** — PrismML Bonsai (Qwen3 architecture) chat templates parsed from GGUF metadata, including the Ternary Bonsai 2 27B hybrid (`qwen35`) architecture. Interactive and single-shot modes.
- **Model-declared defaults** — each GGUF's `general.sampling.*` (temp, top-k, top-p, min-p) and its `context_length` are read and used, so a model runs the way its author tuned it instead of on one hardcoded profile.
- **Sampler controls** — temperature, top-k, top-p, min-p, typical-p, repetition penalty, seed; every one is settable from the CLI, the service protocol, and the web UI.
- **Web chat UI** — `hearth-ui` serves a browser chat over the engine's line-JSON service, with a settings drawer seeded from the loaded model (sampling, context, device/residency) and live tok/s while it replies.
- **GPU stubs** — `hearth-compute` has skeleton API signatures for fused dequant+matmul, flash attention, RMS norm, and RoPE. All methods return `None`/`false` — not yet implemented.

## Supported Quantization Formats

| Format           | Bits/elem | Block size | Kernel                 |
| ---------------- | --------- | ---------- | ---------------------- |
| Q1_0 / Q1_0_G128 | 1         | 128        | AVX2 LUT + FMA         |
| Q2_0             | 2         | 128        | AVX2 LUT + FMA         |
| Q2_K             | 2         | 256        | QK block dequant       |
| Q3_K             | 3         | 256        | QK block dequant       |
| Q4_0             | 4         | 32         | Portable + Q8_0 fusion |
| Q4_1             | 4         | 32         | Portable               |
| Q4_K             | 4         | 256        | QK block dequant       |
| Q5_0             | 5         | 32         | Portable               |
| Q5_1             | 5         | 32         | Portable               |
| Q5_K             | 5         | 256        | QK block dequant       |
| Q6_K             | 6         | 256        | QK block dequant       |
| Q8_0             | 8         | 32         | Portable + Q8_0 fusion |
| F16 / F32        | —         | —          | matrixmultiply sgemm   |

Bold = AVX2-optimized. Portable = scalar but still parallelized.

## Quick Start

### 1. Prerequisites

- **[Rust](https://rustup.rs) 1.85+** (tested on 1.95). If you don't have Rust, install it with `winget install Rustlang.Rustup` on Windows or `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` on Linux/macOS.
- **A Bonsai GGUF model file** placed in a folder you can reference. Two tested models:
  - `Bonsai-1.7B-Q1_0.gguf` — 1-bit, fastest, ~350MB
  - `Ternary-Bonsai-1.7B-Q2_0.gguf` — 2-bit ternary, higher quality, ~630MB
- **x86_64 CPU** with AVX2 recommended (SSE4.1 minimum). Most CPUs from 2015+ have AVX2.

Download models from the [PrismML Bonsai releases](https://huggingface.co/prism-ml).

### 2. Build

Clone the repo and compile. The first build downloads dependencies and takes 2–5 minutes. Subsequent builds are much faster.

```bash
git clone https://github.com/yourname/hearth.git
cd hearth
cargo build --release
```

The build config uses `target-cpu=native`, `lto=fat`, `codegen-units=1` for maximum performance — LLVM optimizes specifically for your CPU.

### 3. Run

The binary builds to `target\release\hearth-chat-cli.exe`. Models typically live wherever you saved them — we'll use `C:\Users\you\models\` as an example below. Replace with your actual paths.

**Verify it works** (quick 20-token test, takes ~1 second):

```powershell
C:\Users\you\hearth\target\release\hearth-chat-cli.exe C:\Users\you\models\Bonsai-1.7B-Q1_0.gguf --temp 0 --max-tokens 20 --prompt "Hello" --prompt-raw
```

**One-shot question** (generates a response and exits):

```powershell
C:\Users\you\hearth\target\release\hearth-chat-cli.exe C:\Users\you\models\Bonsai-1.7B-Q1_0.gguf --prompt "Write a haiku about Rust" --max-tokens 100
```

**Interactive chat** (back-and-forth conversation, Ctrl+C to exit):

```powershell
C:\Users\you\hearth\target\release\hearth-chat-cli.exe C:\Users\you\models\Bonsai-1.7B-Q2_0.gguf --max-tokens 256
```

Interactive commands: `/quit`, `/clear`, `/system <message>`, `/config` (print every active setting).

**Web chat UI:**

```powershell
cargo build --release -p hearth-ui
.\target\release\hearth-ui.exe            # opens http://127.0.0.1:8899
.\target\release\hearth-ui.exe --port 9000 --no-browser
```

The page lists the GGUF files in your model folder, loads one on demand, and streams the reply  
(with `<think>` reasoning in a collapsible drawer). The settings drawer shows **the loaded model's  
sampling values** and has a "reset to model defaults" button; a device/residency select covers  
CPU, GPU, and layer-split loads. During a reply it shows live progress (`streaming · N tok ·
m:ss · X tok/s`) — a 27B model runs at ~1 tok/s, so a long reply is slow but visible rather than  
silent.

#### What do the flags mean?

Nothing here is hardcoded: every Bonsai GGUF ships its own `general.sampling.*` recommendations and  
its own `context_length`, and Hearth follows them unless you pass a flag. Pass **only the fields you  
want to change** — the rest keep the model's values. Run `hearth-chat-cli --help` for the same list.

| Flag                       | Default                  | What it does                                                   |
| -------------------------- | ------------------------ | -------------------------------------------------------------- |
| `<model.gguf>`             | required                 | Path to your model file (comes first, no `--` prefix)          |
| `--temp`                   | model's, else 0.7        | Creativity level. `0` = deterministic, `1.0+` = more random    |
| `--top-k`                  | model's, else 40         | Only pick from the top K most likely next words                |
| `--top-p`                  | model's, else 0.9        | Nucleus sampling: pick from words totaling 90% probability     |
| `--min-p`                  | model's, else 0.0        | Drop tokens below this fraction of the top token's probability |
| `--typical-p`              | 0.0 (off)                | Typical sampling: keep the most information-typical tokens     |
| `--repeat-pen`             | 1.1                      | Penalize repeated words (1.0 = no penalty)                     |
| `--seed`                   | 0 (random)               | Fix the RNG for reproducible replies                           |
| `--max-tokens`             | 512                      | Max new tokens per reply. Lower = faster                       |
| `--max-context`            | model's `context_length` | Context window for prompt + reply, in tokens                   |
| `--prompt`                 | —                        | One-shot mode: ask a question and get one reply, then exit     |
| `--prompt-raw`             | —                        | Send prompt as-is without wrapping it in a chat template       |
| `--show-think`             | off                      | Print `<think>` reasoning instead of hiding it                 |
| `--gpu` / `--gpu-layers N` | cpu                      | Load all layers on GPU, or the first N layers                  |
| `--pin-weights`            | off                      | Read weights into RAM at load instead of mmap                  |
| `--service`                | —                        | Run as the line-JSON stdio service that `hearth-ui` drives     |
| `--dump-info`              | —                        | Print the GGUF metadata and layer-0 tensor dtypes, then exit   |
| `--kv-f32`                 | —                        | Accepted for compatibility; F32 is the only KV backend today   |
| `-h`, `--help`             | —                        | Usage                                                          |

#### Environment variables

| Variable              | Default                                        | What it does                                                                                       |
| --------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `HEARTH_MODELS_DIR`   | repo `models/`, else `%APPDATA%\hearth\models` | Model folder used by the UI dropdown and by `.gguf` resolution                                     |
| `HEARTH_CTX_CAP`      | model's `context_length`                       | Upper bound on the KV cache / context window, in tokens                                            |
| `HEARTH_THREADS`      | ~3/4 of logical cores                          | Worker count for the 27B `qwen35` path (Bonsai models use n/2)                                     |
| `HEARTH_PIN_WEIGHTS`  | off                                            | `=1` is the same as `--pin-weights`                                                                |
| `HEARTH_DUMP_TENSORS` | —                                              | Write layer-0 activations as `<prefix><tensor>.f32` for diffing against a reference implementation |

## Architecture

```
hearth-chat-cli
  └─ hearth-llm         ← inference engine, forward pass, thread pool
       ├─ hearth-core    ← shared types (Model, PipelineRequest)
       ├─ hearth-gguf    ← zero-copy mmap GGUF parser
       ├─ hearth-quant   ← AVX2 dot-product kernels (Q1/Q2/Q8/...)
       ├─ hearth-tokenizer ← BPE tokenizer, chat templates
       ├─ hearth-sampler ← temp, top-k, top-p, min-p, rep-pen
       └─ hearth-compute ← GPU stubs (type-compat only)
```


Each crate is self-contained — you can use `hearth-gguf` to parse GGUF files, `hearth-quant` for quantized dot products, or `hearth-tokenizer` for tokenization independently of the full engine.

### How the Thread Pool Works

Hearth replaces Rayon with a custom `ThreadPool` (`crates/hearth-llm/src/pool.rs`):

1. **Startup**: N-1 OS threads are spawned (one core reserved for main thread). Workers spin-wait on a generation counter — zero syscalls per dispatch.
2. **Dispatch**: Matmul calls write a `WorkParams` struct (8× `usize` + function pointer + `tile_size`) to a pre-allocated shared buffer. Zero allocation per call.
3. **Signal**: Main thread increments an atomic generation counter. Workers see the change via `Acquire` load and latch the new params.
4. **Execute**: Each worker processes its static chunk of rows in L2-sized tiles. No work-stealing — matmul rows are perfectly balanced.
5. **Sync**: Main thread spin-waits on per-worker done flags. Workers spin-loop with occasional `yield_now()` to avoid starving the main thread.

Results: 5.93× parallel scaling on 8 cores vs Rayon's 3.26× and OpenMP's 4.38×.

### AVX2 Kernel Design

The Q1_0 kernel (`hearth-quant/src/q1_0g128.rs`) uses a 2KB lookup table to expand 1-bit weight signs to 8-element `i8` vectors, avoiding branches in the hot path:

```
For each 128-element block:
  ┌─ Weight scale (f16)
  └─ 16 bytes of packed sign bits
      ├─ LUT lookup: byte → [±1, ±1, ..., ±1] × 8
      ├─ vpmovsxbw: activation i8 → i16
      ├─ vpmaddwd: i16 × i16 → i32 (8 lanes)
      ├─ vcvtdq2ps + vfmadd213ps: f32 accumulate across sub-blocks
      └─ Single vhaddps chain at end of row
```

The Q2_0 kernel is similar but uses 2-bit LUT entries (-1, 0, +1, +2). Both kernels accumulate all blocks within a row via `_mm256_fmadd_ps` before a single horizontal sum.

## License

MIT or Apache-2.0, at your option.