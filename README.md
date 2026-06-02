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

Bonsai 1.7B (Qwen3, 28 layers, d=2048) on AMD Ryzen 7 8840HS (Zen 4, 8C/16T, DDR5):

| Model | Hearth | llama.cpp | Ratio |
|-------|--------|-----------|-------|
| Q1_0 (1-bit, 128-el blocks) | **33.5 tok/s** | 34.3 tok/s | tied |
| Q2_0 (2-bit ternary, 128-el blocks) | **24.8 tok/s** | 4.8 tok/s | **5.2× faster** |

Full benchmarks with raw data: [`BENCHMARKS.md`](BENCHMARKS.md)

## Features

- **Pure Rust stack** — GGUF parser, BPE tokenizer, samplers, quant kernels, KV cache, thread pool. No C, no CMake, no `libllama`.
- **Hand-tuned AVX2 kernels** — Q1_0, Q2_0, Q8_0 dot products with lookup-table acceleration and FMA accumulation. SSE4.1 + scalar fallbacks for non-AVX2 CPUs.
- **Custom thread pool** — replaces Rayon with `std::thread::park`/`unpark`. Workers sleep at 0% CPU idle, wake in <1µs. Static work partitioning — no work-stealing overhead. Scales 5.93× on 8 cores (OpenMP: 4.38×).
- **Q8_0 KV cache** — quantized key/value cache reduces memory by 4× with negligible accuracy loss.
- **Chat with templates** — PrismML Bonsai (Qwen3 architecture) chat templates parsed from GGUF metadata. Interactive and single-shot modes.
- **Sampler controls** — temperature, top-k, top-p, min-p, repetition penalty.
- **GPU stubs** — `hearth-compute` has skeleton API signatures for fused dequant+matmul, flash attention, RMS norm, and RoPE. All methods return `None`/`false` — not yet implemented.

## Supported Quantization Formats

| Format | Bits/elem | Block size | Kernel |
|--------|-----------|------------|--------|
| Q1_0 / Q1_0_G128 | 1 | 128 | AVX2 LUT + FMA |
| Q2_0 | 2 | 128 | AVX2 LUT + FMA |
| Q2_K | 2 | 256 | QK block dequant |
| Q3_K | 3 | 256 | QK block dequant |
| Q4_0 | 4 | 32 | Portable + Q8_0 fusion |
| Q4_1 | 4 | 32 | Portable |
| Q4_K | 4 | 256 | QK block dequant |
| Q5_0 | 5 | 32 | Portable |
| Q5_1 | 5 | 32 | Portable |
| Q5_K | 5 | 256 | QK block dequant |
| Q6_K | 6 | 256 | QK block dequant |
| Q8_0 | 8 | 32 | Portable + Q8_0 fusion |
| F16 / F32 | — | — | matrixmultiply sgemm |

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

#### What do the flags mean?

| Flag | Default | What it does |
|------|---------|--------------|
| `<model.gguf>` | required | Path to your model file (comes first, no `--` prefix) |
| `--temp` | 0.7 | Creativity level. `0` = deterministic, `1.0+` = more random |
| `--max-tokens` | 512 | How many words to generate. Lower = faster |
| `--prompt` | — | One-shot mode: ask a question and get one reply, then exit |
| `--prompt-raw` | — | Send prompt as-is without wrapping it in a chat template |
| `--top-k` | 40 | Only pick from the top 40 most likely next words |
| `--top-p` | 0.9 | Nucleus sampling: pick from words totaling 90% probability |
| `--repeat-pen` | 1.1 | Penalize repeated words (1.0 = no penalty) |
| `--kv-f32` | — | Use full-precision KV cache instead of Q8_0 (uses more RAM) |

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

1. **Startup**: N-1 OS threads are spawned (one core reserved for main thread). Workers immediately `park()` — zero CPU idle.
2. **Dispatch**: Matmul calls write a `WorkParams` struct (7× `usize` + function pointer) to a pre-allocated shared buffer. Zero allocation per call.
3. **Signal**: Main thread sets per-worker atomic flags and calls `thread::unpark()`. Workers wake in <1µs.
4. **Execute**: Each worker processes its static chunk of rows sequentially. No work-stealing — matmul rows are perfectly balanced.
5. **Sync**: Main thread spin-waits on per-worker done flags. Workers `park()` again.

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