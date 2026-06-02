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
- **Custom thread pool** — replaces Rayon with `std::thread::park`/`unpark`. Workers sleep at 0% CPU idle, wake in <1µs. Static work partitioning — no work-stealing overhead. Scales 5.9× on 8 cores (OpenMP: 4.4×).
- **Q8_0 KV cache** — quantized key/value cache reduces memory by 4× with negligible accuracy loss.
- **Chat with templates** — PrismML Bonsai (Qwen3 architecture) chat templates parsed from GGUF metadata. Interactive and single-shot modes.
- **Sampler controls** — temperature, top-k, top-p, min-p, repetition penalty.
- **GPU stubs** — `hearth-compute` has a fully-architected GPU backend API (wgpu-ready) with fused dequant+matmul, flash attention, RMS norm, RoPE shader signatures. Not yet implemented.

## Supported Quantization Formats

| Format | Bits/elem | Block size | Kernel |
|--------|-----------|------------|--------|
| Q1_0 / Q1_0_G128 | 1 | 128 | AVX2 LUT + FMA |
| Q2_0 | 2 | 128 | AVX2 LUT + FMA |
| Q4_0 | 4 | 32 | Portable + Q8_0 fusion |
| Q4_1 | 4 | 32 | Portable |
| Q4_K | 4 | 256 | QK block dequant |
| Q5_0 | 5 | 32 | Portable |
| Q5_1 | 5 | 32 | Portable |
| Q6_K | 6 | 256 | QK block dequant |
| Q8_0 | 8 | 32 | Portable + Q8_0 fusion |
| F16 / F32 | — | — | matrixmultiply sgemm |

Bold = AVX2-optimized. Portable = scalar but still parallelized.

## Quick Start

### Prerequisites

- Rust 1.85+ (tested on 1.95)
- A PrismML Bonsai GGUF model file
- x86_64 CPU with AVX2 recommended (SSE4.1 minimum)

### Build

```bash
git clone https://github.com/yourname/hearth.git
cd hearth
cargo build --release
```

The build config uses `target-cpu=native`, `lto=fat`, `codegen-units=1` for maximum performance. LLVM optimizes specifically for your CPU.

### Run

```bash
# Single-shot — generate and exit
./target/release/hearth-chat-cli model.gguf \
  --temp 0.7 \
  --max-tokens 512 \
  --prompt "Write a haiku about Rust"

# Interactive chat
./target/release/hearth-chat-cli model.gguf

# Benchmark (deterministic greedy sampling)
./target/release/hearth-chat-cli model.gguf \
  --temp 0 \
  --max-tokens 60 \
  --prompt "Hello" \
  --prompt-raw
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `<model.gguf>` | required | Path to GGUF model file |
| `--temp T` | 0.7 | Sampling temperature |
| `--top-k K` | 40 | Top-K sampling |
| `--top-p P` | 0.9 | Top-P (nucleus) sampling |
| `--repeat-pen R` | 1.1 | Repetition penalty |
| `--max-tokens N` | 512 | Max new tokens per reply |
| `--prompt TEXT` | — | Single-shot mode |
| `--prompt-raw` | — | Skip chat template wrapping |
| `--kv-f32` | — | Use F32 KV cache (default: Q8_0) |
| `--gpu` | — | Attempt GPU load (stub — falls back to CPU) |
| `--gpu-layers N` | — | First N layers on GPU (stub) |

## Architecture

```
hearth-chat-cli
  └─ hearth-llm         ← inference engine, forward pass, thread pool
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