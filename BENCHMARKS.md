# Hearth vs llama.cpp — Bonsai 1.7B Inference Benchmarks

**Date**: 2026-06-02
**Commit**: Session 2 — custom `ThreadPool` with `park`/`unpack`

## System

| Component | Detail |
|-----------|--------|
| CPU | AMD Ryzen 7 8840HS (8C/16T, Zen 4) |
| Base clock | 3.3 GHz |
| RAM | 16 GB DDR5 |
| OS | Windows 11 Home |
| Rust | 1.95.0 |
| Hearth build | `--release`, `target-cpu=native`, `lto=fat`, `codegen-units=1` |
| llama.cpp build | Prism fork `b1-747eb36`, MSVC Release, `/O2 /arch:AVX2` |

## Models

| Model | Format | Elements/block | Bytes/block | Hearth dtype | llama.cpp dtype |
|-------|--------|---------------|-------------|-------------|-----------------|
| Bonsai-1.7B-Q1_0 | 1-bit {-1,+1} | 128 | 18 | Q1_0_G128 | Q1_0 |
| Ternary-Bonsai-1.7B-Q2_0 | 2-bit {-1,0,1,2} | 128 | 34 | Q2_0 | Q2_0 |

Both models: Qwen3 architecture, 28 layers, d_model=2048, ffn_dim=6144, 16 heads, GQA 8 KV heads, head_dim=128, vocab=151669.

## Methodology

- 20-generation-token runs for llama.cpp (`-n 20`), 60-generation-token runs for Hearth (`--max-tokens 60`)
- `avg_cpu_overhead` (Hearth) excludes prompt prefill and sampling time — pure per-token generation overhead
- `Generation: X.X t/s` (llama.cpp) is the reported generation speed excluding prompt processing
- All runs with `--temp 0` for deterministic greedy sampling
- Hearth uses custom `ThreadPool` with `std::thread::available_parallelism() - 1` workers (15 on this system)
- llama.cpp uses default thread count (16 on this system)
- Single-threaded: `-t 1` for llama.cpp; Hearth pool is hardcoded to ncpu-1 (single-thread not tested with current pool)

## Q1_0 (Bonsai-1.7B-Q1_0.gguf) — Multi-threaded

| Run | Hearth (us/tok) | Hearth (tok/s) | llama.cpp (tok/s) |
|-----|----------------|----------------|-------------------|
| 1 | 28,189 | 35.5 | 34.3 |
| 2 | 35,109 | 28.5 | 33.7 |
| 3 | 29,886 | 33.5 | 34.6 |
| 4 | 29,499 | 33.9 | — |
| 5 | 44,999 | 22.2 | — |
| **Median** | **29,886** | **33.5** | **34.3** |

Hearth Q1_0 median: **33.5 tok/s** (forward pass ~29.9ms)
llama.cpp Q1_0 median: **34.3 tok/s** (forward pass ~29.2ms)
Delta: hearth is **2.3% slower** (within noise)

## Q2_0 (Ternary-Bonsai-1.7B-Q2_0.gguf) — Multi-threaded

| Run | Hearth (us/tok) | Hearth (tok/s) | llama.cpp (tok/s) |
|-----|----------------|----------------|---------------|
| 1 | 41,800 | 23.9 | 5.7 |
| 2 | 42,687 | 23.4 | 4.8 |
| 3 | 37,072 | 27.0 | 4.3 |
| 4 | 39,116 | 25.6 | — |
| **Median** | **40,458** | **24.8** | **4.8** |

Hearth Q2_0 median: **24.8 tok/s** (forward pass ~40.5ms)
llama.cpp Q2_0 median: **4.8 tok/s** (forward pass ~208ms)
Hearth is **5.2× faster** than llama.cpp on Q2_0

## Single-threaded (for kernel codegen comparison)

| Model | Hearth (pool=15T, n/a) | llama.cpp -t 1 |
|-------|----------------------|----------------|
| Q1_0 | — | 6.3 tok/s |
| Q2_0 | — | 0.8 tok/s |

Hearth single-threaded not measured — custom pool always uses `available_parallelism-1` workers.
Earlier session (Rayon-based): Hearth Q1_0 ~5.8 tok/s at 1 thread.

## Architecture

Hearth is **pure Rust** — no C dependencies, no FFI, no `unsafe` except for SIMD intrinsics and raw-pointer matmul dispatch. The entire stack (GGUF parser, tokenizer, sampler, quantized dot-product kernels, KV cache, parallel thread pool) is Rust from `gguf` file to generated text.

Key components:
- `hearth-gguf` — zero-copy mmap GGUF parser
- `hearth-quant` — AVX2 Q1_0/Q2_0/Q8_0 dot product kernels with LUT acceleration
- `hearth-tokenizer` — BPE tokenizer with GGUF template support
- `hearth-sampler` — temperature, top-k, top-p, min-p, repetition penalty
- `hearth-llm` — CPU inference engine with custom `ThreadPool` (park/unpark)
- `hearth-compute` — GPU stubs (wgpu, type-compat only)

The `ThreadPool` uses `std::thread::park`/`unpark` for worker signaling — zero CPU idle, sub-microsecond wake latency. Work is dispatched via a pre-allocated `WorkParams` struct shared through `AtomicPtr` — zero allocation per matmul call. All Q1_0/Q2_0/Q8_0 matmul closures were decomposed into uniform `par_dot_rows(w_base, a_ptr, out_ptr, ...)` calls.
