# Perf: Bonsai 1.7B (Q1_0) + Ternary-Bonsai 1.7B (Q2_0)
Target: **33 tok/s (30ms/token)**.

> **⚠️ BOTH MODELS ARE CRITICAL.** Optimize Q1_0 AND Q2_0. Every change must be benchmarked against both.

## Current (2026-06-02, final)

| Model | Hearth | Reference | Delta | Forward pass |
|-------|--------|-----------|-------|-------------|
| Q1_0  | **34.4** | 32.0    | **+7.5%** | 29.1ms |
| Q2_0  | **27.2** | 5.1     | **+433%** | 36.8ms |

**Hearth now BEATS the reference on Q1_0!** 18.7→34.4 tok/s (+84% from start). Custom thread pool replaced all Rayon dispatch.
**Q2_0**: 17.4→27.2 tok/s (+56%). Still bottlenecked by larger Q2_0 block size (34 vs 18 bytes per 128 elements).

1-thread: Hearth 5.8 tok/s vs Ref 7.3 tok/s → 1.26× kernel codegen gap. But custom pool parallel scaling (5.93×) exceeds reference OpenMP (4.38×).

## Q2_0 kernel: AVX2 REWRITTEN (done 2026-06-02)

Q2_0 kernel (`hearth-quant/src/q2_0.rs`) rewritten from portable `wide::i32x8` to raw AVX2 intrinsics matching the Q1_0 LUT kernel pattern. Processes 16 elements per batch (was 8) using:
1. `_mm_loadu_si128` + `_mm256_cvtepi8_epi16` for activation loading
2. Q2V LUT entries packed via stack array + `_mm_loadu_si128` for weight loading
3. `_mm256_madd_epi16` for dot product (was `i32x8` portable mul-add)
4. `_mm256_cvtepi32_ps` + `_mm256_fmadd_ps` for FMA accumulation across sub-blocks
5. `hsum_float_8` once per Q2_0 block
SSE4.1 and scalar fallbacks retained.

Result: 11.2 → 17.4 tok/s (+55%). Forward pass: 66ms → 47ms (-29%).

## Q1_0 remaining gap (lower priority)
1.27× kernel gap at 1-thread (LLVM codegen vs MSVC). ~17% parallel scaling gap (Rayon 3.4× vs OpenMP 4.0×). Diminishing returns on further work.

## What worked (chronological)

### Session: SIMD inner kernels
- Q2_0: Q2V[256] LUT + `wide::i32x8` SIMD. 4.0→7.5 tok/s (+87%)
- Q1_0: AVX2 intrinsics (`vpmovsxbw`+`vpmaddwd`) + SSE4.1 fallback. 5.5→10.3 tok/s (+87%)

### Session: Pointer-chasing outer loop
- Raw-pointer dot kernels + pointer arithmetic in matmul outer loop. +5-8%

### Session: format! elimination + fused matmul
- Pre-computed tensor names at load time → eliminated 336 `format!()` per forward. +10%
- Fused gate+up matmul: single par_iter over 2×ffn_dim rows. ~0% gain

### 🔴 Session: Q1_0 NaN fix
Q1_0 (type 41) has 128-el blocks in Prism fork, was dispatched to 32-el kernel → NaN logits. Fixed by routing to 128-el kernel. +27%

### Session: LUT AVX2 kernel
Replaced reference-style shuffle+bitmask with Q1V[256][8] LUT kernel (+32%). Current default.

### Session: Static chunking
OpenMP-style `par_for_static` via `rayon::scope`. +8% for Q1_0.

### 🔴 Session 2026-06-01/02: target-cpu=native (game changer)
Added `target-cpu=native` to `.cargo/config.toml`. LLVM optimizes for Zen 4. **2× single-threaded, +38% multi-threaded.**

### Session 2026-06-01/02: par_iter + with_min_len
Replaced `rayon::scope` (fork-join per matmul, ~1800 barriers/forward) with `into_par_iter().for_each()`. **+30% overall.**

### Session 2026-06-01/02: Architecture fixes
- QK head norm: 560 `.to_vec()` allocations → pre-allocated buffer
- lm_head name: cached at load time

### 🔴 Session 2026-06-02: Q2_0 AVX2 kernel rewrite
Replaced portable `wide::i32x8` SIMD with raw AVX2 intrinsics. 16-element batches (was 8). **Q2_0: 11.2 → 17.4 tok/s (+55%).** Forward pass: 66ms → 47ms (-29%).

### 🔴 Session 2026-06-02: Fixed Q1_0 infinite recursion bug
`dot_q1_0g128_fast` fallback path (non-msvc-kernel) called itself recursively instead of `hearth_quant::dot_q1_0g128_q8_0_ptr`. Fixed — Q1_0 restored to 18.7 tok/s.

### 🔴 Session 2026-06-02: `rayon::broadcast` replaces `par_iter` (+5-6%)
`par_for_static` now uses `rayon::broadcast()` instead of `into_par_iter().with_min_len().for_each()`. Each thread computes its own chunk — no work-item queue pushes, just a single barrier per call. Q1_0: 18.7→19.8 tok/s. Q2_0: 17.4→18.3 tok/s. File: `crates/hearth-llm/src/parallel.rs`.

### 🔴🔴 Session 2026-06-02: Custom thread pool (park/unpark) — GAME CHANGER (+55%/+49%)
Replaced all Rayon dispatch with custom `ThreadPool` using `thread::park`/`unpark` for worker signaling. Workers sleep at 0% CPU between matmuls. Main writes `WorkParams` to shared memory, signals workers, spin-waits on done flags. Zero allocation per dispatch. Q1_0: 19.8→34.4 tok/s (+74%). Q2_0: 18.3→27.2 tok/s (+49%). Key files: `crates/hearth-llm/src/pool.rs` (new), `crates/hearth-llm/src/model/matmul.rs` (converted paths).

### Session 2026-06-02: Raw threads + LM head F32 + quant fusion + LLVM flags
- Raw `std::thread::scope`: catastrophic on Windows (5.2/2.8 tok/s). Fixed by using persistent threads with park/unpark.
- Spin-wait pool: consumed 100% CPU, starved main thread. Fixed by using park/unpark.
- LM head F32: both models have Q1_0/Q2_0 lm_head dtype, not F32.
- Q8_0 quant fusion: negligible (<0.5% of forward pass).
- LLVM codegen flags: `+-slow-unaligned-mem-256` not recognized by Rust LLVM.
- Kernel hsum optimization: accumulated f32 across blocks via FMA, single hsum per row. Marginal.

## What didn't work

### 🔴 MSVC FFI kernel (2026-06-02)
Compiled reference's scalar C kernel with MSVC `/O2 /arch:SSE2` as FFI. In release mode, this kernel is **4-6× slower** than LLVM + `target-cpu=native`. MSVC generates generic 128-bit code; LLVM with `target-cpu=native` generates Zen 4-optimized AVX2. The reference's 28 tok/s advantage is NOT from the inner kernel alone — it's the entire MSVC-compiled pipeline.

### Don't retry (cumulative)
- Prism-style conditional add/sub: 5× slower than LUT (branchy codegen)
- `wide::i32x8::new([...])`: construction overhead > SIMD benefit
- LM head chunking via `par_chunks_mut`: slower
- SSE4.1 path: 4.5× slower than AVX2
- QKV fusion: <3% gain
- Reference-style AVX2 kernel rewrite: ~0% gain
- Static chunking via `rayon::scope`: +8% but obsolete (par_iter is better)
- MSVC FFI kernel: 4-6× slower than LLVM with target-cpu=native

## Remaining gap: 1.50× (18.7 vs 28.0 tok/s)

### ~25-30% at 1-thread (LLVM codegen)
`target-cpu=native` closed 60%. Remaining is irreducible LLVM vs MSVC codegen difference. MSVC FFI doesn't help (kernel alone is slower).

### ~17% parallel scaling gap
Rayon scales 3.4× vs OpenMP's 4.0×. Rayon's global pool has inherent overhead vs OpenMP's static scheduling.

## To try (next sessions)

### 1. Parallel scaling: raw threads + work-stealing
Rayon global pool overhead is ~17%. Try replacing `par_for_static` with raw `std::thread::scope` spawning N threads, each with a fixed chunk of rows. Avoids Rayon's job-stealing scheduler entirely. Matmul rows are perfectly balanced (same size), so static work distribution is ideal — no work-stealing needed. Could recover 5-10%.

### 2. LM head: F32 weight path
lm_head is 10-12% of total time. Currently quantized via Q8_0. The output.weight tensor might be F32 in the GGUF — switching to a direct F32 matmul path could be faster for the final projection since there's only one matmul (not per-layer).

### 3. Q8_0 activation quantization: fuse into matmul
Activation quantization is ~0.5% per matmul. Fuse the quant loop into the first row of the matmul to save a pass over the input vector. Marginal (~5%) but clean.

### 4. LLVM codegen experiments
Try `-C llvm-args=-enable-unsafe-fp-math` or `-C target-feature=+avx2,+fma,+-slow-unaligned-mem-256` to see if LLVM can generate better Zen 4 code. Also try `codegen-units=1` for more aggressive inlining.

## Per-token forward pass

### Q1_0 (~42ms)
| Section | μs/token | % total |
|---|---|---|
| ffn_gate_up_matmul | ~14,000 | ~33% |
| ffn_down_matmul | ~9,500 | ~23% |
| qkv_matmul | ~6,500 | ~15% |
| lm_head_matmul | ~5,000 | ~12% |
| attn_output_matmul | ~4,500 | ~11% |
| attention | ~1,500 | ~4% |
| rope | ~1,000 | ~2% |
| Other | ~1,000 | ~2% |
| **TOTAL** | **~42,000** | **100%** |

### Q2_0 (~47ms)
| Section | μs/token | % total |
|---|---|---|
| ffn_gate_up_matmul | ~15,000 | ~34% |
| qkv_matmul | ~7,000 | ~16% |
| ffn_down_matmul | ~8,000 | ~18% |
| lm_head_matmul | ~4,500 | ~10% |
| attn_output_matmul | ~4,000 | ~9% |
| attention | ~2,000 | ~5% |
| rope | ~1,000 | ~2% |
| Other | ~2,500 | ~6% |
| **TOTAL** | **~47,000** | **100%** |

Q2_0 forward pass reduced from 66ms to 47ms via AVX2 kernel rewrite. Now within ~12% of Q1_0 (42ms).
