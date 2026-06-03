# Session 14: AVX-512 512-bit Q2_0 VNNI kernel (neutral)

**Session 14 results:**
1. **512-bit Q2_0 VNNI kernel:** Added `dot_q2_0_q8_0_vnni_avx512_2sub` using `_mm512_dpbusd_epi32` to process 64 elements (2 Q8_0 sub-blocks) per iteration. Uses `_mm512_extracti64x4_epi64` to split 512-bit result into two 256-bit halves with independent Q8_0 scale factors. Requires `avx512f` + `avx512vnni` + `avx512dq`.
   - **Microbenchmark:** ~8.5-9.7% faster in debug mode, ±0.7% neutral in release mode
   - **End-to-end:** All 6 models within system variance of Session 13. No regression.
   - **Analysis:** Zen 4 double-pumps 512-bit→256-bit; the 512-bit path has identical µop throughput. The ~9% debug advantage comes from reduced loop overhead that LLVM already eliminates in release via unrolling/scheduling.
   - **Kept:** Neutral, correct, future-proof for CPUs with native 512-bit units.

2. **Added `bench_vnni_256_vs_512` regression test** that directly compares per-iteration results of both VNNI kernels for correctness and reports timing. Runs in both `debug` and `release` test profiles.

---

## Next optimization targets (re-evaluated after Session 14)

### Target 1: Q1_0 AVX-512 VNNI kernel
Map Q1_0 {-1,+1} to u8 {0,1} via `(w+1)/2`, then `true_dot = 2*vpdpbusd(w_u8,act) - vpdpbusd(ones,act)`. The shuffle kernel already very efficient, but VNNI could save ~1-2 instructions per 32-element block. Worth microbenchmarking.

### Target 2: Batched prefill quantize (performance_ops.md Tier 1)
Current `matmul_batch` quantizes each token sequentially. Batch quantize across all prompt tokens for 10-30% TTFT improvement. Doesn't affect decode benchmarks.

### Target 3: Raw AVX2 intrinsics for attention
`attention()` and `attention_batch()` use `wide::f32x8` for dot products (5-7% total forward time). Hand-tuned AVX2 could be 2-3× faster on the hot inner loop.

### Target 4: Prefetch with `_MM_HINT_T0` (L1 hint) for large models
Previously tried T1 (L2 prefetch) caused ~5-11% regression on Q1_0. Try L1 prefetch (`_MM_HINT_T0`) only for d>=2560 models where DRAM bandwidth is the bottleneck.

---

## Key files
- `crates/hearth-quant/src/q2_0.rs` — Q2_0 dot kernel (256-bit VNNI, 512-bit VNNI, AVX2 LUT, SSE4.1, scalar)
- `crates/hearth-llm/src/model/matmul.rs` — matmul dispatch for Q1_0/Q2_0
- `crates/hearth-quant/src/q1_0g128.rs` — Q1_0 dot kernel (shuffle AVX2)
- `crates/hearth-llm/src/pool.rs` — gen-counter thread pool (par_dot_rows dispatch)
- `crates/hearth-llm/src/ops.rs` — attention, rms_norm, silu via `wide::f32x8`

## Key ref files (Prism fork)
- `ggml/src/ggml-cpu/quants.c:177` — `ggml_vec_dot_q2_0_q8_0_generic` (purely scalar, no SIMD path for Q2_0)
- `ggml/src/ggml-common.h:187-192` — `block_q2_0` struct (128 elements, 34 bytes)
