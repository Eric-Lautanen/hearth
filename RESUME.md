# Session 15: Q1_0 AVX-512 VNNI kernel (correct but ~33% regression — REVERTED)

**Key finding:** `vpdpbusd(~sm & 2, act) - sum_act` produces wrong answers for Q1_0 VNNI. The correct formula is `dpbusd(1, xor(act,sm)-sm) = Σ sign_corrected_act`. The shuffle kernel's SIMD-only pipeline is fundamentally better on Zen 4. VNNI adds LUT stores + 512-bit double-pumping overhead.

---

## Next optimization targets (re-evaluated after Session 15)

### Target 1: Batched prefill quantize (performance_ops.md Tier 1 #2)
Current `matmul_batch` quantizes each token sequentially in a for loop. Batch quantize across all prompt tokens for 10-30% TTFT improvement. Doesn't affect decode tok/s benchmarks but improves first-token latency for multi-token prompts.

### Target 2: Raw AVX2 intrinsics for attention
`attention()` and `attention_batch()` use `wide::f32x8` (~5-7% of total forward time). Hand-tuned AVX2 could be 2-3× faster on the hot inner loop (score dot product). Saves ~3-5% total decode time.

### Target 3: Prefetch with `_MM_HINT_T0` (L1 hint) for large models
Previously tried T1 (L2 prefetch) caused ~5-11% regression on Q1_0. Try L1 prefetch (`_MM_HINT_T0`) only for d>=2560 models where DRAM bandwidth is the bottleneck.

---

## Key files
- `crates/hearth-quant/src/q1_0g128.rs` — Q1_0 dot kernel (shuffle AVX2, VNNI kernels kept as dead_code)
- `crates/hearth-quant/src/q2_0.rs` — Q2_0 dot kernel (256-bit VNNI, 512-bit VNNI, AVX2 LUT, SSE4.1, scalar)
- `crates/hearth-llm/src/model/matmul.rs` — matmul dispatch for Q1_0/Q2_0
- `crates/hearth-llm/src/pool.rs` — gen-counter thread pool (par_dot_rows dispatch)
- `crates/hearth-llm/src/ops.rs` — attention, rms_norm, silu via `wide::f32x8`

## Key ref files (Prism fork)
- `ggml/src/ggml-cpu/quants.c:177` — `ggml_vec_dot_q1_0_q8_0_generic` (scalar, no SIMD)
- `ggml/src/ggml-common.h:187-192` — `block_q1_0` struct, `block_q2_0` struct
