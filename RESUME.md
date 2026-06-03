# Session 16: Batched prefill dispatch (par_dot_rows_batched) — NEUTRAL

**Summary:** Added `par_dot_rows_batched` to ThreadPool, replacing the sequential per-token `par_dot_rows` loop in `matmul_batch` with a single dispatch processing all seq_len tokens. No decode regression on any model. All 6 models within ±system variance of S15 baseline.

---

## Next optimization targets (re-evaluated after Session 16)

### Target 1: Pre-quantize activation once for Q/K/V matmul_batch in forward_batch
Each `matmul_batch` call quantizes the full `residual[seq_len × d]` independently. For Q/K/V matmul_batch (3 calls per layer), the same residual is quantized 3 times. Add `matmul_batch_with_q8` or modify `matmul_batch` to accept a pre-quantized Q8 buffer, then in `forward_batch` quantize once and share.

**Potential:** Saves ~2µs/token/layer × seq_len × (n_layers × 3 - n_layers). For 10-token prefill on 28 layers: ~560µs. Small absolute gain but clean optimization.

### Target 2: Raw AVX2 intrinsics for attention
`attention()` and `attention_batch()` use `wide::f32x8` (~5-7% of total forward time). Hand-tuned AVX2 could be 2-3× faster on the hot inner loop (score dot product). Currently ~1ms per decode token for 1.7B Q1_0 at seq_len=50, growing with seq_len.

### Target 3: Prefetch with `_MM_HINT_T0` (L1 hint) for large models
Previously tried T1 (L2 prefetch) caused ~5-11% regression on Q1_0. Try L1 prefetch (`_MM_HINT_T0`) only for d>=2560 models where DRAM bandwidth is the bottleneck.

---

## Key files
- `crates/hearth-llm/src/pool.rs` — `par_dot_rows_batched` method + worker loop with seq_len outer iteration
- `crates/hearth-llm/src/model/matmul.rs` — `matmul_batch` now uses single `par_dot_rows_batched` call instead of per-token loop
- `crates/hearth-quant/src/q1_0g128.rs` — Q1_0 dot kernel (shuffle AVX2)
- `crates/hearth-quant/src/q2_0.rs` — Q2_0 dot kernel
- `crates/hearth-llm/src/ops.rs` — attention, rms_norm, silu via `wide::f32x8`
