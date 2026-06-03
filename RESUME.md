# Session 17: Pre-quantize activation everywhere — NEUTRAL

**Summary:** Extended the pre-quantized activation optimization across all batch matmul paths. `forward_batch` and `encode_text` now share a single `batch_q8` buffer across Q/K/V (was 3 separate quantize loops), ffn_gate/ffn_up (was 2), attn_output, and ffn_down. No decode regression on any model. All 6 models within ±system variance of S16 baseline.

---

## Remaining opportunities

### 1. Matmul inner kernels are saturated
Q1_0 shuffle AVX2 and Q2_0 LUT+SIMD kernels are well-optimized for this system. Further gains require different kernel approaches (e.g., VNNI on future CPUs, or format changes).

### 2. Attention function (~5-7% of decode time)
`f32x8` SIMD already efficient. Raw AVX2 intrinsics would save only `reduce_add()` overhead (~0.3% total). Not worth the code complexity.

### 3. Fused QKV dispatch for forward_batch
Currently 3 separate `par_dot_rows_batched` calls (Q/K/V) per layer. Fusing into one dispatch saves 2 gen-counter handshakes (~15μs total for 28 layers). Negligible gain.

### 4. KV cache format (F32 → Q8_0)
All models use F32 KV cache. Switching to Q8_0 halves cache memory bandwidth but requires per-position dequant in attention. Could benefit large models (8B) where cache bandwidth pressure is higher.

---

## Key files
- `crates/hearth-llm/src/model/matmul.rs` — `matmul_batch` now takes optional `x_q8: Option<&[u8]>`
- `crates/hearth-llm/src/model/mod.rs` — `forward_batch` and `encode_text` pre-quantize once per activation group
- `crates/hearth-llm/src/model/scratch.rs` — `BatchScratch` has `batch_q8: Vec<u8>` field
