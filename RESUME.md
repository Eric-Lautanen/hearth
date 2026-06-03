# Session 18: Eliminate .to_vec() allocations in batch/encode paths — NEUTRAL on decode

**Summary:** Eliminated per-head/per-iteration `Vec` allocations in `forward_batch` and `encode_text` QK head norm loops by adding a reusable `head_norm_tmp` scratch buffer to `BatchScratch`. Also eliminated `.to_vec()` in `encode_text` output norm by reusing `norm_tmp`. Hoisted `inv_sqrt_hd` in attention functions (replaces per-position division with multiplication). Decode `forward()` path completely unaffected by all changes — no decode regression.

---

## Remaining opportunities

### 1. Matmul inner kernels are saturated
Q1_0 shuffle AVX2 and Q2_0 LUT+SIMD kernels are well-optimized for this system. Further gains require different kernel approaches (e.g., VNNI on future CPUs, or format changes).

### 2. Attention function (~5-7% of decode time)
`f32x8` SIMD already efficient. Raw AVX2 intrinsics would save only `reduce_add()` overhead (~0.3% total). Not worth the code complexity.

### 3. Fused QKV dispatch for forward_batch
Currently 3 separate `par_dot_rows_batched` calls (Q/K/V) per layer. Fusing into one dispatch saves 2 gen-counter handshakes (~15μs total for 28 layers). Negligible gain.

### 4. KV cache format (F32 → Q8_0)
All models use F32 KV cache. Switching to Q8_0 halves cache memory bandwidth but requires per-position dequant in attention. Analysis shows this is roughly break-even at seq_len=50 and net-negative at short contexts due to dequant overhead vs bandwidth savings at this scale. May benefit at very long contexts (seq_len > 4096).

---

## Key files
- `crates/hearth-llm/src/model/scratch.rs` — `BatchScratch` has new `head_norm_tmp: Vec<f32>` field
- `crates/hearth-llm/src/model/mod.rs` — `forward_batch`/`encode_text` use `head_norm_tmp` instead of `.to_vec()` allocations; `ensure_batch_size` takes `head_dim` parameter
- `crates/hearth-llm/src/ops.rs` — `attention`/`attention_batch` use `inv_sqrt_hd = 1.0 / sqrt(head_dim)` hoisted out of score loop
