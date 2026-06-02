# Hearth Performance Review: Q1_0 / Q2_0 Bonsai Inference

## Executive Summary

The codebase has a solid kernel architecture with single-hsum FMA accumulation, a custom thread pool, and fused matmul dispatch. The opportunities below are ranked by expected impact per effort invested.

---

## Tier 1 — High Impact, Low/Medium Effort

### 1. Eliminate per-matmul activation Vec allocations (3 allocs/token)

**Location:** `crates/hearth-llm/src/model/matmul.rs:491-495` (`quantize_act`), called by: `attn_output` matmul, `ffn_down` matmul, and `lm_head` matmul — all 3 pass `x_q8: None`.

```rust
fn quantize_act(&self, x: &[f32]) -> Vec<u8> {
    let mut q8 = Vec::with_capacity(x.len().div_ceil(32) * 34);
    hearth_quant::quantize_q8_0(x, &mut q8);
    q8
}
```

This allocates a new `Vec<u8>` on every call, despite `ForwardScratch` already having `x_q8` and `ffn_q8` buffers. Fix: add a third scratch buffer (e.g., `scratch_q8`) and reuse it. For `lm_head` (vocab_size × d_model, potentially 128K × 2048 = 256MB of Q8_0), scratch reuse saves significant allocation pressure. For attn_output and ffn_down the savings are smaller (d_model-range buffers) but the allocator overhead per token adds up.

**Estimated gain:** 5-10% on token throughput (mainly from reduced allocation/latency jitter in the generate loop).

### 2. Fuse attention output matmul's quantization with the attention output

**Location:** `crates/hearth-llm/src/model/mod.rs:562-574`. The current flow is:
1. Quantize `attn_out` (d_model floats) to Q8_0 → alloc + compute
2. Run matmul

But `attn_out` is computed in `ops::attention()` just above. The activation could be quantized as a fused step during the attention value-accumulation, avoiding the separate quantization pass entirely. This is a known pattern in llama.cpp where `ggml_compute_forward_mul_mat` takes Q8_0 activations directly from the attention output buffer.

**Estimated gain:** 3-5% per token (eliminates one quantization pass per layer per token).

### 3. Avoid `.to_vec()` allocation in Q8_0 KV cache attention path

**Location:** `crates/hearth-llm/src/model/mod.rs:521-523`:
```rust
let ks = caches[layer].k_slice_dequant(kvh, seq_len).to_vec();
let vs = caches[layer].v_slice_dequant(kvh, seq_len).to_vec();
```

For Q8_0 KV caches, this allocates two `Vec<f32>` (each `seq_len * head_dim` floats) per KV head per attention call. At seq_len=2048, head_dim=128: 2 × 2048 × 128 × 4 = 2MB alloc per KV head per layer per token, then immediately freed. Fix: pass the dequant buffer slice directly to `attention()` (it already returns `&[f32]` — the `.to_vec()` just copies into a fresh allocation for no reason).

**Estimated gain:** 2-5% on high-seq-len scenarios, eliminates major GC/allocation pressure.

### 4. Coarsen ThreadPool synchronization — reduce park/unpark per token

**Location:** `crates/hearth-llm/src/pool.rs:141-153`. Each `par_dot_rows` call does:
- One `SeqCst` fence
- N worker `unpark()` syscalls (one per thread)
- N spin-wait loops

With ~7-8 matmuls per layer × 36 layers = ~250+ park/unpark cycles per token. The custom pool was built to avoid Rayon's per-task overhead, but `thread::park()/unpark()` are still kernel transitions (~1-3µs each on Windows).

**Option A (quick):** Group the 3 QKV `par_dot_rows` calls into a single dispatch where workers process Q, then K, then V row ranges without returning to sleep between them. A `par_dot_rows_multi` that accepts 3 sets of params.

**Option B (better):** Replace park/unpark with a shared atomic counter + spin-loop protocol. Workers spin-wait on a generation counter. Master sets params, increments gen, then spin-waits for completion. This eliminates all syscalls for intra-token matmul parallelism.

**Estimated gain:** 5-15% (higher on Windows where park/unpark is slower).

---

## Tier 2 — Medium Impact, Medium Effort

### 5. Use batched prefill instead of per-token forward for the prompt

**Location:** `crates/hearth-llm/src/model/mod.rs:2065-2083`. Prompt tokens are processed one at a time via `forward()`. The `forward_batch` method exists (line ~1560+) but isn't wired into `generate_text`. For a typical 50-token prompt on a 1.7B model, that's 50 sequential forward passes instead of 1 batched pass.

The `forward_batch` code already has `matmul_batch` support for Q1_0_G128 and Q2_0 dtypes (which use `dot_q1_0g128_f32` and `dot_q2_0_f32` — the f32-activation dot kernels, not the Q8_0 ones). This could be accelerated further by using Q8_0 quantization for the batched activation too.

**Estimated gain:** 10-30% on prompt processing (time-to-first-token), 0% on per-token generation.

### 6. Replace scalar RoPE with SIMD

**Location:** `crates/hearth-llm/src/ops.rs:66-135`. RoPE processes each complex pair with individual `sin_cos()` calls and scalar multiply-add. For head_dim=128, 36 layers, Qwen3 with QK-norm: that's 128/2 × (n_heads + n_kv_heads) × 36 calls per token.

**Fix:** Precompute sin/cos tables for all positions up to max_seq_len at model load time. Then apply rotation via SIMD (two `f32x8` lanes per 8 pairs). This is what llama.cpp does (`ggml_rope_ext`). The `wide` crate used elsewhere in ops.rs already has the primitives.

**Estimated gain:** 2-4% per token, plus reduced jitter.

---

## Tier 3 — Medium Impact, Higher Effort

### 7. Add Q1_0_G128 × Q8_0 kernel using `_mm256_maddubs_epi16` (no LUT)

**Location:** `crates/hearth-quant/src/q1_0g128.rs:575` (`dot_q1_0g128_q8_0_lut_avx2`). The current active kernel loads Q1V LUT entries and converts via `_mm256_cvtepi8_epi16` + `_mm256_madd_epi16`. The throughput is limited by:
- 8 LUT loads per sub-block (32 bytes from L1)
- 4 `_mm256_cvtepi8_epi16` + 4 `_mm256_madd_epi16` + 1 `_mm256_cvtepi32_ps` + 1 `_mm256_fmadd_ps` per sub-block

The reference-style kernel already exists at line 219 (`dot_q1_0g128_q8_0_ptr_avx2`) and uses `_mm256_shuffle_epi8` for sign expansion with `_mm256_maddubs_epi16` (which processes 32 i8 → 16 i16 in ONE instruction). This path has:
- 0 LUT loads (purely register/ALU)
- 1 `_mm256_maddubs_epi16` + 1 `_mm256_madd_epi16` per 32 elements
- No `_mm256_cvtepi8_epi16` needed

But the LUT path is used because the MSVC-compiled kernel (`dot_q1_0g128_fast`) reportedly beats LLVM AVX2 by 1.3-1.5×. The benchmark at `crates/hearth-quant/src/lib.rs:222-257` only tests LUT vs LLVM-AVX2 — it should also test against the shuffle-based reference kernel.

**Action:** Benchmark all 3 variants (LUT-LLVM, LUT-MSVC, shuffle-based) on the target hardware. The shuffle-based path may win on Intel since it eliminates L1 data cache traffic entirely.

**Estimated gain:** 5-15% on matmul (the dominant cost) if the shuffle kernel beats LUT.

### 8. Pre-quantize weight rows to a more dot-friendly layout

The Q1_0_G128 block format stores bit-packed bytes. Every dot product unpacks these via Q1V LUT or shuffle. An alternative: at model load time, expand each weight row to a pre-expanded sign array in a SIMD-friendly layout (e.g., 8×i8 per 128-bit lane, interleaved). This trades ~8× memory for weight storage (from 18 bytes/block to 128 bytes/block) but eliminates all bit-unpacking from the hot loop. For a 1.7B model at 128-elem blocks: weight storage goes from ~37MB to ~265MB. For the 8B model: from ~175MB to ~1.25GB. This is viable for the 1.7B and 4B models on systems with 16GB+ RAM.

**Estimated gain:** 15-25% on matmul throughput for smaller models.

---

## Tier 4 — Lower Priority / Future

### 9. NEON (ARM) kernel paths

The scalar fallback in `crates/hearth-quant/src/q1_0g128.rs:628-665` is the only path for non-x86_64. Adding NEON intrinsics (ARMv8 `int8x16_t` + `vmlal_s8` + `vpaddl_s32`) would enable Apple Silicon and ARM server deployment. The block structure (128-element Q1 blocks × 32-element Q8 sub-blocks) maps well to 128-bit NEON lanes.

### 10. AVX-512 or VNNI kernels

For Ice Lake+ or Zen 4+, `VPDPBUSD` (VNNI) can process 4× i8×u8→i32 in one instruction. Q1_0 weights are i8 {-1,+1}, Q8_0 activations are i8. Using VNNI would give 4× the throughput of the current `_mm256_madd_epi16` approach per instruction.

### 11. CPU flash attention

The current attention is O(seq_len × head_dim) per head. For long contexts (2048+), tiled flash attention with online softmax would keep KV in cache and avoid the full `seq_len × head_dim` write to scratch.

### 12. Persistent matmul weights in shared L3 cache

Currently, weight rows are read from `Vec<u8>` through the custom pool. On repeated tokens, the same weight rows are re-read from DRAM. With enough LLC, streaming matmul patterns can be optimized. This is already partially addressed by the weight data being in heap memory (not mmap), but software prefetch (`_mm_prefetch`) for the next row/block could hide DRAM latency for large models.

---

## Summary Table

| # | Opportunity | Est. Gain | Effort | Hotspot |
|---|------------|-----------|--------|---------|
| 1 | Reuse scratch Q8 buffers — eliminate 3 allocs/token | 5-10% | Low | `matmul.rs:491` |
| 2 | Fuse attn_out quantization into attention kernel | 3-5% | Medium | `ops.rs:215`, `mod.rs:562` |
| 3 | Remove `.to_vec()` in Q8_0 KV cache attention | 2-5% | Low | `mod.rs:521-523` |
| 4 | Coarsen ThreadPool sync — reduce park/unpark | 5-15% | Medium | `pool.rs:141` |
| 5 | Wire up batched prefill for prompt processing | 10-30% (TTFT) | Low | `mod.rs:2065` |
| 6 | SIMD RoPE with precomputed sin/cos | 2-4% | Medium | `ops.rs:66` |
| 7 | Benchmark/switch to shuffle-based AVX2 kernel | 5-15% | Medium | `q1_0g128.rs:219` |
| 8 | Pre-expand weight rows to sign arrays | 15-25% | High | Model load |
| 9-12 | NEON, VNNI, flash attention, prefetch | 10-40% cumulative | High | Various |

Items 1-6 combined would conservatively yield a **25-50% improvement in tokens/second** for both Q1_0 and Q2_0 Bonsai models, with most of the work concentrated in a few files (`matmul.rs`, `pool.rs`, `mod.rs`).
