# Hearth Performance Ops — Remaining Opportunities

Cross-referenced against BUG_TRACKER.md change history. Items marked **DONE**, **REVERTED**, or **REJECTED** removed. Only items that haven't been tried or are partially done remain.

---

## Tier 1 — High Impact, Low/Medium Effort

### 1. Wire `scratch_q8` buffer for attn_output / ffn_down / lm_head matmuls

**Status:** Buffers exist (`ForwardScratch::scratch_q8`) but never used.

`x_q8` is reused for QKV (mod.rs:377-378) and `ffn_q8` is reused for ffn_gate_up (mod.rs:582-583). But `ffn_down` (mod.rs:609, passes `x_q8: None`) and `attn_output` and `lm_head` matmuls all call `quantize_act()` which allocates a fresh `Vec<u8>` each time. `scratch_q8` was added to ForwardScratch for this purpose but never wired into those matmul calls.

Fix: pass `Some(&sc.scratch_q8[..])` to those matmuls after clearing/repopulating it.

**Estimated gain:** 3-5% (eliminates 2-3 alloc/free cycles per layer per token).

### 2. Batched prefill — use `forward_batch` as default for multi-token prompts

**Status:** Wired in (BUG_TRACKER: "Batched prefill: DONE") but `matmul_batch` still processes tokens sequentially in a for loop over `seq_len`. Each token quantizes one at a time and dispatches `par_dot_rows` per token.

Fix: batch the Q8_0 quantize across all prompt tokens (col-major SIMD quantize), or process multiple tokens in parallel threads.

**Estimated gain:** 10-30% on prompt TTFT.

---

## Tier 2 — Medium Impact, Medium Effort

### 3. NEON (ARM) kernel paths

**Status:** Not started. Scalar fallback only path for non-x86_64.

Add ARM NEON intrinsics (`int8x16_t`, `vmlal_s8`, `vpaddl_s32`) for Q1_0 and Q2_0 dot kernels. The block structure (128-element Q1 blocks × 32-element Q8 sub-blocks) maps well to 128-bit NEON lanes.

### 4. CPU flash attention

**Status:** Not started. Current attention is O(seq_len × head_dim) per head with full f32 score matrix write.

For long contexts (2048+), tiled flash attention with online softmax keeps KV cache in L1/L2 and avoids writing the full score vector to scratch.

### 5. Q1_0 AVX-512 VNNI kernel

**Status:** Not started. Q2_0 VNNI kernel exists (neutral perf). Q1_0 uses shuffle AVX2.

Q1_0 weights are {-1,+1} × scale. Using VNNI `vpdpbusd` on 512-bit would process 64 elements per instruction. The main challenge: Q1_0 weights are {-1,+1} as i8, but vpdpbusd takes u8 first operand. Encoding {-1,+1} as u8 {0,1} with correction subtracts the activation sum.

---

## Tier 3 — Lower Priority / Future

### 6. Investigate why Q2_0 pre-expansion failed while Q1_0 pre-expansion also failed

Both rejected for the same reason: memory bandwidth bound. Q1_0 expansion was 7.2× (18→130 bytes/block), Q2_0 would be 3.8× (34→128). The 1.7B Q2_0 (554 MB → 2.1 GB) could be compute-bound enough to benefit, but this is unproven.

### 7. Multi-row matmul dispatch

Process multiple weight rows in a single dot call to amortize function-call overhead and keep activation data hotter in registers. Currently each row calls `dot_fn` independently via function pointer.

---

## Summary Table

| # | Opportunity | Est. Gain | Effort | Status |
|---|------------|-----------|--------|--------|
| 1 | Wire `scratch_q8` for remaining matmuls | 3-5% | Low | Buffer exists, not wired |
| 2 | Batch prefill quantize across tokens | 10-30% (TTFT) | Medium | `forward_batch` exists, serial per-token loop |
| 3 | NEON kernel paths | N/A (ARM) | High | Not started |
| 4 | CPU flash attention | 0-5%* | High | Not started |
| 5 | Q1_0 AVX-512 VNNI | 5-15% | Medium | Not started |
| 6 | Q2_0 pre-expansion revisit | 0-15% (risky) | High | Previously failed for Q1_0 |
| 7 | Multi-row matmul dispatch | 2-5% | Medium | Not started |

*Flash attention only helps at seq_len > 2048.
