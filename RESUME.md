# Session 5 Resume — Results from performance_ops.md

Cross-referenced 12 opportunities against `BUG_TRACKER.md`. 5 done, 7 not tried.

## Done (from performance_ops.md)

| # | Opportunity | Gain | Status |
|---|---|---|---|
| 1 | Scratch Q8 buffer reuse | 5-10% (neutral) | Sess 3 — `ForwardScratch` reuse |
| 3 | KV cache `.to_vec()` removal | 2-5% (+29% on 4B) | Sess 3 — `k_slice_dequant_into` |
| 4 | Gen counter spin-wait (was Option B) | 5-15% (+33%) | Sess 3 — replaced park/unpark |
| 5 | **Batched prefill for prompt processing** | **10-22% TTFT** | **Sess 5 — DONE** |
| 7 | Shuffle-based Q1_0 AVX2 kernel | 5-15% (+25%) | Sess 3 — `dot_q1_0g128_q8_0_ptr_avx2` |

## Batched prefill results (Session 5)

**Achievements:**
- Wired `forward_batch()` into `generate_text()` for CPU multi-token prompts
- Replaced f32-activation rayon matmul with Q8_0 quantized activations + custom ThreadPool
- Pre-allocated KV cache (`KVCache::new` now allocates full-size `k`/`v` vectors at construction, eliminating per-layer 256MB resizes in `write_kv`)
- Sequential quantize (removed rayon contention with pool spinners)

**4B Q1_0 benchmarks (7-token prefill → 50 decode tokens):**
| Config | Prefill (7t) | Total (57t) | Tok/s |
|---|---|---|---|
| Baseline (single-token, pre-alloc KV) | ~322ms (7×46ms) | ~3013ms | 18.9 |
| Batched (Q8_0 matmul, pre-alloc KV) | 264-420ms (38-60ms/tok) | 3213-3589ms | 15.9-17.7 |
| Batched (NO pre-alloc KV) | 1305-1750ms | 5191-6418ms | 6.4-11.0 |

**Key insight:** KV cache resize was the dominant bottleneck — each `write_kv` first call did `resize(total, 0.0f32)` for 128MB K + 128MB V (256MB zeroing per layer = 9.2GB total). Pre-allocating at construction time eliminated this, reducing per-layer time from 33→103ms (growing) to stable 5-7ms.

## Not tried — ranked by estimated impact

### 1. Fuse attn_out quant into attention kernel [Tier 1, #2]
**Est:** 3-5%
**Files:** `ops.rs:215`, `mod.rs:562-574`
**What:** `attn_out` is computed in `ops::attention()` then separately quantized to Q8_0. Fuse quant into the attention value-accumulation step — avoid the separate pass.
**Caveat:** Q2_0 models use Q2_0×Q8_0 matmul for attn_output, so the quant is Q8_0. Q1_0 models use Q1_0×Q8_0. Both need the activation in Q8_0 format. The fusion would write Q8_0 blocks directly during attention output accumulation.

### 2. SIMD RoPE with precomputed sin/cos [Tier 2, #6]
**Est:** 2-4%
**File:** `ops.rs:66-135`
**What:** Current RoPE uses per-element `sin_cos()` + scalar multiply-add. Precompute sin/cos tables at load time for all positions up to `max_seq_len`. Apply rotation via `f32x8` (one `wide` load per 8 complex pairs). Same pattern as `ggml_rope_ext`. head_dim=128, 36 layers.

### 3. Pre-expand weight rows to sign arrays [Tier 3, #8]
**Est:** 15-25% on small models
**What:** At load time, expand Q1_0 weight rows from bit-packed 18B/128el to raw i8 sign arrays (128B/block). 8× memory trade (e.g., 1.7B Q1_0: 37MB→265MB). Eliminates all bit-unpacking from hot loop.

### 4. Coarsen ThreadPool: group QKV into single dispatch [Tier 1, #4 Option A]
**Est:** 1-3%
**What:** Currently Q, K, V matmuls are 3 separate `par_dot_rows()` calls. Each gen-counter increment + spin-wait adds ~1-3µs overhead.

### 5. Remaining Tier 4 items
- NEON kernel paths for ARM
- AVX-512 VNNI (`VPDPBUSD`) kernel
- CPU flash attention (tiled softmax)
- Software prefetch for weight rows

## Build & verify
```powershell
cargo build --release
cargo fmt
cargo clippy -p hearth-quant -- -D warnings
cargo test -p hearth-quant -- --test-threads=1
```
