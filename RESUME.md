# Session 6 Resume — SIMD RoPE with precomputed sin/cos

## Done

| # | Opportunity | Gain | Status |
|---|---|---|---|
| 1 | Scratch Q8 buffer reuse | 5-10% (neutral) | Sess 3 — `ForwardScratch` reuse |
| 3 | KV cache `.to_vec()` removal | 2-5% (+29% on 4B) | Sess 3 — `k_slice_dequant_into` |
| 4 | Gen counter spin-wait (was Option B) | 5-15% (+33%) | Sess 3 — replaced park/unpark |
| 5 | Batched prefill for prompt processing | 10-22% TTFT | Sess 5 — DONE |
| 6 | **SIMD RoPE with precomputed sin/cos** | **2-4% (rope: ~960µs→~10µs)** | **Sess 6 — DONE** |
| 7 | Shuffle-based Q1_0 AVX2 kernel | 5-15% (+25%) | Sess 3 — `dot_q1_0g128_q8_0_ptr_avx2` |

## Session 6 — SIMD RoPE

**What changed:**
- `ops.rs`: Re-laid-out `RopeCache` table from interleaved `[sin, cos, sin, cos, ...]` to contiguous `[sin0..sin63, cos0..cos63]` per position, enabling SIMD loads
- `ops.rs`: `RopeCache::apply` now uses `f32x8` to process 8 complex pairs per iteration (was scalar loop with per-element sin/cos calc)
- `mod.rs`: Wired `RopeCache` into `LlamaModel` — initialized at load time, replaced all 6 `ops::rope()` calls (3 forward paths: `forward()`, `forward_batch()`, `encode_text()`)
- Removed unused `scaling_type`/`scaling_factor`/`orig_ctx` locals where they were only used by rope

**Results (4B Q1_0, single-token decode):**
- `rope` timing: 5-13µs per forward pass (was ~960µs when sin_cos computed per element)
- Net forward time reduction: ~950µs → ~47ms → net ~2% improvement
- Removed 192 sin_cos calls per layer (36 layers × 2 heads × 80 dim) = 6,912 trig calls/token

**Key insight:** `RopeCache` struct was already defined but never wired into the model. The `ops::rope()` standalone function recomputed `theta.powf(2*i/dim)` and `sin_cos()` on every call. Precomputation at load time eliminated all trig from the hot path.

## Not tried — ranked by estimated impact

### 1. Fuse attn_out quant into attention kernel [Tier 1, #2]
**Est:** 3-5%
**Files:** `ops.rs:301`, `mod.rs:560-574`
**What:** `attn_out` is computed in `ops::attention()` then separately quantized to Q8_0. Fuse quant into the attention value-accumulation step — avoid the separate pass.

### 2. Pre-expand weight rows to sign arrays [Tier 3, #8]
**Est:** 15-25% on small models
**What:** At load time, expand Q1_0 weight rows from bit-packed 18B/128el to raw i8 sign arrays (128B/block). 8× memory trade (e.g., 1.7B Q1_0: 37MB→265MB). Eliminates all bit-unpacking from hot loop.

### 3. Coarsen ThreadPool: group QKV into single dispatch [Tier 1, #4 Option A]
**Est:** 1-3%
**What:** Currently Q, K, V matmuls are 3 separate `par_dot_rows()` calls. Each gen-counter increment + spin-wait adds ~1-3µs overhead.

### 4. Remaining Tier 4 items
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
