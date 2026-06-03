# Hearth Performance Ops — Prefill Optimization Opportunities

> **⚠️ CRITICAL RULE: NEVER modify the decode generation path.** All optimizations in this document target ONLY `forward_batch()` and `encode_text()` (prefill). The single-token decode `forward()` function must remain completely untouched. Decode is at peak performance — Q1_0 shuffle AVX2 and Q2_0 LUT+SIMD kernels saturated on Zen 4 8C/16T. Any change to `forward()`, `matmul()`, `par_dot_rows()`, `attention()`, or `ops.rs` functions used by decode risks regressing the 1.7B-8B models by 10-33%.

**Scope:** **Prefill-only** (prompt processing / TTFT). Batch functions: `forward_batch()`, `encode_text()`, `matmul_batch()`, `par_dot_rows_batched()`, `attention_batch()`, `norm_batch()`.

**System:** AMD Ryzen 7 8840HS (8C/16T Zen 4), AVX-512 VNNI (double-pumped 256-bit), AVX2, FMA. DRAM bandwidth constrained (single-channel DDR5 5600 in some configurations, 2-channel in others).

**Models:** Qwen3 Bonsai family (1.7B/4B/8B), Q1_0 and Q2_0 quant formats, head_dim=128, d_model=2048/2560/4096, 28-36 layers.

---

## Prefill bottleneck analysis

Current prefill timing breakdown (1.7B Q1_0, 10-token prefill, ~158ms total):

| Section | % of prefill | Notes |
|---|---|---|
| Matmul (QKV + attn_out + gate/up + down + lm_head) | ~70-80% | Same matmul kernels as decode, but run seq_len times |
| Attention (softmax + weighted sum) | ~5-15% | O(seq_len² × head_dim) — grows with prompt length |
| Quantize activations (4× per layer) | ~4-5% | Serial per-token loop |
| Head norm (Q/K) | ~1-2% | Serial over seq_len × heads |
| RoPE | <0.1% | Already SIMD optimized |
| Other (copy, add, SiLU, etc.) | ~4-6% | |

**Key insight:** Unlike decode (bandwidth-bound), prefill is **compute-bound** due to:
1. Matmuls process `seq_len` tokens against weight matrices — weight data is reused across tokens, shifting bottleneck from DRAM bandwidth to compute
2. Attention is O(seq_len² × head_dim) — becomes dominant at long context
3. All non-matmul ops scale with seq_len

---

## Tier 1 — High Impact, Medium Effort

### 1. Q8_0 KV Cache (replace F32)

**Current state:** All models use F32 KV cache. `KVCache` stores k: Vec<f32>[n_kv_heads × max_seq × head_dim] and v similarly. For 1.7B: 16 heads × 8 KV heads × 8192 × 128 × 4B = 512 MB per cache (1 GB total for K+V).

**Change:** Store KV cache as Q8_0 blocks (f16 scale + 32 i8 values = 34 bytes per 32 elements = 1.0625× compression ratio vs 4× for F32→Q8_0, actually Q8_0 is 34/32 = 1.0625 bytes/element vs 4 bytes/element for f32 = **3.76× compression**).

**Benefit:**
- KV cache memory bandwidth reduced by 3.76× during attention
- At seq_len=2048, 1.7B model KV cache = 16 MB F32 → 4.3 MB Q8_0 — fits in L2
- turboquant_plus data (Apple M-series): Q8_0 KV cache gives 7.4% prefill improvement at long context vs F32
- At seq_len=8192: 64 MB → 17 MB — fits in LLC

**Cost:**
- Per-position dequantization in attention (f16 scale + 32 × i8 → f32)
- Dequant overhead is ~2-4 f32 operations per element (multiply by scale)
- At short context (seq_len < 128), dequant overhead may outweigh bandwidth savings
- Need to modify: `KVCache` struct, `write_kv`, `k_slice_dequant` (exists?), attention read paths

**Estimated gain:** 5-15% on prefill at seq_len ≥ 512. Break-even at seq_len ≈ 128. Net-negative at seq_len < 64.

### 2. CPU Tiled Attention (Flash Attention for CPU)

**Current state:** `attention_batch()` computes full score vector `scratch[0..attended_len]` then softmax then weighted sum. This writes O(seq_len) scores per head and reads the entire attended KV span per head.

**Change:** Tile the KV cache dimension — process attention in fixed-size tiles (e.g., 32 or 64 positions) using online softmax:
1. Load Q tile (head_dim)
2. For each KV tile (tile_size × head_dim):
   a. Compute scores for this tile: Q·K_tile^T
   b. Apply online softmax: update running max, running sum, rescale prior accumulation
   c. Accumulate weighted V: out = Σ softmax(score) × V_tile

**Benefit:**
- KV cache read: each element read once (vs current approach of reading KV into score computation then reading again for weighted sum → 2× reads)
- Score vector stays in registers (no scratch buffer write)
- Better cache utilization for KV data (tile fits in L1)
- Eliminates `scratch[seq_len]` allocation

**Cost:**
- Online softmax requires per-tile rescaling (exp correction factor) — ~10% extra arithmetic
- Works best with tile_size that keeps K_tile + V_tile in L1 (tile_size × head_dim × 2 × 4B for F32, less for Q8_0)
- For head_dim=128: tile_size=32 → 32×128×2×4 = 32KB — fits in L1 (32KB Zen 4 data cache)
- With Q8_0 KV: tile_size=64 → 64×128×2×1.0625 = 17KB — even better

**Reference:** FlashAttention (Dao et al., 2022) — the algorithm is architecture-agnostic. CPU implementation differs only in tile size and prefetch strategy.

**Estimated gain:** 0-5% at short context, 10-30% at seq_len ≥ 4096. Marginal at typical seq_len < 128.

### 3. Parallel Batch Quantization

**Current state:** Quantization of activations before matmul_batch is serial per-token:
```rust
for s in 0..seq_len {
    quantize_q8_0(&residual[s*d..(s+1)*d], &mut batch_q8);
}
```

**Change:** Dispatch quantization across thread pool workers — each worker quantizes a subset of tokens. For 8 workers and seq_len tokens: each worker handles seq_len/8 tokens.

**Benefit:**
- Quantization currently ~4-5% of prefill time (from timing data)
- Parallelizing across 8 workers could reduce to ~0.5-1% (not 8× due to memory bandwidth)
- Frees CPU time for earlier matmul dispatch

**Cost:**
- Need to pre-allocate batch_q8 (already done) and write to known offsets
- Thread synchronization (gen counter handshake) — already exists in pool
- Memory bandwidth: each worker writes to different region of batch_q8, no contention

**Estimated gain:** 2-4% on prefill TTFT.

### 4. Fused RMS Norm + Quantize

**Current state:** In each forward_batch layer:
```rust
self.norm_batch(&hidden, attn_norm, eps, &mut residual, seq_len, d);
// separate step:
quantize residual → batch_q8
```

RMS norm reads `hidden[seq_len × d]` and writes `residual[seq_len × d]`. Quantize reads `residual` and writes `batch_q8`. Two separate passes over the same data.

**Change:** Fuse the two: after computing `out[i] = w[i] * x[i] / rms`, immediately quantize the result to Q8_0 and write to batch_q8. Eliminates the intermediate f32 write+read.

**Benefit:**
- Saves one full read+write of `seq_len × d` f32 values per layer per activation
- For 1.7B (d=2048, seq_len=10): 10 × 2048 × 4B = 80KB saved per fusion
- 4 fusions per layer × 28 layers = 8.96MB total bandwidth saved

**Cost:**
- More complex inner loop — must interleave norm and quantize operations
- Need per-element access to norm output (can't use SIMD norm + separate SIMD quantize as easily)
- May interfere with auto-vectorization

**Estimated gain:** 1-3% on prefill TTFT.

---

## Tier 2 — Medium Impact, Medium Effort

### 5. Parallel Head Norm Across Batch

**Current state:** Q/K head norm loops are serial:
```rust
for s in 0..seq_len {
    for h in 0..n_heads {
        self.norm(&q_heads[s*nq + h*head_dim..], q_norm, eps, &mut ...);
    }
}
```

**Change:** Use thread pool to parallelize the outer `s` loop. Each worker processes a subset of tokens for all heads.

**Estimated gain:** 1-2% on prefill at batch size ≥ 8.

### 6. Optimize Attention Batch Inner Loop

**Current state:** `attention_batch` weighted sum uses `to_array()` + `copy_from_slice` per SIMD chunk per position:
```rust
vacc = vv.mul_add(vatt, vacc);
out_row[start..start + 8].copy_from_slice(&vacc.to_array());
```

**Change:** Accumulate the weighted sum across all positions per SIMD chunk, write once:
```rust
for i in 0..chunks {
    let mut vacc = f32x8::ZERO;
    for pos in 0..attended_len {
        let vv = f32x8::from(&vs[pos * head_dim + i*8..][..8]);
        vacc = vv.mul_add(f32x8::splat(scratch[pos]), vacc);
    }
    out_row[i*8..(i+1)*8].copy_from_slice(&vacc.to_array());
}
```

**Benefit:**
- Eliminates O(attended_len × chunks) round-trips through memory for out_row
- One write per chunk instead of one write per position per chunk
- For seq_len=2048, head_dim=128, chunks=16: 2048×16=32768 writes → 16 writes (99.95% reduction)

**Cost:**
- Read vs data `chunks` times instead of once (but vs is small and cache-resident)
- Each chunk buffers all positions' partial sums, then writes once at end

**Estimated gain:** 2-5% on attention time (proportional to seq_len). Significant at long context.

### 7. Pre-Quantize KV Cache Write

**Current state:** `write_kv` stores f32 values. If switching to Q8_0 KV, the quantize-on-write path already exists in the KVCache struct (need to verify).

**Change:** Already implied by Item 1. Quantize KV during write_kv, dequant during attention read.

---

## Tier 3 — Speculative / Long-term

### 8. Speculative Prefill (SpecPrefill)
- Use a lightweight model to select a subset of "important" prompt tokens
- Only process important tokens through the full model
- Up to 7× TTFT improvement on GPU (SpecPrefill paper, 2025)
- Requires: secondary small model, token importance estimation
- **Gain:** 2-7× TTFT (but adds complexity and risk of quality loss)

### 9. Weight Prefetch for Prefill
- During prefill, weight rows are read `seq_len` times (once per token)
- Prefetch next weight row into L2 while current row is being processed
- Already has tile-in-L2 dispatch, but prefetch could be more aggressive
- **Risk:** Previous prefetch attempt (T1 hint) caused regression on Q1_0 models

### 10. Dynamic Shape GEMM (Sandwich approach)
- Sandwich generates custom GEMM kernels for prefill's dynamic shapes
- Uses micro-kernel tiling optimized for specific CPU cache hierarchy
- **Gain:** Up to 2× throughput on CPU (Sandwich paper, 2025)
- **Cost:** Requires code generation infrastructure — very high effort

---

## Summary Table

| # | Opportunity | Est. Gain | Effort | Depends on |
|---|---|---|---|---|
| 1 | Q8_0 KV Cache | 5-15% (long ctx) | Medium | KVCache rewrite |
| 2 | CPU Tiled Attention | 10-30% (long ctx) | High | New attention kernel |
| 3 | Parallel Batch Quantize | 2-4% | Low | ThreadPool dispatch |
| 4 | Fused RMS Norm + Quantize | 1-3% | Medium | Code refactor |
| 5 | Parallel Head Norm | 1-2% | Low | ThreadPool |
| 6 | Attention inner loop | 2-5% | Low | Code refactor |
| 7 | KV cache quantize (prereq for 1) | — | Medium | KVCache format |
| 8 | SpecPrefill | 2-7× | Very High | External model |
| 9 | Weight prefetch | 0-5% | Medium | Prefetch hints |
| 10 | Sandwich-style GEMM | Up to 2× | Very High | Code gen infra |

**Recommended order:**
1. Start with **#6** (attention inner loop) and **#3** (parallel quantize) — low effort, quick wins
2. Then **#1** (Q8_0 KV cache) — unlocks larger long-context gains
3. Then **#2** (tiled attention) — needs Q8_0 KV for best tile size
4. Then **#4** (fused norm+quantize) — only if profiling shows norm+quantize as >5%

---

## Reference: Key Papers & Implementations

| Paper/Project | Key Idea | CPU Relevance |
|---|---|---|
| Sandwich (Zhao et al., 2025) | Separate prefill/decode compilation, custom GEMM MKs | High — 2.01× CPU throughput |
| Litespark-Inference (Dade et al., 2026) | Custom SIMD for ternary {-1,0,+1}, VNNI on Zen4 | Medium — VNNI approach confirmed |
| FlashAttention (Dao et al., 2022) | Tiled online-softmax attention | High — architecture agnostic |
| SpecPrefill (Liu et al., 2025) | Token importance estimation, skip unimportant tokens | Medium — up to 7× TTFT |
| turboquant_plus (TheTom, 2025) | Q8_0 KV cache, 7.4% prefill gain | High — validates Q8_0 KV approach |
| llama.cpp (Gerganov et al.) | Batch prompt processing, n_batch parameter | Reference — batch size tuning |
| vLLM chunked prefill | Split long prompts into chunks for better scheduling | Low — GPU-focused |
