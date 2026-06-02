# Hearth Performance Tracker

System: AMD Ryzen 7 8840HS (8C/16T), 16 GB DDR5, Windows 11. Hearth: 8 workers (n/2). Ref: 16 threads default. Bench: 50-token `--temp 0 --prompt "Hello" --prompt-raw`, `avg_cpu_overhead` tok/s.

All 6 models: Qwen3 architecture, head_dim=128, vocab=151669, YaRN rope scaling factor 4.0, Q/K head norms.

| Model | Format | d_model | ffn_dim | Layers | Heads | KV | Size |
|---|---|---|---|---|---|---|---|
| 1.7B Q1_0 | Q1_0 128/18 | 2048 | 6144 | 28 | 16 | 8 | 293 MB |
| 1.7B Q2_0 | Q2_0 128/34 | 2048 | 6144 | 28 | 16 | 8 | 554 MB |
| 4B Q1_0 | Q1_0 128/18 | 2560 | 9728 | 36 | 32 | 8 | 546 MB |
| 4B Q2_0 | Q2_0 128/34 | 2560 | 9728 | 36 | 32 | 8 | 1025 MB |
| 8B Q1_0 | Q1_0 128/18 | 4096 | 12288 | 36 | 32 | 8 | 1105 MB |
| 8B Q2_0 | Q2_0 128/34 | 4096 | 12288 | 36 | 32 | 8 | 2081 MB |

## Current status (2026-06-02, Session 6 — SIMD RoPE)

### SIMD RoPE with precomputed sin/cos: DONE
`RopeCache` struct was already defined but never wired into the model. The standalone `ops::rope()` function recomputed `theta.powf()` and `sin_cos()` on every token — 6,912 trig calls per forward (36 layers × 2 head groups × 80 dim ÷ 2 halves × 2 Q+K). 

**Changes:**
- Re-laid-out table: interleaved `[sin, cos, sin, cos]` → contiguous `[sin0..sin63, cos0..cos63]` per position, enabling `f32x8` loads
- `RopeCache::apply` now uses SIMD: 8 complex pairs per `f32x8` iteration (was scalar)
- Wired into `LlamaModel` — initialized at load time, all 6 `ops::rope()` calls replaced with `self.rope_cache.apply()`
- Removed unused `scaling_type`/`scaling_factor`/`orig_ctx` locals

**Result:** `rope` timing dropped from ~960µs to ~10µs per forward. Net ~2% forward time reduction.

### Batched prefill: DONE (from Session 5)
Wired `forward_batch` into `generate_text()` for CPU multi-token prompts. Optimized `matmul_batch` with Q8_0 quantized activations + custom ThreadPool + sequential quantize (no rayon contention).

| Config | Prefill 7t | Total 57t | Tok/s |
|---|---|---|---|
| Single-token (pre-alloc KV) | ~322ms | 3013ms | 18.9 |
| Batched (pre-alloc KV) | 264-420ms | 3213-3589ms | 15.9-17.7 |

### Critical bug fixed: KV cache lazy resize (from Session 5)
`KVCache::new()` created empty `k`/`v` Vecs. On first `write_kv` call, `resize(total, 0.0f32)` zeroed 128MB per vector (256MB per layer = 9.2GB total). Fixed by pre-allocating at construction.

### Remaining bottlenecks (4B Q1_0, single-token forward ~47ms)
From `[timing]` output (decode tokens):
- qkv_matmul ~30-37%  
- ffn_gate_up_matmul ~30-37%
- ffn_down_matmul ~15-23%
- attn_output_matmul ~9-12%
- lm_head_matmul ~4-8%

Next target: Fuse attn_out quant into attention kernel (est 3-5%).

| Model | Hearth | Ref | H/Ref | Forward |
|---|---|---|---|---|
| 1.7B Q1_0 | ~36 | 32.0 | 1.13× | ~22ms |
| 1.7B Q2_0 | ~24 | 5.1 | 4.70× | ~37ms |
| 4B Q1_0 | ~18 | 17.4 | 1.03× | ~48ms |
| 4B Q2_0 | ~11 | 2.8 | 3.86× | ~90ms |
| 8B Q1_0 | ~10.5 | 8.2 | 1.28× | ~90ms |
| 8B Q2_0 | ~5 | 1.5 | 3.07× | ~260ms |

## Change history

### 1.7B Q1_0

2026-05-xx SIMD inner kernel (AVX2 intrinsics + SSE4.1): 5.5→10.3 tok/s (+87%)
2026-05-xx Pointer-chasing outer loop: +5-8%
2026-05-xx format! elimination at load time (336/forward): +10%
2026-05-xx Q1_0 NaN fix (routed type 41 to 128-el kernel): +27%
2026-05-xx LUT AVX2 kernel (Q1V[256][8]): +32%
2026-05-xx Static chunking via rayon::scope: +8%
2026-06-01 target-cpu=native (.cargo/config.toml): +38% MT (2× single-threaded)
2026-06-01 par_iter + with_min_len (replaces rayon::scope): +30%
2026-06-02 Q1_0 recursion bug fix (fallback called itself): restored to 18.7 tok/s
2026-06-02 rayon::broadcast (replaces par_iter): 18.7→19.8 tok/s (+5-6%)
2026-06-02 Custom thread pool park/unpark (replaces Rayon): 19.8→34.4 tok/s (+74%)
2026-06-02 Gen counter + 8 workers + yield (Sess 3): 30.6→40.7 tok/s (+33%)
2026-06-02 Shuffle kernel (replaces LUT, fixes 8B collapse): 34.4→43 tok/s (+25%)
2026-06-02 Scratch buffer reuse (3 fewer allocs/forward): neutral
2026-06-02 i16 LUT (no Q1_0 changes): ~36 tok/s (system variance)

### 1.7B Q2_0

2026-05-xx Q2V[256] LUT + wide::i32x8 SIMD: 4.0→7.5 tok/s (+87%)
2026-06-02 AVX2 kernel rewrite (raw intrinsics, 16-el batches): 11.2→17.4 tok/s (+55%)
2026-06-02 rayon::broadcast: 17.4→18.3 tok/s (+5%)
2026-06-02 Custom thread pool park/unpark: 18.3→27.2 tok/s (+49%)
2026-06-02 Gen counter + 8 workers + yield (Sess 3): 19.3→24.5 tok/s (+27%)
2026-06-02 i16 LUT (Q2V_I16 pre-extended, skip sign extension): ~24 tok/s (within noise)
2026-06-02 SIMD RoPE (precomputed sin/cos table + f32x8 apply): rope 960µs→10µs

### 4B Q1_0

2026-06-02 Shuffle kernel (fixes 8B collapse, also helps 4B): 15.4→19 tok/s (+23%)
2026-06-02 Gen counter + 8 workers + yield (Sess 3): 9.0→15.2 tok/s (+68%)
2026-06-02 KV cache .to_vec() removal: 15.2→19.6 tok/s (+29%)
2026-06-02 Scratch buffer reuse: neutral
2026-06-02 i16 LUT, thread tuning: ~20 tok/s (variance)
2026-06-02 SIMD RoPE (precomputed sin/cos table + f32x8 apply): rope 960µs→10µs

### 4B Q2_0

2026-06-02 Gen counter + 8 workers + yield + KV cache (Sess 3): 8.6 tok/s
2026-06-02 i16 LUT (Q2V_I16, skip sign-extend): 8.6→10.8 tok/s (+26%)
2026-06-02 Thread tuning (10 workers): 10.8→12.0 tok/s (+11%)

### 8B Q1_0

2026-05-xx LUT kernel at d=4096 collapses: 0.63× ref (L1 thrash, 1.13× vs Q2_0)
2026-06-02 Shuffle kernel (no L1 cache pressure): 5.2→10.1 tok/s (+94%), 1.23× ref
2026-06-02 Gen counter + 8 workers + yield (Sess 3): 7.8→9.3 tok/s (+19%)
2026-06-02 Ref 8B Q1_0 scaling only 5.9× (vs 6.0× for 4B) — bandwidth-limited even on reference

### 8B Q2_0

2026-06-02 Gen counter + 8 workers + yield + KV cache (Sess 3): 4.6 tok/s
2026-06-02 i16 LUT: no change (bandwidth-bound)
2026-06-02 Thread tuning (10 workers): 4.6→6.5 tok/s (+41%)

## What didn't work

Prism-style conditional add/sub kernel: 5× slower than LUT (branchy codegen)
wide::i32x8::new([...]): construction overhead > SIMD benefit
LM head chunking via par_chunks_mut: slower
SSE4.1 path: 4.5× slower than AVX2
QKV fusion: <3% gain (diminishing returns)
Reference-style AVX2 kernel rewrite: ~0% gain (LLVM already optimized)
Kernel hsum optimization (FMA accumulate across blocks, single hsum per row): marginal
LM head F32: both 1.7B models have Q1_0/Q2_0 lm_head, not F32 (can't skip dequant)
Q8_0 quant fusion: negligible (<0.5% of forward pass)
MSVC FFI kernel: 4-6× slower than LLVM + target-cpu=native
Raw std::thread::scope: catastrophic on Windows (5.2/2.8 tok/s)
Spin-wait pool (no yield): 100% CPU, starved main thread
LLVM codegen flags: +-slow-unaligned-mem-256 not recognized by Rust LLVM
6 workers: worse on all models tested (4B Q2_0: 10.8→8.4, 8B Q2_0: 4.6→3.5)
10 workers with 1.7B models: catastrophic (36→8 tok/s, d=2048 too small)

## Key insights

target-cpu=native was the largest single gain (~2× ST, +38% MT)
Custom thread pool (park/unpark) was the second largest (+74%/+49%)
Shuffle kernel fixed 8B Q1_0 collapse (L1 thrash at 15 threads × 2KB LUT)
8 workers (n/2) eliminated SMT contention for bandwidth-bound matmuls
Q1_0 shuffle kernel beats reference on ALL model sizes
Q2_0 is 3-5× faster than reference at all sizes
LLVM generates better AVX2 code than MSVC SSE2 for quant kernels
10 workers helps large models (4B/8B Q2_0: +11%/+41%) but kills small ones
RopeCache was dead code — struct defined but never instantiated; ops::rope() recomputed trig every token

## Per-token forward pass (warm, non-prefill)

### 1.7B Q1_0 (~22ms)
ffn_gate_up_matmul 41% | ffn_down_matmul 30% | qkv_matmul 16% | attn_output_matmul 4% | lm_head_matmul 5% | rest 5%

### 1.7B Q2_0 (~47ms)
ffn_gate_up_matmul 34% | ffn_down_matmul 18% | qkv_matmul 16% | attn_output_matmul 9% | lm_head_matmul 10% | attention 5% | rope ~0.02% | rest 6%

### 4B Q1_0 (~47ms)
ffn_gate_up_matmul 30-37% | ffn_down_matmul 15-23% | qkv_matmul 30-37% | attn_output_matmul 9-12% | lm_head_matmul 4-8% | rope ~0.02% | rest ~2%

### 4B Q2_0 (profile pre-S3, ~355ms; post-S4 ~90ms)
ffn_gate_up_matmul 40% | ffn_down_matmul 19% | qkv_matmul 16% | kv_cache_write 12% | attn_output_matmul 10% | lm_head_matmul 2% | rest 3%

### 8B Q1_0 (profile pre-S3, ~393ms; post-S4 ~90ms)
ffn_gate_up_matmul 39% | ffn_down_matmul 24% | qkv_matmul 15% | attn_output_matmul 9% | lm_head_matmul 5% | rest 8%

### 8B Q2_0 (profile pre-S3, ~634ms; post-S4 ~260ms)
ffn_gate_up_matmul 46% | ffn_down_matmul 25% | qkv_matmul 16% | attn_output_matmul 9% | lm_head_matmul 2% | rest 2%

## Next up

Fuse Q8_0 activation quant into first matmul row: eliminates full-vector read per dispatch. Marginal (~5%)
Model-size-aware thread count: d>=2560 → 10 workers, d<2560 → 8 workers
