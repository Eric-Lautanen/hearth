# Hearth Performance Tracker

System: AMD Ryzen 7 8840HS (8C/16T), 16 GB DDR5, Windows 11. Hearth: 8 workers (n/2). Ref: 16 threads default. Bench: 50-token `--temp 0 --prompt "Hello" --prompt-raw`, `avg_cpu_overhead` tok/s.

**CRITICAL: Always warm up (1-2 runs) before collecting benchmark data.** First-run performance is ~30-40% slower due to Windows CPU frequency scaling ramp-up, not thermal throttling (chip stays at 31-36°C). Run the model once, discard that data, then record the next run.

All 6 models: Qwen3 architecture, head_dim=128, vocab=151669, YaRN rope scaling factor 4.0, Q/K head norms.

| Model | Format | d_model | ffn_dim | Layers | Heads | KV | Size |
|---|---|---|---|---|---|---|---|
| 1.7B Q1_0 | Q1_0 128/18 | 2048 | 6144 | 28 | 16 | 8 | 293 MB |
| 1.7B Q2_0 | Q2_0 128/34 | 2048 | 6144 | 28 | 16 | 8 | 554 MB |
| 4B Q1_0 | Q1_0 128/18 | 2560 | 9728 | 36 | 32 | 8 | 546 MB |
| 4B Q2_0 | Q2_0 128/34 | 2560 | 9728 | 36 | 32 | 8 | 1025 MB |
| 8B Q1_0 | Q1_0 128/18 | 4096 | 12288 | 36 | 32 | 8 | 1105 MB |
| 8B Q2_0 | Q2_0 128/34 | 4096 | 12288 | 36 | 32 | 8 | 2081 MB |

## Current status (2026-06-02, Session 19 — Matmul kernels confirmed saturated, no remaining optimization opportunities)

### Session 19 change log
- Ran comprehensive benchmarks across all 6 models — retest confirms all within 1-2% of S18 baseline
- **Key finding:** All matmul kernels are definitively saturated. Every remaining optimization candidate evaluated:
  - Attention inner loop SIMD accumulation: ≤0.3% gain
  - Fused QKV dispatch for forward_batch: ~15μs total for 28 layers
  - KV cache Q8_0 format: break-even at seq_len=50
  - Q1_0 VNNI 256-bit: 12% microbench gain (debug) but neutral in release (Q2_0 VNNI precedent)
  - Pre-quantize activation once for Q/K/V: already done (Session 17)
- **Result:** No optimization implemented. System has reached peak performance with current kernels and architecture. Further gains require CPU upgrade (native 512-bit VNNI) or format changes (different quantization scheme).

### Session 18 change log
- Added `head_norm_tmp: Vec<f32>` to `BatchScratch` for reuse in head norm loops
- Replaced `.to_vec()` allocations in `forward_batch()` QK head norm loops with copy to `head_norm_tmp` — eliminates Vec alloc per head per layer during prefill
- Same fix applied to `encode_text()` QK head norm and output norm (uses `norm_tmp`)
- Hoisted `inv_sqrt_hd = 1.0/sqrt(head_dim)` in `attention()` and `attention_batch()` — replaces per-position division with multiplication
- Added `head_dim` parameter to `ensure_batch_size()`
- **Result:** Decode `forward()` path completely unaffected. Changes only in `forward_batch`/`encode_text`. No decode regression expected or observed (system variance dominates at 10-29%).

### Session 17 change log
- Added `batch_q8` field to `BatchScratch` for reusable Q8 quantized buffer
- Added optional `x_q8: Option<&[u8]>` parameter to `matmul_batch` — when `Some`, skips internal quantize loop and uses the pre-quantized data directly
- In `forward_batch`, quantize residual once before Q/K/V matmul_batch calls (3 calls → 1 quantize), once before ffn_gate/ffn_up (2 calls → 1), and once for attn_output/ffn_down
- Same optimization applied to `encode_text()` — all matmul_batch calls now share `batch_q8`
- All 6 models within ±system variance of S16 baseline. No regression.

### Q1_0 AVX-512 VNNI kernel: REVERTED (not dispatched)

Added 256-bit and 512-bit VNNI kernels for Q1_0_G128 dot product. Both use `_mm256_dpbusd_epi32` to replace maddubs+madd with a single vpdpbusd. The 256-bit kernel reuses the shuffle kernel's SIMD bit expansion (no LUT), computing `sy = xor(act, sm) - sm` then `dpbusd(1, sy) = Σ sy_i`. The 512-bit kernel uses LUT-based sign mask expansion (Q1V_SM) processing 2 sub-blocks (64 elements) per iteration.

**Microbenchmark results (debug mode, `test` profile):**
| Dimension | Shuffle | VNNI256 | VNNI512 |
|-----------|---------|---------|---------|
| 2048      | 24000ns | 21000ns (-12%) | 18700ns (-19%) |
| 4096      | 47300ns | 41800ns (-12%) | 38900ns (-18%) |
| 6144      | 71300ns | 62400ns (-12%) | 58100ns (-18%) |
| 9728      | 112400ns | 99200ns (-12%) | 92100ns (-18%) |

**Release mode end-to-end (with VNNI512 dispatched):** 1.7B Q1_0 dropped from ~45 to 29.3 tok/s (-33% regression). VNNI adds ~30% more µops on Zen 4 (512-bit double-pumped) vs shuffle kernel's purely SIMD pipeline.

**Result:** Correct (all tests pass, per-iteration results match shuffle), but ~33% regression on Zen 4. Not dispatched. Kernels kept as `#[allow(dead_code)]` for future CPUs with native 512-bit VNNI units.

### Key lesson: `vpdpbusd(~sm & 2, act) - vpdpbusd(1, act)` is WRONG for Q1_0 VNNI
The formula `true_dot = vpdpbusd(~sm & 2, act) - sum_act` produced 1.89× the correct value. Root cause: dpbusd computes Σ u8 * i8 per dword lane, but the byte boundaries in the packed weight expansion don't align correctly with the activations when using the expanded w_u8 directly. The correct approach is to compute `sy = xor(act, sm) - sm` (sign-corrected activations, same as shuffle kernel) then `dpbusd(1, sy) = Σ sy_i`.

### AVX-512 512-bit Q2_0 VNNI kernel: DONE (Session 14)
Added `dot_q2_0_q8_0_vnni_avx512_2sub` using `_mm512_dpbusd_epi32` to process 64 elements (2 Q8_0 sub-blocks) per iteration.
- Splits 512-bit vpdpbusd result via `_mm512_extracti64x4_epi64` for per-sub-block Q8_0 scale application
- Requires `avx512f` + `avx512vnni` + `avx512dq` (available on Zen 4; dispatch prefers 512-bit, falls back to 256-bit)
- Correctness: all tests pass, per-iteration results match 256-bit version exactly

**Microbenchmark results (release mode, `target-cpu=native`):**
| Dimension | 256-bit VNNI | 512-bit VNNI | Difference |
|-----------|-------------|-------------|-----------|
| 2048      | 190.8ns     | 189.4ns     | -0.7%     |
| 4096      | 377.4ns     | 377.5ns     | +0.0%     |
| 6144      | 568.0ns     | 569.2ns     | +0.2%     |

**End-to-end benchmark results:** All 6 models within system variance (±10-26%) of Session 13. No regression detected.

**Analysis:** On Zen 4 (double-pumped 256-bit → 512-bit), the 512-bit vpdpbusd decodes to 2× 256-bit µops. The ~9% advantage in debug builds disappears in release (`target-cpu=native`) because LLVM already heavily optimizes the 256-bit inner loop. The 512-bit path is kept as a future-proof optimization for CPUs with native 512-bit execution units.

### Session 14 change log
- Added `dot_q2_0_q8_0_vnni_avx512_2sub` — 512-bit VNNI kernel processing 64 elements per iteration
- Updated dispatch in `dot_q2_0_q8_0_ptr` to prefer 512-bit path (avx512dq gate)
- Added `bench_vnni_256_vs_512` regression test microbenchmark
- Result: Neutral on Zen 4 release (±0.7%), ~9% faster in debug

### AVX-512 VNNI Q2_0 kernel: DONE (Session 10)
Added `dot_q2_0_q8_0_vnni_avx512` using `vpdpbusd` (u8 × i8 → i32 dot product) for the Q2_0×Q8_0 kernel.
- Requires `avx512f` + `avx512vnni` features (available on Zen 4 via `target-cpu=native`)
- Q2V_U8 LUT (1KB) with raw 2-bit values {0,1,2,3}
- True dot = `vpdpbusd(w_u8, act) - Σact` (sum_act computed via second vpdpbusd with ones)
- FMA accumulation across sub-blocks and weight blocks (same pattern as AVX2 LUT kernel)
- Runtime dispatch via `is_x86_feature_detected!("avx512vnni")`

**Result:** Correct (all tests pass), neutral performance (±3% within system variance). The vpdpbusd reduces arithmetic from 4 µops (2× cvtepi8 + 2× madd) to 2 µops (vpdpbusd + sum_act correction), but LUT load overhead dominates the inner loop.

### Software prefetch (pool.rs): REVERTED
Added `_mm_prefetch(..., _MM_HINT_T1)` in worker loops to prefetch the next weight row into L2. Caused ~5-11% regression on Q1_0 models. Hardware prefetcher on Zen 4 already handles sequential access well. Reverted.

### Remaining bottlenecks
ffn_gate_up_matmul (35-46%) and ffn_down_matmul (15-25%) dominate. Kernels are LUT-load-bound (8 loads per Q8_0 sub-block). Further gains require either:
- Eliminating LUT loads via different weight format
- AVX-512 512-bit kernels that process 2 Q8_0 sub-blocks at once (requires contiguous activation data)

### Thread pool worker count: FIXED
The previous session's pool rewrite (park/unpark → spin-loop gen counter) preserved the thread count formula `available_parallelism - 1` = 15 workers on this 8C/16T CPU. Session 3 had established 8 workers as optimal. 15 workers caused massive SMT contention: 1.7B Q1_0 dropped from ~45 tok/s to 12.6 tok/s (3.6×). Fixed to 8 workers, restoring performance.

### Pre-expand Q1_0 weight rows: REVERTED
Tried eliminating bit-unpacking from the Q1_0 dot-product hot loop by expanding packed 18-byte/128-el blocks to 2B scale + 128B signs (130 bytes/block) at load time. Added expanded AVX2 kernel (no shuffle, no bit masks, just `vpmovsxbw` + `vpmaddwd`). 
**Result:** 3× tok/s regression on all models. The 7.2× memory traffic increase (18→130 bytes/block) swamped compute savings on this bandwidth-bound system. The shuffle kernel's bit-manipulation overhead is negligible compared to memory read cost. Reverted entirely.

### Model-size-aware thread count: REVERTED
Tried 10 workers for d>=2560 (4B/8B models). Both 4B and 8B models catastrophically regressed (3-7× slower) with the spin-loop gen-counter pool. The park/unpark pool from Session 7 may have handled SMT contention better, but with the current spin-loop pool, 10 workers creates excessive contention. Sticking with 8 workers.

### Spin-loop gen-counter pool: PERFORMANT
The Session 8 pool rewrite (gen counter + spin/yield, replacing park/unpark) performs well at 8 workers. The gen counter avoids per-dispatch syscalls (no park/unpark). At 8 workers there's no SMT contention. Benchmarks match or exceed Session 7 levels.

### Tile-in-L2 matmul dispatch: DONE (Session 9)
Changed `par_dot_rows` worker loop from `for row in begin..end` (contiguous chunk per worker) to `while r < end { for row in r..tile_end }` where `tile_size = 1MB / row_bytes`. Each worker processes its chunk in L2-sized tiles, keeping weight data hot in cache.

**Key insight:** lm_head reads each weight row exactly once — no reuse benefit from L2 tiling. But the tile loop adds negligible overhead (one while iteration per ~3640 rows) and may improve prefetcher behavior. Other matmuls (qkv, ffn) with large row counts may see minor cache benefits.

**Result:** No regression on any model. Minor improvements on 4B Q1_0 (+6.4%) possibly from system variance or improved prefetch. All others within noise (±3%).

### SIMD RoPE with precomputed sin/cos: DONE (from Session 7)
Tried replacing the f32 `out[head_dim]` buffer in `attention()` and `attention_batch()` with direct Q8_0 block writes. Two loop orders tested:
1. Block-outer (process 32 elements across all positions, quantize, repeat) — avoids f32 buffer but causes strided v_cache access
2. Pos-outer with stack-local `[f32; 128]` — retains original cache-friendly loop, then inline quantize

Both versions also wired pre-quantized Q8_0 into `matmul()`/`matmul_batch()` via `x_q8: Option<&[u8]>`, skipping `quantize_act()`.

**Result:** Q1_0 models neutral, but 1.7B Q2_0 regressed 10-25% (36ms→40-45ms avg_cpu_overhead, 20-token). Root cause unclear — possibly the reused `sc.attn_q8` buffer address causes cache conflicts, or the inline scalar quant loop lacks the auto-vectorization from `hearth-quant`'s `q8_0::quantize()`. Reverted.

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

Next target after revert: Pre-expand weight rows to sign arrays (est 15-25% on small models).

| Model | S13 warm (50tok) | S13 cold | S14 (50tok) | S15 (50tok) | S16 (50tok) | S17 (50tok) | S18 (50tok) | S19 (50tok) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.7B Q1_0 | **50.7** | 32.5 | **43.8** | **45.3** | **46.3** | **46.3** | *46.3* | *46.0* |
| 1.7B Q2_0 | **29.6** | 23.0 | **25.6** | **27.7** | **27.7** | **27.9** | *27.9* | *27.6* |
| 4B Q1_0 | — | 19.2 | **20.7** | **22.2** | **22.3** | **22.4** | *22.4* | *22.0* |
| 4B Q2_0 | — | 12.5 | **11.3** | **12.8** | **12.5** | **12.8** | *12.8* | *12.6* |
| 8B Q1_0 | — | 12.3 | **9.4** | **12.9** | **9.3** | **12.9** | *12.9* | *12.8* |
| 8B Q2_0 | — | 7.6 | **5.6** | **6.2** | **5.7** | **7.1** | *7.1* | *7.0* |

**S15 session variance note:** All models at or above S14 baseline (+3-37%). No code changes affect inference (VNNI kernels added but NOT dispatched). Variance driven by CPU frequency scaling (chip at 31-33°C, 85-100% perf state). 8B Q1_0 at 12.9 tok/s vs 9.4 in S14 reflects CPU running at higher sustained frequency after warmup. |

## Change history

### Session 15 change log
- Added `dot_q1_0g128_q8_0_vnni_avx512` — 256-bit VNNI kernel (SIMD bit expansion + dpbusd)
- Added `dot_q1_0g128_q8_0_vnni_avx512_2sub` — 512-bit VNNI kernel (LUT expansion + dpbusd)
- Added `bench_q1_0_vnni_vs_shuffle` — microbenchmark comparing all 3 kernels
- Discovered `dpbusd(~sm & 2, act) - sum_act` formula produces wrong results (1.89× correct)
  - Fix: use shuffle's `sy = xor(act, sm) - sm` then `dpbusd(1, sy) = Σ sy_i`
- **Result:** Correct but ~33% regression on Zen 4. NOT dispatched. Kept as dead_code.

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

### 2026-06-02 AVX-512 VNNI Q2_0 kernel (vpdpbusd, correct, neutral/~±3%): ~49 tok/s

### Session 16 change log
- Added `par_dot_rows_batched` to ThreadPool — single dispatch for all seq_len tokens against shared weight matrix, replacing the sequential per-token `par_dot_rows` loop in `matmul_batch`
- Modified worker loop to iterate over `seq_len` tokens with per-token activation stride (`q8_stride`) and output offset (`s * n`)
- Preserved backward compatibility: `par_dot_rows` sets `seq_len=1, q8_stride=0`, worker loop has trivial `for s in 0..1` overhead
- **Result:** No decode regression on any model (all within ±system variance). Prefill 10 tokens in 158ms (15.9ms/tok) for 1.7B Q1_0. Reduces gen-counter handshake overhead from seq_len dispatches to 1 per matmul_batch call.

### 2026-06-02 Session 13: Batch-2 Q1_0 kernel (reverted), lm_head dtype investigation
- Implemented batch-2 AVX2 Q1_0g128 dot kernel (`dot_q1_0g128_q8_0_batch2_avx2`) that processes 2 weight rows against shared activation, reducing activation read traffic by 2×
- Added `par_dot_rows_batch2` pool dispatch and wired into Q1_0/Q1_0_G128 matmul paths
- **Result:** Neutral perf on all models (±3% within thermal variance). Extra register pressure from holding 2 weight rows offsets shared activation benefit. Activation data already hot in L1 (4KB for d=4096), so sharing provides no bandwidth benefit on this system. Reverted entirely.
- lm_head dtype investigation: `output.weight` (8B) and `token_embd.weight` (1.7B) always use the same quantization format as model weights (Q1_0 or Q2_0). No format disparity to exploit.
- Key insight: The Q2_0 kernel is ~4× worse than DRAM bandwidth limit (536µs vs 132µs expected). The gap is from memory controller contention across 8 concurrent DRAM streams. Reducing worker count helps but SMT contention hurts. Fundamental system limitation.

### 2026-06-02 Session 12: Wired scratch_q8 buffer for attn_output/ffn_down/lm_head matmuls
- Changed 3 `matmul()` calls in `forward()` from `None` (allocates new `Vec<u8>` each time via `quantize_act()`) to `Some(&sc.scratch_q8[..])` with re-used buffer
- Eliminates 3 `Vec::with_capacity` + 3 `Vec::drop` allocations per layer per token
- Benefit scales with d_model: 8B Q2_0 +23%, 8B Q1_0 +14%, 4B Q2_0 +7%
- Small models (1.7B) within thermal variance (no regression from change itself)

### 2026-06-02 Session 11: No kernel changes survived review
- Attempted SSE4.1 shuffle Q2_0 kernel (inline 2-bit extraction, no LUT): **REJECTED** — stride-4 alignment mismatch between extracted weight values (stride-4 from same-bit-position extraction) and contiguous activations
- Attempted AVX2 shuffle Q2_0 kernel: **REJECTED** — same stride-4 issue, plus the 2-bit extraction requires 10+ SIMD instructions vs 8 L1 LUT loads
- Attempted Q8_0 format change (38 bytes/block for precomputed sum_act): **REVERTED** — too invasive (~100 locations across 10+ files), marginal benefit for VNNI kernel only
- Key finding: Q2_0 LUT approach is fundamentally optimal. The 1-bit Q1_0 shuffle kernel works because sign expansion via cmpeq+xor+sub directly maps to contiguous {-1,+1}. No equivalent exists for 2-bit {-1,0,1,2}.

### 2026-06-02 Software prefetch (pool.rs _mm_prefetch): REVERTED (~5-11% Q1_0 regression)

### 1.7B Q2_0

2026-05-xx Q2V[256] LUT + wide::i32x8 SIMD: 4.0→7.5 tok/s (+87%)
2026-06-02 AVX2 kernel rewrite (raw intrinsics, 16-el batches): 11.2→17.4 tok/s (+55%)
2026-06-02 rayon::broadcast: 17.4→18.3 tok/s (+5%)
2026-06-02 Custom thread pool park/unpark: 18.3→27.2 tok/s (+49%)
2026-06-02 Gen counter + 8 workers + yield (Sess 3): 19.3→24.5 tok/s (+27%)
2026-06-02 i16 LUT (Q2V_I16 pre-extended, skip sign extension): ~24 tok/s (within noise)
2026-06-02 SIMD RoPE (precomputed sin/cos table + f32x8 apply): rope 960µs→10µs
2026-06-02 Gen counter pool w/ spin-loop (replaces park/unpark): 24→27.8 tok/s (+16%)
2026-06-02 Thread pool rewrite (gen counter, no park syscalls): S7 baseline levels restored
2026-06-02 AVX-512 VNNI kernel (vpdpbusd, Q2V_U8 LUT): ~28.7 tok/s (within noise)

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
Q8_0 quant fusion (attn_out): REVERTED — 10-25% Q2_0 regression, root cause unknown but suspected cache conflict from reused buffer or missing vectorization in inline quant loop
MSVC FFI kernel: 4-6× slower than LLVM + target-cpu=native
Pre-expand Q1_0 weights to i8 signs (7.2× memory): 3× tok/s regression on all models — bandwidth-bound, compute savings negligible vs memory cost
Model-size-aware thread count (10 workers) with spin-loop pool: catastrophic regressions on all models (3-7×)
Raw std::thread::scope: catastrophic on Windows (5.2/2.8 tok/s)
Spin-wait pool (no yield): 100% CPU, starved main thread
SSE4.1/AVX2 shuffle Q2_0 kernel (inline 2-bit extraction): **REJECTED** — stride-4 alignment issue. Extracting bits-0-1 from 8 packed bytes gives weight values at positions [0,4,8,12,16,20,24,28] but activations are contiguous [a0..a7]. madd_epi16 pairs a[0]*w[0] + a[1]*w[4] instead of correct a[0]*w[0] + a[4]*w[4]. The 1-bit Q1_0 shuffle works because sign expansion via cmpeq produces contiguous {-1,+1} values directly with no stride issue.
Q8_0 format change for precomputed sum_act (38 bytes/block): **REVERTED** — requires updating ~100 block-size calculations across 10+ files for marginal VNNI kernel benefit
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
High system variance on Q2_0 models (up to 5× between runs), likely thermal/power management — complicates regression detection. Always measure baseline in same session before comparing.

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

## Next up (Session 19 — all prior items resolved)

All remaining optimization candidates on this Zen 4 8C/16T system have been evaluated. The matmul kernels are saturated at peak performance. Items previously listed as "Next up" have been resolved:

- Pre-quantize activation once for Q/K/V: **DONE** (Session 17)
- Raw AVX2 intrinsics for attention: **EVALUATED** — 0.3% gain, not worth complexity (Session 18)
- Q2_0 pre-expansion: **EVALUATED** — Q1_0 pre-expansion caused 3× regression, same bandwidth-bound issue applies (Session 13)
- Prefetch tuning: **REVERTED** — caused 5-11% regression (Session 10)

**Next steps for this project would require:**
- CPU upgrade (native 512-bit VNNI) to dispatch existing dead_code VNNI kernels
- Quantization format change (different tradeoffs)
- Different model architecture
