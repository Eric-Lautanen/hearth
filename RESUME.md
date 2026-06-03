# Session 13: Batch-2 Q1_0 kernel (reverted), lm_head investigation

**Session 13 results:**
1. **Batch-2 Q1_0 dot kernel:** Implemented `dot_q1_0g128_q8_0_batch2_avx2` that processes 2 weight rows against a shared activation (halving activation read traffic). Added `par_dot_rows_batch2` pool dispatch. **Result: Neutral** — activation data is already hot in L1 (4KB at d=4096), so sharing provides no bandwidth benefit. Extra register pressure from holding 2× weight bits offsets the instruction savings. **Reverted.**
2. **lm_head dtype:** Checked all 6 models. lm_head always uses the same format as model weights (Q1_0 or Q2_0). No format disparity to exploit.
3. **Root cause analysis:** The Q2_0 kernel is ~4× worse than the DRAM bandwidth limit (536µs actual vs 132µs theoretical per layer for 1.7B Q2_0 ffn_gate_up). The gap comes from memory controller contention with 8 concurrent DRAM streams — a fundamental system limitation on this 8C/16T CPU.

---

## Next optimization targets (re-evaluated)

### Target 1: Reduce Q2_0 kernel instruction count by replacing LUT loads with SIMD shuffle extraction
The Q2_0 LUT AVX2 kernel spends significant uops on `copy_nonoverlapping` LUT loads (512 copies per dot call at d=2048). Replace with `vpshufb`-based inline 2-bit extraction similar to the Q1_0 shuffle kernel. Challenge: stride-4 alignment (2-bit pairs from same byte map to strided positions). Worth re-investigating with a 16-element-at-a-time approach rather than the previously failed 32-element approach. Benchmark on 1.7B Q2_0 first (most compute-bound).

### Target 2: Process multiple Q8_0 sub-blocks in VNNI kernel
The current VNNI kernel processes 1 sub-block (32 elements) per iteration. Using 512-bit AVX-512 would process 2 sub-blocks (64 elements). On Zen 4 (double-pumped 256-bit FPU), main benefit is reduced instruction count and fewer scale conversions. However, VNNI kernel was neutral for Q2_0 (LUT load overhead dominated), so this is unlikely to help without first fixing Target 1.

### Target 3: Prefetch tuning (revisit with model-size-aware threshold)
Software prefetch was previously tried with `_MM_HINT_T1` and caused ~5-11% regression on Q1_0 models. Try `_MM_HINT_T0` (L1 prefetch) instead of T1 (L2 prefetch), or only for large models (d>=2560) where DRAM bandwidth is the bottleneck and prefetch can overlap latency.

### Target 4: Q2_0 pre-expansion (revisit for 1.7B Q2_0 only)
Earlier attempt failed for Q1_0 (7.2× memory, 3× tok/s regression). For Q2_0, expansion is 3.8× (34→128 bytes/block). 1.7B Q2_0 is the most compute-bound model (smallest weight:activation ratio), so eliminating LUT loads may help. Risk: 3.8× memory traffic might still swamp compute savings. Build a single expanded row and microbenchmark before committing to full model expansion.

---

## Key files
- `crates/hearth-llm/src/model/matmul.rs` — matmul dispatch for Q1_0/Q2_0
- `crates/hearth-quant/src/q2_0.rs` — Q2_0 dot kernel (LUT+q2v_i16+q2v_u8)
- `crates/hearth-quant/src/q1_0g128.rs` — Q1_0 dot kernel (shuffle AVX2)
- `crates/hearth-llm/src/pool.rs` — gen-counter thread pool (par_dot_rows dispatch)

## Key ref files (Prism fork)
- `ggml/src/ggml-cpu/quants.c:177` — `ggml_vec_dot_q2_0_q8_0_generic` (purely scalar, no SIMD path for Q2_0)
- `ggml/src/ggml-common.h:187-192` — `block_q2_0` struct (128 elements, 34 bytes)
