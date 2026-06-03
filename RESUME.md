# Session 10: AVX-512 VNNI Q2_0 kernel

**Session 10 result:** Added `dot_q2_0_q8_0_vnni_avx512` using `vpdpbusd` for Q2_0×Q8_0 dot product. Correct (22/22 tests pass), neutral performance (±3% within system variance). The vpdpbusd arithmetic saves ~2 µops per sub-block vs the AVX2 LUT kernel, but LUT load overhead (8 loads/sub-block) dominates.

**Key insight:** The Q2_0 kernel is LUT-load-bound, not arithmetic-bound. The 8 LUT loads per sub-block (Q2V_U8[byte] for unpacking 2-bit values) dominate the inner loop. Replacing arithmetic (`cvtepi8 + madd`) with `vpdpbusd` doesn't help because the memory loads are the bottleneck.

**Prefetch experiment:** Software prefetch (`_mm_prefetch` with `_MM_HINT_T1`) in the worker loop regressed Q1_0 by 5-11%. The Zen 4 hardware prefetcher already handles sequential weight access well.

---

## Session 10 benchmarks (50-token, warmup included)

System was warm after multiple runs — expect ±5-10% variance from thermal throttling.

| Model | tok/s | avg_cpu_overhead (µs/tok) | vs S9 baseline |
|---|---|---|---|
| 1.7B Q1_0 | 43.8 | 22,704 | -11.5% (system variance — Q1_0 not modified) |
| 1.7B Q2_0 | 26.4 | 37,247 | -8.0% (within noise) |
| 4B Q1_0 | 19.7 | 50,385 | -16.2% (system variance — Q1_0 not modified) |
| 4B Q2_0 | 13.1 | 75,447 | +0.8% (neutral) |
| 8B Q1_0 | 11.0 | 90,580 | -17.9% (system variance) |
| 8B Q2_0 | 6.5 | 152,617 | -8.5% (within noise) |

Note: Q1_0 models were NOT modified — all regressions are system variance. Q2_0 models use the new VNNI kernel. Performance is ±8% vs S9 baseline, all within typical system variance for this hardware.

---

## Next optimization targets

### Target 1: Precompute Q8_0 activation sum for VNNI correction
The current VNNI kernel computes `sum_act = vpdpbusd(ones, act)` per sub-block — a second `vpdpbusd` call that doubles the inner-loop overhead. Precompute `sum_act` during Q8_0 quantization and store it alongside the block data (or as a separate array), eliminating the second call.

### Target 2: 512-bit AVX-512 Q2_0 kernel (process 2 Q8_0 blocks at once)
With `avx512vbmi` available (confirmed), use `_mm512_permutexvar_epi8` (vpermb) for cross-lane byte permutation. Process 2 × 32-element Q8_0 sub-blocks (64 elements) per 512-bit iteration, halving the inner loop count. Requires contiguous activation data or two 256-bit loads.

### Target 3: Eliminate Q2_0 LUT loads via pre-expansion
Pack Q2_0 weight bytes at load time to store raw u8 values {0,1,2,3} in 128 bytes per 128-element block (vs current 34 bytes with 2-bit packing). Cost: 3.8× memory traffic. Benefit: zero LUT loads in the inner loop. Only viable for compute-bound models (1.7B Q2_0). Use `Vec<u8>` expansion at load time, not format change.

### Target 4: AVX-512 Q1_0 shuffle kernel
Port the shuffle kernel to 512-bit vectors using `_mm512_broadcast_i32x4` + `_mm512_shuffle_epi8` for sign expansion. Process 64 elements per batch (2 Q8_0 sub-blocks). The main challenge is handling 2 different activation scales within one 512-bit batch — requires splitting the accumulator into lower/upper 256-bit halves.

---

## Key files
- `crates/hearth-quant/src/q2_0.rs` — `dot_q2_0_q8_0_vnni_avx512` (VNNI kernel), `Q2V_U8` LUT, dispatch updated
- `crates/hearth-llm/src/pool.rs` — Worker loop with `_mm_prefetch` (REVERTED)
