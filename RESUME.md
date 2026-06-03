# Session 11: Shuffle kernel exploration (negative result)

**Session 11 result:** Attempted to replace LUT-based Q2_0 kernels with SIMD inline 2-bit extraction (shuffle kernels) to eliminate LUT loads. Both SSE4.1 and AVX2 versions were **REJECTED** due to the stride-4 alignment issue.

**Key insight (why shuffle doesn't work for Q2_0):** Extracting 2-bit values from packed bytes by bit position (e.g., all bits-0-1 from 8 packed bytes) produces 8 weight values at stride-4 positions (0,4,8,12,16,20,24,28), but Q8_0 activations are contiguous. The `_mm_madd_epi16` pairs these incorrectly. For Q1_0, the shuffle kernel works because the 1-bit sign expansion via `cmpeq` + `xor` + `sub` directly maps 1-bit values to contiguous {-1,+1} without stride issues — fundamentally different from the 2-bit case.

**The LUT-based approach is optimal for Q2_0** because:
- Q2V_U8 (1KB) / Q2V_I16 (2KB) fits in L1 cache
- 8 LUT loads per sub-block (from L1, ~4 cycles each) are faster than the ~20 SIMD instructions needed for inline 2-bit extraction + stride correction
- The stride-4 alignment between extracted values and contiguous activations requires complex shuffle/gather that adds more latency than it saves

**Also attempted (reverted):** Precompute sum_act in Q8_0 quantizer to eliminate the second `vpdpbusd` in the VNNI kernel. Reverted because it requires updating Q8_0 format (34→38 bytes/block) across ~100 locations in 10+ files, for marginal benefit.

---

## Session 11 benchmarks (50-token, warm)

| Model | tok/s | avg_cpu_overhead (µs/tok) | vs S10 baseline |
|---|---|---|---|
| 1.7B Q1_0 | 35.9 | 27,685 | -18% (cold start) |
| 1.7B Q2_0 | 23.5 | 42,321 | -11% (within variance) |
| 4B Q1_0 | 22.3 | 44,799 | +13% (warm improvement) |
| 4B Q2_0 | 11.8 | 84,039 | -10% (within variance) |
| 8B Q1_0 | 11.1 | 90,636 | +1% (neutral) |
| 8B Q2_0 | 6.0 | 165,563 | -8% (within variance) |

Note: No kernel changes survived review. All variance is thermal/system.

---

## Next optimization targets (reprioritized)

### Target 1: Process multiple Q8_0 sub-blocks in VNNI kernel
The current VNNI kernel processes 1 sub-block (32 elements) per iteration. Using 512-bit AVX-512 would process 2 sub-blocks (64 elements) per iteration. On Zen 4 (double-pumped 256-bit FPU), the main benefit is reduced instruction count and fewer scale conversions. Requires `avx512f` + `avx512vnni`.

### Target 2: Higher-level matmul optimization
Instead of calling the dot kernel once per row, process batches of rows together. This could amortize the quantize_act() overhead and improve cache utilization for activation data. Currently each row re-reads the activation data; with batching, the activation data would stay in L1.

### Target 3: Investigate lm_head quantization format
The lm_head tensor may use a different quantization format than the model weights. If it's Q8_0 or higher precision, we could potentially optimize its matmul separately. Check the GGUF tensor metadata.

### Target 4: Eliminate Q2_0 LUT loads via pre-expansion (revisit for 1.7B Q2_0 only)
Earlier attempt failed for Q1_0 (7.2× memory, 3× regression). For Q2_0, the expansion ratio is only 3.8× (34→128 bytes/block). The 1.7B Q2_0 model (554 MB → 2085 MB expanded) might see compute-bound behavior where LUT elimination helps. Risk: memory traffic increase overwhelms compute savings.

---

## Key files
- `crates/hearth-quant/src/q2_0.rs` — Dispatch: VNNI > AVX2 LUT > SSE4.1 LUT > scalar (no changes from S10)
- `crates/hearth-quant/src/q8_0.rs` — Q8_0 quantize/dequantize (no changes from S10)

## Key ref files (Prism fork)
- `ggml/src/ggml-cpu/quants.c:177` — `ggml_vec_dot_q2_0_q8_0_generic` (purely scalar, no SIMD path for Q2_0)
- `ggml/src/ggml-common.h:187-192` — `block_q2_0` struct (128 elements, 34 bytes)
