# Session 12: scratch_q8 buffer wiring (success)

**Session 12 result:** Wired the existing `scratch_q8` buffer (`ForwardScratch::scratch_q8`) into 3 `matmul()` calls in `forward()` that were passing `None` (triggering a fresh `Vec<u8>` allocation per call via `quantize_act()`).

**Changed matmuls:** `attn_output`, `ffn_down`, `lm_head` — each now clears the shared `scratch_q8` buffer, quantizes the activation into it, and passes `Some(&sc.scratch_q8[..])`.

**Benefit:** Eliminates 3 `Vec::with_capacity` + 3 `Vec::drop` cycles per layer per token (~84 deallocations per forward pass for 28-layer models). The benefit scales with d_model because larger allocations are more expensive.

**Performance impact:**
| Model | S12 tok/s | S11 tok/s | Change |
|---|---|---|---|
| 1.7B Q1_0 | 27.6 | 35.9 | thermal variance |
| 1.7B Q2_0 | 23.2 | 23.5 | ~0% |
| 4B Q1_0 | 18.9 | 22.3 | thermal variance |
| 4B Q2_0 | 12.6 | 11.8 | +7% |
| 8B Q1_0 | 12.7 | 11.1 | +14% |
| 8B Q2_0 | 7.4 | 6.0 | +23% |

Large models benefit most: 8B Q2_0 +23%, 8B Q1_0 +14%, 4B Q2_0 +7%. Small models within thermal variance.

---

## Next optimization targets (reprioritized)

### Target 1: Multi-row matmul dispatch
Process multiple weight rows in a single dot call to amortize quantize_act() overhead and keep activation data hotter in L1. Currently each row calls `dot_fn` independently via function pointer. Activation data (Q8_0) is re-read from L1 for every weight row. With batching, process B rows at once with a single Q8_0 quantize and multiple dot calls from the same activation data.

### Target 2: Investigate lm_head quantization format
The lm_head tensor may use a different quantization format than the model weights. If it's Q8_0 or higher precision, we could potentially optimize its matmul separately. Check the GGUF tensor metadata. Currently the matmul dispatch selects the kernel based on the tensor's dtype.

### Target 3: Process multiple Q8_0 sub-blocks in VNNI kernel
The current VNNI kernel processes 1 sub-block (32 elements) per iteration. Using 512-bit AVX-512 would process 2 sub-blocks (64 elements). On Zen 4 (double-pumped 256-bit FPU), main benefit is reduced instruction count and fewer scale conversions.

### Target 4: Eliminate Q2_0 LUT loads via pre-expansion (revisit for 1.7B Q2_0 only)
Earlier attempt failed for Q1_0 (7.2× memory, 3× regression). For Q2_0, the expansion ratio is only 3.8× (34→128 bytes/block). The 1.7B Q2_0 model (554 MB → 2085 MB expanded) might see compute-bound behavior where LUT elimination helps. Risk: memory traffic increase overwhelms compute savings.

---

## Key files
- `crates/hearth-llm/src/model/mod.rs` — `forward()` lines 551-558, 608-618, 635-644 (changed `None` → `Some(&sc.scratch_q8[..])`)
- `crates/hearth-llm/src/model/scratch.rs` — `ForwardScratch::scratch_q8` was already defined but unused
- `crates/hearth-llm/src/model/matmul.rs` — `quantize_act()` is now called by fewer matmuls

## Key ref files (Prism fork)
- `ggml/src/ggml-cpu/quants.c:177` — `ggml_vec_dot_q2_0_q8_0_generic` (purely scalar, no SIMD path for Q2_0)
- `ggml/src/ggml-common.h:187-192` — `block_q2_0` struct (128 elements, 34 bytes)
