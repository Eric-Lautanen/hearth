# Session 19: Prefill optimization research — reorienting decode→prefill

> **⚠️ CRITICAL RULE: NEVER touch decode.** ALL optimization work from this point forward targets ONLY the prefill path (`forward_batch`, `encode_text`, `matmul_batch`, `par_dot_rows_batched`, `attention_batch`, `norm_batch`). The single-token decode `forward()` function is frozen — changing it risks regressing all 6 models by 10-33%.

**Decode is frozen at peak performance.** All prior work (Sessions 1-19) optimized the single-token decode path. Matmul kernels are saturated on this Zen 4 8C/16T system. No further decode changes permitted.

**New focus: Prefill-only (prompt processing / TTFT).** Prefill is compute-bound (unlike decode's bandwidth-bound profile), so different optimization strategies apply. Batch-only functions may be modified freely. This session surveyed the 2025-2026 research landscape and identified concrete opportunities.

---

## Prefill architecture (current)

`forward_batch()` processes all seq_len prompt tokens through all layers:

1. Embed tokens (dequantize rows)
2. Per layer:
   - RMS norm hidden → residual
   - Quantize residual → batch_q8 (serial per-token)
   - matmul_batch Q (seq_len tokens × weight matrix)
   - matmul_batch K (reuses same batch_q8)
   - matmul_batch V
   - Q/K head norm (serial over seq_len × heads)
   - RoPE (SIMD, already fast)
   - KV cache write (per head per position)
   - Attention (O(seq_len² × head_dim), f32 scores + softmax + weighted sum)
   - Quantize attn_out → batch_q8
   - matmul_batch attn_output
   - Residual add
   - RMS norm → residual
   - Quantize residual → batch_q8
   - matmul_batch gate + up
   - SiLU + multiply
   - Quantize ffn_tmp → batch_q8
   - matmul_batch ffn_down
   - Residual add
3. Output norm + lm_head

**Key files:**
- `crates/hearth-llm/src/model/mod.rs` — `forward_batch()` (lines ~749-1075), `encode_text()` (lines ~1077-1410)
- `crates/hearth-llm/src/model/matmul.rs` — `matmul_batch()` (line 498), uses `par_dot_rows_batched`
- `crates/hearth-llm/src/pool.rs` — `par_dot_rows_batched()` (line 189)
- `crates/hearth-llm/src/ops.rs` — `attention_batch()` (line 410), f32x8 SIMD
- `crates/hearth-llm/src/kvcache.rs` — `KVCache` with F32 storage

---

## Priority optimization order

### Session 20: Quick wins (low effort, measurable)

1. **Parallel batch quantization** (`pool.rs` + `mod.rs`)
   - Dispatch per-token Q8_0 quantize across thread pool workers
   - Each worker writes to known offset in pre-allocated `batch_q8`
   - Est: 2-4% TTFT reduction

2. **Attention inner loop accumulate** (`ops.rs` `attention_batch`)
   - Swap position-outer + chunk-inner to chunk-outer + position-inner
   - Eliminates per-position `to_array()` + `copy_from_slice` round-trip
   - Est: 2-5% of attention time (scales with seq_len)

### Session 21: Q8_0 KV Cache

3. **Switch KV cache from F32 to Q8_0** (`kvcache.rs` + `ops.rs` + `mod.rs`)
   - `write_kv`: quantize f32 to Q8_0 blocks (f16 scale + 32× i8) on write
   - `attention`/`attention_batch`: dequantize Q8_0 blocks to f32 on read
   - 3.76× memory compression — KV cache fits in L2 at seq_len=2048 for 1.7B
   - turboquant_plus data: 7.4% prefill improvement at long context
   - Must benchmark break-even point (expected ~seq_len=128)
   - **Requires:** `KVCache` enum for storage type (already has `is_q8_0()` check)

### Session 22+: CPU Tiled Attention

4. **Flash-style tiled attention** (`ops.rs` new `attention_tiled`)
   - Process KV in fixed tiles (32-64 positions) with online softmax
   - Keep KV tile in L1/L2, no scratch buffer write
   - Combine with Q8_0 KV for best tile utilization
   - Est: 10-30% attention time at seq_len ≥ 4096

---

## Research references (2025-2026)

- **Sandwich** (Zhao et al., 2025): CPU-specific prefill/decode separation, custom GEMM micro-kernels, 2.01× throughput. [arxiv:2507.18454]
- **Litespark-Inference** (Dade et al., 2026): Custom SIMD for ternary weights, VNNI on Zen4, 9.2× TTFT. [arxiv:2605.06485]
- **SpecPrefill** (Liu et al., 2025): Skip unimportant prompt tokens via lightweight importance model, up to 7× TTFT. [arxiv:2502.02789]
- **FlashAttention** (Dao et al., 2022): Tiled online-softmax attention — architecture-agnostic algorithm.
- **turboquant_plus** (TheTom, 2025): Q8_0 KV cache gives 7.4% prefill improvement on Apple Silicon.

## Key insight

Prefill optimization strategies DIFFER from decode:
- **Decode** is bandwidth-bound → optimize weight memory footprint, reduce DRAM reads
- **Prefill** is compute-bound → optimize matmul utilization, parallelize non-matmul ops, reduce attention O(seq_len²)
- Quantization helps prefill differently: KV cache quantization (reducing attention memory) > weight quantization (matmul already compute-bound)
- Parallelism helps prefill more: batch quantize, batch norm, even parallel attention across heads
