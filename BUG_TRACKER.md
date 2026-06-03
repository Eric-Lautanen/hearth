# Hearth Performance Tracker — CPU Final State → GPU Planning

## System
AMD Ryzen 7 8840HS (8C/16T), Radeon 780M iGPU (12 CU RDNA3), 16 GB DDR5, Windows 11. Hearth: 8 workers. Ref: llama.cpp-prism.

## Current Benchmark (2026-06-03, post wgpu Phase 0, proper warmup)

| Model | Format | Tok/s | Notes |
|-------|--------|-------|-------|
| 1.7B | Q1_0 | 47.7 | Full decode warmup required (built-in 32ms warmup insufficient) |
| 1.7B | Q2_0 | 28.7 | |
| 4B | Q1_0 | 23.0 | |
| 4B | Q2_0 | 13.2 | |
| 8B | Q1_0 | 9.9 | |
| 8B | Q2_0 | 6.0 | |

**Warmup note:** the built-in `[warmup] CPU clock ramp: 32ms` runs just one throwaway forward pass — insufficient for this CPU to reach full boost. A full 50-token decode run as warmup (first run discarded, second measured) reliably gives 47-48 tok/s. The cold single-run tok/s is ~31 tok/s due to Windows frequency scaling ramp-up across the first 30-40 tokens.

## Final CPU Benchmark (1.7B Q1_0, 50-tok decode, pre-wgpu baseline, peak)

| Metric | Hearth | Ref (llama.cpp-prism) | Speedup |
|--------|--------|----------------------|---------|
| Decode tok/s (50tok) | 50.0 tok/s | 35 tok/s | **1.43×** |
| Decode tok/s (Q2_0) | 29.0 tok/s | 5.8 tok/s | **5.0×** |
| Prefill (100t) | 12.5 ms/tok | 19 ms/tok | **1.5×** |
| Prefill (800t) | 13.1 ms/tok | 19 ms/tok | **1.5×** |

## CPU Optimization History (All Sessions)

| Session | Change | Decode Gain | Prefill Gain |
|---------|--------|-------------|--------------|
| S1-9 | Custom thread pool, SIMD kernels, LUT optimization | 4.0→45 tok/s (+1050%) | — |
| S10-13 | Q2_0 VNNI, shuffle kernel, scratch reuse | 45→46 tok/s (+2%) | — |
| S14-15 | AVX-512 VNNI (neutral on Zen 4), f16 LUT | neutral | — |
| S16 | Batch matmul dispatch (par_dot_rows_batched) | neutral | Prefill enabled |
| S17 | Pre-quantize activation once for Q/K/V | neutral | ~5% |
| S18 | Head norm reuse, inv_sqrt_hd hoisting | neutral | ~2% |
| S19 | Matmul confirmed saturated — no decode gains possible | none | none |
| S20 | Parallel batch quantize + chunk-outer attention | neutral | **~15%** |
| S21 | Q8_0 KV cache (REVERTED — net-negative) | — | — |
| S22 | Fused RMS Norm + Quantize | neutral | <1% |
| S23 | Fused SiLU+Mul+Quantize | neutral | <1% |
| S24 | Head-parallel attention dispatch + SFC loop interchange | neutral | **~7%** |
| S24b | Chunk-outer weighted sum for decode attention | **~2×** at 800tctx | — |

### Cumulative: ~17% prefill improvement, decode saturated at 50 tok/s (1.7B Q1_0)

## What Didn't Work (Archive)

- Q8_0 KV cache: dequant compute > bandwidth savings at all context lengths
- 16 threads: SMT contention kills SIMD throughput (10× regression)
- VNNI kernels: Zen 4 double-pumping makes 512-bit neutral vs 256-bit
- Weight pre-expansion: 7.2× memory traffic increase swamped compute savings
- QKV fusion: <3% gain, not worth complexity
- Prefetch hints: hardware prefetcher already optimal on Zen 4
- Tiled attention: compute-bound system, chunk-outer already optimal

## GPU Architecture

See `performance_ops.md` for full GPU planning.

### Phase Plan

| Phase | What | Depends On | Est. Days | Status |
|-------|------|------------|-----------|--------|
| 0 | wgpu infrastructure (device, buffers, dispatch) | Nothing | 3-5 | **DONE** |
| 1 | Q1_0/Q2_0 dequant+matmul WGSL kernel | Phase 0 | 5-7 | — |
| 2 | Full decode on GPU (all layer shaders) | Phase 1 | 5-7 | — |
| 3 | Prefill on GPU (batch matmul) | Phase 2 | 3-5 | — |
| 4 | Image generation GPU (hearth-diffusion) | Phase 1 | 5-10 | — |
| 5 | Optimization (fusion, tuning) | Phase 2/3 | Ongoing | — |

### Phase 0 Deliverables (2026-06-03)
- `hearth-compute/Cargo.toml` — wgpu 29, pollster 0.4 deps added
- `src/device.rs` — `GpuDevice` (wgpu Instance/Adapter/Device/Queue init, Vulkan backend)
- `src/buffers.rs` — `create_storage_buffer`, `upload_to_buffer`, `download_from_buffer`
- `src/pipeline.rs` — `create_compute_pipeline`, `dispatch_compute`
- `src/shaders/simplest.wgsl` — trivial `output[i] = input[i] + 1.0` compute shader
- `src/shaders/mod.rs` — shader include_str!() module
- `src/lib.rs` — `GpuCompute` struct with real `wgpu::Buffer` (re-exported as `GpuBuffer`), all methods still return None/false except: `new()`, `upload_f32()`, `upload_f16_packed()`, `upload_bytes()`, `create_storage_buffer()`, `readback_f32()`, `run_simplest_test()`
- `test_simplest_shader` — unit test initializes GPU, runs compute shader, verifies output
- All hearth-llm code compiles unchanged (type migration `()` → `GpuBuffer` was transparent)
- CPU path unchanged — no decode/prefill regression
- GPU detected: AMD Radeon 780M via Vulkan backend

### Key Technical Decisions

1. **wgpu + WGSL** over ROCm (780M unsupported) and DirectML (maintenance mode, no custom kernels)
2. **Unified memory** (shared DDR5) means zero-copy weight access — no PCIe transfers
3. **Reference project:** `wgpu-llm` (github.com/Beledarian/wgpu-llm) — 12 WGSL shaders for Llama
4. **Fused dequant+matmul** kernels required for Q1_0/Q2_0 weights (WGSL custom)
5. **Kernel fusion critical** — wgpu dispatch latency (20-50µs) makes per-op dispatch expensive
6. **FP16 matmul** for diffusion (FP16 throughput on RDNA3 is 2× FP32)
