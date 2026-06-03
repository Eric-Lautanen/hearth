# Hearth GPU Performance Ops — Planning & Optimization Guide

## System: Ryzen 7 8840HS + Radeon 780M iGPU

| Component | Spec |
|-----------|------|
| CPU | 8C/16T Zen 4, up to 5.1 GHz |
| GPU | Radeon 780M, 12 CUs, RDNA3, up to 2.7 GHz |
| Memory | 16GB DDR5 (shared, ~40-50 GB/s) |
| GPU TFLOPs | ~2.0 FP32, ~4.0 FP16 |
| GPU-CPU BW | Unified memory, no PCIe (same DDR5) |
| GPU API | Vulkan 1.3, DX12 Ultimate |
| ROCm | NOT supported on 780M (`gfx1103`) |

## GPU vs CPU Characteristics

| Factor | CPU (8C Zen 4) | GPU (12 CU RDNA3) |
|--------|----------------|-------------------|
| Peak FP32 | ~1.6 TFLOPS | ~2.0 TFLOPS |
| Peak FP16 | ~1.6 TFLOPS (no native) | ~4.0 TFLOPS |
| Memory BW | ~40 GB/s (shared) | ~40 GB/s (shared) |
| Launch latency | ~1µs | ~10-50µs (wgpu dispatch) |
| Best at | Small matrices, branching | Large matrices, throughput |

## When GPU Helps

| Operation | GPU Advantage | Why |
|-----------|--------------|-----|
| Large matmul (rows > 1024) | 2-5× | GPU compute throughput |
| Batch matmul (prefill) | 3-8× | Multiple tokens × weights parallelized |
| Image gen (Flux transformer) | 5-20× | Massive matmuls, FP16 throughput |
| Fused dequant+matmul | 2-3× | GPU can fuse where CPU must iterate |

## When GPU Doesn't Help

| Operation | CPU Advantage | Why |
|-----------|--------------|-----|
| Small matmul (rows < 256) | 1-2× | GPU launch overhead > compute time |
| Attention (decode, seq < 512) | 1-2× | Small KV cache, CPU SIMD efficient |
| Element-wise ops (norm, SiLU) | 2-5× | CPU SIMD, no launch overhead |
| Single-token decode (batch=1) | Tie | Both limited by memory bandwidth |

## Break-even Analysis

For the iGPU to beat CPU on a matmul:

```
GPU_time = launch_overhead + compute_time + readback_time
CPU_time = compute_time (already optimal)

launch_overhead ≈ 20µs (wgpu dispatch + synchronization)
compute_time = ops / TFLOPS
```

For a 6144×2048 matmul (FFN gate, 1.7B):
- Ops: 6144 × 2048 × 2 (mul+add) = 25M FMAs
- CPU: 25M / 1.6 TFLOPS = 15.6µs (but actually ~100µs due to quantized format overhead)
- GPU: 20µs + 25M/2.0 TFLOPS = 20µs + 12.5µs = 32.5µs

GPU should win for this size. For smaller matrices (QKV at 2048×2048), the margin narrows.

For prefill batch (seq_len=10, 6144×2048):
- CPU does 10 separate matmuls (10 × 100µs = 1ms)
- GPU does 1 batch matmul (20µs + 10 × 12.5µs = 145µs)
- GPU wins by ~7× for batch matmul

## wgpu-llm Benchmarks (Reference)

From wgpu-llm (TinyLlama 1.1B on Snapdragon Adreno iGPU):
- f16 decode: 25.5 tok/s
- INT8 decode: 32.8 tok/s
- CPU-only (same model): ~20 tok/s

This suggests ~1.3-1.6× GPU advantage for decode on a less powerful iGPU. Our 780M is more powerful, so we might see 2-3× on decode and 5-10× on prefill matmuls.

## Image Generation GPU Potential

Flux transformer at 1024×1024 (seq_img=4096, d=3072):
- Self-attention: O(seq² × d) = 4096² × 3072 = 51B FMAs per step
- MLP: 4 × seq × d × 4d = 4 × 4096 × 3072 × 12288 = 618B FMAs per step
- 50 steps: 33 trillion FMAs
- CPU: 33T / 1.6 TFLOPS = 20,000 seconds (5.5 hours) — currently unusable
- GPU: 33T / 4 TFLOPS FP16 = 8,250 seconds BUT actual GPU matmul is bandwidth-limited

With proper GPU matmul: ~200-500 seconds (3-8 minutes) — usable but slow.
With GPU attention + FP16: potentially ~30-60 seconds.

## wgpu Dispatch Overhead

From "Characterizing WebGPU Dispatch Overhead" (2604.02344, Feb 2026):
- Per-dispatch latency: 11-50µs depending on GPU and buffer complexity
- **Kernel fusion improves throughput 53%** on Vulkan (reduces dispatch count)
- For LLM inference with 50+ dispatches per forward pass, fusion is critical

## Implementation Strategy

### Phase 0: Foundation (3-5 days)
```
hearth-compute/
├── src/
│   ├── lib.rs          # GpuCompute struct, public API
│   ├── device.rs        # wgpu adapter/device/queue init
│   ├── buffers.rs       # Buffer allocation, upload/download
│   └── shaders/         # WGSL source files
│       └── simplest.wgsl # Test shader
```

### Phase 1: Matmul Kernel (5-7 days)
```
hearth-compute/src/shaders/
├── q1_0_matmul.wgsl     # Q1_0 dequant + matmul fused
├── q2_0_matmul.wgsl     # Q2_0 dequant + matmul fused
├── f16_matmul.wgsl      # Standard f16 matmul (for non-quantized)
└── helpers.wgsl          # Shared WGSL utilities

hearth-compute/src/
├── matmul.rs             # Matmul dispatch, batch support
└── dequant.rs            # On-host dequant fallback
```

### Phase 2: LLM Inference (5-7 days)
```
hearth-compute/src/shaders/
├── rms_norm.wgsl
├── rope.wgsl
├── silu.wgsl
├── softmax.wgsl
├── attention.wgsl
└── kv_cache.wgsl

hearth-compute/src/
├── pipeline.rs           # Build pipeline from shaders
└── forward.rs            # Full forward pass orchestration
```

### Phase 3-5: Prefill, Diffusion, Optimization
Continuing expansion of shaders and pipeline orchestration.

## Key Metrics to Track

| Metric | Current CPU | Target GPU | Measurement |
|--------|-------------|------------|-------------|
| Decode tok/s (1.7B Q1_0) | 50 | 100-150 | `hearth-chat-cli` |
| Prefill ms/tok (100t) | 12.5 | 3-5 | `[prefill]` output |
| Prefill ms/tok (800t) | 13.1 | 4-6 | `[prefill]` output |
| Image gen (1024², 50 steps) | ~hours | <5 min | `hearth-diffuse` |
| Peak GPU utilization | 0% | >80% | GPU metrics |
| wgpu dispatch overhead | N/A | <10% of total | Profiling |
