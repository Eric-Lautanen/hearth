# Hearth GPU Acceleration — Phase 1: Dequant+Matmul WGSL Kernels

> **Phase 0 complete (2026-06-03).** wgpu 29 infrastructure working on Radeon 780M Vulkan. All stubs replaced with real buffer ops. Trivial compute shader verified output = input + 1.0.

## Phase 0 Summary

| Item | Status |
|------|--------|
| wgpu 29 + pollster 0.4 deps | Done |
| `GpuDevice` (Instance/Adapter/Device/Queue) | Done |
| Buffer create/upload/download | Done |
| Compute pipeline create + dispatch | Done |
| `simplest.wgsl` test shader | Done (verified) |
| `GpuBuffer` type alias (re-exported `wgpu::Buffer`) | Done |
| All CPU paths unchanged, no regression | Verified (35.9 tok/s) |
| `hearth-llm` compiles unchanged | Verified |

## Phase 1: Q1_0/Q2_0 Dequant + Matmul Shader (NEXT SESSION)

### Files to Create/Modify

1. **`crates/hearth-compute/src/shaders/q1_0_matmul.wgsl`**
   - Input: Q1_0 weight blocks (18 bytes each: 2-byte scale f16 + 16-byte sign bits for 128 elements)
   - Input: Q8_0 activation blocks (34 bytes each: 2-byte scale f16 + 32-byte quantized values for 32 elements)
   - Algorithm:
     ```
     For each weight block (128 elements):
       1. Load 18 bytes: scale_f16 (2B) + sign_bits (16B = 128 bits)
       2. For each of the 128 elements:
          - Extract sign bit → {-1, +1}
          - Multiply by block scale → f16 value
       3. For each activation block (32 elements):
          - Load 34 bytes: a_scale + quantized bytes
          - Dequant to f16: a_val = a_scale * quantized_byte
       4. Dot product: sum(sign * w_scale * a_scale * a_quant)
     ```
   - Workgroup size: 64 or 128 (RDNA3 wave32 native, but can emulate wave64)
   - Output: f32 accumulation per output row element (or f32 per workgroup thread with atomic add)

2. **`crates/hearth-compute/src/shaders/q2_0_matmul.wgsl`**
   - Similar but Q2_0 blocks are 34 bytes: scale_f16 (2B) + 32-byte LUT indices (2 bits per element → 128 elements)
   - LUT: {-1, 0, 1, 2} or {0, 1, 2, 3} — need to check reference implementation
   - Dequant: extract 2-bit LUT index → map to LUT value → multiply by scale

3. **`crates/hearth-compute/src/shaders/helpers.wgsl`**
   - Shared utilities: bit extraction, f16→f32 conversion, coalesced loads

4. **`crates/hearth-compute/src/matmul.rs`** (new)
   - `dequant_matmul_fused()` implementation:
     - Determine dispatch size: ceil(n_rows * n_cols / workgroup_size)
     - Create bind group with weight buffer, activation buffer, output buffer
     - Dispatch compute pipeline
   - Support both single-row (decode, batch=1) and batched (prefill, batch>1)

5. **`crates/hearth-compute/src/lib.rs`** (modify)
   - Wire `dequant_matmul_fused()` to real implementation
   - Wire `has_dequant()` to return true for Q1_0, Q1_0G128, Q2_0
   - Add pipeline cache for Q1_0 and Q2_0 shaders

### Key Technical Decisions for Phase 1

- **Tile size**: 64×64 or 128×128? RDNA3 780M has 12 CUs × 64 threads/CU = 768 threads. 64×64 tile needs 4096 threads — too many. Start with 16×16 or 32×8.
- **FP16 accumulation**: RDNA3 FP16 is 2× throughput of FP32. Use f16 for intermediate dot products, f32 for final accumulation.
- **Buffer layout**: Q1_0 weights already in GGUF format (18-byte blocks per 128 elements). No conversion needed — shader reads raw bytes.
- **Q8_0 activation buffer**: Already pre-quantized on CPU side. Shader reads Q8_0 blocks directly.
- **Readback**: GPU output is f32 buffer. Copy to staging, map, read on CPU. Already implemented in `download_from_buffer()`.

### Reference Files
- `C:\Users\ericl\Documents\hearth\crates\hearth-quant\src\q1_0g128.rs` — CPU Q1_0 dot product implementation (reference for format)
- `C:\Users\ericl\Documents\hearth\crates\hearth-quant\src\q2_0.rs` — CPU Q2_0 dot product implementation
- `C:\Users\ericl\.cargo\registry\src\...\wgpu-29.0.3\` — wgpu source for API reference
- `wgpu-llm` (github.com/Beledarian/wgpu-llm) — reference WGSL shaders for Llama ops

### Verification Steps
```powershell
cargo check -p hearth-compute
cargo clippy -p hearth-compute -- -D warnings
cargo fmt --check
cargo test -- -p hearth-compute --test-threads=1
cargo build --release
# Run simplest test to verify no regression
& ".\target\release\hearth-chat-cli.exe" "$env:USERPROFILE\AppData\Roaming\hearth\models\Bonsai-1.7B-Q1_0.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw
```

### Notes for Next Session
- `wgpu::PollType::wait_indefinitely()` replaces old `Maintain::Wait`
- `wgpu::InstanceDescriptor::new_without_display_handle()` for headless compute
- `DeviceDescriptor` has `experimental_features` and `trace` fields in wgpu 29
- `request_device()` takes only 1 arg (no trace path) in wgpu 29
- `ComputePipelineDescriptor` has `cache` field in wgpu 29
- `ExperimentalFeatures::disabled()` not `empty()`
