# GPU Acceleration — Session Start Prompt

## Context

You are continuing the Hearth project — a Rust LLM inference engine. All CPU optimizations are complete (50 tok/s decode, 13ms/tok prefill on 1.7B Q1_0 — ~5× faster than the reference llama.cpp-prism). Now we're adding GPU acceleration via wgpu/WGSL for the Radeon 780M iGPU (shared DDR5 memory, 12 CUs RDNA3).

## Key Files to Read First

1. `performance_ops.md` — Full GPU plan, break-even analysis, implementation phases
2. `RESUME.md` — Current state and phase plan
3. `crates/hearth-compute/src/lib.rs` — The GPU stubs to replace (197 lines, all `()` typed)
4. `crates/hearth-llm/src/model/gpu.rs` — Weight upload logic
5. `crates/hearth-llm/src/model/mod.rs` — Search for `forward_gpu` to see how GPU is wired
6. `crates/hearth-llm/src/model/matmul.rs` — GPU dispatch paths in matmul()

## Architecture Decision

- **wgpu** (WebGPU Rust implementation, translates WGSL → Vulkan on Windows)
- **NOT ROCm** (Radeon 780M `gfx1103` not supported)
- **NOT DirectML** (maintenance mode, no custom Q1_0/Q2_0 Q8_0 formats)
- Reference project: `wgpu-llm` (github.com/Beledarian/wgpu-llm) — 12 WGSL shaders for Llama ops

## Phase 0 Task: wgpu Infrastructure

**Goal:** Replace `hearth-compute` stubs with real wgpu device/buffer/dispatch code. Verify with a trivial compute shader.

### Step-by-step:

1. **Add dependencies** to `crates/hearth-compute/Cargo.toml`:
   - `wgpu = "24"` (latest stable as of June 2026)
   - `pollster = "0.4"` (block_on for async wgpu init)
   - `bytemuck = { workspace = true }` (cast buffers)
   - `half = { workspace = true }` (f16 for GPU)

2. **Create `crates/hearth-compute/src/device.rs`:**
   ```rust
   use wgpu::{Adapter, Device, Queue, Instance};
   
   pub(crate) struct GpuDevice {
       pub instance: Instance,
       pub adapter: Adapter,
       pub device: Device,
       pub queue: Queue,
   }
   
   impl GpuDevice {
       pub async fn new() -> Option<Self> {
           // Create instance
           // Request adapter (Vulkan backend)
           // Request device
           // Return GpuDevice
       }
   }
   ```

3. **Create `crates/hearth-compute/src/buffers.rs`:**
   ```rust
   use wgpu::{Buffer, BufferDescriptor, BufferUsages};
   
   pub(crate) fn create_storage_buffer(device: &Device, size: u64, label: &str) -> Buffer {
       // Create wgpu buffer with STORAGE | COPY_DST | COPY_SRC usage
   }
   
   pub(crate) fn upload_to_buffer(device: &Device, queue: &Queue, data: &[u8], label: &str) -> Buffer {
       // Create buffer, write data via queue.write_buffer()
   }
   
   pub(crate) fn download_from_buffer(device: &Device, queue: &Queue, buffer: &Buffer, size: u64) -> Vec<u8> {
       // Create staging buffer, copy from storage buffer, map and read
   }
   ```

4. **Rewrite `crates/hearth-compute/src/lib.rs`:**
   - Replace all `()` typed buffers with `wgpu::Buffer`
   - Keep the same public API signatures but implement real GPU operations
   - For now, all methods can return `None`/`false`/do nothing EXCEPT:
     - `GpuCompute::new()` should initialize wgpu
     - `upload_f32()`, `upload_bytes()` should create GPU buffers
     - Add a test compute shader

5. **Create `crates/hearth-compute/src/shaders/simplest.wgsl`:**
   ```wgsl
   @group(0) @binding(0) var<storage, read> input: array<f32>;
   @group(0) @binding(1) var<storage, read_write> output: array<f32>;
   
   @compute @workgroup_size(64)
   fn main(@builtin(global_invocation_id) id: vec3<u32>) {
       let i = id.x;
       output[i] = input[i] + 1.0;
   }
   ```

6. **Create `crates/hearth-compute/src/shaders/mod.rs`:**
   ```rust
   pub const SIMPLEST_WGSL: &str = include_str!("simplest.wgsl");
   // Future shaders added here
   ```

7. **Create `crates/hearth-compute/src/pipeline.rs`:**
   ```rust
   use wgpu::{ComputePipeline, ComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource};
   
   pub(crate) fn create_compute_pipeline(
       device: &Device,
       shader_source: &str,
       label: &str,
   ) -> ComputePipeline {
       let shader = device.create_shader_module(ShaderModuleDescriptor {
           label: Some(label),
           source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
       });
       device.create_compute_pipeline(&ComputePipelineDescriptor {
           label: Some(label),
           layout: None, // auto layout
           module: &shader,
           entry_point: "main",
           compilation_options: Default::default(),
       })
   }
   ```

8. **Implement dispatch helper in `lib.rs`:**
   ```rust
   pub fn dispatch_compute(&self, pipeline: &ComputePipeline, bind_group: &BindGroup, x: u32, y: u32, z: u32) {
       let mut encoder = self.device.create_command_encoder(&Default::default());
       {
           let mut cpass = encoder.begin_compute_pass(&Default::default());
           cpass.set_pipeline(pipeline);
           cpass.set_bind_group(0, bind_group, &[]);
           cpass.dispatch_workgroups(x, y, z);
       }
       self.queue.submit(Some(encoder.finish()));
   }
   ```

9. **Wire into `hearth-llm`:**
   - After GpuCompute returns a real object (not `None`), the existing `forward_gpu()` should naturally flow to GPU execution
   - Test with `hearth-chat-cli.exe` on a small model to verify no crashes

### Verification

```powershell
cargo build --release
# Should compile with wgpu dependencies
cargo test -p hearth-compute
# Should pass (add a simple test that initializes wgpu and runs the simplest shader)
```

### Key Constraints
- Must compile and run on Windows 11 with AMD drivers (Vulkan)
- Must not break the CPU-only path (GPU is optional, falls back to CPU)
- Buffer handling must support unified memory (use `Device::buffer_get_mapped_range` or staging buffers)
- All wgpu init is async — use `pollster::block_on(GpuCompute::new())`

## FAQ

**Q: Why wgpu and not raw Vulkan?**  
A: wgpu is pure Rust, handles all the boilerplate (surface, swapchain, synchronization), and is well-maintained. Raw Vulkan would require thousands of lines of unsafe C-like code.

**Q: Will this work on NVIDIA too?**  
A: Yes! wgpu translates WGSL to Vulkan on Windows/Linux, and Metal on macOS. Same WGSL shaders work everywhere.

**Q: What about FP16 support?**  
A: RDNA3 has native FP16 with 2× throughput. WGSL supports `f16` type via the `f16` feature. Use compile-time WGSL string injection to select f32 vs f16.

**Q: How do I handle Q1_0/Q2_0 dequant on GPU?**  
A: Load block bytes → extract sign bits (Q1_0) or LUT indices (Q2_0) → convert to f16 → multiply by block scale → accumulate in f32. This is Phase 1 work; for Phase 0 just get wgpu initialized and verify with a trivial shader.
