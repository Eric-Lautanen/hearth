mod buffers;
mod device;
mod matmul;
mod pipeline;
mod shaders;

use std::collections::HashMap;

use hearth_gguf::GgmlDType;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferDescriptor, BufferUsages,
    ComputePipeline,
};

pub use wgpu::Buffer as GpuBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4K,
    Q1_0,
    Q1_0G128,
    Q2_0,
}

impl QuantKind {
    pub fn from_gguf_dtype(dtype: GgmlDType) -> Option<Self> {
        match dtype {
            GgmlDType::F32 => Some(QuantKind::F32),
            GgmlDType::F16 => Some(QuantKind::F16),
            GgmlDType::Q4_0 => Some(QuantKind::Q4_0),
            GgmlDType::Q8_0 => Some(QuantKind::Q8_0),
            GgmlDType::Q4_K => Some(QuantKind::Q4K),
            GgmlDType::Q1_0 => Some(QuantKind::Q1_0),
            GgmlDType::Q2_0 => Some(QuantKind::Q2_0),
            GgmlDType::Q1_0_G128 => Some(QuantKind::Q1_0G128),
            _ => None,
        }
    }
}

/// Cached pipelines for the general-purpose ops (norm, rope, attention, etc.).
pub(crate) struct GeneralPipelines {
    pub rms_norm: ComputePipeline,
    pub rms_norm_batch: ComputePipeline,
    pub head_rms_norm: ComputePipeline,
    pub head_rms_norm_batch: ComputePipeline,
    pub rope: ComputePipeline,
    pub rope_batch: ComputePipeline,
    pub kv_cache: ComputePipeline,
    pub kv_cache_batch_write: ComputePipeline,
    pub flash_attention: ComputePipeline,
    pub flash_attention_prefill: ComputePipeline,
    pub flash_attention_bidirectional: ComputePipeline,
    pub silu_mul: ComputePipeline,
    pub add_inplace: ComputePipeline,
    pub conv2d_3x3: ComputePipeline,
    pub group_norm: ComputePipeline,
}

impl GeneralPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let create = |src: &str, label: &str| pipeline::create_compute_pipeline(device, src, label);
        Self {
            rms_norm: create(shaders::RMS_NORM_WGSL, "rms_norm"),
            rms_norm_batch: create(shaders::RMS_NORM_BATCH_WGSL, "rms_norm_batch"),
            head_rms_norm: create(shaders::HEAD_RMS_NORM_WGSL, "head_rms_norm"),
            head_rms_norm_batch: create(shaders::HEAD_RMS_NORM_BATCH_WGSL, "head_rms_norm_batch"),
            rope: create(shaders::ROPE_WGSL, "rope"),
            rope_batch: create(shaders::ROPE_BATCH_WGSL, "rope_batch"),
            kv_cache: create(shaders::KV_CACHE_WGSL, "kv_cache"),
            kv_cache_batch_write: create(
                shaders::KV_CACHE_BATCH_WRITE_WGSL,
                "kv_cache_batch_write",
            ),
            flash_attention: create(shaders::FLASH_ATTENTION_WGSL, "flash_attention"),
            flash_attention_prefill: create(
                shaders::FLASH_ATTENTION_PREFILL_WGSL,
                "flash_attention_prefill",
            ),
            flash_attention_bidirectional: create(
                shaders::FLASH_ATTENTION_BIDIRECTIONAL_WGSL,
                "flash_attention_bidirectional",
            ),
            silu_mul: create(shaders::SILU_MUL_WGSL, "silu_mul"),
            add_inplace: create(shaders::ADD_INPLACE_WGSL, "add_inplace"),
            conv2d_3x3: create(shaders::CONV2D_3X3_WGSL, "conv2d_3x3"),
            group_norm: create(shaders::GROUP_NORM_WGSL, "group_norm"),
        }
    }
}

pub struct GpuCompute {
    pub pool: std::sync::Mutex<HashMap<String, GpuBuffer>>,
    pub norm_buffers: HashMap<String, GpuBuffer>,
    pub kv_cache: Vec<GpuBuffer>,
    pub cache_max_seq: u32,
    device: device::GpuDevice,
    simplest_pipeline: Option<wgpu::ComputePipeline>,
    matmul_pipelines: matmul::DequantMatmulPipelines,
    general_pipelines: GeneralPipelines,
    fused_gate_up_pipeline: Option<wgpu::ComputePipeline>,
    fused_gate_up_batch_pipeline: Option<wgpu::ComputePipeline>,
    fused_qkv_pipeline: Option<wgpu::ComputePipeline>,
    fused_qkv_batch_pipeline: Option<wgpu::ComputePipeline>,
    argmax_pipeline: wgpu::ComputePipeline,
    /// When Some, all dispatches and copy operations are recorded on this encoder
    /// and submitted together in end_batch(). When None, each operation creates its
    /// own encoder and submits immediately.
    current_encoder: std::sync::Mutex<Option<wgpu::CommandEncoder>>,
    /// Reusable staging buffer for readback. Created on first use, resized as needed.
    /// Inner: (buffer, current_size_in_bytes)
    readback_staging: std::sync::Mutex<Option<(wgpu::Buffer, u64)>>,
}

#[allow(unused_variables, clippy::too_many_arguments)]
impl GpuCompute {
    pub async fn new() -> Option<Self> {
        let dev = device::GpuDevice::new().await?;

        let simplest_pipeline = Some(pipeline::create_compute_pipeline(
            &dev.device,
            shaders::SIMPLEST_WGSL,
            "simplest",
        ));

        let matmul_pipelines = matmul::DequantMatmulPipelines::new(&dev.device);
        let general_pipelines = GeneralPipelines::new(&dev.device);

        eprintln!("[gpu] Q1_0/Q2_0 dequant+matmul pipelines created");
        eprintln!("[gpu] General op pipelines created (norm, rope, attention, etc.)");

        let fused_gate_up_pipeline = pipeline::create_compute_pipeline(
            &dev.device,
            shaders::FUSED_GATE_UP_WGSL,
            "fused_gate_up",
        )
        .into();
        let fused_gate_up_batch_pipeline = pipeline::create_compute_pipeline(
            &dev.device,
            shaders::FUSED_GATE_UP_BATCH_WGSL,
            "fused_gate_up_batch",
        )
        .into();
        let fused_qkv_pipeline =
            pipeline::create_compute_pipeline(&dev.device, shaders::FUSED_QKV_WGSL, "fused_qkv")
                .into();
        let fused_qkv_batch_pipeline = pipeline::create_compute_pipeline(
            &dev.device,
            shaders::FUSED_QKV_BATCH_WGSL,
            "fused_qkv_batch",
        )
        .into();

        let argmax_pipeline =
            pipeline::create_compute_pipeline(&dev.device, shaders::ARGMAX_WGSL, "argmax");

        Some(Self {
            pool: std::sync::Mutex::new(HashMap::new()),
            norm_buffers: HashMap::new(),
            kv_cache: Vec::new(),
            cache_max_seq: 0,
            device: dev,
            simplest_pipeline,
            matmul_pipelines,
            general_pipelines,
            fused_gate_up_pipeline,
            fused_gate_up_batch_pipeline,
            fused_qkv_pipeline,
            fused_qkv_batch_pipeline,
            argmax_pipeline,
            current_encoder: std::sync::Mutex::new(None),
            readback_staging: std::sync::Mutex::new(None),
        })
    }

    pub fn has_dequant(&self, kind: &QuantKind) -> bool {
        matches!(
            kind,
            QuantKind::Q1_0 | QuantKind::Q1_0G128 | QuantKind::Q2_0
        )
    }

    pub fn has_matmul_f16(&self) -> bool {
        false
    }

    pub fn warmup(&self) {
        let one = self.upload_f32(&[1.0f32], "warmup");
        let two = self.upload_f32(&[2.0f32], "warmup_act");
        let _ = self.dequant_matmul_fused(&one, &QuantKind::Q1_0, &two, 1, 1, 1);
    }

    /// Begin a batch: all subsequent dispatches are recorded into one command encoder
    /// and submitted together when end_batch() is called. This reduces CPU→GPU submit
    /// overhead by replacing N submits with a single submit of N dispatches.
    pub fn begin_batch(&self) {
        let encoder = self
            .device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        *self.current_encoder.lock().unwrap() = Some(encoder);
    }

    /// End a batch: submit all recorded dispatches and poll for completion.
    pub fn end_batch(&self) {
        let encoder = self.current_encoder.lock().unwrap().take();
        if let Some(enc) = encoder {
            self.device.queue.submit(Some(enc.finish()));
        }
        self.device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();
    }

    pub fn create_storage_buffer(&self, size: u64, label: &str) -> GpuBuffer {
        buffers::create_storage_buffer(&self.device.device, size, label)
    }

    pub fn upload_f32(&self, data: &[f32], label: &str) -> GpuBuffer {
        let bytes = bytemuck::cast_slice(data);
        buffers::upload_to_buffer(&self.device.device, &self.device.queue, bytes, label)
    }

    pub fn upload_f16_packed(&self, data: &[u8], label: &str) -> GpuBuffer {
        buffers::upload_to_buffer(&self.device.device, &self.device.queue, data, label)
    }

    pub fn upload_bytes(&self, data: &[u8], label: &str) -> GpuBuffer {
        buffers::upload_to_buffer(&self.device.device, &self.device.queue, data, label)
    }

    pub fn readback_f32(&self, buf: &GpuBuffer, n: usize) -> Vec<f32> {
        let bytes = buffers::download_from_buffer(
            &self.device.device,
            &self.device.queue,
            buf,
            (n * 4) as u64,
        );
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Ensure the reusable staging buffer exists and is at least `min_size` bytes.
    /// Returns a reference to the staging buffer.
    pub fn ensure_readback_staging(&self, min_size: u64) -> wgpu::Buffer {
        let mut guard = self.readback_staging.lock().unwrap();
        if let Some((ref buf, size)) = *guard {
            if size >= min_size {
                return buf.clone();
            }
        }
        // Create new (or larger) staging buffer
        let new_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size: min_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *guard = Some((new_buf.clone(), min_size));
        new_buf
    }

    /// Record a copy from `src` to the staging buffer on the batch encoder.
    /// Must be called between begin_batch() and end_batch().
    /// After end_batch(), call finish_readback_f32() to retrieve the data.
    pub fn copy_to_readback_staging(&self, src: &GpuBuffer, size: u64) {
        let staging = self.ensure_readback_staging(size);
        let mut enc_guard = self.current_encoder.lock().unwrap();
        if let Some(ref mut encoder) = *enc_guard {
            encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        } else {
            // Fallback: no batch active, create a temporary encoder
            let mut encoder = self
                .device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
            self.device.queue.submit(Some(encoder.finish()));
            self.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .ok();
        }
    }

    /// Read back f32 data from the staging buffer after end_batch() has been called.
    /// `n` is the number of f32 elements to read.
    pub fn finish_readback_f32(&self, n: usize) -> Vec<f32> {
        let size = (n as u64) * 4;
        let guard = self.readback_staging.lock().unwrap();
        let staging = guard.as_ref().map(|(b, _)| b.clone());
        drop(guard);
        let staging = staging.expect("finish_readback_f32 called without staging buffer");
        let slice = staging.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging.unmap();
        bytemuck::cast_slice(&result).to_vec()
    }

    /// Compute argmax (index of maximum value) for an f32 array on GPU.
    ///
    /// This method:
    /// 1. Records a reduction dispatch on the current batch encoder
    /// 2. Copies partial results to the staging buffer
    /// 3. Ends the batch (submit + poll)
    /// 4. Reads back partial results (small, ~hundreds of bytes)
    /// 5. CPU-side final reduction to find global argmax
    ///
    /// Returns the index of the maximum element.
    ///
    /// Requires an active batch (call begin_batch() first). This method takes
    /// ownership of the batch encoder and submits it, so the caller must NOT
    /// call end_batch() afterwards.
    ///
    /// This is significantly faster than readback_f32 for large arrays because
    /// it only reads back O(N / 2048) bytes instead of O(N) bytes.
    pub fn argmax_f32(&self, data: &GpuBuffer, n: usize) -> u32 {
        let elems_per_thread: u32 = 16;
        let workgroup_size: u32 = 128;
        let elems_per_wg = workgroup_size * elems_per_thread;
        let num_wgs = (n as u32).div_ceil(elems_per_wg);

        // Allocate partial result buffers
        let partial_size = num_wgs as u64;
        let partial_max = self.create_storage_buffer(partial_size * 4, "argmax_partial_max");
        let partial_idx = self.create_storage_buffer(partial_size * 4, "argmax_partial_idx");
        let n_buf = self.create_uniform_buffer(bytemuck::cast_slice(&[n as u32]), "argmax_n");
        let ept_buf =
            self.create_uniform_buffer(bytemuck::cast_slice(&[elems_per_thread]), "argmax_ept");

        let bind_group = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("argmax_bg"),
                layout: &self.argmax_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(data.as_entire_buffer_binding()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(
                            partial_max.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(
                            partial_idx.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(n_buf.as_entire_buffer_binding()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Buffer(ept_buf.as_entire_buffer_binding()),
                    },
                ],
            });

        // Take the current batch encoder, record dispatch + copies, then submit
        let mut enc_guard = self.current_encoder.lock().unwrap();
        let mut encoder = enc_guard
            .take()
            .expect("argmax_f32 requires an active batch (call begin_batch first)");

        pipeline::dispatch_compute_on_encoder(
            &mut encoder,
            &self.argmax_pipeline,
            &bind_group,
            num_wgs,
            1,
            1,
        );

        // Copy partial results to staging buffer
        let staging_size = partial_size * 8;
        let staging = self.ensure_readback_staging(staging_size);
        encoder.copy_buffer_to_buffer(&partial_max, 0, &staging, 0, partial_size * 4);
        encoder.copy_buffer_to_buffer(
            &partial_idx,
            0,
            &staging,
            partial_size * 4,
            partial_size * 4,
        );

        // Submit and wait
        self.device.queue.submit(Some(encoder.finish()));
        drop(enc_guard);
        self.device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();

        // Read back partial results from staging buffer
        let slice = staging.slice(..staging_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let bytes = data.to_vec();
        drop(data);
        staging.unmap();

        // CPU-side final reduction over partial results
        let partial_vals: &[f32] = bytemuck::cast_slice(&bytes[..(partial_size as usize) * 4]);
        let partial_idcs: &[u32] = bytemuck::cast_slice(&bytes[(partial_size as usize) * 4..]);

        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..partial_size as usize {
            if partial_vals[i] > best_val {
                best_val = partial_vals[i];
                best_idx = partial_idcs[i];
            }
        }
        best_idx
    }

    /// Helper: create a small uniform buffer and write data into it.
    fn create_uniform_buffer(&self, data: &[u8], label: &str) -> Buffer {
        let buf = self.device.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device.queue.write_buffer(&buf, 0, data);
        buf
    }

    /// Helper: dispatch a compute shader with a given bind group.
    /// If a batch encoder is active (via begin_batch()), records on it;
    /// otherwise creates a temporary encoder and submits immediately.
    fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        bind_group: &wgpu::BindGroup,
        x: u32,
        y: u32,
        z: u32,
    ) {
        let mut enc_guard = self.current_encoder.lock().unwrap();
        if let Some(ref mut encoder) = *enc_guard {
            pipeline::dispatch_compute_on_encoder(encoder, pipeline, bind_group, x, y, z);
        } else {
            pipeline::dispatch_compute(&self.device, pipeline, bind_group, x, y, z);
        }
    }

    pub fn dequant_matmul_fused(
        &self,
        quant_buf: &GpuBuffer,
        kind: &QuantKind,
        x_buf: &GpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Option<GpuBuffer> {
        if n != 1 {
            return None;
        }
        let pipeline = self.matmul_pipelines.get(kind)?;
        let mut enc_guard = self.current_encoder.lock().unwrap();
        let batch_encoder: Option<&mut wgpu::CommandEncoder> = enc_guard.as_mut();
        Some(matmul::dispatch_dequant_matmul(
            &self.device,
            pipeline,
            quant_buf,
            x_buf,
            m,
            k,
            batch_encoder,
        ))
    }

    pub fn matmul_f16(
        &self,
        w_buf: &GpuBuffer,
        x_buf: &GpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Option<GpuBuffer> {
        None
    }

    pub fn mat_vec(
        &self,
        w_buf: &GpuBuffer,
        x_buf: &GpuBuffer,
        m: u32,
        n: u32,
    ) -> Option<Vec<f32>> {
        None
    }

    pub fn flash_attention(
        &self,
        q_buf: &GpuBuffer,
        kv_buf: &GpuBuffer,
        kv_offset: u64,
        kv_total_bytes: u64,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_len_q: u32,
        seq_len_kv: u32,
        pos: u32,
    ) -> Option<GpuBuffer> {
        let kv_repeat = n_heads / n_kv_heads;
        let num_kv = (pos + 1).min(seq_len_kv);
        let output_buf =
            self.create_storage_buffer((n_heads * head_dim) as u64 * 4, "flash_attn_out");
        let params = [
            n_heads, n_kv_heads, head_dim, seq_len_q, seq_len_kv, pos, kv_repeat, num_kv,
        ];
        let params_buf = self.create_storage_buffer(32, "flash_attn_params");
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("flash_attn_bg"),
            layout: &self
                .general_pipelines
                .flash_attention
                .get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(q_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(kv_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(output_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(params_buf.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.flash_attention,
            &bind_group,
            n_heads,
            1,
            1,
        );
        Some(output_buf)
    }

    /// Flash attention for batch prefill with causal masking.
    /// Q buffer: [seq_len_q × n_heads × head_dim] f32 row-major.
    /// KV cache: the same layout as flash_attention (K then V, per-head).
    /// Output: [seq_len_q × n_heads × head_dim] f32 row-major.
    ///
    /// Dispatch: n_heads × seq_len_q workgroups, each computes one attention result.
    /// Causal: query at position s attends to positions 0..=s.
    pub fn flash_attention_prefill(
        &self,
        q_buf: &GpuBuffer,
        kv_buf: &GpuBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_len_q: u32,
        cache_stride: u32,
    ) -> Option<GpuBuffer> {
        let kv_repeat = n_heads / n_kv_heads;
        let output_buf = self.create_storage_buffer(
            (seq_len_q * n_heads * head_dim) as u64 * 4,
            "flash_attn_prefill_out",
        );
        let params_data = [
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len_q,
            cache_stride,
            kv_repeat,
            0, // dummy num_kv (computed per-position as s+1 in shader)
        ];
        let params_buf = self.create_storage_buffer(28, "flash_attn_prefill_params");
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("flash_attn_prefill_bg"),
            layout: &self
                .general_pipelines
                .flash_attention_prefill
                .get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(q_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(kv_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(output_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(params_buf.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.flash_attention_prefill,
            &bind_group,
            n_heads,
            seq_len_q,
            1,
        );
        Some(output_buf)
    }

    /// Flash attention for bidirectional (non-causal) batch, used by diffusion models.
    /// Q, K, V buffers are separate (not KV cache format).
    /// Q: [batch × n_heads × head_dim] f32 row-major
    /// K, V: [seq_len × n_heads × head_dim] f32 row-major
    /// Output: [batch × n_heads × head_dim] f32 row-major
    pub fn flash_attention_bidirectional(
        &self,
        q_buf: &GpuBuffer,
        k_buf: &GpuBuffer,
        v_buf: &GpuBuffer,
        n_heads: u32,
        head_dim: u32,
        batch: u32,
        seq_len: u32,
    ) -> Option<GpuBuffer> {
        let output_buf = self.create_storage_buffer(
            (batch * n_heads * head_dim) as u64 * 4,
            "flash_attn_bidir_out",
        );
        let params_data = [n_heads, head_dim, seq_len, 0];
        let params_buf = self.create_storage_buffer(16, "flash_attn_bidir_params");
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("flash_attn_bidir_bg"),
            layout: &self
                .general_pipelines
                .flash_attention_bidirectional
                .get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(q_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(k_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(v_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(output_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(params_buf.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.flash_attention_bidirectional,
            &bind_group,
            n_heads,
            batch,
            1,
        );
        Some(output_buf)
    }

    pub fn rms_norm(
        &self,
        x_buf: &GpuBuffer,
        w_buf: &GpuBuffer,
        out_buf: &GpuBuffer,
        d: u32,
        eps: f32,
        post_norm: bool,
    ) {
        let params = [d as f32, eps, if post_norm { 1.0 } else { 0.0 }, 0.0];
        let uniform = self.create_uniform_buffer(bytemuck::cast_slice(&params), "rms_norm_params");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("rms_norm_bg"),
            layout: &self.general_pipelines.rms_norm.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(w_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(out_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(&self.general_pipelines.rms_norm, &bind_group, 1, 1, 1);
    }

    /// Batch RMS norm: dispatch seq_len workgroups, each normalizing one row.
    /// x_buf and out_buf: [seq_len × d] f32 row-major.
    /// w_buf: [d] f32 (shared weight).
    pub fn rms_norm_batch(
        &self,
        x_buf: &GpuBuffer,
        w_buf: &GpuBuffer,
        out_buf: &GpuBuffer,
        seq_len: u32,
        d: u32,
        eps: f32,
        post_norm: bool,
    ) {
        let params = [d as f32, eps, if post_norm { 1.0 } else { 0.0 }, 0.0];
        let uniform =
            self.create_uniform_buffer(bytemuck::cast_slice(&params), "rms_norm_batch_params");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("rms_norm_batch_bg"),
            layout: &self
                .general_pipelines
                .rms_norm_batch
                .get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(w_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(out_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.rms_norm_batch,
            &bind_group,
            seq_len,
            1,
            1,
        );
    }

    pub fn head_rms_norm(
        &self,
        x_buf: &GpuBuffer,
        w_buf: &GpuBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
        post_norm: bool,
    ) {
        let params = [head_dim as f32, eps, if post_norm { 1.0 } else { 0.0 }, 0.0];
        let uniform =
            self.create_uniform_buffer(bytemuck::cast_slice(&params), "head_rms_norm_params");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("head_rms_norm_bg"),
            layout: &self
                .general_pipelines
                .head_rms_norm
                .get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(w_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.head_rms_norm,
            &bind_group,
            n_heads,
            1,
            1,
        );
    }

    /// Batch head RMS norm: dispatch (n_heads, seq_len, 1) workgroups.
    /// x_buf layout: [seq_len × n_heads × head_dim] f32 row-major, processed in-place.
    /// w_buf: [head_dim] f32 (shared weight for all heads).
    pub fn head_rms_norm_batch(
        &self,
        x_buf: &GpuBuffer,
        w_buf: &GpuBuffer,
        n_heads: u32,
        head_dim: u32,
        seq_len: u32,
        eps: f32,
        post_norm: bool,
    ) {
        let params = [
            n_heads as f32,
            head_dim as f32,
            eps,
            if post_norm { 1.0 } else { 0.0 },
        ];
        let uniform =
            self.create_uniform_buffer(bytemuck::cast_slice(&params), "head_rms_norm_batch_params");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("head_rms_norm_batch_bg"),
            layout: &self
                .general_pipelines
                .head_rms_norm_batch
                .get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(w_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.head_rms_norm_batch,
            &bind_group,
            n_heads,
            seq_len,
            1,
        );
    }

    pub fn rope_combined(
        &self,
        x_buf: &GpuBuffer,
        n_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        pos: u32,
        theta: f32,
        freq_scale: f32,
        yarn_corr_low: f32,
        yarn_corr_high: f32,
        yarn_mscale: f32,
        yarn_n_dims: f32,
    ) {
        let params = [
            n_heads as f32,
            head_dim as f32,
            (rope_dim / 2) as f32,
            pos as f32,
            theta,
            freq_scale,
            yarn_mscale,
            yarn_n_dims,
        ];
        let params_buf = self.create_storage_buffer(32, "rope_params");
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));
        let yarn = [yarn_corr_low, yarn_corr_high, yarn_mscale, 0.0];
        let yarn_buf = self.create_storage_buffer(16, "rope_yarn");
        self.device
            .queue
            .write_buffer(&yarn_buf, 0, bytemuck::cast_slice(&yarn));
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("rope_bg"),
            layout: &self.general_pipelines.rope.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(params_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(yarn_buf.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(&self.general_pipelines.rope, &bind_group, n_heads, 1, 1);
    }

    /// Batch RoPE: dispatch (n_heads, seq_len, 1) workgroups, each head-position pair.
    /// x_buf layout: [seq_len × n_heads × head_dim] f32 row-major, processed in-place.
    pub fn rope_batch(
        &self,
        x_buf: &GpuBuffer,
        n_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        seq_len: u32,
        theta: f32,
        freq_scale: f32,
        yarn_corr_low: f32,
        yarn_corr_high: f32,
        yarn_mscale: f32,
        yarn_n_dims: f32,
    ) {
        let params = [
            n_heads as f32,
            head_dim as f32,
            (rope_dim / 2) as f32,
            0.0, // pos not used in batch shader (uses workgroup_id.y)
            theta,
            freq_scale,
            yarn_mscale,
            yarn_n_dims,
        ];
        let params_buf = self.create_storage_buffer(32, "rope_batch_params");
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));
        let yarn = [yarn_corr_low, yarn_corr_high, yarn_mscale, 0.0];
        let yarn_buf = self.create_storage_buffer(16, "rope_batch_yarn");
        self.device
            .queue
            .write_buffer(&yarn_buf, 0, bytemuck::cast_slice(&yarn));
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("rope_batch_bg"),
            layout: &self.general_pipelines.rope_batch.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(params_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(yarn_buf.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.rope_batch,
            &bind_group,
            n_heads,
            seq_len,
            1,
        );
    }

    pub fn write_cache_kv(
        &self,
        k_buf: &GpuBuffer,
        v_buf: &GpuBuffer,
        cache_buf: &GpuBuffer,
        pos: u32,
        n_kv_heads: u32,
        head_dim: u32,
        cache_stride: u32,
    ) {
        let params = [pos, n_kv_heads, head_dim, cache_stride];
        let uniform = self.create_uniform_buffer(bytemuck::cast_slice(&params), "kv_cache_params");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("kv_cache_bg"),
            layout: &self.general_pipelines.kv_cache.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(k_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(v_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(cache_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        let total = n_kv_heads * head_dim;
        self.dispatch(
            &self.general_pipelines.kv_cache,
            &bind_group,
            total.div_ceil(128),
            1,
            1,
        );
    }

    /// Batch KV cache write: writes all seq_len positions from batched K/V buffers into the cache.
    /// k_buf/v_buf layout: [seq_len × n_kv_heads × head_dim] f32 row-major.
    pub fn write_cache_kv_batch(
        &self,
        k_buf: &GpuBuffer,
        v_buf: &GpuBuffer,
        cache_buf: &GpuBuffer,
        seq_len: u32,
        n_kv_heads: u32,
        head_dim: u32,
        cache_stride: u32,
    ) {
        let params = [seq_len, n_kv_heads, head_dim, cache_stride];
        let uniform =
            self.create_uniform_buffer(bytemuck::cast_slice(&params), "kv_cache_batch_params");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("kv_cache_batch_bg"),
            layout: &self
                .general_pipelines
                .kv_cache_batch_write
                .get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(k_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(v_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(cache_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        let total = seq_len * n_kv_heads * head_dim;
        self.dispatch(
            &self.general_pipelines.kv_cache_batch_write,
            &bind_group,
            total.div_ceil(128),
            1,
            1,
        );
    }

    pub fn dequant_matmul_fused_add_inplace(
        &self,
        quant_buf: &GpuBuffer,
        kind: &QuantKind,
        x_buf: &GpuBuffer,
        accum_buf: &GpuBuffer,
        m: u32,
        k: u32,
    ) -> bool {
        let pipeline = match self.matmul_pipelines.get_add_inplace(kind) {
            Some(p) => p,
            None => return false,
        };
        let mut enc_guard = self.current_encoder.lock().unwrap();
        let batch_encoder: Option<&mut wgpu::CommandEncoder> = enc_guard.as_mut();
        matmul::dispatch_dequant_matmul_add_inplace(
            &self.device,
            pipeline,
            quant_buf,
            x_buf,
            accum_buf,
            m,
            batch_encoder,
        );
        true
    }

    /// Batch dequant+matmul for n > 1 (e.g., prefill with seq_len > 1).
    /// Act buffer layout: [batch × k] f32 row-major.
    /// Returns: [batch × m] f32 row-major, or None if unsupported.
    pub fn dequant_matmul_batch(
        &self,
        quant_buf: &GpuBuffer,
        kind: &QuantKind,
        act_buf: &GpuBuffer,
        m: u32,
        k: u32,
        batch: u32,
    ) -> Option<GpuBuffer> {
        let pipeline = self.matmul_pipelines.get_batch(kind)?;
        let mut enc_guard = self.current_encoder.lock().unwrap();
        let batch_encoder: Option<&mut wgpu::CommandEncoder> = enc_guard.as_mut();
        Some(matmul::dispatch_dequant_matmul_batch(
            &self.device,
            pipeline,
            quant_buf,
            act_buf,
            matmul::BatchMatmulDims { m, k, batch },
            batch_encoder,
        ))
    }

    pub fn add_inplace(&self, a: &GpuBuffer, b: &GpuBuffer, n: u32) {
        let uniform = self.create_uniform_buffer(bytemuck::cast_slice(&[n]), "add_inplace_n");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("add_inplace_bg"),
            layout: &self.general_pipelines.add_inplace.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(a.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(b.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.add_inplace,
            &bind_group,
            n.div_ceil(128),
            1,
            1,
        );
    }

    pub fn silu_mul(&self, gate: &GpuBuffer, up: &GpuBuffer, out: &GpuBuffer, n: u32) {
        let uniform = self.create_uniform_buffer(bytemuck::cast_slice(&[n]), "silu_mul_n");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("silu_mul_bg"),
            layout: &self.general_pipelines.silu_mul.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(gate.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(up.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.silu_mul,
            &bind_group,
            n.div_ceil(128),
            1,
            1,
        );
    }

    pub fn fused_gate_up(
        &self,
        gate_w: &GpuBuffer,
        up_w: &GpuBuffer,
        x_buf: &GpuBuffer,
        gate_out: &GpuBuffer,
        up_out: &GpuBuffer,
        m: u32,
        n: u32,
    ) -> bool {
        let pipeline = match self.fused_gate_up_pipeline {
            Some(ref p) => p,
            None => return false,
        };
        let mut enc_guard = self.current_encoder.lock().unwrap();
        let batch_encoder: Option<&mut wgpu::CommandEncoder> = enc_guard.as_mut();

        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("fused_gate_up_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(gate_w.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(up_w.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(gate_out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(up_out.as_entire_buffer_binding()),
                },
            ],
        });

        match batch_encoder {
            Some(enc) => pipeline::dispatch_compute_on_encoder(enc, pipeline, &bind_group, m, 1, 1),
            None => pipeline::dispatch_compute(&self.device, pipeline, &bind_group, m, 1, 1),
        }
        true
    }

    /// Batch fused gate+up for prefill: dispatch (ffn_dim, seq_len, 1) workgroups.
    /// x layout: [seq_len × k] f32 row-major
    /// gate_out, up_out: [seq_len × ffn_dim] f32 row-major
    pub fn fused_gate_up_batch(
        &self,
        gate_w: &GpuBuffer,
        up_w: &GpuBuffer,
        x_buf: &GpuBuffer,
        gate_out: &GpuBuffer,
        up_out: &GpuBuffer,
        ffn_dim: u32,
        k: u32,
        seq_len: u32,
    ) -> bool {
        let pipeline = match self.fused_gate_up_batch_pipeline {
            Some(ref p) => p,
            None => return false,
        };
        let params = self.create_storage_buffer(8, "fused_gate_up_batch_params");
        let params_data: [u32; 2] = [k, ffn_dim];
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::cast_slice(&params_data));
        let mut enc_guard = self.current_encoder.lock().unwrap();
        let batch_encoder: Option<&mut wgpu::CommandEncoder> = enc_guard.as_mut();

        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("fused_gate_up_batch_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(gate_w.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(up_w.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(gate_out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(up_out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Buffer(params.as_entire_buffer_binding()),
                },
            ],
        });

        match batch_encoder {
            Some(enc) => pipeline::dispatch_compute_on_encoder(
                enc,
                pipeline,
                &bind_group,
                ffn_dim,
                seq_len,
                1,
            ),
            None => {
                pipeline::dispatch_compute(&self.device, pipeline, &bind_group, ffn_dim, seq_len, 1)
            }
        }
        true
    }

    pub fn fused_qkv(
        &self,
        qw: &GpuBuffer,
        kw: &GpuBuffer,
        vw: &GpuBuffer,
        x: &GpuBuffer,
        q_out: &GpuBuffer,
        k_out: &GpuBuffer,
        v_out: &GpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> bool {
        let pipeline = match self.fused_qkv_pipeline {
            Some(ref p) => p,
            None => return false,
        };
        let params = self.create_storage_buffer(4, "fused_qkv_params");
        // Write n_kv_rows as the first u32 element (shader reads params[0])
        let params_data = [n];
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::cast_slice(&params_data));
        let mut enc_guard = self.current_encoder.lock().unwrap();
        let batch_encoder: Option<&mut wgpu::CommandEncoder> = enc_guard.as_mut();

        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("fused_qkv_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(qw.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(kw.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(vw.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(q_out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Buffer(k_out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Buffer(v_out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::Buffer(params.as_entire_buffer_binding()),
                },
            ],
        });

        match batch_encoder {
            Some(enc) => pipeline::dispatch_compute_on_encoder(enc, pipeline, &bind_group, m, 1, 1),
            None => pipeline::dispatch_compute(&self.device, pipeline, &bind_group, m, 1, 1),
        }
        true
    }

    /// Batch fused QKV for prefill: dispatch (nq, seq_len, 1) workgroups.
    /// x layout: [seq_len × k] f32 row-major
    /// q_out: [seq_len × nq] f32 row-major
    /// k_out: [seq_len × nkv] f32 row-major
    /// v_out: [seq_len × nkv] f32 row-major
    pub fn fused_qkv_batch(
        &self,
        qw: &GpuBuffer,
        kw: &GpuBuffer,
        vw: &GpuBuffer,
        x: &GpuBuffer,
        q_out: &GpuBuffer,
        k_out: &GpuBuffer,
        v_out: &GpuBuffer,
        nq: u32,
        nkv: u32,
        k: u32,
        seq_len: u32,
    ) -> bool {
        let pipeline = match self.fused_qkv_batch_pipeline {
            Some(ref p) => p,
            None => return false,
        };
        let params = self.create_storage_buffer(12, "fused_qkv_batch_params");
        let params_data: [u32; 3] = [nkv, k, nq];
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::cast_slice(&params_data));
        let mut enc_guard = self.current_encoder.lock().unwrap();
        let batch_encoder: Option<&mut wgpu::CommandEncoder> = enc_guard.as_mut();

        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("fused_qkv_batch_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(qw.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(kw.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(vw.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(q_out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Buffer(k_out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Buffer(v_out.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::Buffer(params.as_entire_buffer_binding()),
                },
            ],
        });

        match batch_encoder {
            Some(enc) => {
                pipeline::dispatch_compute_on_encoder(enc, pipeline, &bind_group, nq, seq_len, 1)
            }
            None => pipeline::dispatch_compute(&self.device, pipeline, &bind_group, nq, seq_len, 1),
        }
        true
    }

    pub fn copy_buffer(&self, src: &GpuBuffer, dst: &GpuBuffer) {
        let size = src.size().min(dst.size());
        let mut enc_guard = self.current_encoder.lock().unwrap();
        if let Some(ref mut encoder) = *enc_guard {
            encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
        } else {
            let mut encoder = self
                .device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
            self.device.queue.submit(Some(encoder.finish()));
        }
    }

    /// Copy a byte range from one buffer to another.
    /// Both buffers must have COPY_DST and COPY_SRC usage respectively.
    pub fn copy_buffer_range(
        &self,
        src: &GpuBuffer,
        src_offset: u64,
        dst: &GpuBuffer,
        dst_offset: u64,
        size: u64,
    ) {
        let mut enc_guard = self.current_encoder.lock().unwrap();
        if let Some(ref mut encoder) = *enc_guard {
            encoder.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
        } else {
            let mut encoder = self
                .device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
            self.device.queue.submit(Some(encoder.finish()));
        }
    }

    pub fn run_simplest_test(&self, input: &[f32]) -> Vec<f32> {
        use wgpu::{BindGroupDescriptor, BindGroupEntry, BindingResource};

        let pipeline = match &self.simplest_pipeline {
            Some(p) => p,
            None => {
                eprintln!("[gpu] simplest pipeline not available");
                return vec![];
            }
        };

        let input_buf = self.upload_f32(input, "simplest-input");
        let output_buf = self.create_storage_buffer((input.len() * 4) as u64, "simplest-output");

        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("simplest-bindgroup"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(input_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(output_buf.as_entire_buffer_binding()),
                },
            ],
        });

        pipeline::dispatch_compute(
            &self.device,
            pipeline,
            &bind_group,
            input.len() as u32 / 64 + 1,
            1,
            1,
        );

        self.readback_f32(&output_buf, input.len())
    }

    /// Conv2D 3x3 with padding=1, stride=1.
    /// input: [c_in × h × w] f32
    /// weight: [c_out × c_in × 3 × 3] f32
    /// bias: [c_out] f32
    /// Returns: [c_out × h × w] f32
    pub fn conv2d_3x3(
        &self,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        bias: &GpuBuffer,
        c_in: u32,
        c_out: u32,
        h: u32,
        w: u32,
    ) -> GpuBuffer {
        let output = self.create_storage_buffer((c_out * h * w) as u64 * 4, "conv2d_out");
        let params = [c_in, c_out, h, w];
        let uniform = self.create_uniform_buffer(bytemuck::cast_slice(&params), "conv2d_params");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("conv2d_bg"),
            layout: &self.general_pipelines.conv2d_3x3.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(input.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(weight.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(bias.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(output.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        let total = c_out * h * w;
        self.dispatch(
            &self.general_pipelines.conv2d_3x3,
            &bind_group,
            total.div_ceil(128),
            1,
            1,
        );
        output
    }

    /// Group normalization.
    /// x: [c × hw] f32 (channel-first, in-place modified)
    /// weight: [c] f32
    /// bias: [c] f32
    /// n_groups: number of groups
    pub fn group_norm(
        &self,
        x: &GpuBuffer,
        weight: &GpuBuffer,
        bias: &GpuBuffer,
        c: u32,
        hw: u32,
        n_groups: u32,
        eps: f32,
    ) {
        let params = [c as f32, hw as f32, n_groups as f32, eps];
        let uniform =
            self.create_uniform_buffer(bytemuck::cast_slice(&params), "group_norm_params");
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("group_norm_bg"),
            layout: &self.general_pipelines.group_norm.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(x.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(weight.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(bias.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(uniform.as_entire_buffer_binding()),
                },
            ],
        });
        self.dispatch(
            &self.general_pipelines.group_norm,
            &bind_group,
            n_groups,
            1,
            1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create Q1_0G128 quantized row data from f32 values.
    /// Each 128-element block: 2 bytes f16 scale (LE) + 16 bytes sign bits.
    /// sign bit = 1 → value = +scale, sign bit = 0 → value = -scale
    fn quantize_q1_0_row(data: &[f32]) -> Vec<u8> {
        assert!(
            data.len().is_multiple_of(128),
            "Q1_0 requires multiple of 128 elements"
        );
        let blocks = data.len() / 128;
        let mut result = Vec::with_capacity(blocks * 18);
        for b in 0..blocks {
            let start = b * 128;
            // Scale: max absolute value in the block
            let max_abs = data[start..start + 128]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs };
            let scale_f16 = half::f16::from_f32(scale);
            let scale_bytes = scale_f16.to_le_bytes();
            result.push(scale_bytes[0]);
            result.push(scale_bytes[1]);
            // 16 bytes of sign bits (128 elements, LSB-first within each byte)
            for byte_idx in 0..16 {
                let mut byte = 0u8;
                for bit in 0..8 {
                    let elem_idx = start + byte_idx * 8 + bit;
                    if elem_idx < data.len() && data[elem_idx] >= 0.0 {
                        byte |= 1 << bit;
                    }
                }
                result.push(byte);
            }
        }
        result
    }

    /// Compute expected dot product for a Q1_0G128 weight row: dequantize and dot with activation.
    #[allow(dead_code)]
    fn cpu_q1_0_dot(row_data: &[u8], activation: &[f32]) -> f32 {
        assert!(row_data.len() >= 2);
        let blocks = row_data.len() / 18;
        let mut sum = 0.0f32;
        for b in 0..blocks {
            let off = b * 18;
            let scale_lo = row_data[off] as u16;
            let scale_hi = row_data[off + 1] as u16;
            let scale_bits = (scale_hi << 8) | scale_lo;
            let scale = half::f16::from_bits(scale_bits).to_f32();
            for bit_idx in 0..128 {
                let byte_idx = off + 2 + bit_idx / 8;
                let bit = (row_data[byte_idx] >> (bit_idx & 7)) & 1;
                let w_val = if bit == 1 { scale } else { -scale };
                let elem = b * 128 + bit_idx;
                if elem < activation.len() {
                    sum += w_val * activation[elem];
                }
            }
        }
        sum
    }

    #[test]
    fn test_simplest_shader() {
        let gpu = pollster::block_on(GpuCompute::new());
        let gpu = match gpu {
            Some(g) => g,
            None => {
                eprintln!("[test] no GPU adapter found — skipping");
                return;
            }
        };

        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = gpu.run_simplest_test(&input);

        assert_eq!(output.len(), input.len(), "output length mismatch");
        for i in 0..input.len() {
            assert!(
                (output[i] - (input[i] + 1.0)).abs() < 0.001,
                "output[{}] = {} expected {}",
                i,
                output[i],
                input[i] + 1.0
            );
        }
    }

    #[test]
    fn test_q1_0_matmul() {
        let gpu = pollster::block_on(GpuCompute::new());
        let gpu = match gpu {
            Some(g) => g,
            None => {
                eprintln!("[test] no GPU adapter found — skipping Q1_0 matmul test");
                return;
            }
        };
        assert!(gpu.has_dequant(&QuantKind::Q1_0));
        assert!(gpu.has_dequant(&QuantKind::Q1_0G128));
    }

    #[test]
    fn test_q2_0_matmul() {
        let gpu = pollster::block_on(GpuCompute::new());
        let gpu = match gpu {
            Some(g) => g,
            None => {
                eprintln!("[test] no GPU adapter found — skipping Q2_0 matmul test");
                return;
            }
        };
        assert!(gpu.has_dequant(&QuantKind::Q2_0));
    }

    #[test]
    fn test_q1_0_matmul_correctness() {
        let gpu = pollster::block_on(GpuCompute::new());
        let gpu = match gpu {
            Some(g) => g,
            None => {
                eprintln!("[test] no GPU adapter found — skipping matmul correctness test");
                return;
            }
        };

        // Weight matrix: 4 rows × 256 cols (2 blocks per row)
        let m = 4u32;
        let k = 256u32;
        // All weights = +1.0 → scale=1.0, all sign bits=1
        let mut weight_f32 = vec![0.0f32; (m * k) as usize];
        weight_f32.fill(1.0);
        // Quantize row by row
        let mut quant_data = Vec::new();
        for r in 0..m as usize {
            let row = &weight_f32[r * k as usize..(r + 1) * k as usize];
            quant_data.extend_from_slice(&quantize_q1_0_row(row));
        }
        let weight_buf = gpu.upload_bytes(&quant_data, "test_weight");

        // Activation: all 1.0
        let act: Vec<f32> = vec![1.0; k as usize];
        let act_buf = gpu.upload_f32(&act, "test_act");

        // GPU matmul
        let result_buf = gpu
            .dequant_matmul_fused(&weight_buf, &QuantKind::Q1_0G128, &act_buf, m, 1, k)
            .expect("dequant_matmul_fused returned None");
        let gpu_out = gpu.readback_f32(&result_buf, m as usize);

        // Expected: for each row, sum of activation = 256.0 (all weights = +1.0)
        let expected = 256.0f32;
        for (i, &v) in gpu_out.iter().enumerate().take(m as usize) {
            assert!(
                (v - expected).abs() < 0.1,
                "gpu_out[{}] = {} vs expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_q1_0_matmul_mixed_weights() {
        let gpu = pollster::block_on(GpuCompute::new());
        let gpu = match gpu {
            Some(g) => g,
            None => {
                eprintln!("[test] no GPU adapter found — skipping mixed-weights test");
                return;
            }
        };

        // 2 rows × 128 cols (1 block per row)
        let m = 2u32;
        let k = 128u32;
        // Row 0: all +1.0; Row 1: all -1.0
        let mut weight_f32 = vec![0.0f32; (m * k) as usize];
        for i in 0..k as usize {
            weight_f32[i] = 1.0; // row 0 = +1.0
            weight_f32[k as usize + i] = -1.0; // row 1 = -1.0
        }
        let mut quant_data = Vec::new();
        for r in 0..m as usize {
            quant_data.extend_from_slice(&quantize_q1_0_row(
                &weight_f32[r * k as usize..(r + 1) * k as usize],
            ));
        }
        let weight_buf = gpu.upload_bytes(&quant_data, "test_weight_mixed");

        // Activation: ascending values [0..127]
        let act: Vec<f32> = (0..k).map(|x| x as f32).collect();
        let act_buf = gpu.upload_f32(&act, "test_act_mixed");

        let result_buf = gpu
            .dequant_matmul_fused(&weight_buf, &QuantKind::Q1_0G128, &act_buf, m, 1, k)
            .expect("dequant_matmul_fused returned None");
        let gpu_out = gpu.readback_f32(&result_buf, m as usize);

        // Row 0: all +1.0 → sum(act) = sum(0..127) = 8128
        // Row 1: all -1.0 → -sum(act) = -8128
        let act_sum: f32 = (0..k).sum::<u32>() as f32;
        assert!(
            (gpu_out[0] - act_sum).abs() < 0.1,
            "row 0: {} vs expected {}",
            gpu_out[0],
            act_sum
        );
        assert!(
            (gpu_out[1] + act_sum).abs() < 0.1,
            "row 1: {} vs expected {}",
            gpu_out[1],
            -act_sum
        );
    }

    #[test]
    fn test_q1_0_matmul_add_inplace() {
        let gpu = pollster::block_on(GpuCompute::new());
        let gpu = match gpu {
            Some(g) => g,
            None => {
                eprintln!("[test] no GPU adapter found — skipping add_inplace test");
                return;
            }
        };

        // 2 rows × 128 cols
        let m = 2u32;
        let k = 128u32;
        let weight_f32 = vec![1.0; (m * k) as usize]; // all +1.0
        let mut quant_data = Vec::new();
        for r in 0..m as usize {
            quant_data.extend_from_slice(&quantize_q1_0_row(
                &weight_f32[r * k as usize..(r + 1) * k as usize],
            ));
        }
        let weight_buf = gpu.upload_bytes(&quant_data, "test_weight_add");

        // Activation: all 1.0
        let act: Vec<f32> = vec![1.0; k as usize];
        let act_buf = gpu.upload_f32(&act, "test_act_add");

        // Accumulator: all 3.0
        let accum = vec![3.0f32; m as usize];
        let accum_buf = gpu.upload_f32(&accum, "test_accum");

        // GPU: accum[i] = accum[i] + dot(weight[i], act)
        // = 3.0 + 128.0 = 131.0
        let ok = gpu.dequant_matmul_fused_add_inplace(
            &weight_buf,
            &QuantKind::Q1_0G128,
            &act_buf,
            &accum_buf,
            m,
            k,
        );
        assert!(ok, "dequant_matmul_fused_add_inplace returned false");

        let gpu_out = gpu.readback_f32(&accum_buf, m as usize);
        let expected = 3.0 + 128.0;
        for (i, &v) in gpu_out.iter().enumerate().take(m as usize) {
            assert!(
                (v - expected).abs() < 0.1,
                "gpu_out[{}] = {} vs expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_fused_gate_up_correctness() {
        let gpu = pollster::block_on(GpuCompute::new());
        let gpu = match gpu {
            Some(g) => g,
            None => {
                eprintln!("[test] no GPU adapter found — skipping fused_gate_up test");
                return;
            }
        };

        // Gate: 4 rows × 128 cols, Up: 4 rows × 128 cols
        let m = 4u32;
        let k = 128u32;

        // Gate weights: all +1.0; Up weights: alternating +1.0/-1.0
        let mut gate_f32 = vec![0.0f32; (m * k) as usize];
        let mut up_f32 = vec![0.0f32; (m * k) as usize];
        for r in 0..m as usize {
            for i in 0..k as usize {
                gate_f32[r * k as usize + i] = 1.0;
                up_f32[r * k as usize + i] = if i % 2 == 0 { 1.0 } else { -1.0 };
            }
        }

        let mut gate_quant = Vec::new();
        let mut up_quant = Vec::new();
        for r in 0..m as usize {
            gate_quant.extend_from_slice(&quantize_q1_0_row(
                &gate_f32[r * k as usize..(r + 1) * k as usize],
            ));
            up_quant.extend_from_slice(&quantize_q1_0_row(
                &up_f32[r * k as usize..(r + 1) * k as usize],
            ));
        }
        let gate_w = gpu.upload_bytes(&gate_quant, "test_gate_weight");
        let up_w = gpu.upload_bytes(&up_quant, "test_up_weight");

        // Activation: all 1.0
        let act: Vec<f32> = vec![1.0; k as usize];
        let act_buf = gpu.upload_f32(&act, "test_act_gate_up");

        // Output buffers (pre-filled with 0)
        let gate_out = gpu.create_storage_buffer((m as u64) * 4, "gate_out");
        let up_out = gpu.create_storage_buffer((m as u64) * 4, "up_out");

        let ok = gpu.fused_gate_up(&gate_w, &up_w, &act_buf, &gate_out, &up_out, m, k);
        assert!(ok, "fused_gate_up returned false");

        let gate_result = gpu.readback_f32(&gate_out, m as usize);
        let up_result = gpu.readback_f32(&up_out, m as usize);

        // Expected: gate = sum(1.0 * 1.0 * 128) = 128.0 per row
        // up = sum of (+1.0 * 1.0 for even cols, -1.0 * 1.0 for odd cols) = 0.0 per row
        let expected_gate = k as f32;
        let expected_up = 0.0f32;
        for i in 0..m as usize {
            assert!(
                (gate_result[i] - expected_gate).abs() < 0.1,
                "gate[{}] = {} vs expected {}",
                i,
                gate_result[i],
                expected_gate
            );
            assert!(
                (up_result[i] - expected_up).abs() < 0.1,
                "up[{}] = {} vs expected {}",
                i,
                up_result[i],
                expected_up
            );
        }
    }

    #[test]
    fn test_fused_qkv_correctness() {
        let gpu = pollster::block_on(GpuCompute::new());
        let gpu = match gpu {
            Some(g) => g,
            None => {
                eprintln!("[test] no GPU adapter found — skipping fused_qkv test");
                return;
            }
        };

        // Q: 16 rows × 128 cols, K: 8 rows × 128 cols, V: 8 rows × 128 cols
        let nq = 16u32;
        let nkv = 8u32;
        let k = 128u32;

        // Q weights: all +1.0; K weights: all +1.0; V weights: all +1.0
        let make_ones = |r: u32| -> Vec<f32> { vec![1.0; (r * k) as usize] };
        let q_f32 = make_ones(nq);
        let k_f32 = make_ones(nkv);
        let v_f32 = make_ones(nkv);

        let mut q_quant = Vec::new();
        let mut k_quant = Vec::new();
        let mut v_quant = Vec::new();
        for r in 0..nq as usize {
            q_quant.extend_from_slice(&quantize_q1_0_row(
                &q_f32[r * k as usize..(r + 1) * k as usize],
            ));
        }
        for r in 0..nkv as usize {
            k_quant.extend_from_slice(&quantize_q1_0_row(
                &k_f32[r * k as usize..(r + 1) * k as usize],
            ));
            v_quant.extend_from_slice(&quantize_q1_0_row(
                &v_f32[r * k as usize..(r + 1) * k as usize],
            ));
        }
        let qw = gpu.upload_bytes(&q_quant, "test_qw");
        let kw = gpu.upload_bytes(&k_quant, "test_kw");
        let vw = gpu.upload_bytes(&v_quant, "test_vw");

        // Activation: all 1.0
        let act: Vec<f32> = vec![1.0; k as usize];
        let act_buf = gpu.upload_f32(&act, "test_act_qkv");

        // Output buffers
        let q_out = gpu.create_storage_buffer((nq as u64) * 4, "q_out");
        let k_out = gpu.create_storage_buffer((nkv as u64) * 4, "k_out");
        let v_out = gpu.create_storage_buffer((nkv as u64) * 4, "v_out");

        let ok = gpu.fused_qkv(&qw, &kw, &vw, &act_buf, &q_out, &k_out, &v_out, nq, nkv, k);
        assert!(ok, "fused_qkv returned false");

        let q_result = gpu.readback_f32(&q_out, nq as usize);
        let k_result = gpu.readback_f32(&k_out, nkv as usize);
        let v_result = gpu.readback_f32(&v_out, nkv as usize);

        // All weights = +1.0, activation = 1.0 → dot = 128.0 for each row
        let expected = k as f32;
        for (i, &v) in q_result.iter().enumerate().take(nq as usize) {
            assert!(
                (v - expected).abs() < 0.1,
                "q[{}] = {} vs expected {}",
                i,
                v,
                expected
            );
        }
        for i in 0..nkv as usize {
            assert!(
                (k_result[i] - expected).abs() < 0.1,
                "k[{}] = {} vs expected {}",
                i,
                k_result[i],
                expected
            );
            assert!(
                (v_result[i] - expected).abs() < 0.1,
                "v[{}] = {} vs expected {}",
                i,
                v_result[i],
                expected
            );
        }
    }

    #[test]
    fn test_argmax_correctness() {
        let gpu = pollster::block_on(GpuCompute::new());
        let gpu = match gpu {
            Some(g) => g,
            None => {
                eprintln!("[test] no GPU adapter found — skipping argmax test");
                return;
            }
        };

        // Test 1: simple case — known max at specific position
        let n = 256u32;
        let mut data = vec![0.0f32; n as usize];
        data[42] = 99.0; // max at index 42
        data[200] = 50.0; // second max
        let buf = gpu.upload_f32(&data, "argmax_test1");
        gpu.begin_batch();
        let idx = gpu.argmax_f32(&buf, n as usize);
        assert_eq!(idx, 42, "argmax should be 42, got {}", idx);
        eprintln!("[test] argmax test 1 passed: idx={}", idx);

        // Test 2: all negative, max should be the least negative
        let n = 128u32;
        let mut data = vec![0.0f32; n as usize];
        for (i, v) in data.iter_mut().enumerate() {
            *v = -(i as f32) - 1.0; // -1, -2, -3, ...
        }
        // Max (least negative) is at index 0: -1.0
        let buf = gpu.upload_f32(&data, "argmax_test2");
        gpu.begin_batch();
        let idx = gpu.argmax_f32(&buf, n as usize);
        assert_eq!(idx, 0, "argmax should be 0, got {}", idx);
        eprintln!("[test] argmax test 2 passed: idx={}", idx);

        // Test 3: large array (151669 elements like vocab)
        let n = 151669usize;
        let mut data = vec![-100.0f32; n];
        data[123456] = 42.0; // max at index 123456
        let buf = gpu.upload_f32(&data, "argmax_test3");
        gpu.begin_batch();
        let idx = gpu.argmax_f32(&buf, n);
        assert_eq!(idx, 123456, "argmax should be 123456, got {}", idx);
        eprintln!("[test] argmax test 3 (large) passed: idx={}", idx);

        // Test 4: first element is max
        let n = 512u32;
        let mut data = vec![0.0f32; n as usize];
        data[0] = 999.0;
        for (i, v) in data.iter_mut().enumerate().skip(1) {
            *v = (i as f32) * 0.1;
        }
        let buf = gpu.upload_f32(&data, "argmax_test4");
        gpu.begin_batch();
        let idx = gpu.argmax_f32(&buf, n as usize);
        assert_eq!(idx, 0, "argmax should be 0, got {}", idx);
        eprintln!("[test] argmax test 4 passed: idx={}", idx);

        // Test 5: last element is max
        let n = 1000u32;
        let mut data = vec![0.0f32; n as usize];
        for v in data.iter_mut().take(n as usize - 1) {
            *v = 0.5;
        }
        data[n as usize - 1] = 999.0;
        let buf = gpu.upload_f32(&data, "argmax_test5");
        gpu.begin_batch();
        let idx = gpu.argmax_f32(&buf, n as usize);
        assert_eq!(idx, n - 1, "argmax should be {}, got {}", n - 1, idx);
        eprintln!("[test] argmax test 5 (last element) passed: idx={}", idx);
    }
}
