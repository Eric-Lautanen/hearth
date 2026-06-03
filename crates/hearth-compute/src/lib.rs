mod buffers;
mod device;
mod pipeline;
mod shaders;

use std::collections::HashMap;

use hearth_gguf::GgmlDType;

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

pub struct GpuCompute {
    pub pool: std::sync::Mutex<HashMap<String, GpuBuffer>>,
    pub norm_buffers: HashMap<String, GpuBuffer>,
    pub kv_cache: Vec<GpuBuffer>,
    pub cache_max_seq: u32,
    device: device::GpuDevice,
    simplest_pipeline: Option<wgpu::ComputePipeline>,
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

        Some(Self {
            pool: std::sync::Mutex::new(HashMap::new()),
            norm_buffers: HashMap::new(),
            kv_cache: Vec::new(),
            cache_max_seq: 0,
            device: dev,
            simplest_pipeline,
        })
    }

    pub fn has_dequant(&self, kind: &QuantKind) -> bool {
        false
    }

    pub fn has_matmul_f16(&self) -> bool {
        false
    }

    pub fn warmup(&self) {}

    pub fn begin_batch(&self) {}

    pub fn end_batch(&self) {}

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

    pub fn dequant_matmul_fused(
        &self,
        quant_buf: &GpuBuffer,
        kind: &QuantKind,
        x_buf: &GpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Option<GpuBuffer> {
        None
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
        None
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
    }

    pub fn dequant_matmul_fused_add_inplace(
        &self,
        quant_buf: &GpuBuffer,
        x_buf: &GpuBuffer,
        accum_buf: &GpuBuffer,
        m: u32,
        n: u32,
    ) -> bool {
        false
    }

    pub fn add_inplace(&self, a: &GpuBuffer, b: &GpuBuffer, n: u32) {}

    pub fn silu_mul(&self, gate: &GpuBuffer, up: &GpuBuffer, out: &GpuBuffer, n: u32) {}

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
        false
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
        false
    }

    pub fn copy_buffer(&self, src: &GpuBuffer, dst: &GpuBuffer) {}

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
