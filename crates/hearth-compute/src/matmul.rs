use crate::device::GpuDevice;
use crate::shaders;
use crate::QuantKind;
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, ComputePipeline};

pub(crate) struct DequantMatmulPipelines {
    pub q1_0: ComputePipeline,
    pub q2_0: ComputePipeline,
    pub q1_0_add_inplace: ComputePipeline,
    pub q2_0_add_inplace: ComputePipeline,
    pub q1_0_batch: ComputePipeline,
    pub q2_0_batch: ComputePipeline,
}

impl DequantMatmulPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let q1_0 = super::pipeline::create_compute_pipeline(
            device,
            shaders::Q1_0_MATMUL_WGSL,
            "q1_0_matmul",
        );
        let q2_0 = super::pipeline::create_compute_pipeline(
            device,
            shaders::Q2_0_MATMUL_WGSL,
            "q2_0_matmul",
        );
        let q1_0_add_inplace = super::pipeline::create_compute_pipeline(
            device,
            shaders::Q1_0_MATMUL_ADD_INPLACE_WGSL,
            "q1_0_matmul_add_inplace",
        );
        let q2_0_add_inplace = super::pipeline::create_compute_pipeline(
            device,
            shaders::Q2_0_MATMUL_ADD_INPLACE_WGSL,
            "q2_0_matmul_add_inplace",
        );
        let q1_0_batch = super::pipeline::create_compute_pipeline(
            device,
            shaders::Q1_0_MATMUL_BATCH_WGSL,
            "q1_0_matmul_batch",
        );
        let q2_0_batch = super::pipeline::create_compute_pipeline(
            device,
            shaders::Q2_0_MATMUL_BATCH_WGSL,
            "q2_0_matmul_batch",
        );
        Self {
            q1_0,
            q2_0,
            q1_0_add_inplace,
            q2_0_add_inplace,
            q1_0_batch,
            q2_0_batch,
        }
    }

    pub fn get(&self, kind: &QuantKind) -> Option<&ComputePipeline> {
        match kind {
            QuantKind::Q1_0 | QuantKind::Q1_0G128 => Some(&self.q1_0),
            QuantKind::Q2_0 => Some(&self.q2_0),
            _ => None,
        }
    }

    pub fn get_add_inplace(&self, kind: &QuantKind) -> Option<&ComputePipeline> {
        match kind {
            QuantKind::Q1_0 | QuantKind::Q1_0G128 => Some(&self.q1_0_add_inplace),
            QuantKind::Q2_0 => Some(&self.q2_0_add_inplace),
            _ => None,
        }
    }

    pub fn get_batch(&self, kind: &QuantKind) -> Option<&ComputePipeline> {
        match kind {
            QuantKind::Q1_0 | QuantKind::Q1_0G128 => Some(&self.q1_0_batch),
            QuantKind::Q2_0 => Some(&self.q2_0_batch),
            _ => None,
        }
    }
}

/// Dispatch a dequant+matmul for a single input vector (n=1).
///
/// weight_buf: GPU buffer containing the quantized weight matrix (m rows × k cols)
/// act_buf: GPU buffer containing the f32 activation vector (length k)
/// m: number of weight rows (output dimension)
/// batch_encoder: if Some, records the dispatch on the given encoder (batching mode);
///                if None, creates a temporary encoder and submits immediately.
///
/// Returns: GPU buffer of length m with f32 results.
pub fn dispatch_dequant_matmul(
    device: &GpuDevice,
    pipeline: &ComputePipeline,
    weight_buf: &Buffer,
    act_buf: &Buffer,
    m: u32,
    _k: u32,
    batch_encoder: Option<&mut wgpu::CommandEncoder>,
) -> Buffer {
    let output_buf =
        super::buffers::create_storage_buffer(&device.device, (m as u64) * 4, "dequant_matmul_out");

    // WGSL maxComputeWorkgroupsPerDimension is 65535; split large dispatches across Y.
    let max_x = 65535u32;
    let dx = m.min(max_x);
    let dy = m.div_ceil(max_x);

    let group_size_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dequant_matmul_gsx"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    device
        .queue
        .write_buffer(&group_size_buf, 0, bytemuck::cast_slice(&[dx]));

    let bind_group = device.device.create_bind_group(&BindGroupDescriptor {
        label: Some("dequant_matmul_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(weight_buf.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(act_buf.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(output_buf.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Buffer(group_size_buf.as_entire_buffer_binding()),
            },
        ],
    });

    match batch_encoder {
        Some(enc) => {
            super::pipeline::dispatch_compute_on_encoder(enc, pipeline, &bind_group, dx, dy, 1)
        }
        None => super::pipeline::dispatch_compute(device, pipeline, &bind_group, dx, dy, 1),
    }

    output_buf
}

/// Dispatch a dequant+matmul with fused add-inplace.
///
/// Computes: output[i] = output[i] + dot(weight[i], activation)
/// where output is pre-initialized with the accumulator values.
///
/// weight_buf: GPU buffer containing the quantized weight matrix (m rows × k cols)
/// act_buf: GPU buffer containing the f32 activation vector (length k)
/// output_buf: GPU buffer of length m, read for initial accumulator, written with result
/// m: number of weight rows (output dimension)
/// batch_encoder: if Some, records the dispatch on the given encoder; else creates one.
pub fn dispatch_dequant_matmul_add_inplace(
    device: &GpuDevice,
    pipeline: &ComputePipeline,
    weight_buf: &Buffer,
    act_buf: &Buffer,
    output_buf: &Buffer,
    m: u32,
    batch_encoder: Option<&mut wgpu::CommandEncoder>,
) {
    let bind_group = device.device.create_bind_group(&BindGroupDescriptor {
        label: Some("dequant_matmul_add_inplace_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(weight_buf.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(act_buf.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(output_buf.as_entire_buffer_binding()),
            },
        ],
    });

    match batch_encoder {
        Some(enc) => {
            super::pipeline::dispatch_compute_on_encoder(enc, pipeline, &bind_group, m, 1, 1)
        }
        None => super::pipeline::dispatch_compute(device, pipeline, &bind_group, m, 1, 1),
    }
}

/// Dispatch a batch dequant+matmul for multiple input vectors (n > 1).
///
/// weight_buf: GPU buffer containing the quantized weight matrix (m rows × k cols)
/// Geometry of a batched dequant matmul: `m` weight rows per activation vector,
/// `k` input columns, and `batch` activation vectors.
#[derive(Clone, Copy)]
pub struct BatchMatmulDims {
    pub m: u32,
    pub k: u32,
    pub batch: u32,
}

/// act_buf: GPU buffer containing the f32 activation matrix [batch × k] row-major
/// (the output geometry comes from `dims`)
///
/// Returns: GPU buffer of length batch × m with f32 results [batch × m] row-major.
pub fn dispatch_dequant_matmul_batch(
    device: &GpuDevice,
    pipeline: &ComputePipeline,
    weight_buf: &Buffer,
    act_buf: &Buffer,
    dims: BatchMatmulDims,
    batch_encoder: Option<&mut wgpu::CommandEncoder>,
) -> Buffer {
    let BatchMatmulDims { m, k, batch } = dims;
    let output_buf = super::buffers::create_storage_buffer(
        &device.device,
        (batch as u64) * (m as u64) * 4,
        "dequant_matmul_batch_out",
    );

    // Params: [m, k]
    let params_buf =
        super::buffers::create_storage_buffer(&device.device, 8, "matmul_batch_params");
    {
        let params_data: [u32; 2] = [m, k];
        device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));
    }

    let bind_group = device.device.create_bind_group(&BindGroupDescriptor {
        label: Some("dequant_matmul_batch_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(weight_buf.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(act_buf.as_entire_buffer_binding()),
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

    match batch_encoder {
        Some(enc) => {
            super::pipeline::dispatch_compute_on_encoder(enc, pipeline, &bind_group, m, batch, 1)
        }
        None => super::pipeline::dispatch_compute(device, pipeline, &bind_group, m, batch, 1),
    }

    output_buf
}
