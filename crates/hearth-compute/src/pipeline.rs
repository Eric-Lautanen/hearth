use wgpu::{ComputePipeline, Device};

use crate::device::GpuDevice;

pub(crate) fn create_compute_pipeline(
    device: &Device,
    shader_source: &str,
    label: &str,
) -> ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Record a compute dispatch on an already-created encoder. Does NOT submit.
/// Used for batching multiple dispatches into one command buffer.
pub(crate) fn dispatch_compute_on_encoder(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group: &wgpu::BindGroup,
    x: u32,
    y: u32,
    z: u32,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, bind_group, &[]);
    cpass.dispatch_workgroups(x, y, z);
}

/// Create a temporary encoder, record one dispatch, submit immediately.
/// Used when no batch is active (legacy path).
pub(crate) fn dispatch_compute(
    device: &GpuDevice,
    pipeline: &ComputePipeline,
    bind_group: &wgpu::BindGroup,
    x: u32,
    y: u32,
    z: u32,
) {
    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    dispatch_compute_on_encoder(&mut encoder, pipeline, bind_group, x, y, z);
    device.queue.submit(Some(encoder.finish()));
}
