use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

pub(crate) fn create_storage_buffer(device: &Device, size: u64, label: &str) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub(crate) fn upload_to_buffer(device: &Device, queue: &Queue, data: &[u8], label: &str) -> Buffer {
    let buffer = create_storage_buffer(device, data.len() as u64, label);
    queue.write_buffer(&buffer, 0, data);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    buffer
}

pub(crate) fn download_from_buffer(
    device: &Device,
    queue: &Queue,
    buffer: &Buffer,
    size: u64,
) -> Vec<u8> {
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("download-staging"),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let _ = rx.recv();

    let data = slice.get_mapped_range();
    let result = data.to_vec();
    drop(data);
    staging.unmap();

    result
}
