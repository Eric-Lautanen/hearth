use wgpu::{Adapter, Device, DeviceDescriptor, Instance, Queue};

pub(crate) struct GpuDevice {
    #[allow(dead_code)]
    pub instance: Instance,
    #[allow(dead_code)]
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl GpuDevice {
    pub async fn new() -> Option<Self> {
        let instance = Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("hearth-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .ok()?;

        eprintln!(
            "[gpu] device={:?} backend={:?}",
            adapter.get_info().name,
            adapter.get_info().backend,
        );

        Some(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}
