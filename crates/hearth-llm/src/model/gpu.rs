use std::collections::HashMap;

use hearth_compute::GpuCompute;
use hearth_gguf::GgmlDType;

use super::tensor::TensorEntry;

pub(crate) fn upload_tensors_to_gpu(gpu: &GpuCompute, tensors: &HashMap<String, TensorEntry>) {
    let mut pool = gpu.pool.lock().unwrap();
    for (name, entry) in tensors {
        let buf = match entry.dtype {
            GgmlDType::F32 => gpu.upload_f32(
                bytemuck::cast_slice(&entry.data),
                &format!("weight:{}", name),
            ),
            GgmlDType::F16 => {
                if gpu.has_matmul_f16() {
                    gpu.upload_f16_packed(&entry.data, &format!("weight:{}", name))
                } else {
                    let n = entry.data.len() / 2;
                    let mut f32_data = vec![0.0f32; n];
                    let src = bytemuck::cast_slice::<u8, half::f16>(&entry.data);
                    for (i, &v) in src.iter().enumerate() {
                        if i >= n {
                            break;
                        }
                        f32_data[i] = v.to_f32();
                    }
                    gpu.upload_f32(&f32_data, &format!("weight:{}", name))
                }
            }
            GgmlDType::BF16 => {
                let n = entry.data.len() / 2;
                let mut f32_data = vec![0.0f32; n];
                for (i, chunk) in entry.data.chunks_exact(2).enumerate() {
                    if i >= n {
                        break;
                    }
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32_data[i] = f32::from_bits((bits as u32) << 16);
                }
                gpu.upload_f32(&f32_data, &format!("weight:{}", name))
            }
            _ => gpu.upload_bytes(&entry.data, &format!("weight:{}", name)),
        };
        pool.insert(name.clone(), buf);
    }
}
