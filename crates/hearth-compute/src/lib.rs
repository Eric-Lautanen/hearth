use hearth_gguf::GgmlDType;
use std::collections::HashMap;

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
    pub pool: std::sync::Mutex<HashMap<String, ()>>,
    pub norm_buffers: HashMap<String, ()>,
    pub kv_cache: Vec<()>,
    pub cache_max_seq: u32,
}

#[allow(unused_variables, clippy::too_many_arguments)]
impl GpuCompute {
    pub async fn new() -> Option<Self> {
        None
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
    pub fn create_storage_buffer(&self, _size: u64, _label: &str) {}

    pub fn upload_f32(&self, _data: &[f32], _label: &str) {}
    pub fn upload_f16_packed(&self, _data: &[u8], _label: &str) {}
    pub fn upload_bytes(&self, _data: &[u8], _label: &str) {}

    pub fn readback_f32(&self, buf: &(), n: usize) -> Vec<f32> {
        vec![]
    }

    pub fn dequant_matmul_fused(
        &self,
        quant_buf: &(),
        kind: &QuantKind,
        x_buf: &(),
        m: u32,
        n: u32,
        k: u32,
    ) -> Option<()> {
        None
    }

    pub fn matmul_f16(&self, w_buf: &(), x_buf: &(), m: u32, n: u32, k: u32) -> Option<()> {
        None
    }

    pub fn mat_vec(&self, w_buf: &(), x_buf: &(), m: u32, n: u32) -> Option<Vec<f32>> {
        None
    }

    pub fn flash_attention(
        &self,
        q_buf: &(),
        kv_buf: &(),
        kv_offset: u64,
        kv_total_bytes: u64,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_len_q: u32,
        seq_len_kv: u32,
        pos: u32,
    ) -> Option<()> {
        None
    }

    pub fn rms_norm(
        &self,
        x_buf: &(),
        w_buf: &(),
        out_buf: &(),
        d: u32,
        eps: f32,
        post_norm: bool,
    ) {
    }

    pub fn head_rms_norm(
        &self,
        x_buf: &(),
        w_buf: &(),
        n_heads: u32,
        head_dim: u32,
        eps: f32,
        post_norm: bool,
    ) {
    }

    pub fn rope_combined(
        &self,
        x_buf: &(),
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
        k_buf: &(),
        v_buf: &(),
        cache_buf: &(),
        pos: u32,
        n_kv_heads: u32,
        head_dim: u32,
        cache_stride: u32,
    ) {
    }

    pub fn dequant_matmul_fused_add_inplace(
        &self,
        quant_buf: &(),
        x_buf: &(),
        accum_buf: &(),
        m: u32,
        n: u32,
    ) -> bool {
        false
    }

    pub fn add_inplace(&self, a: &(), b: &(), n: u32) {}

    pub fn silu_mul(&self, gate: &(), up: &(), out: &(), n: u32) {}

    pub fn fused_gate_up(
        &self,
        gate_w: &(),
        up_w: &(),
        x_buf: &(),
        gate_out: &(),
        up_out: &(),
        m: u32,
        n: u32,
    ) -> bool {
        false
    }

    pub fn fused_qkv(
        &self,
        qw: &(),
        kw: &(),
        vw: &(),
        x: &(),
        q_out: &(),
        k_out: &(),
        v_out: &(),
        m: u32,
        n: u32,
        k: u32,
    ) -> bool {
        false
    }

    pub fn copy_buffer(&self, src: &(), dst: &()) {}
}
