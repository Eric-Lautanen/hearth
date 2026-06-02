use crate::ggml::GgmlDType;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlDType,
    pub dtype_id: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        let mut n: u64 = 1;
        for &d in &self.shape {
            n = n.saturating_mul(d);
        }
        n
    }

    pub fn byte_size(&self) -> usize {
        self.dtype.byte_size(self.element_count() as usize)
    }
}
