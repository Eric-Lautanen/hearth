use hearth_gguf::GgmlDType;

pub(crate) struct TensorEntry {
    pub(crate) data: Vec<u8>,
    pub(crate) dtype: GgmlDType,
    pub(crate) shape: Vec<u64>,
}

impl TensorEntry {
    pub(crate) fn n_rows(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[1] as usize
        } else {
            1
        }
    }

    pub(crate) fn n_cols(&self) -> usize {
        if !self.shape.is_empty() {
            self.shape[0] as usize
        } else {
            0
        }
    }

    pub(crate) fn row_data(&self, row: usize) -> &[u8] {
        let cols = self.n_cols();
        let row_bytes = self.dtype.byte_size(cols);
        let start = row * row_bytes;
        let end = (start + row_bytes).min(self.data.len());
        &self.data[start..end]
    }

    #[allow(dead_code)]
    pub(crate) fn can_batch(&self) -> bool {
        self.shape.len() >= 2 && self.shape[0] > 1
    }
}
