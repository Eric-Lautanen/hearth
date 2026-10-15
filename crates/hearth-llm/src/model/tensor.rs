use std::sync::Arc;

use hearth_gguf::GgmlDType;
use memmap2::Mmap;

/// Backing storage for a tensor.
///
/// Small tensors are copied into an owned buffer (cheap, keeps hot weights off
/// the page cache), while large tensors borrow directly from the GGUF mmap so a
/// multi-gigabyte model can load on a machine with far less RAM than the file.
pub(crate) enum TensorStorage {
    Owned(Vec<u8>),
    Mapped {
        mmap: Arc<Mmap>,
        start: usize,
        len: usize,
    },
}

pub(crate) struct TensorEntry {
    pub(crate) storage: TensorStorage,
    pub(crate) dtype: GgmlDType,
    pub(crate) shape: Vec<u64>,
}

impl TensorEntry {
    pub(crate) fn owned(data: Vec<u8>, dtype: GgmlDType, shape: Vec<u64>) -> Self {
        TensorEntry {
            storage: TensorStorage::Owned(data),
            dtype,
            shape,
        }
    }

    pub(crate) fn mapped(
        mmap: Arc<Mmap>,
        start: usize,
        len: usize,
        dtype: GgmlDType,
        shape: Vec<u64>,
    ) -> Self {
        TensorEntry {
            storage: TensorStorage::Mapped { mmap, start, len },
            dtype,
            shape,
        }
    }

    #[inline]
    pub(crate) fn data(&self) -> &[u8] {
        match &self.storage {
            TensorStorage::Owned(v) => v,
            TensorStorage::Mapped { mmap, start, len } => &mmap[*start..*start + *len],
        }
    }

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
        let end = (start + row_bytes).min(self.data().len());
        &self.data()[start..end]
    }

    #[allow(dead_code)]
    pub(crate) fn can_batch(&self) -> bool {
        self.shape.len() >= 2 && self.shape[0] > 1
    }
}
