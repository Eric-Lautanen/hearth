use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::ggml::GgmlDType;
use crate::meta::MetaValue;
use crate::tensor::TensorInfo;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF magic: expected GGUF, got {0:?}")]
    InvalidMagic([u8; 4]),
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("File too small")]
    FileTooSmall,
    #[error("Unexpected end of file at offset {0}")]
    UnexpectedEof(usize),
    #[error("Unknown dtype id {0} for tensor {1}")]
    UnknownDtype(u32, String),
    #[error(
        "tensor layout mismatch on {name}: the file gives it {actual} bytes but {dtype} {shape} needs \
         {expected}, so the file stores {actual_per_block} bytes per {block}-element block where \
         {dtype} assumes {expected_per_block}. Decoding it would misalign every block after the \
         first, so this file is refused rather than run."
    )]
    TensorLayoutMismatch {
        name: String,
        dtype: &'static str,
        shape: String,
        expected: usize,
        actual: u64,
        block: usize,
        expected_per_block: usize,
        actual_per_block: f64,
    },
    #[error("tensor {name} is truncated: {actual} bytes remain after offset {offset} but {dtype} needs {expected}")]
    TensorTruncated {
        name: String,
        dtype: &'static str,
        offset: u64,
        expected: usize,
        actual: u64,
    },
    #[error("{0}")]
    Other(String),
}

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, GgufError> {
    if *pos >= data.len() {
        return Err(GgufError::UnexpectedEof(*pos));
    }
    let v = data[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, GgufError> {
    if *pos + 2 > data.len() {
        return Err(GgufError::UnexpectedEof(*pos));
    }
    let bytes: [u8; 2] = data[*pos..*pos + 2].try_into().unwrap();
    *pos += 2;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, GgufError> {
    if *pos + 4 > data.len() {
        return Err(GgufError::UnexpectedEof(*pos));
    }
    let bytes: [u8; 4] = data[*pos..*pos + 4].try_into().unwrap();
    *pos += 4;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(data: &[u8], pos: &mut usize) -> Result<i32, GgufError> {
    if *pos + 4 > data.len() {
        return Err(GgufError::UnexpectedEof(*pos));
    }
    let bytes: [u8; 4] = data[*pos..*pos + 4].try_into().unwrap();
    *pos += 4;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, GgufError> {
    if *pos + 8 > data.len() {
        return Err(GgufError::UnexpectedEof(*pos));
    }
    let bytes: [u8; 8] = data[*pos..*pos + 8].try_into().unwrap();
    *pos += 8;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(data: &[u8], pos: &mut usize) -> Result<i64, GgufError> {
    if *pos + 8 > data.len() {
        return Err(GgufError::UnexpectedEof(*pos));
    }
    let bytes: [u8; 8] = data[*pos..*pos + 8].try_into().unwrap();
    *pos += 8;
    Ok(i64::from_le_bytes(bytes))
}

fn read_f32(data: &[u8], pos: &mut usize) -> Result<f32, GgufError> {
    Ok(f32::from_bits(read_u32(data, pos)?))
}

fn read_f64(data: &[u8], pos: &mut usize) -> Result<f64, GgufError> {
    Ok(f64::from_bits(read_u64(data, pos)?))
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, GgufError> {
    let len = read_u64(data, pos)? as usize;
    if *pos + len > data.len() {
        return Err(GgufError::UnexpectedEof(*pos));
    }
    let s = String::from_utf8_lossy(&data[*pos..*pos + len]).to_string();
    *pos += len;
    Ok(s)
}

fn read_meta_value(data: &[u8], pos: &mut usize, type_id: u32) -> Result<MetaValue, GgufError> {
    Ok(match type_id {
        0 => MetaValue::U8(read_u8(data, pos)?),
        1 => MetaValue::I8(read_u8(data, pos)? as i8),
        2 => MetaValue::U16(read_u16(data, pos)?),
        3 => MetaValue::I16(read_u16(data, pos)? as i16),
        4 => MetaValue::U32(read_u32(data, pos)?),
        5 => MetaValue::I32(read_i32(data, pos)?),
        6 => MetaValue::F32(read_f32(data, pos)?),
        7 => MetaValue::Bool(read_u8(data, pos)? != 0),
        8 => MetaValue::String(read_string(data, pos)?),
        9 => {
            let elem_type = read_u32(data, pos)?;
            let count = read_u64(data, pos)?;
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_meta_value(data, pos, elem_type)?);
            }
            MetaValue::Array(elem_type, items)
        }
        10 => MetaValue::U64(read_u64(data, pos)?),
        11 => MetaValue::I64(read_i64(data, pos)?),
        12 => MetaValue::F64(read_f64(data, pos)?),
        _ => MetaValue::String(format!("<unknown_type_{}>", type_id)),
    })
}

/// Type 42 names two incompatible layouts: the Prism fork packs 128 weights into a 34-byte
/// block, stock llama.cpp packs 64 into an 18-byte one. The Ternary-Bonsai 2.5 `Q2_0` file is
/// the stock layout, and reading it as the fork's gave 36 bytes per 128 weights, which decoded
/// as garbage. The id cannot tell them apart, so classify by the footprint the file actually
/// gives each tensor; the stride check then confirms the guess.
fn resolve_q2_0_group_size(
    tensors: &mut [TensorInfo],
    file_len: u64,
    data_offset: u64,
    alignment: u64,
) {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&i| tensors[i].offset);

    for (pos, &i) in order.iter().enumerate() {
        if tensors[i].dtype != GgmlDType::Q2_0 {
            continue;
        }
        let stride = match order.get(pos + 1) {
            Some(&next) => tensors[next].offset.saturating_sub(tensors[i].offset),
            None => file_len.saturating_sub(data_offset + tensors[i].offset),
        };
        let elems = tensors[i].element_count() as usize;
        let grouped_64 = GgmlDType::Q2_0G64.byte_size(elems) as u64;
        let grouped_128 = tensors[i].byte_size() as u64;
        let fits = |want: u64| stride >= want && stride - want < alignment;

        // Only reinterpret when the stock layout fits and the fork layout does not. A tensor
        // whose footprint fits neither keeps type 42 and is refused by the stride check.
        if fits(grouped_64) && !fits(grouped_128) {
            tensors[i].dtype = GgmlDType::Q2_0G64;
        }
    }
}

/// Every tensor occupies the gap between its own offset and the next tensor's, padded to
/// `alignment`. If a file lays its blocks out at a different width than its dtype claims — a
/// `Q2_0` pack written at 36 bytes per 128-element block, say — the first block of each tensor
/// still decodes and every block after it is read from the wrong place, so the model runs at full
/// speed and emits well-shaped garbage. The arithmetic is cheap, so check it before decoding.
fn verify_tensor_layout(
    tensors: &[TensorInfo],
    file_len: u64,
    data_offset: u64,
    alignment: u64,
) -> Result<(), GgufError> {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&i| tensors[i].offset);

    for pair in order.windows(2) {
        let (cur, next) = (&tensors[pair[0]], &tensors[pair[1]]);
        check_tensor_stride(cur, next.offset - cur.offset, alignment)?;
    }

    if let Some(&last) = order.last() {
        let info = &tensors[last];
        let start = data_offset + info.offset;
        let available = file_len.saturating_sub(start);
        let expected = info.byte_size() as u64;
        if expected > 0 && available < expected {
            return Err(GgufError::TensorTruncated {
                name: info.name.clone(),
                dtype: info.dtype.name(),
                offset: start,
                expected: expected as usize,
                actual: available,
            });
        }
    }

    Ok(())
}

fn check_tensor_stride(info: &TensorInfo, gap: u64, alignment: u64) -> Result<(), GgufError> {
    let elems = info.element_count() as usize;
    let expected = info.byte_size();
    if elems == 0 || expected == 0 {
        return Ok(());
    }

    // Writers pad each tensor up to the file's alignment, so the gap is the tensor's own size or
    // up to one alignment step more than it — never less.
    let slack = gap as i64 - expected as i64;
    if (0..alignment as i64).contains(&slack) {
        return Ok(());
    }

    let block = info.dtype.block_size();
    Err(GgufError::TensorLayoutMismatch {
        name: info.name.clone(),
        dtype: info.dtype.name(),
        shape: format!("{:?}", info.shape),
        expected,
        actual: gap,
        block,
        expected_per_block: info.dtype.block_bytes(),
        actual_per_block: gap as f64 * block as f64 / elems as f64,
    })
}

pub struct GgufFile {
    mmap: Arc<Mmap>,
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub alignment: u64,
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let mmap = Arc::new(mmap);
        let data = mmap.as_ref();

        if data.len() < 4 {
            return Err(GgufError::FileTooSmall);
        }

        let magic: [u8; 4] = [data[0], data[1], data[2], data[3]];
        if &magic != b"GGUF" {
            return Err(GgufError::InvalidMagic(magic));
        }

        let mut pos = 4;

        let version = read_u32(data, &mut pos)?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = read_u64(data, &mut pos)? as usize;
        let metadata_kv_count = read_u64(data, &mut pos)? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(data, &mut pos)?;
            let type_id = read_u32(data, &mut pos)?;
            let value = read_meta_value(data, &mut pos, type_id)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|v| v as u64)
            .unwrap_or(32);

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(data, &mut pos)?;
            let n_dims = read_u32(data, &mut pos)? as usize;

            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u64(data, &mut pos)?);
            }

            let dtype_id = read_u32(data, &mut pos)?;
            let offset = read_u64(data, &mut pos)?;

            let dtype = GgmlDType::from_id(dtype_id)
                .ok_or_else(|| GgufError::UnknownDtype(dtype_id, name.clone()))?;

            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                dtype_id,
                offset,
            });
        }

        let data_offset = if version == 3 {
            let padding = (alignment - (pos as u64 % alignment)) % alignment;
            pos += padding as usize;
            pos as u64
        } else {
            pos as u64
        };

        resolve_q2_0_group_size(&mut tensors, data.len() as u64, data_offset, alignment);
        verify_tensor_layout(&tensors, data.len() as u64, data_offset, alignment)?;

        Ok(GgufFile {
            mmap,
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }

    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = self.data_offset as usize + info.offset as usize;
        let end = start + info.byte_size();
        &self.mmap[start..end]
    }

    /// Shared handle to the backing mmap, for zero-copy tensor storage that
    /// outlives a borrow of `self`.
    pub fn mmap_arc(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }

    pub fn meta_array(&self, key: &str) -> Option<&[MetaValue]> {
        self.metadata.get(key).and_then(|v| v.as_array())
    }

    pub fn meta_bool(&self, key: &str) -> Option<bool> {
        self.metadata.get(key).and_then(|v| v.as_bool())
    }
}
