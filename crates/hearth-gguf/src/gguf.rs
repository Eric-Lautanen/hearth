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
