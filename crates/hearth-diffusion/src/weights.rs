use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const MAGIC: u32 = 0x30513248;

#[derive(Debug, Clone)]
pub struct Q2Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct F32Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Bf16Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum StoredTensor {
    Q2(Q2Tensor),
    F32(F32Tensor),
    Bf16(Bf16Tensor),
}

pub struct ModelWeights {
    pub tensors: HashMap<String, StoredTensor>,
}

fn read_u16(r: &mut impl Read) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_bytes(r: &mut impl Read, n: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn load_weights(path: &Path) -> std::io::Result<ModelWeights> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);
    let mut tensors = HashMap::new();

    let magic = read_u32(&mut r)?;
    if magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad magic: {:#x}", magic),
        ));
    }

    let n_quant = read_u32(&mut r)? as usize;
    for _ in 0..n_quant {
        let name_len = read_u16(&mut r)? as usize;
        let name = String::from_utf8(read_bytes(&mut r, name_len)?)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let rows = read_u64(&mut r)? as usize;
        let cols = read_u64(&mut r)? as usize;
        let byte_len = read_u32(&mut r)? as usize;
        let data = read_bytes(&mut r, byte_len)?;
        tensors.insert(name, StoredTensor::Q2(Q2Tensor { rows, cols, data }));
    }

    let n_skip = read_u32(&mut r)? as usize;
    for _ in 0..n_skip {
        let name_len = read_u16(&mut r)? as usize;
        let name = String::from_utf8(read_bytes(&mut r, name_len)?)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let rows = read_u64(&mut r)? as usize;
        let cols = read_u64(&mut r)? as usize;
        let dtype = read_bytes(&mut r, 1)?[0];
        let byte_len = read_u32(&mut r)? as usize;
        let data = read_bytes(&mut r, byte_len)?;
        match dtype {
            0 => {
                let f32_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                tensors.insert(
                    name,
                    StoredTensor::F32(F32Tensor {
                        rows,
                        cols,
                        data: f32_data,
                    }),
                );
            }
            1 => {
                tensors.insert(name, StoredTensor::Bf16(Bf16Tensor { rows, cols, data }));
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown dtype: {}", dtype),
                ));
            }
        }
    }

    let end_magic = read_u32(&mut r)?;
    if end_magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad end magic",
        ));
    }

    Ok(ModelWeights { tensors })
}

impl ModelWeights {
    pub fn q2(&self, name: &str) -> Option<&Q2Tensor> {
        match self.tensors.get(name) {
            Some(StoredTensor::Q2(t)) => Some(t),
            _ => None,
        }
    }

    pub fn f32(&self, name: &str) -> Option<&F32Tensor> {
        match self.tensors.get(name) {
            Some(StoredTensor::F32(t)) => Some(t),
            _ => None,
        }
    }

    pub fn bf16_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        match self.tensors.get(name) {
            Some(StoredTensor::Bf16(t)) => {
                let mut out = Vec::with_capacity(t.data.len() / 2);
                for chunk in t.data.chunks_exact(2) {
                    let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(f32::from_bits((raw as u32) << 16));
                }
                Some(out)
            }
            _ => None,
        }
    }
}
