use half::f16;

const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

pub enum KVStorage {
    F32,
    Q8_0,
}

pub struct KVCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub k_q8: Vec<u8>,
    pub v_q8: Vec<u8>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub current_len: usize,
    storage: KVStorage,
    dequant_buf: Vec<f32>,
}

impl KVCache {
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        KVCache {
            k: Vec::new(),
            v: Vec::new(),
            k_q8: Vec::new(),
            v_q8: Vec::new(),
            n_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
            storage: KVStorage::F32,
            dequant_buf: Vec::new(),
        }
    }

    pub fn new_q8_0(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let total_positions = n_kv_heads * max_seq_len;
        let blocks_per_position = head_dim.div_ceil(Q8_0_BLOCK_SIZE);
        let q8_bytes = total_positions * blocks_per_position * Q8_0_BLOCK_BYTES;
        KVCache {
            k: Vec::new(),
            v: Vec::new(),
            k_q8: vec![0u8; q8_bytes],
            v_q8: vec![0u8; q8_bytes],
            n_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
            storage: KVStorage::Q8_0,
            dequant_buf: vec![0.0f32; head_dim],
        }
    }

    pub fn new_with_storage(
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        storage: KVStorage,
    ) -> Self {
        match storage {
            KVStorage::F32 => Self::new(n_kv_heads, head_dim, max_seq_len),
            KVStorage::Q8_0 => Self::new_q8_0(n_kv_heads, head_dim, max_seq_len),
        }
    }

    pub fn is_q8_0(&self) -> bool {
        matches!(self.storage, KVStorage::Q8_0)
    }

    pub fn memory_bytes(&self) -> usize {
        let per_entry = self.n_kv_heads * self.max_seq_len;
        match self.storage {
            KVStorage::F32 => per_entry * self.head_dim * 4 * 2,
            KVStorage::Q8_0 => {
                let blocks = self.head_dim.div_ceil(Q8_0_BLOCK_SIZE);
                per_entry * blocks * Q8_0_BLOCK_BYTES * 2
            }
        }
    }

    fn q8_position_offset(&self, head: usize, pos: usize) -> usize {
        let blocks_per_pos = self.head_dim.div_ceil(Q8_0_BLOCK_SIZE);
        head * self.max_seq_len * blocks_per_pos * Q8_0_BLOCK_BYTES
            + pos * blocks_per_pos * Q8_0_BLOCK_BYTES
    }

    pub fn write_kv(&mut self, pos: usize, head: usize, k: &[f32], v: &[f32]) {
        match self.storage {
            KVStorage::F32 => {
                if self.k.len() < self.n_kv_heads * self.head_dim * self.max_seq_len {
                    let total = self.n_kv_heads * self.head_dim * self.max_seq_len;
                    self.k.resize(total, 0.0f32);
                    self.v.resize(total, 0.0f32);
                }
                let offset = head * self.head_dim * self.max_seq_len + pos * self.head_dim;
                self.k[offset..offset + self.head_dim].copy_from_slice(k);
                self.v[offset..offset + self.head_dim].copy_from_slice(v);
            }
            KVStorage::Q8_0 => {
                let off = self.q8_position_offset(head, pos);
                quantize_q8_0_block(k, &mut self.k_q8[off..]);
                quantize_q8_0_block(v, &mut self.v_q8[off..]);
            }
        }
        if pos >= self.current_len {
            self.current_len = pos + 1;
        }
    }

    pub fn k_slice(&self, head: usize, seq_len: usize) -> &[f32] {
        match self.storage {
            KVStorage::F32 => {
                let start = head * self.head_dim * self.max_seq_len;
                let end = start + seq_len * self.head_dim;
                &self.k[start..end]
            }
            KVStorage::Q8_0 => {
                unreachable!("Use k_slice_dequant for Q8_0 cache")
            }
        }
    }

    pub fn v_slice(&self, head: usize, seq_len: usize) -> &[f32] {
        match self.storage {
            KVStorage::F32 => {
                let start = head * self.head_dim * self.max_seq_len;
                let end = start + seq_len * self.head_dim;
                &self.v[start..end]
            }
            KVStorage::Q8_0 => {
                unreachable!("Use v_slice_dequant for Q8_0 cache")
            }
        }
    }

    pub fn k_slice_dequant(&mut self, head: usize, seq_len: usize) -> &[f32] {
        match self.storage {
            KVStorage::F32 => {
                let start = head * self.head_dim * self.max_seq_len;
                let end = start + seq_len * self.head_dim;
                &self.k[start..end]
            }
            KVStorage::Q8_0 => {
                let needed = seq_len * self.head_dim;
                if self.dequant_buf.len() < needed {
                    self.dequant_buf.resize(needed, 0.0f32);
                }
                for pos in 0..seq_len {
                    let off = self.q8_position_offset(head, pos);
                    let blocks = self.head_dim.div_ceil(Q8_0_BLOCK_SIZE);
                    let total_bytes = blocks * Q8_0_BLOCK_BYTES;
                    dequantize_q8_0_block(
                        &self.k_q8[off..off + total_bytes],
                        &mut self.dequant_buf[pos * self.head_dim..(pos + 1) * self.head_dim],
                        self.head_dim,
                    );
                }
                &self.dequant_buf[..needed]
            }
        }
    }

    pub fn v_slice_dequant(&mut self, head: usize, seq_len: usize) -> &[f32] {
        match self.storage {
            KVStorage::F32 => {
                let start = head * self.head_dim * self.max_seq_len;
                let end = start + seq_len * self.head_dim;
                &self.v[start..end]
            }
            KVStorage::Q8_0 => {
                let needed = seq_len * self.head_dim;
                if self.dequant_buf.len() < needed {
                    self.dequant_buf.resize(needed, 0.0f32);
                }
                for pos in 0..seq_len {
                    let off = self.q8_position_offset(head, pos);
                    let blocks = self.head_dim.div_ceil(Q8_0_BLOCK_SIZE);
                    let total_bytes = blocks * Q8_0_BLOCK_BYTES;
                    dequantize_q8_0_block(
                        &self.v_q8[off..off + total_bytes],
                        &mut self.dequant_buf[pos * self.head_dim..(pos + 1) * self.head_dim],
                        self.head_dim,
                    );
                }
                &self.dequant_buf[..needed]
            }
        }
    }

    pub fn clear(&mut self) {
        self.current_len = 0;
    }

    pub fn truncate_left(&mut self, keep: usize) {
        if self.current_len <= keep {
            return;
        }
        let shift = self.current_len - keep;
        match self.storage {
            KVStorage::F32 => {
                for head in 0..self.n_kv_heads {
                    let base = head * self.head_dim * self.max_seq_len;
                    for dst_pos in 0..keep {
                        let src_pos = dst_pos + shift;
                        if src_pos < self.current_len {
                            let dst_off = base + dst_pos * self.head_dim;
                            let src_off = base + src_pos * self.head_dim;
                            self.k
                                .copy_within(src_off..src_off + self.head_dim, dst_off);
                            self.v
                                .copy_within(src_off..src_off + self.head_dim, dst_off);
                        }
                    }
                }
            }
            KVStorage::Q8_0 => {
                let blocks_per_pos = self.head_dim.div_ceil(Q8_0_BLOCK_SIZE);
                let bytes_per_pos = blocks_per_pos * Q8_0_BLOCK_BYTES;
                for head in 0..self.n_kv_heads {
                    let base = self.q8_position_offset(head, 0);
                    for dst_pos in 0..keep {
                        let src_pos = dst_pos + shift;
                        if src_pos < self.current_len {
                            let dst_off = base + dst_pos * bytes_per_pos;
                            let src_off = base + src_pos * bytes_per_pos;
                            self.k_q8
                                .copy_within(src_off..src_off + bytes_per_pos, dst_off);
                            self.v_q8
                                .copy_within(src_off..src_off + bytes_per_pos, dst_off);
                        }
                    }
                }
            }
        }
        self.current_len = keep;
    }
}

fn quantize_q8_0_block(values: &[f32], dst: &mut [u8]) {
    let n = values.len();
    let blocks = n.div_ceil(Q8_0_BLOCK_SIZE);
    for b in 0..blocks {
        let start = b * Q8_0_BLOCK_SIZE;
        let mut max_abs: f32 = 0.0;
        for i in 0..Q8_0_BLOCK_SIZE {
            let idx = start + i;
            if idx < n {
                max_abs = max_abs.max(values[idx].abs());
            }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let scale_f16 = f16::from_f32(scale);
        let dst_off = b * Q8_0_BLOCK_BYTES;
        dst[dst_off] = scale_f16.to_le_bytes()[0];
        dst[dst_off + 1] = scale_f16.to_le_bytes()[1];
        for i in 0..Q8_0_BLOCK_SIZE {
            let idx = start + i;
            let q = if idx < n {
                (values[idx] / scale).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
            dst[dst_off + 2 + i] = q as u8;
        }
    }
}

fn dequantize_q8_0_block(data: &[u8], out: &mut [f32], n: usize) {
    let blocks = n.div_ceil(Q8_0_BLOCK_SIZE);
    for b in 0..blocks {
        let bp = b * Q8_0_BLOCK_BYTES;
        let d_f32 = f16::from_le_bytes([data[bp], data[bp + 1]]).to_f32();
        let vals_start = bp + 2;
        let base = b * Q8_0_BLOCK_SIZE;
        for i in 0..Q8_0_BLOCK_SIZE {
            let idx = base + i;
            if idx < n {
                out[idx] = d_f32 * (data[vals_start + i] as i8) as f32;
            }
        }
    }
}
