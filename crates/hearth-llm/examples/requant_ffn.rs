//! Convert a PTQ1_0 hybrid's ffn tensors to 1-bit Q1_0_G128, to measure what the
//! 5-level ternary -> 1-bit drop costs.
//!
//! The repo cannot do this today: `hearth_gguf` only parses (no writer), and
//! `hearth_quant::quantize_f32` covers Q8_0 and F32 only, so there is no Q1_0
//! quantizer outside `hearth-compute`'s test module. This example sidesteps the
//! missing writer: it copies the source file, rewrites each converted tensor's
//! payload at its existing offset with the smaller Q1_0 row layout, and patches
//! that tensor's dtype id (143 -> 43) in the header. Regions keep their extent,
//! so every other offset stays valid; the tail of each converted region keeps
//! stale bytes that no reader looks at, because a tensor's byte length comes
//! from its dtype and shape, not from the gap to the next tensor.
//!
//! Usage: `requant_ffn <src.gguf> <dst.gguf> [all]`
//! Without `all`, only tensors whose name contains `ffn_` are converted — that
//! is the shape of the recommendation (ffn in Q1_0, everything else unchanged).

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};

use half::f16;
use hearth_gguf::ggml::GgmlDType;
use memmap2::Mmap;

const PTQ1_0_ELEMS: usize = 128;
const PTQ1_0_BYTES: usize = 28;
const Q1_0_BYTES: usize = 18;
const ID_PTQ1_0: u32 = 143;
const ID_Q1_0_G128: u32 = 43;

struct TensorInfo {
    name: String,
    shape: Vec<u64>,
    dtype_id: u32,
    /// File offset of this tensor's dtype field, for patching it.
    dtype_pos: u64,
    /// Offset of the payload relative to `data_offset`.
    offset: u64,
}

fn u32_at(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]])
}

fn u64_at(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes([
        b[p],
        b[p + 1],
        b[p + 2],
        b[p + 3],
        b[p + 4],
        b[p + 5],
        b[p + 6],
        b[p + 7],
    ])
}

fn read_str(b: &[u8], pos: &mut usize) -> Result<String, String> {
    let len = u64_at(b, *pos) as usize;
    *pos += 8;
    let s = String::from_utf8_lossy(&b[*pos..*pos + len]).into_owned();
    *pos += len;
    Ok(s)
}

/// Skip one metadata value of `type_id`, returning the alignment if it is
/// `general.alignment`.
fn skip_value(b: &[u8], pos: &mut usize, type_id: u32) -> Result<Option<u64>, String> {
    let fixed = match type_id {
        0 | 1 | 7 => Some(1), // u8, i8, bool
        2 | 3 => Some(2),     // u16, i16
        4..=6 => Some(4),     // u32, i32, f32
        10..=12 => Some(8),   // u64, i64, f64
        _ => None,
    };
    if let Some(n) = fixed {
        let v = if n == 4 {
            Some(u32_at(b, *pos) as u64)
        } else {
            None
        };
        *pos += n;
        return Ok(v);
    }
    match type_id {
        8 => {
            read_str(b, pos)?;
            Ok(None)
        }
        9 => {
            let elem_type = u32_at(b, *pos);
            *pos += 4;
            let len = u64_at(b, *pos) as usize;
            *pos += 8;
            for _ in 0..len {
                skip_value(b, pos, elem_type)?;
            }
            Ok(None)
        }
        other => Err(format!("unknown metadata value type {other}")),
    }
}

fn parse(b: &[u8]) -> Result<(u64, Vec<TensorInfo>), String> {
    if &b[0..4] != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    let version = u32_at(b, 4);
    if version != 2 && version != 3 {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let mut pos = 8usize;
    let n_tensors = u64_at(b, pos) as usize;
    pos += 8;
    let n_kv = u64_at(b, pos) as usize;
    pos += 8;

    let mut alignment = 32u64;
    for _ in 0..n_kv {
        let key = read_str(b, &mut pos)?;
        let type_id = u32_at(b, pos);
        pos += 4;
        let v = skip_value(b, &mut pos, type_id)?;
        if key == "general.alignment" {
            if let Some(a) = v {
                alignment = a;
            }
        }
    }

    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = read_str(b, &mut pos)?;
        let n_dims = u32_at(b, pos) as usize;
        pos += 4;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u64_at(b, pos));
            pos += 8;
        }
        let dtype_pos = pos as u64;
        let dtype_id = u32_at(b, pos);
        pos += 4;
        let offset = u64_at(b, pos);
        pos += 8;
        tensors.push(TensorInfo {
            name,
            shape,
            dtype_id,
            dtype_pos,
            offset,
        });
    }

    let data_offset = if version == 3 {
        let pad = (alignment - (pos as u64 % alignment)) % alignment;
        pos as u64 + pad
    } else {
        pos as u64
    };
    Ok((data_offset, tensors))
}

/// Q1_0_G128 block: f16 scale, then 16 bytes of signs, 1 = +scale.
/// `scale` is chosen as the mean absolute value, which is the right per-block
/// scale for ternary weights: max-abs would push every one of PTQ1_0's zeros to
/// a full-scale weight.
fn quantize_q1_0_row(src: &[f32], dst: &mut [u8], scale: f32) {
    for (blk, chunk) in src.as_chunks::<PTQ1_0_ELEMS>().0.iter().enumerate() {
        let out = &mut dst[blk * Q1_0_BYTES..(blk + 1) * Q1_0_BYTES];
        out[0..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
        out[2..].fill(0);
        for (i, &v) in chunk.iter().enumerate() {
            if v >= 0.0 {
                out[2 + i / 8] |= 1 << (i % 8);
            }
        }
    }
}

fn mean_abs(v: &[f32]) -> f32 {
    let sum: f32 = v.iter().map(|x| x.abs()).sum();
    let m = sum / v.len() as f32;
    if m == 0.0 {
        1.0
    } else {
        m
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: requant_ffn <src.gguf> <dst.gguf> [all]".into());
    }
    let all = args.iter().any(|a| a == "all");
    let src_path = &args[0];
    let dst_path = &args[1];

    let src = File::open(src_path).map_err(|e| format!("open {src_path}: {e}"))?;
    let map = unsafe { Mmap::map(&src) }.map_err(|e| format!("mmap: {e}"))?;
    let (data_offset, tensors) = parse(&map)?;

    // Region extent per tensor: the gap to the next payload, so the rewrite stays
    // inside it and every offset in the header remains valid.
    let mut starts: Vec<(u64, usize)> = tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.offset, i))
        .collect();
    starts.sort_unstable();

    std::fs::copy(src_path, dst_path).map_err(|e| format!("copy: {e}"))?;
    let mut dst = OpenOptions::new()
        .write(true)
        .open(dst_path)
        .map_err(|e| format!("open {dst_path} for write: {e}"))?;

    let mut total_err = 0.0f64;
    let mut total_ref = 0.0f64;
    let mut converted = 0usize;
    let mut zeros = 0u64;
    let mut count = 0u64;

    for (rank, &(offset, idx)) in starts.iter().enumerate() {
        let t = &tensors[idx];
        if t.dtype_id != ID_PTQ1_0 {
            continue;
        }
        if !all && !t.name.contains("ffn_") {
            continue;
        }
        let cols = t.shape[0] as usize;
        if !cols.is_multiple_of(PTQ1_0_ELEMS) {
            return Err(format!("{}: cols {cols} not a multiple of 128", t.name));
        }
        let numel: usize = t.shape.iter().map(|d| *d as usize).product();
        let rows = numel / cols;
        let old_row = cols / PTQ1_0_ELEMS * PTQ1_0_BYTES;
        let new_row = cols / PTQ1_0_ELEMS * Q1_0_BYTES;
        let base = data_offset as usize + offset as usize;
        let region = match starts.get(rank + 1) {
            Some(&(next, _)) => (next - offset) as usize,
            None => map.len() - base,
        };
        if rows * new_row > region {
            return Err(format!(
                "{}: {} -> {} bytes does not fit {}",
                t.name,
                rows * old_row,
                rows * new_row,
                region
            ));
        }

        let mut abs_sum = 0.0f64;
        let mut abs_count = 0u64;
        let mut t_err = 0.0f64;
        let mut t_ref = 0.0f64;
        let mut f32buf = vec![0.0f32; cols];
        let mut q = vec![0u8; rows * new_row];
        for r in 0..rows {
            let row = &map[base + r * old_row..base + (r + 1) * old_row];
            hearth_quant::dequantize(GgmlDType::PTQ1_0, row, &mut f32buf)
                .map_err(|e| format!("{}: {e}", t.name))?;
            let scale = mean_abs(&f32buf);
            quantize_q1_0_row(&f32buf, &mut q[r * new_row..(r + 1) * new_row], scale);

            for &v in &f32buf {
                abs_sum += (v as f64).abs();
                abs_count += 1;
                if v == 0.0 {
                    zeros += 1;
                }
            }
            count += f32buf.len() as u64;
            // Reconstruct and accumulate the relative L2 error.
            let mut back = vec![0.0f32; cols];
            hearth_quant::dequantize(
                GgmlDType::Q10G128,
                &q[r * new_row..(r + 1) * new_row],
                &mut back,
            )
            .map_err(|e| format!("{}: {e}", t.name))?;
            for (a, b) in back.iter().zip(f32buf.iter()) {
                t_err += ((a - b) as f64).powi(2);
                t_ref += (*b as f64).powi(2);
            }
        }

        dst.seek(SeekFrom::Start(base as u64))
            .map_err(|e| format!("seek: {e}"))?;
        dst.write_all(&q)
            .map_err(|e| format!("write payload: {e}"))?;

        dst.seek(SeekFrom::Start(t.dtype_pos))
            .map_err(|e| format!("seek: {e}"))?;
        dst.write_all(&ID_Q1_0_G128.to_le_bytes())
            .map_err(|e| format!("write dtype: {e}"))?;

        total_err += t_err;
        total_ref += t_ref;
        converted += 1;
        println!(
            "{:<24} {:>5} rows x {:>5}  {:>8.1} MB -> {:>8.1} MB  relL2 {:.4}  mean|w| {:.3e}",
            t.name,
            rows,
            cols,
            (rows * old_row) as f64 / 1e6,
            (rows * new_row) as f64 / 1e6,
            (t_err / t_ref.max(f64::MIN_POSITIVE)).sqrt(),
            abs_sum / abs_count.max(1) as f64
        );
    }

    println!(
        "\n{converted} tensors converted  ·  {:.1}% of weights are exact zeros (the level 1-bit cannot carry)",
        100.0 * zeros as f64 / count.max(1) as f64
    );
    println!(
        "aggregate relative L2 error {:.4}",
        (total_err / total_ref.max(f64::MIN_POSITIVE)).sqrt()
    );
    dst.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(())
}
