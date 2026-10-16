// Brute-force search over rows 0..250000 + rotations to find which embedding row
// matches the fork's attn_norm-0 dump.
use hearth_gguf::{GgufFile, MetaValue};

fn wht(x: &mut [f32], bs: usize) {
    let s = 1.0 / (bs as f32).sqrt();
    for off in (0..x.len()).step_by(bs) {
        let mut h = 1usize;
        while h < bs {
            for i in off..off + bs {
                if (i - off) & h != 0 {
                    continue;
                }
                let a = x[i];
                let b = x[i | h];
                x[i] = a + b;
                x[i | h] = a - b;
            }
            h <<= 1;
        }
        for slot in &mut x[off..off + bs] {
            *slot *= s;
        }
    }
}

fn main() {
    let gguf = GgufFile::open("models/Ternary-Bonsai-2-27B-PTQ1_0.gguf").unwrap();
    let emb = gguf.find_tensor("token_embd.weight").unwrap();
    let d = emb.shape[0] as usize;
    let row_bytes = emb.dtype.byte_size(d);
    let data = gguf.tensor_data(emb);

    // signs
    let widths = gguf.metadata.get("prism.hadamard.sign_widths").unwrap();
    let values = gguf.metadata.get("prism.hadamard.sign_values").unwrap();
    let (MetaValue::Array(_, witems), MetaValue::Array(_, vitems)) = (widths, values) else {
        panic!()
    };
    let mut signs: Vec<f32> = Vec::new();
    let mut off = 0usize;
    for w in witems {
        let MetaValue::I32(width) = w else { panic!() };
        let width = *width as usize;
        if width == 5120 {
            for v in &vitems[off..off + width] {
                let MetaValue::I32(s) = v else { panic!() };
                signs.push(*s as f32);
            }
            break;
        }
        off += width;
    }

    // ref dump (normed) â€” drop the norm weight by comparing pre-norm instead:
    // norm is elementwise positive scaling, so cos(a*w, b*w) == cos(a, b) if we
    // multiply both sides by w. Instead, de-norm the ref with the norm weight.
    let nw = gguf.find_tensor("blk.0.attn_norm.weight").unwrap();
    let mut w = vec![0.0f32; d];
    hearth_quant::dequantize(nw.dtype, gguf.tensor_data(nw), &mut w).unwrap();
    let rdata = std::fs::read(r"C:\Users\ericl\AppData\Local\Temp\.attn_norm-0.f32").unwrap();
    // as_chunks::<4> needs Rust 1.88+; the remainder (short tail) is dropped,
    // same as chunks_exact did.
    let (rchunks, _remainder) = rdata.as_chunks::<4>();
    let r: Vec<f32> = rchunks.iter().map(|c| f32::from_le_bytes(*c)).collect();
    // de-norm (divide by w, where w ~= 0 breaks)
    let mut rdn = vec![0.0f32; d];
    for i in 0..d {
        rdn[i] = if w[i].abs() > 1e-6 { r[i] / w[i] } else { 0.0 };
    }

    let mut z = vec![0.0f32; d];
    // scan all rows, rotation 1 (signs->wht) and 0
    let mut best1 = (0usize, -2.0f64);
    let mut best0 = (0usize, -2.0f64);
    for tok in 0..(data.len() / row_bytes) {
        hearth_quant::dequantize(
            emb.dtype,
            &data[tok * row_bytes..(tok + 1) * row_bytes],
            &mut z,
        )
        .unwrap();
        for (rot, best) in [(1usize, &mut best1), (0usize, &mut best0)] {
            let mut zz = z.clone();
            if rot == 1 {
                for (v, s) in zz.iter_mut().zip(signs.iter()) {
                    *v *= s;
                }
                wht(&mut zz, 1024);
            }
            let dot: f64 = zz
                .iter()
                .zip(rdn.iter())
                .map(|(&x, &y)| x as f64 * y as f64)
                .sum();
            let na: f64 = zz
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = rdn
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .sqrt();
            let c = dot / (na * nb + 1e-30);
            if c > best.1 {
                *best = (tok, c);
            }
        }
    }
    eprintln!(
        "best rot=1 (signs->WHT): tok={} cos={:.4}",
        best1.0, best1.1
    );
    eprintln!("best rot=0 (raw):       tok={} cos={:.4}", best0.0, best0.1);
}
