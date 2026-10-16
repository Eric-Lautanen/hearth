// Tests which (token, rotation variant) reproduces the reference fork's
// layer-0 attn_norm dump. If a variant matches ~1.0, the embedding inverse is
// pinned down exactly; if none do, the token/id path is the divergence.
use hearth_gguf::{GgufFile, MetaValue};

fn wht(x: &mut [f32], bs: usize) {
    if bs == 0 || bs > x.len() || !x.len().is_multiple_of(bs) {
        return;
    }
    let s = 1.0 / (bs as f32).sqrt();
    for off in (0..x.len()).step_by(bs) {
        let mut h = 1usize;
        while h < bs {
            let mut i = off;
            while i < off + bs {
                for j in 0..h {
                    let a = x[i + j];
                    let b = x[i + j + h];
                    x[i + j] = a + b;
                    x[i + j + h] = a - b;
                }
                i += 2 * h;
            }
            h <<= 1;
        }
        for i in off..off + bs {
            x[i] *= s;
        }
    }
}

fn load_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).expect("read dump");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut d = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64) * (a[i] as f64);
        nb += (b[i] as f64) * (b[i] as f64);
    }
    d / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn main() {
    let gguf = GgufFile::open("models/Ternary-Bonsai-2-27B-PTQ1_0.gguf").unwrap();
    println!(
        "prism.hadamard.block_size = {:?}",
        gguf.metadata.get("prism.hadamard.block_size")
    );
    let emb = gguf.find_tensor("token_embd.weight").unwrap();
    let d = emb.shape[0] as usize;
    let row_bytes = emb.dtype.byte_size(d);
    let data = gguf.tensor_data(emb);

    // signs for width d
    let widths = gguf.metadata.get("prism.hadamard.sign_widths").unwrap();
    let values = gguf.metadata.get("prism.hadamard.sign_values").unwrap();
    let (MetaValue::Array(_, witems), MetaValue::Array(_, vitems)) = (widths, values) else {
        panic!("bad metadata")
    };
    let mut signs: Vec<f32> = Vec::new();
    let mut widths_seen: Vec<usize> = Vec::new();
    let mut off = 0usize;
    for w in witems {
        let MetaValue::I32(width) = w else { panic!() };
        let width = *width as usize;
        widths_seen.push(width);
        if width == d {
            for v in &vitems[off..off + width] {
                let MetaValue::I32(s) = v else { panic!() };
                signs.push(*s as f32);
            }
        }
        off += width;
    }
    println!(
        "sign widths = {:?}, got signs for d={} : {}",
        widths_seen,
        d,
        signs.len()
    );

    // norm weight
    let nw = gguf.find_tensor("blk.0.attn_norm.weight").unwrap();
    let mut w = vec![0.0f32; d];
    hearth_quant::dequantize(nw.dtype, gguf.tensor_data(nw), &mut w).unwrap();

    let refx = load_f32(r"C:\Users\ericl\AppData\Local\Temp\refx.attn_norm-0.f32");
    println!(
        "ref n={} mean={:.5}",
        refx.len(),
        refx.iter().map(|v| v.abs()).sum::<f32>() / refx.len() as f32
    );

    let block = gguf
        .metadata
        .get("prism.hadamard.block_size")
        .and_then(|v| match v {
            MetaValue::U32(x) => Some(*x as usize),
            MetaValue::I32(x) => Some(*x as usize),
            _ => None,
        })
        .unwrap_or(0);
    let mut blocks: Vec<usize> = vec![block];
    for bs in [64usize, 128, 256, 512, 1024] {
        if bs != block && d.is_multiple_of(bs) {
            blocks.push(bs);
        }
    }

    let tokens: Vec<usize> = vec![9419, 248044, 248045, 0, 1];
    let mut z = vec![0.0f32; d];
    let mut out = vec![0.0f32; d];
    let mut best_overall = (String::new(), -2.0f64);

    for &tok in &tokens {
        if tok >= data.len() / row_bytes {
            continue;
        }
        hearth_quant::dequantize(
            emb.dtype,
            &data[tok * row_bytes..(tok + 1) * row_bytes],
            &mut z,
        )
        .unwrap();
        let mut variants: Vec<(String, Vec<f32>)> = vec![("raw".into(), z.clone())];
        for &bs in &blocks {
            let mut a = z.clone();
            for (v, s) in a.iter_mut().zip(signs.iter()) {
                *v *= s;
            }
            wht(&mut a, bs);
            variants.push((format!("signs+wht{}", bs), a));
            let mut b = z.clone();
            wht(&mut b, bs);
            for (v, s) in b.iter_mut().zip(signs.iter()) {
                *v *= s;
            }
            variants.push((format!("wht{}+signs", bs), b));
        }
        {
            let mut c = z.clone();
            for (v, s) in c.iter_mut().zip(signs.iter()) {
                *v *= s;
            }
            variants.push(("signs_only".into(), c));
        }
        println!("== token {}", tok);
        for (name, v) in variants {
            hearth_llm::ops::rms_norm(&v, &w, 1e-6, &mut out);
            let c = cos(&out, &refx);
            if c > best_overall.1 {
                best_overall = (format!("tok={} {}", tok, name), c);
            }
            if c > 0.5 {
                println!("   MATCH {:<18} cos={:.5}", name, c);
            }
        }
    }
    println!("best: {} cos={:.5}", best_overall.0, best_overall.1);
}
