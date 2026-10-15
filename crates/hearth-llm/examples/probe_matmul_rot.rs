// Which transform does the folded weight need? Uses the reference's exact
// layer-0 input (attn_norm-0 dump, which matches ours bit-exactly) and compares
// the resulting in_proj output against the reference's qkv_mixed-0 dump.
use hearth_gguf::GgufFile;

fn wht(x: &mut [f32], bs: usize) {
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

fn load_f32(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64) * (a[i] as f64);
        nb += (b[i] as f64) * (b[i] as f64);
    }
    d / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn signs_for(gguf: &GgufFile, width: usize) -> Vec<f32> {
    use hearth_gguf::MetaValue;
    let widths = gguf.metadata.get("prism.hadamard.sign_widths").unwrap();
    let values = gguf.metadata.get("prism.hadamard.sign_values").unwrap();
    let (MetaValue::Array(_, wi), MetaValue::Array(_, vi)) = (widths, values) else {
        panic!()
    };
    let mut off = 0usize;
    for w in wi {
        let MetaValue::I32(wd) = w else { panic!() };
        let wd = *wd as usize;
        if wd == width {
            return vi[off..off + width]
                .iter()
                .map(|v| match v {
                    MetaValue::I32(s) => *s as f32,
                    _ => panic!(),
                })
                .collect();
        }
        off += wd;
    }
    panic!("no signs for width {}", width)
}

fn main() {
    let gguf = GgufFile::open("models/Ternary-Bonsai-2-27B-PTQ1_0.gguf").unwrap();
    let w = gguf.find_tensor("blk.0.attn_qkv.weight").unwrap();
    let n_in = w.shape[0] as usize;
    let n_out = w.shape[1] as usize;
    let row_bytes = w.dtype.byte_size(n_in);
    let data = gguf.tensor_data(w);
    let signs = signs_for(&gguf, n_in);
    println!(
        "weight {} in={} out={} dtype={:?}",
        w.name, n_in, n_out, w.dtype
    );

    let x = load_f32(r"C:\Users\ericl\AppData\Local\Temp\refx.attn_norm-0.f32");
    let refy = load_f32(r"C:\Users\ericl\AppData\Local\Temp\refx.linear_attn_qkv_mixed-0.f32");
    println!("x n={} refy n={}", x.len(), refy.len());

    let mut variants: Vec<(String, Vec<f32>)> = vec![("raw".into(), x.clone())];
    for bs in [1024usize, 128] {
        let mut a = x.clone();
        for (v, s) in a.iter_mut().zip(signs.iter()) {
            *v *= s;
        }
        wht(&mut a, bs);
        variants.push((format!("signs+wht{}", bs), a));
        let mut b = x.clone();
        wht(&mut b, bs);
        for (v, s) in b.iter_mut().zip(signs.iter()) {
            *v *= s;
        }
        variants.push((format!("wht{}+signs", bs), b));
        let mut c = x.clone();
        wht(&mut c, bs);
        variants.push((format!("wht{}", bs), c));
    }
    {
        let mut s = x.clone();
        for (v, sg) in s.iter_mut().zip(signs.iter()) {
            *v *= sg;
        }
        variants.push(("signs".into(), s));
    }

    let mut row = vec![0.0f32; n_in];
    let mut y = vec![0.0f32; n_out];
    for (name, xr) in &variants {
        for r in 0..n_out {
            hearth_quant::dequantize(w.dtype, &data[r * row_bytes..(r + 1) * row_bytes], &mut row)
                .unwrap();
            let mut acc = 0.0f32;
            for i in 0..n_in {
                acc += row[i] * xr[i];
            }
            y[r] = acc;
        }
        println!(
            "activation-rot {:<14} cos={:.6}  mean|y|={:.5}",
            name,
            cos(&y, &refy),
            y.iter().map(|v| v.abs()).sum::<f32>() / n_out as f32
        );
    }

    // weight-side transforms: rotate each row instead of the activation
    for (wname, sign_first) in [("wrow signs+wht", true), ("wrow wht+signs", false)] {
        for r in 0..n_out {
            hearth_quant::dequantize(w.dtype, &data[r * row_bytes..(r + 1) * row_bytes], &mut row)
                .unwrap();
            if sign_first {
                for (v, s) in row.iter_mut().zip(signs.iter()) {
                    *v *= s;
                }
                wht(&mut row, 1024);
            } else {
                wht(&mut row, 1024);
                for (v, s) in row.iter_mut().zip(signs.iter()) {
                    *v *= s;
                }
            }
            let mut acc = 0.0f32;
            for i in 0..n_in {
                acc += row[i] * x[i];
            }
            y[r] = acc;
        }
        println!(
            "weight-rot    {:<14} cos={:.6}  mean|y|={:.5}",
            wname,
            cos(&y, &refy),
            y.iter().map(|v| v.abs()).sum::<f32>() / n_out as f32
        );
    }
    println!(
        "meany ref = {:.5}",
        refy.iter().map(|v| v.abs()).sum::<f32>() / refy.len() as f32
    );
}
