// Which activation transform does the folded ssm_out weight need?
// Input: reference final_output-0 (bit-exact match with ours).
// Target: reference linear_attn_out-0.
use hearth_gguf::{GgufFile, MetaValue};

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

/// Second check: is the residual FFN error a real math bug or just activation
/// quantization precision? Compares an f32 brute-force ffn_down (using our own
/// silu(gate)*up vector, which matches the reference bit-exactly) against the
/// reference's `ffn_out` — which is the FFN branch output, before the residual
/// add (so it must not have `attn_residual` subtracted from it).
fn check_ffn(gguf: &GgufFile) {
    let tmp = std::env::var("TEMP").unwrap_or_else(|_| "C:/Users/ericl/AppData/Local/Temp".into());
    let p = |n: &str| format!("{}\\{}", tmp, n);
    let x = match std::fs::read(p("ffn_down_in-0.f32")) {
        Ok(b) => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>(),
        Err(_) => {
            println!("ffn check skipped (no ffn_down_in dump)");
            return;
        }
    };
    let ref_branch = load_f32(&p("refx.ffn_out-0.f32"));

    let w = gguf.find_tensor("blk.0.ffn_down.weight").unwrap();
    let n_in = w.shape[0] as usize;
    let n_out = w.shape[1] as usize;
    let row_bytes = w.dtype.byte_size(n_in);
    let data = gguf.tensor_data(w);
    let signs = signs_for(gguf, n_in);
    println!(
        "ffn_down: in={} out={} | x n={} ref_branch n={}",
        n_in,
        n_out,
        x.len(),
        ref_branch.len()
    );

    // our runtime input: signs+WHT(1024) rotation
    let mut xr = x.clone();
    for (v, s) in xr.iter_mut().zip(signs.iter()) {
        *v *= s;
    }
    wht(&mut xr, 1024);

    let mut q8: Vec<u8> = Vec::new();
    hearth_quant::quantize_q8_0(&xr, &mut q8);
    let mut row = vec![0.0f32; n_in];
    let mut y32 = vec![0.0f32; n_out];
    let mut yq = vec![0.0f32; n_out];
    for r in 0..n_out {
        hearth_quant::dequantize(w.dtype, &data[r * row_bytes..(r + 1) * row_bytes], &mut row)
            .unwrap();
        y32[r] = (0..n_in).map(|i| row[i] * xr[i]).sum();
        yq[r] = unsafe {
            hearth_quant::dot_ptq1_0_q8_0_ptr(data.as_ptr().add(r * row_bytes), q8.as_ptr(), n_in)
        };
    }
    println!(
        "ffn_down f32-brute  cos={:.6} mean|y|={:.5}",
        cos(&y32, &ref_branch),
        y32.iter().map(|v| v.abs()).sum::<f32>() / n_out as f32
    );
    println!(
        "ffn_down q8-kernel cos={:.6} mean|y|={:.5}",
        cos(&yq, &ref_branch),
        yq.iter().map(|v| v.abs()).sum::<f32>() / n_out as f32
    );
    println!(
        "reference branch    mean|y|={:.5}",
        ref_branch.iter().map(|v| v.abs()).sum::<f32>() / ref_branch.len() as f32
    );
}

fn main() {
    let gguf = GgufFile::open("models/Ternary-Bonsai-2-27B-PTQ1_0.gguf").unwrap();
    check_ffn(&gguf);
    let w = gguf.find_tensor("blk.0.ssm_out.weight").unwrap();
    let n_in = w.shape[0] as usize;
    let n_out = w.shape[1] as usize;
    let row_bytes = w.dtype.byte_size(n_in);
    let data = gguf.tensor_data(w);
    let signs = signs_for(&gguf, n_in);
    println!("ssm_out in={} out={} dtype={:?}", n_in, n_out, w.dtype);
    println!(
        "gdn_v_grouped = {:?}",
        gguf.metadata.get("prism.hadamard.gdn_v_grouped")
    );
    println!(
        "block_size = {:?}",
        gguf.metadata.get("prism.hadamard.block_size")
    );

    let x = load_f32(r"C:\Users\ericl\AppData\Local\Temp\refx.final_output-0.f32");
    let target = load_f32(r"C:\Users\ericl\AppData\Local\Temp\refx.linear_attn_out-0.f32");
    println!("x n={} target n={}", x.len(), target.len());

    // head layout guesses: n_v v-heads of hd each; nk k-groups; rep = n_v/nk
    let hd = 128usize;
    let n_v = n_in / hd;
    let mut variants: Vec<(String, Vec<f32>)> = Vec::new();
    for nk in [16usize, 12, 24, 48, 8, 4, 1] {
        if n_v % nk != 0 {
            continue;
        }
        let rep = n_v / nk;
        if rep == 1 && nk != n_v {
            continue;
        }
        // tiled -> grouped  (our current direction)
        let mut a = vec![0.0f32; n_in];
        for h in 0..n_v {
            let kg = h % nk;
            let r = h / nk;
            let dst = (kg * rep + r) * hd;
            a[dst..dst + hd].copy_from_slice(&x[h * hd..(h + 1) * hd]);
        }
        variants.push((format!("nk{} tiled->grouped", nk), a));
        // grouped -> tiled (inverse)
        let mut b = vec![0.0f32; n_in];
        for h in 0..n_v {
            let kg = h % nk;
            let r = h / nk;
            let src = (kg * rep + r) * hd;
            b[h * hd..(h + 1) * hd].copy_from_slice(&x[src..src + hd]);
        }
        variants.push((format!("nk{} grouped->tiled", nk), b));
    }
    variants.push(("identity".into(), x.clone()));

    let mut row = vec![0.0f32; n_in];
    let mut y = vec![0.0f32; n_out];
    let mut results: Vec<(String, f64, f32)> = Vec::new();
    for (name, v) in &variants {
        for (rotname, rot) in [("signs+wht", 0usize), ("wht+signs", 1), ("raw", 2)] {
            let mut a = v.clone();
            match rot {
                0 => {
                    for (p, s) in a.iter_mut().zip(signs.iter()) {
                        *p *= s;
                    }
                    wht(&mut a, 1024);
                }
                1 => {
                    wht(&mut a, 1024);
                    for (p, s) in a.iter_mut().zip(signs.iter()) {
                        *p *= s;
                    }
                }
                _ => {}
            }
            for r in 0..n_out {
                hearth_quant::dequantize(
                    w.dtype,
                    &data[r * row_bytes..(r + 1) * row_bytes],
                    &mut row,
                )
                .unwrap();
                let mut acc = 0.0f32;
                for i in 0..n_in {
                    acc += row[i] * a[i];
                }
                y[r] = acc;
            }
            let c = cos(&y, &target);
            let mean = y.iter().map(|v| v.abs()).sum::<f32>() / n_out as f32;
            results.push((format!("{:<22} {}", name, rotname), c, mean));
        }
    }
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, c, mean) in results.iter().take(8) {
        println!("{} cos={:.5} mean|y|={:.5}", name, c, mean);
    }
    println!(
        "target mean = {:.5}",
        target.iter().map(|v| v.abs()).sum::<f32>() / target.len() as f32
    );

    // ---- check the runtime's own rotated activation ----
    let rp = r"C:\Users\ericl\AppData\Local\Temp\ssm_rot_in-0.f32";
    if std::path::Path::new(rp).exists() {
        let rt = load_f32(rp);
        println!(
            "runtime rot_in n={} mean|.|={:.5}",
            rt.len(),
            rt.iter().map(|v| v.abs()).sum::<f32>() / rt.len() as f32
        );
        // expected rotated input from the reference's final_output
        let nk = 16usize;
        let rep = n_v / nk;
        let mut a = vec![0.0f32; n_in];
        for h in 0..n_v {
            let kg = h % nk;
            let r = h / nk;
            let dst = (kg * rep + r) * hd;
            a[dst..dst + hd].copy_from_slice(&x[h * hd..(h + 1) * hd]);
        }
        for (p, s) in a.iter_mut().zip(signs.iter()) {
            *p *= s;
        }
        wht(&mut a, 1024);
        println!("runtime-vs-expected rot_in cos = {:.6}", cos(&rt, &a));
        // dot the runtime activation against the weights
        for r in 0..n_out {
            hearth_quant::dequantize(w.dtype, &data[r * row_bytes..(r + 1) * row_bytes], &mut row)
                .unwrap();
            let mut acc = 0.0f32;
            for i in 0..n_in.min(rt.len()) {
                acc += row[i] * rt[i];
            }
            y[r] = acc;
        }
        println!(
            "runtime rot_in -> y cos = {:.6} mean|y|={:.5}",
            cos(&y, &target),
            y.iter().map(|v| v.abs()).sum::<f32>() / n_out as f32
        );

        // ---- kernel path: quantize the activation then use the q8 dot ----
        let mut q8: Vec<u8> = Vec::new();
        hearth_quant::quantize_q8_0(&rt, &mut q8);
        println!("q8 bytes = {} (expected {})", q8.len(), n_in / 32 * 34);
        let wbase = data.as_ptr();
        for r in 0..6 {
            hearth_quant::dequantize(w.dtype, &data[r * row_bytes..(r + 1) * row_bytes], &mut row)
                .unwrap();
            let brute: f32 = (0..n_in).map(|i| row[i] * rt[i]).sum();
            let kern = unsafe {
                hearth_quant::dot_ptq1_0_q8_0_ptr(wbase.add(r * row_bytes), q8.as_ptr(), n_in)
            };
            println!(
                "  row {} brute={:.5} kernel={:.5} diff={:.4}",
                r,
                brute,
                kern,
                (kern - brute).abs()
            );
        }
    }
}
