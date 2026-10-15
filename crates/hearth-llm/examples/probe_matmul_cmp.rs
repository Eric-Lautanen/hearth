// Rust mirror of T:/mmprobe.cpp â€” validates my dequant, dot kernels and WHT
// against the C reference values for blk.0.attn_qkv.weight.
use hearth_gguf::{ggml::GgmlDType, GgufFile};

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
        for i in off..off + bs {
            x[i] *= s;
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf".into());
    let gguf = GgufFile::open(&path).expect("open failed");
    let info = gguf.find_tensor("blk.0.attn_qkv.weight").expect("tensor");
    let n_cols = info.shape[0] as usize;
    let n_rows = info.shape[1] as usize;
    eprintln!("rows={} cols={} dtype={:?}", n_rows, n_cols, info.dtype);

    // same deterministic vector as the C probe (f64 math, cast to f32)
    let x: Vec<f32> = (0..n_cols)
        .map(|c| {
            (0.7 * ((c % 997) as f64)).sin() * (1.3 * ((c % 31) as f64)).cos()
                + 0.05 * (((c * 7) % 13) as f64)
        } as f32)
        .collect();
    let mut xr = x.clone();
    wht(&mut xr, 1024);

    // dequantize rows (52M elements is fine)
    let mut w = vec![0.0f32; n_rows * n_cols];
    hearth_quant::dequantize(info.dtype, gguf.tensor_data(info), &mut w).expect("dequant");

    let n_show = 8;
    let mut abs_raw = 0.0f64;
    let mut abs_rot = 0.0f64;
    for r in 0..n_rows {
        let row = &w[r * n_cols..(r + 1) * n_cols];
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        for c in 0..n_cols {
            s1 += row[c] as f64 * x[c] as f64;
            s2 += row[c] as f64 * xr[c] as f64;
        }
        abs_raw += s1.abs();
        abs_rot += s2.abs();
        if r < n_show {
            eprintln!("row{:02} raw={:.5} rot={:.5}", r, s1, s2);
        }
    }
    eprintln!(
        "mean|raw|={:.6} mean|rot|={:.6}",
        abs_raw / n_rows as f64,
        abs_rot / n_rows as f64
    );

    // Also exercise the actual runtime path: quantize x to Q8 and use the dot kernel.
    let mut q8 = Vec::new();
    hearth_quant::quantize_q8_0(&x, &mut q8);
    let row_bytes = info.dtype.byte_size(n_cols);
    let data = gguf.tensor_data(info);
    let dot = hearth_quant::dot_ptq1_0_q8_0_ptr;
    let mut q8_sum = 0.0f64;
    for r in 0..n_show.min(n_rows) {
        let row = &data[r * row_bytes..(r + 1) * row_bytes];
        let y = unsafe { dot(row.as_ptr(), q8.as_ptr(), n_cols) };
        if r < n_show {
            eprintln!("q8row{:02} raw={:.5}", r, y);
        }
        q8_sum += y.abs() as f64;
    }
    eprintln!("(q8 path shown for first {} rows)", n_show);
    let _ = (abs_raw, abs_rot, GgmlDType::Q8_0);
}
