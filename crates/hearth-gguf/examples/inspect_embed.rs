// Minimal test: check if GPU token embedding dequantize matches CPU
use hearth_gguf::GgufFile;
use hearth_quant::dequantize;

fn main() {
    let models_dir = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into());
    let model_path = format!(
        "{}\\AppData\\Roaming\\hearth\\models\\Bonsai-1.7B-Q1_0.gguf",
        models_dir
    );

    let gguf = GgufFile::open(&model_path).expect("Failed to open");

    // Get token_embd.weight
    for t in &gguf.tensors {
        if t.name == "token_embd.weight" {
            println!("token_embd.weight: shape={:?} dtype={:?}", t.shape, t.dtype);
            // t.shape is [n_cols, n_rows] in GGUF order; only the row width is
            // needed here (the tensor is read one row at a time).
            let n_cols = t.shape[0] as usize; // 2048

            // Dequantize first row (token 0)
            let data = gguf.tensor_data(t);
            let row_bytes = t.dtype.byte_size(n_cols);
            let row_data = &data[0..row_bytes];

            let mut out = vec![0.0f32; n_cols];
            dequantize(t.dtype, row_data, &mut out).expect("dequantize failed");

            println!("First 10 values of token_embd[0]:");
            for (i, v) in out.iter().enumerate().take(10) {
                println!("  [{}] = {:.6}", i, v);
            }
            println!("...");
            println!("Last 10 values of token_embd[0]:");
            for (i, v) in out.iter().enumerate().skip(n_cols - 10) {
                println!("  [{}] = {:.6}", i, v);
            }

            // Also check: sum and norm
            let sum: f32 = out.iter().sum();
            let sum_sq: f32 = out.iter().map(|x| x * x).sum();
            println!("sum={:.4} sum_sq={:.4}", sum, sum_sq);
        }
    }

    // Check blk.0.attn_q.weight first row
    for t in &gguf.tensors {
        if t.name == "blk.0.attn_q.weight" {
            println!(
                "\nblk.0.attn_q.weight: shape={:?} dtype={:?}",
                t.shape, t.dtype
            );
            let n_cols = t.shape[0] as usize;
            let data = gguf.tensor_data(t);
            let row_bytes = t.dtype.byte_size(n_cols);
            let row_data = &data[0..row_bytes];

            let mut out = vec![0.0f32; n_cols];
            dequantize(t.dtype, row_data, &mut out).expect("dequantize failed");

            println!("First 10 values of attn_q[0]:");
            for (i, v) in out.iter().enumerate().take(10) {
                println!("  [{}] = {:.6}", i, v);
            }
            let sum_sq: f32 = out.iter().map(|x| x * x).sum();
            println!("sum_sq={:.4}", sum_sq);
        }
    }
}
