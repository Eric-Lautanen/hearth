use hearth_gguf::GgufFile;

fn main() {
    let models_dir = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into());
    let model_path = format!(
        "{}\\AppData\\Roaming\\hearth\\models\\Bonsai-1.7B-Q1_0.gguf",
        models_dir
    );

    let gguf = GgufFile::open(&model_path).expect("Failed to open");

    // Check a weight tensor's dtype and size
    for t in &gguf.tensors {
        if t.name.contains("blk.0.attn_q.weight") {
            println!("Tensor: {}", t.name);
            println!("  shape: {:?}", t.shape);
            println!("  dtype: {:?} (raw_id={})", t.dtype, t.dtype_id);
            println!("  byte_size: {}", t.byte_size());
            let n_cols = t.shape[0] as usize;
            let n_rows = t.shape[1] as usize;
            println!("  n_cols={} n_rows={}", n_cols, n_rows);

            let data = gguf.tensor_data(t);
            println!("  data.len() = {}", data.len());

            let block_size = t.dtype.block_size();
            let block_bytes = t.dtype.block_bytes();
            println!("  block_size={} block_bytes={}", block_size, block_bytes);

            let expected_bytes = (n_rows * n_cols).div_ceil(block_size) * block_bytes;
            println!("  expected_bytes (row-major scanline) = {}", expected_bytes);

            // Dequantize first row
            let row_bytes = t.dtype.byte_size(n_cols);
            println!("  bytes_per_row = {}", row_bytes);
            let row_data = &data[0..row_bytes];
            println!("  first_row_data.len() = {}", row_data.len());

            // Print first few bytes of each block
            let blocks_per_row = n_cols / block_size;
            for b in 0..blocks_per_row.min(2) {
                let bo = b * block_bytes;
                let scale_lo = row_data[bo] as u16;
                let scale_hi = row_data[bo + 1] as u16;
                let scale_bits = (scale_hi << 8) | scale_lo;
                let scale = half::f16::from_bits(scale_bits).to_f32();
                println!("  block {}: scale={} (bits={:04x})", b, scale, scale_bits);
                // Print first 4 sign bytes
                let mut sign_bits_str = String::new();
                for j in 0..4 {
                    let byte = row_data[bo + 2 + j];
                    sign_bits_str.push_str(&format!("{:08b} ", byte));
                }
                println!("    sign bytes: {}", sign_bits_str);
            }

            // Check: is the data actually Q1_0 format (18-byte blocks) or something else?
            // For Q1_0_G128: 18 bytes per 128 elements
            // For Q2_0: 34 bytes per 128 elements
            let bytes_per_128 = data.len() / (n_rows * n_cols / 128);
            println!("  bytes_per_128_elements = {}", bytes_per_128);
        }
    }
}
