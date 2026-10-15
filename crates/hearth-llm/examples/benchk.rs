// Raw throughput of the quantized dot kernels: how many GB/s of weights can a
// single thread (and the full row set) process?
use hearth_gguf::{GgmlDType, GgufFile};
use std::time::Instant;

fn main() {
    let gguf = GgufFile::open("models/Ternary-Bonsai-2-27B-PTQ1_0.gguf").unwrap();
    let names: Vec<String> = gguf
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("blk.0.") && t.dtype == GgmlDType::PTQ1_0)
        .map(|t| t.name.clone())
        .collect();
    println!("{} PTQ1_0 tensors in blk.0", names.len());

    let mut total_bytes = 0usize;
    let mut total_secs = 0.0f64;
    let mut total_flops = 0usize;

    for name in &names {
        let t = gguf.find_tensor(name).unwrap();
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;
        let row_bytes = t.dtype.byte_size(n_in);
        let data = gguf.tensor_data(t);
        let bytes = row_bytes * n_out;

        // activation: deterministic pseudo-random, then q8
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 2654435761usize % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let mut q8: Vec<u8> = Vec::new();
        hearth_quant::quantize_q8_0(&x, &mut q8);

        // warm
        let mut acc = 0.0f32;
        for r in 0..8 {
            acc += unsafe {
                hearth_quant::dot_ptq1_0_q8_0_ptr(
                    data.as_ptr().add(r * row_bytes),
                    q8.as_ptr(),
                    n_in,
                )
            };
        }

        let reps = 3;
        let t0 = Instant::now();
        for _ in 0..reps {
            for r in 0..n_out {
                acc += unsafe {
                    hearth_quant::dot_ptq1_0_q8_0_ptr(
                        data.as_ptr().add(r * row_bytes),
                        q8.as_ptr(),
                        n_in,
                    )
                };
            }
        }
        let secs = t0.elapsed().as_secs_f64() / reps as f64;
        let gbs = bytes as f64 / secs / 1e9;
        println!(
            "{:26} out={:5} in={:5} {:7.2} MB  {:8.3} s  {:6.2} GB/s (1 thread)  acc={:.2}",
            name,
            n_out,
            n_in,
            bytes as f64 / 1e6,
            secs,
            gbs,
            acc
        );
        total_bytes += bytes;
        total_secs += secs;
        total_flops += n_out * n_in;
    }
    println!(
        "blk.0 total: {:.2} GB, {:.3} s -> {:.2} GB/s single-thread",
        total_bytes as f64 / 1e9,
        total_secs,
        total_bytes as f64 / total_secs / 1e9
    );
    println!(
        "layer-0 MACs: {:.2} G; model (64 layers) ~ {:.1} GB weights",
        total_flops as f64 / 1e9,
        total_bytes as f64 * 64.0 / 1e9
    );
}
