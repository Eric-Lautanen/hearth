// Weight stats probe: mean|x| and max|x| via hearth_quant dequantize (mirror of T:/wstats.cpp).
use hearth_gguf::GgufFile;

fn stats(gguf: &GgufFile, name: &str) {
    let info = match gguf.find_tensor(name) {
        Some(t) => t,
        None => {
            eprintln!("{:28} NOT FOUND", name);
            return;
        }
    };
    let n = info.shape.iter().product::<u64>() as usize;
    let mut buf = vec![0.0f32; n];
    if let Err(e) = hearth_quant::dequantize(info.dtype, gguf.tensor_data(info), &mut buf) {
        eprintln!("{:28} dequant error: {}", name, e);
        return;
    }
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    for &v in &buf {
        let a = v.abs() as f64;
        sum += a;
        if v.abs() > max {
            max = v;
        }
    }
    eprintln!(
        "{:28} dtype={:?} ne={:8} mean={:.6} max={:.4}",
        name,
        info.dtype,
        n,
        sum / n as f64,
        max
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf".into());
    let gguf = GgufFile::open(&path).expect("open failed");
    stats(&gguf, "token_embd.weight");
    stats(&gguf, "blk.0.attn_qkv.weight");
    stats(&gguf, "blk.0.attn_gate.weight");
    stats(&gguf, "blk.0.ssm_beta.weight");
    stats(&gguf, "blk.0.ssm_alpha.weight");
    stats(&gguf, "blk.0.ssm_conv1d.weight");
    stats(&gguf, "blk.0.ssm_out.weight");
    stats(&gguf, "blk.0.ffn_gate.weight");
    stats(&gguf, "output.weight");
}
