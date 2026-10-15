use hearth_gguf::GgufFile;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_gguf <model.gguf>");
    let gguf = GgufFile::open(&path).expect("open failed");

    println!("=== METADATA ===");
    for (k, v) in &gguf.metadata {
        // Show arrays in full (up to a sane cap) — truncation hid critical info.
        let rendered = format!("{:?}", v);
        if rendered.len() > 6000 {
            println!(
                "  {} = <{} bytes of {:?}...>",
                k,
                rendered.len(),
                &k[..k.len().min(40)]
            );
        } else {
            println!("  {} = {}", k, rendered);
        }
    }

    println!("\n=== TENSORS BY LAYER ===");
    let mut current = String::new();
    for t in &gguf.tensors {
        let layer = t
            .name
            .split('.')
            .nth(1)
            .filter(|s| s.chars().all(|c| c.is_ascii_digit()))
            .map(|s| s.to_string())
            .unwrap_or_default();
        if layer != current {
            current = layer;
            println!("--- blk.{} ---", current);
        }
        println!(
            "  {} shape={:?} dtype={} byte_size={}",
            t.name,
            t.shape,
            t.dtype.name(),
            t.byte_size()
        );
    }
}
