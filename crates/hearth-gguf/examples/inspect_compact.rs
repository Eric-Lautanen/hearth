use hearth_gguf::meta::MetaValue;
use hearth_gguf::GgufFile;
use std::env;

fn dump_val(v: &MetaValue) -> String {
    match v {
        MetaValue::String(s) => format!("\"{}\"", s),
        MetaValue::U8(n) => n.to_string(),
        MetaValue::I8(n) => n.to_string(),
        MetaValue::U16(n) => n.to_string(),
        MetaValue::I16(n) => n.to_string(),
        MetaValue::U32(n) => n.to_string(),
        MetaValue::I32(n) => n.to_string(),
        MetaValue::U64(n) => n.to_string(),
        MetaValue::I64(n) => n.to_string(),
        MetaValue::F32(n) => n.to_string(),
        MetaValue::F64(n) => n.to_string(),
        MetaValue::Bool(b) => b.to_string(),
        MetaValue::Array(_, items) => format!("Array[{}]", items.len()),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: inspect_compact <model.gguf>");
        std::process::exit(1);
    }

    let path = &args[1];
    let gguf = GgufFile::open(path).expect("Failed to open GGUF file");

    println!("=== {} ===", path);
    println!(
        "Version: {}, Tensors: {}, Metadata entries: {}",
        gguf.version,
        gguf.tensors.len(),
        gguf.metadata.len()
    );

    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in keys {
        println!("  {} = {}", key, dump_val(&gguf.metadata[key]));
    }

    println!();
    println!("=== Tensors ({}) ===", gguf.tensors.len());
    for t in &gguf.tensors {
        let shape_str: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        println!(
            "  {:50} {}  shape=[{}]  bytes={}",
            t.name,
            t.dtype.name(),
            shape_str.join(","),
            t.byte_size()
        );
    }
}
