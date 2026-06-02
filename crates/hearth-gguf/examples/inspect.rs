use hearth_gguf::meta::MetaValue;
use hearth_gguf::GgufFile;
use std::env;

fn dump_value(v: &MetaValue) -> String {
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
        MetaValue::Array(_, items) => {
            let vs: Vec<String> = items.iter().map(dump_value).collect();
            format!("[{}]", vs.join(", "))
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: inspect <model.gguf>");
        std::process::exit(1);
    }

    let path = &args[1];
    let gguf = GgufFile::open(path).expect("Failed to open GGUF file");

    println!("=== GGUF File: {} ===", path);
    println!("Version:      {}", gguf.version);
    println!("Alignment:    {}", gguf.alignment);
    println!("Data offset:  0x{:x}", gguf.data_offset);
    println!("Tensor count: {}", gguf.tensors.len());
    println!();

    println!("=== All Metadata ({}) ===", gguf.metadata.len());
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in keys {
        let val = &gguf.metadata[key];
        println!("  {} = {}", key, dump_value(val));
    }

    println!();
    println!("=== Tensors ===");
    for t in &gguf.tensors {
        let shape_str: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        let elems = t.element_count();
        println!(
            "  {:50} {}  shape=[{}]  elems={}  bytes={}  offset=0x{:x}",
            t.name,
            t.dtype.name(),
            shape_str.join(","),
            elems,
            t.byte_size(),
            t.offset,
        );
    }
}
