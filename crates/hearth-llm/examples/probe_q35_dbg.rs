// Vocab-style probe for any model.
use hearth_core::LoadStrategy;
use hearth_llm::LlamaModel;
fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let model = LlamaModel::load(&path, LoadStrategy::CpuOnly).expect("load");
    let toks = model
        .tokenizer()
        .lock()
        .unwrap()
        .encode("You are\nwhat", false);
    let tok = model.tokenizer().lock().unwrap();
    let t = &tok.token_to_id;
    eprintln!("file: {}", path);
    eprintln!(
        "  'Ġthe'={} ' the'={} 'Ġ'={} 'Ċ'={} '\n'={}",
        t.contains_key("Ġthe"),
        t.contains_key(" the"),
        t.contains_key("Ġ"),
        t.contains_key("Ċ"),
        t.contains_key("\n")
    );
    eprintln!("  'You are\nwhat' -> {:?}", toks);
}
