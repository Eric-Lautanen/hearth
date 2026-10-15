use hearth_gguf::GgufFile;
use hearth_tokenizer::Tokenizer;
fn main() {
    let gguf = GgufFile::open("models/Ternary-Bonsai-2-27B-PTQ1_0.gguf").unwrap();
    let mut tok = Tokenizer::from_gguf(&gguf).unwrap();
    let ids = tok.encode(" Paris", false);
    println!("' Paris' -> {:?}", ids);
    let ids2 = tok.encode("Paris", false);
    println!("'Paris' -> {:?}", ids2);
    // newline and colon
    for s in [" is", " is\n", "\n", ":"] {
        println!("{:?} -> {:?}", s, tok.encode(s, false));
    }
}
