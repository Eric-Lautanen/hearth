//! Renders a GGUF model's chat prompt exactly as the engine will send it, so a
//! template change can be diffed against the renders the model card documents.
//!
//! ```text
//! cargo run -p hearth-tokenizer --example render_prompt -- \
//!     models/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf \
//!     --system "You are a helpful assistant." --user "Hello"
//! ```
//!
//! `--history "user text|assistant text"` can be repeated for earlier turns and
//! `--nothink` renders with `enable_thinking=false`.

use hearth_gguf::GgufFile;
use hearth_tokenizer::Tokenizer;

fn main() -> Result<(), String> {
    let mut path: Option<String> = None;
    let mut system = String::new();
    let mut user = String::new();
    let mut thinking = true;
    let mut history: Vec<(String, String)> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--system" => system = args.next().unwrap_or_default(),
            "--user" => user = args.next().unwrap_or_default(),
            "--nothink" => thinking = false,
            "--history" => {
                let turn = args.next().unwrap_or_default();
                let (question, answer) = turn.split_once('|').unwrap_or((turn.as_str(), ""));
                history.push((question.to_string(), answer.to_string()));
            }
            other if path.is_none() => path = Some(other.to_string()),
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    let path = path.ok_or_else(|| {
        "usage: render_prompt <model.gguf> [--system TEXT] [--user TEXT] \
         [--history \"user|assistant\"] [--nothink]"
            .to_string()
    })?;

    let gguf = GgufFile::open(&path).map_err(|e| format!("{path}: {e:?}"))?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let turns: Vec<(&str, &str)> = history
        .iter()
        .map(|(question, answer)| (question.as_str(), answer.as_str()))
        .collect();

    print!(
        "{}",
        tokenizer.apply_template(&system, &turns, &user, thinking)
    );
    Ok(())
}
