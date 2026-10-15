//! Headless generation smoke test: `cargo run --release -p hearth-llm --example gen -- <model.gguf> [prompt] [max_tokens] [temp]`
//!
//! Drives the same `Model::run` pipeline the CLI and UI use (so it exercises the
//! hybrid qwen35 core as well as the dense path) and prints the decoded text
//! followed by throughput. Handy for regression-checking every model in
//! `models/` without the UI.
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;
use std::sync::Arc;

use hearth_core::{EngineOutput, LoadStrategy, Model, PipelineRequest, SamplerConfig};
use hearth_llm::LlamaModel;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: gen <model.gguf> [prompt] [max_tokens] [temp]");
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let max_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(24);
    let temp: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let model = Arc::new(LlamaModel::load(&path, LoadStrategy::CpuOnly).expect("load model"));

    let sampler = SamplerConfig {
        temperature: temp,
        ..Default::default()
    };
    let request = PipelineRequest::TextGen {
        prompt: prompt.clone(),
        sampler,
        max_new_tokens: max_tokens,
        max_context_len: None,
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = channel();
    let runner = {
        let model = Arc::clone(&model);
        let cancel = cancel.clone();
        std::thread::spawn(move || model.run(request, cancel, tx))
    };

    let mut text = String::new();
    let mut done = None;
    for ev in rx {
        match ev {
            EngineOutput::TextToken(t) => {
                text.push_str(&t);
                print!("{}", t);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            EngineOutput::Done(stats) => done = Some(stats),
            EngineOutput::Error(e) => eprintln!("\n[error] {:?}", e),
            _ => {}
        }
    }
    runner.join().ok();

    println!();
    match done {
        Some(s) => eprintln!(
            "[{} new tokens ({} prompt), {} ms, {:.2} tok/s]",
            s.tokens, s.prompt_tokens, s.elapsed_ms, s.tokens_per_s
        ),
        None => eprintln!("[no stats]"),
    }
}
