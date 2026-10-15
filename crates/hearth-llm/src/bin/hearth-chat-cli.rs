use std::env;
use std::io::{self, Write};
use std::sync::{atomic::AtomicBool, mpsc, Arc};

use hearth_core::{EngineOutput, LoadStrategy, Model, PipelineRequest, SamplerConfig, ThinkFilter};
use hearth_gguf::GgufFile;
use hearth_llm::LlamaModel;

// Service mode implementation lives in a sibling folder (bin crate roots
// resolve `mod` against src/bin/, so the path is given explicitly).
#[path = "hearth-chat-cli/service.rs"]
mod service;

/// Sampler settings passed on the command line. Every field is optional: an
/// unset field falls back to the loaded model's own `general.sampling.*`
/// recommendation, then to the built-in default. This matches how the service
/// (and therefore the web UI) resolves an omitted protocol field.
#[derive(Default)]
struct SamplerOverrides {
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    typical_p: Option<f32>,
    repeat_pen: Option<f32>,
    seed: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::SamplerOverrides;
    use hearth_core::SamplerConfig;

    /// What a model declares in its `general.sampling.*` metadata.
    fn model_defaults() -> SamplerConfig {
        SamplerConfig {
            temperature: 0.5,
            top_k: 20,
            top_p: 0.85,
            min_p: 0.05,
            typical_p: 0.0,
            repeat_pen: 1.05,
            seed: 7,
        }
    }

    /// No flags: the model's values are used verbatim (not the built-in
    /// 0.7/0.9/40 profile this code used to hardcode).
    #[test]
    fn no_flags_keeps_model_defaults() {
        let merged = SamplerOverrides::default().apply(model_defaults());
        assert_eq!(merged.temperature, 0.5);
        assert_eq!(merged.top_k, 20);
        assert_eq!(merged.top_p, 0.85);
        assert_eq!(merged.min_p, 0.05);
        assert_eq!(merged.repeat_pen, 1.05);
        assert_eq!(merged.seed, 7);
        assert!(!SamplerOverrides::default().any());
    }

    /// The regression this guards: `--temp 0` used to discard the model's
    /// top-k/top-p/min-p along with it.
    #[test]
    fn one_flag_overrides_only_its_own_field() {
        let overrides = SamplerOverrides {
            temperature: Some(0.0),
            ..Default::default()
        };
        assert!(overrides.any());
        let merged = overrides.apply(model_defaults());
        assert_eq!(merged.temperature, 0.0);
        assert_eq!(merged.top_k, 20, "model top_k must survive --temp");
        assert_eq!(merged.top_p, 0.85, "model top_p must survive --temp");
        assert_eq!(merged.min_p, 0.05, "model min_p must survive --temp");
    }
}

impl SamplerOverrides {
    fn any(&self) -> bool {
        self.temperature.is_some()
            || self.top_k.is_some()
            || self.top_p.is_some()
            || self.min_p.is_some()
            || self.typical_p.is_some()
            || self.repeat_pen.is_some()
            || self.seed.is_some()
    }

    /// Flag values win field-by-field; everything else comes from `base`
    /// (the model's declared defaults).
    fn apply(&self, base: SamplerConfig) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_k: self.top_k.unwrap_or(base.top_k),
            top_p: self.top_p.unwrap_or(base.top_p),
            min_p: self.min_p.unwrap_or(base.min_p),
            typical_p: self.typical_p.unwrap_or(base.typical_p),
            repeat_pen: self.repeat_pen.unwrap_or(base.repeat_pen),
            seed: self.seed.unwrap_or(base.seed),
        }
    }
}

struct CliArgs {
    model_path: String,
    strategy: LoadStrategy,
    overrides: SamplerOverrides,
    max_tokens: usize,
    max_context: Option<usize>,
    prompt: Option<String>,
    prompt_raw: bool,
    kv_f32: bool,
    dump_info: bool,
    show_think: bool,
    pin_weights: bool,
    service: bool,
}

fn print_usage() {
    eprintln!("Usage: hearth-chat-cli <model.gguf> [options]");
    eprintln!();
    eprintln!(
        "Sampling (each falls back to the model's general.sampling.*, else the built-in default):"
    );
    eprintln!("  --temp T           Temperature (built-in default 0.7)");
    eprintln!("  --top-k K          Top-K (40)");
    eprintln!("  --top-p P          Top-P (0.9)");
    eprintln!("  --min-p P          Min-P (0.0)");
    eprintln!("  --typical-p P      Typical-P (0.0 = disabled)");
    eprintln!("  --repeat-pen R     Repeat penalty (1.1)");
    eprintln!("  --seed N           RNG seed (0 = random per run)");
    eprintln!();
    eprintln!("Generation:");
    eprintln!("  --max-tokens N     Max new tokens per reply (512)");
    eprintln!("  --max-context N    Context window for prompt + reply (model's limit)");
    eprintln!("  --show-think       Show <think> reasoning instead of hiding it");
    eprintln!("  --prompt TEXT      Single-shot prompt (non-interactive)");
    eprintln!("  --prompt-raw       Send the prompt as-is, without a chat template");
    eprintln!();
    eprintln!("Loading:");
    eprintln!("  --gpu              Load all layers on GPU (default: cpu)");
    eprintln!("  --gpu-layers N     Load first N layers on GPU, rest on CPU");
    eprintln!("  --pin-weights      Read weights into RAM at load instead of mmap");
    eprintln!();
    eprintln!("Other:");
    eprintln!("  --service          Run as a line-JSON stdio service (used by hearth-ui)");
    eprintln!("  --dump-info        Print the GGUF metadata and layer-0 tensor dtypes, then exit");
    eprintln!(
        "  --kv-f32           F32 KV cache (accepted for compatibility; F32 is the only backend)"
    );
    eprintln!("  -h, --help         Show this help");
    eprintln!();
    eprintln!("Environment: HEARTH_THREADS (worker count), HEARTH_CTX_CAP (context ceiling),");
    eprintln!(
        "             HEARTH_MODELS_DIR (model folder; hearth-ui also walks up from the exe),"
    );
    eprintln!("             HEARTH_PIN_WEIGHTS=1 (same as --pin-weights)");
    eprintln!();
    eprintln!("Interactive commands: /quit, /clear, /system <msg>, /config");
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "-h" || args[1] == "--help" {
        print_usage();
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    // `--service` may appear as the first argument (no model is loaded in
    // service mode); everything else keeps the <model.gguf> first convention.
    let service_first = args[1] == "--service";
    let model_path = if service_first {
        String::new()
    } else {
        args[1].clone()
    };
    let mut strategy = LoadStrategy::CpuOnly;
    let mut overrides = SamplerOverrides::default();
    let mut max_tokens = 512usize;
    let mut max_context: Option<usize> = None;
    let mut prompt = None;
    let mut prompt_raw = false;
    let mut kv_f32 = false;
    let mut dump_info = false;
    let mut show_think = false;
    let mut pin_weights = false;
    let mut service = service_first;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" => {
                strategy = LoadStrategy::GpuFull;
            }
            "--gpu-layers" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        strategy = LoadStrategy::LayerSplit(n);
                    }
                    i += 1;
                }
            }
            "--temp" => {
                if i + 1 < args.len() {
                    if let Ok(t) = args[i + 1].parse::<f32>() {
                        overrides.temperature = Some(t);
                    }
                    i += 1;
                }
            }
            "--top-k" => {
                if i + 1 < args.len() {
                    if let Ok(k) = args[i + 1].parse::<usize>() {
                        overrides.top_k = Some(k);
                    }
                    i += 1;
                }
            }
            "--top-p" => {
                if i + 1 < args.len() {
                    if let Ok(p) = args[i + 1].parse::<f32>() {
                        overrides.top_p = Some(p);
                    }
                    i += 1;
                }
            }
            "--min-p" => {
                if i + 1 < args.len() {
                    if let Ok(p) = args[i + 1].parse::<f32>() {
                        overrides.min_p = Some(p);
                    }
                    i += 1;
                }
            }
            "--typical-p" => {
                if i + 1 < args.len() {
                    if let Ok(p) = args[i + 1].parse::<f32>() {
                        overrides.typical_p = Some(p);
                    }
                    i += 1;
                }
            }
            "--repeat-pen" => {
                if i + 1 < args.len() {
                    if let Ok(r) = args[i + 1].parse::<f32>() {
                        overrides.repeat_pen = Some(r);
                    }
                    i += 1;
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    if let Ok(s) = args[i + 1].parse::<u64>() {
                        overrides.seed = Some(s);
                    }
                    i += 1;
                }
            }
            "--max-tokens" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        max_tokens = n;
                    }
                    i += 1;
                }
            }
            "--max-context" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        max_context = Some(n);
                    }
                    i += 1;
                }
            }
            "--prompt" => {
                if i + 1 < args.len() {
                    prompt = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--kv-f32" => {
                kv_f32 = true;
            }
            "--dump-info" => {
                dump_info = true;
            }
            "--show-think" => {
                show_think = true;
            }
            "--prompt-raw" => {
                prompt_raw = true;
            }
            "--service" => {
                service = true;
            }
            "--pin-weights" => {
                pin_weights = true;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
            }
        }
        i += 1;
    }

    CliArgs {
        model_path,
        strategy,
        overrides,
        max_tokens,
        max_context,
        prompt,
        prompt_raw,
        kv_f32,
        dump_info,
        show_think,
        pin_weights,
        service,
    }
}

fn generate(
    model: &LlamaModel,
    prompt: String,
    sampler_config: &SamplerConfig,
    max_tokens: usize,
    max_context: Option<usize>,
    show_think: bool,
) {
    let request = PipelineRequest::TextGen {
        prompt,
        sampler: sampler_config.clone(),
        max_new_tokens: max_tokens,
        max_context_len: max_context,
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<EngineOutput>();

    // F32 is the only KV backend today; say so instead of implying Q8_0.
    eprintln!("[Generating with F32 KV cache...]");

    let cancel_clone = cancel.clone();
    std::thread::scope(|s| {
        s.spawn(|| {
            model.run(request, cancel_clone, tx);
        });
    });

    let mut filter = if show_think {
        None
    } else {
        Some(ThinkFilter::new())
    };
    let mut reply = String::new();
    for received in rx {
        match received {
            EngineOutput::TextToken(token) => {
                let clean = if let Some(ref mut f) = filter {
                    f.process(&token)
                } else {
                    token.clone()
                };
                if !clean.is_empty() {
                    print!("{}", clean);
                    io::stdout().flush().ok();
                    reply.push_str(&token);
                }
            }
            EngineOutput::Done(stats) => {
                println!();
                eprintln!(
                    "\n[{} new tokens ({} prompt), {} ms, {:.2} tok/s]",
                    stats.tokens, stats.prompt_tokens, stats.elapsed_ms, stats.tokens_per_s
                );
                break;
            }
            EngineOutput::Error(e) => {
                eprintln!("\nError: {}", e);
                break;
            }
            _ => {}
        }
    }
    let _ = reply;
}

fn main() {
    let cli = parse_args();

    // Service mode: line-JSON IPC over stdio. This is the single pipeline
    // hearth-ui uses to reach the engine; the UI never links hearth-llm.
    if cli.service {
        service::run();
        return;
    }

    eprintln!("Loading model: {} ...", cli.model_path);
    let strategy_label = match &cli.strategy {
        LoadStrategy::GpuFull => "GPU (all layers)",
        LoadStrategy::LayerSplit(n) => &*format!("GPU {} layers + CPU", n).leak(),
        LoadStrategy::CpuOffload => "CPU offload",
        LoadStrategy::CpuOnly => "CPU only",
    };
    eprintln!("Strategy: {}", strategy_label);
    eprintln!("Max tokens: {}", cli.max_tokens);
    if let Some(c) = cli.max_context {
        eprintln!("Max context: {}", c);
    }
    if cli.show_think {
        eprintln!("Show thinking: yes");
    }
    if cli.kv_f32 {
        eprintln!("KV cache: F32 (the only backend today; flag accepted for compatibility)");
    }

    if cli.dump_info {
        let gguf = GgufFile::open(&cli.model_path).expect("Failed to open GGUF for dump");
        eprintln!("\n=== GGUF METADATA (ALL) ===");
        let arch = gguf.meta_str("general.architecture").unwrap_or("unknown");
        eprintln!("architecture: {}", arch);
        for (k, v) in &gguf.metadata {
            eprintln!("  {} = {:?}", k, v);
        }
        eprintln!("\n=== LAYER 0 TENSOR DTYPES ===");
        for t in &gguf.tensors {
            if t.name.contains("blk.0.")
                || t.name == "token_embd.weight"
                || t.name == "output.weight"
                || t.name == "output_norm.weight"
            {
                let byte_size = t.byte_size();
                let elem_count = t.element_count();
                eprintln!(
                    "  {} shape={:?} dtype={:?} (raw_id={}) byte_size={} elem_count={}",
                    t.name, t.shape, t.dtype, t.dtype_id, byte_size, elem_count
                );
            }
        }
        eprintln!("\n=== ALL UNIQUE DTYPES ===");
        use std::collections::HashSet;
        let mut dtypes: HashSet<String> = HashSet::new();
        for t in &gguf.tensors {
            dtypes.insert(format!("{:?}", t.dtype));
        }
        for d in &dtypes {
            eprintln!("  {}", d);
        }
        return;
    }

    if cli.pin_weights {
        // Read by the loaders; must be set before the model is opened.
        std::env::set_var("HEARTH_PIN_WEIGHTS", "1");
    }

    let model = match LlamaModel::load(&cli.model_path, cli.strategy) {
        Ok(m) => {
            eprintln!("Model loaded successfully.");
            m
        }
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let tk = model.template_kind();
    eprintln!("Template: {:?}", tk);
    model.debug_template_info();
    let model_cfg = model.debug_config();
    eprintln!("[CONFIG] {:?}", model_cfg);

    // Resolve the sampler per field: a flag wins, otherwise the model's own
    // `general.sampling.*` recommendation, otherwise the built-in fallback.
    let config = cli.overrides.apply(model_cfg.sampler_defaults());
    eprintln!(
        "Sampler: temp={} top_k={} top_p={} min_p={} typical_p={} repeat_pen={} seed={}{}",
        config.temperature,
        config.top_k,
        config.top_p,
        config.min_p,
        config.typical_p,
        config.repeat_pen,
        config.seed,
        if cli.overrides.any() {
            "  (unset fields from the model's general.sampling.*; set them with --temp/--top-k/--top-p/--min-p/--repeat-pen/--seed)"
        } else {
            "  (model general.sampling.* defaults; override with --temp/--top-k/--top-p/--min-p/--typical-p/--repeat-pen/--seed)"
        }
    );

    // Single-shot mode
    if let Some(ref prompt_text) = cli.prompt {
        let system = "You are a helpful assistant.";
        let history: Vec<(&str, &str)> = Vec::new();
        let prompt = if cli.prompt_raw {
            prompt_text.clone()
        } else {
            model.build_chat_prompt(system, &history, prompt_text, cli.show_think)
        };
        generate(
            &model,
            prompt,
            &config,
            cli.max_tokens,
            cli.max_context,
            cli.show_think,
        );
        return;
    }

    // Interactive mode
    eprintln!();
    eprintln!("Interactive chat. Type your message and press Enter.");
    eprintln!("Commands: /quit, /clear, /system <msg>, /config");
    eprintln!();

    let mut system_prompt = "You are a helpful assistant.".to_string();
    let mut history: Vec<(String, String)> = Vec::new();

    loop {
        eprint!("> ");
        io::stderr().flush().ok();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() || input.is_empty() {
            break;
        }
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }
        if input == "/quit" || input == "/exit" {
            break;
        }
        if input == "/clear" {
            history.clear();
            eprintln!("[History cleared]");
            continue;
        }
        if let Some(stripped) = input.strip_prefix("/system ") {
            system_prompt = stripped.to_string();
            eprintln!("[System prompt set: {}]", system_prompt);
            continue;
        }
        if input == "/config" {
            eprintln!(
                "temp={} top_k={} top_p={} min_p={} typical_p={} repeat_pen={} seed={} max_tokens={} max_context={:?}",
                config.temperature,
                config.top_k,
                config.top_p,
                config.min_p,
                config.typical_p,
                config.repeat_pen,
                config.seed,
                cli.max_tokens,
                cli.max_context
            );
            continue;
        }

        let history_refs: Vec<(&str, &str)> = history
            .iter()
            .map(|(u, a)| (u.as_str(), a.as_str()))
            .collect();
        let prompt = model.build_chat_prompt(&system_prompt, &history_refs, &input, cli.show_think);

        let (tx, rx) = mpsc::channel::<EngineOutput>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = cancel.clone();

        std::thread::scope(|s| {
            s.spawn(|| {
                model.run(
                    PipelineRequest::TextGen {
                        prompt,
                        sampler: config.clone(),
                        max_new_tokens: cli.max_tokens,
                        max_context_len: cli.max_context,
                    },
                    cancel_clone,
                    tx,
                );
            });
        });

        let mut filter = if cli.show_think {
            None
        } else {
            Some(ThinkFilter::new())
        };
        let mut reply = String::new();
        for received in rx {
            match received {
                EngineOutput::TextToken(token) => {
                    let clean = if let Some(ref mut f) = filter {
                        f.process(&token)
                    } else {
                        token.clone()
                    };
                    if !clean.is_empty() {
                        print!("{}", clean);
                        io::stdout().flush().ok();
                        reply.push_str(&token);
                    }
                }
                EngineOutput::Done(stats) => {
                    println!();
                    eprintln!(
                        "\n[{} new tokens ({} prompt), {} ms, {:.2} tok/s]",
                        stats.tokens, stats.prompt_tokens, stats.elapsed_ms, stats.tokens_per_s
                    );
                    break;
                }
                EngineOutput::Error(e) => {
                    eprintln!("\nError: {}", e);
                    break;
                }
                _ => {}
            }
        }

        let reply_trimmed = reply.trim().to_string();
        if !reply_trimmed.is_empty() {
            history.push((input, reply_trimmed));
        }
    }
}
