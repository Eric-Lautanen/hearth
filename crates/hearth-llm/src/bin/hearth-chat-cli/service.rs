//! `hearth-chat-cli --service` — the single engine pipeline exposed to hearth-ui.
//!
//! Speaks line-delimited JSON over stdio (one JSON object per line). The UI
//! spawns this binary as a child process and never links the engine directly.
//!
//! Requests (UI -> service):
//!   {"id":N,"cmd":"list_models"}                       -> event "models"
//!   {"id":N,"cmd":"load","path":"...","device":"cpu|gpu|gpu_split|offload",
//!    "gpu_layers":N,"pin_weights":B}                   -> event "loaded" | error
//!   {"id":N,"cmd":"unload"}                            -> event "unloaded"
//!   {"id":N,"cmd":"status"}                            -> event "status"
//!   {"id":N,"cmd":"generate","system":S,"history":[[u,a],...],"message":M,
//!    "temp":T,"top_k":K,"top_p":P,"min_p":P,"typical_p":P,"repeat_pen":R,
//!    "seed":N,"max_tokens":N,"max_context":C,"show_think":B,"enable_think":B}
//!    (any sampling field may be omitted; the loaded model's
//!     general.sampling.* recommendation is used instead)
//!                                                       -> "reason"* "token"*
//!                                                          then "done"
//!   {"id":N,"cmd":"stop"}                              -> cancels active generate
//!   {"id":N,"cmd":"shutdown"}                          -> exits the service
//!
//! Replies always carry the request `id` and `ok`. Streamed events reuse the
//! `id` of the generate request that produced them.

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use hearth_core::json::{self, Json};
use hearth_core::{EngineOutput, LoadStrategy, Model, PipelineRequest, SamplerConfig};
use hearth_llm::LlamaModel;

// ---------------------------------------------------------------------------
// <think> handling
// ---------------------------------------------------------------------------

/// Splits a raw token stream into (reasoning, answer) text. Feed ONE token
/// per `feed` call; each returns the fragment for each channel (may be empty).
///
/// The chat template places the think markers in the prompt itself, so the
/// initial state is derived from the prompt: a trailing open `<think>` means
/// the model is already inside the think block (everything it emits streams
/// as reasoning until a `</think>` arrives); a trailing `</think>` means it
/// answers directly. Prompts without any marker fall back to sniffing.
struct StreamSplit {
    state: SplitState,
    tail: String,
}

enum SplitState {
    /// No markers known yet; tags may be split across tokens so everything is
    /// buffered until one is positively found (or the stream ends).
    Prologue,
    /// Inside `<think>...</think>`: text goes to the reasoning channel.
    Reasoning,
    /// After the closing tag: text goes to the answer channel.
    Answer,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

impl StreamSplit {
    fn for_prompt(prompt: &str) -> Self {
        let state = match (prompt.rfind(THINK_OPEN), prompt.rfind(THINK_CLOSE)) {
            (Some(o), Some(c)) if o > c => SplitState::Reasoning,
            (Some(_), Some(_)) => SplitState::Answer,
            (Some(_), None) => SplitState::Reasoning,
            _ => SplitState::Prologue,
        };
        StreamSplit {
            state,
            tail: String::new(),
        }
    }

    /// Feed ONE raw token; returns (reasoning_fragment, answer_fragment),
    /// either of which may be empty. Call exactly once per token.
    fn feed(&mut self, token: &str) -> (String, String) {
        match self.state {
            SplitState::Answer => {
                self.tail.push_str(token);
                let out = std::mem::take(&mut self.tail);
                (String::new(), out)
            }
            SplitState::Reasoning => {
                self.tail.push_str(token);
                if let Some(pos) = self.tail.find(THINK_CLOSE) {
                    let reason_part = self.tail[..pos].to_string();
                    let rest = self.tail[pos + THINK_CLOSE.len()..].to_string();
                    self.tail.clear();
                    self.state = SplitState::Answer;
                    return (reason_part, rest);
                }
                // Hold back a suffix that could be the start of a `</think>`
                // split across tokens, so the closer never leaks into the
                // visible reasoning text.
                let hold = self.partial_close_len();
                let emit_len = self.tail.len() - hold;
                let out = self.tail[..emit_len].to_string();
                self.tail.drain(..emit_len);
                (out, String::new())
            }
            SplitState::Prologue => {
                // Buffer until a marker is positively found (or the stream
                // ends) since the opener/closer may straddle tokens.
                self.tail.push_str(token);
                if let Some(pos) = self.tail.find(THINK_OPEN) {
                    let rest = self.tail[pos + THINK_OPEN.len()..].to_string();
                    self.tail.clear();
                    self.state = SplitState::Reasoning;
                    // Closer may ride in the same remainder (empty think block).
                    if let Some(cp) = rest.find(THINK_CLOSE) {
                        self.state = SplitState::Answer;
                        return (
                            rest[..cp].to_string(),
                            rest[cp + THINK_CLOSE.len()..].to_string(),
                        );
                    }
                    return (rest, String::new());
                }
                if let Some(pos) = self.tail.find(THINK_CLOSE) {
                    // Pre-closed think block: empty reasoning, rest is answer.
                    let rest = self.tail[pos + THINK_CLOSE.len()..].to_string();
                    self.tail.clear();
                    self.state = SplitState::Answer;
                    return (String::new(), rest);
                }
                if self.tail.len() > 64 * 1024 {
                    // Degenerate stream with no markers: all of it is the answer.
                    let out = std::mem::take(&mut self.tail);
                    self.state = SplitState::Answer;
                    return (String::new(), out);
                }
                (String::new(), String::new())
            }
        }
    }

    /// Length of the trailing suffix of `tail` that is a proper prefix of
    /// `</think>` (max 7 bytes), i.e. bytes that must not be emitted yet.
    fn partial_close_len(&self) -> usize {
        let max = (THINK_CLOSE.len() - 1).min(self.tail.len());
        for l in (1..=max).rev() {
            if self.tail.ends_with(&THINK_CLOSE[..l]) {
                return l;
            }
        }
        0
    }

    /// At end of stream: anything still buffered had no markers. An unfinished
    /// think block is all reasoning; an unmarked stream is all answer.
    fn flush(&mut self) -> (String, String) {
        let buf = std::mem::take(&mut self.tail);
        match self.state {
            SplitState::Prologue => (String::new(), buf),
            SplitState::Reasoning => (buf, String::new()),
            SplitState::Answer => (String::new(), buf),
        }
    }
}

// ---------------------------------------------------------------------------
// Worker commands
// ---------------------------------------------------------------------------

enum Cmd {
    ListModels {
        id: i64,
    },
    Load {
        id: i64,
        path: String,
        strategy: LoadStrategy,
        pin_weights: bool,
    },
    Unload {
        id: i64,
    },
    Status {
        id: i64,
    },
    Generate {
        id: i64,
        req: GenRequest,
    },
    Quit,
}

/// Every sampler field is optional: an absent field falls back to the loaded
/// model's own `general.sampling.*` recommendation, then to Hearth's built-in
/// defaults. Nothing about sampling is hardcoded per-request anymore.
struct GenRequest {
    system: String,
    history: Vec<(String, String)>,
    message: String,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    typical_p: Option<f32>,
    repeat_pen: Option<f32>,
    seed: Option<u64>,
    max_tokens: usize,
    max_context: Option<usize>,
    /// Show the model's <think> reasoning as it streams.
    show_think: bool,
    /// Ask the model to reason at all (template: don't pre-close the think
    /// block). Defaults to `show_think` when absent.
    enable_think: bool,
}

// ---------------------------------------------------------------------------
// Thread-safe line writer
// ---------------------------------------------------------------------------

struct Out {
    stdout: Mutex<std::io::Stdout>,
}

impl Out {
    fn new() -> Self {
        Out {
            stdout: Mutex::new(std::io::stdout()),
        }
    }

    fn send(&self, value: &Json) {
        let mut lock = self.stdout.lock().unwrap();
        let _ = writeln!(lock, "{}", value);
        let _ = lock.flush();
    }

    fn send_error(&self, id: i64, message: &str) {
        let mut v = Json::obj();
        v.set("id", Json::Num(id as f64));
        v.set("ok", Json::Bool(false));
        v.set("event", Json::str("error"));
        v.set("error", Json::str(message));
        self.send(&v);
    }
}

// ---------------------------------------------------------------------------
// Service state (worker thread only)
// ---------------------------------------------------------------------------

struct Service {
    out: Arc<Out>,
    model: Option<Arc<LlamaModel>>,
    model_path: Option<String>,
    /// Cancel flag of the generation currently in flight, shared with the main
    /// thread so `stop` can reach it while the worker is busy streaming.
    cancel_slot: Arc<Mutex<Option<Arc<AtomicBool>>>>,
}

impl Service {
    fn model_info_json(&self, model: &LlamaModel, path: &str) -> Json {
        let cfg = model.debug_config();
        let name = Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path)
            .to_string();
        let mut info = Json::obj();
        info.set("path", Json::str(path));
        info.set("name", Json::str(name));
        info.set("arch", Json::str(cfg.arch.clone()));
        info.set("n_layers", Json::num(cfg.n_layers as f64));
        info.set("d_model", Json::num(cfg.d_model as f64));
        info.set("n_heads", Json::num(cfg.n_heads as f64));
        info.set("n_kv_heads", Json::num(cfg.n_kv_heads as f64));
        info.set("d_ffn", Json::num(cfg.d_ffn as f64));
        info.set("vocab_size", Json::num(cfg.vocab_size as f64));
        info.set("max_seq_len", Json::num(cfg.max_seq_len as f64));
        // KV cache allocated at load (bytes). The UI shows this so a model whose
        // context costs gigabytes of RAM is visible before it is chatted with.
        info.set("kv_bytes", Json::num(model.kv_cache_bytes() as f64));
        // Which kernel each weight format actually runs, plus the CPU class that
        // chose it. Exposed as data so the UI never has to parse the load banner:
        // a stale dispatch (a feature gate that silently answers "no") shows up
        // here instead of only in a 34%-slower tok/s number.
        let dispatch = model.dispatch();
        let mut kernels = Json::obj();
        kernels.set("avx512_full_width", Json::Bool(dispatch.avx512_full_width));
        kernels.set("cpu_family", Json::str(dispatch.cpu_family.clone()));
        kernels.set("cpu", Json::str(dispatch.cpu));
        for (format, kernel) in &dispatch.kernels {
            kernels.set(format, Json::str(*kernel));
        }
        info.set("kernels", kernels);
        info.set(
            "template",
            Json::str(format!("{:?}", model.template_kind())),
        );
        // The model author's sampling recommendations, so the UI can seed its
        // controls with them instead of a global hardcoded set.
        let sd = cfg.sampler_defaults();
        info.set("sampling_temperature", Json::num(sd.temperature as f64));
        info.set("sampling_top_p", Json::num(sd.top_p as f64));
        info.set("sampling_top_k", Json::num(sd.top_k as f64));
        info.set("sampling_min_p", Json::num(sd.min_p as f64));
        info.set("sampling_repeat_pen", Json::num(sd.repeat_pen as f64));
        info.set(
            "sampling_from_model",
            Json::Bool(
                cfg.sampling_temperature.is_some()
                    || cfg.sampling_top_p.is_some()
                    || cfg.sampling_top_k.is_some()
                    || cfg.sampling_min_p.is_some(),
            ),
        );
        info
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Quit => {}

            Cmd::ListModels { id } => {
                // Every root the loader accepts, not just one: an installed
                // build keeps models in `<install>/models`, a dev tree in the
                // repo, older ones in `%APPDATA%\hearth\models`. One filename
                // in two places is listed once, first root wins.
                let roots = hearth_core::paths::models_roots();
                let mut seen = HashSet::new();
                let mut models = Vec::new();
                for dir in &roots {
                    let Ok(entries) = std::fs::read_dir(dir) else {
                        continue;
                    };
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let is_gguf = path
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|e| e.eq_ignore_ascii_case("gguf"))
                            .unwrap_or(false);
                        if !is_gguf {
                            continue;
                        }
                        let name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or_default()
                            .to_string();
                        if name.is_empty() {
                            continue;
                        }
                        let key = name.to_lowercase();
                        if !seen.insert(key) {
                            continue;
                        }
                        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                        let mut item = Json::obj();
                        item.set("name", Json::str(name));
                        item.set("path", Json::str(path.to_string_lossy().to_string()));
                        item.set("size_bytes", Json::num(size as f64));
                        models.push(item);
                    }
                }
                models.sort_by(|a, b| {
                    let ka = a.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let kb = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    ka.cmp(kb)
                });
                let primary = hearth_core::paths::primary_models_dir();
                let mut v = Json::obj();
                v.set("id", Json::Num(id as f64));
                v.set("ok", Json::Bool(true));
                v.set("event", Json::str("models"));
                v.set("dir", Json::str(primary.to_string_lossy().to_string()));
                v.set(
                    "dirs",
                    Json::Arr(
                        roots
                            .iter()
                            .map(|d| Json::str(d.to_string_lossy().to_string()))
                            .collect(),
                    ),
                );
                v.set("models", Json::Arr(models));
                self.out.send(&v);
            }

            Cmd::Load {
                id,
                path,
                strategy,
                pin_weights,
            } => {
                // Weight residency is read by the loader at open time.
                if pin_weights {
                    std::env::set_var("HEARTH_PIN_WEIGHTS", "1");
                } else {
                    std::env::remove_var("HEARTH_PIN_WEIGHTS");
                }
                // Load into a temporary first so a failed load keeps the
                // currently loaded model usable.
                match LlamaModel::load(&path, strategy) {
                    Ok(m) => {
                        let info = self.model_info_json(&m, &path);
                        self.model = Some(Arc::new(m));
                        self.model_path = Some(path);
                        let mut v = Json::obj();
                        v.set("id", Json::Num(id as f64));
                        v.set("ok", Json::Bool(true));
                        v.set("event", Json::str("loaded"));
                        v.set("model", info);
                        self.out.send(&v);
                    }
                    Err(e) => {
                        self.out.send_error(id, &e);
                    }
                }
            }

            Cmd::Unload { id } => {
                self.model = None;
                self.model_path = None;
                let mut v = Json::obj();
                v.set("id", Json::Num(id as f64));
                v.set("ok", Json::Bool(true));
                v.set("event", Json::str("unloaded"));
                self.out.send(&v);
            }

            Cmd::Status { id } => {
                let mut v = Json::obj();
                v.set("id", Json::Num(id as f64));
                v.set("ok", Json::Bool(true));
                v.set("event", Json::str("status"));
                match (&self.model, &self.model_path) {
                    (Some(m), Some(p)) => {
                        v.set("loaded", Json::Bool(true));
                        v.set("model", self.model_info_json(m, p));
                    }
                    _ => {
                        v.set("loaded", Json::Bool(false));
                    }
                }
                v.set(
                    "generating",
                    Json::Bool(self.cancel_slot.lock().unwrap().is_some()),
                );
                self.out.send(&v);
            }

            Cmd::Generate { id, req } => {
                let model = match self.model.clone() {
                    Some(m) => m,
                    None => {
                        self.out.send_error(id, "No model loaded");
                        return;
                    }
                };

                let history_refs: Vec<(&str, &str)> = req
                    .history
                    .iter()
                    .map(|(u, a)| (u.as_str(), a.as_str()))
                    .collect();
                let prompt = model.build_chat_prompt(
                    &req.system,
                    &history_refs,
                    &req.message,
                    req.enable_think,
                );
                // The template's trailing think markers decide how the token
                // stream is split into reasoning vs answer; read them before
                // the prompt is moved into the generation thread.
                let split = StreamSplit::for_prompt(&prompt);

                let cancel = Arc::new(AtomicBool::new(false));
                *self.cancel_slot.lock().unwrap() = Some(cancel.clone());

                // Resolve sampling against the model's own recommendation for
                // any field the client left unset.
                let model_defaults = model.debug_config().sampler_defaults();
                let sampler = SamplerConfig {
                    temperature: req.temperature.unwrap_or(model_defaults.temperature),
                    top_k: req.top_k.unwrap_or(model_defaults.top_k),
                    top_p: req.top_p.unwrap_or(model_defaults.top_p),
                    min_p: req.min_p.unwrap_or(model_defaults.min_p),
                    typical_p: req.typical_p.unwrap_or(model_defaults.typical_p),
                    repeat_pen: req.repeat_pen.unwrap_or(model_defaults.repeat_pen),
                    seed: req.seed.unwrap_or(model_defaults.seed),
                };
                let request = PipelineRequest::TextGen {
                    prompt,
                    sampler,
                    max_new_tokens: req.max_tokens,
                    max_context_len: req.max_context,
                };

                let (tx, rx) = mpsc::channel::<EngineOutput>();
                let gen_model = model.clone();
                let gen_cancel = cancel.clone();

                let mut reply = String::new();
                let mut reason = String::new();
                let mut stopped = false;

                std::thread::scope(|s| {
                    s.spawn(move || {
                        gen_model.run(request, gen_cancel, tx);
                    });

                    // The worker has nothing else to do while generating (all
                    // other commands queue up), so a blocking drain is fine.
                    // Stop requests arrive through the shared cancel flag.
                    //
                    // Both modes share one StreamSplit: with `show_think` the
                    // reasoning is also streamed to the UI; hidden it is simply
                    // dropped (but the answer still streams live).
                    let mut split = split;
                    // Counted whether or not the reasoning is streamed, so a
                    // reply that was all hidden reasoning can be explained.
                    let mut reason_token_count = 0usize;

                    for event in rx {
                        match event {
                            EngineOutput::TextToken(token) => {
                                let (think, answer) = split.feed(&token);
                                if !think.is_empty() {
                                    reason_token_count += 1;
                                    if req.show_think {
                                        let mut v = Json::obj();
                                        v.set("id", Json::Num(id as f64));
                                        v.set("ok", Json::Bool(true));
                                        v.set("event", Json::str("reason"));
                                        v.set("text", Json::str(&think));
                                        self.out.send(&v);
                                    }
                                    reason.push_str(&think);
                                }
                                if !answer.is_empty() {
                                    reply.push_str(&answer);
                                    let mut v = Json::obj();
                                    v.set("id", Json::Num(id as f64));
                                    v.set("ok", Json::Bool(true));
                                    v.set("event", Json::str("token"));
                                    v.set("text", Json::str(answer));
                                    self.out.send(&v);
                                }
                            }
                            EngineOutput::Done(stats) => {
                                stopped = cancel.load(Ordering::Relaxed);
                                // Splitter leftovers: a stream that ended inside
                                // <think> is all reasoning; an unmarked stream is
                                // all answer.
                                let (r_tail, a_tail) = split.flush();
                                if !r_tail.is_empty() {
                                    reason_token_count += 1;
                                    reason.push_str(&r_tail);
                                    if req.show_think {
                                        let mut v = Json::obj();
                                        v.set("id", Json::Num(id as f64));
                                        v.set("ok", Json::Bool(true));
                                        v.set("event", Json::str("reason"));
                                        v.set("text", Json::str(r_tail));
                                        self.out.send(&v);
                                    }
                                }
                                if !a_tail.is_empty() {
                                    reply.push_str(&a_tail);
                                    let mut v = Json::obj();
                                    v.set("id", Json::Num(id as f64));
                                    v.set("ok", Json::Bool(true));
                                    v.set("event", Json::str("token"));
                                    v.set("text", Json::str(a_tail));
                                    self.out.send(&v);
                                }
                                let mut v = Json::obj();
                                v.set("id", Json::Num(id as f64));
                                v.set("ok", Json::Bool(true));
                                v.set("event", Json::str("done"));
                                v.set("tokens", Json::num(stats.tokens as f64));
                                // Prompt tokens separately: a client that only
                                // sees one number cannot tell "the model said
                                // nothing" from "the model said one word".
                                v.set("prompt_tokens", Json::num(stats.prompt_tokens as f64));
                                v.set("elapsed_ms", Json::num(stats.elapsed_ms as f64));
                                v.set("tokens_per_s", Json::num(stats.tokens_per_s as f64));
                                v.set("stopped", Json::Bool(stopped));
                                v.set("text", Json::str(reply.clone()));
                                if reason_token_count > 0 {
                                    v.set("reason_tokens", Json::num(reason_token_count as f64));
                                }
                                if req.show_think && !reason.is_empty() {
                                    v.set("reason_text", Json::str(reason.clone()));
                                }
                                self.out.send(&v);
                                break;
                            }
                            EngineOutput::Error(e) => {
                                self.out.send_error(id, &e.to_string());
                                break;
                            }
                            _ => {}
                        }
                    }
                });

                *self.cancel_slot.lock().unwrap() = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Field helpers
// ---------------------------------------------------------------------------

fn get_str(v: &Json, key: &str, default: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or(default)
        .to_string()
}

fn get_usize(v: &Json, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(|x| x.as_usize())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

fn get_opt_f32(v: &Json, key: &str) -> Option<f32> {
    v.get(key).and_then(|x| x.as_f64()).map(|n| n as f32)
}

fn get_opt_usize(v: &Json, key: &str) -> Option<usize> {
    v.get(key).and_then(|x| x.as_usize()).filter(|n| *n > 0)
}

fn get_opt_u64(v: &Json, key: &str) -> Option<u64> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .filter(|n| *n >= 0.0)
        .map(|n| n as u64)
}

fn get_bool(v: &Json, key: &str, default: bool) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Entry point (called from hearth-chat-cli main with --service)
// ---------------------------------------------------------------------------

pub fn run() {
    let out = Arc::new(Out::new());
    let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
    let cancel_slot: Arc<Mutex<Option<Arc<AtomicBool>>>> = Arc::new(Mutex::new(None));

    // Greeting so the UI can verify the protocol before sending commands.
    let mut ready = Json::obj();
    ready.set("id", Json::Num(0.0));
    ready.set("ok", Json::Bool(true));
    ready.set("event", Json::str("ready"));
    ready.set("protocol", Json::num(1.0));
    ready.set("service", Json::str("hearth-chat-cli"));
    out.send(&ready);

    let worker_out = out.clone();
    let worker_cancel_slot = cancel_slot.clone();
    let worker = std::thread::Builder::new()
        .name("hearth-service".into())
        .spawn(move || {
            let mut svc = Service {
                out: worker_out,
                model: None,
                model_path: None,
                cancel_slot: worker_cancel_slot,
            };
            while let Ok(cmd) = cmd_rx.recv() {
                if matches!(cmd, Cmd::Quit) {
                    break;
                }
                svc.handle(cmd);
            }
        })
        .expect("spawn hearth-service worker");

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value = match json::parse(trimmed) {
            Ok(v) => v,
            Err(e) => {
                out.send_error(0, &format!("bad JSON: {}", e));
                continue;
            }
        };

        let id = value.get("id").and_then(|x| x.as_f64()).unwrap_or(0.0) as i64;
        let cmd = value.get("cmd").and_then(|x| x.as_str()).unwrap_or("");

        match cmd {
            "stop" => {
                if let Some(flag) = cancel_slot.lock().unwrap().as_ref() {
                    flag.store(true, Ordering::Relaxed);
                }
                let mut v = Json::obj();
                v.set("id", Json::Num(id as f64));
                v.set("ok", Json::Bool(true));
                v.set("event", Json::str("stopped"));
                out.send(&v);
            }
            "list_models" => {
                let _ = cmd_tx.send(Cmd::ListModels { id });
            }
            "load" => {
                let path = get_str(&value, "path", "");
                if path.is_empty() {
                    out.send_error(id, "load: missing path");
                } else {
                    let strategy = match get_str(&value, "device", "").as_str() {
                        "gpu" => LoadStrategy::GpuFull,
                        "gpu_split" => {
                            let n = get_usize(&value, "gpu_layers", 0);
                            if n > 0 {
                                LoadStrategy::LayerSplit(n)
                            } else {
                                LoadStrategy::CpuOnly
                            }
                        }
                        "offload" => LoadStrategy::CpuOffload,
                        _ => LoadStrategy::CpuOnly,
                    };
                    let pin_weights = get_bool(&value, "pin_weights", false);
                    let _ = cmd_tx.send(Cmd::Load {
                        id,
                        path,
                        strategy,
                        pin_weights,
                    });
                }
            }
            "unload" => {
                let _ = cmd_tx.send(Cmd::Unload { id });
            }
            "status" => {
                let _ = cmd_tx.send(Cmd::Status { id });
            }
            "generate" => {
                let mut history = Vec::new();
                if let Some(items) = value.get("history").and_then(|x| x.as_arr()) {
                    for item in items {
                        if let Some(pair) = item.as_arr() {
                            if pair.len() == 2 {
                                let u = pair[0].as_str().unwrap_or_default().to_string();
                                let a = pair[1].as_str().unwrap_or_default().to_string();
                                history.push((u, a));
                            }
                        }
                    }
                }
                let max_context = get_opt_usize(&value, "max_context");
                let show_think = get_bool(&value, "show_think", false);
                let req = GenRequest {
                    system: get_str(&value, "system", "You are a helpful assistant."),
                    history,
                    message: get_str(&value, "message", ""),
                    temperature: get_opt_f32(&value, "temp"),
                    top_k: get_opt_usize(&value, "top_k"),
                    top_p: get_opt_f32(&value, "top_p"),
                    min_p: get_opt_f32(&value, "min_p"),
                    typical_p: get_opt_f32(&value, "typical_p"),
                    repeat_pen: get_opt_f32(&value, "repeat_pen"),
                    seed: get_opt_u64(&value, "seed"),
                    max_tokens: get_usize(&value, "max_tokens", 512),
                    max_context,
                    show_think,
                    // The UI may explicitly control "reason at all" separately
                    // from "show the reasoning"; default: reasoning on when it
                    // will be shown, off otherwise.
                    enable_think: match value.get("enable_think") {
                        Some(v) => v.as_bool().unwrap_or(show_think),
                        None => show_think,
                    },
                };
                if req.message.trim().is_empty() {
                    out.send_error(id, "generate: empty message");
                } else {
                    let _ = cmd_tx.send(Cmd::Generate { id, req });
                }
            }
            "shutdown" => {
                if let Some(flag) = cancel_slot.lock().unwrap().as_ref() {
                    flag.store(true, Ordering::Relaxed);
                }
                let _ = cmd_tx.send(Cmd::Quit);
                break;
            }
            other => {
                out.send_error(id, &format!("unknown cmd: {}", other));
            }
        }
    }

    // stdin closed (UI exited) or shutdown requested: cancel any in-flight
    // generation and let the process tear down.
    if let Some(flag) = cancel_slot.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
    let _ = cmd_tx.send(Cmd::Quit);
    let _ = worker.join();
}
