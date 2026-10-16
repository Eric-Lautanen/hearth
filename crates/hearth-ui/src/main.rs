//! hearth-ui — a local web chat interface for the Hearth inference engine.
//!
//! Std-only (no external crates). Serves a modern dark chat UI on
//! `http://127.0.0.1:<port>` and proxies every engine operation through
//! `hearth-chat-cli --service` over line-JSON stdio — the single pipeline
//! into the engine. The UI never links hearth-llm directly.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use hearth_core::json::{self, Json};

mod downloads;
mod web;

// ---------------------------------------------------------------------------
// Protocol types
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ModelEntry {
    name: String,
    path: String,
    size_bytes: u64,
}

#[derive(Clone)]
struct LoadedModel {
    name: String,
    path: String,
    arch: String,
    n_layers: u32,
    d_model: u32,
    n_heads: u32,
    n_kv_heads: u32,
    d_ffn: u32,
    vocab_size: u32,
    max_seq_len: u32,
    /// KV cache bytes allocated at load, so the UI can show the model's real
    /// RAM cost next to the context window it buys.
    kv_bytes: u64,
    template: String,
    /// The model author's `general.sampling.*` recommendations (or Hearth's
    /// fallbacks when the GGUF carries none). The UI seeds its controls from
    /// these instead of a global hardcoded set.
    sampling_temperature: f32,
    sampling_top_p: f32,
    sampling_top_k: u32,
    sampling_min_p: f32,
    sampling_repeat_pen: f32,
    sampling_from_model: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum GenState {
    Idle,
    Generating,
}

struct GenProgress {
    text: String,
    /// Tokens the model generated (not counting the prompt).
    tokens: usize,
    /// Prompt tokens processed before the first generated token.
    prompt_tokens: usize,
    elapsed_ms: u64,
    tokens_per_s: f64,
    state: GenState,
    error: Option<String>,
    /// Chain-of-thought streamed on the reasoning channel (show_think on).
    reason: String,
    reason_tokens: usize,
}

impl GenProgress {
    fn new() -> Self {
        GenProgress {
            text: String::new(),
            tokens: 0,
            prompt_tokens: 0,
            elapsed_ms: 0,
            tokens_per_s: 0.0,
            state: GenState::Idle,
            error: None,
            reason: String::new(),
            reason_tokens: 0,
        }
    }
}

/// One pending UI request (list / load / unload / status).
struct RequestSlot {
    inner: Mutex<ReqInner>,
}

struct ReqInner {
    done: bool,
    ok: bool,
    error: String,
    models: Vec<ModelEntry>,
    models_dir: String,
    model: Option<LoadedModel>,
    loaded: bool,
    generating: bool,
}

impl RequestSlot {
    fn new() -> Self {
        RequestSlot {
            inner: Mutex::new(ReqInner {
                done: false,
                ok: false,
                error: String::new(),
                models: Vec::new(),
                models_dir: String::new(),
                model: None,
                loaded: false,
                generating: false,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared app context
// ---------------------------------------------------------------------------

/// Fan-out hub so every connected SSE client (every browser tab) receives a
/// copy of every event. A single mpsc receiver would let a second tab steal
/// token events from the first and break its streaming.
#[derive(Default)]
struct EventHub {
    subs: HashMap<u64, mpsc::Sender<Json>>,
    next_sub: u64,
}

impl EventHub {
    fn subscribe(&mut self) -> (u64, mpsc::Receiver<Json>) {
        let (tx, rx) = mpsc::channel();
        self.next_sub += 1;
        let id = self.next_sub;
        self.subs.insert(id, tx);
        (id, rx)
    }

    fn unsubscribe(&mut self, id: u64) {
        self.subs.remove(&id);
    }

    fn publish(&mut self, ev: &Json) {
        // Unbounded channels; failed sends are disconnected clients, reaped here.
        self.subs.retain(|_, tx| tx.send(ev.clone()).is_ok());
    }
}

struct AppCtx {
    /// Broadcast hub for SSE clients (tokens, done, errors).
    events: Mutex<EventHub>,
    /// Pending request slots keyed by protocol request id.
    slots: Mutex<HashMap<i64, Arc<RequestSlot>>>,
    progress: Mutex<GenProgress>,
    /// True while a generation is in flight (UI-level guard).
    generating: AtomicBool,
    next_id: AtomicI64,
    /// Model-file downloader (HuggingFace -> project-root models/).
    dl: Arc<downloads::Manager>,
    /// Open chat tabs by id with their last heartbeat. When the map stays
    /// empty, nobody is watching anymore and the server exits (so closing
    /// the chat closes the server too).
    tabs: Mutex<HashMap<String, Instant>>,
    /// Set once the first tab ever says hello (boot grace).
    ever_seen_tab: AtomicBool,
}

impl AppCtx {
    fn new_id(&self) -> i64 {
        self.next_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn register_slot(&self, id: i64) -> Arc<RequestSlot> {
        let slot = Arc::new(RequestSlot::new());
        self.slots.lock().unwrap().insert(id, slot.clone());
        slot
    }

    fn drop_slot(&self, id: i64) {
        self.slots.lock().unwrap().remove(&id);
    }

    fn take_slot(&self, id: i64) -> Option<Arc<RequestSlot>> {
        self.slots.lock().unwrap().remove(&id)
    }

    fn push_event(&self, ev: Json) {
        self.events.lock().unwrap().publish(&ev);
    }

    fn finish_generation(&self, error: Option<String>, stopped: bool) {
        let p = {
            let mut g = self.progress.lock().unwrap();
            g.state = GenState::Idle;
            if let Some(ref e) = error {
                g.error = Some(e.clone());
            }
            GenProgress {
                text: g.text.clone(),
                tokens: g.tokens,
                prompt_tokens: g.prompt_tokens,
                elapsed_ms: g.elapsed_ms,
                tokens_per_s: g.tokens_per_s,
                state: GenState::Idle,
                error: g.error.clone(),
                reason: g.reason.clone(),
                reason_tokens: g.reason_tokens,
            }
        };
        let mut ev = Json::obj();
        ev.set("type", Json::str("done"));
        ev.set("tokens", Json::num(p.tokens as f64));
        ev.set("prompt_tokens", Json::num(p.prompt_tokens as f64));
        ev.set("elapsed_ms", Json::num(p.elapsed_ms as f64));
        ev.set("tokens_per_s", Json::num(p.tokens_per_s));
        ev.set("stopped", Json::Bool(stopped));
        ev.set("text", Json::str(p.text));
        if p.reason_tokens > 0 {
            ev.set("reason_tokens", Json::num(p.reason_tokens as f64));
            ev.set("reason_text", Json::str(p.reason));
        }
        if let Some(ref e) = p.error {
            ev.set("error", Json::str(e));
        }
        self.push_event(ev);
    }
}

// ---------------------------------------------------------------------------
// Service worker: owns the hearth-chat-cli --service child process
// ---------------------------------------------------------------------------

static SERVICE_CMD: OnceLock<mpsc::Sender<String>> = OnceLock::new();

fn send_cmd(value: &Json) {
    if let Some(tx) = SERVICE_CMD.get() {
        let _ = tx.send(value.to_string());
    }
}

fn spawn_service(ctx: Arc<AppCtx>) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {}", e))?;
    // hearth-ui.exe sits in target/{debug,release}/ next to hearth-chat-cli.exe.
    let cli = exe
        .parent()
        .map(|p| p.join("hearth-chat-cli.exe"))
        .unwrap_or_else(|| std::path::PathBuf::from("hearth-chat-cli.exe"));

    if !cli.exists() {
        return Err(format!(
            "hearth-chat-cli.exe not found next to hearth-ui.exe (expected {}). \
             Build the workspace with: cargo build --release",
            cli.display()
        ));
    }

    let mut child = std::process::Command::new(&cli)
        .arg("--service")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        // The service chatters on stderr during generation; discard it.
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn {}: {}", cli.display(), e))?;

    let stdin = child.stdin.take().ok_or("service: no stdin")?;
    let stdout = child.stdout.take().ok_or("service: no stdout")?;

    // Writer thread: forwards request lines to the service.
    let (cmd_tx, cmd_rx) = mpsc::channel::<String>();
    std::thread::Builder::new()
        .name("hearth-ui-stdin".into())
        .spawn(move || {
            let mut w = stdin;
            for line in cmd_rx {
                if writeln!(w, "{}", line).is_err() || w.flush().is_err() {
                    break;
                }
            }
        })
        .ok();

    // Reaper thread: if the service exits unexpectedly, notify and exit the UI.
    let ctx_reap = ctx.clone();
    std::thread::Builder::new()
        .name("hearth-ui-reap".into())
        .spawn(move || {
            let status = child.wait();
            let msg = match status {
                Ok(s) => format!("hearth-chat-cli service exited: {}", s),
                Err(e) => format!("hearth-chat-cli service wait failed: {}", e),
            };
            eprintln!("[hearth-ui] {}", msg);
            let mut ev = Json::obj();
            ev.set("type", Json::str("service_exited"));
            ev.set("message", Json::str(msg));
            ctx_reap.push_event(ev);
            // Give SSE clients a moment to see the event, then exit.
            std::thread::sleep(Duration::from_millis(300));
            std::process::exit(1);
        })
        .ok();

    // Reader thread: parses service lines, updates state, resolves slots,
    // streams tokens to SSE clients.
    std::thread::Builder::new()
        .name("hearth-ui-reader".into())
        .spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
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
                        eprintln!("[hearth-ui] bad JSON from service: {} — {:?}", e, trimmed);
                        continue;
                    }
                };
                handle_service_event(&ctx, value);
            }
            eprintln!("[hearth-ui] service stdout closed");
        })
        .ok();

    SERVICE_CMD
        .set(cmd_tx)
        .map_err(|_| "service channel already initialized".to_string())?;
    Ok(())
}

fn parse_loaded_model(v: &Json) -> LoadedModel {
    LoadedModel {
        name: v
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or("model")
            .to_string(),
        path: v
            .get("path")
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string(),
        arch: v
            .get("arch")
            .and_then(|x| x.as_str())
            .unwrap_or("?")
            .to_string(),
        n_layers: v.get("n_layers").and_then(|x| x.as_f64()).unwrap_or(0.0) as u32,
        d_model: v.get("d_model").and_then(|x| x.as_f64()).unwrap_or(0.0) as u32,
        n_heads: v.get("n_heads").and_then(|x| x.as_f64()).unwrap_or(0.0) as u32,
        n_kv_heads: v.get("n_kv_heads").and_then(|x| x.as_f64()).unwrap_or(0.0) as u32,
        d_ffn: v.get("d_ffn").and_then(|x| x.as_f64()).unwrap_or(0.0) as u32,
        vocab_size: v.get("vocab_size").and_then(|x| x.as_f64()).unwrap_or(0.0) as u32,
        max_seq_len: v.get("max_seq_len").and_then(|x| x.as_f64()).unwrap_or(0.0) as u32,
        kv_bytes: v.get("kv_bytes").and_then(|x| x.as_f64()).unwrap_or(0.0) as u64,
        template: v
            .get("template")
            .and_then(|x| x.as_str())
            .unwrap_or("?")
            .to_string(),
        sampling_temperature: get_f32(v, "sampling_temperature", 0.7),
        sampling_top_p: get_f32(v, "sampling_top_p", 0.9),
        sampling_top_k: v
            .get("sampling_top_k")
            .and_then(|x| x.as_f64())
            .unwrap_or(40.0) as u32,
        sampling_min_p: get_f32(v, "sampling_min_p", 0.0),
        sampling_repeat_pen: get_f32(v, "sampling_repeat_pen", 1.1),
        sampling_from_model: get_bool(v, "sampling_from_model", false),
    }
}

fn handle_service_event(ctx: &Arc<AppCtx>, v: Json) {
    let id = v.get("id").and_then(|x| x.as_f64()).unwrap_or(0.0) as i64;
    let event = v.get("event").and_then(|x| x.as_str()).unwrap_or("");

    match event {
        "ready" => eprintln!("[hearth-ui] service ready (protocol v1)"),
        "models" => {
            let models = v
                .get("models")
                .and_then(|x| x.as_arr())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| {
                            Some(ModelEntry {
                                name: m.get("name")?.as_str()?.to_string(),
                                path: m.get("path")?.as_str()?.to_string(),
                                size_bytes: m
                                    .get("size_bytes")
                                    .and_then(|x| x.as_f64())
                                    .unwrap_or(0.0)
                                    as u64,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let dir = v
                .get("dir")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            if let Some(slot) = ctx.take_slot(id) {
                let mut g = slot.inner.lock().unwrap();
                g.models = models;
                g.models_dir = dir;
                g.done = true;
                g.ok = true;
            }
        }
        "loaded" => {
            let loaded = v
                .get("model")
                .map(parse_loaded_model)
                .unwrap_or_else(|| LoadedModel {
                    name: "model".into(),
                    path: String::new(),
                    arch: "?".into(),
                    n_layers: 0,
                    d_model: 0,
                    n_heads: 0,
                    n_kv_heads: 0,
                    d_ffn: 0,
                    vocab_size: 0,
                    max_seq_len: 0,
                    kv_bytes: 0,
                    template: "?".into(),
                    sampling_temperature: 0.7,
                    sampling_top_p: 0.9,
                    sampling_top_k: 40,
                    sampling_min_p: 0.0,
                    sampling_repeat_pen: 1.1,
                    sampling_from_model: false,
                });
            if let Some(slot) = ctx.take_slot(id) {
                let mut g = slot.inner.lock().unwrap();
                g.model = Some(loaded);
                g.loaded = true;
                g.done = true;
                g.ok = true;
            }
        }
        "unloaded" => {
            if let Some(slot) = ctx.take_slot(id) {
                let mut g = slot.inner.lock().unwrap();
                g.loaded = false;
                g.done = true;
                g.ok = true;
            }
        }
        "status" => {
            let loaded = v.get("loaded").and_then(|x| x.as_bool()).unwrap_or(false);
            let model = if loaded {
                v.get("model").map(parse_loaded_model)
            } else {
                None
            };
            if let Some(slot) = ctx.take_slot(id) {
                let mut g = slot.inner.lock().unwrap();
                g.model = model;
                g.loaded = loaded;
                g.generating = v
                    .get("generating")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false);
                g.done = true;
                g.ok = true;
            }
        }
        "token" => {
            if let Some(text) = v.get("text").and_then(|x| x.as_str()) {
                {
                    let mut p = ctx.progress.lock().unwrap();
                    p.state = GenState::Generating;
                    p.text.push_str(text);
                    p.tokens += 1;
                }
                let mut ev = Json::obj();
                ev.set("type", Json::str("token"));
                ev.set("text", Json::str(text));
                ctx.push_event(ev);
            }
        }
        "reason" => {
            // Chain-of-thought fragment, streamed separately from the answer.
            if let Some(text) = v.get("text").and_then(|x| x.as_str()) {
                {
                    let mut p = ctx.progress.lock().unwrap();
                    p.state = GenState::Generating;
                    p.reason.push_str(text);
                    p.reason_tokens += 1;
                }
                let mut ev = Json::obj();
                ev.set("type", Json::str("reason"));
                ev.set("text", Json::str(text));
                ctx.push_event(ev);
            }
        }
        "done" => {
            let stopped = v.get("stopped").and_then(|x| x.as_bool()).unwrap_or(false);
            {
                let mut p = ctx.progress.lock().unwrap();
                p.tokens = v.get("tokens").and_then(|x| x.as_f64()).unwrap_or(0.0) as usize;
                if let Some(pt) = v.get("prompt_tokens").and_then(|x| x.as_f64()) {
                    p.prompt_tokens = pt as usize;
                }
                p.elapsed_ms = v.get("elapsed_ms").and_then(|x| x.as_f64()).unwrap_or(0.0) as u64;
                p.tokens_per_s = v
                    .get("tokens_per_s")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0);
                // Reasoning count is reported even when the stream was hidden;
                // the (optional) reasoning text only arrives when shown.
                if let Some(rt) = v.get("reason_tokens").and_then(|x| x.as_f64()) {
                    p.reason_tokens = rt as usize;
                }
                if let Some(txt) = v.get("reason_text").and_then(|x| x.as_str()) {
                    p.reason = txt.to_string();
                }
            }
            ctx.finish_generation(None, stopped);
            ctx.generating.store(false, Ordering::Relaxed);
        }
        "stopped" => {
            // Ack for the stop command; the cancel flag was already applied.
        }
        "error" => {
            let msg = v
                .get("error")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown service error")
                .to_string();
            // If a generation was in flight, finish it with the error.
            if ctx.generating.swap(false, Ordering::Relaxed) {
                ctx.finish_generation(Some(msg.clone()), false);
            }
            if let Some(slot) = ctx.take_slot(id) {
                let mut g = slot.inner.lock().unwrap();
                g.error = msg;
                g.done = true;
                g.ok = false;
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(reader: &mut BufReader<TcpStream>) -> Option<Request> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).ok()? == 0 {
            return None;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; content_length.min(4 * 1024 * 1024)];
    if !body.is_empty() && reader.read_exact(&mut body).is_err() {
        return None;
    }

    // Strip any query string; the API is POST-body driven.
    let path = target.split('?').next().unwrap_or(&target).to_string();

    Some(Request { method, path, body })
}

fn http_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: keep-alive\r\n\r\n",
        status,
        content_type,
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn json_response(stream: &mut TcpStream, value: &Json) {
    http_response(
        stream,
        "200 OK",
        "application/json; charset=utf-8",
        value.to_string().as_bytes(),
    );
}

fn error_response(stream: &mut TcpStream, message: &str) {
    let mut v = Json::obj();
    v.set("ok", Json::Bool(false));
    v.set("error", Json::str(message));
    json_response(stream, &v);
}

fn json_body(req: &Request) -> Json {
    let text = String::from_utf8_lossy(&req.body);
    json::parse(text.trim()).unwrap_or_else(|_| Json::obj())
}

fn get_str(v: &Json, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string()
}

fn get_f32(v: &Json, key: &str, default: f32) -> f32 {
    v.get(key)
        .and_then(|x| x.as_f64())
        .map(|n| n as f32)
        .unwrap_or(default)
}

fn get_usize(v: &Json, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(|x| x.as_usize())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

fn get_bool(v: &Json, key: &str, default: bool) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
}

fn wait_slot(slot: &RequestSlot, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if slot.inner.lock().unwrap().done {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

// ---------------------------------------------------------------------------
// API handlers
// ---------------------------------------------------------------------------

fn model_json(m: &LoadedModel) -> Json {
    let mut v = Json::obj();
    v.set("name", Json::str(m.name.clone()));
    v.set("path", Json::str(m.path.clone()));
    v.set("arch", Json::str(m.arch.clone()));
    v.set("n_layers", Json::num(m.n_layers as f64));
    v.set("d_model", Json::num(m.d_model as f64));
    v.set("n_heads", Json::num(m.n_heads as f64));
    v.set("n_kv_heads", Json::num(m.n_kv_heads as f64));
    v.set("d_ffn", Json::num(m.d_ffn as f64));
    v.set("vocab_size", Json::num(m.vocab_size as f64));
    v.set("max_seq_len", Json::num(m.max_seq_len as f64));
    v.set("kv_bytes", Json::num(m.kv_bytes as f64));
    v.set("template", Json::str(m.template.clone()));
    v.set(
        "sampling_temperature",
        Json::num(m.sampling_temperature as f64),
    );
    v.set("sampling_top_p", Json::num(m.sampling_top_p as f64));
    v.set("sampling_top_k", Json::num(m.sampling_top_k as f64));
    v.set("sampling_min_p", Json::num(m.sampling_min_p as f64));
    v.set(
        "sampling_repeat_pen",
        Json::num(m.sampling_repeat_pen as f64),
    );
    v.set("sampling_from_model", Json::Bool(m.sampling_from_model));
    v
}

fn handle_models(ctx: &Arc<AppCtx>, stream: &mut TcpStream) {
    let id = ctx.new_id();
    let slot = ctx.register_slot(id);
    let mut cmd = Json::obj();
    cmd.set("id", Json::Num(id as f64));
    cmd.set("cmd", Json::str("list_models"));
    send_cmd(&cmd);

    if wait_slot(&slot, Duration::from_secs(20)) {
        let g = slot.inner.lock().unwrap();
        let mut v = Json::obj();
        v.set("ok", Json::Bool(true));
        v.set("dir", Json::str(g.models_dir.clone()));
        let items: Vec<Json> = g
            .models
            .iter()
            .map(|m| {
                let mut e = Json::obj();
                e.set("name", Json::str(m.name.clone()));
                e.set("path", Json::str(m.path.clone()));
                e.set("size_bytes", Json::num(m.size_bytes as f64));
                e
            })
            .collect();
        v.set("models", Json::Arr(items));
        json_response(stream, &v);
    } else {
        error_response(stream, "list_models timed out");
    }
    ctx.drop_slot(id);
}

fn handle_load(ctx: &Arc<AppCtx>, stream: &mut TcpStream, body: &Json) {
    let path = get_str(body, "path");
    if path.trim().is_empty() {
        error_response(stream, "load: missing path");
        return;
    }
    let path = path.as_str();
    if ctx.generating.load(Ordering::Relaxed) {
        error_response(stream, "cannot switch models while generating");
        return;
    }

    // Switching models: unload whatever is loaded (no-op reply if none).
    let unload_id = ctx.new_id();
    let unload_slot = ctx.register_slot(unload_id);
    let mut c = Json::obj();
    c.set("id", Json::Num(unload_id as f64));
    c.set("cmd", Json::str("unload"));
    send_cmd(&c);
    let _ = wait_slot(&unload_slot, Duration::from_secs(30));
    ctx.drop_slot(unload_id);

    let id = ctx.new_id();
    let slot = ctx.register_slot(id);
    let mut cmd = Json::obj();
    cmd.set("id", Json::Num(id as f64));
    cmd.set("cmd", Json::str("load"));
    cmd.set("path", Json::str(path));
    // Device/residency are adjustable per load (the engine used to be forced
    // to CPU-only from this path). Defaults to CPU when the client sends
    // nothing.
    let device = get_str(body, "device");
    if !device.is_empty() {
        cmd.set("device", Json::str(device));
    }
    if let Some(n) = body
        .get("gpu_layers")
        .and_then(|x| x.as_usize())
        .filter(|n| *n > 0)
    {
        cmd.set("gpu_layers", Json::num(n as f64));
    }
    if get_bool(body, "pin_weights", false) {
        cmd.set("pin_weights", Json::Bool(true));
    }
    send_cmd(&cmd);

    // Large GGUFs take a while; 10 minutes covers the biggest local files.
    if wait_slot(&slot, Duration::from_secs(600)) {
        let g = slot.inner.lock().unwrap();
        if g.ok {
            let mut v = Json::obj();
            v.set("ok", Json::Bool(true));
            if let Some(ref m) = g.model {
                v.set("model", model_json(m));
            }
            json_response(stream, &v);
        } else {
            error_response(stream, &g.error);
        }
    } else {
        error_response(stream, "load timed out");
    }
    ctx.drop_slot(id);
}

fn handle_unload(ctx: &Arc<AppCtx>, stream: &mut TcpStream) {
    let id = ctx.new_id();
    let slot = ctx.register_slot(id);
    let mut cmd = Json::obj();
    cmd.set("id", Json::Num(id as f64));
    cmd.set("cmd", Json::str("unload"));
    send_cmd(&cmd);

    if wait_slot(&slot, Duration::from_secs(60)) {
        let mut v = Json::obj();
        v.set("ok", Json::Bool(true));
        json_response(stream, &v);
    } else {
        error_response(stream, "unload timed out");
    }
    ctx.drop_slot(id);
}

fn handle_status(ctx: &Arc<AppCtx>, stream: &mut TcpStream) {
    let id = ctx.new_id();
    let slot = ctx.register_slot(id);
    let mut cmd = Json::obj();
    cmd.set("id", Json::Num(id as f64));
    cmd.set("cmd", Json::str("status"));
    send_cmd(&cmd);

    if wait_slot(&slot, Duration::from_secs(20)) {
        let g = slot.inner.lock().unwrap();
        let mut v = Json::obj();
        v.set("ok", Json::Bool(true));
        v.set("loaded", Json::Bool(g.loaded));
        if let Some(ref m) = g.model {
            v.set("model", model_json(m));
        }
        v.set("generating", Json::Bool(g.generating));
        json_response(stream, &v);
    } else {
        error_response(stream, "status timed out");
    }
    ctx.drop_slot(id);
}

fn handle_generate(ctx: &Arc<AppCtx>, req: &Request, stream: &mut TcpStream) {
    if ctx
        .generating
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        error_response(stream, "already generating");
        return;
    }

    let body = json_body(req);
    let message = get_str(&body, "message");
    if message.trim().is_empty() {
        ctx.generating.store(false, Ordering::Relaxed);
        error_response(stream, "message is empty");
        return;
    }

    // Reset stream state for this generation.
    {
        let mut p = ctx.progress.lock().unwrap();
        *p = GenProgress::new();
    }

    // History as [[user, assistant], ...].
    let mut history = Vec::new();
    if let Some(items) = body.get("history").and_then(|x| x.as_arr()) {
        for item in items {
            if let Some(pair) = item.as_arr() {
                if pair.len() == 2 {
                    let u = pair[0].as_str().unwrap_or_default();
                    let a = pair[1].as_str().unwrap_or_default();
                    history.push(Json::Arr(vec![Json::str(u), Json::str(a)]));
                }
            }
        }
    }

    let id = ctx.new_id();
    let mut cmd = Json::obj();
    cmd.set("id", Json::Num(id as f64));
    cmd.set("cmd", Json::str("generate"));
    cmd.set("message", Json::str(message));
    cmd.set("system", Json::str(get_str(&body, "system")));
    cmd.set("history", Json::Arr(history));
    // Forward only the sampling fields the client actually supplied; anything
    // omitted falls back to the model's own `general.sampling.*` defaults in
    // the service. This keeps per-request inference settings client-driven.
    for (body_key, cmd_key) in [
        ("temp", "temp"),
        ("top_p", "top_p"),
        ("min_p", "min_p"),
        ("typical_p", "typical_p"),
        ("repeat_pen", "repeat_pen"),
        ("seed", "seed"),
    ] {
        if let Some(v) = body.get(body_key).and_then(|x| x.as_f64()) {
            cmd.set(cmd_key, Json::num(v));
        }
    }
    if let Some(k) = body
        .get("top_k")
        .and_then(|x| x.as_usize())
        .filter(|n| *n > 0)
    {
        cmd.set("top_k", Json::num(k as f64));
    }
    cmd.set(
        "max_tokens",
        Json::num(get_usize(&body, "max_tokens", 512) as f64),
    );
    if let Some(c) = body
        .get("max_context")
        .and_then(|x| x.as_usize())
        .filter(|n| *n > 0)
    {
        cmd.set("max_context", Json::num(c as f64));
    }
    let show_think = get_bool(&body, "show_think", false);
    cmd.set("show_think", Json::Bool(show_think));
    // Reasoning is enabled unless the UI explicitly turns it off; by default a
    // hidden-think request still lets the model reason (it's just not shown).
    let enable_think = match body.get("enable_think") {
        Some(v) => v.as_bool().unwrap_or(true),
        None => true,
    };
    cmd.set("enable_think", Json::Bool(enable_think));
    send_cmd(&cmd);

    // Respond immediately; tokens stream over /api/events.
    let mut v = Json::obj();
    v.set("ok", Json::Bool(true));
    v.set("gen_id", Json::Num(id as f64));
    json_response(stream, &v);
}

// ---------------------------------------------------------------------------
// Model downloads (HuggingFace -> project-root models/)
// ---------------------------------------------------------------------------

fn handle_downloads(ctx: &Arc<AppCtx>, stream: &mut TcpStream) {
    json_response(stream, &ctx.dl.snapshot());
}

fn handle_dl_start(ctx: &Arc<AppCtx>, req: &Request, stream: &mut TcpStream) {
    let body = json_body(req);
    let id = body
        .get("id")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    let url = body
        .get("url")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    let filename = body
        .get("filename")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    match ctx.dl.start(id, url, filename) {
        Ok((id, state)) => {
            let mut v = Json::obj();
            v.set("ok", Json::Bool(true));
            v.set("id", Json::str(id));
            v.set("state", Json::str(state));
            json_response(stream, &v);
        }
        Err(e) => error_response(stream, &e),
    }
}

// ---------------------------------------------------------------------------
// Tab presence: the server exits when no chat tab is watching anymore.
// ---------------------------------------------------------------------------

fn tab_id(req: &Request) -> String {
    json_body(req)
        .get("tab")
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string()
}

fn handle_hello(ctx: &Arc<AppCtx>, req: &Request, stream: &mut TcpStream) {
    let id = tab_id(req);
    if !id.is_empty() {
        ctx.tabs.lock().unwrap().insert(id, Instant::now());
        ctx.ever_seen_tab.store(true, Ordering::Relaxed);
    }
    let mut v = Json::obj();
    v.set("ok", Json::Bool(true));
    json_response(stream, &v);
}

fn handle_ping(ctx: &Arc<AppCtx>, req: &Request, stream: &mut TcpStream) {
    let id = tab_id(req);
    if !id.is_empty() {
        ctx.tabs.lock().unwrap().insert(id, Instant::now());
    }
    let mut v = Json::obj();
    v.set("ok", Json::Bool(true));
    json_response(stream, &v);
}

fn handle_bye(ctx: &Arc<AppCtx>, req: &Request, stream: &mut TcpStream) {
    let id = tab_id(req);
    if !id.is_empty() {
        ctx.tabs.lock().unwrap().remove(&id);
    }
    let mut v = Json::obj();
    v.set("ok", Json::Bool(true));
    json_response(stream, &v);
}

/// Background reaper: when no tab has been seen for a while (and nothing is
/// generating or downloading), the chat is closed — exit so the server does
/// not linger. Reloads re-register within seconds, well inside the grace.
fn spawn_tab_reaper(ctx: Arc<AppCtx>) {
    // Testing escape hatch (screenshots, endpoint probes): set
    // HEARTH_NO_AUTO_SHUTDOWN=1 to keep the server up with no tabs open.
    // Default (unset) is close-chat-closes-server.
    if std::env::var("HEARTH_NO_AUTO_SHUTDOWN").map(|v| v == "1").unwrap_or(false) {
        return;
    }
    std::thread::Builder::new()
        .name("hearth-ui-reaper".into())
        .spawn(move || {
            let grace = Duration::from_secs(20);
            loop {
                std::thread::sleep(Duration::from_secs(5));
                if !ctx.ever_seen_tab.load(Ordering::Relaxed) {
                    continue;
                }
                // Drop heartbeats older than the grace window; live tabs ping
                // every 15 s, reloads re-register within a couple of seconds.
                let now = Instant::now();
                let mut tabs = ctx.tabs.lock().unwrap();
                tabs.retain(|_, seen| now.duration_since(*seen) < grace);
                let empty = tabs.is_empty();
                drop(tabs);
                if empty && !ctx.generating.load(Ordering::Relaxed) && !ctx.dl.has_active() {
                    // Confirm the quiet held across one more check (reload race).
                    std::thread::sleep(Duration::from_secs(5));
                    if ctx.tabs.lock().unwrap().is_empty()
                        && !ctx.generating.load(Ordering::Relaxed)
                        && !ctx.dl.has_active()
                    {
                        eprintln!("[hearth-ui] no open tabs — shutting down");
                        std::process::exit(0);
                    }
                }
            }
        })
        .ok();
}

fn handle_dl_cancel(ctx: &Arc<AppCtx>, req: &Request, stream: &mut TcpStream) {
    let body = json_body(req);
    let id = get_str(&body, "id");
    if id.trim().is_empty() {
        error_response(stream, "cancel: missing id");
        return;
    }
    if ctx.dl.cancel(&id) {
        let mut v = Json::obj();
        v.set("ok", Json::Bool(true));
        v.set("id", Json::str(id));
        json_response(stream, &v);
    } else {
        error_response(stream, "nothing to cancel for that id");
    }
}

fn handle_stop(_ctx: &Arc<AppCtx>, stream: &mut TcpStream) {
    let mut cmd = Json::obj();
    cmd.set("id", Json::Num(0.0));
    cmd.set("cmd", Json::str("stop"));
    send_cmd(&cmd);
    // The "done" event from the service clears `generating`; the UI shows a
    // stopping state until then.
    let mut v = Json::obj();
    v.set("ok", Json::Bool(true));
    json_response(stream, &v);
}

// ---------------------------------------------------------------------------
// SSE streaming
// ---------------------------------------------------------------------------

fn sse_loop(ctx: &Arc<AppCtx>, stream: &mut TcpStream) {
    let _ = stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\n\r\n",
    );
    let _ = stream.flush();

    // Subscribe to the broadcast hub: the connection stays open across
    // generations until the client disconnects or the server exits.
    let (sub_id, rx) = ctx.events.lock().unwrap().subscribe();
    let mut last_keepalive = Instant::now();
    loop {
        let mut batch: Vec<Json> = Vec::new();
        loop {
            match rx.recv_timeout(Duration::from_millis(25)) {
                Ok(ev) => batch.push(ev),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            if batch.len() >= 64 {
                break;
            }
        }

        for ev in &batch {
            let payload = format!("data: {}\n\n", ev);
            if stream.write_all(payload.as_bytes()).is_err() {
                // Client went away; the hub reaps our sender on next publish.
                ctx.events.lock().unwrap().unsubscribe(sub_id);
                return;
            }
        }
        if !batch.is_empty() && stream.flush().is_err() {
            ctx.events.lock().unwrap().unsubscribe(sub_id);
            return;
        }

        if last_keepalive.elapsed() > Duration::from_secs(15) {
            if stream.write_all(b": keepalive\n\n").is_err() || stream.flush().is_err() {
                return;
            }
            last_keepalive = Instant::now();
        }
    }
}

// ---------------------------------------------------------------------------
// Connection handling
// ---------------------------------------------------------------------------

fn handle_client(ctx: Arc<AppCtx>, stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(300)));
    let peer = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut reader = BufReader::new(peer);
    let mut stream = stream;

    loop {
        let req = match read_request(&mut reader) {
            Some(r) => r,
            None => return,
        };

        let path = req.path.trim_end_matches('/');
        let path = if path.is_empty() { "/" } else { path };

        match (req.method.as_str(), path) {
            ("GET", "/") => {
                http_response(
                    &mut stream,
                    "200 OK",
                    "text/html; charset=utf-8",
                    web::INDEX_HTML.as_bytes(),
                );
            }
            ("GET", "/api/models") => handle_models(&ctx, &mut stream),
            ("GET", "/api/downloads") => handle_downloads(&ctx, &mut stream),
            ("POST", "/api/downloads/start") => handle_dl_start(&ctx, &req, &mut stream),
            ("POST", "/api/downloads/cancel") => handle_dl_cancel(&ctx, &req, &mut stream),
            ("POST", "/api/hello") => handle_hello(&ctx, &req, &mut stream),
            ("POST", "/api/ping") => handle_ping(&ctx, &req, &mut stream),
            ("POST", "/api/bye") => handle_bye(&ctx, &req, &mut stream),
            ("POST", "/api/load") => {
                let body = json_body(&req);
                handle_load(&ctx, &mut stream, &body);
            }
            ("POST", "/api/unload") => handle_unload(&ctx, &mut stream),
            ("GET", "/api/status") => handle_status(&ctx, &mut stream),
            ("POST", "/api/generate") => handle_generate(&ctx, &req, &mut stream),
            ("POST", "/api/stop") => handle_stop(&ctx, &mut stream),
            ("GET", "/api/events") => {
                // Streams until the client disconnects or the server shuts
                // down; stays open across generations.
                sse_loop(&ctx, &mut stream);
                return;
            }
            _ => {
                http_response(&mut stream, "404 Not Found", "text/plain", b"not found");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let mut port = 8899u16;
    let mut open = true;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or(8899);
                    i += 1;
                }
            }
            "--no-browser" => {
                open = false;
            }
            other => {
                eprintln!(
                    "Unknown argument: {}\nUsage: hearth-ui [--port N] [--no-browser]",
                    other
                );
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Bind the requested port, scanning a few ahead if it is taken.
    let listener = {
        let mut chosen = None;
        for p in port..port.saturating_add(20) {
            if let Ok(l) = TcpListener::bind(("127.0.0.1", p)) {
                chosen = Some(l);
                port = p;
                break;
            }
        }
        match chosen {
            Some(l) => l,
            None => {
                eprintln!("Could not bind 127.0.0.1:{}-{}", port, port + 19);
                std::process::exit(1);
            }
        }
    };

    let dl = downloads::Manager::new();
    let ctx = Arc::new(AppCtx {
        events: Mutex::new(EventHub::default()),
        slots: Mutex::new(HashMap::new()),
        progress: Mutex::new(GenProgress::new()),
        generating: AtomicBool::new(false),
        next_id: AtomicI64::new(0),
        dl: dl.clone(),
        tabs: Mutex::new(HashMap::new()),
        ever_seen_tab: AtomicBool::new(false),
    });
    // Download progress/completion fans out to every open tab over SSE.
    {
        let ctx_ev = ctx.clone();
        dl.set_sink(Arc::new(move |ev: Json| ctx_ev.push_event(ev)));
    }
    spawn_tab_reaper(ctx.clone());

    if let Err(e) = spawn_service(ctx.clone()) {
        eprintln!("[hearth-ui] fatal: {}", e);
        std::process::exit(1);
    }

    let url = format!("http://127.0.0.1:{}", port);
    println!("Hearth chat UI running at {}", url);
    println!("Ctrl+C to quit.");
    if open {
        open_browser(&url);
    }

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let ctx = ctx.clone();
                std::thread::Builder::new()
                    .name("hearth-ui-conn".into())
                    .spawn(move || handle_client(ctx, stream))
                    .ok();
            }
            Err(e) => eprintln!("[hearth-ui] accept error: {}", e),
        }
    }
}

fn open_browser(url: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}
