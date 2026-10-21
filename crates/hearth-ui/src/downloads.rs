//! Model downloader for hearth-ui — std only.
//!
//! The weights can't live on github, so the UI fetches them straight from
//! HuggingFace into the project-root `models/` dir. One download runs at a
//! time (laptop-friendly); extra requests queue FIFO. Each entry keeps its
//! own progress so the frontend can render per-model bars and let the user
//! pick which models to fetch.
//!
//! Plain-`http://` transfers use the hand-rolled client in `http.rs` (no
//! subprocess); `https://` still runs in the system `curl` subprocess until
//! the hand-rolled TLS transport lands, so users without curl are covered as
//! soon as it does. Either way: resume, redirect handling, and no new
//! dependencies for this crate.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use hearth_core::json::Json;

const HF: &str = "https://huggingface.co/prism-ml";

/// One fetchable model. URLs are the verified `.../resolve/main/<file>`
/// links; every one below was HEAD-checked with a 200 before being listed.
#[derive(Clone)]
pub struct CatalogEntry {
    pub id: String,
    pub filename: String,
    pub url: String,
    pub size_bytes: u64,
    pub label: String,
    pub blurb: String,
}

fn entry(
    id: &str,
    repo: &str,
    filename: &str,
    size_bytes: u64,
    label: &str,
    blurb: &str,
) -> CatalogEntry {
    CatalogEntry {
        id: id.to_string(),
        filename: filename.to_string(),
        url: format!("{}/{}/resolve/main/{}", HF, repo, filename),
        size_bytes,
        label: label.to_string(),
        blurb: blurb.to_string(),
    }
}

pub fn builtin_catalog() -> Vec<CatalogEntry> {
    vec![
        entry(
            "bonsai-1.7b-q1_0",
            "Bonsai-1.7B-gguf",
            "Bonsai-1.7B-Q1_0.gguf",
            248_302_272,
            "Bonsai 1.7B · 1-bit",
            "Fastest chat on any CPU. Start here.",
        ),
        entry(
            "ternary-bonsai-1.7b-q2_0",
            "Ternary-Bonsai-1.7B-gguf",
            "Ternary-Bonsai-1.7B-Q2_0.gguf",
            463_290_464,
            "Ternary Bonsai 1.7B · 2-bit",
            "Higher-quality 1.7B, still quick.",
        ),
        entry(
            "bonsai-4b-q1_0",
            "Bonsai-4B-gguf",
            "Bonsai-4B-Q1_0.gguf",
            572_270_624,
            "Bonsai 4B · 1-bit",
            "Balanced size and quality.",
        ),
        entry(
            "ternary-bonsai-4b-q2_0",
            "Ternary-Bonsai-4B-gguf",
            "Ternary-Bonsai-4B-Q2_0.gguf",
            1_074_969_344,
            "Ternary Bonsai 4B · 2-bit",
            "Better 4B reasoning.",
        ),
        entry(
            "bonsai-8b-q1_0",
            "Bonsai-8B-gguf",
            "Bonsai-8B-Q1_0.gguf",
            1_158_654_496,
            "Bonsai 8B · 1-bit",
            "Strong quality at ~13 tok/s.",
        ),
        entry(
            "ternary-bonsai-8b-q2_0",
            "Ternary-Bonsai-8B-gguf",
            "Ternary-Bonsai-8B-Q2_0.gguf",
            2_182_184_672,
            "Ternary Bonsai 8B · 2-bit",
            "Best 8B quality, slower.",
        ),
        entry(
            "ternary-bonsai-2-2b-ptq1_0",
            "Ternary-Bonsai-2-2B-gguf",
            "Ternary-Bonsai-2-2B-PTQ1_0.gguf",
            429_183_456,
            "Ternary Bonsai 2 2B · PTQ1_0",
            "Newest-gen small model, 1-bit.",
        ),
        entry(
            "ternary-bonsai-2-2b-pq2_0",
            "Ternary-Bonsai-2-2B-gguf",
            "Ternary-Bonsai-2-2B-PQ2_0.gguf",
            517_312_992,
            "Ternary Bonsai 2 2B · PQ2_0",
            "Newest-gen small model, 2-bit.",
        ),
        entry(
            "ternary-bonsai-2-4b-ptq1_0",
            "Ternary-Bonsai-2-4B-gguf",
            "Ternary-Bonsai-2-4B-PTQ1_0.gguf",
            949_549_920,
            "Ternary Bonsai 2 4B · PTQ1_0",
            "Newest-gen 4B, 1-bit.",
        ),
        entry(
            "ternary-bonsai-2-4b-pq2_0",
            "Ternary-Bonsai-2-4B-gguf",
            "Ternary-Bonsai-2-4B-PQ2_0.gguf",
            1_146_465_120,
            "Ternary Bonsai 2 4B · PQ2_0",
            "Newest-gen 4B, 2-bit.",
        ),
        entry(
            "ternary-bonsai-2-9b-ptq1_0",
            "Ternary-Bonsai-2-9B-gguf",
            "Ternary-Bonsai-2-9B-PTQ1_0.gguf",
            1_997_486_976,
            "Ternary Bonsai 2 9B · PTQ1_0",
            "Large newest-gen, 1-bit.",
        ),
        entry(
            "ternary-bonsai-2-9b-pq2_0",
            "Ternary-Bonsai-2-9B-gguf",
            "Ternary-Bonsai-2-9B-PQ2_0.gguf",
            2_416_851_840,
            "Ternary Bonsai 2 9B · PQ2_0",
            "Large newest-gen, 2-bit.",
        ),
        entry(
            "ternary-bonsai-2-27b-ptq1_0",
            "Ternary-Bonsai-2-27B-gguf",
            "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            5_946_648_928,
            "Ternary Bonsai 2 27B · PTQ1_0",
            "Flagship reasoning. ~1 tok/s — slow but visible.",
        ),
        entry(
            "ternary-bonsai-2-27b-pq2_0",
            "Ternary-Bonsai-2-27B-gguf",
            "Ternary-Bonsai-2-27B-PQ2_0.gguf",
            7_206_168_928,
            "Ternary Bonsai 2 27B · PQ2_0",
            "Flagship quality. Needs patience and RAM.",
        ),
    ]
}

/// Where new weights go. Shares the engine's own resolution
/// (`hearth_core::paths`) so the downloader and the model list can never
/// disagree about which directory is *the* directory.
pub fn resolve_models_dir() -> PathBuf {
    hearth_core::paths::primary_models_dir()
}

/// The download dir first, then every other root the engine lists — a model
/// the user already has in `%APPDATA%\hearth\models` counts as downloaded
/// instead of being fetched a second time.
fn roots_with(dir: &Path) -> Vec<PathBuf> {
    let mut roots = vec![dir.to_path_buf()];
    for root in hearth_core::paths::models_roots() {
        if !roots.iter().any(|seen| seen == &root) {
            roots.push(root);
        }
    }
    roots
}

/// Largest copy of `filename` across `roots` (0 when absent everywhere).
fn best_len(roots: &[PathBuf], filename: &str) -> u64 {
    roots
        .iter()
        .filter_map(|root| std::fs::metadata(root.join(filename)).ok())
        .map(|m| m.len())
        .max()
        .unwrap_or(0)
}

fn sanitize_filename(name: &str) -> Option<String> {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    if !base.to_lowercase().ends_with(".gguf") || base.len() > 128 {
        return None;
    }
    if base
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        Some(base.to_string())
    } else {
        None
    }
}

#[derive(Clone, PartialEq)]
enum DlPhase {
    Queued,
    Active,
    Done,
    Error,
    Cancelled,
}

struct ActiveDl {
    id: String,
    cancel: Arc<AtomicBool>,
}

struct Inner {
    dir: PathBuf,
    catalog: Vec<CatalogEntry>,
    /// Custom URL entries added through the UI this session.
    custom: Vec<CatalogEntry>,
    queue: VecDeque<String>,
    active: Option<ActiveDl>,
    /// Terminal states remembered for this session (done/error/cancelled).
    terminal: HashMap<String, (DlPhase, String)>,
    sink: Option<Arc<dyn Fn(Json) + Send + Sync>>,
    custom_seq: u64,
}

pub struct Manager {
    inner: Mutex<Inner>,
    kick: mpsc::Sender<()>,
}

impl Manager {
    pub fn new() -> Arc<Self> {
        let dir = resolve_models_dir();
        let catalog = load_catalog_override(&dir).unwrap_or_else(builtin_catalog);
        let (kick_tx, kick_rx) = mpsc::channel::<()>();
        let mgr = Arc::new(Manager {
            kick: kick_tx,
            inner: Mutex::new(Inner {
                dir,
                catalog,
                custom: Vec::new(),
                queue: VecDeque::new(),
                active: None,
                terminal: HashMap::new(),
                sink: None,
                custom_seq: 0,
            }),
        });
        // Single worker: FIFO, one transfer at a time.
        let worker = mgr.clone();
        std::thread::Builder::new()
            .name("hearth-ui-downloads".into())
            .spawn(move || {
                while kick_rx.recv().is_ok() {
                    worker.pump();
                }
            })
            .ok();
        mgr
    }

    pub fn set_sink(&self, f: Arc<dyn Fn(Json) + Send + Sync>) {
        self.inner.lock().unwrap().sink = Some(f);
    }

    fn emit(&self, id: &str, phase: &str, downloaded: u64, total: u64, error: &str) {
        let sink = self.inner.lock().unwrap().sink.clone();
        if let Some(sink) = sink {
            let mut ev = Json::obj();
            ev.set("type", Json::str("download".to_string()));
            ev.set("id", Json::str(id.to_string()));
            ev.set("state", Json::str(phase.to_string()));
            ev.set("downloaded_bytes", Json::num(downloaded as f64));
            ev.set("size_bytes", Json::num(total as f64));
            if !error.is_empty() {
                ev.set("error", Json::str(error.to_string()));
            }
            sink(ev);
        }
    }

    fn find(&self, id: &str) -> Option<CatalogEntry> {
        let g = self.inner.lock().unwrap();
        g.catalog
            .iter()
            .chain(g.custom.iter())
            .find(|e| e.id == id)
            .cloned()
    }

    fn on_disk(&self, e: &CatalogEntry) -> u64 {
        let dir = self.inner.lock().unwrap().dir.clone();
        best_len(&roots_with(&dir), &e.filename)
    }

    /// Snapshot for `GET /api/downloads`.
    pub fn snapshot(&self) -> Json {
        let (dir, entries) = {
            let g = self.inner.lock().unwrap();
            let mut all = g.catalog.clone();
            all.extend(g.custom.clone());
            (g.dir.clone(), all)
        };
        let roots = roots_with(&dir);
        let items: Vec<Json> = entries
            .iter()
            .map(|e| {
                // Final file (completion truth) vs .part-or-final (shown bytes,
                // so a leftover .part renders instead of an empty bar).
                let have_final = best_len(&roots, &e.filename);
                let shown = part_len(&dir, &e.filename).max(have_final);
                let (mut phase, mut err) = {
                    let g = self.inner.lock().unwrap();
                    if let Some(a) = g.active.as_ref() {
                        if a.id == e.id {
                            (DlPhase::Active, String::new())
                        } else if let Some(t) = g.terminal.get(&e.id) {
                            (t.0.clone(), t.1.clone())
                        } else {
                            (DlPhase::Queued, String::new())
                        }
                    } else if g.queue.contains(&e.id) {
                        (DlPhase::Queued, String::new())
                    } else if let Some(t) = g.terminal.get(&e.id) {
                        (t.0.clone(), t.1.clone())
                    } else {
                        (DlPhase::Queued, String::new())
                    }
                };
                // A complete file on disk always reads as done, even across restarts.
                let complete = e.size_bytes > 0 && have_final >= e.size_bytes
                    || e.size_bytes == 0 && have_final > 0;
                let mut downloaded = shown;
                if complete {
                    phase = DlPhase::Done;
                    err = String::new();
                    downloaded = if e.size_bytes > 0 {
                        e.size_bytes
                    } else {
                        have_final
                    };
                }
                let phase_s = match phase {
                    DlPhase::Queued => {
                        if shown > 0 && !complete {
                            "partial"
                        } else {
                            "idle"
                        }
                    }
                    DlPhase::Active => "active",
                    DlPhase::Done => "done",
                    DlPhase::Error => "error",
                    DlPhase::Cancelled => "cancelled",
                };
                let mut o = Json::obj();
                o.set("id", Json::str(e.id.clone()));
                o.set("filename", Json::str(e.filename.clone()));
                o.set("label", Json::str(e.label.clone()));
                o.set("blurb", Json::str(e.blurb.clone()));
                o.set("url", Json::str(e.url.clone()));
                o.set("size_bytes", Json::num(e.size_bytes as f64));
                o.set("state", Json::str(phase_s.to_string()));
                o.set("downloaded_bytes", Json::num(downloaded as f64));
                o.set("exists", Json::Bool(complete));
                if !err.is_empty() {
                    o.set("error", Json::str(err));
                }
                o
            })
            .collect();
        let mut v = Json::obj();
        v.set("ok", Json::Bool(true));
        v.set("dir", Json::str(dir.to_string_lossy().to_string()));
        v.set("curl", Json::Bool(curl_available()));
        let items_json: Vec<Json> = items;
        v.set("models", Json::Arr(items_json));
        v
    }

    /// Queue a catalog entry (or a custom URL) for download. Returns the id
    /// and the phase it landed in.
    pub fn start(
        &self,
        id: Option<String>,
        url: Option<String>,
        filename: Option<String>,
    ) -> Result<(String, String), String> {
        if !curl_available() {
            return Err("curl was not found on PATH. Install curl, or copy .gguf files into the models folder manually.".to_string());
        }
        let target_id = if let Some(u) = url {
            let u = u.trim().to_string();
            // Plaintext fetch of a multi-GB binary is refused unless the
            // operator explicitly opts in (local mirrors, offline labs).
            let allow_http = std::env::var("HEARTH_DL_ALLOW_HTTP")
                .map(|v| v == "1")
                .unwrap_or(false);
            if u.starts_with("http://") && !allow_http {
                return Err(
                    "plain http:// refused (https only). Set HEARTH_DL_ALLOW_HTTP=1 to override."
                        .to_string(),
                );
            }
            if !(u.starts_with("https://") || u.starts_with("http://")) {
                return Err("URL must start with https://".to_string());
            }
            let fname = filename
                .and_then(|f| sanitize_filename(&f))
                .ok_or_else(|| {
                    "filename must be a plain name ending in .gguf (e.g. Bonsai-1.7B-Q1_0.gguf)"
                        .to_string()
                })?;
            let mut g = self.inner.lock().unwrap();
            g.custom_seq += 1;
            let cid = format!("custom-{}", g.custom_seq);
            let label = fname.replace(".gguf", "").replace(['-', '_'], " ");
            g.custom.push(CatalogEntry {
                id: cid.clone(),
                filename: fname,
                url: u,
                size_bytes: 0,
                label,
                blurb: "Custom link added in this session.".to_string(),
            });
            cid
        } else {
            id.ok_or_else(|| "start: missing id (or url + filename)".to_string())?
        };
        {
            let g = self.inner.lock().unwrap();
            if !g
                .catalog
                .iter()
                .chain(g.custom.iter())
                .any(|e| e.id == target_id)
            {
                return Err("unknown model id".to_string());
            }
            if let Some(a) = g.active.as_ref() {
                if a.id == target_id {
                    return Ok((target_id, "active".to_string()));
                }
            }
            if g.queue.contains(&target_id) {
                return Ok((target_id, "queued".to_string()));
            }
        }
        // Already complete on disk? Report done without queueing. A complete
        // .part with no final file (cancelled at the last byte, crashed
        // rename) heals itself here via rename instead of re-downloading.
        if let Some(e) = self.find(&target_id) {
            let dir = self.inner.lock().unwrap().dir.clone();
            let have = self.on_disk(&e);
            if e.size_bytes > 0 && have >= e.size_bytes {
                return Ok((target_id, "done".to_string()));
            }
            if e.size_bytes > 0
                && part_len(&dir, &e.filename) >= e.size_bytes
                && have < e.size_bytes
            {
                // Complete .part but incomplete final: adopt the partial.
                let part = dir.join(format!("{}.part", e.filename));
                let dest = dir.join(&e.filename);
                let _ = std::fs::remove_file(&dest);
                if std::fs::rename(&part, &dest).is_ok() {
                    return Ok((target_id, "done".to_string()));
                }
            }
        }
        {
            let mut g = self.inner.lock().unwrap();
            g.terminal.remove(&target_id);
            g.queue.push_back(target_id.clone());
        }
        let _ = self.kick.send(());
        Ok((target_id, "queued".to_string()))
    }

    pub fn cancel(&self, id: &str) -> bool {
        let mut g = self.inner.lock().unwrap();
        let was_queued = g.queue.iter().any(|q| q == id);
        g.queue.retain(|q| q != id);
        if let Some(a) = g.active.as_ref() {
            if a.id == id {
                a.cancel.store(true, Ordering::Relaxed);
                return true;
            }
        }
        if was_queued {
            g.terminal
                .insert(id.to_string(), (DlPhase::Cancelled, String::new()));
            return true;
        }
        false
    }

    /// True while a transfer is running or queued (blocks auto-shutdown).
    pub fn has_active(&self) -> bool {
        let g = self.inner.lock().unwrap();
        g.active.is_some() || !g.queue.is_empty()
    }

    fn pump(&self) {
        loop {
            let next = {
                let mut g = self.inner.lock().unwrap();
                if g.active.is_some() {
                    return;
                }
                let Some(id) = g.queue.pop_front() else {
                    return;
                };
                let entry = g
                    .catalog
                    .iter()
                    .chain(g.custom.iter())
                    .find(|e| e.id == id)
                    .cloned();
                let Some(entry) = entry else { continue };
                let flag = Arc::new(AtomicBool::new(false));
                g.active = Some(ActiveDl {
                    id: id.clone(),
                    cancel: flag.clone(),
                });
                (entry, flag)
            };
            self.run_one(&next.0, &next.1);
        }
    }

    fn run_one(&self, e: &CatalogEntry, cancel: &Arc<AtomicBool>) {
        let dir = self.inner.lock().unwrap().dir.clone();
        if let Err(err) = std::fs::create_dir_all(&dir) {
            self.finish(
                e,
                Err(format!(
                    "cannot create models dir {}: {}",
                    dir.display(),
                    err
                )),
            );
            return;
        }
        let dest = dir.join(&e.filename);
        let part = dir.join(format!("{}.part", e.filename));
        self.emit(
            &e.id,
            "active",
            part_len(&dir, &e.filename),
            e.size_bytes,
            "",
        );
        // Resume partials with -C -; --fail surfaces HTTP errors.
        // NOTE: curl is the temporary transport for https:// — the hand-rolled
        // pure-std client (http.rs) takes plain http:// today and https:// once
        // the TLS transport lands, so users without curl are covered.
        // -s: silent (a progress bar would go nowhere; output is nulled).
        let transfer: Result<(), String> = if e.url.starts_with("http://") {
            // Hand-rolled pure-std path (no subprocess).
            self.run_http(e, cancel, &part)
        } else {
            self.run_curl(e, cancel, &dir, &part)
        };
        let result = match transfer {
            Err(msg) if msg == "cancelled" => Err((true, msg)),
            Err(msg) => Err((false, msg)),
            Ok(()) => {
                // Atomic publish: part -> final.
                let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
                if got == 0 {
                    Err((false, "download finished but received 0 bytes".to_string()))
                } else if e.size_bytes > 0 && got < e.size_bytes {
                    Err((
                        false,
                        format!("incomplete: got {} of {} bytes", got, e.size_bytes),
                    ))
                } else {
                    match std::fs::rename(&part, &dest) {
                        Ok(()) => Ok(()),
                        Err(err) => Err((false, format!("cannot move into place: {}", err))),
                    }
                }
            }
        };
        match result {
            Ok(()) => self.finish(e, Ok(())),
            Err((cancelled, msg)) => {
                if cancelled {
                    self.finish(e, Err("cancelled".to_string()));
                } else {
                    self.finish(e, Err(msg));
                }
            }
        }
    }

    /// Hand-rolled transfer for plain-http URLs: resume, redirect and
    /// progress all come from http.rs, no subprocess involved.
    fn run_http(
        &self,
        e: &CatalogEntry,
        cancel: &Arc<AtomicBool>,
        part: &std::path::Path,
    ) -> Result<(), String> {
        let mut resume = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
        for attempt in 0..2 {
            // write+append: append keeps all writes at the end (resume).
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(part)
                .map_err(|err| format!("open partial: {}", err))?;
            if attempt == 1 {
                // Server ignored Range on attempt 0: drop the stale partial
                // and restart from scratch (delete+recreate dodges
                // truncate-on-append-handle quirks on Windows).
                drop(f);
                std::fs::remove_file(part).map_err(|err| format!("reset partial: {}", err))?;
                resume = 0;
                f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(part)
                    .map_err(|err| format!("open partial: {}", err))?;
            }
            let mut last_emit = std::time::Instant::now();
            let r = crate::http::get(
                &e.url,
                resume,
                Duration::from_secs(15),
                cancel,
                &mut |got, total_opt| {
                    // Prefer the server's total for size-unknown (custom) URLs.
                    let total = total_opt.unwrap_or(e.size_bytes);
                    if last_emit.elapsed() > Duration::from_millis(500) {
                        self.emit(&e.id, "active", got, total, "");
                        last_emit = std::time::Instant::now();
                    }
                },
                &mut f,
            );
            match r {
                Ok(_) => return Ok(()),
                Err(m) if m == "cancelled" => return Err(m),
                Err(m) if m == "range_ignored" && attempt == 0 => continue,
                // 416 while resuming: the partial already holds everything.
                // Publish it iff its length matches the catalog size; the
                // shared finalize below re-checks before renaming.
                Err(m) if m == "range_complete" && e.size_bytes > 0 => {
                    let got = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
                    if got >= e.size_bytes {
                        return Ok(());
                    }
                    return Err(format!(
                        "server says complete but partial is short: {} of {} bytes",
                        got, e.size_bytes
                    ));
                }
                Err(m) => return Err(m),
            }
        }
        Err("server ignored Range twice".to_string())
    }

    /// Subprocess transfer for https:// until the TLS transport lands.
    fn run_curl(
        &self,
        e: &CatalogEntry,
        cancel: &Arc<AtomicBool>,
        dir: &std::path::Path,
        part: &std::path::Path,
    ) -> Result<(), String> {
        let mut child = match crate::hide_console(&mut std::process::Command::new("curl"))
            .args(["-s", "-L", "--fail", "-C", "-", "--retry", "2", "-o"])
            .arg(part)
            .arg(&e.url)
            .stderr(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(err) => return Err(format!("failed to launch curl: {}", err)),
        };
        loop {
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                return Err("cancelled".to_string());
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    if status.success() {
                        return Ok(());
                    } else {
                        return Err(format!("curl exited with status {}", status));
                    }
                }
                Ok(None) => {
                    self.emit(
                        &e.id,
                        "active",
                        part_len(dir, &e.filename),
                        e.size_bytes,
                        "",
                    );
                    std::thread::sleep(Duration::from_millis(500));
                }
                Err(err) => {
                    let _ = child.kill();
                    return Err(format!("curl wait failed: {}", err));
                }
            }
        }
    }

    fn finish(&self, e: &CatalogEntry, result: Result<(), String>) {
        let (phase, msg) = match result {
            Ok(()) => ("done", String::new()),
            Err(m) if m == "cancelled" => ("cancelled", String::new()),
            Err(m) => ("error", m),
        };
        {
            let mut g = self.inner.lock().unwrap();
            g.active = None;
            let dl_phase = match phase {
                "done" => DlPhase::Done,
                "cancelled" => DlPhase::Cancelled,
                _ => DlPhase::Error,
            };
            g.terminal.insert(e.id.clone(), (dl_phase, msg.clone()));
        }
        // Report real on-disk bytes (a failed transfer must not show a full bar).
        let dir = self.inner.lock().unwrap().dir.clone();
        let on_disk = part_len(&dir, &e.filename);
        let total = if e.size_bytes > 0 {
            e.size_bytes
        } else {
            on_disk
        };
        self.emit(&e.id, phase, on_disk, total, &msg);
        // Hand the worker the next queued id, if any.
        let _ = self.kick.send(());
    }
}

fn part_len(dir: &std::path::Path, filename: &str) -> u64 {
    let part = dir.join(format!("{}.part", filename));
    if let Ok(m) = std::fs::metadata(&part) {
        return m.len();
    }
    std::fs::metadata(dir.join(filename))
        .map(|m| m.len())
        .unwrap_or(0)
}

fn curl_available() -> bool {
    static CURL_OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CURL_OK.get_or_init(|| {
        crate::hide_console(&mut std::process::Command::new("curl"))
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Optional catalog override: `HEARTH_MODEL_CATALOG_JSON` env or
/// `<models_dir>/catalog.json`, an array of
/// `{id?, filename, url, size_bytes?, label?, blurb?}`.
fn load_catalog_override(dir: &std::path::Path) -> Option<Vec<CatalogEntry>> {
    let text = std::env::var("HEARTH_MODEL_CATALOG_JSON")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::fs::read_to_string(dir.join("catalog.json")).ok())?;
    let v = hearth_core::json::parse(text.trim()).ok()?;
    let arr = v.as_arr()?;
    let mut out = Vec::new();
    for item in arr {
        let filename = item.get("filename")?.as_str()?.to_string();
        let url = item.get("url")?.as_str()?.to_string();
        if sanitize_filename(&filename).is_none() {
            continue;
        }
        let id = item
            .get("id")
            .and_then(|x| x.as_str())
            .unwrap_or(&filename)
            .to_string();
        let size_bytes = item
            .get("size_bytes")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0) as u64;
        let label = item
            .get("label")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| filename.replace(".gguf", "").replace(['-', '_'], " "));
        let blurb = item
            .get("blurb")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        out.push(CatalogEntry {
            id,
            filename,
            url,
            size_bytes,
            label,
            blurb,
        });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}
