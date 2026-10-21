# Hearth — next session

> **Session 2026-10-20 in one paragraph.** The operator installed the published `v0.2.0`
> MSI and the model list showed **2 of the 9 models on the machine**. Cause, not symptom:
> both halves of the app resolved a *single* `models/` directory (the downloader planted
> weights in `%LOCALAPPDATA%\Programs\Hearth\models`, and the service's
> `default_models_dir()` returned that same folder by walking up from its exe), so
> `%APPDATA%\hearth\models` and the source checkout were never scanned. `hearth-core` now
> owns the answer — `paths::models_roots()` (CWD, every exe ancestor, `%APPDATA%`,
> `%LOCALAPPDATA%`, XDG/macOS per-user dirs; absolute, de-duplicated) and
> `paths::primary_models_dir()` for new downloads — and `HEARTH_MODELS_DIR` still wins
> outright but now accepts a `;`-separated list. `list_models` aggregates every root,
> de-duplicates by filename and returns `dirs` beside `dir`; the downloader treats a copy
> in **any** root as already on disk, so nothing is fetched twice; the empty state names
> every folder searched. Verified on the real install: 9 models over 2 roots, 14 with the
> repo added by the env list, `/` + `/api/downloads` 200, a `%APPDATA%` model loaded, and
> the UI screenshotted at both counts. Clippy/fmt/check clean on x86_64 and aarch64;
> `cargo test -p hearth-core` 7 passed. **The published MSI still ships the old
> binary** — the install on this machine was updated by hand.

Earlier sessions: `docs/BUG_TRACKER.md`, newest entries at the top. Packaging:
`packaging/windows/README.md`, `packaging/SIGNING.md`.

## Do these first

1. **Cut v0.2.1 — but the operator has to say go.** The model-root fix is only in the
   working tree and in this machine's `%LOCALAPPDATA%\Programs\Hearth`; anyone who
   downloads `v0.2.0` still sees one folder. Bump the nine crates + `resources/Info.plist`,
   add a `CHANGELOG.md` line, `git tag -a v0.2.1`, push, `gh run watch`. A `workflow_dispatch`
   dry run is only needed when the *packaging* changed — this session touched no `.wxs`.
2. **Add a folder from the UI.** The scan now finds the standard places, but weights in an
   arbitrary directory still need `HEARTH_MODELS_DIR`, and that is a restart plus an env
   var on Windows. Cheapest honest shape: a "Add folder…" row in the downloads drawer
   writing a `roots.json` (`<primary models dir>/roots.json`) that `models_roots()` appends
   after the standard roots — file input on the web UI is a path field, since a browser
   file picker cannot hand a directory to the server.
3. **Two files with the same name in two roots are one dropdown entry today** (first root
   wins, silently). Showing the folder in the option label — or in a `title=` tooltip —
   would make the choice explicit once multiple roots are normal.
4. **Re-baseline the performance table on a quiet box.** Both the six-model rows in
   `docs/BUG_TRACKER.md` and the README table come from busy windows. Each model alone,
   warm-up, mean of two runs, with `% Processor Performance` and top CPU consumers recorded
   beside the number. A run whose CPU/wall ratio is not ~800% for 8 workers is a loaded
   measurement. This session changed no kernel code, so nothing is pending a re-measure.
5. **`docs/RESUME_img.md` / `docs/BUG_img.md`** are the image-lane equivalents; keep them
   in sync if the downloader or the drawer changes.

## Goal 2 (UI / docs / downloader) — remaining

1. **README screenshots are stale** (both predate the 2026-10-16 UI commits). Re-shoot `/`
   and the downloads drawer at ~1600x1000 over `docs/screenshots/hearth-{chat,downloads}.png`
   now that `preview_screenshot` works from the agent side.
2. **Icon pair check whenever artwork changes**: `padTop == padBottom` and
   `padLeft == padRight` from `getBoundingClientRect` on the served SVG. Icons come from
   `python scripts/make-icons.py` (needs `rsvg-convert`) and are committed.
3. **The downloader's DNS fallback is unit-tested only** — check it against a real
   dual-stack host and against a host whose first address is unreachable.

## Goal 1 (inference) — open items, most valuable first

1. **aarch64 has no SIMD backend**; every kernel is the scalar fallback and only
   `arm-qemu-test` has ever run it. NEON first suits the `Q2V_U8` LUT dot (`vqtbl` is a
   table lookup) and the f32 glue in `ops.rs`. A real ARM runner is the prerequisite for
   any number (`macos-latest` is arm64: tests are free, timing is not).
2. **Worker counts follow the *path* and every kernel change owes a count sweep on both
   passes.** Dense = `physical` (8 here); hybrid `Q2_0`/`PQ2_0` = `min(logical, physical +
   physical/4)` (10); hybrid `PTQ1_0` = ¾ of logical (12). `HEARTH_THREADS` overrides.
   Judge a count on its **spread**, not its mean — the dense case at 9 is bimodal — with
   `scripts/count-sweep.py` (quiet and loaded, rotated order). Untested hypothesis:
   `pool.rs` splits statically, so an atomic work-stealing split should tighten the
   batched path; if it does not, that falsifies the mechanism.
3. **Per-CPU calibration runner** is half-built: the record must be a (kernel, count, path)
   triple that refuses to switch inside the noise band. `scripts/baseline-invariance.py`
   (kernel axis) and `scripts/count-sweep.py` (count axis) exist; persisting the winner and
   folding the sweep into `benchk` remain. A baseline is part of the measurement and so is
   the window.
4. **Speculative decoding: measure acceptance `a` before designing anything.** 2B PTQ1_0
   decodes at 50.8 ms/token against the 27B's ~722, so the draft is 14x cheaper, but the
   gain rests entirely on an unmeasured `a`, and the two are not in the same weight basis
   (`prism.hadamard.version` 2/512 vs 1/1024). 3 of every 4 `qwen35` layers are recurrent,
   so a rejected draft must rewind conv+SSM state, not just a KV cache.
5. **Chunked prefill: measure `f` first** — same prompt on `PQ2_0` (2-bit) versus `PTQ1_0`
   (1-bit). `dot_ptq1_0_q8_0x2_ptr` is the tested primitive (bit-identical to two single
   calls, currently unreferenced — do not re-wire it into `matmul_batch`); an x4 variant
   regressed 3.2x in pt 5.
6. **Kernel rules that made past fixes findable**: a `#[target_feature]` body names every
   feature it uses and its dispatch gate names the same set; `.cargo/config.toml` pins
   `x86-64-v3` and must not drop below what any dispatched body uses. Zen 5 and Intel must
   not regress — the AVX-512 gate keys on 19h and there is no such hardware here. A new
   quant format needs `kernel_name()` and a `selected_kernel` arm once it has two
   implementations.
7. **GPU scaffolding: still not started** (opt-in via `--device`/env, graceful without a
   ROCm device, must not break the CPU path or Windows/macOS builds).
8. **P-256 verification is 345.6 µs** with known levers (base-point table, windowing,
   Montgomery product, Fermat inverse). Do not start before measuring — it is paid once per
   connection and each change makes `p256.rs` less obviously correct.

## hearth-tls lane

1. **An anchor source in `hearth-ui`** — a pinned set with fingerprints and a documented
   update path, reviewable on one screen. **No root store in `hearth-tls`**, and do not
   widen `CertificateVerifier` (a `compile_fail` doctest enforces that): no accept-all type,
   flag or env var on any path a URL can reach.
2. **The downloader's `Transport` driving `ClientHandshake`** — `http.rs`, the seam is its
   `https://` branch plus `downloads.rs::run_transfer`. First time any of it meets a socket.
3. **CRL refresh failure** must not mean "the stale list keeps answering". No OCSP, no
   stapling, no fetching — by design.
4. `aarch64` PMULL/AES for macOS slice-3 numbers (AES-128 68.8 MB/s, GCM 26 MB/s);
   `HEARTH_TLS_NO_HW=1` and the software reference already exist.

## Standings and hazards

- **The model list is a multi-root scan now.** `hearth_core::paths` is the only place that
  answers "where are the models"; do not re-add a walk-up in `hearth-ui` or the service, and
  if a new root is needed add it there so the downloader's `done` state keeps agreeing with
  the dropdown.
- **`p256.rs` is verification-only and variable-time by design** (it branches on scalar
  bits); unfit for signing or ECDH. TLS 1.3 here uses X25519 and the crate never signs.
- **Do not re-derive the prefill/decode asymmetry** — the 27B's prefill is *faster* than
  its decode (635 vs 722 ms/token); the old "1.8-2.3 s per prompt token" was a 75%-loaded
  window with a cold cache.
- **Do not "fix" the first token** — it is storage I/O on the first read of the weights; a
  prefault changed nothing and cost 259 ms-2.4 s of load. `HEARTH_PIN_WEIGHTS=1` died
  silently on the 27B on this 16 GB box — never a default.
- **Verify through the interface, not the summary**: drive the real API
  (`HEARTH_MODELS_DIR`, `HEARTH_MODEL_CATALOG_JSON`, `--no-browser`,
  `HEARTH_NO_AUTO_SHUTDOWN=1`). `--temp 0 --prompt-raw` for benchmarks.
- **`hearth-tls` is not a general-purpose TLS stack** — scope is "the downloader can reach
  `https://` with no dependencies"; unneeded features are *refused*, not ignored.
- **MSI facts worth not re-learning**: an OS-version launch condition cannot pass on a
  machine that emulates `VersionNT`; a `File` installs into its component's directory;
  `RemoveFolder` removes empty directories only (which is what preserves `models/`); the
  ARP row registers under `HKLM` while the payload stays in the profile (`ALLUSERS=2`); the
  `RemoveFolder` table cannot be read through the Installer COM API — use
  `wix msi decompile`; `wix build` needs `-acceptEula wix7`, and custom-action FileRefs
  need `ExeCommand=""`.
- **Runner labels fail silently.** `macos-13` was retired and queues forever; the
  replacements are `macos-15-intel` and `macos-15`. `ubuntu-latest` moved to Ubuntu 26 on
  2026-10-19, so the Linux leg is pinned to `ubuntu-24.04` to hold the glibc floor.

## Verification habits for a shared checkout

- `cargo fmt --all` reformats the other agent's files — use `cargo fmt -p <crate>`, same for
  clippy.
- `target/release/hearth-chat-cli.exe` is often locked by a running UI: `netstat -ano` for
  the port, kill that PID, and remember the bash wrapper's PID is not the exe's. Build to a
  spare `CARGO_TARGET_DIR` rather than killing a process you do not own.
- **A measurement needs its window recorded.** Absolute levels drift ~10% over an hour and
  the second run in a pair is worth ~2-3%; use `b,a,a,b` groups and compare means.
- **A constant copied from memory is not a vector** — derive it and have an independent
  implementation agree before embedding it.
- **A test whose two sides run the same code proves nothing**, and **a control that passes
  is a control that was never tested** — assert the domain directly.
- **A claim in a doc comment is a claim**: run it once and look before writing that a check
  passes or a property holds.
- **Suspect your own harness before calling someone else's data broken.** Every finding of
  this kind so far was my own bug (CRLF from a helper, a port reused, an off-by-four
  offset, a PowerShell variable clash, a discarded return value).
