# Hearth — next session

> **Session 2026-10-17 (pt 21) in one paragraph.** The release path exists: versions
> bumped 0.1.0 → 0.2.0 across the nine crates and `resources/Info.plist`, `CHANGELOG.md`
> written, and a Windows **MSI** (`packaging/windows/hearth.wxs`, WiX 7) built and
> verified end to end on this machine, with the portable zip beside it. Four defects came
> out of the *install*, not the build: setup died with 1603 inside `LaunchConditions`
> because a launch condition on `VersionNT` can never be satisfied here (Windows
> Installer reports 603/9600 for this packaging shape — the editor's shipped installer
> logs the same); `models/README.txt` installed beside the exes (a `File` lands in its
> *component's* directory); an empty `models/` survived uninstall and kept
> `Programs\Hearth` alive (`RemoveFolder` on the model folder, empty-only); and the
> artifact glob would have shipped WiX's `.wixpdb`. `hearth-ui` is now a GUI-subsystem
> binary whose children get `CREATE_NO_WINDOW`, so nothing flashes a console.
> `.github/workflows/release.yml` builds on five runners, and its `workflow_dispatch`
> dry run passes **five of five green**, uploading the five artifacts and stopping
> before publish — nothing is tagged or published yet (operator call). That run also
> caught a runner that no longer existed: `macos-13` retired in December 2025 and a
> job on a retired label queues forever, so the Intel leg was `macos-15-intel` (the
> last Intel image) and arm64 moved to `macos-15`; `ubuntu-latest` migrates to
> Ubuntu 26 on 2026-10-19, so that leg is pinned to `ubuntu-24.04` to keep the
> artifacts' glibc floor where it is. Finally, the six-model sweep on the
> **final** binary read 49.26 / 25.39 / 22.71 / 11.41 / 12.85 / 6.07 tok/s against a
> 46.7 / 23.6 / 23.6 / 12.2 / 13.0 / 6.2 baseline — the small models *above*, the 4B/8B
> rows below, with re-runs spreading ±10-20% and an instrumented run measuring the CLI
> at **666% of one core where 8 busy workers need ~800%**. Nothing in the measured path
> changed (only `hearth-ui` sources and version strings), so that is the box, not a
> regression — and the table needs re-baselining on a quiet machine before it is used to
> judge anything.

Earlier sessions: the history is in `docs/BUG_TRACKER.md` (newest entries at the top).
Packaging specifics: `packaging/windows/README.md`, `packaging/SIGNING.md`.

## Do these first

1. **Close the release (needs the operator).** Ask before `git tag v0.2.0` and before
   any publish — both are public and awkward to undo. The dry run is done: run
   `39015532967` is **five of five green**, `publish` skipped, and all five artifacts
   were downloaded and inspected (Windows `.zip` + `.msi` with no `.wixpdb`, both
   `Hearth.app` architectures with `Info.plist` and `hearth.icns`, both Linux tarballs
   with the `.desktop`, the icon and `models/README.txt`). What has *never* run is the
   `publish` job itself — a tag push is the only way to exercise it, and it is the last
   untested step in the release. Re-run
   `gh workflow run release.yml -f version=v0.0.0-dry-run` after any packaging change;
   it is 5 minutes and it is what caught the retired `macos-13` label.
2. **Re-baseline the performance table on a quiet box** (see the sweep note above).
   Both the six-model rows in `docs/BUG_TRACKER.md` and the README's model table are
   from busier windows than they claim. Run each model alone, warm up, take the mean of
   two runs, and record the machine state (`% Processor Performance`, top CPU
   consumers) beside every number from now on. **A run whose CPU/wall ratio is not
   ~800% for 8 workers is a loaded measurement, not a fast or slow one.**
3. **The README has no install section, and it should.** `README.md` is SparkOC's
   claimed file (`task2-ui-docs-downloader`) — ask in the room before editing it. What
   it needs: download the installer or the zip from the releases page, per-OS lines
   (MSI per-user, `Hearth.app` Gatekeeper note, the Linux tarball's `.desktop`), and a
   pointer to `docs/INSTALL_UX.md`.
4. **`docs/RESUME_img.md` and `docs/BUG_img.md`** are the image-lane equivalents of
   this file and the tracker; keep them in sync if the packaging work touched anything
   they describe.

## Goal 1 (inference) — open items, most valuable first

1. **aarch64 has no SIMD backend at all.** Every dispatched kernel is the scalar
   fallback and `arm-qemu-test` is the only thing that has ever executed it. NEON is
   the largest item in `hearth-quant`, but a real ARM target is the *prerequisite* for
   any number (`macos-latest` is arm64: `cargo test` there is free, timing is not).
   The two shapes that suit NEON best: the `Q2V_U8` LUT dot (`vqtbl` is exactly a table
   lookup) and the f32 glue in `hearth-llm/src/ops.rs`.
2. **The worker count follows the *path* — and any kernel change owes a count sweep on
   both passes.** As shipped: dense = `physical` (8 here); hybrid `Q2_0`/`PQ2_0` =
   `min(logical, physical + physical/4)` (10); hybrid `PTQ1_0` = ¾-of-logical (12).
   `HEARTH_THREADS` overrides everything. The dense rule is *robustness*: the batched
   prefill at count 9 is bimodal (mode ~7% faster, tail 1.7x slower) and a laptop
   default is judged on the tail. **Judge a count on its spread, not its mean**, and
   use `scripts/count-sweep.py` (quiet *and* loaded, rotated order).
   - **Untested hypothesis, worth one session:** `pool.rs`'s split is static
     (`chunk = n.div_ceil(nt)`), so 9 workers over 109 rows give the ninth 5 rows
     against 13 — an atomic work-stealing split for the batched path should remove the
     imbalance and bound the sharing penalty to the final row. If the spread does *not*
     tighten, that falsifies the mechanism. Mirrored, both passes, counts 8/9/10.
3. **The per-CPU calibration runner is still half-built, and both axes now have tools.**
   The record has to be a (kernel, count, path) triple that **refuses to switch on a
   difference inside the noise band**. `scripts/baseline-invariance.py` (kernel axis) and
   `scripts/count-sweep.py` (count axis) exist; what remains is persisting the winner
   and folding the count sweep into `benchk`'s report. Hazards it exists to prevent,
   both already misread once: **a baseline is part of the measurement** (a v3 run ranked
   `avx2_lut` above `vnni512x2` and was "corrected" by a `native` rerun that was
   measuring different codegen), and **a window is part of the measurement** (pt 19's
   park table compared two windows).
4. **Speculative decoding: measure acceptance before designing anything.** 2B PTQ1_0
   decodes at 50.8 ms/token against the 27B's ~722, so the draft is 14x cheaper; the
   gain is `(1-a^(k+1))/(1-a)` per `1+0.07k` target-equivalents. **`a` is unmeasured and
   the whole plan rests on it.** Tokenizers are byte-identical (verified); the chat
   template differs and `prism.hadamard.version` is 2/512 on the 2B vs 1/1024 on the
   27B, so the two are *not* in the same weight basis — argue from measurement. 3 of
   every 4 `qwen35` layers are recurrent, so a rejected draft must rewind conv+SSM
   state, not just a KV cache.
5. **Chunked prefill: measure `f` first.** The hybrid prefill has zero weight reuse and
   re-streams every weight row per prompt token, so its per-token cost is ~= decode's.
   The ceiling is roughly `1 - f` for the unpack share; the cheap probe is the same
   prompt on `PQ2_0` (2-bit) versus `PTQ1_0` (1-bit). `dot_ptq1_0_q8_0x2_ptr` is the
   tested primitive (bit-identical to two single calls, **currently unreferenced** — do
   not re-wire it into `matmul_batch`, that path cannot reach PTQ1_0). An x4 variant
   regressed 3.2x in pt 5.
6. **Keep the kernel rules that made today's fixes findable.** A `#[target_feature]`
   body must name every feature it uses and its dispatch gate must name the same set
   (`q1_0g128::avx2_kernel_available`, `q2_0::avx2_lut_available`,
   `ptq1_0::avx2_available`, plus the feature-report test); `.cargo/config.toml` pins
   `x86-64-v3` for x86_64 and must not go below what any dispatched body uses. Zen 5
   (1Ah) and Intel must not regress — the AVX-512 gate keys on 19h and there is no such
   hardware here. A new quant format needs a `kernel_name()` and a `selected_kernel`
   arm (only when it has more than one implementation).
7. **GPU scaffolding: still not started.** Opt-in behind `--device`/env, graceful when
   no ROCm device exists, must not break the CPU path or the Windows/macOS builds.
8. **P-256 verification is 345.6 µs and the levers are known** (no base-point table, no
   windowing, coarse Montgomery product, Fermat inverse). **Do not start before
   measuring**: it is paid once per connection against a download measured in seconds,
   and every one of those changes makes `p256.rs` less obviously correct.

## Goal 2 (UI / docs / downloader)

1. **Screenshots are still blocked.** Both README screenshots predate the 2026-10-16 UI
   commits; `browser.screenshot` refuses without a *visible* desktop window. Bring the
   window forward, then re-shoot `/` (chat) and the downloads drawer at ~1600x1000 over
   `docs/screenshots/hearth-{chat,downloads}.png`.
2. **Verify the icon pair whenever the artwork changes**: `padTop == padBottom` and
   `padLeft == padRight` measured with `getBoundingClientRect` on the served SVG, not by
   eye. `resources/hearth.{ico,icns}` and `hearth-setup.ico` are generated by
   `python scripts/make-icons.py` (needs `rsvg-convert`) and are committed.
3. **The downloader's DNS fallback** is unit-tested only: check it against a real
   dual-stack host and against a host whose first address is unreachable.
4. **Model downloader is otherwise done and verified live** (`/api/downloads` catalog
   with per-model state, `/start`, `/cancel`, one file at a time, resume, the drawer).

## hearth-tls lane

The crate can now *accept*: `PinnedVerifier::new(anchors, revocation, policy)` drives
`chain.rs` / `verify.rs` / `leaf.rs` / `crl.rs`, and every rule has a negative fixture
and an OpenSSL-oracle case. What is left, in order of proof:

1. **An anchor source in `hearth-ui`** — a pinned set with fingerprints and a documented
   update path, reviewable in one screen, before any OS trust reader. **Do not put a root
   store in `hearth-tls`**, and do not widen `CertificateVerifier` (it is deliberately
   not implemented for closures, enforced by a `compile_fail` doctest): no accept-all
   type, flag or environment variable on any path a URL can reach.
2. **The downloader's `Transport` driving `ClientHandshake`** — `http.rs` is SparkOC's
   file, ask before touching it; the seam is its `https://` branch and
   `downloads.rs::run_transfer`. This is the first time any of this meets a socket.
3. **The CRL refresh question**: what happens when a refresh fails must not be "the
   stale list keeps answering". No OCSP, no stapling, no fetching — by design.
4. `aarch64` PMULL/AES so macOS gets slice-3 numbers instead of the software path
   (AES-128 68.8 MB/s, GCM 26 MB/s); `HEARTH_TLS_NO_HW=1` and the software reference
   already exist to test against.

## Standings and hazards

- **`p256.rs` is verification-only and variable-time by design** (it branches on scalar
  bits). Unfit for signing or ECDH as written; TLS 1.3 here uses X25519 and the crate
  never signs. If that changes, do not reuse `mul` as it stands.
- **Do not re-derive the prefill/decode asymmetry.** It was answered and the premise was
  wrong: the 27B's prefill is *faster* than its decode (635 vs 722 ms/token). The old
  "1.8-2.3 s per prompt token" was a 75%-loaded window with a cold cache.
- **Do not "fix" the first token.** It is storage I/O on the first read of the weights;
  a prefault changed nothing (173.0 vs 173.4 ms) and cost 259 ms-2.4 s of load.
  `HEARTH_PIN_WEIGHTS=1` **died silently on the 27B** on this 16 GB box — never a default.
- **Verify through the interface, not the summary.** Findings come from driving the real
  API (`HEARTH_MODELS_DIR`, `HEARTH_MODEL_CATALOG_JSON`, `--no-browser`,
  `HEARTH_NO_AUTO_SHUTDOWN=1`) against a local server. `--temp 0 --prompt-raw` for
  benchmarks; the CLI prints per-token stage timings by default, so filter the output.
- **`hearth-tls` is not a general-purpose TLS stack.** Its scope is "the downloader can
  reach `https://` on Linux, macOS and Windows with no dependencies"; unneeded features
  are *refused*, not ignored, and that is what keeps it reviewable.
- **MSI facts worth not re-learning the hard way** (all in `packaging/windows/README.md`):
  an OS-version launch condition cannot pass on a machine that emulates `VersionNT`;
  a `File` installs into its component's directory; `RemoveFolder` removes empty
  directories only, which is what preserves `models/`; the ARP row registers under
  `HKLM` while the payload stays in the profile (`ALLUSERS=2`); the `RemoveFolder` table
  cannot be read through the Installer COM API at all — use `wix msi decompile`;
  `wix build` needs `-acceptEula wix7`, and MSI custom-action FileRefs need
  `ExeCommand=""` (WIX0038/WIX0044).
- **Runner labels are load-bearing and fail silently.** `macos-13` was retired in
  December 2025 and a job on it queues forever rather than erroring; the replacements
  are `macos-15-intel` (Intel, until Aug 2027) and `macos-15`/`macos-14` (arm64), and
  `macos-14` is itself being retired. `ubuntu-latest` moves to Ubuntu 26 on
  2026-10-19, so it is pinned to `ubuntu-24.04` to hold the glibc floor of the
  published binaries. `actions/upload-artifact@v5` still declares Node 20, so that
  deprecation annotation is GitHub's, not ours.

## Verification habits for a shared checkout

- `cargo fmt --all` reformats the other agent's files too — use `cargo fmt -p <crate>`,
  same for clippy.
- `target/release/hearth-chat-cli.exe` is often locked by a running UI: build to a spare
  `CARGO_TARGET_DIR` rather than killing a process you do not own.
- **A measurement needs its window recorded.** Absolute levels drift ~10% over an hour
  and the second run in a pair is worth ~2-3%, so use `b,a,a,b` / `a,b,b,a` groups,
  compare group means, and never compare across windows. The CLI's 50-token run has a
  ±5% spread on its own, and a *loaded* row is a different measurement, not a faster one.
- **A constant copied from memory is not a vector.** Derivations, not transcriptions:
  have an independent implementation agree before embedding anything.
- **A test whose two sides run the same code proves nothing, however many assertions it
  has** (`assert_eq!(g.double().double(), ladder(4))` is `assert_eq!(x, x)`).
- **A control that passes is a control that was never tested**, and **when a bug cancels
  another bug the test can pass harder** — assert the *domain* directly rather than
  checking that an inverse multiplies back.
- **A claim in a doc comment is a claim.** Before writing that a file reproduces, a check
  passes or a property holds, run it once and look.
- **Suspect your own harness before calling someone else's data broken.** Every "finding"
  of this kind so far was my own bug (CRLF from a helper, a port reused by a Python
  server, an off-by-four offset from a leftover handshake header, a PowerShell variable
  clashing with `$Zip`/`$PID`, a discarded `Execute` return value leaking a null row).
  Print the raw value and compare it against the claim.
