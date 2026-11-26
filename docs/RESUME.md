# Hearth — next session

> **Session 2026-11-26, second half and release, in one paragraph.** Two things landed.
> First, the rotate+quantize path — the Hadamard fold plus the Q8 activation quantize, this
> file's own first "do this" item — is AVX2, and the bit-identity is asserted by nine unit
> tests that compare the vector path against the scalar reference **byte for byte** (whole and
> partial blocks, all-zero blocks, exact ±0.5 ties, the max/127 boundary, a scale that
> underflows, a denormal scale, a NaN, an infinity, signed zeros, and the transform itself on
> `to_bits()`). On the 27B `PQ2_0` the two rows that measure it are **4.5x faster**:
> `ffn q in` 104.7 → 23.3 ms and `ffn q mid` 351.1 → 77.6 ms over 30 decode tokens, i.e.
> **11.8 ms/token** off a ~480 ms token. Every generated text was byte-identical between the
> arms on all seven models measured; the dense models gain 0.7–6.7% end to end and the 27B
> 1.9%. Second, the release: the workspace is at **0.3.0** (ten manifests, `Cargo.lock`, both
> `Info.plist` keys, a `CHANGELOG` section), the Windows installer was verified end to end on
> a real machine (upgrade from 0.2.0 with two downloaded models, same-version reinstall,
> uninstall that keeps the models, the installed binary answering all five endpoints), the
> five-runner release path was exercised with a dry run, and `v0.3.0` is published
> (six assets plus `SHA256SUMS`; 6 of 6 hashes match and the *shipped* MSI passes the 14-check
> table gate). The operator then asked for the chat attachments to be in that release too, so
> `v0.3.0` is re-tagged at the tree that carries them and republished under the same version.
> The release build is now warning-free under the toolchain CI uses. The one scar
> this session left is the most useful lesson in it: **the `kernel-baseline` CI job was red
> for half a day and nobody saw it**, because the failing job was not the one anybody was
> watching — see "Do not re-open".
> Detail: `docs/BUG_TRACKER.md`, newest entry first.
>
> Earlier the same day: `docs/BUG_TRACKER.md`. Packaging: `packaging/windows/README.md`,
> `packaging/SIGNING.md`.

## Baseline to compare against (2026-11-26, 50 tokens, temp 0, prompt-raw "Hello")

Paired A/B medians from **one** binary — `HEARTH_QUANT_SCALAR=1` picks the arm, alternating
`b,a,a,b`, 4 pairs (8 runs per arm), text compared between arms:

| model | before (scalar) | after (AVX2) | delta |
| --- | --- | --- | --- |
| Bonsai-1.7B-Q1_0 | 42.89 tok/s | 43.62 tok/s | +1.7% |
| Ternary-Bonsai-1.7B-Q2_0 | 22.95 tok/s | 23.36 tok/s | +1.8% |
| Bonsai-4B-Q1_0 | 18.85 tok/s | 20.11 tok/s | +6.7% |
| Ternary-Bonsai-4B-Q2_0 | 10.63 tok/s | 10.97 tok/s | +3.2% |
| Bonsai-8B-Q1_0 | 11.41 tok/s | 11.49 tok/s | +0.7% |
| Ternary-Bonsai-8B-Q2_0 | 5.94 tok/s | 6.03 tok/s | +1.5% |
| Ternary-Bonsai-2.5-27B-PQ2_0 | 1.87 tok/s | 1.91 tok/s | +1.9% |

**The README's dense table is stale on this box.** The *before* arm — the pre-change code path,
byte for byte — reads 42.89 tok/s on the 1.7B Q1_0 where the table says 48.1, and 18.85
against 24.4 on the 4B Q1_0. That is 6–18% of machine state (other agents, thermals), not of
this change: the paired A/B is immune to it, because both arms are the same binary minutes
apart. **Re-baseline all 15 packs before quoting an absolute number.** The previous baseline
paragraph (`48.1 / 27.1 / 24.4 / 11.7 / 12.4 / 6.2` for the dense six, `25.3 / 25.1 / 12.0 /
11.2 / 6.7 / 6.3` for the hybrid 2B/4B/9B, `2.17 / 1.98` for the 27B pair, `2.16 / 1.94 / 2.15`
for the 2.5 trio) is still the reference for *shape*, not for absolute tok/s.

## Do these first

1. **Re-baseline the README table (all 15 packs, one at a time).** The table is 6–18% above
   today's box. Sweep it with the same protocol as the A/B above (paired arms, 4 pairs,
   medians) and put the numbers in the README and in `docs/BENCHMARKS.md`.
2. **The GDN scan is the biggest remaining prefill lever.** It is still per-token
   (`qwen35.rs` `gdn_scan_heads`), and the 2026-02 Huawei GDN operator analysis says the state
   traffic dominates: a **chunked scan** (typical C = 64/128, C ~ (dk·dv)^(1/3)) is the fix.
   `PREFILL_CHUNK = 4` chunks the *weight* passes only, not the recurrence. This changes
   numerics, so it needs an output-quality check, not a bit-identity test.
3. **The embedding cache (FableH's LEAD 2).** `forward_pair` de-quants and inverse-WHTs each
   embedding row per token, and the result is position-independent. A bounded cache (256 rows
   ≈ 5 MB) is a self-contained prefill win; upstream llama.cpp PR 27032 does the same. **Probe
   the `embed` timer first** — it reads 32–818 us per token on the 27B, so if it stays under
   ~5 ms/token, skip it. The inverse Hadamard it pays is now the AVX2 kernel, which makes the
   miss cheaper than it was.
4. **The hybrid worker counts need a sweep at width 4.** The Q2_0/PQ2_0 "physical + physical/4"
   rule was derived from a width-2 measurement (13346 ms at 10 workers vs 13940 at 8, over 109
   tokens). Judge a count on its **spread** (`scripts/count-sweep.py`), not its mean, and either
   re-measure at width 4 or label those numbers historical.
5. **Tool calling is the next template gap, and it is unchecked.** The 2.5 template's `tools`
   branch parses, but no fixture exercises it: `TemplateContext` has no `tools` field. Add the
   field and a fixture case rendered by the reference engine before wiring tools through the UI.
6. **Refusal-or-load for foreign `qwen35` packs.** `ModelConfig` accepts any `qwen35` file; the
   loader then fails on the first missing tensor. A ten-line pre-flight check (arch → required
   tensor names → Hadamard metadata) would name the reason instead of
   `qwen35: missing blk.0.attn_qkv.weight`.
7. **Decide on MTP.** The `-mtp` files add a draft head Hearth cannot use yet. Keep them out of
   the catalog (current state) or wire the head when speculative decoding lands.
8. **The 27B tile finding, so nobody re-chases it.** `PREFILL_CHUNK = 4` is **neutral** on the
   27B PTQ1_0 because a 1 MB weight tile does not stay resident at d = 5120 — the second pair
   re-reads from DRAM anyway. Chunk 4 pays off where the tile *is* resident.
9. **The release page after `v0.3.0`**: the tag's assets are on the releases page with
   `SHA256SUMS` beside them. If the human wants signed Windows/macOS builds, the missing piece
   is the signing secrets in `packaging/SIGNING.md` — without them the workflow publishes
   unsigned and the OS says so.

## Do not re-open (each of these is measured, with the number)

- **The `kernel-baseline` guard is not advisory — it catches a real 5.7x.** Every push for half
  a day was red in that one job while clippy, fmt, the tests, the cross-target checks and the
  arm emulators were all green: the AVX2 quantize kernel declared `avx2` but converted each
  block scale with `half::from_f32`, and `avx2` does not imply `f16c`, so the conversion was an
  out-of-line call at the weak `x86-64-v2` baseline the guard builds at. **At `x86-64-v3` the
  omission is invisible** — the release build looked perfect locally. The fix is in
  `dd788bc`: a `f32_to_f16_bits` helper (`#[inline]`, `target_feature f16c`, one `vcvtps2ph`,
  round-to-nearest-even so it matches `half` bit for bit), the kernel declaring `f16c` as a
  **second attribute** (`enable = "avx2, f16c"` parses as one feature named `" f16c"` and is
  rejected), and the dispatch requiring `is_x86_feature_detected!("f16c")` because AMD
  Excavator has AVX2 without F16C. **Run the guard locally before pushing a kernel:**
  `RUSTFLAGS="-C target-cpu=x86-64-v2" cargo rustc -p hearth-quant --release --lib
  --target-dir target/baseline-asm -- --emit=asm=target/baseline-lib.s` and the same for
  `--tests`, then `python scripts/check-kernel-baseline.py` on each listing. Both must print
  `baseline-invariant`.
- **The rotate+quantize path.** It is AVX2 now and byte-identical to the scalar reference
  (nine tests). `ffn q in` 104.7 → 23.3 ms and `ffn q mid` 351.1 → 77.6 ms over 30 decode
  tokens on the 27B PQ2_0 = 11.8 ms/token. The remaining time there is the transform itself
  (~10 butterfly stages over 17k values per row), not a missing optimization.
- **The rounding rules in the Q8 quantize.** They are not free choices: the value is rounded
  **half away from zero** (what `f32::round` does, *not* `roundps`'s half-to-even), the clamp
  happens **before** the rounding, a NaN block falls back to the scalar path, and an all-zero
  block stores the fp16 scale **1.0**. The *scale* is written with the hardware `vcvtps2ph`
  (nearest-even, which is what `half` does) — that is what keeps it byte-identical. Change any
  of this and the activation bytes change.
- **The 27B's ffn decomposition.** Sub-timers sum to the row exactly. +24 ms is the two weight
  passes at 14.6-14.8 GB/s (a kernel-shaped floor ~15% above the byte floor, because the read
  benchmark does not run 17.1 Gweights of MACs), +15.1 ms was the rotate+quantize (now 4.5x
  faster), +1.1 ms silu, +0.8 ms norms and residuals. Nothing else is hiding there.
- **Dispatch count as an explanation.** The gate/up fusion is real and ~12 ms; a longer spin
  before parking is worse; forcing a park costs ~3 ms. The 0.45 ms/dispatch reading was wrong
  by an order of magnitude.
- **The PTQ1_0 decode datapath.** 16-lane `i16` is done (6.75 -> 9.46 Gweights/s per thread,
  ceiling 12.0 -> 16.3-16.6 GB/s); the table-driven Q2_0 form was measured 35% *slower* here, so
  do not retry it. Decode sits at ~90% of its own ceiling and DRAM is ~25% used.
- **Paired prefill.** `forward_pair` pairs every projection and runs the scans per token in
  order, which is why output is bit-identical; `HEARTH_Q35_NOPAIR=1` reverts. One copy of the
  per-token FFN exists, in `ffn_block`.
- **Stage-level waste in the 27B.** `matvec_dense` is split over the pool (`gdn gates` 20 -> 3-4
  ms), and what remains is size-limited throughput: 6.7-11.2 MiB projections read 77-85% of
  their own ceiling, so a stage number below `kernel-bench`'s 140 MiB headline is expected.
- **The Q1_0 ffn lever.** Post-hoc 1-bit destroys a converted model (32.8% exact zeros, relL2
  0.575, EOS). Needs a 27B made 1-bit from source; the converter that measured it is in git at
  `1c9f5ef`, not in the tree — do not re-add one unless the answer changes.
- **The first token.** It is storage I/O on the first read of the weights; `HEARTH_PIN_WEIGHTS=1`
  died silently on the 27B on this 16 GB box.
- **The Windows packaging contract.** It is verified, not assumed: upgrade in place, one
  Add/Remove Programs row (a same-version reinstall still leaves exactly one), the models
  surviving an uninstall, no console window, and the payload under the per-user program files
  folder. `packaging/windows/README.md` lists the checks and `scripts/msi-query.ps1` is the
  gate (14 checks, exits non-zero).

## Open items (most valuable first)

1. **The scan is still the biggest stage after the FFN** — 161-169 of 436-451 ms/token prefill
   and 210-224 ms of ~590 decode, against 258-270 for the FFN. Layout and head split are done.
   What remains is not the strided read (~5 GB/s aggregate), so suspect the dispatch cadence
   (one `par_for` per GDN layer, ~48/token) and an L2-resident working set. Count bytes per
   layer against dispatches before changing anything.
2. **aarch64 has no SIMD backend**; the 1-bit/2-bit dots, the Hadamard rotation and the Q8
   activation quantize are x86-only and fall back to the scalar reference there. The elementwise
   ops and the 4/5-bit dots are portable SIMD and do run on NEON. NEON first suits the
   `Q2V_U8` LUT dot (`vqtbl`). Needs a real ARM runner before any number means anything.
3. **Worker counts follow the path** — dense = `physical`; hybrid `Q2_0`/`PQ2_0` = `physical +
   physical/4`; hybrid `PTQ1_0` = ¾ logical. `HEARTH_THREADS` overrides. Judge a count on its
   **spread** (`scripts/count-sweep.py`), not its mean.
4. **Speculative decoding: measure acceptance `a` first.** 3 of every 4 `qwen35` layers are
   recurrent, so a rejected draft must rewind conv+SSM state, not just a KV cache.
5. **GPU scaffolding: still not started** (opt-in via `--device`/env, graceful without a ROCm
   device, must not break the CPU path or Windows/macOS builds).
6. **Kernel rules that made past fixes findable**: a `#[target_feature]` body names every
   feature it uses and its dispatch gate names the same set; `.cargo/config.toml` pins
   `x86-64-v3` and must not drop below what any dispatched body uses. A new quant format needs
   `kernel_name()` and a `selected_kernel` arm once it has two implementations. Q5_0/Q5_1 remain
   0.35 Gweights/s single-thread (per-element `% 2` unpack before the SIMD madd) — rewrite only
   if such a model lands.
7. **The app shows no version.** The version lives in the packaging (the MSI `ProductVersion`,
   the macOS `Info.plist`), not in the binaries, and there is no `--version` flag — the CLI
   treats the argument as a model path. A UI "about" line would need `CARGO_PKG_VERSION` wired
   into the served page; it was left out of 0.3.0 on purpose (a rebuild to change a string).

## Goal 2 (UI / docs / downloader)

1. **The hero chips are grouped by family now** — any prose that describes the model list as a
   flat chip row is stale (README, `docs/INSTALL_UX.md`).
2. **The downloader screenshot shows per-model state** ("On disk" + bars), which is what the
   README's downloader paragraph should promise.
3. **The catalog only grows by hand.** A `models/catalog.json` in the repo would let a release
   add packs without a rebuild; the override path already exists (`HEARTH_MODEL_CATALOG_JSON`).
4. **The downloader's DNS fallback is unit-tested only** — check it against a real dual-stack
   host and a host whose first address is unreachable.
5. **The release notes body is static.** `.github/workflows/release.yml` carries one arm64
   paragraph and one install table; anything that changes between releases has to be edited
   there, and the arm64 paragraph was wrong until this session (it claimed every kernel falls
   back to scalar off x86-64).

## Standings and hazards

- **Re-baseline before believing a delta.** The README's dense rows sit 6-18% above today's box
  *including the pre-change code path*, so an absolute comparison is meaningless right now and a
  paired one is the only kind worth quoting.
- **This box's noise is bigger than most of the wins.** Per-run spread reaches ±10% (and much
  more with a browser loading the machine): one 4-pair run of the 1.7B Q1_0 read -8.4% with a
  spread of 14.1-43.3 tok/s, and the same model read +1.7% in a quiet window with a spread of
  41.3-44.4. Two pairs cannot resolve a 1-3% effect — use 4, and quote the sub-timer rows when
  the effect is smaller than the noise.
- **[generate] avg_cpu_overhead=Nus/token is mislabelled** — it prints the whole step including
  the forward, so it is not overhead and nothing is wasting that time.
- **The model list is a multi-root scan.** `hearth_core::paths` is the only answer to "where are the
  models"; do not re-add a walk-up elsewhere.
- **`p256.rs` is verification-only and variable-time by design**; TLS 1.3 here uses X25519 and the
  crate never signs.
- **Do not re-derive the prefill/decode asymmetry** — the 27B's prefill stays *faster* than its
  decode, and both are weight-bound.
- **Warm up before timing.** A cold 27B run reads 605 ms/token prefill against 504 warm, and a
  cold 8B Q2_0 read 2.88 tok/s decode against 6.64 warm. Bracket A/B as `b,a,a,b`.
- **Verify through the interface, not the summary**: drive the real API (`HEARTH_MODELS_DIR`,
  `HEARTH_MODEL_CATALOG_JSON`, `--no-browser`, `HEARTH_NO_AUTO_SHUTDOWN=1`), `--temp 0 --prompt-raw`
  for benchmarks.
- **`hearth-tls` is not a general-purpose TLS stack** — scope is "the downloader can reach `https://`
  with no dependencies"; unneeded features are *refused*, not ignored.
- **MSI facts worth not re-learning**: an OS-version launch condition cannot pass on a machine that
  emulates `VersionNT`; a `File` installs into its component's directory; `RemoveFolder` removes
  empty directories only (which is what preserves `models/`); `wix build` needs `-acceptEula wix7`;
  custom-action FileRefs need `ExeCommand=""`.
- **Runner labels fail silently.** `macos-13` is retired; use `macos-15-intel` / `macos-15`. The
  Linux leg is pinned to `ubuntu-24.04` to hold the glibc floor.
- **A live UI server holds the exe.** `target/release/hearth-ui.exe` and its `hearth-chat-cli.exe`
  child must be killed before anything can relink `target/release` (`netstat -ano` for the port;
  the bash wrapper's PID is not the exe's).
- **The shipped artifact must come from the shipped tree.** The MSI built before the `f16c` fix
  was thrown away and rebuilt; the proof that the installed engine is the right build is a
  SHA-256 match against `target/release`, not the file size.
- **CI can be red in one job while everything local is green** — the `kernel-baseline` guard
  builds at `x86-64-v2` on purpose, so it sees what a `x86-64-v3` release build hides. Watch the
  *whole* job list on the pushed SHA, not the job you remember.

## Verification habits for a shared checkout

- **Measure a before/after pair from one binary.** `HEARTH_QUANT_SCALAR=1` turns every vector
  path off, so the "before" arm is the same binary with the reference implementation. No second
  build, no file surgery on a tree other agents are editing, and no chance of comparing two
  different checkouts. Read once per process.
- **A paired A/B is the only kind worth quoting here** — see the noise note above.
- **Prove bit-identity with a test, not with matching texts.** Nine tests compare the vector
  path against the scalar reference byte for byte, including the cases where the two could
  disagree (zero block, ties, the clamp boundary, a scale that underflows, denormals, NaN,
  infinity, a partial tail block). That is stronger than five texts matching, and it is cheap.
- **Gate with the toolchain CI uses, not the default.** Stable is 1.100.0; the box's default is
  1.99.1 and it does not report every lint 1.100 does (it missed `needless_range_loop` in
  `model/mod.rs` for weeks of red builds, and it missed cargo's `unused_dependencies` entirely).
  `rustup toolchain install 1.100.0 --profile minimal --component clippy,rustfmt`, then
  `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`, `cargo +1.100.0 fmt --all
  --check`, and the same cross-target clippy with `--target` for `x86_64-unknown-linux-gnu` and
  the three aarch64 targets (clippy type-checks, so it needs no foreign linker).
- **`cargo fmt --all` reformats the other agent's files** — use `cargo fmt -p <crate>`, and if a
  shared file is not yours, fix your hunks by hand and let the owner reformat theirs.
- **`target/release/hearth-chat-cli.exe` is often locked by a running UI**: `netstat -ano` for the
  port, kill that PID; the bash wrapper's PID is not the exe's.
- **A constant copied from memory is not a vector** — derive it and have an independent
  implementation agree before embedding it.
- **A test whose two sides run the same code proves nothing**, and **a control that passes is a
  control that was never tested** — assert the domain directly.
- **Suspect your own harness before calling someone else's data broken.**
- **Timer rows need an exact field match.** `[q35timing]` rows are `$1="[q35timing]"`, `$2=name`
  (so `gdn gates` is `$2="gdn"`), and the ffn sub-rows put the value in `$3`; match `$2=="ffn"` or a
  regex on ` us `, never `/ffn /`, or the sub-rows arrive as extra matches.
- **In a shared checkout, check the room before touching a file that is already dirty.** Two
  agents can hold the same claim within seconds of each other; the one with the working patch in
  the tree lands it and the other takes the next task.
