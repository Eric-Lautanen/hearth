# Hearth — next session

> **Session 2026-10-15 (pt 2) in one paragraph.** Peer-reviewed SparkOC's downloader, verified every
> claim through the real `hearth-ui` HTTP API instead of by reading, and found one new defect (since
> fixed by them and re-verified end to end, both the self-heal and the refuse-to-publish-a-short-part
> half). Started the clippy clear-up (36 → 31 sites, including a real no-op `drop` of a reference).
> Corrected
> a research claim about speculative decoding and measured the numbers it depends on. Retired the
> "hybrid prefill is 2× slower than decode" premise that had been in this file since pt 6 — it was
> false (27B: prefill 635 ms/token vs decode 722). Started the hand-rolled TLS work the operator asked
> for: a new **`crates/hearth-tls`** crate (std only, **zero dependencies**) with SHA-256, HMAC-SHA256
> and the RFC 8446 key schedule, all vector-tested. **All of it is on `origin/main`** — SparkOC pushed
> my `mod.rs` slice as `50c88bc` while I was out, finished the remaining sites as `f3c18d4` (CI green),
> and I then re-derived their benchmark gate from scratch builds: no regression. **Operator rule this
> session: std Rust only, no new dependencies** — so the TLS stack is hand-rolled end to end, no
> `rustls`. History and numbers: `docs/BUG_TRACKER.md`.

## Next session — in this order

0. **Clippy pass: DONE (SparkOC finished it, 2026-10-15 night).**
   `cargo clippy --workspace --all-targets -- -D warnings` is green on
   stable 1.99.1, 98 tests pass, fmt clean. What it took beyond the 31
   counted sites: the newer toolchain added `redundant-reference-in-format`
   (32 sites, all `&wnames[` in cold error strings → `.as_str()`; two needed
   `|i: usize|` annotations), `ptr_arg` (`tmp: &mut Vec` → slice; `q8`
   keeps `Vec` for `clear()`), `chunks_exact_to_as_chunks` (quant BF16,
   gpu.rs, probes, diffusion weights — needs Rust 1.88+, README still says
   1.85), 8 signature groupings into params structs (pool ×2, fused ×2,
   attention ×3, forward_gpu), all hot index loops to order-preserving
   iterators, diffusion loops/signatures/dead-stub removal, 5 probe files,
   chat-cli test module moved to file end. Benchmark gate HELD then
   CLEARED: post-change sweep first looked 10–25% down on Q2_0 models, but
   an interleaved old-vs-new A/B (pre-pass binary from an isolated worktree
   at `d571b8a`) agreed within ~3% on all four disputed models — machine
   drift across the evening, not code. Lesson: single-run charts carry
   ±20% machine noise; interleave before believing a delta.
   **Independently re-derived 2026-10-16 (buffy), gate closed:** clippy exits 0 here too, and both
   revisions were rebuilt from scratch (worktree + separate target dirs) and interleaved on all 6
   mandate models — everything within ±3%. The Q2_0 dip shows in **both** arms (8B Q2_0: 5.05 pre vs
   5.06 post against the chart's 6.75), which is what proves it is machine state rather than code.
   Nothing left to do on this item.
1. **TLS slices 1–2: DONE (buffy, 2026-10-16).** `crates/hearth-tls` now has `sha256.rs`, `hkdf.rs`,
   `aes.rs`, `gcm.rs` and `x25519.rs` — **40 tests green**, clippy `-D warnings` and fmt clean, `std`
   only with an empty `[dependencies]`. AES-128 (S-box derived at compile time, not transcribed),
   AES-128-GCM (table-free branchless GHASH, constant-time tag compare, verify-before-decrypt,
   no API returning unauthenticated plaintext) and X25519 (51-bit limbs, constant-time ladder,
   small-order points refused with no unchecked twin for protocol code). Every constant was
   re-derived from the OpenSSL-backed `cryptography` package before being pinned, and the oracle
   scripts assert the published spec values too. Details and the throughput table: BUG_TRACKER
   "hearth-tls slice 2".
2. **hardware acceleration: DONE (buffy, 2026-10-16) — measured, not assumed.** AES-NI and
   PCLMULQDQ are runtime-dispatched (`aesni.rs`, `ghash_x86.rs`), software kept as the reference,
   `Aes128Gcm::accelerated()` reports which paths an instance took, and `HEARTH_TLS_NO_HW=1` forces
   the software path for an A/B on one binary (it can only pick the *slower* path, never a weaker
   check). Interleaved result: AES-128 **68.8 → 1304 MB/s**, GCM **26.4 → 522-547 MB/s** (20x), so a
   4 GiB model's decrypt drops from 159 s of CPU to 8 s. All 43 tests pass on **both** paths, so the
   published vectors themselves run through the SIMD code. Details, the derivation of the reduction
   (no magic constant) and one test-side OR-vs-XOR mistake worth not repeating: BUG_TRACKER
   "hearth-tls slice 3".
   Note the corrected framing in that entry: software GCM at 26 MB/s still stays ahead of a 100 Mbps
   link, so the old "56% of the transfer" claim was an overstatement — it is CPU time, and it only
   caps throughput on gigabit links and local mirrors.
3. **Record layer: DONE (buffy, 2026-10-16).** `record.rs` does AEAD framing, derives the nonce from
   the sequence number (owned by the struct, no setter, `SequenceExhausted` rather than wrapping), uses
   the encrypted header as AAD, and implements the record-type-in-inner-plaintext trick with optional
   padding. **55 tests green on both the hardware and the forced-software path.** The fixture is
   RFC 8448's own protected `application_data` record, obtained via `researcher-web` and then
   **re-derived here with the independent OpenSSL-backed library before being embedded** — the
   published key, the IV and the whole record all reproduce from the traffic secret, which pins the
   nonce, the AAD, the inner content type and the layout at once. Details: BUG_TRACKER "slice 4".
4. **Next: the handshake**, the last piece before the downloader can use any of this. Scope:
   ClientHello/ServerHello with `TLS_AES_128_GCM_SHA256`, `x25519` key share (already available), the
   full key schedule from `HKDF-Extract` through `Derive-Secret` and the `traffic upd`/`key`/`iv`
   labels, the transcript hash over the handshake messages, and **Finished MAC verification where a
   mismatch aborts the connection**. RFC 8448 has published vectors for every intermediate secret and
   for the handshake messages themselves, so this slice can be driven entirely by published values
   the same way slices 2-4 were — obtain them through `researcher-web` and re-derive them locally
   before trusting them. Also on the list: the alert protocol, `key_update` (which reuses
   `record::TrafficKeys::from_secret`), and `aarch64` PMULL/AES so macOS gets the slice-3 numbers.
   Only then certificate verification, and **nothing may take an `https://` path until that exists**.
   **Verification policy is fixed: fatal, never downgradeable — no switch that a URL or a user can
   reach can weaken it.**
   - `Transport` in `hearth-ui/src/http.rs` is the seam and is already written for this; the `https://`
     branch of `downloads.rs::run_transfer` is what gets rewired once the record layer works. Do not
     touch `http.rs` without asking — it is SparkOC's file.
   - Measured, for calibration: `curl_available()` + the whole downloader is only the *transport*
     question; the primitives are ~600–900 lines of arithmetic, where X.509 chain validation plus
     RSA/ECDSA verification plus a pinned root store is the 2000–3000 line half. Budget accordingly.
   - **Operator rule received 2026-10-15 (closes the open question in the pt-2 notes): std Rust only,
     no dependencies.** There is therefore no `rustls` escape hatch and no "bundle a vetted verifier"
     option — the X.509/chain/RSA/ECDSA half is hand-rolled too, which makes the fatal, never-
     downgradeable verify policy the *entire* safety story rather than a belt-and-braces extra. Build
     it in the order given above and land it with negative tests (bad signature, expired, unknown
     root, hostname mismatch) before any URL is allowed to use it. Do not let the downloader take an
     `https://` path through this code until those tests exist.
5. **The downloader defect I found: FIXED by SparkOC (`c8782dc`) and re-verified by me end to end —
   closed.** The dead end was a full-size `.part` plus a **416** to `Range: bytes=<size>-`. Both
   candidate fixes landed (heal on 416, and `part_len()` awareness). I re-ran my own repro against
   the fixed binary rather than trusting the diff: the full-size case now goes `partial 100/100` →
   press → `done`, the file is adopted, no error; and the safety half holds too — a 416 with a
   *short* partial refuses to claim completion instead of publishing a truncated file. Also landed
   in the same commit: the redirect now keeps the `Range` header. **The DNS fallback is implemented
   but only unit-tested** — `to_socket_addrs().next()` (first address only) is gone, yet nobody has
   checked it against a host with a real A/AAAA pair, which is latent while catalog URLs are
   `https://` through curl and live the moment TLS routes them here. That is the one loose thread
   left in this item.
6. **Size the chunked-prefill win before designing it.** There is still **zero weight reuse** in the
   hybrid prefill: it re-streams every weight row once per prompt token, so prefill's per-token cost
   is ~= decode's (4B: 144 vs 117 ms/token). A chunked forward would amortise the weight *unpacking*
   across the chunk while the MAC count scales with tokens, so the ceiling is roughly `1 - f` where
   `f` is the share of per-token time spent unpacking rather than multiplying. **Measure `f` first**
   — the cheap probe is the same prompt at a fixed length on PQ2_0 (2-bit) versus PTQ1_0 (1-bit):
   a materially different prefill/decode ratio isolates unpack cost. `dot_ptq1_0_q8_0x2_ptr` is the
   tested primitive it needs (bit-identical to two single calls, **currently unreferenced** — do not
   re-wire it into `matmul_batch`, that path cannot reach PTQ1_0 models). An x4 variant regressed
   3.2× in pt 5.
7. **Speculative decoding: measure acceptance before designing anything.** This is the strongest
   decode idea on the table and the cheapest thing to falsify. 2B PTQ1_0 decodes at **50.8 ms/token
   (18.15 tok/s)** against the 27B's ~722 ms, i.e. the draft is 14× cheaper, so with acceptance `a`
   and draft length `k` the gain is `(1-a^(k+1))/(1-a)` per `1+0.07k` target-equivalents: ≈1.8× at
   a=0.6, ≈2.2× at a=0.7, ≈2.6× at a=0.8 (k=4). **`a` is unmeasured and the whole plan rests on it.**
   Prerequisite is **green**: the 2B and 27B tokenizers are byte-identical (same 248320-token list,
   same 247587 merges, same `token_type`, same bos/eos/padding 248044/248046/248044). Two caveats:
   `tokenizer.chat_template` differs (render with the target's template, feed both the same ids) and
   `prism.hadamard.version` is 2/512 on the 2B vs 1/1024 on the 27B, so the two are not in the same
   weight basis — argue acceptance from measurement, never from "same family". Losslessness is a
   property of the *acceptance rule*: the greedy/argmax form is exact and matches our `--temp 0`
   benchmark path; the sampling form needs rejection plus residual correction over the
   post-truncation distribution and must handle the stateful repeat penalty. Cost to plan for: **3 of
   every 4 `qwen35` layers are recurrent**, so a rejected draft token must rewind conv+SSM state, not
   just a KV cache.
8. **Sweep for other silently-wrong feature gates.** `cpu_is_amd_zen4()` was the only family check,
   but the bug class is "a gate that is always-false or always-true". Grep `hearth-quant` and
   `hearth-llm` for `is_x86_feature_detected!`, `target_feature` and `avx512`, and confirm each
   gated kernel is on the *faster* path here. `cargo run --release -p hearth-llm --example benchk --
   <model.gguf> <dtype>` prints the CPUID decode, the kernel `dot_*_ptr` selects, and every variant's
   GB/s. Still open from the last session; nothing regressed while it waited.
9. **Zen 5 (family 1Ah) and Intel must not regress.** The fix keys on 19h specifically so they keep
   full-width AVX-512, and there is no such hardware here to confirm it.
10. **Keep dispatch introspection honest for each new format.** `selected_kernel(dtype)` +
   `dispatch_report()` cover PTQ1_0, PQ2_0, Q2_0, Q1_0 and Q1_0_G128; the `[kernel]` banner prints at
   load on both paths and the service exposes `event.model.kernels`. A new quant type must add a
   `kernel_name()` arm and a `selected_kernel` arm or it will silently have no reported dispatch.
   Agreed rule: add a format to `selected_kernel` only when it has more than one implementation to
   choose between.
11. **Three small things.** (a) AGENTS.md's verifications block says `cargo test --test-threads=1 -p
   hearth-quant`, which cargo rejects — it must be `cargo test -p hearth-quant -- --test-threads=1`.
   (b) Re-run the benchmark mandate on a fully idle box interleaved with the reference fork
   (`bench_results/run_cpu_bench.bat`). (c) `cargo check -p hearth-diffusion` warns that
   `FluxTransformer::gpu_f32_bufs` (`transformer.rs:46`) is never read — pre-existing, in SparkOC's
   area, deliberately left alone.

## Standings and hazards

- **Do not re-derive the prefill/decode asymmetry.** It was answered and the premise was wrong: with
  `HEARTH_Q35_TIMERS=1`, the 27B PTQ1_0 is 635 ms/token prefill against 722 decode (prefill is
  *faster*), and the 4B is 124/144 prefill vs 117/229 decode. The old "1.8–2.3 s per prompt token"
  came from a pt-6 run under ~75 % external load with a cold file cache.
- **Do not "fix" the first token.** The excess is storage I/O on the first read of the weight file.
  A page-touch prefault was written and measured to change nothing (173.0 vs 173.4 ms) while costing
  259 ms (4B) / 2402 ms (9B) of load, so it was reverted. **`HEARTH_PIN_WEIGHTS=1` on the 27B died
  silently** on this 16 GB box with other models resident — never recommend it as a default.
- **Do not trust a summary; verify through the interface.** Every claim in the review above came from
  driving the real API with a catalog override (`HEARTH_MODELS_DIR` + `HEARTH_MODEL_CATALOG_JSON`,
  `--no-browser`, `HEARTH_NO_AUTO_SHUTDOWN=1`) against a local server that logs `Range` headers — not
  from reading code. Three of my earlier findings tested clean, and the one defect that remains was
  only visible that way.

## Verification habits for a shared checkout

- `cargo fmt --all` reformats the *whole workspace* including the other agent's files — use
  `cargo fmt -p <crate>`. Same for clippy with `-D warnings`.
- `target/release/hearth-chat-cli.exe` is often locked by a running UI: build to a spare
  `CARGO_TARGET_DIR` instead of killing a process you do not own.
- **Suspect your own harness before reporting someone else's data as broken.** Twice this session a
  "finding" was my own bug: a helper that wrote CRLF made curl report HTTP 000 on 14 valid URLs, and a
  Python `allow_reuse_address` server left listening on my test port silently answered a later test
  (Windows lets a second bind succeed), which briefly made a real edge case look handled.
- Test fixtures must be *correct*, not just the right length: a 50-byte prefix of `b"\x00"` is not a
  prefix of `bytes(range(100))`, and that mistake made a valid resume look like a content mismatch.
