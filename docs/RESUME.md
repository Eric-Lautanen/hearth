# Hearth — next session

> **Session 2026-10-15 (pt 2) in one paragraph.** Peer-reviewed SparkOC's downloader, verified every
> claim through the real `hearth-ui` HTTP API instead of by reading, and found one new defect. Corrected
> a research claim about speculative decoding and measured the numbers it depends on. Retired the
> "hybrid prefill is 2× slower than decode" premise that had been in this file since pt 6 — it was
> false (27B: prefill 635 ms/token vs decode 722). Started the hand-rolled TLS work the operator asked
> for: a new **`crates/hearth-tls`** crate (std only, **zero dependencies**) with SHA-256, HMAC-SHA256
> and the RFC 8446 key schedule, all vector-tested. Everything from this session is committed locally;
> nothing was pushed. History and numbers: `docs/BUG_TRACKER.md`.

## Next session — in this order

1. **TLS slice 2: AES-128-GCM, then X25519.** `crates/hearth-tls` exists and is in the workspace;
   `sha256.rs` (SHA-256 + HMAC-SHA256) and `hkdf.rs` (`hkdf_extract`/`hkdf_expand`/`hkdf_expand_label`/
   `derive_secret`) are done, 15 tests, all against published vectors (FIPS 180-4, RFC 4231, RFC 5869).
   Next, in the order the handshake needs them: **AES-128-GCM** (the mandatory TLS 1.3 suite — GCM
   needs a constant-time GHASH; test with the NIST GCM vectors *and* a tamper test proving a flipped
   tag bit is rejected) then **X25519** (RFC 7748 vectors, including the low-order/zero-output cases,
   which must fail rather than produce a silent all-zero shared secret). Then the record layer, then the
   handshake, then certificate verification last. **Verification policy is fixed: fatal, never
   downgradeable — no switch that a URL or a user can reach can weaken it.**
   - `Transport` in `hearth-ui/src/http.rs` is the seam and is already written for this; the `https://`
     branch of `downloads.rs::run_transfer` is what gets rewired once the record layer works. Do not
     touch `http.rs` without asking — it is SparkOC's file.
   - Measured, for calibration: `curl_available()` + the whole downloader is only the *transport*
     question; the primitives are ~600–900 lines of arithmetic, where X.509 chain validation plus
     RSA/ECDSA verification plus a pinned root store is the 2000–3000 line half. Budget accordingly.
2. **The downloader defect I found, unfixed as of writing.** A full-size `.part` can never complete:
   with `<name>.gguf.part` holding exactly `size_bytes` and the server answering **416** to
   `Range: bytes=<size>-`, pressing Download yields `state=error error="HTTP 416" exists=false`, and
   every retry reproduces it — the file must be deleted by hand while the UI shows 100 % beside an
   error. Cause: `start()`'s completion shortcut reads `on_disk()` (final name only), and
   `http::get` treats 416 as fatal while `run_http` retries only on `range_ignored`. Fix is either
   "416 while resuming = range already satisfied → publish", or checking `part_len()` in `start()`.
   SparkOC owns this file. **Also unfixed, code-reading only:** `to_socket_addrs().next()` uses just
   the first address with no fallback across A/AAAA records (curl tries them all) — latent while
   catalog URLs are `https://` and go through curl, live the moment TLS routes them here.
3. **Size the chunked-prefill win before designing it.** There is still **zero weight reuse** in the
   hybrid prefill: it re-streams every weight row once per prompt token, so prefill's per-token cost
   is ~= decode's (4B: 144 vs 117 ms/token). A chunked forward would amortise the weight *unpacking*
   across the chunk while the MAC count scales with tokens, so the ceiling is roughly `1 - f` where
   `f` is the share of per-token time spent unpacking rather than multiplying. **Measure `f` first**
   — the cheap probe is the same prompt at a fixed length on PQ2_0 (2-bit) versus PTQ1_0 (1-bit):
   a materially different prefill/decode ratio isolates unpack cost. `dot_ptq1_0_q8_0x2_ptr` is the
   tested primitive it needs (bit-identical to two single calls, **currently unreferenced** — do not
   re-wire it into `matmul_batch`, that path cannot reach PTQ1_0 models). An x4 variant regressed
   3.2× in pt 5.
4. **Speculative decoding: measure acceptance before designing anything.** This is the strongest
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
5. **Sweep for other silently-wrong feature gates.** `cpu_is_amd_zen4()` was the only family check,
   but the bug class is "a gate that is always-false or always-true". Grep `hearth-quant` and
   `hearth-llm` for `is_x86_feature_detected!`, `target_feature` and `avx512`, and confirm each
   gated kernel is on the *faster* path here. `cargo run --release -p hearth-llm --example benchk --
   <model.gguf> <dtype>` prints the CPUID decode, the kernel `dot_*_ptr` selects, and every variant's
   GB/s. Still open from the last session; nothing regressed while it waited.
6. **Zen 5 (family 1Ah) and Intel must not regress.** The fix keys on 19h specifically so they keep
   full-width AVX-512, and there is no such hardware here to confirm it.
7. **Keep dispatch introspection honest for each new format.** `selected_kernel(dtype)` +
   `dispatch_report()` cover PTQ1_0, PQ2_0, Q2_0, Q1_0 and Q1_0_G128; the `[kernel]` banner prints at
   load on both paths and the service exposes `event.model.kernels`. A new quant type must add a
   `kernel_name()` arm and a `selected_kernel` arm or it will silently have no reported dispatch.
   Agreed rule: add a format to `selected_kernel` only when it has more than one implementation to
   choose between.
8. **Three small things.** (a) AGENTS.md's verifications block says `cargo test --test-threads=1 -p
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
