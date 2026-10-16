# Hearth — next session

> **Session 2026-10-16 (pt 2) in one paragraph.** Finished the hand-rolled TLS handshake. `crates/hearth-tls`
> gained `util.rs` (one shared constant-time compare), `keyschedule.rs`, `handshake.rs` (framing,
> ClientHello, strict ServerHello, transcript), `client.rs` (the 1-RTT state machine: the encrypted
> server flight, fatal Finished check, our own Finished, `KeyUpdate` both ways, `close_notify`) and
> `tests/rfc8448_flight.rs`. **101 tests green on both the hardware and the forced-software path**,
> clippy `-D warnings` and `cargo fmt --check` clean, `std`-only, empty `[dependencies]`. The fixture is
> RFC 8448's real 679-octet encrypted server flight, which the AEAD tag and the published Finished MAC
> pin byte for byte, and the client's own Finished record re-seals to the trace's 58 bytes. Two things
> found by deriving rather than reading: a `researcher-web` transcription of the client Finished had
> **three wrong bytes** (29 of 32 matched, which is what gave it away; the RFC's text agrees with the
> derivation) and RFC 8448's `Certificate` message is internally inconsistent by two bytes (asserted,
> documented, not "fixed"). **Certificate verification is the remaining piece and it is the dangerous
> one**: until it exists the crate must not be wired to any `https://` path. All work is on
> `origin/main`. History, the anomaly and the full test list: `docs/BUG_TRACKER.md`.

## Next session — in this order

1. **Certificate verification — the last piece before the downloader can use any of this.**
   The seam already exists and is the only place to plug in:
   `client::CertificateVerifier`, called with `CertificateVerifyInput { certificate_list,
   signature_scheme, signature, signed_data }`, where `signed_data` is already the RFC 8446 §4.4.3
   construction (64 spaces, `"TLS 1.3, server CertificateVerify"`, a zero byte, then the transcript
   hash through `Certificate`). A `ClientHandshake` cannot be built without a verifier and the crate
   ships exactly one, `RejectAll`, which can only refuse — **keep it that way**: no accept-all type, no
   flag, no environment variable, nothing a URL can reach that weakens verification. A failed chain, a
   wrong signature, an expired certificate or a hostname mismatch must abort the connection.
   - **The verifier interface is settled and enforced — do not widen it.** `CertificateVerifier` is
     deliberately **not** implemented for closures (a `compile_fail` doctest in `client.rs` requires
     `|_| Ok(())` to be rejected), so every verifier is a named type, and `CertificateVerifyInput`
     carries `server_name` beside the chain, the scheme, the signature and the 130-byte `signed_data`.
     This came out of SparkOC's review: the blanket `Fn` impl had made an accept-all a one-liner, and
     the tests were using it. Keep `RejectAll` the only verifier in the crate.
   - Build it in the order that can be *proven* first. Start with **signature verification**, because
     RFC 8448 gives a vector for it: the trace's `CertificateVerify` is `rsa_pss_rsae_sha256` over the
     130-byte `signed_data` we already reconstruct byte-for-byte, with a 128-byte signature (the trace's
     test key is **RSA-1024** — its DER modulus is 128 bytes). The trace also publishes that key's
     *private* parts, so signing can be checked too. Then `ecdsa_secp256r1_sha256`, which is the other
     suite we advertise (P-256 needs its own field arithmetic; the x25519 limb approach ports, the
     curve does not). Generate extra vectors locally with OpenSSL/`cryptography` and pin them the same
     way slices 2-5 did, with the values re-derived here before being embedded.
   - Then the X.509 side: DER parsing (SEQUENCE/INTEGER/BIT STRING/OID), the fields that matter for a
     downloader (basicConstraints CA, keyUsage, validity, subjectAltName dNSName, the signature
     algorithm and the issuer/subject linkage), chain building to a pinned root, and hostname matching
     that is an exact/case-insensitive dNSName match with no wildcard surprises.
   - **The root store is a decision to make explicitly, not to drift into.** `std` has no way to read
     the OS trust store on all three platforms, so the choice is a small pinned set in-tree with
     fingerprints and a documented update path, versus a per-platform reader. Pick the pinned set, say
     so in the module docs, and make its contents reviewable in one screen.
   - Land it with the negative tests before any URL can use it: **bad signature, expired, not-yet-valid,
     unknown root, hostname mismatch, a CA used as a leaf (basicConstraints), a chain that is too long,
     and an RSA key that is too small**. Do not let the downloader take an `https://` path through this
     code until those exist. Measured calibration from the last session still stands: the primitives
     were ~600-900 lines, and this half is the 2000-3000 line one.
   - `Transport` in `hearth-ui/src/http.rs` is the seam and the `https://` branch of
     `downloads.rs::run_transfer` is what gets rewired afterwards. **Do not touch `http.rs` without
     asking — it is SparkOC's file.** Loose thread to close at the same time: the downloader's DNS
     fallback (`to_socket_addrs().next()` is gone, but the replacement is unit-tested only) has never
     been checked against a host with a real A/AAAA pair, and that becomes live the moment TLS routes
     through this code.
2. **`aarch64` PMULL/AES** so macOS gets the slice-3 numbers instead of the software path (AES-128
   68.8 MB/s, GCM 26 MB/s). Small, self-contained, and the software reference and
   `HEARTH_TLS_NO_HW=1` already exist to test it against.
3. **The downloader's DNS thread from item 1** if it is not closed with the TLS work: verify the
   fallback against a real dual-stack host and against a host whose first address is unreachable.
4. **Size the chunked-prefill win before designing it.** There is still **zero weight reuse** in the
   hybrid prefill: it re-streams every weight row once per prompt token, so prefill's per-token cost
   is ~= decode's (4B: 144 vs 117 ms/token). A chunked forward would amortise the weight *unpacking*
   across the chunk while the MAC count scales with tokens, so the ceiling is roughly `1 - f` where
   `f` is the share of per-token time spent unpacking rather than multiplying. **Measure `f` first** —
   the cheap probe is the same prompt at a fixed length on PQ2_0 (2-bit) versus PTQ1_0 (1-bit): a
   materially different prefill/decode ratio isolates unpack cost. `dot_ptq1_0_q8_0x2_ptr` is the
   tested primitive it needs (bit-identical to two single calls, **currently unreferenced** — do not
   re-wire it into `matmul_batch`, that path cannot reach PTQ1_0 models). An x4 variant regressed
   3.2× in pt 5.
5. **Speculative decoding: measure acceptance before designing anything.** This is the strongest
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
6. **Sweep for other silently-wrong feature gates.** `cpu_is_amd_zen4()` was the only family check,
   but the bug class is "a gate that is always-false or always-true". Grep `hearth-quant` and
   `hearth-llm` for `is_x86_feature_detected!`, `target_feature` and `avx512`, and confirm each
   gated kernel is on the *faster* path here. `cargo run --release -p hearth-llm --example benchk --
   <model.gguf> <dtype>` prints the CPUID decode, the kernel `dot_*_ptr` selects, and every variant's
   GB/s. Still open from two sessions ago; nothing regressed while it waited.
7. **Zen 5 (family 1Ah) and Intel must not regress.** The fix keys on 19h specifically so they keep
   full-width AVX-512, and there is no such hardware here to confirm it.
8. **Keep dispatch introspection honest for each new format.** `selected_kernel(dtype)` +
   `dispatch_report()` cover PTQ1_0, PQ2_0, Q2_0, Q1_0 and Q1_0_G128; the `[kernel]` banner prints at
   load on both paths and the service exposes `event.model.kernels`. A new quant type must add a
   `kernel_name()` arm and a `selected_kernel` arm or it will silently have no reported dispatch.
   Agreed rule: add a format to `selected_kernel` only when it has more than one implementation to
   choose between.
9. **Three small things.** (a) AGENTS.md's verifications block says `cargo test --test-threads=1 -p
   hearth-quant`, which cargo rejects — it must be `cargo test -p hearth-quant -- --test-threads=1`.
   (b) Re-run the benchmark mandate on a fully idle box interleaved with the reference fork
   (`bench_results/run_cpu_bench.bat`). (c) `cargo check -p hearth-diffusion` warns that
   `FluxTransformer::gpu_f32_bufs` (`transformer.rs:46`) is never read — pre-existing, in SparkOC's
   area, deliberately left alone.

## Standings and hazards

- **Do not re-derive the prefill/decode asymmetry.** It was answered and the premise was wrong: with
  `HEARTH_Q35_TIMERS=1`, the 27B PTQ1_0 is 635 ms/token prefill against 722 decode (prefill is
  *faster*), and the 4B is 124/144 prefill vs 117/229 decode. The old "1.8-2.3 s per prompt token"
  came from a pt-6 run under ~75 % external load with a cold file cache.
- **Do not "fix" the first token.** The excess is storage I/O on the first read of the weight file.
  A page-touch prefault was written and measured to change nothing (173.0 vs 173.4 ms) while costing
  259 ms (4B) / 2402 ms (9B) of load, so it was reverted. **`HEARTH_PIN_WEIGHTS=1` on the 27B died
  silently** on this 16 GB box with other models resident — never recommend it as a default.
- **Do not trust a summary; verify through the interface.** Every claim in earlier reviews came from
  driving the real API with a catalog override (`HEARTH_MODELS_DIR` + `HEARTH_MODEL_CATALOG_JSON`,
  `--no-browser`, `HEARTH_NO_AUTO_SHUTDOWN=1`) against a local server that logs `Range` headers — not
  from reading code. Three of those findings tested clean, and the one defect that remained was only
  visible that way.
- **`crates/hearth-tls` is not a general-purpose TLS stack.** Its scope is "the downloader can reach
  `https://` on Linux, macOS and Windows with no dependencies". Every unneeded feature (client auth,
  resumption, 0-RTT, ALPN, post-handshake auth) is *refused* rather than ignored, which is the
  property that keeps the surface small enough to review.

## Verification habits for a shared checkout

- `cargo fmt --all` reformats the *whole workspace* including the other agent's files — use
  `cargo fmt -p <crate>`. Same for clippy with `-D warnings`.
- `target/release/hearth-chat-cli.exe` is often locked by a running UI: build to a spare
  `CARGO_TARGET_DIR` instead of killing a process you do not own.
- **Suspect your own harness before reporting someone else's data as broken.** Three times now a
  "finding" was my own bug: a helper that wrote CRLF made curl report HTTP 000 on 14 valid URLs, a
  Python `allow_reuse_address` server silently answered a later test on the same port on Windows, and
  in the handshake slice a ClientHello slice that still carried its 4-byte handshake header shifted my
  session-id offsets by four bytes.
- **A published vector is only as good as the copying.** Every constant in `hearth-tls` is either
  re-derived locally from an independent implementation (OpenSSL via Python's `cryptography`) or
  asserted against a second published value that covers it. The handshake slice's client Finished
  arrived with three wrong bytes; only the derivation caught it.
- Test fixtures must be *correct*, not just the right length: a 50-byte prefix of `b"\x00"` is not a
  prefix of `bytes(range(100))`, and that mistake made a valid resume look like a content mismatch.
