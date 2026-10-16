# Hearth — next session

> **Session 2026-10-16 (pt 4, continued) in one paragraph.** Chain building exists. `chain.rs` takes the
> four fields a verifier is handed — `certificate_list`, the caller's trust anchors, the validation
> instant, a `Policy` — and answers whether the path frames, parses, respects its structural rules, is
> signed by the next certificate's key at every link, and ends at an anchor's key. Structure is checked
> before links, so a chain of the wrong shape is refused before any public-key operation runs on hostile
> input, and a **name match alone anchors nothing**: the key must verify. That last claim is provable
> rather than asserted, because a new fixture (`ROOT_SHADOW`) is a self-signed CA that carries the root's
> *subject name* with a different key — without it, the path a name-only matcher accepts had no negative
> test, which is the same pattern as the shared-key defect of slice 9, one level up. `client.rs`'s
> certificate framing now calls the same entry walk the verifier uses, so the rule exists once. What is
> still missing is the last check before a `https://` URL can use any of this: **host name matching**,
> plus the two certificate rules that need their own failure fixtures (`pathLenConstraint`,
> `keyUsage.keyCertSign`) and the fact that there is **no revocation checking of any kind**. **195
> checks green on the hardware path and on `HEARTH_TLS_NO_HW=1`** (177 before), clippy
> `--all-targets -D warnings`, fmt, `cargo doc` (zero warnings) and the workspace check clean, `std`-only,
> empty `[dependencies]`, still unreachable from any `https://` path. History: `docs/BUG_TRACKER.md`.

## Next session — in this order

1. **Certificate verification — one check short of a decision.** A certificate parses (`x509.rs`), every
   suite verifies (`rsa.rs`, `ecdsa.rs`), `verify.rs` decides which scheme may verify which key, and
   `chain.rs` builds the path from the wire bytes to a caller-supplied anchor with the structural rules
   and the validity window. **What is missing is host name matching**, which is now the single check
   standing between this crate and an `https://` URL — a chain that verifies today may still be for the
   wrong server. The pieces are also still wired to nothing: nothing in this crate calls `chain.rs`, and
   `RejectAll` remains the only `CertificateVerifier`.
   - The seam is unchanged and is the only place to plug in: `client::CertificateVerifier`, called with
     `CertificateVerifyInput { server_name, certificate_list, signature_scheme, signature,
     signed_data }`, where `signed_data` is already the RFC 8446 §4.4.3 construction (64 spaces,
     `"TLS 1.3, server CertificateVerify"`, a zero byte, then the transcript hash through
     `Certificate`). A `ClientHandshake` cannot be built without a verifier and the crate ships exactly
     one, `RejectAll`, which can only refuse — **keep it that way**: no accept-all type, no flag, no
     environment variable, nothing a URL can reach that weakens verification.
   - **The verifier interface is settled and enforced — do not widen it.** `CertificateVerifier` is
     deliberately **not** implemented for closures (a `compile_fail` doctest in `client.rs` requires
     `|_| Ok(())` to be rejected), so every verifier is a named type. This came out of SparkOC's
     review: the blanket `Fn` impl had made an accept-all a one-liner, and the tests were using it.
   - Build the checks in the order each one can be *proven*:
     1. **Key-to-scheme matching — done, in `verify.rs`.** `Suite::from_certificate_algorithm` maps a
        certificate's own `AlgorithmIdentifier` to a suite (refusing a PSS salt length that is not the
        hash length) and `Suite::from_handshake_scheme` maps a wire code point the same way, so `0x0804`
        (`rsa_pss_rsae_sha256`) is only ever verified with an RSA key and `0x0403`
        (`ecdsa_secp256r1_sha256`) only with a P-256 one; a mismatch is
        `VerifyError::SchemeKeyMismatch`, naming both. Cross-checking a scheme against a key of the
        wrong type is the classic way a verifier is walked into an RSA key being read as an EC point or
        a curve being substituted, so it is checked before any arithmetic. **What is left is the
        calling:** each chain link's `AlgorithmIdentifier` and its issuer's key go to
        `verify_certificate_signature`, and `CertificateVerify`'s scheme with the leaf's key goes to
        `verify_handshake_signature` — which accepts only the two schemes RFC 8446 kept, so a peer
        cannot get PKCS#1 v1.5 accepted as a handshake signature. The suite constants are
        `handshake::SIG_RSA_PSS_RSAE_SHA256` and `handshake::SIG_ECDSA_SECP256R1_SHA256`.
     2. **Chain building to a pinned root — done, in `chain.rs`.** Links are compared as the raw DER of
        the names (never as parsed strings), every issuer must be a CA (`basicConstraints`) and the leaf
        must not be, chain length is capped by `Policy::max_chain_len`, and a minimum RSA key size is
        enforced on every certificate in the path *and* on the anchor — `rsa.rs` deliberately accepts a
        1024-bit key because RFC 8448's fixture is one, so that policy lives here and not there. **What
        is deliberately not in it yet**: `pathLenConstraint` and `keyUsage.keyCertSign`, each of which
        needs a fixture that can make it fail — a CA with `path_len = 0` that has another CA below it,
        and a CA whose `keyUsage` lacks `keyCertSign` (both need a child certificate to test the link,
        so two fixtures each). Also absent: revocation of any kind, and `extendedKeyUsage` on issuers.
     3. **Hostname matching** — exact or case-insensitive `dNSName` match against `server_name`, no
        wildcard surprises (if wildcards ship at all, one leading label only, and the prefix must not
        be empty or match a public suffix).
     4. **The validity window — done, in `chain.rs`**, and checked on every certificate in the path and
        on the anchor, inclusively at both ends, against the `UnixTime` the handshake already carries.
        Still to do: `keyUsage` / `extendedKeyUsage` sanity on the *leaf* (`serverAuth`), plus the two
        revert-check-less call sites named above (the anchor's window and its key size have no negative
        test because no fixture anchor can fail them while the chain is valid).
   - **The root store is a decision to make explicitly, not to drift into.** `std` has no way to read
     the OS trust store on all three platforms, so the choice is a small pinned set in-tree with
     fingerprints and a documented update path, versus a per-platform reader. Pick the pinned set, say
     so in the module docs, and make its contents reviewable in one screen.
   - Land it with the negative tests before any URL can use it: **expired, not-yet-valid, unknown
     root, hostname mismatch, a CA used as a leaf (basicConstraints), a chain that is too long, an RSA
     key that is too small, and a scheme that does not match the key type**. The "bad signature" and
     "malformed encoding" cases now exist at the primitive level (`tests/ecdsa_vectors.rs`,
     `tests/rsa_vectors.rs`), so what is left is the chain-level set. Do not let the downloader take an
     `https://` path through this code until those exist. Calibration from the three slices: RSA was
     ~600 lines, the X.509 reading half ~1000, and P-256 + ECDSA ~1000 including their vectors — so the
     decision half is plausibly another 600–900, most of it policy and tests rather than arithmetic.
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
10. **P-256 verification is 345.6 µs, and the levers are known.** Measured in release with the
   `--ignored` timing test in `tests/ecdsa_vectors.rs`. That is several times slower than OpenSSL and
   the reasons are not mysterious, in descending order of payoff: (a) the fixed-base multiply has no
   precomputed table and no windowing, so `u1 * G` costs the same as `u2 * Q` even though `G` is
   constant — a small comb or window table for the base is the single biggest win; (b) the scalar
   multiplication is plain left-to-right double-and-add rather than a signed-digit or sliding-window
   form; (c) the multiplication is the coarse "multiply then reduce" Montgomery product from `rsa.rs`
   rather than an interleaved CIOS one; (d) the scalar inverse (and the field inverse per point) uses
   Fermat, ~384 multiplications, where a binary extended Euclidean inverse is a few dozen. **Do not do
   any of this before measuring**, and note that (a) and (b) make the *code* harder to audit in the one
   place where an error is a security bug — this is a 350 µs cost paid once per connection, against a
   download measured in seconds. Profile first, and treat anything that makes `p256.rs` less obviously
   correct as a bad trade unless a connection setup is actually on a critical path.

## Standings and hazards

- **`p256.rs` is verification-only, and variable time by design.** Scalar multiplication branches on
  the scalar's bits, which is safe because every scalar this crate multiplies by comes from a signature
  or a hash. It is therefore *unfit for signing or ECDH as written*: either would need a ladder with
  conditional swaps and its own tests. TLS 1.3 here uses X25519 and this crate never signs, so keep it
  that way — and if that ever changes, do not reuse `mul` as it stands.

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
- **A constant copied from memory is not a vector.** The X.509 slice's ten `UnixTime` expectations were
  written by hand and every one was wrong; the code was right. Re-derive from an *independent*
  implementation (CPython's `datetime` beside the `days_from_civil` algorithm, or OpenSSL beside the
  RFC's text) and let the two of them agree before embedding anything. A published vector is only as
  good as the copying — the handshake slice's client Finished arrived from a web research pass with
  three wrong bytes, and only the derivation caught it.
- **A test whose two sides run the same code proves nothing, however many assertions it has.** The
  P-256 slice's doubling formula was correct only when `Z == 1`, and it passed every comparison in the
  suite because the comparisons were between the ladder and the group law *invoked with the same
  operand sequence*: `assert_eq!(g.double().double(), ladder(4))` is `assert_eq!(x, x)`. It surfaced as
  arithmetic instead — 7G came back off the curve — and the repair was to pin values from an
  independent implementation (2G…7G x coordinates from the Python in `scripts/tls_p256_params.py`).
  Before writing an agreement test, ask what each side computes.
- **A control that passes is a control that was never tested.** The fixture slice's "a certificate
  verified with the wrong identity fails" check returned `Ok(())`, and the cause was the *fixture*, not
  the verifier: the generator had built `SELF_SIGNED_PSS` with `root_key`, so that certificate really was
  signed by the key it was being checked against. Two habits follow. When a negative case passes, suspect
  the fixture before the code — **two fixtures that share a key cannot tell verification from a key
  lookup**, and every identity gets its own seed for that reason. And do not pin *which* rule fires when
  the input is garbage: a wrong key's RSA operation produces an effectively random block, so the refusal
  can come from any of four checks, and pinning one pins an accident of that key.
- **When a bug cancels against another bug, the test that should have caught it can pass harder.** Two
  Montgomery-domain mistakes in the same file (`ONE` as the limbs `1`, and an inverse returning a plain
  value into a world where every value is Montgomery) made `inverse * value == ONE` hold while the
  curve arithmetic was wrong. Assert the *domain* directly — `ONE` has to read back as the bytes `1`,
  and a value has to survive a round trip through its Montgomery form — because no test that multiplies
  an inverse back can see which domain it is in.
- **A claim in a doc comment is a claim, and gets checked the same way.** The same slice's paragraph
  here said the fixtures "regenerate byte-identically"; two of the six do not, because RSA-PSS salts
  randomly and ECDSA nonces randomly. Nothing in the test suite could catch that — the tests passed
  either way — and only regenerating to a scratch path and diffing exposed it. Before writing that a
  file reproduces, a check passes, or a property holds, run it once and look.
- **Suspect your own harness before reporting someone else's data as broken.** Three times now a
  "finding" was my own bug: a helper that wrote CRLF made curl report HTTP 000 on 14 valid URLs, a
  Python `allow_reuse_address` server silently answered a later test on the same port on Windows, and
  in the handshake slice a ClientHello slice that still carried its 4-byte handshake header shifted my
  session-id offsets by four bytes. In the X.509 slice the same reflex caught three more: an assumed
  TBS offset of zero, an assumed signature length following the subject's key instead of the issuer's,
  and a `keyUsage` bit order read from bit 0 instead of the high bit.
- Test fixtures must be *correct*, not just the right length: a 50-byte prefix of `b"\x00"` is not a
  prefix of `bytes(range(100))`, and that mistake made a valid resume look like a content mismatch.
  Prefer *generated* fixtures (``scripts/tls_x509_fixtures.py`` writes the Rust array from seeded
  keys) over transcribed ones, and parse the expectations back out of the artifact instead of typing
  them.
