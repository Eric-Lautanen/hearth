# Hearth — next session

> **Session 2026-10-16 (pt 18) in one paragraph.** The Q2_0 dispatch was preferring the *slower*
> kernel — the fourth instance in a day of the same cause: a body whose `#[target_feature]` list was
> incomplete when its dispatch order was decided. `dot_q2_0_q8_0_ptr` led with the 512-bit VNNI arm
> on the strength of a `benchk` table taken *before* the AVX2 LUT kernel's declarations were fixed
> (256 → 167 instructions in the change log). Re-measured on real weights at the v3 baseline we ship,
> `avx2_lut` reads **2.56–2.57 GB/s against 1.74–1.76** for the VNNI pair, on every tensor of two
> models, with the two arms that are the same function agreeing within 1%; the 512-bit arm went back
> behind `crate::avx512_full_width` (the gate PTQ1_0 already uses), so a full-width AVX-512 part is
> the one case this measurement does not cover. End to end that is **+31% decode and −4 to −7%
> prefill on all three dense Q2_0 models**, with the Q1_0 rows as an unchanged control. The *count*
> half of the change was wrong on the first attempt and the mirror caught it: the dense prefill is a
> **batched** path (`matmul_batch` → `batched_rows_chunk`, plus `flash_attention_prefill`) and it is
> *unreliable* once the workers outnumber the physical cores — eight runs alternating 8 and 9 workers
> on dense 8B Q2_0, 109-token prompt, gave a **2.4x spread at 9 on identical invocations** (8320 to
> 19933 ms) against a tight 8 (9011/8973/9158), and the boundary moves with the model (4B Q2_0 is
> still fine at 9 and falls at 10). So the dense rule stays at `physical`, while the hybrids — whose
> prefill is token-by-token, i.e. the *stable* single-token path — keep the counts their kernels want:
> `min(logical, physical + physical/4)` = 10 for `Q2_0`/`PQ2_0` and ¾-of-logical = 12 for `PTQ1_0`.
> The single-token path *is* stable above physical and does want more (dense 4B Q2_0 decode 13.4
> tok/s at 10 against 11.7 at 8), so **a count tuned on decode alone is not tuned** — that is the
> finding to carry. Also: the `kernel-baseline` CI job now builds the *test* target too, because a
> `#[cfg(test)]` kernel is one the library build never sees — which is exactly where the Q1_0 VNNI
> kernel hid.

> **Session 2026-10-16 (pt 17) in one paragraph.** Two things landed, both found by measuring rather
> than by reading. (1) The dispatch wait no longer spins through the whole wait:
> `ThreadPool::wait_for_workers` spins for `SPIN_BUDGET` and then yields, worth +15–25% wherever the
> pool outnumbers the physical cores (dense 8B Q1_0 12.1 → 13.2 tok/s at 8 workers and 6.0 → 6.9 at
> 12; hybrid `Ternary-Bonsai-2-4B-PQ2_0` 5.2 → 6.4 at 8) and neutral at the default. That **closes**
> the carry-over item that wanted the dispatching thread to take a chunk instead of waiting: the
> remaining cliff is memory bandwidth and SMT sharing, not a starved worker, so the per-path counts
> stand. *(pt 18: the hybrid's did — only its `Q2_0`/`PQ2_0` count moved to the quarter; the dense
> count stayed at physical because the batched prefill is unstable above it.)* (2) **aarch64 became a gated platform, and it had never been built.** One
> `cargo check --target aarch64-unknown-linux-gnu --workspace --all-targets` showed `hearth-tls` did
> not compile there at all — AES-NI and PCLMUL calls reached from branches gated only on the runtime
> probe, naming modules that exist only on x86_64 — and `hearth-quant` had six dead-code warnings
> plus two more in its test build. `macos-latest` is an arm64 runner now, so that same break had been
> failing `check (macos-latest)` on `main` since the TLS crate landed. New CI: `arm-cross-check`
> (three aarch64 targets, check + clippy `-D warnings`, no linker needed) and `arm-qemu-test` (the
> CPU-only suites linked and *run* under `qemu-user` — the first execution of the aarch64 scalar
> fallback anywhere, `pool::tests::physical_cores_is_always_a_usable_count` included). Run
> `38988216344`: 8/8 green. Commits `5fd87ed` (pool), `7a918ad` (aarch64). (3) **The Q1_0 VNNI kernel
> had never been dispatched, for the same reason as (1) of the previous session.** `q1_0g128`'s
> 256-bit VNNI kernel was `#[cfg(test)]` behind a "~33% regression on Zen 4" note that was measured at
> a `native` baseline, where the intrinsic still inlined; at the v3 baseline we ship it was missing
> `avx512vl` and `fma`, so it measured +180% and nobody re-measured. With the declarations fixed it is
> 4–6.6% faster than shuffle in L1, +6% per tensor on real weights (`benchk`), and +4.5% / +2.7% /
> +3.6% end to end on the 1.7B / 4B / 8B Q1_0. Dispatched, with the guard now covering it.

> **Session 2026-10-17 (pt 19) in one paragraph — a negative result, and the experiment that keeps the
> dense count honest.** The pt-18 entry left one question: *why* the dense batched prefill is unstable
> above the physical count when the single-token path is stable *and* faster there. The obvious suspect
> was the dispatcher's own wait (`wait_for_workers` spins `SPIN_BUDGET` then `yield_now`-loops, and
> `yield_now` only offers the core to a thread ready on the same *logical* processor). Made
> per-dispatch — a batched dispatch blocks on a condvar the workers notify under the same lock, a
> single-token dispatch still yields — the park read 8582/8605/8720/8617 ms at 9 against
> 12999/8320/13256/19933/8439 for yield, but *those two columns are different windows* (pt 20 corrects
> this): the only same-window pair is at 8, where the park buys nothing (9281 against 9357). Either way
> it is **reverted** — parking *unconditionally* costs single-token decode 7%, so it had to be
> conditioned on `seq_len > 1` at all, and the count stays `physical` with the wait on yield. What the
> sweep did pin down is the other half: count 9 is a repeatable **+2.8 to +6.6% decode** on all six
> dense models
> (count-isolated, same binary: 1.7B Q1_0 48.84 → 50.24, 4B Q1_0 22.01 → 22.13, 4B Q2_0 12.58 → 13.02,
> 8B Q1_0 11.72 → 12.45, 8B Q2_0 5.83 → 6.22), and the *only* thing blocking it as a default is the
> batched prefill's tail. Two probes then killed the obvious suspects: it is not bandwidth (a
> dispatch streams ~2.3 GB of weights in ~9.2 s ≈ 250 MB/s) and not DVFS (the perf counter reads mean
> 108% at 8 workers against 103-105% at 9, boost peaks 123-125% in both, 36.8 C throughout). And pt
> 18's "2.4x spread at 9" was then read as mostly the box, because a quiet machine had 9 beating 8 at
> *every* prompt length measured (19/55/109 tokens: 1977→1831, 5039→4700, 9396→8952 ms) — but pt 20's
> controlled sweeps retired that reading too (one sample per point is a *mode*, not the whole story):
> same window, count 9's 109-token prefill is 8528/8599/8679/**15115**/8296 ms against
> 9062/9134/9217/9008/8991 for count 8, i.e. a ~7% better mode and a 1.7x worse tail. So the dense
> rule is **robustness, not speed**, and the tail is what a laptop default is judged on.

> **Session 2026-10-17 (pt 20) in one paragraph — count 9's prefill is bimodal, and the mechanism is
> the static split.** pt 19 left the count question as "9 is better quiet and worse busy", with its
> park experiment reverted and pt 18's 2.4x outlier blamed on the box. Both readings are corrected by
> the controlled sweep. First the record: the park table's 9-worker row compared the park window
> against pt 18's window, so the park's effect at 9 was never measured against the shipped arm at all —
> the only same-window pair is at 8, where it bought nothing (9281 against 9357 ms), and "the park
> reduced the spread" was a cross-window artefact of exactly the kind this exercise exists to catch.
> Second the finding: `scripts/count-sweep.py` (new — it sweeps a count on both passes, on a quiet box
> *and* under a controlled busy load, rotates the run order, prints a per-count mean **and** spread,
> and flags a count whose spread exceeds a threshold) measured dense 8B Q2_0's 109-token prefill at
> counts 8 and 9, five runs each, same window, no added load: count 8 read 9062/9134/9217/9008/8991
> (spread 2.5%) and count 9 read 8528/8599/8679/**15115**/8296 (spread 69.3%). So count 9 is
> **bimodal**, not erratic: four runs in five it is the ~7% faster arm pt 19's quiet sweep saw (clean
> state ~8400 ms, 0.93x of count 8), and the fifth is 1.7x slower than its own clean state. A second
> quiet window read the same shape (60.4% against 1.3%), and pt 18's window read the tail end of it —
> the "cliff" was the tail and the quiet sweep was the mode, both true samples of one distribution. The
> mechanism is in `pool.rs`'s *static* split (`chunk = n.div_ceil(nt)`, worker `i` takes
> `i*chunk..(i+1)*chunk`): 9 workers over 109 rows hand the ninth 5 rows against 13 for the other
> eight, and 9 threads on 8 physical cores must leave one core holding two of them — two long chunks
> sharing a core cost the whole dispatch (a dispatch cannot finish before its last worker), while a
> long chunk beside the 5-row one ends early. The 6-thread load arm confirms the oversubscription
> (everything doubles: 14 runnable threads on 8 cores) but is *not* what produces the tail — it moved
> the tail around instead (4B Q2_0 loaded prefill 13795/13959 ms for 8/9; 8B Q2_0 loaded prefill
> 27070/26395, count 9 ahead) — so "the box did it" is retired as well. The dense count stays
> `physical` for a measured reason now: its spread is 1-3%, count 9's mode is 5-7% better and its tail
> 1.7-2.2x worse, and a default has to be judged on the tail. **A count is chosen on its spread, not
> its mean** — and the tool that says so is in the tree (`scripts/count-sweep.py`).

## Do these first (carry-over from pt 19/20)

1. **aarch64 runs, and it is slow by construction: there is no SIMD backend for it.** Every
   dispatched kernel there is the scalar fallback, and `arm-qemu-test` is the only thing that has
   ever executed that path. So NEON is the largest Goal-1 item left in `hearth-quant` — but *measure
   first*: qemu is not a benchmark, so a real ARM target is the prerequisite for any number
   (`macos-latest` is arm64 and already in the matrix; `cargo test` there is free, `cargo bench`-like
   timing is not). The two shapes that suit NEON best are the `Q2V_U8` LUT dot — `vqtbl` is a table
   lookup in one instruction, which is exactly what that kernel is — and the f32 glue in
   `hearth-llm/src/ops.rs`.
2. **The worker count follows the *path* — and, on the hybrid, the kernel — and pt 20 measured why the
   dense one is the odd one out.** As shipped: dense = `physical` (8 here); hybrid `Q2_0`/`PQ2_0` =
   `min(logical, physical + physical/4)` (10, measured 8.5 tok/s decode and 13346 ms prefill against
   7.9 / 13940 at 8); hybrid `PTQ1_0` = ¾-of-logical (12, 6.7 tok/s against 5.9 at 8). The dense rule
   is not about its kernel: it is the **batched** prefill, whose spread at 9 workers is bimodal — the
   measurement is the first bullet below and the mechanism is the second. `HEARTH_THREADS` overrides
   everything.
   - **The dense rule is about the batched prefill's *spread*, and that spread is bimodal.** Over 109
     batched rows, count 9 reads 8528/8599/8679/**15115**/8296 ms against 9062/9134/9217/9008/8991 for
     count 8 (same window, no added load): its mode is ~7% *faster* than 8 and its tail is 1.7x slower,
     while count 8's spread stays 1-3%. A default is judged on the tail, so the count stays `physical`
     and count 9 stays a `HEARTH_THREADS=9` opt-in — the dense *decode* half (+2.8-6.6%, item 3's count
     axis) does not pay for that tail. The protocol lesson still holds — **any kernel change owes a
     count sweep on *both* passes, and any count change owes an interleaved pair on both passes** — and
     pt 19 added that the dispatcher's wait is *not* the cause (the park is not in the tree).
   - **The mechanism is `pool.rs`'s static split, and it names a candidate change.** A batch is divided
     as `chunk = n.div_ceil(nt)`, worker `i` taking `i*chunk..(i+1)*chunk`, so 9 workers over 109 rows
     hand the ninth worker 5 rows against 13 for the other eight — an unbalanced split *and* 9 threads
     on 8 physical cores, i.e. one core always holding two workers. Two long chunks sharing a core cost
     the whole dispatch (it cannot finish before its last worker); a long chunk beside the 5-row one
     ends early. **Untested hypothesis for the next session:** an atomic work-stealing split for the
     batched path (`batched_rows_chunk` taking a `fetch_add`ed row range) removes the imbalance and
     bounds the sharing penalty to the final row, which should turn count 9's tail from 1.7x into
     something small — and if the spread does *not* tighten, that falsifies the mechanism above.
     Measure it the protocol's way (mirrored, both passes, counts 8/9/10, with
     `scripts/count-sweep.py`); do not ship a split change on the reasoning alone.
3. **The `benchk`-driven per-CPU calibration runner is still the unfinished half of Goal 1 — but both
   of its axes now have tools.** It is not just "pick the winning kernel per dtype": the worker count
   is one decision *per path* (item 2), so the record has to be a (kernel, count, path) triple, and it
   has to *refuse* to switch on a difference inside the noise band. `benchk` already prints every arm's
   GB/s and the CPUID decode, with a 300 ms warm-up per arm and a minimum of two passes; what it does
   not do is persist a winner, sweep the count, or refuse.
   - **Done — `scripts/baseline-invariance.py`** (the kernel axis): `NAME=TARGET_DIR` pairs, `benchk`
     twice per baseline, *arm drift* and *dispatch drift* reported separately, validated by catching
     both on the pre-change artifacts in `target/buffy-ab/`.
   - **Done — `scripts/count-sweep.py`** (the count axis, new in pt 20): a count on both passes, on a
     quiet box *and* under a controlled busy load, run order rotated, a per-count mean **and** spread,
     a fragility flag, and an explicit "inside the noise, no switch justified". It is what established
     item 2, and it carries the rule that matters: **judge a count on its spread**, because the loaded
     arm alone is not enough — it confirms the oversubscription mechanism (everything doubles) without
     being a sharper finder of the tail than the quiet window (4B Q2_0 loaded prefill 13795/13959 ms
     for 8/9 against 4656/4467 quiet; 8B Q2_0 loaded prefill read count 9 *ahead* at 27070/26395).
   - **What remains:** persisting the winner as a (kernel, count, path) triple with the refusal rule,
     and folding the count sweep into `benchk`'s report instead of leaving it a separate script.
   The hazard the harness exists to prevent is real and was misread once: a v3 run that ranked
   `avx2_lut` 2.51 ahead of `vnni512x2` 1.70 was dismissed as clock ramp-up because a `native` rerun
   reversed it — but the v3 reading was right, and `native` was measuring different codegen for the
   same source. **A baseline is part of the measurement; only compare within one** — and pt 20's
   correction of its own park table says the same about a *window*.
4. **Closed in pt 18: the Q2_0 rows were not noise — the dispatch was choosing the slower kernel.**
   With the kernel fixed and the count back at physical they read **+31% decode and −4 to −7% prefill**
   on all three dense Q2_0 models (mirrored, with the Q1_0 rows as an unchanged control). The "count
   alone bought +45%+" reading was wrong and is retired: that measurement used a short prompt, i.e.
   decode only, and the same count pays up to 2.4x on a bad prefill run (item 2 — 1.7x in a
   controlled window). The lesson to keep: three sweeps of
   the same regression looked like drift because the explanation ("machine state plus the thread
   count") was plausible, and the measurement that would have falsified it — the kernel arms
   themselves, on real weights — was in `benchk` the whole time.
5. **Goal 2: the code side is closed; the screenshots are blocked on a visible window.** The
   downloader is implemented and verified live (`GET /api/downloads` returns the catalog with per-model
   `state`/`downloaded_bytes` and HuggingFace URLs; `/api/downloads/start` and `/cancel` queue one file
   at a time; the drawer in `web/index.html` has checkboxes, per-item progress, resume and an add-a-URL
   box). Done in pt 20:
   - `resources/hearth.svg` is centred, and *measured* centred: the flame path's bbox is
     (5.5,2)-(18.5,21.7), so the transform is now `translate(32,32) scale(2) translate(-12,-11.85)` and
     the file's comment records that bbox. Rendered as a document, `getBoundingClientRect` reads
     padTop = padBottom = 90.78 px and padLeft = padRight = 164.06 px; the old `translate(8,6)` read
     65.6 against 116 px, a 50 px asymmetry. Verify any later tweak the same way, not by eye.
   - `/favicon.svg` serves `resources/hearth.svg` from the binary (`web::FAVICON_SVG`), linked from
     `index.html` with `type="image/svg+xml"`, so the browser tab, the desktop entry and the macOS
     bundle share one icon and cannot drift.
   - the README's 14-model table is re-measured (warm-up plus two runs, mean shown: 4B Q1_0 18.8 → 23.6,
     8B Q1_0 12.2 → 13.0, the rest inside the documented spread) and the env table is corrected
     (`HEARTH_THREADS` overrides *both* paths; dense = physical cores, not a fixed 8; `HEARTH_PIN_WEIGHTS`
     was missing and is added).
   **Left, and blocked:** both README screenshots predate the 2026-10-16 UI commits. `browser.screenshot`
   refuses without a *visible* desktop window and no browser tool can restore one — bring the browser
   window to the front, then re-shoot `/` (chat) and the downloads drawer (about 1600x1000) over
   `docs/screenshots/hearth-{chat,downloads}.png`.
6. **GPU scaffolding (Goal 1, still not started):** opt-in behind `--device`/env, graceful when no
   ROCm device exists, and it must not break the CPU path or the Windows/macOS builds. The aarch64
   cross-check now guards the last of those three for the ARM half.
7. **A/B protocol — mirror the order.** On this box the second run in a pair is worth ~2–3%, so plain
   `before, after` pairs invent a win: today's first pass read +6.5% / +3.7% / +3.9% and the
   order-mirrored re-run read +4.5% / +2.7% / +3.6%. Use `b,a,a,b` and `a,b,b,a` groups, compare
   **group means**, and never compare across windows — the absolute level drifts ~10% over an hour.
   The CLI's 50-token run has a ±5% spread on its own; one transient group in the 4B read 18.0 tok/s
   where every other group read 21–22. **A *loaded* row is a different measurement, not a faster one**:
   `scripts/count-sweep.py --load N` is deliberately contended (never run it beside real work, and never
   compare its rows to quiet ones) — it answers "does this count keep its shape when the box is busy",
   which is the question a laptop default has to answer.

## Earlier sessions — the history lives in `docs/BUG_TRACKER.md`

- **pt 16** — CPU baseline → `x86-64-v3`; the AVX2 kernels made baseline-invariant by declaring
  `fma`/`f16c`; the VNNI `avx512vl` fix; the per-path worker counts. *Tracker: "Baseline is
  x86-64-v3…".*
- **pt 15** — the allow-block sweep: zero `#[allow]`/`#[expect]` left anywhere, 25 sites fixed at
  the cause.
- **pt 14** — `PinnedVerifier`: the first thing in `hearth-tls` that can *accept*. *Tracker: slice 20.*
- **pt 13** — CRLs, `Revocation::{Unchecked, Required}`, and the CRL `version` field two readers had
  wrong together. *slice 19.*
- **pt 12** — AKI/SKI matching, a link check no signature can make. *slice 18.*
- **pt 11** — key rollover and the self-issued `pathLenConstraint` count. *slice 17.*
- **pt 10** — what the OpenSSL oracle says about chain *shape*. *slice 16.*
- **pt 9** — the oracle itself: fifteen paths put to `openssl verify`, verdicts recorded. *slice 15.*

## hearth-tls lane — next session, in this order

1. **Certificate verification — the decision exists; its *inputs* and its wiring are the work.**
   A certificate parses (`x509.rs`), every suite verifies (`rsa.rs`, `ecdsa.rs`), `verify.rs` decides
   which scheme may verify which key, `chain.rs` builds the path from the wire bytes to a
   caller-supplied anchor with the structural rules, the validity window and revocation, `crl.rs`
   reads the revocation lists, `leaf.rs` checks that the certificate names the server and may be used
   as one, and **`verifier.rs`'s `PinnedVerifier` calls all of them in order** — so there is now a
   `CertificateVerifier` that can accept, not just refuse. What remains is (a) where the anchors come
   from (below), (b) supplying CRLs or writing `Revocation::Unchecked` at the call site, and (c) the
   transport that would use either — nothing in this crate has ever opened a socket. There is **no
   OCSP, no stapling and no CRL fetching**, and there must be no accept-all type, flag or environment
   variable on any path a URL can reach.
   - **Closed in pt 11: the self-issued branch is now reachable.** `ROLLOVER_CA_A` / `ROLLOVER_CA_B` (the
     same DN, a new key) / `LEAF_OF_ROLLOVER_CA_B` make a key rollover a path, legal only because the
     self-issued certificate does not spend its parent's allowance of none; neutering `self_issued` turns it
     into `PathLengthExceeded`. Those three are also the only fixtures carrying AKI/SKI, because without
     them OpenSSL cannot tell two same-DN certificates apart and refuses sound rollover paths.
   - **The lesson from that slice, which applies to every remaining rule:** a rule whose branch no fixture
     can reach is a rule nobody has checked, and it looks exactly like one that works. When adding a rule,
     add the fixture that makes it *refuse* something in the same change — and prefer one where the rule is
     the only thing standing between the path and `Ok`.
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
        1024-bit key because RFC 8448's fixture is one, so that policy lives here and not there. **Every
        structural rule is now in it**, including `pathLenConstraint`, `keyUsage.keyCertSign` (see
        item 4) and revocation (see item 5). Also absent, and deliberately: `extendedKeyUsage` on
        issuers, and any policy processing (`policyConstraints`, `inhibitAnyPolicy`, name constraints).
     3. **Host name matching — done, in `leaf.rs`.** `subjectAltName` is the only place a name is looked
        for (no `commonName` fallback — RFC 6125 §6.4.4 deprecated it, and `LEAF_EC` is the fixture that
        refuses its own subject's name), only `dNSName` entries are compared, comparison is ASCII
        case-insensitive with one trailing dot ignored, a wildcard is one whole leftmost label or
        nothing at all (and must leave two labels, so `*.com` matches nothing), and an IP literal is
        refused rather than matched. The rule is a pure function — `leaf::dns_name_matches` — so it is
        unit-tested exhaustively without a certificate.
     4. **The validity window — done, in `chain.rs`**, checked on every certificate in the path and on
        the anchor, inclusively at both ends, against the `UnixTime` the handshake already carries. The
        leaf's purpose is also done (`leaf::verify_server_auth`): `extendedKeyUsage` including
        `serverAuth` when present, and `keyUsage` including `digitalSignature` when present, each with a
        fixture that makes it fail. `pathLenConstraint` and `keyUsage.keyCertSign` are done too, on both
        sides — a CA inside the chain and the anchor outside it — and the anchor-side limit is proven
        with the anchor excluded from its own count. The anchor's own window and key size are tested too
        (`EXPIRED_ANCHOR`, `WEAK_ANCHOR`), so **no call site in `chain.rs` is without a negative test any
        more** — the first time that has been true of this file. The anchor's `basicConstraints` is the
        third rule read out of a CA certificate: stated `cA: FALSE` is refused when the anchor verifies
        someone else's certificate, an absent extension is allowed, and a certificate pinned as *itself*
        is an identity pin and allowed (`check_anchor_is_ca`).
     5. **Revocation — done in pt 13, in `crl.rs` and `chain.rs`.** `Revocation::Required` with CRLs the
        caller supplies: every certificate in the path needs a list from its issuer, the issuer must
        claim `cRLSign`, every supplied list for that issuer must verify its signature, the freshest
        `thisUpdate` speaks, and equally fresh lists that contradict each other are refused. Every rule
        was proven by removing it (13 rounds). The open questions are **not** in this crate: nothing
        fetches a CRL, and the design that matters next is the caller's — where CRLs come from, when they
        are refreshed, and what happens when the refresh fails. The failure to design *out* is the one
        where a refresh fails quietly and the stale list keeps answering.
   - **The oracles are in place — extend them instead of trusting prose.** `VERIFY_CASES` holds
     twenty-three chain paths and `CRL_CASES` seventeen revocation paths, each with the OpenSSL CLI's verdicts
     recorded; the tests compare verdicts at the same instant and permit a disagreement only where
     OpenSSL accepts and this crate refuses. New paths go in `verify_cases()` / `crl_cases()` in
     `scripts/tls_x509_fixtures.py`, then regenerate. Three things they exposed and do *not* model:
     `openssl verify -x509_strict` refuses every fixture with error 85 ("Missing Authority Key
     Identifier"), so the stricter extension rules behind it are not implemented here; the five recorded
     chain divergences (`CaUsedAsLeaf`, `Policy::min_rsa_bits`, and the three about chain *shape*) and the
     two revocation ones (a delta CRL, and `-crl_check`'s leaf-only scope) are this crate's own choices,
     ours to keep or to reconsider rather than accidents; and `openssl verify` needs its option arguments
     to *precede* the certificate path, or every case comes back error 20 and looks like a verdict.
   - **The root store is the caller's decision, and the crate is built that way.** `client.rs` already
     says this crate has no root store and no business having one: [`chain::verify_chain`] takes the
     anchors as an argument. So the question moves to the downloader — a pinned set in `hearth-ui` with
     fingerprints and a documented update path, versus a per-platform OS trust reader that `std` cannot
     give on all three platforms — and either way its contents have to be reviewable in one screen and
     its failures loud. **Do not put a root store in `hearth-tls`.**
   - The negative set is now complete: the chain-level **bad signature, malformed encoding, expired,
     not-yet-valid, unknown root, name mismatch, CA used as a leaf, non-CA issuer, broken link, chain too
     long, RSA key too small, scheme that does not match the key type, framing failures, a purpose the
     leaf is not allowed, `pathLenConstraint` on both sides, a CA that does not claim `keyCertSign`, the
     anchor's own window and key size, and an anchor that states `cA: FALSE`** all have tests.
   - **Closed in pt 14: the verifier exists, and it is where a new check belongs.**
     `PinnedVerifier::new(anchors, revocation, policy)` + `CertificateVerifier`; add a rule inside
     `verify()` and it is covered by the three fixture tests that drive it end to end, or by nothing —
     the six neutering rounds in `target/buffy-ab/neuter_verifier.py` are what show which. Two things
     about it are decided and should not be revisited casually: the anchors are **an argument** (no
     root store in this crate — see the downloader note below), and an empty anchor set is a valid
     verifier that refuses everything, not an error to raise at construction. The `CertificateVerify`
     fixtures (`CV_VECTORS`, `CV_SIGNED_DATA_HEX`) live in the generated table and are re-verified by
     the OpenSSL CLI on every regeneration; PSS and ECDSA signatures change every run, so a test may
     only assert that a vector *verifies*, never that it matches pinned bytes.
   - **What the next slice should be, in order of proof:** (a) an anchor source in `hearth-ui` — a
     pinned set with fingerprints and a documented update path is reviewable in one screen and needs
     no platform API, which is why it comes before an OS trust reader; (b) the downloader's `Transport`
     impl driving `ClientHandshake` with `PinnedVerifier`, which is the first time any of this meets a
     socket; (c) the CRL refresh question from pt 13 — what happens when a refresh fails must not be
     "the stale list keeps answering".
   - **Closed in pt 13: revocation exists, as CRLs the caller supplies.** `crl.rs` reads them and
     `Revocation::{Unchecked, Required}` decides what is checked; the file's doc comments state what each
     refusal means and which rules are copied from OpenSSL versus this crate's own. What is still absent
     is **OCSP, stapling, and fetching**: a CRL is bytes the caller brings, so keeping one fresh — and
     deciding what to do when a refresh fails — belongs to the downloader, and the failure mode to design
     *out* is the one where the refresh fails quietly and the stale list keeps answering. `Unchecked` also
     needs a policy decision at the call site: a caller that never supplies CRLs gets no revocation
     checking at all, and the word at the call site is the only thing that makes that visible.
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
6. **Feature-gate sweep — closed in quant, still open elsewhere.** *Done:* every
   `is_x86_feature_detected!` / `target_feature` / `avx512` site in `hearth-quant` was read, and the
   one real defect is fixed — `q2_0`'s dispatch *and* its `kernel_name()` wrapped the AVX-512 arms
   in `#[cfg(target_feature = …)]`, making a kernel's existence a property of the build machine; both
   now share one cached runtime probe, and a negative-target-feature build
   (`-C target-feature=-avx512f,-avx512vnni,-avx512dq`) is the regression test. `ptq1_0` and
   `q1_0g128` were already runtime-dispatched (`q1_0g128` deliberately has no AVX-512 path: ~33%
   regression, measured, kept as a test-only kernel).
   *Closed the same day* by the workspace allow-block sweep (pt 15): all 25 remaining sites are gone,
   `needless_range_loop` fix included, and the whole-repo gate is green. `hearth-llm`'s audit found
   the same class of thing the quant sweep did — a scalar reference dot nobody called, VAE scaffolding
   nothing reached, a diffusion constructor uploading weights no path read.
   *Still open, and now the biggest number in this file:* (b) what the release build promises.
   `.cargo/config.toml` is now **`x86-64-v2`** (SparkOC, 26eab4b) after auditing that every
   AVX2/AVX-512 path sits behind a runtime probe, so only autovectorised code is affected. **Measured
   the same day: that baseline cost 3.4× on Bonsai-1.7B-Q1_0** (43.4 → 12.8 tok/s mean of three
   interleaved rounds; the old native binary read 44.9 with byte-identical output). The commit claims
   1-3%; on the flagship 1.7B pair it was not — and the audit behind the commit was right about what
   it checked (the *dispatch* is runtime-probed everywhere) and wrong about what that implies, because
   a kernel can be reached correctly and still run slowly. That distinction is the whole entry below.
   **Solved at the cause, and the flag does not need to change — closed 2026-10-16.** The loss was
   never the baseline: it was kernels that used FMA and F16C without declaring them, so at any
   baseline lacking those features every `_mm256_fmadd_ps` and every f16 scale conversion became an
   out-of-line call (511 instructions per row against 100, 16 `vzeroupper`s per row). Declaring
   `fma`/`f16c` on the kernels and converting f16 scales with a declared-`f16c`
   `crate::f16_bits_to_f32` (instead of `half`'s runtime-probed `to_f32`, which inlines both of its
   arms into the block loop) took the *unchanged* `x86-64-v2` baseline from **13.6–13.9 to 43.0–43.5
   tok/s on Bonsai-1.7B-Q1_0** (v3 at the same source: 45.0–45.4; native: 50.6–51.1), with the same
   generated text in every arm. Below the kernel level the thing to read is the assembly, not the
   clock: 511 instructions per row with 15 out-of-line calls became 112 with none. The remaining
   `v2`-vs-`v3` difference is a preference about minimum CPU (SSE4.2 2011 vs Haswell 2013), not a
   regression — the 3.4× is gone at either baseline. Full tables and the assembly evidence are in
   BUG_TRACKER under "the `x86-64-v2` baseline cost 3.4×". **SparkOC moved `.cargo/config.toml` to
   `x86-64-v3` at 19:54 today on the strength of this entry's earlier reading** — it is their file and
   either baseline is now defensible, but read that entry before treating v3 as *required*: it would
   have hidden this defect rather than fixed it, and the 4B/Q2_0 pairs could not be separated from
   box noise today (the same binary read 10.67 and 12.70 tok/s back to back).
   **Next session, in this order:** (1) re-run the 6-model sweep on a *quiet* box and put the numbers
   back in the tables — today's box was busy (two agents benchmarking; one binary spanned 2× across
   consecutive runs), so the 4B and Q2_0 pairs are still unmeasured, and the rows recorded before
   today are native while the committed baseline is now v3; (2) hold the rule that
   made this findable: **a kernel's `#[target_feature]` list must name every feature its body
   executes, and the dispatch gate must name the same set** — the three per-file predicates
   (`q1_0g128::avx2_kernel_available`, `q2_0::avx2_lut_available`, `ptq1_0::avx2_available`) and the
   feature-report test are where that rule now lives, and any new format or kernel arm has to join
   them; (3) audit what *else* in the workspace trusts the build's baseline for more than SSE4.2
   (hearth-llm's `wide`-based ops are the obvious place to look) — the stage timings say the f32 glue
   is fine at v2 today, but that is the same reasoning that missed this bug, so check the emitted
   assembly rather than the timing when in doubt. `cargo rustc -p hearth-quant --release --
   --emit=asm` plus a scan for `callq` to `*fmadd*`/`*f16_to_f32*`/`*dpbusd*` inside an annotated
   kernel is the whole method: three scripts' worth of it is in `target/buffy-ab/`
   (`cmp_asm.py`, `scan2.py`, `rewrite_f16.py`).
   *Harness notes:* `cargo run --release -p hearth-llm --example benchk -- <model.gguf> <dtype>`
   prints the CPUID decode and every variant's GB/s; **`benchk … | head` exits 101** (SIGPIPE) so
   redirect to a file or use `tail` when the exit status is the verdict; and **cold runs are ~2×
   under warm ones** (today: dispatched 1.62 cold against 3.09 recorded warm), so always warm up
   before comparing against the numbers in the comments.
7. **Zen 5 (family 1Ah) and Intel must not regress.** The fix keys on 19h specifically so they keep
   full-width AVX-512, and there is no such hardware here to confirm it.
8. **Keep dispatch introspection honest for each new format.** `selected_kernel(dtype)` +
   `dispatch_report()` cover PTQ1_0, PQ2_0, Q2_0, Q1_0 and Q1_0_G128; the `[kernel]` banner prints at
   load on both paths and the service exposes `event.model.kernels`. A new quant type must add a
   `kernel_name()` arm and a `selected_kernel` arm or it will silently have no reported dispatch.
   Agreed rule: add a format to `selected_kernel` only when it has more than one implementation to
   choose between.
9. **Three small things.** (a) *Done (pt 15):* AGENTS.md's verifications block said
   `cargo test --test-threads=1 -p hearth-quant`, which cargo rejects; both occurrences now read
   `cargo test -p hearth-quant -- --test-threads=1`, and the same file's "pre-existing clippy warnings
   in GPU stubs are OK" line is gone, since there are no accepted warnings left to point at.
   (b) Re-run the benchmark mandate on a fully idle box interleaved with the reference fork
   (`bench_results/run_cpu_bench.bat`). (c) *Done (pt 15):* the `gpu_f32_bufs` warning is closed by
   deleting the field and the upload behind it; the diffusion GPU path can re-add it the day it
   consumes those buffers.
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
