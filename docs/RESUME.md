# Hearth — next session

> **Session 2026-10-16 (pt 15) in one paragraph.** The allow-block rule is now a property of the
> workspace rather than a job in progress: **zero `#[allow(...)]` and zero `#[expect(...)]` anywhere**,
> nothing silenced to get there, and the whole-repo gate green — `cargo clippy --workspace
> --all-targets -- -D warnings`, `cargo fmt --check`, `cargo check --workspace --all-targets`
> warning-free, `cargo test --workspace -- --test-threads=1` green (42 quant, 170 tls, rest), release
> build clean. The 25 sites the quant sweep left were all real: 19 `too_many_arguments` (15 of them
> `hearth-compute` dispatch methods, now taking named `Copy` shape structs in `args.rs` while the
> buffers stay explicit; `ensure_batch_size` went from seven shape arguments to two, because the rest
> were `self.config`; `matmul_batch`'s row count was never read and is deleted), 4 `needless_range_loop`
> (both cache batches now iterate `caches.iter_mut().enumerate()`), `GgmlDType` variants camel-cased
> with `name()` still returning the spec spellings so no log line moved, and dead weight deleted rather
> than allowed — the never-wired FLUX VAE scaffolding, `GpuDevice`'s instance/adapter handles, and the
> diffusion constructor's upload of every BF16/F32 weight as f32 that no forward path read. My own
> scripted pass was off by one and deleted a row-buffer argument instead of the dead row count; one
> build caught it, and the repair (and both exact deleted-line counts) is in the tracker. The AGENTS.md
> section now states the rule — no allows ever, never a "pre-existing" warning, a finding that cannot be
> fixed without changing behaviour is reported and measured — and its two stale claims (`cargo test
> --test-threads=1 -p …`, "GPU stub warnings are OK") are corrected. Diffusion's VAE is still the
> placeholder path; the real decoder is the next feature there.

> **Session 2026-10-16 (pt 14) in one paragraph.** The crate now has a verifier, which is the first
> thing in it that can *accept* anything: `verifier.rs`'s `PinnedVerifier` runs the three stages in a
> stated order — the chain from the wire bytes (framing, parse, structure, links, anchor, windows,
> revocation), the leaf's purpose and name, then the `CertificateVerify` signature with the leaf's own
> key — and returns a typed failure per stage, with the trait's `String` being that same `Display`.
> The anchors and the CRLs stay the caller's (borrowed, not copied, and no root store in this crate);
> `RejectAll` remains the verifier a caller who has decided nothing gets. The slice's real work was
> making the composition *testable*: a new `LEAF_EC_VALID` fixture (an EC leaf whose window sits inside
> its issuer's, which no previous fixture allowed — `LEAF_EC`'s opens after its issuer's closes), so the
> whole verifier is exercised with `0x0403` as well as `0x0804`, and generated `CertificateVerify`
> vectors signed with the fixture leaves' keys over the §4.4.3 construction, each **verified by the
> OpenSSL CLI before being written down**, plus the RSA one with a flipped byte, which the generator
> refuses to emit if OpenSSL still accepts it. Six neutering rounds (chain call, caller's `Revocation`,
> leaf check, signature check, `signed_data`/`signature` swap, the trait's entry point) all caught.
> **242 checks green on both paths** (+4), clippy, fmt, `cargo doc`, workspace and regeneration
> stability clean. Still no OCSP, no stapling, no CRL fetching and no transport — nothing in this crate
> has ever opened a socket.

> **Session 2026-10-16 (pt 13) in one paragraph.** The last *absent* check is now present: revocation,
> as caller-supplied CRLs. `crl.rs` reads a `CertificateList` strictly (unknown critical extensions
> refused, which is what refuses a delta CRL; `certificateIssuer` refused by name; a v1 CRL with
> extensions refused), and `chain.rs` gains a `Revocation` argument on `verify_chain` — `Unchecked`,
> which has to be written out because a revoked certificate verifies in that mode, or `Required(&[crls])`,
> where every certificate in the path needs a list from its issuer, the issuer must claim `cRLSign`,
> every supplied list must verify its signature, the freshest `thisUpdate` speaks, and two equally fresh
> lists that contradict each other are refused rather than guessed between. Seventeen cases were put to
> the OpenSSL CLI with `-crl_check_all`/`-crl_check` and recorded; two are divergences with this crate
> the stricter side (a delta CRL, which OpenSSL accepts as a complete one, and the leaf-only scope of
> `-crl_check`). **The slice's real find was a bug in the parser *and* the fixtures together**: a CRL's
> `version` is a plain `INTEGER`, not the `[0] EXPLICIT` a certificate uses, so thirteen unit tests were
> passing on CRLs OpenSSL could not even load, and every real v2 CRL would have been refused — caught
> only because the generator makes OpenSSL read every fixture and raises when it cannot. All 13 new
> rules were proven load-bearing by removing them one at a time (238 checks green on both paths, +21),
> and the neutering script's own bug — it ran one test target, so three rules looked unwatched — is
> recorded. **Still nothing calls any of it**: `RejectAll` is the only `CertificateVerifier`, and there is
> no OCSP, no stapling and no CRL fetching.

> **Session 2026-10-16 (pt 12) in one paragraph.** The question slice 17 raised — are key identifiers worth more
> than `-x509_strict`? — is answered by measurement: with a leaf whose `authorityKeyIdentifier` points at twenty
> bytes nothing owns, **OpenSSL refuses the path (error 20) while this crate accepted it**, which is the
> forbidden direction and so could not be written off as a divergence. `x509.rs` now reads
> `subjectKeyIdentifier` and `authorityKeyIdentifier` (whose `keyIdentifier` is `[0]` **implicit**, the trap in
> that extension), the generated table records both for all 30 fixtures so the parser is compared against
> OpenSSL's reading of the same bytes, and `chain.rs` refuses a link when the child states a key identifier and
> its issuer states a different one (`AuthorityKeyMismatch`, RFC 5280 §4.2.1.1). **The signature check cannot
> make this one**: a certificate whose identifier lies was still signed by whoever signed it, which is exactly
> why the fixture that proves the rule has a *valid* signature asserted beside the refusal. Absences are not
> refusals (no `authorityKeyIdentifier` on the child, no `subjectKeyIdentifier` on the issuer) and the anchor is
> checked on the same rule. Neutering it turns both the direct test and the oracle red. **217 checks green on
> both paths** (216 before), clippy, fmt, `cargo doc`, workspace clean; 30 fixtures, 23 oracle paths.

> **Session 2026-10-16 (pt 11) in one paragraph.** A gap found by reading the code rather than by a failing
> test: `intermediates_below` excludes self-issued certificates from a `pathLenConstraint` count — the rule
> that makes a **key rollover** legal — and in all 24 fixtures then every intermediate was non-self-issued,
> so the branch that does the excluding never excluded anything. It now has a rollover path
> (`ROLLOVER_CA_A` → `ROLLOVER_CA_B`, same DN, new key → a leaf), legal *only* because the rollover is
> self-issued, with the contrast in the same test and the branch proven load-bearing by neutering
> `self_issued` (the path is then `PathLengthExceeded { ca: Chain(2), allowed: 0, found: 1 }`). The oracle
> then produced the most useful result of the day: with no key identifiers it refused two **sound** rollover
> paths with error 7 and error 19 *while this crate accepted them* — a disagreement in the forbidden
> direction, which the test therefore refused to accept as a divergence. The fix was the fixtures (these
> three, and only these three, now carry AKI/SKI) and both paths then agree, which also re-reads the earlier
> `-x509_strict` finding: key identifiers are how a real verifier tells two same-DN certificates apart, so
> their absence was unrealistic fixtures, not a strict-mode nicety. **216 checks green on both paths** (215
> before), clippy, fmt, `cargo doc`, workspace and generator stability clean; 27 fixtures, 21 oracle paths.

> **Session 2026-10-16 (pt 10) in one paragraph.** The oracle's first real findings, from four paths about
> the *shape* of the presented list rather than about any rule. **OpenSSL selects a path out of the pool;
> this crate builds exactly the list it is given.** An unordered chain, a repeated certificate and an
> unrelated certificate padded in are all accepted by OpenSSL 3.5.5 and all refused here (`BrokenLink`,
> `NotACa`), which is deliberate and is RFC 8446 §4.4.2's own requirement — each certificate must certify
> the one before it — but it is a **real interoperability cost**, now stated in `chain.rs`'s "what this does
> not do": a server whose chain is out of order or padded is refused, and a caller who needs that tolerance
> has to select the path first. Being the stricter side is the only divergence the oracle test permits, so
> all three are recorded with their reason. They also happen to exercise "structure before links" on input
> nobody built to be a path — the refusals name the offending certificate's position as `NotACa`, not a
> broken link at the end — which is the property the module docs claim and no old fixture could show. 19
> paths, 5 recorded divergences, all five strict. The oracle test now collects *every* disagreement and
> prints the set, so adding a path costs one run rather than one run per surprise. **215 checks green on
> both paths** (unchanged — this slice added paths, not tests), clippy, fmt, `cargo doc`, workspace and
> generator stability all clean.

> **Session 2026-10-16 (pt 9) in one paragraph.** The crate gets an **oracle**. Every check so far compared
> our reading of a certificate against OpenSSL's *parser*, which left the policy — which rules run, and what
> each means — checked only against my own reading of RFC 5280. Fifteen paths are now put to a second
> implementation of path validation at the same instants (`openssl verify -partial_chain -auth_level 0`,
> 3.5.5), with the verdicts recorded in the generated `VERIFY_CASES` table rather than taken at test time,
> so nothing silently skips when the tool is missing. The test enforces the direction a disagreement may
> take: OpenSSL accepting while this crate refuses, never the reverse. It paid for itself three times over.
> It **measured** the identity-pin claim from pt 8, which had been recalled from memory rather than shown
> (OpenSSL accepts a self-signed `cA: FALSE` certificate pinned as its own anchor). It **independently
> corroborates** the new anchor rule (error 79, "invalid CA certificate", exactly as pt 8 reasoned). And
> with `keyCertSign` neutered the oracle **fails by name** — a second implementation catching a regression
> in a rule it was never told about. Two divergences remain, both with this crate the stricter side: a CA
> certificate used as the server (`CaUsedAsLeaf`) and the `Policy::min_rsa_bits` floor. **215 checks green
> on the hardware path and on `HEARTH_TLS_NO_HW=1`** (214 before), clippy `--all-targets -D warnings`, fmt,
> `cargo doc` (zero warnings) and the workspace check clean, `std`-only, empty `[dependencies]`, and still
> nothing in the crate calls
> any of it: `RejectAll` is the only `CertificateVerifier`, and there is **no revocation checking of any
> kind**. History: `docs/BUG_TRACKER.md`.

## Next session — in this order

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
