# Hearth — next session

> **Session 2026-11-25 in one paragraph.** PrismML's `Ternary-Bonsai-2.5-27B-GGUF`
> (`prism-ml/bonsai-25-27b`) was checked against the loader and added. Ranged 12 MB header
> fetches of all three base packs show the same `qwen35` geometry and the same Hadamard
> metadata as Bonsai 2 27B (`version 1`, `block_size 1024`, sign widths 5120/6144/17408,
> latent `token_embd`), and identical tensor names — so **no engine change was needed**.
> The three packs are in the built-in downloader catalog with exact byte sizes, plus the
> README model table and AGENTS key paths. `-mtp` (draft head) and `mmproj` (vision) are
> deliberately unlisted. PTQ1_0 loads and decodes; interleaved 20-token pairs put it at
> parity with Bonsai 2 (1.24/0.92 vs 1.31/0.71 tok/s), so no speed claim and no perf-table
> row. Then a paired hybrid prefill (two prompt tokens per FFN weight pass) cut the 27B's
> prefill from 663 to 546 ms/token — 1.41× on the FFN stage — with byte-identical output on
> both a PTQ1_0 and a PQ2_0 hybrid; the recurrent projections are the remaining half.
> Detail: `docs/BUG_TRACKER.md`, newest entry first.

Earlier sessions: `docs/BUG_TRACKER.md`. Packaging: `packaging/windows/README.md`,
`packaging/SIGNING.md`.

## Do these first

1. **Download and smoke the other two 2.5 packs** (`PQ2_0` 7,253,354,848 B, `Q2_0`
   7,673,194,848 B). Their headers match PTQ1_0, but only PTQ1_0 has actually decoded a
   token here. `Q2_0` is the llama.cpp-standard 2-bit pack and exercises the Q2 kernel path
   of the hybrid forward, which PTQ1_0 does not.
2. **Decide on MTP.** The `-mtp` files add a draft head that Hearth cannot use yet. Either
   keep them out of the catalog (current state) or, if speculative decoding lands, wire the
   head and add them. Do not list a file the engine cannot exploit.
3. **Refusal-or-load for foreign `qwen35` packs.** `ModelConfig` accepts any `qwen35` file;
   the loader then fails on the first missing tensor name. A ten-line pre-flight check
   (arch → required tensor names → Hadamard metadata) would name the reason instead of
   `qwen35: missing blk.0.attn_qkv.weight`.
4. **Re-baseline the README performance table** before changing any number in it. The 27B
   cells came from a busy window; this session's pair spread was 1.3× within one model.

## Open inference items (most valuable first)

1. **aarch64 has no SIMD backend**; every kernel is the scalar fallback. NEON first suits
   the `Q2V_U8` LUT dot (`vqtbl` is a table lookup) and the f32 glue in `ops.rs`. Needs a
   real ARM runner before any number means anything.
2. **Worker counts follow the path** — dense = `physical`; hybrid `Q2_0`/`PQ2_0` =
   `physical + physical/4`; hybrid `PTQ1_0` = ¾ logical. `HEARTH_THREADS` overrides. Judge a
   count on its **spread** (`scripts/count-sweep.py`), not its mean.
3. **Speculative decoding: measure acceptance `a` first.** The 27B's decode is ~720 ms/token
   and the 2B's ~50 ms, but 3 of every 4 `qwen35` layers are recurrent, so a rejected draft
   must rewind conv+SSM state, not just a KV cache.
4. **Paired prefill is in; the projections are the next half.** `qwen35::forward_pair` runs
   two prompt tokens through one FFN weight pass (`PREFILL_CHUNK = 2`; `HEARTH_Q35_NOPAIR`
   reverts). On the 2.5 27B the FFN stage went 26.0 → 18.45 s for 76 tokens and total prefill
   663 → 546 ms/token, with byte-identical greedy output. What remains is `gdn_layer` and
   `attn_layer`, which still project one token at a time: split each into a batchable
   `project` (qkv/gate/alpha/beta, attn q/k/v/o) and a sequential `scan` (conv, delta rule,
   KV append), then call `mm_batched` there too. Same primitive, no new kernels. Do not
   widen the chunk — the x4 variant of the paired kernel regressed 3.2×.
5. **Kernel rules that made past fixes findable**: a `#[target_feature]` body names every
   feature it uses and its dispatch gate names the same set; `.cargo/config.toml` pins
   `x86-64-v3` and must not drop below what any dispatched body uses. A new quant format
   needs `kernel_name()` and a `selected_kernel` arm once it has two implementations.
6. **GPU scaffolding: still not started** (opt-in via `--device`/env, graceful without a
   ROCm device, must not break the CPU path or Windows/macOS builds).

## Goal 2 (UI / docs / downloader)

1. **README screenshots are stale**; re-shoot `/` and the downloads drawer at ~1600×1000.
2. **The catalog only grows by hand.** Twenty entries now, three of them new today. A
   `models/catalog.json` shipped in the repo would let a release add packs without a
   rebuild; the override path already exists (`HEARTH_MODEL_CATALOG_JSON`).
3. **The downloader's DNS fallback is unit-tested only** — check it against a real
   dual-stack host and a host whose first address is unreachable.

## Standings and hazards

- **The model list is a multi-root scan.** `hearth_core::paths` is the only answer to
  \"where are the models\"; do not re-add a walk-up elsewhere.
- **`p256.rs` is verification-only and variable-time by design**; TLS 1.3 here uses X25519
  and the crate never signs.
- **Do not re-derive the prefill/decode asymmetry** — the 27B's prefill is *faster* than
  its decode (635 vs 722 ms/token).
- **Do not \"fix\" the first token** — it is storage I/O on the first read of the weights;
  `HEARTH_PIN_WEIGHTS=1` died silently on the 27B on this 16 GB box.
- **Verify through the interface, not the summary**: drive the real API
  (`HEARTH_MODELS_DIR`, `HEARTH_MODEL_CATALOG_JSON`, `--no-browser`,
  `HEARTH_NO_AUTO_SHUTDOWN=1`), `--temp 0 --prompt-raw` for benchmarks.
- **A measurement needs its window recorded.** Absolute levels drift ~10% over an hour and
  the second run in a pair is worth ~2-3%; use `b,a,a,b` groups and compare means.
- **`hearth-tls` is not a general-purpose TLS stack** — scope is \"the downloader can reach
  `https://` with no dependencies\"; unneeded features are *refused*, not ignored.
- **MSI facts worth not re-learning**: an OS-version launch condition cannot pass on a
  machine that emulates `VersionNT`; a `File` installs into its component's directory;
  `RemoveFolder` removes empty directories only (which is what preserves `models/`);
  `wix build` needs `-acceptEula wix7`; custom-action FileRefs need `ExeCommand=\"\"`.
- **Runner labels fail silently.** `macos-13` is retired; use `macos-15-intel` /
  `macos-15`. The Linux leg is pinned to `ubuntu-24.04` to hold the glibc floor.

## Verification habits for a shared checkout

- `cargo fmt --all` reformats the other agent's files — use `cargo fmt -p <crate>`.
- `target/release/hearth-chat-cli.exe` is often locked by a running UI: `netstat -ano` for
  the port, kill that PID; the bash wrapper's PID is not the exe's.
- **A constant copied from memory is not a vector** — derive it and have an independent
  implementation agree before embedding it.
- **A test whose two sides run the same code proves nothing**, and **a control that passes
  is a control that was never tested** — assert the domain directly.
- **Suspect your own harness before calling someone else's data broken.**
