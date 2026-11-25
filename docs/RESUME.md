# Hearth — next session

> **Session 2026-11-25 in one paragraph.** PrismML's `Ternary-Bonsai-2.5-27B-GGUF` was
> verified against the loader and added to the downloader catalog, README and AGENTS (same
> `qwen35` geometry and Hadamard metadata as Bonsai 2, so no engine change; `-mtp` and
> `mmproj` deliberately unlisted). Then paired hybrid prefill, in two halves:
> `qwen35::forward_pair` now runs **two prompt tokens through one weight pass for every
> per-token matmul** — FFN first (1.41× on that stage), then the mix projections. This
> session's half split `gdn_layer`/`attn_layer` into a batchable projection (`attn_qkv`,
> `attn_gate`, `attn_q/k/v`, `ssm_out`, `attn_output` through `mm_batched`) and a
> sequential `gdn_scan`/`attn_scan` (conv, L2 norm, gates, delta rule, KV append,
> attention) that is the decode body verbatim. 27B prefill 650 → 511 ms/token (1.27-1.30×,
> mix stage 290 → 253 ms/token, FFN 1.36×), greedy output byte-identical on 27B/9B/4B
> hybrids and a dense control, decode untouched. Then the duplicated per-token FFN was
> folded into one `ffn_block` that both paths call (1 row for decode, `PREFILL_CHUNK` for
> prefill), again byte-identical with no measurable cost. Detail: `docs/BUG_TRACKER.md`,
> newest entry first.

Earlier sessions: `docs/BUG_TRACKER.md`. Packaging: `packaging/windows/README.md`,
`packaging/SIGNING.md`.

## Do these first

1. **Test `PREFILL_CHUNK = 4` and stop conflating it with the x4 kernel.** The rejected x4
   was a *kernel* variant. `pool::batched_rows_chunk` tiles the weights at 1 MB and runs the
   token loop *inside* the tile, pairing two at a time, so `seq_len = 4` needs no new kernel:
   the tile is read once and reused across four activations, halving DRAM traffic per token
   again. Everything downstream is width-agnostic (scans stay per token, scratch is sized
   from the constant), so it is a one-line change plus identity and timing runs. If the 1 MB
   tile is not actually resident, the number will be flat or worse — measure.
2. **Download and smoke the other two 2.5 packs** (`PQ2_0` 7,253,354,848 B, `Q2_0`
   7,673,194,848 B). Headers match PTQ1_0, but only PTQ1_0 has decoded a token here, and
   `Q2_0` exercises the Q2 kernel path of the hybrid forward that PTQ1_0 does not.
3. **Decide on MTP.** The `-mtp` files add a draft head Hearth cannot use yet. Keep them out
   of the catalog (current state) or wire the head when speculative decoding lands. Do not
   list a file the engine cannot exploit.
4. **Refusal-or-load for foreign `qwen35` packs.** `ModelConfig` accepts any `qwen35` file;
   the loader then fails on the first missing tensor. A ten-line pre-flight check (arch →
   required tensor names → Hadamard metadata) would name the reason instead of
   `qwen35: missing blk.0.attn_qkv.weight`.
5. **Re-baseline the README performance table** before changing a number in it. The 27B cells
   came from a busy window; the same model's paired runs spread 1.3× inside one session.

## Open inference items (most valuable first)

1. **The scans are now the majority of prefill** — 216 of 511 ms/token, against 246 for the
   FFN. The delta rule is `hd x hd` state work per v-head with a column-strided inner loop
   (`state[col + i * hd]`), and attention walks every cached key per head. Candidate: keep
   the recurrence but make the state layout row-major per head so both the decay and the
   `q`/`k` products are contiguous, then measure; do not vectorize before the layout
   changes, because the strided read is the cost.
2. **aarch64 has no SIMD backend**; every kernel is the scalar fallback. NEON first suits
   the `Q2V_U8` LUT dot (`vqtbl`) and the f32 glue in `ops.rs`. Needs a real ARM runner
   before any number means anything.
3. **Worker counts follow the path** — dense = `physical`; hybrid `Q2_0`/`PQ2_0` =
   `physical + physical/4`; hybrid `PTQ1_0` = ¾ logical. `HEARTH_THREADS` overrides. Judge a
   count on its **spread** (`scripts/count-sweep.py`), not its mean.
4. **Speculative decoding: measure acceptance `a` first.** The 27B's decode is ~650 ms/token
   and the 2B's ~50 ms, but 3 of every 4 `qwen35` layers are recurrent, so a rejected draft
   must rewind conv+SSM state, not just a KV cache.
5. **Paired prefill is done; do not re-derive the split.** `forward_pair` pairs every
   projection (mix, `ssm_out`, `attn_output`, FFN) and runs `gdn_scan`/`attn_scan` per token
   in order, which is why the output is bit-identical. `HEARTH_Q35_NOPAIR=1` reverts.
   `gdn_out_q8` holds the grouped-order Hadamard permutation for `ssm_out`, and the whole
   per-token FFN exists once in `ffn_block` (called with `rows = 1` from `forward`, `rows =
   PREFILL_CHUNK` from `forward_pair`), so do not reintroduce a second copy of it.
6. **Kernel rules that made past fixes findable**: a `#[target_feature]` body names every
   feature it uses and its dispatch gate names the same set; `.cargo/config.toml` pins
   `x86-64-v3` and must not drop below what any dispatched body uses. A new quant format
   needs `kernel_name()` and a `selected_kernel` arm once it has two implementations.
7. **GPU scaffolding: still not started** (opt-in via `--device`/env, graceful without a
   ROCm device, must not break the CPU path or Windows/macOS builds).

## Goal 2 (UI / docs / downloader)

1. **README screenshots are stale**; re-shoot `/` and the downloads drawer at ~1600×1000.
2. **The catalog only grows by hand.** Twenty entries, three added this session. A
   `models/catalog.json` in the repo would let a release add packs without a rebuild; the
   override path already exists (`HEARTH_MODEL_CATALOG_JSON`).
3. **The downloader's DNS fallback is unit-tested only** — check it against a real
   dual-stack host and a host whose first address is unreachable.

## Standings and hazards

- **The model list is a multi-root scan.** `hearth_core::paths` is the only answer to
  "where are the models"; do not re-add a walk-up elsewhere.
- **`p256.rs` is verification-only and variable-time by design**; TLS 1.3 here uses X25519
  and the crate never signs.
- **Do not re-derive the prefill/decode asymmetry** — the 27B's prefill stays *faster* than
  its decode (511 vs ~650 ms/token), and both are weight-bound.
- **Do not "fix" the first token** — it is storage I/O on the first read of the weights;
  `HEARTH_PIN_WEIGHTS=1` died silently on the 27B on this 16 GB box.
- **Warm up before timing.** A cold 27B run reads 605 ms/token prefill against 504 warm, and
  a cold 8B Q2_0 read 2.88 tok/s decode against 6.64 warm. Bracket A/B as `b,a,a,b`.
- **Verify through the interface, not the summary**: drive the real API
  (`HEARTH_MODELS_DIR`, `HEARTH_MODEL_CATALOG_JSON`, `--no-browser`,
  `HEARTH_NO_AUTO_SHUTDOWN=1`), `--temp 0 --prompt-raw` for benchmarks.
- **`hearth-tls` is not a general-purpose TLS stack** — scope is "the downloader can reach
  `https://` with no dependencies"; unneeded features are *refused*, not ignored.
- **MSI facts worth not re-learning**: an OS-version launch condition cannot pass on a
  machine that emulates `VersionNT`; a `File` installs into its component's directory;
  `RemoveFolder` removes empty directories only (which is what preserves `models/`);
  `wix build` needs `-acceptEula wix7`; custom-action FileRefs need `ExeCommand=""`.
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
