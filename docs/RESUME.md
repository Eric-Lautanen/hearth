# Hearth — next session

> **Session 2026-11-26 in one paragraph.** Started by settling what the gate+up fusion was worth:
> thirteen alternating pairs put it at a ~12 ms median (11 of 13 favour it), not the 29 ms the entry
> hoped for, and the old arm drifted 326.9 -> 279.7 ms *within* one set, which is why single pairs
> are useless on this box. Then the next suspect died: idle workers were given a longer bounded spin
> before parking (`HEARTH_POOL_SPIN_US`) and a knob for the iteration budget itself — extending the
> spin is *worse* (medians 258 / 265 / 263 ms for windows of 0 / 200 / 2000 us) and forcing the
> workers to park costs ~3 ms per token, so a wake is 0.02-0.04 ms across 128 dispatches and the
> 0.45 ms figure is dead. Both knobs were reverted; `pool.rs` is untouched. The throwaway Q1_0
> converter was deleted now that its number is recorded (git `1c9f5ef`), with the recipe kept in the
> tracker. Then the ffn bucket was opened up: five sub-timers in `ffn_block`, summing to the row
> exactly, which show **+24 ms in the two weight passes** (14.6-14.8 GB/s against 16.28 read-only,
> partly real arithmetic the read-only 228.5 ms figure never did) and **+15.1 ms in the serial
> rotate+quantize**. Finished with the mandated sweep over all 15 packs on this machine and a
> re-baselined README: the dense six were never stale (±5%), the hybrid rows were 1.6-1.75x low.
> Then the two missing 2.5 packs were downloaded and benched: `PQ2_0` is a clean 1.94 tok/s, but
> `Q2_0` runs at 2.15 tok/s of `!` — the first hybrid pack whose weights are not all pre-rotated.
> Detail: `docs/BUG_TRACKER.md`, newest entry first.

Earlier sessions: `docs/BUG_TRACKER.md`. Packaging: `packaging/windows/README.md`,
`packaging/SIGNING.md`.

## Baseline to compare against (2026-11-26, 50 tokens, temp 0, prompt-raw "Hello")

`48.1 / 27.1 / 24.4 / 11.7 / 12.4 / 6.2` tok/s for the dense six in README order (1.7B Q1_0,
1.7B Q2_0, 4B Q1_0, 4B Q2_0, 8B Q1_0, 8B Q2_0); `25.3 / 25.1 / 12.0 / 11.2 / 6.7 / 6.3` for the
hybrid 2B/4B/9B in PTQ1_0, PQ2_0 order; `2.17 / 1.98` for the 27B pair, and `2.16 / 1.94 / 2.15` for the 2.5
refresh (`PTQ1_0`, `PQ2_0`, `Q2_0` — the last one is a broken pack, see below). Per-file drift is not uniform — the 2B `PTQ1_0` read 27.81 then 22.81 in one
session — so compare the model you changed, and use alternating pairs, never two single runs.

## Do these first

1. **SIMD the rotate+quantize — it is the largest removable term left in decode.** The ffn
   sub-timers put it at **15.1 ms/token on the 27B** (3.4% of the token: 3.4 ms on the 5120-wide
   normed row, 11.6 ms on the 17408-wide gated row) at ~96 Mvalues/s, serial and scalar. That is the
   Hadamard fold plus the Q8 quantize in `rotate_quantize_into`; the per-128 block math is
   independent per block, so a SIMD pass can stay **bit-identical** — verify with the usual
   five-model identity run, then read the `ffn q in` / `ffn q mid` rows to see the win.
2. **Test `PREFILL_CHUNK = 4` and stop conflating it with the x4 kernel.** The rejected x4 was a
   *kernel* variant. `pool::batched_rows_chunk` tiles the weights at 1 MB and runs the token loop
   *inside* the tile, pairing two at a time, so `seq_len = 4` needs no new kernel: the tile is read
   once and reused across four activations, halving DRAM traffic per token again. Everything
   downstream is width-agnostic (scans stay per token, scratch is sized from the constant), so it is
   a one-line change plus identity and timing runs. If the 1 MB tile is not actually resident, the
   number will be flat or worse — measure.
3. **Refuse (or implement) the 2.5 `Q2_0` pack: it is a 36-byte-per-128 layout Hearth reads as 34.**
   Measured, not hypothesised: the tokenizer is identical to the working packs (same SHA over
   248,320 tokens and 247,587 merges, and `!` is id 0 in all), the hidden state is already 11,718x
   too large straight out of the embedding dequant (`emb_raw` 132.0 against 0.0113), and the tensor
   table steps 36 bytes per 128 elements where `hearth-quant/src/q2_0.rs` hard-codes
   `BLOCK_BYTES = 34`. Only block 0 lines up, which is why the first bytes match and the output does
   not. An earlier guess in the tracker blamed the Hadamard metadata; that was wrong — do not
   "fix" the rotation, it would break the packs that work. The cheap remedy is a load-time stride
   check (next tensor offset minus this one, per 128 elements, against `dtype.byte_size(shape)`),
   which turns silent garbage into a named error.
4. **Tool calling is the next template gap, and it is unchecked.** The 2.5 template's `tools` branch
   (`|tojson`, `tool_call.function`, `{% for k, v in …|items %}`, the `<tool_call>` XML format)
   parses, but no fixture exercises it: `TemplateContext` has no `tools` field. Before wiring tools
   through the UI, add the field and a fixture case rendered by the reference engine.
5. **Refusal-or-load for foreign `qwen35` packs.** `ModelConfig` accepts any `qwen35` file; the
   loader then fails on the first missing tensor. A ten-line pre-flight check (arch → required
   tensor names → Hadamard metadata) would name the reason instead of
   `qwen35: missing blk.0.attn_qkv.weight`.
6. **Decide on MTP.** The `-mtp` files add a draft head Hearth cannot use yet. Keep them out of the
   catalog (current state) or wire the head when speculative decoding lands.

## Do not re-open (each of these is measured, with the number)

- **The 27B's ffn decomposition.** Sub-timers sum to the row exactly. +24 ms is the two weight passes
  at 14.6-14.8 GB/s (a kernel-shaped floor ~15% above the byte floor, because the read benchmark
  does not run 17.1 Gweights of MACs), +15.1 ms is the rotate+quantize, +1.1 ms silu, +0.8 ms norms
  and residuals. Nothing else is hiding there.
- **Dispatch count as an explanation.** The gate/up fusion is real and ~12 ms; a longer spin before
  parking is worse; forcing a park costs ~3 ms. The 0.45 ms/dispatch reading was wrong by an order
  of magnitude.
- **The PTQ1_0 decode datapath.** 16-lane `i16` is done (6.75 -> 9.46 Gweights/s per thread, ceiling
  12.0 -> 16.3-16.6 GB/s); the table-driven Q2_0 form was measured 35% *slower* here, so do not retry
  it. Decode sits at ~90% of its own ceiling and DRAM is ~25% used.
- **Paired prefill.** `forward_pair` pairs every projection and runs the scans per token in order,
  which is why output is bit-identical; `HEARTH_Q35_NOPAIR=1` reverts. One copy of the per-token FFN
  exists, in `ffn_block`.
- **Stage-level waste in the 27B.** `matvec_dense` is split over the pool (`gdn gates` 20 -> 3-4 ms),
  and what remains is size-limited throughput: 6.7-11.2 MiB projections read 77-85% of their own
  ceiling, so a stage number below `kernel-bench`'s 140 MiB headline is expected.
- **The Q1_0 ffn lever.** Post-hoc 1-bit destroys a converted model (32.8% exact zeros, relL2 0.575,
  EOS). Needs a 27B made 1-bit from source; the converter that measured it is in git at `1c9f5ef`,
  not in the tree — do not re-add one unless the answer changes.
- **The first token.** It is storage I/O on the first read of the weights; `HEARTH_PIN_WEIGHTS=1`
  died silently on the 27B on this 16 GB box.

## Open items (most valuable first)

1. **The scan is still the biggest stage after the FFN** — 161-169 of 436-451 ms/token prefill and
   210-224 ms of ~590 decode, against 258-270 for the FFN. Layout and head split are done. What
   remains is not the strided read (~5 GB/s aggregate), so suspect the dispatch cadence (one
   `par_for` per GDN layer, ~48/token) and an L2-resident working set. Count bytes per layer against
   dispatches before changing anything.
2. **aarch64 has no SIMD backend**; every kernel is the scalar fallback. NEON first suits the
   `Q2V_U8` LUT dot (`vqtbl`) and the f32 glue in `ops.rs`. Needs a real ARM runner before any number
   means anything.
3. **Worker counts follow the path** — dense = `physical`; hybrid `Q2_0`/`PQ2_0` = `physical +
   physical/4`; hybrid `PTQ1_0` = ¾ logical. `HEARTH_THREADS` overrides. Judge a count on its
   **spread** (`scripts/count-sweep.py`), not its mean.
4. **Speculative decoding: measure acceptance `a` first.** 3 of every 4 `qwen35` layers are
   recurrent, so a rejected draft must rewind conv+SSM state, not just a KV cache.
5. **GPU scaffolding: still not started** (opt-in via `--device`/env, graceful without a ROCm
   device, must not break the CPU path or Windows/macOS builds).
6. **Kernel rules that made past fixes findable**: a `#[target_feature]` body names every feature it
   uses and its dispatch gate names the same set; `.cargo/config.toml` pins `x86-64-v3` and must not
   drop below what any dispatched body uses. A new quant format needs `kernel_name()` and a
   `selected_kernel` arm once it has two implementations. Q5_0/Q5_1 remain 0.35 Gweights/s
   single-thread (per-element `% 2` unpack before the SIMD madd) — rewrite only if such a model lands.

## Goal 2 (UI / docs / downloader)

1. **README screenshots are stale**; re-shoot `/` and the downloads drawer at ~1600×1000.
2. **The catalog only grows by hand.** A `models/catalog.json` in the repo would let a release add
   packs without a rebuild; the override path already exists (`HEARTH_MODEL_CATALOG_JSON`).
3. **The downloader's DNS fallback is unit-tested only** — check it against a real dual-stack host
   and a host whose first address is unreachable.

## Standings and hazards

- **Re-baseline before believing a delta.** The README's hybrid rows sat 1.6-1.75x low for weeks
  because nobody re-measured them; every hybrid improvement since then was invisible in the table.
- **`[generate] avg_cpu_overhead=Nus/token` is mislabelled** — it prints the whole step including
  the forward, so it is not overhead and nothing is wasting that time.
- **The model list is a multi-root scan.** `hearth_core::paths` is the only answer to "where are the
  models"; do not re-add a walk-up elsewhere.
- **`p256.rs` is verification-only and variable-time by design**; TLS 1.3 here uses X25519 and the
  crate never signs.
- **Do not re-derive the prefill/decode asymmetry** — the 27B's prefill stays *faster* than its
  decode, and both are weight-bound.
- **Warm up before timing.** A cold 27B run reads 605 ms/token prefill against 504 warm, and a cold
  8B Q2_0 read 2.88 tok/s decode against 6.64 warm. Bracket A/B as `b,a,a,b`.
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

## Verification habits for a shared checkout

- `cargo fmt --all` reformats the other agent's files — use `cargo fmt -p <crate>`.
- `target/release/hearth-chat-cli.exe` is often locked by a running UI: `netstat -ano` for the port,
  kill that PID; the bash wrapper's PID is not the exe's.
- **A constant copied from memory is not a vector** — derive it and have an independent
  implementation agree before embedding it.
- **A test whose two sides run the same code proves nothing**, and **a control that passes is a
  control that was never tested** — assert the domain directly.
- **Suspect your own harness before calling someone else's data broken.**
- **Timer rows need an exact field match.** `[q35timing]` rows are `$1="[q35timing]"`, `$2=name`
  (so `gdn gates` is `$2="gdn"`), and the ffn sub-rows put the value in `$3`; match `$2=="ffn"` or a
  regex on ` us `, never `/ffn /`, or the sub-rows arrive as extra matches.
