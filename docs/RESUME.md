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
> prefill), again byte-identical with no measurable cost. Last, the GDN recurrence: state
> laid out `[j][i]` per head so a delta-rule column is contiguous, and the 48 v-heads spread
> over the pool — 1.57× the prefill scan stage, decode 1.47 → 1.67 tok/s on the 27B (1.3× on
> the 4B/9B). Then the chat template: the evaluator became a real Jinja2 subset (`set`,
> `namespace`, macros, filters, `is` tests, slicing, whitespace control) and the silent
> synthetic fallback is gone — all 15 packs now render byte-identically to jinja2 3.1, and
> the 2.5 27B gets its card prompt (67 tokens with `--show-think`) instead of the synthetic
> 33. Then `kernel-bench`: a per-format row-kernel microbenchmark (weight GB/s and Gweights/s
> per thread count, median of five passes over a buffer past L3), sharing one dispatch table
> with `matmul` — it put Q5_0/Q5_1 at 0.35 vs PTQ1_0's 6.5 Gweights/s per thread and showed 16
> threads losing to 12 on every pooled format. Mapping its ceiling onto the 27B's stages says
> decode runs at ~92% of the PTQ1_0 kernel ceiling (ffn 10.83, gdn/attn 10.22, head 11.50 GB/s
> against 11.6), so the constraint is per-weight decode arithmetic, not bandwidth or dispatch.
> Detail: `docs/BUG_TRACKER.md`, newest entry first.

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

6. **Tool calling is the next template gap, and it is unchecked.** The 2.5 template's
   `tools` branch (`|tojson`, `tool_call.function`, `{% for k, v in …|items %}`, the
   `<tool_call>` XML format) parses, but no fixture exercises it: `TemplateContext` has no
   `tools` field. Before wiring tools through the UI, add the field and a fixture case
   rendered by the reference engine, or the first real tool call goes out in a format
   nobody checked.

7. **The PTQ1_0 decode is done; do not re-litigate it.** Widening the decode from 8-byte
   halves to 16 lanes (`dot_ptq1_0_q8_0_avx2_i16`, weights kept as `i16`) moved one thread from
   6.75 to 9.46 Gweights/s and the 12-worker ceiling from 12.0 to 16.3-16.6 GB/s, worth 1.27-1.38x
   on every PTQ1_0 hybrid and 1.13x on prefill. The table-driven form Q2_0 uses was measured first
   and is **35% slower** here (0.92 against 1.42 GB/s) because a PTQ1_0 byte's trits land 16
   elements apart, so do not retry it. Decode now sits at ~90% of its own ceiling again (14.40
   against 16.4 GB/s), which is the same shape the previous entry described: the ceiling is
   arithmetic, DRAM is ~25% used, and dispatch is under 10 ms of 417. A wider datapath is the only
   remaining lever on this kernel, and this Zen 4 has no full-width AVX-512 to give it.
   A kernel claim should name the GB/s, the format and the thread count, plus the whole-model
   tok/s it was meant to move. Do not go looking for PTQ1_0's 8-lane defect in the other formats:
   Q2_0's LUT already emits `i16` and Q1_0_G128's VNNI expands bits 32 lanes at a time, measured
   at 9.3-10.0 and 22.6-24.3 Gweights/s per thread with 7.6-8.1x scaling, and building Q2_0's
   weight vector in registers instead of a stack buffer changed nothing (2.57 vs 2.55 GB/s). Before
   trusting a stage number, check the kernel's curve: PTQ1_0 tops out at 11.2 GB/s (12 threads),
   Q1_0_G128 at 25.3 (12), Q8_0 at 48.7 (12), and above eight threads the cells move by up to 40%
   between runs. The one open finding it recorded is Q5_0/Q5_1 at 0.35 Gweights/s single-thread,
   a per-element `% 2` unpack into a `[f32; 32]` before the SIMD madd; nothing in the catalog is
   Q5, so it is a rewrite-for-a-LUT-kernel job only if such a model ever lands.

## Open inference items (most valuable first)

0. **The 27B's stage-level waste is gone; do not re-open it.** `matvec_dense` is split over the
   pool now (`gdn gates` 20 ms -> 3-4 ms, `gdn/attn` 154.5 -> 138.9, five hybrid models still
   byte-identical), the PTQ1_0 decode is 16 lanes wide, and the scan measures 8.5-9.7 ms with
   nothing to win without changing summation. What remains in the stage is size-limited matmul
   throughput: 6.7-11.2 MiB projections read 77-85% of the ceiling for their own size, so a stage
   number below `kernel-bench`'s 140 MiB headline is expected rather than a leak. The split also
   pays on prefill (per prompt token: `gdn gates` 19.4 -> 3.0 ms on the 27B, 5.2 -> 0.9 on the 9B;
   prefill -6.1% per token on the 27B), and both prefill paths now time the call — before that the
   row only caught the leftover token when the prompt was not a multiple of `PREFILL_CHUNK`.
1. **The 27B's ffn is 258-269 ms against a 228.5 ms single-pass stream, and neither dispatches nor
   park/wake explain the gap.** Fusing gate+up (192 -> 128 dispatches) is bit-identical and worth
   ~9 ms, not 29. Giving idle workers a longer bounded spin before they park is *worse* (medians
   258 / 265 / 263 ms for windows of 0 / 200 / 2000 us), and forcing them to park costs ~3 ms, so a
   wake is 0.02-0.04 ms per dispatch: the 0.45 ms figure is dead and both knobs were reverted. Next
   is sub-timers *inside* the bucket — silu/gating over 17408 values per layer, the two per-layer
   activation quantizations, and 19.5 MB tensors read one dispatch at a time out of a 5.4 GB mmap
   instead of one contiguous 3.74 GB pass. Measure before changing.
   The Q1_0 ffn lever is *priced and blocked*, not open: no writer or Q1_0 quantizer ships, and the
   throwaway converter that measured it is in history at `1c9f5ef`, not in the tree — do not re-add
   one unless the answer changes. A 9B with ffn in 1-bit loses the 32.8% of weights that are exact
   zeros for relL2 0.575 and then answers with EOS, so it needs a 27B made 1-bit from source.
2. **The scan is still the biggest stage after the FFN** — 161-169 of 436-451 ms/token
   prefill, and 210-224 ms of ~590 decode, against 258 for the FFN. Layout and head split
   are done: `GdnState` is `[j][i]` per head and `gdn_scan_heads` runs the 48 heads across the
   pool, worth 1.57x on the prefill stage and 1.14-1.31x end to end. What remains is not the
   strided read: 4 heads per worker per layer and ~12 MB of state access is ~5 GB/s
   aggregate, so suspect the dispatch cadence (one `par_for` per GDN layer, ~48/token on top
   of ~450 matmul dispatches) and an L2-resident working set. Count bytes per layer against
   dispatches before changing anything.
3. **aarch64 has no SIMD backend**; every kernel is the scalar fallback. NEON first suits
   the `Q2V_U8` LUT dot (`vqtbl`) and the f32 glue in `ops.rs`. Needs a real ARM runner
   before any number means anything.
4. **Worker counts follow the path** — dense = `physical`; hybrid `Q2_0`/`PQ2_0` =
   `physical + physical/4`; hybrid `PTQ1_0` = ¾ logical. `HEARTH_THREADS` overrides. Judge a
   count on its **spread** (`scripts/count-sweep.py`), not its mean.
5. **Speculative decoding: measure acceptance `a` first.** The 27B's decode is ~650 ms/token
   and the 2B's ~50 ms, but 3 of every 4 `qwen35` layers are recurrent, so a rejected draft
   must rewind conv+SSM state, not just a KV cache.
6. **Paired prefill is done; do not re-derive the split.** `forward_pair` pairs every
   projection (mix, `ssm_out`, `attn_output`, FFN) and runs `gdn_scan`/`attn_scan` per token
   in order, which is why the output is bit-identical. `HEARTH_Q35_NOPAIR=1` reverts.
   `gdn_out_q8` holds the grouped-order Hadamard permutation for `ssm_out`, and the whole
   per-token FFN exists once in `ffn_block` (called with `rows = 1` from `forward`, `rows =
   PREFILL_CHUNK` from `forward_pair`), so do not reintroduce a second copy of it.
7. **Kernel rules that made past fixes findable**: a `#[target_feature]` body names every
   feature it uses and its dispatch gate names the same set; `.cargo/config.toml` pins
   `x86-64-v3` and must not drop below what any dispatched body uses. A new quant format
   needs `kernel_name()` and a `selected_kernel` arm once it has two implementations.
8. **GPU scaffolding: still not started** (opt-in via `--device`/env, graceful without a
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
