# Hearth — Session 2026-09-19 (pt 6): the 27B's context was allocated from its declared size

> The 27B chat ran (or didn't) out of RAM: the KV cache is allocated at load from the context
> ceiling, and pt 4 had raised the hybrid cap to 32768 while the model declares 262144 — 4.3 GiB of
> KV at 128 KiB/token on a 16 GB box. Context is now sized by a byte budget (`model/kv_budget.rs`,
> 1.25 GiB): dense models keep 4096, the 27B gets 10240. `RunStats` also stopped counting the prompt
> as generated tokens, which is why an empty reply read as "28 tokens generated". Same session: the
> idle worker pool spent **414% of one core** in a spin/`yield_now` loop that never blocks — it now
> parks on a condvar with a syscall-free dispatch path (`pool.rs`), idle CPU 0.0%, throughput
> unchanged. See BUG_TRACKER pt 6.

## Next session — things to do/try

1. **27B prefill is the one thing that still makes the chat feel broken.** Measured this session:
   1.8–2.3 s per *prompt* token (a 33-token chat prompt = 60–76 s before the first token) against
   ~0.9 s per *decoded* token. That is pt 5's item 1 — paired kernels for the formats still on the
   single-token path (`dot_ptq1_0_q8_0x2_ptr`, `dot_pq2_0_q8_0x2_ptr` wired into the `dot_fn2`
   table) — and it is a bigger UX win than any decode work. Also check *why* prefill is 2× slower
   per token than decode on this model, since for a hybrid that keeps a fixed-size recurrence the
   two should cost about the same.
2. **Re-run the 6-model benchmark on an idle machine** (`--temp 0 --max-tokens 50 --prompt "Hello"
   --prompt-raw`). This session's pass ran under ~75% CPU load from unrelated processes and the same
   1.7B Q1_0 binary produced 18–33 tok/s across four identical runs, so it cannot be compared with
   the 2026-09-18 table. Interleave against the reference fork when it is available.
3. **Make the KV budget RAM-aware.** `DEFAULT_KV_BUDGET_BYTES` is a constant (1.25 GiB) chosen by
   hand for a 16 GB machine. Available physical memory needs an OS call (`GlobalMemoryStatusEx` on
   Windows; the repo has no FFI today) — decide whether that is worth it, or keep the constant and
   let `HEARTH_CTX_CAP` carry the machine-specific case.
4. **The UI's empty-reply messages are now four distinct cases** (stopped before text / zero tokens
   generated / reasoning only / tokens that decoded to nothing). Reproduce each one deliberately and
   check the wording reads well; the zero-token case should also suggest the context window if the
   prompt was longer than it.
5. **`max_context` vs the allocated ceiling.** The request's window is a separate knob from the
   allocation; the UI seeds its slider from `max_seq_len`, which is now the budgeted ceiling. Confirm
   a 27B request with a small window (e.g. 1024) does not allocate the full 1.25 GiB — today the
   allocation happens at load, so it does, and per-request demand allocation is the follow-up.

---

# Hearth — Session 2026-09-19 (pt 5): prefill weight reuse across prompt tokens

> Prefill re-expanded every weight row once per prompt token. Now a paired kernel (`DotFn2` in
> `pool.rs`) expands each row once for two tokens: **Q2_0 prefill 25.5 → 17.5 ms/tok (31%),
> Q1_0 8–11%, 8B Q1_0 4%**, bit-identical output, decode untouched. An x4 variant regressed 3.2×
> and was reverted. Full table + analysis in BUG_TRACKER "pt 5".

## Next session — things to do/try

1. **Push the paired kernel to PTQ1_0 and PQ2_0** (`dot_ptq1_0_q8_0x2_ptr`, `dot_pq2_0_q8_0x2_ptr`)
   and wire them into the `dot_fn2` table in `model/matmul.rs`. The 27B models are the ones where
   prefill hurts most, and PTQ1_0 is the one hot format still on the single-token path. Follow the
   existing pattern exactly: share the weight-side expansion only, keep accumulation order
   identical, and assert bit-identity against the single-token kernel.
2. **Retry the four-token kernel correctly.** The x4 attempt failed on register pressure, not on the
   idea: keep only *one* accumulator chain per token by folding `w_scale * a_scale` into a single
   per-sub-block FMA (numerically equivalent, no longer bit-identical — check the generated text),
   or expand a whole weight block into a stack buffer once outside the token loop so the loop body
   needs 4 accumulators, not 8. Q2_0's weight side is ~half its uops, so x4 predicts another ~25%
   there. Measure against the pt 5 baseline, not against x1.
3. **Q1_0 is no longer purely issue-bound** (8B Q1_0 only gained 4%). Before adding more kernel
   work, find the wall: try tile sizes other than the fixed 1 MiB `tile_size` in `pool.rs` (the tile
   is exactly Zen 4's L2, leaving no room for the activation row + output stream), and time
   `forward_batch` stage-by-stage (`ForwardTimers` already exists for the decode path but the batch
   path has no timers — add them behind `HEARTH_BATCH_TIMERS`). Attention is only ~2% of prefill at
   840 tokens but ~8% at 4096, so re-check it once long-context prefill is the target.
4. **Prefill parallel-scaling check.** `attention_batch_*_parallel` parallelizes over `n_kv_heads`
   only (8), so a model with fewer KV heads leaves workers idle, and `par_quantize`/head-norm/RoPE
   live in serial loops. `silu_mul_q8` (seq_len × ffn_dim `exp` calls per layer, ~5% of a 840-token
   prefill) is the cheapest of these to parallelize through `pool.par_for`.

## Benchmark hygiene

Absolute prefill ms/tok on this box swings by 3× with unrelated load (12.8 → 41 ms/tok for the same
1.7B Q1_0 build). Interleave before/after runs and compare ratios, not absolute numbers. The
`--service` process holds `target/release/hearth-chat-cli.exe` open, so `cargo build --release` fails
with "Access is denied" until it is stopped (`CARGO_TARGET_DIR=<tmp> cargo build` works around it).

---

# Hearth — Session 2026-09-19 (pt 4): model-driven, adjustable inference settings

> Every inference knob was hardcoded and the models' own `general.sampling.*` + `context_length`
> metadata was ignored. Now `ModelConfig::sampler_defaults()` drives the CLI and the UI seeds all
> sliders from the loaded model; the 27B's frozen 2048 context is gone; min-p/typical-p/seed and
> the load strategy are adjustable end to end. See BUG_TRACKER "pt 4" for the full table.

**One resolution rule everywhere:** a value you set wins *field by field*; every field you leave
unset comes from the model's `general.sampling.*`, then the built-in default. The CLI now tracks
`SamplerOverrides` (all `Option`) instead of a `sampler_explicit` bool, so `--temp 0` alone no longer
discards the model's top-k/top-p/min-p — it used to, and the service already merged per field.
`hearth-chat-cli --help` now prints the full flag/environment reference (it previously had no
`--help` at all and treated the flag as a model path, and the usage text was missing `--prompt-raw`
and `--dump-info`). `README.md` was brought in line: real defaults, the four new flags, the
environment-variable table, and a `hearth-ui` section.

## Next session — things to do/try

1. **Benchmark all 6+ models** against the pt 4 baseline. The only inference-path change is the
   qwen35 KV cap (2048 → 32768, ~1 GB for the 27B); confirm 27B decode tok/s did not regress and
   tune `HEARTH_CTX_CAP` if it did. 1.7B Q1_0 CLI path was re-checked at 41.3 tok/s (baseline).
2. **KV cache type is still F32-only.** `KVStorage::Q8_0` exists and `--kv-f32` is accepted but
   inert. To expose it, add a `kv_q8_0: bool` to `hearth_core::PipelineRequest::TextGen` (hearth-core
   can't depend on hearth-llm, so no KVStorage in the request) and map it in both `generate_text`
   allocs; then surface a UI toggle. Watch for the decode regression seen when Q8 KV was reverted.
3. **Per-model `max reply tokens`.** The UI still defaults to 512 for every model; a ~1 tok/s 27B
   reply takes ~8 min. Consider deriving a default from model size, or showing an ETA next to the
   live tok/s readout.
4. **`enable_think` vs real GGUF templates.** The synthetic Qwen3 template honors it, but a model
   whose bundled `tokenizer.chat_template` never branches on `enable_thinking` will still reason
   when the toggle is off (reasoning is then hidden). The UI now reports the reasoning token count
   so this is at least visible; a next step is scanning the template for the flag and disabling the
   toggle when the model can't honor it.
5. **Models declare only temp/top_k/top_p/min_p.** `repeat_pen` (1.1), `typical_p` (0) and `seed`
   (0 = random) have no metadata source, so they are still effectively fixed defaults: the same
   value for every model unless the user touches them. Check whether Prism GGUFs carry
   `general.sampling.repeat_penalty`/`*.seed` under another key (scan all `general.sampling.*` keys
   with `--dump-info`), and if not, decide whether a per-model heuristic beats one shared value.
6. **Keep `--help` and the README flag table in lockstep.** Both now list the same flags/defaults
   by hand; anything added to `print_usage()` has to be mirrored in `README.md` or the docs go stale
   again — which is exactly how the four missing flags got documented here.

---

# Hearth — Session 2026-09-19 (pt 3): model-dropdown fix + pt 2 stream-cleanliness work

> pt 3: 27B models missing from the dropdown — `default_models_dir()` now walks up from the
> executable's directory (plus sibling checks) so the repo `models/` is found no matter the
> launch CWD; verified by launching from `C:\\Windows\\Temp`. Absolute model paths load fine.
>
> pt 2 (below): found and fixed the real causes of "weird stuff in chat" — literal
> `{% endif %}` in replies, U+FFFD garbage characters, and answers cut off right after the
> reasoning block.

## Fixed this pass

1. **`{% endif %}` / template-tag leak in chat.** The jinja evaluator's `eval_if` rendered the
   winning branch verbatim and stopped at the FIRST `{% else %}`/`{% endif %}`, so any nested
   conditional (the new `{% if enable_thinking %}` inside `{% if add_generation_prompt %}`, and
   any real GGUF template with nesting) leaked its tags into the prompt, which 1-bit models
   echo back. Rewrote it as `parse_if_branches` (nesting-aware if/elif/else split) rendering
   through the full evaluator; `eval_for` now renders loop bodies through the full evaluator
   with a single-message context too. Message-relative conditions (`message['role'] == ...`)
   work inside both. Regression-tested (`test_nested_if_in_template`).
2. **U+FFFD (�) garbage characters.** `decode_token` did per-token `from_utf8_lossy`, so a
   multi-byte character split across two tokens emitted replacement chars mid-stream. Now the
   tokenizer buffers incomplete sequences (`utf8_stream`) and both model cores call
   `flush_stream_decode()` before `Done` (also on cancel) so nothing is lost or leaked across
   generations. Regression-tested (`test_split_utf8_streaming_decode`, `test_flush_stream_decode`).
3. **`</think>` killed generation.** `is_control_token` returned true for every added token
   (`special_tokens` map), so the engine STOPPED as soon as the model emitted `</think>` —
   the visible answer after it never arrived, and the models looked like they "never close the
   block". Now only `<|...|>` and classic BOS/EOS/PAD-form tokens stop generation;
   user_defined tokens like `<think>`/`</think>` stream through. Regression-tested
   (`test_think_tokens_are_not_control_tokens`).

## Verification (per model, live through the hearth-ui service)

- 1.7B Q1_0: regular / reasoning-shown / reasoning-hidden all PASS (no template leaks, no
  U+FFFD, reason channel streams token-by-token).
- 4B Q2_0: all three paths PASS; emoji (multi-byte) round-trips clean — the U+FFFD fix works.
- 27B PQ2_0: reason channel streams (40 events, coherent reasoning).

Note: the 27B CLI "empty output" scare was a red herring — without `--show-think` the CLI's
`ThinkFilter` hides in-think reasoning by design; with the toggle it streams fine.

---

# Hearth — Session 2026-09-19: hearth-ui think streaming + context window

> The web chat UI now streams reasoning and answer on separate channels, and the
> context slider clamps to the loaded model's real KV-cache limit.

## State

| Item | Status |
| --- | --- |
| hearth-ui think streaming | ✅ `reason` SSE channel; drawer auto-opens while thinking, collapses on answer |
| `done` event | ✅ now carries `text` (+ `reason_text`/`reason_tokens`) — previously the UI read `ev.reply`/nothing and appended nothing |
| Context slider | ✅ max 32768, clamped to `model.max_seq_len` on load |
| All 6+ models load/chat via UI | ✅ tested on Bonsai-1.7B-Q1_0 (~58 tok/s through the service path) |

## What changed this session

- `hearth-tokenizer`: Qwen3 synthetic template now honors `enable_thinking` — when false it
  pre-closes `<think>\n\n</think>` (model answers directly); when true it opens `<think>\n\n`
  and lets the model finish the block. `TemplateContext` gained `enable_thinking`
  (`jinja.rs` evaluates `{% if enable_thinking %}`). `apply_template(.., thinking)` maps the
  UI toggle to it. Also fixed the pre-existing `make_mock_tokenizer` compile error (missing
  `gpt2_style`/`gpt2_byte_encoder` fields).
- `hearth-chat-cli/service.rs`: new `StreamSplit` routes each raw token to a reasoning or
  answer channel. Initial state derived from the prompt's trailing think markers
  (`StreamSplit::for_prompt`) — with an open `<think>` in the prompt, Bonsai 1.7B never emits
  `</think>`, so sniffing the token stream alone dead-ends. `partial_close_len` holds back a
  split-across-tokens `</think>` suffix so the tag never leaks into visible text. New events:
  `reason` (fragment) and `done.text`/`done.reason_text`/`done.reason_tokens`. New request
  field `enable_think` (defaults to `show_think`; UI sends both).
- `hearth-ui`: renders the reason channel in a collapsible drawer (`.think`), auto-open while
  generating with no answer yet; stats show `N think`; context slider 256–32768 with
  `syncCtxSlider()` clamping to the model.

## Known model-behavior quirk (not a bug)

Bonsai-1.7B-Q1_0 with an open `<think>` in the prompt sometimes never emits `</think>` and
puts its answer inside the think block. The UI handles this (drawer holds the whole reply,
"internal reasoning" placeholder when hidden), but don't be surprised in CLI testing.
Also seen on 1-bit models: stray `{% endif %}` echo in replies — pre-existing, happens with
the unmodified template too.

## Next session ideas

1. **Benchmark before/after mandate**: nothing perf-relevant changed this session, but the
   service now sends ~2× the SSE lines during thinking — confirm tok/s is unchanged
   (`hearth-ui` path vs CLI path).
2. **True context > 4096 on dense models.** `mod.rs` load caps `config.max_seq_len` at
   `kv_ctx_cap = 4096` (F32 KV cache safety); the 27B hybrid caps at 2048. To go higher:
   cap by free RAM instead of a constant (KV bytes = `2 · n_layers · n_kv_heads · head_dim ·
   pos · 4B` F32, ÷~4 with Q8_0), and make `max_seq_len`-relative position wrap
   (`truncate_left(64)` already exists) safe for RoPE (consider rope scaling on wrap).
3. **Continue-generation for split answers.** When the reply fills `max_tokens` mid-sentence,
   offer a "continue" button that re-queries with the partial answer appended as assistant
   history instead of the current all-or-nothing budget.
4. **`stop` mid-think** leaves the drawer open with partial reasoning — fine, but verify the
   stopped generation's `reason_text` renders correctly after `done.stopped=true`.
5. **UI polish**: persist settings (system prompt, think toggle, ctx) to localStorage; show
   tokens/s live during streaming (currently only at done); auto-scroll currently forces to
   bottom when near-bottom — verify long-think sessions behave.
6. **Multi-turn history with think**: the engine history carries only visible answers
   (`reply`), which is right for quality — but check whether feeding back a compact
   "reasoning summary" helps the 1-bit models stay on topic in long chats.
7. Old idea list (AVX-512 PTQ1_0 kernel, PQ2_0 kernel, batched 27B prefill, GPU 780M, f16
   activations, KV F32-vs-Q8_0 retest) — still open, see previous sessions in BUG_TRACKER.

## Useful commands

```bash
# Watch the reason channel without a browser:
curl -N http://127.0.0.1:8899/api/events > /tmp/sse.txt &
curl -X POST http://127.0.0.1:8899/api/generate -H "Content-Type: application/json" \
  -d '{"message":"Hi","show_think":true,"enable_think":true,"temp":0,"max_tokens":120,"max_context":4096}'

# Template sanity (both must parse; diff the trailing think markers):
./target/release/hearth-chat-cli.exe models/Bonsai-1.7B-Q1_0.gguf --temp 0 --max-tokens 40 \
  --prompt "What is 15+27?" --show-think

cargo test -p hearth-tokenizer -- --test-threads=1
```
