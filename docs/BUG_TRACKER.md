# Hearth Performance Tracker — CPU Final State → GPU Planning

## System
AMD Ryzen 7 8840HS (8C/16T), Radeon 780M iGPU (12 CU RDNA3), 16 GB DDR5, Windows 11. Hearth: 8 workers. Ref: llama.cpp-prism.

### Clippy pass completed (SparkOC, 2026-10-15 night) — CI green, no perf change

Finished buffy's pass: 64 sites across `hearth-llm` (format borrows,
8 signature groupings, 22 hot-loop iterator rewrites), `hearth-diffusion`
(loops, signatures, dead-stub removal), examples and bins. `cargo clippy
--workspace --all-targets -- -D warnings` green on stable 1.99.1, 98 tests
pass, fmt clean. Benchmark gate: post-change sweep vs chart baseline looked
down on Q2_0 models, but an interleaved A/B against a pre-pass binary built
from an isolated worktree agreed within ~3% on all four disputed models —
machine drift, not code. Charts stand.

### Sweep 2026-10-15 — all 14 models, post-Zen4-fix snapshot (SparkOC)

Release build at `50c88bc` (includes the Zen4 per-kernel dispatch fix).
One run each, `--temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw`,
idle machine, warmed up first. ±10–15% run-to-run noise applies; the 6-model
rows replace the old 10-run averages in the README chart, the 8 Bonsai-2 rows
are new.

| Model | tok/s | ms |
| ----- | ----- | -- |
| Bonsai-1.7B-Q1_0 | 47.13 | 1061 |
| Ternary-Bonsai-1.7B-Q2_0 | 24.53 | 2038 |
| Bonsai-4B-Q1_0 | 18.84 | 2654 |
| Ternary-Bonsai-4B-Q2_0 | 12.41 | 4028 |
| Bonsai-8B-Q1_0 | 12.20 | 4098 |
| Ternary-Bonsai-8B-Q2_0 | 6.75 | 7409 |
| Ternary-Bonsai-2-2B-PTQ1_0 | 16.31 | 3065 |
| Ternary-Bonsai-2-2B-PQ2_0 | 17.79 | 2811 |
| Ternary-Bonsai-2-4B-PTQ1_0 | 6.63 | 7542 |
| Ternary-Bonsai-2-4B-PQ2_0 | 7.47 | 6692 |
| Ternary-Bonsai-2-9B-PTQ1_0 | 4.23 | 11832 |
| Ternary-Bonsai-2-9B-PQ2_0 | 4.80 | 10426 |
| Ternary-Bonsai-2-27B-PTQ1_0 | 1.35 | 36919 |
| Ternary-Bonsai-2-27B-PQ2_0 | 1.53 | 32626 |

Raw per-model lines: [`bench_results/sweep-2026-10-15.log`](../bench_results/sweep-2026-10-15.log).

### Session 2026-10-15 — the Zen 4 AVX-512 gate could never fire, so the slower kernel was selected everywhere

**Trigger:** benchmarking the Ternary Bonsai 2 file set. Every PTQ1_0 model ran at roughly *half the
effective bandwidth* of its PQ2_0 sibling (4B: 6.6 vs 9.2 tok/s, 9B: 4.0 vs 6.3) even though
`gguf_info.py` showed the two files carry **identical block counts** (69.9M for both 9B files) — so
per-token cost should have been equal or better for the 1-bit format, not 1.6× worse.

**Root cause.** `cpu_is_amd_zen4()` (`crates/hearth-quant/src/lib.rs`) read CPUID leaf 1 EAX bits 8–11
— the *base* family field — and compared it to 25. Modern AMD always reports base family **0xF** with
the real family in the *extended* field (bits 20–27), so Zen 4 arrives as base 15 + ext 0xA = **0x19**,
and the predicate could never be true. `avx512_full_width()` therefore always returned `true` on this
8840HS and every kernel gated on it took the AVX-512 path. On Zen 4's double-pumped 256-bit datapath
that is the *slower* choice for the LUT-style kernels: measured PTQ1_0 **1.28 GB/s (AVX-512) vs
1.76 GB/s (AVX2)**, +34%.

**Fix, in two parts — the right answer turned out to be per-kernel, not a blanket ban.**

| # | Change | Why |
|---|--------|-----|
| 1 | `cpu_is_amd_zen4()` reconstructs the effective family (`base == 0xF ? base + ext : base`) and compares to **0x19**. `avx512_full_width()` is now correctly `false` here, so PTQ1_0 dispatches to AVX2 | PTQ1_0's AVX-512 kernel is strictly slower on this part |
| 2 | The Q2_0 VNNI kernels stopped requiring `avx512_full_width()` and are gated on the VNNI feature bits alone (`dot_q2_0_q8_0_ptr`) | Fixing (1) alone would have pushed Q2_0 onto the *slower* AVX2 LUT kernel — but VNNI **wins** on Zen 4 because `vpdpbusd` cuts front-end work enough to pay for the half-width datapath |

**Kernel measurements (single thread, `benchk`, warm).**

| format | tensor set | dispatched | avx2 | avx512 | scalar |
|--------|-----------|-----------|------|--------|--------|
| PTQ1_0 | 4B `blk.0` aggregate | **1.70** (avx2) | 1.72 | 1.28 | 0.67 |
| PQ2_0 (Q2_0 kernels) | 4B `blk.0` aggregate | **3.09** (vnni256) | 2.91 (avx2 LUT) | 3.10 vnni256 / 3.02 vnni512x2 | — |

**Verification.** `benchk` now prints the CPUID decode *and* the kernel `dot_ptq1_0_q8_0_ptr` actually
selects (new `ptq1_0::kernel_name()`, which reuses the real predicates rather than copying them):
`base_family=15 ext_family=0xA real_family=0x19`, **`dispatch selects: avx2`**. Two side findings: the
`scalar` kernel really is ~0.67 GB/s, so an earlier "dispatched = 0.66 = scalar" reading was a
cold-clock artifact on the first-measured variant rather than a dispatch failure; and the bench needed a
time-based warmup before per-variant numbers were comparable at all.

**Model results** (`--temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw`, 2nd (warm) run):

| model | before | after | Δ |
|-------|--------|-------|---|
| Ternary-Bonsai-2-2B-PTQ1_0 | 15.6 | **19.1** | +22% |
| Ternary-Bonsai-2-4B-PTQ1_0 | 6.6 | **7.7** | +17% |
| Ternary-Bonsai-2-9B-PTQ1_0 | 4.0 | **4.8–5.0** | +20–25% |
| Ternary-Bonsai-2-27B-PTQ1_0 | 1.23 | **1.41** | +15% |

The six mandate models were re-run and show no regression (all inside the ±4–5% single-run spread pt 6
documented): 1.7B Q1_0 45.0 (sess. baseline 45.9), 1.7B Q2_0 25.1, 4B Q1_0 21.1 (21.3), 4B Q2_0 11.6,
8B Q1_0 12.0 (12.2), 8B Q2_0 6.5. The Q2_0 models are unchanged **by design** — VNNI is selected both
before and after fix (2), which is exactly what part 2 was for.

**Correctness.** All 33 `hearth-quant` tests pass; `cargo clippy -p hearth-quant --all-targets -- -D
warnings` and `cargo fmt --all --check` are clean. Generated text is **byte-identical** to the pre-fix
binary on all four PTQ1_0 models (AVX2 and AVX-512 agree exactly), so this is a pure speed change.

**Also fixed:** `scripts/gguf_info.py` listed PTQ1_0 as dtype **45** with an **18-byte**/128-element
block. The Rust parser uses dtype **143** and a **28-byte**/128-element block; PQ2_0 is **142**, not 44.
The script's dtype census was silently wrong for the new formats (it reported 71.4 MB where the file
really holds 111.2 MB), which would have hidden exactly the kind of layout mistake this session chased.

**Follow-up in the same session — making the same mistake impossible to repeat.** The bug was a
predicate that could never be true and that nothing in the suite could see, so both halves got fixed:

- `cpuid_family(eax)` and `cpuid_is_amd(...)` are pure `const fn`s, and the policy is now a separate
  `avx512_full_width_for(is_amd, family, has_avx512f, has_avx512bw)`. `avx512_full_width()` only feeds
  them detected values, so the decision is testable for parts this machine is not.
- **33 → 40 tests.** `raw_family_field_can_never_match_19h` pins the bug itself: a real Zen 4 reports
  `0xF` in the 4-bit family field, so any comparison against `0x19` — or the decimal 25 — is false on
  every Zen 4 part. Around it, a decision matrix: Zen 3 (19h, no AVX-512), Zen 4 (19h, loses),
  Zen 5 (1Ah, keeps), Intel (keeps), an *Intel* part reporting 19h (must keep — the check is
  vendor-qualified), and the missing-feature cases. Writing the matrix caught a factual error in the
  test data itself: **Zen 3 is family 19h too**, so family alone cannot separate Zen 3 from Zen 4 — the
  feature bits do.
- **Dispatch is now self-reporting.** `ptq1_0::kernel_name()` was joined by `q2_0::kernel_name()` and
  `q1_0g128::kernel_name()`, each reusing the real predicates rather than copying them, plus
  `hearth_quant::selected_kernel(dtype)` and `dispatch_report(dtypes) -> DispatchReport`
  (`cpu`, `cpu_family`, `avx512_full_width`, per-format `kernels`). Before this, only PTQ1_0 could say
  which kernel it had chosen; Q2_0 and Q1_0_G128 could not, which is exactly how a 34% loss hid.

Load banner, dense and hybrid paths alike (same `·` separator as the `[kv]` line):

```
[kernel] PTQ1_0=avx2 · zen4 · avx512_full_width=false
[kernel] PQ2_0=vnni512x2 · zen4 · avx512_full_width=false
```

And machine-readable for the UI, in the service's loaded event at `event.model.kernels`:

```json
{"avx512_full_width": false, "cpu": "zen4", "cpu_family": "0x19", "ptq1_0": "avx2"}
```

Verified by driving `hearth-chat-cli --service` with line-JSON (`load`/`unload`/`shutdown`) and parsing
stdout, so the UI can report the live dispatch without parsing log lines. `cpu` distinguishes Zen 3 from
Zen 4 by whether AVX-512 is actually present, not by the shared 19h family.

**Files touched:** `crates/hearth-quant/src/lib.rs` (family decode + `DispatchReport` + tests), `q2_0.rs`,
`ptq1_0.rs`, `q1_0g128.rs`;
`crates/hearth-llm/src/model/mod.rs` (banner + `dispatch()` accessor), `model/qwen35.rs` (hybrid path),
`src/bin/hearth-chat-cli/service.rs` (`kernels` JSON), `examples/benchk.rs` (rewritten as a head-to-head
kernel bench that also reports the ground-truth dispatch); `scripts/gguf_info.py`. **Nothing committed**
— the tree also holds another agent's in-progress edits (see RESUME).

**Correction and a bigger finding (same session, after the above).** I first wrote that "PTQ1_0 and
PQ2_0 have no x2 (paired) kernel". **That was wrong about PQ2_0**: it shares Q2_0's paired kernel
(`matmul.rs`: `Q2_0 | PQ2_0 => Some(dot_q2_0_q8_0x2_ptr)`). Only PTQ1_0 lacked one. Then the
replacement turned out to be inert for a more interesting reason.

`dot_ptq1_0_q8_0x2_ptr` was written (weights decoded once into `wvals[128]`, two activation rows, and
**bit-identical to two single calls** — pinned by `paired_matches_two_single_calls`, which also asserts
real magnitude rather than merely "not zero", after a first fixture attempt produced an exactly-zero
dot that would have let a stub pass). 33 → 41 tests. But **no model can reach it**: `DotFn2` is consumed
only by `matmul_batch`, `matmul_batch` is called only from the dense path (`model/mod.rs`), and every
PTQ1_0 model in the catalog is a `qwen35` hybrid. The hybrid prefill is explicitly token-by-token
(`qwen35.rs`: *"Prefill: token by token (GDN recurrence is inherently sequential)"*) and dispatches
single-token kernels only (`qwen35.rs` x1 tables). So the wiring was **reverted** rather than left as a
no-op that reads like a prefill speedup; the kernel stays as the tested, currently-unreferenced
primitive, with its doc comment saying so and pointing at where it belongs. **No performance claim.**

**What that uncovered — the hybrid prefill has no weight reuse at all.** It re-streams every weight row
once per prompt token, i.e. *identical memory traffic per token* to decode. That reframes the earlier
number: "~1.8–2.3 s per *prompt* token vs ~0.9 s per *decoded* token" cannot be explained by matmul work
(it should be about equal), so the 2× is either attention over the growing prefix, first-token/lazy-init
effects, or a measurement artifact — **currently unexplained and possibly free speed**.

**Where the real prefill win is:** not another kernel but a *chunked* hybrid prefill — carry two tokens
through the dense projections per weight pass while keeping the GDN recurrence sequential. The x2 kernel
is exactly the primitive that needs (one weight row, two Q8 activation rows, bit-identical). That is a
hybrid-forward redesign, needs its own design note, and is **not** claimed here.

**Then the measurement killed two of my own claims.** `HEARTH_Q35_TIMERS=1` prints per-stage ms/token per
phase, so prefill and decode are directly comparable:

| model | prompt | prefill ms/token | decode ms/token |
|-------|--------|-----------------|-----------------|
| 27B PTQ1_0 | 16 words | **635** | **722** |
| 27B PTQ1_0 | 1 word | 6934 (first token) | 750 |
| 4B PTQ1_0 | 32 words | 124 | 117 |
| 4B PTQ1_0 | 128 words | 144 | 229 |

1. **"Prefill is ~2× slower per token than decode" is FALSE.** On the 27B, prefill is 635 ms/token
   against decode's 722 — *faster*, not slower. That premise had been in RESUME since pt 6 and came from
   a run under ~75% external CPU load with a cold file cache. The stage split is stable across lengths
   (gdn/attn 54–58%, ffn 34–38%, head 7–12%), i.e. weight-streaming matmul dominates exactly as
   expected when there is no weight reuse. Prefill per-token *does* rise with length on the 4B
   (marginal 113 ms/tok from 8→32, 151 ms/tok from 32→128) — attention over the growing prefix is real
   but modest.
2. **The first-token excess is storage I/O, not page faults and not clock ramp.** A page-touch prefault
   of every mapped tensor was written and **measured to do nothing**: 4B first token 173.0 ms with it
   vs 173.4 ms without, while it added **259 ms** to load at 4B and **2402 ms** at 9B — a net regression,
   so it was reverted (`git diff` on `mod.rs` is empty again). Equally, my earlier reading that
   `HEARTH_PIN_WEIGHTS=1` "removed 1044 ms of the 4B first token while leaving the arithmetic identical,
   which identifies this as page-fault cost" was **wrong**: pinning moves the *first read of the file*
   into load, and I only measured the prefill phase, not the total. With a warm page cache there is
   essentially no first-token penalty at all (4B: 173 ms vs ~121 ms steady), and the 27B's 6934 ms first
   token against 635 ms steady is the 5.9 GB file being read for the first time.

**Also measured, and a hazard:** `HEARTH_PIN_WEIGHTS=1` on the 27B **died with no output** on this 16 GB
box while another agent's models were resident — copying 5.9 GB plus a 1.25 GiB KV cache is not always
survivable, and the failure is silent (the process is killed, nothing is printed). The plain
(non-pinned) path is the safe one.

**Net:** no performance change shipped from the prefill investigation. Its value is negative-result
clarity — the 2× premise is retired, the first-token stall is storage-bound and inherent, and the only
real prefill lever left is the chunked/weight-reuse redesign (≈2× on the projections, not on attention).

### Peer-review round (same session) — verified fixes, one new defect, one corrected research claim

Read-only review of the other agent's downloader work (`8327093`, `86b0bd1`), then **behavioural**
verification of every claim through the real `hearth-ui` HTTP API (catalog override + a local server
that logs `Range`), rather than by reading the code.

**Three defects I reported earlier are fixed, and the fixes are measured:**

| finding | before | after (measured) |
|---|---|---|
| `curl_available()` spawned `curl --version` on every `/api/downloads` poll, and the frontend polls at 1 Hz | 68/67/71 ms per poll, +1 process | **median 0.56 ms**, p95 0.76 ms, n=25 (`OnceLock`) |
| `finish()` emitted the *expected* size as `downloaded_bytes` on failure | 100 % bar beside an error | `state=error`, `downloaded_bytes=100` of 5000, `"incomplete: got 100 of 5000 bytes"` |
| `snapshot()` read only the final filename, so a resumable `.part` read `idle`/0 | `idle`, 0 bytes | `partial`, real `.part` bytes (50 of 100) |

**Also verified positively** (neither was covered by a test through the Manager before): a 206 resume
sends exactly one `Range: bytes=50-` and publishes a byte-exact file; a server that ignores `Range`
(answers 200) is detected *before any byte is written*, the `.part` is reset and re-fetched, and the
published file is still byte-exact — no corruption. `http.rs`'s 7 tests pass; `/`, `/api/models`,
`/api/downloads` all 200.

**One new defect, reproduced twice — a full-size `.part` can never complete.** With `<name>.gguf.part`
holding exactly `size_bytes` and a server answering **416** to `Range: bytes=<size>-`, pressing Download
gives `state=error error="HTTP 416" exists=false` and the `.part` stays; pressing again gives the
identical error. It never self-heals — the file must be deleted by hand, while the UI shows 100 % next
to an error. Cause: `start()`'s "already complete" shortcut uses `on_disk()`, which reads only the
**final** name, so a full-size `.part` is not recognised; and `http::get` treats 416 as fatal while
`run_http` retries only on `range_ignored`. Reachable by cancelling at the last byte (cancel keeps the
`.part` by design), a failed rename, or any catalog `size_bytes` larger than the CDN serves. Fix: either
"416 while resuming = range already satisfied → publish", or check `part_len()` in `start()`.

**`http.rs` notes (code reading, no repro):** `to_socket_addrs().next()` takes only the first address,
with no fallback across A/AAAA records where curl tries them all — latent while catalog URLs are
`https://` (still curl), live once TLS routes them here. Cosmetic: `read_head` returns a vestigial empty
`Vec`, its cap check is off by one, and nothing tests that `Range` survives a redirect (it does).

**Corrected a research claim.** "Speculative decoding from the 2B draft is bit-identical by
construction" is not automatic, and the win is ~2×, not ~10×: losslessness is a property of the
*acceptance rule* (the greedy/argmax form is exact; the sampling form needs rejection plus residual
correction over the post-truncation distribution, and the stateful repeat penalty must be handled), and
the target forward pass is never skipped. Measured economics: **2B PTQ1_0 decodes at 50.8 ms/tok
(18.15 tok/s), prefill 81.4 ms/tok** — 14× cheaper per token than the 27B's ~722 ms, so with acceptance
`a` and draft length `k` the expected accepted tokens per target forward is `(1-a^(k+1))/(1-a)` at a cost
of `1+0.07k` target-equivalents: ≈1.8× at a=0.6, ≈2.2× at a=0.7, ≈2.6× at a=0.8 (k=4). **The acceptance
rate is the unmeasured quantity the whole plan rests on.** The prerequisite is **green**: the 2B and 27B
tokenizers are identical (same 248320-token list, same 247587 merges, same `token_type`, same
bos/eos/padding 248044/248046/248044). Two non-blocking differences: `tokenizer.chat_template` differs
(render with the target's template, feed both the same ids) and `prism.hadamard.version` is 2/512 on the
2B vs 1/1024 on the 27B — the two are *not* in the same weight basis, which is why acceptance must be
measured rather than argued. Hidden cost for the design note: 3 of every 4 `qwen35` layers are
recurrent, so a rejected draft token must rewind conv+SSM state, not just a KV cache.

**Files touched:** docs only — no source change from this review round.

### Clippy clear-up (same session) — 36 → 31 sites, and one real bug behind a lint

`cargo clippy -p hearth-llm --all-targets -- -D warnings` was failing on commits going back several
sessions, which is why the team's CI was red. **Counting lesson first:** a single `--all-targets` run
reports each site once and then appends two `could not compile ... due to N previous errors` summary
lines, so 33 `error:` lines == **31 sites**; summing a `lib` run and a `lib-test` run double-counts the
same source and produced the "68 errors" figure in the room. The real starting count was 36.

**Five sites fixed in `crates/hearth-llm/src/model/mod.rs` (7 → 2), and one of them was a genuine bug:**
`test_gpu_decode_with_real_model` called `drop(gpu)` where `gpu` is a **reference** (`gpu_model.gpu
.as_ref().unwrap()`), so the drop was a no-op that read like a resource release. The pool lock it looked
like it was releasing was already a temporary inside the preceding statement, so the line was removed
outright rather than restated. Also `manual_checked_division` in the timers display (now `checked_div`,
identical output) and three test-only index loops (`cpu_q`/`cpu_k`/`cpu_v` rows), rewritten as
`iter_mut().enumerate().take(n_rows)` — which unlike the original also cannot panic if a row count ever
exceeded the allocation.

**No benchmark gate applied to this slice:** all five sites are in a `#[test]` function or in the
timers-display path, not in inference. Verified with `cargo test -p hearth-llm --lib` (10 passed) and a
real 1.7B Q1_0 run (20 tokens, 34.48 tok/s, coherent output) after the edit.

**Remaining 31 sites** (`ops.rs` 21, `qwen35.rs` 4, `mod.rs` 2, `pool.rs` 2, `matmul.rs` 2) are
`needless_range_loop` in the SIMD helpers plus eight `too_many_arguments` signatures. Both classes are
ABI/codegen-relevant, so they need before/after benchmarks on all six models before merging; the plan
and the exact line list are in RESUME item 0.`#[allow]` is not an option — AGENTS.md bans it.

### Session 2026-09-19 pt 6 — the 27B chat was unusable: KV cache sized from the model's *declared* context

**Trigger:** "the 27b q1 on html UI chat" returned only *"The model returned no visible text (28 tokens
 generated). Try a different model or lower Max reply tokens."*

**Two root causes, both mine from pt 4.**

| # | Cause | Evidence | Fix |
|---|-------|----------|-----|
| 1 | The hybrid KV cap went 2048 → **32768** while the model declares **262144**. The 27B's KV is **128 KiB per token** (16 attention layers × 4 kv heads × head_dim 256 × 2 × 4 B), so 32768 = **4.3 GiB** and the declared context would be **32 GiB**, on a 15.3 GB machine with ~6 GB of weights. The engine ran, but the box paged: a 34-token chat prompt took **106 s** and the UI looked dead | `head_dim: 256, n_kv_heads: 4`. `Get-Process` free RAM 0.9 GB of 15.3 GB; `[34 tokens, 106583 ms, 0.3 tok/s]` | new `model/kv_budget.rs`: context is now sized by a **byte budget** (1.25 GiB) instead of by the declared context |
| 2 | `RunStats::tokens` counted the **prompt**: `total_tokens` started at `input_ids.len()`. So "28 tokens generated" was really *28 prompt tokens and zero generated* — the UI blamed the reply budget for a generation that never produced a token | 1.7B control run: a 2-token reply reported `tokens: 33` | `RunStats { tokens (generated), prompt_tokens }`; tok/s from generated only |

**KV policy (`kv_budget.rs`).** `ctx_cap(declared, kv_layers, kv_heads, head_dim, legacy_cap)`:
budget 1.25 GiB ÷ bytes-per-token, lowered by an optional *legacy* cap, never above what the model
declares, with `HEARTH_CTX_CAP` still overriding in either direction.

| Model | KV/token | Before | After | KV cache |
|-------|----------|--------|-------|----------|
| Bonsai 1.7B | 224 KiB | 4096 | **4096** (unchanged) | 896 MiB |
| Bonsai 4B / 8B | 288 KiB | 4096 | **4096** (unchanged, 1.125 GiB) | 1.125 GiB |
| Ternary Bonsai 2 27B | 128 KiB | 32768 (pt 4) / 2048 (before) | **10240** | 1.25 GiB |

Dense models keep `DENSE_CTX_CAP = 4096` (the context they are benchmarked at) so nothing that already
worked moves; only the hybrid, which had no historic cap worth keeping, follows the budget. The loader
now prints `[kv] context N tokens · X per token · Y KV cache` and the service reports `kv_bytes`, which
the UI shows in the model note — the RAM cost of a context is visible before chatting.

**Honest accounting.** `RunStats.tokens` is generated tokens; `prompt_tokens` is new. CLI prints
`[N new tokens (P prompt), ms, tok/s]`; the service/UI forward `prompt_tokens`. The UI's empty-reply
messages now name the real case instead of always blaming the reply budget: stopped before any text
(with the per-prompt-token cost when it is slow), zero tokens generated, reasoning-only, or tokens that
decoded to nothing.

Why the old default stalled rather than merely ran slowly: `KVCache::new` builds
`vec![0.0f32; …]` for K and V, so the whole cache is **touched at load**. At 32768 that is 4.3 GiB of
zero-fill plus page-file traffic on a box already holding 6 GB of weights — the stall was the
allocation, before a single token.

**Verified (release build).** Unit tests: dense 4096 preserved for 1.7B and for the 288 KiB/token 8B;
27B → 10240 (1.25 GiB) instead of 32 GiB; absurd KV floored at 512. Live 27B through the UI service:
load returns `max_seq_len: 10240, kv_bytes: 1342177280` (1.25 GiB) and a chat streams `Hello` with
`done: tokens=1, prompt_tokens=33, elapsed 59.4 s` — the UI now renders the answer and reports
`1 new tok · 33 prompt · 59.4 s · 0.02 tok/s` instead of an empty bubble blaming the reply budget.
CLI banner: `[kv] context 10240 tokens · 128 KiB per token (16 attn layers) · 1.25 GiB KV cache`.

**Still slow, and now measurable:** the 27B spends **~1.8-2.3 s per *prompt* token** (33-token chat
prompt = 60-76 s before the first token) against ~0.9 s per *decoded* token. Prefill — not decode,
and not memory — is the remaining lever, i.e. pt 5's item 1 (`dot_ptq1_0_q8_0x2_ptr` /
`dot_pq2_0_q8_0x2_ptr`), which is the one hot format still on the single-token path. No kernel or
forward-path changes here, and dense contexts are unchanged at 4096.

**Follow-up — the idle service was burning 4 cores.** Reported as "sitting there it's at 50% CPU".
The worker pool's idle branch spun `SPIN_BUDGET` iterations and then called `thread::yield_now()`
— which returns immediately when nothing else is runnable — so it never actually blocked. Every
worker in the pool therefore stayed at ~100% of a core for as long as the process lived. Measured on
the 27B service (12 workers, 18 threads): **414% of one core while completely idle** (~26% of this
16-thread machine with other apps at 75% load; on an otherwise idle box the workers take whatever is
free, which is the ~50% Task Manager showed). The dense model path was just as bad.

Fix (`pool.rs`): an idle worker now **blocks on a condvar** instead of spinning forever, with the
hot path left syscall-free — `dispatch()` bumps the generation and only pays for a `notify_all()`
when `parked > 0`, so during generation the workers still catch each dispatch by spinning and never
touch the kernel. Two details that matter: the worker re-checks the generation *under the wait lock*
before sleeping (a dispatch between its first read and the sleep cannot be lost), and `spins` stays at
the budget across a wait so a spurious wake goes straight back to sleep instead of re-spinning —
otherwise the 100 ms backstop wake-up would itself cost CPU. `Drop` notifies too, so unload/exit is
prompt rather than waiting out the timeout.

**Verified.** Idle CPU with the 27B loaded: **414% → 0.0% of one core**; same with the dense 1.7B
loaded. Throughput unchanged: 6 interleaved runs of each binary on 1.7B Q1_0 (50 tokens) gave
old mean 31.6 / max 33.9 tok/s vs new mean 31.9 / max 35.2 tok/s on this loaded box. A generation
issued after 12 s of idle (every worker parked) streams tokens immediately, confirming the wake path.

**Benchmark caveat.** The 6-model pass ran under ~75% CPU load from unrelated processes (the box's
absolute numbers swing >3× with load — pt 5's note). The 1.7B Q1_0 binary measured 18.1, 20.8, 25.7
and 32.8 tok/s across four identical runs, so these numbers are load-dominated and not comparable to
the 2026-09-18 table; the reference fork binary was not present to interleave against. The dense
models' context and KV size are provably unchanged (`kv_budget` tests), and nothing in the forward
path was touched, so no regression is expected — but the mandate's before/after is outstanding and
should be redone on an idle machine.

### Session 2026-09-19 pt 5 — prefill now reuses each weight row across two prompt tokens

**Trigger:** prefill ran at ~13.7 ms/tok (1.7B Q1_0) — only 1.5× faster per token than single-token
decode despite ~800× the arithmetic intensity. Prefill re-read and re-expanded every weight row
once per prompt token, even though a row's unpacking depends only on the weights, not the activation.

**Change.** `par_dot_rows_batched` takes an optional paired kernel (`DotFn2`). For a prompt it now
expands each weight row once and applies it to two tokens: ~64 uops per 128 elements → ~52, and the
weight bytes move from L2 once per pair instead of once per token. Q1_0_G128 keeps its shuffle-based
expansion (`dot_q1_0g128_q8_0x2_ptr`); Q2_0 replaces its `Q2V_I16` table gathers with one
per-pair expansion (`dot_q2_0_q8_0x2_ptr`). Accumulation order is unchanged, so both tokens are
**bit-identical** to the single-token kernel — asserted by `test_dot_*_q8_0x2_matches_single` over
n = 128/256/2048/6144. Decode is untouched (the paired path needs `seq_len > 1`); the odd tail token
falls back to the single kernel.

Prefill ms/tok, 420–840-token prompts, warm:

| Model | Before | After | Gain |
|-------|--------|-------|------|
| 1.7B Q1_0 | 14.1 | 12.9 | **8%** |
| 1.7B Q2_0 | 25.5 | 17.5 | **31%** |
| 4B Q1_0 | 34.2 | 30.5 | **11%** |
| 4B Q2_0 | 61.3 | 41.7 | **32%** |
| 8B Q1_0 | 60.6 | 58.2 | **4%** |
| 8B Q2_0 | 111.9 | 76.4 | **32%** |

**Why Q2_0 gains 4× more than Q1_0.** The two kernels are bound differently. Q2_0 expands each
weight block through eight `Q2V_I16` table gathers, which is roughly *half* its per-token uops —
sharing that half across two tokens predicts a ~32% uop cut, and that is exactly what is measured,
so Q2_0 prefill is issue-bound. Q1_0's shuffle-based expansion is only ~36% of its per-token uops
(~18% once paired), and its gain shrinks further at 8B (4%), where the weight stream is large enough
that the kernel is partly memory-bound. Decode tok/s was re-checked on 1.7B Q1_0 and is unchanged.

**Regression that was reverted:** see "What Didn't Work" — a four-token x4 kernel regressed 3.2×.

**Note for benchmarking:** absolute ms/tok moves by more than 3× with machine load (an idle box gave
1.7B Q1_0 = 12.8 ms/tok; the same build under 77% load from unrelated processes gave 41 ms/tok).
Interleave before/after runs rather than comparing across the session; the 8%/31% ratios above were
re-confirmed by interleaved runs under load (7–12% and 31%).

### Session 2026-09-19 pt 4 — inference settings were hardcoded; now model-driven + adjustable

**Trigger:** the web chat showed *"The model used all 28 tokens on internal reasoning…"* for the
27B. Investigation found the whole stack ignored each model's own metadata and hardcoded one
sampling/context set for every model. Nothing was wrong with the model — the knobs were fixed.

| # | Hardcoded setting | Where | Fix |
|---|-------------------|-------|-----|
| 1 | `general.sampling.{temp,top_p,top_k,min_p}` ignored — every GGUF ships them, Hearth always ran 0.7/0.9/40 | `ModelConfig` had no sampling fields | parsed into `ModelConfig` + `sampler_defaults()`; CLI uses them unless a flag is passed; service resolves omitted fields against them; UI seeds all sliders on load ("Reset to model defaults" button) |
| 2 | 27B context frozen at **2048** (`kv_ctx_cap`) and `max_context_len` silently dropped | `qwen35.rs` (`generate_text`, `from_gguf`) | cap now 32768 by default, `HEARTH_CTX_CAP` override; the request's context window truncates the prompt and bounds the decode loop |
| 3 | Load strategy hardcoded `CpuOnly`; no way to pick GPU/pin from the UI | `service.rs` `Cmd::Load` | `load` takes `device` (`cpu\|gpu\|gpu_split\|offload`), `gpu_layers`, `pin_weights`; UI has a Device select + pin toggle |
| 4 | `min_p`, `typical_p`, `seed` existed in `SamplerConfig` but were unreachable (no CLI flag, no protocol field, no UI control) | CLI / service / UI | `--min-p`, `--typical-p`, `--seed`, `--max-context` flags; service fields; UI sliders + seed input |
| 5 | Context slider default 8192 / max 32768 regardless of model | `index.html` | seeded from the loaded model's `max_seq_len` |
| 6 | An all-reasoning reply reported `reason_tokens` only when reasoning was *shown*, so hidden reasoning looked like "28 mystery tokens" | `service.rs` | `reason_tokens` is always reported (text still only when shown); UI names the count instead of guessing |
| 7 | `--kv-f32` implied Q8_0 was the default; the engine is F32-only | CLI help/messages | messaging corrected (flag kept for compatibility) |

**Verified (release binaries, live through the service):** 4B Q1_0 reports its own 0.5/0.85/20·ctx
4096 (`sampling_from_model:true`); 27B PTQ1_0 reports ctx **32768** + 1.0/0.95/20 and streams
reasoning token-by-token at ~1.07 tok/s (`reason_tokens:8`); with reasoning off it answers `OK`
directly; the settings drawer shows the model's values on load. 1.7B Q1_0 CLI benchmark unchanged
(41.3 tok/s vs 41.2 baseline). No inference-hot-path changes outside the qwen35 context wiring.

**Follow-up — one merge rule, one help text, docs that match.** The CLI had resolved flags
all-or-nothing (`sampler_explicit`): passing `--temp 0` alone silently threw away the model's
top-k/top-p/min-p and ran on the built-in 0.7/0.9/40 profile, while the service already merged per
field. The CLI now uses `SamplerOverrides` (all `Option`) and merges field by field, so the rule is
identical on both paths: *flag wins, else the model's `general.sampling.*`, else the built-in
default*. Locked by `hearth-chat-cli` unit tests `no_flags_keeps_model_defaults` and
`one_flag_overrides_only_its_own_field`. There was also no `--help` at all — `--help` was parsed as
a model path — and the usage text omitted `--prompt-raw`/`--dump-info`; `print_usage()` now covers
every flag plus `HEARTH_THREADS`/`HEARTH_CTX_CAP`/`HEARTH_MODELS_DIR`/`HEARTH_PIN_WEIGHTS`. Live
check (4B Q2_0, debug bin): no flags → `temp=0.5 top_k=20 top_p=0.85`; `--temp 0` → `temp=0` with
`top_k=20 top_p=0.85` intact; `--temp 0 --top-k 7` → only those two change. `README.md` was
rewritten to match (real defaults, the four new flags, an environment-variable table, a `hearth-ui`
section, 27B throughput, and a link that actually resolved — it pointed at `BUG_TRACKER.md` in the
repo root). One duplicated `## Current Benchmark` heading left in the working tree was collapsed.

### Session 2026-09-19 — hearth-ui think streaming + context clamp (no perf changes)

**Bug fixed — "Show <think> reasoning" showed nothing in the web chat.** Six stacked causes found across the stack (template → engine → service → UI):

| # | Bug | Symptom | Root cause / fix |
|---|-----|---------|------------------|
| 1 | Template ignored the thinking flag | with the toggle ON the model never opened its own `<think>` | Qwen3 synthetic template always injected `<think>\n\n</think>` (pre-closed) regardless of `thinking`; now gated on new `TemplateContext::enable_thinking` |
| 2 | Service buffered think text until EOF | reasoning arrived (if at all) only at the end, and the literal guard matched nothing | `ThinkGuard` suppressed think text when the toggle was OFF but had no channel for streaming it ON; new `StreamSplit` (seeded from the prompt's trailing think marker) routes tokens to `reason` vs `token` events |
| 3 | UI dropped the final text | the bubble stayed empty; only stats appeared | UI read `ev.text` from `done` but the service sent `reply`; service now sends `text` (+ `reason_text`/`reason_tokens`), UI renders reasoning in a collapsible drawer |

Also: context slider capped at 8192 with no relation to the model; now 32768 max and clamped
to the loaded model's KV-cache limit on load. No inference-path changes; tok/s unchanged
(1.7B service path ≈ 58 tok/s before and after).

### Session 2026-09-19 pt 3 — 27B models missing from the model dropdown

**Bug fixed — the two Ternary-Bonsai-2 27B GGUFs never appeared in the UI model list.**

`default_models_dir()` checked `HEARTH_MODELS_DIR`, then CWD `models/`, then fell back to
`%APPDATA%\\hearth\\models`. When hearth-ui is launched outside the repo (double-click,
shortcut — CWD is `target\\release` or wherever), the project's `models/` was never found and
the dropdown silently showed the AppData set instead (no 27B files there). Fix: after the CWD
check, walk up from `current_exe()`'s directory (up to 6 levels, also testing each level's
sibling `models/`) so the repo's model folder is found regardless of launch CWD, and portable
`exe + models/` layouts work too. Verified by launching from `C:\\Windows\\Temp`: the API now
returns `dir: C:\\github\\hearth\\models` with all 8 GGUFs; absolute-path loads work.

### Session 2026-09-19 pt 2 — stream-cleanliness review (three more root causes)

Follow-up review of the same feature; each fixed with a regression test in `hearth-tokenizer`:

| # | Bug | Symptom | Root cause / fix |
|---|-----|---------|------------------|
| 4 | Jinja evaluator couldn't nest | literal `{% endif %}` echoed at the top of replies (1-bit models mirror prompt noise) | `eval_if` pushed the branch body verbatim and stopped at the first `{% else %}/{% endif %}`; rewritten as nesting-aware `parse_if_branches` + full-evaluator rendering; `eval_for` bodies likewise (`test_nested_if_in_template`) |
| 5 | Per-token UTF-8 lossy decode | U+FFFD (�) at the end of replies when a multi-byte char straddled two tokens | tokenizer now buffers incomplete sequences (`utf8_stream`); model cores call `flush_stream_decode()` before `Done` (incl. cancel) (`test_split_utf8_streaming_decode`, `test_flush_stream_decode`) |
| 6 | `</think>` treated as a stop token | answers cut off right after reasoning; models seemed to "never close" the block; 27B pre-closed-think path produced nothing | `is_control_token` returned true for ALL added tokens; narrowed to `<\|...\|>` + classic EOS/BOS forms so user_defined tokens (`<think>`, `</think>`, `<tool_call>`) stream through (`test_think_tokens_are_not_control_tokens`) |

Verified live through the service on 1.7B Q1_0, 4B Q2_0, and 27B PQ2_0: regular,
reasoning-shown, and reasoning-hidden paths all clean (no template leaks, no U+FFFD, reasoning
streams token-by-token). 27B note: its CLI "empty output" was the CLI's own ThinkFilter hiding
in-think reasoning without `--show-think`, not an engine bug.

## Current Benchmark (2026-09-18, CPU, 50-token decode, `--temp 0 --prompt "Hello" --prompt-raw`)

| Model | Format | Tok/s | Notes |
|-------|--------|-------|-------|
| Bonsai 1.7B | Q1_0 | 41.2 | cold single run (doc'd cold baseline was ~31) |
| Bonsai 4B | Q1_0 | 21.7 | |
| Bonsai 8B | Q1_0 | 12.5 | |
| Ternary Bonsai 1.7B | Q2_0 | 26.3 | |
| Ternary Bonsai 4B | Q2_0 | 11.9 | |
| Ternary Bonsai 8B | Q2_0 | 6.1 | |
| **Ternary Bonsai 2 27B** | **PTQ1_0** | **1.3** | was 0.2 before this session's kernel work; reference fork = 0.17 |
| **Ternary Bonsai 2 27B** | **PQ2_0** | **0.8** | Q2_0 kernel is ~2× slower per weight byte than PTQ1_0 |

### Session 2026-09-18 — PrismML `qwen35` (Ternary Bonsai 2 27B) support

New architecture brought up from scratch (64 layers, 3:1 gated-delta-net → full attention
hybrid). Layer 0 now reproduces the reference fork **bit-exactly** for a single-token
forward pass (cos = 1.000000 on `attn_norm`, `linear_attn_qkv_mixed`, `beta`,
`final_output`, `linear_attn_out`, `attn_residual`; 0.999996 on the FFN branch), and
greedy decode matches the reference token-for-token on the same prompt.

**Bugs found (all fixed):**

| # | Bug | Symptom | Root cause |
|---|-----|---------|------------|
| 1 | Tokenizer pre-tokenization | all models silently fed mangled prompts (spaces dropped, newlines lost) | `pre_tokenize` split on whitespace then failed vocab lookup; needed real GPT-2/Qwen rules + byte→unicode (Ġ/Ċ) mapping |
| 2 | Embedding inverse order | zero correlation with reference (`cos≈0.02`) at layer-0 input, correct magnitude | lookup applied the fold `R = D·H`; the inverse must be `R⁻¹ = H·D` (WHT **then** signs) |
| 3 | Stale Q8 scratch in `ssm_out` | GDN output 25× too large | `quantize_q8_0` **appends** to the `Vec`; the new path forgot `proj_q8.clear()`, so the kernel dotted a stale, wrong-width activation |
| 4 | FFN residual never added | layer output ~10× too large | second residual add was missing entirely |
| 5 | Conv weight indexing transposed | GDN conv output wrong | weight/window are `[channel][tap]`, not `[tap][channel]` |
| 6 | GDN v-head → k-group map | wrong head pairing | ggml `repeat` tiles: group = `v_head % n_groups` |
| 7 | `ssm_out` head permutation | wrong `ssm_out` output | activation must be permuted tiled→grouped (`kg*rep + r`) before the fold |
| 8 | PTQ1_0 dot kernel | 0.16 GB/s/thread (model at 0.2 tok/s) | fully scalar LUT-gather decode |
| 9 | Thread pool default | 8 of 12 usable, then 16-of-16 SMT contention | now `HEARTH_THREADS` override, default 3/4 of logical |

**Speed:** `dot_ptq1_0_q8_0_avx2` (SIMD base-3 decode + `vpmaddwd` ternary dot) is **8× faster**
than the scalar path (1.28 GB/s/thread), taking the 27B from 0.2 → 1.3 tok/s and beating
the reference fork's 0.17 tok/s on the same machine.

**Removed:** the `msvc-kernel` feature (an MSVC-compiled Q1_0 kernel plus its `build.rs`,
C source, FFI declaration and tests). Earlier experiments measured it *slower* than the
LLVM/AVX2 path, it required a manually-configured MSVC toolchain to compile, and it was
the only FFI in `hearth-quant`. The Q1_0 call sites now use
`hearth_quant::dot_q1_0g128_q8_0_ptr` directly, so there is no wrapper indirection left.

**Headless runner:** `cargo run --release -p hearth-llm --example gen -- <model.gguf> [prompt] [max_tokens] [temp]`
drives the same `Model::run` pipeline as the CLI/UI and prints text + tok/s — used to
regression-check all 8 models without the GUI.

**Diagnostic tooling added:** `HEARTH_DUMP_TENSORS=<prefix>` writes layer-0/pos-0 activations
for element-wise diffing against the reference fork; `HEARTH_Q35_DEBUG=1` prints stage
statistics; `HEARTH_THREADS=N` tunes the pool. `crates/hearth-llm/examples/probe_*.rs`
contain the comparison harnesses (see `examples/README.md`).

**Traps that cost time (avoid next session):**

- Comparing dumps from *different* passes: the reference callback overwrites its dump on
  every decode, so a greedy loop leaves the *last* token's tensor, not the prompt's. Use a
  deterministic single-decode mode (`oracle <model> TOK <id> 1`).
- Self-comparison: `ref_*` files were once copies of our own dumps → cos 1.0000 that meant
  nothing. Always confirm the two files came from different binaries.
- Statistics are not proof: a wrong-but-orthogonal rotation preserves mean/‖x‖. Only
  element-wise cos against a reference identifies the right transform.
- mmap is *not* the 27B bottleneck (350 MB RSS is normal); the file pages in at ~5 GB/s and
  the kernels were ~5× slower than even a disk-bound path.

**Warmup note:** the built-in `[warmup] CPU clock ramp: 32ms` runs just one throwaway forward pass — insufficient for this CPU to reach full boost. A full 50-token decode run as warmup (first run discarded, second measured) reliably gives 47-48 tok/s. The cold single-run tok/s is ~31 tok/s due to Windows frequency scaling ramp-up across the first 30-40 tokens.

## Final CPU Benchmark (1.7B Q1_0, 50-tok decode, pre-wgpu baseline, peak)

| Metric | Hearth | Ref (llama.cpp-prism) | Speedup |
|--------|--------|----------------------|---------|
| Decode tok/s (50tok) | 50.0 tok/s | 35 tok/s | **1.43×** |
| Decode tok/s (Q2_0) | 29.0 tok/s | 5.8 tok/s | **5.0×** |
| Prefill (100t) | 12.5 ms/tok | 19 ms/tok | **1.5×** |
| Prefill (800t) | 13.1 ms/tok | 19 ms/tok | **1.5×** |

## CPU Optimization History (All Sessions)

| Session | Change | Decode Gain | Prefill Gain |
|---------|--------|-------------|--------------|
| S1-9 | Custom thread pool, SIMD kernels, LUT optimization | 4.0→45 tok/s (+1050%) | — |
| S10-13 | Q2_0 VNNI, shuffle kernel, scratch reuse | 45→46 tok/s (+2%) | — |
| S14-15 | AVX-512 VNNI (neutral on Zen 4), f16 LUT | neutral | — |
| S16 | Batch matmul dispatch (par_dot_rows_batched) | neutral | Prefill enabled |
| S17 | Pre-quantize activation once for Q/K/V | neutral | ~5% |
| S18 | Head norm reuse, inv_sqrt_hd hoisting | neutral | ~2% |
| S19 | Matmul confirmed saturated — no decode gains possible | none | none |
| S20 | Parallel batch quantize + chunk-outer attention | neutral | **~15%** |
| S21 | Q8_0 KV cache (REVERTED — net-negative) | — | — |
| S22 | Fused RMS Norm + Quantize | neutral | <1% |
| S23 | Fused SiLU+Mul+Quantize | neutral | <1% |
| S24 | Head-parallel attention dispatch + SFC loop interchange | neutral | **~7%** |
| S24b | Chunk-outer weighted sum for decode attention | **~2×** at 800tctx | — |
| **S25** | **hearth-ui crate created** (no perf impact) | **neutral** | **neutral** |
| **S26** | **Paired-token prefill kernels (x2 weight reuse)** | **neutral** | **8–32%** |

### Cumulative: prefill 25.5 → 17.5 ms/tok (Q2_0) and 14.1 → 12.9 ms/tok (Q1_0) on 840-token prompts; decode unchanged at 50 tok/s (1.7B)

## What Didn't Work (Archive)

- Q8_0 KV cache: dequant compute > bandwidth savings at all context lengths
- 16 threads: SMT contention kills SIMD throughput (10× regression)
- VNNI kernels: Zen 4 double-pumping makes 512-bit neutral vs 256-bit
- Weight pre-expansion: 7.2× memory traffic increase swamped compute savings
- QKV fusion: <3% gain, not worth complexity
- Prefetch hints: hardware prefetcher already optimal on Zen 4
- Tiled attention: compute-bound system, chunk-outer already optimal
- Four-token prefill kernel (x4): **3.2× regression** — 55.9 ms/tok vs 17.5 for x2 (1.7B Q2_0).
  Eight live `ymm` accumulators (4 block + 4 global) plus an sret-returned `[f32; 4]` spill the
  inner loop; 16 architectural registers leave no room. x2 is the sweet spot. To revisit x4,
  drop the block accumulators (fold `w_scale * a_scale` per sub-block into one chain) or expand
  a weight block into a stack buffer *outside* the token loop, so neither needs 8 live vectors.

## GPU Architecture (Updated 2026-06-12)

See `performance_ops.md` for full GPU planning.

### Phase Plan

| Phase | What | Depends On | Est. Days | Status |
|-------|------|------------|-----------|--------|
| 0 | wgpu infrastructure (device, buffers, dispatch) | Nothing | 3-5 | **DONE** |
| 1 | Q1_0/Q2_0 dequant+matmul WGSL kernel | Phase 0 | 5-7 | **DONE** |
| 2 | Full decode on GPU (all layer shaders) | Phase 1 | 5-7 | **DONE** |
| 3 | Prefill on GPU (batch matmul) | Phase 2 | 3-5 | 🔜 Planned |
| 4 | Image generation GPU (hearth-diffusion) | Phase 1 | 5-10 | — |
| 5 | Optimization (fusion, tuning) | Phase 2/3 | Ongoing | 🔜 Planned |

### Current GPU Status (2026-06-12)

**Pipeline:** Full decode pipeline works correctly on GPU (cosine similarity 0.999967 vs CPU).
**Performance:** GPU on Radeon 780M iGPU is ~4.7× slower than CPU due to lm_head matmul bottleneck and synchronous readback.

#### Benchmark Results (Radeon 780M iGPU, 50-token decode)

| Model | Format | CPU tok/s | GPU tok/s | GPU/CPU |
|-------|--------|-----------|-----------|---------|
| 1.7B  | Q1_0   | 34.6      | 7.3       | 0.21×   |

(GPU benchmarks pending for other models — pattern expected to be similar since lm_head dispatch is common across all model sizes)

#### Debugging History (Phase 2 — 3 bugs found and fixed)

**Bug 1: KV cache buffer size (CRITICAL)**
- Location: `model/mod.rs` line 221
- Issue: `per_layer = (n_kv_heads × cache_max_seq × head_dim) × 2` — missing `× 4` for bytes
- Effect: KV cache buffer was 4× too small. V-part writes went out-of-bounds, corrupting adjacent GPU memory.
- Fix: Added `* 4` to multiply by f32 byte size.

**Bug 2: RoPE rope_half (CORRECTNESS)**
- Location: `lib.rs` rope_combined/rope_batch
- Issue: params[2] set to `rope_dim` instead of `rope_dim/2`. Shader interpreted this as pairing stride.
- Effect: RoPE read/wrote past head boundary, corrupting adjacent head data.
- Fix: Changed to `(rope_dim / 2) as f32`.

**Bug 3: RoPE mscale (CORRECTNESS)**
- Location: `lib.rs` + `rope.wgsl`
- Issue: yarn_mscale was hardcoded to 0.0 in the yarn array, and shader never applied it.
- Effect: Standard RoPE (non-YaRN) had mscale=0.0, causing all rotated values to become 0.
- Fix: Rust code passes `yarn_mscale = 1.0` for standard RoPE, shader multiplies sin/cos by mscale.

#### Readback Optimization

- Pre-allocated staging buffer for lm_head readback (avoids per-token buffer allocation)
- Recorded copy-to-staging on batch encoder (avoids extra command encoder + poll)
- Result: Readback of 606KB (151669 f32s) takes ~1ms for the copy, but `device.poll()` adds ~126ms waiting for GPU to complete.

#### Key Bottleneck

The lm_head matmul (151669 rows × 2048 columns) on GPU dispatches 196,605 workgroups, which stalls the iGPU's limited compute units for ~120ms. The actual per-layer compute is very fast (~3ms for 28 layers).

**Potential solutions (Phase 5):**
- GPU-side argmax/top-k sampling to avoid full logit readback
- Use CPU for lm_head matmul while GPU runs layers (async overlap)
- Reduce readback to only the hidden state (8KB) and compute lm_head on CPU
- Fuse lm_head with output norm (hard due to weight quantized layout)

### Phase 0 Deliverables (2026-06-03)
- `hearth-compute/Cargo.toml` — wgpu 29, pollster 0.4 deps added
- `src/device.rs` — `GpuDevice` (wgpu Instance/Adapter/Device/Queue init, Vulkan backend)
- `src/buffers.rs` — `create_storage_buffer`, `upload_to_buffer`, `download_from_buffer`
- `src/pipeline.rs` — `create_compute_pipeline`, `dispatch_compute`
- `src/shaders/simplest.wgsl` — trivial `output[i] = input[i] + 1.0` compute shader
- `src/shaders/mod.rs` — shader include_str!() module
- `src/lib.rs` — `GpuCompute` struct with real `wgpu::Buffer` (re-exported as `GpuBuffer`), all methods still return None/false except: `new()`, `upload_f32()`, `upload_f16_packed()`, `upload_bytes()`, `create_storage_buffer()`, `readback_f32()`, `run_simplest_test()`
- `test_simplest_shader` — unit test initializes GPU, runs compute shader, verifies output
- All hearth-llm code compiles unchanged (type migration `()` → `GpuBuffer` was transparent)
- CPU path unchanged — no decode/prefill regression
- GPU detected: AMD Radeon 780M via Vulkan backend

### Session 2026-06-07: hearth-ui crate
- **New crate:** `hearth-ui` — egui chat interface for Hearth LLM inference
  - `crates/hearth-ui/Cargo.toml` — eframe 0.34, egui 0.34
  - `crates/hearth-ui/src/main.rs` — full chat app (see RESUME.md for details)
- **Workspace:** added to workspace members
- **Binary:** `target/release/hearth-ui.exe` (12.5 MB), zero clippy warnings

### Key Technical Decisions

1. **wgpu + WGSL** over ROCm (780M unsupported) and DirectML (maintenance mode, no custom kernels)
2. **Unified memory** (shared DDR5) means zero-copy weight access — no PCIe transfers
3. **Reference project:** `wgpu-llm` (github.com/Beledarian/wgpu-llm) — 12 WGSL shaders for Llama
4. **Fused dequant+matmul** kernels required for Q1_0/Q2_0 weights (WGSL custom)
5. **Kernel fusion critical** — wgpu dispatch latency (20-50µs) makes per-op dispatch expensive
6. **FP16 matmul** for diffusion (FP16 throughput on RDNA3 is 2× FP32)
