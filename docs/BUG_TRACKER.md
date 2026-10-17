# Hearth Performance Tracker — CPU Final State → GPU Planning

## System
AMD Ryzen 7 8840HS (8C/16T), Radeon 780M iGPU (12 CU RDNA3), 16 GB DDR5, Windows 11. Hearth: 8 workers dense (`physical`), 10 for hybrid `Q2_0`/`PQ2_0`, 12 for hybrid `PTQ1_0` — see the two entries below for why they differ. Ref: llama.cpp-prism.

### Count 9's batched-prefill cliff is bimodal — and it was never the dispatcher's wait (SparkOC, 2026-10-17)

Follow-up to the entry below, which left one question open: *why* the dense **batched** prefill
(`matmul_batch` into `batched_rows_chunk`) loses 2.4x at 9 workers on an 8C/16T box when the
single-token path is stable there and *faster* (decode +2.8 to +6.6% on all six dense models in two
mirrored sweeps). First hypothesis: the dispatcher's own wait. `wait_for_workers` spins
`SPIN_BUDGET` and then `yield_now`-loops, and `yield_now` only offers the core to a thread ready on
the same *logical* processor, so the dispatcher keeps competing with the worker that shares its core
— and a batched chunk runs for milliseconds.

Tested by making the wait per-dispatch (batched dispatch blocks on a condvar the workers notify under
the same lock; single-token dispatch keeps yielding), mirrored against the shipped binary:

| dense 8B Q2_0, 109-token prefill | yield | park |
| --- | --- | --- |
| 8 workers (the shipped count) | 9357 ms | 9281 ms |
| 9 workers | 12999/8320/13256/19933/8439 | 8582/8605/8720/8617 |

**The 9-worker row is not a comparison.** The yield numbers are pt 18's window, the park numbers this
one, and those two windows are not interchangeable — which is the trap the whole entry is about. The
only same-window pair is the 8-worker row, where the park buys nothing (9281 against 9357 is inside
the noise), and parking *unconditionally* costs single-token decode 7% (dense 4B Q2_0 11.3 tok/s
against 12.2), which is why it had to be conditioned on `seq_len > 1` at all. So it was **reverted**:
the count stays `physical`, the wait stays yield, and the park was not kept, because a change with no
same-window measurement behind it is just code. (A later window did read the park tight at 9 —
8582/8605/8720/8617 — but with no yield arm beside it in that same window, that establishes nothing.)

The count-9 decode half is real and repeatable, though, and count-isolated runs pin it down (same
binary, `HEARTH_THREADS` 8 against 9): dense 1.7B Q1_0 48.84 → 50.24, 4B Q1_0 22.01 → 22.13, plus
4B Q2_0 12.58 → 13.02, 8B Q1_0 11.72 → 12.45, 8B Q2_0 5.83 → 6.22 against the shipped arm. The two
paths genuinely want different counts; the batched prefill's tail is the only thing holding the
dense count at physical.

Two follow-up probes killed the obvious suspects and moved the answer. **Bandwidth**: a batched
prefill streams its ~2.3 GB of weights once per dispatch, so the 9.2 s at count 8 is ~250 MB/s — under
1% of what the memory system delivers — so it is not bandwidth bound. **DVFS**: sampling
`\Processor Information(_total)\% Processor Performance` through the prefill reads a mean of 108% at 8
workers against 103-105% at 9, boost peaks 123-125% in both, temperature 36.8 C — the frequency does
not collapse.

The same session also found the mode: on a quiet machine 9 workers beats 8 at *every* prompt length
measured — 19 tokens 1977 → 1831 ms, 55 tokens 5039 → 4700, 109 tokens 9396 → 8952 (ratios 0.93 /
0.93 / 0.95). One sample per point, so those runs cannot see a spread — and when the controlled sweep
could, the ambient-load explanation for pt 18's outlier fell over: a 6-thread busy load (the runner's
loaded condition) doubles every row — 14 runnable threads on 8 physical cores — yet it moves the tail
around rather than concentrating it on 9 (4B Q2_0 loaded prefill 13795/13959 ms for 8/9, inside 5%;
8B Q2_0 loaded prefill 27070/26395, i.e. 9 ahead, while the loaded *decode* spread read 15.9% at 8
against 59.4% at 9). So the load arm confirms the mechanism but is no sharper than the quiet window
for finding the tail — the tail is count 9's own.

**What the controlled sweep found** (`scripts/count-sweep.py` — the tool RESUME item 3 asks for: a
per-count mean *and* spread per condition, the run order rotated, a fragility flag, and an explicit
"inside the noise, no switch justified"). Dense 8B Q2_0, 109-token prefill, `HEARTH_THREADS` 8 against
9, no added load:

| | count 8 | count 9 |
| --- | --- | --- |
| raw (5 repeats) | 9062 9134 9217 9008 8991 | 8528 8599 8679 **15115** 8296 |
| mean / spread | 9082 ms / 2.5% | 9843 ms / 69.3% |
| clean state | 9082 ms | ~8400 ms (0.93x of count 8) |

Count 9's batched prefill is **bimodal**, not erratic: four runs in five it is the ~7% faster arm the
length sweep saw, and the fifth is 1.7x slower than its own clean state. A second quiet window read
the same shape (60.4% spread against 1.3% for count 8), and pt 18's window read the tail end of it
(12999/19933). Both earlier readings were true samples of one distribution — the "cliff" was the tail,
the quiet sweep was the mode.

The mechanism is the split. `pool.rs` divides a batch statically (`chunk = n.div_ceil(nt)`, worker `i`
takes `i*chunk..(i+1)*chunk`), so 9 workers over 109 rows hand the ninth 5 rows against 13 for the
other eight, and 9 threads on 8 physical cores always leaves one core holding two of them. When that
pair is two long chunks the dispatch waits on the slower one for the whole run (the tail); when it is
a long chunk and the 5-row one, the penalty ends early (the mode). Count 8 has neither problem: eight
14-row chunks on eight cores, balanced.

So the dense rule is **robustness, not speed**. Count = `physical` gives exactly one runnable worker
per core, so external load costs it a little and its spread stays 1-3%. At 9+ the pool is one over and
a dispatch cannot finish before its last worker, so one bad placement costs the run. The same
mechanism is behind the older 8B Q1_0 reading (13.2 tok/s at 8 against 6.9 at 12) and behind 10 workers
being a consistent 2.1x. Count 9's *mode* is 5-7% better and its *tail* is 1.7-2.2x worse, and for a
default on a laptop the tail is the number that matters. **So the calibration runner has to measure the
spread, not just the mean** (RESUME item 3): a count that wins by 5% quiet and loses 2x on one run in
five is not a winner — `scripts/count-sweep.py` flagged count 9 fragile the first time it ran.

### The Q2_0 dispatch preferred a kernel 47% slower per thread, and the worker count follows the prefill (SparkOC, 2026-10-16 late)

Fourth instance of the missing-declaration class, and the first where the consequence was a dispatch
*order* rather than a dead kernel. `dot_q2_0_q8_0_ptr` preferred the 512-bit VNNI arm, with a comment
saying `vpdpbusd` "cuts the front-end work enough to pay for the half-width datapath" and citing a
warm-machine `benchk` table (3.10 VNNI512 / 3.09 VNNI256 / 2.91 LUT). That table was taken *before
the LUT kernel's declarations were fixed* — it went 256 → 167 instructions in the change log — and a
body that cannot inline is not the body that ships.

Re-measured on real weights at the shipped `x86-64-v3` baseline:

| model (`blk.0`, every tensor) | dispatched (VNNI512x2) | vnni256 | avx2_lut |
| --- | --- | --- | --- |
| dense 4B Q2_0 | 1.76 | 1.61 | **2.57** |
| hybrid 4B PQ2_0 | 1.74 | 1.61 | **2.56** |

`avx2_lut` leads on *every* tensor, and the two arms that are the same function (`dispatched` and
`vnni512x2`) agree within 1%, so the ranking is not a position artifact. The earlier session saw the
same v3 reading (2.51 against 1.70) and dismissed it as clock ramp-up because a `native` rerun
reversed it — but we ship v3, and `native` changes the codegen of a body written in 256-bit
intrinsics.

Two things came off the same table besides the order. The 512-bit arm is now gated on
`crate::avx512_full_width`, which is the gate PTQ1_0 already puts on its AVX-512 kernel — the change
log's earlier entry had *removed* that gate from Q2_0, to keep Zen 4 off the AVX2 LUT kernel, and
that is the decision this measurement reverses. The 256-bit VNNI arm now sits below the LUT arm,
where it is the fallback for a part whose AVX2 probe fails rather than a live choice (every part with
AVX-512F also has the AVX2, FMA and F16C that arm needs). On this box both gates skip it
(`avx512_full_width=false`), so the banner still reads `Q2_0=avx2_lut`. **A full-width AVX-512 part
(Intel) is unmeasured — the 512-bit arm keeps the slot it was written for, and that is the one
question the calibration runner below is the answer to.**

**The kernel and the worker count are one decision — but which decision depends on the *prefill*.**
The hybrid `Ternary-Bonsai-2-4B-PQ2_0` sweep, one run per point then mirrored groups for 8 and 10,
with the LUT kernel it now dispatches:

| workers | 4 | 6 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- | --- |
| decode tok/s | 4.98 | 6.38 | 7.1–7.4 | **7.7–7.9** | 4.3–4.9 |
| 109-token prefill ms | — | — | 13940 | **13346** | — |

so that path wants a quarter past physical, and that is what it takes: its prefill is token-by-token,
which is the *stable* single-token path. The dense path does **not**, and the first version of this
change got that wrong — it moved `default_worker_count()` to the same quarter-past count, which
regressed dense prefill up to 2.4x while it helped decode. The dense prefill is *batched*
(`matmul_batch` → `batched_rows_chunk`, plus `flash_attention_prefill`), and that path is unreliable
once the workers outnumber the physical cores:

| model, 109-token prompt | 8 workers | 9 workers | 10 workers |
| --- | --- | --- | --- |
| dense 8B Q2_0 | 9011, 8973, 9158 | 12999, 8320, 13256, 19933, 8439 | 19841, 20854 |
| dense 4B Q2_0 | 4746, 4825 | 4561, 4560 | 9847, 10153 |
| dense 8B Q1_0 | 6448, 6394 | 6030, 5967 | 12557, 11768 |

Eight runs alternating 8 and 9 workers on the 8B Q2_0 give a **2.4x spread at 9 on identical
invocations** (8320 to 19933 ms) against a tight 8 (9011/8973/9158). That is not a slope to tune
around, it is a stability problem, and the boundary moves with the model (4B Q2_0 is still fine at 9
and falls at 10; 8B Q2_0 is already unstable at 9), so the honest rule stays at the physical count
until the cause is known. The first hypothesis to test is the pool's spin-then-yield wait stealing a
core from a worker while the batched chunk runs (`docs/RESUME.md`).

The single-token path *is* stable above physical and does want more — dense 4B Q2_0 decode 13.4 tok/s
at 10, 12.9 at 9, 11.7 at 8; dense 8B Q2_0 7.0 at 9 against 6.5 at 8 — so the count that serves both
passes is the physical one. The three rules, all measured:

- dense: `physical` (batched prefill, capped where it is stable).
- hybrid `Q2_0`/`PQ2_0`: `min(logical, physical + physical/4)` = 10 (token-by-token prefill only).
- hybrid `PTQ1_0`: ¾-of-logical = 12 (no fast kernel; it hides the pool's barrier behind threads —
  6.7 tok/s at 12 against 5.9 at 8).

End to end, mirrored (before = VNNI Q2_0 dispatch + 8 dense / 12 hybrid workers; after = LUT + the
rules above), decode then a 109-token prefill. The first pass of this table was taken with the
quarter-past dense count, and it is what exposed the regression — its 8B Q2_0 prefill read 21856 ms
against 10060. With the count back at physical, the same row is:

| model | decode before → after | prefill before → after |
| --- | --- | --- |
| Bonsai-1.7B-Q1_0 | 43.14 → 43.23 (+0.2%) | 1364 → 1375 ms (+0.8%) |
| Ternary-Bonsai-1.7B-Q2_0 | 20.63 → 27.14 (+31.6%) | 2082 → 1972 ms (−5.3%) |
| Bonsai-4B-Q1_0 | 21.14 → 21.98 (+4.0%) | 3548 → 3590 ms (+1.2%) |
| Ternary-Bonsai-4B-Q2_0 | 9.00 → 11.81 (+31.2%) | 5273 → 4901 ms (−7.0%) |
| Bonsai-8B-Q1_0 | 12.57 → 12.77 (+1.6%) | 6660 → 6818 ms (+2.4%) |
| Ternary-Bonsai-8B-Q2_0 | 4.84 → 6.34 (+31.0%) | 9792 → 9370 ms (−4.3%) |
| hybrid 4B PQ2_0 | 7.13 → 8.34 (+17.1%) | 15541 → 13433 ms (−13.6%) |
| hybrid 4B PTQ1_0 | 7.14 → 7.49 (+4.9%) | 15882 → 15642 ms (−1.5%) |

The Q1_0 rows are the control — kernel *and* count unchanged from the previous commit, so their ±4%
is this box's noise band. The dense Q2_0 rows are the kernel alone (the count is back at 8): +31%
decode, −4 to −7% prefill. The hybrid `PQ2_0` row is kernel plus count (12 → 10). The hybrid `PTQ1_0`
row is unchanged in both, so its +4.9%/−1.5% is noise as well. `kernel_name()` reports `avx2_lut` for
both paths, so the `[kernel]` banner and the service's `kernels` JSON stay honest without a per-path
variant.

**Baseline invariance, measured.** The dispatch change was re-checked at the weak baseline, because
the arm it makes the default (the Q2_0 LUT kernel) had read 3x slower at `x86-64-v2` before its f16c
fix (0.87 against 2.64 GB/s). Same source, two side target dirs (`RUSTFLAGS="-C target-cpu=x86-64-v2"`
against the shipped v3), `benchk` twice each, best of two:

| model (`blk.0`) | arm | v2 | v3 |
| --- | --- | --- | --- |
| dense 4B Q2_0 | `avx2_lut` | 2.75 | 2.84 |
| dense 4B Q2_0 | `dispatched` | 2.69 | 2.81 |
| dense 4B Q2_0 | `vnni256` | 1.72 | 1.72 |
| dense 4B Q2_0 | `vnni512x2` | 1.85 | 1.87 |
| hybrid 4B PQ2_0 | `avx2_lut` | 2.75 | 2.83 |
| hybrid 4B PQ2_0 | `dispatched` | 2.72 | 2.80 |

Every arm is within 3% across the two baselines and `dispatched` tracks the LUT arm at both, so the
new order is not a v3-only artifact: at v2 the old order would have selected `vnni512x2` (1.85) where
the new one selects `avx2_lut` (2.75) — **+49% at the weak baseline too**.

`scripts/baseline-invariance.py` is that check as a tool. It builds nothing (it takes
`NAME=TARGET_DIR` pairs) and reports the two failure modes separately: *arm drift*, where one arm
reads lower at one baseline, and *dispatch drift*, where `dispatched` is not the fastest arm at that
baseline. Validated against artifacts rather than by assertion — run on the pre-change benchk in
`target/buffy-ab/v3-target` it reports `old: dispatched 1.85 vs best arm avx2_lut 2.77 (0.67) <--
dispatch drift`, which is the bug this commit fixes, plus `vnni256` 0.67 → 1.73 (0.39), which is the
declaration fix on that kernel. It is deliberately not a CI gate: shared runners are not a
benchmark, so the static guard stays the gate and this is the measurement it stands in for.

Lesson, four times in one day: **a dispatch order is a measurement, and a measurement of a body that
cannot inline is not a measurement of the body that ships.** The guard now covers the test target as
well as the library (see the entry below). The worker-count half of this entry is a second lesson:
**a count tuned on decode alone is not tuned**, because the dense prefill is a different code path
(batched) with a different and model-dependent stability boundary — and the first version of this
change cost 2.4x on the 8B before mirrored *prefill* runs caught it. Both axes belong in the
calibration runner (`docs/RESUME.md`).

### The kernel-baseline guard only built the library, so a test-only kernel was never checked (SparkOC, 2026-10-16 late)

`scripts/check-kernel-baseline.py` is the guard for the whole missing-declaration class, and the
`kernel-baseline` job fed it one listing: `cargo rustc -p hearth-quant --release --lib`. A kernel
behind `#[cfg(test)]` is not in that listing. That is exactly where the Q1_0_G128 VNNI kernel sat,
measuring +180% against the kernel it was meant to replace, with nothing able to dispatch it and so
nothing ever re-measuring it. The job now builds `--tests` at the same weak baseline and checks that
listing too.

One exemption was needed: the F16 rule is scoped by function name, and a function inside a `tests`
module legitimately calls `half`'s runtime-dispatched conversion (`f32_to_f16_x86_f16c`) — that is
how `half` dispatches, and the harness itself is compiled at the weak baseline. Kernels are never
named `...tests...`, so every kernel the tests reach is still checked, and the FMA/VNNI rule stays
unconditional because it holds for every caller.

Validated both ways: with `avx512vnni` removed from the test-only `_2sub` kernel, the test-build
guard fails with `dot_q1_0g128_q8_0_vnni_avx512_2sub -> __mm512_dpbusd_epi32`; restored, both
listings are clean.

### The Q1_0 VNNI kernel was never dispatched because of the same missing-declaration bug — and it is 6% faster (SparkOC, 2026-10-16 late)

`hearth-quant/src/q1_0g128.rs` carried a 256-bit VNNI kernel as `#[cfg(test)]` with a note claiming a
"~33% regression on Zen 4 vs shuffle — keep for future CPUs with native 512-bit VNNI units". That
figure was measured at a `native` baseline, where `_mm256_dpbusd_epi32` still inlined. At the
`x86-64-v3` baseline we ship, the body declared `avx2` + `avx512f` + `avx512vnni` but **not**
`avx512vl` (the 256-bit EVEX form) or `fma` (`_mm256_fmadd_ps`), so the intrinsic and every fmadd
were out-of-line calls. Third instance of the same bug class in one day.

Against the shuffle kernel, from the kernel's own benchmark test:

| arm | n=2048 | n=4096 | n=6144 | n=9728 |
| --- | --- | --- | --- | --- |
| VNNI256 as it was (no VL, no FMA) | +178% | +185% | +189% | +190% |
| VNNI256, declarations fixed | **−3.8%** | **−6.6%** | **−5.2%** | **−4.2%** |
| VNNI512 2-sub (AVX2 + FMA added) | +337% | +348% | +356% | +345% |

The 512-bit kernel stays out — double-pumping is real — but the 256-bit one is *faster*, so it is
now dispatched ahead of shuffle, behind `crate::avx512_vnni_available()` (moved from `q2_0` into
`lib.rs` so both formats and both `kernel_name`s read one probe). Real weights agree: `benchk` on
the 1.7B Q1_0, all seven `blk.0.` tensors, **3.35 GB/s against 3.16** for shuffle, +6% on every
tensor. End to end, interleaved against the previous binary, group means:

| model | before | after | change |
| --- | --- | --- | --- |
| Bonsai-1.7B-Q1_0 | 45.66 / 43.94 / 43.27 / 42.47 | 47.33 / 45.10 / 45.11 / 45.52 | **+4.5%** (4 groups, 8 runs/arm) |
| Bonsai-4B-Q1_0 | 22.42 / 22.13 / 18.02 / 20.53 / 21.31 / 21.27 | 22.88 / 21.72 / 21.76 / 21.32 / 21.93 / 21.85 | **+2.7% median** (one group caught a transient) |
| Bonsai-8B-Q1_0 | 11.45 / 11.88 / 11.89 / 12.06 | 12.12 / 12.25 / 12.30 / 12.31 | **+3.6%** (4 groups, 8 runs/arm) |
| Ternary-Bonsai-4B-Q2_0 | 8.97 / 8.93 | 9.33 / 8.77 | noise, as it should be — that path did not change |

**Method note, because it cost an hour.** The first pass ran `before` then `after` inside each pair
and read +6.5% / +3.7% / +3.9%. Re-running the two binaries against each other with the order
*mirrored* (`b,a,a,b` groups and `a,b,b,a` groups) shrank every number to the ones above: on this
box the second run in a pair is worth ~2–3%, so an unmirrored A/B invents a win. The unit is the
group mean, and every group here is order-mirrored.

`hearth-quant`'s 42 tests pass with it dispatched and the VNNI sums are bit-identical to shuffle. The
`kernel-baseline` guard covers it now — with `avx512vl` removed it fails with
`dot_q1_0g128_q8_0_vnni_avx512 -> __mm256_dpbusd_epi32`. It did not before because the kernel was
`#[cfg(test)]` and the guard built only the lib, so the guard now builds the test target too (see the
entry above): a bench-only kernel is a kernel nothing checks.

### aarch64 was a release platform with no CI coverage — and it did not build (SparkOC, 2026-10-16 late)

**`macos-latest` is an arm64 runner now, so `check (macos-latest)` had been red on `main` since the
TLS crate landed.** The failure was visible, the cause was not: nobody had ever run the workspace
for aarch64 locally. One `cargo check --workspace --all-targets --target aarch64-unknown-linux-gnu`
reproduced the whole thing.

- **`hearth-tls` did not compile for aarch64 at all.** `Aes128::encrypt_block` and `Gcm::ghash`
  called `crate::aesni` / `crate::ghash_x86` from a branch gated only on the runtime probe. Those
  modules are `#[cfg(target_arch = "x86_64")]`, and a branch the optimizer can prove dead is still
  a name the compiler has to resolve — so the *call* has to sit behind the architecture gate too,
  which is also where the probe is decided (`aes_ni_available` / `pclmul_available` already return
  `false` off x86_64).
- **`hearth-quant` reported six dead-code warnings there** (`AMD_FAMILY_SPLIT_AVX512`,
  `cpuid_family`, `cpuid_is_amd`, `avx512_full_width_for`, `Q2V_I16`, `Q2V_U8`), plus `Q1V_SM` in the
  test build and an unused `Instant` import. Every one is x86-only by construction — the AVX-512
  width policy and the SIMD lookup tables have no non-x86 reader — so they are now
  `#[cfg(target_arch = "x86_64")]`, and the four CPUID-policy tests are gated with the code they
  test (they still run on x86, and hearth-quant runs 35 of its 42 tests on aarch64).
- `benchk` built its variant list with a `mut` only the x86 arms needed, and kept a `have` stub
  nothing called.

**New CI:** `arm-cross-check` (check + clippy `-D warnings` for `aarch64-unknown-linux-gnu`,
`aarch64-apple-darwin`, `aarch64-pc-windows-msvc`; `check` needs no linker, so it runs on the x86
runners already in the matrix) and `arm-qemu-test` (links and *runs* the CPU-only suites under
`qemu-user` — the first execution of the aarch64 scalar fallback anywhere, including
`pool::tests::physical_cores_is_always_a_usable_count`, which now covers the non-x86 stub). Run
`38988216344`: all 8 jobs green, macOS included. Commits `7a918ad` (CI + the four fixes).

### Baseline is x86-64-v3, the kernels are baseline-invariant, and one missing attribute was worth 2.7x (SparkOC, 2026-10-16)

**`26eab4b` set the crate baseline to `x86-64-v2` and lost 3.4x. Root cause, proven from the
listing rather than argued.** `#[target_feature(enable = "avx2")]` does not imply `fma` or `f16c`.
Under a v2 baseline an AVX2 kernel that reaches for `_mm256_fmadd_ps` or converts an f16 scale
cannot inline those, so LLVM emits an out-of-line call *per use* while the dispatch banner still
reports `avx2`. The v2-built `dot_q1_0g128_q8_0_ptr_avx2` was 494 instructions with 5
`__mm256_fmadd_ps` calls, 4 `f16_to_f32_x86_f16c` calls and 5 `detect_and_initialize` calls;
with `-C target-feature=+fma,+f16c` it is 135 instructions and 0 calls. `f16c` alone was not
enough (v2+f16c still logged 0.67 GB/s: the 5 fmadd calls remained).

| arm | benchk Q1_0 (GB/s) | Bonsai-1.7B-Q1_0 (tok/s) |
| --- | --- | --- |
| x86-64-v2 | 0.55 | ~13 |
| x86-64-v2 (buffy's fma/f16c declarations) | 2.73 | 43.0 |
| x86-64-v3 (new default) | 3.48 | 45.4 |
| `-C target-cpu=native` | 3.12 | 50.6 |

**Fixes landed.** (1) `.cargo/config.toml` scopes its rustflags to
`[target.'cfg(target_arch = "x86_64")']` and pins `x86-64-v3` — the old `[build]` entry would have
failed every aarch64 build with "unknown CPU". (2) buffy declared `fma`/`f16c` on the AVX2 bodies,
folded both into the availability predicates, and routed the 35 in-kernel conversions through a
declared `crate::f16_bits_to_f32`, which is what stops `half`'s runtime probe (and both of its
arms) from being inlined into the block loop. The declarations are the true cause-fix and stay:
they make the kernels baseline-invariant, so a lower-baseline build keeps fast AVX2 even where v3
is not an option. (3) **`dot_q2_0_q8_0_vnni_avx512` declared `avx512f` + `avx512vnni` but used the
256-bit EVEX form, which also needs `avx512vl`** — the same bug class, and the whole reason
`vnni256` read 0.59 GB/s under v3 while `native` (whose baseline includes VL) reached 1.58. With
VL added to the body and to `avx512_vnni_available()`, **vnni256 is 1.59 GB/s at v3**, level with
native. (4) `scripts/check-kernel-baseline.py` + a `kernel-baseline` CI job build `hearth-quant`
at `x86-64-v2` and fail on any out-of-line call to an FMA/VNNI intrinsic (never legitimate) or to
an f16 helper *inside* an `avx2`/`avx512`/`vnni` body. Validated in both directions: it rejects
the pre-fix listing (41 calls) and accepts the fixed tree. `fmadd`-only or `f16`-only greps are
wrong in opposite ways — the fixed crate still makes 61 legitimate f16 calls in scalar/SSE41 and
dequantize paths, and a blanket grep false-fires on them. `RUSTFLAGS` does override the
`[target.'cfg…']` rustflags (v3 listing 409390 B, native override 389917 B), which is what the
job relies on.

**Worker count: the ¾-of-logical rule was a 2x regression on the dense 8B.** It was already
committed in the hybrid path; SparkOC's uncommitted slice spread it to the dense path, and the
first sweep caught it (8B Q1_0 6.0 against native 12.6–13.2). The loss is a cliff, not a slope:
the main thread spins at 100% while it waits, so SMT oversubscription starves the workers that are
doing the work. Thread sweep, dense 8B Q1_0: 6→10.9, **8→12.1, 10→12.0**, 12→6.0, 16→4.0. The
1.7B also collapses at 16 (5.4). But the optimum is path- and kernel-specific, not machine-wide:
Ternary-Bonsai-2-4B-PQ2_0 (hybrid) is 7.3–7.5 at 12 against 4.8–5.7 at 8, and Ternary-Bonsai-4B-Q2_0
(dense) is 9.8–10.8 at 12 against 8.1 at 8 — the opposite of the dense Q1_0 8B. One shared count
would cost ~2x on whichever path loses, so the decision is: **dense uses `pool::default_worker_count()`**
(physical cores, read from CPUID leaf 0x1F → 0xB → logical fallback; honours `HEARTH_THREADS`; 8 on
this box, equal to the old hardcoded 8, and correct on a 4C or 32C host), and **the hybrid path keeps
its own ¾-of-logical rule (12)**. CPUID is safe on 1.99 (`__cpuid`/`__cpuid_count` are not unsafe),
so there is no suppression anywhere.

**6-model sweep at the new default** (one run each, warmed, `--temp 0 --max-tokens 50`, ±10–15%).
The `pt 17` column is the same tree plus the dispatch-wait yield (and the aarch64 `#[cfg]` gates,
which are x86 no-ops), so it is also the check that the yield did not cost anything. The three Q1_0
rows then gained a further +2–4.5% from the VNNI dispatch (entry at the top of this file) and were
not re-swept, because the absolute level drifts ±10% between windows and the mirrored A/B is the
better evidence:

| Model | pt 16 | pt 17 (final binary) | 2026-10-15 chart |
| ----- | ----- | -------------------- | ---------------- |
| Bonsai-1.7B-Q1_0 | 43.40 | 46.82 | 47.13 |
| Ternary-Bonsai-1.7B-Q2_0 | 15.56 | 19.28 | 24.53 |
| Bonsai-4B-Q1_0 | 18.66 | 21.56 | 18.84 |
| Ternary-Bonsai-4B-Q2_0 | 7.82 | 8.91 | 12.41 |
| Bonsai-8B-Q1_0 | 11.3–12.3 | 11.60 | 12.20 |
| Ternary-Bonsai-8B-Q2_0 | 4.33 | 4.64 | 6.75 |

Every row is equal or better than pt 16, and the 1.7B/4B/8B Q1_0 rows are now at or above the chart
(46.82, 21.56, 11.60 against 47.13, 18.84, 12.20).

All three Q2_0 rows read ~36% under the chart, the same family-wide
drift pt 15 already documented (±25% on both arms). It is not a v3 artifact: interleaved against
the native binary, Ternary-Bonsai-4B-Q2_0 is **10.41 native against 8.15 v3 at 8 workers, and 10.3
v3 at 12 workers** — v3 matches native once the thread counts match, so the gap to the chart is
machine state plus the thread count, not the baseline. `benchk` kernel ranks are also noisy enough
to mislead: at v3 one run ranked `avx2_lut` 2.51 ahead of `vnni512x2` 1.70, the reverse of the
dispatch comment, but a native rerun gives `avx2_lut` 1.35 against `vnni512x2` 1.68 — VNNI ahead,
dispatch order correct, and the first reading was clock ramp-up.

**Follow-up: the thread cliff is bandwidth/SMT, not the spinning main thread.**
`ThreadPool::wait_for_workers` now spins for a budget and then `yield_now`s instead of spinning
unconditionally, because a thread that never yields keeps its own workers off a core it is already
oversubscribing. Measured on the dense 8B Q1_0 the same afternoon: 12.1 → 13.2 tok/s at 8 workers,
6.0 → 6.9 at 12, 4.0 → 4.8 at 16 (three runs at 8, spread 12.33–13.82, against a pre-change
12.1–12.31), 1.7B Q1_0 43.4 → 45.3, hybrid 4B PQ2_0 5.2 → 6.4 at 8 workers and 7.4 → 7.4–7.9 at 12.
So the spin was worth ~15–25% of the oversubscribed loss, and it is gone, but it is not the cliff:
12 workers is still about half of 8 on the dense 8B. That leaves memory bandwidth plus SMT sharing
as the mechanism, which also means the proposed "let the dispatching thread take a chunk" change
would only add a 9th/13th runnable thread to a saturated machine — it is dropped, not deferred. A
single anomalous hybrid reading (6.14 against 6.71/7.31/7.53) was re-measured and did not reproduce.

### Clippy pass completed (SparkOC, 2026-10-15 night) — CI green, no perf change

Finished buffy's pass: 64 sites across `hearth-llm` (format borrows,
8 signature groupings, 22 hot-loop iterator rewrites), `hearth-diffusion`
(loops, signatures, dead-stub removal), examples and bins. `cargo clippy
--workspace --all-targets -- -D warnings` green on stable 1.99.1, 98 tests
pass, fmt clean. Benchmark gate: post-change sweep vs chart baseline looked
down on Q2_0 models, but an interleaved A/B against a pre-pass binary built
from an isolated worktree agreed within ~3% on all four disputed models —
machine drift, not code. Charts stand.

**Independently re-derived (buffy, 2026-10-16), because a gate that only one agent ran is a claim.**
`cargo clippy -p hearth-llm --all-targets -- -D warnings` exits 0 here too, and then both revisions
were rebuilt **from scratch** (d571b8a in an isolated worktree, plus HEAD) into separate target dirs
and interleaved pre/post, 2 reps, one process at a time, on the 6 mandate models.

| model | pre (tok/s) | post (tok/s) | Δ |
|-------|-------------|--------------|---|
| Bonsai-1.7B-Q1_0 | 49.16, 50.35 | 48.92, 51.44 | +0.8% |
| Ternary-Bonsai-1.7B-Q2_0 | 21.36, 20.60 | 20.73, 21.51 | +0.7% |
| Bonsai-4B | 22.73, 25.65 | 22.40, 24.74 | −2.6% |
| Ternary-Bonsai-4B-Q2_0 | 9.05, 8.83 | 9.10 (rerun 9.54) | +2..7% |
| Bonsai-8B | 12.04 (rerun 14.31) | 11.45, 12.47 | −0.7% |
| Ternary-Bonsai-8B-Q2_0 | 5.05, 5.08 | 5.06, 5.19 | +1.2% |

Two findings worth keeping. (1) The Q2_0 dip is machine state and provably not the code: on the 8B
Q2_0 **both** arms read ~5.05 against the chart's 6.75, i.e. −25% on pre and post alike — the family
is down as a family, not on one side of the pass. (2) A single run is worthless here: the *same*
8B Q1_0 pre binary returned 12.04 and then 14.31 a few minutes later, which is the ±10–15% spread
pt 6 already documented. Only the alternating pair carries information. One harness trap for the
record: two runs came back with no summary line purely because my `grep` missed it in the stream —
raw reruns gave 9.54 and 14.31 tok/s, coherent text, exit 0, so it was never a crash. And the
tracked `target/release` binary is byte-size identical to the from-scratch HEAD build, confirming it
does contain `f3c18d4` despite an mtime 10 minutes older than that commit.

### hearth-quant — five allow-blocks removed, and a dispatch gate that followed the build machine (buffy, 2026-10-16)

Started as the open audit item ("a gate that is always-false or always-true") and found two
rule violations instead of a wrong predicate.

- **AGENTS.md forbids `#[allow(...)]`, and `hearth-quant` had five.** Both `unused_unsafe` ones
  were *measured* to be load-bearing before being touched: deleting them prints
  "unnecessary `unsafe` block" twice, because `__cpuid` is safe when the build has SSE and every
  build here has it through `.cargo/config.toml`'s `-C target-cpu=native`. The cause is now
  removed rather than the lint silenced — both call through an `unsafe fn` wrapper, which makes
  the call site's `unsafe` necessary in *both* build configurations. The three `dead_code` ones
  hid the rejected AVX-512 Q1_0 VNNI kernels and the LUT they share: their only consumer is
  `bench_q1_0_vnni_vs_shuffle`, so they are `#[cfg(test)]` now — compiled where they are
  measured, absent from a shipped binary, and re-dispatching one means deleting a line on
  purpose. `hearth-quant` now has zero `allow` attributes (the two remaining grep hits are doc
  prose about the removed one).
- **The bug they were sitting next to, which is the same class as the 34% PTQ1_0 one:**
  `q2_0::dot_q2_0_q8_0_ptr` *and* `q2_0::kernel_name()` wrapped their AVX-512 arms in
  `#[cfg(target_feature = …)]`. That makes a kernel's **existence** a property of the machine the
  crate was built on rather than the machine it runs on: a binary built anywhere without AVX-512
  VNNI cannot select the fastest Q2_0 kernel even on a CPU that has it, and the report could name
  a kernel the dispatch could not reach. Both now call one cached runtime probe per arm (the
  arrangement `ptq1_0` has always used), so drift between report and dispatch is impossible;
  the kernels already carry `#[target_feature]`, which is what allows them to be compiled
  unconditionally and called behind a runtime check.
- **Evidence, all reproducible.** Built with `-C target-feature=-avx512f,-avx512vnni,-avx512dq`
  — simulating a build made on a machine without AVX-512 — the suite is green and the report
  names `vnni512x2`, which is what the *running* CPU implies. With the old `cfg` gate restored
  under that same build, the new test `the_reported_kernel_is_the_one_the_features_imply` fails,
  so that test is the regression test rather than a restatement. Three neutering rounds (gate
  restored, order swapped, oracle falsified), all caught. 42 tests, `clippy --all-targets -D
  warnings`, `fmt`, and a warning-free release build.
- **benchk pq2_0 (4B PQ2_0, `blk.0.`), before/after, cold machine:** dispatched 1.62 → 1.65,
  `vnni512x2` 1.59 → 1.55, `vnni256` 1.51 → 1.54, `avx2_lut` 1.30 → 1.28 GB/s. Selection and
  ordering unchanged, which is the point of the change: it removes a build-time dependency
  without moving dispatch on a build that already had the features.
- **The recorded warm figures in those comments do not reproduce cold.** The comments quote
  3.09 (`vnni512x2`) / 3.10 (`vnni256`) / 2.91 (`avx2_lut`) GB/s from a warm machine; today's cold
  runs are ~2× lower at the same ordering. Same trap as the tok/s warmup note, in a different
  harness, and now written down in the comment beside the numbers.
- **Harness trap worth keeping:** `benchk … | head` exits **101** — SIGPIPE from `head` closing
  the pipe, which reads like a crash if the exit status is taken. Redirect to a file or use
  `tail` when the verdict is the exit code.
- **One correction for the record:** that commit's message says "four" allow-blocks; the real
  count was five (three `dead_code`, two `unused_unsafe`). Left in history rather than
  force-pushed over — the operator had just finished a history rewrite, and a second rewrite to
  fix a number in a message is a worse trade than this note.
- **What remained of the same rule:** `hearth-llm` 13 (5 `dead_code`, 4 `needless_range_loop`,
  3 `too_many_arguments`, 1 combined), `hearth-diffusion` 7, `hearth-compute` 4, `hearth-gguf` 1 —
  all of them closed the same day, next entry.

### Workspace allow-block sweep — every one removed, each fixed at its cause (buffy, 2026-10-16)

Closes the rule instead of closing a crate: **zero `#[allow(...)]` and zero `#[expect(...)]` in the
workspace**, and nothing silenced to get there. Started from the 25 sites the quant pass left
(`hearth-llm` 13, `hearth-diffusion` 7, `hearth-compute` 4, `hearth-gguf` 1).

- **`too_many_arguments`, 19 sites.** `hearth-compute`'s 15 dispatch methods take named `Copy` shape
  structs ([args.rs](../crates/hearth-compute/src/args.rs)) while the *buffers* stay explicit — the
  split is deliberate, since a struct that hides the buffers hides the aliasing that matters. Method
  bodies are unchanged: each destructures its shape back into the old local names on the way in, so
  the diff is a signature plus one line. Call sites now name their fields, which is what the lint was
  really about (`seq_len_q` and `seq_len_kv` differed only by position).
- **`ensure_batch_size` took seven shape arguments and needs two.** `d`, `nq`, `nkv`, `ffn_dim` and
  `head_dim` are all `self.config` properties and both callers re-derived them; they are read where
  they belong now. `matmul_batch`'s row count was never read at all — deleted, not turned into a new
  hard error, since that would change behaviour on model families this box cannot test and the tensor
  already knows its rows. `n_cols` stays: the body checks it against the tensor, which is what catches
  a weight name mixed up.
- **`GgmlDType` variants are UpperCamelCase** (`Q2K`, `IQ4NL`, `Q10G128`): the lint fires when a
  *cased* letter sits next to an underscore, so `Q1_0`, `Q2_0`, `Q4_0_4_4` were never flagged and are
  untouched. `name()` still returns the spec spellings (`"Q2_K"`, `"Q1_0_G128"`), so logs, the UI and
  error messages are byte-identical; only identifiers moved, and the enum's doc comment says so.
- **Both `needless_range_loop`s were the KV-cache batches:** `for layer in 0..n_layers` indexing
  `caches[layer]` sixteen times became `iter_mut().enumerate().take(n_layers)`, with `layer` kept for
  `layer_names` and the `blk.{layer}` name. Same machine code (iterating a slice is the same pointer
  walk), six bounds checks a layer gone from the source. `forward` also lost a parameter by naming
  its `(token, position)` pair: `DecodeStep`.
- **Deleted rather than allowed:** the never-wired FLUX VAE scaffolding (`VaeDecoder` config stub plus
  local copies of `ops::group_norm`/`resnet_block`/`upsample`, which duplicated `ops`) and
  `GpuDevice`'s `instance`/`adapter` handles. Not just tidying: the diffusion constructor *stopped
  uploading every BF16/F32 weight as f32*, which no forward path read — real VRAM and construction
  time for nothing. `cpu_q1_0_dot` (an unused scalar reference) is now used: the fused gate/up test
  cross-checks the shader against it instead of only against its own arithmetic.
- **My own mistake, kept:** the first scripted pass over the 14 `matmul_batch` call sites was off by
  one and deleted the row-buffer argument instead of the dead row count. One build caught it
  (`expected &mut Vec<f32>, found usize`); the repair script put the argument back per enclosing
  function (`rb` where it is a reference, `&mut rb` where it is a local) and both deleted-line counts
  came out exact.
- **Verified:** `cargo clippy --workspace --all-targets -- -D warnings` exit 0,
  `cargo fmt --check` exit 0, `cargo check --workspace --all-targets` warning-free,
  `cargo test --workspace -- --test-threads=1` green (42 quant, 170 tls, rest), release build clean.
- **Benchmark: this sweep is neutral, and the sweep itself found the real regression.** The 6-model
  mandate could not be run against the recorded baseline: the first attempts read 13-17 tok/s on
  Bonsai-1.7B-Q1_0 where the 2026-10-15 sweep recorded 47.1, and back-to-back runs of the *same*
  binary varied 25% (Taskmgr, Chrome, three other agents' apps and a second build were all live). So
  it was replaced with an interleaved A/B that needs no idle box: three arms alternating, three
  rounds, one model, 36.6 °C / 120% perf throughout.

  | arm | r1 | r2 | r3 | mean |
  |-----|----|----|----|------|
  | previous binary (2026-10-16 00:12, `-C target-cpu=native`) | 48.54 | 44.44 | 41.81 | 44.9 |
  | this sweep, built `-C target-cpu=native` | 42.12 | 44.72 | 43.40 | 43.4 |
  | this sweep, the committed `x86-64-v2` baseline | 12.64 | 13.05 | 12.65 | 12.8 |

  All three arms produced **byte-identical** text (`ed6e8cb4d000e460`, 50 tokens, temp 0). Two
  conclusions, and the second is the one that matters: **the allow sweep costs nothing** (44.9 vs 43.4
  mean, inside the 41.8-48.5 spread of the *same* arm), and **`26eab4b`'s `import x86-64-v2` baseline
  costs 3.4× on the 1.7B Q1_0** — a single-variable comparison (same source, `RUSTFLAGS` override in a
  side target dir, nothing else changed), not the 1-3% the commit message claims. The kernels are
  unaffected because they are `#[target_feature]` behind runtime probes; what is lost is the
  autovectorised f32 per-token work (rms_norm, silu, softmax, Q8 activation quantisation), which is a
  large fraction of a 1.7B decode step. Reported to SparkOC in the room with the binaries left in
  place (`target/buffy-ab/post-target`, `target/buffy-ab/native-target`, `target/release`); the
  native-vs-baseline decision is theirs to make, and the 6-model mandate should be re-run on an idle
  box once the baseline is settled.

### The `x86-64-v2` baseline cost 3.4× — because the kernels did not declare FMA and F16C (buffy, 2026-10-16)

**A note on how this entry got here.** Its first version called the committed baseline the problem and
`x86-64-v3` the answer. That was the interim reading, and it was wrong in the way that matters: v3
"recovered" the speed only because it puts FMA and F16C in the *baseline*, which supplies by accident
what the kernel attributes should have stated. Raising the flag would have hidden a real defect in
this crate behind a portability claim nobody could keep. `x86-64-v2` stays.

The symptom, end-to-end, 50 tokens, temp 0, same box, same prompts, the **same generated text** in all
arms (portable baseline before the fix → after the fix, still `x86-64-v2` → `x86-64-v3` → native):

| Bonsai-1.7B-Q1_0 | v2 before | v2 after | v3 after | native |
|---|---|---|---|---|
| tok/s, repeated runs | 13.61 / 13.92 / 13.89 | **43.03 / 43.48** | 45.37 / 44.96 / 40.13 | 50.56 / 51.12 |

The other two models this touches were measured and do **not** resolve on today's box: the same
binary read 10.67 and 12.70 tok/s on the 4B in consecutive runs (v3: 10.21, 12.52 — overlapping), and
9.55 against 19.23 on Ternary-1.7B-Q2_0, against a *before* figure that was steady at 15.17 / 15.54.
Two agents were benchmarking on this machine; only the 1.7B Q1_0 pair is far enough above the noise
floor (3×) to read. Re-run those two on an idle box before quoting them.

**Correction, because the working tree moved under this measurement.** `.cargo/config.toml` was
switched to `x86-64-v3` by SparkOC at 19:54, on the strength of this entry's *earlier* reading. Every
`v2` figure above was therefore re-taken with `RUSTFLAGS="-C target-cpu=x86-64-v2"` overriding that
file (`target/buffy-ab/v2fix-target`), and the first version of this table called a 50.05/50.86 run
"v2 after" when it had in fact been built at v3. The kernel-level numbers did not move when re-taken
(2.75 / 2.80 GB/s at v2, 3.11 at v3), so the mechanism is unchanged — but the conclusion changed
shape: **the fix removes the 3.4×, not the baseline.**

**The mechanism, from the emitted assembly rather than from theory.** Stage timings first put the loss
in the five matmul stages (ffn_gate_up 4,852 → 32,266 µs at v2), not in the f32 glue (`qkv_quant`,
`rms_norm`, `silu_mul` all within 1–2×). The bare kernel agreed — same source, same CPUID decode, same
reported `avx2` dispatch, **0.52 GB/s against 3.12** — so the selection was right and the code behind
it compiled worse. `--emit=asm` said why:

| `dot_q1_0g128_q8_0_ptr_avx2` | v2 before | v2, declarations only | v2, + helper | v3 / native |
|---|---|---|---|---|
| instructions per row | 511 | 500 | **112** | 100 |
| out-of-line calls | 15 (`_mm256_fmadd_ps` ×5, `f16_to_f32_x86_f16c` ×5, probe ×5) | probe ×5 | **0** | 0 |
| `benchk` GB/s (1.7B Q1_0) | 0.52 / 0.52 | 1.76 | **2.75 / 2.80** | 3.11 / 3.12 |

Two separate omissions, both the same class of bug:

1. **`fma` and `f16c` were missing from every kernel that used them.** A body compiled with
   `#[target_feature(enable = "avx2")]` alone cannot inline `_mm256_fmadd_ps` or `half`'s f16
   conversion, because those callees need features this function never claimed: each use became a
   call, 15 of them per row with 16 `vzeroupper`s around them. Declaring them recovered 0.52 → 1.76.
2. **`half`'s runtime-probed `to_f32` inlined both arms.** With `f16c` now declared, the probe and
   *its software fallback* inline into the block loop; the register pressure that costs is worth more
   than the conversion. `crate::f16_bits_to_f32` (one `vcvtph2ps`, F16C declared on the caller) took
   1.76 → 2.83. Only 35 sites inside declared kernels changed; every other conversion in the crate
   keeps `half`'s runtime probe, so a CPU without F16C still runs — that is the whole point of the
   baseline.

Same treatment for Q2_0 (`dot_q2_0_q8_0_lut_avx2` 256 → 167 instructions) and for **PTQ1_0**
(`dot_ptq1_0_q8_0x2_avx2` 464 → 301, F16C only — it uses no FMA), which covers the Ternary-Bonsai-2
family. The AVX-512 arms were already fine: LLVM's `avx512f` implies FMA and F16C, which the v2
assembly confirmed before anything was changed (no out-of-line calls there at all).

**Three things the fix also put straight.** (a) The gates now name all three features, so dispatch and
`kernel_name()` still read one predicate per file — `avx2_kernel_available()`,
`avx2_lut_available()`, `ptq1_0::avx2_available()` — and the feature-report test restates the same
rule. (b) That closes a latent SIGILL: the old `avx2`-only gate let a body execute FMA on a CPU that
reported AVX2 without FMA (a masked CPUID under a VM is the realistic case). (c) The three unannotated
256-bit `hsum` helpers were given their own `avx2` attribute — a true statement about their bodies,
with no measured effect; recorded here so nobody re-derives it as the cause.

**The durable lesson:** the `#[target_feature]` list must name every feature the body executes. A
tree of AVX2 kernels compiled at a baseline without AVX2-FMA-F16C loses 2–6× *silently* — the startup
banner keeps reporting `avx2`, because the banner describes the CPU, not the code that was build.

- **Directional only:** small-tensor rows (3.34 MB) vary run to run, and `vnni256` reads slow at every
  baseline (0.55–0.59) because dispatch does not select it on Zen 4. Not a live path.
- **Every perf number recorded before today was taken with `target-cpu=native`.** The baseline now
  costs ~1% rather than 3.4×, so the older tables are comparable again — but the box was busy today,
  so re-run the 6-model sweep on an idle one before treating any row as a baseline.

- **AGENTS.md now states the rule**, not just the practice: no `#[allow(...)]` anywhere, never a
  "pre-existing" or "OK here" warning, the three-part gate, and a finding that cannot be fixed
  without changing behaviour is a finding to report and measure. Two stale claims in that file were
  corrected: `cargo test --test-threads=1 -p …` does not parse (and never did — cargo rejects it),
  and "pre-existing clippy warnings in GPU stubs are OK" described a state that no longer exists.

### hearth-tls slice 20 — the verifier: the first thing in the crate that can accept (buffy, 2026-10-16)

Every rule existed and nothing called any of them: `RejectAll` was the only `CertificateVerifier`,
so no `https://` path could be opened on this crate's account. This slice is the composition, as one
named type, plus the fixtures that make the composition testable.

- **`verifier.rs` — `PinnedVerifier`**, over anchors and CRLs the *caller* supplies and a `Policy`.
  Three checks in a stated order: the chain from the wire bytes (`parse_certificate_list` then
  `verify_chain`), then the leaf's purpose and name (`leaf::verify_leaf`, with the name from our own
  ClientHello), then the `CertificateVerify` signature with the leaf's key
  (`verify::verify_handshake_signature`). A failure is a typed `VerifyFailure` per stage
  (`Chain`/`Leaf`/`Signature`), and the `CertificateVerifier` impl renders it — the trait's `String`
  is that same `Display`, asserted equal in a test, so a log and a match cannot tell two stories.
  Order is argued, not incidental: the chain first because a name match against an unanchored
  certificate means nothing, purpose before name because `leaf.rs` already argues that, and the
  signature last because it is the only stage that costs arithmetic and the only one that proves the
  peer holds the key rather than pointing at it.
- **The anchors are an argument, and the root store stays the caller's.** `chain::verify_chain`
  already took them; `PinnedVerifier` borrows them (nothing is copied, so a root store can be parsed
  once at startup and shared). Decided against putting a store here *again*, explicitly, because the
  tempting version — a small pinned set inside `hearth-tls` — would make the downloader's trust
  anchors invisible and unreviewable. An empty anchor set is a working verifier that refuses
  everything; there is no default and no accept-all.
- **New fixture `LEAF_EC_VALID`, for a reason that is a bug class.** `LEAF_EC`'s window opens after
  its issuer's has closed, so *no* instant verifies its chain — which meant the whole verifier could
  only ever be exercised with an RSA leaf. A verifier that only works for RSA is the classic wrong
  dispatch. The new leaf is P-256 with a window inside the intermediate's, so the same
  `CertificateVerify` construction is verified end to end with `0x0403`; the oracle also checks the
  path (`ec-leaf-and-intermediate-to-root`, OpenSSL accepts).
- **`CertificateVerify` vectors in the generated table**, signed with the fixture leaves' own keys
  over the RFC 8446 §4.4.3 bytes (64 spaces, the context string, a zero byte, a 32-byte hash): one
  RSA-PSS, one ECDSA, and the RSA one with its last byte flipped. **Each accepted vector is verified
  by the OpenSSL CLI before it is written down** (`openssl dgst -verify`, with the PSS options), and
  the generator *raises* if the forged one verifies — so an accepted vector is one a second
  implementation accepts, and the refusal is one a second implementation also refuses. The PSS salt
  is 32 bytes because §4.2.3 requires salt = hash length. RSA-PSS and ECDSA both draw fresh
  randomness, so those bytes change on every regeneration: the tests assert they *verify*, not that
  they match, which is why a fresh regeneration is green.
- **Reachability, run not argued: 6 rounds, every stage caught.** The chain call removed, the caller's
  `Revocation` replaced with `Unchecked`, the leaf check removed, the signature check replaced with
  `Ok(())`, `signed_data`/`signature` swapped, and the trait's entry point short-circuited. The last
  one is caught by the *unit* test in `verifier.rs` and by the fixture suite both — the same
  two-nets lesson slice 19 recorded, which this script was written with from the start.
- **Three test-authoring bugs of mine, caught by running rather than by reading:** the context-string
  slice compared 33 bytes against a 34-byte literal (the string is 33 bytes *plus* the separator) —
  now sliced by `context.len() + 1`; a `ROOT_SHADOW` anchor was expected to be `Unanchored` when the
  documented behaviour is `Signature { certificate: Chain(1) }`, because it *is* a name-matched
  candidate that fails its signature (the wrong-key-anchor case, not the no-candidate case — both are
  now controls, the second using `SELF_SIGNED_EC`); and a leftover `good` from the previous test.
- **No inference code touched, so the benchmark mandate did not apply.** **242 checks green** on the
  hardware path and on `HEARTH_TLS_NO_HW=1` (238 before), clippy `-p hearth-tls --all-targets -D
  warnings` exit 0, `cargo fmt --check` clean, `cargo doc -p hearth-tls --no-deps` zero warnings, and
  the generated file fmt-stable and green after a fresh regeneration. 33 certificate fixtures, 14
  CRLs, 24 oracle paths. Still `std`-only with an empty `[dependencies]`.
- **Still absent, and stated as such in the code:** no OCSP, no stapling, no CRL fetching (a list is
  bytes the caller brings, and keeping it fresh needs a network), and no transport — nothing in this
  crate has ever opened a socket, so the downloader's `https://` wiring is still the next real step.


### hearth-tls slice 19 — revocation: CRLs, and the version field two readers had wrong together (buffy, 2026-10-16)

Revocation was the last **absent** check rather than an untested one. It is now present as
caller-supplied CRLs, and the slice's most useful result was a bug in *both* the parser and the
fixture generator, which agreed with each other and with nothing else.

- **`crl.rs`**: a strict reader for `CertificateList` — issuer name, `thisUpdate`/`nextUpdate`,
  revoked serials as magnitudes, and the bytes a signature covers. Critical extensions it does not
  understand are refusals (§5.2), which is what refuses a **delta CRL** standing in for a complete
  one and an **indirect** entry (`certificateIssuer`, refused by name whether or not it is critical).
  `nextUpdate` is optional and its absence means "no stated expiry", and v1 (no version field, no
  extensions) and v2 are both read.
- **`chain.rs` gains `Revocation`**: `Unchecked` (a word at the call site, because a revoked
  certificate verifies in that mode — as it does for every stack that never turned checking on) or
  `Required(&[CertificateRevocationList])`. In `Required` mode every certificate in the path needs a
  CRL from its issuer (`NoCrlFor` — unknown is refused, not accepted), the issuer must claim
  `cRLSign` when it states a `keyUsage`, every supplied CRL for that issuer must verify its
  signature, **the freshest `thisUpdate` speaks**, a stale or not-yet-valid decisive list is refused,
  and two equally fresh lists that contradict each other are `AmbiguousRevocation` (this crate's own
  rule: there is no honest way to pick). The anchor is checked only against a list it published about
  itself, which is the one case OpenSSL's `-crl_check_all` catches and a permissive reading would not.
- **The bug: a CRL's `version` is a plain `INTEGER`, not the `[0] EXPLICIT` a certificate wraps its
  version in** (a `TBSCertList` starts with it, so nothing needs disambiguating it; a certificate
  starts with a serial number, which is why the tag is there). The parser read the certificate's
  shape and my hand-built fixtures *wrote* it, so thirteen unit tests passed on CRLs that OpenSSL
  could not load at all — and every real v2 CRL would have been refused. It surfaced only because
  the generator asks OpenSSL to read every fixture and **raises when the tool cannot**: "Could not
  find CRL" is what an invalid version tag looks like from the outside. Fixed in both, with a test
  that the tagged form is refused here too, and the reachability round for it turns **seven** tests
  red.
- **Measured against the OpenSSL CLI, 17 recorded cases** (`openssl verify -crl_check_all` /
  `-crl_check -CRLfile`): clean path, leaf revoked (23), intermediate revoked (23), **anchor revoked
  by its own list (23 at depth 2)**, missing CRL at either depth (3), expired (12), not yet valid
  (11), no `nextUpdate` (accepted), v1 (accepted, and a v1 revoking list still revokes), an unknown
  critical extension (36), a delta indicator (**accepted** — a divergence, we refuse), a wrong-key
  signature (8), a stale revoking list beside a fresh clean one (**accepted** — the freshest speaks,
  which is what that rule copies), and a CA without `cRLSign` (35). Two divergences, both
  recorded with their reason, both with this crate the stricter side; one case asked with
  `-crl_check` shows why the leaf alone is not enough.
- **Reachability, run not argued: 13 rounds, every rule caught.** cRLSign, CRL signature verification,
  freshest-speaks, ambiguity, the window, the revoked decision, `NoCrlFor`, the anchor exception, the
  unknown-critical refusal, `certificateIssuer`, the algorithm-agreement rule, the version read, and
  the v1-with-extensions rule. Three first appeared to survive — because the script ran only the
  integration target and their nets are the unit tests in `crl.rs`; the script now runs both, which is
  itself the finding worth keeping.
- **Harness faults, three of them, all of which looked like verdicts:** `openssl verify` reads the
  argument after an option as its value, so an option placed *after* the leaf path makes every case
  come back error 20 (and passing no `-untrusted` does the same); `line.startswith("Serial Number:")`
  reads *no* serials out of any CRL with entries, because entries are printed indented; and appending
  an extension to an existing `[0]` element without rebuilding its length produces a CRL OpenSSL
  rejects for a reason that has nothing to do with the extension. The generator now asks whether the
  tool can **read** a CRL before recording any verdict about it.
- **No inference code touched, so the benchmark mandate did not apply.** 238 checks green on the
  hardware path and on `HEARTH_TLS_NO_HW=1` (217 before), clippy `--all-targets -D warnings`, fmt,
  `cargo doc` and the workspace check clean, generator output fmt-stable and passing after a fresh
  regeneration. 32 certificate fixtures, 14 CRLs. Still `std`-only, still an empty `[dependencies]`,
  still nothing in the crate calling any of it: `RejectAll` remains the only `CertificateVerifier`, and
  there is still **no OCSP, no stapling and no CRL fetching** — a CRL is bytes the caller brings.

### hearth-tls slice 18 — AKI/SKI matching: a link check the signature cannot make (buffy, 2026-10-16)

Slice 17 ended with the question of whether key identifiers matter beyond `-x509_strict`. Measured, and they do:
OpenSSL refuses a certificate whose `authorityKeyIdentifier` points nowhere **while this crate accepted it** — a
disagreement in the direction the oracle test forbids, so it could not be written off.

- **The parser learned the two context-tagged extensions.** `x509.rs` now reads `subjectKeyIdentifier` (an `OCTET
  STRING` inside the extension) and `authorityKeyIdentifier`, whose `keyIdentifier` field is **`[0]` implicit** —
  the contents are the identifier bytes with no nested `OCTET STRING` — with the issuer-and-serial form read past
  rather than read. The generated table now records both for every fixture, so all 30 are compared against
  OpenSSL's parsing of the same bytes, including the deliberately wrong one.
- **The rule: when a child states a key identifier and its issuer states one, they must agree**
  (`ChainError::AuthorityKeyMismatch`, RFC 5280 §4.2.1.1). The signature check **cannot** catch this on its own,
  and that is the whole point of the rule: a certificate whose identifier points elsewhere was still signed by
  whoever signed it, so without the comparison the path verifies. OpenSSL's answer is to treat it as a link that
  cannot be made at all — error 20, "unable to get local issuer certificate" — and our error now says which
  certificate and why.
- **Two absences are not refusals**, and the tests say so rather than the comments: a certificate may carry no
  `authorityKeyIdentifier` (most fixtures here do not) and an issuer may carry no `subjectKeyIdentifier` (`ROOT`
  does not), and there is then no contradiction to refuse.
- **Checked on both sides of the anchor**, the way the other anchor rules are: a candidate naming a different key
  than the chain's last certificate is not this chain's anchor, so the next candidate is tried.
- **Two independent nets caught the regression**: neutering the check turns both the direct fixture test into
  `Ok(())` where `AuthorityKeyMismatch(Chain(0))` is expected, and the oracle into "1 of 23 paths disagree".
- **217 checks green** on both paths (216 before), clippy, fmt, `cargo doc` zero warnings, workspace exit 0.
  30 fixtures, 23 oracle paths, 5 recorded divergences (all strict, none new).

### hearth-tls slice 17 — the one rule no fixture could reach: key rollover and the self-issued count (buffy, 2026-10-16)

`intermediates_below` excludes self-issued certificates from a `pathLenConstraint` count, which is the rule
that makes a **key rollover** possible, and **in all 24 fixtures at the time every intermediate was
non-self-issued** — so the branch that does the excluding never excluded anything. Found by reading the code
after the oracle work, not by a failing test, which is the point: a rule with no fixture is a rule nobody has
checked, and it looked exactly like a rule that worked.

- **A key rollover is now a fixture.** `ROLLOVER_CA_A` (signed by the root, `pathLenConstraint: 0`),
  `ROLLOVER_CA_B` (signed by `ROLLOVER_CA_A`'s key, **the same distinguished name**, a different key) and a
  leaf under `ROLLOVER_CA_B`. The path is legal *only* because `ROLLOVER_CA_B` is self-issued — the same DN in
  the subject and issuer fields, whatever key signed it — and therefore does not spend `ROLLOVER_CA_A`'s
  allowance of none.
- **Proven load-bearing by neutering `self_issued` to `false`:** the rollover path is then refused with
  `PathLengthExceeded { ca: Chain(2), allowed: 0, found: 1 }`. Restored, and the restore verified by an empty
  `git diff` on `chain.rs`.
- **The test carries its own contrast**, so the assertion is about self-issuance rather than about
  permissiveness: the sibling strict CA, one property different, does refuse its non-self-issued child.
- **The oracle agreed with us only after the fixtures became realistic, and that disagreement was the most
  useful thing in the slice.** With no key identifiers, OpenSSL matched these certificates **by subject name
  alone** — a rollover is ambiguous that way by construction — and refused two sound paths with error 7
  ("certificate signature failure") and error 19 ("self-signed certificate in certificate chain") *while this
  crate accepted them*. That is a disagreement in the direction the oracle test forbids, and it correctly
  refused to let me write it off as a divergence. The fix was the fixtures: these three, and only these three,
  now carry `subjectKeyIdentifier`/`authorityKeyIdentifier`, and both paths then agree. So the earlier
  `-x509_strict` finding (error 85, missing AKI/SKI) is not a strict-mode nicety — key identifiers are how a
  real verifier tells two same-DN certificates apart, and without them the *fixtures* are what is unrealistic.
- It also closes the loop on the `-auth_level`/`-partial_chain` flags: no flag made OpenSSL take the rollover
  path, because no flag substitutes for a key identifier.
- **216 checks green** on both paths (215 before), clippy, fmt, `cargo doc` zero warnings, workspace check exit
  0, generator output fmt-stable. 27 fixtures, 21 oracle paths, 5 recorded divergences (all strict).

### hearth-tls slice 16 — what the oracle says about chain *shape*: selection, not ordering (buffy, 2026-10-16)

Four paths added to the oracle table, all about the shape of the presented list rather than about a rule:
the root included in the chain, the chain out of order, the leaf twice, and an unrelated certificate padded in.
The last three are a **finding**, and the first is reassurance.

| path | OpenSSL 3.5.5 | this crate |
|------|---------------|------------|
| root included in the chain | OK | `Ok` |
| chain out of order | **OK** | `BrokenLink { index: 0 }` |
| chain with the leaf twice | **OK** | `NotACa(Chain(1))` |
| chain with an unrelated certificate | **OK** | `NotACa(Chain(2))` |

- **OpenSSL selects a path out of the pool; this crate builds exactly the list it is given.**
  `verify_chain` compares each certificate's issuer name with the next certificate's subject name, so an
  unordered list is a broken link, and a padded or repeated one puts a non-CA where an issuer belongs. Both
  refusals are deliberate and both are RFC 8446 §4.4.2's own requirement — each certificate must certify
  the one before it — and OpenSSL, like the browsers, rebuilds instead. Being stricter is the safe
  direction, which is the direction the oracle test permits, so all three are recorded as divergences with
  their reason attached. The cost is real and now written down in `chain.rs`'s "what this does not do":
  **a server whose chain is out of order or padded is refused, and a caller who needs that tolerance has to
  select the path before calling here.**
- **These three cases also exercise "structure before links" with junk input**, which is the one thing the
  old fixture set could not: the refusals are `NotACa` at the position of the offending certificate, not a
  broken link at the end, because the structural pass runs first. That is exactly the property the module
  docs claim, now shown on input nobody constructed to be a path.
- **The oracle test reports every disagreement at once.** Adding a path used to mean one test run per
  surprise, because the assertion stopped at the first; it now collects them and prints the set, which is
  what a reviewer needs — and the fmt check caught two of my own edits in the same round, which is the
  cheap kind of failure.

19 paths now, 5 recorded divergences, all five in the strict direction.

### hearth-tls slice 15 — an oracle outside the crate: the same paths, put to the OpenSSL CLI (buffy, 2026-10-16)

Every check so far compares our reading of a certificate against OpenSSL's *parser*. The **policy** — which
rules run, and what each one means — was still only compared against my own reading of RFC 5280. So fifteen
paths are now put to a second implementation of path validation, at the same instants, and the verdicts are
recorded in the generated table (`VERIFY_CASES`) rather than taken at test time: a test that shells out to a
tool is a test that quietly skips when the tool is absent.

`openssl verify -partial_chain -auth_level 0 -attime <instant>`, OpenSSL 3.5.5. The flags are part of the
comparison and documented in the generator: `-partial_chain` is the closest equivalent to this crate's rule
that an anchor need not be self-signed, and `-auth_level 0` keeps OpenSSL's separate security-level policy
out of a comparison about structure.

| path | OpenSSL 3.5.5 | this crate |
|------|---------------|------------|
| leaf + intermediate → root | OK | `Ok` |
| intermediate pinned as the anchor | OK (`-partial_chain`) | `Ok` |
| three deep, root allows one | error 25 at depth 3 | `PathLengthExceeded(Anchor(0))` |
| issuer without `keyCertSign` | error 79 at depth 1 | `MissingKeyCertSign(Chain(1))` |
| anchor stating `cA: FALSE` | error 79 at depth 1 | `NotACa(Anchor(0))` |
| `cA: FALSE` certificate as a link | error 79 at depth 1 | `NotACa(Chain(1))` |
| self-signed `cA: FALSE` pinned as itself | **OK** | `Ok` |
| no `basicConstraints`, pinned as itself | OK | `Ok` |
| root the caller does not trust | error 20 at depth 1 | `Unanchored` |
| anchor whose window closed | error 10 at depth 1 | `Expired(Anchor(0))` |
| issuer expired under a valid leaf | error 10 at depth 2 | `Expired(Chain(1))` |
| broken link, intermediate omitted | error 20 at depth 0 | `BrokenLink { index: 0 }` |
| strict sibling CA as the anchor | error 25 at depth 2 | `PathLengthExceeded(Anchor(0))` |
| CA certificate as the server | **OK** | `CaUsedAsLeaf` — divergence, this crate is stricter |
| 1024-bit anchor | **OK** (`-auth_level 0`) | `KeyTooSmall` — divergence, this crate is stricter |

What the oracle actually bought:

- **It measured the claim I had been repeating from memory.** The identity-pin exemption from slice 14 — a
  self-signed `cA: FALSE` certificate pinned as its own anchor — is accepted by OpenSSL, which is what the
  slice-14 commit said and could not previously have shown.
- **It independently corroborates slice 14's new rule**: OpenSSL refuses a non-CA anchor verifying another
  certificate with error 79, the same verdict this crate reaches for the same reason.
- **Two divergences, both in the safe direction**, and the test enforces the direction: a disagreement is
  only permitted where OpenSSL accepts and this crate refuses. Being the more permissive side is a failure,
  not a footnote. `CaUsedAsLeaf` is TLS-specific and correct; the key-size floor is this crate's own policy.
- **With the `keyCertSign` rule neutered, the oracle fails by name** — "issuer-without-keyCertSign: this
  crate says `Ok(())`, OpenSSL says ok=false (error 79)" — a second implementation catching a regression in
  a rule it was never told about.

Two harness traps for the record, both of which produced results that looked like verdicts and were not:
`-untrusted` with an empty file makes openssl refuse to run at all ("Could not find untrusted
certificates"), and `-x509_strict` fails *every* fixture with error 85, "Missing Authority Key Identifier",
because these certificates carry no AKI/SKI. The strict mode is therefore not usable here without new
extensions — a candidate for the next fixture round, not a verdict about the crate.

Also worth writing down: **the blamed certificate is not comparable.** OpenSSL numbers depths from the leaf
and reports the highest expired one; this crate reports the first it reaches. For the expired-issuer path
they are four depths apart while both refusing correctly, so the test compares verdicts only.

**215 checks green** on both paths (214 before), clippy `-D warnings`, fmt, `cargo doc` zero warnings,
workspace check exit 0, and the generator's output is `cargo fmt`-stable (regenerating does not move the
file). No inference code touched, so the benchmark mandate did not apply.

### hearth-tls slice 14 — the anchor's `basicConstraints`, and the documentation defect it exposed (buffy, 2026-10-16)

SparkOC's review (#93) settled the asymmetry reported at the end of slice 13: close it, with
`basicConstraints` **present** requiring `cA: TRUE` and **absent** allowed, since a bare pin or a v1-style
certificate has no extension to lie with and refusing it would break the pinning the anchor docs allow.
Implemented in `check_anchor_is_ca`, with two refinements the fixtures forced.

- **The rule as first written broke a test, and the test was right.** `SELF_SIGNED_PSS` is `cA: FALSE`
  **present** and is pinned as its own anchor by `every_suite_verifies_as_a_chain_link`. Refusing that would
  mean a caller cannot pin a self-signed server certificate. The distinction the rule needed: an anchor that
  *is* the certificate it verifies is an **identity pin** — it proves its own identity rather than claiming
  a right to issue — so the exemption is `anchor.der() == verified.der()`, which the name check above can
  only have matched while that certificate is self-issued. It cannot be walked out of: the same certificate
  used to authenticate anything else is refused, at the anchor and as a chain link. OpenSSL behaves the same
  way — a self-signed cert in the trust store verifies itself, a non-CA used as an issuer is "invalid CA
  certificate" (error 24).
- **`is_ca` could not express the rule.** It is false both for a stated `cA: FALSE` and for an absent
  extension, so `Certificate` gained `has_basic_constraints()` (`x509.rs` tracks it in the extensions pass),
  the generator records it per fixture, and the automatic parse test compares it — which pins the absent
  case on `SELF_SIGNED_EC`, the certificate with no extensions at all.
- **Two new certificates (24 total)**: `NON_CA_SIGNER` (its issuer marked it non-CA, it claims
  `keyCertSign` anyway, and it signs a child) and that child. The test asserts the signature really verifies,
  so the `basicConstraints` claim is the only defect.
- **Reachability run again**: neutering the check turns the refusal `Ok(())`. Separately, the identity-pin
  half was proven by accident at first — the check without the exemption breaks a pre-existing test — and
  then pinned deliberately by `pinning_a_certificate_is_an_identity_and_not_an_issuance`.
- **A documentation defect fixed, which is the part worth remembering.** `chain.rs`'s "the two rules that
  are read out of a CA certificate" section still said `keyCertSign` "is **not** required of an anchor" —
  false since slice 12, when the check moved onto the anchor, and nothing failed, because no test reads a
  doc comment. It now enumerates three rules and states what each site requires.
- **214 checks green** on both paths (212 before), clippy `-D warnings`, fmt, `cargo doc` zero warnings,
  workspace check exit 0. No inference code touched, so the benchmark mandate did not apply.

### hearth-tls slice 13 — the anchor's own window and key size, and two tests that had assumed a 2048-bit world (buffy, 2026-10-16)

`chain.rs` was covered everywhere except two call sites, and they were the same shape: the *anchor's* own
validity window and the *anchor's* own key size only run when the rest of the path is sound, so no fixture
anchor could reach either.

- **Four new certificates (22 total).** `EXPIRED_ANCHOR` stopped being valid in 2020 and `LEAF_OF_EXPIRED_ANCHOR`
  is valid from 2026, so at the instant of the check the leaf is inside its own window and its issuer is not;
  `WEAK_ANCHOR` is an ordinary CA holding a real 1024-bit key, below `Policy::min_rsa_bits`. Each has a leaf it
  really did sign — the test asserts that signature — so the anchor check is the only thing that can refuse
  the path.
- **Reachability in two passes, because the test stops at its first assertion.** Deleting both anchor checks
  turns the expired path `Ok(())`; deleting only the key check then turns the weak-anchor path `Ok(())`. Both
  restored, and the restore was verified by an **empty `git diff`** on `chain.rs` rather than by eye — which is
  worth doing, because my first restore spliced a brace and only the diff caught it.
- **The 1024-bit fixture broke two older tests, and they were wrong, not the fixture.** One hard-coded "an
  RSA signature is 256 bytes" (a signature is the *issuer's* modulus length) and the generated modulus hex was
  padded to a fixed 256 bytes. The generator now emits `signature_len`, read out of each fixture's own DER with
  a small TLV walker, and the modulus at its real width — so both assertions compare against OpenSSL's numbers
  instead of an assumption that every anchor in the set is 2048 bits. That new field also pins the `BIT STRING`
  reading: treating the unused-bits octet as part of the value would leave every signature one byte too long.
- **Call sites with no negative test: none.** For the first time in this file, every rule has one, on every
  site it is checked at.
- **212 checks green** on the hardware path and with `HEARTH_TLS_NO_HW=1` (211 before), clippy
  `-p hearth-tls --all-targets -D warnings` exit 0, `cargo fmt --check` clean, `cargo doc` zero warnings,
  `cargo check --workspace --all-targets` exit 0. No inference code touched, so the benchmark mandate did not
  apply.

### hearth-tls slice 12 — `pathLenConstraint` and `keyUsage.keyCertSign`, each proven by removing it (buffy, 2026-10-16)

`chain.rs`: the last two structural rules, and the first slice where the *test* for each rule was made to
fail on purpose to show the rule is what produces the refusal.

- **`pathLenConstraint` counts non-self-issued CAs below the CA that stated it**, so the leaf never counts
  and a CA sitting in the chain does not count against its own limit when it is also the anchor. Both sides
  are tested: `TOP_CA` (allows one) has one below it in a legal path, and the root (allows one) refuses the
  same path with two below it, at `Position::Anchor(0)`.
- **Two mirrored sub-hierarchies make the contrast exact.** The strict subtree shares *both* subject names
  and both key usages with the normal one and differs only in one CA's limit, and `TOP_CA`'s key really did
  sign everything in it — so the test asserts every link name and every signature itself instead of
  claiming them in a comment. That correction came from the first reachability run: with the original
  fixtures the refused path also had a bad link signature, so "the limit is the only difference" was
  prose. With the mirrored subtree, removing the limit check turns the path `Ok(())`.
- **`keyUsage.keyCertSign`**: RFC 5280 §4.2.1.9. `CA_NO_KEY_CERT_SIGN` is `CA:TRUE` with
  `pathLenConstraint: 0` and a `keyUsage` of `digitalSignature`+`cRLSign`, so the two extensions disagree.
  Refused as an issuer *and* as a pinned anchor — in the anchor case the test also asserts the signature it
  would otherwise pass, which pins the ordering: the usage is checked before any trust is conferred.
- **Reachability, run rather than argued, in three passes.** With `check_key_cert_sign` neutered the
  key-usage test fails with `left: Ok(())`. With the *chain* `path_len` site neutered the limit test fails
  with `left: Ok(())` at the chain assertion. With only the *anchor* site neutered — the first two rounds
  stop the test before its anchor assertion, so that path was not yet proven — the chain assertions pass
  and the anchor assertion fails with `left: Ok(())`. Every site was restored immediately and the restore
  is in the diff (`chain.rs`: +125/−7, the two rules and their docs).
- **Six new certificates (18 total)** from `scripts/tls_x509_fixtures.py`, plus a harness fix: the generator
  now records each fixture's `issuer_cn`, so the signature test iterates `EXPECTATIONS` and finds each
  certificate's issuer by name *and* key — a fixture can no longer be silently omitted from it, and the
  `ROOT_SHADOW` name collision that used to need a hand-written table is handled by trying both.
- **211 checks green** on the hardware path and with `HEARTH_TLS_NO_HW=1` (209 before, the two new tests),
  clippy `-p hearth-tls --all-targets -D warnings` exit 0, `cargo fmt --check` clean, `cargo doc` zero
  warnings, `cargo check --workspace --all-targets` exit 0. No inference code touched, so the 6-model
  benchmark mandate did not apply.

### hearth-tls slice 11 — host name matching and the leaf's purpose, and two wrong rules found by their own tests (buffy, 2026-10-16)

`leaf.rs`: the two checks on the leaf itself. A chain says a certificate is authentic; these say it is the
*right* certificate, and they were the last two between this crate and an `https://` URL.

- **`subjectAltName` is the only place a name is looked for.** No `commonName` fallback (RFC 6125 §6.4.4
  deprecated it, and a name in a free-text field is not a constraint), and only `dNSName` entries are
  compared — `LEAF_EC` has a `commonName` of `ec.example` and no SAN, and is the fixture that refuses its
  own subject's name with `NoDnsNames` rather than matching it.
- **The wildcard policy is a pure function**, `leaf::dns_name_matches(pattern, name)`, so the part that is
  usually wrong in a name matcher is unit-tested without a certificate: one whole leftmost label only, the
  pattern must leave at least two labels (`*.com` matches nothing — the strongest rule available without a
  public suffix list), no `*` anywhere else, and a server name that is itself a pattern never matches.
- **A test found one wrong rule, and fixing it exposed a worse one.** `::1` was reported as a *malformed
  name* rather than an address, because a colon is not a host-name byte and the address check ran second —
  which left the IPv6 branch of `is_ip_literal` unreachable and the module docs describing behaviour the
  code could not reach. Reordering the checks fixed the diagnosis and then failed again immediately:
  "contains a colon" was calling `models.example:443` an address literal. The rule is now address-shaped
  — four digit runs, or two or more colons with nothing but hex digits, colons and dots. Every version
  refused the connection, so nothing unsafe shipped; what was wrong twice was the *diagnosis*, and only a
  test that asserts the error could see it.
- **Two fixtures make the purpose checks provable**, each with its own key per the generator's
  one-key-per-identity rule: `LEAF_CLIENT_AUTH` (a perfectly good chain whose `extKeyUsage` is
  `clientAuth` alone) and `LEAF_NO_DIGITAL_SIGNATURE` (`keyUsage` without the bit that signs a
  `CertificateVerify`).
- An absent `extKeyUsage` means any purpose is allowed (RFC 5280 §4.2.1.12), and the parser refuses an
  *empty* extension, so an empty slice can only mean "absent" — the purpose check relies on that, and
  says so in the code that relies on it.

209 checks green on the hardware path and on `HEARTH_TLS_NO_HW=1` (195 before), fmt, clippy
`--all-targets -D warnings`, `cargo doc` (zero warnings) and `cargo check --workspace --all-targets` all
clean. Still nothing calls any of it: `RejectAll` remains the only `CertificateVerifier`, and there is
still **no revocation checking of any kind**.

### hearth-tls slice 10 — chain building, and the fixture a name-only matcher needs (buffy, 2026-10-16)

`chain.rs`: framing, structure, links and anchoring, over the scheme dispatch in `verify.rs`. The
entry point is `verify_certificate_list(certificate_list, anchors, time, policy)` — exactly the four
fields a verifier is handed, so a `CertificateVerifier` impl is now a call rather than a project.

- **Structure first, then links, then the anchor**, so a chain of the wrong shape is refused before any
  public-key operation runs on hostile input.
- **A name match is not trust, and it is now provable rather than asserted.** `ROOT_SHADOW` is a
  self-signed CA carrying the root's *subject name* with a different key, so anchoring the fixture chain
  on it fails with `Signature { certificate: Chain(1) }` — and with both anchors present the real one is
  found, so a failed candidate does not end the search. That path, the one a name-only matcher would
  accept, had no negative test until the fixture existed: the same pattern as the shared-key defect in
  slice 9, one level up.
- **One framing rule, two callers.** `client.rs`'s `parse_certificate` now calls `chain::split_entries`
  instead of keeping its own copy of the entry walk, because the verifier applies the same rule to the
  same bytes and two copies are two chances to disagree. Its existing tests were the check that the
  extraction changed nothing.
- **A test expectation that was wrong, kept in the file as a comment**: dropping a list's final byte
  truncates an *entry body*, so the refusal names that entry's offset — it does not leave a partial
  length prefix, which needs the list cut one or two bytes past an entry. Both cases are asserted, with
  offsets derived from the fixture lengths rather than transcribed.
- Every instant in the chain tests is derived from the certificate under test (`not_before`,
  `not_after`), so both validity bounds are checked inclusively and exactly, and a regeneration cannot
  silently move them.

**Deliberately not here:** revocation of any kind (no CRL, OCSP or stapling), host name matching,
`pathLenConstraint`, `keyUsage.keyCertSign`, and any name-constraint or policy processing. Revocation
and name matching are the downloader's remaining blockers; the two extension rules need fixtures that
can make them *fail* and are named in `RESUME.md`. Two call sites in the anchor loop — the anchor's own
validity window and its key size — have no negative test, because every fixture anchor satisfies both
whenever the chain is still valid; the functions they call are negatively tested on chain elements.

195 checks green on the hardware path and on `HEARTH_TLS_NO_HW=1` (177 before), fmt, clippy
`--all-targets -D warnings`, `cargo doc` with zero warnings and `cargo check --workspace --all-targets`
all clean. Still `std`-only, still `[dependencies]`-empty, still unreachable from any `https://` path.

### hearth-tls slice 9 — PKCS#1 v1.5, the scheme/key decision, and a fixture that shared a key (buffy, 2026-10-16)

`rsa.rs` gained `sha256WithRSAEncryption` — the one RSA suite this crate lacked and the one nearly
every certificate in the world carries — and a new `verify.rs` holds the decision half's first pieces:
`Suite`, `VerifyError`, `verify_certificate_signature` and `verify_handshake_signature`.

- **The `DigestInfo` is checked structurally, never searched for.** `check_pkcs1v15_sha256` accounts for
every byte by position: block type, at least eight `0xFF`, the zero separator, a `DigestInfo` of exactly
the SHA-256 length, then the hash in the last 32 bytes. Bleichenbacher's 2006 attack needs a verifier that
looks for the `DigestInfo` *anywhere* and ignores its surroundings; there is no such code path here, and
the SHA-1-style prefix encodings that a searching verifier accepts are negative tests. The 19-byte prefix
is transcribed **and then re-derived** in a test from the SHA-256 OID through `oid_from_dotted`, with each
field asserted, so a wrong byte cannot pass.
- **PKCS#1 v1.5 is for certificates only, and that is enforced by shape rather than by a comment.**
`verify_handshake_signature` takes a wire scheme and accepts exactly `0x0804` (`rsa_pss_rsae_sha256`) and
`0x0403` (`ecdsa_secp256r1_sha256`); `0x0401` (`rsa_pkcs1_sha256`) is `UnsupportedScheme`. Proven on RFC
8448's own CertificateVerify, which now verifies *through the dispatch* under the scheme byte read out of
the trace and is refused under `0x0401` with the same key and the same bytes.
- **A fixture defect, found by a control that passed when it should not have.** Verifying the
`SELF_SIGNED_PSS` fixture with the *root's* key returned `Ok(())`. Not a verification bug: the generator
built that certificate with `root_key`, so it was signed by, and carried, the root's key. Two fixtures
sharing a key means the fixture set cannot distinguish verification from a key lookup, and "a certificate
verified with the wrong identity fails" was not a check it could make. Fixed in
`scripts/tls_x509_fixtures.py` (one seed per identity, with a note saying why), the pinned data
regenerated, and the control now fails for a genuine reason. Same class as the previous slice's circular
Montgomery-domain tests: the assertion held for a reason unrelated to the property it named.
- **Which rule fires on a wrong key is deliberately not pinned.** A wrong key's RSA operation produces an
effectively random block, so the refusal can come from the block type, the padding, the `DigestInfo` or
the hash. The test asserts only that it fails at the RSA layer, and prints the actual error if it does
not — pinning the specific rule would pin an accident of that key.

177 checks green on the hardware path and on `HEARTH_TLS_NO_HW=1` (169 before), fmt and
`clippy --all-targets -D warnings` clean, `cargo doc` zero warnings, `cargo check --workspace
--all-targets` clean. `[dependencies]` still empty, and `grep` for the crate name outside itself still
finds nothing, so no `https://` path can reach any of this yet.

### hearth-tls slice 4 — the record layer, pinned by RFC 8448's own record (buffy, 2026-10-16)

`record.rs`: AEAD framing, the nonce from the sequence number, the encrypted
header as additional data, and the record-type-in-inner-plaintext construction.
**55 tests, green on both the hardware and the forced-software path**, clippy
`-D warnings` and fmt clean, still `std`-only with an empty `[dependencies]`.

**The fixture is a real published record, and it was verified before it was
trusted.** `researcher-web` was sent for RFC 8448 §3's client application traffic
secret and was told to report exact bytes and to say "not published" rather than
infer. The returned values were then re-derived here with the independent
OpenSSL-backed library before being embedded: `HKDF-Expand-Label(secret, "key", "", 16)`
and `("iv", "", 12)` reproduce the RFC's `17422dda596ed5d9acd890e3c63f5051` /
`5b78923dee08579033e523d9`, and rebuilding the record from scratch (nonce = IV XOR
sequence 0, AAD = `17 03 03 00 43`, inner = the 50-byte payload plus `0x17`)
reproduces the RFC's ciphertext and tag exactly. Only then did the crate assert
it. That one fixture pins four separate decisions at once: the nonce
construction, the header-as-AAD rule, the inner content-type byte, and the
`ciphertext || tag` layout. A second test walks the same secret through
`TrafficKeys::from_secret`, which is what validates the `tls13 key` / `tls13 iv`
labels and the empty context in `hkdf_expand_label`.

**Design decisions worth keeping.** The sequence number lives in `RecordLayer`,
is incremented exactly when a record is sealed or opened, has no setter, and
returns `SequenceExhausted` rather than wrapping — nonce reuse is prevented
structurally, since the primitive cannot detect it. A record that fails
authentication or framing returns an error and no plaintext, and does **not**
advance the sequence; a record that authenticates but carries an invalid content
type is likewise fatal. `seal_padded` supports the optional zero padding of
§5.4 and the receiver strips it without being told it was there. `key_update` and
the alert protocol are deliberately left to the handshake slice, with
`TrafficKeys::from_secret` as the seam a rekey uses.

Coverage beyond the fixture: round-trip at every content type across lengths
0/1/15/16/17/255/1000/16384; the padding path; the sequence number provably
being part of the nonce (a receiver that skips a record fails); repeated
plaintext never producing repeated ciphertext; tampering with each of the five
fields (ciphertext, tag, outer type, length, version) rejected with the specific
error and without consuming a sequence number; the length limits; and the
`0x00` content type being treated as padding rather than as data.

**Two of the failures in this slice were my own test expectations, not the code.
Both are the kind of mistake that looks like a product bug.** The `length field`
tamper moved the *high* byte of a length that is already below 256, so it changed
nothing and the record legitimately opened; and a test reused a receiver after a
fatal record, forgetting that a failed record does not advance the sequence, so
the second open failed authentication for a legitimate reason. Worth recording
because in both cases the assertion was the wrong thing, not the implementation.

### hearth-tls slice 7 — reading DER and X.509, and the four bugs my own test data had (buffy, 2026-10-16)

**The reading half of certificate verification.** `crates/hearth-tls/src/der.rs` is a strict DER
(X.690) reader and `x509.rs` reads a certificate over it. 152 checks (121 before, plus 15 DER
unit tests, 7 in `x509.rs` and 9 fixture tests), clippy
`-D warnings` and `cargo fmt --check` clean, `cargo doc` clean, workspace check clean, `std`-only,
empty `[dependencies]`, 152 on the hardware path *and* on `HEARTH_TLS_NO_HW=1`. Still not reachable
from any URL: `grep` for the crate name outside itself finds nothing, so the seam stays closed.

**DER is strict by construction, not by option.** Definite lengths only; the short form must be
minimal and long form must not have a leading zero byte or be used below 128; no high-tag-number
form; no trailing bytes after the outer element; a truncated length or body is an error rather than
an empty read. The reader's `take`/`take_if` leave it *untouched* on a tag mismatch, which `take_if`'s
contract already promised — a first version consumed first and checked after, and its own test caught
it. Every one of those rules has a hand-built malformed encoding behind it, plus the DER of the real
RFC 8448 certificate as the positive case.

**The fixtures are generated, not transcribed.** `scripts/tls_x509_fixtures.py` builds six
certificates with OpenSSL via Python's `cryptography` — an RSA root, an EC root, an intermediate, an
RSA leaf (SANs, EKU serverAuth), an EC leaf, and a PSS-signed self-signed EC cert — with keys derived
from a fixed seed, and with the expectations (including the key-usage *names* as OpenSSL reports
them) parsed back out of the generated certs rather than written by hand. Six rather than five
because the EC leaf is signed by an *RSA* issuer, so a PSS-signed self-signed EC cert was needed to
cover that `AlgorithmIdentifier` shape. Two rules came
out of RFC 5280 and have their own fixture: a critical extension the parser does not understand must
be **rejected** (§4.2), and the `signatureAlgorithm` inside the signed part must be byte-identical to
the one outside it (§4.1.1.2).

**The first version of this paragraph said the fixtures "regenerate byte-identically", and that was
wrong — the mistake was in a claim, not in code, and only regenerating and diffing caught it.**
Regenerating to a scratch path produced four hunks: the RSA-PSS fixture's signature (a random salt),
an ECDSA signature whose *length* also changed, and the `der_len` that follows from it. So two of the
six fixtures are not byte-reproducible; four are. The generator's own docstring had the same claim
in a subtler form ("the RSA certificates regenerate byte for byte" — the PSS certificate *is* an RSA
certificate). Both now say which two, and why. The property that actually matters was then verified
rather than assumed: the regenerated file was swapped in and the fixture tests were run against it,
**9/9 green including the changed `der_len`**, which is what proves the expectations are parsed out
of the certificates instead of typed. The file in the tree is therefore an *equivalent* pin, not a
byte-exact one, and it is now commented as such.

**Four of the failures in this slice were wrong test data, not wrong code, and three of them were
copied from memory.** (1) I wrote the expected `UnixTime` constants for ten calendar rows out of my
head and **every single one was wrong**; recomputing them from CPython's `datetime` *and* from the
`days_from_civil` algorithm independently gave two implementations that agreed with each other and
with the Rust code. A memorized constant is not a vector. (2) `keyUsage` bit order was genuinely
wrong — it is a `BIT STRING` counting from the most-significant bit of the first byte, not from bit 0.
(3) A test asserted the TBS started at offset zero; it starts after the certificate's outer header.
(4) A test assumed the signature length follows the *subject's* key — it follows the **issuer's**,
which is exactly why the EC leaf carries a 256-byte RSA signature.

**`x509` refuses anywhere two readers could disagree.** `Certificate` borrows its input, so nothing
is copied out of the DER; both names are kept as their own DER for byte-exact chain linkage rather
than as parsed strings; the signature covers `tbs_certificate` and nothing else (asserted); SPKI,
`basicConstraints`, `keyUsage`, `extendedKeyUsage` and `subjectAltName` are read, with non-DNS SAN
entries kept but only DNS ones usable as names. Not yet present, and the next session's work: chain
building to a pinned root, ECDSA verification, name matching, the validity check and a minimum key
size. Until those exist `RejectAll` remains the only verifier in the crate.

### hearth-tls slice 8 — P-256 and ECDSA, and the bug two of my own tests hid (buffy, 2026-10-16)

**Both halves of signature verification now exist.** `p256.rs` is the curve — the field, the group
and the two scalar multiplications ECDSA needs — and `ecdsa.rs` is `ecdsa_secp256r1_sha256` over it.
**169 checks green on the hardware path and on `HEARTH_TLS_NO_HW=1`** (152 before), clippy
`--all-targets -D warnings`, `cargo fmt --check`, `cargo doc` (zero warnings) and a workspace check
all clean, `std`-only, empty `[dependencies]`. Still unreachable from any URL: grep for the crate name
outside itself finds nothing, and `RejectAll` is still the only verifier in the crate.

**The constants are read out of OpenSSL, not remembered.** `scripts/tls_p256_params.py` runs
`openssl ecparam -name prime256v1 -param_enc explicit -text` and then *checks the group* before
printing Rust: `p` and `n` prime (sympy, which shares no code with OpenSSL), `a == p - 3`, the
generator on the curve, `n * G` the identity via a plain-Python affine multiply, `n` inside the Hasse
interval around `p + 1` with cofactor 1, and `p < 2n` with `n > 2^255` — the last two are what make a
single conditional subtraction a valid reduction. The Rust asserts the first four again by itself, and
the two Montgomery constants (`-m^-1 mod 2^64`, `2^512 mod m`) are **derived in `const fn`** from the
modulus, with the script printing its own values only so the test can check that derivation. Nothing in
`p256.rs` is a literal that was not printed by a program that first proved it.

**Four of five test expectations I wrote were wrong; the fifth hid a real bug.** The one that matters:
`Fe::ONE` was `[1, 0, 0, 0]` — the limbs one, in a type where *every* value is in Montgomery form —
and `Fe::invert` returned a plain value into that same Montgomery world. The two mistakes cancelled:
`R / R` is one, so `inverse * value == ONE` passed, and the coefficient-of-one doubling formula quietly
stopped producing a point on the curve. What exposed it was arithmetic rather than review — 7G came
back off the curve and `n * G` was not the identity — and the reason it survived a first run is worth
recording: my "independent" checks were **circular**. `assert_eq!(4G_from_the_ladder, 4G_from_the_group_law)`
compared `g.double().double()` with a ladder that computes exactly the same thing, so a doubling formula
correct only when `Z == 1` passed every comparison. The fix is a `mont_pow` primitive that makes the
domain explicit, `ONE` written as a conversion rather than a literal, and **pinned 2G/3G/4G/6G/7G x
coordinates taken from the independent Python implementation**, one of them reached by doubling a point
whose `Z` is not 1. The inverse test now asserts the domain directly (`ONE` reads back as the bytes `1`,
and a value survives the round trip through its Montgomery form) because no test that multiplies an
inverse back can see the difference.

**The ECDSA vectors carry a chain of evidence, not a snapshot.** `scripts/tls_p256_vectors.py` derives a
private scalar from a label, signs six messages with OpenSSL, and then requires that each vector passes
an **independent verifier written in plain Python integers over affine coordinates** *and* OpenSSL's own
verify, before writing any of it out. Six shapes of signature integer appear (both 32- and 33-byte
encodings of `r` and `s`), and one of the messages is exactly the RFC 8446 §4.4.3 construction — 64
spaces, `"TLS 1.3, server CertificateVerify"`, a zero byte, a 32-byte transcript hash — so the shape
the real handshake signs is covered. The end-to-end case is the `SELF_SIGNED_EC` X.509 fixture: DER
framing, the certificate fields, the P-256 point parser and the equation, over an OpenSSL certificate,
with a second half that flips one byte of the serial number and requires the same signature to fail —
which is what shows the signature covers the bytes that were *read* rather than a re-encoding.

**Two kinds of negative vector, checked differently.** A *cryptographic* negative (a flipped bit of `r`
or `s`, `r = 0`, `s = n`, `s = n + 1`) must come back as `VerificationFailed` or `ScalarOutOfRange`, and
the test asserts which. An *encoding* negative (a redundant leading zero, a negative `INTEGER`, a
trailing byte, a third integer, `SET` instead of `SEQUENCE`, a truncated sequence, the raw `r || s`
form) must be refused **as an encoding**, and each is pinned to the rule that has to fire rather than to
a boolean — a boolean cannot tell "refused this signature" from "refused this byte string". The
redundant leading zero is what makes that concrete: it is legal BER naming the same integer, so a
lenient reader reads a signature DER does not have. One claim I did *not* get to keep: I had written
that OpenSSL's decoder is lenient about some of these, and the generator records which negatives a
lenient reader accepts — in this run it refused every one of them, so the honest statement is that the
pinning is about which rule fires, not about a disagreement in outcomes.

**High-S is accepted, deliberately.** `(r, n - s)` is the other valid signature over the same message.
Requiring the smaller one is a Bitcoin consensus rule, not an X9.62 or TLS one, so a conforming server
may produce it and this verifier must not invent a reason to refuse it. That is pinned as a test so the
decision is stated rather than assumed.

**Cost, measured rather than guessed: 345.6 µs per verification** (release, `--ignored` timing test on
the TLS-shaped vector). That is slow for P-256 — OpenSSL is several times quicker — and the reasons are
known and recorded rather than mysterious: no windowing or precomputed table for the fixed base, no
signed-digit recoding, a coarse "multiply then reduce" Montgomery product, and a Fermat inversion
(~384 multiplications) instead of a binary extended Euclidean one. For one handshake per download that
is not worth optimising yet; it is in `RESUME.md` with the plan if it ever is.

### hearth-tls slice 6 — RSA-PSS verification, pinned to RFC 8448's own signature (buffy, 2026-10-16)

**The first half of certificate verification, and the half that can be *proven* today.**
`crates/hearth-tls/src/rsa.rs` adds `RsaPublicKey` (modulus + exponent), `rsa_vp1` (RFC 8017 §5.2.2)
and `verify_pss_sha256` (RSASSA-PSS-VERIFY, §9.1.2) with MGF1-SHA-256 and a 32-byte salt. 121 checks
(the suite was 106), clippy `-D warnings` and `cargo fmt --check` clean, `std`-only, still no
dependencies. Nothing calls it yet — the public key has to come out of a certificate first — so the
only verifier the crate ships is still `RejectAll` and no `https://` path exists.

**The vector is the RFC's own recorded signature, not a value we produced.** RFC 8448's trace
contains a complete `rsa_pss_rsae_sha256` signature, made by OpenSSL, over the 130 bytes the client
state machine already rebuilds. `tests/rfc8448_flight.rs` now decrypts the recorded flight with the
crate's own record layer, re-derives the transcript through `Certificate`, and verifies that signature
with the server's modulus read out of the trace's 432-byte certificate. Three independent values pin
it: `Hash(ClientHello..Certificate)` = `764d6632…` (derived here from the trace *and* computed by the
crate), `SHA-256(signed_data)` = `beed95af…`, and the 128-byte modulus, which the test requires to
appear verbatim inside the certificate so the two constants cannot drift apart. Flip any byte of the
signed data — including inside the trailing hash — and it stops verifying, which is the whole point:
the signature covers that exact transcript or nothing.

**Second vector set: both `emLen` cases, and the negative ones.** `tests/rsa_vectors.rs` carries 24
constants generated by *other implementations*: `s^e mod n` from CPython's `pow` at 4, 8, 9, 16, 24
and 32 limbs (exponent 3 as well as 65537), real PSS signatures from OpenSSL, and two primes from
`sympy.randprime` for a key OpenSSL's generator refuses to make. That last one is a **1025-bit key**:
1025 ≡ 1 (mod 8), so `k = 129` while `emLen = 128`, and a 2048-bit key would never notice the
difference. A signature that is a real signature but with a 20-byte salt is refused, because RFC 8446
§4.2.3 says the salt length MUST equal the digest length — OpenSSL will happily produce such a
signature, so it is a live case rather than a hypothetical. Also refused: every single-bit change to a
signature, a different message, a signature one byte short or long, `s == n` and `s == n + 1`, all
zeros, and a 512-bit key that cannot hold the encoding at all (`KeyTooSmall { bits: 512, needed_bits:
522 }`, checked before any arithmetic, so the bounds cannot underflow).

**Two mistakes were mine, and the tests caught both.** `limbs_from_be` packed the input byte string
in the wrong order — most-significant chunk into the least-significant limb — which reverses the limbs
of any value wider than one limb and doubles its apparent width; a 65-bit modulus reported as 128 bits.
Taking the chunks **from the end**, zero-padding each on its high side, is the fix, and a test now pins
the endianness at one, two and 17 limbs. Separately, a generator loop that waited for OpenSSL to
produce a 2049-bit modulus never terminated: `generate_private_key` always sets the top bit, so it
never returns one. Every retry loop in the generator is now bounded. Both are the same lesson as the
transcription bug in slice 5, one layer up: a wrong expectation and an unbounded loop look like
"waiting" rather than "broken".

**The generators are committed, so the provenance is checkable.** `scripts/tls_rfc8448_rsa.py` derives
the modulus and both hash constants from the RFC's hex with OpenSSL and prints them; its output was
diffed against the constants in `tests/rfc8448_flight.rs` and is identical character for character.
`scripts/tls_rsa_vectors.py` regenerates the vector table: run against the committed file it
reproduces **all 18 modpow constants byte for byte** and emits new PSS ones, which is the documented
behaviour rather than a failure — OpenSSL seeds itself from the OS, so only the CPython-derived half
can be reproduced, and that half now demonstrably is. Both live in `scripts/` with the existing Python
helpers instead of in `target/`, where the earlier slices' generators were written and lost.

**A measurement, since a number in a document should have been measured.** The public-key operation on
a 2048-bit modulus takes **94.5 µs** (`cargo test --release -p hearth-tls --lib -- --ignored
--nocapture measure`), which is 17 Montgomery multiplications plus the two conversions: schoolbook
multiply into a double-width buffer, then reduce from the bottom up. That is ~4x OpenSSL and entirely
irrelevant here — a downloader handshake verifies one signature per connection — so the coarse-grained
form was kept over the interleaved CIOS one because it is easier to check by reading. There is no
division routine in the module at all: `R^2` is reached by 2·64·limbs modular doublings.

**Not the end of the job.** Still missing before any URL can use this: DER parsing, chain building to
a pinned root, `ecdsa_secp256r1_sha256` (the other scheme this client offers), name matching, the
validity window check, and a **minimum key size**, which is deliberately *not* in this module — it
accepts a 1024-bit key because RFC 8448's fixture is one, and the policy belongs where the key comes
from.

### hearth-tls slice 5 — the handshake: key schedule, messages, and the client state machine (buffy, 2026-10-16)

**101 tests, green on both the hardware and the forced-software path**, clippy
`--all-targets -D warnings` and `cargo fmt --check` clean, still `std`-only with an empty
`[dependencies]`. Four new modules plus one integration test file:

- `util.rs` — one constant-time comparison, shared by the GCM tag check and the Finished MAC
  check. It lives here rather than beside its first user because a second copy is how a
  security-relevant comparison drifts: someone eventually "optimises" theirs with `==`.
- `keyschedule.rs` — early/handshake/master secrets, `c hs traffic`/`s hs traffic`,
  `c ap traffic`/`s ap traffic`, `finished_key`, the Finished `verify_data`, `key_update`'s
  `traffic upd`, and both application secrets. All pinned to RFC 8448's printed values.
- `handshake.rs` — message framing (4-byte header, not 5), the ClientHello we send, a strict
  ServerHello parser, and the running transcript hash. Nothing panics on hostile input:
  every truncation and every single-bit mutation is asserted to be an error, not a panic.
- `client.rs` — the 1-RTT state machine: ServerHello → the encrypted
  `EncryptedExtensions`/`Certificate`/`CertificateVerify`/`Finished` flight → our own Finished →
  application keys, plus `KeyUpdate` in both directions, `close_notify`, and the alert path. It
  does no I/O (bytes in, bytes out) so a caller owns the transport and the CSPRNG.
- `tests/rfc8448_flight.rs` — the published flight as an external user of the API.

**The certificate verifier cannot be skipped, by construction.** `ClientHandshake::new` requires a
`CertificateVerifier`, and the crate ships exactly one implementation: `RejectAll`, which can only
refuse. There is no insecure flag, no environment variable and no accept-all type — a caller that
wants a usable connection has to write and own the verification step. `HEARTH_TLS_NO_HW` (slice 3)
can only ever select the *slower* implementation; it cannot weaken a check.

**The strongest fixture is RFC 8448's real encrypted server flight, and it is self-proving.** The
679-octet record contains all four messages. Three published values pin each other: the transcript
hash through `ClientHello..ServerHello` fixes the hello bytes; the AEAD tag fixes every ciphertext
byte; and the server's published Finished MAC
(`HMAC(finished_key(s hs traffic), Hash(ClientHello..CertificateVerify))`) fixes `EncryptedExtensions`,
`Certificate` and `CertificateVerify` message by message. The test also re-seals the *client's*
Finished record and requires it to equal the trace's 58 bytes, which checks the sending direction
against a real implementation's output rather than against ourselves.

**A transcription error was caught by derivation, and 29-of-32 bytes matching is what caught it.**
`researcher-web` reported the client's Finished `verify_data` as `…c1 fc eb 31 92 5f 9e…`. Deriving it
from the published `c hs traffic` and transcript hash gave `…c1 fc eb e1 1a 03 9e…` — the same value
in 29 of 32 bytes. A hash cannot differ in three bytes, so one of the two was a copy error: the RFC's
own text prints `eb e1 1a 03`, so the transcription was wrong and the derivation was right. The
generator re-derives the value before emitting it, so the constant in the repository is the RFC's.
The general lesson stands: **a published vector is only as good as the copying**, so every value in
this slice is either re-derived locally or asserted against a second published value that covers it.

**RFC 8448's `Certificate` message is internally inconsistent, and the fixture records that rather
than smoothing it over.** It declares a 437-byte `certificate_list` whose single entry is 432 bytes,
accounting for 435; the two unaccounted bytes are `00 00` at the end. Both the AEAD tag and the
published Finished MAC reproduce over exactly those bytes, and the RFC's own text shows the same
lengths, so the anomaly is the RFC's. It is asserted by its own test so nobody later "fixes" the
fixture. `parse_certificate` in the client is strict about this framing — real servers emit a
consistent list — which is why the fixture drives that trace through the record and transcript layer
instead of through the strict parser.

**Three failures in this slice were again my own test expectations.** The ClientHello slice passed to
`Server::accept` still carries its 4-byte handshake header, so my session-id offsets were four bytes
late (every test that needed keys failed, which is how it surfaced); one negative case I wrote as
"empty chain" (`00 00 00`, a zero-length entry) is malformed framing rather than an empty list; and a
`KeyUpdate` test left the server's *receive* generation behind, so the next record failed to
authenticate — which is exactly the signal that a rekey is real rather than bookkeeping, and that
assertion is now part of the test. Also worth recording: `cargo fmt` had never been run across the
whole crate in the earlier slices, so `cargo fmt --check` reported pre-existing diffs in
`keyschedule.rs`, `handshake.rs` and `util.rs`; those are now formatted.

**What this does not do yet.** No certificate verification (so the crate must not be wired to an
`https://` path), no client authentication (a `CertificateRequest` is refused, not ignored), no
resumption/PSK or 0-RTT, no ALPN, no post-handshake authentication, and `aarch64` PMULL/AES is still
missing for macOS. The verifier interface, however, is now settled: chain, signed data, signature
scheme, the name we asked for and the instant all arrive as arguments, so the certificate slice can be
written as a pure function and tested against expired/not-yet-valid fixtures that never go stale.
`NewSessionTicket` and similar post-handshake messages are refused explicitly rather than swallowed.

**Peer review caught a real hole in the "no accept-all" claim. It is fixed, and now enforced
(SparkOC, same session).** The trait carried a blanket `impl CertificateVerifier for F where F:
Fn(..)`, which left `ClientHandshake::new(config, |_| Ok(()))` available as a one-line accept-all at
any future call site — the exact shorthand the module docs said did not exist. Worse, my own tests
used it (`accept_anything`, `verifier_accepting`), so the tests were themselves the pattern a later
contributor would copy. The blanket impl is gone: every verifier is a named type (`AcceptTestChain`
and `ExpectingChain` in the tests, `RejectAll` in the crate). SparkOC's related note is fixed too:
`CertificateVerifyInput` now carries `server_name`, the one input a verifier cannot derive from the
messages it is handed, so a verifier no longer has to capture the expected host name — and a captured
copy cannot silently drift from the name on the wire.

**The second half of that review was right too, and it is the same defect one field over (SparkOC,
same session): the validity time was missing.** `certificate_list`, `signature_scheme`,
`signature`, `signed_data` and `server_name` covered everything except *when* — expiry and
not-before checks need a `now`, and a verifier that calls `SystemTime::now()` itself cannot be
handed an expired or not-yet-valid chain in a test without those tests expiring along with the
fixtures they use. This crate's one rule for the verifier interface is that the handshake supplies
every input the verifier cannot derive from the messages, and the time is not in any message, so it
belongs in the input rather than in the verifier's captured state. Fixed by adding
`ClientConfig::validation_time` and `CertificateVerifyInput::validation_time`, both a new
**`UnixTime`** (whole seconds since the epoch, `i64` so a subtraction against a validity bound cannot
wrap, with `from_system_time` saturating rather than panicking on a clock set absurdly far out and a
one-line `outside(not_before, not_after)` so the comparison the certificate slice turns on is written
once). Roots and policy stay construction-captured on the named verifier type, as before; only the
instant moves. The state machine still touches no clock — the caller reads it (`SystemTime::now()
.into()`, or anything else) and the value flows through untouched, which is what makes the future
negative suite deterministic.

**That new property is checked by mutation, not by assertion.**
`the_verifier_is_given_the_callers_instant_and_not_the_clock` runs a complete honest handshake at four
instants that no clock would ever report — one second *before* 1970, 1970 itself, 2023, and 2100 — and
requires the verifier to be shown each verbatim. It was then verified to be load-bearing rather than
tautological: replacing the call site with `UnixTime::from_system_time(SystemTime::now())` makes it
fail with `left: [UnixTime { secs: 1792130736 }] / right: [UnixTime { secs: -1 }]`, and the mutant was
reverted. Without that mutation check, a test that hands in a fixed time and reads it back could pass
for the wrong reason. 106 checks green on both paths (100 unit + 5 integration + 1 doctest).

`cargo doc -p hearth-tls` also surfaced two pre-existing rustdoc warnings in slices 1-2, both of them
a doc-list link resolving to the wrong kind of item: `[`x25519`]` is both a module and a function, so
the module list needed `mod@x25519`, and `[`verify_finished`](crate::keyschedule::verify_finished)`
carried an explicit target that the label already resolves to. Both fixed; the crate now documents with zero warnings.

**The property is now checkable rather than asserted in prose, which is the part worth keeping.** A
`compile_fail` doctest in the module docs requires the closure form to be rejected, so CI enforces it.
It was verified to fail for the *right* reason rather than for a typo: with the closure line disabled
the same snippet compiles, and with it enabled the error is exactly `E0277: the trait bound
`{closure}: CertificateVerifier` is not satisfied`. Both halves of that check were run before the
doctest was left in place. 102 checks green on both the hardware and the forced-software path
(96 unit + 5 integration + 1 doctest) at that point; the clock input above brought it to 106.

### hearth-tls slice 3 — AES-NI and PCLMULQDQ, runtime-dispatched (buffy, 2026-10-16)

Measured on one binary, interleaved, with `HEARTH_TLS_NO_HW=1` forcing the software path
(the switch can only ever select the *slower* implementation — it cannot weaken
verification, which is why it was allowed to exist at all):

| path | AES-128 block | AES-128-GCM seal | GCM open | X25519 |
|------|---------------|------------------|----------|--------|
| hardware (AES-NI + PCLMULQDQ) | **1304 MB/s** | **522-547 MB/s** | 556 MB/s | 14.1 k ops/s |
| software | 68.8 MB/s | 26.4 MB/s | 26.5 MB/s | 14.1 k ops/s |

Run 1 and run 3 (both hardware, software run between them) agreed within 0.5%, so this is
not this box's usual 10-15% drift. **20x on GCM**, and a 4 GiB model drops from 159 s of
CPU to 8 s. X25519 is unchanged because it is already field arithmetic with no table.

**The reduction constant is derived, not transcribed.** The usual implementation
byte-swaps and applies `0xc200000000000000`, a constant that works and that nobody can
re-derive from the code. Instead: GCM reads a block with the leftmost bit as `x^0` and
`x^128 ≡ x^7+x^2+x+1`, PCLMULQDQ reads the rightmost bit as `x^0`, and reversing the 128
bits maps one onto the other — reversal commutes with multiplication, so reverse,
multiply, reverse. In the reversed domain `·x^k` is a left shift, so the reduction is
`p_lo ^ (p_hi · 0x87)` — shifts and XORs, no constant to trust. Two folds finish it, since
`p_hi` has degree ≤ 126 and one fold leaves degree ≤ 133. The reasoning is written out in
`ghash_x86.rs` and pinned by a 3800-pair differential test against the software multiply
(including all-ones, single-high-bit and both-folds-needed shapes).

**The acceleration is verified end-to-end, not only differentially:** the entire suite —
FIPS-197, the NIST GCM cases, RFC 7748 including the 1000-iteration chain — passes **both**
with hardware enabled and with `HEARTH_TLS_NO_HW=1`, 43 tests either way. That means the
published vectors themselves run through the SIMD path.

**One bug was mine, in a test, and the lesson is worth keeping.**
`multiply_by_087_matches_a_shift_free_definition` failed against a correct implementation
because my reference reconstruction accumulated the four shifted copies with bitwise **OR**,
while polynomial addition in GF(2) is **XOR** — and the copies genuinely collide. The
differential test had already passed, which is what told me to suspect the test rather than
the code. A plausible-looking manual check can fail a correct implementation; prefer the
comparison against an independent implementation.

Remaining on this front: Apple silicon's crypto extensions (`aarch64` PMULL/AES) so macOS
gets the same treatment, and multi-block pipelining plus aggregated GHASH powers if the
record layer ever needs more than ~520 MB/s.

### hearth-tls slice 2 — AES-128-GCM and X25519 (buffy, 2026-10-16)

`crates/hearth-tls` gained `aes.rs` (AES-128, forward only), `gcm.rs` (AES-128-GCM)
and `x25519.rs` (X25519). **40 tests, all green**, clippy `--all-targets -D warnings`
and fmt clean, still `std` only with an empty `[dependencies]`.

**Method: an oracle, not my memory.** `cryptography` 46.0.6 (OpenSSL 3.5.5) is on
the box, so every constant was re-derived from an independent implementation before
being pinned in a test, and the oracle scripts assert the published spec values too —
so spec and oracle are checked against each other, not merely against me. Scripts:
`target/buffy-ab/gcm_vectors.py`, `target/buffy-ab/x25519_vectors.py`. Two memorised
AES constants were confirmed this way (AES-128 zero vector, FIPS-197 C.1) and one
key-schedule expectation I was not sure of was **deleted rather than guessed**, since
the C.1 cipher vector validates the whole schedule anyway. The AES S-box is derived
at compile time from the GF(2^8) inverse plus the affine transform instead of being
transcribed — 256 chances for a one-character error, replaced by a derivation that
`derived_sbox_matches_known_entries` pins.

Coverage: FIPS-197 C.1 + the zero block; NIST/GCM cases 1–4; odd 1-byte and 15/17-byte
inputs where padding bugs hide; a 1000-byte message; all 128 single-bit tag flips
rejected; tampered ciphertext, AAD, nonce and truncated ciphertext rejected; tag
changes under every one of the 128 key-bit flips. RFC 7748 §5.2 vectors 1–2, the §6.1
DH example, the **iterated 1- and 1000-iteration** chain, three random oracle vectors,
clamping (low 3 bits and bit 255 ignored, bit 254 forced) and the high bit of `u`
ignored.

**Fail-closed, with no way to opt out.** `diffie_hellman` returns `Err` on an all-zero
result rather than handing back a peer-chosen secret; OpenSSL refuses those same points,
so the behaviour is not merely our preference. `x25519` (raw) exists only because the
RFC vectors are defined in terms of it and is documented as not-for-protocol-code.
GCM verifies before it decrypts and has no API that returns unauthenticated plaintext.

**Measured software throughput** (`cargo run --release -p hearth-tls --example crypto_bench`):

| primitive | result |
|-----------|--------|
| AES-128 block | 63.6 MB/s (240 ns/block) |
| AES-128-GCM seal | 24.2 MB/s (1 MiB and 16 KiB alike) |
| AES-128-GCM open | 24.3 MB/s |
| X25519 | 78 us/op (12.9 k ops/s) |**This is the finding that set the next slice.** GHASH dominated: solving
`1/24.2 = 1/63.6 + 1/g` gives g ≈ 39 MB/s for GHASH alone, so AES-NI alone would only
have lifted GCM to ~39 MB/s — PCLMULQDQ GHASH was the half that mattered. X25519 at
78 us was a non-issue either way.

**Corrected framing, and worth reading because I first got it wrong.** The 169 s for a
4 GiB file is *CPU time*, not added latency. A 100 Mbps link delivers ~12 MB/s and the
software path sustains 26 MB/s, so it stays ahead of the download and the cost is a busy
core, not a slower transfer. It only becomes the bottleneck below the link rate — i.e. on
a gigabit link or a local mirror. "56% on top of a 5-minute transfer" in the first draft
conflated the two and was an overstatement; the corrected numbers are in slice 3 below.

### hearth-tls slice 1 — SHA-256, HMAC-SHA256 and the RFC 8446 key schedule (buffy, 2026-10-15)

New workspace crate **`crates/hearth-tls`**: **std only, `[dependencies]` empty** (operator rule this
session: no new dependencies, so no `rustls`). 15 tests, every one against a published vector — FIPS
180-4 (empty, `abc`, the two-block 448-bit message, the 1 M `a` case), RFC 4231 HMAC 1–7 including
the 131-byte key, RFC 5869 HKDF 1–3, plus `hkdf_expand_label`/`derive_secret` round-trips. clippy
`--all-targets -D warnings` and fmt clean, workspace check green. A **lib** crate on purpose: an
unwired public API in a module would warn as dead code, and nothing consumes it yet. It is the bottom
of the TLS 1.3 stack for the downloader's `https://` path; verification is to be fatal and never
downgradeable.

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
