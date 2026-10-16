# Hearth perf research — decode/prefill levers + portable future (2026-10-15)

> Status: research only, no code. Perf code ownership stays with buffy
> (Task 1); nothing here is a commitment, just sized options with prior art.
> Cross-cutting rule from this session: anything adopted must build on
> Linux, Mac and Windows and must not crash low-RAM laptops.
>
> 2026-10-15 update: the prefill/decode asymmetry premise is RETIRED
> (buffy measured: 27B prefill 635 vs decode 722 ms/tok — prefill is fine;
> the old 2x came from a loaded-machine run). First-token excess is warm
> page-cache I/O, inherent. Pin hazard: HEARTH_PIN_WEIGHTS=1 on the 27B
> died silently on 16 GB — never default pinning for large models.

## 1. Where the time goes (roofline framing)

- **Decode (1 tok/step)** is memory-bandwidth bound: each step re-streams
  every weight row once. Kernel speed ≈ bytes moved, not FLOPs. This is why
  the Q2_0 VNNI win and the Zen4 dispatch fix moved tok/s directly.
- **Prefill (prompt processing)** can reuse weights across tokens (batch),
  so it is the only place extra arithmetic buys bandwidth savings (x2
  kernels: Q2_0 prefill −31%).
- The 27B hybrid anomaly (prompt tok slower than decode tok despite
  identical per-token traffic) is under investigation (buffy, Task 1) —
  resolve that before any bigger prefill architecture.

## 2. Decode levers (biggest first)

1. **Speculative decoding (draft + verify).** A small same-family draft
   (Ternary-Bonsai 2B) proposes K tokens; the big model verifies them in
   ONE batched forward pass. Same vocab => no re-tokenization tricks.
   Prior art: llama.cpp speculative mode, LM Studio (1.5–3x reported),
   all on memory-bandwidth-bound rigs exactly like this one. CPU fit is
   GOOD: verification is a batch prefill, our fastest path. Effort:
   medium (draft loop + verify-accept logic + KV rollback). Risk: low —
   output is bit-identical to the target model by construction.
   Suggested pair: 2B-PTQ1_0 drafts for 27B-PTQ1_0.
2. **Keep dispatch honest.** The Zen4 bug class (a predicate that can never
   fire) needs the regression-guard pattern buffy added: synthetic CPUID
   vectors through family reconstruction + a `kernels` report in load
   output. Extend the same treatment to any new kernel (NEON, I8MM).
3. **Batch > 1 is N/A** for a single-user UI; skip.

## 3. Prefill levers

1. **Asymmetry: CLOSED, negative result** (buffy, 2026-10-15). Do not
   sequence around it. Remaining lever is below.
2. **Chunked hybrid prefill** is the real 27B win. The GDN recurrence is
   sequential per token, but the DENSE projections (the bandwidth hogs)
   can carry 2+ tokens per weight pass while the recurrence steps
   sequentially — needs the GDN chunk form written down first. Prior art:
   Mamba-2 State Space Duality (parallel scan / chunkwise form) and
   Mamba selective-scan chunking; our x2 kernel primitive
   (`dot_ptq1_0_q8_0x2_ptr`) is exactly the (one weight row, two Q8 rows)
   shape chunking needs. Multi-session, needs a design note, touches the
   hybrid forward path — do NOT start blind.
3. Dense-path x2 coverage is otherwise complete (Q2_0/PQ2_0 share,
   Q1_0_G128 has one, PTQ1_0 primitive exists but unreachable from hybrids).

## 4. Future-proofing other systems

- **ARM (Mac + ARM laptops): correctness already portable** (scalar
  fallbacks everywhere); speed is x86-only today. Path: NEON VDOT kernels
  mirroring the AVX2 LUT design, then I8MM where present (prior art: Arm's
  llama.cpp I8MM blog, Forge-LLM on Apple AMX). Gate by arch + runtime
  detection, same dispatch-report pattern. Note: on Apple Silicon the
  bigger lever is the GPU (unified memory, Metal backend in llama.cpp) —
  CPU NEON is second.
- **GPU — ROCm vs portable.** Reality check (Oct 2026): ROCm has no
  official Windows support and the 780M (gfx1103) is unofficial even on
  Linux; a hand-rolled ROCm backend would help Linux AMD-dGPU users only —
  NOT this laptop, NOT Mac. A portable compute backend (Vulkan/WebGPU
  kernels: matmul, RMS norm, RoPE, flash attention) covers the 780M on
  Windows+Linux, NVIDIA/Intel iGPUs, and Mac via MoltenVK with ONE
  codebase. Recommendation: keep `hearth-compute` API backend-agnostic,
  prototype portable GPU before any ROCm-specific work; ROCm hand-roll
  only if Linux-AMD-dGPU users materialize. (Portable GPU likely wants a
  crate like wgpu — operator call, same class of decision as rustls.)
- **Keep laptop-safe defaults:** GPU paths must be opt-in (`--device`),
  CPU path always works; no hard AVX-512/NEON requirement; per-size
  gating per the model-size-aware policy in AGENTS.md.

## 5. Suggested order

1. ~~Prefill/decode asymmetry~~ closed (negative result, premise retired).
2. Speculative decoding design + prototype (2B→27B).
3. NEON kernels for ARM correctness-with-speed (behind dispatch report).
4. Chunked hybrid prefill design note, then implementation.
5. Portable GPU prototype evaluation (needs the dep decision first).
6. Hand-rolled TLS completion (separate track, already decided).
