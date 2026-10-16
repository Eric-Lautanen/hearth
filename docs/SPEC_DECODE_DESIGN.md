# Speculative decoding for Hearth — design note (2026-10-15)

> Status: design only, no code. Builds on buffy's review corrections
> (acceptance rule owns exactness; GDN rollback is the hidden cost).
> Target pair: Ternary-Bonsai-2-2B-PTQ1_0 drafts for
> Ternary-Bonsai-2-27B-PTQ1_0. Vocabularies verified identical.

## Why it fits this engine

Decode is memory-bandwidth bound: every step re-streams all weights.
Verification of K draft tokens is one batched prefill — our fastest path
(x2 kernels where they apply). Draft cost is ~7% of a target token
(2B PTQ1_0 50.8 ms/tok vs 27B ~722 ms/tok, measured). Expected win at
acceptance a=0.7, K=4 is ~2.2x (27B ~1.4 → ~2.5–3.6 tok/s). Not 10x;
worth doing anyway since the 27B is the UX bottleneck.

## Acceptance rules (exactness lives here, not in the draft)

- **Greedy (temp 0, our benchmark path): genuinely exact.** Accept draft
  token iff it equals the target's own argmax at that position.
- **Sampling: needs the standard machinery.** Rejection test plus residual
  correction applied to the POST-truncation distribution (Hearth truncates
  via top-k/top-p/min-p/typical before sampling — the correction must see
  the same distribution the sampler sees). The stateful repeat penalty
  must advance identically on accept and on correction; test, don't assume.
- **Measure acceptance first, no new code:** score 2B proposals against
  27B logits on a fixed prompt set. The pair shares vocab but not weight
  basis (hadamard v2/block 512 vs v1/1024), so the rate is empirical.

## GDN state rollback (the hidden cost)

3 of every 4 qwen35 layers are recurrent: a rejected draft must rewind
conv + SSM state, not just a KV cache. Two options: snapshot state per
proposed token (memory cost × K) or re-run the accepted prefix through
the draft path (time cost, cheap at 7% of a target token). Prefer
recompute; snapshot only if measurement says otherwise. Prompt rendering
uses the TARGET template for both models (their chat templates differ);
feed identical ids.

## UI implications

Verification steps should surface in the existing live progress line
(`streaming · N tok · m:ss · X tok/s`), e.g. accepted/rejected counts,
so a 27B reply stays visibly alive. No new endpoints expected: draft
loop lives in the engine/service, streams over the current SSE channels.

## Suggested build order

1. Acceptance-rate measurement harness (fixed prompts, 2B vs 27B logits).
2. Greedy path only (temp 0): draft loop + verify + KV/GDN handling.
3. Sampling variant with rejection + residual tests.
4. UI counters in the progress line.
