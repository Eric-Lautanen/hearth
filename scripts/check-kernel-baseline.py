#!/usr/bin/env python3
"""Guard the baseline-invariance of the runtime-dispatched SIMD kernels.

A kernel body declares the features it needs with `#[target_feature(enable = ...)]`.
`avx2` does NOT imply `fma` or `f16c`, so an AVX2 kernel that uses `_mm256_fmadd_ps`
or converts an f16 scale must declare `fma`/`f16c` too. When the crate baseline
already has them (x86-64-v3 and up) the omission is invisible; when it does not,
the intrinsic stops inlining and the kernel runs as per-instruction calls. That is
a silent ~5.7x on the 1.7B Q1_0 dot kernel (0.55 vs 3.12 GB/s) with the dispatch
banner still reporting `avx2` — the bug this script exists to catch.

Usage: check-kernel-baseline.py <hearth_quant assembly>

The listing comes from a deliberately weak baseline, so that any kernel which
forgot to declare a feature shows its calls. Both the library and its test build
are checked:

    RUSTFLAGS="-C target-cpu=x86-64-v2" \
      cargo rustc -p hearth-quant --release --lib --target-dir target/baseline-asm -- --emit=asm
    RUSTFLAGS="-C target-cpu=x86-64-v2" \
      cargo rustc -p hearth-quant --release --tests --target-dir target/baseline-asm-tests -- --emit=asm

`scripts/baseline-invariance.py` is the measurement this stands in for: build `benchk` at two
baselines, then check that every arm and the dispatch read the same at both.

The test build matters as much as the library one: a kernel only the tests reach is
a kernel the library build never sees, and the Q1_0_G128 VNNI kernel sat there behind
`#[cfg(test)]` while it measured +180% against the kernel it was meant to replace.
Nothing re-measured it, because nothing could dispatch it.
"""
import re
import sys

# A call to one of these can only mean lost inlining: each is a single
# instruction, so it should never be a function call anywhere in the crate.
# FMA3's vfmadd/vfmsub family and AVX-VNNI/AVX-512-VNNI's vpdpbusd/vpdpwssd.
# Never scope this rule by function name -- it holds for every caller.
SINGLE_INSN = re.compile(r"fn?madd|fn?msub|dpbusd|dpwssd")

# F16 conversions are a legitimate call in code that does not declare `f16c`
# (that is how `half` runtime-dispatches), which is correct in scalar/SSE
# fallbacks -- but a body that is only ever reached behind an AVX2/FMA/F16C
# probe must have the hardware conversion inlined.
F16 = re.compile(r"f16_to_f32|f32_to_f16")
HOT = re.compile(r"avx2|avx512|vnni", re.IGNORECASE)

# A function inside a `tests` module is a harness, not a `#[target_feature]` body: it may
# legitimately call `half`'s runtime-dispatched conversion, which is how `half` dispatches
# and is correct at the weak baseline the harness itself is compiled at. Kernels are never
# named `...tests...`, so every kernel the tests reach is still checked -- by this rule, and
# unconditionally by SINGLE_INSN, which holds for every caller and needs no name scoping.
HARNESS = re.compile(r"tests")

LABEL = re.compile(r"^(_R\S+):$")
CALL = re.compile(r"\bcallq?\s+(\S+)")


def main(path: str) -> int:
    function = "?"
    bad: list[tuple[str, str]] = []
    with open(path, encoding="utf-8", errors="replace") as listing:
        for line in listing:
            label = LABEL.match(line.strip())
            if label:
                function = label.group(1)
                continue
            call = CALL.search(line)
            if not call:
                continue
            target = call.group(1)
            if SINGLE_INSN.search(target) or (
                F16.search(target)
                and HOT.search(function)
                and not HARNESS.search(function)
            ):
                bad.append((function, target))
    if bad:
        print(f"{path}: {len(bad)} out-of-line intrinsic call(s) -- a kernel is missing fma/f16c:")
        for function, target in bad[:20]:
            print(f"  in {function.split('hearth_quant')[-1]}  ->  {target}")
        return 1
    print(f"{path}: baseline-invariant (no out-of-line FMA/VNNI calls, no f16 calls in SIMD bodies)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv) == 2 else 2)
