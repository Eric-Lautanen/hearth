#!/usr/bin/env python3
"""Check that one kernel source measures the same at every compile baseline.

The property this checks is the one the 2026-10-16 bug violated: `benchk` reads
per-arm GB/s for the *same source* built at different `-C target-cpu` levels. It
should not matter which baseline a kernel was compiled at, because every kernel
body is runtime-dispatched and declares its own features -- when it does not, the
intrinsic stops inlining and the arm is silently slower at the weak baseline with
the dispatch banner still naming it (0.55 vs 3.12 GB/s on the 1.7B Q1_0, and 0.87
vs 2.64 for the Q2_0 LUT arm before its f16c fix).

Two failures are reported, and they are different bugs:

  * **arm drift** -- the same arm reads > tolerance lower at one baseline. The
    body is missing a declaration (or calls a runtime-probed helper), which is
    what `scripts/check-kernel-baseline.py` catches statically; this is the
    measurement that confirms it.
  * **dispatch drift** -- `dispatched` is not within tolerance of the fastest arm
    *at that baseline*. The order was tuned against one baseline (usually native
    or v3) and picks a slower arm where it is compiled weakly.

The static guard is the CI gate; this script is the measurement it is a proxy for,
so it builds nothing and takes the target directories as arguments.

Usage:
    cargo build --release -p hearth-llm --example benchk --target-dir target/inv-v2   # RUSTFLAGS="-C target-cpu=x86-64-v2"
    cargo build --release -p hearth-llm --example benchk --target-dir target/inv-v3
    scripts/baseline-invariance.py --baseline v2=target/inv-v2 --baseline v3=target/inv-v3 \
        --model models/Ternary-Bonsai-4B-Q2_0.gguf --dtype q2_0 --prefix blk.0.

RUSTFLAGS overrides `build.rustflags` in `.cargo/config.toml`, which is how a side
target directory gets a baseline other than the shipped one.
"""
import argparse
import os
import re
import subprocess
import sys

AGGREGATE = re.compile(r"^\s*(\S+):\s+([0-9.]+) GB/s single-thread aggregate")
DISPATCHED = "dispatched"


def benchk_path(target_dir: str) -> str:
    exe = "benchk.exe" if os.name == "nt" else "benchk"
    return os.path.join(target_dir, "release", "examples", exe)


def run_baseline(name: str, target_dir: str, model: str, dtype: str, prefix: str) -> dict:
    """Per-arm GB/s, taking the best of two passes (benchk's own protocol)."""
    exe = benchk_path(target_dir)
    if not os.path.exists(exe):
        sys.exit(f"{name}: no benchk at {exe} -- build it with --target-dir {target_dir}")
    best: dict[str, float] = {}
    for _ in range(2):
        out = subprocess.run(
            [exe, model, dtype, prefix], capture_output=True, text=True, check=True
        ).stdout
        for line in out.splitlines():
            m = AGGREGATE.match(line)
            if m:
                arm, gbps = m.group(1), float(m.group(2))
                best[arm] = max(best.get(arm, 0.0), gbps)
    if not best:
        sys.exit(f"{name}: benchk printed no aggregate rows (is {dtype} in {model}?)")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="append", required=True, metavar="NAME=TARGET_DIR")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", required=True, help="ptq1_0 | pq2_0 | q2_0 | q1_0 | q1_0_g128")
    ap.add_argument("--prefix", default="blk.0.", help="tensor-name prefix (default blk.0.)")
    ap.add_argument("--tolerance", type=float, default=10.0, help="percent an arm may fall (default 10)")
    args = ap.parse_args()

    baselines = []
    for spec in args.baseline:
        name, _, target_dir = spec.partition("=")
        if not target_dir:
            sys.exit(f"--baseline wants NAME=TARGET_DIR, got {spec!r}")
        baselines.append((name, target_dir))

    results = {name: run_baseline(name, d, args.model, args.dtype, args.prefix) for name, d in baselines}
    arms = sorted({arm for r in results.values() for arm in r})
    names = [name for name, _ in baselines]
    floor = 1.0 - args.tolerance / 100.0

    width = max(len(a) for a in arms) + 2
    header = "arm".ljust(width) + "".join(n.rjust(9) for n in names) + "   min/max"
    print(f"\n{args.model}  dtype={args.dtype}  prefix={args.prefix}")
    print(header)
    print("-" * len(header))
    failures = []
    for arm in arms:
        row = [results[n].get(arm) for n in names]
        cells = "".join((f"{v:.2f}" if v is not None else "--").rjust(9) for v in row)
        present = [v for v in row if v is not None]
        ratio = min(present) / max(present)
        flag = "  <-- drift" if ratio < floor else ""
        print(f"{arm.ljust(width)}{cells}   {ratio:6.2f}{flag}")
        if arm != DISPATCHED and ratio < floor:
            failures.append(f"arm {arm} reads {ratio:.2f} of its best across baselines")

    # The dispatch invariant: at every baseline, `dispatched` must be the fastest arm.
    print()
    for n in names:
        r = results[n]
        if DISPATCHED not in r:
            continue
        others = {a: v for a, v in r.items() if a != DISPATCHED}
        best_arm, best = max(others.items(), key=lambda kv: kv[1])
        ratio = r[DISPATCHED] / best
        note = "" if ratio >= floor else f"  <-- dispatch drift (should be {best_arm})"
        print(f"{n}: dispatched {r[DISPATCHED]:.2f} vs best arm {best_arm} {best:.2f}  ({ratio:.2f}){note}")
        if ratio < floor:
            failures.append(f"{n}: dispatch picks {DISPATCHED} over {best_arm} ({ratio:.2f})")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\nOK: every arm and the dispatch are within {args.tolerance:.0f}% across {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
