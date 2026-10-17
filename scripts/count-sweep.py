#!/usr/bin/env python3
"""Sweep the worker count for a model, and judge it on the loaded number.

The worker count is a per-path decision (`crates/hearth-llm/src/pool.rs`): dense
runs `physical`, the hybrids run a quarter past physical for `Q2_0`/`PQ2_0` and
three quarters of logical for `PTQ1_0`. That shape came from measurement, and the
measurement has a trap this script exists to close: **a count chosen on a quiet box
is not chosen.**

Count 9 on an 8C/16T box is the case study, and the reason this script has a load
arm. Quiet, one window read 9 as faster than 8 at every prompt length (prefill
1977 -> 1831 ms at 19 prompt tokens, 5039 -> 4700 at 55, 9396 -> 8952 at 109); an
earlier window of the *same* build read 12999/8320/13256/19933/8439 at 9 against
8571-9158 for 8, and that 2.4x spread was blamed on the code for a day. A dispatch
takes as long as its *slowest* worker, so a count over the physical core count pays
for its extra runnable thread in full, while the count-equals-physical arm loses only
a little -- which is why the loaded row is the one a default has to survive.

The load arm is a stand-in, and it is not a proof: the first controlled run (dense 4B
Q2_0, counts 8/9/10, load 0 and 6) put 8 and 9 within 5% of each other in *both*
conditions and 10 clearly worse (prefill 4656/4467/8782 ms quiet, 13795/13959/15912
loaded), and it did not reproduce the 2.4x at all. So the default stays at `physical`
because it is never worse -- and count 9 is the open question this script exists to
settle on hardware other than this laptop.

So every count is measured in *both* conditions, repeated, with the order rotated so
no count sits in the same position every repeat (the second run of a pair is worth
2-3% on this box, and the absolute level drifts ~10% an hour, so only within-window
comparisons mean anything). The verdict names the count that is best on the *loaded*
spread, not the quiet mean, and it refuses to recommend a switch for a difference
inside the noise band. It reports; it does not decide.

Usage:
    scripts/count-sweep.py --model models/Ternary-Bonsai-4B-Q2_0.gguf \
        --counts 8,9,10 --load 0,6 --repeats 3
"""
import argparse
import os
import re
import subprocess
import sys
import time

TIMING = re.compile(r"\[(\d+) new tokens \((\d+) prompt\), ([0-9.]+) ms, ([0-9.]+) tok/s\]")
PREFILL_PROMPT = (
    "The quick brown fox jumps over the lazy dog and then considers the implications "
    "of that action. "
) * 6


def cli_path(explicit: str | None) -> str:
    """The release CLI, `target/release/` by default -- same shape as benchk_path."""
    if explicit:
        return explicit
    exe = "hearth-chat-cli.exe" if os.name == "nt" else "hearth-chat-cli"
    return os.path.join("target", "release", exe)


def run_once(binary: str, model: str, count: int, prompt: str, max_tokens: int) -> tuple[float, int]:
    """One generation, as `HEARTH_THREADS=<count>`. Returns (ms, prompt tokens).

    The timing line goes to **stderr** (the CLI writes its banner and timings there;
    stdout carries only the generated text), so that is where this reads, with stdout
    as a fallback. If neither has it, this exits rather than returning a stale number,
    which is how a bad regex silently reports the previous run's time.
    """
    env = {**os.environ, "HEARTH_THREADS": str(count)}
    proc = subprocess.run(
        [binary, model, "--temp", "0", "--max-tokens", str(max_tokens), "--prompt", prompt, "--prompt-raw"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    m = TIMING.search(proc.stderr) or TIMING.search(proc.stdout)
    if not m:
        sys.exit(f"no timing line from {binary} at count {count} (prompt {prompt[:32]!r}...)")
    return float(m.group(3)), int(m.group(2))


class Load:
    """`n` subprocesses burning a core each: a stand-in for the browser and the virus
    scanner a real machine has running, not for a benchmark peer.

    Deliberately coarse and dependency-free (std lib only, works on the three
    platforms). Zero load is the quiet arm and is always measured too, because the
    point is the *difference* between the two, not either on its own.
    """

    def __init__(self, n: int):
        self.n = n
        self.procs: list[subprocess.Popen] = []

    def __enter__(self) -> "Load":
        for _ in range(self.n):
            self.procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", "while True: pass"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        if self.n:
            time.sleep(1.0)  # let them land on cores before the first measurement
        return self

    def __exit__(self, *exc) -> bool:
        for p in self.procs:
            p.kill()
        for p in self.procs:
            p.wait()
        return False


def sweep(
    binary: str, model: str, counts: list[int], repeats: int, prompt: str, max_tokens: int, load: int
) -> tuple[dict[int, list[float]], int]:
    """ms per count. Order rotates per repeat so each count sees every position."""
    rows: dict[int, list[float]] = {c: [] for c in counts}
    prompt_tokens = 0
    with Load(load):
        for r in range(repeats):
            shift = r % len(counts)
            for c in counts[shift:] + counts[:shift]:
                ms, prompt_tokens = run_once(binary, model, c, prompt, max_tokens)
                rows[c].append(ms)
    return rows, prompt_tokens


def stats(vals: list[float]) -> tuple[float, float, float, float]:
    """(mean, min, max, spread% of mean)."""
    mean = sum(vals) / len(vals)
    lo, hi = min(vals), max(vals)
    return mean, lo, hi, 100.0 * (hi - lo) / mean


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default=None, help="default: target/release/hearth-chat-cli[.exe]")
    ap.add_argument("--counts", required=True, help="comma list, e.g. 8,9,10")
    ap.add_argument("--load", default="0", help="comma list of load threads to repeat under (default 0)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=50)
    ap.add_argument("--passes", choices=("decode", "prefill", "both"), default="both")
    ap.add_argument("--raw", action="store_true", help="print every repeat, not just the summary")
    ap.add_argument("--tolerance", type=float, default=5.0, help="percent: a smaller gap is noise (default 5)")
    ap.add_argument("--spread", type=float, default=10.0, help="percent: a spread above this is fragile (default 10)")
    args = ap.parse_args()

    binary = cli_path(args.binary)
    if not os.path.exists(binary):
        sys.exit(f"no CLI at {binary} -- cargo build --release -p hearth-llm --bin hearth-chat-cli")
    counts = [int(c) for c in args.counts.split(",")]
    loads = [int(l) for l in args.load.split(",")]
    if len(counts) < 2:
        sys.exit("--counts wants at least two counts to compare")

    passes = [
        (f"decode ({args.decode_tokens} tokens)", args.decode_tokens, "Hello"),
        ("prefill", 1, PREFILL_PROMPT),
    ]
    if args.passes != "both":
        passes = [p for p in passes if p[0].startswith(args.passes)]

    # One warm-up per pass, so the clock ramp is not charged to the first count.
    for _, max_tokens, prompt in passes:
        run_once(binary, args.model, counts[0], prompt, max_tokens)

    print(f"\n{os.path.basename(args.model)}  counts={counts}  repeats={args.repeats}")
    print(f"load={loads} (busy subprocesses)   tolerance={args.tolerance:.0f}%   spread={args.spread:.0f}%")

    results: dict[str, dict[int, dict[int, tuple[float, float, float, float]]]] = {}
    for label, max_tokens, prompt in passes:
        results[label] = {}
        for load in loads:
            rows, prompt_tokens = sweep(binary, args.model, counts, args.repeats, prompt, max_tokens, load)
            results[label][load] = {c: stats(v) for c, v in rows.items()}
            print(f"\n{label}  ({prompt_tokens} prompt tokens)  load={load}")
            print("  " + "count".ljust(8) + "".join(f"{c:>9}" for c in counts))
            print("  " + "mean".ljust(8) + "".join(f"{results[label][load][c][0]:9.0f}" for c in counts))
            print("  " + "spread".ljust(8) + "".join(f"{results[label][load][c][3]:8.1f}%" for c in counts))
            if args.raw:
                for c in counts:
                    print("  " + f"raw {c}".ljust(8) + "  ".join(f"{v:.0f}" for v in rows[c]))

    def verdict(label: str, load: int) -> None:
        row = results[label][load]
        best_mean = min(counts, key=lambda c: row[c][0])
        worst_spread = max(counts, key=lambda c: row[c][3])
        base = row[best_mean][0]
        spread = [c for c in counts if row[c][3] > args.spread]
        line = f"  {label:24} load={load:<3} best mean {best_mean} ({base:.0f} ms)"
        inside = [c for c in counts if 100.0 * (row[c][0] / base - 1.0) <= args.tolerance]
        if len(inside) > 1:
            line += f"   {sorted(inside)} are within {args.tolerance:.0f}% -- no switch justified"
        if spread:
            line += f"   fragile spread: {spread} (worst {worst_spread} {row[worst_spread][3]:.1f}%)"
        print(line)

    print("\nverdict")
    for label in results:
        for load in loads:
            verdict(label, load)
    print(
        "\nA count is a default only if it holds up in the load rows: the quiet mean is where a\n"
        "count looks best and the loaded spread is where it costs a user prefill latency."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
