#!/usr/bin/env python3
"""Find tuning parameters that do not earn their place in the search.

Grid search costs the product of every grid, so a parameter that changes
nothing multiplies the run time of every other parameter for free. This reads a
sweep log and, for each (bench, case, parameter), reports how much the metric
moves across that parameter's values when the others are held fixed.

Usage:
    python3 evaluation/tuning/sensitivity.py build/tuning/sweep_log.txt
    python3 evaluation/tuning/sensitivity.py build/tuning/sweep_log.txt --noise 1.5

Read it as: "holding everything else constant, sweeping this parameter changes
the metric by X% in the median case". Below the noise floor the parameter is
not being tuned, it is being sampled.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# The benchmarks colour their output; the arg columns follow the name column.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The name column has no trailing space before the first argument -- rows read
# "(BM_SYEV<float, batchlas::Backend::CUDA>)1024    256 ...". Splitting on
# whitespace alone glues ")1024" to the template arguments and silently drops
# arg0, which shifts every column by one and makes the iteration count look
# like the last tuned parameter. Match the closing paren explicitly.
ROW = re.compile(r"\(BM_([A-Z_0-9]+)<[^)]*\)\s*(.*)$")

# arg_spec per benchmark executable, matching evaluation/tuning/spaces/*.json.
# Only args that any space actually tunes need naming; the rest anchor position.
ARG_NAMES: Dict[str, List[str]] = {
    "STEDC": ["n", "batch", "recursion_threshold", "flat", "threads_per_root", "wg_multiplier"],
    "STEQR": ["n", "batch", "n_sweeps", "wg_multiplier", "shift_kind"],
    "SYTRD_BLOCKED": ["n", "batch", "nb", "uplo"],
    "ORMQR_BLOCKED": ["n", "batch", "side", "trans", "block_size"],
    "SYEV": ["n", "batch", "nb", "wg", "fuse"],
    "GESVD_BLOCKED": ["n", "batch", "jobu", "jobvh"],
    "LATRD_LOWER_PANEL": ["n", "batch", "ib", "j0", "fuse"],
    "SYEV_TWO_STAGE": ["n", "batch", "kd"],
    "SB2ST_HH": ["n", "batch", "kd"],
}

# Which positional args identify the *case* rather than a tuned parameter.
CASE_KEYS = {"n", "batch", "uplo", "side", "trans", "jobu", "jobvh", "j0", "ib", "n_sweeps"}


def parse(log_path: Path) -> List[Tuple[str, List[int], float]]:
    rows: List[Tuple[str, List[int], float]] = []
    for raw in log_path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        m = ROW.search(line)
        if not m:
            continue
        bench = m.group(1)
        # arg0 arg1 ... iterations avg_ms stddev_ms [GFLOPS] metric
        fields = m.group(2).split()
        nums: List[str] = [f for f in fields if re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", f)]
        if len(nums) < 4:
            continue
        metric = float(nums[-1])
        nargs = len(ARG_NAMES.get(bench, []))
        if nargs == 0 or len(nums) < nargs:
            continue
        args = [int(float(x)) for x in nums[:nargs]]
        rows.append((bench, args, metric))
    return rows


def parse_profile(profile_path: Path) -> List[Tuple[str, List[int], float]]:
    """Rows from a profile's `measurements`, preferred over log scraping.

    The profile knows each bench's arg_spec and env_spec, so env-only
    parameters -- which never appear in the benchmark's CSV columns -- are
    covered here and cannot be covered from a log.
    """
    import json

    profile = json.loads(profile_path.read_text())
    rows: List[Tuple[str, List[int], float]] = []
    for entry in profile.get("results", []):
        if not isinstance(entry, dict):
            continue
        bench = str(entry.get("bench"))
        arg_spec = [str(a) for a in (entry.get("arg_spec") or [])]
        env_spec = entry.get("env_spec") or {}
        # Env params become extra trailing columns with synthetic names.
        env_names = sorted(str(k) for k in env_spec)
        env_vars = [str(env_spec[k]) for k in env_names]
        names = arg_spec + env_names
        ARG_NAMES[bench] = names
        for meas in entry.get("measurements") or []:
            args = [int(v) for v in meas.get("args", [])]
            env = meas.get("env") or {}
            if len(args) != len(arg_spec):
                continue
            try:
                extra = [int(env[v]) for v in env_vars]
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((bench, args + extra, float(meas["value"])))
    return rows


def analyse(rows, noise_pct: float):
    # (bench, param) -> list of per-slice spreads, where a slice fixes every
    # other column. This is a one-at-a-time sensitivity, which is the right
    # question for "can I delete this axis from the grid".
    spreads: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)

    by_bench: Dict[str, List[Tuple[List[int], float]]] = defaultdict(list)
    for bench, args, metric in rows:
        by_bench[bench].append((args, metric))

    for bench, entries in by_bench.items():
        names = ARG_NAMES[bench]
        for idx, name in enumerate(names):
            if name in CASE_KEYS:
                continue
            slices: Dict[Tuple[int, ...], Dict[int, float]] = defaultdict(dict)
            for args, metric in entries:
                key = tuple(v for i, v in enumerate(args) if i != idx)
                # Keep the best observation per value: repeats are re-runs.
                prev = slices[key].get(args[idx])
                if prev is None or metric < prev:
                    slices[key][args[idx]] = metric
            for key, vals in slices.items():
                if len(vals) < 2:
                    continue
                lo, hi = min(vals.values()), max(vals.values())
                if lo <= 0:
                    continue
                spreads[(bench, name)].append(100.0 * (hi - lo) / lo)
                counts[(bench, name)] += len(vals)

    print(f"{'bench':<20}{'parameter':<22}{'median':>9}{'p90':>9}{'max':>9}  {'obs':>6}  verdict")
    print("-" * 92)
    dead = []
    for (bench, name), vals in sorted(spreads.items(), key=lambda kv: -statistics.median(kv[1])):
        med = statistics.median(vals)
        p90 = sorted(vals)[max(0, int(0.9 * len(vals)) - 1)]
        mx = max(vals)
        if mx < noise_pct:
            verdict = "DEAD - never moves the metric"
            dead.append((bench, name, mx))
        elif med < noise_pct:
            verdict = "mostly flat - shrink the grid"
        else:
            verdict = "keep"
        print(f"{bench:<20}{name:<22}{med:>8.2f}%{p90:>8.2f}%{mx:>8.2f}%  {counts[(bench,name)]:>6}  {verdict}")

    if dead:
        print(f"\nDead axes (max spread < {noise_pct}% across every slice):")
        for bench, name, mx in dead:
            print(f"  {bench}.{name}: max {mx:.2f}%")
    return dead


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, nargs="+",
                    help="profile JSON from tune.py (preferred), or sweep log(s)")
    ap.add_argument("--noise", type=float, default=1.0,
                    help="percent spread below which a parameter counts as inert (default 1.0)")
    args = ap.parse_args()

    rows: List[Tuple[str, List[int], float]] = []
    for p in args.log:
        rows.extend(parse_profile(p) if p.suffix == ".json" else parse(p))
    if not rows:
        print("no benchmark rows parsed", file=sys.stderr)
        return 1
    print(f"parsed {len(rows)} measurements from {len(args.log)} log(s)\n")
    analyse(rows, args.noise)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
