#!/usr/bin/env python3
"""getrf data gate: the OLD router against the transcribed tables at random off-grid points.

    python3 tools/transcribe/getrf_gate.py /tmp/gt [--points 3000] [--seed 1]

/tmp/gt is tools/transcribe/getrf_transcribe.cc built against the old tree (its header has the
g++ line); its --eval mode is the oracle. For each dtype and device table, random (n, batch)
points (log-uniform, n in [1, 2048], batch in [1, 65536], grid cells excluded) are looked up with
the same nearest() as Table::nearest, and the first entry passing a can_run stand-in is compared
with the old Auto choice. The stand-in is src/ops/getrf/getrf.cc's can_run with the device facts
fixed: tiny n <= getrf_tiny_max_n<T>(), cta n <= CTA_MAX, blocked always, vendor iff present.
Every (CTA_MAX, vendor) scenario is checked; disagreeing regions are listed.
"""

import argparse
import math
import os
import random
import subprocess
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import sweep_to_table as stt  # noqa: E402

DTYPES = ("float", "double", "cfloat", "cdouble")
TINY_MAX = {"float": 32, "double": 32, "cfloat": 32, "cdouble": 16}
UNLIMITED = 1 << 30
CTA_CAPS = (UNLIMITED, 48, 24)
LAST_RESORT = ("blocked", "vendor")


def runnable(c, dtype, n, cta_max, vendor):
    return {"tiny": n <= TINY_MAX[dtype], "cta": n <= cta_max, "blocked": True, "vendor": vendor}[c]


def new_choice(row, dtype, n, cta_max, vendor):
    for c in [e[0] for e in row[1]] + list(LAST_RESORT):
        if runnable(c, dtype, n, cta_max, vendor):
            return c
    return "none"


def points(rng, count, grid):
    out = []
    while len(out) < count:
        n = max(1, round(math.exp(rng.uniform(0, math.log(2048)))))
        b = max(1, round(math.exp(rng.uniform(0, math.log(65536)))))
        if (n, b) not in grid:
            out.append((n, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle")
    ap.add_argument("--points", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    worst = 1.0
    for device in ("sm_89", "sm_120"):
        for dtype in DTYPES:
            with open(stt.table_path("getrf", dtype, device)) as f:
                _, keyspec, parsed = stt.parse_table(f.read())
            grid = {k for k, _ in parsed}
            pts = points(rng, args.points, grid)
            both_off = sum(1 for n, b in pts if n not in {k[0] for k in grid} and b not in {k[1] for k in grid})
            for cap in CTA_CAPS:
                text = "".join(f"{dtype} {n} {b}\n" for n, b in pts)
                out = subprocess.run([args.oracle, "--eval", str(cap)], input=text, capture_output=True,
                                     text=True, check=True).stdout.split("\n")
                bad = Counter()
                total = agree = 0
                for (n, b), line in zip(pts, out):
                    old_v, old_f, _ = line.split()
                    row = stt.nearest(parsed, (n, b), keyspec)
                    for vendor, old in ((True, old_v), (False, old_f)):
                        new = new_choice(row, dtype, n, cap, vendor)
                        total += 1
                        if new == old:
                            agree += 1
                        else:
                            bad[(vendor, old, new, f"row n={row[0][0]} batch={row[0][1]}")] += 1
                rate = agree / total
                worst = min(worst, rate)
                cap_s = "unlimited" if cap == UNLIMITED else str(cap)
                print(f"{device} {dtype} cta_max={cap_s}: {agree}/{total} agree ({100 * rate:.2f}%), "
                      f"{len(pts)} points, {both_off} off-grid in both keys")
                for (vendor, old, new, where), k in sorted(bad.items()):
                    print(f"   vendor={'yes' if vendor else 'no'} old={old} new={new} at {where}: {k}")
    print(f"worst agreement {100 * worst:.2f}%")
    return 0 if worst >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
