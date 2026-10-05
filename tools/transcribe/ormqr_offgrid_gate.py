#!/usr/bin/env python3
"""ormqr data gate (docs/design/flat-select-p5/ormqr.md): random OFF-GRID points across the key
space, the OLD predicate (the transcriber's --points mode, built against 424a45bc) against the
new selection replayed in Python (sweep_to_table.nearest on the shipped table, then the first
entry can_run accepts, then the last resort). can_run: blocked = GPU and not complex Trans;
vendor = the factorization library compiled in. Both vendor-present and vendor-free builds.

    tools/transcribe/ormqr_offgrid_gate.py /tmp/ot [N] [seed]
"""

import math
import os
import random
import subprocess
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import sweep_to_table as st  # noqa: E402

GRID_M = (1, 8, 64, 512)
GRID_Q = (1, 32, 1024)
GRID_BATCH = (128, 2048, 32768)
LAST_RESORT = ("blocked", "vendor")


def log_uniform(rng, lo, hi):
    return max(lo, min(hi, int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))))


def can_run(choice, dtype, trans, vendor):
    if choice == "blocked":
        return not (dtype.startswith("c") and trans == "T")
    return vendor


def first_runnable(ranked, dtype, trans, vendor, last_resort):
    for c in ranked:
        if can_run(c, dtype, trans, vendor):
            return c
    if last_resort:
        for c in LAST_RESORT:
            if can_run(c, dtype, trans, vendor):
                return c
    return "none"


def main():
    binary = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 20261005)
    points = []
    while len(points) < n:
        m = log_uniform(rng, 1, 8192)
        k = log_uniform(rng, 1, m)
        q = log_uniform(rng, 1, 16384)
        b = log_uniform(rng, 1, 131072)
        if m in GRID_M and k in GRID_M and q in GRID_Q and b in GRID_BATCH:
            continue  # on-grid: equal by construction
        points.append((rng.choice(["float", "double", "cfloat", "cdouble"]), rng.choice("LR"), rng.choice("NTC"),
                       m, k, q, b))
    stdin = "".join(" ".join(map(str, p)) + "\n" for p in points)
    old = subprocess.run([binary, "--points"], input=stdin, capture_output=True, text=True, check=True)
    old_ranked = [line.split("|") for line in old.stdout.split()]
    assert len(old_ranked) == len(points), (len(old_ranked), len(points))
    tables = {}
    for dev in ("sm_89", "sm_120"):
        for dt in ("float", "double", "cfloat", "cdouble"):
            with open(st.table_path("ormqr", dt, dev)) as f:
                tables[(dt, dev)] = st.parse_table(f.read())
    total, agree = Counter(), Counter()
    bad = Counter()
    for p, ranked in zip(points, old_ranked):
        dt, side, trans, m, k, q, b = p
        for dev in ("sm_89", "sm_120"):
            _, keyspec, rows = tables[(dt, dev)]
            hit = st.nearest(rows, (side, trans, m, k, q, b), keyspec)
            new_ranked = [c for c, _ in hit[1]]
            for vendor in (True, False):
                # The old dispatch always resolved with the vendor "available" and threw at the
                # vendor call when it was compiled out: its first-runnable has no last resort.
                o = first_runnable(ranked, dt, trans, vendor, last_resort=False)
                nw = first_runnable(new_ranked, dt, trans, vendor, last_resort=True)
                key = (dt, dev, "vendor" if vendor else "vendor-free")
                total[key] += 1
                if o == nw:
                    agree[key] += 1
                else:
                    bad[(dt, dev, key[2], side, trans, o, nw)] += 1
    worst = 1.0
    for key in sorted(total):
        rate = agree[key] / total[key]
        worst = min(worst, rate)
        print(f"{key[0]:8s} {key[1]:7s} {key[2]:12s} {agree[key]:5d}/{total[key]:5d} = {100 * rate:.2f}%")
    print(f"points: {len(points)} off-grid; lookups: {sum(total.values())}; worst agreement {100 * worst:.2f}%")
    for k, v in sorted(bad.items()):
        print("disagree", k, v)
    return 0 if worst >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
