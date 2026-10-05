#!/usr/bin/env python3
"""gesv off-grid data gate (docs/design/flat-select-p5/gesv.md).

Draws random OFF-grid (dtype, n, nrhs) points, asks the OLD router for its first choice with the
real tiny capacities applied (gesv_transcribe --points, built against a tree that still has
route_gesv.hh), and compares it with what select::choose takes on the new tables: the nearest row
(scripts/sweep_to_table.py's nearest(), the same rule as Table::nearest), first entry that passes
can_run's capacity terms on a sub-group-32 GPU with max_wg >= 64.

    tools/transcribe/gesv_offgrid_gate.py /tmp/gt [--per-dtype 2500] [--seed 1]
"""
import argparse
import collections
import math
import os
import random
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import sweep_to_table as stt  # noqa: E402

DTYPES = ("float", "double", "cfloat", "cdouble")
TINY_CAP = {"float": 32, "double": 32, "cfloat": 32, "cdouble": 16}  # gesv_tiny.cc tiny_cap
TINY_MAX_RHS = 4  # solve_native.hh kGesvTinyMaxRhs


def can_run(choice, dtype, n, nrhs):
    if choice == "tiny":
        return n <= TINY_CAP[dtype] and nrhs <= TINY_MAX_RHS
    return True


def grid():
    rows = stt.parse_table(open(os.path.join(REPO, "tuned", "gesv.float.sm_89.txt")).read())[2]
    return {k for k, _ in rows}


def draw(rng, cells, per_dtype):
    pts = []
    for dt in DTYPES:
        got = 0
        while got < per_dtype:
            if rng.random() < 0.4:
                n = rng.randint(8, 48)  # the window and ceiling edges
            else:
                n = max(1, int(round(math.exp(rng.uniform(0, math.log(6000))))))
            nrhs = max(1, int(round(math.exp(rng.uniform(0, math.log(400))))))
            if (n, nrhs) in cells:
                continue
            pts.append((dt, n, nrhs))
            got += 1
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcriber")
    ap.add_argument("--per-dtype", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pts = draw(rng, grid(), args.per_dtype)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("".join(f"{d},{n},{r}\n" for d, n, r in pts))
        path = f.name
    out = subprocess.run([args.transcriber, "--points", path], capture_output=True, text=True, check=True)
    os.unlink(path)
    old = {}
    for line in out.stdout.splitlines()[1:]:
        d, n, r, c = line.split(",")
        old[(d, int(n), int(r))] = c
    worst = 1.0
    for dev in ("sm_89", "sm_120"):
        for dt in DTYPES:
            hdr, keyspec, rows = stt.parse_table(open(os.path.join(REPO, "tuned", f"gesv.{dt}.{dev}.txt")).read())
            bad = collections.Counter()
            total = agree = 0
            for d, n, r in pts:
                if d != dt:
                    continue
                row = stt.nearest(rows, (n, r), keyspec)
                new = next(c for c, _ in row[1] if can_run(c, dt, n, r))
                total += 1
                if new == old[(d, n, r)]:
                    agree += 1
                else:
                    bad[(n, r, old[(d, n, r)], new, row[0])] += 1
            rate = agree / total
            worst = min(worst, rate)
            print(f"{dev} {dt}: {agree}/{total} agree ({100 * rate:.2f}%)")
            for (n, r, o, nw, k), cnt in sorted(bad.items()):
                print(f"   n={n} nrhs={r}: old {o}, new {nw} (row n={k[0]} nrhs={k[1]}) x{cnt}")
    print(f"worst agreement {100 * worst:.2f}% over {len(pts)} off-grid points per device")
    return 0 if worst >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
