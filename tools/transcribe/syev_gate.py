#!/usr/bin/env python3
"""syev data gate (docs/design/flat-select-p5/syev.md): the OLD router (syev_transcribe --points)
against select's choice on the shipped tables (scripts/sweep_to_table.py nearest() plus syev's
can_run capacities) at random off-grid points.
  tools/transcribe/syev_gate.py gen POINTS        # 2500 off-grid points per dtype
  /tmp/st --points POINTS > OLD                   # the transcriber, built as in its header
  tools/transcribe/syev_gate.py cmp POINTS OLD    # agreement per device and dtype
DROP_N=33,449 drops grid rows before the lookup, to show the gate can fail."""
import math
import os
import random
import sys
from collections import Counter, defaultdict

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(W, "scripts"))
import sweep_to_table as st  # noqa: E402

GRID_N = {1, 2, 3, 4, 6, 8, 9, 12, 16, 20, 24, 25, 28, 32, 33, 40, 48, 64, 96, 128, 192, 256, 257, 320, 321,
          384, 448, 449, 512, 513, 640, 768, 1024, 1025, 1536, 2048, 4096}
GRID_B = {128, 512, 2048, 8192, 32768}
DTYPES = ("float", "double", "cfloat", "cdouble")
PER_DTYPE = 2500


def gen(path):
    rng = random.Random(20261005)
    with open(path, "w") as f:
        for dt in DTYPES:
            k = 0
            while k < PER_DTYPE:
                n = int(round(2 ** rng.uniform(0, math.log2(6000))))
                b = int(round(2 ** rng.uniform(0, 16)))
                if n in GRID_N or b in GRID_B or n < 1:
                    continue
                f.write(f"{dt},{rng.choice('NV')},{n},{b}\n")
                k += 1
            # Every off-grid n up to 64 and two either side of each large-n window edge.
            edges = [t + d for t in (256, 320, 448, 512, 1024) for d in (-2, -1, 2, 3)]
            for n in [m for m in range(1, 65)] + edges:
                if n in GRID_N:
                    continue
                for b in (1, 100, 3000, 50000):
                    for j in "NV":
                        f.write(f"{dt},{j},{n},{b}\n")


def can_run(c, n, vendor):
    if c in ("cta", "cta_fused", "jacobi"):
        return 1 <= n <= 32
    if c in ("blocked", "two_stage"):
        return n >= 1
    return vendor


def cmp(points, old_out):
    tables = {}
    for dev in ("sm_89", "sm_120"):
        for dt in DTYPES:
            with open(os.path.join(W, "tuned", f"syev.{dt}.{dev}.txt")) as f:
                header, keyspec, rows = st.parse_table(f.read())
            drop = {int(x) for x in os.environ.get("DROP_N", "").split(",") if x}
            tables[(dev, dt)] = (keyspec, [r for r in rows if r[0][1] not in drop])
    stats = defaultdict(Counter)
    bad = defaultdict(list)
    with open(old_out) as f:
        for line in f:
            dt, jobz, n, b, ranked = line.strip().split(",")
            n, b = int(n), int(b)
            old = ranked.split("|")
            old_vendor = next(c for c in old if can_run(c, n, True))
            old_free = next((c for c in old if c != "vendor" and can_run(c, n, False)), None)
            for dev in ("sm_89", "sm_120"):
                keyspec, rows = tables[(dev, dt)]
                row = st.nearest(rows, (jobz, n, b), keyspec)
                new = [c for c, _ in row[1]]
                new_vendor = next(c for c in new if can_run(c, n, True))
                new_free = next((c for c in new if c != "vendor" and can_run(c, n, False)), None)
                full_old = [c for c in old if can_run(c, n, True)]
                full_new = [c for c in new if can_run(c, n, True)]
                s = stats[(dev, dt)]
                s["points"] += 1
                s["first_vendor_build"] += new_vendor == old_vendor
                s["first_vendor_free"] += new_free == old_free
                s["full_rank"] += full_new == full_old
                if new_vendor != old_vendor or new_free != old_free or full_new != full_old:
                    bad[(dev, dt)].append((jobz, n, b, row[0], "|".join(old), "|".join(new)))
    worst = 1.0
    for (dev, dt), s in sorted(stats.items()):
        p = s["points"]
        rates = {k: s[k] / p for k in ("first_vendor_build", "first_vendor_free", "full_rank")}
        worst = min(worst, *rates.values())
        print(f"{dev} {dt}: {p} points  first(vendor build) {100 * rates['first_vendor_build']:.2f}%  "
              f"first(vendor-free) {100 * rates['first_vendor_free']:.2f}%  full rank {100 * rates['full_rank']:.2f}%")
        for d in bad[(dev, dt)][:20]:
            print("   disagree", d)
    print(f"worst agreement {100 * worst:.2f}%")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2])
    else:
        cmp(sys.argv[2], sys.argv[3])
