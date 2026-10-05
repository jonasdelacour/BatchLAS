#!/usr/bin/env python3
"""gesvd data gate (docs/design/flat-select-p5/gesvd.md): the OLD router at random OFF-GRID points
against select's choice on the transcribed tables.

The old decision comes from the transcriber's `points` mode (the real RouteTable<Op::gesvd> code of
424a45bc); the new one from scripts/sweep_to_table.py's nearest() (the Table::nearest mirror) and a
can_run mirror of src/ops/gesvd/gesvd.cc. Both the vendor and the vendor-free build are replayed.

    python3 -P tools/transcribe/gesvd_gate.py REPO TRANSCRIBER [N] [DEVICE]
"""
import importlib.util
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter

REPO, GT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "sm_89"
_spec = importlib.util.spec_from_file_location("sweep_to_table", os.path.join(REPO, "scripts", "sweep_to_table.py"))
st = importlib.util.module_from_spec(_spec)
sys.modules["sweep_to_table"] = st
_spec.loader.exec_module(st)

GRID = {1, 2, 4, 8, 16, 24, 32, 33, 48, 64, 65, 128, 256, 512, 1024}  # choice.hh grid_mn
DTYPES = ("float", "double", "cfloat", "cdouble")
rng = random.Random(20261005)


def dim():
    if rng.random() < 0.3:
        return rng.randint(20, 80)  # dense around the 32/33 and 64/65 thresholds
    return max(1, int(round(2 ** rng.uniform(0, 11))))


pts = []
while len(pts) < N:
    herm = rng.choice(["N", "N", "N", "L", "U"])
    m = dim()
    n = m if herm != "N" and rng.random() < 0.9 else dim()
    if m in GRID and n in GRID:
        continue  # off-grid only: at least one coordinate between grid points
    pts.append((rng.choice(DTYPES), herm, rng.choice("NAT"), rng.choice("NAT"), m, n))

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.writelines("%s %s %s %s %d %d\n" % p for p in pts)
    path = f.name
out = subprocess.run([GT, "points", path], capture_output=True, text=True, check=True).stdout.splitlines()
os.unlink(path)

tables = {}
for dt in DTYPES:
    with open(os.path.join(REPO, "tuned", f"gesvd.{dt}.{DEVICE}.txt")) as f:
        tables[dt] = st.parse_table(f.read())


def canon(job, dim_, k):
    return "A" if job == "T" and k == dim_ else job  # enums.hh canonical_jobu / canonical_jobvh


def can_run(c, dt, herm, vec, m, n, vendor):
    """src/ops/gesvd/gesvd.cc can_run on a sub-group-32 GPU."""
    real, md = dt in ("float", "double"), max(m, n)
    if c == "vendor":
        return vendor
    if c == "jacobi":
        return herm == "N" and md <= (32 if dt == "cdouble" and vec != "none" else 64)
    if c == "cta":
        return md <= 32 and vec != "thin" and (m == n if herm != "N" else real)
    if c == "blocked":
        return (m == n and herm == "L") if herm != "N" else real
    raise ValueError(c)


LAST = ["blocked", "vendor", "jacobi", "cta"]  # choice.hh last_resort, then candidate order
stats = {v: Counter() for v in (True, False)}
bad = {v: Counter() for v in (True, False)}
regions = {v: Counter() for v in (True, False)}
for line in out:
    dt, herm, ju, jv, m, n, ranked = line.split()
    m, n = int(m), int(n)
    k = min(m, n)
    cu, cv = canon(ju, m, k), canon(jv, n, k)
    vec = "thin" if "T" in (cu, cv) else ("none" if cu == cv == "N" else "all")
    old = ranked.split("|")
    _, keyspec, rows = tables[dt]
    row = st.nearest(rows, (herm, vec, m, n), keyspec)
    for vendor in (True, False):
        old_c = old[0] if vendor else next((c for c in old if c != "vendor"), "none")
        new_c = next((c for c, _ in row[1] if can_run(c, dt, herm, vec, m, n, vendor)), None)
        if new_c is None:
            new_c = next((c for c in LAST if can_run(c, dt, herm, vec, m, n, vendor)), "none")
        stats[vendor][dt] += 1
        if old_c != new_c:
            bad[vendor][dt] += 1
            regions[vendor][f"{dt} herm={herm} vec={vec} max(m,n)~{max(m, n)}: old {old_c} new {new_c}"] += 1

worst = 100.0
for vendor in (True, False):
    print("vendor build" if vendor else "vendor-free build")
    for dt in DTYPES:
        tot, b = stats[vendor][dt], bad[vendor][dt]
        worst = min(worst, 100.0 * (tot - b) / tot)
        print(f"  {dt:8s} {tot - b}/{tot} agree ({100.0 * (tot - b) / tot:.2f}%)")
    for r, c in regions[vendor].most_common(25):
        print(f"   DISAGREE x{c} {r}")
sys.exit(0 if worst >= 99.0 else 1)
