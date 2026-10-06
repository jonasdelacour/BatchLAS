#!/usr/bin/env python3
"""symm transcription gate (temporary; docs/design/flat-select-l3/symm.md section 5).

  symm_gate.py --fidelity REPO TRANSCRIBER_SRC
      every verbatim range the transcriber names must appear in it byte for byte, as
      `git show ff340fc6:<file>` has it.
  symm_gate.py --gate REPO TRANSCRIBER_BIN [--points N] [--seed S]
      N random off-grid points per dtype through the old router (transcriber point mode)
      and through sweep_to_table.nearest + a can_run model on the written tables, in the
      vendor-present (A) and vendor-free (B) scenarios, plus two negative controls.
"""

import argparse
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter

SHA = "ff340fc6"
RANGES = [
    ("src/backends/symm_custom_dispatch.cc", 36, 72),
    ("src/backends/symm_custom_dispatch.cc", 137, 150),
    ("src/backends/symm_custom_dispatch.cc", 152, 191),
    ("src/backends/triangular_expand.hh", 45, 63),
    ("src/ops/level3/level3.cc", 47, 64),
]
GRID_MN = [1, 2, 4, 8, 16, 32, 64, 128, 255, 256, 512, 1024, 2048, 4096]
GRID_B = [1, 2, 3, 4, 8, 128, 1024, 8192, 32768]


def fidelity(repo, src):
    text = open(src).read()
    bad = 0
    for f, a, b in RANGES:
        old = subprocess.run(["git", "-C", repo, "show", f"{SHA}:{f}"], check=True, capture_output=True,
                             text=True).stdout.splitlines()
        block = "\n".join(old[a - 1:b])
        ok = block in text
        print(f"fidelity {f}:{a}-{b}: {'ok' if ok else 'DIFFERS'} ({b - a + 1} lines)")
        bad += not ok
    return bad


def form_of(a, b):
    return "sq" if 2 * min(a, b) >= max(a, b) else ("tall" if a > 2 * b else "wide")


def log_uniform(rng, lo, hi):
    return max(lo, min(hi, int(round(math.exp(rng.uniform(math.log(lo), math.log(hi + 0.5)))))))


def sample(rng, n):
    pts = []
    while len(pts) < n:
        m, nn, b = log_uniform(rng, 1, 4096), log_uniform(rng, 1, 4096), log_uniform(rng, 1, 32768)
        if m in GRID_MN and nn in GRID_MN and b in GRID_B:
            continue
        pts.append((m, nn, b, rng.choice("LR")))
    return pts


def sample_band(rng, n):
    """Off-grid points crowding the two thresholds (batch 3|4 at max(m,n) < 256; max(m,n) 255|256
    at batch < 4), where a uniform sample has little mass; the negative controls are read here."""
    pts = []
    while len(pts) < n:
        if rng.random() < 0.5:
            m, nn, b = log_uniform(rng, 1, 255), log_uniform(rng, 1, 255), rng.randint(2, 6)
        else:
            m, nn, b = log_uniform(rng, 128, 512), log_uniform(rng, 128, 512), rng.randint(1, 3)
        if m in GRID_MN and nn in GRID_MN and b in GRID_B:
            continue
        pts.append((m, nn, b, rng.choice("LR")))
    return pts


def new_choice(st, rows, keyspec, m, n, b, vendor):
    row = st.nearest(rows, (form_of(m, n), m, n, b), keyspec)
    for c, _ in row[1]:
        if c == "expand" or (c == "vendor" and vendor):
            return c
    # last resort {"expand", "vendor"}: expand runs every homogeneous GPU shape
    return "expand"


def agreement(st, rows, keyspec, olds, vendor):
    match = total = 0
    gains = Counter()
    misses = []
    for (dtype, m, n, b, side, vp, vf) in olds:
        new = new_choice(st, rows, keyspec, m, n, b, vendor)
        want = vp if vendor else vf
        if want == "throw":
            gains[new] += 1
            continue
        total += 1
        if new == want:
            match += 1
        else:
            misses.append((m, n, b, side, want, new))
    return match, total, gains, misses


def run_old(binary, dtype, pts):
    with tempfile.TemporaryDirectory() as tmp:
        pin, pout = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        with open(pin, "w") as f:
            for m, n, b, s in pts:
                f.write(f"{dtype} {m} {n} {b} {s}\n")
        subprocess.run([binary, "points", pin, pout], check=True)
        olds = []
        for line in open(pout):
            d, m, n, b, s, vp, vf = line.split()
            olds.append((d, int(m), int(n), int(b), s, vp, vf))
    assert len(olds) == len(pts)
    return olds


def gate(repo, binary, npoints, seed):
    # Loaded by path: the caller's directory may hold an inspect.py that shadows the stdlib.
    import importlib.util
    spec = importlib.util.spec_from_file_location("sweep_to_table", os.path.join(repo, "scripts", "sweep_to_table.py"))
    st = importlib.util.module_from_spec(spec)
    sys.modules["sweep_to_table"] = st
    spec.loader.exec_module(st)
    rng = random.Random(seed)
    worst = 100.0
    for dtype in ("float", "double"):
        for label, pts in (("uniform", sample(rng, npoints)), ("threshold-band", sample_band(rng, npoints))):
            olds = run_old(binary, dtype, pts)
            print(f"{dtype} {label}: {len(pts)} off-grid points; old vendor-present "
                  f"{dict(Counter(o[5] for o in olds))}, old vendor-free {dict(Counter(o[6] for o in olds))}")
            for dev in ("sm_89", "sm_120"):
                text = open(os.path.join(repo, "tuned", f"symm.{dtype}.{dev}.txt")).read()
                header, keyspec, rows = st.parse_table(text)
                for scen, vendor in (("A vendor-present", True), ("B vendor-free", False)):
                    match, total, gains, misses = agreement(st, rows, keyspec, olds, vendor)
                    pct = 100.0 * match / total if total else 100.0
                    worst = min(worst, pct)
                    extra = f"; old threw at {sum(gains.values())}, new takes {dict(gains)}" if gains else ""
                    print(f"  {dev} {scen}: {match}/{total} = {pct:.2f}%{extra}")
                    for mm in misses[:20]:
                        print(f"    disagree m={mm[0]} n={mm[1]} batch={mm[2]} side={mm[3]} old={mm[4]} new={mm[5]}")
                if dev == "sm_89" and dtype == "float":
                    for what, drop in (("batch=4 rows removed", lambda k: k[3] == 4),
                                       ("m=256 and n=256 rows removed", lambda k: k[1] == 256 or k[2] == 256)):
                        cut = [r for r in rows if not drop(r[0])]
                        match, total, _, _ = agreement(st, cut, keyspec, olds, True)
                        print(f"  negative control, A ({what}): {match}/{total} = {100.0 * match / total:.2f}%")
    print(f"worst agreement: {worst:.2f}%")
    return 0 if worst >= 99.0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", nargs=2, metavar=("REPO", "SRC"))
    ap.add_argument("--gate", nargs=2, metavar=("REPO", "BIN"))
    ap.add_argument("--points", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261006)
    a = ap.parse_args()
    if a.fidelity:
        sys.exit(fidelity(*a.fidelity))
    if a.gate:
        sys.exit(gate(a.gate[0], a.gate[1], a.points, a.seed))
    ap.print_help()


if __name__ == "__main__":
    main()
