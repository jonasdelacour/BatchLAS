#!/usr/bin/env python3
"""syr2k transcription gates (temporary; docs/design/flat-select-l3/syr2k.md).

  syr2k_gate.py --fidelity TRANSCRIBER_CC
      every BEGIN/END VERBATIM block equals the named lines of ff340fc6, byte for byte.
  syr2k_gate.py --data VP_BIN VF_BIN [--points N] [--seed S] [--drop KEY=VALUE]
      off-grid data gate: the old router (transcriber point mode, vendor present and vendor-free)
      against nearest() on tuned/syr2k.*.txt plus a can_run model. --drop removes the rows with
      that key value first (the negative control). Exit 1 below 99%.
"""
import argparse
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter

REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                      cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or os.getcwd()
sys.path.insert(0, os.path.join(REPO, "scripts"))
import sweep_to_table as stt  # noqa: E402

SHA = "ff340fc6"
GRID_N = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
GRID_K = {1, 8, 64, 512, 4096}
GRID_B = {1, 2, 3, 128, 1024, 8192, 32768}
CANDIDATES = {"float": ["triangular", "vendor"], "double": ["vendor"]}
LAST_RESORT = ["triangular", "vendor"]


def fidelity(path):
    text = open(path).read()
    blocks = re.findall(r"// ---- BEGIN VERBATIM (\S+):(\d+)-(\d+) ----\n(.*?)// ---- END VERBATIM ----\n",
                        text, re.S)
    if len(blocks) != 3:
        print(f"expected 3 verbatim blocks, found {len(blocks)}")
        return 1
    bad = 0
    for f, a, b, body in blocks:
        src = subprocess.run(["git", "-C", REPO, "show", f"{SHA}:{f}"], capture_output=True, text=True,
                             check=True).stdout.splitlines(keepends=True)
        want = "".join(src[int(a) - 1:int(b)])
        ok = want == body
        bad += not ok
        print(f"{f}:{a}-{b}: {'identical' if ok else 'DIFFERS'} ({len(body)} bytes)")
    return 1 if bad else 0


def can_run(choice, dtype, vendor, n, k, batch, trans):
    if choice not in CANDIDATES[dtype]:
        return False
    if choice == "vendor":
        return vendor
    t = (n + 127) // 128
    return trans != "C" and n >= 1 and k >= 1 and 1 <= batch <= 65535 and t * (t + 1) // 2 <= 65535


def new_choice(rows, keyspec, dtype, vendor, n, k, batch, trans):
    row = stt.nearest(rows, (n, k, batch), keyspec)
    for c, _ in row[1]:
        if can_run(c, dtype, vendor, n, k, batch, trans):
            return c
    for c in LAST_RESORT + CANDIDATES[dtype]:
        if can_run(c, dtype, vendor, n, k, batch, trans):
            return c
    return "throw"


def sample(rng, count):
    pts = set()
    while len(pts) < count:
        n = max(1, round(math.exp(rng.uniform(0, math.log(4096)))))
        k = max(1, round(math.exp(rng.uniform(0, math.log(4096)))))
        b = max(1, round(math.exp(rng.uniform(0, math.log(32768)))))
        if n in GRID_N and k in GRID_K and b in GRID_B:
            continue
        pts.add((n, k, b, rng.choice("NT")))
    return sorted(pts)


def old_outcomes(binary, dtype, pts):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for n, k, b, t in pts:
            f.write(f"{dtype} {n} {k} {b} {t}\n")
        path = f.name
    out = subprocess.run([binary, "points", path], capture_output=True, text=True, check=True).stdout.split("\n")
    os.unlink(path)
    res = {}
    for line in out:
        p = line.split()
        if len(p) == 7:
            res[(int(p[1]), int(p[2]), int(p[3]), p[4])] = p[6]
    return res


def data(vp, vf, count, seed, drop):
    rng = random.Random(seed)
    worst = 100.0
    for dtype in ("float", "double"):
        pts = sample(rng, count)
        old_vp, old_vf = old_outcomes(vp, dtype, pts), old_outcomes(vf, dtype, pts)
        for dev in ("sm_89", "sm_120"):
            header, keyspec, rows = stt.parse_table(open(stt.table_path("syr2k", dtype, dev)).read())
            if drop:
                name, value = drop.split("=")
                i = [kname for kname, _, _ in keyspec].index(name)
                rows = [r for r in rows if r[0][i] != int(value)]
            a_ok = a_bad = 0
            b_ok = b_bad = 0
            gains = Counter()
            misses = []
            for p in pts:
                n, k, b, t = p
                nv = new_choice(rows, keyspec, dtype, True, n, k, b, t)
                if nv == old_vp[p]:
                    a_ok += 1
                else:
                    a_bad += 1
                    misses.append(("A", p, old_vp[p], nv))
                nf = new_choice(rows, keyspec, dtype, False, n, k, b, t)
                if old_vf[p] == "throw":
                    gains[nf] += 1
                elif nf == old_vf[p]:
                    b_ok += 1
                else:
                    b_bad += 1
                    misses.append(("B", p, old_vf[p], nf))
            pa = 100.0 * a_ok / (a_ok + a_bad)
            pb = 100.0 * b_ok / (b_ok + b_bad) if b_ok + b_bad else 100.0
            worst = min(worst, pa, pb)
            print(f"{dtype:6} {dev:6} points={len(pts)} A(vendor) {a_ok}/{a_ok + a_bad} = {pa:.2f}%  "
                  f"B(vendor-free, old served) {b_ok}/{b_ok + b_bad} = {pb:.2f}%  "
                  f"B(old threw -> new) {dict(gains)}")
            for m in misses[:8]:
                print("   disagree", m)
    print(f"worst {worst:.2f}% -> {'PASS' if worst >= 99.0 else 'FAIL'}")
    return 0 if worst >= 99.0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity")
    ap.add_argument("--data", nargs=2, metavar=("VP_BIN", "VF_BIN"))
    ap.add_argument("--points", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--drop")
    a = ap.parse_args()
    if a.fidelity:
        return fidelity(a.fidelity)
    if a.data:
        return data(a.data[0], a.data[1], a.points, a.seed, a.drop)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
