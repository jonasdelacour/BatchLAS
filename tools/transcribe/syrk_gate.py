#!/usr/bin/env python3
"""TEMPORARY (deleted at integration): syrk's transcription gates.

  --fidelity          every VERBATIM block of syrk_transcribe.cc equals its ff340fc6 lines
  --data BIN          off-grid data gate: random points through the old rule (BIN points mode)
                      and through sweep_to_table.nearest + a can_run model on tuned/syrk.*
  --drop-n N          negative control: delete the rows at n=N before the lookup

Python 3 standard library only; run from anywhere.
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHA = "ff340fc6"
sys.path.insert(0, os.path.join(REPO, "scripts"))
import sweep_to_table as stt  # noqa: E402

GRID_N = [1, 2, 4, 8, 16, 32, 64, 128, 129, 256, 257, 384, 385, 512, 513, 640, 641, 768, 769, 896, 897, 1024,
          1025, 1152, 1153, 1536, 1537, 2176, 2177, 4096]
GRID_K = [1, 7, 8, 64, 512, 4096]
GRID_B = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 16, 26, 27, 128, 1024, 32768]
CANDIDATES = {"float": ["gram", "triangular", "vendor"], "double": ["gram", "vendor"]}
LAST_RESORT = ["triangular", "gram", "vendor"]


def fidelity():
    src = open(os.path.join(REPO, "tools/transcribe/syrk_transcribe.cc")).read().split("\n")
    blocks, bad = 0, 0
    i = 0
    while i < len(src):
        m = re.match(r"// VERBATIM (\S+) (\d+) (\d+)$", src[i])
        if not m:
            i += 1
            continue
        end = src.index("// END VERBATIM", i)
        body = src[i + 1:end]
        old = subprocess.run(["git", "-C", REPO, "show", f"{SHA}:{m.group(1)}"], capture_output=True, text=True,
                             check=True).stdout.split("\n")
        want = old[int(m.group(2)) - 1:int(m.group(3))]
        blocks += 1
        if body != want:
            bad += 1
            print(f"FIDELITY FAIL: {m.group(1)}:{m.group(2)}-{m.group(3)}")
        i = end + 1
    print(f"fidelity: {blocks} blocks, {bad} differ")
    return bad == 0 and blocks > 0


def form_of(a, b):
    return "sq" if 2 * min(a, b) >= max(a, b) else ("tall" if a > 2 * b else "wide")


def can_run(c, dtype, n, vendor):
    if c not in CANDIDATES[dtype]:
        return False
    if c == "gram":
        return n <= 128
    if c == "triangular":
        return True
    return vendor


def new_choice(rows, keyspec, dtype, n, k, b, vendor):
    row = stt.nearest(rows, (form_of(n, k), n, k, b), keyspec)
    for c, _ in row[1]:
        if can_run(c, dtype, n, vendor):
            return c
    for c in LAST_RESORT:
        if can_run(c, dtype, n, vendor):
            return c
    return "throw"


def loguni(rng, hi):
    return max(1, min(hi, int(round(2 ** rng.uniform(0, math.log2(hi))))))


def data(binary, count, seed, drop_n):
    rng = random.Random(seed)
    ok_all = True
    for dtype in ("float", "double"):
        pts = []
        while len(pts) < count:
            n, k, b = loguni(rng, 4096), loguni(rng, 4096), loguni(rng, 32768)
            if n in GRID_N and k in GRID_K and b in GRID_B:
                continue
            pts.append((n, k, b))
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("".join(f"{dtype} {n} {k} {b}\n" for n, k, b in pts))
            path = f.name
        out = subprocess.run([binary, "points", path], capture_output=True, text=True, check=True).stdout.split("\n")
        os.unlink(path)
        old = [ln.split() for ln in out if ln.strip()]
        assert len(old) == len(pts)
        for dev in ("sm_89", "sm_120"):
            _, keyspec, rows = stt.parse_table(open(stt.table_path("syrk", dtype, dev)).read())
            if drop_n:
                rows = [r for r in rows if r[0][1] != drop_n]
            agree_a, served, agree_b, gains, mism = 0, 0, 0, Counter(), []
            for (n, k, b), (ovp, ovf) in zip(pts, old):
                nvp = new_choice(rows, keyspec, dtype, n, k, b, True)
                nvf = new_choice(rows, keyspec, dtype, n, k, b, False)
                if nvp == ovp:
                    agree_a += 1
                else:
                    mism.append(f"A {dtype} n={n} k={k} batch={b}: old {ovp} new {nvp}")
                if ovf != "throw":
                    served += 1
                    if nvf == ovf:
                        agree_b += 1
                    else:
                        mism.append(f"B {dtype} n={n} k={k} batch={b}: old {ovf} new {nvf}")
                else:
                    gains[nvf] += 1
            pa = 100.0 * agree_a / len(pts)
            pb = 100.0 * agree_b / served if served else 100.0
            ok = pa >= 99.0 and pb >= 99.0
            ok_all &= ok
            print(f"{dtype} {dev}: A (vendor-present) {agree_a}/{len(pts)} = {pa:.2f}%; "
                  f"B (vendor-free, old served) {agree_b}/{served} = {pb:.2f}%; "
                  f"old threw {sum(gains.values())}: new {dict(gains)} {'PASS' if ok else 'FAIL'}")
            for m in mism[:10]:
                print("   ", m)
            if len(mism) > 10:
                print(f"    ... {len(mism) - 10} more")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--data", metavar="BIN")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--drop-n", type=int, default=0)
    a = ap.parse_args()
    ok = True
    if a.fidelity:
        ok &= fidelity()
    if a.data:
        ok &= data(a.data, a.count, a.seed, a.drop_n)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
