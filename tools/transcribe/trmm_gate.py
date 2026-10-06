#!/usr/bin/env python3
"""trmm transcription gates (temporary; docs/design/flat-select-l3/trmm.md).

  --fidelity             every VERBATIM block of trmm_transcribe.cc equals its ff340fc6 source lines
  --data TR_V TR_VF      off-grid data gate: >= 2000 random points per dtype through the old router
                         (the transcriber's point mode, vendor and vendor-free builds) and through
                         sweep_to_table.nearest + a can_run model on the written tables; plus the
                         negative controls (threshold rows deleted)
"""
import argparse
import math
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import sweep_to_table as stt  # noqa: E402

SHA = "ff340fc6"
DTYPES = ("float", "double", "cfloat", "cdouble")
GRID_ORDER = (1, 16, 32, 64, 128, 256, 512, 1024, 2048)
GRID_Q = (1, 16, 128, 1024, 4096)
GRID_BATCH = (1, 128, 1024, 32768)


def fidelity():
    src = open(os.path.join(HERE, "trmm_transcribe.cc")).read().splitlines()
    blocks, bad, i = 0, 0, 0
    while i < len(src):
        line = src[i]
        if line.startswith("// VERBATIM "):
            spec = line.split()[2]
            path, rng = spec.split(":")
            a, b = (int(x) for x in rng.split("-"))
            j = src.index("// END VERBATIM", i)
            body = src[i + 1:j]
            orig = subprocess.run(["git", "-C", REPO, "show", f"{SHA}:{path}"], capture_output=True, text=True,
                                  check=True).stdout.splitlines()[a - 1:b]
            blocks += 1
            if body != orig:
                bad += 1
                print(f"MISMATCH {spec}")
            i = j
        i += 1
    print(f"fidelity: {blocks} verbatim blocks, {bad} mismatches")
    return bad == 0 and blocks > 0


def points(rng, n):
    out = []
    while len(out) < n:
        side = rng.choice("LR")
        o = max(1, min(4096, round(2 ** rng.uniform(0, 12))))
        q = max(1, min(4096, round(2 ** rng.uniform(0, 12))))
        b = max(1, min(32768, round(2 ** rng.uniform(0, 15))))
        if o in GRID_ORDER and q in GRID_Q and b in GRID_BATCH:
            continue  # off-grid only
        out.append((side, o, q, b))
    return out


def old_choices(binary, dtype, pts):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for s, o, q, b in pts:
            f.write(f"{dtype} {s} {o} {q} {b}\n")
        path = f.name
    out = subprocess.run([binary, "points", path], capture_output=True, text=True, check=True).stdout.split("\n")
    os.unlink(path)
    return [line.split()[-1] for line in out if line.strip()]


def can_run(choice, side, batch, vendor):
    if choice == "triangular":
        return side == "L" and batch <= 65535
    if choice == "expand":
        return batch <= 65535
    return vendor


def new_choice(rows, keyspec, pt, vendor):
    side, o, q, b = pt
    row = stt.nearest(rows, (side, o, q, b), keyspec)
    for c, _ in row[1]:
        if can_run(c, side, b, vendor):
            return c
    for c in ("expand", "triangular", "vendor"):  # choice.hh last_resort
        if can_run(c, side, b, vendor):
            return c
    return "throw"


def gate(rows, keyspec, pts, old_v, old_vf):
    agree_a = sum(new_choice(rows, keyspec, p, True) == o for p, o in zip(pts, old_v))
    served = [(p, o) for p, o in zip(pts, old_vf) if o != "throw"]
    agree_b = sum(new_choice(rows, keyspec, p, False) == o for p, o in served)
    gains = {}
    for p, o in zip(pts, old_vf):
        if o == "throw":
            c = new_choice(rows, keyspec, p, False)
            gains[c] = gains.get(c, 0) + 1
    return agree_a, agree_b, len(served), gains


def data(tr_v, tr_vf, n):
    ok = True
    rng = random.Random(20261006)
    for dev in ("sm_89", "sm_120"):
        for dt in DTYPES:
            header, keyspec, rows = stt.parse_table(open(stt.table_path("trmm", dt, dev)).read())
            pts = points(rng, n)
            old_v, old_vf = old_choices(tr_v, dt, pts), old_choices(tr_vf, dt, pts)
            a, b, served, gains = gate(rows, keyspec, pts, old_v, old_vf)
            pa, pb = 100.0 * a / len(pts), (100.0 * b / served if served else 100.0)
            ok &= pa >= 99.0 and pb >= 99.0
            print(f"{dev} {dt}: {len(pts)} pts; A vendor-present {a}/{len(pts)} = {pa:.2f}%; "
                  f"B vendor-free served {b}/{served} = {pb:.2f}%; old-threw -> new {gains}")
            if dev == "sm_89":
                for drop in ("L", "R"):
                    kept = [r for r in rows if r[0][0] != drop]
                    na, _, _, _ = gate(kept, keyspec, pts, old_v, old_vf)
                    print(f"   negative control, side={drop} rows deleted: A {na}/{len(pts)} = "
                          f"{100.0 * na / len(pts):.2f}%")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--data", nargs=2, metavar=("TR_V", "TR_VF"))
    ap.add_argument("--points", type=int, default=2500)
    a = ap.parse_args()
    ok = True
    if a.fidelity:
        ok &= fidelity()
    if a.data:
        ok &= data(a.data[0], a.data[1], a.points)
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
