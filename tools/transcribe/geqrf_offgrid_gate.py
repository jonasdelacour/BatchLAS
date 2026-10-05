#!/usr/bin/env python3
"""The geqrf data gate (docs/design/flat-select-p5/geqrf.md): replay random OFF-GRID points
through the transcribed tables and compare with the OLD router.

Input: the CSV of `geqrf_transcribe --offgrid COUNT SEED dtype:tiny:cta_m:cta_elems...`, whose rows
carry the old Auto choice (vendor present and vendor-free) at the given capacities. For each point
this script takes the nearest row of tuned/geqrf.<dtype>.<device>.txt with the converter's
nearest() (the same rule as select's Table::nearest) and walks it with a can_run model at the
same capacities: tiny needs m == n <= tiny_max_n, cta m <= cta_max_m and m*n <= cta_max_elems,
blocked m >= n >= 1, vendor only when present. Then the last resort (blocked, vendor).

    python3 tools/transcribe/geqrf_offgrid_gate.py points.csv [--device sm_89 sm_120]
"""
import argparse
import csv
import importlib.util
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("s2t", os.path.join(REPO, "scripts", "sweep_to_table.py"))
s2t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2t)


def key_of(m, n):
    form = "sq" if m == n else ("tall" if m > n else "wide")
    lo, hi = min(m, n), max(m, n)
    return (form, n, 1 if lo < 1 else hi // lo)


def can_run(c, m, n, caps, vendor):
    if c == "vendor":
        return vendor
    if m < n or n < 1:
        return False
    tiny, cta_m, cta_e = caps
    if c == "tiny":
        return m == n and n <= tiny
    if c == "cta":
        return m <= cta_m and m * n <= cta_e
    return c == "blocked"


def new_choice(row, m, n, caps, vendor):
    for c, _ in row:
        if can_run(c, m, n, caps, vendor):
            return c
    for c in ("blocked", "vendor"):
        if can_run(c, m, n, caps, vendor):
            return c
    return "none"


def bucket(m, n):
    form, kn, a = key_of(m, n)
    return f"{form} n~{1 << (kn.bit_length() - 1)} aspect~{1 << (a.bit_length() - 1)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("--device", nargs="+", default=["sm_89", "sm_120"])
    ap.add_argument("--tuned-dir", default=os.path.join(REPO, "tuned"), help="tables to replay (default tuned/)")
    args = ap.parse_args()
    tables = {}
    for dev in args.device:
        for dt in ("float", "double", "cfloat", "cdouble"):
            with open(os.path.join(args.tuned_dir, f"geqrf.{dt}.{dev}.txt")) as f:
                _, keyspec, rows = s2t.parse_table(f.read())
            tables[(dt, dev)] = (keyspec, rows, {k for k, _ in rows})
    total, agree, ongrid = Counter(), Counter(), Counter()
    regions = defaultdict(Counter)
    with open(args.points) as f:
        for r in csv.DictReader(f):
            dt, m, n = r["dtype"], int(r["m"]), int(r["n"])
            caps = (int(r["tiny_max_n"]), int(r["cta_max_m"]), int(r["cta_max_elems"]))
            for dev in args.device:
                keyspec, rows, keys = tables[(dt, dev)]
                k = key_of(m, n)
                if k in keys:
                    ongrid[(dt, dev)] += 1
                    continue
                hit = s2t.nearest(rows, k, keyspec)
                for vendor, old in ((True, r["old_vendor"]), (False, r["old_vendor_free"])):
                    old = old if (vendor or old != "vendor") else "none"
                    got = new_choice(hit[1], m, n, caps, vendor)
                    tag = (dt, dev, "vendor" if vendor else "vendor-free")
                    total[tag] += 1
                    if got == old:
                        agree[tag] += 1
                    else:
                        regions[tag][f"{bucket(m, n)}: old {old} -> new {got} (e.g. {m}x{n}, row {hit[0]})"] += 1
    worst = 1.0
    for tag in sorted(total):
        rate = agree[tag] / total[tag]
        worst = min(worst, rate)
        print(f"{tag[0]:8s} {tag[1]:7s} {tag[2]:12s} off-grid {total[tag]:6d}  agree {agree[tag]:6d}  {100 * rate:7.3f}%"
              f"  (on-grid skipped {ongrid[(tag[0], tag[1])]})")
        for what, cnt in regions[tag].most_common(8):
            print(f"    {cnt:5d}  {what}")
    print(f"worst agreement {100 * worst:.3f}%")
    return 0 if worst >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
