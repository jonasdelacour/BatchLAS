#!/usr/bin/env python3
"""spmm data gate (docs/design/flat-select-p5/spmm.md): replay the OLD router at random
off-grid points against select's choice on the shipped tables.

Input: `spmm_transcribe --random N SEED` output (dtype,transA,transB,m,k,nrhs,batch,ranked: the
old preference with N/T/C spellings). For every point and device table, fold C -> T, take
Table::nearest (scripts/sweep_to_table.py's mirror), and walk the row under can_run's
assumptions for a valid CSR call: direct always runnable (both bodies compiled; capacities in
can_run, none for spmm), vendor runnable when present. Compared with the old first choice in
the vendor-present build and the vendor-free build. Exit 1 if agreement < 99% anywhere.
"""

import collections
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import sweep_to_table as stt  # noqa: E402


def first(ranked, vendor):
    return next(c for c in ranked if vendor or c != "vendor")


def main(points_csv, devices=("sm_89", "sm_120", "cpu")):
    tables = {}
    with open(points_csv, newline="") as f:
        pts = list(csv.DictReader(f))
    tally = collections.Counter()
    bad = collections.Counter()
    for p in pts:
        old = p["ranked"].split("|")
        key = ("N" if p["transA"] == "N" else "T", "N" if p["transB"] == "N" else "T",
               int(p["m"]), int(p["nrhs"]), int(p["batch"]))
        for dev in devices:
            ident = (p["dtype"], dev)
            if ident not in tables:
                with open(stt.table_path("spmm", p["dtype"], dev)) as f:
                    _, keyspec, rows = stt.parse_table(f.read())
                tables[ident] = (keyspec, rows)
            keyspec, rows = tables[ident]
            row = stt.nearest(rows, key, keyspec)
            new = [c for c, _ in row[1]]
            for vendor in (True, False):
                cell = (p["dtype"], dev, "vendor" if vendor else "vendor-free")
                tally[cell] += 1
                if first(new, vendor) != first(old, vendor):
                    bad[cell] += 1
                    bad[cell + (p["transA"] + p["transB"],)] += 1
    worst = 1.0
    for cell in sorted(tally):
        rate = 1.0 - bad[cell] / tally[cell]
        worst = min(worst, rate)
        print(f"{cell[0]:8s} {cell[1]:7s} {cell[2]:12s} {tally[cell]:5d} points  agree {rate * 100:7.3f}%")
    for cell, n in sorted(bad.items()):
        if len(cell) == 4:
            print(f"  disagree: {cell} x{n}")
    print(f"points {len(pts)}, worst agreement {worst * 100:.3f}%")
    return 0 if worst >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
