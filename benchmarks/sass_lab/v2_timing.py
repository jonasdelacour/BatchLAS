#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarise sass_lab timing CSV rows for the v2 parity set (ops named "<op>@<WG>").

  v2_timing.py TIMING_CSV [TIMING_CSV ...] [--md]
With several CSVs (repeated passes) the per-pass medians are combined by their median.
Columns: hand median us, lean/hand, v2/hand, v2/lean, v2_scaled/hand (nrm2), max rel err.
"""
import csv
import statistics
import sys

NAMES = {"0": "hand", "3": "lean", "9": "v2", "10": "v2_scaled"}


def main():
    md = "--md" in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    data = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            if "@" not in r["op"] or r["variant"] not in NAMES:
                continue
            key = (r["op"], r["type"], int(r["n"]))
            v = data.setdefault(key, {}).setdefault(NAMES[r["variant"]], {"us": [], "err": 0.0})
            v["us"].append(float(r["median_us"]))
            v["err"] = max(v["err"], float(r["max_rel_err"]))
    head = ["op@WG", "type", "n", "hand_us", "lean/hand", "v2/hand", "v2/lean", "v2_scaled/hand", "max_rel_err"]
    print(("| " + " | ".join(head) + " |\n|" + "---|" * len(head)) if md else ",".join(head))
    ratios = []
    for key in sorted(data):
        g = {k: statistics.median(v["us"]) for k, v in data[key].items()}
        h = g.get("hand")
        if not h:
            continue
        f = lambda a, b: f"{g[a] / g[b]:.3f}" if a in g and b in g and g[b] > 0 else "-"
        if "v2" in g and "lean" in g:
            ratios.append(g["v2"] / g["lean"])
        err = max(v["err"] for v in data[key].values())
        vals = [key[0], key[1], str(key[2]), f"{h:.1f}", f("lean", "hand"), f("v2", "hand"), f("v2", "lean"),
                f("v2_scaled", "hand"), f"{err:.1e}"]
        print(("| " + " | ".join(vals) + " |") if md else ",".join(vals))
    if ratios:
        print(f"\nv2/lean over {len(ratios)} rows: min {min(ratios):.3f} median {statistics.median(ratios):.3f} "
              f"max {max(ratios):.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
