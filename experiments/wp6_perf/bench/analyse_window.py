#!/usr/bin/env python3
"""SCORE A CANDIDATE preferred() PREDICATE AGAINST EVERY CELL THIS DIRECTORY
MEASURED, rather than against the table it was read off.

This exists because a window read off one summary table and then justified by
that same table is not evidence -- the campaign's recorded failure mode is a
one-cell window that reproduces only where it was fitted. Every getrs cell from
every sweep here (grid, nrhs, w8, flat, flat2, flat3, flat4) is pooled, deduplicated on
(type, n, nrhs, batch), and each candidate predicate is asked two questions:

  INSIDE  the window -- how many cells, how many are LOSSES, and what is the
          worst one? A window with losses inside it moves traffic onto a slower
          route in every build.
  OUTSIDE the window -- how many WINS were left to the vendor, and how large?
          That is the price of the window, and it has to be stated, not hidden.

A cell counts once. Where two sweeps measured the same cell the ratios are
averaged and the SPREAD between them is reported at the top as the cross-pass
reproduction check -- which is the honest answer to a relative sd that a
heavy-tailed rep distribution inflates.
"""
import glob
import math
import os
import sys
from collections import defaultdict

from analyse import ARMROUTE, geo, getrs_route, load, med

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = ["grid", "nrhs", "w8", "flat", "flat2", "flat3", "flat4"]


def pool():
    """(type, n, nrhs, batch) -> list of cuBLAS_ms / cta_ms, one per sweep."""
    out = defaultdict(list)
    for tag in TAGS:
        v = os.path.join(HERE, "%s_vendor.csv" % tag)
        c = os.path.join(HERE, "%s_cta.csv" % tag)
        if not (os.path.exists(v) and os.path.exists(c)):
            continue
        av, ac = load(v), load(c)
        for k, rv in av.items():
            rc = ac.get(k)
            if rc is None:
                continue
            if rv["flag"] != "ok" or rc["flag"] != "ok":
                continue
            try:
                if float(rv["relsd"]) > 0.10 or float(rc["relsd"]) > 0.10:
                    continue
            except ValueError:
                continue
            if getrs_route(rv) != "vendor:auto" or getrs_route(rc) != "native:cta":
                continue
            a, b = med(rv), med(rc)
            if a and b:
                out[(k[1], k[2], k[3], k[4])].append(a / b)
    return out


CANDIDATES = [
    ("C1  nrhs <= 1, every type, every order",
     lambda t, n, r, b: r <= 1),
    ("C2  nrhs <= 2, every type, every order",
     lambda t, n, r, b: r <= 2),
    ("C3  C2 + (float and nrhs <= 4)",
     lambda t, n, r, b: r <= 2 or (t == "float" and r <= 4)),
    ("C4  C3 + (float and nrhs <= 8)",
     lambda t, n, r, b: r <= 2 or (t == "float" and r <= 8)),
    ("C5  C3 + (nrhs <= 4 and n >= 128)",
     lambda t, n, r, b: r <= 2 or (t == "float" and r <= 4) or (r <= 4 and n >= 128)),
    ("C6  nrhs <= 4, every type, every order",
     lambda t, n, r, b: r <= 4),
    ("C7  nrhs <= 8, every type, every order (i.e. the whole capability)",
     lambda t, n, r, b: r <= 8),
    # C5's three losses are all n = 2048 at batch 4-8, and the mechanism is not
    # the order: the fused tier is ONE WORK-GROUP PER MATRIX, so the CTA count IS
    # the batch, and batch 4 on 128 SMs occupies 4 of them. These two candidates
    # guard that starvation the two available ways -- by the CTA count directly,
    # and by an upper order bound that only correlates with it.
    ("C8  C5 but the nrhs 3..4 clause also requires batch >= 16 (CTA count)",
     lambda t, n, r, b: r <= 2 or (t == "float" and r <= 4) or (r <= 4 and n >= 128 and b >= 16)),
    ("C9  C5 but the nrhs 3..4 clause is bounded above at n <= 1024",
     lambda t, n, r, b: r <= 2 or (t == "float" and r <= 4) or (r <= 4 and 128 <= n <= 1024)),
]


def main():
    data = pool()
    print("== pooled: distinct (type, n, nrhs, batch) cells where BOTH arms are ok,")
    print("   relsd <= 10 pct, and BOTH routes verified from the printed route column")
    print("   (%d cells)" % len(data))
    dup = [(k, vs) for k, vs in data.items() if len(vs) > 1]
    print("   %d cells measured by more than one sweep; cross-pass spread max/min:" % len(dup))
    worst = sorted(dup, key=lambda kv: max(kv[1]) / min(kv[1]))[-6:]
    for k, vs in worst:
        print("      %-8s n=%-5d nrhs=%-4d b=%-6d  %s   spread %.3f"
              % (k[0], k[1], k[2], k[3], " ".join("%.3f" % x for x in vs),
                 max(vs) / min(vs)))
    print()

    flat = {k: sum(vs) / len(vs) for k, vs in data.items()}

    for name, pred in CANDIDATES:
        ins = [(k, x) for k, x in flat.items() if pred(*k)]
        out = [(k, x) for k, x in flat.items() if not pred(*k)]
        losses = sorted([kv for kv in ins if kv[1] < 1.0], key=lambda kv: kv[1])
        left = sorted([kv for kv in out if kv[1] > 1.0], key=lambda kv: -kv[1])
        print("%s" % name)
        print("   INSIDE : %3d cells, geomean %6.3f, min %6.3f, %d LOSSES"
              % (len(ins), geo([x for _, x in ins]), min(x for _, x in ins) if ins else float("nan"),
                 len(losses)))
        for k, x in losses[:8]:
            print("        loss %-8s n=%-5d nrhs=%-4d b=%-6d  %.3f" % (k[0], k[1], k[2], k[3], x))
        if len(losses) > 8:
            print("        ... and %d more" % (len(losses) - 8))
        print("   OUTSIDE: %3d cells, %d of them are WINS handed to the vendor, best %s"
              % (len(out), len(left), ("%.3f" % left[0][1]) if left else "-"))
        for k, x in left[:5]:
            print("        left  %-8s n=%-5d nrhs=%-4d b=%-6d  %.3f" % (k[0], k[1], k[2], k[3], x))
        print()


if __name__ == "__main__":
    main()
