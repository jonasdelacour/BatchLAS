#!/usr/bin/env python3
"""Render the complex<double> transposed refinement grid as vendor/native ratios.

Prints one grid per input file, then the CROSS-PASS grid (per-cell median over
the passes given) and the exact rectangle a preferred() clause could claim.
"""
import csv, sys

paths = sys.argv[1:] or ["refine_c_p1.csv", "refine_c_p2.csv"]


def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("type") in (None, "FAILED"):
                continue
            d[(int(r["m"]), int(r["n"]), r["arm"])] = (
                float(r["median_ms"]), float(r["GBs"]), float(r["relerr"]))
    return d


passes = [load(p) for p in paths]
ms = sorted({k[0] for p in passes for k in p})
ns = sorted({k[1] for p in passes for k in p})


def ratio(m, n, p):
    v = p.get((m, n, "vendor"))
    c = p.get((m, n, "native:cta"))
    if not v or not c:
        return None
    return v[0] / c[0]


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


for title, sel in [("cross-pass median", None)] + [(p, i) for i, p in enumerate(paths)]:
    print(f"=== vendor_ms / native_ms  ({title}) ===")
    print("   m\\n " + "".join(f"{n:>8}" for n in ns))
    for m in ms:
        row = f"{m:>6} "
        for n in ns:
            rs = [ratio(m, n, p) for p in (passes if sel is None else [passes[sel]])]
            rs = [r for r in rs if r]
            row += f"{med(rs):>8.2f}" if rs else f"{'-':>8}"
        print(row)
    print()

maxerr = max(v[2] for p in passes for v in p.values())
print(f"max relerr over every row of every pass: {maxerr:.1e}")
print()
print("=== the rectangle: cells where EVERY pass is >= 1.15x ===")
win = []
for m in ms:
    for n in ns:
        rs = [ratio(m, n, p) for p in passes]
        if all(r and r >= 1.15 for r in rs):
            win.append((m, n, med([r for r in rs])))
if win:
    print(f"  m in [{min(w[0] for w in win)}, {max(w[0] for w in win)}]"
          f"  n in [{min(w[1] for w in win)}, {max(w[1] for w in win)}]"
          f"  ({len(win)} cells, {min(w[2] for w in win):.2f}x .. {max(w[2] for w in win):.2f}x)")
else:
    print("  none")
print()
print("=== cells INSIDE that rectangle that do NOT win (the cost of the clause) ===")
if win:
    m0, m1 = min(w[0] for w in win), max(w[0] for w in win)
    n0, n1 = min(w[1] for w in win), max(w[1] for w in win)
    bad = []
    for m in ms:
        for n in ns:
            if m0 <= m <= m1 and n0 <= n <= n1:
                rs = [ratio(m, n, p) for p in passes]
                rs = [r for r in rs if r]
                if rs and med(rs) < 1.15:
                    bad.append((m, n, med(rs)))
    for m, n, r in bad:
        print(f"  m={m:>5} n={n:>5}  {r:.2f}x")
    if not bad:
        print("  none -- every cell in the rectangle wins")
    print()
    print(f"worst cell inside the rectangle: "
          f"{min([b[2] for b in bad], default=min(w[2] for w in win)):.2f}x")
