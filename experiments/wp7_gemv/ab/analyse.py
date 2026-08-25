#!/usr/bin/env python3
"""Render the WP7 native-vs-vendor gemv A/B as a table, and apply the lead's
acceptance gate.

THE GATE (B6):
  * every measured cell records native/vendor; target >= 0.85x
  * any cell below 0.50x is a BLOCKER -- fix it, or state it in the report as a
    known weakness with its mechanism
  * preferred() may ship a clause only where a >= 1.15x MEDIAN win reproduces
    across TWO independent passes

RATIO CONVENTION: vendor_ms / native_ms, so 1.00 is parity and > 1 means the
native kernel is faster. Reported for the DEFAULT native route -- the one
RouteTable<Op::gemv,T> would actually pick in a vendor-free build:
  transA = N        -> native:direct   (CTA has no NoTrans body)
  transA = T or C   -> native:cta      (Direct is the no-sub-group fallback)
The Direct arm on a transposed shape is reported too, in its own column,
because it is what a native_cpu queue and any device without an enumerated
sub-group size of 32 will run.
"""
import csv, sys, collections

paths = sys.argv[1:] or ["ab_p1.csv"]


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("type") in (None, "FAILED"):
                continue
            key = (r["type"], int(r["m"]), int(r["n"]), int(r["batch"]), r["transA"])
            rows[(key, r["arm"])] = r
    return rows


passes = [load(p) for p in paths]
keys = sorted(set(k for p in passes for (k, _) in p), key=lambda k: (k[4], k[0], k[1], k[2]))

print(f"passes: {len(passes)}  ({', '.join(paths)})")
print()
hdr = (f"{'transA':>6} {'type':>8} {'m':>5} {'n':>5} {'batch':>6} "
       f"{'vendor ms':>10} {'nat ms':>10} {'route':>14} {'ratio':>7} "
       f"{'direct ms':>10} {'d-ratio':>8} {'relerr':>9}")
print(hdr)
print("-" * len(hdr))

worst = None
blockers = []
wins = []
for k in keys:
    ty, m, n, b, tr = k
    default_arm = "native:direct" if tr == "N" else "native:cta"

    def med(arm, p):
        r = p.get((k, arm))
        return float(r["median_ms"]) if r else None

    def route(arm, p):
        r = p.get((k, arm))
        return r["route"] if r else "-"

    vs = [med("vendor", p) for p in passes if med("vendor", p)]
    ns = [med(default_arm, p) for p in passes if med(default_arm, p)]
    ds = [med("native:direct", p) for p in passes if med("native:direct", p)]
    if not vs or not ns:
        continue
    v = sorted(vs)[len(vs) // 2]
    nn = sorted(ns)[len(ns) // 2]
    d = sorted(ds)[len(ds) // 2] if ds else float("nan")
    ratio = v / nn
    dratio = v / d if d == d else float("nan")
    rerr = max(float(p[(k, default_arm)]["relerr"]) for p in passes if (k, default_arm) in p)
    rt = route(default_arm, passes[0])
    print(f"{tr:>6} {ty:>8} {m:>5} {n:>5} {b:>6} "
          f"{v:>10.4f} {nn:>10.4f} {rt:>14} {ratio:>7.2f} "
          f"{d:>10.4f} {dratio:>8.2f} {rerr:>9.1e}")
    if worst is None or ratio < worst[0]:
        worst = (ratio, k)
    if ratio < 0.50:
        blockers.append((ratio, k))
    if ratio >= 1.15:
        wins.append((ratio, k))

print()
print(f"worst default-route cell: {worst[0]:.2f}x at {worst[1]}")
print(f"cells below 0.50x (BLOCKERS): {len(blockers)}")
for r, k in sorted(blockers):
    print(f"    {r:.2f}x  {k}")
print(f"cells at or above 1.15x (preferred() candidates): {len(wins)}")
for r, k in sorted(wins, reverse=True):
    print(f"    {r:.2f}x  {k}")

# Cross-pass reproducibility of the ratio itself.
if len(passes) > 1:
    print()
    print("cross-pass ratio spread (max/min over passes), default route:")
    worst_spread = 0.0
    for k in keys:
        tr = k[4]
        arm = "native:direct" if tr == "N" else "native:cta"
        rs = []
        for p in passes:
            if (k, "vendor") in p and (k, arm) in p:
                rs.append(float(p[(k, "vendor")]["median_ms"]) /
                          float(p[(k, arm)]["median_ms"]))
        if len(rs) > 1:
            sp = max(rs) / min(rs)
            worst_spread = max(worst_spread, sp)
    print(f"    worst spread over all cells: {worst_spread:.3f}")
