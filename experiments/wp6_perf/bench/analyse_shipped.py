#!/usr/bin/env python3
"""WHAT THE SHIPPED VENDOR-FREE BUILD NOW DELIVERS, over the whole headline grid.

The per-nrhs tables report the two native TIERS separately, which is the right
frame for the routing question and the wrong one for "what does a vendor-free
caller get". A vendor-free build takes the FUSED tier wherever supports() admits
it (native_tier_preferred) and the COMPOSITION everywhere else, so the shipped
time at a cell is cta where the cta pin took and blocked where it fell through --
which is exactly what the route column says, cell by cell.

BEFORE is the same grid with every cell on the composition, i.e. the library as
WP6 shipped it.
"""
import sys

from analyse import geo, getrs_route, load, med


def main(tag="grid"):
    arms = {a: load("%s_%s.csv" % (tag, a)) for a in ("vendor", "blocked", "cta")}
    keys = list(arms["vendor"])

    rows, nfused, ncomp = [], 0, 0
    skipped = []
    for k in keys:
        rv, rb, rc = arms["vendor"].get(k), arms["blocked"].get(k), arms["cta"].get(k)
        if not (rv and rb and rc):
            skipped.append((k, "missing arm"))
            continue
        if rv["flag"] != "ok" or rb["flag"] != "ok" or rc["flag"] != "ok":
            skipped.append((k, "flag"))
            continue
        try:
            if max(float(rv["relsd"]), float(rb["relsd"]), float(rc["relsd"])) > 0.10:
                skipped.append((k, "relsd"))
                continue
        except ValueError:
            skipped.append((k, "relsd unreadable"))
            continue
        if getrs_route(rv) != "vendor:auto" or getrs_route(rb) != "native:blocked":
            skipped.append((k, "vendor/blocked pin did not take"))
            continue
        # THE SHIPPED ARM IS WHAT THE ROUTE COLUMN SAYS, not what the pin asked
        # for: where the cta pin fell through the capacity gate, the shipped
        # vendor-free route at that shape IS the composition.
        if getrs_route(rc) == "native:cta":
            after, which = med(rc), "fused"
            nfused += 1
        else:
            after, which = med(rb), "composition"
            ncomp += 1
        rows.append((k, med(rv), med(rb), after, which))

    print("== %s : %d cells usable, %d skipped" % (tag, len(rows), len(skipped)))
    for k, why in skipped:
        print("   skip %-8s n=%-5d nrhs=%-4d b=%-6d : %s" % (k[1], k[2], k[3], k[4], why))
    print("   shipped vendor-free route: %d cells FUSED, %d cells COMPOSITION" % (nfused, ncomp))
    print()
    print("%-8s %-6s %-6s %-7s %10s %10s %10s %-12s %8s %8s"
          % ("type", "n", "nrhs", "batch", "cuBLAS", "BEFORE", "AFTER", "arm", "b/cub", "a/cub"))
    for k, v, b, a, which in rows:
        print("%-8s %-6d %-6d %-7d %10.4f %10.4f %10.4f %-12s %8.3f %8.3f"
              % (k[1], k[2], k[3], k[4], v, b, a, which, v / b, v / a))
    print()
    gb = geo([v / b for _, v, b, _, _ in rows])
    ga = geo([v / a for _, v, _, a, _ in rows])
    wb = sum(1 for _, v, b, _, _ in rows if v / b > 1.0)
    wa = sum(1 for _, v, _, a, _ in rows if v / a > 1.0)
    sp = geo([b / a for _, _, b, a, _ in rows])
    print("WHOLE GRID, %d cells, cuBLAS_ms / native_ms:" % len(rows))
    print("   BEFORE (composition everywhere) geomean %.3f, %d wins" % (gb, wb))
    print("   AFTER  (shipped vendor-free)    geomean %.3f, %d wins" % (ga, wa))
    print("   speed-up of the op itself, BEFORE_ms / AFTER_ms: geomean %.3f" % sp)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grid")
