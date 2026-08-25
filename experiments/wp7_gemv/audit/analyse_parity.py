#!/usr/bin/env python3
"""WP7 AUDIT -- render the parity gate and apply the lead's acceptance rule.

Ratio convention: vendor_ms / native_ms, so 1.00 is parity and > 1 means the
native kernel is faster.

The "default" arm is the route a VENDOR-FREE build actually picks:
  NoTrans -> native:direct   (there is no NoTrans CTA body)
  Trans / ConjTrans -> native:cta, falling back to native:direct on a device
  with no enumerated sub-group size 32.
That is the arm the gate is applied to. native:direct on a transposed shape is
reported separately because it is what a CPU device runs, not what a GPU picks.

Usage: analyse_parity.py p1.csv [p2.csv]
"""
import csv, sys, collections

def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["route"] == "FAILED" or not r["median_ms"]:
                continue
            key = (r["type"], r["transA"], int(r["out_len"]), int(r["red_len"]),
                   int(r["batch"]), r["arm"])
            d[key] = (float(r["median_ms"]), float(r["GBs"]), r["route"],
                      float(r["relerr"]))
    return d

def main():
    passes = [load(p) for p in sys.argv[1:]]
    p1 = passes[0]

    # Route-column audit: a pinned arm whose resolved route is not the pin means
    # the row measured something other than what it is labelled.
    bad = [(k, v[2]) for k, v in p1.items() if not v[2].startswith(k[5].split(":")[0])]
    print("route-column audit: %d rows, %d disagree with the pin" % (len(p1), len(bad)))
    for k, rt in bad[:10]:
        print("   MISMATCH", k, "->", rt)
    nz = [k for k, v in p1.items() if v[3] != 0.0]
    print("correctness audit : %d rows with relerr != 0" % len(nz))
    for k in nz[:10]:
        print("   RELERR", k, p1[k][3])
    print()

    cells = sorted({k[:5] for k in p1})
    hdr = ("type", "transA", "out", "red", "batch", "MB", "vendor GB/s",
           "native GB/s", "route", "ratio p1", "ratio p2")
    print("%-8s %-2s %6s %6s %6s %8s %11s %11s %-14s %9s %9s" % hdr)
    rows = []
    for c in cells:
        ty, tr, ol, rl, b = c
        wid = {"float": 4, "double": 8, "cfloat": 8, "cdouble": 16}[ty]
        mb = ol * rl * b * wid / 1048576.0
        default = "native:direct" if tr == "N" else "native:cta"
        ratios = []
        for p in passes:
            v = p.get(c + ("vendor",))
            nn = p.get(c + (default,))
            ratios.append(v[0] / nn[0] if v and nn else None)
        v1 = p1.get(c + ("vendor",))
        n1 = p1.get(c + (default,))
        if not (v1 and n1):
            continue
        print("%-8s %-2s %6d %6d %6d %8.1f %11.1f %11.1f %-14s %9s %9s" % (
            ty, tr, ol, rl, b, mb, v1[1], n1[1], n1[2],
            "%.2f" % ratios[0],
            "%.2f" % ratios[1] if len(ratios) > 1 and ratios[1] else "-"))
        rows.append((c, mb, ratios, v1, n1))

    print()
    print("=" * 78)
    print("ACCEPTANCE GATE (B6), applied to the DEFAULT native route")
    print("=" * 78)
    def worst(rs):
        return min(r for r in rs if r is not None)
    allr = [worst(r[2]) for r in rows]
    print("cells: %d   >=0.85x: %d   in [0.50,0.85): %d   BELOW 0.50x: %d" % (
        len(allr), sum(1 for r in allr if r >= 0.85),
        sum(1 for r in allr if 0.50 <= r < 0.85),
        sum(1 for r in allr if r < 0.50)))
    print()
    print("--- BLOCKERS: every cell below 0.50x on its worst pass ---")
    for c, mb, rs, v1, n1 in sorted(rows, key=lambda t: worst(t[2])):
        if worst(rs) >= 0.50:
            continue
        print("  %-8s %-2s out=%-5d red=%-5d batch=%-5d %8.1f MB  vendor %8.1f  native %8.1f  ratio %s" % (
            c[0], c[1], c[2], c[3], c[4], mb, v1[1], n1[1],
            " / ".join("%.2f" % r for r in rs if r is not None)))
    print()
    print("--- SUB-0.85x (target misses that are not blockers) ---")
    for c, mb, rs, v1, n1 in sorted(rows, key=lambda t: worst(t[2])):
        if not (0.50 <= worst(rs) < 0.85):
            continue
        print("  %-8s %-2s out=%-5d red=%-5d batch=%-5d %8.1f MB  vendor %8.1f  native %8.1f  ratio %s" % (
            c[0], c[1], c[2], c[3], c[4], mb, v1[1], n1[1],
            " / ".join("%.2f" % r for r in rs if r is not None)))
    print()
    print("--- NATIVE WINS >= 1.15x on the worst pass ---")
    for c, mb, rs, v1, n1 in sorted(rows, key=lambda t: -worst(t[2])):
        if worst(rs) < 1.15:
            continue
        print("  %-8s %-2s out=%-5d red=%-5d batch=%-5d %8.1f MB  vendor %8.1f  native %8.1f  ratio %s" % (
            c[0], c[1], c[2], c[3], c[4], mb, v1[1], n1[1],
            " / ".join("%.2f" % r for r in rs if r is not None)))

    # The portable arm: Direct on a transposed shape is what a native_cpu queue
    # and any device without sub-group 32 must run. Reported, never gated.
    print()
    print("--- PORTABLE ARM: native:direct on transposed shapes (vs vendor, p1) ---")
    for c in cells:
        if c[1] == "N":
            continue
        v = p1.get(c + ("vendor",)); d = p1.get(c + ("native:direct",))
        if not (v and d):
            continue
        print("  %-8s %-2s out=%-5d red=%-5d batch=%-5d  vendor %8.1f  direct %8.1f  ratio %.2f" % (
            c[0], c[1], c[2], c[3], c[4], v[1], d[1], v[0] / d[0]))

main()
