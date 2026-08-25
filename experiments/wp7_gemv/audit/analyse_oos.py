#!/usr/bin/env python3
"""Score the surviving preferred() candidates on cells they were NOT fitted on."""
import csv, sys

def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        if r["route"] == "FAILED":
            continue
        d[(r["transA"], int(r["m"]), int(r["n"]), int(r["batch"]), r["arm"])] = (
            float(r["median_ms"]), float(r["GBs"]), int(r["MB"]))
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
cells = sorted({k[:4] for k in a})

P = {
    "P1  m in [64,320] and n >= 256":
        lambda tr, m, n, bt, mb: 64 <= m <= 320 and n >= 256,
    "P2  m in [64,320] and n*batch >= 131072":
        lambda tr, m, n, bt, mb: 64 <= m <= 320 and n * bt >= 131072,
    "P3  m in [64,320] and A >= 512 MB":
        lambda tr, m, n, bt, mb: 64 <= m <= 320 and mb >= 512,
}

print("%-2s %5s %6s %6s %7s %11s %11s %8s %8s   %s" % (
    "tr", "m", "n", "batch", "MB", "vendor GB/s", "native GB/s", "p1", "p2",
    "  ".join(k.split()[0] for k in P)))
res = []
for k in cells:
    tr, m, n, bt = k
    va, na = a.get(k + ("vendor",)), a.get(k + ("native:cta",))
    vb, nb = b.get(k + ("vendor",)), b.get(k + ("native:cta",))
    if not (va and na and vb and nb):
        continue
    r1, r2 = va[0] / na[0], vb[0] / nb[0]
    mb = va[2]
    flags = [p(tr, m, n, bt, mb) for p in P.values()]
    print("%-2s %5d %6d %6d %7d %11.1f %11.1f %8.2f %8.2f   %s" % (
        tr, m, n, bt, mb, va[1], na[1], r1, r2,
        "   ".join("Y" if f else "." for f in flags)))
    res.append((k, r1, r2, mb, flags))

print()
print("=" * 92)
print("OUT-OF-SAMPLE VERDICT   (a candidate is REFUTED by admitting a cell below 1.00x)")
print("=" * 92)
for i, name in enumerate(P):
    adm = [r for r in res if r[4][i]]
    if not adm:
        print("%-44s admits 0 of %d cells" % (name, len(res)))
        continue
    worst = min(min(r1, r2) for _, r1, r2, _, _ in adm)
    nwin = sum(1 for _, r1, r2, _, _ in adm if min(r1, r2) >= 1.15)
    nloss = sum(1 for _, r1, r2, _, _ in adm if min(r1, r2) < 1.00)
    verdict = "REFUTED" if nloss else ("HOLDS" if nwin else "HOLDS but wins nothing")
    print("%-44s admits %2d   >=1.15x %2d   BELOW 1.00x %2d   worst %.2f   %s" % (
        name, len(adm), nwin, nloss, worst, verdict))
    for k, r1, r2, mb, _ in sorted(adm, key=lambda t: min(t[1], t[2])):
        if min(r1, r2) < 1.00:
            print("      ADMITS A LOSER: tr=%s m=%d n=%d batch=%d %dMB -> %.2f / %.2f"
                  % (k[0], k[1], k[2], k[3], mb, r1, r2))
