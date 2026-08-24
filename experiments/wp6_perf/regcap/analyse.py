#!/usr/bin/env python3
"""The register-cap A/B, scored across three passes per arm.

The two arms are two BUILDS of the .so, so they cannot be interleaved in one
process the way an ordinary A/B here is. The substitute evidence is the
CROSS-PASS MEDIAN SPREAD, printed for both arms on every cell: a cell whose three
passes agree to a fraction of a percent has a median that is a property of the
build and not of the session, whatever the within-pass relative sd says.
"""
import csv, os, statistics

D = os.path.dirname(os.path.abspath(__file__))

def load(tag):
    out = {}
    for p in (1, 2, 3):
        fn = os.path.join(D, "%s_p%d.csv" % (tag, p))
        for r in csv.reader(open(fn)):
            if len(r) < 12 or r[0] != "getrs":
                continue
            try:
                med = float(r[5])
            except ValueError:
                continue
            assert "native:cta" in r[11], "PIN DID NOT TAKE: %s" % r
            out.setdefault((r[1], int(r[2]), int(r[3]), int(r[4])), []).append(med)
    return out

b, a = load("before"), load("after")
print("%-8s %6s %5s %7s | %-24s | %-24s | %6s" %
      ("type", "n", "nrhs", "batch", "BEFORE med (3 passes)", "AFTER med (3 passes)", "b/a"))
gains = []
for k in sorted(b):
    if k not in a:
        continue
    bs, as_ = b[k], a[k]
    mb, ma = statistics.median(bs), statistics.median(as_)
    sb, sa = max(bs)/min(bs), max(as_)/min(as_)
    gains.append(mb/ma)
    print("%-8s %6d %5d %7d | %s (x%.3f) | %s (x%.3f) | %6.3f" %
          (k[0], k[1], k[2], k[3],
           " ".join("%7.4f" % x for x in bs), sb,
           " ".join("%7.4f" % x for x in as_), sa,
           mb/ma))
print()
print("cells %d   min %.3f   max %.3f   geomean %.4f" %
      (len(gains), min(gains), max(gains),
       __import__("math").exp(sum(__import__("math").log(g) for g in gains)/len(gains))))
