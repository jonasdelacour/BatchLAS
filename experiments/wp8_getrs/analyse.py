#!/usr/bin/env python3
# WP8-GETRS ladder analyser.
#
# THE DISCARD RULE is experiments/wp6_perf/bench/analyse.py's, verbatim in
# intent, plus the foreign-process column this directory's runner adds:
#   drop and NAME a cell when any arm's flag is not "ok", any arm's relsd > 0.10,
#   an arm is missing, the printed route disagrees with the arm, or any arm saw a
#   foreign compute process on the pinned device.
#
# READ BY POSITION, NOT BY DictReader. lubench6 prints 16 columns for
# getrf/getri and 15 for getrs under one 16-column header, so a DictReader
# returns flag=None on every getrs row. This runner appends `foreign` after
# lubench6's own output, so for a getrs row the layout is
#   0 op 1 type 2 n 3 nrhs 4 batch 5 med 6 mean 7 relsd 8 GFLOPs 9 resid
#   10 ws 11 route 12 extra 13 ntpiv 14 flag 15 foreign
# i.e. flag is the SECOND-TO-LAST field, not the last. That is checked, not
# assumed: a row whose field 14 is not ok/BAD is refused outright.
import sys
import math
from collections import OrderedDict

ROUTE_OK = {"nv": "native:blocked", "v": "vendor:auto"}


def load(path, arm):
    """(type,n,nrhs,batch) -> dict, with every refusal recorded."""
    rows, bad = OrderedDict(), []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 16 or p[0] != "getrs":
                bad.append((line.strip(), "malformed"))
                continue
            key = (p[1], int(p[2]), int(p[3]), int(p[4]))
            flag, foreign = p[14], p[15]
            if flag not in ("ok", "BAD"):
                bad.append((str(key), f"flag field is {flag!r}, not ok/BAD"))
                continue
            if flag != "ok":
                bad.append((str(key), "host-oracle flag BAD"))
                continue
            try:
                med, relsd = float(p[5]), float(p[7])
            except ValueError:
                bad.append((str(key), "unparseable timing"))
                continue
            if relsd > 0.10:
                bad.append((str(key), f"relsd {relsd:.3f} > 0.10"))
                continue
            if int(foreign) != 0:
                bad.append((str(key), f"{foreign} foreign compute processes"))
                continue
            got = p[11].split("|")[-1]
            if got != ROUTE_OK[arm]:
                bad.append((str(key), f"getrs route {got!r} != {ROUTE_OK[arm]!r}"))
                continue
            rows[key] = {"med": med, "route": got, "ws": int(p[10])}
    return rows, bad


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main(nv1, v1, nv2, v2):
    arms = {}
    allbad = []
    for name, path, arm in (("nv1", nv1, "nv"), ("v1", v1, "v"),
                            ("nv2", nv2, "nv"), ("v2", v2, "v")):
        r, b = load(path, arm)
        arms[name] = r
        allbad += [(name,) + x for x in b]

    keys = [k for k in arms["nv1"] if all(k in arms[a] for a in arms)]
    print(f"# paired cells: {len(keys)}   refused rows: {len(allbad)}")
    for x in allbad:
        print("#   REFUSED", x)

    print("type,n,nrhs,batch,nv_p1,v_p1,r_p1,nv_p2,v_p2,r_p2,quoted,spread_nv,spread_v")
    out = {}
    for k in keys:
        a1, b1 = arms["nv1"][k]["med"], arms["v1"][k]["med"]
        a2, b2 = arms["nv2"][k]["med"], arms["v2"][k]["med"]
        r1, r2 = b1 / a1, b2 / a2
        quoted = min(r1, r2)          # the conservative pass
        snv = max(a1, a2) / min(a1, a2)
        sv = max(b1, b2) / min(b1, b2)
        out[k] = quoted
        print(f"{k[0]},{k[1]},{k[2]},{k[3]},{a1:.4f},{b1:.4f},{r1:.4f},"
              f"{a2:.4f},{b2:.4f},{r2:.4f},{quoted:.4f},{snv:.4f},{sv:.4f}")

    # cross-pass spread summary
    sp = []
    for k in keys:
        for a, b in (("nv1", "nv2"), ("v1", "v2")):
            x, y = arms[a][k]["med"], arms[b][k]["med"]
            sp.append(max(x, y) / min(x, y))
    sp.sort()
    if sp:
        print(f"# cross-pass median spread {sp[len(sp)//2]:.4f}  worst {sp[-1]:.4f}  "
              f"above 1.10: {sum(1 for s in sp if s > 1.10)} of {len(sp)}")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:5])
