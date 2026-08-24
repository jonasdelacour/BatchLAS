#!/usr/bin/env python3
"""The A/B reader for this directory. THREE arms, and every rule it applies is
stated here rather than in prose downstream.

THE DISCARD RULE, fixed before it was applied (wp6_lu/bench/README.md section 10):
a cell is dropped and NAMED when

  * any arm is flagged BAD (the harness's in-process host-oracle verdict), or
  * any arm's relative sd exceeds 10 %, or
  * an arm is missing for that cell, or
  * THE PIN DID NOT TAKE -- the getrs half of the printed route is not the arm's
    own route. This last one is not a formality: an unsupported forced route
    falls through to automatic() (route_resolve.hh:165 then :175), so a
    `native:cta` pin above the fused tier's capacity ceiling silently measures the
    COMPOSITION. Those rows are named as PIN-FELL-THROUGH, which for the cta arm
    is the fused tier's CAPACITY ceiling doing its job and not a bad run.

Ratios are always BASE / OTHER, i.e. greater than 1 means the arm named second
is FASTER.
"""
import csv
import math
import sys
from collections import OrderedDict

ARMROUTE = {"vendor": "vendor:auto", "blocked": "native:blocked", "cta": "native:cta"}


# THE ROW IS READ BY POSITION AND THE FLAG IS THE LAST FIELD, NOT A NAMED COLUMN.
# lubench6.cpp prints a DIFFERENT NUMBER OF COLUMNS PER OP -- getrf 16, getri 16,
# getrs 15 -- under one 16-column header, so csv.DictReader silently hands back
# flag=None for every getrs row and the pass/fail verdict lands in the column
# called `extra2`. The first version of this reader used DictReader and dropped
# all 168 cells with "flag=None", which is the good failure; the bad one is a
# reader that treats None as "not BAD" and quotes a geomean over unchecked rows.
COLS = ["op", "type", "n", "nrhs", "batch", "med_ms", "mean_ms", "relsd",
        "GFLOPs", "resid", "ws", "route"]


def load(path):
    rows = OrderedDict()
    with open(path) as f:
        for raw in csv.reader(f):
            if not raw or raw[0] == "op" or len(raw) < len(COLS) + 1:
                continue
            r = dict(zip(COLS, raw[:len(COLS)]))
            r["flag"] = raw[-1]
            try:
                key = (r["op"], r["type"], int(r["n"]), int(r["nrhs"]), int(r["batch"]))
            except ValueError:
                continue
            rows[key] = r
    return rows


def med(r):
    try:
        return float(r["med_ms"])
    except (TypeError, ValueError):
        return None


def getrs_route(r):
    return (r["route"] or "").split("|")[-1]


def getrf_route(r):
    return (r["route"] or "").split("|")[0]


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def triage(names, arms, keys):
    kept, dropped = [], []
    for k in keys:
        why = []
        for a in names:
            r = arms[a].get(k)
            if r is None:
                why.append("%s MISSING" % a)
                continue
            if r["flag"] != "ok":
                why.append("%s flag=%s" % (a, r["flag"]))
            try:
                if float(r["relsd"]) > 0.10:
                    why.append("%s relsd=%.3f" % (a, float(r["relsd"])))
            except (TypeError, ValueError):
                why.append("%s relsd unreadable" % a)
            if k[0] == "getrs" and a in ARMROUTE:
                want, got = ARMROUTE[a], getrs_route(r)
                if got != want:
                    why.append("%s PIN-FELL-THROUGH to %s" % (a, got))
        if why:
            dropped.append((k, why))
        else:
            kept.append((k, why))
    return kept, dropped


def main(argv):
    tag = argv[1]
    names = argv[2:]
    arms = OrderedDict((a, load("%s_%s.csv" % (tag, a))) for a in names)
    keys = list(arms[names[0]])
    kept, dropped = triage(names, arms, keys)

    print("== %s : %d cells, %d kept, %d dropped" % (tag, len(keys), len(kept), len(dropped)))
    for k, why in dropped:
        print("   DROP %-6s %-8s n=%-5d nrhs=%-4d b=%-6d : %s"
              % (k[0], k[1], k[2], k[3], k[4], "; ".join(why)))
    print()

    for i, base in enumerate(names):
        for other in names[i + 1:]:
            rs = []
            print("-- %s to %s   (ratio = %s_ms / %s_ms; above 1 means %s FASTER)"
                  % (base, other, base, other, other))
            print("%-8s %-6s %-6s %-7s %10s %10s %8s"
                  % ("type", "n", "nrhs", "batch", base, other, "ratio"))
            for k, _ in kept:
                a, b = med(arms[base][k]), med(arms[other][k])
                if not a or not b:
                    continue
                rs.append(a / b)
                print("%-8s %-6d %-6d %-7d %10.4f %10.4f %8.3f"
                      % (k[1], k[2], k[3], k[4], a, b, a / b))
            if rs:
                wins = sum(1 for x in rs if x > 1.0)
                print("   GEOMEAN %.3f over %d cells, %d wins, min %.3f, max %.3f\n"
                      % (geo(rs), len(rs), wins, min(rs), max(rs)))
            else:
                print("   no cells\n")


if __name__ == "__main__":
    main(sys.argv)
