#!/usr/bin/env python3
"""Append a `transcribed` run to the sygst ledgers from the measured sm_89 float/cfloat rows of a table
generated from the ledger (no timing): double <- float and cdouble <- cfloat on sm_89; all four dtypes on
sm_120 from the matching sm_89 measured dtype; and, for sm_89 float/cfloat, the grid cells the deep run
skipped over the memory cap (copied from the nearest measured row). Keys: the choice.hh grid plus every
measured key. Ranking at a key: the measured row there, else the nearest (3 |dlog n| + |dlog batch|).
Wrote run transcribed-fd79f23c (tuned/README.md). Run from the repository root:
  scripts/transcribe/sygst_transcribe.py TABLE SHA DATE [--ledger-dir DIR]"""
import argparse
import json
import math
import os

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("table", help="a table generated from the ledger, e.g. tuned/sygst.sm_89.txt")
ap.add_argument("sha", help="the commit the run id names")
ap.add_argument("date", help="the run lines' date, YYYY-MM-DD")
ap.add_argument("--ledger-dir", default="benchmarks/results/tuning/ledger", help="ledgers to append to")
args = ap.parse_args()
table_path, sha, date = args.table, args.sha, args.date
grid_n = [1, 2, 4, 8, 16, 32, 33, 64, 128, 240, 256, 480, 512, 1024, 1200]
grid_b = [128, 512, 2048, 8192, 32768]
cands_all = "blocked:nb=32|blocked:nb=64|blocked:nb=128|blocked:nb=256|twosided|vendor"

measured = {}  # dtype -> {(n, b): [spelling...]}
for line in open(table_path).read().splitlines():
    if not line or line.startswith("#"):
        continue
    head, rest = line.split("|", 1)
    if rest.rsplit("#", 1)[1].strip() == "transcribed":
        continue
    kv = dict(f.split("=") for f in head.split())
    spell = [e.strip().split()[0] for e in rest.rsplit("#", 1)[0].split("|") if e.strip()]
    measured.setdefault(kv["dtype"], {})[(int(kv["n"]), int(kv["batch"]))] = spell
assert measured.get("float") and measured.get("cfloat"), "no measured float/cfloat rows"


def nearest(src, key):
    if key in src:
        return src[key]
    d = lambda k: 3 * abs(math.log(k[0] / key[0])) + abs(math.log(k[1] / key[1]))
    return src[min(src, key=d)]


plan = {"sm_89": {"float": "float", "cfloat": "cfloat", "double": "float", "cdouble": "cfloat"},
        "sm_120": {"float": "float", "double": "float", "cfloat": "cfloat", "cdouble": "cfloat"}}
rid = "transcribed-" + sha
for dev, dts in plan.items():
    out = []
    for dt, frm in dts.items():
        src = measured[frm]
        keys = sorted(set((n, b) for n in grid_n for b in grid_b) | set(src))
        if dev == "sm_89" and dt == frm:
            keys = [k for k in keys if k not in src]  # only the cells the measured run skipped
        why = "no timing: %s %s rows copy the nearest measured sm_89 %s row%s (docs/perf/sygvd.md#sygst-measured)" % (
            dev, dt, frm, "" if dt == frm else "; FP64 is not tuned on a 1:64 GPU")
        out.append({"kind": "run", "dtype": dt, "run_id": rid, "host": "", "device": dev, "device_name": "",
                    "batchlas": sha, "argv": "transcribed table tuned/sygst.%s.txt (%s)" % (dev, why), "date": date,
                    "tier": "transcribed", "keys": "n:log:3 batch:log", "candidates": cands_all})
        for key in keys:
            spell = nearest(src, key)
            rec = {"kind": "cell", "dtype": dt, "run_id": rid, "tier": "transcribed", "key": "n=%d,batch=%d" % key,
                   "round": 0, "date": date, "ranked": "|".join(spell), "cands": "|".join(spell)}
            for i in range(len(spell)):
                rec.update({"h%d" % i: "", "s%d" % i: "ok", "r%d" % i: "", "m%d" % i: None, "lo%d" % i: None,
                            "hi%d" % i: None, "n%d" % i: 0})
            out.append(rec)
    with open(os.path.join(args.ledger_dir, "sygst.%s.jsonl" % dev), "a") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(dev, len(out), "lines appended")
