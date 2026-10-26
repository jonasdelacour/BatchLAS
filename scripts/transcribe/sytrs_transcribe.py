#!/usr/bin/env python3
"""Append transcribed sytrs cells to a ledger: every row of a measured table, its ranking kept.

Usage (from the repo root):
  scripts/transcribe/sytrs_transcribe.py TABLE LEDGER DEVICE RUN_ID SHA DATE SRC_DTYPE:DST_DTYPE[,...] ARGV

Each SRC:DST pair copies TABLE's dtype=SRC rows into LEDGER as dtype=DST cells of tier
`transcribed` (untimed). Run lines for every DST come first, then the cells, in pair order.
The shipped rows of run `transcribed-sytrs-d67ad754` (tuned/README.md) came from:
  sm_89:  TABLE=tuned/sytrs.sm_89.txt  pairs float:double,cfloat:cdouble
  sm_120: TABLE=tuned/sytrs.sm_89.txt  pairs float:float,cfloat:cfloat,float:double,cfloat:cdouble
"""
import json, sys

if len(sys.argv) != 9:
    sys.exit(__doc__)
table, ledger, device, run_id, sha, date, maps, argv = sys.argv[1:9]
pairs = [m.split(":") for m in maps.split(",")]
rows = []
for line in open(table):
    if line.startswith("#") or not line.strip():
        continue
    head, rest = line.split("|", 1)
    ranked = [e.split()[0] for e in rest.split("#")[0].split("|") if e.strip()]
    kv = dict(f.split("=") for f in head.split())
    rows.append((kv, ranked))
cands_all = "cta|blocked|wg|vendor"
out = []
for src, dst in pairs:
    out.append({"kind": "run", "dtype": dst, "run_id": run_id, "host": "", "device": device, "device_name": "",
                "batchlas": sha, "argv": argv, "date": date, "tier": "transcribed",
                "keys": "n:log:2 nrhs:log batch:log", "candidates": cands_all})
for src, dst in pairs:
    for kv, ranked in rows:
        if kv["dtype"] != src:
            continue
        c = {"kind": "cell", "dtype": dst, "run_id": run_id, "tier": "transcribed",
             "key": "n=%s,nrhs=%s,batch=%s" % (kv["n"], kv["nrhs"], kv["batch"]), "round": 0, "date": date,
             "ranked": "|".join(ranked), "cands": "|".join(ranked)}
        for i in range(len(ranked)):
            c.update({"h%d" % i: "", "s%d" % i: "ok", "r%d" % i: "", "m%d" % i: None, "lo%d" % i: None,
                      "hi%d" % i: None, "n%d" % i: 0})
        out.append(c)
with open(ledger, "a") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print("appended", len(out), "lines to", ledger)
