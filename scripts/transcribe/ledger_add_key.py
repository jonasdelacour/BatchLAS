#!/usr/bin/env python3
"""Give every cell of a ledger that lacks a new exact table key the value it was measured or
transcribed at, prefixed in the order given (the tuner's axis order), so that sweep_to_table.py
can read the ledger once choice.hh adds the key. Rewrites the file in place; run lines and every
other field are kept byte for byte. Used when sygst gained itype and uplo (both ledgers:
itype=1 uplo=L, the only values the earlier runs measured). Run from the repository root:
  scripts/transcribe/ledger_add_key.py LEDGER NAME=VALUE [NAME=VALUE ...]"""
import argparse
import json

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("ledger", help="benchmarks/results/tuning/ledger/<op>.<device>.jsonl")
ap.add_argument("fields", nargs="+", help="NAME=VALUE, in the order the new keys are prefixed")
args = ap.parse_args()
fields = [f.split("=", 1) for f in args.fields]

out, changed = [], 0
for line in open(args.ledger).read().split("\n"):
    if not line.strip():
        out.append(line)
        continue
    rec = json.loads(line)
    if rec.get("kind") != "cell":
        out.append(line)
        continue
    have = {kv.split("=", 1)[0] for kv in rec["key"].split(",")}
    add = ["%s=%s" % (n, v) for n, v in fields if n not in have]
    if add:
        rec["key"] = ",".join(add + [rec["key"]])
        line, changed = json.dumps(rec), changed + 1
    out.append(line)
with open(args.ledger, "w") as f:
    f.write("\n".join(out))
print(args.ledger, changed, "cells rekeyed")
