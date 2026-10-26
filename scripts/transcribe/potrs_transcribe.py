#!/usr/bin/env python3
"""Append `transcribed` potrs cells: the best current sm_89 float/cfloat record's ranking at each measured key,
copied to sm_89 double (from float) and cdouble (from cfloat), and to all four sm_120 dtypes (float, double from
float; cfloat, cdouble from cfloat). Untimed (m = null), the tuned/README.md convention for FP64 on a 1:64 part
and for a device without a measurement. Wrote run `transcribed-136b6464`.

usage: scripts/transcribe/potrs_transcribe.py SHA DATE [--repo DIR] [--ledger-dir DIR]
  --repo        the source tree whose spec and kernel hashes select the records (default: this checkout)
  --ledger-dir  where potrs.<device>.jsonl are read and appended (default: <repo>/benchmarks/results/tuning/ledger)
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("sha")
ap.add_argument("date")
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ap.add_argument("--ledger-dir")
args = ap.parse_args()
repo, sha, date = args.repo, args.sha, args.date
sys.path.insert(0, f"{repo}/scripts")
import sweep_to_table as S  # noqa: E402

op = "potrs"
led_dir = args.ledger_dir or f"{repo}/benchmarks/results/tuning/ledger"
src_path = f"{led_dir}/{op}.sm_89.jsonl"
block = S.parse_kernel_block(S.read_spec_source(op, repo))
cands = ["fused", "trsm"]
fam = S.family_hashes(block, cands, repo)
led = S.read_ledger(src_path)
runs = {r["dtype"]: r for r in led.runs if r.get("tier") != "transcribed"}
best = {}
for dt in ("float", "cfloat"):
    cells = [c for c in led.cells if c["dtype"] == dt and c["tier"] not in S.UNHASHED]
    best[dt] = S.best_records(cells, fam)
    print(f"{dt}: {len(best[dt])} measured keys", file=sys.stderr)

plan = {"sm_89": {"double": "float", "cdouble": "cfloat"},
        "sm_120": {"float": "float", "double": "float", "cfloat": "cfloat", "cdouble": "cfloat"}}
run_id = f"transcribed-{sha}"
for dev, m in plan.items():
    lines = []
    for dt, src in m.items():
        srun = runs[src]
        lines.append({"kind": "run", "dtype": dt, "run_id": run_id, "host": "", "device": dev, "device_name": "",
                      "batchlas": sha, "argv": f"transcribed: sm_89 {src} best records of run {srun['run_id']} (no timing)",
                      "date": date, "tier": "transcribed", "keys": srun["keys"], "candidates": srun["candidates"]})
        for key, c in sorted(best[src].items(), key=lambda kv: kv[0]):
            if not c["ranked"]:
                continue
            ranked = c["ranked"] + [x for x in cands if x not in c["ranked"]]
            rec = {"kind": "cell", "dtype": dt, "run_id": run_id, "tier": "transcribed",
                   "key": ",".join(f"{k}={v}" for k, v in key), "round": 0, "date": date,
                   "ranked": "|".join(ranked), "cands": "|".join(cands)}
            for i, _ in enumerate(cands):
                rec.update({f"h{i}": "", f"s{i}": "ok", f"r{i}": "", f"m{i}": None, f"lo{i}": None, f"hi{i}": None, f"n{i}": 0})
            lines.append(rec)
    path = f"{led_dir}/{op}.{dev}.jsonl"
    with open(path, "a") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    print(f"{path}: +{len(lines)} lines", file=sys.stderr)
