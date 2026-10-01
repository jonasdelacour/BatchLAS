#!/usr/bin/env python3
"""Check that the LIBRARY's potrf route choice is the fitted model's choice, cell by cell.

    python3 evaluation/routing/check_model_routes.py --profile sm_120 \
        --results benchmarks/results/routing/sm120_potrf_sweep*.jsonl [--device] [--windows]

For every measured cell it compares:
  * fit.py's full-model pick (Python `choose` over the committed profile JSON) against
    potrf_plan_dump's `auto_model` (RouteTable + the generated header, priced in C++);
  * with --device, the library's real potrf_route on GPU 0 (`device_route`) against the same
    pick when the profile's gate passed, or against today's windows (`auto`) when it did not
    -- run with BATCHLAS_ROUTING_PROFILE=sm_89 for the "byte-identical to main" check;
  * the cells where the model moved a decision against the report's route-diff table.
Exit 1 on any mismatch.
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit  # noqa: E402


def model_of(profile):
    m = {}
    for route, by_d in profile["routes"].items():
        for d, by_u in by_d.items():
            for u, e in by_u.items():
                m[(route, d, u)] = {"constants": [e["constants"].get(k, 0.0) for k in fit.PARAMS],
                                    "support": e["support"]}
    return m


def report_diffs(md_path):
    """{(dtype, uplo, n, batch): (today, model)} from the report's per-cell route-diff table."""
    out = {}
    if not os.path.exists(md_path):
        return out
    pat = re.compile(r"^\| (\w+) ([LU]) n=(\d+) b=(\d+) \| ([\w:]+) \| ([\w:]+) \|")
    in_table = False
    for line in open(md_path):
        if line.startswith("Per cell:"):
            in_table = True
        elif in_table:
            m = pat.match(line)
            if m:
                out[(m[1], m[2], int(m[3]), int(m[4]))] = (m[5], m[6])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--plan-dump", default=os.path.join(fit.REPO, "build/tests/potrf_plan_dump"))
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--json", help="the profile JSON (default profiles/<profile>.json)")
    args = ap.parse_args()

    pj = args.json or os.path.join(os.path.dirname(fit.FACTS_FILE), args.profile + ".json")
    with open(pj) as f:
        profile = json.load(f)
    enabled = profile["gate"]["verdict"] == "PASS"
    facts = fit.load_facts(args.profile)
    model = model_of(profile)
    cells, _ = fit.load_rows(args.results, "potrf", False)
    shapes = sorted({(d, u, n, b) for (d, u, n, b, _r) in cells})

    cmd = [args.plan_dump, "--local-mem", str(facts["local_mem"]), "--max-wg", str(facts["max_wg"]),
           "--cus", str(facts["cus"]), "--max-threads-per-cu", str(facts["max_threads_per_cu"]),
           "--max-groups-per-cu", str(facts["max_groups_per_cu"]), "--profile", args.profile]
    for dtype, r in sorted(facts["regs"].items()):
        cmd += ["--regs", dtype + "=" + ",".join(f"{k}:{v}" for k, v in sorted(r.items()))]
    if args.device:
        cmd.append("--device")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/dpcpp-cuda/lib:" + env.get("LD_LIBRARY_PATH", "")
    stdin = "".join(f"{d} {n} {b} {u}\n" for (d, u, n, b) in shapes)
    res = subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env, check=True)
    feats = {}
    for line in res.stdout.splitlines():
        j = json.loads(line)
        feats[(j["dtype"], j["uplo"], j["n"], j["batch"])] = j

    bad = []
    moved = {}
    for s in shapes:
        f = feats[s]
        py, _ = fit.choose(model, f, profile["margin"])
        py = py or "vendor"
        if enabled and f["auto_model"] != py:
            bad.append(f"auto_model {s}: C++ {f['auto_model']} != fit.py {py}")
        expect = py if enabled else f["auto"]
        if args.device and f["device_route"] != expect:
            bad.append(f"device_route {s}: library {f['device_route']} != expected {expect}")
        got = f["device_route"] if args.device else f["auto_model"]
        if got != f["auto"]:
            moved[s] = (f["auto"], got)

    md = report_diffs(os.path.splitext(pj)[0] + ".md")
    measured = {(d, u, n, b) for (d, u, n, b, r) in cells}
    if enabled:
        for s, pair in md.items():
            if s in measured and moved.get(s) != pair:
                bad.append(f"report route-diff {s} {pair} but library moved {moved.get(s)}")
        for s, pair in moved.items():
            if s not in md:
                bad.append(f"library moved {s} {pair}, not in the report's route-diff")
    print(json.dumps({"profile": args.profile, "model_enabled": enabled, "cells": len(shapes),
                      "moved_vs_windows": len(moved), "report_route_diff": len(md),
                      "device": args.device, "mismatches": len(bad)}))
    for b in bad[:40]:
        print("  MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
