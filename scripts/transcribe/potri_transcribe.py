#!/usr/bin/env python3
"""Append a transcribed potri run to the sm_89 and sm_120 ledgers (no timing).

usage: scripts/transcribe/potri_transcribe.py SHA MODE [--date YYYY-MM-DD] [--ledger-dir DIR]
  mode provisional: every dtype and device ranks cta first where it fits the device, else blocked
                    (wrote run transcribed-c6aa0c2b).
  mode measured:    double from sm_89 float's measured winners, cdouble from cfloat's; sm_120 every
                    dtype from sm_89 (float -> float, double, cfloat -> cfloat, cdouble), filtered to
                    the families the dtype compiles and can run (wrote transcribed-f517330f and -gap).
  --date            the run and cell date (default: today)
  --ledger-dir      where potri.<device>.jsonl are read and appended
                    (default: benchmarks/results/tuning/ledger of this checkout)
The run line names the rule, so the provenance is in the ledger (tuned/README.md).
"""
import argparse
import datetime
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATE = datetime.date.today().isoformat()
GRID_N = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
GRID_B = [128, 512, 2048, 8192, 32768]
CANDS = {'float': ['cta', 'blocked:nb=64', 'blocked:nb=128']}
for d in ('double', 'cfloat', 'cdouble'):
    CANDS[d] = ['cta', 'blocked:nb=64']
SIZES = {'float': (4, 4), 'double': (8, 8), 'cfloat': (8, 4), 'cdouble': (16, 8)}
BUDGET = {'sm_89': 99 * 1024 - 4096, 'sm_120': 99 * 1024 - 4096}  # LOCAL_MEM_SIZE less the reserve
MAX_WG = 1024


def hole(b):
    return 49920 if 47104 < b <= 49664 else b


def cta_max_n(dt, dev):
    sd, sr = SIZES[dt]
    ts = 2 if sd > 8 else 4
    best = 0
    for m in range(1, 1025):
        mp = (m + ts - 1) // ts * ts
        t = mp // ts
        wg = min((t * (t + 1) // 2 + 31) // 32 * 32, 512, MAX_WG)
        if hole((mp + 1) * mp * sd + mp * sr + 64) > BUDGET[dev] or t * (t + 1) // 2 > 2 * wg:
            break
        best = m
    return best


def runnable(dt, dev, fam, n):
    cap = cta_max_n(dt, dev)
    if fam == 'cta':
        return n <= cap
    return min(int(fam.split('=')[1]), n) <= cap


def emit(path, sha, rule, dev, ranked_of):
    date = DATE
    run_id = 'transcribed-' + sha
    with open(path, 'a') as f:
        for dt in ('float', 'double', 'cfloat', 'cdouble'):
            cands = CANDS[dt]
            f.write(json.dumps({"kind": "run", "dtype": dt, "run_id": run_id, "host": "", "device": dev,
                                "device_name": "", "batchlas": sha, "argv": rule, "date": date,
                                "tier": "transcribed", "keys": "n:log:3 batch:log",
                                "candidates": "|".join(cands)}) + "\n")
            for n in GRID_N:
                for b in GRID_B:
                    ranked = [c for c in ranked_of(dt, n, b) if c in cands and runnable(dt, dev, c, n)]
                    if not ranked:
                        continue
                    cell = {"kind": "cell", "dtype": dt, "run_id": run_id, "tier": "transcribed",
                            "key": f"n={n},batch={b}", "round": 0, "date": date, "ranked": "|".join(ranked),
                            "cands": "|".join(cands)}
                    for i in range(len(cands)):
                        cell.update({f"h{i}": "", f"s{i}": "ok", f"r{i}": "", f"m{i}": None, f"lo{i}": None,
                                     f"hi{i}": None, f"n{i}": 0})
                    f.write(json.dumps(cell) + "\n")


def measured_winners(path):
    """(dtype, n, batch) -> ranked families from the newest timed sm_89 records of float and cfloat."""
    best = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") != "cell" or r.get("tier") in ("transcribed", "converted", "prior"):
            continue
        if r.get("dtype") not in ("float", "cfloat") or not r.get("ranked"):
            continue
        kv = dict(x.split('=') for x in r["key"].split(','))
        best[(r["dtype"], int(kv["n"]), int(kv["batch"]))] = (r["date"], r["ranked"].split('|'))
    return {k: v[1] for k, v in best.items()}


def nearest(win, dt, n, b):
    import math
    keys = [k for k in win if k[0] == dt]
    k = min(keys, key=lambda k: 3 * abs(math.log(k[1] / n)) + abs(math.log(k[2] / b)))
    return win[k]


def main():
    global DATE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sha")
    ap.add_argument("mode", choices=("provisional", "measured"))
    ap.add_argument("--date", default=DATE)
    ap.add_argument("--ledger-dir", default=REPO + '/benchmarks/results/tuning/ledger')
    args = ap.parse_args()
    sha, mode, DATE = args.sha, args.mode, args.date
    led = args.ledger_dir + '/potri.%s.jsonl'
    if mode == 'provisional':
        rule = "transcribed: cta first where its tile fits the device, else blocked:nb=64, then nb=128 (no timing; replaced by measured rows)"
        order = lambda dt, n, b: ['cta', 'blocked:nb=64', 'blocked:nb=128']
        for dev in ('sm_89', 'sm_120'):
            emit(led % dev, sha, rule, dev, order)
        return
    win = measured_winners(led % 'sm_89')
    src = {'float': 'float', 'double': 'float', 'cfloat': 'cfloat', 'cdouble': 'cfloat'}
    rule89 = "transcribed: double from the measured sm_89 float ranking, cdouble from cfloat's, nearest cell, filtered to the dtype's runnable families"
    rule120 = "transcribed: every dtype from the measured sm_89 float/cfloat ranking (double from float, cdouble from cfloat), nearest cell, filtered to runnable families"
    order = lambda dt, n, b: nearest(win, src[dt], n, b) + ['blocked:nb=64']
    dedup = lambda f: (lambda dt, n, b: list(dict.fromkeys(f(dt, n, b))))
    # sm_89: only double and cdouble are transcribed; the measured float/cfloat rows stand.
    emit_dt(led % 'sm_89', sha, rule89, 'sm_89', dedup(order), ('double', 'cdouble'))
    # float and cfloat cells the campaign could not hold (over the 8 GiB operand cap): their nearest measured cell.
    rule_gap = "transcribed: sm_89 float/cfloat cells over the campaign's operand cap, from the nearest measured cell"
    emit_dt(led % 'sm_89', sha + '-gap', rule_gap, 'sm_89', dedup(order), ('float', 'cfloat'),
            skip=lambda dt, n, b: (dt, n, b) in win)
    emit_dt(led % 'sm_120', sha, rule120, 'sm_120', dedup(order), ('float', 'double', 'cfloat', 'cdouble'))


def emit_dt(path, sha, rule, dev, ranked_of, dtypes, skip=lambda dt, n, b: False):
    date = DATE
    run_id = 'transcribed-' + sha
    with open(path, 'a') as f:
        for dt in dtypes:
            cands = CANDS[dt]
            f.write(json.dumps({"kind": "run", "dtype": dt, "run_id": run_id, "host": "", "device": dev,
                                "device_name": "", "batchlas": sha, "argv": rule, "date": date,
                                "tier": "transcribed", "keys": "n:log:3 batch:log",
                                "candidates": "|".join(cands)}) + "\n")
            for n in GRID_N:
                for b in GRID_B:
                    if skip(dt, n, b):
                        continue
                    ranked = [c for c in ranked_of(dt, n, b) if c in cands and runnable(dt, dev, c, n)]
                    if not ranked:
                        continue
                    cell = {"kind": "cell", "dtype": dt, "run_id": run_id, "tier": "transcribed",
                            "key": f"n={n},batch={b}", "round": 0, "date": date, "ranked": "|".join(ranked),
                            "cands": "|".join(cands)}
                    for i in range(len(cands)):
                        cell.update({f"h{i}": "", f"s{i}": "ok", f"r{i}": "", f"m{i}": None, f"lo{i}": None,
                                     f"hi{i}": None, f"n{i}": 0})
                    f.write(json.dumps(cell) + "\n")


if __name__ == '__main__':
    main()
