#!/usr/bin/env python3
"""getrs data gate (5b): the OLD predicate (tools/transcribe/getrs_transcribe.cc --replay, the
real route_getrs.hh of 424a45bc) vs the NEW selection (tuned/getrs.<dtype>.<device>.txt via
sweep_to_table.nearest, the same rule as Table::nearest, then the first entry passing the
can_run-equivalent capacities, then the last resort) at random OFF-GRID points.

Usage: getrs_gate.py [points] [transcriber binary, built as getrs_transcribe.cc's header says]"""
import collections
import math
import os
import random
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(REPO, "scripts"))
import sweep_to_table as stt  # noqa: E402

GT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gt"
GRID_N = {1, 2, 4, 8, 16, 24, 31, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}
GRID_R = {1, 2, 3, 4, 5, 8, 16, 32, 63, 64, 127, 128, 256, 512, 1024}
GRID_B = {1, 16, 127, 128, 512, 2048, 8192, 32768}
MAX_NRHS = 8  # kGetrsFusedMaxRhs, a build constant
# Two capacity regimes, applied identically to both sides: unlimited, and a device-like
# n*nrhs ceiling (the order of the real ones; the value itself does not matter here).
CAPS = {"unlimited": {d: 10**15 for d in stt.DTYPES.values()},
        "device": {"float": 23264, "double": 11632, "cfloat": 11632, "cdouble": 5816}}
N_POINTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500


def logu(rng, lo, hi):
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def points(seed):
    rng = random.Random(seed)
    out = []
    while len(out) < N_POINTS:
        n, r, b = logu(rng, 1, 3000), logu(rng, 1, 3000), logu(rng, 1, 100000)
        if n in GRID_N or r in GRID_R or b in GRID_B:
            continue  # off-grid in every key
        out.append((n, r, b))
    # Threshold neighbourhoods, densely: every combination just around each threshold.
    for n in (29, 30, 33, 34, 35):
        for r in (6, 7, 9, 10, 60, 62, 65, 66, 120, 126, 129, 130):
            for b in (100, 126, 129, 130, 200):
                out.append((n, r, b))
    return out


def new_choice(rows, keyspec, dt, n, r, b, vendor, cap):
    def can(c):
        if c == "cta":
            return r <= MAX_NRHS and n * r <= cap
        if c == "blocked":
            return True
        return vendor
    _, ranked = stt.nearest(rows, (n, r, b), keyspec)
    for c, _ in ranked:
        if can(c):
            return c
    for c in ("blocked", "vendor", "cta"):
        if can(c):
            return c
    return "none"


def main():
    pts = points(20261005)
    total_bad = 0
    report = []
    for dev in ("sm_89", "sm_120"):
        for dt in ("float", "double", "cfloat", "cdouble"):
            with open(os.path.join(REPO, "tuned", f"getrs.{dt}.{dev}.txt")) as f:
                _, keyspec, rows = stt.parse_table(f.read())
            for capname, caps in CAPS.items():
                cap = caps[dt]
                for vendor in (1, 0):
                    lines = "".join(f"{dt} {n} {r} {b} {vendor} {cap} {MAX_NRHS}\n" for n, r, b in pts)
                    old = subprocess.run([GT, "--replay"], input=lines, capture_output=True, text=True,
                                         check=True).stdout.split()
                    assert len(old) == len(pts)
                    bad = collections.Counter()
                    for (n, r, b), o in zip(pts, old):
                        nw = new_choice(rows, keyspec, dt, n, r, b, vendor, cap)
                        if nw != o:
                            bad[(o, nw)] += 1
                            if sum(bad.values()) <= 5:
                                print(f"  DISAGREE {dev} {dt} {capname} vendor={vendor} n={n} nrhs={r} batch={b}:"
                                      f" old={o} new={nw}")
                    nbad = sum(bad.values())
                    total_bad += nbad
                    rate = 100.0 * (len(pts) - nbad) / len(pts)
                    report.append((dev, dt, capname, vendor, len(pts), nbad, rate, dict(bad)))
    for dev, dt, capname, vendor, npts, nbad, rate, bad in report:
        print(f"{dev:6s} {dt:7s} cap={capname:9s} vendor={vendor} points={npts} disagree={nbad} "
              f"agreement={rate:.3f}% {bad if bad else ''}")
    print(f"TOTAL disagreements: {total_bad}")


if __name__ == "__main__":
    main()
