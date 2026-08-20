#!/usr/bin/env python3
"""The forced-128x128 pad curve (router removed) next to the auto-routed one.

auto rows come from raw/ + raw2/ (sweep.sh, sweep2.sh); forced rows from
forced/ (forced_curve.sh). Warm-up convention throughout: each CSV holds two
passes over the k list, only the second is reported.
"""
import csv, glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "forced_curve.csv")

rows = []
for d in ("raw", "raw2"):
    for path in sorted(glob.glob(os.path.join(HERE, d, "outer-native-b*.csv"))):
        m = re.match(r"outer-native-b(\d)-pad(\d+)$", os.path.basename(path)[:-4])
        if not m:
            continue
        beta, pad = int(m.group(1)), int(m.group(2))
        recs = list(csv.DictReader(open(path)))
        for r in recs[len(recs) // 2:]:
            ms, sd = float(r["avg_ms"]), float(r["stddev_ms"])
            rows.append(dict(mode="auto", beta=beta, pad=pad, k=int(r["arg2"]), ms=ms,
                             rsd=round(100.0 * sd / ms, 2), gflops=float(r["GFLOPS"])))
for path in sorted(glob.glob(os.path.join(HERE, "forced", "t-force128-b*.csv"))):
    m = re.match(r"t-force128-b(\d)-pad(\d+)$", os.path.basename(path)[:-4])
    if not m:
        continue
    beta, pad = int(m.group(1)), int(m.group(2))
    recs = list(csv.DictReader(open(path)))
    for r in recs[len(recs) // 2:]:
        ms, sd = float(r["avg_ms"]), float(r["stddev_ms"])
        rows.append(dict(mode="force128", beta=beta, pad=pad, k=int(r["arg2"]), ms=ms,
                         rsd=round(100.0 * sd / ms, 2), gflops=float(r["GFLOPS"])))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["k"], r["beta"], r["mode"], r["pad"])):
        w.writerow(r)

for r in rows:
    if r["rsd"] > 10.0:
        print("NOISY {mode} beta={beta} pad={pad} k={k} {ms:.4f} rsd={rsd}%".format(**r))

for beta in (1, 0):
    for k in (128, 256):
        sub = [r for r in rows if r["beta"] == beta and r["k"] == k]
        if not sub:
            continue
        pads = sorted({r["pad"] for r in sub if
                       any(x["mode"] == "force128" and x["pad"] == r["pad"] for x in sub)})
        print(f"\n== outer m=128 n=1024 k={k} batch=512 float NN beta={beta} ==")
        print("pad        " + "".join(f"{p:>7}" for p in pads))
        cell = {}
        for mode in ("auto", "force128"):
            line = []
            for p in pads:
                hit = [r for r in sub if r["mode"] == mode and r["pad"] == p]
                cell[(mode, p)] = hit[0]["ms"] if hit else None
                line.append("      -" if not hit else f"{hit[0]['ms']:7.3f}")
            print(f"{mode:11}" + "".join(line))
        for mode in ("auto", "force128"):
            b = cell[(mode, 0)]
            print(f"{mode[:4]}/pad0  " + "".join(
                "      -" if not cell[(mode, p)] else f"{cell[(mode, p)] / b:7.2f}" for p in pads))
print(f"\nwrote {OUT}")
