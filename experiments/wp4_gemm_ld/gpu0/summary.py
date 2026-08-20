#!/usr/bin/env python3
"""One table per shape: native (auto route), native (128x128 forced), cuBLAS.

Values are GFLOP/s at beta=1, batch=512, float NN, RTX 4090 (GPU 0).
Sources: pad_curve_merged.csv (sweeps 1+2, auto route and vendor) and
forced_curve.csv (the 128x128-forced runs).
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
merged = list(csv.DictReader(open(os.path.join(HERE, "pad_curve_merged.csv"))))
forced = list(csv.DictReader(open(os.path.join(HERE, "forced_curve.csv"))))

def g(rows, **kw):
    for r in rows:
        if all(str(r[k]) == str(v) for k, v in kw.items()):
            return float(r["gflops"])
    return None

pads = [0, 1, 2, 3, 4, 8, 16, 32, 64, 96, 128, 129, 192, 256, 384, 512]
for shape, m, k in (("outer", 128, 128), ("outer", 128, 256),
                    ("inner", 32, 32), ("inner", 32, 96)):
    print(f"\n== {shape} m={m} n=1024 k={k} batch=512 float NN beta=1 -- GFLOP/s ==")
    print("pad          " + "".join(f"{p:>7}" for p in pads))
    lines = {}
    for label, get in (
        ("native-auto", lambda p: g(merged, shape=shape, k=k, route="native", pad=p)),
        ("cuBLAS", lambda p: g(merged, shape=shape, k=k, route="vendor", pad=p)),
        ("nat-force128", lambda p: (g(forced, k=k, beta="1", mode="force128", pad=p)
                                    if shape == "outer" else None)),
    ):
        vals = [get(p) for p in pads]
        lines[label] = vals
        print(f"{label:13}" + "".join("      -" if v is None else f"{v:7.0f}" for v in vals))
    na, cb = lines["native-auto"], lines["cuBLAS"]
    print("cuBLAS/nat   " + "".join(
        "      -" if not (a and b) else f"{a / b:7.2f}" for a, b in zip(na, cb)))
