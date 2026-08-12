#!/usr/bin/env python3
"""Fold every RESULT line from every log into one table, ratio'd against cuBLAS."""
import re, glob, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for path in ["bench.log", "bench_complex_split.log", "bench_gapfill_b.log"]:
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        continue
    for line in open(p):
        if "RESULT" not in line:
            continue
        cand = re.search(r"CAND=(\S+)", line)
        d = dict(re.findall(r"(\w+)=([^\s|]+)", line))
        if "ms" not in d or d["ms"] in ("FAILED", "nan"):
            rows.append(dict(cand=cand.group(1) if cand else "?", dtype=d.get("dtype"),
                             m=int(d.get("m", 0)), batch=int(d.get("batch", 0)),
                             beta=d.get("beta"),
                             tile=d.get("tile", "-"), ms=None, tf=None,
                             err=d.get("maxrelerr")))
            continue
        rows.append(dict(cand=cand.group(1) if cand else "?", dtype=d["dtype"],
                         m=int(d["m"]), batch=int(d["batch"]), beta=d["beta"],
                         tile=d.get("tile", "-"), ms=float(d["ms"]),
                         tf=float(d["tflops"]), err=d.get("maxrelerr")))

base = {}
for r in rows:
    if r["cand"] == "cublas" and r["ms"]:
        base[(r["dtype"], r["m"], r["batch"], r["beta"])] = r["tf"]

order = {"cublas": 0, "incumbent-tiled16": 1, "128x128-t8x4": 2, "128x64-t8x4": 3,
         "64x64-k16-t4x4": 4, "complex-split": 5}
rows.sort(key=lambda r: (str(r["dtype"]), r["m"] or 0, str(r["beta"]),
                         order.get(r["cand"], 9), str(r["tile"])))

hdr = f"{'dtype':8} {'shape':16} {'b':2} {'candidate':18} {'tile':22} {'ms':>10} {'TFLOP/s':>8} {'vs cuBLAS':>9} {'maxrelerr':>10}"
print(hdr); print("-" * len(hdr))
prev = None
for r in rows:
    key = (r["dtype"], r["m"], r["beta"])
    if prev and prev != key:
        print()
    prev = key
    b = base.get((r["dtype"], r["m"], r["batch"], r["beta"]))
    if r["ms"] is None:
        print(f"{r['dtype']:8} {'-':16} {r['beta'] or '-':2} {r['cand']:18} {r['tile']:22} "
              f"{'UNLAUNCHABLE':>10}")
        continue
    ratio = f"{r['tf']/b:.3f}x" if b else "-"
    shape = f"{r['m']}^3 b{r['batch']}"
    print(f"{r['dtype']:8} {shape:16} {r['beta']:2} {r['cand']:18} {r['tile']:22} "
          f"{r['ms']:10.4f} {r['tf']:8.2f} {ratio:>9} {r['err'] or '-':>10}")
