#!/usr/bin/env python3
"""Inner shape: which operand's padded ld costs the time."""
import csv, glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "inner_operand")
OUT = os.path.join(HERE, "inner_operand.csv")

rows = []
for path in sorted(glob.glob(os.path.join(D, "i-*.csv"))):
    m = re.match(r"i-p(\d+)-(\w+)$", os.path.basename(path)[:-4])
    if not m:
        continue
    pad, which = int(m.group(1)), m.group(2)
    recs = list(csv.DictReader(open(path)))
    for r in recs[len(recs) // 2:]:
        ms, sd = float(r["avg_ms"]), float(r["stddev_ms"])
        rows.append(dict(pad=pad, which=which, k=int(r["arg2"]), ms=ms,
                         rsd=round(100.0 * sd / ms, 2), gflops=float(r["GFLOPS"])))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["k"], r["pad"], r["which"])):
        w.writerow(r)

for r in rows:
    if r["rsd"] > 10.0:
        print("NOISY pad={pad} {which} k={k} {ms:.4f} rsd={rsd}%".format(**r))

order = ["none", "A", "B", "C", "ABC"]
for k in sorted({r["k"] for r in rows}):
    print(f"\n== inner m=32 n=1024 k={k} batch=512 float NN beta=1, native ==")
    print("padded    " + "".join(f"{w:>9}" for w in order))
    for pad in sorted({r["pad"] for r in rows}):
        line, base = [], None
        for w in order:
            hit = [r for r in rows if r["pad"] == pad and r["which"] == w and r["k"] == k]
            v = hit[0]["ms"] if hit else None
            if w == "none":
                base = v
            line.append("        -" if v is None else f"{v:9.3f}")
        print(f"pad={pad:<6}" + "".join(line))
        rel = []
        for w in order:
            hit = [r for r in rows if r["pad"] == pad and r["which"] == w and r["k"] == k]
            rel.append("        -" if not hit else f"{hit[0]['ms'] / base:9.2f}")
        print("  x none" + "".join(rel))
print(f"\nwrote {OUT}")
