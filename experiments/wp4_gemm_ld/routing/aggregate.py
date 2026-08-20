#!/usr/bin/env python3
"""Aggregate the WP4 routing experiments into one CSV + printed tables.

Every row is the mean of the two repeated k entries inside one CSV (the
benchmark is asked for "k,k" so the repeat is inside a single process, which
is what pins the within-process spread). Cells whose gpu_guard line reported a
foreign process were re-run; only re-run values are here.
"""
import csv, glob, os, re, statistics, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def read(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["arg0"]), int(r["arg1"]), int(r["arg2"]), int(r["arg3"]),
                         float(r["avg_ms"]), float(r["stddev_ms"])))
    if not rows:
        return None
    ms = [r[4] for r in rows]
    sd = max(r[5] / r[4] for r in rows)
    return dict(m=rows[0][0], n=rows[0][1], k=rows[0][2], batch=rows[0][3],
                ms=statistics.mean(ms), rsd=sd)


def collect(pattern, keyfn):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        d = read(p)
        if d is None:
            continue
        out[keyfn(os.path.basename(p))] = d
    return out


def main():
    rows = []

    # E1: B-only pad, m scaling
    e1 = collect("e1-*.csv", lambda b: tuple(b[3:-4].rsplit("-padB", 1)))
    for tag in ("m128", "m256", "m512", "m1024"):
        a, b = e1[(tag, "0")], e1[(tag, "384")]
        rows.append(dict(exp="e1", shape=f"{a['m']}x{a['n']}x{a['k']}", batch=a["batch"],
                         cfg="native-auto", pad0_ms=a["ms"], pad384_ms=b["ms"],
                         ratio=b["ms"] / a["ms"], rsd=max(a["rsd"], b["rsd"]),
                         note="pad applied to B only"))

    # E2/E4/E5/E6: auto vs forced-128x128 vs vendor
    def dump(prefix, tags):
        got = {}
        for p in sorted(glob.glob(os.path.join(ROOT, prefix + "*.csv"))):
            b = os.path.basename(p)[:-4]
            mm = re.match(r"e\d-([A-Z0-9]+)-(auto|f128|vendor)-pad(\d+)(?:-r\d)?$", b)
            if not mm:
                continue
            d = read(p)
            if d:
                got.setdefault((mm.group(1), mm.group(2), mm.group(3)), []).append(d)
        for tag in tags:
            for pad in ("0", "384"):
                cells = {}
                for cfg in ("auto", "f128", "vendor"):
                    ds = got.get((tag, cfg, pad))
                    if ds:
                        cells[cfg] = statistics.mean(d["ms"] for d in ds)
                if "auto" not in cells or "f128" not in cells:
                    continue
                d0 = got[(tag, "auto", pad)][0]
                rows.append(dict(exp=prefix.rstrip("-"),
                                 shape=f"{d0['m']}x{d0['n']}x{d0['k']}", batch=d0["batch"],
                                 cfg=f"pad{pad}", auto_ms=cells["auto"], f128_ms=cells["f128"],
                                 vendor_ms=cells.get("vendor"),
                                 auto_speedup=cells["auto"] / cells["f128"],
                                 f128_vs_vendor=(cells.get("vendor") / cells["f128"]
                                                 if cells.get("vendor") else None)))

    dump("e2-", ["R1", "R2", "R3"])
    dump("e5-", ["S1", "S2", "S3", "S4", "S5", "S6", "S7"])
    dump("e6-", ["T1", "T2", "T3", "T4", "T5"])

    fields = ["exp", "shape", "batch", "cfg", "pad0_ms", "pad384_ms", "ratio", "rsd",
              "auto_ms", "f128_ms", "vendor_ms", "auto_speedup", "f128_vs_vendor", "note"]
    out = os.path.join(os.path.dirname(ROOT), "summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
