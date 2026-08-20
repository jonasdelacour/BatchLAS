#!/usr/bin/env python3
"""Aggregate experiments/wp4_complex/gpu0/raw into one tidy CSV + a ratio table.

One raw CSV per (shape-tag, arm, beta, pad, rep); each contains one row per
scalar type. rep 0 is the discarded warm-up pass.
"""
import csv
import glob
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

TYPE_RE = [
    ("cfloat", "complex<float>"),
    ("cdouble", "complex<double>"),
    ("float", "float"),
    ("double", "double"),
]

FNAME = re.compile(r"^(?P<tag>[^-]+)-(?P<arm>auto|wide|vendor)-b(?P<beta>\d)-pad(?P<pad>\d+)-r(?P<rep>\d+)\.csv$")


def scalar_of(name):
    if "complex<float>" in name:
        return "cfloat"
    if "complex<double>" in name:
        return "cdouble"
    if "double" in name:
        return "double"
    return "float"


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        m = FNAME.match(os.path.basename(path))
        if not m:
            continue
        if m.group("rep") == "0":
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append({
                    "tag": m.group("tag"),
                    "arm": m.group("arm"),
                    "beta": int(m.group("beta")),
                    "pad": int(m.group("pad")),
                    "rep": int(m.group("rep")),
                    "type": scalar_of(r["name"]),
                    "m": int(r["arg0"]), "n": int(r["arg1"]),
                    "k": int(r["arg2"]), "batch": int(r["arg3"]),
                    "iters": int(r["iterations"]),
                    "avg_ms": float(r["avg_ms"]),
                    "sd_ms": float(r["stddev_ms"]),
                })
    return rows


def main():
    rows = load()
    if not rows:
        print("no data", file=sys.stderr)
        return 1

    key = lambda r: (r["type"], r["tag"], r["beta"], r["pad"], r["arm"])
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    tidy_path = os.path.join(HERE, "tidy.csv")
    with open(tidy_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "tag", "m", "n", "k", "batch", "beta", "pad", "arm",
                    "reps", "median_ms", "min_ms", "max_ms", "rel_spread",
                    "worst_within_run_rsd"])
        summary = {}
        for kk in sorted(groups):
            g = groups[kk]
            ms = sorted(r["avg_ms"] for r in g)
            # MIN, not median. Interference on a shared GPU only ever ADDS
            # time, so the minimum over repetitions is the robust estimator
            # here; `rel_spread` stays as the diagnostic that says how much
            # interference there was.
            med = min(ms)
            spread = (max(ms) - min(ms)) / med if med else 0.0
            rsd = max((r["sd_ms"] / r["avg_ms"]) if r["avg_ms"] else 0.0 for r in g)
            g0 = g[0]
            w.writerow([kk[0], kk[1], g0["m"], g0["n"], g0["k"], g0["batch"],
                        kk[2], kk[3], kk[4], len(g),
                        f"{med:.5f}", f"{min(ms):.5f}", f"{max(ms):.5f}",
                        f"{spread:.4f}", f"{rsd:.4f}"])
            summary[kk] = (med, spread, rsd, g0)

    ratio_path = os.path.join(HERE, "ratios.csv")
    hdr = ["type", "tag", "m", "n", "k", "batch", "beta", "pad",
           "auto_ms", "wide_ms", "vendor_ms",
           "ratio_auto_over_wide", "ratio_vendor_over_wide",
           "ratio_vendor_over_auto", "GBps_best", "flag"]
    out = []
    for (t, tag, beta, pad, arm), v in summary.items():
        if arm != "auto":
            continue
        a = summary.get((t, tag, beta, pad, "auto"))
        wv = summary.get((t, tag, beta, pad, "wide"))
        vv = summary.get((t, tag, beta, pad, "vendor"))
        if not (a and wv):
            continue
        g0 = a[3]
        flags = []
        for nm, s in (("auto", a), ("wide", wv), ("vendor", vv)):
            if s and (s[1] > 0.10 or s[2] > 0.10):
                flags.append(f"{nm}:spread{s[1]:.2f}/rsd{s[2]:.2f}")
        # Compulsory traffic at beta=1: read A, read B, read C, write C.
        # beta=0 drops the read of C. This is the DRAM roof the shape sits
        # under; a 4090 measures ~1000 GB/s on a pure stream.
        esz = {"cfloat": 8, "cdouble": 16, "float": 4, "double": 8}[t]
        cterms = 2 if beta else 1
        bytes_moved = (g0["m"] * g0["k"] + g0["k"] * g0["n"]
                       + cterms * g0["m"] * g0["n"]) * esz * g0["batch"]
        best_ms = min(a[0], wv[0])
        gbps = bytes_moved / (best_ms * 1e-3) / 1e9
        out.append([t, tag, g0["m"], g0["n"], g0["k"], g0["batch"], beta, pad,
                    f"{a[0]:.5f}", f"{wv[0]:.5f}",
                    f"{vv[0]:.5f}" if vv else "",
                    f"{a[0]/wv[0]:.3f}",
                    f"{vv[0]/wv[0]:.3f}" if vv else "",
                    f"{vv[0]/a[0]:.3f}" if vv else "",
                    f"{gbps:.0f}",
                    ";".join(flags)])

    order = {"cfloat": 0, "cdouble": 1, "float": 2, "double": 3}
    out.sort(key=lambda r: (order.get(r[0], 9), r[1], r[6], r[7]))
    with open(ratio_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(out)

    widths = [max(len(str(r[i])) for r in [hdr] + out) for i in range(len(hdr))]
    for r in [hdr] + out:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)).rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
