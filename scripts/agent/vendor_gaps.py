#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Where does the vendor still beat every native choice? Read from the tuned tables (each row
lists every timed choice), at saturating batch only.

    python3 scripts/agent/vendor_gaps.py gemm [--device sm_89] [--min-batch 128] [--dtype float]
                                              [--min-ratio 1.05] [--top 15]

Prints, per exact-key group (dtype, transposes, layout, uplo, ...), how many timed rows the vendor
wins and the geometric-mean and worst native/vendor ratio, then the worst rows. Rows with no
vendor time or no native time (transcribed '-') are skipped and counted. The tables are the
ledger's distillate; for a cell's spread or race history, read the ledger itself.
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUMERIC_KEYS = {"m", "n", "k", "q", "nrhs", "order", "batch", "nnz", "rows", "cols"}


def parse(path):
    for line in open(path):
        if line.startswith("#") or "|" not in line:
            continue
        body = line.split("#", 1)[0]
        parts = [p.strip() for p in body.split("|")]
        keys = dict(kv.split("=", 1) for kv in parts[0].split())
        times = {}
        for p in parts[1:]:
            m = re.match(r"(\S+)\s+(\d[0-9.]*(?:[eE][+-]?\d+)?)$", p)
            if m:
                times[m.group(1)] = float(m.group(2))
        yield keys, times


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("op")
    ap.add_argument("--device", default="sm_89")
    ap.add_argument("--min-batch", type=int, default=128)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--min-ratio", type=float, default=1.05, help="report rows the vendor wins by this much")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    path = os.path.join(REPO, "tuned", "%s.%s.txt" % (args.op, args.device))
    if not os.path.isfile(path):
        sys.exit("vendor_gaps: no table %s" % path)

    groups = defaultdict(list)
    worst, skipped, timed = [], 0, 0
    for keys, times in parse(path):
        if int(keys.get("batch", args.min_batch)) < args.min_batch:
            continue
        if args.dtype and keys.get("dtype") != args.dtype:
            continue
        native = {c: t for c, t in times.items() if not c.startswith("vendor")}
        if "vendor" not in times or not native:
            skipped += 1
            continue
        timed += 1
        best_c = min(native, key=native.get)
        ratio = native[best_c] / times["vendor"]
        exact = " ".join("%s=%s" % (k, v) for k, v in keys.items() if k not in NUMERIC_KEYS)
        groups[exact].append(ratio)
        if ratio >= args.min_ratio:
            shape = " ".join("%s=%s" % (k, v) for k, v in keys.items() if k in NUMERIC_KEYS)
            worst.append((ratio, exact, shape, best_c))

    print("%s.%s: %d timed row(s) at batch >= %d, %d skipped (no vendor or native time)"
          % (args.op, args.device, timed, args.min_batch, skipped))
    print("%-48s %6s %8s %8s  (ratio = best native / vendor; >1 vendor wins)" % ("group", "rows", "geomean", "worst"))
    for g, rs in sorted(groups.items()):
        wins = sum(r >= args.min_ratio for r in rs)
        gm = math.exp(sum(math.log(r) for r in rs) / len(rs))
        print("%-48s %6d %8.3f %8.3f  vendor wins %d" % (g[:48], len(rs), gm, max(rs), wins))
    print("\nworst %d row(s) the vendor wins by >= %.2fx:" % (min(args.top, len(worst)), args.min_ratio))
    for ratio, exact, shape, best_c in sorted(worst, reverse=True)[:args.top]:
        print("  %5.2fx  %s  %s  (best native %s)" % (ratio, exact, shape, best_c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
