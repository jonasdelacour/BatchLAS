#!/usr/bin/env python3
"""Compare two syev_cta_benchmark CSV runs.

Reports the best (minimum) avg_ms over the wg-multiplier sweep for each
(n, batch, jobz, uplo) case, so tuning noise in `arg4` does not hide the
underlying kernel change.

Usage: compare_syev_cta.py BASE.csv NEW.csv
"""
import csv
import sys
from collections import defaultdict


def best_by_case(path):
    best = defaultdict(lambda: float("inf"))
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["arg0"]), int(row["arg1"]), int(row["arg2"]), int(row["arg3"]))
            best[key] = min(best[key], float(row["avg_ms"]))
    return best


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    base, new = best_by_case(sys.argv[1]), best_by_case(sys.argv[2])

    print(f"{'n':>4} {'batch':>7} {'jobz':>5} {'uplo':>5} "
          f"{'base ms':>10} {'new ms':>10} {'speedup':>8}")
    total_base = total_new = 0.0
    for key in sorted(set(base) & set(new)):
        b, x = base[key], new[key]
        total_base += b
        total_new += x
        print(f"{key[0]:>4} {key[1]:>7} {key[2]:>5} {key[3]:>5} "
              f"{b:>10.5f} {x:>10.5f} {b / x:>7.2f}x")
    if total_new:
        print(f"{'':>4} {'':>7} {'':>5} {'TOTAL':>5} "
              f"{total_base:>10.5f} {total_new:>10.5f} {total_base / total_new:>7.2f}x")


if __name__ == "__main__":
    main()
