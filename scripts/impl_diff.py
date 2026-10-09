#!/usr/bin/env python3
"""Cross-implementation accuracy A/B: one accuracy harness, two build trees, identical inputs.

  impl_diff.py run --tree-a build-dpcpp --tree-b build-acpp-tests --harness eigensolver_accuracy \
      --out DIR --tag float_n32 -- --type=float --n=32 --impl=all
  impl_diff.py compare A.csv B.csv [--threshold 2] [--md table.md] [--cells cells.csv]

Harnesses: steqr_accuracy, eigensolver_accuracy, orthogonality_accuracy (benchmarks/). Each writes
one row per (impl, sample) with an `input_hash` of the generated matrix. Samples pair on
(impl, n, neigs, dtype, scheme, cta_shift, sample); a cell is that key minus the sample, plus the
decade of target_log10_cond. Per cell and metric: median and max for A and B, their B/A ratios, the
geometric mean of the paired per-sample ratio, the largest paired relative difference, and how
many inputs and results are bit-identical.
Values below the dtype's eps are clamped to eps before a ratio, so two errors at rounding level
never flag. A cell flags when a ratio leaves [1/t, t] or the non-finite counts differ.

The input is generated on the device by each tree (Philox, ortho, gemm, tridiagonal reduction), so
it can round differently. `run` therefore passes `--inputs DIR` to both: tree A writes its matrices,
tree B reads them. `--own-inputs` measures generator plus solver instead. The `inputs =` column
says how many pairs really saw the same matrix.
Plan: docs/design/sycl-implementations.md section 6; results: benchmarks/results/sycl-implementations/accuracy/.
"""
import argparse
import csv
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict

METRICS = ("R", "O", "max_relerr", "relerr", "orthogonality")
KEY_COLS = ("impl", "n", "neigs", "backend", "dtype", "scheme", "cta_shift")
EPS = {"float": 2.0 ** -23, "double": 2.0 ** -52}


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    if not rows:
        sys.exit(f"impl_diff: no rows in {path}")
    return rows


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.3g}"


def cell_of(row):
    key = tuple(row.get(c, "") for c in KEY_COLS)
    decade = math.floor(num(row["target_log10_cond"]) + 1e-9)
    return key + (decade,)


def pair(rows_a, rows_b):
    index_b = {}
    for r in rows_b:
        index_b[tuple(r.get(c, "") for c in KEY_COLS) + (r["sample"],)] = r
    pairs = defaultdict(list)
    unpaired = 0
    for r in rows_a:
        other = index_b.pop(tuple(r.get(c, "") for c in KEY_COLS) + (r["sample"],), None)
        if other is None:
            unpaired += 1
            continue
        pairs[cell_of(r)].append((r, other))
    return pairs, unpaired, len(index_b)


def summarize(pairs, metrics, threshold):
    cells = []
    for cell, prs in sorted(pairs.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        dtype = cell[KEY_COLS.index("dtype")]
        eps = EPS.get(dtype, EPS["double"])
        same_in = sum(1 for a, b in prs if a.get("input_hash") and a.get("input_hash") == b.get("input_hash"))
        for m in metrics:
            if m not in prs[0][0]:
                continue
            va = [num(a[m]) for a, _ in prs]
            vb = [num(b[m]) for _, b in prs]
            fa = [x for x in va if math.isfinite(x)]
            fb = [x for x in vb if math.isfinite(x)]
            nonfin_a, nonfin_b = len(va) - len(fa), len(vb) - len(fb)
            ident = sum(1 for x, y in zip(va, vb) if x == y or (math.isnan(x) and math.isnan(y)))
            reldiff = max((abs(y - x) / max(abs(x), eps) for x, y in zip(va, vb)
                           if math.isfinite(x) and math.isfinite(y)), default=float("nan"))
            logs = [math.log(max(abs(y), eps) / max(abs(x), eps))
                    for x, y in zip(va, vb) if math.isfinite(x) and math.isfinite(y)]
            med_a = statistics.median(fa) if fa else float("nan")
            med_b = statistics.median(fb) if fb else float("nan")
            max_a = max(fa) if fa else float("nan")
            max_b = max(fb) if fb else float("nan")

            def ratio(x, y):
                if math.isnan(x) or math.isnan(y):
                    return float("nan")
                return max(abs(y), eps) / max(abs(x), eps)

            r_med, r_max = ratio(med_a, med_b), ratio(max_a, max_b)
            gmean = math.exp(statistics.fmean(logs)) if logs else float("nan")
            out = lambda r: (not math.isnan(r)) and (r > threshold or r < 1.0 / threshold)
            reasons = []
            if nonfin_a != nonfin_b:
                reasons.append(f"nonfinite {nonfin_a}->{nonfin_b}")
            if out(r_med):
                reasons.append("median")
            if out(r_max):
                reasons.append("max")
            if out(gmean):
                reasons.append("paired")
            cells.append({
                **{c: v for c, v in zip(KEY_COLS, cell)}, "log10_cond_decade": cell[-1],
                "metric": m, "samples": len(prs), "inputs_equal": same_in, "results_identical": ident,
                "median_a": med_a, "median_b": med_b, "ratio_median": r_med,
                "max_a": max_a, "max_b": max_b, "ratio_max": r_max, "paired_gmean": gmean,
                "max_paired_reldiff": reldiff,
                "nonfinite_a": nonfin_a, "nonfinite_b": nonfin_b, "flag": ";".join(reasons),
            })
    return cells


def write_cells(cells, path):
    if not cells:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        for c in cells:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in c.items()})


def markdown(cells, flagged_only):
    head = ("| impl | n | dtype | cond 1e | metric | samples | inputs = | results = | median A | median B "
            "| B/A med | max A | max B | B/A max | flag |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for c in cells:
        if flagged_only and not c["flag"]:
            continue
        lines.append("| " + " | ".join(str(x) for x in (
            c["impl"], c["n"], c["dtype"], c["log10_cond_decade"], c["metric"], c["samples"],
            c["inputs_equal"], c["results_identical"], fmt(c["median_a"]), fmt(c["median_b"]),
            fmt(c["ratio_median"]), fmt(c["max_a"]), fmt(c["max_b"]), fmt(c["ratio_max"]),
            c["flag"] or "")) + " |")
    return "\n".join(lines)


def compare(path_a, path_b, threshold, md=None, cells_out=None, flagged_only=False, quiet=False):
    rows_a, rows_b = load(path_a), load(path_b)
    pairs, only_a, only_b = pair(rows_a, rows_b)
    cells = summarize(pairs, METRICS, threshold)
    n_pairs = sum(len(v) for v in pairs.values())
    same_in = sum(1 for prs in pairs.values() for a, b in prs
                  if a.get("input_hash") and a.get("input_hash") == b.get("input_hash"))
    flagged = [c for c in cells if c["flag"]]
    summary = (f"{os.path.basename(path_a)} vs {os.path.basename(path_b)}: {n_pairs} pairs, "
               f"{same_in} identical inputs, {only_a} only in A, {only_b} only in B, "
               f"{len(cells)} cell-metrics, {len(flagged)} flagged (t={threshold})")
    table = markdown(cells, flagged_only)
    if cells_out:
        write_cells(cells, cells_out)
    if md:
        with open(md, "w") as f:
            f.write(summary + "\n\n" + table + "\n")
    if not quiet:
        print(summary)
        print(table)
    return cells, summary


def run(args):
    os.makedirs(args.out, exist_ok=True)
    outputs = {}
    shared = None
    if not args.own_inputs:
        shared = os.path.join(args.inputs_root or args.out, f"inputs.{args.harness}.{args.tag}")
        shutil.rmtree(shared, ignore_errors=True)
    for side, tree in (("a", args.tree_a), ("b", args.tree_b)):
        exe = os.path.join(tree, "benchmarks", args.harness)
        label = os.path.basename(os.path.normpath(tree))
        csv_path = os.path.join(args.out, f"{args.harness}.{args.tag}.{label}.csv")
        log_path = csv_path[:-4] + ".log"
        cmd = [exe, *args.harness_args, f"--output={csv_path}"]
        if shared:
            cmd.append(f"--inputs={shared}")
        with open(log_path, "w") as log:
            log.write("# " + " ".join(cmd) + "\n")
            log.flush()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            sys.exit(f"impl_diff: {exe} exited {rc}; see {log_path}")
        outputs[side] = csv_path
    if shared:
        shutil.rmtree(shared, ignore_errors=True)
    stem = os.path.join(args.out, f"{args.harness}.{args.tag}")
    _, summary = compare(outputs["a"], outputs["b"], args.threshold, md=stem + ".diff.md",
                         cells_out=stem + ".cells.csv", flagged_only=args.flagged_only, quiet=True)
    print(summary)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    r = sub.add_parser("run")
    r.add_argument("--tree-a", required=True)
    r.add_argument("--tree-b", required=True)
    r.add_argument("--harness", required=True,
                   choices=("steqr_accuracy", "eigensolver_accuracy", "orthogonality_accuracy"))
    r.add_argument("--out", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--own-inputs", action="store_true",
                   help="each tree generates its own matrices (default: tree A writes, tree B reads)")
    r.add_argument("--inputs-root", help="where the shared matrices go (default --out; deleted after)")
    r.add_argument("harness_args", nargs=argparse.REMAINDER)
    for q in (c, r):
        q.add_argument("--threshold", type=float, default=2.0)
        q.add_argument("--flagged-only", action="store_true")
    c.add_argument("--md")
    c.add_argument("--cells")
    args = p.parse_args()
    if args.cmd == "compare":
        cells, _ = compare(args.a, args.b, args.threshold, args.md, args.cells, args.flagged_only)
        sys.exit(1 if any(x["flag"] for x in cells) else 0)
    if args.harness_args and args.harness_args[0] == "--":
        args.harness_args = args.harness_args[1:]
    run(args)


if __name__ == "__main__":
    main()
