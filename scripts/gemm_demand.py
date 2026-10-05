#!/usr/bin/env python3
"""What GEMM shapes does BatchLAS actually issue, and which gemm choice served them?

WHY THIS EXISTS. A routing change is judged by the calls the library ACTUALLY ISSUES, not by
the shape cross-product: sweeping cells nothing asks for measures capability, not demand.
The coverage instrument answers that exactly. Capture a run with scripts/route_diff.sh (or
any run with BATCHLAS_COVERAGE_OUT set, merged with scripts/coverage_merge.sh), then:

    scripts/gemm_demand.py .route-diff/<label>.csv [--minus=<probe capture>]

Since P3.4 (flat kernel selection) a gemm coverage row names the choice that ran: origin
`native` or `vendor`, and the choice spelling (`tiled`, `small`, `reg:m=128:n=128:k=8:u=1`,
`wide:m=64:n=64:k=16`, ...) in chosen_algo. Rows carry the key fields the table is keyed on
(src/ops/gemm/choice.hh: ta, tb, layout, m, n, k, batch), except layout.

This script used to carry a Python replica of the deleted router's gemm preferred() so it could
say what a Vendor->Auto flip "would" move. Both the predicate and the flip are gone: Auto
reads tuned/gemm.<dtype>.<arch>.txt, so "what Auto does" is the capture itself. To ask what a
different table would do, capture again under that table (or under BATCHLAS_GEMM_ROUTE=native
for the best native entry of every row) and diff the two with scripts/route_diff.sh. No
replica means nothing here can drift from the library.
"""

import csv
import sys
import collections

# enums.hh: NoTrans=0, Trans=1, ConjTrans=2
FORM = {"0": "N", "1": "T", "2": "C"}

COL = {name: i for i, name in enumerate(
    "kind op scalar backend shape_class m n k batch chosen_origin chosen_algo calls "
    "native_route_existed native_route_supported library uplo side diag transA transB".split())}


def _gemm_rows(path):
    return [r for r in csv.reader(open(path))
            if r and r[0] == "reached" and r[1] == "gemm"]


def _shape_key(r):
    g = lambda name: r[COL[name]]
    return (g("scalar"), g("m"), g("n"), g("k"), g("batch"), g("transA"), g("transB"))


def subtract_probes(rows, probe_path):
    """Remove a probe suite's rows from a full-suite capture.

    WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL FOR A DEMAND FIGURE. Test suites that sweep a
    synthetic shape cross-product through gemm (the deleted route_gemm_equivalence_tests.cc
    was one; gemm_candidates_tests pins every candidate on every form) are probes of what the
    selector CAN be asked, not demand for what the library ISSUES. On the 2026-08-19 capture
    they were 2312 of 2795 gemm rows and 71051 of 113526 calls, and counting them inverted
    conclusions (a gate looked like it fired on 3.56% of non-float calls; 0.64% without).

    Subtraction is per (scalar, m, n, k, batch, transA, transB) on the call COUNT, not
    row removal: a shape the probe suite touches may also be issued for real.
    """
    probe = collections.Counter()
    for r in _gemm_rows(probe_path):
        probe[_shape_key(r)] += int(r[COL["calls"]] or 0)

    out, dropped_rows, dropped_calls = [], 0, 0
    for r in rows:
        key = _shape_key(r)
        calls = int(r[COL["calls"]] or 0)
        take = calls - probe.get(key, 0)
        if take <= 0:
            dropped_rows += 1
            dropped_calls += calls
            continue
        dropped_calls += calls - take
        r = list(r)
        r[COL["calls"]] = str(take)
        out.append(r)
    print(f"subtracted probe capture {probe_path}: dropped {dropped_rows} rows / "
          f"{dropped_calls} calls, {len(out)} rows remain")
    if not out:
        sys.exit("every row was accounted for by the probe capture -- that is not a "
                 "demand table, check that the two captures are from different runs")
    return out


def _family(algo):
    return algo.split(":", 1)[0] if algo else "?"


def main(path, minus=None):
    rows = _gemm_rows(path)
    if not rows:
        sys.exit(f"{path}: no gemm 'reached' rows -- was BATCHLAS_COVERAGE_OUT set, and were "
                 f"the per-pid shards merged?")
    if minus:
        rows = subtract_probes(rows, minus)

    g = lambda r, name: r[COL[name]]
    calls_of = lambda r: int(g(r, "calls") or 0)
    by_type = collections.Counter()
    by_choice = collections.Counter()
    by_family = collections.Counter()
    by_form = collections.Counter()
    total_calls = 0
    for r in rows:
        c = calls_of(r)
        total_calls += c
        t, origin, algo = g(r, "scalar"), g(r, "chosen_origin"), g(r, "chosen_algo")
        by_type[t] += c
        by_choice[(t, origin, algo)] += c
        by_family[(t, origin, _family(algo))] += c
        form = FORM.get(g(r, "transA"), "?") + FORM.get(g(r, "transB"), "?")
        by_form[(t, form, origin)] += c

    total = len(rows)
    print(f"gemm coverage rows: {total}   (calls: {total_calls})")
    print("calls by type:", dict(by_type))

    print("\nchoice actually taken (call-weighted -- rows are not calls, and the")
    print("distinction matters: a hot shape and a one-off cost the same one row):")
    for kk, v in sorted(by_choice.items()):
        print(f"   {kk[0]:>16}  {kk[1]:<6} {kk[2]:<28} {v}")

    print("\nby family:")
    for kk, v in sorted(by_family.items()):
        print(f"   {kk[0]:>16}  {kk[1]:<6} {kk[2]:<8} {v}")

    print("\nby transpose form (panel updates are mostly transposed):")
    for kk, v in sorted(by_form.items()):
        print(f"   {kk[0]:>16}  {kk[1]}  {kk[2]:<6} {v}")

    native_calls = sum(v for kk, v in by_choice.items() if kk[1] == "native")
    print(f"\nserved native: {native_calls} / {total_calls} calls "
          f"({100.0 * native_calls / max(total_calls, 1):.1f}%)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    if "--check" in sys.argv:
        sys.exit("--check re-derived the deleted router's gemm preferred() (gone since P3.4); "
                 "diff two captures with scripts/route_diff.sh instead")
    minus = None
    for a in sys.argv[1:]:
        if a.startswith("--minus="):
            minus = a.split("=", 1)[1]
    main(args[0], minus=minus)
