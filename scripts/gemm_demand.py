#!/usr/bin/env python3
"""What GEMM shapes does BatchLAS actually issue, and what would routing changes move?

WHY THIS EXISTS. WP2 widens the GEMM routing window, and the plan's stated deliverable is to
flip BATCHLAS_GEMM_VARIANT's default from Vendor to Auto. Both are changes to `preferred()`,
and the honest question about either is not "is the kernel fast?" but "how many calls that
the library ACTUALLY ISSUES would move off cuBLAS, and which ones?".

The coverage instrument answers that exactly. Capture a run with scripts/route_diff.sh (or
any run with BATCHLAS_COVERAGE_OUT set, merged with scripts/coverage_merge.sh), then:

    scripts/gemm_demand.py .route-diff/<label>.csv

This is deliberately a *demand* table, not a *capability* table. Sweeping the shape
cross-product measures cells nothing asks for; this measures the cells the library issues.

The preferred() transcription below MUST be kept in step with
include/batchlas/blas/dispatch/route_gemm.hh. It is a replica, and a replica that drifts is
worse than no replica -- see tests/route_gemm_equivalence_tests.cc, which exists because an
earlier transcription of this same predicate drifted. --check re-derives the verdict for rows
that actually took a route and reports any disagreement, which is the cheap guard against
that drift.
"""

import csv
import sys
import collections

# enums.hh: NoTrans=0, Trans=1, ConjTrans=2
NOTRANS, TRANS, CONJTRANS = 0, 1, 2

COL = {name: i for i, name in enumerate(
    "kind op scalar backend shape_class m n k batch chosen_origin chosen_algo calls "
    "native_route_existed native_route_supported library uplo side diag transA transB".split())}


def supports(m, n, k, heterogeneous=False):
    """Transcription of RouteTable<Op::gemm, T>::supports() for the Native origin.

    preferred() calls supports() FIRST (route_gemm.hh:80). Omitting that guard here
    over-counted by four degenerate m=n=k=0 rows -- a small error, but the kind that
    makes a demand figure quietly wrong, which is the one thing this table must not be.
    Heterogeneous rows are not distinguishable in the CSV, so this reports the
    homogeneous answer; the coverage rows for heterogeneous calls take the vendor
    anyway.
    """
    return m > 0 and n > 0 and k > 0 and not heterogeneous


def preferred(t, m, n, k, batch, tA, tB):
    """Transcription of RouteTable<Op::gemm, T>::preferred() -- keep in step with the header."""
    if not supports(m, n, k):
        return False
    if t.startswith("complex"):
        return False                       # rejected outright
    if not (m == n == k) or batch < 64:    # square only, enough batch to fill the device
        return False
    mx = max(m, n, k)
    if t == "float":
        # WP2 E4 narrowed float to NN and max_dim <= 32. The transposed window
        # (0.34-0.55x of cuBLAS) and the 128..512 NN window (0.40-0.98x) were
        # both measured losses; see docs/perf/gemm.md#float-nn-at-max_dim-32.
        if tA != NOTRANS or tB != NOTRANS:
            return False
        return mx <= 32
    if t == "double":
        return mx <= 512
    return False


def _gemm_rows(path):
    return [r for r in csv.reader(open(path))
            if r and r[0] == "reached" and r[1] == "gemm"]


def _shape_key(r):
    g = lambda name: r[COL[name]]
    return (g("scalar"), g("m"), g("n"), g("k"), g("batch"), g("transA"), g("transB"))


def subtract_probes(rows, probe_path):
    """Remove a probe suite's rows from a full-suite capture.

    WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL FOR A DEMAND FIGURE.

    tests/route_gemm_equivalence_tests.cc pins routing against a replica of the legacy
    behaviour by sweeping a synthetic `dims[] x batches[]` cross-product (lines 119-120)
    straight through resolve_gemm_route. Those calls never execute a GEMM. They are
    probes of what the resolver CAN be asked, not demand for what the library ISSUES --
    and they are the large majority of the table: 2312 of 2795 gemm rows, 71051 of
    113526 calls, on the 2026-08-19 capture.

    Counting them does not merely add noise, it inverts conclusions. The wide-scalar
    kernel's routing gate looks like it fires on 3.56% of non-float calls with the
    probes in and 0.64% with them out, and every probe hit is a large square aligned
    shape -- exactly the cells a new tile wants to claim credit for.

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


def main(path, check=False, minus=None):
    rows = _gemm_rows(path)
    if not rows:
        sys.exit(f"{path}: no gemm 'reached' rows -- was BATCHLAS_COVERAGE_OUT set, and were "
                 f"the per-pid shards merged?")
    if minus:
        rows = subtract_probes(rows, minus)

    g = lambda r, name: r[COL[name]]
    by_type = collections.Counter(g(r, "scalar") for r in rows)
    by_route = collections.Counter((g(r, "scalar"), g(r, "chosen_origin"), g(r, "chosen_algo"))
                                   for r in rows)

    would, would_calls, cells, disagree = 0, 0, collections.Counter(), []
    total_calls = 0
    for r in rows:
        t = g(r, "scalar")
        m, n, k = int(g(r, "m")), int(g(r, "n")), int(g(r, "k"))
        b = int(g(r, "batch"))
        tA, tB = int(g(r, "transA")), int(g(r, "transB"))
        calls = int(g(r, "calls") or 0)
        total_calls += calls
        p = preferred(t, m, n, k, b, tA, tB)
        if p:
            would += 1
            would_calls += calls
            cells[(t, "square" if m == n == k else "non-square",
                   f"max_dim={max(m,n,k)}", f"batch={b}")] += calls
        # A row that took native under an UNFORCED call should have been preferred.
        # Forced routes bypass preferred(), so a mismatch here is a hint, not a proof.
        if check and g(r, "chosen_origin") == "native" and not p:
            disagree.append((t, m, n, k, b, tA, tB))

    total = len(rows)
    print(f"gemm coverage rows: {total}   (calls: {total_calls})")
    print("by type:", dict(by_type))
    print("\nroute actually taken:")
    for kk, v in sorted(by_route.items()):
        print(f"   {kk[0]:>16}  {kk[1]}/{kk[2]:<16} {v}")

    native_now = sum(v for kk, v in by_route.items() if kk[1] == "native")
    print(f"\nrouted native today: {native_now} / {total} ({100.0*native_now/total:.1f}%)")
    print(f"preferred() would accept: {would} rows / {would_calls} calls "
          f"({100.0*would/total:.1f}% of rows, {100.0*would_calls/max(total_calls,1):.1f}% of calls)")
    print(f"  -> flipping the unset default from Vendor to Auto moves ~{would - native_now} "
          f"rows off cuBLAS")

    print("\nwhat would move, by type (call-weighted -- rows are not calls, and the")
    print("distinction matters: a hot shape and a one-off cost the same one row):")
    mv = collections.Counter()
    for kk, v in cells.items():
        mv[kk[0]] += v
    for t, v in sorted(mv.items()):
        print(f"   {t:>16} {v}")

    if check:
        print(f"\nrows that took native but preferred() rejects: {len(disagree)}")
        print("   (expected non-zero: the forced-variant tests bypass preferred())")
        for d in disagree[:8]:
            print("   ", d)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    minus = None
    for a in sys.argv[1:]:
        if a.startswith("--minus="):
            minus = a.split("=", 1)[1]
    main(args[0], check="--check" in sys.argv, minus=minus)
