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
        if tA != NOTRANS or tB != NOTRANS:
            if tA == CONJTRANS or tB == CONJTRANS:
                return False               # meaningless for a real type
            return batch >= 128 and 128 <= mx <= 512
        if mx <= 32:
            return True
        return 128 <= mx <= 512
    if t == "double":
        return mx <= 512
    return False


def main(path, check=False):
    rows = [r for r in csv.reader(open(path))
            if r and r[0] == "reached" and r[1] == "gemm"]
    if not rows:
        sys.exit(f"{path}: no gemm 'reached' rows -- was BATCHLAS_COVERAGE_OUT set, and were "
                 f"the per-pid shards merged?")

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
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], check="--check" in sys.argv)
