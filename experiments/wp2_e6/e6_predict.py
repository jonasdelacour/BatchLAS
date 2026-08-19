#!/usr/bin/env python3
"""Predict, BEFORE flipping, exactly which decisions E6 should move.

The route-diff discipline is that a routing step enumerates its intended moves in advance
and the diff must match them line for line. Anything the diff shows that is not in this
list is a surprise, and a surprise in a routing change is the thing to catch.

Post-E3/E4 preferred() accepts, for an unforced call on a GPU:
  double : square, batch >= 64, max_dim <= 512
  float  : square, batch >= 64, NN, max_dim <= 32
  complex: never
  heterogeneous batch: never
"""
import csv, sys, collections

COL = {n: i for i, n in enumerate(
    "kind op scalar backend shape_class m n k batch chosen_origin chosen_algo calls "
    "native_route_existed native_route_supported library uplo side diag transA transB".split())}
NOTRANS = 0


def preferred(t, m, n, k, batch, tA, tB):
    if not (m > 0 and n > 0 and k > 0):
        return False
    if t.startswith('complex'):
        return False
    if not (m == n == k) or batch < 64:
        return False
    mx = max(m, n, k)
    if t == 'float':
        if tA != NOTRANS or tB != NOTRANS:
            return False
        return mx <= 32
    if t == 'double':
        return mx <= 512
    return False


moves = collections.Counter()
rows = 0
for r in csv.reader(open(sys.argv[1])):
    if not r or r[0] != 'reached' or len(r) <= COL['transB'] or r[COL['op']] != 'gemm':
        continue
    rows += 1
    try:
        m, n, k, b = (int(r[COL[x]]) for x in ('m', 'n', 'k', 'batch'))
        tA, tB = int(r[COL['transA']] or 0), int(r[COL['transB']] or 0)
    except ValueError:
        continue
    t = r[COL['scalar']]
    origin = r[COL['chosen_origin']]
    if origin != 'vendor':
        continue
    # The capture's rows include forced calls; a forced Vendor stays Vendor after
    # the flip, so this is an UPPER BOUND on the moves, not an exact prediction.
    if preferred(t, m, n, k, b, tA, tB):
        moves[(t, r[COL['shape_class']], tA, tB)] += 1

print("gemm reached rows in capture: %d" % rows)
print("decisions that COULD move vendor -> native after the flip: %d distinct" % len(moves))
for (t, sc, tA, tB), c in sorted(moves.items()):
    print("  %-16s shape_class=%-6s transA=%d transB=%d   (%d rows)" % (t, sc, tA, tB, c))
