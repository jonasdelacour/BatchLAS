#!/usr/bin/env python3
"""How much of the real COMPLEX gemm population could a routing relaxation reach?

Reads build/coverage.csv (the whole-test-suite capture). Counts complex gemm
calls only, and classifies each against the three things that stand between it
and Tiled64x64RegisterK16Wide:

  transposed      -- the kernel is NN-only; a router cannot fix this
  min_dim < 256   -- the ROUTING floor at gemm_kernels.cc:632
  extents ragged  -- m%64 or n%64 or k%16, which forces the PREDICATED leg
                     (reachable; the leg exists and is correct)

Alignment cannot be judged from coverage.csv (no ld), and for complex it is
nearly moot anyway: can_use_64x64_k16_wide_fast_path's VecLen is 16/sizeof(T),
i.e. 2 for complex<float> and 1 for complex<double>, so the ld/stride tests are
%2 and %1.
"""
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "..", "build", "coverage.csv")

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["op"] != "gemm":
            continue
        if "complex" not in r["scalar"]:
            continue
        rows.append(r)

# route_gemm_equivalence_tests feeds synthetic probe rows straight to the
# resolver; they are recorded as kind != "reached".
kinds = collections.Counter(r["kind"] for r in rows)
real = [r for r in rows if r["kind"] == "reached"]

tot = sum(int(r["calls"]) for r in real)
print(f"kinds: {dict(kinds)}")
print(f"complex gemm call sites (reached): {len(real)}, calls: {tot}")


def cls(r):
    m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
    tr = r["transA"] != "0" or r["transB"] != "0"
    return m, n, k, tr


buckets = collections.Counter()
big = collections.Counter()
for r in real:
    m, n, k, tr = cls(r)
    c = int(r["calls"])
    if tr:
        buckets["transposed (kernel cannot serve)"] += c
        continue
    if min(m, n, k) >= 256 and m % 64 == 0 and n % 64 == 0 and k % 16 == 0:
        buckets["already routed to wide today"] += c
        continue
    ragged = (m % 64) or (n % 64) or (k % 16)
    floor = min(m, n, k) < 256
    if floor and ragged:
        buckets["NN, blocked by BOTH floor and ragged extents"] += c
    elif floor:
        buckets["NN, blocked by the min_dim floor ALONE"] += c
    else:
        buckets["NN, blocked by ragged extents ALONE"] += c
    if max(m, n) >= 128:
        big["non-small NN reachable by relaxation"] += c

for kk, v in buckets.most_common():
    print(f"  {v:8d}  {kk}")
print()
for kk, v in big.most_common():
    print(f"  {v:8d}  {kk}")

print("\ntop NN non-small complex shapes a relaxation would newly capture:")
agg = collections.Counter()
for r in real:
    m, n, k, tr = cls(r)
    if tr or max(m, n) < 128:
        continue
    if min(m, n, k) >= 256 and m % 64 == 0 and n % 64 == 0 and k % 16 == 0:
        continue
    agg[(m, n, k, r["scalar"])] += int(r["calls"])
for (m, n, k, s), v in agg.most_common(20):
    print(f"  {v:8d}  {m}x{n}x{k}  {s}")
