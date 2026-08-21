#!/usr/bin/env python3
"""Re-derive potrf_cta_max_n<T>() from the MEASURED SLM ceiling.

Formula: WP4_POTRF_SPEC.md section 4.1, plus the W9 fix (off[] has no term in the
spec's formula; WP4_POTRF_SPEC_CORRECTIONS.md:515 says add 4*ceil_div(n-nb, TS)).

    LDA            = n | 1
    slm_per_matrix = LDA*n*sizeof(T) + NB*sizeof(real_t) + 64
    slm_per_wg     = G * slm_per_matrix + off_bytes          # off[] hoisted per WG
    off_bytes      = 4 * ceil_div(n - nb, TS)                # W9

off[] is placed PER WORK-GROUP, not per matrix: it depends only on (m2, TS) and
m2 = n - j - ib is work-group-uniform, so all G matrices in a work-group decode
the same table (Open question 7 in the corrections doc). At G = 1 -- which is what
potrf_cta_max_n<T>() is derived at -- the two placements are numerically identical;
the choice only matters for the packed small-n case.

Constants: spec section 3.1 (spec:177). Note cdouble NB = 8, not 16 -- spec:278's
own spot-check used 16 and is wrong (corrections doc, W9 closing paragraph).

Measured hardware facts (this file's whole reason to exist), all on GPU 0,
RTX 4090 / sm_89:
  sycl local_mem_size                      = 101376
  cudaDeviceProp.sharedMemPerBlockOptin    = 101376
  cudaDeviceProp.sharedMemPerMultiprocessor= 102400
  cudaDeviceProp.reservedSharedMemPerBlock = 1024
  hard cap on (static + dynamic) per block = 101376        [bisected]
  blocks/SM limited by shared             = floor(102400 / (dyn + static + 1024))
                                             [matches ncu on all 11 sizes probed]
"""
import math

SM_POOL   = 102400      # cudaDeviceProp.sharedMemPerMultiprocessor
RESERVED  = 1024        # cudaDeviceProp.reservedSharedMemPerBlock
HARD_CAP  = 101376      # measured: largest static+dynamic that launches
STATIC    = 256         # measured static shared of a kernel using reduce_over_group

TYPES = [
    # name,           sizeof(T), sizeof(real_t), NB, TS
    ("float",              4, 4, 16, 4),
    ("double",             8, 8, 16, 4),
    ("complex<float>",     8, 4, 16, 4),
    ("complex<double>",   16, 8,  8, 2),
]

def cdiv(a, b): return -(-a // b)

def slm(n, szT, szR, NB, TS, G=1, with_off=True):
    lda = n | 1
    per_matrix = lda * n * szT + NB * szR + 64
    off = 4 * cdiv(max(n - NB, 0), TS) if with_off else 0
    return G * per_matrix + off

def max_n(budget, szT, szR, NB, TS, G=1, with_off=True):
    n = 0
    while slm(n + 1, szT, szR, NB, TS, G, with_off) <= budget:
        n += 1
        if n > 4096: break
    return n

def blocks_per_sm(dyn, static=STATIC):
    return SM_POOL // (dyn + static + RESERVED)

# Budget candidates.
#   The "-4096" reserve is the project's own convention
#   (cmake/BatchLASDetectSYCL.cmake:57-67, batchlas_subgroup_workspace_budget_bytes).
occ_budget = lambda k: SM_POOL // k - RESERVED - STATIC   # largest dyn with >= k blocks/SM
BUDGETS = [
    ("45056  spec  (hardcoded 49152 - 4096)",            45056),
    ("49920  >=2 blocks/SM  (measured occupancy)",       occ_budget(2)),
    ("97280  runtime 101376 - 4096 (corrections doc)",   97280),
    ("100352 runtime - 1024 static reserve",             HARD_CAP - 1024),
    ("101120 measured ceiling at static=256",            HARD_CAP - STATIC),
    ("32853  >=3 blocks/SM",                             occ_budget(3)),
    ("24320  >=4 blocks/SM",                             occ_budget(4)),
    ("19200  >=5 blocks/SM",                             occ_budget(5)),
    ("15786  >=6 blocks/SM",                             occ_budget(6)),
    ("11520  >=8 blocks/SM",                             occ_budget(8)),
]

# The launch HOLE, measured: a dynamic request in (49152 - static, 49152] neither
# fits the 48 KB non-opt-in limit nor triggers the CUDA opt-in in the UR adapter.
HOLE_HI = 49152
def in_hole(dyn, static=STATIC):
    return HOLE_HI - static < dyn <= HOLE_HI

print("== occupancy consequence: blocks/SM = floor(102400 / (dyn + 256 + 1024)) ==")
print(f"{'budget':>8} {'blocks/SM':>10} {'warps/SM at wg=128':>20}")
for b in sorted({b for _, b in BUDGETS}):
    k = blocks_per_sm(b)
    print(f"{b:8d} {k:10d} {k*4:20d}")
print()

for label, budget in BUDGETS:
    print(f"== budget {label}  ->  blocks/SM = {blocks_per_sm(budget)} ==")
    for name, szT, szR, NB, TS in TYPES:
        n  = max_n(budget, szT, szR, NB, TS)
        s0 = slm(n,     szT, szR, NB, TS)
        s1 = slm(n + 1, szT, szR, NB, TS)
        lda, lda1 = n | 1, (n + 1) | 1
        off0 = 4 * cdiv(max(n - NB, 0), TS)
        off1 = 4 * cdiv(max(n + 1 - NB, 0), TS)
        hole = " *** IN LAUNCH HOLE ***" if in_hole(s0) else ""
        print(f"  {name:16s} max_n = {n:4d}   "
              f"fits: {lda}*{n}*{szT} + {NB}*{szR} + 64 + {off0} = {s0} <= {budget}"
              f"   | first miss: n={n+1} -> {lda1}*{n+1}*{szT} + {NB}*{szR} + 64 + {off1} = {s1} > {budget}"
              f"   | blocks/SM = {blocks_per_sm(s0)}{hole}")
    print()

# How much does the W9 off[] term actually cost in n?
print("== cost of the W9 off[] term, at budget 97280 ==")
for name, szT, szR, NB, TS in TYPES:
    a = max_n(97280, szT, szR, NB, TS, with_off=True)
    b = max_n(97280, szT, szR, NB, TS, with_off=False)
    print(f"  {name:16s} with off[] = {a}, without = {b}  (delta {b-a})")
print()

# Does any n at or below the big fit ceilings land in the launch hole?
print("== launch-hole audit: every n whose slm request lands in (48896, 49152] ==")
for name, szT, szR, NB, TS in TYPES:
    bad = [n for n in range(1, 400) if in_hole(slm(n, szT, szR, NB, TS))]
    print(f"  {name:16s} {bad if bad else 'none'}")

# The hole's position depends on the potrf kernel's OWN static shared, which is not
# known until the kernel is written. Conservative audit over static in [0, 1024]:
# any n whose request lands in (48128, 49152] is at risk and must be padded.
print()
print("== conservative launch-hole audit, static shared unknown in [0,1024] ==")
print("   at-risk band for the local_accessor request: (48128, 49152]")
for name, szT, szR, NB, TS in TYPES:
    bad = [(n, slm(n, szT, szR, NB, TS)) for n in range(1, 400)
           if 48128 < slm(n, szT, szR, NB, TS) <= 49152]
    print(f"  {name:16s} {bad if bad else 'none'}")
