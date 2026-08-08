# SYEV plan — what actually landed, measured

Companion to `SYEV_PERF_IMPLEMENTATION_PLAN.md` (what to do) and
`SYEV_PLAN_BASELINES.md` (the before numbers and the gates). This is the after.

Same rig throughout: this worktree's `build-cuda`, device 1 of two RTX 4090s, one process,
idle GPU, clocks warmed, first run of a fresh process discarded. µs/matrix,
`--warmup=2 --min_iters=5` unless noted. The rig reproduces the research document's published
figures to within 0.5%, which is what licenses these comparisons.

## Summary

| WP | item | gate | measured | verdict |
|---|---|---|---|---|
| WP0 | harness unblockers | cfloat rows appear | rows appear | **passed** |
| WP1 | A2 — `stebz` values-only | ≥ 1.20× float n=256 | **1.44×** | **passed, exceeds estimate** |
| WP4 | A3 — per-type tile/subs | ≥ 1.10× cfloat, float unchanged | **1.139×**, float unchanged | **passed** |
| WP3 | A1 — `her2k` trailing update | ≥ 1.05× cfloat | 1.043× | **marginal miss** |
| WP2 | B3 — `cta-large-n` | ≥ 1.15× vs blocked | **0.012× (85× slower)** | **rejected** |

Build: clean, exit 0, no new warnings. Tests: `syev_blocked`, `sytrd_blocked`,
`sytrd_sb2st`, `sytrd_sb2st_hh` all pass (145 s).

## WP1 (A2) — the largest win, and it beat its own estimate

Values-only, `provider=blocked`, batch 1024:

| n | float before | float after | | cfloat before | cfloat after | |
|---|---|---|---|---|---|---|
| 64 | 1.1614 | 0.62228 | **1.87×** | 1.5055 | 1.0465 | **1.44×** |
| 128 | 4.5231 | 2.9285 | **1.54×** | 7.3078 | 5.6837 | **1.29×** |
| 192 | 11.566 | 7.6783 | **1.51×** | 21.486 | 16.834 | **1.28×** |
| 256 | 20.774 | **14.476** | **1.44×** | 47.218 | **38.250** | **1.23×** |
| 320 | 42.804 | 31.438 | **1.36×** | 101.45 | 87.666 | **1.16×** |

The gate was ≥ 1.20× on float at n = 256; the plan estimated "up to 1.35×". Measured 1.44×,
and 1.87× at n = 64 where the discarded eigenvector solve is a larger share still.

**The win is where the theory said it is, which is the part worth checking.** stedc runs in
real arithmetic, so its absolute cost is identical for both scalar types — it is 28.3% of the
float solve but only ~12% of the cfloat solve at n = 256. Float should therefore gain
substantially more than cfloat, and it does: 1.44× against 1.23×, and the gap narrows
monotonically as n grows and the reduction dominates. A result where cfloat gained as much as
float would have meant the saving was coming from somewhere unintended.

## WP4 (A3) — the per-type design, empirically justified

`provider=two_stage`, eigenvectors, n = 512, batch 512:

| | before | after | |
|---|---|---|---|
| cfloat | 971.17 | **852.46** | **1.139×** |
| float | 334.28 | **331.25** | unchanged (within noise) |

Compare the global flip measured in `SYEV_PLAN_BASELINES.md`, where the same constants were
forced through the env knobs for both types: cfloat 852.77, float **384.30**. So the per-type
selection captures the entire complex win (852.46 vs 852.77 — identical) while float keeps its
own optimum instead of paying 1.15×. That is the package's whole thesis, and it is now measured
rather than argued.

## WP3 (A1) — a marginal miss, predicted in advance

`provider=blocked`, eigenvectors, n = 256, batch 1024:

| | before | after | |
|---|---|---|---|
| cfloat | 66.628 | **63.899** | **1.043×** |
| float | 31.657 | 31.037 | unchanged |

The gate was ≥ 1.05×, so this misses — by 0.7 of a percentage point.

It is a miss that was called in advance, and by measurement rather than hindsight. The
primitive A/B in `SYEV_PLAN_BASELINES.md` found `her2k` beating the GEMM pair by 1.32–1.33× at
the panel's shapes, but with a time almost independent of k — the signature of being bound by
the n₂² product-buffer traffic rather than by arithmetic. Propagating 1.33× through the
measured phase share predicted **~1.04% end to end**; the solver delivered 1.043×. The plan's
1.05–1.12× estimate was optimistic at its lower bound.

**Recommendation: keep it.** It is a real 1.04×, it costs nothing at runtime, float is
untouched, and the guard work it forced (single-definition route prediction in
`expansion_budget.hh`) is worth having on its own. But the gate should be restated at 1.04×
rather than quietly declared met.

## WP2 (B3) — rejected, and it was the plan's own top-ranked item

Forced `provider=cta` against `blocked`, float, eigenvectors, batch 256:

| n | CTA | blocked | |
|---|---|---|---|
| 33 | 150.17 | 1.7715 | **85× slower** |
| 64 | 760.68 | 3.6021 | **211× slower** |

The gate was ≥ 1.15× *faster*. This is not a near miss and it worsens with n. At batch 2048 a
single sweep over n = 33..128 did not complete one measured iteration in ten minutes while
holding 24 GB, so the shipped sweep was abandoned for these two points.

This matters beyond the package. `SYEV_PERF_RESEARCH.md` ranked B3 **first** in its suggested
order — *"the highest value-per-hour item in Tier B, because the code already exists"* — on the
strength of a complete, tested, unmerged implementation with, as that document itself noted,
*"no performance measurement at all"*. The implementation cost was indeed sunk. The value was
not there. A CTA-resident solve skips ~15 kernel launches, but at n ≥ 33 it is running an
unblocked, level-2 algorithm in one work-group per matrix against a blocked level-3 pipeline,
and that trade is catastrophic long before shared memory runs out.

**Recommendation: do not merge the routing lift.** The port itself is worth keeping on a branch
for the sub-group→work-group partition refactor it contains, which is independently useful at
n ≤ 32. Reorder the plan to drop B3 out of first place.

## WP0 — the unblockers work

`sb2st_hh_benchmark --type=cfloat` now produces rows where it silently produced none
(n = 512, batch 512, kd = 32):

| | float | cfloat | ratio |
|---|---|---|---|
| CHASE | 56.578 | 267.70 | 4.73× |
| BACK | 99.095 | 317.03 | 3.20× |

The CHASE ratio matches the 4.85× the research document measured through the full solver. The
BACK ratio is 3.20× rather than 3.90× **because A3 is active in this build** — the complex
back-transform is exactly what WP4 sped up. The float BACK row (99.095) also matches the
≈101 µs/matrix the WP0 implementer derived when it rejected the plan's stated accept criterion
as a category error (972 µs/matrix is a whole two-stage solve, not a back-transform row). Both
cross-checks land.

The `latrd` grid escape hatch (`BATCHLAS_LATRD_GRID_FORCE_UNSAFE`) is in but **not yet
exercised** — it is deadlock-capable by construction and belongs with the A5 measurement, not
with this batch.

## WP2 is excluded from the PR, and why that is not just the perf result

The B3 port is reverted out of this branch (`git revert -m 1` of its merge). It survives on
`worktree-wf_3b4af334-426-9` with the measurements above.

The performance rejection alone would not force that — a dead forced-provider path is
harmless. The deciding fact is different: the port also **rewrites the n ≤ 32 CTA kernels**
(`sytrd_cta.cc`, `ormqr_cta.cc`, `syev_cta.cc`, and the sub-group→work-group partition
machinery in `sg_compat.hh`), and `Auto` *does* route there — `cta` owns cfloat at n = 9..32.
Its tests pass, but its effect on that live route was never measured. Carrying an unmeasured
change to a routed path, in service of a feature measured as 85–211× slower, is not a trade
worth making. If the partition refactor is wanted for its own sake it should come back as its
own change, with n ≤ 32 numbers attached.

## An incremental build is not trustworthy across this revert

Recorded because it cost an hour and would cost it again.

Reverting WP2 removed a member from `DeviceCaps` (`include/blas/dispatch/context.hh`), which
main does not have and the port added. `cmake --build` reported success and the test suites
passed, but `two_stage` then **segfaulted for both scalar types at n ≥ 256**, in
`~DeviceCaps` freeing a `std::string` at the wrong offset — objects built against the two
struct layouts had been mixed. A stale-object ODR violation, not a code defect:

```
#0  __GI___libc_free (mem=0x18c00)
#7  ~DeviceCaps () at include/blas/dispatch/context.hh:11
#8  ormqr_buffer_size_dispatch<CUDA, complex<float>>
#11 syev_two_stage_buffer_size<CUDA, complex<float>>
```

`cmake --build --target clean` plus a full rebuild fixed it, and every figure in this document
was then re-measured on that clean build:

| | pre-revert | clean rebuild | |
|---|---|---|---|
| WP1 float values n=256 | 14.476 | **14.463** | reproduces |
| WP1 cfloat values n=256 | 38.250 | **38.358** | reproduces |
| WP3 cfloat vectors n=256 | 63.899 | **63.889** | reproduces |
| WP4 cfloat two-stage n=512 | 852.46 | **852.30** | reproduces |
| WP4 float two-stage n=512 | 331.25 | **334.50** | baseline 334.28 — unchanged |

Two lessons. First, after reverting a merge that changed a struct in a widely-included header,
rebuild clean before believing anything — including the tests, which passed against the broken
library. Second, the intermediate spot-check that read float values-only at 15.301 instead of
14.463 was an artifact of that same broken build; a 6% anomaly is exactly the size that gets
rationalised as thermal drift rather than investigated. It was worth chasing.

Note also that benchmark targets here are `EXCLUDE_FROM_ALL`: after `--target clean`, a plain
`cmake --build` rebuilds the tests but leaves no benchmark binaries, and the next measurement
fails with "No such file or directory" rather than anything informative.

## What this changes in the plan

1. **B3 leaves the front of the queue** and becomes a recorded negative result. The plan's
   sequence was built on it being cheap *and* promising; it was only cheap.
2. **A2 is the best item in Tier A by a wide margin** — 1.36–1.87× across the whole values-mode
   region for float, not the "up to 1.35×" estimated.
3. **A1's expected value is 1.04×**, not 1.05–1.12×, and it is capped by product-buffer traffic
   rather than by arithmetic, so a larger `nb` will not improve it.
4. **A3 and B2 remain non-additive**, and A3 has now taken its share: any B2 estimate must be
   measured against 852, not 971.
5. Unchanged and still the biggest item: **B1**, whose counter premise was re-verified here
   (L1TEX 52,499 MB to the byte, 2.7× headroom) and is not affected by any of the above.
