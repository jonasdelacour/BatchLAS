# PlanTree prototype: potrf selection as a plan value

This is a prototype of the "PlanTree" design. `plan_op()` builds a `plan::Node`, and
`Exec<B, Tier>::run` executes that node. It covers potrf only, and it runs next to the
shipping path in the same library:

- `BATCHLAS_POTRF_PLANNER=1`, or `plan::set_potrf_planner(1)`, routes `potrf` and
  `potrf_buffer_size` through the new path.
- With the planner off, the library behaves exactly as before.

Machine: threadripper02, RTX PRO 6000 Blackwell (sm_120, 188 CUs), GPU 2. Build: dpcpp-cuda
Release, `nvptx64-nvidia-cuda` sm_120. Every result below exists for two builds: the vendor build
and a real `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` build.

## Code

| file | role |
|---|---|
| `src/plan/plan.hh` | Core types, all SYCL-free: Facts, Legality, Candidate, Decision, Node, Request, the TierDescriptor concept, TierList (`each`, `visit` by name, `owner_index`/`owner_name` per Route), the strategies, and the ABI tag macro |
| `src/plan/planner.hh` | The one generic `plan_op`, `plan_tier`, CostModelChooser, WindowPolicy (two WindowChoosers), FirstLegalChooser, warn-once, `explain()` |
| `src/plan/potrf_tiers.hh` | The descriptors Tiny, Cta, **CtaWg** (the extension; `owns_route = false`), LPanel, Blocked and Vendor. Also the hand windows as `WindowRow` data, the profile looked up by tier name, candidacy (`measured`), `selectable`, and `plan_trsm`/`plan_gemm` for Blocked's children |
| `src/plan/potrf_exec.hh` | `Exec<B, Tier>`, which has no primary template, plus bind, the covering size, facts, request, the per-thread memo in front of the cache, `potrf_planned`, and `potrf_planned_buffer_size` |
| `src/extensions/potrf_{cta,blocked}.cc` | `potrf_cta_dispatch_planned(geometry)`, `potrf_blocked_{dispatch,buffer_size}_planned(BlockedParams)`. They execute the node's geometry and parameters and do not re-derive them |
| `src/extensions/potrf_{cta,lpanel,tiny}.cc` | The launch record: at each submit site, what the kernel actually launched (`PotrfLaunchRecord`, off by default) |
| `src/extensions/potrf_launch_plan.hh` | `cta_plan_from(geometry)` and `cta_wg_geometry` (the extension) |
| `src/dispatch/entry_points/factorization.cc` | The plan cache (16 shards, a shared_mutex each, FIFO-bounded), the plan log, and the switch in `potrf`/`potrf_buffer_size` |
| `tests/potrf_planner_tests.cc` | Equivalence, tier-order liveness, sizing, execution, plan == launch, pins, EXPLAIN/cache, bounds, the extension, and the candidacy net |
| `mutants.sh` | Deliberate breaks, one per axis, plus the missing-Exec compile check |

## Design decisions made in the repair round

**Sizing contract: main's, kept.** `potrf_buffer_size` reports the vendor's size and, when the
chosen tree is native, the max over the bound plans of every legal native tier.
`options.hh` reads the route pin twice, once for the size query and once for the call, so
a caller whose environment changes in between must not under-allocate. Each of those plans is
exact, because Blocked's W is clamped to n - nb, so the 512 MiB over-estimate is still gone.
`potrf()` itself requires only the chosen tree's `workspace()`.
`PotrfBlockedTest.BufferSizeCoversEverySupportedNativeTier` now measures the blocked call's
real draw. With the planner on, that is the clamped W, re-derived from `blocked_params` in the
test. It no longer uses the unclamped kernel query.

**Tier identity is the name.** Routes may be shared, and each shared Route has exactly one
owner (`static_assert`). A route pin, the shipping resolver's answer and a coverage row all mean
the owner. The grid compares tier names.

**Candidacy is enforced in every chooser.** `Candidate::selectable` (not in `kUnmeasured`) is
read by CostModel (both the priced pick and its first-legal fallback), by both WindowChoosers,
and by FirstLegal. `candidate` (measured against the vendor) is deliberately NOT required in the
native fallback. Blocked and CTA are not measured against the vendor for most dtypes, yet they are
the only native servants for large n. Requiring it would make those shapes unservable
vendor-free (AGENTS.md s9: supports != preferred).

**ODR isolation.** Every planner body whose meaning a macro can change lives in
`inline namespace BATCHLAS_PLAN_ABI`, named from `BATCHLAS_PLAN_MUTANT` and the drop-Exec mock,
for example `abi_m0_x0`. `libbatchlas_backends.so` exports 346 `abi_m0_x0` symbols, and the mutant-1 test object
defines 331 `abi_m1_x0` symbols and no `abi_m0_x0` ones. The cache key carries the tag too, so a
mutant TU never reads the library's plans.

## Results (all on GPU 2)

**Equivalence.** The cells are the 570 distinct (dtype, n, batch) cells of
`sm120_potrf_sweep{,_edges}.jsonl`. Each cell is run as Lower and as Upper, giving 1140 cells. Each
cell is run in 16 modes: two profiles × {auto, vendor-free walk, 6 pins}. For every comparison,
the planned TIER NAME must equal the owner of `backend::potrf_route`'s Route, and the Route must
match too. The result is 0 mismatches for the library's node and 0 for the test TU's own
instantiation. This holds in the vendor build and in the vendor-free build alike.

**Tier order is not live** for a tier that shares a route or is unmeasured.
`TierOrderOfASharedOrUnmeasuredTierIsNotLive` plans every cell with CtaWg placed first and
placed last. It covers auto, a bare `native` pin and `native:cta`, with and without the vendor.
That is 13,680 plans: 0 differ from the canonical order, and 0 pick `native:cta_wg`.

**Sizing for float b=8192** (old is main's max over tiers, at the unclamped W):

| shape | old (B) | new (B) | the chosen tree alone (B) | chosen |
|---|---|---|---|---|
| n=36 | 537,100,288 | 753,664 | 32,768 | CTA |
| n=128 | 537,100,288 | 134,447,104 | 32,768 | LPanel |
| n=36 pinned `native:blocked` | 537,100,288 | 753,664 | 753,664 | Blocked |
| float n=640 b=512 | 33,568,768 | 33,568,768 | 33,568,768 | Blocked |
| cdouble n=20 b=8192 | 229,376 | 229,376 | 32,768 | LPanel |

The "new" figure is the covering size. It equals an independent re-derivation from the kernels'
own size queries in every row. Across the sweep, sizes are never larger than main's. In the vendor
build, 1071 cells are the same and 69 are smaller. In the vendor-free build, 775 are the same, 95
are smaller, and 270 are unservable cells where both paths throw `NoRouteError`.

**Plan == launch is a test now.** Each potrf kernel records what it actually launched at its
submit site: kernel, the instantiated N or NB, the scope, n, batch, the work-group size, the grid,
L and G. `PlanEqualsLaunch` runs each case through the public facade, which uses the library's
`Exec`, and through the test TU's `potrf_planned`, which uses the TU's `Exec`. It asserts the
record against the node. For Blocked, it checks that there are ceil(n/nb) leaf launches and that
the first one matches the planned leaf child. Cases: float n=16 (Tiny<16>), n=36 (CTA SubGroup
L=32 G=4), n=128 (LPanel<8>), n=128 with `BATCHLAS_POTRF_LPANEL_NB=16` (LPanel<16>), cdouble n=20
(LPanel<8> G=4), the cta_wg pin (WorkGroup L=64 G=1), cta pins at n=24 and n=60 (SubGroup and
WorkGroup), and float n=640 (Blocked, 5 leaves of WorkGroup L=256).

`BATCHLAS_POTRF_LPANEL_NB` is a new plan-level knob. Without it every LPanel plan carries the
default NB, so an Exec that dropped NB would launch the same kernel and no check could see it.

**Mutants** (`mutants.sh`). Each one rebuilds only the test TU. The library column (`mismatches=`)
stays 0 in every row, and only the TU column moves:

| mutant | result |
|---|---|
| 1: window edge 256→255 | tu-mismatches=4 (sm_89 auto, plus the tiny and cta pins), library 0 |
| 2: margin ignored | tu-mismatches 36/31/20/14/10 on sm_120, library 0 |
| 3: LPanel and Blocked swapped | green. This is the FirstLegal tie order, and no window or model path reaches it |
| 4: LPanel accepts Upper | tu-mismatches 202 (vendor-free and `native`) and 502 (the lpanel pin), library 0 |
| 5: CtaWg before Cta | green, as intended: the shared route resolves to its owner |
| 6: CtaWg selectable | `TierOrder...` red: 45 permuted plans pick `native:cta_wg` |
| 7: LPanel Exec ignores NB | `PlanEqualsLaunch` red, `[tu exec]` only: "panel width" at LPANEL_NB=16 |
| 8: Cta Exec re-derives scope | `PlanEqualsLaunch` red, `[tu exec]` only: scope, work-group size and grid at n=24 and n=36 |
| drop `Exec<B,CtaWg>` | compile error: the `static_assert` "a potrf tier has no Exec<B, Tier> specialisation" |

**Plan cache.**

- The cache has 16 shards, each a `shared_mutex` over a map plus a FIFO.
- `BATCHLAS_PLAN_CACHE_CAPACITY` sets the capacity; the default is 1024 entries.
- A per-thread memo sits in front of it and skips both the key string and the lock when the
  previous call had the same shape and request. That is the case for a sized call.
- The key omits stride, and omits ld except at batch == 1, where cuSOLVER's bufferSize reads lda.
- `CacheIsBounded` inserts 1500 distinct shapes and ends with size=1024 and evictions=476.

From the breaker's `overhead.cc`, re-run on the repaired build:

| measurement | planner | today's path |
|---|---|---|
| cached lookup, 1 thread | 2.42 Mcalls/s | `potrf_route` 1.27 Mcalls/s |
| cached lookup, 8 threads | 14.31 Mcalls/s | 6.53 Mcalls/s |
| cached lookup, 16 threads | 23.75 Mcalls/s | 2.38 Mcalls/s |
| `potrf_buffer_size` | 0.39 us | 2.80 us |
| buffer_size + potrf (n=8, b=256) | 3.25 us | 6.13 us |
| 200,000 distinct batch sizes at n=640 | maxrss +2.2 MB (bounded) | |
| 50,000 distinct ld at b=64 | 1 miss | |

A miss costs more than before (9.6 us at n=640): it now binds one plan per legal native tier for
the covering size.

**potrf_tests** (all 209 cases) passes 209/209 with the planner off,
on, and on with `BATCHLAS_POTRF_ROUTE=native`. That holds in the vendor build and in the
vendor-free build. `potrf_planner_tests` passes 15/15 in both builds.

**ROCm.** Run with `ROCM_PATH=/opt/rocm-5.3.0` (this box has no 6.2.4),
`scripts/rocm_syntax_check.sh` passes rocsolver.cc and rocsparse.cc. rocblas.cc fails with 4
`std::__invoke` errors, and the base commit 14ccca77 shows the same 4, so they are pre-existing.
The same forced-ROCm `-fsyntax-only` recipe on `factorization.cc`, `potrf_cta.cc` and
`potrf_blocked.cc` gives 0 errors.

**Execution.** Every run uses non-natural ld and stride, one HPD matrix in every item, and 1e6
poison in the other triangle. The results are unchanged from the first round: residuals at about
1e-7 and 1e-16, every item bit-identical, the poison intact on every native tier, and the output
bit-identical to the old path except for CtaWg.
