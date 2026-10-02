# PlanTree prototype: potrf selection as a plan value

This is a prototype of the "PlanTree" design. `plan_op()` builds a `plan::Node`, and
`Exec<B, Tier>::run` executes that node. It covers potrf only, and it runs next to the
shipping path in the same library:

- `BATCHLAS_POTRF_PLANNER=1`, or `plan::set_potrf_planner(1)`, routes `potrf` and
  `potrf_buffer_size` through the new path.
- With the planner off, the library behaves exactly as before.

Machine: threadripper02, RTX PRO 6000 Blackwell (sm_120, 188 CUs), GPU 2. Build: dpcpp-cuda
Release, `nvptx64-nvidia-cuda` sm_120.

## Code

| file | role |
|---|---|
| `src/plan/plan.hh` | Core types. All SYCL-free: Facts, Legality, Candidate, Decision, Node, Request, the TierDescriptor concept, TierList (`each`, `visit` by name, derived `routes`/`names`), and the strategies |
| `src/plan/planner.hh` | The one generic `plan_op`, CostModelChooser, WindowPolicy (two WindowChoosers), FirstLegalChooser, warn-once, `explain()` |
| `src/plan/potrf_tiers.hh` | The descriptors Tiny, Cta, **CtaWg** (the extension), LPanel, Blocked and Vendor. Also the hand windows as `WindowRow` data, the profile looked up by tier name, the single candidacy function, and `plan_trsm`/`plan_gemm` for Blocked's children |
| `src/plan/potrf_exec.hh` | `Exec<B, Tier>`, which has no primary template, plus bind, facts, request, the cache key, `potrf_planned`, and `potrf_planned_buffer_size` |
| `src/extensions/potrf_{cta,blocked}.cc` | `potrf_cta_dispatch_planned(geometry)`, `potrf_blocked_{dispatch,buffer_size}_planned(BlockedParams)`. These execute the node's geometry and parameters and do not re-derive them |
| `src/extensions/potrf_launch_plan.hh` | `cta_plan_from(geometry)` and `cta_wg_geometry` (the extension) |
| `src/dispatch/entry_points/factorization.cc` | The plan cache (leaked map with a mutex), the plan log, and the switch in `potrf`/`potrf_buffer_size` |
| `tests/potrf_planner_tests.cc` | Equivalence, sizing, execution, pins, EXPLAIN/cache, extension, and the candidacy net |
| `mutants.sh` | Deliberate breaks, one per axis, plus the missing-Exec compile check |

## Results (all on GPU 2)

**Equivalence.** The cells are the 570 distinct (dtype, n, batch) cells of `sm120_potrf_sweep{,_edges}.jsonl`.
Every sweep row is Lower, so each cell was also run as Upper, giving 1140 cells.

Each cell was run in 16 modes:

- profile: the device default (sm_120, cost model) or `BATCHLAS_ROUTING_PROFILE=sm_89` (windows)
- walk: auto or vendor-free
- pin: `native`, `native:{tiny,cta,lpanel,blocked}` or `vendor`

That is 18,240 comparisons against `backend::potrf_route` on the device. There were 0
mismatches, both for the library's cached node and for the same planner instantiated in the
test translation unit.

Pins that the new path records as not honoured include:

- `pin=native`: 270 per profile
- `native:lpanel`: 638
- `native:blocked`: 570

Today these are replaced silently.

**Intended differences.** No route differs from today's.

- **Sizing.** 330 of the 1140 cells get smaller and none get larger. They are listed in
  `sizing_old_vs_new_sm120.txt`. The largest saving is float n=36..80 at b=32768: 2,148,401,152 B
  shrinks to 131,072 B.
- **Model versus windows.** `strategy_model_vs_windows_sm120.txt` lists 177 of the 570 Lower cells
  where `BATCHLAS_STRATEGY=model` and `=windows` disagree. For example, cdouble n=20 is LPanel under
  the model and vendor under the windows. These are reported, not silent.

**Sizing for float b=8192** (old is the max over tiers, new is the chosen tree):

| shape | old (B) | new (B) | chosen tier |
|---|---|---|---|
| n=36 | 537,100,288 | 32,768 | CTA |
| n=128 | 537,100,288 | 32,768 | LPanel |
| n=36, pinned `native:blocked` | 537,100,288 | 753,664 | Blocked, W clamped to n-nb |

Other shapes:

| shape | old (B) | new (B) | chosen tier |
|---|---|---|---|
| float n=640 b=512 | 33,568,768 | 33,568,768 | Blocked |
| cdouble n=20 b=8192 | 229,376 | 32,768 | LPanel |

**Execution.** Every run used non-natural ld and stride, one HPD matrix in every item, and a 1e6
poison in the other triangle. Residuals are all at about 1e-7 in single precision and about 1e-16
in double. All items were bit-identical, and the poison stayed intact on every native tier.
Output was bit-identical to the old path on every tier except CtaWg, which is a different
kernel scope.

| shape | how it was chosen | tier |
|---|---|---|
| float n=16 b=4096 | auto | Tiny |
| float n=36 b=8192 | auto | CTA |
| float n=128 b=8192 | auto | LPanel |
| cdouble n=20 b=8192 | auto | LPanel |
| float n=640 b=512 | auto | Blocked, with 8 children |
| double n=300 b=256 | pinned | Blocked, with gemm children on register_tiled |
| float Upper n=128 | auto | vendor |
| cfloat Upper n=24 | pinned CTA | CTA |
| float n=20, cdouble Upper n=24 | name pin | CtaWg |

**Plan equals launch.** Each SYCL_UR_TRACE=2 trace matches its EXPLAIN:

- f128: `PotrfLpanelKernel<float,8>` 1048576/128
- f36: `PotrfCtaKernel<float,8,4,SubGroup>` 262144/128
- f16: `PotrfTinyKernel<float,16>` 65536/64
- cd20: `PotrfLpanelKernel<cdouble,8>` 262144/128
- f640 Blocked leaf: `PotrfCtaKernel<..,WorkGroup>` 131072/256

The coverage rows confirm that the planned child routes are what the routed sub-ops actually chose:

- gemm: vendor:auto
- trsm: native:blocked

**Mutants** (`mutants.sh`, each one rebuilds only the test TU):

| mutant | result |
|---|---|
| drop `Exec<B,CtaWg>` | compile error: the `static_assert` "a potrf tier has no Exec<B, Tier> specialisation" |
| 1: window edge 256→255 | 4 mismatches, sm_89 auto float L n=256 |
| 2: margin ignored | 36 mismatches, sm_120 auto only |
| 3: LPanel and Blocked swapped in Tiers | 0 mismatches. Order is now only the FirstLegal tie order, and no window or model path reaches it |
| 4: LPanel legality accepts Upper | 202 mismatches, vendor-free and `native:lpanel` pins only |

**Pins.** With `BATCHLAS_ROUTE_STRICT=1`, Upper + `native:lpanel` throws `invalid_argument`.
Without strict mode, the decision is recorded with honoured=0 and why=pin-rejected, and three
unservable calls print exactly one warning.

**potrf_tests with BATCHLAS_POTRF_PLANNER=1** (CUDA instantiations, with and without
`BATCHLAS_POTRF_ROUTE=native`): 205 of 209 pass. The 4 failures are
`PotrfBlockedTest/{4..7}.BufferSizeCoversEverySupportedNativeTier`. That test asserts the old
contract, "size under pin A, run under pin B". The planner deliberately drops that contract.

**Cost per call.** From `ExplainAndCache`, float n=128:

| path | time per call |
|---|---|
| cached plan | 0.56 us |
| uncached plan | 1.28 us |
| today's `potrf_route` | 0.79 us |

A sized call today runs `potrf_route` twice, and the planner runs one miss and one hit. The cached
lookup misses the 0.2 us target, because the cache key is built from strings and the settings are
re-parsed on every call.
