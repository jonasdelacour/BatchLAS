# Kernel selection prototype: route descriptors, pluggable choosers, Selection value

This directory holds the evidence for the "Route Descriptor Tables with Pluggable Choosers" design
for potrf. **The CUDA facade now runs it**: `potrf<Backend::CUDA, T>` and
`potrf_buffer_size<Backend::CUDA, T>` go through `potrf_v2::potrf` and `potrf_v2::buffer_size`
(`src/dispatch/potrf_select.cc`, a library TU). The pre-registry facade, which resolves with
`RouteTable<Op::potrf>` and runs the if-chain, is kept as `backend::potrf_legacy` and
`backend::potrf_buffer_size_legacy`. It serves the non-CUDA backends, and it is the reference
the tests compare against in one process.

Machine: threadripper02, RTX PRO 6000 Blackwell Max-Q (cc 12.0, 188 SMs, local_mem 101,376 B,
max work-group 1024), GPU 1. DPC++ `/opt/dpcpp-cuda`, CUDA 13.2, Release, SYCL target
`nvptx64-nvidia-cuda` (sm_120), `BATCHLAS_CPU_TARGET=none`. Base commit `14ccca77`.

## Layout

| file | role |
|---|---|
| `src/dispatch/selection/select.hh` | op-agnostic: `Verdict`, `Candidate(s)`, `Decision`, `Chooser`, `ModelChooser`, `WindowChooser`, `Pin`, `decide()` |
| `src/dispatch/selection/route_table.hh` | `SelectableOp` and `RouteDescriptor` concepts, `Table<Op, R...>` (the Plan variant, keys and candidate rows derived from the type list), `Selection`, `explain()` |
| `src/dispatch/selection/run.hh` | op-agnostic glue: `parse_pin`, pin policy (warn once / strict), `decide_selection`, `enforce_pin`, `size_selection`, `run_selection` (checked launch), `explain(Selection)`, `record_coverage` |
| `src/backends/potrf_routes.hh` | potrf descriptors `Tiny`, `Cta`, `LPanel`, `Blocked`, `CtaWg`, `Cusolver`, `PotrfOp` (incl. `matches`), `PotrfTable` |
| `src/backends/potrf_windows.hh` | the sm_89 hand windows as first-match data rows |
| `src/backends/potrf_select.hh`, `src/dispatch/potrf_select.cc` | the library entry points `potrf_v2::{select, run, explain, buffer_size, potrf}`, the descriptor bodies, the cost-book shim and the shape builder |
| `potrf_select_tests.cc` | gtest T0..T12, target `potrf_select_tests` (EXCLUDE_FROM_ALL, CUDA + cuSOLVER builds) |
| `getrs_offline.cc`, `build_offline.sh` | getrs on the same core, through the same glue, offline (g++) |
| `mutate_compile.py`, `mutate_runtime.py` | deliberate breaks, at compile time and on the GPU |
| `sweep_cells.py`, `sweep_cells.txt` | the 577 sweep cells |

Other library changes: `potrf_cta_dispatch_geometry` (the CTA launcher with a caller-decided
geometry) and `potrf_blocked_{dispatch,buffer_size}_params` (Blocked with caller-decided nb/W).
The shipped launchers forward to both. The `settings_epoch` API of the first prototype is gone,
and `run()` no longer checks it (see "Selection and run" below).

## Build and run

    cmake --build build --target potrf_select_tests -j48
    CUDA_VISIBLE_DEVICES=1 ./build/tests/potrf_select_tests
    BUILD_DIR=build experiments/kernel_selection/descriptor-registry/build_offline.sh
    BUILD_DIR=build python3 experiments/kernel_selection/descriptor-registry/mutate_runtime.py <build-script>

`BATCHLAS_V2_EQUIV_CSV=<prefix>` makes T1 write every decision as a CSV.
`--gtest_also_run_disabled_tests --gtest_filter='*Probe*'` reprints the `kOnlyCtaWg` ranges.

## What changed for library users (CUDA)

- `potrf_buffer_size` returns the CHOSEN route's own bytes. Float L n=36 b=8192 drops from
  537,100,288 B to 32,768 B. Pinned Blocked double L n=36 b=8192 drops from 67,338,240 B to
  1,277,952 B, because W is clamped to n - nb. Pricing still uses the unclamped W.
- A pin the route cannot honour is no longer silent. `BATCHLAS_POTRF_ROUTE=native:lpanel` on
  Upper runs the vendor and prints the explain table once per (op, pin).
  `BATCHLAS_PIN_POLICY=strict` turns that into `std::invalid_argument`.
- `native:cta_wg` is an exact-key pin word. The shipped parser's "not a recognised route"
  warning no longer fires on the public path, because the exact key is matched before the shared
  grammar.
- A route change between `potrf_buffer_size` and `potrf` raises `std::length_error` before any
  launch. The legacy facade over-allocated instead (max over tiers). This is the one public
  contract change: `PotrfBlockedTest.BufferSizeCoversEverySupportedNativeTier` now asserts it on
  CUDA (A and info untouched, a re-query succeeds), and keeps the old assertion elsewhere.
- Vendor-free: Upper orders that subgroup CTA cannot hold but cta_wg can are served instead of
  throwing `NoRouteError`. The ranges are float 78..155, double/cfloat 55..109 and
  cdouble 39..77.
- A bare `native` pin on Upper in those ranges now runs cta_wg. Before, it silently ran the
  vendor. T12 times this change (below): cta_wg is 0.99x to 2.92x the vendor's time.
- A routing profile measured on another architecture (`ProfileChoice::nearest`) prints one
  warning per process.

## Results (GPU 1)

**T1: route and launch-geometry equivalence against an explicit list.** For each cell, T1
compares the legacy `backend::potrf_route` (vendor and vendor-free) with `select()` on the
vendor table and on the vendor-free table. It does this under the default profile (sm_120,
model) and under `BATCHLAS_ROUTING_PROFILE=sm_89` (windows). Every difference must be in
`kOnlyCtaWg`: vendor-free, Upper, the order in the dtype's range, legacy without a native route,
new on cta_wg. Every listed cell must also differ. Where both pick the same native route, the
planned geometry is compared with the shipped launcher's own debug hook. The hooks are
`potrf_{tiny,cta,lpanel}_debug_launch` and `potrf_blocked_debug_params`, with W clamped. The
group count must also follow the packing.

| set | decisions | listed and moved | geometry compared | unintended |
|---|---|---|---|---|
| sweep (577 cells) | 2,308 | 2 of 2 | 1,821 | 0 |
| boundary grid (4 dtypes x L/U x 41 orders x 6 batches) | 7,872 | 348 of 348 | 4,021 | 0 |

The grid now includes both edges of every listed range (38/39, 54/55, 77/78, 109/110, 155/156).
The list was recorded by `DISABLED_ProbeCtaWgOnlyRanges` for local_mem 101,376 and max_wg 1024.
On another device T1 fails until the list is re-derived.

**T2 sizing** (legacy = max over tiers; new = chosen route):

| shape | route | legacy | new |
|---|---|---|---|
| float L n=36 b=8192 | native:cta | 537,100,288 | 32,768 |
| double L n=36 b=8192, pinned blocked | native:blocked | 67,338,240 | 1,277,952 |
| float L n=128 b=8192 | native:lpanel | 537,100,288 | 32,768 |
| float L n=640 b=512 | native:blocked | 33,568,768 | 33,568,768 |
| cdouble L n=20 b=2048 | native:lpanel | 57,344 | 8,192 |
| double L n=36 b=8192 | native:lpanel | 67,338,240 | 32,768 |
| float U n=128 b=8192 | vendor:cusolver | 32,768 | 32,768 |

The Blocked figure is asserted equal to the layout replay with params rebuilt in the test from
the driver's own `potrf_blocked_debug_params`, and at most 1/16 of the legacy figure. Sufficiency
is checked by running, not by re-computing: every T3, T5 and T8 run uses exactly `sel.workspace`
bytes followed by a 4 KiB canary, and the canary must be untouched.

**T3 execution**: 20 runs. Each compares bitwise with the legacy path on the same input, and
checks the residual, info, untouched poisoned padding (1e6, in range) and the canary. The per-route
set (natural layout) covers tiny, cta, lpanel (float, cdouble), blocked (float n=640, n=1024,
pinned double n=36), vendor (cdouble L, float U) and cta_wg (cfloat U n=48, float U/L n=128). The
strided set uses ld = n + {1..7} and stride = ld*n + {1..64} on tiny, cta, lpanel, blocked,
pinned blocked, vendor and cta_wg (Upper and Lower). Every run has 0 items differing from legacy,
except cta_wg, which legacy cannot run. Padding and canary are untouched everywhere. Residuals are
1.2e-07 (float), 1.8e-16 (double) and 1.4e-16 (cdouble) or below the 50*n*eps bound.

**T5 saturating batches**, every item the same matrix, bit-identical to item 0:
- Blocked: double L n=36 b=8192, also identical to legacy.
- cta_wg (L=256, so cross-sub-group SLM traffic is exercised): float U and L n=128 b=2048,
  cfloat U n=64 b=4096 and cdouble L n=48 b=4096. 0 items differ in every case.

**T8 (rule 9)**: cta_wg is launched at its advertised ceiling (the largest order whose cta_wg row
is eligible), Upper and Lower, b=256. The ceilings are float 155, double 109, cfloat 109 and
cdouble 77. All run with info 0 and residual within bounds. The scan must end below n=2048, so a
missing capacity gate fails.

**T6** asserts `extrapolated=0` at f128 b=8192, and `extrapolated=1` at b=4M, both decided
`by=model`. **T9**: on a heterogeneous batch every native row is illegal, the vendor table picks
vendor, and the vendor-free table has no route. **T10**: a CostBook without a vendor row makes
`ModelChooser` decline, and windows decide. **T11**: `run()` rejects a DecideOnly selection, a
selection made for another n or uplo, and a short workspace. It accepts a `reload_settings()`
between select and run. **T7**: legacy `potrf_route` 0.85 us, `select` decide-only 0.77 us,
select + size 0.93 us.

**T12: the behaviour changes, timed.** Upper, vendor vs pinned cta_wg. Each figure is the median
of 10 reps after 3 warm-up reps, in two passes in A/B/B/A order; the table shows the minimum of
the two. Fresh input is copied on the device per rep, outside the timed region. This is a
screen, not a verdict.

| dtype | n | b=1024 cta_wg/vendor | b=8192 cta_wg/vendor |
|---|---|---|---|
| float | 78 / 96 / 128 / 140 | 1.01 / 1.37 / 2.15 / 1.79 | 0.99 / 1.15 / 1.67 / 1.34 |
| double | 64 / 96 | 1.64 / 1.95 | 1.59 / 1.80 |
| cfloat | 64 / 96 | 1.85 / 2.92 | 1.52 / 1.81 |
| cdouble | 48 / 64 | 1.23 / 1.89 | 1.21 / 1.77 |

cta_wg has no cost row, so Auto on a vendor build never picks it. The slowdown applies only to an
explicit `native` / `native:cta_wg` pin. Vendor-free, the alternative is `NoRouteError`.

**Library regression (GPU 1, this build):** `potrf_tests` 209/209, `posv_tests` 40/40,
`potrf_plan_tests`, `settings_tests`, `options_api_tests`, `linalg_layer_tests`,
`error_model_tests`, `routing_profile_tests`, `route_vocabulary_tests` and
`resident_capacity_tests` all pass. `cond_tests`: 1 failure,
`CondTest/0.RandomHermitianTridiagonalLogCondSpectral` (float). It is deterministic and does not
reach potrf (Spectral cond goes through syev), and it is listed among this machine's pre-existing
failures. It was not re-attributed by a base rebuild.

**Vendor-free build** (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`, cuBLAS and cuSOLVER 0): the library
links, with 36 `potrf_v2` symbols and no `potrf_vendor` reference. `potrf_select_tests` is not
configured there, because CMake gates it on cuSOLVER. `potrf_tests` passes 209/209. `posv_tests`
has 1 failure, `PosvTest/3.FusedSolveArmSolvesOnBothTriangles` (cdouble Upper n=100 with the
`native` pin). That order is above cdouble cta_wg's ceiling of 77, so no route exists and the
call throws `NoRouteError`. T1 shows the legacy vendor-free walk has no native route there
either. Float, double and cfloat at n=64 and n=100 are now served by cta_wg.

## Deliberate breaks

These are in `mutate_runtime.py`. Each one edits a source in place, relinks the library, runs
everything except T7/T12, and restores the file (md5-checked). All 14 go red, and each red set is
narrow. After the run, all 6 sources matched their pre-run md5.

| break | red tests | what moved |
|---|---|---|
| no-extension-tier | T0, T1 x2, T3 x2, T4, T5 cta_wg, T8 | the 348 + 2 listed cells no longer move |
| margin-zero | T1 x2 | only default-model/vendor cells, vendor to native (48 grid, 26 sweep) |
| window-off-by-one | T1 x2 | only sm_89 float L n=36 (lpanel to cta) |
| lpanel-not-lower-only | T0, T1 x2, T4 | only vendor-free Upper (vendor to lpanel) |
| ctawg-no-capacity-gate | T1 x2, T8 | 804 unlisted vendor-free Upper cells above each ceiling |
| blocked-no-w-clamp | T1 x2, T2 | Blocked planned W != the launcher's clamped W (double/cdouble L) |
| cta-unpacked-geometry | T1 x2 | CTA planned G != the launcher's packed G |
| extrapolated-inverted | T6 | - |
| no-heterogeneous-check | T9 | - |
| unpriced-vendor-rows-dropped | T1 x2, T2, T3 x2, T6 | default-model cells only (model declines, windows decide) |
| tiny-no-cap-check | T6 | explain text only: Tiny's plan is unfit above its cap anyway |
| model-ignores-unpriced-vendor | T10 | - |
| run-skips-shape-check | T11 | - |
| run-skips-sized-check | T11 | - |

## Compile-time guards

These are in `mutate_compile.py`. Each one fails to build:
- `launch` or `kernel` is missing, or a `Geometry` has no `.fits`: `RouteDescriptor` fails.
- An op has no `matches()`: `SelectableOp` fails.
- A key is duplicated: "two descriptors share a key".
- A window names an unknown route: "a potrf window names no route in the table".
- A `launch` is declared but not defined: the symbol is left undefined.

## Extension (a): a new tier, `native:cta_wg`

This is CTA forced to WorkGroup scope with one matrix per work-group. It took one descriptor in
`potrf_routes.hh`, one entry in each of the two table aliases, and 8 lines in `potrf_select.cc`.
No facade, `buffer_size`, resolver, coverage or `explain` edit was needed.

## Extension (b): getrs on the same core (offline)

`getrs_offline.cc` writes only getrs's Shape, Args, `matches()`, three descriptors and three
window rows. Decisions go through the shared `decide_selection` and `explain(Selection)`. They
match `resolve_route_uninstrumented<Op::getrs>` in 20,736 of 20,736 cells, and the 3-D
ModelChooser checks pass (0 failed). Pin parsing, the policy, sizing, the checked run and coverage
are also shared (`run.hh`), but a live getrs would exercise them on the GPU, and that is not done
here.

## Selection and run

`Selection` carries the shape it was made for and a `sized` flag. `run_selection` refuses a
selection with no route, a DecideOnly selection, a different call (`Op::matches`: n, batch, uplo,
heterogeneity) and a short workspace. It does not compare settings epochs. The plan carries every
knob it was made with, so a later `configure()` or `reload_settings()` on another thread cannot
make it inconsistent. The epoch check was a spurious failure mode, and it is gone.

## Deviations from the design and open issues

- The core headers live in `src/dispatch/selection/` and include `src/util/launch_plan.hh`, which
  is not installed.
- The CostBook is re-keyed by name at first use from the positional generated profile; the
  generator does not emit it directly.
- Blocked's gemm and trsm sub-ops go through `gemm<B,T>` and `trsm<B,T>`, not a nested
  `select<gemm>`.
- Tiny and LPanel still re-derive their geometry in the launcher. T1 compares the plan against the
  launcher's derivation over 5,842 decisions. Cta, CtaWg and Blocked launch the plan as given.
- `select()` makes two device property queries per call; they are not memoised.
- Blocked is priced with the unclamped W and laid out with the clamped W.
- Only CUDA routes through the registry. ROCm and NETLIB keep the legacy facade.
