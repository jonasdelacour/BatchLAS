# Kernel selection prototype: route descriptors, pluggable choosers, Selection value

This directory holds a working prototype of the "Route Descriptor Tables with Pluggable
Choosers" design for potrf. It runs next to the shipped path (`RouteTable<Op::potrf>` +
`resolve_route` + the facade if-chain in `src/dispatch/entry_points/factorization.cc`), so the two
can be compared cell by cell in one process. The shipped path is unchanged apart from three
caller-decided launcher entry points, listed below.

Machine: threadripper02, RTX PRO 6000 Blackwell Max-Q (cc 12.0, 188 SMs), GPU 1, DPC++
`/opt/dpcpp-cuda`, CUDA 13.2, Release, SYCL target `nvptx64-nvidia-cuda` (sm_120), base commit
`14ccca77` (origin/routing-cost-model-core).

## Layout

| file | role |
|---|---|
| `src/dispatch/selection/select.hh` | op-agnostic core: `Verdict`, `Candidate(s)`, `Decision`, `Chooser` concept, `ModelChooser`, `WindowChooser`, `Pin`, `decide()` |
| `src/dispatch/selection/route_table.hh` | `RouteDescriptor` concept, `Table<Op, R...>` (derives the Plan variant, the keys and the candidate rows), `Selection`, `explain()` |
| `src/backends/potrf_routes.hh` | potrf descriptors `Tiny`, `Cta`, `LPanel`, `Blocked`, `CtaWg` (the extension), `Cusolver`, and `PotrfTable` |
| `src/backends/potrf_windows.hh` | the sm_89 hand windows as first-match data rows |
| `potrf_routes.cc` (here) | descriptor `workspace()`/`launch()`, `select()`, `run()`, `potrf_buffer_size()`, `potrf()`, `explain()`, pin parsing, coverage |
| `potrf_select_tests.cc` | gtest T1 to T7, target `potrf_select_tests` (EXCLUDE_FROM_ALL, CUDA builds only) |
| `getrs_offline.cc`, `build_offline.sh` | getrs on the same core, offline (g++) |
| `mutate_compile.py`, `mutate_runtime.py` | deliberate breaks: compile-time and on the GPU |
| `sweep_cells.py`, `sweep_cells.txt` | the 577 (dtype, uplo, n, batch) cells of `benchmarks/results/routing/sm120_potrf_sweep{,_edges}.jsonl` |

Library changes, all additive:
- `potrf_cta_dispatch_geometry`: the CTA launcher with a caller-decided `CtaGeometry`.
- `potrf_blocked_{dispatch,buffer_size}_params`: the Blocked driver with caller-decided nb/W.
- `detail::settings_epoch()`.

The shipped entry points now forward to these, with unchanged behaviour (`potrf_tests`,
`posv_tests`, `potrf_plan_tests` and `settings_tests` pass).

## Build and run

    cmake --build build --target potrf_select_tests -j20
    CUDA_VISIBLE_DEVICES=1 ./build/tests/potrf_select_tests     # T1..T7
    experiments/kernel_selection/descriptor-registry/build_offline.sh       # getrs + compile guards
    python3 experiments/kernel_selection/descriptor-registry/mutate_runtime.py <build-script>

`BATCHLAS_V2_EQUIV_CSV=<prefix>` makes T1 write every decision as a CSV.

## Results

**T1 equivalence.** For each cell, the test calls the shipped `backend::potrf_route` with
vendor_available true and false, and the new `select()` with the vendor table and the
vendor-free table. It does this under the default profile (sm_120, the model) and under
`BATCHLAS_ROUTING_PROFILE=sm_89` (the windows).

| set | decisions | differences | unintended |
|---|---|---|---|
| sweep cells (577 cells x 2 profiles x {vendor, vendor-free}) | 2,308 | 2 | 0 |
| boundary grid (4 dtypes x L/U x 31 orders x 6 batches x 2 x 2) | 5,952 | 168 | 0 |

Every difference is the same intended class: the vendor-free walk on Upper shapes that CTA
cannot hold (float n=78..128, double/cfloat n=64..96, cdouble n=48..77). The shipped walk finds
no native route there and answers vendor (that is, "no route" in a vendor-free build). The new
table also lists the extension tier `native:cta_wg`, which can hold those shapes, so it picks it
as the only eligible route. In the sweep set this is float Upper n=128 b=8192 under both
profiles.

With the extension removed from the type list, the result is **0 differences in all 8,260
decisions** (`mutate_runtime.py no-extension-tier`).

**T2 sizing.** The workspace that `potrf_buffer_size` reports. Old is the maximum over the
supported tiers; new is the chosen route's own size.

| shape | route | old | new |
|---|---|---|---|
| float L n=36 b=8192 | native:cta | 537,100,288 B | 32,768 B |
| float L n=128 b=8192 | native:lpanel | 537,100,288 B | 32,768 B |
| double L n=36 b=8192 | native:lpanel | 67,338,240 B | 32,768 B |
| cdouble L n=20 b=2048 | native:lpanel | 57,344 B | 8,192 B |
| float L n=640 b=512 | native:blocked | 33,568,768 B | 33,568,768 B |
| float U n=128 b=8192 | vendor:cusolver | 32,768 B | 32,768 B |

The pinned Blocked run of double n=36 b=8192 needs 1,277,952 B (W clamped to n-nb=4), against
67,338,240 B on the shipped path.

**T3 execution.** Each case runs through `select()` + `run()` with exactly `sel.workspace`
bytes. The residual is ||LL^H-A||_F/||A||_F on items 0, b/2 and b-1. The result is then
compared bitwise, item by item, against the shipped `potrf` on the same input.

| shape | route | residual | items differing from shipped |
|---|---|---|---|
| float L n=16 b=8192 | native:tiny | 1.2e-07 | 0 |
| float L n=36 b=8192 | native:cta | 8.8e-08 | 0 |
| float L n=128 b=8192 | native:lpanel | 1.0e-07 | 0 |
| cdouble L n=20 b=2048 | native:lpanel | 1.4e-16 | 0 |
| float L n=640 b=512 | native:blocked (routed trsm and gemm) | 1.1e-07 | 0 |
| float L n=1024 b=128 | native:blocked | 1.7e-07 | 0 |
| double L n=36 b=8192, pinned blocked | native:blocked, W clamped | 2.1e-16 | 0 |
| cdouble L n=32 b=512 | vendor:cusolver | 2.7e-16 | 0 |
| float U n=128 b=8192 | vendor:cusolver | 1.2e-07 | 0 |
| cfloat U n=48 b=2048, pinned cta_wg | native:cta_wg (WorkGroup, L=64, G=1) | 7.8e-08 | n/a (shipped runs CTA) |
| float U n=128 b=512, pinned cta_wg | native:cta_wg (WorkGroup, L=256, G=1) | 1.3e-07 | n/a (shipped runs the vendor) |

The f128 costs in `explain()` are lpanel 1270.3 us, blocked 2420.5 us and vendor 1727.4 us,
which match the verified figures.

**T4 pins.**
- `native:lpanel` on Upper: honoured=0, the vendor runs, exactly one warning across two
  calls, and Strict mode throws with "Lower only". The shipped path does the same routing
  silently.
- `native:cta` at n=128: honoured=0 and reported as `unfit`.
- `native:blocked`: honoured=1.
- A bare `native` on Upper n=128 now reaches `native:cta_wg`.

**T5.** Saturating batch through Blocked (double n=36, b=8192, every item the same matrix,
pinned blocked): 0 items differ from item 0, and 0 differ from the shipped unclamped-W path.

**T7 host cost** of one decision (f128): shipped `potrf_route` 0.76 us, new `select` 0.67 us.

**Deliberate breaks.** These are in `mutate_runtime.py`, and each turns a narrow set red.
- margin = 0: only `default-model/vendor` cells move (32 grid, 26 sweep), all vendor to native.
- Window boundary `cta_last<float>` 35 to 36: only `sm_89-windows` float L n=36 moves
  (lpanel to cta).
- `lower_only` removed from LPanel's `legal()`: only vendor-free Upper cells move (vendor to
  lpanel), plus T4. The model never picks it on its own, because it has no Upper cost row for
  lpanel; candidacy is data.
- Extension removed from the type list: T1 goes to 0 differences, and T3 and T4 go red (the
  pin names no row).

**Compile-time guards.** These are in `mutate_compile.py`, and each one fails to build:
- forgetting `launch` or `kernel`, or a `Geometry` without `.fits`: the `RouteDescriptor`
  constraint fails;
- a duplicated key: "two descriptors share a key";
- a window naming an unknown route: "a potrf window names no route in the table";
- a declared but undefined `launch`: the symbol is left undefined at link.

## Extension (a): a new tier, `native:cta_wg`

This is the CTA kernel forced to WorkGroup scope with one matrix per work-group, so its
capacity is the device SLM budget rather than the occupancy slice. It took:
- `src/backends/potrf_routes.hh`: one 30-line descriptor and one type-list entry in each of the
  two table aliases;
- `potrf_routes.cc`: 8 lines (`workspace`, `launch`).

It also uses `potrf_cta_dispatch_geometry`, which the `Cta` descriptor uses too.

The new tier did not need edits to the facade, `buffer_size`, the resolver, the coverage code,
`explain()` or any parallel array. Auto routing on a vendor build is unchanged, because the
tier has no cost row and no window. Pins and the vendor-free walk reach it.

## Extension (b): getrs on the same core (offline)

`getrs_offline.cc` defines `GetrsOp` with Shape (n, nrhs, batch, trans), a 3-D `coords()`,
`cost_variant = trans`, three descriptors, and the shipped `preferred()` as 3 window rows. Its
decisions match `resolve_route_uninstrumented<Op::getrs>` in 20,736 of 20,736 cells (4 dtypes x
vendor/vendor-free). The unchanged `ModelChooser` prices a synthetic 3-D CostBook correctly:
- the nrhs box is honoured;
- the trans variant selects its own rows;
- extrapolation is flagged outside the n box.

None of the chooser code was copied.

## Deviations from the design and open issues

- The core headers live in `src/dispatch/selection/`, not in `include/batchlas/blas/dispatch/`.
  They include `src/util/launch_plan.hh`, which is not installed.
- `potrf_routes.cc` and the v2 entry points are compiled into the test executable, not into
  the library. This avoids a device relink per iteration. The design says
  `src/backends/potrf_routes.cc` in the library, with `BATCHLAS_INTERNAL_API` entry points.
- The CostBook is re-keyed by name at first use from the positional generated profile, instead
  of being emitted by `gen_profile_header.py`.
- Coverage rows are recorded once, with backend = B. They carry no pin, honoured or by
  columns yet. `cta_wg` records as `native:cta`, because the key, not the Route, is its
  identity.
- `Blocked`'s gemm and trsm sub-ops still go through the shipped `gemm<B,T>` and `trsm<B,T>`,
  not through a nested `select<gemm>`.
- Tiny and LPanel launchers still re-derive their geometry: LPanel gets `nb` from the plan, and
  Tiny gets nothing. Cta, CtaWg and Blocked launch exactly the planned geometry.
- `select()` makes two device property queries per call (`potrf_device_facts`). They are not
  memoised per device yet.
- Blocked is priced with the unclamped W, as today, and laid out with the clamped W. Pricing
  the clamped W moves 12 offline cells (double L n=33..36, b >= 8192), which needs an A/B
  measurement first.
- `Selection` cannot travel between the existing `potrf_buffer_size` and `potrf` calls.
  `run()` checks `settings_epoch()` and the workspace size instead.
