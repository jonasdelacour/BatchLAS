# geqrf on flat kernel selection (phase 5)

As-built notes for the integrator. Specs: `docs/design/flat-kernel-selection.md` (§2, §4, §5, §8,
§12, §13) and `docs/design/flat-kernel-selection-phase3-plan.md`. Templates mirrored: posv and trsm.

Branch `flat-select-p5-geqrf`, based on `424a45bc`.

## What changed

| Path | Change |
|---|---|
| `src/ops/geqrf/choice.hh` | the vocabulary: `Tiny`, `Cta`, `Blocked`, `Vendor` (all fieldless), aliases, last resort, key names, the grid |
| `src/ops/geqrf/geqrf.cc` | the public `geqrf` / `geqrf_buffer_size`, `choose`, `launch`, `workspace`, `native_facts`, and `geqrf_buffer_size_bound` |
| `src/ops/geqrf/can_run.hh` | `can_run`, in a header so the host test `GeqrfCanRunDevice` can call it with a synthetic `select::Device` |
| `src/ops/geqrf/geqrf.hh` | internal header declaring `geqrf_buffer_size_bound` |
| `src/dispatch/entry_points/factorization.cc` | geqrf's entry points and `geqrf_throw_native_unimplemented` removed |
| `include/batchlas/blas/dispatch/route_geqrf.hh`, `src/backends/geqrf_route.hh` | deleted |
| `src/extensions/geqrf_blocked.cc`, `geqrf_native.hh` | the default trailing-gemm seam is now the public gemm. These are **the same hunks as P3.4 (`08d07619`)**, so the two branches merge without a conflict there |
| `src/extensions/band_reduction.cc`, `sytrd_sy2sb.cc` | size with `geqrf_buffer_size_bound` (see deviation 1) |
| `tools/transcribe/geqrf_transcribe.cc` | the transcriber, plus a `--offgrid` mode for the data gate |
| `tools/transcribe/geqrf_offgrid_gate.py` | replays off-grid points through the tables (the data gate) |
| `tuned/transcribed/geqrf.csv`, `tuned/geqrf.<dtype>.{sm_89,sm_120}.txt` | the transcription, with the same rows on both devices |
| `scripts/sweep_to_table.py` | the `GEQRF` OpSpec |
| `tests/geqrf_candidates_tests.cc` | new (§8) |
| `tests/geqrf_tests.cc`, `tests/route_vocabulary_tests.cc` | old-predicate tests deleted or ported (list below) |
| `tests/tuned_tables_tests.cc` | the geqrf candidates, key names, and a grid-and-both-devices row check |
| `benchmarks/factor_bench.cc` | `pin_parsed` reads a geqrf pin with `select::parse`, as for potrf and posv |

## Families

One family per driver. Each driver derives its own knobs (tiny bucket, CTA packing, blocked `nb`,
panel leaf), so no family has fields.

| Family | Driver | `can_run` (R3: the driver's own checks) |
|---|---|---|
| `tiny` | `geqrf_tiny_dispatch` | GPU, sg32, homogeneous batch, `m == n`, `1 <= n <= geqrf_tiny_max_n_for_slm(budget)` |
| `cta` | `geqrf_cta_dispatch` | GPU, sg32, homogeneous, `m >= n >= 1`, `geqrf_cta_fits(m, n, budget)` |
| `blocked` | `geqrf_blocked_dispatch`, with the public gemm injected | GPU, sg32, homogeneous, `m >= n >= 1`, `geqrf_blocked_available<T>()`, CTA tier present (`geqrf_cta_max_elems_for_slm >= 1`, inherited from the old table: its panel leaf is that kernel) |
| `vendor` | `backend::geqrf_vendor` | `d.has_vendor_solver`, which carries `factorization_vendor_available<B>` (cuBLAS **and** cuSOLVER on CUDA) via `select::device_of<B>(q, factorization_vendor_available<B>)` |

`can_run(cta)` uses `geqrf_cta_fits`, which is what the driver throws on. The old `supports()`
tested only `m <= cta_max_m && m*n <= cta_max_elems`. `geqrf_cta_fits` adds a check that the
padded launch hole fits. At the 97,280 B and 45,056 B budgets the two agree for float and
cdouble on every `(m, n) <= 512 x 512` with `m >= n` (checked with a host program).

Last resort (§5.5): `blocked`, then `vendor`. Aliases: `native:tiny`, `native:cta`, `native:blocked`.
The class words `native`, `vendor` and `auto` come from select. A pin that `can_run` refuses
throws `invalid_argument` (R6), and so does an unknown spelling.

A vendor-free build where nothing can run (a wide view, a heterogeneous batch, a CPU queue)
throws `dispatch::NoRouteError` from `throw_no_vendor_route`, as the old facade did, so coverage
still records the `miss` row.

## Keys and grid

`keys: form:exact n:log:3 aspect:log`

- `form`: `sq` (m == n), `tall` (m > n) or `wide` (m < n).
- `n`: the number of columns.
- `aspect`: `max(m, n) / min(m, n)`, integer division.

Work is about `n^3 * aspect`, so `n` has weight 3. Every old threshold is an axis-aligned step in
`(form, n, aspect)`. The tall-panel clause `m >= 128 && n >= 32 && m >= A*n` equals
`n >= 32 && aspect >= A`, because `A >= 4` makes `m >= 128` follow. The grid holds a point on
each side of every threshold the old predicates read:

| Threshold | Grid points |
|---|---|
| tiny window, float: 4..16, 21..32 | 3, 4, 16, 17, 20, 21, 32, 33 |
| tiny window, cfloat: 5..8, 11..16, 24..32; native tie-break `n <= 16 \|\| n >= 22` | 4, 5, 8, 9, 10, 11, 16, 17, 21, 22, 23, 24, 32, 33 |
| order floor: float 64, double 76, cfloat 48, cdouble 256 | 63/64, 75/76, 47/48/49, 255/256 |
| tall clause: n >= 32; aspect >= 4 (32-bit) or 8 (64-bit) | n 31/32/33; aspect 3/4/5, 7/8 |
| CTA/Blocked crossover: float 96, double 48 | 96/97, 48/49 |

Elsewhere the grid is a coarse log grid. `n` runs up to 8192 and `aspect` is 1..256. Wide shapes
get a 4 x 3 grid, because every wide row is `vendor` (no native family takes `m < n`).

The capacities (tiny ceiling, CTA area) are not keys. The transcription runs with unlimited
capacities, and `can_run` applies them at run time (§13).

## Transcription

`tools/transcribe/geqrf_transcribe.cc` evaluates the real `RouteTable<Op::geqrf, T>` and
`resolve_route_uninstrumented` from `424a45bc`'s headers. At each cell it records the winner,
makes that winner unable to run, and resolves again. "Unable to run" is spelled as the capacity
that refuses the tier, so the old `preferred()`, through `best_native_tier`, sees the same
exclusion. Once the walk has returned the vendor, the remaining natives are ranked by the
vendor-free walk. The vendor-present Auto choice therefore comes first, followed by the choice
the old router made on a device where the earlier entries do not fit. Without a vendor, that is
the old vendor-free tie-break.

    g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/geqrf_transcribe.cc -o gt
    ./gt sm_89 sm_120 > tuned/transcribed/geqrf.csv
    python3 scripts/sweep_to_table.py --transcribe tuned/transcribed/geqrf.csv ...

The CSV in the tree is byte-identical to a fresh run. `sweep_to_table.py --check` re-derives all
eight tables from it, and they match (636 rows each).

## Deviations

1. **`geqrf_buffer_size` is the chosen family's need only (R5).** The old function returned the
   maximum over every supported tier plus the vendor. `band_reduction.cc` (twice) and
   `sytrd_sy2sb.cc` size once at a bounding panel and then factor smaller sub-views, which can
   choose a different family. They now call the internal `geqrf_buffer_size_bound` (in
   `src/ops/geqrf/geqrf.hh`). It returns the maximum over every family this device can run at all.
   Every family's sizing is monotone in `(m, n, batch)`, and the test
   `BoundCoversEverySubViewChoice` checks the bound against every sub-view under Auto and under
   every pin.
2. **The default trailing-gemm seam.** The salvaged work made it mandatory (it threw). I replaced
   that with P3.4's exact hunk: an empty seam means the public gemm through `with_backend`.
   `geqrf.cc` still injects `gemm<B, T>` explicitly.
3. **No `max_wg` clause in `can_run(tiny)`.** The salvaged draft had one, but the tiny driver does
   not check it, so it was removed (R3).
4. **Docs not updated.** `docs/perf/qr.md`, `docs/design/vendor-free-status.md`,
   `docs/design/vendor-independence.md`, `docs/design/small-n-factorization-plan.md` and
   `docs/perf/blackwell.md` still cite `route_geqrf.hh` line numbers. No anchor was renamed, so
   `check_evidence_anchors.py` is unaffected. For the integrator: point the routing paragraphs at
   `src/ops/geqrf/geqrf.cc` and `tuned/geqrf.*.txt`. The AGENTS.md §9 list of `route_*.hh` tables
   loses geqrf.
5. **Not edited, to avoid conflicts:** `src/backends/orgqr_route.hh` has two comments naming
   `geqrf_route.hh`, and the orgqr migration deletes that file. `tests/CMakeLists.txt`'s
   `geqrf_tests` native re-run (`BATCHLAS_GEQRF_ROUTE=native|...`) stays, because `native` is
   select's class word.

## Old-predicate tests

Deleted from `tests/route_vocabulary_tests.cc`, the whole `RouteGeqrf.*` suite:

- `VendorFreeFallbackHandsOverTheNativeRoute` → `VendorFreeWalkIsTheOldTieBreak`
- `WideIsUnsupportedByEveryNativeArm`, `CorrectnessGatesAreNotSpeedGates`, `CtaCapacityIsAnAreaAndAHeightNotTwoExtentBounds`, `Sg32GatesBothNativeArms` → `CanRunFalsePinsThrow`, `CanRunEqualsLaunch`, `GeqrfCandidatesCpu.CpuQueueRunsNoNativeFamily`, and for the device clauses `GeqrfCanRunDevice.DeviceClausesRefuseTheNativeFamilies`. The queue-driven tests skip without a GPU or sub-group size 32, and every GPU here has both plus SLM for a CTA element, so on these boxes only the host test (synthetic `select::Device`: `is_gpu = false`, `has_sg32 = false`, and an SLM budget straddling the smallest that holds one CTA element) can catch a dropped `is_gpu`, `has_sg32` or Blocked CTA-tier clause. It ports `Sg32GatesBothNativeArms`, which ran on the host
- `PreferredIsTheMeasuredOrderFloorAndTheTallClause`, `NativeWalkTakesTinyInTheWindowGaps` → `AutoReadsTheTranscribedTable`, `GeqrfTranscribedTable.RowsHoldTheOldPreference`
- `BareOriginResolvesToASpecificAlgorithm`, `BatchlasGeqrfRouteIsActuallyRead` → `LegacyAliasesAndClassWords`
- `RectangularIsSupportedAndSquarenessIsNotAGate`, `AbsentKernelIsUnsupportedRatherThanSelectable` → `CanRunEqualsLaunch` on tall shapes

In `tests/geqrf_tests.cc`:

- `NativeTierTieBreakPicksTheFasterNativeVendorFree` (G9b) was deleted. The crossover is now table
  data, checked by `AutoReadsTheTranscribedTable` (96/97 float, 48/49 double).
- The geqrf half of `RouteTableAndTheVendorFreeFallback` was deleted.
- `WideIsRefusedAndTallAndSquareAreNot` (a) now asks pins instead of `supports()`.
- The env pins became `select::ScopedPin`.

## Gate results (RTX 4090 box, GPU 0)

### (a) Correctness

Targets: `geqrf_tests`, `geqrf_candidates_tests`, `geqrf_tests_native`, `orgqr_tests`,
`ormqr_tests`, `ortho_tests`, `gesvd_tests`, `sytrd_sy2sb_tests`, `sytrd_sb2st_tests`,
`syev_two_stage_tests`, `select_tests`, `tuned_tables_tests`, `route_vocabulary_tests`,
`backend_dispatch_tests` and `options_api_tests`. Each was built and run in `build` and in
`build-vf`, and the failing names were compared with the same targets built from `424a45bc`.

- **Vendor build:** the failing set is the same on both sides: `ortho_tests` segfaults (a known
  defect on this box). Every other test passes, `geqrf_candidates_tests` included.
- **Vendor-free build:** the failing case names are the same on both sides, a set of 71 that
  were already failing at `424a45bc` (`options_api`, `ortho`, `ormqr`, `orgqr`, `gesvd` and
  `sytrd_sb2st`). `geqrf_candidates_tests` passes. Its one vendor-free failure was a test bug,
  fixed in `535523dc`: wide sub-views have no route without a vendor.

**Deliberate breaks** in `geqrf.cc`. Each was restored from a saved copy and md5-verified
(`7204a3a2…`). Each red set below covers all four CUDA dtypes:

| Break | Red set |
|---|---|
| `key_of`: aspect always 1 | `AutoReadsEveryKeyField`, `AutoReadsTheTranscribedTable`, `TraceShowsTheLookupKey` |
| `can_run(cta)`: the fit check dropped | `CanRunEqualsLaunch`, `CanRunFalsePinsThrow`, `PinnedCandidatesStraddleTheirLimits`, `AutoReadsTheTranscribedTable` |
| `workspace(blocked)` returns 0 | `ExactWorkspaceInAPoisonedArena` plus every test that runs blocked (the driver's allocator throws) |
| `launch(cta)` runs the blocked driver | `PinnedRunIsTheDirectKernelBitForBit` plus every test that runs cta (cta's 0-byte workspace is too small for blocked) |

Review round, breaks in `can_run.hh` (restored from a saved copy, md5 `80b5bec2…` verified after
each; `geqrf_candidates_tests` in `build`, GPU 0, all four dtypes):

| Break | Red set |
|---|---|
| `d.is_gpu` dropped | `GeqrfCanRunDevice.DeviceClausesRefuseTheNativeFamilies`, `GeqrfCandidatesCpu.CpuQueueRunsNoNativeFamily` (the NETLIB instantiations get a CPU queue) |
| `d.has_sg32` dropped | `GeqrfCanRunDevice.DeviceClausesRefuseTheNativeFamilies` only |
| Blocked's `geqrf_cta_max_elems_for_slm(budget) >= 1` dropped | `GeqrfCanRunDevice.DeviceClausesRefuseTheNativeFamilies` only |

The oracle `expect_runs()` now also carries Blocked's CTA-tier clause.

### (b) Data gate: off-grid replay

`geqrf_transcribe --offgrid` evaluates the OLD `resolve_route` at 3,000 random points per dtype
and capacity set. `n` and `m` are log-uniform in [1, 8192]: a third square, half tall (half of
those within aspect 16), the rest wide. Each point gets the old choice with a vendor and without
one. `tools/transcribe/geqrf_offgrid_gate.py` takes the nearest row of each new table (the
converter's `nearest`, which applies the same rule as `Table::nearest`), walks it with a
`can_run` model at the same capacities, then the last resort, and skips the points that fall on
the grid.

| Capacities | Off-grid points per (dtype, device, vendor or vendor-free) | Agreement |
|---|---|---|
| 97,280 B budget (sm_89 and sm_120): tiny 32/32/32/16, CTA area 11776/5888/5888/2944 | 2,316 - 2,329 | **100.000%** in all 16 cells |
| 45,056 B budget (a 48 KiB device) | ~2,320 | 100.000% |
| small capacities (tiny 16/8/16/8, CTA max_m and area 3000/2500, 1500/1200, 4000/1000, 700/700 for float/double/cfloat/cdouble) | ~2,300 | 100.000% |
| negative control: grid points 31/63 (float) and 47/75 (double) removed | 2,328 / 2,333 | 99.01% / 99.10% (vendor): **exit 1** |

No region disagrees. The gate requires 100% agreement (any disagreement exits 1): the
transcription is exact by construction, and a missing threshold costs only about 1%, so a 99%
tolerance (the first version) passed the negative control. The gate checks the tables against
the OLD router through a Python model: its `can_run` is the old `supports()` (`m <= cta_max_m`
and `m*n <= cta_max_elems`), not `geqrf_cta_fits`, and its `nearest` is the converter's. The
shipped C++ select path is covered by `AutoReadsTheTranscribedTable` and by (c).

### (c) Cross-check against the 424a45bc binary

A host program runs Auto `geqrf` on 24 cells (each dtype, threshold edges and off-grid shapes),
one process per cell, with `BATCHLAS_COVERAGE_OUT`. It was linked once against `424a45bc` and
once against the branch.

- **Vendor build:** 24/24 cells choose the same family. The old row spells the vendor
  `vendor,auto` and the new one `vendor,vendor`.
- **Vendor-free build:** 23/23 runnable cells agree. The wide cell (`float 10x20`) throws
  `NoRouteError` in both builds. The old build also wrote a `reached ... vendor,auto` row for
  it, a route that never ran. The new build writes only the `miss` row.

### Repository checks

`run_local_checks.sh` passes everything except `check_cmake_syntax`, which trips over CMake's
generated files under the in-tree `build-vf/` directory. That is an artifact of where the
directory sits, not of this change. `check_tuned_tables.py` prints `no tools/tune/geqrf_spec.cc,
geqrf tables not checked`. The tables are transcribed, so there is no tuner spec yet.
`check_evidence_anchors.py` reports 0 dangling references. `factor_bench geqrf ...
--arms=vendor,cta,native:blocked,bogus` reports `pin_parsed` 1/1/1/0.
