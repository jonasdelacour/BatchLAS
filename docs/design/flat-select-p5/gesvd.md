# gesvd: flat kernel selection (phase 5)

Branch `flat-select-p5-gesvd`, based on `424a45bc` (the flat-selection stack head). This file holds
the as-built notes, deviations, gate results and the doc changes the integrator should fold into
`docs/design/flat-kernel-selection.md` §12, `tuned/README.md` and AGENTS.md. Those shared files are
not edited on this branch.

## What changed

- **`src/ops/gesvd/{choice.hh,gesvd.cc}`** hold the whole selection path (R1). Families `jacobi`
  (`gesvdj_cta`), `cta` (`gesvd_cta`), `blocked` (`gesvd_blocked`) and `vendor`
  (`backend::gesvd_vendor`). All are `NoFields`, the same for every dtype: the old code chose no knob.
  The Jacobi rung and the bidiagonalisation mode stay derived inside the drivers.
- **Public entry points moved out of the header.** `gesvd` and `gesvd_buffer_size` (general and
  Hermitian overloads) were inline templates in `include/batchlas/blas/functions/gesvd.hh`
  (`gesvd_dispatch`, `gesvd_buffer_size_dispatch`, `detail::gesvd_route`, `gesvd_op_shape`). They
  are now defined in `gesvd.cc` and explicitly instantiated per backend and dtype. The header keeps
  the `BATCHLAS_API` declarations and the forwarders, and gains `sig::gesvd`, `sig::gesvd_hermitian`,
  `sig::gesvd_buffer_size` and `sig::gesvd_buffer_size_hermitian`. gesvd never had an entry in
  `src/dispatch/entry_points/`.
- **`src/extensions/gesvd_native.hh`** (new, sycl-free) holds the drivers' ceilings:
  `gesvd_jacobi_max_dim<T>(want_vectors)` (moved from `route_gesvd.hh`; `gesvdj_cta_max_dim` now
  returns it) and `kGesvdCtaMaxDim = 32` (the four `> 32` literals in `gesvd_blocked.cc`'s
  `gesvd_cta` / `gesvd_cta_buffer_size` now read it). The drivers have no other change.
- **Deleted:** `include/batchlas/blas/dispatch/route_gesvd.hh` (installed header). There was no
  `src/backends/gesvd_route.hh`. `Op::gesvd`, `BATCHLAS_GESVD_PROVIDER` in `route_env.hh` and the
  `{"gesvd", ...}` linked-coverage row in `src/dispatch/coverage.cc` stay for the integrator.
  `functions/gesvd.hh` still includes `dispatch/{route,no_route,vendor_available,route_env}.hh`
  although it no longer uses them. Removing them recompiles ~30 TUs that may depend on them
  transitively, so that is left for the integrator's sweep.

## can_run (R3)

Common native term: `d.is_gpu && m >= 1 && n >= 1 && batch >= 1`. The jobs are canonicalised once
(`canonical_jobu/jobvh`, Thin -> All where they coincide), and `can_run`, `key_of`, the workspace
and the launch all see the same jobs.

| family | adds |
|---|---|
| jacobi | `has_sg32 && !herm && max(m,n) <= gesvd_jacobi_max_dim<T>(vectors)` (64; cdouble with vectors 32) |
| cta | `has_sg32 && max(m,n) <= 32 && !thin && (herm ? m == n : real T)` |
| blocked | `herm ? m == n && herm == Lower : real T` |
| vendor | `d.has_vendor_solver` |

These are the old `supports()` clauses, with `max_sub_group >= 32` spelled `has_sg32`. Two clauses
are deliberately not R3-exact, and `CanRunEqualsLaunch` documents both:
- **blocked Hermitian Upper** is refused, although `syev_blocked` mirrors Upper into Lower and
  launches. The old router refused it too, so the Upper > 32 routing (vendor, or no route
  vendor-free) is unchanged. Opening it needs its own correctness evidence. See "Found" below: the
  Hermitian drivers already mishandle triangles.
- **vendor** is `has_vendor_solver` alone, as the old `supports()` was. cuSOLVER's gesvdjBatched
  refuses max(m,n) > 32, non-packed batches and thin factors at launch, and it reads the whole
  matrix. Auto reaches it only where no native family can run, so this is the old outcome.

## Keys, grid, tables

- `# keys: herm:exact vec:exact m:log:1.5 n:log:1.5`. `herm` is `N|L|U`. `vec` is `none|all|thin`
  from the canonical jobs (`thin` if either side stays Thin). These are the only job facts the old
  predicates read. The work grows as m·n·min(m,n), hence 1.5 + 1.5. batch is not a key, because the
  old predicates read it only as `batch >= 1`.
- Grid `grid_mn = {1,2,4,8,16,24,32,33,48,64,65,128,256,512,1024}` for both m and n. Both sides of
  every threshold the old predicates use (32 | 33 for cta, the real-Jacobi preference and cdouble
  Jacobi with vectors; 64 | 65 for the Jacobi cap), plus a coarse log grid. The grid is the full
  Cartesian product, so the weighted nearest point decomposes per axis, and each axis keeps its
  side of every threshold.
- **Tables: transcribed old routing for both devices** (maintainer decision). The old predicates
  read no architecture, so one transcription is written as `tuned/gesvd.<dtype>.sm_89.txt` and
  `tuned/gesvd.<dtype>.sm_120.txt`: the same 2025 rows, with the device in the header and
  `source=transcribed:424a45bc`. `tuned/transcribed/gesvd.csv` holds 16200 rows (2 devices x 4 dtypes
  x 2025).
- `tools/transcribe/gesvd_transcribe.cc` is host g++ against `424a45bc` headers. It is the
  posv/trsm technique: resolve with the vendor present, record the winner, mark it unsupported, and
  repeat. After the vendor, the remaining natives are ranked by the vendor-free walk. Capacities are
  build constants (the Jacobi cap), so they are applied; the device is a sub-group-32 GPU. Rebuilding
  it reproduces the CSV md5 (`dedd3cd8...`). Its `points` mode replays arbitrary cells for the
  data gate. Row census over the CSV: `vendor` 12032, `blocked|vendor` 1748, `jacobi|vendor` 996,
  `blocked|vendor|jacobi` 612 (real 33..64: Jacobi supported but not preferred),
  `jacobi|cta|blocked|vendor` 392, `jacobi|blocked|vendor` 196, `cta|blocked|vendor` 112,
  `cta|vendor` 112.
- `scripts/sweep_to_table.py`: `GESVD` OpSpec (no sweep source). `--check` re-derives all eight
  tables from the CSV: OK.
- `last_resort {"blocked","vendor"}`.

## Pins (R6)

- `BATCHLAS_GESVD_ROUTE` and `ScopedPin` as for every migrated op. Aliases: `native:{jacobi,cta,blocked}`,
  `batchlas_{jacobi,cta,blocked}`, `batchlas-{...}`, `vendor:auto`, `netlib`, `netlib:auto`.
- **`BATCHLAS_GESVD_PROVIDER`** (the legacy variable, which the old `parse_route_env` read after
  `_ROUTE`) is still honoured when neither a ScopedPin nor `_ROUTE` is set. It is applied as a
  one-shot ScopedPin around `choose()`, so the same R6 rules apply. In error messages it appears as
  `ScopedPin=...`.
- Behaviour changes: an unknown spelling, or a pin whose `can_run` is false, throws
  `std::invalid_argument`. It used to mean Auto silently. On a NETLIB queue, a native pin now throws.
  The old code skipped resolution on NETLIB, so it ignored every pin there. Bare `batchlas` (an old
  origin word for native) is not accepted, because a class word cannot be aliased; use `native`.
- Coverage: the `reached` row is recorded at launch only, not in `gesvd_buffer_size`, as for
  potrf/posv. The TraceScope opens before the workspace sizing, so a call whose driver refuses at
  sizing still leaves its row.

## Tests

- `tests/gesvd_candidates_tests.cc` (label `eig`, 65 cases on GPU 0 after the review follow-up) implements §8:
  - every native family pinned on 11 shapes straddling 32/33/64/65, square and not, times
    herm N/L/U, times 4 job sets, with closed-form singular values and reconstruction;
  - `CanRunEqualsLaunch` against the direct drivers, plus an oracle that reads the drivers' own
    ceilings;
  - the pinned run against the direct driver, bit for bit;
  - a saturating batch of 1024 identical items, bit-identical;
  - the exact workspace in a poisoned arena, with one byte less refused;
  - unknown pins, legacy aliases through all three channels, and pin precedence;
  - `AutoReadsTheTranscribedTable` (ported from `gesvd_tests`' `DefaultProviderRoutesSmallGeneralToJacobi`),
    `AutoReadsEveryKeyField` on a synthetic table, `TraceShowsTheLookupKey`,
    `VendorFreeLastResortIsBlocked`, the coverage row (backend, m/n/k, uplo, spelling, native
    flags), CPU queues, and `RowsHoldTheOldPreference` on both devices.

  Inputs use a non-natural ld (m+3), a batch stride that is not ld·n, complex data with nonzero
  imaginary parts, and Hermitian poison of 4e3-2e3i in the unreferenced triangle.
- `gesvd_tests`: `DefaultProviderRoutesSmallGeneralToJacobi` is deleted (it read the old
  `RouteTable`; ported as above). `CtaRejectsGenuinelyThinButDispatchStillSucceeds` became
  `CtaRejectsGenuinelyThinAndItsPinThrows`, because a `cta` pin on a thin shape now throws, and
  Auto still solves it. `run_gesvd_with_provider` pins with `ScopedPin`.
- `tuned_tables_tests` gains `GesvdTablesDeclareChoiceKeyNames` and
  `GesvdTablesHoldExactlyTheChoiceGridOnBothDevices` (the grid from `choice.hh`, sm_89 == sm_120 row
  for row), and the generic table checks learn gesvd's candidates.
- **Deliberate breaks** of `gesvd.cc`, each restored from a saved copy and md5-verified. The red sets
  come from `gesvd_candidates_tests` (the `/4`-`/7` instantiations are float, double, cfloat and
  cdouble on CUDA):
  - B1 `cta` ignores thin -> `CanRunEqualsLaunch` and `PinnedCandidatesStraddleTheirLimits`, float
    and double;
  - B2 `key_of` vec without thin -> `AutoReadsEveryKeyField` and `TraceShowsTheLookupKey`, float and
    double;
  - B3 Jacobi cap ignores vectors -> `CanRunEqualsLaunch` and
    `PinnedCandidatesStraddleTheirLimits`, cdouble only;
  - B4 blocked accepts Hermitian Upper -> the same two tests, all four dtypes;
  - B5 `key_of` swaps m and n -> `AutoReadsEveryKeyField` and `TraceShowsTheLookupKey`, float and
    double;
  - B6 the legacy variable ignored -> `LegacyAliasesAndClassWords`, all four dtypes.
- Review follow-up: `EmptyShapesRunNoNativeFamily` (m, n or batch = 0: every native pin refused,
  Auto takes the base outcome) and `OutOfOrderQueueRunsThePinnedDriver` (a pinned run on an
  out-of-order `Queue` equals the direct driver bit for bit). Breaks:
  - B7 the empty-shape term dropped from `native` -> `EmptyShapesRunNoNativeFamily`, all dtypes;
  - B8 the in-order wrapper in `run()` disabled -> `OutOfOrderQueueRunsThePinnedDriver`, all dtypes.
  The event join of the wrapper (`enqueue(dep)`) is not separately covered.
- Known gaps: the `has_sg32` and `is_gpu` terms cannot go red on this hardware. There is no vendor
  Auto cell inside cuSOLVER's envelope (only non-square Hermitian input goes there).

## Gate results (threadripper02, GPU 0, 2026-10-05)

**(a) Correctness.** The vendor build (`build`) and the vendor-free build (`build-vf`) each ran
gesvd_candidates_tests, gesvd_tests, gesvdj_cta_tests, tuned_tables_tests, select_tests and gesvd's
callers' suites (linalg_layer_tests, error_model_tests, cond_tests). Failing names were compared
with the same targets built from `424a45bc` (`build-base`, `build-base-vf`):
- Vendor build: identical failing names. There is one failure on both:
  `CondTest/2.RandomHermitianTridiagonalLogCondSpectral` (float CUDA). gesvd_candidates_tests passes
  57/57 (65/65 after the review follow-up, re-run in both trees with identical failing names).
- Vendor-free build: identical failing names on both. These are 14 `gesvd_tests` NETLIB cases and
  24 `cond_tests` cases, because the host backend has no LAPACKE in this build.
  gesvd_candidates_tests passes.
- `scripts/facade_symbol_check.sh gesvd`: OK. The facade exports 8 gesvd symbols and the cuBLAS
  component exports none.
- Case lists differ only by the deleted `DefaultProviderRoutesSmallGeneralToJacobi` (x4) and the
  renamed `CtaRejects...` (x4).

**(b) Data gate** (`python3 -P tools/transcribe/gesvd_gate.py . <transcriber> 4000 <device>`).
4000 random OFF-GRID points (at least one of m, n off the grid; 30% dense in 20..80; herm N/L/U;
jobs N/A/T canonicalised as the entry point does; four dtypes). The old router (the transcriber's
`points` mode, the real `424a45bc` RouteTable code) is compared with the nearest-row walk through
a `can_run` mirror. The result is **100.00% in every dtype**, in both the vendor and the vendor-free
replay, on both sm_89 and sm_120 (float 1013, double 970, cfloat 1029, cdouble 988 points). There
are no disagreeing regions.

The gate can fail. With the 33 and 65 grid points removed it drops to float 96.8% and double 97.4%:
real general 33..47 maps to the n = 32 row (`jacobi`) where the old router says `blocked`.
Complex dtypes stay at 100% because the Jacobi cap lives in `can_run`.

**(c) Cross-check against the `424a45bc` binary.** Each cell ran as one process per build (a
throwaway probe, not committed: `$JOB/tmp/p5_gesvd_probe_tests.cc`), with `BATCHLAS_COVERAGE_OUT`
readback. 27 cells were checked: four dtypes; herm N/L/U; jobs N/A/T; m, n from 1 to 300 across
32/33, 64/65 and 48/40, 20/70, 300/20; batch 128-4096. 25 cells agree on the `reached` row (jacobi,
blocked, cta). The other two, cfloat 70x70 and cdouble 40x40 general, are vendor on `424a45bc`, and
both builds throw the identical cuSOLVER refusal ("requires m <= 32, n <= 32"). On the branch the
refusal comes from `gesvd_buffer_size`, which records no row.

## Found (pre-existing, not fixed here)

1. **The Hermitian drivers read the unreferenced triangle.** `gesvd_cta` with Upper, and
   `gesvd_blocked` with Lower at n <= 32 (8 and 32 tested; 48 is fine), return wrong singular values
   when the unreferenced triangle is not the mirror. `DISABLED_HermitianFamiliesIgnoreTheUnreferencedTriangle`
   reproduces it: float/double/cfloat/cdouble, cta U and blocked L at n = 8 and 32. Old and new Auto
   both route Hermitian Upper n <= 32 to `cta`, so a caller who stores only the upper triangle gets
   a wrong answer today. The driver source is unchanged on this branch, apart from the named
   constant. Suggested `known-defects.md` entry #14.
2. **cuSOLVER gesvdjBatched, values-only and non-square, faults** (`CUDA_ERROR_ILLEGAL_ADDRESS`).
   Reproduced on `424a45bc` with `BATCHLAS_GESVD_ROUTE=vendor`, float 8x4, jobs N/N, batch 3. Auto
   never sends such a shape there; the vendor envelope in the tests excludes it.

## Doc changes for the integrator

- flat-kernel-selection.md §12: add a "Phase 5, gesvd" paragraph from the sections above (families,
  `can_run`, keys/grid, both-device transcription, legacy `_PROVIDER` handling, gate numbers).
- tuned/README.md: list `gesvd.<dtype>.{sm_89,sm_120}.txt` as transcribed (`424a45bc`), CSV
  `tuned/transcribed/gesvd.csv`, transcriber `tools/transcribe/gesvd_transcribe.cc`, gate
  `tools/transcribe/gesvd_gate.py`.
- AGENTS.md §9: gesvd no longer has a RouteTable. Its pins are `BATCHLAS_GESVD_ROUTE` /
  `_PROVIDER` with select spellings, and a bad pin throws.
- known-defects.md: Found #1 (and #2 if wanted).
- Stale citations of the deleted header: `docs/design/vendor-free-status.md:115`
  (`route_gesvd.hh:100`, the wide-band rule, which is now the transcribed `blocked|vendor|jacobi`
  rows for real 33..64) and `docs/design/vendor-independence.md:137` (`GesvdShape`).
- Code comments citing the old router (`gesvd_dispatch`, `gesvd_supports_cta/_blocked`, "a forced
  provider resets to Auto") are already reworded on this branch: `src/extensions/gesvd_blocked.cc`
  (both thin refusals in `gesvd_cta`), `src/backends/netlib_lapack.cc` (Thin), `tests/gesvd_tests.cc`
  and `tests/gesvdj_cta_tests.cc`. The only remaining `route_gesvd.hh` mentions are in
  `tools/transcribe/gesvd_transcribe.cc`, which is built against `424a45bc` on purpose.
