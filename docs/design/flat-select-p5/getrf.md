# getrf: flat kernel selection (phase 5), as built

Branch `flat-select-p5-getrf`, based on `424a45bc` (the flat-selection stack head). This page
holds the as-built notes, deviations, gate results and the doc changes the integrator should fold
into `docs/design/flat-kernel-selection.md` §12 and `tuned/README.md`. It mirrors posv and trsm
(`src/ops/{posv,trsm}/`).

## What changed

| Piece | File |
|---|---|
| Vocabulary: `Tiny`, `Cta`, `Blocked`, `Vendor`, all fieldless; aliases; last resort; keys; grid | `src/ops/getrf/choice.hh` |
| `getrf` / `getrf_buffer_size` public entry points, `can_run`, `choose`, `launch`, `workspace`, `native_facts` | `src/ops/getrf/getrf.cc` |
| Removed from the facade (with `getrf_throw_native_unimplemented`) | `src/dispatch/entry_points/factorization.cc` |
| Deleted: the old table and its shape builder | `include/batchlas/blas/dispatch/route_getrf.hh`, `src/backends/getrf_route.hh` |
| Transcriber (old `RouteTable<Op::getrf,T>` + `resolve_route`, ranked per cell) | `tools/transcribe/getrf_transcribe.cc` |
| Transcriber CSV, 8 tables (4 dtypes x sm_89, sm_120, identical rows) | `tuned/transcribed/getrf.csv`, `tuned/getrf.<dtype>.<device>.txt` |
| OpSpec `GETRF` | `scripts/sweep_to_table.py` |
| Off-grid data gate | `tools/transcribe/getrf_gate.py` |
| Tests | `tests/getrf_candidates_tests.cc` (new), `tests/getrf_tests.cc`, `tests/route_vocabulary_tests.cc`, `tests/tuned_tables_tests.cc` |
| `factor_bench`: getrf pins parsed as choice spellings; gesv `composed` pins `tiny` only where it fits | `benchmarks/factor_bench.cc` |

## Families (one per kernel driver)

- `tiny` = `getrf_tiny_dispatch` (register-resident, n <= 32, cdouble 16).
- `cta` = `getrf_cta_dispatch` (SLM-resident, n <= the occupancy-scaled SLM ceiling).
- `blocked` = `getrf_blocked_dispatch` with the public `gemm` and `trsm` injected.
- `vendor` = `backend::getrf_vendor` (cuBLAS/cuSOLVER, rocSOLVER, LAPACKE).

No int fields: the old router chose no knob. The leaf (`BATCHLAS_GETRF_LEAF`), the laswp mode,
the blocking factor and the tiny bucket stay derived inside the drivers.

`can_run` (R3), each clause the driver's own check:
- every native family: backend != NETLIB (LAPACKE writes int64 pivots, the native tiers packed
  1-based int32), GPU, sub-group size 32 offered, homogeneous batch, square, n >= 1, batch >= 1;
- `tiny`: n <= `getrf_tiny_max_n<T>()` and MAX_WORK_GROUP_SIZE >= `kGetrfTinyWgSize` (64; new
  constant in `getrf_native.hh`, `static_assert`-ed equal to the kernel's `kTinyWg`);
- `cta`: n <= `getrf_cta_max_n_for_slm<T>(slm_budget)`;
- `blocked`: `getrf_blocked_available<T>()` and `getrf_cta_max_n_for_slm<T>(slm_budget, 1) >= 1`;
- `vendor`: `Device::has_vendor_solver`, which getrf fills with
  `factorization_vendor_available<B>` (cuBLAS + cuSOLVER on CUDA), via a local `device_of<B>`.

Last resort: `blocked`, `vendor`. Aliases: `native:tiny`, `native:cta`, `native:blocked`.
A vendor-free build with nothing runnable (non-square, heterogeneous, CPU queue) throws
`dispatch::NoRouteError` through `throw_no_vendor_route`, as before, so coverage `miss` rows
are kept.

## Tables (maintainer decision: transcribed, both devices)

Keys `n:log:3 batch:log` (work ~ n^3 batch; no categorical key). The old predicates read no
architecture, so one transcription is written for sm_89 and sm_120, `source=transcribed:424a45bc`.
CTA capacity unlimited in the transcription (`can_run` applies the device's ceiling at run time);
the tiny ceiling is a build constant (32, cdouble 16) and is applied.

Grid: both sides of every threshold the old predicates read, plus a coarse log grid.
- n: 1 2 3 4 | 5 6 7 | 8 | 9 12 16 | 17 24 | 25 32 | 33 48 64 96 128 192 255 | 256 384 511 | 512 768 1024
  (tiny window float 5..32, cfloat 5..7 and 9..24; cdouble tiny ceiling 16; double CTA tier
  order <= 32 in the vendor-free walk; blocked float >= 256, cfloat >= 512 or >= 256 at batch >= 256).
- batch: 1 8 64 128 255 | 256 512 2048 8192 32768.

Row order per cell: the old Auto choice with the vendor present, then the vendor, then the
remaining natives in the order the old vendor-free walk (`native_tier_preferred`, then the plain
walk) takes them once each higher one is removed. So a vendor-free build, or a device where the
first native cannot run, lands where the old router did.

Regenerate: build the transcriber against a 424a45bc tree (`g++ -std=c++20 -I$OLD/include
-I$OLD/build/include tools/transcribe/getrf_transcribe.cc`), `gt sm_89 sm_120 >
tuned/transcribed/getrf.csv`, then `scripts/sweep_to_table.py --transcribe
tuned/transcribed/getrf.csv --sha 424a45bc`. `--check` passes.

## Deviations from the old router

1. A concrete pin that cannot run throws (R6). The old router sent a forced, unsupported route to
   `automatic()`.
2. `getrf_buffer_size` returns exactly the chosen family's need (R5), not
   max(every supported native tier, vendor).
3. `blocked`'s SLM clause is the driver's `getrf_cta_max_n_for_slm<T>(budget, 1) >= 1` (the panel
   leaf's argmax slots at the whole budget), not the old occupancy-scaled `cta_max_n >= 1`. Both
   are true on every sm_89/sm_120 dtype; they differ only on a device with a tiny SLM.
4. `tiny` also requires MAX_WORK_GROUP_SIZE >= 64 (the driver threw there; the old table did not
   check it).
5. The trailing-update gemm seam of `getrf_blocked_dispatch` is mandatory (it was defaulted to
   `sycl_gemm::gemm_custom`), the same change as geqrf's. getrf injects the public `gemm`, which
   picks its own kernel. The direct-refusal test now also asserts the empty-gemm-seam throw.
6. `factor_bench` gesv `composed` pins getrf `tiny` only where it fits (`native` above), because
   a `tiny` pin past the ceiling now throws.

## Tests

- `tests/getrf_candidates_tests.cc` (label `blas`), design §8: pinned candidates straddling
  4/5, 8/9, 16/17, 24/25, 32/33 and the CTA ceiling; pinned run == direct driver bit for bit;
  saturating batch 1024 bit-identical; `CanRunEqualsLaunch`; heterogeneous and non-square
  (ported from `RouteGetrf.Squareness*`); exact workspace in a poisoned arena (with and without
  `info`); unknown and can_run-false pins throw; NETLIB backend runs no native family (ported
  from `RouteLuPivotFormat`); legacy aliases and class words via `ScopedPin` and the environment;
  `ScopedPin` beats the environment; Auto on the real device against the old router at 21
  off-grid cells (the port of `getrf_tests`' `TinyRoutesInsideItsMeasuredWindowAndNowhereElse`,
  `RouteTableAndTheVendorFreeFallback` getrf half, and `RouteGetrf.*` windows, vendor-present and
  vendor-free); `AutoReadsEveryKeyField`; vendor-free last resort; coverage row (backend, key,
  native flags); CPU queue; `GetrfTranscribedTable.RowsHoldTheOldPreferenceOnBothDevices`
  (Table::nearest on both devices at each threshold).
- Deleted: `route_vocabulary_tests` `RouteGetrf.*` (synthetic `GetrfShape` cases) and getrf's
  half of `RouteLuPivotFormat`; `getrf_tests` `TinyRoutesInsideItsMeasuredWindowAndNowhereElse`
  and the `getrf_route` readbacks (the bit-exact guards stay; a pin that cannot run now throws).
- `tuned_tables_tests`: getrf candidates, key names, and
  `GetrfTablesHoldExactlyTheChoiceGridOnBothDevices` (sm_89 rows == sm_120 rows).

## Gate results (2026-10-05, RTX 4090 GPU 0)

**(a) Correctness.** Targets `getrf_candidates_tests getrf_tests (+ getrf_tests_native) gesv_tests
inverse_tests linalg_layer_tests options_api_tests select_tests tuned_tables_tests
route_vocabulary_tests resident_capacity_tests`.
- `build`: 11/11 pass. `getrf_candidates_tests`: 69 pass, 68 skipped (the NETLIB instantiations
  on a GPU queue and the CUDA ones of the CPU suite).
- `build-vf`: 10/11; the one failure, `OptionsApi.Blas3OptionsMatchPositional`, also fails in the
  424a45bc vendor-free baseline. That baseline's other failure,
  `LuTest/{4,6}.TinyRoutesInsideItsMeasuredWindowAndNowhereElse`, is a test this branch deletes
  (ported into `AutoReproducesTheOldRouterOnTheRealDevice`). The vendor-present baseline is all
  green, the same as this branch. No new failing names.
- Deliberate breaks (restored from a saved copy, md5 `93e0b3c0...` checked): `key_of`'s batch fixed
  at 128 makes `AutoReadsEveryKeyField` red for all 4 dtypes and `AutoReproducesTheOldRouterOnTheRealDevice`
  red for cfloat only (its batch-gated 256..511 window). A `cta` `can_run` that admits one past the
  ceiling makes `CanRunEqualsLaunch`, `CanRunFalsePinsThrow` and `PinnedCandidatesStraddleTheirLimits`
  red. Nothing else turned red.

**(b) Data gate** (`tools/transcribe/getrf_gate.py`, oracle = the transcriber's `--eval` mode built
against 424a45bc): 3000 random log-uniform off-grid points per dtype and device (n in [1, 2048],
batch in [1, 65536]; about 1850-1900 per set are off-grid in both keys). Each point is checked
vendor-present and vendor-free, at CTA capacity unlimited, 48 and 24, which is 18000 comparisons
per dtype and device. **Agreement: 100.00% in every dtype on both devices**, with no disagreeing
region. The grid is a full Cartesian product with points on both sides of every threshold, so
nearest-point lookup separates per key and reproduces each step exactly. Check that the gate can
fail: removing the n = 4, 25, 255, 511 rows from `getrf.cfloat.sm_89.txt` drops it to 95.70% and
names the 256/512 rows. The file was restored and md5-checked afterwards.

**(c) Coverage cross-check** (`factor_bench getrf <t> n n 0 batch 1 --arms=auto`, coverage
`reached` rows, 424a45bc binary against this branch): 22 cells straddling every threshold (float
4/5, 32/33, 100, 255/256, 600; cfloat 8/9, 24/25, 300 at batch 128/512, 512; double 16, 64, 520;
cdouble 8, 16/17, 300). **22/22 agree** on the origin and the kernel: native tiny at float 5 and 32
and cfloat 9 and 24; native blocked at float 256 and 600, cfloat 300 at batch 512 and cfloat 512 at
batch 128; vendor everywhere else. The only textual difference is the algorithm column of a vendor
row, which reads `auto` in the old router and `vendor` (the choice spelling) in the new one.

## Doc changes for the integrator

- `flat-kernel-selection.md` §12: add a "Phase 5, getrf" paragraph from this page (families,
  vendor flag = factorization library, deviations 1-6).
- `tuned/README.md`: list `getrf.<dtype>.sm_89.txt` and `.sm_120.txt` as transcribed from 424a45bc.
- `docs/perf/lu.md`, `docs/design/vendor-independence.md`, `vendor-free-status.md`,
  `small-n-factorization-plan.md`, `docs/perf/qr.md` still cite `route_getrf.hh` /
  `getrf_route.hh`; the windows they quote now live in `tuned/getrf.*.txt` rows.
- Comment-only references to `GetrfShape::has_sg32` / `getrf_route.hh` remain in
  `src/backends/{getrs,getri,orgqr}_route.hh`; those files belong to ops other agents migrate, so
  they were left alone. They go when those headers are deleted.
