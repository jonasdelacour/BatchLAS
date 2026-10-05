# gemv: flat kernel selection (phase 5), as built

Branch `flat-select-p5-gemv`, based on `424a45bc` (the flat-selection stack head: phases 1-2,
P3.0 select infra, posv, tuner, blackwell kernels, trsm). This page holds the as-built notes,
deviations, gate results and the doc changes the integrator should fold into
`docs/design/flat-kernel-selection.md` §12 and `tuned/README.md`.

## What changed

| | before (`424a45bc`) | after |
|---|---|---|
| decision | `RouteTable<Op::gemv, T>` in `include/batchlas/blas/dispatch/route_gemv.hh` + the shape builder/env adapter `src/backends/gemv_route.hh` | `src/ops/gemv/{choice.hh,gemv.cc}`; `select::choose` over `tuned/gemv.<dtype>.<device>.txt` |
| public entry | `gemv` in `src/dispatch/entry_points/level3.cc` | `src/ops/gemv/gemv.cc` (instantiated for CUDA, ROCM and NETLIB on the device family) |
| `preferred()` window | code: complex<double>, Trans/ConjTrans, 64 <= red <= 352, out >= 256, batch >= 320 | data: transcribed rows, identical on `sm_89` and `sm_120` |
| bad pin | silently fell through to the vendor (`PinningCtaOnAShapeCtaCannotServeFallsThroughSilently`) | throws (R6) |

Families (`choice.hh`), all fieldless, one per kernel driver:

| family | driver | `can_run` (R3) |
|---|---|---|
| `cta` | `sycl_gemv::gemv_native_cta` (bodies 3/5) | agreement checks, `gemv_cta_available<T>()`, `d.is_gpu`, `d.has_sg32`, trans != N |
| `direct` | `sycl_gemv::gemv_native_direct` (bodies 1/2/4) | agreement checks, `gemv_direct_available<T>()`; no GPU gate (vendor-free native_cpu) |
| `vendor` | `backend::gemv_vendor` | `d.has_vendor_blas` (level-3 library: cuBLAS / rocBLAS / netlib) |

"Agreement checks" are exactly what the deleted `gemv_op_shape` returned `nullopt` on, plus the
old `supports()` clauses: A homogeneous, `rows, cols >= 0`, `batch >= 1`, x and y batch equal to
A's, `x.size() == red`, `y.size() == out`. There is still no gemv validator (known-defects #1):
a non-conforming call goes to the vendor, as before, and vendor-free it throws
`throw_no_vendor_route` (as before, through the `choose` catch so coverage records a `miss`).

The body split (1/2/4 inside Direct, 3/5 inside Cta) and the segment width W stay derived in the
drivers (`BATCHLAS_GEMV_SEGT` still steers W); no int fields, because the old code never chose a
knob. gemv takes no workspace, so R5 has nothing to size.

Keys: `trans:exact out:log red:log batch:log`. out/red are y's and x's lengths (they swap with
trans, the old `out_len()`/`red_len()`); ConjTrans folds to T. All log weights are 1: the
transcribed grid is a full product with both sides of every threshold, so per-dimension nearest is
the decision whatever the weights.

Aliases: `native:cta` -> `cta`, `native:direct` -> `direct`. Bare `native` / `vendor` / `auto`
are select's class words; bare `native` is the best runnable non-vendor of the row (cta under
T/C on a sub-group-32 GPU, direct otherwise), which is what the old "first supported native route"
gave. There was never a legacy `BATCHLAS_GEMV_VARIANT`/`_PROVIDER` spelling.

Coverage row: `m = A.rows`, `n = A.cols`, `k = A.rows`, `transA`, backend, `heterogeneous_batch`,
as the old builder wrote them; native flags via `native_facts`.

## Tables

Maintainer decision: transcribed old routing for both devices. `tools/transcribe/gemv_transcribe.cc`
(host-only, compiled against the `424a45bc` headers that still hold `route_gemv.hh`) evaluates the
real `RouteTable<Op::gemv,T>` + `resolve_route` at every grid cell and derives the full ranked
order by exclusion (resolve with vendor, record winner, mark it unsupported, re-resolve; after the
vendor, the vendor-free walk ranks the rest). Device scenario: sub-group-32 GPU, both kernels
compiled; the predicates read no architecture and gemv has no capacity limit, so `sm_89` and
`sm_120` get the same rows (`tuned_tables_tests` asserts this).

Grid (both sides of every threshold the old predicates read, plus a coarse log grid):

- out: 1, 8, 64, **255, 256**, 1024, 4096, 32768
- red: 1, 8, 32, **63, 64**, 128, **352, 353**, 1024, 4096, 32768
- batch: 1, 16, 128, **319, 320**, 1024, 8192, 32768
- trans: N, T

1408 rows per table. Rankings: `cta|vendor|direct` on the 48 complex<double> T cells inside the
window, `vendor|cta|direct` on every other T cell, `vendor|direct` under N.

Regenerate:

```
g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/gemv_transcribe.cc -o /tmp/gt
/tmp/gt sm_89  > tuned/transcribed/gemv.sm_89.csv
/tmp/gt sm_120 > tuned/transcribed/gemv.sm_120.csv
python3 scripts/sweep_to_table.py --transcribe tuned/transcribed/gemv.sm_89.csv tuned/transcribed/gemv.sm_120.csv --sha 424a45bc
```

Both steps were re-run on this branch and reproduce the committed CSVs and tables byte for byte.
`scripts/sweep_to_table.py` gained a `GEMV` OpSpec (no sweep source).

## Deleted

- `include/batchlas/blas/dispatch/route_gemv.hh`, `src/backends/gemv_route.hh`, their two
  comment-density waivers.
- The `RouteGemv.*` suite in `tests/route_vocabulary_tests.cc` (16 tests). Ported into
  `tests/gemv_candidates_tests.cc`: HelperIsArmed (by `CanRunEqualsLaunch`), DirectHasNoGpuGate
  (`GemvCandidatesCpu.CpuQueueRunsDirectOrTheVendor`), CtaRequiresTransposedGpu... and
  ZeroExtent... (`CanRunFalsePinsThrow`, `PinnedCandidatesStraddleTheirLimits`),
  HeterogeneousBatch... (`HeterogeneousBatchHasNoNativeRoute`), the window/boundary/preferred
  tests (`AutoReadsTheTranscribedTable`, `GemvTranscribedTable.RowsHoldTheOldPreference`),
  OutLenAndRedLenSwap (`TraceKeyOutRedFollowTrans`), AutoTakesTheVendor...
  (`VendorFreeLastResortIsDirect`, `AutoReadsTheTranscribedTable`), BareNative...
  (`LegacyAliasesAndClassWords`). Dropped as asserting deleted mechanics:
  PinningCtaOnAShapeCtaCannotServeFallsThroughSilently (now throws, `ScopedPinBeatsTheEnvironment`),
  ShapeDoesNotShadowOpShapeFields, NativeAutoIsNotItselfASupportedRoute, OrderIsCtaThenDirectThenVendor,
  TheWindowNeverOutrunsTheCapability (the table cannot pick a non-runnable entry by construction).
- The `gemv_route` readback print in `gemv_tests.cc` (`SegTransCasesAreReachable`).

## Doc changes needed (for the integrator)

- `docs/perf/gemv.md`: a "Flat selection" paragraph was added under "What ships" (done on this
  branch); the `route_gemv.hh:line` citations below it now describe deleted code.
- `docs/design/known-defects.md:65` and `docs/design/vendor-independence.md:114` cite
  `gemv_route.hh` lines; the agreement checks now live in `can_run` (`src/ops/gemv/gemv.cc`).
- `docs/design/small-n-factorization-plan.md:1141` and the `orgqr_route.hh` waiver text cite
  `gemv_route.hh`'s density as a sibling; harmless history.
- flat-kernel-selection.md §12: add gemv to the migrated-op list; `tuned/README.md`: add the
  gemv tables (transcribed, both devices).

## Deviations

- None in behaviour beyond R6 (bad pins throw). The vendor is kept everywhere the old router
  chose it (cuBLAS gemv is at the DRAM roof).
- The off-grid gate uses a Python replica of `Table::nearest` (`sweep_to_table.nearest`); the
  C++ `Table::nearest` is checked on 12 on/off-grid keys per device by
  `GemvTranscribedTable.RowsHoldTheOldPreference`.

## Gate results (threadripper02, GPU 0 = RTX PRO 6000 Blackwell, sm_120)

**(a) Correctness.** Same targets built from this branch and from `424a45bc` (a `git archive`
copy), vendor (`build`) and vendor-free (`build-vf`) trees, run on GPU 0.

| binary | branch | base `424a45bc` |
|---|---|---|
| gemv_candidates_tests (vendor / vf) | 68 / 69 passed, 0 failed | n/a |
| gemv_tests | segfault at `GemvMatrixViewTest/7.SingleGemvNoTranspose` (known-defects #13) | same segfault, same case |
| gemv_tests `-*/7.*` | 326 passed | 326 passed |
| gemv_tests, `BATCHLAS_GEMV_ROUTE=native` | 372 passed | 372 passed |
| gemv_tests (vf) | 372 passed | 372 passed |
| select_tests (vendor, vf) | 37, 37 | 37, 37 |
| tuned_tables_tests (vendor, vf) | 13, 13 (+2 gemv tests) | 11, 11 |
| route_vocabulary_tests | 88 (16 `RouteGemv.*` deleted) | 104 |
| linalg_layer_tests (vendor, vf) | 22, 22 | 22, 22 |
| options_api_tests | 17; vf: 16 + `OptionsApi.Blas3OptionsMatchPositional` FAILED | identical |
| ortho_tests `-*/7.*` | 14; vf: 6 + 8 `Ortho*Test/{0..3}` (Backend 6) FAILED | identical |
| ortho_tests `*/7.*` with `BATCHLAS_GEMV_ROUTE=native` | 2 passed | 2 passed |

Failing-name diff branch vs base: empty in both trees. In the vendor build the complex<double>
CUDA cases that would launch `cublasZgemvStridedBatched` are skipped by name after a child-process
probe (known-defects #13); the vendor-free run covers that instantiation's native paths.

Deliberate breaks (restored from a saved copy, md5 `5a95cd0b...` verified):
1. `can_run(Cta)` without `transA != NoTrans` -> exactly `CanRunEqualsLaunch`,
   `CanRunFalsePinsThrow`, `PinnedCandidatesStraddleTheirLimits`, `ScopedPinBeatsTheEnvironment`
   red, for each of the 4 CUDA types (16 failures; 52 passed).
2. `key_of` "red" reading the out length -> `AutoReadsEveryKeyField` and
   `TraceKeyOutRedFollowTrans` red (float/double/cfloat; cdouble `TraceKeyOutRedFollowTrans`);
   complex<double> `AutoReadsTheTranscribedTable` / `AutoReadsEveryKeyField` then reach the
   cuBLAS Zgemv and segfault (known-defects #13) rather than failing cleanly.

**(b) Data gate.** Host replay: 3000 random points per dtype, off-grid in every coordinate (one
third over out, red in [1, 131072], batch in [1, 65536]; two thirds concentrated around the
window, out in [32, 4096], red in [16, 2048], batch in [32, 4096]), trans drawn from N/T/C. The
OLD answer is the transcriber binary (`--choices`, the real `RouteTable` + `resolve_route`); the
NEW answer is the nearest row of each shipped table (`sweep_to_table.nearest`) filtered by a
can_run model, in four device scenarios (sub-group 32 yes/no x vendor yes/no), for both the
`sm_89` and `sm_120` tables.

- Agreement: **100.00% in all 32 (device, dtype, scenario) cells** (3000/3000 each); the full
  ranked list equals the old exclusion-derived ranking at all 3000 points per table.
- 202 of the 12000 points sit in the old cdouble cta window.
- Armed: dropping the `red=63` rows from `gemv.cdouble.sm_89.txt` gives 2976/3000 (99.20%) with
  every disagreement at red in [47, 61] (old vendor, new cta), so the gate catches one missing edge.

**(c) Coverage cross-check.** One Auto gemv per process, `BATCHLAS_COVERAGE_OUT` `reached` row,
424a45bc binary vs this branch, 26 cells straddling every window edge plus the other dtypes:

- vendor build: 21/26 identical (`native:cta` = `native:cta`, `vendor:auto` = `vendor:vendor`);
  the other 5 (complex<double> just outside the window, and N) crash inside cuBLAS Zgemv in both
  binaries before coverage is written (known-defects #13); the branch's select trace names
  `vendor` for all 5, so both sides took the vendor.
- vendor-free build: 26/26 identical (cta for every T/C cell, direct for N), those 5 included.

`run_local_checks.sh`: all checks pass except `check_cmake_syntax`, which trips only on CMake's
generated files under this worktree's untracked `build-vf/` directory.
