# trmm: level-3 flat selection, as built

Branch `flat-select-l3-trmm`, from `origin/main` = `ff340fc6`. These notes are folded into
`docs/design/flat-kernel-selection.md` §12 by the integration commit; until then they are the
record. Every `file:line` is this branch's.

## What moved

- **Entry point.** `trmm<B, T>` left `src/ops/level3/level3.cc` for `src/ops/trmm/trmm.cc`
  (R1: validate, empty-problem return, `choose()`, `TraceScope`, one `std::visit` launch). It is
  instantiated for CUDA, ROCm and NETLIB in all four dtypes, as before.
- **Families** (`src/ops/trmm/choice.hh`), all `NoFields`, one list for every dtype:
  - `triangular`: `detail::trmm_triangular_tiles<T>` (Side::Left only). Its row tile
    (`trmm_row_tile`, `BATCHLAS_TRMM_TILE_M`) stays derived.
  - `expand`: `expand_triangular<T>` into a Queue workspace lease, then the public `gemm` at
    beta 0 (`launch_expand`, `trmm.cc:106`). This body was the first half of cublas.cc's
    `trmm_vendor_impl`, which is now the `cublas?trmm` loop only (D3).
  - `vendor`: `backend::trmm_vendor<B, T>`, never `*_vendor_cuda_raw` or the level-3 fallback.
- **Removed from the live path:** level3.cc's float gate and `record_level3_route` decline row;
  cublas.cc's forced-cuBLASDx branch and non-float tile branch in `trmm_vendor`. The old
  `trmm_custom_dispatch.{cc,hh}` and `trmm_vendor_cuda_raw` stay in the tree, dead and
  unreferenced from any live path, for the integration commit to delete (D8).

## can_run (R3), `trmm.cc:59-75`

| family | terms |
|---|---|
| common native | `B == Backend::CUDA` (D6), `d.is_gpu`, A, B and C homogeneous, `d.max_wg >= 256`, C extents >= 1, `1 <= batch <= 65535` |
| triangular | + `detail::trmm_tiles_supported(A, B, C, side)` (Side::Left, shapes, homogeneous) |
| expand | + `expansion_fits(q, order, batch, expanded_workspace_bytes)` (n^2 batch <= INT_MAX; bytes <= GLOBAL_MEM/4 or `BATCHLAS_EXPAND_MAX_BYTES`) |
| vendor | `d.has_vendor_blas`, and on CUDA a homogeneous batch |

- **Grid ceiling, measured** on the main binary (`ff340fc6`), float Left (tile kernel) and
  float/cdouble Right (expansion), order 1-3: batch 65535 runs and is correct; batch 65536
  throws `sycl::exception: Number of work-groups exceed limit for dimension 2 : 65536 > 65535`.
  The term stays. `GridBatchCeiling` launches at 65535, refuses at 65536 (the direct launch
  throws), and Auto takes the vendor there (vendor-free: `NoRouteError`).
- **Heterogeneous batch (design §2.3/§2.4 check).** On the main binary the old expansion
  threw `invalid_argument` ("GEMM: incompatible per-batch matrix dimensions"), so the
  homogeneity term on `expand` stays. The vendor loop, reached on main only with
  `BATCHLAS_EXPAND_MAX_BYTES=0`, runs every item at the full storage order: wrong for
  Upper/Left (maxrel 0.92) and Lower/Right (0.76); right only where the extra storage is
  never referenced (Lower/Left). **Deviation:** the vendor family refuses a heterogeneous
  batch on CUDA, so Auto keeps throwing (now `runtime_error` "no runnable kernel", or
  `NoRouteError` vendor-free) instead of moving to a silent wrong answer. ROCm and NETLIB
  vendors keep accepting one, as before. For `known-defects.md` (integrator): the cuBLAS
  trmm loop shares known-defect #12's full-storage-order behaviour.
- **Empty problem.** `trmm.cc:158` returns a no-op event for batch 0 or an empty C under any
  pin (the gemm precedent, `src/ops/gemm/gemm.cc:202`). Measured on main: batch 0 threw a
  `sycl::exception` from the tile kernel and from the expansion; on this branch before the
  return it reached the cuBLAS loop, which faulted (`CUBLAS error: 7`, then an abort).

## Keys, grid, tables

- Keys `side:exact order:log:2 q:log batch:log`; order = A.rows(), q = B.cols() (Left) or
  B.rows() (Right).
- Grid: order {1,16,32,64,128,256,512,1024,2048} x q {1,16,128,1024,4096} x batch
  {1,128,1024,32768} x side = 360 rows per (dtype, device). The old rule read only the side
  (and capacities), so there is no threshold to straddle beyond L|R.
- `tuned/trmm.<dtype>.{sm_89,sm_120}.txt`, `source=transcribed:ff340fc6`, one CSV for both
  devices (`tuned/transcribed/trmm.csv`, geqrf precedent). Rows: Left
  `triangular | expand | vendor`, Right `expand | vendor`, every dtype and both devices.
- last_resort `{expand, triangular, vendor}`: expand serves both sides.

## Transcriber and gates

- `tools/transcribe/trmm_transcribe.cc` (committed for provenance; the integration commit
  deletes it and records `git show <I^>:tools/transcribe/trmm_transcribe.cc` in
  `tuned/README.md`). Host-only, no BatchLAS headers; nine VERBATIM blocks of `ff340fc6`
  (level3.cc:178-206, trmm_custom_dispatch.cc:21-56, 78-110, 114-207, level3_vendor_fallback.cc:86-102,
  cublas.cc:661-662, 712-751, 785-795, trmm_triangular_tiles.hh:352-373) on stub types, built
  once per flavour (`-DVENDOR_BUILD=0|1`). Build lines are in its header. The template it is
  generated from splices the blocks with `git show`; the cuBLASDx fused kernel is stubbed
  absent (MathDx is absent on both boxes).
- `tools/transcribe/trmm_gate.py --fidelity`: 9 blocks, 0 mismatches.
- **Data gate** (`trmm_gate.py --data`): 2500 random off-grid points per (dtype, device),
  log-uniform order and q in [1, 4096], batch in [1, 32768], side uniform; new choice =
  `sweep_to_table.nearest` + a can_run model.
  - A, vendor present: 2500/2500 (100.00%) for all 8 (dtype, device) pairs.
  - B, vendor-free where the old build served (float Left only): 1253/1253 (sm_89) and
    1296/1296 (sm_120) = 100%. Where the old build threw: float Right -> expand; double,
    cfloat, cdouble Left -> triangular, Right -> expand (counts in the gate log).
  - **Negative control (deviation):** the design's "delete the side=R rows" cannot fail
    for trmm: a Right point then snaps to an L row, whose `triangular` can_run refuses, and
    the next entry is `expand`, the old choice. Measured: still 100%. Deleting the side=L
    rows drops A to 49.2-50.7% for every dtype, so that is the control used.
- **Live cross-check**: 24 cells, one process per cell, on the main binary and this branch
  (coverage `reached` rows; the old non-float path records none, so its route is read from
  the kernel trace). 20 report the same route, with bit-identical residuals. 3 are the
  expected rename: float Right went from `vendor,vendor` to `native,expand`. They have the
  same residual, because the old "vendor" was the expansion. 1 differs on purpose: float
  Left at batch 65536 threw a `sycl::exception` on main and runs the vendor here. cdouble
  Right at 65536 also threw on main, and its "same" label is the classifier's default.

## Pins

`auto`, `native`, `vendor` (class words), `triangular`, `expand`. `cublasdx`, `gram` and every
removed word throw `invalid_argument` (`UnknownPinsThrow`, `trmm_tests` RemovedRouteWordsThrow,
which now lists `cublasdx` and no longer `expand`). A `triangular` pin on Side::Right, an
`expand` pin over the scratch budget, and a native pin on a heterogeneous batch throw "cannot
run this shape".

## Behaviour changes

- `BATCHLAS_TRMM_ROUTE=vendor` means the `cublas?trmm` loop; `expand` is the old meaning of
  `vendor` (D3). Right-side Auto coverage reads `native,expand` where it read `vendor,vendor`.
- Double and complex Left now record a coverage row (the old cublas.cc tile branch recorded
  none); the scalar and backend columns are the real ones (the old recorder wrote F32/CUDA).
- `cublasdx` throws `invalid_argument` (was `batchlas::unsupported`, or a forced attempt).
- Batch > 65535 goes to the vendor (was a `sycl::exception` from either native launch).
- A heterogeneous batch on CUDA throws `runtime_error` "no runnable kernel" (was
  `invalid_argument` from the expansion's gemm, or a wrong answer from the loop when the
  expansion did not fit).
- An empty problem is a no-op (batch 0 threw a `sycl::exception`).
- A native pin on a NETLIB (CPU) queue throws "cannot run this shape"; the old router never
  read `BATCHLAS_TRMM_ROUTE` off CUDA.
- **Vendor-free gains:** every dtype, both sides (float Left was the only route before).

## Tests

- `tests/trmm_candidates_tests.cc` (label `blas`): PinnedCandidatesStraddleTheirLimits,
  ExpandFitStraddlesTheScratchBudget (exact bytes and bytes - 1), EveryCombinationOnEveryCandidate
  (side x uplo x N/T/C x diag, non-natural ld and stride on A, B and C, alpha with an
  imaginary part, A's unreferenced triangle and a Unit diagonal poisoned with a large finite
  value, everything outside C checked bit for bit), PinnedRunIsTheDirectKernelBitForBit,
  SaturatingBatchIsBitIdenticalToItsRepresentative (batch 1024, every row tile and the
  expansion), CanRunEqualsLaunch, GridBatchCeiling, UnknownPinsThrow, CanRunFalsePinsThrow,
  InvalidShapesThrowBeforeChoose, EmptyProblemIsANoOp, HeterogeneousBatchHasNoRoute,
  ClassWordsAndSpellings, ScopedPinBeatsTheEnvironment, AutoReadsTheTranscribedTable,
  AutoReadsEveryKeyField, TraceKeyQFollowsSide, VendorOnlyTableFallsToTheLastResort,
  CoverageRowCarriesBackendKeyAndNativeFlags, CpuQueueRunsNoNativeFamily (NETLIB), and the
  GPU-free TrmmTranscribedTable.RowsHoldTheChoiceGridAndTheOldPreference.
- `tests/trmm_tests.cc`: ForcedCuBLASDxPathMatchesVendor and its `trmm_cublasdx_fused.hh`
  include deleted; the no-scratch sweep runs only with a vendor library (vendor-free there
  is no no-scratch route for Side::Right).

## Deliberate breaks (AGENTS §8)

Each break was applied to `trmm.cc` (or the tables), followed by a rebuild and a run of
`trmm_candidates_tests` and `trmm_tests` in the vendor tree. The source was then restored and
md5-verified. Red sets, per CUDA dtype (all four go red unless noted):

| break | red |
|---|---|
| triangular can_run ignores side | CanRunEqualsLaunch, CanRunFalsePinsThrow, PinnedCandidatesStraddleTheirLimits, ScopedPinBeatsTheEnvironment |
| key q always B.cols() | AutoReadsEveryKeyField, TraceKeyQFollowsSide |
| expand fit term dropped | CanRunFalsePinsThrow, ExpandFitStraddlesTheScratchBudget |
| grid ceiling term dropped | GridBatchCeiling |
| vendor arm flips uplo | every test that runs the vendor: 8 candidates tests, CpuQueueRunsNoNativeFamily (NETLIB), trmm_tests NETLIB and the CUDA no-scratch sweep |
| coverage k = C.rows() | CoverageRowCarriesBackendKeyAndNativeFlags |
| vendor accepts heterogeneous | HeterogeneousBatchHasNoRoute |
| expand Left ignores transA | EveryCombinationOnEveryCandidate, ExpandFitStraddlesTheScratchBudget, PinnedRunIsTheDirectKernelBitForBit, SaturatingBatchIsBitIdenticalToItsRepresentative |
| tables: expand dropped from Right rows | AutoReadsTheTranscribedTable, CoverageRowCarriesBackendKeyAndNativeFlags, TrmmTranscribedTable |

## Failing names against main

The same targets were run from the `ff340fc6` builds (`mainbase/build`, `mainbase/build-vf`):
trmm_tests, trmm_candidates_tests, tuned_tables_tests, select_tests, options_api_tests,
error_model_tests, ormqr_tests and ormqr_blocked_tests.

- **Vendor tree:** everything passes on main and here. trmm_tests has 17 cases here, against 18
  on main, because the cuBLASDx test was deleted.
- **Vendor-free tree:** no test newly fails. 8 CUDA trmm_tests cases are fixed:
  AllCombinations and IgnoresUnreferencedTriangleAndUnitDiagonal for all four dtypes. Main
  failed them on every dtype except float Left. The netlib-less NETLIB instantiations of
  trmm_tests, options_api Blas3OptionsMatchPositional, ormqr_tests and ormqr_blocked_tests
  fail identically in both. trmm_candidates_tests passes all 81 cases in both trees.

## Conflict notes for the integrator

S0 was not on `flat-select-level3` when this branch started (it pointed at `ff340fc6`), so
this branch carries its own share of it: the level3.cc removal of trmm and its instantiation
macro `ALL_TYPE_OPS_ONE`, `target_sources(... ops/trmm/trmm.cc)` in `src/CMakeLists.txt`, the
TRMM OpSpec appended to `OPS` in `scripts/sweep_to_table.py`, the trmm registry entry in
`tests/tuned_tables_tests.cc`, and `trmm_candidates_tests` in `tests/CMakeLists.txt`. The
other three op branches touch the same lines.

Left for the integration commit: `error_model_tests.cc:409` cites `tests/trmm_tests.cc:126`,
the message check inside the deleted cuBLASDx test; the CUDA::cudart regex for `trmm_tests`
in `tests/CMakeLists.txt` (it no longer includes a cuBLASDx header); the
`EveryOpShipsATableForEveryDtypeOnEveryShippedDevice` op list; `tuned/README.md`.
