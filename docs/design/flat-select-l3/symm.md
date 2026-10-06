# symm: flat selection, as built

Branch `flat-select-l3-symm`, from `ff340fc6` (merged PR #144). Working notes for the level-3
integration commit, which folds them into `docs/design/flat-kernel-selection.md` §12 and deletes
this file. Every `file:line` for the old code refers to `ff340fc6`.

## What moved

- `src/ops/symm/{choice.hh,symm.cc}` hold the whole path (R1): public `symm()` ->
  `shape::validate_product<std::invalid_argument>` -> `choose()` -> `TraceScope` -> one `std::visit`
  launch. `level3.cc` no longer defines symm (its ROCm block never instantiated it; `rocblas.cc` has
  no symm wrapper, and `src/extensions/symm.cc` keeps serving ROCm/MKL as before).
- Families (both `NoFields`; the old router chose no knob):
  - `expand`: `expand_mirrored<T, false>` into a lease on the queue's arena, then the public
    `gemm<B, T>` (`full * B` on the Left, `B * full` on the Right). Lifted from
    `symm_custom_dispatch.cc:74-133` and generalised from float to T, so double gains a native route.
  - `vendor`: `backend::symm_vendor<B, T>` (the per-item `cublas?symm` / `cblas_?symm` loop), never
    through `symm_vendor_cuda_raw` or `level3_vendor_fallback`.
- `candidates<T>()` is `{expand, vendor}` for float and double; `last_resort {"expand", "vendor"}`.
- The old `symm_custom_dispatch.{cc,hh}` is now unreferenced from the entry point. Per design D8
  it stays in the tree, dead, until the integration commit deletes it with the other dispatchers,
  `symm_vendor_cuda_raw` and the fallback/fused helpers (they are shared with the other three ops
  and with `cublas.cc` includes that this branch must not touch).

## can_run (R3)

| family | term | why |
|---|---|---|
| expand | `B == Backend::CUDA` | the old reach (ROCm and the host stay vendor) |
| expand | `d.is_gpu` | the expansion is a SYCL kernel wired for the GPU queue |
| expand | A, B and C homogeneous | see "Heterogeneous batches" |
| expand | `C.rows() >= 1 && C.cols() >= 1 && batch >= 1` | the old `problem_supported` (`m > 0 && n > 0`), plus trsm's empty-batch rule |
| expand | `batch <= kMaxGridBatch` (65535) | `expand_mirrored` puts the batch in SYCL dim 0 = CUDA grid z. Measured: a direct expand + gemm at batch 65535 runs, at 65536 it throws (`GridBatchCeiling`) |
| expand | `expansion_fits(q, k, batch, expanded_workspace_bytes<T>(q, k, batch))` | new: the old symm never checked it (`docs/perf/level3.md` debt 3). `BATCHLAS_EXPAND_MAX_BYTES` still lowers the ceiling, through `can_run` |
| vendor | `d.has_vendor_blas` | the library is compiled in |

There is no `symm_buffer_size`; `expand` leases its scratch from `Queue::workspace`, so R5 has
nothing to size.

### Heterogeneous batches

The design asked for `HeterogeneousBatchGoesToVendor` to be run against the old build before
relying on the homogeneity term. It was (a temporary case in `symm_tests`, ff340fc6 tree, float,
n = 16, batch 4, active orders 16/14/12/10, Left/Lower):

| heterogeneous | old `expand` pin | old `vendor` pin |
|---|---|---|
| A only | ran; equal to the vendor (7e-7); both are the storage-order product, off the active-order reference by 3.8 | ran (storage order) |
| B and C | **threw** `GEMM: incompatible per-batch matrix dimensions for heterogeneous dispatch` | ran (storage order) |
| all three | **threw** (same) | ran (storage order) |

So the old Auto **threw** on a heterogeneous B or C at batch >= 4 (the rule took the expansion),
and on a heterogeneous A it gave the vendor's storage-order answer. Neither honours per-item
extents, so the term stays on all three operands: Auto now takes the vendor (and, vendor-free,
throws `NoRouteError`). The vendor's own storage-order behaviour is the known-defects #12 class
(potrf/trsm), now also true of symm; the integration commit should add symm to that entry.
`HeterogeneousBatchHasNoNativeRoute` keeps the A-only wrong answer of the direct expansion as a
check, so the term is revisited if the expansion ever learns active extents.

## Keys, grid and tables

- Keys `form:exact m:log n:log batch:log`, with `form = sq|tall|wide` of C's (m, n)
  (`2*min >= max` is sq; else tall if m > 2n). The old rule's `squareish(m, n, k)` with
  `k = A.rows()` reduces to `squareish(m, n)` (k is m on the Left, n on the Right) and its
  `max_dim` to `max(m, n)`; side is never read, so it is not a key. The transcriber asserts the old
  choice is side-invariant at every cell.
- Grid: form x m, n in {1, 2, 4, ..., 128, 255, 256, 512, ..., 4096} x batch in
  {1, 2, 3, 4, 8, 128, 1024, 8192, 32768} = 5292 rows per (dtype, device). 255|256 and 3|4 straddle
  the two thresholds (`triangular_expand.hh:45-46`).
- A cell whose (m, n) contradicts its form holds the row of its form representative R_f
  (M = max(m, n): sq (M, M); tall (M, max(1, (M-1)/2)); wide the transpose). For M <= 2 the tall
  and wide representatives are squareish, so those 48 cells per table rank expand first at
  batch >= 4; no tall or wide query can snap there (a tall query has m >= 3, which snaps to 4).
- Rows: float `expand | vendor` where the old rule took the expansion (squareish and
  (batch >= 4 or max(m, n) >= 256)), else `vendor | expand`; double `vendor | expand` everywhere.
  sm_89 and sm_120 identical. Header `source=transcribed:ff340fc6`, CSV
  `tuned/transcribed/symm.csv` (both devices).
- `scripts/sweep_to_table.py`: OpSpec `SYMM` (key function `symm_key`), appended to `OPS`.
  `--check` passes on every file.

## Transcriber and data gate

`tools/transcribe/symm_transcribe.cc` (host g++, stub types) holds verbatim copies of
`symm_custom_dispatch.cc:36-72, 137-150, 152-191`, `triangular_expand.hh:45-63` and
`level3.cc:47-64`; `tools/transcribe/symm_gate.py --fidelity` checks each byte for byte against
`git show ff340fc6:<file>` (and fails on a one-character edit). Stubs model both boxes' build: no
MathDx (`symm_fused_try` -> NoKernel), no route pins, a GPU queue. Both files are committed for
provenance and are deleted in the integration commit with the rest of `tools/transcribe/`.

Gate (`symm_gate.py --gate`, seed 20261006): per dtype, 3000 log-uniform off-grid points and 3000
off-grid points crowding the thresholds, replayed through the old router (point mode) and through
`sweep_to_table.nearest` + a `can_run` model (homogeneous GPU, unlimited capacity).

| dtype | sample | old vendor-present | A: vendor-present agreement (sm_89, sm_120) | B: vendor-free, where the old build served | old vendor-free threw -> new |
|---|---|---|---|---|---|
| float | uniform | 2570 vendor, 430 expand | 3000/3000, 3000/3000 (100.00%) | 430/430 (100.00%) | 2570 -> expand |
| float | threshold band | 1998 vendor, 1002 expand | 3000/3000, 3000/3000 (100.00%) | 1002/1002 (100.00%) | 1998 -> expand |
| double | uniform | 3000 vendor | 3000/3000, 3000/3000 (100.00%) | none served | 3000 -> expand |
| double | threshold band | 3000 vendor | 3000/3000, 3000/3000 (100.00%) | none served | 3000 -> expand |

No point disagrees. Negative controls (float, sm_89, scenario A): removing the batch=4 rows gives
99.70% on the uniform sample and 97.73% on the band sample; removing the m=256 and n=256 rows gives
99.97% and 86.97%. A uniform sample puts little mass on a single threshold edge, so the band sample
is the one that shows the gate can fail.

## Pins (R6)

`BATCHLAS_SYMM_ROUTE` / `ScopedPin` take `auto`, `native` (-> expand), `vendor`, `expand`,
case-folded and trimmed. `cublasdx`, `triangular`, `gram`, `loop`, other ops' families and fields on
a fieldless family throw `invalid_argument`. Behaviour changes against ff340fc6:

- `cublasdx` throws `invalid_argument` (unknown family) instead of `batchlas::unsupported`.
- an `expand` pin on a CPU queue, a heterogeneous batch, a batch over 65535 or a scratch over the
  budget throws (it used to go to the vendor silently, or throw from the gemm).
- symm no longer reads `BATCHLAS_EXPAND_ROUTE` (hemm, herk, her2k still do).
- invalid shapes throw `invalid_argument` before `choose()` on every backend (netlib used to throw
  `runtime_error` from inside its host task).
- coverage: `chosen_algo` is the spelling (`expand`, `vendor`) as before; the scalar and backend
  columns are now real (the old recorder hard-coded F32/CUDA), and double rows exist.

Vendor-free gains: float and double symm on every homogeneous CUDA GPU shape (`expand`).

## Tests

`tests/symm_candidates_tests.cc` (label `blas`, typed float/double x NETLIB/CUDA; GPU cases skip on
the CPU): PinnedCandidatesStraddleTheirLimits (extents across the 32-wide mirror tile and 255|256,
the budget at exactly the lease and one byte short), EveryCombinationOnEveryCandidate,
PinnedRunIsTheDirectKernelBitForBit, SaturatingBatchIsBitIdenticalToItsRepresentative (1024 items),
CanRunEqualsLaunch, GridBatchCeiling, InvalidShapesThrowBeforeChoose, UnknownPinsThrow,
CanRunFalsePinsThrow, HeterogeneousBatchHasNoNativeRoute, ClassWordsAndSpellings,
ScopedPinBeatsTheEnvironment, AutoReadsTheTranscribedTable (18 cells x 2 sides around every
threshold), AutoReadsEveryKeyField (synthetic table; vendor build only, since vendor-free only expand
can run), TraceKeyIsCsShape, VendorFreeLastResortIsExpand,
CoverageRowCarriesBackendKeyAndNativeFlags, CpuQueueRunsNoNativeFamily, and the GPU-free
SymmTranscribedTable.HoldsTheChoiceGridAndTheOldRule. Every operand sits at a padded `ld` and a
batch stride that is not `ld * cols`, rows != cols, alpha 1.5, beta -0.75, A's unreferenced triangle
and every pad hold a large finite poison, and everything outside C is compared bit for bit.

`symm_tests.cc`: the three ForcedExpand* cases pin through `ScopedPin`; RemovedRouteWordsThrow adds
`cublasdx`.

### Deliberate breaks

Each break was applied to `src/ops/symm/*`, the vendor tree was rebuilt and `symm_candidates_tests`
and `symm_tests` were run on GPU 1. The files were then restored and checked by md5
(symm.cc `8b581d5f`, choice.hh `2ace9b3c` before the comment trim). Red sets (f = float, d = double,
CUDA unless marked):

| break | red set |
|---|---|
| `form_of` with `3*min` | AutoReadsEveryKeyField f d, AutoReadsTheTranscribedTable f, SymmTranscribedTable.HoldsTheChoiceGridAndTheOldRule |
| `key_of` m and n swapped | AutoReadsEveryKeyField f d, TraceKeyIsCsShape f d |
| `expansion_fits` dropped (`... \|\| native`) | PinnedCandidatesStraddleTheirLimits f d, CanRunFalsePinsThrow f d |
| homogeneity term dropped | CanRunFalsePinsThrow f d, HeterogeneousBatchHasNoNativeRoute f (double's row ranks the vendor first) |
| grid-z batch term dropped | GridBatchCeiling f d |
| coverage `k` = C.cols | CoverageRowCarriesBackendKeyAndNativeFlags f d |
| vendor arm with uplo flipped | every case that runs the vendor: the vendor-ranked Auto cases, the vendor pins, CpuQueueRunsNoNativeFamily (NETLIB), and symm_tests' vendor comparisons |
| expand arm with Left/Right gemm swapped | every case that runs expand, and symm_tests' ForcedExpand* |

A first "fits" break (`native && fits` -> `native \|\| fits`) also removed the native terms and so
went wider (adding GridBatchCeiling, HeterogeneousBatchHasNoNativeRoute and the CPU queue); the narrow
version above replaced it.

## Gates

Run on the sm_120 box (RTX PRO 6000 Blackwell). The base is `ff340fc6`, built in the job's
`mainbase` trees. Targets: `symm_tests`, `symm_candidates_tests` (new), `select_tests`,
`tuned_tables_tests` and `options_api_tests`. symm has no library-internal caller
(`src/extensions/symm.cc` serves ROCm/MKL only), so there is no consumer suite to re-run.

| tree | base failing names | new failing names |
|---|---|---|
| vendor | none | none (symm_candidates_tests 37 passed, the rest skip on the CPU) |
| vendor-free | symm_tests: SymmTest/0-3.MatchesSymmetrizedGemmReference, SymmCudaCustomTest.ForcedExpandPath{MatchesVendor,IgnoresUnreferencedTriangle,OrdersExpansionOnOutOfOrderQueue}; options_api_tests: OptionsApi.Blas3OptionsMatchPositional | symm_tests: SymmTest/0,1 (NETLIB float, double: no netlib in this tree, so no route, as before) |

Vendor-free, the CUDA symm cases and OptionsApi.Blas3OptionsMatchPositional now pass, because expand
serves them. `python3 scripts/sweep_to_table.py --check` passes on every file. The comment-density
gate passes.

Live check: one Auto call per process (`BATCHLAS_COVERAGE_OUT`; the old double path records no row,
so its route is read from `BATCHLAS_KERNEL_TRACE`, where the expansion's `symm_cuda_custom.expand`
scope shows up on the old float expand cells). Cells: 24 float and 6 double, both sides, across
batch 3|4, max(m, n) 255|256 and every form edge (100x49 | 100x50, 600x299 | 600x300, ...).

- Vendor build: the `reached` chosen_algo is identical at all 60 cells (float 24 expand and 24
  vendor; double vendor).
- Vendor-free build: the 24 cells where the old build served (expand) are identical. At the other 36
  the old build threw `NoRouteError` and the new one takes expand.
