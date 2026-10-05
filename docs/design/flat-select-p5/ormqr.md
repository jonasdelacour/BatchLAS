# P5: ormqr flat selection (as built)

Branch `flat-select-p5-ormqr`, based on `424a45bc`. Notes for the integrator who merges the P5
branches and folds them into `docs/design/flat-kernel-selection.md` §12. This file is the only
place the branch records its deviations. AGENTS.md, the design doc and `tuned/README.md` are not
edited here.

## What changed

- **Vocabulary** (`src/ops/ormqr/choice.hh`): `Blocked` (`ormqr_blocked`: larft plus level-3 WY
  updates) and `Vendor` (cuSOLVER / rocSOLVER / LAPACKE `ormqr`/`unmqr`). Both are `NoFields` and
  the same for every dtype. Blocked's WY width is still derived (a positive `block_size_hint`
  clamped to [1, k], else `tuning::ormqr_block_size_for_n(A.rows())`). The old code never chose it,
  so it is not a field. Aliases: `native:blocked` -> `blocked`, `vendor:auto` -> `vendor`.
  `last_resort {"blocked", "vendor"}`.
- **Keys**: `side:exact trans:exact m:log k:log q:log batch:log`. Here `m` is A.rows (the order of
  Q), `k = min(A.rows, A.cols)` (the reflectors) and `q` is the extent of C that Q does not act on
  (C.cols for Left, C.rows for Right). Work grows as m·k·q·batch, so every weight is 1. T and C stay
  apart, because complex T is not ormqr. The old predicates read only `trans` (for complex) and
  `is_gpu`. The log keys exist so a future timed table can use them, and they move no transcribed
  decision.
- **`src/ops/ormqr/ormqr.cc`** now holds the public `ormqr` and `ormqr_buffer_size`. They were
  header templates in `functions/ormqr.hh`, instantiated from `src/dispatch/entry_points/eigen.cc`.
  Both header bodies are deleted, and so are `ormqr_dispatch`, `ormqr_buffer_size_dispatch`,
  `ormqr_route`, `ormqr_op_shape` and `resolve_ormqr_block_size`. The path is now `choose()` ->
  `workspace()` / `launch()` through `std::visit`. Sizing and running use the same `choose()` (R5).
  The out-of-order-queue sequencing of the old `ormqr_dispatch` is kept.
- **`can_run`** (R3): Blocked is `is_gpu && !(complex && Trans)`. Vendor is
  `has_vendor_solver && !(complex && Trans)`, with `factorization_vendor_available<B>`, i.e. cuBLAS
  plus cuSOLVER, as the solver flag. The drivers do their own dimension checks.
- **Deleted**: `include/batchlas/blas/dispatch/route_ormqr.hh` (an installed header) and its
  comment-density waiver. ormqr never had a `src/backends/ormqr_route.hh`: its shape was built in
  `functions/ormqr.hh` (`ormqr_op_shape`). `Op::ormqr` stays in the enum, and so does
  `legacy_variable_for(Op::ormqr)`, for phase 5.
- **Tests**: the new `tests/ormqr_candidates_tests.cc` (labels `ortho`; typed over 4 dtypes x
  backends, plus a CPU suite and a table suite). The two old-predicate tests in
  `tests/ormqr_tests.cc` were ported: `BufferSizeAgreesWithDispatchUnderAnUnmatchedForcedRoute` ->
  `BufferSizeIsTheChosenFamilysAndRuns` + `UnknownPinsThrow` (an unmatched `cta` now throws), and
  `ForcingCannotSelectAnUnsupportedRoute` -> `ComplexTransIsRefusedUnderEveryPin` +
  `CanRunEqualsLaunch` + `CpuQueueRunsNoNativeFamily`. Because those tests are gone,
  `tests/ormqr_tests.cc` drops under the density ceiling, and its waiver line is removed.
  `tuned_tables_tests` gains `OrmqrTablesDeclareChoiceKeyNames` and
  `OrmqrTablesHoldExactlyTheChoiceGridOnBothDevices`.
- **Tables**: `tuned/ormqr.<dtype>.{sm_89,sm_120}.txt`, 540 untimed rows each,
  `source=transcribed:424a45bc`, generated from `tuned/transcribed/ormqr.csv` (4320 rows) by
  `scripts/sweep_to_table.py --transcribe` (new `ORMQR` OpSpec). The transcriber
  (`tools/transcribe/ormqr_transcribe.cc`, host g++ against the 424a45bc headers) rebuilds the CSV
  with an identical md5 (`c48be05d...`), and `--check` passes.

## Behaviour changes visible to callers (deviations)

1. **Complex `Transpose::Trans` throws `batchlas::invalid_argument`** ("... is not an ormqr
   operation ... use Transpose::ConjTrans"). It is raised from both `ormqr` and
   `ormqr_buffer_size`, before `choose()`, under Auto and under every pin. Before, the old router
   sent it to the vendor, and cuSOLVER rejected it at run time (`CUSOLVER error: 3`; reproduced by
   the 424a45bc cross-check below). No library caller passes complex Trans (sytrd_sy2sb,
   band_reduction and ormqr_tests all use ConjTrans for complex). The complex `trans=T` rows of the
   transcribed tables (`vendor` alone, the old choice) can therefore never be read. They are kept
   so that the transcription stays verbatim.
2. **The vendor arm spells a real `ConjTrans` as `Trans`.** Real `ormqr` takes only N/T, and
   cuSOLVER returns status 3 on C. Under the old router a real ConjTrans reached the vendor only
   through a pin, where it failed. Auto never sent it there.
3. **Bad pins throw** (R6): `BATCHLAS_ORMQR_ROUTE=cta` and similar. These used to mean Auto.
4. **The legacy `BATCHLAS_ORMQR_PROVIDER` still pins**, below `BATCHLAS_ORMQR_ROUTE` and below a
   `ScopedPin`, and it goes through the same parser, so a bad value throws too
   (`LegacyProviderVariablePins`). It is read in `ormqr.cc` (`legacy_pin`). Delete it in phase 5
   together with `legacy_variable_for`.
5. **The vendor-free build calls `choose` with the real vendor availability.** The old
   `resolve_ormqr_route` took the `vendor_available = true` default (docs/perf/qr.md debt 12, now
   marked resolved). On a GPU nothing moves, because blocked was always first. On a host queue with
   no vendor, Auto throws `NoRouteError` and the coverage `miss` row is recorded.
6. **Coverage row**: the backend is now the real one. The old builder left it at `AUTO`.
   `native_route_existed` and `native_route_supported` come from `select::native_facts`. m, n and k
   are unchanged (A's extents and the reflector count).

## Known gaps

- **batch > 65535 with `blocked` throws** "Number of work-groups exceed limit for dimension 2"
  (float L T m=2 k=1 q=1 batch=65536, cross-check below). This is the same on 424a45bc. The driver
  does not check it itself, so `can_run` does not model it. It is a pre-existing kernel limit in
  ormqr_blocked's sub-kernels, outside this migration. Candidate for known-defects.md.
- `tests/route_vocabulary_tests.cc` (RouteOrgqr, around lines 971-1031) still cites
  `route_ormqr.hh` in comments and one message string. It is left alone because the orgqr agent
  owns that suite. `src/backends/orgqr_route.hh` and `route_orgqr.hh` cite it as well.
  **Integrator:** make sure the orgqr branch, or the merge, rewrites these citations.
  `docs/design/vendor-free-status.md` (the `is_gpu` list, the `preferred()` table row, and the
  `resolve_ormqr_route` debt, now struck through as resolved) was fixed on this branch.
- The `-ffp-model`/ROCm syntax checks were not run (no ROCm headers on this box).

## Gate

All runs were on threadripper02, GPU 0 (`CUDA_VISIBLE_DEVICES=0`), 2026-10-05.

**(a) Correctness.** These targets were run in this branch's vendor build (`build`), in its
vendor-free build (`build-vf`, `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`), and in the 424a45bc
baseline trees (`git archive` copy, vendor and vendor-free): `ormqr_candidates_tests`,
`ormqr_tests`, `ormqr_blocked_tests`, `ormqr_cta_tests`, `orgqr_tests`, `geqrf_tests`,
`sytrd_sy2sb_tests`, `sytrd_cta_tests`, `syev_two_stage_tests`, `select_tests`,
`tuned_tables_tests` and `route_vocabulary_tests`. The baseline has no `ormqr_candidates_tests`.
- Vendor build: 12/12 pass, and the baseline passes all of the same targets.
- Vendor-free build: the failing ctest targets are the same on both sides: `ormqr_tests`,
  `ormqr_cta_tests`, `ormqr_blocked_tests` and `orgqr_tests`. These are NETLIB instantiations with
  no host LAPACK route, plus `OrmqrBlockedTest/{4,5}`, whose reference is the vendor. The
  failing-name diff (baseline minus branch) is exactly the 8
  `OrmqrTest/*.BufferSizeAgreesWithDispatchUnderAnUnmatchedForcedRoute` names, which the branch
  deletes (ported). No name is added. `ormqr_candidates_tests` passes vendor-free.
- Deliberate breaks of `ormqr.cc`. Each was restored from a saved copy and md5-verified
  (`fbfa62e9...`). Red sets, where /4-7 means the CUDA dtype instantiations:
  - B1, the complex-Trans validation dropped -> `CanRunEqualsLaunch`,
    `ComplexTransIsRefusedUnderEveryPin` and `PinnedCandidatesStraddleTheirLimits` (complex only,
    /6-7).
  - B2, C folded to T in `key_of` -> `AutoReadsEveryKeyField` and
    `TraceKeyFollowsSideAndKeepsTrans` (/4-7), and `AutoReadsTheTranscribedTable` (/6-7: complex C
    lands on the vendor-only T rows).
  - B3, q always `C.cols` -> `AutoReadsEveryKeyField` and `TraceKeyFollowsSideAndKeepsTrans`.
  - B4, no real ConjTrans -> Trans for the vendor -> `PinnedCandidatesStraddleTheirLimits`,
    `PinnedRunIsTheDirectKernelBitForBit`, `OutOfOrderQueueIsSequenced` and
    `AutoReadsEveryKeyField` (real only, /4-5).
  - B5, block-size hint ignored -> `PinnedRunIsTheDirectKernelBitForBit`. The first run was green;
    a hinted spec where the hint differs from the ladder (m=100, k=37, hint 5) was then added.
  - B6, blocked allowed on the CPU -> `OrmqrCandidatesCpu/*.CpuQueueRunsNoNativeFamily` only.
  - B7, the Blocked arm runs the vendor -> broad (46), as expected for a launch swap.
  - B8, the legacy variable ignored -> `LegacyProviderVariablePins` only.
  - B10, vendor workspace sized as blocked -> every test that runs a vendor pin or Auto-vendor (38).
  - B11, key k = A.rows -> `AutoReadsEveryKeyField` and `TraceKeyFollowsSideAndKeepsTrans`.

**(b) Data gate** (`tools/transcribe/ormqr_offgrid_gate.py`, seed 20261005). The test drew 4000
random off-grid points: m in [1, 8192], k in [1, m], q in [1, 16384] and batch in [1, 131072],
all log-uniform, with a random dtype, side and trans. At each point the OLD predicate (the
transcriber's `--points` mode against the 424a45bc headers) was compared with
`sweep_to_table.nearest` on the shipped table plus `can_run` on a GPU, for both devices and with
the vendor present and absent (16000 lookups). Agreement is **100.00% in every
(dtype, device, vendor) cell**, and no region disagrees. This was expected: the old predicates
read no extent, so every row with the same (dtype, trans) is identical and any nearest row
reproduces the decision. No grid refinement was needed.

**(c) Cross-check against the 424a45bc binary.** 22 cells were run one process each, with the
coverage `reached` row as the readback (probe: `ormqr_buffer_size` + `ormqr`). The cells covered
4 dtypes, L/R, N/T/C, m 1-1024, k 1-512, q 1-4096 and batch 7-65536, on and off the grid.
- 18 cells agree (`native:blocked` -> `blocked`). This includes float L T b65536, which throws
  the same work-group-limit error in both builds (see Known gaps).
- The 4 complex-T cells (cfloat L/R, cdouble L/R) differ, as intended. Old: `vendor:auto`, which
  then threw `CUSOLVER error: 3`. New: `invalid_argument` before selection, with no coverage
  row. This is deviation 1.

`scripts/sweep_to_table.py --check`: OK (8 ormqr tables, 540 rows each, re-derived from the CSV).
Comment density: every touched file is under 18%, except `eigen.cc`, which was already waived.

**Not run:** `ctest` beyond the targets above, the ROCm syntax check (no headers on this box), and
`run_local_checks.sh`.
