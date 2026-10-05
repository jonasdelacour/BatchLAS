# Phase 5: spmm on flat kernel selection (as built)

Branch `flat-select-p5-spmm`, based on `424a45bc` (the flat-selection stack head: phases 1-2,
P3.0-P3.3). Written for the integrator. Nothing in `docs/design/flat-kernel-selection.md`,
`AGENTS.md` or `tuned/README.md` was edited; the changes they need are listed at the end.

## What moved

| Old | New |
|---|---|
| `include/batchlas/blas/dispatch/route_spmm.hh` (installed header, `RouteTable<Op::spmm,T>`, `SpmmShape`) | deleted |
| `src/backends/spmm_route.hh` (`spmm_op_shape`, `spmm_route`) | deleted; its shape checks are `one_spmm()` in `spmm.cc` |
| `src/dispatch/entry_points/sparse.cc` (public `spmm`, `spmm_buffer_size`) | deleted; the entry points live in `src/ops/spmm/spmm.cc` |
| 25 `RouteSpmm.*` tests in `tests/route_vocabulary_tests.cc` | deleted; behaviour-describing ones ported to `tests/spmm_candidates_tests.cc` |
| — | `src/ops/spmm/{choice.hh,spmm.cc}`, `tools/transcribe/spmm_transcribe.cc`, `tools/transcribe/spmm_offgrid_gate.py`, `tuned/spmm.<dtype>.{sm_89,sm_120,cpu}.txt`, `tuned/transcribed/spmm.{sm_89,sm_120,cpu}.csv` |

`Op::spmm`, `sparse_vendor_available`, `kSparseLibrary` and the `coverage.cc` linked-table row
stay (shared infrastructure, deleted by the integrator with the rest).

## Vocabulary (`choice.hh`)

- Families: `Direct` (spelling `direct`, `sycl_spmm::spmm_native_csr`) and `Vendor` (`vendor`,
  `backend::spmm_vendor`: cuSPARSE, rocSPARSE, netlib). Both `NoFields`, same list for every
  dtype. The old table had exactly these two routes (`{Native, Direct}`, `{Vendor, Auto}`).
- Derived inside Direct, never chosen: the body (gather for `transA == N`, scale + atomic scatter
  otherwise), the gather's column block (2 at `nrhs <= 2`, else `kNCmax`), and the complex pair
  load (taken when A's values, B and C are all `sizeof(D)`-aligned). The old router chose none of
  these, so none is a field (the "cfloat pair-load history" is an alignment leg inside the driver,
  not a routing decision; the cfloat routing term is the transB exclusion below).
- Aliases: `native:direct` -> `direct`. `direct`, `vendor`, `native`, `auto` parse as before. The
  old spellings that silently meant Auto (`cta`, `blocked`, a typo) now throw (R6).
- `last_resort {"vendor", "direct"}`: the old resolver's answer when nothing was supported was
  the vendor; reached only when no table row has a runnable entry.
- Keys `transA:exact transB:exact m:log nrhs:log batch:log`. ConjTrans folds to T on both
  operands (the bodies differ from Trans only by a conjugation; the old predicates never
  distinguished T from C). `m` = `A.rows()` as stored (the old `SpmmShape::m`), `nrhs` =
  `C.cols()`. Weights 1: work ~ nnz·nrhs·batch with nnz ~ m at a fixed row density.
  **No nnz/density key**, as in the old `SpmmShape`: the honest per-item nnz is in device memory,
  and `spmm_buffer_size` runs the same `choose()` (reading it there segfaulted before). `k` is not
  a key either: no old predicate read it; the coverage row still records it.

## What the old predicates read, and the grid

`supports(Direct)`: format == CSR, no heterogeneous dense operand, m, k, n >= 0, batch >= 1, and
the body's compiled flag (`gather_available` for N, `scatter_available` otherwise; both true in
every build). `supports(Vendor)`: always. `preferred(Direct)`: CSR and `transA == N` and not
(complex<float> and `transB != N`). Nothing else: no extent, batch, nnz, is_gpu or architecture
term. The shape builder sent any inconsistent view to the vendor.

So the only thresholds are categorical, and both sides of each are exact keys. The size axes are a
coarse log grid kept for a later tuner (m {1,16,256,4096,65536} x nrhs {1,2,4,16,64} x batch
{1,8,128,1024,16384}; batch reaches 1 because lanczos/LOBPCG call at small batch). 500 rows per
table. Rows: `direct | vendor` inside the window, `vendor | direct` outside (transA=T, and
cfloat transA=N transB=T), for all three devices.

## can_run (R3)

- `Direct`: CSR only (`if constexpr`), the compiled-body flag for this transA, `one_spmm()` (the
  old shape builder's checks: extents agree, batch sizes agree, `ldb, ldc > 0`, CSR
  `offset_stride >= m + 1`, `matrix_stride >= 0`), no heterogeneous B or C, `nrhs >= 0`,
  `batch >= 1`. No GPU gate (the NETLIB native_cpu queue relies on it). The driver itself has no
  checks (a refused shape would read out of bounds), so `CanRunEqualsTheOracle` tests refusal
  through the pin, acceptance through a correct answer.
- `Vendor`: `d.has_vendor_blas`, which **carries the sparse library** for spmm:
  `ops::spmm::device<B>()` calls `select::device_of<B>(q, solver_vendor_available<B>,
  sparse_vendor_available<B>)`. `select::Device` has no sparse flag and `select.hh` is shared, so
  no field was added; the integrator may want a `has_vendor_sparse`.
- **Two new CUDA vendor terms (R3, not in the old `supports()`)**, both measured on GPU 0 of
  this box (cuSPARSE from HPC SDK 26.5 / CUDA 13.2) by calling `backend::spmm_vendor` directly,
  so they are vendor behaviour and not the facade:
  1. complex (cfloat, cdouble) with `transB == ConjTrans` and a single-row B (nrhs 1): cuSPARSE
     returns an error status the vendor arm never checks, C is left unwritten (a silent wrong
     answer: 5 failures per probe, any transA). The old Auto *did* route here (complex transA=T/C
     with transB=C, and cfloat transA=N transB=C), so these shapes now run `direct` instead of
     producing a wrong answer. Together with the NETLIB term below, the only Auto changes.
  2. complex<double> `N/N` with one column: segfault on the host inside cuSPARSE (cf.
     known-defects #13, the cuBLASLt unit-dimension crash). Never on the old Auto path (N/N is
     `direct` first), so only a `vendor` pin changes: it now falls back to Auto (the class word's
     §5.3 behaviour) instead of crashing.
  Both terms carry `B == Backend::CUDA`; known-defects.md should gain these two (#13 extension).
- **A NETLIB vendor term (R3, review fix)**: `!(B == Backend::NETLIB && !nn && batch > 0)`.
  netlib's host loop (`netlib_lapack.cc`) throws `unsupported` per item on any transpose, so
  `can_run` now equals `launch` for every family. On a NETLIB queue Auto for transA=T (any
  dtype) and cfloat transA=N transB=T/C used to pick `vendor` (first in the cpu row) and throw;
  it now takes `direct`, the row's second entry, which serves all nine spellings
  (`DirectHasNoGpuGate`). A `vendor` pin there falls back to Auto (the class word, §5.3)
  instead of throwing. Side effect: the `spmm_tests` "backend REFUSED ... a MISSING ROUTE"
  skips on NETLIB transposes now run and pass (same names, skip -> pass).
- **Known R3 gap, inherited unchanged**: cuSPARSE silently mis-handles operands off their natural
  alignment (`spmm_tests` skips misaligned cases unless pinned native). Alignment is a property
  of the pointers, not of the key, and modelling it would move routing; left for the
  vendor-capability phase (cf. trsm's known-defects #12 deviation). known-defects.md should
  record it as an explicit R3 waiver.

## choose / launch / workspace

R1-R5 as in trsm: `choose()` -> `select::choose("spmm", ...)`, a `runtime_error` with no runnable
candidate in a vendor-free build becomes `throw_no_vendor_route(Op::spmm, B, kSparseLibrary<B>)`
(coverage `miss` row kept). `launch` is one `std::visit`. `spmm_buffer_size` calls the same
`choose()` and returns 0 for `direct`, `spmm_vendor_buffer_size` for `vendor` (R5); the old code
already skipped the vendor sizer on a native route. `DirectSizesNoWorkspace` (review fix) pins
both: exactly 0 under a Direct pin on all three transA spellings, and the vendor sizer's own
answer under a Vendor pin. No validate_params, as before (a bad shape
goes to the vendor). `TraceScope` with the key fields; the coverage row keeps the old field
mapping (m = A.rows, n = nrhs, k = A.cols, transA/transB unfolded, is_gpu, heterogeneous) and now
carries the real backend; `native_facts` gives `native_route_existed/supported`.

## Deviations from the brief

1. **A third transcribed device, `cpu`.** The brief named sm_89 and sm_120. A CPU device never
   borrows a GPU table (§12), so without `tuned/spmm.<dtype>.cpu.txt` a NETLIB queue would fall
   to the shape-independent last resort and could not reproduce the old transA-dependent choice
   (Direct for NoTrans, netlib otherwise). The cpu tables are the same transcription (the old
   predicates read no device fact, not even is_gpu).
2. `Vendor` uses `has_vendor_blas` carrying the sparse flag (above), not `has_vendor_solver`/blas
   as such.
3. The cuSPARSE-alignment R3 gap above is kept. Three vendor terms are added: the two CUDA terms
   (one moves Auto off a wrong answer) and the NETLIB transpose term (moves NETLIB Auto off a
   throw onto Direct), see can_run.
6. cuSPARSE's `CUSPARSE_SPMM_ALG_DEFAULT` is not bit-reproducible run to run on this box, so
   `spmm_candidates_tests` identifies a pinned vendor by its trace line plus "not the gather's
   exact bits", and its saturating-batch bit-identity check covers Direct only.
4. No `tools/tune/spmm_spec.cc` (no tuner spec in scope), so the staleness checker skips spmm
   tables (`kernels=unknown`, no warning).
5. Old pins that silently fell through (`cta`, `blocked`, typos) now throw; a `direct` pin on an
   inconsistent shape throws instead of silently taking the vendor.

## Gates

All on GPU 0 (`CUDA_VISIBLE_DEVICES=0`), threadripper02 (sm_120), 2026-10-05.

- (a) correctness: `spmm_tests spmm_tests_native spmm_candidates_tests select_tests
  tuned_tables_tests route_vocabulary_tests lanczos_tests syevx_tests syevx_range_tests
  ritz_values_tests`, built in `build` and `build-vf`, against the same targets built from
  424a45bc (`build-base`, `build-base-vf`). Failing gtest names are identical to the baseline in
  both trees: in `build`, `lanczos_tests::LanczosTestBase.{LanczosTest,ToeplitzEigenpairs}` and
  `syevx_tests::SyevxInfoTest.InfoIsZeroWhenEveryItemConverges` (pre-existing on this box); in
  `build-vf` the same two lanczos cases plus 20 `syevx_tests` cases (Jacobi/LOBPCG/range, all
  failing identically on 424a45bc vendor-free). No new failure, none fixed; `spmm_candidates_tests` passes (5 repeats clean).
  Deliberate breaks (restored from saved copies, md5 verified):
  - `key_of` ignores transB -> red exactly `AutoReadsEveryKeyField` + `TraceKeyIsKeyOf` (all 8
    configs) and `AutoReadsTheTranscribedTables` (cfloat, both backends);
  - conj single-row vendor term disabled -> red `VendorRefusesAConjugatedSingleRowB`,
    `PinnedCandidatesRunEveryTransposeCombination` (CUDA cfloat, cdouble) and
    `AutoReadsTheTranscribedTables` (CUDA cdouble);
  - Vendor launch arm runs Direct -> red `PinnedRunIsTheDirectKernelBitForBit` (all 8),
    `PinnedCandidatesRunEveryTransposeCombination` + `LegacyAliasesAndClassWords` (NETLIB: netlib
    no longer throws on a transpose). Before the "not the gather's bits" check this break was
    green on CUDA.
- (b) data gate: `spmm_transcribe --random 2500 20261005` (10 000 off-grid points: log-uniform
  m, k in [1, 3e5], nrhs [1, 256], batch [1, 65536], all nine N/T/C spellings) replayed by
  `tools/transcribe/spmm_offgrid_gate.py` against all 12 tables, vendor-present and vendor-free:
  **100.000% agreement in every (dtype, device, build) cell**, no disagreeing region. It cannot be
  otherwise: rows are constant across the size axes and the categorical keys are exact. The
  gate is armed: swapping the old ranking on 50 points drops the float cells to 98% and lists them.
  A third column, `cuda-can_run`, applies the two CUDA vendor terms (sm_89/sm_120 only): cfloat
  96.2%, cdouble 97.24%, every disagreement in `{N,T,C}C nrhs=1` (cfloat) / `{T,C}C nrhs=1`
  (cdouble), i.e. exactly the conj single-row region where the old router picked a vendor that
  returns a wrong answer. No grid point can or should change that; the gate exits on the
  table-only cells.
- (c) cross-check vs the 424a45bc binary: `spmm_tests` under Auto with `BATCHLAS_COVERAGE_OUT`
  in both builds, `reached` rows keyed (scalar, m, n, k, batch, transA, transB): 80 cells common,
  **80/80 choose the same family** (direct for N/N, N/T, N/C except cfloat N/T, N/C; vendor for
  every transposed A), no cell only in one build.

### Review round 1 (NETLIB R3 term, R5 sizing test)

Same box, GPU 0, own `build` / `build-vf` trees, same targets as (a).

- (a) Failing gtest names identical to the 424a45bc baseline in both trees (3 in `build`, 22 in
  `build-vf`, the same lists as above); `spmm_tests`, `spmm_tests_native`,
  `spmm_candidates_tests` pass in both. `spmm_candidates_tests` passed 10 then 5 repeats per tree.
  `spmm_tests` on this tree: 396 passed, 12 skipped (all `Misaligned*` on CUDA), zero
  NETLIB "REFUSED ... MISSING ROUTE" skips left.
- Deliberate breaks (restored from a saved copy, md5 `e9d844dd...` verified):
  - NETLIB term removed -> red exactly `PinnedCandidatesRunEveryTransposeCombination`,
    `VendorRefusesAConjugatedSingleRowB`, `LegacyAliasesAndClassWords`,
    `AutoReadsTheTranscribedTables`, `AutoReadsEveryKeyField` on the 4 NETLIB configs; CUDA green.
  - Direct arm of `workspace()` returns `spmm_vendor_buffer_size` -> red exactly
    `DirectSizesNoWorkspace` on the 4 CUDA configs (cuSPARSE asks 512 bytes); NETLIB green (its
    vendor sizer is 0, so the guard is vacuous there).
  - Vendor launch arm runs Direct -> red exactly `PinnedRunIsTheDirectKernelBitForBit` in all 8.
- Flake fixed: `PinnedRunIsTheDirectKernelBitForBit` failed once in 5 repeats (CUDA float, vendor
  NN). Two agreeing direct cuSPARSE runs were taken to mean "reproducible", and then the pinned
  run was required to match them bit for bit. Now a pinned vendor run must match one of the two
  direct runs or, failing that, must not carry the gather's exact bits.
- `AutoReadsEveryKeyField` on NETLIB: the transposed probe rows expect `direct` (the vendor can no
  longer run there), so the transA/transB rows are read on CUDA/ROCm only.
- (b) data gate rerun on the same 10 000 points: table-only cells 100.000% everywhere (exit 0).
  The new `cpu-can_run` column (netlib transpose term) reads float 32.44%, double 34.6%, cfloat
  10.64%, cdouble 34.16%. Every disagreement is a transA=T shape, or a cfloat transA=N shape
  with transB=T/C, which is exactly where old Auto picked netlib and threw.

## Doc changes the integrator should make

- `docs/design/known-defects.md`: the two CUDA cuSPARSE vendor terms (#13 extension) and the
  cuSPARSE misalignment R3 waiver (above).

- `docs/design/flat-kernel-selection.md` §12: a "Phase 5, spmm" paragraph (this file).
- `docs/perf/spmm.md` cites `route_spmm.hh:...` line numbers for the window and `supports()`
  (sections "Route arms", "The preferred window, as implemented", "supports(), and what is
  deliberately not in it", "Open debts"); they now live in `src/ops/spmm/spmm.cc` `can_run` and
  the transcribed tables. Anchors are unchanged.
- `docs/design/vendor-free-status.md:113` and `docs/design/vendor-independence.md` (lines 111,
  115, 132, 138, 146, 218) cite `route_spmm.hh`, `spmm_route.hh`, `SpmmShape` and
  `entry_points/sparse.cc`.
- `tuned/README.md`: list the spmm tables (three devices, transcribed).
- `AGENTS.md` §9: `BATCHLAS_SPMM_ROUTE` values are now `auto|native|vendor|direct|native:direct`;
  anything else throws.
