# Level-3 wave: syr2k on flat kernel selection (as built)

Branch `flat-select-l3-syr2k`, based on `origin/main` = `ff340fc6` (no S0 scaffold commit existed when
the branch was cut, so this branch carries syr2k's share of S0 itself; see Deviations). This page holds
the as-built notes, deviations, gate results and the doc changes left to the integrator. AGENTS.md,
`flat-kernel-selection.md`, `tuned/README.md` and `docs/perf/` were left untouched on purpose.

## What changed

| Piece | File |
|---|---|
| Vocabulary: `Triangular` (`detail::syr2k_triangular_tiles<float>`) and `Vendor` (`backend::syr2k_vendor`), both `NoFields`; float lists both, double only the vendor (R7); keys, grid, grid ceilings | `src/ops/syr2k/choice.hh` |
| Public `syr2k`, `key_of`, `can_run`, `choose`, `native_facts`, `launch`, TraceScope; instantiations for CUDA, ROCm and NETLIB, real types | `src/ops/syr2k/syr2k.cc` (moved out of `src/ops/level3/level3.cc`) |
| Transcriber (byte-for-byte copies of the old rule) and its gates | `tools/transcribe/syr2k_transcribe.cc`, `tools/transcribe/syr2k_gate.py` (provenance; delete in I) |
| Transcriber CSV; tables (identical rows, `source=transcribed:ff340fc6`) | `tuned/transcribed/syr2k.csv`; `tuned/syr2k.{float,double}.{sm_89,sm_120}.txt` |
| OpSpec `SYR2K` (appended to `OPS`) | `scripts/sweep_to_table.py` |
| Candidate tests (17 typed cases, a CPU case, a no-GPU table case) | `tests/syr2k_candidates_tests.cc` |
| Old-rule tests: `ForcedCuBLASDxPathMatchesVendor` and the `syr2k_cublasdx_fused.hh` include deleted; `cublasdx`, `cta`, `tiny` added to `RemovedRouteWordsThrow` | `tests/syr2k_tests.cc` |
| Deleted: the old rule (`syr2k_pin`, `problem_supported`, `triangular_supported`, `prefer_triangular_tiles`, `use_cuda_custom`, `cuda_custom`, `cuda_custom_forced`) | `src/backends/syr2k_custom_dispatch.cc` (whole file), its CMake line and its comment-density waiver |
| The cublasdx-forced hook (`cublas.cc:622-635`); `syr2k_vendor` is the library loop only | `src/backends/cublas.cc` |
| `syr2k_custom_dispatch.hh` keeps only `syr2k_vendor_cuda_raw` (the dead `level3_vendor_fallback.cc` still names it) | `src/backends/syr2k_custom_dispatch.hh` |

Shared files, each a small local edit: `src/CMakeLists.txt` (one `target_sources` line after level3's),
`src/backends/CMakeLists.txt` (one line removed from the `*_custom_dispatch.cc` list),
`src/ops/level3/level3.cc` (syr2k body, include and four instantiation lines removed; line 1 comment),
`tests/CMakeLists.txt` (target and `blas` label), `tests/tuned_tables_tests.cc` (registry, real-only),
`.github/ci/comment_density_waivers.txt` (the deleted file's waiver).

## The old rule and the transcription

At `ff340fc6`, float on a CUDA GPU queue under Auto: `problem_supported` (no ConjTrans, C square, equal
batches, consistent extents, n, k > 0) and homogeneous and `batch >= 2` -> triangular, else the vendor
(`syr2k_custom_dispatch.cc:29-106`, `level3.cc:157-175`). Double: the vendor always (`cublas.cc:622-637`
only adds the cublasdx throw). Vendor-free: the same gate, and every vendor decision throws `NoRouteError`.

Keys `n:log:2 k:log batch:log` (n = C's order, k = op(A)'s inner extent; work ~ n^2 k batch). The only
threshold is batch 1|2; the grid is a full product, n {1, 2, 4, ..., 4096} x k {1, 8, 64, 512, 4096} x
batch {1, 2, 3, 128, 1024, 8192, 32768} = 455 rows per (dtype, device). Rows: float `vendor | triangular`
at batch 1 (130 rows) and `triangular | vendor` elsewhere (325); double `vendor` (455). sm_89 and sm_120
are identical (the rule read no architecture). `python3 scripts/sweep_to_table.py --check` passes on every
table.

## can_run (R3)

- `triangular`: `B == CUDA` (design D6, the old reach), float, `d.is_gpu`, `d.max_wg >= 256` (the kernel's
  16x16 work-group), `transA != ConjTrans`, A, B, C homogeneous, n, k, batch >= 1, **batch <= 65535**, and
  **`T(T+1)/2 <= 65535` tiles, T = ceil(n/128)**.
  - Grid z (batch, SYCL dim 0) is confirmed by launch: main's binary at batch 65536 throws
    `Number of work-groups exceed limit for dimension 2 : 65536 > 65535` (coverage cross-check below), and
    the `no_batch_ceiling` break turns `GridCeilingsAreCanRunTerms` red. With the term, Auto goes to the
    vendor there (vendor-free: `NoRouteError`).
  - Grid y (the triangular tile list, SYCL dim 1) is new (not in the design): n = 46208 (361 tiles a side,
    65341 tiles) launches and is correct in `GridCeilingsAreCanRunTerms` (a 8.5 GB C on the 96 GB card;
    skipped below 16 GB), and the `no_tile_ceiling` break shows the n = 46209 launch failing.
  - ConjTrans stays refused although the kernel would read a real ConjTrans as Trans and answer correctly.
    That is the old reach (design §2.2); opening it is a routing change, not a migration.
- `vendor`: `d.has_vendor_blas`.

Not terms: the kernel's aligned/predicated leg (derived inside `syr2k_triangular_tiles`, as before).

## Behaviour changes

Vendor present: `BATCHLAS_SYR2K_ROUTE=cublasdx` throws `invalid_argument` (unknown family; it used to
throw `batchlas::unsupported` after the fused attempt), as does every unknown word; `triangular` on double
throws "not a compiled syr2k double candidate" (it used to be ignored); batch > 65535 goes to the vendor
(it used to throw from the launch); invalid shapes throw `invalid_argument` from `validate_rank_2k` before
`choose()` on every backend (NETLIB used to throw `runtime_error` from the deferred host task); coverage
rows carry the real scalar and backend (the old recorder hard-coded F32/CUDA), and double now records rows.

Vendor-free gains: float at batch 1 runs the tile kernel (it threw `NoRouteError`); 4 failing names
disappear from `syr2k_tests` (below). Double still has no route.

## Gates

**(a) Tests, failing names against main** (`/home/jonaslacour/.claude/jobs/4f777db1/tmp/mainbase`, same
targets: `syr2k_tests syr2k_candidates_tests tuned_tables_tests select_tests options_api_tests
sytrd_blocked_tests settings_tests error_model_tests`, GPU 2/3 of the sm_120 box):

- vendor tree: no failures on either side. `syr2k_tests` has 13 cases (main 14: the deleted cuBLASDx case);
  `syr2k_candidates_tests` 30 pass, the rest skip by type (NETLIB GPU suite, double native cases).
- vendor-free tree: branch failures are a strict subset of main's. Gone: `Syr2kCudaCustomTest.{Adversarial
  ShapesLeaveTheOtherHalfUntouched, AutoRouteLeavesTheOtherHalfUntouched, BetaNeverReadsOutsideTheTriangle,
  TriangularTilesLeaveTheOtherHalfUntouched}` (batch-1 shapes now run natively). Unchanged on both sides:
  `Syr2kTest/{0,1,3}.*` (NETLIB with no netlib, double CUDA: no route), `options_api_tests` 1,
  `sytrd_blocked_tests` 18.

**(b) Off-grid data gate** (`tools/transcribe/syr2k_gate.py --data`, 2500 random log-uniform off-grid points
per dtype, n, k in [1, 4096], batch in [1, 32768], N/T): vendor-present agreement 100.00% (2500/2500) for
float and double on sm_89 and sm_120; vendor-free where the old build served 100.00% (float 2389/2389);
where it threw, the new build runs float `triangular` 111 times and double still throws (2500).
`--fidelity`: the three verbatim blocks are byte-identical to `ff340fc6`. Negative control: dropping the
`batch=1` rows drops float to 95.56% (exit 1); dropping the design's `batch=2` rows alone stays at 100%,
because batch 2 snaps to the batch=3 row, which holds the same ranking.

**(c) Coverage cross-check** (one probe process per cell, `BATCHLAS_COVERAGE_OUT` plus
`BATCHLAS_KERNEL_TRACE`, main's and this branch's libraries, 44 cells: both dtypes, batch 1|2|3 at
off-grid n and k, N/T/C, L/U, batch 40000, 65535 and 65536):

- vendor: float 21/22 identical (`native:triangular` with the `syr2k_cuda_custom.triangular_tiles` kernel,
  or `vendor:vendor`); the 22nd is batch 65536, main `THROW Number of work-groups exceed limit`, branch
  `vendor:vendor`. Double: main records no row and launches no native kernel (the vendor, by
  `cublas.cc`); the branch records `vendor:vendor` in all 22.
- vendor-free: main records a `vendor:vendor` row and then throws `NoRouteError` at batch 1; the branch runs
  `native:triangular` (8 cells). ConjTrans and batch 65536 throw `NoRouteError` on the branch with a `miss`
  row instead of main's false `reached` row (and main's launch failure at 65536). Double throws in both.

**Deliberate breaks** (each one edit to `src/ops/syr2k/syr2k.cc`, rebuilt, both binaries run, restored and
md5-checked against the committed file):

| Break | Red set (`/2` = float CUDA, `/3` = double CUDA, `Cpu/0,1` = NETLIB) |
|---|---|
| `key_k_wrong_extent`: `key_of` reads `A.cols()` for k | `AutoReadsEveryKeyField/2`, `TraceKeyKFollowsTrans/2,3` |
| `no_conjtrans_term` | `CanRunFalsePinsThrow/2`, `EveryCombinationOnEveryCandidate/2` |
| `no_batch_ceiling` | `GridCeilingsAreCanRunTerms/2` (the pin launches and throws `Number of work-groups exceed limit for dimension 2 : 65536 > 65535`) |
| `no_tile_ceiling` | `GridCeilingsAreCanRunTerms/2` (`... dimension 1 : 65703 > 65535` at n = 46209) |
| `no_het_term` | `CanRunFalsePinsThrow/2`, `HeterogeneousBatchHasNoNativeRoute/2` |
| `vendor_uplo_flipped` (the vendor arm passes the other triangle) | 25 names, every case that runs the vendor: the candidates suite's vendor pins and batch-1 Auto cases, `CpuQueueRunsNoNativeFamily` on NETLIB, and `syr2k_tests`' vendor comparisons |
| `triangular_arm_runs_vendor` (swapped launch arm) | `PinnedChoiceRunsItsOwnKernel/2` |
| `coverage_k_is_n` | `CoverageRowCarriesBackendKeyAndNativeFlags/2,3` |
| `no_validation` (no `validate_rank_2k` at entry) | `InvalidShapesThrowBeforeChoosing/2` (double still throws from cuBLAS's own check) |
| table: float sm_120 batch=1 rows reordered `triangular \| vendor` (through `BATCHLAS_TUNED_DIR`, no rebuild) | `AutoReadsTheTranscribedTable/2`, `ClassWordsAndSpellings/2`, `CoverageRowCarriesBackendKeyAndNativeFlags/2`, `Syr2kTranscribedTable`; also `AutoReadsEveryKeyField/2` and `VendorFreeLastResortIsTriangular/2`, only because the override directory also replaces the synthetic tables those two install |

## Deviations from the shared design

1. **No S0 commit**: the branch starts at `ff340fc6`, so it moves syr2k's entry point, adds the CMake line,
   the OpSpec, the registry entry and the test target itself. Merge conflicts with the other three op
   branches are expected only in `level3.cc` (adjacent deletions and line 1), `src/CMakeLists.txt` (adjacent
   additions), `src/backends/CMakeLists.txt`, `tests/CMakeLists.txt`, `tests/tuned_tables_tests.cc` and
   `scripts/sweep_to_table.py` (each op appends its own block).
2. **The old rule is deleted here, not in I** (the task asked for it): `syr2k_custom_dispatch.cc` is gone;
   the header keeps `syr2k_vendor_cuda_raw` until I deletes `level3_vendor_fallback.*`. `syr2k_fused_try`
   (`level3_fused*`) and `syr2k_vendor_fallback` are now unreferenced and left for I.
3. **Two extra `can_run` terms**: the grid-y tile ceiling and `max_wg >= 256` (both launch limits).
4. **`PinnedRunIsTheDirectKernelBitForBit` became `PinnedChoiceRunsItsOwnKernel`**: the vendor pin is
   compared bit for bit with `backend::syr2k_vendor`; the triangular pin is identified by the kernel trace in
   a child process. The tile kernel is header-only, so calling it from the test would compile a second copy
   of the kernel with the same SYCL kernel name into the test binary.
5. **`CanRunEqualsLaunch` is folded into `GridCeilingsAreCanRunTerms`**: the kernel has no refusals of its
   own, so the only launch-versus-`can_run` boundaries are the two grid limits.
6. **Negative control on `batch=1`**, not `batch=2` (see gate b).

## Left for the integrator

- `tuned/README.md`: inventory row (syr2k, float and double, transcribed `ff340fc6`, `syr2k.csv`, both
  devices) and provenance (`git show <I^>:tools/transcribe/syr2k_transcribe.cc`); then delete
  `tools/transcribe/`.
- `EveryOpShipsATableForEveryDtypeOnEveryShippedDevice`: add syr2k with real dtypes only.
- AGENTS.md §9 and `docs/perf/dispatch.md#the-level-3-pin-words`: syr2k is flat; words `auto`, `native`,
  `vendor`, `triangular`; `cublasdx` throws.
- `docs/perf/level3.md` (`syr2k_custom_dispatch.cc` line cites at :41 and :127), `docs/perf/dispatch.md`
  (:53, :73, :155), `known-defects.md:257` (cites `syr2k_custom_dispatch.cc:210`): historical now.
- `level3.cc` header ("The four level-3 custom-route gates"), `vendor.hh:60-63` and `coverage.cc:158`
  comments; `tests/CMakeLists.txt:303` (`syr2k_tests` no longer includes a cuBLASDx header).
- `scripts/rocm_syntax_check.sh` could not run here (no ROCm headers on the sm_120 box); the ROCm
  instantiations moved into `src/ops/syr2k/syr2k.cc`.
