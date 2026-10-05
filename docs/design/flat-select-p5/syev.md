# Phase 5: syev on flat kernel selection (as built)

Branch `flat-select-p5-syev`, based on `424a45bc` (the flat-selection stack head). This file is the
op's §12 entry for [flat-kernel-selection.md](../flat-kernel-selection.md); the integrator folds
it in. It also lists the shared-doc edits this PR leaves to the integrator (last section).

## What was built

- `src/ops/syev/{choice.hh,syev.cc}`: the public `syev` and `syev_buffer_size` (moved out of the
  installed header, where they were inline, and out of `src/dispatch/entry_points/eigen.cc`, which
  only instantiated them). Three hops: `syev()` → `choose()` → `std::visit` launch; sizing calls
  the same `choose()` (R5) and returns the chosen family's own `*_buffer_size`.
- **Families**, one per kernel driver, all `NoFields`, the same for every dtype, in tie-break
  order: `cta` (`syev_cta`), `cta_fused` (`syev_cta_fused`), `jacobi` (`syev_jacobi_cta`),
  `blocked` (`syev_blocked`), `two_stage` (`syev_two_stage`), `vendor` (`backend::syev_vendor`).
  The old router had one `CTA` route and picked the driver inside it
  (`syev_choose_small_kernel<T>`, from the type and `n <= 8`); that pick was a choice the old code
  made, so the three drivers are families now. CTA STEQR parameters (`max_sweeps = 400`,
  Wilkinson), `StedcParams{}`, the wg multipliers and every sub-op choice stay derived.
- **Keys**: `jobz:exact n:log:3 batch:log` (work ~ n³·batch). uplo is not a key (both large-n
  drivers mirror Upper into Lower; the old router never read it). batch is read by no live old
  predicate (`syev_prefer_vendor` was dead for n <= 32 by construction) but is kept so a timed
  table can use it; every transcribed row is batch-invariant.
- **Grid** (`choice.hh`, spelled again in the transcriber): both sides of every old threshold,
  n = 8|9 (small-kernel pick), 24|25 (cdouble CTA vs vendor), 32|33 (CTA cap), 256|257 (cdouble
  vectors), 320|321 (values-only), 448|449 (real vectors), 512|513 (cfloat vectors), 1024|1025
  (float vectors), plus a log grid 1..4096; batch 128..32768. Because each threshold t has both t
  and t+1 on the grid, nearest-point lookup lands every integer n on the same side of every
  threshold as the old predicate (the data gate confirms it).
- **`can_run`** (R3): native families need a non-NETLIB backend, a GPU, a square A and n >= 1;
  the small three add n <= 32 and `has_sg32` (the drivers check `sub_group_sizes` for 32);
  blocked and two_stage add batch >= 1 (their drivers' own check; the small drivers have none);
  vendor is `d.has_vendor_solver`. No heterogeneity term: no driver and no old predicate checks it.
- `last_resort {"blocked", "vendor"}`. **Aliases** (old `BATCHLAS_SYEV_ROUTE` /
  `BATCHLAS_SYEV_PROVIDER` spellings): `native:{cta,blocked,two_stage,two-stage,jacobi,cta_fused}`,
  `two-stage`, `batchlas_{cta,blocked,two_stage}`, `batchlas-{cta,blocked,two-stage}`, `fused`,
  `netlib` and `vendor:auto` (→ `vendor`, concrete). `BATCHLAS_SYEV_PROVIDER` is still read when
  neither a `ScopedPin` nor `BATCHLAS_SYEV_ROUTE` is set; an error from it names the variable.
- Trace and coverage through `TraceScope` with the key (`jobz n batch`) and `native_facts`;
  `syev_supports_{cta,blocked,two_stage}` (the Python binding's introspection) are now exported
  out-of-line wrappers over the same `can_run`.
- Deleted: `RouteTable<Op::syev,T>`, `kSyevOrder`, `SyevShape`, `syev_op_shape`, `syev_route`,
  `syev_dispatch`, `syev_buffer_size_dispatch`, `syev_prefer_vendor*`, `syev_cta_max_n*`,
  `syev_saturated_algorithm_for_n*`, `SyevSmallKernel`/`syev_choose_small_kernel`,
  `syev_cta_steqr_params` from `include/batchlas/blas/functions/syev.hh` (installed header).
  The public function declarations, `sig::`, the vendor declarations and the `_or_throw` helpers
  are unchanged. syev had no `route_syev.hh` or `src/backends/syev_route.hh`; `Op::syev` stays.

## Transcription

`tools/transcribe/syev_transcribe.cc` is built with DPC++'s `clang++` (no `-fsycl`) against the
424a45bc headers and that tree's `libbatchlas_util`/`libbatchlas_core` (for `settings()`); the
command is in its header. It specialises `RouteTable` for a private `Op` that forwards to the real
`RouteTable<Op::syev,T>` but hides already-ranked routes, and calls the real
`resolve_route_uninstrumented` repeatedly (vendor present until the vendor is ranked, then the
vendor-free walk), so the order comes from the deleted predicates. The CTA slot is spelled by the
real `syev_choose_small_kernel<T>`. Shape: backend CUDA, GPU, sub-group 32, Lower; the env knobs
are unset. The old predicates read no architecture, so one transcription is written for both
devices: `tuned/transcribed/syev.sm_{89,120}.csv` (1480 rows each; md5 reproduced by a rebuild)
and `tuned/syev.<dtype>.sm_{89,120}.txt` (370 rows each, `source=transcribed:424a45bc`).
`scripts/sweep_to_table.py --check` re-derives all eight. Row patterns (every batch):

| dtype | jobz=V | jobz=N |
|---|---|---|
| float | n<=8 `jacobi`, 9-32 `cta_fused`, 33-448 `blocked`, 449-1024 `two_stage`, >1024 `vendor` | n<=8 `jacobi`, 9-32 `cta_fused`, 33-320 `blocked`, >320 `two_stage` |
| double | n<=32 `jacobi`, 33-448 `blocked`, >448 `vendor` | n<=32 `jacobi`, 33-320 `blocked`, >320 `two_stage` |
| cfloat | n<=8 `cta_fused`, 9-32 `cta`, 33-512 `blocked`, >512 `vendor` | as V up to 320, >320 `two_stage` |
| cdouble | n<=24 `cta`, 25-32 `vendor` then `cta`, 33-256 `blocked`, >256 `vendor` | n<=32 `cta`, 33-320 `blocked`, >320 `two_stage` |

Each row then lists the rest in the old order (n<=32: `blocked | two_stage | vendor`; n>32 the
vendor, then the vendor-free natives).

## Behaviour changes and deviations

1. **Vendor-free builds no longer throw where the old Auto preferred the vendor.** The old
   `syev_route` called `resolve_route` with the default `vendor_available = true`, so a vendor-free
   build threw `NoRouteError` at cdouble V n=25-32 and every vendor window above. The table walk
   now skips `vendor` and takes the next native (`cta`, `blocked`). 11 vendor-free failing test
   names disappear (below).
2. **Non-square A throws `invalid_argument`** before `choose()`, under Auto and every pin. The old
   router sent it to the vendor, which read `rows()` as the order.
3. **NETLIB pins are evaluated.** The old dispatch skipped resolution on NETLIB (always vendor) and
   ignored any pin; a native pin on NETLIB now throws (R6). `syev_blocked_tests`'
   `TwoStageProvider*Smoke` NETLIB instantiations (which never ran two_stage) now skip.
4. **`cta` names `syev_cta`.** The old `cta`/`native:cta` meant the small tier with its driver
   picked by type and n; `jacobi` used to parse to an unsupported route and silently mean Auto.
5. **Retired knobs:** `BATCHLAS_SYEV_SMALL_KERNEL` (pin `cta|cta_fused|jacobi` instead) and
   `BATCHLAS_SYEV_CTA_MAX_N` (the window is table data) are no longer read; `syev` warns once if
   either is set. Their `Settings` fields and comments in `include/batchlas/settings.hh` are left
   (editing that header rebuilds everything; integrator follow-up). `docs/cpp-api.md` marks them.
6. **ROCm** (no AMD GPU here, untested): the old `preferred()` windows were CUDA-only, so ROCm took
   CTA for n<=32 and Blocked above. A ROCm device (key `rocm`) has no syev table and borrows the sm
   tables with the R8 warning, so it now follows the CUDA windows (two_stage, vendor).
   `scripts/rocm_syntax_check.sh` was not run (no ROCm headers on this box).
7. Batch 0: small families stay eligible (as before); blocked/two_stage require batch >= 1.
8. The coverage `reached` row names the family (`jacobi`, `cta_fused`, `cta`) where it said `cta`,
   and `vendor:vendor` where it said `vendor:auto`.

## Located defects

`OtherTriangleIsNeverRead` (other triangle = large finite poison) is red when its skip list is
removed (break B10 below, all four dtypes), for exactly these pinned drivers; the drivers are
unchanged by this PR and the old Auto reached the same ones:
- `cta` with Upper (n = 5, 17, 32) reads the lower triangle. Old and new Auto send cfloat
  n = 9..32 and cdouble n <= 32 (vectors or not) there.
- `blocked` with Lower at n <= 32 reads the upper triangle (Auto reaches it only through a pin,
  or on a device without a 32-wide sub-group).
- `two_stage` with Lower at n = 40 reads the upper triangle (Auto sends n >= 321 jobz=N and
  float V 449..1024 there).
Integrator: add them to `docs/design/known-defects.md`.

## Tests

`tests/syev_candidates_tests.cc` (label `eig`, typed over the 4 dtypes x NETLIB/CUDA; the NETLIB
half skips, GPU-only): every candidate pinned across n = 1, 4|5, 9, 17, 32|33, 40, both jobz and
both triangles; non-natural ld/stride with large finite poison around A; graded spectra ±10^[-2,2]
with a host Householder Q (no reference through the code under test); pinned-equals-direct-driver
bit for bit; a saturating batch of 1024 (4 representatives, bit-identical); `CanRunEqualsLaunch`;
exact workspace in a poisoned arena plus need-1 refused; unknown and can_run-false pins throw;
the binding's `syev_supports_*` refuse a rectangle and straddle 32|33; legacy aliases and the
class words through `ScopedPin`, `BATCHLAS_SYEV_ROUTE` and `BATCHLAS_SYEV_PROVIDER`; pin
precedence; `AutoReadsEveryKeyField` (synthetic table); the trace key; Auto against the shipped
table at 8|9, 24|25, 32|33; the vendor-free last resort; coverage columns; a CPU queue runs no
native family; `SyevTranscribedTable.RowsHoldTheOldPreference` (C++ `Table::nearest` on both
devices' tables, 20 threshold rows x 3 batches). `tuned_tables_tests` gains
`SyevTablesDeclareChoiceKeyNames` and `SyevTablesHoldExactlyTheChoiceGridOnBothDevices`. No old
predicate test existed for syev (`route_vocabulary_tests` only parses the generic vocabulary).

Deliberate breaks (each restored from a saved copy, md5-verified; red sets per CUDA dtype /4-/7):

| # | break | red |
|---|---|---|
| B1 | small cap 33 | `CanRunEqualsLaunch`, `CanRunFalsePinsThrow`, `PinnedCandidatesStraddleTheirLimits` |
| B2 | jobz key fixed to V | `AutoReadsEveryKeyField`, `TraceLineShowsTheKey` |
| B3 | batch key fixed to 128 | `AutoReadsEveryKeyField`, `TraceLineShowsTheKey` |
| B4 | `cta_fused` launches `syev_cta` | 9 tests/dtype (wrong kernel and workspace: bit-for-bit, workspace, every Auto test that lands on cta_fused) |
| B5 | blocked workspace halved | 10 tests/dtype (every test that runs blocked) |
| B6 | alias `fused` removed | `LegacyAliasesAndClassWords` |
| B7 | `BATCHLAS_SYEV_PROVIDER` ignored | `LegacyAliasesAndClassWords`, `PinPrecedence` (not double: its Auto is jacobi too) |
| B8 | cdouble sm_120 row n=25 V reordered | `AutoReadsTheTranscribedTable` (cdouble), `RowsHoldTheOldPreference` |
| B9 | square term dropped from blocked/two_stage `can_run` | `CanRunFalsePinsThrow` (green before the `syev_supports_*` checks were added: the public entry validates first) |
| B10 | `OtherTriangleIsNeverRead` skip list removed | `OtherTriangleIsNeverRead` (the located defects) |
| BX1 | `batch >= 1` dropped from `large` | `CanRunEqualsLaunch` (batch=0 cells; `sytrd_blocked: invalid batch size` from the sizing call) |
| BX2 | `n >= 1` dropped from `native` | `CanRunEqualsLaunch` (n=0 cells) |
| BX3 | `d.is_gpu` dropped from `native` | `SyevCandidatesCpu.CpuQueueRunsNoNativeFamily` (the binding's `syev_supports_*` on a CPU queue: `Backend::AUTO` skips the NETLIB term) |
| — | Jacobi workspace halved | nothing: `syev_jacobi_cta_buffer_size` is 0 |

BX1-BX3 came from review: `CanRunEqualsLaunch` now also runs n=0 (batch 2), batch=0 (n=8 and
n=40) and Upper as well as Lower; the vendor accepts every degenerate cell, as `can_run` says.
Known gaps: the NETLIB term of `can_run` cannot go red here (no test has a GPU queue on the NETLIB
backend); `has_sg32` cannot go red on this hardware.

## Gate results (threadripper02, GPU 0, sm_120)

**(a) Correctness.** Suites: `syev_candidates_tests syev_tests syev_blocked_tests
syev_two_stage_tests syev_cta_tests syev_jacobi_cta_tests syev_cta_fused_tests syevx_tests
syevx_range_tests iluk_tests lanczos_tests options_api_tests linalg_layer_tests
route_vocabulary_tests select_tests tuned_tables_tests ortho_tests`, against the same targets
built from 424a45bc (`git archive`), failing gtest names compared:
- vendor build: identical failing names (6): `LanczosTestBase.{LanczosTest,ToeplitzEigenpairs}`,
  `SyevBlockedTest/{0,1,2}.EigenvaluesOnlyLowerMatchesNetlib`,
  `SyevxInfoTest.InfoIsZeroWhenEveryItemConverges`; `ortho_tests` segfaults at
  `OrthoMatrixTest/7.OrthogonalizeMatrix` in both (known-defects #13) and `syev_blocked_tests`
  segfaults in a NETLIB instantiation in both (nondeterministic: base and branch each segfaulted
  in some runs and not others; it follows a "no image for the selected device" exception). The
  CUDA instantiations of `syev_blocked_tests`, run alone, pass 32/32 on both. `syev_candidates_tests`
  passes (60 GPU cases + CPU + table cases).
- vendor-free: base 91 failing names, branch 80, a strict subset. Gone: the 8 NETLIB
  `TwoStageProvider*Smoke` (now skip), `SyevBlockedTest/7.{AutoEigenvectorsAtRetunedPanelWidth,
  AutoEigenvectorsSmallNKernelBoundaries}` (cdouble V n=28 used to pick the vendor and throw; it
  runs `cta` now) and `SyevCtaTest/3.EigenvectorsN32RandomLowerResidualAndOrtho` (cdouble). The
  rest are vendor references and `syr2k` inside blocked without cuBLAS, the same on both.
- `scripts/facade_symbol_check.sh syev`: OK. `run_local_checks.sh`: only `check_cmake_syntax`
  fails, on the untracked `build-vf/` tree in the worktree.

**(b) Data gate** (`tools/transcribe/syev_gate.py`): 3028 off-grid points per dtype (2500 random,
n log-uniform in [1, 6000], batch in [1, 65536], plus every off-grid n <= 64 and n = t-2, t-1,
t+2, t+3 around each large-n edge at 4 batches, both jobz), the old ranking from the transcriber's
`--points` mode against `sweep_to_table.nearest()` on the shipped tables with syev's capacities:
**100.00% on all 8 (device, dtype) pairs** for the first runnable choice with the vendor, the
first runnable choice vendor-free, and the whole runnable ranking (24224 lookups). Sensitivity:
with the n = 33 and 449 rows dropped (`DROP_N=33,449`) agreement falls to 98.08-99.44%, so the
gate can fail.

**(c) Cross-check against the 424a45bc binary**: 25 cells (4 dtypes, both jobz, every threshold
pair, off-grid n = 100, 600, 700, 1500 and batch 1-5000), one process per cell, the coverage
`reached` row of an Auto `syev` built against each tree: **25/25 agree** (old `cta` compared with
the transcriber's small-kernel pick; old `vendor:auto` = new `vendor:vendor`).

**Re-run after the review fixes** (16 suites, `ortho_tests` not included). Vendor build: 7 failing
names, the 6 above plus `SyevBlockedTest/3.EigenvaluesOnlyLowerMatchesNetlib`. That name is the
NETLIB "no image" exception. Earlier runs never reached it because the nondeterministic segfault
came first, and run alone it fails identically on the 424a45bc binary. CUDA instantiations of
`syev_blocked_tests` pass 32/32, and `syev_candidates_tests` passes 65. Vendor-free: 72 failing
names, which are the earlier 80 minus the 8 `ortho_tests` names, so nothing new. The data gate
again gives 100.00% on all 8 pairs (98.08% worst with `DROP_N=33,449`).

## Shared-doc edits left to the integrator

- `flat-kernel-selection.md` §12: a syev entry (this file).
- `docs/design/known-defects.md`: the three located defects above.
- `AGENTS.md` (§9 routing line: "syev's table is in functions/syev.hh"), `docs/perf/dispatch.md:61`
  `docs/design/vendor-independence.md:124` and `docs/design/vendor-free-status.md:114`
  (`syev.hh:357-385`, the `syev` row's preferred-window citation) cite the deleted header table.
  The in-tree code comments that cited it (`sytrd_blocked.cc`, `sytrd_sb2st_hh.cc`,
  `syev_blocked.cc`, `entry_points/eigen.cc`, `level3_coverage.hh`, `syev_blocked_tests.cc`) are
  fixed in this PR, and `benchmarks/eigensolver_accuracy.cc` labels runs by the new family words
  (`BATCHLAS_SYEV_ROUTE` first, then `BATCHLAS_SYEV_PROVIDER`; `cta_fused`, `jacobi` added).
  The old root-level notes (`SYEV_PERF_*.md`, `SYEV_RETUNE_*.md`, `GESVD_IMPL_SPEC.md`) cite
  pre-move `include/blas/functions/syev.hh` lines and were already stale before this PR.
- `tuned/README.md`: syev's tables are transcribed (source=transcribed:424a45bc) for sm_89 and sm_120.
- `include/batchlas/settings.hh`: mark `syev_small_kernel` and `syev_cta_max_n` retired.
