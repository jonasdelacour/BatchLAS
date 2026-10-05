# getrs on flat kernel selection (phase 5)

Branch `flat-select-p5-getrs`, based on `424a45bc` (the flat-selection stack head). These are
the as-built notes for the integrator. They cover the deviations, the gate results and the doc
edits this branch leaves for others, because the shared docs are not edited here.

## What changed

| | before (`424a45bc`) | after |
|---|---|---|
| routing | `RouteTable<Op::getrs,T>` in `include/batchlas/blas/dispatch/route_getrs.hh` + shape builder `src/backends/getrs_route.hh` | `src/ops/getrs/{choice.hh,getrs.cc}`, `select::choose` on `tuned/getrs.<dtype>.<device>.txt` |
| public `getrs` / `getrs_buffer_size` | `src/dispatch/entry_points/factorization.cc` | `src/ops/getrs/getrs.cc` (added to `batchlas_dispatch_obj` in `src/CMakeLists.txt`) |
| tables | none | `tuned/getrs.{float,double,cfloat,cdouble}.{sm_89,sm_120}.txt`, `source=transcribed:424a45bc` |
| transcriber | none | `tools/transcribe/getrs_transcribe.cc` -> `tuned/transcribed/getrs.csv` |
| data gate | none | `tools/transcribe/getrs_gate.py` |
| tests | `RouteGetrs.*` (route_vocabulary_tests), getrs parts of `getrf_tests` | `tests/getrs_candidates_tests.cc`; getrf_tests ported to `ScopedPin`/can_run; tuned_tables_tests learns getrs |

Deleted: `route_getrs.hh`, `src/backends/getrs_route.hh` and its waiver line. Shared dispatch
infrastructure (`route.hh`, `route_resolve.hh`, `route_env.hh`, coverage, `no_route.hh`, `op.hh`,
`vendor_available.hh`) is left untouched for the integrator.

## Vocabulary

There are three fieldless families, one per kernel driver. The old code chose no knob, so every
family is fieldless: `nb`, the accumulator width, the work-group size and the permutation spelling
are all derived inside the drivers.

| family | driver | can_run (R3) |
|---|---|---|
| `cta` | `sycl_getrs::getrs_fused_dispatch` | native gate, `getrs_fused_available<T>`, `max_wg >= 32`, `nrhs <= kGetrsFusedMaxRhs` (8), `n*nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)` |
| `blocked` | `sycl_getrs::getrs_blocked_dispatch` with two public `trsm` | native gate, `getrs_blocked_available<T>` |
| `vendor` | `backend::getrs_vendor` | `d.has_vendor_solver`, from `factorization_vendor_available<B>` via `device_of<B>(q, ...)` |

The native gate is `B != NETLIB`, `is_gpu`, `has_sg32`, a conforming pair (A square,
`B.rows == n`, equal batch), no heterogeneous A or B, and `n, nrhs, batch >= 1`. This is exactly
what the old `supports()` and `getrs_op_shape`'s `nullopt` cases checked, so a non-conforming
pair still goes to the vendor, or to `throw_no_vendor_route` in a vendor-free build.

- Aliases: `native:cta` -> `cta`, `native:blocked` -> `blocked`. `native` and `vendor` are the
  select class words, and `auto` is Auto. An unknown spelling, or a pin whose `can_run` is false,
  throws (R6).
- Last resort: `blocked`, then `vendor`.
- Keys: `n:log:2 nrhs:log batch:log`, because work is ~ n^2 nrhs batch. `transA` is not a key,
  since no old predicate read it, but it still goes into the coverage row as before.
- The workspace goes through the same `choose` (R5). `getrs_buffer_size` now returns the chosen
  family's need. The old code returned `max(native, vendor)` (see Deviations).

## Grid and transcription

The old predicates read no architecture, so one transcription is written twice: as
`getrs.<dtype>.sm_89.txt` and `getrs.<dtype>.sm_120.txt`, with the same rows. Only the device is
different, in the header. Both capacities are unlimited in the transcriber: the fused tier's
`n*nrhs` ceiling is a device fact and `kGetrsFusedMaxRhs` is a build fact, so both stay in
`can_run`.

The grid puts points on both sides of every threshold the old predicates use:

- n: 31/32 (the CTA order floor), on a log grid 1..1024 (18 points);
- nrhs: 2/3 (clause A), 4/5 (float clause B), 63/64 (float Blocked), 127/128 (double Blocked),
  plus 8 (`kGetrsFusedMaxRhs`), 1..1024 (15 points);
- batch: 127/128 (Blocked's floor), on a log grid 1..32768 (8 points).

That gives 2160 rows per table. Each row's rank order is derived from the real old router
(`resolve_route_uninstrumented` with the winner excluded, then the vendor-free walk):

| ranked | cells (all 4 dtypes x 2 devices) |
|---|---|
| `vendor cta blocked` | 13720 |
| `cta vendor blocked` | 1760 |
| `blocked vendor cta` | 1800 |

The full table is a tensor grid and the lookup distance is a weighted L1 in log2, so the nearest
row splits axis by axis. Every threshold has adjacent integers on the grid, so any integer key
snaps to the correct side.

Regenerate:

    OLD=<checkout of 424a45bc, configured into build/>
    g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/getrs_transcribe.cc -o /tmp/gt
    /tmp/gt sm_89 sm_120 > tuned/transcribed/getrs.csv
    python3 scripts/sweep_to_table.py --transcribe tuned/transcribed/getrs.csv --sha 424a45bc

The re-derivation on this branch was byte-identical, for both the CSV and all 8 tables.
`sweep_to_table.py --transcribe` reads every LFS sweep, so run `git lfs pull` first, or it fails on
the potrf pointer file.

## Gate results (box: 2x RTX PRO 6000 Blackwell, sm_120; GPU 0 only)

### (a) Correctness, failing names vs a `424a45bc` baseline built from the same targets

Targets: getrs_candidates_tests (new), getrf_tests, getrf_tests_native, gesv_tests,
options_api_tests, linalg_layer_tests, select_tests, tuned_tables_tests, route_vocabulary_tests.

| tree | branch failing names | baseline failing names |
|---|---|---|
| build (vendor) | none (9/9 pass) | none (8/8 pass) |
| build-vf | `OptionsApi.Blas3OptionsMatchPositional`; `LuTest/4,6.TinyRoutesInsideItsMeasuredWindowAndNowhereElse` (getrf_tests and getrf_tests_native) | identical |

The vendor-free failures are pre-existing and belong to getrf's tiny-route window and options_api;
neither is getrs. getrs_candidates_tests passes in both trees: 73 run and 72 skipped. Every skip is
a NETLIB-backend instantiation (`GetrsCandidates/0-3`) or the CPU-queue suite on a CUDA backend,
none of them on the CUDA-backend GPU instantiations.

Deliberate breaks of `src/ops/getrs/getrs.cc`, each restored and md5-verified
(`310929e9...`). Every one gives a narrow, named red set:

| break | red |
|---|---|
| B1 drop `nrhs <= kGetrsFusedMaxRhs` from can_run(cta) | CanRunEqualsLaunch, CanRunFalsePinsThrow, PinnedCandidatesStraddleTheirLimits (x4 dtypes) |
| B2 key_of fixes nrhs = 1 | AutoReadsEveryKeyField, AutoReadsTheTranscribedTable, LegacyAliasesAndClassWords, TraceLineShowsTheKey |
| B3 Blocked launches the fused kernel | PinnedRunIsTheDirectKernelBitForBit, CtaLaunchesAtItsResidentCapacity, Saturating..., Workspace..., Straddle..., ScopedPinBeatsTheEnvironment, AutoReadsTheTranscribedTable (float/double) |
| B4 drop the NETLIB gate | NetlibBackendRunsNoNativeFamily |
| B5 drop the conforming-pair gate | CanRunFalsePinsThrow |
| B5b drop `A.batch_size() >= 1` (review fix; the batch-0 cases were added for it) | CanRunFalsePinsThrow (x4 dtypes: "cta batch 0 was accepted", "blocked batch 0 was accepted") |
| B6 SLM capacity x2 | CanRunEqualsLaunch, CtaLaunchesAtItsResidentCapacity |
| B7 table: float sm_120 n=32 nrhs 1/2 rows vendor-first (`BATCHLAS_TUNED_DIR`) | GetrsCandidates/4 (float) AutoReadsTheTranscribedTable + AutoReadsEveryKeyField, GetrsTranscribedTable.RowsHoldTheOldPreference... |

### (b) Data gate: old predicate vs table lookup off the grid

Command: `tools/transcribe/getrs_gate.py 3000 /tmp/gt`. It draws 3000 random log-uniform points,
off the grid in every key, with n, nrhs in [1, 3000] and batch in [1, 1e5]. It adds 300 points in
the threshold neighbourhoods (n 29-35, nrhs 6-10/60-66/120-130, batch 100-200). The old side is the
real `route_getrs.hh` of `424a45bc`, run through the transcriber's `--replay`. The new side is
`sweep_to_table.nearest`, which is the same rule as `Table::nearest`, followed by the first entry
that passes the can_run-equivalent capacities, then the last resort. Each case is run for 2
devices x 4 dtypes x {vendor, vendor-free} x {unlimited, device-like n*nrhs capacity}:

**agreement 100.000% in every one of the 32 configurations (3300 points each, 0 disagreements).**
No disagreeing region, so no grid refinement was needed.

### (c) Coverage cross-check, Auto on the real binaries

The `424a45bc` `factor_bench getrs ... --arms=auto` was run against the branch's, with
`BATCHLAS_COVERAGE_OUT` set, reading the `reached` getrs row. 22 cells were checked: float, double
and cfloat at the 31/32, 2/3, 4/5, 63/64, 127/128 and batch 127/128 straddles, plus n=700 nrhs=2.
The two binaries agree on **22/22**: 7 cta, 3 blocked and 12 vendor, matching cell for cell.
cdouble was left out because the cuBLAS 13.4 complex<double> crash (known-defects #13) hits the
vendor getrf that factor_bench uses to factor.

### Repo checks

`run_local_checks.sh` passes except `check_cmake_syntax`, which parses the generated files in this
worktree's `build-vf/` directory. That is a local artifact, not a source file. The comment density
check passes on every touched C++ file. `check_tuned_tables` notes that there is no
`tools/tune/getrs_spec.cc`. That is expected, because the getrs tables are transcribed.

## Deviations

1. `getrs_buffer_size` returns the chosen family's workspace (R5), where the old code returned
   `max(native, vendor)`. A caller that sizes once and then pins a different family needs to
   re-query, as it does for every migrated op.
2. Coverage rows now carry the choice spelling in `chosen_algo` (§5.6). The native rows read as
   before (`native,cta` / `native,blocked`), but a vendor row now reads `vendor,vendor` where the
   old one read `vendor,auto`.
3. The old "bare origin" behaviour of `BATCHLAS_GETRS_ROUTE=native` (preferred, then the native
   tier walk) is now select's class word `native`. It takes the first runnable native entry of the
   nearest row, then the native last resort. On the transcribed tables that is the same choice the
   old vendor-free walk made.

## Doc edits left for the integrator

- `tuned/README.md` "What is here": add the row
  `| getrs | sm_89, sm_120 | **transcribed** old router, untimed (source=transcribed:424a45bc) | getrs.<dtype>.<device>.txt, from transcribed/getrs.csv |`.
- `docs/design/flat-kernel-selection.md` §12: getrs migrated, with its three fieldless families,
  keys `n:log:2 nrhs:log batch:log`, and one transcription for both devices.
- Stale `route_getrs.hh:<line>` / `getrs_route.hh` citations, now historical: `docs/perf/lu.md`
  (:51, :87, :1472), `docs/perf/small-n-baseline.md:229`, `docs/design/vendor-free-status.md`
  (:58, :111), `docs/design/vendor-independence.md:84`, `docs/design/small-n-factorization-plan.md:1141`
  and the `orgqr_route.hh` waiver justification in `.github/ci/comment_density_waivers.txt:48`. The
  predicates they quote now live as rows of `tuned/getrs.*.txt`, and `tools/transcribe/getrs_transcribe.cc`
  reproduces them from `424a45bc`.
- AGENTS.md §9 lists route tables under `include/batchlas/blas/dispatch/route_*.hh`. getrs no
  longer has one.
