# P5: getri on flat kernel selection (as built)

Branch `flat-select-p5-getri`, based on `424a45bc` (the flat-selection stack head). This page holds the
as-built notes, deviations, gate results and the doc changes the integrator still has to make. The
other docs (AGENTS.md, flat-kernel-selection.md, tuned/README.md) were left untouched on purpose.

## What changed

| Piece | File |
|---|---|
| Vocabulary: `Blocked` (getri_blocked_dispatch: P written into C, then two public trsm calls) and `Vendor` (cublas<t>getriBatched / rocSOLVER / LAPACKE). Both are fieldless, because the driver derives its work-group size. | `src/ops/getri/choice.hh` |
| Public `getri` / `getri_buffer_size`, `key_of`, `can_run`, `choose`, `launch`, `workspace`, `native_facts`, TraceScope | `src/ops/getri/getri.cc` (moved out of `src/dispatch/entry_points/factorization.cc`) |
| Transcriber (old `RouteTable<Op::getri,T>` + `resolve_route`, evaluated at every grid cell, plus `--offgrid` probes) | `tools/transcribe/getri_transcribe.cc` |
| Transcriber CSVs | `tuned/transcribed/getri.sm_89.csv`, `tuned/transcribed/getri.sm_120.csv` |
| Tables (identical rows, device in the header, `source=transcribed:424a45bc`) | `tuned/getri.{float,double,cfloat,cdouble}.{sm_89,sm_120}.txt` |
| OpSpec `GETRI` | `scripts/sweep_to_table.py` |
| Candidate tests (design §8, plus the ported RouteGetri.* and getrf_tests getri readbacks) | `tests/getri_candidates_tests.cc` |
| Deleted | `include/batchlas/blas/dispatch/route_getri.hh`, `src/backends/getri_route.hh`, and the waiver line for route_getri.hh |

Shared files touched, each with a small, localized edit: `src/CMakeLists.txt` (one `target_sources` line),
`tests/CMakeLists.txt` (target and blas label), `tests/tuned_tables_tests.cc` (candidates,
key-names test, grid test), `tests/route_vocabulary_tests.cc` (RouteGetri.* removed, getri halves of
RouteLuPivotFormat and NativeTierPreferred removed), `tests/getrf_tests.cc` (getri route readbacks
removed, a `ScopedPin` replaces the env pin), and `src/dispatch/entry_points/factorization.cc` (getri removed).
`.github/ci/comment_density_waivers.txt` loses the waiver line for the deleted header.

## Keys and grid

`# keys: n:log:3 batch:log`. The old predicates read only the order: `preferred()` is float `n >= 128`
and cfloat `n >= 256`, and never true for double or cdouble. `supports()` reads structure only (square, GPU,
sg32, homogeneous, n >= 1, batch >= 1, not NETLIB, driver compiled). The grid is
`n ∈ {1,2,3,4,6,8,12,16,24,32,48,64,96,127,128,192,255,256,384,512,768,1024,2048,4096}` ×
`batch ∈ {128,512,2048,8192,32768}`, so both thresholds have an integer grid point on each side
(127/128 and 255/256). Nearest-point lookup on an integer n therefore reproduces each threshold exactly.

**Deviation: `batch` is a key even though the old router never read it** (beyond the `batch >= 1`
correctness term, which stays in `can_run`). This follows the convention of the other migrated ops
(potrf, posv and trsm all carry `batch:log`), so that a phase-4 retune can split by batch without changing
the table format. Every batch row holds the same ranking for a given n, and the grid is a full product, so the
batch key cannot change any decision (the data gate below confirms this). If the integrator prefers
"keys = exactly what the old predicates read", dropping `batch` needs edits in four places: choice.hh
`key_names`/`grid_batch`, `key_of`, the transcriber, and the OpSpec. After that, re-transcribe.

## can_run (R3)

- `Blocked`: `d.is_gpu && d.has_sg32 && B != NETLIB && getri_blocked_available<T>() && square && n >= 1 && batch >= 1 && !heterogeneous`.
  These are getri_blocked_dispatch's own checks on A, plus the pivot-format clause (a NETLIB backend on a GPU
  queue writes genuine int64, while Blocked reads packed int32). C's extents and A/C aliasing remain argument
  errors that the driver itself reports. The choice is a function of A alone, because `getri_buffer_size` has no C.
- `Vendor`: `d.has_vendor_solver && d.has_vendor_blas`. On CUDA this equals `factorization_vendor_available`
  (cuBLAS && cuSOLVER), and on NETLIB both reduce to kHasNetlib. On ROCm it additionally requires rocBLAS, which
  a rocSOLVER build always has. There is no dedicated "factorization" flag on `select::Device`. If the
  integrator adds one for getrf, getrs, geqrf and the rest, switch this to that flag.
- No capacity terms exist, so none are capped in the transcription.

Sizing stays pure: `getri_buffer_size` → `choose` → `workspace` reads metadata only. A `SizingReadsMetadataOnly` test
passes a null data pointer under Auto and under every pin. Workspace is the chosen family's own: 0 for Blocked,
`getri_vendor_buffer_size` for Vendor. The old code returned `max(native, vendor)`.

Aliases: `native:blocked` → `blocked`. The class words `native`, `vendor` and `auto` come from select. Bad pins
throw (R6). The `getrf_tests_native` ctest variant still sets `BATCHLAS_GETRI_ROUTE=native`, which is valid
as a class word.

## Gate results

### (a) Correctness: GPU 0 (RTX PRO 6000 Blackwell, sm_120)

Targets: getri_candidates_tests, getrf_tests, inverse_tests, options_api_tests, linalg_layer_tests,
syevx_tests (inv callers), select_tests, tuned_tables_tests, route_vocabulary_tests, trsm_candidates_tests.
Baseline: the same targets built from `git archive 424a45bc`.

| tree | branch failing names | 424a45bc failing names |
|---|---|---|
| build | `SyevxInfoTest.InfoIsZeroWhenEveryItemConverges` | same |
| build-vf | getrf_tests `LuTest/{4,6}.TinyRoutesInsideItsMeasuredWindowAndNowhereElse`; `OptionsApi.Blas3OptionsMatchPositional`; 20 syevx names | identical set |

The failing-name diff is empty in both trees. Test counts: route_vocabulary_tests 104 → 98 (the six RouteGetri.*
cases moved), tuned_tables_tests 11 → 13, getri_candidates_tests is new (build: 81 pass, 80 typed skips for
NETLIB/CPU instantiations; build-vf: 73 pass).

One vendor-free defect was found in the salvaged test and fixed. `NetlibBackendOnAGpuQueueRefusesBlocked` called
the NETLIB getri with a `native` pin and expected only the warning. In build-vf there is no netlib, so
the warned fall-back to Auto has nothing left and throws `NoRouteError`. This is correct behaviour. The test
left stderr captured and aborted the binary. It now expects `NoRouteError` exactly when
`!factorization_vendor_available<NETLIB>`.

Deliberate breaks: each one was applied, getri_candidates_tests was rebuilt and run in build on GPU 0, and the file was
then restored from a saved copy and md5-verified (getri.cc `3bd61b11…`, getri.float.sm_120.txt `7c68f2bc…`).
Every red set is narrow and named. The /4-/7 suffixes are the CUDA float, double, cfloat and cdouble instantiations.

| break | red set |
|---|---|
| drop `B != NETLIB` from can_run(Blocked) | `NetlibBackendOnAGpuQueueRefusesBlocked` ×4 |
| drop squareness | `CanRunEqualsLaunch`, `CanRunFalsePinsThrow` ×4 |
| drop `!is_heterogeneous` | `HeterogeneousBatchHasNoNativeRoute` ×4 |
| key batch fixed at 128 | `AutoReadsEveryKeyField`, `TraceLineShowsTheKey` ×4 |
| key n fixed at 64 | `AutoReadsEveryKeyField`, `TraceLineShowsTheKey` ×4, `AutoReadsTheTranscribedTable` /4 and /6 (double and cdouble are vendor at every n) |
| Blocked launches the vendor | `PinnedRunIsTheDirectKernelBitForBit` ×4 |
| Vendor workspace 0 | `ExactWorkspaceInAPoisonedArena` ×4, `GetriCandidatesCpu.CpuQueueRunsNoNativeFamily` ×4 |
| float sm_120 row n=127 batch=512 flipped to blocked | `GetriTranscribedTable.RowsHoldTheOldPreference` |

### (b) Data gate: old predicate vs select on the new tables, off-grid

`getri_transcribe --offgrid 4000 7`: 4000 random off-grid points, log-uniform n ∈ [1, 8192] and batch ∈ [1, 65536],
grid cells excluded, scored for all 4 dtypes × 2 devices × {vendor, vendor-free}. Each point is compared with the
nearest row (`sweep_to_table.nearest`, the Python mirror of `Table::nearest`, with weighted log keys) under
these can_run assumptions: a CUDA sg32 GPU, a square homogeneous A, so both families run with a vendor and only
Blocked runs without one.

**Agreement: 100.000% in every (device, dtype, vendor/vendor-free) cell: 4000/4000 each.** There are no
disagreeing regions. Sanity check on the gate itself: dropping the below-edge grid points 127 and 255 from the
tables drops agreement to 98.075% for float and 98.650% for cfloat: float n ≈ 111-127 and cfloat n ≈ 222-255, the upper halves of the
log gaps 96-128 and 192-256, go to blocked, so the gate can
see a misplaced edge.

### (c) Cross-check against the 424a45bc binary (Auto coverage readback)

There are 22 cells: both sides of each edge, off-grid n, batches below the grid (2, 3, 4, 7, 8, 16), and all
four dtypes. A one-cell probe runs getrf then getri under Auto with `BATCHLAS_COVERAGE_OUT`, and the
`reached,getri,` row is compared. All 22 agree on origin and kernel (float 128/129/200/700 and cfloat
256/257/600 native blocked; everything else vendor). The one textual difference is the algorithm column of
the vendor row, which was `vendor,auto` and is now `vendor,vendor`, because flat selection records the choice spelling.

## Doc changes needed (for the integrator)

- `docs/perf/lu.md:39`, `:66`, `:68`, `:132` cite `route_getri.hh` / `getri_route.hh` line numbers. The window now
  lives in the transcribed tables (`tuned/getri.*.txt`). The evidence anchor `#getri-window-evidence` is
  unchanged.
- `docs/design/vendor-free-status.md:58` and `:112` cite `route_getri.hh:39` and `:65-72`. The equivalent is now
  `can_run` in `src/ops/getri/getri.cc` (vendor-free last resort `blocked`).
- `docs/design/flat-kernel-selection.md` §12/§13: add getri to the migrated ops, recording the transcribed
  sm_89 and sm_120 tables, `batch` as a no-op key, and the Vendor can_run flag pair noted above.
- `tuned/README.md`: list `getri.*.{sm_89,sm_120}.txt` as transcribed.
