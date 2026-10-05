# gesv: flat kernel selection (phase 5), as built

Branch `flat-select-p5-gesv`, based on `424a45bc` (the flat-selection stack head). This page holds
the as-built notes, deviations, gate results and the doc changes the integrator should fold into
`docs/design/flat-kernel-selection.md` §12, `tuned/README.md` and AGENTS.md. It mirrors posv
(`src/ops/posv/`), the other op with no vendor arm on any backend.

## What changed

| Piece | File |
|---|---|
| Vocabulary: `Tiny`, `Blocked` (fieldless), aliases, last resort, keys, grid | `src/ops/gesv/choice.hh` |
| `gesv` / `gesv_buffer_size` entry points, `can_run`, `choose`, `launch`, `workspace`, `native_facts` | `src/ops/gesv/gesv.cc` |
| Removed from the facade (`SOLVE_ONE`/`SOLVE_ALL`, `solve_throw_unroutable`) | `src/dispatch/entry_points/factorization.cc` |
| Deleted: the old table and its shape builder | `include/batchlas/blas/dispatch/route_gesv.hh`, `src/backends/gesv_route.hh` |
| `kGesvTinyWgSize = 64` (new, sycl-free) + `static_assert` against the kernel's `kTinyWg` | `src/extensions/solve_native.hh`, `src/extensions/gesv_tiny.cc` |
| Transcriber (old `RouteTable<Op::gesv,T>` + `resolve_route`, ranked per cell; `--points` mode for the gate) | `tools/transcribe/gesv_transcribe.cc` |
| Transcriber CSVs, 8 tables (4 dtypes x sm_89, sm_120, identical rows) | `tuned/transcribed/gesv.sm_{89,120}.csv`, `tuned/gesv.<dtype>.<device>.txt` |
| OpSpec `GESV` | `scripts/sweep_to_table.py` |
| Off-grid data gate | `tools/transcribe/gesv_offgrid_gate.py` |
| Tests | `tests/gesv_candidates_tests.cc` (new), `tests/gesv_tests.cc` (G7/G8 moved), `tests/tuned_tables_tests.cc`, `tests/linalg_layer_tests.cc` (comment) |
| `factor_bench`: gesv pins parsed as choice spellings | `benchmarks/factor_bench.cc` |
| Pointer from the old measurement record to the new code | `docs/perf/lu.md` (note at the top of "The fused gesv tier"; no heading renamed) |
| Comment updates | `include/batchlas/blas/dispatch/route.hh` (Op enum comment), `include/batchlas/blas/linalg-ops.hh` |

## Families (one per kernel driver)

- `tiny` = `sycl_gesv::gesv_tiny_dispatch` (fused LU factor + solve, register-resident).
- `blocked` = public `getrf`, then public `getrs` (NoTrans); each child picks its own kernel.

No int fields: the old router chose no knob. Tiny's N/NR buckets stay derived in the driver.
No vendor family: no vendor ships a batched gesv; a `vendor` pin warns once and runs Auto (as posv).

`can_run` (R3), each clause the driver's own check (`gesv_tiny_dispatch`, `gesv_tiny.cc:455-515`):
- `tiny`: backend != NETLIB (kept from the old `supports()`: the GPU kernel packs 1-based int32
  pivots into the int64 span's low half, netlib writes true int64), GPU, sub-group 32 offered,
  homogeneous A and B, n, nrhs, batch >= 1, n <= `gesv_tiny_max_n<T>()` (32; cdouble 16),
  nrhs <= `kGesvTinyMaxRhs` (4), MAX_WORK_GROUP_SIZE >= `kGesvTinyWgSize` (64). The last term is
  **new**: the old `supports()` lacked the driver's work-group check (same as posv P3.1).
- `blocked`: homogeneous batch. Its children decide everything else.

Heterogeneous batches and empty problems (n, nrhs or batch 0) throw `internal_error` before
`choose()` (`throw_if_unservable`), so a pin cannot take them either; the old router refused both
the same way (empty: through `solve_throw_unroutable`). Last resort: `blocked`. Aliases:
`native:tiny`, `native:blocked`. Unknown spellings (`cta`, `native:cta`, `composed`, `tiny:1`)
throw (R6); under the old router they silently meant Auto. Wider break: spellings that the old
`dispatch::parse_route_value` (`route_env.hh`) *recognised* also throw now, namely
`batchlas_tiny`/`batchlas-tiny` and `batchlas_blocked`/`batchlas-blocked` (old: Tiny / Blocked),
the `batchlas:` origin such as `batchlas:tiny` (old: native origin), and `netlib` (old: vendor
origin, so Auto for gesv). posv made the same choice; a grep of `benchmarks/`, `scripts/` and
`evaluation/` finds no user. Integrator: carry this into the AGENTS.md §9 / design §12 note.

Workspace (R5): `gesv_buffer_size` runs the same `choose()`. Tiny: `gesv_tiny_buffer_size`.
Blocked: `getrf_buffer_size + getrs_buffer_size` (a sum: the launch cuts the span at getrf's size).

Trace: `TraceScope("gesv", ...)` with key fields `n`, `nrhs`, `batch`; the coverage row keeps the old
spelling (m = k = order, n = nrhs), and `native_facts` computes existed/supported from the
candidate list and `can_run` only when coverage is recording.

## Tables (maintainer decision: transcribed, both devices)

Keys `n:log:3 nrhs:log` (work 2n^3/3 + 2 n^2 nrhs). These are exactly what the old predicates
read; batch only entered as `>= 1`, and there is no categorical key. The old predicates read no
architecture, so one transcription is written for sm_89 and sm_120, `source=transcribed:424a45bc`.
Capacities (tiny order and nrhs ceilings) are unlimited in the transcription; `can_run` re-applies
them at run time. The old preference then depends on n only:
float `tiny|blocked` for n <= 32, cfloat for n <= 16, `blocked` everywhere else (double and
cdouble have no window). The list stops at `blocked`, which `can_run` never refuses.

Grid (both sides of every old threshold, log-spaced elsewhere):
- n: 1 2 3 4 6 8 12 16 | 17 20 24 28 32 | 33 40 48 64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096
  (cfloat window 16|17, cdouble tiny ceiling 16|17, float window and tiny ceiling 32|33).
- nrhs: 1 2 4 | 5 8 16 64 256 (tiny nrhs ceiling 4|5).

`sweep_to_table.py --transcribe` regenerates the eight tables byte-identically from the CSVs, and the
transcriber rebuilt against the 424a45bc headers regenerates the CSVs byte-identically.
`sweep_to_table.py --check`: OK (needs `git lfs pull` of `benchmarks/results/routing/` first).

## Gate results

**(a) Correctness**, GPU 0 (RTX PRO 6000 Blackwell, sm_120), `ctest -R
'^(gesv_candidates_tests|gesv_tests|select_tests|tuned_tables_tests|linalg_layer_tests)$'`:

| tree | result |
|---|---|
| `build` (vendor) | 5/5 passed |
| `build-vf` (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) | 5/5 passed |
| baseline 424a45bc (`gesv_tests select_tests tuned_tables_tests linalg_layer_tests`) | 4/4 passed |

No failing names on either side, so the name diff is empty. gesv_candidates_tests: 57 cases run on
CUDA (14 typed x 4 dtypes + the table test) plus `GesvNetlib`; the NETLIB typed instantiations skip
(CPU queue), which is why `GesvNetlib.TinyRefusedBlockedSolves` exists.

Deliberate breaks of `src/ops/gesv/gesv.cc` (one build, three disjoint axes; restored from a saved
copy, md5 `5e2f5d77...` verified):
- drop `Bk != Backend::NETLIB` from Tiny's `can_run` -> red: `GesvNetlib.TinyRefusedBlockedSolves`
  only ("gpu: a Tiny pin was accepted on NETLIB").
- drop Tiny's `nrhs <= kGesvTinyMaxRhs` -> red: `PinnedCandidatesStraddleTheirLimits`,
  `CanRunEqualsLaunch`, `CanRunFalsePinsThrow` (all four dtypes), `AutoReadsTheTranscribedTables`
  (float: the n=8 nrhs=5 row), `CoverageRowCarriesNativeFlags` (float, cfloat: n=8 nrhs=9).
- fix `key_of`'s nrhs at 1 -> red: `AutoReadsEveryKeyField` (all dtypes) only.

**(b) Off-grid data gate** (`tools/transcribe/gesv_offgrid_gate.py <transcriber> --per-dtype 2500
--seed 1`): 10000 random off-grid (dtype, n, nrhs) points per device, 40% of them in n 8..48 (the
window and ceiling edges), n up to 6000, nrhs up to 400. The old router's first choice with the
real capacities (transcriber `--points`, i.e. the real `RouteTable` + `resolve_route`) against the
new table's nearest row (the `Table::nearest` rule, weights included) filtered by `can_run`'s
capacity terms:

| device | float | double | cfloat | cdouble |
|---|---|---|---|---|
| sm_89 | 2500/2500 | 2500/2500 | 2500/2500 | 2500/2500 |
| sm_120 | 2500/2500 | 2500/2500 | 2500/2500 | 2500/2500 |

100.00% everywhere, no disagreeing region: every old threshold sits between two adjacent integers
that are both grid points, and nrhs does not change the transcribed ranking.

**(c) Live cross-check** against the 424a45bc binary: `factor_bench gesv <t> n n nrhs 256 1
--arms=native` with `BATCHLAS_COVERAGE_OUT`, 21 cells (float n 4/16/25/32/33/64/8 with nrhs
1..9 incl. 32x5 and 8x9; cfloat 9/16/17/32/12x5; double 4/16/32/100; cdouble 8/16/17/40). The
`reached,gesv` row's chosen algorithm agrees on 21/21 (old and new). `native` is the old bare-origin
walk, which for gesv is Auto (no vendor), and Auto under the new pin vocabulary.

## Deviations and notes for the integrator

- AGENTS.md / flat-kernel-selection.md §12: add gesv to the migrated-op list; `route_gesv.hh` and
  `src/backends/gesv_route.hh` are gone, `BATCHLAS_GESV_ROUTE` takes choice spellings.
- `tuned/README.md`: list `gesv.<dtype>.{sm_89,sm_120}.txt` as transcribed from 424a45bc with
  `tools/transcribe/gesv_transcribe.cc`.
- `src/CMakeLists.txt` gets its own `target_sources(... ops/gesv/gesv.cc)` line to keep the merge
  local; fold it into the migrated-ops line.
- `run_local_checks.sh` passes except `check_cmake_syntax.py`, which scans the local `build-vf/` tree
  (generated CMake files); no repository file fails.
- The Tiny work-group gate is new (see `can_run`). On every GPU BatchLAS targets the limit is >= 64,
  so no routing changes.
- `factor_bench`'s gesv `composed` arm pins `BATCHLAS_GETRF_ROUTE=tiny`, which throws above getrf's
  tiny ceiling once getrf is migrated; the getrf branch already fixes that (its own note).
