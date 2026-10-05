# Phase 5: orgqr on flat kernel selection

As-built notes for the integrator. Branch `flat-select-p5-orgqr`, based on `424a45bc` (the
flat-selection stack head after P3.3 trsm). The spec is
[flat-kernel-selection.md](../flat-kernel-selection.md); this file records what orgqr did, where
it deviates, the gate results, and the edits to shared docs that the integrator still has to
make.

## What moved

| before (424a45bc) | after |
|---|---|
| `include/batchlas/blas/dispatch/route_orgqr.hh` (`RouteTable<Op::orgqr,T>`, `kOrgqrOrder`) | deleted |
| `src/backends/orgqr_route.hh` (`orgqr_op_shape`, `orgqr_route`) | deleted |
| `orgqr` / `orgqr_buffer_size` in `src/dispatch/entry_points/factorization.cc` | `src/ops/orgqr/orgqr.cc` |
| `preferred()`: native iff `rows <= 512 && cols <= 512` | `tuned/orgqr.<dtype>.{sm_89,sm_120}.txt`, transcribed |
| `RouteOrgqr.*` in `tests/route_vocabulary_tests.cc` | ported into `tests/orgqr_candidates_tests.cc` |

`src/ops/orgqr/choice.hh` holds two fieldless families in tie order: `Blocked`
(`orgqr_blocked_dispatch`, the identity fill plus the public `ormqr`; its `orgqr_nb` stays
derived from type and `min(m, n)`, because the old code never chose it) and `Vendor`
(`backend::orgqr_vendor`, the per-item library loop). Aliases are `native:blocked` -> `blocked`
and `vendor:auto` -> `vendor`. The last-resort order is `vendor, blocked`, ordered by
generality: the vendor runs every shape, including n > m, CPU queues and heterogeneous batches.

`can_run` mirrors `orgqr_blocked_dispatch`'s own checks (R3): GPU, `orgqr_blocked_available<T>`,
homogeneous batch, `m, n, batch >= 1`, `n <= m`. The tau-length check is an argument error and is
left to the driver. The old table's complex-`Trans` exclusion is gone, because the apply is fixed
at `(Left, NoTrans)` and that clause could never fire. Vendor's `can_run` is
`d.has_vendor_solver`. `device()` fills that field from `factorization_vendor_available<B>`, the
same constant that compiles the Vendor arm (cuBLAS + cuSOLVER on CUDA). It does **not** use
`solver_vendor_available`, which differs on CUDA.

The workspace (R5) comes from the same `choose()`. Blocked is sized by
`orgqr_blocked_buffer_size` through the public `ormqr_buffer_size`, and Vendor by
`orgqr_vendor_buffer_size`. The old `max(native, vendor)` path is gone. This keeps the fix for the
[orgqr_buffer_size latent defect](../../perf/qr.md#the-orgqr_buffer_size-latent-defect): a native
call is never sized by the batch-linear vendor loop.

The trace and coverage key is `m`, `n` (+ `batch`). The coverage shape keeps the old builder's
fields: `k = min(m, n)`, `side = Left`, `transA = NoTrans`, and the backend is set. Vendor-free
burn-down still records a `miss` row, through `throw_no_vendor_route` when nothing can run.

## Keys and grid

The old predicates read `m` and `n` only: no batch, no architecture, no categorical field. Keys
are `m:log n:log:2` (work is about m n^2). The grid on each axis is
`1 2 4 8 16 32 64 128 256 384 512 513 768 1024 2048 4096 8192`, restricted to `n <= m` (153
cells per table). 512/513 puts a point on both sides of the only threshold. Because the grid is a
product grid with the same points on both axes, weighted nearest-point lookup snaps each axis
independently, so the 512 edge is reproduced exactly. Above the diagonal Blocked cannot run, and
`can_run` sends those shapes to the vendor at run time.

## Transcription

`tools/transcribe/orgqr_transcribe.cc` is built against the 424a45bc headers. It ranks the old
`resolve_route` order at every cell by excluding the winner and resolving again. It writes
`tuned/transcribed/orgqr.csv`, and
`scripts/sweep_to_table.py --transcribe tuned/transcribed/orgqr.csv --sha 424a45bc` (OpSpec
`ORGQR`) writes the eight tables. sm_89 and sm_120 have identical rows (checked by
`TunedTables.OrgqrTablesHoldExactlyTheChoiceGridOnBothDevices`). The CSV and the tables were
regenerated from scratch on this branch, and the output is byte-identical (CSV md5
`b6736287ce08e59da35a47a2d6a9e6d3`). `sweep_to_table.py --check` passes all eight tables. Note:
`--check` needs the LFS objects under `benchmarks/results/routing/`
(`git lfs pull --include="benchmarks/results/routing/*"`). On a pointer checkout it dies parsing
`sm120_potrf_sweep.jsonl`.

## Gate results

**(a) Correctness.** These targets ran on GPU 0: orgqr_candidates_tests, orgqr_tests,
geqrf_tests (orgqr's caller in tests), ortho_tests (`src/extensions/ortho.cc` calls orgqr),
select_tests, tuned_tables_tests, route_vocabulary_tests and options_api_tests.

- `build`: 7/8 pass. ortho_tests segfaults, as it does in the 424a45bc baseline. The set of
  failing gtest case names is empty on both sides.
- `build-vf`: orgqr_candidates_tests, select_tests, tuned_tables_tests, route_vocabulary_tests
  and geqrf_tests pass. options_api_tests, ortho_tests and orgqr_tests fail, and those 21
  failing case names (`OptionsApi.Blas3OptionsMatchPositional`, `OrgqrTest/0-3` on Backend 6,
  `OrthoMatrixTest` / `OrthoAgainstMTest`) are identical to the 424a45bc vendor-free baseline.
  orgqr_candidates_tests first failed three cases in build-vf: its fixture factored wide inputs
  with a geqrf that has no vendor-free route. That is fixed in the test.
- Deliberate breaks, each restored from a saved copy and md5-verified
  (`bee1b1e3a7c9161864578d8cbcb78a21`):
  - Dropping `n <= m` from Blocked's `can_run` turns exactly `CanRunEqualsLaunch`,
    `CanRunFalsePinsThrow` and `PinnedCandidatesStraddleTheirLimits` red (x4 dtypes).
  - Keying `n` on `A.rows()` in `key_of` turns exactly `AutoReadsEveryKeyField` and
    `TraceKeyIsMAndN` red.

**(b) Data gate.** The transcriber's `--random` mode evaluates the OLD router at 2500 off-grid
points per dtype. The points are log-uniform `(m, n)` with log2 extents in [0, 14), both
triangles, and none of them on the grid. A second band of 2500 per dtype has log2 extents in
[8, 10), around the edge. Each point is replayed through `sweep_to_table.nearest` (the same rule
as `Table::nearest`) on the new table plus can_run (GPU, homogeneous batch), for both devices,
with and without a vendor (`p5_orgqr_gate.py`). The result is **100.00% agreement in all 32
(dtype x device x vendor) cells, for both bands**, with no disagreeing regions.

The gate is not vacuous. With the 513 rows deleted, float agreement drops to 98.36% on the full
band and 91.92% on the edge band.

**(c) Coverage cross-check.** `factor_bench orgqr <dt> m n 0 batch 1 --arms=auto` was run with
`BATCHLAS_COVERAGE_OUT` on 20 cells: the 512/513 edges on each axis, near-diagonal cells, tiny
and huge batch, and all four dtypes. Every cell reached the same family in the 424a45bc binary
and in this branch: 11 native/blocked and 9 vendor. The only difference is the vendor algorithm
spelling, `auto` before and `vendor` now, as for trsm.

## Deviations and notes

- **Bad pins throw (R6).** A `blocked` pin on a shape Blocked cannot run used to fall through to
  the vendor. It now throws `invalid_argument` ("cannot run this shape"). An unrecognised
  `BATCHLAS_ORGQR_ROUTE` used to mean Auto and now throws.
- **Two variables still govern a native orgqr.** The Blocked arm calls the public `ormqr`, which
  makes its own selection (`BATCHLAS_ORMQR_ROUTE`). ormqr is migrating in parallel
  (`flat-select-p5-ormqr`). orgqr depends only on the public `ormqr` / `ormqr_buffer_size`
  signatures, so the two branches do not conflict.
- **Entry-point instantiation** follows posv and trsm. It is keyed on the device family. Without
  the vendor library the Vendor arm compiles to a throw.
- The `orgqr_route.hh` waiver line was removed from `.github/ci/comment_density_waivers.txt`.

## Doc changes the integrator still needs

- `docs/design/flat-kernel-selection.md` §12 (as-built): add orgqr to the migrated-op list and
  mention the both-devices transcription.
- `tuned/README.md`: list `orgqr.*.{sm_89,sm_120}.txt` as transcribed from 424a45bc.
- `AGENTS.md` §9: nothing orgqr-specific.
- These were already updated on this branch: `docs/perf/qr.md` (route-arms table, the shipped
  ceiling section, gate references), `docs/design/vendor-free-status.md` (orgqr row and GPU-only
  list), `include/batchlas/blas/functions/orgqr.hh` and `src/extensions/orgqr_blocked.cc`
  comments.
