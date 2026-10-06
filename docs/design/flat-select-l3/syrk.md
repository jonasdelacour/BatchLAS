# syrk: level-3 flat selection, as built

Branch `flat-select-l3-syrk`, from `origin/main` = `ff340fc6`. These notes are folded into
`docs/design/flat-kernel-selection.md` at integration. Every `file:line` of old code refers to
`ff340fc6`.

## Layout (R1)

- `src/ops/syrk/choice.hh`: families `gram`, `triangular`, `vendor` (all `NoFields`), the
  per-dtype candidate lists (float `{gram, triangular, vendor}`, double `{gram, vendor}`),
  `last_resort = {triangular, gram, vendor}`, keys `form:exact n:log:2 k:log batch:log`, the grid,
  `form_of`, `kMaxGridBatch`, gram's launch geometry, and `herk_gram_pinned()`.
- `src/ops/syrk/syrk.cc`: the public `syrk<B, T>` (moved out of `src/ops/level3/level3.cc`):
  `validate_rank_k<invalid_argument>` -> empty batch is a no-op -> `choose()` -> `TraceScope`
  -> one `std::visit` launch. Instantiated for CUDA, ROCM and NETLIB, float and double.
- No workspace (R5): syrk has no `*_buffer_size`.

## can_run (R3)

| family | term | why |
|---|---|---|
| all native | `B == CUDA && d.is_gpu` | the old reach: the kernels are wired for CUDA only |
| all native | A and C homogeneous | one launch has one (n, k, ld, stride) |
| all native | `transA != ConjTrans` | the tile kernels take BLAS syrk's N/T; `syrk_gram_supported` refuses C (see deviations) |
| all native | n, k, batch >= 1 | |
| all native | `batch <= 65535` | the batch is SYCL dim 0 (grid z for triangular, grid y for gram's 2-D range). Confirmed by launch: at 65535 both run and are correct; at 65536 the direct call throws `Number of work-groups exceed limit for dimension 2 (1) : 65536 > 65535` (`GridBatchCeiling`) |
| gram | `syrk_gram_supported<T>(A, C, transA, false)` (n <= 128, shapes) | the tile is the whole of C. Past it the kernel does not throw, it answers wrongly (`CanRunEqualsLaunch`, n = 129) |
| gram | `max_wg >= gram_threads(n)`, `slm_budget >= 32 * tile * sizeof(T)` | the launch's work-group (64/160/544) and SLM (<= 32 KiB) |
| triangular | `T == float` | 128-bit float packets; not instantiated for double (`if constexpr` in the visitor) |
| vendor | `d.has_vendor_blas` | `level3_vendor_available<B>` |

## Tables

Transcribed for sm_89 and sm_120, identical rows, `source=transcribed:ff340fc6`, CSV
`tuned/transcribed/syrk.csv`, 9180 rows per (dtype, device): 3 forms x 30 n x 6 k x 17 batch.
The transcriber (`tools/transcribe/syrk_transcribe.cc`, deleted at integration) holds verbatim
copies of the old predicates (`syrk_custom_dispatch.cc:22,34-117,129-209`,
`triangular_tiles.hh:126-136`, `syrk_gram_tiles.hh:65,314-338`, `cublas.cc:544-580`,
`level3.cc:117-145`) and runs the old facade once with and once without a vendor library;
`tools/transcribe/syrk_gate.py --fidelity` diffs every block against `git show ff340fc6:` (8
blocks, 0 differ). A cell whose (n, k) contradicts its form holds the decision at the form
representative (k' = n, max(1, (n-1)/2), 2n+1).

The old float rule, as transcribed: n <= 128 gram; above, triangular iff squareish
(`2 min(n,k) >= max(n,k)`) or (n >= 257, k >= 8, batch * T(T+1)/2 >= 160 with T = ceil(n/128));
otherwise the vendor. The cuBLASDx heuristic term `tiled_work >= 8` is always true above n = 128,
so it reduces to squareish. Double: gram iff n <= 128 (GPU, not pinned vendor), else the vendor.

Rows: float `gram | triangular | vendor` (n <= 128), `triangular | vendor`, `vendor | triangular`;
double `gram | vendor`, `vendor`.

## Gates

- **(b) data gate**, `syrk_gate.py --data`, 3000 random off-grid points per dtype (log-uniform
  n, k in [1, 4096], batch in [1, 32768]), both devices:
  - vendor-present agreement with the old Auto choice: float 100.00%, double 100.00%;
  - vendor-free, where the old build served (float 2450 points): 100.00%; where it threw: float
    550 -> triangular, double 1724 -> gram and 1276 -> no route (as before);
  - negative controls: dropping the n=129 rows gives float 96.20%; dropping the n=128 rows gives
    float 95.50% / 94.49% (B) and double 95.97%.
- **(a) tests** (sm_120 box, against the same binaries built from `ff340fc6`), failing gtest names:
  - vendor tree: `syrk_tests`, `syrk_candidates_tests` (39 run, 36 skip off-GPU), `herk_tests`,
    `tuned_tables_tests`, `select_tests`, `options_api_tests` green as on main; `ortho_tests`
    segfaults in `OrthoMatrixTest/7.OrthogonalizeMatrix` exactly as on main (known-defects #13),
    and its other 14 cases pass in both.
  - vendor-free tree: `syrk_candidates_tests` green; `syrk_tests` fails 6 NETLIB cases (no netlib
    in this build) where main fails 11: the 5 CUDA cases main lost for want of a vendor now pass.
    `herk_tests` (16), `ortho_tests` (8), `options_api_tests` (1) fail the same names as main.
  - deliberate breaks of `syrk.cc`/`choice.hh`, each restored and md5-checked against the
    committed file (red sets; /2 float, /3 double CUDA):
    - gram ceiling 127: `PinnedCandidatesStraddleTheirLimits`, `CanRunEqualsLaunch`,
      `SaturatingBatchIsBitIdentical`, `AutoReadsTheTranscribedTable` (/2, /3);
    - `form_of` with 3 min: `AutoReadsTheTranscribedTable` /2 only;
    - k from `A.cols()` always: `AutoReadsEveryKeyField` /2 /3, `TraceLineCarriesTheKey` /2 /3,
      `AutoReadsTheTranscribedTable` /2;
    - coverage k = n: `CoverageRowCarriesBackendKeyAndNativeFlags` /2 /3 only;
    - grid term dropped: `GridBatchCeiling` /2 /3 only;
    - triangular arm launching gram: five float cases plus the two `SyrkCudaCustomTest` triangle
      tests;
    - vendor arm with uplo flipped: every case that reaches the vendor (16 candidates cases, both
      CPU cases, 8 `syrk_tests` cases), and nothing that does not.
- **(c) live coverage**: `syrk_benchmark`, one process per cell, 26 cells on both sides of every
  threshold (22 float, 4 double; old double read from the kernel trace, it recorded no row):
  vendor-present 26/26 identical; vendor-free 14/26 identical and the other 12 are gains (old
  `NoRouteError`, new triangular or gram); the two double cells past n = 128 throw in both.

## Behaviour changes

- Vendor-free gains: every float GPU shape (gram or triangular), double n <= 128 (gram).
- `BATCHLAS_SYRK_ROUTE=gram` at n > 128 now throws (it used to run gram, which answers wrongly
  there). `cublasdx` and `expand` throw `invalid_argument` (unknown family). `triangular` on
  double throws "not a compiled syrk double candidate" (double used to ignore every word but
  `vendor`). Pins on a CPU queue or a heterogeneous batch throw.
- A batch above 65535 goes to the vendor (vendor-free: `NoRouteError`) instead of aborting in the
  launch.
- An invalid shape throws `invalid_argument` in every build (vendor-free used to throw
  `NoRouteError`); an empty batch is a no-op under every pin.
- The coverage row's scalar and backend columns are the real ones (the old float rows were
  hard-coded F32/CUDA; old double rows were not recorded at all).
- herk's opt-in reads `ops::syrk::herk_gram_pinned()` (exactly `gram`); herk no longer throws on
  a syrk word it does not understand.

## Deviations from the shared design

- No S0 scaffold existed on `origin/main`, so this branch does syrk's share of it: the entry point
  moved out of `level3.cc` (its syrk lines only), one `target_sources` line, the SYRK OpSpec, the
  registry entry in `tuned_tables_tests`, the test target.
- `syrk_custom_dispatch.{cc,hh}` stay in the tree, dead (D8: integration deletes them); nothing
  calls `syrk_use_cuda_custom` / `syrk_cuda_custom` / `syrk_route_*` any more.
- ConjTrans stays a can_run term although real C equals T (the triangular kernel would answer it
  correctly): the table has no trans key, so admitting it would move real ConjTrans off the
  vendor, which the old rule never did. The gram kernel's own predicate refuses it.
- The empty-batch early return is new (gemm's precedent).
