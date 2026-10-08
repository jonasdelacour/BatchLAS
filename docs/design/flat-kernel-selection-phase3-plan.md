# Flat selection: phase 3 plan {#design_flat_selection_phase3_plan}

> **Status:** historical, superseded by @ref design_flat_selection (section 12 "As built",
> section 13 "Phase 3 decisions").

Phase 3 moved posv, trsm and gemm from the old route tables to flat selection
(`src/ops/<op>/choice.hh`, ranked by `tuned/<op>.*.txt`). This page records the planned family
sets, `can_run` clauses, table sources and gate method. Every PR it planned has landed. Where the
build departed from this plan, the maintainer decisions in @ref design_flat_selection govern.
Its file:line references describe a tree that no longer exists.

## 1. Families and `can_run`

### 1.1 posv

Files: `src/ops/posv/{choice.hh,posv.cc}`.

| Family | Driver | Workspace |
|---|---|---|
| `tiny` | `posv_tiny_dispatch` (fused factor and solve) | `posv_tiny_buffer_size` |
| `cta` | public `potrf` + `potrs_fused_dispatch` | `potrf_buffer_size` |
| `blocked` | public `potrf` + 2x public `trsm` | `potrf_buffer_size` |

- Families are fieldless. Every knob derives from the shape.
- Keys: `uplo:exact`, `n:log:3`, `nrhs:log`, `batch:log`. Work is about n³/3 + 2n²·nrhs.
- Grid: n from the potrf grid, nrhs {1, 2, 4, 8, 16, 64}, batch {128, 512, 2048, 8192, 32768}.
- Last resort: `blocked`. There is no vendor family.
- Names stay `cta` and `blocked` even though they are misnomers, so benchviz and the scripts
  keep working. The class word `native` behaves as Auto.

`can_run`:

| Family | Clause |
|---|---|
| `tiny` | `native && n <= posv_tiny_max_n<T>()` (16 for cdouble, else 32) `&& nrhs <= 4 && d.max_wg >= kPosvTinyWgSize` |
| `cta` | `native && nrhs <= 8 && n*nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)` |
| `blocked` | `!A.het && !B.het` |

`native` is `d.is_gpu && d.has_sg32 && !A.het && !B.het && n, nrhs, batch >= 1`. Empty shapes
throw before selection.

### 1.2 trsm

Files: `src/ops/trsm/{choice.hh,trsm.cc}`.

| Family | Driver | `can_run` |
|---|---|---|
| `cta` | `trsm_native_v1_dispatch` | `native && order <= 32` |
| `sg_left` | sub-group Side::Left kernel (order <= 32), ported from the blackwell branch | as `cta`, Side::Left only |
| `blocked` | `trsm_native_blocked` with the public `gemm` | `native && trsm_blocked_available<T>()` |
| `vendor` | `trsm_vendor` | `d.has_vendor_blas` |

- Keys: `side:exact`, `trans:exact` (C folds to T), `order:log:2`, `q:log`, `batch:log`.
  Work is proportional to order²·q·batch.
- `uplo` and `diag` are not keys. This is valid only if a one-time A/B shows at most 3% effect for
  each family; if vendor fails that A/B, add `uplo:exact`.
- A batch mismatch (A.batch ≠ B.batch) throws `invalid_argument` in `trsm_validate_params`. It used
  to go silently to vendor.
- Last resort: `blocked`, then `vendor`.
- Fields (`cta:wg`, `blocked:outer`) were deferred to phase 4.

### 1.3 gemm

Files: `src/ops/gemm/{choice.hh,gemm.cc}`. The vendor TUs keep only the vendor implementation.

| Family | Fields | `can_run` |
|---|---|---|
| `direct`, `tiled` | none | true |
| `small` | none | `!complex && max(m,n,k) <= 64 && d.max_wg >= 128` |
| `reg` | `m n k u` | float only; instantiated for the (ta, tb) pair after C→T fold |
| `wide` | `m n k` | `wide_trans_matches<T>(cfg, ta, tb)`, plus a measured launch ceiling for double and complex |
| `vendor` | none | `d.has_vendor_blas` |

- Aligned vs predicated legs, the SmallBatched bucket and the TR/TC/Stages constants are derived
  inside the launchers, never fields.
- Dropped: the 5 experimental variants, the 4 pin-only register variants, and split-K (it allocates
  per call, which breaks the workspace rule). cuBLASDx was removed because MathDx is absent.
- Keys: `ta:exact tb:exact layout:exact m:log n:log k:log batch:log`. `layout = packed` means
  A, B and C are contiguous with 16-byte bases; otherwise `strided`. Heterogeneous batches are
  handled before selection.
- Last resort: `direct`, then `vendor`. `can_run` keeps `precision != Default` and the CPU on vendor.
- Before the rewrite, 18 NN-only register variants computed wrong answers on transposed calls. A
  pin on a `can_run`-false transposed call now throws.

## 2. Table source

No posv, trsm or gemm data could be converted under the old format: posv had no `cta` arm and
pins were never confirmed; trsm had only a pre-barrier native-versus-vendor split; gemm had one
variant per shape against vendor only.

| Option | Verdict |
|---|---|
| A. Measure on sm_120 with existing harnesses | Used for posv only. Trsm and gemm had no harness that met the protocol. |
| B. Pull the tuner core forward (P3.2) | Adopted. It is the only route for trsm and gemm. |
| C. Borrow sm_120 tables on sm_89 | Rejected. It inverts sm_89 tuning. |
| D. **Transcribe old routing** as untimed ranked rows | **Adopted for sm_89.** Example: posv in the tiny window gives `tiny - \| cta - \| blocked -`. Header `source=transcribed:<sha>`. |
| E. Measure on sm_89 | Deferred. Estimated 15 h for posv, 22 h for trsm, 60-80 h for gemm on the shared box. |

Transcription uses a host-only generator linked against the parent build, run before the old code
was deleted. Cross-checks: a `route_diff` coverage readback on a sample, and for gemm an untimed
`BATCHLAS_KERNEL_TRACE=1` readback.

Keep the sm_89 transcribed rows until a phase-4 retune on the 4090. Off-grid cells where the old
predicate and the nearest transcribed row disagree need a timed check.

## 3. Key space and grids

**trsm.** Per-dtype grid: order {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
768, 1024}, q {1, 2, 4, ..., 4096}, batch {128, 512, 2048, 8192, 32768}. Side × trans(N/T) is the
recommended key set: 4 combinations, about 5.5 h on 4 GPUs. The full side × uplo × trans × diag
set (24 combinations) was infeasible. Cta is never timed above order 32.

**gemm.** A full lattice is infeasible, so the grid is declared per (ta, tb, layout):

- Squares: m = n = k in {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}.
- Panels: m, n in {32, 64, 128, 256, 512, 1024, 2048} × k in {8, 16, 32, 64, 96, 128}, for the forms
  the library issues (NN, NT/NC, TN/CN).
- Skinny: m×32×k and 32×n×k.
- Batch: {128, 2048, 32768}. Real types use 4 (ta, tb) combinations; complex types use the issued
  ones only. `packed` layout appears on squares and panels with m, n >= 128, plus small squares,
  because a `packed` call at small m sees `packed` rows.
- A screening pass (one pass, keep candidates within 1.5x of the best) cuts the sweep to about 6-8 h
  on 4 GPUs. Shipped times still come from the full protocol.

## 4. Gate method

For each op:

1. Run the parent build with `BATCHLAS_COVERAGE_OUT` (gemm: `BATCHLAS_KERNEL_TRACE=1`) and record
   the old choice at each gate cell. Map it to a spelling.
2. In the branch binary, time `pin:<old spelling>` against `auto`, interleaved in one process, two
   passes with the order reversed.
3. Fail if auto/pinned > 1.05 and the loss reproduces. Skip cells where both pick the same choice,
   and list them in the evidence CSV.

Correctness gates per op: the op's `_tests`, its `_candidates_tests`, `tuned_tables_tests` and
`select_tests`; for trsm also the callers (potrf, getrs, getri, gesv, ortho); for gemm also
`device_calls_tests` and every gemm consumer label. Diff failing names against
`tests/known-failures.txt`. Run one measuring process per box, on device 1 of the 4090.

## 5. Risks and maintainer decisions

| Risk | Disposition |
|---|---|
| K1. Without a posv table, Auto falls to `blocked`, 2-20x slower at n <= 32 | Table is a merge blocker. `AutoReadsEveryKeyField` guards it. |
| K3. trsm batch <= 5 moves from vendor to native | Caller suites run in the P3.3 gate. |
| K4. Complex trsm vendor on CUDA is BatchLAS's own kernel, with a possible `int` overflow (cfloat order 512 × batch 8192 = 2³¹) | Host verify in the tuner. Unverified at the time of planning. |
| K5. symm, syrk, syr2k and trmm called `gemm_vendor` directly and lost native gemm | Route them through the public `gemm` (decided, see below). |
| K7. A multi-GPU sweep breaks the one-process-per-box rule | Re-measure a sample single-GPU. |

Maintainer answers, from @ref design_flat_selection section 13:

- **Q1 (transcribed sm_89 tables): yes.** Until a retune on the 4090.
- **Q4 (blackwell kernels): before trsm and gemm,** as kernels only, with the `is_sm120_family` and
  `cuda_cc` predicates dropped.
- **Q8 (delete experimental gemm variants): yes.**
- **Q10 (keep `cta` and `blocked` names): yes.**
- **Q5 (screening pass for gemm): not approved** at planning time; the full protocol is used.
- **Q3 (vendor before direct for gemm last resort): not approved.** Gemm's last resort is `direct`
  then `vendor`, as built.
- **Q7 (symm, syrk, syr2k, trmm through public gemm): recorded as decided on 2026-10-04.** Section 13
  of @ref design_flat_selection carries the final entry.
- **Q6 (trsm batch mismatch throws): yes.**
- **Q9 (trsm uplo and diag not keys): yes,** subject to the ≤3% A/B.
