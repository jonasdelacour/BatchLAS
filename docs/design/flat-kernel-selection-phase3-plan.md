# Flat selection: phase 3 plan {#design_flat_selection_phase3_plan}

> **Status:** historical, superseded by @ref design_flat_selection. · checked 2026-10-21

The plan that moved posv, trsm and gemm to flat selection (`src/ops/<op>/choice.hh`, ranked by
`tuned/<op>.*.txt`). Every step landed. The page is kept because source comments cite its section
numbers for the grids and family sets (`§1.1`-`§1.3`, `§3`). Current per-op facts: @ref design_flat_selection
§12; rules: §13 there. Each section ends with a "Now" line where the code departs from the plan.

## 1. Families and `can_run`

### 1.1 posv

Files: `src/ops/posv/{choice.hh,posv.cc}`.

| Family | Driver | Workspace | `can_run` |
|---|---|---|---|
| `tiny` | `posv_tiny_dispatch` (fused factor and solve) | `posv_tiny_buffer_size` | `native && n <= posv_tiny_max_n<T>()` (16 for cdouble, else 32) `&& nrhs <= 4 && max_wg >= kPosvTinyWgSize` (64) |
| `cta` | public `potrf` + `potrs_fused_dispatch` | `potrf_buffer_size` | `native && nrhs <= 8 && n*nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)` |
| `blocked` | public `potrf` + 2× public `trsm` | `potrf_buffer_size` | `!A.het && !B.het` |

- `native` = `is_gpu && has_sg32 && !A.het && !B.het && n, nrhs, batch >= 1`. Empty shapes and
  heterogeneous batches throw before selection.
- Families are field-less; every knob derives from the shape.
- Keys: `uplo:exact n:log:3 nrhs:log batch:log` (work \f$\approx n^3/3 + 2n^2\,\mathrm{nrhs}\f$).
- Grid: potrf's n values up to 1024 (33), nrhs {1, 2, 4, 8, 16, 64}, batch {128, 512, 2048, 8192, 32768}.
- Last resort: `blocked`. No vendor family; a `vendor` pin warns and runs Auto.
- The names `cta` and `blocked` are kept although they are misnomers, so benchviz and the scripts
  keep working.

Now: as planned (`src/ops/posv/posv.cc:42-59`).

### 1.2 trsm

Files: `src/ops/trsm/{choice.hh,trsm.cc}`.

| Family | Driver | Planned `can_run` |
|---|---|---|
| `cta` | `trsm_native_v1_dispatch` | `native && order <= 32` |
| `sg_left` | `trsm_native_sg_left_dispatch`, Side::Left, order ≤ 32 | as `cta`, Side::Left only |
| `blocked` | `trsm_native_blocked` with the public `gemm` | `native && trsm_blocked_available<T>()` |
| `vendor` | `backend::trsm_vendor` | `has_vendor_blas` |

- Keys: `side:exact trans:exact order:log:2 q:log batch:log` (ConjTrans folds to T; work
  \f$\propto \mathrm{order}^2\,q\,\mathrm{batch}\f$).
- `uplo` and `diag` are not keys, valid only if a one-time A/B shows at most 3% effect per family.
  Result: no first-choice change in 56 cells (@ref design_flat_selection §12, trsm).
- A batch mismatch (A.batch ≠ B.batch) throws `invalid_argument` in `trsm_validate_params`.
- Last resort: `blocked`, then `vendor`.
- Fields (`cta:wg`, `blocked:outer`): deferred; none exist.

Now: `cta` and `blocked` also need `max_wg >= 32`; `sg_left` needs sub-group 32 and
`max_wg >= 128`; `vendor` is `has_vendor` and homogeneous operands (`src/ops/trsm/trsm.cc:46-62`).

### 1.3 gemm

Files: `src/ops/gemm/{choice.hh,gemm.cc}`. The vendor TUs keep only the vendor implementation.

| Family | Fields | Planned `can_run` |
|---|---|---|
| `direct`, `tiled` | none | true |
| `small` | none | `!complex && max(m,n,k) <= 64 && max_wg >= 128` |
| `reg` | `m n k u` | float only; instantiated for the (ta, tb) pair after the C→T fold |
| `wide` | `m n k` | `wide_trans_matches<T>(cfg, ta, tb)`, plus a measured launch ceiling for double and complex |
| `vendor` | none | `has_vendor_blas` |

- Aligned vs predicated legs, the small bucket and the TR/TC/stages constants are derived inside the
  launchers, never fields.
- Dropped: the 5 experimental variants, the 4 pin-only register variants, split-K (it allocated per
  call, which broke the workspace rule), and cuBLASDx (MathDx absent).
- Keys: `ta:exact tb:exact layout:exact m:log n:log k:log batch:log`. `layout = packed` means A, B and
  C are contiguous with 16-byte bases, otherwise `strided`. Heterogeneous batches are split before
  selection.
- Last resort: `direct`, then `vendor`. `can_run` keeps `precision != Default` and the CPU on vendor.
- Before the rewrite, 18 NN-only register variants computed wrong answers on transposed calls; a pin
  on a `can_run`-false transposed call now throws.

Now (`src/ops/gemm/gemm.cc:114-161`; [the gemm page](../perf/gemm.md), "GEMM: what ships"):

| Planned | Now |
|---|---|
| `small` real only | every scalar; `max(m,n,k) <= small_max_dim<T>()` (32 for 16-byte scalars), work-group, sub-group and local-memory terms (`small_fits`) |
| `reg`, `wide` form match only | also `max_wg >= threads()`, a local-memory term, and `batch <= max_groups_yz` |
| split-K dropped | `splitk:m:n:k:s` family over six wide host tiles, partials in the queue's scratch lease (`scratch_budget`) |
| 10 `reg`, 8 `wide` configs (as built, 2026-10-06) | 13 `reg_configs`, 9 `wide_configs` |

## 2. Table source

No posv, trsm or gemm data could be converted under the old format: posv had no `cta` arm and pins
were never confirmed; trsm had only a pre-barrier native-versus-vendor split; gemm had one variant
per shape against vendor only.

| Option | Verdict |
|---|---|
| A. Measure on sm_120 with existing harnesses | Used for posv only. trsm and gemm had no harness that met the protocol. |
| B. Pull the tuner core forward (P3.2) | Adopted. The only route for trsm and gemm. |
| C. Borrow sm_120 tables on sm_89 | Rejected. It inverts sm_89 tuning. |
| D. **Transcribe old routing** as untimed ranked rows | **Adopted for sm_89.** Example: posv in the tiny window gives `tiny -`, then `cta -`, then `blocked -`. |
| E. Measure on sm_89 | Deferred. Estimated 15 h for posv, 22 h for trsm, 60-80 h for gemm on the shared box. |

Transcription used a host-only generator linked against the parent build, run before the old code
was deleted, cross-checked by a `route_diff` coverage readback on a sample (gemm: an untimed
`BATCHLAS_KERNEL_TRACE=1` readback). Off-grid cells where the old predicate and the nearest
transcribed row disagree need a timed check.

Now: sm_89 tables are still transcribed; sm_120 tables are deep-measured (gemm float and cfloat
2026-10-21; double and cdouble transcribed). Provenance per table: @ref tuned_tables_readme.

## 3. Key space and grids

**trsm.** Per-dtype grid: order {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
768, 1024}, q {1, 2, 4, ..., 1024, 4096} (no 2048), batch {128, 512, 2048, 8192, 32768}
(`src/ops/trsm/choice.hh`). Side × trans (N/T) is the key set: 4 combinations, about 5.5 h on 4 GPUs.
The full side × uplo × trans × diag set (24 combinations) was infeasible. `cta` is never timed above
order 32.

**gemm.** A full lattice is infeasible, so the grid is declared per (ta, tb, layout)
(`src/ops/gemm/choice.hh`, `grid_*`):

| Part | Values |
|---|---|
| Squares | m = n = k ∈ {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}, every form, both layouts |
| Panels | m, n ∈ {32, 64, 128, 256, 512, 1024, 2048} × k ∈ {8, 16, 32, 64, 96, 128}, the issued forms (NN, NT/NC, TN/CN); `packed` from m, n ≥ 128 |
| Skinny | 32 × n and m × 32 with m, n ∈ {64, ..., 2048}, k ∈ {256, 1024} (planned: m×32×k and 32×n×k) |
| Large k (added, not transcribed for sm_89) | Gram m = n ∈ {16, 32, 64} × k ∈ {1024, 2048, 4096}; 32 × skinny at k = 2048; panel forms, strided |
| Batch | {128, 2048, 32768} |

Real types use 4 (ta, tb) combinations; complex types the issued ones only. A planned screening pass
(one pass, keep candidates within 1.5× of the best; about 6-8 h on 4 GPUs) was not approved; shipped
times come from the full protocol.

## 4. Gate method

1. Run the parent build with `BATCHLAS_COVERAGE_OUT` (gemm: `BATCHLAS_KERNEL_TRACE=1`) and record the
   old choice at each gate cell; map it to a spelling.
2. In the branch binary, time `pin:<old spelling>` against `auto`, interleaved in one process, two
   passes with the order reversed (`batchlas_tune --gate`).
3. Fail if auto/pinned > 1.05 and the loss reproduces. Skip cells where both pick the same choice,
   and list them in the evidence CSV.

Correctness gates per op: the op's `_tests`, its `_candidates_tests`, `tuned_tables_tests` and
`select_tests`; for trsm also the callers (potrf, getrs, getri, gesv, ortho); for gemm also
`device_calls_tests` and every gemm consumer label. Diff failing names against
`tests/known-failures.txt`. One measuring process per box, on device 1 of the 4090.

## 5. Risks and maintainer decisions

| Risk | Disposition |
|---|---|
| K1. Without a posv table, Auto falls to `blocked`, 2-20× slower at n ≤ 32 | Table was a merge blocker. `AutoReadsEveryKeyField` guards it. |
| K3. trsm batch ≤ 5 moves from vendor to native | Caller suites run in the P3.3 gate. |
| K4. Complex trsm vendor on CUDA is BatchLAS's own kernel, with a possible `int` overflow (cfloat order 512 × batch 8192 = \f$2^{31}\f$) | Confirmed and fixed: batch offsets are 64-bit (@ref design_flat_selection §12, trsm). |
| K5. symm, syrk, syr2k and trmm called `gemm_vendor` directly and lost native gemm | Routed through the public `gemm` (Q7). |
| K7. A multi-GPU sweep breaks the one-process-per-box rule | Re-measure a sample single-GPU. |

Maintainer answers (2026-10-04):

| Question | Answer |
|---|---|
| Q1 transcribed sm_89 tables | Yes, until a retune on the 4090. |
| Q3 vendor before direct in gemm's last resort | Not approved: `direct`, then `vendor`. |
| Q4 blackwell kernels | Before trsm and gemm, as kernels only, with the `is_sm120_family` and `cuda_cc` predicates dropped. |
| Q5 screening pass for gemm | Not approved; the full protocol is used. |
| Q6 trsm batch mismatch throws | Yes. |
| Q7 symm, syrk, syr2k, trmm through the public gemm | Yes (@ref design_flat_selection §13). |
| Q8 delete experimental gemm variants | Yes. |
| Q9 trsm `uplo` and `diag` not keys | Yes, subject to the ≤ 3% A/B (passed). |
| Q10 keep the `cta` and `blocked` names | Yes. |
