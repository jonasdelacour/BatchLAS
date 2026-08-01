# STEDC Merge Kernel — Optimization Research Notes

Research log for speeding up the STEDC merge path. Captures the profiling picture,
the code-level findings, the relevant literature, and a prioritized plan.

Status legend: **[done]** implemented · **[open]** not started.

---

## 1. Profiling picture

From `.github/skills/nvidia-kernel-profiling/references/batchlas-stedc.md`
(RTX 4090, sm_89, `stedc_benchmark --backend=CUDA --type=float 256 256 16 0 32 1`):

| Metric | `StedcFusedCtaMerge<CUDA, float, 32>` |
| --- | --- |
| Duration | 101.50 us |
| SM throughput | 8.80% |
| Memory throughput | 8.80% |
| DRAM throughput | 0.14% |
| Registers/thread | 78 |
| Block size | 32 |
| Grid size | 256 |
| Theoretical occupancy | 50.00% |
| Achieved occupancy | 3.96% |
| Waves per SM | 0.08 |

Kernel share of total GPU time: `SteqrCTAKernel` 35.7%, `StedcFusedCtaMerge` 30.3%,
`Matrix::Identity` 13.9%, `PermutedCopyKernel` 5.4%, `ampere_sgemm_128x128_nn` 4.9%.

**Conclusion: the merge kernel is neither compute- nor bandwidth-bound.** It is
serialization/latency-bound. Achieved occupancy is ~1/12 of theoretical, which
points at intra-kernel serialization rather than launch geometry alone.

---

## 2. Code-level findings

### 2.1 Rescale and normalize were serialized over `dd` — **[done]**

`stedc_merge_cta.cc`, `maybe_rescale_vectors` and `normalize_vectors`.

Both looped `for eid = 0 .. dd-1` performing a **whole-work-group**
`reduce_over_group` plus a `group_barrier` per index. The root solve directly
above them is already partition-parallel (`for root_ix = part_id; root_ix < dd;
root_ix += parts_per_wg`), so the kernel went wide for the secular solve and then
collapsed to one column at a time for `2*dd` barrier-separated steps.

Both outer loops are embarrassingly parallel:

- **rescale**, index `eid`: reads row `eid` of `Q` (not written in this phase),
  writes only `v(eid)`. No cross-index dependency.
- **normalize**, index `eig`: reads all of `v`, reads and writes only column
  `eig` of `Q`. No cross-index dependency.

Fix applied: both are now templated on a reduction adapter plus a
`(first, stride)` index range.

- CTA variant passes `PartitionAdapter` with `(part_id, parts_per_wg)` — one
  index per sub-group partition, all running concurrently, **zero work-group
  barriers inside either loop**. A single `group_barrier(wg)` separates the two
  phases (rescale reads `Q` / writes `v`; normalize overwrites `Q` / reads `v`).
- WG variant passes `WorkgroupAdapter` with `(0, 1)` — bit-identical to the
  previous behaviour, retained as the reference path for A/B comparison.

Supporting changes: `reduce_product` and a no-op/barrier `sync()` added to both
adapters; `nrm2_column` added as an adapter-generic Blue's-scaled 2-norm
mirroring `internal::nrm2` in `src/math-helpers.hh` (the three accumulators are
reduced as scalars rather than a `sycl::vec<R,3>`, which is componentwise
identical).

**Numerical note.** Narrowing the reduction group reassociates both reductions.
This is safe here because neither involves cancellation: the Löwner reduction is
a *product* of ratios, and `nrm2` is a *sum of squares*. Only benign relative
error accumulates. This is materially different from the failed "fast path"
documented in `stedc_merge_kernels.cc:44-52`, which changed *where the poles were
stored* during the solve and produced a bimodal orthogonality distribution.
Regardless, orthogonality must be validated, not assumed — see §5.

### 2.2 The Löwner rescale does not need to read `Q` at all — **[open]**

`maybe_rescale_vectors` reads `Q_bid(eid, j)` with `j` strided across lanes. The
matrix is column-major (`include/blas/matrix.hh:75-80`,
`data_[b*stride + j*ld + i]`), so this is an `ld`-stride row walk — the only
uncoalesced access in the kernel.

It is also unnecessary. At that point `Q_bid(eid, j)` holds
`(d_eid - origin_j) - tau_j`, i.e. `d_eid - lambda_j`, and the denominator is
`d_eid - d_j`. Both operands are already in shared memory (`d_local`) or cheaply
available (`temp_lambdas`). Recomputing instead of reloading turns the entire
rescale into a shared-memory-only pass with **zero global traffic**. This is
exactly LAPACK's `dlaed3` formulation.

Follow-on: `write_denominator_column` then no longer needs to materialize
denominators to global `Q` purely so the next two phases can read them back.
Combined, per-merge `Q` traffic drops from roughly 3 reads + 2 writes per element
to a single write.

Caveat: recomputing changes rounding relative to the stored value (the stored
denominator has a zero-clamp applied). Needs the same orthogonality validation.

### 2.3 Re-tune work-group size — **[open]**

`params.secular_cta_wg_size_multiplier` / `choose_wg_size`. Widening the WG was
previously useless because the serial rescale/normalize tail meant a 256-thread
WG reduced over only `dd` elements, wasting most lanes. With §2.1 landed, more
partitions per WG now translate into real concurrency, so this should be
re-swept. **Do this after §2.1 and §2.2, not before.**

### 2.4 Back-transform GEMM ignores deflation and block structure — **[open]**

`src/extensions/stedc.cc:424` issues a full dense `n x n x n` GEMM per merge
level. LAPACK's `dlaed2`/`dlaed3` permute columns by `CTOT` into those touching
only block 1, only block 2, or both, and issue two smaller GEMMs — roughly a
**2x FLOP saving** — and deflated columns are skipped entirely (`dd =
n_reduced[bid] <= n`). The `perm_map` machinery already exists to make this
tractable. Only 4.9% of time at n=256, but it is the term that dominates as n
grows.

### 2.5 Eigenvalues-only still pays for eigenvectors — **[open]**

`jobz` is threaded through `stedc_impl` but between the recursion
(`stedc.cc:110-111`) and the GEMM (`:424`) there is no `NoEigenVectors` branch —
the merge appears to build and back-transform eigenvectors unconditionally.
Worth confirming; if true it is free money for that path, and see §3 for the
matching algorithm.

### 2.6 What is already good — leave alone

The secular root solver (`stedc_merge_cta.cc`, `solve_root_roc_generic`) is the
rocSOLVER middle-way / fixed-weight hybrid with bound-clamped steps and a proper
`|f| <= eps * err` convergence exit. That is the literature-recommended scheme
(Li's middle way; Gu–Eisenstat stability). Do not touch it.

---

## 3. Literature

- **Gu & Eisenstat, "A Stable and Efficient Algorithm for the Rank-One
  Modification of the Symmetric Eigenproblem", SIMAX 1994.**
  <https://epubs.siam.org/doi/10.1137/S089547989223924X>
  The foundation of the Löwner-based eigenvector formula used here: computes
  eigenvectors stably without extended precision. Confirms the current approach.

- **Jakovčević Stor, Slapničar & Barlow, "Forward stable eigenvalue decomposition
  of rank-one modifications of diagonal matrices", arXiv:1405.7537.**
  <https://arxiv.org/pdf/1405.7537>
  Computes each eigenvalue *and all eigenvector components* to high relative
  accuracy in O(n). Could let us drop the `enable_rescale` pass entirely rather
  than optimize it — strictly better than §2.2 if it holds up.

- **Liao, Li, Xia et al., "New fast divide-and-conquer algorithms for the
  symmetric tridiagonal eigenvalue problem", arXiv:1510.04591.**
  <https://arxiv.org/abs/1510.04591>
  The rank-one eigenvector matrix is Cauchy-like with off-diagonally low rank;
  HSS approximation gives O(N^2 r) instead of O(N^3), reported >6x over MKL on
  large matrices with few deflations. Relevant only to a large-n path, not to
  batched small n.

- **Zhan & Zhang, "Reducing Internal State in Eigenvalue-Only Divide-and-Conquer
  Tridiagonal Eigensolvers", arXiv:2605.26599 (May 2026).**
  <https://arxiv.org/pdf/2605.26599>
  Boundary-row D&C: the conquer phase needs only selected boundary rows/columns,
  not the accumulated eigenvector matrix. Quadratic to linear memory, and the
  matrix update disappears. Directly targets §2.5.

- **FMM-accelerated secular evaluation** (Gu–Eisenstat lineage). Evaluating the
  secular function at all roots simultaneously is a Cauchy matvec — O(dd log dd)
  instead of O(dd^2). Our per-iteration `evaluate_roc_secular` is O(dd) per root.
  Probably not worth it below dd ~ 512, but it is the asymptotic answer.

- **LAPACK reference:** [dlaed2](https://netlib.org/lapack/explore-3.2-html/dlaed2.f.html),
  [dlaed3](https://www.netlib.org/lapack/explore-3.2-html/dlaed3.f.html) — the
  `CTOT` block partitioning behind §2.4.

- **[rocSOLVER tuning guide](https://rocm.docs.amd.com/projects/rocSOLVER/en/develop/reference/tuning.html)**
  — our solver follows their STEDC scheme; useful for the minimum-block-size
  heuristics.

---

## 4. Prioritized plan

1. **[done]** Partition-parallelize rescale + normalize (removes `2*dd` barriers).
2. **[open]** Recompute denominators from `d_local`/`temp_lambdas` instead of
   round-tripping through `Q` (kills the uncoalesced access and most global
   traffic); then drop the `write_denominator_column` global store.
3. **[open]** Re-sweep `wg_size` / `secular_cta_wg_size_multiplier` — only
   meaningful after 1 and 2.
4. **[open]** Deflation- and block-aware GEMM at `stedc.cc:424`.
5. **[open]** `NoEigenVectors` short-circuit (+ boundary-row D&C from
   arXiv:2605.26599).

Items 1–3 are contained within `stedc_merge_cta.cc`; 4–5 touch the driver.
Given the 3.96% achieved occupancy, 1–3 should be the bulk of the merge-kernel
win.

---

## 5. Validation protocol

Any change in this area must be checked for **orthogonality**, not just
residuals. The comment at `src/extensions/stedc_merge_kernels.cc:44-52` records a
prior regression that passed residual checks while producing a bimodal
orthogonality distribution for float STEDC at n <= 64.

- `tests/stedc_tests.cc` — correctness gate.
- `benchmarks/orthogonality_accuracy.cc`, `benchmarks/eigensolver_accuracy.cc` —
  compare the *distribution*, not just the max, against the WG variant
  (`StedcMergeVariant::FusedWg` / the baseline 3-kernel path), which is
  deliberately left bit-identical as the reference.
- `benchmarks/stedc_benchmark.cc` — timing, plus re-profile with the `ncu`
  recipe in the skill reference to confirm occupancy actually moved.
