# STEDC: the divide-and-conquer merge {#perf_stedc}

**Covers:** the performance work on the `stedc` merge path: the Nsight Compute profile of the
fused CTA merge kernel, partition-parallel rescale and normalize, the work-group width retune,
the deflation-aware back-transform GEMM, what was tried and reverted, the open eigenvalues-only
item, the literature, and the validation protocol for any change here.
**Status:** mostly current; stale parts are marked inline. This page is the distilled form of the
August 2026 research log (formerly `STEDC_MERGE_OPTIMIZATION.md`).
**Machine:** RTX 4090 (sm_89), CUDA backend, GPU 1 (`CUDA_VISIBLE_DEVICES=1`).
**Measured:** 2026-08-01 (commits `296a5aba`, `2a65e165`, `6c2664a7`, `d4e057f1`); the
syev-side retune quoted in [stedc: current tuning values](#stedc-current-tuning-values) on
2026-08-07.

The mathematics of the leaf solver is in @ref algo_steqr. The Nsight Compute recipe for this
kernel is agent tooling kept at `.github/skills/nvidia-kernel-profiling/references/batchlas-stedc.md`.

## stedc: merge kernel profile

`stedc_benchmark --backend=CUDA --type=float 256 256 16 0 32 1` under Nsight Compute, before any
of the work below:

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

Share of total GPU time: `SteqrCTAKernel` 35.7%, `StedcFusedCtaMerge` 30.3%,
`Matrix::Identity` 13.9%, `PermutedCopyKernel` 5.4%, `ampere_sgemm_128x128_nn` 4.9%.

**The merge kernel is neither compute- nor bandwidth-bound; it is serialization/latency-bound.**
Achieved occupancy is about 1/12 of theoretical, which points at serialization inside the kernel
rather than at launch geometry alone.

## stedc: partition-parallel rescale and normalize

Status: **done**. End-to-end gain on the CTA path 4-9% on its own, depending on n (about 5.5% in
the matched-settings attribution, see
[stedc: end-to-end merge speedup](#stedc-end-to-end-merge-speedup)).

`maybe_rescale_vectors` and `normalize_vectors` (`src/extensions/stedc_merge_cta.cc`) each looped
`for eid = 0 .. dd-1` with a whole-work-group `reduce_over_group` plus a `group_barrier` per
index. The root solve just above them was already partition-parallel
(`for root_ix = part_id; root_ix < dd; root_ix += parts_per_wg`), so the kernel went wide for the
secular solve and then collapsed to one column at a time for `2*dd` barrier-separated steps.

Both loops are embarrassingly parallel:

- **rescale**, index `eid`: reads row `eid` of `Q` (not written in this phase), writes only
  `v(eid)`;
- **normalize**, index `eig`: reads all of `v`, reads and writes only column `eig` of `Q`.

Both are now templated on a reduction adapter and a `(first, stride)` index range:

- the CTA variant passes `PartitionAdapter` with `(part_id, parts_per_wg)` for both loops: one
  index per sub-group partition, all concurrent, no work-group barrier inside either loop. One
  `group_barrier(wg)` separates the phases (rescale reads `Q` and writes `v`; normalize overwrites
  `Q` and reads `v`);
- the WG variant passes `WorkgroupAdapter` with `(0, 1)`, bit-identical to the old behaviour and
  kept as the A/B reference path.

Supporting changes: `reduce_product` and a no-op/barrier `sync()` on both adapters, and
`nrm2_column`, an adapter-generic Blue's-scaled 2-norm mirroring `internal::nrm2` in
`src/math-helpers.hh` (three accumulators reduced as scalars rather than a `sycl::vec<R,3>`,
componentwise identical).

Two hazards created by narrowing the reduction group were fixed along the way and are kept:

- **Löwner product range.** A lane multiplies `dd/width` factors before any reduction, so going
  from width 32 to width 4 makes each serial product 8x longer, and it can leave float range. The
  product is accumulated as a (mantissa, exponent) pair with `frexp` renormalization. Rescaling is
  by exact powers of two, so wherever the naive product was in range the result is bit-for-bit the
  same.
- **Uniform trip counts.** `dd` is rarely a multiple of `parts_per_wg` (deflation makes it
  arbitrary). A plain `eid < dd` bound lets some partitions leave the loop while others are still
  in it, and the reductions are sub-group shuffles that need every lane of the warp. Both loops
  iterate a uniform number of times and mask the work.

**Numerical note.** Narrowing the reduction group reassociates both reductions. That is safe
because neither involves cancellation: the Löwner reduction is a product of ratios and `nrm2` a
sum of squares, so only benign relative error accumulates. This differs from the failed "fast
path" recorded in the fused merge kernel (`src/extensions/stedc_merge_kernels.cc`, the comment
above the root solve; the log cited it as lines 44-52, it has since moved), which changed where
the poles were stored during the solve and produced a bimodal orthogonality distribution.
Orthogonality must still be validated, not assumed; see
[stedc: validation protocol for merge changes](#stedc-validation-protocol-for-merge-changes).

## stedc: the dd == 1 secular-solve fault

Status: **fixed**. Making `normalize_vectors` partition-parallel surfaced
`CUDA_ERROR_ILLEGAL_ADDRESS` in the merge kernel. The cause was pre-existing.

`solve_root_ext_generic` computed `prev = dd - 2` and dereferenced `d_prob(prev)`
unconditionally. When a merge subproblem deflates down to `dd == 1`, `prev` is `-1`. `d_prob`
aliases the shared-memory `d_local` through a generic pointer, so the negative offset resolves
outside the shared window and faults instead of reading garbage. The work-group path made the
same out-of-range access (confirmed by device assert) but happened not to fault under its
register and shared-memory layout; the partition version changed the layout enough to make it
fatal.

Fix: solve the single-pole case in closed form. With one pole the secular equation
\f$1/\rho + z_0^2/(d_0 - x) = 0\f$ gives \f$x = d_0 + \rho z_0^2\f$, so the CTA solver returns
`{d_prob(0), rho * z0 * z0}` before computing `prev` (`stedc_merge_cta.cc`, `solve_root_ext_generic`).
The non-CTA `sec_solve_ext_roc` in `src/extensions/stedc_secular.cc` had the identical bug
(`prev_index = dd - 2`) and is fixed the same way. That path updates `D` in place, so the guard
also writes the denominator \f$D_0 - x = -\rho z_0^2\f$, matching what the CTA path's
`(d - origin) - tau` produces and what `apply_shift_to_poles` leaves in the general case.

**How it was found.** `compute-sanitizer` cannot attach ("CUDA initialized before the
Sanitizer"), because SYCL initializes CUDA first. The bounds checks in
`KernelMatrixView`/`VectorView::at` are compiled out by `NDEBUG`, so the tool that worked was a
rebuild with device asserts on:

```bash
cmake -B build-assert -S . -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -UNDEBUG" ...
```

CMake puts `-DNDEBUG` in `$DEFINES`, and the rule is `$DEFINES $INCLUDES $FLAGS`, so `-UNDEBUG`
in the flags wins. The device assert then names the function and the index.

### stedc: the heavy-deflation test gap

Status: **fixed, but the guard is weak**. The gap was not partition width:
`StedcTest.FusedCtaPartitionWidths` already covered \f$P = 4, 8, 16, 32\f$ at n = 64. It was the
deflation regime: plain random tridiagonals essentially never deflate a subproblem to `dd == 1`.
`StedcTest.FusedCtaConditionedHeavyDeflation` (`tests/stedc_tests.cc`) builds matrices with
`random_hermitian_tridiagonal_with_log10_cond_metric` at log10 cond 1, 3 and 5 and runs all four
widths, mirroring the `stedc_acc` case that exposed the bug. It reproduced
`CUDA_ERROR_ILLEGAL_ADDRESS` with the fix reverted and passed with it.

**Trap (still open):** the test asserts only that eigenvalues are finite and sorted, never
accuracy or orthogonality. "The tests pass" is weaker evidence than it looks, for this fix and for
the merge-variant choice below. Its comment also says the tuning tables select
`secular_threads_per_root = 4` for n <= 64; they now select 8 (`STEDC_THREADS_PER_ROOT_*` in
`include/batchlas/tuning_params.hh`).

## stedc: recomputing denominators costs occupancy

Status: **tried, reverted (regression)**.

The idea: `maybe_rescale_vectors` read `Q_bid(eid, j)` with `j` strided across lanes, an
`ld`-stride row walk of a column-major matrix and the only uncoalesced access in the kernel. That
value is exactly `secular_denominator(d_eid, origin_j, tau_j)`, so it can be recomputed from shared
memory instead. That also removes `write_denominator_column`'s global store and folds the 2-norm
into the write pass. It was implemented, verified bit-identical, and is 5-10% slower
(batch 256, float, ms):

| n | partition-parallel only | with recompute | control: partition-parallel + 2 unused shared arrays |
| --- | --- | --- | --- |
| 64 | 0.840 | 0.981 | 0.990 |
| 256 | 4.698 | 5.112 | 5.190 |
| 512 | 12.704 | 13.807 | 14.105 |

The control column decides it: the same code with the two extra `local_accessor`s allocated and
written but never read is just as slow. The whole regression is shared-memory footprint; the
recompute itself is worth about 1-2% (it beats the control at every size) but cannot pay for the
occupancy. Storing `origin` and `tau` per root doubles shared usage from
\f$2 \cdot n_{loc} \cdot \mathrm{sizeof}(T)\f$ to \f$4 \cdot n_{loc} \cdot \mathrm{sizeof}(T)\f$,
which at 32 threads per work-group halves the blocks resident per SM.

**The lesson generalizes:** this kernel runs at 0.14% DRAM throughput. Removing global traffic
buys nothing, while anything that grows shared memory costs occupancy, the binding constraint.
Optimizations here should reduce resident state, not trade memory for arithmetic. If revisited,
add no shared memory: keep `tau` plus a 1-byte origin-pole selector, or stage the shifts in global
scratch (coalesced, L1-resident, no occupancy cost). Expected upside is the 1-2% above.

## stedc: the work-group multiplier after the barriers went

Status: **done**; the values have since been re-chosen through `syev`
(see [stedc: current tuning values](#stedc-current-tuning-values)).

`STEDC_WG_MULTIPLIER_*` went from 1 / 1 / 2 / 2 / 4 to 8 for every size class. This was the
largest single win, and it exists only because the partition-parallel loops removed the barriers
(n = 64, batch 256, float, ms):

| multiplier | baseline (serial rescale/normalize) | partition-parallel |
| --- | --- | --- |
| 1 | 0.919 | 0.880 |
| 4 | 0.850 | 0.740 |
| 8 | 0.877 (worse than 4) | 0.729 |

On the baseline, widening past 4 hurts: a wide work-group reducing over only `dd` elements wastes
most of its lanes. With partition-parallel loops the extra width becomes extra concurrent
columns. 16 was also tried: marginally better at n = 64, worse at n >= 256.

**Trap fixed here: launch ceiling.** `choose_wg_size` clamped only against the device's
`max_work_group_size` (1024). At about 80 registers per work-item, 1024 work-items do not launch,
and the call throws:

```
Exceeded the number of registers available on the hardware.
The kernel uses 80 registers per work-item for a total of 1024 work-items.
```

It now also clamps to `kernel_device_specific::work_group_size` for the specific kernel
(`kernel_max_work_group_size` in `stedc_merge_cta.cc`), so an aggressive multiplier degrades to the
largest launchable size. The log explained the failure as 81,920 registers against a 65,536 file;
the correct bound is per sub-partition (16,384 each), which caps the kernel at 6 warps per
sub-partition, 768 work-items, as the comment above `choose_wg_size` records.

**Benchmark trap.** `stedc_benchmark` used to substitute fixed fallbacks
(`threads_per_root = 32`, `wg_multiplier = 1`) for 0 arguments, so it measured a configuration
the library never runs by default. It now passes non-positive values through and `StedcParams`
resolves them from the tuning tables, so `stedc_benchmark ... <n> 256 16 2 0 0` measures what a
real caller such as `syev` gets. Also: **argument 3 selects the merge variant** (`-1` Auto,
`0` Baseline, `1` Fused, `2` FusedCta). An early measurement used `0`, the Baseline 3-kernel path,
which does not run this code at all, and produced a spurious result.

## stedc: deflation-aware back-transform GEMM

Status: **done**, gated at \f$n \ge 512\f$ (`stedc_deflation_gemm_min_n`,
`src/extensions/stedc.cc`).

`stedc.cc` issued a full dense \f$n \times n \times n\f$ GEMM per merge level. `Qprime` is
identity-filled and only its first `n_reduced` columns carry secular eigenvectors, so as a block
it is \f$M = [W \mid I]\f$. Permuting columns commutes with the product:

\f[
A \, M_{:,\pi} = (A M)_{:,\pi},
\qquad
A M = [\, A W \mid A_{:,\,dd:} \,],
\f]

so the sort can be applied after the multiply, and the deflated columns of the product are
columns of \f$A\f$ that need no multiply. Only the first \f$dd\f$ columns need a GEMM.

**Kept on cuBLAS.** `dd` varies per batch item, and a per-item GEMM would be ragged, dropping off
the vendor batched kernel onto the heterogeneous path. A single batch-wide \f$dd_{\max}\f$ keeps
one uniform batched call and is still exact: for an item with \f$dd < dd_{\max}\f$, columns
\f$dd .. dd_{\max}-1\f$ of \f$M\f$ really are identity columns. No extra workspace: \f$A\f$ goes
into `temp_Q`, which frees `eigvects` for the narrow GEMM result, folded back over \f$A\f$'s head.

How much deflates (random tridiagonals, batch 64):

| merge size | mean `dd` | kept | min-max |
| --- | --- | --- | --- |
| 256 | 43.8 | 17.1% | 25-61 |
| 128 | 43.9 | 34.3% | 19-69 |
| 64 | 41.3 | 64.5% | 21-57 |
| 32 | 29.0 | 90.6% | 10-32 |

Deflation is heaviest where the GEMM is most expensive, and the batch is homogeneous enough that
\f$dd_{\max}\f$ is close to the mean.

### stedc: the deflation GEMM threshold

The catch is a host sync: \f$dd_{\max}\f$ must be known on the host to size the GEMM, which
stalls the enqueue pipeline. Feature off vs on, interleaved, idle GPU, batch 256, float, ms:

| n | off | on | gain |
| --- | --- | --- | --- |
| 256 | 3.998 | 4.020 | -0.5% (noise) |
| 512 | 11.166 | 10.525 | +5.8% |
| 1024 | 39.388 | 29.937 | +24.0% |

Hence the gate at n >= 512. Below it the recursion has many small merge nodes and the syncs cost
more than the saved flops; a threshold of 128 regressed n = 256 by about 5%. A second guard,
`stedc_deflation_gemm_max_kept_fraction = 0.75`, skips the narrow path unless deflation removed
at least 25% of the columns, so weak deflation cannot lose more than the one sync.

**Not done, deliberately:** LAPACK's `dlaed2`/`dlaed3` `CTOT` split of the non-deflated columns
into two blocks (a further ~2x). It would split the one uniform GEMM into two with per-item widths,
the ragged shape that leaves the vendor kernel.

## stedc: eigenvalues-only still builds eigenvectors

Status: **open inside `stedc`; avoided by `syev`**.

`stedc` has no `NoEigenVectors` branch between the recursion and the back-transform GEMM: the
merges consume the leaves' eigenvectors, so every merge and back-transform runs whatever `jobz`
says. In `src/extensions/stedc.cc` the level driver (`stedc_levels_impl`) never reads `jobz`
(its workspace sizing discards it with `(void)jobz`), and the recursive driver (`stedc_impl`)
only forwards it to the leaf `steqr_dispatch`. The `syev` tiers
work around it rather than fix it: `syev_blocked` and `syev_two_stage` call `stebz` (bisection)
for values-only and always call `stedc` with `EigenVectors` (see the NOTE in
`src/extensions/syev_blocked.cc`). A direct `stedc(..., NoEigenVectors, ...)` caller still pays
for vectors. **Unverified hazard:** because the recursive driver passes the caller's `jobz` to the
leaf solve while every merge consumes leaf eigenvectors, a direct
`stedc(..., NoEigenVectors, ...)` on the recursive driver may merge from vectors that were never
computed; this needs a test before it is filed as a defect. The matching algorithm is boundary-row divide and conquer
([stedc: literature on the merge](#stedc-literature-on-the-merge), Zhan & Zhang).

## stedc: what is already good

The secular root solver (`solve_root_roc_generic` in `stedc_merge_cta.cc`) is the rocSOLVER
middle-way / fixed-weight hybrid with bound-clamped steps and a proper
\f$|f| \le \varepsilon \cdot \mathrm{err}\f$ convergence exit: the literature-recommended scheme
(Li's middle way, Gu-Eisenstat stability). Leave it alone.

## stedc: literature on the merge

- **Gu & Eisenstat, "A Stable and Efficient Algorithm for the Rank-One Modification of the
  Symmetric Eigenproblem", SIMAX 1994.** <https://epubs.siam.org/doi/10.1137/S089547989223924X>.
  The Löwner-based eigenvector formula used here; stable eigenvectors without extended precision.
- **Jakovčević Stor, Slapničar & Barlow, "Forward stable eigenvalue decomposition of rank-one
  modifications of diagonal matrices", arXiv:1405.7537.** <https://arxiv.org/pdf/1405.7537>.
  Every eigenvalue and every eigenvector component to high relative accuracy in O(n). Could remove
  the `enable_rescale` pass entirely, which would beat the reverted denominator recompute if it
  holds up. Not tried.
- **Liao, Li, Xia et al., "New fast divide-and-conquer algorithms for the symmetric tridiagonal
  eigenvalue problem", arXiv:1510.04591.** <https://arxiv.org/abs/1510.04591>. The rank-one
  eigenvector matrix is Cauchy-like with off-diagonally low rank; HSS gives
  \f$O(N^2 r)\f$ instead of \f$O(N^3)\f$, reported >6x over MKL on large matrices with few
  deflations. Relevant to a large-n path only, not batched small n.
- **Zhan & Zhang, "Reducing Internal State in Eigenvalue-Only Divide-and-Conquer Tridiagonal
  Eigensolvers", arXiv:2605.26599 (May 2026).** <https://arxiv.org/pdf/2605.26599>. Boundary-row
  D&C: the conquer phase needs only selected boundary rows/columns, not the accumulated
  eigenvector matrix; memory goes from quadratic to linear and the matrix update disappears.
  Targets the open eigenvalues-only item.
- **FMM-accelerated secular evaluation** (Gu-Eisenstat lineage). Evaluating the secular function
  at all roots at once is a Cauchy matvec, \f$O(dd \log dd)\f$ instead of \f$O(dd^2)\f$; our
  per-iteration `evaluate_roc_secular` is \f$O(dd)\f$ per root. Probably not worth it below
  \f$dd \approx 512\f$.
- **LAPACK** [dlaed2](https://netlib.org/lapack/explore-3.2-html/dlaed2.f.html) and
  [dlaed3](https://www.netlib.org/lapack/explore-3.2-html/dlaed3.f.html): the `CTOT` block
  partitioning discussed under the deflation GEMM.
- **[rocSOLVER tuning guide](https://rocm.docs.amd.com/projects/rocSOLVER/en/develop/reference/tuning.html)**:
  our solver follows their STEDC scheme; useful for minimum-block-size heuristics.

## stedc: where the remaining headroom is

As of the log (2026-08-01): the partition-parallel loops and the multiplier addressed the merge
kernel's serialization and occupancy, and the deflation GEMM addressed the back-transform. The
recompute control experiment shows the merge kernel is limited by resident state, not memory
traffic, so further merge work should cut registers or shared memory rather than restructure data
flow. The largest remaining item is eigenvalues-only.

**Superseded:** the log closed with "`SteqrCTAKernel` is now the biggest single kernel at ~25%
of GPU time and has not been looked at". It has since been: the lockstep series in
[STEQR: the CTA tridiagonal solver](../perf/steqr.md) (flat solver, full-warp partition, tuned
multiplier) measured `stedc` float n = 64 with \f$P = 32\f$ leaves at 1.14 (random) / 1.09 (graded)
over the previous `main`; see [Cumulative result](../perf/steqr.md#cumulative-result).

## stedc: validation protocol for merge changes

Check **orthogonality**, not just residuals. The comment on the fused merge kernel's root solve
(`src/extensions/stedc_merge_kernels.cc`) records a regression that passed residual checks while
producing a bimodal orthogonality distribution for float STEDC at n <= 64.

- `tests/stedc_tests.cc`: the correctness gate (with the weak heavy-deflation guard noted above).
- `benchmarks/orthogonality_accuracy.cc`, `benchmarks/eigensolver_accuracy.cc`: compare the
  *distribution*, not only the max, against the work-group reference, deliberately kept
  bit-identical: `stedc_merge_fused_wg` (the `WorkgroupAdapter` kernel `StedcFusedWgMerge`,
  which `FusedCta` falls back to when `secular_threads_per_root` exceeds the sub-group size) or
  the `Baseline` 3-kernel path. (The log called it `StedcMergeVariant::FusedWg`; no such
  enumerator exists, the enum is `Auto`, `Baseline`, `Fused`, `FusedCta`.)
- `benchmarks/stedc_benchmark.cc`: timing; re-profile with the `ncu` recipe in the skill
  reference to confirm occupancy moved.

**Measurement hygiene.** The box has two RTX 4090s and other jobs may be running; a contended GPU
inflated an early reading of this work from 5.5% to 8.6%. Pin to an idle device
(`CUDA_VISIBLE_DEVICES=1`) and check
`nvidia-smi --query-compute-apps=pid,process_name --format=csv` before trusting a number.

## stedc: merge correctness results

`stedc_acc --backend=CUDA --samples=64`, float and double, n = 16..512: 0.00000 Fail% everywhere,
residual and orthogonality at machine-epsilon level. Results were bit-identical across the
partition-parallel loops, the denominator recompute and the multiplier change: the refactors are
numerically exact, and the work-group width does not enter the per-column math because the
reductions are partition-local.

Orthogonality \f$\|Z^T Z - I\|_F / n\f$, float, `--samples=256`:

| n | baseline | with the change |
| --- | --- | --- |
| 16 | 1.21899e-07 | 1.21899e-07 |
| 32 | 9.36432e-08 | 9.38560e-08 |
| 64 | 7.41322e-08 | 7.44386e-08 |

`stedc_tests` CUDA 44 passed; `syev_tests` 8/8, `syev_cta_tests` 28/28, `syev_blocked_tests`
32/32. Pre-existing failures at the time, reproduced identically with the changes reverted:
8 `FlatTraceCollapsesByDepth[Ragged]` in `stedc_tests` and 4 `steqr_tests` failures on the host
(CPU) backend. **Historical:** `FlatTraceCollapsesByDepth` no longer exists in
`tests/stedc_tests.cc`; the current accepted failures are in `tests/known-failures.txt`.

## stedc: end-to-end merge speedup

End-to-end `stedc`, batch 256, float, idle GPU, with tuning-resolved parameters as a real caller
gets them (`stedc_benchmark ... <n> 256 16 2 0 0`), ms:

| n | baseline (main) | this work | speedup |
| --- | --- | --- | --- |
| 64 | 0.930 | 0.782 | 15.9% |
| 128 | 2.042 | 1.633 | 20.0% |
| 256 | 4.885 | 3.967 | 18.8% |
| 512 | 13.144 | 11.162 | 15.1% |

Attribution at matched settings (multiplier 1, so partition-parallel loops alone): 0.933 to 0.882
ms at n = 64 and 4.948 to 4.676 ms at n = 256, about 5.5%. The rest is the multiplier change,
which the loops made possible. The deflation GEMM only engages at n >= 512.

## stedc: current tuning values

The live values are in `include/batchlas/tuning_params.hh`; read them there, not here.
At the time of writing:

- `STEDC_MERGE_VARIANT_*` = 2 (FusedCta) for every size class. A commit (`3072ea6`) had moved it
  to 1 (Fused), claiming FusedCta was wrong and 2-12% slower. Re-tested at `0bb92fb`, neither
  held: with variant 2 forced all 16 CUDA `stedc_tests` passed in both precisions, plus
  `syev_tests` 8/8, `syev_blocked_tests` 44/44 and `syev_two_stage_tests` 20/20 (the one red
  test on the dev box, `StedcTest/1.BatchedMatrices`, is double + NETLIB, i.e. the host OpenBLAS
  dgemm, not `stedc`). FusedCta was 11-35% faster than Fused on `stedc` across n = 64..640 on
  both drivers; variant 1 was the whole cause of a 1.24x-1.37x `syev` regression at power-of-two
  n. The NaN seen under FusedCta (in `FusedCtaConditionedHeavyDeflation`) was a symptom of a
  deadlock in the same kernel, since fixed, not a separate defect in the partition root solve.
  The heavy-deflation test caveat above applies to this verdict: strengthen that test before
  trusting variant 2 on hardware it has not been measured on.
- `STEDC_THREADS_PER_ROOT_*` = 8 and `STEDC_WG_MULTIPLIER_*` = 8, chosen through `syev`, not
  through the `stedc` benchmark, which disagrees with its consumer. The 2026-08-07 sweep picked
  multiplier 2-4 and threads-per-root 4-8 on `stedc` alone; through `syev` with everything else
  fixed (ms):

| syev n | wgm=2 | wgm=4 | wgm=8 | tpr=4 | tpr=8 | tpr=16 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.9987 | 0.9225 | 0.9227 | 0.9937 | 0.9229 | 0.9422 |
| 128 | 4.626 | 4.535 | 4.514 | 4.518 | 4.469 | 4.534 |
| 256 | 23.067 | 22.110 | 21.383 | 21.557 | 21.332 | 21.330 |
| 512 | 151.54 | 151.46 | 151.84 | 151.63 | 151.62 | 151.67 |
| 1024 | 903.40 | 900.12 | 903.01 | 902.35 | 904.74 | 903.35 |

Adopting `stedc`'s own winners cost `syev` 2.7% at n = 256. The `stedc` benchmark measures the
merge in isolation, while `syev` pays for the whole tridiagonal solve. Below n = 512 the
difference is real (up to 7.6% at n = 64); at n >= 512 every value is within noise.

- `STEDC_RECURSION_THRESHOLD_*` = 32: the CTA leaf-size invariant (a leaf of 33+ falls from
  `steqr_cta` to `steqr_wg`), not a tuning constant.
