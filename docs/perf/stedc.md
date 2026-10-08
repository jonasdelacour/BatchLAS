# STEDC: the divide-and-conquer merge {#perf_stedc}

> **Status:** mostly current · RTX 4090 (sm_89), CUDA, GPU 1 · measured 2026-08-01; `syev` tuning
> re-measured 2026-08-07

This page covers the performance work on the `stedc` merge path: the merge-kernel profile, the
partition-parallel rescale and normalize, the work-group width, the deflation-aware back-transform
GEMM, rejected alternatives, open debts, and the validation protocol. The leaf-solver mathematics is
in @ref algo_steqr; the live tuning values are in `include/batchlas/tuning_params.hh`. Each section's
status line marks what is stale.

## stedc: merge kernel profile

Nsight Compute, `stedc_benchmark --backend=CUDA --type=float 256 256 16 0 32 1`, before the changes
below (`StedcFusedCtaMerge<CUDA, float, 32>`): 101.5 us, 8.8% SM throughput, 0.14% DRAM throughput,
78 registers, 3.96% achieved occupancy against 50% theoretical. The merge is serialization- and
latency-bound, not compute- or bandwidth-bound.

## stedc: partition-parallel rescale and normalize

Status: **done**. End-to-end gain on the CTA path is 4-9%, depending on n (about 5.5% at matched
settings, see [stedc: end-to-end merge speedup](#stedc-end-to-end-merge-speedup)).

`maybe_rescale_vectors` and `normalize_vectors` (`src/extensions/stedc_merge_cta.cc`) looped over
`eid = 0 .. dd-1` with a whole-work-group reduction and a barrier per index. Both loops are
independent per index, so the CTA variant runs one index per sub-group partition, concurrently, with
one `group_barrier(wg)` between the phases. The WG variant (`WorkgroupAdapter`, `(0, 1)`) is
bit-identical to the old code and is kept as the A/B reference.

> **Warning:** Narrowing the reduction group makes each serial Löwner product 8x longer going from
> width 32 to width 4. At n = 64 the naive product overflowed or underflowed and poisoned the
> eigenvalues. The product is accumulated as a (mantissa, exponent) pair with `frexp` renormalization.

> **Warning:** `dd` is rarely a multiple of `parts_per_wg`. A plain `eid < dd` bound lets some
> partitions leave early, while the sub-group shuffles need every lane. Both loops run a uniform trip
> count and mask the work.

Narrowing the reduction reassociates both sums. That is safe because neither involves cancellation,
but orthogonality must still be validated
([stedc: validation protocol for merge changes](#stedc-validation-protocol-for-merge-changes)).

### Rejected: recomputed denominators

Recomputing the rescale denominators from shared memory removes a global store but adds two shared
arrays, which cut occupancy. Batch 256, float, ms:

| n | partition-parallel | with recompute | control: + 2 unused shared arrays |
| --- | --- | --- | --- |
| 64 | 0.840 | 0.981 | 0.990 |
| 256 | 4.698 | 5.112 | 5.190 |
| 512 | 12.704 | 13.807 | 14.105 |

The regression is shared-memory footprint, not the recompute. Any revisit must add no shared memory:
keep `tau` plus a 1-byte origin selector, or stage the shifts in global scratch.

## stedc: the dd == 1 secular-solve fault

Status: **fixed**. Making `normalize_vectors` partition-parallel exposed `CUDA_ERROR_ILLEGAL_ADDRESS`
in the merge kernel; the cause predates the change. `solve_root_ext_generic` computed `prev = dd - 2`
and dereferenced `d_prob(prev)` unconditionally. When a subproblem deflates to `dd == 1`, `prev` is
-1, outside the shared window.

Fix: solve the single-pole case in closed form. With one pole, \f$x = d_0 + \rho z_0^2\f$, so the CTA
solver returns `{d_prob(0), rho * z0 * z0}` before computing `prev`. The non-CTA `sec_solve_ext_roc`
(`src/extensions/stedc_secular.cc`) gets the same guard.

To find such faults, `compute-sanitizer` cannot attach (SYCL initializes CUDA first). Rebuild with
device asserts on instead:

```bash
cmake -B build-assert -S . -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -UNDEBUG" ...
```

`-UNDEBUG` in the flags overrides the `-DNDEBUG` in `$DEFINES`.

### stedc: the heavy-deflation test gap

Status: **fixed; the guard is weak**. Random tridiagonals almost never deflate a subproblem to
`dd == 1`. `StedcTest.FusedCtaConditionedHeavyDeflation` (`tests/stedc_tests.cc`) uses
log10 condition 1, 3 and 5 across all four widths, and reproduced the fault with the fix reverted.

> **Warning:** The test asserts only that eigenvalues are finite and sorted, never accuracy or
> orthogonality. Strengthen it before trusting the merge-variant choice on new hardware.

## stedc: the work-group multiplier after the barriers went

Status: **done**; the values were later re-chosen through `syev` (see
[stedc: current tuning values](#stedc-current-tuning-values)).

`STEDC_WG_MULTIPLIER_*` went from 1 / 1 / 2 / 2 / 4 to 8 for every size class. This was the largest
single win, and it exists only because the partition-parallel loops removed the barriers. n = 64,
batch 256, float, ms: baseline 0.919 / 0.877 at multiplier 1 / 8, partition-parallel 0.880 / 0.729 at
multiplier 1 / 8 (multiplier 4: 0.740). On the baseline, widening past 4 hurts, because a wide group
reducing over `dd` elements wastes lanes; with partition-parallel loops the extra width becomes extra
concurrent columns.

> **Warning:** `choose_wg_size` clamped only to the device's `max_work_group_size` (1024). At about 80
> registers per work-item, 1024 work-items do not launch and the call throws. It now also clamps to
> `kernel_device_specific::work_group_size` (`kernel_max_work_group_size` in `stedc_merge_cta.cc`),
> which caps the kernel at 768 work-items (6 warps per sub-partition).

> **Note:** `stedc_benchmark` passes non-positive arguments through, and `StedcParams` resolves them
> from the tuning tables. Argument 3 selects the merge variant: `-1` Auto, `0` Baseline, `1` Fused,
> `2` FusedCta. Variant 0 runs the 3-kernel path, which does not exercise this code.

## stedc: deflation-aware back-transform GEMM

Status: **done**, gated at \f$n \ge 512\f$ (`stedc_deflation_gemm_min_n`, `src/extensions/stedc.cc`).

Each merge level issued a dense \f$n \times n \times n\f$ GEMM. `Qprime` is identity beyond its first
`n_reduced` columns, so as a block it is \f$M = [W \mid I]\f$, and
\f$A M = [\, A W \mid A_{:,\,dd:} \,]\f$. Only the first \f$dd\f$ columns need a GEMM; the sort is applied
after the multiply.

- **Kept on cuBLAS.** `dd` varies per item. A batch-wide \f$dd_{\max}\f$ keeps one uniform call and stays
  exact: for an item with \f$dd < dd_{\max}\f$, the extra columns of \f$M\f$ are identity columns.
- **No extra workspace.** \f$A\f$ goes into `temp_Q`, which frees `eigvects` for the narrow result.

Deflation on random tridiagonals, batch 64: merge size 256 keeps 17.1% of the columns (mean `dd`
43.8); merge size 32 keeps 90.6% (mean `dd` 29.0).

### stedc: the deflation GEMM threshold

The catch is a host sync to learn \f$dd_{\max}\f$, which stalls enqueue. Feature off vs on, idle GPU,
batch 256, float, ms:

| n | off | on | gain |
|---|---|---|---|
| 256 | 3.998 | 4.020 | -0.5% (noise) |
| 512 | 11.166 | 10.525 | +5.8% |
| 1024 | 39.388 | 29.937 | +24.0% |

The gate is n >= 512; a threshold of 128 regressed n = 256 by about 5%. A second guard,
`stedc_deflation_gemm_max_kept_fraction = 0.75`, skips the narrow path unless deflation removed at
least 25% of the columns. LAPACK's `dlaed2`/`dlaed3` `CTOT` split would gain a further ~2x, but it
splits the uniform GEMM into ragged calls and leaves the vendor kernel; not done.

## stedc: eigenvalues-only still builds eigenvectors

Status: **open inside `stedc`; avoided by `syev`**.

`stedc` has no `NoEigenVectors` branch. Merges consume the leaves' eigenvectors, so every merge and
back-transform runs whatever `jobz` says. The level driver (`stedc_levels_impl`) never reads `jobz`.
The recursive driver (`stedc_impl`) forwards it only to the leaf `steqr_dispatch`. `syev_blocked` and
`syev_two_stage` avoid this: they use `stebz` (bisection) for values-only.

> **Warning:** Unverified hazard. A direct `stedc(..., NoEigenVectors, ...)` on the recursive driver
> may merge from vectors that were never computed. This needs a test before it is filed as a defect.

The matching algorithm is boundary-row divide and conquer
([stedc: literature on the merge](#stedc-literature-on-the-merge), Zhan & Zhang).

## stedc: the level-synchronous driver

`src/extensions/stedc.cc` has two drivers over one size-uniform merge that combines P independent
subproblems of size s pairwise.

- **Recursive driver:** P = batch, one tree node at a time, depth-first. The \f$2^l\f$ sibling merges at
  level l run one after another, each with only `batch` work-groups. Cost is \f$O(2^L)\f$ launches.
- **Level-synchronous driver:** all nodes of a level are merged by one launch over `nodes * batch`
  work-groups, so launches drop to \f$O(L)\f$. To keep levels size-uniform, n is padded to
  \f$N = \text{leaf}\cdot 2^L\f$ with a diagonal tail above the Gershgorin bound; padded eigenvalues sort
  last. Padding costs \f$(N/n)^3\f$ in the top-level GEMM, so the planner prefers the tree that pads least,
  subject to the leaf cap.

Layout choices (RTX 4090): merges write into their parent's sub-blocks (one extra launch per level,
about 2/3 less assembly traffic, up to 5% faster than copying). Leaves are copied, not written in
place: doubling the leaf leading dimension made the CTA STEQR kernel 10% slower at n = 64.

## stedc: the leaf cap at the sub-group width

`plan_stedc_levels` (`src/extensions/stedc_levels_plan.hh`) never lets a leaf exceed the device
sub-group width. `steqr` uses `steqr_cta` only up to that width and falls to `steqr_wg` above it,
about 14x slower one step past the edge (batch 10,416, eigenvectors: n = 32 0.26 us, n = 36 3.76 us,
n = 40 4.87 us).

Weighting leaf width against padding instead regressed `syev` at n = 320 and 640 (3.25x and 1.51x),
because the padding term dominates and a zero-padding wide leaf always won. History and threshold
sweep: [syev: the stedc merge-variant and leaf-cliff regression](syev.md#syev-the-stedc-merge-variant-and-leaf-cliff-regression).

## stedc: the absolute deflation tolerance

Deflation uses LAPACK's absolute tolerance \f$\mathrm{tol} = 8\varepsilon\max(\|D\|_\infty, \|z\|_\infty)\f$
for both small-\f$|z|\f$ and eigenvalue-proximity deflation. The earlier relative tolerance
\f$64\varepsilon\max(1, |D_j|, |D_{j+1}|)\f$ under-deflated clustered small eigenvalues and produced
near-parallel eigenvectors: good residuals, bad orthogonality, bimodal orthogonality distribution.
After the fix, native-T accumulation matched double to the reported digits (n = 16 to 256).

## stedc: the rejected private-pole fast path

The fused merge kernel (`src/extensions/stedc_merge_kernels.cc`) initializes each column of `Q_bid`
with the poles and runs the secular solver in place. A variant that kept the poles in a private
`T d_priv[128]` array and copied back after the solve gave a bimodal orthogonality distribution for
float n <= 64. The in-place form matches the baseline exactly.

## stedc: convergence reporting through info

Every secular root solver reports whether its iteration met tolerance. Only the tolerance exit counts
as convergence; a root that hit the iteration cap used to return a silently wrong eigenvalue.

- The flag is an out-parameter with no default, so a new call site cannot drop the status silently.
- Reports from several threads of one item merge through an atomic `fetch_max`.
- The public `stedc` clears `info` once. Callers that run `stedc` more than once on the same items
  (for example `gesvd`'s two tridiagonal solves) must use the no-clear entry point.

The device-versus-host choice between secular routines tests `is_gpu`, not `B != Backend::NETLIB`:
the host runtime cannot safely invoke the ROCm-style root routines inside SYCL kernels.

## stedc: open debts

- **`max_sec_iter` does not reach one exit arm.** In `stedc_secular.cc` the tolerance/budget split
  uses a hardcoded 100 on one arm.
- **Direct eigenvalues-only calls** still build eigenvectors
  ([eigenvalues-only still builds eigenvectors](#stedc-eigenvalues-only-still-builds-eigenvectors)).
- **The heavy-deflation test** asserts only finite and sorted
  ([the heavy-deflation test gap](#stedc-the-heavy-deflation-test-gap)).

## stedc: literature on the merge

The secular root solver (`solve_root_roc_generic` in `stedc_merge_cta.cc`) is the rocSOLVER middle-way
/ fixed-weight hybrid with bound-clamped steps and a \f$|f| \le \varepsilon \cdot \mathrm{err}\f$ exit.
It follows the literature-recommended scheme; leave it alone.

- **Gu & Eisenstat, "A Stable and Efficient Algorithm for the Rank-One Modification of the Symmetric
  Eigenproblem", SIMAX 1994.** <https://epubs.siam.org/doi/10.1137/S089547989223924X>. The
  Löwner-based eigenvector formula used here.
- **Jakovčević Stor, Slapničar & Barlow, arXiv:1405.7537.** <https://arxiv.org/pdf/1405.7537>. Relative
  accuracy for every eigenvalue and eigenvector component in O(n). Could remove the `enable_rescale`
  pass. Not tried.
- **Liao, Li, Xia et al., arXiv:1510.04591.** <https://arxiv.org/abs/1510.04591>. HSS compression of the
  Cauchy-like eigenvector matrix, \f$O(N^2 r)\f$ instead of \f$O(N^3)\f$. Relevant to large n only.
- **Zhan & Zhang, "Reducing Internal State in Eigenvalue-Only Divide-and-Conquer Tridiagonal
  Eigensolvers", arXiv:2605.26599.** <https://arxiv.org/pdf/2605.26599>. Boundary-row D&C: memory drops
  from quadratic to linear and the matrix update disappears. Targets the open eigenvalues-only item.
- **FMM-accelerated secular evaluation.** A Cauchy matvec, \f$O(dd \log dd)\f$ instead of
  \f$O(dd^2)\f$; probably not worth it below \f$dd \approx 512\f$.
- **LAPACK** [dlaed2](https://netlib.org/lapack/explore-3.2-html/dlaed2.f.html) and
  [dlaed3](https://www.netlib.org/lapack/explore-3.2-html/dlaed3.f.html): the `CTOT` partitioning.
- **[rocSOLVER tuning guide](https://rocm.docs.amd.com/projects/rocSOLVER/en/develop/reference/tuning.html)**:
  the STEDC scheme this solver follows.

## stedc: current tuning values

The live values are in `include/batchlas/tuning_params.hh`.

- **`STEDC_MERGE_VARIANT_*` = 2 (FusedCta)** for every size class. Commit `3072ea6` moved it to 1
  (Fused), claiming FusedCta was 2-12% slower; re-tested at `0bb92fb`, that did not hold. FusedCta was
  11-35% faster than Fused on `stedc` for n = 64 to 640 on both drivers, and passed all 16 CUDA
  `stedc_tests` in both precisions, plus `syev` suites. Variant 1 caused a 1.24-1.37x `syev` regression
  at power-of-two n. The NaN seen under FusedCta was a symptom of a deadlock in the same kernel, since
  fixed.

  > **Warning:** The heavy-deflation test caveat applies to this verdict. Strengthen that test before
  > trusting variant 2 on hardware it has not been measured on.

- **`STEDC_THREADS_PER_ROOT_*` = 8 and `STEDC_WG_MULTIPLIER_*` = 8.** These were chosen through `syev`,
  not the `stedc` benchmark, which disagrees with its consumer (its own sweep picked multiplier 2-4 and
  threads-per-root 4-8). Through `syev`, ms, everything else fixed:

| syev n | wgm=2 | wgm=4 | wgm=8 | tpr=4 | tpr=8 | tpr=16 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.9987 | 0.9225 | 0.9227 | 0.9937 | 0.9229 | 0.9422 |
| 256 | 23.067 | 22.110 | 21.383 | 21.557 | 21.332 | 21.330 |
| 1024 | 903.40 | 900.12 | 903.01 | 902.35 | 904.74 | 903.35 |

  Adopting `stedc`'s own winners would cost `syev` 2.7% at n = 256 and up to 7.6% at n = 64. At n >= 512
  every value is within noise.

- **`STEDC_RECURSION_THRESHOLD_*` = 32.** This is the CTA leaf-size invariant (a leaf of 33 or more
  falls from `steqr_cta` to `steqr_wg`), not a tuning constant.

## stedc: validation protocol for merge changes

Check **orthogonality**, not just residuals. Two regressions passed residual checks while producing a
bimodal orthogonality distribution for float STEDC at n <= 64:
[the private-pole fast path](#stedc-the-rejected-private-pole-fast-path) and
[the absolute deflation tolerance](#stedc-the-absolute-deflation-tolerance).

- `tests/stedc_tests.cc`: the correctness gate (with the weak heavy-deflation guard noted above).
- `benchmarks/orthogonality_accuracy.cc` and `benchmarks/eigensolver_accuracy.cc`: compare the
  *distribution*, not only the max, against the bit-identical reference (`StedcFusedWgMerge`, or the
  `Baseline` 3-kernel path).
- `benchmarks/stedc_benchmark.cc`: timing. Re-profile with the `ncu` recipe in
  `.github/skills/nvidia-kernel-profiling/references/batchlas-stedc.md` to confirm occupancy moved.

Accuracy results: `stedc_acc --backend=CUDA --samples=64`, float and double, n = 16 to 512, 0.00000
Fail% everywhere. Orthogonality \f$\|Z^T Z - I\|_F / n\f$, float, `--samples=256`, baseline / with the
change: n = 16 1.21899e-07 / 1.21899e-07; n = 32 9.36432e-08 / 9.38560e-08; n = 64 7.41322e-08 /
7.44386e-08. Results are bit-identical across the partition-parallel loops, the denominator recompute
and the multiplier change.

**Measurement hygiene.** The box has two RTX 4090s and other jobs may run on them. A contended GPU once
inflated this work's gain from 5.5% to 8.6%. Pin an idle device (`CUDA_VISIBLE_DEVICES=1`) and check
`nvidia-smi --query-compute-apps=pid,process_name --format=csv` before trusting a number.

## stedc: end-to-end merge speedup

End-to-end `stedc`, batch 256, float, idle GPU, tuning-resolved parameters (`stedc_benchmark ... <n> 256 16 2 0 0`), ms:

| n | baseline (main) | this work | speedup |
|---|---|---|---|
| 64 | 0.930 | 0.782 | 15.9% |
| 128 | 2.042 | 1.633 | 20.0% |
| 256 | 4.885 | 3.967 | 18.8% |
| 512 | 13.144 | 11.162 | 15.1% |

At matched settings (multiplier 1, so the partition-parallel loops alone), the gain is about 5.5%
(0.933 to 0.882 ms at n = 64, 4.948 to 4.676 ms at n = 256). The rest is the multiplier change, which
the loops made possible. The deflation GEMM engages only at n >= 512.
