# STEDC: the divide-and-conquer merge {#perf_stedc}

> **Status:** current · RTX 4090 (sm_89), CUDA · measured 2026-08-01; `syev` tuning 2026-08-07

The `stedc` merge is serialization- and latency-bound (Nsight: 8.8% SM, 4% achieved occupancy before
this work). This page records why the merge, drivers and tuning constants are shaped as they are. The
leaf solver is in @ref algo_steqr; live values are in `include/batchlas/tuning_params.hh`.

## stedc: partition-parallel rescale and normalize

`maybe_rescale_vectors` and `normalize_vectors` (`src/ops/syev/stedc/stedc_merge_cta.cc`) run one
index per sub-group partition, concurrently, with one `group_barrier(wg)` between phases, instead of a
whole-work-group reduction and a barrier per index. The WG variant (executor `v2::RtGroup`) keeps the old,
bit-identical code as the A/B reference. Alone this gains about 5.5%; with the multiplier change it
enabled, 15-20% end to end (batch 256, float, \f$n = 64\f$ 0.930 -> 0.782 ms, \f$n = 512\f$ 13.144 -> 11.162 ms).

> **Warning:** Narrow groups make each serial Löwner product longer (8x from width 32 to 4); the naive
> product under/overflowed at \f$n = 64\f$. It is accumulated as a (mantissa, exponent) pair with `frexp`.

> **Warning:** Wide groups are the opposite hazard: a group-wide product of mantissas in \f$[0.5, 1)\f$ over
> \f$W = 256\f$ lanes is zero in float (eigen_residual 0.56 at \f$n = 512\f$,
> `StedcTest.StedcFusedWideWorkgroupRescale`). The reduction runs in rounds of 32 lanes and renormalizes.

> **Warning:** `dd` is rarely a multiple of `parts_per_wg`, and the sub-group shuffles need every lane:
> both loops run a uniform trip count and mask the work.

| Alternative | Result | Verdict |
| --- | --- | --- |
| Recompute rescale denominators from shared memory | 8-17% slower (\f$64 \le n \le 512\f$); a control with 2 unused shared arrays regressed the same | Rejected: any revisit must add no shared memory |

## stedc: the dd == 1 secular-solve fault

When a subproblem deflates to `dd == 1`, `solve_root_ext_generic` dereferenced `d_prob(dd - 2)`, out
of the shared window (`CUDA_ERROR_ILLEGAL_ADDRESS`). `device::laed4` (`detail::solve_last`), the one
solver of every merge, now returns the closed form \f$x = d_0 + \rho z_0^2\f$ first.
`StedcTest.FusedCtaConditionedHeavyDeflation` (log10 condition 1, 3, 5, all four widths) reproduces
the fault when the fix is reverted.

`compute-sanitizer` cannot attach (SYCL initializes CUDA first). Find such faults with device asserts:
configure with `-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -UNDEBUG"`
(`-UNDEBUG` overrides the `-DNDEBUG` in `$DEFINES`).

## stedc: the work-group multiplier after the barriers went

`STEDC_WG_MULTIPLIER_*` is 8 for every size class (was 1 / 1 / 2 / 2 / 4). The width pays only because
the partition-parallel loops removed the barriers: \f$n = 64\f$, batch 256, float, multiplier 1 / 8 gave
0.880 / 0.729 ms with them and 0.919 / 0.877 ms without.

> **Warning:** At about 80 registers per work-item, 1024 work-items do not launch. `choose_wg_size`
> also clamps to the kernel's own limit (`kernel_max_work_group_size`, 768 on sm_89).

`stedc_benchmark` argument 3 selects the merge variant: `-1` Auto, `0` Baseline (3-kernel path), `1`
Fused, `2` FusedCta; non-positive arguments resolve from the tuning tables.

## stedc: deflation-aware back-transform GEMM

`Qprime` is identity beyond its first `dd` columns, so \f$A [W \mid I] = [AW \mid A_{:,dd:}]\f$ and only
the first \f$dd_{\max}\f$ columns need a GEMM. One uniform cuBLAS call at the batch-wide
\f$dd_{\max}\f$ stays exact (extra columns are identity) and needs no workspace (\f$A\f$ goes into
`temp_Q`). Random tridiagonals keep 17% of the columns at merge size 256 and 91% at 32.

### stedc: the deflation GEMM threshold

Learning \f$dd_{\max}\f$ costs a host sync, so the path is gated at \f$n \ge 512\f$
(`stedc_deflation_gemm_min_n`) and skipped unless deflation removed at least 25% of the columns
(`stedc_deflation_gemm_max_kept_fraction = 0.75`). Off vs on, batch 256, float, ms:

| n | off | on | gain |
| --- | --- | --- | --- |
| 256 | 3.998 | 4.020 | noise (a gate at 128 regressed \f$n = 256\f$ by ~5%) |
| 512 | 11.166 | 10.525 | +5.8% |
| 1024 | 39.388 | 29.937 | +24.0% |

LAPACK's `dlaed2`/`dlaed3` `CTOT` split could gain ~2x more but breaks the uniform vendor GEMM into
ragged calls; not done.

## stedc: eigenvalues-only still builds eigenvectors

Open as a cost, avoided by `syev` (which uses `stebz` for values-only; no internal caller passes
`NoEigenVectors` to `stedc`). Each merge builds its secular vector from the last row of the left
child's eigenvectors and the first row of the right child's. The recursive driver used to forward
`NoEigenVectors` to the leaf `steqr_dispatch`, which then wrote no vectors, so every merge read the
zeroed matrix: all of z deflated and the "eigenvalues" were the sorted torn diagonal, finite and
plausible (values error \f$3.6 \times 10^{-3}\f$ against a bound of \f$3.4 \times 10^{-4}\f$ at \f$n = 180\f$, float, Metal). With no
eigenvector matrix the call threw instead. Fixed 2026-10-30: `stedc_dispatch` turns
`NoEigenVectors` into a vector solve into a workspace Z (`stedc_buffer_size` adds \f$n^2\f$ reals
per item) and never touches the caller's matrix. Guards: `NoEigenVectorsMatchesReferenceOnPoisonedWorkspace`
and `NoEigenVectorsWithoutAnEigenvectorMatrix` in `tests/stedc_tests.cc`. The cost is still a full
vector solve; boundary-row D&C (Zhan & Zhang, arXiv:2605.26599), which carries only the two rows
the merge reads, is the matching algorithm.

## stedc: the level-synchronous driver

`src/ops/syev/stedc/stedc.cc` has two drivers over one size-uniform merge of P subproblems pairwise.
The recursive driver merges one node at a time (\f$O(2^L)\f$ launches, `batch` work-groups each). The
level-synchronous driver merges a whole level in one launch over `nodes * batch` work-groups
(\f$O(L)\f$ launches); \f$n\f$ is padded to \f$\text{leaf}\cdot 2^L\f$ with a diagonal tail above the
Gershgorin bound (padded eigenvalues sort last), and the planner picks the tree that pads least, since
padding costs \f$(N/n)^3\f$ in the top GEMM. Merges write into the parent's sub-blocks (up to 5% faster
than copying); leaves are copied, because doubling the leaf leading dimension slowed CTA STEQR 10% at
\f$n = 64\f$.

## stedc: the leaf cap at the sub-group width

`plan_stedc_levels` (`stedc_levels_plan.hh`) never lets a leaf exceed the sub-group width:
`steqr_cta` stops there and `steqr_wg` is about 14x slower one step past it (\f$n = 32\f$ 0.26 us,
\f$n = 36\f$ 3.76 us, batch 10,416). Trading leaf width against padding regressed `syev` 3.25x at \f$n = 320\f$; see
[syev: the stedc merge-variant and leaf-cliff regression](syev.md#syev-the-stedc-merge-variant-and-leaf-cliff-regression).

## stedc: the absolute deflation tolerance

Deflation uses LAPACK's absolute \f$\mathrm{tol} = 8\varepsilon\max(\lVert D\rVert_\infty, \lVert z\rVert_\infty)\f$
for both small-\f$|z|\f$ and close-pole deflation. The earlier relative tolerance under-deflated
clustered small eigenvalues: good residuals, bimodal orthogonality.

## stedc: the rejected private-pole fast path

The fused merge once ran its own secular solver in place in `Q_bid`; keeping the poles in a private
`T d_priv[128]` array gave bimodal orthogonality for float \f$n \le 64\f$. It now runs `device::laed4`
on the poles in global memory ([below](#stedc-one-secular-solver)).

## stedc: one secular solver

Every merge (`FusedCta`, `Fused`, `Baseline`, and the CPU device) solves its roots with
`device::laed4`; the per-thread rocSOLVER copy (`sec_solve_roc`, `sec_solve_ext_roc`) and the
original `Legacy` solver (`StedcSecularSolver`) are gone. `Legacy` was 3.4-4.5x slower than
`FusedCta` (float, \f$n = 64, 128\f$) and 3.7x in double at \f$n = 64\f$. Two changes made `laed4` the
faster body on one work-item per root:

- `s += w * ratio` as one expression (an FMA; two statements were a DMUL and a DADD).
- FP64 keeps \f$d - o\f$ in the root's `Q(:, k)` scratch, one subtraction per term less; float ignores
  the scratch, whose strided reads cost it 4-6% (`Baseline` float \f$n = 64\f$: 0.876 with, 0.825
  without).

`stedc` jobz = V, batch 1024 / 512 / 256 / 128 at \f$n\f$ = 64 / 128 / 256 / 512, RTX 4090, 16
alternating reps, new / old:

| merge | float | double |
| --- | --- | --- |
| `Baseline` | 0.823-0.932 | 0.987-1.013 (\f$n = 64\f$: 1.013) |
| `Fused` | 0.875-0.941 | 0.936-0.980 |
| `FusedCta` (default) | 0.997-1.000 | 0.991-0.995 |

`Baseline` double at \f$n = 64\f$ was 1.030 with neither change and 1.021 with the FMA alone; before the
FMA its solve kernel ran 16% more FP64 instructions than the old solver (ncu).

## stedc: convergence reporting through info

`device::laed4` reports whether each root met tolerance (`laed4_root::converged`); a root that hit
`max_sec_iter` is a failure, not a silently wrong eigenvalue.

- Reports from several threads of one item merge through an atomic `fetch_max`.
- The public `stedc` clears `info` once; callers running `stedc` twice on the same items (`gesvd`)
  must use the no-clear entry point.

## stedc: open debts

- **Direct eigenvalues-only calls** still build eigenvectors
  ([above](#stedc-eigenvalues-only-still-builds-eigenvectors)).

## stedc: current tuning values

- **`STEDC_MERGE_VARIANT_*` = 2 (FusedCta)** everywhere: 11-35% faster than Fused (1) on `stedc` for
  \f$64 \le n \le 640\f$ on both drivers; Fused cost `syev` 1.24-1.37x at power-of-two \f$n\f$. An earlier NaN under
  FusedCta was a since-fixed deadlock.
- **`STEDC_THREADS_PER_ROOT_*` = 8, `STEDC_WG_MULTIPLIER_*` = 8**, chosen through `syev`, not the
  `stedc` benchmark (whose own sweep picked multiplier 2-4). Adopting `stedc`'s winners would cost
  `syev` 2.7% at \f$n = 256\f$ and up to 7.6% at \f$n = 64\f$; at \f$n \ge 512\f$ all values are within noise. `syev`, ms:

| syev n | wgm=2 | wgm=4 | wgm=8 | tpr=4 | tpr=8 | tpr=16 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.9987 | 0.9225 | 0.9227 | 0.9937 | 0.9229 | 0.9422 |
| 256 | 23.067 | 22.110 | 21.383 | 21.557 | 21.332 | 21.330 |

- **`STEDC_RECURSION_THRESHOLD_*` = 32** is the CTA leaf-size invariant, not a tuning constant.

## stedc: validation protocol for merge changes

Check **orthogonality**, not just residuals: the private-pole path and the relative deflation
tolerance both passed residual checks with bimodal float orthogonality at \f$n \le 64\f$.

- `tests/stedc_tests.cc` is the gate (it checks eigenvalues against LAPACKE, orthogonality and
  residual).
- `benchmarks/orthogonality_accuracy.cc`, `benchmarks/eigensolver_accuracy.cc`: compare the
  *distribution* against the bit-identical reference (`StedcFusedWgMerge` or the Baseline path).
- `benchmarks/stedc_benchmark.cc` for timing; the `ncu` recipe is
  `.github/skills/nvidia-kernel-profiling/references/batchlas-stedc.md`.

## stedc: literature on the merge

The secular solver (`device::laed4`) is the rocSOLVER middle-way / fixed-weight hybrid with
bound-clamped steps; it follows the recommended scheme, leave it alone. Untried leads: relative-accuracy
Löwner vectors (Jakovčević Stor, Slapničar & Barlow, arXiv:1405.7537; could remove `enable_rescale`),
HSS-compressed eigenvectors for large n (arXiv:1510.04591), and boundary-row D&C for values-only
(arXiv:2605.26599). Eigenvector formula: Gu & Eisenstat, SIMAX 1994.
