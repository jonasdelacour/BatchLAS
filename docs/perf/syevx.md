# SYEVX {#perf_syevx}

> **Status:** current · RTX 4090 (sm_89), CUDA backend, float unless a row says otherwise

Measurements behind `syevx`: the dense crossovers that set `Auto` routing, the sparse Filtered-vs-LOBPCG
sweep, the Chebyshev-degree A/B and the LOBPCG hardening. Design is in @ref design_syevx; range selection
in @ref design_syevx_range. LOBPCG iteration counts are CPU (NETLIB) builds with
`BATCHLAS_SYEVX_CHECK_EVERY=1` unless a row says otherwise.

## syevx: routing thresholds as they stand

`syevx_select_algorithm` (`src/ops/syevx/syevx.cc`, constants at the top of the file):

| condition | route |
|---|---|
| `n <= kSyevxSmallN` (64) | `Direct` |
| eigenvalues only (`jobz != EigenVectors`) | `Direct` |
| eigenvectors, real `T`, `n >= kSyevxSubsetMinN` (1024) **and** `n * batch >= kSyevxSubsetMinWork` (128*1024) | `DirectSubset` |
| everything else dense | `Direct` |
| sparse (`MatrixFormat::CSR`) | `LOBPCG`, or `Filtered` only when explicitly requested |

`k` does not enter the rule. `Auto` never picks `Filtered`. An explicit `DirectSubset` on complex input
degrades to `Direct`. `BATCHLAS_SYEVX_ALGORITHM` overrides `SyevxParams::method`.

## syevx: the Direct baseline was never cuSOLVER

Before commit `fc4fa7a9`, every dense `Direct` solve ran the in-house blocked `syev`, not the vendor. With
`syev` on the vendor (now via flat selection, see @ref perf_syev), `syevx` `Auto` (`neigs = 8`,
`n = 64..2048`, `batch = 1..128`) gained a geometric mean 3.72x (eigenvalues) and 3.00x (eigenvectors);
worst 0.99x.

## syevx: DirectSubset batch crossover with eigenvectors

`BM_SYEVX_CrossoverVectors`, float, eigenvectors. Cells are Direct time / DirectSubset time (above 1:
DirectSubset wins).

| shape | b=1 | b=4 | b=16 | b=64 | b=256 |
|---|---|---|---|---|---|
| `n=1024, k=8` | 0.09 | 0.34 | 0.36 | 1.00 | 2.40 |
| `n=2048, k=8` | 0.06 | 0.28 | 0.43 | 1.12 | 1.98 |

- The batch-parallel reduction starves at small batch: the win needs `batch >= 128` at `n = 1024` and
  `batch >= 64` at `n = 2048`, matching `n * batch >= 128 * 1024`. Above `n = 2048` it is extrapolated.
- `k` is absent on purpose: at `n = 1024, batch = 128` the ratio is 1.38 to 1.57 for `k = 8..102`; it
  ties only at `k = n/2`.
- Eigenvalues only, Direct wins every shape by 3x to 5x.
- `SyevxSelectAlgorithmTest` pins the batch dependence; both paths are correct, so a routing regression
  is otherwise silent.

## syevx: the first GPU crossover measurement (superseded baseline)

2026-08-02, commit `c25f5050`. Its Direct column is the blocked `syev`, so ratios against Direct are
invalid. What remains valid: Filtered's dense niche is narrow (`n >= 1024`, `k/n` about 1 %, small batch,
under 2x) and it has a convergence failure mode, so it stays opt-in; LOBPCG on dense input was 10x to
100x behind Direct.

## syevx: Filtered vs LOBPCG on CSR

`BM_SYEVX_SparseShape`, 2026-08-06, commit `826bc5cb`, float, eigenvectors, 16 nonzeros/row,
`diagonal_boost` 10, `k` about 1 % of `n`, tol 1e-6, 200 iterations, Chol2 ortho. LOBPCG time / Filtered
time (above 1: Filtered wins). Raw: `benchmarks/results/syevx_sparse_shape_rtx4090.csv` (LFS).

| n | batch | largest end | smallest end |
|---|---|---|---|
| 1024 | 1  | **8.8**  | **2.7** |
| 2048 | 64 | **1.4**  | 0.73 |
| 4096 | 64 | 0.88     | 0.51 |

Filtered wins 15 of 18 (shape, end) cells and is 5x to 10x more accurate, but loses at batch 64 for
`n >= 2048`. `Auto` stays on LOBPCG for CSR: a sparse rule needs a batch term and more than one operator
family (see the open debts). `SyevxRangeRoutingTest.SparseHonoursExplicitFilteredRequest` guards that an
explicit Filtered request on CSR is honoured (it was silently ignored before this sweep).

### syevx: the diagonal boost gap sweep and LOBPCG stagnation

`BM_SYEVX_SparseGap`, `n = 2048`, batch 8, `k = 20`, `diagonal_boost` 1 to 30
(`benchmarks/results/syevx_sparse_gap_rtx4090.csv`). Times are flat within +/-1.5 %: scaling a random
diagonal does not close the extremal gap, so `diagonal_boost` is not a difficulty knob. LOBPCG runs the
full 200-iteration budget every time (errors 6x to 14x above 1e-6); Filtered is 5x to 10x more accurate in
2.4x less time through cheaper iterations, not faster convergence. A stencil family would test
[the Chebyshev-Davidson verdict](../design/syevx.md#chebyshev-davidson-consider-as-the-lobpcg-replacement-for-unpreconditioned-problems).

### syevx: automatic Chebyshev filter degree

`kDefaultFilterDegree = 10` (`syevx_filtered.cc`, untuned) unless pinned (`BATCHLAS_SYEVX_FILTER_DEGREE`
or `params.filter_degree`). The derived degree behind `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=1` stays off:
`n = 1024`, `neigs = 8`, derived / fixed-10 was 2.18x at batch 1, 2.25x at 2, then 0.70x, 0.50x, 0.48x
at batch 4, 8, 16. The batch reduction (max need, clamped to `[10, 40]`) makes every item run at the most
demanding item's degree; that explanation is unverified (a per-iteration degree trace at batch 8 would
settle it). Precedence and derivation: @ref design_syevx.

## syevx Tier 1: stebz and stein

`stebz` (batched bisection, All/Index/Value) clamps \f$|q|\f$ below `pivmin` in the Sturm recurrence, as
`dlaebz` does; without it the count is not monotone. `stein` (inverse iteration, then MGS within clusters
closer than `ortho_threshold * ||T||`). **Testing trap:** doubled eigenvalues of a block-diagonal matrix
are orthogonal for free; use the Wilkinson matrix \f$W^+_{2m+1}\f$ (with `ortho_threshold = 0`, columns 4
and 5 have a dot product of 8.6e-4).

## syevx Tier 2: DirectSubset

`sytrd_sy2sb` → band-to-tridiagonal → `stebz` → `stein` → back-transforms on the `k` columns (Q2 via
`unmqr_hb2st`, then Q1 via `ormqr`). Real types, dense only. Verified at `BATCHLAS_SYEV_TWO_STAGE_KD` in
{1, 2, 8, 32, 64}. **Trap:** `apply_phase_rows` must take its column count from `cols()`; `rows()` writes
out of bounds on the n x k block and corrupts neighbouring batch items.

## syevx Tier 3: Filtered

Bounds → Chebyshev filter → `ortho` → Rayleigh-Ritz; dense and CSR, real and complex, no preconditioner,
reachable only explicitly. Its two NaN rules (normalise at the extreme Ritz value; cap the degree at an
\f$\epsilon^{-1/4}\f$ budget) are in [the Filtered design notes](../design/syevx.md#syevx-filtered-the-scaled-chebyshev-filter).
Tests assert eigenvalues against a reference `syev`: a self-consistent pair need not be a *wanted* pair.
**Known cost:** one host read of a device flag per outer iteration, as in
[LOBPCG](#lobpcg-host-synchronization-only-at-convergence-checks).

### syevx: the parallel Gershgorin bounds kernel

One work-item per row (`kBoundsWG = 64` rows per work-group), then a second kernel reduces the
per-row-group partials. The old one-work-item-per-matrix kernel took 80.2 % of the solve at `n = 1024`.
The result is bit-identical to it because the inner `j` loop stays serial and ascending; keep that order.

## LOBPCG: host synchronization only at convergence checks

The loop drains to the host only at the convergence check (every `BATCHLAS_SYEVX_CHECK_EVERY`, default 4)
or for unstaged instrumentation, not every iteration (about 30 drains per solve before). Overshooting by
up to `check_every - 1` iterations costs less than the drains.

Also shipped: a single-pass residual kernel (R and its column norms in one pass), the
backward-stable criterion \f$\|r\| / (\|Ax\| + |\lambda|\,\|x\|)\f$ (the old \f$\|r\|/(\|x\|\,|\lambda|)\f$ never
converged for \f$\lambda \to 0\f$), and `iluk_apply` reading R's strided slice with no repack.

## LOBPCG: device-staged instrumentation

The residual kernel writes a pool staging buffer; one host pass after the loop scatters into the caller's
spans, which are not guaranteed device-accessible (the Python binding passes a `std::vector`). If
instrumentation is set after `syevx_buffer_size` there is no staging room and the host-read path runs;
`BATCHLAS_SYEVX_INSTR_HOST=1` forces it, and `tests/syevx_tests.cc` A/Bs the two paths.

## LOBPCG: soft locking by column masking

Off by default; `BATCHLAS_SYEVX_SOFT_LOCK` enables it (inverted parser, see `include/batchlas/settings.hh`).
A column locks when `residual_j <= BATCHLAS_SYEVX_LOCK_FACTOR * tol` (0.1); locking at exactly `tol`
oscillates. Because `ortho` mixes columns (\f$R\,U^{-1}\f$), both sides are masked: before `ortho` a locked
column gets a deterministic pseudo-random vector scaled to the largest live residual (else \f$R^TR\f$ is
singular), after it the column is zeroed and its `StAS` diagonal gets a sentinel \f$\pm(\|StAS\|_F + 1)\f$ so
the spurious 0 eigenvalue sorts to the unwanted end. No measured win: CPU counts moved by at most +1 over
six cases, and masking saves no flops. The batch-wide staircase is not built; it first needs the three
`Nvecs` computations collapsed into one.

## LOBPCG: guard vectors by default

`extra_directions == 0` means `max(2, neigs/4)`; an explicit width wins. Over eight shapes (`n = 64..256`,
`neigs = 4..16`, `find_largest`, tol 1e-5, batch 4) it cut iterations by 34 % to 50 % (`n = 256, neigs = 4`:
78 to 39), at a cost linear in the width. Comparisons with other paths use a guard block too.

## LOBPCG: column-reversal kernels removed

X stays in `syev`'s ascending order instead of being reversed for `find_largest` each iteration; column
j pairs with `lambda[eig_offset + j]`, `eig_offset = num_eigvals - block_vectors` when `find_largest`,
and the `X_best` snapshot supplies largest-first order. One cold launch remains for `iterations == 0`.
`SyevxLobpcgVectorsTest` checks per-pair residuals against `A`; `NoEigenVectors` tests cannot catch a
column-to-eigenvalue mismatch.

## LOBPCG: X-only fill and power-iteration start

The random fill covers only X (start bit-identical, a third of the traffic). For `find_largest`,
\f$X \leftarrow \mathrm{ortho}(AX)\f$ runs `init_power_iterations` times first (default 4,
`BATCHLAS_SYEVX_INIT_POWER`). Double, batch 4, tol 1e-5, mean of 3 seeds: `p = 4` gives 12 % to 44 % fewer
iterations (`n = 64, k = 16`: 8.33 to 4.67). `p = 8` is better but rank-deficient blocks make the Cholesky
`ortho` return NaN, so the default stays 4. The \f$\sigma I - A\f$ variant for the smallest end was flat
and is not built.

## LOBPCG: Jacobi preconditioners

`SyevxParams::preconditioner_type`; neither form is picked by `Auto`. `DISABLED_IterationSweep` in
`tests/syevx_tests.cc` reproduces the fp32 counts.

- **`Jacobi`** (\f$\mathrm{diag}(A)^{-1}\f$): rejected for `find_largest` (forcing it took 21-47 iterations
  to 127-300). On a dominant positive diagonal 2.1x to 7.3x (70 to 11 at `n = 512, k = 2`); a per-item
  test disables it where `diag(A)` is not positive definite; a no-op on random symmetric input.
- **`JacobiShifted`** (\f$(\mathrm{diag}(A) - \lambda I)^{-1}\f$): legal at both ends but degenerates where
  `Jacobi` wins (0.85x to 1.2x random, 0.2x to 0.9x graded). Kept only as the Jacobi form legal for
  `find_largest`. On a constant diagonal it is a per-column scalar, to which LOBPCG is invariant;
  `SyevxJacobiIterations.ShiftedIsANoOpOnConstantDiagonal` allows one check interval of difference and
  must run without soft locking. It does not catch a swapped Ritz index.

## LOBPCG: the accuracy pin on the projected syev

The tuned `syev` tables rank by time only, and on sm_120 they put `vendor` or `jacobi` first for small
eigenvector cells. On threadripper02 (RTX PRO 6000), float, n = 16, cuSOLVER's eigenvalues are about
1.9e-5 off against 2.7e-6 for CTA, and with the projected solves off CTA
`ILUKTests.SyevxInstrumentationAndPreconditioner` (`iluk_k3`, `d0.06_b0.5`) stalls at 5.2e-6 against a
4.2e-6 target (2026-10-09).

`projected_syev_pin` (`syevx_lobpcg.cc`) pins every projected solve (XtAX, StAS, restart) to `cta_fused`
when CTA can take the shape (n <= 32, 32-wide sub-groups); the buffer-size query uses the same pin. Above
32 the table decides; an explicit `BATCHLAS_SYEV_ROUTE` or outer `select::ScopedPin` still wins, so
`Pinned/SyevxProjectedSyevRouteTest.*` exercises each family.
`SyevxProjectedSyevPin.CtaFusedAtOrBelow32EvenWhenTheTableRanksTheVendorFirst` guards the pin. The speed
side (CTA 2.21x slower than cuSOLVER at n = 30) is on [the syev page](syev.md#syev-the-lobpcg-projected-solve-knob).
`BATCHLAS_SYEVX_PROJECTED_VENDOR` changes workspace size; do not change it between query and call.

## syevx: open measurement debts

- **`BM_SYEVX_RangePosition` has not been run** (@ref design_syevx_range).
- **Soft locking under Chol2 on GPU:** the only A/B ran on CPU, where Householder is forced; owed before
  the default could flip.
- **Projected syev:** only `n = 30` is measured in `1 <= n <= 32`; `syev_jacobi_cta` never was.
- **Filtered's large-batch collapse is unprofiled** (`n = 1024`, `k/n` about 1 %: wins at batch 1, 6.7x
  behind per matrix at batch 64). Suspect the `ortho` / Rayleigh-Ritz tail; profile before tuning.
- **Sparse routing** needs a mesh/stencil family and a gap-closing experiment.
- **Filtered degree:** the constant 10 is untuned; the auto-degree batch reduction needs a design change.
- **Above `n = 2048`** the DirectSubset gate is extrapolated.
- **Not done in LOBPCG:** batch-wide staircase, warm starts, a Chebyshev preconditioner, SYCL-graph capture.
