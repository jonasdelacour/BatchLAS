# SYEVX {#perf_syevx}

> **Status:** current · RTX 4090 (sm_89), CUDA backend, float unless a row says otherwise

Measurements behind the partial symmetric eigensolver `syevx`: the dense crossovers that set `Auto`
routing, the sparse Filtered-vs-LOBPCG sweep, the Chebyshev-degree A/B, and the LOBPCG hardening
experiments. The design is in @ref design_syevx; range selection is in @ref design_syevx_range.

Raw data:

- `benchmarks/syevx_crossover_rtx4090.csv`: eigenvalues-only and eigenvector grid, 2026-08-02, before the
  baseline fix (see [the superseded grid](#syevx-the-first-gpu-crossover-measurement-superseded-baseline)).
- `benchmarks/results/syevx_sparse_shape_rtx4090.csv`, `benchmarks/results/syevx_sparse_gap_rtx4090.csv`:
  Git LFS, 2026-08-06.

LOBPCG iteration counts in the hardening sections are CPU (NETLIB) builds with
`BATCHLAS_SYEVX_CHECK_EVERY=1`, unless a row says otherwise.

## syevx: routing thresholds as they stand

The dense `Auto` rule, from `syevx_select_algorithm` (`src/extensions/syevx.cc:299-317`):

| condition | route |
|---|---|
| `n <= kSyevxSmallN` (64) | `Direct` |
| eigenvalues only (`jobz != EigenVectors`) | `Direct` |
| eigenvectors, real `T`, `n >= kSyevxSubsetMinN` (1024) **and** `n * batch >= kSyevxSubsetMinWork` (128*1024) | `DirectSubset` |
| everything else dense | `Direct` |
| sparse (`MatrixFormat::CSR`) | `LOBPCG`, or `Filtered` only when explicitly requested |

`k` does not enter the rule. `Auto` never picks `Filtered`, even where it wins on sparse input
([Filtered vs LOBPCG on CSR](#syevx-filtered-vs-lobpcg-on-csr)).

## syevx: the Direct baseline was never cuSOLVER

Before commit `fc4fa7a9`, `syev`'s `Auto` order put the in-house blocked solver ahead of the vendor, so
every dense `Direct` solve ran it. `syev_prefer_vendor` (`include/batchlas/blas/functions/syev.hh:181`)
now sends `syev` to the vendor on CUDA. Over `n = 64..1024`, `batch = 1..512`, the vendor won 24 of 25
shapes (1.27x to 15.4x). Blocked leads by up to 1.37x on `320 <= n <= 640` with `batch >= 128`; that
carve-out is kept as measured, since the boundary is not monotone in `n`.

End to end on `syevx` `Auto` (`neigs = 8`, `n = 64..2048`, `batch = 1..128`): geometric mean 3.72x
(eigenvalues only) and 3.00x (with eigenvectors); best 33.7x, worst 0.99x.

## syevx: DirectSubset batch crossover with eigenvectors

`BM_SYEVX_CrossoverVectors` (`benchmarks/syevx_benchmark.cc`), RTX 4090, float, eigenvectors. Cells are
Direct time / DirectSubset time, so above 1 means DirectSubset wins.

| shape | b=1 | b=4 | b=16 | b=64 | b=256 |
|---|---|---|---|---|---|
| `n=1024, k=8` | 0.09 | 0.34 | 0.36 | 1.00 | 2.40 |
| `n=2048, k=8` | 0.06 | 0.28 | 0.43 | 1.12 | 1.98 |

- The reduction is batch-parallel and starves at small batch, so the win needs `batch >= 128` at
  `n = 1024` and `batch >= 64` at `n = 2048`. Both match the gate `n * batch >= 128 * 1024`. Above
  `n = 2048` the gate is extrapolated.
- `k` is deliberately absent from the gate. At `n = 1024, batch = 128` the ratio is 1.38 to 1.57 for
  `k = 8` to 102, and it ties only at `k = n/2`.
- Eigenvalues only, Direct won every shape by 3x to 5x; DirectSubset has no back-transform to narrow.
- `SyevxSelectAlgorithmTest` pins the batch dependence. Both algorithms return correct answers, so a
  routing regression is otherwise silent.

Spectrum position does not enter either path's cost (@ref design_syevx_range). `BM_SYEVX_RangePosition`
that would check this has not been run ([open debts](#syevx-open-measurement-debts)).

## syevx: the first GPU crossover measurement (superseded baseline)

Superseded. Recorded 2026-08-02 (commit `c25f5050`). Its Direct column is the blocked `syev`, not
cuSOLVER, so ratios against Direct from this grid are invalid. CSV columns are `n, batch, neigs,
algorithm` with algorithm 1..4 = Direct, DirectSubset, Filtered, LOBPCG. The DirectSubset and Filtered
columns remain valid:

- DirectSubset was 3x to 5x slower than Direct in eigenvalues-only mode at every shape. Its best
  eigenvector result was 1.46x (`n = 1024`, batch 64), or 2.40x against the corrected baseline.
- Filtered's dense niche is narrow (`n >= 1024`, `k/n` about 1 %, small batch, margin under 2x). It has
  a convergence failure mode, so it stays opt-in.
- LOBPCG on dense input was 10x to 100x behind Direct.

## syevx: Filtered vs LOBPCG on CSR

`BM_SYEVX_SparseShape` and `BM_SYEVX_SparseGap` (`benchmarks/syevx_benchmark.cc`), 2026-08-06, commit
`826bc5cb`. CUDA, float, eigenvectors, 16 nonzeros per row, `diagonal_boost` 10, `k` about 1 % of `n`,
tolerance 1e-6, `iterations = 200`, Chol2 ortho, guard block `max(1, neigs/4)`. Ratio is LOBPCG time /
Filtered time, so above 1 means Filtered wins.

| n | batch | largest end | smallest end |
|---|---|---|---|
| 1024 | 1  | **8.8**  | **2.7** |
| 1024 | 64 | **2.2**  | **1.5** |
| 2048 | 1  | **10.9** | **4.6** |
| 2048 | 64 | **1.4**  | 0.73 |
| 4096 | 64 | 0.88     | 0.51 |

Full 9-shape grid: `benchmarks/results/syevx_sparse_shape_rtx4090.csv`.

- Filtered is ahead in 15 of 18 (shape, end) ratios, by up to 10.9x, and it is 5x to 10x more accurate
  (7e-07 to 2e-06 against 6e-06 to 1.4e-05 for LOBPCG). Report accuracy next to every timing.
- It loses at batch 64 for `n = 2048` (smallest end) and both ends of `n = 4096`. A sparse routing rule
  needs a batch term, as DirectSubset's does.
- ILU(k) is 15 % to 45 % slower than unpreconditioned LOBPCG, formation included, and buys accuracy
  (5.5e-06 vs 8.9e-06 at `n = 1024`).

`Auto` stays on LOBPCG for sparse input: the batch-64 crossover is real, and one operator family (random
sparse Hermitian with an imposed dominant diagonal) is too thin a basis for a routing rule. The next step
is a mesh/stencil family.

> **Note:** Explicit `params.method = Filtered` on CSR was silently ignored before this sweep, so runs
> labelled Filtered on CSR earlier actually ran LOBPCG. Guarded by
> `SyevxRangeRoutingTest.SparseHonoursExplicitFilteredRequest`.

### syevx: the diagonal boost gap sweep and LOBPCG stagnation

`BM_SYEVX_SparseGap`: `n = 2048`, batch 8, `k = 20`, `diagonal_boost` 1 to 30 (raw:
`benchmarks/results/syevx_sparse_gap_rtx4090.csv`).

- **`diagonal_boost` is not a difficulty knob.** Times are flat within +/-1.5 % over a 30x change in
  diagonal dominance. Scaling a random diagonal rescales the spectrum without closing the extremal gap.
  A stencil operator with a clustered interior spectrum would test the argument in
  [the Chebyshev-Davidson verdict](../design/syevx.md#chebyshev-davidson-consider-as-the-lobpcg-replacement-for-unpreconditioned-problems).
- **LOBPCG stagnates on CSR.** Its time ignores `diagonal_boost` and the requested end of the spectrum,
  and its errors are 6x to 14x above 1e-6. It runs the full 200-iteration budget every time.
- Filtered is also flat, and returns an answer 5x to 10x more accurate in 2.4x less time. Its advantage is
  cheaper iterations, not faster convergence, so do not claim "converges faster".

### syevx: automatic Chebyshev filter degree

`syevx_filtered` uses `kDefaultFilterDegree = 10` (`src/extensions/syevx_filtered.cc:55`) unless the degree
is pinned (`BATCHLAS_SYEVX_FILTER_DEGREE` or `params.filter_degree`). The per-matrix derived degree behind
`BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=1` **must not ship**.

RTX 4090, float, `n = 1024`, `neigs = 8`, derived / fixed-10 (above 1 means derived wins). EigenVectors
agrees with NoEigenVectors to within 0.02x.

| batch | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| ratio | 2.18x | 2.25x | 0.70x | 0.50x | 0.48x |

The loss tracks batch size: the source takes the maximum derived need across the batch, clamps to
`[10, 40]`, then trims by the minimum precision cap, so every matrix runs at the most demanding item's
degree. Unverified; a per-iteration degree trace at batch 8 would settle it. The constant 10 has never
been tuned on this hardware.

## syevx implementation status by tier

Every tier except Tier 5 exists (plan: @ref design_syevx).

### syevx Tier 0: dispatch and the Direct path

- `SyevxAlgorithm {Auto, Direct, DirectSubset, Filtered, LOBPCG}` on `SyevxParams::method`;
  `BATCHLAS_SYEVX_ALGORITHM` overrides it. Tests that pin an algorithm skip under a conflicting override.
- `syevx_select_algorithm` is deterministic, so `syevx` and `syevx_buffer_size` agree. An unimplemented
  choice degrades to its nearest neighbour (DirectSubset on complex input becomes Direct).
- `syevx_direct` (`src/extensions/syevx_direct.cc`) runs `syev` on a private copy of `A`. `syevx.cc` is
  only the dispatcher; LOBPCG is in `src/extensions/syevx_lobpcg.cc`.

### syevx Tier 1: stebz and stein

- `stebz` (`src/extensions/stebz.cc`): batched tridiagonal bisection, All/Index/Value ranges, one
  work-group per batch item. The Sturm recurrence clamps \f$|q|\f$ below `pivmin`, as LAPACK `dlaebz`
  does; without it the count is not monotone in \f$x\f$.
- `stein` (`src/extensions/stein.cc`): batched inverse iteration with tridiagonal LU (`dgttrf`/`dgttrs`),
  then modified Gram-Schmidt within clusters closer than `ortho_threshold * ||T||`.
- Testing trap: a block-diagonal matrix with doubled eigenvalues does not test reorthogonalization (the
  pairs are orthogonal for free). Use the Wilkinson matrix \f$W^+_{2m+1}\f$; with `ortho_threshold = 0`,
  columns 4 and 5 have a dot product of 8.6e-4.
- Tests: `stebz_tests` (10), `stein_tests` (8), float and double.

### syevx Tier 2: DirectSubset

`syevx_direct_subset` (`src/extensions/syevx_direct_subset.cc`): `sytrd_sy2sb` → band-to-tridiagonal →
`stebz` → `stein` → back-transforms on the `k` selected columns. Real types, dense input only.

- Eigenvector mode reduces at the tuned band width. Q2 is applied to the `k` columns (`unmqr_hb2st`)
  before Q1 (`ormqr`). Eigenvalue-only solves keep the Givens chase. Verified at
  `BATCHLAS_SYEV_TWO_STAGE_KD` in {1, 2, 8, 32, 64}.
- Trap: `apply_phase_rows` must take its column count from `cols()`. `z.rows()` assumes a square matrix;
  on the n x k block it writes out of bounds and corrupts neighbouring batch items.

### syevx Tier 3: Filtered

`syevx_filtered` (`src/extensions/syevx_filtered.cc`): spectral bounds → Chebyshev filter → `ortho` →
Rayleigh-Ritz, looping until the wanted residuals converge. Dense and CSR, real and complex, no
preconditioner, not limited to the smallest eigenpairs. Reachable only explicitly (`params.method` or
`BATCHLAS_SYEVX_ALGORITHM=filtered`).

Two numerical rules, each caught by a test:

1. **Normalise the scaled recurrence at the extreme Ritz value, not the Gershgorin bound.** Gershgorin
   overestimates the spectral radius by \f$O(n)\f$ against the true \f$O(\sqrt{n})\f$; dividing by \f$T_m\f$
   of a far-outside point underflows the block to zero, and orthogonalizing zero gives NaN.
2. **Bound the filter degree by precision.** Past the precision's amplification budget the Cholesky-based
   `ortho` returns NaN. The degree is capped per iteration from the Ritz values, then reduced across the
   batch so the recurrence length is uniform. The cap also applies when the least-wanted direction sits in
   the damped band (growth \f$\cosh(d\,\mathrm{acosh}(y_{far}))\f$). A budget of \f$\epsilon^{-1/2}\f$ was too
   generous; \f$\epsilon^{-1/4}\f$ holds.

Tests assert eigenvalues against a reference `syev`, because a self-consistent pair does not prove it is
one of the *wanted* pairs.

> **Known cost:** the convergence test reads a device flag on the host once per outer iteration, the same
> defect as [LOBPCG's host synchronization](#lobpcg-host-synchronization-only-at-convergence-checks).

Derivations and degree precedence: [the Filtered design notes](../design/syevx.md#syevx-filtered-the-scaled-chebyshev-filter).

### syevx: the parallel Gershgorin bounds kernel

`syevx_filtered` starts from Gershgorin bounds (`lo`/`hi` are the min/max over rows of
\f$a_{ii} \mp \sum_{j \ne i}|a_{ij}|\f$). The bounds must be conservative; a loose interval costs filter
sharpness, never correctness.

The shipped kernel gives each work-item one row (`kBoundsWG = 64` rows per work-group), so a work-group
reads a contiguous column run; a second kernel reduces the per-(matrix, row-group) partials. The legacy
one-work-item-per-matrix kernel took 80.2 % of the solve at `n = 1024` (RTX 4090).

- Bit-identical to the legacy kernel, because the inner `j` loop stays serial and ascending. Keep that
  order if the kernel is touched.
- `BATCHLAS_SYEVX_BOUNDS_LEGACY=1` restores the serial kernel for A/B.

## LOBPCG hardening in syevx

Tier 4. Iteration counts are CPU (NETLIB) unless stated.

### LOBPCG: host synchronization only at convergence checks

The loop used to read unified memory on the host every iteration (about 30 forced drains per solve). It
now drains only at the convergence check (every `check_every`, default 4, `BATCHLAS_SYEVX_CHECK_EVERY`) or
for instrumentation that cannot be staged on the device. Overshooting by up to `check_every - 1`
iterations costs less than the drains. Verified at {1, 4, 16}.

### LOBPCG: device-staged instrumentation

The residual kernel writes a staging buffer carved from the pool, and one host pass after the loop
scatters into the caller's spans.

- `SyevxInstrumentation`'s spans are not guaranteed device-accessible (the Python binding passes a plain
  `std::vector<int32_t>`, `python/batchlas/bindings/ops_spectral.cc`), so no kernel writes them.
- If instrumentation is set after `syevx_buffer_size`, there is no staging room and `syevx_lobpcg` uses the
  host-read path. `BATCHLAS_SYEVX_INSTR_HOST=1` forces it; `tests/syevx_tests.cc` A/Bs the two paths.
- Values match the host path to 6 to 7 significant figures in float.

### LOBPCG: single-pass residual kernel

One pass forms R and accumulates the per-column norms, replacing `2 * neigs` sequential group reductions
(128 barriers at `neigs = 64`).

### LOBPCG: backward-stable convergence criterion

The criterion is \f$\|r\| / (\|Ax\| + |\lambda|\,\|x\|)\f$ (Duersch et al.). The former
\f$\|r\| / (\|x\|\,|\lambda|)\f$ never converged for \f$\lambda \to 0\f$.

### LOBPCG: soft locking by column masking

Column masking (variant a) is implemented and **off by default**: `BATCHLAS_SYEVX_SOFT_LOCK=1` enables it
(the parser for this knob is inverted; see `include/batchlas/settings.hh`). The batch-wide staircase
(variant b) is not implemented.

The residual kernel flags `col_converged[bid][j]` when `residual_j <= 0.1 * tol`
(`BATCHLAS_SYEVX_LOCK_FACTOR`). **`ortho` mixes columns** (R becomes \f$R\,U^{-1}\f$, U upper triangular), so
masking on one side only fails: masking after `ortho` deletes content the unconverged columns were
orthogonalised against (NaN by iteration 26 at `n = 128, k = 8, tol = 1e-4`, where the unlocked run converged
at 14), and masking before `ortho` makes \f$R^T R\f$ singular, so `potrf` returns NaN.

The implementation masks both sides. Before `ortho`, converged columns get a deterministic pseudo-random
vector scaled to the largest surviving residual; after `ortho` they are zeroed. A zeroed column left in
`StAS` adds a spurious eigenvalue 0, which is fatal for `find_largest = false` on positive-definite A, so
each locked diagonal gets a sentinel \f$\pm(\|StAS\|_F + 1)\f$ that sorts it to the unwanted end.

The locking threshold must be stricter than the tolerance. A column locked at exactly `tol` drifts back above
it and oscillates; that made the final residual worse in 3 of 8 ILU(k) sweep cases (`tests/iluk_tests.cc`).

Measured, no win: CPU iteration counts moved by at most +1 across six cases (`n = 256, b = 4, k = 16,
tol 1e-5`: 19 to 20). Masking saves no flops, since masked columns still occupy the fixed shapes. The regime
where it should help is unreachable on CPU, because `Backend::NETLIB` forces Householder.
**A GPU A/B under Chol2 is still owed** before the default could flip.

Variant (b) would need the active blocks to be non-contiguous, which breaks the single
`S({0,n},{0,Nvecs})` slice every gemm uses. `Nvecs` is computed in three places that must agree, so collapse
those into one value first.

### LOBPCG: guard vectors by default

`extra_directions == 0` means `max(2, neigs/4)`, the convention `filter_degree` also uses; an explicit width
wins. Iterations to converge (`find_largest`, tol 1e-5, batch 4):

| n / neigs | no guard | guard | change |
|---|---|---|---|
| 64 / 4   | 32 | 20 | -38 % |
| 256 / 4  | 78 | 39 | -50 % |
| 256 / 8  | 50 | 33 | -34 % |

Across eight shapes (`n` = 64 to 256, `neigs` = 4 to 16) the guard cut iterations by 34 % to 50 %, at a cost
linear in the extra width. Comparisons of LOBPCG with other paths use a guard block too.

### LOBPCG: preconditioner staging copy removed

`iluk_apply` reads R's strided slice directly, so the `R_contiguous` repack is gone. `R_preconditioned` stays
separate: the forward solve writes `out` while still reading `rhs`.

### LOBPCG: column-reversal kernels removed

The two per-iteration batch-wide launches that reversed columns for `find_largest` are gone. X stays in
`syev`'s ascending order; no replacement was available (`MatrixView` clamps `ld`, so there is no negative
stride for vendor gemm; `syev` returns ascending only; `gemm` has no permutation). Column j pairs with
`lambda[eig_offset + j]`, where `eig_offset = num_eigvals - block_vectors` when `find_largest`. The
largest-first order comes from the `X_best` snapshot. One cold-path launch remains, for `params.iterations == 0`.

`SyevxLobpcgVectorsTest` checks per-pair residuals against the original `A`. Tests that request only
`NoEigenVectors` cannot catch a column-to-eigenvalue mismatch.

### LOBPCG: X-only fill and power-iteration start

- The random fill covers only the X block: the start is bit-for-bit unchanged, with a third of the traffic.
- A block power-iteration start, \f$X \leftarrow \mathrm{ortho}(AX)\f$ (`SyevxParams::init_power_iterations`,
  default 4, `BATCHLAS_SYEVX_INIT_POWER`), runs before the first Rayleigh-Ritz when `find_largest`. Dense random
  Hermitian, double, batch 4, tol 1e-5, mean iterations over 3 seeds:

| n, k | p=0 | p=1 | p=4 | p=8 |
|---|---|---|---|---|
| 64, 4   | 22.67 | 20.67 | 18.67 | 16.00 |
| 64, 16  | 8.33  | 7.00  | 4.67  | 3.00  |
| 256, 8  | 36.00 | 35.33 | 31.00 | 27.33 |

`p = 4` gives 12 % to 44 % fewer iterations (`n = 64, k = 16`: 23.4 to 11.3 ms). `p = 8` is better still, but the
default stays at 4: an over-compressed block is rank-deficient, and the Cholesky `ortho` then returns NaN.

Negative result: the same start with \f$\sigma I - A\f$ for `find_largest = false` was flat (23.00 to 21.67 at
`n = 64, k = 4`) and is not implemented.

### LOBPCG: Jacobi preconditioners

`SyevxParams::preconditioner_type` (`SyevxPreconditioner`; `Auto` keeps the old behaviour). `diag(A)` is
extracted once, and the apply is one in-place kernel over R. `DISABLED_IterationSweep` in
`tests/syevx_tests.cc` reproduces the fp32 counts.

- **`Jacobi`** = \f$\mathrm{diag}(A)^{-1}\f$ approximates \f$A^{-1}\f$, so `find_largest` is rejected (forcing it
  turned 21 to 47 iterations into 127 to 300). On a dominant positive diagonal it is worth **2.1x to 7.3x**
  (70 to 11 at `n = 512, k = 2`). Where `diag(A)` is not positive definite a per-item test disables it. On
  random symmetric input it is an exact no-op.
- **`JacobiShifted`** = \f$(\mathrm{diag}(A) - \lambda I)^{-1}\f$ is legal at both ends, but it degenerates where
  `Jacobi` wins: as \f$\mathrm{diag}(A) \to A\f$ it becomes the inverse of \f$A - \lambda I\f$, and orthogonalization
  against X annihilates the new direction. It measured 0.85x to 1.2x on random symmetric input and 0.2x to 0.9x on
  graded input. It is kept only as the one Jacobi form legal for `find_largest`.
- Neither is selected by `Auto`. On a constant diagonal `JacobiShifted` is a per-column scalar, and LOBPCG is
  invariant to that, so its counts equal the unpreconditioned ones. `SyevxJacobiIterations.ShiftedIsANoOpOnConstantDiagonal`
  checks that the shift tracks the right Ritz value; it must run without soft locking.
- The Chebyshev "preconditioner" (a low-degree \f$p(A)\f$ applied to R) is not attempted.

### LOBPCG: the projected syev provider

The projected problem is a batched `3k x 3k` dense eigensolve per iteration. `syev`'s `Auto` gave every
`n <= 32` to CTA, so it never reached the vendor. At `n = 30`, batch 8, float, eigenvectors: CTA 229.6 µs per call
against cuSOLVER's 103.7 µs (2.21x), and nsys attributes 29.4 % of LOBPCG GPU time to it.

`syev_prefer_vendor_over_cta` (`include/batchlas/blas/functions/syev.hh`) would send eigenvector-mode CUDA solves
with `n > BATCHLAS_SYEV_CTA_MAX_N` to the vendor. It ships **off** (default 32, `syev_cta_max_n_default_for<T>()`):
it measured 1.10x to 1.15x through LOBPCG at batch 8 but flips a marginal case in
`ILUKTests.SyevxInstrumentationAndPreconditioner`. The owner decision is on the [syev page](syev.md#syev-the-lobpcg-projected-solve-knob).
`BATCHLAS_SYEVX_PROJECTED_VENDOR` forces the vendor for the projected solve; it changes workspace size, so it must
not change between a buffer-size query and the call.

## syevx: open measurement debts

- **`BM_SYEVX_RangePosition` has not been run** (@ref design_syevx_range).
- **Soft locking under Chol2 on GPU:** the only A/B ran on CPU, where Householder is forced.
- **Projected-syev provider:** only `n = 30` is measured in `1 <= n <= 32`; `syev_jacobi_cta` was never measured.
- **Filtered's large-batch collapse is unprofiled.** At `n = 1024`, `k/n` about 1 %, Filtered wins at batch 1
  (62 ms vs 114 ms for the superseded grid's blocked Direct) and loses at batch 64 (23.5 vs 3.5 ms per matrix,
  6.7x). Suspect the `ortho` / Rayleigh-Ritz tail, not the GEMMs; profile before tuning.
- **Sparse routing** needs a mesh/stencil family and a spectral-gap experiment that closes the gap.
- **Filtered degree:** `kDefaultFilterDegree = 10` is untuned; the auto-degree batch reduction needs a design change.
- **Above `n = 2048`** the DirectSubset gate is extrapolated.
- **Not done in LOBPCG:** a batch-wide staircase, warm starts, a Chebyshev preconditioner, SYCL-graph capture.
