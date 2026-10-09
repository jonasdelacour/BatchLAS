# syevx: measured crossovers, LOBPCG hardening and tier status {#perf_syevx}

**Covers:** every measurement behind the partial symmetric eigensolver `syevx`: the dense
crossovers that set `Auto` routing (`syevx_select_algorithm` in `src/extensions/syevx.cc`),
the sparse Filtered-vs-LOBPCG sweep, the Chebyshev-degree A/B, the LOBPCG hardening
experiments, and which implementation tiers exist.
**Status:** current. The dense grid of 2026-08-02 in
[the first GPU crossover measurement](#syevx-the-first-gpu-crossover-measurement-superseded-baseline)
is kept for the record but was measured against a wrong baseline. It was replaced the same day
by [the routing thresholds as they stand](#syevx-routing-thresholds-as-they-stand).
**Machine:** RTX 4090 (sm_89), CUDA backend, float, unless a row says otherwise. The LOBPCG
iteration-count tables were measured on the NETLIB/CPU build and are marked as such.
**Dates:** tiers 0 to 4 on 2026-08-01/02; the GPU crossover and re-routing on 2026-08-02; the CSR
sweep on 2026-08-06. The dates come from the commits that recorded each result.
**Raw data:** `benchmarks/syevx_crossover_rtx4090.csv` (eigenvalues-only `BM_SYEVX_Crossover`,
2026-08-02, from **before** the baseline fix, see below),
`benchmarks/results/syevx_sparse_shape_rtx4090.csv` and
`benchmarks/results/syevx_sparse_gap_rtx4090.csv` (Git LFS, 2026-08-06).

Design rationale, the flop-count cost model these numbers overturned, and the algorithm survey
are in @ref design_syevx. The LAPACK-style `Index`/`Value` range selection is in
@ref design_syevx_range.

## syevx: routing thresholds as they stand

The dense `Auto` rule, quoted from `syevx_select_algorithm` (`src/extensions/syevx.cc:299-317`):

| condition | route |
|---|---|
| `n <= kSyevxSmallN` (64) | `Direct` |
| eigenvalues only (`jobz != EigenVectors`) | `Direct` |
| eigenvectors, real `T`, `n >= kSyevxSubsetMinN` (1024) **and** `n * batch >= kSyevxSubsetMinWork` (128*1024) | `DirectSubset` |
| everything else dense | `Direct` |
| sparse (`MatrixFormat::CSR`) | `LOBPCG`, or `Filtered` only when explicitly requested |

`k` (the number of wanted eigenpairs) appears nowhere in the rule. `Filtered` is never picked by
`Auto`: see [Filtered vs LOBPCG on CSR](#syevx-filtered-vs-lobpcg-on-csr) for why that holds
even on sparse input where it wins.

### syevx: the Direct baseline was never cuSOLVER

`Direct` calls `syev`. Until commit `fc4fa7a9` (2026-08-02), `syev`'s `Auto` order listed
`BatchLAS_Blocked` ahead of `Vendor`, and `syev_supports_blocked` is unconditionally true on a
GPU for `n > 32`. So every dense `Direct` solve ran our own blocked solver, not cuSOLVER. The
blocked reduction's panel factorization is parallel over the *batch*, one work-group per matrix.
At `n = 1024, batch = 1` its panel kernel (`LatrdLowerPanelKernelLegacy`) took 357 ms of a 405 ms
solve (88 %), moving about 134 MB per panel at roughly 1/48 of device bandwidth on one SM of 128. So the old
baseline was slow for exactly the reason DirectSubset is slow at small batch, and the two
comparing "evenly" at batch 1 was two starved kernels, not a fair comparison.

The fix (`syev_prefer_vendor`, `include/batchlas/blas/functions/syev.hh:181`) sends `syev`
to the vendor on CUDA. Over `n = 64..1024` x `batch = 1..512` the vendor won 24 of 25 shapes, by
1.27x to 15.4x. The one exception is the box `320 <= n <= 640`, `batch >= 128`, where blocked
leads by up to 1.37x. It is carved out as measured rather than smoothed, because the boundary is
not monotone in `n`. Direct got up to 15.4x faster, and the syevx thresholds had to be measured
again against it.

End to end on `syevx` `Auto`, `neigs = 8`, over `n = 64..2048` x `batch = 1..128`, the two
fixes together gave a geometric mean of 3.72x (eigenvalues only) and 3.00x (with eigenvectors).
The best case was 33.7x (`n = 2048, batch = 1`) and the worst 0.99x, so no shape regressed.

### syevx: DirectSubset batch crossover with eigenvectors

`BM_SYEVX_CrossoverVectors` (`benchmarks/syevx_benchmark.cc`), RTX 4090, float, eigenvectors.
The ratio is Direct time / DirectSubset time, so a value above 1 means DirectSubset wins:

| shape | | | | | | |
|---|---|---|---|---|---|---|
| `n=1024, k=8` | b=1 0.09 | b=4 0.34 | b=16 0.36 | b=64 1.00 | b=256 2.40 | |
| `n=2048, k=8` | b=1 0.06 | b=4 0.28 | b=16 0.43 | b=64 1.12 | b=256 1.98 | |
| `n=1024, b=128` | k=8 1.47 | k=25 1.57 | k=51 1.38 | k=102 1.51 | | |
| `n=1024, b=256` | k=8 2.12 | k=25 1.93 | k=51 1.96 | k=102 1.83 | k=256 1.43 | k=512 1.00 |

- DirectSubset's reduction is batch-parallel, exactly like the blocked `syev` it used to be
  compared with, so it starves at small batch: it loses by up to 16x at batch 1 and wins by up
  to 2.4x at batch 256.
- Two anchors bound the win region. `n = 1024` needs `batch >= 128` and `n = 2048` needs
  `batch >= 64`. Both are `n * batch >= 128 * 1024`, which is the form of the gate. Above
  `n = 2048` the gate extrapolates rather than interpolates, in the direction the two anchors
  already move.
- **`k` is deliberately absent.** The ratio is flat in `k` from 0.8 % to 25 % of the spectrum and
  decays to a tie only at 50 %.
- Eigenvalues only: `Direct` won at every measured shape, by 3x to 5x. The subset path pays the
  full reduction and has no back-transform to narrow, so there is nothing for it to win with.
  That conclusion comes from the older grid below, and the baseline fix only made `Direct`
  faster, so it can only have become stronger.
- `SyevxSelectAlgorithmTest` pins the batch dependence. Both algorithms return correct answers,
  so a routing regression is silent everywhere else.

Where in the spectrum the `k` eigenpairs sit is absent for a structural reason, argued in
@ref design_syevx_range. The benchmark that would check it, `BM_SYEVX_RangePosition`, has not
been run (see [open measurement debts](#syevx-open-measurement-debts)).

## syevx: the first GPU crossover measurement (superseded baseline)

Recorded 2026-08-02 (commit `c25f5050`) with `BM_SYEVX_Crossover` plus a separate
eigenvector-mode sweep. **Its `Direct` column is the blocked `syev`, not cuSOLVER** (see above).
The DirectSubset and Filtered columns are still valid measurements of those paths at that date.
The *ratios* against Direct are not. The CSV `benchmarks/syevx_crossover_rtx4090.csv` is this
eigenvalues-only grid (args `n, batch, neigs, algorithm`, with algorithm 1..4 = Direct,
DirectSubset, Filtered, LOBPCG).

Eigenvalues only, microseconds per matrix, batch 8:

| n | k/n | Direct (blocked) | DirectSubset | Filtered |
|---|---|---|---|---|
| 256 | 1.95 % | **721** | 3609 | 1762 |
| 512 | 1.95 % | **2612** | 8211 | 9257 |
| 1024 | 0.98 % | 13943 | 50419 | **13179** |
| 1024 | 9.96 % | **13980** | 50526 | 108668 |

With eigenvectors, microseconds per matrix:

| n | batch | k/n | Direct (blocked) | DirectSubset | Filtered |
|---|---|---|---|---|---|
| 512 | 8 | 4.88 % | **2973** | 4317 | 25322 |
| 1024 | 1 | 0.98 % | 114325 | 85390 | **62039** |
| 1024 | 1 | 4.98 % | 113976 | **87966** | 274046 |
| 1024 | 64 | 4.98 % | 3547 | **2567** | 23512 |
| 1024 | 64 | 25.0 % | 3568 | **3035** | 114622 |

What this grid established, with the later status of each point:

1. **The ~3x flop-count ceiling for a direct subset solve was never approached.** The best
   DirectSubset result here was 1.46x (`n = 1024`, batch 64, eigenvectors). The page at the time
   blamed "cuSOLVER's full eigensolve being better optimised than our two-stage +
   `stebz`/`stein` chain". **That explanation is wrong**: the comparison never involved cuSOLVER.
   Against the corrected baseline, DirectSubset's best measured ratio is 2.40x.
2. **DirectSubset is a 3x to 5x pessimisation in eigenvalues-only mode at every measured shape.**
   There the reduction is pure cost with no back-transform to narrow. The cost model treated the
   two modes as one problem, and they are not. This still holds.
3. **`Auto` was routing into that loss.** Before this grid it sent all dense real input with
   `n > 64` and `k/n <= 25 %` to DirectSubset, including eigenvalues-only calls. The first fix
   was "Direct unless eigenvectors are wanted and `n >= 1024`". The later baseline fix added the
   batch term.
4. **Filtered's dense niche is real but narrow**: `n >= 1024` at `k/n` about 1 %, and only at
   small batch. At batch 64 Direct won even there. Its best margin is under 2x, and it is the only
   dense path with a convergence failure mode, so it stays opt-in.
5. The one prediction that held: iterative methods are hopeless on dense input at moderate `k/n`.
   LOBPCG was 10x to 100x behind Direct across the sweep. At `k/n = 0.5` it cannot run at all,
   because `[X,P,R]` needs `3 * block_vectors` columns in `n` dimensions and `ortho` refuses once
   they do not fit. The sweep skips those combinations.

A claim made before this grid should also be corrected. The page once said the syevx benchmarks
"could not run on this machine". That was inferred from a CPU-only build (`BATCHLAS_ENABLE_CUDA=OFF`),
where the `native_cpu` backend aborts with `UR_RESULT_ERROR_INVALID_NULL_POINTER` at every shape,
including the untouched LOBPCG path and the pre-existing `BM_SYEVX`. The abort is real but
specific to that backend. A CUDA build configures, compiles, and passed all 44 syevx tests at the
time.

## syevx: Filtered vs LOBPCG on CSR

`BM_SYEVX_SparseShape` and `BM_SYEVX_SparseGap` (`benchmarks/syevx_benchmark.cc`), 2026-08-06,
commit `826bc5cb`. Raw data: `benchmarks/results/syevx_sparse_shape_rtx4090.csv` and
`syevx_sparse_gap_rtx4090.csv`. The dense sweeps cannot answer this question: on sparse input,
Filtered's only rival is LOBPCG, another iterative method, not a tuned vendor eigensolver.

**A dispatcher bug had to be fixed before this sweep could measure anything.**
`syevx_select_algorithm` short-circuited all sparse input with
`if (!dense) return SyevxAlgorithm::LOBPCG;`, placed **above** the explicit-method switch. So
`params.method = Filtered` on a CSR matrix was silently discarded, even though `syevx_filtered`
is instantiated for every `MatrixFormat` and has real `spmm` branches. The first run exposed it:
configs 0 and 1 returned identical times and identical eigenvalue errors to five significant
figures. It also refuted the earlier Tier 3 claim that "the suite passes forced onto each
algorithm, which exercises Filtered on CSR": those runs had silently executed LOBPCG.
`SyevxRangeRoutingTest.SparseHonoursExplicitFilteredRequest` guards it. `Auto` still lands on
LOBPCG for sparse input.

Setup: CUDA, float, eigenvectors, 16 nonzeros per row (held constant as `n` grows),
`diagonal_boost` 10, `k` about 1 % of `n`, tolerance 1e-6, `iterations = 200`, Chol2 ortho,
guard block `max(1, neigs/4)`. The ratio is LOBPCG time / Filtered time, so **a value above 1
means Filtered wins**:

| n | batch | largest end | smallest end | Filtered err | LOBPCG err |
|---|---|---|---|---|---|
| 1024 | 1  | **8.8** | **2.7**  | 7.5e-07 | 6.4e-06 |
| 1024 | 8  | **3.8** | **3.1**  | 7.5e-07 | 6.4e-06 |
| 1024 | 64 | **2.2** | **1.5**  | 7.5e-07 | 6.4e-06 |
| 2048 | 1  | **10.9**| **4.6**  | 9.3e-07 | 7.4e-06 |
| 2048 | 8  | **2.5** | **2.0**  | 9.3e-07 | 7.4e-06 |
| 2048 | 64 | **1.4** | 0.73     | 1.4e-06 | 1.35e-05 |
| 4096 | 1  | **2.6** | **2.3**  | 1.5e-06 | 8.8e-06 |
| 4096 | 8  | **1.9** | **1.2**  | 1.5e-06 | 8.8e-06 |
| 4096 | 64 | 0.88    | 0.51     | 2.1e-06 | 1.44e-05 |

1. **On sparse input, Filtered wins by far more than its best dense margin**: up to 10.9x, ahead
   in 15 of 18 shapes. The dense "narrow niche" conclusion does not transfer, and it never had
   grounds to: on dense input Filtered competes with a direct solver, on sparse input with
   another iterative method.
2. **It is also the more accurate path, by 5x to 10x.** Both were asked for 1e-6. Filtered
   returns 7e-07 to 2e-06. LOBPCG returns 6e-06 to 1.4e-05, never reaches the tolerance, and its
   error grows with `n`. That is why the sweep reports accuracy next to every timing: without it
   the two look like an ordinary speed trade-off.
3. **The batch-64 collapse from the dense sweep reproduces.** Filtered loses at `n = 2048`,
   batch 64 (smallest end) and at both ends of `n = 4096`, batch 64. Any future sparse routing
   rule needs a batch term, as DirectSubset's does.
4. **ILU(k) never pays for itself on time**: 15 % to 45 % slower than unpreconditioned LOBPCG
   at every measured shape, formation included. It does buy accuracy (5.5e-06 vs 8.9e-06 at
   `n = 1024`). Where Filtered wins at all, it beats ILU-preconditioned LOBPCG on both axes.

`Auto` is deliberately **not** rerouted to Filtered on sparse input. Filtered is still the only
path with a convergence failure mode, the batch-64 crossover is real, and one operator family
(random sparse Hermitian with an imposed dominant diagonal) is a thin basis for a routing rule.
The next step is a second operator family (a mesh/stencil matrix, with a clustered rather than
boosted spectrum) before `Auto` changes.

### syevx: the diagonal boost gap sweep and LOBPCG stagnation

`BM_SYEVX_SparseGap`: `n = 2048`, batch 8, `k = 20`, `diagonal_boost` from 1.0 to 30.0.
Times in ms:

| config | boost 1 | 2 | 5 | 10 | 30 | eig error range |
|---|---|---|---|---|---|---|
| Filtered, largest  | 231.3 | 226.5 | 230.8 | 226.9 | 225.8 | 8.6e-07 to 1.2e-06 |
| LOBPCG, largest    | 553.4 | 553.3 | 554.4 | 560.3 | 554.0 | 5.9e-06 to 7.9e-06 |
| Filtered, smallest | 283.7 | 285.1 | 287.0 | 283.1 | 283.0 | 1.4e-06 to 2.8e-06 |
| LOBPCG, smallest   | 561.1 | 553.1 | 552.5 | 551.6 | 551.5 | 8.3e-06 to 1.35e-05 |
| LOBPCG + ILU(k)    | 692.3 | 694.8 | 627.7 | 631.8 | 633.0 | 6.2e-06 to 9.7e-06 |

**Null result: `diagonal_boost` is not a difficulty knob.** Every row is flat to within
+/-1.5 % across a 30x change in diagonal dominance, and the error columns barely move. Scaling a
*random* diagonal rescales the spectrum without closing the extremal gap, so the sweep did not
vary the quantity it was built to vary. It neither confirms nor refutes the spectral-separation
argument of [the Chebyshev-Davidson verdict](../design/syevx.md#chebyshev-davidson-consider-as-the-lobpcg-replacement-for-unpreconditioned-problems),
and should not be cited as if it did. A stencil operator, whose interior spectrum is genuinely
clustered, is what that experiment needs.

**Confirmation: LOBPCG stagnates on CSR.** LOBPCG's runtime is invariant to `diagonal_boost` and
to *which end of the spectrum is requested* (553.4 vs 561.1 ms at boost 1), while it returns
eigenvalue errors 6x to 14x above the requested 1e-6. A solver whose convergence test fired would
finish in a time that tracked problem difficulty. One whose time is constant to +/-1.5 % across
every axis and which never reaches tolerance is running its full `iterations = 200` budget every
time. That is the documented stagnation of unpreconditioned LOBPCG, observed here directly.

Filtered's times are flat too, and it sits at roughly its requested tolerance, so part of its
advantage is cheaper iterations rather than fewer. In the same fixed budget Filtered gets an
answer 5x to 10x more accurate in 2.4x less time. "Converges faster" is not what was measured
and should not be claimed.

### syevx: automatic Chebyshev filter degree

`syevx_filtered` uses the constant `kDefaultFilterDegree = 10`
(`src/extensions/syevx_filtered.cc:55`) unless the degree is pinned
(`BATCHLAS_SYEVX_FILTER_DEGREE` or `params.filter_degree`). A per-matrix degree derived from the
problem exists behind `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=1` and **cannot ship on**. RTX 4090,
float, `n = 1024`, `neigs = 8`, ratio derived / fixed-10 (above 1 means the derived degree wins):

| batch | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| NoEigenVectors | 2.18x | 2.25x | 0.70x | 0.50x | 0.48x |
| EigenVectors   | 2.20x | 2.30x | 0.69x | 0.49x | 0.48x |

The two modes agree to within 0.02x, so this is not a `jobz` effect. The mechanism is the batch
reduction, not the per-matrix formula. The degree is derived per matrix and then reduced to a
**min** across the batch, so that the GEMM shapes stay uniform. As the batch grows, the chance
that some matrix forces a low degree tends to 1, the whole batch runs at that degree, and the
outer iteration count explodes, which gives the plateau at about 0.48x. A fix would change the
batch reduction (for example a quantile instead of the min, which keeps the batched GEMM shape),
not the formula. The date of this measurement is not recorded in the source comment.

**Stale-claim warning (found 2026-09-30 while moving the comments, not re-measured).** The code
does not reduce the *derived* degree by min. Read from `syevx_filtered.cc` (the loop after the
interval kernel): the per-item derived needs are reduced by **max**, clamped to `[10, 40]`, and
then trimmed by the **min** of the per-item precision caps, which applies with or without the
derivation (see
[the degree cap](../design/syevx.md#syevx-filtered-normalising-the-filter-and-capping-its-degree)).
Since the derived degree is never below the fixed 10, "some matrix forces a low degree" cannot be
what makes auto lose at batch >= 4; a batch-max of the *needs*, which runs every matrix at the most
demanding item's degree, fits the code better. The mechanism above should be treated as
unverified until a per-iteration degree trace at batch 8 settles it.

The constant 10 itself has never been tuned on this hardware. It is a mid-range guess: too low
and each outer iteration barely separates the spectrum, too high and the extra matvecs are
wasted because the block is reorthogonalized anyway.

## syevx implementation status by tier

The tier plan is in @ref design_syevx. Every tier except Tier 5 exists.

### syevx Tier 0: dispatch and the Direct path

- `SyevxAlgorithm {Auto, Direct, DirectSubset, Filtered, LOBPCG}` on `SyevxParams::method`, with
  the `BATCHLAS_SYEVX_ALGORITHM` override. The environment **wins** over `params.method`
  (as `BATCHLAS_SYEV_ROUTE` does for syev; at the time, the since-retired `BATCHLAS_SYEV_PROVIDER`), so a whole application can be forced onto one algorithm.
  Tests that pin an algorithm skip under a conflicting override instead of failing, which keeps
  "run the suite under every algorithm" sweeps meaningful.
- `syevx_select_algorithm` is deterministic in its inputs, so `syevx` and `syevx_buffer_size`
  always agree. An unimplemented or unsupported choice degrades to its nearest implemented
  neighbour (for example DirectSubset on complex input degrades to Direct), so `Auto` never fails.
- `syevx_direct` (`src/extensions/syevx_direct.cc`): `syev` on a private copy of `A` plus a
  selection kernel. `A` is left unmodified. LOBPCG moved to `src/extensions/syevx_lobpcg.cc` and
  `syevx.cc` is only the dispatcher.
- Preconditioner-argument validation moved from `syevx_lobpcg` to the dispatcher. It had been
  equivalent only while every path led to LOBPCG. Once dense input routed to Direct/DirectSubset,
  an illegal combination on a dense matrix reached a solver that ignored it instead of rejecting it.

### syevx Tier 1: stebz and stein

- `stebz` (`src/extensions/stebz.cc`): batched tridiagonal bisection. All/Index/Value ranges,
  either sort order, no global scratch. One work-group per batch item, one work-item per wanted
  eigenvalue. The Sturm recurrence clamps \f$|q|\f$ below `pivmin` as LAPACK `dlaebz` does. Without
  that guard an underflowing pivot yields an infinity and the count stops being monotone in
  \f$x\f$, which breaks the bisection invariant.
- `stein` (`src/extensions/stein.cc`): batched inverse iteration. Phase 1 is one work-item per
  vector doing a tridiagonal LU with partial pivoting (LAPACK `dgttrf`/`dgttrs`). Phase 2 is
  modified Gram-Schmidt within clusters of eigenvalues closer than `ortho_threshold * ||T||`.
- **Testing note.** The obvious clustered-spectrum test, a block-diagonal matrix with doubled
  eigenvalues, does *not* test reorthogonalization: its degenerate pairs have disjoint supports
  and come out orthogonal for free. The test uses a Wilkinson matrix \f$W^+_{2m+1}\f$ instead,
  whose near-degenerate pairs share support. With `ortho_threshold` forced to 0, columns 4 and 5
  have a dot product of 8.6e-4, so the test has teeth.
- Tests at landing: `stebz_tests` (10) and `stein_tests` (8), float and double.

### syevx Tier 2: DirectSubset

`syevx_direct_subset` (`src/extensions/syevx_direct_subset.cc`): `sytrd_sy2sb` -> band-to-tridiagonal
-> `stebz` -> `stein` -> back-transforms narrowed to `k` columns. Real scalar types and dense
input only. `syevx_direct_subset_supported<T, MFormat>()` gates it. Shared helpers were lifted
out of `syev_two_stage.cc` into `src/extensions/two_stage_common.hh` rather than duplicated.

- **As first landed**, eigenvector mode forced `kd = 1`, so the Givens `sytrd_sb2st` left no Q2
  to apply (see the `kd = 1` correction in @ref design_syevx). That made stage 1 an unblocked
  BLAS-2 reduction, the dominant cost done the slow way.
- **Now**: once `sytrd_sb2st_hh` retained Q2 (PR #45, applied to this path in commit `95033223`,
  2026-08-03), eigenvector mode reduces at the tuned band width and applies Q2 to the `k`
  selected columns via `unmqr_hb2st` before the Q1 `ormqr`. Both back-transforms run on `k`
  columns, not `n`. Eigenvalue-only solves keep the cheaper Givens chase, which needs no
  back-transform. Verified correct at `BATCHLAS_SYEV_TWO_STAGE_KD` in {1, 2, 8, 32, 64}.
- **Bug found:** `apply_phase_rows` computed its column count from `z.rows()`, which assumes a
  square matrix. That holds in `syev_two_stage`, which only passes an n x n block. On the subset
  path's n x k block it wrote out of bounds and corrupted neighbouring batch items. The
  eigenvectors came back normalized but failed the residual on every column, while eigenvalues
  stayed correct. Fixed to use `cols()`.
- Tests check both orderings x both `JobType` values, with residuals against the *original* A
  (which is what validates the back-transform) plus orthonormality.

### syevx Tier 3: Filtered

`syevx_filtered` (`src/extensions/syevx_filtered.cc`): spectral bounds -> Chebyshev filter ->
`ortho` -> Rayleigh-Ritz, looping until the wanted residuals converge. Dense and CSR, real and
complex. It needs no preconditioner and no factorization, and unlike the ILU(k) path it is not
restricted to the smallest eigenpairs. Two numerical findings, both caught by tests:

1. **Normalise the scaled recurrence at the extreme Ritz value, not at the Gershgorin bound.**
   Gershgorin overestimates the spectral radius of a random symmetric matrix by roughly
   \f$O(n)\f$ against the true \f$O(\sqrt{n})\f$. Dividing by \f$T_m\f$ of a point that far outside
   the spectrum underflows the block to zero, and orthogonalizing a zero block yields NaN.
2. **The filter degree must be bounded by precision.** A sharp filter drives every column toward
   the dominant direction. Once the amplification ratio between the most- and least-wanted
   directions exceeds what the working precision can hold, the block is numerically
   rank-deficient and the Cholesky-based `ortho` returns NaN. The degree is capped per iteration
   from the Ritz values, then reduced across the batch so the recurrence length stays uniform.
   The first version waived the cap when the least-wanted direction fell inside the damped band,
   which is the worst case, where growth is the full \f$\cosh(d\,\mathrm{acosh}(y_{far}))\f$.
   A ratio budget of \f$\epsilon^{-1/2}\f$ was measured to be still too generous;
   \f$\epsilon^{-1/4}\f$ holds.

The filtered tests assert the eigenvalues against a reference `syev`, not only residuals. A
self-consistent pair proves nothing about whether it is one of the *wanted* pairs, which is how a
filtered solver fails. `Filtered` is reachable only explicitly (`params.method` or
`BATCHLAS_SYEVX_ALGORITHM=filtered`).

**Known cost:** the convergence test reads a device flag on the host once per outer iteration.
That is the same defect as [LOBPCG's host synchronization](#lobpcg-host-synchronization-only-at-convergence-checks),
but it costs one sync per outer iteration, not per matvec.

The derivations behind both numerical findings, and the degree precedence, are in
[the Filtered design notes](../design/syevx.md#syevx-filtered-the-scaled-chebyshev-filter).

### syevx: the parallel Gershgorin bounds kernel

`syevx_filtered` starts from Gershgorin bounds: row \f$i\f$ contributes
\f$[a_{ii} - \sum_{j \ne i}|a_{ij}|,\ a_{ii} + \sum_{j \ne i}|a_{ij}|]\f$, and `lo`/`hi` are the
min/max over rows. They are cheap, need no matvec, and only have to be conservative: a loose
interval costs filter sharpness, never correctness.

The legacy kernel ran this as **one work-item per matrix** walking all \f$n^2\f$ elements with a
column-major stride, so every load was its own sector. It was measured at **80.2 % of the whole
solve at `n = 1024`**, and independent of batch. (RTX 4090; the date, dtype and batch of that
profile are not recorded in the source comment.)

The shipped kernel gives each work-item one row (`kBoundsWG = 64` rows per work-group), so at a
fixed `j` a work-group reads a contiguous run of a column: fully coalesced. Row-groups per matrix
scale as `n / 64`, so parallelism grows linearly while work grows as \f$n^2\f$. The per-(matrix,
row-group) partial `[min, max]` goes to a second reduce kernel.

- **Bit-identical, not merely equivalent.** The inner `j` loop stays serial and ascending, so
  each row's radius is accumulated in exactly the legacy order and the per-row endpoints are
  bit-identical. Only the outer min/max is re-associated, and min/max is exact under any
  association. Keep that loop order if the kernel is touched.
- `BATCHLAS_SYEVX_BOUNDS_LEGACY=1` restores the serial kernel for A/B.
- **Workspace lockstep.** The partials (`2 * ceil(n/64) * batch` reals) are a `UnifiedVector`
  outside the `BumpAllocator`, the same pattern as `converged` and `degree_cap`, so the sequence
  of pool allocations, and `syevx_filtered_buffer_size`, are byte-for-byte unchanged. The filter
  degree allocates nothing either: the recurrence reuses `Y`/`Yprev` whatever its length.

### syevx: pre-existing bugs found while building the tiers

All confirmed present on `main` before the change that exposed them:

1. **LOBPCG workspace missed the restart size.** The projected `syev` was sized for
   `block_vectors` and `3 * block_vectors`, but the restart iteration solves a
   `2 * block_vectors` problem. The SYEV provider is chosen per size (CTA only up to `n = 32`), so
   the largest matrix does not necessarily need the largest workspace. It threw
   `syev: insufficient workspace for chosen provider` for example at `n = 60, neigs = 3,
   extra_directions = 10`, which is what made `SyevxOperationsTest.RandomMatrix` fail on main.
2. `SyevxOperationsTest.ComplexShiftInverToeplitzEigenpairs` sized its workspace from a CSR view
   but solved a dense matrix. This was latent until the two could select different algorithms.
3. `syevx_buffer_size`'s `C_p` stand-in was built with only `(data, rows, cols, ld)`, so
   `batch_size` defaulted to 1 and the orthogonalization workspace was sized for one item. It
   became a hard allocation failure when the guard-vector default made `n = 64,
   block_vectors = 20` the default shape.
4. The single-matrix `ortho(X)` was never included in the workspace maximum at all.

## LOBPCG hardening in syevx

The LOBPCG defect list and its fixes (Tier 4). Unless stated, iteration counts were measured on
NETLIB/CPU with `BATCHLAS_SYEVX_CHECK_EVERY=1`, so the counts are exact.

### LOBPCG: host synchronization only at convergence checks

Every iteration used to end with `residual_evt.wait_and_throw()` and a host read of unified
memory: about 30 forced pipeline drains per solve, which can dominate at small `n` and large
batch. The pipeline is now drained only when a host reader needs that iteration's data: the
convergence check (every `check_every`, default 4, `BATCHLAS_SYEVX_CHECK_EVERY`) or
instrumentation that could not be staged on the device. Overshooting the stopping point by up to
`check_every - 1` iterations is far cheaper than the drains it replaces. Correctness verified at
`check_every` in {1, 4, 16}. Capturing the iteration body in a SYCL graph was the proposed
follow-on and has not been done.

### LOBPCG: device-staged instrumentation

The histories used to be filled by a host loop over `batch x neigs` after a forced drain every
iteration, so turning instrumentation on changed the performance it measured. Now the residual
kernel stores into a compact staging buffer carved from the same pool, and one host pass after
the loop scatters into the caller's spans.

- **Constraint:** `SyevxInstrumentation`'s spans and `iterations_done` are caller-supplied and
  **not guaranteed device-accessible**. The Python binding passes a plain `std::vector<int32_t>`
  (`python/batchlas/bindings/ops_spectral.cc`). So no kernel writes them. The kernel touches only
  pool memory, and the caller's buffers are written only from the host.
- `convergence_rate_history` is not staged. It is a ratio against the previous sample, so the
  scatter computes it from the staged best-residual series.
- A caller who fills in `params.instrumentation` only *after* `syevx_buffer_size` gets a
  workspace with no room for staging. `syevx_lobpcg` notices and falls back to the host-read path
  instead of throwing. `BATCHLAS_SYEVX_INSTR_HOST=1` forces the same fallback, which is how
  `tests/syevx_tests.cc` A/Bs the two paths.
- The values match the host path only up to the run-to-run noise of the residual kernel's
  work-group atomics: about 6 to 7 significant figures in float, which is what two runs of the
  unmodified build already differed by.

### LOBPCG: single-pass residual kernel

It was `2 * neigs` sequential group reductions, 128 work-group barriers at `neigs = 64`. Now one
pass forms R and accumulates the per-column norms together, flushing a running partial on column
change. The block is column-major, so a work-item stays in one column for long runs, which turns
one atomic per element into roughly one per thread per column.

### LOBPCG: backward-stable convergence criterion

It was \f$\|r\| / (\|x\|\,|\lambda|)\f$, which collapses as \f$\lambda \to 0\f$: a good eigenpair
with a near-zero eigenvalue could never converge. It is now
\f$\|r\| / (\|Ax\| + |\lambda|\,\|x\|)\f$, which stays bounded away from zero for any nonzero A
(Duersch et al.).

### LOBPCG: soft locking by column masking

Variant (a), column masking, is implemented and **defaults to off**. Enable it with
`BATCHLAS_SYEVX_SOFT_LOCK=1` (note that this knob's parser is inverted, see
`include/batchlas/settings.hh`). Variant (b), a batch-wide staircase, was not attempted.

The residual kernel writes a per-column flag `col_converged[bid][j]`
(`residual_j <= 0.1 * tol`, where 0.1 is `BATCHLAS_SYEVX_LOCK_FACTOR`). The non-obvious part is
that **`ortho` mixes columns**: Householder QR and the Chol/Chol2 triangular solve both replace
R by \f$R\,U^{-1}\f$ with U upper triangular, so column j of the result spans
\f$r_0 \ldots r_j\f$. Neither one-sided placement works:

- Masking **only after** `ortho` deletes content the unconverged columns were orthogonalised
  against, and they lose their correction directions. Measured (`n = 128, k = 8, tol = 1e-4`):
  convergence froze at the first locking iteration (residuals stuck at their iteration-13 values)
  and went NaN by iteration 26, where the unlocked run converged at iteration 14.
- Masking **only before** `ortho` makes \f$R^T R\f$ singular, and the Cholesky algorithms produce
  NaN from `potrf` rather than raising.

Implemented: **before** `ortho`, each converged column is overwritten with a deterministic
pseudo-random vector scaled to the largest surviving residual column. That keeps full rank and a
well-scaled Gram matrix, and cannot produce an all-zero block. **After** `ortho`, those columns
are zeroed so the filler never reaches `AR` or `StAS`. A masked column makes its row and column
of `StAS` exactly zero, which would add a spurious eigenvalue 0. That is fatal for
`find_largest = false` on a positive-definite A, where the zeros sort below the wanted spectrum.
So a sentinel of \f$\pm(\|StAS\|_F + 1)\f$ is planted on each locked diagonal. It is provably
outside the real spectrum and sorts to the unwanted end.

The locking threshold must be stricter than the tolerance. A masked column is not frozen: its
Ritz vector is still recombined every iteration. So a column at exactly `tol` drifts back above
it, unlocks, and oscillates. Locking at exactly `tol` made the final residual worse on 3 of 8
cases of the ILU(k) convergence sweep in `tests/iluk_tests.cc`.

**Measured: no win, small cost.** Iterations to converge, CPU build:

| case | off | on (f=0.1) |
|---|---|---|
| n128 b4 k8 largest tol1e-4 | 16 | 16 |
| n128 b4 k16 largest tol1e-5 | 14 | 14 |
| n256 b4 k16 largest tol1e-5 | 19 | 20 |
| n256 b8 k32 largest tol1e-5 | 14 | 14 |
| n256 b4 k16 largest tol1e-5 (dense off-diag) | 19 | 20 |
| n512 b2 k16 largest tol1e-5 | 28 | 28 |

1. **It saves no flops.** Shapes are fixed, so a masked column is still multiplied by A and
   still occupies its row and column of `StAS`. Only variant (b) recovers that.
2. **The regime where it should help was unreachable in this measurement.** `Backend::NETLIB`
   forces `OrthoAlgorithm::Householder` (`src/extensions/ortho.cc`), which already handles a
   near-null column gracefully. The conditioning argument is about Chol2, the GPU default. **A GPU
   A/B under Chol2 is still owed** before the default could flip.

#### LOBPCG: why the batch-wide staircase was not attempted

Variant (b) shrinks the active block width (for example in steps of 8) only when *every* batch
member has converged that many columns. That keeps shapes uniform, and it is where the speedup is
(`A*R` and `S^T A S` widths drop from `3k` toward `k`). It is not a local change:

- **The active blocks stop being contiguous.** Soft locking keeps the converged X columns and
  drops only their P/R counterparts, so the basis becomes `[X(k) | P(k') | R(k')]`, no longer the
  single `S({0,n},{0,Nvecs})` slice every gemm in the loop takes. Either the loop grows to 3x3
  block gemms (9 calls where there is now 1, four times over), or X/AX/P/AP get permuted on each
  shrink. Permuting requires the converged set to be a contiguous prefix, which it is not: at the
  first locking iteration of the `n = 128, k = 8` case, columns 0, 1, 2, 4 were locked and 3, 5,
  6, 7 were not.
- **`Nvecs` is computed in three places that must agree**: the host loop, the residual kernel
  (`num_eigvals = it < 2 ? (it+1)*block_vectors : 3*block_vectors`) and the instrumentation
  block. The residual kernel uses it for the `find_largest` reverse indexing into `lambdas`, so
  one out of step silently returns wrong eigenvalues.
- **Workspace sizing is not monotone in `Nvecs`.** `syev` picks its provider from the shape, so
  the `3k` figure does not bound the `2k` one (the restart bug above). A staircase of step 8 adds
  `k/8` projected shapes, each of which must be covered in the runtime max *and* in
  `syevx_lobpcg_buffer_size`.
- It reintroduces a host-side shape decision inside the loop, which the host-synchronization fix
  removed. That is affordable only if folded into the periodic convergence check.

Prerequisite before attempting it: collapse the three `Nvecs`/`num_eigvals` computations into one
value, and name the hard-coded block offsets (`k`, `2k`).

### LOBPCG: guard vectors by default

`extra_directions == 0` now means "choose one": `max(2, neigs/4)`, the same convention
`filter_degree` uses. An explicit width still wins. Iterations to converge, `find_largest`,
tol 1e-5, batch 4, via `SyevxInstrumentation::iterations_done`:

| n / neigs | no guard | guard | change |
|---|---|---|---|
| 64 / 4   | 32 | 20 | -38 % |
| 64 / 8   | 36 | 18 | -50 % |
| 128 / 4  | 49 | 26 | -47 % |
| 128 / 8  | 39 | 25 | -36 % |
| 128 / 16 | 32 | 18 | -44 % |
| 256 / 4  | 78 | 39 | -50 % |
| 256 / 8  | 50 | 33 | -34 % |
| 256 / 16 | 48 | 26 | -46 % |

A third to a half of the iterations, for a cost linear in the extra width. Benchmarks that
compare LOBPCG against another path use a guard block too, because omitting it would handicap
LOBPCG for a reason unrelated to the comparison.

### LOBPCG: preconditioner staging copy removed

`R_contiguous`, a full `n x k x batch` staging buffer that repacked R before `iluk_apply`, is
gone: `iluk_apply` indexes as `b*stride_ + col*ld_`, so it reads R's strided slice directly. This
removed a copy per iteration, its allocation, and the two `wait_and_throw` drains around it. The
separate `R_preconditioned` destination stays: the forward solve writes into `out` as its
temporary while still reading `rhs`, so aliasing them would corrupt the solve.

### LOBPCG: column-reversal kernels removed

Two batch-wide launches per iteration only reversed the column order of a small block for
`find_largest`. The three proposed fixes were all unavailable:

- (a) a reversed view: `MatrixView`/`KernelMatrixView` slicing only offsets the base pointer and
  the constructor clamps `ld_(ld > 0 ? ld : rows)`, so a negative-stride view cannot be built,
  and vendor `gemm` requires `ld >= max(1, rows)` anyway;
- (b) have `syev` emit descending order: the public `syev` takes no ordering argument, and the
  vendor path is permanently ascending, so this would only relocate the permutation;
- (c) fold it into the GEMM: `gemm` has no permutation argument.

What landed: X stays in `syev`'s ascending order, since nothing in the iteration depends on the
column order of a basis. Only the residual/W indexing cares (column j pairs with
`lambda[eig_offset + j]`, `eig_offset = num_eigvals - block_vectors` when `find_largest`), plus
presentation order. The largest-first flip rides on the `X_best` snapshot the residual kernel
already writes. One cold-path launch remains, for `params.iterations == 0`, where no snapshot
happens.

**Testing gap this exposed:** every earlier LOBPCG test asked for `NoEigenVectors`, so nothing
checked that returned column j belongs to eigenvalue j. `SyevxLobpcgVectorsTest` checks per-pair
residuals against the original A. With the snapshot permutation forced to the identity, it fails
`Largest` and passes `Smallest`, so it has teeth.

### LOBPCG: X-only fill and power-iteration start

- The random fill now covers only the X block, reproducing `fill_random`'s linear index, so the
  start is bit-for-bit unchanged with a third of the traffic.
- A block power-iteration start, \f$X \leftarrow \mathrm{ortho}(AX)\f$
  (`SyevxParams::init_power_iterations`, default 4, `BATCHLAS_SYEVX_INIT_POWER`), runs before
  the first Rayleigh-Ritz when `find_largest` is true. NETLIB/CPU, double, dense random
  Hermitian, batch 4, tol 1e-5, mean iterations over 3 seeds:

| n | k | p=0 | p=1 | p=2 | p=4 | p=8 |
|---|---|-----|-----|-----|-----|-----|
| 64 | 4 | 22.67 | 20.67 | 20.00 | 18.67 | 16.00 |
| 64 | 16 | 8.33 | 7.00 | 6.00 | 4.67 | 3.00 |
| 128 | 16 | 19.00 | 17.33 | 16.00 | 15.00 | 12.67 |
| 256 | 8 | 36.00 | 35.33 | 34.33 | 31.00 | 27.33 |
| 256 | 16 | 27.67 | 26.33 | 25.33 | 24.00 | 21.00 |

12 % to 44 % fewer iterations at `p = 4`, with wall time moving the same way (`n = 64, k = 16`:
23.4 -> 11.3 ms). `p = 8` was better still on this family. 4 was kept as the default for margin:
an over-compressed block is rank-deficient in floating point, and the Cholesky-based `ortho` then
returns NaN rather than an error.

**Negative result:** the same start for `find_largest = false`, using \f$\sigma I - A\f$ with a
Gershgorin \f$\sigma\f$, was flat (23.00 -> 21.67 at `n = 64, k = 4`; unchanged at the other
shapes), because \f$(\sigma-\lambda_1)/(\sigma-\lambda_{k+1}) \approx 1\f$ for a wide spectrum. It
is not implemented, and the steps are skipped when `find_largest` is false. Warm-starting from a
previous solution across a sequence of related problems is still open.

### LOBPCG: Jacobi preconditioners

`SyevxParams::preconditioner_type` selects a family (`SyevxPreconditioner`; `Auto` keeps the old
behaviour). `diag(A)` is extracted once (for CSR, by a per-row search) and the apply is one
in-place kernel over R. Measured iteration counts: fp32, `BATCHLAS_SYEVX_CHECK_EVERY=1`,
`n` in {64, 128, 256, 512}, `k` in {2, 8, 16}. `DISABLED_IterationSweep` in
`tests/syevx_tests.cc` reproduces them.

- **`Jacobi`** = \f$\mathrm{diag}(A)^{-1}\f$. Like ILU(k) it approximates \f$A^{-1}\f$, so
  `find_largest` is rejected: forcing it turned 21 to 47 iterations into 127 to 300. Where the
  diagonal is dominant and positive it is worth **2.1x to 7.3x** (for example 70 -> 11 at
  `n = 512, k = 2`). Where `diag(A)` is not positive definite it is not a valid preconditioner, so
  a per-batch-item test disables it there. On random symmetric input it is therefore an exact
  no-op.
- **`JacobiShifted`** = \f$(\mathrm{diag}(A) - \lambda I)^{-1}\f$, legal at both ends because the
  shift tracks the wanted Ritz value. It **degenerates exactly where the unshifted form wins**: as
  \f$\mathrm{diag}(A) \to A\f$ it becomes the exact inverse of \f$A - \lambda I\f$, so
  \f$Tr \to x\f$ and the new direction is annihilated by orthogonalization against X. Measured
  0.85x to 1.2x on random symmetric input and 0.2x to 0.9x on graded input. It is kept only
  because it is the one Jacobi form legal for `find_largest`.
- Neither is selected by `Auto`. On a constant diagonal, `JacobiShifted` is a per-column scalar
  and LOBPCG is invariant to per-column scaling, so its iteration counts on `csr-tridiag` are
  identical to the unpreconditioned ones. That is the sharpest check that the shift is indexed to
  the right Ritz value per column (`SyevxJacobiIterations.ShiftedIsANoOpOnConstantDiagonal`,
  which must run without soft locking because the filler vectors break the exact equality).
- The Chebyshev "preconditioner" (a low-degree \f$p(A)\f$ applied to R) was not attempted.

### LOBPCG: the projected syev provider

The projected problem is a batched `3k x 3k` dense eigensolve per iteration (`192 x 192` at
`k = 64`). The sweep found that `syev`'s `Auto` let CTA claim every `n <= 32`, so the projected
solve never reached the vendor. At `n = 30`, batch 8, float, with eigenvectors: CTA
229.6 us/call against cuSOLVER 103.7 us/call (2.21x), and nsys attributed 29.4 % of all LOBPCG
GPU time to that solve. The fix is in the `syev` dispatcher, so LOBPCG and Filtered both get it:
`syev_prefer_vendor_over_cta` (`include/batchlas/blas/functions/syev.hh`) sends eigenvector-mode
CUDA solves with `n > BATCHLAS_SYEV_CTA_MAX_N` to the vendor. The default was 16 when this
was recorded. It was later measured through LOBPCG (1.10x to 1.15x at batch 8) and ships
**off** (default 32, with a type-dependent `syev_cta_max_n_default_for<T>()`), because it flips
a marginal case in `ILUKTests.SyevxInstrumentationAndPreconditioner`. The measurement and the
open owner decision are in
[syev: the LOBPCG projected-solve knob](syev.md#syev-the-lobpcg-projected-solve-knob).
`BATCHLAS_SYEVX_PROJECTED_VENDOR` forces the vendor for the
projected solve. It changes the workspace size, so it must not change between a buffer-size query
and the call. The original plan also named `syev_jacobi_cta` as a candidate provider for the
small-`k` end of the projected solve. That was never measured.

### LOBPCG: the accuracy pin on the projected syev

**Since 2026-10-09 (deep sm_120 tables).** The tuned `syev` tables rank families by time
only, and the deep sm_120 run ranks cuSOLVER (`vendor`) or `jacobi` first for many small
eigenvector-mode cells. LOBPCG cannot afford that trade. On threadripper02 (RTX PRO 6000,
sm_120), float, n = 16, cuSOLVER's eigenvalues are about 1.9e-5 off, against 2.7e-6 for the CTA
solver. With the projected solves on `vendor` or `jacobi`,
`ILUKTests.SyevxInstrumentationAndPreconditioner` (`iluk_k3`, case `d0.06_b0.5`) stalls at
a residual of 5.2e-6 against its 4.2e-6 target. It passes on CTA.

`projected_syev_pin` (`src/extensions/syevx_lobpcg.cc`) therefore pins every projected solve
(XtAX, StAS and its restart size) to `cta_fused` when the CTA solvers can take the shape
(n <= 32, a GPU with 32-wide sub-groups). The buffer-size query uses the same pin, so the
workspace and the run branch identically. `cta` has the same `can_run` as `cta_fused`, so it is
never a fallback rung. Above n = 32 the table decides. An explicit route (`BATCHLAS_SYEV_ROUTE`
or an outer `select::ScopedPin`) still wins, which keeps the five
`Pinned/SyevxProjectedSyevRouteTest.*` cases exercising each family.
`SyevxProjectedSyevPin.CtaFusedAtOrBelow32EvenWhenTheTableRanksTheVendorFirst` guards the pin
through the selection trace, under a synthetic table that ranks the vendor first.

## syevx: open measurement debts

- **`BM_SYEVX_RangePosition` has not been run.** It is the one benchmark point that checks that
  spectrum position does not enter either dense path's cost (@ref design_syevx_range).
- **Soft locking under Chol2 on GPU.** The only A/B ran on CPU, where Householder is forced.
- **The `syev` CTA-vs-vendor crossover** for the projected solve: when first recorded, `n = 30`
  was the only measured point in `1 <= n <= 32`, and the eigenvalues-only case at `n <= 32` was
  left on CTA unmeasured. The later LOBPCG-level sweep and the decision it is blocked on are on
  the [syev page](syev.md#syev-the-lobpcg-projected-solve-knob).
- **Profile Filtered's large-batch collapse before tuning it** (an unmeasured idea from the
  SYEV ideation notes). Filtered wins at `n = 1024`, `k/n` about 1 %, batch 1 (62 ms against
  114 ms for the blocked Direct of the superseded grid) and loses at batch 64 (23.5 ms against
  3.5 ms per matrix, 6.7x). A path that wins when the GPU is starved and loses when it is
  saturated suggests that the filter (two GEMMs per Chebyshev step) is not the cost and that the
  `ortho` / Rayleigh-Ritz tail is. If so, the same fix would help Filtered and LOBPCG, which both
  spend their non-GEMM time in `ortho` plus a small projected `syev`.
- **A second sparse operator family** (mesh/stencil) before `Auto` could route sparse input to
  Filtered, and a spectral-gap experiment that actually closes the gap.
- **The Filtered degree**: `kDefaultFilterDegree = 10` is untuned, and the auto-degree batch
  reduction needs a design change before it can ship. Its loss mechanism is itself unverified
  (see the stale-claim warning in
  [the automatic degree section](#syevx-automatic-chebyshev-filter-degree)).
- **The projected-syev provider sweep** (SYEVX_PLAN s7.11), listed as "now possible" in s13 and
  still undone as a sweep: only the `n = 30` point and the LOBPCG-level CTA-max-n A/B exist (see
  [the projected syev provider](#lobpcg-the-projected-syev-provider)), and `syev_jacobi_cta` as a
  provider for the small-`k` end was never measured.
- **Above `n = 2048`** the DirectSubset gate is extrapolated, not measured.
- **Not done in LOBPCG:** the batch-wide staircase, warm starts, the Chebyshev preconditioner,
  and capturing the iteration in a SYCL graph.
