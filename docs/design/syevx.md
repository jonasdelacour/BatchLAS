# syevx: design of the partial symmetric eigensolver {#design_syevx}

> **Status:** current · written 2026-08-01, corrected in place where measurement disagreed.

`syevx` returns \f$k\f$ eigenpairs of a symmetric or Hermitian matrix, by extreme end or by an
index or value range. It dispatches to one of four algorithms. This page gives the algorithms,
the routing rule, the cost model, the verdict on each candidate, and the rules the dispatcher
enforces. Measured crossovers are on @ref perf_syevx. Range semantics are on @ref design_syevx_range.

## syevx algorithms

| Algorithm | Input | Method | Chosen by Auto when |
|---|---|---|---|
| `Direct` | dense; all types, ranges and `jobz` | `syev`, then select | \f$n \le 64\f$, or eigenvalues only |
| `DirectSubset` | dense | band reduction, `stebz`, `stein`, back-transforms on \f$k\f$ columns | eigenvectors at large \f$n\f$ and batch |
| `LOBPCG` | CSR (dense on explicit request) | batched LOBPCG, optional ILU(k) or Jacobi preconditioner | sparse input, extremal ranges only |
| `Filtered` | CSR or dense | scaled Chebyshev filter with Rayleigh-Ritz | never; explicit request only |

On dense input with eigenvectors, Auto picks `DirectSubset` when \f$n \ge 1024\f$ and
\f$n \cdot \mathrm{batch} \ge 128 \cdot 1024\f$, and `Direct` otherwise. The gates are
`kSyevxSubsetMinN` and `kSyevxSubsetMinWork` in `src/extensions/syevx.cc`. The rule has no
\f$k/n\f$ term: the `DirectSubset` advantage is flat in \f$k\f$ from 0.8 % to 25 % of the spectrum.

Routing gave most of the practical gain (3.0x to 3.7x geometric mean). `DirectSubset` adds up to
2.4x, at large \f$n\f$ and batch only.

## syevx: the flop-count cost model

Leading-order flops for real symmetric \f$n \times n\f$ with \f$k\f$ wanted eigenpairs. These are
estimates; measurement did not confirm their performance predictions.

| Path | Flops |
|---|---|
| Full, one-stage, with vectors | \f$\approx 4.7n^3\f$ (sytrd \f$\tfrac43 n^3\f$, stedc \f$\tfrac43 n^3\f$, ormtr \f$2n^3\f$) |
| Full, two-stage, with vectors | \f$\approx 6.7n^3\f$; the back-transform is applied twice |
| `DirectSubset` | \f$\approx \tfrac43 n^3 + 4n^2k\f$ |
| Chebyshev subspace iteration, dense, degree \f$d\f$ | \f$4dn^2k\f$ per outer pass |
| LOBPCG, dense, \f$it\f$ iterations | \f$2\,it\,n^2k + O(it\,nk^2)\f$ |

- The subset ceiling is about 3x as \f$k/n \to 0\f$, not 10x, because tridiagonalization stays
  \f$O(n^3)\f$. Measured best: 2.40x. Eigenvalues-only subset solves lost 3x to 5x.
- Iterative methods on dense input break even at \f$k < n/(3d)\f$ (about 2.2 % of \f$n\f$ for
  \f$d = 15\f$). Measured, they lose at moderate \f$k/n\f$.

## syevx for batches of small matrices

### syevx small n: why Direct is the answer

For \f$n \le 64\f$ a CTA-resident solver is bound by occupancy, shared-memory traffic and sweep
count, not flops. A subset kernel would land within noise of `syev_cta`, so `syevx` calls `syev`
and selects for \f$n \le\f$ `kSyevxSmallN` (64). The path exists for API compatibility, not speed.

Not built:

- A fused filtered CTA solver (hold A in shared memory, \f$d\f$ Chebyshev steps, Rayleigh-Ritz in
  registers). Estimated 3x to 8x at \f$n = 32\f$, \f$k = 2\f$ if the spectrum is well separated. Gate
  on a measured gap estimate first.
- `stebz` as an eigenvalues-only alternative to `stedc` for small \f$n\f$. Not measured.

## syevx for batches of medium matrices

This is where `DirectSubset` applies: \f$n\f$ from about 128 to 4096, batch 8 to 1000.

| Stage | Component |
|---|---|
| dense to band | `sytrd_sy2sb` |
| band to tridiagonal (keeps Q2) | `sytrd_sb2st_hh` |
| subset eigenvalues | `stebz` |
| subset eigenvectors | `stein` |
| back-transform through Q2 | `unmqr_hb2st`, on \f$k\f$ columns |
| back-transform through Q1 | `ormqr_blocked`, on \f$k\f$ columns |

- The tridiagonal subset stage is off the critical path: at \f$n = 1024\f$, \f$k = 64\f$, bisection
  costs about 2.6 MFlop against 1.4 GFlop for `sy2sb`. It must be correct and batched, not fast.
  That favours bisection with inverse iteration over MRRR.
- `stein` runs one work-item per vector. Each solves \f$(T - \lambda I)x = b\f$ with a tridiagonal LU,
  which is serial in \f$n\f$. Vectors in a cluster are reorthogonalized by modified Gram-Schmidt.
- Batch items converge at different rates, so per-matrix adaptivity (variable filter degree or block
  width) breaks the uniform shapes batched GEMM needs. Take the max or min over the batch, or bucket.
- Memory: LOBPCG allocates four \f$n \times 3k \times \mathrm{batch}\f$ buffers, about 800 MB at
  \f$n = 4096\f$, \f$k = 64\f$, batch 32, fp64. `DirectSubset` needs about an order of magnitude less.

## syevx vendor gap

- `cusolverDnXsyevdx` and MAGMA `syevdx`: subset, not batched.
- `cusolverDnXsyevBatched`: batched, full spectrum.
- `cusolverDnSsyevjBatched` and `gesvdjBatched`: batched, full spectrum, \f$n \le 32\f$.

No vendor routine is a batched subset eigensolver. That gap is the product case for `DirectSubset`.

## syevx algorithm verdicts

### Bisection and inverse iteration, stebz and stein: build

Bisection counts eigenvalues below a shift from Sturm sign changes. Each eigenvalue is independent,
so it maps to one lane with no communication. Inverse iteration's weakness, loss of orthogonality
in tight clusters, costs \f$O(nk^2)\f$ to repair. With \f$k \ll n\f$ that does not bind. Built as
`stebz` and `stein`.

### MRRR (stemr, dsyevr): do not build

\f$O(n^2)\f$ with no reorthogonalization, but hard to implement correctly. For medium \f$n\f$ the saving
is negligible against the \f$O(n^3)\f$ tridiagonalization. LAPACK's `dsyevr` falls back to `dstebz` and
`dstein` on some subset paths.

### Chebyshev-filtered subspace iteration (ChFSI, ChASE): built as syevx_filtered

All work is GEMM (dense) or SpMM (sparse). It needs a spectral bound estimate, reasonable separation,
and a degree uniform across the batch. It is the strong path on sparse input, not dense.

### Chebyshev-Davidson: consider as the LOBPCG replacement for unpreconditioned problems

Not built. The polynomial filter serves as the expansion operator, so no preconditioner is needed.
`Filtered` covers much of the same ground. Unpreconditioned LOBPCG stagnates, and that is the
default configuration unless the caller supplies ILU(k) or Jacobi.

### LOBPCG: keep for preconditioned sparse, and harden

Kept for sparse \f$A\f$ with a good preconditioner, where no \f$O(n^3)\f$ alternative exists. The
hardening follows Duersch, Shao, Yang and Gu (SISC 2018): Hetmaniuk-Lehoucq basis selection,
`svqb`/`svqbDrop` with condition-based dropping, and a backward-stable criterion. The old criterion
\f$\|r\| / (\|x\|\,|\lambda|)\f$ was not backward stable. Open: staircase locking, warm start, and a
Chebyshev preconditioner.

### Spectrum slicing, FEAST, contour integration, shift-and-invert: no

Slicing adds orthogonality bookkeeping that the batch dimension already makes unnecessary. FEAST and
shift-and-invert need a solve with \f$A - zI\f$ per shift per matrix, strictly more expensive than the
direct route for batched dense input. The `syevr` and generalized `sygvx` non-goals are on
@ref design_syevx_range.

## syevx dispatcher rules

### syevx dispatcher: problem arguments are validated before dispatch

`validate_syevx_preconditioner_params` and `validate_syevx_range_params` run in `syevx` and
`syevx_buffer_size` before `syevx_select_algorithm`. They check the problem, not the algorithm.
Otherwise an illegal combination on dense input would reach `Direct` or `DirectSubset`, which ignore
preconditioners.

`value_range_reportable` is false only for solve overloads without an `m` output. A short `m` span is
rejected on the host, because the device write it would cause is unchecked in release builds.

### syevx dispatcher: which preconditioner is legal at which end

- ILU(k) approximates \f$A^{-1}\f$, so it helps only the smallest eigenpairs. `find_largest` with an
  ILU(k) factor throws.
- `Jacobi`, \f$\mathrm{diag}(A)^{-1}\f$, inherits that restriction. With `find_largest` it turned 21 to
  47 iterations into 127 to 300, which is non-convergence at the cap.
- `JacobiShifted`, \f$(\mathrm{diag}(A) - \lambda I)^{-1}\f$ with the current Ritz value as \f$\lambda\f$, is
  legal at either end. Its iteration ratio with `find_largest` is 0.85x to 1.2x on random input and 0.2x
  to 0.9x on graded input.
- An explicit `preconditioner_type` must agree with the ILU(k) fields. Otherwise a caller's factor goes
  unused, or a family with no factor is requested. Both throw.

### syevx dispatcher: environment override versus explicit request

`algorithm_from_env` resolves `BATCHLAS_SYEVX_ALGORITHM` against `SyevxParams::method`. An environment
default degrades where an explicit request throws, so that a whole suite can run on one algorithm for
diagnosis.

- A non-extremal range under `lobpcg` or `filtered` degrades to `Direct`, with a once-per-process
  warning. An explicit request for the same combination throws.
- Trap: the variable wins whenever it is set. An unrecognised value parses to `Auto`, which ignores
  `params.method`. This is preserved on purpose.
- `BATCHLAS_SYEVX_PRECONDITIONER` supplies only the default. A configured ILU(k) factor always wins.
  `ILUK` from the environment is dropped, since there is no factor and `syevx` does not build one.
  `Jacobi` from the environment is dropped for `find_largest` calls.

### syevx: why the preconditioner environment variable is only a default

An algorithm can be substituted, because only performance changes. A preconditioner cannot: an ILU(k)
factor the caller built has no substitute, and silently ignoring it is a correctness surprise. Only
LOBPCG uses the preconditioner.

`Auto` never picks a Jacobi form, since neither is a free win on the measured matrices.
`JacobiShifted` is offered because it is safe in general and is the only Jacobi form legal with
`find_largest`.

### syevx dispatcher: sparse input and an explicit Filtered request

`Direct` and `DirectSubset` need dense \f$A\f$, so on CSR they degrade to LOBPCG. `Filtered` works on CSR
through `spmm`, so an explicit `Filtered` on CSR is honoured. `Auto` still lands on LOBPCG.

The CSR branch must stay below the non-extremal range check. Otherwise sparse input with an interior
range would return extremal eigenpairs instead of throwing. LOBPCG and `Filtered` only see extremal
ranges, so `syevx` fills `m` with the resolved block size before the solve, on the same in-order queue.

## syevx path details

### syevx Direct: the range is checked at its own entry point

`syevx_direct` and `syevx_direct_subset` are public, so they re-check Index and Value ranges. An
out-of-range block would index past the \f$n\f$-entry eigenvalue array. The check throws
`std::invalid_argument`; `syevx_resolve_range` clamps as a second defence. `DirectSubset` validates
before the reduction, since `stebz` would otherwise reject the range after \f$O(n^3)\f$ work.

`neigs` above \f$n\f$ is accepted and clamped. `neigs = 0` is rejected by `DirectSubset`, because
`stein` needs \f$k \ge 1\f$.

### syevx DirectSubset: reported as converged, with a caveat

`info` is 0 for every item. That is a caveat, not a check: `stebz` gives no exit status, and `stein` runs
a fixed number of inverse iterations with no convergence test. A real flag needs a residual check, which
is not implemented. Writing 0 is deliberate, so that an untouched span cannot be read as "converged".

### syevx Filtered: the scaled Chebyshev filter

One outer iteration:

```
Y = p_m(A) X            Chebyshev filter, m matvecs
Y = ortho(Y)
H = Y^H A Y             projected problem
H = Z diag(theta) Z^H   syev
X = Y Z                 Ritz vectors
```

\f$p_m\f$ maps the unwanted interval into \f$[-1, 1]\f$, where \f$|T_m| \le 1\f$. The wanted end lies
outside, where \f$T_m\f$ grows like \f$\cosh(m\,\mathrm{acosh}\,x)\f$. The cut is the first unwanted Ritz value.
The filter needs only matvecs, so no preconditioner or factorization.

The recurrence is the scaled Zhou/Saad form:

\f[
Y_1 = \frac{\sigma_1}{e}(A - cI)X, \qquad
Y_{j+1} = \frac{2\sigma'}{e}(A - cI)Y_j - \sigma\sigma' Y_{j-1}.
\f]

Unscaled \f$T_m\f$ overflows: \f$T_{25}(3) \approx 10^{38}\f$, which is float infinity. The \f$\sigma\f$ factors
keep intermediates near unit magnitude. The block is orthonormalized afterwards.

Guard directions (`extra_directions`, default `max(2, k/4)`) keep the last wanted pair from converging
slowly. The start block is a hash of the index, so it is reproducible across runs and backends.

### syevx Filtered: normalising the filter and capping its degree

**Normalise at the extreme Ritz value**, not the Gershgorin bound. Gershgorin overestimates a random
symmetric matrix's spectral radius (\f$O(n)\f$ against \f$O(\sqrt n)\f$). Normalising there underflows the block
at degree 40, and orthogonalizing a zero block gives NaN.

**Cap the degree by precision.** The filter amplifies the most-wanted direction over the least-wanted one by

\f[
\frac{\cosh(d\,\mathrm{acosh}\,y_{far})}{\cosh(d\,\mathrm{acosh}\,y_{edge})}.
\f]

Past what the working precision holds, the block is rank-deficient and the Cholesky-based `ortho`
returns NaN. The cap is

\f[
d_{cap} = \frac{-\tfrac14 \log \epsilon}{\mathrm{acosh}\,y_{far} - \mathrm{acosh}\,y_{edge}},
\f]

a ratio budget of \f$\epsilon^{-1/4}\f$ (about 300 in float). A budget of \f$\epsilon^{-1/2}\f$ still failed at
degree 40. When \f$y_{edge} \le 1\f$ the edge term is taken as 0, the worst case, so the cap still applies.
Each item's cap is reduced by **min** across the batch, so every GEMM shape stays uniform.

### syevx Filtered: the problem-derived degree

Behind `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=1`, off by default. The degree that finishes in one more outer
iteration is

\f[
d_{need} = \frac{\log(\text{worst residual} / \text{target})}{r}, \qquad r = \mathrm{acosh}\,y_{edge},
\f]

in the units of the convergence test. Each item's \f$d_{need}\f$ is clamped to `[kAutoDegreeMin, kAutoDegreeMax]`
= `[10, 40]`, reduced by **max** across the batch, then trimmed by the batch-minimum cap. The lower bound is
the old constant, so enabling the derivation never makes a step less selective.

Precedence: `BATCHLAS_SYEVX_FILTER_DEGREE` > `params.filter_degree` > derived > `kDefaultFilterDegree`. An
explicit value turns the derivation off. `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=0` restores the constant exactly.

**Trap:** `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO` and `BATCHLAS_SYEVX_BOUNDS_LEGACY` are parsed with `atoi`, not
`env_truthy`, so `true` and `on` read as false. This is kept, so existing spellings keep their meaning.

## syevx validation

- The residual \f$\|Ax_i - \lambda_i x_i\| / (\|A\|\,\|x_i\|)\f$ for every returned pair.
- Orthogonality \f$\|V^TV - I\|\f$, the binding constraint on clusters. The Wilkinson-matrix `stein` test is the gate.
- Eigenvalue agreement with `dsyevx` and `dstebz`, including repeated, Wilkinson and graded spectra, and
  \f$\lambda \approx 0\f$.
- Performance sweep: \f$n\f$ in 32 to 4096, \f$k/n\f$ in {0.01 … 0.5}, batch in {1, 8, 64, 512}, both `JobType`s.
  Baselines: `syev`, every `syevx` algorithm, and the vendor non-batched subset routine looped over the batch.

fp64 CPU failures on the primary machine are usually the broken OpenBLAS Cooperlake `dgemm`. Check a
known-good BLAS first.

## syevx status

| Component | State |
|---|---|
| `Direct`, `DirectSubset`, `stebz`, `stein` (Index and Value ranges) | done |
| `Filtered` | done; opt-in only |
| LOBPCG hardening | mostly done; open: staircase locking, warm start, Chebyshev preconditioner |
| Fused filtered CTA solver for small \f$n\f$ | not built; gated on a gap estimate |

Known risks: inverse iteration on variable-size clusters (orthogonality is a hard test gate); a bad
spectral bound in `Filtered` gives no acceleration or overflow (bounds are conservative, the degree is
clamped, and divergence is detected; two NaN modes were found and fixed); and per-feature choices across
the batch (max-over-batch or bucketing, not single-matrix adaptivity).

## syevx sources

Bisection, inverse iteration, MRRR:

- [ALGLIB: symmetric bisection and inverse iteration](https://www.alglib.net/eigen/symmetric/symmbisectionandinverseiteration.php)
- [ALGLIB: tridiagonal EVD, bisection and inverse iteration](https://www.alglib.net/eigen/symmetric/tdbisectionandinverseiteration.php)
- [Dhillon, Parlett and Voemel: the design and implementation of the MRRR algorithm (TOMS)](https://dl.acm.org/doi/10.1145/1186785.1186788)
- [Petschow and Bientinesi: MRRR-based eigensolvers for multi-core processors and supercomputers](https://arxiv.org/pdf/1401.4950)
- [Petschow et al.: high-performance solvers for dense Hermitian eigenproblems](https://arxiv.org/pdf/1205.2107)
- [A modified bisection algorithm for eigenvalues of a symmetric tridiagonal matrix (Numer. Math.)](https://link.springer.com/article/10.1007/BF01396441)
- [Dongarra et al.: numerical eigen-spectrum slicing, accurate orthogonality (IJHPCA 2024)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/eigen-improve-ijhpca-2024.pdf)

Polynomial filtering:

- [ChASE: Chebyshev Accelerated Subspace iteration Eigensolver](https://arxiv.org/pdf/1805.10121)
- [ChASE: distributed hybrid CPU-GPU eigensolver](http://fulir.irb.hr/7504/1/3539781.3539792.pdf)
- [Advancing the distributed multi-GPU ChASE library (SC'23 Workshops)](https://dl.acm.org/doi/10.1145/3624062.3624249)
- [Residual-based Chebyshev filtered subspace iteration tolerant to inexact matvecs](https://arxiv.org/pdf/2503.22652)
- [AdaPolySI: adaptive polynomial filtered subspace iteration (ACM ICS)](https://doi.org/10.1145/3797905.3800553)
- [Zhou and Saad: a Chebyshev-Davidson algorithm for large symmetric eigenproblems (SIMAX)](https://dx.doi.org/10.1137/050630404)
- [Zhou: a block Chebyshev-Davidson method with inner-outer restart](https://www.sciencedirect.com/science/article/abs/pii/S0021999110004791)
- [Saad et al.: parallel self-consistent-field calculations via Chebyshev-filtered subspace acceleration](https://arxiv.org/pdf/cond-mat/0703239)

LOBPCG:

- [Knyazev: toward the optimal preconditioned eigensolver, LOBPCG (SISC)](https://epubs.siam.org/doi/10.1137/S1064827500366124)
- [Duersch, Shao, Yang and Gu: a robust and efficient implementation of LOBPCG](https://arxiv.org/pdf/1704.07458)
- [Anzt, Tomov and Dongarra: accelerating LOBPCG on GPUs using a blocked SpMV](https://icl.utk.edu/files/publications/2014/icl-utk-771-2014.pdf)
- [Efficient implementation of the LOBPCG algorithm on a CPU-GPU cluster](https://link.springer.com/chapter/10.1007/978-981-96-2830-8_6)
- [Local convergence behavior of extended LOBPCG](https://arxiv.org/pdf/2505.08218)

Dense eigensolver performance on GPUs:

- [Extracting the potential of emerging hardware accelerators for symmetric eigenvalue decomposition](https://arxiv.org/html/2410.02170v1)
- [cuSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
- [pytorch/pytorch#175585: linalg.eigh performance cliff at n=32 for batched CUDA inputs](https://github.com/pytorch/pytorch/issues/175585)
