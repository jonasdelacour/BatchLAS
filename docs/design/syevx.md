# syevx: design of the partial symmetric eigensolver {#design_syevx}

> **Status:** current · checked 2026-10-29

`syevx` returns \f$k\f$ eigenpairs of a symmetric or Hermitian matrix, by extreme end or by an
index or value range. It dispatches to one of four algorithms by a hand-written rule in
`syevx_select_algorithm` (no `tuned/` table). Measured crossovers are on @ref perf_syevx.

| Piece | Source |
|---|---|
| entry points, validation, `syevx_select_algorithm`, `syevx_resolve_range`, thresholds | `src/ops/syevx/syevx.cc` |
| `Direct`, `DirectSubset`, `Filtered`, `LOBPCG` | `src/ops/syevx/kernels/syevx_{direct,direct_subset,filtered,lobpcg}.cc` |
| `SyevxParams`, `SyevxResolvedRange`, public declarations | `include/batchlas/blas/extensions.hh` |

## syevx algorithms

| Algorithm | Input | Method | Chosen by Auto when |
|---|---|---|---|
| `Direct` | dense; all types, ranges and `jobz` | the public `syev`, then select | \f$n \le 64\f$, eigenvalues only, complex, or below the `DirectSubset` gates |
| `DirectSubset` | dense, real only (`syevx_direct_subset_supported`); in-order Queue | `sy2sb`, `sb2st`, `stebz`, `stein`, back-transforms on \f$k\f$ columns | eigenvectors at large \f$n\f$ and batch |
| `LOBPCG` | CSR (dense on explicit request) | batched LOBPCG, optional ILU(k) or Jacobi preconditioner | sparse input, extremal ranges only |
| `Filtered` | CSR or dense | scaled Chebyshev filter with Rayleigh-Ritz | never; explicit request only |

On dense real input with eigenvectors, Auto picks `DirectSubset` when \f$n \ge\f$ `kSyevxSubsetMinN`
(1024) and \f$n \cdot \mathrm{batch} \ge\f$ `kSyevxSubsetMinWork` (\f$128 \cdot 1024\f$), and `Direct`
otherwise; `kSyevxSmallN` = 64. The rule has no \f$k/n\f$ term: the `DirectSubset` advantage is flat in
\f$k\f$ from 0.8 % to 25 % of the spectrum. An explicit `DirectSubset` request on complex input runs
`Direct`. No vendor routine is a batched subset eigensolver (`cusolverDnXsyevdx` and MAGMA `syevdx` are
not batched; the batched cuSOLVER routines solve the full spectrum), which is the case for `DirectSubset`.

## syevx: the flop-count cost model

Leading-order flops, real \f$n \times n\f$, \f$k\f$ wanted pairs. Measurement did not confirm their
performance predictions; they bound what a subset path can win.

| Path | Flops |
|---|---|
| Full, one-stage, with vectors | \f$\approx 4.7n^3\f$ (sytrd \f$\tfrac43 n^3\f$, stedc \f$\tfrac43 n^3\f$, ormtr \f$2n^3\f$) |
| Full, two-stage, with vectors | \f$\approx 6.7n^3\f$; the back-transform is applied twice |
| `DirectSubset` | \f$\approx \tfrac43 n^3 + 4n^2k\f$ |
| Chebyshev subspace iteration, dense, degree \f$d\f$ | \f$4dn^2k\f$ per outer pass |
| LOBPCG, dense, \f$it\f$ iterations | \f$2\,it\,n^2k + O(it\,nk^2)\f$ |

The subset ceiling is about 3x as \f$k/n \to 0\f$ because tridiagonalization stays \f$O(n^3)\f$
(measured best 2.40x); eigenvalues-only subset solves lost 3x to 5x. Iterative methods on dense input
break even only at \f$k < n/(3d)\f$ and measured losses at moderate \f$k/n\f$.

## syevx small n: why Direct is the answer

For \f$n \le\f$ `kSyevxSmallN` (64) a CTA-resident solver is bound by occupancy, shared-memory traffic
and sweep count, not flops, so a subset kernel would land within noise of `syev_cta`. `syevx` calls
`syev` and selects; the path exists for API compatibility. A fused filtered CTA solver (estimated 3x to
8x at \f$n = 32\f$, \f$k = 2\f$ on a well-separated spectrum) is not built; gate it on a measured gap
estimate first.

## syevx for batches of medium matrices

`DirectSubset` stages: `sytrd_sy2sb` (dense to band), `sytrd_sb2st_hh` (band to tridiagonal, keeps Q2),
`stebz` (subset eigenvalues), `stein` (subset eigenvectors), `unmqr_hb2st` (through Q2) and
`ormqr_blocked` (through Q1; host LAPACKE `ormqr` on `Backend::NETLIB`), both on \f$k\f$ columns.

- The tridiagonal subset stage is off the critical path (at \f$n = 1024\f$, \f$k = 64\f$, bisection is
  about 2.6 MFlop against 1.4 GFlop for `sy2sb`). It must be correct and batched, not fast, which favours
  bisection with inverse iteration over MRRR.
- `stein` runs one work-item per vector, a serial tridiagonal LU solve of \f$(T - \lambda I)x = b\f$;
  vectors in a cluster are reorthogonalized by modified Gram-Schmidt.
- Per-matrix adaptivity (filter degree, block width) breaks the uniform shapes batched GEMM needs: take
  the max or min over the batch.
- LOBPCG allocates four \f$n \times 3k \times \mathrm{batch}\f$ buffers (about 800 MB at \f$n = 4096\f$,
  \f$k = 64\f$, batch 32, fp64); `DirectSubset` needs about an order of magnitude less.

## syevx algorithm verdicts

### Bisection and inverse iteration, stebz and stein: build

Bisection counts eigenvalues below a shift from Sturm sign changes; each eigenvalue maps to one lane
with no communication. Inverse iteration's loss of orthogonality in tight clusters costs \f$O(nk^2)\f$
to repair, which does not bind at \f$k \ll n\f$. Built as `stebz` and `stein`.

### Chebyshev-Davidson: consider as the LOBPCG replacement for unpreconditioned problems

Not built. The polynomial filter is the expansion operator, so no preconditioner is needed.
Unpreconditioned LOBPCG stagnates, and that is the default configuration; `Filtered` covers much of
the same ground.

| Alternative | Verdict | Why |
|---|---|---|
| MRRR (`stemr`, `dsyevr`) | do not build | hard to implement correctly; its \f$O(n^2)\f$ saving is negligible against the \f$O(n^3)\f$ reduction |
| Chebyshev-filtered subspace iteration (ChFSI) | built as `Filtered` | all GEMM/SpMM; strong on sparse input, not dense |
| LOBPCG | keep for preconditioned sparse | no \f$O(n^3)\f$ alternative there; hardened after Duersch, Shao, Yang and Gu (SISC 2018) |
| Spectrum slicing, FEAST, shift-and-invert | no | the batch dimension already gives the parallelism; a solve per shift per matrix costs more than the direct route |
| Folded-spectrum LOBPCG, band-pass filter, `syevr` entry, `sygvx`, per-item ranges | deferred / non-goal | `DirectSubset` answers interior ranges at no extra cost |

## syevx dispatcher rules

### syevx dispatcher: problem arguments are validated before dispatch

`validate_syevx_preconditioner_params` and `validate_syevx_range_params` run in `syevx` and
`syevx_buffer_size` before `syevx_select_algorithm`. They check the problem, not the algorithm, so an
illegal combination cannot reach `Direct` or `DirectSubset` (which ignore preconditioners) unseen. Every
rejection is a `batchlas::invalid_argument`. `value_range_reportable` is false only for overloads
without an `m` output; a short `m` span is rejected on the host because the device write it would cause
is unchecked in release builds.

### syevx dispatcher: which preconditioner is legal at which end

- An ILU(k) factor comes from the caller (`SyevxParams::preconditioner`) or is built from the workspace
  (`build_preconditioner`, CSR only). Both at once throw.
- ILU(k) and `Jacobi` approximate \f$A^{-1}\f$ and help only the smallest pairs: either with
  `find_largest` throws ([measured](../perf/syevx.md#lobpcg-jacobi-preconditioners)).
- `JacobiShifted`, \f$(\mathrm{diag}(A) - \lambda I)^{-1}\f$ at the current Ritz value, is legal at either
  end.
- An explicit `preconditioner_type` must agree with the ILU(k) fields, else it throws.

### syevx dispatcher: environment override versus explicit request

`algorithm_from_env` resolves `BATCHLAS_SYEVX_ALGORITHM` against `SyevxParams::method`. An environment
default degrades where an explicit request throws, so a whole suite can run on one algorithm.

- Trap: the variable wins whenever it is set. An unrecognised value parses to `Auto`, which ignores
  `params.method`. Kept on purpose.
- `BATCHLAS_SYEVX_PRECONDITIONER` supplies only the default (`syevx_select_preconditioner`). A configured
  ILU(k) factor always wins; `ILUK` from the environment is dropped (no factor); `Jacobi` from the
  environment is dropped for `find_largest`.

### syevx: why the preconditioner environment variable is only a default

An algorithm can be substituted because only performance changes. A preconditioner cannot: an ILU(k)
factor the caller built has no substitute, and silently ignoring it is a correctness surprise. `Auto`
never picks a Jacobi form, since neither is a free win on the measured matrices; `JacobiShifted` is
offered because it is safe and the only Jacobi form legal with `find_largest`.

### syevx dispatcher: sparse input and an explicit Filtered request

On CSR, `Direct` and `DirectSubset` degrade to LOBPCG; an explicit `Filtered` is honoured (it runs on
`spmm`); `Auto` lands on LOBPCG. Trap: the CSR branch must stay below the non-extremal range check, or
sparse input with an interior range would return extremal pairs instead of throwing. LOBPCG and
`Filtered` see only extremal ranges, so `syevx` fills `m` with the block size before the solve.

## syevx path details

### syevx Direct: the range is checked at its own entry point

`syevx_direct` and `syevx_direct_subset` are public, so they re-check Index and Value ranges
(`batchlas::invalid_argument`); `syevx_resolve_range` clamps as a second defence, since an out-of-range
block would index past the eigenvalue array. `syevx_direct_subset` also throws on an out-of-order Queue
and `batchlas::unsupported` on complex or CSR input, and validates before the reduction, because `stebz`
would otherwise reject the range after \f$O(n^3)\f$ work. `neigs` above \f$n\f$ is clamped; `neigs = 0` is
rejected by `DirectSubset` because `stein` needs \f$k \ge 1\f$.

### syevx DirectSubset: reported as converged, with a caveat

`info` is 0 for every item (`detail::info_clear`). That is not a check: `stebz` gives no exit status
and `stein` runs a fixed number of inverse iterations with no convergence test. A real flag needs a
residual check (not implemented). Writing 0 is deliberate, so an untouched span cannot read as
"converged".

### syevx Filtered: the scaled Chebyshev filter

One outer iteration: \f$Y = p_m(A)X\f$ (\f$m\f$ matvecs), orthonormalize, \f$H = Y^HAY\f$, `syev` of
\f$H\f$, Ritz vectors \f$X = YZ\f$. \f$p_m\f$ maps the unwanted interval into \f$[-1, 1]\f$, where
\f$|T_m| \le 1\f$; the wanted end grows like \f$\cosh(m\,\mathrm{acosh}\,x)\f$. The cut is the first
unwanted Ritz value. The recurrence is the scaled Zhou/Saad form

\f[
Y_1 = \frac{\sigma_1}{e}(A - cI)X, \qquad
Y_{j+1} = \frac{2\sigma'}{e}(A - cI)Y_j - \sigma\sigma' Y_{j-1},
\f]

because unscaled \f$T_{25}(3) \approx 10^{38}\f$ is float infinity. Guard directions
(`extra_directions`; 0 means `max(2, k/4)`) keep the last wanted pair from converging slowly. The start
block is a hash of the index, reproducible across runs and backends.

### syevx Filtered: normalising the filter and capping its degree

**Normalise at the extreme Ritz value**, not the Gershgorin bound: Gershgorin overestimates a random
matrix's spectral radius (\f$O(n)\f$ against \f$O(\sqrt n)\f$), the block underflows at degree 40, and
orthogonalizing a zero block gives NaN.

**Cap the degree by precision.** Past what the working precision holds, the block is rank-deficient
and the Cholesky-based `ortho` returns NaN. The cap

\f[
d_{cap} = \frac{-\tfrac14 \log \epsilon}{\mathrm{acosh}\,y_{far} - \mathrm{acosh}\,y_{edge}}
\f]

is a ratio budget of \f$\epsilon^{-1/4}\f$ (about 300 in float; \f$\epsilon^{-1/2}\f$ still failed at
degree 40). When \f$y_{edge} \le 1\f$ the edge term is 0. The cap is reduced by **min** across the batch
so every GEMM shape stays uniform.

### syevx Filtered: the problem-derived degree

Behind `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=1`, off by default
([measured](../perf/syevx.md#syevx-automatic-chebyshev-filter-degree)). Each item's
\f$d_{need} = \log(\text{worst residual} / \text{target}) / \mathrm{acosh}\,y_{edge}\f$ is clamped to
[`kAutoDegreeMin`, `kAutoDegreeMax`] = [10, 40], reduced by **max** across the batch, then trimmed by the
batch-minimum cap. The lower bound equals `kDefaultFilterDegree` (10), so the derivation never makes a
step less selective. Precedence: `BATCHLAS_SYEVX_FILTER_DEGREE` > `params.filter_degree` > derived >
`kDefaultFilterDegree`; an explicit value turns the derivation off. **Trap:** the switch is parsed with
`atoi`, not `env_truthy`, so `true` and `on` read as false; kept so existing spellings keep their
meaning.

## syevx: range selection {#design_syevx_range}

`SyevxSelect::Index` selects pairs `il..iu` of the ascending spectrum (LAPACK `range = 'I'`); `Value`
selects \f$v_l < \lambda \le v_u\f$ (`range = 'V'`). Both work from C++ and Python on `Direct` and
`DirectSubset`. `stebz` already supports all three LAPACK ranges: a Value range becomes an index range per
item through two Sturm counts, exact even inside a tight cluster.

### syevx range: the m output and what neigs means

For a Value range the count is known only after the reduction and differs per item. As in LAPACK, the
caller declares a capacity, the library writes \f$\min(m[b], \mathrm{neigs})\f$ pairs, lowest first, and
reports the true \f$m[b]\f$; \f$m[b] > \mathrm{neigs}\f$ is the overflow signal, never a silent truncation.

| `select` | `neigs` means | `m[b]` is |
|---|---|---|
| `Extremal` | number wanted | always `neigs` |
| `Index` | must equal `iu - il + 1` | always `neigs` |
| `Value` | capacity of `W` and `V` per item | the true count; may exceed `neigs` |

`W` past `m[b]` is untouched; `V` past `m[b]` is exactly zero. The asymmetry is deliberate: the
`DirectSubset` back-transforms run over a uniform column count and need inert columns, both dense paths
must agree, and `W`'s tail stays usable as a "was this slot written" sentinel. Explicit ranges return
ascending order unless `SyevxParams::order` says otherwise; `Extremal` keeps descending under
`find_largest`. `stein` needs ascending input, so descending output comes from a reversal in the
finalize kernel. Rejected: counting then solving in two calls (the count needs the full reduction) and
a per-item variable layout (breaks `MatrixView`).

### syevx range: SyevxSelect rather than EigenRangeType

`SyevxSelect { Extremal, Index, Value }` defaults to `Extremal`, normalized internally to an Index range
but kept separate so no caller depends on a changed default. `EigenRangeType` is not reused because its
default `All` means every eigenvalue. `SyevxParams` adds `il`, `iu` (inclusive, 0-based, `-1` means
\f$n-1\f$), `vl`, `vu`, `abstol` (non-positive means \f$\epsilon\|T\|\f$) and `order`.

`syevx`, `syevx_direct` and `syevx_direct_subset` each have an overload taking `Span<int32_t> m` after
`W`; overloads without `m` throw on `Value`. The `m` overload is told apart by parameter 5 (`size_t
neigs`), since `Span` converts implicitly from `T&` at position 4. The `m`-less overloads are inline
forwarders; `SyevxRangeValidationTest.SolverEntryPointsKeepTheirMLessOverloads` fails if they go missing.
Python: `bl.syevx` returns `m` only when a range is requested; `bl.syevx_range` returns per-item lists
sliced to \f$\min(m[b], \mathrm{capacity})\f$ plus `counts`, `truncated` and `capacity`, requires `neigs`
for a Value range, and rejects `select="extremal"`.

### syevx range: normalization and validation

`syevx_resolve_range` is the single normalization point for every solve and every `buffer_size`.

| Input | Result |
|---|---|
| `Extremal` | with \f$c = \min(\mathrm{neigs}, n)\f$: `il = find_largest ? n-c : 0`, `iu = il+c-1`, `reverse = find_largest` (`order` ignored) |
| `Index` | `il`, `iu` clamped into \f$[0, n-1]\f$, `reverse = (order == Descending)` |
| `Value` | `value_range = true`, `max_count = min(neigs, n)`, `reverse = (order == Descending)` |

The work count is clamped to \f$n\f$; the caller's `neigs` stays the output stride. An inverted block
resolves to the empty block (`il = 0`, `iu = -1`, `max_count = 0`). `validate_syevx_range_params` throws
for Index with \f$il < 0\f$, \f$iu \ge n\f$, \f$il > iu\f$ or `neigs != iu - il + 1`; Value with
\f$v_l \ge v_u\f$ (a swapped pair would cost a full reduction for an empty answer); and Value without
`m`. Not enforced: `Extremal` with a contradicting `order` (`SortOrder` has no "unset", and Python sends
every field) and capacity above \f$n\f$ (clamped).

### syevx range: throw, do not degrade

Only `Direct` and `DirectSubset` answer a non-extremal range. LOBPCG converges to whichever extreme its
trial block favours and has no \f$il/iu\f$; `Filtered` is a high-pass, and an interior interval has
unwanted spectrum on both sides. Neither would fail, both would answer a different question, so:

- sparse input with a non-extremal range throws;
- an explicit `LOBPCG` or `Filtered` with a non-extremal range throws, and both entry points reject one
  themselves;
- `BATCHLAS_SYEVX_ALGORITHM=lobpcg|filtered` with a non-extremal range degrades to `Direct` with a
  once-per-process warning.

### syevx range: why the thresholds carry over unchanged

The crossovers depend on \f$(n, \mathrm{batch}, \mathrm{jobz})\f$, not on \f$k\f$, and spectrum position
cannot change the cost of either dense path: the reduction, bisection steps and back-transform slice are
the same wherever the block sits. For a Value range the selector sees the capacity, since both paths
scale with it. A routing unit test pins the decision. `BM_SYEVX_RangePosition` has not been run; if the
argument is wrong the symptom is a slower correct answer. `Direct`'s spread across bottom, middle and top
is the noise floor; if `DirectSubset` leaves that band, the gate needs a range-aware term.

### syevx range: stein's per-item count bound

`stein` takes one uniform \f$k\f$; with a Value range item \f$b\f$ has \f$m[b] \le k\f$ valid eigenvalues and a
stale tail. Phase 1 needs the `kb` bound, or inverse iteration runs on garbage shifts. Phase 2's bound is
for cost only: modified Gram-Schmidt reads only earlier columns, but a stale tail would join the last real
eigenvalue into one bogus cluster. `stein` zeroes columns at or beyond \f$m[b]\f$; the back-transforms keep
the capacity column count (orthogonal maps send zero to zero), so a batch with 3 and 200 pairs pays for
200 columns everywhere. `stebz` needs `w.size()` \f$\ge n\f$ for a Value range.

`SteinTest.PerItemCountsIgnorePoisonedTail` checks this. Its orthonormality-and-residual assertion
does not discriminate (zero failures with the bound removed); the second one does: item 1's unused
columns, pre-poisoned, must come back exactly zero. Keep both.

### syevx range: stein workspace scaling

`stein` allocates five length-\f$n\f$ arrays and pivot flags per (item, vector): \f$21\,nk \cdot
\mathrm{batch}\f$ bytes in float, about 2.8 GB at \f$n = 1024\f$, batch 128, `neigs = n`. Callers must pass
a realistic capacity (documented in the C++ declarations and `bl.syevx_range`). Open: index scratch by
resident work-item, \f$O(n \cdot \mathrm{resident})\f$ for every `stein` caller.

### syevx range: the host reference oracle

A full host `syev`, sorted ascending, then selection by index or by \f$(v_l, v_u]\f$; the Python
`test_syevx_*range*` cases use `numpy.linalg.eigvalsh`. Value cases use `Matrix::TriDiagToeplitz`
(closed-form eigenvalues) with boundaries in spectral gaps: empty, everything, items that disagree
(`m = {n, 0}`) and overflow (`m == 20`, `neigs = 5`). An Index block touching either end must be
bit-identical to the `Extremal` call.

### syevx range: Direct and DirectSubset may disagree on m by one

`Direct` counts by a sorted search, `DirectSubset` by a Sturm count, so a boundary within rounding
distance of an eigenvalue can make them differ by one. This is the documented tolerance; one test puts a
boundary on an eigenvalue and asserts \f$|m_{\mathrm{direct}} - m_{\mathrm{subset}}| \le 1\f$.

## syevx validation

Every returned pair is checked by residual \f$\|Ax_i - \lambda_i x_i\| / (\|A\|\,\|x_i\|)\f$ and
orthogonality \f$\|V^TV - I\|\f$ (the binding constraint on clusters; the Wilkinson-matrix `stein` test is
the gate), through `EXPECT_VERIFY` (@ref design_verification). fp64 CPU failures on the primary machine
are usually the broken OpenBLAS Cooperlake `dgemm`; check a known-good BLAS first.

## syevx open debts

- `DirectSubset` reports `info = 0` without a residual check.
- `BM_SYEVX_RangePosition` not run ([measurement debts](../perf/syevx.md#syevx-open-measurement-debts)).
- `stein` scratch scales with \f$nk \cdot \mathrm{batch}\f$.
- LOBPCG: staircase locking, warm start, a Chebyshev preconditioner.
- Fused filtered CTA solver for small \f$n\f$: not built.

## syevx sources

- [Dhillon, Parlett and Voemel: MRRR (TOMS)](https://dl.acm.org/doi/10.1145/1186785.1186788)
- [ChASE: Chebyshev Accelerated Subspace iteration Eigensolver](https://arxiv.org/pdf/1805.10121)
- [Zhou and Saad: a Chebyshev-Davidson algorithm (SIMAX)](https://dx.doi.org/10.1137/050630404)
- [Knyazev: LOBPCG (SISC)](https://epubs.siam.org/doi/10.1137/S1064827500366124)
- [Duersch, Shao, Yang and Gu: a robust and efficient implementation of LOBPCG](https://arxiv.org/pdf/1704.07458)
