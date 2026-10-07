# syevx: design of the partial symmetric eigensolver {#design_syevx}

**Covers:** why `syevx` routes over several algorithms instead of being one, the flop-count cost
model it was designed from, the small- and medium-matrix regimes, the algorithm survey and its
verdicts, what not to build, and the validation plan and risks.
**Status:** design record, written 2026-08-01 (before implementation) and corrected in place where
the implementation or the measurements proved it wrong. The corrections are marked.
**The cost model's predictions were largely overturned by measurement**: see @ref perf_syevx for
the measured crossovers, the current routing rule, and the implementation status of each tier.
Range selection (`Index`/`Value`) has its own record: @ref design_syevx_range.

## syevx before routing: one algorithm under a dispatcher's name

When this design was written, `src/extensions/syevx.cc` was a batched LOBPCG (Locally Optimal
Block Preconditioned Conjugate Gradient) for dense and CSR input, all scalar types, all backends.
It was the only `syevx` path. Per iteration, on dense input with block width \f$k\f$:

| step | cost |
|---|---|
| residual \f$R = AX - X\,\mathrm{diag}(\lambda)\f$, norms, best snapshot | \f$O(nk)\f$ bandwidth, `2*neigs` group reductions |
| host sync + convergence read | a full pipeline drain, every iteration |
| optional ILU(k) apply (+2 full copies) | \f$O(nk)\f$ x 3 |
| `ortho(R, XP)` | \f$O(nk^2)\f$ |
| \f$AR\f$ (or \f$AP\f$ on restart) | \f$2n^2k\f$ dense, or one SpMM |
| \f$S^T(AS)\f$ | \f$2n(3k)^2 = 18nk^2\f$ |
| `syev` on the `3k x 3k` projected matrix | batched small/medium dense eigensolve |
| 4 gemms `n x 3k x k` to form X, AX, P, AP | \f$8nk^2\f$ |

That is about \f$2n^2k + 30nk^2\f$ flops per iteration, dominated by the matvec block when
\f$k \ll n\f$. It had no locking or deflation, no polynomial acceleration, and only ILU(k), a sparse
construct, as a preconditioner. `extra_directions` defaulted to 0, the setting LOBPCG converges
worst under. Its convergence measure \f$\|r\| / (\|x\|\,|\lambda|)\f$ was not backward stable.
None of that is wrong for the sparse case it was built for. It is wrong as the universal answer to
"give me k eigenpairs": a dense caller asking for 10 % of the spectrum ran LOBPCG, which is slower
than `syev` followed by discarding eigenpairs. The central claim of the design was that **the
biggest available win is routing, not a new kernel**. The measured routing is in
[the perf page](../perf/syevx.md#syevx-routing-thresholds-as-they-stand).

## syevx: the flop-count cost model

Leading-order flops for real symmetric \f$n \times n\f$ with \f$k\f$ wanted eigenpairs. **These are
estimates, and the measurements did not bear out their performance predictions** (see
[the predicted three-way crossover](#syevx-cost-model-the-predicted-three-way-crossover)).

### syevx cost model: full eigendecomposition

One-stage, with vectors (`dsyevd`-shaped):

```
sytrd    (4/3)n^3      ~50 % SYMV -> memory-bound
stedc    ~(4/3)n^3     highly variable, deflation-dependent
ormtr     2n^3         GEMM
--------------------------
total    ~4.7n^3
```

Two-stage (`sy2sb` + `sb2st`):

```
sy2sb    (4/3)n^3      almost entirely SYR2K/GEMM -> compute-bound
sb2st    O(n^2 b)      cheap in flops, poorly parallel, latency-bound
stedc    ~(4/3)n^3
apply Q2  2n^3         back-transform through the bulge-chasing reflectors
apply Q1  2n^3         back-transform through the band reduction
--------------------------
total    ~6.7n^3, at far better flop efficiency
```

Two-stage's weakness is that it pays the back-transform twice, and that is exactly the term that
scales with the number of wanted vectors.

### syevx cost model: direct subset

```
sy2sb          (4/3)n^3
sb2st          O(n^2 b)
tridiag subset O(40 n k) bisection + O(nk) per inverse-iteration step
apply Q2       2n^2 k
apply Q1       2n^2 k
--------------------------
total ~ (4/3)n^3 + 4n^2 k
```

At \f$k = 0.05n\f$: \f$(1.33 + 0.20)n^3\f$ against \f$4.7n^3\f$, about 3.0x. Eigenvalues only:
\f$1.33n^3\f$ against \f$4.7n^3\f$, about 3.5x. So about 3x is the honest ceiling for a direct dense
subset solve, reached as \f$k/n \to 0\f$. It is not 10x, because tridiagonalization is \f$O(n^3)\f$
however few eigenpairs are wanted. The work deleted (`stedc` and the back-transforms) is the
memory-bound, badly scaling part, so the design expected the wall-clock ratio to exceed the flop
ratio. **Measured, the best ratio was 2.40x, at large n and large batch only, and eigenvalues-only
was a 3x to 5x loss.**

### syevx cost model: the kd = 1 correction

Found while implementing Tier 2. BatchLAS's `syev_two_stage` did *not* pay a double
back-transform with eigenvectors: it set `kd = 1`, which makes band-to-tridiagonal a pure
extract. The `sb2st` reflectors (`tau_sb2st`) were computed and never applied anywhere, which was
correct only because `kd = 1` leaves no bulge chasing to undo. So the eigenvector path was really
one-stage tridiagonalization plus a single `ormqr`. The ~3x ceiling survived, but from different
terms: with `kd = 1` the reduction cost is the same in the full and subset solves and cancels, and
the saving is (a) full `stedc` -> subset bisection/inverse iteration and (b) `ormqr` on `n`
columns -> on `k` columns:

```
full ED ~ (4/3)n^3 [sytrd] + (4/3)n^3 [stedc] + 2n^3  [ormqr, n cols] ~ 4.7n^3
subset  ~ (4/3)n^3 [sytrd] + O(nk)            + 2n^2k [ormqr, k cols] ~ 1.43n^3 at k=0.05n
```

Eigenvalue-only solves were unaffected: they use a wide band (`kd` = 16/32), get the GEMM-heavy
reduction, and need no back-transform. **Superseded later**: once `sytrd_sb2st_hh` retained Q2,
the subset path's eigenvector mode moved to the tuned band width plus a Q2 back-transform on `k`
columns (see [Tier 2 status](../perf/syevx.md#syevx-tier-2-directsubset)). Any other `kd > 1`
eigenvector path must apply the `sb2st` reflectors, which is noted at the top of
`src/extensions/two_stage_common.hh`.

### syevx cost model: iterative methods on dense input

Chebyshev-filtered subspace iteration, degree \f$d\f$, block width \f$m \approx 2k\f$, one outer
pass: \f$d \cdot 2n^2m = 4dn^2k\f$. Break-even against \f$\tfrac{4}{3}n^3\f$ is \f$k < n/(3d)\f$, so
with \f$d = 15\f$, \f$k < 2.2\,\%\f$ of \f$n\f$. LOBPCG on dense, `it` iterations:
\f$2\,it\,n^2k + O(it\,nk^2)\f$. Break-even is \f$it \cdot k < 0.67n\f$, so with an unpreconditioned
\f$it = 30\f$, also \f$k < 2.2\,\%\f$ of \f$n\f$, by coincidence of the constants.

### syevx cost model: the predicted three-way crossover

The prediction was: for dense input, iterative methods win only below about \f$k/n \approx 2\,\%\f$;
above that a direct subset solver wins; and above perhaps 25 % to 30 % a plain full `syev` wins,
because the subset machinery's constants and the loss of `stedc` deflation eat the margin.

**Measured outcome** (see @ref perf_syevx): the crossover has a batch dimension the model did not
have, and no `k/n` dimension at all. `Direct` wins all eigenvalues-only shapes. With eigenvectors,
DirectSubset wins only when `n >= 1024` and `n * batch >= 128 * 1024`, flat in `k` from 0.8 % to
25 %. Filtered has a narrow small-batch niche on dense input and is never routed to. The
prediction that iterative methods lose badly on dense input at moderate `k/n` held.

## syevx for batches of small matrices

The prior that "a subset eigensolver for small matrices almost never makes sense" is correct, with
one exception.

### syevx small n: why Direct is the answer

For `n <= 32..64` in a CTA-resident solver, run time is not flop-bound. It is bound by work-group
occupancy, shared-memory traffic and the sweep count of the tridiagonal solver. Cutting the
back-transform from `k = n` to `k = 2` columns removes work that was already hidden behind the
tridiagonalization and the QR sweeps, so against `syev_cta`/`syev_jacobi_cta` a subset variant
should land within noise. **Verdict: for `n <= 64`, `syevx` calls `syev` and selects**
(`kSyevxSmallN = 64`). It is exposed for API compatibility, not speed. No small-matrix subset
kernel.

### syevx small n: the filtered CTA exception (Tier 5)

One route skips tridiagonalization entirely. Hold A (`n <= 64`, at most 32 KB in fp64) in shared
memory for the whole kernel. Run \f$d\f$ steps of a block Chebyshev filter \f$p_d(A)X\f$ in
registers/SLM, each one \f$n \times n\f$ by \f$n \times k\f$ product (4 K flops at
\f$n = 32, k = 2\f$). Then Rayleigh-Ritz on a \f$k \times k\f$ or \f$2k \times 2k\f$ matrix in registers.
At \f$n = 32, k = 2, d = 12\f$ that is about 50 K flops against `syev_cta`'s tridiagonalization plus
100 to 400 QR sweeps, so a 3x to 8x win is plausible *if* the spectrum is well separated. With a
cluster at the wanted end, the needed degree explodes and the method degrades smoothly into
"slower than full ED". **Verdict: build it last, gated on a measured gap estimate, as an
opportunistic path. Not built.**

### syevx small n: bisection as a stedc alternative

The subset machinery's building block could help the small-matrix *full-spectrum* path. Bisection
assigns one eigenvalue per lane, is embarrassingly parallel, has no shared-memory ping-pong, and
has a fixed iteration count (about 40 to 60 Sturm evaluations for fp64 to machine precision). For
`n <= 32` on a 32-wide sub-group it is a natural fit and sidesteps the STEDC merge serialization.
The plan was to measure `stebz` as an eigenvalues-only alternative to `stedc` for small `n`. That
comparison has not been recorded.

## syevx for batches of medium matrices

This is where a subset solver earns its keep: `n` about 128 to 4096, batch 8 to 1000. The direct
route is assembled almost entirely from existing parts:

| stage | component |
|---|---|
| dense -> band | `sytrd_sy2sb` |
| band -> tridiagonal | `sytrd_sb2st` (Givens) or `sytrd_sb2st_hh` (retains Q2) |
| subset eigenvalues | `stebz` (bisection), new |
| subset eigenvectors | `stein` (inverse iteration), new |
| back-transform through Q2 | `unmqr_hb2st`, on `k` columns |
| back-transform through Q1 | `ormqr_blocked`, on `k` columns |

**The tridiagonal subset solver is not on the critical path.** At `n = 1024, k = 64`, bisection
costs about \f$40nk = 2.6\f$ MFlop against `sy2sb`'s 1.4 GFlop. It needs to be correct and batched,
not fast. That argues for simple bisection + inverse iteration over MRRR.

### syevx medium n: batched-specific constraints

- **Uniform shapes.** Batch members converge at different rates and have different cluster
  structure. Per-matrix adaptivity (variable filter degree, variable active block width) breaks
  the uniform shapes batched GEMM needs. Resolve by taking the max (or min) over the batch, or by
  bucketing. The auto filter degree later failed exactly this way (see
  [the automatic Chebyshev filter degree](../perf/syevx.md#syevx-automatic-chebyshev-filter-degree)).
- **Inverse iteration on clusters.** LAPACK's `stein` reorthogonalizes within clusters where
  \f$\lambda_{i+1} - \lambda_i < 10^{-3}\|T\|\f$. Cluster sizes vary per matrix. This was the main
  correctness risk of Tier 1.
- **Memory.** The LOBPCG path allocates four `n x 3k x batch` buffers, about 800 MB at
  `n = 4096, k = 64, batch = 32`, fp64. The direct path needs the band `n x b`, the tridiagonal
  `2n` and the `n x k` output, an order of magnitude less, which can make medium-`n` batched
  partial eigensolves feasible where LOBPCG runs out of memory.

## syevx: no vendor ships a batched subset eigensolver

- `cusolverDnXsyevdx`: subset, **not batched**.
- `cusolverDnXsyevBatched`: batched, **full spectrum**. It replaced `cusolverDnSsyevjBatched`,
  whose `n <= 32` cap was a leftover of the old Jacobi implementation. PyTorch's dispatch still
  carried that stale gate (pytorch/pytorch#175585).
- `cusolverDnSsyevjBatched` / `gesvdjBatched`: batched, full spectrum, `n <= 32`.
- MAGMA `syevdx`: subset via divide and conquer, **not batched**.

That gap is the product argument for the DirectSubset tier.

## syevx algorithm survey

Ranked by fit to BatchLAS, with verdicts.

### Bisection and inverse iteration, stebz and stein: build

Bisection computes any subset of the tridiagonal spectrum from Sturm sequence sign counts. Every
eigenvalue is independent, so it maps onto a lane-per-eigenvalue kernel with no communication. It
gives eigenvalues to any (including reduced) accuracy at proportional cost, and incomplete Sturm
sequences allow early termination when few eigenvalues are wanted. Inverse iteration then
recovers vectors at \f$O(n)\f$ per vector per step. Its known weakness, loss of orthogonality on
tight clusters requiring reorthogonalization at up to \f$O(nk^2)\f$, is the reason MRRR exists. In
this regime \f$k \ll n\f$ and \f$O(nk^2) \ll O(n^3)\f$, so the weakness does not bind. **Built** as
`stebz` and `stein`.

### MRRR (stemr, dsyevr): do not build

\f$O(n^2)\f$ total with no reorthogonalization, via relatively robust representations and twisted
factorizations. It is the theoretically right answer and a notoriously hard implementation
(differential qd transforms, RRR trees, careful shift selection, two decades of errata). LAPACK's
own `dsyevr` falls back to `dstebz` + `dstein` on some subset paths, so the reference
implementation agrees that bisection is the pragmatic subset choice.

### Chebyshev-filtered subspace iteration (ChFSI, ChASE): build for small k/n

A Chebyshev polynomial maps the unwanted interval to \f$[-1, 1]\f$ and the wanted spectrum to large
values, followed by Rayleigh-Ritz. All the work is GEMM (dense) or SpMM (sparse), which suits a
batched library. ChASE scales this to 526 A100s and optimizes the degree per accuracy target.
AdaPolySI (ACM ICS 2026) and residual-based ChFSI (arXiv:2503.22652) adapt the degree per
eigenpair and tolerate inexact matvecs. It needs a spectral bound estimate and reasonable
separation, and for batched work a degree that is uniform across the batch. **Built** as
`syevx_filtered`. It turned out to be the strong path on sparse input, not dense.

### Chebyshev-Davidson: consider as the LOBPCG replacement for unpreconditioned problems

Zhou and Saad's Chebyshev-Davidson uses the polynomial filter as the *expansion* operator in a
Davidson framework, so it needs no preconditioner. The block variant with inner-outer restart
uses roughly half the subspace dimension of ARPACK/TRLan. The relevant fact: **unpreconditioned
LOBPCG is documented to stagnate**, with residuals oscillating around a floor. BatchLAS's `syevx`
has no preconditioner unless the caller supplies ILU(k) (or Jacobi, for the smallest end), so
that failure mode is the default configuration. The CSR sweep later observed it directly (see
[the gap sweep](../perf/syevx.md#syevx-the-diagonal-boost-gap-sweep-and-lobpcg-stagnation)). Not
built. Filtered covers much of the same ground.

### LOBPCG: keep for preconditioned sparse, and harden

Its domain is sparse A with a good preconditioner, where no \f$O(n^3)\f$ alternative exists.
Duersch, Shao, Yang and Gu (SISC 2018) is the reference for robustness: Hetmaniuk-Lehoucq basis
selection, `svqb`/`svqbDrop` with condition-based dropping, and a backward-stable convergence
criterion. The implementation had none of the three. The criterion was fixed. The hardening that
was done is in [LOBPCG hardening](../perf/syevx.md#lobpcg-hardening-in-syevx).

### Spectrum slicing (EVSL-style): not for batched

Splitting the spectrum into subintervals, each with its own polynomial or rational filter, is
excellent for one huge matrix on many nodes. For a batch of medium matrices the parallelism is
already in the batch dimension, and slicing only adds per-slice orthogonality bookkeeping.

### FEAST, contour integration, shift-and-invert Lanczos: no

All need linear solves with \f$A - zI\f$ per shift. For batched dense input that is a factorization
per shift per matrix, strictly more expensive than the direct route.

## syevx: the original tier plan

Kept for orientation. The status of each tier is on @ref perf_syevx.

| tier | content | outcome |
|---|---|---|
| 0 | `SyevxAlgorithm` + env override, `Auto` routing, `Direct` (`syev` + select), crossover benchmark | done. Routing now measured, not flop-derived |
| 1 | `stebz` (bisection, Index and Value ranges) + `stein` (inverse iteration, cluster reorthogonalization) | done |
| 2 | `syevx_direct_subset`: two-stage reduction -> `stebz` -> `stein` -> narrowed back-transforms | done |
| 3 | `syevx_filtered`: bounds, uniform-degree Chebyshev filter, Rayleigh-Ritz | done, opt-in only |
| 4 | LOBPCG hardening | mostly done. Staircase locking, warm start and Chebyshev preconditioner open |
| 5 | CTA-fused filtered solver for small `n`, gated on a gap estimate | not built |

The original `Auto` proposal routed on `k/n`: `n <= 64` -> Direct, `k/n > 0.25` -> Direct,
`0.02 < k/n <= 0.25` -> DirectSubset, `k/n <= 0.02` -> Filtered, CSR -> LOBPCG. **Superseded**
by the measured rule, which has no `k/n` term (see
[routing thresholds](../perf/syevx.md#syevx-routing-thresholds-as-they-stand)).

## syevx: what not to build

- **MRRR.** Its advantage over bisection + inverse iteration is \f$O(nk^2)\f$ in this regime,
  negligible against \f$O(n^3)\f$ tridiagonalization for medium `n` and irrelevant for small `n`.
  The implementation cost is very large, and LAPACK's own `dsyevr` drops to `dstebz` + `dstein`
  on subset paths.
- **FEAST, contour integration, shift-and-invert.** They need linear solves with \f$A - zI\f$,
  which is strictly more expensive than direct for batched dense input.
- **Spectrum slicing across subintervals.** The parallelism already lives in the batch dimension.
- **A small-matrix (`n <= 64`) direct subset kernel.** It will not beat `syev_cta` (see
  [why Direct is the answer](#syevx-small-n-why-direct-is-the-answer)).
- A `syevr` entry point and generalized `sygvx` ranges, recorded as non-goals in
  @ref design_syevx_range.

## syevx validation plan

Accuracy, per tier, following `benchmarks/syevx_acc.cc` and `tests/syevx_tests.cc`:

- the residual \f$\|Ax_i - \lambda_i x_i\| / (\|A\|\,\|x_i\|)\f$ for every returned pair;
- orthogonality \f$\|V^TV - I\|\f$, the binding constraint for inverse iteration on clusters;
- eigenvalue agreement with LAPACK `dsyevx`/`dstebz` on the same input, including clustered
  spectra (repeated eigenvalues, Wilkinson matrices, graded matrices) and \f$\lambda \approx 0\f$
  cases that expose a non-backward-stable criterion.

Performance sweep axes: `n` in {32 .. 4096} x `k/n` in {0.01, 0.02, 0.05, 0.1, 0.25, 0.5} x
`batch` in {1, 8, 64, 512}, both `JobType` values. Baselines: `syev` (all providers), `syevx`
(all algorithms), and the vendor non-batched subset routine looped over the batch. The measured
crossover map replaced the cost-model estimates (@ref perf_syevx). Double-precision CPU accuracy
failures on the primary machine are usually the broken OpenBLAS Cooperlake `dgemm`, not BatchLAS,
so validate fp64 references against a known-good BLAS first.

## syevx risks, ranked

1. **Inverse iteration on clusters (Tier 1).** Batched, uniform-shape reorthogonalization within
   variable-size clusters. Mitigation: cap cluster size, fall back per matrix (or to `stedc`) when
   exceeded, and make orthogonality a hard test gate. The Wilkinson-matrix `stein` test is that
   gate.
2. **The 3x ceiling may not justify the work.** Mitigation: Tier 0 routing is cheap and delivers
   most of the practical win. Outcome: routing alone gave 3.0x to 3.7x geometric mean, and
   DirectSubset tops out at 2.4x in a narrow window.
3. **Filter degree instability (Tier 3).** A bad spectral bound gives no acceleration or
   overflow. Mitigation: conservative bounds, a clamped degree, divergence detection. Outcome: two
   NaN modes were found and fixed (see [Tier 3 status](../perf/syevx.md#syevx-tier-3-filtered)).
4. **Batched adaptivity vs uniform shapes.** Every adaptive idea in the literature assumes a
   single matrix. Decide max-over-batch or bucketing per feature, and do not import adaptivity
   uncritically.
5. **Two-stage back-transform narrowing.** Resolved, though not as expected: there was no Q2 to
   narrow at `kd = 1`, and Q1's `ormqr` narrowed cleanly. The square-matrix assumption did exist,
   in `apply_phase_rows`, and caused an out-of-bounds write on an n x k block. Fixed.

## syevx implementation notes

Rationale that used to live as comment blocks in `src/extensions/syevx*.cc`, `stein.cc` and
`stebz.cc` (moved 2026-09-30). The code keeps the invariant and points here. Measurements stay on
@ref perf_syevx, and range-selection rules on @ref design_syevx_range.

### syevx dispatcher: problem arguments are validated before dispatch

Preconditioner and range arguments describe the *problem*, not the algorithm, so
`validate_syevx_preconditioner_params` and `validate_syevx_range_params` run in `syevx` and
`syevx_buffer_size` before `syevx_select_algorithm`. The preconditioner checks used to live
inside `syevx_lobpcg`. That was equivalent only while every path led there. Once dense input
routed to Direct/DirectSubset, an illegal combination on a dense matrix silently reached a solver
that ignores preconditioners (see [Tier 0](../perf/syevx.md#syevx-tier-0-dispatch-and-the-direct-path)).

Two range rules from the plan are deliberately not in the validator:

- **"`select == Extremal` and `order` contradicts `find_largest`" is not implemented.**
  `SortOrder` has only `Ascending` and `Descending`, and `SyevxParams::order` defaults to
  `Ascending`, so an explicit `Ascending` cannot be told from an unset one. The rule as written
  would reject the library's own defaults (Extremal + `find_largest = true` + Ascending), which is
  nearly every existing call, and the Python `SyevxOptions` sends every field on every call.
  `order` is documented as ignored for Extremal instead (see
  [normalization and validation](syevx-range-selection.md#syevx-range-normalization-and-validation)).
- **"A non-extremal range may not resolve to LOBPCG or Filtered, and sparse input may not ask for
  one" lives in `syevx_select_algorithm`**, because it has to distinguish an explicit
  `SyevxParams::method` (throws) from a `BATCHLAS_SYEVX_ALGORITHM` override (degrades), and the
  selector is the only place that sees both.

The `value_range_reportable` argument of the range validator is false only for the solve
overloads that have no `m` output. The sizing entry points pass true: they write no counts, and
rejecting a Value range there would make sizing a Value-range solve impossible. A short `m` span
is also rejected on the host, because the device write it causes has no diagnostic in a release
build (`Span::operator[]`'s assert is compiled out).

### syevx dispatcher: which preconditioner is legal at which end

- **ILU(k)** approximates \f$A^{-1}\f$, so it accelerates only the smallest eigenpairs. For the
  largest it damps exactly what is sought. `find_largest` with an ILU(k) factor throws.
- **`Jacobi`** \f$= \mathrm{diag}(A)^{-1}\f$ is an approximate \f$A^{-1}\f$ as well, only cruder, so
  it inherits the restriction verbatim. This is measured, not theoretical: forcing it on with
  `find_largest` turned 21 to 47 iterations into 127 to 300, that is non-convergence at the cap
  (see [the Jacobi preconditioners](../perf/syevx.md#lobpcg-jacobi-preconditioners)).
- **`JacobiShifted`** \f$= (\mathrm{diag}(A) - \lambda I)^{-1}\f$ is a different operator. Its shift
  is the *current Ritz value*, so it is a diagonal approximation to \f$(A - \lambda I)^{-1}\f$ and
  amplifies whatever is near \f$\lambda\f$, which is the wanted end by construction at either end
  of the spectrum. Allowing it with `find_largest` is deliberate, backed by the same sweep (0.85x
  to 1.2x on random symmetric input either way).
- An explicit `preconditioner_type` must agree with the ILU(k) fields. Anything else silently
  drops one of the two requests: either a factor the caller built at real cost is never applied,
  or a family is requested that has nothing behind it. Both combinations throw.

### syevx dispatcher: environment override versus explicit request

`algorithm_from_env` resolves `BATCHLAS_SYEVX_ALGORITHM` against `SyevxParams::method` and
reports through `from_env` which of the two won. The distinction is load-bearing: an environment
default **degrades** where an explicit request **throws**. The variable exists so that a whole
application or test suite can be forced onto one algorithm for diagnosis, and aborting on the
first call it cannot serve would make that sweep impossible rather than informative.

- A non-extremal range under `BATCHLAS_SYEVX_ALGORITHM=lobpcg` or `filtered` degrades to Direct
  with a once-per-process warning. The substitute is always Direct, the universal fallback (every
  scalar type, every range, every `jobz`), not the shape heuristics, because a diagnostic sweep
  wants one substitute rather than a shape-dependent one. The rule is recorded as
  [throw, do not degrade](syevx-range-selection.md#syevx-range-throw-do-not-degrade).
- **Trap:** the environment wins whenever the variable is set at all, including to an
  unrecognized value. That parses to `Auto`, that is "ignore `params.method` and use the
  heuristics". This is pre-existing behaviour and is preserved deliberately.
- `syevx_select_preconditioner` applies the same asymmetry to `BATCHLAS_SYEVX_PRECONDITIONER`. A
  configured ILU(k) factor wins over any environment default, because it is the strongest signal
  of intent and was paid for before the call. `ILUK` from the environment is not actionable:
  there is no factor, and `syevx` will not build one behind the caller's back, which would need
  CSR input and `find_largest = false`, neither of which the environment can know. `Jacobi` from
  the environment is dropped for `find_largest` calls instead of throwing.

### syevx: why the preconditioner environment variable is only a default

Precedence for `SyevxPreconditioner` differs deliberately from `SyevxAlgorithm`.
`BATCHLAS_SYEVX_ALGORITHM` overrides `params.method`; `BATCHLAS_SYEVX_PRECONDITIONER` only supplies
the *default* that `Auto` resolves to, and never overrides an explicit `preconditioner_type`. An
algorithm can always be substituted for another, because that changes only performance. A
preconditioner cannot: an ILU(k) factor a caller built and handed in has no substitute, and
silently ignoring it, or silently ignoring a request for one, would be a correctness surprise
rather than a performance one. Only the LOBPCG path uses the preconditioner; Direct, DirectSubset
and Filtered ignore it.

Neither Jacobi form is chosen by `Auto`. On the matrices measured, neither is a free win (see
[the Jacobi preconditioners](../perf/syevx.md#lobpcg-jacobi-preconditioners)), so picking one
implicitly would be a regression for somebody. `JacobiShifted` in particular is offered because
the constant-diagonal case is provably a no-op and the general case is safe, not because it was
found to pay. The other half of the reason is on the perf page: it is the only Jacobi form legal
for `find_largest`. Its measured effect there is neutral on random symmetric input (0.85-1.2x
iterations) and 0.2-0.9x on graded input, against 2.1-7.3x fewer iterations for plain `Jacobi` on
graded input. (Moved from the `SyevxPreconditioner` comment in `include/batchlas/blas/enums.hh`,
2026-09-30.)

### syevx dispatcher: sparse input and an explicit Filtered request

Direct and DirectSubset tridiagonalize a dense A and are not defined on CSR, so an explicit
request for them on sparse input degrades to LOBPCG. `Filtered` is not a dense path:
`syevx_filtered` is instantiated for every `MatrixFormat` (`BATCHLAS_FOR_EACH_MATRIX_FORMAT_2`)
and has real CSR branches through `spmm`, because a Chebyshev iteration needs nothing from A but
the ability to multiply by it. So an explicit `Filtered` on CSR is honoured. `Auto`, and every
dense-only request, still lands on LOBPCG.

This branch used to `return LOBPCG` unconditionally, above the explicit-method switch. It silently
discarded `params.method = Filtered` on sparse input, which is the one thing the rest of the
selector is careful never to do, and it meant every "run the suite forced onto Filtered" sweep
passed on CSR inputs without once running Filtered. The measurement that exposed it is in
[Filtered vs LOBPCG on CSR](../perf/syevx.md#syevx-filtered-vs-lobpcg-on-csr). The branch must
stay below the non-extremal range check, or sparse input with an interior range would get LOBPCG's
extremal answer instead of an exception.

The LOBPCG and Filtered arms take no `m` argument. They only ever see an Extremal range, so the
count is static, and `syevx` fills `m` itself with the resolved block size. The fill is submitted
*before* the solve, so the solve's Event, which is what the caller waits on, covers it on the
in-order queue the library assumes. It aliases nothing the solvers touch.

### syevx Direct: the range is checked at its own entry point

`syevx_direct` and `syevx_direct_subset` are public entry points in their own right: the test
suite calls them directly, and so may a caller who wants to pin the algorithm. So they re-check an
Index or Value range themselves. An out-of-range index block would otherwise reach a device kernel
that indexes an n-entry eigenvalue array with `il..iu`. `syevx_resolve_range` clamps as a second
line of defence, so the throw is about telling the caller rather than memory safety, and both are
wanted. The wording and exception type (`std::invalid_argument`, not the `std::runtime_error`
the shape checks use) match `validate_syevx_range_params`. DirectSubset validates before the
reduction for a second reason: `stebz` would otherwise reject the range two layers down, after
the whole \f$O(n^3)\f$ reduction had run.

`neigs` above `n` is accepted by both: it is a capacity, clamped inside `syevx_resolve_range`, and
the tail of `W` and `V` goes unwritten. For `syevx_direct` this was the one place an existing
call's outcome changed from a throw to a success when capacity semantics landed. `neigs = 0` is
still rejected by DirectSubset, because `stein` requires `k >= 1`.

### syevx DirectSubset: reported as converged, with a caveat

`syevx_direct_subset` clears `info` to 0 ("converged") for every item. `sytrd_sy2sb` and
`sytrd_sb2st` are direct, but the two tridiagonal solvers are not: `stebz` bisects to a tolerance
and cannot tell its three loop exits apart afterwards, and `stein` runs a **fixed** number of
inverse iterations with no convergence test at all (LAPACK's `?stein` reports how many vectors
failed; nothing here measures it). A real flag would need a residual check, which is new
arithmetic rather than surfacing something that exists, so it was left out of the work package
that added `info` and recorded as deferred. Writing 0 rather than leaving the span untouched is
the deliberate part: an uninitialised span is worse than a conservative one, because the caller
cannot tell "converged" from "never written".

### stein: one work-item per vector

Phase 1 of `stein` solves \f$(T - \lambda I)x = b\f$ a few times from a deterministic
pseudo-random start, with a tridiagonal LU with partial pivoting (LAPACK `dgttrf`/`dgttrs`). The
factorization depends on \f$\lambda\f$, so it is per vector and serial in `n`, hence one work-item
rather than one work-group per vector. That is affordable because for medium `n` the tridiagonal
stage is far off the critical path (see
[medium matrices](#syevx-for-batches-of-medium-matrices)). Phase 2 reorthogonalizes vectors whose
eigenvalues form a cluster by modified Gram-Schmidt: inverse iteration alone does not deliver
orthogonality on clusters, and that \f$O(nk^2)\f$-within-clusters cost is the price bisection pays
relative to MRRR. The per-item `kb` bounds of both phases are in
[stein's per-item count bound](syevx-range-selection.md#syevx-range-steins-per-item-count-bound).

### syevx Filtered: the scaled Chebyshev filter

One outer iteration of `syevx_filtered`:

```
Y = p_m(A) X            Chebyshev filter, m matvecs
Y = ortho(Y)
H = Y^H A Y             projected problem
H = Z diag(theta) Z^H   syev
X = Y Z                 Ritz vectors
```

\f$p_m\f$ is the degree-\f$m\f$ Chebyshev polynomial of the spectrum mapped so that the *unwanted*
interval falls in \f$[-1, 1]\f$, where \f$|T_m| \le 1\f$, while the wanted end falls outside, where
\f$T_m\f$ grows like \f$\cosh(m\,\mathrm{acosh}\,x)\f$. A handful of iterations does what many
unpreconditioned Krylov steps could not. The cut is the first *unwanted* Ritz value, so the damped
interval is exactly the part of the spectrum being rejected.

Why this rather than more LOBPCG: the filter needs no preconditioner and no factorization, only
matvecs. Unpreconditioned LOBPCG is documented to stagnate (see
[Chebyshev-Davidson](#chebyshev-davidson-consider-as-the-lobpcg-replacement-for-unpreconditioned-problems)),
and the ILU(k) preconditioner `syevx_lobpcg` can build is valid only for the smallest eigenpairs.
Filtering has no such restriction.

**The recurrence is the scaled Zhou/Saad form**, not the textbook one:

\f[
Y_1 = \frac{\sigma_1}{e}(A - cI)X, \qquad
Y_{j+1} = \frac{2\sigma'}{e}(A - cI)Y_j - \sigma\sigma' Y_{j-1}.
\f]

\f$T_m\f$ evaluated directly overflows quickly: with a spectrum reaching \f$x = 3\f$ on the mapped
axis, \f$T_{25}(3) \approx 10^{38}\f$, which is float infinity. Carrying the \f$\sigma\f$ factors keeps
every intermediate near unit magnitude. The block is orthonormalized immediately afterwards, so the
overall scale is irrelevant; only the ratio between wanted and unwanted components matters.

The block has guard directions (`extra_directions`, default `max(2, k/4)`): without any, the cut
sits exactly at the edge of the wanted block and the last wanted pair converges slowly. The start
block comes from a hash of the index, so it does not depend on host RNG state and is reproducible
across runs and backends. The per-iteration host read of the convergence flag is a known cost, see
[Tier 3 status](../perf/syevx.md#syevx-tier-3-filtered).

### syevx Filtered: normalising the filter and capping its degree

**Normalise at the extreme Ritz value.** `far`, the point where the scaled polynomial equals 1,
must be the extreme *Ritz* value, not the Gershgorin bound. Gershgorin overestimates the spectral
radius of a random symmetric matrix badly (\f$O(n)\f$ against the true \f$O(\sqrt n)\f$), and
normalising there divides by \f$T_m\f$ of a point far outside the spectrum. At degree 40 that
underflows the whole block to zero, and orthogonalizing a zero block yields NaN. Normalising at the
wanted end keeps \f$p \approx 1\f$ exactly where the wanted vectors live.

**Cap the degree by precision.** The filter amplifies the most-wanted direction over the
least-wanted one by

\f[
\frac{\cosh(d\,\mathrm{acosh}\,y_{far})}{\cosh(d\,\mathrm{acosh}\,y_{edge})}.
\f]

Once that ratio exceeds what the working precision holds, the least-wanted columns are numerically
swamped, the block is rank-deficient, and the Cholesky-based `ortho` returns NaN rather than a slow
answer. Bounding \f$d\f$ keeps every outer iteration well-conditioned; the lost sharpness is
recovered by another iteration, which is cheap by comparison. The cap is

\f[
d_{cap} = \frac{-\tfrac14 \log \epsilon}{\mathrm{acosh}\,y_{far} - \mathrm{acosh}\,y_{edge}},
\f]

a ratio budget of \f$\epsilon^{-1/4}\f$ (about 300 in float). A budget of \f$\epsilon^{-1/2}\f$ was
measured to be too generous: the block still went rank-deficient and Chol2 returned NaN at
degree 40. When \f$y_{edge} \le 1\f$ the least-wanted direction is inside the damped band, where
\f$|T_d| \le 1\f$. That is the *worst* case for the ratio, not a case to skip: growth is then the
full \f$\cosh(d\,\mathrm{acosh}\,y_{far})\f$, so the edge term is taken as 0. The first version
waived the cap there. Each item's cap is reduced by **min** across the batch so the recurrence
length, and with it every GEMM shape, stays uniform.

### syevx Filtered: the problem-derived degree

Behind `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=1` (off by default; see
[the measurement](../perf/syevx.md#syevx-automatic-chebyshev-filter-degree)). Inside the damped
band \f$|T_d| \le 1\f$. The least-amplified *wanted* direction sits at mapped coordinate
\f$y_{edge} > 1\f$ and grows like \f$\cosh(d\,\mathrm{acosh}\,y_{edge}) \approx e^{d r}/2\f$ with
\f$r = \mathrm{acosh}\,y_{edge}\f$. One outer iteration therefore shrinks the unwanted content by
about \f$e^{-dr}\f$, and the degree that would finish in one more iteration is

\f[
d_{need} = \frac{\log(\text{worst residual} / \text{target})}{r},
\f]

in the scaled units the convergence test uses. This is the geometry the precision cap already
computes, and the cap stays an upper bound on top of it, because it guards the measured failure
mode (rank-deficient block, NaN from Cholesky). The per-item \f$d_{need}\f$ is clamped to
`[kAutoDegreeMin, kAutoDegreeMax] = [10, 40]` and reduced by **max** across the batch, then trimmed
by the batch-minimum cap. The lower bound is the old constant, so turning the derivation on never
makes a step *less* selective than before. The upper bound is the usual ChASE-style ceiling: past
roughly 40 the extra matvecs stop paying for themselves and the precision cap tends to bind anyway.

Degree precedence: `BATCHLAS_SYEVX_FILTER_DEGREE` > `params.filter_degree` > derived >
`kDefaultFilterDegree`. Either explicit value pins the degree and turns the derivation off, and
`BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=0` restores the constant exactly. **Trap:** that variable and
`BATCHLAS_SYEVX_BOUNDS_LEGACY` are parsed with `atoi`, not `env_truthy`, so `true` and `on` read
as false. Kept as is, so no existing spelling changes meaning.

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
