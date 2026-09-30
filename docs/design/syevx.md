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
