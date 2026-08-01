# SYEVX: Research Findings and a Performance Plan for Partial Symmetric Eigensolves

Status: **design only.** No code in this document has been implemented.

Scope: what "high-performance SYEVX" should mean in BatchLAS, for two regimes the
user asked about separately — *batches of small matrices* and *batches of medium
matrices* — plus a concrete optimization list for the existing LOBPCG solver that
currently sits behind the `syevx` name.

The central claim of this document is that **`syevx` in BatchLAS is currently one
algorithm wearing a name that should belong to a dispatcher over four.** The
single biggest available win is not a new kernel; it is routing.

---

## Table of contents

1. [What `syevx` is today](#1-what-syevx-is-today)
2. [The cost model — where the ceiling actually is](#2-the-cost-model)
3. [Regime A: batches of small matrices](#3-regime-a-small)
4. [Regime B: batches of medium matrices](#4-regime-b-medium)
5. [The vendor gap](#5-the-vendor-gap)
6. [Algorithm survey and verdicts](#6-algorithm-survey)
7. [LOBPCG: concrete defects and fixes in the current code](#7-lobpcg-fixes)
8. [Implementation plan, tiered](#8-implementation-plan)
9. [What not to build](#9-what-not-to-build)
10. [Validation and benchmarking plan](#10-validation)
11. [Risks, ranked](#11-risks)
12. [Sources](#12-sources)

---

## 1. What `syevx` is today

`src/extensions/syevx.cc` is a batched LOBPCG (Locally Optimal Block Preconditioned
Conjugate Gradient), instantiated for both `MatrixFormat::Dense` and
`MatrixFormat::CSR`, for all scalar types and all backends.

Structure per iteration (line references to `src/extensions/syevx.cc`):

| Step | Lines | Cost (dense, block width `k`) |
|---|---|---|
| residual `R = AX − Xdiag(λ)`, norms, best-snapshot | 279–375 | `O(nk)` bandwidth, `2·neigs` group reductions |
| host sync + convergence read | 375, 424–430 | **full pipeline drain, every iteration** |
| optional ILU(k) apply (+2 full copies) | 435–443 | `O(nk)` × 3 |
| `ortho(R, XP)` | 446 | `O(nk²)` |
| `A·R` (or `A·P` on restart) | 467–474 | `2n²k` dense / one SpMM |
| `StAS = Sᵀ(AS)` | 482 | `2n(3k)² = 18nk²` |
| `syev` on `3k × 3k` | 486–490 | batched small/medium dense ED |
| 4 × `gemm` `n×3k×k` to form X,AX,P,AP | 552–562 | `8nk²` |

Per iteration ≈ **`2n²k + ~30nk²`** flops on dense, dominated by the single matvec
block when `k ≪ n`.

The important structural facts:

- **It is the only `syevx` path.** A dense caller asking for 10 % of the spectrum
  runs LOBPCG, which for that ratio is essentially guaranteed to be slower than
  just calling `syev` and throwing eigenpairs away (§2).
- It has **no locking/deflation**, **no polynomial acceleration**, and its only
  preconditioner is ILU(k), which is a sparse-matrix construct.
- `extra_directions` defaults to `0`, i.e. the trial block is exactly `neigs` wide
  — the setting LOBPCG converges worst under.
- Convergence is measured as `‖r‖ / (‖x‖·|λ|)` (line 326), which is not backward
  stable and degenerates as `λ → 0`.

None of this is wrong for the sparse case it was clearly built for. It is wrong as
the universal answer to "give me `k` eigenpairs".

---

## 2. The cost model

This section determines what is worth building. All counts are flops for real
symmetric `n × n`, `k` wanted eigenpairs, leading-order only.

### 2.1 Full eigendecomposition (the thing we must beat)

One-stage, with vectors (`dsyevd`-shaped):

```
sytrd    (4/3)n³      ~50 % SYMV → memory-bound
stedc    ~(4/3)n³     highly variable, deflation-dependent
ormtr     2n³         GEMM
--------------------------
total    ~4.7n³
```

Two-stage (`sy2sb` + `sb2st`, which BatchLAS already has):

```
sy2sb    (4/3)n³      almost entirely SYR2K/GEMM → compute-bound, fast
sb2st    O(n²b)       cheap in flops, poorly parallel, latency-bound
stedc    ~(4/3)n³
apply Q₂  2n³         back-transform through bulge-chasing reflectors
apply Q₁  2n³         back-transform through the band reduction
--------------------------
total    ~6.7n³ flops, but far better flop efficiency
```

Two-stage's well-known weakness is that it **pays the back-transform twice**. That
is exactly the term that scales with the number of wanted vectors.

### 2.2 Direct subset (the thing we should build)

```
sy2sb          (4/3)n³
sb2st          O(n²b)
tridiag subset O(n·k·40)  bisection  +  O(nk) per inverse-iteration step
apply Q₂       2n²k
apply Q₁       2n²k
--------------------------
total ≈ (4/3)n³ + 4n²k
```

Ratio against a one-stage full ED at `k = 0.05n`:

```
(1.33 + 0.20) n³   vs   4.7 n³    →   ~3.0×
```

and eigenvalues-only (`k` vectors not requested):

```
1.33 n³  vs  4.7 n³              →   ~3.5×
```

**So ~3× is the honest ceiling for a direct dense SYEVX, and it is reached
asymptotically as `k/n → 0`.** It is not 10×, because tridiagonalization is `O(n³)`
no matter how few eigenpairs you want. But it is a *good* 3×: the work being
deleted (`stedc` and the two `ormqr` back-transforms) is the memory-bound,
badly-scaling part, so the wall-clock ratio should exceed the flop ratio.

Two-stage's double back-transform, its main liability for full ED, **disappears in
the subset regime.** This is the key asymmetry that makes SYEVX worth building on
top of the machinery already in `syev_two_stage.cc`.

### 2.3 Iterative methods on dense input

Chebyshev-filtered subspace iteration, degree `d`, block width `m ≈ 2k`, one outer
pass:

```
d · 2n²m  =  4dn²k
```

Break-even against `(4/3)n³`:  `4dk < 1.33n` → `k < n/(3d)`. With a realistic
`d = 15`: **`k < 2.2 % of n`.**

LOBPCG on dense, `it` iterations:

```
2·it·n²k   (+ O(it·nk²))
```

Break-even: `it·k < 0.67n`. With a realistic unpreconditioned `it = 30`:
**`k < 2.2 % of n`** — the same threshold, by coincidence of the constants.

### 2.4 The conclusion that drives everything else

> For **dense** input, iterative methods (LOBPCG, Chebyshev filtering) can only win
> below roughly **`k/n ≈ 2 %`**. Above that, a direct subset solver wins, and above
> maybe 25–30 % even a plain full `syev` wins because the subset machinery's
> constants and the loss of `stedc` deflation eat the margin.

This is a *three-way* crossover, and BatchLAS currently implements only the branch
that is correct in the narrowest of the three bands.

---

## 3. Regime A: batches of small matrices

The user's prior — "SYEVX for small matrices probably almost never makes sense" —
is **correct, with one exception.**

### 3.1 Why it doesn't make sense

For `n ≤ 32–64` in a CTA-resident solver, the run time is not flop-bound. It is
bound by work-group occupancy, shared-memory traffic, and the sweep count of the
tridiagonal solver. Cutting the back-transform from `k = n` to `k = 2` columns
removes work that was already hidden behind the tridiagonalization and the QR
sweeps. Measured against `syev_cta`/`syev_jacobi_cta`, a subset variant should be
expected to land within noise.

**Recommendation: for `n ≤ 64`, `syevx` should call `syev` and select.** Expose it
for API compatibility, not for speed. Add the dispatch, document the reason, and do
not build a small-matrix subset kernel.

### 3.2 The exception: filtered CTA solver for `k ≪ n`, `n ≤ 64`

There is one route that skips tridiagonalization altogether and is therefore not
subject to the argument above:

- Hold `A` (`n ≤ 64`, so ≤ 32 KB in fp64) in shared memory for the whole kernel.
- Run `d` steps of a block Chebyshev filter `p_d(A)·X` entirely in registers/SLM.
  Each step is one `n×n × n×k` product — for `n=32, k=2` that is 4 K flops.
- Rayleigh–Ritz on a `k×k` (or `2k×2k`) matrix, solved in-register.

For `n=32, k=2, d=12` this is ~50 K flops against `syev_cta`'s tridiagonalization
plus 100–400 QR sweeps. A 3–8× win is plausible *if* the spectrum is well
separated, which is the load-bearing caveat: with a cluster at the wanted end,
the filter degree needed to separate them explodes and the method degrades
smoothly into "slower than full ED".

**Verdict: build this last (Tier 5), gate it behind a measured gap estimate, and
treat it as opportunistic.** It is the only small-matrix subset idea with a real
mechanism behind it.

### 3.3 The thing small matrices *do* benefit from

Interestingly, the subset machinery's *building block* helps the small-matrix
**full-spectrum** path. A bisection-based tridiagonal eigenvalue kernel (§8, Tier 1)
assigns one eigenvalue per lane, is embarrassingly parallel, has no shared-memory
ping-pong, and has a fixed iteration count (~40–60 Sturm sequence evaluations for
fp64 to machine precision). For `n ≤ 32` with a 32-wide sub-group this is a natural
fit and sidesteps the STEDC merge serialization already identified as a bottleneck
in this codebase (see `stedc-merge-optimization-plan`).

So: **build bisection for SYEVX, then also use it as an eigenvalues-only
alternative to STEDC for small `n`.** Two payoffs, one kernel.

---

## 4. Regime B: batches of medium matrices

This is where SYEVX earns its keep. `n ≈ 128–4096`, batch of 8–1000.

The plan is the direct route of §2.2, assembled almost entirely from parts that
already exist in the tree:

| Stage | Existing component | Change needed |
|---|---|---|
| dense → band | `sytrd_sy2sb` (`src/extensions/sytrd_sy2sb.cc`) | none |
| band → tridiagonal | `sytrd_sb2st` (`src/extensions/sytrd_sb2st.cc`) | none |
| tridiagonal subset eigenvalues | — | **new: `stebz` (bisection)** |
| tridiagonal subset vectors | — | **new: `stein` (inverse iteration)** |
| back-transform through Q₂ | `sytrd_sb2st` reflector application | narrow to `k` columns |
| back-transform through Q₁ | `ormqr_blocked` (`internal/ormqr_blocked.hh`) | narrow to `k` columns |

Both back-transform steps in `syev_two_stage.cc` already take a matrix of vectors;
narrowing them to `k` columns is a shape change, not an algorithm change. The
genuinely new code is the tridiagonal subset solver.

**And that solver is not on the critical path.** At `n = 1024, k = 64`, bisection
costs ~`40·n·k = 2.6 MFlop` against sy2sb's `1.4 GFlop`. It needs to be *correct
and batched*, not fast. This strongly argues for the simple, well-understood
bisection + inverse iteration pair over MRRR (§9).

### 4.1 Batched-specific wrinkles

- **Uniform shapes.** Different batch members converge at different rates and may
  have different cluster structure. Any per-matrix adaptivity (variable filter
  degree, variable active block width) breaks the uniform shapes that batched GEMM
  needs. Resolve by taking the max over the batch, or by bucketing the batch.
- **Inverse iteration on clusters.** LAPACK's `stein` reorthogonalizes within
  clusters where `λᵢ₊₁ − λᵢ < 10⁻³·‖T‖`. Cluster sizes vary per matrix; the batched
  kernel must handle a worst-case cluster size or fall back per-matrix. This is the
  main correctness risk of Tier 1 (§11).
- **Memory.** The current LOBPCG allocates four `n × 3k × batch` buffers. At
  `n=4096, k=64, batch=32`, fp64, that is ~800 MB. The direct path needs only the
  band `n×b`, the tridiagonal `2n`, and `n×k` output — an order of magnitude less.
  This alone may make medium-`n` batched partial eigensolves feasible where they
  currently OOM.

---

## 5. The vendor gap

Worth stating because it defines the opportunity:

- `cusolverDnXsyevdx` — subset, **not batched**.
- `cusolverDnXsyevBatched` — batched, **full spectrum**. (It replaced
  `cusolverDnSsyevjBatched`, whose `n ≤ 32` cap was a leftover from the old Jacobi
  implementation; PyTorch's dispatch still carries the stale `n ≤ 32` gate, see
  pytorch/pytorch#175585.)
- `cusolverDnSsyevjBatched` / `gesvdjBatched` — batched, full spectrum, `n ≤ 32`.
- MAGMA `syevdx` — subset via divide-and-conquer, **not batched**.

**No vendor ships a batched subset symmetric eigensolver.** That is the whole
product argument for Tier 2.

---

## 6. Algorithm survey

Ranked by fit to BatchLAS, with verdicts.

### 6.1 Bisection + inverse iteration (`stebz` + `stein`) — **build**

Bisection computes any subset of the tridiagonal spectrum from Sturm sequence sign
counts. Every eigenvalue is independent, so it is embarrassingly parallel and maps
onto a lane-per-eigenvalue GPU kernel with no communication. It gives eigenvalues
to arbitrary (including reduced) accuracy at proportionally reduced cost, and
incomplete Sturm sequences let you terminate early when only a few eigenvalues are
wanted. Inverse iteration then recovers vectors at `O(n)` per vector per step.

Its known weakness — loss of orthogonality on tight clusters, requiring explicit
reorthogonalization, potentially `O(nk²)` — is the reason MRRR exists. In our
regime `k ≪ n` and `O(nk²) ≪ O(n³)`, so the weakness does not bind.

### 6.2 MRRR (`stemr`/`dsyevr`) — **do not build** (see §9)

`O(n²)` total with no reorthogonalization, via relatively robust representations
and twisted factorizations. It is the theoretically right answer and it is a
notorious implementation (differential qd transforms, RRR trees, careful shift
selection). Note also that LAPACK's `dsyevr` itself falls back to `dstebz` +
`dstein` when only a subset is requested on some paths — i.e. the reference
implementation agrees that bisection is the pragmatic subset choice.

### 6.3 Chebyshev-filtered subspace iteration (ChFSI / ChASE) — **build for `k/n ≲ 2 %`**

Apply a Chebyshev polynomial that maps the wanted spectral interval to large values
and the rest to `[−1,1]`, then Rayleigh–Ritz. All the work is GEMM (dense) or SpMM
(sparse), which is exactly what a batched library wants. ChASE demonstrates this
scaling to 526 A100s and optimizes the filter degree per accuracy target to
minimize flops. Recent work (AdaPolySI, ACM ICS 2026; residual-based ChFSI,
arXiv:2503.22652) adapts the degree per-eigenpair as the iteration progresses and
tolerates inexact matvecs, both directly relevant.

Requirements: a spectral bound estimate (a few Lanczos steps) and reasonable
separation. For batched work, keep the degree uniform across the batch.

### 6.4 Chebyshev-Davidson — **consider as the LOBPCG replacement for unpreconditioned problems**

Zhou & Saad's Chebyshev–Davidson uses the polynomial filter as the *expansion*
operator in a Davidson framework, so it needs no preconditioner at all, and the
block variant with inner–outer restart gets by with roughly half the subspace
dimension of ARPACK/TRLan. The relevant comparison for us: **unpreconditioned
LOBPCG is documented to stagnate**, with residuals oscillating around a floor
rather than converging. Since BatchLAS's `syevx` has no preconditioner unless the
caller supplies an ILU(k), this is not a hypothetical failure mode — it is the
default configuration.

### 6.5 LOBPCG — **keep for preconditioned sparse; harden**

Its domain is sparse `A` with a good preconditioner, where no `O(n³)` alternative
exists. Duersch, Shao, Yang & Gu (SISC 2018) is the reference for making it robust:
the Hetmaniuk–Lehoucq basis selection, `svqb`/`svqbDrop` with condition-based
dropping, and a backward-stable convergence criterion. The current implementation
has none of the three (§7).

### 6.6 Spectrum slicing (EVSL-style) — **not for batched**

Split the spectrum into subintervals and extract each independently, with a
polynomial or rational filter per slice. Excellent for one huge matrix on many
nodes; for a batch of medium matrices the parallelism is already in the batch
dimension and slicing only adds per-slice orthogonality bookkeeping.

### 6.7 FEAST / contour integration, shift-and-invert Lanczos — **no**

Both require solving linear systems with `(A − zI)` per shift. For batched dense
that means a factorization per shift per matrix — strictly more expensive than the
direct route.

---

## 7. LOBPCG: concrete defects and fixes

Ordered by expected impact. Line numbers refer to `src/extensions/syevx.cc` as of
this branch.

### 7.1 A full host synchronization every iteration — **highest impact**

```
375:  residual_evt.wait_and_throw();
424:  for (int64_t b = 0; b < batch_size; ++b)
425:      if (converged_flags[b] == 0) { all_converged = false; break; }
```

Every iteration drains the device pipeline and reads unified memory from the host.
With ~30 iterations this is 30 forced round-trips. For small `n` and large batch —
precisely the regime BatchLAS targets — this can dominate total run time.

Fix: reduce the per-batch flags to a single device-side counter, and read it back
only every `check_every` iterations (default 4–8). Overshooting by a few iterations
is far cheaper than 30 pipeline drains. A follow-on is to capture the whole
iteration body in a SYCL graph.

### 7.2 Instrumentation does host-side unified-memory reads every iteration

Lines 378–420 loop over `batch_size × neigs` on the host reading `residuals[]`,
`best_residuals[]` and `lambdas[]`. Even when the histories are wanted, these should
be device-side stores from the residual kernel into the history spans. As written,
enabling instrumentation changes the performance characteristics of the thing being
instrumented.

### 7.3 The residual kernel serializes over `neigs`

```
312:  for (size_t i = 0; i < neigs; i++){
320:      const float_type r_sum = sycl::reduce_over_group(...);
321:      const float_type x_sum = sycl::reduce_over_group(...);
```

That is `2·neigs` sequential group reductions, each a full work-group barrier. For
`neigs = 64` that is 128 barriers in one kernel. Rewrite as a single pass over
`n × neigs` accumulating per-column partials in local memory, then one reduction
tree over all columns at once — or assign one sub-group per column.

### 7.4 The convergence criterion is not backward stable

```
326:  const auto denom = x_norm * sycl::fabs(eigval);
```

Relative-to-`|λ|` breaks down as `λ → 0`: the residual is divided by something
approaching zero and the eigenpair never registers as converged, regardless of how
good it is. Duersch et al. give the fix — measure `‖r‖ / (‖A‖·‖x‖)`, or
`‖r‖ / (‖Ax‖ + |λ|·‖x‖)`. `‖A‖` can be estimated once at setup (BatchLAS already
has `norm` in `src/extra/norm.cc`).

### 7.5 No soft locking or deflation

Converged columns continue to be multiplied by `A`, orthogonalized, and included in
the projected problem for every remaining iteration. The batched setting makes the
classic soft-locking recipe awkward — different batch members converge at different
times and shrinking the block per-matrix breaks batched GEMM shapes.

Two workable variants:
- **Column masking (cheap, stability-only win):** zero the residual columns that
  have converged so they stop polluting the search space. Same flops, better
  conditioning of `[X,P,R]`.
- **Batch-wide staircase (real win):** shrink the active block width to the next
  step (e.g. multiple of 8) only when *every* batch member has converged that many
  columns. Preserves uniform shapes and recovers most of the benefit when
  convergence is correlated across the batch, which it usually is for
  same-distribution batches.

### 7.6 `extra_directions` defaults to 0

A guard block (`extra_directions ≈ 0.1–0.25 · neigs`) is standard practice and
often cuts iteration count substantially, at a cost linear in the extra width. This
is a one-line default change and should be the first experiment run.

### 7.7 An unnecessary staging copy in the preconditioner path

```
437:  MatrixView<...>::copy(ctx, R_contiguous, R);
438:  ctx.wait_and_throw();
439:  iluk_apply<B>(ctx, *params.preconditioner, R_contiguous, R_preconditioned);
440:  MatrixView<...>::copy(ctx, R, R_preconditioned);
441:  ctx.wait_and_throw();
```

`R_contiguous` is a full `n × k × batch` staging buffer (allocated at 112–122)
whose only purpose is to repack `R` into a packed-batch layout. **It is not
needed:** `iluk_apply` already indexes through `kernel_view()` with explicit
`stride_` and `ld_` (`src/extensions/iluk.cc:449–472`), so it accepts `R`'s
`n·3k` batch stride directly. Dropping `R_contiguous` removes one full copy per
iteration *and* its workspace allocation.

The two `ctx.wait_and_throw()` calls (438, 441) are additional full pipeline
drains per iteration on top of §7.1, and appear unnecessary given the queue
ordering.

### 7.8 Column-reversal kernels for `find_largest`

Lines 226–247 and 506–529 launch dedicated kernels whose entire job is to reverse
the column order of a `k×k` (resp. `Nvecs×k`) block. Each is a batch-wide launch
doing trivial work — pure launch overhead, twice per iteration.

Fixes, in increasing order of effort: (a) index `Z`'s columns in reverse when
forming the view, so no data movement happens at all; (b) have the projected `syev`
emit descending order; (c) fold the permutation into the subsequent GEMM.

### 7.9 `S.fill_random` fills 3× more than needed

Line 187 fills the whole `n × 3k × batch` buffer when only the `X` block is read.
Beyond the wasted bandwidth, random initialization is a weak start: a few block
power-iteration steps, or reuse of the previous solution when solving a sequence of
related problems (the ChASE use case), converge markedly faster.

### 7.10 No cheap preconditioner

The only preconditioner is ILU(k). Two additions, both nearly free:
- **Jacobi/diagonal** `(diag(A) − λI)⁻¹` — one kernel, works for dense and CSR.
- **Chebyshev "preconditioner"** — apply a low-degree `p(A)` to `R`. Pure GEMM/SpMM,
  no factorization, and it is the natural bridge to the Tier 3 filtered solver.

### 7.11 The projected `syev` is worth benchmarking as-is

The `3k × 3k` projected solve runs every iteration for every batch member. At
`k = 64` that is a batched `192 × 192` dense ED per iteration. The
`BATCHLAS_SYEVX_PROJECTED_VENDOR` knob (line 150) already exists to A/B this — it
should be swept, and `syev_jacobi_cta` from the Jacobi plan is a candidate provider
for the small-`k` end.

---

## 8. Implementation plan

### Tier 0 — Dispatch and honesty (small, do first)

1. Add `SyevxAlgorithm { Auto, Direct, DirectSubset, Filtered, LOBPCG }` to
   `SyevxParams`, mirroring the `Provider` pattern already used by `syev`
   (`include/blas/functions/syev.hh`) and the `BATCHLAS_*` env override convention.
2. `Auto` for `MatrixFormat::Dense`:
   - `n ≤ 64` → `Direct` (full `syev` + select).
   - `k/n > ~0.25` → `Direct`.
   - `0.02 < k/n ≤ 0.25` → `DirectSubset` (Tier 2; falls back to `Direct` until
     Tier 2 lands).
   - `k/n ≤ 0.02` → `Filtered` (Tier 3; falls back to `LOBPCG` until Tier 3 lands).
   - `MatrixFormat::CSR` → `LOBPCG`.
3. Add a `Direct` implementation: `syev` + a selection/sort kernel (`internal/sort.hh`
   already exists). This is ~50 lines and immediately guarantees `syevx` is never
   worse than `syev` on dense.
4. Extend `benchmarks/syevx_benchmark.cc` to sweep `(n, k/n, batch)` and compare all
   available algorithms. **The crossover thresholds above are derived from flop
   counts and must be replaced by measured ones.**

Expected outcome: large speedups on dense at moderate `k/n`, purely from routing.

### Tier 1 — Batched tridiagonal subset solver

New files, following the `*_cta` conventions of `steqr_cta.cc` / `sytrd_cta.cc`:

- **`stebz_cta`** — bisection. Gershgorin bounds for the initial interval; count
  eigenvalues below a point via the Sturm sequence `q_i = d_i − λ − e_{i−1}²/q_{i−1}`
  with the standard guard against `q_i = 0`; one lane (or sub-group for large `n`)
  per wanted eigenvalue; LAPACK `dlaebz`-style interval bookkeeping. Support
  `IndexRange` (`il..iu`) and `ValueRange` (`vl..vu`) selection.
- **`stein_cta`** — inverse iteration. Random start, 2–5 solves of `(T − λI)x = b`
  by tridiagonal LU with partial pivoting, reorthogonalization within clusters
  separated by less than `10⁻³·‖T‖`.

Also wire `stebz_cta` in as an eigenvalues-only alternative to `stedc` for small
`n` (§3.3) — measure against the existing STEDC path.

### Tier 2 — `syevx_direct` for medium dense

Compose: `sytrd_sy2sb` → `sytrd_sb2st` → `stebz_cta` → (`stein_cta` if vectors) →
narrowed Q₂ apply → narrowed `ormqr_blocked`. Model this on `syev_two_stage.cc`,
which already sequences four of these five stages; the changes are substituting the
tridiagonal solver and passing a `k`-column matrix to the back-transforms.

Target: ~3× over full ED at `k/n = 0.05`, more in wall-clock (§2.2).

### Tier 3 — Filtered subspace iteration

- Spectral bound estimate: reuse `src/extensions/lanczos.cc` for a few steps, or
  Gershgorin as a cheap conservative fallback.
- Degree selection from the target accuracy and the estimated gap, per ChASE;
  **uniform across the batch** to preserve GEMM shapes.
- Filter via `gemm`/`symm` (dense) or `spmm` (CSR); Rayleigh–Ritz reusing the
  existing projected-`syev` machinery; lock converged columns using the same
  staircase strategy as §7.5.

This subsumes the useful part of LOBPCG for unpreconditioned problems and is the
right default for `k/n ≲ 2 %`.

### Tier 4 — LOBPCG hardening

Apply §7 in order: 7.1 (async convergence), 7.6 (`extra_directions` default), 7.3
(residual kernel), 7.4 (criterion), 7.7 (ILU copies), 7.8 (reversal kernels), 7.5
(locking), 7.10 (Jacobi preconditioner), 7.2, 7.9.

7.1, 7.6, 7.7 and 7.8 are small and independently landable; do them first and
measure each.

### Tier 5 — CTA-fused filtered solver for small `n`

Only after Tiers 1–3 are measured. Gate on an estimated spectral gap; fall back to
`Direct` when the gap is small (§3.2).

---

## 9. What not to build

- **MRRR.** Its `O(n²)`-without-reorthogonalization advantage over bisection +
  inverse iteration is `O(nk²)` in our regime, which is negligible against `O(n³)`
  tridiagonalization for medium `n` and irrelevant for small `n`. The
  implementation cost (differential qd, RRR trees, shift selection, the accumulated
  errata of two decades) is very large. Reference LAPACK's own `dsyevr` drops to
  `dstebz` + `dstein` on subset paths.
- **FEAST / contour integration / shift-and-invert.** Require linear solves with
  `(A − zI)`; strictly more expensive than direct for batched dense.
- **Spectrum slicing across subintervals.** Parallelism already lives in the batch
  dimension; slicing adds cross-slice orthogonality bookkeeping for no gain.
- **A small-matrix (`n ≤ 64`) direct subset kernel.** §3.1 — it will not beat
  `syev_cta`.

---

## 10. Validation and benchmarking plan

Accuracy, per tier, following the pattern in `benchmarks/syevx_acc.cc` and
`tests/syevx_tests.cc`:

- Residual `‖Axᵢ − λᵢxᵢ‖ / (‖A‖‖xᵢ‖)` for every returned pair.
- Orthogonality `‖VᵀV − I‖` — the binding constraint for inverse iteration on
  clusters, and the reason Tier 1's cluster handling is the main risk.
- Eigenvalue agreement against LAPACK `dsyevx`/`dstebz` on the same input,
  including deliberately clustered spectra (repeated eigenvalues, Wilkinson
  matrices, graded matrices) and `λ ≈ 0` cases that expose §7.4.

Performance sweeps:

- `n ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}` × `k/n ∈ {0.01, 0.02, 0.05, 0.1,
  0.25, 0.5}` × `batch ∈ {1, 8, 64, 512}`, both `JobType` values.
- Baselines: `syev` (all providers), `syevx` (all algorithms), and the vendor
  non-batched subset routine looped over the batch.
- The deliverable of Tier 0 is the **measured** crossover map that replaces the
  flop-count estimates in §2.4.

Note the caveat in memory for this machine: double-precision CPU numerical failures
here are usually the broken OpenBLAS Cooperlake `dgemm` kernel, not BatchLAS
(`broken-openblas-dgemm-cooperlake`). Validate fp64 reference results against a
known-good BLAS before believing an accuracy regression.

---

## 11. Risks, ranked

1. **Inverse iteration on clusters (Tier 1).** Batched, uniform-shape
   reorthogonalization within variable-size clusters is the hardest part of the
   plan. Mitigation: cap cluster size, fall back to a per-matrix path (or to
   `stedc`) when exceeded, and make the orthogonality test a hard gate in CI.
2. **The 3× ceiling is not enough to justify the work.** Mitigation: Tier 0 is
   cheap and delivers most of the practical win by routing alone; measure before
   committing to Tier 2.
3. **Filter degree selection instability (Tier 3).** A bad spectral bound produces
   either no acceleration or overflow. Mitigation: conservative Gershgorin fallback,
   clamp the degree, and detect divergence by monitoring the Ritz values.
4. **Batched adaptivity vs uniform shapes.** Every adaptive idea in the literature
   (per-eigenpair degree, per-matrix locking, variable block width) assumes a single
   matrix. Mitigation: max-over-batch or bucketing, decided per feature; do not
   import adaptivity uncritically.
5. **Two-stage back-transform narrowing may not be a clean shape change.** The Q₂
   application in `sytrd_sb2st` may have assumptions tied to a square vector matrix.
   Mitigation: verify early in Tier 2; this is the one place the "parts already
   exist" claim could fail.

---

## 12. Sources

Bisection, inverse iteration, MRRR:
- [ALGLIB — symmetric bisection and inverse iteration](https://www.alglib.net/eigen/symmetric/symmbisectionandinverseiteration.php)
- [ALGLIB — tridiagonal EVD: bisection and inverse iteration](https://www.alglib.net/eigen/symmetric/tdbisectionandinverseiteration.php)
- [Dhillon, Parlett & Vömel — The design and implementation of the MRRR algorithm (TOMS)](https://dl.acm.org/doi/10.1145/1186785.1186788)
- [Petschow & Bientinesi — MRRR-based eigensolvers for multi-core processors and supercomputers](https://arxiv.org/pdf/1401.4950)
- [Petschow et al. — High-performance solvers for dense Hermitian eigenproblems](https://arxiv.org/pdf/1205.2107)
- [A modified bisection algorithm for eigenvalues of a symmetric tridiagonal matrix (Numer. Math.)](https://link.springer.com/article/10.1007/BF01396441)
- [Dongarra et al. — Numerical eigen-spectrum slicing, accurate orthogonality (IJHPCA 2024)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/eigen-improve-ijhpca-2024.pdf)

Polynomial filtering:
- [ChASE — Chebyshev Accelerated Subspace iteration Eigensolver](https://arxiv.org/pdf/1805.10121)
- [ChASE — distributed hybrid CPU-GPU eigensolver](http://fulir.irb.hr/7504/1/3539781.3539792.pdf)
- [Advancing the distributed multi-GPU ChASE library (SC'23 Workshops)](https://dl.acm.org/doi/10.1145/3624062.3624249)
- [Residual-based Chebyshev filtered subspace iteration tolerant to inexact matvecs](https://arxiv.org/pdf/2503.22652)
- [AdaPolySI — adaptive polynomial filtered subspace iteration (ACM ICS)](https://doi.org/10.1145/3797905.3800553)
- [Zhou & Saad — A Chebyshev–Davidson algorithm for large symmetric eigenproblems (SIMAX)](https://dx.doi.org/10.1137/050630404)
- [Zhou — A block Chebyshev–Davidson method with inner–outer restart](https://www.sciencedirect.com/science/article/abs/pii/S0021999110004791)
- [Saad et al. — Parallel self-consistent-field calculations via Chebyshev-filtered subspace acceleration](https://arxiv.org/pdf/cond-mat/0703239)

LOBPCG:
- [Knyazev — Toward the optimal preconditioned eigensolver: LOBPCG (SISC)](https://epubs.siam.org/doi/10.1137/S1064827500366124)
- [Duersch, Shao, Yang & Gu — A robust and efficient implementation of LOBPCG](https://arxiv.org/pdf/1704.07458)
- [Anzt, Tomov & Dongarra — Accelerating the LOBPCG method on GPUs using a blocked SpMV](https://icl.utk.edu/files/publications/2014/icl-utk-771-2014.pdf)
- [Efficient implementation of the LOBPCG algorithm on a CPU-GPU cluster](https://link.springer.com/chapter/10.1007/978-981-96-2830-8_6)
- [Local convergence behavior of extended LOBPCG](https://arxiv.org/pdf/2505.08218)

Dense eigensolver performance on GPUs:
- [Extracting the potential of emerging hardware accelerators for symmetric eigenvalue decomposition](https://arxiv.org/html/2410.02170v1)
- [cuSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
- [pytorch/pytorch#175585 — `linalg.eigh` performance cliff at n=32 for batched CUDA inputs](https://github.com/pytorch/pytorch/issues/175585)
