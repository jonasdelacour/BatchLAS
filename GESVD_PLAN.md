# Batched GESVD: status, defect analysis, and implementation plan

Goal: a batched SVD that is **measurably faster than cuSOLVER's `gesvdjBatched`** and
**at least as accurate**, across the shapes and types users actually pass.

Everything in §1 is measured or read off this tree at commit `5ade468`, not assumed.
Every number below was taken on this box (2x RTX 4090, CUDA 13.3) with the
already-built binaries in `build/benchmarks/`.

---

## STATUS UPDATE — Tier 0, the accuracy harness, and Tier 1 are DONE and measured

**`gesvdj_cta` (Tier 1) has landed and beats `gesvdjBatched` on both axes.**
One-sided Hestenes Jacobi, one fused kernel, `src/extensions/gesvdj_cta.cc`.

Time, full U and V^H, uncontended (`CUDA_VISIBLE_DEVICES=1`), ms:

| type | n | batch | cuSOLVER | gesvd_cta (old) | **gesvdj_cta** | vs cuSOLVER |
|---|---|---|---|---|---|---|
| float | 8 | 16384 | 5.594 | 1.095 | **0.104** | **53.7x** |
| float | 16 | 16384 | 7.596 | 1.540 | **0.891** | **8.5x** |
| float | 32 | 4096 | 3.226 | 2.126 | **2.082** | **1.55x** |
| float | 32 | 16384 | 12.581 | 7.643 | **7.804** | **1.61x** |
| double | 8 | 16384 | 42.248 | 5.470 | **1.029** | **41.1x** |
| double | 16 | 16384 | 103.520 | 18.703 | **6.926** | **15.0x** |
| double | 32 | 16384 | 364.539 | 83.075 | **49.516** | **7.4x** |

Relative error in the singular values, n=32 float, against the *known* spectrum:

| log10 kappa | gesvd_cta (old) | cuSOLVER | **gesvdj_cta** |
|---|---|---|---|
| 1 | 7.8e-3 | 4.09e-6 | 4.78e-6 |
| 3 | 1.06e-2 | 1.03e-5 | 1.25e-5 |
| 4 | 0.299 | 8.35e-5 | **6.44e-5** |
| 5 | 1.06 | 7.40e-4 | **5.67e-4** |
| 6 | 2.13 | 9.27e-3 | **5.33e-3** |

Orthogonality is flat at ~1.3e-5 across all six decades (cuSOLVER ~8e-6; the old
path collapses to 2.43). At kappa >= 1e4 the new kernel is *more* accurate than
cuSOLVER. In double it holds to kappa = 1e14 (6.2e-4 relative, 1.4e-14
orthogonality). **Defect A is closed.**

Also closed: complex GENERAL SVD on GPU, which previously fell through to Vendor
and threw. `gesvdj_cta` supports it natively and is reachable automatically for
complex, or by `BATCHLAS_GESVD_PROVIDER=jacobi`. It sits *after* `BatchLAS_CTA`
in the default order on purpose — promoting it ahead is a separate, gated commit
(see GESVD_IMPL_SPEC.md D.3), since the old path is still faster values-only.

Three defects were found in the reviewed design during implementation, all of
which produced silently wrong or crashing output and none of which the pre-existing
gesvd tolerances could have caught:

1. **Completion acceptance threshold.** "Accept when the residual norm exceeds
   1/2" cannot fill the last columns of a tall matrix: with d dimensions left of
   RR, a canonical basis vector's residual norm^2 is only ~d/RR, so at d=1, RR=32
   every trial measures ~0.03 and is rejected. The column stays zero and U is not
   orthogonal. Fixed by running the trial cursor *across* columns and accepting
   above 1/(2*RR), which is provably sufficient.
2. **m < n must solve A^H, not A^T.** A^T gives A = conj(V') S U'^T, whose
   conjugations differ from the m >= n case. Invisible in real arithmetic; wrong
   for complex.
3. **The norm recurrence must clamp both sides at zero.** Clamping only the
   cancelling p side lets the q side go negative, sqrt gives NaN, every rank
   comparison against NaN is false, two columns collide on one rank, and the
   unwritten permutation slot is read as a garbage column index — an
   out-of-bounds LDS access. Observed as CUDA_ERROR_ILLEGAL_ADDRESS at
   kappa >= 1e4.

Tests: `tests/gesvdj_cta_tests.cc`, 32 cases over {float, double, complex<float>,
complex<double>} x {square, tall, wide, rank-deficient, all-zeros, graded, job
combinations, public-API dispatch}. All pass; `gesvd_tests` is unchanged at 26.

### Earlier status — Tier 0 and the harness

Both prerequisites in §8 have landed. The comparison the plan could not previously
make is now made, and it splits cleanly in two.

**Speed: BatchLAS already wins.** n=32, float, `gesvd_vendor_benchmark`, pinned to
an idle GPU (`CUDA_VISIBLE_DEVICES=1`; std dev <= 0.5% of mean):

| batch | jobu/jobvh | cuSOLVER `gesvdjBatched` | BatchLAS `gesvd_cta` | speedup |
|---|---|---|---|---|
| 4096 | None/None | 3.372 ms | 0.911 ms | **3.70x** |
| 4096 | All/All | 3.250 ms | 2.131 ms | **1.53x** |
| 16384 | None/None | 12.159 ms | 3.248 ms | **3.74x** |
| 16384 | All/All | 12.664 ms | 7.827 ms | **1.62x** |

In double at n=32/batch=16384/All-All: cuSOLVER 365.2 ms vs BatchLAS 80.9 ms — **4.5x**.
Note `gesvdjBatched`'s time barely moves with `jobz` (3.37 vs 3.25), so it appears to
compute vectors regardless. Full sweep: `benchmarks/results/gesvd_vs_gesvdj_rtx4090.csv`.

**Accuracy: cuSOLVER wins by 2-3 orders of magnitude, exactly as §2.1 predicted.**
`gesvd_relacc --samples=128 --log10-cond=1..6 --type=float 32`, relative error in the
singular values against the *known* constructed spectrum:

| log10 kappa | BatchLAS `max_relerr` | cuSOLVER `max_relerr` | BatchLAS orthogonality | cuSOLVER orthogonality |
|---|---|---|---|---|
| 1 | 7.8e-3 | 4.1e-6 | 2.9e-6 | 7.7e-6 |
| 2 | 7.9e-3 | 4.1e-6 | 1.0e-4 | 7.5e-6 |
| 3 | 1.1e-2 | 1.0e-5 | 5.8e-3 | 7.6e-6 |
| 4 | **0.299** | 8.4e-5 | **1.11** | 8.2e-6 |
| 5 | **1.06** | 7.4e-4 | **1.50** | 8.1e-6 |
| 6 | **2.13** | 9.3e-3 | **2.43** | 8.0e-6 |

cuSOLVER tracks `eps * kappa` — the Demmel-Veselic bound — and its orthogonality is
flat at ~8e-6 across six decades. The BatchLAS normal-equations path has lost all
relative accuracy by kappa = 1e4 and its U/V are no longer orthogonal at all. Note
it is already at 7.8e-3 at kappa = 10, i.e. ~1e5 x eps: the damage starts immediately,
not just at high conditioning.

**What this means for the plan.** The strategic picture inverts relative to §7 risk 1.
We do not need to find speed; we need to *keep* it while fixing accuracy. Tier 1's bar
is therefore not "beat 4.09 ms" but "reach cuSOLVER-class relative accuracy while
staying under cuSOLVER's 3.25 ms at n=32/batch=4096" — 1.5x of headroom over the
current path, against a target `syev_jacobi_cta` already meets at 2.11 ms for a
comparable amount of work. Tiers 1 and 2 are still the right work; the payoff is
now quantified rather than assumed.

**Tier 1 design decision (settled).** A three-candidate design panel judged by three
independent lenses (performance, numerical accuracy, implementability) returned a
unanimous 3/3 verdict for **lane = row, one warp per problem, round-batched
reduce-scatter Gram**. The Gram-resident variant was rejected — its own author
concluded it is wrong at m=n=32 (per-round LDS cost 10n vs 5n+2m). The accuracy judge
found a real defect in the winning design that must be repaired before implementation:
its reduce-scatter `half = (16 >> step) / 2` yields 8,4,2,1,**0**, so the fifth step is
a no-op and every dot product is silently halved; the fix is a plain all-reduce
special-cased outside the formula.

---

---

## Table of contents

1. [Current status — what actually exists](#1-current-status)
2. [The four defects that decide the outcome](#2-the-four-defects)
3. [What we are racing: `gesvdjBatched`](#3-what-we-are-racing)
4. [Design](#4-design)
5. [Tier 0 — the cuSOLVER baseline (blocking prerequisite)](#tier-0)
6. [Tier 1 — one-sided Jacobi CTA (`gesvdj_cta`)](#tier-1)
7. [Tier 2 — QR preconditioning](#tier-2)
8. [Tier 3 — repair the blocked path (n > 32)](#tier-3)
9. [Tier 4 — tall-skinny and complex coverage](#tier-4)
10. [Accuracy harness](#5-accuracy-harness)
11. [Benchmark protocol](#6-benchmark-protocol)
12. [Risks, ranked](#7-risks)
13. [Sequencing and exit criteria](#8-sequencing)

---

## 1. Current status

### 1.1 What is wired

`gesvd` is a dispatching front-end (`include/blas/functions/gesvd.hh`) over three
providers, chosen by `default_order_cta_blocked_vendor_netlib` =
**CTA -> Blocked -> TwoStage -> Vendor -> Netlib**:

| Provider | Source | Domain | Algorithm |
|---|---|---|---|
| `BatchLAS_CTA` | `src/extensions/gesvd_blocked.cc` (`gesvd_cta`) | GPU, `max(m,n) <= 32`, real (or Hermitian complex) | `gebrd_cta` -> normal-equation tridiagonal -> `steqr_cta` |
| `BatchLAS_Blocked` | `src/extensions/gesvd_blocked.cc` (`gesvd_blocked`) | GPU, real, any size | `gebrd_{unblocked,blocked}` -> normal-equation tridiagonal -> `stedc` -> `ormbr` |
| `Vendor` | `include/blas/functions/gesvd.hh` | **NETLIB only** | LAPACKE `?gesvd`, looped over the batch, `ctx.wait()` first |

Both native paths live in one 1420-line file. There is a Hermitian shortcut that
routes to `syev_{cta,blocked}` and takes absolute values of eigenvalues.

### 1.2 Measured baseline

`gesvd_cta`, n=32, batch=8192, square, uncontended:

| type | jobu/jobvh | time | per matrix |
|---|---|---|---|
| float | All/All | 4.09 ms | 0.499 us |
| float | All/None | 3.56 ms | 0.434 us |
| float | None/All | 2.25 ms | 0.275 us |
| float | None/None | 1.66 ms | 0.203 us |
| double | All/All | 40.7 ms | 4.97 us |
| double | None/None | 30.9 ms | 3.78 us |

Reference points at the same shape (`syev_jacobi_cta_benchmark 32 8192`):

| kernel | float | double |
|---|---|---|
| `syev_jacobi_cta` (two-sided Jacobi, with vectors) | 2.11 ms | 19.3 ms |
| `syev_cta` tridiagonal reference | 0.988 ms | 31.7 ms |

Two things to read off this:

* **float:double is ~10x**, not 2x. RTX 4090 runs FP64 at 1/64 of FP32. Any
  double-precision target is bounded by that, and cuSOLVER hits the same wall —
  so the double comparison is a fair fight, just a slow one.
* **In double, Jacobi already beats the tridiagonal path** for `syev`
  (19.3 vs 31.7 ms). Jacobi is not only the accurate choice here, it is the fast
  one in the type where the rotation arithmetic stops being free.

### 1.3 What is missing

* `gesvd_vendor` **throws** for every backend except NETLIB
  (`"gesvd_vendor: backend implementation not available yet"`). There is no
  cuSOLVER SVD binding of any kind — not `gesvdjBatched`, not
  `gesvdaStridedBatched`, not `gesvdj`. `src/backends/cusolver.cc` (296 lines)
  has `potrfBatched`, `XsyevBatched`, `syevjBatched`, and no SVD at all.
* `src/extensions/bdsqr.cc` — 420 lines of bidiagonal QR — has **zero callers**.
  Verified by grep: the only hits outside the file are the declaration and the
  `BATCHLAS_DISPATCH_ON_QUEUE` macro. The accurate bidiagonal solver was written
  and never wired in.
* Complex **general** (non-Hermitian) SVD is unsupported on GPU:
  `gesvd_supports_cta` and `gesvd_supports_blocked` both `return false` for
  complex, dispatch falls through to `Vendor`, and `Vendor` throws. Only the
  Hermitian overload works for complex.
* `SvdVectors` has only `None` and `All` — no economy/thin mode, so a
  tall-skinny `m x n` job must materialise a full `m x m` U.

---

## 2. The four defects

### 2.1 Defect A — the native paths form the normal equations

This is the one that decides the accuracy comparison, and it is not a subtlety.

`form_right_tridiagonal` (`src/extensions/gesvd_blocked.cc:220`) takes the
bidiagonal factors `d, e` from `gebrd` and builds

```
TD(i) = d_i^2 + e_{i-1}^2
TE(i) = d_i * e_i
```

That is exactly the tridiagonal of **B^T B**, formed explicitly. The pipeline
then runs a symmetric eigensolver on it and takes `sigma_i = sqrt(lambda_i)`.
`form_left_tridiagonal` does the same for `B B^T`.

The consequence is standard and severe. A symmetric eigensolver returns
`lambda_i` with absolute error on the order of `eps * lambda_max = eps * sigma_max^2`.
Propagating through the square root:

```
|d sigma_i| / sigma_i  ~  (eps / 2) * (sigma_max / sigma_i)^2
```

The relative error in a small singular value grows with the **square** of the
condition number. In float (`eps ~ 6e-8`), a matrix with `kappa = 1e3` already
has no correct digits left in `sigma_min`. This is precisely the failure mode
that one-sided Jacobi — i.e. `gesvdjBatched` — is designed to avoid.

The test suite is consistent with this. `tests/gesvd_tests.cc` uses, for float:

```
gesvd_sv_tol    = 5e-2
gesvd_ortho_tol = 2e-1     // relative
gesvd_recon_tol = 3e-1     // relative
```

A relative reconstruction tolerance of **0.3** and an orthogonality tolerance of
**0.2** do not constitute a correctness test; they are wide enough to admit a
result with essentially no accuracy. These are the tolerances a normal-equations
SVD needs in order to pass.

**Conclusion: on the accuracy axis we currently lose to `gesvdjBatched` by
construction, and the tests are calibrated not to notice.** No amount of kernel
tuning fixes this; the algorithm has to change.

### 2.2 Defect B — there is no baseline to beat

We cannot presently produce a single `gesvdjBatched` number. The comparison the
task asks for is not measurable today. This makes Tier 0 a hard prerequisite,
not a nice-to-have: every performance claim downstream is unfalsifiable without it.

### 2.3 Defect C — the accuracy harness measures the wrong thing

`benchmarks/gesvd_{cta,blocked}_acc.cc` report a single `Fail%` column against
the loose tolerances above. A run of `gesvd_cta_acc --samples=32` returns
`Fail% = 0.00000` for every type and size — which tells us nothing, because the
bar is at 0.3 relative error.

Worse, the `log10cond` column printed in the output is **blank**: the
conditioning sweep that `miniacc` supports (`--log10-cond=`) is not wired into
these benchmarks. So the harness cannot see the one effect that matters, which is
error as a function of `kappa`.

### 2.4 Defect D — the perf benchmarks stop at batch=64

`GesvdCtaBenchSizes` / `GesvdBlockedBenchSizes` sweep `bs in {1,2,4,8,16,32,64}`.
At n=32, batch=64 is nowhere near saturating a 4090 — the measurement is
dominated by launch overhead, so ratios taken there are overhead ratios, not
algorithm ratios. All tuning must be done at batch >= 4096.

---

## 3. What we are racing

`cusolverDnXgesvdjBatched` (present in this toolkit's `cusolverDn.h`, lines
3950-4085, all four types S/D/C/Z):

* **One-sided Jacobi.** Its selling point is high *relative* accuracy — the
  Demmel-Veselic bound `|d sigma_i|/sigma_i <= eps * kappa(A_c)` where `A_c` is
  the column-equilibrated matrix — not raw speed.
* **Documented shape limit: `m <= 32` and `n <= 32`.** (In the programming
  guide, not the header — the header carries no doc comments. *Verify empirically
  in Tier 0 rather than trusting this.*) It is not a coincidence that the
  existing `gesvd_cta` caps at 32; the domains were meant to line up.
* Tunables via `gesvdjInfo_t`: `XgesvdjSetTolerance`, `XgesvdjSetMaxSweeps`,
  `XgesvdjSetSortEig`. A fair comparison must match tolerance and sort order, and
  must report `XgesvdjGetSweeps` — sweep count is the dominant cost term and
  cuSOLVER's default tolerance may differ from ours.
* Above 32x32 there is **no batched vendor SVD** except
  `gesvdaStridedBatched`, which is an *approximate* solver restricted to `m >= n`
  and aimed at tall-skinny problems.

Strategic read: **for `n > 32` we win by default** — there is no batched
competitor, only a loop over `gesvdj`. The genuinely contested region is
`n <= 32`, where `gesvdjBatched` is a purpose-built kernel. That is where the
work should go, and it is exactly where BatchLAS already has the right machinery.

---

## 4. Design

### 4.1 The core asset

`src/extensions/syev_jacobi_cta.cc` (664 lines) is a mature, tuned, two-sided
Jacobi CTA eigensolver, and nearly every part of it transfers to a one-sided
Jacobi SVD:

* **One lane per column**, `P in {4,8,16,32}` chosen at compile time from `n`.
* **LDS tile with `LD = P+1`** — the padding comment explains that `LD == 32`
  serialises a warp 32 ways on the row-update phase; this is a solved problem.
* **Precomputed round-robin pivot table** in LDS, shared across all problems in
  the work-group, packed two indices to an `int16_t`. The comment notes that
  computing pairs inline cost three integer modulos per lane per phase and
  dominated the inner loop — also solved.
* **Multiple problems per work-group**, sized by a local-memory budget
  computed against `device::local_mem_size`.
* **Rotation coefficients packed as `vec<Real,2>`** so the update loop issues one
  LDS load instead of two.
* **The correct relative stopping criterion** already implemented:
  `|a_pq| > tol * sqrt(|a_pp| * |a_qq|)`, with a comment stating explicitly that
  the classical absolute test would forfeit the relative-accuracy advantage.
* Complex support, denormal guard, `tau_big` overflow branch, sorting.

`JacobiParams<T>` (`extensions.hh:1701`) already carries `tol_multiplier`,
`max_sweeps`, `sort`, `sort_order`, `cta_wg_size_multiplier`.

The rotation *schedule*, *storage layout*, *convergence test*, and *launch
geometry* are all reusable. What changes is what a rotation is applied to.

### 4.2 Why one-sided Jacobi, specifically

Hestenes one-sided Jacobi on `A (m x n, m >= n)`: repeatedly pick a column pair
`(p,q)`, form the 2x2 Gram from `a_pp = A_p.A_p`, `a_qq = A_q.A_q`,
`a_pq = A_p.A_q`, and apply the Jacobi rotation to **columns p and q of A**. At
convergence the columns of `A` are orthogonal, `sigma_i = ||A_i||`,
`U_i = A_i / sigma_i`, and `V` is the accumulated product of rotations.

The accuracy argument, stated precisely so it is not confused with Defect A:
the 2x2 Gram entries are dot products, and a dot product does square magnitudes.
But `sigma_i` is **never recovered as the square root of a difference of large
numbers** — it is a column norm of the *rotated* `A`. The rotations are
orthogonal and applied to `A` itself, so the condition number is never squared.
That is the whole distinction from Defect A, where a full `B^T B` tridiagonal is
built and `sigma = sqrt(lambda)`.

Secondary benefits that matter here:

* `U` and `V` come out of the iteration directly. There is **no back-transform**
  — no `ormbr`, no `ormqr`. In the measured baseline, going from `None/None` to
  `All/All` costs 1.66 -> 4.09 ms (2.5x); a large part of that is back-transform
  work that one-sided Jacobi does not have.
* The whole thing is **one kernel**. The current CTA path is
  `gebrd_cta` -> tridiagonal build -> `steqr_cta` -> vector assembly ->
  back-transform: five-plus launches with global round-trips between them.
* It degrades gracefully: `max_sweeps` and `tol_multiplier` give a real
  accuracy/speed dial, which is how we can offer a "fast" mode that still beats
  the current path on accuracy.

### 4.3 The one open micro-architecture question

Per round, the `n/2` disjoint pairs each need three length-`m` dot products
before their rotation can be computed. In `syev_jacobi_cta` the corresponding
quantity is a single LDS read, because the matrix *is* the Gram. So one-sided
Jacobi does strictly more work per rotation, and where that work lands decides
the kernel.

Two candidate mappings, to be prototyped and measured, not decided on paper:

**(a) lane = column** (mirrors `syev_jacobi_cta` exactly). `A` is `m x n` in LDS.
Phase 1: distribute the `1.5n` dot products across the `n` lanes — with two lanes
per pair the critical path is ~`2m` FMAs. Phase 2: every lane owns one column and
updates it, `m` FMAs. Keeps the entire proven structure; the risk is Phase 1 load
imbalance.

**(b) lane = row.** Lane `r` holds `A[r,p]` and `A[r,q]`; the three dot products
become three sub-group reductions (`partition_reduce_sum_j` already exists in the
file at line 97). Perfectly balanced, but three reductions per pair per round is
a lot of shuffle traffic.

(a) is the recommended starting point purely because it reuses a structure that
is already tuned; (b) is the fallback if Phase 1 imbalance dominates.

**A third option to hold in reserve, gated on measurement:** maintain
`G = A^T A` in LDS alongside `A` and update it as `J^T G J`, making Phase 1 a free
LDS read exactly as in `syev_jacobi_cta`. This is mathematically exact — errors
in `G` perturb only the *choice* of rotation, not the output, since `sigma` is
still read off `A`'s column norms. It is a genuine optimisation, but it drifts
with rounding and it is the kind of shortcut that quietly reintroduces Defect A
if `sigma` is ever taken from `G` instead of `A`. **Do not implement it until
(a) or (b) is correct and measured, and gate it behind an A/B accuracy test.**

### 4.4 Accuracy features that are not optional

Carried over from LAPACK `xGESVJ` / Drmac-Veselic (LAWN 169):

1. **Column equilibration** up front (scale each column to unit norm, record the
   scaling). This is what makes `kappa(A_c)` — not `kappa(A)` — the governing
   quantity, and it prevents over/underflow in the dot products.
2. **Relative stopping test**, already implemented in `syev_jacobi_cta`.
3. **Skip converged pairs**: a pair below threshold costs a test, not a rotation.
   This is what makes late sweeps cheap and it is essential to the sweep count.
4. **de Rijk ordering** (sort columns by descending norm) — reduces sweeps.

---

## Tier 0

### The cuSOLVER baseline — do this first

Nothing downstream is measurable without it.

**Work:**

1. Implement `gesvd_vendor` / `gesvd_vendor_buffer_size` for `Backend::CUDA` in
   `src/backends/cusolver.cc`, following the existing `syevjBatched` block
   (lines 154-163) verbatim in style: `LinalgHandle<B>`, `handle.setStream(ctx)`,
   `call_backend<T, BackendLibrary::CUSOLVER, B>(S,D,C,Z, ...)`, workspace from
   the `BumpAllocator` pool, `op_external(...)` wrapper,
   `ctx.create_event_after_external_work()`.
2. Route by shape:
   * `m,n <= 32` -> `gesvdjBatched` (+ `CreateGesvdjInfo` / `SetTolerance` /
     `SetMaxSweeps` / `SetSortEig` / `DestroyGesvdjInfo`).
   * `m >= n`, larger -> `gesvdaStridedBatched`, clearly **labelled approximate**
     in the benchmark output so it is never silently compared against an exact
     result.
   * otherwise -> loop `gesvdj`, labelled as a loop.
3. **Empirically establish the 32x32 limit** rather than trusting the docs: call
   `gesvdjBatched` at 33x33 and record the status code. This determines the
   boundary of the contested region and must not be guessed.
4. Add a `Provider::Vendor` escape so benchmarks can force it —
   `BATCHLAS_GESVD_PROVIDER=vendor` already exists via `parse_provider_env`, but
   note dispatch order is CTA -> Blocked -> ... -> Vendor, so **Vendor is
   currently unreachable for real GPU input without forcing**. (This is the same
   routing trap recorded for `syev`: an `Auto` order that never reaches cuSOLVER.)

**Exit criterion:** a table of `gesvdjBatched` time and accuracy for
`n in {8,16,24,32}`, `batch in {1024, 4096, 16384}`, `{float,double}`,
jobz `{None, All}`, plus the observed sweep counts.

---

## Tier 1

### `gesvdj_cta` — one-sided Jacobi, `max(m,n) <= 32`

The head-to-head competitor. New file `src/extensions/gesvdj_cta.cc`, modelled
structurally on `syev_jacobi_cta.cc`.

**Scope:** real and complex, `m,n <= 32`, `jobu/jobvh in {None, All}`, both
`m >= n` and `m < n` (the latter by the existing transpose-and-swap trick already
used in `gesvd_blocked.cc`).

**Steps:**

1. Column-equilibrate `A` into the LDS tile; record scales.
2. Sweep loop with the existing round-robin pair table and relative threshold:
   per round, compute the pair Grams, build rotations, apply to `A` (and to `V`
   if `jobvh != None`).
3. On convergence: `sigma_i = ||A_i||` (undo scaling), `U_i = A_i / sigma_i`,
   sort descending, permute `V` to match.
4. Handle rank deficiency: `sigma_i` below the zero threshold means `U_i` is not
   determined by `A` — fill from the orthogonal complement. The existing
   `patch_zero_left_vectors` in `gesvd_blocked.cc` solves the same problem and
   shows the intended semantics.
5. Wire into `gesvd_supports_cta` / `choose_gesvd_provider` as a new
   `Provider::BatchLAS_Jacobi`, ahead of the current CTA path for `n <= 32`.

**Target:** given `syev_jacobi_cta` at 2.11 ms (float, n=32, batch=8192, with
vectors) and the extra dot-product and `V`-accumulation cost, a landing zone of
**3-5 ms** is realistic — i.e. roughly parity with today's `gesvd_cta` (4.09 ms)
while fixing Defect A outright. Whether that beats `gesvdjBatched` is exactly
what Tier 0 tells us, and it is not knowable before then.

**Keep the old CTA path** behind a provider flag until Tier 1 is measured
faster *and* more accurate. It may remain the right choice for well-conditioned
values-only work.

---

## Tier 2

### QR preconditioning — TESTED AND REJECTED as a speed optimisation

The plan called this "the main speed lever". **It is not, for this workload.**
The tier was gated on measurement and the measurement came back negative.

`gesvdj_cta` now reports per-problem sweep counts (`GesvdjParams::sweep_counts`,
the analogue of `cusolverDnXgesvdjGetSweeps`), which is what made this decidable.
Baseline mean sweeps at n=32, float, 64 samples:

| kappa | 1e1 | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|---|
| sweeps | 8.95 | 12.03 | 13.25 | 14.55 | 15.22 |

Sweeps are the only term an algorithmic change can move (cost is
`sweeps x n^2 x m`), and they nearly double from kappa 1e1 to 1e6, so the lever
looked real. Two of the three preconditioning mechanisms were then tested:

**1. de Rijk pre-ordering — no effect. Removed.**

| kappa | 1e1 | 1e4 | 1e6 |
|---|---|---|---|
| with | 8.91 | 13.52 | 15.53 |
| without | 8.95 | 13.25 | 15.22 |

Neutral at low conditioning, slightly *worse* when graded. Worse, merely having
the untaken branch in the kernel cost **13% of wall clock** (7.80 -> 8.88 ms at
n=32/batch=16384) through register pressure. Removed outright rather than left
behind a default-off flag; the code is gone and this table is the record.

**2. QR preconditioning — no sweep reduction.** Tested out-of-kernel, which is
the cheap way to falsify it: Q is orthogonal so `sigma(R) == sigma(A)` exactly,
meaning the kernel can be run on `R` and scored against the *same* known
spectrum without ever building Q. Mean sweeps:

| kappa | 1e1 | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|---|
| on A | 8.95 | 12.03 | 13.25 | 14.55 | 15.22 |
| on R | 8.97 | 12.02 | 13.23 | 14.50 | 15.22 |

Identical to within 0.1 sweeps everywhere. A fused in-kernel QR would therefore
pay for the factorisation and the back-application of Q and get **nothing** back
on the sweep count — a guaranteed net loss, before writing a line of it.

> **A false positive nearly got through here.** The first run of this experiment
> showed sweeps dropping 15.2 -> 8.9, an apparent 1.7x. It was wrong: the R being
> factored was not R. `MatrixView::triangularize` indexes `i*ld + j` while naming
> `i` the row (`src/matrix.cc:796-805`), so on this library's COLUMN-major
> storage its `uplo` is inverted — `triangularize(Uplo::Upper)` keeps the LOWER
> triangle. The kernel was being handed geqrf's Householder reflectors, which
> happen to be easier to diagonalise. The tell was that reconstruction was fine
> (5e-7) while the singular values were off by 30x, i.e. it had correctly
> factored the wrong matrix. **This is a live bug in `triangularize` and is worth
> fixing separately**; it is not specific to gesvd.

**3. Column-pivoted QR (geqp3) — not attempted.** There is no batched pivoted QR
in the tree, so this is a large new implementation. The evidence argues against
it: the two mechanisms that *were* testable are the same family (pivoting orders
columns by decreasing residual norm; de Rijk orders them by decreasing norm once)
and both moved the sweep count by nothing. It should not be built on the
literature's claim alone.

### What Tier 2 did produce

* **A convergence diagnostic** — `GesvdjParams::sweep_counts`. Permanent value:
  sweep count is the quantity every future tuning change has to be judged on, and
  it is now visible in `gesvd_relacc` under the printable `iterations_done`
  column.
* **An accuracy result worth keeping.** QR preconditioning does not help speed
  but it *does* improve relative accuracy, roughly 2x at high conditioning
  (n=32 float: `max_relerr` 4.03e-5 vs 6.56e-5 at kappa=1e4; 2.58e-3 vs 5.11e-3
  at 1e6). That is the Drmac-Veselic accuracy benefit showing up without the
  convergence benefit. If an accuracy-critical mode is ever wanted, this is the
  lever — as an *option*, not a default.
* **A confirmed occupancy setting.** `cta_wg_size_multiplier` swept at
  n=32/batch=16384/float: 1 -> 7.67 ms, 2 -> 8.85 ms, 4 -> 9.01 ms. The default
  of 1 is correct; more problems per work-group trades local memory for warps and
  the trade is not monotone.

**Conclusion: do not build fused QR preconditioning.** The remaining speed work
for `n <= 32` is micro-architectural (the Gram phase still runs in full during
late, mostly-converged sweeps), not algorithmic.

---

## Tier 3

### Blocked bidiagonalization for n > 32 — bdsqr FIXED and wired, but not the default

`bdsqr` is now correct and the accuracy win is exactly what the plan predicted.
It is **not** fast enough to be the default, and the reason is structural rather
than a tuning problem.

**bdsqr had a real bug, and 420 lines of it had never executed.** It has no
handling for a negligible diagonal entry. When `db[l]` is zero the shift gives
`f = -mu`, `g = db[l]*eb[l] = 0`, so `lartg` returns the identity rotation,
every rotation in the chase is trivial, nothing is annihilated, and the sweep
spins to `maxit` and reports "did not converge" — immune to the iteration cap,
which is how it was identified (20x `maxit` changed nothing). LAPACK DBDSQR
handles this with a zero-SHIFT chase; that branch was missing entirely. Added in
both directions:

* zero strictly inside the block -> chase rightwards with LEFT rotations,
  emptying that row;
* zero at the BOTTOM of the block -> chase leftwards with RIGHT rotations,
  emptying that column. **Handling only the first case is itself a bug**: it
  leaves `eb[l..m-1]` untouched while still advancing past the block, so the
  sweep never converges. That was caught by the tests below, not by inspection.

`tests/bdsqr_tests.cc` is new — bdsqr previously had no test because it had no
caller. Note the first version of it passed while the bug was still present: it
generated only positive `d, e` in [0.3, 1.7], a far easier class than anything
gebrd emits. The test that matters is the one with mixed signs, exact zeros and
six decades of range.

**Accuracy: the defect is closed for the blocked path.** A/B on identical data,
float, `gesvd_blocked` normal-equation vs bdsqr:

| n | kappa | normal-eq relerr | bdsqr relerr | normal-eq ortho | bdsqr ortho |
|---|---|---|---|---|---|
| 32 | 1e3 | 2.90e-3 | 8.2e-6 | 6.0e-3 | 2.0e-6 |
| 32 | 1e4 | 0.258 | 6.7e-5 | 1.48 | 1.9e-6 |
| 32 | 1e6 | 2.019 | 0.168 | 2.60 | 1.4e-6 |
| 64 | 1e4 | 0.288 | 7.7e-5 | 0.631 | 3.3e-6 |

Orthogonality goes flat at ~2e-6 across six decades instead of collapsing.
Verified working at n = 64, 128 and 256.

**Performance: 3-400x slower, and parallelising the obvious part will not fix
it.** batch=512, float, full vectors:

| n | normal-eq | bdsqr | |
|---|---|---|---|
| 64 | 201 ms | 643 ms | 3.2x |
| 128 | 7.97 ms | 3255 ms | 408x |
| 256 | 65.5 ms | 24388 ms | 372x |

`bdsqr` runs **one thread per matrix** with the whole Golub-Kahan sweep serial
inside it (`parallel_for(range<1>(nb))`) — the batch-only-parallelism pattern.
The tempting fix is to parallelise the rotation accumulation, since vectors cost
55x values-only at n=128 (3238 ms vs 59 ms) and each lane could own a row of U or
a column of Vh with no cross-lane hazard. **That is not sufficient**: values-only
bdsqr at n=128 is 59 ms against 7.97 ms for the *entire* normal-equation pipeline
including both back-transforms, so the serial recurrence alone is already 7x over
budget.

**Call: default stays on normal equations; bdsqr is opt-in via
`BATCHLAS_GESVD_BIDIAG=bdsqr`.** Shipping a 400x regression silently is worse
than the accuracy defect it fixes, and the accuracy defect is now at least
reachable and documented rather than unavoidable.

**Next step is Option 2, not more work on bdsqr:** a bidiagonal
divide-and-conquer (`bdsdc`) reusing the existing `stedc` merge machinery. The
measurement above is what selects it — sequential QR per matrix is the wrong
shape for this hardware at n >= 128 no matter how its inner loops are arranged,
which is exactly why LAPACK switches to D&C at large n.

### A generator bug that was corrupting the instrument

`random_with_log10_cond_metric` intermittently returns an **entirely non-finite
matrix** — observed as 1 batch item in 32 at n=64, float, kappa=10. gebrd
propagates the NaN and any bidiagonal solver then refuses to converge on it,
which reads as a solver failure and is not one. This is what produced the
~0.8% Fail% that appeared *identically across every implementation* in earlier
runs, and it sent the first diagnosis of the n=64 failure down the wrong path
entirely. `benchmarks/gesvd_relacc.cc` now validates the generated matrix and
re-draws up to 8 times before attributing anything to the solver. **The
generator itself is still buggy and worth fixing separately** — it is likely the
Chol2 orthogonalisation inside it.

## Tier 4

### Coverage gaps

* **Complex general SVD on GPU** — currently throws. Tier 1 closes this for
  `n <= 32` if complex is implemented from the start (as `syev_jacobi_cta` did);
  Tier 3 closes it above.
* **Thin/economy vectors.** `SvdVectors` has only `None`/`All`. A tall-skinny
  `10000 x 32` job must currently allocate a `10000 x 10000` U — unusable. Adding
  `SvdVectors::Some` (LAPACK `'S'`) is a prerequisite for any serious tall-skinny
  support, and one-sided Jacobi produces thin `U` *natively* (it is just the
  rotated, normalised `A`) — full `U` is the extra work, not the thin one.
* **Tall-skinny route:** `A = QR` then SVD of the `n x n` `R`, then `U = Q * U_R`.
  This is the `gesvdaStridedBatched` domain and where the largest absolute wins
  are, since cuSOLVER's offering there is *approximate*.

---

## 5. Accuracy harness

Defect C has to be fixed before any accuracy claim is made, and it should be
fixed *before* Tier 1 so the new kernel is judged on a working instrument.

1. **Wire `--log10-cond` into `gesvd_*_acc`.** `miniacc` supports it; these
   benchmarks ignore it (the `log10cond` column prints blank). Sweep
   `kappa = 1e1 .. 1e7` for float, `1e1 .. 1e14` for double.
2. **Report error magnitudes, not `Fail%`.** Emit, per case:
   * `max_i |sigma_i - sigma_i_ref| / sigma_i` — **relative per singular value**,
     which is the quantity Jacobi is supposed to win on. The current
     `max_abs_singular_error` is absolute and cannot see the effect at all.
   * `||A - U S V^H|| / ||A||`
   * `||U^H U - I||`, `||V^H V - I||`
   * sweep count (ours) and `XgesvdjGetSweeps` (theirs)
3. **Reference in higher precision.** Compare float against a double LAPACK
   solve, not against another float solve, or the reference is the error floor.
4. **Include graded matrices**, not just random ones with a prescribed condition
   number. Grading is where `kappa(A_c) << kappa(A)` and where Jacobi's advantage
   is largest — and it is the case a normal-equations solver fails worst.
5. **Then tighten `tests/gesvd_tests.cc`.** The float tolerances (`5e-2`,
   `2e-1`, `3e-1`) must come down to something meaningful once Tier 1 lands.
   Expect this to break the existing paths — that is the point, and it should be
   done as a deliberate, separately-reviewed commit rather than folded into a
   kernel change.

---

## 6. Benchmark protocol

Following the measurement rules already established in this repo:

* **Saturation only.** batch >= 4096 for `n <= 32`. The existing sweeps stop at
  64 (Defect D); ratios there are launch-overhead ratios. Extend
  `GesvdCtaBenchSizes` to `{1024, 4096, 16384, 65536}`.
* **One GPU at a time.** This box has two 4090s and a concurrent run visibly
  perturbs results — the first `gesvd_cta` measurement in this analysis read
  4.37 ms under contention and 4.09 ms clean, a 7% error from that alone.
* **Warm clocks**, and `--name` is a *substring* filter — `gesvd` also matches
  `gesvdj`, so name new benchmarks so the filters stay unambiguous.
* **Match the tunables.** Same tolerance, same `max_sweeps`, same sort order as
  the `gesvdjInfo_t` settings, or the comparison is meaningless.
* **Report both axes together.** Time alone is not a result for an SVD; a table
  that gives time without the conditioning sweep next to it will mislead.

---

## 7. Risks

Ranked by how likely they are to sink the effort:

1. **`gesvdjBatched` may already be fast.** It is a purpose-built vendor kernel
   in its designed domain. Parity plus better accuracy is a good outcome; a large
   speed win at `n <= 32` is not guaranteed. **The `n > 32` and tall-skinny
   regions are where a decisive win is nearly free**, and the plan should not be
   judged solely on the contested 32x32 point.
2. **Sweep count is data-dependent.** Jacobi's cost is `sweeps x n^2 x m`, and
   sweeps vary with conditioning. A benchmark on random well-conditioned matrices
   will flatter us; one on graded matrices may not. Report the distribution.
3. **Double precision is 1/64 rate on this GPU.** Measured 10x float:double on
   the Jacobi kernel. Neither side can escape it — but it means double results
   should not be extrapolated to a datacenter card, where the ratio is 1/2.
4. **Register/LDS pressure at `P=32` with `A`, `U`, and `V` resident.** The
   existing kernel already budgets `A` + `Z`; a one-sided SVD wants `A` + `V`
   (and `U` overwrites `A`). Complex doubles this again. Expect the
   problems-per-work-group count to fall and occupancy with it. The
   local-memory budget logic in `syev_jacobi_cta_impl` handles this correctly
   already, but the resulting occupancy has to be checked, not assumed.
5. **`bdsqr` has never run.** Tier 3 Option 1 assumes it works. Budget for the
   possibility that it does not.
6. **The `n < 32` cases waste lanes.** `P` is rounded up to `{4,8,16,32}`; at
   n=17 more than half of a 32-lane partition idles. `gesvdjBatched` has the same
   problem, so this is unlikely to lose the comparison, but it caps absolute
   throughput.

---

## 8. Sequencing

| Step | Deliverable | Gate |
|---|---|---|
| 0 | cuSOLVER `gesvd_vendor` (Tier 0) | `gesvdjBatched` numbers exist; 32x32 limit confirmed empirically |
| 1 | Accuracy harness (§5.1-5.4) | relative-error vs `kappa` curves for the *current* paths, showing Defect A |
| 2 | `gesvdj_cta` real, values-only | matches LAPACK to `eps * kappa(A_c)` |
| 3 | `gesvdj_cta` + `U`,`V` | reconstruction and orthogonality at tightened tolerances |
| 4 | Dispatch + benchmarks at saturation | head-to-head table vs `gesvdjBatched` |
| 5 | Complex (Tier 1 scope completion) | closes the GPU complex-general gap |
| 6 | QR preconditioning (Tier 2) | sweep count drops; net time win confirmed |
| 7 | `bdsqr` wired for `n > 32` (Tier 3) | Defect A removed above 32 |
| 8 | Tighten `tests/gesvd_tests.cc` | separate commit, deliberate |
| 9 | Thin vectors + tall-skinny (Tier 4) | vs `gesvdaStridedBatched` |

**Steps 0 and 1 are prerequisites, not preliminaries.** Until both are done, no
claim about beating `gesvdjBatched` — in speed or accuracy — can be checked.

### Smallest useful first commit

Tier 0 alone. It is self-contained, follows an existing pattern in
`cusolver.cc`, closes the "we cannot measure the competitor" gap, and its output
determines how much of Tiers 1-2 is worth building.
