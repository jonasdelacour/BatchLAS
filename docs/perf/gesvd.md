# gesvd: measured evidence for the batched SVD {#perf_gesvd}

**Covers:** every measurement behind the batched SVD's routing and design: the
head-to-head against cuSOLVER `gesvdjBatched`, the four defects that decided the plan, the
one-sided Jacobi kernel (`gesvdj_cta`), the bidiagonal solvers for the blocked path
(`bdsdc`, `bdsqr`, normal equations), the preconditioning tier that was tested and rejected,
the gebrd panel width, and the instrument bugs found along the way.
**Status:** current for routing and the kernels in the tree; sections marked *historical*
record the state before a later change and say which change superseded them.
**Machine:** 2x RTX 4090 (sm_89), one GPU pinned per run (`CUDA_VISIBLE_DEVICES=1`,
uncontended), CUDA 13.3 as recorded in the source plan. **Dates:** 2026-08-06 to 2026-08-08
(commits `c76a01f` .. `3fa2abf`).
**Sources:** distilled from the root `GESVD_PLAN.md` (written at `5ade468`) and the
later commit messages that measured against it (the original is
`git show bdcdfa5c:GESVD_PLAN.md`). The design rationale lives in
@ref design_gesvd; the raw head-to-head grid is `benchmarks/results/gesvd_vs_gesvdj_rtx4090.csv`
(Git LFS).

## gesvd: what routes today

Checked against the tree on 2026-10-06, after
[flat kernel selection](../design/flat-kernel-selection.md) replaced the `RouteTable` router.
`gesvd` lives in `src/ops/gesvd/gesvd.cc`; its families are spelled in
`src/ops/gesvd/choice.hh`. `select::run` takes the first entry that `can_run` admits from the
nearest row of `tuned/gesvd.<dtype>.<arch>.txt` (arch `sm_89` or `sm_120`), keyed on `herm`
(N, L or U), `vec` (none, all or thin, taken from the *canonical* jobs) and m, n.

| family | kernel | `can_run` (correctness only) |
|---|---|---|
| `jacobi` | `gesvdj_cta` (`src/extensions/gesvdj_cta.cc`) | GPU, sub-group 32, not Hermitian, \f$\max(m,n) \le 64\f$ (32 for `complex<double>` with vectors; `gesvd_jacobi_max_dim`) |
| `cta` | `gesvd_cta` (`src/extensions/gesvd_blocked.cc`) | GPU, sub-group 32, \f$\max(m,n) \le 32\f$, no thin request, real general or square Hermitian |
| `blocked` | `gesvd_blocked`, `gebrd` + `bdsdc` + `ormbr` | GPU, real general, or square Hermitian Lower |
| `vendor` | `gesvdjBatched` (CUDA) / LAPACKE loop (NETLIB) | the solver library is linked |

The tables are `source=transcribed:424a45bc`, the old router's preference order replayed per
grid cell (`tuned/README.md`), so the rules on this page are what they encode:

| input | first family in the row |
|---|---|
| real general, \f$\max(m,n) \le 32\f$ | `jacobi` (`cta`, `blocked` behind it) |
| real general, \f$\max(m,n) \ge 33\f$ | `blocked`; in 33..64 `jacobi` stays in the row behind `vendor`, so a pin reaches it |
| complex general | `jacobi` up to its ceiling, then `vendor` |
| Hermitian Lower, square | `cta` to 32, `blocked` above |
| Hermitian Upper, square | `cta` to 32, `vendor` above |
| Hermitian, non-square | `vendor` |

`BATCHLAS_GESVD_ROUTE=<family>` (or a `select::ScopedPin`) pins a family; a pin that `can_run`
refuses throws `invalid_argument`. The pre-flat knob `BATCHLAS_GESVD_PROVIDER`, which some
sections below name, is how those measurements were taken; it is no longer read. The rows below
are the evidence for each of the choices above.

## gesvd: the head-to-head against gesvdjBatched

`gesvd_vendor_benchmark`, float and double, `jobu = jobvh = All`, ms per call. cuSOLVER and
`gesvd_cta` columns are from the Tier 0 run; `gesvdj_cta` from the Tier 1 run
(2026-08-06, `da26852`).

| type | n | batch | cuSOLVER | `gesvd_cta` (normal eq.) | **`gesvdj_cta`** | vs cuSOLVER |
|---|---|---|---|---|---|---|
| float | 8 | 16384 | 5.594 | 1.095 | **0.104** | **53.7x** |
| float | 16 | 16384 | 7.596 | 1.540 | **0.891** | **8.5x** |
| float | 32 | 4096 | 3.226 | 2.126 | **2.082** | **1.55x** |
| float | 32 | 16384 | 12.581 | 7.643 | **7.804** | **1.61x** |
| double | 8 | 16384 | 42.248 | 5.470 | **1.029** | **41.1x** |
| double | 16 | 16384 | 103.520 | 18.703 | **6.926** | **15.0x** |
| double | 32 | 16384 | 364.539 | 83.075 | **49.516** | **7.4x** |

The committed CSV is the re-run after de Rijk ordering was removed (2026-08-07, `a0a9753`;
three arms, `All/All`, batch 4096 and 16384). Its float n=32/16384 Jacobi row is
7.670 ms (0.468 µs/matrix); later no-op refactors re-measured it at 0.470 and 0.450
µs/matrix (`53c28ad`, `8bb04f2`).

Relative error in the singular values, n=32 float, against the *known* constructed spectrum
(`gesvd_relacc`; sample count not recorded for this run):

| \f$\log_{10}\kappa\f$ | `gesvd_cta` | cuSOLVER | **`gesvdj_cta`** |
|---|---|---|---|
| 1 | 7.8e-3 | 4.09e-6 | 4.78e-6 |
| 3 | 1.06e-2 | 1.03e-5 | 1.25e-5 |
| 4 | 0.299 | 8.35e-5 | **6.44e-5** |
| 5 | 1.06 | 7.40e-4 | **5.67e-4** |
| 6 | 2.13 | 9.27e-3 | **5.33e-3** |

`gesvdj_cta` orthogonality is flat at ~1.3e-5 over all six decades (cuSOLVER ~8e-6;
`gesvd_cta` collapses to 2.43). At \f$\kappa \ge 10^4\f$ Jacobi is *more* accurate than
cuSOLVER. In double it holds to \f$\kappa = 10^{14}\f$ (6.2e-4 relative, 1.4e-14
orthogonality).

At landing (`da26852`) `tests/gesvdj_cta_tests.cc` had 32 cases over {float, double,
complex\<float\>, complex\<double\>} × {square, tall, wide, rank-deficient, all-zeros, graded,
job combinations, public-API dispatch}, all passing, with `gesvd_tests` unchanged at 26. The
same commit closed the complex *general* GPU gap (previously it fell through to Vendor and
threw). Three defects of the reviewed design surfaced only while implementing it (completion
acceptance threshold, \f$A^H\f$ vs \f$A^T\f$ for m < n, one-sided norm clamp); none was
detectable by the gesvd tolerances of the time. They are recorded with their mechanisms in
@ref design_gesvd.

*Historical targets, superseded by the table above:* the plan's first Tier 1 bar (before the
CSV existed) was "cuSOLVER-class accuracy while staying under cuSOLVER's 3.25 ms at
n=32/batch=4096", with a projected landing zone of 3-5 ms at batch 8192 (from
`syev_jacobi_cta`'s 2.11 ms plus the extra dot products and V accumulation). The impl spec
then reset the bar to the batch-16384 numbers quoted in the Tier 0 section below.

### gesvd README headline: Jacobi vs gesvdjBatched per matrix

The number `README.md` quotes, read from `benchmarks/results/gesvd_vs_gesvdj_rtx4090.csv`
(float, `All/All`, RTX 4090, 2026-08-07):

| n | batch | `gesvdj_cta` µs/matrix | `gesvdjBatched` µs/matrix | ratio |
|---|---|---|---|---|
| 8 | 16384 | 0.00636 | 0.339 | 53x |
| 32 | 16384 | 0.468 | 0.768 | 1.64x |

Both are saturated-batch, single-machine numbers. The vendor time includes the V to
\f$V^H\f$ conjugate transpose BatchLAS's contract requires (see the fairness note in
`benchmarks/gesvd_vendor_benchmark.cc`).

### gesvd Tier 0 baseline: the normal-equations CTA path already won on speed (historical)

Before the Jacobi kernel existed, n=32 float, `CUDA_VISIBLE_DEVICES=1`, std dev ≤ 0.5% of mean:

| batch | jobu/jobvh | `gesvdjBatched` | `gesvd_cta` | speedup |
|---|---|---|---|---|
| 4096 | None/None | 3.372 ms | 0.911 ms | 3.70x |
| 4096 | All/All | 3.250 ms | 2.131 ms | 1.53x |
| 16384 | None/None | 12.159 ms | 3.248 ms | 3.74x |
| 16384 | All/All | 12.664 ms | 7.827 ms | 1.62x |

From the same CSV at n=32/16384, all four job combinations (`gesvd_cta` vs `gesvdjBatched`):
None/None 3.230 vs 12.148 (3.8x), None/All 4.608 vs 12.654 (2.7x), All/None 6.920 vs 12.245
(1.8x), All/All 7.691 vs 12.680 (1.6x). Double, n=32/16384/All: 365.2 vs 80.9 ms (4.5x).
`gesvdjBatched` barely moves with `jobz` (3.37 vs 3.25 ms), so it appears to compute vectors
regardless.

And its accuracy (`gesvd_relacc --samples=128 --log10-cond=1..6 --type=float 32`):

| \f$\log_{10}\kappa\f$ | BatchLAS max_relerr | cuSOLVER max_relerr | BatchLAS ortho | cuSOLVER ortho |
|---|---|---|---|---|
| 1 | 7.8e-3 | 4.1e-6 | 2.9e-6 | 7.7e-6 |
| 2 | 7.9e-3 | 4.1e-6 | 1.0e-4 | 7.5e-6 |
| 3 | 1.1e-2 | 1.0e-5 | 5.8e-3 | 7.6e-6 |
| 4 | 0.299 | 8.4e-5 | 1.11 | 8.2e-6 |
| 5 | 1.06 | 7.4e-4 | 1.50 | 8.1e-6 |
| 6 | 2.13 | 9.3e-3 | 2.43 | 8.0e-6 |

cuSOLVER tracks \f$\varepsilon\kappa\f$ (the Demmel–Veselić bound) with flat orthogonality;
the normal-equations path is at ~1e5 ε already at \f$\kappa=10\f$ and has lost all
orthogonality by \f$\kappa = 10^4\f$. **This inverted the plan:** the problem was never
speed but keeping speed while fixing accuracy, so the Tier 1 bar became "beat 7.69 ms
(strictly better than `gesvd_cta`) and 12.68 ms (still beats the vendor) at float/n=32/
batch=16384/All/All, with cuSOLVER-class relative accuracy". The design review projected
5.5-8 ms; it landed at 7.80, then 7.67.

Pre-plan baseline of `gesvd_cta` (n=32, batch=8192, square, uncontended): float All/All
4.09 ms, All/None 3.56, None/All 2.25, None/None 1.66; double All/All 40.7, None/None 30.9.
Reference points at the same shape: `syev_jacobi_cta` 2.11 ms float / 19.3 ms double,
`syev_cta` 0.988 / 31.7. float:double is ~10x on this card (FP64 at 1/64 rate), and in
double two-sided Jacobi already beat the tridiagonal path.

## gesvd: promoting Jacobi to the default at n up to 32

`da920c3` (2026-08-08) moved `native:jacobi` ahead of `native:cta`. Before it, Jacobi sat
after CTA and was reachable only for complex input or by forcing. n=32 float, 256 samples:

| \f$\log_{10}\kappa\f$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| CTA max_relerr | 1.4e-6 | 4.7e-5 | 3.1e-3 | 0.235 | 1.046 | 1.857 |
| Jacobi max_relerr | 4.8e-6 | 5.9e-6 | 1.2e-5 | 7.1e-5 | 6.3e-4 | 5.6e-3 |
| cuSOLVER max_relerr | 4.1e-6 | 4.1e-6 | 1.1e-5 | 8.3e-5 | 7.7e-4 | 8.7e-3 |

CTA orthogonality over the same sweep runs 2.8e-6 → 0.879 → 4.371; Jacobi stays near 1.3e-5.
Time per matrix, float, saturated batch, `jobu=All`: Jacobi is 23x faster at n=8, 4.1x at
n=16, parity at n=32. Adding U costs CTA 17x at n=8 (0.065 → 1.126 µs) and Jacobi nothing
(0.0448 → 0.0452 µs), because U *is* the rotated, normalised A.

The one regime where CTA wins is **values-only at n ≥ 16 (2.2x at n=32)**, exactly where it
has no correct digits past \f$\kappa = 10^3\f$. That is not a default; it stays reachable
with `BATCHLAS_GESVD_ROUTE=cta` (measured as `BATCHLAS_GESVD_PROVIDER=cta`, since retired). The earlier plan to gate the promotion on
"`gesvdj_cta` ≤ `gesvd_cta` in time at every point" (the impl spec's F.6 exit criterion)
was superseded by this measurement: the deciding axis is accuracy.

The test that pins this was checked to fail on the old routing (\f$\sigma_{27}\f$ off by 38%
at \f$\kappa = 10^5\f$), and needed a dense test matrix: the pre-existing tiny-spectrum
generator built a diagonal matrix whose columns are already orthogonal, which every path
solves exactly.

## gesvd: the wide band 33 to 64

`8bb04f2` (2026-08-08) extended `gesvdj_cta` to \f$\max(m,n) \le 64\f$ (tile capacity
\f$C = 64\f$, two rows per lane). It did **not** close the n=32 → 33 cliff. batch=4096,
float, full vectors, µs per matrix:

| n | 32 | 33 | 40 | 48 | 64 |
|---|---|---|---|---|---|
| Jacobi | 0.502 | 3.177 | 3.746 | 4.856 | 7.253 |
| Blocked | — | 1.846 | — | 2.930 | 4.855 |

Jacobi is 1.5-1.7x slower than Blocked across 33..64: n=33 pays the full \f$C = 64\f$ tile
(64x65 elements, 63 rounds), ~6x the n=32 cost before occupancy. Accuracy at n=64, float,
256 samples:

| \f$\log_{10}\kappa\f$ | 1 | 3 | 5 | 6 |
|---|---|---|---|---|
| Blocked max_relerr | 1.1e-6 | 1.7e-5 | 4.6e-3 | 0.526 |
| Jacobi max_relerr | 1.2e-5 | 2.3e-5 | 7.3e-4 | 6.2e-3 |
| Blocked ortho | 7.4e-7 | 3.0e-6 | 4.9e-4 | 0.144 |
| Jacobi ortho | 2.4e-5 | 3.0e-5 | 3.5e-5 | 3.7e-5 |

(The Blocked ortho row predates the `bdsdc` threshold fix below, which brought 0.144 to
1.1e-4.) Blocked is faster **and** more accurate below \f$\kappa \approx 10^4\f$, so there is
no regime-free win and the regime is not knowable from the shape. Hence the wide-band rule:
for real input Jacobi heads the tuned row only at \f$\max(m,n) \le 32\f$, but `can_run`
admits it up to 64 and it stays in the 33..64 rows behind `blocked` and `vendor`, so
`BATCHLAS_GESVD_ROUTE=jacobi` still reaches it.
(Under the old router this was `preferred()` against `supports()`.) Complex general input in
the band runs Jacobi automatically, because the alternative is a throw.

## gesvd defect A: the normal equations square kappa

`form_right_tridiagonal` (`src/extensions/gesvd_blocked.cc`) builds, from `gebrd`'s
bidiagonal \f$d, e\f$,

\f[ T_{ii} = d_i^2 + e_{i-1}^2, \qquad T_{i,i+1} = d_i e_i, \f]

the tridiagonal of \f$B^T B\f$, formed explicitly; the pipeline then takes
\f$\sigma_i = \sqrt{\lambda_i}\f$. A symmetric eigensolver returns \f$\lambda_i\f$ with absolute
error \f$\sim \varepsilon \sigma_{\max}^2\f$, so

\f[ \frac{|\delta\sigma_i|}{\sigma_i} \sim \frac{\varepsilon}{2}\left(\frac{\sigma_{\max}}{\sigma_i}\right)^2 . \f]

The relative error of a small singular value grows with \f$\kappa^2\f$; in float
(\f$\varepsilon \approx 6\times10^{-8}\f$) \f$\kappa = 10^3\f$ leaves no correct digit in
\f$\sigma_{\min}\f$. The Tier 0 accuracy table above is the measurement. No kernel tuning fixes
this; the algorithm has to change, which is why the design moved to one-sided Jacobi
(\f$n \le 32\f$) and to `bdsdc` (above). The normal-equations bidiagonal branch still exists,
selected by `BATCHLAS_GESVD_BIDIAG=normal` for A/B only.

At the time, `tests/gesvd_tests.cc` passed it with float tolerances 5e-2 (singular values),
2e-1 (orthogonality), 3e-1 (relative reconstruction), wide enough to admit a result with no
accuracy. *Superseded* by `de294cd` (2026-08-08): the float tolerances are now
2e-3 / 1e-3 / 1e-4 and solver-aware; only `BATCHLAS_GESVD_BIDIAG=normal` keeps the old
bounds (`tests/gesvd_tests.cc`, `gesvd_sv_tol` and siblings), with 48 passing cases verified
on each of the three `BATCHLAS_GESVD_BIDIAG` settings.

## gesvd defect B: no vendor baseline existed

At plan time `gesvd_vendor` threw for every backend except NETLIB, and
`src/backends/cusolver.cc` had no SVD binding at all. Every performance claim was
unfalsifiable until Tier 0 bound `gesvdjBatched`.

## gesvd defect C: an accuracy harness that could not see relative error

`benchmarks/gesvd_{cta,blocked}_acc.cc` reported only `Fail%` against the loose tolerances
(`gesvd_cta_acc --samples=32` gave 0.00000 for every type and size), used an **absolute**
singular-value error that cannot express loss of relative accuracy in the small
\f$\sigma\f$, never wired `--log10-cond` (the `log10cond` column printed blank), and chose
metric names outside miniacc's terminal whitelist. `benchmarks/gesvd_relacc.cc` fixes all of
it: \f$\max_i |\sigma_i - \sigma_i^{\mathrm{ref}}| / \sigma_i^{\mathrm{ref}}\f$ (`max_relerr`),
\f$\|A - U S V^H\|_F / \|A\|_F\f$ (`R`), \f$\max(\|U^H U - I\|, \|V V^H - I\|)\f$ (`O`), a
recorded `log10_cond`, `NormType::Spectral` (the Frobenius generator throws when
\f$\log_{10}\kappa_F < \log_{10} n\f$), and an **exact** reference: the generator builds
\f$A = U S V^H\f$ from a constructed spectrum, so a second floating-point solve never sets the
error floor. The superseded `_acc` harnesses were planned for deletion with the tolerance
tightening.

Harness mechanics that are easy to get wrong again: miniacc's default target
\f$\log_{10}\kappa\f$ is NaN and prints as an empty cell (the blank column), so the harness
records a metric literally named `log10_cond`, also on its failure path; the terminal summary
prints only whitelisted metric names (everything else reaches only `--csv`); per-sample rows
are retained only when `--csv` is passed; and miniacc filters with `--benchmark_filter=GLOB`,
not minibench's substring `--name`. The plan's intended sweep range was \f$\kappa = 10^1..10^7\f$
for float and \f$10^1..10^{14}\f$ for double.

Still open from the plan: a graded-matrix arm (\f$A = D G\f$, where
\f$\kappa(A_c) \ll \kappa(A)\f$ and Jacobi's advantage is largest; random-orthogonal
\f$U S V^H\f$ is the case where it has the *least* advantage), and a `C·ε·κ`-based `ok`
predicate (today it is a finiteness check).

## gesvd defect D: perf sweeps that stopped at batch 64

`GesvdCtaBenchSizes` / `GesvdBlockedBenchSizes` swept batch ∈ {1..64}. At n=32 batch=64 is
launch overhead, not the algorithm. `benchmarks/gesvd_vendor_benchmark.cc` sweeps from
saturation (batch ≥ 4096). Protocol notes that produced the numbers on this page:

- One GPU at a time: the first `gesvd_cta` measurement read 4.37 ms under contention from
  the other card and 4.09 ms clean, a 7% error from that alone.
- `--name` is a substring filter (`gesvd` matches `gesvdj`), which is why the head-to-head
  lives in its own binary.
- Match `gesvdjInfo_t` tolerance, `max_sweeps` and sort order, and report both axes
  together. Only `SetSortEig` is applied today; see open debts.

## gesvd Tier 0: the cuSOLVER gesvdjBatched binding

`gesvd_vendor` for `Backend::CUDA` in `src/backends/cusolver.cc` calls
`cusolverDnXgesvdjBatched`. Its documented limit is \f$m, n \le 32\f$
(`kGesvdjBatchedMaxDim = 32`). Above 32 there is no batched exact vendor SVD, only
`gesvdaStridedBatched` (approximate, \f$m \ge n\f$) or a loop over `gesvdj`. The binding
**throws** outside the batched route on purpose: substituting a looped or approximate
solver under the same name would corrupt the comparison it exists to make. It also refuses
Thin (the batched call has no `econ` flag; at \f$\le 32\f$ square input canonicalises Thin to
All anyway).

Binding facts not to regress: `SvdVectors` maps to `CUSOLVER_EIG_MODE_*` by hand;
cuSOLVER returns V and the contract is \f$V^H\f$, so an out-of-place conjugate transpose
(`gesvd_detail::write_vh_from_v`) runs last and the returned event is a SYCL event, not
external work; every view is checked for tight packing (the call takes no strides); the
buffer-size query mirrors each allocation in order through `BumpAllocator`. The event after
the cuSOLVER call itself must come from `create_event_after_external_work()`, not
`get_event()`, because external library calls never update the queue's cached last event.
The complex forms take a *real* S, hence `base_float_ptr_convert` on the singular-value
pointer. `cusolverDnDestroyGesvdjInfo` is skipped (leaks the info object) if a
`check_status` between create and destroy throws; this matches `syev_vendor`, and
`op_external` is a tag, not a scope guard.

Strategic read at the time: for \f$n > 32\f$ BatchLAS wins by default (there is no batched
competitor), so the contested region is \f$n \le 32\f$.

## gesvdj_cta: Tier 2 preconditioning, tested and rejected

The plan called QR preconditioning "the main speed lever". Measured negative
(`a0a9753`, 2026-08-07). Per-problem sweep counts (`GesvdjParams::sweep_counts`, the analogue
of `cusolverDnXgesvdjGetSweeps`) made it decidable. Mean sweeps, n=32 float, 64 samples:

| \f$\kappa\f$ | 1e1 | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|---|
| on A (baseline) | 8.95 | 12.03 | 13.25 | 14.55 | 15.22 |
| on R from geqrf | 8.97 | 12.02 | 13.23 | 14.50 | 15.22 |

Cost is \f$\text{sweeps}\times n^2 m\f$ and sweeps nearly double over the range, so the lever
looked real. QR was tested out of kernel: Q is orthogonal, so \f$\sigma(R) = \sigma(A)\f$
exactly and the kernel can be scored on R against the same known spectrum without building
Q. Identical to within 0.1 sweeps: a fused in-kernel QR would pay for the factorisation and
the back-application of Q and get nothing back.

**de Rijk pre-ordering** (sort columns by descending norm): no effect, removed.

| \f$\kappa\f$ | 1e1 | 1e4 | 1e6 |
|---|---|---|---|
| with | 8.91 | 13.52 | 15.53 |
| without | 8.95 | 13.25 | 15.22 |

Merely having the untaken branch in the kernel cost **13% of wall clock** (7.80 → 8.88 ms at
n=32/batch=16384) through register pressure, so it was deleted rather than left behind a
flag. This table is the record.

**Column-pivoted QR (geqp3):** not attempted. There is no batched pivoted QR in the tree,
and the two testable mechanisms of the same family (ordering columns by norm) moved the
sweep count by nothing. Do not build it on the literature's claim alone.

**A false positive nearly got through.** The first run showed sweeps dropping 15.2 → 8.9
(an apparent 1.7x). The R being factored was not R: `MatrixView::triangularize` had its
`uplo` inverted on column-major storage, so the kernel was handed geqrf's Householder
reflectors. The tell was a 5e-7 reconstruction with singular values off by 30x. Fixed in
`bdb1807` (`src/matrix.cc`, `triangularize`).

What Tier 2 did produce:

- The sweep-count diagnostic, visible in `gesvd_relacc` as `iterations_done`.
- QR preconditioning **does** improve relative accuracy about 2x at high conditioning
  (n=32 float: `max_relerr` 4.03e-5 vs 6.56e-5 at \f$\kappa=10^4\f$; 2.58e-3 vs 5.11e-3 at
  \f$10^6\f$). If an accuracy-critical mode is ever wanted, this is the lever, as an option.
- `cta_wg_size_multiplier` at n=32/batch=16384/float: 1 → 7.67 ms, 2 → 8.85, 4 → 9.01. The
  default of 1 is right; more problems per work-group trades local memory for warps and
  the trade is not monotone.

Remaining speed work at \f$n \le 32\f$ is micro-architectural (the Gram phase still runs in
full in late, mostly converged sweeps), not algorithmic.

## gesvd Tier 3: bdsdc as the bidiagonal solver

Above 32 the Blocked path runs `gebrd` → bidiagonal SVD → `ormbr`. `bdsdc`
(`src/extensions/bdsdc.cc`, `3235d99`, 2026-08-07) is the default bidiagonal solver; it
reduces the order-\f$n\f$ bidiagonal SVD to an order-\f$2n\f$ tridiagonal eigenproblem solved by
`stedc` (derivation in @ref design_gesvd). The measurement that chose this over porting
LAPACK's `dlasd*`, `stedc` at order \f$2n\f$ vs `bdsqr` on the same bidiagonal, batch=512,
float, vectors:

| gesvd n | `stedc` at 2n | `bdsqr` |
|---|---|---|
| 64 | 0.85 ms | 643 ms |
| 128 | 3.19 ms | 3255 ms |
| 256 | 11.1 ms | 24388 ms |

Accuracy, `gesvd_relacc`, float, 1024 samples, n=64:

| \f$\kappa\f$ | normal-eq relerr | bdsdc relerr | normal-eq ortho | bdsdc ortho |
|---|---|---|---|---|
| 1e2 | 5.0e-5 | 1.8e-6 | 1.1e-4 | 1.1e-6 |
| 1e3 | 9.4e-2 | 9.7e-6 | 3.8e-1 | 2.8e-6 |
| 1e4 | 4.1e-1 | 9.4e-5 | 6.8e-1 | 1.9e-5 |
| 1e6 | 8.5e-1 | 5.0e-1 | 1.6e+0 | 1.4e-4 |

Confirmed at n=128 (\f$\kappa = 10^4\f$: relerr 0.433 → 2.5e-4, ortho 0.475 → 3.5e-5). Cost,
float, full vectors:

| n | batch | normal-eq | bdsdc | ratio | bdsqr |
|---|---|---|---|---|---|
| 64 | 512 | 202 ms | 201 ms | 1.00x | 643 ms |
| 128 | 512 | 8.4 ms | 10.0 ms | 1.19x | 3255 ms |
| 256 | 512 | 66.3 ms | 76.4 ms | 1.15x | 24388 ms |
| 512 | 256 | 291 ms | 324 ms | 1.11x | — |

(The n=64 row is the unblocked-gebrd cliff below, not a solver cost.) Price: a
\f$2n \times 2n\f$ eigenvector matrix per item, ~4x the tridiagonal path's memory.
`BATCHLAS_GESVD_BIDIAG=normal` restores the normal equations, `=bdsqr` the sequential
sweep. `bdsqr` keeps one advantage: zero-shift QR keeps relative accuracy for tiny
\f$\sigma\f$ where divide and conquer does not (\f$\kappa=10^6\f$, n=64: relerr 0.198 vs 0.497),
at 300-400x the cost. That cost is structural, not a tuning problem: `bdsqr` runs one
*thread* per matrix with the whole sweep serial inside it, and values-only `bdsqr` at n=128
(59 ms) is already 7x the entire normal-equations pipeline including its back-transforms.

Workspace consequence: the bidiagonal solver's own workspace (for `bdsdc`, a
\f$2k \times 2k\f$ eigenvector matrix per item plus `stedc` at order \f$2k\f$) is the dominant
term of `gesvd_native_buffer_size` on a direct solve. `bdsqr`'s used to fit inside the
tridiagonal path's over-allocation by accident; `bdsdc`'s does not, which is why the sizing
function mirrors the run path's direct-solve branch exactly. The per-item eigenvector matrix is
about 4x what the tridiagonal path allocates; `BATCHLAS_GESVD_BIDIAG=normal` trades that memory
back for the squared condition number.

**Values floor above 32.** `bdsdc` cannot fix \f$\sigma\f$ at \f$\kappa = 10^6\f$ (0.52
relative): that is `gebrd`'s own backward-stability floor, since it perturbs A by
\f$\varepsilon\|A\|\f$. Only a method that never forms the bidiagonal (one-sided Jacobi, up to
64) reaches past it.

### gesvd bdsdc: null-space repair criteria that failed

With \f$k\f$ zero singular values the \f$2k\f$-dimensional null space of the Golub–Kahan matrix is
degenerate as a whole, and the selected columns' halves need not be mutually orthogonal.
Two plausible fixes were implemented and measured wrong:

- <b>Repair from the partner (\f$-\sigma\f$) column:</b> wrong for well-separated \f$\sigma\f$; both
  partners have equal half-norms, the tie-break flips u's sign: 1.18 relative
  reconstruction error on a matrix with no zero singular values. Also fails at null-space
  dimension 3.
- **Detect damage by a vanishing half-norm:** at n=16 with two zero singular values every
  column had norm 1.000 and healthy halves, yet \f$U_{14}\cdot U_{15} = 1.0\f$ with V
  orthogonal. Only \f$\sigma\f$ sees it coming.

What ships: columns with small \f$\sigma\f$ are rebuilt by Gram–Schmidt against the columns
already final. Two performance traps in that repair:

- **The tolerance must not carry a factor of n.** \f$8n\varepsilon\f$ is 2.4e-4 relative at
  n=256 float, sweeps in informative \f$\sigma\f$, and made the repair fire on ordinary random
  matrices: 298 ms of a 354 ms solve.
- **Choose the candidate axis, do not search it.** Walking \f$e_0, e_1, \dots\f$ is quadratic in
  the wrong place: the left null vector of an upper bidiagonal sits at the end of the range
  (n=256: V accepted on attempt 1, U on attempt 251, 93 ms of a 162 ms solve). The residual
  of \f$e_c\f$ against an orthonormal \f$Q\f$ is \f$1 - \sum_t Q_{ct}^2\f$, so one pass ranks
  every axis.

### gesvd bdsdc: the repair threshold from the resolution limit

The plan shipped the threshold \f$16\varepsilon\sigma_{\max}\f$. *Superseded* by `3fa2abf`
(2026-08-08): the \f$\pm\sigma_i\f$ pair has gap \f$2\sigma_i\f$; once `stedc` cannot resolve it
the two eigenvectors are an arbitrary basis of the pair's 2-D eigenspace and both normalise
to the same (v, u), so U gets parallel columns well before \f$\sigma = 0\f$. Column i
contributes \f$\sim \varepsilon\sigma_{\max}/(2\sigma_i)\f$ orthogonality error, so a target
\f$\tau\f$ needs \f$\sigma_i \ge \varepsilon\sigma_{\max}/(2\tau)\f$; with \f$\tau = 10^{-3}\f$ the
multiplier is 500 (`bdsdc_repair_degenerate`, `kOrthTarget`). `gesvd_relacc`, float, 128
samples:

| n=64 | \f$\kappa \le 10^4\f$ | \f$10^5\f$ | \f$10^6\f$ |
|---|---|---|---|
| ortho | unchanged | 5.2e-4 → 4.6e-5 | 0.143 → 1.1e-4 |
| recon | unchanged | 1.3e-6 → 7.2e-5 | 3.1e-6 → 7.3e-5 |

n=128 at \f$\kappa = 10^6\f$: ortho 0.200 → 2.5e-4. 500 is the knee: 3000 gives 6.6e-5
orthogonality but reconstruction at \f$\kappa = 10^4\f$ falls to 4.2e-4. Double unchanged. No
performance change (n=64 batch=4096 5.01 vs 4.86 µs/matrix). The regression test is graded,
n=64, with vectors, and was verified to fail on 16 (1.6e-2 mean, 1.41 on one item).

### gesvd bdsqr: the missing zero-shift chase

`bdsqr` had 420 lines and zero callers, and a real bug: no handling of a negligible diagonal
entry. With `db[l] = 0` the shift gives \f$f = -\mu\f$, \f$g = 0\f$, `lartg` returns the identity
and the sweep spins to `maxit` (20x `maxit` changed nothing, which is how it was found).
LAPACK DBDSQR's zero-shift chase was added in both directions (zero inside the block: chase
right with left rotations; zero at the bottom: chase left with right rotations). Handling
only the first case is itself a bug that never converges; the tests caught it. The first
`tests/bdsqr_tests.cc` passed with the bug present because it drew only positive
\f$d, e \in [0.3, 1.7]\f$; the case that matters has mixed signs, exact zeros and six decades.

## gesvd: the unblocked-gebrd cliff

Found by noticing that the Tier 3 cost table ran n=64 at 201 ms and n=128 at 10.0 ms.
`gesvd_use_blocked_gebrd` returned \f$n \ge 128\f$, so \f$33 \le n \le 127\f$ ran the level-2
panel-serial bidiagonalisation. float, batch=256, full vectors, ms:

| n | 33 | 36 | 48 | 64 | 96 | 127 | 128 |
|---|---|---|---|---|---|---|---|
| unblocked | 10.92 | 15.50 | 44.83 | 99.94 | — | 609.1 | — |
| blocked | 1.06 | 1.13 | 1.44 | 2.12 | 3.84 | 5.83 | 6.02 |
| speedup | 10.3x | 13.7x | 31.2x | 47.2x | — | 104x | — |

Blocked n=40 was 1.25 ms, so the blocked curve is continuous across the CTA hand-off. Blocked
won at every n from 33 up in both types; the old threshold of 128 was never a measured
tuning constant. n=127 cost 101x what n=128 cost, and n=64 cost 16x what a problem eight
times larger cost. Double, batch=256: n=64 107.8 → 10.6 ms (10.2x), n=127 646.9 →
36.8 ms (17.6x). Accuracy unchanged (n=64, \f$\kappa=10^4\f$, float, 512 samples: ortho 1.78e-5 →
1.79e-5, residual 1.29e-6 → 1.32e-6). The plan set the threshold to 33 (`50fb37a`), on the
reasoning that the CTA path takes over below it.
*Superseded* by `fea82ed`: the threshold is now **1**, because that reasoning holds only for
square input. The CTA and Jacobi predicates both require \f$\max(m,n) \le 32\f$, so tall input
with \f$n \le 32 < m\f$ fails both and lands here, where the m rows make the level-2 path
ruinous. No benchmark covered that band (it needs \f$\min(m,n) \le 32\f$ to reach it and
\f$m > 32\f$ to be slow) because every gesvd benchmark built a square `Random(n, n)` until the
m/n split. m=1024, batch=128, float, `jobu=None jobvh=All`, µs per matrix:

| n | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| unblocked | 138.73 | 604.48 | 1535.2 | 2537.2 |
| blocked | 2.15 | 5.69 | 9.42 | 14.15 |
| speedup | 64x | 106x | 163x | 179x |

Square input is faster blocked too (8x8 0.935 → 0.370 µs, 32x32 48.67 → 1.84 µs, 26x), so
there is no n at which the unblocked path is the right choice and the threshold is a floor,
not a tuned constant. The threshold lives in `gesvd_use_blocked_gebrd`
(`src/extensions/gesvd_blocked.cc`). `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` overrides it, which is
how the tables above were taken; set it above the largest n to get the unblocked path back.
The setting is parsed with bare `atoi` on purpose: an unparseable value yields 0, which is
*below* the default of 1 and so silently widens the blocked path rather than falling back to
the unblocked one. Routing it through `env_int_or` would change that.

## gesvd: the gebrd panel-width split from ORMQR

`gebrd_blocked`'s panel width is `tuning::gebrd_block_size_for_n` (`GEBRD_BLOCK_SIZE_*` in
`include/batchlas/tuning_params.hh`). Until `ca349b73` (2026-08-06) `gesvd` sized its
bidiagonal reduction from `ORMQR_BLOCK_SIZE_*`, an unrelated kernel with the opposite gradient
(flat for gebrd, 2.16x between 16 and 56 for ormqr), so one knob pinned the steep parameter at
the flat one's optimum. Only 2 of the 13 `ormqr_block_size_for_n` call sites were gebrd (the
solve and its buffer-size twin, which must move together). Verified orthogonal after the split,
gesvd n=512, batch=256, float, RTX 4090:

| setting | gebrd | left back-transform |
|---|---|---|
| baseline | 230.1 ms | 18.33 ms |
| `BATCHLAS_TUNE_GEBRD_BLOCK_SIZE=48` | 260.8 ms | 18.39 ms (only gebrd moves) |
| `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=48` | 230.7 ms | 14.78 ms (only ormqr moves) |

Before the split the ORMQR override at 48 moved both and cost gesvd-with-vectors 7.9%; after it, the
same value is a 1.8% win (1136.7 → 1116.9 µs). The split shipped 16 in every bucket, exactly the
old values, so it changed no behaviour by itself.

The nb curve that justified the split, `gesvd_blocked` n=512, batch=256, float, RTX 4090,
`gesvd.gebrd` stage timer (2026-08-06):

| nb | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|
| `gesvd.gebrd` (ms) | 234.9 | 232.1 | **230.7** | 235.5 | 240.9 | 256.6 |

gebrd's optimum at n=512 is 16 and the curve is flat around it (within 2% from 8 to 24); ormqr's
is steep over the same range, which is why one shared knob could not serve both.

The retune that used it (`924b3a59`, 2026-08-07, float, A/B'd end to end at the consumers) set
`GEBRD_BLOCK_SIZE_{SMALL,MEDIUM} = 8` (the small sizes want 8, n ≥ 512 wants 16). gesvd
before → after for the whole retune (which also moved ORMQR, stedc and sy2sb constants, so
these are not gebrd-only): n=128 batch=1024 9.2465 → 8.0684 ms (1.146x); n=256 batch=512
94.987 → 92.016 ms (1.032x); n=512 batch=256 with vectors 1143.3 → 1110.3 ms (1.030x). As with
all tuning in the tree, this was float-only.

The retune's per-case nb winners, at `jobu = jobvh = None` where `gesvd.gebrd` is about 95% of
the call: n=128: 8, n=256: 8, n=512: 16, n=1024: 16. The axis is worth 3.5-48% depending on n, so
nb is flat only near n=512, not everywhere. Shipped, by bucket of
`tuning::gebrd_block_size_for_n` (n ≤ 64 / ≤ 128 / ≤ 256 / ≤ 512 / above):
`GEBRD_BLOCK_SIZE_{TINY,SMALL,MEDIUM,LARGE,XLARGE}` = 16, 8, 8, 16, 16. The TINY bucket was not
in the sweep and keeps the pre-split 16.

## gesvd: the non-finite generator that corrupted the instrument

`random_with_log10_cond_metric` intermittently returned an entirely non-finite matrix (1
item in 32 at n=64; at n=128 Fail% 50% even with re-draws). `gebrd` propagated the NaN, any
bidiagonal solver then refused to converge, and the ~0.8% Fail% that appeared identically
across every implementation, plus the first misdiagnosis of the n=64 `bdsqr` failure, came
from this. `gesvd_relacc` validates and re-draws. Root cause and fix (`bdb1807`): Chol2
orthonormalisation squares an unconditioned input's \f$\kappa\f$ and `potrf` fails silently;
the default is now CGS2. Householder was measured and rejected because it leaves items
numerically singular (achieved \f$\log_{10}\kappa\f$ up to 17.39 when 5 was requested), which
is worse because it is invisible.

## gesvd: open debts

- **gesvdjBatched tunables.** Only `SetSortEig` is applied; tolerance and `max_sweeps` are
  cuSOLVER's defaults, and cuSOLVER's executed sweeps are not reported. The comparison
  above is therefore "default vs default". The spec's plan: optional
  `SetTolerance`/`SetMaxSweeps` with 0 meaning "cuSOLVER default" (so existing numbers stay
  valid), and `cusolverDnXgesvdjGetSweeps` exposed separately for `gesvd_relacc` only,
  because it is a host-blocking read that must not sit inside the async `gesvd_vendor`.
- **Sweep-count distribution, not the mean.** Jacobi's cost is \f$\text{sweeps}\times n^2 m\f$
  and sweeps depend on the data; random well-conditioned input flatters it. Only means are
  recorded above.
- **The 32x32 vendor limit was never probed empirically.** `kGesvdjBatchedMaxDim` still comes
  from the programming guide. The planned probe: call `cusolverDnSgesvdjBatched_bufferSize`
  at 32/33/48/64 and record the raw status (not through `check_status`, which folds it into
  a string); if 33+ succeeds the contested region moves.
- **No looped-gesvdj or gesvdaStridedBatched route** above 32; the binding throws there by
  design until each is added under its own label. Facts for whoever adds them:
  `gesvdaStridedBatched` takes strides, a `rank` argument and a host output `h_R_nrmF`, is
  restricted to \f$m \ge n\f$ and is *approximate*, so it must never share a benchmark name
  with the exact route; looped `gesvdj` has an extra `econ` argument the batched form lacks.
  Both still need `write_vh_from_v`, and every new allocation must be mirrored in
  `gesvd_vendor_buffer_size`.
- **The update-phase barrier in `gesvdj_cta` is still there.** Under lane = row it guards no
  cross-lane hazard (see @ref design_gesvd); the spec planned its deletion as a separate,
  measured change. Not done, not measured.
- **Graded-matrix harness arm and a meaningful `ok` predicate** in `gesvd_relacc` (defect C).
- **The n=32 → 33 cliff for real input.** A \f$C = 48\f$ rung would help 33..48 but needs
  \f$P = 16\f$ (three rows per lane); a separate measured change.
- **Rescale headroom on the C=64 rung.** `gesvdj_cta`'s global rescale reduces `nmax`/`nmin`
  over columns `lane < CC` only, so columns 32..63 do not influence \f$\beta\f$. Correctness is
  unaffected (\f$\beta\f$ is a power of two), but the overflow/underflow headroom for graded
  33..64 input is narrower than the scaling argument in @ref design_gesvd claims. Observed,
  not measured; a fix would cover all C columns in C/P passes, as the exact-norm reduction does.
- **`complex<double>` with vectors** stops at 32 (local memory; see @ref design_gesvd).
- **Tall-skinny route** (QR, then SVD of R, then \f$U = Q U_R\f$): the `gesvdaStridedBatched`
  domain, where cuSOLVER's offering is only approximate. `SvdVectors::Thin` (`2fcc1db`,
  `ceb4b59`) was the prerequisite and has landed.
- **Double results do not transfer to a 1:2 FP64 card**; the float:double ratio here is ~10x
  on the Jacobi kernel.
