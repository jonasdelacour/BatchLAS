# GESVD {#perf_gesvd}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · checked 2026-10-06

Measurements behind the batched SVD's routing: the head-to-head against cuSOLVER `gesvdjBatched`,
accuracy against constructed spectra, the bidiagonal solvers of the blocked path, and the gebrd panel
width. Runs were made 2026-08-06 to 2026-08-08 on 2x RTX 4090, one GPU per run (`CUDA_VISIBLE_DEVICES=1`).
Design rationale: @ref design_gesvd. Raw head-to-head grid: `benchmarks/results/gesvd_vs_gesvdj_rtx4090.csv`
(Git LFS).

## gesvd: what routes today {#gesvd-what-routes-today}

`gesvd` lives in `src/ops/gesvd/gesvd.cc`, its families in `src/ops/gesvd/choice.hh`. `select::run` takes the
first family whose `can_run` admits the shape, from the nearest row of `tuned/gesvd.<dtype>.<arch>.txt`,
keyed on `herm`, `vec`, m and n. See [flat kernel selection](../design/flat-kernel-selection.md).

| family | kernel | `can_run` (correctness only) |
|---|---|---|
| `jacobi` | `gesvdj_cta` (`src/extensions/gesvdj_cta.cc`) | GPU, sub-group 32, not Hermitian, \f$\max(m,n) \le 64\f$ (32 for `complex<double>` with vectors) |
| `cta` | `gesvd_cta` (`src/extensions/gesvd_blocked.cc`) | GPU, sub-group 32, \f$\max(m,n) \le 32\f$, no thin request, real general or square Hermitian |
| `blocked` | `gesvd_blocked`: `gebrd` + `bdsdc` + `ormbr` | GPU, real general, or square Hermitian Lower |
| `vendor` | `gesvdjBatched` (CUDA) / LAPACKE loop (NETLIB) | the solver library is linked |

The tables are `source=transcribed:424a45bc` (the old router's order replayed per cell, not timed). First
family per input:

| input | first family |
|---|---|
| real general, \f$\max(m,n) \le 32\f$ | `jacobi` (`cta`, `blocked` behind it) |
| real general, \f$\max(m,n) \ge 33\f$ | `blocked`; in 33..64 `jacobi` stays behind `vendor` |
| complex general | `jacobi` up to its ceiling, then `vendor` |
| Hermitian, square | `cta` to 32, then `blocked` (Lower) or `vendor` (Upper) |
| Hermitian, non-square | `vendor` |

Pin with `BATCHLAS_GESVD_ROUTE=<family>` or `select::ScopedPin`; a pin that `can_run` refuses throws
`invalid_argument`.

## gesvd: the head-to-head against gesvdjBatched

`gesvd_vendor_benchmark`, `jobu = jobvh = All`, ms per call, saturated batch (Tier 0 run 2026-08-06 for
cuSOLVER and `gesvd_cta`; Tier 1 run for `gesvdj_cta`).

| type | n | batch | cuSOLVER | `gesvd_cta` | **`gesvdj_cta`** | vs cuSOLVER |
|---|---|---|---|---|---|---|
| float | 8 | 16384 | 5.594 | 1.095 | **0.104** | **53.7x** |
| float | 16 | 16384 | 7.596 | 1.540 | **0.891** | **8.5x** |
| float | 32 | 4096 | 3.226 | 2.126 | **2.082** | **1.55x** |
| float | 32 | 16384 | 12.581 | 7.643 | **7.804** | **1.61x** |
| double | 8 | 16384 | 42.248 | 5.470 | **1.029** | **41.1x** |
| double | 16 | 16384 | 103.520 | 18.703 | **6.926** | **15.0x** |
| double | 32 | 16384 | 364.539 | 83.075 | **49.516** | **7.4x** |

The committed CSV gives float n=32, batch 16384 `gesvdj_cta` at 7.670 ms (0.468 µs/matrix). `gesvdjBatched`
costs the same for every job combination, so it always computes vectors.

Relative error in the singular values, float n=32, against the constructed spectrum (`gesvd_relacc`):

| \f$\log_{10}\kappa\f$ | `gesvd_cta` | cuSOLVER | **`gesvdj_cta`** |
|---|---|---|---|
| 1 | 7.8e-3 | 4.09e-6 | 4.78e-6 |
| 4 | 0.299 | 8.35e-5 | **6.44e-5** |
| 5 | 1.06 | 7.40e-4 | **5.67e-4** |
| 6 | 2.13 | 9.27e-3 | **5.33e-3** |

Jacobi orthogonality stays near 1.3e-5 across six decades (cuSOLVER ~8e-6). Above \f$\kappa = 10^4\f$ Jacobi beats
cuSOLVER on accuracy. In double it holds to \f$\kappa = 10^{14}\f$. `tests/gesvdj_cta_tests.cc` covers the four types
over square, tall, wide, rank-deficient, zero and graded inputs.

### gesvd README headline: Jacobi vs gesvdjBatched per matrix {#gesvd-readme-headline-jacobi-vs-gesvdjbatched-per-matrix}

Float, `All/All`, saturated batch, 2026-08-07 (the figure quoted on the documentation front page):

| n | batch | `gesvdj_cta` µs/matrix | `gesvdjBatched` µs/matrix | ratio |
|---|---|---|---|---|
| 8 | 16384 | 0.00636 | 0.339 | 53x |
| 32 | 16384 | 0.468 | 0.768 | 1.64x |

The vendor time includes the V to \f$V^H\f$ transpose that the BatchLAS contract requires.

## gesvd: promoting Jacobi to the default at n up to 32 {#gesvd-promoting-jacobi-to-the-default-at-n-up-to-32}

Jacobi moved ahead of CTA in the default order on 2026-08-08. Float, n=32, 256 samples, `max_relerr`:

| \f$\log_{10}\kappa\f$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| CTA | 1.4e-6 | 4.7e-5 | 3.1e-3 | 0.235 | 1.046 | 1.857 |
| Jacobi | 4.8e-6 | 5.9e-6 | 1.2e-5 | 7.1e-5 | 6.3e-4 | 5.6e-3 |
| cuSOLVER | 4.1e-6 | 4.1e-6 | 1.1e-5 | 8.3e-5 | 7.7e-4 | 8.7e-3 |

CTA orthogonality runs 2.8e-6 → 0.879 → 4.371 over the sweep; Jacobi stays near 1.3e-5. Jacobi is 23x faster
than CTA at n=8 and 4.1x at n=16, and ties at n=32. CTA wins only for values-only at n ≥ 16 (2.2x at n=32), where
it has no correct digits past \f$\kappa = 10^3\f$; it stays reachable with `BATCHLAS_GESVD_ROUTE=cta`.

> **Note:** the pinning test needs a dense matrix. The older tiny-spectrum generator built an already-orthogonal
> diagonal matrix, which every path solves exactly.

## gesvd: the wide band 33 to 64 {#gesvd-the-wide-band-33-to-64}

`gesvdj_cta` covers \f$\max(m,n) \le 64\f$ (tile capacity \f$C = 64\f$) but does not close the 32 → 33 cliff. Float,
batch 4096, full vectors, µs per matrix:

| n | 32 | 33 | 48 | 64 |
|---|---|---|---|---|
| Jacobi | 0.502 | 3.177 | 4.856 | 7.253 |
| Blocked | — | 1.846 | 2.930 | 4.855 |

Jacobi is 1.5–1.7x slower than Blocked over 33..64, and pays the full 64x65 tile at n=33. Accuracy, float n=64,
`max_relerr` at \f$\kappa = 10^1 / 10^5 / 10^6\f$: Blocked 1.1e-6 / 4.6e-3 / 0.526, Jacobi 1.2e-5 / 7.3e-4 / 6.2e-3.
Blocked orthogonality at \f$10^6\f$ is 1.1e-4 after the `bdsdc` repair (0.144 before it).

Blocked is faster and more accurate below \f$\kappa \approx 10^4\f$, and the regime cannot be read from the shape.
So Jacobi heads the real tuned row only at \f$\max(m,n) \le 32\f$. In 33..64 it stays behind `blocked` and `vendor`,
and `BATCHLAS_GESVD_ROUTE=jacobi` still reaches it. Complex input in the band runs Jacobi, because the alternative
is a throw.

## gesvd defect A: the normal equations square kappa {#gesvd-defect-a-the-normal-equations-square-kappa}

`form_right_tridiagonal` (`src/extensions/gesvd_blocked.cc`) builds the tridiagonal of \f$B^T B\f$ from the bidiagonal,
so the singular values come from eigenvalues with absolute error \f$\sim \varepsilon\sigma_{\max}^2\f$:

\f[ \frac{|\delta\sigma_i|}{\sigma_i} \sim \frac{\varepsilon}{2}\left(\frac{\sigma_{\max}}{\sigma_i}\right)^2 . \f]

In float, \f$\kappa = 10^3\f$ leaves no correct digit in \f$\sigma_{\min}\f$. No tuning fixes this, which is why the
design uses one-sided Jacobi (\f$n \le 32\f$) and `bdsdc` above. The normal-equations branch remains behind
`BATCHLAS_GESVD_BIDIAG=normal`, for A/B only.

## gesvd defect C: an accuracy harness that could not see relative error {#gesvd-defect-c-an-accuracy-harness-that-could-not-see-relative-error}

`benchmarks/gesvd_relacc.cc` reports `max_relerr` (against the constructed spectrum, which is exact), `R`
(\f$\|A - U S V^H\|_F / \|A\|_F\f$), `O` (unitarity of U and V) and `log10_cond`. Sweeps run \f$\kappa = 10^1..10^7\f$
in float and \f$10^1..10^{14}\f$ in double.

> **Note:** the generator uses CGS2. Chol2 squares \f$\kappa\f$ and `potrf` fails silently, and Householder items came
> out numerically singular. miniacc's default \f$\log_{10}\kappa\f$ is NaN, so the harness records it explicitly.

`benchmarks/gesvd_cta_acc.cc` and `benchmarks/gesvd_blocked_acc.cc` take `sv_max_abs_err` against double `?gesdd`
(`batchlas::verify::singular_values`) since 2026-10-09; earlier CSVs used `?gesvd` in the working precision, so a float
column from before that date is not comparable like for like.

## gesvd defect D: perf sweeps that stopped at batch 64 {#gesvd-defect-d-perf-sweeps-that-stopped-at-batch-64}

`GesvdCtaBenchSizes` and `GesvdBlockedBenchSizes` swept batch 1..64, where at n=32 the timing is launch overhead.
`benchmarks/gesvd_vendor_benchmark.cc` sweeps from saturation (batch ≥ 4096). Run one GPU at a time (a busy
neighbour moved one run by 7%), and match the `gesvdjInfo_t` tolerance, `max_sweeps` and sort order.

## gesvd Tier 0: the cuSOLVER gesvdjBatched binding {#gesvd-tier-0-the-cusolver-gesvdjbatched-binding}

`gesvd_vendor` for `Backend::CUDA` (`src/backends/cusolver.cc`) calls `cusolverDnXgesvdjBatched` for
\f$m, n \le 32\f$ (`kGesvdjBatchedMaxDim = 32`). It throws outside that route on purpose, since a looped or approximate
solver under the same name would corrupt the comparison. It refuses Thin, which has no `econ` flag.

Facts not to regress:

- cuSOLVER returns V, so `gesvd_detail::write_vh_from_v` runs last, out of place.
- Views are checked for tight packing (the call takes no strides); the buffer-size query mirrors every allocation.
- The event comes from `create_event_after_external_work()`, not `get_event()`.
- `cusolverDnDestroyGesvdjInfo` is skipped if `check_status` throws in between (as in `syev_vendor`).

## gesvdj_cta Tier 2: preconditioning, tested and rejected {#gesvdj_cta-tier-2-preconditioning-tested-and-rejected}

QR preconditioning did not reduce sweeps (float n=32, 64 samples: 8.95 at \f$\kappa = 10^1\f$, 15.22 at \f$10^6\f$, on A
and on R from geqrf alike), so it is not in the kernel.

| Alternative | Result | Verdict |
|---|---|---|
| QR preconditioning (on R, out of kernel) | sweeps identical to within 0.1 | rejected |
| de Rijk pre-ordering (columns by descending norm) | 8.91 vs 8.95 sweeps at \f$10^1\f$, 15.53 vs 15.22 at \f$10^6\f$ | removed |
| Untaken pre-ordering branch kept in the kernel | 13% slower (7.80 → 8.88 ms, n=32, batch 16384), register pressure | deleted |
| Column-pivoted QR (`geqp3`) | no batched pivoted QR in the tree; no sweep gain | not built |

What it did produce: QR preconditioning improves relative accuracy about 2x at high conditioning (float n=32,
\f$\kappa=10^6\f$: 2.58e-3 vs 5.11e-3), so it is the lever if an accuracy-only option is ever wanted. The
`cta_wg_size_multiplier` default of 1 stays (7.67 ms at 1, 8.85 at 2, 9.01 at 4).

> **Note:** an early run showed sweeps falling from 15.2 to 8.9 because `MatrixView::triangularize` had its `uplo`
> inverted, so the kernel factored geqrf's reflectors. The tell was a 5e-7 reconstruction with singular values off
> by 30x. Check reconstruction before trusting a preconditioning result.

## gesvd Tier 3: bdsdc as the bidiagonal solver {#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver}

Above 32 the blocked path runs `gebrd`, a bidiagonal SVD, then `ormbr`. `bdsdc` (`src/extensions/bdsdc.cc`) is the
default bidiagonal solver: an order-\f$2n\f$ tridiagonal eigenproblem solved by `stedc` (derivation in @ref design_gesvd).

| n (batch 512, float) | `stedc` at 2n | `bdsqr` |
|---|---|---|
| 64 | 0.85 ms | 643 ms |
| 256 | 11.1 ms | 24388 ms |

Accuracy, `gesvd_relacc`, float, n=64, 1024 samples (`max_relerr`, normal equations → bdsdc):

| \f$\kappa\f$ | 1e2 | 1e3 | 1e4 | 1e6 |
|---|---|---|---|---|
| normal-eq | 5.0e-5 | 9.4e-2 | 4.1e-1 | 8.5e-1 |
| bdsdc | 1.8e-6 | 9.7e-6 | 9.4e-5 | 5.0e-1 |

Full vectors, float, normal equations vs bdsdc (ms):

| n | batch | normal-eq | bdsdc | ratio | bdsqr |
|---|---|---|---|---|---|
| 64 | 512 | 202 | 201 | 1.00x | 643 |
| 128 | 512 | 8.4 | 10.0 | 1.19x | 3255 |
| 256 | 512 | 66.3 | 76.4 | 1.15x | 24388 |

The n=64 row is dominated by the unblocked gebrd (see the cliff section). `bdsdc` needs a \f$2n \times 2n\f$ eigenvector
matrix per item, about 4x the tridiagonal memory, and its workspace sizing mirrors the direct-solve branch.
`BATCHLAS_GESVD_BIDIAG=normal` restores the normal equations, and `=bdsqr` selects the sequential sweep.

`bdsqr` keeps one advantage: zero-shift QR keeps relative accuracy for tiny \f$\sigma\f$ (\f$\kappa = 10^6\f$, n=64: relerr
0.198 vs 0.497), at 300–400x the cost, because it runs one thread per matrix. Above 32, `bdsdc` cannot fix \f$\sigma\f$ at
\f$\kappa = 10^6\f$ (0.52 relative), which is gebrd's own backward-stability floor. Only one-sided Jacobi (up to 64) gets
past it.

### gesvd bdsdc: null-space repair criteria that failed {#gesvd-bdsdc-null-space-repair-criteria-that-failed}

With \f$k\f$ zero singular values the null space of the Golub–Kahan matrix is degenerate as a whole, and the selected
columns' halves need not be orthogonal. Rejected:

| Alternative | Result | Verdict |
|---|---|---|
| Repair from the partner (\f$-\sigma\f$) column | equal half-norms and a flipped sign tie-break: 1.18 relative reconstruction error with no zero singular values | rejected |
| Detect damage by a vanishing half-norm | n=16, two zero \f$\sigma\f$: norms 1.000, yet \f$U_{14}\cdot U_{15} = 1.0\f$ | rejected: only \f$\sigma\f$ sees it |

Columns with small \f$\sigma\f$ are rebuilt by Gram–Schmidt against the final columns. Two traps:

- The tolerance must not carry a factor of n: \f$8n\varepsilon\f$ is 2.4e-4 relative at n=256 in float, which fired on
  ordinary random matrices (298 ms of a 354 ms solve).
- Choose the candidate axis; do not search it. Searching took 251 attempts for U (93 ms of a 162 ms solve). The residual
  \f$1 - \sum_t Q_{ct}^2\f$ ranks every axis in one pass.

### gesvd bdsdc: the repair threshold from the resolution limit {#gesvd-bdsdc-the-repair-threshold-from-the-resolution-limit}

Column \f$i\f$ contributes orthogonality error \f$\sim \varepsilon\sigma_{\max}/(2\sigma_i)\f$. Once `stedc` cannot resolve the
\f$\pm\sigma_i\f$ gap of \f$2\sigma_i\f$, U gets parallel columns well before \f$\sigma = 0\f$. For target \f$\tau\f$ the repair
must cover \f$\sigma_i \ge \varepsilon\sigma_{\max}/(2\tau)\f$. With \f$\tau = 10^{-3}\f$ the multiplier is 500
(`bdsdc_repair_degenerate`, `kOrthTarget`). The earlier \f$16\varepsilon\sigma_{\max}\f$ threshold fails the graded
regression test (n=64, with vectors).

| n=64, float | \f$\kappa \le 10^4\f$ | \f$10^5\f$ | \f$10^6\f$ |
|---|---|---|---|
| ortho | unchanged | 5.2e-4 → 4.6e-5 | 0.143 → 1.1e-4 |
| recon | unchanged | 1.3e-6 → 7.2e-5 | 3.1e-6 → 7.3e-5 |

The multiplier 500 is the knee: 3000 gives 6.6e-5 orthogonality, but reconstruction at \f$\kappa = 10^4\f$ rises to 4.2e-4.
Double accuracy and speed are unchanged.

### gesvd bdsqr: the missing zero-shift chase {#gesvd-bdsqr-the-missing-zero-shift-chase}

`bdsqr` did not handle a negligible diagonal entry: with `db[l] = 0` the sweep ran to `maxit`. LAPACK's zero-shift chase
is needed in both directions (right inside the block, left at the bottom). Handling one direction never converges.

> **Note:** the test must use mixed-sign \f$d, e\f$, exact zeros and six decades of conditioning. A first version drew only
> positive \f$d, e\f$ and passed with the bug present.

## gesvd: the unblocked-gebrd cliff {#gesvd-the-unblocked-gebrd-cliff}

The blocked gebrd threshold is 1 (`gesvd_use_blocked_gebrd`, `src/extensions/gesvd_blocked.cc`). Blocked wins at every
n from 33 in both types. Float, batch 256, full vectors, ms (unblocked → blocked):

| n | 33 | 48 | 64 | 127 |
|---|---|---|---|---|
| unblocked | 10.92 | 44.83 | 99.94 | 609.1 |
| blocked | 1.06 | 1.44 | 2.12 | 5.83 |
| speedup | 10.3x | 31.2x | 47.2x | 104x |

Tall input with \f$n \le 32 < m\f$ lands here, because CTA and Jacobi need \f$\max(m,n) \le 32\f$. At m=1024 (float, batch 128,
µs per matrix) blocked is 64x faster at n=8 and 179x at n=32. Square input also wins blocked (32x32: 48.67 → 1.84 µs), so the
threshold is a floor, not a tuned constant. `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` overrides it.

> **Note:** the setting is parsed with bare `atoi` on purpose. An unparseable value yields 0, which is below the default
> and silently selects the blocked path. `env_int_or` would change that.

## gesvd: the gebrd panel-width split from ORMQR {#gesvd-the-gebrd-panel-width-split-from-ormqr}

`gebrd_blocked`'s panel width is `tuning::gebrd_block_size_for_n` (`GEBRD_BLOCK_SIZE_*` in
`include/batchlas/tuning_params.hh`), separate from `ORMQR_BLOCK_SIZE_*`. The two have opposite gradients: gebrd is flat,
ormqr varies 2.16x between 16 and 56. The solve and its buffer-size twin must move together. Panel-width sweep,
n=512, batch 256, float, `gesvd.gebrd` stage (ms):

| nb | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|
| `gesvd.gebrd` | 234.9 | 232.1 | **230.7** | 235.5 | 240.9 | 256.6 |

The optimum is 16, flat within 2% from 8 to 24. Per-case winners at `jobu = jobvh = None`: n=128: 8, n=256: 8, n=512: 16,
n=1024: 16. Shipped buckets (`GEBRD_BLOCK_SIZE_{TINY,SMALL,MEDIUM,LARGE,XLARGE}`) are 16, 8, 8, 16, 16; TINY was not swept.
End to end the retune gives 1.03–1.15x (float, with vectors and without).

## gesvd: open debts

- **gesvdjBatched tunables.** Only `SetSortEig` is applied. Tolerance and `max_sweeps` are cuSOLVER defaults, and executed
  sweeps are not reported.
- **Sweep-count distribution.** Only means are recorded. Random well-conditioned input flatters Jacobi.
- **32x32 vendor limit unprobed.** `kGesvdjBatchedMaxDim` comes from the programming guide. Planned: probe
  `cusolverDnSgesvdjBatched_bufferSize` at 32/33/48/64.
- **No looped `gesvdj` or `gesvdaStridedBatched` route above 32.** `gesvdaStridedBatched` is approximate, so it must not
  share a benchmark name with the exact route.
- **Update-phase barrier in `gesvdj_cta` unmeasured.** Under lane = row it guards no cross-lane hazard (see @ref design_gesvd).
- **Graded arm and a real `ok` predicate in `gesvd_relacc`.** `ok` is a finiteness check today.
- **32 → 33 cliff for real input.** A \f$C = 48\f$ rung needs \f$P = 16\f$ (three rows per lane).
- **Rescale headroom at \f$C = 64\f$.** The global rescale ignores columns 32..63, so overflow headroom for graded 33..64 input
  is narrower than @ref design_gesvd claims. Observed, not measured.
- **`complex<double>` with vectors stops at 32** (local memory; see @ref design_gesvd).
- **Tall-skinny route** (QR, SVD of R) is the `gesvdaStridedBatched` domain, where cuSOLVER is only approximate.
- **Double results do not transfer to a 1:2 FP64 card.** The float:double ratio here is about 10x on the Jacobi kernel.
