# GESVD {#perf_gesvd}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · checked 2026-10-29

Measurements behind the batched SVD's routing and tuning constants. Runs on 2x RTX 4090, one GPU per run,
2026-08-06 to 2026-10-23. Design rationale: @ref design_gesvd. Raw head-to-head grid:
`benchmarks/results/gesvd_vs_gesvdj_rtx4090.csv` (Git LFS).

## gesvd: what routes today {#gesvd-what-routes-today}

Families in `src/ops/gesvd/choice.hh`, `can_run` in `src/ops/gesvd/gesvd.cc`; order from the nearest row of
`tuned/gesvd.<arch>.txt` (keys dtype, `herm`, `vec`, m, n). See [flat kernel selection](../design/flat-kernel-selection.md).

| family | kernel | `can_run` (correctness only) |
|---|---|---|
| `jacobi` | `gesvdj_cta` | GPU, sub-group 32, not Hermitian, \f$\max(m,n) \le\f$ `gesvd_jacobi_max_dim` (64; 32 for `complex<double>` with vectors) |
| `cta` | `gesvd_cta` | GPU, sub-group 32, \f$\max(m,n) \le 32\f$, no thin request, real general or square Hermitian |
| `blocked` | `gebrd` + `bdsdc` + `ormbr` | GPU; any general input, or square Hermitian (either triangle) |
| `vendor` | `gesvdjBatched` (CUDA) / LAPACKE loop | solver linked; CUDA only inside cuSOLVER's envelope (\f$\max(m,n) \le 32\f$, packed A and U, not thin, not values-only non-square); never ROCm |

Every family also needs a homogeneous, unmasked batch. A row ranking `vendor` above a shape the binding refuses
falls through to the next entry or the last resort (`blocked`). The sm_89 table is transcribed (old router order,
untimed); sm_120 is deep-measured. First family on sm_89:

| input | first family |
|---|---|
| real general, \f$\max(m,n) \le 32\f$ | `jacobi`, then `cta`, `blocked`, `vendor` |
| real general, 33..64 | `blocked`, `vendor`, `jacobi` |
| complex\<float\> general, 33..64 | `blocked`, `jacobi`, `vendor` |
| complex\<double\> general, 33..64 | values only: `jacobi`, `blocked`; with vectors: `blocked` |
| any general, above 64 | `blocked`, `vendor` |
| Hermitian, square | `cta` to 32, then `blocked`; Upper rows copy Lower |
| Hermitian, non-square | `vendor` |

Pin with `BATCHLAS_GESVD_ROUTE=<family>` or `select::ScopedPin`; a pin `can_run` refuses throws `invalid_argument`.

## gesvd README headline: Jacobi vs gesvdjBatched per matrix {#gesvd-readme-headline-jacobi-vs-gesvdjbatched-per-matrix}

`gesvd_vendor_benchmark`, float, `jobu = jobvh = All`, batch 16384, 2026-08-07 (quoted on the front page). The
vendor time includes the V to \f$V^H\f$ transpose the contract requires; `gesvdjBatched` always computes vectors.
`gesvdj_cta` vs `gesvdjBatched`, µs per matrix: n=8 0.00636 vs 0.339 (53x); n=32 0.468 vs 0.768 (1.64x). Double at
n = 8 / 16 / 32: 41x / 15x / 7.4x (1:64 FP64 card; does not transfer to a 1:2 card).

## gesvd: promoting Jacobi to the default at n up to 32 {#gesvd-promoting-jacobi-to-the-default-at-n-up-to-32}

Float, n=32, 256 samples, `max_relerr` at \f$\kappa = 10^3 / 10^4 / 10^6\f$ (2026-08-08): CTA 3.1e-3 / 0.235 / 1.86,
Jacobi 1.2e-5 / 7.1e-5 / 5.6e-3, cuSOLVER 1.1e-5 / 8.3e-5 / 8.7e-3. CTA orthogonality reaches 4.4; Jacobi stays near 1.3e-5. Jacobi is 23x faster than CTA at n=8, 4.1x at
n=16, tied at n=32. CTA wins only values-only at n ≥ 16 (2.2x at n=32) with no correct digits past \f$\kappa = 10^3\f$.
The pinning test needs a dense matrix: a diagonal test matrix is solved exactly by every path.

## gesvd: the wide band 33 to 64 {#gesvd-the-wide-band-33-to-64}

Float, batch 4096, full vectors, µs per matrix: Jacobi 3.18 / 4.86 / 7.25, Blocked 1.85 / 2.93 / 4.86 at n = 33 / 48 / 64
(Jacobi 0.50 at 32: the tile is 64x65 from n=33). Accuracy, float n=64, `max_relerr` at \f$\kappa = 10^1/10^5/10^6\f$:
Blocked 1.1e-6 / 4.6e-3 / 0.526, Jacobi 1.2e-5 / 7.3e-4 / 6.2e-3. Blocked is faster and more accurate below
\f$\kappa \approx 10^4\f$ and the regime is not visible from the shape, so real 33..64 rows put `blocked` first;
`BATCHLAS_GESVD_ROUTE=jacobi` reaches Jacobi.

## gesvd defect A: the normal equations square kappa {#gesvd-defect-a-the-normal-equations-square-kappa}

Singular values taken from the eigenvalues of \f$B^T B\f$ have \f$|\delta\sigma_i|/\sigma_i \sim
\tfrac{\varepsilon}{2}(\sigma_{\max}/\sigma_i)^2\f$: in float, \f$\kappa = 10^3\f$ leaves no correct digit in
\f$\sigma_{\min}\f$. Hence Jacobi to 32 and `bdsdc` above; the normal-equations branch remains behind
`BATCHLAS_GESVD_BIDIAG=normal` for A/B only.

## gesvd defect C: an accuracy harness that could not see relative error {#gesvd-defect-c-an-accuracy-harness-that-could-not-see-relative-error}

`benchmarks/gesvd_relacc.cc` reports `max_relerr` (against the exact constructed spectrum), `R`
(\f$\|A - U S V^H\|_F / \|A\|_F\f$), `O` (unitarity) and `log10_cond`, over \f$\kappa = 10^1..10^7\f$ (float) and
\f$10^1..10^{14}\f$ (double). The generator uses CGS2 (Chol2 squares \f$\kappa\f$; Householder items came out singular);
miniacc's default \f$\log_{10}\kappa\f$ is NaN, so the harness records it. `gesvd_cta_acc` and `gesvd_blocked_acc`
compare `sv_max_abs_err` against double `?gesdd` since 2026-10-09; float CSVs from before used working-precision
`?gesvd` and are not comparable.

## gesvd defect D: perf sweeps that stopped at batch 64 {#gesvd-defect-d-perf-sweeps-that-stopped-at-batch-64}

At n=32, batch ≤ 64 measures launch overhead. `benchmarks/gesvd_vendor_benchmark.cc` sweeps from batch 4096. Run
one GPU at a time (a busy neighbour moved a run by 7%) and match `gesvdjInfo_t` tolerance, `max_sweeps` and sort order.

## gesvd Tier 0: the cuSOLVER gesvdjBatched binding {#gesvd-tier-0-the-cusolver-gesvdjbatched-binding}

`gesvd_vendor` on CUDA (`src/backends/cusolver.cc`) calls `cusolverDnXgesvdjBatched` for \f$m, n \le 32\f$
(`kGesvdjBatchedMaxDim`) and throws outside it, Thin included: a looped or approximate solver under the same name would
corrupt the comparison. Do not regress: V is transposed into \f$V^H\f$ last, out of place (`write_vh_from_v`); views must
be tightly packed and the buffer-size query mirrors every allocation; the event comes from
`create_event_after_external_work()`; `cusolverDnDestroyGesvdjInfo` runs even when `check_status` throws.

## gesvdj_cta Tier 2: preconditioning, tested and rejected {#gesvdj_cta-tier-2-preconditioning-tested-and-rejected}

Float n=32, 64 samples; mean sweeps 8.95 at \f$\kappa = 10^1\f$, 15.22 at \f$10^6\f$.

| Alternative | Result | Verdict |
|---|---|---|
| QR preconditioning (on R, out of kernel) | sweeps unchanged within 0.1; relerr 2x better at \f$10^6\f$ (2.58e-3 vs 5.11e-3) | rejected; the lever for an accuracy-only option |
| de Rijk pre-ordering | 8.91 / 15.53 sweeps | removed |
| untaken pre-ordering branch in the kernel | 13% slower (7.80 → 8.88 ms, batch 16384), register pressure | deleted |
| `cta_wg_size_multiplier` 2 / 4 | 8.85 / 9.01 ms vs 7.67 at 1 | default 1 kept |

Check reconstruction before trusting a preconditioning result: an inverted `uplo` in `MatrixView::triangularize` once
showed a false sweep drop (5e-7 reconstruction, singular values off by 30x).

## gesvd Tier 3: bdsdc as the bidiagonal solver {#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver}

Above 32 the blocked path runs `gebrd`, a bidiagonal SVD, then `ormbr`. `bdsdc` (`src/ops/gesvd/bdsdc/bdsdc.cc`), the
default, solves the order-\f$2n\f$ Golub–Kahan tridiagonal with `stedc` (@ref design_gesvd). Float, batch 512, full
vectors, n = 128 / 256: normal-eq 8.4 / 66.3 ms, `bdsdc` 10.0 / 76.4 ms, `bdsqr` 3255 / 24388 ms. Accuracy, float n=64, `max_relerr` at \f$\kappa = 10^3/10^4/10^6\f$: normal-eq 9.4e-2 / 0.41 / 0.85, `bdsdc` 9.7e-6 /
9.4e-5 / 0.50. `bdsdc` needs a \f$2n \times 2n\f$ eigenvector matrix per item (about 4x the tridiagonal memory).
`BATCHLAS_GESVD_BIDIAG=normal|bdsqr` switches solver. `bdsqr`'s zero-shift QR keeps tiny \f$\sigma\f$ more accurate
(\f$\kappa = 10^6\f$: 0.198 vs 0.497) at 300–400x the cost (one thread per matrix). The 0.5 floor at \f$10^6\f$ is gebrd's
backward stability; only Jacobi (to 64) gets past it.

### gesvd bdsdc: null-space repair criteria that failed {#gesvd-bdsdc-null-space-repair-criteria-that-failed}

With zero singular values the Golub–Kahan null space is degenerate and the selected columns' halves need not be
orthogonal. Columns with small \f$\sigma\f$ are rebuilt by Gram–Schmidt against the final columns.

| Alternative | Result | Verdict |
|---|---|---|
| Repair from the partner (\f$-\sigma\f$) column | flipped sign tie-break: 1.18 relative reconstruction error with no zero \f$\sigma\f$ | rejected |
| Detect damage by a vanishing half-norm | n=16, two zero \f$\sigma\f$: norms 1.000 yet \f$U_{14}\cdot U_{15} = 1.0\f$ | rejected |
| Tolerance \f$8n\varepsilon\f$ | fired on ordinary random matrices at n=256 (298 ms of 354) | rejected: no factor of n |
| Search the candidate axis | 251 attempts for U (93 ms of 162) | rejected: rank axes by \f$1 - \sum_t Q_{ct}^2\f$ in one pass |

### gesvd bdsdc: the repair threshold from the resolution limit {#gesvd-bdsdc-the-repair-threshold-from-the-resolution-limit}

Column \f$i\f$ contributes orthogonality error \f$\sim \varepsilon\sigma_{\max}/(2\sigma_i)\f$, so for target \f$\tau\f$ the
repair must cover \f$\sigma_i \ge \varepsilon\sigma_{\max}/(2\tau)\f$; \f$\tau = 10^{-3}\f$ (`kOrthTarget`) gives multiplier
500. Float n=64: orthogonality at \f$10^6\f$ goes 0.143 → 1.1e-4, reconstruction 3.1e-6 → 7.3e-5; unchanged at
\f$\kappa \le 10^4\f$. 500 is the knee: 3000 gives 6.6e-5 orthogonality but 4.2e-4 reconstruction at \f$10^4\f$. The
earlier \f$16\varepsilon\sigma_{\max}\f$ fails the graded regression test.

### gesvd bdsqr: the missing zero-shift chase {#gesvd-bdsqr-the-missing-zero-shift-chase}

A negligible diagonal entry needs LAPACK's zero-shift chase in both directions (right inside the block, left at the
bottom); one direction never converges. The test needs mixed-sign \f$d, e\f$, exact zeros and six decades of
conditioning: positive-only data passed with the bug present.

## gesvd: L2 batch chunking {#gesvd-l2-batch-chunking}

The general blocked path runs the batch in parts (A plus the requested U and \f$V^H\f$ per item; workspace for one
part); the Hermitian arm and cta/jacobi are not chunked. `GESVD_BLOCKED_L2_CHUNK_PERCENT` = 150 for real scalars;
complex is not L2-chunked (unmeasured). Every type's parts are capped at `max_groups_yz / 2`: without it ormbr's
`PackPPanelKernel` aborted from batch 65505 at n=40
([defect 21](../design/known-defects.md#defect-21-range-rounded-past-the-grid)).

Cold A/B, float, pinned `blocked`, time relative to unchunked at 75 / **150** / 300 %, 2026-10-23 (72 MB L2): n=128
vectors b8192 1.093 / **0.975** / 0.996; n=256 vectors b1024 0.878 / **0.712** / 0.874; worst case n=128 values-only
b16384 1.017 / **1.021** / 1.004.

## gesvd: the unblocked-gebrd cliff {#gesvd-the-unblocked-gebrd-cliff}

The blocked gebrd threshold is 1 (`gesvd_use_blocked_gebrd`): blocked wins at every n. Float, batch 256, full vectors:
10.3x at n=33, 47x at 64, 104x at 127; tall \f$m = 1024\f$, \f$n \le 32\f$ (which lands here): 64–179x; square 32x32:
48.67 → 1.84 µs. `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` overrides it, parsed with bare `atoi` on purpose: an unparseable
value yields 0, which still selects blocked (`env_int_or` would change that).

## gesvd: the gebrd panel-width split from ORMQR {#gesvd-the-gebrd-panel-width-split-from-ormqr}

`gebrd_blocked`'s panel width is `tuning::gebrd_block_size_for_n` (`GEBRD_BLOCK_SIZE_*`), separate from
`ORMQR_BLOCK_SIZE_*`: gebrd is flat where ormqr varies 2.16x between 16 and 56. The solve and its buffer-size twin must
move together. n=512, batch 256, float, `gesvd.gebrd` stage: nb 8 / 16 / 24 / 48 = 234.9 / **230.7** / 235.5 / 256.6 ms.
Per-case winners (values only): 8 at n = 128, 256; 16 at 512, 1024. Shipped TINY..XLARGE = 16, 8, 8, 16, 16 (TINY
unswept); end to end 1.03–1.15x.

## gesvd: complex general on the blocked path {#gesvd-complex-general-on-the-blocked-path}

`gebrd` reduces complex A to a *real* bidiagonal (real-beta `zlarfg`, `src/ops/gesvd/gebrd/larfg_real_beta.hh`), the
real `bdsdc` solves it, its vectors are widened into U and \f$V^H\f$, and the complex `unmbr` runs unchanged; m < n
solves \f$A^H\f$. complex\<float\>, ms per call, `blocked` vs `jacobi`: n=33 b2048 N/N 3.43 vs 39.7; n=64 b2048 A/A
13.76 vs 464.6; n=64 b128 A/A 2.83 vs 33.0. Above 64 (no route before): 6.64 / 32.7 / 379.0 ms at n = 128 / 256 / 512,
b128 A/A. So complex\<float\> 33..64 rows put `blocked` first (ledger run `transcribed-167a4eaf`); Jacobi keeps better relative
accuracy at \f$\kappa \gtrsim 10^4\f$ and stays reachable by pin. complex\<double\> was not timed (1:64 FP64).
`GesvdGeneralComplexTest.BlockedGradedPaddedShapesAndJobs` checks values at \f$\kappa = 10^6\f$; `bdsdc` vectors lose
orthogonality as ~eps·κ (3e-3 float, 3e-11 double), so vectors are checked at κ = 10.

## gesvd: open debts

- **gesvdjBatched tunables.** Only `SetSortEig` is applied; tolerance and `max_sweeps` are defaults; sweeps unreported.
  Its 32x32 limit (`kGesvdjBatchedMaxDim`) is from the guide, unprobed at 33/48/64.
- **Sweep-count distribution.** Only means are recorded; random well-conditioned input flatters Jacobi.
- **No looped `gesvdj` or `gesvdaStridedBatched` route above 32.** The latter is approximate and must not share a
  benchmark name with the exact route; it is also the tall-skinny (QR, SVD of R) domain.
- **Update-phase barrier in `gesvdj_cta` unmeasured** (guards no cross-lane hazard under lane = row; @ref design_gesvd).
- **`gesvd_relacc` lacks a graded arm and a real `ok` predicate** (`ok` is a finiteness check).
- **32 → 33 cliff for real input.** A \f$C = 48\f$ rung needs \f$P = 16\f$ (three rows per lane).
- **Rescale headroom at \f$C = 64\f$.** The global rescale ignores columns 32..63, so headroom for graded 33..64 input
  is narrower than @ref design_gesvd claims. Observed, not measured.
- **`complex<double>` with vectors stops at 32** (local memory; @ref design_gesvd).
