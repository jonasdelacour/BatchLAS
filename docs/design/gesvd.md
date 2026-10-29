# gesvd design {#design_gesvd}

> **Status:** current · checked 2026-10-29

Batched SVD has four families, ranked per shape by `tuned/gesvd.<device>.txt`: one-sided Jacobi
(`jacobi`, \f$\max(m,n) \le 64\f$, lower where local memory is short), the normal-equations `cta`
path (\f$\le 32\f$), the bidiagonal `blocked` path (`gebrd`, `bdsdc` on the batched `stedc`,
`ormbr`), and the vendor. Complex general input takes `blocked`: `gebrd` uses a real-beta
`zlarfg`, so the bidiagonal is real and the real `bdsdc` solves it; its vectors are widened before
the complex `unmbr`. Measurements: @ref perf_gesvd "the GESVD performance page".

| Piece | Source |
|---|---|
| entry point, `can_run`, workspace, launch | `src/ops/gesvd/gesvd.cc` |
| families and table keys (`herm`, `vec`, `m`, `n`) | `src/ops/gesvd/choice.hh` |
| ceilings (`gesvd_jacobi_max_dim`, `kGesvdCtaMaxDim`, `kGesvdVendorCudaMaxDim`) | `src/ops/gesvd/gesvd_native.hh` |
| one-sided Jacobi kernel | `src/ops/gesvd/kernels/gesvdj_cta.cc` |
| `cta` and `blocked` drivers | `src/ops/gesvd/kernels/gesvd_blocked.cc` |
| sub-ops | `src/ops/gesvd/{gebrd,bdsdc,bdsqr,ormbr}/` |
| `GesvdjParams` (`max_sweeps` 30, `tol_multiplier` 1) | `include/batchlas/blas/extensions.hh` |

## gesvd design: why one-sided Jacobi

The accuracy target is cuSOLVER `gesvdjBatched`: relative accuracy
\f$|\delta\sigma_i|/\sigma_i \le O(\varepsilon)\,\kappa(A_c)\f$ with \f$A_c\f$ column-equilibrated
(Demmel–Veselić 1992). Paths through the tridiagonal of \f$B^T B\f$ square \f$\kappa\f$
([defect A](../perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa)).

Hestenes one-sided Jacobi rotates column pairs until the columns are orthogonal; then
\f$\sigma_i = \|A_i\|\f$, \f$U_i = A_i/\sigma_i\f$, and V is the accumulated rotation. The
rotation test is relative, on the Gram of the current columns:
\f$|a_{pq}| > t\,n\,\varepsilon\sqrt{|a_{pp}|\,|a_{qq}|}\f$. An absolute test against
\f$\max|a_{kl}|\f$ loses the accuracy (LAWN 169, Remark 2.2). Jacobi also needs no back-transform
and runs as one kernel, where `cta` needs five or more launches. `gesvdj_cta` is
`syev_jacobi_cta` without its \f$A \leftarrow U^H A\f$ update, plus a Gram phase and an SVD
epilogue. It skips converged pairs as `xGESVJ` does, and uses neither column equilibration
([scaling](#gesvdj_cta-global-power-of-two-scaling)) nor de Rijk ordering
([rejected](../perf/gesvd.md#gesvdj_cta-tier-2-preconditioning-tested-and-rejected)).

## gesvd design: why bdsdc goes through Golub-Kahan

`bdsdc` is not a port of `dlasd*`. For upper bidiagonal B the matrix
\f$T = \begin{pmatrix} 0 & B^T \\ B & 0 \end{pmatrix}\f$ has eigenvalues \f$\pm\sigma_i\f$; under
the perfect shuffle it is a zero-diagonal tridiagonal of order 2n with off-diagonal
\f$(d_0, e_0, d_1, \dots, d_{n-1})\f$, which the batched `stedc` solves without squaring anything.
That reuses existing device code instead of ~2000 lines of bidiagonal divide and conquer, and is
about 1000x faster than the sequential `bdsqr`
([table](../perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver)).

- **Extraction is exact:** the \f$+\sigma\f$ eigenvector is \f$(v; u)/\sqrt{2}\f$ interleaved. A
  sign flip on u when the \f$\pm\sigma\f$ vectors mix costs at most \f$2\sigma\f$ in the residual.
- **Null space:** repeated zero or unresolvable \f$\sigma\f$ collapse eigenvectors.
  `bdsdc_repair_degenerate` rebuilds every column with \f$\sigma_i < 500\,\varepsilon\,\sigma_{\max}\f$
  by Gram–Schmidt (no factor of n;
  [rejected criteria](../perf/gesvd.md#gesvd-bdsdc-null-space-repair-criteria-that-failed)).
  `tests/bdsdc_tests.cc` covers graded spectra, rank deficiency 1-5 and the zero matrix.
- `BATCHLAS_GESVD_BIDIAG` (`gesvd_bidiag_solver()`): unset or unknown is `bdsdc`; `bdsqr` keeps
  relative accuracy for tiny \f$\sigma\f$ at 300-400x the cost; `normal` squares \f$\kappa\f$ and is
  overridden by a direct solve when U is thin and tall.

## gesvdj_cta: the lane-equals-row mapping decision

Lane r owns row r of A and V, so both operands of a rotation sit in one lane: no partner
shuffles, no cross-lane hazard in the update, and the converged-pair skip (`if (sk == 0)
continue;`) is warp-uniform. Keeping
\f$G = A^H A\f$ resident by congruence was rejected: its error is normwise
(\f$\varepsilon\sigma_{\max}^2\f$), which turns the relative test into an absolute one.

## gesvdj_cta: local-memory budget formula

Partition width \f$P = \min(C, 32)\f$ lanes; tile capacity C is keyed on \f$\max(m,n)\f$: 4, 8, 16,
32, 64 (the 64 rung has 2 rows per lane). `gesvdj_slm_bytes<T>(C, want_vectors)` in
`gesvd_native.hh` is the formula: per problem, A (and V) as \f$(C+1) \times C\f$ tiles, column
norms, rotation cosines and sines (plus a phase for complex), an inverse permutation; per work-group,
an `int16` pair table. `gesvd_jacobi_max_dim` returns the widest C whose smallest work-group fits
the device budget, and both `can_run` and the kernel read it. On the RTX 4090 (101,376 B)
`complex<double>` with vectors needs 138,816 B at C = 64 and is capped at 32.

- The pair table packs `p | (q << 8)` (`static_assert(C <= 64)`); it breaks at \f$C \ge 128\f$.
- U is never held in local memory (A + V matches syev's occupancy).
- LD = C + 1 is odd: the \f$V^H\f$ writeback reads `V_local[r + c_i*LD]` with \f$c_i\f$ a
  permutation, and LD = 32 serialises 32 ways.
- Default `cta_wg_size_multiplier = 1`: at n = 32, batch 16384, float, 1 problem per work-group
  took 7.67 ms, 2 took 8.85 ms.

## gesvdj_cta: the reduce-scatter G3 trap

The Gram phase scatters V = P/2 dot products over L = P lanes (pair k to lanes 2k, 2k+1). That is
\f$\log_2 V\f$ scatter steps plus \f$\log_2(L/V)\f$ all-reduce steps: 4 + 1 at P = 32, not five
halving steps. Written as five halvings, the fifth has `half == 0`, the loop never runs, and each
dot product sums 16 of 32 rows (`max(1, half)` reads a stale `g[1]`). The exact-norm reduction
(32 values over 32 lanes) is 5 + 0. The tree sum keeps the error at \f$\log_2(m)\,\varepsilon\f$; a
lane-sequential sum (\f$m\varepsilon\f$) churns the threshold. At C = 64 a round is two chunks of
16 pairs on disjoint columns.

## gesvdj_cta: global power-of-two scaling

Per-column equilibration computes the SVD of \f$AD\f$, which does not factor A. The kernel applies
one power-of-two rescale \f$\beta = 2^{-e}\f$ centred on the geometric mean,
\f$e = \operatorname{round}(\tfrac14(\log_2 \max_c\|A_c\|^2 + \log_2 \min_{\|A_c\|>0}\|A_c\|^2))\f$:
exact round trip, and for float a tolerated column-norm ratio of \f$1.7\times10^{38}\f$ against
\f$9.2\times10^{18}\f$ when centred on the maximum. Zero input keeps \f$\beta = 1\f$. The norm
epilogue uses a scaled sum of squares; otherwise an underflowing norm gives \f$\sigma = 0\f$ and a
fabricated U column. Only the first P = 32 columns feed the max/min, so on the C = 64 rung
columns 32..63 get no overflow headroom (read from code;
[unverified candidate](known-defects.md#unverified-candidates)).

## gesvdj_cta: sweep, convergence and extraction rules

- `tol = tol_multiplier * CC * eps`, `tau_big = 1/sqrt(eps)`; rotation formulas match
  `syev_jacobi_cta`. A rotation with s == 0 is inactive, or every problem runs all sweeps (30x).
- The norm recurrence clamps both sides at zero (one side let the other go negative: NaN, then an
  out-of-bounds access at \f$\kappa \ge 10^4\f$). Exact norms are recomputed every sweep.
- Termination needs two consecutive zero-rotation sweeps; `info` tests exactly that predicate.
- \f$\sigma_i\f$ is a final exact column norm times \f$1/\beta\f$; reading `Nrm_local` reintroduces
  the normal-equations defect.
- Sort descending with a rank sort that breaks ties on index; do not also reverse.
- In the \f$V^H\f$ writeback the lane is the output row, so the Thin bound applies to `lane`, not to
  the inner loop (truncating the loop gives a wrongly shaped factor with plausible numbers).

## gesvdj_cta: rank deficiency, Thin and m < n

The zero threshold is \f$\mathtt{zero\_sigma\_multiplier}\cdot\varepsilon\cdot\sigma_{\max}\f$;
\f$\varepsilon\max(1, \sigma_{\max})\f$ zeroes everything at \f$\sigma_{\max} = 10^{-10}\f$. Columns
CC..RR-1 of a full U, and columns below the threshold, are completed in-kernel by two passes of
classical Gram–Schmidt with a trial cursor across output columns (accept above \f$1/(2 RR)\f$; a
per-column restart leaves the last columns unfilled). `SvdVectors::Thin` skips completion but
still repairs deficient columns. m < n solves \f$A^H\f$ transposed at load; \f$A^T\f$ would be wrong
for complex.

## gesvdj_cta: traps and dead ends

- `gesvdj_cta.cc` must stay in `EXTENSIONS_CTA_SOURCES` (`src/ops/CMakeLists.txt`, configured
  `NO_CPU_TARGETS`); elsewhere it silently gets a CPU AOT image.
- `GesvdjParams` is its own struct: `JacobiParams`' ascending `sort_order` default invites a silent
  reversal.
- Live registers are `ap + aq + g`, 48 T at C = 32 and 80 at C = 64; check `-Xcuda-ptxas -v`. Index
  `pk`, `qk`, `g`, `ap`, `aq` only at compile time, or the array spills.
- The pair-table stride is `kRotSlots`, not `pairs_per_round` (they differ when CC < P), and its fill
  ends in a barrier, so it precedes the `prob_id >= nb` early return.
- `[[sycl::reqd_sub_group_size(32)]]` is required. Off NVPTX, `group_barrier` on a
  `SubGroupPartition` is a no-op and correctness rests on sub-group lockstep.
- Do not fully unroll the round loop (about 6k instructions against a 32 KB I-cache).
- A is destroyed; benchmarks wrap it in `bench::pristine`.

## gesvd design: why SvdVectors::Thin exists

`SvdVectors` follows LAPACK `jobu`/`jobvt`: `None`, `All` ('A'), `Thin` ('S'). For tall-skinny input
All is unusable (a 10000 x 32 float problem needs a 400 MB U). Thin equals All for U when
\f$m \le n\f$ and for \f$V^H\f$ when \f$m \ge n\f$; entry points canonicalise with `canonical_jobu` /
`canonical_jobvh`, and a `*_buffer_size` must canonicalise exactly as its run path does. LAPACK 'O'
is absent; if added, append the enumerator so ordinals stay stable.

## gesvd design: vendor binding and dispatch

- `include/batchlas/blas/functions/gesvd.hh` declares `backend::gesvd_vendor` only; each backend TU
  (`cusolver.cc`, `rocsolver.cc`, `netlib_lapack.cc`) defines it. A header definition would collide.
- `launch` and `workspace` in `gesvd.cc` are `std::visit`s over `GesvdChoice`.
- `can_run` (correctness only) requires a homogeneous, unmasked batch and, for natives, a GPU.
  `jacobi`: not Hermitian, sub-group 32, \f$\max(m,n) \le\f$ `gesvd_jacobi_max_dim`. `cta`:
  sub-group 32, \f$\max(m,n) \le 32\f$, not thin, real general or square Hermitian. `blocked`: general
  or square Hermitian. `vendor`: on CUDA \f$\max(m,n) \le 32\f$, packed, not thin, vectors or square;
  never on ROCm.
- A pin `can_run` refuses throws (`BATCHLAS_GESVD_ROUTE=jacobi` at 128x128); only `native` and
  `vendor` fall back to auto.
- Order lives in the tables; Jacobi ranks ahead of `cta` on accuracy
  ([evidence](../perf/gesvd.md#gesvd-promoting-jacobi-to-the-default-at-n-up-to-32)) and stays
  admissible to 64 so a pin reaches 33..64
  ([evidence](../perf/gesvd.md#gesvd-what-routes-today)). Where a table ranks `cta` first for real
  input with vectors,
  [known defect 18](known-defects.md#defect-18-gesvd_cta-left-vectors-lose-orthogonality-as-eps-times-kappa-squared)
  applies.

## gesvd design: test and benchmark traps

- A test binary needs both `TEST_TARGETS` and `BATCHLAS_TEST_LABELS_eig` in `tests/CMakeLists.txt`,
  or `ctest -L eig` drops it silently.
- Fill host-written test matrices with `Matrix<T,F>(n, n, batch)`; `Matrix::Zeros` and
  `Matrix::Diagonal` race with the host writes.
- Pin or assert the route (`BATCHLAS_GESVD_ROUTE`, `select::ScopedPin`) on an observable only that
  path produces, e.g. \f$\|U^H U - I\|\f$ at \f$\kappa = 10^5\f$, and check with `EXPECT_VERIFY`
  (@ref design_verification).
- minibench positional arguments replace the whole sweep and missing ones read as 0: pass all four
  (`32 16384 1 1`).
