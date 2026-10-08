# gesvd design {#design_gesvd}

> **Status:** current.

Batched SVD uses one-sided Jacobi (`gesvdj_cta`) up to n = 64 (complex\<double\> with vectors up
to 32). Above that it bidiagonalises with `gebrd`, solves the bidiagonal problem with `bdsdc` on
the batched `stedc`, and back-transforms with `ormbr`. This page gives the design rules, the
kernel's local-memory and scaling rules, and the traps. Measurements are in @ref perf_gesvd.

## gesvd design: why one-sided Jacobi

The accuracy target is cuSOLVER `gesvdjBatched`, which gives relative accuracy (Demmel–Veselić):

\f[ \frac{|\delta\sigma_i|}{\sigma_i} \le O(\varepsilon)\,\kappa(A_c), \f]

where \f$A_c\f$ is the column-equilibrated matrix. Paths that form the tridiagonal of
\f$B^T B\f$ take \f$\sigma = \sqrt{\lambda}\f$ and square the condition number (see
[the normal-equations defect](../perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa)).

Hestenes one-sided Jacobi applies 2x2 rotations to column pairs of A until the columns are
orthogonal. Then \f$\sigma_i = \|A_i\|\f$, \f$U_i = A_i / \sigma_i\f$, and V is the accumulated
rotation. Each pair's Gram is recomputed from the current columns, so the test is relative:

\f[ |a_{pq}| > \mathrm{tol}\cdot\sqrt{|a_{pp}|\,|a_{qq}|}, \qquad \mathrm{tol} = t\cdot n\cdot\varepsilon \f]

An absolute test \f$|a_{pq}| \le \mathrm{tol}\cdot\max|a_{kl}|\f$ loses this accuracy (LAWN 169,
Remark 2.2).

Why Jacobi here:

- **No back-transform.** U and V come out of the iteration.
- **One kernel.** The former CTA path needed five or more launches with global round trips.
- **Dial.** `max_sweeps` and `tol_multiplier` (`GesvdjParams`).
- **Reuse.** `gesvdj_cta` is `syev_jacobi_cta` without its Phase 2 (the \f$A \leftarrow U^H A\f$
  update), plus a Gram phase, column norms and an SVD epilogue.

The kernel keeps the relative stopping test and skips converged pairs from LAPACK `xGESVJ`. It
does not use column equilibration ([scaling](#gesvdj_cta-global-power-of-two-scaling)) or de Rijk
ordering ([evidence](../perf/gesvd.md#gesvdj_cta-tier-2-preconditioning-tested-and-rejected)).

References: Hestenes (1958); Demmel & Veselić, SIAM J. Matrix Anal. Appl. 13(4) (1992); Drmač &
Veselić, LAWN 169/170; Golub & Van Loan, Alg. 8.5.1.

## gesvd design: why bdsdc goes through Golub-Kahan

`bdsdc` is not a port of LAPACK `dlasd*`. For B upper bidiagonal with diagonal d and
superdiagonal e,

\f[ T = \begin{pmatrix} 0 & B^T \\ B & 0 \end{pmatrix} \f]

has eigenvalues \f$\pm\sigma_i\f$. Under the perfect-shuffle permutation it is a symmetric
tridiagonal matrix of order 2n with zero diagonal and off-diagonal
\f$(d_0, e_0, d_1, e_1, \dots, d_{n-1})\f$, which the batched `stedc` solves. Nothing is squared.
A native bidiagonal divide and conquer would be about 2000 lines of new device code. Solving the
doubled problem with `stedc` is about 1000x faster than the sequential `bdsqr`
([table](../perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver)).

**Extraction is exact.** For \f$\sigma \ne 0\f$ the \f$+\sigma\f$ eigenvector is \f$(v; u)/\sqrt{2}\f$
interleaved, so even and odd rows give v and u. The only residue is a possible sign flip on u
when the \f$\pm\sigma\f$ vectors mix by more than 45°. It costs at most \f$2\sigma\f$ in the
residual, and only when \f$\sigma \approx \varepsilon\|B\|\f$.

**Null space.** Repeated zero singular values, or a \f$\pm\sigma\f$ pair too close to resolve, make
eigenvectors collapse onto the same (v, u). `bdsdc_repair_degenerate` rebuilds each column with
\f$\sigma_i < 500\,\varepsilon\,\sigma_{\max}\f$ by Gram–Schmidt against the final columns. Any
orthonormal completion satisfies \f$B = U S V^T\f$ to within \f$2\sigma\f$. The threshold has no
factor of n. The rejected criteria are in the
[perf page](../perf/gesvd.md#gesvd-bdsdc-null-space-repair-criteria-that-failed).

`bdsdc` writes the vectors it computes, so \f$U^T U = I\f$ is a property of the method.
`tests/bdsdc_tests.cc` checks it on graded spectra, a hostile gebrd-like generator, rank
deficiency of multiplicity 1-5, and the all-zero matrix. `bdsqr` stays behind
`BATCHLAS_GESVD_BIDIAG=bdsqr`: it keeps relative accuracy for tiny \f$\sigma\f$ at 300-400x the cost.

## gesvdj_cta: the lane-equals-row mapping decision

Each round has n/2 disjoint pairs, and each needs three length-m dot products before its rotation
is known. Three mappings were compared at m = n = 32, P = 32, float, with vectors. These counts are
projected, not measured. `syev_jacobi_cta` is 266.

| design | per round | vs syev |
|---|---|---|
| lane = row, reduce-scatter Gram (chosen) | ~181 | 0.68 |
| lane = column | ~251-264 | 0.94-1.00 |
| Gram-resident (\f$G = A^H A\f$ kept by congruence) | 334 + refresh | 1.26-1.39 |

Lane = row wins because both operands of a rotation sit in one lane, so no partner shuffles are
needed. Converged pairs are free: the skip `if (sk == 0) continue;` is warp-uniform, which lane =
column would not be. Lane r alone reads and writes row r of A and V, so the update has no
cross-lane hazard.

The Gram-resident design is rejected. Congruence updates give normwise error
\f$\sim\varepsilon\sigma_{\max}^2\f$ rather than columnwise error, so the relative threshold
degrades to an absolute test. It also needs 51,200 B per `complex<double>` problem, above the
49,152 B the design assumed.

## gesvdj_cta: local-memory budget formula

\f$P\f$ is the partition width (lanes, at most 32, the sub-group width). \f$C\f$ is the tile capacity.
For every rung up to 32, \f$P = C\f$. The 64 rung is \f$(P, C) = (32, 64)\f$, with `kRPL = C/P` rows
per lane. The rung is keyed on \f$\max(m,n)\f$: 4, 8, 16, 32, then (32, 64).

```cpp
LD            = C + 1;                   // odd
kTileElems    = LD * C;                  // 1056 at C=32
kRotSlots     = C / 2;                   // 16 at C=32
kPairSlots    = (C - 1) * kRotSlots;     // 496 at C=32
bytes_per_prob = (1 + ComputeV) * kTileElems * sizeof(T)      // A (+ V)
               + C             * sizeof(Real)                 // Nrm_local
               + 2 * kRotSlots * sizeof(Real)                 // Rcs_local (vec<Real,2>)
               + (complex ? kRotSlots * sizeof(T) : 0)        // Rd_local, the phase
               + C             * sizeof(int16_t);             // Inv_local
wg_fixed       = kPairSlots * sizeof(int16_t);                // pair table, 992 B at C=32
```

Bytes per problem with V resident, plus the work-group pair table:

| T | C = 32 | C = 64 (per problem + table) |
|---|---|---|
| float | 8,768 (+992) | 33,920 + 4,032 = 37,952 |
| double | 17,472 (+992) | 67,712 + 4,032 = 71,744 |
| complex\<float\> | 17,344 (+992) | 67,712 + 4,032 = 71,744 |
| complex\<double\> | 34,624 (+992) | 134,784 + 4,032 = 138,816 |

The device reports 101,376 B of local memory. `complex<double>` with vectors therefore does not
launch at C=64 and is capped at 32. `gesvd_jacobi_max_dim` (`src/extensions/gesvd_native.hh`) sets
the cap, and both the kernel guard `gesvdj_cta_max_dim` and `can_run` read it. Other types drop to
1-2 work-groups per SM, against 10 at C=32. Occupancy, not the cap, is the binding limit.

- The pair table is `int16` `p | (q << 8)`, with `static_assert(C <= 64)`. The packing breaks at
  \f$C \ge 128\f$.
- **U is never held in local memory.** A + V matches syev's occupancy. Holding A, U and V would
  cost about 20%.
- **LD is odd.** Hot accesses are conflict-free for any LD. The V^H writeback reads
  `V_local[r + c_i*LD]` with \f$c_i\f$ a permutation, which needs \f$\gcd(LD, 32) = 1\f$. LD = 32
  serialises 32 ways.
- **Clamp per work-group** against `local_mem_size - wg_fixed`, as a multiple of `32/P`, and against
  `max_work_group_size`. The `syev_jacobi_cta` clamp under-counts by `32/P`.
- **Default `cta_wg_size_multiplier = 1`.** At n=32, batch=16384, float: 7.67 ms with 1 problem per
  work-group, 8.85 ms with 2, 9.01 ms with 4.
- `complex<double>` with vectors runs at about 2 warps per SM. This was accepted, not engineered
  around.

## gesvdj_cta: the reduce-scatter G3 trap

The Gram phase computes `kGramChunk = P/2` dot products per chunk and scatters them so that pair k
lands in lanes 2k and 2k+1. **Scattering V values over L lanes takes \f$\log_2 V\f$ scatter steps
plus \f$\log_2(L/V)\f$ all-reduce steps.** At P=32, V=16 that is 4 + 1, not 5 + 0:

```cpp
for (step = 0; step < 4; ++step) {        // masks 16, 8, 4, 2
    mask = kGramChunk >> step; hi = lane & mask; half = mask / 2;
    for (j = 0; j < half; ++j) {
        own  = hi ? g[j + half] : g[j];
        send = hi ? g[j]        : g[j + half];
        g[j] = own + permute_group_by_xor(part, send, mask);
    }
}
g[0] = g[0] + permute_group_by_xor(part, g[0], 1);   // the final ALL-REDUCE
```

Written as five halving steps, `half` is 0 at the fifth step, the loop never runs, and each dot
product sums only 16 of the 32 rows. `max(1, half)` reads a stale `g[1]`. The exact-norm reduction
(32 values over 32 lanes) is 5 + 0 and correct.

The cost is 16 shuffles for all sixteen dot products, against 80 for independent butterflies. The
tree sum's error is \f$\sim\log_2(m)\,\varepsilon\f$, against \f$\mathrm{tol} = n\varepsilon\f$.
A lane-sequential sum would give \f$\sim m\varepsilon\f$, which causes threshold churn.

At C=64 a round runs as two chunks of 16 pairs. The chunks touch disjoint columns, and the live
register set is 80 T rather than 160. The exact-norm reduction stays 32 lanes wide, over C/P passes.

## gesvdj_cta: global power-of-two scaling

Per-column equilibration is wrong as an algorithm step. Scaling columns by D computes the SVD of
\f$AD\f$, and \f$D^{-1}W\f$ is not orthogonal, so the result does not factor A. \f$\kappa(A_c)\f$ is an
analysis quantity only.

The kernel applies one global power-of-two rescale centred on the geometric mean of the column
norms:

\f[ \beta = 2^{-e}, \qquad e = \operatorname{round}\!\Big(\tfrac14\big(\log_2 \max_c\|A_c\|^2 + \log_2 \min_{c,\ \|A_c\| > 0}\|A_c\|^2\big)\Big). \f]

A power of two keeps the round trip exact. For float, geometric centring tolerates a column-norm
ratio of \f$1.69\times10^{38}\f$, against \f$9.2\times10^{18}\f$ for centring on the maximum. An all-zero
input keeps \f$\beta = 1\f$. The column-norm epilogue uses a scaled sum of squares. Without it, an
underflowing norm gives \f$\sigma = 0\f$ and fabricates a U column.

> **Note:** the \f$\max\f$/\f$\min\f$ reductions read `Nrm_local` only for the first 32 columns. On the
> C=64 rung, columns 32..63 do not affect \f$\beta\f$. Correctness holds, but the overflow headroom
> covers only those 32 columns. Observed, not measured.

## gesvdj_cta: sweep, convergence and extraction rules

- **Threshold:** `tol = tol_multiplier * CC * eps`, `tau_big = 1/sqrt(eps)`. Rotation formulas
  match `syev_jacobi_cta`.
- **A rotation with s == 0 is inactive.** Otherwise the rotation count never reaches 0, and every
  problem runs all `max_sweeps`: a 30x slowdown.
- **Norm recurrence** between recomputations. Both sides are clamped at zero. Clamping one side let
  the other go negative, `sqrt` gave NaN, and an out-of-bounds local access followed at
  \f$\kappa \ge 10^4\f$.
- **Exact norms** are recomputed at the start of every sweep. Termination requires **two consecutive**
  zero-rotation sweeps (`zero_sweeps >= 2`), and `info` tests exactly that predicate.
- **\f$\sigma\f$ comes from A**: a final exact column norm times \f$1/\beta\f$. Reading it from
  `Nrm_local` reintroduces the normal-equations defect.
- **Sort descending** with a parallel rank sort that breaks ties on index. Do not also reverse
  indices.
- **Two writebacks, opposite lane roles.** U: lane is the output row. V^H: lane is the output row
  and reads `V_local[r + c_i*LD]`. In the transposed orientation the Thin bound applies to `lane`,
  not the inner loop. Truncating the inner loop gives a wrongly shaped factor with plausible
  numbers.

## gesvdj_cta: rank deficiency, Thin and m < n

The zero threshold is relative to \f$\sigma_{\max}\f$ only:
\f$\mathrm{tol}_0 = \mathtt{zero\_sigma\_multiplier}\cdot\varepsilon\cdot\sigma_{\max}\f$. The
absolute form \f$\varepsilon\max(1, \sigma_{\max})\f$ declares every \f$\sigma\f$ zero when
\f$\sigma_{\max} = 10^{-10}\f$.

**Completion.** Columns CC..RR-1 of a full U, and any column below \f$\mathrm{tol}_0\f$, are filled
in-kernel by two passes of classical Gram–Schmidt against the accepted columns. The trial cursor
runs across output columns and accepts above \f$1/(2 \cdot RR)\f$. A per-column restart leaves the
last columns unfilled, because a canonical vector's residual is about \f$d/RR\f$ when d dimensions
remain. The branch is gated on a warp-uniform predicate.

**Thin** (`SvdVectors::Thin`) sets `left_cols = CC` and skips completion. A numerically deficient
column is still repaired.

**m < n solves \f$A^H\f$**, transposed at load time. Solving \f$A^T\f$ instead gives
\f$\overline{V'} S U'^T\f$, whose conjugations are invisible in real arithmetic and wrong for complex.

## gesvdj_cta: traps and dead ends

- **Build group.** `gesvdj_cta.cc` must stay in `EXTENSIONS_CTA_SOURCES`
  (`src/extensions/CMakeLists.txt`). Moving it gives the 32-lane kernel a CPU AOT image with no error.
- **`GesvdjParams` is its own struct.** Reusing `JacobiParams` would bring its ascending
  `sort_order` default, which invites a silent reversal.
- **Explicit instantiations** each need a `template ...;` line spelled with `BATCHLAS_UNPAREN fp`.
  A missing one is an undefined symbol at final link.
- **Register pressure.** `ap + aq + g` is 48 live T at C=32 and 80 at C=64. Check `-Xcuda-ptxas -v`
  before trusting a timing.
- **Compile-time indices only** into `pk`, `qk`, `g`, `ap`, `aq`. A runtime index spills the array.
- **Pair-table stride** is the compile-time `kRotSlots`, not `pairs_per_round`. They differ when
  CC < P. The fill ends in a barrier, so it must precede the `prob_id >= nb` early return.
- **`[[sycl::reqd_sub_group_size(32)]]` is required.** The butterflies are wrong at any other width.
- **Off CUDA, correctness rests on sub-group lockstep.** `group_barrier` on a `SubGroupPartition`
  is a no-op off NVPTX. The ROCm and NETLIB arms are the least tested.
- **Do not fully unroll the round loop** to keep V in registers. The code would be about 6k
  instructions against a 32 KB Ada I-cache.
- **Round-robin partner** is unused: under lane = row every lane needs all pairs.
- **A is destroyed**, per the gesvd contract. Benchmarks wrap it in `bench::pristine`.

## gesvd design: why SvdVectors::Thin exists

`SvdVectors` (`include/batchlas/blas/enums.hh`) follows LAPACK `jobu`/`jobvt`: `None`, `All`
('A') and `Thin` ('S'). For tall-skinny input All is unusable. A 10000 x 32 problem needs a
10000 x 10000 U, 400 MB per matrix in float.

For \f$k = \min(m, n)\f$, a thin U equals the full U when \f$m \le n\f$, and a thin \f$V^H\f$ equals the
full one when \f$m \ge n\f$. Entry points canonicalise Thin to All when the shapes coincide
(`canonical_jobu` / `canonical_jobvh`).

> **Warning:** a `*_buffer_size` and its run path must canonicalise identically, or the workspace
> is sized for a different computation.

LAPACK 'O' is absent. If added, append an enumerator so existing ordinals stay stable.

## gesvd design: vendor binding and dispatch

- **The header declares `backend::gesvd_vendor` only.** Each backend TU defines and instantiates
  it for its own `Backend`. A throwing definition in the header would collide with `cusolver.cc`.
- **One branch per family, checked by the compiler.** `launch` and `workspace` in
  `src/ops/gesvd/gesvd.cc` are `std::visit`s over `GesvdChoice`.
- **A pin that `can_run` refuses throws.** `BATCHLAS_GESVD_ROUTE=jacobi` on a 128x128 real matrix
  raises `std::invalid_argument` (`src/select/select.hh`). Only `native` and `vendor` fall back.
- **Order lives in the tables.** Jacobi is ahead of CTA on accuracy evidence
  ([evidence](../perf/gesvd.md#gesvd-promoting-jacobi-to-the-default-at-n-up-to-32)). The order is
  the rows of `tuned/gesvd.<dtype>.<arch>.txt`. Jacobi stays admissible to 64, so a pin still
  reaches the 33..64 band
  ([evidence](../perf/gesvd.md#gesvd-what-routes-today)).

## gesvd design: test and benchmark traps

- Register a test binary in both `TEST_TARGETS` and the `eig` label list in `tests/CMakeLists.txt`.
  A missing label drops it from `ctest -L eig` without an error.
- Fill test matrices with the plain `Matrix<T,F>(n, n, batch)` constructor when the host writes to
  them afterwards. `Matrix::Zeros` and `Matrix::Diagonal` race with those writes.
- A test through `gesvd` must assert the route taken, on an observable only that path produces.
  For example, \f$\|U^H U - I\|\f$ at \f$\kappa = 10^5\f$.
- minibench positional arguments replace the whole sweep. `state.range(i)` returns 0 when missing,
  so pass all four (`32 16384 1 1`).
