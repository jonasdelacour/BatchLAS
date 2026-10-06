# gesvd: design of the batched SVD {#design_gesvd}

**Covers:** why the batched SVD is one-sided Jacobi for small matrices and Golub–Kahan +
`stedc` for the bidiagonal above that, and the still-valid design rationale of the
`gesvdj_cta` kernel: the lane mapping, the local-memory budget, the reduce-scatter trap, the
scaling rule, the extraction and completion rules, and the traps recorded while building it.
**Status:** current. Each rule below was re-checked against `src/extensions/gesvdj_cta.cc`,
`src/extensions/bdsdc.cc` and `include/batchlas/blas/dispatch/route_gesvd.hh` in this tree;
where the shipped kernel departs from the reviewed design the section says so.
**Sources:** the root `GESVD_PLAN.md` (design sections 4.x) and `GESVD_IMPL_SPEC.md`
(Parts C and D). The impl spec was pinned to the since-merged `gesvd-batched-plan` worktree
at `c76a01f` plus uncommitted work, and was mostly a task list with `file:line` citations
into that snapshot; the task lists, build-order checklist (its Part F), test case list
(Part E) and commit bookkeeping were dropped here as done or stale, and its line numbers are
not carried over; the traps from Part E are kept in the last section. Both root files were
removed after this migration; the originals are `git show bdcdfa5c:GESVD_IMPL_SPEC.md` and
`git show bdcdfa5c:GESVD_PLAN.md`. Measurements are in @ref perf_gesvd.

## gesvd design: why one-sided Jacobi

The accuracy target is cuSOLVER `gesvdjBatched`, which is one-sided Jacobi and sells
**relative** accuracy, the Demmel–Veselić bound

\f[ \frac{|\delta\sigma_i|}{\sigma_i} \le O(\varepsilon)\,\kappa(A_c), \f]

where \f$A_c\f$ is the column-equilibrated matrix. The pre-existing native paths formed the
tridiagonal of \f$B^T B\f$ and took \f$\sigma = \sqrt{\lambda}\f$, which squares the condition
number (see [the normal-equations defect](../perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa)).

Hestenes one-sided Jacobi on \f$A\f$ (\f$m \times n\f$, \f$m \ge n\f$): repeatedly take a column
pair \f$(p,q)\f$, form the 2x2 Gram \f$a_{pp} = A_p^H A_p\f$, \f$a_{qq} = A_q^H A_q\f$,
\f$a_{pq} = A_p^H A_q\f$, and apply the Jacobi rotation to columns p and q of A. At convergence
the columns are orthogonal, \f$\sigma_i = \|A_i\|\f$, \f$U_i = A_i / \sigma_i\f$, and V is the
accumulated product of rotations.

The distinction from the normal-equations defect, stated precisely: the Gram entries are dot
products and do square magnitudes, but \f$\sigma_i\f$ is **never** the square root of a
difference of large numbers. It is a column norm of the rotated A, and the rotations are
orthogonal and applied to A itself. The Gram of a pair is recomputed fresh from the current
columns every time, so its rounding error is columnwise relative, which is what makes the
threshold

\f[ |a_{pq}| > \mathrm{tol}\cdot\sqrt{|a_{pp}|\,|a_{qq}|}, \qquad \mathrm{tol} = t\cdot n\cdot\varepsilon \f]

a genuinely relative test. The classical absolute test \f$|a_{pq}| \le \mathrm{tol}\cdot\max|a_{kl}|\f$
would forfeit the whole advantage (LAWN 169, Remark 2.2).

Secondary reasons that mattered on this workload:

- **No back-transform.** U and V come out of the iteration; there is no `ormbr`/`ormqr`.
  U costs Jacobi essentially nothing (it is the rotated, normalised A), while the old CTA
  path paid 17x at n=8 for it.
- **One kernel.** The old CTA path was `gebrd_cta` → tridiagonal build → `steqr_cta` →
  vector assembly → back-transform, five-plus launches with global round trips.
- **An accuracy/speed dial:** `max_sweeps` and `tol_multiplier` (`GesvdjParams`).
- **Reuse.** `syev_jacobi_cta` (two-sided Jacobi) already had the round-robin pair table in
  local memory, the \f$LD = P+1\f$ padding, multiple problems per work-group sized against
  local memory, packed rotation coefficients, the relative threshold, complex support, the
  denormal and `tau_big` guards. Structurally `gesvdj_cta` is `syev_jacobi_cta` with its
  Phase 2 (the \f$A \leftarrow U^H A\f$ row update) deleted, which is exactly the difference
  between two-sided and one-sided Jacobi and a third of the reference kernel's local-memory
  traffic, plus a Gram phase, column norms and an SVD epilogue.

Of the LAPACK `xGESVJ` / Drmač–Veselić accuracy features the plan listed, the relative
stopping test and skipping converged pairs are implemented. **Column equilibration was
refuted as an algorithm step** ([scaling](#gesvdj_cta-global-power-of-two-scaling)). **de Rijk ordering** was implemented,
measured to save nothing and cost 13% through register pressure, and removed
([evidence](../perf/gesvd.md#gesvdj_cta-tier-2-preconditioning-tested-and-rejected)).

References cited by the kernel: Hestenes, "Inversion of matrices by biorthogonalization"
(1958); Demmel & Veselić, SIAM J. Matrix Anal. Appl. 13(4) (1992); Drmač & Veselić, LAPACK
Working Notes 169/170 (threshold form, SVA recurrence, convergence test); Golub & Van Loan,
*Matrix Computations*, Alg. 8.5.1 (the 2x2 rotation).

## gesvd design: why bdsdc goes through Golub-Kahan

Above the Jacobi cap the Blocked path bidiagonalises (`gebrd`), solves the bidiagonal SVD,
and back-transforms (`ormbr`). The bidiagonal solver is `bdsdc`, and it is **not** a port of
LAPACK's `dlasd0/2/3/4/8`. For B upper bidiagonal with diagonal d and superdiagonal e,

\f[ T = \begin{pmatrix} 0 & B^T \\ B & 0 \end{pmatrix} \f]

has eigenvalues \f$\pm\sigma_i\f$, and under the perfect-shuffle permutation
\f$y = (v_0, u_0, v_1, u_1, \dots)\f$ it is tridiagonal with zero diagonal and off-diagonal
\f$(d_0, e_0, d_1, e_1, \dots, d_{n-1})\f$. So the order-n bidiagonal SVD is an order-2n
symmetric tridiagonal eigenproblem, which the batched, tuned `stedc` already solves. Nothing
is squared: T's eigenvalues are \f$\pm\sigma\f$, not \f$\sigma^2\f$.

**Why not a native bidiagonal divide and conquer.** It needs its own secular equation (the
factored \f$(d_j - \sigma)(d_j + \sigma)\f$ form), its own two-kind deflation and its own Löwner
vector recomputation: about 2000 lines of new device code with no reuse, to solve a problem
of half the size. Solving twice the size with the right algorithm beat solving the right size
with a sequential one (`bdsqr`) by ~1000x ([table](../perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver)).

**The extraction is exact.** For \f$\sigma \ne 0\f$, \f$T(a v; b u) = (b\sigma v; a\sigma u)\f$
forces \f$a = b\f$, so the eigenvector for \f$+\sigma\f$ is exactly \f$(v; u)/\sqrt{2}\f$
interleaved; splitting into even and odd rows and normalising each half recovers v and u.
It stays exact when \f$+\sigma\f$ and \f$-\sigma\f$ are numerically mixed (a mixed vector is still
\f$(\alpha v; \beta u)\f$). The only residue is a possible sign flip on u past 45° of mixing,
worth at most \f$2\sigma\f$ in the residual, reached only when \f$\sigma\f$ is at
\f$\varepsilon\|B\|\f$.

**Where it fails, and the repair.** The null space: with k zero singular values the whole
2k-dimensional \f$\mathrm{span}\{(v_j; 0), (0; u_j)\}\f$ is degenerate at once, and more
generally once the \f$\pm\sigma_i\f$ pair (gap \f$2\sigma_i\f$) is unresolvable its two eigenvectors
normalise to the same (v, u). `bdsdc_repair_degenerate` rebuilds every column with
\f$\sigma_i < 500\,\varepsilon\,\sigma_{\max}\f$ by Gram–Schmidt against the columns already
final, taking the best canonical axis directly (the residual of \f$e_c\f$ against an orthonormal
Q is \f$1 - \sum_t Q_{ct}^2\f$). Those vectors are arbitrary anyway: any orthonormal completion
satisfies \f$B = U S V^T\f$ to within \f$2\sigma\f$. The threshold carries no factor of n. The
two alternative criteria that failed, and the derivation of 500, are in
[the perf page](../perf/gesvd.md#gesvd-bdsdc-null-space-repair-criteria-that-failed).
Unlike `bdsqr`, which accumulates rotations from the identity, `bdsdc` writes vectors it
computed, so \f$U^T U = I\f$ is a claim about the method; `tests/bdsdc_tests.cc` checks it,
over values-only, vectors, graded spectra, the hostile gebrd-like generator (mixed signs, exact
zeros, six decades), rank deficiency of multiplicity 1-5, and the all-zero matrix.

`bdsqr` (zero-shift QR) is kept behind `BATCHLAS_GESVD_BIDIAG=bdsqr`: it keeps relative
accuracy for tiny \f$\sigma\f$ where divide and conquer does not, at 300-400x the cost.

## gesvdj_cta: the lane-equals-row mapping decision

Per round the n/2 disjoint pairs each need three length-m dot products before their rotation
is known; in `syev_jacobi_cta` the corresponding quantity is one local-memory read because the
matrix *is* the Gram. Three mappings were reviewed by three independent judges (performance,
numerical accuracy, implementability); all three ranked **lane = row, one warp per problem,
round-batched reduce-scatter Gram** first and all three disqualified the Gram-resident one.

Projected local-memory + shuffle operations per round at m=n=32, P=32, float, with vectors,
against `syev_jacobi_cta`'s 266 (a design-time count, not a measurement):

| design | per round | vs syev |
|---|---|---|
| lane = row, reduce-scatter Gram (chosen) | ~181 | 0.68 |
| lane = column | ~251-264 | 0.94-1.00 |
| Gram-resident (\f$G = A^H A\f$ kept by congruence) | 334 + refresh | 1.26-1.39 |

Why lane = row:

- Both operands of a rotation live in the same lane. Under lane = column the update re-fetches
  the partner element with one shuffle per matrix element, 64 per round for A and V, more
  than the traffic it saves.
- `ap`/`aq` loaded for the Gram stay live into the update, saving 32 local loads per round.
- **Converged pairs are genuinely free.** The skip `if (sk == 0) continue;` is warp-uniform
  (every lane reads the same `Rcs_local` slot). Under lane = column it is lane-varying, so a
  round with 3 of 16 pairs live still issues the full update predicated off. Jacobi's last
  sweeps are almost entirely partial rounds (projected 10-20% of wall clock).
- The global load `A(lane, c)` is coalesced; the U writeback `U(lane, dst)` is too.
- Under lane = row, lane r exclusively reads and writes row r of A and V, so the update phase
  has no cross-lane local-memory hazard at all. The only hand-offs are `Rcs_local`/`Rd_local`
  and `Nrm_local`, ordered by barriers.

**Why the Gram-resident design is rejected outright.** Maintaining \f$G\f$ by congruence
accumulates *normwise* error \f$\sim\varepsilon\sigma_{\max}^2\f$, not the columnwise-relative
error of a fresh Gram. The relative threshold then degenerates into an absolute test for
every pair with \f$\sigma_p\sigma_q \ll \sigma_{\max}^2\f$, which is the normal-equations defect
one level removed. Independently, `complex<double>` with vectors needed 51,200 B per problem
plus the pair table, a hard launch failure at the 49,152 B the design assumed.

## gesvdj_cta: local-memory budget formula

`P` is the partition width (lanes, ≤ 32, the sub-group width); `C` is the tile capacity
(rows and columns of the resident matrix). On every rung up to 32 they are equal; the
64 rung is \f$(P, C) = (32, 64)\f$ and each lane owns `kRPL = C/P` rows. Splitting them
(`53c28ad`, `8bb04f2`) kept the \f$\le 32\f$ path textually identical. Ladder, keyed on
\f$\max(m,n)\f$ because lane = row needs the rows to fit: 4, 8, 16, 32, then (32, 64).

As shipped (`gesvdj_cta_impl`):

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

Recomputed per problem with V resident (plus the work-group-fixed pair table):

| T | C = 32 | C = 64 (per problem + table) |
|---|---|---|
| float | 8,768 (+992) | 33,920 + 4,032 = 37,952 |
| double | 17,472 (+992) | 67,712 + 4,032 = 71,744 |
| complex\<float\> | 17,344 (+992) | 67,712 + 4,032 = 71,744 |
| complex\<double\> | 34,624 (+992) | 134,784 + 4,032 = 138,816 |

Values-only drops one `kTileElems * sizeof(T)` term. The device reports 101,376 B
(`8bb04f2`; the design assumed 49,152), so `complex<double>` with vectors does not launch at
C=64 and is capped at 32 (`gesvdj_cta_max_dim`, mirrored in `gesvd_jacobi_max_dim` in
`route_gesvd.hh`; the two must agree). The others fall to 2 or 1 work-groups per SM against
10 at C=32; occupancy, not the hard cap, is the binding constraint.

Departures from the reviewed spec, all deliberate:

- The spec's pair table was `uint32` with four `uint8` fields; the shipped one is `int16`
  `p | (q << 8)`, with a `static_assert(C <= 64)` because that packing has one bit of
  headroom and breaks at \f$C \ge 128\f$.
- The spec kept both `Rank_local` and `Inv_local`; only the inverse permutation is stored.
- The design check that mattered still holds: **U is never materialised in local memory.**
  A + V replaces syev's A + Z one tile for one, so occupancy of the 2.11 ms `syev_jacobi_cta`
  reference carries over; holding A, U and V would have cost ~20% occupancy.

**Why LD is odd.** Under lane = row every hot access is `base + lane + c*LD` with lanes
differing by 1, conflict-free at any LD. The pad is needed by one phase, the \f$V^H\f$
writeback, where lane i reads `V_local[r + c_i*LD]` with \f$c_i\f$ a permutation. Banks are
\f$(r + c_i \cdot LD) \bmod 32\f$; \f$\gcd(LD, 32) = 1\f$ makes \f$c \mapsto c \cdot LD \bmod 32\f$ a
bijection, whereas LD = 32 serialises 32 ways.

**Clamp problems per work-group directly.** `syev_jacobi_cta` clamps its multiplier, but usage
is `probs_per_wg * bytes_per_prob` with `probs_per_wg = multiplier * 32/P`, so the clamp
under-counts by 32/P (8x at P=4). Harmless there with one small tile; a launch failure with
two. `gesvdj_cta` clamps `probs_per_wg` against `local_mem_size - wg_fixed`, keeps it a
multiple of `32/P`, and also against `max_work_group_size`.

**Default `cta_wg_size_multiplier = 1`.** Design-time projection at P=32 float: 1 problem →
10 blocks → 10 warps/SM; 2 → 5 blocks → 10 warps; 4 → 2 blocks → 8 warps (worse); double 1 →
5 warps, 2 → 4. Measured: 1 → 7.67 ms, 2 → 8.85, 4 → 9.01 (n=32, batch=16384, float). The
multiplier is a knob, not a default. `local_mem_size` bounds the per-block request;
occupancy is set by the SM's shared pool, which SYCL does not report. Verify warps/SM with
`ncu` rather than assuming: the spec's critical-path analysis put the issue-bound/latency-bound
transition at ~3 warps per sub-partition, between the candidate answers.

**`complex<double>` with vectors is the known cliff even at C=32:** ~35.6 KB per problem plus
table → 2 blocks/SM → 2 warps/SM (design-time count, not measured). At 1/64 FP64 rate with
4x complex multiply cost two warps roughly saturate the FP64 pipe, so it was accepted rather
than engineered around.

## gesvdj_cta: the reduce-scatter G3 trap

The Gram phase computes `kGramChunk = P/2` dot products per chunk (16 at P=32) and scatters
them across P lanes so that pair k lands in lanes 2k and 2k+1. **Scattering V values over L
lanes needs \f$\log_2 V\f$ scatter steps plus \f$\log_2(L/V)\f$ all-reduce steps.** At P=32,
V=16 that is 4 + 1, not 5 + 0:

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

Written as five halving steps with `half = (16 >> step) / 2`, `half` is 0 at the fifth step,
the inner loop never runs, and every dot product is silently summed over only 16 of the 32
rows. `max(1, half)` is not a fix: at that step it makes the `hi` lane read a stale `g[1]`.
The design review's accuracy judge found this before the kernel was written. The exact-norm
reduction (32 values over 32 lanes) is 5 + 0 and correct as written.

Cost: 8+4+2+1+1 = 16 shuffles for all sixteen dot products, against 80 for sixteen
independent butterflies (complex: two interleaved reduce-scatters, 32). The tree sum gives
dot-product error \f$\sim\log_2(m)\,\varepsilon\f$ against \f$\mathrm{tol} = n\varepsilon\f$; a
lane-sequential sum would give \f$\sim m\varepsilon\f$, the same order as the threshold, showing
up as threshold churn and an inflated sweep count.

At C=64 a round has 32 pairs and runs as two chunks. Chunking is safe because a round's
pairs are a perfect matching (chunks touch disjoint columns, so neither the Gram/apply of
one chunk nor the `Nrm_local` writes can disturb another), and it holds the register arrays
at 80 live T (`ap[16][2] + aq[16][2] + g[16]`) instead of the 160 a whole C=64 round would
need, against 48 at C=32. Keeping the chunk at `P/2` is also what leaves the reduce-scatter,
the `k_of_lane = lane >> 1` mapping and the even-lane guards textually unchanged from the
C=32 kernel.

The exact-norm reduction likewise keeps its accumulator at `Real x[P]`, not `Real x[C]`, on
the C=64 rung. Widening it would cost 64 `Real` registers per lane and a sixth reduction
step; the hard-coded 5 steps stay correct because the *lane* count is still 32. The C
columns are instead covered in C/P passes of the unchanged 32-wide reduce-scatter, each lane
first summing its kRPL rows into `x[c]`. The same "sum own rows first, then a 32-wide
butterfly" rule applies to every dot product in the completion path.

## gesvdj_cta: global power-of-two scaling

**Per-column equilibration (the plan's item 4.4.1) is wrong as an algorithm step.** Scaling
columns by D computes the SVD of \f$AD = U\Sigma_c W^H\f$, so \f$A = U \Sigma_c (D^{-1}W)^H\f$ and
\f$D^{-1}W\f$ is not orthogonal: the result is not a factorisation of A. \f$\kappa(A_c)\f$ is an
*analysis* quantity, delivered because the rotation and threshold formulas are already
per-pair scale invariant. LAPACK `?GESVJ` likewise scales only by a scalar. All three
candidate designs and two of three judges refuted it independently.

What is admissible, and shipped, is a single global power-of-two rescale centred on the
geometric mean of the column norms:

\f[ \beta = 2^{-e}, \qquad e = \operatorname{round}\!\Big(\tfrac14\big(\log_2 \max_c\|A_c\|^2 + \log_2 \min_{c,\ \|A_c\| > 0}\|A_c\|^2\big)\Big). \f]

A power of two makes \f$\beta \cdot (1/\beta) = 1\f$ exactly, so the round trip is lossless.
Centring on the geometric mean rather than the maximum: in float the norms must satisfy
\f$\max < 1.84\times10^{19}\f$ (so \f$\max^2\f$ is finite) and \f$\min > 1.09\times10^{-19}\f$ (so
\f$\min^2\f$ is normal); centring on the max tolerates a column-norm ratio of \f$9.2\times10^{18}\f$,
geometric centring \f$1.69\times10^{38}\f$, for one extra min-reduction. Graded matrices, the
class the accuracy argument is about, are where that ratio is large. An identically zero
input keeps \f$\beta = 1\f$.

The spec also asked for a scaled sum-of-squares guard in the column-norm epilogue: without
it a column whose norm underflows in the squaring gives \f$\sigma = 0\f$, trips the
rank-deficiency path and fabricates a U column.

*Observed while verifying (not measured):* the \f$\max\f$/\f$\min\f$ reductions read
`Nrm_local[base_n + lane]` for `lane < CC`, i.e. only the first P = 32 columns. On the
C=64 rung columns 32..63 do not influence \f$\beta\f$. Correctness is unaffected (any power of
two is exact); the overflow headroom argument above holds only for the first 32 columns.

## gesvdj_cta: sweep, convergence and extraction rules

- **Relative threshold** with `tol = tol_multiplier * CC * eps`, `tiny = numeric_limits::min()`,
  `tau_big = 1/sqrt(eps)`, rotation formulas byte-for-byte from `syev_jacobi_cta`.
- **A rotation with s == 0 is inactive.** Otherwise a rotation that rounds to the identity is
  counted, the rotation count never reaches 0, and every problem burns all `max_sweeps`: a
  silent 30x slowdown.
- **Analytic norm recurrence** between exact recomputations:
  \f$a_{pp}' = a_{pp} - t\,a_{pq}\f$, \f$a_{qq}' = a_{qq} + t\,a_{pq}\f$, exact in exact arithmetic
  because \f$t^2 + 2\tau t - 1 = 0\f$ (LAPACK `?GESVJ` maintains SVA this way). **Both** sides are
  clamped at zero: clamping only the cancelling p side let the q side go negative, sqrt gave
  NaN, every rank comparison against NaN was false, two columns got the same rank and the
  unwritten permutation slot was read as a column index, an out-of-bounds local access seen as
  `CUDA_ERROR_ILLEGAL_ADDRESS` at \f$\kappa \ge 10^4\f$.
- **Drift and the verification sweep.** A drifted (inflated) \f$a_{pp}\f$ raises the threshold,
  a genuinely non-negligible \f$a_{pq}\f$ is skipped, the loop exits, and \f$\sigma\f$ is a column
  norm of a non-converged A: an output error, not a schedule error. The spec proposed a
  mid-sweep refresh when a norm shrinks by more than `drift_refresh_ratio`, plus one
  verification sweep. *Superseded in the shipped kernel* by exact norms recomputed at the start
  of every sweep and termination only after **two consecutive** zero-rotation sweeps; the
  per-item `info` convergence status tests exactly that predicate (`zero_sweeps >= 2`, not
  `sweeps_used < max_sweeps`: an item can leave the loop early without its second clean
  sweep). The verification sweep applies no rotations, so it costs only the Gram and
  threshold pass. `GesvdjParams` has no `drift_refresh_ratio` or `derijk` field.
- **\f$\sigma\f$ comes from A, always.** A final exact column norm, times \f$1/\beta\f$. The
  incrementally maintained `Nrm_local` exists only to choose rotations; reading \f$\sigma\f$ from
  it is a one-line shortcut that passes every existing test and reintroduces the
  normal-equations defect through the side door.
- **Sort descending**, as a parallel rank sort with ties broken on index so the permutation is
  a bijection, seeded with the identity first so a hypothetical collision degrades to a wrong
  permutation rather than a garbage address. Descending is the gesvd contract
  (`finalize_values_only` produces it by index reversal; `has_tiny_singular_values` and
  `patch_zero_left_vectors` read `sb[0]` as \f$\sigma_{\max}\f$). Do not also reverse indices.
- **Two writebacks, opposite lane roles.** U: lane is the output row (`U(lane, dst)`,
  coalesced). \f$V^H\f$: lane is the output row i and reads `V_local[r + c_i*LD]`
  (conflict-free because LD is odd). In the transposed orientation the Thin bound lands on
  `lane`, not on the inner loop; truncating the inner loop writes a wrongly shaped factor with
  plausible numbers.

## gesvdj_cta: rank deficiency, Thin and m < n

**Zero threshold relative to \f$\sigma_{\max}\f$ only:**
\f$\mathrm{tol}_0 = \mathtt{zero\_sigma\_multiplier}\cdot\varepsilon\cdot\sigma_{\max}\f$, in the
unscaled domain. `gesvd_blocked.cc`'s \f$\varepsilon \max(1, \sigma_{\max})\f$ declares every
\f$\sigma\f$ zero on a uniformly small input (\f$\sigma_{\max} = 10^{-10}\f$) and fabricates every U
column.

**Completion.** Columns CC..RR-1 of a full U, and any column below \f$\mathrm{tol}_0\f$, are not
determined by A. `patch_zero_left_vectors` fills them from a second tridiagonal eigensolve,
which a fused kernel does not have, so `gesvdj_cta` completes in-kernel: two passes of
classical Gram–Schmidt against the accepted columns (one pass against an ill-conditioned set
loses exactly the orthogonality the patch exists to provide). The trial cursor **runs across
output columns** and accepts above \f$1/(2 \cdot RR)\f$. The reviewed design restarted per column
and accepted above 1/2; with d dimensions left, a canonical vector's residual norm² is about
\f$d/RR\f$ (0.03 at d=1, RR=32), so the last columns were never filled (defect exactly 1.0 on a
32x8 input). With the running cursor, once \f$j_0\f$ trials are consumed some remaining one has
residual \f$\ge d/(RR - j_0) \ge 1/RR\f$, so a trial always passes. The branch is gated on a
warp-uniform predicate, so a well-conditioned square input pays one compare.

**Thin** (`SvdVectors::Thin`, LAPACK 'S') is natively cheap here: the thin U is what the solve
produces, so a Thin request sets `left_cols = CC` and skips completion, except that a
numerically deficient column is still repaired. Thin and All differ on at most one side, so
entry points canonicalise Thin to All whenever the shapes coincide.

**m < n solves \f$A^H\f$, not \f$A^T\f$**, transposing at load time (the tile is a few KB, so this
is free and avoids `gesvd_blocked`'s out-of-place transpose + recursion).
\f$A^H = U' S V'^H\f$ gives \f$A = V' S U'^H\f$: the same role mapping as \f$m \ge n\f$, swapped
between outputs. Solving \f$A^T\f$ gives \f$A = \overline{V'} S U'^T\f$, whose conjugations differ:
invisible in real arithmetic, wrong for complex.

## gesvdj_cta: traps and dead ends

- **Build group: `EXTENSIONS_CTA_SOURCES`, and nothing warns if it moves.** `gesvdj_cta.cc`
  sits in that list (`src/extensions/CMakeLists.txt`), whose object library is the only one
  built `NO_CPU_TARGETS`, and the shared library is the device-link unit. Moving it next to
  `gesvd_blocked.cc` in the factorization list gives no error: the 32-lane kernel just
  silently acquires a CPU AOT image.
- **The kernel name tag `GesvdjCTAKernel` lives outside the anonymous namespace**, so it does
  not depend on internal-linkage entities.
- **`GesvdjParams` is its own struct, not `JacobiParams`.** `JacobiParams::sort_order` defaults
  to ascending while gesvd's contract is descending; reusing it invites a silent reversal, and
  changing its default would alter `syev_jacobi_cta`. There is no sort-order field at all:
  the contract admits one order. The shared fields (`tol_multiplier`, `max_sweeps`,
  `cta_wg_size_multiplier`) keep identical names and semantics.
- **No Hermitian overload** (one-sided Jacobi has no use for the symmetry shortcut, and a
  second overload of the same arity is the option-struct trap), and
  `gesvdj_cta_buffer_size` returns 0: everything is local-memory resident. The workspace
  argument stays because the dispatcher passes one.
- **Every explicitly instantiated overload needs its own `template ...;` line**, spelled with
  `BATCHLAS_UNPAREN fp` because the type list passes `std::complex<float>` parenthesised; a
  missing line is an undefined symbol at final link, not a compile error in the `.cc`.
- **Register pressure grows with the scalar.** `ap + aq + g` per chunk is 48 live T at C=32
  (80 at C=64): ~60-75 registers in float, ~200+ for `complex<double>` (design-time estimate).
  The spec proposed half-rounds of 8 pairs for \f$\mathrm{sizeof}(T) \ge 16\f$; not
  implemented. Check `-Xcuda-ptxas -v` before believing a timing.
- **Every index into `pk`, `qk`, `g`, `ap`, `aq` must be compile-time.** The per-lane pair
  `k = lane >> 1` is re-read from `Pair_local` rather than indexing `pk[]` with a runtime index,
  which would spill the arrays.
- **The pair table is filled and read with the compile-time stride `kRotSlots`**, not
  `pairs_per_round`; they differ whenever CC < P, and a mismatch makes the unrolled loop
  dereference garbage column indices. Unused slots hold a sentinel (C) that fails `< CC`. The
  fill ends in a work-group barrier, so it must precede the `prob_id >= nb` early return or a
  ragged tail work-group deadlocks.
- **Conditionally unused accessors are sized 1, never 0**, with their base forced to 0.
- **`[[sycl::reqd_sub_group_size(32)]]` is load-bearing** (`189c381`): the 5-step and 4+1-step
  butterflies, `probs_per_warp = 32/P` and `part_id` are all silently wrong at any other width.
- **Off-CUDA correctness rests on sub-group lockstep:** `group_barrier` on a
  `SubGroupPartition` is a no-op off NVPTX. Lane = row has no cross-lane hazard in the update,
  but the `Rcs_local`/`Nrm_local` hand-offs still depend on it. The ROCm/NETLIB arms are the
  least defended.
- **Do not fully unroll the round loop** to keep V's row in registers (it would delete 64 of
  ~181 local-memory ops per round): 31 rounds × 16 pairs × ~12 instructions is ~6k
  instructions (48-96 KB) against a 32 KB Ada I-cache.
- **If de Rijk ordering is ever re-added** (it was removed, see the perf page), the spec's
  form was a permuted reload from global folded into V's initialisation,
  \f$V_{ij} = [i = \pi(j)]\f$; inverting the permutation returns a valid-looking but wrong V that
  reconstruction tests on symmetric-ish input do not catch.
- **The closed-form round-robin partner** (`partner(0) = (t mod ring) + 1`; for j ≥ 1,
  `u = j-1`, `v = ((2t-u) mod ring + ring) mod ring`, `partner(j) = v == u ? 0 : v+1`) is correct
  only because `ring = mp - 1` is odd. Useless under lane = row (every lane needs all pairs), kept
  here for any future lane = column variant.
- **n < P wastes lanes twice** (padded rows and inactive pair slots), and the reduce-scatter
  is sized at compile time. `gesvdjBatched` shares the cliff; it caps throughput at 17-31.
- **A is destroyed** (it becomes the rotated matrix), per the gesvd contract; benchmarks wrap
  it in `bench::pristine`.

## gesvd design: why SvdVectors::Thin exists

`SvdVectors` (`include/batchlas/blas/enums.hh`) follows LAPACK `jobu`/`jobvt`: `None`, `All`
('A') and `Thin` ('S'). Thin exists because All is unusable on tall-skinny input: a
10000 x 32 problem has to materialise a 10000 x 10000 U, 400 MB per matrix in float, so
batch=4 needs 1.6 GB for a factor whose last 9968 columns are an arbitrary orthonormal
completion the caller did not ask for.

The identity most of the implementation rests on: **Thin and All differ on at most one
side.** With \f$k = \min(m, n)\f$, for \f$m \le n\f$ a thin U (m x k) is exactly the full U, and
for \f$m \ge n\f$ a thin \f$V^H\f$ is exactly the full \f$V^H\f$; square input has Thin == All on
both sides. Entry points therefore canonicalise Thin to All whenever the shapes coincide
(`canonical_jobu` / `canonical_jobvh`), and only the genuinely thinner side has to be handled,
or rejected, by any given route.

LAPACK's 'O' (overwrite A with one of the factors) is deliberately absent. Add it as a further
enumerator if it is ever wanted: appending keeps the existing ordinals stable for the
benchmarks that pass jobs as ints.

## gesvd design: vendor binding and dispatch

- **The header declares `backend::gesvd_vendor` only**; each backend TU defines and explicitly
  instantiates the primary template for its own `Backend`. Function templates cannot be
  partially specialised on `Backend B`, so a throwing definition in the header made a
  `cusolver.cc` definition a redefinition error. This is the house idiom (`syev_vendor`,
  `ormqr_vendor`).
- **Explicit dispatch branch per route.** The tail of `gesvd_dispatch` is an unguarded
  `return gesvd_blocked(...)`, so a route without its own branch silently runs the blocked
  path under another label.
- **A forced route that `supports()` rejects falls back to Auto with no diagnostic**
  (`resolve_route`). Forcing `jacobi` on a 128x128 real matrix runs `blocked`; any test or
  benchmark that forces a route must verify the route taken.
- **The spec's ordering policy is superseded.** It put Jacobi *after* CTA until measured
  faster in time at every point; `da920c3` promoted it first on the accuracy evidence, and the
  `Provider` enum it extended was later replaced by `RouteTable<Op::gesvd, T>`
  ([evidence](../perf/gesvd.md#gesvd-promoting-jacobi-to-the-default-at-n-up-to-32)).
  The wide-band rule for 33..64 is in `preferred()`, not `supports()`, so it cannot make the
  band unservable vendor-free.

## gesvd design: test and benchmark traps

Recorded by the impl spec while wiring the tests and benchmarks; each one fails silently.

- **Register a new test binary in two lists** in `tests/CMakeLists.txt`: `TEST_TARGETS` and the
  `eig` label list. Omitting the label is not an error; the test still runs under bare
  `ctest` but disappears from `ctest -L eig` and every scoped command.
- **Fill test matrices by hand** with the plain `Matrix<T,F>(n, n, batch)` constructor when the
  host writes into them afterwards: `Matrix::Zeros`/`Matrix::Diagonal` launch asynchronous
  device kernels that race with those host writes on the same USM memory.
- **A test that goes through `gesvd` must assert the route taken**, on an observable only the
  intended path produces (for example \f$\|U^H U - I\|\f$ at \f$\kappa = 10^5\f$, which the
  normal-equations path cannot meet), because a forced route degrades silently.
- **Benchmarks wrap A in `bench::pristine`:** structured-mode setup runs once and gesvd
  destroys A, so otherwise every iteration after the first measures a different problem.
  Compute the workspace size before `SetKernel`, which moves from its arguments.
- **minibench positional arguments replace the whole registered sweep** for every benchmark
  in the binary, and `state.range(i)` returns 0 for a missing argument, so pass all four
  (`32 16384 1 1`), never just `32 16384`.
