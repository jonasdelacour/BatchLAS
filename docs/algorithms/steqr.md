# STEQR: implicit QR/QL on symmetric tridiagonals {#algo_steqr}

**Covers:** the mathematics implemented by `steqr`, `steqr_wg` and `steqr_cta` (and the leaf
solve of `stedc` and the fused small-n `syev`): deflation, scaling, shifts, the Givens bulge
chase, split book-keeping, the two CTA update schemes and eigenvector accumulation.
**Status:** current. Performance evidence for the CTA solver is in
[STEQR: the CTA tridiagonal solver](../perf/steqr.md); the stedc merge that consumes the leaf
solves is in @ref perf_stedc.

The solver takes a batch of tridiagonal matrices

\f[
T = \operatorname{tridiag}(e_1, \dots, e_{n-1}; d_1, \dots, d_n)
\f]

with diagonal entries \f$d_i\f$ and off-diagonal entries \f$e_i\f$, and computes

\f[
T = Q \Lambda Q^T,
\f]

where \f$\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)\f$ and, if requested,
\f$Q\f$ is the orthogonal eigenvector matrix.

The method is implicit shifted QR/QL with deflation, specialised for batched execution. The
work-group path and the CTA path share the mathematical core. They differ in how they represent
the active subproblems, how they book-keep splits, and how they accumulate the local orthogonal
transformations.

Which path runs (`steqr_dispatch`, `src/extensions/steqr.cc`): `steqr_cta` when
\f$0 < n \le\f$ the device's largest sub-group size, otherwise `steqr_wg`. On ROCm the CTA path
is disabled outright, because its chunked sub-group operations gave wrong eigenvalues on AMD
gfx1200 (`should_use_cta`).

## steqr: problem representation

For each matrix in the batch the solver stores only \f$d\f$ and \f$e\f$:

\f[
T =
\begin{bmatrix}
d_1 & e_1 &        &        & 0 \\
e_1 & d_2 & e_2    &        &   \\
    & e_2 & d_3    & \ddots &   \\
    &     & \ddots & \ddots & e_{n-1} \\
0   &     &        & e_{n-1} & d_n
\end{bmatrix}.
\f]

It never forms dense similarity transforms on \f$T\f$. Givens rotations chase a bulge through
the tridiagonal structure while preserving symmetry.

## steqr: the relative deflation criterion

If an off-diagonal entry is small enough the matrix splits into independent subproblems.
BatchLAS uses the LAPACK-style relative test

\f[
|e_i|^2 \le \varepsilon^2 |d_i| |d_{i+1}| + \mathrm{safmin},
\f]

where \f$\varepsilon\f$ is the machine precision and \f$\mathrm{safmin}\f$ the safe minimum. When
it holds, the implementation sets \f$e_i \leftarrow 0\f$ and

\f[
T =
\begin{bmatrix}
T_{11} & 0 \\
0      & T_{22}
\end{bmatrix}.
\f]

**Trap: `SteqrParams::zero_threshold` does not control deflation.** It is threaded through to
`steqr_wg`, `steqr_cta` (including the fused `syev` kernel) and `steqr_legacy`, and discarded in
all of them (`(void)zero_threshold` in `src/extensions/steqr_wg.cc` and
`src/extensions/steqr_cta_device.hh`). Every path uses the relative criterion above.

## steqr: safe scaling

Before sweeping an active block the implementation computes a block norm

\f[
\|T\|_\infty^{(\mathrm{block})} \approx \max\left( |d_i|, |e_i| \right)
\f]

over the block and rescales to avoid overflow or underflow: by
\f$\alpha = \mathrm{ssfmax} / \|T\|\f$ if the norm is too large, and by
\f$\alpha = \mathrm{ssfmin} / \|T\|\f$ if it is too small but nonzero. After the block
converges its entries are scaled back by \f$\alpha^{-1}\f$.

## steqr: shift selection

Each implicit step uses a shift \f$\mu\f$ derived from a \f$2 \times 2\f$ trailing or leading
principal block, depending on whether the iteration is QR or QL.

### Wilkinson shift

Given

\f[
B = \begin{bmatrix} a & b \\ b & c \end{bmatrix},
\qquad
\lambda_{\pm} = \frac{a+c}{2} \pm \sqrt{\left(\frac{a-c}{2}\right)^2 + b^2},
\f]

the Wilkinson shift is the eigenvalue closer to the endpoint being converged:

\f[
\mu =
\begin{cases}
\lambda_+ & \text{if } |\lambda_+ - c| < |\lambda_- - c|, \\
\lambda_- & \text{otherwise.}
\end{cases}
\f]

In the QL case the code forms the same expression from the leading \f$2 \times 2\f$ block and
takes the eigenvalue closest to the top endpoint.

### LAPACK-style stable shift

The CTA path also has the stable implicit-shift formula. For a local \f$2 \times 2\f$ block

\f[
g = \frac{d_2 - d_1}{2 e_1}, \qquad
r = \sqrt{g^2 + 1}, \qquad
\mu = d_1 - \frac{e_1}{g + \operatorname{copysign}(r, g)},
\f]

which avoids the subtractive cancellation of the classical quadratic formula.

### Which path uses which shift

- `steqr_wg` always uses the Wilkinson shift.
- `steqr_cta` takes either through `SteqrParams::cta_shift_strategy`
  (`SteqrShiftStrategy::Lapack`, the default, or `Wilkinson`).
- `syev`'s CTA arms override the default: `syev_cta_steqr_params`
  (`include/batchlas/blas/functions/syev.hh`) sets `Wilkinson` and `max_sweeps = 400`,
  deliberately slower and more robust, because `syev` runs inside `syevx`, where an inaccurate
  Ritz solve stagnates the outer iteration.

## steqr: Givens rotations

The basic transformation is

\f[
G(c,s) =
\begin{bmatrix}
c & -s \\
s &  c
\end{bmatrix},
\qquad c^2 + s^2 = 1,
\qquad
G(c,s)^T
\begin{bmatrix} x \\ y \end{bmatrix}
=
\begin{bmatrix} r \\ 0 \end{bmatrix},
\f]

with \f$(c, s, r)\f$ from `lartg(x, y)`. An implicit QR or QL step applies a sequence of these
as orthogonal similarities \f$T \leftarrow G_k^T T G_k\f$, which preserves symmetry, introduces a
single bulge and then chases it.

## steqr: the local similarity update on the band

The central local operation is a similarity with an adjacent Givens rotation on rows and
columns \f$i\f$ and \f$i+1\f$:

\f[
\widehat G_i = \operatorname{diag}(I_{i-1}, G(c,s), I_{n-i-1}),
\qquad
T^+ = \widehat G_i^T T \widehat G_i .
\f]

Because \f$T\f$ is tridiagonal this changes only the band around indices \f$i-1, i, i+1, i+2\f$.

### The local window being updated

Before the update the window is

\f[
T_{\mathrm{loc}} =
\begin{bmatrix}
\alpha & \beta & 0 & 0 \\
\beta  & d_i   & e_i & 0 \\
0       & e_i   & d_{i+1} & \gamma \\
0       & 0     & \gamma & \delta
\end{bmatrix},
\qquad
\alpha = d_{i-1},\ \beta = e_{i-1},\ \gamma = e_{i+1},\ \delta = d_{i+2},
\f]

whenever those entries exist. After the similarity on the middle \f$2 \times 2\f$ block

\f[
T_{\mathrm{loc}}^+ =
\begin{bmatrix}
\alpha & c\beta - s b_{\mathrm{prev}} & -s\beta & 0 \\
c\beta - s b_{\mathrm{prev}} & d_i^+ & e_i^+ & -s\gamma \\
-s\beta & e_i^+ & d_{i+1}^+ & c\gamma \\
0 & -s\gamma & c\gamma & \delta
\end{bmatrix},
\f]

with \f$b_{\mathrm{prev}} = 0\f$ for the first rotation of a sweep. The entry \f$-s\gamma\f$ is
the new bulge, chased to the next position. One step therefore touches only
\f$(e_{i-1}, d_i, e_i, d_{i+1}, e_{i+1})\f$.

### Scalar formulas of the explicit update

The work-group path and the CTA `EXP` scheme use

\f[
\begin{aligned}
d_i^+     &= c(c d_i - e_i s) - s(e_i c - s d_{i+1}), \\
d_{i+1}^+ &= c(c d_{i+1} + e_i s) + s(e_i c + s d_i), \\
e_i^+     &= c(c e_i + s d_i) - s(c d_{i+1} + s e_i), \\
e_{i-1}^+ &= c e_{i-1} - s b_{\mathrm{prev}}, \\
e_{i+1}^+ &= c e_{i+1},
\end{aligned}
\f]

and the outgoing bulge is \f$b_{\mathrm{new}} = -s e_{i+1}\f$. One step annihilates the current
off-band bulge, updates one adjacent \f$2 \times 2\f$ pivot block, and creates the next bulge.

### First step versus interior step

The first rotation of a shifted step is generated from the shifted active corner
\f$(d_{\mathrm{edge}} - \mu,\ e_{\mathrm{edge}})^T\f$, so the shift creates the bulge. Every
later rotation is generated from \f$(e_{i-1}^{(\mathrm{current})},\ b_{\mathrm{prev}})^T\f$: the
bulge is moved one position, not recreated.

## steqr: choosing QR versus QL

For an active block with endpoints \f$d_\ell\f$ and \f$d_r\f$, BatchLAS converges from the
smaller-magnitude endpoint: QR if \f$|d_\ell| \le |d_r|\f$, QL otherwise. QR chases the bulge
top to bottom and converges the bottom eigenvalue first; QL chases bottom to top and converges
the top eigenvalue first. This is the LAPACK policy.

## steqr: the exact 2x2 solve

A deflated block of size 2 is diagonalised analytically. For
\f$T_2 = \begin{bmatrix} a & b \\ b & c \end{bmatrix}\f$ the eigenvalues are \f$\lambda_\pm\f$ as
above. With eigenvectors requested the implementation also computes

\f[
Q_2 = \begin{bmatrix} c_2 & -s_2 \\ s_2 & c_2 \end{bmatrix},
\qquad
Q_2^T T_2 Q_2 = \operatorname{diag}(\lambda_1, \lambda_2),
\f]

and accumulates that rotation into the global eigenvector matrix.

## steqr: split structure and book-keeping

After deflation the solver iterates on maximal active blocks separated by zero off-diagonals,
never on the whole matrix.

### Definition of a split

Let \f$\mathcal Z = \{ i \in \{1,\dots,n-1\} : e_i = 0 \}\f$. The active blocks are the maximal
intervals \f$I_k = [s_k, t_k] \subseteq \{1,\dots,n\}\f$ with \f$e_j \neq 0\f$ for
\f$j = s_k, \dots, t_k-1\f$; equivalently

\f[
e_{s_k-1} = 0 \text{ or } s_k = 1,
\qquad
e_{t_k} = 0 \text{ or } t_k = n,
\qquad
T = \operatorname{diag}(T_{I_1}, T_{I_2}, \dots, T_{I_p}).
\f]

Each \f$T_{I_k}\f$ is processed independently.

### Work-group path: a global block list per pass

On every outer pass, for each batch item, `steqr_wg` scans \f$e\f$ left to right and records each
maximal run \f$e_s, \dots, e_{t-1}\f$ of nonzeros as the diagonal block \f$[s, t]\f$. In code the
pair is `(start_ix, end_ix)` with `end_ix` exclusive, so the subproblem is `d[s:t]`, `e[s:t-1]`.
The per-matrix descriptors are compacted by an inclusive scan into one global list of
`(batch index, start index, end index)` triples, and all blocks across the batch are processed in
parallel. Hence the two-stage outer iteration:

1. discover and compact active blocks;
2. apply one Francis sweep to each block.

New zeros may appear during the sweep, so the next pass rebuilds the list from scratch.

### CTA path: a left-to-right walk per matrix

`steqr_cta` builds no global list. Each sub-group partition holds one matrix in registers and
walks its block decomposition locally, keeping a running left boundary \f$b_{\mathrm{next}}\f$.
Each outer iteration sets \f$b_{\mathrm{begin}} \leftarrow b_{\mathrm{next}}\f$, enforces the
left boundary with \f$e_{b_{\mathrm{begin}}-1} \leftarrow 0\f$ when that index exists, deflates
the tail, and takes the first zero to the right as the right boundary:

\f[
b_{\mathrm{end}} = \min\{ i \ge b_{\mathrm{begin}} : e_i = 0 \}
\quad (b_{\mathrm{end}} = n \text{ if none}),
\qquad
b_{\mathrm{next}} \leftarrow b_{\mathrm{end}} + 1.
\f]

### How inner splits shrink the active window

Once a block \f$[\ell:r]\f$ is selected, the iteration re-tests deflation after every implicit
step and locates the first (QL) or last (QR) new zero.

For QL, with active window \f$[\ell:m]\f$, it finds
\f$p = \min\{ i \in [\ell, m-1] : e_i = 0 \}\f$:

- \f$p = \ell\f$: the left-edge eigenvalue has converged;
- \f$p = \ell+1\f$: a \f$2 \times 2\f$ block remains and is solved exactly;
- otherwise the next step is applied only to \f$[\ell:p]\f$.

For QR, with active window \f$[m:r]\f$, it finds
\f$p = \max\{ i \in [m+1, r] : e_{i-1} = 0 \}\f$:

- \f$p = r\f$: the right-edge eigenvalue has converged;
- \f$p = r-1\f$: a \f$2 \times 2\f$ block remains and is solved exactly;
- otherwise the next step is applied only to \f$[p:r]\f$.

This nested structure (split loop, loop over \f$\ell\f$, sweep loop) is what the CTA
[lockstep flat solver](#steqr-the-lockstep-flat-cta-solver) restructures.

## steqr: pseudoalgorithms

### Algorithm 1: BatchLAS STEQR

```text
Input:  d = (d_1, ..., d_n), e = (e_1, ..., e_{n-1}), jobz, params
Output: eigenvalues lambda and, optionally, eigenvectors Q

if jobz = EigenVectors and back_transform = false then
  Q <- I_n
end if
copy (d, e) into working storage

if the CTA path is available and n <= sub-group capacity then
  (d, e, Q) <- Algorithm 4 applied independently to each batch item
else
  repeat n-1 times
    construct the list of maximal active blocks (Algorithm 2)
    for each active block in parallel do
      apply one shifted Francis sweep (Algorithm 3)
    end for
    if jobz = EigenVectors then
      apply the stored rotations to Q
    end if
  end repeat
end if

lambda <- d
if sorting is requested then
  sort (lambda, Q) jointly
end if
```

### Algorithm 2: active blocks from deflation

```text
Input:  e = (e_1, ..., e_{n-1})
Output: maximal active blocks I_1, ..., I_p

p <- 0; i <- 1
while i <= n-1 do
  if e_i = 0 then
    i <- i + 1
  else
    s <- i
    while i <= n-1 and e_i != 0 do
      i <- i + 1
    end while
    t <- i
    p <- p + 1
    I_p <- [s, t]
  end if
end while
```

\f$I_p = [s,t]\f$ is the subproblem on \f$d_s, \dots, d_t\f$ and \f$e_s, \dots, e_{t-1}\f$.

### Algorithm 3: one shifted sweep on an active block

```text
Input:  active block [s:t], local (d, e)
Output: updated (d, e) and optionally the stored Givens rotations

if t = s then return
if t = s + 1 then diagonalize the 2x2 block exactly; return

scale the block into a safe range if necessary
if |d_s| <= |d_t| then use QR ordering
else use QL ordering by reversing the local index map

form shift mu from the trailing 2x2 block in the chosen ordering
compute the first Givens rotation from (d_edge - mu, e_edge)
apply the local similarity update
for each remaining local index do
  compute the next rotation from (current coupling, incoming bulge)
  apply the same local similarity update
end for

apply the relative deflation test to every local off-diagonal
undo scaling
```

### Algorithm 4: CTA STEQR on one matrix

```text
Input:  one tridiagonal matrix (d, e)
Output: converged d and optional eigenvector matrix Q

load d and e into registers
cache Q in shared memory if jobz = EigenVectors
budget <- n * max_sweeps
next_block_begin <- 1

while next_block_begin <= n do
  block_begin <- next_block_begin
  if block_begin > 1 then e_(block_begin-1) <- 0
  deflate the tail [block_begin:n]
  block_end <- first i >= block_begin with e_i = 0, or n if none
  next_block_begin <- block_end + 1
  if block_end <= block_begin then continue

  scale the block [block_begin:block_end] if necessary
  if |d_block_end| < |d_block_begin| then
    QL from the left edge: repeatedly locate the first split p,
      solve 1x1 or 2x2 edge cases when reached,
      otherwise apply one implicit QL step on [ell:p]
  else
    QR from the right edge: repeatedly locate the last split p,
      solve 1x1 or 2x2 edge cases when reached,
      otherwise apply one implicit QR step on [p:r]
  end if
  rescale the block back
  decrease budget after every implicit step
  if budget is exhausted then mark non-convergence and stop
end while
```

### QR and QL in virtual coordinates

For QR the work-group path uses the physical ordering \f$(d_1, \dots, d_n)\f$,
\f$(e_1, \dots, e_{n-1})\f$. The first rotation annihilates the first subdiagonal of the shifted
matrix,

\f[
G_1^T
\begin{bmatrix} d_1 - \mu \\ e_1 \end{bmatrix}
=
\begin{bmatrix} r \\ 0 \end{bmatrix},
\f]

and each later rotation annihilates the bulge created by the previous one.

For QL the same formulas run on reversed local indices. The implementation defines, per active
block, the virtual coordinates

\f[
\tilde d_k = d_{n-k+1}, \qquad \tilde e_k = e_{n-k},
\f]

and applies the same QR-style bulge chase to \f$(\tilde d_1, \dots, \tilde d_{n_{\mathrm{local}}})\f$,
\f$(\tilde e_1, \dots, \tilde e_{n_{\mathrm{local}}-1})\f$. That is a QL iteration in the
original coordinates.

## steqr: the CTA update schemes EXP and PG

`steqr_cta` has two mathematically equivalent, operationally different ways to write one
implicit step, chosen by `SteqrParams::cta_update_scheme`. The default is `EXP`
(`include/batchlas/blas/extensions.hh`); the enum's comment calling `PG` "current default" is
stale.

### EXP: explicit similarity update

This mirrors the work-group formulas. The rotation of each step is built from
\f$(d_1 - \mu, e_1)\f$ for the first step, or from
\f$(e_{j-1}^{(\mathrm{updated})}, b_{j-1})\f$ later, and then the local band is updated with the
explicit scalar formulas above. The state is the local band plus the current bulge. Every lane
conceptually owns one diagonal and one off-diagonal entry, and one step updates the
\f$4 \times 4\f$ window. This makes the CTA path mathematically closest to `steqr_wg`.

### PG: Parlett-Gray scalar recurrence

`PG` uses the classical implicit recurrence and propagates a few scalars
\f$(g, p, c, s)\f$ across the sweep instead of writing every local similarity. In QR form, step
\f$i\f$ is

\f[
\begin{aligned}
f_i &= s_{i-1} e_i, \\
(c_i, s_i, r_i) &= \operatorname{lartg}(g_{i-1}, f_i), \\
g_i^{\star} &= d_i - p_{i-1}, \\
r_i^{\star} &= (d_{i+1} - g_i^{\star}) s_i + 2 c_i c_{i-1} e_i, \\
p_i &= s_i r_i^{\star}, \\
d_i &\leftarrow g_i^{\star} + p_i, \\
g_i &= c_i r_i^{\star} - c_{i-1} e_i .
\end{aligned}
\f]

The QL version is the mirrored recurrence with the sweep direction reversed. It is the same
similarity in a different state representation, closer to the compact scalar formulas of
LAPACK's implicit tridiagonal QR/QL kernels.

## steqr: the lockstep flat CTA solver

The CTA solve runs one tridiagonal per chunk of \f$P\f$ lanes, several chunks per warp. With the
nested loop structure above, chunks of a warp realign only at loop exits, so the warp pays the
slowest chunk's sweep count per eigenvalue. `steqr_cta_solve_flat` hoists the sweep out of the
nest: each chunk first *settles* (opens a block, tests, advances past deflated eigenvalues,
2x2-solves) until it has a sweep \f$[\ell, m]\f$ to run or is done, and then every chunk with a
sweep chases in the same pass. The per-chunk operation sequence is the nested solver's, so the
results are bitwise equal and the mathematics on this page is unchanged. It ships only for some
(type, \f$P\f$) pairs. The design, the `kSteqrCtaFlatPays` gate, the sub-group-wide form and what
was measured worse are recorded in
[Lockstep flat solver](../perf/steqr.md#lockstep-flat-solver); the full-warp partition and the
work-group multiplier are on the same page.

## steqr: eigenvector accumulation

With eigenvectors requested the solver accumulates the similarities into \f$Q\f$. With
`back_transform = false` it starts from \f$Q \leftarrow I\f$; otherwise it treats the given
matrix as an incoming basis and right-applies the tridiagonal eigenvector transforms. Each local
rotation on adjacent columns \f$k, k+1\f$ does

\f[
Q_{:,\{k,k+1\}} \leftarrow Q_{:,\{k,k+1\}} G(c,s).
\f]

In the QL case the physical column order is reversed relative to the virtual sweep order, but
the operation is still a right multiplication by the corresponding adjacent rotation.
`steqr_wg` stores the rotations of a sweep and applies them afterwards; `steqr_cta` usually
applies them immediately to a shared-memory cache of \f$Q\f$.

## steqr: sorting

After convergence the final diagonal is copied out, \f$\lambda_i \leftarrow d_i\f$. With
`sort = true` the eigenpairs are sorted, ascending by default
(\f$\lambda_1 \le \lambda_2 \le \dots \le \lambda_n\f$), and the same permutation is applied to
the columns of \f$Q\f$.

## steqr: what is specific to BatchLAS

The mathematics is standard implicit tridiagonal QR/QL. The BatchLAS-specific choices are:

- batched from the ground up;
- a dynamic choice between the general work-group path and the small-n CTA path;
- two CTA update schemes, `PG` and `EXP`;
- the user-facing `zero_threshold` is not the deflation criterion;
- eigenpair sorting is part of the solver pipeline.

In short,

\f[
T_0 \xrightarrow{\text{deflation + implicit shifted QR/QL steps}} T_k = \Lambda,
\qquad
Q = G_1 G_2 \cdots G_p,
\qquad
Q^T T_0 Q = \Lambda .
\f]
