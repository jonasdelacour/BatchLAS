# STEQR: implicit QR/QL on symmetric tridiagonals {#algo_steqr}

> **Status:** current.

`steqr`, `steqr_wg` and `steqr_cta` solve batches of symmetric tridiagonal eigenproblems by implicit
shifted QR/QL with deflation. This page gives the mathematics: deflation, scaling, shifts, the
Givens bulge chase, split bookkeeping, the two CTA update schemes and eigenvector accumulation. It
also covers the leaf solve of `stedc` and the fused small-n `syev`. CTA performance is in
[STEQR: the CTA tridiagonal solver](../perf/steqr.md); the stedc merge is in @ref perf_stedc.

The solver takes a batch of tridiagonal matrices

\f[
T = \operatorname{tridiag}(e_1, \dots, e_{n-1}; d_1, \dots, d_n)
\f]

and computes

\f[
T = Q \Lambda Q^T,
\f]

where \f$\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)\f$ and, if requested,
\f$Q\f$ is the orthogonal eigenvector matrix.

Which path runs (`steqr_dispatch`, `src/extensions/steqr.cc`): `steqr_cta` when
\f$0 < n \le\f$ the device's largest sub-group size, otherwise `steqr_wg`.

> **Note:** On ROCm the CTA path is disabled (`should_use_cta`). Its chunked sub-group operations
> gave wrong eigenvalues on AMD gfx1200.

## steqr: problem representation

Each matrix is stored as its diagonal \f$d\f$ and off-diagonal \f$e\f$ only:

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

No dense similarity transform is formed. Givens rotations chase a bulge through the band.

## steqr: the relative deflation criterion

An off-diagonal entry deflates when

\f[
|e_i|^2 \le \varepsilon^2 |d_i| |d_{i+1}| + \mathrm{safmin},
\f]

where \f$\varepsilon\f$ is the machine precision and \f$\mathrm{safmin}\f$ the safe minimum (LAPACK
style). The code then sets \f$e_i \leftarrow 0\f$, and \f$T\f$ becomes block diagonal:

\f[
T =
\begin{bmatrix}
T_{11} & 0 \\
0      & T_{22}
\end{bmatrix}.
\f]

> **Warning:** `SteqrParams::zero_threshold` does not control deflation. It is passed to
> `steqr_wg`, `steqr_cta` (including the fused `syev` kernel) and `steqr_legacy`, and discarded in
> each (`(void)zero_threshold` in `src/extensions/steqr_wg.cc` and
> `src/extensions/steqr_cta_device.hh`). All paths use the relative criterion.

## steqr: safe scaling

Before sweeping an active block, the solver computes its norm

\f[
\|T\|_\infty^{(\mathrm{block})} \approx \max\left( |d_i|, |e_i| \right)
\f]

and rescales against overflow and underflow: by \f$\alpha = \mathrm{ssfmax} / \|T\|\f$ if the norm
is too large, and by \f$\alpha = \mathrm{ssfmin} / \|T\|\f$ if it is too small but nonzero. After
convergence the block is scaled back by \f$\alpha^{-1}\f$.

## steqr: shift selection

Each implicit step uses a shift \f$\mu\f$ from a \f$2 \times 2\f$ principal block at the trailing
(QR) or leading (QL) end.

### Wilkinson shift

For \f$B = \begin{bmatrix} a & b \\ b & c \end{bmatrix}\f$,

\f[
\lambda_{\pm} = \frac{a+c}{2} \pm \sqrt{\left(\frac{a-c}{2}\right)^2 + b^2},
\qquad
\mu =
\begin{cases}
\lambda_+ & \text{if } |\lambda_+ - c| < |\lambda_- - c|, \\
\lambda_- & \text{otherwise.}
\end{cases}
\f]

The QL case uses the leading block and the eigenvalue closest to the top endpoint.

### LAPACK-style stable shift

For a local \f$2 \times 2\f$ block with leading entries \f$d_1, d_2, e_1\f$, the CTA path can use

\f[
g = \frac{d_2 - d_1}{2 e_1}, \qquad
r = \sqrt{g^2 + 1}, \qquad
\mu = d_1 - \frac{e_1}{g + \operatorname{copysign}(r, g)},
\f]

which avoids the cancellation of the quadratic formula.

### Which path uses which shift

- `steqr_wg` always uses the Wilkinson shift.
- `steqr_cta` selects it with `SteqrParams::cta_shift_strategy`: `SteqrShiftStrategy::Lapack`
  (default) or `Wilkinson`.
- `syev`'s CTA arms (`syev_cta_steqr_params`, `include/batchlas/blas/functions/syev.hh`) set
  `Wilkinson` and `max_sweeps = 400`. The solver runs inside `syevx`, where an inaccurate Ritz
  solve stalls the outer iteration, so the slower, more robust setting is used.

## steqr: Givens rotations

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

with \f$(c, s, r)\f$ from `lartg(x, y)`. An implicit step applies a sequence of these as
similarities \f$T \leftarrow G_k^T T G_k\f$. Each similarity preserves symmetry, introduces a single
bulge and chases it.

## steqr: the local similarity update on the band

A similarity with an adjacent rotation on rows and columns \f$i, i+1\f$ is

\f[
\widehat G_i = \operatorname{diag}(I_{i-1}, G(c,s), I_{n-i-1}),
\qquad
T^+ = \widehat G_i^T T \widehat G_i ,
\f]

which changes only the band around indices \f$i-1, \dots, i+2\f$. The local window before the
update is

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

with entries that do not exist omitted. After the similarity,

\f[
T_{\mathrm{loc}}^+ =
\begin{bmatrix}
\alpha & c\beta - s b_{\mathrm{prev}} & -s\beta & 0 \\
c\beta - s b_{\mathrm{prev}} & d_i^+ & e_i^+ & -s\gamma \\
-s\beta & e_i^+ & d_{i+1}^+ & c\gamma \\
0 & -s\gamma & c\gamma & \delta
\end{bmatrix},
\f]

with \f$b_{\mathrm{prev}} = 0\f$ for the first rotation of a sweep. The entry \f$-s\gamma\f$ is the
new bulge. One step touches only \f$(e_{i-1}, d_i, e_i, d_{i+1}, e_{i+1})\f$.

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

and the outgoing bulge is \f$b_{\mathrm{new}} = -s e_{i+1}\f$.

### First step versus interior step

The first rotation is generated from the shifted corner
\f$(d_{\mathrm{edge}} - \mu,\ e_{\mathrm{edge}})^T\f$, so the shift creates the bulge. Each later
rotation is generated from \f$(e_{i-1}^{(\mathrm{current})},\ b_{\mathrm{prev}})^T\f$. The bulge
moves one position and is not recreated.

## steqr: choosing QR versus QL

For a block with endpoints \f$d_\ell\f$ and \f$d_r\f$, the solver converges from the endpoint of
smaller magnitude: QR if \f$|d_\ell| \le |d_r|\f$, QL otherwise. QR chases the bulge top to bottom
and converges the bottom eigenvalue first. QL is the reverse. This is the LAPACK policy.

## steqr: the exact 2x2 solve

A deflated block of size 2, \f$T_2 = \begin{bmatrix} a & b \\ b & c \end{bmatrix}\f$, is solved
analytically. With eigenvectors requested, it also yields a rotation

\f[
Q_2 = \begin{bmatrix} c_2 & -s_2 \\ s_2 & c_2 \end{bmatrix},
\qquad
Q_2^T T_2 Q_2 = \operatorname{diag}(\lambda_1, \lambda_2),
\f]

which is accumulated into the eigenvector matrix.

## steqr: split structure and book-keeping

The solver iterates on maximal active blocks, never on the whole matrix. Let
\f$\mathcal Z = \{ i : e_i = 0 \}\f$. The active blocks are the maximal intervals
\f$I_k = [s_k, t_k]\f$ with \f$e_j \neq 0\f$ for \f$j = s_k, \dots, t_k-1\f$, so that

\f[
T = \operatorname{diag}(T_{I_1}, T_{I_2}, \dots, T_{I_p}),
\f]

and each \f$T_{I_k}\f$ is processed independently.

### Work-group path: a global block list

On each outer pass, `steqr_wg` scans \f$e\f$ and records each maximal run of nonzeros
\f$e_s, \dots, e_{t-1}\f$ as the block \f$[s, t]\f$ (`(start_ix, end_ix)`, `end_ix` exclusive). An
inclusive scan compacts the descriptors into one list of
`(batch index, start index, end index)` triples, and all blocks across the batch run in parallel.
Each outer pass has two stages:

1. find and compact the active blocks;
2. apply one Francis sweep to each block.

A sweep can create new zeros, so the list is rebuilt on every pass.

### CTA path: a left-to-right walk

`steqr_cta` builds no global list. Each sub-group partition holds one matrix in registers and walks
its blocks with a running left boundary \f$b_{\mathrm{next}}\f$. Each iteration sets
\f$b_{\mathrm{begin}} \leftarrow b_{\mathrm{next}}\f$, enforces \f$e_{b_{\mathrm{begin}}-1} \leftarrow 0\f$
when that index exists, and takes the first zero to the right as the right boundary:

\f[
b_{\mathrm{end}} = \min\{ i \ge b_{\mathrm{begin}} : e_i = 0 \}
\quad (b_{\mathrm{end}} = n \text{ if none}),
\qquad
b_{\mathrm{next}} \leftarrow b_{\mathrm{end}} + 1.
\f]

### Inner splits shrink the window

After each implicit step the solver re-tests deflation and locates the first (QL) or last (QR)
new zero.

- **QL**, window \f$[\ell:m]\f$: with \f$p = \min\{ i \in [\ell, m-1] : e_i = 0 \}\f$, the left
  eigenvalue has converged if \f$p = \ell\f$; a \f$2 \times 2\f$ block remains if \f$p = \ell+1\f$
  (solved exactly); otherwise the next step runs on \f$[\ell:p]\f$.
- **QR**, window \f$[m:r]\f$: with \f$p = \max\{ i \in [m+1, r] : e_{i-1} = 0 \}\f$, the right
  eigenvalue has converged if \f$p = r\f$; a \f$2 \times 2\f$ block remains if \f$p = r-1\f$;
  otherwise the next step runs on \f$[p:r]\f$.

The nested loops (split, \f$\ell\f$, sweep) are what the
[lockstep flat solver](#steqr-the-lockstep-flat-cta-solver) restructures.

## steqr: CTA control flow

### Control flow of `steqr_cta` on one matrix

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

QR uses the physical ordering. The first rotation annihilates the first subdiagonal of the shifted
matrix,

\f[
G_1^T
\begin{bmatrix} d_1 - \mu \\ e_1 \end{bmatrix}
=
\begin{bmatrix} r \\ 0 \end{bmatrix},
\f]

and each later rotation annihilates the bulge left by the previous one.

QL runs the same formulas on reversed local indices. Per active block, define

\f[
\tilde d_k = d_{n-k+1}, \qquad \tilde e_k = e_{n-k},
\f]

and apply the QR-style chase to \f$(\tilde d_k)\f$ and \f$(\tilde e_k)\f$. This is a QL iteration in
the original coordinates.

## steqr: the CTA update schemes EXP and PG

`steqr_cta` writes one implicit step in one of two equivalent forms, chosen by
`SteqrParams::cta_update_scheme`. The default is `EXP`. (The enum comment that calls `PG` the
"current default" is stale.)

- **`EXP`** (explicit similarity) applies the scalar formulas of the local band update above. Each
  lane owns one diagonal and one off-diagonal entry, and one step updates a \f$4 \times 4\f$
  window. This is the form closest to `steqr_wg`.
- **`PG`** (Parlett-Gray) propagates the scalars \f$(g, p, c, s)\f$ across the sweep instead of
  writing each local similarity. In QR form, step \f$i\f$ is

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

  QL is the mirrored recurrence with the sweep reversed.

## steqr: the lockstep flat CTA solver

The CTA solver runs one tridiagonal per chunk of \f$P\f$ lanes, several chunks per warp. In the
nested loop structure, chunks realign only at loop exits, so the warp pays the slowest chunk's
sweep count per eigenvalue. `steqr_cta_solve_flat` moves the sweep out of the nest. Each chunk
first settles (opens a block, tests, advances past deflated eigenvalues, solves 2x2 blocks) until it
has a sweep \f$[\ell, m]\f$ or is done. Then all chunks with a sweep run it in the same pass. The
per-chunk operation sequence is unchanged, so results are bitwise equal to the nested solver. The
flat solver ships only for some (type, \f$P\f$) pairs. The `kSteqrCtaFlatPays` gate, the
sub-group-wide form and the measured losses are in
[Lockstep flat solver](../perf/steqr.md#lockstep-flat-solver).

## steqr: eigenvector accumulation

With `back_transform = false` the solver starts from \f$Q \leftarrow I\f$. Otherwise it treats the
input matrix as a basis and right-applies the rotations. Each local rotation on columns \f$k, k+1\f$
does

\f[
Q_{:,\{k,k+1\}} \leftarrow Q_{:,\{k,k+1\}} G(c,s).
\f]

In QL the physical column order is reversed relative to the sweep, but each operation is still a
right multiplication by an adjacent rotation. `steqr_wg` stores a sweep's rotations and applies
them afterwards. `steqr_cta` usually applies them at once to a shared-memory cache of \f$Q\f$.

## steqr: sorting

After convergence \f$\lambda_i \leftarrow d_i\f$. With `sort = true` the eigenpairs are sorted
ascending, \f$\lambda_1 \le \lambda_2 \le \dots \le \lambda_n\f$, and the same permutation is applied
to the columns of \f$Q\f$.

In short,

\f[
T_0 \xrightarrow{\text{deflation + implicit shifted QR/QL steps}} T_k = \Lambda,
\qquad
Q = G_1 G_2 \cdots G_p,
\qquad
Q^T T_0 Q = \Lambda .
\f]
