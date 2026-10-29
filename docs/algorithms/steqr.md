# STEQR: implicit QR/QL on symmetric tridiagonals {#algo_steqr}

> **Status:** current.

`steqr`, `steqr_wg` and `steqr_cta` solve batches of symmetric tridiagonal eigenproblems
\f$T = Q \Lambda Q^T\f$ by implicit shifted QR/QL with deflation; `steqr_cta` is also the leaf
solve of `stedc` and the solver of the fused small-n `syev`. Measurements are in
[STEQR: the CTA tridiagonal solver](../perf/steqr.md); the stedc merge is in @ref perf_stedc.

Which path runs (`should_use_cta`, `src/ops/syev/steqr/steqr.cc`): `steqr_cta` when
\f$0 < n \le 32\f$ and the device supports sub-group size 32, otherwise `steqr_wg`. On ROCm the
CTA path is disabled: its chunked sub-group operations gave wrong eigenvalues on AMD gfx1200.

## steqr: problem representation

Each matrix is stored as its diagonal \f$d_1, \dots, d_n\f$ and off-diagonal
\f$e_1, \dots, e_{n-1}\f$ only. No dense similarity transform is formed; Givens rotations chase a
bulge through the band.

## steqr: the relative deflation criterion

An off-diagonal entry is set to zero when

\f[
|e_i|^2 \le \varepsilon^2 |d_i| |d_{i+1}| + \mathrm{safmin},
\f]

with \f$\varepsilon\f$ the machine precision and \f$\mathrm{safmin}\f$ the safe minimum (LAPACK
style), splitting \f$T\f$ into independent diagonal blocks. Both paths use this relative test;
there is no absolute deflation threshold.

## steqr: safe scaling

Before sweeping an active block the solver takes \f$a = \max_i(|d_i|, |e_i|)\f$ over the block and
scales the block by \f$\mathrm{ssfmax}/a\f$ if \f$a > \mathrm{ssfmax}\f$, or by
\f$\mathrm{ssfmin}/a\f$ if \f$0 < a < \mathrm{ssfmin}\f$. After convergence it scales back.

## steqr: shift selection

Each implicit step takes a shift \f$\mu\f$ from the \f$2 \times 2\f$ block at the end being
converged.

- **Wilkinson.** For \f$\begin{bmatrix} a & b \\ b & c \end{bmatrix}\f$,
  \f$\lambda_\pm = \tfrac{a+c}{2} \pm \sqrt{(\tfrac{a-c}{2})^2 + b^2}\f$, and \f$\mu\f$ is the
  \f$\lambda_\pm\f$ closer to the corner entry being converged.
- **LAPACK-style stable shift.** With leading entries \f$d_1, d_2, e_1\f$,

\f[
g = \frac{d_2 - d_1}{2 e_1}, \qquad
r = \sqrt{g^2 + 1}, \qquad
\mu = d_1 - \frac{e_1}{g + \operatorname{copysign}(r, g)},
\f]

  which avoids the cancellation of the quadratic formula.

`steqr_wg` always uses Wilkinson. `steqr_cta` selects with `SteqrParams::cta_shift_strategy`:
`SteqrShiftStrategy::Lapack` (default) or `Wilkinson`. `syev`'s CTA arms (`cta_steqr_params`,
`src/ops/syev/syev.cc`) set `Wilkinson` and `max_sweeps = 400`: the solver runs inside `syevx`,
where an inaccurate Ritz solve stalls the outer iteration, so the slower, more robust setting wins.

## steqr: the Givens bulge chase

With \f$G(c,s) = \begin{bmatrix} c & -s \\ s & c \end{bmatrix}\f$ and \f$(c, s, r)\f$ from
`lartg(x, y)` so that \f$G^T (x, y)^T = (r, 0)^T\f$, an implicit step applies adjacent rotations
as similarities \f$T \leftarrow G_i^T T G_i\f$ on rows and columns \f$i, i+1\f$. Each touches only
\f$(e_{i-1}, d_i, e_i, d_{i+1}, e_{i+1})\f$. The work-group path and the CTA `EXP` scheme write it
explicitly:

\f[
\begin{aligned}
d_i^+     &= c(c d_i - e_i s) - s(e_i c - s d_{i+1}), \\
d_{i+1}^+ &= c(c d_{i+1} + e_i s) + s(e_i c + s d_i), \\
e_i^+     &= c(c e_i + s d_i) - s(c d_{i+1} + s e_i), \\
e_{i-1}^+ &= c e_{i-1} - s b_{\mathrm{prev}}, \\
e_{i+1}^+ &= c e_{i+1},
\end{aligned}
\f]

and the outgoing bulge is \f$b_{\mathrm{new}} = -s e_{i+1}\f$. The first rotation of a sweep is
generated from the shifted corner \f$(d_{\mathrm{edge}} - \mu, e_{\mathrm{edge}})\f$ with
\f$b_{\mathrm{prev}} = 0\f$, so the shift creates the bulge; each later rotation is generated from
\f$(e_{i-1}, b_{\mathrm{prev}})\f$ and moves the bulge one position.

## steqr: choosing QR versus QL

For a block \f$[\ell, r]\f$ the solver converges the endpoint of smaller magnitude first, as
`dsteqr` does: QL (converge \f$d_\ell\f$ first) if \f$|d_\ell| \le |d_r|\f$, QR (converge
\f$d_r\f$ first) otherwise. The inverted rule loses relative accuracy on graded matrices.

- `steqr_wg` runs the chase in virtual indices oriented so the converged end is the shift end,
  and records the orientation (`ApplyOrder::Forward`/`Backward`) for the eigenvector update.
- `steqr_cta` has only a QL loop. For QR it mirrors the block, \f$T \leftarrow J T J\f$ and
  \f$Q \leftarrow Q J\f$ with \f$J\f$ the reversal on \f$[\ell, r]\f$ (`reverse_block`), runs QL,
  and mirrors back, also on failure. All chunks therefore share one loop nest.

## steqr: the exact 2x2 solve

A block of size 2 is solved analytically (`laev2`). With eigenvectors requested it also yields
the rotation \f$Q_2\f$ with \f$Q_2^T T_2 Q_2 = \operatorname{diag}(\lambda_1, \lambda_2)\f$, which
is accumulated into \f$Q\f$.

## steqr: split structure and book-keeping

The solver iterates on maximal blocks \f$[s, t]\f$ with \f$e_j \neq 0\f$ for
\f$s \le j < t\f$, never on the whole matrix.

- **`steqr_wg`: a global block list.** Each outer pass scans \f$e\f$, records each block as
  `(start_ix, end_ix)` (`end_ix` exclusive), compacts the descriptors of the whole batch with an
  inclusive scan into `(batch, start, end)` triples, then applies one Francis sweep to every block
  in parallel. A sweep can create new zeros, so the list is rebuilt every pass.
- **`steqr_cta`: a left-to-right walk.** Each sub-group partition holds one matrix in registers
  (lane \f$i\f$ owns \f$d_i, e_i\f$). From a running left boundary \f$b\f$ it sets
  \f$e_{b-1} \leftarrow 0\f$, deflates the tail, takes the first zero \f$e_i\f$ with
  \f$i \ge b\f$ as the block end, and continues after it.

Inside a block the QL window \f$[\ell, m]\f$ shrinks after each step: with \f$m\f$ the first zero
in \f$e_\ell, \dots\f$, the eigenvalue \f$d_\ell\f$ has converged if \f$m = \ell\f$, a 2x2 block
is solved exactly if \f$m = \ell + 1\f$, and otherwise one implicit step runs on \f$[\ell, m]\f$.

### CTA control flow on one matrix

```text
budget <- n * max_sweeps
for each block [bb, be] found by the left-to-right walk:
  scale the block if necessary
  rev <- |d_be| < |d_bb|;  if rev: mirror the block (QR as QL)
  l <- bb
  while l <= be:
    up to max_sweeps times:
      deflate [l, be]; m <- first i in [l, be) with e_i = 0, else be
      if m == l:      l += 1; next l
      if m == l + 1:  exact 2x2 on (l, l+1); l += 2; next l
      if budget == 0: fail
      budget -= 1; one implicit QL step on [l, m]
    no progress after max_sweeps: fail
  mirror back if rev; rescale back
  stop on failure
```

## steqr: the CTA update schemes EXP and PG

`steqr_cta` writes one implicit step in one of two equivalent forms, chosen by
`SteqrParams::cta_update_scheme` (default `EXP`).

- **`EXP`** (explicit similarity) applies the scalar band update of
  [the bulge chase](#steqr-the-givens-bulge-chase). QL runs it as a QR-style chase in virtual
  reversed indices inside \f$[\ell, m]\f$. This is the form closest to `steqr_wg`.
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
g_i &= c_i r_i^{\star} - c_{i-1} e_i ;
\end{aligned}
\f]

  QL is the mirrored recurrence.

## steqr: the lockstep flat CTA solver

A warp holds several chunks of \f$P\f$ lanes, one tridiagonal each. In the nested loop above,
chunks realign only at loop exits, so the warp pays the slowest chunk's sweep count per
eigenvalue. `steqr_cta_solve_flat` hoists the sweep out of the nest: each chunk settles (opens a
block, deflates, advances, solves 2x2 blocks) until it has a sweep \f$[\ell, m]\f$ or is done, then
every chunk with a sweep chases in the same pass. The per-chunk operation sequence is unchanged,
so results are bitwise equal to the nested solver. It ships where `kSteqrCtaFlatPays` holds
(\f$P = 16\f$, or \f$P = 8\f$ in FP64); the measurements are in
[Lockstep flat solver](../perf/steqr.md#lockstep-flat-solver).

## steqr: eigenvector accumulation and sorting

With `back_transform = false` the solver starts from \f$Q = I\f$; otherwise it right-applies the
rotations to the caller's matrix. Each rotation on columns \f$k, k+1\f$ does
\f$Q_{:,\{k,k+1\}} \leftarrow Q_{:,\{k,k+1\}} G(c,s)\f$, so
\f$Q = G_1 G_2 \cdots G_p\f$ and \f$Q^T T Q = \Lambda\f$. `steqr_wg` stores a sweep's rotations and
applies them afterwards; `steqr_cta` applies them at once to a shared-memory cache of \f$Q\f$.

After convergence \f$\lambda_i = d_i\f$. With `sort = true` the eigenpairs are sorted in
`sort_order` (ascending by default) and the same permutation is applied to the columns of \f$Q\f$.
