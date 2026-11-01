# syev: symmetric and Hermitian eigensolver drivers {#algo_syev}

> **Status:** current · checked 2026-11-01.

`syev` computes \f$ A = Z \Lambda Z^H \f$ for batches of dense Hermitian (real symmetric) matrices,
either by tridiagonal reduction, a real tridiagonal solver and a back-transform, or directly by
cyclic Jacobi. `syev_align` rotates eigenvectors of clustered eigenvalues onto a reference basis.
Timings: @ref perf_syev.

## syev: the factorization chain

For \f$ A = A^H \in \mathbb{F}^{n \times n} \f$, \f$ \mathbb{F} \in \{\mathbb{R}, \mathbb{C}\} \f$,
the tridiagonal routes compute

\f[
A = Q\,T\,Q^H, \qquad T = S\,T'\,S^H, \qquad T' = W \Lambda W^T
\quad\Longrightarrow\quad Z = Q\,S\,W,
\f]

with \f$ Q \f$ the Householder product of the reduction (@ref algo_sytrd),
\f$ S = \operatorname{diag}(\sigma_i) \f$ unitary, and \f$ W \f$ the eigenvectors of the real
tridiagonal \f$ T' \f$ (@ref algo_steqr, @ref algo_stedc). With \f$ e_i = t_{i+1,i} \f$, choosing

\f[
\sigma_0 = 1, \qquad \sigma_{i+1} = \sigma_i\, e_i / |e_i| \quad (\sigma_{i+1} = \sigma_i \text{ if } e_i = 0)
\f]

gives \f$ t'_{i+1,i} = \bar\sigma_{i+1} e_i \sigma_i = |e_i| \ge 0 \f$.

- Complex, all routes: `src/ops/syev/kernels/syev_blocked.cc:L205-L228`,
  `src/kernels/common/two_stage_common.hh:L74-L88`, `src/ops/syev/kernels/syev_cta.cc:L254-L266`,
  `src/ops/syev/kernels/syev_cta_fused.cc:L215-L225` (Upper: subdiagonal \f$ \bar e_i \f$).
- Real, `blocked`/`two_stage`: \f$ \sigma_i = \pm 1 \f$ (`syev_blocked.cc:L384-L390`). Real
  `cta`/`cta_fused` pass the signed \f$ e \f$ to `steqr`: \f$ S = I \f$ (`syev_cta.cc:L461`,
  `syev_cta_fused.cc:L226-L228`).
- Lift \f$ S W \f$ (row \f$ i \f$ times \f$ \sigma_i \f$): `syev_blocked.cc:L303`, `L450`;
  `two_stage_common.hh:L140-L144`; `syev_cta.cc:L310`; `syev_cta_fused.cc:L387-L400`.

Every factor is unitary, so \f$ Z, \Lambda \f$ are backward stable when the tridiagonal solver is
[1, §8.3]. Unlike `DSYEV` [3], \f$ A \f$ is not pre-scaled; only `steqr` rescales \f$ T \f$, per
block (`src/ops/syev/steqr/steqr_wg.cc:L148-L152`).

## syev: route selection

| Choice | Pipeline | `can_run` term | \f$ n \f$ |
| --- | --- | --- | --- |
| `cta` | `sytrd_cta` → `steqr` → `ormqx_cta` | `cta_steqr` | \f$ \le 32 \f$ |
| `cta_fused` | the same stages in one kernel | `cta_steqr` | \f$ \le 32 \f$ |
| `jacobi` | cyclic two-sided Jacobi, one kernel | `small` | \f$ \le 32 \f$ |
| `blocked` | `sytrd_blocked` → `stedc` → `ormqr_blocked` | `large` | any |
| `two_stage` | `sy2sb` → `sb2st` → `stedc` → \f$ Q_2 \f$, \f$ Q_1 \f$ | `large` | any |
| `vendor` | `backend::syev_vendor` | `has_vendor`, homogeneous, unmasked unless NETLIB | any |

`native` is GPU, not NETLIB, homogeneous, square, \f$ n \ge 1 \f$; `small` adds \f$ n \le 32 \f$
and a 32-wide sub-group; `cta_steqr` excludes ROCm; `large` adds batch \f$ \ge 1 \f$
(`src/ops/syev/syev.cc:L55-L69`). `tuned/syev.<device>.txt` orders admitted choices by speed. CTA
arms run `steqr` with the Wilkinson shift and 400 sweeps (`syev.cc:L74-L80`).

## syev: the blocked and two-stage pipelines

Both reduce in Lower storage; Upper input is mirrored first, \f$ a_{cr} := \bar a_{rc} \f$ for
\f$ r < c \f$ (`src/ops/syev/kernels/uplo_mirror.cc:L40-L47`).

- **blocked.** \f$ Q = \operatorname{diag}(1, Q_s) \f$; reflector \f$ i \f$ of \f$ Q_s \f$ occupies
  rows \f$ i+2, \dots, n-1 \f$ of column \f$ i \f$ (0-based), so the view shifted down one row is a
  QR reflector block (`src/ops/syev/kernels/syev_blocked.cc:L81-L96`) and
  \f$ Z_{1:,:} := Q_s Z_{1:,:} \f$ is one `ormqr_blocked` call (complex `L307-L330`, real
  `L454-L477`; @ref algo_geqrf).
- **two_stage.** \f$ A = Q_1 B Q_1^H \f$ (band width \f$ k_d = \min(32, n-1) \f$ by default,
  `src/kernels/common/two_stage_common.hh:L29-L36`), \f$ B = Q_2 T Q_2^H \f$ [6];
  \f$ Z = Q_1 (Q_2 (S W)) \f$ right to left (`src/ops/syev/kernels/syev_two_stage.cc:L316-L362`).
- **jobz = N.** `stebz` bisection (@ref algo_stebz_stein) replaces `stedc`, whose merge needs
  eigenvectors (`syev_blocked.cc:L234-L258` complex, `L400-L421` real; `syev_two_stage.cc:L246-L263`).

## syev: the CTA tridiagonal pipelines

One sub-group partition of \f$ P \in \{4, 8, 16, 32\} \f$ lanes owns one matrix. Both reduce in
Upper storage (`cta` mirrors Lower input first: `src/ops/syev/kernels/syev_cta.cc:L137-L162`,
@ref design_known_defects). Upper reduction gives \f$ Q = H_{n-2} \cdots H_0 \f$,
\f$ H_k = I - \tau_k v_k v_k^H \f$, where \f$ v_k \f$ has entries \f$ A_{0:k-1,\,k+1} \f$, a 1 at row
\f$ k \f$ and zeros below: QL-ordered reflectors.

- **cta.** Separate launches: `sytrd_cta`, `steqr` from \f$ W = I \f$, then `ormqx_cta` in QL mode
  forms \f$ Z := Q (S W) \f$ (complex: `src/ops/syev/kernels/syev_cta.cc:L228-L312`, `L386-L396`)
  or \f$ Z := Q W \f$ (real: `L438-L461`, `L533-L544`).
- **cta_fused, real.** `DSYEV`'s order (`DORGTR`, then `DSTEQR` seeded with \f$ Q \f$) [3]:
  `steqr`'s QL/QR sweeps accumulate onto \f$ Q \f$, formed in place. For \f$ k = 0, \dots, n-2 \f$,
  lane \f$ j < k \f$ applies \f$ q_j \leftarrow q_j - \tau_k v_k (v_k^T q_j) \f$ and lane \f$ k \f$
  sets \f$ q_k = \hat e_k - \tau_k v_k \f$ (unit vector \f$ \hat e_k \f$); \f$ q_{n-1} = \hat e_{n-1} \f$
  (`src/ops/syev/kernels/syev_cta_fused.cc:L278-L330`, sweeps `L341-L345`).
- **cta_fused, complex.** The sweeps accumulate \f$ W \f$ on a real tile; then \f$ C := S W \f$ and
  \f$ C \leftarrow H_k C \f$ for \f$ k = 0, \dots, n-2 \f$
  (`src/ops/syev/kernels/syev_cta_fused.cc:L360-L434`).

## syev: cyclic two-sided Jacobi

`jacobi` loads \f$ A \f$ from one triangle, sets \f$ Z = I \f$
(`src/ops/syev/kernels/syev_jacobi_cta.cc:L254-L274`), and rotates \f$ A \leftarrow U^H A U \f$,
\f$ Z \leftarrow Z U \f$ until \f$ A \f$ is diagonal to a relative tolerance [1, Alg. 8.5.3].

**One rotation** on the pair \f$ p < q \f$. The phase \f$ \delta = \bar a_{pq}/|a_{pq}| \f$ makes the
pivot block real with off-diagonal \f$ g = |a_{pq}| \f$ (real data: \f$ \delta = 1 \f$,
\f$ g = a_{pq} \f$); then [1, Alg. 8.5.1]

\f[
\zeta = \frac{a_{qq} - a_{pp}}{2g}, \quad
t = \frac{\operatorname{sign}\zeta}{|\zeta| + \sqrt{1 + \zeta^2}}, \quad
c = \frac{1}{\sqrt{1 + t^2}}, \quad s = t c, \quad
U_{pq} = \begin{bmatrix} c & s \\ -\delta s & \delta c \end{bmatrix}.
\f]

\f$ t \f$ is the smaller-magnitude root of \f$ t^2 + 2\zeta t - 1 = 0 \f$ (\f$ |t| \le 1 \f$). For
\f$ |\zeta| > \varepsilon^{-1/2} \f$, \f$ 1 + \zeta^2 \f$ rounds to \f$ \zeta^2 \f$, so
\f$ t = 1/(2\zeta) \f$ exactly to rounding, and \f$ \zeta^2 \f$ cannot overflow
(`syev_jacobi_cta.cc:L202`, `L320-L338`).

**Relative threshold.** The pair rotates only if

\f[
|a_{pq}| > \mathrm{tol}\,\sqrt{|a_{pp}|\,|a_{qq}|}, \qquad \mathrm{tol} = m_{\mathrm{tol}}\, n\, \varepsilon
\ (m_{\mathrm{tol}} = 1),
\f]

and \f$ |a_{pq}| \f$ exceeds the smallest normal number; a rotation with \f$ s = 0 \f$ is dropped
(`syev_jacobi_cta.cc:L194-L199`, `L312-L314`, `L342`). For positive definite input this rule keeps
small eigenvalues relatively accurate [2]; the absolute values only guard indefinite input.
Cyclic-by-rows Jacobi converges globally, ultimately quadratically [1, §8.5]; for the ordering
below this is assumed, not proved, and a non-converged item reports `info = 1`.

**Parallel ordering** [1, §8.5], [4]. \f$ n \f$ is padded to even \f$ \bar n \f$. Round
\f$ r = 0, \dots, \bar n-2 \f$ pairs index 0 with \f$ (r \bmod (\bar n-1)) + 1 \f$ and, for
\f$ k = 1, \dots, \bar n/2-1 \f$, \f$ ((r+k) \bmod (\bar n-1)) + 1 \f$ with
\f$ ((r-k) \bmod (\bar n-1)) + 1 \f$: \f$ \bar n/2 \f$ disjoint pairs per round, every pair once
per sweep; pairs touching the pad are skipped (`syev_jacobi_cta.cc:L84-L98`, `L118`, `L298`).
Lines of `syev_jacobi_cta.cc`:

```text
for sweep = 1 .. 30:
  for round r = 0 .. nbar-2:
    lane k < nbar/2: (c, s, delta) of pair k, or identity      L306-L359
    no pair rotates: next round                                L367-L369
    A <- A U_r, Z <- Z U_r  (lane = row)                       L374-L411
    A <- U_r^H A  (lane = column); a_pq, a_qp := 0             L420-L462
  no rotation in the sweep: converged                          L468
info := !converged; lambda_j := a_jj, Z(:,j), sorted by rank   L475-L505
```

\f$ U_r \f$ is the direct sum of the round's \f$ 2 \times 2 \f$ blocks. \f$ P \f$ is the smallest of
4, 8, 16, 32 that holds \f$ n \f$ (`L571-L579`).

## syev_align: the clusterwise Procrustes rotation

Given \f$ V_{\mathrm{old}}, V \in \mathbb{F}^{n \times m} \f$, ascending \f$ \lambda \f$ and the
caller's tolerance \f$ \mathrm{tol}_c \f$, columns \f$ j-1, j \f$ share a cluster \f$ \mathcal{C} \f$
when \f$ \lambda_j - \lambda_{j-1} \le \mathrm{tol}_c \cdot \max(|\lambda_0|, |\lambda_{m-1}|) \f$
(`src/ops/syev_align/syev_align.cc:L189-L192`; rationale: @ref design_syev_align). Per cluster,

\f[
\min_{R^H R = I} \| V_{\mathrm{old},\mathcal{C}} - V_{\mathcal{C}} R \|_F:\qquad
M = V_{\mathrm{old}}^H V,\quad M_{\mathcal{C}\mathcal{C}} = X \Sigma Y^H \ \Longrightarrow\ R = Y X^H
\f]

[1, §6.4.1], [5]; then \f$ V_{\mathcal{C}} := V_{\mathcal{C}} R \f$. Lines of
`src/ops/syev_align/syev_align.cc`:

- \f$ M \f$: one `gemm` (`L176-L177`).
- One-sided Jacobi SVD of \f$ G = M_{\mathcal{C}\mathcal{C}} \f$ [2], at most 64 sweeps: for
  columns \f$ g_i, g_k \f$ with \f$ a = \|g_i\|^2 \f$, \f$ b = \|g_k\|^2 \f$,
  \f$ \gamma = g_i^H g_k \ne 0 \f$ and \f$ |\gamma| > \varepsilon\sqrt{ab} \f$, take
  \f$ \zeta = (b-a)/(2|\gamma|) \f$, \f$ t, c, s \f$ as above, \f$ y = g_k \bar\gamma/|\gamma| \f$,
  and set \f$ (g_i, g_k) \leftarrow (c g_i - s y,\ s g_i + c y) \f$; \f$ Y \f$ accumulates the same
  (`L81-L112`). At convergence \f$ G \f$ has been overwritten by \f$ G Y = X \Sigma \f$.
- Columns are normalised (`L113-L129`); those with
  \f$ \sigma \le |\mathcal{C}|\,\varepsilon\,\sigma_{\max} \f$ are replaced by Gram-Schmidt on unit
  vectors, so \f$ X \f$ stays unitary (`L131-L153`).
- \f$ R^H = X Y^H \f$ overwrites \f$ M_{\mathcal{C}\mathcal{C}} \f$ (`L155-L162`);
  \f$ V_{ij} := \sum_{q \in \mathcal{C}} V^{\mathrm{copy}}_{iq} \overline{(R^H)_{jq}} \f$ (`L205-L216`).

## syev: references

1. G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press,
   2013 (§6.4.1, §8.3, §8.5; Algs. 8.5.1, 8.5.3).
2. J. Demmel, K. Veselić, "Jacobi's method is more accurate than QR", *SIAM J. Matrix Anal.
   Appl.* 13(4):1204-1245, 1992. https://doi.org/10.1137/0613074
3. LAPACK reference routine `DSYEV`:
   https://netlib.org/lapack/explore-html/dd/d4c/dsyev_8f_source.html
4. R. P. Brent, F. T. Luk, "The solution of singular-value and symmetric eigenvalue problems on
   multiprocessor arrays", *SIAM J. Sci. Stat. Comput.* 6(1):69-84, 1985.
   https://doi.org/10.1137/0906007
5. P. H. Schönemann, "A generalized solution of the orthogonal Procrustes problem",
   *Psychometrika* 31:1-10, 1966. https://doi.org/10.1007/BF02289451
6. P. Luszczek, H. Ltaief, J. Dongarra, "Two-stage tridiagonal reduction for dense symmetric
   matrices using tile algorithms on multicore architectures", LAWN 244, 2011.
   https://netlib.org/lapack/lawnspdf/lawn244.pdf
