# stebz, stein: bisection and inverse iteration on symmetric tridiagonals {#algo_stebz_stein}

> **Status:** current · checked 2026-11-01.

`stebz` finds selected eigenvalues of a batch of symmetric tridiagonal matrices by bisection on a
Sturm count. `stein` turns them into eigenvectors by inverse iteration, then applies modified
Gram-Schmidt within clusters of close eigenvalues. Each Sturm count costs \f$O(n)\f$, so \f$k\f$
eigenvalues cost \f$O(nk)\f$ per bisection step. Measurements:
[syevx Tier 1: stebz and stein](../perf/syevx.md#syevx-tier-1-stebz-and-stein).

## stebz: the Sturm count

\f$T\f$ is \f$n \times n\f$ symmetric tridiagonal with diagonal \f$d_i\f$, off-diagonal \f$e_i\f$
(\f$e_0 = e_n = 0\f$) and eigenvalues \f$\lambda_0 \le \dots \le \lambda_{n-1}\f$ (0-based, as in
the code). The caller wants an index range \f$[i_l, i_u]\f$ or a value range \f$(v_l, v_u]\f$.
\f$\varepsilon\f$ is the machine epsilon, \f$\sigma\f$ the smallest normal number.

Elimination without pivoting gives \f$T - xI = L D L^T\f$, \f$L\f$ unit lower bidiagonal,
\f$D = \mathrm{diag}(q_1,\dots,q_n)\f$:

\f[
q_1 = d_1 - x, \qquad q_i = d_i - x - \frac{e_{i-1}^2}{q_{i-1}}, \quad i = 2,\dots,n.
\f]

If no \f$q_i\f$ vanishes, Sylvester's law of inertia gives
\f$\#\{i : q_i < 0\} = \#\{j : \lambda_j < x\}\f$ [1, §8.4]. A pivot with
\f$|q_i| < \pi_{\min}\f$ is replaced by \f$-\pi_{\min}\f$, as `dlaebz` does [3], with
\f$\pi_{\min} = \sigma \max(1, \max_i e_i^2)\f$ as in `dstebz` [2]. This keeps the division finite
and counts tiny pivots as non-positive, which defines

\f[
\nu(x) = \#\{\, i : q_i \le 0 \,\} = \#\{\, j : \lambda_j \le x \,\}.
\f]

With monotone (correctly rounded) arithmetic the computed \f$\nu\f$ is nondecreasing in \f$x\f$,
the invariant bisection needs [5].

- recurrence and guard: `src/ops/syev/tridiag/stebz.cc:L47-L58`
- \f$\pi_{\min}\f$: `src/ops/syev/tridiag/stebz.cc:L139`, `L144`, `L154`

## stebz: start interval and tolerance

Every eigenvalue lies in the Gershgorin hull \f$[g_l^0, g_u^0]\f$, widened with the intent that
\f$\nu(g_l) = 0\f$ and \f$\nu(g_u) = n\f$ (not checked in the code; narrower than LAPACK's pad):

\f[
r_i = |e_{i-1}| + |e_i|, \quad
g_l^0 = \min_i (d_i - r_i), \quad g_u^0 = \max_i (d_i + r_i), \quad
g_{l,u} = g_{l,u}^0 \mp \bigl(2\varepsilon \max(|g_l^0|, |g_u^0|) + \sigma\bigr).
\f]

Bisection stops at interval width

\f[
\tau = \begin{cases} \texttt{abstol} & \texttt{abstol} > 0, \\
\varepsilon \max\bigl(|g_l|,|g_u|,1\bigr) & \text{otherwise.} \end{cases}
\f]

- bounds: `src/ops/syev/tridiag/stebz.cc:L133-L144`; widening: `src/ops/syev/tridiag/stebz.cc:L148-L153`
- \f$\tau\f$: `src/ops/syev/tridiag/stebz.cc:L164-L165`

## stebz: bisection

A value range becomes an index range: the eigenvalues in \f$(v_l, v_u]\f$ are
\f$\lambda_j\f$, \f$j \in [\nu(v_l), \nu(v_u) - 1]\f$, so \f$m = \nu(v_u) - \nu(v_l)\f$
(`src/ops/syev/tridiag/stebz.cc:L172-L181`). For each wanted \f$j\f$ bisection keeps
\f$\nu(a) \le j < \nu(b)\f$, i.e. \f$\lambda_j \in (a, b]\f$:

```text
a, b := g_l, g_u
repeat at most max_iterations (128) times:
    if b - a <= tau: stop
    c := a + (b - a)/2
    if c <= a or c >= b: stop                 # no representable midpoint
    if nu(c) >= j + 1: b := c  else: a := c
w_j := a + (b - a)/2
```

- loop: `src/ops/syev/tridiag/stebz.cc:L196-L209`; output and descending order: `src/ops/syev/tridiag/stebz.cc:L211-L213`

One bit per step: at most \f$\lceil \log_2((g_u - g_l)/\tau) \rceil\f$ counts per eigenvalue. Each
computed count is the exact count of a matrix within a few ulps of \f$T\f$, so if the width test
ends the loop, \f$|w_j - \lambda_j| \le \tau/2 + O(\varepsilon\|T\|)\f$ [2, 5]. An early stop
(cap or midpoint guard) is not reported; callers clear `info`
(e.g. `src/ops/syevx/kernels/syevx_direct_subset.cc:L91`).

**Kernel.** One work-group per batch item, \f$\min(256, k)\f$ work-items
(`src/ops/syev/tridiag/stebz.cc:L109-L111`); group reductions give \f$g_l^0, g_u^0,
\max_i e_i^2\f$ (`src/ops/syev/tridiag/stebz.cc:L142-L144`). Each work-item bisects every
`local_size`-th wanted index (`src/ops/syev/tridiag/stebz.cc:L191-L192`); no global workspace.

## stein: inverse iteration

\f$u_i\f$ are the exact eigenvectors. Given \f$w_j\f$ from `stebz`, inverse iteration from
\f$x^{(0)}\f$ is

\f[
(T - \mu_j I)\, y^{(t+1)} = x^{(t)}, \qquad x^{(t+1)} = y^{(t+1)} / \|y^{(t+1)}\|.
\f]

One step scales the \f$u_i\f$ component by \f$1/(\lambda_i - \mu_j)\f$; with
\f$|\lambda_j - \mu_j| = O(\varepsilon\|T\|)\f$ one or two steps suffice for a well-separated
\f$\lambda_j\f$, and the near-singular solve is benign because its error lies mostly along
\f$u_j\f$ [4]. With \f$\|T\|_\infty\f$ clamped below by \f$\sigma\f$
(`src/ops/syev/tridiag/stein.cc:L172-L185`), `stein` runs per vector \f$j\f$:

0. **Shift and start.** \f$\mu_j = w_j\f$, except that if
   \f$|w_j - w_{j-1}| < \varepsilon\|T\|_\infty\f$ then \f$\mu_j = w_j + j\,\varepsilon\|T\|_\infty\f$
   so coincident eigenvalues give distinct systems (`src/ops/syev/tridiag/stein.cc:L213-L219`); the
   offset grows with \f$j\f$, up to \f$(k-1)\varepsilon\|T\|_\infty\f$, weakening the argument
   above for large \f$k\f$. \f$x^{(0)}\f$ is uniform in \f$[-1,1)\f$ from a seeded LCG (`src/ops/syev/tridiag/stein.cc:L93-L96`, `L221-L225`).
1. Normalise \f$x \leftarrow x/\|x\|_2\f$ (`src/ops/syev/tridiag/stein.cc:L228-L233`).
2. Rebuild \f$T - \mu_j I\f$ (`src/ops/syev/tridiag/stein.cc:L237-L244`) and factor it by LU with
   partial pivoting as in `dgttrf` [6]:
   \f$T - \mu_j I = P_1 L_1 P_2 L_2 \cdots P_{n-1} L_{n-1} U\f$, each \f$P_i\f$ the identity or the
   swap of rows \f$i, i+1\f$, each \f$L_i\f$ an elementary unit lower bidiagonal matrix, \f$U\f$
   upper with two superdiagonals (`src/ops/syev/tridiag/stein.cc:L46-L68`). A non-interchanged
   pivot (and \f$u_{nn}\f$) smaller than \f$\varepsilon\|T\|_\infty\f$ becomes
   \f$\pm\varepsilon\|T\|_\infty\f$, cf. `dlagts` with `job = -1` [8].
3. Solve by applying \f$L_i^{-1}P_i\f$ in turn, then back-substituting with \f$U\f$
   (`src/ops/syev/tridiag/stein.cc:L75-L88`); rescale by \f$1/\|y\|_\infty\f$ against overflow
   (`src/ops/syev/tridiag/stein.cc:L249-L254`).

Steps 1-3 repeat a fixed `max_iterations` (default 3) times with no convergence test; the result is
normalised in the 2-norm into column \f$z_j\f$ of \f$Z\f$ (`src/ops/syev/tridiag/stein.cc:L257-L261`).

## stein: reorthogonalisation within clusters

Inverse iteration does not make vectors of close eigenvalues orthogonal [4, 7]. With \f$w\f$
ascending, a gap \f$w_j - w_{j-1} > \theta\|T\|_\infty\f$ (\f$\theta\f$ = `ortho_threshold`,
default \f$10^{-3}\f$ like `dstein`'s `ORTOL` [9]) makes \f$j\f$ the first index \f$c\f$ of a new
cluster. Each vector is orthogonalised against its cluster predecessors by modified Gram-Schmidt:

\f[
z_j \leftarrow z_j - (z_i^T z_j)\, z_i, \quad i = c, \dots, j-1; \qquad z_j \leftarrow z_j / \|z_j\|_2 .
\f]

- gap tolerance: `src/ops/syev/tridiag/stein.cc:L286-L295`; split: `src/ops/syev/tridiag/stein.cc:L305-L309`
- projections: `src/ops/syev/tridiag/stein.cc:L310-L320`; renormalisation: `src/ops/syev/tridiag/stein.cc:L322-L337`

Cost: \f$O(n\,k_c^2)\f$ for a cluster of \f$k_c\f$ vectors.

**Kernels.**

- `SteinIterateKernel`: one work-group per batch item, \f$\min(256, k)\f$ work-items, each handling
  every `local_size`-th vector (`src/ops/syev/tridiag/stein.cc:L146-L148`, `L192`); serial LU and
  solve in global workspace.
- `SteinOrthoKernel`: one work-group per batch item; vectors in sequence, each dot product and AXPY
  parallel over rows.
- Per-item counts \f$m_b\f$: only the first \f$\min(\max(m_b,0),k)\f$ columns are computed; the rest
  are zeroed so uniform-width back-transforms stay exact (`src/ops/syev/tridiag/stein.cc:L189-L200`).

## stebz: differences from LAPACK

| | LAPACK `dstebz` / `dstein` | BatchLAS |
| --- | --- | --- |
| splitting | unreduced blocks | none |
| Gershgorin pad | \f$2.1(n\varepsilon\,\mathrm{tnorm} + 2\pi_{\min})\f$ low, \f$2.1(n\varepsilon\,\mathrm{tnorm} + \pi_{\min})\f$ high [2] | \f$2\varepsilon\max(\lvert g_l^0\rvert,\lvert g_u^0\rvert) + \sigma\f$ |
| stopping width | \f$\max(\texttt{abstol}, \pi_{\min}, \texttt{rtol}\max(\lvert a\rvert,\lvert b\rvert))\f$ [3] | \f$\tau\f$, no `rtol`, no \f$\pi_{\min}\f$ floor |
| count monotonicity | clamped explicitly [3] | monotone arithmetic only |
| LU | once per eigenvalue (`dlagtf`) | rebuilt every step |
| close shifts | \f$x_j = x_{j-1} + 10\lvert\varepsilon x_j\rvert\f$ | \f$w_j + j\varepsilon\lVert T\rVert_\infty\f$ |
| orthogonalisation | inside every iteration | once, after the last step |
| stopping, failures | norm test passed 3 times, at most 5 steps, `ifail` | fixed step count, none reported |

## stebz: where the library calls it

`stebz` and `stein` have no `choice.hh`; they are stages of other ops.

- @ref algo_syevx route `DirectSubset`: `stebz` ascending (required by the clustering), then `stein`
  with per-item counts \f$m_b\f$ only when eigenvectors are wanted
  (`src/ops/syevx/kernels/syevx_direct_subset.cc:L249-L275`, guard `L263`).
- @ref algo_syev routes `blocked` and `two_stage`, eigenvalues only: `stebz` over all indices
  (`src/ops/syev/kernels/syev_blocked.cc:L234-L257`, `L397-L419`;
  `src/ops/syev/kernels/syev_two_stage.cc:L245-L263`).

## tridiagonal_solver: explicit shifted QR

Public low-level API (`include/batchlas/blas/extensions.hh:L695`), no internal route (maintained
solvers: @ref algo_steqr). One 32-wide work-group per item; on the leading block of order \f$p\f$:

- **Deflation**: iterate while \f$|e_{p-1}| > \varepsilon(|d_{p-1}| + |d_p|)\f$, at most 30 steps
  (`src/ops/syev/tridiag/tridiag_solver.cc:L226-L229`); a failure throws `convergence_error`
  (`src/ops/syev/tridiag/tridiag_solver.cc:L273-L274`).
- **Shift**: the Wilkinson shift \f$\mu\f$, the eigenvalue of the trailing \f$2\times 2\f$ block
  closer to \f$d_p\f$ [1, §8.3] (`src/ops/syev/tridiag/tridiag_solver.cc:L230-L231`).
- **Step**: \f$T \leftarrow T - \mu I\f$ (`src/ops/syev/tridiag/tridiag_solver.cc:L62-L63`);
  \f$T = H_1 \cdots H_{p-1} R\f$ with \f$2\times 2\f$ Householder reflectors \f$H_i\f$ on rows
  \f$i, i+1\f$ (`src/ops/syev/tridiag/tridiag_solver.cc:L76-L99`); then
  \f$T \leftarrow R H_1 \cdots H_{p-1} + \mu I\f$ (`src/ops/syev/tridiag/tridiag_solver.cc:L101-L136`)
  and \f$Q \leftarrow Q H_1 \cdots H_{p-1}\f$ (`src/ops/syev/tridiag/tridiag_solver.cc:L233-L235`).

## stebz_stein: references

1. G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press,
   2013. https://www.press.jhu.edu/books/title/10678/matrix-computations
2. LAPACK `dstebz`. https://netlib.org/lapack/explore-html/d4/d48/dstebz_8f_source.html
3. LAPACK `dlaebz`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dlaebz.f
4. I. C. F. Ipsen, "Computing an eigenvector with inverse iteration", *SIAM Review* 39(2),
   254-291, 1997. https://doi.org/10.1137/S0036144596300773
5. J. W. Demmel, I. Dhillon, H. Ren, "On the correctness of some bisection-like parallel
   eigenvalue algorithms in floating point arithmetic", *ETNA* 3, 116-149, 1995. Earlier version:
   "On the correctness of parallel bisection in floating point", UCB/CSD-94-805, 1994.
   https://www2.eecs.berkeley.edu/Pubs/TechRpts/1994/5599.html
6. LAPACK `dgttrf`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgttrf.f
7. E. R. Jessup, I. C. F. Ipsen, "Improving the accuracy of inverse iteration", *SIAM J. Sci.
   Stat. Comput.* 13(2), 550-572, 1992. https://doi.org/10.1137/0913031
8. LAPACK `dlagts`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dlagts.f
9. LAPACK `dstein`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dstein.f
