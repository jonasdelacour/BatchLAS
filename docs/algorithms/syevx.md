# syevx: partial eigensolvers by Rayleigh-Ritz projection {#algo_syevx}

> **Status:** current · checked 2026-11-01.

`syevx` returns \f$k\f$ eigenpairs of batched Hermitian matrices; its iterative routes (LOBPCG,
Chebyshev-filtered subspace iteration) extract Ritz pairs from a small orthonormal trial space.
`lanczos` builds the full Lanczos tridiagonal; `ritz_values` evaluates Rayleigh quotients.
Dispatch and traps: @ref design_syevx; measurements: @ref perf_syevx.

## syevx: problem and routes

For Hermitian \f$A \in \mathbb{C}^{n \times n}\f$ with eigenvalues
\f$\lambda_1 \le \dots \le \lambda_n\f$, find the \f$k\f$ largest or smallest eigenpairs, an index
range \f$[i_l, i_u]\f$ or a value range \f$(v_l, v_u]\f$. `syevx_select_algorithm`
(`src/ops/syevx/syevx.cc:L246-L330`) picks the route by a fixed rule; there is no `tuned/` table.

| Route | Method | Auto picks it when |
|---|---|---|
| `Direct` | full `syev`, then copy the wanted columns (`src/ops/syevx/kernels/syevx_direct.cc:L128`, `:L162`) | \f$n \le 64\f$, eigenvalues only, complex, or below the `DirectSubset` gate (`syevx.cc:L311-L329`) |
| `DirectSubset` | `sy2sb`, `sb2st` (@ref algo_sytrd), `stebz`, `stein` (@ref algo_stebz_stein), back-transforms on \f$k\f$ columns (`src/ops/syevx/kernels/syevx_direct_subset.cc:L188-L318`) | dense real with vectors, \f$n \ge 1024\f$, \f$n \cdot \mathrm{batch} \ge 2^{17}\f$ (`syevx.cc:L322-L325`) |
| `LOBPCG` | below | CSR input (`syevx.cc:L295-L298`) or explicit request |
| `Filtered` | below | explicit request only |

The iterative routes converge to an end of the spectrum. A non-extremal range throws for sparse
input or an explicit LOBPCG/Filtered request; a `BATCHLAS_SYEVX_ALGORITHM` pin falls back to
`Direct` with a warning (`syevx.cc:L262-L289`).

## syevx: Rayleigh-Ritz on an orthonormal trial space

For \f$S \in \mathbb{C}^{n \times s}\f$ with \f$S^H S = I\f$, Rayleigh-Ritz [5, Alg. 4.5] solves

\f[
G = S^H A S = Z \Theta Z^H, \qquad X = S Z
\f]

(Filtered: `src/ops/syevx/kernels/syevx_filtered.cc:L344-L353`). The Ritz pairs
\f$(\theta_j, x_j)\f$ satisfy \f$r_j = A x_j - \theta_j x_j \perp \operatorname{span} S\f$, and for
\f$\|x_j\| = 1\f$ some eigenvalue obeys \f$|\lambda - \theta_j| \le \|r_j\|\f$ [5, Cor. 3.3]: the
residual certifies the pair.

- \f$\operatorname{ortho}(Y; M)\f$ repeats `ortho_iterations` times (default 2):
  \f$Y \leftarrow Y - M(M^H Y)\f$, then orthonormalise \f$Y\f$ (default CholeskyQR2; @ref algo_ortho)
  (`src/ops/ortho/ortho.cc:L519-L528`), so \f$G\f$ is a standard eigenproblem for batched `syev`.
- Both routes carry \f$g = \max(2, \lfloor k/4 \rfloor)\f$ guard columns beyond the \f$k\f$ wanted
  (`extra_directions` overrides \f$g\f$); only the \f$k\f$ wanted columns are tested.

## syevx: the LOBPCG iteration

LOBPCG [1] takes the trial space \f$\operatorname{span}\{X, P, R\}\f$: the current Ritz block, the
previous directions and the (preconditioned) residuals; each step is the Rayleigh-Ritz optimum
over that space (locally optimal block preconditioned CG). The block width
is \f$b = k + g\f$, or \f$\max(k, \lfloor n/3 \rfloor)\f$ when \f$3(k+g) > n\f$ with the default
\f$g\f$ (`src/ops/syevx/kernels/syevx_lobpcg.cc:L80-L91`). References below are to this file.

| Step | Formula | Lines |
|---|---|---|
| residual | \f$R = AX - X\Theta\f$ | `:L688` |
| precondition (optional) | \f$R \leftarrow (\tilde L \tilde U)^{-1}R\f$ (ILU(k), @ref algo_iluk; smallest pairs only), or \f$r_{ij} \leftarrow r_{ij}/(a_{ii} - \theta_j)\f$ (`JacobiShifted`; `Jacobi`: \f$\theta_j = 0\f$, positive diagonal only), skipping near-zero denominators | `:L256-L260`, `:L865`, `:L978-L1000` |
| orthogonalise | \f$R \leftarrow \operatorname{ortho}(R;\,[X\;P])\f$, so \f$S = [X\;P\;R]\f$ is orthonormal | `:L1009` |
| one block matvec | \f$AR = A R\f$ (`gemm` or `spmm`) | `:L1059-L1066` |
| project | \f$G = S^H (AS)\f$ | `:L1073` |
| solve | \f$G = Z \Theta Z^H\f$; \f$Z_X\f$ = the \f$b\f$ columns at the wanted end, \f$Z_X^{PR}\f$ its rows below the first \f$b\f$ | `:L1128-L1133`, `:L1143-L1144` |
| new Ritz block | \f$X \leftarrow S Z_X\f$, \f$AX \leftarrow (AS) Z_X\f$ | `:L1163-L1164` |
| new directions | \f$C_P = \operatorname{ortho}\big(\left[\begin{smallmatrix}0 \\ Z_X^{PR}\end{smallmatrix}\right];\,Z_X\big)\f$, \f$P \leftarrow S C_P\f$, \f$AP \leftarrow (AS) C_P\f$ | `:L1149-L1158`, `:L1166-L1170` |

- \f$C_P\f$ is the Hetmaniuk-Lehoucq basis [2]. Because \f$S\f$ is orthonormal, \f$[X\;P]\f$ stays
  orthonormal without forming the ill-conditioned difference \f$X_{i+1} - X_i\f$.
- The first iteration has no \f$P\f$: \f$R\f$ is orthogonalised against \f$X\f$ only, copied into
  the \f$P\f$ slot (the matvec writes \f$AP\f$), and \f$s = 2b\f$ (`:L590`, `:L1009`,
  `:L1041-L1061`).
- A null trial column would give \f$G\f$ a spurious zero eigenvalue; its diagonal entry is set to
  \f$\pm(\|G\|_F + 1)\f$, on the unwanted side (`:L1115-L1121`).
- Start: a Philox-random \f$X\f$ (`:L520`); for the largest pairs, four block power steps
  \f$X \leftarrow A\,\operatorname{ortho}(X)\f$ (`:L531-L541`); then one Rayleigh-Ritz on \f$X\f$
  (`:L544-L575`).

**Convergence.** Column \f$j\f$ is converged when

\f[
\rho_j = \frac{\|r_j\|}{\|A x_j\| + |\theta_j|\,\|x_j\|} \le \tau_L = \max(\tau_{\mathrm{abs}}, \tau_{\mathrm{rel}})
\f]

(`:L715-L716`, `:L595`). Since \f$\|A x_j\| \le \|A\|\,\|x_j\|\f$, this implies the
backward-stable test \f$\|r_j\| / ((\|A\| + |\theta_j|)\|x_j\|) \le \tau_L\f$ of [3] without a norm
estimate. The iterate with
the smallest \f$\max_j \rho_j\f$ is snapshotted and returned (`:L732-L737`); an item is done when
all \f$k\f$ snapshot residuals pass (`:L770-L777`).

**Kernels.** Every step is a batched call on \f$n \times 3b\f$ blocks. The projected `syev` is
pinned to the CTA solver (@ref algo_steqr) when the shape fits, except on NETLIB, under an outer
`syev` pin, or with the vendor projected solver opted in (`:L183-L190`, `:L387-L388`).

## syevx: Chebyshev-filtered subspace iteration

`Filtered` is subspace iteration in which the power step is a Chebyshev polynomial filter, followed
by Rayleigh-Ritz [5, Alg. 7.5], [7]. The block width is \f$m = \min(k + g, n)\f$
(`src/ops/syevx/kernels/syevx_filtered.cc:L113-L116`); references below are to this file.

The Chebyshev polynomial \f$C_d\f$ satisfies \f$|C_d(y)| \le 1\f$ on \f$[-1, 1]\f$ and
\f$C_d(y) = \cosh(d \operatorname{arccosh} y)\f$ for \f$y \ge 1\f$. For the largest pairs (Ritz
values ascending) the damped interval is \f$[\ell, u]\f$ with \f$\ell\f$ the Gershgorin lower bound
(`:L213-L246`, `:L303-L309`) and \f$u = \theta_{m-k}\f$, the first unwanted Ritz value; the
smallest pairs mirror this with the Gershgorin upper bound (`:L452-L462`). With
\f$c = (\ell + u)/2\f$, \f$e = (u - \ell)/2\f$ (`:L464-L466`) and \f$\theta_{\mathrm{far}}\f$ the
extreme Ritz value, the filter is

\f[
p_d(t) = \frac{C_d\big((t - c)/e\big)}{C_d\big((\theta_{\mathrm{far}} - c)/e\big)} .
\f]

\f$Y = p_d(A) X\f$ is formed by the scaled three-term recurrence of [4, Alg. 3.1], which keeps
\f$p_d(\theta_{\mathrm{far}}) = 1\f$ instead of letting \f$C_d\f$ overflow:

\f[
\begin{aligned}
\sigma_1 &= \frac{e}{\theta_{\mathrm{far}} - c}, &
Y_0 &= X, \quad Y_1 = \frac{\sigma_1}{e}(A - cI)X, \\
\sigma_{j+1} &= \frac{1}{2/\sigma_1 - \sigma_j}, &
Y_{j+1} &= \frac{2\sigma_{j+1}}{e}(A - cI)Y_j - \sigma_j \sigma_{j+1} Y_{j-1}
\end{aligned}
\f]

(`:L475`, `:L557-L559`, `:L573`, `:L579`, `:L621-L624`). An eigencomponent at \f$\lambda\f$ is
scaled by \f$p_d(\lambda)\f$, \f$|p_d(\lambda)| = |C_d(y_\lambda)| / C_d(y_{\mathrm{far}})\f$,
\f$y_t = |t - c|/e\f$: at most \f$1/C_d(y_{\mathrm{far}})\f$ inside \f$[\ell, u]\f$, growing like
\f$e^{d \operatorname{arccosh} y_\lambda}\f$ outside, which sets the convergence rate [5, §7.5].

```text
X <- ortho(random);  (Theta, X) <- RayleighRitz(X)                      L315-L355
repeat
  stop if ||A x_j - theta_j x_j|| <= tau_F * max(|theta_j|, 1) for the k wanted j   L389-L405
  l, u, c, e, sigma_1 from the Gershgorin end and the Ritz values          L452-L476
  Y <- p_d(A) X                    (d block matvecs)                       L578-L632
  X <- ortho(Y);  (Theta, X) <- RayleighRitz(X)                            L635-L637
```

- \f$\tau_F = \tau_{\mathrm{abs}}\f$, or \f$100\varepsilon\f$ if that is zero (`:L356-L359`).
- Degree: `BATCHLAS_SYEVX_FILTER_DEGREE`, else `filter_degree`, else 10 (`:L58`, `:L118-L124`;
  opt-in residual-driven degree: @ref design_syevx). With \f$x = \operatorname{arccosh} y\f$, the
  cap below keeps \f$\cosh(d x_{\mathrm{far}})/\cosh(d x_{\mathrm{edge}}) \le e^{d(x_{\mathrm{far}} - x_{\mathrm{edge}})} \le \varepsilon^{-1/4}\f$
  (empirical exponent), so CholeskyQR stays full rank:

\f[
d_{\mathrm{cap}} = \left\lfloor \frac{-\tfrac14 \ln \varepsilon}{\operatorname{arccosh} y_{\mathrm{far}} - \operatorname{arccosh} y_{\mathrm{edge}}} \right\rfloor ,
\f]

  with \f$y_{\mathrm{edge}}\f$ for the innermost wanted Ritz value and its term 0 when
  \f$y_{\mathrm{edge}} \le 1\f$ (`:L481-L500`). The batch takes the minimum cap, keeping GEMM
  shapes uniform (`:L592-L595`).

## lanczos: full tridiagonalisation by the Lanczos recurrence

For real symmetric \f$A\f$ and a unit \f$v_0\f$, the Lanczos recurrence [6, §10.1], [5, §6.3] builds
an orthonormal basis \f$V_j = [v_0, \dots, v_{j-1}]\f$ of the Krylov space
\f$\mathcal{K}_j(A, v_0)\f$ with

\f[
A V_j = V_j T_j + \beta_{j-1} v_j \mathbf{e}_j^T, \qquad
T_j = \operatorname{tridiag}(\beta_{0:j-2},\, \alpha_{0:j-1},\, \beta_{0:j-2}) .
\f]

Step \f$j\f$ in `src/ops/lanczos/lanczos.cc`:

\f[
\alpha_j = v_j^T A v_j, \qquad
w = A v_j - \beta_{j-1} v_{j-1} - \alpha_j v_j, \qquad
\beta_j = \|w\|, \qquad
v_{j+1} = w / \beta_j
\f]

(`:L154-L158`, `:L168-L174`, `:L176-L186`).

- In floating point the \f$v_j\f$ lose orthogonality as Ritz values converge [6, §10.3]. Every
  `reorthogonalization_iterations`-th step (default 2) and at the last step, \f$v_j\f$ gets two
  passes of block classical Gram-Schmidt against \f$V_j\f$, each followed by renormalisation
  (`:L117-L125`). That may flip \f$v_j\f$'s sign, so \f$\beta_{j-1}\f$ takes the sign of
  \f$v_{j-1}^T A v_j\f$ (`:L159-L167`).
- The recurrence runs to \f$j = n - 1\f$. If \f$\dim \mathcal{K}_n(A, v_0) = n\f$ (distinct
  eigenvalues, \f$v_0\f$ not orthogonal to any eigenvector), \f$A = V_n T_n V_n^T\f$ in exact
  arithmetic. `steqr` gives \f$T_n = Q \Lambda Q^T\f$ (`:L206-L214`, @ref algo_steqr), the
  eigenvectors are \f$V_n Q\f$ (`:L216-L222`), then, if `sort_enabled`, a sort (`:L224-L230`).
  Real types only (`:L299`).
- Otherwise \f$\beta_j = 0\f$ exactly at some \f$j < n-1\f$ (e.g. \f$A = I\f$), and with no
  breakdown test this divides by zero (`:L184`).
- Kernels: one work-group per item per step (`:L139-L187`); \f$A v_j\f$ is a two-column
  `gemm`/`spmm` (`:L127-L133`).

## ritz_values: per-column Rayleigh quotients

For a dense block \f$V = [v_1, \dots, v_k]\f$, per item,

\f[
\theta_j = \operatorname{Re} \frac{v_j^H A v_j}{v_j^H v_j}
\f]

(`src/ops/ritz_values/ritz_values.cc:L105-L144`): \f$AV\f$ by `gemm` or `spmm` (`:L53-L59`), then
work-groups grid-stride over (item, column) pairs (`:L96`), reducing numerator (`:L105-L119`) and
denominator (`:L125-L138`). These are Ritz values of \f$\operatorname{span} V\f$ only when the
columns are already Ritz vectors; otherwise use \f$V^H A V\f$ and `syev` (@ref guide_ritz_values).

## syevx: references

1. A. V. Knyazev, "Toward the optimal preconditioned eigensolver: locally optimal block
   preconditioned conjugate gradient method", *SIAM J. Sci. Comput.* 23(2):517-541, 2001.
   [doi:10.1137/S1064827500366124](https://doi.org/10.1137/S1064827500366124)
2. U. Hetmaniuk, R. Lehoucq, "Basis selection in LOBPCG", *J. Comput. Phys.* 218:324-332, 2006.
   [doi:10.1016/j.jcp.2006.02.007](https://doi.org/10.1016/j.jcp.2006.02.007)
3. J. A. Duersch, M. Shao, C. Yang, M. Gu, "A robust and efficient implementation of LOBPCG",
   *SIAM J. Sci. Comput.* 40(5):C655-C676, 2018.
   [doi:10.1137/17M1129830](https://doi.org/10.1137/17M1129830),
   [arXiv:1704.07458](https://arxiv.org/abs/1704.07458)
4. Y. Zhou, Y. Saad, "A Chebyshev-Davidson algorithm for large symmetric eigenproblems",
   *SIAM J. Matrix Anal. Appl.* 29(3):954-971, 2007.
   [doi:10.1137/050630404](https://doi.org/10.1137/050630404)
5. Y. Saad, *Numerical Methods for Large Eigenvalue Problems*, 2nd ed., SIAM, 2011.
   [doi:10.1137/1.9781611970739](https://doi.org/10.1137/1.9781611970739),
   [author's copy](https://www-users.cse.umn.edu/~saad/eig_book_2ndEd.pdf)
6. G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press,
   2013, ch. 10. [publisher](https://www.press.jhu.edu/books/title/10678/matrix-computations)
7. Y. Zhou, Y. Saad, M. L. Tiago, J. R. Chelikowsky, "Self-consistent-field calculations using
   Chebyshev-filtered subspace iteration", *J. Comput. Phys.* 219:172-184, 2006.
   [doi:10.1016/j.jcp.2006.03.017](https://doi.org/10.1016/j.jcp.2006.03.017)
