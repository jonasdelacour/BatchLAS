# sytrf: symmetric indefinite factorization by Bunch-Kaufman pivoting {#algo_sytrf}

> **Status:** current · checked 2026-11-01.

`sytrf` factors a batch of symmetric, possibly indefinite \f$A = M D M^T\f$, \f$D\f$ block
diagonal with \f$1\times1\f$ and \f$2\times2\f$ blocks, choosing the same pivots as LAPACK
`?sytf2` (Bunch-Kaufman [1, 3]); the blocked route follows `?lasyf` [4] with its own panel
partition. `sytrs` solves with the factor (`?sytrs` [5] or the converted `?sytrs2` [6, 7]);
`sysv` is the two in sequence. Measurements: @ref perf_sytrf.

## sytrf: problem and the block elimination identity

\f$A = A^T \in \mathbb{F}^{n\times n}\f$, \f$\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}\f$ (complex
is symmetric: transposes are never conjugated); the kernels read the lower triangle. One step on
the active matrix, after a symmetric interchange \f$\Pi\f$, with pivot block \f$E\f$ of order
\f$s \in \{1, 2\}\f$:

\f[
\Pi A \Pi^T = \begin{bmatrix} E & C^T \\ C & B \end{bmatrix}
= \begin{bmatrix} I & 0 \\ C E^{-1} & I \end{bmatrix}
  \begin{bmatrix} E & 0 \\ 0 & B - C E^{-1} C^T \end{bmatrix}
  \begin{bmatrix} I & E^{-T} C^T \\ 0 & I \end{bmatrix}.
\f]

Repeating on the Schur complement gives LAPACK's product form [3]

\f[
A = M D M^T, \qquad M = P_1 L_1 P_2 L_2 \cdots P_m L_m,
\f]

\f$P_j\f$ a transposition, \f$L_j\f$ unit lower with multipliers \f$C E^{-1}\f$ in the \f$s\f$
columns of step \f$j\f$. In place, the triangle receives \f$D\f$ and the multipliers;
`ipiv[k] = p+1 > 0` marks a \f$1\times1\f$ step swapping \f$k, p\f$, and
`ipiv[k] = ipiv[k+1] = -(p+1)` a \f$2\times2\f$ step swapping \f$k+1, p\f$
([`src/ops/sytrf/sytrf_device.hh:L270-L277`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L270-L277)).

## sytrf: Bunch-Kaufman pivot selection

At step \f$k\f$, with \f$|z| = |\mathrm{Re}\,z| + |\mathrm{Im}\,z|\f$ (`cabs1`;
[`src/ops/sytrf/sytrf_device.hh:L28-L33`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L28-L33)) and \f$\alpha = (1+\sqrt{17})/8 \approx 0.6404\f$
([`:L189`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L189)):

\f[
\omega_1 = \max_{i>k} |a_{ik}| = |a_{rk}|, \qquad
\omega_r = \max_{j \ge k,\, j \ne r} |a_{rj}|.
\f]

| Test, in order | Pivot | Interchange |
| --- | --- | --- |
| \f$\max(\lvert a_{kk}\rvert, \omega_1) = 0\f$ or \f$a_{kk}\f$ NaN | none; the first sets `info` | none |
| \f$\lvert a_{kk}\rvert \ge \alpha\,\omega_1\f$ | \f$1\times1\f$, \f$a_{kk}\f$ | none |
| \f$\lvert a_{kk}\rvert\,\omega_r \ge \alpha\,\omega_1^2\f$ | \f$1\times1\f$, \f$a_{kk}\f$ | none |
| \f$\lvert a_{rr}\rvert \ge \alpha\,\omega_r\f$ | \f$1\times1\f$, \f$a_{rr}\f$ | \f$k \leftrightarrow r\f$ |
| otherwise | \f$2\times2\f$, \f$E = \left[\begin{smallmatrix} a_{kk} & a_{rk} \\ a_{rk} & a_{rr}\end{smallmatrix}\right]\f$ | \f$k+1 \leftrightarrow r\f$ |

Code: \f$|a_{kk}|\f$ [`:L197`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L197); \f$\omega_1, r\f$ [`:L200-L208`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L200-L208) (first maximum, as `I?AMAX`); the
first two tests [`:L166-L171`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L166-L171); \f$\omega_r\f$ [`:L211-L218`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L211-L218); the rest [`:L176-L179`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L176-L179); `info`
[`:L222`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L222); the interchange [`:L224-L232`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L224-L232). Unlike `I?AMAX`, NaN entries count as \f$+\infty\f$ in
\f$\omega_1, \omega_r\f$ ([`:L155-L158`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L155-L158)); a NaN \f$|a_{rr}|\f$ forces a \f$2\times2\f$ ([`:L179`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L179)).

## sytrf: the 1x1 and 2x2 elimination steps

- **\f$1\times1\f$** (\f$E = a_{kk}\f$, \f$C = c\f$): \f$B \leftarrow B - c\,c^T/a_{kk}\f$ on the
  lower triangle, then \f$\ell = c/a_{kk}\f$ ([`src/ops/sytrf/sytrf_device.hh:L234-L242`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L234-L242)).
- **\f$2\times2\f$** (\f$E = \left[\begin{smallmatrix} d_1 & e \\ e & d_2\end{smallmatrix}\right]\f$):
  the rows \f$[\ell_{jk}\; \ell_{j,k+1}] = [a_{jk}\; a_{j,k+1}]\,E^{-1}\f$ are formed scaled by
  \f$e\f$, as `dsytf2` [3]:

\f[
\delta_{11} = \frac{d_2}{e},\quad \delta_{22} = \frac{d_1}{e},\quad
\eta = \frac{1}{e\,(\delta_{11}\delta_{22} - 1)},\quad
\ell_{jk} = \eta\,(\delta_{11} a_{jk} - a_{j,k+1}),\quad
\ell_{j,k+1} = \eta\,(\delta_{22} a_{j,k+1} - a_{jk}),
\f]

  then \f$a_{ij} \leftarrow a_{ij} - a_{ik} \ell_{jk} - a_{i,k+1} \ell_{j,k+1}\f$ for
  \f$i \ge j \ge k+2\f$, and \f$\ell\f$ overwrites the two columns
  ([`src/ops/sytrf/sytrf_device.hh:L243-L267`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L243-L267)).

## sytrf: stability

- A \f$2\times2\f$ means \f$|d_1| < \alpha|e|\f$, \f$|d_1|\,\omega_r < \alpha e^2\f$,
  \f$|d_2| < \alpha\,\omega_r\f$ with \f$|e| = \omega_1\f$; for real data
  \f$|\det E| > (1-\alpha^2)\,e^2\f$, so \f$E\f$ is safely nonsingular [1].
- Element growth in the reduced matrices and \f$D\f$ is at most
  \f$(1 + 1/\alpha)^{n-1} \approx 2.56^{n-1}\f$ for real data [1] (complex: modestly larger
  constants, `cabs1` being within \f$\sqrt 2\f$ of the modulus); the bound is essentially
  attained [2].
- \f$L\f$ is unbounded, yet the method is normwise backward stable, with backward error of order
  \f$\rho_n u \|A\|\f$ (\f$\rho_n\f$ the growth factor) as for Gaussian elimination with partial
  pivoting [8]; see [9, Ch. 11].

## sytrf: upper storage by index reversal

With \f$J\f$ the reversal \f$i \mapsto n-1-i\f$, the lower triangle of \f$JAJ\f$ is the upper
triangle of \f$A\f$, and \f$JAJ = MDM^T\f$ gives \f$A = (JMJ)(JDJ)(JMJ)^T\f$ with \f$JMJ\f$ unit
upper (`dsytf2`'s upper loop). `Uplo::Upper` runs the lower code through a reversed index map
([`src/ops/sytrf/sytrf_device.hh:L68-L79`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L68-L79)); `ipiv` and `info` hold physical indices.

## sytrf: the blocked panel and uniform-rank update

`?lasyf` [4] factors a panel while accumulating \f$W = L_{:,K} D_{K,K}\f$, so the trailing matrix
is touched once per panel. The panel starts at \f$k_0\f$ with \f$n_b = 31\f$; its first column is
\f$\sigma = k_0\f$, or \f$k_0+1\f$ when the previous panel's last \f$2\times2\f$ took \f$k_0\f$.
Column \f$k\f$ of the partially updated matrix is

\f[
w_k = a_{k:n,k} - L_{k:n,\,\sigma:k}\, W_{k,\,\sigma:k}^T
\f]

([`src/ops/sytrf/sytrf_device.hh:L316-L324`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L316-L324), [`:L328`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L328); candidate column \f$r\f$ at [`:L343`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L343)). The
pivot tests run on \f$w_k\f$; \f$\ell = w_k / w_{kk}\f$ for \f$1\times1\f$ ([`:L372-L374`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L372-L374)), the
scaled \f$E^{-1}\f$ for \f$2\times2\f$ ([`:L375-L392`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L375-L392)). After the panel,
\f$B \leftarrow B - L_{21} W_{21}^T\f$.

- **Uniform rank.** Panels end per item, but the batch shares one `gemm`. The step at
  \f$k_0+n_b-1\f$ is always taken, so a \f$2\times2\f$ may take \f$k_0+n_b\f$ and a panel ends at
  \f$k_0+n_b\f$ or \f$k_0+n_b+1\f$. \f$W\f$ has \f$n_b+1 = 32\f$ columns, uncomputed ones zero
  ([`src/ops/sytrf/sytrf_device.hh:L312-L313`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L312-L313), [`:L407-L408`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L407-L408)): rank 32 for every item.
- **Trailing update.** A native kernel updates the lower triangles of 128-wide diagonal blocks
  with each item's own rank, plus column \f$k_0+n_b\f$ for an item whose panel ended there
  ([`src/ops/sytrf/kernels/sytrf_kernels.cc:L390-L405`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/kernels/sytrf_kernels.cc#L390-L405)); the public `gemm` (`NoTrans`, `Trans`,
  \f$-1, 1\f$, rank 32) does the rest ([`:L459-L475`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/kernels/sytrf_kernels.cc#L459-L475)).
- **Standard form.** An in-panel interchange moves the un-updated column
  ([`src/ops/sytrf/sytrf_device.hh:L365-L367`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L365-L367)) and swaps rows of the finished columns and of
  \f$W\f$ ([`:L368-L369`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L368-L369)), so \f$L_{21}\f$ matches the permuted \f$B\f$; after the update
  `ldl_undo` reverts the row swaps in reverse ([`:L415-L428`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L415-L428)), as `?lasyf`'s last loop.
- **Driver.** Panel, trailing kernel, `gemm` per panel; `?sytf2` on the last \f$\le n_b\f$ columns
  ([`src/ops/sytrf/kernels/sytrf_kernels.cc:L453-L479`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/kernels/sytrf_kernels.cc#L453-L479), tail [`:L345-L352`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/kernels/sytrf_kernels.cc#L345-L352)).

## sytrs: the solve in product form

\f$x = M^{-T} D^{-1} M^{-1} b\f$, applied factor by factor as `dsytrs` [5] (the `wg` route,
[`src/ops/sytrf/kernels/sytrf_kernels.cc:L271`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/kernels/sytrf_kernels.cc#L271); [`src/ops/sytrf/sytrf_device.hh:L448-L503`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L448-L503)).
\f$K_j\f$ is the column set of step \f$j\f$, \f$T_j\f$ the rows after it:

```text
for j = 1..m:                    # M^{-1}, then D^{-1}
  b <- P_j b
  b[T_j] -= L[T_j, K_j] b[K_j]
  b[K_j] <- E_j^{-1} b[K_j]
for j = m..1:                    # M^{-T}
  b[K_j] -= L[T_j, K_j]^T b[T_j]
  b <- P_j b
```

For \f$2\times2\f$, with \f$(v_1, v_2) = b[K_j]\f$, \f$\beta_i = v_i/e\f$ and
\f$\delta = \delta_{11}\delta_{22} - 1\f$: \f$x_1 = (\delta_{11}\beta_1 - \beta_2)/\delta\f$,
\f$x_2 = (\delta_{22}\beta_2 - \beta_1)/\delta\f$ ([`src/ops/sytrf/sytrf_device.hh:L470-L479`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf_device.hh#L470-L479)).

## sytrs: one permutation and two triangular sweeps

`?syconv` [7] moves each later transposition left past the earlier \f$L_j\f$:
\f$L_j P_i = P_i (P_i L_j P_i)\f$ for \f$i > j\f$, with \f$P_i L_j P_i\f$ unit lower of the same
column pattern, so

\f[
A = P \tilde L D \tilde L^T P^T, \qquad P = P_1 P_2 \cdots P_m,
\f]

column \f$j\f$ of \f$\tilde L\f$ being that of \f$L_j\f$ with rows permuted by
\f$P_m \cdots P_{j+1}\f$; the off-diagonal of a \f$2\times2\f$ block at column \f$j\f$ moves to a
separate `e[j]` and is zeroed in \f$\tilde L\f$ ([`src/ops/sytrs/sytrs_device.hh:L54-L70`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs_device.hh#L54-L70); one
gather per column, [`:L114-L138`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs_device.hh#L114-L138)). The solve is `?sytrs2` [6]:

\f[
y = P^T b,\quad \tilde L v = y,\quad z = D^{-1} v,\quad \tilde L^T u = z,\quad x = P u.
\f]

- **`cta`**: sub-groups take right-hand sides round-robin ([`:L213`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs_device.hh#L213)), row \f$t\f$ in lane
  \f$t \bmod 32\f$: \f$\tilde L\f$ sweeps [`:L169-L176`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs_device.hh#L169-L176), [`:L196-L203`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs_device.hh#L196-L203); \f$D^{-1}\f$ [`:L142-L152`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs_device.hh#L142-L152),
  [`:L177-L195`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs_device.hh#L177-L195); \f$P\f$ [`:L73-L86`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs_device.hh#L73-L86), [`:L219-L230`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs_device.hh#L219-L230).
- **`blocked`** ([`src/ops/sytrs/kernels/sytrs_kernels.cc`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/kernels/sytrs_kernels.cc)): gather [`:L209-L216`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/kernels/sytrs_kernels.cc#L209-L216), `trsm`
  (`NoTrans`, `Unit`) [`:L357`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/kernels/sytrs_kernels.cc#L357), \f$D^{-1}\f$ [`:L242-L252`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/kernels/sytrs_kernels.cc#L242-L252), `trsm` (`Trans`) [`:L361`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/kernels/sytrs_kernels.cc#L361), scatter
  [`:L279-L281`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/kernels/sytrs_kernels.cc#L279-L281).

## sysv: factor then solve

`compose` runs the routed `sytrf`, then the routed `sytrs` with `info` as a skip list, so an item
with `info > 0` keeps its \f$B\f$, as `?sysv` ([`src/ops/sysv/sysv.cc:L73-L80`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sysv/sysv.cc#L73-L80)). `cta` fuses both
in one launch: sub-group 0 factors, then, if `info` \f$= 0\f$ ([`src/ops/sytrf/kernels/sytrf_kernels.cc:L148`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/kernels/sytrf_kernels.cc#L148)),
all sub-groups solve ([`:L149-L154`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/kernels/sytrf_kernels.cc#L149-L154)).

## sytrf: routes and kernels

Native routes need a non-NETLIB backend, a GPU with sub-group size 32 and work-groups of at least
32, a homogeneous batch, and for `sytrs` \f$\mathrm{nrhs} \ge 1\f$.

| Op | Route | Kernel | Extra predicate |
| --- | --- | --- | --- |
| sytrf | `cta` | sub-group per matrix, triangle in local memory | \f$n \le 32\f$ |
| sytrf | `wg` | work-group per matrix, in place | |
| sytrf | `blocked` | `ldl_panel`, trailing kernel, `gemm` | |
| sytrf | `vendor` | cuSOLVER / LAPACKE | cuSOLVER: no batch mask |
| sytrs | `cta` | converted, local memory | \f$n \le 128\f$, fits local memory |
| sytrs | `blocked` | converted, two `trsm` | |
| sytrs | `wg` | `?sytrs`, work-group per right-hand side | |
| sytrs | `vendor` | LAPACKE | NETLIB only |
| sysv | `cta` | fused | \f$n \le 32\f$ |
| sysv | `compose` | `sytrf` then `sytrs` | |

Predicates: [`src/ops/sytrf/sytrf.cc:L44-L57`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrf/sytrf.cc#L44-L57), [`src/ops/sytrs/sytrs.hh:L41-L56`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sytrs/sytrs.hh#L41-L56),
[`src/ops/sysv/sysv.cc:L41-L50`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sysv/sysv.cc#L41-L50).

## sytrf: references

1. J. R. Bunch, L. Kaufman, "Some stable methods for calculating inertia and solving symmetric
   linear systems", *Math. Comp.* 31(137):163-179, 1977.
   https://doi.org/10.1090/S0025-5718-1977-0428694-0
2. A. Druinsky, S. Toledo, "The growth-factor bound for the Bunch-Kaufman factorization is tight",
   *SIAM J. Matrix Anal. Appl.* 32(3):928-937, 2011. https://doi.org/10.1137/100801548
3. LAPACK, `DSYTF2`. https://www.netlib.org/lapack/lapack-3.1.1/SRC/dsytf2.f
4. LAPACK, `DLASYF`. https://www.netlib.org/lapack/lapack-3.1.1/SRC/dlasyf.f
5. LAPACK, `DSYTRS`. https://www.netlib.org/lapack/lapack-3.1.1/SRC/dsytrs.f
6. LAPACK 3.3+, `DSYTRS2`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dsytrs2.f
7. LAPACK 3.3+, `DSYCONV`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dsyconv.f
8. N. J. Higham, "Stability of the diagonal pivoting method with partial pivoting", *SIAM J. Matrix
   Anal. Appl.* 18(1):52-65, 1997. https://doi.org/10.1137/S0895479895290371
9. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
   https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/
