# sytrd: tridiagonal reduction, one-stage and two-stage {#algo_sytrd}

> **Status:** current · checked 2026-11-01.

A Hermitian \f$A\f$ is reduced to a real symmetric tridiagonal \f$T_r\f$ by Householder
similarities, \f$A = Q D T_r D^H Q^H\f$, before a tridiagonal eigensolver (@ref algo_steqr,
@ref algo_stedc) runs. The one-stage path (`sytrd_blocked`) aggregates reflectors LATRD-style [3];
the two-stage path (`sytrd_sy2sb` + `sytrd_sb2st_hh`) goes through a band of width \f$k_d\f$ and
chases bulges [4]. Measurements: @ref perf_sytrd and @ref perf_syev.

Indices are 1-based (the code is 0-based); \f$u\f$ is the unit roundoff. Only `Uplo::Lower` is
implemented: `syev` mirrors `Upper` input into Lower first ([`src/ops/syev/kernels/uplo_mirror.cc:39-47`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/kernels/uplo_mirror.cc#L39-L47)).

## sytrd: the Householder similarity step

A reflector \f$H = I - \tau v v^H\f$ with \f$v_1 = 1\f$ maps \f$(\alpha, x)\f$ to \f$\beta e_1\f$:

\f[
\beta = -\frac{\alpha}{|\alpha|}\,\big\|(\alpha, x)\big\|_2, \qquad
\tau = \frac{\beta - \alpha}{\beta}, \qquad
v = \begin{pmatrix} 1 \\ x / (\alpha - \beta) \end{pmatrix}
\f]

with \f$\alpha/|\alpha| := \operatorname{sign}\alpha\f$ for real data and \f$:= 1\f$ for
\f$\alpha = 0\f$. If the length is 1, or \f$x = 0\f$ (and \f$\operatorname{Im}\alpha = 0\f$ for complex
data), then \f$\tau = 0\f$, \f$\beta = \alpha\f$ ([`src/math-helpers.hh:401-434`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/math-helpers.hh#L401-L434)).

Step \f$i\f$ takes \f$(\alpha, x) = A(i{+}1{:}n, i)\f$, stores \f$\beta\f$ as the subdiagonal
\f$t_i\f$ and \f$v_{2:}\f$ in \f$A(i{+}2{:}n, i)\f$, and replaces
\f$A_{22} = A(i{+}1{:}n, i{+}1{:}n)\f$ by \f$H A_{22} H\f$ as a rank-2 correction [1, §8.3]:

\f[
p = \tau A_{22} v, \qquad
w = p - \tfrac{\tau}{2}\,(v^H p)\, v, \qquad
A_{22} \leftarrow A_{22} - v w^H - w v^H .
\f]

[`src/ops/syev/sytrd/sytrd_cta.cc:239-276`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_cta.cc#L239-L276) runs this (`sytd2`) for \f$n \le 32\f$, one sub-group
per matrix, tile in local memory. With \f$H_i\f$ embedded as
\f$\operatorname{diag}(I_i, H)\f$, \f$Q = H_1 \cdots H_{n-1}\f$ and \f$Q^H A Q = T\f$. Backward
stability [1, §8.3.1], [2, §19.3]: \f$\hat T = \tilde Q^H (A + E) \tilde Q\f$ with \f$\tilde Q\f$
exactly unitary and \f$\|E\|_F \le c(n)\,u\,\|A\|_F\f$, provided nothing over- or underflows:
unlike `xLARFG`, `larfg` does not rescale tiny \f$|\beta|\f$.

## sytrd: complex reflectors and the phase similarity

For complex data \f$\tau = 1 + |\alpha| / \|(\alpha,x)\|_2\f$ is real, so \f$H = H^H\f$ and
\f$\beta\f$, hence \f$t_i\f$, has the phase of \f$-\alpha\f$ (`ZLARFG` instead makes \f$\beta\f$
real, \f$\tau\f$ complex [7]). The diagonal unitary
\f$D = \operatorname{diag}(\delta_1, \dots, \delta_n)\f$,

\f[
\delta_1 = 1, \qquad \delta_{i+1} = \delta_i \frac{t_i}{|t_i|} \quad (\delta_{i+1} = \delta_i \text{ if } t_i = 0),
\f]

gives \f$D^H T D = T_r\f$ with off-diagonal \f$|t_i|\f$. If \f$T_r = Z \Lambda Z^T\f$, the
eigenvectors of \f$T\f$ are \f$D Z\f$.

- One-stage, complex only: \f$D\f$ [`src/ops/syev/kernels/syev_blocked.cc:205-228`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/kernels/syev_blocked.cc#L205-L228),
  \f$Z \leftarrow D Z\f$ [`:287-304`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/kernels/syev_blocked.cc#L287-L304).
- Two-stage, every type: `sytrd_sb2st_hh` stores \f$|t_i|\f$ and the signed \f$t_i\f$
  ([`src/ops/syev/sytrd/sytrd_sb2st_hh.cc:427-430`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st_hh.cc#L427-L430)), so real data get a sign vector \f$D\f$.
  \f$D\f$ [`src/kernels/common/two_stage_common.hh:59-91`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/kernels/common/two_stage_common.hh#L59-L91), \f$Z \leftarrow D Z\f$ [`:121-147`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/kernels/common/two_stage_common.hh#L121-L147).

## sytrd: the LATRD panel and the lazy rank-2k update

The rank-2 updates of \f$b\f$ consecutive steps are deferred and applied as one rank-\f$2b\f$ update
[3], [5]. Panel columns \f$j_0{+}1{:}j_0{+}b\f$ build \f$V, W \in \mathbb{C}^{m \times b}\f$,
\f$m = n - j_0\f$, column by column. Column \f$a_i\f$ of the panel first receives the earlier
columns' share of the update:

\f[
a_i \leftarrow a_i - V_{:,1:i-1} W_{i,1:i-1}^H - W_{:,1:i-1} V_{i,1:i-1}^H
\f]

([`src/ops/syev/sytrd/latrd_lower_panel.cc:270-297`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/latrd_lower_panel.cc#L270-L297)). The reflector is generated ([`:302-357`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/latrd_lower_panel.cc#L302-L357)), and
\f$w_i\f$ is formed against the not-yet-updated \f$A\f$:

\f[
w_i = \tau_i\big(A v_i - V (W^H v_i) - W (V^H v_i)\big), \qquad
w_i \leftarrow w_i - \tfrac{\tau_i}{2} (v_i^H w_i)\, v_i
\f]

(symv [`:368-396`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/latrd_lower_panel.cc#L368-L396), correction [`:398-419`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/latrd_lower_panel.cc#L398-L419), \f$\times\tau_i\f$ [`:421-423`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/latrd_lower_panel.cc#L421-L423), last term [`:425-436`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/latrd_lower_panel.cc#L425-L436)).
After the panel, with \f$n_2 = n - j_0 - b\f$:

\f[
A_{22} \leftarrow A_{22} - V W^H - W V^H
\f]

([`src/ops/syev/sytrd/sytrd_blocked.cc:753-813`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_blocked.cc#L753-L813)). The symv \f$A v_i\f$ stays BLAS-2 and holds about
half of the \f$\tfrac43 n^3\f$ flops [3].

| Piece | Variant | Where chosen |
| --- | --- | --- |
| \f$n \le 32\f$, sub-group 32 | `sytrd_cta` (above) | [`sytrd_blocked.cc:656-660`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_blocked.cc#L656-L660) |
| \f$n \le 64\f$, non-native backend, float (any type if forced), \f$n(n{+}1)\f$ fit in local memory | local-memory `sytd2` | [`sytrd_blocked.cc:662-700`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_blocked.cc#L662-L700) |
| panel | one work-group per matrix; grid kernel when \f$m \ge\f$ `latrd_grid_min_n` (768) and more than one group fits | [`latrd_lower_panel.cc:61-65`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/latrd_lower_panel.cc#L61-L65), [`1097-1102`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/latrd_lower_panel.cc#L1097-L1102) |
| trailing \f$n_2 \le 128\f$ | fused into the panel ([`latrd_lower_panel.cc:444-469`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/latrd_lower_panel.cc#L444-L469)) or `update_vw_lower_small_legacy` ([`sytrd_blocked.cc:120-172`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_blocked.cc#L120-L172)) | [`sytrd_blocked.cc:739`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_blocked.cc#L739), [`:758-763`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_blocked.cc#L758-L763) |
| trailing \f$n_2 > 128\f$ | `syr2k`/`her2k` (float, cfloat with a batched route), else two `gemm` | [`sytrd_blocked.cc:764-812`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_blocked.cc#L764-L812) |

`restore_tridiag_lower` copies \f$d\f$ out and writes the stored \f$t_i\f$ back to both
off-diagonals ([`sytrd_blocked.cc:554-561`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_blocked.cc#L554-L561)).

## sytrd: dense to band (sy2sb)

Stage 1 [4], [8] reduces \f$A\f$ to a band of semibandwidth \f$k_d\f$ (default 32, clamped to
\f$n-1\f$; [`src/kernels/common/two_stage_common.hh:29-36`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/kernels/common/two_stage_common.hh#L29-L36)). For \f$i = 1, 1{+}k_d, \dots\f$, with
panel width \f$p_k = \min(k_d, n-i-k_d+1)\f$:

1. QR of the block below the band, \f$A(i{+}k_d{:}n,\, i{:}i{+}p_k{-}1) = Q R\f$, with
   \f$Q = I - V \mathcal{T} V^H\f$ in compact WY form [6]
   ([`src/ops/syev/sytrd/sytrd_sy2sb.cc:439`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sy2sb.cc#L439); \f$\mathcal{T}\f$ by `larft`, [`:286-288`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sy2sb.cc#L286-L288)).
2. Two-sided update \f$A_{22} \leftarrow Q^H A_{22} Q\f$, \f$A_{22} = A(i{+}k_d{:}n, i{+}k_d{:}n)\f$,
   reading the lower triangle only:

\f[
Y = A_{22} V \mathcal{T}, \qquad
Y \leftarrow Y - \tfrac12 V\big(\mathcal{T}^H V^H Y\big), \qquad
A_{22} \leftarrow A_{22} - V Y^H - Y V^H
\f]

   ([`sytrd_sy2sb.cc:289-297`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sy2sb.cc#L289-L297): four `gemm`, one `symm`/`hemm`, then `syr2k`/`her2k`).
3. A short last panel (\f$p_k < k_d\f$) also applies \f$Q^H\f$ to columns
   \f$i{+}p_k{:}i{+}k_d{-}1\f$ ([`:444-449`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sy2sb.cc#L444-L449)); LAPACK gets the same by factoring the full
   \f$(n{-}i{-}k_d{+}1) \times k_d\f$ block [8].

Only the panel QR (\f$O(n^2 k_d)\f$ flops) is not BLAS-3. The band goes to
\f$AB(1{+}r{-}c, c) = A(r, c)\f$, \f$c \le r \le c{+}k_d\f$ ([`:172-175`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sy2sb.cc#L172-L175), the `xSYTRD_SY2SB` layout [8]).
The reflectors of \f$Q_1\f$ stay in \f$A(k_d{+}1{:}n, 1{:}n{-}k_d)\f$.

## sytrd: band to tridiagonal by bulge chasing (sb2st)

Stage 2 [4], [9] annihilates one column per sweep \f$s = 1, \dots, n-2\f$ with reflectors of length
\f$\le k_d\f$, the `xSB2ST_KERNELS` sequence [10] ([`src/ops/syev/sytrd/sytrd_sb2st_hh.cc:391-412`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st_hh.cc#L391-L412)):

```text
r = [s+1, s+kd]
H = reflector(B(r, s));  B(r,r) <- H B(r,r) H                    type 1
while the next block J = [r_end+1, r_end+kd] exists:
    B(J, r) <- B(J, r) H          fills the bulge
    H = reflector(B(J, r_first))  annihilates the bulge's first column
    B(J, r_first+1:r_end) <- H B(J, r_first+1:r_end)               type 2
    B(J,J) <- H B(J,J) H                                           type 3
    r = J
```

\f$\tau\f$ is real (`larfg`, [`sytrd_sb2st_hh.cc:363`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st_hh.cc#L363)), so the diagonal-block update is the rank-2
step of the first section ([`:252-300`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st_hh.cc#L252-L300)).

- The bulge reaches \f$k_d\f$ rows below the band, so the working band holds \f$2k_d\f$
  subdiagonals ([`sytrd_sb2st_hh.hh:58-62`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st_hh.hh#L58-L62)).
- The schedule depends only on \f$(n, k_d)\f$; the host replays it for the back-transform
  ([`sytrd_sb2st_hh.hh:30-51`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st_hh.hh#L30-L51)).
- One 256-lane work-group chases one matrix; parallelism comes from the window and the batch,
  not from pipelined sweeps as in [4].

`sytrd_sb2st` is the Givens chase (`xSBTRD` [11]; Schwarz [12], Kaufman [13]): `lartg`/`largv`
([`src/ops/syev/sytrd/sytrd_sb2st.cc:263-275`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st.cc#L263-L275)) and `lar2v` ([`:348-357`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st.cc#L348-L357)) per sweep. It discards
\f$Q_2\f$, so `syev` uses it only for eigenvalues, under `BATCHLAS_SYEV_TWO_STAGE_CHASE=givens`
([`two_stage_common.hh:42-45`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/kernels/common/two_stage_common.hh#L42-L45)). The public `sytrd_band_reduction` (successive band reduction [14])
is not called by `syev`.

## sytrd: eigenvector back-transformation

The eigenvectors of \f$A\f$ are \f$X = Q_1 Q_2 D Z\f$ (two-stage) or \f$X = Q D Z\f$ (one-stage),
applied right to left, never formed:

- \f$Z \leftarrow D Z\f$ (phase section).
- \f$Z \leftarrow Q_2 Z\f$, \f$Q_2 = H_1 \cdots H_m\f$, \f$H_m\f$ first, each step
  \f$z \leftarrow z - \tau v (v^H z)\f$. Reflectors of one sweep act on disjoint rows and commute,
  so a sweep is one wave of concurrent updates ([`sytrd_sb2st_hh.hh:85-106`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st_hh.hh#L85-L106),
  [`sytrd_sb2st_hh.cc:530-557`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/sytrd/sytrd_sb2st_hh.cc#L530-L557)).
- \f$Z(k_d{+}1{:}n, :) \leftarrow Q_1 Z(k_d{+}1{:}n, :)\f$ by blocked `ormqr` on the stage-1
  reflectors in place ([`src/ops/syev/kernels/syev_two_stage.cc:341-361`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/kernels/syev_two_stage.cc#L341-L361)). One-stage: the same with
  the reflectors in \f$A(2{:}n, 1{:}n{-}1)\f$ ([`syev_blocked.cc:82-96`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/kernels/syev_blocked.cc#L82-L96), [`:307-330`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/kernels/syev_blocked.cc#L307-L330)).

Route selection between `cta`, `blocked` and `two_stage` is documented in @ref algo_syev.

## sytrd: references

1. G. H. Golub, C. F. Van Loan. *Matrix Computations*, 4th ed., Johns Hopkins University Press,
   2013, §8.3. ISBN 978-1-4214-0794-4. https://www.press.jhu.edu/books/title/10678/matrix-computations
2. N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, §19.3.
   https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/
3. J. J. Dongarra, D. C. Sorensen, S. J. Hammarling. Block reduction of matrices to condensed forms
   for eigenvalue computations. *J. Comput. Appl. Math.* 27(1-2):215-227, 1989.
   https://doi.org/10.1016/0377-0427(89)90367-1
4. A. Haidar, H. Ltaief, J. Dongarra. Parallel reduction to condensed forms for symmetric eigenvalue
   problems using aggregated fine-grained and memory-aware kernels. *SC11*, Article 8, 2011.
   https://doi.org/10.1145/2063384.2063394
5. LAPACK 3.12.1, `DLATRD`.
   https://netlib.org/lapack/explore-html/d1/d45/group__latrd_ga41f90f3c76acde52c34616324bbf1a0f.html
6. R. Schreiber, C. Van Loan. A storage-efficient WY representation for products of Householder
   transformations. *SIAM J. Sci. Stat. Comput.* 10(1):53-57, 1989. https://doi.org/10.1137/0910005
7. LAPACK 3.12.1, `ZLARFG`.
   https://netlib.org/lapack/explore-html/d8/d0d/group__larfg_gad96b19d25ca0cd3fc34292a8bfaf6e4c.html
8. LAPACK 3.12.1, `DSYTRD_SY2SB` source.
   https://netlib.org/lapack/explore-html/d0/d52/dsytrd__sy2sb_8f_source.html
9. LAPACK 3.12.1, `DSYTRD_SB2ST`.
   https://netlib.org/lapack/explore-html/d2/ded/group__hetrd__hb2st_ga5c342791b25b37be0460027bc066e74d.html
10. LAPACK 3.12.1, `DSB2ST_KERNELS`.
    https://netlib.org/lapack/explore-html/d9/dfb/group__hb2st__kernels_gada55e34b69155debaf385324a295221f.html
11. LAPACK 3.12.1, `DSBTRD`.
    https://netlib.org/lapack/explore-html/da/dec/group__hbtrd_ga8c524d4eb120c9e098a2de3ff6c35334.html
12. H. R. Schwarz. Tridiagonalization of a symmetric band matrix. *Numer. Math.* 12(4):231-241,
    1968. https://doi.org/10.1007/BF02162505
13. L. Kaufman. Banded eigenvalue solvers on vector machines. *ACM Trans. Math. Softw.*
    10(1):73-85, 1984. https://doi.org/10.1145/356068.356074
14. C. H. Bischof, B. Lang, X. Sun. A framework for symmetric band reduction. *ACM Trans. Math.
    Softw.* 26(4):581-601, 2000. https://doi.org/10.1145/365723.365735
