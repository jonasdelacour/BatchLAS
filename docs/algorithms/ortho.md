# ortho: orthonormal bases for batches of vector blocks {#algo_ortho}

> **Status:** current · checked 2026-11-01.

`ortho` replaces each block of \f$k\f$ vectors of length \f$m\f$ by an orthonormal basis of its span,
in place, by CholeskyQR (once, twice, or shifted three times), SVQB (once or twice), CGS2 or
Householder QR. A second entry point first projects out an external orthonormal basis.
Measurements: @ref perf_ortho.

## ortho: problem and notation

Given \f$A \in \mathbb{F}^{m \times k}\f$ with \f$\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}\f$ and
\f$k \le m\f$ (asserted, not thrown: `src/ops/ortho/ortho.cc:L166`), find \f$Q\f$ with
\f$Q^H Q = I_k\f$ and \f$\operatorname{range}(Q) = \operatorname{range}(A)\f$. Notation:

- \f$u\f$ is the unit roundoff and \f$\varepsilon = 2u\f$ is machine epsilon (`numeric_limits::epsilon()`).
- \f$\kappa(A) = \sigma_1(A)/\sigma_k(A)\f$, \f$C = A^H A\f$ is the Gram matrix, and
  \f$p = mk + k(k+1)\f$.
- `transA = NoTrans`: the vectors are the columns of `A`; otherwise the rows, handled through the
  companion transpose `inv_trans` (`src/ops/ortho/ortho.cc:L163-L165`). Complex rows need
  `ConjTrans`: with `Trans` the Gram matrix is \f$AA^T\f$ (`src/ops/ortho/ortho.cc:L195`) and the
  CGS2 projection is wrong.
- `OrthoAlgorithm` selects the method. The default is `Chol2` (`include/batchlas/blas/extensions.hh:L149`).

## ortho: CholeskyQR and CholeskyQR2

If \f$A\f$ has full column rank, \f$C\f$ is Hermitian positive definite, so \f$C = R^H R\f$
(Cholesky) and

\f[
A = Q R, \qquad Q = A R^{-1}, \qquad Q^H Q = R^{-H} C R^{-1} = I .
\f]

- Gram (`src/ops/ortho/ortho.cc:L186-L196`): `syrk` writes only the lower triangle. It is used
  for real `T` with \f$k \le 64\f$ (float) or \f$k \le 128\f$ (double) where
  `select::level3_tile_substitution_enabled` holds. Otherwise a full GEMM is used, which
  `BATCHLAS_ORTHO_GRAM=gemm` forces.
- `potrf` gives \f$C = LL^H\f$ (\f$L = R^H\f$). `trsm` then sets \f$A \leftarrow A L^{-H}\f$, or
  \f$A \leftarrow L^{-1}A\f$ for rows (`src/ops/ortho/ortho.cc:L210-L215`).
- `Chol2` runs the pass twice (`src/ops/ortho/ortho.cc:L422-L425`).
- One pass loses orthogonality as \f$O(\kappa(A)^2 u)\f$, and `potrf` fails once
  \f$\kappa(A)^2 u \gtrsim 1\f$ [1], [2].
- The second pass starts from \f$\kappa(Q_1) \approx 1\f$. If \f$8\,\kappa(A)\sqrt{pu} \le 1\f$, it
  reaches \f$\|Q^H Q - I\|_F = O(u)\f$ and \f$\|A - QR\|_F = O(u)\|A\|\f$ [1].

## ortho: shifted CholeskyQR3

`ShiftChol3` first factors a shifted Gram matrix [2], then applies CholeskyQR2:

\f[
R_0^H R_0 = C + s I, \qquad
s = 11\,p\,\varepsilon\,\|A\|_F^2, \qquad
A \leftarrow A R_0^{-1}.
\f]

- If \f$6k^2u\kappa < 1\f$, \f$mku \le 1/64\f$, \f$k(k+1)u \le 1/64\f$ and
  \f$11\,p\,u\|A\|_2^2 \le s \le \|A\|_2^2/100\f$, `potrf` completes and
  \f$\kappa(AR_0^{-1}) \lesssim u^{-1/2}\f$, inside CholeskyQR2's range; overall
  \f$\kappa(A) \le u^{-1}/(96p)\f$ suffices [2].
- The code uses \f$\varepsilon\f$ and \f$\|A\|_F^2 = \operatorname{tr}(C)\f$ where [2] uses \f$u\f$
  and \f$\|A\|_2^2\f$ ([2] allows \f$\|A\|_F\f$). The shift is therefore at least twice that of [2],
  and it can exceed \f$\|A\|_2^2/100\f$ once \f$2200\,pku > 1\f$, which float can reach.
- Gram `src/ops/ortho/ortho.cc:L294`. One work-group of \f$k\f$ items sums the diagonal into
  \f$\operatorname{tr}(C)\f$ and adds \f$s\f$ (`src/ops/ortho/ortho.cc:L306-L309`). Shifted
  factor/solve `src/ops/ortho/ortho.cc:L313-L314`, CholeskyQR2 tail
  `src/ops/ortho/ortho.cc:L315-L316`.

## ortho: SVQB

SVQB [3], [4] orthonormalises through the eigendecomposition of the scaled Gram matrix, so its second
factor is not triangular. With \f$D = \operatorname{diag}(d_j)\f$, \f$d_j = C_{jj}^{-1/2}\f$:

\f[
D C D = U \Theta U^H, \qquad Q = A D U \Theta^{-1/2}, \qquad
Q^H Q = \Theta^{-1/2} U^H (D C D) U \Theta^{-1/2} = I .
\f]

| Step | Formula | Lines |
| --- | --- | --- |
| Gram (always GEMM: all of \f$C\f$ is scaled) | \f$C = A^H A\f$ | `src/ops/ortho/ortho.cc:L326` |
| Scaling | \f$d_j = C_{jj}^{-1/2}\f$, or \f$0\f$ if \f$C_{jj} \le\f$ `numeric_limits::min()` | `src/ops/ortho/ortho.cc:L342-L344` |
| Scale | \f$C_{ij} \leftarrow d_i C_{ij} d_j\f$ | `src/ops/ortho/ortho.cc:L360-L363` |
| Eigensolve | `syev`, ascending \f$\theta\f$, \f$U\f$ overwrites \f$C\f$ | `src/ops/ortho/ortho.cc:L367` |
| Clamp, form \f$D U \Theta^{-1/2}\f$ | \f$C_{ij} \leftarrow d_i U_{ij} \max(\theta_j, \tau)^{-1/2}\f$, \f$\tau = \varepsilon\,\theta_k = \varepsilon \max_j\theta_j\f$ | `src/ops/ortho/ortho.cc:L382-L386` |
| Apply | \f$Q = A\,(D U \Theta^{-1/2})\f$ into scratch | `src/ops/ortho/ortho.cc:L390` |
| Copy back | \f$A \leftarrow Q\f$, conjugated for rows | `src/ops/ortho/ortho.cc:L404-L413` |

- The clamp is the "replace" rule of [4, Alg. 4]. That rule leaves the threshold free;
  \f$\tau = \varepsilon\theta_k\f$ is the value BatchLAS picks. \f$Q^HQ = I\f$ holds only if no clamp
  fires, and \f$d_j = 0\f$ gives a zero column.
- Like CholeskyQR it works on \f$A^H A\f$, so `SVQB2` repeats it. For columns the two passes alternate
  between scratch and `A` instead of copying back (`src/ops/ortho/ortho.cc:L442-L452`).

## ortho: classical Gram-Schmidt with reorthogonalisation

`CGS2` builds \f$Q\f$ one vector at a time, with \f$Q_{i-1} = [q_1, \dots, q_{i-1}]\f$. Each vector
gets two classical projections ("twice is enough") and a normalisation:

```text
for i = 1 .. k:
  if i > 1: repeat 2: c <- Q_{i-1}^H a_i ; a_i <- a_i - Q_{i-1} c     (two gemv)
  q_i <- a_i / ||a_i||_2                                              (norm kernel)
```

- Projections: `src/ops/ortho/ortho.cc:L249-L256`. For rows, the row is conjugated before and after
  (`src/ops/ortho/ortho.cc:L248`, `src/ops/ortho/ortho.cc:L257`).
- Norm: squares into local memory (all \f$m\f$ of them, `src/ops/ortho/ortho.cc:L261`), a group
  reduction, a divide (`src/ops/ortho/ortho.cc:L277-L287`). No zero guard: an exactly zero residual
  gives `NaN`; a numerically dependent one is normalised to roundoff noise, silently.
- One CGS pass loses orthogonality as \f$O(\kappa(A)^2 u)\f$. With reorthogonalisation the loss is
  \f$O(u)\f$, provided \f$A\f$ is numerically full rank [5].

## ortho: Householder QR

`geqrf` writes \f$A = QR\f$ with \f$Q = H_1 \cdots H_k\f$ and \f$H_i = I - \tau_i v_i v_i^H\f$. `orgqr`
then overwrites \f$A\f$ with the first \f$k\f$ columns of \f$Q\f$ [6]
(`src/ops/ortho/ortho.cc:L431-L432`).

- Rows go through the unconjugated transpose \f$A^T \in \mathbb{F}^{m \times k}\f$, both ways
  (`src/ops/ortho/ortho.cc:L55`, `src/ops/ortho/ortho.cc:L430`, `src/ops/ortho/ortho.cc:L433`). The
  rows of \f$Q^T\f$ are orthonormal and span the row space.
- Householder QR is backward stable. The computed \f$Q\f$ is orthonormal to \f$O(u)\f$ for any
  \f$\kappa(A)\f$ [7, §19.3], and it spans \f$\operatorname{range}(A)\f$ when \f$A\f$ has full rank.

## ortho: orthogonalising against an external basis

Given \f$M \in \mathbb{F}^{m \times n_M}\f$ with \f$M^H M = I\f$ and \f$k + n_M \le m\f$ (else
`batchlas::invalid_argument`, `src/ops/ortho/ortho.cc:L500-L502`), each of `iterations` passes
(default 2) is a block classical Gram-Schmidt step followed by the inner method:

\f[
B = M^H A, \qquad A \leftarrow A - M B, \qquad A \leftarrow \operatorname{ortho}(A).
\f]

Loop: `src/ops/ortho/ortho.cc:L519-L528`. It is the `ortho` procedure of [3] ([4, Alg. 3]) with a fixed
pass count instead of a tolerance; two passes match BCGS2 [8].

## ortho: choosing an algorithm

| `OrthoAlgorithm` | Valid for \f$\kappa(A)\f$ up to | Kernels |
| --- | --- | --- |
| `Cholesky` | \f$\ll u^{-1/2}\f$ (orthogonality \f$\kappa^2 u\f$) | Gram, `potrf`, `trsm` |
| `Chol2` (default) | \f$1/(8\sqrt{pu})\f$ [1] | 2 x the above |
| `ShiftChol3` | \f$u^{-1}/(96p)\f$ [2] | shift kernel + 3 x the above |
| `SVQB`, `SVQB2` | as CholeskyQR, CholeskyQR2 (Gram-based) | GEMM, 3 scaling kernels, `syev`, GEMM, copy-back |
| `CGS2` | numerically full rank [5] | \f$2(k-1)\f$ `gemv` pairs, \f$k\f$ norm kernels (+ \f$2(k-1)\f$ conj kernels for complex rows) |
| `Householder` | any (orthonormality) [7] | `geqrf`, `orgqr` (+ 2 transpose copies for rows) |

- Routing: there is no `choice.hh`. The one override sends every request on a non-GPU device to
  Householder. It applies in the op, in its size query and in the external-basis entry
  (`src/ops/ortho/ortho.cc:L77-L79`, `src/ops/ortho/ortho.cc:L160-L162`,
  `src/ops/ortho/ortho.cc:L467-L469`, `src/ops/ortho/ortho.cc:L486-L488`). The reason is given in
  @ref perf_ortho.
- Launches: one work-group per matrix for the shift, SVQB scalings, CGS2 norm and copy-back;
  flat element-wise for transpose and conjugation (`src/ops/ortho/ortho.cc:L50`,
  `src/ops/ortho/ortho.cc:L221`). All skip masked-off items except the SVQB scalings, which write
  only scratch (`src/ops/ortho/ortho.cc:L333-L335`).
- Workspace carve: `src/ops/ortho/ortho.cc:L93-L140`.

## ortho: references

1. Y. Yamamoto, Y. Nakatsukasa, Y. Yanagisawa, T. Fukaya, "Roundoff error analysis of the
   CholeskyQR2 algorithm," *Electron. Trans. Numer. Anal.* 44 (2015) 306-326.
   https://etna.ricam.oeaw.ac.at/vol.44.2015/pp306-326.dir/pp306-326.pdf
2. T. Fukaya, R. Kannan, Y. Nakatsukasa, Y. Yamamoto, Y. Yanagisawa, "Shifted Cholesky QR for
   computing the QR factorization of ill-conditioned matrices," *SIAM J. Sci. Comput.* 42(1)
   (2020) A477-A503. doi:10.1137/18M1218212; https://arxiv.org/abs/1809.11085
3. A. Stathopoulos, K. Wu, "A block orthogonalization procedure with constant synchronization
   requirements," *SIAM J. Sci. Comput.* 23(6) (2002) 2165-2182. doi:10.1137/S1064827500370883
4. J. A. Duersch, M. Shao, C. Yang, M. Gu, "A robust and efficient implementation of LOBPCG,"
   *SIAM J. Sci. Comput.* 40(5) (2018) C655-C676. doi:10.1137/17M1129830;
   https://arxiv.org/abs/1704.07458
5. L. Giraud, J. Langou, M. Rozložník, J. van den Eshof, "Rounding error analysis of the classical
   Gram-Schmidt orthogonalization process," *Numer. Math.* 101 (2005) 87-100.
   doi:10.1007/s00211-005-0615-4; https://icl.utk.edu/files/publications/2005/icl-utk-250-2005.pdf
6. LAPACK 3.12, `DGEQRF` and `DORGQR` reference documentation.
   https://www.netlib.org/lapack/explore-html/d0/da1/group__geqrf_gade26961283814bb4e62183d9133d8bf5.html,
   https://netlib.org/lapack/explore-html/d4/dfc/group__ungqr_ga9f4abcfe1543a5d6d90fcd4dd21f12f0.html
7. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, ch. 19
   "QR Factorization". doi:10.1137/1.9780898718027
8. J. L. Barlow, A. Smoktunowicz, "Reorthogonalized block classical Gram-Schmidt," *Numer. Math.*
   123 (2013) 395-423. doi:10.1007/s00211-012-0496-2; https://arxiv.org/abs/1108.4209
