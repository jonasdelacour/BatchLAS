# sygvd: generalized symmetric-definite eigenproblem {#algo_sygvd}

> **Status:** current · checked 2026-11-01.

`sygvd` solves batches of \f$A x = \lambda B x\f$ and its two relatives by the Cholesky method:
factor \f$B\f$, reduce to a standard Hermitian problem with `sygst`, solve it with `syev`, and map
the eigenvectors back with one triangular solve or product [1], [2]. Routes and timings are in
@ref perf_sygvd.

## sygvd: problem and reduction

\f$A = A^H\f$ and \f$B = B^H\f$ positive definite, both \f$n \times n\f$; `itype` picks the problem.
`potrf` gives \f$B = F F^H\f$ with \f$F = L\f$ (`uplo = Lower`) or \f$F = U^H\f$ (`Upper`,
\f$B = U^H U\f$). Substituting \f$x = F^{-H} y\f$ or \f$x = F y\f$ turns each problem into
\f$C y = \lambda y\f$ with Hermitian \f$C\f$ [2]:

| itype | problem | \f$C\f$ (`sygst`) | back-transform | normalization |
|---|---|---|---|---|
| 1 | \f$A x = \lambda B x\f$ | \f$F^{-1} A F^{-H}\f$ | \f$x = F^{-H} y\f$ | \f$X^H B X = I\f$ |
| 2 | \f$A B x = \lambda x\f$ | \f$F^H A F\f$ | \f$x = F^{-H} y\f$ | \f$X^H B X = I\f$ |
| 3 | \f$B A x = \lambda x\f$ | \f$F^H A F\f$ | \f$x = F y\f$ | \f$X^H B^{-1} X = I\f$ |

For itype 1, multiplying \f$A x = \lambda F F^H x\f$ by \f$F^{-1}\f$ gives
\f$F^{-1} A F^{-H} (F^H x) = \lambda F^H x\f$, and orthonormal \f$Y\f$ from `syev` gives
\f$X^H B X = Y^H Y = I\f$ [3]. \f$C\f$ is similar to \f$B^{-1} A\f$, \f$AB\f$ or \f$BA\f$
(itypes 1, 2, 3) and Hermitian, so the eigenvalues are real.

**Stability.** For itype 1 the standard backward-error bound is a multiple of \f$\kappa_2(B)\,u\f$,
so accuracy can degrade with an ill-conditioned \f$B\f$; [4] shows the error is often much smaller,
and that iterative refinement repairs the unstable cases. `potrf` does not pivot, as in LAPACK.

## sygvd: the compose pipeline

```text
compose(A, B, itype, jobz, uplo):
  F, finfo  := potrf(B)                  # skipped by sygvd_factored
  F_b       := I  where finfo_b != 0     # finite data, not a NaN partial factor
  A         := C = sygst(A, F, itype)    # uplo triangle only
  (w, Y)    := syev(C)                   # A := Y when jobz = V
  X         := F^{-H} Y  (itype 1, 2)  |  F Y  (itype 3)
  info_b    := n + finfo_b  where finfo_b != 0   # as ?sygvd [3]
```

| step | lines |
|---|---|
| `potrf` | [`src/ops/sygvd/kernels/sygvd_compose.cc:84`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/kernels/sygvd_compose.cc#L84) |
| identity (`uplo` triangle) on failed items | [`src/ops/sygvd/kernels/sygvd_compose.cc:86-87`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/kernels/sygvd_compose.cc#L86-L87) → [`src/ops/sygvd/kernels/sygvd_native.cc:40-47`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/kernels/sygvd_native.cc#L40-L47) |
| `sygst`, then `syev` | [`src/ops/sygvd/kernels/sygvd_compose.cc:89-90`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/kernels/sygvd_compose.cc#L89-L90) |
| \f$X = F^{-H} Y\f$: `trsm`, Left, \f$L^H\f$ or \f$U\f$ | [`src/ops/sygvd/kernels/sygvd_compose.cc:92-93`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/kernels/sygvd_compose.cc#L92-L93) |
| \f$X = F Y\f$: out-of-place `trmm` (\f$L\f$ or \f$U^H\f$), then copy | [`src/ops/sygvd/kernels/sygvd_compose.cc:94-99`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/kernels/sygvd_compose.cc#L94-L99) |
| \f$\mathrm{info}_b = n + i\f$ (unless factored) | [`src/ops/sygvd/kernels/sygvd_compose.cc:102-104`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/kernels/sygvd_compose.cc#L102-L104) → [`src/ops/sygvd/kernels/sygvd_native.cc:60-61`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/kernels/sygvd_native.cc#L60-L61) |

## sygst: the unblocked diagonal kernel

`sygst_diag` reduces a \f$k_b \times k_b\f$ block (\f$k_b \le 32\f$) per item, one work-item
per column. Since \f$(F^{-1} A)^H = A F^{-H}\f$, two identical column passes give \f$C\f$:

\f[
Z = F^{-1} A, \qquad C = F^{-1} Z^H \quad (\text{itype } 1); \qquad
Z = F^H A, \qquad C = F^H Z^H \quad (\text{itypes } 2, 3).
\f]

- Load the named triangle and \f$F\f$ into SLM: [`src/ops/sygst/kernels/sygst_diag.cc:133-143`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_diag.cc#L133-L143);
  mirror column \f$j\f$, \f$\Re\f$ on the diagonal: [`src/ops/sygst/kernels/sygst_diag.cc:151-154`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_diag.cc#L151-L154).
- Forward substitution \f$c_i \leftarrow (c_i - \sum_{p<i} F_{ip} c_p) / F_{ii}\f$:
  [`src/ops/sygst/kernels/sygst_diag.cc:86-93`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_diag.cc#L86-L93), [`107`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_diag.cc#L107).
- Top-down product \f$c_i \leftarrow F_{ii} c_i + \sum_{p>i} \overline{F_{pi}}\, c_p\f$ (in place:
  \f$c_p\f$, \f$p > i\f$, is still unwritten): [`src/ops/sygst/kernels/sygst_diag.cc:94-104`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_diag.cc#L94-L104), [`108`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_diag.cc#L108).
- Pass 2 reads row \f$j\f$ of \f$Z\f$ conjugated as its column: [`src/ops/sygst/kernels/sygst_diag.cc:163-164`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_diag.cc#L163-L164).
- Store the `uplo` triangle (optionally the full Hermitian result to workspace):
  [`src/ops/sygst/kernels/sygst_diag.cc:171-180`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_diag.cc#L171-L180).

LAPACK's `dsygs2` forms the same \f$C\f$ with BLAS-2 updates [6].

## sygst: the blocked itype-1 recurrence

Partition with \f$F = L\f$, diagonal block order \f$k_b = \min(nb, n - k)\f$ at offset \f$k\f$
([`src/ops/sygst/kernels/sygst_blocked.cc:144`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L144)), trailing order \f$m\f$:

\f[
A = \begin{bmatrix} A_{11} & A_{21}^H \\ A_{21} & A_{22} \end{bmatrix}, \quad
L = \begin{bmatrix} L_{11} & 0 \\ L_{21} & L_{22} \end{bmatrix}, \quad
C = L^{-1} A L^{-H}.
\f]

Matching blocks of \f$A = L C L^H\f$ gives \f$C_{11} = L_{11}^{-1} A_{11} L_{11}^{-H}\f$ and
\f$C_{21} = L_{22}^{-1}(A_{21} L_{11}^{-H} - L_{21} C_{11})\f$. With
\f$P = A_{21} L_{11}^{-H} - \tfrac12 L_{21} C_{11}\f$,

\f[
A_{22} - P L_{21}^H - L_{21} P^H = L_{22} C_{22} L_{22}^H,
\f]

so the trailing block is the same problem of order \f$m\f$ with factor \f$L_{22}\f$ [5].
Per block (`blocked_range`), with \f$\hat C_{11}\f$ a full Hermitian copy of \f$C_{11}\f$ (`H` in the code):

| step | operation | lines |
|---|---|---|
| \f$C_{11}\f$ and \f$\hat C_{11}\f$ | diagonal kernel | [`src/ops/sygst/kernels/sygst_blocked.cc:147`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L147) |
| \f$P = A_{21} L_{11}^{-H}\f$ | `trsm` Right | [`src/ops/sygst/kernels/sygst_blocked.cc:155`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L155) |
| \f$P \mathrel{-}= \tfrac12 L_{21} \hat C_{11}\f$ (now \f$P\f$ as above) | `gemm` | [`src/ops/sygst/kernels/sygst_blocked.cc:156`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L156) |
| \f$A_{22} \mathrel{-}= P L_{21}^H + L_{21} P^H\f$ | `her2k` / `syr2k` | [`src/ops/sygst/kernels/sygst_blocked.cc:157`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L157) |
| \f$P \mathrel{-}= \tfrac12 L_{21} \hat C_{11}\f$ | `gemm` | [`src/ops/sygst/kernels/sygst_blocked.cc:158`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L158) |
| \f$C_{21} = L_{22}^{-1} P\f$ | `trsm` Left | [`src/ops/sygst/kernels/sygst_blocked.cc:159`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L159) |

The two half-steps make the update a symmetric rank-\f$2k_b\f$ one; LAPACK's `symm` against
\f$C_{11}\f$ becomes a `gemm` against \f$\hat C_{11}\f$. `Upper` runs the conjugate-transposed
schedule on \f$A_{12}\f$, \f$U_{12}\f$: [`src/ops/sygst/kernels/sygst_blocked.cc:160-167`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L160-L167).

## sygst: the blocked itype-2/3 recurrence

\f$C = L^H A L\f$ is built left to right. If the leading \f$k \times k\f$ block holds
\f$L_{11}^H A_{11} L_{11}\f$, appending block row 2 (\f$A_{11}, L_{11}\f$ of order \f$k\f$;
\f$A_{21}, L_{21}\f$ of size \f$k_b \times k\f$; \f$A_{22}, L_{22}\f$ of order \f$k_b\f$) gives

\f[
\begin{bmatrix} L_{11}^H A_{11} L_{11} + L_{21}^H A_{21} L_{11} + L_{11}^H A_{21}^H L_{21} + L_{21}^H A_{22} L_{21} & \ast \\
L_{22}^H (A_{21} L_{11} + A_{22} L_{21}) & L_{22}^H A_{22} L_{22} \end{bmatrix}.
\f]

With \f$W = A_{21} L_{11} + \tfrac12 A_{22} L_{21}\f$ the corner update is the rank-\f$2k_b\f$ term
\f$W^H L_{21} + L_{21}^H W\f$ [5]; at \f$k = n\f$ the leading block is \f$C\f$. \f$\hat A_{22}\f$
is a full Hermitian copy of \f$A_{22}\f$:

| step | operation | lines |
|---|---|---|
| \f$\hat A_{22} = A_{22}\f$ | `sygst_mirror` | [`src/ops/sygst/kernels/sygst_blocked.cc:175`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L175) |
| \f$W = A_{21} L_{11}\f$ (out of place) | `trmm` Right | [`src/ops/sygst/kernels/sygst_blocked.cc:180`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L180) |
| \f$W \mathrel{+}= \tfrac12 \hat A_{22} L_{21}\f$ | `gemm` | [`src/ops/sygst/kernels/sygst_blocked.cc:181`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L181) |
| \f$A_{11} \mathrel{+}= W^H L_{21} + L_{21}^H W\f$ | `her2k` / `syr2k` | [`src/ops/sygst/kernels/sygst_blocked.cc:182`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L182) |
| \f$W \mathrel{+}= \tfrac12 \hat A_{22} L_{21}\f$ | `gemm` | [`src/ops/sygst/kernels/sygst_blocked.cc:183`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L183) |
| \f$A_{21} = L_{22}^H W\f$ | `trmm` Left | [`src/ops/sygst/kernels/sygst_blocked.cc:184`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L184) |
| \f$A_{22} = L_{22}^H A_{22} L_{22}\f$ | diagonal kernel | [`src/ops/sygst/kernels/sygst_blocked.cc:196`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L196) |

`Upper`: [`src/ops/sygst/kernels/sygst_blocked.cc:185-194`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L185-L194).

## sygst: block sizes and batching

- `nb` \f$\in \{32, 64, 128, 256\}\f$ is a route field ([`src/ops/sygst/sygst.cc:39`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/sygst.cc#L39),
  [`src/ops/sygst/choice.hh:31`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/choice.hh#L31)). A diagonal block with \f$k_b > 32\f$ is reduced by the same
  recurrence with inner block size \f$\max(32, k_b/4)\f$: [`src/ops/sygst/kernels/sygst_blocked.cc:138`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L138).
- \f$n \le \min(nb, 32)\f$ is one diagonal-kernel launch with no workspace
  (`single_block`, [`src/ops/sygst/kernels/sygst_blocked.cc:28`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L28)).
- Every child is a batched call over a pass of items: [`src/ops/sygst/kernels/sygst_blocked.cc:207-215`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L207-L215).
  The pass size is potrf's L2 chunk, at least 4 items per CU, and a single block never splits:
  [`src/ops/sygst/kernels/sygst_blocked.cc:37-47`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_blocked.cc#L37-L47).

## sygst: the two-sided alternative

`twosided` mirrors \f$A\f$ to full storage ([`src/ops/sygst/kernels/sygst_twosided.cc:37`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_twosided.cc#L37)) and
applies \f$F\f$ from both sides with whole-matrix level-3 calls, about twice the blocked flops:

- itype 1: \f$A \leftarrow F^{-1} A\f$, then \f$A \leftarrow A F^{-H}\f$ (two `trsm`):
  [`src/ops/sygst/kernels/sygst_twosided.cc:39-40`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_twosided.cc#L39-L40).
- itypes 2, 3: \f$W = F^H A\f$, then \f$A = W F\f$ (two `trmm`, \f$n \times n\f$ workspace):
  [`src/ops/sygst/kernels/sygst_twosided.cc:48-49`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/kernels/sygst_twosided.cc#L48-L49).

## sygvd: routes and predicates

*Homogeneous* means every item has the same \f$n \ge 1\f$ ([`src/ops/sygst/sygst.cc:45`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/sygst.cc#L45)).

| op | route | `can_run` | lines |
|---|---|---|---|
| sygst | `blocked:nb` | homogeneous, GPU, compiled `nb`, `max_wg >= 128`, diagonal-kernel SLM fits | [`src/ops/sygst/sygst.cc:49-53`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/sygst.cc#L49-L53) |
| sygst | `twosided` | homogeneous | [`src/ops/sygst/sygst.cc:54`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/sygst.cc#L54) |
| sygst | `vendor` | homogeneous, NETLIB only, vendor present (LAPACKE per item) | [`src/ops/sygst/sygst.cc:55`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/sygst.cc#L55) |
| sygvd | `compose` | homogeneous; no batch mask on NETLIB | [`src/ops/sygvd/sygvd.cc:47`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/sygvd.cc#L47) |
| sygvd | `vendor` | homogeneous, cuSOLVER or NETLIB, vendor present, not `sygvd_factored`; a mask only on NETLIB | [`src/ops/sygvd/sygvd.cc:48-49`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/sygvd.cc#L48-L49) |

A GPU without a table borrows the nearest one ([`src/select/select.cc:545-552`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/select/select.cc#L545-L552)). The CPU never
borrows ([`src/select/select.cc:554-555`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/select/select.cc#L554-L555)); with no runnable row it takes the last resort `vendor, blocked,
twosided` ([`src/ops/sygst/choice.hh:35`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygst/choice.hh#L35)) or `vendor, compose` ([`src/ops/sygvd/choice.hh:22`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/sygvd/choice.hh#L22)).
All children are public routed ops, so `BATCHLAS_<OP>_ROUTE` pins apply inside them.

## sygvd: references

1. E. Anderson et al., "Generalized Symmetric Definite Eigenproblems", *LAPACK Users' Guide*,
   3rd ed., SIAM, 1999. https://netlib.org/lapack/lug/node54.html
2. R. S. Martin, J. H. Wilkinson, "Reduction of the symmetric eigenproblem \f$Ax = \lambda Bx\f$
   and related problems to standard form", *Numer. Math.* 11:99-110, 1968.
   https://doi.org/10.1007/BF02165306
3. LAPACK, `DSYGVD` reference source. https://www.netlib.org/lapack/double/dsygvd.f
4. P. I. Davies, N. J. Higham, F. Tisseur, "Analysis of the Cholesky Method with Iterative
   Refinement for Solving the Symmetric Definite Generalized Eigenproblem", *SIAM J. Matrix Anal.
   Appl.* 23(2):472-493, 2001. https://doi.org/10.1137/S0895479800373498
5. LAPACK, `DSYGST` reference source. https://www.netlib.org/lapack/double/dsygst.f
6. LAPACK, `DSYGS2` reference source. https://www.netlib.org/lapack/double/dsygs2.f
