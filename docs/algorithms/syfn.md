# syfn: matrix functions of Hermitian matrices {#algo_syfn}

> **Status:** current · checked 2026-11-01.

`syfn` evaluates \f$ f(A) \f$ for a batch of Hermitian matrices by the spectral method: one
eigendecomposition, a diagonal scaling of the eigenvectors, and one level-3 product.
`syfn_from_eigen` is the second half on its own, for eigenpairs the caller already holds. The API
contract and test plan are in @ref perf_syfn.

## syfn: problem and the spectral definition

Let \f$ A \in \mathbb{C}^{n \times n} \f$ be Hermitian. By the spectral theorem

\f[
A = V \Lambda V^H, \qquad V^H V = I, \qquad \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n),
\ \lambda_j \in \mathbb{R}.
\f]

For a scalar function \f$ f \f$ defined on the spectrum, the primary matrix function is [1, §1.2]

\f[
f(A) = V f(\Lambda) V^H = \sum_{j=1}^{n} f(\lambda_j)\, v_j v_j^H .
\qquad (1)
\f]

It is independent of the basis chosen inside a repeated eigenspace.

The supported functions (`SyfnKind`, [`include/batchlas/blas/functions/syfn.hh:L24-L30`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/include/batchlas/blas/functions/syfn.hh#L24-L30)):

| Kind | \f$ f(\lambda) \f$ | Domain \f$ \mathcal{D} \f$ | Sign of \f$ f \f$ |
| --- | --- | --- | --- |
| `InvSqrt` | \f$ \lambda^{-1/2} \f$ (Löwdin \f$ S^{-1/2} \f$ [3]) | \f$ \lambda > 0 \f$ | \f$ > 0 \f$ |
| `Sqrt` | \f$ \lambda^{1/2} \f$ | \f$ \lambda \ge 0 \f$ | \f$ \ge 0 \f$ |
| `Exp` | \f$ e^{\lambda} \f$ | \f$ \mathbb{R} \f$ | \f$ > 0 \f$ |
| `Inverse` | \f$ \lambda^{-1} \f$ | \f$ \lambda \ne 0 \f$ | either |
| `Weights` | caller's \f$ u_j \f$, one per eigenpair | \f$ \mathbb{R} \f$ | either |

## syfn: two factored forms of the sum

Write (1) as a scaled product. With a diagonal \f$ G = \operatorname{diag}(g_1, \dots, g_n) \f$ and
\f$ Y = V G \f$ (column \f$ j \f$ of \f$ V \f$ times \f$ g_j \f$):

- **Rank-\f$ k \f$ form**, for \f$ f \ge 0 \f$. Take \f$ g_j = \sqrt{f(\lambda_j)} \f$; then
  \f[
  f(A) = (V G)(V G)^H = Y Y^H .
  \qquad (2)
  \f]
  Only one triangle is needed, so a `herk` (real: `syrk`) [4] does it in about half the flops of
  a general product. The scales are \f$ \lambda^{-1/4} \f$, \f$ \lambda^{1/4} \f$ and
  \f$ e^{\lambda/2} \f$, computed as `1/sqrt(sqrt(w))`, `sqrt(sqrt(w))` and `exp(w*0.5)`
  ([`src/ops/syfn/kernels/syfn_kernels.cc:L45-L47`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/kernels/syfn_kernels.cc#L45-L47)).
- **General form**, for \f$ f \f$ of either sign. Take \f$ g_j = f(\lambda_j) \f$; then
  \f[
  f(A) = (V G) V^H = Y V^H ,
  \qquad (3)
  \f]
  one `gemm` with \f$ V^H \f$ as the conjugate-transposed operand. The scales are
  \f$ 1/\lambda \f$ and \f$ u_j \f$ ([`src/ops/syfn/kernels/syfn_kernels.cc:L48-L49`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/kernels/syfn_kernels.cc#L48-L49)).

The choice is by kind, not by the values (`rank_k_form`, [`src/ops/syfn/syfn.cc:L31`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/syfn.cc#L31)): `Weights`
always takes (3), even when every \f$ u_j \ge 0 \f$. `Exp` is unscaled: \f$ F \f$ overflows once
\f$ e^{\lambda_{\max}} \f$ exceeds the overflow threshold, and `info` does not flag it.

## syfn: out-of-domain eigenvalues

An eigenvalue outside \f$ \mathcal{D} \f$ gets \f$ g_j = 0 \f$
([`src/ops/syfn/kernels/syfn_kernels.cc:L31-L38`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/kernels/syfn_kernels.cc#L31-L38), [`L43`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/kernels/syfn_kernels.cc#L43)). With
\f$ J = \{ j : \lambda_j \in \mathcal{D} \} \f$ and
\f$ \mathcal{R} = \operatorname{span}\{ v_j : j \in J \} \f$ the result is

\f[
F = \sum_{j \in J} f(\lambda_j)\, v_j v_j^H = f(A|_{\mathcal{R}})\, P_{\mathcal{R}},
\qquad P_{\mathcal{R}} = \sum_{j \in J} v_j v_j^H ,
\f]

i.e. \f$ f \f$ on \f$ \mathcal{R} \f$, zero on its complement. For `Inverse` with exact zeros in
the computed spectrum (of \f$ A + E \f$, see below) this is the Moore-Penrose pseudo-inverse
\f$ V \Lambda^{+} V^H \f$ [5]: it satisfies Penrose's four equations because \f$ V \f$ is unitary.
No threshold is applied, so a rounding-level \f$ \lambda_j \ne 0 \f$ is inverted as is.
`info[b]` counts \f$ |\{1..n\} \setminus J| \f$, or is \f$ n + \f$ syev's status when the
eigensolver failed ([`src/ops/syfn/kernels/syfn_kernels.cc:L64-L70`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/kernels/syfn_kernels.cc#L64-L70)); then \f$ F_b \f$ is
unspecified.

## syfn: the algorithm

```text
per batch item b, all items in one launch per step:
  V := A                                   full copy; A untouched
  (V, w) := syev(V, uplo)                  reads the uplo triangle; w ascending
  info_b                                   as above
  g_j := sqrt(f(w_j)) | f(w_j) | 0         (2) | (3) | out of domain
  Y := V diag(g)
  F := Y Y^H  (herk/syrk, lower)    if kind in {InvSqrt, Sqrt, Exp}
  F := Y V^H  (gemm)                otherwise
  F_ij := conj(F_ji) for i < j;  F_ii := Re F_ii
```

The last line, `hermitize(Lower)`, makes \f$ F \f$ exactly Hermitian (`herk` fills one triangle;
a `gemm`'s two triangles round independently): it copies \f$ \bar F_{ij} \f$, \f$ i > j \f$, into
\f$ F_{ji} \f$ and zeroes \f$ \operatorname{Im} F_{ii} \f$ ([`src/matrix.cc:L1022-L1035`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/matrix.cc#L1022-L1035)).

**Stability.** `syev` is backward stable: the computed \f$ (\hat V, \hat\Lambda) \f$ is nearly the
exact eigendecomposition of \f$ A + E \f$ with \f$ \|E\|_2 \le p(n)\, \epsilon \|A\|_2 \f$
(\f$ \epsilon \f$ the unit roundoff), and \f$ \hat V \f$ is unitary to \f$ O(\epsilon) \f$ [7]. Unlike a non-normal diagonalization, no
\f$ \kappa_2(V) \f$ factor enters, so
\f$ \hat F = f(A + E) + O(\epsilon \max_j |f(\lambda_j)|) \f$ when \f$ f(\hat\Lambda) \f$ is accurate
[1, §4.5], [2, §9.1], [6]. Inaccurate individual eigenvectors of clustered eigenvalues do not
matter; what remains is the conditioning of \f$ f \f$ at \f$ A \f$ [1, Ch. 3], e.g.
\f$ \lambda^{-1/2} \f$ near \f$ \lambda = 0 \f$.

## syfn: the given-eigenpair form

`syfn_from_eigen` takes \f$ V \in \mathbb{C}^{n \times m} \f$, \f$ m \le n \f$, and
\f$ w \in \mathbb{R}^m \f$ in any order, and returns

\f[
F = V \operatorname{diag}(f(w)) V^H = \sum_{j=1}^{m} f(w_j)\, v_j v_j^H ,
\f]

by the same (2)/(3) with \f$ Y \in \mathbb{C}^{n \times m} \f$. \f$ V \f$ need not be orthonormal;
then \f$ F \f$ is a weighted Gram matrix rather than \f$ f(A) \f$, e.g. a density matrix
\f$ P = C\, \operatorname{diag}(u)\, C^H \f$ from occupied orbitals \f$ C \f$ and occupations
\f$ u \f$ passed as `Weights`. `info[b]` counts the out-of-domain \f$ w_j \f$ among the \f$ m \f$
given, with no syev term ([`src/ops/syfn/syfn.cc:L174`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/syfn.cc#L174)). For \f$ m = 0 \f$, \f$ F = 0 \f$
([`src/ops/syfn/syfn.cc:L78`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/syfn.cc#L78), [`src/ops/syfn/kernels/syfn_kernels.cc:L75-L85`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/kernels/syfn_kernels.cc#L75-L85)).

## syfn: mapping to kernels

`syfn` has no `choice.hh`, no route spelling and no tuned table: it is a composition, and each
sub-op selects its own route (`BATCHLAS_SYEV_ROUTE`, the `herk`/`syrk`/`gemm` routes; see
@ref perf_syev, @ref perf_level3). Every step runs on one in-order queue over the whole batch
([`src/ops/syfn/syfn.cc:L135`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/syfn.cc#L135)).

| Step | Implementation | Source |
| --- | --- | --- |
| copy, eigendecomposition | `MatrixView::copy`, `syev` (`JobType::EigenVectors`) | [`src/ops/syfn/syfn.cc:L140-L141`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/syfn.cc#L140-L141) |
| `info` | `InfoKernel`, one work-item per item | [`src/ops/syfn/kernels/syfn_kernels.cc:L61-L71`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/kernels/syfn_kernels.cc#L61-L71) |
| \f$ g_j \f$ | `WeightsKernel`, one work-item per \f$ (b, j) \f$ | [`src/ops/syfn/kernels/syfn_kernels.cc:L96-L100`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/kernels/syfn_kernels.cc#L96-L100) |
| \f$ Y = V G \f$ | `ScaleKernel`, one work-item per entry of \f$ Y \f$ | [`src/ops/syfn/kernels/syfn_kernels.cc:L104-L108`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/kernels/syfn_kernels.cc#L104-L108) |
| (2) or (3) | `herk`/`syrk` (`Uplo::Lower`) or `gemm` (`NoTrans`, `ConjTrans`/`Trans`) | [`src/ops/syfn/syfn.cc:L82-L88`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/syfn.cc#L82-L88) |
| exact Hermiticity | `MatrixView::hermitize(Lower)` | [`src/ops/syfn/syfn.cc:L90`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/syfn.cc#L90) |

Workspace per item: `syfn` needs \f$ V \f$ (\f$ n^2 \f$), \f$ w \f$ (\f$ n \f$), \f$ Y \f$
(\f$ n^2 \f$), \f$ g \f$ (\f$ n \f$) and syev's own ([`src/ops/syfn/syfn.cc:L56-L64`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/syfn.cc#L56-L64));
`syfn_from_eigen` only \f$ Y \f$ (\f$ n m \f$) and \f$ g \f$ (\f$ m \f$)
([`src/ops/syfn/syfn.cc:L34-L39`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syfn/syfn.cc#L34-L39)).

## syfn: references

1. N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008 (§1.2 "Definitions
   of f(A)", §4.5 "Diagonalization"). https://doi.org/10.1137/1.9780898717778
2. G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University
   Press, 2013 (§9.1 "Eigenvalue Methods"). ISBN 978-1-4214-0794-4.
3. P.-O. Löwdin, "On the Non-Orthogonality Problem Connected with the Use of Atomic Wave Functions
   in the Theory of Molecules and Crystals", *J. Chem. Phys.* 18(3), 365-375, 1950.
   https://doi.org/10.1063/1.1747632
4. Reference BLAS, `ZHERK`: \f$ C := \alpha A A^H + \beta C \f$ on one triangle, real diagonal.
   https://www.netlib.org/blas/zherk.f
5. R. Penrose, "A generalized inverse for matrices", *Proc. Cambridge Philos. Soc.* (now *Math. Proc.*) 51(3),
   406-413, 1955. https://doi.org/10.1017/S0305004100030401
6. NAG Library, `f01fff` (complex Hermitian matrix function by spectral factorization), Accuracy
   section. https://support.nag.com/numeric/nl/nagdoc_latest/flhtml/f01/f01fff.html
7. E. Anderson et al., *LAPACK Users' Guide*, 3rd ed., SIAM, 1999, "Further Details: Error Bounds
   for the Symmetric Eigenproblem". https://netlib.org/lapack/lug/node90.html
