# Algorithms {#algorithms_index}

> **Status:** current · section index

Derivations and pseudo-algorithms for the implemented methods. Performance: @ref perf_evidence.

| Page | What you find there |
| --- | --- |
| **Factorizations** | |
| @subpage algo_potrf "potrf: Cholesky factorization, solve and inverse" | Blocked and batched Cholesky, triangular solves, inverse from the factor. |
| @subpage algo_getrf "getrf: LU with partial pivoting" | Pivot search, row swaps, panel and trailing updates. |
| @subpage algo_geqrf "geqrf: Householder QR, and forming and applying Q" | Reflector generation, compact WY, forming and applying Q. |
| @subpage algo_sytrf "sytrf: symmetric indefinite factorization by Bunch-Kaufman pivoting" | 1x1 and 2x2 pivot choice and the block diagonal factor. |
| **Reductions** | |
| @subpage algo_sytrd "sytrd: tridiagonal reduction, one-stage and two-stage" | Householder tridiagonalization, band reduction and bulge chasing. |
| **Tridiagonal solvers** | |
| @subpage algo_steqr "steqr: implicit QR/QL on symmetric tridiagonals" | Deflation, shifts, Givens chasing, CTA and work-group variants, eigenvector accumulation. |
| @subpage algo_stedc "stedc: divide and conquer for symmetric tridiagonals" | Rank-one tearing, secular equation, deflation, eigenvector merge. |
| @subpage algo_stebz_stein "stebz, stein: bisection and inverse iteration" | Sturm-count bisection for eigenvalues, inverse iteration for eigenvectors. |
| **Drivers** | |
| @subpage algo_syev "syev: symmetric and Hermitian eigensolver drivers" | How reduction, tridiagonal solve and back-transform compose. |
| @subpage algo_sygvd "sygvd: generalized symmetric-definite eigenproblem" | Reduction to standard form via Cholesky and back-transform. |
| @subpage algo_syevx "syevx: partial eigensolvers by Rayleigh-Ritz projection" | Subspace selection, projection and Ritz extraction. |
| @subpage algo_gesvd "gesvd: singular value decomposition of batched small matrices" | One-sided Jacobi / bidiagonal SVD for small batched matrices. |
| **Others** | |
| @subpage algo_syfn "syfn: matrix functions of Hermitian matrices" | f(A) from the eigendecomposition. |
| @subpage algo_ortho "ortho: orthonormal bases for batches of vector blocks" | Orthonormalization of vector blocks. |
| @subpage algo_iluk "iluk: incomplete LU with level of fill" | Symbolic level-of-fill pattern and numeric ILU(k). |
