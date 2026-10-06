# Inbox from shard H1-matrix

## -> docs/design/symbol-visibility.md: symbol visibility: enums used as template arguments

Clang and GCC give a template instantiation the **minimum** of the template's own visibility
and the visibility of its template **arguments**. `Backend` and `MatrixFormat` are non-type
template parameters across the whole public surface (207 `template <Backend ...>`
declarations alone), so under `-fvisibility=hidden` an unannotated enum drags every one of
those instantiations to hidden, and `BATCHLAS_API` on the function is inert.

Measured, not assumed: annotating the function alone left
`batchlas::gemm<Backend::CUDA, float>` as a local `t` symbol; annotating the enum flipped it
to an exported `W`. 63% of the symbols a consumer links (688 of 1,083) were affected. The
failure mode is an undefined reference at consumer link, never a compile diagnostic, so
nothing in-tree would have caught it: no test links a monolithic install.

Only the enums that appear as template arguments need this. Adding a template parameterised
on another enum in `include/batchlas/blas/enums.hh` means annotating that enum too.

Code sites that now point here:
- `include/batchlas/blas/enums.hh` (comment above `namespace batchlas`):
  `evidence: docs/design/symbol-visibility.md#symbol-visibility-enums-used-as-template-arguments`

## -> docs/design/symbol-visibility.md: symbol visibility: the export attribute on the Matrix class template

`BATCHLAS_API` is on the **class template** `Matrix` (and `MatrixView`, `VectorView`), not on
the 313 explicit instantiations in `src/matrix.cc`, and the difference is 291 symbols. Only 22
of those lines are `template class ...;`; the other 291 individually instantiate **member
templates** (the constrained constructors, the Identity / Random / Zeros / Ones / Diagonal /
Triangular / TriDiagToeplitz / RandomSparseHermitian factories, `convert_to`,
`to_row_major` / `to_column_major`, and `MatrixView`'s `at` / `deep_copy` / `fill_*` /
`symmetrize` / `hermitize` / `triangularize`), which a whole-class instantiation does not
reach. A class-level attribute propagates to every member and every specialisation and so
covers all 313; annotating the instantiation block would cover 22.

Separately, an attribute on an explicit instantiation is not portable: measured, g++ 13
rejects `template class __attribute__((visibility("default"))) F<double,1>;` with
"'F' is not a class template" where clang accepts it.

Code sites that now point here:
- `include/batchlas/blas/matrix.hh` (comment above `class BATCHLAS_API Matrix`):
  `evidence: docs/design/symbol-visibility.md#symbol-visibility-the-export-attribute-on-the-matrix-class-template`

## -> docs/design/gesvd.md: gesvd design: why SvdVectors::Thin exists

`SvdVectors::Thin` (LAPACK `'S'`) exists because `All` is unusable on tall-skinny input: a
10000 x 32 problem has to materialise a 10000 x 10000 U, 400 MB per matrix in `float`, so
batch = 4 needs 1.6 GB for a factor whose last 9968 columns are an arbitrary orthonormal
completion the caller did not ask for.

The identity most of the implementation rests on: Thin and All **differ on at most one side**.
For m <= n, k == m, so a thin U (m x k) is exactly a full U (m x m); for m >= n, k == n, so a
thin V^H is exactly a full V^H. Square input has Thin == All on both sides. Entry points
therefore canonicalise Thin to All whenever the shapes coincide (`canonical_jobu` /
`canonical_jobvh` in `enums.hh`), and only the genuinely thinner side has to be handled, or
rejected, by any given route. `*_buffer_size` and the run path must canonicalise identically,
or the workspace is sized for a different computation than the one performed.

LAPACK's `'O'` (overwrite A with one of the factors) is deliberately absent. If it is ever
wanted, append it as a new enumerator: appending keeps the existing ordinals stable for the
benchmarks that pass jobs as ints.

Code sites that now point here:
- `include/batchlas/blas/enums.hh` (above `enum class SvdVectors`):
  `evidence: docs/design/gesvd.md#gesvd-design-why-svdvectorsthin-exists`

## -> docs/design/syevx.md: syevx: why the preconditioner environment variable is only a default

`SyevxPreconditioner` precedence differs deliberately from `SyevxAlgorithm`. For the algorithm,
`BATCHLAS_SYEVX_ALGORITHM` **wins** over `SyevxParams::method`, so a whole application can be
forced onto one algorithm for diagnosis. For the preconditioner, `BATCHLAS_SYEVX_PRECONDITIONER`
only supplies the **default** that `Auto` falls back to; it never overrides an explicit request.
An algorithm can always be substituted for another; a preconditioner cannot. An ILU(k) factor a
caller built and handed in has no substitute, and silently ignoring it (or, worse, silently
ignoring a request for it) would be a correctness surprise rather than a performance one.

On `JacobiShifted`: the removed `enums.hh` comment said it "is offered because the
constant-diagonal case is provably a no-op and the general case is safe, not because it was
found to pay". `docs/perf/syevx.md#lobpcg-jacobi-preconditioners` gives the other half of the
reason (it is the only Jacobi form legal for `find_largest`); both belong together. Its
iteration counts (neutral 0.85-1.2x on random symmetric input, 0.2-0.9x on graded input;
`Jacobi` 2.1-7.3x fewer iterations on graded input) are already in that perf section, and
`enums.hh` also points there.

Code sites that now point here:
- `include/batchlas/blas/enums.hh` (above `enum class SyevxPreconditioner`):
  `evidence: docs/design/syevx.md#syevx-why-the-preconditioner-environment-variable-is-only-a-default`

## -> docs/pages/architecture.md: register the matrix-model page

New page `docs/design/matrix-model.md` (label `design_matrix_model`) needs a row in the design
table, for example:
`| @subpage design_matrix_model "Matrix model" | Column-major storage, ld and batch stride, Matrix vs MatrixView, KernelMatrixView, heterogeneous batches, CSR strides, the strong integer types. |`

## -> docs/design/known-defects.md: matrix and vector container defects

`docs/design/matrix-model.md#matrix-model-open-debts` lists five located, unfixed defects in
`matrix.hh` / `src/matrix.cc` (an undefined `MatrixView::transpose`, packed-only addressing and
identical items in `fill_triangular_random` / `fill_tridiag_toeplitz`, `fill_random` writing
padding, a no-op slice assert in `KernelMatrixView`, a length assert in
`fill_diagonal(ctx, Span, k)` that can fire for `k != 0`). The owner may want a row per defect
in the known-defects table pointing there.
