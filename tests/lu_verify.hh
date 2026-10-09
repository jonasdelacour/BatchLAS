// SPDX-License-Identifier: MIT
// The LU-family tests hold raw pointers with padded ld and a non-natural stride; this wraps them as
// views for batchlas::verify and keeps the pass/fail decision in verify::pass.
#pragma once

#include <gtest/gtest.h>

#include <batchlas/blas/matrix.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <vector>

namespace lu_verify {

/// A read-only view over host or unified memory; stride 0 means ld * cols.
template <class T>
batchlas::MatrixView<T, batchlas::MatrixFormat::Dense> view_over(const T* base, int rows, int cols, int ld,
                                                                 int stride = 0, int batch = 1) {
    return batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>(const_cast<T*>(base), rows, cols, ld,
                                                                  stride > 0 ? stride : ld * cols, batch);
}

/// ||P A0 - L U||_F / ||A0||_F of the m x n factor @p F (ld @p ld) with 1-based pivots.
template <class T>
double factor_residual(const T* A0, const T* F, const int* ipiv, int m, int n, int ld) {
    batchlas::VectorView<std::int32_t> piv(const_cast<std::int32_t*>(reinterpret_cast<const std::int32_t*>(ipiv)),
                                           std::min(m, n), 1);
    return batchlas::verify::getrf_residual(view_over(A0, m, n, ld), view_over(F, m, n, ld), piv);
}

/// ||op(A0) X - B0||_F / (||A0||_F ||X||_F). op(A0) is materialised exactly (transpose, conjugate).
template <class T>
double solve_residual(const T* A0, const T* X, const T* B0, int n, int nrhs, int lda, int ldb,
                      batchlas::Transpose op) {
    std::vector<T> opa;
    const T* a = A0;
    int ld = lda;
    if (op != batchlas::Transpose::NoTrans) {
        opa.resize(static_cast<std::size_t>(n) * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T v = A0[static_cast<std::size_t>(i) * lda + j];
                if constexpr (batchlas::verify::is_complex<T>::value)
                    if (op == batchlas::Transpose::ConjTrans) v = std::conj(v);
                opa[static_cast<std::size_t>(j) * n + i] = v;
            }
        a = opa.data();
        ld = n;
    }
    return batchlas::verify::solve_residual(view_over(a, n, n, ld), view_over(X, n, nrhs, ldb),
                                            view_over(B0, n, nrhs, ldb));
}

/// ||A0 C - I||_F / (||A0||_F ||C||_F).
template <class T>
double inverse_residual(const T* A0, const T* C, int n, int lda, int ldc) {
    return batchlas::verify::solve_residual(view_over(A0, n, n, lda), view_over(C, n, n, ldc),
                                            batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>());
}

/// verify::pass with the value and the bound in the failure message.
template <class T>
::testing::AssertionResult within(batchlas::verify::Check kind, int n, double value) {
    if (batchlas::verify::pass<T>(kind, n, value)) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "value " << value << " exceeds bound "
                                         << batchlas::verify::bound<T>(kind, n) << " (n=" << n << ")";
}

/// Machine epsilon (twice the unit roundoff) for slack in non-residual comparisons.
template <class T>
constexpr double machine_eps() { return 2.0 * batchlas::verify::eps<T>(); }

}  // namespace lu_verify
