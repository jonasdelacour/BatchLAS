// SPDX-License-Identifier: MIT
// Host LAPACK references (docs/design/verification.md): eigenvalues, singular values and getrf pivots
// from LAPACKE, never from a BatchLAS path under test. Without LAPACKE (no host backend) every
// function returns false and the caller reports NaN, so a values-only check is `bad`, never passed.
#pragma once

#include <batchlas/verify/norms.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#if BATCHLAS_VERIFY_HAVE_LAPACKE
#include <lapacke.h>
#endif

namespace batchlas::verify {

/// Packed (ld = rows) column-major copy of item @p item, promoted to double / complex<double>.
template <class View> auto copy_item(const View& A, int item) {
    const auto m = detail::item_of(A, item);
    using D = promoted_t<std::remove_cv_t<std::remove_pointer_t<decltype(m.data)>>>;
    std::vector<D> out(static_cast<std::size_t>(m.rows) * m.cols);
    for (int j = 0; j < m.cols; ++j)
        for (int i = 0; i < m.rows; ++i)
            out[static_cast<std::size_t>(j) * m.rows + i] = up(m.data[static_cast<long long>(j) * m.ld + i]);
    return out;
}

/// Packed (ld = rows) column-major copy of item @p item in the view's own element type.
template <class View> auto copy_item_native(const View& A, int item) {
    const auto m = detail::item_of(A, item);
    using E = std::remove_cv_t<std::remove_pointer_t<decltype(m.data)>>;
    std::vector<E> out(static_cast<std::size_t>(m.rows) * m.cols);
    for (int j = 0; j < m.cols; ++j)
        for (int i = 0; i < m.rows; ++i) out[static_cast<std::size_t>(j) * m.rows + i] = m.data[static_cast<long long>(j) * m.ld + i];
    return out;
}

/// Ascending eigenvalues of the n x n Hermitian @p a (column-major, ld = n, both triangles valid; destroyed).
template <class D> bool eigenvalues(int n, std::vector<D>& a, std::vector<double>& w) {
    w.assign(static_cast<std::size_t>(n), 0.0);
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if (n == 0) return true;
    if constexpr (std::is_same_v<D, double>)
        return LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'L', n, a.data(), n, w.data()) == 0;
    else
        return LAPACKE_zheevd(LAPACK_COL_MAJOR, 'N', 'L', n, reinterpret_cast<lapack_complex_double*>(a.data()), n,
                              w.data()) == 0;
#else
    (void)a;
    return false;
#endif
}

/// Descending singular values of the m x n @p a (column-major, ld = m; destroyed).
template <class D> bool singular_values(int m, int n, std::vector<D>& a, std::vector<double>& s) {
    s.assign(static_cast<std::size_t>(std::min(m, n)), 0.0);
    if (s.empty()) return true;
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if constexpr (std::is_same_v<D, double>)
        return LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'N', m, n, a.data(), m, s.data(), nullptr, 1, nullptr, 1) == 0;
    else
        return LAPACKE_zgesdd(LAPACK_COL_MAJOR, 'N', m, n, reinterpret_cast<lapack_complex_double*>(a.data()), m,
                              s.data(), nullptr, 1, nullptr, 1) == 0;
#else
    (void)a;
    return false;
#endif
}

// A residual bound is satisfied by ANY valid pivot choice, so a kernel that pivots on |z| instead of
// LAPACK's |re|+|im|, or breaks ties the other way, passes every residual. This is the check that does not.
/// 1-based pivots of the m x n @p a (column-major, ld = m; overwritten by its LU factor): min(m, n)
/// of them, from ?getrf in E's own precision (float, double, complex<float>, complex<double>).
// Never promote float data to dgetrf: this box's host dgetrf is wrong from n = 10, sgetrf is not.
// evidence: docs/perf/lu.md#the-host-dgetrf-oracle-is-broken-on-this-box
template <class E> bool getrf_pivots(int m, int n, std::vector<E>& a, std::vector<std::int32_t>& ipiv) {
    ipiv.assign(static_cast<std::size_t>(std::min(m, n)), 0);
    if (ipiv.empty()) return true;
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    std::vector<lapack_int> p(ipiv.size());
    lapack_int info;
    if constexpr (std::is_same_v<E, float>)
        info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, a.data(), m, p.data());
    else if constexpr (std::is_same_v<E, double>)
        info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, a.data(), m, p.data());
    else if constexpr (std::is_same_v<E, std::complex<float>>)
        info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_float*>(a.data()), m, p.data());
    else
        info = LAPACKE_zgetrf(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_double*>(a.data()), m, p.data());
    // info > 0 is an exactly singular U: the pivots are still valid, so only a negative info fails.
    if (info < 0) return false;
    for (std::size_t i = 0; i < p.size(); ++i) ipiv[i] = static_cast<std::int32_t>(p[i]);
    return true;
#else
    (void)a;
    return false;
#endif
}

/// LAPACK geqrf of the m x n @p a (column-major, ld = m; overwritten by R and the reflectors) and its
/// min(m, n) @p tau.
template <class D> bool geqrf_tau(int m, int n, std::vector<D>& a, std::vector<D>& tau) {
    tau.assign(static_cast<std::size_t>(std::min(m, n)), D(0));
    if (tau.empty()) return true;
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if constexpr (std::is_same_v<D, double>)
        return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a.data(), m, tau.data()) == 0;
    else
        return LAPACKE_zgeqrf(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_double*>(a.data()), m,
                              reinterpret_cast<lapack_complex_double*>(tau.data())) == 0;
#else
    (void)a;
    return false;
#endif
}

/// Eigenvalues of the symmetric tridiagonal (diagonal @p d, off-diagonal @p e): ascending, written to @p d.
inline bool tridiagonal_eigenvalues(std::vector<double>& d, std::vector<double>& e) {
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if (d.empty()) return true;
    return LAPACKE_dsterf(static_cast<lapack_int>(d.size()), d.data(), e.data()) == 0;
#else
    (void)d;
    (void)e;
    return false;
#endif
}

}  // namespace batchlas::verify
