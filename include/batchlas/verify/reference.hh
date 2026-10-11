// SPDX-License-Identifier: MIT
// Host LAPACK references (docs/design/verification.md): eigenvalues, singular values and getrf pivots
// from LAPACKE, never from a BatchLAS path under test. Without LAPACKE (no host backend) every
// function returns false, empty input included, and the caller skips: never a pass. With LAPACKE
// (have_reference) false is a failed reference, and the caller fails.
#pragma once

#include <batchlas/blas/matrix.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <span>
#include <vector>

// Undefined would read as 0 and silently compile the "no reference" branch.
#ifndef BATCHLAS_VERIFY_HAVE_LAPACKE
#error "batchlas/verify/reference.hh: link batchlas::verify, which defines BATCHLAS_VERIFY_HAVE_LAPACKE"
#endif

#if BATCHLAS_VERIFY_HAVE_LAPACKE
#include <lapacke.h>

namespace batchlas::verify::detail {
/// The host dgemm probe (libbatchlas_verify, cached): empty when correct, else the diagnostic.
std::string host_dgemm_defect();
}  // namespace batchlas::verify::detail
#endif

namespace batchlas::verify {

/// True when this build has host LAPACKE references; a reference that returns false here failed.
inline constexpr bool have_reference = BATCHLAS_VERIFY_HAVE_LAPACKE != 0;

namespace detail {
// Every double reference except ?sterf runs blocked code on the host dgemm, which OpenBLAS 0.3.20
// gets wrong on Cooperlake without OPENBLAS_CORETYPE: a wrong reference fails a correct result or
// passes a wrong one, so refuse it. evidence: docs/perf/lu.md#the-host-dgetrf-oracle-is-broken-on-this-box
template <class D> void require_host_dgemm() {
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if constexpr (std::is_same_v<D, double> || std::is_same_v<D, std::complex<double>>) {
        static const std::string defect = host_dgemm_defect();
        if (!defect.empty()) throw std::runtime_error("batchlas::verify: host LAPACKE reference refused\n" + defect);
    }
#endif
}
}  // namespace detail

namespace detail {
// Item @p item of @p A as a packed std::vector, converted to U.
template <class U, class T> std::vector<U> item_vector(const MatrixView<T, MatrixFormat::Dense>& A, int item) {
    const auto m = A.template astype<U>(std::span<const int>(&item, 1));
    return std::vector<U>(m.data().data(), m.data().data() + std::size_t(A.rows()) * std::size_t(A.cols()));
}
}  // namespace detail

/// Ascending eigenvalues of the n x n Hermitian @p a (column-major, ld = n, both triangles valid; destroyed).
template <class D> bool eigenvalues(int n, std::vector<D>& a, std::vector<double>& w) {
    w.assign(static_cast<std::size_t>(n), 0.0);
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if (n == 0) return true;
    detail::require_host_dgemm<D>();
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
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if (s.empty()) return true;
    detail::require_host_dgemm<D>();
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
/// @pre @p a holds the data in its native precision: copy_item_native, never copy_item, which
/// promotes float data and sends it to dgetrf. The view overload below does this for you.
// Never promote float data to dgetrf: OpenBLAS 0.3.20's dgetrf on Cooperlake Xeons (its broken dgemm
// kernel) is wrong from n = 10 while sgetrf is not. evidence: docs/perf/lu.md#the-host-dgetrf-oracle-is-broken-on-this-box
template <class E> bool getrf_pivots(int m, int n, std::vector<E>& a, std::vector<std::int32_t>& ipiv) {
    static_assert(std::is_same_v<E, float> || std::is_same_v<E, double> || std::is_same_v<E, std::complex<float>> ||
                      std::is_same_v<E, std::complex<double>>,
                  "getrf_pivots: float, double, complex<float> or complex<double> only");
    ipiv.assign(static_cast<std::size_t>(std::min(m, n)), 0);
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if (ipiv.empty()) return true;
    detail::require_host_dgemm<E>();
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

/// getrf_pivots of item @p item of a dense view, copied in the view's own precision.
template <class T> bool getrf_pivots(const MatrixView<T, MatrixFormat::Dense>& A, int item, std::vector<std::int32_t>& ipiv) {
    auto a = detail::item_vector<T>(A, item);
    return getrf_pivots(A.rows(), A.cols(), a, ipiv);
}

/// LAPACK geqrf of the m x n @p a (column-major, ld = m; overwritten by R and the reflectors) and its
/// min(m, n) @p tau.
template <class D> bool geqrf_tau(int m, int n, std::vector<D>& a, std::vector<D>& tau) {
    static_assert(std::is_same_v<D, double> || std::is_same_v<D, std::complex<double>>,
                  "geqrf_tau: double or complex<double> only");
    tau.assign(static_cast<std::size_t>(std::min(m, n)), D(0));
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    if (tau.empty()) return true;
    detail::require_host_dgemm<D>();
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

/// eigenvalues() of item @p item of the Hermitian @p A (both triangles valid), promoted.
template <class T> bool eigenvalues(const MatrixView<T, MatrixFormat::Dense>& A, int item, std::vector<double>& w) {
    auto a = detail::item_vector<typename MatrixView<T, MatrixFormat::Dense>::promoted_type>(A, item);
    return eigenvalues(A.rows(), a, w);
}

/// singular_values() of item @p item of @p A, promoted.
template <class T> bool singular_values(const MatrixView<T, MatrixFormat::Dense>& A, int item, std::vector<double>& s) {
    auto a = detail::item_vector<typename MatrixView<T, MatrixFormat::Dense>::promoted_type>(A, item);
    return singular_values(A.rows(), A.cols(), a, s);
}

/// geqrf_tau() of item @p item of @p A, promoted; the factor is discarded.
template <class T>
bool geqrf_tau(const MatrixView<T, MatrixFormat::Dense>& A, int item,
               std::vector<typename MatrixView<T, MatrixFormat::Dense>::promoted_type>& tau) {
    auto a = detail::item_vector<typename MatrixView<T, MatrixFormat::Dense>::promoted_type>(A, item);
    return geqrf_tau(A.rows(), A.cols(), a, tau);
}

}  // namespace batchlas::verify
