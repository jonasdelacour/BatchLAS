#pragma once

// Host LAPACK references for the tuner's eigen and SVD specs: the eigenvalues or singular values
// of one item, in double / complex<double>, from LAPACKE (?syevd / ?heevd, ?gesdd), never from a
// BatchLAS path the tuner is ranking. Without LAPACKE (no host backend) they return false and the
// caller reports a NaN residual, so a values-only arm is `bad`, never silently passed.

#include "residuals.hh"  // sample_indices

#include <algorithm>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#if BATCHLAS_TUNE_HAVE_LAPACKE
#include <lapacke.h>
#endif

namespace batchlas::tune {

// Ascending eigenvalues of the n x n Hermitian `a` (column-major, both triangles valid; destroyed).
template <class D>
bool host_eigenvalues(int n, std::vector<D>& a, std::vector<double>& w) {
    w.assign(std::size_t(n), 0.0);
#if BATCHLAS_TUNE_HAVE_LAPACKE
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

// Descending singular values of the m x n `a` (column-major; destroyed).
template <class D>
bool host_singular_values(int m, int n, std::vector<D>& a, std::vector<double>& s) {
    s.assign(std::size_t(std::min(m, n)), 0.0);
    if (s.empty()) return true;
#if BATCHLAS_TUNE_HAVE_LAPACKE
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

}  // namespace batchlas::tune
