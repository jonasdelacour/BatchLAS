// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief `lagtf` (xLAGTF): the LU factorization with row interchanges of a shifted tridiagonal
/// \f$T - \lambda I\f$, serial on one item, as `stein` uses it. Not covered by the compatibility
/// promise.
/// @ingroup api_device_lowlevel

#include <batchlas/device/exec.hh>

namespace batchlas::device {

/// @brief xLAGTF, dstein's factorization of \f$T - \lambda I\f$ in xGTTRF's storage, serial on
/// one item; solve with lagts().
///
/// n = `dd.size()`; dl, du and swapped need n - 1 entries and du2 n - 2 (views at least that
/// long). A pivot below tol in magnitude becomes +tol or -tol (dlagtf's TOL perturbation: inverse
/// iteration is near-singular on purpose).
/// @param[in,out] dl       the sub-diagonal, already shifted; the multipliers on return
/// @param[in,out] dd       the diagonal, already shifted; the diagonal of U on return
/// @param[in,out] du       the super-diagonal; the first super-diagonal of U on return
/// @param[out]    du2      the second super-diagonal of U (the fill)
/// @param[out]    swapped  swapped(i) = 1 where rows i and i + 1 were interchanged, else 0
/// @param[in]     tol      the smallest pivot magnitude kept
/// @ingroup api_device_lowlevel
template <class V, class P, class Real>
BATCHLAS_DEVICE_HELPER void lagtf(const Thread&, const V& dl, const V& dd, const V& du, const V& du2, const P& swapped, Real tol) {
    const int n = dd.size();
    for (int i = 0; i < n - 1; ++i) {
        if (sycl::fabs(dd(i)) >= sycl::fabs(dl(i))) {
            swapped(i) = 0;
            if (sycl::fabs(dd(i)) < tol) dd(i) = (dd(i) < Real(0)) ? -tol : tol;
            const Real fact = dl(i) / dd(i);
            dl(i) = fact;
            dd(i + 1) -= fact * du(i);
            if (i < n - 2) du2(i) = Real(0);
        } else {
            swapped(i) = 1;
            const Real fact = dd(i) / dl(i);
            dd(i) = dl(i);
            dl(i) = fact;
            const Real tmp = du(i);
            du(i) = dd(i + 1);
            dd(i + 1) = tmp - fact * dd(i + 1);
            if (i < n - 2) {
                du2(i) = du(i + 1);
                du(i + 1) = -fact * du(i + 1);
            }
        }
    }
    if (sycl::fabs(dd(n - 1)) < tol) dd(n - 1) = (dd(n - 1) < Real(0)) ? -tol : tol;
}

}  // namespace batchlas::device
