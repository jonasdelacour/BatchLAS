// SPDX-License-Identifier: MIT
#pragma once
// xLAGTF (dstein's factorization of T - lambda I) in xGTTRF's storage, serial on one item. In:
// dl the sub-, dd the main and du the super-diagonal, already shifted (n = dd.size(); dl, du and
// swapped n - 1 entries, du2 n - 2: views at least that long). Out: dl the multipliers, dd / du /
// du2 the diagonals of U (du2 the second super-diagonal fill), swapped(i) = 1 where rows i and
// i + 1 were interchanged. A pivot below tol in magnitude becomes +-tol (dlagtf's TOL
// perturbation: inverse iteration is near-singular on purpose). Solve with lagts.

#include <batchlas/device/exec.hh>

namespace batchlas::device {

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
