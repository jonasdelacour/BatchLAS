// SPDX-License-Identifier: MIT
#pragma once
// xLAGTF (dstein's factorization of T - lambda I) in xGTTRF's storage, serial on one item. In:
// dl the sub-, dd the main and du the super-diagonal, already shifted. Out: dl the multipliers,
// dd / du / du2 the diagonals of U (du2 the second super-diagonal fill), swapped[i] = 1 where
// rows i and i + 1 were interchanged. A pivot below tol in magnitude becomes +-tol (dlagtf's
// TOL perturbation: inverse iteration is near-singular on purpose). Solve with lagts.

#include <cstdint>

#include <batchlas/device/exec.hh>

namespace batchlas::device {

template <class T>
BATCHLAS_DLA_HELPER void lagtf(const Thread&, T* dl, T* dd, T* du, T* du2, unsigned char* swapped, std::int64_t n, T tol) {
    for (std::int64_t i = 0; i < n - 1; ++i) {
        if (sycl::fabs(dd[i]) >= sycl::fabs(dl[i])) {
            swapped[i] = 0;
            if (sycl::fabs(dd[i]) < tol) dd[i] = (dd[i] < T(0)) ? -tol : tol;
            const T fact = dl[i] / dd[i];
            dl[i] = fact;
            dd[i + 1] -= fact * du[i];
            if (i < n - 2) du2[i] = T(0);
        } else {
            swapped[i] = 1;
            const T fact = dd[i] / dl[i];
            dd[i] = dl[i];
            dl[i] = fact;
            const T tmp = du[i];
            du[i] = dd[i + 1];
            dd[i + 1] = tmp - fact * dd[i + 1];
            if (i < n - 2) {
                du2[i] = du[i + 1];
                du[i + 1] = -fact * du[i + 1];
            }
        }
    }
    if (sycl::fabs(dd[n - 1]) < tol) dd[n - 1] = (dd[n - 1] < T(0)) ? -tol : tol;
}

}  // namespace batchlas::device
