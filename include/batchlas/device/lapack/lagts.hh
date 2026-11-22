// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief `lagts` (xLAGTS): the solve with the `lagtf` factors of a shifted tridiagonal, serial
/// on one item, as `stein` uses it. Not covered by the compatibility promise.
/// @ingroup api_device_lowlevel

#include <batchlas/device/exec.hh>

namespace batchlas::device {

/// @brief xLAGTS without transpose, as xGTTRS: \f$b := (T - \lambda I)^{-1} b\f$ with the
/// factors from lagtf(), serial on one item.
///
/// Applies the row interchanges and the multipliers forward, then back-substitutes with U.
/// n = `dd.size()`; the factors' views are as for lagtf().
/// @param[in]     dl       the multipliers
/// @param[in]     dd       the diagonal of U
/// @param[in]     du       the first super-diagonal of U
/// @param[in]     du2      the second super-diagonal of U
/// @param[in]     swapped  the interchange flags
/// @param[in,out] b        n entries: the right-hand side, the solution on return
/// @ingroup api_device_lowlevel
template <class V, class P, class B>
BATCHLAS_DEVICE_HELPER void lagts(const Thread&, const V& dl, const V& dd, const V& du, const V& du2, const P& swapped, const B& b) {
    const int n = dd.size();
    for (int i = 0; i < n - 1; ++i) {
        if (!swapped(i)) {
            b(i + 1) -= dl(i) * b(i);
        } else {
            const auto tmp = b(i);
            b(i) = b(i + 1);
            b(i + 1) = tmp - dl(i) * b(i);
        }
    }
    b(n - 1) = b(n - 1) / dd(n - 1);
    if (n > 1) b(n - 2) = (b(n - 2) - du(n - 2) * b(n - 1)) / dd(n - 2);
    for (int i = n - 3; i >= 0; --i) {
        b(i) = (b(i) - du(i) * b(i + 1) - du2(i) * b(i + 2)) / dd(i);
    }
}

}  // namespace batchlas::device
