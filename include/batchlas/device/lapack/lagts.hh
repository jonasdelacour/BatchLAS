// SPDX-License-Identifier: MIT
#pragma once
// xLAGTS (as xGTTRS, no transpose): b := (T - lambda I)^{-1} b with the lagtf factors, serial
// on one item: the row interchanges and multipliers forward, then U back-substitution
// (n = dd.size(); the factors' views as lagtf's, b n entries).

#include <batchlas/device/exec.hh>

namespace batchlas::device {

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
