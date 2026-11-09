// SPDX-License-Identifier: MIT
#pragma once
// house2, the tridiagonal solver's helper core lacks (design A.2 scalar.hh); apply: core rot.

#include <array>

#include <batchlas/device.hh>

namespace batchlas::device::inline v2::tridiag {

// v with (I - 2 v v^T) (a0, a1)^T = (-sign(a0) ||a||, 0)^T; unscaled norms (inputs are O(||T||)).
template <class R> BATCHLAS_DLA_INLINE std::array<R, 2> house2(R a0, R a1) {
    const R anorm = sycl::sqrt(a0 * a0 + a1 * a1);
    const R alpha = -sycl::copysign(anorm, a0);
    const R v0 = a0 - alpha, v1 = a1;
    const R vnorm = sycl::sqrt(v0 * v0 + v1 * v1);
    const R inv = R(1) / vnorm;
    return {v0 * inv, v1 * inv};
}

}  // namespace batchlas::device::tridiag
