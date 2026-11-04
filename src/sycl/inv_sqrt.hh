// SPDX-License-Identifier: MIT
#pragma once

// 1 / sqrt(a). NVPTX folds a plain float `1 / sqrt(a)` into rsqrt.approx by code shape, so float
// runs rsqrt.approx (asm) + one FMA Newton step, y + (y/2)(1 - (a y) y); a (y y) overflows for
// denormal a. Not in device_scalar.hh, a tune-hash input of other families.

#include <sycl/sycl.hpp>

#include <type_traits>

namespace batchlas::sycl_device {

// Lapack: 1/inf = 0, not NaN. FiniteGeOne: a finite a >= 1 (Jacobi cosine 1 + t^2), no guard.
enum class InvSqrtEdge { Lapack, FiniteGeOne };

// sqrt_a = sqrt(a): the fallback's divisor, for callers that hold it.
template <InvSqrtEdge E = InvSqrtEdge::Lapack, typename R>
inline R inv_sqrt(R a, R sqrt_a) {
#if !BATCHLAS_SYCL_IMPL_ACPP && defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    if constexpr (std::is_same_v<R, float>) {
        float y;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(a));
        const float e = sycl::fma(-(a * y), y, 1.0f);
        const float r = sycl::fma(0.5f * y, e, y);
        if constexpr (E == InvSqrtEdge::Lapack) return sycl::isinf(a) ? 0.0f : r;
        else return r;
    }
#endif
    (void)a;
    return R(1) / sqrt_a;
}

template <InvSqrtEdge E = InvSqrtEdge::Lapack, typename R>
inline R inv_sqrt(R a) {
    return inv_sqrt<E>(a, sycl::sqrt(a));
}

}  // namespace batchlas::sycl_device
