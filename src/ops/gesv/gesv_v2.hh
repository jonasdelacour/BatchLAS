// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops private to the fused gesv tiny tier (getf2's pivot scaling and the rank-1
// row update by the broadcast pivot row); the rest comes from v2 core.

#include <cstdint>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::device::inline v2::lu {

// Multiply by the reciprocal when it is finite and nonzero, else divide (the sfmin arm), else
// (an exact zero pivot) the caller's singular value. Branches on the uniform flags, never a
// select: a select pays the divide every time. evidence: docs/perf/lu.md#the-rhs-pad-column-cost
template <class T> struct RecipOrDiv {
    T d;
    T rc;
    bool zero;
    bool use_mul;
    BATCHLAS_DLA_INLINE static RecipOrDiv make(T d) {
        const bool z = sycl_device::dev_is_zero(d);  // EXACT zero, no epsilon
        const T rc = arith_t<arith::Fma, T>::recip(d);
        return {d, rc, z, !z && sycl_device::dev_isfinite(rc) && !sycl_device::dev_is_zero(rc)};
    }
    BATCHLAS_DLA_INLINE T solve(T y, T if_singular) const {
        using Ar = arith_t<arith::Fma, T>;
        T x = if_singular;
        if (use_mul) x = Ar::mul(y, rc);
        else if (!zero) x = Ar::div(y, d);
        return x;
    }
};

// x[k] -= m * x[k]@src for k0 <= k < min(N, kend), applied where act: the source lane is
// excluded by its own act flag. Collective (px.bcast): every lane reaches every k.
template <int N, class Px, class T>
BATCHLAS_DLA_INLINE void ger_frag_bcast(const Px& px, T m, T (&x)[N], std::uint32_t src, int k0, int kend, bool act) {
    using Ar = arith_t<arith::Fma, T>;
BATCHLAS_UNROLL_FULL
    for (int k = k0; k < N; ++k) {
        if (k >= kend) continue;
        const T u = px.bcast(x[k], src);
        if (act) x[k] = Ar::sub(x[k], Ar::mul(m, u));
    }
}

}  // namespace batchlas::device::lu
