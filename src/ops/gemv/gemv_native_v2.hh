// SPDX-License-Identifier: MIT
#pragma once
// Native gemv's last private op: fold_down<Count>, the shift-down sum of lanes {l + stride*t :
// t < Count} onto head lane l. Core shift_reduce folds at stride 1 only (body 4 folds at stride m).

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>

namespace batchlas::ops::gemv::dla {


// Every lane of g must call (lockstep). Closed when Count * stride <= the sub-group size; lanes
// at or above that shift in values no head lane reads.
template <int Count, class G, class T, class Stride = device::Unit>
BATCHLAS_DEVICE_INLINE T fold_down(const G& g, T v, Stride stride = {}) {
    static_assert(Count > 0 && (Count & (Count - 1)) == 0, "gemv fold_down: Count must be a power of two");
    using A = device::Arith<T>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        R re = A::re(v), im = A::im(v);
#pragma unroll
        for (int w = Count / 2; w >= 1; w >>= 1) {
            const int off = int(stride) * w;
            re += sycl::shift_group_left(g, re, off);
            im += sycl::shift_group_left(g, im, off);
        }
        return A::make(re, im);
    } else {
#pragma unroll
        for (int w = Count / 2; w >= 1; w >>= 1) v += sycl::shift_group_left(g, v, int(stride) * w);
        return v;
    }
}

}  // namespace batchlas::ops::gemv::dla
