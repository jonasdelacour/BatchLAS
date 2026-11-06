// SPDX-License-Identifier: MIT
#pragma once
// ?GESV's tiny-tier step: the rank-1 row update by the broadcast pivot row (two roundings, a
// source lane, a start column), in gesv_detail until core ger_row takes them.

#include <cstdint>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::device::inline v2::gesv_detail {

// x[k] -= m * x[k]@src, k0 <= k < min(N, kend), where act; collective: every lane reaches every k.
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

}  // namespace batchlas::device::gesv_detail
