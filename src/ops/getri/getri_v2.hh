// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops private to getri's permutation kernel (everything else comes from v2 core):
//   perm_index    where index i lands under an interchange list, with no index array
//   LocalMinSlot  a work-group minimum in one local int, through relaxed atomics (LAPACK info)

#include <climits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/dla.hh>
#include <src/kernels/common/dla_v2_scalar.hh>

namespace batchlas::device::v2::lu {

// One entry of laswp(ipiv(k1 .. k2-1)) applied to the identity: Forward applies k1 first,
// Backward k2-1 first (the inverse). Pivots are LAPACK's, 1-based, as in core laswp.
template <class Dir = dir::Forward, class P> BATCHLAS_DLA_INLINE int perm_index(const P& ipiv, int k1, int k2, int i) {
    int r = i;
    auto step = [&](int k) {
        const int p = static_cast<int>(ipiv(k)) - 1;
        if (r == k) r = p;
        else if (r == p) r = k;
    };
    if constexpr (std::is_same_v<Dir, dir::Forward>)
        for (int k = k1; k < k2; ++k) step(k);
    else
        for (int k = k2 - 1; k >= k1; --k) step(k);
    return r;
}

// The first index that fails, as LAPACK info: the leader resets, a barrier, every item offers,
// a barrier, info() (0 when nobody offered). No tree, no Scratch.
struct LocalMinSlot {
    int* p;
    template <class Ex> BATCHLAS_DLA_INLINE void reset(const Ex& ex) const {
        if (ex.lid() == 0) *p = INT_MAX;
    }
    BATCHLAS_DLA_INLINE void offer(int v) const {
        using Ref = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>;
        Ref ref(*p);
        ref.fetch_min(v);
    }
    BATCHLAS_DLA_INLINE int info() const {
        const int v = *p;
        return v == INT_MAX ? 0 : v;
    }
};

}  // namespace batchlas::device::v2::lu
