// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_sparse
/// @brief routine_call for the sparse routine tags, so `launch<Routine, Tier, T>::run(args...)`
/// calls spmm, iluk_factorize and iluk_apply in tier Tier.
///
/// The sparse counterpart of `<batchlas/device/lapack_run.hh>`; `<batchlas/device/sparse.hh>`
/// does not include it, so include this header to call a sparse routine through `run`.

#include <batchlas/device/launch.hh>
#include <batchlas/device/sparse.hh>

namespace batchlas::device {

template <Transpose TA, Transpose TB> struct routine_call<routine::spmm<TA, TB>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::spmm<TA, TB, Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::iluk_factorize> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::iluk_factorize<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::iluk_apply> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::iluk_apply<Tier>(static_cast<A&&>(a)...);
    }
};

}  // namespace batchlas::device
