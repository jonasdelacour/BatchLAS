#pragma once

// The ormqr vendor call gated on the library being compiled in: with it absent the
// call is never instantiated, so there is no symbol to link, and the call throws
// NoRouteError instead.

#include <batchlas/blas/functions/ormqr.hh>

#include "../../select/vendor.hh"

#include <cstddef>
#include <utility>

namespace batchlas::blas::dispatch::detail {

template <Backend B, typename T, typename... Args>
Event ormqr_vendor_or_throw(Args&&... args) {
    if constexpr (!batchlas::select::factorization_vendor_available<B>) {
        batchlas::select::throw_no_vendor_route<T>(batchlas::Op::ormqr, B, batchlas::select::kFactorizationLibrary<B>);
    } else {
        return batchlas::backend::ormqr_vendor<B, T>(std::forward<Args>(args)...);
    }
}

template <Backend B, typename T, typename... Args>
std::size_t ormqr_vendor_buffer_size_or_throw(Args&&... args) {
    if constexpr (!batchlas::select::factorization_vendor_available<B>) {
        batchlas::select::throw_no_vendor_route<T>(batchlas::Op::ormqr, B, batchlas::select::kFactorizationLibrary<B>);
    } else {
        return batchlas::backend::ormqr_vendor_buffer_size<B, T>(std::forward<Args>(args)...);
    }
}

} // namespace batchlas::blas::dispatch::detail
