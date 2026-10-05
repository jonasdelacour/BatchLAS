#pragma once

// The syev vendor call gated on the library being compiled in: with it absent the
// call is never instantiated, so there is no symbol to link, and the call throws
// NoRouteError instead.

#include <batchlas/blas/functions/syev.hh>

#include "../../select/vendor.hh"

#include <cstddef>
#include <utility>

namespace batchlas::blas::dispatch::detail {

template <Backend B, typename T, typename... Args>
Event syev_vendor_or_throw(Args&&... args) {
    if constexpr (!batchlas::select::solver_vendor_available<B>) {
        batchlas::select::throw_no_vendor_route<T>(batchlas::Op::syev, B, batchlas::select::kSolverLibrary<B>);
    } else {
        return batchlas::backend::syev_vendor<B, T>(std::forward<Args>(args)...);
    }
}

template <Backend B, typename T, typename... Args>
std::size_t syev_vendor_buffer_size_or_throw(Args&&... args) {
    if constexpr (!batchlas::select::solver_vendor_available<B>) {
        batchlas::select::throw_no_vendor_route<T>(batchlas::Op::syev, B, batchlas::select::kSolverLibrary<B>);
    } else {
        return batchlas::backend::syev_vendor_buffer_size<B, T>(std::forward<Args>(args)...);
    }
}

} // namespace batchlas::blas::dispatch::detail
