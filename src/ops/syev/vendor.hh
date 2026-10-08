#pragma once

/// @file
/// @brief The syev vendor call for callers that bypass selection, gated on the library. @ingroup api_selection_ops
// Without the library the call is never instantiated (no symbol to link); it throws NoRouteError instead.

#include <batchlas/blas/functions/syev.hh>

#include "../../select/vendor.hh"

#include <cstddef>
#include <utility>

namespace batchlas::blas::dispatch::detail {

template <Backend B, typename T, typename... Args>  /// backend::syev_vendor. @throws NoRouteError without the library.
Event syev_vendor_or_throw(Args&&... args) {
    if constexpr (!batchlas::select::solver_vendor_available<B>) {
        batchlas::select::throw_no_vendor_route<T>(batchlas::Op::syev, B, batchlas::select::kSolverLibrary<B>);
    } else {
        return batchlas::backend::syev_vendor<B, T>(std::forward<Args>(args)...);
    }
}

template <Backend B, typename T, typename... Args>  /// Its workspace size. @throws NoRouteError without the library.
std::size_t syev_vendor_buffer_size_or_throw(Args&&... args) {
    if constexpr (!batchlas::select::solver_vendor_available<B>) {
        batchlas::select::throw_no_vendor_route<T>(batchlas::Op::syev, B, batchlas::select::kSolverLibrary<B>);
    } else {
        return batchlas::backend::syev_vendor_buffer_size<B, T>(std::forward<Args>(args)...);
    }
}

} // namespace batchlas::blas::dispatch::detail
