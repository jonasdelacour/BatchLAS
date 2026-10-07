#pragma once

#include "math-helpers.hh"
#include "queue.hh"

#include <batchlas/settings.hh>
#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string_view>

// Sizing, fit ceiling and her2k route predicate of the scratch expansions, outside src/backends/ so
// sytrd_blocked.cc asks the very predicate her2k_vendor answers. One definition: a copy drifts silently.
// evidence: docs/perf/level3.md#level-3-scratch-expansions-and-their-ceilings
namespace batchlas::backend::detail {

// Every element is written, so the caller's ld is irrelevant: pack, pad to 16 bytes for packet loads.
template <typename T>
int expanded_ld(int n) {
    constexpr int elements_per_packet = std::max<int>(1, 16 / sizeof(T));
    return ::batchlas::internal::ceil_div(n, elements_per_packet) * elements_per_packet;
}

template <typename T>
std::size_t expanded_workspace_bytes(Queue& ctx, int n, int batch) {
    auto sizer = BumpAllocator::measuring();
    sizer.allocate<T>(ctx, static_cast<std::size_t>(expanded_ld<T>(n)) *
                               static_cast<std::size_t>(n) *
                               static_cast<std::size_t>(batch));
    return sizer.required_bytes();
}

// Two hard ceilings, not tuned: the int-linearised SYCL range (2^31 elements) and a quarter of global
// memory, which BATCHLAS_EXPAND_MAX_BYTES lowers. Past either the caller takes its no-scratch route.
inline bool expansion_fits(const Queue& ctx, int n, int batch, std::size_t bytes) {
    const std::size_t elements = static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(batch);
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    std::size_t budget = ctx.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE) / 4;
    if (const char* capped = batchlas::settings().geometry.expand_max_bytes.get()) {
        budget = std::min(budget, static_cast<std::size_t>(std::strtoull(capped, nullptr, 10)));
    }
    return bytes <= budget;
}

// BATCHLAS_EXPAND_ROUTE: 1 = expand, 0 = loop, -1 = unset; herk and her2k read it here. Not in the
// backend, because a sytrd guard that modelled half of her2k's predicate shipped as a bug.
inline int expansion_route_pin() {
    if (const char* route = batchlas::settings().selection.expand_route.get()) {
        if (std::string_view(route) == "expand") return 1;
        if (std::string_view(route) == "loop") return 0;
    }
    return -1;
}

// her2k's GEMM-plus-fold vs the per-item cublas?her2k loop.
// evidence: docs/perf/level3.md#herk-and-her2k-the-gemm-plus-fold-crossovers
inline bool her2k_gemm_preferred(int n, int batch) {
    const int pin = expansion_route_pin();
    if (pin >= 0) return pin != 0;
    return batch >= 2 || n >= 128;
}

// The whole of her2k_vendor's batched-GEMM condition, never half of it.
inline bool her2k_takes_gemm_route(const Queue& ctx, int n, int batch, std::size_t bytes) {
    return her2k_gemm_preferred(n, batch) && expansion_fits(ctx, n, batch, bytes);
}

}  // namespace batchlas::backend::detail
