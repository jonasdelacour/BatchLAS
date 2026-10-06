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

// Expansion sizing, fit ceiling and route pin, CUDA-free so sytrd_blocked.cc can ask
// her2k's exact question. One definition: never copy these predicates into a caller.
// evidence: docs/perf/level3.md#level-3-scratch-expansions-and-their-ceilings
namespace batchlas::backend::detail {

// Leading dimension of an expanded copy: packed, padded to 16 bytes for packet loads.
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

// Two HARD ceilings, not tuned: the global range must fit an int, the scratch a
// quarter of device memory (BATCHLAS_EXPAND_MAX_BYTES lowers it).
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

// BATCHLAS_EXPAND_ROUTE: 1 = "expand", 0 = "loop", -1 = unset. The only parse
// (cublas.cc's rankk_route_pin delegates here).
inline int expansion_route_pin() {
    if (const char* route = batchlas::settings().selection.expand_route.get()) {
        if (std::string_view(route) == "expand") return 1;
        if (std::string_view(route) == "loop") return 0;
    }
    return -1;
}

// evidence: docs/perf/level3.md#herk-and-her2k-the-gemm-plus-fold-crossovers
inline bool her2k_gemm_preferred(int n, int batch) {
    const int pin = expansion_route_pin();
    if (pin >= 0) return pin != 0;
    return batch >= 2 || n >= 128;
}

// The WHOLE condition her2k_vendor uses for its batched-GEMM route; ask this, not a half.
inline bool her2k_takes_gemm_route(const Queue& ctx, int n, int batch, std::size_t bytes) {
    return her2k_gemm_preferred(n, batch) && expansion_fits(ctx, n, batch, bytes);
}

}  // namespace batchlas::backend::detail
