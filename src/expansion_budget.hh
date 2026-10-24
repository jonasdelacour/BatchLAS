// SPDX-License-Identifier: MIT
#pragma once

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/ops/symm/choice.hh>

#include <batchlas/settings.hh>
#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

// Sizing and fit ceilings of the level-3 scratch expansions (symm, hemm, trmm expand; herk and her2k
// fold), shared so the chunk a launch leases and the can_run that admits it cannot drift.
// evidence: docs/perf/level3.md#level-3-scratch-expansions-and-their-ceilings
namespace batchlas::backend::detail {

// Every element is written, so the caller's ld is irrelevant: pack, pad to 16 bytes for packet loads.
template <typename T>
int expanded_ld(int n) {
    constexpr int elements_per_packet = std::max<int>(1, 16 / sizeof(T));
    return ::batchlas::detail::ceil_div(n, elements_per_packet) * elements_per_packet;
}

template <typename T>
std::size_t expanded_workspace_bytes(Queue& ctx, int n, int batch) {
    auto sizer = BumpAllocator::measuring();
    sizer.allocate<T>(ctx, static_cast<std::size_t>(expanded_ld<T>(n)) *
                               static_cast<std::size_t>(n) *
                               static_cast<std::size_t>(batch));
    return sizer.required_bytes();
}

// A quarter of global memory; BATCHLAS_EXPAND_MAX_BYTES lowers it.
inline std::size_t expansion_budget_bytes(const Queue& ctx) {
    std::size_t budget = ctx.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE) / 4;
    if (const char* capped = batchlas::settings().geometry.expand_max_bytes.get()) {
        budget = std::min(budget, static_cast<std::size_t>(std::strtoull(capped, nullptr, 10)));
    }
    return budget;
}

// Three hard ceilings, not tuned: the batch in grid z (expand_mirrored, the folds), the
// int-linearised SYCL range (2^31 elements of the PADDED expanded_ld x n x batch scratch) and
// expansion_budget_bytes. max_grid_z is select::Device::max_groups_yz (CUDA 65535, Metal unbounded).
template <typename T>
inline bool expansion_fits(const Queue& ctx, int n, int batch, std::size_t bytes,
                           std::int64_t max_grid_z = ::batchlas::ops::symm::kMaxGridBatch) {
    if (batch > max_grid_z) {
        return false;
    }
    const std::size_t elements = static_cast<std::size_t>(expanded_ld<T>(n)) *
                                 static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(batch);
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    return bytes <= expansion_budget_bytes(ctx);
}

// Items per launch of a chunked expansion: the most (at most batch) whose scratch passes
// expansion_fits; 0 only when a single item does not, the one shape a chunked family refuses.
template <typename T>
inline int expansion_chunk(Queue& ctx, int n, int batch, std::int64_t max_grid_z) {
    const std::size_t per_item = static_cast<std::size_t>(expanded_ld<T>(n)) * static_cast<std::size_t>(n);
    if (per_item == 0 || batch <= 0) return std::max(batch, 0);
    const std::size_t by_bytes = expansion_budget_bytes(ctx) / (per_item * sizeof(T));
    const std::size_t by_range = static_cast<std::size_t>(std::numeric_limits<int>::max()) / per_item;
    std::size_t c = std::min({static_cast<std::size_t>(batch), static_cast<std::size_t>(std::max<std::int64_t>(
                                  max_grid_z, 0)), by_bytes, by_range});
    // The allocator's alignment padding can push the last item over: back off until it fits.
    while (c > 0 && !expansion_fits<T>(ctx, n, int(c), expanded_workspace_bytes<T>(ctx, n, int(c)), max_grid_z)) --c;
    return static_cast<int>(c);
}

}  // namespace batchlas::backend::detail
