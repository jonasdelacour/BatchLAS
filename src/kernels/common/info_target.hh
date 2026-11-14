// SPDX-License-Identifier: MIT
#pragma once

// Host half of the per-item info contract (device half: info_span.hh); no kernel, no device code.
// evidence: docs/design/runtime-internals.md#runtime-internals-the-per-item-info-span-contract

#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-span.hh>
#include <src/queue.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas::detail {

// The caller's span if long enough, else pool scratch: a *_buffer_size of info_scratch_bytes holds.
inline Span<int32_t> info_target(Queue& ctx, BumpAllocator& pool, Span<int32_t> info_out, size_t count) {
    if (info_out.size() >= count) return info_out;
    return pool.allocate<int32_t>(ctx, count);
}

// The workspace info_target may draw: the whole *_buffer_size of a route with no other scratch.
inline std::size_t info_scratch_bytes(Queue& ctx, int64_t batch) {
    if (batch < 1) return 0;
    return workspace_bytes(
        [&](BumpAllocator& p) { return p.allocate<int32_t>(ctx, static_cast<std::size_t>(batch)); });
}

inline bool any_nonzero(const int32_t* flags, int64_t count) {  // USM shared / host, already waited for
    for (int64_t b = 0; b < count; ++b)
        if (flags[b] != 0) return true;
    return false;
}

}  // namespace batchlas::detail
