#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include <batchlas/util/sycl-span.hh>

#include "../queue.hh"

// Per-item status span: caller-owned USM, 0 = converged, >0 LAPACK-like, empty = not requested.
// An ACCUMULATOR: cleared once by the entry point, then only raised. evidence: docs/design/runtime-internals.md#runtime-internals-the-per-item-info-span-contract
namespace batchlas::detail {

// A too-short span is ignored, not rejected: this is the inner-loop spelling.
inline int32_t* info_ptr(Span<int32_t> info, int64_t batch) {
    if (batch <= 0) return nullptr;
    if (info.size() < static_cast<size_t>(batch)) return nullptr;
    return info.data();
}

inline void info_clear(Queue& ctx, Span<int32_t> info, int64_t batch) {
    if (int32_t* out = info_ptr(info, batch)) {
        ctx->memset(out, 0, sizeof(int32_t) * static_cast<size_t>(batch));
    }
}

// fetch_max, not a store: several producers share an item and the answer is "did ANY fail".
inline void info_report(int32_t* info, int64_t item, int32_t status) {
    if (!info || status <= 0) return;
    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>
        slot(info[item]);
    slot.fetch_max(status);
}

// SINGLE-writer store (no separate clear to race on an out-of-order queue). Never with shared producers.
inline void info_store(int32_t* info, int64_t item, int32_t status) {
    if (!info) return;
    info[item] = status;
}

// Single-writer copy of a leaf tier's flags. syevx_lobpcg/_filtered write 1 for CONVERGED, hence
// `one_means_converged`. The caller keeps `flags` alive until the kernel has run.
inline void info_from_flags(Queue& ctx,
                            Span<int32_t> info,
                            const int32_t* flags,
                            int64_t batch,
                            bool one_means_converged) {
    int32_t* out = info_ptr(info, batch);
    if (!out || !flags) return;
    const bool invert = one_means_converged;
    ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> id) {
            const int32_t flag = flags[id[0]];
            out[id[0]] = invert ? (flag != 0 ? 0 : 1) : (flag > 0 ? flag : 0);
        });
    });
}

// Kernel problem index -> caller's batch item (stedc's level driver runs nodes_per_item * batch problems).
inline int64_t info_item(int64_t problem, int64_t nodes_per_item) {
    return nodes_per_item > 1 ? problem / nodes_per_item : problem;
}

}  // namespace batchlas::detail
