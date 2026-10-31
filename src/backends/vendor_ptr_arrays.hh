// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/workspace.hh>
#include <src/queue.hh>
#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace batchlas::detail {

/// The device T*[batch] arrays a pointer-array vendor call (cublas<t>trsmBatched, ...) takes,
/// one per view, in one lease on the queue's arena.
template <typename T, std::size_t N>
struct VendorPtrArrays {
    WorkspaceLease lease;
    std::array<T**, N> ptrs{};
    T** operator[](std::size_t i) const { return ptrs[i]; }
};

/// Fills one pointer array per view from its base and stride, never the view's own data_ptrs():
/// a view over the caller's USM has none, and two slices of one Matrix share theirs, so filling
/// the second repoints the first. Keep the result alive until the vendor call is enqueued; the
/// arena's next borrower is ordered after it (in-order queue, or the drain an out-of-order release does).
template <typename T, typename... Views>
VendorPtrArrays<T, sizeof...(Views)> vendor_ptr_arrays(Queue& ctx, const Views&... views) {
    constexpr std::size_t N = sizeof...(Views);
    const std::array<const MatrixView<T, MatrixFormat::Dense>*, N> vs{&views...};
    const std::size_t batch = static_cast<std::size_t>(vs[0]->batch_size());
    auto sizer = BumpAllocator::measuring();
    for (std::size_t v = 0; v < N; ++v) (void)sizer.allocate<T*>(ctx, batch);
    VendorPtrArrays<T, N> out{ctx.workspace(sizer.required_bytes())};
    BumpAllocator pool(out.lease.span());
    std::array<T*, N> base{};
    std::array<std::int64_t, N> stride{};
    for (std::size_t v = 0; v < N; ++v) {
        out.ptrs[v] = pool.allocate<T*>(ctx, batch).data();
        base[v] = vs[v]->data_ptr();
        stride[v] = vs[v]->stride();
    }
    const auto ptrs = out.ptrs;
    ctx->parallel_for(sycl::range<1>(batch), [=](sycl::id<1> id) {
        const std::size_t b = id[0];
        for (std::size_t v = 0; v < N; ++v) ptrs[v][b] = base[v] + static_cast<std::int64_t>(b) * stride[v];
    });
    if (!ctx.in_order()) ctx.wait();  // the native call is not ordered after the fill
    return out;
}

}  // namespace batchlas::detail
