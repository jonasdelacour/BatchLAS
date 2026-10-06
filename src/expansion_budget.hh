#pragma once

#include "math-helpers.hh"
#include "queue.hh"

#include <batchlas/settings.hh>
#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>

// Sizing and the fit ceiling for the scratch expansions (src/backends/triangular_expand.hh)
// and the herk/her2k fold's product (src/backends/accumulate_hermitian.hh). The expanding
// families' can_run (src/ops/{symm,hemm,trmm,herk,her2k}) call expansion_fits, so a family is
// refused exactly where its scratch cannot be built. Nothing below touches CUDA.
namespace batchlas::backend::detail {

// Leading dimension of an expanded copy. The caller's own ld is irrelevant --
// the expansion writes every element -- so pack the columns and pad only to
// 16 bytes, which is the alignment the vendor and native GEMM kernels want
// before they will use packet loads.
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

// Whether an n x n x batch expansion can be built at all. Two ceilings, both
// hard rather than tuned:
//
//   - SYCL linearises the global id, and the runtime rejects a range whose
//     product does not fit in an int. The grid is one work item per element, so
//     it hits that at 2^31 elements -- measured, as a thrown sycl::exception at
//     n = 2048 batch = 512.
//   - The scratch shares the device with A, B and C, which for a square problem
//     are together about three times its size. A quarter of global memory
//     leaves room for them; at n = 2048 batch = 256 that is 4.3 GB of scratch
//     inside 17 GB of live operands, which runs.
//
// A caller that exceeds either has to fall back to whatever route needs no
// scratch.
//
// BATCHLAS_EXPAND_MAX_BYTES lowers the memory ceiling, for sharing a device
// with something else -- and for reaching the no-scratch fallback from a test
// without allocating gigabytes to get there.
inline bool expansion_fits(const Queue& ctx, int n, int batch, std::size_t bytes) {
    const std::size_t elements = static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(batch);
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    // The default is a DEVICE property, so the Settings field carries only the
    // override and the strtoull stays here beside the min() it feeds.
    std::size_t budget = ctx.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE) / 4;
    if (const char* capped = batchlas::settings().geometry.expand_max_bytes.get()) {
        budget = std::min(budget, static_cast<std::size_t>(std::strtoull(capped, nullptr, 10)));
    }
    return bytes <= budget;
}

// Work-group shape of expand_triangular and accumulate_hermitian: rows first, so that a
// group's lanes walk a column and both the load and the store coalesce, and
// only as many rows as the matrix actually has, so that a batch of tiny
// matrices does not retire mostly-idle groups.
struct ExpandGroupShape {
    int rows;
    int cols;
};

inline ExpandGroupShape expand_group_shape(int n) {
    constexpr int kItemsPerGroup = 256;
    constexpr int kMaxGroupRows = 32;
    int rows = 1;
    while (rows < kMaxGroupRows && rows < n) {
        rows *= 2;
    }
    return {rows, kItemsPerGroup / rows};
}

// The padded range of an expand_group_shape launch fits an int (-fsycl-id-queries-fit-in-int
// throws at submit otherwise); tighter than expansion_fits' n^2 batch term.
// evidence: docs/perf/level3.md#the-padded-launch-range
inline bool expand_grid_fits(int n, int batch) {
    const auto shape = expand_group_shape(n);
    const std::size_t range = static_cast<std::size_t>(batch) *
                              static_cast<std::size_t>(::batchlas::internal::ceil_div(n, shape.cols) * shape.cols) *
                              static_cast<std::size_t>(::batchlas::internal::ceil_div(n, shape.rows) * shape.rows);
    return range <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

}  // namespace batchlas::backend::detail
