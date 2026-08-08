#pragma once

#include "math-helpers.hh"
#include "queue.hh"

#include <util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>

// Sizing and the fit ceiling for the scratch expansions in
// src/backends/triangular_expand.hh. They live here, outside src/backends/,
// because callers outside the backend need the *same* predicate the backend
// uses to pick its route: src/extensions/sytrd_blocked.cc has to know whether
// her2k will take its batched-GEMM route or its per-batch host loop before it
// decides to call her2k at all. triangular_expand.hh cannot be included from
// there -- it pulls cublasdx_dispatch_common.hh, whose <cuda_runtime_api.h> is
// unguarded, and sytrd_blocked.cc is also instantiated for ROCm and the host
// backend. Nothing below touches CUDA.
//
// One definition, not a copy: a call site that reimplemented these ceilings
// would drift from the backend's, and the failure mode of disagreeing is
// silent -- the caller believes it got the fast route and gets the host loop.
namespace batchlas::backend::detail {

// Leading dimension of an expanded copy. The caller's own ld is irrelevant --
// the expansion writes every element -- so pack the columns and pad only to
// 16 bytes, which is the alignment the vendor and cuBLASDx GEMM kernels want
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

    std::size_t budget = ctx.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE) / 4;
    if (const char* capped = std::getenv("BATCHLAS_EXPAND_MAX_BYTES")) {
        budget = std::min(budget, static_cast<std::size_t>(std::strtoull(capped, nullptr, 10)));
    }
    return bytes <= budget;
}

}  // namespace batchlas::backend::detail
