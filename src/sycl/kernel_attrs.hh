// SPDX-License-Identifier: MIT
#pragma once
// Kernel attributes and loop pragmas; on DPC++ each expands to exactly the spelling it replaced.
#include <src/sycl/sscp_target.hh>   // batchlas::sycl_impl::on_ptx(), target_arch()

#include <batchlas/backend_config.h>

#if BATCHLAS_SYCL_IMPL_ACPP

#define BATCHLAS_REQD_SG_SIZE(n)   // ignored by acpp: kernels run at the native sub-group size
#define BATCHLAS_LAUNCH_BOUNDS(max_threads, min_blocks)   // ignored: .maxntid = launch size
#define BATCHLAS_LAUNCH_BOUNDS_WITH_REQD_WG(max_threads, min_blocks)
#define BATCHLAS_UNROLL_FULL _Pragma("clang loop unroll(full)")   // SSCP JIT: bare unroll is partial

#else

#define BATCHLAS_REQD_SG_SIZE(n) [[sycl::reqd_sub_group_size(n)]]
// BATCHLAS_LAUNCH_BOUNDS goes FIRST, before any other attribute of operator(): on clang 24 it is
// a get(properties_tag) member, and an attribute in front of it would land on that.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)   // SPIR-V: breaks the icpx CPU AOT link
#if __has_cpp_attribute(intel::max_work_group_size)
#define BATCHLAS_LAUNCH_BOUNDS(max_threads, min_blocks) \
    [[intel::max_work_group_size(1, 1, max_threads), intel::min_work_groups_per_cu(min_blocks)]]
#define BATCHLAS_LAUNCH_BOUNDS_WITH_REQD_WG(max_threads, min_blocks) \
    BATCHLAS_LAUNCH_BOUNDS(max_threads, min_blocks)
#else
// clang 24 (b87b3566) dropped the attribute; the kernel property still emits .maxntid, and
// without it ptxas ignores .minnctapersm for its register budget (getrf tiny 64 -> 94 registers).
#define BATCHLAS_LAUNCH_BOUNDS(max_threads, min_blocks)                                         \
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {                           \
        return sycl::ext::oneapi::experimental::properties{                                     \
            sycl::ext::oneapi::experimental::max_work_group_size<1, 1, max_threads>};           \
    }                                                                                           \
    [[intel::min_work_groups_per_cu(min_blocks)]]
// Beside a live reqd_work_group_size(1, 1, max_threads) (the gemm and symm functors, under
// BATCHLAS_NVPTX_REQD_WG_OK): its .reqntid bounds the registers, and ptxas rejects .maxntid too.
#if BATCHLAS_NVPTX_REQD_WG_OK
#define BATCHLAS_LAUNCH_BOUNDS_WITH_REQD_WG(max_threads, min_blocks) \
    [[intel::min_work_groups_per_cu(min_blocks)]]
#else
#define BATCHLAS_LAUNCH_BOUNDS_WITH_REQD_WG(max_threads, min_blocks) \
    BATCHLAS_LAUNCH_BOUNDS(max_threads, min_blocks)
#endif
#endif
#else
#define BATCHLAS_LAUNCH_BOUNDS(max_threads, min_blocks)
#define BATCHLAS_LAUNCH_BOUNDS_WITH_REQD_WG(max_threads, min_blocks)
#endif
#define BATCHLAS_UNROLL_FULL _Pragma("unroll")

#endif
// BATCHLAS_UNROLL_FULL only on compile-time trip counts: unroll(full) ignores a runtime count.
