#pragma once
// Kernel attributes and loop pragmas; on DPC++ each expands to exactly the spelling it replaced.
#include "sscp_target.hh"   // batchlas::sycl_impl::on_ptx(), target_arch()

#include <batchlas/backend_config.h>

#if BATCHLAS_SYCL_IMPL_ACPP

#define BATCHLAS_REQD_SG_SIZE(n)   // ignored by acpp: kernels run at the native sub-group size
#define BATCHLAS_LAUNCH_BOUNDS(max_threads, min_blocks)   // ignored: .maxntid = launch size
#define BATCHLAS_UNROLL_FULL _Pragma("clang loop unroll(full)")   // SSCP JIT: bare unroll is partial

#else

#define BATCHLAS_REQD_SG_SIZE(n) [[sycl::reqd_sub_group_size(n)]]
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)   // SPIR-V: breaks the icpx CPU AOT link
#define BATCHLAS_LAUNCH_BOUNDS(max_threads, min_blocks) \
    [[intel::max_work_group_size(1, 1, max_threads), intel::min_work_groups_per_cu(min_blocks)]]
#else
#define BATCHLAS_LAUNCH_BOUNDS(max_threads, min_blocks)
#endif
#define BATCHLAS_UNROLL_FULL _Pragma("unroll")

#endif
// BATCHLAS_UNROLL_FULL only on compile-time trip counts: unroll(full) ignores a runtime count.
