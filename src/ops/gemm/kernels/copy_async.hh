#pragma once

// Global -> local copies that skip the registers: float 128x128 (4 B) and wide v2 128x128 (8, 16 B).

#include <sycl/sycl.hpp>

#include <cstdint>

namespace batchlas::sycl_gemm {

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && defined(__SYCL_CUDA_ARCH__) && __SYCL_CUDA_ARCH__ >= 800
#define BATCHLAS_GEMM_CP_ASYNC 1
#else
#define BATCHLAS_GEMM_CP_ASYNC 0
#endif

// One Bytes-wide copy (4, 8 or 16; src and dst aligned to Bytes): the first src_bytes are read and
// the rest zero-filled (0: nothing is read, src may be out of bounds). cp.async on sm_80+
// (dst_local is dst's shared-window address; .cg needs 16 bytes), a plain copy elsewhere, ordered
// the same way by the barrier after the wait.
// evidence: docs/perf/gemm.md#gemm-wp-n2-the-u4-stack-frame-and-zero-filled-predicated-copies
template <int Bytes, typename T>
inline void copy_async_zfill(T* dst, [[maybe_unused]] std::uint32_t dst_local, const T* src, std::uint32_t src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp.async copies 4, 8 or 16 bytes");
    static_assert(Bytes % sizeof(T) == 0, "whole elements per copy");
#if BATCHLAS_GEMM_CP_ASYNC
    if constexpr (Bytes == 16)
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(dst_local), "l"(src),
                     "r"(src_bytes) : "memory");
    else
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;" ::"r"(dst_local), "l"(src), "n"(Bytes),
                     "r"(src_bytes) : "memory");
#else
#pragma unroll
    for (int e = 0; e < int(Bytes / sizeof(T)); ++e) dst[e] = sizeof(T) * e < src_bytes ? src[e] : T(0);
#endif
}
// A bool would read as src_bytes = 1.
template <int Bytes, typename T>
void copy_async_zfill(T*, std::uint32_t, const T*, bool) = delete;

inline void copy_async_commit() {
#if BATCHLAS_GEMM_CP_ASYNC
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
// Wait until at most N of this thread's committed groups are in flight.
template <int N>
inline void copy_async_wait() {
#if BATCHLAS_GEMM_CP_ASYNC
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
}

}  // namespace batchlas::sycl_gemm
