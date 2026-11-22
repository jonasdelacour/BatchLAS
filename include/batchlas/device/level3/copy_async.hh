// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Level 3 global-to-local copies that skip the registers, 4, 8 or 16 bytes each
/// (@ref api_device_lowlevel, not covered by the compatibility promise).
///
/// `cp.async` on CUDA `sm_80` and newer, a plain copy elsewhere; both are ordered by the barrier
/// after `copy_async_wait`.
/// @ingroup api_device_lowlevel

#include <sycl/sycl.hpp>

#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && defined(__SYCL_CUDA_ARCH__) && __SYCL_CUDA_ARCH__ >= 800
/// @brief 1 when the device pass compiles `cp.async` (CUDA `sm_80` and newer), else 0.
/// @ingroup api_device_lowlevel
#define BATCHLAS_GEMM_CP_ASYNC 1
#else
/// @brief 1 when the device pass compiles `cp.async` (CUDA `sm_80` and newer), else 0.
/// @ingroup api_device_lowlevel
#define BATCHLAS_GEMM_CP_ASYNC 0
#endif

namespace batchlas::device::l3 {

// evidence: docs/perf/gemm.md#gemm-wp-n2-the-u4-stack-frame-and-zero-filled-predicated-copies
/// @brief One `Bytes`-wide asynchronous copy: the first `src_bytes` bytes are read, the rest zero-filled.
///
/// `cp.async` on `sm_80` and newer (`.cg` for 16 bytes, `.ca` otherwise), a plain copy elsewhere;
/// either way the data is ready after `copy_async_wait` and a barrier.
/// @ingroup api_device_lowlevel
/// @tparam Bytes 4, 8 or 16, a whole number of `T`.
/// @pre `src` and `dst` are aligned to `Bytes`.
/// @param dst the local destination.
/// @param dst_local `dst`'s shared-window address (read by `cp.async` only).
/// @param src the global source; with `src_bytes == 0` nothing is read and `src` may be out of bounds.
/// @param src_bytes the bytes read, at most `Bytes`.
template <int Bytes, class T>
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
/// @brief Deleted: a `bool` byte count would read as `src_bytes = 1`.
/// @ingroup api_device_lowlevel
template <int Bytes, class T>
void copy_async_zfill(T*, std::uint32_t, const T*, bool) = delete;

/// @brief Closes this item's current group of asynchronous copies (nothing without `cp.async`).
/// @ingroup api_device_lowlevel
inline void copy_async_commit() {
#if BATCHLAS_GEMM_CP_ASYNC
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
/// @brief Waits until at most `N` of this item's committed copy groups are in flight (nothing without `cp.async`).
/// @ingroup api_device_lowlevel
/// @tparam N the groups that may stay pending.
template <int N>
inline void copy_async_wait() {
#if BATCHLAS_GEMM_CP_ASYNC
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
}

}  // namespace batchlas::device::l3
