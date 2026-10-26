// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief Limits of the symm / hemm `mirror` family, sycl-free for can_run and the tuner.
/// The kernel (src/ops/symm/kernels/symm_mirror.cc) static_asserts them.

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::ops::mirror {

inline constexpr int kWg = 256;  ///< wide v2's 16 x 16 lanes; double's tiled kernel is 16 x 16 too

/// Output tile edge: wide v2's 64 (float, complex), the tiled kernel's 16 (double).
template <typename T>
constexpr int tile() {
    return std::is_same_v<T, double> ? 16 : 64;
}

/// k slab: 16, or 8 for complex<double> (two 64 x 16 slabs of 16-byte scalars are 64 KiB).
template <typename T>
constexpr int tile_k() {
    return std::is_same_v<T, double> ? 16 : (sizeof(T) == 16 ? 8 : 16);
}

/// Local memory, at most: two slabs of both operands, the dense one's k-contiguous rows padded by
/// 16 bytes on the element leg (wide v2), or the tiled kernel's two 16 x 16 tiles.
template <typename T>
constexpr std::int64_t slm_bytes() {
    if constexpr (std::is_same_v<T, double>) return 2 * 16 * 16 * std::int64_t(sizeof(T));
    else return 2 * std::int64_t(tile_k<T>()) * (2 * 64 * std::int64_t(sizeof(T)) + 16);
}

}  // namespace batchlas::ops::mirror
