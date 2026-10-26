// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief Launch limits of the symm / hemm `mirror` family (symm_mirror.cc static_asserts them).

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::ops::mirror {

inline constexpr int kWg = 256;  ///< 16 x 16 lanes on both kernels

template <typename T>  /// Output tile edge: wide v2's 64, the double tiled kernel's 16.
constexpr int tile() {
    return std::is_same_v<T, double> ? 16 : 64;
}

template <typename T>  /// k slab: 8 for complex<double> (16 would need 64 KiB), else 16.
constexpr int tile_k() {
    return std::is_same_v<T, double> ? 16 : (sizeof(T) == 16 ? 8 : 16);
}

template <typename T>  /// Local memory, at most: two slabs of both operands, a 16-byte row pad.
constexpr std::int64_t slm_bytes() {
    if constexpr (std::is_same_v<T, double>) return 2 * 16 * 16 * std::int64_t(sizeof(T));
    else return 2 * std::int64_t(tile_k<T>()) * (2 * 64 * std::int64_t(sizeof(T)) + 16);
}

}  // namespace batchlas::ops::mirror
