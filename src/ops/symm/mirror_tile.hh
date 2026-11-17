// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief Launch limits of the symm / hemm `mirror` family (symm_mirror.cc static_asserts them).

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::ops::mirror {

inline constexpr int kWg = 256;  ///< 16 x 16 lanes on both kernels

template <typename Scalar>  /// Output tile edge: wide v2's 64, the double tiled kernel's 16.
constexpr int tile() {
    return std::is_same_v<Scalar, double> ? 16 : 64;
}

template <typename Scalar>  /// k slab: 8 for complex<double> (16 would need 64 KiB), else 16.
constexpr int tile_k() {
    return std::is_same_v<Scalar, double> ? 16 : (sizeof(Scalar) == 16 ? 8 : 16);
}

template <typename Scalar>  /// Local memory, at most: two slabs of both operands, a 16-byte row pad.
constexpr std::int64_t slm_bytes() {
    if constexpr (std::is_same_v<Scalar, double>) return 2 * 16 * 16 * std::int64_t(sizeof(Scalar));
    else return 2 * std::int64_t(tile_k<Scalar>()) * (2 * 64 * std::int64_t(sizeof(Scalar)) + 16);
}

}  // namespace batchlas::ops::mirror
