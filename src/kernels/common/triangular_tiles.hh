// SPDX-License-Identifier: MIT
#pragma once

// Tile grid sizes for the level-3 tile kernels and can_run; device side: v2 ext/l3-triangular-tiles.hh.
// evidence: docs/perf/level3.md#syrk-and-syr2k-triangular-tiles-kernel-design

#include <complex>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <batchlas/backend_config.h>
#include <batchlas/blas/device/detail/group_blas_common.hh>
#include <src/sycl/kernel_attrs.hh>

namespace batchlas::backend::detail {

// Replaces sycl::detail::is_complex, a DPC++ detail other SYCL implementations lack.
template <typename T> struct is_std_complex : std::false_type {};
template <typename R> struct is_std_complex<std::complex<R>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_complex_v = is_std_complex<std::remove_cv_t<T>>::value;

inline constexpr int kTriangularTile = 128;
inline constexpr int kTriangularTileK = 8;

inline int triangular_tiles_per_side(int n) {
    return (n + kTriangularTile - 1) / kTriangularTile;
}

inline int triangular_tile_count(int n) {
    const int t = triangular_tiles_per_side(n);
    return t * (t + 1) / 2;
}

} // namespace batchlas::backend::detail
