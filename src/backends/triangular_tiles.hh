#pragma once

// The tile grid the triangular-output kernels share.
//
// SYRK and SYR2K both write one triangle of a symmetric n x n C, and both do it
// by indexing a 128x128 tile grid over the triangular tile set rather than the
// square one, so that a tile lying entirely outside the requested triangle is
// never launched. The decode, the tile geometry it counts, and the 128-bit
// vector type the staging rides on live here so the two kernels cannot drift
// apart on which half of the grid they visit.

#include <sycl/sycl.hpp>

namespace batchlas::backend::detail {

// A vector of four T with the alignment the 128-bit load/store forms need.
template <typename T>
struct alignas(4 * sizeof(T)) TileVec4 {
    T v[4];
};

template <typename T>
inline const TileVec4<T>& tile_vec4(const T* p) {
    return *reinterpret_cast<const TileVec4<T>*>(p);
}

template <typename T>
inline TileVec4<T>& tile_vec4(T* p) {
    return *reinterpret_cast<TileVec4<T>*>(p);
}

// Four contiguous elements into registers.
//
// For float the 4-wide packet is exactly a 128-bit LDS and the reinterpret is
// how we get one. For anything wider it is not: four doubles are 32 bytes and
// four complex<double> are 64, which no load form covers, and -- worse -- the
// reinterpret asserts an alignment `sycl::local_accessor` never promised, since
// it aligns to T and not to 4*sizeof(T). So wider scalars stay scalar. This is
// the only thing standing between these kernels and a misaligned access in
// double, and it is silent when wrong.
template <typename T>
inline void tile_load4(const T* p, T (&out)[4]) {
    if constexpr (sizeof(T) == sizeof(float)) {
        const TileVec4<T>& v = tile_vec4(p);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[i] = v.v[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[i] = p[i];
        }
    }
}

inline constexpr int kTriangularTile = 128;
inline constexpr int kTriangularTileK = 8;

inline int triangular_tiles_per_side(int n) {
    return (n + kTriangularTile - 1) / kTriangularTile;
}

inline int triangular_tile_count(int n) {
    const int t = triangular_tiles_per_side(n);
    return t * (t + 1) / 2;
}

// Tile t of the lower triangle, packed row-major: t maps to (bi, bj) with
// bj <= bi via the inverse of bi*(bi+1)/2. sqrt is only a seed here; the two
// correction loops make the result exact regardless of what the device's sqrt
// rounds to. Uplo::Upper is the same set with the pair swapped.
inline void triangular_tile_decode(int tile, int& bi, int& bj) {
    bi = static_cast<int>((sycl::sqrt(8.0 * tile + 1.0) - 1.0) * 0.5);
    while (bi > 0 && bi * (bi + 1) / 2 > tile) {
        --bi;
    }
    while ((bi + 1) * (bi + 2) / 2 <= tile) {
        ++bi;
    }
    bj = tile - bi * (bi + 1) / 2;
}

} // namespace batchlas::backend::detail
