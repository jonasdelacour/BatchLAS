#pragma once

// The triangular tile grid, packet type and MAC helpers the level-3 tile
// kernels share, in one place so they cannot drift on which half they visit.
// evidence: docs/perf/level3.md#syrk-and-syr2k-triangular-tiles-kernel-design

// <complex> is needed for std::conj even if a CUDA consumer included it first.
#include <complex>
#include <type_traits>
#include <sycl/sycl.hpp>

namespace batchlas::backend::detail {

// Replaces sycl::detail::is_complex, which is a DPC++ implementation detail and
// absent from other SYCL implementations.
template <typename T> struct is_std_complex : std::false_type {};
template <typename R> struct is_std_complex<std::complex<R>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_complex_v = is_std_complex<std::remove_cv_t<T>>::value;

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

// Conjugation is a no-op on a real scalar, which is what lets one kernel serve
// both the plain and the ^H spellings rather than two near-copies.
template <typename T>
inline T conj_if(const T& value) {
    if constexpr (is_std_complex_v<T>) {
        return std::conj(value);
    } else {
        return value;
    }
}

// accum + a * b, written out: std::complex operator* is the __mulsc3 libcall.
// Returns by value on purpose: a `T&` into the accumulator array spills it.
// evidence: docs/perf/level3.md#level-3-the-complex-mac-and-return-by-value-rules
template <typename T>
inline T accumulate(const T& accum, const T& a, const T& b) {
    if constexpr (is_std_complex_v<T>) {
        using Real = typename T::value_type;
        const Real ar = a.real();
        const Real ai = a.imag();
        const Real br = b.real();
        const Real bi = b.imag();
        return T(accum.real() + ar * br - ai * bi,
                 accum.imag() + ar * bi + ai * br);
    } else {
        return accum + a * b;
    }
}

template <typename T>
inline void tile_store4(T* p, const TileVec4<T>& in) {
    if constexpr (sizeof(T) == sizeof(float)) {
        tile_vec4(p) = in;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            p[i] = in.v[i];
        }
    }
}

// Four contiguous elements into registers, by value. Trap: 128-bit only for
// float; a wider reinterpret assumes an alignment local_accessor never promised.
// evidence: docs/perf/level3.md#level-3-the-complex-mac-and-return-by-value-rules
template <typename T>
inline TileVec4<T> tile_load4(const T* p) {
    if constexpr (sizeof(T) == sizeof(float)) {
        return tile_vec4(p);
    } else {
        TileVec4<T> out;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out.v[i] = p[i];
        }
        return out;
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
