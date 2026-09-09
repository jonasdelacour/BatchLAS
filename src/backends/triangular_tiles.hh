#pragma once

// The tile grid the triangular-output kernels share.
//
// SYRK and SYR2K both write one triangle of a symmetric n x n C, and both do it
// by indexing a 128x128 tile grid over the triangular tile set rather than the
// square one, so that a tile lying entirely outside the requested triangle is
// never launched. The decode, the tile geometry it counts, and the 128-bit
// vector type the staging rides on live here so the two kernels cannot drift
// apart on which half of the grid they visit.

// std::conj below needs <complex>. It compiled without it only because every
// consumer in the CUDA build happens to include <complex> earlier through
// linalg-impl.hh; including this header first, or from a non-CUDA translation
// unit, fails with "no member named 'conj' in namespace 'std'".
#include <complex>
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

// Conjugation is a no-op on a real scalar, which is what lets one kernel serve
// both the plain and the ^H spellings rather than two near-copies.
template <typename T>
inline T conj_if(const T& value) {
    if constexpr (sycl::detail::is_complex<T>::value) {
        return std::conj(value);
    } else {
        return value;
    }
}

// accum + a * b, written out rather than delegated to std::complex.
//
// `std::complex<float>::operator*` lowers to the __mulsc3 libcall, which
// implements C99 Annex G -- a branch on Inf and NaN around every single
// multiply. In the innermost loop of a GEMM that is ruinous and it is invisible
// in the source: the first complex build of the Gram kernel ran at 1.2 TFLOP/s
// against float's 13.8, and at n = 128 took 38 ms where a cuBLAS GEMM took 1.5.
// Four real multiplies and two adds is the whole operation, and it folds to the
// four FMAs a complex MAC should be; there is no exceptional case here worth a
// branch, because a NaN in the input is already a NaN in the answer.
//
// Returns the new accumulator rather than updating one through a reference.
// That is not a style choice: taking the address of an element of the
// register-resident accumulator array is enough for the compiler to stop
// believing it can stay in registers, and the array goes to local memory. It
// cost 43% on float at m = 512 (0.659 -> 0.944 ms) when this was first written
// with a `T&` out-parameter, with no other change.
template <typename T>
inline T accumulate(const T& accum, const T& a, const T& b) {
    if constexpr (sycl::detail::is_complex<T>::value) {
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

// Four contiguous elements into registers.
//
// For float the 4-wide packet is exactly a 128-bit LDS and the reinterpret is
// how we get one. For anything wider it is not: four doubles are 32 bytes and
// four complex<double> are 64, which no load form covers, and -- worse -- the
// reinterpret asserts an alignment `sycl::local_accessor` never promised, since
// it aligns to T and not to 4*sizeof(T). So wider scalars stay scalar. This is
// the only thing standing between these kernels and a misaligned access in
// double, and it is silent when wrong.
//
// Returned by value for the same reason `accumulate` is: an out-parameter array
// reference is an address the fragment registers do not survive.
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
