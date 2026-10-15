#pragma once

// The 4-wide register value of the reg: and small gemm tiles: a clang ext_vector, or under
// BATCHLAS_DEVICE_NO_VECTOR_TYPES (Metal: its emitter has no vector types) a 16-byte struct
// with the members the kernels use (.x-.w, [i], scalar * Vec4).

#include <batchlas/backend_config.h>

namespace batchlas::sycl_gemm {

#if BATCHLAS_DEVICE_NO_VECTOR_TYPES

// Element-wise copies on purpose: LLVM folds a trivial global->private->local copy pair into
// a byte-loop memmove (AdaptiveCpp's emitter needs patch 0005 to compile it at all).
template <typename T>
struct alignas(4 * sizeof(T)) Vec4 {
    T x, y, z, w;
    Vec4() = default;
    Vec4(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}
    Vec4(const Vec4& o) : x(o.x), y(o.y), z(o.z), w(o.w) {}
    Vec4& operator=(const Vec4& o) {
        x = o.x, y = o.y, z = o.z, w = o.w;
        return *this;
    }
    T& operator[](int i) { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
    T operator[](int i) const { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
    friend Vec4 operator*(T a, const Vec4& v) { return {a * v.x, a * v.y, a * v.z, a * v.w}; }
};

#else

template <typename T>
using Vec4 = T __attribute__((ext_vector_type(4)));

#endif

} // namespace batchlas::sycl_gemm
