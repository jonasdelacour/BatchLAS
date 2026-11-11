// SPDX-License-Identifier: MIT
#pragma once
// Level 3 packets (v22 design B.1): the 4-wide register value of the reg: and small gemm tiles
// (Vec4) and the aligned packet load of the stagers. Vec4 is a clang ext_vector, or under
// BATCHLAS_DEVICE_NO_VECTOR_TYPES (Metal: its emitter has no vector types) a 16-byte struct with
// the members the kernels use (.x-.w, [i], scalar * Vec4).

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

namespace batchlas::device::l3 {

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

template <typename T>
inline constexpr bool supports_packet4_v = std::is_same_v<T, float>;

template <typename T, int Width>
inline constexpr bool supports_packet_v = false;

template <typename T>
inline constexpr bool supports_packet_v<T, 4> = supports_packet4_v<T>;

template <typename T, int Width>
inline bool supports_aligned_packet_loads(const T* ptr, int ld, int stride) {
    if constexpr (!supports_packet_v<T, Width>) {
        static_cast<void>(ptr);
        static_cast<void>(ld);
        static_cast<void>(stride);
        return false;
    } else {
        const auto address = reinterpret_cast<std::uintptr_t>(ptr);
        return (address % alignof(sycl::vec<T, Width>) == 0) && (ld % Width == 0) && (stride % Width == 0);
    }
}

template <typename T, int Width>
inline sycl::vec<T, Width> packet_load_aligned(const T* ptr, int offset) {
    return *reinterpret_cast<const sycl::vec<T, Width>*>(ptr + offset);
}

}  // namespace batchlas::device::l3
