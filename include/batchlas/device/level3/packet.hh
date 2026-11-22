// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Level 3 packets of the tuned gemm: the 4-wide register value `l3::Vec4` and the stagers'
/// aligned packet load (@ref api_device_lowlevel, not covered by the compatibility promise).
///
/// `Vec4` is a clang `ext_vector_type(4)`, or, under `BATCHLAS_DEVICE_NO_VECTOR_TYPES` (Metal,
/// whose emitter has no vector types), a 16-byte struct with the members the kernels use
/// (`.x` to `.w`, `[i]`, scalar times `Vec4`).
/// @ingroup api_device_lowlevel

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/config.hh>

namespace batchlas::device::l3 {

#if BATCHLAS_DEVICE_NO_VECTOR_TYPES

// Element-wise copies on purpose: LLVM folds a trivial global->private->local copy pair into
// a byte-loop memmove (AdaptiveCpp's emitter needs patch 0005 to compile it at all).
/// @brief A 4-wide register value of `T`: the struct form for targets without vector types.
/// @ingroup api_device_lowlevel
template <class T>
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

/// @brief A 4-wide register value of `T`, a clang `ext_vector_type(4)`.
/// @ingroup api_device_lowlevel
template <class T>
using Vec4 = T __attribute__((ext_vector_type(4)));

#endif

/// @brief `true` when 4-wide packet loads of `T` are supported (`float` only).
/// @ingroup api_device_lowlevel
template <class T>
inline constexpr bool supports_packet4_v = std::is_same_v<T, float>;

/// @brief `true` when `Width`-wide packet loads of `T` are supported.
/// @ingroup api_device_lowlevel
template <class T, int Width>
inline constexpr bool supports_packet_v = false;

/// @brief `true` when 4-wide packet loads of `T` are supported.
/// @ingroup api_device_lowlevel
template <class T>
inline constexpr bool supports_packet_v<T, 4> = supports_packet4_v<T>;

/// @brief Whether a batch at `ptr` may be staged with `Width`-wide aligned packet loads.
/// @ingroup api_device_lowlevel
/// @tparam Width the packet width.
/// @param ptr the batch's first element.
/// @param ld the leading dimension.
/// @param stride the batch stride.
/// @return `true` when packets of `T` are supported, `ptr` is aligned to a packet, and `ld` and
///   `stride` are multiples of `Width`.
template <class T, int Width>
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

/// @brief One aligned `Width`-wide load of `ptr[offset]` to `ptr[offset + Width - 1]`.
/// @ingroup api_device_lowlevel
/// @pre `ptr + offset` is aligned to `sycl::vec<T, Width>`.
/// @tparam Width the packet width.
/// @param ptr the base pointer.
/// @param offset the first element's offset.
template <class T, int Width>
inline sycl::vec<T, Width> packet_load_aligned(const T* ptr, int offset) {
    return *reinterpret_cast<const sycl::vec<T, Width>*>(ptr + offset);
}

}  // namespace batchlas::device::l3
