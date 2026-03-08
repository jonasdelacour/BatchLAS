#pragma once

#include <cstdint>
#include <type_traits>
#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

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

} // namespace batchlas::sycl_gemm