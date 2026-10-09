#pragma once

// Counter-based Philox4x32-10 (Salmon et al., SC'11; Random123 constants). Every value is a pure
// function of (seed, stream, index): no state, no discard loop, and the same bits on host, DPC++
// and AdaptiveCpp. Only the uniform paths are bit-exact across them; normal() goes through the
// implementation's log/sin/cos.

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::philox {

struct Words {
    std::uint32_t w[4];
};

inline constexpr std::uint32_t kMul0 = 0xD2511F53u;
inline constexpr std::uint32_t kMul1 = 0xCD9E8D57u;
inline constexpr std::uint32_t kWeyl0 = 0x9E3779B9u;
inline constexpr std::uint32_t kWeyl1 = 0xBB67AE85u;
inline constexpr int kRounds = 10;

constexpr Words philox4x32_10(Words ctr, std::uint32_t k0, std::uint32_t k1) noexcept {
    for (int r = 0; r < kRounds; ++r) {
        const std::uint64_t p0 = static_cast<std::uint64_t>(kMul0) * ctr.w[0];
        const std::uint64_t p1 = static_cast<std::uint64_t>(kMul1) * ctr.w[2];
        const auto hi0 = static_cast<std::uint32_t>(p0 >> 32);
        const auto lo0 = static_cast<std::uint32_t>(p0);
        const auto hi1 = static_cast<std::uint32_t>(p1 >> 32);
        const auto lo1 = static_cast<std::uint32_t>(p1);
        ctr = Words{{hi1 ^ ctr.w[1] ^ k0, lo1, hi0 ^ ctr.w[3] ^ k1, lo0}};
        k0 += kWeyl0;
        k1 += kWeyl1;
    }
    return ctr;
}

// Key = seed, counter = (index, stream): one block of 128 random bits per (seed, stream, index).
constexpr Words draw(std::uint64_t seed, std::uint64_t stream, std::uint64_t index) noexcept {
    return philox4x32_10(Words{{static_cast<std::uint32_t>(index), static_cast<std::uint32_t>(index >> 32),
                                static_cast<std::uint32_t>(stream), static_cast<std::uint32_t>(stream >> 32)}},
                         static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32));
}

// Integer mantissa of component `slot` (0 or 1): 24 bits from one word for float, 53 bits from
// two words for double, so a complex<double> consumes the whole block.
template <typename R>
constexpr std::uint64_t mantissa_bits(const Words& b, int slot) noexcept {
    static_assert(std::is_same_v<R, float> || std::is_same_v<R, double>, "float or double");
    if constexpr (std::is_same_v<R, float>) {
        return b.w[slot] >> 8;
    } else {
        return ((static_cast<std::uint64_t>(b.w[2 * slot]) << 32) | b.w[2 * slot + 1]) >> 11;
    }
}

template <typename R>
inline constexpr int kMantissaDigits = std::is_same_v<R, float> ? 24 : 53;

template <typename R>
constexpr R ulp_scale() noexcept {
    return R(1) / static_cast<R>(std::uint64_t(1) << kMantissaDigits<R>);
}

// [0, 1), exact multiples of 2^-24 (float) or 2^-53 (double).
template <typename R>
constexpr R unit(const Words& b, int slot) noexcept {
    return static_cast<R>(mantissa_bits<R>(b, slot)) * ulp_scale<R>();
}

// [-1, 1), exact: k * 2^(1-digits) - 1 needs no more bits than the mantissa holds.
template <typename R>
constexpr R symmetric(const Words& b, int slot) noexcept {
    return static_cast<R>(mantissa_bits<R>(b, slot)) * (R(2) * ulp_scale<R>()) - R(1);
}

template <typename T>
struct is_complex : std::false_type {};
template <typename R>
struct is_complex<std::complex<R>> : std::true_type {};

template <typename T>
struct real_of {
    using type = T;
};
template <typename R>
struct real_of<std::complex<R>> {
    using type = R;
};

// Each component uniform in [-1, 1); real and imaginary parts come from disjoint bits.
template <typename T>
constexpr T uniform_symmetric(std::uint64_t seed, std::uint64_t stream, std::uint64_t index) noexcept {
    using R = typename real_of<T>::type;
    const Words b = draw(seed, stream, index);
    if constexpr (is_complex<T>::value) {
        return T(symmetric<R>(b, 0), symmetric<R>(b, 1));
    } else {
        return symmetric<R>(b, 0);
    }
}

// Each component uniform in [0, 1).
template <typename T>
constexpr T uniform_unit(std::uint64_t seed, std::uint64_t stream, std::uint64_t index) noexcept {
    using R = typename real_of<T>::type;
    const Words b = draw(seed, stream, index);
    if constexpr (is_complex<T>::value) {
        return T(unit<R>(b, 0), unit<R>(b, 1));
    } else {
        return unit<R>(b, 0);
    }
}

// Box-Muller on one block. Each component is N(0, 1) (a complex value has E|z|^2 = 2).
template <typename T>
inline T normal(std::uint64_t seed, std::uint64_t stream, std::uint64_t index) {
    using R = typename real_of<T>::type;
    const Words b = draw(seed, stream, index);
    const R u1 = R(1) - unit<R>(b, 0);  // (0, 1]: log stays finite
    const R angle = R(6.283185307179586476925286766559) * unit<R>(b, 1);
    const R radius = sycl::sqrt(R(-2) * sycl::log(u1));
    if constexpr (is_complex<T>::value) {
        return T(radius * sycl::cos(angle), radius * sycl::sin(angle));
    } else {
        return radius * sycl::cos(angle);
    }
}

}  // namespace batchlas::philox
