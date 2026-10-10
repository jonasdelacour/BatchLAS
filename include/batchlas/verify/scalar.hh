// SPDX-License-Identifier: MIT
// batchlas::verify scalars: every check promotes to double / complex<double> (docs/design/verification.md).
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace batchlas::verify {

template <class T> struct is_complex : std::false_type {};
template <class R> struct is_complex<std::complex<R>> : std::true_type {};

template <class T> struct real_of { using type = T; };
template <class R> struct real_of<std::complex<R>> { using type = R; };

template <class T> using real_t = typename real_of<T>::type;
template <class T> using promoted_t = std::conditional_t<is_complex<T>::value, std::complex<double>, double>;

inline double up(float x) { return double(x); }
inline double up(double x) { return x; }
inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
inline std::complex<double> up(std::complex<double> x) { return x; }

inline double abs(double x) { return std::fabs(x); }
inline double abs(std::complex<double> x) { return std::abs(x); }
inline float conj(float x) { return x; }
inline double conj(double x) { return x; }
inline std::complex<float> conj(std::complex<float> x) { return std::conj(x); }
inline std::complex<double> conj(std::complex<double> x) { return std::conj(x); }

/// |re| + |im| (LAPACK's cabs1, the partial-pivoting metric); |x| for a real x.
template <class T> double cabs1(T x) {
    const auto z = up(x);
    if constexpr (is_complex<T>::value) return std::fabs(z.real()) + std::fabs(z.imag());
    else return std::fabs(z);
}

/// True when every component of @p x is finite.
template <class T> bool finite(T x) {
    const auto z = up(x);
    if constexpr (is_complex<T>::value) return std::isfinite(z.real()) && std::isfinite(z.imag());
    else return std::isfinite(z);
}

template <class T> T make(double re, double im) {
    if constexpr (is_complex<T>::value) return T(real_t<T>(re), real_t<T>(im));
    else { (void)im; return T(re); }
}

/// make<T>(z.real(), z.imag()): a promoted value rounded back to T (the imaginary part dropped for a real T).
template <class T> T make(std::complex<double> z) { return make<T>(z.real(), z.imag()); }

/// Unit roundoff of T's real type: 2^-24 for float, 2^-53 for double.
template <class T> constexpr double eps() { return std::numeric_limits<real_t<T>>::epsilon() * 0.5; }

// std::max returns its first argument when the second is NaN, so a poisoned residual would read
// as a perfect one. Every maximum in this library goes through here.
inline double nanmax(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return a > b ? a : b;
}

// factor_bench.cc's historical LCG (its constants and draws, unchanged): seeded sequences, and so
// archived sweeps and ledger records, reproduce across harnesses.
class Rng {
public:
    explicit Rng(std::uint64_t seed) : s_(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    /// Uniform in [-1, 1).
    double next() {
        s_ = s_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(std::int32_t(std::uint32_t(s_ >> 32))) / 2147483648.0;
    }
    double uniform01() { const double u = next() * 0.5 + 0.5; return u <= 0.0 ? 1e-12 : u; }
    double normal() { return std::sqrt(-2.0 * std::log(uniform01())) * std::cos(6.283185307179586 * uniform01()); }

private:
    std::uint64_t s_;
};

}  // namespace batchlas::verify
