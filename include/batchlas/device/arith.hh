// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Device arithmetic: the Arith trait, the `arith::` policies a routine's loop takes, the
/// scalar helpers built on them, and the store epilogue.
/// @ingroup api_device_arith
///
/// Arith has one specialisation per scalar kind: real, `std::complex`, and any POD {re, im}
/// complex such as Cx. The policies are Explicit, Native, ExplicitInline, Fma and FmaSplit; the
/// helpers include abs1, abs2, conj, select, tag_mul, tag_mac and store_epi.

#include <complex>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>

namespace batchlas::device {

/// @brief A plain aggregate complex for device code.
/// @ingroup api_device_arith
///
/// It is layout-compatible with `std::complex<Real>` (a launcher may reinterpret_cast at the
/// pointer boundary) and free of Annex-G operators.
/// @tparam Real float or double
template <class Real> struct Cx {
    Real re;  ///< The real part.
    Real im;  ///< The imaginary part.
};

namespace detail {
// A POD {re, im} complex: two data members of one floating type, trivially copyable.
template <class C>
concept PodComplex = requires { &C::re; &C::im; } && std::is_same_v<decltype(C::re), decltype(C::im)> &&
                     std::is_floating_point_v<decltype(C::re)> && std::is_trivially_copyable_v<C>;
}  // namespace detail

/// @brief The arithmetic trait: every inner loop multiplies through Arith only.
/// @ingroup api_device_arith
///
/// For a complex Scalar it is explicit real arithmetic, never the Annex-G `operator*` (isnan
/// branches and __mulsc3). Any POD {re, im} aggregate (Cx, or the library's own device scalar)
/// gets the explicit complex Arith (ComplexArith); specialise Arith for another scalar type.
/// @tparam Scalar the scalar type
template <class Scalar> struct Arith {
    using real_t = Scalar;  ///< The real type of Scalar.
    static constexpr bool is_complex = false;  ///< False here; true for a complex Scalar.
    /// @brief a * b.
    BATCHLAS_DEVICE_INLINE static Scalar mul(Scalar a, Scalar b) { return a * b; }
    /// @brief acc += a * b.
    BATCHLAS_DEVICE_INLINE static void mac(Scalar& acc, Scalar a, Scalar b) { acc += a * b; }
    /// @brief acc += conj(a) * b.
    BATCHLAS_DEVICE_INLINE static void mac_conj(Scalar& acc, Scalar a, Scalar b) { acc += a * b; }
    /// @brief conj(a); the identity on a real Scalar.
    BATCHLAS_DEVICE_INLINE static Scalar conj(Scalar a) { return a; }
    /// @brief The real part.
    BATCHLAS_DEVICE_INLINE static Scalar re(Scalar a) { return a; }
    /// @brief The imaginary part; 0 for a real Scalar.
    BATCHLAS_DEVICE_INLINE static Scalar im(Scalar) { return Scalar(0); }
    /// @brief The scalar (r, i); r for a real Scalar.
    BATCHLAS_DEVICE_INLINE static Scalar make(Scalar r, Scalar) { return r; }
    /// @brief a + b.
    BATCHLAS_DEVICE_INLINE static Scalar add(Scalar a, Scalar b) { return a + b; }
    /// @brief a - b.
    BATCHLAS_DEVICE_INLINE static Scalar sub(Scalar a, Scalar b) { return a - b; }
    /// @brief -a.
    BATCHLAS_DEVICE_INLINE static Scalar neg(Scalar a) { return -a; }
    /// @brief 0.
    BATCHLAS_DEVICE_INLINE static Scalar zero() { return Scalar(0); }
    /// @brief 1.
    BATCHLAS_DEVICE_INLINE static Scalar one() { return Scalar(1); }
    /// @brief c -= a * b as ONE expression: it contracts to one FMA, where two statements would
    /// round twice.
    BATCHLAS_DEVICE_INLINE static void msub(Scalar& c, const Scalar& a, const Scalar& b) { c = c - a * b; }
    /// @brief a / b.
    BATCHLAS_DEVICE_INLINE static Scalar div(const Scalar& a, const Scalar& b) { return a / b; }
    /// @brief 1 / a.
    BATCHLAS_DEVICE_INLINE static Scalar recip(const Scalar& a) { return Scalar(1) / a; }
    /// @brief a / s for a real s.
    BATCHLAS_DEVICE_INLINE static Scalar div_real(const Scalar& a, const Scalar& s) { return a / s; }
    /// @brief s * a for a real s.
    BATCHLAS_DEVICE_INLINE static Scalar scale_real(const Scalar& s, const Scalar& a) { return s * a; }
    /// @brief True when a is exactly zero.
    BATCHLAS_DEVICE_INLINE static bool is_zero(const Scalar& a) { return a == Scalar(0); }
    /// @brief True when a is finite (both parts, for a complex Scalar).
    BATCHLAS_DEVICE_INLINE static bool isfinite(const Scalar& a) { return sycl::isfinite(a); }
};

namespace detail {
// The field access of a complex kind, by reference (one part loaded at a time): std::complex
// (.real() / .imag(), Scalar(r, i)) and a POD {re, im} (.re / .im, Scalar{r, i}).
template <class Scalar> struct StdComplexParts {
    using R = typename Scalar::value_type;
    static constexpr bool pod = false;
    BATCHLAS_DEVICE_INLINE static R re(const Scalar& a) { return a.real(); }
    BATCHLAS_DEVICE_INLINE static R im(const Scalar& a) { return a.imag(); }
    BATCHLAS_DEVICE_INLINE static Scalar make(R r, R i) { return Scalar(r, i); }
};
template <class C> struct PodComplexParts {
    using R = decltype(C::re);
    static constexpr bool pod = true;
    BATCHLAS_DEVICE_INLINE static R re(const C& a) { return a.re; }
    BATCHLAS_DEVICE_INLINE static R im(const C& a) { return a.im; }
    BATCHLAS_DEVICE_INLINE static C make(R r, R i) { return C{r, i}; }
};
}  // namespace detail

/// @brief The complex Arith, written once over a Parts accessor (`std::complex` or a POD {re, im}).
/// @ingroup api_device_lowlevel
///
/// `Arith<std::complex<Real>>` and Arith of a POD complex derive from it; the members are Arith's.
/// div and recip are Smith's algorithm (no Annex-G, no (c - di) / (c^2 + d^2) overflow).
/// @tparam T_ the complex type
/// @tparam P the parts accessor
// Where the two kinds' legacy kernels spelled a member differently (mac / mac_conj: split
// temporaries vs compound updates; msub, div and recip: parts read into locals first vs in place;
// scale_real: s * x vs x * s) each keeps its form: under -ffp-contract=on and the register
// allocator these are not the same code.
template <class T_, class P> struct ComplexArith {
    using T = T_;  ///< The complex type.
    using R = typename P::R;  ///< Its real type.
    using real_t = R;  ///< Its real type.
    using parts = P;  ///< The parts accessor.
    static constexpr bool is_complex = true;  ///< True.
    /// @brief a * b.
    BATCHLAS_DEVICE_INLINE static T mul(T a, T b) {
        return P::make(P::re(a) * P::re(b) - P::im(a) * P::im(b), P::re(a) * P::im(b) + P::im(a) * P::re(b));
    }
    /// @brief acc += a * b.
// std::complex: the latrd split-loop form, bit-identical to the legacy kernel's mac().
    BATCHLAS_DEVICE_INLINE static void mac(T& acc, T a, T b) {
        if constexpr (P::pod) {
            acc.re += a.re * b.re - a.im * b.im;
            acc.im += a.re * b.im + a.im * b.re;
        } else {
            const R pr = a.real() * b.real() - a.imag() * b.imag();
            const R pi = a.real() * b.imag() + a.imag() * b.real();
            acc = T(acc.real() + pr, acc.imag() + pi);
        }
    }
    /// @brief acc += conj(a) * b.
    BATCHLAS_DEVICE_INLINE static void mac_conj(T& acc, T a, T b) {
        if constexpr (P::pod) {
            acc.re += a.re * b.re + a.im * b.im;
            acc.im += a.re * b.im - a.im * b.re;
        } else {
            const R pr = a.real() * b.real() + a.imag() * b.imag();
            const R pi = a.real() * b.imag() - a.imag() * b.real();
            acc = T(acc.real() + pr, acc.imag() + pi);
        }
    }
    /// @brief conj(a).
    BATCHLAS_DEVICE_INLINE static T conj(T a) { return P::make(P::re(a), -P::im(a)); }
    /// @brief The real part.
    BATCHLAS_DEVICE_INLINE static R re(T a) { return P::re(a); }
    /// @brief The imaginary part.
    BATCHLAS_DEVICE_INLINE static R im(T a) { return P::im(a); }
    /// @brief The scalar (r, i).
    BATCHLAS_DEVICE_INLINE static T make(R r, R i) { return P::make(r, i); }
    /// @brief a + b.
    BATCHLAS_DEVICE_INLINE static T add(T a, T b) { return P::make(P::re(a) + P::re(b), P::im(a) + P::im(b)); }
    /// @brief a - b.
    BATCHLAS_DEVICE_INLINE static T sub(T a, T b) { return P::make(P::re(a) - P::re(b), P::im(a) - P::im(b)); }
    /// @brief -a.
    BATCHLAS_DEVICE_INLINE static T neg(T a) { return P::make(-P::re(a), -P::im(a)); }
    /// @brief 0.
    BATCHLAS_DEVICE_INLINE static T zero() { return P::make(R(0), R(0)); }
    /// @brief 1.
    BATCHLAS_DEVICE_INLINE static T one() { return P::make(R(1), R(0)); }
    /// @brief c -= a * b.
    // FmaArith<C> overrides msub / div / recip with the fma forms. On the library's device
    // scalar, div_real / scale_real / is_zero / isfinite are bit for bit its dev_div_real /
    // dev_mul_real / dev_is_zero / dev_isfinite.
    BATCHLAS_DEVICE_INLINE static void msub(T& c, const T& a, const T& b) {
        if constexpr (P::pod) {
            c = T{c.re - (a.re * b.re - a.im * b.im), c.im - (a.re * b.im + a.im * b.re)};
        } else {
            const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
            c = T(c.real() - (ar * br - ai * bi), c.imag() - (ar * bi + ai * br));
        }
    }
    /// @brief a / b by Smith's algorithm: no Annex-G and no (c - di) / (c^2 + d^2) overflow past
    /// ~1e19 (float) or ~1e154 (double).
    BATCHLAS_DEVICE_INLINE static T div(const T& a, const T& b) {
        if constexpr (P::pod) {
            if (sycl::fabs(b.re) >= sycl::fabs(b.im)) {
                const R r = b.im / b.re, den = b.re + b.im * r;
                return T{(a.re + a.im * r) / den, (a.im - a.re * r) / den};
            }
            const R r = b.re / b.im, den = b.im + b.re * r;
            return T{(a.re * r + a.im) / den, (a.im * r - a.re) / den};
        } else {
            const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
            if (sycl::fabs(br) >= sycl::fabs(bi)) {
                const R r = bi / br, den = br + bi * r;
                return T((ar + ai * r) / den, (ai - ar * r) / den);
            }
            const R r = br / bi, den = bi + br * r;
            return T((ar * r + ai) / den, (ai * r - ar) / den);
        }
    }
    /// @brief 1 / d by Smith's algorithm.
    BATCHLAS_DEVICE_INLINE static T recip(const T& d) {
        if constexpr (P::pod) {
            if (sycl::fabs(d.re) >= sycl::fabs(d.im)) {
                const R r = d.im / d.re, den = d.re + d.im * r;
                return T{R(1) / den, -r / den};
            }
            const R r = d.re / d.im, den = d.im + d.re * r;
            return T{r / den, R(-1) / den};
        } else {
            const R dr = d.real(), di = d.imag();
            if (sycl::fabs(dr) >= sycl::fabs(di)) {
                const R r = di / dr, den = dr + di * r;
                return T(R(1) / den, -r / den);
            }
            const R r = dr / di, den = di + dr * r;
            return T(r / den, R(-1) / den);
        }
    }
    /// @brief a / s for a real s.
    BATCHLAS_DEVICE_INLINE static T div_real(const T& a, const R& s) { return P::make(P::re(a) / s, P::im(a) / s); }
    /// @brief s * a for a real s.
    BATCHLAS_DEVICE_INLINE static T scale_real(const R& s, const T& a) {
        if constexpr (P::pod) return P::make(P::re(a) * s, P::im(a) * s);
        else return P::make(s * P::re(a), s * P::im(a));
    }
    /// @brief True when both parts of a are exactly zero.
    // The && pair reads its fields directly: through an accessor the second load is not
    // speculated and lands behind a branch (getri's Cx perm kernel, +1 BRA).
    BATCHLAS_DEVICE_INLINE static bool is_zero(const T& a) {
        if constexpr (P::pod) return a.re == R(0) && a.im == R(0);
        else return a.real() == R(0) && a.imag() == R(0);
    }
    /// @brief True when both parts of a are finite.
    BATCHLAS_DEVICE_INLINE static bool isfinite(const T& a) {
        if constexpr (P::pod) return sycl::isfinite(a.re) && sycl::isfinite(a.im);
        else return sycl::isfinite(a.real()) && sycl::isfinite(a.imag());
    }
};

/// @cond
template <class Real> struct Arith<std::complex<Real>> : ComplexArith<std::complex<Real>, detail::StdComplexParts<std::complex<Real>>> {};
template <class C> requires detail::PodComplex<C> struct Arith<C> : ComplexArith<C, detail::PodComplexParts<C>> {};
/// @endcond

/// @brief The real type of Scalar (cv removed): Scalar itself, or the complex type's value type.
/// @ingroup api_device_arith
/// @tparam Scalar the scalar type
template <class Scalar> using real_t = typename Arith<std::remove_cv_t<Scalar>>::real_t;

/// @brief True when Scalar is complex (`std::complex` or a POD {re, im}): the device-side question.
/// @ingroup api_device_arith
///
/// Host code asks batchlas::is_std_complex_v.
/// @tparam Scalar the scalar type
template <class Scalar> inline constexpr bool is_complex_v = Arith<std::remove_cv_t<Scalar>>::is_complex;

// std::complex<Real> -> Cx<Real>: launchers re-type operands at the pointer boundary, so no
// Annex-G operator* reaches the kernel.
namespace detail {
template <class Scalar> struct dev_scalar_of {
    using type = Scalar;
};
template <class Real> struct dev_scalar_of<std::complex<Real>> {
    using type = Cx<Real>;
};
}  // namespace detail
/// @brief The device scalar of a host scalar: `std::complex<Real>` becomes `Cx<Real>`; every other
/// Scalar is itself.
/// @ingroup api_device_arith
/// @tparam Scalar the host scalar type
template <class Scalar> using dev_scalar_t = typename detail::dev_scalar_of<std::remove_cv_t<Scalar>>::type;

/// @brief Re(a), read through the reference (no copy of the imaginary part); a itself for a real
/// Scalar.
/// @ingroup api_device_arith
/// @param a the value
/// @return the real part
template <class Scalar> BATCHLAS_DEVICE_INLINE real_t<Scalar> re(const Scalar& a) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    if constexpr (A::is_complex) return A::parts::re(a);
    else return a;
}

/// @brief The arithmetic policies: the Ops template argument a routine's loop takes.
namespace arith {
/// @brief The default: `Arith<T>`, explicit real arithmetic.
/// @ingroup api_device_arith
struct Explicit {};
/// @brief T's own operators (`std::complex` keeps its Annex-G multiply).
/// @ingroup api_device_arith
///
/// For setup code outside inner loops, and for strict ports whose machine code and rounding must
/// not move. A strict port of a kernel that uses `std::complex` operators takes Native: bit
/// identity, Annex-G kept.
struct Native {};
/// @brief `Arith<T>` with mac and mac_conj as plain `inline` calls, the hand kernels' helper form.
/// @ingroup api_device_arith
// LLVM applies `#pragma unroll 4` while the call is still in the body, which leaves the loop open
// to runtime unrolling after inlining. The always_inline form unrolls only by 4 (latrd symv: 47 vs
// 78 FFMA at f32 WG 128, hand = 78). D6: explicit arithmetic everywhere costs occupancy.
struct ExplicitInline {};
/// @brief Explicit fma chains: the arithmetic of gemm, gemv, potri, trsm, getrs and geqrf.
/// @ingroup api_device_arith
///
/// Real mac is one fma and complex mac four; mul is two; div and recip are Smith's with fma
/// denominators. Explicit, ExplicitInline and Fma are for algorithm ports under tolerance tests
/// with an Inf / NaN case.
// Bit-equal to the library's device scalar dev_mul / fma_acc / dev_recip / dev_div.
struct Fma {};
}  // namespace arith
namespace detail {
template <class Ops> inline constexpr bool is_native_v = std::is_same_v<Ops, arith::Native>;
}  // namespace detail

/// @brief The arithmetic of arith::ExplicitInline: `Arith<Scalar>` with mac and mac_conj plain
/// `inline`.
/// @ingroup api_device_lowlevel
///
/// Operands by reference and complex parts read one at a time, as in the hand helper: the loads sit
/// inside the call. Any complex kind goes through its Parts.
/// @tparam Scalar the scalar type
// A by-value std::complex is one 8- or 16-byte load (other unrolling).
template <class Scalar> struct ArithInline : Arith<Scalar> {
    /// @brief acc += a * b.
    static inline void mac(Scalar& acc, const Scalar& a, const Scalar& b) {
        if constexpr (Arith<Scalar>::is_complex) {
            using P = typename Arith<Scalar>::parts;
            using R = typename Arith<Scalar>::real_t;
            const R ar = P::re(a), ai = P::im(a), br = P::re(b), bi = P::im(b);
            const R pr = ar * br - ai * bi;
            const R pi = ar * bi + ai * br;
            acc = P::make(P::re(acc) + pr, P::im(acc) + pi);
        } else {
            acc += a * b;
        }
    }
    /// @brief acc += conj(a) * b.
    static inline void mac_conj(Scalar& acc, const Scalar& a, const Scalar& b) {
        if constexpr (Arith<Scalar>::is_complex) {
            using P = typename Arith<Scalar>::parts;
            using R = typename Arith<Scalar>::real_t;
            const R ar = P::re(a), ai = P::im(a), br = P::re(b), bi = P::im(b);
            const R pr = ar * br + ai * bi;
            const R pi = ar * bi - ai * br;
            acc = P::make(P::re(acc) + pr, P::im(acc) + pi);
        } else {
            acc += a * b;
        }
    }
};

/// @brief The arithmetic of arith::Native: Scalar's own operators.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the scalar type
template <class Scalar> struct NativeArith {
    using real_t = typename Arith<Scalar>::real_t;  ///< The real type of Scalar.
    static constexpr bool is_complex = Arith<Scalar>::is_complex;  ///< True for a complex Scalar.
    /// @brief `std::conj(a)`; the identity on a real Scalar.
    BATCHLAS_DEVICE_INLINE static Scalar cj(Scalar a) {
        if constexpr (is_complex) return std::conj(a);
        else return a;
    }
    /// @brief a * b.
    BATCHLAS_DEVICE_INLINE static Scalar mul(Scalar a, Scalar b) { return a * b; }
    /// @brief acc += a * b.
    BATCHLAS_DEVICE_INLINE static void mac(Scalar& acc, Scalar a, Scalar b) { acc += a * b; }
    /// @brief acc += conj(a) * b.
    BATCHLAS_DEVICE_INLINE static void mac_conj(Scalar& acc, Scalar a, Scalar b) { acc += cj(a) * b; }
    /// @brief conj(a).
    BATCHLAS_DEVICE_INLINE static Scalar conj(Scalar a) { return cj(a); }
    /// @brief a + b.
    BATCHLAS_DEVICE_INLINE static Scalar add(Scalar a, Scalar b) { return a + b; }
    /// @brief a - b.
    BATCHLAS_DEVICE_INLINE static Scalar sub(Scalar a, Scalar b) { return a - b; }
    /// @brief -a.
    BATCHLAS_DEVICE_INLINE static Scalar neg(Scalar a) { return -a; }
    /// @brief 0.
    BATCHLAS_DEVICE_INLINE static Scalar zero() { return Scalar(0); }
    /// @brief 1.
    BATCHLAS_DEVICE_INLINE static Scalar one() { return Scalar(1); }
    /// @brief The real part.
    BATCHLAS_DEVICE_INLINE static real_t re(const Scalar& a) { return Arith<Scalar>::re(a); }
    /// @brief The imaginary part.
    BATCHLAS_DEVICE_INLINE static real_t im(const Scalar& a) { return Arith<Scalar>::im(a); }
    /// @brief The scalar (r, i).
    BATCHLAS_DEVICE_INLINE static Scalar make(real_t r, real_t i) { return Arith<Scalar>::make(r, i); }
    /// @brief c -= a * b.
    BATCHLAS_DEVICE_INLINE static void msub(Scalar& c, const Scalar& a, const Scalar& b) { c -= a * b; }
    /// @brief a / b.
    BATCHLAS_DEVICE_INLINE static Scalar div(const Scalar& a, const Scalar& b) { return a / b; }
    /// @brief 1 / a.
    BATCHLAS_DEVICE_INLINE static Scalar recip(const Scalar& a) { return Scalar(1) / a; }
    /// @brief a / s for a real s.
    BATCHLAS_DEVICE_INLINE static Scalar div_real(const Scalar& a, const real_t& s) { return a / s; }
    /// @brief s * a for a real s.
    BATCHLAS_DEVICE_INLINE static Scalar scale_real(const real_t& s, const Scalar& a) { return s * a; }
    /// @brief True when a == 0.
    BATCHLAS_DEVICE_INLINE static bool is_zero(const Scalar& a) { return a == Scalar(0); }
    /// @brief True when a is finite.
    BATCHLAS_DEVICE_INLINE static bool isfinite(const Scalar& a) { return Arith<Scalar>::isfinite(a); }
};

/// @brief The arithmetic of arith::Fma: explicit fma chains.
/// @ingroup api_device_lowlevel
///
/// Real mac is one fma, complex four; mul is two; div and recip are Smith's with fma denominators.
/// @tparam Scalar the scalar type
template <class Scalar> struct FmaArith : Arith<Scalar> {
    using A = Arith<Scalar>;  ///< `Arith<Scalar>`.
    using R = typename A::real_t;  ///< The real type.
    /// @brief a * b.
    BATCHLAS_DEVICE_INLINE static Scalar mul(const Scalar& a, const Scalar& b) {
        if constexpr (A::is_complex)
            return A::make(sycl::fma(A::re(a), A::re(b), -A::im(a) * A::im(b)), sycl::fma(A::re(a), A::im(b), A::im(a) * A::re(b)));
        else return a * b;
    }
    /// @brief acc += a * b.
    BATCHLAS_DEVICE_INLINE static void mac(Scalar& acc, const Scalar& a, const Scalar& b) {
        if constexpr (A::is_complex) {
            R re = sycl::fma(A::re(a), A::re(b), A::re(acc));
            re = sycl::fma(-A::im(a), A::im(b), re);
            R im = sycl::fma(A::re(a), A::im(b), A::im(acc));
            im = sycl::fma(A::im(a), A::re(b), im);
            acc = A::make(re, im);
        } else {
            acc = sycl::fma(a, b, acc);
        }
    }
    /// @brief acc += conj(a) * b.
    BATCHLAS_DEVICE_INLINE static void mac_conj(Scalar& acc, const Scalar& a, const Scalar& b) {
        if constexpr (A::is_complex) {
            R re = sycl::fma(A::re(a), A::re(b), A::re(acc));
            re = sycl::fma(A::im(a), A::im(b), re);
            R im = sycl::fma(A::re(a), A::im(b), A::im(acc));
            im = sycl::fma(-A::im(a), A::re(b), im);
            acc = A::make(re, im);
        } else {
            acc = sycl::fma(a, b, acc);
        }
    }
    /// @brief c -= a * b.
    BATCHLAS_DEVICE_INLINE static void msub(Scalar& c, const Scalar& a, const Scalar& b) {
        if constexpr (A::is_complex) {
            R re = sycl::fma(-A::re(a), A::re(b), A::re(c));
            re = sycl::fma(A::im(a), A::im(b), re);
            R im = sycl::fma(-A::re(a), A::im(b), A::im(c));
            im = sycl::fma(-A::im(a), A::re(b), im);
            c = A::make(re, im);
        } else {
            c = sycl::fma(-a, b, c);
        }
    }
    /// @brief 1 / d.
    BATCHLAS_DEVICE_INLINE static Scalar recip(const Scalar& d) {
        if constexpr (A::is_complex) {
            const R dr = A::re(d), di = A::im(d);
            if (sycl::fabs(dr) >= sycl::fabs(di)) {
                const R r = di / dr;
                const R den = sycl::fma(di, r, dr);
                return A::make(R(1) / den, -r / den);
            }
            const R r = dr / di;
            const R den = sycl::fma(dr, r, di);
            return A::make(r / den, R(-1) / den);
        } else {
            return Scalar(1) / d;
        }
    }
    /// @brief a / b.
    BATCHLAS_DEVICE_INLINE static Scalar div(const Scalar& a, const Scalar& b) {
        if constexpr (A::is_complex) {
            const R ar = A::re(a), ai = A::im(a), br = A::re(b), bi = A::im(b);
            if (sycl::fabs(br) >= sycl::fabs(bi)) {
                const R r = bi / br;
                const R den = sycl::fma(bi, r, br);
                return A::make(sycl::fma(ai, r, ar) / den, sycl::fma(-ar, r, ai) / den);
            }
            const R r = br / bi;
            const R den = sycl::fma(br, r, bi);
            return A::make(sycl::fma(ar, r, ai) / den, sycl::fma(ai, r, -ar) / den);
        } else {
            return a / b;
        }
    }
};

/// @brief The arithmetic of policy Ops for Scalar: NativeArith, ArithInline, FmaArith or
/// `Arith<Scalar>` (arith::Explicit).
/// @ingroup api_device_arith
/// @tparam Ops an `arith::` policy other than FmaSplit (see ops_arith_t)
/// @tparam Scalar the scalar type
template <class Ops, class Scalar>
using arith_t = std::conditional_t<
    std::is_same_v<Ops, arith::Native>, NativeArith<std::remove_cv_t<Scalar>>,
    std::conditional_t<std::is_same_v<Ops, arith::ExplicitInline>, ArithInline<std::remove_cv_t<Scalar>>,
                       std::conditional_t<std::is_same_v<Ops, arith::Fma>, FmaArith<std::remove_cv_t<Scalar>>, Arith<std::remove_cv_t<Scalar>>>>>;

/// @brief |a|^2 = re^2 + im^2, by value.
/// @ingroup api_device_arith
///
/// Its result is a separate rounding: it cannot contract into the caller's accumulation, so
/// accumulate with sumsq or acc_abs2 (level1.hh) instead.
/// @param a the value
/// @return |a|^2
template <class Scalar> BATCHLAS_DEVICE_INLINE real_t<Scalar> abs2(Scalar a) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    if constexpr (A::is_complex) return A::re(a) * A::re(a) + A::im(a) * A::im(a);
    else return a * a;
}

/// @brief BLAS |re| + |im| (dcabs1): the key of asum, amax and iamax.
/// @ingroup api_device_arith
/// @param a the value
/// @return |re a| + |im a|
template <class Scalar> BATCHLAS_DEVICE_INLINE real_t<Scalar> abs1(Scalar a) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    if constexpr (A::is_complex) return sycl::fabs(A::re(a)) + sycl::fabs(A::im(a));
    else return sycl::fabs(a);
}

/// @brief The conjugate of a; the identity on a real Scalar.
/// @ingroup api_device_arith
///
/// Under arith::Native (the default) it is `std::conj`, plain inline, by reference: the legacy
/// kernels' form.
/// @tparam Ops the arithmetic policy
/// @param a the value
/// @return conj(a)
template <class Ops = arith::Native, class Scalar> BATCHLAS_DEVICE_HELPER constexpr Scalar conj(const Scalar& a) {
    if constexpr (!std::is_same_v<Ops, arith::Native>) return arith_t<Ops, Scalar>::conj(a);
    else if constexpr (Arith<std::remove_cv_t<Scalar>>::is_complex) return std::conj(a);
    else return a;
}

/// @brief Selects a when cond, else b, component by component.
/// @ingroup api_device_arith
///
/// LLVM builds no `select` of a complex aggregate, and a register array read through one stops
/// being promotable.
/// @param cond the condition
/// @param a the value when cond
/// @param b the value otherwise
/// @return a or b
// The components are read before the ternaries: a call inside a `?:` arm leaves a branchier CFG
// (complex<float> machine code moves).
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar select(bool cond, Scalar a, Scalar b) {
    using A = Arith<Scalar>;
    if constexpr (A::is_complex) {
        const auto ar = A::re(a), ai = A::im(a), br = A::re(b), bi = A::im(b);
        return A::make(cond ? ar : br, cond ? ai : bi);
    } else {
        return cond ? a : b;
    }
}

/// @brief s * x for a scalar tag s; Zero never reads x.
/// @ingroup api_device_arith
/// @param s the tag (One, Zero, NegOne or `Scalar<T>`)
/// @param x the value
/// @return s * x
template <class Scalar, class S> BATCHLAS_DEVICE_INLINE Scalar tag_mul(S s, Scalar x) {
    if constexpr (std::is_same_v<S, One>) return x;
    else if constexpr (std::is_same_v<S, Zero>) return Arith<Scalar>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<Scalar>::neg(x);
    else return Arith<Scalar>::mul(static_cast<Scalar>(s.v), x);
}

// arith::FmaSplit is the getrf / potri / getrs / jacobi hand form.
namespace arith {
/// @brief `Arith<T>` with msub(c, a, b) = sub(c, mul(a, b)): two roundings, never one fma.
/// @ingroup api_device_arith
struct FmaSplit {};
}  // namespace arith
/// @brief The arithmetic of arith::FmaSplit: `Arith<Scalar>` with a two-rounding msub.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the scalar type
template <class Scalar> struct FmaSplitArith : Arith<Scalar> {
    /// @brief c := c - a * b with two roundings.
    BATCHLAS_DEVICE_INLINE static void msub(Scalar& c, const Scalar& a,
                                            const Scalar& b) { c = Arith<Scalar>::sub(c, Arith<Scalar>::mul(a, b)); }
};
/// @brief The arithmetic of any `arith::` policy, FmaSplit included.
/// @ingroup api_device_arith
/// @tparam Ops the arithmetic policy
/// @tparam Scalar the scalar type
template <class Ops, class Scalar>
using ops_arith_t = std::conditional_t<std::is_same_v<Ops, arith::FmaSplit>, FmaSplitArith<std::remove_cv_t<Scalar>>, arith_t<Ops, Scalar>>;

/// @brief max(|re|, |im|): the infinity-norm magnitude bound, ABSSQ-free.
/// @ingroup api_device_arith
/// @param a the value
/// @return max(|re a|, |im a|)
template <class Scalar> BATCHLAS_DEVICE_INLINE real_t<Scalar> abs_inf(Scalar a) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    if constexpr (A::is_complex) return sycl::fmax(sycl::fabs(A::re(a)), sycl::fabs(A::im(a)));
    else return sycl::fabs(a);
}
/// @brief (re, 0): the real part as Scalar (a Hermitian diagonal, a phase-free reflector beta).
/// @ingroup api_device_arith
/// @param a the value
/// @return (re a, 0)
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar force_real(Scalar a) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    if constexpr (A::is_complex) return A::make(A::re(a), typename A::real_t(0));
    else return a;
}
/// @brief A real r as Scalar (r, 0).
/// @ingroup api_device_arith
/// @param r the real value
/// @return (r, 0)
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar lift(real_t<Scalar> r) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    if constexpr (A::is_complex) return A::make(r, typename A::real_t(0));
    else return r;
}
/// @brief {re, im} of a value, as by_parts returns them (im = 0 for a real Scalar).
/// @ingroup api_device_arith
/// @tparam Real the real type
template <class Real> struct Parts {
    ///@{
    /// @brief The real and the imaginary part.
    Real re, im;
    ///@}
};
/// @brief {re, im} of a (im = 0 for a real Scalar).
/// @ingroup api_device_arith
/// @param a the value
/// @return the Parts
template <class Scalar> BATCHLAS_DEVICE_INLINE Parts<real_t<Scalar>> by_parts(Scalar a) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    return {A::re(a), A::im(a)};
}
/// @brief conj(a) when C (compile time); the identity on a real Scalar.
/// @ingroup api_device_arith
/// @tparam C true to conjugate
/// @param a the value
/// @return conj(a) or a
template <bool C, class Scalar>
    requires std::is_floating_point_v<real_t<Scalar>>
BATCHLAS_DEVICE_INLINE Scalar conj_if(const Scalar& a) {
    if constexpr (C) return Arith<std::remove_cv_t<Scalar>>::conj(a);
    else return a;
}
/// @brief conj(a) when on (run time, one select); the identity on a real Scalar.
/// @ingroup api_device_arith
/// @param on the run-time switch
/// @param a the value
/// @return conj(a) or a
template <class Scalar>
    requires std::is_floating_point_v<real_t<Scalar>>
BATCHLAS_DEVICE_INLINE Scalar conj_if(bool on, const Scalar& a) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    if constexpr (A::is_complex) return on ? A::conj(a) : a;
    else return a;
}
/// @brief s * x for a REAL s, componentwise: no complex multiply.
/// @ingroup api_device_arith
///
/// x is taken by value: one load of an x in memory, not one per part.
/// @param s the real factor
/// @param x the value
/// @return s * x
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar scale_real(real_t<Scalar> s, Scalar x) { return Arith<std::remove_cv_t<Scalar>>::scale_real(s, x); }
/// @brief x / s for a REAL s, componentwise.
/// @ingroup api_device_arith
/// @param x the value
/// @param s the real divisor
/// @return x / s
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar div_real(const Scalar& x, real_t<Scalar> s) { return Arith<std::remove_cv_t<Scalar>>::div_real(x, s); }
/// @brief a - s * x for a REAL s, componentwise.
/// @ingroup api_device_arith
/// @param a the minuend
/// @param s the real factor
/// @param x the value
/// @return a - s * x
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar sub_scale_real(const Scalar& a, real_t<Scalar> s,
                                                                     const Scalar& x) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    if constexpr (A::is_complex) return A::make(A::re(a) - s * A::re(x), A::im(a) - s * A::im(x));
    else return a - s * x;
}

namespace detail {
// The value a scalar tag stands for (MaybeZero: its value, the flag ignored).
template <class Scalar, class S> BATCHLAS_DEVICE_INLINE Scalar tag_value(const S& s) {
    if constexpr (std::is_same_v<S, One>) return Arith<Scalar>::one();
    else if constexpr (std::is_same_v<S, Zero>) return Arith<Scalar>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<Scalar>::neg(Arith<Scalar>::one());
    else return static_cast<Scalar>(s.v);
}
template <class S> inline constexpr bool is_maybe_zero_v = false;
template <class Scalar> inline constexpr bool is_maybe_zero_v<MaybeZero<Scalar>> = true;
// out += s * x under Ar: Zero adds nothing, One / NegOne add exactly, a value is one mac.
template <class Ar, class Scalar, class S> BATCHLAS_DEVICE_INLINE void mac_term(Scalar& out, const S& s,
                                                                                const Scalar& x) {
    if constexpr (std::is_same_v<S, Zero>) return;
    else if constexpr (std::is_same_v<S, One>) out = Ar::add(out, x);
    else if constexpr (std::is_same_v<S, NegOne>) out = Ar::sub(out, x);
    else Ar::mac(out, static_cast<Scalar>(s.v), x);
}
// s * x under Ar; Zero never reads x.
template <class Ar, class Scalar, class S> BATCHLAS_DEVICE_INLINE Scalar mul_term(const S& s, const Scalar& x) {
    if constexpr (std::is_same_v<S, One>) return x;
    else if constexpr (std::is_same_v<S, Zero>) return Ar::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Ar::neg(x);
    else return Ar::mul(static_cast<Scalar>(s.v), x);
}
// beta == 0 as the epilogue sees it: a Zero tag, a MaybeZero flag, else false (a value is not
// compared: only epi::Linear / LinearNegZero compare beta with 0, as their sources do).
template <class S> BATCHLAS_DEVICE_INLINE bool known_zero(const S& s) {
    if constexpr (std::is_same_v<S, Zero>) return true;
    else if constexpr (is_maybe_zero_v<S>) return s.zero;
    else return false;
}
}  // namespace detail

/// @brief out += s * x under Ops: Zero adds nothing, and MaybeZero adds nothing when its flag is
/// set.
/// @ingroup api_device_arith
/// @tparam Ops the arithmetic policy
/// @param out the accumulator
/// @param s_ the scalar: a tag, `MaybeZero<T>` or a plain value
/// @param x the value
template <class Ops = arith::Explicit, class Scalar, class S> BATCHLAS_DEVICE_INLINE void tag_mac(Scalar& out, S s_,
                                                                                                  const Scalar& x) {
    using Ar = ops_arith_t<Ops, Scalar>;
    const auto s = as_tag(s_);
    if constexpr (detail::is_maybe_zero_v<std::remove_cv_t<decltype(s)>>) {
        if (!s.zero) Ar::mac(out, s.v, x);
    } else {
        detail::mac_term<Ar>(out, s, x);
    }
}

/// @brief The epilogue shapes of store_epi: each is the exact expression of the code it replaces.
namespace epi {
/// @brief add(alpha acc, beta y): the core store, the default.
/// @ingroup api_device_arith
struct Sum {};
/// @brief out = 0; out += alpha acc; out += beta y.
/// @ingroup api_device_arith
// gemv store_fma, sparse store_cols.
struct MacChain {};
/// @brief out = alpha acc; out += beta y; for a real T one expression.
/// @ingroup api_device_arith
// gemm epilogue_store_fused.
struct MulMac {};
/// @brief beta == 0 ? alpha acc : alpha acc + beta y, beta compared at run time.
/// @ingroup api_device_arith
// gemm LinearEpilogue.
struct Linear {};
/// @brief alpha acc + (beta == 0 ? -0 : beta) y; reads y always.
/// @ingroup api_device_arith
// The wide kernels' lin_epi.
struct LinearNegZero {};
}  // namespace epi

/// @brief THE epilogue: y := alpha acc + beta y, in one of five shapes.
/// @ingroup api_device_arith
///
/// alpha and beta are each a tag (One, Zero, NegOne, `Scalar<T>`), a `MaybeZero<T>` (a value with
/// a host-decided zero flag) or a plain value. A Zero beta, or a set MaybeZero flag, never reads y,
/// except in epi::LinearNegZero, whose source reads it always.
/// `store_epi<epi::MacChain, arith::Fma>` is gemv's store_fma. The Frag and Tile spill is
/// operand.hh's store.
/// @tparam Shape epi::Sum (default), MacChain, MulMac, Linear or LinearNegZero
/// @tparam Ops the arithmetic policy
/// @param alpha_ the scale of acc
/// @param acc the accumulated value
/// @param beta_ the scale of y
/// @param y the destination, read unless beta is known zero
template <class Shape = epi::Sum, class Ops = arith::Explicit, class AS, class T, class BS, class Ref>
BATCHLAS_DEVICE_INLINE void store_epi(AS alpha_, T acc, BS beta_, Ref&& y) {
    using Ar = ops_arith_t<Ops, T>;
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    using A = std::remove_cv_t<decltype(alpha)>;
    using B = std::remove_cv_t<decltype(beta)>;
    if constexpr (std::is_same_v<Shape, epi::Sum>) {
        T ax;
        if constexpr (detail::is_maybe_zero_v<A>) ax = detail::mul_term<Ar>(Scalar<T>{alpha.v}, acc);
        else ax = detail::mul_term<Ar>(alpha, acc);
        if constexpr (std::is_same_v<B, Zero>) {
            y = ax;
        } else if constexpr (detail::is_maybe_zero_v<B>) {
            y = beta.zero ? ax : Ar::add(ax, Ar::mul(beta.v, static_cast<T>(y)));
        } else if constexpr (std::is_same_v<B, One>) {
            y = Ar::add(ax, static_cast<T>(y));
        } else {
            y = Ar::add(ax, detail::mul_term<Ar>(beta, static_cast<T>(y)));
        }
    } else if constexpr (std::is_same_v<Shape, epi::MacChain>) {
        T out = Ar::zero();
        if constexpr (detail::is_maybe_zero_v<A>) detail::mac_term<Ar>(out, Scalar<T>{alpha.v}, acc);
        else detail::mac_term<Ar>(out, alpha, acc);
        if constexpr (detail::is_maybe_zero_v<B>) {
            if (!beta.zero) detail::mac_term<Ar>(out, Scalar<T>{beta.v}, static_cast<T>(y));
        } else {
            detail::mac_term<Ar>(out, beta, static_cast<T>(y));
        }
        y = out;
    } else if constexpr (std::is_same_v<Shape, epi::MulMac>) {
        const T a = detail::tag_value<T>(alpha);
        if constexpr (std::is_same_v<B, Zero>) {
            y = Ar::mul(a, acc);
        } else {
            const T b = detail::tag_value<T>(beta);
            const bool bz = detail::known_zero(beta);
            if constexpr (Ar::is_complex) {
                T out = Ar::mul(a, acc);
                if (!bz) Ar::mac(out, b, static_cast<T>(y));
                y = out;
            } else {
                y = bz ? a * acc : a * acc + b * static_cast<T>(y);
            }
        }
    } else if constexpr (std::is_same_v<Shape, epi::Linear>) {
        const T a = detail::tag_value<T>(alpha), b = detail::tag_value<T>(beta);
        if constexpr (Ar::is_complex) {
            y = Ar::is_zero(b) ? Ar::mul(a, acc) : Ar::add(Ar::mul(a, acc), Ar::mul(b, static_cast<T>(y)));
        } else {
            y = b == T(0) ? a * acc : a * acc + b * static_cast<T>(y);
        }
    } else {
        static_assert(std::is_same_v<Shape, epi::LinearNegZero>, "store: Shape is epi::Sum, MacChain, MulMac, Linear or LinearNegZero");
        const T a = detail::tag_value<T>(alpha), b = detail::tag_value<T>(beta);
        const T p = static_cast<T>(y);
        if constexpr (Ar::is_complex) {
            using R = typename Ar::real_t;
            const bool bz = Ar::is_zero(b);
            const R ar = Ar::re(a), ai = Ar::im(a), cr = Ar::re(acc), ci = Ar::im(acc), pr = Ar::re(p), pi = Ar::im(p);
            const R br = bz ? R(-0.0) : Ar::re(b), bi_re = bz ? R(0) : Ar::im(b), bi_im = bz ? R(-0.0) : Ar::im(b);
            y = Ar::make(ar * cr - ai * ci + br * pr - bi_re * pi, ar * ci + ai * cr + br * pi + bi_im * pr);
        } else {
            y = a * acc + (b == T(0) ? T(-0.0) : b) * p;
        }
    }
}

}  // namespace batchlas::device
