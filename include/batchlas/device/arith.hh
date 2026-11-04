// SPDX-License-Identifier: MIT
#pragma once
// Device LA arithmetic: the Arith trait (one per scalar kind: real, std::complex, any POD {re, im}
// complex such as Cx), the arith:: policies an op's loop takes (Explicit, Native, ExplicitInline,
// Fma) and the scalar helpers built on them (abs1, abs2, conj, select, tag_mul, store).

#include <complex>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>

namespace batchlas::device::inline v2 {

// A plain aggregate complex for device code: layout-compatible with std::complex<R> (a launcher
// may reinterpret_cast at the pointer boundary) and free of Annex-G operators.
template <class R> struct Cx {
    R re;
    R im;
};

namespace detail {
// A POD {re, im} complex: two data members of one floating type, trivially copyable.
template <class C>
concept PodComplex = requires { &C::re; &C::im; } && std::is_same_v<decltype(C::re), decltype(C::im)> &&
                     std::is_floating_point_v<decltype(C::re)> && std::is_trivially_copyable_v<C>;
}  // namespace detail

// ---- arithmetic trait ---------------------------------------------------------------------
// Inner loops multiply through Arith only: for complex it is explicit real arithmetic, never
// the Annex-G operator* (isnan branches + __mulsc3). Any POD {re, im} aggregate (Cx below, the
// src device scalar sycl_device::Cx) gets the explicit complex Arith; specialise it for another
// scalar type.
template <class T> struct Arith {
    using real_t = T;
    static constexpr bool is_complex = false;
    BATCHLAS_DLA_INLINE static T mul(T a, T b) { return a * b; }
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, T a, T b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static T conj(T a) { return a; }
    BATCHLAS_DLA_INLINE static T re(T a) { return a; }
    BATCHLAS_DLA_INLINE static T im(T) { return T(0); }
    BATCHLAS_DLA_INLINE static T make(T r, T) { return r; }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return a + b; }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return a - b; }
    BATCHLAS_DLA_INLINE static T neg(T a) { return -a; }
    BATCHLAS_DLA_INLINE static T zero() { return T(0); }
    BATCHLAS_DLA_INLINE static T one() { return T(1); }
    // c -= a * b as ONE expression (contracts to one FMA; two statements would round twice).
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) { c = c - a * b; }
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) { return a / b; }
    BATCHLAS_DLA_INLINE static T recip(const T& a) { return T(1) / a; }
    BATCHLAS_DLA_INLINE static T div_real(const T& a, const T& s) { return a / s; }
    BATCHLAS_DLA_INLINE static T scale_real(const T& s, const T& a) { return s * a; }
    BATCHLAS_DLA_INLINE static bool is_zero(const T& a) { return a == T(0); }
    BATCHLAS_DLA_INLINE static bool isfinite(const T& a) { return sycl::isfinite(a); }
};

template <class R> struct Arith<std::complex<R>> {
    using T = std::complex<R>;
    using real_t = R;
    static constexpr bool is_complex = true;
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        return T(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
    }
    // The latrd split-loop form: bit-identical to the legacy kernel's mac().
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        const R pr = a.real() * b.real() - a.imag() * b.imag();
        const R pi = a.real() * b.imag() + a.imag() * b.real();
        acc = T(acc.real() + pr, acc.imag() + pi);
    }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, T a, T b) {
        const R pr = a.real() * b.real() + a.imag() * b.imag();
        const R pi = a.real() * b.imag() - a.imag() * b.real();
        acc = T(acc.real() + pr, acc.imag() + pi);
    }
    BATCHLAS_DLA_INLINE static T conj(T a) { return T(a.real(), -a.imag()); }
    BATCHLAS_DLA_INLINE static R re(T a) { return a.real(); }
    BATCHLAS_DLA_INLINE static R im(T a) { return a.imag(); }
    BATCHLAS_DLA_INLINE static T make(R r, R i) { return T(r, i); }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return T(a.real() + b.real(), a.imag() + b.imag()); }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return T(a.real() - b.real(), a.imag() - b.imag()); }
    BATCHLAS_DLA_INLINE static T neg(T a) { return T(-a.real(), -a.imag()); }
    BATCHLAS_DLA_INLINE static T zero() { return T(R(0), R(0)); }
    BATCHLAS_DLA_INLINE static T one() { return T(R(1), R(0)); }
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) {
        const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
        c = T(c.real() - (ar * br - ai * bi), c.imag() - (ar * bi + ai * br));
    }
    // Smith's algorithm (no Annex-G, no (c - di) / (c^2 + d^2) overflow past ~1e19 / ~1e154).
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) {
        const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
        if (sycl::fabs(br) >= sycl::fabs(bi)) {
            const R r = bi / br, den = br + bi * r;
            return T((ar + ai * r) / den, (ai - ar * r) / den);
        }
        const R r = br / bi, den = bi + br * r;
        return T((ar * r + ai) / den, (ai * r - ar) / den);
    }
    BATCHLAS_DLA_INLINE static T recip(const T& d) {
        const R dr = d.real(), di = d.imag();
        if (sycl::fabs(dr) >= sycl::fabs(di)) {
            const R r = di / dr, den = dr + di * r;
            return T(R(1) / den, -r / den);
        }
        const R r = dr / di, den = di + dr * r;
        return T(r / den, R(-1) / den);
    }
    BATCHLAS_DLA_INLINE static T div_real(const T& a, const R& s) { return T(a.real() / s, a.imag() / s); }
    BATCHLAS_DLA_INLINE static T scale_real(const R& s, const T& a) { return T(s * a.real(), s * a.imag()); }
    BATCHLAS_DLA_INLINE static bool is_zero(const T& a) { return a.real() == R(0) && a.imag() == R(0); }
    BATCHLAS_DLA_INLINE static bool isfinite(const T& a) { return sycl::isfinite(a.real()) && sycl::isfinite(a.imag()); }
};

template <class C> requires detail::PodComplex<C> struct Arith<C> {
    using T = C;
    using R = decltype(C::re);
    using real_t = R;
    static constexpr bool is_complex = true;
    BATCHLAS_DLA_INLINE static T mul(T a, T b) { return T{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        acc.re += a.re * b.re - a.im * b.im;
        acc.im += a.re * b.im + a.im * b.re;
    }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, T a, T b) {
        acc.re += a.re * b.re + a.im * b.im;
        acc.im += a.re * b.im - a.im * b.re;
    }
    BATCHLAS_DLA_INLINE static T conj(T a) { return T{a.re, -a.im}; }
    BATCHLAS_DLA_INLINE static R re(T a) { return a.re; }
    BATCHLAS_DLA_INLINE static R im(T a) { return a.im; }
    BATCHLAS_DLA_INLINE static T make(R r, R i) { return T{r, i}; }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return T{a.re + b.re, a.im + b.im}; }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return T{a.re - b.re, a.im - b.im}; }
    BATCHLAS_DLA_INLINE static T neg(T a) { return T{-a.re, -a.im}; }
    BATCHLAS_DLA_INLINE static T zero() { return T{R(0), R(0)}; }
    BATCHLAS_DLA_INLINE static T one() { return T{R(1), R(0)}; }
    // The rest mirrors Arith<std::complex<R>>; FmaArith<C> overrides msub/div/recip with the fma
    // forms. On sycl_device::Cx, div_real / scale_real / is_zero / isfinite are bit for bit
    // dev_div_real / dev_mul_real / dev_is_zero / dev_isfinite (src/sycl/device_scalar.hh).
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) {
        c = T{c.re - (a.re * b.re - a.im * b.im), c.im - (a.re * b.im + a.im * b.re)};
    }
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) {
        if (sycl::fabs(b.re) >= sycl::fabs(b.im)) {
            const R r = b.im / b.re, den = b.re + b.im * r;
            return T{(a.re + a.im * r) / den, (a.im - a.re * r) / den};
        }
        const R r = b.re / b.im, den = b.im + b.re * r;
        return T{(a.re * r + a.im) / den, (a.im * r - a.re) / den};
    }
    BATCHLAS_DLA_INLINE static T recip(const T& d) {
        if (sycl::fabs(d.re) >= sycl::fabs(d.im)) {
            const R r = d.im / d.re, den = d.re + d.im * r;
            return T{R(1) / den, -r / den};
        }
        const R r = d.re / d.im, den = d.im + d.re * r;
        return T{r / den, R(-1) / den};
    }
    BATCHLAS_DLA_INLINE static T div_real(const T& a, const R& s) { return T{a.re / s, a.im / s}; }
    BATCHLAS_DLA_INLINE static T scale_real(const R& s, const T& a) { return T{s * a.re, s * a.im}; }
    BATCHLAS_DLA_INLINE static bool is_zero(const T& a) { return a.re == R(0) && a.im == R(0); }
    BATCHLAS_DLA_INLINE static bool isfinite(const T& a) { return sycl::isfinite(a.re) && sycl::isfinite(a.im); }
};

template <class T> using real_t = typename Arith<std::remove_cv_t<T>>::real_t;

// The arithmetic an op's loop uses. arith::Explicit (default) is Arith<T>. arith::Native is
// T's own operators (std::complex keeps its Annex-G multiply): setup code outside inner loops,
// and strict ports whose SASS and rounding must not move (D6: explicit arithmetic everywhere
// costs occupancy). arith::ExplicitInline is Arith<T> with mac/mac_conj as plain `inline`
// calls, the hand kernels' helper form: LLVM applies `#pragma unroll 4` while the call is still
// in the body, which leaves the loop open to runtime unrolling after inlining. The always_inline
// form unrolls only by 4 (latrd symv: 47 vs 78 FFMA at f32 WG 128, hand = 78).
// arith::Fma is explicit fma chains, bit-equal to sycl_device::dev_mul / fma_acc / dev_recip /
// dev_div (src/sycl/device_scalar.hh): the arithmetic of gemm, gemv, potri, trsm, getrs, geqrf.
// Complex rule (D6): a strict port of a kernel that uses std::complex operators takes Native
// (bit identity, Annex-G kept); explicit real arithmetic (Explicit, ExplicitInline, Fma) is for
// algo ports, under tolerance tests with an Inf/NaN case.
namespace arith {
struct Explicit {};
struct Native {};
struct ExplicitInline {};
struct Fma {};
}  // namespace arith

// Operands by reference and complex parts read one at a time, as in the hand helper: the loads
// sit inside the call, and a by-value std::complex is one 8/16-byte load (other unrolling).
template <class T> struct ArithInline : Arith<T> {
    static inline void mac(T& acc, const T& a, const T& b) {
        if constexpr (Arith<T>::is_complex) {
            using R = typename Arith<T>::real_t;
            const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
            const R pr = ar * br - ai * bi;
            const R pi = ar * bi + ai * br;
            acc = T(acc.real() + pr, acc.imag() + pi);
        } else {
            acc += a * b;
        }
    }
    static inline void mac_conj(T& acc, const T& a, const T& b) {
        if constexpr (Arith<T>::is_complex) {
            using R = typename Arith<T>::real_t;
            const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
            const R pr = ar * br + ai * bi;
            const R pi = ar * bi - ai * br;
            acc = T(acc.real() + pr, acc.imag() + pi);
        } else {
            acc += a * b;
        }
    }
};

template <class T> struct NativeArith {
    using real_t = typename Arith<T>::real_t;
    static constexpr bool is_complex = Arith<T>::is_complex;
    BATCHLAS_DLA_INLINE static T cj(T a) {
        if constexpr (is_complex) return std::conj(a);
        else return a;
    }
    BATCHLAS_DLA_INLINE static T mul(T a, T b) { return a * b; }
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, T a, T b) { acc += cj(a) * b; }
    BATCHLAS_DLA_INLINE static T conj(T a) { return cj(a); }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return a + b; }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return a - b; }
    BATCHLAS_DLA_INLINE static T neg(T a) { return -a; }
    BATCHLAS_DLA_INLINE static T zero() { return T(0); }
    BATCHLAS_DLA_INLINE static T one() { return T(1); }
    BATCHLAS_DLA_INLINE static real_t re(const T& a) { return Arith<T>::re(a); }
    BATCHLAS_DLA_INLINE static real_t im(const T& a) { return Arith<T>::im(a); }
    BATCHLAS_DLA_INLINE static T make(real_t r, real_t i) { return Arith<T>::make(r, i); }
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) { c -= a * b; }
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) { return a / b; }
    BATCHLAS_DLA_INLINE static T recip(const T& a) { return T(1) / a; }
    BATCHLAS_DLA_INLINE static T div_real(const T& a, const real_t& s) { return a / s; }
    BATCHLAS_DLA_INLINE static T scale_real(const real_t& s, const T& a) { return s * a; }
    BATCHLAS_DLA_INLINE static bool is_zero(const T& a) { return a == T(0); }
    BATCHLAS_DLA_INLINE static bool isfinite(const T& a) { return Arith<T>::isfinite(a); }
};

// Explicit fma chains (src/sycl/device_scalar.hh, bit for bit): real mac is one fma, complex
// four; mul is dev_mul's two; div / recip are Smith's with fma denominators.
template <class T> struct FmaArith : Arith<T> {
    using A = Arith<T>;
    using R = typename A::real_t;
    BATCHLAS_DLA_INLINE static T mul(const T& a, const T& b) {
        if constexpr (A::is_complex)
            return A::make(sycl::fma(A::re(a), A::re(b), -A::im(a) * A::im(b)), sycl::fma(A::re(a), A::im(b), A::im(a) * A::re(b)));
        else return a * b;
    }
    BATCHLAS_DLA_INLINE static void mac(T& acc, const T& a, const T& b) {
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
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, const T& a, const T& b) {
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
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) {
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
    BATCHLAS_DLA_INLINE static T recip(const T& d) {
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
            return T(1) / d;
        }
    }
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) {
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

template <class Ops, class T>
using arith_t = std::conditional_t<
    std::is_same_v<Ops, arith::Native>, NativeArith<std::remove_cv_t<T>>,
    std::conditional_t<std::is_same_v<Ops, arith::ExplicitInline>, ArithInline<std::remove_cv_t<T>>,
                       std::conditional_t<std::is_same_v<Ops, arith::Fma>, FmaArith<std::remove_cv_t<T>>, Arith<std::remove_cv_t<T>>>>>;

// |a|^2 = re^2 + im^2, by value. Its result is a separate rounding: it cannot contract into the
// caller's accumulation (syevx), so accumulate with sumsq / acc_abs2 (level1.hh) instead.
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs2(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return A::re(a) * A::re(a) + A::im(a) * A::im(a);
    else return a * a;
}

// BLAS |re| + |im| (dcabs1): the key of asum, amax and iamax.
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs1(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return sycl::fabs(A::re(a)) + sycl::fabs(A::im(a));
    else return sycl::fabs(a);
}

// Conjugate, the identity on a real T. arith::Native is std::conj, plain inline by reference: the
// legacy kernels' form (always_inline / by value moves their complex SASS).
template <class Ops = arith::Native, class T> BATCHLAS_DLA_HELPER constexpr T conj(const T& a) {
    if constexpr (!std::is_same_v<Ops, arith::Native>) return arith_t<Ops, T>::conj(a);
    else if constexpr (Arith<std::remove_cv_t<T>>::is_complex) return std::conj(a);
    else return a;
}

// cond ? a : b component by component: LLVM builds no `select` of a complex aggregate, and
// a register array read through one stops being promotable. The components are read before
// the ternaries: a call inside a `?:` arm leaves a branchier CFG (complex<float> SASS moves).
template <class T> BATCHLAS_DLA_INLINE T select(bool cond, T a, T b) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        const auto ar = A::re(a), ai = A::im(a), br = A::re(b), bi = A::im(b);
        return A::make(cond ? ar : br, cond ? ai : bi);
    } else {
        return cond ? a : b;
    }
}

// s * x for a tag s; Zero never reads x.
template <class T, class S> BATCHLAS_DLA_INLINE T tag_mul(S s, T x) {
    if constexpr (std::is_same_v<S, One>) return x;
    else if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(x);
    else return Arith<T>::mul(static_cast<T>(s.v), x);
}


// ---- v2.2 scalar helpers (design A.2 arith.hh) ---------------------------------------------
// arith::FmaSplit: Arith<T> with msub(c, a, b) = sub(c, mul(a, b)): two roundings, never one
// fma (the getrf / potri / getrs / jacobi hand form).
namespace arith {
struct FmaSplit {};
}  // namespace arith
template <class T> struct FmaSplitArith : Arith<T> {
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) { c = Arith<T>::sub(c, Arith<T>::mul(a, b)); }
};
template <class Ops, class T>
using ops_arith_t = std::conditional_t<std::is_same_v<Ops, arith::FmaSplit>, FmaSplitArith<std::remove_cv_t<T>>, arith_t<Ops, T>>;

// max(|re|, |im|): LAPACK's ABSSQ-free magnitude bound (izmax1's key is abs1; this is the inf norm).
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs_inf(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return sycl::fmax(sycl::fabs(A::re(a)), sycl::fabs(A::im(a)));
    else return sycl::fabs(a);
}
// (re, 0): the real part as T (a Hermitian diagonal, a phase-free reflector beta).
template <class T> BATCHLAS_DLA_INLINE T force_real(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return A::make(A::re(a), typename A::real_t(0));
    else return a;
}
// A real r as T (r, 0).
template <class T> BATCHLAS_DLA_INLINE T lift(real_t<T> r) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return A::make(r, typename A::real_t(0));
    else return r;
}
// {re, im} of a (im = 0 for real T).
template <class R> struct Parts {
    R re, im;
};
template <class T> BATCHLAS_DLA_INLINE Parts<real_t<T>> by_parts(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    return {A::re(a), A::im(a)};
}
// conj(a) when C (compile time) or when on (run time, one select); the identity on real T.
template <bool C, class T>
    requires std::is_floating_point_v<real_t<T>>
BATCHLAS_DLA_INLINE T conj_if(const T& a) {
    if constexpr (C) return Arith<std::remove_cv_t<T>>::conj(a);
    else return a;
}
template <class T>
    requires std::is_floating_point_v<real_t<T>>
BATCHLAS_DLA_INLINE T conj_if(bool on, const T& a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return on ? A::conj(a) : a;
    else return a;
}
// s * x and x / s for a REAL s (componentwise; no complex multiply), and a - s * x.
template <class T> BATCHLAS_DLA_INLINE T scale_real(real_t<T> s, const T& x) { return Arith<std::remove_cv_t<T>>::scale_real(s, x); }
template <class T> BATCHLAS_DLA_INLINE T div_real(const T& x, real_t<T> s) { return Arith<std::remove_cv_t<T>>::div_real(x, s); }
template <class T> BATCHLAS_DLA_INLINE T sub_scale_real(const T& a, real_t<T> s, const T& x) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return A::make(A::re(a) - s * A::re(x), A::im(a) - s * A::im(x));
    else return a - s * x;
}

namespace detail {
// The value a scalar tag stands for (MaybeZero: its value, the flag ignored).
template <class T, class S> BATCHLAS_DLA_INLINE T tag_value(const S& s) {
    if constexpr (std::is_same_v<S, One>) return Arith<T>::one();
    else if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(Arith<T>::one());
    else return static_cast<T>(s.v);
}
template <class S> inline constexpr bool is_maybe_zero_v = false;
template <class T> inline constexpr bool is_maybe_zero_v<MaybeZero<T>> = true;
// out += s * x under Ar: Zero adds nothing, One / NegOne add exactly, a value is one mac.
template <class Ar, class T, class S> BATCHLAS_DLA_INLINE void mac_term(T& out, const S& s, const T& x) {
    if constexpr (std::is_same_v<S, Zero>) return;
    else if constexpr (std::is_same_v<S, One>) out = Ar::add(out, x);
    else if constexpr (std::is_same_v<S, NegOne>) out = Ar::sub(out, x);
    else Ar::mac(out, static_cast<T>(s.v), x);
}
// s * x under Ar; Zero never reads x.
template <class Ar, class T, class S> BATCHLAS_DLA_INLINE T mul_term(const S& s, const T& x) {
    if constexpr (std::is_same_v<S, One>) return x;
    else if constexpr (std::is_same_v<S, Zero>) return Ar::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Ar::neg(x);
    else return Ar::mul(static_cast<T>(s.v), x);
}
// beta == 0 as the epilogue sees it: a Zero tag, a MaybeZero flag, else false (a value is not
// compared: only epi::Linear / LinearNegZero compare beta with 0, as their sources do).
template <class S> BATCHLAS_DLA_INLINE bool known_zero(const S& s) {
    if constexpr (std::is_same_v<S, Zero>) return true;
    else if constexpr (is_maybe_zero_v<S>) return s.zero;
    else return false;
}
}  // namespace detail

// out += s * x under Ops (Zero adds nothing; MaybeZero adds nothing when its flag is set).
template <class Ops = arith::Explicit, class T, class S> BATCHLAS_DLA_INLINE void tag_mac(T& out, S s_, const T& x) {
    using Ar = ops_arith_t<Ops, T>;
    const auto s = as_tag(s_);
    if constexpr (detail::is_maybe_zero_v<std::remove_cv_t<decltype(s)>>) {
        if (!s.zero) Ar::mac(out, s.v, x);
    } else {
        detail::mac_term<Ar>(out, s, x);
    }
}

// ---- THE epilogue: y := alpha acc + beta y in one of five shapes ---------------------------
// alpha / beta: a tag (One, Zero, NegOne, Scalar<T>), MaybeZero<T> (a value with a host-decided
// zero flag) or a plain value. Zero beta (or a set MaybeZero flag) never reads y, except in
// LinearNegZero, whose source reads it always. Each shape is the exact expression of the code
// it replaces (design A.2):
//   epi::Sum           add(alpha acc, beta y)                 (the v2.1 core store; default)
//   epi::MacChain      out = 0; out += alpha acc; out += beta y  (gemv store_fma, sparse store_cols)
//   epi::MulMac        out = alpha acc; out += beta y; real: one expression (gemm epilogue_store_fused)
//   epi::Linear        beta == 0 ? alpha acc : alpha acc + beta y, beta compared at run time
//                      (gemm LinearEpilogue)
//   epi::LinearNegZero alpha acc + (beta == 0 ? -0 : beta) y (the wide kernels' lin_epi)
// store<epi::MacChain, arith::Fma> is gemv's store_fma (a core store_fma would meet gemv's by ADL
// until M4 deletes it). store(alpha, acc, beta, y) with no Shape is the v2.1 core store.
namespace epi {
struct Sum {};
struct MacChain {};
struct MulMac {};
struct Linear {};
struct LinearNegZero {};
}  // namespace epi

template <class Shape = epi::Sum, class Ops = arith::Explicit, class AS, class T, class BS, class Ref>
BATCHLAS_DLA_INLINE void store(AS alpha_, T acc, BS beta_, Ref&& y) {
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
