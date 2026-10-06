#pragma once

// The POD device scalar shared by the SYCL kernels. std::complex must never
// reach device code (Annex-G operator* = isnan branch + __mulsc3): launchers
// re-type operands AND scalars to these aggregates at the pointer boundary.
// evidence: docs/perf/gemm.md#gemm-the-pod-device-scalar

#include <sycl/sycl.hpp>

#include <complex>
#include <type_traits>

namespace batchlas::sycl_device {

// A plain aggregate complex. Layout-compatible with std::complex, which is what
// lets the launcher reinterpret_cast at the boundary.
template <typename R>
struct Cx {
    R re;
    R im;
};

template <typename T>
struct DevMap {
    using type = T;
    using real = T;
    static constexpr bool is_complex = false;
};

template <typename R>
struct DevMap<std::complex<R>> {
    using type = Cx<R>;
    using real = R;
    static constexpr bool is_complex = true;
};

// DevMap<T>::is_complex asked of the DEVICE type D. Not sizeof: Cx<float> and
// double are both 8 bytes.
template <typename D> struct IsDevComplex           : std::false_type {};
template <typename R> struct IsDevComplex<Cx<R>>    : std::true_type  {};
template <typename D> inline constexpr bool dev_is_complex_v = IsDevComplex<D>::value;

static_assert(sizeof(Cx<float>) == sizeof(std::complex<float>), "layout");
static_assert(sizeof(Cx<double>) == sizeof(std::complex<double>), "layout");
static_assert(alignof(Cx<float>) == alignof(std::complex<float>), "layout");
static_assert(alignof(Cx<double>) == alignof(std::complex<double>), "layout");

// --- zero test -------------------------------------------------------------

template <typename R>
inline bool dev_is_zero(R x) {
    return x == R(0);
}
template <typename R>
inline bool dev_is_zero(Cx<R> x) {
    return x.re == R(0) && x.im == R(0);
}

// --- multiply-accumulate, written out --------------------------------------
// Real is one FMA; complex is four, with no branches and no libcall.

inline void fma_acc(float& acc, float a, float b) { acc = sycl::fma(a, b, acc); }
inline void fma_acc(double& acc, double a, double b) { acc = sycl::fma(a, b, acc); }

template <typename R>
inline void fma_acc(Cx<R>& acc, Cx<R> a, Cx<R> b) {
    acc.re = sycl::fma(a.re, b.re, acc.re);
    acc.re = sycl::fma(-a.im, b.im, acc.re);
    acc.im = sycl::fma(a.re, b.im, acc.im);
    acc.im = sycl::fma(a.im, b.re, acc.im);
}

// ===========================================================================
// The arithmetic a triangular solve needs and a GEMM does not.
// ===========================================================================

// --- construction and conjugation ------------------------------------------

template <typename D>
inline D dev_one() {
    if constexpr (std::is_same_v<D, float> || std::is_same_v<D, double>) {
        return D(1);
    } else {
        return D{typename std::remove_reference_t<decltype(D{}.re)>(1),
                 typename std::remove_reference_t<decltype(D{}.re)>(0)};
    }
}

inline float dev_conj(float x) { return x; }
inline double dev_conj(double x) { return x; }

template <typename R>
inline Cx<R> dev_conj(Cx<R> x) {
    return Cx<R>{x.re, -x.im};
}

// --- plain multiply and subtract -------------------------------------------

inline float dev_mul(float a, float b) { return a * b; }
inline double dev_mul(double a, double b) { return a * b; }

template <typename R>
inline Cx<R> dev_mul(Cx<R> a, Cx<R> b) {
    // Written out for the same reason fma_acc is: keep Annex-G out of it.
    return Cx<R>{sycl::fma(a.re, b.re, -a.im * b.im),
                 sycl::fma(a.re, b.im, a.im * b.re)};
}

inline float dev_sub(float a, float b) { return a - b; }
inline double dev_sub(double a, double b) { return a - b; }

template <typename R>
inline Cx<R> dev_sub(Cx<R> a, Cx<R> b) {
    return Cx<R>{a.re - b.re, a.im - b.im};
}

// --- finiteness ------------------------------------------------------------
// Both components: they go non-finite independently.

inline bool dev_isfinite(float x) { return sycl::isfinite(x); }
inline bool dev_isfinite(double x) { return sycl::isfinite(x); }

template <typename R>
inline bool dev_isfinite(Cx<R> x) {
    return sycl::isfinite(x.re) && sycl::isfinite(x.im);
}

// --- division and reciprocal -----------------------------------------------
// Smith's algorithm, NOT (c-di)/(c^2+d^2), which overflows past ~1e19 (float) /
// ~1e154 (double) and returns 0. evidence: docs/perf/gemm.md#gemm-the-pod-device-scalar

template <typename R>
inline Cx<R> dev_recip(Cx<R> d) {
    if (sycl::fabs(d.re) >= sycl::fabs(d.im)) {
        const R r = d.im / d.re;
        const R den = sycl::fma(d.im, r, d.re);
        return Cx<R>{R(1) / den, -r / den};
    }
    const R r = d.re / d.im;
    const R den = sycl::fma(d.re, r, d.im);
    return Cx<R>{r / den, R(-1) / den};
}

inline float dev_recip(float d) { return 1.0f / d; }
inline double dev_recip(double d) { return 1.0 / d; }

template <typename R>
inline Cx<R> dev_div(Cx<R> a, Cx<R> b) {
    if (sycl::fabs(b.re) >= sycl::fabs(b.im)) {
        const R r = b.im / b.re;
        const R den = sycl::fma(b.im, r, b.re);
        return Cx<R>{sycl::fma(a.im, r, a.re) / den, sycl::fma(-a.re, r, a.im) / den};
    }
    const R r = b.re / b.im;
    const R den = sycl::fma(b.re, r, b.im);
    return Cx<R>{sycl::fma(a.re, r, a.im) / den, sycl::fma(a.im, r, -a.re) / den};
}

inline float dev_div(float a, float b) { return a / b; }
inline double dev_div(double a, double b) { return a / b; }

// --- real component, real construction, real scaling, real division --------
// For a REAL divisor/scale (a Cholesky diagonal) without the complex paths.
// Deliberate asymmetry: callers use dev_div_real where reference ?trsm divides
// and dev_mul_real by a reciprocal where ?potf2 scales; unifying them moves one
// off its LAPACK rounding. evidence: docs/perf/gemm.md#gemm-the-pod-device-scalar

inline float  dev_real(float x)  { return x; }
inline double dev_real(double x) { return x; }
template <typename R>
inline R dev_real(Cx<R> x) { return x.re; }

template <typename D, typename R>
inline D dev_from_real(R x) {
    if constexpr (std::is_same_v<D, R>) {
        return x;
    } else {
        return D{x, R(0)};
    }
}

inline float  dev_mul_real(float a, float s)   { return a * s; }
inline double dev_mul_real(double a, double s) { return a * s; }
template <typename R>
inline Cx<R> dev_mul_real(Cx<R> a, R s) { return Cx<R>{a.re * s, a.im * s}; }

inline float  dev_div_real(float a, float d)   { return a / d; }
inline double dev_div_real(double a, double d) { return a / d; }
template <typename R>
inline Cx<R> dev_div_real(Cx<R> a, R d) { return Cx<R>{a.re / d, a.im / d}; }

}  // namespace batchlas::sycl_device
