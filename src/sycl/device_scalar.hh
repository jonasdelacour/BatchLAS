// SPDX-License-Identifier: MIT
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
template <typename Real>
struct Cx {
    Real re;
    Real im;
};

template <typename Scalar>
struct DevMap {
    using type = Scalar;
    using real = Scalar;
    static constexpr bool is_complex = false;
};

template <typename Real>
struct DevMap<std::complex<Real>> {
    using type = Cx<Real>;
    using real = Real;
    static constexpr bool is_complex = true;
};

// DevMap<T>::is_complex asked of the DEVICE type D. Not sizeof: Cx<float> and
// double are both 8 bytes.
template <typename D> struct IsDevComplex           : std::false_type {};
template <typename Real> struct IsDevComplex<Cx<Real>>    : std::true_type  {};
template <typename D> inline constexpr bool dev_is_complex_v = IsDevComplex<D>::value;

static_assert(sizeof(Cx<float>) == sizeof(std::complex<float>), "layout");
static_assert(sizeof(Cx<double>) == sizeof(std::complex<double>), "layout");
static_assert(alignof(Cx<float>) == alignof(std::complex<float>), "layout");
static_assert(alignof(Cx<double>) == alignof(std::complex<double>), "layout");

// --- zero test -------------------------------------------------------------

template <typename Real>
inline bool dev_is_zero(Real x) {
    return x == Real(0);
}
template <typename Real>
inline bool dev_is_zero(Cx<Real> x) {
    return x.re == Real(0) && x.im == Real(0);
}

// --- multiply-accumulate, written out --------------------------------------
// Real is one FMA; complex is four, with no branches and no libcall.

inline void fma_acc(float& acc, float a, float b) { acc = sycl::fma(a, b, acc); }
inline void fma_acc(double& acc, double a, double b) { acc = sycl::fma(a, b, acc); }

template <typename Real>
inline void fma_acc(Cx<Real>& acc, Cx<Real> a, Cx<Real> b) {
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

inline float dev_abs(float x) { return sycl::fabs(x); }
inline double dev_abs(double x) { return sycl::fabs(x); }
template <typename Real>
inline Real dev_abs(Cx<Real> x) { return sycl::hypot(x.re, x.im); }

inline float dev_conj(float x) { return x; }
inline double dev_conj(double x) { return x; }

template <typename Real>
inline Cx<Real> dev_conj(Cx<Real> x) {
    return Cx<Real>{x.re, -x.im};
}

// --- plain multiply and subtract -------------------------------------------

inline float dev_mul(float a, float b) { return a * b; }
inline double dev_mul(double a, double b) { return a * b; }

template <typename Real>
inline Cx<Real> dev_mul(Cx<Real> a, Cx<Real> b) {
    // Written out for the same reason fma_acc is: keep Annex-G out of it.
    return Cx<Real>{sycl::fma(a.re, b.re, -a.im * b.im),
                 sycl::fma(a.re, b.im, a.im * b.re)};
}

inline float dev_sub(float a, float b) { return a - b; }
inline double dev_sub(double a, double b) { return a - b; }

template <typename Real>
inline Cx<Real> dev_sub(Cx<Real> a, Cx<Real> b) {
    return Cx<Real>{a.re - b.re, a.im - b.im};
}

// --- finiteness ------------------------------------------------------------
// Both components: they go non-finite independently.

inline bool dev_isfinite(float x) { return sycl::isfinite(x); }
inline bool dev_isfinite(double x) { return sycl::isfinite(x); }

template <typename Real>
inline bool dev_isfinite(Cx<Real> x) {
    return sycl::isfinite(x.re) && sycl::isfinite(x.im);
}

// --- division and reciprocal -----------------------------------------------
// Smith's algorithm, NOT (c-di)/(c^2+d^2), which overflows past ~1e19 (float) /
// ~1e154 (double) and returns 0. evidence: docs/perf/gemm.md#gemm-the-pod-device-scalar

template <typename Real>
inline Cx<Real> dev_recip(Cx<Real> d) {
    if (sycl::fabs(d.re) >= sycl::fabs(d.im)) {
        const Real r = d.im / d.re;
        const Real den = sycl::fma(d.im, r, d.re);
        return Cx<Real>{Real(1) / den, -r / den};
    }
    const Real r = d.re / d.im;
    const Real den = sycl::fma(d.re, r, d.im);
    return Cx<Real>{r / den, Real(-1) / den};
}

inline float dev_recip(float d) { return 1.0f / d; }
inline double dev_recip(double d) { return 1.0 / d; }

template <typename Real>
inline Cx<Real> dev_div(Cx<Real> a, Cx<Real> b) {
    if (sycl::fabs(b.re) >= sycl::fabs(b.im)) {
        const Real r = b.im / b.re;
        const Real den = sycl::fma(b.im, r, b.re);
        return Cx<Real>{sycl::fma(a.im, r, a.re) / den, sycl::fma(-a.re, r, a.im) / den};
    }
    const Real r = b.re / b.im;
    const Real den = sycl::fma(b.re, r, b.im);
    return Cx<Real>{sycl::fma(a.re, r, a.im) / den, sycl::fma(a.im, r, -a.re) / den};
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
template <typename Real>
inline Real dev_real(Cx<Real> x) { return x.re; }

template <typename D, typename Real>
inline D dev_from_real(Real x) {
    if constexpr (std::is_same_v<D, Real>) {
        return x;
    } else {
        return D{x, Real(0)};
    }
}

inline float  dev_mul_real(float a, float s)   { return a * s; }
inline double dev_mul_real(double a, double s) { return a * s; }
template <typename Real>
inline Cx<Real> dev_mul_real(Cx<Real> a, Real s) { return Cx<Real>{a.re * s, a.im * s}; }

inline float  dev_div_real(float a, float d)   { return a / d; }
inline double dev_div_real(double a, double d) { return a / d; }
template <typename Real>
inline Cx<Real> dev_div_real(Cx<Real> a, Real d) { return Cx<Real>{a.re / d, a.im / d}; }

}  // namespace batchlas::sycl_device
