// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 arithmetic for the POD device scalar sycl_device::Cx (src-only type), so v2 ops
// run on kernels that re-type std::complex at the pointer boundary.

#include <batchlas/blas/device/v2/core.hh>
#include <src/sycl/device_scalar.hh>

namespace batchlas::device::v2 {

template <class R> struct Arith<sycl_device::Cx<R>> {
    using T = sycl_device::Cx<R>;
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
    // The rest mirrors Arith<std::complex<R>> (core.hh); FmaArith<Cx> overrides msub/div/recip
    // with the dev_* fma forms. div_real / scale_real / is_zero / isfinite are bit for bit
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

}  // namespace batchlas::device::v2
