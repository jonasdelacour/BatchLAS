// SPDX-License-Identifier: MIT
#pragma once

// ZLARFG: H^H [alpha; x] = [beta; 0] with beta REAL, H = I - tau v v^H, v(0) = 1. gebrd needs the
// real beta so that complex input reduces to a REAL bidiagonal the real bdsdc/bdsqr can solve;
// detail::larfg's complex branch keeps beta's phase (beta = -sgn(alpha) |.|) and would leave d, e
// complex. Real T is detail::larfg unchanged.

#include <src/math-helpers.hh>

#include <sycl/sycl.hpp>

namespace batchlas::gebrd_detail {

template <typename T>
inline detail::larfg_result<T> larfg_real_beta(const T& alpha, detail::base_float_t<T> xnorm, int len) {
    if constexpr (!detail::is_complex<T>::value) {
        return detail::larfg(alpha, xnorm, len);
    } else {
        using R = detail::base_float_t<T>;
        detail::larfg_result<T> r{T(0), alpha, T(0)};
        // Unlike the real case, len == 1 still reflects: a complex alpha must still become real.
        if (len < 1 || (xnorm == R(0) && alpha.imag() == R(0))) return r;
        const R norm = sycl::hypot(sycl::hypot(alpha.real(), alpha.imag()), xnorm);
        if (norm == R(0)) return r;
        const R beta = sycl::signbit(alpha.real()) ? norm : -norm;
        r.beta = T(beta);
        r.tau = T((beta - alpha.real()) / beta, -alpha.imag() / beta);
        r.scale = T(1) / (alpha - r.beta);
        return r;
    }
}

template <typename T>
inline T conj(const T& x) {
    if constexpr (detail::is_complex<T>::value) return T(x.real(), -x.imag());
    else return x;
}

}  // namespace batchlas::gebrd_detail
