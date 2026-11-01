// SPDX-License-Identifier: MIT
#pragma once

// Negligibility, the QL shift and block scaling of the symmetric tridiagonal T.

#include <batchlas/blas/extensions.hh>
#include <src/math-helpers.hh>

namespace batchlas::tridiag {

// e_i is negligible: e^2 <= eps^2 |d_i| |d_{i+1}| + safmin.  steqr_cta_device.hh:220-221
template <typename T>
__attribute__((always_inline)) inline bool negligible(T e, T d0, T d1) {
    const T rhs = detail::eps2<T>() * sycl::fabs(d0) * sycl::fabs(d1) + detail::safmin<T>();
    return (sycl::fabs(e) * sycl::fabs(e) <= rhs);
}

// The eigenvalue of [[d_l, e_l], [e_l, d_l1]] nearest d_l. Lapack: d_l - e_l / (g + sign(g) hypot(g, 1)),
// g = (d_l1 - d_l) / (2 e_l); Wilkinson: the lae2 eigenvalue nearer d_l.  steqr_cta_device.hh:335-343
template <typename T>
__attribute__((always_inline)) inline T wilkinson_shift(SteqrShiftStrategy strategy, T d_l, T e_l, T d_l1) {
    T mu = T(0);
    if (strategy == SteqrShiftStrategy::Wilkinson) {
        mu = detail::wilkinson_shift(d_l1, e_l, d_l);
    } else {
        const T gg = (d_l1 - d_l) / (T(2) * e_l);
        const T rr = sycl::hypot(gg, T(1));
        mu = d_l - e_l / (gg + sycl::copysign(rr, gg));
    }
    return mu;
}

// factor * max|B| in [ssfmin, ssfmax]; back() undoes it with inverse = 1 / factor.
template <typename T> struct BlockScale {
    T factor = T(1), inverse = T(1);

    // steqr_cta_device.hh:599-605
    __attribute__((always_inline)) static BlockScale for_norm(T anorm) {
        T scale = T(1);
        if (anorm > detail::ssfmax<T>()) {
            scale = detail::ssfmax<T>() / anorm;
        } else if (anorm < detail::ssfmin<T>() && anorm != T(0)) {
            scale = detail::ssfmin<T>() / anorm;
        }
        const T inv_scale = T(1) / scale;
        return {scale, inv_scale};
    }
    __attribute__((always_inline)) bool active() const { return factor != T(1); }
    __attribute__((always_inline)) T to_safe(T x) const { return x * factor; }
    __attribute__((always_inline)) T back(T x) const { return x * inverse; }
};

} // namespace batchlas::tridiag
