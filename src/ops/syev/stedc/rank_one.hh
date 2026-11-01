// SPDX-License-Identifier: MIT
#pragma once

// The Cuppen tear T = diag(T1, T2) + rho u u^T and the deflation tests of the merge.

#include <algorithm>
#include <cmath>
#include <limits>

namespace batchlas::tridiag {

// Cutting at e_m = beta: d_m -= |beta|, d_{m+1} -= |beta|, rho = 2|beta|, z_2 carries sign(beta).
template <typename T> struct Tear {
    T beta;
    __attribute__((always_inline)) T rho() const { return std::abs(2 * beta); }        // stedc.cc:366
    __attribute__((always_inline)) T sign() const { return (beta >= T(0)) ? T(1) : T(-1); }  // stedc.cc:180
};

// Absolute: tol = 8 eps max(|D|_inf, |z|_inf).  stedc.cc:239
template <typename T> struct DeflationTolerance {
    T tol;
    __attribute__((always_inline)) static DeflationTolerance from(T max_abs_d, T max_abs_z) {
        return {T(8.0) * std::numeric_limits<T>::epsilon() * std::max(max_abs_d, max_abs_z)};
    }
};

// |d_hi - d_lo| <= tol: one Givens on the columns of Q zeroes one weight.  stedc.cc:242
template <typename T>
__attribute__((always_inline)) inline bool coincident_poles(T d_lo, T d_hi, DeflationTolerance<T> t) {
    return std::abs(d_hi - d_lo) <= t.tol;
}

// |beta z_n| <= tol drops the pole; NaN drops too.  stedc.cc:284
template <typename T>
__attribute__((always_inline)) inline bool negligible_weight(T beta, T zn, DeflationTolerance<T> t) {
    return !(std::abs(beta * zn) > t.tol);
}

} // namespace batchlas::tridiag
