// SPDX-License-Identifier: MIT
#pragma once

// Löwner weights |zhat_j| = sqrt|prod_i delta_ji / prod_{i!=j} (d_i - d_j)|, kept in range.

#include <sycl/sycl.hpp>

#include <cstdint>

namespace batchlas::tridiag {

// value = mant * 2^exp with |mant| in [0.5, 1) after every factor. frexp rescales by exact
// powers of two, so an in-range product is bit-identical to the naive one.
template <typename T> struct ScaledProduct {
    T mant = T(1);
    int exp = 0;

    // stedc_merge_kernels.hh:89-96
    __attribute__((always_inline)) ScaledProduct times(T factor) const {
        ScaledProduct p = *this;
        p.mant *= factor;
        if (p.mant != T(0) && sycl::isfinite(p.mant)) {
            int e = 0;
            p.mant = sycl::frexp(p.mant, &e);
            p.exp += e;
        }
        return p;
    }

    // sqrt|value|, with neither argument nor result out of range.  stedc_merge_kernels.hh:117-120
    __attribute__((always_inline)) T sqrt_value() const {
        const int32_t half = exp >> 1;
        const int32_t rem = exp - 2 * half;
        T mag = sycl::sqrt(sycl::fabs(mant) * (rem ? T(2) : T(1)));
        mag = sycl::ldexp(mag, half);
        return mag;
    }
};

} // namespace batchlas::tridiag
