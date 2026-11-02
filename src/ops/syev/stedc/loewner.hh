// SPDX-License-Identifier: MIT
#pragma once

// Löwner weights |zhat_j| = sqrt|prod_i delta_ji / prod_{i!=j} (d_i - d_j)|, kept in range.

#include <batchlas/blas/matrix.hh>
#include <src/math-helpers.hh>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

namespace batchlas::tridiag {

// value = mant * 2^exp with |mant| in [0.5, 1) after every factor. frexp rescales by exact
// powers of two, so an in-range product is bit-identical to the naive one.
template <typename T> struct ScaledProduct {
    T mant = T(1);
    int exp = 0;

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

    // sqrt|value|, with neither argument nor result out of range.
    __attribute__((always_inline)) T sqrt_value() const {
        const int32_t half = exp >> 1;
        const int32_t rem = exp - 2 * half;
        T mag = sycl::sqrt(sycl::fabs(mant) * (rem ? T(2) : T(1)));
        mag = sycl::ldexp(mag, half);
        return mag;
    }
};

// |zhat_i|^2 = |prod_j delta_ij / prod_{j!=i} (d_i - d_j)|, delta_ii alone at j == i (dlaed3.f:206-219,
// without the uniform 1/rho). Each factor pairs a numerator with a denominator; interlacing
// fixes only its sign and side of 1 (< 1 for j < i, > 1 and unbounded for j > i), so
// ScaledProduct renormalises after every factor. The partial products are lane-strided over
// the reduction group. Every group calls
// this with the same trip count (`active` masks the work): the reductions are sub-group shuffles.
// evidence: docs/perf/stedc.md#stedc-partition-parallel-rescale-and-normalize
template <typename T, typename Reduce, typename QBatch>
__attribute__((always_inline)) inline ScaledProduct<T> loewner_weight(const Reduce& red, bool active, int32_t i,
                                                                      int32_t K, const QBatch& Q,
                                                                      const VectorView<T>& d) {
    const int32_t lane = red.lane_id();
    const int32_t width = red.width();
    const T Di = active ? d(i) : T(0);
    ScaledProduct<T> part;
    for (int32_t j = lane; active && j < K; j += width) {
        const T q_elem = Q(i, j);
        T ratio;
        if (j == i) {
            ratio = q_elem;
        } else {
            const T denom = Di - d(j);
            ratio = q_elem / denom;
        }
        part = part.times(ratio);
    }

    // Each lane's |mant| is in [0.5, 1), so one product over a wide group bottoms out at
    // 0.5^width (2^-256 at width 256 is zero in float). Reduce 32 lanes per round and
    // renormalize between rounds: every partial stays >= 2^-33.
    ScaledProduct<T> all{T(1), red.template reduce_sum<int32_t>(part.exp)};
    for (int32_t r0 = 0; r0 < width; r0 += 32) {
        const bool mine = lane >= r0 && lane < r0 + 32;
        all = all.times(red.reduce_product(mine ? part.mant : T(1)));
    }
    return all;
}

// Blue's scaled 2-norm of Q(0:K, col), reduced over the group. Mirrors detail::nrm2
// (src/math-helpers.hh) with the three accumulators reduced as scalars (componentwise equal).
template <typename T, typename Reduce, typename QBatch>
inline T nrm2_column(const Reduce& red, QBatch& Q, int32_t K, int32_t col) {
    using R = detail::base_float_t<T>;

    const int radix = std::numeric_limits<R>::radix;
    const int p = std::numeric_limits<R>::digits;
    const int emin = std::numeric_limits<R>::min_exponent;
    const int emax = std::numeric_limits<R>::max_exponent;

    auto floor_div = [](int a, int b) {
        int q = a / b;
        if (((a ^ b) < 0) && (a % b)) --q;
        return q;
    };
    auto ceil_div = [](int a, int b) {
        int q = a / b;
        if (((a ^ b) >= 0) && (a % b)) ++q;
        return q;
    };

    const R tsml = sycl::pown(static_cast<R>(radix), ceil_div(emin - 1, 2));
    const R tbig = sycl::pown(static_cast<R>(radix), floor_div(emax - p + 1, 2));
    const R ssml = sycl::pown(static_cast<R>(radix), -floor_div(emin - p, 2));
    const R sbig = sycl::pown(static_cast<R>(radix), -ceil_div(emax + p - 1, 2));

    R abig = R(0), amed = R(0), asml = R(0);
    for (int32_t i = red.lane_id(); i < K; i += red.width()) {
        const R ax = sycl::fabs(static_cast<R>(detail::abs(Q(i, col))));
        if (ax > tbig) {
            const R t = ax * sbig;
            abig += t * t;
        } else if (ax < tsml) {
            const R t = ax * ssml;
            asml += t * t;
        } else {
            amed += ax * ax;
        }
    }

    abig = red.reduce_sum(abig);
    amed = red.reduce_sum(amed);
    asml = red.reduce_sum(asml);

    const R zero = R(0);
    const R one = R(1);
    const R maxN = std::numeric_limits<R>::max();

    R scl = one;
    R sumsq = zero;

    if (abig > zero) {
        if ((amed > zero) || (amed > maxN) || (amed != amed)) {
            abig += (amed * sbig) * sbig;
        }
        scl = one / sbig;
        sumsq = abig;
    } else if (asml > zero) {
        if ((amed > zero) || (amed > maxN) || (amed != amed)) {
            const R am = sycl::sqrt(amed);
            const R as = sycl::sqrt(asml) / ssml;
            const R ymin = sycl::fmin(am, as);
            const R ymax = sycl::fmax(am, as);
            scl = one;
            sumsq = ymax * ymax * (one + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = one / ssml;
            sumsq = asml;
        }
    } else {
        scl = one;
        sumsq = amed;
    }

    return scl * sycl::sqrt(sumsq);
}

// Eigenvector j of D + rho z z^T: u = zhat ./ delta_j, then u / ||u||_2, in place over the
// column of Q that holds delta_j (dlaed3.f:223-232). zhat is v(:, bid). Inactive groups still
// take the norm (of an empty range) so its reduction stays uniform.
template <typename T, typename Reduce, typename QBatch, typename VView>
__attribute__((always_inline)) inline void cauchy_column(const Reduce& red, bool active, int32_t K, QBatch& Q,
                                                         int32_t j, const VView& v, int32_t bid) {
    const int32_t lane = red.lane_id();
    const int32_t width = red.width();
    for (int32_t i = lane; active && i < K; i += width) {
        Q(i, j) = v(i, bid) / Q(i, j);
    }
    // No sync: the norm walks the column with the same lane stride, reading only what this lane wrote.
    const T nrm2 = nrm2_column<T>(red, Q, active ? K : 0, active ? j : 0);
    for (int32_t i = lane; active && i < K; i += width) {
        Q(i, j) /= nrm2;
    }
}

} // namespace batchlas::tridiag
