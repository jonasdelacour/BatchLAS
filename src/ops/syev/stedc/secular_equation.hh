// SPDX-License-Identifier: MIT
#pragma once

// The secular equation of the rank-one merge, in the origin-shifted form of Li / dlaed4:
//   f(tau) = 1/rho + sum_j z_j^2 / delta_j,  delta_j = (d_j - origin) - tau,  lambda = origin + tau.

#include <batchlas/blas/matrix.hh>

#include <cstdint>

namespace batchlas::tridiag {

// lambda = d[origin] + tau; `converged` is the backward-error stop, not max_iter.
template <typename T> struct SecularRoot { int32_t origin; T tau; bool converged; };

// Poles left out of the split sums (today's `skip`): none, the pole k, or k and k+1.
enum class Exclude : int8_t { None = -1, Pole = 0, PoleAndNext = 1 };

// w = f = 1/rho + psi + phi split at the pole: psi sums the poles below, phi those above;
// dw = f', and err is the running-sum bound of the evaluation error.
template <typename T> struct SecularSplit { T w, dw, psi, dpsi, phi, dphi, err; };

// lo <= tau* <= hi. f increases on the interval: f(tau) <= 0 raises lo, f(tau) > 0 lowers hi.
template <typename T> struct Bracket {
    T lo, hi;

    // stedc_merge_cta.cc:147-148
    __attribute__((always_inline)) Bracket tighten(T tau, T w) const {
        Bracket b = *this;
        if (w <= T(0)) b.lo = sycl::fmax(b.lo, tau);
        if (w > T(0)) b.hi = sycl::fmin(b.hi, tau);
        return b;
    }

    // eta, or half the way to the bound on the root's side when tau + eta leaves the bracket.
    // stedc_merge_cta.cc:157-160
    __attribute__((always_inline)) T step_inside(T tau, T eta, T w) const {
        T step = eta;
        if (tau + step > hi || tau + step < lo) {
            step = (w < T(0)) ? (hi - tau) / T(2) : (lo - tau) / T(2);
        }
        return step;
    }
};

// How the two-pole model of f fixes its constant: FixedWeight keeps the exact weight of the
// origin pole (dlaed4's fixed weight), MiddleWay splits f' between psi' and phi'.
enum class RationalModel : uint8_t { FixedWeight, MiddleWay };

// The step eta of the two-pole model through f, f' at tau: qc eta^2 - qa eta + qb = 0 with
// qa = (dk + dk1) w - dk dk1 w', qb = dk dk1 w; Newton when eta goes uphill. The split is by
// value: callers never read back the derivatives MiddleWay adds to.  stedc_merge_cta.cc:225-277
template <typename T>
__attribute__((always_inline)) inline T rational_step(RationalModel model, SecularSplit<T> eval, T dk, T dk1, T zk2, T zk12,
                                                      T pk, T pk1, bool origin_lower, T ratio_sq) {
    const bool use_fixed = (model == RationalModel::FixedWeight);
    T qc;
    if (use_fixed) {
        if (origin_lower) {
            qc = eval.w - dk1 * eval.dw - (pk - pk1) * zk2 / (dk * dk);
        } else {
            qc = eval.w - dk * eval.dw - (pk1 - pk) * zk12 / (dk1 * dk1);
        }
    } else {
        if (origin_lower) {
            eval.dpsi += ratio_sq;
        } else {
            eval.dphi += ratio_sq;
        }
        qc = eval.w - dk * eval.dpsi - dk1 * eval.dphi;
    }

    T qa = (dk + dk1) * eval.w - dk * dk1 * eval.dw;
    T qb = dk * dk1 * eval.w;
    T step;

    if (qc == T(0)) {
        if (qa == T(0)) {
            if (use_fixed) {
                if (origin_lower) {
                    qa = zk2 + dk1 * dk1 * (eval.dpsi + eval.dphi);
                } else {
                    qa = zk12 + dk * dk * (eval.dpsi + eval.dphi);
                }
            } else {
                qa = dk * dk * eval.dpsi + dk1 * dk1 * eval.dphi;
            }
        }
        step = qb / qa;
    } else {
        const T disc = sycl::sqrt(sycl::fabs(qa * qa - T(4) * qb * qc));
        step = (qa <= T(0)) ? (qa - disc) / (T(2) * qc) : T(2) * qb / (qa + disc);
    }

    if (eval.w * step >= T(0)) {
        step = -eval.w / eval.dw;
    }

    return step;
}

// Views of the K surviving poles d and weights z (shared memory; never copied). Reduce is
// PartitionAdapter or WorkgroupAdapter: lane_id, width, reduce_sum.
template <typename T, typename Reduce> struct SecularEquation {
    const Reduce& red;
    VectorView<T> d, z;
    int32_t K; T inv_rho;

    // f and f' at origin + tau, split at pole k.  stedc_merge_cta.cc:174-212
    __attribute__((always_inline)) SecularSplit<T> split(int32_t k, Exclude exclude, T origin, T tau) const {
        const int32_t lane = red.lane_id();
        const int32_t width = red.width();
        const int32_t skip = static_cast<int32_t>(exclude);

        const int32_t lower_end = k + (skip < 0 ? 1 : 0);
        const int32_t upper_start = k + 1 + (skip < 0 ? 0 : skip);

        T low_s = T(0), low_d = T(0), low_err = T(0);
        T up_s = T(0), up_d = T(0), up_err = T(0);

        for (int32_t i = lane; i < lower_end; i += width) {
            const T shifted = d(i) - origin - tau;
            const T w = z(i);
            const T ratio = w / shifted;
            const T wr = w * ratio;
            low_s += wr;
            low_d += ratio * ratio;
            low_err += low_s;
        }
        for (int32_t i = upper_start + lane; i < K; i += width) {
            const T shifted = d(i) - origin - tau;
            const T w = z(i);
            const T ratio = w / shifted;
            const T wr = w * ratio;
            up_s += wr;
            up_d += ratio * ratio;
            up_err += up_s;
        }

        low_s = red.template reduce_sum<T>(low_s);
        low_d = red.template reduce_sum<T>(low_d);
        low_err = red.template reduce_sum<T>(low_err);
        up_s = red.template reduce_sum<T>(up_s);
        up_d = red.template reduce_sum<T>(up_d);
        up_err = red.template reduce_sum<T>(up_err);

        return {inv_rho + low_s + up_s, low_d + up_d, low_s, low_d, up_s, up_d,
                sycl::fabs(low_err) + sycl::fabs(up_err)};
    }
};

} // namespace batchlas::tridiag
