// SPDX-License-Identifier: MIT
#pragma once

// The secular equation of the rank-one merge, in the origin-shifted form of Li / dlaed4:
//   f(tau) = 1/rho + sum_j z_j^2 / delta_j,  delta_j = (d_j - origin) - tau,  lambda = origin + tau.

#include <batchlas/blas/matrix.hh>

#include <cstdint>
#include <limits>

namespace batchlas::tridiag {

// lambda = d[origin] + tau, pole = d[origin]; `converged` is the backward-error stop, not max_iter.
// pole spares callers a reload of d[origin], which costs registers.
template <typename T> struct SecularRoot { int32_t origin; T pole; T tau; bool converged; };

// Poles left out of the split sums (today's `skip`): none, the pole k, or k and k+1.
enum class Exclude : int8_t { None = -1, Pole = 0, PoleAndNext = 1 };

// w = 1/rho + psi + phi over the poles not excluded (f itself only for Exclude::None;
// split_at_origin adds pole o back): psi sums the included poles at or below k, phi those above;
// dw = dpsi + dphi, and err sums each lane's running sums: an ERRETM-like bound that depends on
// the width, not dlaed4's (WP6d).
template <typename T> struct SecularSplit { T w, dw, psi, dpsi, phi, dphi, err; };

// lo <= tau* <= hi. f increases on the interval: f(tau) <= 0 raises lo, f(tau) > 0 lowers hi.
template <typename T> struct Bracket {
    T lo, hi;

    __attribute__((always_inline)) Bracket tighten(T tau, T w) const {
        Bracket b = *this;
        if (w <= T(0)) b.lo = sycl::fmax(b.lo, tau);
        if (w > T(0)) b.hi = sycl::fmin(b.hi, tau);
        return b;
    }

    // eta, or half the way to the bound on the root's side when tau + eta leaves the bracket.
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
// value: callers never read back the derivatives MiddleWay adds to.
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

// f split at the origin pole o with o's own term z_o^2/delta_o added to w and dw (not to psi,
// phi); err is Li's ERRETM form applied to split's width-dependent err (WP6d), origin_ratio_sq = (z_o/delta_o)^2 the term MiddleWay adds back.
template <typename T> struct OriginSplit { SecularSplit<T> f; T origin_ratio_sq; };

// Views of the K surviving poles d and weights z (shared memory; never copied). Reduce is
// PartitionAdapter or WorkgroupAdapter: lane_id, width, reduce_sum, broadcast.
template <typename T, typename Reduce> struct SecularEquation {
    const Reduce& red;
    const VectorView<T>& d;
    const VectorView<T>& z;
    int32_t K; T inv_rho;

    // f and f' at origin + tau, split at pole k.
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

    // f, f' at origin + tau around the origin pole o = the root's nearer pole.
    __attribute__((always_inline)) OriginSplit<T> split_at_origin(int32_t o, T origin, T tau) const {
        OriginSplit<T> at{split(o, Exclude::Pole, origin, tau), T(0)};
        const T delta_o = d(o) - origin - tau;
        const T ratio = z(o) / delta_o;
        at.origin_ratio_sq = ratio * ratio;
        at.f.dw += at.origin_ratio_sq;
        const T pole_term = z(o) * ratio;
        at.f.w += pole_term;
        at.f.err = at.f.err + T(8) * (at.f.phi - at.f.psi) + T(2) * inv_rho + T(3) * sycl::fabs(pole_term)
                   + sycl::fabs(tau) * at.f.dw;
        return at;
    }

    // Li's stop |f| <= eps * err, broadcast from lane 0 so every lane leaves the loop together
    // (a divergent exit deadlocks the butterfly reductions).
    __attribute__((always_inline)) bool converged(const SecularSplit<T>& s) const {
        return red.broadcast(sycl::fabs(s.w) <= std::numeric_limits<T>::epsilon() * s.err);
    }
};

// FixedWeight for the second interior step: f, signed toward the origin, still exceeds a tenth
// of |f_prev|.
// differs from dlaed4 SWTCH, see WP6c
template <typename T>
__attribute__((always_inline)) inline bool fixed_weight_after_first_step(bool origin_lower, T f, T f_prev) {
    return (origin_lower ? -f : f) > (sycl::fabs(f_prev) / T(10));
}

__attribute__((always_inline)) inline RationalModel rational_model(bool fixed_weight) {
    return fixed_weight ? RationalModel::FixedWeight : RationalModel::MiddleWay;
}

// lambda_k in (d_k, d_k+1) for k < K-1: dlaed4's interior case with the two-pole model only (base
// behaviour). It never takes the SWTCH3 three-pole dlaed6 step (dlaed4.f:509-518, 597, 763), and
// K = 2 comes here, not to dlaed5 (dlaed4.f:136).
template <typename T, typename Reduce>
__attribute__((always_inline)) inline SecularRoot<T> solve_interior_root(const SecularEquation<T, Reduce>& eq,
                                                                         int32_t k, int32_t max_iter) {
    const T pole_k = eq.d(k);
    const T pole_k1 = eq.d(k + 1);
    const T gap = pole_k1 - pole_k;

    // g = f without poles k, k+1 at the midpoint; the sign of f there puts the root in the
    // half nearer the origin pole.
    const T midpoint = (pole_k + pole_k1) / T(2);
    const T g = eq.split(k, Exclude::PoleAndNext, T(0), midpoint).w;
    const T zk2 = eq.z(k) * eq.z(k);
    const T zk12 = eq.z(k + 1) * eq.z(k + 1);
    const T f_mid = g + T(2) * (zk12 - zk2) / gap;
    const bool origin_lower = (f_mid > T(0));

    // tau0: the root of g + z_k^2/delta_k + z_k+1^2/delta_k+1 = 0, by the stable quadratic formula.
    int32_t o;
    T origin, tau;
    Bracket<T> br;
    if (origin_lower) {
        origin = pole_k;
        o = k;
        br = {T(0), gap / T(2)};
        const T a = g * gap + zk2 + zk12;
        const T b = zk2 * gap;
        const T disc = sycl::sqrt(sycl::fabs(a * a - T(4) * b * g));
        tau = (a > T(0)) ? T(2) * b / (a + disc) : (a - disc) / (T(2) * g);
    } else {
        origin = pole_k1;
        o = k + 1;
        br = {-gap / T(2), T(0)};
        const T a = g * gap - zk2 - zk12;
        const T b = zk12 * gap;
        const T disc = sycl::sqrt(sycl::fabs(a * a + T(4) * b * g));
        tau = (a < T(0)) ? T(2) * b / (a - disc) : -(a + disc) / (T(2) * g);
    }

    // f, f' at origin + t; one step tau <- tau + eta of the two-pole model, eta kept inside the
    // bracket, returning f before the step.
    auto f_at = [&](T t) { return eq.split_at_origin(o, origin, t); };
    OriginSplit<T> at = f_at(tau);
    auto advance = [&](RationalModel model) __attribute__((always_inline)) {
        br = br.tighten(tau, at.f.w);
        const T delta_k = eq.d(k) - origin - tau;
        const T delta_k1 = eq.d(k + 1) - origin - tau;
        T eta = rational_step(model, at.f, delta_k, delta_k1, zk2, zk12, pole_k, pole_k1, origin_lower,
                              at.origin_ratio_sq);
        eta = br.step_inside(tau, eta, at.f.w);
        const T f_prev = at.f.w;
        tau += eta;
        at = f_at(tau);
        return f_prev;
    };

    if (eq.converged(at.f)) {
        return {o, origin, tau, true};
    }
    // The first step is FixedWeight; then fixed_weight_after_first_step picks the model.
    T f_prev = advance(RationalModel::FixedWeight);
    bool fixed_weight = fixed_weight_after_first_step(origin_lower, at.f.w, f_prev);

    bool converged = false;
    for (int32_t iter = 1; iter < max_iter; ++iter) {
        if (eq.converged(at.f)) {
            converged = true;
            break;
        }
        f_prev = advance(rational_model(fixed_weight));
        // f kept its sign and shrank by less than ten: switch models (dlaed4.f:823-824). Inline on
        // purpose: as a helper or a named bool it moves registers.
        if (at.f.w * f_prev > T(0) && sycl::fabs(at.f.w) > sycl::fabs(f_prev) / T(10)) {
            fixed_weight = !fixed_weight;
        }
    }

    return {o, origin, tau, converged};
}

// The two-pole step of the last root (poles K-2 and K-1, f' split as psi' and phi'), Newton when
// it goes uphill. Not rational_step: no fixed weight, and the other root of the quadratic.
template <typename T>
__attribute__((always_inline)) inline T last_root_step(const SecularSplit<T>& s, T delta_prev, T delta_last) {
    const T qc = s.w - delta_prev * s.dpsi - delta_last * s.dphi;
    const T qa = (delta_prev + delta_last) * s.w - delta_prev * delta_last * (s.dpsi + s.dphi);
    const T qb = delta_prev * delta_last * s.w;

    const T disc = sycl::sqrt(sycl::fabs(qa * qa - T(4) * qb * qc));
    T step = (qa >= T(0)) ? (qa + disc) / (T(2) * qc) : T(2) * qb / (qa - disc);

    if (s.w * step > T(0)) {
        step = -s.w / (s.dpsi + s.dphi);
    }
    return step;
}

// lambda_K-1 in (d_K-1, d_K-1 + rho) (dlaed4's I == N case), origin d_K-1. Takes rho itself:
// 1/inv_rho is not bitwise rho.
template <typename T, typename Reduce>
__attribute__((always_inline)) inline SecularRoot<T> solve_last_root(const SecularEquation<T, Reduce>& eq, T rho,
                                                                     int32_t max_iter) {
    // One pole: 1/rho + z0^2/(d0 - lambda) = 0 gives lambda = d0 + rho z0^2 exactly.
    // evidence: docs/perf/stedc.md#stedc-the-dd--1-secular-solve-fault
    if (eq.K <= 1) {
        const T z0 = eq.z(0);
        return {0, eq.d(0), rho * z0 * z0, true};
    }

    const int32_t last = eq.K - 1;
    const int32_t prev = eq.K - 2;
    const T d_last = eq.d(last);
    const T d_prev = eq.d(prev);
    const T origin = d_last;

    // g = f without the last two poles at origin + rho/2; f there picks the half of (0, rho)
    // holding tau, and the two-pole model gives tau0.
    const T mid_tau = rho / T(2);
    const T g = eq.split(prev, Exclude::PoleAndNext, origin, mid_tau).w;
    const T zk2 = eq.z(prev) * eq.z(prev);
    const T zk12 = eq.z(last) * eq.z(last);
    const T f_init = g + zk2 / (d_prev - origin - mid_tau) - T(2) * zk12 * eq.inv_rho;

    T tau;
    Bracket<T> br;
    const T gap = d_last - d_prev;

    if (f_init > T(0)) {
        br = {T(0), rho / T(2)};
        const T a = -g * gap + zk2 + zk12;
        const T b = zk12 * gap;
        const T disc = sycl::sqrt(sycl::fabs(a * a + T(4) * b * g));
        tau = (a < T(0)) ? T(2) * b / (disc - a) : (a + disc) / (T(2) * g);
    } else {
        br = {rho / T(2), rho};
        const T bound_check = zk2 / (gap + rho) + zk12 / rho;
        if (g <= bound_check) {
            tau = rho;
        } else {
            const T a = -g * gap + zk2 + zk12;
            const T b = zk12 * gap;
            const T disc = sycl::sqrt(sycl::fabs(a * a + T(4) * b * g));
            tau = (a < T(0)) ? T(2) * b / (disc - a) : (a + disc) / (T(2) * g);
        }
    }

    // f, f' at origin + tau over all poles; err is dlaed4's I == N ERRETM plus |phi| (split's running sum also covers the last pole, and runs per lane; as in base).
    auto split_all = [&](T t) {
        SecularSplit<T> s = eq.split(prev, Exclude::None, origin, t);
        s.err = s.err + sycl::fabs(t) * (s.dphi + s.dpsi) - T(8) * (s.phi + s.psi) - s.phi + eq.inv_rho;
        return s;
    };

    SecularSplit<T> s = split_all(tau);
    // Not broadcast, unlike every later exit test (as in base).
    if (sycl::fabs(s.w) <= std::numeric_limits<T>::epsilon() * s.err) {
        return {last, origin, tau, true};
    }

    // First step: |qc|, and qc == 0 jumps to the top of the bracket.
    br = br.tighten(tau, s.w);
    {
        const T delta_prev = d_prev - origin - tau;
        const T delta_last = d_last - origin - tau;
        const T qc = sycl::fabs(s.w - delta_prev * s.dpsi - delta_last * s.dphi);
        const T qa = (delta_prev + delta_last) * s.w - delta_prev * delta_last * (s.dpsi + s.dphi);
        const T qb = delta_prev * delta_last * s.w;
        T step;
        if (qc == T(0)) {
            step = br.hi - tau;
        } else {
            const T disc = sycl::sqrt(sycl::fabs(qa * qa - T(4) * qb * qc));
            step = (qa >= T(0)) ? (qa + disc) / (T(2) * qc) : T(2) * qb / (qa - disc);
        }
        if (s.w * step > T(0)) {
            step = -s.w / (s.dpsi + s.dphi);
        }
        step = br.step_inside(tau, step, s.w);
        tau += step;
    }
    s = split_all(tau);

    bool converged = false;
    for (int32_t iter = 1; iter < max_iter; ++iter) {
        if (eq.converged(s)) {
            converged = true;
            break;
        }
        br = br.tighten(tau, s.w);
        const T eta = last_root_step(s, d_prev - origin - tau, d_last - origin - tau);
        tau += br.step_inside(tau, eta, s.w);
        s = split_all(tau);
    }

    return {last, origin, tau, converged};
}

} // namespace batchlas::tridiag
