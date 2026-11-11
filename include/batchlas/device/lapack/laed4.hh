// SPDX-License-Identifier: MIT
#pragma once
// xLAED4: one root of a divide-and-conquer merge's secular equation
//     f(x) = 1 / rho + sum_i z_i^2 / (d_i - x) = 0,   d ascending, rho > 0,
// collective over an executor (the ROCm stedc root iteration with LAPACK's error bound). Root k
// < dd - 1 lies in (d_k, d_{k+1}), root dd - 1 in (d_{dd-1}, d_{dd-1} + rho). Every evaluation
// sums over the executor (shuffles on a partition, work-group reductions on a group).
//   laed4                    the public op: root k as origin + tau (origin a pole, so the
//                            caller forms d_i - x as (d_i - origin) - tau without cancellation)
//   laed4_detail::solve_inner / solve_last   the two root forms (the host merges' bodies)
//   laed4_detail::uniform    the leader's value in every item (a loop exit on a reduced value)
// Moved from src/ops/syev/stedc/stedc_merge_cta.cc unchanged: the host's fused merges and
// device::stedc run this code. evidence: docs/perf/stedc.md#stedc-the-dd--1-secular-solve-fault

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device {

/// Root of xLAED4: x = origin + tau; converged = the iteration met its error bound (false: it
/// ran out of max_iter, the merge reports info > 0).
template <class T> struct laed4_root {
    T origin;
    T tau;
    bool converged;
};

namespace laed4_detail {

template <class Ex> inline constexpr bool is_group_v = is_work_group_v<Ex>;
template <int SG> inline constexpr bool is_group_v<RtGroup<Scope::Group, SG>> = true;

// The leader's v in every item (a loop exit on a reduced value, else a partition may diverge and
// deadlock); work-group: the legacy broadcast of a bool (core's group bcast moves 4/8-byte words).
template <class Ex, class T> BATCHLAS_DLA_INLINE T uniform(const Ex& ex, T v) {
    if constexpr (is_thread_v<Ex>) return v;
    else if constexpr (is_group_v<Ex>) return batchlas::portable::group_broadcast(ex.it.get_group(), v, 0);
    else return ex.uniform(v);
}

template <typename T, typename Ex> struct Eval {
    T secular_value;
    T secular_derivative;
    T lower_sum;
    T lower_derivative;
    T upper_sum;
    T upper_derivative;
    T error_estimate;
};

template <typename T> BATCHLAS_DLA_HELPER void update_bounds(T secular_value, T tau, T& lower_bound, T& upper_bound) {
    if (secular_value <= T(0)) lower_bound = sycl::fmax(lower_bound, tau);
    if (secular_value > T(0)) upper_bound = sycl::fmin(upper_bound, tau);
}

template <typename T> BATCHLAS_DLA_HELPER T bounded_step(T step, T tau, T lower_bound, T upper_bound, T secular_value) {
    if (tau + step > upper_bound || tau + step < lower_bound) {
        step = (secular_value < T(0)) ? (upper_bound - tau) / T(2) : (lower_bound - tau) / T(2);
    }
    return step;
}

// The secular function at x = origin + tau without the terms i in [lower_end, upper_start):
// per item the sums of z_i^2 / (d_i - x), of (z_i / (d_i - x))^2 and of the running sums (the
// laed4 error term) below and above that window, then six reductions over the executor.
template <typename T, typename Ex, typename DV>
BATCHLAS_DLA_HELPER Eval<T, Ex> evaluate(const Ex& ex, const DV& d_prob, const DV& z_prob, std::int32_t dd, std::int32_t pole_index,
                                         std::int32_t skip, T origin, T tau, T rho_inv) {
    const std::int32_t lower_end = pole_index + (skip < 0 ? 1 : 0);
    const std::int32_t upper_start = pole_index + 1 + (skip < 0 ? 0 : skip);

    T low_s = T(0), low_d = T(0), low_err = T(0);
    T up_s = T(0), up_d = T(0), up_err = T(0);
    auto accumulate = [&](std::int32_t i, T& s, T& d, T& err) {
        const T shifted = d_prob(i) - origin - tau;
        const T w = z_prob(i);
        const T ratio = w / shifted;
        const T wr = w * ratio;
        s += wr;
        d += ratio * ratio;
        err += s;
    };
    for_each_owned(ex, lower_end, [&](std::int32_t i) { accumulate(i, low_s, low_d, low_err); });
    for_each_owned(ex, Range{upper_start, dd}, [&](std::int32_t i) { accumulate(i, up_s, up_d, up_err); });

    const auto r = allreduce(ex, std::array<T, 6>{low_s, low_d, low_err, up_s, up_d, up_err});
    low_s = r[0], low_d = r[1], low_err = r[2], up_s = r[3], up_d = r[4], up_err = r[5];

    return {rho_inv + low_s + up_s, low_d + up_d, low_s, low_d, up_s, up_d, sycl::fabs(low_err) + sycl::fabs(up_err)};
}

template <typename T, typename Ex> struct EvalState {
    Eval<T, Ex> eval;
    T err;
    T ratio_sq;
};

template <typename T, typename Ex>
BATCHLAS_DLA_HELPER T roc_step(Eval<T, Ex>& eval, T sk, T sk1, T zk2, T zk12, T pole_k, T pole_k1, bool origin_lower, bool use_fixed,
                               T wi_ratio_sq) {
    T qc;
    if (use_fixed) {
        if (origin_lower) {
            qc = eval.secular_value - sk1 * eval.secular_derivative - (pole_k - pole_k1) * zk2 / (sk * sk);
        } else {
            qc = eval.secular_value - sk * eval.secular_derivative - (pole_k1 - pole_k) * zk12 / (sk1 * sk1);
        }
    } else {
        if (origin_lower) {
            eval.lower_derivative += wi_ratio_sq;
        } else {
            eval.upper_derivative += wi_ratio_sq;
        }
        qc = eval.secular_value - sk * eval.lower_derivative - sk1 * eval.upper_derivative;
    }

    T qa = (sk + sk1) * eval.secular_value - sk * sk1 * eval.secular_derivative;
    T qb = sk * sk1 * eval.secular_value;
    T step;

    if (qc == T(0)) {
        if (qa == T(0)) {
            if (use_fixed) {
                if (origin_lower) {
                    qa = zk2 + sk1 * sk1 * (eval.lower_derivative + eval.upper_derivative);
                } else {
                    qa = zk12 + sk * sk * (eval.lower_derivative + eval.upper_derivative);
                }
            } else {
                qa = sk * sk * eval.lower_derivative + sk1 * sk1 * eval.upper_derivative;
            }
        }
        step = qb / qa;
    } else {
        const T disc = sycl::sqrt(sycl::fabs(qa * qa - T(4) * qb * qc));
        step = (qa <= T(0)) ? (qa - disc) / (T(2) * qc) : T(2) * qb / (qa + disc);
    }

    if (eval.secular_value * step >= T(0)) {
        step = -eval.secular_value / eval.secular_derivative;
    }

    return step;
}

// Root k < dd - 1, in (d_k, d_{k+1}): the origin is the nearer pole (the sign of f at the
// midpoint), then ROC steps with the fixed-weight / middle-way switch.
template <typename T, typename Ex, typename DV>
BATCHLAS_DLA_HELPER laed4_root<T> solve_inner(const Ex& ex_in, const DV& d_prob, const DV& z_prob, std::int32_t dd, std::int32_t k, T rho,
                                              std::int32_t max_iter) {
    const Ex ex = ex_in;  // one load of the group fields, not one per owned loop
    const T rho_inv = T(1) / rho;
    const T pole_k = d_prob(k);
    const T pole_k1 = d_prob(k + 1);
    const T gap = pole_k1 - pole_k;

    const T midpoint = (pole_k + pole_k1) / T(2);
    auto eval = evaluate<T>(ex, d_prob, z_prob, dd, k, 1, T(0), midpoint, rho_inv);
    const T sec_no_poles = eval.secular_value;
    const T zk2 = z_prob(k) * z_prob(k);
    const T zk12 = z_prob(k + 1) * z_prob(k + 1);
    const T f_mid = sec_no_poles + T(2) * (zk12 - zk2) / gap;

    const bool origin_lower = (f_mid > T(0));
    T origin, tau, lower_bound, upper_bound;
    std::int32_t wi;

    if (origin_lower) {
        origin = pole_k;
        wi = k;
        lower_bound = T(0);
        upper_bound = gap / T(2);
        const T a = sec_no_poles * gap + zk2 + zk12;
        const T b = zk2 * gap;
        const T disc = sycl::sqrt(sycl::fabs(a * a - T(4) * b * sec_no_poles));
        tau = (a > T(0)) ? T(2) * b / (a + disc) : (a - disc) / (T(2) * sec_no_poles);
    } else {
        origin = pole_k1;
        wi = k + 1;
        lower_bound = -gap / T(2);
        upper_bound = T(0);
        const T a = sec_no_poles * gap - zk2 - zk12;
        const T b = zk12 * gap;
        const T disc = sycl::sqrt(sycl::fabs(a * a + T(4) * b * sec_no_poles));
        tau = (a < T(0)) ? T(2) * b / (a - disc) : -(a + disc) / (T(2) * sec_no_poles);
    }

    auto update_eval = [&](T current_tau) {
        EvalState<T, Ex> state;
        state.eval = evaluate<T>(ex, d_prob, z_prob, dd, wi, 0, origin, current_tau, rho_inv);
        const T shifted_wi = d_prob(wi) - origin - current_tau;
        const T z_wi = z_prob(wi);
        const T ratio_wi = z_wi / shifted_wi;
        state.ratio_sq = ratio_wi * ratio_wi;
        state.eval.secular_derivative += state.ratio_sq;
        const T pole_contrib = z_wi * ratio_wi;
        state.eval.secular_value += pole_contrib;

        state.err = state.eval.error_estimate + T(8) * (state.eval.upper_sum - state.eval.lower_sum) + T(2) * rho_inv +
                    T(3) * sycl::fabs(pole_contrib) + sycl::fabs(current_tau) * state.eval.secular_derivative;
        return state;
    };

    auto state = update_eval(tau);
    eval = state.eval;
    T err = state.err;
    T ratio_sq = state.ratio_sq;

    if (uniform(ex, sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err)) {
        return {origin, tau, true};  // converged before the first iteration
    }

    update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

    T sk = d_prob(k) - origin - tau;
    T sk1 = d_prob(k + 1) - origin - tau;
    T step = roc_step(eval, sk, sk1, zk2, zk12, pole_k, pole_k1, origin_lower, true, ratio_sq);
    step = bounded_step(step, tau, lower_bound, upper_bound, eval.secular_value);

    T prev_f = eval.secular_value;
    tau += step;

    state = update_eval(tau);
    eval = state.eval;
    err = state.err;
    ratio_sq = state.ratio_sq;

    bool use_fixed = (origin_lower ? -eval.secular_value : eval.secular_value) > (sycl::fabs(prev_f) / T(10));

    // Hoisted out of the loop condition: a root that ran out of max_iter must be told apart
    // from one that converged (the merge reports it through info).
    bool converged = false;
    for (std::int32_t iter = 1; iter < max_iter; ++iter) {
        if (uniform(ex, sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err)) {
            converged = true;
            break;
        }

        update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

        sk = d_prob(k) - origin - tau;
        sk1 = d_prob(k + 1) - origin - tau;

        step = roc_step(eval, sk, sk1, zk2, zk12, pole_k, pole_k1, origin_lower, use_fixed, ratio_sq);
        step = bounded_step(step, tau, lower_bound, upper_bound, eval.secular_value);

        prev_f = eval.secular_value;
        tau += step;

        state = update_eval(tau);
        eval = state.eval;
        err = state.err;
        ratio_sq = state.ratio_sq;

        if (eval.secular_value * prev_f > T(0) && sycl::fabs(eval.secular_value) > sycl::fabs(prev_f) / T(10)) {
            use_fixed = !use_fixed;
        }
    }

    return {origin, tau, converged};
}

// Root dd - 1, in (d_{dd-1}, d_{dd-1} + rho), origin the last pole; dd == 1 is exact.
template <typename T, typename Ex, typename DV>
BATCHLAS_DLA_HELPER laed4_root<T> solve_last(const Ex& ex_in, const DV& d_prob, const DV& z_prob, std::int32_t dd, T rho,
                                             std::int32_t max_iter) {
    const Ex ex = ex_in;  // one load of the group fields, not one per owned loop
    const T rho_inv = T(1) / rho;

    // dd == 1 has no second pole: `prev` below would be -1 and d_prob(prev) would read outside
    // the problem (a negative offset can fault). With one pole the secular equation
    // 1/rho + z0^2/(d0 - x) = 0 solves exactly: x = d0 + rho z0^2.
    if (dd <= 1) {
        const T z0 = z_prob(0);
        return {d_prob(0), rho * z0 * z0, true};
    }

    const std::int32_t last = dd - 1;
    const std::int32_t prev = dd - 2;
    const T d_last = d_prob(last);
    const T d_prev = d_prob(prev);
    const T origin = d_last;

    const T mid_tau = rho / T(2);
    auto eval = evaluate<T>(ex, d_prob, z_prob, dd, prev, 1, origin, mid_tau, rho_inv);
    const T sec_no_poles = eval.secular_value;
    const T zk2 = z_prob(prev) * z_prob(prev);
    const T zk12 = z_prob(last) * z_prob(last);
    const T f_init = sec_no_poles + zk2 / (d_prev - origin - mid_tau) - T(2) * zk12 * rho_inv;

    T tau, lower_bound, upper_bound;
    const T gap = d_last - d_prev;

    if (f_init > T(0)) {
        lower_bound = T(0);
        upper_bound = rho / T(2);
        const T a = -sec_no_poles * gap + zk2 + zk12;
        const T b = zk12 * gap;
        const T disc = sycl::sqrt(sycl::fabs(a * a + T(4) * b * sec_no_poles));
        tau = (a < T(0)) ? T(2) * b / (disc - a) : (a + disc) / (T(2) * sec_no_poles);
    } else {
        lower_bound = rho / T(2);
        upper_bound = rho;
        const T bound_check = zk2 / (gap + rho) + zk12 / rho;
        if (sec_no_poles <= bound_check) {
            tau = rho;
        } else {
            const T a = -sec_no_poles * gap + zk2 + zk12;
            const T b = zk12 * gap;
            const T disc = sycl::sqrt(sycl::fabs(a * a + T(4) * b * sec_no_poles));
            tau = (a < T(0)) ? T(2) * b / (disc - a) : (a + disc) / (T(2) * sec_no_poles);
        }
    }

    eval = evaluate<T>(ex, d_prob, z_prob, dd, prev, -1, origin, tau, rho_inv);

    T err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative) -
            T(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;

    if (sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err) {
        return {origin, tau, true};  // converged before the first iteration
    }

    update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

    T sk = d_prev - origin - tau;
    T sk1 = d_last - origin - tau;
    T qc = sycl::fabs(eval.secular_value - sk * eval.lower_derivative - sk1 * eval.upper_derivative);
    T qa = (sk + sk1) * eval.secular_value - sk * sk1 * (eval.lower_derivative + eval.upper_derivative);
    T qb = sk * sk1 * eval.secular_value;
    T step;

    if (qc == T(0)) {
        step = upper_bound - tau;
    } else {
        const T disc = sycl::sqrt(sycl::fabs(qa * qa - T(4) * qb * qc));
        step = (qa >= T(0)) ? (qa + disc) / (T(2) * qc) : T(2) * qb / (qa - disc);
    }

    if (eval.secular_value * step > T(0)) {
        step = -eval.secular_value / (eval.lower_derivative + eval.upper_derivative);
    }

    step = bounded_step(step, tau, lower_bound, upper_bound, eval.secular_value);
    tau += step;

    eval = evaluate<T>(ex, d_prob, z_prob, dd, prev, -1, origin, tau, rho_inv);
    err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative) -
          T(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;

    bool converged = false;
    for (std::int32_t iter = 1; iter < max_iter; ++iter) {
        if (uniform(ex, sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err)) {
            converged = true;
            break;
        }

        update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

        sk = d_prev - origin - tau;
        sk1 = d_last - origin - tau;

        qc = eval.secular_value - sk * eval.lower_derivative - sk1 * eval.upper_derivative;
        qa = (sk + sk1) * eval.secular_value - sk * sk1 * (eval.lower_derivative + eval.upper_derivative);
        qb = sk * sk1 * eval.secular_value;

        const T disc = sycl::sqrt(sycl::fabs(qa * qa - T(4) * qb * qc));
        step = (qa >= T(0)) ? (qa + disc) / (T(2) * qc) : T(2) * qb / (qa - disc);

        if (eval.secular_value * step > T(0)) {
            step = -eval.secular_value / (eval.lower_derivative + eval.upper_derivative);
        }

        step = bounded_step(step, tau, lower_bound, upper_bound, eval.secular_value);
        tau += step;

        eval = evaluate<T>(ex, d_prob, z_prob, dd, prev, -1, origin, tau, rho_inv);
        err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative) -
              T(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;
    }

    return {origin, tau, converged};
}

}  // namespace laed4_detail

/// Root k (0-based) of the dd-pole secular equation 1/rho + sum_i z_i^2 / (d_i - x) = 0, d
/// ascending and distinct, z without zeros, rho > 0 (a deflated merge: laed2). COLLECTIVE over
/// ex: every item calls with the same k; d and z must be visible to every item. The result is
/// uniform. max_iter >= 1 caps the iteration (StedcParams::max_sec_iter).
template <class Ex, class DV, class T>
BATCHLAS_DLA_HELPER laed4_root<T> laed4(const Ex& ex, const DV& d, const DV& z, std::int32_t dd, std::int32_t k, T rho, std::int32_t max_iter) {
    if (k == dd - 1) return laed4_detail::solve_last<T>(ex, d, z, dd, rho, max_iter);
    return laed4_detail::solve_inner<T>(ex, d, z, dd, k, rho, max_iter);
}

}  // namespace batchlas::device
