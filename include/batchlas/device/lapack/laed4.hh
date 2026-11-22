// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief `laed4` (xLAED4): one root of a divide-and-conquer merge's secular equation, collective
/// over an executor, as `stedc` and the host merges run it. Not covered by the compatibility
/// promise: its signature follows those callers.
/// @ingroup api_device_lowlevel
// xLAED4: one root of a divide-and-conquer merge's secular equation
//     f(x) = 1 / rho + sum_i z_i^2 / (d_i - x) = 0,   d ascending, rho > 0,
// collective over an executor (the ROCm stedc root iteration with LAPACK's error bound). Root k
// < dd - 1 lies in (d_k, d_{k+1}), root dd - 1 in (d_{dd-1}, d_{dd-1} + rho). Every evaluation
// sums over the executor (shuffles on a partition, work-group reductions on a group).
//   laed4                    the public op: info, and root k as origin + tau (origin a pole, so
//                            the caller forms d_i - x as (d_i - origin) - tau without cancellation)
//   detail::solve_inner / solve_last   the two root forms (the host merges' bodies)
//   detail::uniform    the leader's value in every item (a loop exit on a reduced value)
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

namespace detail {

// A secular root x = origin + tau; converged = the iteration met its error bound (false: it ran
// out of max_iter, the merge reports info > 0). laed4 returns it as (info, origin, tau).
template <class Real> struct SecularRoot {
    Real origin;
    Real tau;
    bool converged;
};

template <class Ex> inline constexpr bool is_group_v = is_work_group_v<Ex>;
template <int SG> inline constexpr bool is_group_v<RtGroup<Scope::Group, SG>> = true;

// The leader's v in every item (a loop exit on a reduced value, else a partition may diverge and
// deadlock); work-group: portable::group_broadcast, which moves a bool too (core's group bcast moves
// 4/8-byte words; DPC++'s own reaches half the items of a CPU work-group at sub-group 32).
template <class Ex, class T> BATCHLAS_DEVICE_INLINE T uniform(const Ex& ex, T v) {
    if constexpr (is_thread_v<Ex>) return v;
    else if constexpr (is_group_v<Ex>) return batchlas::portable::group_broadcast(ex.it.get_group(), v, 0);
    else return ex.uniform(v);
}

template <class Real, class Ex> struct Eval {
    Real secular_value;
    Real secular_derivative;
    Real lower_sum;
    Real lower_derivative;
    Real upper_sum;
    Real upper_derivative;
    Real error_estimate;
};

template <class Real> BATCHLAS_DEVICE_HELPER void update_bounds(Real secular_value, Real tau, Real& lower_bound,
                                                                Real& upper_bound) {
    if (secular_value <= Real(0)) lower_bound = sycl::fmax(lower_bound, tau);
    if (secular_value > Real(0)) upper_bound = sycl::fmin(upper_bound, tau);
}

template <class Real> BATCHLAS_DEVICE_HELPER Real bounded_step(Real step, Real tau, Real lower_bound, Real upper_bound,
                                                               Real secular_value) {
    if (tau + step > upper_bound || tau + step < lower_bound) {
        step = (secular_value < Real(0)) ? (upper_bound - tau) / Real(2) : (lower_bound - tau) / Real(2);
    }
    return step;
}

// The secular function at x = origin + tau without the terms i in [lower_end, upper_start):
// per item the sums of z_i^2 / (d_i - x), of (z_i / (d_i - x))^2 and of the running sums (the
// laed4 error term) below and above that window, then six reductions over the executor.
template <class Real, class Ex, class DV, class ZV>
BATCHLAS_DEVICE_HELPER Eval<Real, Ex> evaluate(const Ex& ex, const DV& d_prob, const ZV& z_prob, std::int32_t dd, std::int32_t pole_index,
                                         std::int32_t skip, Real origin, Real tau, Real rho_inv) {
    const std::int32_t lower_end = pole_index + (skip < 0 ? 1 : 0);
    const std::int32_t upper_start = pole_index + 1 + (skip < 0 ? 0 : skip);

    Real low_s = Real(0), low_d = Real(0), low_err = Real(0);
    Real up_s = Real(0), up_d = Real(0), up_err = Real(0);
    auto accumulate = [&](std::int32_t i, Real& s, Real& d, Real& err) {
        const Real shifted = d_prob(i) - origin - tau;
        const Real w = z_prob(i);
        const Real ratio = w / shifted;
        s += w * ratio;  // one expression: contracted to an FMA (two statements are not)
        d += ratio * ratio;
        err += s;
    };
    for_each_owned(ex, lower_end, [&](std::int32_t i) { accumulate(i, low_s, low_d, low_err); });
    for_each_owned(ex, Range{upper_start, dd}, [&](std::int32_t i) { accumulate(i, up_s, up_d, up_err); });

    const auto r = allreduce(ex, std::array<Real, 6>{low_s, low_d, low_err, up_s, up_d, up_err});
    low_s = r[0], low_d = r[1], low_err = r[2], up_s = r[3], up_d = r[4], up_err = r[5];

    return {rho_inv + low_s + up_s, low_d + up_d, low_s, low_d, up_s, up_d, sycl::fabs(low_err) + sycl::fabs(up_err)};
}

// No scratch: every evaluation forms d_i - x as (d_i - origin) - tau.
struct NoScratch {};

// d - origin into one item's scratch (a Thread's), once per root: the evaluations then run on
// (scratch, 0), the same values with one subtraction per term less. Returns the poles to use.
template <class Real, class Ex, class DV, class SV>
BATCHLAS_DEVICE_INLINE const auto& shift_poles(const DV& d_prob, const SV& scratch, std::int32_t dd, Real origin) {
    if constexpr (std::is_same_v<SV, NoScratch>) {
        return d_prob;
    } else {
        static_assert(is_thread_v<Ex>, "laed4: the scratch vector is one work-item's (Thread)");
        for (std::int32_t i = 0; i < dd; ++i) scratch(i) = d_prob(i) - origin;
        return scratch;
    }
}

template <class Real, class Ex> struct EvalState {
    Eval<Real, Ex> eval;
    Real err;
    Real ratio_sq;
};

template <class Real, class Ex>
BATCHLAS_DEVICE_HELPER Real roc_step(Eval<Real, Ex>& eval, Real sk, Real sk1, Real zk2, Real zk12, Real pole_k, Real pole_k1, bool origin_lower, bool use_fixed,
                               Real wi_ratio_sq) {
    Real qc;
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

    Real qa = (sk + sk1) * eval.secular_value - sk * sk1 * eval.secular_derivative;
    Real qb = sk * sk1 * eval.secular_value;
    Real step;

    if (qc == Real(0)) {
        if (qa == Real(0)) {
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
        const Real disc = sycl::sqrt(sycl::fabs(qa * qa - Real(4) * qb * qc));
        step = (qa <= Real(0)) ? (qa - disc) / (Real(2) * qc) : Real(2) * qb / (qa + disc);
    }

    if (eval.secular_value * step >= Real(0)) {
        step = -eval.secular_value / eval.secular_derivative;
    }

    return step;
}

// Root k < dd - 1, in (d_k, d_{k+1}): the origin is the nearer pole (the sign of f at the
// midpoint), then ROC steps with the fixed-weight / middle-way switch.
template <class Real, class Ex, class DV, class SV = NoScratch>
BATCHLAS_DEVICE_HELPER SecularRoot<Real> solve_inner(const Ex& ex_in, const DV& d_prob, const DV& z_prob, std::int32_t dd, std::int32_t k, Real rho,
                                              std::int32_t max_iter, const SV& scratch = {}) {
    const Ex ex = ex_in;  // one load of the group fields, not one per owned loop
    const Real rho_inv = Real(1) / rho;
    const Real pole_k = d_prob(k);
    const Real pole_k1 = d_prob(k + 1);
    const Real gap = pole_k1 - pole_k;

    const Real midpoint = (pole_k + pole_k1) / Real(2);
    auto eval = evaluate<Real>(ex, d_prob, z_prob, dd, k, 1, Real(0), midpoint, rho_inv);
    const Real sec_no_poles = eval.secular_value;
    const Real zk2 = z_prob(k) * z_prob(k);
    const Real zk12 = z_prob(k + 1) * z_prob(k + 1);
    const Real f_mid = sec_no_poles + Real(2) * (zk12 - zk2) / gap;

    const bool origin_lower = (f_mid > Real(0));
    Real origin, tau, lower_bound, upper_bound;
    std::int32_t wi;

    if (origin_lower) {
        origin = pole_k;
        wi = k;
        lower_bound = Real(0);
        upper_bound = gap / Real(2);
        const Real a = sec_no_poles * gap + zk2 + zk12;
        const Real b = zk2 * gap;
        const Real disc = sycl::sqrt(sycl::fabs(a * a - Real(4) * b * sec_no_poles));
        tau = (a > Real(0)) ? Real(2) * b / (a + disc) : (a - disc) / (Real(2) * sec_no_poles);
    } else {
        origin = pole_k1;
        wi = k + 1;
        lower_bound = -gap / Real(2);
        upper_bound = Real(0);
        const Real a = sec_no_poles * gap - zk2 - zk12;
        const Real b = zk12 * gap;
        const Real disc = sycl::sqrt(sycl::fabs(a * a + Real(4) * b * sec_no_poles));
        tau = (a < Real(0)) ? Real(2) * b / (a - disc) : -(a + disc) / (Real(2) * sec_no_poles);
    }

    constexpr bool kScratch = !std::is_same_v<SV, NoScratch>;
    const auto& dv = shift_poles<Real, Ex>(d_prob, scratch, dd, origin);
    const Real o = kScratch ? Real(0) : origin;

    auto update_eval = [&](Real current_tau) {
        EvalState<Real, Ex> state;
        state.eval = evaluate<Real>(ex, dv, z_prob, dd, wi, 0, o, current_tau, rho_inv);
        const Real shifted_wi = dv(wi) - o - current_tau;
        const Real z_wi = z_prob(wi);
        const Real ratio_wi = z_wi / shifted_wi;
        state.ratio_sq = ratio_wi * ratio_wi;
        state.eval.secular_derivative += state.ratio_sq;
        const Real pole_contrib = z_wi * ratio_wi;
        state.eval.secular_value += pole_contrib;

        state.err = state.eval.error_estimate + Real(8) * (state.eval.upper_sum - state.eval.lower_sum) + Real(2) * rho_inv +
                    Real(3) * sycl::fabs(pole_contrib) + sycl::fabs(current_tau) * state.eval.secular_derivative;
        return state;
    };

    auto state = update_eval(tau);
    eval = state.eval;
    Real err = state.err;
    Real ratio_sq = state.ratio_sq;

    if (uniform(ex, sycl::fabs(eval.secular_value) <= std::numeric_limits<Real>::epsilon() * err)) {
        return {origin, tau, true};  // converged before the first iteration
    }

    update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

    Real sk = dv(k) - o - tau;
    Real sk1 = dv(k + 1) - o - tau;
    Real step = roc_step(eval, sk, sk1, zk2, zk12, pole_k, pole_k1, origin_lower, true, ratio_sq);
    step = bounded_step(step, tau, lower_bound, upper_bound, eval.secular_value);

    Real prev_f = eval.secular_value;
    tau += step;

    state = update_eval(tau);
    eval = state.eval;
    err = state.err;
    ratio_sq = state.ratio_sq;

    bool use_fixed = (origin_lower ? -eval.secular_value : eval.secular_value) > (sycl::fabs(prev_f) / Real(10));

    // Hoisted out of the loop condition: a root that ran out of max_iter must be told apart
    // from one that converged (the merge reports it through info).
    bool converged = false;
    for (std::int32_t iter = 1; iter < max_iter; ++iter) {
        if (uniform(ex, sycl::fabs(eval.secular_value) <= std::numeric_limits<Real>::epsilon() * err)) {
            converged = true;
            break;
        }

        update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

        sk = dv(k) - o - tau;
        sk1 = dv(k + 1) - o - tau;

        step = roc_step(eval, sk, sk1, zk2, zk12, pole_k, pole_k1, origin_lower, use_fixed, ratio_sq);
        step = bounded_step(step, tau, lower_bound, upper_bound, eval.secular_value);

        prev_f = eval.secular_value;
        tau += step;

        state = update_eval(tau);
        eval = state.eval;
        err = state.err;
        ratio_sq = state.ratio_sq;

        if (eval.secular_value * prev_f > Real(0) && sycl::fabs(eval.secular_value) > sycl::fabs(prev_f) / Real(10)) {
            use_fixed = !use_fixed;
        }
    }

    return {origin, tau, converged};
}

// Root dd - 1, in (d_{dd-1}, d_{dd-1} + rho), origin the last pole; dd == 1 is exact.
template <class Real, class Ex, class DV, class SV = NoScratch>
BATCHLAS_DEVICE_HELPER SecularRoot<Real> solve_last(const Ex& ex_in, const DV& d_prob, const DV& z_prob, std::int32_t dd, Real rho,
                                             std::int32_t max_iter, const SV& scratch = {}) {
    const Ex ex = ex_in;  // one load of the group fields, not one per owned loop
    const Real rho_inv = Real(1) / rho;

    // dd == 1 has no second pole: `prev` below would be -1 and d_prob(prev) would read outside
    // the problem (a negative offset can fault). With one pole the secular equation
    // 1/rho + z0^2/(d0 - x) = 0 solves exactly: x = d0 + rho z0^2.
    if (dd <= 1) {
        const Real z0 = z_prob(0);
        return {d_prob(0), rho * z0 * z0, true};
    }

    const std::int32_t last = dd - 1;
    const std::int32_t prev = dd - 2;
    const Real d_last = d_prob(last);
    const Real d_prev = d_prob(prev);
    const Real origin = d_last;
    constexpr bool kScratch = !std::is_same_v<SV, NoScratch>;
    const auto& dv = shift_poles<Real, Ex>(d_prob, scratch, dd, origin);
    const Real o = kScratch ? Real(0) : origin;
    Real d_prev_o = d_prev, d_last_o = d_last;  // d - o at the two poles
    if constexpr (kScratch) d_prev_o = scratch(prev), d_last_o = scratch(last);

    const Real mid_tau = rho / Real(2);
    auto eval = evaluate<Real>(ex, dv, z_prob, dd, prev, 1, o, mid_tau, rho_inv);
    const Real sec_no_poles = eval.secular_value;
    const Real zk2 = z_prob(prev) * z_prob(prev);
    const Real zk12 = z_prob(last) * z_prob(last);
    const Real f_init = sec_no_poles + zk2 / (d_prev_o - o - mid_tau) - Real(2) * zk12 * rho_inv;

    Real tau, lower_bound, upper_bound;
    const Real gap = d_last - d_prev;

    if (f_init > Real(0)) {
        lower_bound = Real(0);
        upper_bound = rho / Real(2);
        const Real a = -sec_no_poles * gap + zk2 + zk12;
        const Real b = zk12 * gap;
        const Real disc = sycl::sqrt(sycl::fabs(a * a + Real(4) * b * sec_no_poles));
        tau = (a < Real(0)) ? Real(2) * b / (disc - a) : (a + disc) / (Real(2) * sec_no_poles);
    } else {
        lower_bound = rho / Real(2);
        upper_bound = rho;
        const Real bound_check = zk2 / (gap + rho) + zk12 / rho;
        if (sec_no_poles <= bound_check) {
            tau = rho;
        } else {
            const Real a = -sec_no_poles * gap + zk2 + zk12;
            const Real b = zk12 * gap;
            const Real disc = sycl::sqrt(sycl::fabs(a * a + Real(4) * b * sec_no_poles));
            tau = (a < Real(0)) ? Real(2) * b / (disc - a) : (a + disc) / (Real(2) * sec_no_poles);
        }
    }

    eval = evaluate<Real>(ex, dv, z_prob, dd, prev, -1, o, tau, rho_inv);

    Real err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative) -
            Real(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;

    if (sycl::fabs(eval.secular_value) <= std::numeric_limits<Real>::epsilon() * err) {
        return {origin, tau, true};  // converged before the first iteration
    }

    update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

    Real sk = d_prev_o - o - tau;
    Real sk1 = d_last_o - o - tau;
    Real qc = sycl::fabs(eval.secular_value - sk * eval.lower_derivative - sk1 * eval.upper_derivative);
    Real qa = (sk + sk1) * eval.secular_value - sk * sk1 * (eval.lower_derivative + eval.upper_derivative);
    Real qb = sk * sk1 * eval.secular_value;
    Real step;

    if (qc == Real(0)) {
        step = upper_bound - tau;
    } else {
        const Real disc = sycl::sqrt(sycl::fabs(qa * qa - Real(4) * qb * qc));
        step = (qa >= Real(0)) ? (qa + disc) / (Real(2) * qc) : Real(2) * qb / (qa - disc);
    }

    if (eval.secular_value * step > Real(0)) {
        step = -eval.secular_value / (eval.lower_derivative + eval.upper_derivative);
    }

    step = bounded_step(step, tau, lower_bound, upper_bound, eval.secular_value);
    tau += step;

    eval = evaluate<Real>(ex, dv, z_prob, dd, prev, -1, o, tau, rho_inv);
    err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative) -
          Real(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;

    bool converged = false;
    for (std::int32_t iter = 1; iter < max_iter; ++iter) {
        if (uniform(ex, sycl::fabs(eval.secular_value) <= std::numeric_limits<Real>::epsilon() * err)) {
            converged = true;
            break;
        }

        update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

        sk = d_prev_o - o - tau;
        sk1 = d_last_o - o - tau;

        qc = eval.secular_value - sk * eval.lower_derivative - sk1 * eval.upper_derivative;
        qa = (sk + sk1) * eval.secular_value - sk * sk1 * (eval.lower_derivative + eval.upper_derivative);
        qb = sk * sk1 * eval.secular_value;

        const Real disc = sycl::sqrt(sycl::fabs(qa * qa - Real(4) * qb * qc));
        step = (qa >= Real(0)) ? (qa + disc) / (Real(2) * qc) : Real(2) * qb / (qa - disc);

        if (eval.secular_value * step > Real(0)) {
            step = -eval.secular_value / (eval.lower_derivative + eval.upper_derivative);
        }

        step = bounded_step(step, tau, lower_bound, upper_bound, eval.secular_value);
        tau += step;

        eval = evaluate<Real>(ex, dv, z_prob, dd, prev, -1, o, tau, rho_inv);
        err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative) -
              Real(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;
    }

    return {origin, tau, converged};
}

// laed4's body: root k as a SecularRoot. Not force-inlined: the callers' code stays the one it
// was when laed4 itself returned the root.
template <class Ex, class DV, class Real, class SV>
BATCHLAS_DEVICE_HELPER SecularRoot<Real> secular_root(const Ex& ex, const DV& d, const DV& z, std::int32_t dd, std::int32_t k, Real rho,
                                                      std::int32_t max_iter, const SV& scratch) {
    if constexpr (sizeof(Real) < 8 || std::is_same_v<SV, NoScratch>) {
        if (k == dd - 1) return solve_last<Real>(ex, d, z, dd, rho, max_iter);
        return solve_inner<Real>(ex, d, z, dd, k, rho, max_iter);
    } else {
        if (k == dd - 1) return solve_last<Real>(ex, d, z, dd, rho, max_iter, scratch);
        return solve_inner<Real>(ex, d, z, dd, k, rho, max_iter, scratch);
    }
}

}  // namespace detail

/// @brief xLAED4: root k of the dd-pole secular equation
/// \f$1/\rho + \sum_i z_i^2 / (d_i - x) = 0\f$, returned as x = origin + tau.
///
/// d ascending and distinct, z without zeros and rho > 0, as a deflated merge (laed2()) leaves
/// them. Root k < dd - 1 lies in \f$(d_k, d_{k+1})\f$, root dd - 1 in
/// \f$(d_{dd-1}, d_{dd-1} + \rho)\f$. origin is a pole, so the caller forms \f$d_i - x\f$ as
/// \f$(d_i - \mathrm{origin}) - \tau\f$ without cancellation. COLLECTIVE over `ex`: every item
/// calls with the same k, and d and z must be visible to every item.
/// @param[in]  ex        the executor (a partition or a work-group)
/// @param[in]  d         the dd poles
/// @param[in]  z         the dd weights
/// @param[in]  dd        the number of poles
/// @param[in]  k         the root's index, 0-based
/// @param[in]  rho       the rank-one weight, > 0
/// @param[out] origin    the pole the root is measured from
/// @param[out] tau       the root's offset from origin
/// @param[in]  max_iter  the iteration cap, >= 1 (the host StedcParams::max_sec_iter)
/// @param      scratch   `Thread` executor only: dd writable entries of this item, undefined on
///                       return; the root is the same with or without it
/// @retval 0  converged
/// @retval 1  the iteration ran out of max_iter (xLAED4's info)
/// @return    The value is uniform over the executor.
/// @ingroup api_device_lowlevel
// FP64 holds d - origin in scratch (one subtraction per term less); float ignores it (its strided
// reads cost more than the subtraction). evidence: docs/perf/stedc.md#stedc-one-secular-solver
template <class Ex, class DV, class Real, class SV = detail::NoScratch>
BATCHLAS_DEVICE_INLINE int laed4(const Ex& ex, const DV& d, const DV& z, int dd, int k, Real rho, Real& origin, Real& tau, int max_iter,
                                 const SV& scratch = {}) {
    const auto root = detail::secular_root(ex, d, z, dd, k, rho, max_iter, scratch);
    origin = root.origin;
    tau = root.tau;
    return static_cast<int>(!root.converged);
}

}  // namespace batchlas::device
