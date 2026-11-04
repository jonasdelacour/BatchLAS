// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <src/util/template-instantiations.hh>

#include <batchlas/util/group-invoke.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>
#include <src/kernels/common/dla_v2_partition.hh>
#include <src/ops/syev/stedc/stedc_v2.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>

#include <src/math-helpers.hh>
#include <src/ops/syev/stedc/stedc_secular.hh>
#include <src/ops/syev/stedc/stedc_merge_kernels.hh>
#include <src/kernels/common/info_span.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace batchlas {

namespace v2 = batchlas::device;

template <Backend B, typename T, int32_t P>
class StedcFusedCtaMerge;

template <Backend B, typename T>
class StedcFusedWgMerge;

// `kernel_max_wg` is the largest work-group THIS kernel can launch on this device,
// generally below the device maximum (register file per sub-partition: the merge
// kernel caps at 768 on sm_89, and 1024 throws). 0 means unknown: device limit only.
// evidence: docs/perf/stedc.md#stedc-the-work-group-multiplier-after-the-barriers-went
inline int32_t choose_wg_size(const sycl::device& dev,
                              int32_t base_wg_size,
                              int32_t requested_mul,
                              int32_t kernel_max_wg = 0) {
    int32_t wg_mul = std::max<int32_t>(1, requested_mul);
    int32_t wg_size = base_wg_size * wg_mul;

    int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
    if (kernel_max_wg > 0) {
        max_wg_size = std::min(max_wg_size, kernel_max_wg);
    }
    if (wg_size > max_wg_size) {
        const int32_t max_mul = std::max<int32_t>(1, max_wg_size / base_wg_size);
        wg_mul = std::min(wg_mul, max_mul);
        wg_size = base_wg_size * wg_mul;
    }
    return wg_size;
}

// Largest work-group the given kernel can actually launch on `dev`. Returns 0 if
// the runtime cannot answer, so the caller falls back to the device limit.
template <typename KernelName>
inline int32_t kernel_max_work_group_size(Queue& ctx, const sycl::device& dev) {
    try {
        return static_cast<int32_t>(impl::kernel_max_wg_size<KernelName>(ctx->get_context(), dev));
    } catch (const sycl::exception&) {
        return 0;
    }
}

// Poles and weights into local memory, then a barrier: every root solve reads all of them.
template <typename T, typename Ex>
inline void load_local_problem_vectors(const Ex& ex,
                                       const VectorView<T>& d_prob_global,
                                       const VectorView<T>& z_prob_global,
                                       const VectorView<T>& d_prob,
                                       const VectorView<T>& z_prob) {
    v2::copy(ex, d_prob_global, d_prob);
    v2::copy(ex, z_prob_global, z_prob);
    stedc_v2::group_sync(ex);
}

template <typename T, typename Ex>
struct ShiftedEvalResult {
    T secular_value;
    T secular_derivative;
    T lower_sum;
    T lower_derivative;
    T upper_sum;
    T upper_derivative;
    T error_estimate;
};

template <typename T>
inline void update_bounds(T secular_value, T tau, T& lower_bound, T& upper_bound) {
    if (secular_value <= T(0)) lower_bound = sycl::fmax(lower_bound, tau);
    if (secular_value > T(0)) upper_bound = sycl::fmin(upper_bound, tau);
}

template <typename T>
inline T apply_step_with_bounds(T step,
                                T tau,
                                T lower_bound,
                                T upper_bound,
                                T secular_value) {
    if (tau + step > upper_bound || tau + step < lower_bound) {
        step = (secular_value < T(0)) ? (upper_bound - tau) / T(2) : (lower_bound - tau) / T(2);
    }
    return step;
}

// The secular function at x = origin + tau without the terms i in [lower_end, upper_start):
// per item the sums of z_i^2 / (d_i - x), of (z_i / (d_i - x))^2 and of the running sums (the
// laed4 error term) below and above that window, then six reductions over the executor
// (shuffles on a partition, work-group reductions on a RtGroup).
template <typename T, typename Ex>
inline ShiftedEvalResult<T, Ex> evaluate_shifted_generic(
    const Ex& ex,
    const VectorView<T>& d_prob,
    const VectorView<T>& z_prob,
    int32_t dd,
    int32_t pole_index,
    int32_t skip,
    T origin,
    T tau,
    T rho_inv) {
    const int32_t lower_end = pole_index + (skip < 0 ? 1 : 0);
    const int32_t upper_start = pole_index + 1 + (skip < 0 ? 0 : skip);

    T low_s = T(0), low_d = T(0), low_err = T(0);
    T up_s = T(0), up_d = T(0), up_err = T(0);
    auto accumulate = [&](int32_t i, T& s, T& d, T& err) {
        const T shifted = d_prob(i) - origin - tau;
        const T w = z_prob(i);
        const T ratio = w / shifted;
        const T wr = w * ratio;
        s += wr;
        d += ratio * ratio;
        err += s;
    };
    v2::for_each_owned(ex, lower_end, [&](int32_t i) { accumulate(i, low_s, low_d, low_err); });
    v2::for_each_owned(ex, v2::Range{upper_start, dd}, [&](int32_t i) { accumulate(i, up_s, up_d, up_err); });

    const auto r = v2::allreduce(ex, std::array<T, 6>{low_s, low_d, low_err, up_s, up_d, up_err});
    low_s = r[0], low_d = r[1], low_err = r[2], up_s = r[3], up_d = r[4], up_err = r[5];

    return {rho_inv + low_s + up_s,
            low_d + up_d,
            low_s, low_d,
            up_s, up_d,
            sycl::fabs(low_err) + sycl::fabs(up_err)};
}

template <typename T, typename Ex>
struct RocEvalState {
    ShiftedEvalResult<T, Ex> eval;
    T err;
    T ratio_sq;
};

template <typename T, typename Ex>
inline T compute_roc_step(ShiftedEvalResult<T, Ex>& eval,
                          T sk,
                          T sk1,
                          T zk2,
                          T zk12,
                          T pole_k,
                          T pole_k1,
                          bool origin_lower,
                          bool use_fixed,
                          T wi_ratio_sq) {
    T qc;
    if (use_fixed) {
        if (origin_lower) {
            qc = eval.secular_value - sk1 * eval.secular_derivative
                 - (pole_k - pole_k1) * zk2 / (sk * sk);
        } else {
            qc = eval.secular_value - sk * eval.secular_derivative
                 - (pole_k1 - pole_k) * zk12 / (sk1 * sk1);
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

// Result of a secular-equation root solve. We return origin and tau separately
// so the caller can compute denominators as `(d_prob[i] - origin) - tau` and
// preserve precision when d_prob[i] equals the origin pole (catastrophic
// cancellation destroys tau when computed as `d_prob[i] - (origin + tau)`).
template <typename T>
struct RocRoot {
    T origin;
    T tau;
    // Whether the bisection/interpolation loop below broke on its convergence
    // test rather than running out of max_iter.
    bool converged;
};

template <typename T, typename Ex>
inline RocRoot<T> solve_root_roc_generic(const Ex& ex_in,
                                const VectorView<T>& d_prob,
                                const VectorView<T>& z_prob,
                                int32_t dd,
                                int32_t k,
                                T rho,
                                int32_t max_iter) {
    const Ex ex = ex_in;  // one load of the group fields, not one per owned loop
    const T rho_inv = T(1) / rho;
    const T pole_k = d_prob(k);
    const T pole_k1 = d_prob(k + 1);
    const T gap = pole_k1 - pole_k;

    const T midpoint = (pole_k + pole_k1) / T(2);
    auto eval = evaluate_shifted_generic<T>(ex, d_prob, z_prob, dd, k, 1,
                                            T(0), midpoint, rho_inv);
    const T sec_no_poles = eval.secular_value;
    const T zk2 = z_prob(k) * z_prob(k);
    const T zk12 = z_prob(k + 1) * z_prob(k + 1);
    const T f_mid = sec_no_poles + T(2) * (zk12 - zk2) / gap;

    const bool origin_lower = (f_mid > T(0));
    T origin, tau, lower_bound, upper_bound;
    int32_t wi;

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
        RocEvalState<T, Ex> state;
        state.eval = evaluate_shifted_generic<T>(ex, d_prob, z_prob, dd, wi, 0,
                                                 origin, current_tau, rho_inv);
        const T shifted_wi = d_prob(wi) - origin - current_tau;
        const T z_wi = z_prob(wi);
        const T ratio_wi = z_wi / shifted_wi;
        state.ratio_sq = ratio_wi * ratio_wi;
        state.eval.secular_derivative += state.ratio_sq;
        const T pole_contrib = z_wi * ratio_wi;
        state.eval.secular_value += pole_contrib;

        state.err = state.eval.error_estimate + T(8) * (state.eval.upper_sum - state.eval.lower_sum)
                    + T(2) * rho_inv + T(3) * sycl::fabs(pole_contrib)
                    + sycl::fabs(current_tau) * state.eval.secular_derivative;
        return state;
    };

    auto state = update_eval(tau);
    eval = state.eval;
    T err = state.err;
    T ratio_sq = state.ratio_sq;

    if (stedc_v2::uniform(ex, sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err)) {
        return {origin, tau, true};   // converged before the first iteration
    }

    update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

    T sk = d_prob(k) - origin - tau;
    T sk1 = d_prob(k + 1) - origin - tau;
    T step = compute_roc_step(eval, sk, sk1, zk2, zk12, pole_k, pole_k1,
                              origin_lower, true, ratio_sq);
    step = apply_step_with_bounds(step, tau, lower_bound, upper_bound, eval.secular_value);

    T prev_f = eval.secular_value;
    tau += step;

    state = update_eval(tau);
    eval = state.eval;
    err = state.err;
    ratio_sq = state.ratio_sq;

    bool use_fixed = (origin_lower ? -eval.secular_value : eval.secular_value)
                     > (sycl::fabs(prev_f) / T(10));

    // Hoisted out of the loop condition: on cap exhaustion this function used to
    // just return {origin, tau} with no signal at all, so a root that never
    // converged was indistinguishable from one that did.
    bool converged = false;
    for (int32_t iter = 1; iter < max_iter; ++iter) {
        if (stedc_v2::uniform(ex, sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err)) {
            converged = true;
            break;
        }

        update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

        sk = d_prob(k) - origin - tau;
        sk1 = d_prob(k + 1) - origin - tau;

        step = compute_roc_step(eval, sk, sk1, zk2, zk12, pole_k, pole_k1,
                                origin_lower, use_fixed, ratio_sq);
        step = apply_step_with_bounds(step, tau, lower_bound, upper_bound, eval.secular_value);

        prev_f = eval.secular_value;
        tau += step;

        state = update_eval(tau);
        eval = state.eval;
        err = state.err;
        ratio_sq = state.ratio_sq;

        if (eval.secular_value * prev_f > T(0)
            && sycl::fabs(eval.secular_value) > sycl::fabs(prev_f) / T(10)) {
            use_fixed = !use_fixed;
        }
    }

    return {origin, tau, converged};
}

template <typename T, typename Ex>
inline RocRoot<T> solve_root_ext_generic(const Ex& ex_in,
                                const VectorView<T>& d_prob,
                                const VectorView<T>& z_prob,
                                int32_t dd,
                                T rho,
                                int32_t max_iter) {
    const Ex ex = ex_in;  // one load of the group fields, not one per owned loop
    const T rho_inv = T(1) / rho;

    // dd == 1 has no second pole: `prev` below would be -1 and d_prob(prev)
    // would read outside the shared-memory window (d_prob aliases d_local via a
    // generic pointer, so a negative offset can fault). With a single pole the
    // secular equation 1/rho + z0^2/(d0 - x) = 0 solves exactly:
    //     x = d0 + rho * z0^2.
    // evidence: docs/perf/stedc.md#stedc-the-dd--1-secular-solve-fault
    if (dd <= 1) {
        const T z0 = z_prob(0);
        return {d_prob(0), rho * z0 * z0, true};  // exact: a two-member init left converged false
    }

    const int32_t last = dd - 1;
    const int32_t prev = dd - 2;
    const T d_last = d_prob(last);
    const T d_prev = d_prob(prev);
    const T origin = d_last;

    const T mid_tau = rho / T(2);
    auto eval = evaluate_shifted_generic<T>(ex, d_prob, z_prob, dd, prev, 1,
                                            origin, mid_tau, rho_inv);
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

    eval = evaluate_shifted_generic<T>(ex, d_prob, z_prob, dd, prev, -1,
                                       origin, tau, rho_inv);

    T err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative)
            - T(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;

    if (sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err) {
        return {origin, tau, true};   // converged before the first iteration
    }

    update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

    T sk = d_prev - origin - tau;
    T sk1 = d_last - origin - tau;
    T qc = sycl::fabs(eval.secular_value - sk * eval.lower_derivative - sk1 * eval.upper_derivative);
    T qa = (sk + sk1) * eval.secular_value
           - sk * sk1 * (eval.lower_derivative + eval.upper_derivative);
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

    step = apply_step_with_bounds(step, tau, lower_bound, upper_bound, eval.secular_value);
    tau += step;

    eval = evaluate_shifted_generic<T>(ex, d_prob, z_prob, dd, prev, -1,
                                       origin, tau, rho_inv);
    err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative)
          - T(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;

    bool converged = false;
    for (int32_t iter = 1; iter < max_iter; ++iter) {
        if (stedc_v2::uniform(ex, sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err)) {
            converged = true;
            break;
        }

        update_bounds(eval.secular_value, tau, lower_bound, upper_bound);

        sk = d_prev - origin - tau;
        sk1 = d_last - origin - tau;

        qc = eval.secular_value - sk * eval.lower_derivative - sk1 * eval.upper_derivative;
        qa = (sk + sk1) * eval.secular_value
             - sk * sk1 * (eval.lower_derivative + eval.upper_derivative);
        qb = sk * sk1 * eval.secular_value;

        const T disc = sycl::sqrt(sycl::fabs(qa * qa - T(4) * qb * qc));
        step = (qa >= T(0)) ? (qa + disc) / (T(2) * qc) : T(2) * qb / (qa - disc);

        if (eval.secular_value * step > T(0)) {
            step = -eval.secular_value / (eval.lower_derivative + eval.upper_derivative);
        }

        step = apply_step_with_bounds(step, tau, lower_bound, upper_bound, eval.secular_value);
        tau += step;

        eval = evaluate_shifted_generic<T>(ex, d_prob, z_prob, dd, prev, -1,
                                           origin, tau, rho_inv);
        err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative)
              - T(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;
    }

    return {origin, tau, converged};
}

// q(i) = (d(i) - origin) - tau, the root's denominators. Not d(i) - (origin + tau): at the
// origin pole this form is -tau exactly, where the other cancels catastrophically. The solver
// keeps tau != 0 for a non-deflated root, so only an exact zero (extreme cancellation) is
// clamped, to eps (|d(i)| + 1), to keep the Löwner reciprocal finite.
template <typename T, typename Ex, typename QCol>
inline void write_denominator_column(const Ex& ex, const QCol& q, const VectorView<T>& d_prob, T origin, T tau) {
    v2::map(ex, q, [=](T di) {
        T denom = (di - origin) - tau;
        if (denom == T(0)) {
            denom = std::numeric_limits<T>::epsilon() * (sycl::fabs(di) + T(1));
        }
        return denom;
    }, d_prob);
}

// Reciprocate-and-normalize: Q(:, eig) = z ./ Q(:, eig), then divided by its Blue 2-norm (three
// butterflies on a partition, one vec<R, 3> reduction on a work-group). Same `first`/`stride`
// convention as maybe_rescale_vectors: columns are independent (column `eig` reads z and only
// touches Q(:, eig)), so one column per partition needs no work-group barrier.
template <typename T, typename Ex, typename QMat, typename VView>
inline void normalize_vectors(const Ex& ex,
                              int32_t first,
                              int32_t stride,
                              int32_t dd,
                              const QMat& Q,
                              const VView& v,
                              int32_t bid) {
    // Uniform trip count across reduction groups -- see maybe_rescale_vectors. nrm2
    // reduces with sub-group shuffles, so a partition must not fall out of the loop while
    // others are inside it: an inactive group runs every op on an empty column.
    const int32_t iters = (dd + stride - 1) / stride;
    const auto z = v2::vec(v.batch_item(bid));

    for (int32_t it = 0; it < iters; ++it) {
        const int32_t eig = first + it * stride;
        const bool active = (eig < dd);
        const auto q = Q.col(active ? eig : 0).head(active ? dd : 0);
        v2::map(ex, q, [](T zi, T den) { return zi / den; }, z, q);
        // No barrier: nrm2 walks q with map's owner stride, so every item reads back
        // only what it just wrote.
        const T nrm = v2::nrm2<v2::alg::Blue>(ex, q);
        v2::map(ex, q, [=](T qi) { return qi / nrm; }, q);
        stedc_v2::group_sync(ex);
    }
}

template <Backend B, typename T, int32_t P>
void stedc_merge_fused_cta_impl(Queue& ctx,
                                const VectorView<T>& eigenvalues,
                                const VectorView<T>& v,
                                const Span<T>& rho,
                                const Span<int32_t>& n_reduced,
                                const MatrixView<T, MatrixFormat::Dense>& Qprime,
                                const VectorView<T>& temp_lambdas,
                                const StedcParams<T>& params,
                                int32_t* info,
                                int64_t info_nodes_per_item) {
    const auto batch_size = eigenvalues.batch_size();
    const int32_t nloc = static_cast<int32_t>(Qprime.rows());
    const int32_t sg_size = 32;

    const auto dev = ctx->get_device();
    const int32_t base_wg_size = std::lcm<int32_t>(P, sg_size);
    const int32_t kernel_max_wg = kernel_max_work_group_size<StedcFusedCtaMerge<B, T, P>>(ctx, dev);
    const int32_t wg_size = choose_wg_size(dev, base_wg_size, params.secular_cta_wg_size_multiplier,
                                           kernel_max_wg);

    const bool do_rescale = params.enable_rescale;
    // Locals so the kernel's `[=]` copies a pointer and a scalar; nullptr makes
    // info_report a no-op. See src/kernels/common/info_span.hh.
    int32_t* const info_dev = info;
    const int64_t nodes_per_item = info_nodes_per_item;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto d_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);
        auto z_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);

        batchlas::impl::parallel_for<StedcFusedCtaMerge<B, T, P>>(h,
            sycl::nd_range<1>(batch_size * wg_size, wg_size),
            [=](sycl::nd_item<1> item) BATCHLAS_REQD_SG_SIZE(sg_size) {
                const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
                const v2::RtGroup<v2::Scope::Group, sg_size> wex{item};
                // One root per partition of P lanes; the solver's reductions are shuffles.
                const v2::SgpPartition<P> pex{item.get_sub_group()};
                const int32_t sg_id = static_cast<int32_t>(item.get_sub_group().get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(pex.part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(pex.part.get_group_linear_id());
                const int32_t parts_per_wg = wg_size / P;

                const int32_t dd = n_reduced[bid];
                if (dd <= 0) {
                    return;
                }

                auto Q_bid = Qview.batch_item(bid);
                const auto Q = v2::mat(Q_bid);
                const auto d_prob = VectorView<T>(util::get_raw_ptr(d_local), dd);
                const auto z_prob = VectorView<T>(util::get_raw_ptr(z_local), dd);
                load_local_problem_vectors(wex, eigenvalues.batch_item(bid), v.batch_item(bid), d_prob, z_prob);

                const T abs_2rho = sycl::fabs(T(2) * rho[bid]);

                for (int32_t root_ix = part_id; root_ix < dd; root_ix += parts_per_wg) {
                    RocRoot<T> root;
                    if (root_ix == dd - 1) {
                        root = solve_root_ext_generic(pex, d_prob, z_prob, dd,
                                                      abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    } else {
                        root = solve_root_roc_generic(pex, d_prob, z_prob, dd, root_ix,
                                                      abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    }

                    if (pex.leader()) {
                        temp_lambdas(root_ix, bid) = root.origin + root.tau;
                        // This is the one merge arm that honours params.max_sec_iter,
                        // so it is also the one a forced-non-convergence test can
                        // drive from the public StedcParams.
                        if (!root.converged) {
                            detail::info_report(info_dev,
                                                detail::info_item(static_cast<int64_t>(bid), nodes_per_item),
                                                1);
                        }
                    }
                    write_denominator_column(pex, Q.col(root_ix).head(dd), d_prob, root.origin, root.tau);
                }

                stedc_v2::group_sync(wex);

                // Rescale and normalize are parallel over their outer index, so
                // hand one index to each partition instead of walking them
                // sequentially behind a work-group reduction per index.
                maybe_rescale_vectors(do_rescale, pex, part_id, parts_per_wg, dd,
                                      Q_bid, d_prob, v, bid);
                // Rescale reads all of Q and writes all of v; normalize
                // overwrites Q and reads all of v. One barrier separates the two
                // phases -- the loops themselves no longer need any.
                stedc_v2::group_sync(wex);
                normalize_vectors<T>(pex, part_id, parts_per_wg, dd, Q, v, bid);
            });
    });
}

template <Backend B, typename T>
void stedc_merge_fused_wg(Queue& ctx,
                          const VectorView<T>& eigenvalues,
                          const VectorView<T>& v,
                          const Span<T>& rho,
                          const Span<int32_t>& n_reduced,
                          const MatrixView<T, MatrixFormat::Dense>& Qprime,
                          const VectorView<T>& temp_lambdas,
                          const StedcParams<T>& params,
                          int32_t* info,
                          int64_t info_nodes_per_item) {
    const int32_t nloc = static_cast<int32_t>(Qprime.rows());
    const auto batch_size = eigenvalues.batch_size();
    const auto dev = ctx->get_device();

    const int32_t base_wg_size = 32;  // the CTA kernels' lane count; wider requests come here
    const int32_t kernel_max_wg = kernel_max_work_group_size<StedcFusedWgMerge<B, T>>(ctx, dev);
    const int32_t wg_size = choose_wg_size(dev, base_wg_size, params.secular_cta_wg_size_multiplier,
                                           kernel_max_wg);

    const bool do_rescale = params.enable_rescale;
    int32_t* const info_dev = info;
    const int64_t nodes_per_item = info_nodes_per_item;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto d_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);
        auto z_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);

        batchlas::impl::parallel_for<StedcFusedWgMerge<B, T>>(h,
            sycl::nd_range<1>(batch_size * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
                const v2::RtGroup<> ex{item};

                const int32_t dd = n_reduced[bid];
                if (dd <= 0) {
                    return;
                }

                auto Q_bid = Qview.batch_item(bid);
                const auto Q = v2::mat(Q_bid);
                const auto d_prob = VectorView<T>(util::get_raw_ptr(d_local), dd);
                const auto z_prob = VectorView<T>(util::get_raw_ptr(z_local), dd);
                load_local_problem_vectors(ex, eigenvalues.batch_item(bid), v.batch_item(bid), d_prob, z_prob);

                const T abs_2rho = sycl::fabs(T(2) * rho[bid]);

                // WG variant: the reference path, one root at a time over the whole group.
                for (int32_t root_ix = 0; root_ix < dd; ++root_ix) {
                    RocRoot<T> root;
                    if (root_ix == dd - 1) {
                        root = solve_root_ext_generic(ex, d_prob, z_prob, dd,
                                                      abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    } else {
                        root = solve_root_roc_generic(ex, d_prob, z_prob, dd, root_ix,
                                                      abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    }

                    if (ex.leader()) {
                        temp_lambdas(root_ix, bid) = root.origin + root.tau;
                        if (!root.converged) {
                            detail::info_report(info_dev,
                                                detail::info_item(static_cast<int64_t>(bid), nodes_per_item),
                                                1);
                        }
                    }
                    write_denominator_column(ex, Q.col(root_ix).head(dd), d_prob, root.origin, root.tau);
                }

                stedc_v2::group_sync(ex);

                // Sequential over the index with work-group-wide reductions: (first, stride) = (0, 1).
                maybe_rescale_vectors(do_rescale, ex, 0, 1, dd, Q_bid, d_prob, v, bid);
                normalize_vectors<T>(ex, 0, 1, dd, Q, v, bid);
            });
    });
}

template <Backend B, typename T>
void stedc_merge_fused_cta(Queue& ctx,
                           const VectorView<T>& eigenvalues,
                           const VectorView<T>& v,
                           const Span<T>& rho,
                           const Span<int32_t>& n_reduced,
                           const MatrixView<T, MatrixFormat::Dense>& Qprime,
                           const VectorView<T>& temp_lambdas,
                           const StedcParams<T>& params,
                           int32_t* info,
                           int64_t info_nodes_per_item) {
    const bool has32 = ctx.device().supports_sub_group_size(32);

    if (!has32) {
        stedc_merge_fused<B, T>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
        return;
    }

    const int32_t requested = std::max<int32_t>(4, params.secular_threads_per_root);
    const bool use_wg_path = requested > 32;

    if (use_wg_path) {
        stedc_merge_fused_wg<B, T>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
        return;
    }

    if (requested <= 4) {
        stedc_merge_fused_cta_impl<B, T, 4>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    } else if (requested <= 8) {
        stedc_merge_fused_cta_impl<B, T, 8>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    } else if (requested <= 16) {
        stedc_merge_fused_cta_impl<B, T, 16>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    } else {
        stedc_merge_fused_cta_impl<B, T, 32>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    }
}

#define STEDC_MERGE_FUSED_CTA_INSTANTIATE(back, fp) \
    template void stedc_merge_fused_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<BATCHLAS_UNPAREN fp>&, const Span<int32_t>&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const VectorView<BATCHLAS_UNPAREN fp>&, const StedcParams<BATCHLAS_UNPAREN fp>&, int32_t*, int64_t);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEDC_MERGE_FUSED_CTA_INSTANTIATE)

#undef STEDC_MERGE_FUSED_CTA_INSTANTIATE

} // namespace batchlas
