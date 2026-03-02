#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <util/group-invoke.hh>
#include <util/sycl-local-accessor-helpers.hh>

#include "../math-helpers.hh"
#include "stedc_secular.hh"
#include "stedc_merge_kernels.hh"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace batchlas {

template <Backend B, typename T, int32_t P>
class StedcFusedCtaMerge;

template <Backend B, typename T>
class StedcFusedWgMerge;

inline int32_t device_max_sub_group_size(const Queue& ctx) {
    const auto dev = ctx->get_device();
    const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    int32_t max_sg = 1;
    for (const auto size : sg_sizes) {
        max_sg = std::max(max_sg, static_cast<int32_t>(size));
    }
    return max_sg;
}

inline bool device_has_sub_group_size(const Queue& ctx, int32_t target_size) {
    const auto dev = ctx->get_device();
    const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    for (const auto size : sg_sizes) {
        if (static_cast<int32_t>(size) == target_size) {
            return true;
        }
    }
    return false;
}

inline int32_t choose_wg_size(const sycl::device& dev,
                              int32_t base_wg_size,
                              int32_t requested_mul) {
    int32_t wg_mul = std::max<int32_t>(1, requested_mul);
    int32_t wg_size = base_wg_size * wg_mul;

    const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
    if (wg_size > max_wg_size) {
        const int32_t max_mul = std::max<int32_t>(1, max_wg_size / base_wg_size);
        wg_mul = std::min(wg_mul, max_mul);
        wg_size = base_wg_size * wg_mul;
    }
    return wg_size;
}

template <typename T>
inline void load_local_problem_vectors(const VectorView<T>& d_prob_global,
                                       const VectorView<T>& z_prob_global,
                                       const sycl::local_accessor<T, 1>& d_local,
                                       const sycl::local_accessor<T, 1>& z_local,
                                       int32_t tid,
                                       int32_t bdim,
                                       int32_t dd,
                                       const sycl::group<1>& wg) {
    for (int32_t i = tid; i < dd; i += bdim) {
        d_local[i] = d_prob_global(i);
        z_local[i] = z_prob_global(i);
    }
    sycl::group_barrier(wg);
}

template <typename T, typename Partition>
struct ShiftedEvalResult {
    T secular_value;
    T secular_derivative;
    T lower_sum;
    T lower_derivative;
    T upper_sum;
    T upper_derivative;
    T error_estimate;
};

template <typename Partition>
struct PartitionAdapter {
    Partition partition;

    inline int32_t lane_id() const {
        return static_cast<int32_t>(partition.get_local_linear_id());
    }

    inline int32_t width() const {
        return static_cast<int32_t>(partition.get_local_linear_range());
    }

    template <typename T>
    inline T reduce_sum(T value) const {
        const uint32_t w = static_cast<uint32_t>(partition.get_local_linear_range());
        for (uint32_t offset = w / 2; offset > 0; offset >>= 1) {
            value += sycl::permute_group_by_xor(partition, value, offset);
        }
        return value;
    }
};

struct WorkgroupAdapter {
    sycl::group<1> wg;
    int32_t tid;
    int32_t bdim;

    inline int32_t lane_id() const {
        return tid;
    }

    inline int32_t width() const {
        return bdim;
    }

    template <typename T>
    inline T reduce_sum(T value) const {
        return sycl::reduce_over_group(wg, value, sycl::plus<T>());
    }
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

template <typename T, typename Adapter>
inline ShiftedEvalResult<T, Adapter> evaluate_shifted_generic(
    const Adapter& adapter,
    const VectorView<T>& d_prob,
    const VectorView<T>& z_prob,
    int32_t dd,
    int32_t pole_index,
    int32_t skip,
    T origin,
    T tau,
    T rho_inv) {
    const int32_t lane = adapter.lane_id();
    const int32_t width = adapter.width();

    const int32_t lower_end = pole_index + (skip < 0 ? 1 : 0);
    const int32_t upper_start = pole_index + 1 + (skip < 0 ? 0 : skip);

    T low_s = T(0), low_d = T(0), low_err = T(0);
    T up_s = T(0), up_d = T(0), up_err = T(0);

    for (int32_t i = lane; i < lower_end; i += width) {
        const T shifted = d_prob(i) - origin - tau;
        const T w = z_prob(i);
        const T ratio = w / shifted;
        const T wr = w * ratio;
        low_s += wr;
        low_d += ratio * ratio;
        low_err += low_s;
    }
    for (int32_t i = upper_start + lane; i < dd; i += width) {
        const T shifted = d_prob(i) - origin - tau;
        const T w = z_prob(i);
        const T ratio = w / shifted;
        const T wr = w * ratio;
        up_s += wr;
        up_d += ratio * ratio;
        up_err += up_s;
    }

    low_s = adapter.template reduce_sum<T>(low_s);
    low_d = adapter.template reduce_sum<T>(low_d);
    low_err = adapter.template reduce_sum<T>(low_err);
    up_s = adapter.template reduce_sum<T>(up_s);
    up_d = adapter.template reduce_sum<T>(up_d);
    up_err = adapter.template reduce_sum<T>(up_err);

    return {rho_inv + low_s + up_s,
            low_d + up_d,
            low_s, low_d,
            up_s, up_d,
            sycl::fabs(low_err) + sycl::fabs(up_err)};
}

template <typename T, typename Adapter>
struct RocEvalState {
    ShiftedEvalResult<T, Adapter> eval;
    T err;
    T ratio_sq;
};

template <typename T, typename Adapter>
inline T compute_roc_step(ShiftedEvalResult<T, Adapter>& eval,
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

template <typename T, typename Adapter>
inline T solve_root_roc_generic(const Adapter& adapter,
                                const VectorView<T>& d_prob,
                                const VectorView<T>& z_prob,
                                int32_t dd,
                                int32_t k,
                                T rho,
                                int32_t max_iter) {
    const T rho_inv = T(1) / rho;
    const T pole_k = d_prob(k);
    const T pole_k1 = d_prob(k + 1);
    const T gap = pole_k1 - pole_k;

    const T midpoint = (pole_k + pole_k1) / T(2);
    auto eval = evaluate_shifted_generic<T>(adapter, d_prob, z_prob, dd, k, 1,
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
        RocEvalState<T, Adapter> state;
        state.eval = evaluate_shifted_generic<T>(adapter, d_prob, z_prob, dd, wi, 0,
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

    if (sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err) {
        return origin + tau;
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

    for (int32_t iter = 1; iter < max_iter; ++iter) {
        if (sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err) {
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

    return origin + tau;
}

template <typename T, typename Adapter>
inline T solve_root_ext_generic(const Adapter& adapter,
                                const VectorView<T>& d_prob,
                                const VectorView<T>& z_prob,
                                int32_t dd,
                                T rho,
                                int32_t max_iter) {
    const T rho_inv = T(1) / rho;
    const int32_t last = dd - 1;
    const int32_t prev = dd - 2;
    const T d_last = d_prob(last);
    const T d_prev = d_prob(prev);
    const T origin = d_last;

    const T mid_tau = rho / T(2);
    auto eval = evaluate_shifted_generic<T>(adapter, d_prob, z_prob, dd, prev, 1,
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

    eval = evaluate_shifted_generic<T>(adapter, d_prob, z_prob, dd, prev, -1,
                                       origin, tau, rho_inv);

    T err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative)
            - T(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;

    if (sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err) {
        return origin + tau;
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

    eval = evaluate_shifted_generic<T>(adapter, d_prob, z_prob, dd, prev, -1,
                                       origin, tau, rho_inv);
    err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative)
          - T(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;

    for (int32_t iter = 1; iter < max_iter; ++iter) {
        if (sycl::fabs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * err) {
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

        eval = evaluate_shifted_generic<T>(adapter, d_prob, z_prob, dd, prev, -1,
                                           origin, tau, rho_inv);
        err = eval.error_estimate + sycl::fabs(tau) * (eval.upper_derivative + eval.lower_derivative)
              - T(8) * (eval.upper_sum + eval.lower_sum) - eval.upper_sum + rho_inv;
    }

    return origin + tau;
}

template <typename T, typename Partition>
inline ShiftedEvalResult<T, Partition> evaluate_shifted_partition(
    const Partition& partition,
    const VectorView<T>& d_prob,
    const VectorView<T>& z_prob,
    int32_t dd,
    int32_t pole_index,
    int32_t skip,   // -1 = full (no skip), 0 = skip pole_index, 1 = skip both
    T origin,
    T tau,
    T rho_inv) {
    const PartitionAdapter<Partition> adapter{partition};
    const auto r = evaluate_shifted_generic<T>(adapter, d_prob, z_prob, dd, pole_index, skip,
                                               origin, tau, rho_inv);
    return {r.secular_value,
            r.secular_derivative,
            r.lower_sum,
            r.lower_derivative,
            r.upper_sum,
            r.upper_derivative,
            r.error_estimate};
}

template <typename T, typename Partition>
inline T solve_root_roc_partition(const Partition& partition,
                                  const VectorView<T>& d_prob,
                                  const VectorView<T>& z_prob,
                                  int32_t dd,
                                  int32_t k,
                                  T rho,
                                  int32_t max_iter) {
    const PartitionAdapter<Partition> adapter{partition};
    return solve_root_roc_generic(adapter, d_prob, z_prob, dd, k, rho, max_iter);
}

template <typename T, typename Partition>
inline T solve_root_ext_partition(const Partition& partition,
                                  const VectorView<T>& d_prob,
                                  const VectorView<T>& z_prob,
                                  int32_t dd,
                                  T rho,
                                  int32_t max_iter) {
    const PartitionAdapter<Partition> adapter{partition};
    return solve_root_ext_generic(adapter, d_prob, z_prob, dd, rho, max_iter);
}

template <typename T>
inline ShiftedEvalResult<T, sycl::group<1>> evaluate_shifted_wg(
    const sycl::group<1>& wg,
    int32_t tid,
    int32_t bdim,
    const VectorView<T>& d_prob,
    const VectorView<T>& z_prob,
    int32_t dd,
    int32_t pole_index,
    int32_t skip,
    T origin,
    T tau,
    T rho_inv) {
    const WorkgroupAdapter adapter{wg, tid, bdim};
    const auto r = evaluate_shifted_generic<T>(adapter, d_prob, z_prob, dd, pole_index, skip,
                                               origin, tau, rho_inv);
    return {r.secular_value,
            r.secular_derivative,
            r.lower_sum,
            r.lower_derivative,
            r.upper_sum,
            r.upper_derivative,
            r.error_estimate};
}

template <typename T>
inline T solve_root_roc_wg(const sycl::group<1>& wg,
                           int32_t tid,
                           int32_t bdim,
                           const VectorView<T>& d_prob,
                           const VectorView<T>& z_prob,
                           int32_t dd,
                           int32_t k,
                           T rho,
                           int32_t max_iter) {
    const WorkgroupAdapter adapter{wg, tid, bdim};
    return solve_root_roc_generic(adapter, d_prob, z_prob, dd, k, rho, max_iter);
}

template <typename T>
inline T solve_root_ext_wg(const sycl::group<1>& wg,
                           int32_t tid,
                           int32_t bdim,
                           const VectorView<T>& d_prob,
                           const VectorView<T>& z_prob,
                           int32_t dd,
                           T rho,
                           int32_t max_iter) {
    const WorkgroupAdapter adapter{wg, tid, bdim};
    return solve_root_ext_generic(adapter, d_prob, z_prob, dd, rho, max_iter);
}

template <typename T, typename QBatch>
inline void write_denominator_column(QBatch& Q_bid,
                                     const VectorView<T>& d_prob,
                                     int32_t dd,
                                     int32_t root_ix,
                                     T lambda,
                                     int32_t lane,
                                     int32_t width) {
    for (int32_t i = lane; i < dd; i += width) {
        T denom = d_prob(i) - lambda;
        const T floor = std::numeric_limits<T>::epsilon() * (sycl::fabs(d_prob(i)) + T(1));
        if (sycl::fabs(denom) < floor) {
            const T s = (denom == T(0)) ? T(1) : sycl::copysign(T(1), denom);
            denom = s * floor;
        }
        Q_bid(i, root_ix) = denom;
    }
}

template <typename T, typename QBatch, typename VView>
inline void maybe_rescale_vectors(bool do_rescale,
                                  const sycl::group<1>& wg,
                                  int32_t lane,
                                  int32_t width,
                                  int32_t dd,
                                  QBatch& Q_bid,
                                  const VectorView<T>& d_prob,
                                  const VView& v,
                                  int32_t bid) {
    if (!do_rescale) {
        return;
    }

    for (int32_t eid = 0; eid < dd; ++eid) {
        const T Di = d_prob(eid);
        T partial = T(1);
        for (int32_t j = lane; j < dd; j += width) {
            partial *= (j == eid) ? Q_bid(eid, j) : Q_bid(eid, j) / (Di - d_prob(j));
        }

        const T valf = sycl::reduce_over_group(wg, partial, sycl::multiplies<T>());
        if (lane == 0) {
            const T mag = sycl::sqrt(sycl::fabs(valf));
            const T sgn = (v(eid, bid) >= T(0)) ? T(1) : T(-1);
            v(eid, bid) = sgn * mag;
        }
        sycl::group_barrier(wg);
    }
}

template <typename T, typename QBatch, typename QView, typename VView>
inline void normalize_vectors(const sycl::group<1>& wg,
                              int32_t lane,
                              int32_t width,
                              int32_t dd,
                              QBatch& Q_bid,
                              const QView& Qview,
                              const VView& v,
                              int32_t bid) {
    for (int32_t eig = 0; eig < dd; ++eig) {
        for (int32_t i = lane; i < dd; i += width) {
            Q_bid(i, eig) = v(i, bid) / Q_bid(i, eig);
        }

        const T nrm2 = internal::nrm2<T>(wg, Qview(Slice{0, dd}, eig));
        for (int32_t i = lane; i < dd; i += width) {
            Q_bid(i, eig) /= nrm2;
        }
        sycl::group_barrier(wg);
    }
}

template <Backend B, typename T, int32_t P>
void stedc_merge_fused_cta_impl(Queue& ctx,
                                const VectorView<T>& eigenvalues,
                                const VectorView<T>& v,
                                const Span<T>& rho,
                                const Span<int32_t>& n_reduced,
                                const VectorView<T>& e,
                                int64_t m,
                                int64_t n,
                                const MatrixView<T, MatrixFormat::Dense>& Qprime,
                                const VectorView<T>& temp_lambdas,
                                const StedcParams<T>& params) {
    const auto batch_size = eigenvalues.batch_size();
    const int32_t nloc = static_cast<int32_t>(n);
    const int32_t sg_size = 32;

    const auto dev = ctx->get_device();
    const int32_t base_wg_size = std::lcm<int32_t>(P, sg_size);
    const int32_t wg_size = choose_wg_size(dev, base_wg_size, params.secular_cta_wg_size_multiplier);

    const bool do_rescale = params.enable_rescale;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto d_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);
        auto z_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);

        h.parallel_for<StedcFusedCtaMerge<B, T, P>>(
            sycl::nd_range<1>(batch_size * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
                const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
                const int32_t bdim = static_cast<int32_t>(item.get_local_range(0));
                const auto wg = item.get_group();

                const auto sg = item.get_sub_group();
                const auto partition = sycl::ext::oneapi::experimental::chunked_partition<P>(sg);
                const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(partition.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(partition.get_group_linear_id());
                const int32_t parts_per_wg = wg_size / P;

                const int32_t dd = n_reduced[bid];
                if (dd <= 0) {
                    return;
                }

                auto Q_bid = Qview.batch_item(bid);
                auto z_prob_global = v.batch_item(bid);
                auto d_prob_global = eigenvalues.batch_item(bid);
                load_local_problem_vectors(d_prob_global, z_prob_global,
                                           d_local, z_local,
                                           tid, bdim, dd, wg);

                const auto d_ptr = util::get_raw_ptr(d_local);
                const auto z_ptr = util::get_raw_ptr(z_local);
                const auto d_prob = VectorView<T>(d_ptr, dd);
                const auto z_prob = VectorView<T>(z_ptr, dd);

                const T sign = (e(m - 1, bid) >= T(0)) ? T(1) : T(-1);
                const T abs_2rho = sycl::fabs(T(2) * rho[bid]);

                for (int32_t root_ix = part_id; root_ix < dd; root_ix += parts_per_wg) {
                    T lambda;
                    if (root_ix == dd - 1) {
                        lambda = solve_root_ext_partition(partition, d_prob, z_prob, dd,
                                                          abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    } else {
                        lambda = solve_root_roc_partition(partition, d_prob, z_prob, dd, root_ix,
                                                          abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    }

                    if (lane == 0) {
                        temp_lambdas(root_ix, bid) = lambda * sign;
                    }
                    write_denominator_column(Q_bid, d_prob, dd, root_ix, lambda, lane, P);
                }

                sycl::group_barrier(wg);

                maybe_rescale_vectors(do_rescale, wg, tid, bdim, dd, Q_bid, d_prob, v, bid);
                normalize_vectors<T>(wg, tid, bdim, dd, Q_bid, Qview, v, bid);
            });
    });
}

template <Backend B, typename T>
void stedc_merge_fused_wg(Queue& ctx,
                          const VectorView<T>& eigenvalues,
                          const VectorView<T>& v,
                          const Span<T>& rho,
                          const Span<int32_t>& n_reduced,
                          const VectorView<T>& e,
                          int64_t m,
                          int64_t n,
                          const MatrixView<T, MatrixFormat::Dense>& Qprime,
                          const VectorView<T>& temp_lambdas,
                          const StedcParams<T>& params) {
    const int32_t nloc = static_cast<int32_t>(n);
    const auto batch_size = eigenvalues.batch_size();
    const auto dev = ctx->get_device();

    const int32_t max_sg = std::max<int32_t>(1, device_max_sub_group_size(ctx));
    const int32_t base_wg_size = max_sg;
    const int32_t wg_size = choose_wg_size(dev, base_wg_size, params.secular_cta_wg_size_multiplier);

    const bool do_rescale = params.enable_rescale;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto d_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);
        auto z_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);

        h.parallel_for<StedcFusedWgMerge<B, T>>(
            sycl::nd_range<1>(batch_size * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
                const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
                const int32_t bdim = static_cast<int32_t>(item.get_local_range(0));
                const auto wg = item.get_group();

                const int32_t dd = n_reduced[bid];
                if (dd <= 0) {
                    return;
                }

                auto Q_bid = Qview.batch_item(bid);
                auto z_prob_global = v.batch_item(bid);
                auto d_prob_global = eigenvalues.batch_item(bid);
                load_local_problem_vectors(d_prob_global, z_prob_global,
                                           d_local, z_local,
                                           tid, bdim, dd, wg);

                const auto d_ptr = util::get_raw_ptr(d_local);
                const auto z_ptr = util::get_raw_ptr(z_local);
                const auto d_prob = VectorView<T>(d_ptr, dd);
                const auto z_prob = VectorView<T>(z_ptr, dd);

                const T sign = (e(m - 1, bid) >= T(0)) ? T(1) : T(-1);
                const T abs_2rho = sycl::fabs(T(2) * rho[bid]);

                for (int32_t root_ix = 0; root_ix < dd; ++root_ix) {
                    T lambda;
                    if (root_ix == dd - 1) {
                        lambda = solve_root_ext_wg(wg, tid, bdim, d_prob, z_prob, dd,
                                                   abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    } else {
                        lambda = solve_root_roc_wg(wg, tid, bdim, d_prob, z_prob, dd, root_ix,
                                                   abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    }

                    if (tid == 0) {
                        temp_lambdas(root_ix, bid) = lambda * sign;
                    }
                    write_denominator_column(Q_bid, d_prob, dd, root_ix, lambda, tid, bdim);
                }

                sycl::group_barrier(wg);

                maybe_rescale_vectors(do_rescale, wg, tid, bdim, dd, Q_bid, d_prob, v, bid);
                normalize_vectors<T>(wg, tid, bdim, dd, Q_bid, Qview, v, bid);
            });
    });
}

template <Backend B, typename T>
void stedc_merge_fused_cta(Queue& ctx,
                           const VectorView<T>& eigenvalues,
                           const VectorView<T>& v,
                           const Span<T>& rho,
                           const Span<int32_t>& n_reduced,
                           const VectorView<T>& e,
                           int64_t m,
                           int64_t n,
                           const MatrixView<T, MatrixFormat::Dense>& Qprime,
                           const VectorView<T>& temp_lambdas,
                           const StedcParams<T>& params) {
    const bool has32 = device_has_sub_group_size(ctx, 32);

    if (!has32) {
        stedc_merge_fused<B, T>(ctx, eigenvalues, v, rho, n_reduced, e, m, n, Qprime, temp_lambdas, params);
        return;
    }

    const int32_t requested = std::max<int32_t>(4, params.secular_threads_per_root);
    const int32_t max_sg = device_max_sub_group_size(ctx);
    const bool use_wg_path = requested > max_sg;

    if (use_wg_path) {
        stedc_merge_fused_wg<B, T>(ctx, eigenvalues, v, rho, n_reduced, e, m, n, Qprime, temp_lambdas, params);
        return;
    }

    if (requested <= 4) {
        stedc_merge_fused_cta_impl<B, T, 4>(ctx, eigenvalues, v, rho, n_reduced, e, m, n, Qprime, temp_lambdas, params);
    } else if (requested <= 8) {
        stedc_merge_fused_cta_impl<B, T, 8>(ctx, eigenvalues, v, rho, n_reduced, e, m, n, Qprime, temp_lambdas, params);
    } else if (requested <= 16) {
        stedc_merge_fused_cta_impl<B, T, 16>(ctx, eigenvalues, v, rho, n_reduced, e, m, n, Qprime, temp_lambdas, params);
    } else {
        stedc_merge_fused_cta_impl<B, T, 32>(ctx, eigenvalues, v, rho, n_reduced, e, m, n, Qprime, temp_lambdas, params);
    }
}

#if BATCHLAS_HAS_HOST_BACKEND
template void stedc_merge_fused_cta<Backend::NETLIB, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const VectorView<float>&, int64_t, int64_t, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_fused_cta<Backend::NETLIB, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const VectorView<double>&, int64_t, int64_t, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
template void stedc_merge_fused_cta<Backend::CUDA, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const VectorView<float>&, int64_t, int64_t, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_fused_cta<Backend::CUDA, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const VectorView<double>&, int64_t, int64_t, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
#endif

} // namespace batchlas
