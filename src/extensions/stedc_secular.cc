#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/mempool.hh>
#include <util/sycl-local-accessor-helpers.hh>
#include <internal/sort.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "stedc_secular.hh"

namespace batchlas {

template <typename T>
auto givens_rotation_r(const T& a, const T& b) {
    T r = std::hypot(a, b);
    if (internal::is_numerically_zero(r)) {
        return std::array<T, 3>{T(1), T(0), T(0)};
    }
    return std::array<T, 3>{a / r, b / r, r};
}


template <typename T>
auto sec_eval(const sycl::group<1>& cta, const VectorView<T>& v, const VectorView<T>& d, const T& x, const int32_t i, const int32_t n, const sycl::local_accessor<T, 1>& psi_buffer) {
    auto tid = cta.get_local_linear_id();
    auto bdim = cta.get_local_range()[0];
    auto bid = cta.get_group_linear_id();

    for (int k = tid; k < n; k += bdim) { psi_buffer[k] = v(k, bid) / (d(k, bid) - x); }
    sycl::group_barrier(cta);
    auto psi1 = sycl::joint_reduce(cta, util::get_raw_ptr(psi_buffer), util::get_raw_ptr(psi_buffer) + i + 1, sycl::plus<T>());
    auto psi2 = (i + 1) < n ? sycl::joint_reduce(cta, util::get_raw_ptr(psi_buffer) + i + 1, util::get_raw_ptr(psi_buffer) + n, sycl::plus<T>()) : T(0);
    for (int k = tid; k < n; k += bdim) { psi_buffer[k] = v(k, bid) / ( (d(k, bid) - x) * (d(k, bid) - x)); }
    sycl::group_barrier(cta);
    auto psi1_prime = sycl::joint_reduce(cta, util::get_raw_ptr(psi_buffer), util::get_raw_ptr(psi_buffer) + i + 1, sycl::plus<T>());
    auto psi2_prime = (i + 1) < n ? sycl::joint_reduce(cta, util::get_raw_ptr(psi_buffer) + i + 1, util::get_raw_ptr(psi_buffer) + n, sycl::plus<T>()) : T(0);
    return std::array<T, 4>{psi1, psi1_prime, psi2, psi2_prime};
}

enum class RocSecularEvalMode : int32_t {
    full = 0,
    skip_k = 1,
    skip_k_and_next = 2,
};

template <typename T>
struct RocSecularEvalResult {
    T secular_value;
    T secular_derivative;
    T lower_sum;
    T lower_derivative;
    T upper_sum;
    T upper_derivative;
    T error_estimate;
};

template <typename T>
auto evaluate_roc_secular(const RocSecularEvalMode mode,
                          const int32_t pole_index,
                          const int32_t degree,
                          const VectorView<T>& poles,
                          const VectorView<T>& weights,
                          const T& rho_inverse,
                          const T& correction,
                          const bool apply_shift_to_poles) {
    int32_t lower_exclusive_end = 0;
    int32_t upper_exclusive_start = 0;

    // The evaluation modes match the original ROCm algorithm variants.
    if(mode == RocSecularEvalMode::full)
    {
        lower_exclusive_end = pole_index + 1;
        upper_exclusive_start = pole_index;
    }
    else if(mode == RocSecularEvalMode::skip_k)
    {
        if(apply_shift_to_poles)
        {
            poles(pole_index) = poles(pole_index) - correction;
        }
        lower_exclusive_end = pole_index;
        upper_exclusive_start = pole_index;
    }
    else if(mode == RocSecularEvalMode::skip_k_and_next)
    {
        if(apply_shift_to_poles)
        {
            poles(pole_index) = poles(pole_index) - correction;
            poles(pole_index + 1) = poles(pole_index + 1) - correction;
        }
        lower_exclusive_end = pole_index;
        upper_exclusive_start = pole_index + 1;
    }
    else
    {
        assert(false && "Invalid RocSecularEvalMode");
    }

    T lower_sum = T(0);
    T lower_derivative = T(0);
    T error_estimate = T(0);
    for(int i = 0; i < lower_exclusive_end; ++i)
    {
        T shifted_pole = poles(i) - correction;
        if(apply_shift_to_poles)
        {
            poles(i) = shifted_pole;
        }
        const T weight = weights(i);
        const T ratio = weight / shifted_pole;
        lower_sum += weight * ratio;
        lower_derivative += ratio * ratio;
        error_estimate += lower_sum;
    }
    error_estimate = std::abs(error_estimate);

    T upper_sum = T(0);
    T upper_derivative = T(0);
    for(int i = degree - 1; i > upper_exclusive_start; --i)
    {
        T shifted_pole = poles(i) - correction;
        if(apply_shift_to_poles)
        {
            poles(i) = shifted_pole;
        }
        const T weight = weights(i);
        const T ratio = weight / shifted_pole;
        upper_sum += weight * ratio;
        upper_derivative += ratio * ratio;
        error_estimate += upper_sum;
    }

    return RocSecularEvalResult<T>{
        rho_inverse + lower_sum + upper_sum,
        lower_derivative + upper_derivative,
        lower_sum,
        lower_derivative,
        upper_sum,
        upper_derivative,
        error_estimate,
    };
}

template <typename T>
SYCL_EXTERNAL T sec_solve_ext_roc(const int32_t dd,
                                  const VectorView<T>& D,
                                  const VectorView<T>& z,
                                  const T p)
{
    bool converged = false;
    T lower_bound, upper_bound, quadratic_a, quadratic_b, quadratic_c, root;
    T shift_from_origin, step;
    T last_pole, prev_pole, shifted_last_pole, shifted_prev_pole;
    const int32_t last_index = dd - 1;
    const int32_t prev_index = dd - 2;

    // initialize
    last_pole = D(last_index);
    prev_pole = D(prev_index);
    root = last_pole + p / 2;
    const T rho_inverse = 1 / p;

    // find bounds and initial guess
    auto eval = evaluate_roc_secular(RocSecularEvalMode::skip_k_and_next, prev_index, dd, D, z, rho_inverse, root, false);
    const T secular_without_two_poles = eval.secular_value;
    eval.lower_derivative = z(prev_index) * z(prev_index);
    eval.upper_derivative = z(last_index) * z(last_index);
    eval.secular_value = eval.secular_value + eval.lower_derivative / (prev_pole - root) - 2 * eval.upper_derivative * rho_inverse;
    if(eval.secular_value > 0)
    {
        // if the secular eq at the midpoint is positive, the root is in between
        // D[k] and the midpoint take D[k] as the origin, i.e. x = D[k] + tau with
        // tau in (0, uppb)
        lower_bound = 0;
        upper_bound = p / 2;
        shift_from_origin = last_pole - prev_pole;
        quadratic_a = -secular_without_two_poles * shift_from_origin + eval.lower_derivative + eval.upper_derivative;
        quadratic_b = eval.upper_derivative * shift_from_origin;
        step = std::sqrt(quadratic_a * quadratic_a + 4 * quadratic_b * secular_without_two_poles);
        if(quadratic_a < 0)
            shift_from_origin = 2 * quadratic_b / (step - quadratic_a);
        else
            shift_from_origin = (quadratic_a + step) / (2 * secular_without_two_poles);
    }
    else
    {
        // otherwise, the root is in between the midpoint and D[k+1]
        // take D[k+1] as the origin, i.e. x = D[k+1] + tau with tau in (lowb, 0)
        lower_bound = p / 2;
        upper_bound = p;
        step = eval.lower_derivative / (last_pole - prev_pole + p) + eval.upper_derivative / p;
        if(secular_without_two_poles <= step)
            shift_from_origin = p;
        else
        {
            shift_from_origin = last_pole - prev_pole;
            quadratic_a = -secular_without_two_poles * shift_from_origin + eval.lower_derivative + eval.upper_derivative;
            quadratic_b = eval.upper_derivative * shift_from_origin;
            step = std::sqrt(quadratic_a * quadratic_a + 4 * quadratic_b * secular_without_two_poles);
            if(quadratic_a < 0)
                shift_from_origin = 2 * quadratic_b / (step - quadratic_a);
            else
                shift_from_origin = (quadratic_a + step) / (2 * secular_without_two_poles);
        }
    }
    root = last_pole + shift_from_origin; // initial guess

    // evaluate secular eq and get input values to calculate step correction
    eval = evaluate_roc_secular(RocSecularEvalMode::full, prev_index, dd, D, z, rho_inverse, last_pole, true);
    eval = evaluate_roc_secular(RocSecularEvalMode::full, prev_index, dd, D, z, rho_inverse, shift_from_origin, true);

    // calculate tolerance er for convergence test
    eval.error_estimate += std::abs(shift_from_origin) * (eval.upper_derivative + eval.lower_derivative)
                           - 8 * (eval.upper_sum + eval.lower_sum)
                           - eval.upper_sum
                           + rho_inverse;

    // if the value of secular eq is small enough, no point to continue;
    // converged!!!
    if(std::abs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * eval.error_estimate)
        converged = true;

    // otherwise...
    else
    {
        // update bounds
        lower_bound = (eval.secular_value <= 0) ? std::max(lower_bound, shift_from_origin) : lower_bound;
        upper_bound = (eval.secular_value > 0) ? std::min(upper_bound, shift_from_origin) : upper_bound;

        // calculate first step correction with fixed weight method
        shifted_last_pole = D(last_index);
        shifted_prev_pole = D(prev_index);
        quadratic_c = std::abs(eval.secular_value - shifted_prev_pole * eval.lower_derivative - shifted_last_pole * eval.upper_derivative);
        quadratic_a = (shifted_last_pole + shifted_prev_pole) * eval.secular_value
                      - shifted_last_pole * shifted_prev_pole * (eval.lower_derivative + eval.upper_derivative);
        quadratic_b = shifted_last_pole * shifted_prev_pole * eval.secular_value;
        if(quadratic_c == 0)
        {
            step = upper_bound - shift_from_origin;
        }
        else
        {
            step = std::sqrt(std::abs(quadratic_a * quadratic_a - 4 * quadratic_b * quadratic_c));
            if(quadratic_a >= 0)
                step = (quadratic_a + step) / (2 * quadratic_c);
            else
                step = (2 * quadratic_b) / (quadratic_a - step);
        }

        // verify that the correction eta will get x closer to the root
        // i.e. eta*fx should be negative. If not the case, take a Newton step
        // instead
        if(eval.secular_value * step > 0)
            step = -eval.secular_value / (eval.lower_derivative + eval.upper_derivative);

        // now verify that applying the correction won't get the process out of
        // bounds if that is the case, bisect the interval instead
        if(shift_from_origin + step > upper_bound || shift_from_origin + step < lower_bound)
        {
            if(eval.secular_value < 0)
                step = (upper_bound - shift_from_origin) / 2;
            else
                step = (lower_bound - shift_from_origin) / 2;
        }

        // take the step
        shift_from_origin += step;
        root = last_pole + shift_from_origin;

        // evaluate secular eq and get input values to calculate step correction
        eval = evaluate_roc_secular(RocSecularEvalMode::full, prev_index, dd, D, z, rho_inverse, step, true);

        // calculate tolerance er for convergence test
        eval.error_estimate += std::abs(shift_from_origin) * (eval.upper_derivative + eval.lower_derivative)
                               - 8 * (eval.upper_sum + eval.lower_sum)
                               - eval.upper_sum
                               + rho_inverse;

        // MAIN ITERATION LOOP
        // ==============================================
        for(int i = 1; i < 50; ++i)
        {
            // if the value of secular eq is small enough, no point to continue;
            // converged!!!
            if(std::abs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * eval.error_estimate)
            {
                converged = true;
                break;
            }

            // update bounds
            lower_bound = (eval.secular_value <= 0) ? std::max(lower_bound, shift_from_origin) : lower_bound;
            upper_bound = (eval.secular_value > 0) ? std::min(upper_bound, shift_from_origin) : upper_bound;

                // calculate step correction
            shifted_last_pole = D(last_index);
            shifted_prev_pole = D(prev_index);
            quadratic_c = eval.secular_value - shifted_prev_pole * eval.lower_derivative - shifted_last_pole * eval.upper_derivative;
            quadratic_a = (shifted_last_pole + shifted_prev_pole) * eval.secular_value
                          - shifted_last_pole * shifted_prev_pole * (eval.lower_derivative + eval.upper_derivative);
            quadratic_b = shifted_last_pole * shifted_prev_pole * eval.secular_value;
            step = std::sqrt(std::abs(quadratic_a * quadratic_a - 4 * quadratic_b * quadratic_c));
            if(quadratic_a >= 0)
                step = (quadratic_a + step) / (2 * quadratic_c);
            else
                step = (2 * quadratic_b) / (quadratic_a - step);

            // verify that the correction eta will get x closer to the root
            // i.e. eta*fx should be negative. If not the case, take a Newton step
            // instead
            if(eval.secular_value * step > 0)
                step = -eval.secular_value / (eval.lower_derivative + eval.upper_derivative);

            // now verify that applying the correction won't get the process out of
            // bounds if that is the case, bisect the interval instead
            if(shift_from_origin + step > upper_bound || shift_from_origin + step < lower_bound)
            {
                if(eval.secular_value < 0)
                    step = (upper_bound - shift_from_origin) / 2;
                else
                    step = (lower_bound - shift_from_origin) / 2;
            }

            // take the step
            shift_from_origin += step;
            root = last_pole + shift_from_origin;

            // evaluate secular eq and get input values to calculate step correction
            eval = evaluate_roc_secular(RocSecularEvalMode::full, prev_index, dd, D, z, rho_inverse, step, true);

            // calculate tolerance er for convergence test
            eval.error_estimate += std::abs(shift_from_origin) * (eval.upper_derivative + eval.lower_derivative)
                                   - 8 * (eval.upper_sum + eval.lower_sum)
                                   - eval.upper_sum
                                   + rho_inverse;
        }
    }
    (void)converged;
    return root;
}

template <typename T>
SYCL_EXTERNAL T sec_solve_roc(int32_t dd, const VectorView<T>& d, const VectorView<T>& z, const T& rho, const int32_t k){
    bool converged = false;
    bool origin_at_lower_pole = false;
    bool use_fixed_weight_update = false;
    T lower_bound, upper_bound, quadratic_a, quadratic_b, quadratic_c, root;
    T shift_from_origin, step, previous_secular_value;
    T pole_k, pole_k1, shifted_pole_k, shifted_pole_k1;
    int32_t working_index;
    const int32_t k1 = k + 1;

    // initialize
    pole_k = d(k);
    pole_k1 = d(k1);
    root = (pole_k + pole_k1) / 2; // midpoint of interval
    shift_from_origin = (pole_k1 - pole_k);
    const T rho_inverse = 1 / rho;

    // find bounds and initial guess; translate origin
    auto eval = evaluate_roc_secular(RocSecularEvalMode::skip_k_and_next, k, dd, d, z, rho_inverse, root, false);
    const T secular_without_two_poles = eval.secular_value;
    eval.lower_derivative = z(k) * z(k);
    eval.upper_derivative = z(k1) * z(k1);
    eval.secular_value = eval.secular_value + 2 * (eval.upper_derivative - eval.lower_derivative) / shift_from_origin;
    if(eval.secular_value > 0)
    {
        // if the secular eq at the midpoint is positive, the root is in between
        // D(k) and the midpoint take D(k) as the origin, i.e. x = D(k) + tau with
        // tau in (0, uppb)
        lower_bound = 0;
        upper_bound = shift_from_origin / 2;
        origin_at_lower_pole = true;
        working_index = k; // origin remains the same
        quadratic_a = secular_without_two_poles * shift_from_origin + eval.lower_derivative + eval.upper_derivative;
        quadratic_b = eval.lower_derivative * shift_from_origin;
        step = std::sqrt(std::abs(quadratic_a * quadratic_a - 4 * quadratic_b * secular_without_two_poles));
        if(quadratic_a > 0)
            shift_from_origin = 2 * quadratic_b / (quadratic_a + step);
        else
            shift_from_origin = (quadratic_a - step) / (2 * secular_without_two_poles);
        root = pole_k + shift_from_origin; // initial guess
    }
    else
    {
        // otherwise, the root is in between the midpoint and D(k+1)
        // take D(k+1) as the origin, i.e. x = D(k+1) + tau with tau in (lowb, 0)
        lower_bound = -shift_from_origin / 2;
        upper_bound = 0;
        origin_at_lower_pole = false;
        working_index = k + 1; // translate the origin
        quadratic_a = secular_without_two_poles * shift_from_origin - eval.lower_derivative - eval.upper_derivative;
        quadratic_b = eval.upper_derivative * shift_from_origin;
        step = std::sqrt(std::abs(quadratic_a * quadratic_a + 4 * quadratic_b * secular_without_two_poles));
        if(quadratic_a < 0)
            shift_from_origin = 2 * quadratic_b / (quadratic_a - step);
        else
            shift_from_origin = -(quadratic_a + step) / (2 * secular_without_two_poles);
        root = pole_k1 + shift_from_origin; // initial guess
    }

    // evaluate secular eq and get input values to calculate step correction
    eval = evaluate_roc_secular(RocSecularEvalMode::full, working_index, dd, d, z, rho_inverse, (origin_at_lower_pole ? pole_k : pole_k1), true);
    eval = evaluate_roc_secular(RocSecularEvalMode::skip_k, working_index, dd, d, z, rho_inverse, shift_from_origin, true);
    quadratic_b = z(working_index);
    quadratic_a = quadratic_b / d(working_index);
    eval.secular_derivative += quadratic_a * quadratic_a;
    quadratic_b *= quadratic_a;
    eval.secular_value += quadratic_b;

    // calculate tolerance er for convergence test
    eval.error_estimate += 8 * (eval.upper_sum - eval.lower_sum)
                           + 2 * rho_inverse
                           + 3 * std::abs(quadratic_b)
                           + std::abs(shift_from_origin) * eval.secular_derivative;

    // if the value of secular eq is small enough, no point to continue;
    // converged!!!
    if(std::abs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * eval.error_estimate)
        converged = true;

    // otherwise...
    else
    {
        // update bounds
        lower_bound = (eval.secular_value <= 0) ? std::max(lower_bound, shift_from_origin) : lower_bound;
        upper_bound = (eval.secular_value > 0) ? std::min(upper_bound, shift_from_origin) : upper_bound;

        // calculate first step correction with fixed weight method
        shifted_pole_k = d(k);
        shifted_pole_k1 = d(k1);
        if(origin_at_lower_pole)
            quadratic_c = eval.secular_value - shifted_pole_k1 * eval.secular_derivative
                          - (pole_k - pole_k1) * z(k) * z(k) / shifted_pole_k / shifted_pole_k;
        else
            quadratic_c = eval.secular_value - shifted_pole_k * eval.secular_derivative
                          - (pole_k1 - pole_k) * z(k1) * z(k1) / shifted_pole_k1 / shifted_pole_k1;
        quadratic_a = (shifted_pole_k + shifted_pole_k1) * eval.secular_value - shifted_pole_k * shifted_pole_k1 * eval.secular_derivative;
        quadratic_b = shifted_pole_k * shifted_pole_k1 * eval.secular_value;
        if(quadratic_c == 0)
        {
            if(quadratic_a == 0)
            {
                if(origin_at_lower_pole)
                    quadratic_a = z(k) * z(k) + shifted_pole_k1 * shifted_pole_k1 * (eval.lower_derivative + eval.upper_derivative);
                else
                    quadratic_a = z(k1) * z(k1) + shifted_pole_k * shifted_pole_k * (eval.lower_derivative + eval.upper_derivative);
            }
            step = quadratic_b / quadratic_a;
        }
        else
        {
            step = std::sqrt(std::abs(quadratic_a * quadratic_a - 4 * quadratic_b * quadratic_c));
            if(quadratic_a <= 0)
                step = (quadratic_a - step) / (2 * quadratic_c);
            else
                step = (2 * quadratic_b) / (quadratic_a + step);
        }

        // verify that the correction eta will get x closer to the root
        // i.e. eta*fx should be negative. If not the case, take a Newton step
        // instead
        if(eval.secular_value * step >= 0)
            step = -eval.secular_value / eval.secular_derivative;

        // now verify that applying the correction won't get the process out of
        // bounds if that is the case, bisect the interval instead
        if(shift_from_origin + step > upper_bound || shift_from_origin + step < lower_bound)
        {
            if(eval.secular_value < 0)
                step = (upper_bound - shift_from_origin) / 2;
            else
                step = (lower_bound - shift_from_origin) / 2;
        }

        // take the step
        shift_from_origin += step;
        root = (origin_at_lower_pole ? pole_k : pole_k1) + shift_from_origin;

        // evaluate secular eq and get input values to calculate step correction
        previous_secular_value = eval.secular_value;
        eval = evaluate_roc_secular(RocSecularEvalMode::skip_k, working_index, dd, d, z, rho_inverse, step, true);
        quadratic_b = z(working_index);
        quadratic_a = quadratic_b / d(working_index);
        eval.secular_derivative += quadratic_a * quadratic_a;
        quadratic_b *= quadratic_a;
        eval.secular_value += quadratic_b;

        // calculate tolerance er for convergence test
        eval.error_estimate += 8 * (eval.upper_sum - eval.lower_sum)
                               + 2 * rho_inverse
                               + 3 * std::abs(quadratic_b)
                               + std::abs(shift_from_origin) * eval.secular_derivative;

        // from now on, further step corrections will be calculated either with
        // fixed weights method or with normal interpolation depending on the value
        // of boolean fixed
        quadratic_c = origin_at_lower_pole ? -1 : 1;
        use_fixed_weight_update = (quadratic_c * eval.secular_value) > (std::abs(previous_secular_value) / 10);

        // MAIN ITERATION LOOP
        // ==============================================
        for(int i = 1; i < 50; ++i)
        {
            // if the value of secular eq is small enough, no point to continue;
            // converged!!!
            if(std::abs(eval.secular_value) <= std::numeric_limits<T>::epsilon() * eval.error_estimate)
            {
                converged = true;
                break;
            }

            // update bounds
            lower_bound = (eval.secular_value <= 0) ? std::max(lower_bound, shift_from_origin) : lower_bound;
            upper_bound = (eval.secular_value > 0) ? std::min(upper_bound, shift_from_origin) : upper_bound;

            // calculate next step correction with either fixed weight method or
            // simple interpolation
            shifted_pole_k = d(k);
            shifted_pole_k1 = d(k1);
            if(use_fixed_weight_update)
            {
                if(origin_at_lower_pole)
                    quadratic_c = eval.secular_value - shifted_pole_k1 * eval.secular_derivative
                                  - (pole_k - pole_k1) * z(k) * z(k) / shifted_pole_k / shifted_pole_k;
                else
                    quadratic_c = eval.secular_value - shifted_pole_k * eval.secular_derivative
                                  - (pole_k1 - pole_k) * z(k1) * z(k1) / shifted_pole_k1 / shifted_pole_k1;
            }
            else
            {
                if(origin_at_lower_pole)
                    eval.lower_derivative += quadratic_a * quadratic_a;
                else
                    eval.upper_derivative += quadratic_a * quadratic_a;
                quadratic_c = eval.secular_value - shifted_pole_k * eval.lower_derivative - shifted_pole_k1 * eval.upper_derivative;
            }
            quadratic_a = (shifted_pole_k + shifted_pole_k1) * eval.secular_value
                          - shifted_pole_k * shifted_pole_k1 * eval.secular_derivative;
            quadratic_b = shifted_pole_k * shifted_pole_k1 * eval.secular_value;
            if(quadratic_c == 0)
            {
                if(quadratic_a == 0)
                {
                    if(use_fixed_weight_update)
                    {
                        if(origin_at_lower_pole)
                            quadratic_a = z(k) * z(k) + shifted_pole_k1 * shifted_pole_k1 * (eval.lower_derivative + eval.upper_derivative);
                        else
                            quadratic_a = z(k1) * z(k1) + shifted_pole_k * shifted_pole_k * (eval.lower_derivative + eval.upper_derivative);
                    }
                    else
                        quadratic_a = shifted_pole_k * shifted_pole_k * eval.lower_derivative
                                      + shifted_pole_k1 * shifted_pole_k1 * eval.upper_derivative;
                }
                step = quadratic_b / quadratic_a;
            }
            else
            {
                step = std::sqrt(std::abs(quadratic_a * quadratic_a - 4 * quadratic_b * quadratic_c));
                if(quadratic_a <= 0)
                    step = (quadratic_a - step) / (2 * quadratic_c);
                else
                    step = (2 * quadratic_b) / (quadratic_a + step);
            }

            // verify that the correction eta will get x closer to the root
            // i.e. eta*fx should be negative. If not the case, take a Newton step
            // instead
            if(eval.secular_value * step >= 0)
                step = -eval.secular_value / eval.secular_derivative;

            // now verify that applying the correction won't get the process out of
            // bounds if that is the case, bisect the interval instead
            if(shift_from_origin + step > upper_bound || shift_from_origin + step < lower_bound)
            {
                if(eval.secular_value < 0)
                    step = (upper_bound - shift_from_origin) / 2;
                else
                    step = (lower_bound - shift_from_origin) / 2;
            }

            // take the step
            shift_from_origin += step;
            root = (origin_at_lower_pole ? pole_k : pole_k1) + shift_from_origin;

            // evaluate secular eq and get input values to calculate step correction
            previous_secular_value = eval.secular_value;
            eval = evaluate_roc_secular(RocSecularEvalMode::skip_k, working_index, dd, d, z, rho_inverse, step, true);
            quadratic_b = z(working_index);
            quadratic_a = quadratic_b / d(working_index);
            eval.secular_derivative += quadratic_a * quadratic_a;
            quadratic_b *= quadratic_a;
            eval.secular_value += quadratic_b;

            // calculate tolerance er for convergence test
            eval.error_estimate += 8 * (eval.upper_sum - eval.lower_sum)
                                   + 2 * rho_inverse
                                   + 3 * std::abs(quadratic_b)
                                   + std::abs(shift_from_origin) * eval.secular_derivative;

            // update boolean fixed if necessary
            if(eval.secular_value * previous_secular_value > 0
               && std::abs(eval.secular_value) > std::abs(previous_secular_value) / 10)
                use_fixed_weight_update = !use_fixed_weight_update;
        }
    }
    assert(converged && "sec_solve_roc did not converge!");
    return root; // return the computed root (k^{th} eigenvalue)
}

template <typename T>
Event secular_solver(Queue& ctx, const VectorView<T>& d, const VectorView<T>& v, const MatrixView<T, MatrixFormat::Dense>& Qprime, const VectorView<T>& lambdas, const Span<int32_t>& n_reduced, const Span<T> rho, const T& tol_factor) {
    //Solve the secular equation for each row in d and v
    auto N_max = d.size();
    const T safe_min = std::numeric_limits<T>::min();
    const T safe_denorm = std::numeric_limits<T>::denorm_min() > T(0) ? std::numeric_limits<T>::denorm_min() : safe_min;
    const T log_upper_bound = static_cast<T>(std::log(static_cast<double>(std::numeric_limits<T>::max())));
    const T log_lower_bound = static_cast<T>(std::log(static_cast<double>(safe_denorm)));
    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto shared_mem = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto vhat = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto vsign = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto log_numerators = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto log_denominators = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        h.parallel_for(sycl::nd_range<1>(d.batch_size()*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            auto cta = item.get_group();
            auto n = n_reduced[bid];
            for (int k = tid; k < n; k += bdim) {
                log_numerators[k] = T(0);
                log_denominators[k] = T(0);
            }
            sycl::group_barrier(cta);
            for (int k = tid; k < n; k += bdim) { vsign[k] = (v(k, bid) >= T(0)) ? T(1) : T(-1); }
            for (int k = tid; k < n; k += bdim) { auto v_temp = v(k, bid); v(k, bid) *= v_temp; }

            auto v_norm2 = sycl::joint_reduce(cta, v.batch_item(bid).data_ptr(), v.batch_item(bid).data_ptr() + n, sycl::plus<T>());

            for (int i = 0; i < n; i++) {
                auto di = d(i, bid);
                auto di1 = i < n - 1 ? d(i + 1, bid) : d(n - 1, bid) + v_norm2;
                auto lam = i < n - 1 ? T(0.5) * (di + di1) : di1;
                auto [psi1, psi1_prime, psi2, psi2_prime] = sec_eval(cta, v, d, lam, i, n, shared_mem);

                auto condition = T(1) + psi1 + psi2 > T(0);
                auto shift = condition ? di : di1;
                lam  -= shift;
                if (condition) {di1 -= di;} else {di -= di1;}
                auto iter = 0;
                while (true) {
                    auto [psi1, psi1_prime, psi2, psi2_prime] = sec_eval(cta, v, d, lam + shift, i, n, shared_mem);
                    auto Di = condition ? -lam : di - lam;
                    auto Di1 = condition ? di1 - lam : -lam;
                    auto a = (Di + Di1) * (T(1) + psi1 + psi2) - Di * Di1 * (psi1_prime + psi2_prime);
                    auto b = Di * Di1 * (T(1) + psi1 + psi2);
                    auto c = (T(1) + psi1 + psi2) - Di * psi1_prime - Di1 * psi2_prime;
                    auto disc = a * a - T(4) * b * c;
                    disc = (disc < T(0) && disc < std::numeric_limits<T>::epsilon()) ? T(0) : disc;
                    auto s = std::sqrt(disc);
                    auto eta = (a > T(0)) ? (T(2) * b) / (a + s) : (a - s) / (T(2) * c);
                    auto new_lam = lam + eta;

                    auto stop_threshold = tol_factor * std::numeric_limits<T>::epsilon() * n * (1 + std::abs(psi1) + std::abs(psi2));
                    if (std::abs(eta) <= stop_threshold || iter >= 100) {
                        lam = new_lam;
                        break;
                    } else if (!std::isfinite(new_lam)) {
                        //Fall back to Newton's method
                        eta = -(1 + psi1 + psi2) / (psi1_prime + psi2_prime);
                        new_lam = lam + eta;
                    }

                    lam = new_lam;
                    iter++;
                }

                if(tid == 0) lambdas(i,bid) = lam;

                for (int k = tid; k < n; k += bdim) {
                    auto sign = d(k, bid) - (lam + shift) >= T(0) ? T(1) : T(-1);
                    auto clamped_diff = std::max(sycl::fabs(d(k, bid) - (lam + shift)), std::numeric_limits<T>::epsilon()*T(1e-5));
                    Qview(k, i, bid) =  sign * clamped_diff;
                }


                for (int k = tid; k < n; k += bdim) {
                    auto diff = sycl::fabs(Qview(k, i, bid));
                    diff = sycl::fmax(diff, safe_denorm);
                    auto log_term = sycl::log(diff);
                    if (i == 0) {
                        log_numerators[k] = log_term;
                    } else {
                        log_numerators[k] += log_term;
                    }
                }

                for (int k = tid; k < i; k += bdim) {
                    auto diff = sycl::fabs(d(i, bid) - d(k, bid));
                    diff = sycl::fmax(diff, safe_denorm);
                    shared_mem[k] = sycl::log(diff);
                }
                sycl::group_barrier(cta);
                auto log_den2 = i > 0 ? sycl::joint_reduce(cta, util::get_raw_ptr(shared_mem), util::get_raw_ptr(shared_mem) + i, sycl::plus<T>()) : T(0);

                for (int k = tid; k < n - i - 1; k += bdim) {
                    auto diff = sycl::fabs(d(k + i + 1, bid) - d(i, bid));
                    diff = sycl::fmax(diff, safe_denorm);
                    shared_mem[k] = sycl::log(diff);
                }
                sycl::group_barrier(cta);
                auto log_den1 = (i + 1) < n ? sycl::joint_reduce(cta, util::get_raw_ptr(shared_mem), util::get_raw_ptr(shared_mem) + (n - i - 1), sycl::plus<T>()) : T(0);

                if (tid == 0) log_denominators[i] = log_den1 + log_den2;
            }

            sycl::group_barrier(cta);
            for (int k = tid; k < n; k += bdim) {
                auto exponent = T(0.5) * (log_numerators[k] - log_denominators[k]);
                exponent = sycl::fmax(log_lower_bound, sycl::fmin(log_upper_bound, exponent));
                auto magnitude = sycl::exp(exponent);
                vhat[k] = vsign[k] * magnitude;
            }

            for (int i = 0; i < n; i++) {
                for (int k = tid; k < n; k += bdim) {
                    Qview(k, i, bid) = vhat[k] / Qview(k, i, bid);
                }
            }


            for (int i = 0; i < n; i++) {
                auto norm = internal::nrm2<T>(cta, VectorView<T>(Qview.data() + i * Qview.ld(), n, Qview.batch_size(), 1, Qview.stride()));

                for (int k = tid; k < n; k += bdim) Qview(k, i, bid) = Qview(k, i, bid) / norm;
            }

            for (int k = tid; k < n; k += bdim) {
                auto d_term = lambdas(k, bid) > 0 ? d(k, bid) : (k + 1 < n ? d(k + 1, bid) : d(k, bid) + v_norm2);
                lambdas(k, bid) += d_term;
            }
        });
    });

    return ctx.get_event();
}

// Explicit instantiations to ensure device symbols are emitted for SYCL_EXTERNAL calls

template float sec_solve_ext_roc<float>(const int32_t dd,
                                        const VectorView<float>& D,
                                        const VectorView<float>& z,
                                        const float p);
template double sec_solve_ext_roc<double>(const int32_t dd,
                                          const VectorView<double>& D,
                                          const VectorView<double>& z,
                                          const double p);

template float sec_solve_roc<float>(int32_t dd,
                                    const VectorView<float>& d,
                                    const VectorView<float>& z,
                                    const float& rho,
                                    const int32_t k);
template double sec_solve_roc<double>(int32_t dd,
                                      const VectorView<double>& d,
                                      const VectorView<double>& z,
                                      const double& rho,
                                      const int32_t k);

template Event secular_solver<float>(Queue& ctx,
                                     const VectorView<float>& d,
                                     const VectorView<float>& v,
                                     const MatrixView<float, MatrixFormat::Dense>& Qprime,
                                     const VectorView<float>& lambdas,
                                     const Span<int32_t>& n_reduced,
                                     const Span<float> rho,
                                     const float& tol_factor);

template Event secular_solver<double>(Queue& ctx,
                                      const VectorView<double>& d,
                                      const VectorView<double>& v,
                                      const MatrixView<double, MatrixFormat::Dense>& Qprime,
                                      const VectorView<double>& lambdas,
                                      const Span<int32_t>& n_reduced,
                                      const Span<double> rho,
                                      const double& tol_factor);

} // namespace batchlas
