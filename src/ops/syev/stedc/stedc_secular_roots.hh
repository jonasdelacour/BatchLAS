#pragma once
// Per-root secular equation solvers (device functions). Included by stedc_secular.cc, which
// emits them as SYCL_EXTERNAL symbols, and by every caller when the implementation has no
// SYCL_EXTERNAL (AdaptiveCpp multipass, e.g. --acpp-targets=cuda:sm_XX): each TU then inlines
// its own copy. evidence: docs/design/sycl-implementations.md#sycl-impl-cuda-aot

#include <batchlas/blas/matrix.hh>
#include <src/math-helpers.hh>

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>

#ifdef SYCL_EXTERNAL
#define BATCHLAS_SECULAR_LINKAGE SYCL_EXTERNAL
#else
#define BATCHLAS_SECULAR_LINKAGE inline
#endif

namespace batchlas {

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
BATCHLAS_SECULAR_LINKAGE T sec_solve_ext_roc(const int32_t dd,
                                  const VectorView<T>& D,
                                  const VectorView<T>& z,
                                  const T p,
                                  bool& converged_out)
{
    bool converged = false;
    T lower_bound, upper_bound, quadratic_a, quadratic_b, quadratic_c, root;
    T shift_from_origin, step;
    T last_pole, prev_pole, shifted_last_pole, shifted_prev_pole;
    const int32_t last_index = dd - 1;
    const int32_t prev_index = dd - 2;

    // dd == 1 has no second pole: prev_index would be -1 and D(prev_index)
    // would read out of bounds. With a single pole the secular equation
    // 1/p + z0^2/(D0 - x) = 0 solves exactly as x = D0 + p*z0^2. D is updated
    // in place to the denominator D0 - x = -p*z0^2, matching what the general
    // path leaves behind via apply_shift_to_poles.
    if(dd <= 1)
    {
        const T z0 = z(0);
        const T tau = p * z0 * z0;
        const T eigenvalue = D(0) + tau;
        D(0) = -tau;
        return eigenvalue;
    }

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
    // Was `(void)converged;` -- the flag was computed by both exit paths and then
    // explicitly discarded. It is now the caller's to report.
    converged_out = converged;
    return root;
}

template <typename T>
BATCHLAS_SECULAR_LINKAGE T sec_solve_roc(int32_t dd, const VectorView<T>& d, const VectorView<T>& z, const T& rho, const int32_t k, bool& converged_out){
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
    // Report, do not assert: a device assert is compiled out in the release build.
    converged_out = converged;
    return root; // return the computed root (k^{th} eigenvalue)
}

} // namespace batchlas
