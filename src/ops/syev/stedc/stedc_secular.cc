// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/sort.hh>
#include <batchlas/backend_config.h>
#include <src/math-helpers.hh>
#include <src/ops/syev/stedc/stedc_secular.hh>
#include <src/ops/syev/stedc/stedc_secular_roots.hh>
#include <src/kernels/common/info_span.hh>

namespace batchlas {


template <typename T>
auto sec_eval(const sycl::group<1>& cta, const VectorView<T>& v, const VectorView<T>& d, const T& x, const int32_t i, const int32_t n, const sycl::local_accessor<T, 1>& psi_buffer) {
    auto tid = cta.get_local_linear_id();
    auto bdim = cta.get_local_range()[0];
    auto bid = cta.get_group_linear_id();

    for (int k = tid; k < n; k += bdim) { psi_buffer[k] = v(k, bid) / (d(k, bid) - x); }
    sycl::group_barrier(cta);
    auto psi1 = batchlas::portable::joint_reduce(cta, util::get_raw_ptr(psi_buffer), util::get_raw_ptr(psi_buffer) + i + 1, sycl::plus<T>());
    auto psi2 = (i + 1) < n ? batchlas::portable::joint_reduce(cta, util::get_raw_ptr(psi_buffer) + i + 1, util::get_raw_ptr(psi_buffer) + n, sycl::plus<T>()) : T(0);
    for (int k = tid; k < n; k += bdim) { psi_buffer[k] = v(k, bid) / ( (d(k, bid) - x) * (d(k, bid) - x)); }
    sycl::group_barrier(cta);
    auto psi1_prime = batchlas::portable::joint_reduce(cta, util::get_raw_ptr(psi_buffer), util::get_raw_ptr(psi_buffer) + i + 1, sycl::plus<T>());
    auto psi2_prime = (i + 1) < n ? batchlas::portable::joint_reduce(cta, util::get_raw_ptr(psi_buffer) + i + 1, util::get_raw_ptr(psi_buffer) + n, sycl::plus<T>()) : T(0);
    return std::array<T, 4>{psi1, psi1_prime, psi2, psi2_prime};
}


template <typename T>
Event secular_solver(Queue& ctx, const VectorView<T>& d, const VectorView<T>& v, const MatrixView<T, MatrixFormat::Dense>& Qprime, const VectorView<T>& lambdas, const Span<int32_t>& n_reduced, const Span<T> rho, const T& tol_factor, int32_t* info, int64_t info_nodes_per_item) {
    //Solve the secular equation for each row in d and v
    auto N_max = d.size();
    const T safe_min = std::numeric_limits<T>::min();
    const T safe_denorm = std::numeric_limits<T>::denorm_min() > T(0) ? std::numeric_limits<T>::denorm_min() : safe_min;
    const T log_upper_bound = static_cast<T>(std::log(static_cast<double>(std::numeric_limits<T>::max())));
    const T log_lower_bound = static_cast<T>(std::log(static_cast<double>(safe_denorm)));
    // A local of the submit lambda so the kernel's `[=]` copies a pointer;
    // nullptr when status was not requested, which makes info_report a no-op.
    int32_t* const info_dev = info;
    const int64_t nodes_per_item = info_nodes_per_item;
    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto shared_mem = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto vhat = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto vsign = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto log_numerators = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto log_denominators = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        batchlas::impl::parallel_for(h, sycl::nd_range<1>(d.batch_size()*128, 128), [=](sycl::nd_item<1> item) {
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

            auto v_norm2 = batchlas::portable::joint_reduce(cta, v.batch_item(bid).data_ptr(), v.batch_item(bid).data_ptr() + n, sycl::plus<T>());

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
                    // Only the tolerance arm is convergence; budget exhaustion is not. The
                    // 100 here is a literal that StedcParams::max_sec_iter does not reach.
                    // evidence: docs/perf/stedc.md#stedc-open-debts
                    const bool tol_met = std::abs(eta) <= stop_threshold;
                    if (tol_met || iter >= 100) {
                        lam = new_lam;
                        if (!tol_met) {
                            detail::info_report(info_dev,
                                                detail::info_item(static_cast<int64_t>(bid), nodes_per_item),
                                                1);
                        }
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
                auto log_den2 = i > 0 ? batchlas::portable::joint_reduce(cta, util::get_raw_ptr(shared_mem), util::get_raw_ptr(shared_mem) + i, sycl::plus<T>()) : T(0);

                for (int k = tid; k < n - i - 1; k += bdim) {
                    auto diff = sycl::fabs(d(k + i + 1, bid) - d(i, bid));
                    diff = sycl::fmax(diff, safe_denorm);
                    shared_mem[k] = sycl::log(diff);
                }
                sycl::group_barrier(cta);
                auto log_den1 = (i + 1) < n ? batchlas::portable::joint_reduce(cta, util::get_raw_ptr(shared_mem), util::get_raw_ptr(shared_mem) + (n - i - 1), sycl::plus<T>()) : T(0);

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
                auto norm = detail::nrm2<T>(cta, VectorView<T>(Qview.data() + i * Qview.ld(), n, Qview.batch_size(), 1, Qview.stride()));

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

#ifdef SYCL_EXTERNAL
// Explicit instantiations to ensure device symbols are emitted for SYCL_EXTERNAL calls

template float sec_solve_ext_roc<float>(const int32_t dd,
                                        const VectorView<float>& D,
                                        const VectorView<float>& z,
                                        const float p,
                                        bool& converged_out);
template double sec_solve_ext_roc<double>(const int32_t dd,
                                          const VectorView<double>& D,
                                          const VectorView<double>& z,
                                          const double p,
                                          bool& converged_out);

template float sec_solve_roc<float>(int32_t dd,
                                    const VectorView<float>& d,
                                    const VectorView<float>& z,
                                    const float& rho,
                                    const int32_t k,
                                    bool& converged_out);
template double sec_solve_roc<double>(int32_t dd,
                                      const VectorView<double>& d,
                                      const VectorView<double>& z,
                                      const double& rho,
                                      const int32_t k,
                                      bool& converged_out);

#endif

template Event secular_solver<float>(Queue& ctx,
                                     const VectorView<float>& d,
                                     const VectorView<float>& v,
                                     const MatrixView<float, MatrixFormat::Dense>& Qprime,
                                     const VectorView<float>& lambdas,
                                     const Span<int32_t>& n_reduced,
                                     const Span<float> rho,
                                     const float& tol_factor,
                                     int32_t* info,
                                     int64_t info_nodes_per_item);

template Event secular_solver<double>(Queue& ctx,
                                      const VectorView<double>& d,
                                      const VectorView<double>& v,
                                      const MatrixView<double, MatrixFormat::Dense>& Qprime,
                                      const VectorView<double>& lambdas,
                                      const Span<int32_t>& n_reduced,
                                      const Span<double> rho,
                                      const double& tol_factor,
                                      int32_t* info,
                                      int64_t info_nodes_per_item);

} // namespace batchlas
