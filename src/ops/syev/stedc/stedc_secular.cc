// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/ops/syev/stedc/stedc_v2.hh>
#include <src/sort.hh>
#include <batchlas/backend_config.h>
#include <src/math-helpers.hh>
#include <src/ops/syev/stedc/stedc_secular.hh>
#include <src/ops/syev/stedc/stedc_secular_roots.hh>
#include <src/kernels/common/info_span.hh>

namespace batchlas {


namespace v2 = batchlas::device::v2;

// psi_1(x) = sum_{k <= i} z_k / (d_k - x), psi_2(x) = the same over k > i, and their derivatives
// (terms z_k / (d_k - x)^2), for z = v .* v. Each sum stages its terms in local memory, then
// folds the range.
template <typename T, typename Ex, typename V, typename P>
auto sec_eval(const Ex& ex, const V& v, const V& d, const T& x, const int32_t i, const int32_t n, const P& psi) {
    const auto terms = psi.head(n);
    v2::map(ex, terms, [=](T vk, T dk) { return vk / (dk - x); }, v, d);
    stedc_v2::group_sync(ex);
    auto psi1 = stedc_v2::fold(ex, terms.head(i + 1), sycl::plus<T>());
    auto psi2 = (i + 1) < n ? stedc_v2::fold(ex, terms.tail(i + 1), sycl::plus<T>()) : T(0);
    v2::map(ex, terms, [=](T vk, T dk) { return vk / ( (dk - x) * (dk - x)); }, v, d);
    stedc_v2::group_sync(ex);
    auto psi1_prime = stedc_v2::fold(ex, terms.head(i + 1), sycl::plus<T>());
    auto psi2_prime = (i + 1) < n ? stedc_v2::fold(ex, terms.tail(i + 1), sycl::plus<T>()) : T(0);
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
            const v2::RtGroup<> ex{item};
            auto n = n_reduced[bid];
            const auto vb = v2::vec(v.batch_item(bid)).head(n);
            const auto db = v2::vec(d.batch_item(bid)).head(n);
            const auto lam_out = v2::vec(lambdas.batch_item(bid)).head(n);
            const auto Q = v2::mat(Qview.batch_item(bid)).sub(0, 0, n, n);
            const auto lnum = v2::vec<v2::Mem::Local>(util::get_raw_ptr(log_numerators), n);
            const auto lden = v2::vec<v2::Mem::Local>(util::get_raw_ptr(log_denominators), n);
            const auto sgn = v2::vec<v2::Mem::Local>(util::get_raw_ptr(vsign), n);
            const auto zhat = v2::vec<v2::Mem::Local>(util::get_raw_ptr(vhat), n);
            const auto scratch = v2::vec<v2::Mem::Local>(util::get_raw_ptr(shared_mem), n);

            v2::for_each_owned(ex, n, [&](int k) {
                lnum(k) = T(0);
                lden(k) = T(0);
            });
            stedc_v2::group_sync(ex);
            v2::map(ex, sgn, [](T vk) { return (vk >= T(0)) ? T(1) : T(-1); }, vb);
            v2::map(ex, vb, [](T vk) { return vk * vk; }, vb);  // z = v .* v, in place

            auto v_norm2 = stedc_v2::fold(ex, v2::vec(v.batch_item(bid).data_ptr(), n), sycl::plus<T>());

            for (int i = 0; i < n; i++) {
                auto di = d(i, bid);
                auto di1 = i < n - 1 ? d(i + 1, bid) : d(n - 1, bid) + v_norm2;
                auto lam = i < n - 1 ? T(0.5) * (di + di1) : di1;
                auto [psi1, psi1_prime, psi2, psi2_prime] = sec_eval(ex, vb, db, lam, i, n, scratch);

                auto condition = T(1) + psi1 + psi2 > T(0);
                auto shift = condition ? di : di1;
                lam  -= shift;
                if (condition) {di1 -= di;} else {di -= di1;}
                auto iter = 0;
                while (true) {
                    auto [psi1, psi1_prime, psi2, psi2_prime] = sec_eval(ex, vb, db, lam + shift, i, n, scratch);
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

                if (ex.leader()) { [&]() BATCHLAS_DLA_BODY { lam_out(i) = lam; }(); }

                // Q(:, i) = d - lambda_i, its magnitude clamped away from zero.
                const T root = lam + shift;
                v2::map(ex, Q.col(i), [=](T dk) {
                    auto sign = dk - root >= T(0) ? T(1) : T(-1);
                    auto clamped_diff = std::max(sycl::fabs(dk - root), std::numeric_limits<T>::epsilon()*T(1e-5));
                    return sign * clamped_diff;
                }, db);

                // Lowner numerators: sum over roots of log |d_k - lambda_i|.
                v2::for_each_owned(ex, n, [&](int k) {
                    auto diff = sycl::fabs(Q(k, i));
                    diff = sycl::fmax(diff, safe_denorm);
                    auto log_term = sycl::log(diff);
                    if (i == 0) {
                        lnum(k) = log_term;
                    } else {
                        lnum(k) += log_term;
                    }
                });

                // Lowner denominator i: sum over k != i of log |d_i - d_k|, below then above i.
                v2::map(ex, scratch.head(i), [&](T dk) {
                    auto diff = sycl::fabs(db(i) - dk);
                    diff = sycl::fmax(diff, safe_denorm);
                    return sycl::log(diff);
                }, db);
                stedc_v2::group_sync(ex);
                auto log_den2 = i > 0 ? stedc_v2::fold(ex, scratch.head(i), sycl::plus<T>()) : T(0);

                v2::map(ex, scratch.head(n - i - 1), [&](T dk) {
                    auto diff = sycl::fabs(dk - db(i));
                    diff = sycl::fmax(diff, safe_denorm);
                    return sycl::log(diff);
                }, db.tail(i + 1));
                stedc_v2::group_sync(ex);
                auto log_den1 = (i + 1) < n ? stedc_v2::fold(ex, scratch.head(n - i - 1), sycl::plus<T>()) : T(0);

                if (ex.leader()) { [&]() BATCHLAS_DLA_BODY { lden(i) = log_den1 + log_den2; }(); }
            }

            stedc_v2::group_sync(ex);
            // zhat_k = sign(v_k) sqrt(numerator_k / denominator_k), from the logs, range-clamped.
            v2::map(ex, zhat, [=](T ln, T ld, T s) {
                auto exponent = T(0.5) * (ln - ld);
                exponent = sycl::fmax(log_lower_bound, sycl::fmin(log_upper_bound, exponent));
                auto magnitude = sycl::exp(exponent);
                return s * magnitude;
            }, lnum, lden, sgn);

            // Eigenvector i of the merge: Q(:, i) = zhat ./ (d - lambda_i), normalised.
            for (int i = 0; i < n; i++) {
                v2::map(ex, Q.col(i), [](T zk, T qk) { return zk / qk; }, zhat, Q.col(i));
            }
            for (int i = 0; i < n; i++) {
                const T nrm = v2::nrm2<v2::alg::Blue>(ex, Q.col(i));
                v2::map(ex, Q.col(i), [=](T qk) { return qk / nrm; }, Q.col(i));
            }

            // Roots were solved relative to their nearer pole: add it back.
            v2::for_each_owned(ex, n, [&](int k) {
                auto d_term = lam_out(k) > 0 ? db(k) : (k + 1 < n ? db(k + 1) : db(k) + v_norm2);
                lam_out(k) += d_term;
            });
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
