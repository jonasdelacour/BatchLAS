#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "stedc_secular.hh"
#include "stedc_merge_kernels.hh"

namespace batchlas {

template <Backend B, typename T> class StedcFusedMerge;

// Fused merge kernel that preserves baseline math exactly while collapsing
// the secular solve + rescale + matrix update into one launch.
template <Backend B, typename T>
void stedc_merge_fused(Queue& ctx,
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
    (void)n;
    const auto batch_size = eigenvalues.batch_size();
    const int wg_size = params.merge_threads;
    const bool do_rescale = params.enable_rescale;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();

        h.parallel_for<StedcFusedMerge<B, T>>(
            sycl::nd_range<1>(batch_size * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const auto bid = item.get_group_linear_id();
                const auto bdim = item.get_local_range(0);
                const auto tid = item.get_local_linear_id();
                const auto cta = item.get_group();
                auto Q_bid = Qview.batch_item(bid);
                const auto sign = (e(m - 1, bid) >= T(0)) ? T(1) : T(-1);
                const int dd = n_reduced[bid];

                if (dd <= 0) {
                    return;
                }

                // Root solve: for small dd, solve on private D vectors and write the
                // mutated solver state back to Q_bid(:,k). This preserves the exact
                // downstream math while removing the up-front dd*dd initialization pass.
                if (dd <= 128) {
                    T d_priv[128];
                    auto z_view = v.batch_item(bid);
                    const T abs_2rho = std::abs(T(2) * rho[bid]);
                    for (int k = tid; k < dd; k += bdim) {
                        for (int i = 0; i < dd; ++i) {
                            d_priv[i] = eigenvalues(i, bid);
                        }

                        auto dview = VectorView<T>(d_priv, dd);
                        if (k == dd - 1) {
                            temp_lambdas(k, bid) = sec_solve_ext_roc(dd, dview, z_view, abs_2rho) * sign;
                        } else {
                            temp_lambdas(k, bid) = sec_solve_roc(dd, dview, z_view, abs_2rho, k) * sign;
                        }

                        for (int i = 0; i < dd; ++i) {
                            Q_bid(i, k) = d_priv[i];
                        }
                    }
                } else {
                    // Fallback path for larger dd: keep baseline in-place behavior.
                    for (int k = tid; k < dd * dd; k += bdim) {
                        const int i = k % dd;
                        const int j = k / dd;
                        Q_bid(i, j) = eigenvalues(i, bid);
                    }
                    sycl::group_barrier(cta);

                    for (int k = tid; k < dd; k += bdim) {
                        auto dview = Q_bid(Slice{}, k);
                        if (k == dd - 1) {
                            temp_lambdas(k, bid) = sec_solve_ext_roc(dd, dview, v.batch_item(bid), std::abs(T(2) * rho[bid])) * sign;
                        } else {
                            temp_lambdas(k, bid) = sec_solve_roc(dd, dview, v.batch_item(bid), std::abs(T(2) * rho[bid]), k) * sign;
                        }
                    }
                }
                sycl::group_barrier(cta);

                if (do_rescale) {
                    // Baseline rescale kernel math.
                    for (int eid = 0; eid < dd; ++eid) {
                        const T Di = eigenvalues(eid, bid);
                        T partial = T(1);
                        for (int j = tid; j < dd; j += static_cast<int>(bdim)) {
                            partial *= (j == eid) ? Q_bid(eid, j) : Q_bid(eid, j) / (Di - eigenvalues(j, bid));
                        }

                        T valf = sycl::reduce_over_group(cta, partial, sycl::multiplies<T>());
                        if (tid == 0) {
                            T mag = sycl::sqrt(sycl::fabs(valf));
                            T sgn = (v(eid, bid) >= T(0)) ? T(1) : T(-1);
                            v(eid, bid) = sgn * mag;
                        }
                    }
                    sycl::group_barrier(cta);
                }

                // Baseline matrix update kernel math.
                for (int eig = 0; eig < dd; ++eig) {
                    for (int i = tid; i < dd; i += static_cast<int>(bdim)) {
                        Q_bid(i, eig) = v(i, bid) / Q_bid(i, eig);
                    }

                    auto nrm2 = internal::nrm2(cta, Qview(Slice{0, dd}, eig));
                    for (int i = tid; i < dd; i += static_cast<int>(bdim)) {
                        Q_bid(i, eig) /= nrm2;
                    }
                }
            });
    });
}

template <Backend B, typename T>
void stedc_merge_dispatch(Queue& ctx,
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
    switch (params.merge_variant) {
    case StedcMergeVariant::Fused:
        stedc_merge_fused<B, T>(ctx, eigenvalues, v, rho, n_reduced, e, m, n, Qprime, temp_lambdas, params);
        break;
    default:
        // Baseline path is handled by the caller in stedc.cc.
        break;
    }
}

#if BATCHLAS_HAS_HOST_BACKEND
template void stedc_merge_dispatch<Backend::NETLIB, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const VectorView<float>&, int64_t, int64_t, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_dispatch<Backend::NETLIB, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const VectorView<double>&, int64_t, int64_t, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
template void stedc_merge_dispatch<Backend::CUDA, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const VectorView<float>&, int64_t, int64_t, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_dispatch<Backend::CUDA, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const VectorView<double>&, int64_t, int64_t, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
#endif

} // namespace batchlas
