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
                       const MatrixView<T, MatrixFormat::Dense>& Qprime,
                       const VectorView<T>& temp_lambdas,
                       const StedcParams<T>& params) {
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
                const auto sign = (rho[bid] >= T(0)) ? T(1) : T(-1);
                const int dd = n_reduced[bid];

                if (dd <= 0) {
                    return;
                }

                // Root solve: initialize each column j of Q_bid with eigenvalues(:, bid)
                // and run the secular solver in-place on column k so that
                // `apply_shift_to_poles` updates Q_bid(:, k) directly. A previous
                // "fast path" stored the poles in a private `T d_priv[128]` array
                // and copied back to Q_bid after the solve; that path produced a
                // bimodal orthogonality distribution (float STEDC n<=64) versus the
                // baseline 3-kernel path, while the in-place variant below matches
                // baseline exactly.
                {
                    for (int k = tid; k < dd * dd; k += bdim) {
                        const int i = k % dd;
                        const int j = k / dd;
                        Q_bid(i, j) = eigenvalues(i, bid);
                    }
                    sycl::group_barrier(cta);

                    for (int k = tid; k < dd; k += bdim) {
                        auto dview = Q_bid(Slice{}, k);
                        if (k == dd - 1) {
                            temp_lambdas(k, bid) = sec_solve_ext_roc(dd, dview, v.batch_item(bid), std::abs(T(2) * rho[bid]));
                        } else {
                            temp_lambdas(k, bid) = sec_solve_roc(dd, dview, v.batch_item(bid), std::abs(T(2) * rho[bid]), k);
                        }
                    }
                }
                sycl::group_barrier(cta);

                if (do_rescale) {
                    // Löwner rescale: native T is sufficient once deflation uses the
                    // absolute 8*eps*max(|D|,|z|) tolerance. See stedc.cc.
                    for (int eid = 0; eid < dd; ++eid) {
                        const T Di = eigenvalues(eid, bid);
                        T partial = T(1);
                        for (int j = tid; j < dd; j += static_cast<int>(bdim)) {
                            const T q_elem = Q_bid(eid, j);
                            T ratio;
                            if (j == eid) {
                                ratio = q_elem;
                            } else {
                                const T denom = Di - eigenvalues(j, bid);
                                ratio = q_elem / denom;
                            }
                            partial *= ratio;
                        }

                        T valf = sycl::reduce_over_group(cta, partial, sycl::multiplies<T>());
                        if (tid == 0) {
                            T mag = std::sqrt(std::fabs(valf));
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
                          const MatrixView<T, MatrixFormat::Dense>& Qprime,
                          const VectorView<T>& temp_lambdas,
                          const StedcParams<T>& params) {
    switch (params.merge_variant) {
    case StedcMergeVariant::Fused:
        stedc_merge_fused<B, T>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params);
        break;
    case StedcMergeVariant::FusedCta:
        stedc_merge_fused_cta<B, T>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params);
        break;
    default:
        // Baseline path is handled by the caller in stedc.cc.
        break;
    }
}

#if BATCHLAS_HAS_HOST_BACKEND
template void stedc_merge_fused<Backend::NETLIB, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_fused<Backend::NETLIB, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
template void stedc_merge_dispatch<Backend::NETLIB, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_dispatch<Backend::NETLIB, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
template void stedc_merge_fused<Backend::CUDA, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_fused<Backend::CUDA, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
template void stedc_merge_dispatch<Backend::CUDA, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_dispatch<Backend::CUDA, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
template void stedc_merge_fused<Backend::ROCM, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_fused<Backend::ROCM, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
template void stedc_merge_dispatch<Backend::ROCM, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const Span<float>&, const Span<int32_t>&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const StedcParams<float>&);
template void stedc_merge_dispatch<Backend::ROCM, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const Span<double>&, const Span<int32_t>&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const StedcParams<double>&);
#endif

} // namespace batchlas
