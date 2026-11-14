// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <src/util/template-instantiations.hh>

#include <src/math-helpers.hh>
#include <src/ops/syev/stedc/stedc_merge_kernels.hh>
#include <src/kernels/common/info_span.hh>

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
                       const StedcParams<T>& params,
                       int32_t* info,
                       int64_t info_nodes_per_item) {
    const auto batch_size = eigenvalues.batch_size();
    const int wg_size = params.merge_threads;
    const bool do_rescale = params.enable_rescale;
    const int32_t max_iter = static_cast<int32_t>(params.max_sec_iter);
    // Locals of the enclosing scope so the kernel's `[=]` copies a pointer and a
    // scalar; nullptr makes info_report a no-op.
    int32_t* const info_dev = info;
    const int64_t nodes_per_item = info_nodes_per_item;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();

        batchlas::impl::parallel_for<StedcFusedMerge<B, T>>(h,
            sycl::nd_range<1>(batch_size * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const auto bid = item.get_group_linear_id();
                const device::RtGroup<> ex{item};
                auto Q_bid = Qview.batch_item(bid);
                const int dd = n_reduced[bid];

                if (dd <= 0) {
                    return;
                }
                const auto Q = device::mat(Q_bid).sub(0, 0, dd, dd);
                const auto vb = device::vec(v.batch_item(bid));

                // One root per item (laed4 on Thread, Q(:, k) its scratch, then the denominators).
                // Several items may report one batch item: info_report is an atomic fetch_max.
                const auto poles = device::vec(eigenvalues.batch_item(bid)).head(dd);
                const auto zb = vb.head(dd);
                const T abs_2rho = sycl::fabs(T(2) * rho[bid]);
                device::for_each_owned(ex, dd, [&](int k) {
                    const auto root = device::laed4(device::Thread{}, poles, zb, dd, k, abs_2rho, max_iter, Q.col(k));
                    temp_lambdas(k, bid) = root.origin + root.tau;
                    if (!root.converged) {
                        detail::info_report(info_dev, detail::info_item(static_cast<int64_t>(bid), nodes_per_item), 1);
                    }
                    device::detail::denominators(device::Thread{}, Q.col(k), poles, root.origin, root.tau);
                });
                device::sync<device::Scope::Group>(ex, device::Mem::Global);

                device::detail::rescale<T>(do_rescale, ex, 0, 1, dd, Q_bid, eigenvalues.batch_item(bid), v, bid);
                device::sync<device::Scope::Group>(ex, device::Mem::Global);

                // Eigenvector j of the merge: Q(:, j) = z ./ (D - lambda_j), normalised.
                for (int eig = 0; eig < dd; ++eig) {
                    const auto q = Q.col(eig);
                    device::map(ex, q, [](T zi, T den) { return zi / den; }, vb, q);
                    const T nrm = device::nrm2<device::alg::Blue>(ex, q);
                    device::map(ex, q, [=](T qi) { return qi / nrm; }, q);
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
                          const StedcParams<T>& params,
                          int32_t* info,
                          int64_t info_nodes_per_item) {
    switch (params.merge_variant) {
    case StedcMergeVariant::Fused:
        stedc_merge_fused<B, T>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
        break;
    case StedcMergeVariant::FusedCta:
        stedc_merge_fused_cta<B, T>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
        break;
    default:
        // Baseline path is handled by the caller in stedc.cc.
        break;
    }
}

#define STEDC_MERGE_INSTANTIATE(back, fp) \
    template void stedc_merge_fused<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<BATCHLAS_UNPAREN fp>&, const Span<int32_t>&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const VectorView<BATCHLAS_UNPAREN fp>&, const StedcParams<BATCHLAS_UNPAREN fp>&, int32_t*, int64_t); \
    template void stedc_merge_dispatch<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<BATCHLAS_UNPAREN fp>&, const Span<int32_t>&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const VectorView<BATCHLAS_UNPAREN fp>&, const StedcParams<BATCHLAS_UNPAREN fp>&, int32_t*, int64_t);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEDC_MERGE_INSTANTIATE)

#undef STEDC_MERGE_INSTANTIATE

} // namespace batchlas
