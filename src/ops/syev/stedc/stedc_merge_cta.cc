// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <src/util/template-instantiations.hh>

#include <batchlas/util/group-invoke.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/laed2.hh>
#include <batchlas/device/lapack/laed4.hh>
#include <batchlas/device/lapack/laed3.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>

#include <src/math-helpers.hh>
#include <src/ops/syev/stedc/stedc_merge_kernels.hh>
#include <src/kernels/common/info_span.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace batchlas {

template <Backend B, typename Real, int32_t P>
class StedcFusedCtaMerge;

template <Backend B, typename Real>
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
template <typename Real, typename Ex>
inline void load_local_problem_vectors(const Ex& ex,
                                       const VectorView<Real>& d_prob_global,
                                       const VectorView<Real>& z_prob_global,
                                       const VectorView<Real>& d_prob,
                                       const VectorView<Real>& z_prob) {
    device::copy(ex, d_prob_global, d_prob);
    device::copy(ex, z_prob_global, z_prob);
    device::sync<device::Scope::Group>(ex, device::Mem::Global);
}

template <Backend B, typename Real, int32_t P>
void stedc_merge_fused_cta_impl(Queue& ctx,
                                const VectorView<Real>& eigenvalues,
                                const VectorView<Real>& v,
                                const Span<Real>& rho,
                                const Span<int32_t>& n_reduced,
                                const MatrixView<Real, MatrixFormat::Dense>& Qprime,
                                const VectorView<Real>& temp_lambdas,
                                const StedcParams<Real>& params,
                                int32_t* info,
                                int64_t info_nodes_per_item) {
    const auto batch_size = eigenvalues.batch_size();
    const int32_t nloc = static_cast<int32_t>(Qprime.rows());
    const int32_t sg_size = 32;

    const auto dev = ctx->get_device();
    const int32_t base_wg_size = std::lcm<int32_t>(P, sg_size);
    const int32_t kernel_max_wg = kernel_max_work_group_size<StedcFusedCtaMerge<B, Real, P>>(ctx, dev);
    const int32_t wg_size = choose_wg_size(dev, base_wg_size, params.secular_cta_wg_size_multiplier,
                                           kernel_max_wg);

    const bool do_rescale = params.enable_rescale;
    // Locals so the kernel's `[=]` copies a pointer and a scalar; nullptr makes
    // info_report a no-op. See src/kernels/common/info_span.hh.
    int32_t* const info_dev = info;
    const int64_t nodes_per_item = info_nodes_per_item;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto d_local = sycl::local_accessor<Real, 1>(sycl::range<1>(nloc), h);
        auto z_local = sycl::local_accessor<Real, 1>(sycl::range<1>(nloc), h);

        batchlas::impl::parallel_for<StedcFusedCtaMerge<B, Real, P>>(h,
            sycl::nd_range<1>(batch_size * wg_size, wg_size),
            [=](sycl::nd_item<1> item) BATCHLAS_REQD_SG_SIZE(sg_size) {
                const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
                const device::RtGroup<device::Scope::Group, sg_size> wex{item};
                // One root per partition of P lanes; the solver's reductions are shuffles.
                const device::SgpPartition<P> pex{item.get_sub_group()};
                const int32_t sg_id = static_cast<int32_t>(item.get_sub_group().get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(pex.part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(pex.part.get_group_linear_id());
                const int32_t parts_per_wg = wg_size / P;

                const int32_t dd = n_reduced[bid];
                if (dd <= 0) {
                    return;
                }

                auto Q_bid = Qview.batch_item(bid);
                const auto Q = device::mat(Q_bid);
                const auto d_prob = VectorView<Real>(util::get_raw_ptr(d_local), dd);
                const auto z_prob = VectorView<Real>(util::get_raw_ptr(z_local), dd);
                load_local_problem_vectors(wex, eigenvalues.batch_item(bid), v.batch_item(bid), d_prob, z_prob);

                const Real abs_2rho = sycl::fabs(Real(2) * rho[bid]);

                // Named before the call: the kernel's captures (its parameter layout) keep the
                // order of first use they had with this body inline (params, temp_lambdas, ...).
                const int32_t max_iter = static_cast<int32_t>(params.max_sec_iter);
                const auto on_root = [&](int32_t root_ix, Real origin, Real tau, bool converged) {
                    temp_lambdas(root_ix, bid) = origin + tau;
                    // This is the one merge arm that honours params.max_sec_iter, so it is also
                    // the one a forced-non-convergence test can drive from the public StedcParams.
                    if (!converged) {
                        detail::info_report(info_dev, detail::info_item(static_cast<int64_t>(bid), nodes_per_item), 1);
                    }
                };
                // One root per partition, then the rescale and the vectors with one index per
                // partition: laed3 (no work-group reduction per index).
                device::laed3<Real>(wex, pex, part_id, parts_per_wg, dd, d_prob, z_prob, abs_2rho, max_iter, do_rescale, Q, Q_bid, v, bid, on_root);
            });
    });
}

template <Backend B, typename Real>
void stedc_merge_fused_wg(Queue& ctx,
                          const VectorView<Real>& eigenvalues,
                          const VectorView<Real>& v,
                          const Span<Real>& rho,
                          const Span<int32_t>& n_reduced,
                          const MatrixView<Real, MatrixFormat::Dense>& Qprime,
                          const VectorView<Real>& temp_lambdas,
                          const StedcParams<Real>& params,
                          int32_t* info,
                          int64_t info_nodes_per_item) {
    const int32_t nloc = static_cast<int32_t>(Qprime.rows());
    const auto batch_size = eigenvalues.batch_size();
    const auto dev = ctx->get_device();

    const int32_t base_wg_size = 32;  // the CTA kernels' lane count; wider requests come here
    const int32_t kernel_max_wg = kernel_max_work_group_size<StedcFusedWgMerge<B, Real>>(ctx, dev);
    const int32_t wg_size = choose_wg_size(dev, base_wg_size, params.secular_cta_wg_size_multiplier,
                                           kernel_max_wg);

    const bool do_rescale = params.enable_rescale;
    int32_t* const info_dev = info;
    const int64_t nodes_per_item = info_nodes_per_item;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto d_local = sycl::local_accessor<Real, 1>(sycl::range<1>(nloc), h);
        auto z_local = sycl::local_accessor<Real, 1>(sycl::range<1>(nloc), h);

        batchlas::impl::parallel_for<StedcFusedWgMerge<B, Real>>(h,
            sycl::nd_range<1>(batch_size * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
                const device::RtGroup<> ex{item};

                const int32_t dd = n_reduced[bid];
                if (dd <= 0) {
                    return;
                }

                auto Q_bid = Qview.batch_item(bid);
                const auto Q = device::mat(Q_bid);
                const auto d_prob = VectorView<Real>(util::get_raw_ptr(d_local), dd);
                const auto z_prob = VectorView<Real>(util::get_raw_ptr(z_local), dd);
                load_local_problem_vectors(ex, eigenvalues.batch_item(bid), v.batch_item(bid), d_prob, z_prob);

                const Real abs_2rho = sycl::fabs(Real(2) * rho[bid]);

                // WG variant: the reference path, one root at a time over the whole group, then
                // the rescale and the vectors sequentially over the index: laed3 with (0, 1).
                // on_root named first: the capture order of the inline body (see the CTA kernel).
                const int32_t max_iter = static_cast<int32_t>(params.max_sec_iter);
                const auto on_root = [&](int32_t root_ix, Real origin, Real tau, bool converged) {
                    temp_lambdas(root_ix, bid) = origin + tau;
                    if (!converged) {
                        detail::info_report(info_dev, detail::info_item(static_cast<int64_t>(bid), nodes_per_item), 1);
                    }
                };
                device::laed3<Real>(ex, ex, 0, 1, dd, d_prob, z_prob, abs_2rho, max_iter, do_rescale, Q, Q_bid, v, bid, on_root);
            });
    });
}

template <Backend B, typename Real>
void stedc_merge_fused_cta(Queue& ctx,
                           const VectorView<Real>& eigenvalues,
                           const VectorView<Real>& v,
                           const Span<Real>& rho,
                           const Span<int32_t>& n_reduced,
                           const MatrixView<Real, MatrixFormat::Dense>& Qprime,
                           const VectorView<Real>& temp_lambdas,
                           const StedcParams<Real>& params,
                           int32_t* info,
                           int64_t info_nodes_per_item) {
    // This TU builds without CPU targets, and the OpenCL CPU offers sub-group size 32.
    const bool has32 = ctx.device().type == DeviceType::GPU && ctx.device().supports_sub_group_size(32);

    if (!has32) {
        stedc_merge_fused<B, Real>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
        return;
    }

    const int32_t requested = std::max<int32_t>(4, params.secular_threads_per_root);
    const bool use_wg_path = requested > 32;

    if (use_wg_path) {
        stedc_merge_fused_wg<B, Real>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
        return;
    }

    if (requested <= 4) {
        stedc_merge_fused_cta_impl<B, Real, 4>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    } else if (requested <= 8) {
        stedc_merge_fused_cta_impl<B, Real, 8>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    } else if (requested <= 16) {
        stedc_merge_fused_cta_impl<B, Real, 16>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    } else {
        stedc_merge_fused_cta_impl<B, Real, 32>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    }
}

#define STEDC_MERGE_FUSED_CTA_INSTANTIATE(back, fp) \
    template void stedc_merge_fused_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<BATCHLAS_UNPAREN fp>&, const Span<int32_t>&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const VectorView<BATCHLAS_UNPAREN fp>&, const StedcParams<BATCHLAS_UNPAREN fp>&, int32_t*, int64_t);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEDC_MERGE_FUSED_CTA_INSTANTIATE)

#undef STEDC_MERGE_FUSED_CTA_INSTANTIATE

}  // namespace batchlas
