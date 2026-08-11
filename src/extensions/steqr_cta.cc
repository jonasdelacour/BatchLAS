#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/group-invoke.hh>
#include "sg_compat.hh"
#include <batchlas/util/mempool.hh>
#include <batchlas/backend_config.h>
#include "steqr_internal.hh"
#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/kernel-trace.hh"
#include "../util/template-instantiations.hh"
#include "../sort.hh"
#include "steqr_cta_device.hh"
#include <array>
#include <numeric>

namespace batchlas {

    template <typename T, size_t P, bool ComputeVecs>
    class SteqrCTAKernel;

    template <typename T, size_t P, bool ComputeVecs>
    inline void steqr_cta_impl(Queue& ctx,
                              VectorView<T>& d,
                              VectorView<T>& e,
                              MatrixView<T, MatrixFormat::Dense>& eigvects,
                              int32_t n,
                              size_t max_sweeps,
                              T zero_threshold,
                              SteqrShiftStrategy cta_shift_strategy,
                              SteqrUpdateScheme cta_update_scheme,
                              size_t cta_wg_size_multiplier,
                              int32_t* status,
                              BumpAllocator& pool) {
        (void)pool;
        const auto batch_size = d.batch_size();
        if (n < 1 || n > static_cast<int32_t>(P) || d.size() != n || e.size() != (n - 1)) {
            throw std::runtime_error("steqr_cta_impl: invalid n or vector sizes for CTA partition.");
        }

        ctx->submit([&](sycl::handler& cgh) {
            auto Q_view = eigvects.kernel_view();
            const auto dev = ctx->get_device();
            const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();

            // CTA path assumes warp-sized sub-groups on NVIDIA.
            const int32_t sg_size = 32;

            // Baseline work-group size is LCM(P, sg_size), so we can form fixed-size partitions of size P.
            // Allow scaling it at runtime to tune the number of sub-groups per work-group.
            const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), static_cast<int32_t>(sg_size));
            int32_t wg_size_multiplier = std::max<int32_t>(int32_t(1), cta_wg_size_multiplier);
            int32_t wg_size = base_wg_size * wg_size_multiplier;

            const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
            if (wg_size > max_wg_size) {
                const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
                wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
                wg_size = base_wg_size * wg_size_multiplier;
            }

            const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
            const int32_t num_wg = (batch_size + probs_per_wg - 1) / probs_per_wg;
            const int32_t global_size = num_wg * wg_size;

            auto Q_local = sycl::local_accessor<T, 1>(
                sycl::range<1>(ComputeVecs ? (probs_per_wg * P * P) : 1), cgh);
            cgh.parallel_for<SteqrCTAKernel<T, P, ComputeVecs>>(
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                    const auto wg = it.get_group();
                    const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());

                    const auto sg = it.get_sub_group();
                    const auto partition = make_partition<P>(sg);
                    // NOTE: chunked_partition<P>(sg) partitions *within a sub-group*.
                    // If the work-group contains multiple sub-groups, partition.get_group_linear_id()
                    // repeats for each sub-group. Make part_id unique within the whole work-group.
                    const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                    const int32_t parts_per_sg = static_cast<int32_t>(partition.get_group_linear_range());
                    const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(partition.get_group_linear_id());
                    const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
                    const int32_t prob_id = static_cast<int32_t>(wg_id) * static_cast<int32_t>(probs_per_wg) + part_id;
                    if (prob_id >= static_cast<int32_t>(batch_size)) return;
                    auto d_prob = d.batch_item(prob_id);
                    auto e_prob = e.batch_item(prob_id);

                    // Compile-time selectable eigenvector accumulation (shared-memory Q).
                    const int32_t base_q = part_id * static_cast<int32_t>(P) * static_cast<int32_t>(P);
                    using QLocalAccT = decltype(Q_local);
                    QSharedCache<T, P, P, ComputeVecs, QLocalAccT> qcache(Q_local, base_q, lane, n);

                    if constexpr (ComputeVecs) {
                        auto Q_prob = Q_view.batch_item(prob_id);
                        qcache.load(Q_prob);
                    }

                    // Load D/E into registers (one element per lane).
                    T diag = (lane < n) ? d_prob(lane) : T(0);
                    T offdiag = (lane < (n - 1)) ? e_prob(lane) : T(0);

                    const bool failed = steqr_cta_solve<T, P>(partition, diag, offdiag, qcache, n,
                                                             static_cast<int32_t>(max_sweeps),
                                                             zero_threshold,
                                                             cta_shift_strategy, cta_update_scheme);
                    if (failed && lane == 0 && status) {
                        // We cannot throw from device code; the host decides how to handle it.
                        status[prob_id] = 1;
                    }

                    // Store back D/E (one element per lane).
                    if (lane < n) {
                        d_prob(lane) = diag;
                    }
                    if (lane < (n - 1)) {
                        e_prob(lane) = offdiag;
                    }

                    if constexpr (ComputeVecs) {
                        auto Q_prob = Q_view.batch_item(prob_id);
                        qcache.store(Q_prob);
                    }
                });
        });

    }

    template <Backend B, typename T>
    Event steqr_cta(Queue& ctx, const VectorView<T>& d_in, const VectorView<T>& e_in,
                    const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
                    JobType jobz, SteqrParams<T> params,
                    const MatrixView<T, MatrixFormat::Dense>& eigvects) {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr_cta");
        if (eigvects.rows() != eigvects.cols()) {
            throw std::invalid_argument("Matrix must be square for eigenvalue computation.");
        }
        if (jobz == JobType::EigenVectors && !params.back_transform) {
            eigvects.fill_identity(ctx);
        }

        const int64_t n = d_in.size();
        const int64_t batch_size = d_in.batch_size();
        auto pool = BumpAllocator(ws);

        const auto increment = params.transpose_working_vectors ? batch_size : 1;
        const auto d_stride = params.transpose_working_vectors ? 1 : n;
        const auto e_stride = params.transpose_working_vectors ? 1 : n - 1;

        auto d = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n, increment, d_stride, batch_size)),
                               n, batch_size, increment, d_stride);
        auto e = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n - 1, increment, e_stride, batch_size)),
                               n - 1, batch_size, increment, e_stride);

        auto status = pool.allocate<int32_t>(ctx, std::max<int64_t>(int64_t(1), batch_size)).data();
        ctx->memset(status, 0, sizeof(int32_t) * static_cast<size_t>(std::max<int64_t>(int64_t(1), batch_size)));

        VectorView<T>::copy(ctx, d, d_in);
        VectorView<T>::copy(ctx, e, e_in);

        auto& eigvects_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(eigvects);

        // CTA backend: choose an optimal compile-time partition size P in {4,8,16,32}.
        // Requires warp-sized sub-groups (32) on NVIDIA.
        if (n < 1 || n > 32) {
            throw std::invalid_argument("steqr_cta currently supports 1 <= n <= 32.");
        }

        const auto dev = ctx->get_device();
        bool has32 = false;
        {
            const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
            for (auto sgs : sg_sizes) {
                if (static_cast<int32_t>(sgs) == 32) {
                    has32 = true;
                    break;
                }
            }
        }

        if (!has32) {
            return steqr_wg<B, T>(ctx, d_in, e_in, eigenvalues, ws, jobz, params, eigvects);
        }

        const int32_t n_i32 = static_cast<int32_t>(n);

        auto launch = [&](auto P_tag) {
            constexpr int32_t P = decltype(P_tag)::value;
            if (jobz == JobType::EigenVectors) {
                steqr_cta_impl<T, P, true>(ctx, d, e, eigvects_mut, n_i32,
                                           params.max_sweeps, params.zero_threshold,
                                           params.cta_shift_strategy, params.cta_update_scheme, params.cta_wg_size_multiplier,
                                           status,
                                           pool);
            } else {
                steqr_cta_impl<T, P, false>(ctx, d, e, eigvects_mut, n_i32,
                                            params.max_sweeps, params.zero_threshold,
                                            params.cta_shift_strategy, params.cta_update_scheme, params.cta_wg_size_multiplier,
                                            status,
                                            pool);
            }
        };

        if (n_i32 <= 4) {
            launch(std::integral_constant<int32_t, 4>{});
        } else if (n_i32 <= 8) {
            launch(std::integral_constant<int32_t, 8>{});
        } else if (n_i32 <= 16) {
            launch(std::integral_constant<int32_t, 16>{});
        } else {
            launch(std::integral_constant<int32_t, 32>{});
        }

        // Copy back eigenvalues.
        VectorView<T>::copy(ctx, eigenvalues, d);

        // Optional fail-fast diagnostics: avoids silent non-convergence.
        // Note: checking requires synchronization, so keep it opt-in.
        if (const char* v = std::getenv("BATCHLAS_STEQR_CTA_CHECK")) {
            const bool enabled = (v[0] == '1') || (v[0] == 't') || (v[0] == 'T') || (v[0] == 'y') || (v[0] == 'Y');
            if (enabled) {
                ctx.wait();
                for (int64_t i = 0; i < batch_size; ++i) {
                    if (status[i] != 0) {
                        throw std::runtime_error("steqr_cta: failed to converge within sweep budget.");
                    }
                }
            }
        }

        if (params.sort) {
            auto ws_sort = pool.allocate<std::byte>(ctx, sort_buffer_size<T>(ctx, eigenvalues.data(), eigvects, jobz));
            sort(ctx, eigenvalues, eigvects, jobz, params.sort_order, ws_sort);
        }

        return ctx.get_event();
    }

    template <typename T>
    size_t steqr_cta_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                 const VectorView<T>& eigenvalues, JobType jobz, SteqrParams<T> params) {
        const auto n = d.size();
        const auto batch_size = d.batch_size();
        const auto d_stride = d.stride() > 0 ? d.stride() : n * d.inc();
        const auto e_stride = e.stride() > 0 ? e.stride() : (n - 1) * e.inc();
        const auto d_size = VectorView<T>::required_span_length(n, d.inc(), d_stride, batch_size);
        const auto e_size = VectorView<T>::required_span_length(n - 1, e.inc(), e_stride, batch_size);

        size_t size = BumpAllocator::allocation_size<T>(ctx, d_size)
                + BumpAllocator::allocation_size<T>(ctx, e_size);

        // steqr_cta allocates a per-problem status array (int32_t) for non-convergence tracking.
        size += BumpAllocator::allocation_size<int32_t>(ctx, std::max<int64_t>(int64_t(1), batch_size));

        size += sort_buffer_size<T>(ctx, eigenvalues.data(),
                                    MatrixView<T, MatrixFormat::Dense>(nullptr, n, n, n, n * n, batch_size), jobz);

        const auto dev = ctx->get_device();
        bool has32 = false;
        {
            const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
            for (auto sgs : sg_sizes) {
                if (static_cast<int32_t>(sgs) == 32) {
                    has32 = true;
                    break;
                }
            }
        }

        if (!has32) {
            const auto steqr_size = steqr_wg_buffer_size<T>(ctx, d, e, eigenvalues, jobz, params);
            size = std::max<size_t>(size, steqr_size);
        }
        return size;
    }

#define STEQR_CTA_INSTANTIATE(back, fp) \
    template Event steqr_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&);

#define STEQR_CTA_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(STEQR_CTA_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    STEQR_CTA_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
    STEQR_CTA_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    STEQR_CTA_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

    template size_t steqr_cta_buffer_size<float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, JobType, SteqrParams<float>);
    template size_t steqr_cta_buffer_size<double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, JobType, SteqrParams<double>);

    #undef STEQR_CTA_INSTANTIATE_FOR_BACKEND
    #undef STEQR_CTA_INSTANTIATE
}
