// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <src/util/template-instantiations.hh>

#include <batchlas/util/group-invoke.hh>
#include <src/kernels/common/sg_partition/sg_partition.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>

#include <src/math-helpers.hh>
#include <src/ops/syev/stedc/stedc_secular.hh>
#include <src/ops/syev/stedc/stedc_merge_kernels.hh>
#include <src/ops/syev/stedc/loewner.hh>
#include <src/ops/syev/stedc/secular_equation.hh>
#include <src/kernels/common/info_span.hh>
#include <src/sycl/kernel_attrs.hh>

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace batchlas {

template <Backend B, typename T, int32_t P>
class StedcFusedCtaMerge;

template <Backend B, typename T>
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
            value += permute_group_by_xor(partition, value, offset);
        }
        return value;
    }

    template <typename T>
    inline T reduce_product(T value) const {
        const uint32_t w = static_cast<uint32_t>(partition.get_local_linear_range());
        for (uint32_t offset = w / 2; offset > 0; offset >>= 1) {
            value *= permute_group_by_xor(partition, value, offset);
        }
        return value;
    }

    // Force a value to be identical in every lane of the partition.
    //
    // Needed on every loop-exit test in the root solvers. Those tests read
    // reduced quantities, which are *mathematically* the same in every lane --
    // but the compiler cannot see that, so it treats the branch as divergent.
    // If the hardware then actually diverges, the lanes that left the loop stop
    // participating in the butterfly shuffles that the lanes still iterating are
    // waiting on, and the partition deadlocks. Broadcasting from lane 0 makes
    // the condition provably uniform and the branch convergent.
    template <typename T>
    inline T broadcast(T value) const {
        return select_from_group(partition, value, 0u);
    }

    // Partition-local reductions never need a work-group barrier: every lane of
    // a partition executes the same trip count, so the shuffles are convergent.
    inline void sync() const {}
};

// lambda = origin + tau with the origin pole's value kept apart, so the caller forms delta as
// (d_i - origin) - tau, exact at d_i = origin. `converged`: the loop met its stop, not max_iter.
template <typename T>
struct RocRoot {
    T origin;
    T tau;
    bool converged;
};

// One root of the secular equation, lambda = origin + tau. These six forwarders keep base's
// signatures and RocRoot: for double the *_generic pair stays out of line, so its call ABI is
// part of the register budget (folding them into the kernels moved registers).
template <typename T, typename Adapter>
inline RocRoot<T> interior_root_generic(const Adapter& adapter, const VectorView<T>& d_prob,
                                        const VectorView<T>& z_prob, int32_t dd, int32_t k, T rho, int32_t max_iter) {
    const tridiag::SecularEquation<T, Adapter> eq{adapter, d_prob, z_prob, dd, T(1) / rho};
    const tridiag::SecularRoot<T> root = tridiag::solve_interior_root(eq, k, max_iter);
    return {root.pole, root.tau, root.converged};
}

template <typename T, typename Adapter>
inline RocRoot<T> last_root_generic(const Adapter& adapter, const VectorView<T>& d_prob, const VectorView<T>& z_prob,
                                    int32_t dd, T rho, int32_t max_iter) {
    const tridiag::SecularEquation<T, Adapter> eq{adapter, d_prob, z_prob, dd, T(1) / rho};
    const tridiag::SecularRoot<T> root = tridiag::solve_last_root(eq, rho, max_iter);
    return {root.pole, root.tau, root.converged};
}

template <typename T, typename Partition>
inline RocRoot<T> interior_root_partition(const Partition& partition, const VectorView<T>& d_prob,
                                          const VectorView<T>& z_prob, int32_t dd, int32_t k, T rho, int32_t max_iter) {
    const PartitionAdapter<Partition> adapter{partition};
    return interior_root_generic(adapter, d_prob, z_prob, dd, k, rho, max_iter);
}

template <typename T, typename Partition>
inline RocRoot<T> last_root_partition(const Partition& partition, const VectorView<T>& d_prob,
                                      const VectorView<T>& z_prob, int32_t dd, T rho, int32_t max_iter) {
    const PartitionAdapter<Partition> adapter{partition};
    return last_root_generic(adapter, d_prob, z_prob, dd, rho, max_iter);
}

template <typename T>
inline RocRoot<T> interior_root_wg(const sycl::group<1>& wg, int32_t tid, int32_t bdim, const VectorView<T>& d_prob,
                                   const VectorView<T>& z_prob, int32_t dd, int32_t k, T rho, int32_t max_iter) {
    const WorkgroupAdapter adapter{wg, tid, bdim};
    return interior_root_generic(adapter, d_prob, z_prob, dd, k, rho, max_iter);
}

template <typename T>
inline RocRoot<T> last_root_wg(const sycl::group<1>& wg, int32_t tid, int32_t bdim, const VectorView<T>& d_prob,
                               const VectorView<T>& z_prob, int32_t dd, T rho, int32_t max_iter) {
    const WorkgroupAdapter adapter{wg, tid, bdim};
    return last_root_generic(adapter, d_prob, z_prob, dd, rho, max_iter);
}

// delta_i = (d_i - d_o) - tau into column j of Q, never d_i - lambda: at i = o it is -tau
// exactly. The solver keeps tau != 0, so only an exact 0 from extreme cancellation is clamped
// (to eps (|d_i| + 1)), against a NaN in the Löwner quotient.
template <typename T, typename QBatch>
inline void write_denominator_column(QBatch& Q_bid,
                                     const VectorView<T>& d_prob,
                                     int32_t dd,
                                     int32_t root_ix,
                                     T origin,
                                     T tau,
                                     int32_t lane,
                                     int32_t width) {
    for (int32_t i = lane; i < dd; i += width) {
        T denom = (d_prob(i) - origin) - tau;
        if (denom == T(0)) {
            denom = std::numeric_limits<T>::epsilon() * (sycl::fabs(d_prob(i)) + T(1));
        }
        Q_bid(i, root_ix) = denom;
    }
}

// U(:, j) = normalize(zhat ./ delta_j) for the columns this reduction group owns, same
// (first, stride) convention as maybe_rescale_vectors: column j reads v and touches only Q(:, j),
// so one column per partition needs no work-group barrier. Uniform trip count (cauchy_column).
template <typename T, typename Adapter, typename QBatch, typename VView>
inline void normalize_vectors(const Adapter& adapter,
                              int32_t first,
                              int32_t stride,
                              int32_t dd,
                              QBatch& Q_bid,
                              const VView& v,
                              int32_t bid) {
    const int32_t iters = (dd + stride - 1) / stride;
    for (int32_t it = 0; it < iters; ++it) {
        const int32_t eig = first + it * stride;
        tridiag::cauchy_column<T>(adapter, eig < dd, dd, Q_bid, eig, v, bid);
        adapter.sync();
    }
}

template <Backend B, typename T, int32_t P>
void stedc_merge_fused_cta_impl(Queue& ctx,
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
    const int32_t nloc = static_cast<int32_t>(Qprime.rows());
    const int32_t sg_size = 32;

    const auto dev = ctx->get_device();
    const int32_t base_wg_size = std::lcm<int32_t>(P, sg_size);
    const int32_t kernel_max_wg = kernel_max_work_group_size<StedcFusedCtaMerge<B, T, P>>(ctx, dev);
    const int32_t wg_size = choose_wg_size(dev, base_wg_size, params.secular_cta_wg_size_multiplier,
                                           kernel_max_wg);

    const bool do_rescale = params.enable_rescale;
    // Locals so the kernel's `[=]` copies a pointer and a scalar; nullptr makes
    // info_report a no-op. See src/kernels/common/info_span.hh.
    int32_t* const info_dev = info;
    const int64_t nodes_per_item = info_nodes_per_item;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto d_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);
        auto z_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);

        batchlas::impl::parallel_for<StedcFusedCtaMerge<B, T, P>>(h,
            sycl::nd_range<1>(batch_size * wg_size, wg_size),
            [=](sycl::nd_item<1> item) BATCHLAS_REQD_SG_SIZE(sg_size) {
                const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
                const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
                const int32_t bdim = static_cast<int32_t>(item.get_local_range(0));
                const auto wg = item.get_group();

                const auto sg = item.get_sub_group();
                const auto partition = make_partition<P>(sg);
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

                const T abs_2rho = sycl::fabs(T(2) * rho[bid]);

                for (int32_t root_ix = part_id; root_ix < dd; root_ix += parts_per_wg) {
                    RocRoot<T> root;
                    if (root_ix == dd - 1) {
                        root = last_root_partition(partition, d_prob, z_prob, dd,
                                                   abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    } else {
                        root = interior_root_partition(partition, d_prob, z_prob, dd, root_ix,
                                                       abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    }

                    if (lane == 0) {
                        temp_lambdas(root_ix, bid) = root.origin + root.tau;
                        // This is the one merge arm that honours params.max_sec_iter,
                        // so it is also the one a forced-non-convergence test can
                        // drive from the public StedcParams.
                        if (!root.converged) {
                            detail::info_report(info_dev,
                                                detail::info_item(static_cast<int64_t>(bid), nodes_per_item),
                                                1);
                        }
                    }
                    write_denominator_column(Q_bid, d_prob, dd, root_ix, root.origin, root.tau, lane, P);
                }

                sycl::group_barrier(wg);

                // Rescale and normalize are parallel over their outer index, so
                // hand one index to each partition instead of walking them
                // sequentially behind a work-group reduction per index.
                const PartitionAdapter<decltype(partition)> padapter{partition};
                maybe_rescale_vectors(do_rescale, padapter, part_id, parts_per_wg, dd,
                                      Q_bid, d_prob, v, bid);
                // Rescale reads all of Q and writes all of v; normalize
                // overwrites Q and reads all of v. One barrier separates the two
                // phases -- the loops themselves no longer need any.
                sycl::group_barrier(wg);
                normalize_vectors<T>(padapter, part_id, parts_per_wg, dd, Q_bid, v, bid);
            });
    });
}

template <Backend B, typename T>
void stedc_merge_fused_wg(Queue& ctx,
                          const VectorView<T>& eigenvalues,
                          const VectorView<T>& v,
                          const Span<T>& rho,
                          const Span<int32_t>& n_reduced,
                          const MatrixView<T, MatrixFormat::Dense>& Qprime,
                          const VectorView<T>& temp_lambdas,
                          const StedcParams<T>& params,
                          int32_t* info,
                          int64_t info_nodes_per_item) {
    const int32_t nloc = static_cast<int32_t>(Qprime.rows());
    const auto batch_size = eigenvalues.batch_size();
    const auto dev = ctx->get_device();

    const int32_t base_wg_size = 32;  // the CTA kernels' lane count; wider requests come here
    const int32_t kernel_max_wg = kernel_max_work_group_size<StedcFusedWgMerge<B, T>>(ctx, dev);
    const int32_t wg_size = choose_wg_size(dev, base_wg_size, params.secular_cta_wg_size_multiplier,
                                           kernel_max_wg);

    const bool do_rescale = params.enable_rescale;
    int32_t* const info_dev = info;
    const int64_t nodes_per_item = info_nodes_per_item;

    ctx->submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto d_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);
        auto z_local = sycl::local_accessor<T, 1>(sycl::range<1>(nloc), h);

        batchlas::impl::parallel_for<StedcFusedWgMerge<B, T>>(h,
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

                const T abs_2rho = sycl::fabs(T(2) * rho[bid]);

                for (int32_t root_ix = 0; root_ix < dd; ++root_ix) {
                    RocRoot<T> root;
                    if (root_ix == dd - 1) {
                        root = last_root_wg(wg, tid, bdim, d_prob, z_prob, dd,
                                            abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    } else {
                        root = interior_root_wg(wg, tid, bdim, d_prob, z_prob, dd, root_ix,
                                                abs_2rho, static_cast<int32_t>(params.max_sec_iter));
                    }

                    if (tid == 0) {
                        temp_lambdas(root_ix, bid) = root.origin + root.tau;
                        if (!root.converged) {
                            detail::info_report(info_dev,
                                                detail::info_item(static_cast<int64_t>(bid), nodes_per_item),
                                                1);
                        }
                    }
                    write_denominator_column(Q_bid, d_prob, dd, root_ix, root.origin, root.tau, tid, bdim);
                    // (WG variant: kept sequential as the reference path.)
                }

                sycl::group_barrier(wg);

                // WG variant keeps the original sequential-over-index walk with
                // work-group-wide reductions: (first, stride) = (0, 1).
                const WorkgroupAdapter wadapter{wg, tid, bdim};
                maybe_rescale_vectors(do_rescale, wadapter, 0, 1, dd, Q_bid, d_prob, v, bid);
                normalize_vectors<T>(wadapter, 0, 1, dd, Q_bid, v, bid);
            });
    });
}

template <Backend B, typename T>
void stedc_merge_fused_cta(Queue& ctx,
                           const VectorView<T>& eigenvalues,
                           const VectorView<T>& v,
                           const Span<T>& rho,
                           const Span<int32_t>& n_reduced,
                           const MatrixView<T, MatrixFormat::Dense>& Qprime,
                           const VectorView<T>& temp_lambdas,
                           const StedcParams<T>& params,
                           int32_t* info,
                           int64_t info_nodes_per_item) {
    const bool has32 = ctx.device().supports_sub_group_size(32);

    if (!has32) {
        stedc_merge_fused<B, T>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
        return;
    }

    const int32_t requested = std::max<int32_t>(4, params.secular_threads_per_root);
    const bool use_wg_path = requested > 32;

    if (use_wg_path) {
        stedc_merge_fused_wg<B, T>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
        return;
    }

    if (requested <= 4) {
        stedc_merge_fused_cta_impl<B, T, 4>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    } else if (requested <= 8) {
        stedc_merge_fused_cta_impl<B, T, 8>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    } else if (requested <= 16) {
        stedc_merge_fused_cta_impl<B, T, 16>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    } else {
        stedc_merge_fused_cta_impl<B, T, 32>(ctx, eigenvalues, v, rho, n_reduced, Qprime, temp_lambdas, params, info, info_nodes_per_item);
    }
}

#define STEDC_MERGE_FUSED_CTA_INSTANTIATE(back, fp) \
    template void stedc_merge_fused_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<BATCHLAS_UNPAREN fp>&, const Span<int32_t>&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const VectorView<BATCHLAS_UNPAREN fp>&, const StedcParams<BATCHLAS_UNPAREN fp>&, int32_t*, int64_t);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEDC_MERGE_FUSED_CTA_INSTANTIATE)

#undef STEDC_MERGE_FUSED_CTA_INSTANTIATE

} // namespace batchlas
