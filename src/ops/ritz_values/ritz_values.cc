// SPDX-License-Identifier: MIT
#include <src/util/batch_mask.hh>
#include <src/linalg-impl.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/group-collectives.hh>
#include <src/queue.hh>
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <batchlas/util/kernel-heuristics.hh>

#include <src/util/template-instantiations.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/ritz_values.hh>

namespace batchlas {
    namespace dla = device;

    template <Backend B, typename T, MatrixFormat MFormat>
    struct RitzValuesKernel {};

    // ritz_vals[j] = (v_j^H A v_j) / (v_j^H v_j); contract in blas/extensions.hh.
    template <Backend B, typename T, MatrixFormat MFormat>
    Event ritz_values(Queue& ctx,
                      const MatrixView<T, MFormat>& A,
                      const MatrixView<T, MatrixFormat::Dense>& V,
                      const VectorView<typename base_type<T>::type>& ritz_vals,
                      Span<std::byte> workspace) {
        const auto mask = detail::call_mask("ritz_values", A, V, ritz_vals);
        const std::uint8_t* mk = detail::mask_ptr(mask);
        const bool masked = mk != nullptr;  // a captured null pointer can read as non-null on AdaptiveCpp Metal
        using real_t = typename base_type<T>::type;

        auto pool = BumpAllocator(workspace);
        auto batch_size = A.batch_size();
        auto n = A.rows();
        auto k = V.cols(); // Number of trial vectors

        assert(A.cols() == n && "Matrix A must be square");
        assert(V.rows() == n && "V must have same number of rows as A");
        assert(V.batch_size() == batch_size && "V must have same batch size as A");
        assert(ritz_vals.size() >= k && "ritz_vals must have at least k elements");
        assert(ritz_vals.batch_size() == batch_size && "ritz_vals must have same batch size as A");

        // Allocate workspace for A*V computation
        auto AV_mem = pool.allocate<T>(ctx, n * k * batch_size);
        auto AV = MatrixView<T, MatrixFormat::Dense>(AV_mem.data(), n, k, n, n * k, batch_size, pool.allocate<T*>(ctx, batch_size).data())
                      .with_batch_mask(mask);

        Event last_event;

        // Compute AV = A * V using spmm for sparse or gemm for dense
        if constexpr (MFormat == MatrixFormat::Dense) {
            last_event = gemm<B>(ctx, A, V, AV, GemmOptions<T>{});
        } else {
            auto spmm_ws = pool.allocate<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, V, AV, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
            last_event = spmm<B>(ctx, A, V, AV, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_ws);
        }

        // Ritz values (v_j^H A v_j) / (v_j^H v_j), one work-group per (item, column):
        // device::ritz_values_quotient on the gemm / spmm product.
        auto [global_size, local_size, use_grid_stride] = compute_batched_nd_range_sizes(
            batch_size * k * n, ctx.device(), KernelType::ELEMENTWISE, batch_size, n);

        last_event = ctx -> submit([&](sycl::handler& h) {
            h.depends_on(*last_event);
            auto AV_view = AV.kernel_view();
            auto V_view = V.kernel_view();

            batchlas::impl::parallel_for<RitzValuesKernel<B, T, MFormat>>(h,
                sycl::nd_range<1>(global_size, local_size),
                [=](sycl::nd_item<1> item) {
                    const dla::RtGroup<> ex{item};
                    // Grid-stride loop over (item, column) pairs.
                    for (size_t group_id = item.get_group(0); group_id < batch_size * k; group_id += item.get_group_range(0)) {
                        const int b = group_id / k;
                        const int j = group_id % k;
                        // Uniform per work-group (one (b, j) pair each), so the reductions stay matched.
                        if (masked && !mk[b]) continue;
                        // Column j of item b (no local lambda: SemaSYCL would take it as the kernel body).
                        const auto v = dla::vec(V_view.data() + static_cast<int64_t>(b) * V_view.stride() + j * V_view.ld(), n);
                        const auto av = dla::vec(AV_view.data() + static_cast<int64_t>(b) * AV_view.stride() + j * AV_view.ld(), n);
                        // The public device op's per-column step (a thin launcher over it).
                        const real_t q = dla::ritz_values_quotient(ex, v, av);
                        if (ex.leader()) ritz_vals(j, b) = q;
                    }
                }
            );
        });

        return last_event;
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    size_t ritz_values_buffer_size(Queue& ctx,
                                 const MatrixView<T, MFormat>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& V,
                                 const VectorView<typename base_type<T>::type>& ritz_vals) {
        const auto mask = detail::call_mask("ritz_values", A, V, ritz_vals);
        const auto batch_size = A.batch_size();
        const auto n = A.rows();
        const auto k = V.cols();
        // The run path's allocation sequence, measured: the spmm bytes go through allocate<std::byte>,
        // which rounds, so a hand sum of raw sizes is short.
        return workspace_bytes([&](BumpAllocator& pool) {
            auto av = pool.allocate<T>(ctx, n * k * batch_size);
            (void)pool.allocate<T*>(ctx, batch_size);
            if constexpr (MFormat != MatrixFormat::Dense) {
                auto AV = MatrixView<T, MatrixFormat::Dense>(av.data(), n, k, n, n * k, batch_size, nullptr).with_batch_mask(mask);
                (void)pool.allocate<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, V, AV, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
            }
            return 0;
        });
    }

    // Explicit template instantiations for common types
    #define RITZ_VALUES_INSTANTIATE(back, fp, fmt) \
        template Event ritz_values<back, BATCHLAS_UNPAREN fp, fmt>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, fmt>&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, Span<std::byte>); \
        template size_t ritz_values_buffer_size<back, BATCHLAS_UNPAREN fp, fmt>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, fmt>&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&);

    #define RITZ_VALUES_INSTANTIATE_FOR_BACKEND_TYPE(back, fp) \
        BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(RITZ_VALUES_INSTANTIATE, back, fp)

    #define RITZ_VALUES_INSTANTIATE_FOR_BACKEND(back) \
        BATCHLAS_FOR_EACH_SCALAR_TYPE_1(RITZ_VALUES_INSTANTIATE_FOR_BACKEND_TYPE, back)

    #if BATCHLAS_HAS_HOST_BACKEND
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
    #endif

    #if BATCHLAS_HAS_CUDA_BACKEND
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
    #endif

    #if BATCHLAS_HAS_ROCM_BACKEND
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
    #endif

    #if BATCHLAS_HAS_SYCL_BACKEND
        BATCHLAS_FOR_EACH_DEVICE_SCALAR_TYPE_1(RITZ_VALUES_INSTANTIATE_FOR_BACKEND_TYPE, Backend::SYCL)
    #endif

    #undef RITZ_VALUES_INSTANTIATE_FOR_BACKEND
    #undef RITZ_VALUES_INSTANTIATE_FOR_BACKEND_TYPE
    #undef RITZ_VALUES_INSTANTIATE
}
