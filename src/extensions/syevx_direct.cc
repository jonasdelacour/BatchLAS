// syevx_direct: partial symmetric eigensolve by full decomposition + selection.
//
// This is the correct choice whenever the requested fraction of the spectrum is
// large enough that an iterative method cannot amortize its matvecs, and for small
// n where a subset solver cannot beat the CTA-resident full solver at all.
// See SYEVX_PLAN.md §2 (cost model) and §3.1.

#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <stdexcept>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <blas/functions/syev.hh>
#include "../util/template-instantiations.hh"

namespace batchlas {

template <Backend B, typename T, MatrixFormat MFormat>
struct SyevxDirectSelectKernel;

namespace {

// Shape of the private copy of A that syev consumes. Packed (ld == n,
// stride == n*n) regardless of the input view's own layout.
template <typename T>
inline MatrixView<T, MatrixFormat::Dense> packed_copy_view(T* data,
                                                           int64_t n,
                                                           int64_t batch_size,
                                                           T** ptr_array) {
    return MatrixView<T, MatrixFormat::Dense>(data,
                                              static_cast<int>(n),
                                              static_cast<int>(n),
                                              static_cast<int>(n),
                                              n * n,
                                              static_cast<int>(batch_size),
                                              ptr_array);
}

} // namespace

template <Backend B, typename T, MatrixFormat MFormat>
Event syevx_direct(Queue& ctx,
                   const MatrixView<T, MFormat>& A,
                   Span<typename base_type<T>::type> W,
                   size_t neigs,
                   Span<std::byte> workspace,
                   JobType jobz,
                   const MatrixView<T, MatrixFormat::Dense>& V,
                   const SyevxParams<T>& params) {
    using float_type = typename base_type<T>::type;

    if constexpr (MFormat != MatrixFormat::Dense) {
        (void)ctx; (void)A; (void)W; (void)neigs; (void)workspace;
        (void)jobz; (void)V; (void)params;
        throw std::runtime_error("syevx_direct: only dense matrices are supported");
    } else {
        const int64_t n = A.rows();
        const int64_t batch_size = A.batch_size();
        const bool want_eigenvectors = (jobz == JobType::EigenVectors);

        if (A.rows() != A.cols()) {
            throw std::runtime_error("syevx_direct: A must be square");
        }
        if (static_cast<int64_t>(neigs) > n) {
            throw std::runtime_error("syevx_direct: neigs must not exceed the matrix dimension");
        }

        auto pool = BumpAllocator(workspace);
        auto a_copy_data = pool.allocate<T>(ctx, static_cast<size_t>(n * n * batch_size));
        auto a_copy_ptrs = pool.allocate<T*>(ctx, static_cast<size_t>(batch_size));
        auto lambdas = pool.allocate<float_type>(ctx, static_cast<size_t>(n * batch_size));

        auto A_copy = packed_copy_view<T>(a_copy_data.data(), n, batch_size, a_copy_ptrs.data());

        // syev overwrites its input; syevx must leave A intact.
        MatrixView<T, MatrixFormat::Dense>::copy(ctx, A_copy, A);

        auto syev_ws = pool.allocate<std::byte>(
            ctx, syev_buffer_size<B>(ctx, A_copy, lambdas, jobz, Uplo::Lower));
        syev<B>(ctx, A_copy, lambdas, jobz, Uplo::Lower, syev_ws);

        // syev returns eigenvalues ascending. Select the requested extreme block,
        // matching the LOBPCG path's ordering: descending for find_largest.
        const bool find_largest = params.find_largest;
        const int64_t k = static_cast<int64_t>(neigs);

        const size_t wg = std::min<size_t>(256, static_cast<size_t>(std::max<int64_t>(n, 1)));
        const auto* lam_ptr = lambdas.data();
        auto* w_ptr = W.data();
        const T* src_ptr = A_copy.data_ptr();
        T* dst_ptr = want_eigenvectors ? V.data_ptr() : nullptr;
        const int64_t dst_ld = want_eigenvectors ? V.ld() : 0;
        const int64_t dst_stride = want_eigenvectors ? V.stride() : 0;

        ctx->submit([&](sycl::handler& h) {
            h.parallel_for<SyevxDirectSelectKernel<B, T, MFormat>>(
                sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch_size) * wg}, sycl::range{wg}),
                [=](sycl::nd_item<1> item) {
                    const int64_t tid = static_cast<int64_t>(item.get_local_linear_id());
                    const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                    const int64_t local_size = static_cast<int64_t>(item.get_local_range(0));

                    const auto* lam = lam_ptr + bid * n;
                    auto* w = w_ptr + bid * k;

                    for (int64_t i = tid; i < k; i += local_size) {
                        const int64_t src = find_largest ? (n - 1 - i) : i;
                        w[i] = lam[src];
                    }

                    if (dst_ptr != nullptr) {
                        const auto* vsrc = src_ptr + bid * n * n;
                        auto* vdst = dst_ptr + bid * dst_stride;
                        for (int64_t linear = tid; linear < n * k; linear += local_size) {
                            const int64_t row = linear % n;
                            const int64_t col = linear / n;
                            const int64_t src_col = find_largest ? (n - 1 - col) : col;
                            vdst[row + col * dst_ld] = vsrc[row + src_col * n];
                        }
                    }
                });
        });

        return ctx.get_event();
    }
}

template <Backend B, typename T, MatrixFormat MFormat>
size_t syevx_direct_buffer_size(Queue& ctx,
                                const MatrixView<T, MFormat>& A,
                                Span<typename base_type<T>::type> W,
                                size_t neigs,
                                JobType jobz,
                                const MatrixView<T, MatrixFormat::Dense>& V,
                                const SyevxParams<T>& params) {
    using float_type = typename base_type<T>::type;

    if constexpr (MFormat != MatrixFormat::Dense) {
        (void)ctx; (void)A; (void)W; (void)neigs; (void)jobz; (void)V; (void)params;
        return 0;
    } else {
        (void)W; (void)V; (void)params; (void)neigs;
        const int64_t n = A.rows();
        const int64_t batch_size = A.batch_size();

        size_t work_size = 0;
        work_size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(n * n * batch_size));
        work_size += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch_size));
        work_size += BumpAllocator::allocation_size<float_type>(ctx, static_cast<size_t>(n * batch_size));

        auto A_copy = packed_copy_view<T>(nullptr, n, batch_size, nullptr);
        work_size += BumpAllocator::allocation_size<std::byte>(
            ctx, syev_buffer_size<B>(ctx, A_copy, Span<float_type>(), jobz, Uplo::Lower));

        return work_size;
    }
}

#define SYEVX_DIRECT_INSTANTIATE(back, fp, fmt) \
    template Event syevx_direct<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);\
    template size_t syevx_direct_buffer_size<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);

#define SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND_TYPE(back, fp) \
    BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(SYEVX_DIRECT_INSTANTIATE, back, fp)

#define SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND(back)\
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND_TYPE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND(Backend::CUDA);
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND(Backend::ROCM);
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND(Backend::NETLIB);
#endif

#undef SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND
#undef SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND_TYPE
#undef SYEVX_DIRECT_INSTANTIATE

} // namespace batchlas
