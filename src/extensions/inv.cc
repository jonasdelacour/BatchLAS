#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <batchlas/util/mempool.hh>

#include "../util/template-instantiations.hh"

namespace batchlas {

    template <typename T>
    struct InvWorkspace {
        MatrixView<T, MatrixFormat::Dense> Acopy;
        Span<int64_t> pivots;
        Span<std::byte> getri_ws;
        Span<std::byte> getrf_ws;
    };

    // Single description of inv's workspace; see workspace_bytes() in
    // util/mempool.hh. Note the nested size queries ask about the caller's A,
    // not about Acopy: Acopy lives in the workspace and has no backing memory
    // while this runs. The two are shape-identical, which is what makes that
    // substitution sound.
    template <Backend B, typename T>
    InvWorkspace<T> inv_layout(Queue& ctx,
                               BumpAllocator& pool,
                               const MatrixView<T, MatrixFormat::Dense>& A) {
        auto data = pool.allocate<T>(ctx, A.data().size());
        auto ptrs = pool.allocate<T*>(ctx, A.batch_size());
        return {
            MatrixView<T, MatrixFormat::Dense>(data.data(),
                                               A.rows(), A.cols(), A.ld(), A.stride(), A.batch_size(),
                                               ptrs.data()),
            pool.allocate<int64_t>(ctx, A.rows() * A.batch_size()),
            pool.allocate<std::byte>(ctx, getri_buffer_size<B>(ctx, A)),
            pool.allocate<std::byte>(ctx, getrf_buffer_size<B>(ctx, A)),
        };
    }

    template <Backend B, typename T>
    Event inv(Queue& ctx,
              const MatrixView<T, MatrixFormat::Dense>& A,
              const MatrixView<T, MatrixFormat::Dense>& Ainv,
              Span<std::byte> workspace) {
        BumpAllocator pool(workspace);
        auto ws = inv_layout<B, T>(ctx, pool, A);
        MatrixView<T, MatrixFormat::Dense>::copy(ctx, ws.Acopy, A);
        getrf<B>(ctx, ws.Acopy, ws.pivots, ws.getrf_ws);
        getri<B>(ctx, ws.Acopy, Ainv, ws.pivots, ws.getri_ws);
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t inv_buffer_size(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A) {
        return workspace_bytes([&](BumpAllocator& pool) { return inv_layout<B, T>(ctx, pool, A); });
    }

    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> inv(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& A) {
        Matrix<T, MatrixFormat::Dense> Aout(A.rows(), A.cols(), A.batch_size());
        // Arena-backed rather than a local UnifiedVector: this overload returns
        // without waiting, so a local would sycl::free the workspace while the
        // kernels using it were still only enqueued. Arena memory is owned by the
        // queue and outlives the call; the next lease reuses it, which the
        // in-order queue orders behind this work.
        auto workspace = ctx.workspace(inv_buffer_size<B>(ctx, A));
        inv<B>(ctx, A, Aout.view(), workspace.span());
        return Aout;
    }

#define INV_INSTANTIATE(back, fp) \
    template Event inv<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Span<std::byte>); \
    template size_t inv_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&); \
    template Matrix<BATCHLAS_UNPAREN fp, MatrixFormat::Dense> inv<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&);

#define INV_INSTANTIATE_FOR_BACK(back)\
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(INV_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    INV_INSTANTIATE_FOR_BACK(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    INV_INSTANTIATE_FOR_BACK(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    INV_INSTANTIATE_FOR_BACK(Backend::NETLIB)
#endif

#undef INV_INSTANTIATE_FOR_BACK
#undef INV_INSTANTIATE

}
