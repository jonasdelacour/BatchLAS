#include <blas/extra.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/mempool.hh>
#include "../queue.hh"
namespace batchlas
{
    template <Backend B, typename T>
    Event cond_impl(Queue &ctx,
                    const MatrixView<T, MatrixFormat::Dense> &A,
                    const NormType norm_type,
                    const Span<T> conds,
                    const Span<std::byte> workspace){
                        auto pool = BumpAllocator(workspace);
                        auto Ainv = MatrixView<T, MatrixFormat::Dense>(pool.allocate<T>(ctx, A.data().size()).data(),
                                                                     A.rows(), A.cols(), A.ld(), A.stride(), A.batch_size(),
                                                                     pool.allocate<T*>(ctx, A.batch_size()).data());

                        auto inv_workspace = pool.allocate<std::byte>(ctx, inv_buffer_size<B>(ctx, A));
                        auto A_norms = pool.allocate<T>(ctx, A.batch_size());
                        auto A_inv_norms = pool.allocate<T>(ctx, A.batch_size());

                        inv<B>(ctx, A, Ainv, inv_workspace);
                        
                        norm(ctx, Ainv, norm_type, A_inv_norms);
                        norm(ctx, A, norm_type, A_norms);
                        ctx -> parallel_for(A.batch_size(), [=](size_t i) {
                            conds[i] = A_inv_norms[i] * A_norms[i];
                        });
                        return ctx.get_event();
                    }

    // Memory passed from outside
    template <Backend B, typename T, MatrixFormat MF>
    Event cond(Queue &ctx,
                const MatrixView<T, MF> &A,
                const NormType norm_type,
                const Span<T> conds,
                const Span<std::byte> workspace)
    {
        return cond_impl<B>(ctx, A, norm_type, conds, workspace);
    }

    template <Backend B, typename T, MatrixFormat MF>
    size_t cond_buffer_size(Queue &ctx,
                            const MatrixView<T, MF> &A,
                            const NormType norm_type)
    {
        return  BumpAllocator::allocation_size<std::byte>(ctx, inv_buffer_size<B>(ctx, A)) +
                BumpAllocator::allocation_size<T>(ctx, A.batch_size()) * 2 + // For norms
                BumpAllocator::allocation_size<T>(ctx, A.data().size()) + // For Ainv
                BumpAllocator::allocation_size<T*>(ctx, A.batch_size()) + // For data_ptrs
                BumpAllocator::allocation_size<T>(ctx, A.batch_size()); // For conds
    }

    // Convenience function which allocates and returns the results stored in an array.
    template <Backend B, typename T, MatrixFormat MF>
    UnifiedVector<T> cond(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type)
    {
        UnifiedVector<T> conds(A.batch_size());
        UnifiedVector<std::byte> workspace(cond_buffer_size<B>(ctx, A, norm_type));
        cond_impl<B>(ctx, A, norm_type, conds.to_span(), workspace.to_span()).wait();
        return conds;
    }

    #define COND_INSTANTIATE(back, fp, fmt) \
    template Event cond<back, fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        const NormType,\
        const Span<fp>,\
        const Span<std::byte>);\
    template UnifiedVector<fp> cond<back, fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        const NormType);\
    template size_t cond_buffer_size<back, fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        const NormType);

    #define COND_INSTANTIATE_FOR_BACKEND(back) \
        COND_INSTANTIATE(back, float, MatrixFormat::Dense) \
        COND_INSTANTIATE(back, double, MatrixFormat::Dense) 

    #if BATCHLAS_HAS_CUDA_BACKEND
        COND_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
    #endif
    #if BATCHLAS_HAS_ROCM_BACKEND
        COND_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
    #endif
    #if BATCHLAS_HAS_HOST_BACKEND
        COND_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
    #endif
    

} // namespace batchlas

