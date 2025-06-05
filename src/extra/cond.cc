#include <blas/extra.hh>
#include <blas/functions.hh>
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
                        auto pivots = pool.allocate<int64_t>(ctx, A.batch_size()*A.rows());
                        auto Acopy = MatrixView<T, MatrixFormat::Dense>::deep_copy(A, pool.allocate<T>(ctx, A.data().size()).data(),
                                                                                    pool.allocate<T*>(ctx, A.batch_size()).data());

                        auto Ainv = MatrixView<T, MatrixFormat::Dense>(pool.allocate<T>(ctx, A.data().size()).data(),
                                                                        A.rows(), A.cols(), A.ld(), A.stride(), A.batch_size(),
                                                                        pool.allocate<T*>(ctx, A.batch_size()).data());
                                                                        
                        auto getri_workspace = pool.allocate<std::byte>(ctx, getri_buffer_size<B>(ctx, Acopy));
                        auto getrf_workspace = pool.allocate<std::byte>(ctx, getrf_buffer_size<B>(ctx, Acopy));
                        auto A_norms = pool.allocate<T>(ctx, A.batch_size());
                        auto A_inv_norms = pool.allocate<T>(ctx, A.batch_size());
                        getrf<B>(ctx, Acopy, pivots, getrf_workspace);
                        getri<B>(ctx, Acopy, Ainv, pivots, getri_workspace);
                        
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
        return  BumpAllocator::allocation_size<std::byte>(ctx, getri_buffer_size<B>(ctx, A)) +
                BumpAllocator::allocation_size<std::byte>(ctx, getrf_buffer_size<B>(ctx, A)) +
                BumpAllocator::allocation_size<T>(ctx, A.batch_size()) * 2 + // For norms
                BumpAllocator::allocation_size<T>(ctx, A.data().size()) * 2 + // For Ainv and Acopy
                BumpAllocator::allocation_size<int64_t>(ctx, A.batch_size() * A.rows()) + // For pivots
                BumpAllocator::allocation_size<T*>(ctx, A.batch_size()) * 2 + // For data_ptrs
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

    #define COND_INSTANTIATE(fp, fmt) \
    template Event cond<Backend::CUDA, fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        const NormType,\
        const Span<fp>,\
        const Span<std::byte>);\
    template UnifiedVector<fp> cond<Backend::CUDA, fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        const NormType);\
    template size_t cond_buffer_size<Backend::CUDA, fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        const NormType);
    
    COND_INSTANTIATE(float, MatrixFormat::Dense)
    COND_INSTANTIATE(double, MatrixFormat::Dense)

} // namespace batchlas