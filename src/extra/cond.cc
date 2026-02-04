#include <blas/extra.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/mempool.hh>
#include <complex>
#include <limits>
#include <type_traits>
#include "../queue.hh"
namespace batchlas
{
    template <Backend B, typename T>
    struct CondSpectralKernel;

    template <Backend B, typename T>
    Event cond_spectral_impl(Queue &ctx,
                             const MatrixView<T, MatrixFormat::Dense> &A,
                             const Span<T> conds,
                             const Span<std::byte> workspace){
                        if (A.rows() != A.cols()) {
                            throw std::runtime_error("cond: Spectral norm requires square symmetric/Hermitian matrices");
                        }

                        using Real = typename base_type<T>::type;
                        const int n = A.rows();
                        const int batch_size = A.batch_size();

                        auto pool = BumpAllocator(workspace);
                        auto A_buf = pool.allocate<T>(ctx, A.data().size());
                        auto A_ptrs = pool.allocate<T*>(ctx, A.batch_size());
                        auto A_copy = MatrixView<T, MatrixFormat::Dense>(A_buf.data(),
                                                                         A.rows(), A.cols(), A.ld(), A.stride(), A.batch_size(),
                                                                         A_ptrs.data());

                        Event copy_e = MatrixView<T, MatrixFormat::Dense>::copy(ctx, A_copy, A);
                        ctx.enqueue(copy_e);

                        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                            A_copy.hermitize(ctx, Uplo::Lower).wait();
                        } else {
                            A_copy.symmetrize(ctx, Uplo::Lower).wait();
                        }

                        auto eig = pool.allocate<Real>(ctx, static_cast<size_t>(batch_size) * n);
                        auto eig_span = Span<Real>(eig.data(), eig.size());

                        const size_t ws_bytes = backend::syev_vendor_buffer_size<B, T>(ctx,
                                                                                       A_copy,
                                                                                       eig_span,
                                                                                       JobType::NoEigenVectors,
                                                                                       Uplo::Lower);
                        auto syev_ws = pool.allocate<std::byte>(ctx, ws_bytes);
                        Event e = backend::syev_vendor<B, T>(ctx,
                                                             A_copy,
                                                             eig_span,
                                                             JobType::NoEigenVectors,
                                                             Uplo::Lower,
                                                             syev_ws);
                        ctx.enqueue(e);

                        ctx->parallel_for<CondSpectralKernel<B, T>>(sycl::range<1>(static_cast<size_t>(batch_size)), [=](sycl::id<1> idx) {
                            const size_t b = idx[0];
                            const Real* ev = eig_span.data() + b * n;
                            Real max_v = sycl::fabs(ev[0]);
                            Real min_v = max_v;
                            for (int i = 1; i < n; ++i) {
                                const Real v = sycl::fabs(ev[i]);
                                max_v = sycl::fmax(max_v, v);
                                min_v = sycl::fmin(min_v, v);
                            }
                            const Real ratio = (min_v == Real(0)) ? std::numeric_limits<Real>::infinity() : (max_v / min_v);
                            conds[b] = static_cast<T>(ratio);
                        });

                        return ctx.get_event();
                    }

    template <Backend B, typename T>
    Event cond_impl(Queue &ctx,
                    const MatrixView<T, MatrixFormat::Dense> &A,
                    const NormType norm_type,
                    const Span<T> conds,
                    const Span<std::byte> workspace){
                        if (norm_type == NormType::Spectral) {
                            return cond_spectral_impl<B>(ctx, A, conds, workspace);
                        }
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
        if (norm_type == NormType::Spectral) {
            if constexpr (MF != MatrixFormat::Dense) {
                throw std::runtime_error("cond: Spectral norm only supported for dense symmetric/Hermitian matrices");
            } else {
                using Real = typename base_type<T>::type;
                const size_t eig_size = static_cast<size_t>(A.rows()) * A.batch_size();
                const size_t syev_ws = backend::syev_vendor_buffer_size<B, T>(ctx,
                                                                              A,
                                                                              Span<Real>(nullptr, eig_size),
                                                                              JobType::NoEigenVectors,
                                                                              Uplo::Lower);
                return  BumpAllocator::allocation_size<T>(ctx, A.data().size()) +
                        BumpAllocator::allocation_size<T*>(ctx, A.batch_size()) +
                        BumpAllocator::allocation_size<Real>(ctx, eig_size) +
                        BumpAllocator::allocation_size<std::byte>(ctx, syev_ws) +
                        BumpAllocator::allocation_size<T>(ctx, A.batch_size());
            }
        }
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

