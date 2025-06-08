#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>

namespace batchlas {

    template <Backend B, typename T>
    Event inv(Queue& ctx,
              const MatrixView<T, MatrixFormat::Dense>& A,
              const MatrixView<T, MatrixFormat::Dense>& Ainv,
              Span<std::byte> workspace) {
        BumpAllocator pool(workspace);
        auto Acopy = MatrixView<T, MatrixFormat::Dense>::deep_copy(
            A,
            pool.allocate<T>(ctx, A.data().size()).data(),
            pool.allocate<T*>(ctx, A.batch_size()).data());
        auto pivots = pool.allocate<int64_t>(ctx, A.rows()*A.batch_size());
        auto getri_ws = pool.allocate<std::byte>(ctx, getri_buffer_size<B>(ctx, Acopy));
        getrf<B>(ctx, Acopy, pivots);
        getri<B>(ctx, Acopy, Ainv, pivots, getri_ws);
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t inv_buffer_size(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A) {
        return  BumpAllocator::allocation_size<T>(ctx, A.data().size()) +
                BumpAllocator::allocation_size<T*>(ctx, A.batch_size()) +
                BumpAllocator::allocation_size<int64_t>(ctx, A.rows()*A.batch_size()) +
                BumpAllocator::allocation_size<std::byte>(ctx, getri_buffer_size<B>(ctx, A));
    }

    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> inv(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& A) {
        Matrix<T, MatrixFormat::Dense> Aout(A.rows(), A.cols(), A.batch_size());
        UnifiedVector<std::byte> workspace(inv_buffer_size<B>(ctx, A));
        inv<B>(ctx, A, Aout.view(), workspace);
        return Aout;
    }

#define INV_INSTANTIATE(fp) \
    template Event inv<Backend::CUDA, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, Span<std::byte>); \
    template size_t inv_buffer_size<Backend::CUDA, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&); \
    template Matrix<fp, MatrixFormat::Dense> inv<Backend::CUDA, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&);

    INV_INSTANTIATE(float)
    INV_INSTANTIATE(double)
    INV_INSTANTIATE(std::complex<float>)
    INV_INSTANTIATE(std::complex<double>)

#undef INV_INSTANTIATE

}
