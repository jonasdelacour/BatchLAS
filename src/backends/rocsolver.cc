#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <complex>
#include <blas/linalg.hh>

namespace batchlas {

    template <Backend B, typename T>
    size_t potrf_buffer_size(Queue& ctx,
                            const MatrixView<T,MatrixFormat::Dense>& A,
                            Uplo uplo) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        size_t size = 0;
        if (A.batch_size() == 1) {
            size = BumpAllocator::allocation_size<int>(ctx,1);
        } else {
            size = BumpAllocator::allocation_size<int>(ctx,A.batch_size());
        }
        return size;
    }

    template <Backend B, typename T>
    Event potrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Uplo uplo,
                Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto info = pool.allocate<int>(ctx, A.batch_size());
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_spotrf, rocsolver_dpotrf, rocsolver_cpotrf, rocsolver_zpotrf,
                handle, enum_convert<BackendLibrary::ROCSOLVER>(uplo), A.rows(), A.data_ptr(), A.ld(), info.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_spotrf_strided_batched, rocsolver_dpotrf_strided_batched,
                rocsolver_cpotrf_strided_batched, rocsolver_zpotrf_strided_batched,
                handle, enum_convert<BackendLibrary::ROCSOLVER>(uplo), A.rows(), A.data_ptr(), A.ld(), A.stride(), info.data(), A.batch_size());
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    Event syev(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<typename base_type<T>::type> eigenvalues,
                   JobType jobtype,
                   Uplo uplo,
                   Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto info = pool.allocate<int>(ctx, A.batch_size());
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_ssyevd, rocsolver_dsyevd, rocsolver_cheevd, rocsolver_zheevd,
                handle, enum_convert<BackendLibrary::ROCSOLVER>(jobtype), enum_convert<BackendLibrary::ROCSOLVER>(uplo),
                A.rows(), A.data_ptr(), A.ld(), eigenvalues.data(), info.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_ssyevd_strided_batched, rocsolver_dsyevd_strided_batched,
                rocsolver_cheevd_strided_batched, rocsolver_zheevd_strided_batched,
                handle, enum_convert<BackendLibrary::ROCSOLVER>(jobtype), enum_convert<BackendLibrary::ROCSOLVER>(uplo),
                A.rows(), A.data_ptr(), A.ld(), A.stride(), eigenvalues.data(), eigenvalues.size()/A.batch_size(), info.data(), A.batch_size());
        }
        return ctx.get_event();
    }

    #define POTRF_INSTANTIATE(fp) \
    template Event potrf<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Uplo, Span<std::byte>);
    #define POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t potrf_buffer_size<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Uplo);
    #define SYEV_INSTANTIATE(fp) \
    template Event syev<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<typename base_type<fp>::type>, JobType, Uplo, Span<std::byte>);

    #define ROCSOLVER_INSTANTIATE(fp) \
        POTRF_INSTANTIATE(fp) \
        POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_INSTANTIATE(fp)

    ROCSOLVER_INSTANTIATE(float)
    ROCSOLVER_INSTANTIATE(double)
    ROCSOLVER_INSTANTIATE(std::complex<float>)
    ROCSOLVER_INSTANTIATE(std::complex<double>)

    #undef POTRF_INSTANTIATE
    #undef POTRF_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_INSTANTIATE
    #undef ROCSOLVER_INSTANTIATE
}
