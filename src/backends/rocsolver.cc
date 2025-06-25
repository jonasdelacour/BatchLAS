#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <complex>
#include <algorithm>
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
    Event geqrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgeqrf, rocsolver_dgeqrf,
                                                         rocsolver_cgeqrf, rocsolver_zgeqrf,
                                                         handle, A.rows(), A.cols(),
                                                         A.data_ptr(), A.ld(), tau.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgeqrf_strided_batched,
                                                         rocsolver_dgeqrf_strided_batched,
                                                         rocsolver_cgeqrf_strided_batched,
                                                         rocsolver_zgeqrf_strided_batched,
                                                         handle, A.rows(), A.cols(),
                                                         A.data_ptr(), A.ld(), A.stride(),
                                                         tau.data(), std::min(A.rows(), A.cols()),
                                                         A.batch_size());
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t geqrf_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(tau);
        return 0;
    }

    template <Backend B, typename T>
    Event ormqr(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& C,
                Side side,
                Transpose trans,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = C.rows();
        auto n = C.cols();
        auto k = std::min(A.rows(), A.cols());
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sormqr, rocsolver_dormqr,
                                                         rocsolver_cunmqr, rocsolver_zunmqr,
                                                         handle,
                                                         enum_convert<BackendLibrary::ROCSOLVER>(side),
                                                         enum_convert<BackendLibrary::ROCSOLVER>(trans),
                                                         m, n, k,
                                                         A.data_ptr(), A.ld(),
                                                         tau.data(),
                                                         C.data_ptr(), C.ld());
        } else {
            Queue sub_queue(ctx.device(), false);
            for (int i = 0; i < A.batch_size(); ++i) {
                ormqr<B>(sub_queue, A.batch_item(i), C.batch_item(i), side, trans,
                          tau.subspan(i * k, k), {});
            }
            sub_queue.wait();
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t ormqr_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Side side,
                             Transpose trans,
                             Span<T> tau) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(C);
        static_cast<void>(side);
        static_cast<void>(trans);
        static_cast<void>(tau);
        return 0;
    }

    template <Backend B, typename T>
    Event orgqr(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sorgqr, rocsolver_dorgqr,
                                                         rocsolver_cungqr, rocsolver_zungqr,
                                                         handle, m, n, k, A.data_ptr(), A.ld(),
                                                         tau.data());
        } else {
            Queue sub_queue(ctx.device(), false);
            for (int i = 0; i < A.batch_size(); ++i) {
                orgqr<B>(sub_queue, A.batch_item(i), tau.subspan(i * k, k), {});
            }
            sub_queue.wait();
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t orgqr_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(tau);
        return 0;
    }

    template <Backend B, typename T>
    Event getrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto info = pool.allocate<int>(ctx, A.batch_size());
        auto ipiv = pivots.as_span<int>();
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgetrf, rocsolver_dgetrf,
                                                         rocsolver_cgetrf, rocsolver_zgetrf,
                                                         handle, A.rows(), A.cols(),
                                                         A.data_ptr(), A.ld(), ipiv.data(),
                                                         info.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgetrf_strided_batched,
                                                         rocsolver_dgetrf_strided_batched,
                                                         rocsolver_cgetrf_strided_batched,
                                                         rocsolver_zgetrf_strided_batched,
                                                         handle, A.rows(), A.cols(), A.data_ptr(),
                                                         A.ld(), A.stride(), ipiv.data(),
                                                         std::min(A.rows(), A.cols()), info.data(),
                                                         A.batch_size());
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t getrf_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        return BumpAllocator::allocation_size<int>(ctx, A.batch_size());
    }

    template <Backend Back, typename T>
    Event getrs(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& B,
                Transpose transA,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        static LinalgHandle<Back> handle;
        handle.setStream(ctx);
        auto ipiv = pivots.as_span<int>();
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, Back>(rocsolver_sgetrs, rocsolver_dgetrs,
                                                         rocsolver_cgetrs, rocsolver_zgetrs,
                                                         handle,
                                                         enum_convert<BackendLibrary::ROCSOLVER>(transA),
                                                         A.rows(), B.cols(),
                                                         A.data_ptr(), A.ld(), ipiv.data(),
                                                         B.data_ptr(), B.ld());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, Back>(rocsolver_sgetrs_strided_batched,
                                                         rocsolver_dgetrs_strided_batched,
                                                         rocsolver_cgetrs_strided_batched,
                                                         rocsolver_zgetrs_strided_batched,
                                                         handle,
                                                         enum_convert<BackendLibrary::ROCSOLVER>(transA),
                                                         A.rows(), B.cols(),
                                                         A.data_ptr(), A.ld(), A.stride(),
                                                         ipiv.data(), std::min(A.rows(), A.cols()),
                                                         B.data_ptr(), B.ld(), B.stride(),
                                                         A.batch_size());
        }
        return ctx.get_event();
    }

    template <Backend Back, typename T>
    size_t getrs_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             Transpose transA) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(transA);
        return 0;
    }

    template <Backend B, typename T>
    Event getri(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& C,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto info = pool.allocate<int>(ctx, A.batch_size());
        auto ipiv = pivots.as_span<int>();
        if (A.data_ptr() != C.data_ptr()) {
            ctx->memcpy(C.data_ptr(), A.data_ptr(), sizeof(T) * static_cast<size_t>(A.stride()) * A.batch_size());
        }
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgetri, rocsolver_dgetri,
                                                         rocsolver_cgetri, rocsolver_zgetri,
                                                         handle, A.rows(), C.data_ptr(), C.ld(),
                                                         ipiv.data(), info.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgetri_strided_batched,
                                                         rocsolver_dgetri_strided_batched,
                                                         rocsolver_cgetri_strided_batched,
                                                         rocsolver_zgetri_strided_batched,
                                                         handle, A.rows(), C.data_ptr(), C.ld(), C.stride(),
                                                         ipiv.data(), std::min(A.rows(), A.cols()),
                                                         info.data(), A.batch_size());
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t getri_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        return BumpAllocator::allocation_size<int>(ctx, A.batch_size());
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
        auto ws = pool.allocate<typename base_type<T>::type>(ctx, A.rows() * A.batch_size());
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_ssyev, rocsolver_dsyev, rocsolver_cheev, rocsolver_zheev,
                handle, jobtype, uplo,
                A.rows(), A.data_ptr(), A.ld(), eigenvalues.data(), ws.data(),
                info.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_ssyev_strided_batched, rocsolver_dsyevd_strided_batched,
                rocsolver_cheevd_strided_batched, rocsolver_zheevd_strided_batched,
                handle, jobtype, uplo,
                A.rows(), A.data_ptr(), A.ld(), A.stride(), eigenvalues.data(), A.rows(), ws.data(), A.rows(), info.data(), A.batch_size());
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t syev_buffer_size(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        Span<typename base_type<T>::type> eigenvalues,
                        JobType jobtype,
                        Uplo uplo) {
                            return BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, A.rows() * A.batch_size()) +
                                   BumpAllocator::allocation_size<int>(ctx, A.batch_size());
    }

    #define POTRF_INSTANTIATE(fp) \
    template Event potrf<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Uplo, Span<std::byte>);
    #define POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t potrf_buffer_size<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Uplo);
    #define SYEV_INSTANTIATE(fp) \
    template Event syev<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<typename base_type<fp>::type>, JobType, Uplo, Span<std::byte>);
    #define SYEV_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t syev_buffer_size<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<typename base_type<fp>::type>, JobType, Uplo);
    #define GEQRF_INSTANTIATE(fp) \
    template Event geqrf<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<fp>, Span<std::byte>);
    #define GEQRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t geqrf_buffer_size<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<fp>);
    #define ORMQR_INSTANTIATE(fp) \
    template Event ormqr<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, Side, Transpose, Span<fp>, Span<std::byte>);
    #define ORMQR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t ormqr_buffer_size<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, Side, Transpose, Span<fp>);
    #define ORGQR_INSTANTIATE(fp) \
    template Event orgqr<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<fp>, Span<std::byte>);
    #define ORGQR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t orgqr_buffer_size<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<fp>);
    #define GETRF_INSTANTIATE(fp) \
    template Event getrf<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<int64_t>, Span<std::byte>);
    #define GETRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getrf_buffer_size<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&);
    #define GETRS_INSTANTIATE(fp) \
    template Event getrs<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, Transpose, Span<int64_t>, Span<std::byte>);
    #define GETRS_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getrs_buffer_size<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, Transpose);
    #define GETRI_INSTANTIATE(fp) \
    template Event getri<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, Span<int64_t>, Span<std::byte>);
    #define GETRI_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getri_buffer_size<Backend::ROCM, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&);

    #define ROCSOLVER_INSTANTIATE(fp) \
        POTRF_INSTANTIATE(fp) \
        POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_INSTANTIATE(fp) \
        SYEV_BUFFER_SIZE_INSTANTIATE(fp) \
        GEQRF_INSTANTIATE(fp) \
        GEQRF_BUFFER_SIZE_INSTANTIATE(fp) \
        GETRF_INSTANTIATE(fp) \
        GETRF_BUFFER_SIZE_INSTANTIATE(fp) \
        GETRS_INSTANTIATE(fp) \
        GETRS_BUFFER_SIZE_INSTANTIATE(fp) \
        GETRI_INSTANTIATE(fp) \
        GETRI_BUFFER_SIZE_INSTANTIATE(fp) \
        ORMQR_INSTANTIATE(fp) \
        ORMQR_BUFFER_SIZE_INSTANTIATE(fp) \
        ORGQR_INSTANTIATE(fp) \
        ORGQR_BUFFER_SIZE_INSTANTIATE(fp)

    ROCSOLVER_INSTANTIATE(float)
    ROCSOLVER_INSTANTIATE(double)
    ROCSOLVER_INSTANTIATE(std::complex<float>)
    ROCSOLVER_INSTANTIATE(std::complex<double>)

    #undef POTRF_INSTANTIATE
    #undef POTRF_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_INSTANTIATE
    #undef SYEV_BUFFER_SIZE_INSTANTIATE
    #undef GEQRF_INSTANTIATE
    #undef GEQRF_BUFFER_SIZE_INSTANTIATE
    #undef ORMQR_INSTANTIATE
    #undef ORMQR_BUFFER_SIZE_INSTANTIATE
    #undef ORGQR_INSTANTIATE
    #undef ORGQR_BUFFER_SIZE_INSTANTIATE
    #undef GETRF_INSTANTIATE
    #undef GETRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRS_INSTANTIATE
    #undef GETRS_BUFFER_SIZE_INSTANTIATE
    #undef GETRI_INSTANTIATE
    #undef GETRI_BUFFER_SIZE_INSTANTIATE
    #undef ROCSOLVER_INSTANTIATE
}
