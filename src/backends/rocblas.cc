#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <blas/functions.hh>
#include <complex>

namespace batchlas {

    template <Backend Back, typename T>
    Event gemm(Queue& ctx,
               const MatrixView<T,MatrixFormat::Dense>& A,
               const MatrixView<T,MatrixFormat::Dense>& B,
               const MatrixView<T,MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Transpose transA,
               Transpose transB,
               ComputePrecision precision) {
        static LinalgHandle<Back> handle;
        handle.setStream(ctx);
        auto [m, k] = get_effective_dims(A, transA);
        auto [kB, n] = get_effective_dims(B, transB);
        auto compute_type = enum_convert<BackendLibrary::ROCBLAS, T>(precision);
        if (A.batch_size() <= 1) {
            call_backend<T, BackendLibrary::ROCBLAS, Back>(rocblas_sgemm, rocblas_dgemm, rocblas_cgemm, rocblas_zgemm,
                             handle, transA, transB,
                                m, n, k,
                                &alpha,
                                A.data_ptr(), A.ld(),
                                B.data_ptr(), B.ld(),
                                &beta,
                                C.data_ptr(), C.ld());
        } else {
            call_backend<T, BackendLibrary::ROCBLAS, Back>(rocblas_sgemm_strided_batched, rocblas_dgemm_strided_batched,
                            rocblas_cgemm_strided_batched, rocblas_zgemm_strided_batched,
                            handle,
                            transA, transB,
                            m, n, k,
                            &alpha,
                            A.data_ptr(), A.ld(), A.stride(),
                            B.data_ptr(), B.ld(), B.stride(),
                            &beta,
                            C.data_ptr(), C.ld(), C.stride(),
                            A.batch_size());
        }
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T>
    Event gemv(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A,
        const VectorView<T>& X,
        const VectorView<T>& Y,
        T alpha,
        T beta,
        Transpose transA) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto batch_size = A.batch_size();
        if (batch_size <= 1) {
            call_backend<T, BackendLibrary::ROCBLAS, B>(rocblas_sgemv, rocblas_dgemv, rocblas_cgemv, rocblas_zgemv,
                handle, enum_convert<BackendLibrary::ROCBLAS>(transA), m, n, &alpha,
                A.data_ptr(), A.ld(), X.data_ptr(), 1, &beta, Y.data_ptr(), 1);
        } else {
            call_backend<T, BackendLibrary::ROCBLAS, B>(rocblas_sgemv_strided_batched, rocblas_dgemv_strided_batched,
                rocblas_cgemv_strided_batched, rocblas_zgemv_strided_batched,
                handle, enum_convert<BackendLibrary::ROCBLAS>(transA), m, n, &alpha,
                A.data_ptr(), A.ld(), A.stride(), X.data_ptr(), 1, X.stride(), &beta, Y.data_ptr(), 1, Y.stride(), batch_size);
        }
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T>
    Event trsm(Queue& ctx,
               const MatrixView<T,MatrixFormat::Dense>& A,
               const MatrixView<T,MatrixFormat::Dense>& Bmat,
               Side side,
               Uplo uplo,
               Transpose transA,
               Diag diag,
               T alpha) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto [kB, n] = get_effective_dims(Bmat, Transpose::NoTrans);
        auto batch_size = A.batch_size();
        trsm_validate_params(A, Bmat, side, uplo, transA, diag);
        if (batch_size == 1) {
            call_backend<T, BackendLibrary::ROCBLAS, B>(rocblas_strsm, rocblas_dtrsm, rocblas_ctrsm, rocblas_ztrsm,
                handle, enum_convert<BackendLibrary::ROCBLAS>(side), enum_convert<BackendLibrary::ROCBLAS>(uplo),
                enum_convert<BackendLibrary::ROCBLAS>(transA), enum_convert<BackendLibrary::ROCBLAS>(diag),
                kB, n, &alpha, A.data_ptr(), A.ld(), Bmat.data_ptr(), Bmat.ld());
        } else {
            call_backend<T, BackendLibrary::ROCBLAS, B>(rocblas_strsm_strided_batched, rocblas_dtrsm_strided_batched,
                rocblas_ctrsm_strided_batched, rocblas_ztrsm_strided_batched,
                handle, enum_convert<BackendLibrary::ROCBLAS>(side), enum_convert<BackendLibrary::ROCBLAS>(uplo),
                enum_convert<BackendLibrary::ROCBLAS>(transA), enum_convert<BackendLibrary::ROCBLAS>(diag),
                kB, n, &alpha, A.data_ptr(), A.ld(), A.stride(), Bmat.data_ptr(), Bmat.ld(), Bmat.stride(), batch_size);
        }
        return ctx.create_event_after_external_work();
    }

    // Add further solver routines analogous to cuBLAS implementations using rocSOLVER

    #define GEMM_INSTANTIATE(fp) \
    template Event gemm<Backend::ROCM, fp>(Queue&, const MatrixView<fp,MatrixFormat::Dense>&, const MatrixView<fp,MatrixFormat::Dense>&, const MatrixView<fp,MatrixFormat::Dense>&, fp, fp, Transpose, Transpose, ComputePrecision);
    #define GEMV_INSTANTIATE(fp) \
    template Event gemv<Backend::ROCM, fp>(Queue&, const MatrixView<fp,MatrixFormat::Dense>&, const VectorView<fp>&, const VectorView<fp>&, fp, fp, Transpose);
    #define TRSM_INSTANTIATE(fp) \
    template Event trsm<Backend::ROCM, fp>(Queue&, const MatrixView<fp,MatrixFormat::Dense>&, const MatrixView<fp,MatrixFormat::Dense>&, Side, Uplo, Transpose, Diag, fp);

    #define BLAS_INSTANTIATE(fp) \
        GEMM_INSTANTIATE(fp) \
        GEMV_INSTANTIATE(fp) \
        TRSM_INSTANTIATE(fp)

    BLAS_INSTANTIATE(float)
    BLAS_INSTANTIATE(double)
    BLAS_INSTANTIATE(std::complex<float>)
    BLAS_INSTANTIATE(std::complex<double>)

    #undef GEMM_INSTANTIATE
    #undef GEMV_INSTANTIATE
    #undef TRSM_INSTANTIATE
    #undef BLAS_INSTANTIATE
}
