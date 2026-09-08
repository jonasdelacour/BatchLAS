#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <batchlas/blas/functions.hh>
#include "../util/template-instantiations.hh"
#include <complex>

#include "batch_launch.hh"
#include "level3_shape.hh"
#include "gemm_variant.hh"
#include "../sycl/gemm_kernels.hh"

namespace batchlas {

    namespace backend {

    template <Backend Back, typename T>
    Event gemm_vendor(Queue& ctx,
               const MatrixView<T,MatrixFormat::Dense>& A,
               const MatrixView<T,MatrixFormat::Dense>& B,
               const MatrixView<T,MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Transpose transA,
               Transpose transB,
               ComputePrecision precision) {
        if (!gemm_batch_dimensions_compatible(A, B, C, transA, transB)) {
            throw std::invalid_argument("GEMM: incompatible matrix dimensions");
        }

        if (gemm_has_heterogeneous_batch(A, B, C)) {
            return gemm_over_heterogeneous_batch(ctx, A, B, C, beta, transA, transB,
                [&](const auto& A_i, const auto& B_i, const auto& C_i) {
                    return gemm_vendor<Back, T>(ctx, A_i, B_i, C_i, alpha, beta, transA, transB, precision);
                },
                [&] { return ctx.create_event_after_external_work(); });
        }

        if (gemm_use_sycl_custom(ctx, A, B, C, transA, transB, precision)) {
            return sycl_gemm::gemm_custom(ctx, A, B, C, alpha, beta, transA, transB, precision);
        }

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

    } // namespace backend

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
        return backend::gemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
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
               T alpha,
               Side side,
               Uplo uplo,
               Transpose transA,
               Diag diag) {
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

    template <Backend B, RealScalar T>
    Event syrk(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Uplo uplo,
               Transpose transA) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        const auto [n, k] = backend::shape::validate_rank_k<std::invalid_argument>("SYRK", A, C, transA, /*hermitian=*/false);

        // The two complex slots have no callee because this overload is
        // constrained to a real T; the complex rank-k update is herk.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::ROCBLAS, B>(rocblas_ssyrk, rocblas_dsyrk, nullptr, nullptr,
                handle, uplo, transA, n, k, &alpha,
                A_i.data_ptr(), A_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        backend::for_each_batch_item(launch_single, A, C);
        return ctx.create_event_after_external_work();
    }

    template <Backend B, RealScalar T>
    Event syr2k(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& Bmat,
                const MatrixView<T, MatrixFormat::Dense>& C,
                T alpha,
                T beta,
                Uplo uplo,
                Transpose transA) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        const auto [n, k] = backend::shape::validate_rank_2k<std::invalid_argument>("SYR2K", A, Bmat, C, transA, /*hermitian=*/false);

        // The two complex slots have no callee because this overload is
        // constrained to a real T; the complex rank-2k update is her2k.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::ROCBLAS, B>(rocblas_ssyr2k, rocblas_dsyr2k, nullptr, nullptr,
                handle, uplo, transA, n, k, &alpha,
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        backend::for_each_batch_item(launch_single, A, Bmat, C);
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T>
    Event trmm(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& Bmat,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               Side side,
               Uplo uplo,
               Transpose transA,
               Diag diag) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        const auto [m, n, k] = backend::shape::validate_product<std::invalid_argument>("TRMM", A, Bmat, C, side);
        static_cast<void>(k);  // rocblas_?trmm takes A's order from side, not as an argument

        // ROCm 6.3 changed rocblas_[sdcz]trmm to the 14-arg out-of-place form:
        //   ..., A, lda, B, ldb, C, ldc   (B=input, C=output)
        // The old 16-arg variant (with a duplicate output pair) was removed.
        //
        // The rocblas_float_complex / rocblas_double_complex casts the two
        // complex arms used to spell out by hand are what ptr_convert already
        // emits for BackendLibrary::ROCBLAS (linalg-impl.hh), so all four arms
        // are the same call.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::ROCBLAS, B>(rocblas_strmm, rocblas_dtrmm, rocblas_ctrmm, rocblas_ztrmm,
                handle, side, uplo, transA, diag, m, n, &alpha,
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(),
                C_i.data_ptr(), C_i.ld());
        };

        backend::for_each_batch_item(launch_single, A, Bmat, C);
        return ctx.create_event_after_external_work();
    }

    // Add further solver routines analogous to cuBLAS implementations using rocSOLVER

    // Explicit instantiations. Signatures live in the `sig` namespace beside each
    // public declaration (include/batchlas/blas/functions/*.hh), so changing one is a single
    // header edit rather than one edit per backend TU.
    #define ROCBLAS_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_OP(B, fp, gemm) \
        BATCHLAS_INSTANTIATE_OP(B, fp, gemv) \
        BATCHLAS_INSTANTIATE_OP(B, fp, trsm) \
        BATCHLAS_INSTANTIATE_OP(B, fp, trmm)

    // syrk/syr2k are constrained to RealScalar T -- only instantiate for real types.
    #define ROCBLAS_REAL_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_OP(B, fp, syrk) \
        BATCHLAS_INSTANTIATE_OP(B, fp, syr2k)

    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(ROCBLAS_OPS, Backend::ROCM)
    BATCHLAS_FOR_EACH_REAL_TYPE_1(ROCBLAS_REAL_OPS, Backend::ROCM)

    #undef ROCBLAS_OPS
    #undef ROCBLAS_REAL_OPS
}
