#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <blas/functions.hh>
#include <complex>

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
            throw std::runtime_error("GEMM: incompatible matrix dimensions");
        }

        if (gemm_has_heterogeneous_batch(A, B, C)) {
            Event last_event;
            bool launched = false;
            for (int batch_index = 0; batch_index < A.batch_size(); ++batch_index) {
                const auto [m, k] = get_effective_dims(A, transA, batch_index);
                const auto [k_b, n] = get_effective_dims(B, transB, batch_index);
                static_cast<void>(k_b);
                if (m == 0 || n == 0) {
                    continue;
                }
                if (k == 0) {
                    last_event = scale(ctx, beta, C.batch_item(batch_index));
                    launched = true;
                    continue;
                }
                last_event = gemm_vendor<Back, T>(ctx,
                                                  A.batch_item(batch_index),
                                                  B.batch_item(batch_index),
                                                  C.batch_item(batch_index),
                                                  alpha,
                                                  beta,
                                                  transA,
                                                  transB,
                                                  precision);
                launched = true;
            }
            if (launched) {
                return std::move(last_event);
            }
            return ctx.create_event_after_external_work();
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

    template <Backend B, typename T, typename std::enable_if<std::is_floating_point_v<T>, int>::type>
    Event syrk(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Uplo uplo,
               Transpose transA) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        if (C.rows() != C.cols()) {
            throw std::runtime_error("SYRK: C must be square");
        }
        if (A.batch_size() != C.batch_size()) {
            throw std::runtime_error("SYRK: batch size mismatch");
        }

        const int n = C.rows();
        const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
        const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
        if (expected_n != n || k <= 0) {
            throw std::runtime_error("SYRK: incompatible matrix dimensions");
        }

        const auto roc_uplo = enum_convert<BackendLibrary::ROCBLAS>(uplo);
        const auto roc_trans = enum_convert<BackendLibrary::ROCBLAS>(transA);

        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            if constexpr (std::is_same_v<T, float>) {
                rocblas_ssyrk(handle,
                              roc_uplo,
                              roc_trans,
                              n,
                              k,
                              &alpha,
                              A_i.data_ptr(),
                              A_i.ld(),
                              &beta,
                              C_i.data_ptr(),
                              C_i.ld());
            } else if constexpr (std::is_same_v<T, double>) {
                rocblas_dsyrk(handle,
                              roc_uplo,
                              roc_trans,
                              n,
                              k,
                              &alpha,
                              A_i.data_ptr(),
                              A_i.ld(),
                              &beta,
                              C_i.data_ptr(),
                              C_i.ld());
            }
        };

        if (A.batch_size() <= 1) {
            launch_single(A, C);
        } else {
            for (int batch = 0; batch < A.batch_size(); ++batch) {
                launch_single(A[batch], C[batch]);
            }
        }
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T, typename std::enable_if<std::is_floating_point_v<T>, int>::type>
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

        if (C.rows() != C.cols()) {
            throw std::runtime_error("SYR2K: C must be square");
        }
        if (A.batch_size() != Bmat.batch_size() || A.batch_size() != C.batch_size()) {
            throw std::runtime_error("SYR2K: batch size mismatch");
        }

        const int n = C.rows();
        const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
        const int expected_b_n = transA == Transpose::NoTrans ? Bmat.rows() : Bmat.cols();
        const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
        const int b_k = transA == Transpose::NoTrans ? Bmat.cols() : Bmat.rows();
        if (expected_n != n || expected_b_n != n || b_k != k || k <= 0) {
            throw std::runtime_error("SYR2K: incompatible matrix dimensions");
        }

        const auto roc_uplo = enum_convert<BackendLibrary::ROCBLAS>(uplo);
        const auto roc_trans = enum_convert<BackendLibrary::ROCBLAS>(transA);

        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            if constexpr (std::is_same_v<T, float>) {
                rocblas_ssyr2k(handle,
                               roc_uplo,
                               roc_trans,
                               n,
                               k,
                               &alpha,
                               A_i.data_ptr(),
                               A_i.ld(),
                               B_i.data_ptr(),
                               B_i.ld(),
                               &beta,
                               C_i.data_ptr(),
                               C_i.ld());
            } else if constexpr (std::is_same_v<T, double>) {
                rocblas_dsyr2k(handle,
                               roc_uplo,
                               roc_trans,
                               n,
                               k,
                               &alpha,
                               A_i.data_ptr(),
                               A_i.ld(),
                               B_i.data_ptr(),
                               B_i.ld(),
                               &beta,
                               C_i.data_ptr(),
                               C_i.ld());
            }
        };

        if (A.batch_size() <= 1) {
            launch_single(A, Bmat, C);
        } else {
            for (int batch = 0; batch < A.batch_size(); ++batch) {
                launch_single(A[batch], Bmat[batch], C[batch]);
            }
        }
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

        if (A.rows() != A.cols()) {
            throw std::runtime_error("TRMM: A must be square");
        }
        if (A.batch_size() != Bmat.batch_size() || A.batch_size() != C.batch_size()) {
            throw std::runtime_error("TRMM: batch size mismatch");
        }

        const int m = C.rows();
        const int n = C.cols();
        const int expected_dim = side == Side::Left ? m : n;
        if (A.rows() != expected_dim || Bmat.rows() != m || Bmat.cols() != n) {
            throw std::runtime_error("TRMM: incompatible matrix dimensions");
        }

        const auto roc_side = enum_convert<BackendLibrary::ROCBLAS>(side);
        const auto roc_uplo = enum_convert<BackendLibrary::ROCBLAS>(uplo);
        const auto roc_trans = enum_convert<BackendLibrary::ROCBLAS>(transA);
        const auto roc_diag = enum_convert<BackendLibrary::ROCBLAS>(diag);

        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            if constexpr (std::is_same_v<T, float>) {
                rocblas_strmm(handle,
                              roc_side,
                              roc_uplo,
                              roc_trans,
                              roc_diag,
                              m,
                              n,
                              &alpha,
                              A_i.data_ptr(),
                              A_i.ld(),
                              B_i.data_ptr(),
                              B_i.ld(),
                              C_i.data_ptr(),
                              C_i.ld(),
                              C_i.data_ptr(),
                              C_i.ld());
            } else if constexpr (std::is_same_v<T, double>) {
                rocblas_dtrmm(handle,
                              roc_side,
                              roc_uplo,
                              roc_trans,
                              roc_diag,
                              m,
                              n,
                              &alpha,
                              A_i.data_ptr(),
                              A_i.ld(),
                              B_i.data_ptr(),
                              B_i.ld(),
                              C_i.data_ptr(),
                              C_i.ld(),
                              C_i.data_ptr(),
                              C_i.ld());
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                rocblas_ctrmm(handle,
                              roc_side,
                              roc_uplo,
                              roc_trans,
                              roc_diag,
                              m,
                              n,
                              reinterpret_cast<const rocblas_float_complex*>(&alpha),
                              reinterpret_cast<const rocblas_float_complex*>(A_i.data_ptr()),
                              A_i.ld(),
                              reinterpret_cast<const rocblas_float_complex*>(B_i.data_ptr()),
                              B_i.ld(),
                              reinterpret_cast<rocblas_float_complex*>(C_i.data_ptr()),
                              C_i.ld(),
                              reinterpret_cast<rocblas_float_complex*>(C_i.data_ptr()),
                              C_i.ld());
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                rocblas_ztrmm(handle,
                              roc_side,
                              roc_uplo,
                              roc_trans,
                              roc_diag,
                              m,
                              n,
                              reinterpret_cast<const rocblas_double_complex*>(&alpha),
                              reinterpret_cast<const rocblas_double_complex*>(A_i.data_ptr()),
                              A_i.ld(),
                              reinterpret_cast<const rocblas_double_complex*>(B_i.data_ptr()),
                              B_i.ld(),
                              reinterpret_cast<rocblas_double_complex*>(C_i.data_ptr()),
                              C_i.ld(),
                              reinterpret_cast<rocblas_double_complex*>(C_i.data_ptr()),
                              C_i.ld());
            }
        };

        if (A.batch_size() <= 1) {
            launch_single(A, Bmat, C);
        } else {
            for (int batch = 0; batch < A.batch_size(); ++batch) {
                launch_single(A[batch], Bmat[batch], C[batch]);
            }
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
    #define TRMM_INSTANTIATE(fp) \
    template Event trmm<Backend::ROCM, fp>(Queue&, const MatrixView<fp,MatrixFormat::Dense>&, const MatrixView<fp,MatrixFormat::Dense>&, const MatrixView<fp,MatrixFormat::Dense>&, fp, Side, Uplo, Transpose, Diag);
    #define SYRK_INSTANTIATE(fp) \
    template Event syrk<Backend::ROCM, fp>(Queue&, const MatrixView<fp,MatrixFormat::Dense>&, const MatrixView<fp,MatrixFormat::Dense>&, fp, fp, Uplo, Transpose);
    #define SYR2K_INSTANTIATE(fp) \
    template Event syr2k<Backend::ROCM, fp>(Queue&, const MatrixView<fp,MatrixFormat::Dense>&, const MatrixView<fp,MatrixFormat::Dense>&, const MatrixView<fp,MatrixFormat::Dense>&, fp, fp, Uplo, Transpose);

    #define BLAS_INSTANTIATE(fp) \
        GEMM_INSTANTIATE(fp) \
        GEMV_INSTANTIATE(fp) \
        TRSM_INSTANTIATE(fp) \
        TRMM_INSTANTIATE(fp) \
        SYRK_INSTANTIATE(fp)

    BLAS_INSTANTIATE(float)
    BLAS_INSTANTIATE(double)
    BLAS_INSTANTIATE(std::complex<float>)
    BLAS_INSTANTIATE(std::complex<double>)
    SYR2K_INSTANTIATE(float)
    SYR2K_INSTANTIATE(double)

    #undef GEMM_INSTANTIATE
    #undef GEMV_INSTANTIATE
    #undef TRSM_INSTANTIATE
    #undef TRMM_INSTANTIATE
    #undef SYRK_INSTANTIATE
    #undef SYR2K_INSTANTIATE
    #undef BLAS_INSTANTIATE
}
