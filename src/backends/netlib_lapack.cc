#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <complex>
#include <lapack.h>
#include <blas/linalg.hh>
namespace batchlas{

    template <Backend Back, typename T, MatrixFormat MFormat>
    Event spmm(Queue& ctx,
               const MatrixView<T, MFormat>& A,
               const MatrixView<T, MatrixFormat::Dense>& B,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Transpose transA,
               Transpose transB,
               Span<std::byte> workspace) {
        static_cast<void>(workspace); // no workspace needed for CPU implementation

        if constexpr (MFormat == MatrixFormat::CSR) {
            int batch = A.batch_size();
            for (int b = 0; b < batch; ++b) {
                auto A_b = A[b];
                auto B_b = B[b];
                auto C_b = C[b];

                int m = A_b.rows();
                int k = A_b.cols();
                int n = B_b.cols();

                // Only handle no transpose cases for now
                if (transA != Transpose::NoTrans || transB != Transpose::NoTrans) {
                    throw std::runtime_error("NETLIB spmm only supports NoTrans for now");
                }

                for (int row = 0; row < m; ++row) {
                    for (int col = 0; col < n; ++col) {
                        T sum = beta * C_b.at(row, col);
                        for (int idx = A_b.row_offsets()[row]; idx < A_b.row_offsets()[row + 1]; ++idx) {
                            int a_col = A_b.col_indices()[idx];
                            sum += alpha * A_b.data()[idx] * B_b.at(a_col, col);
                        }
                        C_b.at(row, col) = sum;
                    }
                }
            }
        } else {
            throw std::runtime_error("Unsupported sparse format for NETLIB spmm");
        }
        return ctx.get_event();
    }

    template <Backend Back, typename T, MatrixFormat MFormat>
    size_t spmm_buffer_size(Queue& ctx,
                            const MatrixView<T, MFormat>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B,
                            const MatrixView<T, MatrixFormat::Dense>& C,
                            T alpha,
                            T beta,
                            Transpose transA,
                            Transpose transB) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(C);
        static_cast<void>(alpha);
        static_cast<void>(beta);
        static_cast<void>(transA);
        static_cast<void>(transB);
        return 0;
    }
    
    template <Backend B, typename T>
    Event gemm(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& descrA,
                   const MatrixView<T, MatrixFormat::Dense>& descrB,
                   const MatrixView<T, MatrixFormat::Dense>& descrC,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB,
                   ComputePrecision precision) {
        auto [m, k] = get_effective_dims(descrA, transA);
        auto [kB, n] = get_effective_dims(descrB, transB);

        if (descrA.batch_size() == 1) {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    cblas_sgemm, cblas_dgemm, cblas_cgemm, cblas_zgemm,
                    Layout::ColMajor, transA, transB,
                    m, n, k,
                    alpha,
                    descrA.data_ptr(), descrA.ld(),
                    descrB.data_ptr(), descrB.ld(),
                    beta,
                    descrC.data_ptr(), descrC.ld());    
        } else {
            for (int i = 0; i < descrA.batch_size(); ++i) {
                gemm<Backend::NETLIB>(
                    ctx,
                    descrA[i],
                    descrB[i],
                    descrC[i],
                    alpha,
                    beta,
                    transA,
                    transB,
                    precision);

            }
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    Event gemv(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const VectorView<T>& X,
               const VectorView<T>& Y,
               T alpha,
               T beta,
               Transpose transA) {
        auto [m, n] = get_effective_dims(A, transA);

        if (A.batch_size() > 1) {
            for (int i = 0; i < A.batch_size(); ++i) {
                gemv<B>(ctx,
                       A[i],
                       X.batch_item(i),
                       Y.batch_item(i),
                       alpha,
                       beta,
                       transA);
            }
        } else {
            call_backend_nh<T, BackendLibrary::CBLAS>(
                cblas_sgemv, cblas_dgemv, cblas_cgemv, cblas_zgemv,
                Layout::ColMajor,
                transA,
                m,
                n,
                alpha,
                A.data_ptr(),
                A.ld(),
                X.data_ptr(),
                X.inc(),
                beta,
                Y.data_ptr(),
                Y.inc());
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    Event trsm(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA,
        const MatrixView<T, MatrixFormat::Dense>& descrB,
        Side side,
        Uplo uplo,
        Transpose transA,
        Diag diag,
        T alpha) {
        
        auto [kB, n] = get_effective_dims(descrB, Transpose::NoTrans);
        if (descrA.batch_size() == 1) {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    cblas_strsm, cblas_dtrsm, cblas_ctrsm, cblas_ztrsm,
                    Layout::ColMajor, side, uplo, transA, diag,
                    n, kB,
                    alpha,
                    descrA.data_ptr(), descrA.ld(),
                    descrB.data_ptr(), descrB.ld());    
        } else {
            for (int i = 0; i < descrA.batch_size(); ++i) {
                trsm<Backend::NETLIB>(
                    ctx,
                    descrA[i],
                    descrB[i],
                    side,
                    uplo,
                    transA,
                    diag,
                    alpha);
            }
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    Event potrf(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& descrA,
                    Uplo uplo,
                    Span<std::byte> workspace) {
        if (descrA.batch_size() == 1) {
            call_backend_nh<T, BackendLibrary::LAPACKE>(
                LAPACKE_spotrf, LAPACKE_dpotrf, LAPACKE_cpotrf, LAPACKE_zpotrf,
                Layout::ColMajor, uplo,
                descrA.rows(), descrA.data_ptr(), descrA.ld());
        } else {
            for (int i = 0; i < descrA.batch_size(); ++i) {
                potrf<Backend::NETLIB>(
                    ctx,
                    descrA[i],
                    uplo,
                    workspace);
            }
        }
        return ctx.get_event();    
    }

    template <Backend B, typename T>
    Event syev(Queue& ctx,
                   const MatrixView<T,MatrixFormat::Dense>& descrA,
                   Span<T> eigenvalues,
                   JobType jobtype,
                   Uplo uplo,
                   Span<std::byte> workspace) {
        if (descrA.batch_size() == 1) {
            call_backend_nh<T, BackendLibrary::LAPACKE>(
                LAPACKE_ssyev, LAPACKE_dsyev, LAPACKE_cheev, LAPACKE_zheev,
                Layout::ColMajor, jobtype, uplo,
                descrA.rows(), descrA.data_ptr(), descrA.ld(),
                base_float_ptr_convert(eigenvalues.data()));
        } else {
            for (int i = 0; i < descrA.batch_size(); ++i) {
                syev<Backend::NETLIB>(
                    ctx,
                    descrA[i],
                    eigenvalues.subspan(i*descrA.rows()),
                    jobtype,
                    uplo,
                    workspace);
            }
        }
        return ctx.get_event();
    }

    template <Backend Back, typename T>
    Event getrs(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& B,
                Transpose transA,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        int n = A.rows();
        int nrhs = B.cols();
        for (int i = 0; i < A.batch_size(); ++i) {
            call_backend_nh<T, BackendLibrary::LAPACKE>(
                LAPACKE_sgetrs, LAPACKE_dgetrs, LAPACKE_cgetrs, LAPACKE_zgetrs,
                Layout::ColMajor,
                transA,
                n,
                nrhs,
                A[i].data_ptr(),
                A.ld(),
                pivots.as_span<int>().data() + i * n,
                B[i].data_ptr(),
                B.ld());
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
    Event getrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        int n = A.rows();
        for (int i = 0; i < A.batch_size(); ++i) {
            call_backend_nh<T, BackendLibrary::LAPACKE>(
                LAPACKE_sgetrf, LAPACKE_dgetrf, LAPACKE_cgetrf, LAPACKE_zgetrf,
                Layout::ColMajor,
                n,
                n,
                A[i].data_ptr(),
                A.ld(),
                pivots.as_span<int>().data() + i * n);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t getrf_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        return 0;
    }

    template <Backend B, typename T>
    Event getri(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& C,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        int n = A.rows();
        for (int i = 0; i < A.batch_size(); ++i) {
            call_backend_nh<T, BackendLibrary::LAPACKE>(
                LAPACKE_sgetri, LAPACKE_dgetri, LAPACKE_cgetri, LAPACKE_zgetri,
                Layout::ColMajor,
                n,
                C[i].data_ptr(),
                C.ld(),
                pivots.as_span<int>().data() + i * n);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t getri_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        return 0;
    }

    template <Backend B, typename T>
    Event geqrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        int m = A.rows();
        int n = A.cols();

        for (int i = 0; i < A.batch_size(); ++i) {
            call_backend_nh<T, BackendLibrary::LAPACKE>(
                LAPACKE_sgeqrf, LAPACKE_dgeqrf, LAPACKE_cgeqrf, LAPACKE_zgeqrf,
                Layout::ColMajor,
                m,
                n,
                A[i].data_ptr(),
                A.ld(),
                tau.data() + i * std::min(m, n));
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
    Event orgqr(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        int m = A.rows();
        int n = A.cols();
        int k = std::min(m, n);
        for (int i = 0; i < A.batch_size(); ++i) {
            call_backend_nh<T, BackendLibrary::LAPACKE>(
                LAPACKE_sorgqr, LAPACKE_dorgqr, LAPACKE_cungqr, LAPACKE_zungqr,
                Layout::ColMajor,
                m,
                n,
                k,
                A[i].data_ptr(),
                A.ld(),
                tau.data() + i * k);
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
    Event ormqr(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& C,
                Side side,
                Transpose trans,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        int m = C.rows();
        int n = C.cols();
        int k = std::min(A.rows(), A.cols());
        for (int i = 0; i < A.batch_size(); ++i) {
            call_backend_nh<T, BackendLibrary::LAPACKE>(
                LAPACKE_sormqr, LAPACKE_dormqr, LAPACKE_cunmqr, LAPACKE_zunmqr,
                Layout::ColMajor,
                side,
                trans,
                m,
                n,
                k,
                A[i].data_ptr(),
                A.ld(),
                tau.data() + i * k,
                C[i].data_ptr(),
                C.ld());
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
    size_t potrf_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& descrA,
                             Uplo uplo) {
        static_cast<void>(ctx);
        static_cast<void>(descrA);
        static_cast<void>(uplo);
        return 0;
    }

    template <Backend B, typename T>
    size_t syev_buffer_size(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& descrA,
                   Span<T> eigenvalues,
                   JobType jobtype,
                   Uplo uplo) {
        return 0;
    }


    #define SPMM_INSTANTIATE(fp, F) \
    template Event spmm<Backend::NETLIB, fp, F>( \
        Queue&, \
        const MatrixView<fp, F>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose, Span<std::byte>);

    #define SPMM_BUFFER_SIZE_INSTANTIATE(fp, F) \
    template size_t spmm_buffer_size<Backend::NETLIB, fp, F>( \
        Queue&, \
        const MatrixView<fp, F>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose);

    #define GEMM_INSTANTIATE(fp) \
    template Event gemm<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose, ComputePrecision);

    #define GEMV_INSTANTIATE(fp) \
    template Event gemv<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        fp, fp, Transpose);

    #define TRSM_INSTANTIATE(fp) \
    template Event trsm<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Uplo, Transpose, Diag, fp);

    #define GEQRF_INSTANTIATE(fp) \
    template Event geqrf<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>, \
        Span<std::byte>);

    #define GEQRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t geqrf_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>);

    #define ORGQR_INSTANTIATE(fp) \
    template Event orgqr<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>, \
        Span<std::byte>);

    #define ORGQR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t orgqr_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>);

    #define GETRS_INSTANTIATE(fp) \
    template Event getrs<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose, \
        Span<int64_t>, \
        Span<std::byte>);

    #define GETRS_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getrs_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose);

    #define GETRF_INSTANTIATE(fp) \
    template Event getrf<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<int64_t>, \
        Span<std::byte>);

    #define GETRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getrf_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&);

    #define GETRI_INSTANTIATE(fp) \
    template Event getri<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<int64_t>, \
        Span<std::byte>);

    #define GETRI_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getri_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&);

    #define ORMQR_INSTANTIATE(fp) \
    template Event ormqr<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>, \
        Span<std::byte>);

    #define ORMQR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t ormqr_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>);

    #define POTRF_INSTANTIATE(fp) \
    template Event potrf<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo, Span<std::byte>);

    #define POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t potrf_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo);

    #define SYEV_INSTANTIATE(fp) \
    template Event syev<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>, JobType, Uplo, Span<std::byte>);

    #define SYEV_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t syev_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>, JobType, Uplo);

    #define BLAS_LEVEL3_INSTANTIATE(fp) \
        SPMM_INSTANTIATE(fp, MatrixFormat::CSR) \
        SPMM_BUFFER_SIZE_INSTANTIATE(fp, MatrixFormat::CSR) \
        GEMM_INSTANTIATE(fp) \
        GEMV_INSTANTIATE(fp) \
        TRSM_INSTANTIATE(fp) \
        GEQRF_INSTANTIATE(fp) \
        GEQRF_BUFFER_SIZE_INSTANTIATE(fp) \
        GETRS_INSTANTIATE(fp) \
        GETRS_BUFFER_SIZE_INSTANTIATE(fp) \
        GETRF_INSTANTIATE(fp) \
        GETRF_BUFFER_SIZE_INSTANTIATE(fp) \
        GETRI_INSTANTIATE(fp) \
        GETRI_BUFFER_SIZE_INSTANTIATE(fp) \
        ORMQR_INSTANTIATE(fp) \
        ORMQR_BUFFER_SIZE_INSTANTIATE(fp) \
        ORGQR_INSTANTIATE(fp) \
        ORGQR_BUFFER_SIZE_INSTANTIATE(fp) \
        POTRF_INSTANTIATE(fp) \
        POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_INSTANTIATE(fp) \
        SYEV_BUFFER_SIZE_INSTANTIATE(fp)

    // Instantiate for the floating-point types of interest.
    BLAS_LEVEL3_INSTANTIATE(float)
    BLAS_LEVEL3_INSTANTIATE(double)
    BLAS_LEVEL3_INSTANTIATE(std::complex<float>)
    BLAS_LEVEL3_INSTANTIATE(std::complex<double>)

    #undef SPMM_INSTANTIATE
    #undef SPMM_BUFFER_SIZE_INSTANTIATE
    #undef GEMM_INSTANTIATE
    #undef GEMV_INSTANTIATE
    #undef TRSM_INSTANTIATE
    #undef GEQRF_INSTANTIATE
    #undef GEQRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRS_INSTANTIATE
    #undef GETRS_BUFFER_SIZE_INSTANTIATE
    #undef GETRF_INSTANTIATE
    #undef GETRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRI_INSTANTIATE
    #undef GETRI_BUFFER_SIZE_INSTANTIATE
    #undef ORMQR_INSTANTIATE
    #undef ORMQR_BUFFER_SIZE_INSTANTIATE
    #undef ORGQR_INSTANTIATE
    #undef ORGQR_BUFFER_SIZE_INSTANTIATE
    #undef POTRF_INSTANTIATE
    #undef POTRF_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_INSTANTIATE
    #undef SYEV_BUFFER_SIZE_INSTANTIATE
    #undef BLAS_LEVEL3_INSTANTIATE
}