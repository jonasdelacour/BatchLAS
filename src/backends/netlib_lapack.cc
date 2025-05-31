#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <complex>
#include <lapack.h>
#include <cblas.h>
#include <blas/linalg.hh>
namespace batchlas{
    
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

        if (descrA.batch_size() > 1) {
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
    Event trsm(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA,
        const MatrixView<T, MatrixFormat::Dense>& descrB,
        Side side,
        Uplo uplo,
        Transpose transA,
        Diag diag,
        T alpha) {
        
        auto [kB, n] = get_effective_dims(descrB, Transpose::NoTrans);
        if (descrA.batch_size() > 1) {
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
        if (descrA.batch_size() > 1) {
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
        if (descrA.batch_size() > 1) {
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

    template <Backend B, typename T>
    size_t syev_buffer_size(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& descrA,
                   Span<T> eigenvalues,
                   JobType jobtype,
                   Uplo uplo) {
        return 0;
    }


    #define GEMM_INSTANTIATE(fp) \
    template Event gemm<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose, ComputePrecision);

    #define TRSM_INSTANTIATE(fp) \
    template Event trsm<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Uplo, Transpose, Diag, fp);

    #define POTRF_INSTANTIATE(fp) \
    template Event potrf<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo, Span<std::byte>);

    #define SYEV_INSTANTIATE(fp) \
    template Event syev<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>, JobType, Uplo, Span<std::byte>);

    #define BLAS_LEVEL3_INSTANTIATE(fp) \
        GEMM_INSTANTIATE(fp) \
        TRSM_INSTANTIATE(fp) \
        POTRF_INSTANTIATE(fp) \
        SYEV_INSTANTIATE(fp)

    // Instantiate for the floating-point types of interest.
    BLAS_LEVEL3_INSTANTIATE(float)
    BLAS_LEVEL3_INSTANTIATE(double)
    BLAS_LEVEL3_INSTANTIATE(std::complex<float>)
    BLAS_LEVEL3_INSTANTIATE(std::complex<double>)

    #undef GEMM_INSTANTIATE
    #undef TRSM_INSTANTIATE
    #undef POTRF_INSTANTIATE
    #undef BLAS_LEVEL3_INSTANTIATE
    #undef BLAS_LEVEL3_INSTANTIATE_FOR_FP
}