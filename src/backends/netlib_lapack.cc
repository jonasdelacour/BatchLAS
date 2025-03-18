#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <complex>
#include <lapack.h>
#include <cblas.h>

namespace batchlas{
    
    template <Backend B, typename T, BatchType BT>
    Event gemm(Queue& ctx,
                   DenseMatView<T,BT> descrA,
                   DenseMatView<T,BT> descrB,
                   DenseMatView<T,BT> descrC,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB,
                   ComputePrecision precision) {
        auto [m, k] = get_effective_dims(descrA, transA);
        auto [kB, n] = get_effective_dims(descrB, transB);

        if constexpr (BT == BatchType::Single) {
                call_cblas<T>(
                    cblas_sgemm, cblas_dgemm, cblas_cgemm, cblas_zgemm,
                    descrA.layout_, transA, transB,
                    m, n, k,
                    alpha,
                    descrA.data_, descrA.ld_,
                    descrB.data_, descrB.ld_,
                    beta,
                    descrC.data_, descrC.ld_);    
        } else {
            for (int i = 0; i < get_batch_size(descrA); ++i) {
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

    template <Backend B, typename T, BatchType BT>
    Event trsm(Queue& ctx,
        DenseMatView<T,BT> descrA,
        DenseMatView<T,BT> descrB,
        Side side,
        Uplo uplo,
        Transpose transA,
        Diag diag,
        T alpha) {
        
        auto [kB, n] = get_effective_dims(descrB, Transpose::NoTrans);
        if constexpr (BT == BatchType::Single) {
                call_cblas<T>(
                    cblas_strsm, cblas_dtrsm, cblas_ctrsm, cblas_ztrsm,
                    descrA.layout_, side, uplo, transA, diag,
                    n, kB,
                    alpha,
                    descrA.data_, descrA.ld_,
                    descrB.data_, descrB.ld_);    
        } else {
            for (int i = 0; i < get_batch_size(descrA); ++i) {
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

    template <Backend B, typename T, BatchType BT>
    Event potrf(Queue& ctx,
                    DenseMatView<T,BT> descrA,
                    Uplo uplo,
                    Span<std::byte> workspace) {
        /* if constexpr (BT == BatchType::Single) {
            call_cblas<T>(
                LAPACKE_spotrf, LAPACKE_dpotrf, LAPACKE_cpotrf, LAPACKE_zpotrf,
    */ }


    #define GEMM_INSTANTIATE(fp, BT) \
    template Event gemm<Backend::NETLIB, fp, BT>( \
        Queue&, \
        DenseMatView<fp, BT>, \
        DenseMatView<fp, BT>, \
        DenseMatView<fp, BT>, \
        fp, fp, Transpose, Transpose, ComputePrecision);

    #define TRSM_INSTANTIATE(fp, BT) \
    template Event trsm<Backend::NETLIB, fp, BT>( \
        Queue&, \
        DenseMatView<fp, BT>, \
        DenseMatView<fp, BT>, \
        Side, Uplo, Transpose, Diag, fp);

    #define BLAS_LEVEL3_INSTANTIATE(fp, BT) \
        GEMM_INSTANTIATE(fp, BT) \
        TRSM_INSTANTIATE(fp, BT)

    // Macro that covers all layout and batch type combinations for a given floating-point type.
    #define BLAS_LEVEL3_INSTANTIATE_FOR_FP(fp)        \
        BLAS_LEVEL3_INSTANTIATE(fp, BatchType::Batched) \
        BLAS_LEVEL3_INSTANTIATE(fp, BatchType::Single)

    // Instantiate for the floating-point types of interest.
    BLAS_LEVEL3_INSTANTIATE_FOR_FP(float)
    BLAS_LEVEL3_INSTANTIATE_FOR_FP(double)
    BLAS_LEVEL3_INSTANTIATE_FOR_FP(std::complex<float>)
    BLAS_LEVEL3_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef GEMM_INSTANTIATE
    #undef TRSM_INSTANTIATE
    #undef BLAS_LEVEL3_INSTANTIATE
    #undef BLAS_LEVEL3_INSTANTIATE_FOR_FP
}