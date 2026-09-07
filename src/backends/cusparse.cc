// filepath: /home/jonaslacour/BatchLAS/src/backends/cusparse_matrixview.cc
#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/mempool.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <batchlas/blas/linalg.hh>
#include "../util/template-instantiations.hh"
#include "backend_handle_impl.hh"
#include <complex>
#include <ios>

// This file contains cuSPARSE primitives implementation using MatrixView
namespace batchlas {

    namespace backend {
        template <Backend B, typename T, MatrixFormat MFormat>
        size_t spmm_vendor_buffer_size(Queue& ctx,
                                       const MatrixView<T, MFormat>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B_mat,
                                       const MatrixView<T, MatrixFormat::Dense>& C,
                                       T alpha,
                                       T beta,
                                       Transpose transA,
                                       Transpose transB);

    template <Backend B, typename T, MatrixFormat MFormat>
    Event spmm_vendor(Queue& ctx,
               const MatrixView<T, MFormat>& A,
               const MatrixView<T, MatrixFormat::Dense>& B_mat,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Transpose transA,
               Transpose transB,
               Span<std::byte> workspace) {
        // Call cuSPARSE
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        BumpAllocator pool(workspace);
        auto buffer_size = spmm_vendor_buffer_size<B>(ctx, A, B_mat, C, alpha, beta, transA, transB);
        auto buffer = pool.allocate<std::byte>(ctx, buffer_size);

        cusparseSpMM(
            handle,
            enum_convert<BackendLibrary::CUSPARSE>(transA),
            enum_convert<BackendLibrary::CUSPARSE>(transB),
            &alpha,
            *A,
            *B_mat,
            &beta,
            *C,
            BackendScalar<T,BackendLibrary::CUSPARSE>::type,
            CUSPARSE_SPMM_ALG_DEFAULT,
            buffer.data()
        );
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    size_t spmm_vendor_buffer_size(Queue& ctx,
                          const MatrixView<T, MFormat>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B_mat,
                          const MatrixView<T, MatrixFormat::Dense>& C,
                          T alpha,
                          T beta,
                          Transpose transA,
                          Transpose transB) {
        // Call cuSPARSE
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        size_t size = 0;
        cusparseSpMM_bufferSize(
            handle,
            enum_convert<BackendLibrary::CUSPARSE>(transA),
            enum_convert<BackendLibrary::CUSPARSE>(transB),
            &alpha,
            *A,
            *B_mat,
            &beta,
            *C,
            BackendScalar<T,BackendLibrary::CUSPARSE>::type,
            CUSPARSE_SPMM_ALG_DEFAULT,
            &size
        );
        return BumpAllocator::allocation_size<std::byte>(ctx, size);
    }

    } // namespace backend

    template <Backend B, typename T, MatrixFormat MFormat>
    Event spmm(Queue& ctx,
               const MatrixView<T, MFormat>& A,
               const MatrixView<T, MatrixFormat::Dense>& B_mat,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Transpose transA,
               Transpose transB,
               Span<std::byte> workspace) {
        return backend::spmm_vendor<B, T, MFormat>(ctx, A, B_mat, C, alpha, beta, transA, transB, workspace);
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    size_t spmm_buffer_size(Queue& ctx,
                            const MatrixView<T, MFormat>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B_mat,
                            const MatrixView<T, MatrixFormat::Dense>& C,
                            T alpha,
                            T beta,
                            Transpose transA,
                            Transpose transB) {
        return backend::spmm_vendor_buffer_size<B, T, MFormat>(ctx, A, B_mat, C, alpha, beta, transA, transB);
    }

    // Explicit instantiations. Signatures live in the `sig` namespace beside each
    // public declaration (include/batchlas/blas/functions/*.hh), so changing one is a single
    // header edit rather than one edit per backend TU. CSR is the only sparse
    // format cuSPARSE is wired up for here.
    #define CUSPARSE_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_FORMAT_OP(B, fp, MatrixFormat::CSR, spmm) \
        BATCHLAS_INSTANTIATE_FORMAT_OP(B, fp, MatrixFormat::CSR, spmm_buffer_size)

    // Instantiate for the floating-point types of interest
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(CUSPARSE_OPS, Backend::CUDA)

    #undef CUSPARSE_OPS
}
