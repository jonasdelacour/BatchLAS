// filepath: /home/jonaslacour/BatchLAS/src/backends/cusparse_matrixview.cc
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include <util/mempool.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <blas/linalg.hh>
#include "backend_handle.cc"
#include <complex>
#include <ios>

// This file contains cuSPARSE primitives implementation using MatrixView
namespace batchlas {

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
        // Call cuSPARSE
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        BumpAllocator pool(workspace);
        auto buffer_size = spmm_buffer_size<B>(ctx, A, B_mat, C, alpha, beta, transA, transB);
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
        return ctx.get_event();
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
        return size;
    }

    #define SPMM_INSTANTIATE(fp, F) \
    template Event spmm<Backend::CUDA, fp, F>( \
        Queue&, \
        const MatrixView<fp, F>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose, Span<std::byte>);
    
    #define SPMM_BUFFER_SIZE_INSTANTIATE(fp, F) \
    template size_t spmm_buffer_size<Backend::CUDA, fp, F>( \
        Queue&, \
        const MatrixView<fp, F>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose);

    #define CUSPARSE_INSTANTIATE(fp, F) \
        SPMM_INSTANTIATE(fp, F) \
        SPMM_BUFFER_SIZE_INSTANTIATE(fp, F)

    // Instantiate for all supported sparse formats
    #define CUSPARSE_INSTANTIATE_FOR_FP(fp) \
        CUSPARSE_INSTANTIATE(fp, MatrixFormat::CSR)

    // Instantiate for the floating-point types of interest
    CUSPARSE_INSTANTIATE_FOR_FP(float)
    CUSPARSE_INSTANTIATE_FOR_FP(double)
    CUSPARSE_INSTANTIATE_FOR_FP(std::complex<float>)
    CUSPARSE_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef SPMM_INSTANTIATE
    #undef SPMM_BUFFER_SIZE_INSTANTIATE
    #undef CUSPARSE_INSTANTIATE
    #undef CUSPARSE_INSTANTIATE_FOR_FP
}
