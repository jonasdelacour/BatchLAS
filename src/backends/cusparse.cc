//#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>

// This file contains cuSPARSE primitives implementation
namespace batchlas {

    template <Backend B, typename T, Format F, BatchType BT>
    SyclEvent spmm(SyclQueue& ctx,
                   SparseMatHandle<T, F, BT>& descrA,
                   DenseMatView<T,BT> descrB,
                   DenseMatView<T,BT> descrC,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB,
                   Span<std::byte> workspace) {
        // Call cuSPARSE
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        BumpAllocator pool(workspace);
        auto buffer_size = spmm_buffer_size<B>(ctx, descrA, descrB, descrC, alpha, beta, transA, transB);
        auto buffer = pool.allocate<std::byte>(ctx, buffer_size);

        cusparseSpMM(
            handle,
            enum_convert<BackendLibrary::CUSPARSE>(transA),
            enum_convert<BackendLibrary::CUSPARSE>(transB),
            &alpha,
            *descrA,
            *descrB,
            &beta,
            *descrC,
            BackendScalar<T,B>::type,
            CUSPARSE_SPMM_ALG_DEFAULT,
            buffer.data()
        );
        return ctx.get_event();
    }

    template <Backend B, typename T, Format F, BatchType BT>
    size_t spmm_buffer_size(SyclQueue& ctx,
                            SparseMatHandle<T, F, BT>& descrA,
                            DenseMatView<T,BT> descrB,
                            DenseMatView<T,BT> descrC,
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
            *descrA,
            *descrB,
            &beta,
            *descrC,
            BackendScalar<T,B>::type,
            CUSPARSE_SPMM_ALG_DEFAULT,
            &size
        );
        return size;
    }

    #define SPMM_INSTANTIATE(fp, F, BT) \
    template SyclEvent spmm<Backend::CUDA, fp, F, BT>( \
        SyclQueue&, \
        SparseMatHandle<fp, F, BT>&, \
        DenseMatView<fp, BT>, \
        DenseMatView<fp, BT>, \
        fp, fp, Transpose, Transpose, Span<std::byte>);
    
    #define SPMM_BUFFER_SIZE_INSTANTIATE(fp, F, BT) \
    template size_t spmm_buffer_size<Backend::CUDA, fp, F, BT>( \
        SyclQueue&, \
        SparseMatHandle<fp, F, BT>&, \
        DenseMatView<fp, BT>, \
        DenseMatView<fp, BT>, \
        fp, fp, Transpose, Transpose);

    #define CUSPARSE_INSTANTIATE(fp, F, BT) \
        SPMM_INSTANTIATE(fp, F, BT) \
        SPMM_BUFFER_SIZE_INSTANTIATE(fp, F, BT)

    // Instantiate for all format and batch type combinations
    #define CUSPARSE_INSTANTIATE_FOR_FP(fp) \
        CUSPARSE_INSTANTIATE(fp, Format::CSR, BatchType::Batched) \
        CUSPARSE_INSTANTIATE(fp, Format::CSR, BatchType::Single)

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