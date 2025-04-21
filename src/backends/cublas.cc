#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <complex>

// This file contains cuBLAS primitives implementation
namespace batchlas {

    template <Backend B, typename T, BatchType BT>
    Event gemm(Queue& ctx,
                   const DenseMatView<T,BT>& descrA,
                   const DenseMatView<T,BT>& descrB,
                   const DenseMatView<T,BT>& descrC,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB,
                   ComputePrecision precision) {
        // Call cuBLAS
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        
        auto [m, k] = get_effective_dims(descrA, transA);
        auto [kB, n] = get_effective_dims(descrB, transB);

        if constexpr (BT == BatchType::Single) {
            cublasGemmEx(handle,
                enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                m, n, k,
                &alpha,
                descrA.data_, BackendScalar<T,B>::type, descrA.ld_,
                descrB.data_, BackendScalar<T,B>::type, descrB.ld_,
                &beta,
                descrC.data_, BackendScalar<T,B>::type, descrC.ld_,
                enum_convert<BackendLibrary::CUBLAS, T>(precision),
                CUBLAS_GEMM_DFALT);
        } else {
            cublasGemmStridedBatchedEx(handle,
                enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                m, n, k,
                &alpha,
                descrA.data_, BackendScalar<T,B>::type, descrA.ld_, descrA.stride_,
                descrB.data_, BackendScalar<T,B>::type, descrB.ld_, descrB.stride_,
                &beta,
                descrC.data_, BackendScalar<T,B>::type, descrC.ld_, descrC.stride_,
                descrA.batch_size_,
                enum_convert<BackendLibrary::CUBLAS, T>(precision),
                CUBLAS_GEMM_DFALT);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T, BatchType BT>
    Event gemv(Queue& ctx,
        const DenseMatView<T,BT>& descrA,
        const DenseVecHandle<T,BT>& descrX,
        const DenseVecHandle<T,BT>& descrY,
        T alpha,
        T beta,
        Transpose transA) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = descrA.rows_;
        auto n = descrA.cols_;
        auto batch_size = get_batch_size(descrA);
        
        if constexpr (BT == BatchType::Single) {
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgemv, cublasDgemv, cublasCgemv, cublasZgemv,
                handle, transA, m, n, &alpha, descrA.data_, descrA.ld_, descrX.data_, descrX.inc_, &beta, descrY.data_, descrY.inc_);
        } else {
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgemvStridedBatched, cublasDgemvStridedBatched, cublasCgemvStridedBatched, cublasZgemvStridedBatched,
                handle, transA, m, n, &alpha, descrA.data_, descrA.ld_, descrA.stride_, descrX.data_, descrX.inc_, descrX.stride_, &beta, descrY.data_, descrY.inc_, descrY.stride_, batch_size); 
        }
        return ctx.get_event();
    }

    template <Backend B, typename T, BatchType BT>
    Event trsm(Queue& ctx,
                   const DenseMatView<T,BT>& descrA,
                   const DenseMatView<T,BT>& descrB,
                   Side side,
                   Uplo uplo,
                   Transpose transA,
                   Diag diag,
                   T alpha) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto [kB, n] = get_effective_dims(descrB, Transpose::NoTrans);

        if constexpr (BT == BatchType::Single) {
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasStrsm, cublasDtrsm, cublasCtrsm, cublasZtrsm, 
                handle, side, uplo, transA, diag, kB, n, &alpha, get_data(descrA), descrA.ld_, get_data(descrB), descrB.ld_); 
        } else {
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasStrsmBatched, cublasDtrsmBatched, cublasCtrsmBatched, cublasZtrsmBatched, 
                handle, side, uplo, transA, diag, kB, n, &alpha, get_ptr_arr(ctx, descrA), descrA.ld_, get_ptr_arr(ctx, descrB), descrB.ld_, descrA.batch_size_);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T, BatchType BT>
    Event geqrf(Queue& ctx,
        DenseMatView<T,BT>& A, //In place reflectors (Lower triangle of A)
        Span<T> tau,
        Span<std::byte> work_space) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows_;
        auto n = A.cols_;
        auto batch_size = get_batch_size(A);
        auto pool = BumpAllocator(work_space);
        
        if constexpr (BT == BatchType::Single) {
            cusolverDnParams_t params;
            cusolverDnCreateParams(&params);

            size_t device_l_work, host_l_work;
            cusolverDnXgeqrf_bufferSize(handle, params, m, n,
                BackendScalar<T,B>::type, A.data_, A.ld_,
                BackendScalar<T,B>::type, tau.data(),
                BackendScalar<T,B>::type, &device_l_work, &host_l_work);

            auto device_work_space = pool.allocate<std::byte>(ctx, device_l_work);
            auto host_work_space = pool.allocate<std::byte>(ctx, host_l_work);
            int info;

            cusolverDnXgeqrf(handle, params, m, n,
                BackendScalar<T,B>::type, A.data_, A.ld_,
                BackendScalar<T,B>::type, tau.data(), 
                BackendScalar<T,B>::type, device_work_space.data(), 
                device_l_work, host_work_space.data(), host_l_work, &info);
        } else {
            auto tau_data = tau.data(); // Get base pointer, likely USM device or managed memory
            auto tau_ptrs = pool.allocate<T*>(ctx, batch_size); // Allocate space for pointers, assume USM host/shared

            // Use a SYCL kernel to compute the pointers in parallel
            ctx->parallel_for(sycl::range<1>(batch_size),
                                [=](sycl::id<1> item) {
                size_t i = item.get(0);
                // Calculate the pointer for the i-th batch's tau vector
                tau_ptrs[i] = tau_data + i * n; // Using 'n' as in the original loop
            });
            auto info = pool.allocate<int>(ctx,batch_size);
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgeqrfBatched, cublasDgeqrfBatched, cublasCgeqrfBatched, cublasZgeqrfBatched,
                handle, m, n, get_ptr_arr(ctx, A), A.ld_, tau_ptrs.data(), info.data(), batch_size);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T, BatchType BT>
    size_t geqrf_buffer_size(Queue& ctx,
        const DenseMatView<T,BT>& A, //In place reflectors (Lower triangle of A)
        Span<T> tau) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows_;
        auto n = A.cols_;
        auto batch_size = get_batch_size(A);
        if constexpr (BT == BatchType::Single) {
            size_t device_l_work, host_l_work;
            cusolverDnParams_t params;
            cusolverDnCreateParams(&params);

            cusolverDnXgeqrf_bufferSize(handle, params, m, n,
                BackendScalar<T,B>::type, A.data_, A.ld_,
                BackendScalar<T,B>::type, tau.data(),
                BackendScalar<T,B>::type, &device_l_work, &host_l_work);
            return BumpAllocator::allocation_size<std::byte>(ctx, device_l_work) + BumpAllocator::allocation_size<std::byte>(ctx, host_l_work);
        } else {
            return BumpAllocator::allocation_size<T*>(ctx, batch_size) + BumpAllocator::allocation_size<int>(ctx, batch_size);
        }
    }

    // Template instantiations for cuBLAS functions
    #define GEMM_INSTANTIATE(fp, BT) \
    template Event gemm<Backend::CUDA, fp, BT>( \
        Queue&, \
        const DenseMatView<fp, BT>&, \
        const DenseMatView<fp, BT>&, \
        const DenseMatView<fp, BT>&, \
        fp, fp, Transpose, Transpose, ComputePrecision);

    #define GEMV_INSTANTIATE(fp, BT) \
    template Event gemv<Backend::CUDA, fp, BT>( \
        Queue&, \
        const DenseMatView<fp, BT>&, \
        const DenseVecHandle<fp, BT>&, \
        const DenseVecHandle<fp, BT>&, \
        fp, fp, Transpose);

    #define TRSM_INSTANTIATE(fp, BT) \
    template Event trsm<Backend::CUDA, fp, BT>( \
        Queue&, \
        const DenseMatView<fp, BT>&, \
        const DenseMatView<fp, BT>&, \
        Side, Uplo, Transpose, Diag, fp);

    #define GEQRF_INSTANTIATE(fp, BT) \
    template Event geqrf<Backend::CUDA, fp, BT>( \
        Queue&, \
        DenseMatView<fp, BT>&, \
        Span<fp>, \
        Span<std::byte>);
    
        #define GEQRF_BUFFER_SIZE_INSTANTIATE(fp, BT) \
    template size_t geqrf_buffer_size<Backend::CUDA, fp, BT>( \
        Queue&, \
        const DenseMatView<fp, BT>&, \
        Span<fp>);

    #define BLAS_LEVEL3_INSTANTIATE(fp, BT) \
        GEMM_INSTANTIATE(fp, BT) \
        GEMV_INSTANTIATE(fp, BT) \
        TRSM_INSTANTIATE(fp, BT) \
        GEQRF_INSTANTIATE(fp, BT) \
        GEQRF_BUFFER_SIZE_INSTANTIATE(fp, BT)

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