// filepath: /home/jonaslacour/BatchLAS/src/backends/cublas_matrixview.cc
//#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <blas/functions.hh>
#include <complex>

// This file contains cuBLAS primitives implementation using MatrixView
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
        if (A.batch_size() <= 1) {
            cublasGemmEx(handle,
                enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                m, n, k,
                &alpha,
                A.data_ptr(), BackendScalar<T,Back>::type, A.ld(),
                B.data_ptr(), BackendScalar<T,Back>::type, B.ld(),
                &beta,
                C.data_ptr(), BackendScalar<T,Back>::type, C.ld(),
                enum_convert<BackendLibrary::CUBLAS, T>(precision),
                CUBLAS_GEMM_DFALT);
        } else {
            cublasGemmStridedBatchedEx(handle,
                enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                m, n, k,
                &alpha,
                A.data_ptr(), BackendScalar<T,Back>::type, A.ld(), A.stride(),
                B.data_ptr(), BackendScalar<T,Back>::type, B.ld(), B.stride(),
                &beta,
                C.data_ptr(), BackendScalar<T,Back>::type, C.ld(), C.stride(),
                A.batch_size(),
                enum_convert<BackendLibrary::CUBLAS, T>(precision),
                CUBLAS_GEMM_DFALT);
        }
        return ctx.get_event();
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
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgemv, cublasDgemv, cublasCgemv, cublasZgemv,
                handle, transA, m, n, &alpha, A.data_ptr(), A.ld(), X.data_ptr(), X.inc(), &beta, Y.data_ptr(), Y.inc());
        } else {
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgemvStridedBatched, cublasDgemvStridedBatched, cublasCgemvStridedBatched, cublasZgemvStridedBatched,
                handle, transA, m, n, &alpha, A.data_ptr(), A.ld(), A.stride(), X.data_ptr(), X.inc(), X.stride(), &beta, Y.data_ptr(), Y.inc(), Y.stride(), batch_size);
        }
        return ctx.get_event();
    }

    template <Backend Back, typename T>
    Event trsm(Queue& ctx,
                   const MatrixView<T,MatrixFormat::Dense>& A,
                   const MatrixView<T,MatrixFormat::Dense>& B,
                   Side side,
                   Uplo uplo,
                   Transpose transA,
                   Diag diag,
                   T alpha) {
        static LinalgHandle<Back> handle;
        handle.setStream(ctx);
        auto [kB, n] = get_effective_dims(B, Transpose::NoTrans);
        auto batch_size = A.batch_size();
        trsm_validate_params(A, B, side, uplo, transA, diag);

        if (batch_size == 1) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(cublasStrsm, cublasDtrsm, cublasCtrsm, cublasZtrsm, 
                handle, side, uplo, transA, diag, kB, n, &alpha, A.data_ptr(), A.ld(), B.data_ptr(), B.ld()); 
        } else {
            call_backend<T, BackendLibrary::CUBLAS, Back>(cublasStrsmBatched, cublasDtrsmBatched, cublasCtrsmBatched, cublasZtrsmBatched, 
                handle, side, uplo, transA, diag, kB, n, &alpha, A.data_ptrs(ctx).data(), A.ld(), B.data_ptrs(ctx).data(), B.ld(), batch_size);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    Event geqrf(Queue& ctx,
        MatrixView<T,MatrixFormat::Dense>& A, //In place reflectors (Lower triangle of A)
        Span<T> tau,
        Span<std::byte> work_space) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto batch_size = A.batch_size();
        auto pool = BumpAllocator(work_space);
        if (batch_size <= 1) {
            cusolverDnParams_t params;
            cusolverDnCreateParams(&params);
            size_t device_l_work, host_l_work;
            cusolverDnXgeqrf_bufferSize(handle, params, m, n,
                BackendScalar<T,B>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,B>::type, tau.data(),
                BackendScalar<T,B>::type, &device_l_work, &host_l_work);
            auto device_work_space = pool.allocate<std::byte>(ctx, device_l_work);
            auto host_work_space = pool.allocate<std::byte>(ctx, host_l_work);
            int info;
            cusolverDnXgeqrf(handle, params, m, n,
                BackendScalar<T,B>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,B>::type, tau.data(), 
                BackendScalar<T,B>::type, device_work_space.data(), 
                device_l_work, host_work_space.data(), host_l_work, &info);
        } else {
            auto tau_data = tau.data();
            auto tau_ptrs = pool.allocate<T*>(ctx, batch_size);
            ctx->parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> item) {
                size_t i = item.get(0);
                tau_ptrs[i] = tau_data + i * n;
            });
            auto info = pool.allocate<int>(ctx, batch_size);
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgeqrfBatched, cublasDgeqrfBatched, cublasCgeqrfBatched, cublasZgeqrfBatched,
                handle, m, n, A.data_ptrs(ctx).data(), A.ld(), tau_ptrs.data(), info.data(), batch_size);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t geqrf_buffer_size(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A,
        Span<T> tau) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto batch_size = A.batch_size();
        if (batch_size <= 1) {
            size_t device_l_work, host_l_work;
            cusolverDnParams_t params;
            cusolverDnCreateParams(&params);
            cusolverDnXgeqrf_bufferSize(handle, params, m, n,
                BackendScalar<T,B>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,B>::type, tau.data(),
                BackendScalar<T,B>::type, &device_l_work, &host_l_work);
            return BumpAllocator::allocation_size<std::byte>(ctx, device_l_work) + BumpAllocator::allocation_size<std::byte>(ctx, host_l_work);
        } else {
            return BumpAllocator::allocation_size<T*>(ctx, batch_size) + BumpAllocator::allocation_size<int>(ctx, batch_size);
        }
    }

    template <Backend Back, typename T>
    Event getrs(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A,
        const MatrixView<T,MatrixFormat::Dense>& B,
        Transpose transA,
        Span<int64_t> pivots,
        Span<std::byte> work_space) {
            static LinalgHandle<Back> handle;
            handle.setStream(ctx);
            auto n = A.rows();
            auto nrhs = B.cols();
            auto batch_size = A.batch_size();
            auto pool = BumpAllocator(work_space);
            if (batch_size <= 1) {
                auto info = pool.allocate<int>(ctx, 1);
                cusolverDnParams_t params;
                cusolverDnCreateParams(&params);
                cusolverDnXgetrs(handle, params, enum_convert<BackendLibrary::CUBLAS>(transA), n, nrhs,
                    BackendScalar<T,Back>::type, A.data_ptr(), A.ld(),
                    pivots.data(),
                    BackendScalar<T,Back>::type, B.data_ptr(), B.ld(),
                    info.data());
            } else {
                int info;
                auto reinterpreted_pivots = pivots .as_span<int>();
                call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSgetrsBatched, cublasDgetrsBatched, cublasCgetrsBatched, cublasZgetrsBatched,
                    handle, enum_convert<BackendLibrary::CUBLAS>(transA), n, nrhs,
                    A.data_ptrs(ctx).data(), A.ld(), reinterpreted_pivots.data(),
                    B.data_ptrs(ctx).data(), B.ld(), &info, batch_size);
            }
            return ctx.get_event();
        }
    
    template <Backend Back, typename T>
    size_t getrs_buffer_size(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A,
        const MatrixView<T,MatrixFormat::Dense>& B,
        Transpose transA) {
            return BumpAllocator::allocation_size<int>(ctx, A.batch_size() == 1 ? 1 : 0); //batched getrs just uses a single host integer.
        }

    template <Backend B, typename T>
    Event getrf(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        Span<int64_t> pivots,
        Span<std::byte> work_space) {
            static LinalgHandle<B> handle;
            handle.setStream(ctx);
            auto n = A.rows();
            auto batch_size = A.batch_size();
            auto pool = BumpAllocator(work_space);
            auto info = pool.allocate<int>(ctx, batch_size);
            auto reinterpreted_pivots = pivots.as_span<int>();
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgetrfBatched, cublasDgetrfBatched, cublasCgetrfBatched, cublasZgetrfBatched,
                handle, n,
                A.data_ptrs(ctx).data(), A.ld(), reinterpreted_pivots.data(), info.data(), batch_size);
            return ctx.get_event();
        }

    template <Backend B, typename T>
    size_t getrf_buffer_size(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A) {
            return BumpAllocator::allocation_size<int>(ctx, A.batch_size()); //batched getrf just uses a single host integer.
        }

    template <Backend B, typename T>
    Event getri(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& C, //C is overwritten with inverse of A
        Span<int64_t> pivots,
        Span<std::byte> work_space) {
            static LinalgHandle<B> handle;
            handle.setStream(ctx);
            auto n = A.rows();
            auto batch_size = A.batch_size();
            auto pool = BumpAllocator(work_space);
            auto info_arr = pool.allocate<int>(ctx, batch_size);
            auto reinterpreted_pivots = pivots.as_span<int>();
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgetriBatched, cublasDgetriBatched, cublasCgetriBatched, cublasZgetriBatched,
                handle, n,
                A.data_ptrs(ctx).data(), A.ld(), reinterpreted_pivots.data(),
                C.data_ptrs(ctx).data(), C.ld(), info_arr.data(), batch_size);
            return ctx.get_event();
            
        }

    template <Backend B, typename T>
    size_t getri_buffer_size(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A) {
            static LinalgHandle<B> handle;
            handle.setStream(ctx);
            auto n = A.rows();
            auto batch_size = A.batch_size();
            return BumpAllocator::allocation_size<int>(ctx, batch_size);
        }

    // Template instantiations for cuBLAS functions (MatrixView version)
    #define GEMM_INSTANTIATE(fp) \
    template Event gemm<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose, ComputePrecision);

    #define GEMV_INSTANTIATE(fp) \
    template Event gemv<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        fp, fp, Transpose);

    #define TRSM_INSTANTIATE(fp) \
    template Event trsm<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Uplo, Transpose, Diag, fp);

    #define GEQRF_INSTANTIATE(fp) \
    template Event geqrf<Backend::CUDA, fp>( \
        Queue&, \
        MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>, \
        Span<std::byte>);
    
    #define GEQRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t geqrf_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>);

    #define GETRS_INSTANTIATE(fp) \
    template Event getrs<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose, \
        Span<int64_t>, \
        Span<std::byte>);

    #define GETRS_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getrs_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose);
    #define GETRF_INSTANTIATE(fp) \
    template Event getrf<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<int64_t>,\
        Span<std::byte>);
    #define GETRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getrf_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&);
    #define GETRI_INSTANTIATE(fp) \
    template Event getri<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<int64_t>, \
        Span<std::byte>);
    #define GETRI_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getri_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&);

    #define BLAS_LEVEL3_INSTANTIATE(fp)\
        GEMM_INSTANTIATE(fp)\
        GEMV_INSTANTIATE(fp)\
        TRSM_INSTANTIATE(fp)\
        GEQRF_INSTANTIATE(fp)\
        GEQRF_BUFFER_SIZE_INSTANTIATE(fp)\
        GETRS_INSTANTIATE(fp)\
        GETRS_BUFFER_SIZE_INSTANTIATE(fp)\
        GETRF_INSTANTIATE(fp)\
        GETRF_BUFFER_SIZE_INSTANTIATE(fp)\
        GETRI_INSTANTIATE(fp)\
        GETRI_BUFFER_SIZE_INSTANTIATE(fp)


    BLAS_LEVEL3_INSTANTIATE(float)
    BLAS_LEVEL3_INSTANTIATE(double)
    BLAS_LEVEL3_INSTANTIATE(std::complex<float>)
    BLAS_LEVEL3_INSTANTIATE(std::complex<double>)

    #undef GEMM_INSTANTIATE
    #undef GEMV_INSTANTIATE
    #undef TRSM_INSTANTIATE
    #undef GEQRF_INSTANTIATE
    #undef GEQRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRS_INSTANTIATE
    #undef GETRF_INSTANTIATE
    #undef BLAS_LEVEL3_INSTANTIATE
}
