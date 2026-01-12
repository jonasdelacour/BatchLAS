// filepath: /home/jonaslacour/BatchLAS/src/backends/cublas_matrixview.cc
//#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <internal/ormqr_blocked.hh>

#include <cstdlib>
#include <string>
#include <blas/functions.hh>
#include <complex>

// This file contains cuBLAS primitives implementation using MatrixView
namespace batchlas {

    namespace {
        inline bool _use_blocked_ormqr() {
            if (const char* p = std::getenv("BATCHLAS_ORMQR_IMPL")) {
                if (std::string(p) == "blocked") return true;
            }
            if (const char* p = std::getenv("BATCHLAS_ORMQR_BLOCKED")) {
                if (std::string(p) == "1" || std::string(p) == "true" || std::string(p) == "TRUE" ||
                    std::string(p) == "on" || std::string(p) == "ON") {
                    return true;
                }
            }
            return false;
        }
    } // namespace

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

        if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
            throw std::runtime_error(
                "GEMM: batch size mismatch (A=" + std::to_string(A.batch_size()) +
                ", B=" + std::to_string(B.batch_size()) +
                ", C=" + std::to_string(C.batch_size()) + ")");
        }

        auto [m, k] = get_effective_dims(A, transA);
        auto [kB, n] = get_effective_dims(B, transB);
        if (A.batch_size() <= 1) {
            cublasGemmEx(handle,
                enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                m, n, k,
                &alpha,
                A.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, A.ld(),
                B.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, B.ld(),
                &beta,
                C.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, C.ld(),
                enum_convert<BackendLibrary::CUBLAS, T>(precision),
                CUBLAS_GEMM_DFALT);
        } else {
            cublasGemmStridedBatchedEx(handle,
                enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                m, n, k,
                &alpha,
                A.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, A.ld(), A.stride(),
                B.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, B.ld(), B.stride(),
                &beta,
                C.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, C.ld(), C.stride(),
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
        const MatrixView<T,MatrixFormat::Dense>& A, //In place reflectors (Lower triangle of A)
        Span<T> tau,
        Span<std::byte> work_space) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        auto batch_size = A.batch_size();
        auto pool = BumpAllocator(work_space);
        if (batch_size <= 1) {
            cusolverDnParams_t params;
            cusolverDnCreateParams(&params);
            size_t device_l_work, host_l_work;
            cusolverDnXgeqrf_bufferSize(handle, params, m, n,
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, tau.data(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, &device_l_work, &host_l_work);
            auto device_work_space = pool.allocate<std::byte>(ctx, device_l_work);
            auto host_work_space = pool.allocate<std::byte>(ctx, host_l_work);
            auto d_info = pool.allocate<int>(ctx, 1);
            cusolverDnXgeqrf(handle, params, m, n,
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, tau.data(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, device_work_space.data(),
                device_l_work, host_work_space.data(), host_l_work, d_info.data());
        } else {
            auto tau_data = tau.data();
            auto tau_ptrs = pool.allocate<T*>(ctx, batch_size);
            ctx->parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> item) {
                size_t i = item.get(0);
                tau_ptrs[i] = tau_data + i * k;
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
                BackendScalar<T,BackendLibrary::CUBLAS>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,BackendLibrary::CUBLAS>::type, tau.data(),
                BackendScalar<T,BackendLibrary::CUBLAS>::type, &device_l_work, &host_l_work);
            return BumpAllocator::allocation_size<std::byte>(ctx, device_l_work) + BumpAllocator::allocation_size<std::byte>(ctx, host_l_work) 
                   + BumpAllocator::allocation_size<int>(ctx, 1); // +1 for info
        } else {
            return BumpAllocator::allocation_size<T*>(ctx, batch_size) + BumpAllocator::allocation_size<int>(ctx, batch_size);
        }
    }

    template <Backend B, typename T>
    Event ormqr(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& C,
                Side side,
                Transpose trans,
                Span<T> tau,
                Span<std::byte> workspace) {
        if (_use_blocked_ormqr()) {
            Queue* q_ptr = &ctx;
            Queue in_order_q;
            if (!ctx.in_order()) {
                in_order_q = Queue(ctx, true);
                q_ptr = &in_order_q;
            }
            return ormqr_blocked<B, T>(*q_ptr, A, C, side, trans, tau, workspace);
        }

        // Ensure the non-blocked cuSOLVER path respects the ordering guarantees of the
        // caller queue. In particular, the batched fallback must not run on a separate
        // Queue/stream without synchronizing with prior kernels that produced A/C.
        Queue* q_ptr = &ctx;
        Queue in_order_q;
        if (!ctx.in_order()) {
            in_order_q = Queue(ctx, true);
            q_ptr = &in_order_q;
        }
        static LinalgHandle<B> handle;
        handle.setStream(*q_ptr);
        auto m = C.rows();
        auto n = C.cols();
        auto k = std::min(A.rows(), A.cols());
        auto batch_size = A.batch_size();
        BumpAllocator pool(workspace);
        if (batch_size == 1) {
            int lwork;
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSormqr_bufferSize, cusolverDnDormqr_bufferSize, 
                cusolverDnCunmqr_bufferSize, cusolverDnZunmqr_bufferSize,
                handle,
                enum_convert<BackendLibrary::CUSOLVER>(side),
                enum_convert<BackendLibrary::CUSOLVER>(trans),
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                C.data_ptr(), C.ld(),
                &lwork);
            auto device_ws = pool.allocate<T>(ctx, lwork);
            auto info = pool.allocate<int>(ctx, 1);
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSormqr, cusolverDnDormqr,
                cusolverDnCunmqr, cusolverDnZunmqr,
                handle,
                enum_convert<BackendLibrary::CUSOLVER>(side),
                enum_convert<BackendLibrary::CUSOLVER>(trans),
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                C.data_ptr(), C.ld(),
                device_ws.data(), lwork, info.data());
        } else {
            size_t single_ws = ormqr_buffer_size<B>(*q_ptr, A.batch_item(0), C.batch_item(0), side, trans, tau.subspan(0, k));
            for (int i = 0; i < batch_size; ++i) {
                auto sub_ws = pool.allocate<std::byte>(*q_ptr, single_ws);
                ormqr<B>(*q_ptr, A.batch_item(i), C.batch_item(i), side, trans, tau.subspan(i * k, k), sub_ws);
            }
        }
        return q_ptr->get_event();
    }

    template <Backend B, typename T>
    size_t ormqr_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Side side,
                             Transpose trans,
                             Span<T> tau) {
        if (_use_blocked_ormqr()) {
            return ormqr_blocked_buffer_size<B, T>(ctx, A, C, side, trans, tau);
        }
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = C.rows();
        auto n = C.cols();
        auto k = std::min(A.rows(), A.cols());
        auto batch_size = A.batch_size();
        if (batch_size == 1) {
            int lwork;
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSormqr_bufferSize, cusolverDnDormqr_bufferSize,
                cusolverDnCunmqr_bufferSize, cusolverDnZunmqr_bufferSize,
                handle,
                enum_convert<BackendLibrary::CUSOLVER>(side),
                enum_convert<BackendLibrary::CUSOLVER>(trans),
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                C.data_ptr(), C.ld(),
                &lwork);
            return BumpAllocator::allocation_size<T>(ctx, lwork) + BumpAllocator::allocation_size<int>(ctx, 1); // +1 for info
        } else {
            size_t single = BumpAllocator::allocation_size<std::byte>(ctx, ormqr_buffer_size<B>(ctx, A.batch_item(0), C.batch_item(0), side, trans, tau.subspan(0, k)));
            return single * batch_size;
        }
    }

    template <Backend B, typename T>
    Event orgqr(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        auto batch_size = A.batch_size();
        BumpAllocator pool(workspace);
        if (batch_size == 1) {
            int lwork;
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSorgqr_bufferSize, cusolverDnDorgqr_bufferSize,
                cusolverDnCungqr_bufferSize, cusolverDnZungqr_bufferSize,
                handle,
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                &lwork);
            auto device_ws = pool.allocate<T>(ctx, lwork);
            auto info = pool.allocate<int>(ctx, 1);
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSorgqr, cusolverDnDorgqr,
                cusolverDnCungqr, cusolverDnZungqr,
                handle,
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                device_ws.data(), lwork, info.data());
        } else {
            Queue sub_queue(ctx.device(), false);
            size_t single_ws = orgqr_buffer_size<B>(ctx, A.batch_item(0), tau.subspan(0, k));
            for (int i = 0; i < batch_size; ++i) {
                auto sub_ws = pool.allocate<std::byte>(sub_queue, single_ws);
                orgqr<B>(sub_queue, A.batch_item(i), tau.subspan(i * k, k), sub_ws);
            }
            sub_queue.wait();
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t orgqr_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        auto batch_size = A.batch_size();
        if (batch_size == 1) {
            int lwork;
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSorgqr_bufferSize, cusolverDnDorgqr_bufferSize,
                cusolverDnCungqr_bufferSize, cusolverDnZungqr_bufferSize,
                handle,
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                &lwork);
            return BumpAllocator::allocation_size<T>(ctx, lwork) + BumpAllocator::allocation_size<int>(ctx, 1);
        } else {
            size_t single = BumpAllocator::allocation_size<std::byte>(ctx, orgqr_buffer_size<B>(ctx, A.batch_item(0), tau.subspan(0, k)));
            return single * batch_size;
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
                    BackendScalar<T,BackendLibrary::CUBLAS>::type, A.data_ptr(), A.ld(),
                    pivots.data(),
                    BackendScalar<T,BackendLibrary::CUBLAS>::type, B.data_ptr(), B.ld(),
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
        const MatrixView<fp, MatrixFormat::Dense>&, \
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

    #define ORMQR_INSTANTIATE(fp) \
    template Event ormqr<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>, \
        Span<std::byte>);

    #define ORMQR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t ormqr_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>);

    #define ORGQR_INSTANTIATE(fp) \
    template Event orgqr<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>, \
        Span<std::byte>);

    #define ORGQR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t orgqr_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>);

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
        GETRI_BUFFER_SIZE_INSTANTIATE(fp)\
        ORMQR_INSTANTIATE(fp)\
        ORMQR_BUFFER_SIZE_INSTANTIATE(fp)\
        ORGQR_INSTANTIATE(fp)\
        ORGQR_BUFFER_SIZE_INSTANTIATE(fp)


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
    #undef GETRS_BUFFER_SIZE_INSTANTIATE
    #undef GETRF_INSTANTIATE
    #undef GETRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRI_INSTANTIATE
    #undef GETRI_BUFFER_SIZE_INSTANTIATE
    #undef ORMQR_INSTANTIATE
    #undef ORMQR_BUFFER_SIZE_INSTANTIATE
    #undef ORGQR_INSTANTIATE
    #undef ORGQR_BUFFER_SIZE_INSTANTIATE
    #undef BLAS_LEVEL3_INSTANTIATE
}
