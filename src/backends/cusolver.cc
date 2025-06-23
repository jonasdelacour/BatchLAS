//#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <blas/linalg.hh>

// This file contains cuSOLVER primitives implementation
namespace batchlas {

    #if defined(CUDART_VERSION) && CUDART_VERSION >= 12060
        #define USE_CUSOLVER_X_API 1
    #else
        #define USE_CUSOLVER_X_API 0
        #pragma message("cuSOLVER X API is not available, using legacy API be wary batches of matrices larger than 128x128")
    #endif

    template <Backend B, typename T>
    size_t potrf_buffer_size(Queue& ctx,
                            const MatrixView<T,MatrixFormat::Dense>& A,
                            Uplo uplo) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        int size = 0;
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSpotrf_bufferSize, cusolverDnDpotrf_bufferSize, cusolverDnCpotrf_bufferSize, cusolverDnZpotrf_bufferSize,
                handle, uplo, A.rows(), A.data_ptr(), A.ld(), &size);
            size = BumpAllocator::allocation_size<std::byte>(ctx, size) + BumpAllocator::allocation_size<int>(ctx, 1);
        } else {
            size =  BumpAllocator::allocation_size<int>(ctx, A.batch_size());
        }
        return size;
    }

    template <Backend B, typename T>
    Event potrf(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& descrA,
                    Uplo uplo,
                    Span<std::byte> workspace) {        
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto Lwork = potrf_buffer_size<B>(ctx, descrA, uplo) - BumpAllocator::allocation_size<int>(ctx, 1);
        if (descrA.batch_size() == 1) {
            auto potrf_span = pool.allocate<std::byte>(ctx, Lwork);
            auto info = pool.allocate<int>(ctx, 1);
            auto status = call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSpotrf, cusolverDnDpotrf, cusolverDnCpotrf, cusolverDnZpotrf,
                handle, uplo, descrA.rows(), descrA.data_ptr(), descrA.ld(), reinterpret_cast<T*>(potrf_span.data()), Lwork, info.data());
        } else {
            auto info = pool.allocate<int>(ctx, descrA.batch_size());
            call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSpotrfBatched, cusolverDnDpotrfBatched, cusolverDnCpotrfBatched, cusolverDnZpotrfBatched,
                handle, uplo, descrA.rows(), descrA.data_ptrs(ctx).data(), descrA.ld(), info.data(), descrA.batch_size());
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    Event syev(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& descrA,
                   Span<typename base_type<T>::type> eigenvalues,
                   JobType jobtype,
                   Uplo uplo,
                   Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        size_t l_work_device = 0;
        size_t l_work_host = 0;
        cusolverDnParams_t params;
        cusolverDnCreateParams(&params);
        syevjInfo_t syevj_info;
        cusolverDnCreateSyevjInfo(&syevj_info);
        if (descrA.batch_size() == 1) {
            cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, enum_convert<BackendLibrary::CUSOLVER>(uplo), descrA.rows(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, descrA.data_ptr(), descrA.ld(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, eigenvalues.data(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, &l_work_device, &l_work_host);
        } else {
            #if USE_CUSOLVER_X_API
                cusolverDnXsyevBatched_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, enum_convert<BackendLibrary::CUSOLVER>(uplo), descrA.rows(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, descrA.data_ptr(), descrA.ld(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, eigenvalues.data(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, &l_work_device, &l_work_host, descrA.batch_size());
            #else
                int l_work_device_int = 0;
                call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                    handle, jobtype, uplo, descrA.rows(), descrA.data_ptr(), descrA.ld(), base_float_ptr_convert(eigenvalues.data()), &l_work_device_int, syevj_info, descrA.batch_size());
                l_work_device = static_cast<size_t>(l_work_device_int);
            #endif
        }
        
        auto host_workspace = pool.allocate<std::byte>(ctx, l_work_host);
        auto device_workspace = pool.allocate<std::byte>(ctx, l_work_device);

        if (descrA.batch_size() == 1) {
            auto info = pool.allocate<int>(ctx, 1);
            cusolverDnXsyevd(
                handle,
                params,
                CUSOLVER_EIG_MODE_VECTOR,
                enum_convert<BackendLibrary::CUSOLVER>(uplo),
                descrA.rows(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type,
                descrA.data_ptr(),
                descrA.ld(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type,
                eigenvalues.data(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type,
                device_workspace.data(),
                l_work_device,
                host_workspace.data(),
                l_work_host,
                info.data());
        } else {
            auto info = pool.allocate<int>(ctx, descrA.batch_size());
            #if USE_CUSOLVER_X_API
                
            cusolverDnXsyevBatched(
                handle,
                params,
                CUSOLVER_EIG_MODE_VECTOR,
                enum_convert<BackendLibrary::CUSOLVER>(uplo),
                descrA.rows(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type,
                descrA.data_ptr(),
                descrA.ld(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type,
                eigenvalues.data(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type,
                device_workspace.data(),
                l_work_device,
                host_workspace.data(),
                l_work_host,
                info.data(),
                descrA.batch_size_);
            #else
            int l_work_device_int = static_cast<int>(l_work_device);
            call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsyevjBatched, cusolverDnDsyevjBatched, cusolverDnCheevjBatched, cusolverDnZheevjBatched,
                handle, jobtype, uplo, descrA.rows(), descrA.data_ptr(), descrA.ld(), base_float_ptr_convert(eigenvalues.data()), device_workspace.template as_span<T>().data(), l_work_device_int, info.data(), syevj_info, descrA.batch_size_);
            #endif
        }
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t syev_buffer_size(Queue& ctx,
                            const MatrixView<T,MatrixFormat::Dense>& descrA,                            
                            Span<typename base_type<T>::type> eigenvalues,
                            JobType jobtype,
                            Uplo uplo) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        size_t l_work_device = 0;
        size_t l_work_host;
        cusolverDnParams_t params;
        cusolverDnCreateParams(&params);
        syevjInfo_t syevj_info;
        cusolverDnCreateSyevjInfo(&syevj_info);
        if (descrA.batch_size() == 1) {
            cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, enum_convert<BackendLibrary::CUSOLVER>(uplo), descrA.rows_, BackendScalar<T,BackendLibrary::CUSOLVER>::type, descrA.data_ptr(), descrA.ld(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, eigenvalues.data(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, &l_work_device, &l_work_host);
        } else {
            #if USE_CUSOLVER_X_API
                cusolverDnXsyevBatched_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, enum_convert<BackendLibrary::CUSOLVER>(uplo), descrA.rows_, BackendScalar<T,BackendLibrary::CUSOLVER>::type, descrA.data_ptr(), descrA.ld(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, eigenvalues.data(), BackendScalar<T,BackendLibrary::CUSOLVER>::type, &l_work_device, &l_work_host, descrA.batch_size());
            #else
                int l_work_device_int = 0;
                call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                    handle, jobtype, uplo, descrA.rows(), descrA.data_ptr(), descrA.ld(), base_float_ptr_convert(eigenvalues.data()), &l_work_device_int, syevj_info, descrA.batch_size());
                l_work_device = static_cast<size_t>(l_work_device_int);
            #endif
        };

        return BumpAllocator::allocation_size<std::byte>(ctx, l_work_host) + BumpAllocator::allocation_size<std::byte>(ctx, l_work_device) + BumpAllocator::allocation_size<int>(ctx, descrA.batch_size());
    }

    #define POTRF_INSTANTIATE(fp) \
    template Event potrf<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo, \
        Span<std::byte>);
    
    #define POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t potrf_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo);

    #define SYEV_INSTANTIATE(fp) \
    template Event syev<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo, \
        Span<std::byte>);

    #define SYEV_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t syev_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo);

    #define CUSOLVER_INSTANTIATE(fp) \
        POTRF_INSTANTIATE(fp) \
        POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_INSTANTIATE(fp) \
        SYEV_BUFFER_SIZE_INSTANTIATE(fp)

    // Instantiate for the floating-point types of interest
    CUSOLVER_INSTANTIATE(float)
    CUSOLVER_INSTANTIATE(double)
    CUSOLVER_INSTANTIATE(std::complex<float>)
    CUSOLVER_INSTANTIATE(std::complex<double>)

    #undef POTRF_INSTANTIATE
    #undef POTRF_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_INSTANTIATE
    #undef SYEV_BUFFER_SIZE_INSTANTIATE
    #undef CUSOLVER_INSTANTIATE
    #undef CUSOLVER_INSTANTIATE_FOR_FP
}