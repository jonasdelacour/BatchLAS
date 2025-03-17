//#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>

// This file contains cuSOLVER primitives implementation
namespace batchlas {

    template <Backend B, typename T, BatchType BT>
    size_t potrf_buffer_size(Queue& ctx,
                            DenseMatView<T,BT> A,
                            Uplo uplo) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        int size = 0;
        if constexpr (BT == BatchType::Single) {
            call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSpotrf_bufferSize, cusolverDnDpotrf_bufferSize, cusolverDnCpotrf_bufferSize, cusolverDnZpotrf_bufferSize,
                handle, uplo, A.rows_, get_data(A), A.ld_, &size);
            size = BumpAllocator::allocation_size<std::byte>(ctx, size) + BumpAllocator::allocation_size<int>(ctx, 1);
        } else {
            size =  BumpAllocator::allocation_size<int>(ctx, A.batch_size_);
        }
        return size;
    }

    template <Backend B, typename T, BatchType BT>
    Event potrf(Queue& ctx,
                    DenseMatView<T,BT> descrA,
                    Uplo uplo,
                    Span<std::byte> workspace) {        
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto Lwork = potrf_buffer_size<B>(ctx, descrA, uplo) - BumpAllocator::allocation_size<int>(ctx, 1);
        
        if constexpr (BT == BatchType::Single) {
            auto potrf_span = pool.allocate<std::byte>(ctx, Lwork);
            auto info = pool.allocate<int>(ctx, 1);
            auto status = call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSpotrf, cusolverDnDpotrf, cusolverDnCpotrf, cusolverDnZpotrf,
                handle, uplo, descrA.rows_, get_data(descrA), descrA.ld_, reinterpret_cast<T*>(potrf_span.data()), Lwork, info.data());
        } else {
            auto info = pool.allocate<int>(ctx, descrA.batch_size_);
            call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSpotrfBatched, cusolverDnDpotrfBatched, cusolverDnCpotrfBatched, cusolverDnZpotrfBatched,
                handle, uplo, descrA.rows_, get_ptr_arr(ctx, descrA), descrA.ld_, info.data(), descrA.batch_size_);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T, BatchType BT>
    Event syev(Queue& ctx,
                   DenseMatView<T,BT> descrA,
                   Span<T> eigenvalues,
                   Uplo uplo,
                   Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);

        size_t l_work_device;
        size_t l_work_host;
        
        cusolverDnParams_t params;
        cusolverDnCreateParams(&params);
        if constexpr (BT == BatchType::Single){
            cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, enum_convert<BackendLibrary::CUSOLVER>(uplo), descrA.rows_, BackendScalar<T,B>::type, descrA.data_, descrA.ld_, BackendScalar<T,B>::type, eigenvalues.data(), BackendScalar<T,B>::type, &l_work_device, &l_work_host);
        } else {
            cusolverDnXsyevBatched_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, enum_convert<BackendLibrary::CUSOLVER>(uplo), descrA.rows_, BackendScalar<T,B>::type, descrA.data_, descrA.ld_, BackendScalar<T,B>::type, eigenvalues.data(), BackendScalar<T,B>::type, &l_work_device, &l_work_host, descrA.batch_size_);
        }
        
        auto host_workspace = pool.allocate<std::byte>(ctx, l_work_host);
        auto device_workspace = pool.allocate<std::byte>(ctx, l_work_device);
                                                      
        if constexpr (BT == BatchType::Single) {
            int info;
            cusolverDnXsyevd(
                handle,
                params,
                CUSOLVER_EIG_MODE_VECTOR,
                enum_convert<BackendLibrary::CUSOLVER>(uplo),
                descrA.rows_,
                BackendScalar<T,B>::type,
                get_data(descrA),
                descrA.ld_,
                BackendScalar<T,B>::type,
                eigenvalues.data(),
                BackendScalar<T,B>::type,
                device_workspace.data(),
                l_work_device,
                host_workspace.data(),
                l_work_host,
                &info);
        } else {
            auto info = pool.allocate<int>(ctx, descrA.batch_size_);
            cusolverDnXsyevBatched(
                handle,
                params,
                CUSOLVER_EIG_MODE_VECTOR,
                enum_convert<BackendLibrary::CUSOLVER>(uplo),
                descrA.rows_,
                BackendScalar<T,B>::type,
                descrA.data_,
                descrA.ld_,
                BackendScalar<T,B>::type,
                eigenvalues.data(),
                BackendScalar<T,B>::type,
                device_workspace.data(),
                l_work_device,
                host_workspace.data(),
                l_work_host,
                info.data(),
                descrA.batch_size_);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T, BatchType BT>
    size_t syev_buffer_size(Queue& ctx,
                            DenseMatView<T,BT> descrA,                            
                            Span<T> eigenvalues,
                            Uplo uplo) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        size_t l_work_device;
        size_t l_work_host;
        size_t total = 0;
        cusolverDnParams_t params;
        cusolverDnCreateParams(&params);
        if constexpr (BT == BatchType::Single){
            cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, enum_convert<BackendLibrary::CUSOLVER>(uplo), descrA.rows_, BackendScalar<T,B>::type, descrA.data_, descrA.ld_, BackendScalar<T,B>::type, eigenvalues.data(), BackendScalar<T,B>::type, &l_work_device, &l_work_host);
        } else {
            cusolverDnXsyevBatched_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, enum_convert<BackendLibrary::CUSOLVER>(uplo), descrA.rows_, BackendScalar<T,B>::type, descrA.data_, descrA.ld_, BackendScalar<T,B>::type, eigenvalues.data(), BackendScalar<T,B>::type, &l_work_device, &l_work_host, descrA.batch_size_);
            total = BumpAllocator::allocation_size<int>(ctx, descrA.batch_size_);
        }

        return BumpAllocator::allocation_size<std::byte>(ctx, l_work_host) + BumpAllocator::allocation_size<std::byte>(ctx, l_work_device) + total;
    }

    #define POTRF_INSTANTIATE(fp, BT) \
    template Event potrf<Backend::CUDA, fp, BT>( \
        Queue&, \
        DenseMatView<fp, BT>, \
        Uplo, \
        Span<std::byte>);
    
    #define POTRF_BUFFER_SIZE_INSTANTIATE(fp, BT) \
    template size_t potrf_buffer_size<Backend::CUDA, fp, BT>( \
        Queue&, \
        DenseMatView<fp, BT>, \
        Uplo);

    #define SYEV_INSTANTIATE(fp, BT) \
    template Event syev<Backend::CUDA, fp, BT>( \
        Queue&, \
        DenseMatView<fp, BT>, \
        Span<fp>, \
        Uplo, \
        Span<std::byte>);

    #define SYEV_BUFFER_SIZE_INSTANTIATE(fp, BT) \
    template size_t syev_buffer_size<Backend::CUDA, fp, BT>( \
        Queue&, \
        DenseMatView<fp, BT>, \
        Span<fp>, \
        Uplo);

    #define CUSOLVER_INSTANTIATE(fp, BT) \
        POTRF_INSTANTIATE(fp, BT) \
        POTRF_BUFFER_SIZE_INSTANTIATE(fp, BT) \
        SYEV_INSTANTIATE(fp, BT) \
        SYEV_BUFFER_SIZE_INSTANTIATE(fp, BT)

    // Instantiate for all floating-point and batch type combinations
    #define CUSOLVER_INSTANTIATE_FOR_FP(fp) \
        CUSOLVER_INSTANTIATE(fp, BatchType::Batched) \
        CUSOLVER_INSTANTIATE(fp, BatchType::Single)

    // Instantiate for the floating-point types of interest
    CUSOLVER_INSTANTIATE_FOR_FP(float)
    CUSOLVER_INSTANTIATE_FOR_FP(double)
    CUSOLVER_INSTANTIATE_FOR_FP(std::complex<float>)
    CUSOLVER_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef POTRF_INSTANTIATE
    #undef POTRF_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_INSTANTIATE
    #undef SYEV_BUFFER_SIZE_INSTANTIATE
    #undef CUSOLVER_INSTANTIATE
    #undef CUSOLVER_INSTANTIATE_FOR_FP
}