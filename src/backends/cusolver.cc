//#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <blas/linalg.hh>

#include <blas/functions/syev.hh>
#include <blas/dispatch/op.hh>

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

    namespace backend {
        template <Backend B, typename T>
        Event syev_vendor(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& descrA,
                          Span<typename base_type<T>::type> eigenvalues,
                          JobType jobtype,
                          Uplo uplo,
                          Span<std::byte> workspace) {
            return op_external("cusolver.syev_vendor", [&] {
                static LinalgHandle<B> handle;
                handle.setStream(ctx);
                BumpAllocator pool(workspace);
                size_t l_work_device_bytes = 0;
                size_t l_work_host_bytes = 0;
                int l_work_device_elems = 0; // legacy API returns lwork in elements of T
                cusolverDnParams_t params;
                check_status(cusolverDnCreateParams(&params));
                syevjInfo_t syevj_info;
                check_status(cusolverDnCreateSyevjInfo(&syevj_info));

                const auto eig_mode = enum_convert<BackendLibrary::CUSOLVER>(jobtype);
                const auto fill_mode = enum_convert<BackendLibrary::CUSOLVER>(uplo);
                if (descrA.batch_size() == 1) {
                    check_status(cusolverDnXsyevd_bufferSize(handle, params, eig_mode, fill_mode, descrA.rows(),
                                                            BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                            descrA.data_ptr(), descrA.ld(),
                                                            BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                            eigenvalues.data(),
                                                            BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                            &l_work_device_bytes, &l_work_host_bytes));
                } else {
                    #if USE_CUSOLVER_X_API
                        check_status(cusolverDnXsyevBatched_bufferSize(handle, params, eig_mode, fill_mode, descrA.rows(),
                                                                      BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                                      descrA.data_ptr(), descrA.ld(),
                                                                      BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                                      eigenvalues.data(),
                                                                      BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                                      &l_work_device_bytes, &l_work_host_bytes, descrA.batch_size()));
                    #else
                        call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                            handle, eig_mode, fill_mode, descrA.rows(), descrA.data_ptr(), descrA.ld(), base_float_ptr_convert(eigenvalues.data()), &l_work_device_elems, syevj_info, descrA.batch_size());
                    #endif
                }

                auto host_workspace = pool.allocate<std::byte>(ctx, l_work_host_bytes);
                auto device_workspace_bytes = pool.allocate<std::byte>(ctx, l_work_device_bytes);
                auto device_workspace_elems = pool.allocate<T>(ctx, static_cast<size_t>(l_work_device_elems));

                if (descrA.batch_size() == 1) {
                    auto info = pool.allocate<int>(ctx, 1);
                    check_status(cusolverDnXsyevd(handle,
                                                 params,
                                                 eig_mode,
                                                 fill_mode,
                                                 descrA.rows(),
                                                 BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                 descrA.data_ptr(),
                                                 descrA.ld(),
                                                 BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                 eigenvalues.data(),
                                                 BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                 device_workspace_bytes.data(),
                                                 l_work_device_bytes,
                                                 host_workspace.data(),
                                                 l_work_host_bytes,
                                                 info.data()));
                } else {
                    auto info = pool.allocate<int>(ctx, descrA.batch_size());
                    #if USE_CUSOLVER_X_API
                    check_status(cusolverDnXsyevBatched(handle,
                                                       params,
                                                       eig_mode,
                                                       fill_mode,
                                                       descrA.rows(),
                                                       BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                       descrA.data_ptr(),
                                                       descrA.ld(),
                                                       BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                       eigenvalues.data(),
                                                       BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                       device_workspace_bytes.data(),
                                                       l_work_device_bytes,
                                                       host_workspace.data(),
                                                       l_work_host_bytes,
                                                       info.data(),
                                                       descrA.batch_size()));
                    #else
                    call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsyevjBatched, cusolverDnDsyevjBatched, cusolverDnCheevjBatched, cusolverDnZheevjBatched,
                        handle, eig_mode, fill_mode, descrA.rows(), descrA.data_ptr(), descrA.ld(), base_float_ptr_convert(eigenvalues.data()), device_workspace_elems.data(), l_work_device_elems, info.data(), syevj_info, descrA.batch_size());
                    #endif
                }

                check_status(cusolverDnDestroySyevjInfo(syevj_info));
                check_status(cusolverDnDestroyParams(params));
                return ctx.get_event();
            });
        }

        template <Backend B, typename T>
        size_t syev_vendor_buffer_size(Queue& ctx,
                                       const MatrixView<T,MatrixFormat::Dense>& descrA,
                                       Span<typename base_type<T>::type> eigenvalues,
                                       JobType jobtype,
                                       Uplo uplo) {
            return op_external("cusolver.syev_vendor_buffer_size", [&] {
                static LinalgHandle<B> handle;
                handle.setStream(ctx);
                size_t l_work_device_bytes = 0;
                size_t l_work_host_bytes = 0;
                int l_work_device_elems = 0;
                cusolverDnParams_t params;
                check_status(cusolverDnCreateParams(&params));
                syevjInfo_t syevj_info;
                check_status(cusolverDnCreateSyevjInfo(&syevj_info));

                const auto eig_mode = enum_convert<BackendLibrary::CUSOLVER>(jobtype);
                const auto fill_mode = enum_convert<BackendLibrary::CUSOLVER>(uplo);
                if (descrA.batch_size() == 1) {
                    check_status(cusolverDnXsyevd_bufferSize(handle, params, eig_mode, fill_mode, descrA.rows(),
                                                            BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                            descrA.data_ptr(), descrA.ld(),
                                                            BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                            eigenvalues.data(),
                                                            BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                            &l_work_device_bytes, &l_work_host_bytes));
                } else {
                    #if USE_CUSOLVER_X_API
                        check_status(cusolverDnXsyevBatched_bufferSize(handle, params, eig_mode, fill_mode, descrA.rows(),
                                                                      BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                                      descrA.data_ptr(), descrA.ld(),
                                                                      BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                                      eigenvalues.data(),
                                                                      BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                                      &l_work_device_bytes, &l_work_host_bytes, descrA.batch_size()));
                    #else
                        call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                            handle, eig_mode, fill_mode, descrA.rows(), descrA.data_ptr(), descrA.ld(), base_float_ptr_convert(eigenvalues.data()), &l_work_device_elems, syevj_info, descrA.batch_size());
                    #endif
                };

                check_status(cusolverDnDestroySyevjInfo(syevj_info));
                check_status(cusolverDnDestroyParams(params));

                return BumpAllocator::allocation_size<std::byte>(ctx, l_work_host_bytes)
                     + BumpAllocator::allocation_size<std::byte>(ctx, l_work_device_bytes)
                     + BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(l_work_device_elems))
                     + BumpAllocator::allocation_size<int>(ctx, descrA.batch_size());
            });
        }
    } // namespace backend

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

    #define SYEV_VENDOR_INSTANTIATE(fp) \
    template Event backend::syev_vendor<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo, \
        Span<std::byte>);

    #define SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t backend::syev_vendor_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo);

    #define CUSOLVER_INSTANTIATE(fp) \
        POTRF_INSTANTIATE(fp) \
        POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_INSTANTIATE(fp) \
        SYEV_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_VENDOR_INSTANTIATE(fp) \
        SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE(fp)

    // Instantiate for the floating-point types of interest
    CUSOLVER_INSTANTIATE(float)
    CUSOLVER_INSTANTIATE(double)
    CUSOLVER_INSTANTIATE(std::complex<float>)
    CUSOLVER_INSTANTIATE(std::complex<double>)

    #undef POTRF_INSTANTIATE
    #undef POTRF_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_INSTANTIATE
    #undef SYEV_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_VENDOR_INSTANTIATE
    #undef SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE
    #undef CUSOLVER_INSTANTIATE
    #undef CUSOLVER_INSTANTIATE_FOR_FP
}