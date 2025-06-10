#pragma once
#include <blas/linalg.hh>
#include "queue.hh"
#include <sycl/sycl.hpp>
#include <util/mempool.hh>
#include <execution>

#ifdef BATCHLAS_HAS_CUDA_BACKEND
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
    #include <cublas_v2.h>
    #include <cusparse.h>
    #include <cusolverDn.h>
#endif

#ifdef BATCHLAS_HAS_HOST_BACKEND
    #include <cblas.h>
    #include <lapacke.h>
#endif

#ifdef USE_ROCM
    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>
    #include <rocblas.h>
#endif

#ifdef USE_MAGMA
    #include <magma_v2.h>
#endif

#ifdef USE_MKL
    #include <mkl.h>
#include "linalg.hh"
#endif



namespace batchlas{
    template <typename T, Backend B>
    struct BackendScalar;

    template <typename T, ComputePrecision P, Backend B>
    struct BlasComputeType;

    template <Transpose T, Backend B>
    struct BackendTranspose;

    template <Backend B>
    struct LinalgHandle;


    // Helper type traits to identify our enum types
    template<typename T>
    struct is_linalg_enum : std::false_type {};
    
    template<> struct is_linalg_enum<Side> : std::true_type {};
    template<> struct is_linalg_enum<Uplo> : std::true_type {};
    template<> struct is_linalg_enum<Transpose> : std::true_type {};
    template<> struct is_linalg_enum<Diag> : std::true_type {};
    template<> struct is_linalg_enum<Layout> : std::true_type {};
    template<> struct is_linalg_enum<JobType> : std::true_type {};

    template<class T>
    struct is_complex_or_floating_point : std::is_floating_point<T> { };

    template<class T>
    struct is_complex_or_floating_point<std::complex<T>> : std::is_floating_point<T> { };

    // Individual enum conversions for CUDA backend
    template<BackendLibrary B>
    constexpr auto enum_convert(Side side) {
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSOLVER) {
            return static_cast<cublasSideMode_t>(
                side == Side::Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT
            );
        } else if constexpr (B == BackendLibrary::CBLAS){
            return static_cast<CBLAS_SIDE>(
                side == Side::Left ? CblasLeft : CblasRight
            );
        } else if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                side == Side::Left ? 'L' : 'R'
            );
        } else {
            static_assert(false, "Unsupported backend for Side conversion");
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(JobType job) {
        if constexpr (B == BackendLibrary::CUSOLVER) {
            return static_cast<cusolverEigMode_t>(
                job == JobType::EigenVectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR
            );
        } else if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                job == JobType::EigenVectors ? 'V' : 'N'
            );
        } else {
            static_assert(false, "Unsupported backend for JobType conversion");
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(Diag diag) {
        if constexpr (B == BackendLibrary::CUBLAS) {
            return static_cast<cublasDiagType_t>(
                diag == Diag::NonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT
            );
        } else if constexpr (B == BackendLibrary::CBLAS) {
            return static_cast<CBLAS_DIAG>(
                diag == Diag::NonUnit ? CblasNonUnit : CblasUnit
            );
        } else if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                diag == Diag::NonUnit ? 'N' : 'U'
            );
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(Layout layout) {
        if constexpr (B == BackendLibrary::CUSPARSE) {
            return static_cast<cusparseOrder_t>(
                layout == Layout::RowMajor ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL
            );
        } else if constexpr (B == BackendLibrary::CBLAS) {
            return static_cast<CBLAS_LAYOUT>(
                layout == Layout::RowMajor ? CblasRowMajor : CblasColMajor
            );
        } else if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<lapack_int>(
                layout == Layout::RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR
            );
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(Uplo uplo) {
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSOLVER) {
            return static_cast<cublasFillMode_t>(
                uplo == Uplo::Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER
            );
        } else if constexpr (B == BackendLibrary::CUSPARSE) {
            return static_cast<cusparseFillMode_t>(
                uplo == Uplo::Upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER
            );
        } else if constexpr (B == BackendLibrary::CBLAS) {
            return static_cast<CBLAS_UPLO>(
                uplo == Uplo::Upper ? CblasUpper : CblasLower
            );
        } else if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                uplo == Uplo::Upper ? 'U' : 'L'
            );
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(Transpose trans) {
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSOLVER) {
            return static_cast<cublasOperation_t>(
                trans == Transpose::NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T
            );
        } else if constexpr (B == BackendLibrary::CUSPARSE) {
            return static_cast<cusparseOperation_t>(
                trans == Transpose::NoTrans ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE
            );
        } else if constexpr (B == BackendLibrary::CBLAS) {
            return static_cast<CBLAS_TRANSPOSE>(
                trans == Transpose::NoTrans ? CblasNoTrans : CblasTrans
            );
        } else if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                trans == Transpose::NoTrans ? 'N' : 'T'
            );
        }
    }

    template<BackendLibrary B, typename T>
    constexpr auto enum_convert(ComputePrecision precision) {
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSOLVER || B == BackendLibrary::CUSPARSE) {
            using BaseType = typename base_type<T>::type;
            
            // Handle default precision first
            if (precision == ComputePrecision::Default) {
                if constexpr (std::is_same_v<BaseType, float>) {
                    return CUBLAS_COMPUTE_32F;
                } else if constexpr (std::is_same_v<BaseType, double>) {
                    return CUBLAS_COMPUTE_64F;
                }
            }

            // Handle specific precision requests
            if constexpr (std::is_same_v<BaseType, float>) {
                switch (precision) {
                    case ComputePrecision::F32:  return CUBLAS_COMPUTE_32F;
                    case ComputePrecision::F16:  return CUBLAS_COMPUTE_32F_FAST_16F;
                    case ComputePrecision::BF16: return CUBLAS_COMPUTE_32F_FAST_16BF;
                    case ComputePrecision::TF32: return CUBLAS_COMPUTE_32F_FAST_TF32;
                    default:
                        throw std::runtime_error("Unsupported precision for single precision type");
                }
            } 
            else if constexpr (std::is_same_v<BaseType, double>) {
                if (precision == ComputePrecision::F64) {
                    return CUBLAS_COMPUTE_64F;
                }
                throw std::runtime_error("Only F64 precision supported for double precision type");
            }
        }
        throw std::runtime_error("Unsupported backend or type combination");
    }

    template<BackendLibrary B, typename T>
    constexpr auto ptr_convert(T** ptr) {
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSPARSE || B == BackendLibrary::CUSOLVER) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return ptr; // No conversion needed
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<cuComplex**>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<cuDoubleComplex**>(ptr);
            }
        } else if constexpr (B == BackendLibrary::CBLAS) {
            if constexpr (std::is_same_v<T, float>) {
                return reinterpret_cast<float**>(ptr);
            } else if constexpr (std::is_same_v<T, double>) {
                return reinterpret_cast<double**>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<void**>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<void**>(ptr);
            }
        }
        
        else {
            static_assert(false, "Unsupported backend or type combination");
        }
    }

    


    template<BackendLibrary B, typename T>
    constexpr auto ptr_convert(T* ptr) {
        static_assert(std::is_floating_point<T>::value || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>, "Type must be floating point or complex");
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSPARSE || B == BackendLibrary::CUSOLVER) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return ptr; // No conversion needed
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<cuComplex*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<cuDoubleComplex*>(ptr);
            }
        } else if constexpr (B == BackendLibrary::CBLAS) {
            if constexpr (std::is_same_v<T, float>) {
                return reinterpret_cast<float*>(ptr);
            } else if constexpr (std::is_same_v<T, double>) {
                return reinterpret_cast<double*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<void*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<void*>(ptr);
            }
        } else if constexpr (B == BackendLibrary::LAPACKE) {
            if constexpr (std::is_same_v<T, float>) {
                return reinterpret_cast<float*>(ptr);
            } else if constexpr (std::is_same_v<T, double>) {
                return reinterpret_cast<double*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<lapack_complex_float*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<lapack_complex_double*>(ptr);
            }
        } else {
            static_assert(false, "Unsupported backend or type combination");
        }
    }

    // Const pointer version
    template<BackendLibrary B, typename T>
    constexpr auto ptr_convert(const T* ptr) {
        static_assert(std::is_floating_point<T>::value || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>, "Type must be floating point or complex");
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSPARSE || B == BackendLibrary::CUSOLVER) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return ptr; // No conversion needed
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<const cuComplex*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<const cuDoubleComplex*>(ptr);
            }
        } else {
            static_assert(false, "Unsupported backend or type combination");
        }
    }

    template<BackendLibrary B, typename T>
    constexpr auto float_convert(T& val) {
        static_assert(is_complex_or_floating_point<T>::value, "Type must be floating point or complex");
        if constexpr (B == BackendLibrary::CBLAS) {
            if constexpr (std::is_same_v<T, float>) {
                return static_cast<float>(val);
            } else if constexpr (std::is_same_v<T, double>) {
                return static_cast<double>(val);
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return static_cast<void*>(&val);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return static_cast<void*>(&val);
            }
        } else {
            return val; // No conversion needed
        }
    }

    template <typename T>
    constexpr auto base_float_ptr_convert(T* ptr) {
        if constexpr (std::is_same_v<T, float>) {
            return reinterpret_cast<float*>(ptr);
        } else if constexpr (std::is_same_v<T, double>) {
            return reinterpret_cast<double*>(ptr);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            return reinterpret_cast<float*>(ptr);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            return reinterpret_cast<double*>(ptr);
        }
    }

    /* template<Backend B, typename T>
    constexpr auto ptr_convert(T* ptr) {
        if constexpr (B == Backend::CUDA) {
            using BaseT = std::remove_const_t<T>;
            using CudaT = std::conditional_t<std::is_same_v<BaseT, float> || std::is_same_v<BaseT, double>,
                                           BaseT,
                                           std::conditional_t<std::is_same_v<BaseT, std::complex<float>>,
                                                            cuComplex,
                                                            cuDoubleComplex>>;
            using ReturnT = std::conditional_t<std::is_const_v<T>,
                                             const CudaT*,
                                             CudaT*>;
            
            if constexpr (std::is_same_v<BaseT, float> || std::is_same_v<BaseT, double>) {
                return ptr; // No conversion needed
            } else {
                return reinterpret_cast<ReturnT>(ptr);
            }
        }
    } */

    // Variadic template for converting multiple pointers
    namespace detail {
        template <BackendLibrary B, typename T>
        constexpr auto convert_arg(T&& arg) {
            if constexpr (is_linalg_enum<std::remove_const_t<std::remove_reference_t<T>>>::value) {
                return enum_convert<B>(std::forward<T>(arg));
            } else if constexpr (std::is_integral_v<std::remove_reference_t<T>>) {
                return std::forward<T>(arg);
            } else if constexpr (std::is_integral_v<std::remove_pointer_t<std::remove_reference_t<T>>>) {
                return std::forward<T>(arg);
            } else if constexpr (std::is_pointer_v<std::remove_reference_t<T>> && is_complex_or_floating_point<std::remove_pointer_t<std::remove_pointer_t<std::remove_reference_t<T>>>>::value) {
                return ptr_convert<B>(std::forward<T>(arg));
            } else if constexpr (is_complex_or_floating_point<std::remove_reference_t<T>>::value) {
                return float_convert<B>(std::forward<T>(arg));
            } else {
                return std::forward<T>(arg);
            }
        }

        template <typename F, typename Tuple, std::size_t... I>
        auto apply_tuple_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
            return f(std::get<I>(std::forward<Tuple>(t))...);
        }

        template <typename F, typename Tuple>
        auto apply_tuple(F&& f, Tuple&& t) {
            return apply_tuple_impl(
                std::forward<F>(f),
                std::forward<Tuple>(t),
                std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{}
            );
        }
    }

    // Combined enum and pointer conversion
    template <BackendLibrary B, typename... Args>
    auto backend_convert(Args&&... args) {
        return std::make_tuple(detail::convert_arg<B>(std::forward<Args>(args))...);
    }

    inline auto check_status(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("CUBLAS error: " + std::to_string(status));
        }
        return status;
    }

    inline auto check_status(cusparseStatus_t status) {
        if (status != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("CUSPARSE error: " + std::to_string(status));
        }
        return status;
    }

    inline auto check_status(cusolverStatus_t status) {
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("CUSOLVER error: " + std::to_string(status));
        }
        return status;
    }
    
    template <typename T, BackendLibrary BL, Backend B, typename Fun1, typename Fun2, typename Fun3, typename Fun4, typename... Args>
    auto call_backend(const Fun1& fun1, const Fun2& fun2, const Fun3& fun3, const Fun4& fun4, const LinalgHandle<B>& handle, Args&&... args) {
        if constexpr (std::is_same_v<T,float>) {
            return check_status(std::apply(fun1, std::tuple_cat(std::forward_as_tuple(handle), backend_convert<BL>(std::forward<Args>(args)...))));
        } else if constexpr (std::is_same_v<T,double>) {
            return check_status(std::apply(fun2, std::tuple_cat(std::forward_as_tuple(handle), backend_convert<BL>(std::forward<Args>(args)...))));
        } else if constexpr (std::is_same_v<T,std::complex<float>>) {
            return check_status(std::apply(fun3, std::tuple_cat(std::forward_as_tuple(handle), backend_convert<BL>(std::forward<Args>(args)...))));
        } else if constexpr (std::is_same_v<T,std::complex<double>>) {
            return check_status(std::apply(fun4, std::tuple_cat(std::forward_as_tuple(handle), backend_convert<BL>(std::forward<Args>(args)...))));
        }
    }

    template <typename T, BackendLibrary BL, typename Fun1, typename Fun2, typename Fun3, typename Fun4, typename... Args>
    void call_backend_nh(const Fun1& fun1, const Fun2& fun2, const Fun3& fun3, const Fun4& fun4, Args&&... args) {
        if constexpr (std::is_same_v<T,float>) {
            std::apply(fun1, backend_convert<BL>(std::forward<Args>(args)...));
        } else if constexpr (std::is_same_v<T,double>) {
            std::apply(fun2, backend_convert<BL>(std::forward<Args>(args)...));
        } else if constexpr (std::is_same_v<T,std::complex<float>>) {
            std::apply(fun3, backend_convert<BL>(std::forward<Args>(args)...));
        } else if constexpr (std::is_same_v<T,std::complex<double>>) {
            std::apply(fun4, backend_convert<BL>(std::forward<Args>(args)...));
        }
    }


    // Variadic template for converting multiple enums
/*     template <Backend B, typename... Args>
    auto enum_convert(Args&&... args) {
        static_assert((is_linalg_enum<std::remove_reference_t<Args>>::value && ...), 
                     "All arguments must be linalg enum types");
        return std::make_tuple(enum_convert<B>(std::forward<Args>(args))...);
    }
 */


        


    #ifdef BATCHLAS_HAS_CUDA_BACKEND
        template <>
        struct BackendScalar<float, Backend::CUDA> {
            static constexpr cudaDataType_t type = CUDA_R_32F;
        };

        template <>
        struct BackendScalar<double, Backend::CUDA> {
            static constexpr cudaDataType_t type = CUDA_R_64F;
        };

        template <>
        struct BackendScalar<std::complex<float>, Backend::CUDA> {
            static constexpr cudaDataType_t type = CUDA_C_32F;
        };

        template <>
        struct BackendScalar<std::complex<double>, Backend::CUDA> {
            static constexpr cudaDataType_t type = CUDA_C_64F;
        };

        template <typename T>
        struct BlasComputeType<T, ComputePrecision::Default, Backend::CUDA> {
            static constexpr cublasComputeType_t type = (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_64F;
        };


        /* template <>
        struct BackendTranspose<Transpose::NoTrans, Layout::ColMajor, Backend::CUDA> {
            static constexpr cublasOperation_t type = CUBLAS_OP_N;
        };

        template <>
        struct BackendTranspose<Transpose::NoTrans, Layout::RowMajor, Backend::CUDA> {
            static constexpr cublasOperation_t type = CUBLAS_OP_T;
        };  

        template <>
        struct BackendTranspose<Transpose::Trans, Layout::ColMajor, Backend::CUDA> {
            static constexpr cublasOperation_t type = CUBLAS_OP_T;
        };

        template <>
        struct BackendTranspose<Transpose::Trans, Layout::RowMajor, Backend::CUDA> {
            static constexpr cublasOperation_t type = CUBLAS_OP_N;
        }; */

        inline cublasOperation_t backendTransposeOp(Transpose t) {
            return (t == Transpose::NoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
        }

        template <>
        struct LinalgHandle<Backend::CUDA> {
            cublasHandle_t blas_handle_;
            cusparseHandle_t sparse_handle_;
            cusolverDnHandle_t solver_handle_;

            LinalgHandle() {
                cudaDeviceSynchronize();
                auto blas_status = cublasCreate(&blas_handle_);
                if (blas_status != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "CUBLAS initialization failed with status: " << blas_status << std::endl;
                    throw std::runtime_error("CUBLAS initialization failed");
                }

                auto sparse_status = cusparseCreate(&sparse_handle_);
                if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
                    std::cerr << "CUSPARSE initialization failed with status: " << sparse_status << std::endl;
                    throw std::runtime_error("CUSPARSE initialization failed");
                }

                auto solver_status = cusolverDnCreate(&solver_handle_);
                if (solver_status != CUSOLVER_STATUS_SUCCESS) {
                    std::cerr << "CUSOLVER initialization failed with status: " << solver_status << std::endl;
                    throw std::runtime_error("CUSOLVER initialization failed");
                }
                cudaDeviceSynchronize();
            }

            LinalgHandle(const LinalgHandle&) = delete;
            LinalgHandle& operator=(const LinalgHandle&) = delete;

            LinalgHandle(LinalgHandle&& other) = delete;
            LinalgHandle& operator=(LinalgHandle&& other) = delete;


            ~LinalgHandle() {
                cudaDeviceSynchronize();
                auto blas_status = cublasDestroy(blas_handle_);
                if (blas_status != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "CUBLAS initialization failed with status: " << blas_status << std::endl;
                }
                auto sparse_status = cusparseDestroy(sparse_handle_);
                if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
                    std::cerr << "CUSPARSE initialization failed with status: " << sparse_status << std::endl;
                }
                auto solver_status = cusolverDnDestroy(solver_handle_);
                if (solver_status != CUSOLVER_STATUS_SUCCESS) {
                    std::cerr << "CUSOLVER initialization failed with status: " << solver_status << std::endl;
                }
                cudaDeviceSynchronize();
            }

            constexpr inline operator cublasHandle_t() const {
                return blas_handle_;
            }

            constexpr inline operator cusparseHandle_t() const {
                return sparse_handle_;
            }

            constexpr inline operator cusolverDnHandle_t() const {
                return solver_handle_;
            }

            void setStream(const Queue& ctx) {
                cudaStream_t stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*ctx);
                cudaStreamSynchronize(stream);
                cublasSetStream(blas_handle_, stream);
                cusparseSetStream(sparse_handle_, stream);
                cusolverDnSetStream(solver_handle_, stream);
                cudaStreamSynchronize(stream);
            }
        };

    #endif


    template <>
    struct LinalgHandle<Backend::NETLIB>{};


    template <typename KernelName>
    size_t get_kernel_max_wg_size(const Queue& ctx){
        auto kernel_id = sycl::get_kernel_id<KernelName>();
        auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx -> get_context(), {kernel_id});
        auto kernel = kernel_bundle.get_kernel(kernel_id);
        return kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(ctx -> get_device());
    }
}