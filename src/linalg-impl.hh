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
        if constexpr (B == BackendLibrary::CUBLAS) {
            return static_cast<cublasSideMode_t>(
                side == Side::Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT
            );
        } else if constexpr (B == BackendLibrary::CBLAS){
            return static_cast<CBLAS_SIDE>(
                side == Side::Left ? CblasLeft : CblasRight
            );
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

        template <template <typename, BatchType> class Handle, typename T, BatchType BT>
        auto get_effective_dims(const Handle<T,BT>& handle, Transpose trans_op){
            return trans_op == Transpose::NoTrans ? std::make_tuple(handle.rows_, handle.cols_) : std::make_tuple(handle.cols_, handle.rows_);
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

        template <typename T>
        struct BackendDenseMatrixHandle{
            #ifdef BATCHLAS_HAS_CUDA_BACKEND
              cusparseDnMatDescr_t descr_ = nullptr;
                constexpr inline operator cusparseDnMatDescr_t() const {
                    return descr_;
                }
            #endif
            
            BackendDenseMatrixHandle() = default;

            BackendDenseMatrixHandle(T* data, int rows, int cols, int ld, Layout layout) {
                #ifdef BATCHLAS_HAS_CUDA_BACKEND
                    cusparseCreateDnMat(&descr_, rows, cols, ld, data, BackendScalar<T, Backend::CUDA>::type, enum_convert<BackendLibrary::CUSPARSE>(layout));
                #endif
            }

            BackendDenseMatrixHandle(T* data, int rows, int cols, int ld, Layout layout, int stride, int batch_size) {
                #ifdef BATCHLAS_HAS_CUDA_BACKEND
                    cusparseCreateDnMat(&descr_, rows, cols, ld, data, BackendScalar<T, Backend::CUDA>::type, enum_convert<BackendLibrary::CUSPARSE>(layout));
                    cusparseDnMatSetStridedBatch(descr_, batch_size, stride);
                #endif
            }



            ~BackendDenseMatrixHandle() {
                #ifdef BATCHLAS_HAS_CUDA_BACKEND
                    if(descr_) cusparseDestroyDnMat(descr_);
                #endif
            }
        };

        template <typename T>
        struct BackendSparseMatrixHandle<T, Format::CSR>{
            #ifdef BATCHLAS_HAS_CUDA_BACKEND
              cusparseSpMatDescr_t descr_ = nullptr;
              constexpr inline operator cusparseSpMatDescr_t() const {
                  return descr_;
              }
            #endif

            BackendSparseMatrixHandle() = default;

            BackendSparseMatrixHandle(int nnz, int rows, int cols, int* row_offsets, int* col_indices, T* values, Layout layout) {
                #ifdef BATCHLAS_HAS_CUDA_BACKEND
                    cusparseCreateCsr(&descr_, rows, cols, nnz, row_offsets, col_indices, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, BackendScalar<T, Backend::CUDA>::type);
                #endif
            }

            BackendSparseMatrixHandle(int nnz, int rows, int cols, int* row_offsets, int* col_indices, T* values, Layout layout, int matrix_stride, int offset_stride, int batch_size) {
                #ifdef BATCHLAS_HAS_CUDA_BACKEND
                    cusparseCreateCsr(&descr_, rows, cols, nnz, row_offsets, col_indices, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, BackendScalar<T, Backend::CUDA>::type);
                    cusparseCsrSetStridedBatch(descr_, batch_size, offset_stride, matrix_stride);
                #endif
            }

            ~BackendSparseMatrixHandle() {
                #ifdef BATCHLAS_HAS_CUDA_BACKEND
                    if(descr_) cusparseDestroySpMat(descr_);
                #endif
            }
        };

        template <typename T>
        struct BackendDenseVectorHandle{
            #ifdef BATCHLAS_HAS_CUDA_BACKEND
              cusparseDnVecDescr_t descr_ = nullptr;
                constexpr inline operator cusparseDnVecDescr_t() const {
                    return descr_;
                }
            #endif

            BackendDenseVectorHandle() = default;

            BackendDenseVectorHandle(T* data, int size, int inc) {
                #ifdef BATCHLAS_HAS_CUDA_BACKEND
                    cusparseCreateDnVec(&descr_, size, data, BackendScalar<T, Backend::CUDA>::type);
                #endif
            }

            ~BackendDenseVectorHandle() {
                #ifdef BATCHLAS_HAS_CUDA_BACKEND
                    if(descr_) cusparseDestroyDnVec(descr_);
                #endif
            }
        };

        // Implementation of init_backend methods for vector handles
        template <typename T>
        void DenseVecHandle<T, BatchType::Single>::init_backend() {
            if (!backend_handle_) backend_handle_ = std::make_unique<BackendDenseVectorHandle<T>>(data_, size_, inc_);
        }

        template <typename T>
        void DenseVecHandle<T, BatchType::Batched>::init_backend() {
            if (!backend_handle_) backend_handle_ = std::make_unique<BackendDenseVectorHandle<T>>(data_, size_, inc_);
        }


    template <>
    struct LinalgHandle<Backend::NETLIB>{};

    template <typename T>
    void SparseMatHandle<T, Format::CSR, BatchType::Single>::init_backend(){
        if(!backend_handle_) backend_handle_ = std::make_unique<BackendSparseMatrixHandle<T, Format::CSR>>(nnz_, rows_, cols_, row_offsets_, col_indices_, data_, layout_);
    }

    template <typename T>
    void SparseMatHandle<T, Format::CSR, BatchType::Batched>::init_backend(){
        if(!backend_handle_){
            backend_handle_ = std::make_unique<BackendSparseMatrixHandle<T, Format::CSR>>(nnz_, rows_, cols_, row_offsets_, col_indices_, data_, layout_, matrix_stride_, offset_stride_, batch_size_);
        }
    }

    template <typename T>
    void DenseMatHandle<T, BatchType::Single>::init_backend(){
        if(!backend_handle_) backend_handle_ = std::make_unique<BackendDenseMatrixHandle<T>>(data_, rows_, cols_, ld_, layout_);
    }

    template <typename T>
    void DenseMatHandle<T, BatchType::Batched>::init_backend(){
        if(!backend_handle_) backend_handle_ = std::make_unique<BackendDenseMatrixHandle<T>>(data_, rows_, cols_, ld_, layout_, stride_, batch_size_);
    }

    template <typename T>
    void DenseMatView<T, BatchType::Single>::init_backend(){
        if(!backend_handle_) backend_handle_ = std::make_unique<BackendDenseMatrixHandle<T>>(data_, rows_, cols_, ld_, layout_);
    }

    template <typename T>
    void DenseMatView<T, BatchType::Batched>::init_backend(){
        if(!backend_handle_) backend_handle_ = std::make_unique<BackendDenseMatrixHandle<T>>(data_, rows_, cols_, ld_, layout_, stride_, batch_size_);
    }

    template <typename T>
    void DenseMatHandle<T, BatchType::Single>::init(Queue& ctx)  const{
        //Currently nothing to do here.
    }

    template <typename T>
    void DenseMatHandle<T, BatchType::Batched>::init(Queue& ctx) const{
        //Certain algorithms require array of pointers to data rather than a single pointer + stride
        //Here we ensure that the data_ptrs_ array is correctly populated (lazily) and done so on the device provided by the context
        //if(data_ptrs_.capacity() < batch_size_) data_ptrs_.resize(batch_size_);
        assert(data_ptrs_.size() == batch_size_);
        assert(data_ != nullptr);
        auto data_ptrs = data_ptrs_.data();
        auto data = data_;
        auto stride = stride_;
        ctx->parallel_for(batch_size_, [=](int i) {
            data_ptrs[i] = data + i * stride;
        });
    }

    template <typename T>
    void DenseMatView<T, BatchType::Single>::init(Queue& ctx) const{
       //Currently nothing to do here.
    }

    template <typename T>
    void DenseMatView<T, BatchType::Batched>::init(Queue& ctx) const {
        //Certain algorithms require array of pointers to data rather than a single pointer + stride
        //Here we ensure that the data_ptrs_ array is correctly populated (lazily) and done so on the device provided by the context
        //Allocation is done from outside to allow for persistent allocation across invocations
        assert(data_ptrs_.size() == batch_size_);
        assert(data_ != nullptr);
        auto data_ptrs = data_ptrs_.data();
        auto data = data_;
        auto stride = stride_;
        ctx->parallel_for(batch_size_, [=](int i) {
            data_ptrs[i] = data + i * stride;
        });
    }

    template <typename T>
    void SparseMatHandle<T, Format::CSR, BatchType::Batched>::init(Queue& ctx) const {
        //Currently nothing to do here.
    }

    template <typename T>
    void SparseMatHandle<T, Format::CSR, BatchType::Single>::init(Queue& ctx) const {
        //Currently nothing to do here.
    }





    template <typename T>
    SparseMatHandle<T, Format::CSR, BatchType::Single>::SparseMatHandle(T* data, int* row_offsets, int* col_indices, int nnz, int rows, int cols, Layout layout) 
        : data_(data), row_offsets_(row_offsets), col_indices_(col_indices), nnz_(nnz), rows_(rows), cols_(cols), layout_(layout) {}
    
        
    template <typename T>
    SparseMatHandle<T, Format::CSR, BatchType::Batched>::SparseMatHandle(T* data, int* row_offsets, int* col_indices, int nnz, int rows, int cols, int matrix_stride, int offset_stride, int batch_size) 
        : data_(data), row_offsets_(row_offsets), col_indices_(col_indices), nnz_(nnz), rows_(rows), cols_(cols), matrix_stride_(matrix_stride), offset_stride_(offset_stride), batch_size_(batch_size) {}
        
    template <typename T>
    DenseMatHandle<T, BatchType::Single>::DenseMatHandle(T* data, int rows, int cols, int ld) 
        : data_(data), rows_(rows), cols_(cols), ld_(ld) {}
        
    template <typename T>
    DenseMatHandle<T, BatchType::Batched>::DenseMatHandle(T* data, int rows, int cols, int ld, int stride, int batch_size) 
        : data_(data), rows_(rows), cols_(cols), ld_(ld), stride_(stride), batch_size_(batch_size), data_ptrs_(batch_size) {}

    template <typename T>
    DenseMatView<T, BatchType::Single>::DenseMatView(T* data, int rows, int cols, int ld) 
        : data_(data), rows_(rows), cols_(cols), ld_(ld) {}
        
    template <typename T>
    DenseMatView<T, BatchType::Batched>::DenseMatView(T* data, int rows, int cols, int ld, int stride, int batch_size, Span<T*> data_ptrs) 
        : data_(data), rows_(rows), cols_(cols), ld_(ld), stride_(stride), batch_size_(batch_size), data_ptrs_(data_ptrs) {}

    template <typename T>
    DenseVecHandle<T, BatchType::Batched>::DenseVecHandle(T* data, int size, int inc, int stride, int batch_size)
        : data_(data), size_(size), inc_(inc), stride_(stride), batch_size_(batch_size) {}

    template <typename T>
    DenseVecHandle<T, BatchType::Single>::DenseVecHandle(T* data, int size, int inc)
        : data_(data), size_(size), inc_(inc) {}

    
    /* template <typename T>
    DenseMatView<T, BatchType::Single>::DenseMatView(const DenseMatHandle<T, BatchType::Single>& handle) 
        : data_(handle.data_), rows_(handle.rows_), cols_(handle.cols_), ld_(handle.ld_), layout_(handle.layout_) {}

    template <typename T>
    DenseMatView<T, BatchType::Batched>::DenseMatView(const DenseMatHandle<T, BatchType::Batched>& handle) 
        : data_(handle.data_), rows_(handle.rows_), cols_(handle.cols_), ld_(handle.ld_), stride_(handle.stride_), batch_size_(handle.batch_size_), layout_(handle.layout_), data_ptrs_(handle.data_ptrs_) {} */

    template <typename T>
    DenseMatView<T, BatchType::Single>::DenseMatView(const DenseMatView<T, BatchType::Single>& view) = default;

    template <typename T>
    DenseMatView<T, BatchType::Batched>::DenseMatView(const DenseMatView<T, BatchType::Batched>& view) = default;

    template <typename T>
    DenseMatView<T, BatchType::Single>::DenseMatView(DenseMatView<T, BatchType::Single>&& view) = default;
    
    template <typename T>
    DenseMatView<T, BatchType::Batched>::DenseMatView(DenseMatView<T, BatchType::Batched>&& view) = default;

    template <typename T>
    DenseMatView<T, BatchType::Single>::DenseMatView(const DenseMatHandle<T, BatchType::Single>& handle, int rows, int cols, int ld) 
        : data_(handle.data_), rows_(rows), cols_(cols), ld_(ld), layout_(handle.layout_) {
            auto total_size_handle = handle.layout_ == Layout::RowMajor ? handle.rows_ * handle.ld_ : handle.cols_ * handle.ld_;
            auto total_size_view = layout_ == Layout::RowMajor ? rows * ld : cols * ld;
            if (total_size_handle < total_size_view) {
                throw std::runtime_error("View larger than handle: " + std::to_string(total_size_handle) + " < " + std::to_string(total_size_view));
            }
        }

    template <typename T>
    DenseMatView<T, BatchType::Batched>::DenseMatView(const DenseMatHandle<T, BatchType::Batched>& handle, int rows, int cols, int ld, int stride, int batch_size) 
        : data_(handle.data_), rows_(rows), cols_(cols), ld_(ld), stride_(stride), batch_size_(batch_size), layout_(handle.layout_), data_ptrs_(handle.data_ptrs_.to_span()) {
            auto total_size_handle = handle.stride_ * handle.batch_size_;
            auto total_size_view = stride * batch_size;
            if (total_size_handle < total_size_view) {
                throw std::runtime_error("View larger than handle: " + std::to_string(total_size_handle) + " < " + std::to_string(total_size_view));
            }
        }

    template <typename T>
    DenseMatView<T, BatchType::Single>& DenseMatView<T, BatchType::Single>::operator=(const DenseMatView<T, BatchType::Single>& view) = default;

    template <typename T>
    DenseMatView<T, BatchType::Single>& DenseMatView<T, BatchType::Single>::operator=(DenseMatView<T, BatchType::Single>&& view) = default;

    template <typename T>
    DenseMatView<T, BatchType::Batched>& DenseMatView<T, BatchType::Batched>::operator=(const DenseMatView<T, BatchType::Batched>& view) = default;

    template <typename T>
    DenseMatView<T, BatchType::Batched>& DenseMatView<T, BatchType::Batched>::operator=(DenseMatView<T, BatchType::Batched>&& view) = default;
    
    template <typename T>
    DenseMatView<T, BatchType::Batched>& DenseMatView<T, BatchType::Batched>::operator=(const DenseMatHandle<T, BatchType::Batched>& handle) {
        data_ = handle.data_;
        rows_ = handle.rows_;
        cols_ = handle.cols_;
        ld_ = handle.ld_;
        stride_ = handle.stride_;
        batch_size_ = handle.batch_size_;
        layout_ = handle.layout_;
        data_ptrs_ = handle.data_ptrs_;
        return *this;
    }

    template <typename T>
    DenseMatView<T, BatchType::Single>& DenseMatView<T, BatchType::Single>::operator=(const DenseMatHandle<T, BatchType::Single>& handle) {
        data_ = handle.data_;
        rows_ = handle.rows_;
        cols_ = handle.cols_;
        ld_ = handle.ld_;
        layout_ = handle.layout_;
        return *this;
    }


    template <typename T>
    SparseMatHandle<T, Format::CSR, BatchType::Single>::~SparseMatHandle() = default;
    template <typename T>
    SparseMatHandle<T, Format::CSR, BatchType::Batched>::~SparseMatHandle() = default;
    template <typename T>
    DenseMatHandle<T, BatchType::Single>::~DenseMatHandle() = default;
    template <typename T>
    DenseMatHandle<T, BatchType::Batched>::~DenseMatHandle() = default;
    template <typename T>
    DenseMatView<T, BatchType::Single>::~DenseMatView() = default;
    template <typename T>
    DenseMatView<T, BatchType::Batched>::~DenseMatView() = default;
    template <typename T>
    DenseVecHandle<T, BatchType::Single>::~DenseVecHandle() = default;
    template <typename T>
    DenseVecHandle<T, BatchType::Batched>::~DenseVecHandle() = default;


    template <typename T>
    BackendSparseMatrixHandle<T, Format::CSR>* SparseMatHandle<T, Format::CSR, BatchType::Single>::operator->() {
        init_backend();
        return backend_handle_.get();
    }
    template <typename T>
    BackendSparseMatrixHandle<T, Format::CSR>* SparseMatHandle<T, Format::CSR, BatchType::Batched>::operator->() {
        init_backend();
        return backend_handle_.get();
    }
    template <typename T>
    BackendDenseMatrixHandle<T>* DenseMatHandle<T, BatchType::Single>::operator->() {
        init_backend();
        return backend_handle_.get();
    }
    template <typename T>
    BackendDenseMatrixHandle<T>* DenseMatHandle<T, BatchType::Batched>::operator->() {
        init_backend();
        return backend_handle_.get();
    }
    template <typename T>
    BackendSparseMatrixHandle<T, Format::CSR>& SparseMatHandle<T, Format::CSR, BatchType::Single>::operator*() {
        init_backend();
        return *backend_handle_;
    }
    template <typename T>
    BackendSparseMatrixHandle<T, Format::CSR>& SparseMatHandle<T, Format::CSR, BatchType::Batched>::operator*() {
        init_backend();
        return *backend_handle_;
    }
    template <typename T>
    BackendDenseMatrixHandle<T>& DenseMatHandle<T, BatchType::Single>::operator*() {
        init_backend();
        return *backend_handle_;
    }
    template <typename T>
    BackendDenseMatrixHandle<T>& DenseMatHandle<T, BatchType::Batched>::operator*() {
        init_backend();
        return *backend_handle_;
    }

    template <typename T>
    BackendDenseMatrixHandle<T>& DenseMatView<T, BatchType::Single>::operator*() {
        init_backend();
        return *backend_handle_;
    }

    template <typename T>
    BackendDenseMatrixHandle<T>& DenseMatView<T, BatchType::Batched>::operator*() {
        init_backend();
        return *backend_handle_;
    }

    template <typename T>
    DenseMatView<T, BatchType::Single>::DenseMatView(const DenseMatHandle<T, BatchType::Single>& handle) 
        : data_(handle.data_), rows_(handle.rows_), cols_(handle.cols_), ld_(handle.ld_), layout_(handle.layout_) {}

    template <typename T>
    DenseMatView<T, BatchType::Batched>::DenseMatView(const DenseMatHandle<T, BatchType::Batched>& handle) 
        : data_(handle.data_), rows_(handle.rows_), cols_(handle.cols_), ld_(handle.ld_), stride_(handle.stride_), batch_size_(handle.batch_size_), layout_(handle.layout_), data_ptrs_(handle.data_ptrs_) {}

    template <typename T>
    DenseMatView<T, BatchType::Single>::DenseMatView() = default;

    template <typename T>
    DenseMatView<T, BatchType::Batched>::DenseMatView() = default;

    template <typename KernelName>
    size_t get_kernel_max_wg_size(const Queue& ctx){
        auto kernel_id = sycl::get_kernel_id<KernelName>();
        auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx -> get_context(), {kernel_id});
        auto kernel = kernel_bundle.get_kernel(kernel_id);
        return kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(ctx -> get_device());
    }


    template struct DenseMatView<float, BatchType::Single>;
    template struct DenseMatView<float, BatchType::Batched>;
    template struct DenseMatView<double, BatchType::Single>;
    template struct DenseMatView<double, BatchType::Batched>;
    template struct DenseMatView<std::complex<float>, BatchType::Single>;
    template struct DenseMatView<std::complex<float>, BatchType::Batched>;
    template struct DenseMatView<std::complex<double>, BatchType::Single>;
    template struct DenseMatView<std::complex<double>, BatchType::Batched>;

    template struct DenseMatHandle<float, BatchType::Single>;
    template struct DenseMatHandle<float, BatchType::Batched>;
    template struct DenseMatHandle<double, BatchType::Single>;
    template struct DenseMatHandle<double, BatchType::Batched>;
    template struct DenseMatHandle<std::complex<float>, BatchType::Single>;
    template struct DenseMatHandle<std::complex<float>, BatchType::Batched>;
    template struct DenseMatHandle<std::complex<double>, BatchType::Single>;
    template struct DenseMatHandle<std::complex<double>, BatchType::Batched>;

    template struct SparseMatHandle<float, Format::CSR, BatchType::Single>;
    template struct SparseMatHandle<float, Format::CSR, BatchType::Batched>;
    template struct SparseMatHandle<double, Format::CSR, BatchType::Single>;
    template struct SparseMatHandle<double, Format::CSR, BatchType::Batched>;
    template struct SparseMatHandle<std::complex<float>, Format::CSR, BatchType::Single>;
    template struct SparseMatHandle<std::complex<float>, Format::CSR, BatchType::Batched>;
    template struct SparseMatHandle<std::complex<double>, Format::CSR, BatchType::Single>;
    template struct SparseMatHandle<std::complex<double>, Format::CSR, BatchType::Batched>;

    template struct DenseVecHandle<float, BatchType::Single>;
    template struct DenseVecHandle<float, BatchType::Batched>;
    template struct DenseVecHandle<double, BatchType::Single>;
    template struct DenseVecHandle<double, BatchType::Batched>;
    template struct DenseVecHandle<std::complex<float>, BatchType::Single>;
    template struct DenseVecHandle<std::complex<float>, BatchType::Batched>;
    template struct DenseVecHandle<std::complex<double>, BatchType::Single>;
    template struct DenseVecHandle<std::complex<double>, BatchType::Batched>;
}