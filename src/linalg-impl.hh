#pragma once
#include <batchlas/blas/linalg.hh>
#include "queue.hh"
#include <sycl/sycl.hpp>
#include <batchlas/util/mempool.hh>
#include <execution>
#include <type_traits>
#include <memory>

// Math-library includes are keyed on the LIBRARY axis, never the device family; the CUDA runtime
// stays on the family flag. evidence: docs/design/vendor-independence.md#vendor-independence-headers-keyed-on-the-library-axis
#if BATCHLAS_HAS_CUDA_BACKEND
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
    // cuComplex / cuDoubleComplex are CUDA *runtime* types, used by the
    // pointer-cast helpers below. They used to arrive transitively through
    // cublas_v2.h, which made those helpers silently depend on cuBLAS.
    #include <cuComplex.h>
#endif
#if BATCHLAS_HAS_CUBLAS
    #include <cublas_v2.h>
#endif
#if BATCHLAS_HAS_CUSPARSE
    #include <cusparse.h>
#endif
#if BATCHLAS_HAS_CUSOLVER
    #include <cusolverDn.h>
#endif

#if BATCHLAS_HAS_LAPACKE
    #include <lapacke.h>
#endif
#if BATCHLAS_HAS_CBLAS
    #include <cblas.h>
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
    #define  __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #include <rocblas/rocblas.h>
    #include <rocsparse/rocsparse.h>
    #include <rocsolver/rocsolver.h>
#endif

#include <batchlas/blas/linalg.hh>


namespace batchlas{
    template <typename T, BackendLibrary B>
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

    template <class T>
    struct is_complex_or_floating_point : std::bool_constant<FloatingOrComplexScalar<T>> {};

    template<auto>
    struct always_false : std::false_type {};

    // Individual enum conversions for CUDA backend
    template<BackendLibrary B>
    constexpr auto enum_convert(Side side) {
#if BATCHLAS_HAS_CUBLAS
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSOLVER) {
            return static_cast<cublasSideMode_t>(
                side == Side::Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT
            );
        } else
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCBLAS || B == BackendLibrary::ROCSOLVER) {
            return static_cast<rocblas_side>(side == Side::Left ? rocblas_side_left : rocblas_side_right);
        } else
#endif
#if BATCHLAS_HAS_CBLAS
        if constexpr (B == BackendLibrary::CBLAS){
            return static_cast<CBLAS_SIDE>(
                side == Side::Left ? CblasLeft : CblasRight
            );
        } else
#endif
#if BATCHLAS_HAS_LAPACKE
        if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                side == Side::Left ? 'L' : 'R'
            );
        } else
#endif
        {
            static_assert(always_false<B>::value, "Unsupported backend for Side conversion");
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(JobType job) {
#if BATCHLAS_HAS_CUSOLVER
        if constexpr (B == BackendLibrary::CUSOLVER) {
            return static_cast<cusolverEigMode_t>(
                job == JobType::EigenVectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR
            );
        } else
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCSOLVER) {
            return job == JobType::EigenVectors ? rocblas_evect_original : rocblas_evect_none;
        } else
#endif
#if BATCHLAS_HAS_HOST_BACKEND
        if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                job == JobType::EigenVectors ? 'V' : 'N'
            );
        } else
#endif
        {
            static_assert(always_false<B>::value, "Unsupported backend for JobType conversion");
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(Diag diag) {
#if BATCHLAS_HAS_CUBLAS
        if constexpr (B == BackendLibrary::CUBLAS) {
            return static_cast<cublasDiagType_t>(
                diag == Diag::NonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT
            );
        } else
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCBLAS) {
            return diag == Diag::NonUnit ? rocblas_diagonal_non_unit : rocblas_diagonal_unit;
        } else
#endif
#if BATCHLAS_HAS_CBLAS
        if constexpr (B == BackendLibrary::CBLAS) {
            return static_cast<CBLAS_DIAG>(
                diag == Diag::NonUnit ? CblasNonUnit : CblasUnit
            );
        } else
#endif
#if BATCHLAS_HAS_LAPACKE
        if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                diag == Diag::NonUnit ? 'N' : 'U'
            );
        } else
#endif
        {
            static_assert(always_false<B>::value, "Unsupported backend for Diag conversion");
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(Layout layout) {
#if BATCHLAS_HAS_CUSPARSE
        if constexpr (B == BackendLibrary::CUSPARSE) {
            return static_cast<cusparseOrder_t>(
                layout == Layout::RowMajor ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL
            );
        } else
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCSPARSE) {
            return static_cast<rocsparse_order>(layout == Layout::RowMajor ? rocsparse_order_row : rocsparse_order_column);
        } else
#endif
#if BATCHLAS_HAS_CBLAS
        if constexpr (B == BackendLibrary::CBLAS) {
            return static_cast<CBLAS_LAYOUT>(
                layout == Layout::RowMajor ? CblasRowMajor : CblasColMajor
            );
        } else
#endif
#if BATCHLAS_HAS_LAPACKE
        if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<lapack_int>(
                layout == Layout::RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR
            );
        } else
#endif
        {
            static_assert(always_false<B>::value, "Unsupported backend for Layout conversion");
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(Uplo uplo) {
#if BATCHLAS_HAS_CUBLAS
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSOLVER) {
            return static_cast<cublasFillMode_t>(
                uplo == Uplo::Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER
            );
        } else
#endif
#if BATCHLAS_HAS_CUSPARSE
        if constexpr (B == BackendLibrary::CUSPARSE) {
            return static_cast<cusparseFillMode_t>(
                uplo == Uplo::Upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER
            );
        } else
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCBLAS || B == BackendLibrary::ROCSOLVER) {
            return uplo == Uplo::Upper ? rocblas_fill_upper : rocblas_fill_lower;
        } else if constexpr (B == BackendLibrary::ROCSPARSE) {
            return uplo == Uplo::Upper ? rocsparse_fill_mode_upper : rocsparse_fill_mode_lower;
        } else
#endif
#if BATCHLAS_HAS_CBLAS
        if constexpr (B == BackendLibrary::CBLAS) {
            return static_cast<CBLAS_UPLO>(
                uplo == Uplo::Upper ? CblasUpper : CblasLower
            );
        } else
#endif
#if BATCHLAS_HAS_LAPACKE
        if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                uplo == Uplo::Upper ? 'U' : 'L'
            );
        } else
#endif
        {
            static_assert(always_false<B>::value, "Unsupported backend for Uplo conversion");
        }
    }

    template<BackendLibrary B>
    constexpr auto enum_convert(Transpose trans) {
#if BATCHLAS_HAS_CUBLAS
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSOLVER) {
            switch (trans) {
                case Transpose::NoTrans: return static_cast<cublasOperation_t>(CUBLAS_OP_N);
                case Transpose::Trans: return static_cast<cublasOperation_t>(CUBLAS_OP_T);
                case Transpose::ConjTrans: return static_cast<cublasOperation_t>(CUBLAS_OP_C);
            }
        } else
#endif
#if BATCHLAS_HAS_CUSPARSE
        if constexpr (B == BackendLibrary::CUSPARSE) {
            switch (trans) {
                case Transpose::NoTrans: return static_cast<cusparseOperation_t>(CUSPARSE_OPERATION_NON_TRANSPOSE);
                case Transpose::Trans: return static_cast<cusparseOperation_t>(CUSPARSE_OPERATION_TRANSPOSE);
                case Transpose::ConjTrans: return static_cast<cusparseOperation_t>(CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
            }
        } else
#endif
    #if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCBLAS || B == BackendLibrary::ROCSOLVER) {
            //return trans == Transpose::NoTrans ? rocblas_operation_none : rocblas_operation_transpose;
            switch (trans) {
                case Transpose::NoTrans: return rocblas_operation_none;
                case Transpose::Trans: return rocblas_operation_transpose;
                case Transpose::ConjTrans: return rocblas_operation_conjugate_transpose;
            }
        } else if constexpr (B == BackendLibrary::ROCSPARSE) {
            //return trans == Transpose::NoTrans ? rocsparse_operation_none : rocsparse_operation_transpose;
            switch (trans){
                case Transpose::NoTrans: return rocsparse_operation_none;
                case Transpose::Trans: return rocsparse_operation_transpose;
                case Transpose::ConjTrans: return rocsparse_operation_conjugate_transpose;
            }
        } else
#endif
#if BATCHLAS_HAS_CBLAS
        if constexpr (B == BackendLibrary::CBLAS) {
            return static_cast<CBLAS_TRANSPOSE>(
                trans == Transpose::NoTrans ? CblasNoTrans : trans == Transpose::ConjTrans ? CblasConjTrans : CblasTrans
            );
        } else
#endif
#if BATCHLAS_HAS_LAPACKE
        if constexpr (B == BackendLibrary::LAPACKE) {
            return static_cast<char>(
                trans == Transpose::NoTrans ? 'N' : trans == Transpose::ConjTrans ? 'C' : 'T'
            );
        } else
#endif
        {
            static_assert(always_false<B>::value, "Unsupported backend for Transpose conversion");
        }
    }

    template<BackendLibrary B, typename T>
    constexpr auto enum_convert(ComputePrecision precision) {
#if BATCHLAS_HAS_CUBLAS
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
                        throw batchlas::unsupported("Unsupported precision for single precision type");
                }
            } 
            else if constexpr (std::is_same_v<BaseType, double>) {
                if (precision == ComputePrecision::F64) {
                    return CUBLAS_COMPUTE_64F;
                }
                throw batchlas::unsupported("Only F64 precision supported for double precision type");
            }
        } else
#endif
    #if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCBLAS || B == BackendLibrary::ROCSOLVER || B == BackendLibrary::ROCSPARSE) {
            (void)precision;
            if constexpr (std::is_same_v<T, float>) {
                return rocblas_datatype_f32_r;
            } else if constexpr (std::is_same_v<T, double>) {
                return rocblas_datatype_f64_r;
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return rocblas_datatype_f32_c;
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return rocblas_datatype_f64_c;
            }
        } else
#endif
        {
            throw batchlas::unsupported("Unsupported backend or type combination");
        }
    }

    template<BackendLibrary B, typename T>
    constexpr auto ptr_convert(T** ptr) {
#if BATCHLAS_HAS_CUDA_BACKEND
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSPARSE || B == BackendLibrary::CUSOLVER) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return ptr; // No conversion needed
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<cuComplex**>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<cuDoubleComplex**>(ptr);
            }
        } else
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCBLAS || B == BackendLibrary::ROCSOLVER || B == BackendLibrary::ROCSPARSE) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return ptr;
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<rocblas_float_complex**>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<rocblas_double_complex**>(ptr);
            }
        } else
#endif
#if BATCHLAS_HAS_CBLAS
        if constexpr (B == BackendLibrary::CBLAS) {
            if constexpr (std::is_same_v<T, float>) {
                return reinterpret_cast<float**>(ptr);
            } else if constexpr (std::is_same_v<T, double>) {
                return reinterpret_cast<double**>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<void**>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<void**>(ptr);
            }
        } else
#endif
        {
            static_assert(always_false<B>::value, "Unsupported backend or type combination");
        }
    }

    


    template<BackendLibrary B, typename T>
    constexpr auto ptr_convert(T* ptr) {
        static_assert(std::is_floating_point<T>::value || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>, "Type must be floating point or complex");
#if BATCHLAS_HAS_CUDA_BACKEND
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSPARSE || B == BackendLibrary::CUSOLVER) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return ptr; // No conversion needed
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<cuComplex*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<cuDoubleComplex*>(ptr);
            }
        } else
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCBLAS || B == BackendLibrary::ROCSOLVER || B == BackendLibrary::ROCSPARSE) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return ptr;
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<rocblas_float_complex*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<rocblas_double_complex*>(ptr);
            }
        } else
#endif
#if BATCHLAS_HAS_HOST_BACKEND
        if constexpr (B == BackendLibrary::CBLAS) {
            if constexpr (std::is_same_v<T, float>) {
                return reinterpret_cast<float*>(ptr);
            } else if constexpr (std::is_same_v<T, double>) {
                return reinterpret_cast<double*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<void*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<void*>(ptr);
            }
        } else
#endif
#if BATCHLAS_HAS_LAPACKE
        if constexpr (B == BackendLibrary::LAPACKE) {
            if constexpr (std::is_same_v<T, float>) {
                return reinterpret_cast<float*>(ptr);
            } else if constexpr (std::is_same_v<T, double>) {
                return reinterpret_cast<double*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<lapack_complex_float*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<lapack_complex_double*>(ptr);
            }
        } else
#endif
        {
            static_assert(always_false<B>::value, "Unsupported backend or type combination");
        }
    }

    // Const pointer version
    template<BackendLibrary B, typename T>
    constexpr auto ptr_convert(const T* ptr) {
        static_assert(std::is_floating_point<T>::value || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>, "Type must be floating point or complex");
#if BATCHLAS_HAS_CUDA_BACKEND
        if constexpr (B == BackendLibrary::CUBLAS || B == BackendLibrary::CUSPARSE || B == BackendLibrary::CUSOLVER) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return ptr; // No conversion needed
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<const cuComplex*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<const cuDoubleComplex*>(ptr);
            }
        } else
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        if constexpr (B == BackendLibrary::ROCBLAS || B == BackendLibrary::ROCSOLVER || B == BackendLibrary::ROCSPARSE) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return ptr;
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                return reinterpret_cast<const rocblas_float_complex*>(ptr);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                return reinterpret_cast<const rocblas_double_complex*>(ptr);
            }
        } else
#endif
        {
            static_assert(always_false<B>::value, "Unsupported backend or type combination");
        }
    }

    template<BackendLibrary B, typename T>
    constexpr auto float_convert(const T& val) {
        using BareT = std::remove_const_t<T>;
        static_assert(is_complex_or_floating_point<BareT>::value, "Type must be floating point or complex");
        if constexpr (B == BackendLibrary::CBLAS) {
            if constexpr (std::is_same_v<BareT, float>) {
                return static_cast<float>(val);
            } else if constexpr (std::is_same_v<BareT, double>) {
                return static_cast<double>(val);
            } else if constexpr (std::is_same_v<BareT, std::complex<float>>) {
                return static_cast<const void*>(std::addressof(val));
            } else if constexpr (std::is_same_v<BareT, std::complex<double>>) {
                return static_cast<const void*>(std::addressof(val));
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
            using Arg = std::remove_cv_t<std::remove_reference_t<T>>;
            if constexpr (is_linalg_enum<Arg>::value) {
                return enum_convert<B>(std::forward<T>(arg));
            } else if constexpr (std::is_integral_v<Arg>) {
                return std::forward<T>(arg);
            } else if constexpr (std::is_integral_v<std::remove_pointer_t<Arg>>) {
                return std::forward<T>(arg);
            } else if constexpr (std::is_pointer_v<Arg> && is_complex_or_floating_point<std::remove_pointer_t<std::remove_pointer_t<Arg>>>::value) {
                return ptr_convert<B>(std::forward<T>(arg));
            } else if constexpr (is_complex_or_floating_point<Arg>::value) {
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

#if BATCHLAS_HAS_CUBLAS
    inline auto check_status(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw batchlas::device_error("CUBLAS error: " + std::to_string(status));
        }
        return status;
    }
#endif

#if BATCHLAS_HAS_CUSPARSE
    inline auto check_status(cusparseStatus_t status) {
        if (status != CUSPARSE_STATUS_SUCCESS) {
            throw batchlas::device_error("CUSPARSE error: " + std::to_string(status));
        }
        return status;
    }
#endif

#if BATCHLAS_HAS_CUSOLVER
    inline auto check_status(cusolverStatus_t status) {
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw batchlas::device_error("CUSOLVER error: " + std::to_string(status));
        }
        return status;
    }
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
    inline auto check_status(rocblas_status status) {
        if (status != rocblas_status_success) {
            throw batchlas::device_error("rocBLAS error: " + std::to_string(status));
        }
        return status;
    }

    inline auto check_status(rocsparse_status status) {
        if (status != rocsparse_status_success) {
            throw batchlas::device_error("rocSPARSE error: " + std::to_string(status));
        }
        return status;
    }
#endif
    
#if BATCHLAS_HAS_CUDA_BACKEND
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
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
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
#endif

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

    // Same dispatch, returning the routine's result (LAPACKE's info). A sibling, not a change to
    // call_backend_nh, whose void CBLAS users would break an `auto` return.
    // evidence: docs/design/runtime-internals.md#runtime-internals-the-per-item-info-span-contract
    template <typename T, BackendLibrary BL, typename Fun1, typename Fun2, typename Fun3, typename Fun4, typename... Args>
    auto call_backend_nh_r(const Fun1& fun1, const Fun2& fun2, const Fun3& fun3, const Fun4& fun4, Args&&... args) {
        if constexpr (std::is_same_v<T,float>) {
            return std::apply(fun1, backend_convert<BL>(std::forward<Args>(args)...));
        } else if constexpr (std::is_same_v<T,double>) {
            return std::apply(fun2, backend_convert<BL>(std::forward<Args>(args)...));
        } else if constexpr (std::is_same_v<T,std::complex<float>>) {
            return std::apply(fun3, backend_convert<BL>(std::forward<Args>(args)...));
        } else {
            return std::apply(fun4, backend_convert<BL>(std::forward<Args>(args)...));
        }
    }

    namespace detail {

    // Per-item status target: the caller's span, else pool scratch (short spans silently ignored).
    // A span only REMOVES a pool draw, so *_buffer_size stays correct in both modes.
    // evidence: docs/design/runtime-internals.md#runtime-internals-the-per-item-info-span-contract
    inline Span<int32_t> info_target(Queue& ctx, BumpAllocator& pool,
                                     Span<int32_t> info_out, size_t count) {
        if (info_out.size() >= count) return info_out;
        return pool.allocate<int32_t>(ctx, count);
    }

    }  // namespace detail


    // Variadic template for converting multiple enums
/*     template <Backend B, typename... Args>
    auto enum_convert(Args&&... args) {
        static_assert((is_linalg_enum<std::remove_reference_t<Args>>::value && ...), 
                     "All arguments must be linalg enum types");
        return std::make_tuple(enum_convert<B>(std::forward<Args>(args))...);
    }
 */


        


    #if BATCHLAS_HAS_CUDA_BACKEND
        template <>
        struct BackendScalar<float, BackendLibrary::CUBLAS> {
            static constexpr cudaDataType_t type = CUDA_R_32F;
        };

        template <>
        struct BackendScalar<float, BackendLibrary::CUSPARSE> {
            static constexpr cudaDataType_t type = CUDA_R_32F;
        };

        template <>
        struct BackendScalar<float, BackendLibrary::CUSOLVER> {
            static constexpr cudaDataType_t type = CUDA_R_32F;
        };

        
        template <>
        struct BackendScalar<double, BackendLibrary::CUBLAS> {
            static constexpr cudaDataType_t type = CUDA_R_64F;
        };
        
        template <>
        struct BackendScalar<double, BackendLibrary::CUSPARSE> {
            static constexpr cudaDataType_t type = CUDA_R_64F;
        };
        
        template <>
        struct BackendScalar<double, BackendLibrary::CUSOLVER> {
            static constexpr cudaDataType_t type = CUDA_R_64F;
        };

        template <>
        struct BackendScalar<std::complex<float>, BackendLibrary::CUBLAS> {
            static constexpr cudaDataType_t type = CUDA_C_32F;
        };

        template <>
        struct BackendScalar<std::complex<float>, BackendLibrary::CUSPARSE> {
            static constexpr cudaDataType_t type = CUDA_C_32F;
        };

        template <>
        struct BackendScalar<std::complex<float>, BackendLibrary::CUSOLVER> {
            static constexpr cudaDataType_t type = CUDA_C_32F;
        };

        template <>
        struct BackendScalar<std::complex<double>, BackendLibrary::CUBLAS> {
            static constexpr cudaDataType_t type = CUDA_C_64F;
        };

        template <>
        struct BackendScalar<std::complex<double>, BackendLibrary::CUSPARSE> {
            static constexpr cudaDataType_t type = CUDA_C_64F;
        };

        template <>
        struct BackendScalar<std::complex<double>, BackendLibrary::CUSOLVER> {
            static constexpr cudaDataType_t type = CUDA_C_64F;
        };
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
        template <>
        struct BackendScalar<float, BackendLibrary::ROCBLAS> {
            static constexpr rocblas_datatype type = rocblas_datatype_f32_r;
        };

        template <>
        struct BackendScalar<float, BackendLibrary::ROCSPARSE> {
            static constexpr rocsparse_datatype type = rocsparse_datatype_f32_r;
        };

        template <>
        struct BackendScalar<float, BackendLibrary::ROCSOLVER> {
            static constexpr rocblas_datatype type = rocblas_datatype_f32_r;
        };

        template <>
        struct BackendScalar<double, BackendLibrary::ROCBLAS> {
            static constexpr rocblas_datatype type = rocblas_datatype_f64_r;
        };

        template <>
        struct BackendScalar<double, BackendLibrary::ROCSPARSE> {
            static constexpr rocsparse_datatype type = rocsparse_datatype_f64_r;
        };

        template <>
        struct BackendScalar<double, BackendLibrary::ROCSOLVER> {
            static constexpr rocblas_datatype type = rocblas_datatype_f64_r;
        };

        template <>
        struct BackendScalar<std::complex<float>, BackendLibrary::ROCBLAS> {
            static constexpr rocblas_datatype type = rocblas_datatype_f32_c;
        };

        template <>
        struct BackendScalar<std::complex<float>, BackendLibrary::ROCSPARSE> {
            static constexpr rocsparse_datatype type = rocsparse_datatype_f32_c;
        };

        template <>
        struct BackendScalar<std::complex<float>, BackendLibrary::ROCSOLVER> {
            static constexpr rocblas_datatype type = rocblas_datatype_f32_c;
        };

        template <>
        struct BackendScalar<std::complex<double>, BackendLibrary::ROCBLAS> {
            static constexpr rocblas_datatype type = rocblas_datatype_f64_c;
        };

        template <>
        struct BackendScalar<std::complex<double>, BackendLibrary::ROCSPARSE> {
            static constexpr rocsparse_datatype type = rocsparse_datatype_f64_c;
        };

        template <>
        struct BackendScalar<std::complex<double>, BackendLibrary::ROCSOLVER> {
            static constexpr rocblas_datatype type = rocblas_datatype_f64_c;
        };
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
        template <typename T>
        struct BlasComputeType<T, ComputePrecision::Default, Backend::ROCM> {
            static constexpr rocblas_datatype type = (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) ? rocblas_datatype_f32_r : rocblas_datatype_f64_r;
        };
#endif

// Keyed on the LIBRARIES: these types do not exist under BATCHLAS_HAS_CUDA_BACKEND alone.
#if BATCHLAS_HAS_CUBLAS

        template <typename T>
        struct BlasComputeType<T, ComputePrecision::Default, Backend::CUDA> {
            static constexpr cublasComputeType_t type = (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_64F;
        };
#endif

#if BATCHLAS_HAS_CUBLAS && BATCHLAS_HAS_CUSPARSE && BATCHLAS_HAS_CUSOLVER

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
                    throw batchlas::device_error("CUBLAS initialization failed");
                }

                auto sparse_status = cusparseCreate(&sparse_handle_);
                if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
                    std::cerr << "CUSPARSE initialization failed with status: " << sparse_status << std::endl;
                    throw batchlas::device_error("CUSPARSE initialization failed");
                }

                auto solver_status = cusolverDnCreate(&solver_handle_);
                if (solver_status != CUSOLVER_STATUS_SUCCESS) {
                    std::cerr << "CUSOLVER initialization failed with status: " << solver_status << std::endl;
                    throw batchlas::device_error("CUSOLVER initialization failed");
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

            // Enqueuing calls set the stream inside impl::run_native; this overload is for queries.
            void setStream(const Queue& ctx) {
                setStream(static_cast<cudaStream_t>(impl::query_stream<impl::Native::Cuda>(*ctx)));
            }

            void setStream(cudaStream_t stream) {
                cublasSetStream(blas_handle_, stream);
                cusparseSetStream(sparse_handle_, stream);
                cusolverDnSetStream(solver_handle_, stream);
            }
        };
#elif BATCHLAS_HAS_CUDA_BACKEND
        // CUDA without all math libraries: no handles, but the type must be COMPLETE (native TUs such
        // as ortho.cc declare one). evidence: docs/design/vendor-independence.md#vendor-independence-headers-keyed-on-the-library-axis
        template <>
        struct LinalgHandle<Backend::CUDA> {
            void setStream(const Queue&) {}
            void setStream(cudaStream_t) {}
        };
#endif
    #if BATCHLAS_HAS_ROCM_BACKEND
        template <>
        struct LinalgHandle<Backend::ROCM> {
            rocblas_handle blas_handle_{};
            rocsparse_handle sparse_handle_{};

            LinalgHandle() {
                hipDeviceSynchronize();
                rocblas_create_handle(&blas_handle_);
                rocsparse_create_handle(&sparse_handle_);
                hipDeviceSynchronize();
            }

            LinalgHandle(const LinalgHandle&) = delete;
            LinalgHandle& operator=(const LinalgHandle&) = delete;
            LinalgHandle(LinalgHandle&&) = delete;
            LinalgHandle& operator=(LinalgHandle&&) = delete;

            ~LinalgHandle() {
                hipDeviceSynchronize();
                rocblas_destroy_handle(blas_handle_);
                rocsparse_destroy_handle(sparse_handle_);
                hipDeviceSynchronize();
            }

            constexpr inline operator rocblas_handle() const { return blas_handle_; }
            constexpr inline operator rocsparse_handle() const { return sparse_handle_; }

            void setStream(const Queue& ctx) {
                setStream(static_cast<hipStream_t>(impl::query_stream<impl::Native::Hip>(*ctx)));
            }

            void setStream(hipStream_t stream) {
                rocblas_set_stream(blas_handle_, stream);
                rocsparse_set_stream(sparse_handle_, stream);
            }
        };

#endif
#if BATCHLAS_HAS_HOST_BACKEND
    template <>
    struct LinalgHandle<Backend::NETLIB>{
        void setStream(const Queue& ctx) {}
    };
#endif


    template <typename KernelName>
    size_t get_kernel_max_wg_size(const Queue& ctx){
        try {
            return impl::kernel_max_wg_size_all_devices<KernelName>(ctx -> get_context(), ctx -> get_device());
        } catch (...) {
            return (ctx -> get_device()).get_info<sycl::info::device::max_work_group_size>();
        }
    }
}