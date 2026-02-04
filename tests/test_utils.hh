#pragma once
#include <batchlas/backend_config.h>
#include <tuple>
#include <gtest/gtest.h>
#include <blas/enums.hh>
#include <complex>
#include <type_traits>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <iostream>

namespace test_utils {

// Convert std::tuple<Ts...> to GoogleTest type list
template <class Tuple> struct tuple_to_types;
template <class... Ts>
struct tuple_to_types<std::tuple<Ts...>> { using type = ::testing::Types<Ts...>; };

// Backend name lookup for error messages
inline const char* backend_to_string(batchlas::Backend backend) {
    switch (backend) {
        case batchlas::Backend::CUDA: return "CUDA";
        case batchlas::Backend::ROCM: return "ROCM";
        case batchlas::Backend::MKL: return "MKL";
        case batchlas::Backend::NETLIB: return "NETLIB";
        default: return "UNKNOWN";
    }
}

// Runtime filtering support with error handling
inline bool should_run_backend(batchlas::Backend backend) {
    const char* env = std::getenv("BATCHLAS_TEST_BACKEND");
    if (!env) return true;  // No filter, run all
    
    std::string backend_filter(env);
    std::transform(backend_filter.begin(), backend_filter.end(), backend_filter.begin(), ::toupper);
    
    // Check if filter matches a known backend
    bool recognized = (backend_filter == "CUDA" || backend_filter == "ROCM" || 
                       backend_filter == "MKL" || backend_filter == "NETLIB");
    
    if (!recognized) {
        std::cerr << "Warning: BATCHLAS_TEST_BACKEND=" << env 
                  << " is not recognized. Valid values are: CUDA, ROCM, MKL, NETLIB. "
                  << "Skipping all tests." << std::endl;
        return false;
    }
    
    // Match backend to filter
    if (backend == batchlas::Backend::CUDA && backend_filter == "CUDA") return true;
    if (backend == batchlas::Backend::ROCM && backend_filter == "ROCM") return true;
    if (backend == batchlas::Backend::MKL && backend_filter == "MKL") return true;
    if (backend == batchlas::Backend::NETLIB && backend_filter == "NETLIB") return true;
    
    return false;  // Filter doesn't match, skip
}

// Portable type name checking using template specialization instead of typeid().name()
template <typename T> struct is_float_type : std::false_type {};
template <> struct is_float_type<float> : std::true_type {};
template <> struct is_float_type<std::complex<float>> : std::true_type {};

template <typename T> struct is_double_type : std::false_type {};
template <> struct is_double_type<double> : std::true_type {};
template <> struct is_double_type<std::complex<double>> : std::true_type {};

template <typename T> struct is_complex_type : std::false_type {};
template <> struct is_complex_type<std::complex<float>> : std::true_type {};
template <> struct is_complex_type<std::complex<double>> : std::true_type {};

// Runtime filtering for float types - portable version using template specialization
template <typename T>
inline bool should_run_float_type() {
    const char* env = std::getenv("BATCHLAS_TEST_FLOAT_TYPE");
    if (!env) return true;  // No filter, run all
    
    std::string type_filter(env);
    std::transform(type_filter.begin(), type_filter.end(), type_filter.begin(), ::tolower);
    
    if (type_filter == "float") {
        return is_float_type<T>::value;
    } else if (type_filter == "double") {
        return is_double_type<T>::value;
    } else if (type_filter == "complex") {
        return is_complex_type<T>::value;
    }
    
    return false;  // Filter doesn't match, skip
}

// Deprecated: non-template version kept for backward compatibility
inline bool should_run_float_type(const std::string& type_name) {
    const char* env = std::getenv("BATCHLAS_TEST_FLOAT_TYPE");
    if (!env) return true;  // No filter, run all
    
    std::string type_filter(env);
    std::transform(type_filter.begin(), type_filter.end(), type_filter.begin(), ::tolower);
    
    // typeid().name() returns mangled names (compiler-specific):
    // GCC/Clang: f, d, St7complexIfE, St7complexIdE
    // MSVC: float, double, complex<float>, complex<double> (different format)
    // This fallback supports GCC/Clang but may not work on all compilers
    
    if (type_filter == "float") {
        return type_name == "f" || type_name == "St7complexIfE" ||
               type_name.find("complex<float>") != std::string::npos;
    } else if (type_filter == "double") {
        return type_name == "d" || type_name == "St7complexIdE" ||
               type_name.find("complex<double>") != std::string::npos;
    } else if (type_filter == "complex") {
        return type_name == "St7complexIfE" || type_name == "St7complexIdE" ||
               type_name.find("complex") != std::string::npos;
    }
    
    return false;  // Filter doesn't match, skip
}

// Helper to gather types for all enabled backends (runtime filtering via environment variables)
template <template <typename, batchlas::Backend> class Config>
struct backend_types {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
    std::tuple<Config<float, batchlas::Backend::NETLIB>,
           Config<double, batchlas::Backend::NETLIB>,
           Config<std::complex<float>, batchlas::Backend::NETLIB>,
           Config<std::complex<double>, batchlas::Backend::NETLIB>>{},
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<float, batchlas::Backend::CUDA>,
                   Config<double, batchlas::Backend::CUDA>,
                   Config<std::complex<float>, batchlas::Backend::CUDA>,
                   Config<std::complex<double>, batchlas::Backend::CUDA>>{},
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        std::tuple<Config<float, batchlas::Backend::ROCM>,
                   Config<double, batchlas::Backend::ROCM>,
                   Config<std::complex<float>, batchlas::Backend::ROCM>,
                   Config<std::complex<double>, batchlas::Backend::ROCM>>{},
#endif
#if BATCHLAS_HAS_MKL_BACKEND
        std::tuple<Config<float, batchlas::Backend::MKL>,
                   Config<double, batchlas::Backend::MKL>,
                   Config<std::complex<float>, batchlas::Backend::MKL>,
                   Config<std::complex<double>, batchlas::Backend::MKL>>{},
#endif
        std::tuple<>{}));

    using type = typename tuple_to_types<tuple_type>::type;
};

// Helper to optionally exclude complex types per test suite
template <template <typename, batchlas::Backend> class Config, bool IncludeComplex>
struct backend_types_filtered {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
    std::tuple<Config<float, batchlas::Backend::NETLIB>,
           Config<double, batchlas::Backend::NETLIB>>{},
    std::conditional_t<IncludeComplex,
           std::tuple<Config<std::complex<float>, batchlas::Backend::NETLIB>,
                      Config<std::complex<double>, batchlas::Backend::NETLIB>>,
           std::tuple<>>{},
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<float, batchlas::Backend::CUDA>,
                   Config<double, batchlas::Backend::CUDA>>{},
        std::conditional_t<IncludeComplex,
                   std::tuple<Config<std::complex<float>, batchlas::Backend::CUDA>,
                              Config<std::complex<double>, batchlas::Backend::CUDA>>,
                   std::tuple<>>{},
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        std::tuple<Config<float, batchlas::Backend::ROCM>,
                   Config<double, batchlas::Backend::ROCM>>{},
        std::conditional_t<IncludeComplex,
                   std::tuple<Config<std::complex<float>, batchlas::Backend::ROCM>,
                              Config<std::complex<double>, batchlas::Backend::ROCM>>,
                   std::tuple<>>{},
#endif
#if BATCHLAS_HAS_MKL_BACKEND
        std::tuple<Config<float, batchlas::Backend::MKL>,
                   Config<double, batchlas::Backend::MKL>>{},
        std::conditional_t<IncludeComplex,
                   std::tuple<Config<std::complex<float>, batchlas::Backend::MKL>,
                              Config<std::complex<double>, batchlas::Backend::MKL>>,
                   std::tuple<>>{},
#endif
        std::tuple<>{}));

    using type = typename tuple_to_types<tuple_type>::type;
};

// Choose one available GPU backend at compile time
#if BATCHLAS_HAS_CUDA_BACKEND
constexpr batchlas::Backend gpu_backend = batchlas::Backend::CUDA;
#elif BATCHLAS_HAS_ROCM_BACKEND
constexpr batchlas::Backend gpu_backend = batchlas::Backend::ROCM;
#elif BATCHLAS_HAS_MKL_BACKEND
constexpr batchlas::Backend gpu_backend = batchlas::Backend::MKL;
#endif

// Utility traits and helper functions
template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
constexpr typename batchlas::base_type<T>::type tolerance() {
    using real_t = typename batchlas::base_type<T>::type;
    if constexpr (is_complex<T>::value) {
        if constexpr (std::is_same_v<real_t, float>) return real_t(2e-5f);
        else return real_t(2e-10);
    } else {
        if constexpr (std::is_same_v<real_t, float>) return real_t(1e-5f);
        else return real_t(1e-10);
    }
}

template <typename T>
inline void expect_near(const T& a, const T& b, typename batchlas::base_type<T>::type tol = tolerance<T>()) {
    using real_t = typename batchlas::base_type<T>::type;
    if constexpr (is_complex<T>::value) {
        EXPECT_NEAR(a.real(), b.real(), tol);
        EXPECT_NEAR(a.imag(), b.imag(), tol);
    } else {
        EXPECT_NEAR(a, b, tol);
    }
}

template <typename T>
inline void assert_near(const T& a, const T& b, typename batchlas::base_type<T>::type tol = tolerance<T>()) {
    using real_t = typename batchlas::base_type<T>::type;
    if constexpr (is_complex<T>::value) {
        ASSERT_NEAR(a.real(), b.real(), tol);
        ASSERT_NEAR(a.imag(), b.imag(), tol);
    } else {
        ASSERT_NEAR(a, b, tol);
    }
}

// Unified base test fixture for all BatchLAS tests
template <typename Config>
class BatchLASTest : public ::testing::Test {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr batchlas::Backend BackendType = Config::BackendVal;
    std::shared_ptr<Queue> ctx;
    
    void SetUp() override {
        // Runtime filtering
        if (!should_run_backend(BackendType)) {
            GTEST_SKIP() << "Backend filtered by BATCHLAS_TEST_BACKEND environment variable";
        }
        
        // Use portable template-based type filtering instead of typeid().name()
        if (!should_run_float_type<ScalarType>()) {
            GTEST_SKIP() << "Float type filtered by BATCHLAS_TEST_FLOAT_TYPE environment variable";
        }
        
        // GPU backends require GPU device
        if constexpr (BackendType != batchlas::Backend::NETLIB) {
            try {
                ctx = std::make_shared<Queue>("gpu", true);
                if (ctx->device().type != DeviceType::GPU) {
                    GTEST_SKIP() << "GPU backend requires GPU device, but none was selected";
                }
            } catch (const sycl::exception& e) {
                if (e.code() == sycl::errc::runtime || 
                    e.code() == sycl::errc::feature_not_supported) {
                    GTEST_SKIP() << "GPU queue creation failed: " << e.what();
                }
                throw;
            } catch (const std::exception& e) {
                GTEST_SKIP() << "Queue construction failed: " << e.what();
            }
        } else {
#if !BATCHLAS_HAS_CPU_TARGET
            GTEST_SKIP() << "NETLIB backend requires CPU SYCL target support (no CPU target in fsycl-targets)";
#endif
            ctx = std::make_shared<Queue>("cpu");
        }
    }
    
    void TearDown() override {
        if (ctx) {
            ctx->wait();
        }
    }
};

} // namespace test_utils
