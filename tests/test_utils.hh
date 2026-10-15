#pragma once
#include <batchlas/backend_config.h>
#include <cstdint>
#include <tuple>
#include <gtest/gtest.h>
#include <batchlas/blas/enums.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/verify/tolerance.hh>
#include <complex>
#include <type_traits>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <algorithm>
#include <iostream>

namespace test_utils {

// acpp without BATCHLAS_ACPP_SLM_OPTIN launches at most 48 KiB of local memory per work-group
// (no CUDA opt-in), so every capacity read from the device budget is smaller than the 99 KiB
// that tests, hole rows and shipped tables were written against. Tests whose premise needs the
// opt-in skip on it, naming this; DPC++ and the opt-in variant keep every assertion.
// evidence: docs/design/sycl-implementations.md#sycl-impl-slm-budget
inline constexpr bool kSlmCappedAt48KiB = BATCHLAS_SYCL_IMPL_ACPP && !BATCHLAS_ACPP_SLM_OPTIN;
inline constexpr const char* kSlmCappedReason =
#if BATCHLAS_HAS_SYCL_BACKEND
    "acpp on an Apple GPU: 32 KiB threadgroup memory per work-group, less the runtime's scratch";
#else
    "acpp without BATCHLAS_ACPP_SLM_OPTIN: 48 KiB local memory per work-group (no CUDA opt-in)";
#endif

// True when @p msg reports a kernel with no device image (runtimes word it differently) AND
// BATCHLAS_TEST_ALLOW_MISSING_KERNEL=1: only then may the caller skip. Otherwise it rethrows and
// the test fails, because a silent skip is how a dropped instantiation stays green.
inline bool may_skip_missing_kernel(std::string_view msg) {
    const bool missing = msg.find("No kernel named") != std::string_view::npos ||
                         msg.find("kernel bundle") != std::string_view::npos ||
                         msg.find("PI_ERROR_INVALID_KERNEL_NAME") != std::string_view::npos;
    const char* allow = std::getenv("BATCHLAS_TEST_ALLOW_MISSING_KERNEL");
    return missing && allow && std::string_view(allow) == "1";
}

// A *TranscribedTable test holds untimed sm_89 rows to the transcribed order and skips timed ones. Once
// a retune times every consulted row it must skip loudly rather than pass on zero rows.
class TranscribedRowsHeld {
public:
    bool skip(bool timed) { timed ? ++timed_ : ++checked_; return timed; }
    int checked() const { return checked_; }
    int timed() const { return timed_; }
private:
    int checked_ = 0, timed_ = 0;
};
#define SKIP_IF_NO_TRANSCRIBED_ROWS(held)                                                      \
    if ((held).checked() == 0)                                                                 \
    GTEST_SKIP() << "all " << (held).timed() << " consulted sm_89 rows are timed: the old "   \
                 << "router's preference no longer applies; delete this test with the "        \
                 << "transcribed tables"

// Convert std::tuple<Ts...> to GoogleTest type list
template <class Tuple> struct tuple_to_types;
template <class... Ts>
struct tuple_to_types<std::tuple<Ts...>> { using type = ::testing::Types<Ts...>; };

// Backend name lookup for error messages. Forwards to the public printer in
// <batchlas/blas/enums.hh>; the hand-rolled switch this replaces rendered AUTO, MAGMA
// and SYCL as "UNKNOWN".
inline std::string_view backend_to_string(batchlas::Backend backend) {
    return batchlas::to_string(backend);
}

// Runtime filtering support with error handling
inline bool should_run_backend(batchlas::Backend backend) {
    const char* env = std::getenv("BATCHLAS_TEST_BACKEND");
    if (!env) return true;  // No filter, run all
    
    std::string backend_filter(env);
    std::transform(backend_filter.begin(), backend_filter.end(), backend_filter.begin(), ::toupper);
    
    // Check if filter matches a known backend
    bool recognized = (backend_filter == "CUDA" || backend_filter == "ROCM" ||
                       backend_filter == "SYCL" || backend_filter == "NETLIB");
    
    if (!recognized) {
        std::cerr << "Warning: BATCHLAS_TEST_BACKEND=" << env 
                  << " is not recognized. Valid values are: CUDA, ROCM, SYCL, NETLIB. "
                  << "Skipping all tests." << std::endl;
        return false;
    }
    
    // Match backend to filter
    if (backend == batchlas::Backend::CUDA && backend_filter == "CUDA") return true;
    if (backend == batchlas::Backend::ROCM && backend_filter == "ROCM") return true;
    if (backend == batchlas::Backend::SYCL && backend_filter == "SYCL") return true;
    if (backend == batchlas::Backend::NETLIB && backend_filter == "NETLIB") return true;
    
    return false;  // Filter doesn't match, skip
}

template <typename T>
inline constexpr bool is_float_type_v = std::is_same_v<batchlas::float_t<T>, float>;

template <typename T>
inline constexpr bool is_double_type_v = std::is_same_v<batchlas::float_t<T>, double>;

template <typename T>
inline constexpr bool is_complex_type_v = batchlas::is_std_complex_v<T>;

// Runtime filtering for float types - portable version using template specialization
template <typename T>
inline bool should_run_float_type() {
    const char* env = std::getenv("BATCHLAS_TEST_FLOAT_TYPE");
    if (!env) return true;  // No filter, run all
    
    std::string type_filter(env);
    std::transform(type_filter.begin(), type_filter.end(), type_filter.begin(), ::tolower);
    
    if (type_filter == "float") {
        return is_float_type_v<T>;
    } else if (type_filter == "double") {
        return is_double_type_v<T>;
    } else if (type_filter == "complex") {
        return is_complex_type_v<T>;
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
#if BATCHLAS_HAS_SYCL_BACKEND
        std::tuple<Config<float, batchlas::Backend::SYCL>,
                   Config<std::complex<float>, batchlas::Backend::SYCL>>{},
#if BATCHLAS_HAS_DEVICE_FP64
        std::tuple<Config<double, batchlas::Backend::SYCL>,
                   Config<std::complex<double>, batchlas::Backend::SYCL>>{},
#endif
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
#if BATCHLAS_HAS_SYCL_BACKEND
        std::tuple<Config<float, batchlas::Backend::SYCL>>{},
        std::conditional_t<IncludeComplex, std::tuple<Config<std::complex<float>, batchlas::Backend::SYCL>>,
                           std::tuple<>>{},
#if BATCHLAS_HAS_DEVICE_FP64
        std::tuple<Config<double, batchlas::Backend::SYCL>>{},
        std::conditional_t<IncludeComplex, std::tuple<Config<std::complex<double>, batchlas::Backend::SYCL>>,
                           std::tuple<>>{},
#endif
#endif
        std::tuple<>{}));

    using type = typename tuple_to_types<tuple_type>::type;
};

// Helper for entry points that exist only for complex scalars (hemm, and the
// ?he* family generally -- BLAS has no real spelling of them).
template <template <typename, batchlas::Backend> class Config>
struct backend_types_complex {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
    std::tuple<Config<std::complex<float>, batchlas::Backend::NETLIB>,
           Config<std::complex<double>, batchlas::Backend::NETLIB>>{},
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<std::complex<float>, batchlas::Backend::CUDA>,
                   Config<std::complex<double>, batchlas::Backend::CUDA>>{},
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        std::tuple<Config<std::complex<float>, batchlas::Backend::ROCM>,
                   Config<std::complex<double>, batchlas::Backend::ROCM>>{},
#endif
#if BATCHLAS_HAS_SYCL_BACKEND
        std::tuple<Config<std::complex<float>, batchlas::Backend::SYCL>>{},
#if BATCHLAS_HAS_DEVICE_FP64
        std::tuple<Config<std::complex<double>, batchlas::Backend::SYCL>>{},
#endif
#endif
        std::tuple<>{}));

    using type = typename tuple_to_types<tuple_type>::type;
};

// Choose one available GPU backend at compile time
#if BATCHLAS_HAS_CUDA_BACKEND
constexpr batchlas::Backend gpu_backend = batchlas::Backend::CUDA;
#elif BATCHLAS_HAS_ROCM_BACKEND
constexpr batchlas::Backend gpu_backend = batchlas::Backend::ROCM;
#elif BATCHLAS_HAS_SYCL_BACKEND
constexpr batchlas::Backend gpu_backend = batchlas::Backend::SYCL;
#endif

// The backend a native-kernel test pins (tests that call the CUDA-or-SYCL native kernels
// directly): CUDA, else SYCL (AdaptiveCpp Metal). A build with neither keeps CUDA, whose
// Queue constructor throws, which is how these tests behaved before SYCL existed.
#if BATCHLAS_HAS_CUDA_BACKEND
constexpr batchlas::Backend native_gpu_backend = batchlas::Backend::CUDA;
#elif BATCHLAS_HAS_SYCL_BACKEND
constexpr batchlas::Backend native_gpu_backend = batchlas::Backend::SYCL;
#else
constexpr batchlas::Backend native_gpu_backend = batchlas::Backend::CUDA;
#endif

// True when the native GPU backend has no double kernels (Metal: no device fp64).
inline constexpr bool native_gpu_lacks_fp64 =
    native_gpu_backend == batchlas::Backend::SYCL && !BATCHLAS_HAS_DEVICE_FP64;

// T for a native-kernel case written in double: double (or complex<double>) where the native
// GPU backend has it, else the float type of the same kind, so the kernel still launches.
template <typename T>
using native_gpu_scalar_t = std::conditional_t<
    native_gpu_lacks_fp64,
    std::conditional_t<std::is_same_v<T, double>, float,
                       std::conditional_t<std::is_same_v<T, std::complex<double>>, std::complex<float>, T>>,
    T>;

// The work-group ceiling of SYCL dims 0/1 of a 2-D/3-D launch (CUDA grid z/y). Grid-ceiling
// tests launch at kCudaGridYZ and one past it; on an Apple GPU (acpp Metal) the cap is the
// 32-bit work-item global range instead, so one past CUDA's ceiling must be served. The test's
// own fact, checked against select::Device::max_groups_yz; pass select::Device::family.
inline constexpr std::int64_t kCudaGridYZ = 65535;
inline std::int64_t expected_grid_yz(std::string_view device_family) {
    return device_family == "apple" ? std::int64_t{0xffffffff} : kCudaGridYZ;
}

// Utility traits and helper functions
template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline void expect_near(const T& a, const T& b, typename batchlas::base_type<T>::type tol) {
    if constexpr (is_complex<T>::value) {
        EXPECT_NEAR(a.real(), b.real(), tol);
        EXPECT_NEAR(a.imag(), b.imag(), tol);
    } else {
        EXPECT_NEAR(a, b, tol);
    }
}

// batchlas::verify::pass (recorded) with value, bound, kind and n in the failure message.
template <typename T>
::testing::AssertionResult verify_pass(batchlas::verify::Check kind, int n, double value) {
    if (batchlas::verify::pass<T>(kind, n, value)) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << batchlas::verify::detail::kind_name(kind) << ": value " << value << " exceeds bound "
                                         << batchlas::verify::bound<T>(kind, n) << " (n=" << n << ")";
}

template <typename T>
::testing::AssertionResult verify_pass(batchlas::verify::Check kind, int n, double value, batchlas::verify::Slack slack) {
    if (batchlas::verify::pass<T>(kind, n, value, slack)) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << batchlas::verify::detail::kind_name(kind) << ": value " << value << " exceeds bound "
                                         << batchlas::verify::bound<T>(kind, n, slack) << " (n=" << n << ", slack " << slack.factor
                                         << " (" << slack.reason << "))";
}

// A vendor build's "no route": select::pick's std::runtime_error("<op>: no runnable kernel on ...").
// Any other exception fails, so a vendor that ran and then threw cannot pass for a refusal.
template <typename F>
::testing::AssertionResult throws_no_runnable_kernel(F&& f) {
    try {
        std::forward<F>(f)();
    } catch (const std::exception& e) {
        if (std::string_view(e.what()).find("no runnable kernel") != std::string_view::npos)
            return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure() << "threw, but not 'no runnable kernel': " << e.what();
    }
    return ::testing::AssertionFailure() << "did not throw";
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
        
        // The queue carries the backend. Tests are typed on a Backend, so bind
        // it here rather than passing it at every call: that is what lets a test
        // body say `syev(*ctx, ...)` and still be the BackendType test, and it
        // means the suite exercises the same queue-dispatch path callers use.
        // Pinning also makes a missing backend fail loudly here instead of
        // silently resolving to whatever the device vendor suggests.
        if constexpr (BackendType != batchlas::Backend::NETLIB) {
            try {
                ctx = std::make_shared<Queue>(Device("gpu"), BackendType, true);
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
            ctx = std::make_shared<Queue>(Device("cpu"), BackendType, true);
        }
    }
    
    void TearDown() override {
        if (ctx) {
            ctx->wait();
        }
    }
};

} // namespace test_utils

// EXPECT_VERIFY(T, Check::solve, n, value); EXPECT_VERIFY_SLACK(T, kind, n, value, Slack{factor, "reason"}).
#define EXPECT_VERIFY(T, kind, n, value) EXPECT_TRUE(::test_utils::verify_pass<T>((kind), (n), (value)))
#define EXPECT_VERIFY_SLACK(T, kind, n, value, ...) EXPECT_TRUE(::test_utils::verify_pass<T>((kind), (n), (value), __VA_ARGS__))
#define EXPECT_NO_RUNNABLE_KERNEL(...) EXPECT_TRUE(::test_utils::throws_no_runnable_kernel([&] { __VA_ARGS__; }))
