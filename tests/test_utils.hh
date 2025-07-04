#pragma once
#include <batchlas/backend_config.h>
#include <tuple>
#include <gtest/gtest.h>
#include <blas/enums.hh>
#include <complex>
#include <type_traits>

namespace test_utils {

// Convert std::tuple<Ts...> to GoogleTest type list
template <class Tuple> struct tuple_to_types;
template <class... Ts>
struct tuple_to_types<std::tuple<Ts...>> { using type = ::testing::Types<Ts...>; };

// Helper to gather types for all enabled backends
template <template <typename, batchlas::Backend> class Config>
struct backend_types {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_HOST_BACKEND
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

} // namespace test_utils
