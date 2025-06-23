#pragma once
#include <batchlas/backend_config.h>
#include <tuple>
#include <gtest/gtest.h>

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
                   Config<double, batchlas::Backend::NETLIB>>{},
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<float, batchlas::Backend::CUDA>,
                   Config<double, batchlas::Backend::CUDA>>{},
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        std::tuple<Config<float, batchlas::Backend::ROCM>,
                   Config<double, batchlas::Backend::ROCM>>{},
#endif
#if BATCHLAS_HAS_MKL_BACKEND
        std::tuple<Config<float, batchlas::Backend::MKL>,
                   Config<double, batchlas::Backend::MKL>>{},
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

} // namespace test_utils
