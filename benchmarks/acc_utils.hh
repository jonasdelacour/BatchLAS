#pragma once

#include <batchlas/backend_config.h>
#include <blas/enums.hh>
#include <util/miniacc.hh>

#include <complex>

#if BATCHLAS_HAS_CUDA_BACKEND
#define BATCHLAS_ACC_CUDA(name, sizes) \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::CUDA>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::CUDA>), sizes);

#define BATCHLAS_ACC_CUDA_ALL_TYPES(name, sizes) \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::CUDA>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::CUDA>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::CUDA>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::CUDA>), sizes);
#else
#define BATCHLAS_ACC_CUDA(name, sizes)
#define BATCHLAS_ACC_CUDA_ALL_TYPES(name, sizes)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
#define BATCHLAS_ACC_ROCM(name, sizes) \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::ROCM>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::ROCM>), sizes);

#define BATCHLAS_ACC_ROCM_ALL_TYPES(name, sizes) \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::ROCM>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::ROCM>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::ROCM>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::ROCM>), sizes);
#else
#define BATCHLAS_ACC_ROCM(name, sizes)
#define BATCHLAS_ACC_ROCM_ALL_TYPES(name, sizes)
#endif

#if BATCHLAS_HAS_MKL_BACKEND
#define BATCHLAS_ACC_MKL(name, sizes) \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::MKL>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::MKL>), sizes);

#define BATCHLAS_ACC_MKL_ALL_TYPES(name, sizes) \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::MKL>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::MKL>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::MKL>), sizes); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::MKL>), sizes);
#else
#define BATCHLAS_ACC_MKL(name, sizes)
#define BATCHLAS_ACC_MKL_ALL_TYPES(name, sizes)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
#define BATCHLAS_ACC_NETLIB(name, sizes) \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::NETLIB>), sizes##Netlib);

#define BATCHLAS_ACC_NETLIB_ALL_TYPES(name, sizes) \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_ACC_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::NETLIB>), sizes##Netlib);
#else
#define BATCHLAS_ACC_NETLIB(name, sizes)
#define BATCHLAS_ACC_NETLIB_ALL_TYPES(name, sizes)
#endif

#define BATCHLAS_REGISTER_ACCURACY(name, sizes) \
    BATCHLAS_ACC_CUDA(name, sizes)              \
    BATCHLAS_ACC_ROCM(name, sizes)              \
    BATCHLAS_ACC_NETLIB(name, sizes)

#define BATCHLAS_REGISTER_ACCURACY_ALL_TYPES(name, sizes) \
    BATCHLAS_ACC_CUDA_ALL_TYPES(name, sizes)              \
    BATCHLAS_ACC_ROCM_ALL_TYPES(name, sizes)              \
    BATCHLAS_ACC_NETLIB_ALL_TYPES(name, sizes)
