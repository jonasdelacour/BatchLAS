#pragma once
#include <util/minibench.hh>
#include <batchlas/backend_config.h>

#if BATCHLAS_HAS_CUDA_BACKEND
#define BATCHLAS_BENCH_CUDA(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::CUDA>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::CUDA>), sizes);
#else
#define BATCHLAS_BENCH_CUDA(name, sizes)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
#define BATCHLAS_BENCH_ROCM(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::ROCM>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::ROCM>), sizes);
#else
#define BATCHLAS_BENCH_ROCM(name, sizes)
#endif

#if BATCHLAS_HAS_MKL_BACKEND
#define BATCHLAS_BENCH_MKL(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::MKL>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::MKL>), sizes);
#else
#define BATCHLAS_BENCH_MKL(name, sizes)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
#define BATCHLAS_BENCH_NETLIB(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::NETLIB>), sizes##Netlib);
#else
#define BATCHLAS_BENCH_NETLIB(name, sizes)
#endif

#define BATCHLAS_REGISTER_BENCHMARK(name, sizes) \
    BATCHLAS_BENCH_CUDA(name, sizes) \
    BATCHLAS_BENCH_ROCM(name, sizes) \
    BATCHLAS_BENCH_NETLIB(name, sizes)
