// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/backend_config.h>

// This header is intentionally benchmark-local and should stay lightweight.
// The structured-benchmark implementation lives in `include/batchlas/util/*`.
#include <batchlas/util/minibench_structured.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <algorithm>
#include <complex>
#include <cstdlib>

// beta of the gemm benchmarks, from BATCHLAS_BENCH_BETA (default 1). beta decides whether C is
// READ at all, so a beta=1-only harness cannot see an epilogue defect (see gemm_benchmark.cc).
inline double bench_beta() {
    if (const char* p = std::getenv("BATCHLAS_BENCH_BETA")) {
        return std::atof(p);
    }
    return 1.0;
}

#if BATCHLAS_HAS_CUDA_BACKEND
#define BATCHLAS_BENCH_CUDA(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::CUDA>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::CUDA>), sizes);

#define BATCHLAS_BENCH_CUDA_ALL_TYPES(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::CUDA>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::CUDA>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::CUDA>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::CUDA>), sizes);
#else
#define BATCHLAS_BENCH_CUDA(name, sizes)
#define BATCHLAS_BENCH_CUDA_ALL_TYPES(name, sizes)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
#define BATCHLAS_BENCH_ROCM(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::ROCM>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::ROCM>), sizes);

#define BATCHLAS_BENCH_ROCM_ALL_TYPES(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::ROCM>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::ROCM>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::ROCM>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::ROCM>), sizes);
#else
#define BATCHLAS_BENCH_ROCM(name, sizes)
#define BATCHLAS_BENCH_ROCM_ALL_TYPES(name, sizes)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
#define BATCHLAS_BENCH_NETLIB(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::NETLIB>), sizes##Netlib);

#define BATCHLAS_BENCH_NETLIB_ALL_TYPES(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_BENCHMARK_REGISTER_SIZES((name<double, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::NETLIB>), sizes##Netlib);
#else
#define BATCHLAS_BENCH_NETLIB(name, sizes)
#define BATCHLAS_BENCH_NETLIB_ALL_TYPES(name, sizes)
#endif

// Copies the lower band of batch item 0 of the dense n x n A into AB, which is (kd+1) x n:
// AB(r,j) = A(j+r, j), and rows past the matrix edge are zeroed.
template <typename Scalar>
inline void fill_lower_band_from_dense(const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& A,
                                       batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense> AB,
                                       int n,
                                       int kd) {
    for (int j = 0; j < n; ++j) {
        const int rmax = std::min(kd, n - 1 - j);
        for (int r = 0; r <= rmax; ++r) {
            AB(r, j, 0) = A(j + r, j, 0);
        }
        for (int r = rmax + 1; r <= kd; ++r) {
            AB(r, j, 0) = Scalar(0);
        }
    }
}

// Backend::SYCL (AdaptiveCpp Metal) has double kernels only when the device has fp64.
#if BATCHLAS_HAS_SYCL_BACKEND && BATCHLAS_HAS_DEVICE_FP64
#define BATCHLAS_BENCH_SYCL_FP64(name, sizes, ...) \
    MINI_BENCHMARK_REGISTER_SIZES((name<__VA_ARGS__, batchlas::Backend::SYCL>), sizes);
#else
#define BATCHLAS_BENCH_SYCL_FP64(name, sizes, ...)
#endif

#if BATCHLAS_HAS_SYCL_BACKEND
#define BATCHLAS_BENCH_SYCL(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<float, batchlas::Backend::SYCL>), sizes); \
    BATCHLAS_BENCH_SYCL_FP64(name, sizes, double)

#define BATCHLAS_BENCH_SYCL_ALL_TYPES(name, sizes) \
    BATCHLAS_BENCH_SYCL(name, sizes) \
    BATCHLAS_BENCH_SYCL_COMPLEX_TYPES(name, sizes)

#define BATCHLAS_BENCH_SYCL_COMPLEX_TYPES(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::SYCL>), sizes); \
    BATCHLAS_BENCH_SYCL_FP64(name, sizes, std::complex<double>)
#else
#define BATCHLAS_BENCH_SYCL(name, sizes)
#define BATCHLAS_BENCH_SYCL_ALL_TYPES(name, sizes)
#define BATCHLAS_BENCH_SYCL_COMPLEX_TYPES(name, sizes)
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
#define BATCHLAS_BENCH_CUDA_COMPLEX_TYPES(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::CUDA>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::CUDA>), sizes);
#else
#define BATCHLAS_BENCH_CUDA_COMPLEX_TYPES(name, sizes)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
#define BATCHLAS_BENCH_ROCM_COMPLEX_TYPES(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::ROCM>), sizes); \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::ROCM>), sizes);
#else
#define BATCHLAS_BENCH_ROCM_COMPLEX_TYPES(name, sizes)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
#define BATCHLAS_BENCH_NETLIB_COMPLEX_TYPES(name, sizes) \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<float>, batchlas::Backend::NETLIB>), sizes##Netlib); \
    MINI_BENCHMARK_REGISTER_SIZES((name<std::complex<double>, batchlas::Backend::NETLIB>), sizes##Netlib);
#else
#define BATCHLAS_BENCH_NETLIB_COMPLEX_TYPES(name, sizes)
#endif

// GFLOP count of `batch` gemms of m x n x k (2mnk flops each), for the gemm benchmarks' GFLOPS
// rate. Keep the operation order: the gemm benchmarks' numbers were recorded with it.
inline double gemm_gflop(std::size_t m, std::size_t n, std::size_t k, std::size_t batch) {
    return static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k);
}

// Flop count of one n x n tridiagonal reduction (4/3 n^3), for the syev/sytrd benchmarks' rates.
// Keep the operation order ((4/3) n) n n: the syev/sytrd benchmarks' numbers were recorded with it.
inline double tridiag_flops(std::size_t n) {
    return 4.0 / 3.0 * static_cast<double>(n) * n * n;
}

#define BATCHLAS_REGISTER_BENCHMARK(name, sizes) \
    BATCHLAS_BENCH_CUDA(name, sizes) \
    BATCHLAS_BENCH_ROCM(name, sizes) \
    BATCHLAS_BENCH_SYCL(name, sizes) \
    BATCHLAS_BENCH_NETLIB(name, sizes)

// Use this only for ops that support complex types.
#define BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(name, sizes) \
    BATCHLAS_BENCH_CUDA_ALL_TYPES(name, sizes) \
    BATCHLAS_BENCH_ROCM_ALL_TYPES(name, sizes) \
    BATCHLAS_BENCH_SYCL_ALL_TYPES(name, sizes) \
    BATCHLAS_BENCH_NETLIB_ALL_TYPES(name, sizes)

// Use this for ops that exist only for complex types, such as hemm.
#define BATCHLAS_REGISTER_BENCHMARK_COMPLEX_TYPES(name, sizes) \
    BATCHLAS_BENCH_CUDA_COMPLEX_TYPES(name, sizes) \
    BATCHLAS_BENCH_ROCM_COMPLEX_TYPES(name, sizes) \
    BATCHLAS_BENCH_SYCL_COMPLEX_TYPES(name, sizes) \
    BATCHLAS_BENCH_NETLIB_COMPLEX_TYPES(name, sizes)

// jobz argument convention of the syev benchmarks: 0 = NoEigenVectors, anything else (1) =
// EigenVectors. state.range() returns 0 for an absent argument, so omitting it means eigenvalues only.
inline batchlas::JobType parse_jobz(int v) {
    return (v == 0) ? batchlas::JobType::NoEigenVectors : batchlas::JobType::EigenVectors;
}

// uplo argument of the syev benchmarks: 0 = Lower, anything else (1) = Upper.
inline batchlas::Uplo parse_uplo(int v) {
    return (v == 0) ? batchlas::Uplo::Lower : batchlas::Uplo::Upper;
}

// jobu/jobvh argument of the gesvd cta/vendor benchmarks: 0 = None, anything else (1) = All.
// gesvd_blocked_benchmark keeps its own parse_job, which also maps 2 to Thin.
inline batchlas::SvdVectors parse_svd_job(int v) {
    return (v == 0) ? batchlas::SvdVectors::None : batchlas::SvdVectors::All;
}
