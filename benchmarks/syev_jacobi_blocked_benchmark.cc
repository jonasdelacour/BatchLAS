#include <util/minibench.hh>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>

#include "bench_utils.hh"

#include <cstddef>
#include <cstdint>
#include <memory>

using namespace batchlas;

namespace {

// Head-to-head grid for the blocked Jacobi eigensolver against the routed syev
// path on identical inputs, at the batch sizes where each n saturates the
// device (the same ones syev_benchmark registers). Ratios taken below
// saturation measure launch overhead, not algorithms.
//
// arg0 n, arg1 batch, arg2 jobz, arg3 nb (0 = auto), arg4 inner sweeps.
template <typename Benchmark>
inline void SyevJacobiBlockedBenchSizes(Benchmark* b) {
    auto add = [&](int n, int batch) {
        for (int jobz : {0, 1}) {
            // inner: 0 = the shipped adaptive rule, 30 = exact block solves.
            for (int nb : {0, 8, 16, 24, 32}) {
                for (int inner : {0, 1, 2, 30}) {
                    b->Args({n, batch, jobz, nb, inner});
                }
            }
        }
    };
    add(64, 4096);
    add(128, 2048);
    add(256, 1024);
}

// The reference path does not care about nb or the inner sweep count, so it
// gets the same grid with those two collapsed.
template <typename Benchmark>
inline void SyevRefBenchSizes(Benchmark* b) {
    auto add = [&](int n, int batch) {
        for (int jobz : {0, 1}) {
            b->Args({n, batch, jobz, 0, 0});
        }
    };
    add(64, 4096);
    add(128, 2048);
    add(256, 1024);
}

inline JobType parse_jobz(int v) {
    return (v == 0) ? JobType::NoEigenVectors : JobType::EigenVectors;
}

// Operation count of the *reference* algorithm, so GFLOPS is comparable across
// the two rows. Jacobi does roughly an order of magnitude more arithmetic than
// this; the honest way to read the table is the per-matrix time column.
inline double ref_flops(std::size_t n) {
    return 4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
}

} // namespace

template <typename T, Backend B>
static void BM_SYEV_JACOBI_BLOCKED(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const JobType jobz = parse_jobz(static_cast<int>(state.range(2)));

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    JacobiParams<T> params;
    params.block_size = static_cast<size_t>(state.range(3));
    params.inner_sweeps = static_cast<size_t>(state.range(4));

    const size_t ws_size = syev_jacobi_blocked_buffer_size<B, T>(*q, A.view(), jobz, params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    jobz,
                    Uplo::Lower,
                    std::move(workspace),
                    params,
                    [](Queue& q, auto&&... xs) {
                        syev_jacobi_blocked(q, std::forward<decltype(xs)>(xs)...);
                    });

    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * ref_flops(n)), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// The routed syev, which at these shapes is the vendor path on CUDA.
template <typename T, Backend B>
static void BM_SYEV_ROUTED_REF(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const JobType jobz = parse_jobz(static_cast<int>(state.range(2)));

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    const size_t ws_size = syev_buffer_size(*q, A.view(), W.to_span(), jobz, Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    jobz,
                    Uplo::Lower,
                    std::move(workspace),
                    [](Queue& q, auto&&... xs) {
                        syev(q, std::forward<decltype(xs)>(xs)...);
                    });

    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * ref_flops(n)), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

#if BATCHLAS_HAS_CUDA_BACKEND
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_JACOBI_BLOCKED, SyevJacobiBlockedBenchSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_ROUTED_REF, SyevRefBenchSizes);
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_JACOBI_BLOCKED, SyevJacobiBlockedBenchSizes);
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_ROUTED_REF, SyevRefBenchSizes);
#endif

MINI_BENCHMARK_MAIN();
