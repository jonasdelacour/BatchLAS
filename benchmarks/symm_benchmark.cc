// SPDX-License-Identifier: MIT
#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include <benchmarks/bench_utils.hh>
#include <batchlas/backend_config.h>

using namespace batchlas;

namespace {

template <typename Scalar, Backend B>
void BM_SYMM(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(3);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<Scalar>::Random(n, n, false, batch);
    auto Bm = Matrix<Scalar>::Random(n, n, false, batch);
    auto C = Matrix<Scalar>::Random(n, n, false, batch);

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    Scalar(1),
                    Scalar(1),
                    Side::Left,
                    Uplo::Lower,
                    [](Queue& q, auto&&... xs) {
                        (void)symm(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

}  // namespace

BATCHLAS_REGISTER_BENCHMARK(BM_SYMM, CubeBatchSizes);

MINI_BENCHMARK_MAIN();
