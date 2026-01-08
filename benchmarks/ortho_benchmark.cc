#include <util/minibench.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include "bench_utils.hh"

using namespace batchlas;

template <typename T, Backend B>
static void BM_Ortho(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);
    const OrthoAlgorithm algo = static_cast<OrthoAlgorithm>(state.range(3));

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T, MatrixFormat::Dense>::Random(m, n, false, batch);
    UnifiedVector<std::byte> workspace(ortho_buffer_size<B>(*q, A.view(), Transpose::NoTrans, algo));

    state.SetKernel(q,
                    bench::pristine(A),
                    Transpose::NoTrans,
                    std::move(workspace),
                    algo,
                        [](Queue& q, auto&&... xs) {
                            ortho<B, T>(q, std::forward<decltype(xs)>(xs)...);
                        });
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK(BM_Ortho, OrthoBenchSizes);

MINI_BENCHMARK_MAIN();
