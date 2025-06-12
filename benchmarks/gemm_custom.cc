#include <blas/linalg.hh>
#include <util/simple_benchmark.hh>

using namespace batchlas;

SIMPLE_BENCHMARK(gemm_custom);

static void gemm_custom(simple_bench::State& state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);

    auto A = Matrix<float>::Random(m, k);
    auto B = Matrix<float>::Random(k, n);
    auto C = Matrix<float>::Zeros(m, n);

    Queue queue("gpu");

    state.SetMetric("GFLOPS", 1e-9 * 2.0 * m * n * k, true);

    for (auto _ : state) {
        gemm<Backend::CUDA>(queue, A.view(), B.view(), C.view(), 1.0f, 0.0f,
            Transpose::NoTrans, Transpose::NoTrans);
        queue.wait();
    }
}

static auto* bench_cfg = BENCHMARK_gemm_custom
    ->Args({100, 100, 100});

SIMPLE_BENCHMARK_MAIN();
