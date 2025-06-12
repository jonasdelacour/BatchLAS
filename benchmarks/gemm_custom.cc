#include <blas/linalg.hh>
#include <util/simple_benchmark.hh>

using namespace batchlas;

SIMPLE_BENCHMARK(gemm_custom);

static void gemm_custom(simple_bench::State& state) {
    state.PauseTiming();
    size_t m = state.range(0);
    size_t n = state.range(1);
    size_t k = state.range(2);
    size_t batch_size = state.range(3);

    auto A = Matrix<float>::Random(m, k, false, batch_size);
    auto B = Matrix<float>::Random(k, n, false, batch_size);
    auto C = Matrix<float>::Zeros(m, n, batch_size);

    Queue queue("gpu");

    state.SetMetric("GFLOPS", batch_size * (1e-9 * 2.0 * m * n * k), true);
    state.ResumeTiming();
    for (auto _ : state) {
        gemm<Backend::CUDA>(queue, A.view(), B.view(), C.view(), 1.0f, 0.0f,
            Transpose::NoTrans, Transpose::NoTrans);
    }
    queue.wait();
    state.StopTiming();
}

static auto* bench_cfg = BENCHMARK_gemm_custom
    ->Args({32, 32, 32, 1280}) ->Args({64, 64, 64, 1280})
    ->Args({128, 128, 128, 1280}) ->Args({256, 256, 256, 1280})
    ->Args({512, 512, 512, 1280}) ->Args({1024, 1024, 1024, 1280});

SIMPLE_BENCHMARK_MAIN();
