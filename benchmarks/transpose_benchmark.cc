#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <blas/extra.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched TRANSPOSE benchmark
template <typename T, Backend B>
static void BM_TRANSPOSE(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto A = Matrix<T>::Random(m, n, false, batch);
    auto B_mat = Matrix<T>::Zeros(n, m, batch);

    state.SetKernel([=]() {
        batchlas::transpose(*q, A, B_mat);
    });
    state.SetBatchEndWait(q);
    state.SetMetric("GB/s", static_cast<double>(batch) * (1e-9 * n * m * sizeof(T)), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_TRANSPOSE, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
