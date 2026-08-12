#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/extra.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched TRANSPOSE benchmark
template <typename T, Backend B>
static void BM_TRANSPOSE(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);

    auto A = Matrix<T>::Random(m, n, false, batch);
    auto B_mat = Matrix<T>::Zeros(n, m, batch);

    state.SetKernel(q,
                    std::move(A),
                    std::move(B_mat),
                    [](Queue& q, auto&& A, auto&& B_mat) {
                        batchlas::transpose<T, MatrixFormat::Dense>(q, A, B_mat);
                    });
    state.SetMetric("GB/s", static_cast<double>(batch) * (1e-9 * n * m * sizeof(T)), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_TRANSPOSE, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
