#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched MatrixView copy benchmark
// Exercises MatrixView::copy across batched dense matrices.
template <typename T, Backend B>
static void BM_MATRIX_COPY(minibench::State& state) {
    const int rows = state.range(0);
    const int cols = state.range(1);
    const int batch = state.range(2);
    const int ld  = state.range(3) > 0 ? state.range(3) : rows ;
    const int stride = state.range(4) > 0 ? state.range(4) : ld * cols ;

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    Matrix<T> src(rows, cols, batch, ld, stride);
    Matrix<T> dst(rows, cols, batch, ld, stride);

    state.SetKernel(q,
                    std::move(dst),
                    std::move(src),
                    [](Queue& q, auto&& dst, auto&& src) {
                        (void)MatrixView<T>::copy(q, dst, src);
                    });

    const double bytes_moved = static_cast<double>(rows) * cols * batch * sizeof(T) * 2.0;
    state.SetMetric("GB/s", bytes_moved * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK(BM_MATRIX_COPY, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
