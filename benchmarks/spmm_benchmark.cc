#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Single SPMM benchmark (CSR * Dense)
template <typename T, Backend B>
static void BM_SPMM(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t k = state.range(1);
    const size_t n = state.range(2);
    const size_t batch = state.range(3);

    auto A_dense = Matrix<T>::Random(m, k, false, batch);
    auto A = A_dense.template convert_to<MatrixFormat::CSR>();
    auto Bm = Matrix<T>::Random(k, n, false, batch);
    auto C = Matrix<T>::Random(m, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    size_t ws_size = spmm_buffer_size<B>(queue, A.view(), Bm.view(), C.view(),
                                        T(1), T(0), Transpose::NoTrans,
                                        Transpose::NoTrans);
    UnifiedVector<std::byte> workspace(ws_size);
    state.SetKernel([&]{
        spmm<B>(queue, A.view(), Bm.view(), C.view(), T(1), T(0),
                Transpose::NoTrans, Transpose::NoTrans, workspace.to_span());
    });
    state.SetBatchEnd([&]{ queue.wait(); });
    state.SetMetric("GFLOPS", static_cast<double>(batch) *
                        (1e-9 * 2.0 * A.nnz() * n), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



BATCHLAS_REGISTER_BENCHMARK(BM_SPMM, CubeBatchSizes);

MINI_BENCHMARK_MAIN();
