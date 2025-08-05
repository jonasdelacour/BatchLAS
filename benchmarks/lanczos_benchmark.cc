#include <util/minibench.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <batchlas/backend_config.h>
#include "bench_utils.hh"
using namespace batchlas;

// Symmetric eigenvalue decomposition benchmark
template <typename T, Backend B>
static void BM_LANCZOS(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);

    auto A = Matrix<T>::TriDiagToeplitz(m, 1.0, 0.5, 0.5, batch);
    auto Acsr = A.template convert_to<MatrixFormat::CSR>();
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    auto V = Matrix<T>::Zeros(n, batch);
    LanczosParams<T> params;
    params.ortho_algorithm = OrthoAlgorithm::CGS2;

    size_t ws_size = lanczos_buffer_size<B>(queue, A.view(), W.to_span(),
                                            JobType::NoEigenVectors, V.view(), params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        lanczos<B>(queue, A.view(), W.to_span(),workspace.to_span(),
                   JobType::NoEigenVectors, V.view(), params);
    }
    queue.wait();
    state.StopTiming();
    double flops = 4.0 / 3.0 * static_cast<double>(n) * n * n;
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK(BM_LANCZOS, SquareBatchSizes);
MINI_BENCHMARK_MAIN();
