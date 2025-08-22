#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched STEQR benchmark
template <typename T, Backend B>
static void BM_STEQR(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t m = state.range(1);
    const size_t batch = state.range(2);

    auto diags = Vector<T>::random(n, batch, 1, batch);
    auto off_diags = Vector<T>::random(n - 1, batch, 1, batch);
    //auto diags = Vector<T>::random(n, batch);
    //auto off_diags = Vector<T>::random(n - 1, batch);

    auto eigvals = Vector<T>::zeros(n, batch);
    auto eigvects = Matrix<T>::Identity(n, batch);
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    UnifiedVector<std::byte> ws(steqr_buffer_size<T>(queue, diags, off_diags, eigvals, JobType::EigenVectors, SteqrParams<T>{}));

    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        steqr<B>(queue, diags, off_diags, eigvals, ws.to_span());
    }
    queue.wait();
    state.StopTiming();
    //state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_STEQR, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
