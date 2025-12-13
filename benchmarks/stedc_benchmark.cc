#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched STEQR benchmark
template <typename T, Backend B>
static void BM_STEQR(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t rec_threshold = state.range(2);
    const bool flat = state.range(3) != 0;
    
    JobType jobz = JobType::EigenVectors;

    auto diags = Vector<T>::ones(n, batch);
    auto off_diags = Vector<T>::ones(n - 1, batch);

    StedcParams<T> params;
    params.recursion_threshold = rec_threshold;

    auto eigvals = Vector<T>::zeros(n, batch);
    auto eigvects = Matrix<T>::Identity(n, batch);
    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    UnifiedVector<std::byte> ws(stedc_workspace_size<B, T>(*q, n, batch, jobz, params));

    // Register kernel-only runner so warmup uses the same USM allocations.
    state.SetKernel([=]() {
        if (flat) {
            stedc_flat<B>(*q, diags, off_diags, eigvals, ws.to_span(), jobz, params, eigvects);
        } else {
            stedc<B>(*q, diags, off_diags, eigvals, ws.to_span(), jobz, params, eigvects);    
        }
    });
    // Single wait after each batch of internal iterations to mirror prior behavior.
    state.SetBatchEndWait(q);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_STEQR, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
