#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched STEQR benchmark
template <typename T, Backend B>
static void BM_STEHR(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);

    JobType jobz = JobType::EigenVectors;
    const double avg_deflations_per_eigenvalue = 2.5; // Empirical average
    const double flops_eigvals = avg_deflations_per_eigenvalue * double(n) * double(n) * 35.0/2.0; // Approximate number of flops to compute all eigenvalues.
    const double flops_eigvects = 3 * avg_deflations_per_eigenvalue * double(n) * double(n) * double(n) - 3 * avg_deflations_per_eigenvalue * double(n) * double(n);  //Extra flops if eigvects are required.

    const double total_flops = (flops_eigvals + (jobz == JobType::EigenVectors ? flops_eigvects : 0)) * double(batch);

    Vector<T> diags = Vector<T>::random(n, batch);
    Vector<T> off_diags = Vector<T>::random(n, batch);

    auto eigvals = Vector<T>::zeros(n, batch);
    auto eigvects = Matrix<T>::Identity(n, batch);
    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    UnifiedVector<std::byte> ws(tridiagonal_solver_buffer_size<B, T>(*q, n, batch, jobz));

    // Register kernel-only runner so warmup uses the same USM allocations.
    state.SetKernel([=]() {
        tridiagonal_solver<B>(*q, diags.data(), off_diags.data(), eigvals.data(), ws.to_span(), jobz, eigvects, n, batch);
    });
    // Single wait after each batch of internal iterations to mirror prior behavior.
    state.SetBatchEndWait(q);
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_STEHR, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
