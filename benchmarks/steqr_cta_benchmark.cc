#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched STEQR benchmark
template <typename T, Backend B>
static void BM_STEQR_CTA(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t n_sweeps = state.range(2) > 0 ? state.range(2) : 10;
    const size_t wg_mult = state.range(3) > 0 ? state.range(3) : 1;
    const int shift_kind = state.range(4);
    //const int use_block_rotations = state.range(4);
    JobType jobz = JobType::EigenVectors;
    const double avg_deflations_per_eigenvalue = 2.5; // Empirical average
    const double flops_eigvals = avg_deflations_per_eigenvalue * double(n) * double(n) * 35.0/2.0; // Approximate number of flops to compute all eigenvalues.
    const double flops_eigvects = 3 * avg_deflations_per_eigenvalue * double(n) * double(n) * double(n) - 3 * avg_deflations_per_eigenvalue * double(n) * double(n);  //Extra flops if eigvects are required.

    const double total_flops = (flops_eigvals + (jobz == JobType::EigenVectors ? flops_eigvects : 0)) * double(batch);

    SteqrParams<T> params;
    params.max_sweeps = n_sweeps;
    params.cta_wg_size_multiplier = wg_mult;
    params.cta_shift_strategy = (shift_kind == 1) ? SteqrShiftStrategy::Wilkinson : SteqrShiftStrategy::Lapack;
    params.back_transform = false;
    params.block_rotations = false;
    params.transpose_working_vectors = false;

    Vector<T> diags = Vector<T>::random(n, batch);
    Vector<T> off_diags = Vector<T>::random(n - 1, batch);

    auto eigvals = Vector<T>::zeros(n, batch);
    auto eigvects = Matrix<T>::Identity(n, batch);
    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    UnifiedVector<std::byte> ws(steqr_cta_buffer_size<T>(*q, diags, off_diags, eigvals,
                                                    jobz, params));

    // Register kernel-only runner so warmup uses the same USM allocations.
    state.SetKernel([=]() {
        steqr_cta<B>(*q, diags, off_diags, eigvals,
                 ws.to_span(), jobz, params, eigvects);
    });
    // Single wait after each batch of internal iterations to mirror prior behavior.
    state.SetBatchEndWait(q);
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_STEQR_CTA, SteqrCtaBenchSizes);

MINI_BENCHMARK_MAIN();
