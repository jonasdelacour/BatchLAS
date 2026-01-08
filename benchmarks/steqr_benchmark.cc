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
    const size_t n_sweeps = state.range(2) > 0 ? state.range(2) : 10;
    const size_t transpose_layout = state.range(3);
    //const int use_block_rotations = state.range(4);
    JobType jobz = JobType::EigenVectors;
    const double avg_deflations_per_eigenvalue = 2.5; // Empirical average
    const double flops_eigvals = avg_deflations_per_eigenvalue * double(n) * double(n) * 35.0/2.0; // Approximate number of flops to compute all eigenvalues.
    const double flops_eigvects = 3 * avg_deflations_per_eigenvalue * double(n) * double(n) * double(n) - 3 * avg_deflations_per_eigenvalue * double(n) * double(n);  //Extra flops if eigvects are required.

    const double total_flops = (flops_eigvals + (jobz == JobType::EigenVectors ? flops_eigvects : 0)) * double(batch);
    bool transpose = static_cast<bool>(transpose_layout);

    SteqrParams<T> params;
    params.max_sweeps = n_sweeps;
    params.back_transform = false;
    params.block_rotations = false;
    params.transpose_working_vectors = transpose;

    Vector<T> diags = Vector<T>::random(n, batch, transpose ? 1 : n, transpose ? batch : 1);
    Vector<T> off_diags = Vector<T>::random(n - 1, batch, transpose ? 1 : n - 1, transpose ? batch : 1);
    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto eigvals = Vector<T>::zeros(n, batch);
    auto eigvects = Matrix<T>::Identity(n, batch);
    const size_t ws_size = steqr_buffer_size<T>(*q, diags, off_diags, eigvals, jobz, params);
    UnifiedVector<std::byte> ws(ws_size);

    state.SetKernel(q,
                    bench::pristine(diags),
                    bench::pristine(off_diags),
                    std::move(eigvals),
                    std::move(ws),
                    jobz,
                    params,
                    bench::pristine(eigvects),
                    [](Queue& q, auto&&... xs) {
                        steqr<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_STEQR, SteqrBenchSizes);

MINI_BENCHMARK_MAIN();
