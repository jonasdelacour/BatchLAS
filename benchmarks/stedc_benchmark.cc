#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched STEDC benchmark
template <typename T, Backend B>
static void BM_STEDC(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t rec_threshold = state.range(2);
    const bool flat = state.range(3) != 0;
    
    JobType jobz = JobType::EigenVectors;

    auto diags = Vector<T>::random(n, batch);
    auto off_diags = Vector<T>::random(n - 1, batch);

    StedcParams<T> params;
    params.recursion_threshold = rec_threshold;

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto eigvals = Vector<T>::zeros(n, batch);
    auto eigvects = Matrix<T>::Identity(n, batch);
    UnifiedVector<std::byte> ws(stedc_workspace_size<B, T>(*q, n, batch, jobz, params));

    auto kernel = [q, flat](auto& diags,
                            auto& off_diags,
                            auto& eigvals,
                            auto& ws,
                            JobType jobz,
                            StedcParams<T> params,
                            auto& eigvects) {
        auto d = static_cast<VectorView<T>>(diags);
        auto e = static_cast<VectorView<T>>(off_diags);
        auto w = static_cast<VectorView<T>>(eigvals);
        auto Z = eigvects.view();
        if (flat) {
            return stedc_flat<B, T>(*q, d, e, w, ws.to_span(), jobz, params, Z);
        }
        return stedc<B, T>(*q, d, e, w, ws.to_span(), jobz, params, Z);
    };
    state.SetKernel(q,
                    bench::pristine(diags),
                    bench::pristine(off_diags),
                    std::move(eigvals),
                    std::move(ws),
                    jobz,
                    params,
                    bench::pristine(eigvects),
                    kernel);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_STEDC, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
