#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

template <typename Benchmark>
static void StedcBenchSizes(Benchmark* b) {
    for (int n : {64, 128, 256}) {
        for (int batch : {128, 512, 2048}) {
            for (int flattened : {0, 1}) {
                b->Args({n, batch, 32, static_cast<int>(StedcMergeVariant::FusedCta), flattened, 0, 0});
            }
        }
    }
}

template <typename Benchmark>
static void StedcBenchSizesNetlib(Benchmark* b) {
    for (int n : {32, 64, 128}) {
        for (int batch : {1, 8, 32}) {
            b->Args({n, batch, 32, static_cast<int>(StedcMergeVariant::Baseline), 0, 0, 0});
        }
    }
}

// Batched STEDC benchmark
template <typename T, Backend B>
static void BM_STEDC(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t rec_threshold = state.range(2);
    const StedcMergeVariant merge_variant = static_cast<StedcMergeVariant>(state.range(3));
    const int threads_per_root = state.range(4) > 0 ? state.range(4) : 32;
    const int wg_multiplier = state.range(5) > 0 ? state.range(5) : 1;
    JobType jobz = JobType::EigenVectors;

    auto diags = Vector<T>::random(n, batch);
    auto off_diags = Vector<T>::random(n - 1, batch);

    StedcParams<T> params;
    params.recursion_threshold = rec_threshold;
    params.merge_variant = merge_variant;
    params.secular_threads_per_root = threads_per_root;
    params.secular_cta_wg_size_multiplier = wg_multiplier;

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto eigvals = Vector<T>::zeros(n, batch);
    auto eigvects = Matrix<T>::Identity(n, batch);
    UnifiedVector<std::byte> ws(stedc_workspace_size<B, T>(*q, n, batch, jobz, params));

    auto kernel = [q](auto& diags,
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
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_STEDC, StedcBenchSizes);

MINI_BENCHMARK_MAIN();
