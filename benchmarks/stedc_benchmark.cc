#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

template <typename Benchmark>
static void StedcBenchSizes(Benchmark* b) {
    // 320 and 640 are here deliberately. This sweep used to register only
    // powers of two, where the level planner lands on leaf == threshold exactly
    // -- so the leaf-width cliff that cost syev 3.25x at n=320 was structurally
    // unsamplable, and shipped. Any n whose tree shape is interesting is a
    // non-power-of-two n; keep at least one in the registered sweep.
    for (int n : {64, 128, 256, 320, 640}) {
        for (int batch : {128, 512, 2048}) {
            for (int algorithm : {static_cast<int>(StedcAlgorithm::Levels),
                                  static_cast<int>(StedcAlgorithm::Recursive)}) {
                // Auto so the registered sweep measures what a caller such as
                // syev actually gets; pass 0/1/2 on the CLI to pin a variant.
                b->Args({n, batch, 32, static_cast<int>(StedcMergeVariant::Auto), 0, 0, algorithm});
            }
        }
    }
}

template <typename Benchmark>
static void StedcBenchSizesNetlib(Benchmark* b) {
    for (int n : {32, 64, 128}) {
        for (int batch : {1, 8, 32}) {
            b->Args({n, batch, 32, static_cast<int>(StedcMergeVariant::Baseline), 0, 0,
                     static_cast<int>(StedcAlgorithm::Levels)});
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
    // Pass non-positive values straight through: StedcParams treats <= 0 as
    // "use the BatchLAS tuning tables", which is what real callers such as syev
    // get. Substituting fixed fallbacks here silently benchmarked something the
    // library never actually runs by default.
    const int threads_per_root = static_cast<int>(state.range(4));
    const int wg_multiplier = static_cast<int>(state.range(5));
    const auto algorithm = static_cast<StedcAlgorithm>(state.range(6));
    JobType jobz = JobType::EigenVectors;

    auto diags = Vector<T>::random(n, batch);
    auto off_diags = Vector<T>::random(n - 1, batch);

    StedcParams<T> params;
    params.recursion_threshold = rec_threshold;
    params.merge_variant = merge_variant;
    params.secular_threads_per_root = threads_per_root;
    params.secular_cta_wg_size_multiplier = wg_multiplier;
    params.algorithm = algorithm;

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto eigvals = Vector<T>::zeros(n, batch);
    auto eigvects = Matrix<T>::Identity(n, batch);
    UnifiedVector<std::byte> ws(stedc_buffer_size(*q, n, batch, jobz, params));

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
        return stedc(*q, d, e, w, ws.to_span(), jobz, params, Z);
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
