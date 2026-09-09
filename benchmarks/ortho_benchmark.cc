#include <batchlas/util/minibench.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include "bench_utils.hh"

using namespace batchlas;

template <typename T, Backend B>
static void BM_Ortho(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);
    const OrthoAlgorithm algo = static_cast<OrthoAlgorithm>(state.range(3));

    // arg4 SELECTS THE TRSM SIDE, and without it half of trsm's routing table
    // is unreachable from the only caller that exercises it end to end.
    // ortho.cc:205,289 pick `is_A_trans ? Side::Left : Side::Right`, and
    // is_A_trans is just transA != NoTrans -- so a benchmark hardcoding NoTrans
    // measures Side::Right exclusively. The float Side::Left window is real
    // traffic for anyone calling ortho on row-vectors, and it had no
    // caller-level test at all until this argument existed. Defaults to 0 so
    // every existing invocation keeps its meaning.
    const Transpose tr = state.range(4) ? Transpose::Trans : Transpose::NoTrans;

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T, MatrixFormat::Dense>::Random(m, n, false, batch);
    UnifiedVector<std::byte> workspace(ortho_buffer_size(*q, A.view(), tr, algo));

    state.SetKernel(q,
                    bench::pristine(A),
                    tr,
                    std::move(workspace),
                    algo,
                        [](Queue& q, auto&&... xs) {
                            ortho(q, std::forward<decltype(xs)>(xs)...);
                        });
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// The registered grid keeps arg4 = 0, so every historical row means exactly
// what it did before. Side::Left is reached by passing args explicitly.
template <typename Benchmark>
inline void OrthoBenchSizesNoTrans(Benchmark* b) {
    for (int algo = 0; algo < static_cast<int>(batchlas::OrthoAlgorithm::NUM_ALGORITHMS); ++algo)
        for (int m : {64, 128, 256, 512, 1024})
            for (int n : {64, 128, 256, 512, 1024})
                for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512})
                    if (m >= n) b->Args({m, n, bs, algo, 0});
}

// BATCHLAS_REGISTER_BENCHMARK appends "Netlib" to the sizer name for the host
// backend, so the pair has to exist even though the host grid is much smaller.
template <typename Benchmark>
inline void OrthoBenchSizesNoTransNetlib(Benchmark* b) {
    for (int algo = 0; algo < static_cast<int>(batchlas::OrthoAlgorithm::NUM_ALGORITHMS); ++algo)
        for (int m : {16, 32, 64, 128})
            for (int n : {16, 32, 64, 128})
                for (int bs : {1, 10, 100})
                    if (m >= n) b->Args({m, n, bs, algo, 0});
}

BATCHLAS_REGISTER_BENCHMARK(BM_Ortho, OrthoBenchSizesNoTrans);

MINI_BENCHMARK_MAIN();
