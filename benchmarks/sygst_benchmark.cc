// SPDX-License-Identifier: MIT
#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include <benchmarks/bench_utils.hh>
#include <batchlas/backend_config.h>

using namespace batchlas;

// sygst from potrf's factor of a random SPD B (factored once, untimed). ARGS: n batch itype uplo
// (uplo 0 = Lower, 1 = Upper). Pin a family with BATCHLAS_SYGST_ROUTE (blocked:nb=32, twosided, ...).
// evidence: docs/perf/sygvd.md#sygst-measured
template <typename T, Backend B>
static void BM_SYGST(minibench::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int batch = static_cast<int>(state.range(1));
    const int itype = static_cast<int>(state.range(2));
    const Uplo uplo = state.range(3) == 0 ? Uplo::Lower : Uplo::Upper;

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, true, batch);
    Matrix<T> F(n, n, batch);
    (void)F.view().fill_random_spd(*q, 12345);
    UnifiedVector<int32_t> info(static_cast<std::size_t>(batch), 0);
    UnifiedVector<std::byte> fws(std::max<std::size_t>(1, potrf_buffer_size<B, T>(*q, F.view(), uplo)));
    (void)potrf<B, T>(*q, F.view(), uplo, fws.to_span(), info.to_span());
    q->wait();
    UnifiedVector<std::byte> ws(std::max<std::size_t>(1, sygst_buffer_size<B, T>(*q, A.view(), F.view(), itype, uplo)));

    state.SetKernel(q, bench::pristine(A), std::move(F), itype, uplo, std::move(ws),
                    [](Queue& q, auto&&... xs) { (void)sygst(q, std::forward<decltype(xs)>(xs)...); });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * 1e-9 * static_cast<double>(n) * n * n, minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

template <typename Benchmark>
inline void SygstSizes(Benchmark* b) {
    for (int n : {32, 64, 128, 256, 480, 1024})
        for (int bs : {128, 1024}) b->Args({n, bs, 1, 0});
}
template <typename Benchmark>
inline void SygstSizesNetlib(Benchmark* b) {
    for (int n : {32, 128})
        for (int bs : {16}) b->Args({n, bs, 1, 0});
}

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_SYGST, SygstSizes);

MINI_BENCHMARK_MAIN();
