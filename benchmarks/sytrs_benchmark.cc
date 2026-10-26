// SPDX-License-Identifier: MIT
// sytrs and sysv at saturating batch: Args (n, nrhs, batch). sytrs times the solve alone, from a
// random symmetric (complex: symmetric, not Hermitian) A factored once, untimed, by the routed sytrf;
// sysv times the factor and the solve. The family is the route pin: BATCHLAS_SYTRS_ROUTE (cta,
// blocked, wg) and BATCHLAS_SYSV_ROUTE (cta, compose). evidence: docs/perf/sytrf.md#sytrs-measured
#include <batchlas/util/minibench.hh>
#include <batchlas/blas/functions/sysv.hh>
#include <batchlas/blas/functions/sytrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/settings.hh>
#include <batchlas/backend_config.h>
#include <benchmarks/bench_utils.hh>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

using namespace batchlas;

namespace {

// Real-arithmetic flop counts for every type: sytrs 2 n^2 nrhs, sysv n^3 / 3 + 2 n^2 nrhs.
template <typename T>
Matrix<T> symmetric(Queue& q, int n, int batch) {
    auto A = Matrix<T>::Random(n, n, false, batch, 12345);
    (void)A.view().symmetrize(q, Uplo::Lower);
    q.wait();
    return A;
}

void announce(const char* op) {
    static bool done[2] = {false, false};
    const int i = std::string(op) == "sysv";
    if (done[i]) return;
    done[i] = true;
    try {
        const char* raw = settings().routing.route(op).get();
        std::fprintf(stderr, "%s route: %s\n", op, raw && *raw ? raw : "unset (the tuned table's choice)");
    } catch (const std::invalid_argument&) {  // a build where the op reads no route variable
        std::fprintf(stderr, "%s route: not routed in this build\n", op);
    }
}

}  // namespace

template <typename T, Backend B>
static void BM_SYTRS(minibench::State& state) {
    announce("sytrs");
    const int n = int(state.range(0)), nrhs = int(state.range(1)), batch = int(state.range(2));
    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = symmetric<T>(*q, n, batch);
    auto Bm = Matrix<T>::Random(n, nrhs, false, batch, 777);
    UnifiedVector<int64_t> piv(size_t(n) * batch, 0);
    {
        UnifiedVector<std::byte> fws(std::max<size_t>(1, sytrf_buffer_size<B, T>(*q, A.view(), Uplo::Lower)));
        (void)sytrf<B, T>(*q, A.view(), Uplo::Lower, piv.to_span(), fws.to_span());
        q->wait();
    }
    UnifiedVector<std::byte> ws(std::max<size_t>(1, sytrs_buffer_size<B, T>(*q, A.view(), Bm.view(), Uplo::Lower)));
    state.SetKernel(q, std::move(A), bench::pristine(Bm), Uplo::Lower, std::move(piv), std::move(ws),
                    [](Queue& qq, auto&&... xs) { (void)sytrs(qq, std::forward<decltype(xs)>(xs)...); });
    state.SetMetric("GFLOPS", double(batch) * 1e-9 * 2.0 * n * n * nrhs, minibench::Rate);
    state.SetMetric("Time (us) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

template <typename T, Backend B>
static void BM_SYSV(minibench::State& state) {
    announce("sysv");
    const int n = int(state.range(0)), nrhs = int(state.range(1)), batch = int(state.range(2));
    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = symmetric<T>(*q, n, batch);
    auto Bm = Matrix<T>::Random(n, nrhs, false, batch, 777);
    UnifiedVector<int64_t> piv(size_t(n) * batch, 0);
    UnifiedVector<std::byte> ws(std::max<size_t>(1, sysv_buffer_size<B, T>(*q, A.view(), Bm.view(), Uplo::Lower)));
    state.SetKernel(q, bench::pristine(A), bench::pristine(Bm), Uplo::Lower, std::move(piv), std::move(ws),
                    [](Queue& qq, auto&&... xs) { (void)sysv(qq, std::forward<decltype(xs)>(xs)...); });
    state.SetMetric("GFLOPS", double(batch) * 1e-9 * (double(n) * n * n / 3.0 + 2.0 * n * n * nrhs), minibench::Rate);
    state.SetMetric("Time (us) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// n 8..512, nrhs {1, 4, 16, 64}, batch {128, 1024, 8192}; cells over 4 GB of complex<float> operands
// (A, B and B's pristine copy) are dropped, so every type runs the same grid.
template <typename Benchmark>
inline void SytrsSizes(Benchmark* b) {
    for (int n : {8, 16, 32, 64, 128, 256, 512})
        for (int nrhs : {1, 4, 16, 64})
            for (int bs : {128, 1024, 8192}) {
                const double bytes = 8.0 * bs * (double(n) * n + 2.0 * n * nrhs);
                if (bytes <= 4.0e9) b->Args({n, nrhs, bs});
            }
}

template <typename Benchmark>
inline void SytrsSizesNetlib(Benchmark* b) {
    for (int n : {8, 32, 128})
        for (int nrhs : {1, 16}) b->Args({n, nrhs, 128});
}

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_SYTRS, SytrsSizes);
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_SYSV, SytrsSizes);

MINI_BENCHMARK_MAIN();
