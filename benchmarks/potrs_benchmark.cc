// SPDX-License-Identifier: MIT
// potrs: B := A^-1 B from a Cholesky factor made once, untimed, by the public potrf. Pin a family with
// BATCHLAS_POTRS_ROUTE=fused|trsm (unset: the tuned table's choice). evidence: docs/perf/potrs.md#potrs-selection
#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/settings.hh>
#include <batchlas/backend_config.h>
#include <benchmarks/bench_utils.hh>

#include <algorithm>
#include <cstdio>
using namespace batchlas;

// Saturating batches only (agent-guide §10): n through both fused register buckets, nrhs through the
// accumulator buckets 1, 2, 4, 8 and past the fused ceiling (16).
template <typename Benchmark>
inline void PotrsSizes(Benchmark* b) {
    for (int n : {4, 8, 16, 32, 64, 128, 256, 512})
        for (int nrhs : {1, 2, 4, 8, 16})
            for (int bs : {128, 1024, 8192}) b->Args({n, nrhs, bs});
}

template <typename Benchmark>
inline void PotrsSizesNetlib(Benchmark* b) {
    for (int n : {16, 64})
        for (int nrhs : {1, 8})
            for (int bs : {16, 128}) b->Args({n, nrhs, bs});
}

static void potrs_announce_route_env() {
    static bool done = false;
    if (done) return;
    done = true;
    const char* raw = batchlas::settings().routing.route("potrs").get();
    std::fprintf(stderr, raw && *raw ? "potrs route pinned: BATCHLAS_POTRS_ROUTE=%s (choices: auto | fused | trsm)\n"
                                     : "potrs route: unset (the tuned table's choice)%s\n", raw ? raw : "");
}

template <typename T, Backend B, Uplo U>
static void BM_POTRS_Body(minibench::State& state) {
    potrs_announce_route_env();
    const int n = static_cast<int>(state.range(0));
    const int nrhs = static_cast<int>(state.range(1));
    const int batch = static_cast<int>(state.range(2));

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    Matrix<T> A(n, n, batch);
    (void)A.view().fill_random_spd(*q, 12345);
    UnifiedVector<std::byte> ws(std::max<std::size_t>(1, potrf_buffer_size<B, T>(*q, A.view(), U)));
    UnifiedVector<std::int32_t> info(static_cast<std::size_t>(batch), 0);
    (void)potrf<B, T>(*q, A.view(), U, ws.to_span(), info.to_span());
    q->wait();
    auto Bm = Matrix<T>::Random(n, nrhs, false, batch);

    state.SetKernel(q, std::move(A), bench::pristine(Bm), U, [](Queue& qq, auto&&... xs) {
        (void)potrs(qq, std::forward<decltype(xs)>(xs)...);
    });
    // Two triangular solves: 2 n^2 nrhs flops (real convention for every type).
    state.SetMetric("GFLOPS", double(batch) * (1e-9 * 2.0 * double(n) * double(n) * double(nrhs)), minibench::Rate);
    state.SetMetric("Time (us) / matrix", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
}

template <typename T, Backend B>
static void BM_POTRS(minibench::State& s) { BM_POTRS_Body<T, B, Uplo::Lower>(s); }
template <typename T, Backend B>
static void BM_POTRS_Upper(minibench::State& s) { BM_POTRS_Body<T, B, Uplo::Upper>(s); }

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_POTRS, PotrsSizes);
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_POTRS_Upper, PotrsSizes);

MINI_BENCHMARK_MAIN();
