#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include "bench_utils.hh"
using namespace batchlas;

// Single TRSM benchmark
template <typename T, Backend B>
static void BM_TRSM(minibench::State& state) {
    // SquareBatchSizes (include/batchlas/util/minibench.hh:790) emits
    // Args({s, s, bs}) -- three arguments, (n, n, batch). This read used to be
    // range(1), i.e. the SECOND n, so `batch` was always equal to `n` and the
    // batch argument was silently ignored: every one of the ten batch rows per
    // size ran the identical problem. It is why the table showed a dead-flat
    // 0.0324 ms from batch 1 to batch 512. gemv_benchmark.cc:11-13 and
    // ormqr_benchmark.cc:11-13 read range(2) and were always correct; this file
    // was the only one that did not.
    const size_t n = state.range(0);
    const size_t batch = state.range(2);

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5), batch);
    auto Bm = Matrix<T>::Random(n, n, false, batch);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    state.SetKernel(q,
                    std::move(A),
                    bench::pristine(Bm),
                    T(1),
                    Side::Left,
                    Uplo::Lower,
                    Transpose::NoTrans,
                    Diag::NonUnit,
                    [](Queue& q, auto&&... xs) {
                        trsm(q, std::forward<decltype(xs)>(xs)...);
                    });
    // TRSM solves op(A) X = alpha B with A n x n and B n x q, which is n^2 * q
    // flops, not n^2. Here B is square so q == n. The old expression omitted q
    // and therefore understated the rate by exactly a factor of n -- 8.3 GFLOP/s
    // reported where the kernel was doing 518.
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_TRSM, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
