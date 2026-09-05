#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

#include <cstdlib>

using namespace batchlas;

// beta was hardcoded to 1 here. It has to be settable, because beta is not a
// detail of the epilogue -- it decides whether C is READ at all, and a kernel
// can look completely different on the two sides of that. One kernel in this
// tree scored 26 instead of 41 TFLOP/s with an identical inner loop purely
// because its beta != 0 read of C was one scattered transaction per lane, and
// a beta == 0 measurement structurally cannot see that.
//
// Env rather than a fifth range arg so the registered size sets keep working
// unchanged: BATCHLAS_BENCH_BETA=0 or 1 (default 1, the prior behaviour).
static double bench_beta() {
    if (const char* p = std::getenv("BATCHLAS_BENCH_BETA")) {
        return std::atof(p);
    }
    return 1.0;
}

// Pad added to every leading dimension, so the UNALIGNED-ld case is measurable.
// It is not an exotic one: a panel is a sub-view carrying its parent's ld, so
// unaligned ld is what BatchLAS's own factorisations hand to gemm. Default 0
// keeps the prior behaviour (ld == rows).
static int bench_ld_pad() {
    if (const char* p = std::getenv("BATCHLAS_BENCH_LD_PAD")) {
        return std::atoi(p);
    }
    return 0;
}

template <typename T, Backend B>
static void BM_GEMM_IMPL(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t batch = state.range(3);
    const T beta = static_cast<T>(bench_beta());

    const int pad = bench_ld_pad();
    auto A = pad ? Matrix<T>(m, k, static_cast<int>(batch), static_cast<int>(m) + pad)
                 : Matrix<T>::Random(m, k, false, batch);
    auto Bm = pad ? Matrix<T>(k, n, static_cast<int>(batch), static_cast<int>(k) + pad)
                  : Matrix<T>::Random(k, n, false, batch);
    auto C = pad ? Matrix<T>(m, n, static_cast<int>(batch), static_cast<int>(m) + pad)
                 : Matrix<T>::Random(m, n, false, batch);
    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    T(1),
                    beta,
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    [](Queue& q, auto&&... xs) {
                        gemm(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// Batched GEMM benchmark
template <typename T, Backend B>
static void BM_GEMM(minibench::State& state) {
    BM_GEMM_IMPL<T, B>(state);
}

template <typename T, Backend B>
static void BM_GEMM_FIXED128(minibench::State& state) {
    BM_GEMM_IMPL<T, B>(state);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_GEMM, CubeBatchSizes);
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_GEMM_FIXED128, GemmSquareSizesFixedBatch128);

MINI_BENCHMARK_MAIN();
