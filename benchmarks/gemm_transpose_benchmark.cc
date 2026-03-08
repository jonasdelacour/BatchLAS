#include <util/minibench.hh>

#include <batchlas/backend_config.h>
#include <blas/linalg.hh>

#include "bench_utils.hh"

using namespace batchlas;

namespace {

inline Transpose transpose_from_arg(int value) {
    switch (value) {
    case 0:
        return Transpose::NoTrans;
    case 1:
        return Transpose::Trans;
    case 2:
        return Transpose::ConjTrans;
    default:
        throw std::invalid_argument("transpose arg must be 0=NoTrans, 1=Trans, or 2=ConjTrans");
    }
}

inline void GemmTransposeSizes(minibench::Benchmark* b) {
    for (int batch : {128, 512, 1024}) {
        b->Args({96, 80, 64, batch, 1, 1});
        b->Args({128, 128, 128, batch, 1, 1});
        b->Args({128, 64, 256, batch, 0, 1});
        b->Args({64, 128, 256, batch, 1, 0});
    }
}

inline void GemmTransposeSizesNetlib(minibench::Benchmark* b) {
    for (int batch : {1, 8, 32}) {
        b->Args({96, 80, 64, batch, 1, 1});
        b->Args({128, 128, 128, batch, 1, 1});
        b->Args({128, 64, 256, batch, 0, 1});
        b->Args({64, 128, 256, batch, 1, 0});
    }
}

template <typename T, Backend B>
static void BM_GEMM_TRANSPOSE(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t batch = state.range(3);
    const Transpose transA = transpose_from_arg(state.range(4));
    const Transpose transB = transpose_from_arg(state.range(5));

    const size_t a_rows = transA == Transpose::NoTrans ? m : k;
    const size_t a_cols = transA == Transpose::NoTrans ? k : m;
    const size_t b_rows = transB == Transpose::NoTrans ? k : n;
    const size_t b_cols = transB == Transpose::NoTrans ? n : k;

    auto A = Matrix<T>::Random(a_rows, a_cols, false, batch);
    auto Bm = Matrix<T>::Random(b_rows, b_cols, false, batch);
    auto C = Matrix<T>::Random(m, n, false, batch);
    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    T(1),
                    T(1),
                    transA,
                    transB,
                    [](Queue& q, auto&&... xs) {
                        gemm<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

} // namespace

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_GEMM_TRANSPOSE, GemmTransposeSizes);

MINI_BENCHMARK_MAIN();