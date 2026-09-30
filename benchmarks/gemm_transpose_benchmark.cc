#include <batchlas/util/minibench.hh>

#include <batchlas/backend_config.h>
#include <batchlas/blas/linalg.hh>

#include "bench_utils.hh"

#include <cstdlib>

using namespace batchlas;

namespace {

// Same reason as gemm_benchmark.cc: beta decides whether C is READ at all, so a
// beta=1-only harness cannot see an epilogue defect. Default 1 = prior behaviour.
inline double bench_beta() {
    if (const char* p = std::getenv("BATCHLAS_BENCH_BETA")) {
        return std::atof(p);
    }
    return 1.0;
}

// BATCHLAS_BENCH_LD_PAD, as in gemm_benchmark.cc: panel updates carry their
// parent's ld, so the transposed forms must be measurable at a strided ld too.
// Padded operands are filled deterministically (not left uninitialised).
inline int bench_ld_pad() {
    if (const char* p = std::getenv("BATCHLAS_BENCH_LD_PAD")) {
        return std::atoi(p);
    }
    return 0;
}

template <typename T>
Matrix<T> bench_operand(size_t rows, size_t cols, size_t batch) {
    const int pad = bench_ld_pad();
    if (pad == 0) {
        return Matrix<T>::Random(rows, cols, false, batch);
    }
    Matrix<T> M(static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(batch),
                static_cast<int>(rows) + pad);
    auto host = M.data();
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < host.size(); ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        host[i] = static_cast<T>(static_cast<double>(s) / 4294967296.0 - 0.5);
    }
    return M;
}

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

    auto A = bench_operand<T>(a_rows, a_cols, batch);
    auto Bm = bench_operand<T>(b_rows, b_cols, batch);
    auto C = bench_operand<T>(m, n, batch);
    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    T(1),
                    static_cast<T>(bench_beta()),
                    transA,
                    transB,
                    [](Queue& q, auto&&... xs) {
                        (void)gemm(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

} // namespace

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_GEMM_TRANSPOSE, GemmTransposeSizes);

MINI_BENCHMARK_MAIN();