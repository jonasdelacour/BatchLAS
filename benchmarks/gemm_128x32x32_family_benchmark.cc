#include <batchlas/util/env.hh>
#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>

#include "bench_utils.hh"

#include <cstdlib>
#include <memory>
#include <string>

using namespace batchlas;

namespace {

inline void Gemm128x32x32FamilyNnSizes(minibench::Benchmark* b) {
    b->Args({128, 128, 128, 4096});
    b->Args({256, 256, 256, 1024});
    b->Args({512, 512, 512, 512});
}

inline void Gemm128x32x32FamilyTransposeSizes(minibench::Benchmark* b) {
    b->Args({256, 128, 256, 1024});
}

template <Backend B>
void run_family_variant(minibench::State& state, const char* kernel_name, Transpose transA, Transpose transB) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t batch = state.range(3);

    const size_t a_rows = transA == Transpose::NoTrans ? m : k;
    const size_t a_cols = transA == Transpose::NoTrans ? k : m;
    const size_t b_rows = transB == Transpose::NoTrans ? k : n;
    const size_t b_cols = transB == Transpose::NoTrans ? n : k;

    auto A = Matrix<float>::Random(a_rows, a_cols, false, batch);
    auto Bm = Matrix<float>::Random(b_rows, b_cols, false, batch);
    auto C = Matrix<float>::Random(m, n, false, batch);
    auto q = std::make_shared<Queue>(Device("gpu"), B);

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    1.0f,
                    1.0f,
                    transA,
                    transB,
                    [kernel_name](Queue& q, auto&&... xs) {
                        ScopedEnvVar pin("BATCHLAS_GEMM_ROUTE", kernel_name);  // a gemm choice spelling
                        (void)gemm(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

template <Backend B>
void register_family_variant_benchmark(const char* benchmark_name,
                                       const char* kernel_name,
                                       void (*sizer)(minibench::Benchmark*),
                                       Transpose transA = Transpose::NoTrans,
                                       Transpose transB = Transpose::NoTrans) {
    sizer(minibench::RegisterBenchmark(benchmark_name, [=](minibench::State& state) {
        run_family_variant<B>(state, kernel_name, transA, transB);
    }));
}

#if BATCHLAS_HAS_CUDA_BACKEND
static int register_cuda_family_benchmarks = []() {
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_s2_u1<float, Backend::CUDA>", "reg:m=128:n=32:k=32:u=1",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_s2_u1_tn<float, Backend::CUDA>", "reg:m=128:n=32:k=32:u=1",
                                                     Gemm128x32x32FamilyTransposeSizes,
                                                     Transpose::Trans,
                                                     Transpose::NoTrans);
    return 0;
}();
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
static int register_rocm_family_benchmarks = []() {
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s2_u1<float, Backend::ROCM>", "reg:m=128:n=32:k=32:u=1",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s2_u1_tn<float, Backend::ROCM>", "reg:m=128:n=32:k=32:u=1",
                                                     Gemm128x32x32FamilyTransposeSizes,
                                                     Transpose::Trans,
                                                     Transpose::NoTrans);
    return 0;
}();
#endif

} // namespace

MINI_BENCHMARK_MAIN();
