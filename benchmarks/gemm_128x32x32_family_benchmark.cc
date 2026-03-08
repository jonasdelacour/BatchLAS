#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>

#include "bench_utils.hh"

#include <cstdlib>
#include <memory>
#include <string>

using namespace batchlas;

namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name_);
        if (old) {
            had_old_ = true;
            old_value_ = old;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_old_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool had_old_ = false;
    std::string old_value_;
};

inline void Gemm128x32x32FamilyNnSizes(minibench::Benchmark* b) {
    b->Args({128, 128, 128, 4096});
    b->Args({256, 256, 256, 1024});
    b->Args({512, 512, 512, 512});
}

inline void Gemm128x32x32FamilyTransposeSizes(minibench::Benchmark* b) {
    b->Args({256, 128, 256, 1024});
}

inline void Gemm128x32x32FamilyLargeSplitKSizes(minibench::Benchmark* b) {
    b->Args({256, 256, 256, 1024});
    b->Args({512, 512, 512, 512});
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
    auto q = std::make_shared<Queue>("gpu");

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    1.0f,
                    1.0f,
                    transA,
                    transB,
                    [kernel_name](Queue& q, auto&&... xs) {
                        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
                        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", kernel_name);
                        ScopedEnvVar experimental("BATCHLAS_GEMM_EXPERIMENTAL", "1");
                        gemm<B, float>(q, std::forward<decltype(xs)>(xs)...);
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
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_s2_u1<float, Backend::CUDA>", "128x32x32_s2_u1",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_s2_u1_aligned<float, Backend::CUDA>", "128x32x32_s2_u1_aligned",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_s2_u1_generic<float, Backend::CUDA>", "128x32x32_s2_u1_generic",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_s2_u2<float, Backend::CUDA>", "128x32x32_s2_u2",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_s2_u2_tt8x4<float, Backend::CUDA>", "128x32x32_s2_u2_tt8x4",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_s2_u2_tt4x8<float, Backend::CUDA>", "128x32x32_s2_u2_tt4x8",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_persistent<float, Backend::CUDA>", "128x32x32_persistent",
                                                     Gemm128x32x32FamilyLargeSplitKSizes);
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_splitk4<float, Backend::CUDA>", "128x32x32_splitk4",
                                                     Gemm128x32x32FamilyLargeSplitKSizes);
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x32x32_s2_u1_tn<float, Backend::CUDA>", "128x32x32_s2_u1_tn",
                                                     Gemm128x32x32FamilyTransposeSizes,
                                                     Transpose::Trans,
                                                     Transpose::NoTrans);
    return 0;
}();
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
static int register_rocm_family_benchmarks = []() {
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s1_u1<float, Backend::ROCM>", "128x32x32_s1_u1",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s2_u1<float, Backend::ROCM>", "128x32x32_s2_u1",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s2_u1_aligned<float, Backend::ROCM>", "128x32x32_s2_u1_aligned",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s2_u1_generic<float, Backend::ROCM>", "128x32x32_s2_u1_generic",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s2_u2<float, Backend::ROCM>", "128x32x32_s2_u2",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s2_u2_tt8x4<float, Backend::ROCM>", "128x32x32_s2_u2_tt8x4",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s2_u2_tt4x8<float, Backend::ROCM>", "128x32x32_s2_u2_tt4x8",
                                                     Gemm128x32x32FamilyNnSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_persistent<float, Backend::ROCM>", "128x32x32_persistent",
                                                     Gemm128x32x32FamilyLargeSplitKSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_splitk4<float, Backend::ROCM>", "128x32x32_splitk4",
                                                     Gemm128x32x32FamilyLargeSplitKSizes);
    register_family_variant_benchmark<Backend::ROCM>("BM_GEMM_128x32x32_s2_u1_tn<float, Backend::ROCM>", "128x32x32_s2_u1_tn",
                                                     Gemm128x32x32FamilyTransposeSizes,
                                                     Transpose::Trans,
                                                     Transpose::NoTrans);
    return 0;
}();
#endif

} // namespace

MINI_BENCHMARK_MAIN();
