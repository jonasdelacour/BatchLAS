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

inline void Gemm128x64x32FamilySizes(minibench::Benchmark* b) {
    b->Args({256, 256, 256, 1024});
    b->Args({512, 512, 512, 512});
    b->Args({512, 256, 512, 512});
    b->Args({512, 64, 512, 512});
}

template <Backend B>
void run_family_variant(minibench::State& state, const char* kernel_name) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t batch = state.range(3);

    auto A = Matrix<float>::Random(m, k, false, batch);
    auto Bm = Matrix<float>::Random(k, n, false, batch);
    auto C = Matrix<float>::Random(m, n, false, batch);
    auto q = std::make_shared<Queue>("gpu");

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    1.0f,
                    1.0f,
                    Transpose::NoTrans,
                    Transpose::NoTrans,
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
void register_family_variant_benchmark(const char* benchmark_name, const char* kernel_name) {
    Gemm128x64x32FamilySizes(minibench::RegisterBenchmark(benchmark_name, [=](minibench::State& state) {
        run_family_variant<B>(state, kernel_name);
    }));
}

#if BATCHLAS_HAS_CUDA_BACKEND
static int register_cuda_family_benchmarks = []() {
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x64x32_large<float, Backend::CUDA>", "128x64x32large");
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x64x32_large_u2<float, Backend::CUDA>", "128x64x32large_u2");
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x64x32_large_tt4x8<float, Backend::CUDA>", "128x64x32large_tt4x8");
    register_family_variant_benchmark<Backend::CUDA>("BM_GEMM_128x64x32_large_tt4x8_u2<float, Backend::CUDA>", "128x64x32large_tt4x8_u2");
    return 0;
}();
#endif

} // namespace

MINI_BENCHMARK_MAIN();