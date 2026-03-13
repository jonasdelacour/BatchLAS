#include <util/minibench.hh>
#include <util/bench_structured.hh>

#include <blas/linalg.hh>
#include <batchlas/backend_config.h>

#include <memory>
#include <string>
#include <cstdlib>

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

struct HeterogeneousPattern {
    UnifiedVector<int> a_rows;
    UnifiedVector<int> a_cols;
    UnifiedVector<int> b_rows;
    UnifiedVector<int> b_cols;
    UnifiedVector<int> c_rows;
    UnifiedVector<int> c_cols;
    double total_flops = 0.0;
};

inline HeterogeneousPattern make_shrinking_pattern(int max_m, int max_n, int k, int batch) {
    HeterogeneousPattern pattern{
        UnifiedVector<int>(batch),
        UnifiedVector<int>(batch),
        UnifiedVector<int>(batch),
        UnifiedVector<int>(batch),
        UnifiedVector<int>(batch),
        UnifiedVector<int>(batch),
        0.0,
    };

    const int half_m = max_m / 2;
    const int half_n = max_n / 2;

    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        switch (batch_index % 3) {
        case 0:
            pattern.a_rows[batch_index] = max_m;
            pattern.a_cols[batch_index] = k;
            pattern.b_rows[batch_index] = k;
            pattern.b_cols[batch_index] = max_n;
            pattern.c_rows[batch_index] = max_m;
            pattern.c_cols[batch_index] = max_n;
            pattern.total_flops += 2.0 * static_cast<double>(max_m) * static_cast<double>(max_n) * static_cast<double>(k);
            break;
        case 1:
            pattern.a_rows[batch_index] = half_m;
            pattern.a_cols[batch_index] = k;
            pattern.b_rows[batch_index] = k;
            pattern.b_cols[batch_index] = half_n;
            pattern.c_rows[batch_index] = half_m;
            pattern.c_cols[batch_index] = half_n;
            pattern.total_flops += 2.0 * static_cast<double>(half_m) * static_cast<double>(half_n) * static_cast<double>(k);
            break;
        default:
            pattern.a_rows[batch_index] = 0;
            pattern.a_cols[batch_index] = k;
            pattern.b_rows[batch_index] = k;
            pattern.b_cols[batch_index] = 0;
            pattern.c_rows[batch_index] = 0;
            pattern.c_cols[batch_index] = 0;
            break;
        }
    }

    return pattern;
}

inline void apply_pattern(Matrix<float>& A,
                          Matrix<float>& B,
                          Matrix<float>& C,
                          const HeterogeneousPattern& pattern) {
    A.set_active_dims(pattern.a_rows.to_span(), pattern.a_cols.to_span());
    B.set_active_dims(pattern.b_rows.to_span(), pattern.b_cols.to_span());
    C.set_active_dims(pattern.c_rows.to_span(), pattern.c_cols.to_span());
}

inline void HeterogeneousGemmSizes(minibench::Benchmark* b) {
    b->Args({64, 64, 32, 4096});
    b->Args({128, 128, 32, 1024});
    b->Args({256, 256, 32, 256});
}

template <typename SetupEnv>
void run_heterogeneous_variant(minibench::State& state,
                               SetupEnv setup_env) {
    const int max_m = state.range(0);
    const int max_n = state.range(1);
    const int k = state.range(2);
    const int batch = state.range(3);

    auto q = std::make_shared<Queue>("gpu");
    auto A = std::make_shared<Matrix<float>>(Matrix<float>::Random(max_m, k, false, batch));
    auto B = std::make_shared<Matrix<float>>(Matrix<float>::Random(k, max_n, false, batch));
    auto C = std::make_shared<Matrix<float>>(Matrix<float>::Random(max_m, max_n, false, batch));

    const auto pattern = make_shrinking_pattern(max_m, max_n, k, batch);
    apply_pattern(*A, *B, *C, pattern);

    auto C0 = std::make_shared<Matrix<float>>(C->clone());
    auto env = setup_env();

    state.SetMetric("GFLOPS", pattern.total_flops * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);

    state.SetPrepare([q, A, B, C, C0]() {
        (void)A->view().set_access_device(*q);
        (void)B->view().set_access_device(*q);
        (void)C->view().set_access_device(*q);
        (void)C0->view().set_access_device(*q);
        (void)A->view().prefetch(*q);
        (void)B->view().prefetch(*q);
        (void)C->view().prefetch(*q);
        (void)C0->view().prefetch(*q);
        q->wait();
    });

    state.SetBeforeEachRun([q, C, C0]() {
        MatrixView<float>::copy(*q, C->view(), C0->view()).wait();
    });

    state.SetTimedKernelMs(bench::make_event_timed_kernel_ms(q, [q, A, B, C, env]() {
        gemm_heterogeneous<Backend::CUDA>(*q,
                                          A->view(),
                                          B->view(),
                                          C->view(),
                                          1.0f,
                                          1.0f,
                                          Transpose::NoTrans,
                                          Transpose::NoTrans,
                                          ComputePrecision::Default);
    }));
}

#if BATCHLAS_HAS_CUDA_BACKEND
static int register_cuda_heterogeneous_benchmarks = []() {
    HeterogeneousGemmSizes(minibench::RegisterBenchmark(
        "BM_GEMM_heterogeneous_vendor<float, Backend::CUDA>",
        [](minibench::State& state) {
            run_heterogeneous_variant(state, []() {
                return std::make_tuple(
                    std::make_shared<ScopedEnvVar>("BATCHLAS_GEMM_VARIANT", "vendor")
                );
            });
        }));

    HeterogeneousGemmSizes(minibench::RegisterBenchmark(
        "BM_GEMM_heterogeneous_cublasdx<float, Backend::CUDA>",
        [](minibench::State& state) {
            run_heterogeneous_variant(state, []() {
                return std::make_tuple(
                    std::make_shared<ScopedEnvVar>("BATCHLAS_GEMM_VARIANT", "cublasdx"),
                    std::make_shared<ScopedEnvVar>("BATCHLAS_GEMM_CUBLASDX_KERNEL", "cublasdx_nn")
                );
            });
        }));
    return 0;
}();
#endif

} // namespace

MINI_BENCHMARK_MAIN();