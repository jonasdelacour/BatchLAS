#include <util/minibench.hh>
#include <util/bench_structured.hh>

#include <blas/linalg.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

#include <memory>

using namespace batchlas;

namespace {

inline void GemmSteadyCampaignSizes(minibench::Benchmark* b) {
    b->Args({128, 128, 128, 4096});
    b->Args({256, 256, 256, 1024});
    b->Args({512, 512, 512, 512});
    b->Args({512, 256, 512, 512});
    b->Args({512, 64, 512, 512});
}

inline void GemmSteadyCampaignSizesNetlib(minibench::Benchmark* b) {
    GemmSteadyCampaignSizes(b);
}

template <typename F>
auto make_host_timed_kernel_ms(std::shared_ptr<Queue> q, F&& kernel) {
    return [q = std::move(q), kernel = std::forward<F>(kernel)]() mutable -> double {
        return bench_time_region_ms([&] {
            kernel();
            q->wait();
        });
    };
}

template <typename T, Backend B, typename TimerFactory>
void configure_steady_gemm(minibench::State& state, TimerFactory&& timer_factory) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t batch = state.range(3);

    auto A = std::make_shared<Matrix<T>>(Matrix<T>::Random(m, k, false, batch));
    auto Bm = std::make_shared<Matrix<T>>(Matrix<T>::Random(k, n, false, batch));
    auto C = std::make_shared<Matrix<T>>(Matrix<T>::Random(m, n, false, batch));
    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    bench::ManagedInputs managed(q);
    managed.prepare(*A);
    managed.prepare(*Bm);
    managed.pristine(C);

    state.SetPrepare(managed.make_prepare_once());
    state.SetBeforeEachRun(managed.make_before_each_run());

    auto kernel_once = [q, A, Bm, C]() mutable {
        gemm<B, T>(*q, A->view(), Bm->view(), C->view(), T(1), T(1), Transpose::NoTrans, Transpose::NoTrans);
    };

    state.SetKernel(std::function<void()>(kernel_once));
    state.SetTimedKernelMs(timer_factory(q, kernel_once));
    state.SetBatchEndWait(q);
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

template <typename T, Backend B>
static void BM_GEMM_STEADY_EVENT(minibench::State& state) {
    configure_steady_gemm<T, B>(state, [](std::shared_ptr<Queue> q, auto kernel_once) {
        return bench::make_event_timed_kernel_ms(std::move(q), std::move(kernel_once));
    });
}

template <typename T, Backend B>
static void BM_GEMM_STEADY_HOST(minibench::State& state) {
    configure_steady_gemm<T, B>(state, [](std::shared_ptr<Queue> q, auto kernel_once) {
        return make_host_timed_kernel_ms(std::move(q), std::move(kernel_once));
    });
}

} // namespace

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_GEMM_STEADY_EVENT, GemmSteadyCampaignSizes);
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_GEMM_STEADY_HOST, GemmSteadyCampaignSizes);

MINI_BENCHMARK_MAIN();