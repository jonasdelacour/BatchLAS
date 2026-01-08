#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>
#include <util/sycl-device-queue.hh>
#include "../src/math-helpers.hh"

using namespace batchlas;

// Batched STEQR benchmark
template <typename T, Backend B>
static void BM_STEQR(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t inc = state.range(2);
    const size_t stride = state.range(3);
    const size_t n_writes = state.range(4);
    auto vec1 = Vector<T>::random(n, batch, stride, inc);
    auto vec2 = Vector<T>::random(n, batch, stride, inc);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    state.SetKernel(q,
                    bench::pristine(std::move(vec1)),
                    std::move(vec2),
                    n,
                    batch,
                    n_writes,
                    [](Queue& q,
                       auto&& vec1,
                       auto&& vec2,
                       size_t n,
                       size_t batch,
                       size_t n_writes) {
                        q->submit([&](sycl::handler& cgh) {
                            auto v1 = vec1;
                            auto v2 = vec2;
                            cgh.parallel_for(
                                sycl::nd_range<1>(
                                    sycl::range<1>(internal::ceil_div(batch, size_t(64)) * 64),
                                    sycl::range<1>(64)),
                                [=](sycl::nd_item<1> item) {
                                    size_t i = item.get_global_id(0);
                                    if (i >= batch) return;
                                    for (size_t w = 0; w < n_writes; w++) {
                                        for (size_t j = 0; j < n; j++) {
                                            v1(j, i) += v2(j, i) * v2(j, i);
                                        }
                                    }
                                });
                        });
                    });
    state.SetMetric("TFLOPS", static_cast<double>(batch) * (1e-12 * double(n) * double(n_writes) * 3.0), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_STEQR, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
