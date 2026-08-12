#include <algorithm>
#include <complex>

#include <batchlas/blas/device.hh>
#include <batchlas/blas/matrix.hh>

#include "bench_utils.hh"

#include "../src/queue.hh"

using namespace batchlas;

#ifndef DEVICE_BLAS_LEVEL1_MODE
#define DEVICE_BLAS_LEVEL1_MODE 1
#endif

#ifndef DEVICE_BLAS_LEVEL1_BENCHMARK_NAME
#define DEVICE_BLAS_LEVEL1_BENCHMARK_NAME "device_blas_level1_benchmark"
#endif

namespace {

template <typename Tag>
class DeviceBlasLevel1Kernel;

using DeviceBlasLevel1Scalar = std::complex<float>;
constexpr const char* kDeviceBlasLevel1ScalarName = "std::complex<float>";

inline std::string device_blas_backend_tags() {
    std::string tags;
#if BATCHLAS_HAS_CUDA_BACKEND
    tags += ", batchlas::Backend::CUDA";
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    tags += ", batchlas::Backend::ROCM";
#endif
#if BATCHLAS_HAS_MKL_BACKEND
    tags += ", batchlas::Backend::MKL";
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    tags += ", batchlas::Backend::NETLIB";
#endif
    return tags;
}

inline std::string device_blas_benchmark_name() {
    return std::string("(") + DEVICE_BLAS_LEVEL1_BENCHMARK_NAME + "<" + kDeviceBlasLevel1ScalarName + device_blas_backend_tags() + ">)";
}

inline std::size_t device_blas_level1_local_size(const Queue& queue) {
    const std::size_t max_work_group_size = queue.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    const std::size_t subgroup_size = std::max<std::size_t>(1, queue.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
    const std::size_t subgroup_count = std::max<std::size_t>(
        1,
        std::min<std::size_t>(queue.device().get_property(DeviceProperty::MAX_NUM_SUB_GROUPS),
                              static_cast<std::size_t>(device::detail::subgroup::kMaxSubgroupsPerWorkGroup)));
    return std::min(subgroup_size * subgroup_count, max_work_group_size);
}

template <typename KernelName, typename KernelFunc>
void launch_batched_level1_kernel(Queue& queue, int batch, std::size_t local_size, KernelFunc&& kernel) {
    auto kernel_fn = std::forward<KernelFunc>(kernel);
    queue->parallel_for<KernelName>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * local_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
            kernel_fn(item, static_cast<int>(item.get_group(0)));
        });
}

template <typename Benchmark>
inline void DeviceBlasLevel1Sizes(Benchmark* b) {
    b->Args({1024, 16384})
        ->Args({2048, 8192})
        ->Args({4096, 4096})
        ->Args({8192, 2048})
        ->Args({16384, 1024})
        ->Args({32768, 512});
}

constexpr double kComplexAxpyFlopsPerElement = 8.0;
constexpr double kComplexDotcFlopsPerElement = 8.0;
constexpr double kComplexScalFlopsPerElement = 6.0;

} // namespace

MINI_BENCHMARK(device_blas_level1_benchmark) {
    const int n = state.range(0);
    const int batch = state.range(1);

    auto queue = std::make_shared<Queue>(Device::default_device(), true);
    const std::size_t local_size = device_blas_level1_local_size(*queue);

#if DEVICE_BLAS_LEVEL1_MODE == 1
    auto x = Vector<DeviceBlasLevel1Scalar>::random(n, batch);
    auto y = Vector<DeviceBlasLevel1Scalar>::random(n, batch);
    state.SetKernel(queue,
                    std::move(x),
                    std::move(y),
                    [local_size](Queue& queue, auto x, auto y) {
                        launch_batched_level1_kernel<DeviceBlasLevel1Kernel<std::integral_constant<int, 100>>>(
                            queue, x.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                batchlas::device::axpy(item.get_group(), x.batch_item(bid), y.batch_item(bid), DeviceBlasLevel1Scalar(2.0f, -0.5f));
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * kComplexAxpyFlopsPerElement * n), minibench::Rate);
    state.SetMetric("GB/s", static_cast<double>(batch) * (3.0 * n * sizeof(DeviceBlasLevel1Scalar)) * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / vector", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_LEVEL1_MODE == 2
    auto x = Vector<DeviceBlasLevel1Scalar>::random(n, batch);
    auto y = Vector<DeviceBlasLevel1Scalar>::random(n, batch);
    auto result = Vector<DeviceBlasLevel1Scalar>::zeros(1, batch);
    state.SetKernel(queue,
                    std::move(x),
                    std::move(y),
                    std::move(result),
                    [local_size](Queue& queue, auto x, auto y, auto result) {
                        launch_batched_level1_kernel<DeviceBlasLevel1Kernel<std::integral_constant<int, 200>>>(
                            queue, x.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                const auto value = batchlas::device::dotc(item.get_group(), x.batch_item(bid), y.batch_item(bid));
                                if (item.get_local_linear_id() == 0) {
                                    result.batch_item(bid)(0) = value;
                                }
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * kComplexDotcFlopsPerElement * n), minibench::Rate);
    state.SetMetric("GB/s", static_cast<double>(batch) * (2.0 * n + 1.0) * sizeof(DeviceBlasLevel1Scalar) * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / vector", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_LEVEL1_MODE == 3
    auto x = Vector<DeviceBlasLevel1Scalar>::random(n, batch);
    state.SetKernel(queue,
                    std::move(x),
                    [local_size](Queue& queue, auto x) {
                        launch_batched_level1_kernel<DeviceBlasLevel1Kernel<std::integral_constant<int, 300>>>(
                            queue, x.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                batchlas::device::scal(item.get_group(), x.batch_item(bid), DeviceBlasLevel1Scalar(0.75f, 0.25f));
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * kComplexScalFlopsPerElement * n), minibench::Rate);
    state.SetMetric("GB/s", static_cast<double>(batch) * (2.0 * n * sizeof(DeviceBlasLevel1Scalar)) * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / vector", (1.0 / batch) * 1e6, minibench::Reciprocal);
#else
#error "Unsupported DEVICE_BLAS_LEVEL1_MODE"
#endif
}

namespace {

static auto* bench_device_blas_level1_benchmark = minibench::RegisterBenchmark(device_blas_benchmark_name(), device_blas_level1_benchmark);

template <typename Benchmark>
Benchmark* register_device_blas_level1_sizes(Benchmark* benchmark) {
    DeviceBlasLevel1Sizes(benchmark);
    return benchmark;
}

static auto* bench_device_blas_level1_benchmark_sizes = register_device_blas_level1_sizes(bench_device_blas_level1_benchmark);

} // namespace

MINI_BENCHMARK_MAIN();