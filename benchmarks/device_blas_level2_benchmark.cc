#include <algorithm>
#include <complex>
#include <type_traits>

#include <batchlas/blas/device.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include "bench_utils.hh"

#include "../src/queue.hh"

using namespace batchlas;

#ifndef DEVICE_BLAS_LEVEL2_MODE
#define DEVICE_BLAS_LEVEL2_MODE 1
#endif

#ifndef DEVICE_BLAS_POLICY
#define DEVICE_BLAS_POLICY 0
#endif

#ifndef DEVICE_BLAS_BENCHMARK_NAME
#define DEVICE_BLAS_BENCHMARK_NAME "device_blas_level2_benchmark"
#endif

#ifndef DEVICE_BLAS_LEVEL2_COMPLEX
#define DEVICE_BLAS_LEVEL2_COMPLEX 0
#endif

#ifndef DEVICE_BLAS_LEVEL2_TRANS
#define DEVICE_BLAS_LEVEL2_TRANS 0
#endif

#ifndef DEVICE_BLAS_LEVEL2_TRMV_UPLO
#define DEVICE_BLAS_LEVEL2_TRMV_UPLO Lower
#endif

#ifndef DEVICE_BLAS_LEVEL2_TRMV_DIAG
#define DEVICE_BLAS_LEVEL2_TRMV_DIAG NonUnit
#endif

namespace {

template <typename Tag>
class DeviceBlasLevel2Kernel;

constexpr device::DeviceBlasPolicy kPolicy = static_cast<device::DeviceBlasPolicy>(DEVICE_BLAS_POLICY);
constexpr Transpose kLevel2Trans = static_cast<Transpose>(DEVICE_BLAS_LEVEL2_TRANS);
constexpr Uplo kTrmvUplo = Uplo::DEVICE_BLAS_LEVEL2_TRMV_UPLO;
constexpr Diag kTrmvDiag = Diag::DEVICE_BLAS_LEVEL2_TRMV_DIAG;

#if DEVICE_BLAS_LEVEL2_COMPLEX
using DeviceBlasLevel2Scalar = std::complex<float>;
constexpr const char* kDeviceBlasLevel2ScalarName = "std::complex<float>";
constexpr double kDeviceBlasGemvFlopScale = 8.0;
constexpr double kDeviceBlasHemvFlopScale = 8.0;
constexpr double kDeviceBlasTrmvFlopScale = 8.0;
#else
using DeviceBlasLevel2Scalar = float;
constexpr const char* kDeviceBlasLevel2ScalarName = "float";
constexpr double kDeviceBlasGemvFlopScale = 2.0;
constexpr double kDeviceBlasHemvFlopScale = 2.0;
constexpr double kDeviceBlasTrmvFlopScale = 2.0;
#endif

inline std::string device_blas_backend_tags() {
    // Only CUDA backend for device BLAS benchmarks
    return ", batchlas::Backend::CUDA";
}

inline std::string device_blas_benchmark_name() {
    return std::string("(") + DEVICE_BLAS_BENCHMARK_NAME + "<" + kDeviceBlasLevel2ScalarName + device_blas_backend_tags() + ">)";
}

inline std::size_t device_blas_level2_local_size(const Queue& queue) {
    const std::size_t max_work_group_size = queue.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    const std::size_t subgroup_size = std::max<std::size_t>(1, queue.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
    const std::size_t subgroup_count = std::max<std::size_t>(
        1,
        std::min<std::size_t>(queue.device().get_property(DeviceProperty::MAX_NUM_SUB_GROUPS),
                              static_cast<std::size_t>(device::detail::subgroup::kMaxSubgroupsPerWorkGroup)));
    const std::size_t preferred_local_size = subgroup_size * subgroup_count;
    return std::min(preferred_local_size, max_work_group_size);
}

inline device::DeviceBlasLaunchInfo device_blas_group_launch_info(const Queue& queue, std::size_t local_size) {
    (void)queue;
    return device::make_group_launch_info(static_cast<int>(local_size));
}

template <typename T, typename KernelName, typename KernelFunc>
void launch_batched_level2_kernel_with_workspace(Queue& queue,
                                                 int batch,
                                                 std::size_t local_size,
                                                 std::size_t workspace_elements,
                                                 KernelFunc&& kernel) {
    auto kernel_fn = std::forward<KernelFunc>(kernel);
    queue->submit([&](sycl::handler& h) {
        if (workspace_elements == 0) {
            h.parallel_for<KernelName>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * local_size), sycl::range<1>(local_size)),
                [=](sycl::nd_item<1> item) {
                    kernel_fn(item, static_cast<int>(item.get_group(0)), static_cast<T*>(nullptr));
                });
            return;
        }

        sycl::local_accessor<T, 1> workspace(sycl::range<1>(workspace_elements), h);
        h.parallel_for<KernelName>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * local_size), sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {
                kernel_fn(item, static_cast<int>(item.get_group(0)), batchlas::util::get_raw_ptr(workspace));
            });
    });
}

inline double dense_matrix_vector_bytes(int m, int n, int batch) {
    return static_cast<double>(batch) * (static_cast<double>(m) * n + n + m) * sizeof(DeviceBlasLevel2Scalar);
}

inline double symmetric_vector_bytes(int n, int batch) {
    return static_cast<double>(batch) * (0.5 * static_cast<double>(n) * (n + 1) + 2.0 * n) * sizeof(DeviceBlasLevel2Scalar);
}

inline double triangular_vector_bytes(int n, int batch) {
    return static_cast<double>(batch) * (0.5 * static_cast<double>(n) * (n + 1) + 2.0 * n) * sizeof(DeviceBlasLevel2Scalar);
}

template <typename Benchmark>
inline void DeviceBlasGemvSizes(Benchmark* b) {
    constexpr int rows[] = {32, 64, 96, 128, 192, 256, 384, 512};
    constexpr int cols[] = {1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512};
    constexpr int batches[] = {8192, 4096, 2048, 2048, 1024, 1024, 512, 512};
    for (int i = 0; i < 8; ++i) {
        for (int n : cols) {
            b->Args({rows[i], n, batches[i]});
        }
    }
}

template <typename Benchmark>
inline void DeviceBlasSquareLevel2Sizes(Benchmark* b) {
    constexpr int dims[] = {32, 64, 96, 128, 192, 256, 384, 512};
    constexpr int batches[] = {8192, 4096, 2048, 2048, 1024, 1024, 512, 512};
    for (int i = 0; i < 8; ++i) {
        b->Args({dims[i], batches[i]});
    }
}

} // namespace

MINI_BENCHMARK(device_blas_level2_benchmark) {
    auto queue = std::make_shared<Queue>(Device::default_device(), true);
    const std::size_t local_size = device_blas_level2_local_size(*queue);

#if DEVICE_BLAS_LEVEL2_MODE == 1
    const int m = state.range(0);
    const int n = state.range(1);
    const int batch = state.range(2);
    auto A = Matrix<DeviceBlasLevel2Scalar>::Random(m, n, false, batch);
    auto x = Vector<DeviceBlasLevel2Scalar>::random(kLevel2Trans == Transpose::NoTrans ? n : m, batch);
    auto y = Vector<DeviceBlasLevel2Scalar>::zeros(kLevel2Trans == Transpose::NoTrans ? m : n, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(x),
                    std::move(y),
                    [local_size](Queue& queue, auto a, auto x, auto y) {
                        const auto a_view = a.kernel_view();
                        const auto launch = device_blas_group_launch_info(queue, local_size);
                        const auto workspace_elements = batchlas::device::gemv_workspace_elements<DeviceBlasLevel2Scalar, kPolicy, kLevel2Trans>(
                            launch, a_view.rows(), a_view.cols());
                        launch_batched_level2_kernel_with_workspace<DeviceBlasLevel2Scalar,
                                                                    DeviceBlasLevel2Kernel<std::integral_constant<int, 100 + DEVICE_BLAS_POLICY + 10 * DEVICE_BLAS_LEVEL2_TRANS>>>(
                            queue, a.batch_size(), local_size, workspace_elements, [=](sycl::nd_item<1> item, int bid, DeviceBlasLevel2Scalar* workspace) {
                                batchlas::device::gemv<kPolicy, kLevel2Trans>(item.get_group(),
                                                                              a_view.batch_item(bid),
                                                                              x.batch_item(bid),
                                                                              y.batch_item(bid),
                                                                              DeviceBlasLevel2Scalar(1),
                                                                              DeviceBlasLevel2Scalar(0),
                                                                              workspace);
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * kDeviceBlasGemvFlopScale * m * n), minibench::Rate);
    state.SetMetric("GB/s", dense_matrix_vector_bytes(m, n, batch) * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_LEVEL2_MODE == 2
    const int n = state.range(0);
    const int batch = state.range(1);
    auto A = Matrix<DeviceBlasLevel2Scalar>::Random(n, n, std::is_same_v<DeviceBlasLevel2Scalar, std::complex<float>>, batch);
    auto x = Vector<DeviceBlasLevel2Scalar>::random(n, batch);
    auto y = Vector<DeviceBlasLevel2Scalar>::zeros(n, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(x),
                    std::move(y),
                    [local_size](Queue& queue, auto a, auto x, auto y) {
                        const auto a_view = a.kernel_view();
                        const auto launch = device_blas_group_launch_info(queue, local_size);
                        const auto workspace_elements = std::is_same_v<DeviceBlasLevel2Scalar, std::complex<float>>
                                ? batchlas::device::hemv_workspace_elements<DeviceBlasLevel2Scalar, kPolicy, Uplo::Lower>(
                                launch, a_view.rows())
                                : batchlas::device::symv_workspace_elements<DeviceBlasLevel2Scalar, kPolicy, Uplo::Lower>(
                                launch, a_view.rows());
                        launch_batched_level2_kernel_with_workspace<DeviceBlasLevel2Scalar,
                                                                    DeviceBlasLevel2Kernel<std::integral_constant<int, 200 + DEVICE_BLAS_POLICY + 10 * DEVICE_BLAS_LEVEL2_COMPLEX>>>(
                            queue, a.batch_size(), local_size, workspace_elements, [=](sycl::nd_item<1> item, int bid, DeviceBlasLevel2Scalar* workspace) {
                                if constexpr (std::is_same_v<DeviceBlasLevel2Scalar, std::complex<float>>) {
                                    batchlas::device::hemv<kPolicy, Uplo::Lower>(item.get_group(),
                                                                                 a_view.batch_item(bid),
                                                                                 x.batch_item(bid),
                                                                                 y.batch_item(bid),
                                                                                 DeviceBlasLevel2Scalar(1),
                                                                                 DeviceBlasLevel2Scalar(0),
                                                                                 workspace);
                                } else {
                                    batchlas::device::symv<kPolicy, Uplo::Lower>(item.get_group(),
                                                                                 a_view.batch_item(bid),
                                                                                 x.batch_item(bid),
                                                                                 y.batch_item(bid),
                                                                                 DeviceBlasLevel2Scalar(1),
                                                                                 DeviceBlasLevel2Scalar(0),
                                                                                 workspace);
                                }
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * kDeviceBlasHemvFlopScale * n * n), minibench::Rate);
    state.SetMetric("GB/s", symmetric_vector_bytes(n, batch) * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_LEVEL2_MODE == 3
    const int n = state.range(0);
    const int batch = state.range(1);
    auto A = Matrix<DeviceBlasLevel2Scalar, MatrixFormat::Dense>::RandomTriangular(n, kTrmvUplo, kTrmvDiag, batch);
    auto x = Vector<DeviceBlasLevel2Scalar>::random(n, batch);
    auto y = Vector<DeviceBlasLevel2Scalar>::zeros(n, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(x),
                    std::move(y),
                    [local_size](Queue& queue, auto a, auto x, auto y) {
                        const auto a_view = a.kernel_view();
                        launch_batched_level2_kernel<DeviceBlasLevel2Kernel<std::integral_constant<int, 300 + DEVICE_BLAS_POLICY + 10 * DEVICE_BLAS_LEVEL2_TRANS>>>(
                            queue, a.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                batchlas::device::trmv<kPolicy, kTrmvUplo, kLevel2Trans, kTrmvDiag>(
                                    item.get_group(),
                                    a_view.batch_item(bid),
                                    x.batch_item(bid),
                                    y.batch_item(bid),
                                    DeviceBlasLevel2Scalar(1),
                                    DeviceBlasLevel2Scalar(0));
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * kDeviceBlasTrmvFlopScale * 0.5 * n * (n + 1)), minibench::Rate);
    state.SetMetric("GB/s", triangular_vector_bytes(n, batch) * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#else
#error "Unsupported DEVICE_BLAS_LEVEL2_MODE"
#endif
}

namespace {

static auto* bench_device_blas_level2_benchmark = minibench::RegisterBenchmark(device_blas_benchmark_name(), device_blas_level2_benchmark);

template <typename Benchmark>
Benchmark* register_device_blas_level2_sizes(Benchmark* benchmark) {
    DeviceBlasGemvSizes(benchmark);
    return benchmark;
}

static auto* bench_device_blas_level2_benchmark_sizes = register_device_blas_level2_sizes(bench_device_blas_level2_benchmark);

} // namespace

MINI_BENCHMARK_MAIN();