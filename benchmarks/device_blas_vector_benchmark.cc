#include <algorithm>

#include <blas/device.hh>
#include <blas/matrix.hh>
#include "bench_utils.hh"

#include "../src/queue.hh"

using namespace batchlas;

#ifndef DEVICE_BLAS_VECTOR_MODE
#define DEVICE_BLAS_VECTOR_MODE 1
#endif

#ifndef DEVICE_BLAS_POLICY
#define DEVICE_BLAS_POLICY 0
#endif

#ifndef DEVICE_BLAS_BENCHMARK_NAME
#define DEVICE_BLAS_BENCHMARK_NAME "device_blas_vector_benchmark"
#endif

#ifndef DEVICE_BLAS_TRMV_UPLO
#define DEVICE_BLAS_TRMV_UPLO Lower
#endif

#ifndef DEVICE_BLAS_TRMV_TRANS
#define DEVICE_BLAS_TRMV_TRANS NoTrans
#endif

#ifndef DEVICE_BLAS_TRMV_DIAG
#define DEVICE_BLAS_TRMV_DIAG NonUnit
#endif

namespace {

template <typename Tag>
class DeviceBlasVectorKernel;

constexpr device::DeviceBlasPolicy kPolicy = static_cast<device::DeviceBlasPolicy>(DEVICE_BLAS_POLICY);
constexpr Uplo kTrmvUplo = Uplo::DEVICE_BLAS_TRMV_UPLO;
constexpr Transpose kTrmvTrans = Transpose::DEVICE_BLAS_TRMV_TRANS;
constexpr Diag kTrmvDiag = Diag::DEVICE_BLAS_TRMV_DIAG;

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
    return std::string("(") + DEVICE_BLAS_BENCHMARK_NAME + "<float" + device_blas_backend_tags() + ">)";
}

inline std::size_t device_blas_vector_local_size(const Queue& queue) {
    const std::size_t max_work_group_size = queue.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    const std::size_t subgroup_size = std::max<std::size_t>(1, queue.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
    const std::size_t subgroup_count = std::max<std::size_t>(
        1,
        std::min<std::size_t>(queue.device().get_property(DeviceProperty::MAX_NUM_SUB_GROUPS),
                              static_cast<std::size_t>(device::detail::subgroup::kMaxSubgroupsPerWorkGroup)));
    const std::size_t preferred_local_size = subgroup_size * subgroup_count;
    return std::min(preferred_local_size, max_work_group_size);
}

template <typename KernelName, typename KernelFunc>
void launch_batched_vector_kernel(Queue& queue, int batch, std::size_t local_size, KernelFunc&& kernel) {
    auto kernel_fn = std::forward<KernelFunc>(kernel);
    queue->parallel_for<KernelName>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * local_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
            kernel_fn(item, static_cast<int>(item.get_group(0)));
        });
}

MINI_BENCHMARK(device_blas_vector_benchmark) {
    const int a0 = state.range(0);
    const int a1 = state.range(1);
    const int batch = state.range(2);

    auto queue = std::make_shared<Queue>(Device::default_device(), true);
    const std::size_t local_size = device_blas_vector_local_size(*queue);

#if DEVICE_BLAS_VECTOR_MODE == 1
    auto A = Matrix<float>::Random(a0, a1, false, batch);
    auto x = Vector<float>::random(a1, batch);
    auto y = Vector<float>::zeros(a0, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(x),
                    std::move(y),
                    [local_size](Queue& queue, auto a, auto x, auto y) {
                        const auto a_view = a.kernel_view();
                        launch_batched_vector_kernel<DeviceBlasVectorKernel<std::integral_constant<int, 100 + DEVICE_BLAS_POLICY>>>(
                            queue, a.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                batchlas::device::gemv(
                                    item, a_view.batch_item(bid), x.batch_item(bid), y.batch_item(bid), 1.0f, 0.0f, Transpose::NoTrans, kPolicy);
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * a0 * a1), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_VECTOR_MODE == 2
    auto A = Matrix<float, MatrixFormat::Dense>::RandomTriangular(a0, kTrmvUplo, kTrmvDiag, batch);
    auto x = Vector<float>::random(a0, batch);
    auto y = Vector<float>::zeros(a0, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(x),
                    std::move(y),
                    [local_size](Queue& queue, auto a, auto x, auto y) {
                        const auto a_view = a.kernel_view();
                        launch_batched_vector_kernel<DeviceBlasVectorKernel<std::integral_constant<int, 200 + DEVICE_BLAS_POLICY>>>(
                            queue, a.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                batchlas::device::trmv(item,
                                                       a_view.batch_item(bid),
                                                       x.batch_item(bid),
                                                       y.batch_item(bid),
                                                       1.0f,
                                                       0.0f,
                                               kTrmvUplo,
                                               kTrmvTrans,
                                               kTrmvDiag,
                                                       kPolicy);
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * a0 * a0), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_VECTOR_MODE == 3
    auto A = Matrix<float>::Random(a0, a0, false, batch);
    auto x = Vector<float>::random(a0, batch);
    auto y = Vector<float>::zeros(a0, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(x),
                    std::move(y),
                    [local_size](Queue& queue, auto a, auto x, auto y) {
                        const auto a_view = a.kernel_view();
                        launch_batched_vector_kernel<DeviceBlasVectorKernel<std::integral_constant<int, 300 + DEVICE_BLAS_POLICY>>>(
                            queue, a.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                batchlas::device::symv(
                                    item, a_view.batch_item(bid), x.batch_item(bid), y.batch_item(bid), 1.0f, 0.0f, Uplo::Lower, kPolicy);
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * a0 * a0), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#else
#error "Unsupported DEVICE_BLAS_VECTOR_MODE"
#endif
}

} // namespace

static auto* bench_device_blas_vector_benchmark = minibench::RegisterBenchmark(device_blas_benchmark_name(), device_blas_vector_benchmark)
#if DEVICE_BLAS_VECTOR_MODE == 1
    ->Args({128, 128, 256})->Args({512, 128, 128})->Args({128, 512, 128});
#else
    ->Args({128, 128, 256})->Args({256, 256, 128})->Args({512, 512, 64});
#endif

MINI_BENCHMARK_MAIN();
