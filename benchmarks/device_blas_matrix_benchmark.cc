#include <algorithm>

#include <blas/device.hh>
#include <blas/matrix.hh>
#include "bench_utils.hh"

#include "../src/queue.hh"

using namespace batchlas;

#ifndef DEVICE_BLAS_MATRIX_MODE
#define DEVICE_BLAS_MATRIX_MODE 1
#endif

#ifndef DEVICE_BLAS_POLICY
#define DEVICE_BLAS_POLICY 0
#endif

#ifndef DEVICE_BLAS_BENCHMARK_NAME
#define DEVICE_BLAS_BENCHMARK_NAME "device_blas_matrix_benchmark"
#endif

namespace {

template <typename Tag>
class DeviceBlasMatrixKernel;

template <typename Tag>
class DeviceBlasMatrixTileKernel;

constexpr device::DeviceBlasPolicy kPolicy = static_cast<device::DeviceBlasPolicy>(DEVICE_BLAS_POLICY);

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

template <typename KernelName, typename KernelFunc>
void launch_batched_matrix_kernel(Queue& queue, int batch, std::size_t local_size, KernelFunc&& kernel) {
    auto kernel_fn = std::forward<KernelFunc>(kernel);
    queue->parallel_for<KernelName>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * local_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
            kernel_fn(item, static_cast<int>(item.get_group(0)));
        });
}

template <typename KernelName, typename KernelFunc>
void launch_batched_matrix_tile_kernel(Queue& queue, int rows, int cols, int batch, KernelFunc&& kernel) {
    constexpr std::size_t local_rows = static_cast<std::size_t>(device::detail::subgroup::kRegisterMatrixLocalRows);
    constexpr std::size_t local_cols = static_cast<std::size_t>(device::detail::subgroup::kRegisterMatrixLocalCols);
    constexpr std::size_t tile_rows = static_cast<std::size_t>(device::detail::subgroup::kRegisterMatrixTileM);
    constexpr std::size_t tile_cols = static_cast<std::size_t>(device::detail::subgroup::kRegisterMatrixTileN);
    const std::size_t group_rows = (static_cast<std::size_t>(rows) + tile_rows - 1) / tile_rows;
    const std::size_t group_cols = (static_cast<std::size_t>(cols) + tile_cols - 1) / tile_cols;
    auto kernel_fn = std::forward<KernelFunc>(kernel);
    queue->parallel_for<KernelName>(
        sycl::nd_range<3>(sycl::range<3>(static_cast<std::size_t>(batch), group_rows * local_rows, group_cols * local_cols),
                          sycl::range<3>(1, local_rows, local_cols)),
        [=](sycl::nd_item<3> item) {
            kernel_fn(item, static_cast<int>(item.get_group(0)));
        });
}

MINI_BENCHMARK(device_blas_matrix_benchmark) {
    const int n = state.range(0);
    const int batch = state.range(3);

    auto queue = std::make_shared<Queue>(Device::default_device(), true);
    const std::size_t local_size = std::min<std::size_t>(256, queue->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

#if DEVICE_BLAS_MATRIX_MODE == 1
    auto A = Matrix<float, MatrixFormat::Dense>::RandomTriangular(n, Uplo::Lower, Diag::NonUnit, batch);
    auto B = Matrix<float>::Random(n, n, false, batch);
    auto C = Matrix<float>::Zeros(n, n, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(B),
                    std::move(C),
                    [local_size](Queue& queue, auto a, auto b, auto c) {
                        const auto a_view = a.kernel_view();
                        const auto b_view = b.kernel_view();
                        const auto c_view = c.kernel_view();
                        if (kPolicy == device::DeviceBlasPolicy::Generic || local_size < 256) {
                            launch_batched_matrix_kernel<DeviceBlasMatrixKernel<std::integral_constant<int, 100 + DEVICE_BLAS_POLICY>>>(
                                queue, a.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                    batchlas::device::trmm(item,
                                                           a_view.batch_item(bid),
                                                           b_view.batch_item(bid),
                                                           c_view.batch_item(bid),
                                                           1.0f,
                                                           0.0f,
                                                           Side::Left,
                                                           Uplo::Lower,
                                                           Transpose::NoTrans,
                                                           Diag::NonUnit,
                                                           kPolicy);
                                });
                        } else {
                            launch_batched_matrix_tile_kernel<DeviceBlasMatrixTileKernel<std::integral_constant<int, 100 + DEVICE_BLAS_POLICY>>>(
                                queue, a.rows(), c.cols(), a.batch_size(), [=](sycl::nd_item<3> item, int bid) {
                                    batchlas::device::trmm(item,
                                                           a_view.batch_item(bid),
                                                           b_view.batch_item(bid),
                                                           c_view.batch_item(bid),
                                                           1.0f,
                                                           0.0f,
                                                           Side::Left,
                                                           Uplo::Lower,
                                                           Transpose::NoTrans,
                                                           Diag::NonUnit,
                                                           kPolicy);
                                });
                        }
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_MATRIX_MODE == 2
    auto A = Matrix<float>::Random(n, n, false, batch);
    auto B = Matrix<float>::Random(n, n, false, batch);
    auto C = Matrix<float>::Zeros(n, n, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(B),
                    std::move(C),
                    [local_size](Queue& queue, auto a, auto b, auto c) {
                        const auto a_view = a.kernel_view();
                        const auto b_view = b.kernel_view();
                        const auto c_view = c.kernel_view();
                        if (kPolicy == device::DeviceBlasPolicy::Generic || local_size < 256) {
                            launch_batched_matrix_kernel<DeviceBlasMatrixKernel<std::integral_constant<int, 200 + DEVICE_BLAS_POLICY>>>(
                                queue, a.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                    batchlas::device::symm(
                                        item, a_view.batch_item(bid), b_view.batch_item(bid), c_view.batch_item(bid), 1.0f, 0.0f, Side::Left, Uplo::Lower, kPolicy);
                                });
                        } else {
                            launch_batched_matrix_tile_kernel<DeviceBlasMatrixTileKernel<std::integral_constant<int, 200 + DEVICE_BLAS_POLICY>>>(
                                queue, c.rows(), c.cols(), a.batch_size(), [=](sycl::nd_item<3> item, int bid) {
                                    batchlas::device::symm(
                                        item, a_view.batch_item(bid), b_view.batch_item(bid), c_view.batch_item(bid), 1.0f, 0.0f, Side::Left, Uplo::Lower, kPolicy);
                                });
                        }
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_MATRIX_MODE == 3
    auto A = Matrix<float>::Random(n, n, false, batch);
    auto B = Matrix<float>::Random(n, n, false, batch);
    auto C = Matrix<float>::Zeros(n, n, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(B),
                    std::move(C),
                    [local_size](Queue& queue, auto a, auto b, auto c) {
                        const auto a_view = a.kernel_view();
                        const auto b_view = b.kernel_view();
                        const auto c_view = c.kernel_view();
                        if (kPolicy == device::DeviceBlasPolicy::Generic || local_size < 256) {
                            launch_batched_matrix_kernel<DeviceBlasMatrixKernel<std::integral_constant<int, 300 + DEVICE_BLAS_POLICY>>>(
                                queue, a.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                    batchlas::device::gemm(item,
                                                           a_view.batch_item(bid),
                                                           b_view.batch_item(bid),
                                                           c_view.batch_item(bid),
                                                           1.0f,
                                                           0.0f,
                                                           Transpose::NoTrans,
                                                           Transpose::NoTrans,
                                                           kPolicy);
                                });
                        } else {
                            launch_batched_matrix_tile_kernel<DeviceBlasMatrixTileKernel<std::integral_constant<int, 300 + DEVICE_BLAS_POLICY>>>(
                                queue, c.rows(), c.cols(), a.batch_size(), [=](sycl::nd_item<3> item, int bid) {
                                    batchlas::device::gemm(item,
                                                           a_view.batch_item(bid),
                                                           b_view.batch_item(bid),
                                                           c_view.batch_item(bid),
                                                           1.0f,
                                                           0.0f,
                                                           Transpose::NoTrans,
                                                           Transpose::NoTrans,
                                                           kPolicy);
                                });
                        }
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#else
#error "Unsupported DEVICE_BLAS_MATRIX_MODE"
#endif
}

} // namespace

static auto* bench_device_blas_matrix_benchmark = minibench::RegisterBenchmark(device_blas_benchmark_name(), device_blas_matrix_benchmark)
    ->Args({128, 128, 128, 128})->Args({256, 256, 256, 64})->Args({512, 512, 512, 32});

MINI_BENCHMARK_MAIN();
