#include <algorithm>

#include <blas/device.hh>
#include <blas/matrix.hh>
#include <util/sycl-local-accessor-helpers.hh>
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

#ifndef DEVICE_BLAS_MATRIX_TRANS
#define DEVICE_BLAS_MATRIX_TRANS 0
#endif

#ifndef DEVICE_BLAS_MATRIX_COMPLEX
#define DEVICE_BLAS_MATRIX_COMPLEX 0
#endif

#ifndef DEVICE_BLAS_MATRIX_HERMITIAN
#define DEVICE_BLAS_MATRIX_HERMITIAN 0
#endif

namespace {

template <typename Tag>
class DeviceBlasMatrixKernel;

template <typename Tag>
class DeviceBlasMatrixTileKernel;

constexpr device::DeviceBlasPolicy kPolicy = static_cast<device::DeviceBlasPolicy>(DEVICE_BLAS_POLICY);
constexpr Transpose kRank2kTrans = static_cast<Transpose>(DEVICE_BLAS_MATRIX_TRANS);

#if DEVICE_BLAS_MATRIX_COMPLEX
using DeviceBlasMatrixScalar = std::complex<float>;
constexpr const char* kDeviceBlasScalarName = "std::complex<float>";
constexpr double kDeviceBlasLevel3FlopScale = 8.0;
constexpr double kDeviceBlasRank2kFlopScale = 8.0;
#else
using DeviceBlasMatrixScalar = float;
constexpr const char* kDeviceBlasScalarName = "float";
constexpr double kDeviceBlasLevel3FlopScale = 2.0;
constexpr double kDeviceBlasRank2kFlopScale = 2.0;
#endif

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
    return std::string("(") + DEVICE_BLAS_BENCHMARK_NAME + "<" + kDeviceBlasScalarName + device_blas_backend_tags() + ">)";
}

inline std::size_t device_blas_rank_update_local_size(const Queue& queue) {
    const std::size_t max_work_group_size = queue.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    const std::size_t subgroup_size = std::max<std::size_t>(1, queue.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
    const std::size_t preferred =
        (kPolicy == device::DeviceBlasPolicy::Auto ||
         kPolicy == device::DeviceBlasPolicy::Subgroup16 ||
         kPolicy == device::DeviceBlasPolicy::Subgroup32)
            ? std::min<std::size_t>(256, max_work_group_size)
            : subgroup_size * 2;
    return std::min(preferred, max_work_group_size);
}

inline std::size_t device_blas_subgroup_size(const Queue& queue) {
    return std::max<std::size_t>(1, queue.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
}

inline device::DeviceBlasLaunchInfo device_blas_nd_item_1d_launch_info(const Queue& queue, std::size_t local_size) {
    return device::make_nd_item_1d_launch_info(static_cast<int>(local_size), static_cast<int>(device_blas_subgroup_size(queue)));
}

inline device::DeviceBlasLaunchInfo device_blas_nd_item_3d_launch_info(const Queue& queue) {
    return device::make_nd_item_3d_launch_info(
        device::detail::subgroup::kRegisterMatrixThreadsPerGroup,
        static_cast<int>(device_blas_subgroup_size(queue)));
}

template <typename T, typename KernelName, typename KernelFunc>
void launch_batched_matrix_kernel_with_workspace(Queue& queue,
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

template <typename T, typename KernelName, typename KernelFunc>
void launch_batched_matrix_tile_kernel_with_workspace(Queue& queue,
                                                      int rows,
                                                      int cols,
                                                      int batch,
                                                      std::size_t workspace_elements,
                                                      KernelFunc&& kernel) {
    constexpr std::size_t local_rows = static_cast<std::size_t>(device::detail::subgroup::kRegisterMatrixLocalRows);
    constexpr std::size_t local_cols = static_cast<std::size_t>(device::detail::subgroup::kRegisterMatrixLocalCols);
    constexpr std::size_t tile_rows = static_cast<std::size_t>(device::detail::subgroup::kRegisterMatrixTileM);
    constexpr std::size_t tile_cols = static_cast<std::size_t>(device::detail::subgroup::kRegisterMatrixTileN);
    const std::size_t group_rows = (static_cast<std::size_t>(rows) + tile_rows - 1) / tile_rows;
    const std::size_t group_cols = (static_cast<std::size_t>(cols) + tile_cols - 1) / tile_cols;
    auto kernel_fn = std::forward<KernelFunc>(kernel);

    queue->submit([&](sycl::handler& h) {
        if (workspace_elements == 0) {
            h.parallel_for<KernelName>(
                sycl::nd_range<3>(sycl::range<3>(static_cast<std::size_t>(batch), group_rows * local_rows, group_cols * local_cols),
                                  sycl::range<3>(1, local_rows, local_cols)),
                [=](sycl::nd_item<3> item) {
                    kernel_fn(item, static_cast<int>(item.get_group(0)), static_cast<T*>(nullptr));
                });
            return;
        }

        sycl::local_accessor<T, 1> workspace(sycl::range<1>(workspace_elements), h);
        h.parallel_for<KernelName>(
            sycl::nd_range<3>(sycl::range<3>(static_cast<std::size_t>(batch), group_rows * local_rows, group_cols * local_cols),
                              sycl::range<3>(1, local_rows, local_cols)),
            [=](sycl::nd_item<3> item) {
                kernel_fn(item, static_cast<int>(item.get_group(0)), batchlas::util::get_raw_ptr(workspace));
            });
    });
}

MINI_BENCHMARK(device_blas_matrix_benchmark) {
    const int n = state.range(0);
    const int k = state.range(1);
    const int batch = std::max(1, state.range(3) != 0 ? state.range(3) : state.range(2));
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
                            const auto launch = device_blas_nd_item_1d_launch_info(queue, local_size);
                            const auto workspace_elements = batchlas::device::trmm_workspace_elements<float,
                                                                                                       kPolicy,
                                                                                                       Side::Left,
                                                                                                       Uplo::Lower,
                                                                                                       Transpose::NoTrans,
                                                                                                       Diag::NonUnit>(
                                launch, c_view.rows(), c_view.cols(), false);
                            launch_batched_matrix_kernel_with_workspace<float,
                                                                        DeviceBlasMatrixKernel<std::integral_constant<int, 100 + DEVICE_BLAS_POLICY>>>(
                                queue, a.batch_size(), local_size, workspace_elements, [=](sycl::nd_item<1> item, int bid, float* workspace) {
                                        batchlas::device::trmm<kPolicy,
                                                       Side::Left,
                                                       Uplo::Lower,
                                                       Transpose::NoTrans,
                                                      Diag::NonUnit>(item,
                                                              a_view.batch_item(bid),
                                                              b_view.batch_item(bid),
                                                              c_view.batch_item(bid),
                                                              1.0f,
                                                              0.0f,
                                                              workspace);
                                });
                        } else {
                            const auto launch = device_blas_nd_item_3d_launch_info(queue);
                            const auto workspace_elements = batchlas::device::trmm_workspace_elements<float,
                                                                                                       kPolicy,
                                                                                                       Side::Left,
                                                                                                       Uplo::Lower,
                                                                                                       Transpose::NoTrans,
                                                                                                       Diag::NonUnit>(
                                launch, c_view.rows(), c_view.cols(), false);
                            launch_batched_matrix_tile_kernel_with_workspace<float,
                                                                             DeviceBlasMatrixTileKernel<std::integral_constant<int, 100 + DEVICE_BLAS_POLICY>>>(
                                queue, a.rows(), c.cols(), a.batch_size(), workspace_elements, [=](sycl::nd_item<3> item, int bid, float* workspace) {
                                        batchlas::device::trmm<kPolicy,
                                                       Side::Left,
                                                       Uplo::Lower,
                                                       Transpose::NoTrans,
                                                      Diag::NonUnit>(item,
                                                              a_view.batch_item(bid),
                                                              b_view.batch_item(bid),
                                                              c_view.batch_item(bid),
                                                              1.0f,
                                                              0.0f,
                                                              workspace);
                                });
                        }
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_MATRIX_MODE == 2
    auto A = Matrix<DeviceBlasMatrixScalar>::Random(n, n, false, batch);
    auto B = Matrix<DeviceBlasMatrixScalar>::Random(n, n, false, batch);
    auto C = Matrix<DeviceBlasMatrixScalar>::Zeros(n, n, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(B),
                    std::move(C),
                    [local_size](Queue& queue, auto a, auto b, auto c) {
                        const auto a_view = a.kernel_view();
                        const auto b_view = b.kernel_view();
                        const auto c_view = c.kernel_view();
                        if (kPolicy == device::DeviceBlasPolicy::Generic || local_size < 256 ||
                            !std::is_same_v<DeviceBlasMatrixScalar, float>) {
                            const auto launch = device_blas_nd_item_1d_launch_info(queue, local_size);
                            const auto workspace_elements = batchlas::device::symm_workspace_elements<DeviceBlasMatrixScalar, kPolicy, Side::Left, Uplo::Lower>(
                                launch, c_view.rows(), c_view.cols());
                            launch_batched_matrix_kernel_with_workspace<DeviceBlasMatrixScalar,
                                                                        DeviceBlasMatrixKernel<std::integral_constant<int, 200 + DEVICE_BLAS_POLICY>>>(
                                queue, a.batch_size(), local_size, workspace_elements, [=](sycl::nd_item<1> item, int bid, DeviceBlasMatrixScalar* workspace) {
                                    batchlas::device::symm<kPolicy, Side::Left, Uplo::Lower>(
                                        item,
                                        a_view.batch_item(bid),
                                        b_view.batch_item(bid),
                                        c_view.batch_item(bid),
                                        DeviceBlasMatrixScalar(1),
                                        DeviceBlasMatrixScalar(0),
                                        workspace);
                                });
                        } else {
                            const auto launch = device_blas_nd_item_3d_launch_info(queue);
                            const auto workspace_elements = batchlas::device::symm_workspace_elements<DeviceBlasMatrixScalar, kPolicy, Side::Left, Uplo::Lower>(
                                launch, c_view.rows(), c_view.cols());
                            launch_batched_matrix_tile_kernel_with_workspace<DeviceBlasMatrixScalar,
                                                                             DeviceBlasMatrixTileKernel<std::integral_constant<int, 200 + DEVICE_BLAS_POLICY>>>(
                                queue, c.rows(), c.cols(), a.batch_size(), workspace_elements, [=](sycl::nd_item<3> item, int bid, DeviceBlasMatrixScalar* workspace) {
                                    batchlas::device::symm<kPolicy, Side::Left, Uplo::Lower>(
                                        item,
                                        a_view.batch_item(bid),
                                        b_view.batch_item(bid),
                                        c_view.batch_item(bid),
                                        DeviceBlasMatrixScalar(1),
                                        DeviceBlasMatrixScalar(0),
                                        workspace);
                                });
                        }
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * kDeviceBlasLevel3FlopScale * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_MATRIX_MODE == 3
    auto A = Matrix<DeviceBlasMatrixScalar>::Random(n, n, false, batch);
    auto B = Matrix<DeviceBlasMatrixScalar>::Random(n, n, false, batch);
    auto C = Matrix<DeviceBlasMatrixScalar>::Zeros(n, n, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(B),
                    std::move(C),
                    [local_size](Queue& queue, auto a, auto b, auto c) {
                        const auto a_view = a.kernel_view();
                        const auto b_view = b.kernel_view();
                        const auto c_view = c.kernel_view();
                        if (kPolicy == device::DeviceBlasPolicy::Generic || local_size < 256 ||
                            !std::is_same_v<DeviceBlasMatrixScalar, float>) {
                            const auto launch = device_blas_nd_item_1d_launch_info(queue, local_size);
                            const auto workspace_elements = batchlas::device::gemm_workspace_elements<DeviceBlasMatrixScalar, kPolicy, Transpose::NoTrans, Transpose::NoTrans>(
                                launch,
                                c_view.rows(),
                                c_view.cols(),
                                a_view.cols(),
                                false,
                                false);
                            launch_batched_matrix_kernel_with_workspace<DeviceBlasMatrixScalar,
                                                                        DeviceBlasMatrixKernel<std::integral_constant<int, 300 + DEVICE_BLAS_POLICY>>>(
                                queue, a.batch_size(), local_size, workspace_elements, [=](sycl::nd_item<1> item, int bid, DeviceBlasMatrixScalar* workspace) {
                                    batchlas::device::gemm<kPolicy, Transpose::NoTrans, Transpose::NoTrans>(
                                        item,
                                        a_view.batch_item(bid),
                                        b_view.batch_item(bid),
                                        c_view.batch_item(bid),
                                        DeviceBlasMatrixScalar(1),
                                        DeviceBlasMatrixScalar(0),
                                        workspace);
                                });
                        } else {
                            const auto launch = device_blas_nd_item_3d_launch_info(queue);
                            const auto workspace_elements = batchlas::device::gemm_workspace_elements<DeviceBlasMatrixScalar, kPolicy, Transpose::NoTrans, Transpose::NoTrans>(
                                launch,
                                c_view.rows(),
                                c_view.cols(),
                                a_view.cols(),
                                false,
                                false);
                            launch_batched_matrix_tile_kernel_with_workspace<DeviceBlasMatrixScalar,
                                                                             DeviceBlasMatrixTileKernel<std::integral_constant<int, 300 + DEVICE_BLAS_POLICY>>>(
                                queue, c.rows(), c.cols(), a.batch_size(), workspace_elements, [=](sycl::nd_item<3> item, int bid, DeviceBlasMatrixScalar* workspace) {
                                    batchlas::device::gemm<kPolicy, Transpose::NoTrans, Transpose::NoTrans>(
                                        item,
                                        a_view.batch_item(bid),
                                        b_view.batch_item(bid),
                                        c_view.batch_item(bid),
                                        DeviceBlasMatrixScalar(1),
                                        DeviceBlasMatrixScalar(0),
                                        workspace);
                                });
                        }
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * kDeviceBlasLevel3FlopScale * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_MATRIX_MODE == 4
    auto x = Vector<float>::random(n, batch);
    auto y = Vector<float>::random(k, batch);
    auto A = Matrix<float>::Zeros(n, k, batch);
    state.SetKernel(queue,
                    std::move(x),
                    std::move(y),
                    std::move(A),
                    [](Queue& queue, auto xvec, auto yvec, auto a) {
                        const auto a_view = a.kernel_view();
                        const std::size_t local_size = device_blas_rank_update_local_size(queue);
                        launch_batched_matrix_kernel<DeviceBlasMatrixKernel<std::integral_constant<int, 400 + DEVICE_BLAS_POLICY>>>(
                            queue, a.batch_size(), local_size, [=](sycl::nd_item<1> item, int bid) {
                                batchlas::device::ger<kPolicy>(item.get_group(),
                                                       xvec.batch_item(bid),
                                                       yvec.batch_item(bid),
                                                       a_view.batch_item(bid),
                                                       1.0f);
                            });
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#elif DEVICE_BLAS_MATRIX_MODE == 5
    const int a_rows = kRank2kTrans == Transpose::NoTrans ? n : k;
    const int a_cols = kRank2kTrans == Transpose::NoTrans ? k : n;
    auto A = Matrix<DeviceBlasMatrixScalar>::Random(a_rows, a_cols, false, batch);
    auto B = Matrix<DeviceBlasMatrixScalar>::Random(a_rows, a_cols, false, batch);
    auto C = Matrix<DeviceBlasMatrixScalar>::Zeros(n, n, batch);
    state.SetKernel(queue,
                    std::move(A),
                    std::move(B),
                    std::move(C),
                    [](Queue& queue, auto a, auto b, auto c) {
                        const auto a_view = a.kernel_view();
                        const auto b_view = b.kernel_view();
                        const auto c_view = c.kernel_view();
                        const std::size_t local_size = device_blas_rank_update_local_size(queue);
                        if (kPolicy != device::DeviceBlasPolicy::Auto || local_size < 256) {
                            const auto launch = device_blas_nd_item_1d_launch_info(queue, local_size);
                            const auto workspace_elements = []<typename Scalar>(const device::DeviceBlasLaunchInfo& launch_info,
                                                                                 const auto& av,
                                                                                 const auto& bv,
                                                                                 const auto& cv) {
                                const int extent = cv.rows();
                                const int contract_extent = kRank2kTrans == Transpose::NoTrans ? av.cols() : av.rows();
                                if constexpr (DEVICE_BLAS_MATRIX_HERMITIAN) {
                                    return batchlas::device::her2k_workspace_elements<Scalar, kPolicy, Uplo::Lower, kRank2kTrans>(
                                        launch_info, extent, contract_extent);
                                }
                                return batchlas::device::syr2k_workspace_elements<Scalar, kPolicy, Uplo::Lower, kRank2kTrans>(
                                    launch_info, extent, contract_extent);
                            }.template operator()<DeviceBlasMatrixScalar>(launch, a_view.batch_item(0), b_view.batch_item(0), c_view.batch_item(0));
                            launch_batched_matrix_kernel_with_workspace<DeviceBlasMatrixScalar,
                                                                        DeviceBlasMatrixKernel<std::integral_constant<int, 500 + DEVICE_BLAS_POLICY>>>(
                                queue, a.batch_size(), local_size, workspace_elements, [=](sycl::nd_item<1> item, int bid, DeviceBlasMatrixScalar* workspace) {
                                    if constexpr (DEVICE_BLAS_MATRIX_HERMITIAN) {
                                        batchlas::device::her2k<kPolicy, Uplo::Lower, kRank2kTrans>(
                                            item,
                                            a_view.batch_item(bid),
                                            b_view.batch_item(bid),
                                            c_view.batch_item(bid),
                                            DeviceBlasMatrixScalar(1),
                                            DeviceBlasMatrixScalar(0),
                                            workspace);
                                    } else {
                                        batchlas::device::syr2k<kPolicy, Uplo::Lower, kRank2kTrans>(
                                            item,
                                            a_view.batch_item(bid),
                                            b_view.batch_item(bid),
                                            c_view.batch_item(bid),
                                            DeviceBlasMatrixScalar(1),
                                            DeviceBlasMatrixScalar(0),
                                            workspace);
                                    }
                                });
                        } else {
                            const auto launch = device_blas_nd_item_3d_launch_info(queue);
                            const auto workspace_elements = []<typename Scalar>(const device::DeviceBlasLaunchInfo& launch_info,
                                                                                 const auto& av,
                                                                                 const auto& bv,
                                                                                 const auto& cv) {
                                const int extent = cv.rows();
                                const int contract_extent = kRank2kTrans == Transpose::NoTrans ? av.cols() : av.rows();
                                if constexpr (DEVICE_BLAS_MATRIX_HERMITIAN) {
                                    return batchlas::device::her2k_workspace_elements<Scalar, kPolicy, Uplo::Lower, kRank2kTrans>(
                                        launch_info, extent, contract_extent);
                                }
                                return batchlas::device::syr2k_workspace_elements<Scalar, kPolicy, Uplo::Lower, kRank2kTrans>(
                                    launch_info, extent, contract_extent);
                            }.template operator()<DeviceBlasMatrixScalar>(launch, a_view.batch_item(0), b_view.batch_item(0), c_view.batch_item(0));
                            launch_batched_matrix_tile_kernel_with_workspace<DeviceBlasMatrixScalar,
                                                                             DeviceBlasMatrixTileKernel<std::integral_constant<int, 500 + DEVICE_BLAS_POLICY>>>(
                                queue, c.rows(), c.cols(), a.batch_size(), workspace_elements, [=](sycl::nd_item<3> item, int bid, DeviceBlasMatrixScalar* workspace) {
                                    if constexpr (DEVICE_BLAS_MATRIX_HERMITIAN) {
                                        batchlas::device::her2k<kPolicy, Uplo::Lower, kRank2kTrans>(
                                            item,
                                            a_view.batch_item(bid),
                                            b_view.batch_item(bid),
                                            c_view.batch_item(bid),
                                            DeviceBlasMatrixScalar(1),
                                            DeviceBlasMatrixScalar(0),
                                            workspace);
                                    } else {
                                        batchlas::device::syr2k<kPolicy, Uplo::Lower, kRank2kTrans>(
                                            item,
                                            a_view.batch_item(bid),
                                            b_view.batch_item(bid),
                                            c_view.batch_item(bid),
                                            DeviceBlasMatrixScalar(1),
                                            DeviceBlasMatrixScalar(0),
                                            workspace);
                                    }
                                });
                        }
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * kDeviceBlasRank2kFlopScale * n * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
#else
#error "Unsupported DEVICE_BLAS_MATRIX_MODE"
#endif
}

} // namespace

static auto* bench_device_blas_matrix_benchmark = minibench::RegisterBenchmark(device_blas_benchmark_name(), device_blas_matrix_benchmark)
    ->Args({128, 128, 128, 128})->Args({256, 256, 256, 64})->Args({512, 512, 512, 32});

MINI_BENCHMARK_MAIN();
