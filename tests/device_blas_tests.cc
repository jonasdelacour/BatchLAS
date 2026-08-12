#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>

#include "../src/queue.hh"

using namespace batchlas;

namespace {

template <typename T>
void fill_dense_test_matrix(Matrix<T, MatrixFormat::Dense>& a) {
    auto view = a.view();
    for (int b = 0; b < view.batch_size(); ++b) {
        for (int j = 0; j < view.cols(); ++j) {
            for (int i = 0; i < view.rows(); ++i) {
                view.template at<MatrixFormat::Dense>(i, j, b) = static_cast<T>((j + 1) * 10 + (i + 1));
            }
        }
    }
}

template <typename T>
void expect_vector_near(const VectorView<T>& actual,
                        const std::array<T, 4>& expected,
                        int count,
                        double tol = 1e-5) {
    for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(static_cast<double>(actual(i)), static_cast<double>(expected[static_cast<std::size_t>(i)]), tol)
            << "Mismatch at index " << i;
    }
}

template <typename T, std::size_t N>
void expect_matrix_near(const MatrixView<T, MatrixFormat::Dense>& actual,
                        const std::array<T, N>& expected,
                        double tol = 1e-5) {
    ASSERT_EQ(static_cast<std::size_t>(actual.rows() * actual.cols()), N);

    std::size_t index = 0;
    for (int j = 0; j < actual.cols(); ++j) {
        for (int i = 0; i < actual.rows(); ++i, ++index) {
            EXPECT_NEAR(static_cast<double>(actual.template at<MatrixFormat::Dense>(i, j)),
                        static_cast<double>(expected[index]),
                        tol)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

size_t device_test_work_group_size(const Queue& ctx) {
    const auto max_wg = ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    return std::max<size_t>(1, std::min<size_t>(32, max_wg));
}

bool device_supports_matrix_register_tiles(const Queue& ctx) {
    return ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE) >=
        static_cast<size_t>(batchlas::device::detail::subgroup::kRegisterMatrixThreadsPerGroup);
}

size_t device_test_subgroup_size(const Queue& ctx) {
    return std::max<size_t>(1, ctx.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
}

batchlas::device::DeviceBlasLaunchInfo device_test_group_launch_info(size_t local_size) {
    return batchlas::device::make_group_launch_info(static_cast<int>(local_size));
}

batchlas::device::DeviceBlasLaunchInfo device_test_nd_item_1d_launch_info(const Queue& ctx, size_t local_size) {
    return batchlas::device::make_nd_item_1d_launch_info(static_cast<int>(local_size), static_cast<int>(device_test_subgroup_size(ctx)));
}

batchlas::device::DeviceBlasLaunchInfo device_test_nd_item_3d_launch_info(const Queue& ctx) {
    return batchlas::device::make_nd_item_3d_launch_info(
        batchlas::device::detail::subgroup::kRegisterMatrixThreadsPerGroup,
        static_cast<int>(device_test_subgroup_size(ctx)));
}

template <typename KernelFunc>
void run_group_kernel(Queue& ctx, size_t local_size, KernelFunc&& kernel) {
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           kernel(item.get_group());
                       });
    });
    ctx.wait_and_throw();
}

template <typename T, typename KernelFunc>
void run_group_kernel_with_workspace(Queue& ctx,
                                     size_t local_size,
                                     size_t workspace_elements,
                                     KernelFunc&& kernel) {
    ctx->submit([&](sycl::handler& h) {
        if (workspace_elements == 0) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                           [=](sycl::nd_item<1> item) {
                               kernel(item.get_group(), static_cast<T*>(nullptr));
                           });
            return;
        }

        sycl::local_accessor<T, 1> workspace(sycl::range<1>(workspace_elements), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           kernel(item.get_group(), batchlas::util::get_raw_ptr(workspace));
                       });
    });
    ctx.wait_and_throw();
}

template <typename KernelFunc>
void run_nd_item_kernel(Queue& ctx, size_t local_size, KernelFunc&& kernel) {
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           kernel(item);
                       });
    });
    ctx.wait_and_throw();
}

template <typename T, typename KernelFunc>
void run_nd_item_kernel_with_workspace(Queue& ctx,
                                       size_t local_size,
                                       size_t workspace_elements,
                                       KernelFunc&& kernel) {
    ctx->submit([&](sycl::handler& h) {
        if (workspace_elements == 0) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                           [=](sycl::nd_item<1> item) {
                               kernel(item, static_cast<T*>(nullptr));
                           });
            return;
        }

        sycl::local_accessor<T, 1> workspace(sycl::range<1>(workspace_elements), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           kernel(item, batchlas::util::get_raw_ptr(workspace));
                       });
    });
    ctx.wait_and_throw();
}

template <typename KernelFunc>
void run_nd_item_kernel_3d(Queue& ctx, int rows, int cols, KernelFunc&& kernel) {
    constexpr size_t local_rows = static_cast<size_t>(batchlas::device::detail::subgroup::kRegisterMatrixLocalRows);
    constexpr size_t local_cols = static_cast<size_t>(batchlas::device::detail::subgroup::kRegisterMatrixLocalCols);
    constexpr size_t tile_rows = static_cast<size_t>(batchlas::device::detail::subgroup::kRegisterMatrixTileM);
    constexpr size_t tile_cols = static_cast<size_t>(batchlas::device::detail::subgroup::kRegisterMatrixTileN);
    const size_t group_rows = (static_cast<size_t>(rows) + tile_rows - 1) / tile_rows;
    const size_t group_cols = (static_cast<size_t>(cols) + tile_cols - 1) / tile_cols;

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, group_rows * local_rows, group_cols * local_cols),
                                         sycl::range<3>(1, local_rows, local_cols)),
                       [=](sycl::nd_item<3> item) {
                           kernel(item);
                       });
    });
    ctx.wait_and_throw();
}

template <typename T, typename KernelFunc>
void run_nd_item_kernel_3d_with_workspace(Queue& ctx,
                                          int rows,
                                          int cols,
                                          size_t workspace_elements,
                                          KernelFunc&& kernel) {
    constexpr size_t local_rows = static_cast<size_t>(batchlas::device::detail::subgroup::kRegisterMatrixLocalRows);
    constexpr size_t local_cols = static_cast<size_t>(batchlas::device::detail::subgroup::kRegisterMatrixLocalCols);
    constexpr size_t tile_rows = static_cast<size_t>(batchlas::device::detail::subgroup::kRegisterMatrixTileM);
    constexpr size_t tile_cols = static_cast<size_t>(batchlas::device::detail::subgroup::kRegisterMatrixTileN);
    const size_t group_rows = (static_cast<size_t>(rows) + tile_rows - 1) / tile_rows;
    const size_t group_cols = (static_cast<size_t>(cols) + tile_cols - 1) / tile_cols;

    ctx->submit([&](sycl::handler& h) {
        if (workspace_elements == 0) {
            h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, group_rows * local_rows, group_cols * local_cols),
                                             sycl::range<3>(1, local_rows, local_cols)),
                           [=](sycl::nd_item<3> item) {
                               kernel(item, static_cast<T*>(nullptr));
                           });
            return;
        }

        sycl::local_accessor<T, 1> workspace(sycl::range<1>(workspace_elements), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, group_rows * local_rows, group_cols * local_cols),
                                         sycl::range<3>(1, local_rows, local_cols)),
                       [=](sycl::nd_item<3> item) {
                           kernel(item, batchlas::util::get_raw_ptr(workspace));
                       });
    });
    ctx.wait_and_throw();
}

template <typename T>
std::vector<T> reference_trmm(const MatrixView<T, MatrixFormat::Dense>& a_view,
                              const MatrixView<T, MatrixFormat::Dense>& b_view,
                              const MatrixView<T, MatrixFormat::Dense>& c_initial,
                              T alpha,
                              T beta,
                              Side side,
                              Uplo uplo,
                              Transpose trans,
                              Diag diag);

template <typename T>
void expect_matrix_matches_vector(const MatrixView<T, MatrixFormat::Dense>& actual,
                                  const std::vector<T>& expected,
                                  double tol = 1e-4);

template <Side SideValue, Uplo UploValue, Transpose TransValue, Diag DiagValue>
void run_trmm_nd_item_case(Queue& ctx, size_t local_size) {
    Matrix<float, MatrixFormat::Dense> a(4, 4);
    Matrix<float, MatrixFormat::Dense> b(SideValue == Side::Left ? 4 : 3, SideValue == Side::Left ? 3 : 4);
    Matrix<float, MatrixFormat::Dense> c(b.rows(), b.cols());
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(1 + i + 4 * j);
        }
    }
    for (int j = 0; j < b.cols(); ++j) {
        for (int i = 0; i < b.rows(); ++i) {
            b_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(2 + i + 3 * j);
            c_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(-1 + i - j);
        }
    }

    const auto expected = reference_trmm(a.view(), b.view(), c.view(), 0.9f, -0.2f, SideValue, UploValue, TransValue, DiagValue);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
    const auto workspace_elements = batchlas::device::trmm_workspace_elements<float,
                                                                               batchlas::device::DeviceBlasPolicy::Auto,
                                                                               SideValue,
                                                                               UploValue,
                                                                               TransValue,
                                                                               DiagValue>(
        launch, c_view.rows(), c_view.cols(), false);

    run_nd_item_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](sycl::nd_item<1> item, float* workspace) {
        batchlas::device::trmm<batchlas::device::DeviceBlasPolicy::Auto, SideValue, UploValue, TransValue, DiagValue>(
            item, a_view, b_view, c_view, 0.9f, -0.2f, workspace);
    });

    expect_matrix_matches_vector(c.view(), expected);
}

template <typename T>
std::vector<T> reference_trmm(const MatrixView<T, MatrixFormat::Dense>& a_view,
                              const MatrixView<T, MatrixFormat::Dense>& b_view,
                              const MatrixView<T, MatrixFormat::Dense>& c_initial,
                              T alpha,
                              T beta,
                              Side side,
                              Uplo uplo,
                              Transpose trans,
                              Diag diag) {
    auto a_kernel = a_view.kernel_view();
    auto b_kernel = b_view.kernel_view();
    std::vector<T> out(static_cast<std::size_t>(c_initial.rows() * c_initial.cols()), T{});
    const auto transform = batchlas::device::TriangularTransform{.side = side, .uplo = uplo, .trans = trans, .diag = diag};
    const int contract_extent = side == Side::Left ? b_view.rows() : b_view.cols();

    for (int j = 0; j < c_initial.cols(); ++j) {
        for (int i = 0; i < c_initial.rows(); ++i) {
            T sum{};
            for (int k = 0; k < contract_extent; ++k) {
                if (side == Side::Left) {
                    sum += batchlas::device::detail::triangular_matrix_entry(a_kernel, i, k, transform) * b_kernel(k, j);
                } else {
                    sum += b_kernel(i, k) * batchlas::device::detail::triangular_matrix_entry(a_kernel, k, j, transform);
                }
            }
            out[static_cast<std::size_t>(j * c_initial.rows() + i)] = alpha * sum + beta * c_initial(i, j);
        }
    }

    return out;
}

template <typename T>
std::vector<T> reference_gemm(const MatrixView<T, MatrixFormat::Dense>& a_view,
                              const MatrixView<T, MatrixFormat::Dense>& b_view,
                              const MatrixView<T, MatrixFormat::Dense>& c_initial,
                              T alpha,
                              T beta,
                              Transpose trans_a,
                              Transpose trans_b) {
    auto a_kernel = a_view.kernel_view();
    auto b_kernel = b_view.kernel_view();
    std::vector<T> out(static_cast<std::size_t>(c_initial.rows() * c_initial.cols()), T{});
    const int m = trans_a == Transpose::NoTrans ? a_view.rows() : a_view.cols();
    const int k = trans_a == Transpose::NoTrans ? a_view.cols() : a_view.rows();
    const int n = trans_b == Transpose::NoTrans ? b_view.cols() : b_view.rows();

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T sum{};
            for (int kk = 0; kk < k; ++kk) {
                sum += batchlas::device::detail::matrix_entry(a_kernel, i, kk, trans_a) *
                    batchlas::device::detail::matrix_entry(b_kernel, kk, j, trans_b);
            }
            out[static_cast<std::size_t>(j * c_initial.rows() + i)] = alpha * sum + beta * c_initial(i, j);
        }
    }

    return out;
}

template <typename T>
std::vector<T> reference_symm(const MatrixView<T, MatrixFormat::Dense>& a_view,
                              const MatrixView<T, MatrixFormat::Dense>& b_view,
                              const MatrixView<T, MatrixFormat::Dense>& c_initial,
                              T alpha,
                              T beta,
                              Side side,
                              Uplo uplo) {
    auto a_kernel = a_view.kernel_view();
    auto b_kernel = b_view.kernel_view();
    std::vector<T> out(static_cast<std::size_t>(c_initial.rows() * c_initial.cols()), T{});
    const auto transform = batchlas::device::SymmetricTransform{.side = side, .uplo = uplo};
    const int contract_extent = side == Side::Left ? b_view.rows() : b_view.cols();

    for (int j = 0; j < c_initial.cols(); ++j) {
        for (int i = 0; i < c_initial.rows(); ++i) {
            T sum{};
            for (int k = 0; k < contract_extent; ++k) {
                if (side == Side::Left) {
                    sum += batchlas::device::detail::symmetric_matrix_entry(a_kernel, i, k, transform) * b_kernel(k, j);
                } else {
                    sum += b_kernel(i, k) * batchlas::device::detail::symmetric_matrix_entry(a_kernel, k, j, transform);
                }
            }
            out[static_cast<std::size_t>(j * c_initial.rows() + i)] = alpha * sum + beta * c_initial(i, j);
        }
    }

    return out;
}

template <typename T>
std::vector<T> reference_rank1_update(const VectorView<T>& x,
                                      const VectorView<T>& y,
                                      const MatrixView<T, MatrixFormat::Dense>& a_initial,
                                      T alpha,
                                      bool conjugate_x = false,
                                      bool conjugate_y = false) {
    std::vector<T> out(static_cast<std::size_t>(a_initial.rows() * a_initial.cols()), T{});

    for (int j = 0; j < a_initial.cols(); ++j) {
        for (int i = 0; i < a_initial.rows(); ++i) {
            const T lhs = batchlas::device::detail::maybe_conjugate(x(i), conjugate_x);
            const T rhs = batchlas::device::detail::maybe_conjugate(y(j), conjugate_y);
            out[static_cast<std::size_t>(j * a_initial.rows() + i)] = a_initial(i, j) + alpha * lhs * rhs;
        }
    }

    return out;
}

template <typename T>
std::vector<T> reference_rank2k(const MatrixView<T, MatrixFormat::Dense>& a_view,
                                const MatrixView<T, MatrixFormat::Dense>& b_view,
                                const MatrixView<T, MatrixFormat::Dense>& c_initial,
                                T alpha,
                                T beta,
                                Uplo uplo,
                                Transpose trans,
                                bool hermitian) {
    auto a_kernel = a_view.kernel_view();
    auto b_kernel = b_view.kernel_view();
    std::vector<T> out(static_cast<std::size_t>(c_initial.rows() * c_initial.cols()), T{});
    const int extent = batchlas::device::detail::output_size(a_kernel, trans);
    const int contract_extent = batchlas::device::detail::input_size(a_kernel, trans);
    const Transpose rhs_transform = (trans == Transpose::NoTrans)
        ? (hermitian ? Transpose::ConjTrans : Transpose::Trans)
        : Transpose::NoTrans;
    const T alpha2 = hermitian ? batchlas::device::detail::conj(alpha) : alpha;

    for (int j = 0; j < c_initial.cols(); ++j) {
        for (int i = 0; i < c_initial.rows(); ++i) {
            T value = c_initial(i, j);
            if (i < extent && j < extent && batchlas::device::detail::triangular_storage_contains(uplo, i, j)) {
                T sum{};
                for (int k = 0; k < contract_extent; ++k) {
                    const T lhs1 = batchlas::device::detail::matrix_entry(a_kernel, i, k, trans);
                    const T rhs1 = batchlas::device::detail::matrix_entry(b_kernel, k, j, rhs_transform);
                    const T lhs2 = batchlas::device::detail::matrix_entry(b_kernel, i, k, trans);
                    const T rhs2 = batchlas::device::detail::matrix_entry(a_kernel, k, j, rhs_transform);
                    sum += alpha * lhs1 * rhs1 + alpha2 * lhs2 * rhs2;
                }
                value = beta * c_initial(i, j) + sum;
                if constexpr (ComplexScalar<T>) {
                    if (hermitian && i == j) {
                        value = T(value.real(), typename T::value_type(0));
                    }
                }
            }
            out[static_cast<std::size_t>(j * c_initial.rows() + i)] = value;
        }
    }

    return out;
}

template <typename T>
std::vector<T> reference_rankk(const MatrixView<T, MatrixFormat::Dense>& a_view,
                               const MatrixView<T, MatrixFormat::Dense>& c_initial,
                               T alpha,
                               T beta,
                               Uplo uplo,
                               Transpose trans,
                               bool hermitian) {
    auto a_kernel = a_view.kernel_view();
    std::vector<T> out(static_cast<std::size_t>(c_initial.rows() * c_initial.cols()), T{});
    const int extent = batchlas::device::detail::output_size(a_kernel, trans);
    const int contract_extent = batchlas::device::detail::input_size(a_kernel, trans);
    const Transpose rhs_transform = (trans == Transpose::NoTrans)
        ? (hermitian ? Transpose::ConjTrans : Transpose::Trans)
        : Transpose::NoTrans;

    for (int j = 0; j < c_initial.cols(); ++j) {
        for (int i = 0; i < c_initial.rows(); ++i) {
            T value = c_initial(i, j);
            if (i < extent && j < extent && batchlas::device::detail::triangular_storage_contains(uplo, i, j)) {
                T sum{};
                for (int k = 0; k < contract_extent; ++k) {
                    const T lhs = batchlas::device::detail::matrix_entry(a_kernel, i, k, trans);
                    const T rhs = batchlas::device::detail::matrix_entry(a_kernel, k, j, rhs_transform);
                    sum += alpha * lhs * rhs;
                }
                value = beta * c_initial(i, j) + sum;
                if constexpr (ComplexScalar<T>) {
                    if (hermitian && i == j) {
                        value = T(value.real(), typename T::value_type(0));
                    }
                }
            }
            out[static_cast<std::size_t>(j * c_initial.rows() + i)] = value;
        }
    }

    return out;
}

template <typename T>
void expect_matrix_matches_vector(const MatrixView<T, MatrixFormat::Dense>& actual,
                                  const std::vector<T>& expected,
                                  double tol) {
    ASSERT_EQ(static_cast<std::size_t>(actual.rows() * actual.cols()), expected.size());
    std::size_t index = 0;
    for (int j = 0; j < actual.cols(); ++j) {
        for (int i = 0; i < actual.rows(); ++i, ++index) {
            EXPECT_LE(std::abs(actual.template at<MatrixFormat::Dense>(i, j) - expected[index]),
                      tol)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

template <typename T>
void expect_vector_matches_vector(const VectorView<T>& actual,
                                  const std::vector<T>& expected,
                                  double tol = 1e-5) {
    ASSERT_EQ(static_cast<std::size_t>(actual.size()), expected.size());
    for (int i = 0; i < actual.size(); ++i) {
        EXPECT_LE(std::abs(actual(i) - expected[static_cast<std::size_t>(i)]), tol)
            << "Mismatch at index " << i;
    }
}

} // namespace

TEST(DeviceBlasTest, GemvNoTransposeMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 4, 1, 5, 32);
    Vector<float> x(4);
    Vector<float> y(3);

    fill_dense_test_matrix(a);
    x(0) = 1.0f;
    x(1) = -2.0f;
    x(2) = 0.5f;
    x(3) = 3.0f;
    y(0) = 2.0f;
    y(1) = -1.0f;
    y(2) = 4.0f;

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_view = VectorView<float>(y);
    const float alpha = 1.5f;
    const float beta = -0.25f;
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::gemv_workspace_elements<float, Transpose::NoTrans>(launch, a_view.rows(), a_view.cols());

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto group, float* workspace) {
        batchlas::device::gemv<Transpose::NoTrans>(group, a_view, x_view, y_view, alpha, beta, workspace);
    });

    const std::array<float, 4> expected{
        alpha * (11.0f * 1.0f + 21.0f * -2.0f + 31.0f * 0.5f + 41.0f * 3.0f) + beta * 2.0f,
        alpha * (12.0f * 1.0f + 22.0f * -2.0f + 32.0f * 0.5f + 42.0f * 3.0f) + beta * -1.0f,
        alpha * (13.0f * 1.0f + 23.0f * -2.0f + 33.0f * 0.5f + 43.0f * 3.0f) + beta * 4.0f,
        0.0f,
    };

    expect_vector_near(VectorView<float>(y), expected, 3);
}

TEST(DeviceBlasTest, GemvTransposeMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 4);
    Vector<float> x(3);
    Vector<float> y(4);

    fill_dense_test_matrix(a);
    x(0) = -1.0f;
    x(1) = 2.0f;
    x(2) = 0.25f;
    y(0) = 0.5f;
    y(1) = -1.5f;
    y(2) = 2.5f;
    y(3) = -3.5f;

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_view = VectorView<float>(y);
    const float alpha = -2.0f;
    const float beta = 0.75f;
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::gemv_workspace_elements<float, Transpose::Trans>(launch, a_view.rows(), a_view.cols());

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto group, float* workspace) {
        batchlas::device::gemv<Transpose::Trans>(group, a_view, x_view, y_view, alpha, beta, workspace);
    });

    const std::array<float, 4> expected{
        alpha * (11.0f * -1.0f + 12.0f * 2.0f + 13.0f * 0.25f) + beta * 0.5f,
        alpha * (21.0f * -1.0f + 22.0f * 2.0f + 23.0f * 0.25f) + beta * -1.5f,
        alpha * (31.0f * -1.0f + 32.0f * 2.0f + 33.0f * 0.25f) + beta * 2.5f,
        alpha * (41.0f * -1.0f + 42.0f * 2.0f + 43.0f * 0.25f) + beta * -3.5f,
    };

    expect_vector_near(VectorView<float>(y), expected, 4);
}

TEST(DeviceBlasTest, GemxvComputesMultipleVectorsInOnePass) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a_parent(3, 3, 2);
    Vector<float> x0_parent(3, 2);
    Vector<float> x1_parent(3, 2);
    Vector<float> y0_parent(3, 2);
    Vector<float> y1_parent(3, 2);

    fill_dense_test_matrix(a_parent);
    for (int j = 0; j < a_parent.cols(); ++j) {
        for (int i = 0; i < a_parent.rows(); ++i) {
            a_parent(i, j, 1) = static_cast<float>(100 + (j + 1) * 10 + (i + 1));
        }
    }

    x0_parent(0, 1) = 1.0f;
    x0_parent(1, 1) = 0.0f;
    x0_parent(2, 1) = -1.0f;
    x1_parent(0, 1) = 2.0f;
    x1_parent(1, 1) = -1.0f;
    x1_parent(2, 1) = 0.5f;
    y0_parent(0, 1) = 4.0f;
    y0_parent(1, 1) = -2.0f;
    y0_parent(2, 1) = 1.0f;
    y1_parent(0, 1) = -3.0f;
    y1_parent(1, 1) = 5.0f;
    y1_parent(2, 1) = 7.0f;

    auto a_view = a_parent.view().kernel_view().batch_item(1);
    auto x0 = VectorView<float>(x0_parent).batch_item(1);
    auto x1 = VectorView<float>(x1_parent).batch_item(1);
    auto y0 = VectorView<float>(y0_parent).batch_item(1);
    auto y1 = VectorView<float>(y1_parent).batch_item(1);
    const float alpha0 = 2.0f;
    const float beta0 = -1.0f;
    const float alpha1 = -0.5f;
    const float beta1 = 0.25f;
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::gemv_workspace_elements<float, Transpose::NoTrans>(launch, a_view.rows(), a_view.cols());

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto g, float* workspace) {
        batchlas::device::gemv<Transpose::NoTrans>(g, a_view, x0, y0, alpha0, beta0, workspace);
        batchlas::device::gemv<Transpose::NoTrans>(g, a_view, x1, y1, alpha1, beta1, workspace);
    });

    const std::array<float, 4> expected_y0{
        alpha0 * (111.0f * 1.0f + 121.0f * 0.0f + 131.0f * -1.0f) + beta0 * 4.0f,
        alpha0 * (112.0f * 1.0f + 122.0f * 0.0f + 132.0f * -1.0f) + beta0 * -2.0f,
        alpha0 * (113.0f * 1.0f + 123.0f * 0.0f + 133.0f * -1.0f) + beta0 * 1.0f,
        0.0f,
    };
    const std::array<float, 4> expected_y1{
        alpha1 * (111.0f * 2.0f + 121.0f * -1.0f + 131.0f * 0.5f) + beta1 * -3.0f,
        alpha1 * (112.0f * 2.0f + 122.0f * -1.0f + 132.0f * 0.5f) + beta1 * 5.0f,
        alpha1 * (113.0f * 2.0f + 123.0f * -1.0f + 133.0f * 0.5f) + beta1 * 7.0f,
        0.0f,
    };

    expect_vector_near(y0_parent.batch_item(1), expected_y0, 3);
    expect_vector_near(y1_parent.batch_item(1), expected_y1, 3);
}

TEST(DeviceBlasTest, TrmvLowerNoTransposeMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Vector<float> x(3);
    Vector<float> y(3);
    auto a_host = a.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -1.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 4.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = 9.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = 3.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 5.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = 8.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 7.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = -2.0f;

    x(0) = 1.5f;
    x(1) = -2.0f;
    x(2) = 0.5f;
    y(0) = 4.0f;
    y(1) = -1.0f;
    y(2) = 2.0f;

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_view = VectorView<float>(y);
    const float alpha = 0.75f;
    const float beta = -0.5f;
    const size_t local_size = device_test_work_group_size(ctx);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           batchlas::device::trmv<Uplo::Lower, Transpose::NoTrans, Diag::NonUnit>(
                               item.get_group(), a_view, x_view, y_view, alpha, beta);
                       });
    });
    ctx.wait_and_throw();

    const std::array<float, 4> expected{
        alpha * (2.0f * 1.5f) + beta * 4.0f,
        alpha * (-1.0f * 1.5f + 3.0f * -2.0f) + beta * -1.0f,
        alpha * (4.0f * 1.5f + 5.0f * -2.0f + -2.0f * 0.5f) + beta * 2.0f,
        0.0f,
    };

    expect_vector_near(VectorView<float>(y), expected, 3);
}

TEST(DeviceBlasTest, TrmvUpperTransposeUnitDiagonalMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Vector<float> x(3);
    Vector<float> y(3);
    auto a_host = a.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 4.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -8.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 6.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = -3.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 9.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = -1.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 5.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = 7.0f;

    x(0) = 2.0f;
    x(1) = -1.0f;
    x(2) = 3.0f;
    y(0) = 1.0f;
    y(1) = -4.0f;
    y(2) = 2.0f;

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_view = VectorView<float>(y);
    const float alpha = -1.25f;
    const float beta = 0.5f;
    const size_t local_size = device_test_work_group_size(ctx);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           batchlas::device::trmv<Uplo::Upper, Transpose::Trans, Diag::Unit>(
                               item.get_group(), a_view, x_view, y_view, alpha, beta);
                       });
    });
    ctx.wait_and_throw();

    const std::array<float, 4> expected{
        alpha * (1.0f * 2.0f) + beta * 1.0f,
        alpha * (2.0f * 2.0f + 1.0f * -1.0f) + beta * -4.0f,
        alpha * (-1.0f * 2.0f + 5.0f * -1.0f + 1.0f * 3.0f) + beta * 2.0f,
        0.0f,
    };

    expect_vector_near(VectorView<float>(y), expected, 3);
}

TEST(DeviceBlasTest, FillVectorAndMatrixSetConstantValues) {
    Queue ctx(Device::default_device());

    Vector<float> x(5);
    Matrix<float, MatrixFormat::Dense> a(3, 2);
    auto x_view = VectorView<float>(x);
    auto a_view = a.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](const auto& group) {
        batchlas::device::fill(group, x_view, -2.5f);
    });
    run_group_kernel(ctx, local_size, [=](const auto& group) {
        batchlas::device::fill(group, a_view, 4.0f);
    });

    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(x(i), -2.5f) << "Mismatch at vector index " << i;
    }

    auto a_host = a.view();
    for (int col = 0; col < a_host.cols(); ++col) {
        for (int row = 0; row < a_host.rows(); ++row) {
            EXPECT_FLOAT_EQ(a_host.at<MatrixFormat::Dense>(row, col), 4.0f)
                << "Mismatch at (" << row << ", " << col << ")";
        }
    }
}

TEST(DeviceBlasTest, TrmvUpperNoTransSupportsInPlaceOutput) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Vector<float> x(3);
    auto a_host = a.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 4.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -8.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 6.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = -3.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 9.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = -1.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 5.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = 7.0f;

    x(0) = 2.0f;
    x(1) = -1.0f;
    x(2) = 3.0f;

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](const auto& group) {
        batchlas::device::trmv<Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>(
            group, a_view, x_view, x_view, 1.0f, 0.5f);
    });

    const std::array<float, 4> expected{4.0f, 17.5f, 22.5f, 0.0f};
    expect_vector_near(VectorView<float>(x), expected, 3);
}

TEST(DeviceBlasTest, SymvLowerMatchesReferenceUsingOnlyStoredTriangle) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Vector<float> x(3);
    Vector<float> y(3);
    auto a_host = a.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -1.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 4.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = 99.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = 3.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 5.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = -77.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 88.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = -2.0f;

    x(0) = 1.5f;
    x(1) = -2.0f;
    x(2) = 0.5f;
    y(0) = 4.0f;
    y(1) = -1.0f;
    y(2) = 2.0f;

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_view = VectorView<float>(y);
    const float alpha = 0.75f;
    const float beta = -0.5f;
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::symv_workspace_elements<float, Uplo::Lower>(launch, a_view.rows());

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto group, float* workspace) {
        batchlas::device::symv<Uplo::Lower>(group, a_view, x_view, y_view, alpha, beta, workspace);
    });

    const std::array<float, 4> expected{
        alpha * (2.0f * 1.5f + -1.0f * -2.0f + 4.0f * 0.5f) + beta * 4.0f,
        alpha * (-1.0f * 1.5f + 3.0f * -2.0f + 5.0f * 0.5f) + beta * -1.0f,
        alpha * (4.0f * 1.5f + 5.0f * -2.0f + -2.0f * 0.5f) + beta * 2.0f,
        0.0f,
    };

    expect_vector_near(VectorView<float>(y), expected, 3);
}

TEST(DeviceBlasTest, SymvUpperMatchesReferenceUsingOnlyStoredTriangle) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Vector<float> x(3);
    Vector<float> y(3);
    auto a_host = a.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -42.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 71.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = -1.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = 3.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 64.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = 4.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 5.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = -2.0f;

    x(0) = 1.5f;
    x(1) = -2.0f;
    x(2) = 0.5f;
    y(0) = 1.0f;
    y(1) = -3.0f;
    y(2) = 2.5f;

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_view = VectorView<float>(y);
    const float alpha = -1.25f;
    const float beta = 0.25f;
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::symv_workspace_elements<float, Uplo::Upper>(launch, a_view.rows());

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto group, float* workspace) {
        batchlas::device::symv<Uplo::Upper>(group, a_view, x_view, y_view, alpha, beta, workspace);
    });

    const std::array<float, 4> expected{
        alpha * (2.0f * 1.5f + -1.0f * -2.0f + 4.0f * 0.5f) + beta * 1.0f,
        alpha * (-1.0f * 1.5f + 3.0f * -2.0f + 5.0f * 0.5f) + beta * -3.0f,
        alpha * (4.0f * 1.5f + 5.0f * -2.0f + -2.0f * 0.5f) + beta * 2.5f,
        0.0f,
    };

    expect_vector_near(VectorView<float>(y), expected, 3);
}

TEST(DeviceBlasTest, HemvLowerMatchesReferenceUsingStoredTriangle) {
    Queue ctx(Device::default_device());

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(3, 3);
    Vector<Complex> x(3);
    Vector<Complex> y(3);
    auto a_host = a.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = Complex(3.0f, 0.0f);
    a_host.at<MatrixFormat::Dense>(1, 0) = Complex(1.0f, -2.0f);
    a_host.at<MatrixFormat::Dense>(2, 0) = Complex(-0.5f, 1.5f);
    a_host.at<MatrixFormat::Dense>(1, 1) = Complex(4.0f, 0.0f);
    a_host.at<MatrixFormat::Dense>(2, 1) = Complex(2.0f, 0.25f);
    a_host.at<MatrixFormat::Dense>(2, 2) = Complex(5.0f, 0.0f);

    x(0) = Complex(1.0f, -1.0f);
    x(1) = Complex(0.5f, 2.0f);
    x(2) = Complex(-1.5f, 0.25f);

    y(0) = Complex(0.25f, 0.5f);
    y(1) = Complex(-2.0f, 0.0f);
    y(2) = Complex(1.0f, -0.75f);

    std::array<Complex, 3> expected{};
    const Complex alpha(0.75f, -0.25f);
    const Complex beta(-0.5f, 0.0f);
    for (int row = 0; row < 3; ++row) {
        Complex acc{};
        for (int col = 0; col < 3; ++col) {
            const Complex a_rc = row >= col
                ? a_host.at<MatrixFormat::Dense>(row, col)
                : std::conj(a_host.at<MatrixFormat::Dense>(col, row));
            acc += a_rc * x(col);
        }
        expected[static_cast<std::size_t>(row)] = alpha * acc + beta * y(row);
    }

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<Complex>(x);
    auto y_view = VectorView<Complex>(y);
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::hemv_workspace_elements<Complex, Uplo::Lower>(launch, a_view.rows());

    run_group_kernel_with_workspace<Complex>(ctx, local_size, workspace_elements, [=](auto group, Complex* workspace) {
        batchlas::device::hemv<Uplo::Lower>(group, a_view, x_view, y_view, alpha, beta, workspace);
    });

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(std::abs(y(i) - expected[static_cast<std::size_t>(i)]), 0.0f, 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST(DeviceBlasTest, HemvUpperMatchesReferenceUsingStoredTriangle) {
    Queue ctx(Device::default_device());

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(3, 3);
    Vector<Complex> x(3);
    Vector<Complex> y(3);
    auto a_host = a.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = Complex(3.0f, 0.0f);
    a_host.at<MatrixFormat::Dense>(0, 1) = Complex(1.0f, 2.0f);
    a_host.at<MatrixFormat::Dense>(0, 2) = Complex(-0.5f, -1.5f);
    a_host.at<MatrixFormat::Dense>(1, 1) = Complex(4.0f, 0.0f);
    a_host.at<MatrixFormat::Dense>(1, 2) = Complex(2.0f, -0.25f);
    a_host.at<MatrixFormat::Dense>(2, 2) = Complex(5.0f, 0.0f);

    x(0) = Complex(1.0f, -1.0f);
    x(1) = Complex(0.5f, 2.0f);
    x(2) = Complex(-1.5f, 0.25f);

    y(0) = Complex(-0.75f, 0.5f);
    y(1) = Complex(-2.0f, 1.0f);
    y(2) = Complex(1.0f, -0.75f);

    std::array<Complex, 3> expected{};
    const Complex alpha(-0.5f, 0.75f);
    const Complex beta(0.25f, -0.5f);
    for (int row = 0; row < 3; ++row) {
        Complex acc{};
        for (int col = 0; col < 3; ++col) {
            const Complex a_rc = row <= col
                ? a_host.at<MatrixFormat::Dense>(row, col)
                : std::conj(a_host.at<MatrixFormat::Dense>(col, row));
            acc += a_rc * x(col);
        }
        expected[static_cast<std::size_t>(row)] = alpha * acc + beta * y(row);
    }

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<Complex>(x);
    auto y_view = VectorView<Complex>(y);
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::hemv_workspace_elements<Complex, Uplo::Upper>(launch, a_view.rows());

    run_group_kernel_with_workspace<Complex>(ctx, local_size, workspace_elements, [=](auto group, Complex* workspace) {
        batchlas::device::hemv<Uplo::Upper>(group, a_view, x_view, y_view, alpha, beta, workspace);
    });

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(std::abs(y(i) - expected[static_cast<std::size_t>(i)]), 0.0f, 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST(DeviceBlasTest, HemvLowerNdItemMatchesReferenceUsingStoredTriangle) {
    Queue ctx(Device::default_device());

    using Complex = std::complex<double>;
    Matrix<Complex, MatrixFormat::Dense> a(3, 3);
    Vector<Complex> x(3);
    Vector<Complex> y(3);
    auto a_host = a.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = Complex(3.0, 0.0);
    a_host.at<MatrixFormat::Dense>(1, 0) = Complex(1.0, -2.0);
    a_host.at<MatrixFormat::Dense>(2, 0) = Complex(-0.5, 1.5);
    a_host.at<MatrixFormat::Dense>(1, 1) = Complex(4.0, 0.0);
    a_host.at<MatrixFormat::Dense>(2, 1) = Complex(2.0, 0.25);
    a_host.at<MatrixFormat::Dense>(2, 2) = Complex(5.0, 0.0);

    x(0) = Complex(1.0, -1.0);
    x(1) = Complex(0.5, 2.0);
    x(2) = Complex(-1.5, 0.25);

    y(0) = Complex(0.25, 0.5);
    y(1) = Complex(-2.0, 0.0);
    y(2) = Complex(1.0, -0.75);

    std::array<Complex, 3> expected{};
    const Complex alpha(0.75, -0.25);
    const Complex beta(-0.5, 0.0);
    for (int row = 0; row < 3; ++row) {
        Complex acc{};
        for (int col = 0; col < 3; ++col) {
            const Complex a_rc = row >= col
                ? a_host.at<MatrixFormat::Dense>(row, col)
                : std::conj(a_host.at<MatrixFormat::Dense>(col, row));
            acc += a_rc * x(col);
        }
        expected[static_cast<std::size_t>(row)] = alpha * acc + beta * y(row);
    }

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<Complex>(x);
    auto y_view = VectorView<Complex>(y);
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
    const auto workspace_elements = batchlas::device::hemv_workspace_elements<Complex, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower>(
        launch, a_view.rows());

    run_nd_item_kernel_with_workspace<Complex>(ctx, local_size, workspace_elements, [=](sycl::nd_item<1> item, Complex* workspace) {
        batchlas::device::hemv<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower>(item.get_group(),
                                                    a_view,
                                                    x_view,
                                                    y_view,
                                                    alpha,
                                                    beta,
                                                    workspace);
    });

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(std::abs(y(i) - expected[static_cast<std::size_t>(i)]), 0.0, 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST(DeviceBlasTest, SymvLowerAutoMatchesGenericOnTiledCase) {
    Queue ctx(Device::default_device());

    constexpr int n = 33;
    Matrix<float, MatrixFormat::Dense> a(n, n);
    Vector<float> x(n);
    Vector<float> y_auto(n);
    Vector<float> y_generic(n);
    auto a_host = a.view();

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            a_host.at<MatrixFormat::Dense>(row, col) = 0.0f;
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int row = col; row < n; ++row) {
            a_host.at<MatrixFormat::Dense>(row, col) = 0.05f * static_cast<float>(1 + row - col) + 0.01f * static_cast<float>(col + 1);
        }
    }

    for (int i = 0; i < n; ++i) {
        x(i) = 0.25f * static_cast<float>(i + 1) - 1.5f;
        y_auto(i) = 1.0f - 0.1f * static_cast<float>(i);
        y_generic(i) = y_auto(i);
    }

    const float alpha = 0.8f;
    const float beta = -0.35f;
    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_auto_view = VectorView<float>(y_auto);
    auto y_generic_view = VectorView<float>(y_generic);
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto auto_workspace = batchlas::device::symv_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower>(
        launch, a_view.rows());
    const auto generic_workspace = batchlas::device::symv_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Generic, Uplo::Lower>(
        launch, a_view.rows());

    run_group_kernel_with_workspace<float>(ctx, local_size, auto_workspace, [=](auto group, float* workspace) {
        batchlas::device::symv<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower>(group,
                                                                                        a_view,
                                                                                        x_view,
                                                                                        y_auto_view,
                                                                                        alpha,
                                                                                        beta,
                                                                                        workspace);
    });
    run_group_kernel_with_workspace<float>(ctx, local_size, generic_workspace, [=](auto group, float* workspace) {
        batchlas::device::symv<batchlas::device::DeviceBlasPolicy::Generic, Uplo::Lower>(group,
                                                                                           a_view,
                                                                                           x_view,
                                                                                           y_generic_view,
                                                                                           alpha,
                                                                                           beta,
                                                                                           workspace);
    });

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(y_auto(i), y_generic(i), 1e-4f)
            << "Mismatch at index " << i;
    }
}

TEST(DeviceBlasTest, VectorCopyScalAndAxpyGroupMatchReference) {
    Queue ctx(Device::default_device());

    Vector<float> x(5);
    Vector<float> y(5);
    Vector<float> z(5);

    x(0) = 1.0f;
    x(1) = -2.0f;
    x(2) = 0.5f;
    x(3) = 3.0f;
    x(4) = -1.5f;

    y(0) = -1.0f;
    y(1) = 0.25f;
    y(2) = 2.0f;
    y(3) = -0.75f;
    y(4) = 4.0f;

    for (int i = 0; i < z.size(); ++i) {
        z(i) = 0.0f;
    }

    const auto x_view = VectorView<float>(x);
    const auto y_view = VectorView<float>(y);
    const auto z_view = VectorView<float>(z);
    const std::vector<float> expected_y = {
        -1.0f + 1.5f * 1.0f,
        0.25f + 1.5f * -2.0f,
        2.0f + 1.5f * 0.5f,
        -0.75f + 1.5f * 3.0f,
        4.0f + 1.5f * -1.5f,
    };
    const std::vector<float> expected_z = {2.0f, -4.0f, 1.0f, 6.0f, -3.0f};
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](auto group) {
        batchlas::device::copy(group, x_view, z_view);
        sycl::group_barrier(group);
        batchlas::device::scal(group, z_view, 2.0f);
        sycl::group_barrier(group);
        batchlas::device::axpy(group, x_view, y_view, 1.5f);
    });

    expect_vector_matches_vector(VectorView<float>(y), expected_y);
    expect_vector_matches_vector(VectorView<float>(z), expected_z);
}

TEST(DeviceBlasTest, VectorCopycAndHadamardGroupMatchReference) {
    Queue ctx(Device::default_device());

    using Complex = std::complex<float>;
    Vector<Complex> x(4);
    Vector<Complex> y(4);
    Vector<Complex> xc(4);
    Vector<Complex> z(4);

    x(0) = Complex(1.0f, 2.0f);
    x(1) = Complex(-0.5f, 0.25f);
    x(2) = Complex(2.0f, -1.0f);
    x(3) = Complex(0.75f, 0.5f);

    y(0) = Complex(-1.0f, 0.5f);
    y(1) = Complex(0.25f, -0.75f);
    y(2) = Complex(1.5f, 1.0f);
    y(3) = Complex(-0.5f, 2.0f);

    for (int i = 0; i < xc.size(); ++i) {
        xc(i) = Complex(0.0f, 0.0f);
        z(i) = Complex(0.0f, 0.0f);
    }

    const auto x_view = VectorView<Complex>(x);
    const auto y_view = VectorView<Complex>(y);
    const auto xc_view = VectorView<Complex>(xc);
    const auto z_view = VectorView<Complex>(z);
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](auto group) {
        batchlas::device::copyc(group, x_view, xc_view);
        sycl::group_barrier(group);
        batchlas::device::hadamard(group, xc_view, y_view, z_view);
    });

    for (int i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(std::abs(xc(i) - std::conj(x(i))), 0.0f, 1e-5f)
            << "Mismatch in copyc at index " << i;
        EXPECT_NEAR(std::abs(z(i) - std::conj(x(i)) * y(i)), 0.0f, 1e-5f)
            << "Mismatch in hadamard at index " << i;
    }
}

TEST(DeviceBlasTest, VariadicHadamardGroupFusesPointwiseExpression) {
    Queue ctx(Device::default_device());

    Vector<float> x(5);
    Vector<float> y(5);
    Vector<float> w(5);
    Vector<float> out(5);

    for (int i = 0; i < x.size(); ++i) {
        x(i) = 1.0f + static_cast<float>(i);
        y(i) = -0.5f + 0.25f * static_cast<float>(i);
        w(i) = 2.0f - 0.1f * static_cast<float>(i);
        out(i) = 0.0f;
    }

    const auto x_view = VectorView<float>(x);
    const auto y_view = VectorView<float>(y);
    const auto w_view = VectorView<float>(w);
    const auto out_view = VectorView<float>(out);
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](auto group) {
        batchlas::device::hadamard(group,
                                   out_view,
                                   [](float xv, float yv, float wv) {
                                       return (xv + yv) * wv - xv * yv;
                                   },
                                   x_view,
                                   y_view,
                                   w_view);
    });

    for (int i = 0; i < x.size(); ++i) {
        const float expected = (x(i) + y(i)) * w(i) - x(i) * y(i);
        EXPECT_NEAR(out(i), expected, 1e-5f)
            << "Mismatch in variadic hadamard at index " << i;
    }
}

TEST(DeviceBlasTest, DotcGroupMatchesReference) {
    Queue ctx(Device::default_device());

    using Complex = std::complex<float>;
    Vector<Complex> x(4);
    Vector<Complex> y(4);
    Vector<Complex> out(1);

    x(0) = Complex(1.0f, 2.0f);
    x(1) = Complex(-0.5f, 0.25f);
    x(2) = Complex(2.0f, -1.0f);
    x(3) = Complex(0.75f, 0.5f);

    y(0) = Complex(-1.0f, 0.5f);
    y(1) = Complex(0.25f, -0.75f);
    y(2) = Complex(1.5f, 1.0f);
    y(3) = Complex(-0.5f, 2.0f);

    out(0) = Complex(0.0f, 0.0f);

    Complex expected = Complex(0.0f, 0.0f);
    for (int i = 0; i < x.size(); ++i) {
        expected += std::conj(x(i)) * y(i);
    }

    const auto x_view = VectorView<Complex>(x);
    const auto y_view = VectorView<Complex>(y);
    const auto out_view = VectorView<Complex>(out);
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](auto group) {
        const Complex dot = batchlas::device::dotc(group, x_view, y_view);
        if (group.get_local_linear_id() == 0) {
            out_view(0) = dot;
        }
    });

    EXPECT_NEAR(std::abs(out(0) - expected), 0.0f, 1e-5f);
}

TEST(DeviceBlasTest, GerGroupMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 4);
    Vector<float> x(3);
    Vector<float> y(4);
    auto a_host = a.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(1 + i + 2 * j);
        }
    }
    x(0) = 1.0f;
    x(1) = -2.0f;
    x(2) = 0.5f;
    y(0) = 2.0f;
    y(1) = -1.0f;
    y(2) = 3.0f;
    y(3) = 0.25f;

    const auto x_view = VectorView<float>(x);
    const auto y_view = VectorView<float>(y);
    const auto expected = reference_rank1_update(x_view, y_view, a.view(), 1.25f);
    auto a_view = a.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](auto group) {
        batchlas::device::ger(group, x_view, y_view, a_view, 1.25f);
    });

    expect_matrix_matches_vector(a.view(), expected);
}

TEST(DeviceBlasTest, GercGroupMatchesReference) {
    Queue ctx(Device::default_device());

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(2, 3);
    Vector<Complex> x(2);
    Vector<Complex> y(3);
    auto a_host = a.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = Complex(1.0f, -0.5f);
    a_host.at<MatrixFormat::Dense>(1, 0) = Complex(0.25f, 0.75f);
    a_host.at<MatrixFormat::Dense>(0, 1) = Complex(-2.0f, 0.5f);
    a_host.at<MatrixFormat::Dense>(1, 1) = Complex(1.5f, -1.0f);
    a_host.at<MatrixFormat::Dense>(0, 2) = Complex(0.0f, 1.25f);
    a_host.at<MatrixFormat::Dense>(1, 2) = Complex(-0.5f, -0.25f);

    x(0) = Complex(1.0f, 2.0f);
    x(1) = Complex(-0.5f, 0.25f);
    y(0) = Complex(0.5f, -1.0f);
    y(1) = Complex(-2.0f, 0.5f);
    y(2) = Complex(1.5f, 0.75f);

    const auto x_view = VectorView<Complex>(x);
    const auto y_view = VectorView<Complex>(y);
    const auto expected = reference_rank1_update(x_view, y_view, a.view(), Complex(0.75f, -0.5f), false, true);
    auto a_view = a.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](auto group) {
        batchlas::device::gerc(group, x_view, y_view, a_view, Complex(0.75f, -0.5f));
    });

    expect_matrix_matches_vector(a.view(), expected);
}

TEST(DeviceBlasTest, GerNdItemAutoMatchesGeneric) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a_auto(16, 12);
    Matrix<float, MatrixFormat::Dense> a_generic(16, 12);
    Vector<float> x(16);
    Vector<float> y(12);
    auto a_auto_host = a_auto.view();
    auto a_generic_host = a_generic.view();

    for (int j = 0; j < a_auto.cols(); ++j) {
        for (int i = 0; i < a_auto.rows(); ++i) {
            const float value = static_cast<float>(((3 * i + j) % 9) - 4);
            a_auto_host.at<MatrixFormat::Dense>(i, j) = value;
            a_generic_host.at<MatrixFormat::Dense>(i, j) = value;
        }
    }
    for (int i = 0; i < x.size(); ++i) {
        x(i) = static_cast<float>(i + 1) * 0.25f;
    }
    for (int i = 0; i < y.size(); ++i) {
        y(i) = static_cast<float>(2 - i) * 0.5f;
    }

    const auto x_view = VectorView<float>(x);
    const auto y_view = VectorView<float>(y);
    const auto expected = reference_rank1_update(x_view, y_view, a_auto.view(), 0.75f);
    auto a_auto_view = a_auto.view().kernel_view();
    auto a_generic_view = a_generic.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);

    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::ger<batchlas::device::DeviceBlasPolicy::Auto>(item.get_group(), x_view, y_view, a_auto_view, 0.75f);
    });
    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::ger<batchlas::device::DeviceBlasPolicy::Generic>(item.get_group(), x_view, y_view, a_generic_view, 0.75f);
    });

    expect_matrix_matches_vector(a_auto.view(), expected, 1e-4);
    for (int j = 0; j < a_auto.cols(); ++j) {
        for (int i = 0; i < a_auto.rows(); ++i) {
            EXPECT_NEAR(a_auto.view().template at<MatrixFormat::Dense>(i, j),
                        a_generic.view().template at<MatrixFormat::Dense>(i, j),
                        1e-5f)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(DeviceBlasTest, Syr2kGroupMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 2);
    Matrix<float, MatrixFormat::Dense> b(3, 2);
    Matrix<float, MatrixFormat::Dense> c(3, 3);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(1 + i + 3 * j);
            b_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(-2 + 2 * i - j);
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>((i >= j) ? (5 + i + j) : (-10 - i - j));
        }
    }

    const auto expected = reference_rank2k(a.view(), b.view(), c.view(), -0.75f, 0.5f, Uplo::Lower, Transpose::NoTrans, false);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::syr2k_workspace_elements<float, Uplo::Lower, Transpose::NoTrans>(
        launch, c_view.rows(), a_view.cols());

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto group, float* workspace) {
        batchlas::device::syr2k<Uplo::Lower, Transpose::NoTrans>(group, a_view, b_view, c_view, -0.75f, 0.5f, workspace);
    });

    expect_matrix_matches_vector(c.view(), expected);
}

TEST(DeviceBlasTest, SyrkGroupMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 2);
    Matrix<float, MatrixFormat::Dense> c(3, 3);
    auto a_host = a.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(1 + i + 3 * j);
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>((i >= j) ? (5 + i + j) : (-10 - i - j));
        }
    }

    const auto expected = reference_rankk(a.view(), c.view(), -0.75f, 0.5f, Uplo::Lower, Transpose::NoTrans, false);
    auto a_view = a.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::syrk_workspace_elements<float, Uplo::Lower, Transpose::NoTrans>(launch, c_view.rows(), a_view.cols());

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto group, float* workspace) {
        batchlas::device::syrk<Uplo::Lower, Transpose::NoTrans>(group, a_view, c_view, -0.75f, 0.5f, workspace);
    });

    expect_matrix_matches_vector(c.view(), expected);
}

TEST(DeviceBlasTest, Her2kGroupMatchesReference) {
    Queue ctx(Device::default_device());

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(3, 2);
    Matrix<Complex, MatrixFormat::Dense> b(3, 2);
    Matrix<Complex, MatrixFormat::Dense> c(3, 3);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = Complex(1.0f, 0.5f);
    a_host.at<MatrixFormat::Dense>(1, 0) = Complex(-0.5f, 1.5f);
    a_host.at<MatrixFormat::Dense>(2, 0) = Complex(2.0f, -1.0f);
    a_host.at<MatrixFormat::Dense>(0, 1) = Complex(-1.0f, 0.25f);
    a_host.at<MatrixFormat::Dense>(1, 1) = Complex(0.75f, -0.5f);
    a_host.at<MatrixFormat::Dense>(2, 1) = Complex(1.25f, 0.5f);

    b_host.at<MatrixFormat::Dense>(0, 0) = Complex(0.5f, -1.0f);
    b_host.at<MatrixFormat::Dense>(1, 0) = Complex(1.25f, 0.75f);
    b_host.at<MatrixFormat::Dense>(2, 0) = Complex(-0.25f, 0.5f);
    b_host.at<MatrixFormat::Dense>(0, 1) = Complex(1.5f, 0.0f);
    b_host.at<MatrixFormat::Dense>(1, 1) = Complex(-0.5f, -0.25f);
    b_host.at<MatrixFormat::Dense>(2, 1) = Complex(0.75f, 1.0f);

    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = (i >= j)
                ? Complex(2.0f + i + j, (i == j) ? 0.0f : (0.5f * (i - j)))
                : Complex(-4.0f - i - j, 1.0f + i + j);
        }
    }

    const auto expected = reference_rank2k(a.view(), b.view(), c.view(), Complex(-0.5f, 0.25f), Complex(1.0f, 0.0f), Uplo::Lower, Transpose::NoTrans, true);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::her2k_workspace_elements<Complex, Uplo::Lower, Transpose::NoTrans>(
        launch, c_view.rows(), a_view.cols());

    run_group_kernel_with_workspace<Complex>(ctx, local_size, workspace_elements, [=](auto group, Complex* workspace) {
        batchlas::device::her2k<Uplo::Lower, Transpose::NoTrans>(
            group, a_view, b_view, c_view, Complex(-0.5f, 0.25f), Complex(1.0f, 0.0f), workspace);
    });

    expect_matrix_matches_vector(c.view(), expected);
}

TEST(DeviceBlasTest, HerkGroupMatchesReference) {
    Queue ctx(Device::default_device());

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(3, 2);
    Matrix<Complex, MatrixFormat::Dense> c(3, 3);
    auto a_host = a.view();
    auto c_host = c.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = Complex(1.0f, 0.5f);
    a_host.at<MatrixFormat::Dense>(1, 0) = Complex(-0.5f, 1.5f);
    a_host.at<MatrixFormat::Dense>(2, 0) = Complex(2.0f, -1.0f);
    a_host.at<MatrixFormat::Dense>(0, 1) = Complex(-1.0f, 0.25f);
    a_host.at<MatrixFormat::Dense>(1, 1) = Complex(0.75f, -0.5f);
    a_host.at<MatrixFormat::Dense>(2, 1) = Complex(1.25f, 0.5f);

    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = (i >= j)
                ? Complex(2.0f + i + j, (i == j) ? 0.0f : (0.5f * (i - j)))
                : Complex(-4.0f - i - j, 1.0f + i + j);
        }
    }

    const auto expected = reference_rankk(a.view(), c.view(), Complex(-0.5f, 0.25f), Complex(1.0f, 0.0f), Uplo::Lower, Transpose::NoTrans, true);
    auto a_view = a.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::herk_workspace_elements<Complex, Uplo::Lower, Transpose::NoTrans>(
        launch, c_view.rows(), a_view.cols());

    run_group_kernel_with_workspace<Complex>(ctx, local_size, workspace_elements, [=](auto group, Complex* workspace) {
        batchlas::device::herk<Uplo::Lower, Transpose::NoTrans>(group, a_view, c_view, Complex(-0.5f, 0.25f), Complex(1.0f, 0.0f), workspace);
    });

    expect_matrix_matches_vector(c.view(), expected);
}

TEST(DeviceBlasTest, Syr2kNdItemAutoMatchesGeneric) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(24, 6);
    Matrix<float, MatrixFormat::Dense> b(24, 6);
    Matrix<float, MatrixFormat::Dense> c_auto(24, 24);
    Matrix<float, MatrixFormat::Dense> c_generic(24, 24);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_auto_host = c_auto.view();
    auto c_generic_host = c_generic.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((2 * i + j) % 11) - 5);
            b_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((i + 3 * j) % 13) - 6);
        }
    }
    for (int j = 0; j < c_auto.cols(); ++j) {
        for (int i = 0; i < c_auto.rows(); ++i) {
            const float value = static_cast<float>((i >= j) ? (1 + ((i + j) % 7)) : (-3 - ((i + j) % 5)));
            c_auto_host.at<MatrixFormat::Dense>(i, j) = value;
            c_generic_host.at<MatrixFormat::Dense>(i, j) = value;
        }
    }

    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_auto_view = c_auto.view().kernel_view();
    auto c_generic_view = c_generic.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
    const auto auto_workspace = batchlas::device::syr2k_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
        launch, c_auto_view.rows(), a_view.cols());
    const auto generic_workspace = batchlas::device::syr2k_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Generic, Uplo::Lower, Transpose::NoTrans>(
        launch, c_generic_view.rows(), a_view.cols());

    run_nd_item_kernel_with_workspace<float>(ctx, local_size, auto_workspace, [=](sycl::nd_item<1> item, float* workspace) {
        batchlas::device::syr2k<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
            item, a_view, b_view, c_auto_view, -0.5f, 0.75f, workspace);
    });
    run_nd_item_kernel_with_workspace<float>(ctx, local_size, generic_workspace, [=](sycl::nd_item<1> item, float* workspace) {
        batchlas::device::syr2k<batchlas::device::DeviceBlasPolicy::Generic, Uplo::Lower, Transpose::NoTrans>(
            item, a_view, b_view, c_generic_view, -0.5f, 0.75f, workspace);
    });

    for (int j = 0; j < c_auto.cols(); ++j) {
        for (int i = 0; i < c_auto.rows(); ++i) {
            EXPECT_NEAR(c_auto.view().template at<MatrixFormat::Dense>(i, j),
                        c_generic.view().template at<MatrixFormat::Dense>(i, j),
                        1e-4f)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(DeviceBlasTest, SyrkNdItem3DTiledTransposeMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Matrix register-tiled device BLAS path requires work-group size >= 256";
    }

    Matrix<float, MatrixFormat::Dense> a(48, 96);
    Matrix<float, MatrixFormat::Dense> c(96, 96);
    auto a_host = a.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((5 * i + 3 * j) % 17) - 8);
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>((i >= j) ? (1 + ((i + 2 * j) % 11)) : (-4 - ((i + j) % 7)));
        }
    }

    const auto expected = reference_rankk(a.view(), c.view(), -0.6f, 0.8f, Uplo::Lower, Transpose::Trans, false);
    auto a_view = a.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_3d_launch_info(ctx);
    const auto workspace_elements = batchlas::device::syrk_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::Trans>(
        launch, c_view.rows(), a_view.rows());

    run_nd_item_kernel_3d_with_workspace<float>(ctx, c.rows(), c.cols(), workspace_elements, [=](sycl::nd_item<3> item, float* workspace) {
        batchlas::device::syrk<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::Trans>(
            item, a_view, c_view, -0.6f, 0.8f, workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 5e-4);
}

TEST(DeviceBlasTest, Syr2kNdItem3DTiledTransposeMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Matrix register-tiled device BLAS path requires work-group size >= 256";
    }

    Matrix<float, MatrixFormat::Dense> a(48, 96);
    Matrix<float, MatrixFormat::Dense> b(48, 96);
    Matrix<float, MatrixFormat::Dense> c(96, 96);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((5 * i + 3 * j) % 17) - 8);
            b_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((7 * i + 2 * j) % 19) - 9);
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>((i >= j) ? (1 + ((i + 2 * j) % 11)) : (-4 - ((i + j) % 7)));
        }
    }

    const auto expected = reference_rank2k(a.view(), b.view(), c.view(), -0.6f, 0.8f, Uplo::Lower, Transpose::Trans, false);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_3d_launch_info(ctx);
    const auto workspace_elements = batchlas::device::syr2k_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::Trans>(
        launch, c_view.rows(), a_view.rows());

    run_nd_item_kernel_3d_with_workspace<float>(ctx, c.rows(), c.cols(), workspace_elements, [=](sycl::nd_item<3> item, float* workspace) {
        batchlas::device::syr2k<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::Trans>(
            item, a_view, b_view, c_view, -0.6f, 0.8f, workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 5e-4);
}

TEST(DeviceBlasTest, Her2kNdItem3DTiledMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Matrix register-tiled device BLAS path requires work-group size >= 256";
    }

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(96, 48);
    Matrix<Complex, MatrixFormat::Dense> b(96, 48);
    Matrix<Complex, MatrixFormat::Dense> c(96, 96);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<float>(((3 * i + 5 * j) % 23) - 11), static_cast<float>(((2 * i + j) % 9) - 4));
            b_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<float>(((4 * i + 7 * j) % 21) - 10), static_cast<float>(((i + 3 * j) % 11) - 5));
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = (i >= j)
                ? Complex(static_cast<float>(2 + ((i + j) % 13)), (i == j) ? 0.0f : static_cast<float>(((2 * i - j) % 7) - 3))
                : Complex(static_cast<float>(-5 - ((i + j) % 9)), static_cast<float>(((i + 2 * j) % 8) - 4));
        }
    }

    const auto expected = reference_rank2k(a.view(), b.view(), c.view(), Complex(-0.4f, 0.3f), Complex(0.9f, 0.0f), Uplo::Lower, Transpose::NoTrans, true);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_3d_launch_info(ctx);
    const auto workspace_elements = batchlas::device::her2k_workspace_elements<Complex, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
        launch, c_view.rows(), a_view.cols());

    run_nd_item_kernel_3d_with_workspace<Complex>(ctx, c.rows(), c.cols(), workspace_elements, [=](sycl::nd_item<3> item, Complex* workspace) {
        batchlas::device::her2k<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
            item, a_view, b_view, c_view, Complex(-0.4f, 0.3f), Complex(0.9f, 0.0f), workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 1e-3);
}

TEST(DeviceBlasTest, HerkNdItem3DTiledMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Matrix register-tiled device BLAS path requires work-group size >= 256";
    }

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(96, 48);
    Matrix<Complex, MatrixFormat::Dense> c(96, 96);
    auto a_host = a.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<float>(((3 * i + 5 * j) % 23) - 11), static_cast<float>(((2 * i + j) % 9) - 4));
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = (i >= j)
                ? Complex(static_cast<float>(2 + ((i + j) % 13)), (i == j) ? 0.0f : static_cast<float>(((2 * i - j) % 7) - 3))
                : Complex(static_cast<float>(-5 - ((i + j) % 9)), static_cast<float>(((i + 2 * j) % 8) - 4));
        }
    }

    const auto expected = reference_rankk(a.view(), c.view(), Complex(-0.4f, 0.3f), Complex(0.9f, 0.0f), Uplo::Lower, Transpose::NoTrans, true);
    auto a_view = a.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_3d_launch_info(ctx);
    const auto workspace_elements = batchlas::device::herk_workspace_elements<Complex, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
        launch, c_view.rows(), a_view.cols());

    run_nd_item_kernel_3d_with_workspace<Complex>(ctx, c.rows(), c.cols(), workspace_elements, [=](sycl::nd_item<3> item, Complex* workspace) {
        batchlas::device::herk<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
            item, a_view, c_view, Complex(-0.4f, 0.3f), Complex(0.9f, 0.0f), workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 1e-3);
}

TEST(DeviceBlasTest, Her2kNdItem3DTiledConjTransposeMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Matrix register-tiled device BLAS path requires work-group size >= 256";
    }

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(48, 96);
    Matrix<Complex, MatrixFormat::Dense> b(48, 96);
    Matrix<Complex, MatrixFormat::Dense> c(96, 96);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<float>(((5 * i + 2 * j) % 27) - 13), static_cast<float>(((3 * i + 4 * j) % 11) - 5));
            b_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<float>(((6 * i + 5 * j) % 25) - 12), static_cast<float>(((2 * i + j) % 13) - 6));
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = (i >= j)
                ? Complex(static_cast<float>(3 + ((i + 2 * j) % 9)), (i == j) ? 0.0f : static_cast<float>(((i - 2 * j) % 7) - 3))
                : Complex(static_cast<float>(-6 - ((i + j) % 10)), static_cast<float>(((i + 3 * j) % 9) - 4));
        }
    }

    const auto expected = reference_rank2k(a.view(), b.view(), c.view(), Complex(-0.35f, 0.2f), Complex(0.8f, 0.0f), Uplo::Lower, Transpose::ConjTrans, true);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_3d_launch_info(ctx);
    const auto workspace_elements = batchlas::device::her2k_workspace_elements<Complex, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::ConjTrans>(
        launch, c_view.rows(), a_view.rows());

    run_nd_item_kernel_3d_with_workspace<Complex>(ctx, c.rows(), c.cols(), workspace_elements, [=](sycl::nd_item<3> item, Complex* workspace) {
        batchlas::device::her2k<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::ConjTrans>(
            item, a_view, b_view, c_view, Complex(-0.35f, 0.2f), Complex(0.8f, 0.0f), workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 1e-3);
}

TEST(DeviceBlasTest, Her2kNdItem1DCompactMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Compact 1D complex her2k path requires work-group size >= 256";
    }

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(96, 48);
    Matrix<Complex, MatrixFormat::Dense> b(96, 48);
    Matrix<Complex, MatrixFormat::Dense> c(96, 96);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<float>(((3 * i + 5 * j) % 23) - 11), static_cast<float>(((2 * i + j) % 9) - 4));
            b_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<float>(((4 * i + 7 * j) % 21) - 10), static_cast<float>(((i + 3 * j) % 11) - 5));
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = (i >= j)
                ? Complex(static_cast<float>(2 + ((i + j) % 13)), (i == j) ? 0.0f : static_cast<float>(((2 * i - j) % 7) - 3))
                : Complex(static_cast<float>(-5 - ((i + j) % 9)), static_cast<float>(((i + 2 * j) % 8) - 4));
        }
    }

    const auto expected = reference_rank2k(a.view(), b.view(), c.view(), Complex(-0.4f, 0.3f), Complex(0.9f, 0.0f), Uplo::Lower, Transpose::NoTrans, true);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const size_t local_size = static_cast<size_t>(batchlas::device::detail::subgroup::kComplexRank2kThreadsPerGroup);
    const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
    const auto workspace_elements = batchlas::device::her2k_workspace_elements<Complex, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
        launch, c_view.rows(), a_view.cols());

    run_nd_item_kernel_with_workspace<Complex>(ctx, local_size, workspace_elements, [=](sycl::nd_item<1> item, Complex* workspace) {
        batchlas::device::her2k<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
            item, a_view, b_view, c_view, Complex(-0.4f, 0.3f), Complex(0.9f, 0.0f), workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 1e-3);
}

TEST(DeviceBlasTest, Her2kNdItem1DCompactConjTransposeMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Compact 1D complex her2k path requires work-group size >= 256";
    }

    using Complex = std::complex<float>;
    Matrix<Complex, MatrixFormat::Dense> a(48, 96);
    Matrix<Complex, MatrixFormat::Dense> b(48, 96);
    Matrix<Complex, MatrixFormat::Dense> c(96, 96);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<float>(((5 * i + 2 * j) % 27) - 13), static_cast<float>(((3 * i + 4 * j) % 11) - 5));
            b_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<float>(((6 * i + 5 * j) % 25) - 12), static_cast<float>(((2 * i + j) % 13) - 6));
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = (i >= j)
                ? Complex(static_cast<float>(3 + ((i + 2 * j) % 9)), (i == j) ? 0.0f : static_cast<float>(((i - 2 * j) % 7) - 3))
                : Complex(static_cast<float>(-6 - ((i + j) % 10)), static_cast<float>(((i + 3 * j) % 9) - 4));
        }
    }

    const auto expected = reference_rank2k(a.view(), b.view(), c.view(), Complex(-0.35f, 0.2f), Complex(0.8f, 0.0f), Uplo::Lower, Transpose::ConjTrans, true);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const size_t local_size = static_cast<size_t>(batchlas::device::detail::subgroup::kComplexRank2kThreadsPerGroup);
    const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
    const auto workspace_elements = batchlas::device::her2k_workspace_elements<Complex, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::ConjTrans>(
        launch, c_view.rows(), a_view.rows());

    run_nd_item_kernel_with_workspace<Complex>(ctx, local_size, workspace_elements, [=](sycl::nd_item<1> item, Complex* workspace) {
        batchlas::device::her2k<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::ConjTrans>(
            item, a_view, b_view, c_view, Complex(-0.35f, 0.2f), Complex(0.8f, 0.0f), workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 1e-3);
}

TEST(DeviceBlasTest, Her2kNdItem1DComplexDoubleGenericMatchesReference) {
    Queue ctx(Device::default_device());

    using Complex = std::complex<double>;
    Matrix<Complex, MatrixFormat::Dense> a(64, 32);
    Matrix<Complex, MatrixFormat::Dense> b(64, 32);
    Matrix<Complex, MatrixFormat::Dense> c(64, 64);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<double>(((3 * i + 5 * j) % 29) - 14),
                                                           static_cast<double>(((2 * i + j) % 13) - 6));
            b_host.at<MatrixFormat::Dense>(i, j) = Complex(static_cast<double>(((4 * i + 7 * j) % 31) - 15),
                                                           static_cast<double>(((i + 3 * j) % 17) - 8));
        }
    }
    for (int j = 0; j < c.cols(); ++j) {
        for (int i = 0; i < c.rows(); ++i) {
            c_host.at<MatrixFormat::Dense>(i, j) = (i >= j)
                ? Complex(static_cast<double>(2 + ((i + j) % 19)),
                          (i == j) ? 0.0 : static_cast<double>(((2 * i - j) % 11) - 5))
                : Complex(static_cast<double>(-5 - ((i + j) % 23)),
                          static_cast<double>(((i + 2 * j) % 13) - 6));
        }
    }

    const auto expected = reference_rank2k(a.view(), b.view(), c.view(), Complex(-0.4, 0.3), Complex(0.9, 0.0), Uplo::Lower, Transpose::NoTrans, true);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
    const auto workspace_elements = batchlas::device::her2k_workspace_elements<Complex, batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
        launch, c_view.rows(), a_view.cols());

    run_nd_item_kernel_with_workspace<Complex>(ctx, local_size, workspace_elements, [=](sycl::nd_item<1> item, Complex* workspace) {
        batchlas::device::her2k<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
            item, a_view, b_view, c_view, Complex(-0.4, 0.3), Complex(0.9, 0.0), workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 1e-10);
}

TEST(DeviceBlasTest, TrmmLeftLowerMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Matrix<float, MatrixFormat::Dense> b(3, 2);
    Matrix<float, MatrixFormat::Dense> c(3, 2);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -1.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 4.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = 99.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = 3.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 5.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = -7.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 42.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = -2.0f;

    b_host.at<MatrixFormat::Dense>(0, 0) = 1.0f;
    b_host.at<MatrixFormat::Dense>(1, 0) = -2.0f;
    b_host.at<MatrixFormat::Dense>(2, 0) = 0.5f;
    b_host.at<MatrixFormat::Dense>(0, 1) = 3.0f;
    b_host.at<MatrixFormat::Dense>(1, 1) = 1.0f;
    b_host.at<MatrixFormat::Dense>(2, 1) = -1.0f;

    c_host.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    c_host.at<MatrixFormat::Dense>(1, 0) = -1.0f;
    c_host.at<MatrixFormat::Dense>(2, 0) = 4.0f;
    c_host.at<MatrixFormat::Dense>(0, 1) = 0.5f;
    c_host.at<MatrixFormat::Dense>(1, 1) = -3.0f;
    c_host.at<MatrixFormat::Dense>(2, 1) = 2.0f;

    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const float alpha = 1.5f;
    const float beta = -0.25f;
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::trmm_workspace_elements<Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, float>(
        launch, c_view.rows(), c_view.cols(), false);

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto group, float* workspace) {
        batchlas::device::trmm<Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit>(
            group, a_view, b_view, c_view, alpha, beta, workspace);
    });

    const std::array<float, 6> expected{
        2.5f,
        -10.25f,
        -11.5f,
        8.875f,
        0.75f,
        28.0f,
    };

    expect_matrix_near(c.view(), expected);
}

TEST(DeviceBlasTest, TrmmRightUpperTransposeUnitDiagonalMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Matrix<float, MatrixFormat::Dense> b(2, 3);
    Matrix<float, MatrixFormat::Dense> c(2, 3);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 4.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -8.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 6.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = -3.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 9.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = -1.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 5.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = 7.0f;

    b_host.at<MatrixFormat::Dense>(0, 0) = 1.0f;
    b_host.at<MatrixFormat::Dense>(1, 0) = 3.0f;
    b_host.at<MatrixFormat::Dense>(0, 1) = -2.0f;
    b_host.at<MatrixFormat::Dense>(1, 1) = 1.0f;
    b_host.at<MatrixFormat::Dense>(0, 2) = 0.5f;
    b_host.at<MatrixFormat::Dense>(1, 2) = -1.0f;

    c_host.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    c_host.at<MatrixFormat::Dense>(1, 0) = 0.5f;
    c_host.at<MatrixFormat::Dense>(0, 1) = -1.0f;
    c_host.at<MatrixFormat::Dense>(1, 1) = -3.0f;
    c_host.at<MatrixFormat::Dense>(0, 2) = 4.0f;
    c_host.at<MatrixFormat::Dense>(1, 2) = 2.0f;

    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const float alpha = -1.25f;
    const float beta = 0.5f;
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::trmm_workspace_elements<Side::Right, Uplo::Upper, Transpose::Trans, Diag::Unit, float>(
        launch, c_view.rows(), c_view.cols(), false);

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto group, float* workspace) {
        batchlas::device::trmm<Side::Right, Uplo::Upper, Transpose::Trans, Diag::Unit>(
            group, a_view, b_view, c_view, alpha, beta, workspace);
    });

    const std::array<float, 6> expected{
        5.375f,
        -7.25f,
        -1.125f,
        3.5f,
        1.375f,
        2.25f,
    };

    expect_matrix_near(c.view(), expected);
}

TEST(DeviceBlasTest, TrmmLeftUpperNoTransSupportsInPlaceOutput) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Matrix<float, MatrixFormat::Dense> b(3, 2);
    auto a_host = a.view();
    auto b_host = b.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -8.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 6.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = 3.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = 4.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 9.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = -1.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 5.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = 6.0f;

    b_host.at<MatrixFormat::Dense>(0, 0) = 1.0f;
    b_host.at<MatrixFormat::Dense>(1, 0) = -2.0f;
    b_host.at<MatrixFormat::Dense>(2, 0) = 0.5f;
    b_host.at<MatrixFormat::Dense>(0, 1) = 3.0f;
    b_host.at<MatrixFormat::Dense>(1, 1) = 1.0f;
    b_host.at<MatrixFormat::Dense>(2, 1) = -1.0f;

    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::trmm_workspace_elements<Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, float>(
        launch, b_view.rows(), b_view.cols(), true);

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](const auto& group, float* workspace) {
        batchlas::device::trmm<Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>(
            group, a_view, b_view, b_view, 1.0f, 0.5f, workspace);
    });

    const std::array<float, 6> expected{-4.0f, -6.5f, 3.25f, 11.5f, -0.5f, -6.5f};
    expect_matrix_near(b.view(), expected);
}

TEST(DeviceBlasTest, TrmmRightUpperNoTransSupportsInPlaceOutput) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Matrix<float, MatrixFormat::Dense> b(2, 3);
    auto a_host = a.view();
    auto b_host = b.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -8.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 6.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = 3.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = 4.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 9.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = -1.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 5.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = 6.0f;

    b_host.at<MatrixFormat::Dense>(0, 0) = 1.0f;
    b_host.at<MatrixFormat::Dense>(1, 0) = 3.0f;
    b_host.at<MatrixFormat::Dense>(0, 1) = -2.0f;
    b_host.at<MatrixFormat::Dense>(1, 1) = 1.0f;
    b_host.at<MatrixFormat::Dense>(0, 2) = 0.5f;
    b_host.at<MatrixFormat::Dense>(1, 2) = -1.0f;

    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::trmm_workspace_elements<Side::Right, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, float>(
        launch, b_view.rows(), b_view.cols(), true);

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](const auto& group, float* workspace) {
        batchlas::device::trmm<Side::Right, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>(
            group, a_view, b_view, b_view, 1.0f, 0.5f, workspace);
    });

    const std::array<float, 6> expected{2.5f, 7.5f, -6.0f, 13.5f, -7.75f, -4.5f};
    expect_matrix_near(b.view(), expected);
}

TEST(DeviceBlasTest, GemvNdItemGenericAndAutoMatch) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(8, 8);
    Vector<float> x(8);
    Vector<float> y_auto(8);
    Vector<float> y_generic(8);

    fill_dense_test_matrix(a);
    for (int i = 0; i < 8; ++i) {
        x(i) = static_cast<float>(i + 1);
        y_auto(i) = static_cast<float>(i - 2);
        y_generic(i) = y_auto(i);
    }

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_auto_view = VectorView<float>(y_auto);
    auto y_generic_view = VectorView<float>(y_generic);
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
    const auto auto_workspace = batchlas::device::gemv_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Transpose::NoTrans>(
        launch, a_view.rows(), a_view.cols());
    const auto generic_workspace = batchlas::device::gemv_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Generic, Transpose::NoTrans>(
        launch, a_view.rows(), a_view.cols());

    run_nd_item_kernel_with_workspace<float>(ctx, local_size, auto_workspace, [=](sycl::nd_item<1> item, float* workspace) {
        batchlas::device::gemv<batchlas::device::DeviceBlasPolicy::Auto, Transpose::NoTrans>(
            item, a_view, x_view, y_auto_view, 1.25f, -0.5f, workspace);
    });
    run_nd_item_kernel_with_workspace<float>(ctx, local_size, generic_workspace, [=](sycl::nd_item<1> item, float* workspace) {
        batchlas::device::gemv<batchlas::device::DeviceBlasPolicy::Generic, Transpose::NoTrans>(
            item, a_view, x_view, y_generic_view, 1.25f, -0.5f, workspace);
    });

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(y_auto(i), y_generic(i), 1e-5f) << "Mismatch at index " << i;
    }
}

TEST(DeviceBlasTest, GemvNdItemTiledAutoMatchesGeneric) {
    Queue ctx(Device::default_device());

    const size_t local_size = device_test_work_group_size(ctx);
    if (local_size < 16) {
        GTEST_SKIP() << "Device work-group size is too small to exercise tiled GEMV";
    }

    constexpr int rows = 32;
    constexpr int cols = 32;
    Matrix<float, MatrixFormat::Dense> a(rows, cols);
    Vector<float> x(cols);
    Vector<float> y_auto(rows);
    Vector<float> y_generic(rows);

    auto a_host = a.view();
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            a_host.at<MatrixFormat::Dense>(row, col) = static_cast<float>((row % 7) - 0.25f * (col % 5) + 0.1f * (row - col));
        }
    }
    for (int i = 0; i < cols; ++i) {
        x(i) = static_cast<float>((i % 9) - 4);
    }
    for (int i = 0; i < rows; ++i) {
        const float initial = static_cast<float>(0.5f * i - 3.0f);
        y_auto(i) = initial;
        y_generic(i) = initial;
    }

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_auto_view = VectorView<float>(y_auto);
    auto y_generic_view = VectorView<float>(y_generic);
    const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
    const auto auto_workspace = batchlas::device::gemv_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Transpose::NoTrans>(
        launch, a_view.rows(), a_view.cols());
    const auto generic_workspace = batchlas::device::gemv_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Generic, Transpose::NoTrans>(
        launch, a_view.rows(), a_view.cols());

    run_nd_item_kernel_with_workspace<float>(ctx, local_size, auto_workspace, [=](sycl::nd_item<1> item, float* workspace) {
        batchlas::device::gemv<batchlas::device::DeviceBlasPolicy::Auto, Transpose::NoTrans>(
            item, a_view, x_view, y_auto_view, -0.75f, 1.25f, workspace);
    });
    run_nd_item_kernel_with_workspace<float>(ctx, local_size, generic_workspace, [=](sycl::nd_item<1> item, float* workspace) {
        batchlas::device::gemv<batchlas::device::DeviceBlasPolicy::Generic, Transpose::NoTrans>(
            item, a_view, x_view, y_generic_view, -0.75f, 1.25f, workspace);
    });

    for (int i = 0; i < rows; ++i) {
        EXPECT_NEAR(y_auto(i), y_generic(i), 1e-4f) << "Mismatch at index " << i;
    }
}

TEST(DeviceBlasTest, TrmvNdItemMatchesGroupReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(4, 4);
    Vector<float> x(4);
    Vector<float> y_group(4);
    Vector<float> y_item(4);
    auto a_host = a.view();

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>((j + 1) * 7 + i + 1);
        }
    }
    for (int i = 0; i < 4; ++i) {
        x(i) = static_cast<float>(i + 1);
        y_group(i) = static_cast<float>(2 - i);
        y_item(i) = y_group(i);
    }

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_group_view = VectorView<float>(y_group);
    auto y_item_view = VectorView<float>(y_item);
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](auto group) {
        batchlas::device::trmv<Uplo::Lower, Transpose::NoTrans, Diag::Unit>(group, a_view, x_view, y_group_view, 0.75f, 0.25f);
    });
    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::trmv<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans, Diag::Unit>(
            item, a_view, x_view, y_item_view, 0.75f, 0.25f);
    });

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(y_group(i), y_item(i), 1e-5f) << "Mismatch at index " << i;
    }
}

TEST(DeviceBlasTest, TrmvTransposeNdItemMatchesGroupReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(8, 8);
    Vector<float> x(8);
    Vector<float> y_group(8);
    Vector<float> y_item(8);
    auto a_host = a.view();

    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>((j + 1) * 5 + i + 1);
        }
    }
    for (int i = 0; i < 8; ++i) {
        x(i) = static_cast<float>(i + 1) * 0.5f;
        y_group(i) = static_cast<float>(1 - i);
        y_item(i) = y_group(i);
    }

    auto a_view = a.view().kernel_view();
    auto x_view = VectorView<float>(x);
    auto y_group_view = VectorView<float>(y_group);
    auto y_item_view = VectorView<float>(y_item);
    const size_t local_size = device_test_work_group_size(ctx);

    run_group_kernel(ctx, local_size, [=](auto group) {
        batchlas::device::trmv<Uplo::Lower, Transpose::Trans, Diag::NonUnit>(group, a_view, x_view, y_group_view, 1.25f, -0.25f);
    });
    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::trmv<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::Trans, Diag::NonUnit>(
            item, a_view, x_view, y_item_view, 1.25f, -0.25f);
    });

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(y_group(i), y_item(i), 1e-5f) << "Mismatch at index " << i;
    }
}

TEST(DeviceBlasTest, SymmGroupMatchesReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(3, 3);
    Matrix<float, MatrixFormat::Dense> b(3, 2);
    Matrix<float, MatrixFormat::Dense> c(3, 2);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_initial = c.view();

    a_host.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    a_host.at<MatrixFormat::Dense>(1, 0) = -1.0f;
    a_host.at<MatrixFormat::Dense>(2, 0) = 4.0f;
    a_host.at<MatrixFormat::Dense>(0, 1) = 99.0f;
    a_host.at<MatrixFormat::Dense>(1, 1) = 3.0f;
    a_host.at<MatrixFormat::Dense>(2, 1) = 5.0f;
    a_host.at<MatrixFormat::Dense>(0, 2) = -7.0f;
    a_host.at<MatrixFormat::Dense>(1, 2) = 42.0f;
    a_host.at<MatrixFormat::Dense>(2, 2) = -2.0f;

    b_host.at<MatrixFormat::Dense>(0, 0) = 1.0f;
    b_host.at<MatrixFormat::Dense>(1, 0) = -2.0f;
    b_host.at<MatrixFormat::Dense>(2, 0) = 0.5f;
    b_host.at<MatrixFormat::Dense>(0, 1) = 3.0f;
    b_host.at<MatrixFormat::Dense>(1, 1) = 1.0f;
    b_host.at<MatrixFormat::Dense>(2, 1) = -1.0f;

    c_initial.at<MatrixFormat::Dense>(0, 0) = 2.0f;
    c_initial.at<MatrixFormat::Dense>(1, 0) = -1.0f;
    c_initial.at<MatrixFormat::Dense>(2, 0) = 4.0f;
    c_initial.at<MatrixFormat::Dense>(0, 1) = 0.5f;
    c_initial.at<MatrixFormat::Dense>(1, 1) = -3.0f;
    c_initial.at<MatrixFormat::Dense>(2, 1) = 2.0f;

    const auto expected = reference_symm(a.view(), b.view(), c.view(), 1.5f, -0.25f, Side::Left, Uplo::Lower);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_group_launch_info(local_size);
    const auto workspace_elements = batchlas::device::symm_workspace_elements<float, Side::Left, Uplo::Lower>(launch, c_view.rows(), c_view.cols());

    run_group_kernel_with_workspace<float>(ctx, local_size, workspace_elements, [=](auto group, float* workspace) {
        batchlas::device::symm<Side::Left, Uplo::Lower>(group, a_view, b_view, c_view, 1.5f, -0.25f, workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 5e-4);
}

TEST(DeviceBlasTest, SymmNdItemAutoAndGenericMatchReference) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> a(4, 4);
    Matrix<float, MatrixFormat::Dense> b(2, 4);
    Matrix<float, MatrixFormat::Dense> c_auto(2, 4);
    Matrix<float, MatrixFormat::Dense> c_generic(2, 4);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_auto_host = c_auto.view();
    auto c_generic_host = c_generic.view();

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>((j + 2) * 5 + i);
        }
    }
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 2; ++i) {
            const float value = static_cast<float>(1 + i + j * 2);
            b_host.at<MatrixFormat::Dense>(i, j) = value;
            c_auto_host.at<MatrixFormat::Dense>(i, j) = value * 0.5f;
            c_generic_host.at<MatrixFormat::Dense>(i, j) = value * 0.5f;
        }
    }

    const auto expected = reference_symm(a.view(), b.view(), c_auto.view(), 0.8f, 0.3f, Side::Right, Uplo::Upper);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_auto_view = c_auto.view().kernel_view();
    auto c_generic_view = c_generic.view().kernel_view();
    const size_t local_size = device_test_work_group_size(ctx);
    const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
    const auto auto_workspace = batchlas::device::symm_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Side::Right, Uplo::Upper>(
        launch, c_auto_view.rows(), c_auto_view.cols());
    const auto generic_workspace = batchlas::device::symm_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Generic, Side::Right, Uplo::Upper>(
        launch, c_generic_view.rows(), c_generic_view.cols());

    run_nd_item_kernel_with_workspace<float>(ctx, local_size, auto_workspace, [=](sycl::nd_item<1> item, float* workspace) {
        batchlas::device::symm<batchlas::device::DeviceBlasPolicy::Auto, Side::Right, Uplo::Upper>(
            item, a_view, b_view, c_auto_view, 0.8f, 0.3f, workspace);
    });
    run_nd_item_kernel_with_workspace<float>(ctx, local_size, generic_workspace, [=](sycl::nd_item<1> item, float* workspace) {
        batchlas::device::symm<batchlas::device::DeviceBlasPolicy::Generic, Side::Right, Uplo::Upper>(
            item, a_view, b_view, c_generic_view, 0.8f, 0.3f, workspace);
    });

    expect_matrix_matches_vector(c_auto.view(), expected);
    expect_matrix_matches_vector(c_generic.view(), expected);
}

TEST(DeviceBlasTest, GemmNdItemMatchesReferenceAcrossTransforms) {
    Queue ctx(Device::default_device());
    const size_t local_size = device_test_work_group_size(ctx);

    for (auto trans_a : {Transpose::NoTrans, Transpose::Trans}) {
        for (auto trans_b : {Transpose::NoTrans, Transpose::Trans}) {
            constexpr int m = 4;
            constexpr int n = 3;
            constexpr int k = 5;
            Matrix<float, MatrixFormat::Dense> a(trans_a == Transpose::NoTrans ? m : k,
                                                 trans_a == Transpose::NoTrans ? k : m);
            Matrix<float, MatrixFormat::Dense> b(trans_b == Transpose::NoTrans ? k : n,
                                                 trans_b == Transpose::NoTrans ? n : k);
            Matrix<float, MatrixFormat::Dense> c_auto(m, n);
            Matrix<float, MatrixFormat::Dense> c_generic(m, n);
            auto a_host = a.view();
            auto b_host = b.view();
            auto c_auto_host = c_auto.view();
            auto c_generic_host = c_generic.view();

            for (int j = 0; j < a.cols(); ++j) {
                for (int i = 0; i < a.rows(); ++i) {
                    a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((3 * i + 2 * j) % 11) - 4);
                }
            }
            for (int j = 0; j < b.cols(); ++j) {
                for (int i = 0; i < b.rows(); ++i) {
                    b_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((i + 5 * j) % 13) - 6);
                }
            }
            for (int j = 0; j < c_auto.cols(); ++j) {
                for (int i = 0; i < c_auto.rows(); ++i) {
                    const float value = static_cast<float>(((2 * i + j) % 7) - 2);
                    c_auto_host.at<MatrixFormat::Dense>(i, j) = value;
                    c_generic_host.at<MatrixFormat::Dense>(i, j) = value;
                }
            }

            const auto expected = reference_gemm(a.view(), b.view(), c_auto.view(), 1.25f, -0.35f, trans_a, trans_b);
            auto a_view = a.view().kernel_view();
            auto b_view = b.view().kernel_view();
            auto c_auto_view = c_auto.view().kernel_view();
            auto c_generic_view = c_generic.view().kernel_view();

            auto run_case = [&](auto trans_a_tag, auto trans_b_tag) {
                constexpr auto TransAValue = decltype(trans_a_tag)::value;
                constexpr auto TransBValue = decltype(trans_b_tag)::value;
                const auto launch = device_test_nd_item_1d_launch_info(ctx, local_size);
                const auto auto_workspace = batchlas::device::gemm_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, TransAValue, TransBValue>(
                    launch,
                    c_auto_view.rows(),
                    c_auto_view.cols(),
                    TransAValue == Transpose::NoTrans ? a_view.cols() : a_view.rows(),
                    false,
                    false);
                const auto generic_workspace = batchlas::device::gemm_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Generic, TransAValue, TransBValue>(
                    launch,
                    c_generic_view.rows(),
                    c_generic_view.cols(),
                    TransAValue == Transpose::NoTrans ? a_view.cols() : a_view.rows(),
                    false,
                    false);
                run_nd_item_kernel_with_workspace<float>(ctx, local_size, auto_workspace, [=](sycl::nd_item<1> item, float* workspace) {
                    batchlas::device::gemm<batchlas::device::DeviceBlasPolicy::Auto, TransAValue, TransBValue>(
                        item, a_view, b_view, c_auto_view, 1.25f, -0.35f, workspace);
                });
                run_nd_item_kernel_with_workspace<float>(ctx, local_size, generic_workspace, [=](sycl::nd_item<1> item, float* workspace) {
                    batchlas::device::gemm<batchlas::device::DeviceBlasPolicy::Generic, TransAValue, TransBValue>(
                        item, a_view, b_view, c_generic_view, 1.25f, -0.35f, workspace);
                });
            };

            if (trans_a == Transpose::NoTrans && trans_b == Transpose::NoTrans) {
                run_case(std::integral_constant<Transpose, Transpose::NoTrans>{},
                         std::integral_constant<Transpose, Transpose::NoTrans>{});
            } else if (trans_a == Transpose::NoTrans && trans_b == Transpose::Trans) {
                run_case(std::integral_constant<Transpose, Transpose::NoTrans>{},
                         std::integral_constant<Transpose, Transpose::Trans>{});
            } else if (trans_a == Transpose::Trans && trans_b == Transpose::NoTrans) {
                run_case(std::integral_constant<Transpose, Transpose::Trans>{},
                         std::integral_constant<Transpose, Transpose::NoTrans>{});
            } else {
                run_case(std::integral_constant<Transpose, Transpose::Trans>{},
                         std::integral_constant<Transpose, Transpose::Trans>{});
            }

            expect_matrix_matches_vector(c_auto.view(), expected);
            expect_matrix_matches_vector(c_generic.view(), expected);
        }
    }
}

TEST(DeviceBlasTest, TrmmNdItemMatchesReferenceAcrossTransforms) {
    Queue ctx(Device::default_device());
    const size_t local_size = device_test_work_group_size(ctx);

    run_trmm_nd_item_case<Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Left, Uplo::Lower, Transpose::Trans, Diag::NonUnit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Left, Uplo::Lower, Transpose::Trans, Diag::Unit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::Unit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Left, Uplo::Upper, Transpose::Trans, Diag::NonUnit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Left, Uplo::Upper, Transpose::Trans, Diag::Unit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Right, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Right, Uplo::Lower, Transpose::NoTrans, Diag::Unit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Right, Uplo::Lower, Transpose::Trans, Diag::NonUnit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Right, Uplo::Lower, Transpose::Trans, Diag::Unit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Right, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Right, Uplo::Upper, Transpose::NoTrans, Diag::Unit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Right, Uplo::Upper, Transpose::Trans, Diag::NonUnit>(ctx, local_size);
    run_trmm_nd_item_case<Side::Right, Uplo::Upper, Transpose::Trans, Diag::Unit>(ctx, local_size);
}

TEST(DeviceBlasTest, TrmmNdItem3DTiledMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Matrix register-tiled device BLAS path requires work-group size >= 256";
    }

    Matrix<float, MatrixFormat::Dense> a(160, 160);
    Matrix<float, MatrixFormat::Dense> b(160, 48);
    Matrix<float, MatrixFormat::Dense> c(160, 48);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = i >= j ? static_cast<float>(1 + ((i + 3 * j) % 17)) : 0.0f;
        }
    }
    for (int j = 0; j < b.cols(); ++j) {
        for (int i = 0; i < b.rows(); ++i) {
            b_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((2 * i + j) % 13) - 4);
            c_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((i + 2 * j) % 7) - 2);
        }
    }

    const auto expected = reference_trmm(a.view(), b.view(), c.view(), 0.9f, -0.2f, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_3d_launch_info(ctx);
    const auto workspace_elements = batchlas::device::trmm_workspace_elements<float,
                                                                               batchlas::device::DeviceBlasPolicy::Auto,
                                                                               Side::Left,
                                                                               Uplo::Lower,
                                                                               Transpose::NoTrans,
                                                                               Diag::NonUnit>(
        launch, c_view.rows(), c_view.cols(), false);

    run_nd_item_kernel_3d_with_workspace<float>(ctx, c.rows(), c.cols(), workspace_elements, [=](sycl::nd_item<3> item, float* workspace) {
        batchlas::device::trmm<batchlas::device::DeviceBlasPolicy::Auto,
                           Side::Left,
                           Uplo::Lower,
                           Transpose::NoTrans,
                           Diag::NonUnit>(item,
                                  a_view,
                                  b_view,
                                  c_view,
                                  0.9f,
                                  -0.2f,
                                  workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 5e-4);
}

TEST(DeviceBlasTest, GemmNdItem3DTiledMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Matrix register-tiled device BLAS path requires work-group size >= 256";
    }

    Matrix<float, MatrixFormat::Dense> a(160, 160);
    Matrix<float, MatrixFormat::Dense> b(160, 48);
    Matrix<float, MatrixFormat::Dense> c(160, 48);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((i + 3 * j) % 17) - 4);
        }
    }
    for (int j = 0; j < b.cols(); ++j) {
        for (int i = 0; i < b.rows(); ++i) {
            b_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((2 * i + 5 * j) % 13) - 6);
            c_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((i + 2 * j) % 7) - 2);
        }
    }

    const auto expected = reference_gemm(a.view(), b.view(), c.view(), 0.95f, -0.2f, Transpose::NoTrans, Transpose::NoTrans);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_3d_launch_info(ctx);
    const auto workspace_elements = batchlas::device::gemm_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Transpose::NoTrans, Transpose::NoTrans>(
        launch, c_view.rows(), c_view.cols(), a_view.cols(), false, false);

    run_nd_item_kernel_3d_with_workspace<float>(ctx, c.rows(), c.cols(), workspace_elements, [=](sycl::nd_item<3> item, float* workspace) {
        batchlas::device::gemm<batchlas::device::DeviceBlasPolicy::Auto, Transpose::NoTrans, Transpose::NoTrans>(
            item, a_view, b_view, c_view, 0.95f, -0.2f, workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 5e-4);
}

TEST(DeviceBlasTest, GemmNdItem3DTiledAlignedLargeMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Matrix register-tiled device BLAS path requires work-group size >= 256";
    }

    Matrix<float, MatrixFormat::Dense> a(256, 256);
    Matrix<float, MatrixFormat::Dense> b(256, 128);
    Matrix<float, MatrixFormat::Dense> c(256, 128);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((7 * i + 3 * j) % 29) - 14);
        }
    }
    for (int j = 0; j < b.cols(); ++j) {
        for (int i = 0; i < b.rows(); ++i) {
            b_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((5 * i + 11 * j) % 31) - 15);
            c_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((2 * i + 3 * j) % 17) - 8);
        }
    }

    const auto expected = reference_gemm(a.view(), b.view(), c.view(), 1.1f, -0.25f, Transpose::NoTrans, Transpose::NoTrans);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_3d_launch_info(ctx);
    const auto workspace_elements = batchlas::device::gemm_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Transpose::NoTrans, Transpose::NoTrans>(
        launch, c_view.rows(), c_view.cols(), a_view.cols(), false, false);

    run_nd_item_kernel_3d_with_workspace<float>(ctx, c.rows(), c.cols(), workspace_elements, [=](sycl::nd_item<3> item, float* workspace) {
        batchlas::device::gemm<batchlas::device::DeviceBlasPolicy::Auto, Transpose::NoTrans, Transpose::NoTrans>(
            item, a_view, b_view, c_view, 1.1f, -0.25f, workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 1e-3);
}

TEST(DeviceBlasTest, SymmNdItem3DTiledMatchesReference) {
    Queue ctx(Device::default_device());
    if (!device_supports_matrix_register_tiles(ctx)) {
        GTEST_SKIP() << "Matrix register-tiled device BLAS path requires work-group size >= 256";
    }

    Matrix<float, MatrixFormat::Dense> a(160, 160);
    Matrix<float, MatrixFormat::Dense> b(160, 48);
    Matrix<float, MatrixFormat::Dense> c(160, 48);
    auto a_host = a.view();
    auto b_host = b.view();
    auto c_host = c.view();

    for (int j = 0; j < a.cols(); ++j) {
        for (int i = 0; i < a.rows(); ++i) {
            if (i >= j) {
                a_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(1 + ((i + 5 * j) % 19));
            } else {
                a_host.at<MatrixFormat::Dense>(i, j) = 0.0f;
            }
        }
    }
    for (int j = 0; j < b.cols(); ++j) {
        for (int i = 0; i < b.rows(); ++i) {
            b_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((3 * i + 2 * j) % 11) - 3);
            c_host.at<MatrixFormat::Dense>(i, j) = static_cast<float>(((i + j) % 5) - 1);
        }
    }

    const auto expected = reference_symm(a.view(), b.view(), c.view(), 1.1f, -0.15f, Side::Left, Uplo::Lower);
    auto a_view = a.view().kernel_view();
    auto b_view = b.view().kernel_view();
    auto c_view = c.view().kernel_view();
    const auto launch = device_test_nd_item_3d_launch_info(ctx);
    const auto workspace_elements = batchlas::device::symm_workspace_elements<float, batchlas::device::DeviceBlasPolicy::Auto, Side::Left, Uplo::Lower>(
        launch, c_view.rows(), c_view.cols());

    run_nd_item_kernel_3d_with_workspace<float>(ctx, c.rows(), c.cols(), workspace_elements, [=](sycl::nd_item<3> item, float* workspace) {
        batchlas::device::symm<batchlas::device::DeviceBlasPolicy::Auto, Side::Left, Uplo::Lower>(
            item, a_view, b_view, c_view, 1.1f, -0.15f, workspace);
    });

    expect_matrix_matches_vector(c.view(), expected, 5e-4);
}
