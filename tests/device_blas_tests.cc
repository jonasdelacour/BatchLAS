#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <blas/linalg.hh>

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
void expect_matrix_matches_vector(const MatrixView<T, MatrixFormat::Dense>& actual,
                                  const std::vector<T>& expected,
                                  double tol = 1e-4) {
    ASSERT_EQ(static_cast<std::size_t>(actual.rows() * actual.cols()), expected.size());
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

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           batchlas::device::gemv(item.get_group(), a_view, x_view, y_view, alpha, beta, Transpose::NoTrans);
                       });
    });
    ctx.wait_and_throw();

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

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           batchlas::device::gemv(item.get_group(), a_view, x_view, y_view, alpha, beta, Transpose::Trans);
                       });
    });
    ctx.wait_and_throw();

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

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           batchlas::device::gemxv(
                               item.get_group(),
                               a_view,
                               Transpose::NoTrans,
                               batchlas::device::make_gemv_operand(x0, y0, alpha0, beta0),
                               batchlas::device::make_gemv_operand(x1, y1, alpha1, beta1));
                       });
    });
    ctx.wait_and_throw();

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
                           batchlas::device::trmv(item.get_group(),
                                                  a_view,
                                                  x_view,
                                                  y_view,
                                                  alpha,
                                                  beta,
                                                  Uplo::Lower,
                                                  Transpose::NoTrans,
                                                  Diag::NonUnit);
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
                           batchlas::device::trmv(item.get_group(),
                                                  a_view,
                                                  x_view,
                                                  y_view,
                                                  alpha,
                                                  beta,
                                                  Uplo::Upper,
                                                  Transpose::Trans,
                                                  Diag::Unit);
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

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           batchlas::device::symv(item.get_group(), a_view, x_view, y_view, alpha, beta, Uplo::Lower);
                       });
    });
    ctx.wait_and_throw();

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

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           batchlas::device::symv(item.get_group(), a_view, x_view, y_view, alpha, beta, Uplo::Upper);
                       });
    });
    ctx.wait_and_throw();

    const std::array<float, 4> expected{
        alpha * (2.0f * 1.5f + -1.0f * -2.0f + 4.0f * 0.5f) + beta * 1.0f,
        alpha * (-1.0f * 1.5f + 3.0f * -2.0f + 5.0f * 0.5f) + beta * -3.0f,
        alpha * (4.0f * 1.5f + 5.0f * -2.0f + -2.0f * 0.5f) + beta * 2.5f,
        0.0f,
    };

    expect_vector_near(VectorView<float>(y), expected, 3);
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

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           batchlas::device::trmm(item.get_group(),
                                                  a_view,
                                                  batchlas::device::make_matmat_operand(b_view, c_view, alpha, beta),
                                                  batchlas::device::TriangularTransform{.side = Side::Left,
                                                                                        .uplo = Uplo::Lower,
                                                                                        .trans = Transpose::NoTrans,
                                                                                        .diag = Diag::NonUnit});
                       });
    });
    ctx.wait_and_throw();

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

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) {
                           batchlas::device::trmm(item.get_group(),
                                                  a_view,
                                                  b_view,
                                                  c_view,
                                                  alpha,
                                                  beta,
                                                  Side::Right,
                                                  Uplo::Upper,
                                                  Transpose::Trans,
                                                  Diag::Unit);
                       });
    });
    ctx.wait_and_throw();

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

    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::gemv(
            item, a_view, x_view, y_auto_view, 1.25f, -0.5f, Transpose::NoTrans, batchlas::device::DeviceBlasPolicy::Auto);
    });
    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::gemv(item,
                               a_view,
                               x_view,
                               y_generic_view,
                               1.25f,
                               -0.5f,
                               Transpose::NoTrans,
                               batchlas::device::DeviceBlasPolicy::Generic);
    });

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(y_auto(i), y_generic(i), 1e-5f) << "Mismatch at index " << i;
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
        batchlas::device::trmv(group, a_view, x_view, y_group_view, 0.75f, 0.25f, Uplo::Lower, Transpose::NoTrans, Diag::Unit);
    });
    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::trmv(item,
                               a_view,
                               x_view,
                               y_item_view,
                               0.75f,
                               0.25f,
                               Uplo::Lower,
                               Transpose::NoTrans,
                               Diag::Unit,
                               batchlas::device::DeviceBlasPolicy::Auto);
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
        batchlas::device::trmv(group, a_view, x_view, y_group_view, 1.25f, -0.25f, Uplo::Lower, Transpose::Trans, Diag::NonUnit);
    });
    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::trmv(item,
                               a_view,
                               x_view,
                               y_item_view,
                               1.25f,
                               -0.25f,
                               Uplo::Lower,
                               Transpose::Trans,
                               Diag::NonUnit,
                               batchlas::device::DeviceBlasPolicy::Auto);
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

    run_group_kernel(ctx, local_size, [=](auto group) {
        batchlas::device::symm(group, a_view, b_view, c_view, 1.5f, -0.25f, Side::Left, Uplo::Lower);
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

    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::symm(
            item, a_view, b_view, c_auto_view, 0.8f, 0.3f, Side::Right, Uplo::Upper, batchlas::device::DeviceBlasPolicy::Auto);
    });
    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
        batchlas::device::symm(item,
                               a_view,
                               b_view,
                               c_generic_view,
                               0.8f,
                               0.3f,
                               Side::Right,
                               Uplo::Upper,
                               batchlas::device::DeviceBlasPolicy::Generic);
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

            run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
                batchlas::device::gemm(item,
                                       a_view,
                                       b_view,
                                       c_auto_view,
                                       1.25f,
                                       -0.35f,
                                       trans_a,
                                       trans_b,
                                       batchlas::device::DeviceBlasPolicy::Auto);
            });
            run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
                batchlas::device::gemm(item,
                                       a_view,
                                       b_view,
                                       c_generic_view,
                                       1.25f,
                                       -0.35f,
                                       trans_a,
                                       trans_b,
                                       batchlas::device::DeviceBlasPolicy::Generic);
            });

            expect_matrix_matches_vector(c_auto.view(), expected);
            expect_matrix_matches_vector(c_generic.view(), expected);
        }
    }
}

TEST(DeviceBlasTest, TrmmNdItemMatchesReferenceAcrossTransforms) {
    Queue ctx(Device::default_device());
    const size_t local_size = device_test_work_group_size(ctx);

    for (auto side : {Side::Left, Side::Right}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            for (auto trans : {Transpose::NoTrans, Transpose::Trans}) {
                for (auto diag : {Diag::NonUnit, Diag::Unit}) {
                    Matrix<float, MatrixFormat::Dense> a(4, 4);
                    Matrix<float, MatrixFormat::Dense> b(side == Side::Left ? 4 : 3, side == Side::Left ? 3 : 4);
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

                    const auto expected = reference_trmm(a.view(), b.view(), c.view(), 0.9f, -0.2f, side, uplo, trans, diag);
                    auto a_view = a.view().kernel_view();
                    auto b_view = b.view().kernel_view();
                    auto c_view = c.view().kernel_view();

                    run_nd_item_kernel(ctx, local_size, [=](sycl::nd_item<1> item) {
                        batchlas::device::trmm(item,
                                               a_view,
                                               b_view,
                                               c_view,
                                               0.9f,
                                               -0.2f,
                                               side,
                                               uplo,
                                               trans,
                                               diag,
                                               batchlas::device::DeviceBlasPolicy::Auto);
                    });

                    expect_matrix_matches_vector(c.view(), expected);
                }
            }
        }
    }
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

    run_nd_item_kernel_3d(ctx, c.rows(), c.cols(), [=](sycl::nd_item<3> item) {
        batchlas::device::trmm(item,
                               a_view,
                               b_view,
                               c_view,
                               0.9f,
                               -0.2f,
                               Side::Left,
                               Uplo::Lower,
                               Transpose::NoTrans,
                               Diag::NonUnit,
                               batchlas::device::DeviceBlasPolicy::Auto);
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

    run_nd_item_kernel_3d(ctx, c.rows(), c.cols(), [=](sycl::nd_item<3> item) {
        batchlas::device::gemm(item,
                               a_view,
                               b_view,
                               c_view,
                               0.95f,
                               -0.2f,
                               Transpose::NoTrans,
                               Transpose::NoTrans,
                               batchlas::device::DeviceBlasPolicy::Auto);
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

    run_nd_item_kernel_3d(ctx, c.rows(), c.cols(), [=](sycl::nd_item<3> item) {
        batchlas::device::gemm(item,
                               a_view,
                               b_view,
                               c_view,
                               1.1f,
                               -0.25f,
                               Transpose::NoTrans,
                               Transpose::NoTrans,
                               batchlas::device::DeviceBlasPolicy::Auto);
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

    run_nd_item_kernel_3d(ctx, c.rows(), c.cols(), [=](sycl::nd_item<3> item) {
        batchlas::device::symm(
            item, a_view, b_view, c_view, 1.1f, -0.15f, Side::Left, Uplo::Lower, batchlas::device::DeviceBlasPolicy::Auto);
    });

    expect_matrix_matches_vector(c.view(), expected, 5e-4);
}
