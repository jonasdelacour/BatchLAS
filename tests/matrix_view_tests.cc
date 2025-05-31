// filepath: /home/jonaslacour/BatchLAS/tests/matrix_view_tests.cc
#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include <complex>

using namespace batchlas;

TEST(MatrixViewTest, ConstructFromMatrix) {
    constexpr int rows = 3, cols = 4, batch = 2;
    Matrix<float, MatrixFormat::Dense> mat(rows, cols, batch);
    mat.fill(2.5f);
    auto view = mat.view();
    EXPECT_EQ(view.rows_, rows);
    EXPECT_EQ(view.cols_, cols);
    EXPECT_EQ(view.batch_size_, batch);
    for (auto v : view.data()) EXPECT_FLOAT_EQ(v, 2.5f);
}

TEST(MatrixViewTest, SubmatrixView) {
    constexpr int n = 4;
    Matrix<float, MatrixFormat::Dense> mat(n, n, 1);
    for (int i = 0; i < n * n; ++i) mat.data()[i] = float(i);
    auto sub = mat.view(n, n / 2);
    EXPECT_EQ(sub.rows_, n);
    EXPECT_EQ(sub.cols_, n / 2);
    for (int j = 0; j < n / 2; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_EQ(sub.at<MatrixFormat::Dense>(i, j), mat.data()[j * n + i]);
}

TEST(MatrixViewTest, BatchItemView) {
    constexpr int rows = 2, cols = 2, batch = 3;
    Matrix<float, MatrixFormat::Dense> mat(rows, cols, batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                mat.data()[b * rows * cols + j * rows + i] = float(10 * b + j * rows + i);
    auto view = mat.view();
    for (int b = 0; b < batch; ++b) {
        auto item = view.batch_item(b);
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                EXPECT_EQ(item.at<MatrixFormat::Dense>(i, j), float(10 * b + j * rows + i));
    }
}

TEST(MatrixViewTest, ElementAccessAndModify) {
    Matrix<float, MatrixFormat::Dense> mat(2, 2, 1);
    auto view = mat.view();
    view.at<MatrixFormat::Dense>(0, 0) = 1.0f;
    view.at<MatrixFormat::Dense>(1, 0) = 2.0f;
    view.at<MatrixFormat::Dense>(0, 1) = 3.0f;
    view.at<MatrixFormat::Dense>(1, 1) = 4.0f;
    EXPECT_FLOAT_EQ(mat.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(mat.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(mat.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(mat.data()[3], 4.0f);
}

TEST(MatrixViewTest, OutOfBoundsThrows) {
    Matrix<float, MatrixFormat::Dense> mat(2, 2, 1);
    auto view = mat.view();
    EXPECT_THROW(view.at<MatrixFormat::Dense>(2, 0), std::out_of_range);
    EXPECT_THROW(view.at<MatrixFormat::Dense>(0, 2), std::out_of_range);
    EXPECT_THROW(view.at<MatrixFormat::Dense>(0, 0, 1), std::out_of_range);
}

TEST(MatrixViewTest, CopyAndMoveSemantics) {
    Matrix<float, MatrixFormat::Dense> mat(2, 2, 1);
    mat.fill(5.0f);
    auto view1 = mat.view();
    auto view2 = view1;
    auto view3 = std::move(view2);
    for (auto v : view3.data()) EXPECT_FLOAT_EQ(v, 5.0f);
}

// --- CSR MatrixView tests ---
TEST(MatrixViewTest, ConstructFromCSRMatrix) {
    constexpr int rows = 3, cols = 3, nnz = 5, batch = 2;
    // Example CSR data for two batches
    float values[2 * nnz] = {1, 2, 3, 4, 5, 10, 20, 30, 40, 50};
    int row_offsets[2 * (rows + 1)] = {0, 2, 3, 5, 0, 2, 3, 5};
    int col_indices[2 * nnz] = {0, 2, 1, 0, 2, 0, 2, 1, 0, 2};
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, nnz, rows, cols, nnz, rows + 1, batch);
    auto view = mat.view();
    EXPECT_EQ(view.rows_, rows);
    EXPECT_EQ(view.cols_, cols);
    EXPECT_EQ(view.batch_size_, batch);
    EXPECT_EQ(view.nnz(), nnz);
    for (int i = 0; i < 2 * nnz; ++i) EXPECT_FLOAT_EQ(view.data()[i], values[i]);
    for (int i = 0; i < 2 * (rows + 1); ++i) EXPECT_EQ(view.row_offsets()[i], row_offsets[i]);
    for (int i = 0; i < 2 * nnz; ++i) EXPECT_EQ(view.col_indices()[i], col_indices[i]);
}

TEST(MatrixViewTest, CSRBatchItemView) {
    constexpr int rows = 2, cols = 3, nnz = 3, batch = 2;
    float values[2 * nnz] = {1, 2, 3, 10, 20, 30};
    int row_offsets[2 * (rows + 1)] = {0, 1, 3, 0, 2, 3};
    int col_indices[2 * nnz] = {0, 2, 1, 1, 0, 2};
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, nnz, rows, cols, nnz, rows + 1, batch);
    auto view = mat.view();
    for (int b = 0; b < batch; ++b) {
        auto item = view.batch_item(b);
        for (int i = 0; i < nnz; ++i)
            EXPECT_FLOAT_EQ(item.data()[i], values[b * nnz + i]);
        for (int i = 0; i < rows + 1; ++i)
            EXPECT_EQ(item.row_offsets()[i], row_offsets[b * (rows + 1) + i]);
        for (int i = 0; i < nnz; ++i)
            EXPECT_EQ(item.col_indices()[i], col_indices[b * nnz + i]);
    }
}

TEST(MatrixViewTest, CSRAccessors) {
    constexpr int rows = 2, cols = 2, nnz = 2;
    float values[nnz] = {1.5f, 2.5f};
    int row_offsets[rows + 1] = {0, 1, 2};
    int col_indices[nnz] = {0, 1};
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, nnz, rows, cols, nnz, rows + 1, 1);
    auto view = mat.view();
    EXPECT_EQ(view.nnz(), nnz);
    EXPECT_EQ(view.matrix_stride(), nnz);
    EXPECT_EQ(view.offset_stride(), rows + 1);
    EXPECT_EQ(view.data().size(), nnz);
    EXPECT_EQ(view.row_offsets().size(), rows + 1);
    EXPECT_EQ(view.col_indices().size(), nnz);
}

TEST(MatrixViewTest, CSRSubmatrixViewThrows) {
    constexpr int rows = 2, cols = 2, nnz = 2;
    float values[nnz] = {1, 2};
    int row_offsets[rows + 1] = {0, 1, 2};
    int col_indices[nnz] = {0, 1};
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, nnz, rows, cols, nnz, rows + 1, 1);
    EXPECT_THROW(mat.view(1, 1), std::runtime_error);
}
