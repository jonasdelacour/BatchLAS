#include <gtest/gtest.h>
#include <blas/matrix_handle_new.hh>
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include <vector>
#include <complex>

using namespace batchlas;

TEST(MatrixDenseTest, BasicConstructionAndFill) {
    constexpr int rows = 4;
    constexpr int cols = 3;
    constexpr int batch = 2;
    Matrix<float, MatrixFormat::Dense> mat(rows, cols, batch);
    EXPECT_EQ(mat.rows_, rows);
    EXPECT_EQ(mat.cols_, cols);
    EXPECT_EQ(mat.batch_size_, batch);
    EXPECT_EQ(mat.data().size(), rows * cols * batch);
    
    // Fill and check
    mat.fill(7.5f);
    for (auto v : mat.data()) {
        EXPECT_FLOAT_EQ(v, 7.5f);
    }
}

TEST(MatrixDenseTest, MoveConstructorAndAssignment) {
    Matrix<float, MatrixFormat::Dense> mat1(2, 2, 1);
    mat1.fill(3.14f);
    Matrix<float, MatrixFormat::Dense> mat2(std::move(mat1));
    EXPECT_EQ(mat2.rows_, 2);
    EXPECT_EQ(mat2.cols_, 2);
    EXPECT_EQ(mat2.batch_size_, 1);
    for (auto v : mat2.data()) {
        EXPECT_FLOAT_EQ(v, 3.14f);
    }
    // Move assignment
    Matrix<float, MatrixFormat::Dense> mat3(2, 2, 1);
    mat3 = std::move(mat2);
    for (auto v : mat3.data()) {
        EXPECT_FLOAT_EQ(v, 3.14f);
    }
}

TEST(MatrixDenseTest, StaticFactoryMethods) {
    constexpr int n = 5;
    constexpr int batch = 2;
    // Identity
    auto eye = Matrix<float, MatrixFormat::Dense>::Identity(n, batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float v = eye.data()[b * eye.stride() + j * eye.ld() + i];
                if (i == j) EXPECT_FLOAT_EQ(v, 1.0f);
                else EXPECT_FLOAT_EQ(v, 0.0f);
            }
        }
    }
    // Zeros
    auto zeros = Matrix<float, MatrixFormat::Dense>::Zeros(n, n, batch);
    for (auto v : zeros.data()) EXPECT_FLOAT_EQ(v, 0.0f);
    // Ones
    auto ones = Matrix<float, MatrixFormat::Dense>::Ones(n, n, batch);
    for (auto v : ones.data()) EXPECT_FLOAT_EQ(v, 1.0f);
    // Diagonal
    UnifiedVector<float> diag_vals(n);
    for (int i = 0; i < n; ++i) diag_vals[i] = float(i + 1);
    auto diag = Matrix<float, MatrixFormat::Dense>::Diagonal(diag_vals.to_span(), batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float v = diag.data()[b * diag.stride() + j * diag.ld() + i];
                if (i == j) EXPECT_FLOAT_EQ(v, float(i + 1));
                else EXPECT_FLOAT_EQ(v, 0.0f);
            }
        }
    }
    // Random (just check size and value range)
    auto rnd = Matrix<float, MatrixFormat::Dense>::Random(n, n, batch, 123);
    EXPECT_EQ(rnd.data().size(), n * n * batch);
    for (auto v : rnd.data()) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

// Additional tests for Matrix class
#include <complex>

TEST(MatrixDenseTest, DoubleAndComplexConstruction) {
    Matrix<double, MatrixFormat::Dense> dmat(3, 2, 1);
    dmat.fill(2.0);
    for (auto v : dmat.data()) EXPECT_DOUBLE_EQ(v, 2.0);

    Matrix<std::complex<float>, MatrixFormat::Dense> cmat(2, 2, 1);
    std::complex<float> val(1.0f, -1.0f);
    cmat.fill(val);
    for (auto v : cmat.data()) EXPECT_EQ(v, val);
}

TEST(MatrixDenseTest, DataAccessAndModify) {
    Matrix<float, MatrixFormat::Dense> mat(2, 2, 1);
    auto data = mat.data();
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;
    EXPECT_FLOAT_EQ(mat.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(mat.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(mat.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(mat.data()[3], 4.0f);
}

TEST(MatrixDenseTest, CopyFromView) {
    Matrix<float, MatrixFormat::Dense> src(2, 2, 1);
    src.fill(9.0f);
    Matrix<float, MatrixFormat::Dense> dst(2, 2, 1);
    dst.fill(0.0f);
    dst.copy_from(src.view());
    for (auto v : dst.data()) EXPECT_FLOAT_EQ(v, 9.0f);
}

TEST(MatrixDenseTest, SubmatrixViewThrowsForCSR) {
    Matrix<float, MatrixFormat::CSR> smat(2, 2, 2, 1);
    EXPECT_THROW(smat.view(1, 1, 1, 1), std::runtime_error);
}

TEST(MatrixDenseTest, ExceptionOnCopyFromMismatchedShape) {
    Matrix<float, MatrixFormat::Dense> a(2, 2, 1);
    Matrix<float, MatrixFormat::Dense> b(3, 2, 1);
    EXPECT_THROW(a.copy_from(b.view()), std::runtime_error);
}

// --- CSR Matrix tests ---
TEST(MatrixCSRTest, BasicConstructionAndFill) {
    constexpr int rows = 3, cols = 3, nnz = 4, batch = 2;
    Matrix<float, MatrixFormat::CSR> mat(rows, cols, nnz, batch);
    EXPECT_EQ(mat.rows_, rows);
    EXPECT_EQ(mat.cols_, cols);
    EXPECT_EQ(mat.nnz(), nnz);
    EXPECT_EQ(mat.batch_size_, batch);
    EXPECT_EQ(mat.data().size(), nnz * batch);
    EXPECT_EQ(mat.row_offsets().size(), (rows + 1) * batch);
    EXPECT_EQ(mat.col_indices().size(), nnz * batch);
    mat.fill(7.5f);
    for (auto v : mat.data()) EXPECT_FLOAT_EQ(v, 7.5f);
}

TEST(MatrixCSRTest, ConstructionFromData) {
    constexpr int rows = 2, cols = 3, nnz = 3, batch = 1;
    float values[nnz] = {1.0f, 2.0f, 3.0f};
    int row_offsets[rows + 1] = {0, 2, 3};
    int col_indices[nnz] = {0, 2, 1};
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, nnz, rows, cols, nnz, rows + 1, batch);
    EXPECT_EQ(mat.rows_, rows);
    EXPECT_EQ(mat.cols_, cols);
    EXPECT_EQ(mat.nnz(), nnz);
    for (int i = 0; i < nnz; ++i) EXPECT_FLOAT_EQ(mat.data()[i], values[i]);
    for (int i = 0; i < rows + 1; ++i) EXPECT_EQ(mat.row_offsets()[i], row_offsets[i]);
    for (int i = 0; i < nnz; ++i) EXPECT_EQ(mat.col_indices()[i], col_indices[i]);
}

TEST(MatrixCSRTest, CopyAndMoveSemantics) {
    constexpr int rows = 2, cols = 2, nnz = 2, batch = 1;
    float values[nnz] = {4.0f, 5.0f};
    int row_offsets[rows + 1] = {0, 1, 2};
    int col_indices[nnz] = {0, 1};
    Matrix<float, MatrixFormat::CSR> mat1(values, row_offsets, col_indices, nnz, rows, cols, nnz, rows + 1, batch);
    Matrix<float, MatrixFormat::CSR> mat2(std::move(mat1));
    for (int i = 0; i < nnz; ++i) EXPECT_FLOAT_EQ(mat2.data()[i], values[i]);
    Matrix<float, MatrixFormat::CSR> mat3(rows, cols, nnz, batch);
    mat3 = std::move(mat2);
    for (int i = 0; i < nnz; ++i) EXPECT_FLOAT_EQ(mat3.data()[i], values[i]);
}

TEST(MatrixCSRTest, CopyFromView) {
    constexpr int rows = 2, cols = 2, nnz = 2, batch = 1;
    float values[nnz] = {8.0f, 9.0f};
    int row_offsets[rows + 1] = {0, 1, 2};
    int col_indices[nnz] = {0, 1};
    Matrix<float, MatrixFormat::CSR> src(values, row_offsets, col_indices, nnz, rows, cols, nnz, rows + 1, batch);
    Matrix<float, MatrixFormat::CSR> dst(rows, cols, nnz, batch);
    dst.copy_from(src.view());
    for (int i = 0; i < nnz; ++i) EXPECT_FLOAT_EQ(dst.data()[i], values[i]);
    for (int i = 0; i < rows + 1; ++i) EXPECT_EQ(dst.row_offsets()[i], row_offsets[i]);
    for (int i = 0; i < nnz; ++i) EXPECT_EQ(dst.col_indices()[i], col_indices[i]);
}
 
TEST(MatrixCSRTest, ExceptionOnCopyFromMismatchedShape) {
    Matrix<float, MatrixFormat::CSR> a(2, 2, 2, 1);
    Matrix<float, MatrixFormat::CSR> b(3, 2, 2, 1);
    EXPECT_THROW(a.copy_from(b.view()), std::runtime_error);
}
