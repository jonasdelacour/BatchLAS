#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <algorithm>
#include <vector>
#include <complex>
#include <string>

using namespace batchlas;

TEST (MatrixDenseTest, StridedConstruction) {
    constexpr int rows = 3;
    constexpr int cols = 2;
    constexpr int batch = 2;
    constexpr int ld = 4;      // leading dimension greater than rows
    constexpr int stride = 10; // stride greater than ld * cols

    Matrix<float, MatrixFormat::Dense> mat(rows, cols, batch, ld, stride);
    EXPECT_EQ(mat.rows_, rows);
    EXPECT_EQ(mat.cols_, cols);
    EXPECT_EQ(mat.batch_size_, batch);
    EXPECT_EQ(mat.ld(), ld);
    EXPECT_EQ(mat.stride(), stride);
    EXPECT_EQ(mat.data().size(), stride * batch);

    //Ensure that the underlying data is zero'd first:
    for (auto v : mat.data()) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }

    //Now use matrix fill function (shouldn't fill the padding areas)
    mat.fill(1.0f);
    for (int k = 0; k < batch; ++k) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                mat(i, j, k) = 1.0f;
                EXPECT_FLOAT_EQ(mat.data()[k * stride + j * ld + i], 1.0f);
            }
        }
    }

    for (int k = 0; k < batch; ++k) {
        for (int j = 0; j < cols; ++j) {
            for (int i = rows; i < ld; ++i) {
                // Padding area in leading dimension should remain zero
                EXPECT_FLOAT_EQ(mat.data()[k * stride + j * ld + i], 0.0f);
            }
        }
    }

    for (int k = 0; k < batch; ++k) {
        for (int j = cols; j < stride / ld; ++j) {
            for (int i = 0; i < ld; ++i) {
                // Padding area in stride should remain zero
                EXPECT_FLOAT_EQ(mat.data()[k * stride + j * ld + i], 0.0f);
            }
        }
    }
}


TEST (MatrixDenseTest, StridedIdentity) {
    constexpr int n = 4;
    constexpr int batch = 2;
    constexpr int ld = 6;
    constexpr int stride = 30;

    Queue ctx;

    Matrix<float, MatrixFormat::Dense> mat(n, n, batch, ld, stride);
    auto view = mat.view();
    view.fill_identity(ctx, 1.0f).wait();

    for (int k = 0; k < batch; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                float v = mat.data()[k * stride + j * ld + i];
                if (i == j) EXPECT_FLOAT_EQ(v, 1.0f);
                else EXPECT_FLOAT_EQ(v, 0.0f);
            }
        }
    }
}

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
    auto rnd = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 123);
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
    Matrix<float, MatrixFormat::CSR> smat(2, 2, NonZeros{2}, 1);
    EXPECT_THROW(smat.view(1, 1, 1, 1), std::runtime_error);
}

TEST(MatrixDenseTest, ExceptionOnCopyFromMismatchedShape) {
    Matrix<float, MatrixFormat::Dense> a(2, 2, 1);
    Matrix<float, MatrixFormat::Dense> b(3, 2, 1);
    EXPECT_THROW(a.copy_from(b.view()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Matrix(const T* data, rows, cols, ld, stride, batch_size)
//
// (ld, stride) describe the *source* buffer: element (i, j, b) lives at
// data[b * stride + j * ld + i]. The copy keeps the caller's leading dimension and
// packs the batch items (stride() == ld * cols). The invariant every test below
// checks is that view()(i, j, b) returns what the caller had at their own (i, j, b) -
// this constructor used to size its allocation from rows * cols while advertising
// ld_ = ld, and to test the raw `stride` argument for its fast path, so a batched call
// with the defaulted stride silently copied batch item 0 into every item.
// ---------------------------------------------------------------------------

TEST(MatrixDenseTest, ConstructionFromDataPacked) {
    constexpr int rows = 3, cols = 2;
    // Column-major, no padding
    float src[rows * cols] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Matrix<float, MatrixFormat::Dense> mat(src, rows, cols, rows);
    EXPECT_EQ(mat.rows_, rows);
    EXPECT_EQ(mat.cols_, cols);
    EXPECT_EQ(mat.batch_size_, 1);
    EXPECT_EQ(mat.ld(), rows);
    EXPECT_EQ(mat.stride(), rows * cols);
    EXPECT_EQ(mat.data().size(), rows * cols);

    auto view = mat.view();
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            EXPECT_FLOAT_EQ(view(i, j, 0), src[j * rows + i]);
        }
    }
}

TEST(MatrixDenseTest, ConstructionFromDataPaddedLeadingDimension) {
    constexpr int rows = 3, cols = 2, ld = 5;
    // Padding rows carry a sentinel that must not show up in the copy
    std::vector<float> src(ld * cols, -1.0f);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            src[j * ld + i] = static_cast<float>(j * 10 + i);
        }
    }

    Matrix<float, MatrixFormat::Dense> mat(src.data(), rows, cols, ld);
    EXPECT_EQ(mat.ld(), ld);
    EXPECT_EQ(mat.stride(), ld * cols);
    EXPECT_EQ(mat.data().size(), ld * cols);

    auto view = mat.view();
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            EXPECT_FLOAT_EQ(view(i, j, 0), src[j * ld + i]);
        }
        for (int i = rows; i < ld; ++i) {
            // Source padding is not copied; the destination padding is zeroed
            EXPECT_FLOAT_EQ(mat.data()[j * ld + i], 0.0f);
        }
    }
}

TEST(MatrixDenseTest, ConstructionFromDataExplicitStride) {
    constexpr int rows = 2, cols = 2, ld = 3, stride = 10, batch = 2;
    // stride > ld * cols: the batch items are separated by a gap in the source
    std::vector<float> src(stride * batch, -1.0f);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                src[b * stride + j * ld + i] = static_cast<float>(b * 100 + j * 10 + i);
            }
        }
    }

    Matrix<float, MatrixFormat::Dense> mat(src.data(), rows, cols, ld, stride, batch);
    EXPECT_EQ(mat.batch_size_, batch);
    EXPECT_EQ(mat.ld(), ld);
    EXPECT_EQ(mat.stride(), ld * cols);  // the source gap is not carried into the copy
    EXPECT_EQ(mat.data().size(), ld * cols * batch);

    auto view = mat.view();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                EXPECT_FLOAT_EQ(view(i, j, b), src[b * stride + j * ld + i]);
            }
        }
    }
}

TEST(MatrixDenseTest, ConstructionFromDataDefaultStrideBatched) {
    constexpr int rows = 2, cols = 3, batch = 3;
    // Packed batched source with the defaulted stride - every item must survive
    std::vector<float> src(rows * cols * batch);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                src[b * rows * cols + j * rows + i] = static_cast<float>(b * 100 + j * 10 + i);
            }
        }
    }

    Matrix<float, MatrixFormat::Dense> mat(src.data(), rows, cols, rows, 0, batch);
    EXPECT_EQ(mat.ld(), rows);
    EXPECT_EQ(mat.stride(), rows * cols);
    EXPECT_EQ(mat.data().size(), rows * cols * batch);

    auto view = mat.view();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                EXPECT_FLOAT_EQ(view(i, j, b), src[b * rows * cols + j * rows + i]);
            }
        }
    }
}

TEST(MatrixDenseTest, ConstructionFromDataDefaultStridePaddedBatched) {
    constexpr int rows = 2, cols = 2, ld = 4, batch = 3;
    // Defaulted stride with ld > rows: source items are ld * cols apart
    std::vector<std::complex<double>> src(ld * cols * batch, std::complex<double>(-1.0, -1.0));
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                src[b * ld * cols + j * ld + i] = std::complex<double>(b * 100 + j * 10 + i, -b);
            }
        }
    }

    Matrix<std::complex<double>, MatrixFormat::Dense> mat(src.data(), rows, cols, ld, 0, batch);
    EXPECT_EQ(mat.ld(), ld);
    EXPECT_EQ(mat.stride(), ld * cols);
    EXPECT_EQ(mat.data().size(), ld * cols * batch);

    auto view = mat.view();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                EXPECT_EQ(view(i, j, b), src[b * ld * cols + j * ld + i]);
            }
        }
    }
}

// clone() allocated a packed rows * cols * batch buffer and then flat-copied
// data_, whose length is stride_ * batch_size_. For any padded matrix the copy
// was longer than the destination -- a heap write past the end -- and the
// clone's ld_/stride_ then described a buffer that was never allocated. The
// size assertions below fail deterministically without the fix; the value
// assertions cover the reads that followed.
TEST(MatrixDenseTest, ClonePreservesPaddedLeadingDimension) {
    constexpr int rows = 4, cols = 3, ld = 8, batch = 2;
    std::vector<float> src(ld * cols * batch, -1.0f);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                src[b * ld * cols + j * ld + i] = static_cast<float>(b * 100 + j * 10 + i);
            }
        }
    }

    Matrix<float, MatrixFormat::Dense> mat(src.data(), rows, cols, ld, 0, batch);
    ASSERT_EQ(mat.data().size(), static_cast<size_t>(ld) * cols * batch);

    auto copy = mat.clone();
    EXPECT_EQ(copy.ld(), ld);
    EXPECT_EQ(copy.stride(), ld * cols);
    // The clone must own as many elements as it claims, or the copy above ran past it.
    EXPECT_EQ(copy.data().size(), mat.data().size());

    auto view = copy.view();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                EXPECT_EQ(view(i, j, b), static_cast<float>(b * 100 + j * 10 + i));
            }
        }
    }
}

// Same defect reached through the allocating constructor, which additionally
// allows a stride with gaps between batch items (stride > ld * cols).
TEST(MatrixDenseTest, ClonePreservesGappedBatchStride) {
    constexpr int rows = 3, cols = 2, ld = 5, stride = 16, batch = 3;
    Matrix<float, MatrixFormat::Dense> mat(rows, cols, batch, ld, stride);
    ASSERT_EQ(mat.data().size(), static_cast<size_t>(stride) * batch);

    auto fill = mat.view();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                fill(i, j, b) = static_cast<float>(b * 100 + j * 10 + i);
            }
        }
    }

    auto copy = mat.clone();
    EXPECT_EQ(copy.ld(), ld);
    EXPECT_EQ(copy.stride(), stride);
    EXPECT_EQ(copy.data().size(), mat.data().size());

    auto view = copy.view();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                EXPECT_EQ(view(i, j, b), static_cast<float>(b * 100 + j * 10 + i));
            }
        }
    }
}

// A default-initialised MatrixView (as opposed to a value-initialised one) left
// rows_/cols_/batch_size_ indeterminate, which queue-dispatch.hh's USM check
// reads through addresses_no_elements() to decide whether a null data pointer is
// legal. They are zero-initialised now, so an empty view reports an empty shape.
TEST(MatrixDenseTest, DefaultInitialisedViewHasAZeroShape) {
    MatrixView<float, MatrixFormat::Dense> view;
    EXPECT_EQ(view.rows(), 0);
    EXPECT_EQ(view.cols(), 0);
    EXPECT_EQ(view.batch_size(), 0);
}

TEST(MatrixDenseTest, ConstructionFromDataThrowsOnBadShape) {
    using DenseMatrix = Matrix<float, MatrixFormat::Dense>;
    std::vector<float> src(16, 1.0f);

    EXPECT_THROW(DenseMatrix(nullptr, 2, 2, 2), std::invalid_argument);
    EXPECT_THROW(DenseMatrix(src.data(), 0, 2, 2), std::invalid_argument);
    EXPECT_THROW(DenseMatrix(src.data(), 2, 0, 2), std::invalid_argument);
    EXPECT_THROW(DenseMatrix(src.data(), 2, 2, 1), std::invalid_argument);        // ld < rows
    EXPECT_THROW(DenseMatrix(src.data(), 2, 2, -1), std::invalid_argument);       // negative ld
    EXPECT_THROW(DenseMatrix(src.data(), 2, 2, 2, -1), std::invalid_argument);    // negative stride
    EXPECT_THROW(DenseMatrix(src.data(), 2, 2, 2, 0, 0), std::invalid_argument);  // empty batch
    EXPECT_THROW(DenseMatrix(src.data(), 2, 2, 2, 3, 2), std::invalid_argument);  // stride < ld * cols

    // ld == 0 falls back to rows, like the sizing constructor
    DenseMatrix defaulted(src.data(), 2, 2, 0);
    EXPECT_EQ(defaulted.ld(), 2);
    EXPECT_EQ(defaulted.stride(), 4);
}

TEST(MatrixDenseTest, ConstructionFromSpanChecksLength) {
    using DenseMatrix = Matrix<float, MatrixFormat::Dense>;
    constexpr int rows = 2, cols = 3, batch = 2;
    std::vector<float> src(rows * cols * batch);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i);

    // Mutable span overload
    DenseMatrix mat(Span<float>(src.data(), src.size()), rows, cols, rows, 0, batch);
    EXPECT_EQ(mat.data().size(), rows * cols * batch);
    auto view = mat.view();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                EXPECT_FLOAT_EQ(view(i, j, b), src[b * rows * cols + j * rows + i]);
            }
        }
    }

    // Const span overload
    DenseMatrix const_mat(Span<const float>(src.data(), src.size()), rows, cols, rows, 0, batch);
    EXPECT_FLOAT_EQ(const_mat.view()(rows - 1, cols - 1, batch - 1), src.back());

    // A span that is too short for the requested shape is rejected
    EXPECT_THROW((DenseMatrix(Span<const float>(src.data(), src.size() - 1), rows, cols, rows, 0, batch)),
                 std::invalid_argument);
}

// --- CSR Matrix tests ---

// ---------------------------------------------------------------------------
// The CSR non-zero count is a NonZeros, not an int.
//
// Dense is Matrix(rows, cols, batch_size, ld, stride) and CSR is
// Matrix(rows, cols, NonZeros{nnz}, batch_size). Before NonZeros existed both
// took an int in the third position, so the dense spelling compiled for CSR and
// silently allocated batch_size non-zeros. These asserts pin that it cannot
// come back: the all-int spellings must not be viable at all.
// ---------------------------------------------------------------------------
using CsrF = Matrix<float, MatrixFormat::CSR>;

static_assert(!std::is_constructible_v<CsrF, int, int, int, int>,
              "CSR must not be constructible from (rows, cols, nnz, batch) as plain ints");
static_assert(!std::is_constructible_v<CsrF, int, int, int>,
              "CSR must not be constructible from the dense (rows, cols, batch) spelling");
static_assert(std::is_constructible_v<CsrF, int, int, NonZeros, int>,
              "CSR must be constructible from (rows, cols, NonZeros, batch)");
static_assert(std::is_constructible_v<CsrF, int, int, NonZeros>,
              "batch_size must keep its default");

// The from-data constructor: shape before the count, and the count strongly typed.
static_assert(!std::is_constructible_v<CsrF, const float*, const int*, const int*,
                                       int, int, int, int, int, int>,
              "the old (nnz, rows, cols, ...) CSR from-data order must not compile");
static_assert(std::is_constructible_v<CsrF, const float*, const int*, const int*,
                                      int, int, NonZeros, int, int, int>,
              "CSR from-data must take (values, ro, ci, rows, cols, NonZeros, strides, batch)");

using CsrViewF = MatrixView<float, MatrixFormat::CSR>;
static_assert(!std::is_constructible_v<CsrViewF, float*, int*, int*,
                                       int, int, int, int, int, int>,
              "the old (nnz, rows, cols, ...) CSR view order must not compile");
static_assert(std::is_constructible_v<CsrViewF, float*, int*, int*,
                                      int, int, NonZeros, int, int, int>,
              "CSR view must take (values, ro, ci, rows, cols, NonZeros, strides, batch)");

// NonZeros itself: explicit in, no decay out, usable at compile time.
static_assert(!std::is_convertible_v<int, NonZeros>, "NonZeros must be explicit");
static_assert(!std::is_convertible_v<NonZeros, int>, "NonZeros must not decay to int");
static_assert(std::is_trivially_copyable_v<NonZeros>);
static_assert(NonZeros{7}.value == 7);

TEST(MatrixCSRTest, NonZerosSpellingAllocatesAndRoundTrips) {
    // rows/cols/nnz/batch deliberately all distinct, so a swapped argument would
    // change an observable number rather than land on an equal one.
    constexpr int rows = 4, cols = 5, nnz = 6, batch = 3;

    Matrix<float, MatrixFormat::CSR> mat(rows, cols, NonZeros{nnz}, batch);
    EXPECT_EQ(mat.rows(), rows);
    EXPECT_EQ(mat.cols(), cols);
    EXPECT_EQ(mat.nnz(), nnz);
    EXPECT_EQ(mat.batch_size(), batch);
    EXPECT_EQ(mat.data().size(), static_cast<size_t>(nnz) * batch);
    EXPECT_EQ(mat.col_indices().size(), static_cast<size_t>(nnz) * batch);
    EXPECT_EQ(mat.row_offsets().size(), static_cast<size_t>(rows + 1) * batch);

    // Same shape through the from-data constructor: buffers, shape, count, strides, batch.
    std::vector<float> values(static_cast<size_t>(nnz) * batch);
    std::vector<int> col_indices(static_cast<size_t>(nnz) * batch);
    std::vector<int> row_offsets(static_cast<size_t>(rows + 1) * batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < nnz; ++i) {
            values[static_cast<size_t>(b) * nnz + i] = static_cast<float>(100 * b + i);
            col_indices[static_cast<size_t>(b) * nnz + i] = i % cols;
        }
        // 6 non-zeros over 4 rows: 2, 2, 1, 1.
        const int ro[rows + 1] = {0, 2, 4, 5, 6};
        for (int i = 0; i < rows + 1; ++i)
            row_offsets[static_cast<size_t>(b) * (rows + 1) + i] = ro[i];
    }

    Matrix<float, MatrixFormat::CSR> from_data(values.data(), row_offsets.data(),
                                               col_indices.data(), rows, cols,
                                               NonZeros{nnz}, nnz, rows + 1, batch);
    EXPECT_EQ(from_data.rows(), rows);
    EXPECT_EQ(from_data.cols(), cols);
    EXPECT_EQ(from_data.nnz(), nnz);
    EXPECT_EQ(from_data.batch_size(), batch);
    EXPECT_EQ(from_data.matrix_stride(), nnz);
    EXPECT_EQ(from_data.offset_stride(), rows + 1);
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_FLOAT_EQ(from_data.data()[i], values[i]);
    for (size_t i = 0; i < col_indices.size(); ++i)
        EXPECT_EQ(from_data.col_indices()[i], col_indices[i]);
    for (size_t i = 0; i < row_offsets.size(); ++i)
        EXPECT_EQ(from_data.row_offsets()[i], row_offsets[i]);

    // clone() must carry the full allocation, not just nnz worth of it.
    auto copy = from_data.clone();
    EXPECT_EQ(copy.nnz(), nnz);
    EXPECT_EQ(copy.data().size(), from_data.data().size());
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_FLOAT_EQ(copy.data()[i], values[i]);

    // A view over one batch item sees that item's values.
    auto item = from_data.view().batch_item(1);
    EXPECT_EQ(item.batch_size(), 1);
    EXPECT_EQ(item.nnz(), nnz);
    for (int i = 0; i < nnz; ++i)
        EXPECT_FLOAT_EQ(item.data()[i], values[static_cast<size_t>(nnz) + i]);
}

TEST(MatrixCSRTest, BasicConstructionAndFill) {
    constexpr int rows = 3, cols = 3, nnz = 4, batch = 2;
    Matrix<float, MatrixFormat::CSR> mat(rows, cols, NonZeros{nnz}, batch);
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
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, rows, cols, NonZeros{nnz}, nnz, rows + 1, batch);
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
    Matrix<float, MatrixFormat::CSR> mat1(values, row_offsets, col_indices, rows, cols, NonZeros{nnz}, nnz, rows + 1, batch);
    Matrix<float, MatrixFormat::CSR> mat2(std::move(mat1));
    for (int i = 0; i < nnz; ++i) EXPECT_FLOAT_EQ(mat2.data()[i], values[i]);
    Matrix<float, MatrixFormat::CSR> mat3(rows, cols, NonZeros{nnz}, batch);
    mat3 = std::move(mat2);
    for (int i = 0; i < nnz; ++i) EXPECT_FLOAT_EQ(mat3.data()[i], values[i]);
}

TEST(MatrixCSRTest, CopyFromView) {
    constexpr int rows = 2, cols = 2, nnz = 2, batch = 1;
    float values[nnz] = {8.0f, 9.0f};
    int row_offsets[rows + 1] = {0, 1, 2};
    int col_indices[nnz] = {0, 1};
    Matrix<float, MatrixFormat::CSR> src(values, row_offsets, col_indices, rows, cols, NonZeros{nnz}, nnz, rows + 1, batch);
    Matrix<float, MatrixFormat::CSR> dst(rows, cols, NonZeros{nnz}, batch);
    dst.copy_from(src.view());
    for (int i = 0; i < nnz; ++i) EXPECT_FLOAT_EQ(dst.data()[i], values[i]);
    for (int i = 0; i < rows + 1; ++i) EXPECT_EQ(dst.row_offsets()[i], row_offsets[i]);
    for (int i = 0; i < nnz; ++i) EXPECT_EQ(dst.col_indices()[i], col_indices[i]);
}
 
TEST(MatrixCSRTest, ExceptionOnCopyFromMismatchedShape) {
    Matrix<float, MatrixFormat::CSR> a(2, 2, NonZeros{2}, 1);
    Matrix<float, MatrixFormat::CSR> b(3, 2, NonZeros{2}, 1);
    EXPECT_THROW(a.copy_from(b.view()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// triangularize
//
// `uplo` names the triangle to KEEP. These tests exist because the function
// shipped with that inverted: it named the flat-index quotient the row while
// addressing it as the column, so under column-major storage
// triangularize(Uplo::Upper) left a LOWER triangular matrix. It had no test
// and, in-tree, no caller, so nothing caught it -- PR #66 found it only
// because extracting R from a geqrf result silently handed back the
// Householder reflectors instead, which fabricated a 1.7x speedup that was
// not real.
//
// Each entry is seeded with a value encoding its own (row, col), so a
// transposed or mis-strided write shows up as a wrong *position*, not merely a
// wrong number.
// ---------------------------------------------------------------------------
namespace {

inline float tri_marker(int row, int col) {
    return 100.0f * static_cast<float>(row + 1) + static_cast<float>(col + 1);
}

}  // namespace

TEST(MatrixTriangularize, UpperKeepsUpperTriangle) {
    constexpr int n = 5;
    constexpr int batch = 2;
    constexpr int ld = 7;      // deliberately > rows
    constexpr int stride = 40; // deliberately > ld * cols

    Queue ctx;
    Matrix<float, MatrixFormat::Dense> mat(n, n, batch, ld, stride);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                mat.data()[b * stride + j * ld + i] = tri_marker(i, j);

    mat.view().triangularize(ctx, Uplo::Upper, Diag::NonUnit).wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const float v = mat.data()[b * stride + j * ld + i];
                if (i <= j) {
                    EXPECT_FLOAT_EQ(v, tri_marker(i, j)) << "kept (" << i << "," << j << ")";
                } else {
                    EXPECT_FLOAT_EQ(v, 0.0f) << "strict lower (" << i << "," << j << ")";
                }
            }
        }
    }
}

TEST(MatrixTriangularize, LowerKeepsLowerTriangle) {
    constexpr int n = 5;
    constexpr int batch = 2;

    Queue ctx;
    Matrix<float, MatrixFormat::Dense> mat(n, n, batch);
    const int ld = mat.view().ld();
    const int stride = mat.view().stride();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                mat.data()[b * stride + j * ld + i] = tri_marker(i, j);

    mat.view().triangularize(ctx, Uplo::Lower, Diag::NonUnit).wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const float v = mat.data()[b * stride + j * ld + i];
                if (i >= j) {
                    EXPECT_FLOAT_EQ(v, tri_marker(i, j)) << "kept (" << i << "," << j << ")";
                } else {
                    EXPECT_FLOAT_EQ(v, 0.0f) << "strict upper (" << i << "," << j << ")";
                }
            }
        }
    }
}

TEST(MatrixTriangularize, UnitDiagonalOverwritesDiagonal) {
    constexpr int n = 4;

    Queue ctx;
    Matrix<float, MatrixFormat::Dense> mat(n, n, 1);
    const int ld = mat.view().ld();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            mat.data()[j * ld + i] = tri_marker(i, j);

    mat.view().triangularize(ctx, Uplo::Upper, Diag::Unit).wait();

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const float v = mat.data()[j * ld + i];
            if (i == j)     EXPECT_FLOAT_EQ(v, 1.0f);
            else if (i < j) EXPECT_FLOAT_EQ(v, tri_marker(i, j));
            else            EXPECT_FLOAT_EQ(v, 0.0f);
        }
    }
}

// The element count used to come from data_.size() and the index decode from
// rows_ alone, so a non-square view decoded its own coordinates wrongly.
TEST(MatrixTriangularize, NonSquareTallAndWide) {
    Queue ctx;

    const std::pair<int, int> shapes[] = {{6, 3}, {3, 6}};
    for (const auto& shape : shapes) {
        const int rows = shape.first;
        const int cols = shape.second;

        Matrix<float, MatrixFormat::Dense> mat(rows, cols, 1);
        const int ld = mat.view().ld();
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                mat.data()[j * ld + i] = tri_marker(i, j);

        mat.view().triangularize(ctx, Uplo::Upper, Diag::NonUnit).wait();

        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                const float v = mat.data()[j * ld + i];
                if (i <= j) {
                    EXPECT_FLOAT_EQ(v, tri_marker(i, j))
                        << rows << "x" << cols << " kept (" << i << "," << j << ")";
                } else {
                    EXPECT_FLOAT_EQ(v, 0.0f)
                        << rows << "x" << cols << " zeroed (" << i << "," << j << ")";
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// to_column_major() / to_row_major()
//
// Both used to index the source as `b * stride + i * cols + j` (resp.
// `b * stride + j * rows + i`): the leading dimension was never read, and the
// *source* stride was reused for the destination. So a matrix with a padded ld
// read the wrong elements, and any batched matrix whose stride is not
// rows * cols wrote past the end of the destination buffer -- both silently.
//
// The row pitch used to be *inferred* from ld (`ld > rows ? ld : cols`). That
// inference is gone: ld means the distance between successive columns, so for a
// matrix with rows > cols every legal row pitch in (cols, rows] was silently read
// as "packed", and an inferred pitch above cols made the read run past the end of
// an allocation that is sized ld * cols. to_column_major now takes the pitch, and
// defaults it to cols -- packed, the only row-major layout the class can promise
// it holds, and the one to_row_major produces. The source's batch stride is still
// its own stride(); the destination is packed in the target major.
// ---------------------------------------------------------------------------

namespace {
float major_marker(int i, int j, int b) {
    return static_cast<float>(b * 1000 + i * 10 + j) + 0.5f;
}
}  // namespace

TEST(MatrixMajorConversion, ToColumnMajorHonoursAnExplicitRowPitch) {
    // A padded row-major buffer: the pitch is stated, not guessed.
    constexpr int rows = 3, cols = 4, ld = 6, pitch = 6;
    Matrix<float, MatrixFormat::Dense> row_major(rows, cols, 1, ld);
    auto raw = row_major.data();
    std::fill(raw.begin(), raw.end(), -7.0f);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            raw[i * pitch + j] = major_marker(i, j, 0);  // row-major, row pitch 6

    auto col_major = row_major.to_column_major(pitch);
    EXPECT_EQ(col_major.ld(), rows);
    EXPECT_EQ(col_major.stride(), rows * cols);
    ASSERT_EQ(col_major.data().size(), static_cast<size_t>(rows * cols));

    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            EXPECT_FLOAT_EQ(col_major.data()[j * rows + i], major_marker(i, j, 0))
                << "column-major (" << i << "," << j << ")";
}

TEST(MatrixMajorConversion, ToColumnMajorHonoursExplicitPitchAndStrideBatched) {
    constexpr int rows = 3, cols = 4, batch = 2, ld = 6, stride = 40;  // stride > ld * cols
    constexpr int pitch = 6;
    Matrix<float, MatrixFormat::Dense> row_major(rows, cols, batch, ld, stride);
    auto raw = row_major.data();
    std::fill(raw.begin(), raw.end(), -7.0f);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                raw[b * stride + i * pitch + j] = major_marker(i, j, b);

    auto col_major = row_major.to_column_major(pitch);
    EXPECT_EQ(col_major.ld(), rows);
    EXPECT_EQ(col_major.stride(), rows * cols);
    ASSERT_EQ(col_major.data().size(), static_cast<size_t>(rows * cols * batch));

    auto view = col_major.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                EXPECT_FLOAT_EQ(col_major.data()[b * rows * cols + j * rows + i],
                                major_marker(i, j, b))
                    << "column-major (" << i << "," << j << ") batch " << b;
                EXPECT_FLOAT_EQ(view(i, j, b), major_marker(i, j, b));
            }
}

TEST(MatrixMajorConversion, ToRowMajorHonoursPaddedLdAndStrideBatched) {
    constexpr int rows = 3, cols = 4, batch = 2, ld = 5, stride = 30;  // stride > ld * cols
    Matrix<float, MatrixFormat::Dense> col_major(rows, cols, batch, ld, stride);
    auto raw = col_major.data();
    std::fill(raw.begin(), raw.end(), -7.0f);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                raw[b * stride + j * ld + i] = major_marker(i, j, b);  // column-major

    auto row_major = col_major.to_row_major();
    ASSERT_EQ(row_major.data().size(), static_cast<size_t>(rows * cols * batch));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                EXPECT_FLOAT_EQ(row_major.data()[b * rows * cols + i * cols + j],
                                major_marker(i, j, b))
                    << "row-major (" << i << "," << j << ") batch " << b;
}

TEST(MatrixMajorConversion, RoundTripFromPaddedColumnMajor) {
    constexpr int rows = 3, cols = 4, batch = 2, ld = 5, stride = 30;
    Matrix<float, MatrixFormat::Dense> original(rows, cols, batch, ld, stride);
    auto raw = original.data();
    std::fill(raw.begin(), raw.end(), -7.0f);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                raw[b * stride + j * ld + i] = major_marker(i, j, b);

    // column-major (padded) -> row-major (packed) -> column-major (packed)
    auto back = original.to_row_major().to_column_major();
    EXPECT_EQ(back.ld(), rows);
    EXPECT_EQ(back.stride(), rows * cols);
    auto view = back.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                EXPECT_FLOAT_EQ(view(i, j, b), major_marker(i, j, b))
                    << "round trip (" << i << "," << j << ") batch " << b;
}

TEST(MatrixMajorConversion, QueueOverloadMatchesTheQueueLessForm) {
    Queue ctx;
    constexpr int rows = 3, cols = 4, batch = 2, ld = 6, stride = 40, pitch = 6;
    Matrix<float, MatrixFormat::Dense> mat(rows, cols, batch, ld, stride);
    auto raw = mat.data();
    std::fill(raw.begin(), raw.end(), -7.0f);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                raw[b * stride + i * pitch + j] = major_marker(i, j, b);

    auto own_queue = mat.to_column_major(pitch);
    auto given_queue = mat.to_column_major(ctx, pitch);
    ASSERT_EQ(own_queue.data().size(), given_queue.data().size());
    for (size_t k = 0; k < own_queue.data().size(); ++k)
        EXPECT_FLOAT_EQ(own_queue.data()[k], given_queue.data()[k]) << "element " << k;

    auto rm_own = mat.to_row_major();
    auto rm_given = mat.to_row_major(ctx);
    for (size_t k = 0; k < rm_own.data().size(); ++k)
        EXPECT_FLOAT_EQ(rm_own.data()[k], rm_given.data()[k]) << "element " << k;
}

// --- rows > cols: the shape the inferred pitch got wrong ---------------------
//
// Every case above has rows < cols, which is exactly why the inference survived:
// with ld == rows it happened to return cols. The cases below are non-square the
// other way, where "ld == rows" and "row pitch" are different numbers.

TEST(MatrixMajorConversion, ToColumnMajorReadsTallRowMajorAtTheGivenPitch) {
    // rows > cols with a padded row pitch of 8. The old inference
    // (ld > rows ? ld : cols) folded ld == rows into "packed" and read pitch 6,
    // returning 42 of 48 elements wrong without a word.
    constexpr int rows = 8, cols = 6, ld = 8, stride = 64, pitch = 8;
    Matrix<float, MatrixFormat::Dense> row_major(rows, cols, 1, ld, stride);
    auto raw = row_major.data();
    std::fill(raw.begin(), raw.end(), -999.0f);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            raw[i * pitch + j] = major_marker(i, j, 0);

    auto col_major = row_major.to_column_major(pitch);
    EXPECT_EQ(col_major.ld(), rows);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            EXPECT_FLOAT_EQ(col_major.data()[j * rows + i], major_marker(i, j, 0))
                << "column-major (" << i << "," << j << ")";
}

TEST(MatrixMajorConversion, ToColumnMajorDefaultsToPackedForATallMatrix) {
    // Same shape, and the matrix is packed in *both* senses: ld == rows and the
    // items are rows * cols apart, so packed row-major is the only layout that
    // fits. The default pitch reads it, and it is what to_row_major hands back.
    constexpr int rows = 8, cols = 6;
    Matrix<float, MatrixFormat::Dense> row_major(rows, cols, 1);
    ASSERT_EQ(row_major.ld(), rows);
    ASSERT_EQ(row_major.stride(), rows * cols);
    auto raw = row_major.data();
    std::fill(raw.begin(), raw.end(), -999.0f);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            raw[i * cols + j] = major_marker(i, j, 0);

    auto col_major = row_major.to_column_major();
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            EXPECT_FLOAT_EQ(col_major.data()[j * rows + i], major_marker(i, j, 0))
                << "column-major (" << i << "," << j << ")";
}

// --- the default pitch only applies where it is the only possibility ---------
//
// "Default to packed" is safe exactly when the metadata leaves no room for a
// padded row pitch: with ld == rows and stride == rows * cols the buffer holds
// rows * cols elements per item, and a row-major read at pitch p needs
// (rows-1)*p + cols <= rows*cols, i.e. p <= cols, which the existing p >= cols
// check pins to p == cols. Anywhere else a padded row-major buffer fits, and
// guessing "packed" returned the wrong elements in bounds and without a word --
// 42 of 48 for the 8x6/ld 8/stride 64 case below.

TEST(MatrixMajorConversion, ToColumnMajorDefaultPitchRefusesAnOverAllocatedItem) {
    // ld == rows, but the items are 64 apart, so a row-major pitch of 7 or 8 fits
    // just as well as the packed 6. This is the case the probe fell into.
    constexpr int rows = 8, cols = 6, ld = 8, stride = 64;
    Matrix<float, MatrixFormat::Dense> mat(rows, cols, 1, ld, stride);
    EXPECT_THROW(mat.to_column_major(), std::invalid_argument);

    // Both escapes work: state the padded pitch, or state cols to say "packed".
    EXPECT_NO_THROW(mat.to_column_major(8));
    EXPECT_NO_THROW(mat.to_column_major(cols));

    try {
        mat.to_column_major();
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        // The numbers, and both ways out, are in the message.
        EXPECT_NE(msg.find("rows=8"), std::string::npos) << msg;
        EXPECT_NE(msg.find("cols=6"), std::string::npos) << msg;
        EXPECT_NE(msg.find("ld=8"), std::string::npos) << msg;
        EXPECT_NE(msg.find("stride=64"), std::string::npos) << msg;
        EXPECT_NE(msg.find("row pitch"), std::string::npos) << msg;
    }
}

TEST(MatrixMajorConversion, ToColumnMajorDefaultPitchRefusesAPaddedLd) {
    // ld > rows: the allocation is a column-major extent of ld * cols, which again
    // has room for a padded row-major layout.
    Matrix<float, MatrixFormat::Dense> mat(3, 4, 1, /*ld=*/6);
    EXPECT_THROW(mat.to_column_major(), std::invalid_argument);
    EXPECT_NO_THROW(mat.to_column_major(4));
}

TEST(MatrixMajorConversion, ToColumnMajorDefaultPitchRefusesAGappedBatchStride) {
    Matrix<float, MatrixFormat::Dense> mat(3, 4, 2, /*ld=*/3, /*stride=*/20);
    EXPECT_THROW(mat.to_column_major(), std::invalid_argument);
    EXPECT_NO_THROW(mat.to_column_major(4));
}

TEST(MatrixMajorConversion, ToColumnMajorDefaultPitchAllowsASingleRow) {
    // One row: every pitch reads the same elements, so there is nothing to guess
    // and a throw would be noise.
    Matrix<float, MatrixFormat::Dense> mat(1, 6, 1, /*ld=*/4, /*stride=*/40);
    EXPECT_NO_THROW(mat.to_column_major());
}

TEST(MatrixMajorConversion, ToRowMajorHandlesTallPaddedColumnMajor) {
    constexpr int rows = 8, cols = 6, batch = 2, ld = 10, stride = 70;
    Matrix<float, MatrixFormat::Dense> col_major(rows, cols, batch, ld, stride);
    auto raw = col_major.data();
    std::fill(raw.begin(), raw.end(), -7.0f);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                raw[b * stride + j * ld + i] = major_marker(i, j, b);

    auto row_major = col_major.to_row_major();
    ASSERT_EQ(row_major.data().size(), static_cast<size_t>(rows * cols * batch));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                EXPECT_FLOAT_EQ(row_major.data()[b * rows * cols + i * cols + j],
                                major_marker(i, j, b))
                    << "row-major (" << i << "," << j << ") batch " << b;
}

TEST(MatrixMajorConversion, RoundTripTallPaddedColumnMajor) {
    constexpr int rows = 8, cols = 6, batch = 2, ld = 10, stride = 70;
    Matrix<float, MatrixFormat::Dense> original(rows, cols, batch, ld, stride);
    auto raw = original.data();
    std::fill(raw.begin(), raw.end(), -7.0f);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                raw[b * stride + j * ld + i] = major_marker(i, j, b);

    // column-major (padded) -> row-major (packed) -> column-major (packed), with
    // no pitch bookkeeping: to_row_major packs, and that is the default pitch.
    auto back = original.to_row_major().to_column_major();
    auto view = back.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                EXPECT_FLOAT_EQ(view(i, j, b), major_marker(i, j, b))
                    << "round trip (" << i << "," << j << ") batch " << b;
}

// --- a pitch that cannot be honoured is named, not guessed at ----------------

TEST(MatrixMajorConversion, ToColumnMajorRejectsAPitchBelowTheColumnCount) {
    Matrix<float, MatrixFormat::Dense> mat(8, 6, 1, 8, 64);
    EXPECT_THROW(mat.to_column_major(5), std::invalid_argument);
}

TEST(MatrixMajorConversion, ToColumnMajorRejectsAPitchThatOverrunsTheBuffer) {
    // The allocation is a column-major extent, ld * cols = 60 elements; a
    // row-major read at pitch 10 needs (8-1)*10 + 6 = 76. The old code inferred
    // exactly that pitch from ld and read 16 elements past the end.
    Matrix<float, MatrixFormat::Dense> mat(8, 6, 1, 10);
    ASSERT_EQ(mat.data().size(), 60u);
    EXPECT_THROW(mat.to_column_major(10), std::invalid_argument);
}

TEST(MatrixMajorConversion, ToColumnMajorRejectsAPitchThatStraddlesTheNextBatchItem) {
    // stride 50 < (8-1)*8 + 6 = 62: batch item 1 would be read out of item 0's rows.
    Matrix<float, MatrixFormat::Dense> mat(8, 6, 2, 8, 50);
    EXPECT_THROW(mat.to_column_major(8), std::invalid_argument);
}

TEST(MatrixMajorConversion, ToColumnMajorRejectsANegativePitch) {
    Matrix<float, MatrixFormat::Dense> mat(8, 6, 1, 8, 64);
    EXPECT_THROW(mat.to_column_major(-1), std::invalid_argument);
}

// --- adopting a caller's row-major buffer, the way the docs spell it ---------

TEST(MatrixMajorConversion, AdoptsAPackedRowMajorSpanAndConvertsIt) {
    // docs/cpp-api.md: a row-major source is adopted packed (ld = 0), then
    // converted with the default pitch. Non-square, in the tall direction too.
    constexpr int rows = 4, cols = 3;
    std::vector<float> src(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            src[i * cols + j] = major_marker(i, j, 0);

    Matrix<float, MatrixFormat::Dense> adopted(Span<const float>(src.data(), src.size()),
                                               rows, cols, /*ld=*/0);
    ASSERT_EQ(adopted.ld(), rows);
    auto col_major = adopted.to_column_major();
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            EXPECT_FLOAT_EQ(col_major.data()[j * rows + i], src[i * cols + j])
                << "(" << i << "," << j << ")";
}

TEST(MatrixMajorConversion, SpanCtorRejectsARowPitchPassedAsLd) {
    // The copying constructors are column-major by definition, so "ld = row pitch"
    // is not an adoption spelling: it asks for a 15-element column-major source.
    std::vector<float> src(12, 0.0f);
    EXPECT_THROW((Matrix<float, MatrixFormat::Dense>(Span<const float>(src.data(), src.size()),
                                                     3, 4, /*ld=*/4)),
                 std::invalid_argument);
}

// --- the layout invariant the conversions rely on ----------------------------

TEST(MatrixDenseTest, UninitialisedCtorRejectsLdBelowRows) {
    // ld is a column pitch; ld < rows makes every accessor read into the next
    // column. The from-data constructors have always rejected it; this one did not.
    EXPECT_THROW((Matrix<float, MatrixFormat::Dense>(8, 6, 1, 7, 64)), std::invalid_argument);
    EXPECT_THROW((Matrix<float, MatrixFormat::Dense>(8, 6, 1, 7)), std::invalid_argument);
    EXPECT_THROW((Matrix<float, MatrixFormat::Dense>(8, 6, 1, -1)), std::invalid_argument);
}

TEST(MatrixDenseTest, UninitialisedCtorRejectsStrideBelowLdTimesCols) {
    // The allocation is stride * batch, so a short stride under-allocates and the
    // last columns of every item land outside the buffer.
    EXPECT_THROW((Matrix<float, MatrixFormat::Dense>(3, 4, 2, 5, 10)), std::invalid_argument);
    EXPECT_THROW((Matrix<float, MatrixFormat::Dense>(3, 4, 1, 0, -4)), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Bulk data movement: copy_from and astype
//
// Both used to walk the matrix element by element from the host. They now move
// data on the same tiering the from-data constructor uses: one std::copy when
// both layouts are packed, one copy per column otherwise. The cases below are
// the ones that tell a correct bulk copy from a sloppy one -- a padded ld, and a
// batch stride that is not ld * cols.
// ---------------------------------------------------------------------------

TEST(MatrixDenseTest, CopyFromViewWithPaddedLdAndStride) {
    constexpr int rows = 3, cols = 4, batch = 2;
    constexpr int src_ld = 5, src_stride = 30, dst_ld = 7, dst_stride = 40;
    Matrix<float, MatrixFormat::Dense> src(rows, cols, batch, src_ld, src_stride);
    Matrix<float, MatrixFormat::Dense> dst(rows, cols, batch, dst_ld, dst_stride);
    auto sraw = src.data();
    auto draw = dst.data();
    std::fill(sraw.begin(), sraw.end(), -1.0f);
    std::fill(draw.begin(), draw.end(), -2.0f);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                sraw[b * src_stride + j * src_ld + i] = major_marker(i, j, b);

    dst.copy_from(src.view());
    auto view = dst.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                EXPECT_FLOAT_EQ(view(i, j, b), major_marker(i, j, b))
                    << "(" << i << "," << j << ") batch " << b;

    // The padding is not part of the copy and must be left alone.
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = rows; i < dst_ld; ++i)
                EXPECT_FLOAT_EQ(draw[b * dst_stride + j * dst_ld + i], -2.0f)
                    << "padding (" << i << "," << j << ") batch " << b;
}

TEST(MatrixDenseTest, CopyFromViewPackedBatched) {
    constexpr int rows = 3, cols = 4, batch = 3;
    Matrix<float, MatrixFormat::Dense> src(rows, cols, batch);
    Matrix<float, MatrixFormat::Dense> dst(rows, cols, batch);
    auto sraw = src.data();
    for (size_t k = 0; k < sraw.size(); ++k) sraw[k] = static_cast<float>(k) + 0.25f;
    std::fill(dst.data().begin(), dst.data().end(), 0.0f);

    dst.copy_from(src.view());
    for (size_t k = 0; k < sraw.size(); ++k)
        EXPECT_FLOAT_EQ(dst.data()[k], sraw[k]) << "element " << k;
}

TEST(MatrixDenseTest, AstypeHonoursPaddedLdAndStride) {
    constexpr int rows = 3, cols = 4, batch = 2, ld = 5, stride = 30;
    Matrix<float, MatrixFormat::Dense> src(rows, cols, batch, ld, stride);
    auto raw = src.data();
    std::fill(raw.begin(), raw.end(), -1.0f);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                raw[b * stride + j * ld + i] = major_marker(i, j, b);

    auto converted = src.view().astype<double>();
    EXPECT_EQ(converted.rows_, rows);
    EXPECT_EQ(converted.cols_, cols);
    EXPECT_EQ(converted.batch_size_, batch);
    auto view = converted.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                EXPECT_DOUBLE_EQ(view(i, j, b), static_cast<double>(major_marker(i, j, b)))
                    << "(" << i << "," << j << ") batch " << b;
}

TEST(MatrixDenseTest, AstypePackedBatched) {
    constexpr int rows = 2, cols = 3, batch = 3;
    Matrix<double, MatrixFormat::Dense> src(rows, cols, batch);
    auto raw = src.data();
    for (size_t k = 0; k < raw.size(); ++k) raw[k] = static_cast<double>(k) + 0.5;

    auto converted = src.view().astype<float>();
    auto view = converted.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                EXPECT_FLOAT_EQ(view(i, j, b),
                                static_cast<float>(raw[b * rows * cols + j * rows + i]));
}
