// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/verify/reference.hh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <complex>
#include <string>

#include <tests/test_utils.hh>

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

// A Matrix is a MatrixView of its own storage: every (re)allocation must re-point the view half.
TEST(MatrixDenseTest, ViewHalfFollowsTheOwnedStorage) {
    Matrix<float, MatrixFormat::Dense> a(3, 2, 4, /*ld=*/5, /*stride=*/13);
    a.fill(1.0f);
    EXPECT_EQ(static_cast<const MatrixView<float>&>(a).data_ptr(), a.data().data());

    Matrix<float, MatrixFormat::Dense> b(a);
    EXPECT_NE(b.data_ptr(), a.data_ptr());
    EXPECT_EQ(b.kernel_view().data_, b.data().data());
    b(1, 1, 3) = 9.0f;
    EXPECT_FLOAT_EQ(a(1, 1, 3), 1.0f);

    float* const moved = b.data_ptr();
    Matrix<float, MatrixFormat::Dense> c(std::move(b));
    EXPECT_EQ(c.data_ptr(), moved);
    EXPECT_EQ(b.data().size(), 0u);
    a = c;
    EXPECT_NE(a.data_ptr(), c.data_ptr());
    EXPECT_FLOAT_EQ(a(1, 1, 3), 9.0f);

    UnifiedVector<int> rows(4, 2), cols(4, 1);
    c.set_active_dims(rows.to_span(), cols.to_span());
    EXPECT_EQ(c.rows(2), 2);
    EXPECT_EQ(MatrixView<float>(c).cols(3), 1);
    EXPECT_NE(c.active_rows().data(), rows.data());

    Vector<float> x(4, 0.0f, 2, Stride{6}, Inc{1});
    x(3, 1) = 2.0f;
    Vector<float> y(x);
    y(3, 1) = 5.0f;
    EXPECT_FLOAT_EQ(x(3, 1), 2.0f);
    EXPECT_EQ(y.data_ptr(), y.data().data());
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
    dst.copy_from(src);
    for (auto v : dst.data()) EXPECT_FLOAT_EQ(v, 9.0f);
}

TEST(MatrixDenseTest, SubmatrixViewThrowsForCSR) {
    Matrix<float, MatrixFormat::CSR> smat(2, 2, NonZeros{2}, 1);
    EXPECT_THROW(smat.view(1, 1, 1, 1), std::runtime_error);
}

TEST(MatrixDenseTest, ExceptionOnCopyFromMismatchedShape) {
    Matrix<float, MatrixFormat::Dense> a(2, 2, 1);
    Matrix<float, MatrixFormat::Dense> b(3, 2, 1);
    // A shape mismatch is the CALLER's error, so it is invalid_argument, not the
    // catch-all runtime_error this used to assert. Spelled as the concrete type
    // rather than std::exception so that a future reclassification has to come
    // here and be argued about.
    EXPECT_THROW(a.copy_from(b), batchlas::invalid_argument);
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

// The layout tags get the same treatment. If either of these ever decays back to int the
// Vector/VectorView inc-stride swap becomes writable again, silently.
static_assert(!std::is_convertible_v<int, Inc>, "Inc must be explicit");
static_assert(!std::is_convertible_v<Inc, int>, "Inc must not decay to int");
static_assert(!std::is_convertible_v<int, Stride>, "Stride must be explicit");
static_assert(!std::is_convertible_v<Stride, int>, "Stride must not decay to int");
static_assert(!std::is_convertible_v<int, Ld>, "Ld must be explicit");
static_assert(!std::is_convertible_v<Ld, int>, "Ld must not decay to int");
static_assert(!std::is_convertible_v<int, BatchSize>, "BatchSize must be explicit");
static_assert(!std::is_convertible_v<BatchSize, int>, "BatchSize must not decay to int");
static_assert(Stride{7}.value == 7 && Inc{3}.value == 3);

// Vector's four-argument spelling now names stride and inc; the bare-int order it used to
// accept -- the one that transliterates into VectorView's opposite order -- is deleted.
static_assert(!std::is_constructible_v<Vector<float>, int, int, int, int>,
              "Vector(size, batch, stride, inc) with bare ints must not compile");
static_assert(std::is_constructible_v<Vector<float>, int, int, Stride, Inc>,
              "Vector(size, batch, Stride, Inc) must compile");
static_assert(std::is_constructible_v<Vector<float>, int, int>,
              "the two-argument Vector(size, batch) spelling must keep working");
// VectorView keeps its positional constructors (138 correct call sites) and gains tagged
// ones alongside them.
static_assert(std::is_constructible_v<VectorView<float>, float*, int, int, int, int>,
              "VectorView's positional constructor must keep working");
static_assert(std::is_constructible_v<VectorView<float>, float*, int, int, Inc, Stride>,
              "VectorView must accept the tagged spelling");

TEST(VectorLayoutTagTest, TaggedArgumentsLandInTheRightFields) {
    constexpr int size = 5, batch = 3, stride = 7, inc = 2;
    // stride and inc deliberately distinct and non-default, so a swap changes an
    // observable number rather than landing on an equal one.
    Vector<float> v(size, batch, Stride{stride}, Inc{inc});
    EXPECT_EQ(v.size(), size);
    EXPECT_EQ(v.batch_size(), batch);
    EXPECT_EQ(v.stride(), stride);
    EXPECT_EQ(v.inc(), inc);

    // The view spelled with the tags means the same layout, even though its positional
    // order is the reverse of Vector's.
    VectorView<float> view(v.data_ptr(), size, batch, Inc{inc}, Stride{stride});
    EXPECT_EQ(view.stride(), stride);
    EXPECT_EQ(view.inc(), inc);

    auto z = Vector<float>::zeros(size, batch, Stride{stride}, Inc{inc});
    EXPECT_EQ(z.stride(), stride);
    EXPECT_EQ(z.inc(), inc);
    EXPECT_EQ(Vector<float>::zeros(size, batch).stride(), size);
    EXPECT_EQ(Vector<float>::zeros(size, batch).inc(), 1);
}

TEST(MatrixViewTrapTest, LeadingDimensionSmallerThanRowsThrows) {
    // The trap this guards: MatrixView takes (data, rows, cols, ld, stride, batch_size)
    // while Matrix takes (rows, cols, batch_size, ld, stride), so a caller who learned the
    // order from Matrix writes the batch count into ld. It used to be accepted silently.
    constexpr int n = 8, batch = 2;
    std::vector<float> buffer(static_cast<size_t>(n) * n * batch, 0.0f);
    EXPECT_THROW((MatrixView<float, MatrixFormat::Dense>(buffer.data(), n, n, batch)),
                 std::invalid_argument);
    // The spelling the error message names must be accepted.
    EXPECT_NO_THROW((MatrixView<float, MatrixFormat::Dense>(buffer.data(), n, n, n, 0, batch)));
    // And the innocent shorthands must stay innocent: a shape-only stand-in for a
    // workspace query, and the three-argument spelling.
    EXPECT_NO_THROW((MatrixView<float, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, batch)));
    EXPECT_NO_THROW((MatrixView<float, MatrixFormat::Dense>(buffer.data(), n, n)));
}

TEST(MatrixCSRTest, ConvertToSeparatesPerItemCountFromCapacity) {
    // A heterogeneous batch is the normal result of a dense -> CSR conversion, and the
    // conversion sizes the whole batch by its largest item. nnz() is that capacity;
    // nnz(b) is what item b actually stored.
    constexpr int n = 4, batch = 2;
    Matrix<float, MatrixFormat::Dense> dense(n, n, batch);
    std::fill(dense.data().begin(), dense.data().end(), 0.0f);

    // Item 0: three non-zeros. Item 1: six.
    dense(0, 0, 0) = 1.0f; dense(1, 2, 0) = 2.0f; dense(3, 3, 0) = 3.0f;
    for (int i = 0; i < 3; ++i) { dense(i, i, 1) = 1.0f; dense(i, i + 1, 1) = 2.0f; }
    constexpr int nnz0 = 3, nnz1 = 6;

    auto csr = dense.convert_to<MatrixFormat::CSR>();
    EXPECT_EQ(csr.nnz(), std::max(nnz0, nnz1));      // capacity, the batch maximum
    EXPECT_EQ(csr.nnz_capacity(), std::max(nnz0, nnz1));
    EXPECT_EQ(csr.nnz(0), nnz0);                     // what item 0 actually stored
    EXPECT_EQ(csr.nnz(1), nnz1);

    // Regression guard: the scan writes offsets 1..rows of each item and never element 0,
    // and the allocation does not zero, so this used to be uninitialised memory -- which
    // is exactly what nnz(b) subtracts.
    for (int b = 0; b < batch; ++b) {
        EXPECT_EQ(csr.row_offsets()[static_cast<size_t>(b) * csr.offset_stride()], 0);
    }
}

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
    auto item = from_data.batch_item(1);
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
    dst.copy_from(src);
    for (int i = 0; i < nnz; ++i) EXPECT_FLOAT_EQ(dst.data()[i], values[i]);
    for (int i = 0; i < rows + 1; ++i) EXPECT_EQ(dst.row_offsets()[i], row_offsets[i]);
    for (int i = 0; i < nnz; ++i) EXPECT_EQ(dst.col_indices()[i], col_indices[i]);
}
 
TEST(MatrixCSRTest, ExceptionOnCopyFromMismatchedShape) {
    Matrix<float, MatrixFormat::CSR> a(2, 2, NonZeros{2}, 1);
    Matrix<float, MatrixFormat::CSR> b(3, 2, NonZeros{2}, 1);
    // A shape mismatch is the CALLER's error, so it is invalid_argument, not the
    // catch-all runtime_error this used to assert. Spelled as the concrete type
    // rather than std::exception so that a future reclassification has to come
    // here and be argued about.
    EXPECT_THROW(a.copy_from(b), batchlas::invalid_argument);
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

    mat.triangularize(ctx, Uplo::Upper, Diag::NonUnit).wait();

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
    const int ld = mat.ld();
    const int stride = mat.stride();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                mat.data()[b * stride + j * ld + i] = tri_marker(i, j);

    mat.triangularize(ctx, Uplo::Lower, Diag::NonUnit).wait();

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
    const int ld = mat.ld();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            mat.data()[j * ld + i] = tri_marker(i, j);

    mat.triangularize(ctx, Uplo::Upper, Diag::Unit).wait();

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
        const int ld = mat.ld();
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                mat.data()[j * ld + i] = tri_marker(i, j);

        mat.triangularize(ctx, Uplo::Upper, Diag::NonUnit).wait();

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

    dst.copy_from(src);
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

    dst.copy_from(src);
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

// ---- fill_random_spd / fill_random_pivoted / fill_graded_hermitian ----
// Every case runs on a padded layout (ld > n, stride > ld * n) whose padding holds a poison the
// fills must leave alone, with batch >= 3 so per-item seeding is visible.

template <typename T, Backend B>
struct FillConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using FillTestTypes = typename test_utils::backend_types<FillConfig>::type;

template <typename Config>
class MatrixFillTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = batchlas::float_t<T>;

    static T poison() {
        if constexpr (is_std_complex_v<T>) return T(1234.5, -77.25);
        else return T(1234.5);
    }

    static Matrix<T, MatrixFormat::Dense> padded(int n, int batch) {
        const int ld = n + 3;
        Matrix<T, MatrixFormat::Dense> m(n, n, batch, ld, ld * n + 7);
        for (auto& x : m.data()) x = poison();
        return m;
    }

    static T at(Matrix<T, MatrixFormat::Dense>& m, int i, int j, int b) {
        return m.data()[static_cast<std::size_t>(b) * m.stride() + static_cast<std::size_t>(j) * m.ld() + i];
    }

    static void expect_padding_untouched(Matrix<T, MatrixFormat::Dense>& m) {
        const int n = m.rows();
        std::size_t bad = 0;
        for (std::size_t k = 0; k < m.data().size(); ++k) {
            const std::size_t off = k % m.stride();
            const bool inside = off / m.ld() < static_cast<std::size_t>(n) && off % m.ld() < static_cast<std::size_t>(n);
            if (!inside && !(m.data()[k] == poison())) ++bad;
        }
        EXPECT_EQ(bad, 0u) << "padding entries overwritten";
    }

    static std::vector<T> active(Matrix<T, MatrixFormat::Dense>& m) {
        std::vector<T> out;
        for (int b = 0; b < m.batch_size(); ++b)
            for (int j = 0; j < m.cols(); ++j)
                for (int i = 0; i < m.rows(); ++i) out.push_back(at(m, i, j, b));
        return out;
    }

    static bool items_equal(Matrix<T, MatrixFormat::Dense>& m, int b0, int b1) {
        for (int j = 0; j < m.cols(); ++j)
            for (int i = 0; i < m.rows(); ++i)
                if (!(at(m, i, j, b0) == at(m, i, j, b1))) return false;
        return true;
    }

    static void expect_items_differ(Matrix<T, MatrixFormat::Dense>& m) {
        for (int b = 1; b < m.batch_size(); ++b) EXPECT_FALSE(items_equal(m, 0, b)) << "item " << b << " equals item 0";
    }

    static void expect_exactly_hermitian(Matrix<T, MatrixFormat::Dense>& m) {
        const int n = m.rows();
        std::size_t asym = 0, nonreal = 0, nonfinite = 0;
        for (int b = 0; b < m.batch_size(); ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const T x = at(m, i, j, b);
                    if (!std::isfinite(std::real(x)) || !std::isfinite(std::imag(x))) ++nonfinite;
                    if (i == j && std::imag(x) != R(0)) ++nonreal;
                    if (i != j) {
                        T mirrored = at(m, j, i, b);
                        if constexpr (is_std_complex_v<T>) mirrored = std::conj(mirrored);
                        if (!(x == mirrored)) ++asym;
                    }
                }
        EXPECT_EQ(asym, 0u) << "entries with a_ij != conj(a_ji)";
        EXPECT_EQ(nonreal, 0u) << "diagonal entries with a nonzero imaginary part";
        EXPECT_EQ(nonfinite, 0u) << "non-finite entries";
    }

    template <typename Fill>
    void expect_reproducible(int n, Fill fill) {
        auto a = padded(n, 3), b = padded(n, 3), c = padded(n, 3);
        fill(a.view(), 11u).wait();
        fill(b.view(), 11u).wait();
        fill(c.view(), 12u).wait();
        EXPECT_TRUE(active(a) == active(b)) << "the same seed did not reproduce bit-identically";
        EXPECT_FALSE(active(a) == active(c)) << "a different seed gave the same matrices";
    }
};

TYPED_TEST_SUITE(MatrixFillTest, FillTestTypes);

TYPED_TEST(MatrixFillTest, RandomSpdIsHermitianAndStrictlyDiagonallyDominant) {
    using R = typename TestFixture::R;
    for (int n : {1, 5, 37}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        auto m = TestFixture::padded(n, 4);
        m.fill_random_spd(*this->ctx, 7u).wait();
        TestFixture::expect_padding_untouched(m);
        TestFixture::expect_exactly_hermitian(m);
        if (n > 1) TestFixture::expect_items_differ(m);
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < n; ++i) {
                R off = 0;
                for (int j = 0; j < n; ++j)
                    if (j != i) off += std::abs(TestFixture::at(m, i, j, b));
                const R d = std::real(TestFixture::at(m, i, i, b));
                EXPECT_GT(d, off) << "row " << i << " item " << b << " not strictly dominant";
                EXPECT_GT(d, R(0));
            }
    }
    this->expect_reproducible(9, [&](auto v, unsigned s) { return v.fill_random_spd(*this->ctx, s); });
    if constexpr (TestFixture::BackendType != Backend::NETLIB) {
        auto q = TestFixture::padded(9, 3), d = TestFixture::padded(9, 3);
        q.fill_random_spd(*this->ctx, 5u).wait();
        d.fill_random_spd(5u).wait();
        EXPECT_TRUE(TestFixture::active(q) == TestFixture::active(d)) << "queue-less twin differs";
    }
}

TYPED_TEST(MatrixFillTest, RandomPivotedMovesEveryRowPerItem) {
    using R = typename TestFixture::R;
    for (int n : {4, 9, 40}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        constexpr int batch = 5;
        auto m = TestFixture::padded(n, batch);
        m.fill_random_pivoted(*this->ctx, 3u).wait();
        TestFixture::expect_padding_untouched(m);
        TestFixture::expect_items_differ(m);
        std::vector<std::vector<int>> owner(batch, std::vector<int>(n, -1));
        for (int b = 0; b < batch; ++b) {
            std::vector<int> hits(n, 0);
            for (int c = 0; c < n; ++c)
                for (int i = 0; i < n; ++i)
                    if (std::abs(TestFixture::at(m, i, c, b)) >= R(n - 1)) {
                        owner[b][c] = i;
                        ++hits[i];
                    }
            for (int c = 0; c < n; ++c) {
                ASSERT_GE(owner[b][c], 0) << "column " << c << " item " << b << " has no dominant entry";
                // A random n-cycle moves every row, so the identity is impossible, not just unlikely.
                EXPECT_NE(owner[b][c], c) << "column " << c << " item " << b << " kept its row";
            }
            for (int i = 0; i < n; ++i) EXPECT_EQ(hits[i], 1) << "row " << i << " item " << b;
        }
        bool any_differs = false;
        for (int b = 1; b < batch; ++b) any_differs = any_differs || owner[b] != owner[0];
        EXPECT_TRUE(any_differs) << "every item has the same row permutation";
    }
    this->expect_reproducible(9, [&](auto v, unsigned s) { return v.fill_random_pivoted(*this->ctx, s); });
}

TYPED_TEST(MatrixFillTest, GradedHermitianIsExactlyHermitian) {
    for (int n : {1, 16, 70}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        auto m = TestFixture::padded(n, 3);
        m.fill_graded_hermitian(*this->ctx, 6.0, 21u).wait();
        TestFixture::expect_padding_untouched(m);
        TestFixture::expect_exactly_hermitian(m);
        // n == 1: lambda = 1, and H M H only rounds it (|v|^2 is 1 to a few ulps).
        const auto eps = std::numeric_limits<typename TestFixture::R>::epsilon();
        if (n == 1) EXPECT_LE(std::abs(TestFixture::at(m, 0, 0, 0) - typename TestFixture::T(1)), 8 * eps);
        else TestFixture::expect_items_differ(m);
    }
    this->expect_reproducible(12, [&](auto v, unsigned s) { return v.fill_graded_hermitian(*this->ctx, 3.0, s); });
}

TYPED_TEST(MatrixFillTest, GradedHermitianHasTheRequestedSpectrum) {
    using T = typename TestFixture::T;
    if constexpr (!std::is_same_v<batchlas::float_t<T>, double>) {
        GTEST_SKIP() << "spectrum checked in double precision only";
    } else if constexpr (!batchlas::verify::have_reference) {
        GTEST_SKIP() << "no LAPACKE reference";
    } else {
        for (int n : {16, 64})
            for (double cond : {3.0, 6.0}) {
                SCOPED_TRACE("n=" + std::to_string(n) + " log10_cond=" + std::to_string(cond));
                auto m = TestFixture::padded(n, 3);
                m.fill_graded_hermitian(*this->ctx, cond, 5u).wait();
                for (int b = 0; b < 3; ++b) {
                    std::vector<T> a;
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) a.push_back(TestFixture::at(m, i, j, b));
                    std::vector<double> w;
                    ASSERT_TRUE(batchlas::verify::eigenvalues(n, a, w));
                    EXPECT_NEAR(w.back() / w.front(), std::pow(10.0, cond), 0.01 * std::pow(10.0, cond)) << "item " << b;
                    double worst = 0;
                    for (int i = 0; i < n; ++i) {
                        const double lambda = std::pow(10.0, -cond * double(n - 1 - i) / double(n - 1));
                        worst = std::max(worst, std::abs(w[i] - lambda));
                    }
                    // Normwise relative (lambda_max = 1): a small eigenvalue carries the
                    // reflectors' absolute rounding, ~n eps, not a relative one.
                    EXPECT_LT(worst, 1e-10) << "item " << b;
                }
            }
    }
}

TYPED_TEST(MatrixFillTest, GradedHermitianRunsAtN512) {
    constexpr int n = 512;
    auto m = TestFixture::padded(n, 2);
    m.fill_graded_hermitian(*this->ctx, 6.0, 4u).wait();
    TestFixture::expect_padding_untouched(m);
    TestFixture::expect_exactly_hermitian(m);
    TestFixture::expect_items_differ(m);
    if constexpr (std::is_same_v<typename TestFixture::R, double> && batchlas::verify::have_reference) {
        std::vector<typename TestFixture::T> a;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) a.push_back(TestFixture::at(m, i, j, 1));
        std::vector<double> w;
        ASSERT_TRUE(batchlas::verify::eigenvalues(n, a, w));
        EXPECT_NEAR(w.back() / w.front(), 1e6, 1e4);
    }
}

// Diagonal floor(i/4) + (i%4) 1e-4 plus a Hermitian perturbation of size noise: clusters of four.
TYPED_TEST(MatrixFillTest, NearDegenerateHermitianIsClusteredAndExactlyHermitian) {
    using R = typename TestFixture::R;
    constexpr double noise = 1e-3;
    for (int n : {1, 9, 33}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        auto m = TestFixture::padded(n, 3);
        m.fill_near_degenerate_hermitian(*this->ctx, noise, 1337u).wait();
        TestFixture::expect_padding_untouched(m);
        TestFixture::expect_exactly_hermitian(m);
        if (n > 1) TestFixture::expect_items_differ(m);
        // |Re z| < 1 and |z| <= sqrt(2): every entry is noise-sized around its target.
        const R bound = R(noise * 1.5) + 4 * std::numeric_limits<R>::epsilon() * R(n);
        for (int b = 0; b < 3; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const R target = i == j ? R(i / 4) + R(i % 4) * R(1e-4) : R(0);
                    EXPECT_LE(std::abs(TestFixture::at(m, i, j, b) - typename TestFixture::T(target)), bound)
                        << "(" << i << "," << j << ") item " << b;
                }
    }
    this->expect_reproducible(12, [&](auto v, unsigned s) { return v.fill_near_degenerate_hermitian(*this->ctx, noise, s); });
}

TYPED_TEST(MatrixFillTest, SquareFillsRejectARectangularView) {
    Matrix<typename TestFixture::T, MatrixFormat::Dense> m(5, 4, 3);
    EXPECT_THROW((void)m.fill_random_spd(*this->ctx), batchlas::invalid_argument);
    EXPECT_THROW((void)m.fill_random_pivoted(*this->ctx), batchlas::invalid_argument);
    EXPECT_THROW((void)m.fill_graded_hermitian(*this->ctx, 3.0), batchlas::invalid_argument);
    EXPECT_THROW((void)m.fill_near_degenerate_hermitian(*this->ctx, 1e-3), batchlas::invalid_argument);
}

// Race guard for the two local-memory kernels: a saturating batch must give, item for item,
// the same bits on a packed layout as on a padded one, and the same bits twice.
TYPED_TEST(MatrixFillTest, LocalMemoryFillsAreRaceFreeAtASaturatingBatch) {
    using T = typename TestFixture::T;
    constexpr int n = 32, batch = 1024;
    for (int which = 0; which < 2; ++which) {
        SCOPED_TRACE(which == 0 ? "pivoted" : "graded");
        auto run = [&](Matrix<T, MatrixFormat::Dense>& m) {
            if (which == 0) m.fill_random_pivoted(*this->ctx, 8u).wait();
            else m.fill_graded_hermitian(*this->ctx, 4.0, 8u).wait();
        };
        auto padded = TestFixture::padded(n, batch);
        Matrix<T, MatrixFormat::Dense> packed(n, n, batch), again(n, n, batch);
        run(padded);
        run(packed);
        run(again);
        EXPECT_TRUE(TestFixture::active(packed) == TestFixture::active(padded)) << "layout changed the values";
        EXPECT_TRUE(TestFixture::active(packed) == TestFixture::active(again)) << "a rerun changed the values";
    }
}

// compute_nd_range_sizes caps the grid above INT_MAX / 2 work items, so a kernel that handles one
// element per work-item silently skipped everything past the cap. 2^30 + 32768 floats (4 GiB).
TEST(MatrixKernels, GridCapCoversAllElements) {
    Queue q;
    constexpr int rows = 1 << 15, cols = (1 << 15) + 1;
    const size_t total = static_cast<size_t>(rows) * cols;
    if (q.device().type != DeviceType::GPU ||
        q.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE) < (size_t{10} << 30))
        GTEST_SKIP() << "needs a GPU with at least 10 GB";
    Matrix<float, MatrixFormat::Dense> M(rows, cols, 1);
    M.view().fill(q, 1.0f).wait();
    scale(q, 2.0f, M).wait();
    const float* d = M.data_ptr();
    for (size_t idx : {size_t{0}, size_t{1} << 29, (size_t{1} << 30) - 1, size_t{1} << 30, total - rows, total - 1})
        EXPECT_EQ(d[idx], 2.0f) << "element " << idx << " of " << total;
}

// fill_diagonal on a wide padded view: diagonal k holds min(rows - row_off, cols - col_off) entries
// (none once |k| reaches the edge); nothing else in the storage, padding included, may change.
namespace {
constexpr float kDiagPoison = 1234.5f;

Matrix<float, MatrixFormat::Dense> poisoned_wide() {
    Matrix<float, MatrixFormat::Dense> m(5, 9, 2, 7, 7 * 9 + 3);
    for (auto& x : m.data()) x = kDiagPoison;
    return m;
}

// The number of storage entries that differ from what diagonal k with value 100 b + i + 1 implies.
std::size_t diagonal_mismatches(Matrix<float, MatrixFormat::Dense>& m, int k) {
    const int ro = k < 0 ? -k : 0, co = k > 0 ? k : 0;
    std::size_t bad = 0;
    for (std::size_t s = 0; s < m.data().size(); ++s) {
        const int b = int(s / m.stride()), off = int(s % m.stride());
        const int c = off / m.ld(), r = off % m.ld();
        const bool on = b < m.batch_size() && c < m.cols() && r < m.rows() && r - ro == c - co && r >= ro;
        const float want = on ? float(100 * b + (r - ro) + 1) : kDiagPoison;
        if (m.data()[s] != want) ++bad;
    }
    return bad;
}
}  // namespace

TEST(FillDiagonalVector, WideViewAndOffsets) {
    Queue ctx(Device::default_device());
    UnifiedVector<float> values(9 * 2);
    for (int b = 0; b < 2; ++b)
        for (int i = 0; i < 9; ++i) values[b * 9 + i] = float(100 * b + i + 1);
    for (int k : {-4, 0, 3}) {
        SCOPED_TRACE("k=" + std::to_string(k));
        auto m = poisoned_wide();
        m.fill_diagonal(ctx, VectorView<float>(values, 9, 2, 1, 9), k).wait();
        EXPECT_EQ(diagonal_mismatches(m, k), 0u);
    }
    // A strided source (inc 2, stride 19) with poison in the skipped slots.
    UnifiedVector<float> strided(19 * 2, kDiagPoison);
    for (int b = 0; b < 2; ++b)
        for (int i = 0; i < 9; ++i) strided[b * 19 + 2 * i] = float(100 * b + i + 1);
    for (int k : {-4, 3}) {
        SCOPED_TRACE("inc=2 k=" + std::to_string(k));
        auto m = poisoned_wide();
        m.fill_diagonal(ctx, VectorView<float>(strided, 9, 2, Inc{2}, Stride{19}), k).wait();
        EXPECT_EQ(diagonal_mismatches(m, k), 0u);
    }
    // The Span overload packs one run of exactly min(...) values per item.
    const int d = 5;  // k = 3 on 5x9
    UnifiedVector<float> packed(d * 2);
    for (int b = 0; b < 2; ++b)
        for (int i = 0; i < d; ++i) packed[b * d + i] = float(100 * b + i + 1);
    auto m = poisoned_wide();
    m.fill_diagonal(ctx, packed.to_span(), 3).wait();
    EXPECT_EQ(diagonal_mismatches(m, 3), 0u) << "Span overload";
}

TEST(FillDiagonalVector, OffsetsAtOrPastTheEdge) {
    Queue ctx(Device::default_device());
    UnifiedVector<float> values(9 * 2);
    for (int b = 0; b < 2; ++b)
        for (int i = 0; i < 9; ++i) values[b * 9 + i] = float(100 * b + i + 1);
    for (int k : {8, 9, 12, -5, -6}) {
        SCOPED_TRACE("k=" + std::to_string(k));
        auto m = poisoned_wide();
        ASSERT_NO_THROW(m.fill_diagonal(ctx, VectorView<float>(values, 9, 2, 1, 9), k).wait());
        EXPECT_EQ(diagonal_mismatches(m, k), 0u);
    }
}

// The permutation and the values of the seeded fills come from different Philox streams, so a
// seed that differs by the old XOR salt cannot reproduce one from the other. Sattolo's draws are
// recovered from the permutation; the bug made them a linear function of fill_random_spd's column 0.
TEST(SeededFills, IndependentStreams) {
    constexpr int n = 256;
    constexpr unsigned int seed = 21u;
    Queue ctx(Device::default_device());
    Matrix<float, MatrixFormat::Dense> piv(n, n), spd(n, n);
    piv.fill_random_pivoted(ctx, seed).wait();
    spd.fill_random_spd(ctx, seed ^ 0x5bd1e995u).wait();

    // Row i carries the dominant entry in column perm[i].
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) {
        int arg = 0;
        for (int c = 1; c < n; ++c)
            if (std::fabs(piv.data()[std::size_t(c) * n + i]) > std::fabs(piv.data()[std::size_t(arg) * n + i])) arg = c;
        perm[i] = arg;
    }
    std::vector<int> cur(n);
    for (int i = 0; i < n; ++i) cur[i] = i;
    std::vector<double> x, y;
    for (int i = n - 1; i > 0; --i) {
        const int j = int(std::find(cur.begin(), cur.end(), perm[i]) - cur.begin());
        ASSERT_LT(j, i) << "not a Sattolo cycle at " << i;
        x.push_back((j + 0.5) / i);
        y.push_back(spd.data()[i]);
        std::swap(cur[i], cur[j]);
    }
    const double mx = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    const double my = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double sxy = 0, sxx = 0, syy = 0;
    for (std::size_t t = 0; t < x.size(); ++t) {
        sxy += (x[t] - mx) * (y[t] - my);
        sxx += (x[t] - mx) * (x[t] - mx);
        syy += (y[t] - my) * (y[t] - my);
    }
    // 255 independent pairs: |r| has sd 0.063, so 0.25 is 4 sd; the shared stream gave r ~ 1.
    EXPECT_LT(std::fabs(sxy / std::sqrt(sxx * syy)), 0.25);
}
