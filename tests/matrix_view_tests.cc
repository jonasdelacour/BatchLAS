#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <complex>
#include <cstdint>
#include <limits>

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

// batch_range keeps extents, ld and stride (non-natural here), offsets the base by first * stride in
// 64 bits, takes the matching active extents and drops the pointer array (it would address item 0).
TEST(MatrixViewTest, BatchRangeView) {
    constexpr int rows = 3, cols = 2, ld = 5, stride = 13, batch = 5;
    UnifiedVector<float> mem(std::size_t(stride) * batch, -1.0f);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) mem[std::size_t(b) * stride + j * ld + i] = float(100 * b + 10 * j + i);
    UnifiedVector<float*> ptrs(batch, nullptr);
    const MatrixView<float, MatrixFormat::Dense> view(mem.data(), rows, cols, ld, stride, batch, ptrs.data());
    const auto r = view.batch_range(2, 3);
    EXPECT_EQ(r.data_ptr(), mem.data() + 2 * stride);
    EXPECT_EQ(r.rows(), rows);
    EXPECT_EQ(r.cols(), cols);
    EXPECT_EQ(r.ld(), ld);
    EXPECT_EQ(r.stride(), stride);
    EXPECT_EQ(r.batch_size(), 3);
    for (int b = 0; b < 3; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) EXPECT_EQ(r(i, j, b), float(100 * (b + 2) + 10 * j + i));
    Queue q;
    EXPECT_THROW((void)r.data_ptrs(q), batchlas::invalid_argument);
    EXPECT_NO_THROW((void)view.batch_range(0, batch));
    EXPECT_NO_THROW((void)view.batch_range(batch - 1, 1));
    EXPECT_THROW((void)view.batch_range(-1, 2), batchlas::out_of_range);
    EXPECT_THROW((void)view.batch_range(1, 0), batchlas::out_of_range);
    EXPECT_THROW((void)view.batch_range(3, 3), batchlas::out_of_range);
    EXPECT_THROW((void)view.batch_range(1, std::numeric_limits<int>::max()), batchlas::out_of_range);

    UnifiedVector<int> ar(batch), ac(batch);
    for (int b = 0; b < batch; ++b) ar[b] = 1 + b % 3, ac[b] = 2 - b % 2;
    const auto het = view.with_active_dims(ar.to_span(), ac.to_span()).batch_range(1, 4);
    EXPECT_TRUE(het.is_heterogeneous());
    for (int b = 0; b < 4; ++b) {
        EXPECT_EQ(het.rows(b), ar[b + 1]);
        EXPECT_EQ(het.cols(b), ac[b + 1]);
    }

    // first * stride past INT_MAX: the offset is 64-bit (pointer arithmetic only, nothing is read).
    const int big_stride = 1 << 20, big_batch = 4096;
    const MatrixView<float, MatrixFormat::Dense> wide(mem.data(), rows, cols, ld, big_stride, big_batch);
    const auto far = wide.batch_range(big_batch - 1, 1);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(far.data_ptr()) - reinterpret_cast<std::uintptr_t>(mem.data()),
              std::uintptr_t(big_batch - 1) * big_stride * sizeof(float));
}

TEST(MatrixViewTest, DenseHeterogeneousActiveDimensionsPropagateToViews) {
    constexpr int rows = 4, cols = 5, batch = 3;
    Matrix<float, MatrixFormat::Dense> mat(rows, cols, batch);

    UnifiedVector<int> active_rows(batch);
    UnifiedVector<int> active_cols(batch);
    active_rows[0] = 4;
    active_rows[1] = 2;
    active_rows[2] = 0;
    active_cols[0] = 5;
    active_cols[1] = 3;
    active_cols[2] = 0;

    mat.set_active_dims(active_rows.to_span(), active_cols.to_span());

    auto view = mat.view();
    EXPECT_TRUE(view.is_heterogeneous());
    EXPECT_EQ(view.rows_capacity(), rows);
    EXPECT_EQ(view.cols_capacity(), cols);
    EXPECT_EQ(view.rows(0), 4);
    EXPECT_EQ(view.cols(0), 5);
    EXPECT_EQ(view.rows(1), 2);
    EXPECT_EQ(view.cols(1), 3);
    EXPECT_EQ(view.rows(2), 0);
    EXPECT_EQ(view.cols(2), 0);

    auto item = view.batch_item(1);
    EXPECT_FALSE(item.is_heterogeneous());
    EXPECT_EQ(item.rows(), 2);
    EXPECT_EQ(item.cols(), 3);

    auto kernel_view = view.kernel_view();
    EXPECT_TRUE(kernel_view.is_heterogeneous());
    EXPECT_EQ(kernel_view.rows(1), 2);
    EXPECT_EQ(kernel_view.cols(1), 3);
}

TEST(MatrixViewTest, DenseHeterogeneousMetadataRejectsOutOfCapacityDimensions) {
    Matrix<float, MatrixFormat::Dense> mat(3, 3, 2);
    UnifiedVector<int> active_rows(2);
    UnifiedVector<int> active_cols(2);
    active_rows[0] = 3;
    active_rows[1] = 4;
    active_cols[0] = 3;
    active_cols[1] = 3;

    EXPECT_THROW(mat.set_active_dims(active_rows.to_span(), active_cols.to_span()), std::invalid_argument);
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

#if BATCHLAS_HAS_CPU_TARGET || BATCHLAS_HAS_GPU_BACKEND
TEST(MatrixViewTest, DenseCopyHandlesTallSkinnyViewsWithPadding) {
    constexpr int rows = 5;
    constexpr int cols = 3;
    constexpr int ld = 7;
    constexpr int stride = 29;
    constexpr int batch = 2;

    Matrix<float, MatrixFormat::Dense> src(rows, cols, batch, ld, stride);
    Matrix<float, MatrixFormat::Dense> dst(rows, cols, batch, ld, stride);

    src.fill(-1.0f);
    dst.fill(-9.0f);

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                src(i, j, b) = static_cast<float>(100 * b + 10 * j + i);
            }
        }
    }

    Queue ctx(Device::default_device());
    MatrixView<float, MatrixFormat::Dense>::copy(ctx, dst, src).wait();
    ctx.wait_and_throw();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                EXPECT_FLOAT_EQ(dst(i, j, b), src(i, j, b))
                    << "Mismatch at batch=" << b << " row=" << i << " col=" << j;
            }
        }
    }
}

TEST(MatrixViewTest, DenseCopyHandlesContiguousMatricesWithDifferentBatchStrides) {
    constexpr int rows = 48;
    constexpr int cols = 6;
    constexpr int batch = 3;
    constexpr int src_stride = rows * cols * 3;
    constexpr int dst_stride = rows * cols;

    Matrix<float, MatrixFormat::Dense> src(rows, cols, batch, rows, src_stride);
    Matrix<float, MatrixFormat::Dense> dst(rows, cols, batch, rows, dst_stride);

    src.fill(-1.0f);
    dst.fill(-9.0f);

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                src(i, j, b) = static_cast<float>(1000 * b + 100 * j + i);
            }
        }
    }

    Queue ctx(Device::default_device());
    MatrixView<float, MatrixFormat::Dense>::copy(ctx, dst, src).wait();
    ctx.wait_and_throw();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                EXPECT_FLOAT_EQ(dst(i, j, b), src(i, j, b))
                    << "Mismatch at batch=" << b << " row=" << i << " col=" << j;
            }
        }
    }
}
#endif

// --- CSR MatrixView tests ---
TEST(MatrixViewTest, ConstructFromCSRMatrix) {
    constexpr int rows = 3, cols = 3, nnz = 5, batch = 2;
    // Example CSR data for two batches
    float values[2 * nnz] = {1, 2, 3, 4, 5, 10, 20, 30, 40, 50};
    int row_offsets[2 * (rows + 1)] = {0, 2, 3, 5, 0, 2, 3, 5};
    int col_indices[2 * nnz] = {0, 2, 1, 0, 2, 0, 2, 1, 0, 2};
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, rows, cols, NonZeros{nnz}, nnz, rows + 1, batch);
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
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, rows, cols, NonZeros{nnz}, nnz, rows + 1, batch);
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
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, rows, cols, NonZeros{nnz}, nnz, rows + 1, 1);
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
    Matrix<float, MatrixFormat::CSR> mat(values, row_offsets, col_indices, rows, cols, NonZeros{nnz}, nnz, rows + 1, 1);
    EXPECT_THROW(mat.view(1, 1), std::runtime_error);
}

// A host slice reaching past the dimension used to hand out a view onto foreign memory
// (only an EMPTY slice was rejected).
TEST(MatrixView, SliceOutOfRangeThrows) {
    Matrix<float, MatrixFormat::Dense> M(6, 5, 2);
    auto v = M.view();
    EXPECT_THROW((void)v(Slice(0, 6), Slice(2, 9)), batchlas::out_of_range);
    EXPECT_THROW((void)v(Slice(4, 8), Slice()), batchlas::out_of_range);
    EXPECT_THROW((void)v(Slice(-9, 3), Slice()), batchlas::out_of_range);
    EXPECT_THROW((void)v(Slice(7), Slice()), batchlas::out_of_range);
    EXPECT_THROW((void)v(2, Slice(1, 7)), batchlas::out_of_range);
    EXPECT_THROW((void)v(Slice(0, 7), 1), batchlas::out_of_range);
    EXPECT_THROW((void)M(Slice(0, 6), Slice(2, 9)), batchlas::out_of_range);
    // GUARD: in-range slices, negative bounds included, are still accepted.
    EXPECT_NO_THROW((void)v(Slice(0, 6), Slice(2, 5)));
    EXPECT_NO_THROW((void)v(Slice(-2, 6), Slice(-3)));
    EXPECT_NO_THROW((void)M(Slice(1), Slice(0, -1)));
    EXPECT_THROW((void)v(Slice(3, 3), Slice()), batchlas::invalid_argument) << "empty stays invalid_argument";
}

// The device-side twin cannot throw; its assert used to be assert("string"), which never fires.
TEST(KernelMatrixView, BadSliceAsserts) {
#ifdef NDEBUG
    GTEST_SKIP() << "asserts are compiled out (NDEBUG); covered in a -UNDEBUG build";
#else
    std::vector<float> buf(30, 0.0f);
    KernelMatrixView<float, MatrixFormat::Dense> kv(buf.data(), 6, 5, 6);
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";  // earlier cases started SYCL threads
    EXPECT_DEATH((void)kv(Slice(3, 3), Slice()), "");
    EXPECT_DEATH((void)kv(2, Slice(4, 4)), "");
    EXPECT_DEATH((void)kv(Slice(1, 1), 0), "");
#endif
}
