#include <gtest/gtest.h>

#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>

#include <cmath>
#include <vector>

using namespace batchlas;

namespace {

using Dense = MatrixView<float, MatrixFormat::Dense>;

float& at(const Dense& v, int i, int j, int b) {
    return v.data_ptr()[b * v.stride() + j * v.ld() + i];
}

Matrix<float, MatrixFormat::Dense> from_fn(int rows, int cols, int batch, auto fn) {
    Matrix<float, MatrixFormat::Dense> m(rows, cols, batch);
    auto v = m.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) at(v, i, j, b) = fn(i, j, b);
    return m;
}

// Diagonally dominant, so LU and Cholesky are both well-posed.
Matrix<float, MatrixFormat::Dense> spd(int n, int batch) {
    return from_fn(n, n, batch, [n](int i, int j, int b) {
        return i == j ? float(n + b + 3) : 1.0f / (1.0f + std::abs(i - j));
    });
}

}  // namespace

TEST(LinalgLayer, ElementwiseMatchesScalarReference) {
    Queue q;
    const int rows = 7, cols = 5, batch = 3;  // deliberately non-square, batched

    auto A = from_fn(rows, cols, batch, [](int i, int j, int b) { return 1.0f + i + 2.0f * j + b; });
    auto B = from_fn(rows, cols, batch, [](int i, int j, int b) { return 2.0f + j + 0.5f * i - b; });

    auto S = linalg::add(q, A.view(), B.view());
    auto D = linalg::subtract(q, A.view(), B.view());
    auto P = linalg::multiply(q, A.view(), B.view());
    auto Q = linalg::divide(q, A.view(), B.view());
    auto K = linalg::scaled(q, A.view(), 3.0f);
    q.wait();

    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                const float a = at(A.view(), i, j, b);
                const float bb = at(B.view(), i, j, b);
                ASSERT_FLOAT_EQ(at(S.view(), i, j, b), a + bb) << "add " << i << "," << j;
                ASSERT_FLOAT_EQ(at(D.view(), i, j, b), a - bb) << "sub " << i << "," << j;
                ASSERT_FLOAT_EQ(at(P.view(), i, j, b), a * bb) << "mul " << i << "," << j;
                ASSERT_FLOAT_EQ(at(Q.view(), i, j, b), a / bb) << "div " << i << "," << j;
                ASSERT_FLOAT_EQ(at(K.view(), i, j, b), 3.0f * a) << "scaled " << i << "," << j;
            }
}

// multiply is elementwise, not matrix multiplication. Easy to conflate, and the
// shapes agree for square operands, so nothing would catch it but a value check.
TEST(LinalgLayer, MultiplyIsHadamardNotMatmul) {
    Queue q;
    const int n = 3;
    auto A = from_fn(n, n, 1, [](int i, int j, int) { return float(i + 2 * j + 1); });
    auto B = from_fn(n, n, 1, [](int i, int j, int) { return float(j - i + 4); });

    auto had = linalg::multiply(q, A.view(), B.view());
    auto mm = linalg::matmul(q, A.view(), B.view());
    q.wait();

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            ASSERT_FLOAT_EQ(at(had.view(), i, j, 0),
                            at(A.view(), i, j, 0) * at(B.view(), i, j, 0));

    // ...and matmul really is the matrix product.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k) acc += at(A.view(), i, k, 0) * at(B.view(), k, j, 0);
            ASSERT_NEAR(at(mm.view(), i, j, 0), acc, 1e-4f) << "matmul at " << i << "," << j;
        }
}

TEST(LinalgLayer, ScaleIsInPlaceAndAxpbyAccumulates) {
    Queue q;
    const int rows = 4, cols = 6, batch = 2;
    auto A = from_fn(rows, cols, batch, [](int i, int j, int b) { return 1.0f + i - j + b; });
    std::vector<float> before;
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) before.push_back(at(A.view(), i, j, b));

    linalg::scale<float>(q, A.view(), -2.0f);
    q.wait();

    size_t k = 0;
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                ASSERT_FLOAT_EQ(at(A.view(), i, j, b), -2.0f * before[k++]);

    // C = 2A + 3B
    auto X = from_fn(rows, cols, batch, [](int i, int j, int) { return float(i + j + 1); });
    auto Y = from_fn(rows, cols, batch, [](int i, int j, int) { return float(2 * i - j); });
    Matrix<float, MatrixFormat::Dense> C(rows, cols, batch);
    linalg::axpby_into<float>(q, 2.0f, X.view(), 3.0f, Y.view(), C.view());
    q.wait();

    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                ASSERT_FLOAT_EQ(at(C.view(), i, j, b),
                                2.0f * at(X.view(), i, j, b) + 3.0f * at(Y.view(), i, j, b));
}

TEST(LinalgLayer, MismatchedShapesAreRejected) {
    Queue q;
    auto A = from_fn(4, 4, 1, [](int, int, int) { return 1.0f; });
    auto B = from_fn(4, 5, 1, [](int, int, int) { return 1.0f; });
    EXPECT_THROW(linalg::add_into<float>(q, A.view(), B.view(), A.view()), std::invalid_argument);
}

// The value-returning wrappers must not modify their inputs -- that is the whole
// difference between them and the out-parameter forms.
TEST(LinalgLayer, ValueReturningWrappersLeaveInputsAlone) {
    Queue q;
    const int n = 12, batch = 2;
    auto A = spd(n, batch);

    std::vector<float> original;
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) original.push_back(at(A.view(), i, j, b));

    auto L = linalg::cholesky(q, A.view());
    auto w = linalg::eigvalsh(q, A.view());
    auto e = linalg::eigh(q, A.view());
    q.wait();

    size_t k = 0;
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                ASSERT_FLOAT_EQ(at(A.view(), i, j, b), original[k++]) << "input was modified";

    // eigvalsh and eigh must agree on the eigenvalues.
    ASSERT_EQ(w.size(), e.values.size());
    for (size_t i = 0; i < w.size(); ++i) ASSERT_NEAR(w[i], e.values[i], 1e-4f) << "at " << i;
}

// L L^T must reproduce the lower triangle of A.
TEST(LinalgLayer, CholeskyFactorReproducesInput) {
    Queue q;
    const int n = 8, batch = 2;
    auto A = spd(n, batch);
    auto L = linalg::cholesky(q, A.view(), Uplo::Lower);
    q.wait();

    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {  // lower triangle only
                float acc = 0.0f;
                for (int k = 0; k <= std::min(i, j); ++k)
                    acc += at(L.view(), i, k, b) * at(L.view(), j, k, b);
                ASSERT_NEAR(acc, at(A.view(), i, j, b), 1e-3f)
                    << "L L^T at (" << i << "," << j << ") batch " << b;
            }
}

// A X == B, checked by multiplying back.
TEST(LinalgLayer, SolveSatisfiesTheSystem) {
    Queue q;
    const int n = 10, nrhs = 3, batch = 2;
    auto A = spd(n, batch);
    auto B = from_fn(n, nrhs, batch, [](int i, int j, int b) { return float(1 + i - 2 * j + b); });

    auto X = linalg::solve(q, A.view(), B.view());
    q.wait();

    auto AX = linalg::matmul(q, A.view(), X.view());
    q.wait();

    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                ASSERT_NEAR(at(AX.view(), i, j, b), at(B.view(), i, j, b), 1e-2f)
                    << "A X != B at (" << i << "," << j << ") batch " << b;
}

// A V == V diag(w).
TEST(LinalgLayer, EighProducesConsistentEigenpairs) {
    Queue q;
    const int n = 8, batch = 2;
    auto A = spd(n, batch);
    auto e = linalg::eigh(q, A.view());
    q.wait();

    auto AV = linalg::matmul(q, A.view(), e.vectors.view());
    q.wait();

    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j) {
            const float lambda = e.values[b * n + j];
            for (int i = 0; i < n; ++i)
                ASSERT_NEAR(at(AV.view(), i, j, b), lambda * at(e.vectors.view(), i, j, b), 1e-3f)
                    << "eigenpair " << j << " batch " << b << " row " << i;
        }
}

// The convenience layer leases from the arena; repeated calls must not grow it.
TEST(LinalgLayer, RepeatedCallsDoNotGrowTheArena) {
    Queue q;
    const int n = 10, batch = 2;
    auto A = spd(n, batch);
    auto B = from_fn(n, 2, batch, [](int i, int j, int) { return float(i + j + 1); });

    auto warm = linalg::solve(q, A.view(), B.view());
    q.wait();
    const size_t settled = q.workspace_capacity();

    for (int i = 0; i < 16; ++i) {
        auto X = linalg::solve(q, A.view(), B.view());
        q.wait();
    }
    EXPECT_EQ(q.workspace_capacity(), settled);
}
