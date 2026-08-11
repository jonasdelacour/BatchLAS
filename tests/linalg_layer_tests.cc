#include <gtest/gtest.h>

#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>

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

// Compile guard for docs/cpp-api.md.
//
// Most code blocks in that document appear here. Nothing calls this -- it only
// has to compile. A signature change that invalidates a documented call site
// must fail the build rather than rot in the document. When you add a block to
// the document, add it here too.
[[maybe_unused]] void docs_cpp_api_examples(Queue& ctx,
                                            const Dense& A,
                                            const Dense& B,
                                            const Dense& C,
                                            Span<float> W,
                                            Span<int64_t> pivots,
                                            Span<std::byte> my_span) {
    // "Getting host data in"
    {
        const int n = 4, batch = 2;
        std::vector<float> host(static_cast<size_t>(n) * n * batch);
        Matrix<float, MatrixFormat::Dense> Ah(
            Span<const float>(host.data(), host.size()),
            n, n, /*ld=*/n, /*stride=*/0, /*batch_size=*/batch);

        [[maybe_unused]] auto R = Matrix<float, MatrixFormat::Dense>::Random(n, n, true, batch);
        [[maybe_unused]] auto I = Matrix<float, MatrixFormat::Dense>::Identity(n, batch);
        [[maybe_unused]] auto Z = Matrix<float, MatrixFormat::Dense>::Zeros(n, n, batch);

        Matrix<float, MatrixFormat::Dense> Bh(n, n, batch), dst(n, n, batch);
        UnifiedVector<float> d(n);
        Bh.view().fill_diagonal(ctx, d.to_span());
        Bh.view().fill_zeros(ctx);
        MatrixView<float, MatrixFormat::Dense>::copy(ctx, dst.view(), Bh.view());
    }

    // "The backend comes from the Queue"
    Queue host(Device::default_device(), Backend::NETLIB);
    if (Queue::backend_available(Backend::CUDA)) ctx.set_backend(Backend::CUDA);
    [[maybe_unused]] Backend resolved = ctx.backend();
    with_backend(ctx, [&](auto Back) {
        constexpr Backend Bk = Back.value;
        gemm<Bk>(ctx, A, B, C, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
    });

    // "Options are structs with defaults"
    gemm(ctx, A, B, C, {.alpha = 2.0f, .transA = Transpose::Trans});
    syev(ctx, A, W, {.jobz = JobType::NoEigenVectors});
    getrs(ctx, A, C, pivots, {.trans = Transpose::Trans});

    // "Workspaces come from the queue's arena"
    potrf(ctx, A, {.uplo = Uplo::Lower});
    auto lease = ctx.workspace(1024);
    [[maybe_unused]] Span<std::byte> bytes = lease.span();
    [[maybe_unused]] auto capacity = ctx.workspace_capacity();
    potrf(ctx, A, {.uplo = Uplo::Lower}, my_span);

    // "Options are structs with defaults" -- an empty option struct passed
    // with an explicit workspace names its type. The bare-`{}` spelling is
    // absent because it is ill-formed; the deleted overload that makes it so is
    // pinned by static_asserts in tests/options_api_tests.cc.
    with_backend(ctx, [&](auto Back) {
        potrf<Back.value>(ctx, A, PotrfOptions{}, my_span);
    });

    // "The linalg convenience layer"
    [[maybe_unused]] auto product = linalg::matmul(ctx, A, B);
    [[maybe_unused]] auto chol = linalg::cholesky(ctx, A);
    [[maybe_unused]] auto solution = linalg::solve(ctx, A, B);
    [[maybe_unused]] auto values = linalg::eigvalsh(ctx, A);
    [[maybe_unused]] auto pairs = linalg::eigh(ctx, A);
    [[maybe_unused]] auto sum = linalg::add(ctx, A, B);
    [[maybe_unused]] auto hadamard = linalg::multiply(ctx, A, B);
    linalg::scale<float>(ctx, A, 2.0f);
    linalg::axpby_into<float>(ctx, 2.0f, A, 3.0f, B, C);
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
