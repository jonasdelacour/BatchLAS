#include <gtest/gtest.h>

#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/sycl_interop.hh>   // for the "Device-resident operands" block below

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
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
// Every C++ code block in that document appears here, in the document's order,
// along with the APIs its prose names. Nothing calls this -- it only has to
// compile. A signature change that invalidates a documented call site must fail
// the build rather than rot in the document. Write the call exactly as the
// document writes it, template arguments included, or the guard stops guarding
// the document. When you add a block to the document, add it here too.
[[maybe_unused]] void docs_cpp_api_examples(Queue& ctx,
                                            const Dense& A,
                                            const Dense& B,
                                            const Dense& C,
                                            Span<float> W,
                                            Span<int64_t> pivots,
                                            Span<std::byte> my_span,
                                            Span<float> S,
                                            const Dense& U,
                                            const Dense& Vh) {
    // "The short version"
    {
        const int n = 8, batch = 4;
        Matrix<float> As(n, n, batch), Bs(n, n, batch), Cs(n, n, batch);
        As.view().fill_random(ctx, /*hermitian=*/false, /*seed=*/1);
        Bs.view().fill_random(ctx, /*hermitian=*/false, /*seed=*/2);
        gemm(ctx, As.view(), Bs.view(), Cs.view(), {.alpha = 2.0f});
        ctx.wait();
        [[maybe_unused]] float c00 = Cs(0, 0, 0);
    }

    // "The short template spelling"
    {
        const int n = 4, batch = 2, nnz = 6;
        Matrix<float> As(n, n, batch);
        [[maybe_unused]] MatrixView<float> Vs = As.view();
        [[maybe_unused]] Matrix<std::complex<float>> Zs(n, n, batch);
        [[maybe_unused]] Matrix<float, MatrixFormat::CSR> Ss(n, n, NonZeros{nnz}, batch);
        [[maybe_unused]] Vector<float> y(4);
        static_assert(std::is_same_v<Matrix<float>, Matrix<float, MatrixFormat::Dense>>);
        static_assert(std::is_same_v<MatrixView<float>, MatrixView<float, MatrixFormat::Dense>>);
        static_assert(std::is_same_v<VectorView<float>, VectorView<>>);
    }

    // "Devices and queues"
    {
        auto gpus = Device::get_devices(DeviceType::GPU);
        for (const auto& d : gpus) std::cout << d.get_name() << '\n';
        Queue second(gpus.at(1));
        Queue cpu("cpu");

        Device device = Device::default_device();
        Queue q1;
        Queue q2(device);
        Queue ooo(device, /*in_order=*/false);
        Queue pinned(device, Backend::NETLIB);
        Queue pinned_ooo(device, Backend::NETLIB, /*in_order=*/false);
        Queue sibling(q2, /*in_order=*/true);
    }

    // "What the operations compute" -- mirroring the other triangle
    {
        const int n = 4, k = 3, batch = 2;
        Matrix<float> Asy(n, k, batch), Csy(n, n, batch);
        syrk(ctx, Asy.view(), Csy.view(), {.uplo = Uplo::Lower}).wait();
        Csy.view().symmetrize(ctx, Uplo::Lower).wait();
        Csy.view().hermitize(ctx, Uplo::Lower);
    }

    // "Which type each parameter takes"
    {
        const int m = 6, n = 4, batch = 2;
        [[maybe_unused]] UnifiedVector<int64_t> piv(n * batch);
        [[maybe_unused]] UnifiedVector<float> tau(std::min(m, n) * batch);
        UnifiedVector<float> w(n * batch);
        [[maybe_unused]] Span<float> ws_span = w.to_span();

        Vector<float> x(n, /*batch_size=*/batch), y(m, batch);
        gemv(ctx, A, x.view(), y.view(), {.alpha = 1.0f});

        // The inc/stride argument orders are opposite, and the document says so.
        [[maybe_unused]] Vector<float> v(n, /*batch_size=*/batch, /*stride=*/n, /*inc=*/1);
        [[maybe_unused]] VectorView<float> vv(x.view().data_ptr(), n, /*batch_size=*/batch,
                                              /*inc=*/1, /*stride=*/n);
        [[maybe_unused]] auto vz = Vector<float>::zeros(n, batch);
        [[maybe_unused]] auto vo = Vector<float>::ones(n, batch);
        [[maybe_unused]] auto ve = Vector<float>::standard_basis(n, /*index=*/0, batch);
    }

    // "Column-major, always" -- the element address the document spells out
    {
        const int n = 4, batch = 2;
        Matrix<float> M(n, n, batch);
        MatrixView<float> V = M.view();
        [[maybe_unused]] float x = V.data_ptr()[1 * V.stride() + 2 * V.ld() + 3];
        [[maybe_unused]] float m = M(3, 2, 1);
        [[maybe_unused]] float a = V.at(3, 2, 1);
        [[maybe_unused]] float p = V(3, 2, 1);
    }

    // "Getting host data in"
    {
        const int n = 4, batch = 2;
        std::vector<float> host(static_cast<size_t>(n) * n * batch);
        Matrix<float> Ah(Span<const float>(host.data(), host.size()),
                         n, n, /*ld=*/n, /*stride=*/0, /*batch_size=*/batch);

        [[maybe_unused]] auto R = Matrix<float>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/7);
        [[maybe_unused]] auto I = Matrix<float>::Identity(n, batch);
        [[maybe_unused]] auto Z = Matrix<float>::Zeros(n, n, batch);

        Matrix<float> Bh(n, n, batch), dst(n, n, batch);
        UnifiedVector<float> d(n);
        Bh.view().fill_diagonal(ctx, d.to_span());
        Bh.view().fill_zeros(ctx);
        MatrixView<float>::copy(ctx, dst.view(), Bh.view());

        // "`Random` is deterministic"
        [[maybe_unused]] auto R0 = Matrix<float>::Random(n, n, false, batch);
        [[maybe_unused]] auto R1 = Matrix<float>::Random(n, n, /*hermitian=*/false, batch, /*seed=*/1);
        [[maybe_unused]] auto R2 = Matrix<float>::Random(n, n, /*hermitian=*/false, batch, /*seed=*/2);
    }

    // "Row-major source data" -- the packed adoption spelling
    {
        const int rows = 3, cols = 4;
        std::vector<float> src(static_cast<size_t>(rows) * cols);
        Matrix<float> Arm(Span<const float>(src.data(), src.size()), rows, cols, /*ld=*/0);
        [[maybe_unused]] auto col_major = Arm.to_column_major();
        [[maybe_unused]] auto row_major = col_major.to_row_major();
        [[maybe_unused]] auto row_major_q = col_major.to_row_major(ctx);
    }

    // "Row-major source data" -- a padded row pitch
    {
        const int rows = 4, cols = 3, batch = 2, p = 5;
        Matrix<float> holder(rows, cols, batch, /*ld=*/rows, /*stride=*/(rows - 1) * p + cols);
        [[maybe_unused]] auto col_major = holder.to_column_major(p);
        [[maybe_unused]] auto col_major_q = holder.to_column_major(ctx, p);
    }

    // "Where the memory has to live: the USM contract" -- this call throws at
    // run time, which is the point of the block; it only has to compile here.
    {
        const int n = 4, batch = 2;
        std::vector<float> ha(n * n * batch), hb(n * n * batch), hc(n * n * batch);
        MatrixView<float> Au(ha.data(), n, n, n, n * n, batch);
        MatrixView<float> Bu(hb.data(), n, n, n, n * n, batch);
        MatrixView<float> Cu(hc.data(), n, n, n, n * n, batch);
        gemm(ctx, Au, Bu, Cu, GemmOptions<float>{});
        [[maybe_unused]] bool reachable = ctx.is_device_accessible(ha.data());
    }

    // "Where the memory has to live" -- an argument that addresses no elements
    {
        SyevxParams<float> params;
        syevx(ctx, A, W, size_t(1), my_span, JobType::NoEigenVectors,
              MatrixView<float>(), params);
    }

    // "Device-resident operands"
    {
        const int n = 4, batch = 2;
        std::vector<float> host(static_cast<size_t>(n) * n * batch);
        auto& q = batchlas::sycl_queue(ctx);
        const size_t elems = static_cast<size_t>(n) * n * batch;

        float* dA = sycl::malloc_device<float>(elems, q);
        float** pA = sycl::malloc_device<float*>(batch, q);
        q.memcpy(dA, host.data(), elems * sizeof(float)).wait();

        [[maybe_unused]] MatrixView<float> Ad(dA, n, n, /*ld=*/n, /*stride=*/n * n, batch,
                                              /*data_ptrs=*/pA);
        sycl::free(pA, q);
        sycl::free(dA, q);
    }

    // "Row-major data: the operand swap"
    {
        const int m = 3, k = 4, n = 5;
        Matrix<float> RA(k, m), RB(n, k), RC(n, m);
        MatrixView<float> At(RA.view().data_ptr(), k, m, k);
        MatrixView<float> Bt(RB.view().data_ptr(), n, k, n);
        MatrixView<float> Ct(RC.view().data_ptr(), n, m, n);
        gemm(ctx, Bt, At, Ct, GemmOptions<float>{});
    }

    // "The CSR non-zero count has its own type" -- the owning and from-data
    // constructors of both formats, in the order the document lists them.
    {
        const int rows = 4, cols = 4, batch_size = 2, ld = 4, stride = 16, nnz = 6;
        const int matrix_stride = nnz, offset_stride = rows + 1;
        std::vector<float> data(static_cast<size_t>(stride) * batch_size);
        std::vector<float> values(static_cast<size_t>(nnz) * batch_size);
        std::vector<int> row_offsets(static_cast<size_t>(offset_stride) * batch_size);
        std::vector<int> col_indices(static_cast<size_t>(nnz) * batch_size);

        Matrix<float> Dd(rows, cols, batch_size, ld, stride);
        Matrix<float, MatrixFormat::CSR> Sd(rows, cols, NonZeros{nnz}, batch_size);

        Matrix<float> D(data.data(), rows, cols, ld, stride, batch_size);
        Matrix<float, MatrixFormat::CSR> S(values.data(), row_offsets.data(), col_indices.data(),
                                           rows, cols, NonZeros{nnz},
                                           matrix_stride, offset_stride, batch_size);
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

    // "Which spelling each entry point takes"
    getrf(ctx, A, pivots);
    with_backend(ctx, [&](auto Back) {
        constexpr Backend Bk = Back.value;
        auto ws = ctx.workspace(gesvd_buffer_size<Bk, float>(
                                    ctx, A, S, U, Vh,
                                    SvdVectors::All, SvdVectors::All));
        gesvd<Bk, float>(ctx, A, S, U, Vh,
                         SvdVectors::All, SvdVectors::All, ws.span());
    });

    // "Workspaces come from the queue's arena"
    potrf(ctx, A, {.uplo = Uplo::Lower});
    with_backend(ctx, [&](auto Back) {
        constexpr Backend Bk = Back.value;
        UnifiedVector<std::byte> ws(potrf_buffer_size<Bk, float>(ctx, A, Uplo::Lower));
        potrf<Bk, float>(ctx, A, Uplo::Lower, ws.to_span());
        ctx.wait();
    });
    auto lease = ctx.workspace(1024);
    [[maybe_unused]] Span<std::byte> bytes = lease.span();
    [[maybe_unused]] auto capacity = ctx.workspace_capacity();
    potrf(ctx, A, {.uplo = Uplo::Lower}, my_span);
    lease.release();                       // before reassigning a live lease
    lease = ctx.workspace(2048);
    lease.release();
    [[maybe_unused]] bool trimmed = ctx.trim_workspace();

    // An empty option struct passed with an explicit workspace names its type.
    // The bare-`{}` spelling is absent because it is ill-formed; the deleted
    // overload that makes it so is pinned by tests/options_api_tests.cc.
    with_backend(ctx, [&](auto Back) {
        potrf<Back.value>(ctx, A, PotrfOptions{}, my_span);
    });

    // "Synchronisation and threading"
    {
        Event e = gemm(ctx, A, B, C, GemmOptions<float>{});
        e.wait();
        ctx.wait();
        ctx.wait_and_throw();
        ctx.attach_to_current_thread();
    }

    // "Interop with CUDA and with your own SYCL"
    {
        [[maybe_unused]] void* stream = ctx.native_handle();
        [[maybe_unused]] Event external = ctx.create_event_after_external_work();

        sycl::queue& my_queue = batchlas::sycl_queue(ctx);
        sycl::event mine = my_queue.ext_oneapi_submit_barrier();
        Event e = batchlas::event_from_sycl(mine);
        ctx.enqueue(e);                                        // `enqueue` takes an lvalue
        batchlas::potrf(ctx, A, {.uplo = Uplo::Lower});

        // ... and in the other direction:
        my_queue.ext_oneapi_submit_barrier({batchlas::sycl_event(ctx.get_event())});
    }

    // "The linalg convenience layer"
    [[maybe_unused]] auto product = linalg::matmul(ctx, A, B);
    [[maybe_unused]] auto chol = linalg::cholesky(ctx, A);
    [[maybe_unused]] auto solution = linalg::solve(ctx, A, B);
    [[maybe_unused]] auto values = linalg::eigvalsh(ctx, A);
    [[maybe_unused]] auto pairs = linalg::eigh(ctx, A);
    [[maybe_unused]] auto sum = linalg::add(ctx, A, B);
    [[maybe_unused]] auto difference = linalg::subtract(ctx, A, B);
    [[maybe_unused]] auto hadamard = linalg::multiply(ctx, A, B);
    [[maybe_unused]] auto quotient = linalg::divide(ctx, A, B);
    [[maybe_unused]] auto scaled_copy = linalg::scaled(ctx, A, 2.0f);
    linalg::scale(ctx, A, 2.0f);
    linalg::axpby_into(ctx, 2.0f, A, 3.0f, B, C);
    linalg::add_into(ctx, A, B, C);
    linalg::subtract_into(ctx, A, B, C);
    linalg::multiply_into(ctx, A, B, C);
    linalg::divide_into(ctx, A, B, C);
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
