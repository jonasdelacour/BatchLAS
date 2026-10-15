#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/sycl_interop.hh>   // for the "Device-resident operands" block below
#include <batchlas/verify/reference.hh>

#include <tests/eigen_verify.hh>
#include <tests/test_utils.hh>

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
        (void)As.fill_random(ctx, /*hermitian=*/false, /*seed=*/1);
        (void)Bs.fill_random(ctx, /*hermitian=*/false, /*seed=*/2);
        (void)gemm(ctx, As, Bs, Cs, {.alpha = 2.0f});
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
        syrk(ctx, Asy, Csy, {.uplo = Uplo::Lower}).wait();
        Csy.symmetrize(ctx, Uplo::Lower).wait();
        (void)Csy.hermitize(ctx, Uplo::Lower);
    }

    // "Which type each parameter takes"
    {
        const int m = 6, n = 4, batch = 2;
        [[maybe_unused]] UnifiedVector<int64_t> piv(n * batch);
        [[maybe_unused]] UnifiedVector<float> tau(std::min(m, n) * batch);
        UnifiedVector<float> w(n * batch);
        [[maybe_unused]] Span<float> ws_span = w.to_span();

        Vector<float> x(n, /*batch_size=*/batch), y(m, batch);
        (void)gemv(ctx, A, x, y, {.alpha = 1.0f});

        // The inc/stride argument orders are opposite. Vector now names both with tag
        // types, so the two spellings can no longer be transliterated into each other.
        [[maybe_unused]] Vector<float> v(n, /*batch_size=*/batch, Stride{n}, Inc{1});
        [[maybe_unused]] VectorView<float> vv(x.data_ptr(), n, /*batch_size=*/batch,
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
        (void)Bh.fill_diagonal(ctx, d.to_span());
        (void)Bh.fill_zeros(ctx);
        (void)MatrixView<float>::copy(ctx, dst, Bh);

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
        (void)gemm(ctx, Au, Bu, Cu, GemmOptions<float>{});
        [[maybe_unused]] bool reachable = ctx.is_device_accessible(ha.data());
    }

    // "Where the memory has to live" -- an argument that addresses no elements
    {
        SyevxParams<float> params;
        (void)syevx(ctx, A, W, size_t(1), my_span, JobType::NoEigenVectors,
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
        MatrixView<float> At(RA.data_ptr(), k, m, k);
        MatrixView<float> Bt(RB.data_ptr(), n, k, n);
        MatrixView<float> Ct(RC.data_ptr(), n, m, n);
        (void)gemm(ctx, Bt, At, Ct, GemmOptions<float>{});
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
        (void)gemm<Bk>(ctx, A, B, C, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
    });

    // "Options are structs with defaults"
    (void)gemm(ctx, A, B, C, {.alpha = 2.0f, .transA = Transpose::Trans});
    (void)syev(ctx, A, W, {.jobz = JobType::NoEigenVectors});
    (void)getrs(ctx, A, C, pivots, {.trans = Transpose::Trans});

    // "Which spelling each entry point takes"
    (void)getrf(ctx, A, pivots);
    with_backend(ctx, [&](auto Back) {
        constexpr Backend Bk = Back.value;
        auto ws = ctx.workspace(gesvd_buffer_size<Bk, float>(
                                    ctx, A, S, U, Vh,
                                    SvdVectors::All, SvdVectors::All));
        (void)gesvd<Bk, float>(ctx, A, S, U, Vh,
                         SvdVectors::All, SvdVectors::All, ws.span());
    });

    // "Workspaces come from the queue's arena"
    (void)potrf(ctx, A, {.uplo = Uplo::Lower});
    with_backend(ctx, [&](auto Back) {
        constexpr Backend Bk = Back.value;
        UnifiedVector<std::byte> ws(potrf_buffer_size<Bk, float>(ctx, A, Uplo::Lower));
        (void)potrf<Bk, float>(ctx, A, Uplo::Lower, ws.to_span());
        ctx.wait();
    });
    auto lease = ctx.workspace(1024);
    [[maybe_unused]] Span<std::byte> bytes = lease.span();
    [[maybe_unused]] auto capacity = ctx.workspace_capacity();
    (void)potrf(ctx, A, {.uplo = Uplo::Lower}, my_span);
    lease.release();                       // before reassigning a live lease
    lease = ctx.workspace(2048);
    lease.release();
    [[maybe_unused]] bool trimmed = ctx.trim_workspace();

    // An empty option struct passed with an explicit workspace names its type.
    // The bare-`{}` spelling is absent because it is ill-formed; the deleted
    // overload that makes it so is pinned by tests/options_api_tests.cc.
    with_backend(ctx, [&](auto Back) {
        (void)potrf<Back.value>(ctx, A, PotrfOptions{}, my_span);
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
#if BATCHLAS_SYCL_IMPL_DPCPP
        sycl::event mine = my_queue.ext_oneapi_submit_barrier();
#else
        sycl::event mine = my_queue.single_task([] {});        // acpp: no ext_oneapi_submit_barrier
#endif
        Event e = batchlas::event_from_sycl(mine);
        ctx.enqueue(e);                                        // `enqueue` takes an lvalue
        (void)batchlas::potrf(ctx, A, {.uplo = Uplo::Lower});

        // ... and in the other direction:
#if BATCHLAS_SYCL_IMPL_DPCPP
        my_queue.ext_oneapi_submit_barrier({batchlas::sycl_event(ctx.get_event())});
#else
        my_queue.submit([&](sycl::handler& h) {
            h.depends_on(batchlas::sycl_event(ctx.get_event()));
            h.single_task([] {});
        });
#endif
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
    (void)linalg::scale(ctx, A, 2.0f);
    (void)linalg::axpby_into(ctx, 2.0f, A, 3.0f, B, C);
    (void)linalg::add_into(ctx, A, B, C);
    (void)linalg::subtract_into(ctx, A, B, C);
    (void)linalg::multiply_into(ctx, A, B, C);
    (void)linalg::divide_into(ctx, A, B, C);

    // "The linalg convenience layer" -- the rest of the layer
    [[maybe_unused]] auto svd_result = linalg::svd(ctx, A);
    [[maybe_unused]] auto lu_result = linalg::lu(ctx, A);
    [[maybe_unused]] auto inverse = linalg::inv(ctx, A);
    [[maybe_unused]] auto transposed = linalg::transpose(ctx, A);
    [[maybe_unused]] auto norms = linalg::norm(ctx, A);
    [[maybe_unused]] auto conds = linalg::cond(ctx, A);
    [[maybe_unused]] auto upper = linalg::triu(ctx, A);
    [[maybe_unused]] auto strict_lower = linalg::tril(ctx, A, -1);
    linalg::matmul(ctx, A, B, {.alpha = 2.0f});

    // "The linalg convenience layer" -- the table
    {
        Vector<float> xv(4, 2);
        [[maybe_unused]] auto y = linalg::matvec(ctx, A, xv);
        [[maybe_unused]] auto g = linalg::gram(ctx, A);
        [[maybe_unused]] auto outer = linalg::gram(ctx, A, Transpose::NoTrans);
        [[maybe_unused]] auto t = linalg::solve_triangular(ctx, A, B, {.uplo = Uplo::Upper});
        [[maybe_unused]] auto spd_solution = linalg::solve_spd(ctx, A, B);
        [[maybe_unused]] auto by_index =
            linalg::eigh(ctx, A, SyevxParams<float>{.select = SyevxSelect::Index, .il = 0, .iu = 1});
        [[maybe_unused]] auto by_value =
            linalg::eigvalsh(ctx, A, SyevxParams<float>{.select = SyevxSelect::Value, .vl = 0.0f, .vu = 1.0f});
        [[maybe_unused]] auto mixed =
            linalg::svd(ctx, A, GesvdOptions{.jobu = SvdVectors::Thin, .jobvh = SvdVectors::All});
        [[maybe_unused]] auto hermitian = linalg::svd(ctx, A, GesvdOptions{.hermitian_uplo = Uplo::Lower});
    }
}

// What the layer must NOT accept. Each of these is spelled as a concept rather
// than a bare `!requires(...)` in the static_assert: a requires-expression only
// swallows errors that come from template substitution, so an ill-formed call
// written with concrete types in a non-template context is a hard error and the
// guard would fail to compile instead of passing.
template <typename T>
concept MatmulOptionsHasBeta = requires(linalg::MatmulOptions<T> o) { o.beta; };

template <typename T>
concept HasLinalgTranspose = requires(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    linalg::transpose(ctx, A);
};

template <typename T>
concept HasLinalgCond = requires(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    linalg::cond(ctx, A);
};

template <typename T>
concept MatmulTakesOptionList = requires(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    linalg::matmul(ctx, A, A, {.alpha = T(2)});
};

// matmul allocates its result, so beta has no meaning: it used to be spellable
// through GemmOptions and silently discarded.
static_assert(!MatmulOptionsHasBeta<float>,
              "MatmulOptions must not carry a beta: matmul allocates its result");
static_assert(MatmulTakesOptionList<float>,
              "matmul must still accept a designated-initialiser option list");

// transpose and cond are constrained to real scalars because the entry points
// behind them are instantiated for real scalars only. Without the constraint
// these calls would compile and fail at link time.
static_assert(HasLinalgTranspose<float> && HasLinalgTranspose<double>);
static_assert(!HasLinalgTranspose<std::complex<float>>,
              "complex transpose must be rejected: transpose_impl does not conjugate");
static_assert(HasLinalgCond<float> && HasLinalgCond<double>);
static_assert(!HasLinalgCond<std::complex<float>>,
              "complex cond must be rejected: there is no complex instantiation behind it");

}  // namespace

TEST(LinalgLayer, ElementwiseMatchesScalarReference) {
    Queue q;
    const int rows = 7, cols = 5, batch = 3;  // deliberately non-square, batched

    auto A = from_fn(rows, cols, batch, [](int i, int j, int b) { return 1.0f + i + 2.0f * j + b; });
    auto B = from_fn(rows, cols, batch, [](int i, int j, int b) { return 2.0f + j + 0.5f * i - b; });

    auto S = linalg::add(q, A, B);
    auto D = linalg::subtract(q, A, B);
    auto P = linalg::multiply(q, A, B);
    auto Q = linalg::divide(q, A, B);
    auto K = linalg::scaled(q, A, 3.0f);
    q.wait();

    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                const float a = at(A, i, j, b);
                const float bb = at(B, i, j, b);
                ASSERT_FLOAT_EQ(at(S, i, j, b), a + bb) << "add " << i << "," << j;
                ASSERT_FLOAT_EQ(at(D, i, j, b), a - bb) << "sub " << i << "," << j;
                ASSERT_FLOAT_EQ(at(P, i, j, b), a * bb) << "mul " << i << "," << j;
                ASSERT_FLOAT_EQ(at(Q, i, j, b), a / bb) << "div " << i << "," << j;
                ASSERT_FLOAT_EQ(at(K, i, j, b), 3.0f * a) << "scaled " << i << "," << j;
            }
}

// multiply is elementwise, not matrix multiplication. Easy to conflate, and the
// shapes agree for square operands, so nothing would catch it but a value check.
TEST(LinalgLayer, MultiplyIsHadamardNotMatmul) {
    Queue q;
    const int n = 3;
    auto A = from_fn(n, n, 1, [](int i, int j, int) { return float(i + 2 * j + 1); });
    auto B = from_fn(n, n, 1, [](int i, int j, int) { return float(j - i + 4); });

    auto had = linalg::multiply(q, A, B);
    auto mm = linalg::matmul(q, A, B);
    q.wait();

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            ASSERT_FLOAT_EQ(at(had, i, j, 0),
                            at(A, i, j, 0) * at(B, i, j, 0));

    // ...and matmul really is the matrix product.
    EXPECT_VERIFY(float, verify::Check::blas, n,
                  Dense::gemm_backward_error(q, A, Transpose::NoTrans, B, Transpose::NoTrans, mm, mm,
                                             1.0, 0.0, mm.all_items()));
}

TEST(LinalgLayer, ScaleIsInPlaceAndAxpbyAccumulates) {
    Queue q;
    const int rows = 4, cols = 6, batch = 2;
    auto A = from_fn(rows, cols, batch, [](int i, int j, int b) { return 1.0f + i - j + b; });
    std::vector<float> before;
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) before.push_back(at(A, i, j, b));

    (void)linalg::scale<float>(q, A, -2.0f);
    q.wait();

    size_t k = 0;
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                ASSERT_FLOAT_EQ(at(A, i, j, b), -2.0f * before[k++]);

    // C = 2A + 3B
    auto X = from_fn(rows, cols, batch, [](int i, int j, int) { return float(i + j + 1); });
    auto Y = from_fn(rows, cols, batch, [](int i, int j, int) { return float(2 * i - j); });
    Matrix<float, MatrixFormat::Dense> C(rows, cols, batch);
    (void)linalg::axpby_into<float>(q, 2.0f, X, 3.0f, Y, C);
    q.wait();

    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                ASSERT_FLOAT_EQ(at(C, i, j, b),
                                2.0f * at(X, i, j, b) + 3.0f * at(Y, i, j, b));
}

TEST(LinalgLayer, MismatchedShapesAreRejected) {
    Queue q;
    auto A = from_fn(4, 4, 1, [](int, int, int) { return 1.0f; });
    auto B = from_fn(4, 5, 1, [](int, int, int) { return 1.0f; });
    EXPECT_THROW((void)linalg::add_into<float>(q, A, B, A), std::invalid_argument);
}

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using Real = typename base_type<T>::type;

template <typename T>
inline constexpr bool kIsComplex = !std::is_same_v<T, Real<T>>;

template <typename T>
T scalar_of(double re, double im) {
    if constexpr (kIsComplex<T>) {
        return T(Real<T>(re), Real<T>(im));
    } else {
        (void)im;
        return T(Real<T>(re));
    }
}

template <typename T>
T& elem(const MV<T>& v, int i, int j, int b) {
    return v.data_ptr()[static_cast<int64_t>(b) * v.stride() + static_cast<int64_t>(j) * v.ld() + i];
}

// In range and finite, so a kernel that reads padding or the unread triangle accepts it.
template <typename T>
const T kPoison = scalar_of<T>(777.0, -333.0);

enum class Fill { general, hermitian, spd, pivoted };

template <typename T>
Matrix<T> make(Queue& q, int m, int n, int batch, Fill fill, unsigned seed) {
    Matrix<T> A(m, n, batch);
    switch (fill) {
        case Fill::general: (void)A.fill_random(q, false, seed); break;
        case Fill::hermitian: (void)A.fill_random(q, true, seed); break;
        case Fill::spd: (void)A.fill_random_spd(q, seed); break;
        case Fill::pivoted: (void)A.fill_random_pivoted(q, seed); break;
    }
    q.wait();
    return A;
}

// A's values in storage with ld = rows + 3 and a padded batch stride, the padding poisoned.
template <typename T>
Matrix<T> padded(Queue& q, const MV<T>& A) {
    const int ld = A.rows() + 3;
    const int stride = ld * A.cols() + 5;
    const int total = stride * A.batch_size();
    Matrix<T> P(A.rows(), A.cols(), A.batch_size(), ld, stride);
    (void)MV<T>(P.data_ptr(), total, 1, total, total, 1).fill(q, kPoison<T>);
    (void)MV<T>::copy(q, P, A);
    q.wait();
    return P;
}

// A with the triangle opposite `keep` replaced by poison: proves the wrapper reads only `keep`.
template <typename T>
Matrix<T> poison_other_triangle(Queue& q, const MV<T>& A, Uplo keep) {
    Matrix<T> junk(A.rows(), A.cols(), A.batch_size());
    (void)junk.fill(q, kPoison<T>);
    auto kept = keep == Uplo::Upper ? linalg::triu(q, A) : linalg::tril(q, A);
    auto other = keep == Uplo::Upper ? linalg::tril(q, junk, -1) : linalg::triu(q, junk, 1);
    auto P = linalg::add(q, kept, other);
    q.wait();
    return P;
}

void expect_info(const UnifiedVector<int32_t>& info, int batch, int bad, const char* what) {
    ASSERT_EQ(info.size(), size_t(batch)) << what;
    for (int b = 0; b < batch; ++b) {
        if (b == bad) {
            EXPECT_NE(info[b], 0) << what << ": item " << b << " should report failure";
        } else {
            EXPECT_EQ(info[b], 0) << what << ": item " << b;
        }
    }
}

template <typename T>
void expect_vectors_agree(const VectorView<Real<T>>& w, const VectorView<Real<T>>& ref, int n) {
    std::vector<std::vector<double>> r(static_cast<size_t>(w.batch_size()));
    double scale = 0;
    for (int b = 0; b < w.batch_size(); ++b)
        for (int i = 0; i < w.size(); ++i) {
            r[static_cast<size_t>(b)].push_back(double(ref(i, b)));
            scale = std::max(scale, std::fabs(double(ref(i, b))));
        }
    EXPECT_VERIFY(T, verify::Check::values, n, MV<T>::values_error(w, r, scale));
}

}  // namespace

// The value-returning wrappers must not modify their inputs -- that is the whole
// difference between them and the out-parameter forms.
TEST(LinalgLayer, ValueReturningWrappersLeaveInputsAlone) {
    Queue q;
    const int n = 12, nrhs = 3, batch = 2;
    auto A = spd(n, batch);
    auto B = from_fn(n, nrhs, batch, [](int i, int j, int b) { return float(1 + i - 2 * j + b); });
    Vector<float> x(n, batch);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i) x(i, b) = float(i - b);

    std::vector<float> original_A(A.data_ptr(), A.data_ptr() + A.stride() * batch);
    std::vector<float> original_B(B.data_ptr(), B.data_ptr() + B.stride() * batch);
    std::vector<float> original_x(x.data_ptr(), x.data_ptr() + size_t(n) * batch);

    auto L = linalg::cholesky(q, A);
    auto w = linalg::eigvalsh(q, A);
    auto e = linalg::eigh(q, A);
    auto ws = linalg::eigvalsh(q, A, SyevxParams<float>{.select = SyevxSelect::Index, .il = 1, .iu = 4});
    auto s = linalg::svd(q, A);
    auto f = linalg::lu(q, A);
    auto Ai = linalg::inv(q, A);
    auto up = linalg::triu(q, A);
    auto X = linalg::solve(q, A, B);
    auto Xt = linalg::solve(q, A, B, Transpose::Trans);
    auto Y = linalg::solve_spd(q, A, B);
    auto Z = linalg::solve_triangular(q, A, B);
    auto G = linalg::gram(q, A);
    auto y = linalg::matvec(q, A, x);
    q.wait();

    ASSERT_TRUE(std::equal(original_A.begin(), original_A.end(), A.data_ptr())) << "A was modified";
    ASSERT_TRUE(std::equal(original_B.begin(), original_B.end(), B.data_ptr())) << "B was modified";
    ASSERT_TRUE(std::equal(original_x.begin(), original_x.end(), x.data_ptr())) << "x was modified";

    // eigvalsh and eigh must agree on the eigenvalues.
    ASSERT_EQ(w.size(), e.values.size());
    test_utils::expect_eigenvalues_agree<float>(w, e.values, n, batch);
}

// L L^T must reproduce the lower triangle of A.
TEST(LinalgLayer, CholeskyFactorReproducesInput) {
    Queue q;
    const int n = 8, batch = 2;
    auto A = spd(n, batch);
    auto c = linalg::cholesky(q, A, Uplo::Lower);
    q.wait();

    EXPECT_VERIFY(float, verify::Check::factorization, n,
                  Dense::potrf_residual(q, A, c.factor, Uplo::Lower, A.all_items()));
    expect_info(c.info, batch, -1, "cholesky");
}

// A X == B, checked by multiplying back.
TEST(LinalgLayer, SolveSatisfiesTheSystem) {
    Queue q;
    const int n = 10, nrhs = 3, batch = 2;
    auto A = spd(n, batch);
    auto B = from_fn(n, nrhs, batch, [](int i, int j, int b) { return float(1 + i - 2 * j + b); });

    auto s = linalg::solve(q, A, B);
    q.wait();

    EXPECT_VERIFY(float, verify::Check::solve, n, Dense::solve_residual(q, A, s.x, B, A.all_items()));
    expect_info(s.info, batch, -1, "solve");
}

// ---- linalg::solve after the gesv rewire -----------------------------------
//
// `solve` sends Transpose::NoTrans to the fused `gesv` and keeps the hand-composed
// getrf + getrs for Trans/ConjTrans, because gesv has no Transpose parameter. Both arms
// therefore need their own residual. The NoTrans shapes below straddle the sm_89 gesv
// table's fused window (float n <= 32, complex<float> n <= 16, and nrhs <= 4 for both);
// on a CPU queue, or a device without sub-group 32, the fused tier is gated off and every
// shape takes the composed arm -- so these are correctness rows that hold on any device,
// not routing rows.

namespace {

// Diagonally dominant for every batch item, so LU is well-posed without pivot luck.
// The imaginary part is off-diagonal and odd in (i - j), which is what makes
// ConjTrans a different system from Trans.
template <typename T>
Matrix<T> solve_system(int n, int batch) {
    Matrix<T> A(n, n, batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                elem<T>(A, i, j, b) =
                    i == j ? scalar_of<T>(double(n + b + 3), 0.0)
                           : scalar_of<T>(1.0 / (1.0 + std::abs(i - j)), 0.25 * double(i - j) / double(n));
    return A;
}

template <typename T>
Matrix<T> solve_rhs(int n, int nrhs, int batch) {
    Matrix<T> B(n, nrhs, batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) elem<T>(B, i, j, b) = scalar_of<T>(1.0 + i - 2.0 * j + b, 0.5 - 0.1 * j);
    return B;
}

template <typename T>
void expect_solves(Queue& q, const char* what, int n, int nrhs, int batch, Transpose trans) {
    auto A = solve_system<T>(n, batch);
    auto B = solve_rhs<T>(n, nrhs, batch);

    auto s = linalg::solve(q, A, B, trans);
    q.wait();

    ASSERT_EQ(s.x.rows(), n) << what;
    ASSERT_EQ(s.x.cols(), nrhs) << what;
    ASSERT_EQ(s.x.batch_size(), batch) << what;
    EXPECT_VERIFY(T, verify::Check::solve, n, MV<T>::solve_residual(q, A, trans, s.x, B, A.all_items())) << what;
    expect_info(s.info, batch, -1, what);
}

}  // namespace

// float: n = 16 is inside the fused window, n = 48 is outside it by order, and
// nrhs = 6 is outside it by right-hand sides.
TEST(LinalgLayer, SolveNoTransFloatStraddlesTheFusedWindow) {
    Queue q;
    expect_solves<float>(q, "float inside", 16, 2, 8, Transpose::NoTrans);
    expect_solves<float>(q, "float outside by n", 48, 2, 8, Transpose::NoTrans);
    expect_solves<float>(q, "float outside by nrhs", 16, 6, 8, Transpose::NoTrans);
}

// complex<float>: the window stops at n = 16, half of float's.
TEST(LinalgLayer, SolveNoTransComplexStraddlesTheFusedWindow) {
    Queue q;
    expect_solves<std::complex<float>>(q, "cfloat inside", 8, 2, 4, Transpose::NoTrans);
    expect_solves<std::complex<float>>(q, "cfloat outside by n", 40, 2, 4, Transpose::NoTrans);
}

// The arm the rewire preserved. gesv has no Transpose parameter, so these must still
// go through getrf + getrs -- and ConjTrans must still conjugate.
TEST(LinalgLayer, SolveTransposedKeepsTheComposedArm) {
    Queue q;
    expect_solves<float>(q, "float Trans", 16, 2, 8, Transpose::Trans);
    expect_solves<float>(q, "float Trans, large n", 48, 2, 8, Transpose::Trans);
    expect_solves<std::complex<float>>(q, "cfloat Trans", 8, 2, 4, Transpose::Trans);
    expect_solves<std::complex<float>>(q, "cfloat ConjTrans", 8, 2, 4, Transpose::ConjTrans);
}

// A V == V diag(w).
TEST(LinalgLayer, EighProducesConsistentEigenpairs) {
    Queue q;
    const int n = 8, batch = 2;
    auto A = spd(n, batch);
    auto e = linalg::eigh(q, A);
    q.wait();

    test_utils::expect_eigenpairs<float>(q, A, e.vectors, e.values, n, std::span<const int>(A.all_items()));
    expect_info(e.info, batch, -1, "eigh");
}

// The convenience layer leases from the arena; repeated calls must not grow it.
TEST(LinalgLayer, RepeatedCallsDoNotGrowTheArena) {
    Queue q;
    const int n = 10, batch = 2;
    auto A = spd(n, batch);
    auto B = from_fn(n, 2, batch, [](int i, int j, int) { return float(i + j + 1); });

    auto warm = linalg::solve(q, A, B);
    q.wait();
    const size_t settled = q.workspace_capacity();

    for (int i = 0; i < 16; ++i) {
        auto X = linalg::solve(q, A, B);
        q.wait();
    }
    EXPECT_EQ(q.workspace_capacity(), settled);
}

// alpha is matmul's only scalar knob now that beta is gone.
TEST(LinalgLayer, MatmulAlphaScalesTheProduct) {
    Queue q;
    const int n = 4, batch = 2;
    auto A = from_fn(n, n, batch, [](int i, int j, int b) { return float(1 + i + 2 * j + b); });
    auto B = from_fn(n, n, batch, [](int i, int j, int) { return float(j - i + 3); });

    auto plain = linalg::matmul(q, A, B);
    auto scaled_product = linalg::matmul(q, A, B, {.alpha = 2.0f});
    q.wait();

    EXPECT_VERIFY(float, verify::Check::blas, n,
                  Dense::gemm_backward_error(q, A, Transpose::NoTrans, B, Transpose::NoTrans, plain,
                                             scaled_product, 2.0, 0.0, A.all_items()));
}

// triu/tril mask an existing matrix; fill_triangular generates one. The k
// offsets follow NumPy, and the negative-k case is the one an unsigned diagonal
// difference would get wrong.
TEST(LinalgLayer, TriuTrilMaskAnExistingMatrix) {
    Queue q;
    const int rows = 5, cols = 7, batch = 2;  // rectangular, so row/col are not interchangeable
    auto A = from_fn(rows, cols, batch,
                     [](int i, int j, int b) { return float(1 + i + 10 * j + 100 * b); });

    auto u0 = linalg::triu(q, A);
    auto u2 = linalg::triu(q, A, 2);
    auto l0 = linalg::tril(q, A);
    auto lm1 = linalg::tril(q, A, -1);
    q.wait();

    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                const float a = at(A, i, j, b);
                const int d = j - i;
                ASSERT_FLOAT_EQ(at(u0, i, j, b), d >= 0 ? a : 0.0f) << "triu " << i << "," << j;
                ASSERT_FLOAT_EQ(at(u2, i, j, b), d >= 2 ? a : 0.0f) << "triu2 " << i << "," << j;
                ASSERT_FLOAT_EQ(at(l0, i, j, b), d <= 0 ? a : 0.0f) << "tril " << i << "," << j;
                ASSERT_FLOAT_EQ(at(lm1, i, j, b), d <= -1 ? a : 0.0f) << "tril-1 " << i << "," << j;
            }

    // Aliasing: the documented in-place spelling.
    (void)linalg::triangular_mask_into<float>(q, A, A, Uplo::Upper, 0);
    q.wait();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                ASSERT_FLOAT_EQ(at(A, i, j, b), at(u0, i, j, b)) << "in place " << i << "," << j;
}

TEST(LinalgLayer, TriangularMaskRejectsMismatchedShapes) {
    Queue q;
    auto A = from_fn(4, 4, 1, [](int, int, int) { return 1.0f; });
    auto C = from_fn(4, 5, 1, [](int, int, int) { return 1.0f; });
    EXPECT_THROW((void)linalg::triangular_mask_into<float>(q, A, C, Uplo::Upper, 0),
                 std::invalid_argument);
}

TEST(LinalgLayer, TransposeSwapsExtentsAndElements) {
    Queue q;
    const int rows = 3, cols = 5, batch = 2;
    auto A = from_fn(rows, cols, batch,
                     [](int i, int j, int b) { return float(1 + i + 7 * j + 50 * b); });
    auto At = linalg::transpose(q, A);
    q.wait();

    ASSERT_EQ(At.rows(), cols);
    ASSERT_EQ(At.cols(), rows);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                ASSERT_FLOAT_EQ(at(At, j, i, b), at(A, i, j, b));
}

// norm returns one value per batch item. Unlike the rest of the layer it has
// already waited by the time it returns.
TEST(LinalgLayer, NormMatchesTheHostFrobeniusNorm) {
    const int rows = 4, cols = 3, batch = 3;
    Queue q;
    auto A = from_fn(rows, cols, batch, [](int i, int j, int b) { return float(i - 2 * j + b); });
    auto n = linalg::norm(q, A);

    ASSERT_EQ(n.size(), size_t(batch));
    std::vector<std::vector<double>> ref(batch);
    double scale = 0.0;
    for (int b = 0; b < batch; ++b) {
        double acc = 0.0;
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) acc += double(at(A, i, j, b)) * at(A, i, j, b);
        ref[b].push_back(std::sqrt(acc));
        scale = std::max(scale, std::sqrt(acc));
    }
    EXPECT_VERIFY(float, verify::Check::values, rows * cols,
                  Dense::values_error(VectorView<float>(n.data(), 1, batch), ref, scale));
}

// cond on a diagonal matrix has a closed form: the Frobenius condition number is
// ||A||_F ||A^-1||_F, and both factors are exact for a diagonal.
TEST(LinalgLayer, CondMatchesTheDiagonalReference) {
    Queue q;
    const int n = 4, batch = 2;
    const float d[n] = {1.0f, 2.0f, 4.0f, 8.0f};
    auto A = from_fn(n, n, batch, [&](int i, int j, int) { return i == j ? d[i] : 0.0f; });
    auto k = linalg::cond(q, A);

    double a2 = 0.0, ai2 = 0.0;
    for (int i = 0; i < n; ++i) {
        a2 += double(d[i]) * d[i];
        ai2 += 1.0 / (double(d[i]) * d[i]);
    }
    const double expected = std::sqrt(a2) * std::sqrt(ai2);
    ASSERT_EQ(k.size(), size_t(batch));
    EXPECT_VERIFY(float, verify::Check::values, n,
                  Dense::values_error(VectorView<float>(k.data(), 1, batch),
                                      std::vector<std::vector<double>>(batch, {expected}), expected));
}

TEST(LinalgLayer, SvdRejectsTheVectorlessJob) {
    Queue q;
    auto A = spd(4, 1);
    EXPECT_THROW(linalg::svd(q, A, SvdVectors::None), std::invalid_argument);
    EXPECT_THROW(linalg::svd(q, A, GesvdOptions{.jobu = SvdVectors::None, .jobvh = SvdVectors::None}),
                 std::invalid_argument);
}

TEST(LinalgLayer, EighSubsetRejectsExtremal) {
    Queue q;
    auto A = spd(6, 1);
    EXPECT_THROW(linalg::eigh(q, A, SyevxParams<float>{}), std::invalid_argument);
}

// ---- every wrapper over float, double and complex<float> ----------------------
// Inputs are padded views (ld = rows + 3, padded batch stride, poisoned padding), so a
// wrapper that derives ld or stride instead of reading it fails.

template <typename T>
class LinalgTyped : public ::testing::Test {
protected:
    Queue q;
};

using LinalgTypes = ::testing::Types<float, double, std::complex<float>>;
TYPED_TEST_SUITE(LinalgTyped, LinalgTypes);

TYPED_TEST(LinalgTyped, MatmulTransposesAndRectangles) {
    using T = TypeParam;
    auto& q = this->q;
    const int m = 5, k = 7, n = 3, batch = 3;
    const T alpha = scalar_of<T>(1.5, -0.5);
    for (Transpose ta : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
        for (Transpose tb : {Transpose::NoTrans, Transpose::ConjTrans}) {
            const bool tA = ta != Transpose::NoTrans, tB = tb != Transpose::NoTrans;
            auto A = padded<T>(q, make<T>(q, tA ? k : m, tA ? m : k, batch, Fill::general, 11));
            auto B = padded<T>(q, make<T>(q, tB ? n : k, tB ? k : n, batch, Fill::general, 12));
            auto C = linalg::matmul(q, A, B, {.alpha = alpha, .transA = ta, .transB = tb});
            q.wait();
            ASSERT_EQ(C.rows(), m);
            ASSERT_EQ(C.cols(), n);
            EXPECT_VERIFY(T, verify::Check::blas, k,
                          MV<T>::gemm_backward_error(q, A, ta, B, tb, C, C, alpha, T(0), C.all_items()))
                << "ta=" << int(ta) << " tb=" << int(tb);
        }
}

TYPED_TEST(LinalgTyped, CholeskyBothTrianglesAndInfo) {
    using T = TypeParam;
    auto& q = this->q;
    const int n = 12, batch = 6, bad = 4;
    auto A = make<T>(q, n, n, batch, Fill::spd, 21);
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        auto P = padded<T>(q, poison_other_triangle<T>(q, A, uplo));
        auto c = linalg::cholesky(q, P, uplo);
        q.wait();
        EXPECT_VERIFY(T, verify::Check::factorization, n,
                      MV<T>::potrf_residual(q, A, c.factor, uplo, A.all_items()))
            << "uplo " << int(uplo);
        expect_info(c.info, batch, -1, "cholesky");
        const int i = uplo == Uplo::Lower ? 0 : n - 1, j = uplo == Uplo::Lower ? n - 1 : 0;
        EXPECT_EQ(elem<T>(c.factor, i, j, batch - 1), T(0)) << "the other triangle is zeroed";
    }

    elem<T>(A, 0, 0, bad) = T(-1);  // leading minor of order 1 is negative
    auto c = linalg::cholesky(q, A);
    q.wait();
    expect_info(c.info, batch, bad, "cholesky, one non-SPD item");
    EXPECT_EQ(c.info[bad], 1);
}

TYPED_TEST(LinalgTyped, EighBothTrianglesAndInfo) {
    using T = TypeParam;
    auto& q = this->q;
    const int n = 10, batch = 4;
    auto A = make<T>(q, n, n, batch, Fill::hermitian, 31);
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        auto P = padded<T>(q, poison_other_triangle<T>(q, A, uplo));
        auto e = linalg::eigh(q, P, uplo);
        auto w = linalg::eigvalsh(q, P, uplo);
        q.wait();
        test_utils::expect_eigenpairs<T>(q, A, e.vectors, e.values, n, std::span<const int>(A.all_items()));
        test_utils::expect_eigenvalues_agree<T>(w, e.values, n, batch);
        expect_info(e.info, batch, -1, "eigh");
    }
}

TYPED_TEST(LinalgTyped, EighByIndexRange) {
    using T = TypeParam;
    auto& q = this->q;
    const int n = 12, batch = 4, il = 3, iu = 7, k = iu - il + 1;
    auto A = make<T>(q, n, n, batch, Fill::hermitian, 41);
    auto P = padded<T>(q, A);
    auto full = linalg::eigh(q, A);
    auto e = linalg::eigh(q, P, SyevxParams<T>{.select = SyevxSelect::Index, .il = il, .iu = iu});
    auto w = linalg::eigvalsh(q, P, SyevxParams<T>{.select = SyevxSelect::Index, .il = il, .iu = iu});
    q.wait();

    ASSERT_EQ(e.values.size(), size_t(k) * batch);
    ASSERT_EQ(e.vectors.rows(), n);
    ASSERT_EQ(e.vectors.cols(), k);
    EXPECT_EQ(w.vectors.rows(), 0);
    for (int b = 0; b < batch; ++b) {
        EXPECT_EQ(e.count[b], k) << b;
        EXPECT_EQ(w.count[b], k) << b;
    }
    expect_info(e.info, batch, -1, "eigh subset");
    const VectorView<Real<T>> wv(e.values.data(), k, batch);
    const VectorView<Real<T>> ref(full.values.data() + il, k, batch, 1, n);
    expect_vectors_agree<T>(wv, ref, n);
    expect_vectors_agree<T>(VectorView<Real<T>>(w.values.data(), k, batch), ref, n);
    EXPECT_VERIFY(T, verify::Check::eigen_residual, n, MV<T>::eigen_residual(q, A, e.vectors, wv));
    EXPECT_VERIFY(T, verify::Check::orthogonality_rotations, n, MV<T>::orthogonality(q, e.vectors));
}

// A value range has a per-item count; each must equal the full spectrum's count in (vl, vu].
TYPED_TEST(LinalgTyped, EighByValueRangeReportsCounts) {
    using T = TypeParam;
    auto& q = this->q;
    const int n = 12, batch = 4;
    auto A = make<T>(q, n, n, batch, Fill::hermitian, 51);
    auto full = linalg::eigh(q, A);
    q.wait();
    // Midpoints between eigenvalues of item 0, so no eigenvalue sits on an end.
    const Real<T> vl = (full.values[2] + full.values[3]) / 2, vu = (full.values[8] + full.values[9]) / 2;
    auto e = linalg::eigh(q, A, SyevxParams<T>{.select = SyevxSelect::Value, .vl = vl, .vu = vu});
    q.wait();

    ASSERT_EQ(e.vectors.cols(), n) << "capacity n";
    std::vector<int> counts;
    for (int b = 0; b < batch; ++b) {
        int c = 0, first = -1;
        for (int i = 0; i < n; ++i) {
            const Real<T> v = full.values[size_t(b) * n + i];
            if (v > vl && v <= vu) {
                if (first < 0) first = i;
                ++c;
            }
        }
        ASSERT_EQ(e.count[b], c) << "item " << b;
        counts.push_back(c);
        if (c == 0) continue;
        expect_vectors_agree<T>(VectorView<Real<T>>(e.values.data() + size_t(b) * n, c, 1),
                                VectorView<Real<T>>(full.values.data() + size_t(b) * n + first, c, 1), n);
    }
    EXPECT_EQ(counts[0], 6);
    expect_info(e.info, batch, -1, "eigh value range");
}

TYPED_TEST(LinalgTyped, SvdThinFullMixedAndRectangular) {
    using T = TypeParam;
    auto& q = this->q;
    const int batch = 3;
    struct Case { int m, n; SvdVectors ju, jv; };
    for (Case c : {Case{9, 6, SvdVectors::All, SvdVectors::All}, Case{9, 6, SvdVectors::Thin, SvdVectors::Thin},
                   Case{6, 9, SvdVectors::Thin, SvdVectors::Thin}, Case{6, 9, SvdVectors::All, SvdVectors::Thin},
                   Case{9, 6, SvdVectors::Thin, SvdVectors::All}}) {
        const int k = std::min(c.m, c.n);
        auto A = make<T>(q, c.m, c.n, batch, Fill::general, 61);
        auto s = linalg::svd(q, padded<T>(q, A), GesvdOptions{.jobu = c.ju, .jobvh = c.jv});
        ASSERT_EQ(s.U.cols(), c.ju == SvdVectors::All ? c.m : k);
        ASSERT_EQ(s.Vh.rows(), c.jv == SvdVectors::All ? c.n : k);
        EXPECT_VERIFY(T, verify::Check::factorization, std::max(c.m, c.n),
                      MV<T>::svd_residual(q, A, s.U, VectorView<Real<T>>(s.values.data(), k, batch), s.Vh))
            << c.m << "x" << c.n << " ju=" << int(c.ju) << " jv=" << int(c.jv);
        EXPECT_VERIFY(T, verify::Check::orthogonality, std::max(c.m, c.n), MV<T>::orthogonality(q, s.U));
        expect_info(s.info, batch, -1, "svd");
    }
}

// One side None: the other factor and the values must still come out.
TYPED_TEST(LinalgTyped, SvdOneSidedAndHermitian) {
    using T = TypeParam;
    auto& q = this->q;
    const int m = 8, n = 5, k = 5, batch = 2;
    auto A = make<T>(q, m, n, batch, Fill::general, 71);
    auto full = linalg::svd(q, A, SvdVectors::Thin);
    auto u_only = linalg::svd(q, A, GesvdOptions{.jobu = SvdVectors::Thin, .jobvh = SvdVectors::None});
    EXPECT_EQ(u_only.Vh.rows(), 0);
    EXPECT_VERIFY(T, verify::Check::orthogonality, m, MV<T>::orthogonality(q, u_only.U));
    expect_vectors_agree<T>(VectorView<Real<T>>(u_only.values.data(), k, batch),
                            VectorView<Real<T>>(full.values.data(), k, batch), m);

    const int h = 8;
    auto H = make<T>(q, h, h, batch, Fill::hermitian, 72);
    auto sh = linalg::svd(q, poison_other_triangle<T>(q, H, Uplo::Lower),
                          GesvdOptions{.hermitian_uplo = Uplo::Lower});
    EXPECT_VERIFY(T, verify::Check::factorization, h,
                  MV<T>::svd_residual(q, H, sh.U, VectorView<Real<T>>(sh.values.data(), h, batch), sh.Vh));
    expect_info(sh.info, batch, -1, "hermitian svd");
}

TYPED_TEST(LinalgTyped, LuSolveSolveSpdInvReportSingularItems) {
    using T = TypeParam;
    auto& q = this->q;
    const int nrhs = 3, batch = 6, bad = 3;
    for (int n : {16, 40}) {
        auto A = make<T>(q, n, n, batch, Fill::pivoted, 81);
        auto S = make<T>(q, n, n, batch, Fill::spd, 82);
        auto B = make<T>(q, n, nrhs, batch, Fill::general, 83);
        auto Ap = padded<T>(q, A), Sp = padded<T>(q, S), Bp = padded<T>(q, B);

        auto f = linalg::lu(q, Ap);
        auto x = linalg::solve(q, Ap, Bp);
        auto xt = linalg::solve(q, Ap, Bp, Transpose::ConjTrans);
        auto y = linalg::solve_spd(q, Sp, Bp);
        auto yu = linalg::solve_spd(q, padded<T>(q, poison_other_triangle<T>(q, S, Uplo::Upper)), Bp, Uplo::Upper);
        auto ai = linalg::inv(q, Ap);
        q.wait();

        Matrix<T> X(n, nrhs, batch);
        (void)MV<T>::copy(q, X, B);
        (void)getrs(q, f.factors, X, f.pivots.to_span());
        q.wait();
        const auto items = A.all_items();
        EXPECT_VERIFY(T, verify::Check::solve, n, MV<T>::solve_residual(q, A, X, B, items)) << "lu, n=" << n;
        EXPECT_VERIFY(T, verify::Check::solve, n, MV<T>::solve_residual(q, A, x.x, B, items)) << "solve, n=" << n;
        EXPECT_VERIFY(T, verify::Check::solve, n,
                      MV<T>::solve_residual(q, A, Transpose::ConjTrans, xt.x, B, items)) << "solve^H, n=" << n;
        EXPECT_VERIFY(T, verify::Check::solve, n, MV<T>::solve_residual(q, S, y.x, B, items)) << "solve_spd";
        EXPECT_VERIFY(T, verify::Check::solve, n, MV<T>::solve_residual(q, S, yu.x, B, items)) << "solve_spd U";
        EXPECT_VERIFY(T, verify::Check::solve, n,
                      MV<T>::solve_residual(q, A, ai.inverse, MV<T>(nullptr, 0, 0, 1), items)) << "inv";
        for (auto* info : {&f.info, &x.info, &xt.info, &y.info, &yu.info, &ai.info})
            expect_info(*info, batch, -1, "well-posed");

        for (int i = 0; i < n; ++i) elem<T>(A, i, 0, bad) = T(0);  // U(1,1) == 0 exactly
        elem<T>(S, 0, 0, bad) = T(-1);
        auto fs = linalg::lu(q, A);
        auto xs = linalg::solve(q, A, B);
        auto xts = linalg::solve(q, A, B, Transpose::Trans);
        auto ys = linalg::solve_spd(q, S, B);
        auto ais = linalg::inv(q, A);
        q.wait();
        expect_info(fs.info, batch, bad, "lu, singular item");
        expect_info(xs.info, batch, bad, "solve, singular item");
        expect_info(xts.info, batch, bad, "solve^T, singular item");
        expect_info(ys.info, batch, bad, "solve_spd, non-SPD item");
        expect_info(ais.info, batch, bad, "inv, singular item");
    }
}

TYPED_TEST(LinalgTyped, SolveTriangularEverySideTriangleAndOp) {
    using T = TypeParam;
    auto& q = this->q;
    const int m = 9, n = 4, batch = 3;
    auto L = make<T>(q, m, m, batch, Fill::spd, 91);
    auto R = make<T>(q, n, n, batch, Fill::spd, 92);
    auto B = make<T>(q, m, n, batch, Fill::general, 93);
    auto Bp = padded<T>(q, B);
    const T alpha = scalar_of<T>(2.0, 0.5);
    for (Side side : {Side::Left, Side::Right})
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
                for (Diag diag : {Diag::NonUnit, Diag::Unit}) {
                    const auto& A = side == Side::Left ? L : R;
                    auto X = linalg::solve_triangular(
                        q, padded<T>(q, A), Bp,
                        {.alpha = alpha, .side = side, .uplo = uplo, .trans = t, .diag = diag});
                    q.wait();
                    EXPECT_VERIFY(T, verify::Check::solve, m,
                                  MV<T>::trsm_residual(q, A, side, uplo, t, diag, X, B, alpha))
                        << "side " << int(side) << " uplo " << int(uplo) << " t " << int(t) << " diag " << int(diag);
                }
}

TYPED_TEST(LinalgTyped, MatvecBothOps) {
    using T = TypeParam;
    auto& q = this->q;
    const int m = 7, n = 5, batch = 3;
    auto A = padded<T>(q, make<T>(q, m, n, batch, Fill::general, 101));
    for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        const int len = t == Transpose::NoTrans ? n : m;
        // inc 2 and a padded stride, poison in the gaps.
        Vector<T> x(len, batch, Stride{2 * len + 3}, Inc{2});
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < 2 * len + 3; ++i) x.data_ptr()[b * (2 * len + 3) + i] = kPoison<T>;
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < len; ++i) x(i, b) = scalar_of<T>(0.5 * i - b, 0.25 * (i + b));
        auto y = linalg::matvec(q, A, x, t);
        q.wait();
        ASSERT_EQ(y.size(), t == Transpose::NoTrans ? m : n);
        EXPECT_VERIFY(T, verify::Check::blas, std::max(m, n),
                      MV<T>::gemv_backward_error(q, A, t, x, y, y, T(1), T(0))) << "t " << int(t);
    }
}

TYPED_TEST(LinalgTyped, GramBothSidesBothTriangles) {
    using T = TypeParam;
    auto& q = this->q;
    const int m = 9, n = 5, batch = 3;
    const Transpose h = kIsComplex<T> ? Transpose::ConjTrans : Transpose::Trans;
    auto A = padded<T>(q, make<T>(q, m, n, batch, Fill::general, 111));
    // Complex gram is herk, which has no native family: vendor-free it must throw, not fall back.
    if (kIsComplex<T> && !(BATCHLAS_HAS_CUBLAS || BATCHLAS_HAS_ROCBLAS)) {
        EXPECT_THROW((void)linalg::gram(q, A), NoRouteError);
        return;
    }
    auto inner = linalg::gram(q, A);
    auto outer = linalg::gram(q, A, Transpose::NoTrans);
    q.wait();
    ASSERT_EQ(inner.rows(), n);
    ASSERT_EQ(outer.rows(), m);
    EXPECT_VERIFY(T, verify::Check::blas, m,
                  MV<T>::gemm_backward_error(q, A, h, A, Transpose::NoTrans, inner, inner, T(1), T(0)))
        << "A^H A";
    EXPECT_VERIFY(T, verify::Check::blas, n,
                  MV<T>::gemm_backward_error(q, A, Transpose::NoTrans, A, h, outer, outer, T(1), T(0)))
        << "A A^H";
}

// ---- out-of-order queues -------------------------------------------------------
// A wrapper that copies A and then factors the copy is two launches: on an out-of-order
// Queue they need an explicit order. Saturating batches give the race room to show.

namespace {
constexpr int kRaceN = 32, kRaceBatch = 16384, kRaceReps = 3;
}  // namespace

// posv and syev run an out-of-order caller's call on a temporary in-order queue; the caller's
// Queue::wait() must still cover that work.
TEST(LinalgWrappers, OutOfOrderWaitCoversInOrderDrivers) {
    Queue base;
    Queue q(base, false);
    const int n = 64, nrhs = 4, batch = 4096;
    auto F = spd(n, batch);
    auto E = spd(n, batch);
    auto X = from_fn(n, nrhs, batch, [](int i, int j, int b) { return float(1 + i - j + (b % 5)); });
    UnifiedVector<float> W(static_cast<size_t>(n) * batch);
    auto lease = q.workspace(posv_buffer_size(q, F, X, Uplo::Lower));
    const auto complete = sycl::info::event_command_status::complete;

    Event solved = posv(q, F, X, Uplo::Lower, lease.span(), Span<int32_t>{});
    q.wait();
    EXPECT_EQ(sycl_event(solved).get_info<sycl::info::event::command_execution_status>(), complete) << "posv";

    Event values = syev(q, E, W.to_span(), {.jobz = JobType::NoEigenVectors});
    q.wait();
    EXPECT_EQ(sycl_event(values).get_info<sycl::info::event::command_execution_status>(), complete) << "syev";
}

TEST(LinalgWrappers, OutOfOrderCholeskyLuInvOrderTheirCopy) {
    Queue base;
    Queue q(base, false);
    auto A = make<float>(base, kRaceN, kRaceN, kRaceBatch, Fill::spd, 7);
    auto B = make<float>(base, kRaceN, 2, kRaceBatch, Fill::general, 3);
    for (int rep = 0; rep < kRaceReps; ++rep) {
        auto c = linalg::cholesky(q, A);
        auto f = linalg::lu(q, A);
        auto ai = linalg::inv(q, A);
        q.wait();
        EXPECT_VERIFY(float, verify::Check::factorization, kRaceN,
                      Dense::potrf_residual(base, A, c.factor, Uplo::Lower)) << "cholesky " << rep;
        expect_info(c.info, kRaceBatch, -1, "cholesky");
        Matrix<float> X(kRaceN, 2, kRaceBatch);
        (void)Dense::copy(base, X, B);
        (void)getrs(base, f.factors, X, f.pivots.to_span());
        base.wait();
        EXPECT_VERIFY(float, verify::Check::solve, kRaceN, Dense::solve_residual(base, A, X, B)) << "lu " << rep;
        EXPECT_VERIFY(float, verify::Check::solve, kRaceN,
                      Dense::solve_residual(base, A, ai.inverse, Dense(nullptr, 0, 0, 1))) << "inv " << rep;
    }
}

TEST(LinalgWrappers, OutOfOrderSolvesOrderTheirCopies) {
    Queue base;
    Queue q(base, false);
    auto A = make<float>(base, kRaceN, kRaceN, kRaceBatch, Fill::spd, 7);
    auto B = make<float>(base, kRaceN, 2, kRaceBatch, Fill::general, 3);
    for (int rep = 0; rep < kRaceReps; ++rep) {
        auto x = linalg::solve(q, A, B);
        auto xt = linalg::solve(q, A, B, Transpose::Trans);
        auto y = linalg::solve_spd(q, A, B);
        auto z = linalg::solve_triangular(q, A, B);
        q.wait();
        EXPECT_VERIFY(float, verify::Check::solve, kRaceN, Dense::solve_residual(base, A, x.x, B)) << rep;
        EXPECT_VERIFY(float, verify::Check::solve, kRaceN,
                      Dense::solve_residual(base, A, Transpose::Trans, xt.x, B)) << rep;
        EXPECT_VERIFY(float, verify::Check::solve, kRaceN, Dense::solve_residual(base, A, y.x, B)) << rep;
        EXPECT_VERIFY(float, verify::Check::solve, kRaceN,
                      Dense::trsm_residual(base, A, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit,
                                           z, B, 1.0)) << rep;
    }
}

TEST(LinalgWrappers, OutOfOrderEighSvdGramOrderTheirLaunches) {
    Queue base;
    Queue q(base, false);
    auto A = make<float>(base, kRaceN, kRaceN, kRaceBatch, Fill::hermitian, 7);
    for (int rep = 0; rep < kRaceReps; ++rep) {
        auto e = linalg::eigh(q, A);
        auto es = linalg::eigh(q, A, SyevxParams<float>{.select = SyevxSelect::Index, .il = 4, .iu = 9});
        auto s = linalg::svd(q, A);
        auto g = linalg::gram(q, A);
        q.wait();
        test_utils::expect_eigenpairs<float>(base, A, e.vectors, e.values, kRaceN);
        EXPECT_VERIFY(float, verify::Check::eigen_residual, kRaceN,
                      Dense::eigen_residual(base, A, es.vectors, VectorView<float>(es.values.data(), 6, kRaceBatch)))
            << rep;
        EXPECT_VERIFY(float, verify::Check::factorization, kRaceN,
                      Dense::svd_residual(base, A, s.U, VectorView<float>(s.values.data(), kRaceN, kRaceBatch), s.Vh))
            << rep;
        EXPECT_VERIFY(float, verify::Check::blas, kRaceN,
                      Dense::gemm_backward_error(base, A, Transpose::Trans, A, Transpose::NoTrans, g, g, 1.0, 0.0))
            << rep;
    }
}

// The value-returning wrappers own their scratch (the copied A, the factor); returning while
// the kernels that read it are still enqueued frees it under them. Many calls with no wait in
// between, on an out-of-order queue, give the freed memory every chance to be reused.
TEST(LinalgWrappers, ScratchOutlivesKernels) {
    Queue base;
    Queue q(base, false);
    const int n = 48, nrhs = 3, batch = 256, reps = 20;
    auto A = spd(n, batch);
    auto B = from_fn(n, nrhs, batch, [](int i, int j, int b) { return float(1 + i - 2 * j + (b % 7)); });

    std::vector<linalg::Solve<float>> X, Y;
    std::vector<UnifiedVector<float>> W;
    for (int r = 0; r < reps; ++r) {
        X.push_back(linalg::solve(q, A, B));
        Y.push_back(linalg::solve_spd(q, A, B));
        W.push_back(linalg::eigvalsh(q, A));
    }
    q.wait();

    std::vector<std::vector<double>> ref(static_cast<size_t>(batch));
    double scale = 0.0;
    for (int b = 0; b < batch; ++b) {
        ASSERT_TRUE(batchlas::verify::eigenvalues(A, b, ref[static_cast<size_t>(b)]));
        scale = std::max(scale, std::fabs(ref[static_cast<size_t>(b)].back()));
    }
    for (int r = 0; r < reps; ++r) {
        EXPECT_VERIFY(float, batchlas::verify::Check::solve, n, Dense::solve_residual(base, A, X[r].x, B))
            << "solve, rep " << r;
        EXPECT_VERIFY(float, batchlas::verify::Check::solve, n, Dense::solve_residual(base, A, Y[r].x, B))
            << "solve_spd, rep " << r;
        const VectorView<float> w(W[r].data(), n, batch);
        EXPECT_VERIFY(float, batchlas::verify::Check::values, n, Dense::values_error(w, ref, scale)) << "eigvalsh, rep " << r;
    }
}
