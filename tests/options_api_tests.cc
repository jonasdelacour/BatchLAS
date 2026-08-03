#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-vector.hh>

#include <vector>

using namespace batchlas;

namespace {

Matrix<float, MatrixFormat::Dense> filled(int n, int batch, float scale) {
    Matrix<float, MatrixFormat::Dense> m(n, n, batch);
    auto v = m.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                v.data_ptr()[b * v.stride() + j * v.ld() + i] =
                    scale * static_cast<float>((i * 5 + j * 3 + b * 7) % 13) / 13.0f;
    return m;
}

// Symmetric positive definite, so potrf/syev have something well-posed to chew.
Matrix<float, MatrixFormat::Dense> spd(int n, int batch) {
    Matrix<float, MatrixFormat::Dense> m(n, n, batch);
    auto v = m.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                v.data_ptr()[b * v.stride() + j * v.ld() + i] =
                    (i == j) ? float(n + b + 2)
                             : 1.0f / (1.0f + static_cast<float>((i > j ? i - j : j - i)));
    return m;
}

void expect_same(const MatrixView<float, MatrixFormat::Dense>& a,
                 const MatrixView<float, MatrixFormat::Dense>& b,
                 int n, int batch, const char* what) {
    for (int k = 0; k < batch; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const auto ia = k * a.stride() + j * a.ld() + i;
                const auto ib = k * b.stride() + j * b.ld() + i;
                ASSERT_FLOAT_EQ(a.data_ptr()[ia], b.data_ptr()[ib])
                    << what << " differs at (" << i << "," << j << ") batch " << k;
            }
}

// Overload resolution, pinned at compile time.
//
// The dispatch macro's variadic overload used to accept any argument list at
// all. That made it beat these option-struct overloads and then fail inside its
// own body, on calls it should never have claimed -- so the constraint below is
// load-bearing, and a silent regression of it would only show up as a confusing
// error deep in a header.
namespace resolution {

using M = MatrixView<float, MatrixFormat::Dense>;

// The arena spelling exists and is callable...
static_assert(requires(Queue& q, M A, Span<float> tau) { geqrf(q, A, tau); },
              "geqrf(ctx, A, tau) should lease its workspace from the arena");
static_assert(requires(Queue& q, M A, Span<float> tau, Span<std::byte> w) { geqrf(q, A, tau, w); },
              "geqrf(ctx, A, tau, ws) should still take a caller-managed span");

static_assert(requires(Queue& q, M A, M B, M C) { gemm(q, A, B, C, GemmOptions<float>{}); },
              "gemm should accept an option struct");
static_assert(requires(Queue& q, M A, M B, M C) {
                  gemm(q, A, B, C, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
              },
              "the positional spelling should still deduce its backend from the queue");

// Note these are positive assertions only. The matching negative form --
// asserting that a *wrong* call is rejected -- cannot be written with an
// explicit template argument list, because a mismatch there is a hard error
// rather than something a requires-expression absorbs. The negative direction
// is covered instead by the fact that this file compiles at all: before the
// constraint was added to BATCHLAS_DISPATCH_ON_QUEUE, the calls above resolved
// to the variadic overload and failed to compile.

}  // namespace resolution

}  // namespace

// Every option struct must default to exactly what the positional call defaults
// to. If one drifts, the concise spelling quietly computes something else.
TEST(OptionsApi, DefaultsMatchPositionalDefaults) {
    Queue q;
    const int n = 16, batch = 2;
    auto A = filled(n, batch, 1.0f);
    auto B = filled(n, batch, 0.5f);

    Matrix<float, MatrixFormat::Dense> C_opts(n, n, batch), C_pos(n, n, batch);
    C_opts.view().fill_zeros(q);
    C_pos.view().fill_zeros(q);
    q.wait();

    gemm(q, A.view(), B.view(), C_opts.view(), GemmOptions<float>{});
    gemm(q, A.view(), B.view(), C_pos.view(), 1.0f, 0.0f, Transpose::NoTrans,
         Transpose::NoTrans, ComputePrecision::Default);
    q.wait();
    expect_same(C_opts.view(), C_pos.view(), n, batch, "gemm defaults");
}

TEST(OptionsApi, DesignatedInitialisersSetOnlyWhatTheyName) {
    Queue q;
    const int n = 12, batch = 2;
    auto A = filled(n, batch, 1.0f);
    auto B = filled(n, batch, 0.75f);

    Matrix<float, MatrixFormat::Dense> C_opts(n, n, batch), C_pos(n, n, batch);
    C_opts.view().fill_zeros(q);
    C_pos.view().fill_zeros(q);
    q.wait();

    // alpha and transA named; beta, transB and precision must keep their defaults.
    gemm(q, A.view(), B.view(), C_opts.view(),
         {.alpha = 2.5f, .transA = Transpose::Trans});
    gemm(q, A.view(), B.view(), C_pos.view(), 2.5f, 0.0f, Transpose::Trans,
         Transpose::NoTrans, ComputePrecision::Default);
    q.wait();
    expect_same(C_opts.view(), C_pos.view(), n, batch, "gemm with designated initialisers");
}

TEST(OptionsApi, Blas3OptionsMatchPositional) {
    Queue q;
    const int n = 10, batch = 2;
    auto A = filled(n, batch, 1.0f);
    auto B = filled(n, batch, 0.6f);

    {   // symm
        Matrix<float, MatrixFormat::Dense> Co(n, n, batch), Cp(n, n, batch);
        Co.view().fill_zeros(q); Cp.view().fill_zeros(q); q.wait();
        symm(q, A.view(), B.view(), Co.view(), {.alpha = 1.5f, .side = Side::Right, .uplo = Uplo::Upper});
        symm(q, A.view(), B.view(), Cp.view(), 1.5f, 0.0f, Side::Right, Uplo::Upper);
        q.wait();
        expect_same(Co.view(), Cp.view(), n, batch, "symm");
    }
    {   // syrk
        Matrix<float, MatrixFormat::Dense> Co(n, n, batch), Cp(n, n, batch);
        Co.view().fill_zeros(q); Cp.view().fill_zeros(q); q.wait();
        syrk(q, A.view(), Co.view(), {.alpha = 0.5f, .uplo = Uplo::Upper});
        syrk(q, A.view(), Cp.view(), 0.5f, 0.0f, Uplo::Upper, Transpose::NoTrans);
        q.wait();
        expect_same(Co.view(), Cp.view(), n, batch, "syrk");
    }
    {   // syr2k
        Matrix<float, MatrixFormat::Dense> Co(n, n, batch), Cp(n, n, batch);
        Co.view().fill_zeros(q); Cp.view().fill_zeros(q); q.wait();
        syr2k(q, A.view(), B.view(), Co.view(), {.alpha = 0.25f});
        syr2k(q, A.view(), B.view(), Cp.view(), 0.25f, 0.0f, Uplo::Lower, Transpose::NoTrans);
        q.wait();
        expect_same(Co.view(), Cp.view(), n, batch, "syr2k");
    }
    {   // trmm
        auto Bo = filled(n, batch, 0.6f);
        auto Bp = filled(n, batch, 0.6f);
        Matrix<float, MatrixFormat::Dense> Co(n, n, batch), Cp(n, n, batch);
        Co.view().fill_zeros(q); Cp.view().fill_zeros(q); q.wait();
        trmm(q, A.view(), Bo.view(), Co.view(), {.alpha = 2.0f, .diag = Diag::Unit});
        trmm(q, A.view(), Bp.view(), Cp.view(), 2.0f, Side::Left, Uplo::Lower,
             Transpose::NoTrans, Diag::Unit);
        q.wait();
        expect_same(Co.view(), Cp.view(), n, batch, "trmm");
    }
    {   // trsm writes into B, so give each call its own copy
        auto Bo = filled(n, batch, 0.6f);
        auto Bp = filled(n, batch, 0.6f);
        auto Tri = spd(n, batch);
        trsm(q, Tri.view(), Bo.view(), {.alpha = 1.0f, .diag = Diag::NonUnit});
        trsm(q, Tri.view(), Bp.view(), Side::Left, Uplo::Lower, Transpose::NoTrans,
             Diag::NonUnit, 1.0f);
        q.wait();
        expect_same(Bo.view(), Bp.view(), n, batch, "trsm");
    }
}

// The arena payoff: omitting the workspace must give the same answer as sizing
// and allocating it by hand.
TEST(OptionsApi, OmittedWorkspaceMatchesExplicitWorkspace) {
    Queue q;
    const int n = 24, batch = 3;

    {   // potrf
        auto Ao = spd(n, batch);
        auto Ap = spd(n, batch);
        potrf(q, Ao.view(), {.uplo = Uplo::Lower});
        auto ws = q.workspace(potrf_buffer_size(q, Ap.view(), Uplo::Lower));
        potrf(q, Ap.view(), {.uplo = Uplo::Lower}, ws.span());
        q.wait();
        expect_same(Ao.view(), Ap.view(), n, batch, "potrf");
    }
    {   // geqrf
        auto Ao = filled(n, batch, 1.0f);
        auto Ap = filled(n, batch, 1.0f);
        UnifiedVector<float> tau_o(n * batch), tau_p(n * batch);
        geqrf(q, Ao.view(), tau_o.to_span());
        auto ws = q.workspace(geqrf_buffer_size(q, Ap.view(), tau_p.to_span()));
        geqrf(q, Ap.view(), tau_p.to_span(), ws.span());  // positional: caller-managed
        q.wait();
        expect_same(Ao.view(), Ap.view(), n, batch, "geqrf");
        for (size_t i = 0; i < tau_o.size(); ++i)
            ASSERT_FLOAT_EQ(tau_o[i], tau_p[i]) << "geqrf tau at " << i;
    }
    {   // syev
        auto Ao = spd(n, batch);
        auto Ap = spd(n, batch);
        UnifiedVector<float> Wo(n * batch), Wp(n * batch);
        syev(q, Ao.view(), Wo.to_span(), {.jobz = JobType::NoEigenVectors});
        auto ws = q.workspace(
            syev_buffer_size(q, Ap.view(), Wp.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        syev(q, Ap.view(), Wp.to_span(), {.jobz = JobType::NoEigenVectors}, ws.span());
        q.wait();
        for (size_t i = 0; i < Wo.size(); ++i)
            ASSERT_NEAR(Wo[i], Wp[i], 1e-4f) << "syev eigenvalue at " << i;
    }
}

// Repeating an arena-backed call must not grow the arena without bound -- the
// lease has to actually come back at the end of each call.
TEST(OptionsApi, ArenaBackedCallsDoNotLeak) {
    Queue q;
    const int n = 20, batch = 2;
    auto A = spd(n, batch);

    potrf(q, A.view());
    q.wait();
    const size_t settled = q.workspace_capacity();

    for (int i = 0; i < 24; ++i) {
        auto Ai = spd(n, batch);
        potrf(q, Ai.view());
        q.wait();
    }
    EXPECT_EQ(q.workspace_capacity(), settled);
}

// The option-struct spelling must not have made the older spellings ambiguous.
TEST(OptionsApi, CoexistsWithPositionalAndExplicitBackendSpellings) {
    Queue q;
    const int n = 8, batch = 1;
    auto A = filled(n, batch, 1.0f);
    auto B = filled(n, batch, 0.5f);
    Matrix<float, MatrixFormat::Dense> C(n, n, batch);
    C.view().fill_zeros(q);
    q.wait();

    // option struct
    gemm(q, A.view(), B.view(), C.view(), {.alpha = 1.0f});
    // positional, backend from the queue
    gemm(q, A.view(), B.view(), C.view(), 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
    // positional, backend named
#if BATCHLAS_HAS_CUDA_BACKEND
    if (q.backend() == Backend::CUDA) {
        gemm<Backend::CUDA>(q, A.view(), B.view(), C.view(), 1.0f, 0.0f,
                            Transpose::NoTrans, Transpose::NoTrans);
    }
#endif
    q.wait();
    SUCCEED();
}
