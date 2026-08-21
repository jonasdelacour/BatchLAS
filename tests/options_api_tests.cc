#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/queue-dispatch.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <complex>
#include <fstream>
#include <string>
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

// hemm is the one level-3 entry point constrained to complex scalars: BLAS has
// no real ?hemm, and for a real matrix "Hermitian" and "symmetric" are the same
// statement. Here both directions *are* assertable, because a concept is a
// template: inside one the call is a dependent expression, so a mismatch makes
// the concept false instead of ending the translation unit.
template <typename T>
concept HemmTakesOptions = requires(Queue& q, MatrixView<T, MatrixFormat::Dense> A) {
    hemm(q, A, A, A, HemmOptions<T>{});
};

template <typename T>
concept HemmTakesPositional = requires(Queue& q, MatrixView<T, MatrixFormat::Dense> A) {
    hemm(q, A, A, A, T(1), T(0), Side::Left, Uplo::Lower);
};

static_assert(HemmTakesOptions<std::complex<float>>, "hemm should accept an option struct");
static_assert(HemmTakesPositional<std::complex<float>>,
              "hemm's positional spelling should deduce its backend from the queue");
static_assert(!HemmTakesOptions<float>, "hemm must not accept real operands");
static_assert(!HemmTakesPositional<float>, "hemm must not accept real operands");

// herk and her2k are constrained the same way, and additionally take a real
// alpha (herk) or a real beta (both) while their operands stay complex -- the
// classic ?herk mistake is to give alpha type T, which compiles and computes
// something that is not Hermitian.
template <typename T>
concept HerkTakesOptions = requires(Queue& q, MatrixView<T, MatrixFormat::Dense> A) {
    herk(q, A, A, HerkOptions<T>{});
};

template <typename T>
concept HerkTakesRealScalars = requires(Queue& q, MatrixView<T, MatrixFormat::Dense> A,
                                       typename base_type<T>::type r) {
    herk(q, A, A, r, r, Uplo::Lower, Transpose::NoTrans);
};

template <typename T>
concept Her2kTakesOptions = requires(Queue& q, MatrixView<T, MatrixFormat::Dense> A) {
    her2k(q, A, A, A, Her2kOptions<T>{});
};

template <typename T>
concept Her2kTakesComplexAlphaRealBeta = requires(Queue& q, MatrixView<T, MatrixFormat::Dense> A,
                                                  typename base_type<T>::type r) {
    her2k(q, A, A, A, T(1), r, Uplo::Lower, Transpose::NoTrans);
};

static_assert(HerkTakesOptions<std::complex<float>>, "herk should accept an option struct");
static_assert(HerkTakesRealScalars<std::complex<float>>,
              "herk's alpha and beta are real, as in cublas?herk");
static_assert(!HerkTakesOptions<float>, "herk must not accept real operands");
static_assert(Her2kTakesOptions<std::complex<float>>, "her2k should accept an option struct");
static_assert(Her2kTakesComplexAlphaRealBeta<std::complex<float>>,
              "her2k's alpha is complex and its beta real, as in cublas?her2k");
static_assert(!Her2kTakesOptions<float>, "her2k must not accept real operands");

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
        trsm(q, Tri.view(), Bp.view(), 1.0f, Side::Left, Uplo::Lower, Transpose::NoTrans,
             Diag::NonUnit);
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

// An explicitly-passed empty workspace must be passed through, NOT treated as
// "no workspace given, lease one".
//
// This is a regression test for a real corruption. The option overloads once
// shared a single body with `Span<std::byte> ws = {}` and an
// `if (ws.data() != nullptr)` to decide whether to lease from the arena. That
// makes an empty span mean "I did not pass one" -- but library code that
// sub-allocates from a BumpAllocator runs its algorithm once in sizing mode,
// where every pool allocation legitimately yields an empty span while the input
// matrices stay real. Under the old body the sizing pass therefore leased real
// memory and really ran the factorisation over live data, silently destroying
// it. `ortho`'s Cholesky and SVQB paths both did this, which showed up only as
// LOBPCG failing to converge much further downstream.
//
// A backend call with a genuinely too-small workspace must fail rather than
// quietly succeed by allocating behind the caller's back.
TEST(OptionsApi, EmptyWorkspaceIsUsedNotReplacedByALease) {
    Queue q;
    const int n = 32, batch = 2;
    auto A = spd(n, batch);

    // The discriminating case is an *explicitly passed empty* span. Under the
    // old shared body this took the "no workspace given" branch and leased from
    // the arena; under the split overloads it is forwarded as given. The arena's
    // capacity is the observable difference, so assert on that. Whether the
    // backend then rejects the empty workspace is beside the point and is not
    // asserted -- only that the call did not go allocating behind our back.
    const size_t before = q.workspace_capacity();
    try {
        potrf(q, A.view(), {.uplo = Uplo::Lower}, Span<std::byte>{});
        q.wait();
    } catch (const std::exception&) {
        // A backend is entitled to refuse a zero-sized workspace.
    }
    EXPECT_EQ(q.workspace_capacity(), before)
        << "an explicitly passed empty workspace was silently replaced by an arena lease";

    // And the omitted-workspace spelling must still lease, or the arena-backed
    // convenience would be doing nothing at all.
    Queue fresh;
    auto A2 = spd(n, batch);
    const size_t fresh_before = fresh.workspace_capacity();
    potrf(fresh, A2.view(), {.uplo = Uplo::Lower});
    fresh.wait();
    EXPECT_GT(fresh.workspace_capacity(), fresh_before)
        << "omitting the workspace should lease one from the queue's arena";
}


// WP4 step 0.6. potrf validated NOTHING on the positional path.
//
// require_square / require_info_span are attached only to the OPTION overloads
// (options.hh:548-549, :557-558, :565-566); the workspace-taking <Backend B>
// overload at :539-543 -- the spelling src/extensions/ortho.cc:200 uses -- has
// neither, and there was no potrf_validate_params anywhere in the tree. So a
// non-square view reached the backend and cuSOLVER factorised A.rows() x
// A.rows() out of it, reading past the columns it was given.
//
// The facade now validates before it resolves a route, because the shape
// builder reads A.rows()/A.cols() and must not describe a non-conforming view.
// Both entry points do it, and they must agree: potrf_buffer_size resolves the
// route a second time (options.hh:550 then :551), and the two disagreeing is
// the ormqr defect class -- buffer size 2560 bytes, call demanded 276480.
TEST(OptionsApi, PotrfRejectsANonSquareViewOnEveryEntryPoint) {
    Queue q;
    Matrix<float, MatrixFormat::Dense> oblong(8, 5, 2);

    EXPECT_THROW(potrf_buffer_size(q, oblong.view(), Uplo::Lower),
                 std::invalid_argument)
        << "the buffer-size query reads A.rows()/A.cols() too";

    EXPECT_THROW(potrf(q, oblong.view(), Uplo::Lower, Span<std::byte>{},
                       Span<int32_t>{}),
                 std::invalid_argument)
        << "the POSITIONAL overload is the one that had no validation at all";

    // GUARD: the same calls on a square view must NOT throw, or the two
    // assertions above would pass for a reason that has nothing to do with
    // squareness.
    auto square = spd(8, 2);
    size_t bytes = 0;
    EXPECT_NO_THROW(bytes = potrf_buffer_size(q, square.view(), Uplo::Lower));
    auto ws = q.workspace(bytes);
    EXPECT_NO_THROW(potrf(q, square.view(), Uplo::Lower, ws.span(), Span<int32_t>{}));
    q.wait();
}

// An empty option struct must be written with its type named, never as `{}`.
//
// Regression test for a silent wrong answer. `potrf`'s option overload
//     potrf<B>(ctx, A, const PotrfOptions&, Span<std::byte>)
// has the SAME ARITY as the positional one
//     potrf<B>(ctx, A, Uplo,               Span<std::byte>)
// and a braced-init-list converts to both. Overload resolution picks the
// positional one, so `potrf<B>(ctx, A, {}, ws)` means `Uplo{}` -- and because
// Uplo is `{Upper, Lower}`, `Uplo{}` is Upper while `PotrfOptions{}.uplo` is
// Lower. The call therefore factorises the opposite triangle and returns a
// confidently wrong answer. `ortho`'s Cholesky path did exactly this, and it
// surfaced only as LOBPCG failing to converge several layers up.
//
// This is the one arity collision in the surface; the other option overloads
// differ in arity or in argument type from their positional twins. Naming the
// type is the rule, and this test is what enforces it.
TEST(OptionsApi, NamedEmptyOptionsSelectTheOptionOverload) {
    Queue q;
    const int n = 24, batch = 2;

    auto A_named = spd(n, batch);
    auto A_uplo = spd(n, batch);
    auto A_upper = spd(n, batch);

    with_backend(q, [&](auto Back) {
        constexpr Backend B = Back.value;
        auto w1 = q.workspace(potrf_buffer_size<B, float>(q, A_named.view(), Uplo::Lower));
        auto w2 = q.workspace(potrf_buffer_size<B, float>(q, A_uplo.view(), Uplo::Lower));
        auto w3 = q.workspace(potrf_buffer_size<B, float>(q, A_upper.view(), Uplo::Upper));
        potrf<B>(q, A_named.view(), PotrfOptions{}, w1.span());
        potrf<B>(q, A_uplo.view(), Uplo::Lower, w2.span());
        potrf<B>(q, A_upper.view(), Uplo::Upper, w3.span());
    });
    q.wait();

    // PotrfOptions{} defaults to Lower, so the named spelling must match the
    // explicit Lower call...
    expect_same(A_named.view(), A_uplo.view(), n, batch, "PotrfOptions{} == Uplo::Lower");

    // ...and Lower must be distinguishable from Upper, or the assertion above
    // would hold no matter which overload had been selected.
    bool differs = false;
    for (int b = 0; b < batch && !differs; ++b)
        for (int j = 0; j < n && !differs; ++j)
            for (int i = 0; i < n && !differs; ++i) {
                const auto l = A_uplo.view();
                const auto u = A_upper.view();
                if (std::abs(l.data_ptr()[b * l.stride() + j * l.ld() + i] -
                             u.data_ptr()[b * u.stride() + j * u.ld() + i]) > 1e-4f)
                    differs = true;
            }
    EXPECT_TRUE(differs) << "Lower and Upper potrf are indistinguishable here, so this "
                            "test could not detect the wrong triangle being used";
}

// A bare `{}` in potrf's 4-argument form used to select the positional overload
// -- Uplo{} == Upper -- and silently factorise the wrong triangle. A deleted
// overload taking a dedicated enum type now puts a third candidate at the same
// exact-match rank, so the call is ambiguous and fails to compile.
//
// The negative is checked by static_assert rather than by a compile-fail
// fixture: overload resolution on `{}` cannot be probed with a requires-clause,
// because a braced-init-list is not an expression that can be deduced. What can
// be checked is that the trap type exists and that neither legitimate spelling
// converts to it, which is the property that keeps the fix from breaking callers.
static_assert(!std::is_convertible_v<PotrfOptions, detail::EmptyBracesAreAmbiguous>,
              "PotrfOptions must not convert to the trap type, or the named "
              "spelling would become ambiguous too");
static_assert(!std::is_convertible_v<Uplo, detail::EmptyBracesAreAmbiguous>,
              "Uplo must not convert to the trap type, or the positional "
              "spelling would become ambiguous too");

// ---------------------------------------------------------------------------
// Per-item factorisation status (issue #73).
//
// potrf, getrf and getri all had somewhere to put LAPACK's `info` -- every
// backend allocated the array the vendor call demands -- and every backend then
// dropped it. A batch containing one rank-deficient item returned an ordinary
// Event, no throw and no flag, so the caller could not tell "the batch
// factorised" from "item 1 is garbage and everything downstream of it is noise".
// There was no workaround at the public API.
//
// The batches below are deliberately mixed: item 0 is well-conditioned and item
// 1 is not, so the test fails both if status is never reported (bad item reads
// 0) and if it is reported indiscriminately (good item reads non-zero). The
// `info` buffers start at a sentinel rather than 0, because 0 is the success
// value -- a buffer the backend never touched would otherwise look like "all
// items factorised", which is exactly the bug.
//
// Values are asserted only as zero / non-zero. LAPACK, cuSOLVER, cuBLAS and
// rocSOLVER agree on the sign convention but not always on which index they
// name first, and the API's contract is the zero/non-zero distinction.
// ---------------------------------------------------------------------------
namespace {

// Item 0: diagonally dominant, so Cholesky succeeds. Item 1: the negative of
// it, so the very first leading minor is not positive definite.
Matrix<float, MatrixFormat::Dense> spd_then_negative_definite(int n) {
    Matrix<float, MatrixFormat::Dense> m(n, n, 2);
    auto v = m.view();
    for (int b = 0; b < 2; ++b) {
        const float sign = (b == 0) ? 1.0f : -1.0f;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                v.data_ptr()[b * v.stride() + j * v.ld() + i] =
                    sign * ((i == j) ? float(n + 2) : 0.25f);
    }
    return m;
}

// Item 0: a well-conditioned diagonal. Item 1: all zeros, so the first pivot is
// exactly zero and U is singular.
Matrix<float, MatrixFormat::Dense> nonsingular_then_singular(int n) {
    Matrix<float, MatrixFormat::Dense> m(n, n, 2);
    auto v = m.view();
    for (int b = 0; b < 2; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                v.data_ptr()[b * v.stride() + j * v.ld() + i] =
                    (b == 1) ? 0.0f : ((i == j) ? float(i + 2) : 0.5f);
    return m;
}

constexpr int32_t kNeverWritten = -12345;

}  // namespace

TEST(OptionsApi, FactorisationsReportPerItemStatus) {
    Queue q;
    constexpr int n = 8;
    constexpr int batch = 2;

    {   // potrf: item 1 is not positive definite
        auto A = spd_then_negative_definite(n);
        UnifiedVector<int32_t> info(batch, kNeverWritten);
        potrf(q, A.view(), {.uplo = Uplo::Lower, .info = info.to_span()});
        q.wait();
        EXPECT_EQ(info[0], 0) << "potrf reported failure on a positive definite item";
        EXPECT_NE(info[0], kNeverWritten) << "potrf never wrote info at all";
        EXPECT_GT(info[1], 0) << "potrf did not report the indefinite item";
    }

    {   // getrf: item 1 is singular
        auto A = nonsingular_then_singular(n);
        UnifiedVector<int64_t> pivots(static_cast<size_t>(n) * batch);
        UnifiedVector<int32_t> info(batch, kNeverWritten);
        getrf(q, A.view(), pivots.to_span(), info.to_span());
        q.wait();
        EXPECT_EQ(info[0], 0) << "getrf reported failure on a nonsingular item";
        EXPECT_NE(info[0], kNeverWritten) << "getrf never wrote info at all";
        EXPECT_GT(info[1], 0) << "getrf did not report the singular item";
    }

    {   // getri: same batch, inverted from its own LU
        auto A = nonsingular_then_singular(n);
        Matrix<float, MatrixFormat::Dense> Ainv(n, n, batch);
        UnifiedVector<int64_t> pivots(static_cast<size_t>(n) * batch);
        UnifiedVector<int32_t> info(batch, kNeverWritten);
        getrf(q, A.view(), pivots.to_span());
        getri(q, A.view(), Ainv.view(), pivots.to_span(), info.to_span());
        q.wait();
        EXPECT_EQ(info[0], 0) << "getri reported failure on an invertible item";
        EXPECT_NE(info[0], kNeverWritten) << "getri never wrote info at all";
        EXPECT_GT(info[1], 0) << "getri did not report the singular item";
    }
}

// The whole change is additive, so the spellings that existed before must still
// compile and still run. If `info` had been made a required parameter -- or a
// defaulted one on a signature the sig:: aliases have to restate, which function
// types cannot carry -- every one of these would have broken instead.
TEST(OptionsApi, FactorisationsStillWorkWithoutAnInfoSpan) {
    Queue q;
    constexpr int n = 8;
    constexpr int batch = 2;

    auto A = spd(n, batch);
    potrf(q, A.view(), {.uplo = Uplo::Lower});
    auto Aw = spd(n, batch);
    with_backend(q, [&](auto Back) {
        constexpr Backend Bk = Back.value;
        auto ws = q.workspace(potrf_buffer_size<Bk, float>(q, Aw.view(), Uplo::Lower));
        // The four-argument positional spelling: the arity that had to survive.
        potrf<Bk, float>(q, Aw.view(), Uplo::Lower, ws.span());
    });

    auto B = nonsingular_then_singular(n);
    UnifiedVector<int64_t> pivots(static_cast<size_t>(n) * batch);
    getrf(q, B.view(), pivots.to_span());

    Matrix<float, MatrixFormat::Dense> Binv(n, n, batch);
    getri(q, B.view(), Binv.view(), pivots.to_span());
    q.wait();
    SUCCEED();
}

// An `info` span shorter than the batch is rejected up front rather than
// silently ignored. That direction matters more than it looks: a backend handed
// a short span falls back to its own scratch and writes nothing to the caller's
// buffer, so the caller would read stale zeros -- "every item factorised" -- on
// precisely the batch it was trying to diagnose. An EMPTY span stays legal; it
// is the API's spelling for "do not report".
TEST(OptionsApi, ShortInfoSpanIsRejected) {
    Queue q;
    constexpr int n = 8;
    constexpr int batch = 4;

    auto A = spd(n, batch);
    UnifiedVector<int32_t> too_short(batch - 1, 0);
    EXPECT_THROW(potrf(q, A.view(), {.uplo = Uplo::Lower, .info = too_short.to_span()}),
                 std::invalid_argument);

    UnifiedVector<int64_t> pivots(static_cast<size_t>(n) * batch);
    EXPECT_THROW(getrf(q, A.view(), pivots.to_span(), too_short.to_span()),
                 std::invalid_argument);

    EXPECT_NO_THROW(potrf(q, A.view(), {.uplo = Uplo::Lower, .info = Span<int32_t>{}}));
    q.wait();
}

// The USM contract used to be documented and unenforced: handing ordinary host
// memory to a GPU queue reached the device as a wild address and aborted the
// process from inside the SYCL runtime during teardown, where no catch block
// could help -- while the identical code was correct on the host backend.
TEST(OptionsApi, HostPointerToDeviceQueueThrowsInsteadOfAborting) {
    Queue q;
    if (q.device().type == DeviceType::CPU) {
        GTEST_SKIP() << "host memory is legitimately device-accessible on a CPU device";
    }

    const int n = 4;
    std::vector<float> host(n * n, 1.0f);
    MatrixView<float, MatrixFormat::Dense> A(host.data(), n, n);

    EXPECT_FALSE(q.is_device_accessible(host.data()));
    EXPECT_THROW(gemm(q, A, A, A, GemmOptions<float>{}), std::invalid_argument);

    // The message has to name the entry point and pin the argument down, or it
    // sends the reader hunting. Which of the two labellings appears depends on
    // which overload wins: the variadic dispatch overload forwards an unnamed
    // pack ("argument 1"), the option-struct overload knows the name ("A").
    try {
        gemm(q, A, A, A, GemmOptions<float>{});
        FAIL() << "expected the pointer check to throw";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("gemm"), std::string::npos) << msg;
        EXPECT_TRUE(msg.find("argument 1") != std::string::npos ||
                    msg.find("gemm: A") != std::string::npos) << msg;
        // and it must say what to do about it, not just what went wrong
        EXPECT_NE(msg.find("malloc_shared"), std::string::npos) << msg;
    }
}

// The check must not reject the allocations that genuinely work, or it would
// trade a crash for a false alarm.
TEST(OptionsApi, DeviceAccessibleMemoryIsAccepted) {
    Queue q;
    const int n = 4;

    Matrix<float, MatrixFormat::Dense> owned(n, n, 1);
    EXPECT_TRUE(q.is_device_accessible(owned.view().data_ptr()));
    EXPECT_FALSE(q.is_device_accessible(nullptr));

    // And a real call through the checked path still runs.
    Matrix<float, MatrixFormat::Dense> B(n, n, 1), C(n, n, 1);
    auto a = owned.view(), b = B.view(), c = C.view();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            a.data_ptr()[j * a.ld() + i] = (i == j) ? 2.0f : 0.0f;
            b.data_ptr()[j * b.ld() + i] = (i == j) ? 3.0f : 0.0f;
            c.data_ptr()[j * c.ld() + i] = 0.0f;
        }
    EXPECT_NO_THROW(gemm(q, a, b, c, GemmOptions<float>{}));
    q.wait();
    EXPECT_NEAR(c.data_ptr()[0], 6.0f, 1e-5f);
}

// A default-constructed view is how this API spells "this optional matrix is not
// in use": `syevx(ctx, A, W, k, ws, JobType::NoEigenVectors,
// MatrixView<T, MatrixFormat::Dense>(), params)` is the documented call and ~50
// call sites in this repo write it. It owns no memory, so the USM check has
// nothing to reach and must let it through. When it did not, every such call
// threw "the pointer is null" and four iluk_tests failed -- and iluk_tests is
// `slow`-labelled, so the usual `-LE slow` run could not see it. Hence this
// cheap guard in a `util`-labelled test.
TEST(OptionsApi, EmptyViewSentinelIsNotRejected) {
    Queue q;

    MatrixView<float, MatrixFormat::Dense> absent_matrix{};
    VectorView<float> absent_vector{};
    EXPECT_NO_THROW(detail::require_arg_accessible(q, absent_matrix, "syevx: V"));
    EXPECT_NO_THROW(detail::require_arg_accessible(q, absent_vector, "syevx: v"));

    // ...while a view that does address elements is still checked, or the
    // exemption would have swallowed the contract it is carved out of.
    if (q.device().type != DeviceType::CPU) {
        std::vector<float> host(16, 0.0f);
        MatrixView<float, MatrixFormat::Dense> host_backed(host.data(), 4, 4);
        EXPECT_THROW(detail::require_arg_accessible(q, host_backed, "syevx: A"),
                     std::invalid_argument);
    }
}

// ---------------------------------------------------------------------------
// Every backend-deducing option overload checks its pointers.
//
// The option struct is written as a braced initialiser on purpose. That is the
// spelling that makes the variadic queue-dispatch overload drop out (Args cannot
// be deduced from a braced-init-list), so the call lands on the overload in
// options.hh and nothing else can supply the check on its behalf. `getrs`'s
// workspace-taking overload was the one that had no BATCHLAS_CHECK_ARGS: with a
// braced option struct it reached the backend with a host pointer, while its
// siblings -- potrf and syev in the same shape -- threw.
//
// A missing check here does not fail cleanly: the call reaches the device with a
// wild address and aborts the process. That is the failure this test exists to
// keep from coming back; see also OptionsHeaderKeepsEveryDeducingOverloadChecked
// below, which catches it without running anything.
// ---------------------------------------------------------------------------
TEST(OptionsApi, EveryDeducingOptionOverloadRejectsHostMemory) {
    Queue q;
    if (q.device().type == DeviceType::CPU) {
        GTEST_SKIP() << "host memory is legitimately device-accessible on a CPU device";
    }

    constexpr int n = 4;
    std::vector<float> hf(n * n, 1.0f);
    std::vector<std::complex<float>> hc(n * n, std::complex<float>(1.0f, 0.0f));
    std::vector<int64_t> hpiv(n, 0);
    std::vector<std::byte> hws(4096);

    using DenseF = MatrixView<float, MatrixFormat::Dense>;
    using DenseC = MatrixView<std::complex<float>, MatrixFormat::Dense>;
    DenseF A(hf.data(), n, n), B(hf.data(), n, n), C(hf.data(), n, n);
    DenseC Ac(hc.data(), n, n), Bc(hc.data(), n, n), Cc(hc.data(), n, n);
    VectorView<float> x(hf.data(), n), y(hf.data(), n);
    Span<float> tau(hf.data(), n);
    Span<float> W(hf.data(), n);
    Span<int64_t> pivots(hpiv.data(), hpiv.size());
    Span<std::byte> ws(hws.data(), hws.size());

    // dense BLAS
    EXPECT_THROW(gemm(q, A, B, C, {.alpha = 1.0f}), std::invalid_argument);
    EXPECT_THROW(gemv(q, A, x, y, {.alpha = 1.0f}), std::invalid_argument);
    EXPECT_THROW(symm(q, A, B, C, {.alpha = 1.0f}), std::invalid_argument);
    EXPECT_THROW(hemm(q, Ac, Bc, Cc, {.alpha = std::complex<float>(1.0f)}),
                 std::invalid_argument);
    EXPECT_THROW(herk(q, Ac, Cc, {.alpha = 1.0f}), std::invalid_argument);
    EXPECT_THROW(her2k(q, Ac, Bc, Cc, {.alpha = std::complex<float>(1.0f)}),
                 std::invalid_argument);
    EXPECT_THROW(syrk(q, A, C, {.alpha = 1.0f}), std::invalid_argument);
    EXPECT_THROW(syr2k(q, A, B, C, {.alpha = 1.0f}), std::invalid_argument);
    EXPECT_THROW(trmm(q, A, B, C, {.alpha = 1.0f}), std::invalid_argument);
    EXPECT_THROW(trsm(q, A, B, {.alpha = 1.0f}), std::invalid_argument);

    // dense LAPACK, both the arena spelling and the workspace-taking one
    EXPECT_THROW(potrf(q, A, {.uplo = Uplo::Lower}), std::invalid_argument);
    EXPECT_THROW(potrf(q, A, {.uplo = Uplo::Lower}, ws), std::invalid_argument);
    EXPECT_THROW(getrf(q, A, pivots), std::invalid_argument);
    EXPECT_THROW(getrs(q, A, B, pivots, {.trans = Transpose::NoTrans}),
                 std::invalid_argument);
    EXPECT_THROW(getrs(q, A, B, pivots, {.trans = Transpose::NoTrans}, ws),
                 std::invalid_argument);
    EXPECT_THROW(getri(q, A, B, pivots), std::invalid_argument);
    EXPECT_THROW(geqrf(q, A, tau), std::invalid_argument);
    EXPECT_THROW(orgqr(q, A, tau), std::invalid_argument);
    EXPECT_THROW(syev(q, A, W, {.jobz = JobType::EigenVectors}), std::invalid_argument);
    EXPECT_THROW(syev(q, A, W, {.jobz = JobType::EigenVectors}, ws), std::invalid_argument);
}

// The behavioural test above can only cover the entry points that exist today,
// and a new one added without BATCHLAS_CHECK_ARGS would sail past it. This reads
// options.hh itself and holds every backend-deducing overload -- the ones a
// caller reaches by writing `f(ctx, ...)` -- to the rule. The Backend-explicit
// `f<B>(ctx, ...)` overloads are deliberately exempt: they are the library's own
// inner-loop spelling (src/extensions/*.cc calls them inside iteration loops),
// where a sycl::get_pointer_type per argument per call is not worth paying.
TEST(OptionsApi, OptionsHeaderKeepsEveryDeducingOverloadChecked) {
#ifndef BATCHLAS_OPTIONS_HH_PATH
    GTEST_SKIP() << "options.hh path not passed in by the build";
#else
    std::ifstream in(BATCHLAS_OPTIONS_HH_PATH);
    ASSERT_TRUE(in.is_open()) << "cannot read " << BATCHLAS_OPTIONS_HH_PATH;

    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    ASSERT_FALSE(lines.empty());

    int deducing_overloads = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        // A definition starts at `inline Event NAME(Queue& ctx`.
        if (lines[i].find("inline Event ") == std::string::npos) continue;
        if (lines[i].find("(Queue& ctx") == std::string::npos) continue;

        // Walk back over the template head; `template <Backend B, ...` marks the
        // Backend-explicit forms.
        bool backend_explicit = false;
        for (size_t k = i; k-- > 0;) {
            if (lines[k].find("template <") != std::string::npos) {
                backend_explicit = lines[k].find("Backend B") != std::string::npos;
                break;
            }
            if (lines[k].empty()) break;
        }
        if (backend_explicit) continue;
        ++deducing_overloads;

        // ...and forward over the body, to the closing brace in column 0.
        bool checked = false;
        for (size_t k = i; k < lines.size(); ++k) {
            if (lines[k].find("BATCHLAS_CHECK_ARGS") != std::string::npos) checked = true;
            if (k > i && !lines[k].empty() && lines[k][0] == '}') break;
        }
        EXPECT_TRUE(checked)
            << "backend-deducing overload at options.hh:" << (i + 1)
            << " has no BATCHLAS_CHECK_ARGS -- a caller can hand it host memory and "
               "the process aborts inside the runtime instead of throwing:\n  "
            << lines[i];
    }
    // Guards the scan itself: if the parse stops matching the file, this drops
    // to zero and the loop above silently passes.
    EXPECT_GE(deducing_overloads, 20) << "only found " << deducing_overloads
                                      << " backend-deducing overloads; the scan is stale";
#endif
}
