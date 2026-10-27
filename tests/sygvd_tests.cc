// sygst and sygvd: the generalized Hermitian-definite eigenproblem. Residuals and B-normalization through
// batchlas::verify (reference products, never the routes under test), graded B (cond ~1e2 float, ~1e4
// double), Hermitian A with nonzero imaginary parts, both triangles with the other one poisoned, padded ld
// and stride, every pinned family, a non-positive-definite item, the factored entry, a saturating batch.
// evidence: docs/perf/sygvd.md#sygvd-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/blas/functions/sygvd.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <tests/sygvd_test_common.hh>
#include <tests/test_support/coverage_rows.hh>
#include <tests/test_support/inputs.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <src/ops/sygst/choice.hh>
#include <src/ops/sygvd/choice.hh>
#include <src/select/select.hh>

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace batchlas;

namespace {

namespace ts = test_support;
using ts::Live;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// B's condition scale 10^(2 decades) (see scaled): 1e2 in single precision, 1e4 in double.
template <typename T>
constexpr double b_cond() { return std::is_same_v<batchlas::float_t<T>, float> ? 1e2 : 1e4; }

// The generalized residual and normalization scale with cond(B).
template <typename T>
verify::Slack cond_slack() { return {4.0 * b_cond<T>(), "generalized problem: error ~ cond(B)"}; }

// sygst's residual on a badly scaled B = D S D follows S, not cond(B): at most 17x the bound at cond(B) 1e6
// (double), under 1x at 1e3 (float) (2026-10-27); far tighter than cond_slack, so a subtle defect shows.
inline verify::Slack graded_slack() { return {32.0, "graded B: the reduction's error follows the scaled S"}; }

// S := D S D in place, D_i = 10^(decades (i / (n - 1) - 1/2)): entries spread over 10^(2 decades) on a
// well-conditioned S. Local: no shared generator or MatrixView facility scales rows and columns.
template <typename T>
void scaled(Matrix<T>& S, double decades) {
    const int n = S.rows();
    std::vector<double> D(n);
    for (int i = 0; i < n; ++i) D[i] = std::pow(10.0, decades * (n > 1 ? double(i) / (n - 1) - 0.5 : 0.0));
    for (int b = 0; b < S.batch_size(); ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) S(i, j, b) *= batchlas::float_t<T>(D[i] * D[j]);
}

// A0 (random Hermitian) and B0 = D S D, D taking cond(B) to ~`cond`: S is fill_random_spd's with its diagonal
// raised to 2.5 n, so its off-diagonal stays ~1/(4n) of it (a larger one makes sygst's error follow cond(B)).
// Whole and packed, one item when identical; a and b in padded, poisoned storage (A ld n + 3, stride + 5;
// B ld n + 2, stride + 7) holding their uplo triangles.
template <typename T>
struct Gen {
    Uplo uplo;
    Matrix<T> A0, B0, a, b;
};

using sygvd_test::items_of;

template <typename Config>
class SygvdTest : public sygvd_test::SygvdFixture<Config> {
protected:
    using Base = sygvd_test::SygvdFixture<Config>;
    using T = typename Config::ScalarType;
    using R = batchlas::float_t<T>;
    static constexpr Backend BK = Config::BackendVal;
    using Base::q;
    using Base::vendor_linked;

    Gen<T> gen(int n, int batch, Uplo uplo, unsigned seed, bool identical = false, double cond = b_cond<T>()) {
        const int drawn = identical ? 1 : batch;
        Gen<T> g{uplo, Matrix<T>(n, n, drawn), Matrix<T>(n, n, drawn), ts::padded<T>(n, n, batch),
                 ts::padded<T>(n, n, batch, 2, 7)};
        (void)g.A0.view().fill_random(q(), true, seed);
        (void)g.B0.view().fill_random_spd(q(), seed + 1);
        (void)g.B0.view().fill_diagonal(q(), MV<T>::make(2.5 * n, 0.0));
        q().wait();
        scaled(g.B0, std::log10(cond) / 2);
        ts::store<T>(g.a, g.A0, Live::triangle(uplo));
        ts::store<T>(g.b, g.B0, Live::triangle(uplo));
        return g;
    }

    static constexpr bool sygst_vendor_linked() { return BK == Backend::NETLIB && vendor_linked(); }

    // {residual, normalization} of itype's problem over every item (generalized_eigen_residual, X in A's
    // storage); X^H B X = I (itypes 1, 2) or B = X X^H (3).
    std::pair<double, double> check(const Gen<T>& g, UnifiedVector<R>& w, int itype) {
        const int n = g.a.rows(), batch = g.a.batch_size();
        const std::vector<int> all = g.a.all_items();
        const double res = MV<T>::generalized_eigen_residual(q(), g.A0, g.B0, g.a, VectorView<R>(w.data(), n, batch), itype, all);
        if (itype == 3) return {res, MV<T>::similarity_residual(q(), g.B0, g.a, Matrix<T>::Identity(n, batch), all)};
        Matrix<T> P(n, n, batch);
        (void)MV<T>::product(q(), g.B0.view(), Transpose::NoTrans, g.a.view(), Transpose::NoTrans, P.view());
        q().wait();
        return {res, MV<T>::solve_residual(q(), g.a, Transpose::ConjTrans, P, MV<T>{}, all)};
    }

    // sygvd (or sygvd_factored) with exactly the queried workspace; false on a NoRouteError.
    bool run(Gen<T>& g, UnifiedVector<R>& w, UnifiedVector<int32_t>& info, int itype, JobType jobz,
             bool factored = false) {
        const int batch = g.a.batch_size();
        w = UnifiedVector<R>(static_cast<size_t>(g.a.rows()) * batch, R(-4321));
        info = UnifiedVector<int32_t>(static_cast<size_t>(batch), int32_t(-12345));
        try {
            const size_t need = factored ? sygvd_factored_buffer_size<BK, T>(q(), g.a, g.b, w.to_span(), itype, jobz, g.uplo)
                                         : sygvd_buffer_size<BK, T>(q(), g.a, g.b, w.to_span(), itype, jobz, g.uplo);
            UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
            const Span<std::byte> s(ws.data(), need);
            if (factored)
                (void)sygvd_factored<BK, T>(q(), g.a, g.b, w.to_span(), itype, jobz, g.uplo, s, info.to_span());
            else
                (void)sygvd<BK, T>(q(), g.a, g.b, w.to_span(), itype, jobz, g.uplo, s, info.to_span());
            q().wait();
        } catch (const NoRouteError&) {
            return false;
        }
        return true;
    }

    // B's Cholesky factor in place, the other triangle re-poisoned whatever potrf did to it (a reader of
    // the wrong triangle shows). False when the build has no potrf route.
    bool factor(Gen<T>& g) {
        UnifiedVector<int32_t> info(static_cast<size_t>(g.b.batch_size()), int32_t(-12345));
        if (!ts::potrf_in_place<BK, T>(q(), g.b, g.uplo, info.to_span())) return false;
        for (int it = 0; it < g.b.batch_size(); ++it) EXPECT_EQ(info[it], 0) << "potrf item " << it;
        ts::poison_outside<T>(g.b, Live::triangle(g.uplo));
        return true;
    }

    // sygst's result against its definition: itype 1 ||A0 - Q C Q^H|| / ||A0||, itypes 2, 3
    // ||C - Q A0 Q^H|| / ||C||, with C mirrored from A's uplo triangle and Q the factor (or its ^H).
    double sygst_residual(const Gen<T>& g, int itype) {
        const int n = g.a.rows(), nb = g.a.batch_size();
        const Matrix<T> C = ts::hermitian<T>(g.a, g.uplo, true), H = ts::hermitian<T>(g.b, g.uplo);
        const bool qh = (itype == 1) != (g.uplo == Uplo::Lower);  // Q = F^H: the mirror of F's triangle
        const Uplo other = g.uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower;
        Matrix<T> Q = Matrix<T>::Zeros(n, n, nb);
        ts::store<T>(Q, H, Live::triangle(qh ? other : g.uplo));
        if (qh)
            for (int b = 0; b < nb; ++b)
                for (int i = 0; i < n; ++i) Q(i, i, b) = MV<T>::conj(T(H(i, i, b)));
        const std::vector<int> all = g.a.all_items();
        return itype == 1 ? MV<T>::similarity_residual(q(), g.A0, Q, C, all) : MV<T>::similarity_residual(q(), C, Q, g.A0, all);
    }

    // Whether can_run is documented false for a pinned sygst family on this backend: blocked off the
    // GPU, vendor (LAPACKE) where it is not linked. Only these may skip; any other NoRouteError fails.
    static bool sygst_unreachable(const std::string& fam) {
        if (fam == "vendor") return !sygst_vendor_linked();
        return BK == Backend::NETLIB && fam.rfind("blocked", 0) == 0;
    }

    // sygst under a pin with exactly the queried workspace; false on a NoRouteError, which is a
    // failure unless sygst_unreachable(fam).
    bool run_sygst(Gen<T>& g, int itype, const std::string& fam) {
        select::ScopedPin<ops::sygst::SygstChoice> pin("sygst", fam);
        try {
            const size_t need = sygst_buffer_size<BK, T>(q(), g.a, g.b, itype, g.uplo);
            UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
            (void)sygst<BK, T>(q(), g.a, g.b, itype, g.uplo, Span<std::byte>(ws.data(), need));
            q().wait();
        } catch (const NoRouteError& e) {
            if (!sygst_unreachable(fam)) ADD_FAILURE() << fam << " itype=" << itype << ": " << e.what();
            return false;
        }
        return true;
    }
};

using SygvdTestTypes = typename test_utils::backend_types<ts::Config>::type;

}  // namespace

TYPED_TEST_SUITE(SygvdTest, SygvdTestTypes);

// sygst against its definition, from potrf's factor: itype 1 A0 = L C L^H (Lower) or U^H C U (Upper);
// itypes 2, 3 C = L^H A0 L or U A0 U^H (similarity_residual with the factor built on the host). Every
// family it can pin, both triangles, an order ladder through the CTA width, B read-only (pads too).
// ARMED BREAK: in src/ops/sygst/sygst.cc twosided(), give the itype-1 Right trsm `Transpose::NoTrans` for
// Lower. SEEN (2026-10-25; re-run 2026-10-27 on test_support): RED here for twosided itype 1 Lower only, all
// 8 configs; on CUDA also compose itype 1 Lower in SygvdSolves (double, cdouble) and FactoredEntry (double),
// the same set as before the migration. NETLIB's compose takes LAPACKE's sygst, its last resort.
TYPED_TEST(SygvdTest, SygstMatchesTheTwoSidedDefinition) {
    using T = typename TestFixture::T;
    constexpr Backend BK = TestFixture::BK;
    std::vector<std::string> fams{"twosided"};
    if (TestFixture::sygst_vendor_linked()) fams.push_back("vendor");
    for (const std::string& fam : fams)
        for (int itype : {1, 2, 3})
            for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
                for (int n : {1, 5, 33, 70}) {
                    auto g = this->gen(n, 3, uplo, 7u + n * 13 + itype);
                    if (!this->factor(g)) GTEST_SKIP() << "no potrf route in this build";
                    const Matrix<T> factor_bits = g.b;
                    select::ScopedPin<ops::sygst::SygstChoice> pin("sygst", fam);
                    size_t need = 0;
                    try {
                        need = sygst_buffer_size<BK, T>(this->q(), g.a, g.b, itype, uplo);
                    } catch (const NoRouteError& e) {
                        if (!TestFixture::sygst_unreachable(fam)) FAIL() << fam << ": " << e.what();
                        GTEST_SKIP() << "no sygst route in this build";
                    }
                    UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
                    (void)sygst<BK, T>(this->q(), g.a, g.b, itype, uplo, Span<std::byte>(ws.data(), need));
                    this->q().wait();
                    const std::string at = fam + " itype=" + std::to_string(itype) + " uplo=" + ts::uplo_name(uplo) +
                                           " n=" + std::to_string(n);
                    EXPECT_VERIFY_SLACK(T, verify::Check::factorization, n, this->sygst_residual(g, itype), cond_slack<T>()) << at;
                    ASSERT_TRUE(ts::unwritten_outside<T>(g.b, Live::nothing(), factor_bits)) << at << ": B";
                }
}

// The blocked family pinned at every nb, orders straddling each edge it has (the diagonal kernel's
// compile-time orders 8, 16, 32, nb 32 to 256, a two-level recursion at 260), every itype and triangle, on an ill-conditioned B (cond ~1e3
// float, ~1e6 double), padded ld and stride, complex data with nonzero imaginary parts. It reads and
// writes only the uplo triangles: every element of A outside its triangle and all of B keep their bits.
// Deliberate breaks and their red sets: docs/perf/sygvd.md#sygst-the-deliberate-breaks.
TYPED_TEST(SygvdTest, SygstBlockedMatchesTheDefinitionAcrossBlockEdges) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::BK == Backend::NETLIB) GTEST_SKIP() << "blocked is a GPU family";
    const double cond = std::is_same_v<batchlas::float_t<T>, float> ? 1e3 : 1e6;
    for (const std::string fam : {"blocked:nb=32", "blocked:nb=64", "blocked:nb=128", "blocked:nb=256"})
        for (int itype : {1, 2, 3})
            for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
                for (int n : {1, 7, 8, 16, 17, 31, 32, 33, 48, 64, 65, 97, 128, 130, 260}) {
                    auto g = this->gen(n, 3, uplo, 11u + n * 17 + itype, false, cond);
                    if (!this->factor(g)) GTEST_SKIP() << "no potrf route in this build";
                    const Matrix<T> factor_bits = g.b;
                    if (!this->run_sygst(g, itype, fam)) GTEST_SKIP() << "no sygst route in this build";
                    const std::string at = fam + " itype=" + std::to_string(itype) + " uplo=" + ts::uplo_name(uplo) +
                                           " n=" + std::to_string(n);
                    EXPECT_VERIFY_SLACK(T, verify::Check::factorization, n, this->sygst_residual(g, itype),
                                        graded_slack()) << at;
                    ASSERT_TRUE(ts::unwritten_outside<T>(g.a, Live::triangle(uplo))) << at << ": A";
                    ASSERT_TRUE(ts::unwritten_outside<T>(g.b, Live::nothing(), factor_bits)) << at << ": B";
                }
}

// 1024 identical items on the blocked family, each bit-identical to item 0: n = 5 packs up to 25 items
// per diagonal work-group (the last part-filled), n = 16 the order-16 instance, n = 20 the generic
// order, n = 40 two blocks (orders 32 and 8) and every child.
// The workspace query is exact: itype 2's panel product, one byte short, throws.
TYPED_TEST(SygvdTest, SaturatingSygstBlockedIsBitIdenticalToItemZero) {
    using T = typename TestFixture::T;
    constexpr Backend BK = TestFixture::BK;
    if constexpr (BK == Backend::NETLIB) GTEST_SKIP() << "blocked is a GPU family";
    for (const std::string fam : {"blocked:nb=32", "blocked:nb=64"})
        for (int itype : {1, 2})
            for (int n : {5, 16, 20, 40}) {
                const Uplo uplo = itype == 1 ? Uplo::Upper : Uplo::Lower;
                auto g = this->gen(n, 1024, uplo, 515u + n, true);
                if (!this->factor(g)) GTEST_SKIP() << "no potrf route in this build";
                if (!this->run_sygst(g, itype, fam)) GTEST_SKIP() << "no sygst route in this build";
                ASSERT_TRUE(ts::items_bit_identical<T>(g.a)) << fam << " itype=" << itype << " n=" << n;
            }
    // Passes of 100 items, the last taking the remainder (124), on distinct items: every item meets the
    // definition and nothing outside A's triangle is written. (Children route by the pass's batch, so a
    // bit comparison with one pass would compare different kernels.)
    for (int itype : {1, 2})
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            auto g = this->gen(40, 1024, uplo, 77u + itype);
            if (!this->factor(g)) GTEST_SKIP() << "no potrf route in this build";
            {
                const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "100");
                ASSERT_TRUE(this->run_sygst(g, itype, "blocked:nb=32"));
            }
            const std::string at = std::string("chunked itype=") + std::to_string(itype) + " uplo=" + ts::uplo_name(uplo);
            EXPECT_VERIFY_SLACK(T, verify::Check::factorization, 40, this->sygst_residual(g, itype), graded_slack()) << at;
            ASSERT_TRUE(ts::unwritten_outside<T>(g.a, Live::triangle(uplo))) << at << ": A";
        }
    select::ScopedPin<ops::sygst::SygstChoice> pin("sygst", "blocked:nb=32");
    auto h = this->gen(40, 4, Uplo::Lower, 1u);
    const size_t need = sygst_buffer_size<BK, T>(this->q(), h.a, h.b, 2, Uplo::Lower);
    ASSERT_GT(need, 0u);
    UnifiedVector<std::byte> ws(need);
    EXPECT_THROW(((void)sygst<BK, T>(this->q(), h.a, h.b, 2, Uplo::Lower, Span<std::byte>(ws.data(), need - 1))),
                 batchlas::workspace_error);
}

// Each pinned family is the one that ran: the coverage `reached` row of each call names its spelling.
// One call per family (a row is per shape class and spelling), across the itypes and both triangles.
TYPED_TEST(SygvdTest, CoverageRowNamesEachPinnedSygstFamily) {
    if constexpr (TestFixture::BK == Backend::NETLIB) GTEST_SKIP() << "blocked is a GPU family";
    struct Run {
        int n, itype;
        Uplo uplo;
        std::string fam;
    };
    const std::vector<Run> runs{{40, 1, Uplo::Upper, "blocked:nb=32"}, {70, 2, Uplo::Upper, "blocked:nb=64"},
                                {140, 3, Uplo::Lower, "blocked:nb=128"}, {270, 2, Uplo::Lower, "blocked:nb=256"},
                                {50, 3, Uplo::Upper, "twosided"}};
    const auto rows = ts::reached_rows([&] {
        const ScopedEnvVar clear("BATCHLAS_SYGST_ROUTE", nullptr);
        for (const auto& r : runs) {
            auto g = this->gen(r.n, 2, r.uplo, 9u + r.n);
            if (!this->factor(g) || !this->run_sygst(g, r.itype, r.fam)) throw std::runtime_error("no route for " + r.fam);
        }
    });
    std::map<long long, std::string> algo;  // n -> spelling
    for (const auto& row : rows)
        if (row.op == "sygst") algo[row.m] = row.spelling;
    ASSERT_EQ(algo.size(), runs.size());
    for (const auto& r : runs) EXPECT_EQ(algo[r.n], r.fam) << "n=" << r.n << " itype=" << r.itype;
}

// sygvd on Auto: every itype, both jobz, both triangles, n through the syev CTA width; jobz=N's values
// must equal jobz=V's (whose residual is checked) within the values bound.
// ARMED BREAK: in src/ops/sygvd/sygvd.cc compose(), give the itype-3 trmm `Transpose::NoTrans` for Upper.
// SEEN (re-run 2026-10-27): RED for compose itype 3 Upper only, here and in FactoredEntry, all 8 configs;
// vendor GREEN.
// ARMED BREAK (2026-10-26): in compose(), B.hermitize(q, uplo) after potrf. SEEN: RED here for compose only,
// every itype, both triangles, n > 1, all 8 configs (B's poisoned triangle); vendor GREEN.
TYPED_TEST(SygvdTest, SygvdSolvesEveryItypeOnEveryFamily) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    for (const std::string& fam : this->sygvd_families())
        for (int itype : {1, 2, 3})
            for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
                for (int n : {1, 4, 33, 70}) {
                    select::ScopedPin<ops::sygvd::SygvdChoice> pin("sygvd", fam);
                    auto g = this->gen(n, 3, uplo, 31u + n * 7 + itype);
                    auto gn = g;
                    UnifiedVector<R> w, wn;
                    UnifiedVector<int32_t> info;
                    if (!this->run(g, w, info, itype, JobType::EigenVectors)) GTEST_SKIP() << "no sygvd route";
                    for (int it = 0; it < 3; ++it) ASSERT_EQ(info[it], 0) << fam << " item " << it;
                    const std::string at = fam + " itype=" + std::to_string(itype) + " uplo=" + ts::uplo_name(uplo) +
                                           " n=" + std::to_string(n);
                    if (fam == "compose") ASSERT_TRUE(ts::unwritten_outside<T>(g.b, Live::triangle(uplo))) << at << ": B";
                    for (int it = 0; it < 3; ++it)
                        for (int j = 1; j < n; ++j) ASSERT_LE(w[it * n + j - 1], w[it * n + j]) << "not ascending";
                    const auto [res, nrm] = this->check(g, w, itype);
                    EXPECT_VERIFY_SLACK(T, verify::Check::eigen_residual, n, res, cond_slack<T>()) << at;
                    EXPECT_VERIFY_SLACK(T, verify::Check::orthogonality, n, nrm, cond_slack<T>()) << at;
                    ASSERT_TRUE(this->run(gn, wn, info, itype, JobType::NoEigenVectors));
                    for (int it = 0; it < 3; ++it) ASSERT_EQ(info[it], 0) << fam << " N item " << it;
                    std::vector<std::vector<double>> ref(3);
                    double scale = 1;
                    for (int it = 0; it < 3; ++it)
                        for (int j = 0; j < n; ++j) {
                            ref[it].push_back(double(w[it * n + j]));
                            scale = std::max(scale, std::fabs(ref[it].back()));
                        }
                    const double ve = MV<T>::values_error(VectorView<R>(wn.data(), n, 3), ref, scale, std::vector<int>{0, 1, 2});
                    EXPECT_VERIFY_SLACK(T, verify::Check::values, n, ve, cond_slack<T>()) << at << " jobz=N";
                }
}

// Item 2's B is indefinite at order 4: info = n + 4 there, 0 elsewhere, and every other item's A, B and
// eigenvalues are bit-identical to a clean batch's on the same family. On compose the failed item's
// eigenvalues are finite (its B is replaced by the identity before sygst).
// ARMED BREAK: in src/ops/sygvd/kernels/sygvd_native.cc sygvd_merge_info, store `f` instead of `n + f`.
// SEEN (re-run 2026-10-27): RED for compose only (both triangles, all 8 configs); vendor GREEN.
TYPED_TEST(SygvdTest, NonPositiveDefiniteItemReportsNPlusIAndLeavesOthersBitIdentical) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    const std::vector<int> others{0, 1, 3, 4};
    for (const std::string& fam : this->sygvd_families())
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {6, 40}) {
                select::ScopedPin<ops::sygvd::SygvdChoice> pin("sygvd", fam);
                auto clean = this->gen(n, 5, uplo, 99u + n);
                auto bad = clean;
                bad.b(3, 3, 2) = MV<T>::make(-30.0, 0.0);  // leading minors 1..3 stay definite
                UnifiedVector<R> wc, wb;
                UnifiedVector<int32_t> ic, ib;
                if (!this->run(clean, wc, ic, 1, JobType::EigenVectors)) GTEST_SKIP() << "no sygvd route";
                ASSERT_TRUE(this->run(bad, wb, ib, 1, JobType::EigenVectors));
                const std::string at = fam + " uplo=" + ts::uplo_name(uplo) + " n=" + std::to_string(n);
                for (int it = 0; it < 5; ++it) EXPECT_EQ(ib[it], it == 2 ? n + 4 : 0) << at << " item " << it;
                ASSERT_TRUE(ts::items_bit_identical_to<T>(bad.a, clean.a, others)) << at << " A";
                ASSERT_TRUE(ts::items_bit_identical_to<T>(bad.b, clean.b, others)) << at << " B";
                ASSERT_TRUE(ts::items_bit_identical_to<R>(items_of(wb, n), items_of(wc, n), others)) << at << " w";
                if (fam == "compose")
                    for (int j = 0; j < n; ++j) EXPECT_TRUE(std::isfinite(double(wb[2 * n + j]))) << at << " failed item w";
            }
}

// sygvd_factored on potrf's factor: B is never written, a second A reuses the factor, and the result is
// bit-identical to sygvd's compose family on the unfactored B (the same potrf route factors it).
// ARMED BREAK: in src/ops/sygvd/sygvd.cc sygvd_factored, pass `false` for factored. SEEN: RED here only (NETLIB:
// "B written"; CUDA: the run's workspace check throws, its potrf scratch was never sized).
TYPED_TEST(SygvdTest, FactoredEntryReusesTheFactorAndMatchesCompose) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    for (int itype : {1, 3})
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {5, 40}) {
                const std::string at = "itype=" + std::to_string(itype) + " uplo=" + ts::uplo_name(uplo) + " n=" + std::to_string(n);
                auto ref = this->gen(n, 4, uplo, 501u + n + itype);
                auto g = ref;
                UnifiedVector<R> wr, wf;
                UnifiedVector<int32_t> ir, inf;
                {
                    select::ScopedPin<ops::sygvd::SygvdChoice> pin("sygvd", "compose");
                    if (!this->run(ref, wr, ir, itype, JobType::EigenVectors)) GTEST_SKIP() << "no sygvd route";
                }
                ASSERT_TRUE(this->factor(g));
                const Matrix<T> factor_bits = g.b;
                for (int rep = 0; rep < 2; ++rep) {
                    if (rep == 1) {  // a second A against the same factor
                        auto g2 = this->gen(n, 4, uplo, 777u + n + itype);
                        g.A0 = g2.A0, g.a = g2.a;
                    }
                    ASSERT_TRUE(this->run(g, wf, inf, itype, JobType::EigenVectors, true));
                    ASSERT_TRUE(ts::unwritten_outside<T>(g.b, Live::nothing(), factor_bits)) << at << " B";
                    for (int it = 0; it < 4; ++it) ASSERT_EQ(inf[it], 0) << at;
                    const auto [res, nrm] = this->check(g, wf, itype);
                    EXPECT_VERIFY_SLACK(T, verify::Check::eigen_residual, n, res, cond_slack<T>()) << at << " rep " << rep;
                    EXPECT_VERIFY_SLACK(T, verify::Check::orthogonality, n, nrm, cond_slack<T>()) << at << " rep " << rep;
                    if (rep == 0) {
                        ASSERT_TRUE(ts::items_bit_identical_to<R>(items_of(wf, n), items_of(wr, n))) << at << " w differs from compose";
                        ASSERT_TRUE(ts::items_bit_identical_to<T>(g.a, ref.a)) << at << " X differs from compose";
                    }
                }
            }
}

// 1024 identical items, each bit-identical to item 0, on every family (n = 12: syev's sub-group solvers;
// n = 40: blocked); the workspace query is exact (one byte less throws).
TYPED_TEST(SygvdTest, SaturatingBatchIsBitIdenticalToItemZeroAndWorkspaceIsExact) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend BK = TestFixture::BK;
    for (const std::string& fam : this->sygvd_families())
        for (int n : {12, 40}) {
            select::ScopedPin<ops::sygvd::SygvdChoice> pin("sygvd", fam);
            auto g = this->gen(n, 1024, Uplo::Lower, 4242u + n, true);
            UnifiedVector<R> w;
            UnifiedVector<int32_t> info;
            if (!this->run(g, w, info, 1, JobType::EigenVectors)) GTEST_SKIP() << "no sygvd route";
            for (int it = 0; it < 1024; ++it) ASSERT_EQ(info[it], 0) << fam << " n=" << n << " item " << it;
            ASSERT_TRUE(ts::items_bit_identical<R>(items_of(w, n))) << fam << " n=" << n << " w";
            ASSERT_TRUE(ts::items_bit_identical<T>(g.a)) << fam << " n=" << n << " X";
            auto h = this->gen(n, 4, Uplo::Lower, 1u);
            UnifiedVector<R> wh(static_cast<size_t>(n) * 4);
            const size_t need = sygvd_buffer_size<BK, T>(this->q(), h.a, h.b, wh.to_span(), 1, JobType::EigenVectors, Uplo::Lower);
            if (need == 0) continue;
            UnifiedVector<std::byte> ws(need);
            EXPECT_THROW(((void)sygvd<BK, T>(this->q(), h.a, h.b, wh.to_span(), 1, JobType::EigenVectors, Uplo::Lower,
                                           Span<std::byte>(ws.data(), need - 1), Span<int32_t>{})),
                         batchlas::workspace_error) << fam << " n=" << n;
        }
}

// Arguments no family serves are refused before anything is enqueued.
TYPED_TEST(SygvdTest, RejectsBadArguments) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend BK = TestFixture::BK;
    auto g = this->gen(4, 2, Uplo::Lower, 3u);
    UnifiedVector<R> w(8);
    UnifiedVector<std::byte> ws(1);
    for (int itype : {0, 4})
        EXPECT_THROW(((void)sygvd<BK, T>(this->q(), g.a, g.b, w.to_span(), itype, JobType::EigenVectors, Uplo::Lower,
                                       ws.to_span(), Span<int32_t>{})),
                     batchlas::invalid_argument);
    EXPECT_THROW(((void)sygvd<BK, T>(this->q(), g.a, g.b, Span<R>(w.data(), 7), 1, JobType::EigenVectors, Uplo::Lower,
                                   ws.to_span(), Span<int32_t>{})),
                 batchlas::invalid_argument);
    const MV<T> B3 = g.b.view(3, 3);
    EXPECT_THROW(((void)sygst<BK, T>(this->q(), g.a, B3, 1, Uplo::Lower, ws.to_span())), batchlas::invalid_argument);
    // The complex spellings resolve through the backend-deducing overload, as sygst and sygvd do.
    EXPECT_THROW(((void)hegst(this->q(), g.a.view(), B3, 1, Uplo::Lower, ws.to_span())), batchlas::invalid_argument);
    EXPECT_THROW(((void)hegvd(this->q(), g.a.view(), g.b.view(), w.to_span(), 0, JobType::EigenVectors, Uplo::Lower,
                              ws.to_span(), Span<int32_t>{})),
                 batchlas::invalid_argument);
}
