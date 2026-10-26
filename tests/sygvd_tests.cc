// sygst and sygvd: the generalized Hermitian-definite eigenproblem. Residuals and B-normalization through
// batchlas::verify (reference products, never the routes under test), graded B (cond ~1e2 float, ~1e4
// double), Hermitian A with nonzero imaginary parts, both triangles with the other one poisoned, padded ld
// and stride, every pinned family, a non-positive-definite item, the factored entry, a saturating batch.
// evidence: docs/perf/sygvd.md#sygvd-tests
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/blas/functions/sygvd.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <tests/test_utils.hh>

#include <src/ops/potrf/choice.hh>
#include <src/ops/sygst/choice.hh>
#include <src/ops/sygvd/choice.hh>
#include <src/select/select.hh>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;

template <typename T, Backend B>
struct SygvdConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <typename T>
bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <typename T>
constexpr double grade() { return std::is_same_v<RealOf<T>, float> ? 1.0 : 2.0; }  // decades of D's spread

// cond(B) <= 4 * 10^(2 grade): the generalized residual and normalization scale with it.
template <typename T>
verify::Slack cond_slack() { return {4.0 * std::pow(10.0, 2.0 * grade<T>()), "generalized problem: error ~ cond(B)"}; }

// A and B at padded ld and a stride that is not ld*n. The other triangle, the ld pads and the inter-item
// gaps hold a large finite value a kernel would happily consume.
template <typename T>
struct Gen {
    int n = 0, batch = 0, lda = 0, stra = 0, ldb = 0, strb = 0;
    Uplo uplo = Uplo::Lower;
    UnifiedVector<T> a, b;
    UnifiedVector<T*> ap, bp;
    std::vector<std::vector<T>> A0, B0;  // the whole Hermitian A and B per item, packed n x n

    MV<T> A() { return {a.data(), n, n, lda, stra, batch, ap.data()}; }
    MV<T> B() { return {b.data(), n, n, ldb, strb, batch, bp.data()}; }
    bool owned(int i, int j) const { return uplo == Uplo::Lower ? i >= j : i <= j; }
    T& at_a(int it, int i, int j) { return a[static_cast<size_t>(it) * stra + static_cast<size_t>(j) * lda + i]; }
    T& at_b(int it, int i, int j) { return b[static_cast<size_t>(it) * strb + static_cast<size_t>(j) * ldb + i]; }
    void store(int it) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (owned(i, j)) {
                    at_a(it, i, j) = A0[it][i + static_cast<size_t>(j) * n];
                    at_b(it, i, j) = B0[it][i + static_cast<size_t>(j) * n];
                }
    }
};

template <typename T>
T poison() { return MV<T>::make(9.0e3, -7.0e3); }

// identical: every item is item 0's data (the saturating-batch shape).
template <typename T>
Gen<T> make_gen(int n, int batch, Uplo uplo, unsigned seed, bool identical = false) {
    using R = RealOf<T>;
    Gen<T> g;
    g.n = n, g.batch = batch, g.uplo = uplo;
    g.lda = n + 3, g.stra = g.lda * n + 5, g.ldb = n + 2, g.strb = g.ldb * n + 7;
    g.a = UnifiedVector<T>(static_cast<size_t>(g.stra) * batch, poison<T>());
    g.b = UnifiedVector<T>(static_cast<size_t>(g.strb) * batch, poison<T>());
    g.ap = UnifiedVector<T*>(batch, nullptr);
    g.bp = UnifiedVector<T*>(batch, nullptr);
    g.A0.resize(batch);
    g.B0.resize(batch);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> d(R(-1), R(1));
    for (int it = 0; it < batch; ++it) {
        if (identical && it > 0) {
            g.A0[it] = g.A0[0], g.B0[it] = g.B0[0];
            g.store(it);
            continue;
        }
        std::vector<T> A(static_cast<size_t>(n) * n), S(static_cast<size_t>(n) * n);
        std::vector<double> D(n);
        for (int i = 0; i < n; ++i) D[i] = std::pow(10.0, grade<T>() * (n > 1 ? double(i) / (n - 1) - 0.5 : 0.0));
        for (int j = 0; j < n; ++j) {
            A[j + static_cast<size_t>(j) * n] = MV<T>::make(d(gen), 0.0);
            S[j + static_cast<size_t>(j) * n] = MV<T>::make(2.5 + 0.5 * std::fabs(d(gen)), 0.0);
            for (int i = j + 1; i < n; ++i) {
                const double x = d(gen), y = d(gen), u = d(gen) / n, v = d(gen) / n;
                A[i + static_cast<size_t>(j) * n] = MV<T>::make(x, y);
                A[j + static_cast<size_t>(i) * n] = MV<T>::make(x, -y);
                S[i + static_cast<size_t>(j) * n] = MV<T>::make(u, v);
                S[j + static_cast<size_t>(i) * n] = MV<T>::make(u, -v);
            }
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) S[i + static_cast<size_t>(j) * n] *= RealOf<T>(D[i] * D[j]);
        g.A0[it] = std::move(A), g.B0[it] = std::move(S);
        g.store(it);
    }
    return g;
}

template <typename T>
Matrix<T> packed(const std::vector<std::vector<T>>& m, int n) {
    Matrix<T> M(n, n, static_cast<int>(m.size()));
    for (size_t it = 0; it < m.size(); ++it)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) M(i, j, static_cast<int>(it)) = m[it][i + static_cast<size_t>(j) * n];
    return M;
}

template <typename Config>
class SygvdTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = RealOf<T>;
    static constexpr Backend BK = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        // cuSOLVER's Upper potrf overwrites the lower triangle (agent guide section 8 rule 4), which would
        // wipe B's poison before sygst or the back-transform read it; the native potrf leaves it alone.
        if constexpr (BK != Backend::NETLIB) potrf_pin_.emplace("potrf", "native");
    }

    Queue& q() { return *this->ctx; }
    std::optional<select::ScopedPin<ops::potrf::PotrfChoice>> potrf_pin_;

    // The other triangle of every item's B still holds the poison (only the uplo triangle is read).
    void expect_b_other_triangle_poisoned(Gen<T>& g, const std::string& at) {
        for (int it = 0; it < g.batch; ++it)
            for (int j = 0; j < g.n; ++j)
                for (int i = 0; i < g.n; ++i)
                    if (!g.owned(i, j))
                        ASSERT_TRUE(same_bits(g.at_b(it, i, j), poison<T>())) << at << ": B(" << i << "," << j << ") item " << it;
    }

    // Whether the build links sygvd's (and, on the host, sygst's) vendor family for this backend.
    static constexpr bool vendor_linked() {
        if constexpr (BK == Backend::CUDA) return BATCHLAS_HAS_CUSOLVER;
        if constexpr (BK == Backend::NETLIB) return BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS;
        return false;
    }
    static constexpr bool sygst_vendor_linked() { return BK == Backend::NETLIB && vendor_linked(); }

    // The eigenvectors of the call as a packed batch (A's storage).
    Matrix<T> vectors(Gen<T>& g) {
        Matrix<T> X(g.n, g.n, g.batch);
        for (int it = 0; it < g.batch; ++it)
            for (int j = 0; j < g.n; ++j)
                for (int i = 0; i < g.n; ++i) X(i, j, it) = g.at_a(it, i, j);
        return X;
    }

    // {residual, normalization} of itype's problem: 1: ||A X - B X W||, 2: ||A (B X) - X W||,
    // 3: ||B (A X) - X W||, each over ||op|| ||X||; X^H B X = I (1, 2) or B = X X^H (3).
    std::pair<double, double> check(Gen<T>& g, const UnifiedVector<R>& w, int itype) {
        const int n = g.n, batch = g.batch;
        Matrix<T> A0 = packed(g.A0, n), B0 = packed(g.B0, n), X = vectors(g);
        Matrix<T> P(n, n, batch), XW(n, n, batch), I(n, n, batch);
        (void)MV<T>::product(q(), itype == 3 ? A0.view() : B0.view(), Transpose::NoTrans, X.view(), Transpose::NoTrans, P.view());
        (void)I.view().fill_identity(q());
        q().wait();
        for (int it = 0; it < batch; ++it)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const T s = T(w[static_cast<size_t>(it) * n + j]);
                    XW(i, j, it) = (itype == 1 ? P(i, j, it) : X(i, j, it)) * s;
                }
        const std::vector<int> all = A0.view().all_items();
        double res = 0;
        if (itype == 1) res = MV<T>::solve_residual(q(), A0, X, XW, all);
        if (itype == 2) res = MV<T>::solve_residual(q(), A0, P, XW, all);
        if (itype == 3) res = MV<T>::solve_residual(q(), B0, P, XW, all);
        const double nrm = itype == 3 ? MV<T>::similarity_residual(q(), B0, X, I, all)
                                      : MV<T>::solve_residual(q(), X, Transpose::ConjTrans, P, MV<T>{}, all);
        return {res, nrm};
    }

    size_t ws_size(Gen<T>& g, UnifiedVector<R>& w, int itype, JobType jobz, bool factored) {
        return factored ? sygvd_factored_buffer_size<BK, T>(q(), g.A(), g.B(), w.to_span(), itype, jobz, g.uplo)
                        : sygvd_buffer_size<BK, T>(q(), g.A(), g.B(), w.to_span(), itype, jobz, g.uplo);
    }

    // sygvd (or sygvd_factored) with exactly the queried workspace; false on a NoRouteError.
    bool run(Gen<T>& g, UnifiedVector<R>& w, UnifiedVector<int32_t>& info, int itype, JobType jobz,
             bool factored = false) {
        w = UnifiedVector<R>(static_cast<size_t>(g.n) * g.batch, R(-4321));
        info = UnifiedVector<int32_t>(static_cast<size_t>(g.batch), int32_t(-12345));
        try {
            const size_t need = ws_size(g, w, itype, jobz, factored);
            UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
            const Span<std::byte> s(ws.data(), need);
            if (factored)
                (void)sygvd_factored<BK, T>(q(), g.A(), g.B(), w.to_span(), itype, jobz, g.uplo, s, info.to_span());
            else
                (void)sygvd<BK, T>(q(), g.A(), g.B(), w.to_span(), itype, jobz, g.uplo, s, info.to_span());
            q().wait();
        } catch (const NoRouteError&) {
            return false;
        }
        return true;
    }

    bool factor(Gen<T>& g) {
        UnifiedVector<int32_t> info(static_cast<size_t>(g.batch), 0);
        try {
            const size_t need = potrf_buffer_size<BK, T>(q(), g.B(), g.uplo);
            UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
            (void)potrf<BK, T>(q(), g.B(), g.uplo, Span<std::byte>(ws.data(), need), info.to_span());
            q().wait();
        } catch (const NoRouteError&) {
            return false;
        }
        for (int it = 0; it < g.batch; ++it) EXPECT_EQ(info[it], 0) << "potrf item " << it;
        // Re-poison the other triangle whatever potrf did to it, so a reader of the wrong triangle shows.
        for (int it = 0; it < g.batch; ++it)
            for (int j = 0; j < g.n; ++j)
                for (int i = 0; i < g.n; ++i)
                    if (!g.owned(i, j)) g.at_b(it, i, j) = poison<T>();
        return true;
    }

    // The families a pin must reach on this backend; each element pins sygvd to one.
    std::vector<std::string> sygvd_families() const {
        std::vector<std::string> f{"compose"};
        if (vendor_linked()) f.push_back("vendor");
        return f;
    }
};

using SygvdTestTypes = typename test_utils::backend_types<SygvdConfig>::type;

const char* ul(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }

}  // namespace

TYPED_TEST_SUITE(SygvdTest, SygvdTestTypes);

// sygst against its definition, from potrf's factor: itype 1 A0 = L C L^H (Lower) or U^H C U (Upper);
// itypes 2, 3 C = L^H A0 L or U A0 U^H (similarity_residual with the factor built on the host). Every
// family it can pin, both triangles, an order ladder through the CTA width, B read-only (pads too).
// ARMED BREAK: in src/ops/sygst/sygst.cc twosided(), give the itype-1 Right trsm `Transpose::NoTrans` for
// Lower. SEEN (2026-10-25): RED here for twosided itype 1 Lower only, all 8 configs; on CUDA also compose
// itype 1 Lower in the two sygvd tests below (NETLIB's compose takes LAPACKE's sygst, its last resort).
TYPED_TEST(SygvdTest, SygstMatchesTheTwoSidedDefinition) {
    using T = typename TestFixture::T;
    constexpr Backend BK = TestFixture::BK;
    std::vector<std::string> fams{"twosided"};
    if (TestFixture::sygst_vendor_linked()) fams.push_back("vendor");
    for (const std::string& fam : fams)
        for (int itype : {1, 2, 3})
            for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
                for (int n : {1, 5, 33, 70}) {
                    auto g = make_gen<T>(n, 3, uplo, 7u + n * 13 + itype);
                    if (!this->factor(g)) GTEST_SKIP() << "no potrf route in this build";
                    const UnifiedVector<T> factor_bits = g.b;
                    select::ScopedPin<ops::sygst::SygstChoice> pin("sygst", fam);
                    size_t need = 0;
                    try {
                        need = sygst_buffer_size<BK, T>(this->q(), g.A(), g.B(), itype, uplo);
                    } catch (const NoRouteError&) {
                        GTEST_SKIP() << "no sygst route in this build";
                    }
                    UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
                    (void)sygst<BK, T>(this->q(), g.A(), g.B(), itype, uplo, Span<std::byte>(ws.data(), need));
                    this->q().wait();
                    // C mirrored from the uplo triangle; F the factor with the other triangle zeroed.
                    Matrix<T> C(n, n, 3), F(n, n, 3), A0 = packed(g.A0, n);
                    for (int it = 0; it < 3; ++it)
                        for (int j = 0; j < n; ++j)
                            for (int i = 0; i < n; ++i) {
                                const bool own = g.owned(i, j);
                                C(i, j, it) = own ? g.at_a(it, i, j) : MV<T>::conj(g.at_a(it, j, i));
                                if (i == j) C(i, j, it) = MV<T>::make(double(std::real(C(i, j, it))));
                                F(i, j, it) = own ? g.at_b(it, i, j) : T(0);
                            }
                    // Q such that the check is ||lhs - Q mid Q^H|| / ||lhs||.
                    Matrix<T> Q(n, n, 3);
                    const bool qh = (itype == 1) != (uplo == Uplo::Lower);  // Q = F^H
                    for (int it = 0; it < 3; ++it)
                        for (int j = 0; j < n; ++j)
                            for (int i = 0; i < n; ++i) Q(i, j, it) = qh ? MV<T>::conj(F(j, i, it)) : F(i, j, it);
                    const std::vector<int> all{0, 1, 2};
                    const double r = itype == 1 ? MV<T>::similarity_residual(this->q(), A0, Q, C, all)
                                                : MV<T>::similarity_residual(this->q(), C, Q, A0, all);
                    EXPECT_VERIFY_SLACK(T, verify::Check::factorization, n, r, cond_slack<T>())
                        << fam << " itype=" << itype << " uplo=" << ul(uplo) << " n=" << n;
                    for (size_t i = 0; i < g.b.size(); ++i)
                        ASSERT_TRUE(same_bits(g.b[i], factor_bits[i])) << fam << ": B written at " << i;
                }
}

// sygvd on Auto: every itype, both jobz, both triangles, n through the syev CTA width; jobz=N's values
// must equal jobz=V's (whose residual is checked) within the values bound.
// ARMED BREAK: in src/ops/sygvd/sygvd.cc compose(), give the itype-3 trmm `Transpose::NoTrans` for Upper.
// SEEN: RED for compose itype 3 Upper only, all 8 configs; vendor GREEN.
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
                    auto g = make_gen<T>(n, 3, uplo, 31u + n * 7 + itype);
                    UnifiedVector<R> w, wn;
                    UnifiedVector<int32_t> info;
                    if (!this->run(g, w, info, itype, JobType::EigenVectors)) GTEST_SKIP() << "no sygvd route";
                    for (int it = 0; it < g.batch; ++it) ASSERT_EQ(info[it], 0) << fam << " item " << it;
                    if (fam == "compose") this->expect_b_other_triangle_poisoned(g, fam + " itype=" + std::to_string(itype));
                    for (int it = 0; it < g.batch; ++it)
                        for (int j = 1; j < n; ++j) ASSERT_LE(w[it * n + j - 1], w[it * n + j]) << "not ascending";
                    const auto [res, nrm] = this->check(g, w, itype);
                    const std::string at = fam + " itype=" + std::to_string(itype) + " uplo=" + ul(uplo) +
                                           " n=" + std::to_string(n);
                    EXPECT_VERIFY_SLACK(T, verify::Check::eigen_residual, n, res, cond_slack<T>()) << at;
                    EXPECT_VERIFY_SLACK(T, verify::Check::orthogonality, n, nrm, cond_slack<T>()) << at;
                    auto gn = make_gen<T>(n, 3, uplo, 31u + n * 7 + itype);
                    ASSERT_TRUE(this->run(gn, wn, info, itype, JobType::NoEigenVectors));
                    for (int it = 0; it < g.batch; ++it) ASSERT_EQ(info[it], 0) << fam << " N item " << it;
                    std::vector<std::vector<double>> ref(g.batch);
                    double scale = 0;
                    for (int it = 0; it < g.batch; ++it)
                        for (int j = 0; j < n; ++j) {
                            ref[it].push_back(double(w[it * n + j]));
                            scale = std::max(scale, std::fabs(double(w[it * n + j])));
                        }
                    const std::vector<int> items{0, 1, 2};
                    const double ve = MV<T>::values_error(VectorView<R>(wn.data(), n, g.batch), ref,
                                                          std::max(scale, 1.0), items);
                    EXPECT_VERIFY_SLACK(T, verify::Check::values, n, ve, cond_slack<T>()) << at << " jobz=N";
                }
}

// Item 2's B is indefinite at order 4: info = n + 4 there, 0 elsewhere, and every other item's A, B and
// eigenvalues are bit-identical to a clean batch's on the same family. On compose the failed item's
// eigenvalues are finite (its B is replaced by the identity before sygst).
// ARMED BREAK: in src/ops/sygvd/kernels/sygvd_native.cc sygvd_merge_info, store `f` instead of `n + f`.
// SEEN: RED for compose only (both triangles, both n, all 8 configs); vendor GREEN.
TYPED_TEST(SygvdTest, NonPositiveDefiniteItemReportsNPlusIAndLeavesOthersBitIdentical) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    for (const std::string& fam : this->sygvd_families())
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {6, 40}) {
                select::ScopedPin<ops::sygvd::SygvdChoice> pin("sygvd", fam);
                auto clean = make_gen<T>(n, 5, uplo, 99u + n);
                auto bad = make_gen<T>(n, 5, uplo, 99u + n);
                const R dmax = R(30) * R(std::pow(10.0, grade<T>()));
                bad.B0[2][3 + static_cast<size_t>(3) * n] = MV<T>::make(-double(dmax), 0.0);
                bad.store(2);
                UnifiedVector<R> wc, wb;
                UnifiedVector<int32_t> ic, ib;
                if (!this->run(clean, wc, ic, 1, JobType::EigenVectors)) GTEST_SKIP() << "no sygvd route";
                ASSERT_TRUE(this->run(bad, wb, ib, 1, JobType::EigenVectors));
                const std::string at = fam + " uplo=" + ul(uplo) + " n=" + std::to_string(n);
                for (int it = 0; it < 5; ++it) {
                    EXPECT_EQ(ib[it], it == 2 ? n + 4 : 0) << at << " item " << it;
                    if (it == 2) continue;
                    for (size_t o = 0; o < static_cast<size_t>(clean.stra); ++o)
                        ASSERT_TRUE(same_bits(bad.a[it * clean.stra + o], clean.a[it * clean.stra + o])) << at << " A item " << it;
                    for (size_t o = 0; o < static_cast<size_t>(clean.strb); ++o)
                        ASSERT_TRUE(same_bits(bad.b[it * clean.strb + o], clean.b[it * clean.strb + o])) << at << " B item " << it;
                    for (int j = 0; j < n; ++j) ASSERT_TRUE(same_bits(wb[it * n + j], wc[it * n + j])) << at << " w item " << it;
                }
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
                const std::string at = "itype=" + std::to_string(itype) + " uplo=" + ul(uplo) + " n=" + std::to_string(n);
                auto ref = make_gen<T>(n, 4, uplo, 501u + n + itype);
                UnifiedVector<R> wr, wf;
                UnifiedVector<int32_t> ir, inf;
                {
                    select::ScopedPin<ops::sygvd::SygvdChoice> pin("sygvd", "compose");
                    if (!this->run(ref, wr, ir, itype, JobType::EigenVectors)) GTEST_SKIP() << "no sygvd route";
                }
                auto g = make_gen<T>(n, 4, uplo, 501u + n + itype);
                ASSERT_TRUE(this->factor(g));
                const UnifiedVector<T> factor_bits = g.b;
                for (int rep = 0; rep < 2; ++rep) {
                    if (rep == 1) {  // a second A against the same factor
                        auto g2 = make_gen<T>(n, 4, uplo, 777u + n + itype);
                        g.A0 = g2.A0;
                        for (size_t i = 0; i < g.a.size(); ++i) g.a[i] = g2.a[i];
                    }
                    ASSERT_TRUE(this->run(g, wf, inf, itype, JobType::EigenVectors, true));
                    for (size_t i = 0; i < g.b.size(); ++i) ASSERT_TRUE(same_bits(g.b[i], factor_bits[i])) << at << " B written";
                    for (int it = 0; it < 4; ++it) ASSERT_EQ(inf[it], 0) << at;
                    auto chk = g;
                    chk.B0 = ref.B0;
                    const auto [res, nrm] = this->check(chk, wf, itype);
                    EXPECT_VERIFY_SLACK(T, verify::Check::eigen_residual, n, res, cond_slack<T>()) << at << " rep " << rep;
                    EXPECT_VERIFY_SLACK(T, verify::Check::orthogonality, n, nrm, cond_slack<T>()) << at << " rep " << rep;
                    if (rep == 0) {
                        for (size_t i = 0; i < wr.size(); ++i) ASSERT_TRUE(same_bits(wf[i], wr[i])) << at << " w differs from compose";
                        for (size_t i = 0; i < g.a.size(); ++i) ASSERT_TRUE(same_bits(g.a[i], ref.a[i])) << at << " X differs from compose";
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
            auto g = make_gen<T>(n, 1024, Uplo::Lower, 4242u + n, true);
            UnifiedVector<R> w;
            UnifiedVector<int32_t> info;
            if (!this->run(g, w, info, 1, JobType::EigenVectors)) GTEST_SKIP() << "no sygvd route";
            for (int it = 1; it < g.batch; ++it) {
                ASSERT_EQ(info[it], 0);
                for (int j = 0; j < n; ++j) ASSERT_TRUE(same_bits(w[it * n + j], w[j])) << fam << " n=" << n << " item " << it;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        ASSERT_TRUE(same_bits(g.at_a(it, i, j), g.at_a(0, i, j))) << fam << " n=" << n << " item " << it;
            }
            auto h = make_gen<T>(n, 4, Uplo::Lower, 1u);
            UnifiedVector<R> wh(static_cast<size_t>(n) * 4);
            const size_t need = sygvd_buffer_size<BK, T>(this->q(), h.A(), h.B(), wh.to_span(), 1, JobType::EigenVectors, Uplo::Lower);
            if (need == 0) continue;
            UnifiedVector<std::byte> ws(need);
            EXPECT_THROW(((void)sygvd<BK, T>(this->q(), h.A(), h.B(), wh.to_span(), 1, JobType::EigenVectors, Uplo::Lower,
                                           Span<std::byte>(ws.data(), need - 1), Span<int32_t>{})),
                         batchlas::workspace_error) << fam << " n=" << n;
        }
}

// Arguments no family serves are refused before anything is enqueued.
TYPED_TEST(SygvdTest, RejectsBadArguments) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend BK = TestFixture::BK;
    auto g = make_gen<T>(4, 2, Uplo::Lower, 3u);
    UnifiedVector<R> w(8);
    UnifiedVector<std::byte> ws(1);
    for (int itype : {0, 4})
        EXPECT_THROW(((void)sygvd<BK, T>(this->q(), g.A(), g.B(), w.to_span(), itype, JobType::EigenVectors, Uplo::Lower,
                                       ws.to_span(), Span<int32_t>{})),
                     batchlas::invalid_argument);
    EXPECT_THROW(((void)sygvd<BK, T>(this->q(), g.A(), g.B(), Span<R>(w.data(), 7), 1, JobType::EigenVectors, Uplo::Lower,
                                   ws.to_span(), Span<int32_t>{})),
                 batchlas::invalid_argument);
    MV<T> B3(g.b.data(), 3, 3, g.ldb, g.strb, 2, g.bp.data());
    EXPECT_THROW(((void)sygst<BK, T>(this->q(), g.A(), B3, 1, Uplo::Lower, ws.to_span())), batchlas::invalid_argument);
    // The complex spellings resolve through the backend-deducing overload, as sygst and sygvd do.
    EXPECT_THROW(((void)hegst(this->q(), g.A(), B3, 1, Uplo::Lower, ws.to_span())), batchlas::invalid_argument);
    EXPECT_THROW(((void)hegvd(this->q(), g.A(), g.B(), w.to_span(), 0, JobType::EigenVectors, Uplo::Lower, ws.to_span(),
                              Span<int32_t>{})),
                 batchlas::invalid_argument);
}
