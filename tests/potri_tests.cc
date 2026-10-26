// potri (inverse from a Cholesky factor; potrs's own cases are in potrs_tests.cc): residuals through
// batchlas::verify, per-item info with the other items undisturbed, padded ld and stride with a
// poisoned unreferenced triangle, and a saturating batch. The op is native and vendor-free, so no
// oracle is a vendor library.
// evidence: docs/perf/potri.md#potrs-and-potri
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/potri.hh>
#include <batchlas/no_route.hh>
#include <batchlas/blas/functions/potrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_utils.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/posv/choice.hh>
#include <src/ops/potri/choice.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;

template <typename T, Backend B>
struct PotriConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <typename T>
bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// One Hermitian positive-definite item, strictly diagonally dominant (condition below 8), every
// off-diagonal entry with a nonzero imaginary part. Column-major n x n.
template <typename T>
std::vector<T> make_hpd(int n, std::mt19937& gen) {
    using R = RealOf<T>;
    std::uniform_real_distribution<R> d(R(0.1), R(1));
    std::vector<T> A(static_cast<size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
        A[j + static_cast<size_t>(j) * n] = MV<T>::make(R(2.5) + R(0.5) * d(gen), R(0));
        for (int i = j + 1; i < n; ++i) {
            const R x = (gen() & 1) ? d(gen) : -d(gen);
            const R y = d(gen);
            A[i + static_cast<size_t>(j) * n] = MV<T>::make(x / R(n), y / R(n));
            A[j + static_cast<size_t>(i) * n] = MV<T>::make(x / R(n), -y / R(n));
        }
    }
    return A;
}

// A and B at padded ld and a stride that is not ld*cols. Everything the op may not touch (the
// other triangle of A, the ld pads, the inter-item gaps) holds a large finite value a kernel
// would happily consume: reading it wrecks a residual, writing it is caught by memcmp.
template <typename T>
struct Sys {
    int n = 0, nrhs = 0, batch = 0, lda = 0, stra = 0, ldb = 0, strb = 0;
    Uplo uplo = Uplo::Lower;
    UnifiedVector<T> a, b;
    UnifiedVector<T*> aptr, bptr;
    std::vector<std::vector<T>> full, rhs;  // the whole Hermitian A and the n x nrhs B per item

    MV<T> A() { return {a.data(), n, n, lda, stra, batch, aptr.data()}; }
    MV<T> B() { return {b.data(), n, nrhs, ldb, strb, batch, bptr.data()}; }
    bool owned(int i, int j) const { return uplo == Uplo::Lower ? i >= j : i <= j; }
};

// identical: every item is item 0's data (the saturating-batch shape).
template <typename T>
Sys<T> make_sys(int n, int nrhs, int batch, Uplo uplo, unsigned seed, bool identical = false) {
    using R = RealOf<T>;
    Sys<T> p;
    p.n = n, p.nrhs = std::max(nrhs, 1), p.batch = batch, p.uplo = uplo;
    p.lda = n + 3, p.stra = p.lda * n + 5, p.ldb = n + 2, p.strb = p.ldb * p.nrhs + 7;
    const T poison = MV<T>::make(R(-999), R(777));
    p.a = UnifiedVector<T>(static_cast<size_t>(p.stra) * batch, poison);
    p.b = UnifiedVector<T>(static_cast<size_t>(p.strb) * batch, poison);
    p.aptr = UnifiedVector<T*>(batch, nullptr);
    p.bptr = UnifiedVector<T*>(batch, nullptr);
    p.full.resize(batch);
    p.rhs.resize(batch);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> d(R(-1), R(1));
    for (int it = 0; it < batch; ++it) {
        if (identical && it > 0) {
            p.full[it] = p.full[0];
            p.rhs[it] = p.rhs[0];
        } else {
            p.full[it] = make_hpd<T>(n, gen);
            p.rhs[it].resize(static_cast<size_t>(n) * p.nrhs);
            for (auto& v : p.rhs[it]) v = MV<T>::make(d(gen), d(gen));
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (p.owned(i, j))
                    p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i] =
                        p.full[it][i + static_cast<size_t>(j) * n];
        for (int k = 0; k < p.nrhs; ++k)
            for (int i = 0; i < n; ++i)
                p.b[static_cast<size_t>(it) * p.strb + static_cast<size_t>(k) * p.ldb + i] =
                    p.rhs[it][i + static_cast<size_t>(k) * n];
    }
    return p;
}

template <typename Config>
class PotriTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend BK = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        // NETLIB serves potrf and trsm through LAPACKE/CBLAS only (no CPU-native kernel), so a
        // vendor-free build has nothing to test on the host queue.
        if (BK == Backend::NETLIB && !(BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS))
            GTEST_SKIP() << "NETLIB without LAPACKE/CBLAS has no potrf or trsm route";
    }

    Queue& q() { return *this->ctx; }
    // The fused solve (potri `fused`, posv `cta`) needs a GPU with sub-group size 32; `blocked` runs anywhere.
    bool fused_ok() { return q().device().type == DeviceType::GPU && q().device().supports_sub_group_size(32); }

    // The Cholesky factor in A's triangle, through the public potrf; returns its info.
    // False when the build has no potrf route for the shape (vendor-free: the native potrf does not cover it).
    bool factor(Sys<T>& p, UnifiedVector<int32_t>& info) {
        info = UnifiedVector<int32_t>(static_cast<size_t>(p.batch), int32_t(-12345));
        try {
            const size_t need = potrf_buffer_size<BK, T>(q(), p.A(), p.uplo);
            UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
            (void)potrf<BK, T>(q(), p.A(), p.uplo, Span<std::byte>(ws.data(), need), info.to_span());
        } catch (const NoRouteError&) {
            return false;
        }
        q().wait();
        return true;
    }

    Event run_potri(Sys<T>& p, bool full, UnifiedVector<int32_t>* info = nullptr) {
        const size_t need = potri_buffer_size<BK, T>(q(), p.A(), p.uplo);
        ws_ = UnifiedVector<std::byte>(std::max<size_t>(need, 1));
        Span<int32_t> s = info ? info->to_span() : Span<int32_t>{};
        auto ev = potri<BK, T>(q(), p.A(), p.uplo, full, Span<std::byte>(ws_.data(), need), s);
        ev.wait();
        q().wait();
        return ev;
    }

    // max over items of ||A0 X - I|| / (||A0|| ||X||), X read from item @p x_of(item) of a packed
    // n x n batch, A0 the original full Hermitian matrices.
    double inverse_residual(const Sys<T>& p, const std::vector<std::vector<T>>& inv) {
        Matrix<T> A0(p.n, p.n, p.batch), X(p.n, p.n, p.batch);
        for (int it = 0; it < p.batch; ++it)
            for (int j = 0; j < p.n; ++j)
                for (int i = 0; i < p.n; ++i) {
                    A0(i, j, it) = p.full[it][i + static_cast<size_t>(j) * p.n];
                    X(i, j, it) = inv[it][i + static_cast<size_t>(j) * p.n];
                }
        return MV<T>::solve_residual(q(), A0, X, MV<T>{});
    }

    // Item @p it of A as a packed n x n matrix, the triangle mirrored (Hermitian) onto the rest.
    std::vector<T> mirrored(const Sys<T>& p, int it) const {
        std::vector<T> out(static_cast<size_t>(p.n) * p.n);
        for (int j = 0; j < p.n; ++j)
            for (int i = 0; i < p.n; ++i) {
                const T v = p.owned(i, j) ? p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i]
                                          : p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(i) * p.lda + j];
                out[i + static_cast<size_t>(j) * p.n] = p.owned(i, j) ? v : MV<T>::make(double(std::real(v)), -double(std::imag(v)));
            }
        return out;
    }

    double solve_res(const Sys<T>& p) {
        Matrix<T> A0(p.n, p.n, p.batch), X(p.n, p.nrhs, p.batch), B0(p.n, p.nrhs, p.batch);
        for (int it = 0; it < p.batch; ++it) {
            for (int j = 0; j < p.n; ++j)
                for (int i = 0; i < p.n; ++i) A0(i, j, it) = p.full[it][i + static_cast<size_t>(j) * p.n];
            for (int k = 0; k < p.nrhs; ++k)
                for (int i = 0; i < p.n; ++i) {
                    X(i, k, it) = p.b[static_cast<size_t>(it) * p.strb + static_cast<size_t>(k) * p.ldb + i];
                    B0(i, k, it) = p.rhs[it][i + static_cast<size_t>(k) * p.n];
                }
        }
        return MV<T>::solve_residual(q(), A0, X, B0);
    }

    bool outside_a_untouched(const Sys<T>& p, const UnifiedVector<T>& before, bool full) const {
        for (int it = 0; it < p.batch; ++it)
            for (size_t o = 0; o < static_cast<size_t>(p.stra); ++o) {
                const int col = static_cast<int>(o) / p.lda, row = static_cast<int>(o) % p.lda;
                const bool in = col < p.n && row < p.n && static_cast<int>(o) < p.lda * p.n;
                if (in && (p.owned(row, col) || full)) continue;
                const size_t idx = static_cast<size_t>(it) * p.stra + o;
                if (!same_bits(p.a[idx], before[idx])) return false;
            }
        return true;
    }

    UnifiedVector<std::byte> ws_;
};

using PotriTestTypes = typename test_utils::backend_types<PotriConfig>::type;

}  // namespace

TYPED_TEST_SUITE(PotriTest, PotriTestTypes);

// potri: A * A^-1 = I through verify, both triangles, Auto and each pin, an order ladder through the
// fused ceiling (8), padded ld and stride, the unreferenced triangle poisoned and written by nobody.
// ARMED BREAK: in potri.cc launch() drop the fill_identity. EXPECTED: RED in every potri test that
// reads the inverse, the potrs tests GREEN.
TYPED_TEST(PotriTest, PotriInvertsAndWritesOnlyItsTriangle) {
    using T = typename TestFixture::T;
    using namespace ops::potri;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {1, 2, 5, 8, 9, 16, 31, 64, 100})
            for (int route = 0; route < 3; ++route) {  // 0 auto, 1 fused, 2 blocked
                if (route == 1 && (n > 8 || !this->fused_ok())) continue;
                auto p = make_sys<T>(n, 1, 4, uplo, 900u + n);
                UnifiedVector<int32_t> info;
                if (!this->factor(p, info)) continue;
                const UnifiedVector<T> before = p.a;
                UnifiedVector<int32_t> pinfo(p.batch, int32_t(-7));
                {
                    std::optional<select::ScopedPin<PotriChoice>> pin;
                    if (route == 1) pin.emplace("potri", PotriChoice{Fused{}});
                    if (route == 2) pin.emplace("potri", PotriChoice{Blocked{}});
                    (void)this->run_potri(p, false, &pinfo);
                }
                for (int it = 0; it < p.batch; ++it) ASSERT_EQ(pinfo[it], 0);
                std::vector<std::vector<T>> inv;
                for (int it = 0; it < p.batch; ++it) inv.push_back(this->mirrored(p, it));
                EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, inv))
                    << "n=" << n << " route=" << route << " uplo=" << (uplo == Uplo::Lower ? "L" : "U");
                EXPECT_TRUE(this->outside_a_untouched(p, before, false)) << "n=" << n << " route=" << route;
            }
}

// With full = true the other triangle is exactly the conjugate transpose (bit for bit: a complex
// inverse with nonzero imaginary parts is Hermitian only if the mirror conjugates), and the whole
// matrix is the inverse with no mirroring on the caller's side.
// BREAK (run 2026-10-25): potri_store drops the dev_conj. RED: this test for cfloat and cdouble
// only (NETLIB and CUDA); real types and every other test GREEN.
TYPED_TEST(PotriTest, PotriFullFillsTheOtherTriangleWithTheConjugateTranspose) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {3, 8, 17, 40}) {
            auto p = make_sys<T>(n, 1, 3, uplo, 31u + n);
            UnifiedVector<int32_t> info;
            if (!this->factor(p, info)) continue;
            const UnifiedVector<T> before = p.a;
            (void)this->run_potri(p, true);
            std::vector<std::vector<T>> inv;
            for (int it = 0; it < p.batch; ++it) {
                std::vector<T> whole(static_cast<size_t>(n) * n);
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) whole[i + static_cast<size_t>(j) * n] = p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i];
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) {
                        const T a = whole[i + static_cast<size_t>(j) * n], b = whole[j + static_cast<size_t>(i) * n];
                        // Diagonals: a real value (a bitwise +0 against -0 is not a defect).
                        if (i == j) ASSERT_EQ(std::imag(a), 0) << "(" << i << "," << j << ")";
                        else ASSERT_TRUE(same_bits(a, MV<T>::make(double(std::real(b)), -double(std::imag(b))))) << "(" << i << "," << j << ")";
                    }
                inv.push_back(whole);
            }
            EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, inv)) << "n=" << n;
            EXPECT_TRUE(this->outside_a_untouched(p, before, true));
        }
}

// info follows ?potri: the first exactly-zero factor diagonal, per item, and the other items'
// inverses are bit-identical to a clean batch's.
// BREAK (run 2026-10-25): potri_diag_info starts the scan at i = 1. RED: this test only (the
// zero at index 0), every type and backend.
TYPED_TEST(PotriTest, PotriInfoNamesTheZeroDiagonalAndLeavesOtherItemsAlone) {
    using T = typename TestFixture::T;
    for (int n : {5, 8, 20})
        for (int zero_at : {0, n / 2, n - 1}) {
            const int bad = 1, batch = 4;
            auto clean = make_sys<T>(n, 1, batch, Uplo::Lower, 5u + n);
            UnifiedVector<int32_t> info;
            if (!this->factor(clean, info)) continue;
            auto dirty = make_sys<T>(n, 1, batch, Uplo::Lower, 5u + n);
            dirty.a = clean.a;
            dirty.a[static_cast<size_t>(bad) * dirty.stra + static_cast<size_t>(zero_at) * dirty.lda + zero_at] = MV<T>::make(0.0, 0.0);
            UnifiedVector<int32_t> ic(batch, -1), id(batch, -1);
            (void)this->run_potri(clean, false, &ic);
            (void)this->run_potri(dirty, false, &id);
            for (int it = 0; it < batch; ++it) {
                EXPECT_EQ(ic[it], 0);
                EXPECT_EQ(id[it], it == bad ? zero_at + 1 : 0) << "item " << it;
                if (it == bad) continue;
                for (size_t o = 0; o < static_cast<size_t>(dirty.stra); ++o)
                    ASSERT_TRUE(same_bits(dirty.a[static_cast<size_t>(it) * dirty.stra + o], clean.a[static_cast<size_t>(it) * clean.stra + o]))
                        << "item " << it << " disturbed at " << o;
            }
        }
}

// The workspace is one n x n identity per item: sized by potri_buffer_size, enforced, and the
// empty-batch, empty-problem and heterogeneous cases are refused or no-ops, not wrong answers.
TYPED_TEST(PotriTest, PotriWorkspaceAndDegenerateCalls) {
    using T = typename TestFixture::T;
    auto p = make_sys<T>(12, 1, 3, Uplo::Lower, 3u);
    UnifiedVector<int32_t> info;
    if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
    const size_t need = potri_buffer_size<TestFixture::BK, T>(this->q(), p.A(), p.uplo);
    EXPECT_GE(need, sizeof(T) * 12 * 12 * 3);
    UnifiedVector<std::byte> small(need / 2);
    EXPECT_THROW(([&] { (void)potri<TestFixture::BK, T>(this->q(), p.A(), p.uplo, false, Span<std::byte>(small.data(), small.size()), Span<int32_t>{}); }()), batchlas::workspace_error);
    UnifiedVector<std::byte> ws(need);
    MV<T> empty_batch(p.a.data(), 12, 12, p.lda, p.stra, 0, nullptr);
    EXPECT_NO_THROW(([&] { (void)potri<TestFixture::BK, T>(this->q(), empty_batch, p.uplo, false, Span<std::byte>(ws.data(), need), Span<int32_t>{}).wait(); }()));
    EXPECT_EQ((potri_buffer_size<TestFixture::BK, T>(this->q(), empty_batch, p.uplo)), 0u);
    MV<T> nonsquare(p.a.data(), 12, 11, p.lda, p.stra, 3, nullptr);
    EXPECT_THROW(([&] { (void)potri_buffer_size<TestFixture::BK, T>(this->q(), nonsquare, p.uplo); }()), batchlas::invalid_argument);
    MV<T> zero_n(p.a.data(), 0, 0, 1, 1, 3, nullptr);
    EXPECT_THROW(([&] { (void)potri_buffer_size<TestFixture::BK, T>(this->q(), zero_n, p.uplo); }()), batchlas::internal_error);
}

// The route straddle: the fused pin runs at n = 8 and is refused at n = 9, Auto serves both sides,
// and an unknown pin throws rather than falling back.
TYPED_TEST(PotriTest, PotriFusedPinStraddlesItsCeiling) {
    using T = typename TestFixture::T;
    using namespace ops::potri;
    for (int n : {8, 9}) {
        auto p = make_sys<T>(n, 1, 3, Uplo::Lower, 77u + n);
        UnifiedVector<int32_t> info;
        if (!this->factor(p, info)) continue;
        const size_t need = potri_buffer_size<TestFixture::BK, T>(this->q(), p.A(), p.uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        auto call = [&] { (void)potri<TestFixture::BK, T>(this->q(), p.A(), p.uplo, false, Span<std::byte>(ws.data(), need), Span<int32_t>{}).wait(); };
        {
            const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{Fused{}});
            if (n == 8 && this->fused_ok()) EXPECT_NO_THROW(call());
            else EXPECT_THROW(call(), std::invalid_argument);
        }
        {
            const select::ScopedPin<PotriChoice> pin("potri", std::string_view("lpanel:panel=8"));
            EXPECT_THROW(call(), std::invalid_argument);
        }
        if (n == 9) EXPECT_NO_THROW(call());  // Auto
    }
}

// A saturating batch: 1024 identical items, every result bit-identical to item 0 (a race between
// items or work-groups shows as a differing item; small batches cannot race). Each potri family
// pinned (Auto might never reach the fused SLM kernel here), and potrs on the same factor.
// The race it guards is in the fused SLM solve (getrs_fused.cc's potrs body), not the element-wise store.
// BREAK (run 2026-10-26): drop the forward-pass barrier between the block staging and the block solve.
// RED: CUDA only, every type: this test (potrs item 19 differs), PotrsSolves, PotriInverts, BatchBeyond,
// PotriFull (double, cfloat, cdouble); every NETLIB case GREEN. A wrong item index (BREAK run
// 2026-10-25: the store reads X at item 0) is GREEN here by construction (RED: PotriInverts, PotriFull).
TYPED_TEST(PotriTest, SaturatingBatchOfIdenticalItemsIsBitIdentical) {
    using T = typename TestFixture::T;
    using namespace ops::potri;
    for (int n : {8, 16})
        for (int route : {1, 2}) {  // 1 fused, 2 blocked
            if (route == 1 && (n > 8 || !this->fused_ok())) continue;
            auto p = make_sys<T>(n, 3, 1024, Uplo::Lower, 13u + n, /*identical=*/true);
            UnifiedVector<int32_t> info;
            if (!this->factor(p, info)) continue;
            auto s = make_sys<T>(n, 3, 1024, Uplo::Lower, 13u + n, true);
            s.a = p.a;
            (void)potrs<TestFixture::BK, T>(this->q(), s.A(), s.B(), Uplo::Lower).wait();
            {
                const select::ScopedPin<PotriChoice> pin("potri", route == 1 ? PotriChoice{Fused{}} : PotriChoice{Blocked{}});
                (void)this->run_potri(p, true);
            }
            this->q().wait();
            for (int it = 1; it < p.batch; ++it) {
                for (size_t o = 0; o < static_cast<size_t>(p.stra); ++o)
                    ASSERT_TRUE(same_bits(p.a[static_cast<size_t>(it) * p.stra + o], p.a[o]))
                        << "potri item " << it << " offset " << o << " route " << route;
                for (size_t o = 0; o < static_cast<size_t>(s.strb); ++o)
                    ASSERT_TRUE(same_bits(s.b[static_cast<size_t>(it) * s.strb + o], s.b[o])) << "potrs item " << it << " offset " << o;
            }
            auto one = make_sys<T>(n, 1, 1, Uplo::Lower, 13u + n, true);
            one.full.assign(1, p.full[0]);
            EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(one, {this->mirrored(p, 0)}))
                << "n=" << n << " route " << route;
        }
}

// A coverage guard for batch > 65535 (a CUDA grid y/z limit), at n = 2: every item must equal item 0,
// and item 0 must be the inverse. It is NOT a regression proof for the 1-D grid-strided store: the old
// 2-D range<2>(batch, n*n) also stayed GREEN on DPC++ CUDA (that runtime maps it onto the grid).
// BREAK (run 2026-10-25): the store skips items >= 65535. RED: this test only, every type and backend.
TYPED_TEST(PotriTest, BatchBeyondTheGridYLimitIsCovered) {
    using T = typename TestFixture::T;
    using namespace ops::potri;
    const int n = 2, batch = 70001;
    for (int route : {1, 2}) {  // 1 fused, 2 blocked
        if (route == 1 && !this->fused_ok()) continue;
        auto p = make_sys<T>(n, 1, batch, Uplo::Upper, 4242u, /*identical=*/true);
        UnifiedVector<int32_t> info;
        if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
        UnifiedVector<int32_t> pinfo(batch, int32_t(-7));
        {
            const select::ScopedPin<PotriChoice> pin("potri", route == 1 ? PotriChoice{Fused{}} : PotriChoice{Blocked{}});
            (void)this->run_potri(p, true, &pinfo);
        }
        for (int it = 0; it < batch; ++it) {
            ASSERT_EQ(pinfo[it], 0) << "item " << it;
            for (size_t o = 0; o < static_cast<size_t>(p.stra); ++o)
                ASSERT_TRUE(same_bits(p.a[static_cast<size_t>(it) * p.stra + o], p.a[o]))
                    << "item " << it << " offset " << o << " route " << route;
        }
        auto one = make_sys<T>(n, 1, 1, Uplo::Upper, 4242u, true);
        one.full.assign(1, p.full[0]);
        EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(one, {this->mirrored(p, 0)})) << "route " << route;
    }
}

// A non-empty info span shorter than the batch is refused, not silently ignored.
// BREAK (run 2026-10-25): drop the info-size check in potri.cc. RED: this test only.
TYPED_TEST(PotriTest, ShortInfoSpanIsRefused) {
    using T = typename TestFixture::T;
    auto p = make_sys<T>(4, 1, 3, Uplo::Lower, 8u);
    UnifiedVector<int32_t> info;
    if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
    UnifiedVector<int32_t> short_info(2, int32_t(-7));
    EXPECT_THROW((void)this->run_potri(p, false, &short_info), batchlas::invalid_argument);
}
