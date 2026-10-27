// potri (inverse from a Cholesky factor; potrs's own cases are in potrs_tests.cc): residuals through
// batchlas::verify, per-item info with the other items undisturbed, padded ld and stride with a
// poisoned unreferenced triangle, and a saturating batch. The op is native and vendor-free, so no
// oracle is a vendor library.
// evidence: docs/perf/potri.md#potrs-and-potri
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/potri.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_support/coverage_rows.hh>
#include <tests/test_support/inputs.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/potri/choice.hh>
#include <src/ops/potri/potri_native.hh>
#include <src/select/select.hh>

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace batchlas;

namespace {

namespace ts = test_support;
using ts::Live;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// A0 the whole Hermitian A packed (one item when the batch is identical); a padded and poisoned
// (ld n + 3, stride + 5) holding A0's uplo triangle, the factor after factor(), the inverse after potri.
template <typename T>
struct Sys {
    Uplo uplo;
    Matrix<T> A0, a;
};

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

    // cond > 1: a graded spectrum of that condition, else diagonally dominant; identical: one drawn item everywhere.
    Sys<T> sys(int n, int batch, Uplo uplo, unsigned seed, double cond = 1.0, bool identical = false) {
        Sys<T> p{uplo, ts::hpd<T>(q(), n, identical ? 1 : batch, cond, seed), ts::padded<T>(n, n, batch)};
        ts::store<T>(p.a, p.A0, Live::triangle(uplo));
        return p;
    }

    // The Cholesky factor in A's triangle, through the public potrf; returns its info.
    // False when the build has no potrf route for the shape (vendor-free: the native potrf does not cover it).
    bool factor(Sys<T>& p, UnifiedVector<int32_t>& info) {
        info = UnifiedVector<int32_t>(static_cast<size_t>(p.a.batch_size()), int32_t(-12345));
        try {
            const size_t need = potrf_buffer_size<BK, T>(q(), p.a, p.uplo);
            UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
            (void)potrf<BK, T>(q(), p.a, p.uplo, Span<std::byte>(ws.data(), need), info.to_span());
        } catch (const NoRouteError&) {
            return false;
        }
        q().wait();
        return true;
    }

    Event run_potri(Sys<T>& p, bool full, UnifiedVector<int32_t>* info = nullptr) {
        const size_t need = potri_buffer_size<BK, T>(q(), p.a, p.uplo);
        ws_ = UnifiedVector<std::byte>(std::max<size_t>(need, 1));
        Span<int32_t> s = info ? info->to_span() : Span<int32_t>{};
        auto ev = potri<BK, T>(q(), p.a, p.uplo, full, Span<std::byte>(ws_.data(), need), s);
        ev.wait();
        q().wait();
        return ev;
    }

    // max over `items` (empty: the default sample) of ||A0 X - I|| / (||A0|| ||X||).
    double inverse_residual(const Sys<T>& p, const MV<T>& X, const std::vector<int>& items = {}) {
        return MV<T>::solve_residual(q(), p.A0, X, MV<T>{}, items);
    }
    // The stored triangle of the inverse, mirrored.
    double triangle_residual(const Sys<T>& p) { return inverse_residual(p, ts::hermitian<T>(p.a, p.uplo)); }

    UnifiedVector<std::byte> ws_;
};

using PotriTestTypes = typename test_utils::backend_types<ts::Config>::type;

}  // namespace

TYPED_TEST_SUITE(PotriTest, PotriTestTypes);

// Families: cta (sycl_potri::potri_cta, trtri + lauum in one local-memory launch) and blocked:nb
// (LAPACK's blocked trtri and lauum: the CTA kernel on nb x nb diagonal blocks, public trmm, gemm,
// herk/syrk). Every test pins each family; Auto is exercised beside them. The deliberate breaks and
// their red sets are in docs/perf/potri.md#potri-what-the-tests-pin.

namespace {

using ops::potri::PotriChoice;

// Every compiled candidate: cta, blocked:nb=64 and, for float only, blocked:nb=128.
template <typename T>
std::vector<PotriChoice> potri_pins() {
    const auto c = ops::potri::candidates<T>();
    return {c.begin(), c.end()};
}

std::string pin_name(const PotriChoice& c) { return select::to_string(c); }

// The order the cta pin can run on this queue (the can_run term, read from the same device facts).
template <Backend B, typename T>
int cta_ceiling(Queue& q) {
    const auto& d = select::device_of<B>(q);
    return sycl_potri::cta_max_n<T>(static_cast<std::size_t>(d.slm_budget), d.max_wg);
}

}  // namespace

// A * A^-1 = I through verify, both triangles, Auto and each pin, an order ladder through the tile
// width (4), the block orders (64, 128) and their ragged neighbours, three and more diagonal blocks
// (lauum's gemm leg needs n > 2 nb), padded ld and stride, the
// unreferenced triangle poisoned and written by nobody, info all zero.
TYPED_TEST(PotriTest, PotriInvertsAndWritesOnlyItsTriangle) {
    using T = typename TestFixture::T;
    const int ceiling = cta_ceiling<TestFixture::BK, T>(this->q());
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {1, 2, 3, 4, 5, 8, 9, 16, 31, 32, 33, 64, 65, 100, 129, 200, 300})
            for (int route = -1; route < static_cast<int>(potri_pins<T>().size()); ++route) {
                if (route == 0 && n > ceiling) continue;
                auto p = this->sys(n, 4, uplo, 900u + n);
                UnifiedVector<int32_t> info;
                if (!this->factor(p, info)) continue;
                const Matrix<T> before = p.a;
                UnifiedVector<int32_t> pinfo(4, int32_t(-7));
                {
                    std::optional<select::ScopedPin<PotriChoice>> pin;
                    if (route >= 0) pin.emplace("potri", potri_pins<T>()[route]);
                    (void)this->run_potri(p, false, &pinfo);
                }
                const std::string what = "n=" + std::to_string(n) + " route=" + (route < 0 ? "auto" : pin_name(potri_pins<T>()[route])) +
                                         " uplo=" + (uplo == Uplo::Lower ? "L" : "U");
                for (int it = 0; it < 4; ++it) ASSERT_EQ(pinfo[it], 0) << what;
                EXPECT_VERIFY(T, verify::Check::solve, n, this->triangle_residual(p)) << what;
                EXPECT_TRUE(ts::unwritten_outside<T>(p.a, Live::triangle(uplo), before)) << what;
            }
}

// Graded spectra (condition 1e3 for float and cfloat, 1e6 for double and cdouble) on every family,
// both triangles: the diagonally dominant inputs above cannot see an unstable update order.
TYPED_TEST(PotriTest, PotriInvertsGradedIllConditionedInputs) {
    using T = typename TestFixture::T;
    const int ceiling = cta_ceiling<TestFixture::BK, T>(this->q());
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {7, 33, 100, 200, 300})
            for (const PotriChoice& c : potri_pins<T>()) {
                if (c.index() == 0 && n > ceiling) continue;
                auto p = this->sys(n, 3, uplo, 61u + n, ts::cond_for<T>());
                UnifiedVector<int32_t> info;
                if (!this->factor(p, info)) continue;
                for (int it = 0; it < 3; ++it) ASSERT_EQ(info[it], 0) << "potrf refused a graded input";
                {
                    const select::ScopedPin<PotriChoice> pin("potri", c);
                    (void)this->run_potri(p, false);
                }
                EXPECT_VERIFY(T, verify::Check::solve, n, this->triangle_residual(p))
                    << "n=" << n << " " << pin_name(c) << " uplo=" << (uplo == Uplo::Lower ? "L" : "U");
            }
}

// With full = true the other triangle is exactly the conjugate transpose (bit for bit: a complex
// inverse with nonzero imaginary parts is Hermitian only if the mirror conjugates), the diagonal is
// real, and the whole matrix is the inverse with no mirroring on the caller's side. Every family.
TYPED_TEST(PotriTest, PotriFullFillsTheOtherTriangleWithTheConjugateTranspose) {
    using T = typename TestFixture::T;
    for (const PotriChoice& c : potri_pins<T>())
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {3, 8, 17, 40, 70, 150}) {
                auto p = this->sys(n, 3, uplo, 31u + n);
                UnifiedVector<int32_t> info;
                if (!this->factor(p, info)) continue;
                if (c.index() == 0 && n > cta_ceiling<TestFixture::BK, T>(this->q())) continue;
                const Matrix<T> before = p.a;
                {
                    const select::ScopedPin<PotriChoice> pin("potri", c);
                    (void)this->run_potri(p, true);
                }
                for (int it = 0; it < 3; ++it)
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            const T a = p.a(i, j, it), b = p.a(j, i, it);
                            // Diagonals: a real value (a bitwise +0 against -0 is not a defect).
                            if (i == j) ASSERT_EQ(std::imag(a), 0) << "(" << i << "," << j << ") " << pin_name(c);
                            else ASSERT_TRUE(ts::same_bits(a, MV<T>::conj(b))) << "(" << i << "," << j << ") " << pin_name(c);
                        }
                EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, p.a)) << "n=" << n << " " << pin_name(c);
                EXPECT_TRUE(ts::unwritten_outside<T>(p.a, Live::block(), before)) << pin_name(c);
            }
}

// info follows ?potri: the first exactly-zero factor diagonal, per item; that item comes back
// unchanged (both triangles and its pads, as ?trtri returns before writing), with info and without
// it, `full` or not, both triangles; the other items are bit-identical to a clean batch's, on every
// family. The zero sits in each of blocked's diagonal blocks at n = 150.
// BREAK (run 2026-10-26): blocked's last Lauum launch skips the restore; separately, potri_finish ignores
// the flags. RED: this test only, every type CUDA+NETLIB (the second in the `full` cases alone).
TYPED_TEST(PotriTest, PotriInfoNamesTheZeroDiagonalAndLeavesOtherItemsAlone) {
    using T = typename TestFixture::T;
    for (const PotriChoice& c : potri_pins<T>())
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (bool full : {false, true})
                for (int n : {5, 8, 20, 70, 150})
                    for (int zero_at : {0, n / 2, n - 1}) {
                        if (c.index() == 0 && n > cta_ceiling<TestFixture::BK, T>(this->q())) continue;
                        const int bad = 1, batch = 4;
                        const std::string what = pin_name(c) + " n=" + std::to_string(n) + " zero_at=" + std::to_string(zero_at) +
                                                 (uplo == Uplo::Lower ? " L" : " U") + (full ? " full" : "");
                        auto clean = this->sys(n, batch, uplo, 5u + n);
                        UnifiedVector<int32_t> info;
                        if (!this->factor(clean, info)) continue;
                        Sys<T> dirty = clean;
                        dirty.a(zero_at, zero_at, bad) = MV<T>::make(0.0, 0.0);
                        Sys<T> quiet = dirty;
                        const Matrix<T> before = dirty.a;
                        UnifiedVector<int32_t> ic(batch, -1), id(batch, -1);
                        {
                            const select::ScopedPin<PotriChoice> pin("potri", c);
                            (void)this->run_potri(clean, full, &ic);
                            (void)this->run_potri(dirty, full, &id);
                            (void)this->run_potri(quiet, full);
                        }
                        for (int it = 0; it < batch; ++it) {
                            EXPECT_EQ(ic[it], 0) << what;
                            EXPECT_EQ(id[it], it == bad ? zero_at + 1 : 0) << "item " << it << " " << what;
                        }
                        for (const Sys<T>* s : {&dirty, &quiet}) {
                            const char* how = s == &quiet ? " without info" : "";
                            ASSERT_TRUE(ts::items_bit_identical_to<T>(s->a, before, std::vector<int>{bad})) << "the failed item, " << what << how;
                            ASSERT_TRUE(ts::items_bit_identical_to<T>(s->a, clean.a, std::vector<int>{0, 2, 3})) << what << how;
                        }
                    }
}

// The workspace is the chosen family's: none for cta, none for blocked at n <= nb, and for blocked
// above it one part's n x nb panel and packed triangle (a failed item's copy) per item, below
// n x (nb + n); it is enforced, and the empty-batch, empty-problem and non-square cases are refused
// or no-ops, not wrong answers.
TYPED_TEST(PotriTest, PotriWorkspaceAndDegenerateCalls) {
    using T = typename TestFixture::T;
    constexpr Backend BK = TestFixture::BK;
    auto p = this->sys(100, 3, Uplo::Lower, 3u);
    UnifiedVector<int32_t> info;
    if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
    {
        const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Cta{}});
        if (100 <= cta_ceiling<BK, T>(this->q())) EXPECT_EQ((potri_buffer_size<BK, T>(this->q(), p.a, p.uplo)), 0u);
    }
    const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Blocked{64}});
    const size_t need = potri_buffer_size<BK, T>(this->q(), p.a, p.uplo);
    EXPECT_GE(need, sizeof(T) * (100 * 64 + 100 * 101 / 2) * 3);
    EXPECT_LT(need, sizeof(T) * 100 * (64 + 100) * 3);
    MV<T> small_n(p.a.data_ptr(), 64, 64, p.a.ld(), p.a.stride(), 3, nullptr);
    EXPECT_EQ((potri_buffer_size<BK, T>(this->q(), small_n, p.uplo)), 0u);
    UnifiedVector<std::byte> small(need / 2);
    EXPECT_THROW(([&] { (void)potri<BK, T>(this->q(), p.a, p.uplo, false, Span<std::byte>(small.data(), small.size()), Span<int32_t>{}); }()), batchlas::workspace_error);
    UnifiedVector<std::byte> ws(need);
    MV<T> empty_batch(p.a.data_ptr(), 12, 12, p.a.ld(), p.a.stride(), 0, nullptr);
    EXPECT_NO_THROW(([&] { (void)potri<BK, T>(this->q(), empty_batch, p.uplo, false, Span<std::byte>(ws.data(), need), Span<int32_t>{}).wait(); }()));
    EXPECT_EQ((potri_buffer_size<BK, T>(this->q(), empty_batch, p.uplo)), 0u);
    MV<T> nonsquare(p.a.data_ptr(), 12, 11, p.a.ld(), p.a.stride(), 3, nullptr);
    EXPECT_THROW(([&] { (void)potri_buffer_size<BK, T>(this->q(), nonsquare, p.uplo); }()), batchlas::invalid_argument);
    MV<T> zero_n(p.a.data_ptr(), 0, 0, 1, 1, 3, nullptr);
    EXPECT_THROW(([&] { (void)potri_buffer_size<BK, T>(this->q(), zero_n, p.uplo); }()), batchlas::internal_error);
}

// blocked runs the batch in parts: with the chunk forced to 3 items, a 10-item batch is parts of
// 3, 3 and 4, every item is inverted (each verified, not a default sample), and the workspace is one part's.
TYPED_TEST(PotriTest, PotriBlockedPartsCoverEveryItem) {
    using T = typename TestFixture::T;
    constexpr Backend BK = TestFixture::BK;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        auto p = this->sys(70, 10, uplo, 404u);
        UnifiedVector<int32_t> info;
        if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
        const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Blocked{64}});
        const size_t whole = potri_buffer_size<BK, T>(this->q(), p.a, p.uplo);
        const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "3");
        const size_t part = potri_buffer_size<BK, T>(this->q(), p.a, p.uplo);
        EXPECT_LT(part, whole);
        (void)this->run_potri(p, true);
        EXPECT_VERIFY(T, verify::Check::solve, 70, this->inverse_residual(p, p.a, p.a.all_items()));
    }
}

// The cta pin at its advertised ceiling launches and inverts (a graded input: at n = 176 the
// diagonally dominant one hid a broken substitution); one past it is refused (a bad pin throws,
// never falls back) while Auto still serves it, and a foreign spelling throws.
TYPED_TEST(PotriTest, PotriCtaPinStraddlesItsCeiling) {
    using T = typename TestFixture::T;
    constexpr Backend BK = TestFixture::BK;
    const int ceiling = cta_ceiling<BK, T>(this->q());
    ASSERT_GE(ceiling, 32);
    for (int n : {ceiling, ceiling + 1}) {
        auto p = this->sys(n, 2, Uplo::Lower, 77u + n, 1e2);
        UnifiedVector<int32_t> info;
        if (!this->factor(p, info)) continue;
        const Matrix<T> factor = p.a;
        UnifiedVector<std::byte> ws(1);
        auto call = [&] { (void)potri<BK, T>(this->q(), p.a, p.uplo, false, Span<std::byte>(ws.data(), std::size_t{0}), Span<int32_t>{}).wait(); };
        {
            const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Cta{}});
            if (n == ceiling) {
                EXPECT_NO_THROW(call());
                EXPECT_VERIFY(T, verify::Check::solve, n, this->triangle_residual(p)) << "at the ceiling " << n;
            } else {
                EXPECT_THROW(call(), std::invalid_argument);
            }
        }
        {
            const select::ScopedPin<PotriChoice> pin("potri", std::string_view("lpanel:panel=8"));
            EXPECT_THROW(call(), std::invalid_argument);
        }
        if (n == ceiling + 1) {
            p.a = factor;
            const size_t need = potri_buffer_size<BK, T>(this->q(), p.a, p.uplo);
            ws = UnifiedVector<std::byte>(std::max<size_t>(need, 1));
            EXPECT_NO_THROW(([&] { (void)potri<BK, T>(this->q(), p.a, p.uplo, false, Span<std::byte>(ws.data(), need), Span<int32_t>{}).wait(); }()));
        }
    }
}

// A saturating batch: 1024 identical items, every result bit-identical to item 0 (a race between
// work-items or work-groups shows as a differing item; small batches cannot race). Each family
// pinned, at orders whose CTA work-groups span several sub-groups (n = 37: 64 work-items, n = 100:
// 352) and, for blocked, several diagonal blocks per item. The host queue (no race to find, 77 s
// at n = 100) stops at 37.
TYPED_TEST(PotriTest, SaturatingBatchOfIdenticalItemsIsBitIdentical) {
    using T = typename TestFixture::T;
    const int ceiling = cta_ceiling<TestFixture::BK, T>(this->q());
    for (int n : {8, 37, 100})
        for (const PotriChoice& c : potri_pins<T>()) {
            if ((c.index() == 0 && n > ceiling) || (TestFixture::BK == Backend::NETLIB && n > 37)) continue;
            auto p = this->sys(n, 1024, Uplo::Lower, 13u + n, 1.0, /*identical=*/true);
            UnifiedVector<int32_t> info;
            if (!this->factor(p, info)) continue;
            {
                const select::ScopedPin<PotriChoice> pin("potri", c);
                (void)this->run_potri(p, true);
            }
            ASSERT_TRUE(ts::items_bit_identical<T>(p.a)) << "potri " << pin_name(c) << " n=" << n;
            EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, p.a, {0})) << "n=" << n << " " << pin_name(c);
        }
}

// A coverage guard for batch > 65535 (a CUDA grid y/z limit), at n = 2: every item must equal item 0,
// and item 0 must be the inverse. Both families launch one work-group per item (blocked at n <= nb is
// the CTA kernel twice), so the item index past 65535 is exercised on each.
TYPED_TEST(PotriTest, BatchBeyondTheGridYLimitIsCovered) {
    using T = typename TestFixture::T;
    const int n = 2, batch = 70001;
    for (const PotriChoice& c : {PotriChoice{ops::potri::Cta{}}, PotriChoice{ops::potri::Blocked{64}}}) {
        auto p = this->sys(n, batch, Uplo::Upper, 4242u, 1.0, /*identical=*/true);
        UnifiedVector<int32_t> info;
        if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
        UnifiedVector<int32_t> pinfo(batch, int32_t(-7));
        {
            const select::ScopedPin<PotriChoice> pin("potri", c);
            (void)this->run_potri(p, true, &pinfo);
        }
        for (int it = 0; it < batch; ++it) ASSERT_EQ(pinfo[it], 0) << "item " << it;
        ASSERT_TRUE(ts::items_bit_identical<T>(p.a)) << pin_name(c);
        EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, p.a, {0})) << pin_name(c);
    }
}

// A non-empty info span shorter than the batch is refused, not silently ignored.
// BREAK (run 2026-10-25): drop the info-size check in potri.cc. RED: this test only.
TYPED_TEST(PotriTest, ShortInfoSpanIsRefused) {
    auto p = this->sys(4, 3, Uplo::Lower, 8u);
    UnifiedVector<int32_t> info;
    if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
    UnifiedVector<int32_t> short_info(2, int32_t(-7));
    EXPECT_THROW((void)this->run_potri(p, false, &short_info), batchlas::invalid_argument);
}

// Linked is not reached: the coverage rows of a child process show each pinned family ran, and that
// blocked above nb reached its children (trmm, gemm, and herk or syrk) while cta reached none of them.
TYPED_TEST(PotriTest, PotriCoverageShowsEachFamilyAndTheBlockedChildren) {
    using T = typename TestFixture::T;
    // Factored before coverage is switched on, so potrf's own children leave no rows.
    auto lo = this->sys(16, 3, Uplo::Lower, 5u), hi = this->sys(200, 3, Uplo::Lower, 6u);
    UnifiedVector<int32_t> info;
    if (!this->factor(lo, info) || !this->factor(hi, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
    const auto rows = ts::reached_rows([&] {
        const ScopedEnvVar clear("BATCHLAS_POTRI_ROUTE", nullptr);
        {
            const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Cta{}});
            (void)this->run_potri(lo, false);
        }
        const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Blocked{64}});
        (void)this->run_potri(hi, false);
    });
    std::map<std::string, std::set<std::string>> algos;  // op -> chosen spellings
    std::set<std::string> blocked_children;
    for (const auto& r : rows) {
        algos[r.op].insert(r.spelling);
        if (r.op != "potri") blocked_children.insert(r.op);
    }
    EXPECT_EQ(algos["potri"], (std::set<std::string>{"cta", "blocked:nb=64"}));
    // cta at n = 16 calls no child; blocked at n = 200 (four blocks) calls these, and nothing else.
    EXPECT_EQ(blocked_children, (std::set<std::string>{"trmm", "gemm", is_std_complex_v<T> ? "herk" : "syrk"}));
}
