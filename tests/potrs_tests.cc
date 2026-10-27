// potrs (solve with a Cholesky factor), a routed op with two families: `fused` (both substitutions in
// one launch, B in SLM) and `trsm` (two public trsm calls). Residuals through batchlas::verify on graded
// SPD inputs, each family pinned and checked reached (coverage rows), the fused ceilings straddled and
// launched at, Auto against synthetic and shipped tables, bit-for-bit agreement with posv on the same
// solve, padded ld and stride with a poisoned unreferenced triangle, and a saturating batch.
// evidence: docs/perf/potrs.md#potrs-selection
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/potrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/shipped_table_pick.hh>
#include <tests/test_support/coverage_rows.hh>
#include <tests/test_support/inputs.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/getrs/getrs_native.hh>
#include <src/ops/posv/choice.hh>
#include <src/ops/potrs/choice.hh>
#include <src/ops/potrs/potrs.hh>
#include <src/select/select.hh>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using namespace batchlas;

namespace {

namespace ps = ops::potrs;
namespace ts = test_support;
using C = ps::PotrsChoice;
using Pin = select::ScopedPin<C>;
using ts::Live;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

const char* uplo_s(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }

// A0 (the whole Hermitian A) and B0 packed, one item when the batch is identical; a and b in padded,
// poisoned storage (A ld n + 3, stride + 5; B ld n + 2, stride + 7), a holding A0's uplo triangle.
template <typename T>
struct Sys {
    Uplo uplo;
    Matrix<T> A0, B0, a, b;
};

// The outermost potrs trace line for whatever `run` calls, and the choice it names.
template <typename F>
std::string traced_choice(F&& run) {
    const ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    try {
        run();
    } catch (...) {
        (void)::testing::internal::GetCapturedStderr();
        throw;
    }
    const std::string err = ::testing::internal::GetCapturedStderr();
    std::istringstream in(err);
    for (std::string line; std::getline(in, line);)
        if (line.rfind("potrs ", 0) == 0 && line.find(" -> ") != std::string::npos) {
            const std::string tail = line.substr(line.find(" -> ") + 4);
            return tail.substr(0, tail.find(' '));
        }
    return "<no potrs trace line in: " + err + ">";
}

struct TableGuard {
    ~TableGuard() { select::testing::use_embedded_tables(); }
};

template <typename Config>
class PotrsTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend BK = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        // NETLIB serves potrf and trsm through LAPACKE/CBLAS only, so a vendor-free build has nothing to test there.
        if (BK == Backend::NETLIB && !(BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS))
            GTEST_SKIP() << "NETLIB without LAPACKE/CBLAS has no potrf or trsm route";
    }

    Queue& q() { return *this->ctx; }
    const select::Device& dev() { return select::device_of<BK>(q()); }
    bool fits(int n, int nrhs) { return ps::fused_fits<T>(dev(), n, nrhs); }
    // The largest n the fused solve takes at nrhs = 8, from the device's own SLM budget.
    int fused_ceiling_n() {
        return static_cast<int>(sycl_getrs::getrs_fused_max_rhs_elems<T>(static_cast<size_t>(dev().slm_budget)) / 8);
    }

    // cond > 1: a graded spectrum of that condition; identical: every item is one drawn item.
    Sys<T> sys(int n, int nrhs, int batch, Uplo uplo, unsigned seed, double cond = 1.0, bool identical = false) {
        const int drawn = identical ? 1 : batch;
        Sys<T> p{uplo, ts::hpd<T>(q(), n, drawn, cond, seed), Matrix<T>::Random(n, nrhs, false, drawn, seed + 1),
                 ts::padded<T>(n, n, batch), ts::padded<T>(n, nrhs, batch, 2, 7)};
        ts::store<T>(p.a, p.A0, Live::triangle(uplo));
        ts::store<T>(p.b, p.B0);
        return p;
    }

    // The Cholesky factor in A's triangle, through the public potrf. False when the build has no potrf route.
    bool factor(Sys<T>& p) {
        UnifiedVector<int32_t> info(static_cast<size_t>(p.a.batch_size()), int32_t(-12345));
        try {
            const size_t need = potrf_buffer_size<BK, T>(q(), p.a, p.uplo);
            UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
            (void)potrf<BK, T>(q(), p.a, p.uplo, Span<std::byte>(ws.data(), need), info.to_span());
        } catch (const NoRouteError&) {
            return false;
        }
        q().wait();
        for (int it = 0; it < p.a.batch_size(); ++it) EXPECT_EQ(info[it], 0) << "potrf failed on item " << it;
        return true;
    }
    void solve(Sys<T>& p) {
        (void)potrs<BK, T>(q(), p.a, p.b, p.uplo).wait();
        q().wait();
    }
    // ||A0 X - B0|| over `items` (empty: the default sample), X read in place from the padded b.
    double solve_res(const Sys<T>& p, const std::vector<int>& items = {}) {
        return MV<T>::solve_residual(q(), p.A0, p.b, p.B0, items);
    }
};

using PotrsTestTypes = typename test_utils::backend_types<ts::Config>::type;

}  // namespace
TYPED_TEST_SUITE(PotrsTest, PotrsTestTypes);

// Each family pinned, and Auto, on graded SPD factors (condition 1e6 double, 1e3 single), both triangles,
// an order ladder through the fused block edge (nb = 16: 16/17, 33 = 2*16 + 1, 130) and the accumulator
// buckets (nrhs 1, 2, 3, 4, 5, 8; 9 and 20 past the fused width). The factor is read-only, B's pads survive.
// BREAK (run 2026-10-26): getrs_fused.cc's potrs body stages the Upper forward block from ld_h(j+s, j+t), the
// poisoned triangle. RED: the fused Upper cases of this test, ASecondRightHandSide, SaturatingBatch and
// AutoReadsTheShippedTable (Auto takes fused at U n=12 nrhs=8; re-run 2026-10-27, this suite and the one before
// test_support alike), every CUDA type; trsm, Lower and the posv comparison (same kernel) GREEN.
// BREAK (run 2026-10-26): the packed block solve gives every slot the group's first column (c = g * P). RED:
// exactly the fused cases with nrhs >= 3 (P >= 2) here, in ASecondRightHandSide, PinnedCandidatesStraddle and
// SaturatingBatch, every CUDA type; nrhs 1 and 2 (one slot) GREEN.
TYPED_TEST(PotrsTest, EachFamilySolvesGradedSystemsOnBothTriangles) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
    for (const char* family : {"fused", "trsm", "auto"})
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {1, 3, 8, 16, 17, 33, 130})
                for (int nrhs : {1, 2, 3, 4, 5, 8, 9, 20}) {
                    const bool fused = std::string_view(family) == "fused";
                    if (fused && !this->fits(n, nrhs)) continue;
                    auto p = this->sys(n, nrhs, 3, uplo, 100u + n * 31 + nrhs, ts::cond_for<T>());
                    if (!this->factor(p)) continue;
                    const Matrix<T> factor_bits = p.a;
                    {
                        const Pin pin("potrs", std::string_view(family));
                        this->solve(p);
                    }
                    const std::string what = std::string(family) + " " + uplo_s(uplo) + " n=" + std::to_string(n) +
                                             " nrhs=" + std::to_string(nrhs);
                    EXPECT_VERIFY(T, verify::Check::solve, n, this->solve_res(p)) << what;
                    ASSERT_TRUE(ts::unwritten_outside<T>(p.a, Live::nothing(), factor_bits)) << what << ": A";
                    ASSERT_TRUE(ts::unwritten_outside<T>(p.b, Live::block())) << what << ": B";
                }
}

// The factor is reusable: a second right-hand-side set on the same factor, each family.
TYPED_TEST(PotrsTest, ASecondRightHandSideReusesTheFactor) {
    using T = typename TestFixture::T;
    for (const char* family : {"fused", "trsm"})
        for (int n : {8, 40}) {
            if (std::string_view(family) == "fused" && !this->fits(n, 4)) continue;
            auto p = this->sys(n, 4, 5, Uplo::Upper, 7u + n);
            if (!this->factor(p)) continue;
            const Pin pin("potrs", std::string_view(family));
            this->solve(p);
            auto p2 = this->sys(n, 4, 5, Uplo::Upper, 7u + n);
            p2.a = p.a;
            for (int it = 0; it < 5; ++it)
                for (int k = 0; k < 4; ++k)
                    for (int i = 0; i < n; ++i) p2.B0(i, k, it) = MV<T>::make(double(i + 1), double(-k));
            ts::store<T>(p2.b, p2.B0);
            this->solve(p2);
            EXPECT_VERIFY(T, verify::Check::solve, n, this->solve_res(p2)) << family << " second right-hand side n=" << n;
        }
}

// potrs's `fused` is posv's `cta` solve and `trsm` is posv's `blocked` solve, bit for bit.
// BREAK (run 2026-10-26): potrs.cc's launch passes fused = false whatever the choice. RED: this test only,
// every CUDA type (the fused cases); every other potrs test GREEN, NETLIB GREEN.
TYPED_TEST(PotrsTest, FusedMatchesPosvCtaAndTrsmMatchesPosvBlockedBitForBit) {
    using T = typename TestFixture::T;
    namespace pv = ops::posv;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (auto [n, nrhs, fused] : {std::tuple{8, 4, true}, std::tuple{9, 2, true}, std::tuple{40, 8, true},
                                      std::tuple{8, 4, false}, std::tuple{20, 16, false}}) {
            if (fused && !this->fits(n, nrhs)) continue;
            auto p = this->sys(n, nrhs, 7, uplo, 7u + n);
            auto q2 = this->sys(n, nrhs, 7, uplo, 7u + n);
            {
                const size_t need = posv_buffer_size<TestFixture::BK, T>(this->q(), p.a, p.b, uplo);
                UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
                const select::ScopedPin<pv::PosvChoice> pin("posv", fused ? pv::PosvChoice{pv::Cta{}} : pv::PosvChoice{pv::Blocked{}});
                UnifiedVector<int32_t> info(7, int32_t(-1));
                (void)posv<TestFixture::BK, T>(this->q(), p.a, p.b, uplo, Span<std::byte>(ws.data(), need), info.to_span()).wait();
            }
            if (!this->factor(q2)) continue;
            {
                const Pin pin("potrs", fused ? C{ps::Fused{}} : C{ps::Trsm{}});
                this->solve(q2);
            }
            const std::string what = "n=" + std::to_string(n) + " nrhs=" + std::to_string(nrhs) + " fused=" + std::to_string(fused);
            ASSERT_TRUE(ts::items_bit_identical_to<T>(p.a, q2.a)) << "factor " << what;
            ASSERT_TRUE(ts::items_bit_identical_to<T>(p.b, q2.b)) << "X " << what;
        }
}

// posv composes potrf and the potrs solve, so a non-positive-definite item is flagged in `info` at the
// minor potrf names, and every other item's X is bit-identical to a batch without the bad item. Both
// batches take the same solve, so a wrong solve stays GREEN here (EachFamilySolves catches it); this
// guards info and cross-item isolation. ARMED BREAK: potrf's info names the minor 0-based (success
// stays 0). EXPECTED: RED in this file only here, the bad item's info assertion, every type and both families.
TYPED_TEST(PotrsTest, NonSpdItemIsFlaggedAndTheOthersAreUndisturbed) {
    using T = typename TestFixture::T;
    using namespace ops::posv;
    for (bool blocked : {false, true})
        for (int n : {8, 20}) {
            const int nrhs = blocked ? 16 : 4, bad = 2, batch = 5;
            if (!blocked && !this->fits(n, nrhs)) continue;
            auto clean = this->sys(n, nrhs, batch, Uplo::Lower, 41u + n);
            auto dirty = this->sys(n, nrhs, batch, Uplo::Lower, 41u + n);
            dirty.a(n / 2, n / 2, bad) = MV<T>::make(-1.0, 0.0);  // item 2 loses positive-definiteness at minor n/2 + 1
            UnifiedVector<int32_t> ic(batch, -1), id(batch, -1);
            for (Sys<T>* s : {&clean, &dirty}) {
                const size_t need = posv_buffer_size<TestFixture::BK, T>(this->q(), s->a, s->b, Uplo::Lower);
                UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
                const select::ScopedPin<PosvChoice> pin("posv", blocked ? PosvChoice{Blocked{}} : PosvChoice{Cta{}});
                (void)posv<TestFixture::BK, T>(this->q(), s->a, s->b, Uplo::Lower, Span<std::byte>(ws.data(), need),
                                               (s == &clean ? ic : id).to_span()).wait();
            }
            this->q().wait();
            for (int it = 0; it < batch; ++it) {
                EXPECT_EQ(ic[it], 0);
                EXPECT_EQ(id[it], it == bad ? n / 2 + 1 : 0) << "item " << it;
            }
            ASSERT_TRUE(ts::items_bit_identical_to<T>(dirty.b, clean.b, std::vector<int>{0, 1, 3, 4})) << "disturbed by item " << bad;
        }
}

// The fused pin on both sides of each ceiling: the accumulator width (nrhs 8 runs, 9 is refused) and the
// SLM capacity, LAUNCHED at the advertised ceiling (nrhs = 8, n = cap / 8 runs and solves; n + 1 is
// refused). trsm and Auto run on both sides; an unknown spelling throws. On a CPU queue fused never runs.
// BREAK (run 2026-10-26): fused_fits drops `nrhs <= kGetrsFusedMaxRhs`. RED: this test (the nrhs = 9 pin is
// refused by the kernel's entry check, not by selection) and EachFamilySolves (its fused pin at nrhs 9 and 20
// reaches the kernel and throws), every CUDA type; every other potrs test GREEN, NETLIB GREEN.
TYPED_TEST(PotrsTest, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
    const bool gpu = this->fits(1, 1);
    const int ceil_n = gpu ? this->fused_ceiling_n() : 16;
    if (gpu) ASSERT_GT(ceil_n, 64) << "the fused SLM capacity reads implausibly small";
    struct Cell { int n, nrhs; bool fused_runs; };
    const Cell cells[] = {{24, 8, gpu}, {24, 9, false}, {ceil_n, 8, gpu}, {ceil_n + 1, 8, false}};
    for (const Cell& k : cells) {
        const int batch = k.n > 256 ? 2 : 3;
        for (const char* family : {"fused", "trsm", "auto", "no_such_family"}) {
            auto p = this->sys(k.n, k.nrhs, batch, Uplo::Lower, 5u + k.n);
            if (!this->factor(p)) break;
            const std::string what = std::string(family) + " n=" + std::to_string(k.n) + " nrhs=" + std::to_string(k.nrhs);
            const std::string_view f(family);
            const Pin pin("potrs", f);
            if (f == "no_such_family" || (f == "fused" && !k.fused_runs)) {
                // Refused by selection itself (can_run), not by the kernel's own entry check.
                std::string err;
                try {
                    this->solve(p);
                } catch (const std::invalid_argument& e) {
                    err = e.what();
                }
                EXPECT_NE(err.find(f == "fused" ? "cannot run this shape" : "no_such_family"), std::string::npos) << what << ": " << err;
                continue;
            }
            EXPECT_NO_THROW(this->solve(p)) << what;
            EXPECT_VERIFY(T, verify::Check::solve, k.n, this->solve_res(p)) << what;
        }
    }
}

// Every pinned family records a `reached` coverage row under its own spelling, with potrs's key
// (m = k = order, n = nrhs) and uplo.
TYPED_TEST(PotrsTest, CoverageRowNamesThePinnedFamily) {
    if (!this->fits(16, 4)) GTEST_SKIP() << "the fused family cannot run on this queue";
    const auto rows = ts::reached_rows([&] {
        const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
        for (auto [n, nrhs, uplo, c] : {std::tuple{16, 4, Uplo::Lower, C{ps::Fused{}}}, std::tuple{24, 3, Uplo::Upper, C{ps::Trsm{}}}}) {
            auto p = this->sys(n, nrhs, 2, uplo, 3u);
            if (!this->factor(p)) std::exit(3);
            const Pin pin("potrs", c);
            this->solve(p);
        }
    });
    std::map<std::tuple<long long, long long, long long>, ts::ReachedRow> by_key;  // (m, n, k)
    for (const auto& r : rows)
        if (r.op == "potrs") by_key[{r.m, r.n, r.k}] = r;
    ASSERT_EQ(by_key.size(), 2u);
    for (const auto& [key, algo, uplo] : {std::tuple{std::tuple{16LL, 4LL, 16LL}, "fused", Uplo::Lower},
                                          std::tuple{std::tuple{24LL, 3LL, 24LL}, "trsm", Uplo::Upper}}) {
        ASSERT_TRUE(by_key.count(key)) << "no row m=" << std::get<0>(key) << " n=" << std::get<1>(key);
        EXPECT_EQ(by_key[key].spelling, algo) << std::get<0>(key);
        EXPECT_EQ(by_key[key].uplo, uplo) << std::get<0>(key);
    }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with uplo, n, nrhs and
// batch alone. BREAK (run 2026-10-26): key_of spells nrhs as 1. RED: exactly the nrhs probe of this test
// and AutoReadsTheShippedTable (its nrhs-straddling cells), every CUDA type; every other potrs test GREEN.
TYPED_TEST(PotrsTest, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!this->fits(64, 8)) GTEST_SKIP() << "the fused family cannot run on this queue";
    const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
    const TableGuard restore;
    const std::string dtype(select::dtype_name<T>());
    const std::string devk = this->dev().key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("potrs.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("potrs." + dtype + "." + devk + ".txt",
                       "# op=potrs dtype=" + dtype + " device=" + devk + " kernels=unknown\n"
                       "# keys: uplo:exact n:log:2 nrhs:log batch:log\n"
                       "uplo=L n=8 nrhs=2 batch=128 | fused 1 | trsm 2\n"
                       "uplo=L n=8 nrhs=2 batch=32768 | trsm 1 | fused 2\n"
                       "uplo=L n=8 nrhs=8 batch=128 | trsm 1 | fused 2\n"
                       "uplo=L n=64 nrhs=2 batch=128 | trsm 1 | fused 2\n"
                       "uplo=U n=8 nrhs=2 batch=128 | trsm 1 | fused 2\n");
    select::testing::set_builtin_tables(test_utils::dtype_keyed(std::move(files)));
    struct Probe { Uplo uplo; int n, nrhs, batch; const char* expect; const char* field; };
    const Probe probes[] = {{Uplo::Lower, 8, 2, 128, "fused", "base"},   {Uplo::Lower, 8, 2, 32768, "trsm", "batch"},
                            {Uplo::Lower, 8, 8, 128, "trsm", "nrhs"},    {Uplo::Lower, 64, 2, 128, "trsm", "n"},
                            {Uplo::Upper, 8, 2, 128, "trsm", "uplo"}};
    for (const auto& k : probes) {
        auto p = this->sys(k.n, k.nrhs, k.batch, k.uplo, 43u);
        if (!this->factor(p)) continue;
        EXPECT_EQ(traced_choice([&] { this->solve(p); }), k.expect) << "the " << k.field << " row";
    }
}

// Auto against the shipped table of this device on both sides of the table's crossovers (nrhs 1/2 and
// 4/5 at n 16 and 64, the fused width 8/9, n 12/16, an off-grid batch), so a retune moves the expectation.
TYPED_TEST(PotrsTest, AutoReadsTheShippedTable) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
    struct Cell { Uplo uplo; int n, nrhs, batch; };
    const Cell cells[] = {{Uplo::Lower, 16, 1, 128},  {Uplo::Lower, 16, 2, 128},  {Uplo::Lower, 16, 8, 2048},
                          {Uplo::Lower, 16, 9, 2048}, {Uplo::Lower, 64, 1, 512},  {Uplo::Lower, 64, 5, 512},
                          {Uplo::Upper, 64, 4, 512},  {Uplo::Upper, 12, 8, 300},  {Uplo::Lower, 256, 2, 128},
                          {Uplo::Upper, 256, 8, 128}};
    for (Cell k : cells) {
        if (TestFixture::BK == Backend::NETLIB) k.batch = std::min(k.batch, 16);  // no CPU table: the batch keys nothing
        const std::string label = std::string(uplo_s(k.uplo)) + " n=" + std::to_string(k.n) + " nrhs=" + std::to_string(k.nrhs) +
                                  " batch=" + std::to_string(k.batch);
        auto p = this->sys(k.n, k.nrhs, k.batch, k.uplo, 41u + k.n);
        if (!this->factor(p)) continue;
        const std::string want = test_utils::shipped_table_pick<C>(
            "potrs", select::dtype_name<T>(), this->dev(),
            select::Key{{"uplo", uplo_s(k.uplo)}, {"n", k.n}, {"nrhs", k.nrhs}, {"batch", k.batch}},
            [&] {
                auto s = this->sys(k.n, k.nrhs, 1, k.uplo, 3u);
                (void)this->factor(s);
                this->solve(s);
            });
        const std::string expect = want == test_utils::kNoTableEntryRuns ? "trsm" : want;
        EXPECT_EQ(traced_choice([&] { this->solve(p); }), expect) << label;
        EXPECT_VERIFY(T, verify::Check::solve, k.n, this->solve_res(p)) << "auto " << label;
    }
}

// A saturating batch: 1024 identical items, every result bit-identical to item 0 (a race between items or
// sub-groups shows as a differing item; small batches cannot race). Each family pinned, at one block
// (n = 16), several blocks (n = 40) and a 4-sub-group work-group (n = 130), both triangles.
// BREAK (run 2026-10-26): drop the forward-pass barrier between the block staging and the block solve in
// getrs_fused.cc's potrs body. RED: this test and EachFamilySolves (fused), every CUDA type, and
// PinnedCandidatesStraddle (double, cdouble); trsm and NETLIB GREEN. Re-run 2026-10-27: the same, plus (a race,
// so per run) AutoReadsTheShippedTable and FusedMatchesPosv for some CUDA types, with either suite.
TYPED_TEST(PotrsTest, SaturatingBatchOfIdenticalItemsIsBitIdentical) {
    using T = typename TestFixture::T;
    if (TestFixture::BK == Backend::NETLIB) GTEST_SKIP() << "no shared-local-memory kernel runs on the host queue";
    for (const char* family : {"fused", "trsm"})
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (auto [n, nrhs] : {std::tuple{16, 8}, std::tuple{40, 5}, std::tuple{130, 8}}) {
                if (std::string_view(family) == "fused" && !this->fits(n, nrhs)) continue;
                auto p = this->sys(n, nrhs, 1024, uplo, 13u + n, ts::cond_for<T>(), /*identical=*/true);
                if (!this->factor(p)) continue;
                {
                    const Pin pin("potrs", std::string_view(family));
                    this->solve(p);
                }
                ASSERT_TRUE(ts::items_bit_identical<T>(p.b)) << family << " " << uplo_s(uplo) << " n=" << n;
                EXPECT_VERIFY(T, verify::Check::solve, n, this->solve_res(p, {0})) << family << " n=" << n;
            }
}

// An empty batch is a no-op under any pin (it never reaches selection); an empty problem and a non-square
// A are refused before selection.
TYPED_TEST(PotrsTest, DegenerateCallsAreRefusedOrNoOps) {
    using T = typename TestFixture::T;
    auto p = this->sys(12, 2, 3, Uplo::Lower, 3u);
    MV<T> empty_a(p.a.data_ptr(), 12, 12, p.a.ld(), p.a.stride(), 0, nullptr);
    MV<T> empty_b(p.b.data_ptr(), 12, 2, p.b.ld(), p.b.stride(), 0, nullptr);
    {
        const Pin pin("potrs", std::string_view("no_such_family"));
        EXPECT_NO_THROW(([&] { (void)potrs<TestFixture::BK, T>(this->q(), empty_a, empty_b, Uplo::Lower).wait(); }()));
    }
    MV<T> zero_rhs(p.b.data_ptr(), 12, 0, p.b.ld(), p.b.stride(), 3, nullptr);
    EXPECT_THROW(([&] { (void)potrs<TestFixture::BK, T>(this->q(), p.a, zero_rhs, Uplo::Lower); }()), batchlas::internal_error);
    MV<T> nonsquare(p.a.data_ptr(), 12, 11, p.a.ld(), p.a.stride(), 3, nullptr);
    EXPECT_THROW(([&] { (void)potrs<TestFixture::BK, T>(this->q(), nonsquare, p.b, Uplo::Lower); }()), batchlas::invalid_argument);
}
