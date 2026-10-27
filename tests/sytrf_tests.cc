// sytrf / sytrs / sysv: Bunch-Kaufman LDL^T with LAPACK ?sytrf semantics. Every route is pinned and
// checked against the full symmetric matrix through batchlas::verify; the operand layout is never
// natural (padded ld, padded batch stride, both pads and the unreferenced triangle poisoned with a large
// finite value), complex data has nonzero imaginary parts and is symmetric, not Hermitian.
// evidence: docs/perf/sytrf.md
#include <gtest/gtest.h>

#include <batchlas/blas/functions/sysv.hh>
#include <batchlas/blas/functions/sytrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/tuning_params.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <tests/test_support/coverage_rows.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <src/ops/sysv/choice.hh>
#include <src/ops/sytrf/choice.hh>
#include <src/ops/sytrf/sytrf_native.hh>
#include <src/ops/sytrs/choice.hh>
#include <src/select/select.hh>
#include <src/select/vendor.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace ts = test_support;
using ts::Live;
using verify::Check;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// random: uniform [-1, 1], every pivot kind occurs; dominant: |diagonal| = n + 2 with alternating
// signs, so every pivot is 1 x 1 with no interchange; kkt: [[H, A^T], [A, 0]] with a small H, which
// forces 2 x 2 pivots.
enum class Gen { random, dominant, kkt };

// full: the whole symmetric matrix per item, packed; a: its uplo triangle in padded storage (ld n + 3,
// stride + 7), b: B padded (ld n + 2, stride + 5), b0 a snapshot of b; piv and info hold sentinels.
template <typename T>
struct Sys {
    int n = 0, nrhs = 0;
    Uplo uplo = Uplo::Lower;
    Matrix<T> full, a, b, b0;
    UnifiedVector<int64_t> piv;
    UnifiedVector<int32_t> info;

    int batch() const { return a.batch_size(); }
    const int* ipiv(int item) const { return reinterpret_cast<const int*>(piv.data()) + size_t(item) * n; }
    int* ipiv(int item) { return reinterpret_cast<int*>(piv.data()) + size_t(item) * n; }
    bool stored(int i, int j) const { return Live::triangle(uplo).contains(i, j, n, n); }
};

template <typename T>
T draw(std::mt19937& g, double scale) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const double re = u(g), im = u(g);
    return MV<T>::make(scale * re, scale * im);
}

// Complex symmetric (not Hermitian) item `item` of m: no library generator draws one, nor a KKT matrix.
template <typename T>
void symmetric(Matrix<T>& m, int item, Gen gen, std::mt19937& g) {
    const int n = m.rows();
    const int nh = n - (gen == Gen::kkt ? n / 3 : 0);  // n - nh constraint rows: the zero block
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i) {
            T v = draw<T>(g, 1.0);
            if (gen == Gen::kkt) {
                if (i < nh) v = draw<T>(g, 1e-2);  // H
                else if (j >= nh) v = T{};         // the zero block
            }
            if (gen == Gen::dominant && i == j) v = MV<T>::make((j % 2 ? -1.0 : 1.0) * (n + 2), 0.5);
            m(i, j, item) = m(j, i, item) = v;
        }
}

template <typename T>
Sys<T> make_sys(int n, int nrhs, int batch, Uplo uplo, Gen gen, unsigned seed, bool identical = false) {
    Matrix<T> full(n, n, batch), b(n, nrhs, batch);
    std::mt19937 g(seed);
    for (int bi = 0; bi < batch; ++bi) {
        if (identical && bi > 0) g.seed(seed);
        symmetric(full, bi, gen, g);
        for (int c = 0; c < nrhs; ++c)
            for (int i = 0; i < n; ++i) b(i, c, bi) = draw<T>(g, 1.0);
    }
    Sys<T> p{n, nrhs, uplo, std::move(full), ts::padded<T>(n, n, batch, 3, 7), ts::padded<T>(n, nrhs, batch, 2, 5),
             Matrix<T>(n, nrhs, batch), UnifiedVector<int64_t>(size_t(n) * batch, int64_t(0x0BADBEEF0BADBEEFLL)),
             UnifiedVector<int32_t>(size_t(batch), int32_t(-12345))};
    ts::store<T>(p.a, p.full, Live::triangle(uplo));
    ts::store<T>(p.b, b);
    p.b0 = p.b.clone();
    return p;
}

// The unreferenced triangle and both pads still hold the poison.
template <typename T>
::testing::AssertionResult untouched(const Sys<T>& p) {
    if (auto r = ts::unwritten_outside<T>(p.a, Live::triangle(p.uplo)); !r) return r << " in A";
    return ts::unwritten_outside<T>(p.b, Live::block()) << " in B";
}

// Zeroes item s's stored triangle (a singular item; full keeps the regular matrix).
template <typename T>
void zero_item(Sys<T>& p, int s) {
    ts::store<T>(p.a.batch_item(s), Matrix<T>::Zeros(p.n, p.n, 1), Live::triangle(p.uplo));
}

// LAPACK's encoding, checked: |ipiv| in [1, n], a 2 x 2 block is two equal negative entries.
// Returns {1 x 1 interchanges, 2 x 2 blocks}, or {-1, -1} for a malformed list.
template <typename T>
std::pair<int, int> pivot_kinds(const Sys<T>& p, int item) {
    const int* ip = p.ipiv(item);
    int swaps = 0, blocks = 0;
    for (int k = 0; k < p.n;) {
        const int v = ip[k];
        if (v == 0 || std::abs(v) > p.n) return {-1, -1};
        if (v > 0) { swaps += v != k + 1; ++k; continue; }
        if (k + 1 >= p.n || ip[k + 1] != v) return {-1, -1};
        ++blocks;
        k += 2;
    }
    return {swaps, blocks};
}

// Whether item `item` has a 2 x 2 block whose first LOGICAL column is k (logical i is physical
// n-1-i for Upper): k = nb - 1 is a block across the first blocked-panel boundary.
template <typename T>
bool block_at(const Sys<T>& p, int item, int k) {
    const int* ip = p.ipiv(item);
    auto lp = [&](int i) { return ip[p.uplo == Uplo::Upper ? p.n - 1 - i : i]; };
    for (int j = 0; j < p.n;) {
        if (lp(j) > 0) { ++j; continue; }
        if (j == k) return true;
        j += 2;
    }
    return false;
}

// Item b's ipiv in x equals item b's in y (item y_item's when >= 0), for each of `items` (empty: all).
template <typename T>
::testing::AssertionResult same_pivots(const Sys<T>& x, const Sys<T>& y, const std::vector<int>& items = {}, int y_item = -1) {
    for (int b : items.empty() ? x.a.all_items() : items)
        for (int k = 0; k < x.n; ++k)
            if (x.ipiv(b)[k] != y.ipiv(y_item < 0 ? b : y_item)[k])
                return ::testing::AssertionFailure() << "ipiv differs: item " << b << " entry " << k << " ("
                                                     << x.ipiv(b)[k] << " vs " << y.ipiv(y_item < 0 ? b : y_item)[k] << ")";
    return ::testing::AssertionSuccess();
}

// The logical partner row the 2 x 2 block at logical k of item `item` interchanged with logical
// k + 1 (k + 1 itself when none), or -1 when no block starts at k.
template <typename T>
int block_partner(const Sys<T>& p, int item, int k) {
    if (!block_at(p, item, k)) return -1;
    auto phys = [&](int i) { return p.uplo == Uplo::Upper ? p.n - 1 - i : i; };
    return phys(-p.ipiv(item)[phys(k)] - 1);
}

// A 2 x 2 block forced at logical step k of item s of a dominant system: diagonals k and r (default
// k + 1) zero and a coupling (r, k) that dominates both columns, so ?sytf2 takes kstep 2 at k and
// interchanges rows k + 1 and r (none when r = k + 1).
template <typename T>
void force_block(Sys<T>& p, int s, int k, int r = -1) {
    if (r < 0) r = k + 1;
    auto phys = [&](int i) { return p.uplo == Uplo::Upper ? p.n - 1 - i : i; };
    auto put = [&](int li, int lj, T v) {
        const int i = phys(li), j = phys(lj);
        p.full(i, j, s) = p.full(j, i, s) = v;
        (p.stored(i, j) ? p.a(i, j, s) : p.a(j, i, s)) = v;
    };
    put(k, k, T{});
    put(r, r, T{});
    put(r, k, MV<T>::make(2.0 * (p.n + 2), 0.25));
}

// Sets the stored (referenced) entries of physical row and column j of item s to v.
template <typename T>
void set_index(Sys<T>& p, int s, int j, T v) {
    for (int i = 0; i < p.n; ++i) {
        if (p.stored(i, j)) p.a(i, j, s) = v;
        if (p.stored(j, i)) p.a(j, i, s) = v;
    }
}

template <typename Config>
class SytrfTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend Bk = Config::BackendVal;

    static std::vector<ops::sytrf::SytrfChoice> routes(int n) {
        std::vector<ops::sytrf::SytrfChoice> r;
        if (Bk != Backend::NETLIB) {
            if (n <= 32) r.push_back(ops::sytrf::Cta{});
            r.push_back(ops::sytrf::Wg{});
            r.push_back(ops::sytrf::Blocked{});
        }
        if ((Bk == Backend::CUDA && select::kHasCudaVendor) || (Bk == Backend::NETLIB && select::kHasNetlib))
            r.push_back(ops::sytrf::Vendor{});
        return r;
    }

    void factor(Sys<T>& p, const ops::sytrf::SytrfChoice& c) {
        const select::ScopedPin<ops::sytrf::SytrfChoice> pin("sytrf", c);
        factor_auto(p);
    }

    void factor_auto(Sys<T>& p) {
        const size_t need = sytrf_buffer_size<Bk, T>(*this->ctx, p.a, p.uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        (void)sytrf<Bk, T>(*this->ctx, p.a, p.uplo, p.piv.to_span(), Span<std::byte>(ws.data(), need), p.info.to_span());
        this->ctx->wait();
    }

    void solve(Sys<T>& p) {
        const size_t need = sytrs_buffer_size<Bk, T>(*this->ctx, p.a, p.b, p.uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        (void)sytrs<Bk, T>(*this->ctx, p.a, p.b, p.uplo, p.piv.to_span(), Span<std::byte>(ws.data(), need));
        this->ctx->wait();
    }

    void solve(Sys<T>& p, const ops::sytrs::SytrsChoice& c) {
        const select::ScopedPin<ops::sytrs::SytrsChoice> pin("sytrs", c);
        solve(p);
    }

    void sysv_run(Sys<T>& p, const ops::sysv::SysvChoice* c) {
        std::optional<select::ScopedPin<ops::sysv::SysvChoice>> pin;
        if (c) pin.emplace("sysv", *c);
        const size_t need = sysv_buffer_size<Bk, T>(*this->ctx, p.a, p.b, p.uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        (void)sysv<Bk, T>(*this->ctx, p.a, p.b, p.uplo, p.piv.to_span(), Span<std::byte>(ws.data(), need),
                          p.info.to_span());
        this->ctx->wait();
    }

    // ||full X - B0|| over `items` (empty: every item), X read in place from the padded b.
    double residual(const Sys<T>& p, const std::vector<int>& items = {}) {
        return MV<T>::solve_residual(*this->ctx, p.full, p.b, p.b0, items.empty() ? p.b.all_items() : items);
    }

    std::vector<ops::sysv::SysvChoice> sysv_routes(int n) const {
        std::vector<ops::sysv::SysvChoice> r;
        if (Bk == Backend::NETLIB && !select::kHasNetlib) return r;  // compose's sytrf has no route
        if (Bk != Backend::NETLIB && n <= 32) r.push_back(ops::sysv::Cta{});
        r.push_back(ops::sysv::Compose{});
        return r;
    }

    // Whether sytrs `cta` can take order n here: the register tile and this device's local memory.
    bool cta_fits(int n) const {
        const auto& d = select::describe(this->ctx->device(), Bk, false);
        return Bk != Backend::NETLIB && n <= sycl_sytrf::kSytrsCtaMaxN &&
               static_cast<int64_t>(sycl_sytrf::sytrs_cta_slm_bytes<T>(n)) <= d.slm_budget;
    }

    std::vector<ops::sytrs::SytrsChoice> solve_routes(int n) const {
        std::vector<ops::sytrs::SytrsChoice> r;
        if (Bk != Backend::NETLIB) {
            if (cta_fits(n)) r.push_back(ops::sytrs::Cta{});
            r.push_back(ops::sytrs::Blocked{});
            r.push_back(ops::sytrs::Wg{});
        } else if (select::kHasNetlib) {
            r.push_back(ops::sytrs::Vendor{});
        }
        return r;
    }
};

using SytrfTestTypes = typename test_utils::backend_types<ts::Config>::type;

// Every item but `s`.
std::vector<int> all_but(int batch, int s) {
    std::vector<int> v;
    for (int b = 0; b < batch; ++b)
        if (b != s) v.push_back(b);
    return v;
}

}  // namespace

TYPED_TEST_SUITE(SytrfTest, SytrfTestTypes);

// The test of the tests: each generator produces the pivot mix the suite relies on. random has
// interchanges and 2 x 2 blocks somewhere in the batch, dominant none of either, kkt a 2 x 2 block in
// every item. A generator that stopped forcing its case would turn every test below vacuous.
TYPED_TEST(SytrfTest, GeneratorsForceTheirPivotKinds) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (const auto& c : this->routes(24)) {
            auto r = make_sys<T>(24, 1, 8, uplo, Gen::random, 11u);
            this->factor(r, c);
            int swaps = 0, blocks = 0;
            for (int bi = 0; bi < r.batch(); ++bi) {
                const auto [s, b] = pivot_kinds(r, bi);
                ASSERT_GE(s, 0) << "malformed ipiv, item " << bi << " " << select::to_string(c) << ts::uplo_name(uplo);
                swaps += s;
                blocks += b;
            }
            EXPECT_GT(swaps, 0) << select::to_string(c) << ts::uplo_name(uplo);
            EXPECT_GT(blocks, 0) << select::to_string(c) << ts::uplo_name(uplo);
            auto d = make_sys<T>(24, 1, 4, uplo, Gen::dominant, 12u);
            this->factor(d, c);
            for (int bi = 0; bi < d.batch(); ++bi) EXPECT_EQ(pivot_kinds(d, bi), std::make_pair(0, 0));
            auto k = make_sys<T>(24, 1, 4, uplo, Gen::kkt, 13u);
            this->factor(k, c);
            for (int bi = 0; bi < k.batch(); ++bi) EXPECT_GT(pivot_kinds(k, bi).second, 0) << select::to_string(c);
        }
}

// sytrf on every runnable route, then sytrs: the residual against the full symmetric matrix over every
// item, info = 0, and the unreferenced triangle and the pads bit-unchanged. Orders straddle the CTA
// ceiling (32 | 33), the 256-wide work-group (256 | 257) and the blocked panel (31 | 32, 62 | 63).
TYPED_TEST(SytrfTest, FactorThenSolveEveryRoute) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (Gen gen : {Gen::random, Gen::dominant, Gen::kkt})
            for (int n : {1, 2, 3, 5, 8, 17, 31, 32, 33, 62, 63, 64, 100, 257}) {
                if (gen == Gen::kkt && n < 3) continue;
                for (const auto& c : this->routes(n)) {
                    auto p = make_sys<T>(n, 3, 5, uplo, gen, 100u + unsigned(n));
                    this->factor(p, c);
                    this->solve(p);
                    const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " gen=" +
                                             std::to_string(int(gen)) + " n=" + std::to_string(n);
                    for (int bi = 0; bi < p.batch(); ++bi) EXPECT_EQ(p.info[bi], 0) << what;
                    EXPECT_VERIFY(T, Check::solve, n, this->residual(p)) << what;
                    EXPECT_TRUE(untouched(p)) << what;
                }
            }
}

// sysv on both routes, KKT systems included: the solve, ipiv's encoding and the untouched triangle.
TYPED_TEST(SytrfTest, SysvEveryRoute) {
    using T = typename TestFixture::T;
    if (this->sysv_routes(4).empty()) {  // NETLIB without LAPACKE: the contract is NoRouteError, for both ops
        auto p = make_sys<T>(4, 1, 2, Uplo::Lower, Gen::random, 1u);
        EXPECT_THROW(this->sysv_run(p, nullptr), batchlas::NoRouteError);
        EXPECT_THROW(this->factor_auto(p), batchlas::NoRouteError);
        return;
    }
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (Gen gen : {Gen::random, Gen::kkt})
            for (int n : {4, 9, 32, 33, 96})
                for (int nrhs : {1, 4, 9}) {  // 8 | 9 straddles cta's sub-groups per work-group
                for (const auto& c : this->sysv_routes(n)) {
                    auto p = make_sys<T>(n, nrhs, 6, uplo, gen, 300u + unsigned(n));
                    this->sysv_run(p, &c);
                    const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " n=" + std::to_string(n);
                    for (int bi = 0; bi < p.batch(); ++bi) {
                        EXPECT_EQ(p.info[bi], 0) << what;
                        EXPECT_GE(pivot_kinds(p, bi).first, 0) << what;
                        if (gen == Gen::kkt) EXPECT_GT(pivot_kinds(p, bi).second, 0) << what;
                    }
                    EXPECT_VERIFY(T, Check::solve, n, this->residual(p)) << what;
                    EXPECT_TRUE(untouched(p)) << what;
                }
            }
}

// A singular item reports info > 0 and disturbs nobody: every other item's factor, ipiv, info and X is
// bit-identical to the same batch with a regular matrix in that slot. The zero matrix fails at the
// first step, which is physical column 1 (Lower) or n (Upper), as LAPACK numbers it. sysv keeps the
// singular item's B, as LAPACK ?sysv does.
TYPED_TEST(SytrfTest, SingularItemReportsInfoWithoutDisturbingTheOthers) {
    using T = typename TestFixture::T;
    const int n = 20, s = 2;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (const auto& c : this->sysv_routes(n)) {
            auto good = make_sys<T>(n, 2, 6, uplo, Gen::kkt, 77u);
            auto bad = make_sys<T>(n, 2, 6, uplo, Gen::kkt, 77u);
            zero_item(bad, s);
            this->sysv_run(good, &c);
            this->sysv_run(bad, &c);
            const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo);
            EXPECT_EQ(bad.info[s], uplo == Uplo::Lower ? 1 : n) << what;
            const auto others = all_but(bad.batch(), s);
            for (int bi : others) EXPECT_EQ(bad.info[bi], 0) << what;
            ASSERT_TRUE(ts::items_bit_identical_to<T>(bad.a, good.a, others)) << what;
            ASSERT_TRUE(ts::items_bit_identical_to<T>(bad.b, good.b, others)) << what;
            ASSERT_TRUE(same_pivots(bad, good, others)) << what;
            ASSERT_TRUE(ts::items_bit_identical_to<T>(bad.b, bad.b0, std::vector{s})) << what;
        }
}

// The same algorithm in local memory (cta) and in global memory (wg) does the same arithmetic in the
// same order: for n <= 32 the factor, ipiv and info agree bit for bit. sysv's fused cta then equals
// the cta factor plus sytrs (both reduce over one sub-group).
TYPED_TEST(SytrfTest, CtaAndWgAgreeBitForBit) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {7, 24, 32}) {
            auto p = make_sys<T>(n, 2, 9, uplo, Gen::random, 500u + unsigned(n));
            auto q = make_sys<T>(n, 2, 9, uplo, Gen::random, 500u + unsigned(n));
            auto f = make_sys<T>(n, 2, 9, uplo, Gen::random, 500u + unsigned(n));
            auto k = make_sys<T>(n, 2, 9, uplo, Gen::random, 500u + unsigned(n));
            this->factor(p, ops::sytrf::Cta{});
            this->factor(q, ops::sytrf::Wg{});
            this->factor(k, ops::sytrf::Blocked{});  // below one panel the blocked tier is the wg kernel
            this->solve(p, ops::sytrs::Cta{});
            const ops::sysv::SysvChoice cta = ops::sysv::Cta{};
            this->sysv_run(f, &cta);
            const bool one_panel = n <= sycl_sytrf::kSytrfBlockedNb;
            const std::string what = std::string(" n=") + std::to_string(n) + " " + ts::uplo_name(uplo);
            ASSERT_TRUE(ts::items_bit_identical_to<T>(p.a, q.a)) << "factor" << what;
            ASSERT_TRUE(ts::items_bit_identical_to<T>(p.a, f.a)) << "sysv factor" << what;
            ASSERT_TRUE(ts::items_bit_identical_to<T>(p.b, f.b)) << "X" << what;
            ASSERT_TRUE(same_pivots(p, q)) << what;
            ASSERT_TRUE(same_pivots(p, f)) << what;
            if (one_panel) {
                ASSERT_TRUE(ts::items_bit_identical_to<T>(k.a, q.a)) << "blocked" << what;
                ASSERT_TRUE(same_pivots(k, q)) << "blocked" << what;
            }
        }
}

// The vendor's factor (cuSOLVER ?sytrf) solved by the native sytrs: ipiv's convention and the stored
// D and L are LAPACK's on both sides, or this residual fails.
TYPED_TEST(SytrfTest, VendorFactorSolvesWithNativeSytrs) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk != Backend::CUDA) GTEST_SKIP() << "the CUDA vendor arm only";
    else {
        if (!select::kHasCudaVendor) GTEST_SKIP() << "vendor-free build";
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {6, 32, 65}) {
                auto p = make_sys<T>(n, 2, 4, uplo, Gen::kkt, 900u + unsigned(n));
                this->factor(p, ops::sytrf::Vendor{});
                this->solve(p);
                for (int bi = 0; bi < p.batch(); ++bi) EXPECT_GT(pivot_kinds(p, bi).second, 0);
                EXPECT_VERIFY(T, Check::solve, n, this->residual(p)) << "n=" << n << ts::uplo_name(uplo);
                EXPECT_TRUE(untouched(p)) << "vendor wrote outside the triangle, n=" << n << ts::uplo_name(uplo);
            }
    }
}

// The pivot decisions are ?sytf2's: on random data (no near ties) the wg kernel, whose argmax spans
// several sub-groups above n = 32, picks cuSOLVER's ipiv entry for entry. A residual alone cannot see a
// wrong pivot choice: any nonzero pivot still gives a valid factorization.
TYPED_TEST(SytrfTest, WgPivotsMatchVendor) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk != Backend::CUDA) GTEST_SKIP() << "the CUDA vendor arm only";
    else {
        if (!select::kHasCudaVendor) GTEST_SKIP() << "vendor-free build";
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {24, 100, 200}) {
                auto w = make_sys<T>(n, 1, 3, uplo, Gen::random, 700u + unsigned(n));
                auto v = make_sys<T>(n, 1, 3, uplo, Gen::random, 700u + unsigned(n));
                this->factor(w, ops::sytrf::Wg{});
                this->factor(v, ops::sytrf::Vendor{});
                ASSERT_TRUE(same_pivots(w, v)) << "n=" << n << ts::uplo_name(uplo);
            }
    }
}

// A saturating batch of identical KKT matrices: every item's factor, ipiv, info and X must equal item
// 0's bit for bit, on the local-memory kernels (cta, sysv cta), the work-group kernels (wg, sytrs) and
// the blocked tier with its trailing gemm (n = 170).
TYPED_TEST(SytrfTest, SaturatingBatchIsBitIdenticalToItemZero) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    const int batch = 1024;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int route = 0; route < 4; ++route) {
            const int n = route == 2 ? 70 : route == 3 ? 170 : 32;  // 170: the blocked tier's gemm runs
            auto p = make_sys<T>(n, 2, batch, uplo, Gen::kkt, 4242u, /*identical=*/true);
            if (route == 0) {
                const ops::sysv::SysvChoice cta = ops::sysv::Cta{};
                this->sysv_run(p, &cta);
            } else {
                const ops::sytrf::SytrfChoice c = route == 1   ? ops::sytrf::SytrfChoice{ops::sytrf::Cta{}}
                                                  : route == 2 ? ops::sytrf::SytrfChoice{ops::sytrf::Wg{}}
                                                               : ops::sytrf::SytrfChoice{ops::sytrf::Blocked{}};
                this->factor(p, c);
                this->solve(p);
            }
            EXPECT_GT(pivot_kinds(p, 0).second, 0);
            EXPECT_VERIFY(T, Check::solve, n, this->residual(p, {0, batch - 1})) << route;
            for (int bi = 1; bi < batch; ++bi) ASSERT_EQ(p.info[bi], p.info[0]) << "route " << route << " item " << bi;
            ASSERT_TRUE(ts::items_bit_identical<T>(p.a)) << "route " << route;
            ASSERT_TRUE(ts::items_bit_identical<T>(p.b)) << "route " << route;
            ASSERT_TRUE(same_pivots(p, p, {}, 0)) << "route " << route;
        }
}

// The CTA ceiling straddled both ways: a cta pin runs at 32 and throws at 33, and Auto under the
// shipped table runs cta at 32 and wg at 33 (bit-identical to the pinned run).
TYPED_TEST(SytrfTest, CtaCeilingStraddled) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    for (int n : {32, 33}) {
        auto pinned = make_sys<T>(n, 1, 128, Uplo::Lower, Gen::random, 31u);
        auto autop = make_sys<T>(n, 1, 128, Uplo::Lower, Gen::random, 31u);
        if (n == 33) {
            EXPECT_ANY_THROW(this->factor(pinned, ops::sytrf::Cta{}));
            pinned = make_sys<T>(n, 1, 128, Uplo::Lower, Gen::random, 31u);
        }
        this->factor(pinned, n <= 32 ? ops::sytrf::SytrfChoice{ops::sytrf::Cta{}}
                                     : ops::sytrf::SytrfChoice{ops::sytrf::Wg{}});
        this->factor_auto(autop);
        ASSERT_TRUE(ts::items_bit_identical_to<T>(pinned.a, autop.a)) << "n=" << n;
    }
}

// Validation and the empty problem.
TYPED_TEST(SytrfTest, ArgumentChecks) {
    using T = typename TestFixture::T;
    UnifiedVector<T> buf(64, T{});
    UnifiedVector<int64_t> piv(64, 0);
    const MV<T> rect(buf.data(), 4, 3, 4, 12, 1);
    EXPECT_THROW(((void)sytrf<TestFixture::Bk, T>(*this->ctx, rect, Uplo::Lower, piv.to_span(), Span<std::byte>{},
                                                  Span<int32_t>{})),
                 batchlas::invalid_argument);
    const MV<T> sq(buf.data(), 4, 4, 4, 16, 1);
    const MV<T> b3(buf.data(), 3, 1, 3, 3, 1);
    EXPECT_THROW(((void)sytrs<TestFixture::Bk, T>(*this->ctx, sq, b3, Uplo::Lower, piv.to_span(), Span<std::byte>{})),
                 batchlas::invalid_argument);
    const MV<T> empty(buf.data(), 4, 4, 4, 16, 0);
    EXPECT_NO_THROW(((void)sytrf<TestFixture::Bk, T>(*this->ctx, empty, Uplo::Upper, piv.to_span(), Span<std::byte>{},
                                                     Span<int32_t>{})));
}

// The blocked tier (?lasyf panels of nb = 31 columns, then the public gemm) against the unblocked wg
// kernel and the vendor: on random data ipiv agrees entry for entry; info is 0, the triangle and
// pads are untouched and the solve holds on random and KKT data. Orders straddle the panel
// boundaries, and some item must take a 2 x 2 block across the first one (logical columns 30 and
// 31), where a panel ends one column late and the next starts one late.
TYPED_TEST(SytrfTest, BlockedMatchesWgAndVendor) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    const bool vendor = TestFixture::Bk == Backend::CUDA && select::kHasCudaVendor;
    constexpr int nb = sycl_sytrf::kSytrfBlockedNb;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        int straddles = 0;
        for (Gen gen : {Gen::random, Gen::kkt})
            for (int n : {nb + 1, 2 * nb, 2 * nb + 1, 2 * nb + 2, 100, 200, 300}) {
                const unsigned seed = 1300u + unsigned(n) + (gen == Gen::kkt ? 7u : 0u);
                auto bl = make_sys<T>(n, 2, 16, uplo, gen, seed);
                auto wg = make_sys<T>(n, 2, 16, uplo, gen, seed);
                this->factor(bl, ops::sytrf::Blocked{});
                this->factor(wg, ops::sytrf::Wg{});
                const std::string what = std::string(ts::uplo_name(uplo)) + " gen=" + std::to_string(int(gen)) + " n=" + std::to_string(n);
                for (int bi = 0; bi < bl.batch(); ++bi) {
                    EXPECT_EQ(bl.info[bi], 0) << what;
                    straddles += block_at(bl, bi, nb - 1) ? 1 : 0;
                }
                if (gen == Gen::random) {
                    ASSERT_TRUE(same_pivots(bl, wg)) << what << " vs wg";
                    if (vendor) {
                        auto v = make_sys<T>(n, 2, 16, uplo, gen, seed);
                        this->factor(v, ops::sytrf::Vendor{});
                        ASSERT_TRUE(same_pivots(bl, v)) << what << " vs vendor";
                    }
                }
                EXPECT_TRUE(untouched(bl)) << what;
                this->solve(bl);
                EXPECT_VERIFY(T, Check::solve, n, this->residual(bl)) << what;
            }
        EXPECT_GT(straddles, 0) << ts::uplo_name(uplo) << ": no 2 x 2 block across the first panel boundary";
    }
}

// Zero rows and columns of a dominant item (no interchanges, so they stay zero): info is the first
// in loop order, min + 1 (Lower) or max + 1 (Upper), and every other item is bit-identical to the
// batch without the singular item. At n = 100 blocked's panels are logical [0, 93) and its ?sytf2
// tail [93, 100): {95} Lower and {4} Upper sit in the tail only, {40, 95} and {4, 60} in a panel
// and the tail (the panel's info must survive the hand-off), {40, 70} in panels only.
TYPED_TEST(SytrfTest, SingularColumnsReportTheFirstInLoopOrder) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    const int n = 100, s = 3;
    const std::vector<std::vector<int>> zero_sets{{40, 70}, {95}, {4}, {40, 95}, {4, 60}};
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (const auto& zs : zero_sets)
        for (const ops::sytrf::SytrfChoice c : {ops::sytrf::SytrfChoice{ops::sytrf::Wg{}},
                                                ops::sytrf::SytrfChoice{ops::sytrf::Blocked{}}}) {
            auto good = make_sys<T>(n, 1, 6, uplo, Gen::dominant, 51u);
            auto bad = make_sys<T>(n, 1, 6, uplo, Gen::dominant, 51u);
            for (int j : zs) set_index(bad, s, j, T{});
            this->factor(good, c);
            this->factor(bad, c);
            const auto [lo, hi] = std::minmax_element(zs.begin(), zs.end());
            const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " zero " + std::to_string(*lo) +
                                     (zs.size() > 1 ? "," + std::to_string(*hi) : "");
            EXPECT_EQ(bad.info[s], (uplo == Uplo::Lower ? *lo : *hi) + 1) << what;
            const auto others = all_but(bad.batch(), s);
            for (int bi : others) EXPECT_EQ(bad.info[bi], 0) << what;
            ASSERT_TRUE(ts::items_bit_identical_to<T>(bad.a, good.a, others)) << what;
            ASSERT_TRUE(same_pivots(bad, good, others)) << what;
        }
}

// Termination for any data. NaN and Inf reach every pivot compare, and items aliased by a stride
// shorter than the matrix race each other's writes: lane-divergent pivot decisions would deadlock a
// barrier, so each call must return (ctest's TIMEOUT bounds a hang). A smoke test: it provokes no
// divergence on the 4090 box, so it is not a guard for the broadcast. A NaN first diagonal reports
// info > 0, and the finite items are bit-identical to a batch without the others.
// evidence: docs/perf/sytrf.md#sytrf-termination
TYPED_TEST(SytrfTest, NonFiniteAndAliasedItemsTerminate) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    const T nan = MV<T>::make(std::numeric_limits<double>::quiet_NaN(), 0.0);
    const T inf = MV<T>::make(std::numeric_limits<double>::infinity(), 1.0);
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {20, 100})
            for (const auto& c : this->routes(n)) {
                if (std::holds_alternative<ops::sytrf::Vendor>(c)) continue;
                auto clean = make_sys<T>(n, 1, 6, uplo, Gen::random, 61u);
                auto dirty = make_sys<T>(n, 1, 6, uplo, Gen::random, 61u);
                const int first = uplo == Uplo::Lower ? 0 : n - 1;  // the first logical diagonal
                dirty.a(first, first, 1) = nan;
                dirty.a(uplo == Uplo::Lower ? n - 1 : 0, uplo == Uplo::Lower ? 0 : n - 1, 3) = inf;
                for (int j = 0; j < n; ++j) set_index(dirty, 4, j, nan);
                this->factor(clean, c);
                this->factor(dirty, c);
                const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " n=" + std::to_string(n);
                EXPECT_GT(dirty.info[1], 0) << what;
                ASSERT_TRUE(ts::items_bit_identical_to<T>(dirty.a, clean.a, std::vector{0, 2, 5})) << what;
                auto al = make_sys<T>(n, 1, 8, uplo, Gen::random, 62u);
                const MV<T> A(al.a.data_ptr(), n, n, al.a.ld(), al.a.ld(), al.batch());  // item b+1 = item b shifted one column
                const select::ScopedPin<ops::sytrf::SytrfChoice> pin("sytrf", c);
                const size_t need = sytrf_buffer_size<TestFixture::Bk, T>(*this->ctx, A, uplo);
                UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
                EXPECT_NO_THROW(((void)sytrf<TestFixture::Bk, T>(*this->ctx, A, uplo, al.piv.to_span(),
                                                                 Span<std::byte>(ws.data(), need), al.info.to_span())));
                this->ctx->wait();
            }
}

// sytrs reads ipiv as given: a list sytrf cannot have written (an entry 0 or beyond n, the poison
// 0x0BADBEEF, an unpaired 2 x 2 entry) gives that item an all-NaN X instead of an out-of-bounds
// access, and the other items are still solved. Every native sytrs family.
TYPED_TEST(SytrfTest, SytrsMalformedPivotsGiveNaN) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "LAPACKE ?sytrs does not check ipiv";
    const int n = 40;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (const auto& c : this->solve_routes(n)) {
            auto p = make_sys<T>(n, 2, 5, uplo, Gen::kkt, 81u);
            this->factor(p, ops::sytrf::Wg{});
            p.ipiv(1)[3] = 0;
            p.ipiv(2)[5] = n + 1;
            for (int k = 0; k < n; ++k) p.ipiv(3)[k] = 0x0BADBEEF;
            for (int k = 0; k < n; ++k) p.ipiv(4)[k] = k + 1;
            p.ipiv(4)[uplo == Uplo::Lower ? n - 1 : 0] = -1;  // the last logical entry, alone
            this->solve(p, c);
            const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo);
            for (int bi = 1; bi < 5; ++bi)
                for (int col = 0; col < p.nrhs; ++col)
                    for (int i = 0; i < n; ++i)
                        ASSERT_TRUE(std::isnan(std::real(p.b(i, col, bi))))
                            << what << " item " << bi << " row " << i;
            EXPECT_VERIFY(T, Check::solve, n, this->residual(p, {0})) << what;
            EXPECT_TRUE(untouched(p)) << what;
        }
}

// Every sytrs family on factors with interchanges (random) and 2 x 2 blocks in every item (kkt):
// residual, the pads and the unreferenced triangle untouched. n straddles cta's register-tile steps
// (32 | 33, 64 | 65) and its ceiling (128 | 129); nrhs straddles cta's sub-group count (8 | 9).
TYPED_TEST(SytrfTest, SytrsEveryFamily) {
    using T = typename TestFixture::T;
    if (this->solve_routes(4).empty()) GTEST_SKIP() << "no sytrs route in this build";
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (Gen gen : {Gen::random, Gen::kkt})
            for (int n : {1, 3, 17, 32, 33, 64, 65, 128, 129, 300})
                for (int nrhs : {1, 8, 9}) {
                    if (gen == Gen::kkt && n < 3) continue;
                    if (TestFixture::Bk == Backend::NETLIB && (n > 65 || nrhs == 8)) continue;  // one family: LAPACKE
                    for (const auto& c : this->solve_routes(n)) {
                        auto p = make_sys<T>(n, nrhs, 4, uplo, gen, 2000u + unsigned(n) + 7u * unsigned(nrhs));
                        this->factor_auto(p);
                        this->solve(p, c);
                        const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " gen=" +
                                                 std::to_string(int(gen)) + " n=" + std::to_string(n) +
                                                 " nrhs=" + std::to_string(nrhs);
                        for (int bi = 0; bi < p.batch(); ++bi) {
                            ASSERT_EQ(p.info[bi], 0) << what;
                            if (gen == Gen::kkt) ASSERT_GT(pivot_kinds(p, bi).second, 0) << what << " item " << bi;
                        }
                        EXPECT_VERIFY(T, Check::solve, n, this->residual(p)) << what;
                        EXPECT_TRUE(untouched(p)) << what;
                    }
                }
}

// A 2 x 2 step whose second row is interchanged with a row r != k + 1: the gather must move row
// k + 1, not row k. Forced on a dominant system at (1, n - 1 - s) and, from n = 12, (n / 2, n / 2 + 3),
// checked in ipiv first (else vacuous), then solved on every family.
TYPED_TEST(SytrfTest, SytrsTwoByTwoWithInterchangeEveryFamily) {
    using T = typename TestFixture::T;
    if (this->solve_routes(8).empty()) GTEST_SKIP() << "no sytrs route in this build";
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {8, 33, 65, 129})
            for (const auto& c : this->solve_routes(n)) {
                auto p = make_sys<T>(n, 3, 3, uplo, Gen::dominant, 3100u + unsigned(n));
                std::vector<std::pair<int, int>> forced[3];
                for (int s = 0; s < p.batch(); ++s) {
                    forced[s].push_back({1, n - 1 - s});
                    if (n >= 12) forced[s].push_back({n / 2, n / 2 + 3});
                    for (const auto& [k, r] : forced[s]) force_block(p, s, k, r);
                }
                this->factor_auto(p);
                const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " n=" + std::to_string(n);
                for (int s = 0; s < p.batch(); ++s) {
                    ASSERT_EQ(p.info[s], 0) << what;
                    for (const auto& [k, r] : forced[s])
                        ASSERT_EQ(block_partner(p, s, k), r) << what << " item " << s << " k=" << k;
                }
                this->solve(p, c);
                EXPECT_VERIFY(T, Check::solve, n, this->residual(p)) << what;
                EXPECT_TRUE(untouched(p)) << what;
            }
}

// Wide right-hand sides: cta's sub-groups each take many columns, blocked's trsm a wide B.
TYPED_TEST(SytrfTest, SytrsWideRightHandSides) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {40, 70})
            for (const auto& c : this->solve_routes(n)) {
                if (std::holds_alternative<ops::sytrs::Wg>(c)) continue;  // one work-group per column: slow, covered above
                auto p = make_sys<T>(n, 300, 3, uplo, Gen::kkt, 2500u + unsigned(n));
                this->factor_auto(p);
                this->solve(p, c);
                const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " n=" + std::to_string(n);
                EXPECT_VERIFY(T, Check::solve, n, this->residual(p)) << what;
                EXPECT_TRUE(untouched(p)) << what;
            }
}

// A 2 x 2 block whose rows sit in different register slots of cta (logical 31 | 32, 63 | 64,
// 95 | 96): D^-1 reads the partner across the lane-31 / lane-0 seam. Every family; the forced
// block is checked first, or the case is vacuous.
TYPED_TEST(SytrfTest, SytrsTwoByTwoAcrossTheLaneBoundary) {
    using T = typename TestFixture::T;
    if (this->solve_routes(4).empty()) GTEST_SKIP() << "no sytrs route in this build";
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {64, 128})
            for (const auto& c : this->solve_routes(n)) {
                auto p = make_sys<T>(n, 3, 3, uplo, Gen::dominant, 2600u + unsigned(n));
                for (int s = 0; s < p.batch(); ++s)
                    for (int k = 31; k + 1 < n; k += 32) force_block(p, s, k);
                this->factor_auto(p);
                const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " n=" + std::to_string(n);
                for (int s = 0; s < p.batch(); ++s)
                    for (int k = 31; k + 1 < n; k += 32) ASSERT_TRUE(block_at(p, s, k)) << what << " k=" << k;
                this->solve(p, c);
                EXPECT_VERIFY(T, Check::solve, n, this->residual(p)) << what;
            }
}

// A saturating batch of identical KKT items: every item's X equals item 0's bit for bit, on cta's
// local-memory kernel (n = 32 one register slot, n = 100 four; nrhs = 9, so sub-groups take two
// columns), blocked (its trsm and chunked passes) and wg.
TYPED_TEST(SytrfTest, SytrsSaturatingBatchIsBitIdenticalToItemZero) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    const int batch = 1024;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (const auto& [c, n] : {std::pair{ops::sytrs::SytrsChoice{ops::sytrs::Cta{}}, 32},
                                   std::pair{ops::sytrs::SytrsChoice{ops::sytrs::Cta{}}, 100},
                                   std::pair{ops::sytrs::SytrsChoice{ops::sytrs::Blocked{}}, 150},
                                   std::pair{ops::sytrs::SytrsChoice{ops::sytrs::Wg{}}, 40}}) {
            if (std::holds_alternative<ops::sytrs::Cta>(c) && !this->cta_fits(n)) continue;
            auto p = make_sys<T>(n, 9, batch, uplo, Gen::kkt, 4343u, /*identical=*/true);
            this->factor(p, ops::sytrf::Wg{});
            this->solve(p, c);
            const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " n=" + std::to_string(n);
            EXPECT_GT(pivot_kinds(p, 0).second, 0) << what;
            EXPECT_VERIFY(T, Check::solve, n, this->residual(p, {0, batch - 1})) << what;
            ASSERT_TRUE(ts::items_bit_identical<T>(p.b)) << what;
        }
}

// cta's ceiling straddled both ways: a pin runs at 128 and throws at 129 (when local memory fits
// 128 at all), and Auto solves on both sides.
TYPED_TEST(SytrfTest, SytrsCtaCeilingStraddled) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    for (int n : {sycl_sytrf::kSytrsCtaMaxN, sycl_sytrf::kSytrsCtaMaxN + 1}) {
        auto p = make_sys<T>(n, 2, 128, Uplo::Lower, Gen::random, 37u);
        this->factor_auto(p);
        if (this->cta_fits(n)) {
            this->solve(p, ops::sytrs::Cta{});
            EXPECT_VERIFY(T, Check::solve, n, this->residual(p, {0, 127})) << "cta n=" << n;
        } else {
            EXPECT_ANY_THROW(this->solve(p, ops::sytrs::Cta{})) << "n=" << n;
        }
        auto a = make_sys<T>(n, 2, 128, Uplo::Lower, Gen::random, 37u);
        this->factor_auto(a);
        this->solve(a);
        EXPECT_VERIFY(T, Check::solve, n, this->residual(a)) << "auto n=" << n;
    }
}

// sysv's compose hands its sytrf status to sytrs as the skip list: on every sytrs family the
// singular item keeps its B bit for bit and every other item is solved.
TYPED_TEST(SytrfTest, SytrsSkipListEveryFamily) {
    using T = typename TestFixture::T;
    if (this->sysv_routes(4).empty()) GTEST_SKIP() << "no sysv route in this build";
    const ops::sysv::SysvChoice compose = ops::sysv::Compose{};
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {20, 90})
            for (const auto& c : this->solve_routes(n)) {
                const int s = 2;
                auto p = make_sys<T>(n, 3, 5, uplo, Gen::kkt, 2700u + unsigned(n));
                zero_item(p, s);
                {
                    const select::ScopedPin<ops::sytrs::SytrsChoice> pin("sytrs", c);
                    this->sysv_run(p, &compose);
                }
                const std::string what = select::to_string(c) + " " + ts::uplo_name(uplo) + " n=" + std::to_string(n);
                ASSERT_GT(p.info[s], 0) << what;
                ASSERT_TRUE(ts::items_bit_identical_to<T>(p.b, p.b0, std::vector{s})) << what;
                EXPECT_VERIFY(T, Check::solve, n, this->residual(p, all_but(p.batch(), s))) << what;
            }
}

// blocked's batch chunks (forced to 7 over 23 items: 7 + 7 + 9), in order and out of order: every
// item is solved from its own factor and pivots, a malformed list in the last part gives that item
// NaN, and sysv's skip list reaches the item in the second part.
TYPED_TEST(SytrfTest, SytrsBlockedChunksKeepEachItemsPivots) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    else {
        const int n = 45, batch = 23;
        const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "7");
        ASSERT_EQ(tuning::blocked_chunk_items(batch, 1, 1, 1), 7) << "the override did not reach the accessor";
        const select::ScopedPin<ops::sytrs::SytrsChoice> pin("sytrs", ops::sytrs::Blocked{});
        Queue ooo(*this->ctx, /*in_order=*/false);
        for (Queue* q : {this->ctx.get(), &ooo}) {
            const std::string what = q == &ooo ? "out of order" : "in order";
            for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                auto p = make_sys<T>(n, 3, batch, uplo, Gen::kkt, 2800u);
                this->factor_auto(p);
                p.ipiv(21)[4] = 0;
                UnifiedVector<std::byte> ws(std::max<size_t>(1, sytrs_buffer_size<TestFixture::Bk, T>(*q, p.a, p.b, uplo)));
                (void)sytrs<TestFixture::Bk, T>(*q, p.a, p.b, uplo, p.piv.to_span(), ws.to_span());
                q->wait();
                EXPECT_VERIFY(T, Check::solve, n, this->residual(p, all_but(batch, 21))) << what << " " << ts::uplo_name(uplo);
                for (int i = 0; i < n; ++i) ASSERT_TRUE(std::isnan(std::real(p.b(i, 0, 21)))) << what;
                EXPECT_TRUE(untouched(p)) << what;
            }
        }
        const ops::sysv::SysvChoice compose = ops::sysv::Compose{};
        auto s = make_sys<T>(n, 2, batch, Uplo::Lower, Gen::kkt, 2900u);
        zero_item(s, 9);
        this->sysv_run(s, &compose);
        ASSERT_GT(s.info[9], 0);
        ASSERT_TRUE(ts::items_bit_identical_to<T>(s.b, s.b0, std::vector{9}));
        EXPECT_VERIFY(T, Check::solve, n, this->residual(s, all_but(batch, 9))) << "sysv compose, chunked";
    }
}

// Linked is not reached: each pinned native family leaves a `reached,sytrs` coverage row with its
// own spelling and the call's key (m = k = n, n = nrhs). threadsafe: the child re-executes the binary.
TYPED_TEST(SytrfTest, SytrsPinnedFamiliesAreReached) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    else {
        const std::vector<std::tuple<ops::sytrs::SytrsChoice, int, int, std::string>> runs{
            {ops::sytrs::Cta{}, 16, 4, "cta"}, {ops::sytrs::Blocked{}, 48, 3, "blocked"}, {ops::sytrs::Wg{}, 40, 2, "wg"}};
        const auto rows = ts::reached_rows([&] {
            for (const auto& [c, n, nrhs, name] : runs) {
                auto p = make_sys<T>(n, nrhs, 2, Uplo::Lower, Gen::random, 77u);
                this->factor(p, ops::sytrf::Wg{});
                this->solve(p, c);
            }
        });
        std::map<std::tuple<long long, long long, long long>, std::string> algo;  // (m, n, k) -> spelling
        for (const auto& r : rows)
            if (r.op == "sytrs") algo[{r.m, r.n, r.k}] = r.spelling;
        for (const auto& [c, n, nrhs, name] : runs) {
            const auto it = algo.find({n, nrhs, n});
            ASSERT_NE(it, algo.end()) << "no reached row for " << name << " (" << n << " " << nrhs << ")";
            EXPECT_EQ(it->second, name) << n << " " << nrhs;
        }
    }
}
