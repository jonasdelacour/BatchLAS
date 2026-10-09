// Every trmm candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-kernel-selection.md §12. The limit oracle reads the kernels' own predicates and
// capacities; which kernel ran is read back from the select trace or a bit-for-bit comparison
// with the direct launch, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/sycl_interop.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include "../src/select/coverage.hh"
#include "../src/select/vendor.hh"

#include "test_utils.hh"
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>

#include "../src/backends/triangular_expand.hh"
#include "../src/backends/trmm_triangular_tiles.hh"
#include "../src/expansion_budget.hh"
#include "../src/ops/trmm/choice.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace tm = batchlas::ops::trmm;
using C = tm::TrmmChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;

template <typename T>
T mk(RealOf<T> r, RealOf<T> i) {
    if constexpr (kCx<T>) return T(r, i);
    else return r;
}
template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

const char* side_s(Side s) { return s == Side::Left ? "L" : "R"; }
const char* uplo_s(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }
const char* trans_s(Transpose t) { return t == Transpose::NoTrans ? "N" : (t == Transpose::Trans ? "T" : "C"); }
const char* diag_s(Diag d) { return d == Diag::Unit ? "U" : "N"; }

struct Region {
    std::size_t off = 0;
    int rows = 0, cols = 0, ld = 0, stride = 0;
};

struct Spec {
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
    Diag diag = Diag::NonUnit;
    int n = 8, q = 4, batch = 3;  // n: A's order; q: B's other extent
    int period = 0;               // > 0: item it repeats item it % period
    unsigned seed = 1;
};

// A, B and C in one buffer of a large finite poison, at padded lds and strides that are not
// ld * cols. A's other triangle, a Unit diagonal, the pads and gaps hold the poison; C starts as
// poison too, since trmm overwrites it (beta = 0).
template <typename T>
struct Prob {
    Spec s;
    T alpha{};
    Region a, b, c;
    UnifiedVector<T> mem;
    std::vector<T> mem0;

    MVof<T> A() { return {mem.data() + a.off, s.n, s.n, a.ld, a.stride, s.batch}; }
    MVof<T> B() { return {mem.data() + b.off, b.rows, b.cols, b.ld, b.stride, s.batch}; }
    MVof<T> Cm() { return {mem.data() + c.off, c.rows, c.cols, c.ld, c.stride, s.batch}; }
    std::size_t ai(int it, int i, int j) const { return a.off + std::size_t(it) * a.stride + std::size_t(j) * a.ld + i; }
    std::size_t bi(int it, int i, int j) const { return b.off + std::size_t(it) * b.stride + std::size_t(j) * b.ld + i; }
    std::size_t ci(int it, int i, int j) const { return c.off + std::size_t(it) * c.stride + std::size_t(j) * c.ld + i; }
    bool owned(int i, int j) const { return s.uplo == Uplo::Lower ? i > j : i < j; }
};

template <typename T>
T poison() {
    return mk<T>(RealOf<T>(-999), RealOf<T>(777));
}

template <typename T>
Prob<T> make_prob(const Spec& s) {
    using R = RealOf<T>;
    Prob<T> p;
    p.s = s;
    p.alpha = mk<T>(R(1.5), R(-0.5));
    const int br = s.side == Side::Left ? s.n : s.q, bc = s.side == Side::Left ? s.q : s.n;
    p.a = {0, s.n, s.n, s.n + 3, (s.n + 3) * s.n + 5};
    p.b = {std::size_t(p.a.stride) * s.batch + 11, br, bc, br + 2, (br + 2) * bc + 7};
    p.c = {p.b.off + std::size_t(p.b.stride) * s.batch + 13, br, bc, br + 1, (br + 1) * bc + 3};
    p.mem = UnifiedVector<T>(p.c.off + std::size_t(p.c.stride) * s.batch + 17, poison<T>());
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<R> u(R(-1), R(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % std::max(reps, 1);
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i)
                if ((i == j && s.diag == Diag::NonUnit) || p.owned(i, j))
                    p.mem[p.ai(it, i, j)] = it < reps ? mk<T>(u(gen), u(gen)) : p.mem[p.ai(r, i, j)];
        for (int j = 0; j < bc; ++j)
            for (int i = 0; i < br; ++i) p.mem[p.bi(it, i, j)] = it < reps ? mk<T>(u(gen), u(gen)) : p.mem[p.bi(r, i, j)];
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

// Componentwise backward error of C = alpha op(A) B (Right: B op(A)) for one item through the
// library; A is read as its triangle (Unit: no diagonal), operands from the pristine copy.
template <typename T>
double residual(const Prob<T>& p, int it) {
    using namespace batchlas::verify;
    const Spec& s = p.s;
    auto at0 = [&](const Region& g) {
        return MVof<T>(const_cast<T*>(p.mem0.data()) + g.off, g.rows, g.cols, g.ld, g.stride, s.batch);
    };
    const MVof<T> A0 = at0(p.a), B0 = at0(p.b), C0 = at0(p.c);
    const MVof<T> C(const_cast<T*>(p.mem.data()) + p.c.off, p.c.rows, p.c.cols, p.c.ld, p.c.stride, s.batch);
    const bool unit = s.diag == Diag::Unit;
    const Shape tri = s.uplo == Uplo::Lower ? (unit ? Shape::unit_lower : Shape::lower) : (unit ? Shape::unit_upper : Shape::upper);
    const int one[] = {it};
    if (s.side == Side::Left)
        return gemm_backward_error(A0, tri, s.trans, B0, Shape::general, Transpose::NoTrans, C0, C, Shape::general,
                                   up(p.alpha), up(T(0)), one);
    return gemm_backward_error(B0, Shape::general, Transpose::NoTrans, A0, tri, s.trans, C0, C, Shape::general,
                               up(p.alpha), up(T(0)), one);
}

// Residuals of the checked items, every element outside C bit for bit (A's poison included),
// and, for a repeating batch, every item's C bit-identical to its representative's.
template <typename T>
void expect_trmm(const Prob<T>& p, const std::string& what) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0)
        for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    else if (s.batch <= 8)
        for (int it = 0; it < s.batch; ++it) items.push_back(it);
    else
        items = {0, 1, s.batch / 2, s.batch - 1};
    for (int it : items) {
        const double r = residual(p, it);
        ASSERT_TRUE(batchlas::verify::pass<T>(batchlas::verify::Check::blas, s.n, r))
            << what << " item " << it << " backward error " << r << " exceeds "
            << batchlas::verify::bound<T>(batchlas::verify::Check::blas, s.n);
    }
    std::vector<char> inc(p.mem.size(), 0);
    for (int it = 0; it < s.batch; ++it)
        for (int j = 0; j < p.c.cols; ++j)
            for (int i = 0; i < p.c.rows; ++i) inc[p.ci(it, i, j)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!inc[e]) ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": wrote outside C at element " << e;
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int j = 0; j < p.c.cols; ++j)
                for (int i = 0; i < p.c.rows; ++i)
                    ASSERT_TRUE(same_bits(p.mem[p.ci(it, i, j)], p.mem[p.ci(it % s.period, i, j)]))
                        << what << ": item " << it << " differs from its representative at (" << i << "," << j << ")";
}

std::string label(const Spec& s) {
    return std::string(side_s(s.side)) + uplo_s(s.uplo) + trans_s(s.trans) + diag_s(s.diag) +
           " order=" + std::to_string(s.n) + " q=" + std::to_string(s.q) + " batch=" + std::to_string(s.batch);
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

template <class F>
std::string traced_line(F&& run, std::string* all = nullptr) {
    const ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    try {
        run();
    } catch (...) {
        (void)::testing::internal::GetCapturedStderr();
        throw;
    }
    const std::string err = ::testing::internal::GetCapturedStderr();
    if (all) *all = err;
    std::istringstream in(err);
    for (std::string line; std::getline(in, line);)
        if (line.rfind("trmm ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no trmm trace line in: " + err + ">";
}
template <class F>
std::string traced_choice(F&& run, std::string* all = nullptr) {
    const std::string line = traced_line(std::forward<F>(run), all);
    const auto arrow = line.find(" -> ");
    if (arrow == std::string::npos) return line;
    const std::string tail = line.substr(arrow + 4);
    return tail.substr(0, tail.find(' '));
}

struct TableGuard {
    ~TableGuard() { select::testing::use_embedded_tables(); }
};

// A heterogeneous operand has no route on any backend: one native launch has one order (the
// expansion's gemm throws), and every vendor loop runs each item at the top-level (m, n), so the
// vendor is refused too. Every pin throws (the vendor class word falls back to Auto), and Auto
// throws runtime_error, or NoRouteError without a vendor library.
template <Backend B, class T>
void expect_heterogeneous_has_no_route(Queue& ctx) {
    using R = RealOf<T>;
    const int n = 16, q = 3, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch), Bm(n, q, batch), Cm(n, q, batch);
    A.fill(mk<T>(R(0.5), R(0)));
    Bm.fill(mk<T>(R(1), R(0.5)));
    UnifiedVector<int> act(batch), cols(batch);
    for (int b = 0; b < batch; ++b) act[b] = n - b, cols[b] = q - (b % 2);
    const auto hetA = A.view().with_active_dims(act.to_span(), act.to_span());
    const auto hetB = Bm.view().with_active_dims(act.to_span(), cols.to_span());
    const auto hetC = Cm.view().with_active_dims(act.to_span(), cols.to_span());
    ASSERT_TRUE(hetA.is_heterogeneous());
    auto call = [&] {
        (void)trmm<B, T>(ctx, hetA, hetB, hetC, T(1), Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit);
        ctx.wait();
    };
    auto expect_no_route = [&](const std::string& what) {
        if constexpr (select::level3_vendor_available<B>) EXPECT_THROW(call(), std::runtime_error) << what;
        else EXPECT_THROW(call(), batchlas::NoRouteError) << what;
    };
    for (const C& c : tm::candidates<T>()) {
        const Pin pin("trmm", c);
        if (std::holds_alternative<tm::Vendor>(c)) expect_no_route("vendor pin");
        else EXPECT_THROW(call(), std::invalid_argument) << select::to_string(c);
    }
    const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
    expect_no_route("auto");
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class TrmmCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MVof<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::level3_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU kernels";
        if constexpr (B != Backend::CUDA) GTEST_SKIP() << "the native families are wired for CUDA only";
    }

    std::size_t expand_bytes(int order, int batch) {
        return backend::detail::expanded_workspace_bytes<T>(*this->ctx, order, batch);
    }

    // The limit oracle: the tile kernel's own predicate (Left, homogeneous), the grid-z ceiling,
    // the expansion's fit (BATCHLAS_EXPAND_MAX_BYTES included), the vendor's presence.
    bool expect_runs(const C& c, const Spec& s) {
        const bool grid = s.batch <= 65535;
        if (std::holds_alternative<tm::Triangular>(c)) return grid && s.side == Side::Left;
        if (std::holds_alternative<tm::Expand>(c))
            return grid && backend::detail::expansion_fits(*this->ctx, s.n, s.batch, expand_bytes(s.n, s.batch));
        return kVendor;
    }
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<tm::Vendor>(c) && !kVendor; }

    void run(Prob<T>& p) {
        (void)trmm<B, T>(*this->ctx, p.A(), p.B(), p.Cm(), p.alpha, p.s.side, p.s.uplo, p.s.trans, p.s.diag);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Prob<T>& p) {
        const Pin pin("trmm", c);
        run(p);
    }
    bool pin_accepted(const C& c, Prob<T>& p) {
        const Pin pin("trmm", c);
        try {
            run(p);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }
    std::string auto_choice(const Spec& s) {
        const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
        auto p = make_prob<T>(s);
        return traced_choice([&] { run(p); });
    }

    // Each family's own launch: the tile kernel (it takes no side, so it cannot serve Right), the
    // expansion plus the public gemm, the vendor library.
    bool direct(const C& c, Prob<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const MV A = p.A(), Bm = p.B(), Cm = p.Cm();
        const Spec& s = p.s;
        try {
            if (std::holds_alternative<tm::Triangular>(c)) {
                if (s.side != Side::Left) {
                    if (why) *why = "the tile kernel has no Side::Right";
                    return false;
                }
                (void)backend::detail::trmm_triangular_tiles<T>(q, A, Bm, Cm, p.alpha, s.uplo, s.trans, s.diag);
            } else if (std::holds_alternative<tm::Expand>(c)) {
                const int k = s.n, ld = backend::detail::expanded_ld<T>(k);
                UnifiedVector<T> scratch(std::size_t(ld) * k * std::max(s.batch, 1));
                const MV E(scratch.data(), k, k, ld, ld * k, s.batch);
                backend::detail::expand_triangular<T>(q, E, A, s.uplo, s.diag).wait();
                if (s.side == Side::Left)
                    (void)gemm<B, T>(q, E, Bm, Cm, p.alpha, T(0), s.trans, Transpose::NoTrans, ComputePrecision::Default);
                else
                    (void)gemm<B, T>(q, Bm, E, Cm, p.alpha, T(0), Transpose::NoTrans, s.trans, ComputePrecision::Default);
            } else {
                if constexpr (kVendor) {
                    (void)backend::trmm_vendor<B, T>(q, A, Bm, Cm, p.alpha, s.side, s.uplo, s.trans, s.diag);
                } else {
                    if (why) *why = "no vendor BLAS in this build";
                    return false;
                }
            }
            q.wait();
            return true;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return false;
        }
    }
};

TYPED_TEST_SUITE(TrmmCandidates, Types);

// §8.1: every candidate on both sides, orders across the tile kernel's row tiles (16/32/64/128
// by dtype) and the expansion's packet padding, rhs counts across the 128-wide column tile.
TYPED_TEST(TrmmCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    int ran = 0, refused = 0;
    for (const C& c : tm::candidates<T>())
        for (Side side : {Side::Left, Side::Right})
            for (int n : {1, 16, 17, 33, 65, 129})
                for (int q : {1, 5, 130}) {
                    Spec s{side, n % 2 ? Uplo::Lower : Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, n, q, 3};
                    s.seed = 1000u + 31u * n + q;
                    auto p = make_prob<T>(s);
                    const std::string what = name(c, s);
                    if (TestFixture::vendor_word_falls_back(c)) {
                        EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), this->auto_choice(s)) << what;
                        expect_trmm(p, what + " (vendor-free: Auto)");
                        continue;
                    }
                    if (!this->expect_runs(c, s)) {
                        EXPECT_FALSE(this->pin_accepted(c, p)) << what << " was accepted";
                        ++refused;
                        continue;
                    }
                    ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
                    expect_trmm(p, what);
                    ++ran;
                }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// The expansion's fit term (R3), straddled with BATCHLAS_EXPAND_MAX_BYTES: the exact scratch
// size runs, one byte less is refused, and Auto then takes the other route.
TYPED_TEST(TrmmCandidates, ExpandFitStraddlesTheScratchBudget) {
    using T = typename TestFixture::T;
    for (Side side : {Side::Left, Side::Right}) {
        Spec s{side, Uplo::Upper, Transpose::Trans, Diag::Unit, 37, 6, 4};
        const std::size_t bytes = this->expand_bytes(s.n, s.batch);
        {
            const ScopedEnvVar cap("BATCHLAS_EXPAND_MAX_BYTES", std::to_string(bytes).c_str());
            auto p = make_prob<T>(s);
            ASSERT_TRUE(this->pin_accepted(C{tm::Expand{}}, p)) << label(s) << " at " << bytes << " bytes";
            expect_trmm(p, "expand at the exact budget " + label(s));
        }
        const ScopedEnvVar cap("BATCHLAS_EXPAND_MAX_BYTES", std::to_string(bytes - 1).c_str());
        auto p = make_prob<T>(s);
        EXPECT_FALSE(this->pin_accepted(C{tm::Expand{}}, p)) << label(s) << " at " << bytes - 1 << " bytes";
        auto r = make_prob<T>(s);
        const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
        if (side == Side::Right && !TestFixture::kVendor) {
            EXPECT_THROW(this->run(r), batchlas::NoRouteError) << label(s);
            continue;
        }
        EXPECT_EQ(traced_choice([&] { this->run(r); }), side == Side::Left ? "triangular" : "vendor") << label(s);
        expect_trmm(r, "auto over budget " + label(s));
    }
}

// Every (side, uplo, trans, diag) on every candidate, ConjTrans with an imaginary alpha.
TYPED_TEST(TrmmCandidates, EveryCombinationOnEveryCandidate) {
    using T = typename TestFixture::T;
    for (const C& c : tm::candidates<T>())
        for (int n : {13, 70})
            for (Side side : {Side::Left, Side::Right})
                for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
                    for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
                        for (Diag dg : {Diag::NonUnit, Diag::Unit}) {
                            Spec s{side, uplo, t, dg, n, 9, 3};
                            s.seed = 77u + n;
                            if (!this->expect_runs(c, s)) continue;
                            auto p = make_prob<T>(s);
                            this->run_pinned(c, p);
                            expect_trmm(p, name(c, s));
                            if (::testing::Test::HasFatalFailure()) return;
                        }
}

// The pinned facade runs exactly that family's launch: bit-identical to the direct call.
TYPED_TEST(TrmmCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : tm::candidates<T>())
        for (Side side : {Side::Left, Side::Right})
            for (auto [n, q] : {std::pair{13, 7}, std::pair{40, 3}, std::pair{100, 33}}) {
                Spec s{side, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, n, q, 3};
                s.seed = 4242u + n;
                if (!this->expect_runs(c, s)) continue;
                auto pinned = make_prob<T>(s);
                auto direct = make_prob<T>(s);
                const std::string what = name(c, s);
                this->run_pinned(c, pinned);
                std::string why;
                ASSERT_TRUE(this->direct(c, direct, &why)) << what << ": " << why;
                expect_trmm(pinned, what);
                for (std::size_t e = 0; e < pinned.mem.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.mem[e], direct.mem[e]))
                        << what << ": the pinned facade did not run this family's launch; element " << e;
                ++compared;
            }
    EXPECT_GT(compared, 0);
}

// The SLM kernels at a saturating batch: 1024 items repeating 7 distinct problems, every row
// tile of the tile kernel and the expansion's group shapes.
TYPED_TEST(TrmmCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    constexpr int kBatch = 1024;
    struct Case { C c; Side side; int n, q; };
    const Case cases[] = {{tm::Triangular{}, Side::Left, 16, 40},  {tm::Triangular{}, Side::Left, 48, 9},
                          {tm::Triangular{}, Side::Left, 100, 130}, {tm::Triangular{}, Side::Left, 600, 3},
                          {tm::Expand{}, Side::Left, 33, 5},        {tm::Expand{}, Side::Right, 70, 12}};
    for (const auto& k : cases)
        for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
            Spec s{k.side, Uplo::Upper, t, Diag::NonUnit, k.n, k.q, kBatch};
            s.period = 7;
            s.seed = 777u + k.n;
            if (!this->expect_runs(k.c, s)) continue;
            auto p = make_prob<T>(s);
            this->run_pinned(k.c, p);
            expect_trmm(p, name(k.c, s));
            if (::testing::Test::HasFatalFailure()) return;
        }
}

// §8.2 (R3): on every straddling shape a pin is accepted exactly when the family's own launch runs.
TYPED_TEST(TrmmCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    int disagreements = 0;
    for (const C& c : tm::candidates<T>())
        for (Side side : {Side::Left, Side::Right})
            for (int n : {1, 2, 16, 17, 64, 65, 130})
                for (int q : {1, 9}) {
                    if (TestFixture::vendor_word_falls_back(c)) continue;
                    Spec s{side, Uplo::Lower, Transpose::Trans, Diag::NonUnit, n, q, 2};
                    s.seed = 61u + n;
                    auto a = make_prob<T>(s);
                    auto b = make_prob<T>(s);
                    const bool pin = this->pin_accepted(c, a);
                    std::string why;
                    const bool run = this->direct(c, b, &why);
                    EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the launch "
                                        << (run ? "runs" : "refuses: " + why);
                    EXPECT_EQ(pin, this->expect_runs(c, s)) << name(c, s) << ": can_run disagrees with the oracle";
                    disagreements += pin != run;
                }
    EXPECT_EQ(disagreements, 0);
}

// Both native launches put the batch in grid z: a pin launches at 65535 and is refused at 65536,
// where the direct launch throws; Auto then takes the vendor (vendor-free: no route).
TYPED_TEST(TrmmCandidates, GridBatchCeiling) {
    using T = typename TestFixture::T;
    const int ceiling = int(tm::kMaxGridBatch);
    for (const C& c : {C{tm::Triangular{}}, C{tm::Expand{}}}) {
        Spec at{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 2, 2, ceiling};
        auto p = make_prob<T>(at);
        ASSERT_TRUE(this->pin_accepted(c, p)) << name(c, at);
        expect_trmm(p, name(c, at));
        Spec past = at;
        past.batch = ceiling + 1;
        auto r = make_prob<T>(past);
        EXPECT_FALSE(this->pin_accepted(c, r)) << name(c, past);
        auto d = make_prob<T>(past);
        std::string why;
        EXPECT_FALSE(this->direct(c, d, &why)) << name(c, past) << ": the launch ran past the grid ceiling";
    }
    Spec past{Side::Right, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 2, 2, ceiling + 1};
    auto p = make_prob<T>(past);
    const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
    if constexpr (TestFixture::kVendor) {
        EXPECT_EQ(traced_choice([&] { this->run(p); }), "vendor");
        expect_trmm(p, "auto past the grid ceiling");
    } else {
        EXPECT_THROW(this->run(p), batchlas::NoRouteError);
    }
}

// The tile-grid term (AGENTS §8.9): triangular puts ceil(order/tile_m) x ceil(q/128) work-groups
// in grid y, capped at 65535. float's row tile is 32 at order 16 (one row tile) and 64 at order 65
// (two), so the last q that fits is 65535*128 and 32767*128. There the pin launches and writes C's
// last column; one column past it the direct launch throws, the pin is refused and writes nothing,
// and Auto takes expand. B and C live on the device (~2.2 GB at order 65).
TYPED_TEST(TrmmCandidates, TriangularTileGridCeiling) {
    static constexpr Backend B = TestFixture::B;
    if constexpr (!std::is_same_v<typename TestFixture::T, float>) {
        GTEST_SKIP() << "one dtype exercises the term; each launch costs GBs";
    } else {
        const ScopedEnvVar tile("BATCHLAS_TRMM_TILE_M", nullptr);
        auto& q = batchlas::sycl_queue(*this->ctx);
        for (const auto [n, row_tiles] : {std::pair{16, 1}, std::pair{65, 2}}) {
            const int last = int(65535 / row_tiles) * 128;
            const std::size_t need = 2 * std::size_t(n) * std::size_t(last + 1) * sizeof(float);
            if (q.get_device().template get_info<sycl::info::device::global_mem_size>() < need + (std::size_t(4) << 30))
                GTEST_SKIP() << "needs " << (need >> 30) << " GiB of device memory";
            UnifiedVector<float> a(std::size_t(n) * n, poison<float>());
            for (int j = 0; j < n; ++j)
                for (int i = j; i < n; ++i) a[std::size_t(j) * n + i] = float((i * 7 + j * 3) % 11 - 5) / 4;
            auto want = [&](int i) {  // B is all 1/4: multiples of 1/16, exact in float
                float s = 0;
                for (int t = 0; t <= i; ++t) s += a[std::size_t(t) * n + i] * 0.25f;
                return s;
            };
            for (const int cols : {last, last + 1}) {
                float* b = sycl::malloc_device<float>(std::size_t(n) * cols, q);
                float* c = sycl::malloc_device<float>(std::size_t(n) * cols, q);
                ASSERT_TRUE(b != nullptr && c != nullptr);
                q.fill(b, 0.25f, std::size_t(n) * cols).wait();
                q.fill(c, poison<float>(), std::size_t(n) * cols).wait();
                UnifiedVector<float*> ap(1, nullptr), bp(1, nullptr), cp(1, nullptr);
                const MVof<float> A(a.data(), n, n, n, n * n, 1, ap.data());
                const MVof<float> Bm(b, n, cols, n, n * cols, 1, bp.data());
                const MVof<float> Cm(c, n, cols, n, n * cols, 1, cp.data());
                auto at = [&](int i, int j) {
                    float v = 0;
                    q.memcpy(&v, c + std::size_t(j) * n + i, sizeof v).wait();
                    return v;
                };
                auto call = [&] {
                    (void)trmm<B, float>(*this->ctx, A, Bm, Cm, 1.0f, Side::Left, Uplo::Lower, Transpose::NoTrans,
                                         Diag::NonUnit);
                    this->ctx->wait();
                };
                const std::string what = "order=" + std::to_string(n) + " q=" + std::to_string(cols);
                bool accepted = true;
                try {
                    const Pin pin("trmm", C{tm::Triangular{}});
                    call();
                } catch (const std::invalid_argument& e) {
                    EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
                    accepted = false;
                }
                if (cols == last) {
                    EXPECT_TRUE(accepted) << what;
                    for (const auto& [i, j] : std::initializer_list<std::pair<int, int>>{
                             {n - 1, cols - 1}, {0, cols - 1}, {n - 1, 0}, {n / 2, cols - 129}})
                        EXPECT_EQ(at(i, j), want(i)) << what << " C(" << i << "," << j << ")";
                } else {
                    EXPECT_FALSE(accepted) << what;
                    EXPECT_EQ(at(n - 1, cols - 1), poison<float>()) << what << ": a refused pin wrote";
                    try {
                        (void)backend::detail::trmm_triangular_tiles<float>(*this->ctx, A, Bm, Cm, 1.0f, Uplo::Lower,
                                                                            Transpose::NoTrans, Diag::NonUnit);
                        this->ctx->wait();
                        ADD_FAILURE() << "the direct launch past the ceiling ran, " << what;
                    } catch (const std::exception& e) {
                        EXPECT_NE(std::string(e.what()).find("exceed limit"), std::string::npos) << e.what();
                    }
                    const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
                    EXPECT_EQ(traced_choice(call), "expand") << what;
                    EXPECT_EQ(at(n - 1, cols - 1), want(n - 1)) << what;
                    EXPECT_EQ(at(0, 0), want(0)) << what;
                }
                sycl::free(b, q);
                sycl::free(c, q);
            }
        }
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto: the removed
// level-3 words (cublasdx included), the other ops' families, and malformed spellings.
TYPED_TEST(TrmmCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"cublasdx", "gram", "tiles", "narrow", "gemm", "custom", "dx", "fused", "diag_full_gemm",
                             "triangular_tiles", "gram_tiles", "expand_gemm", "fused_device", "register_tiled",
                             "native:auto", "vendor:auto", "bogus", "triangular:1", "expand:tile=64", "blocked",
                             "cta", "tiled", "native:triangular", "loop"}) {
        auto p = make_prob<T>(Spec{});
        const Pin pin("trmm", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
        for (std::size_t e = 0; e < p.mem.size(); ++e)
            ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << word << ": a refused pin wrote element " << e;
    }
}

// The named can_run-false cases, each with its message: triangular on Side::Right (the tile
// kernel puts the triangle on the row index only), expand over budget.
TYPED_TEST(TrmmCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    for (int n : {1, 8, 200}) {
        Spec s{Side::Right, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, n, 4, 2};
        auto p = make_prob<T>(s);
        const Pin pin("trmm", C{tm::Triangular{}});
        try {
            this->run(p);
            ADD_FAILURE() << label(s) << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
        }
    }
    const ScopedEnvVar cap("BATCHLAS_EXPAND_MAX_BYTES", "0");
    Spec s{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 8, 4, 2};
    auto p = make_prob<T>(s);
    const Pin pin("trmm", C{tm::Expand{}});
    EXPECT_THROW(this->run(p), std::invalid_argument) << "expand with no scratch budget";
}

// A batch mismatch or a non-square A throws invalid_argument before choose(), under every pin.
TYPED_TEST(TrmmCandidates, InvalidShapesThrowBeforeChoose) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    auto p = make_prob<T>(Spec{});
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : tm::candidates<T>()) pins.push_back(c);
    const MVof<T> A2(p.mem.data() + p.a.off, 8, 8, p.a.ld, p.a.stride, 2);
    const MVof<T> Ar(p.mem.data() + p.a.off, 8, 7, p.a.ld, p.a.stride, 3);
    for (const auto& c : pins) {
        std::optional<Pin> pin;
        if (c) pin.emplace("trmm", *c);
        const std::string what = c ? select::to_string(*c) : std::string("auto");
        EXPECT_THROW(((void)trmm<B, T>(*this->ctx, A2, p.B(), p.Cm(), T(1), Side::Left, Uplo::Lower,
                                       Transpose::NoTrans, Diag::NonUnit)), std::invalid_argument) << what;
        EXPECT_THROW(((void)trmm<B, T>(*this->ctx, Ar, p.B(), p.Cm(), T(1), Side::Left, Uplo::Lower,
                                       Transpose::NoTrans, Diag::NonUnit)), std::invalid_argument) << what;
        EXPECT_THROW(((void)trmm<B, T>(*this->ctx, p.A(), p.B(), p.Cm(), T(1), Side::Right, Uplo::Lower,
                                       Transpose::NoTrans, Diag::NonUnit)), std::invalid_argument) << what;
    }
    this->ctx->wait();
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "a refused call wrote element " << e;
}

// An empty batch, order or rhs count is a no-op under Auto and every pin, vendor-free too
// (the old native launches threw on an empty batch, and the cuBLAS loop faults on one).
TYPED_TEST(TrmmCandidates, EmptyProblemIsANoOp) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    auto p = make_prob<T>(Spec{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 8, 4, 2});
    struct Case { int n, q, batch; Side side; };
    const Case cases[] = {{8, 4, 0, Side::Left}, {8, 4, 0, Side::Right}, {8, 0, 2, Side::Left}, {0, 4, 2, Side::Right}};
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : tm::candidates<T>()) pins.push_back(c);
    const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
    for (const auto& k : cases) {
        const int br = k.side == Side::Left ? k.n : k.q, bc = k.side == Side::Left ? k.q : k.n;
        const MVof<T> A(p.mem.data() + p.a.off, k.n, k.n, p.a.ld, p.a.stride, k.batch);
        const MVof<T> Bm(p.mem.data() + p.b.off, br, bc, std::max(br, 1), p.b.stride, k.batch);
        const MVof<T> Cm(p.mem.data() + p.c.off, br, bc, std::max(br, 1), p.c.stride, k.batch);
        for (const auto& c : pins) {
            std::optional<Pin> pin;
            if (c) pin.emplace("trmm", *c);
            EXPECT_NO_THROW(((void)trmm<B, T>(*this->ctx, A, Bm, Cm, p.alpha, k.side, Uplo::Lower, Transpose::NoTrans,
                                              Diag::NonUnit), this->ctx->wait()))
                << (c ? select::to_string(*c) : std::string("auto")) << " n=" << k.n << " q=" << k.q
                << " batch=" << k.batch;
        }
    }
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "an empty problem wrote element " << e;
}

// A heterogeneous operand on CUDA (expect_heterogeneous_has_no_route).
TYPED_TEST(TrmmCandidates, HeterogeneousBatchHasNoRoute) {
    expect_heterogeneous_has_no_route<TestFixture::B, typename TestFixture::T>(*this->ctx);
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment.
TYPED_TEST(TrmmCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
    for (Side side : {Side::Left, Side::Right}) {
        const Spec s{side, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 24, 5, 4};
        const std::string auto_pick = this->auto_choice(s);
        const std::string native_pick = side == Side::Left ? "triangular" : "expand";
        const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
        std::vector<std::pair<const char*, std::string>> expect{
            {"Expand", "expand"}, {" EXPAND ", "expand"}, {"vendor", vendor_pick}, {"native", native_pick},
            {"auto", auto_pick}};
        if (side == Side::Left) expect.push_back({"TRIANGULAR", "triangular"});
        for (const auto& [word, spelling] : expect)
            for (bool via_env : {false, true}) {
                auto p = make_prob<T>(s);
                select::testing::reset_warnings();
                std::string err;
                const std::string got = traced_choice([&] {
                    const ScopedEnvVar env("BATCHLAS_TRMM_ROUTE", via_env ? word : nullptr);
                    std::optional<Pin> pin;
                    if (!via_env) pin.emplace("trmm", std::string_view(word));
                    this->run(p);
                }, &err);
                const std::string what = std::string(word) + (via_env ? " via BATCHLAS_TRMM_ROUTE" : " via ScopedPin") +
                                         " " + label(s);
                EXPECT_EQ(got, spelling) << what;
                const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
                EXPECT_EQ(err.find("trmm pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                    << what << ": " << err;
                expect_trmm(p, what);
            }
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_TRMM_ROUTE, and nested pins restore the outer one.
TYPED_TEST(TrmmCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const Spec right{Side::Right, Uplo::Upper, Transpose::NoTrans, Diag::Unit, 20, 5, 2};
    const ScopedEnvVar env("BATCHLAS_TRMM_ROUTE", "triangular");
    {
        auto p = make_prob<T>(right);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_prob<T>(right);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("trmm", C{tm::Expand{}});
                  this->run(p);
              }), "expand");
    expect_trmm(p, "expand over env triangular");
    const Spec left{Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::Unit, 20, 5, 2};
    const Pin outer("trmm", C{tm::Triangular{}});
    {
        const Pin inner("trmm", C{tm::Expand{}});
        auto r = make_prob<T>(left);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "expand");
    }
    auto r = make_prob<T>(left);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "triangular") << "the inner pin did not restore the outer";
    expect_trmm(r, "outer triangular");
}

// Auto against the transcribed tables: triangular on Side::Left, expand on Side::Right, at grid
// and off-grid cells and in both builds (each row ranks a native family first).
TYPED_TEST(TrmmCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
    const auto tables = select::tables_in_borrow_order("trmm", select::dtype_name<T>(), select::device_of<B>(*this->ctx));
    if (tables.empty() || tables.front()->source != "transcribed:ff340fc6")
        GTEST_SKIP() << "this device reads a measured trmm table";
    struct Row { int n, q, batch; };
    const Row rows[] = {{1, 1, 1}, {16, 16, 128}, {23, 300, 7}, {128, 1, 1024}, {200, 70, 2}, {700, 3, 1}};
    for (const Row& r : rows)
        for (Side side : {Side::Left, Side::Right})
            for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
                Spec s{side, Uplo::Lower, t, Diag::NonUnit, r.n, r.q, r.batch};
                s.seed = 41u + r.n;
                auto p = make_prob<T>(s);
                EXPECT_EQ(traced_choice([&] { this->run(p); }), side == Side::Left ? "triangular" : "expand")
                    << label(s);
                expect_trmm(p, "auto " + label(s));
            }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with side,
// order, q on each side, and batch alone. A fixed field in key_of, or q from the wrong extent
// of B, turns exactly its probe(s) red: side=L sends the Right probes to L rows, whose first
// Right-runnable entry is vendor, so only the base Right probe moves; side=R sends the Left
// probes to R rows, which lead with triangular, so only the base Left probe moves. Vendor-free,
// expand is the only family that runs on Right, so the Right probes are blind there; side and
// q on Right are then covered through the trace key by TraceKeyQFollowsSide.
TYPED_TEST(TrmmCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("trmm.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("trmm." + dtype + "." + dev + ".txt",
                       "# op=trmm dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: side:exact order:log:2 q:log batch:log\n"
                       "side=L order=8 q=8 batch=8 | vendor 1 | expand 2\n"
                       "side=L order=8 q=8 batch=4096 | triangular 1 | expand 2\n"
                       "side=L order=8 q=512 batch=8 | triangular 1 | vendor 2 | expand 3\n"
                       "side=L order=128 q=8 batch=8 | triangular 1 | vendor 2 | expand 3\n"
                       "side=R order=8 q=8 batch=8 | triangular 1 | expand 2\n"
                       "side=R order=8 q=512 batch=8 | triangular 1 | vendor 2 | expand 3\n"
                       "side=R order=128 q=8 batch=8 | triangular 1 | vendor 2 | expand 3\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    const char* v = TestFixture::kVendor ? "vendor" : "expand";
    struct Probe { Side side; int n, q, batch; const char* expect; const char* field; };
    const Probe probes[] = {{Side::Left, 8, 8, 8, v, "side (Left)"},
                            {Side::Left, 8, 8, 4096, "triangular", "batch"},
                            {Side::Left, 8, 512, 8, "triangular", "q (Left: B.cols)"},
                            {Side::Left, 128, 8, 8, "triangular", "order (Left)"},
                            {Side::Right, 8, 8, 8, "expand", "side (Right)"},
                            {Side::Right, 8, 512, 8, v, "q (Right: B.rows)"},
                            {Side::Right, 128, 8, 8, v, "order (Right: A.rows)"}};
    for (const auto& k : probes) {
        Spec s{k.side, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, k.n, k.q, k.batch};
        s.seed = 43u;
        auto p = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row";
        expect_trmm(p, std::string("the ") + k.field + " row");
    }
}

// The trace key: order is A's order and q is B.cols on Side::Left, B.rows on Side::Right.
TYPED_TEST(TrmmCandidates, TraceKeyQFollowsSide) {
    using T = typename TestFixture::T;
    for (Side side : {Side::Left, Side::Right}) {
        Spec s{side, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit, 12, 37, 5};
        auto p = make_prob<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        const std::string want = std::string("side=") + side_s(side) + " order=12 q=37 batch=5 ->";
        EXPECT_NE(line.find(want), std::string::npos) << line;
        expect_trmm(p, line);
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, expand, on
// both sides (expand leads last_resort: it serves both); a vendor build takes the vendor.
TYPED_TEST(TrmmCandidates, VendorOnlyTableFallsToTheLastResort) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("trmm.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("trmm." + dtype + "." + dev + ".txt",
                       "# op=trmm dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: side:exact order:log:2 q:log batch:log\n"
                       "side=L order=32 q=32 batch=128 | vendor 1\n"
                       "side=R order=32 q=32 batch=128 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    for (Side side : {Side::Left, Side::Right}) {
        Spec s{side, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 32, 9, 3};
        auto p = make_prob<T>(s);
        std::string all;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), TestFixture::kVendor ? "vendor" : "expand") << label(s);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_trmm(p, "vendor-only table " + label(s));
    }
}

// The coverage row (§5.6): the real backend and scalar (the old recorder hard-coded CUDA/F32),
// the old key (m = C.rows, n = C.cols, k = A's order, uplo, side, diag, transA), native flags.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(TrmmCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "trmm_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 16, 4, 2};
    const Spec hi{Side::Right, Uplo::Upper, Transpose::ConjTrans, Diag::Unit, 48, 3, 2};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        auto p = make_prob<T>(lo);
        this->run(p);
        auto r = make_prob<T>(hi);
        this->run(r);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,trmm,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string klo = "16 4 16", khi = "3 48 48";
    ASSERT_TRUE(rows.count(klo)) << "no Left 16x4 row";
    ASSERT_TRUE(rows.count(khi)) << "no Right 3x48 row";
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    for (const auto& [key, s, algo] : {std::tuple{klo, lo, "triangular"}, std::tuple{khi, hi, "expand"}}) {
        const auto& f = rows[key];
        EXPECT_EQ(f[3], "CUDA") << key << ": the backend column";
        EXPECT_EQ(f[2], std::string(batchlas::to_string(batchlas::scalar_kind_of<T>))) << key << ": the scalar column";
        EXPECT_EQ(f[9], "native") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[15], as_int(s.uplo)) << key;
        EXPECT_EQ(f[16], as_int(s.side)) << key;
        EXPECT_EQ(f[17], as_int(s.diag)) << key;
        EXPECT_EQ(f[18], as_int(s.trans)) << key;
    }
}

// On a CPU queue (NETLIB) no native family can run: native pins throw, Auto is the vendor.
template <typename Config>
class TrmmCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(TrmmCandidatesCpu, Types);

TYPED_TEST(TrmmCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_TRMM_ROUTE", nullptr);
    Spec s{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 8, 3, 2};
    for (const C& c : tm::candidates<T>()) {
        if (std::holds_alternative<tm::Vendor>(c)) continue;
        auto p = make_prob<T>(s);
        const Pin pin("trmm", c);
        EXPECT_THROW(((void)trmm<B, T>(*this->ctx, p.A(), p.B(), p.Cm(), p.alpha, s.side, s.uplo, s.trans, s.diag)),
                     std::invalid_argument)
            << select::to_string(c);
    }
    auto p = make_prob<T>(s);
    auto call = [&] {
        (void)trmm<B, T>(*this->ctx, p.A(), p.B(), p.Cm(), p.alpha, s.side, s.uplo, s.trans, s.diag);
        this->ctx->wait();
    };
    if constexpr (batchlas::select::level3_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        expect_trmm(p, "cpu vendor");
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

// The netlib loop shares the vendor loops' single top-level (m, n), so a heterogeneous batch has
// no route on a CPU queue either.
TYPED_TEST(TrmmCandidatesCpu, HeterogeneousBatchHasNoRoute) {
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    expect_heterogeneous_has_no_route<TypeParam::BackendVal, typename TypeParam::ScalarType>(*this->ctx);
}

// The transcribed tables (no GPU): every dtype on sm_89 and sm_120 holds exactly choice.hh's
// grid with identical rows, names the transcribed commit and choice.hh's keys, and reads
// `triangular | expand | vendor` on Side::Left and `expand | vendor` on Side::Right.
TEST(TrmmTranscribedTable, RowsHoldTheChoiceGridAndTheOldPreference) {
    std::set<std::string> want;
    for (const char* s : {"L", "R"})
        for (int o : tm::grid_order)
            for (int q : tm::grid_q)
                for (int b : tm::grid_batch)
                    want.insert(std::string(s) + " " + std::to_string(o) + " " + std::to_string(q) + " " +
                                std::to_string(b));
    std::string keys = "# keys:";
    for (auto k : tm::key_names) keys += " " + std::string(k);
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        std::map<std::string, std::string> first;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const std::string file = std::string("trmm.") + dt + "." + dev + ".txt";
            const auto& all = select::embedded_tables();
            const auto e = std::find_if(all.begin(), all.end(), [&](const auto& t) { return t.name == file; });
            ASSERT_NE(e, all.end()) << file;
            EXPECT_NE(std::string(e->text).find("\n" + keys + "\n"), std::string::npos) << file;
            const select::Table t = select::parse_table(e->text, e->name);
            EXPECT_EQ(t.source, "transcribed:ff340fc6") << file;
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                const std::string k = row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3];
                got.insert(k);
                std::string ranked;
                for (const auto& x : row.ranked) ranked += x.spelling + "|";
                EXPECT_EQ(ranked, row.keys[0] == "L" ? "triangular|expand|vendor|" : "expand|vendor|") << file << " " << k;
                EXPECT_FALSE(row.timed) << file;
                if (std::string(dev) == "sm_89") first[k] = ranked;
                else EXPECT_EQ(ranked, first[k]) << file << " " << k << ": sm_120 differs from sm_89";
            }
            EXPECT_EQ(got, want) << file;
            EXPECT_EQ(t.rows.size(), want.size()) << file;
            for (const char* side : {"L", "R"}) {
                const select::Key key{{"side", side}, {"order", 300}, {"q", 7}, {"batch", 50000}};
                const select::TableRow* row = t.nearest(key);
                ASSERT_NE(row, nullptr) << file;
                EXPECT_EQ(row->keys[0], side) << file;
                EXPECT_EQ(row->ranked.front().spelling, std::string(side) == "L" ? "triangular" : "expand") << file;
            }
        }
    }
}

}  // namespace
