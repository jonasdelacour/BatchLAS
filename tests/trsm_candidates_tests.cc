// Every trsm candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// flat-kernel-selection-phase3-plan.md §1.2. The limit oracle reads the drivers' own capacity
// constants; which kernel ran is read back from the select trace or a bit-for-bit comparison
// with the direct driver, never assumed from the pin being accepted.
#include <gtest/gtest.h>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trsm.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/ops/trsm/choice.hh"
#include "../src/sycl/trsm_native.hh"

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace ts = batchlas::ops::trsm;
using C = ts::TrsmChoice;
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

// One operand's footprint inside the shared buffer: item `it` starts at off + it * stride.
struct Region {
    std::size_t off = 0;
    int rows = 0, cols = 0, ld = 0, stride = 0;
};

struct Spec {
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
    Diag diag = Diag::NonUnit;
    int n = 8, q = 4, batch = 3;
    int period = 0;  // > 0: item it repeats item it % period
    unsigned seed = 1;
};

// A and B live in one buffer of a large finite poison, at padded lds and strides that are not
// ld * cols. Everything a solve may not read or write -- A's other triangle, a Unit diagonal,
// the ld pads, the gaps, a parent matrix around sub-views -- holds the poison.
template <typename T>
struct Solve {
    Spec s;
    T alpha{};
    Region a, b;
    UnifiedVector<T> mem;
    std::vector<T> mem0;
    UnifiedVector<T*> aptr, bptr;

    MatrixView<T, MatrixFormat::Dense> A() {
        return {mem.data() + a.off, s.n, s.n, a.ld, a.stride, s.batch, aptr.data()};
    }
    MatrixView<T, MatrixFormat::Dense> B() {
        return {mem.data() + b.off, b.rows, b.cols, b.ld, b.stride, s.batch, bptr.data()};
    }
    std::size_t ai(int it, int i, int j) const { return a.off + std::size_t(it) * a.stride + std::size_t(j) * a.ld + i; }
    std::size_t bi(int it, int i, int j) const { return b.off + std::size_t(it) * b.stride + std::size_t(j) * b.ld + i; }
    bool owned(int i, int j) const { return s.uplo == Uplo::Lower ? i > j : i < j; }
};

template <typename T>
T poison() {
    return mk<T>(RealOf<T>(-999), RealOf<T>(777));
}

// Strictly diagonally dominant owned triangle, every entry with a nonzero imaginary part; a
// Unit diagonal stores poison (a kernel that reads it divides by ~1e3).
template <typename T>
void fill(Solve<T>& p) {
    using R = RealOf<T>;
    const Spec& s = p.s;
    std::mt19937 gen(p.s.seed);
    std::uniform_real_distribution<R> d(R(0.1), R(1)), u(R(-1), R(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < reps; ++it) {
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i) {
                if (i == j)
                    p.mem[p.ai(it, i, j)] = s.diag == Diag::Unit ? mk<T>(R(300), R(-200))
                                                                 : mk<T>(R(2.5) + R(0.5) * d(gen), R(0.3) * d(gen));
                else if (p.owned(i, j))
                    p.mem[p.ai(it, i, j)] = mk<T>((gen() & 1 ? d(gen) : -d(gen)) / R(s.n), d(gen) / R(s.n));
            }
        for (int j = 0; j < p.b.cols; ++j)
            for (int i = 0; i < p.b.rows; ++i) p.mem[p.bi(it, i, j)] = mk<T>(u(gen), u(gen));
    }
    for (int it = reps; it < s.batch; ++it) {
        const int r = it % reps;
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i)
                if (i == j || p.owned(i, j)) p.mem[p.ai(it, i, j)] = p.mem[p.ai(r, i, j)];
        for (int j = 0; j < p.b.cols; ++j)
            for (int i = 0; i < p.b.rows; ++i) p.mem[p.bi(it, i, j)] = p.mem[p.bi(r, i, j)];
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
}

template <typename T>
T alpha_of() {
    return mk<T>(RealOf<T>(1.5), RealOf<T>(-0.5));
}

template <typename T>
void init(Solve<T>& p, std::size_t total) {
    p.alpha = alpha_of<T>();
    p.mem = UnifiedVector<T>(total, poison<T>());
    p.aptr = UnifiedVector<T*>(p.s.batch, nullptr);
    p.bptr = UnifiedVector<T*>(p.s.batch, nullptr);
    fill(p);
}

// Standalone A and B with non-natural ld and stride on both.
template <typename T>
Solve<T> make_solve(const Spec& s) {
    Solve<T> p;
    p.s = s;
    p.a = {0, s.n, s.n, s.n + 3, (s.n + 3) * s.n + 5};
    const int br = s.side == Side::Left ? s.n : s.q, bc = s.side == Side::Left ? s.q : s.n;
    p.b = {std::size_t(p.a.stride) * s.batch + 11, br, bc, br + 2, (br + 2) * bc + 7};
    init(p, p.b.off + std::size_t(p.b.stride) * s.batch + 13);
    return p;
}

// The potrf panel pattern (potrf_blocked.cc): A11 and the block below (Side::Right) or to its
// right (Side::Left, getrf's L11 \ A12) are sub-views of one parent and carry its ld and stride.
template <typename T>
Solve<T> make_panel(const Spec& s, int parent, int j0) {
    Solve<T> p;
    p.s = s;
    const int ld = parent + 5, stride = ld * parent + 9;
    p.a = {std::size_t(j0) * ld + j0, s.n, s.n, ld, stride};
    if (s.side == Side::Right)
        p.b = {std::size_t(j0) * ld + j0 + s.n, s.q, s.n, ld, stride};
    else
        p.b = {std::size_t(j0 + s.n) * ld + j0, s.n, s.q, ld, stride};
    init(p, std::size_t(stride) * s.batch);
    return p;
}

// ||op(A) X - alpha B|| / (||A|| ||X|| + |alpha| ||B||) for one item through the library: A and B
// from the pristine copy, X from the result buffer, A's other triangle and a Unit diagonal unread.
template <typename T>
double residual(const Solve<T>& p, int it) {
    const Spec& s = p.s;
    using View = MatrixView<T, MatrixFormat::Dense>;
    const auto A0 = batchlas::verify::view(p.mem0.data() + p.a.off, s.n, s.n, p.a.ld, p.a.stride, s.batch);
    const auto B0 = batchlas::verify::view(p.mem0.data() + p.b.off, p.b.rows, p.b.cols, p.b.ld, p.b.stride, s.batch);
    const auto X = batchlas::verify::view(p.mem.data() + p.b.off, p.b.rows, p.b.cols, p.b.ld, p.b.stride, s.batch);
    const int one[] = {it};
    return batchlas::verify::trsm_residual(A0, s.side, s.uplo, s.trans, s.diag, X, B0, batchlas::verify::up(p.alpha), one);
}

// Residuals of the checked items (all of them for small batches; the representatives of a
// repeating batch), every element outside B's footprint bit for bit, and, for a repeating
// batch, every item's X bit-identical to its representative's.
template <typename T>
void expect_solved(const Solve<T>& p, const std::string& what) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0) {
        for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    } else if (s.batch <= 8) {
        for (int it = 0; it < s.batch; ++it) items.push_back(it);
    } else {
        items = {0, 1, s.batch / 2, s.batch - 1};
    }
    for (int it : items) {
        const double r = residual(p, it);
        ASSERT_TRUE(batchlas::verify::pass<T>(batchlas::verify::Check::solve, s.n, r))
            << what << " item " << it << " residual " << r << " exceeds "
            << batchlas::verify::bound<T>(batchlas::verify::Check::solve, s.n);
    }
    std::vector<char> inb(p.mem.size(), 0);
    for (int it = 0; it < s.batch; ++it)
        for (int j = 0; j < p.b.cols; ++j)
            for (int i = 0; i < p.b.rows; ++i) inb[p.bi(it, i, j)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!inb[e]) ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": wrote outside B at element " << e;
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int j = 0; j < p.b.cols; ++j)
                for (int i = 0; i < p.b.rows; ++i)
                    ASSERT_TRUE(same_bits(p.mem[p.bi(it, i, j)], p.mem[p.bi(it % s.period, i, j)]))
                        << what << ": X of item " << it << " differs from its representative at (" << i << "," << j
                        << ")";
}

std::string label(const Spec& s) {
    return std::string(side_s(s.side)) + uplo_s(s.uplo) + trans_s(s.trans) + diag_s(s.diag) +
           " order=" + std::to_string(s.n) + " q=" + std::to_string(s.q) + " batch=" + std::to_string(s.batch);
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// The outermost trsm trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("trsm ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no trsm trace line in: " + err + ">";
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

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class TrsmCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::level3_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native tiers are GPU kernels";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    // ---- the limit oracle: the drivers' own constants, not trsm.cc's can_run ----
    static bool expect_runs(const C& c, Side side, int n) {
        if (n < 1) return false;
        if (std::holds_alternative<ts::Cta>(c)) return n <= sycl_trsm::trsm_cta_max_n<T>();
        if (std::holds_alternative<ts::SgLeft>(c)) return side == Side::Left && n <= sycl_trsm::kTrsmSgLeftMaxN;
        if (std::holds_alternative<ts::Blocked>(c)) return sycl_trsm::trsm_blocked_available<T>();
        return kVendor;
    }

    // The Vendor choice spells the class word `vendor`, which falls back to Auto (with a warning)
    // where no vendor can run, instead of throwing (§5.3).
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<ts::Vendor>(c) && !kVendor; }
    std::string auto_choice(const Spec& s) {
        const ScopedEnvVar clear("BATCHLAS_TRSM_ROUTE", nullptr);
        auto p = make_solve<T>(s);
        return traced_choice([&] { run(p); });
    }

    void run(Solve<T>& p) {
        (void)trsm<B, T>(*this->ctx, p.A(), p.B(), p.alpha, p.s.side, p.s.uplo, p.s.trans, p.s.diag);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Solve<T>& p) {
        const Pin pin("trsm", c);
        run(p);
    }
    // trsm has no sizing call, so acceptance is a launch: `p` is consumed.
    bool pin_accepted(const C& c, Solve<T>& p) {
        const Pin pin("trsm", c);
        try {
            run(p);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }

    // The family's own driver: Cta = V1, SgLeft = the sub-group kernel (no side parameter, so it
    // cannot serve Side::Right at all), Blocked = V2 with the public gemm, Vendor = the library.
    bool direct(const C& c, Solve<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const MV A = p.A(), Bm = p.B();
        const Spec& s = p.s;
        try {
            if (std::holds_alternative<ts::Cta>(c)) {
                (void)sycl_trsm::trsm_native_v1_dispatch<T>(q, A, Bm, p.alpha, s.side, s.uplo, s.trans, s.diag);
            } else if (std::holds_alternative<ts::SgLeft>(c)) {
                if (s.side != Side::Left) {
                    if (why) *why = "the sub-group kernel has no Side::Right";
                    return false;
                }
                (void)sycl_trsm::trsm_native_sg_left_dispatch<T>(q, A, Bm, p.alpha, s.uplo, s.trans, s.diag);
            } else if (std::holds_alternative<ts::Blocked>(c)) {
                (void)sycl_trsm::trsm_native_blocked<T>(
                    q, A, Bm, p.alpha, s.side, s.uplo, s.trans, s.diag,
                    [](Queue& c2, const MV& ga, const MV& gb, const MV& gc, T galpha, T gbeta, Transpose gta,
                       Transpose gtb, ComputePrecision gp) {
                        return gemm<B, T>(c2, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                    });
            } else {
                if constexpr (kVendor) {
                    (void)backend::trsm_vendor<B, T>(q, A, Bm, s.side, s.uplo, s.trans, s.diag, p.alpha);
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

TYPED_TEST_SUITE(TrsmCandidates, Types);

// §8.1: each candidate on orders straddling the V1 buckets (8/9, 16/17), the sub-group
// kernel's (4/5) and both order-32 ceilings, both sides, rhs counts across sg_left's chunks.
// Accepted shapes run and are correct; refused ones throw from the pin.
TYPED_TEST(TrsmCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    const int cap = sycl_trsm::trsm_cta_max_n<T>();
    ASSERT_EQ(cap, sycl_trsm::kTrsmSgLeftMaxN) << "the straddles below assume one order-32 ceiling";
    int ran = 0, refused = 0;
    for (const C& c : ts::candidates<T>())
        for (Side side : {Side::Left, Side::Right})
            for (int n : {1, 4, 5, 8, 9, 16, 17, cap, cap + 1, 100})
                for (int q : {1, 5, 17, 40}) {
                    Spec s{side, n % 2 ? Uplo::Lower : Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, n, q, 3};
                    s.seed = 1000u + 31u * n + q;
                    auto p = make_solve<T>(s);
                    const std::string what = name(c, s);
                    if (TestFixture::vendor_word_falls_back(c)) {
                        const std::string want = this->auto_choice(s);
                        EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), want) << what;
                        expect_solved(p, what + " (vendor-free: Auto)");
                        continue;
                    }
                    if (!this->expect_runs(c, side, n)) {
                        EXPECT_FALSE(this->pin_accepted(c, p)) << what << " was accepted";
                        ++refused;
                        continue;
                    }
                    ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
                    expect_solved(p, what);
                    ++ran;
                }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// Every (side, uplo, trans, diag), which covers the 7 combinations issued inside the library:
// potrf's panel R L C N; posv's L L N N, L L C N, L U C N, L U N N; getrs/getri/getrf's L L N U;
// getrs-transposed L U T N (and L L T U). ConjTrans on complex data with an imaginary alpha.
TYPED_TEST(TrsmCandidates, EveryCombinationOnEveryCandidate) {
    using T = typename TestFixture::T;
    for (const C& c : ts::candidates<T>())
        for (int n : {13, 32, 70})
            for (Side side : {Side::Left, Side::Right})
                for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
                    for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
                        for (Diag dg : {Diag::NonUnit, Diag::Unit}) {
                            if (!this->expect_runs(c, side, n)) continue;
                            Spec s{side, uplo, t, dg, n, 7, 3};
                            s.seed = 77u + n;
                            auto p = make_solve<T>(s);
                            this->run_pinned(c, p);
                            expect_solved(p, name(c, s));
                            if (::testing::Test::HasFatalFailure()) return;
                        }
}

// Sub-views carrying the parent's ld and stride: A11 and the panel beside it, as potrf
// (R L C N) and getrf (L L N U) issue them. Everything else in the parent stays bit for bit.
TYPED_TEST(TrsmCandidates, ParentLdSubViews) {
    using T = typename TestFixture::T;
    struct Case { Side side; Uplo uplo; Transpose t; Diag dg; int n, q; };
    const Case cases[] = {{Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, 32, 61},
                          {Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, 16, 3},
                          {Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, 48, 40},
                          {Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit, 32, 61},
                          {Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit, 9, 5},
                          {Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit, 48, 40}};
    int ran = 0;
    for (const C& c : ts::candidates<T>())
        for (const auto& k : cases) {
            if (!this->expect_runs(c, k.side, k.n)) continue;
            Spec s{k.side, k.uplo, k.t, k.dg, k.n, k.q, 3};
            s.seed = 300u + k.n;
            auto p = make_panel<T>(s, 8 + k.n + k.q, 8);
            this->run_pinned(c, p);
            expect_solved(p, name(c, s) + " (parent-ld sub-view)");
            ++ran;
        }
    EXPECT_GT(ran, 0);
}

// ortho's Right-side solves (order 2-3, q 12, batch 2), which left the vendor in P3.3; the
// cdouble ortho suite crashes before reaching them (known-defects #13), so they are pinned here.
TYPED_TEST(TrsmCandidates, OrthoCallerShapes) {
    using T = typename TestFixture::T;
    int ran = 0;
    for (const C& c : ts::candidates<T>())
        for (Uplo u : {Uplo::Lower, Uplo::Upper})
            for (Transpose t : {Transpose::Trans, Transpose::ConjTrans})
                for (int n : {2, 3}) {
                    if (!this->expect_runs(c, Side::Right, n)) continue;
                    Spec s{Side::Right, u, t, Diag::NonUnit, n, 12, 2};
                    s.seed = 500u + n;
                    auto p = make_solve<T>(s);
                    this->run_pinned(c, p);
                    expect_solved(p, name(c, s) + " (ortho shape)");
                    ++ran;
                }
    EXPECT_GT(ran, 0);
}

// The pinned facade runs exactly that family's driver: bit-identical to the direct call on the
// same input. A swapped launch arm shows here even where both kernels are correct.
TYPED_TEST(TrsmCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : ts::candidates<T>())
        for (Side side : {Side::Left, Side::Right})
            for (auto [n, q] : {std::pair{13, 7}, std::pair{29, 3}, std::pair{32, 33}, std::pair{75, 9}}) {
                if (!this->expect_runs(c, side, n)) continue;
                Spec s{side, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, n, q, 3};
                s.seed = 4242u + n;
                auto pinned = make_solve<T>(s);
                auto direct = make_solve<T>(s);
                const std::string what = name(c, s);
                this->run_pinned(c, pinned);
                std::string why;
                ASSERT_TRUE(this->direct(c, direct, &why)) << what << ": " << why;
                expect_solved(pinned, what);
                for (std::size_t e = 0; e < pinned.mem.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.mem[e], direct.mem[e]))
                        << what << ": the pinned facade did not run this family's driver; element " << e;
                ++compared;
            }
    EXPECT_GT(compared, 0);
}

// SLM tiers at a saturating batch: 1024 items repeating 7 distinct systems; each representative
// is correct and every other item is bit-identical to its representative.
TYPED_TEST(TrsmCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    constexpr int kBatch = 1024;
    struct Case { C c; Side side; int n, q; };
    const Case cases[] = {{ts::Cta{}, Side::Left, 17, 5},     {ts::Cta{}, Side::Right, 32, 40},
                          {ts::Cta{}, Side::Left, 32, 300},   {ts::SgLeft{}, Side::Left, 16, 4},
                          {ts::SgLeft{}, Side::Left, 32, 17}, {ts::Blocked{}, Side::Left, 70, 9},
                          {ts::Blocked{}, Side::Right, 70, 9}, {ts::Vendor{}, Side::Left, 24, 6}};
    for (const auto& k : cases) {
        if (!this->expect_runs(k.c, k.side, k.n)) continue;
        for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
            Spec s{k.side, Uplo::Upper, t, Diag::NonUnit, k.n, k.q, kBatch};
            s.period = 7;
            s.seed = 777u + k.n;
            auto p = make_solve<T>(s);
            this->run_pinned(k.c, p);
            expect_solved(p, name(k.c, s));
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// §8.2 (R3): on every straddling shape a pin is accepted exactly when the family's own driver
// launches. sg_left's driver takes no side, so it "refuses" Side::Right by construction.
TYPED_TEST(TrsmCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    int disagreements = 0;
    for (const C& c : ts::candidates<T>())
        for (Side side : {Side::Left, Side::Right})
            for (int n : {1, 4, 5, 8, 9, 16, 17, 32, 33, 48, 100})
                for (int q : {1, 9}) {
                    Spec s{side, Uplo::Lower, Transpose::Trans, Diag::NonUnit, n, q, 2};
                    s.seed = 61u + n;
                    auto a = make_solve<T>(s);
                    auto b = make_solve<T>(s);
                    if (TestFixture::vendor_word_falls_back(c)) continue;  // Auto, not the vendor
                    const bool pin = this->pin_accepted(c, a);
                    std::string why;
                    const bool run = this->direct(c, b, &why);
                    EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the driver "
                                        << (run ? "launches" : "refuses: " + why);
                    EXPECT_EQ(pin, this->expect_runs(c, side, n)) << name(c, s) << ": can_run disagrees with the oracle";
                    disagreements += pin != run;
                }
    EXPECT_EQ(disagreements, 0);
}

// Q6: A.batch != B.batch throws invalid_argument before choose(), under Auto and every pin.
// The buffers hold three items each, so a kernel that ignored the check stays in bounds.
TYPED_TEST(TrsmCandidates, BatchMismatchThrows) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    Spec s{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 8, 4, 3};
    auto p = make_solve<T>(s);
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : ts::candidates<T>()) pins.push_back(c);
    for (auto [ba, bb] : {std::pair{3, 2}, std::pair{2, 3}}) {
        const MVof<T> A(p.mem.data() + p.a.off, 8, 8, p.a.ld, p.a.stride, ba);
        const MVof<T> Bm(p.mem.data() + p.b.off, 8, 4, p.b.ld, p.b.stride, bb);
        for (const auto& c : pins) {
            std::optional<Pin> pin;
            if (c) pin.emplace("trsm", *c);
            const std::string what = (c ? select::to_string(*c) : std::string("auto")) + " A.batch=" +
                                     std::to_string(ba) + " B.batch=" + std::to_string(bb);
            try {
                (void)trsm<B, T>(*this->ctx, A, Bm, p.alpha, Side::Left, Uplo::Lower, Transpose::NoTrans,
                                 Diag::NonUnit);
                this->ctx->wait();
                ADD_FAILURE() << what << " was accepted";
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("same batch size"), std::string::npos) << what << ": " << e.what();
            }
        }
    }
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "a refused call wrote element " << e;
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(TrsmCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "cta:1", "blocked:outer=64", "native:sg_left", "sg-left", "native:vendor",
                             "native:cta:8", "v1",
                             // removed aliases (phase 5): each must stay an error
                             "native:cta", "native:blocked"}) {
        auto p = make_solve<T>(Spec{});
        const Pin pin("trsm", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
    for (const C& c : ts::candidates<T>()) {
        auto p = make_solve<T>(Spec{});
        if (!this->expect_runs(c, Side::Left, 8)) continue;
        EXPECT_NO_THROW(this->run_pinned(c, p)) << select::to_string(c);
    }
}

// The named can_run-false cases, each with its message (absorbs route_vocabulary_tests'
// RouteTrsm.CorrectnessGatesAreNotSpeedGates and AbsentKernelIsUnsupportedRatherThanSelectable:
// order 0, q 0, one past the CTA ceiling, sg_left on Side::Right).
TYPED_TEST(TrsmCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int cap = sycl_trsm::trsm_cta_max_n<T>();
    struct Case { C c; Side side; int n, q; };
    std::vector<Case> cases{{ts::Cta{}, Side::Left, cap + 1, 4},     {ts::Cta{}, Side::Right, cap + 1, 4},
                            {ts::SgLeft{}, Side::Left, sycl_trsm::kTrsmSgLeftMaxN + 1, 4},
                            {ts::SgLeft{}, Side::Right, 8, 4},       {ts::SgLeft{}, Side::Right, 1, 1},
                            {ts::Cta{}, Side::Left, 0, 4},           {ts::Blocked{}, Side::Left, 0, 4},
                            {ts::SgLeft{}, Side::Left, 0, 4},        {ts::Cta{}, Side::Left, 8, 0},
                            {ts::Blocked{}, Side::Right, 8, 0}};
    for (const auto& k : cases) {
        const int br = k.side == Side::Left ? k.n : k.q, bc = k.side == Side::Left ? k.q : k.n;
        UnifiedVector<T> a(std::max(1, k.n * k.n), T(1)), b(std::max(1, br * bc), T(1));
        const MVof<T> A(a.data(), k.n, k.n, std::max(1, k.n), std::max(1, k.n * k.n), 1);
        const MVof<T> Bm(b.data(), br, bc, std::max(1, br), std::max(1, br * bc), 1);
        const Spec s{k.side, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, k.n, k.q, 1};
        const Pin pin("trsm", k.c);
        try {
            (void)trsm<B, T>(*this->ctx, A, Bm, T(1), k.side, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
            this->ctx->wait();
            ADD_FAILURE() << name(k.c, s) << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                << name(k.c, s) << ": " << e.what();
        }
    }
}

// RouteTrsm.AbsentKernelIsUnsupportedRatherThanSelectable, ported: a build without the kernel
// reports cta_max 0 and blocked unavailable, which this build does not; what can be launched is
// that both natives exist here and that blocked serves what cta's ceiling refuses.
TYPED_TEST(TrsmCandidates, BlockedServesPastTheCtaCeiling) {
    using T = typename TestFixture::T;
    const int cap = sycl_trsm::trsm_cta_max_n<T>();
    ASSERT_GE(cap, 1) << "this build has no CTA kernel, so neither native can run";
    ASSERT_TRUE(sycl_trsm::trsm_blocked_available<T>());
    for (int n : {cap + 1, 2 * cap + 1}) {
        Spec s{Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, n, 3, 2};
        auto refused = make_solve<T>(s);
        EXPECT_FALSE(this->pin_accepted(C{ts::Cta{}}, refused)) << n;
        auto p = make_solve<T>(s);
        this->run_pinned(C{ts::Blocked{}}, p);
        expect_solved(p, name(C{ts::Blocked{}}, s));
    }
}

// A heterogeneous A or B: no native family can run it (one launch has one order), so every
// native pin throws. Auto takes the vendor, which runs at the full storage order (known-defects
// #12); vendor-free there is no route.
TYPED_TEST(TrsmCandidates, HeterogeneousBatchHasNoNativeRoute) {
    using T = typename TestFixture::T;
    using R = RealOf<T>;
    static constexpr Backend B = TestFixture::B;
    const int n = 16, q = 3, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch), Bm(n, q, batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) A(i, j, b) = mk<T>(R(i == j ? 4 : (i > j ? 0.1 : 0)), R(0));
    Bm.fill(mk<T>(R(1), R(0.5)));
    UnifiedVector<int> act(batch), cols(batch);
    for (int b = 0; b < batch; ++b) act[b] = n - b, cols[b] = q - (b % 2);
    const auto hetA = A.view().with_active_dims(act.to_span(), act.to_span());
    const auto hetB = Bm.view().with_active_dims(act.to_span(), cols.to_span());
    ASSERT_TRUE(hetA.is_heterogeneous());
    struct Case { const char* what; MVof<T> a, b; };
    const Case cases[] = {{"A heterogeneous", hetA, Bm.view()}, {"B heterogeneous", A.view(), hetB}};
    for (const auto& k : cases) {
        for (const C& c : ts::candidates<T>()) {
            if (std::holds_alternative<ts::Vendor>(c)) continue;
            const Pin pin("trsm", c);
            try {
                (void)trsm<B, T>(*this->ctx, k.a, k.b, T(1), Side::Left, Uplo::Lower, Transpose::NoTrans,
                                 Diag::NonUnit);
                this->ctx->wait();
                ADD_FAILURE() << k.what << " " << select::to_string(c) << " was accepted";
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                    << k.what << " " << select::to_string(c) << ": " << e.what();
            }
        }
        const ScopedEnvVar clear("BATCHLAS_TRSM_ROUTE", nullptr);
        auto call = [&] {
            (void)trsm<B, T>(*this->ctx, k.a, k.b, T(1), Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
            this->ctx->wait();
        };
        if constexpr (TestFixture::kVendor) EXPECT_EQ(traced_choice(call), "vendor") << k.what;
        else EXPECT_THROW(call(), batchlas::NoRouteError) << k.what;
    }
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment.
// Bare `native` is the row's best runnable non-vendor; bare `vendor` falls back to Auto
// (with the warning) where there is no vendor.
TYPED_TEST(TrsmCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_TRSM_ROUTE", nullptr);
    const Spec s16{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 16, 4, 4};
    std::string auto_pick;
    {
        auto p = make_solve<T>(s16);
        auto_pick = traced_choice([&] { this->run(p); });
    }
    ASSERT_TRUE(auto_pick == "cta" || auto_pick == "sg_left" || auto_pick == "blocked" || auto_pick == "vendor")
        << auto_pick;
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    const std::string native_pick = auto_pick == "vendor" ? "cta" : auto_pick;
    const std::pair<const char*, std::string> expect[] = {
        {"CTA", "cta"}, {"cta", "cta"},
        {"sg_left", "sg_left"}, {"Blocked", "blocked"},       {"vendor", vendor_pick}, {"native", native_pick},
        {"auto", auto_pick}};
    for (const auto& [word, spelling] : expect) {
        for (bool via_env : {false, true}) {
            auto p = make_solve<T>(s16);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_TRSM_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("trsm", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_TRSM_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
            EXPECT_EQ(err.find("trsm pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                << what << ": " << err;
            expect_solved(p, what);
        }
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_TRSM_ROUTE, and nested pins restore the outer one.
TYPED_TEST(TrsmCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const Spec s{Side::Right, Uplo::Upper, Transpose::NoTrans, Diag::Unit, 40, 5, 2};
    const ScopedEnvVar env("BATCHLAS_TRSM_ROUTE", "cta");
    {
        auto p = make_solve<T>(s);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_solve<T>(s);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("trsm", C{ts::Blocked{}});
                  this->run(p);
              }),
              "blocked");
    expect_solved(p, "blocked over env cta");
    const Spec s8{Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::Unit, 8, 5, 2};
    const Pin outer("trsm", C{ts::Cta{}});
    {
        const Pin inner("trsm", C{ts::Blocked{}});
        auto r = make_solve<T>(s8);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "blocked");
    }
    auto r = make_solve<T>(s8);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "cta") << "the inner pin did not restore the outer";
    expect_solved(r, "outer cta");
}

// Auto against the transcribed table on a device that reads it (sm_89, sm_120 complex, which
// carries the same rows, or borrowed): cta up to order 32, blocked above (33 lands on the
// order-32 row, where cta cannot run), both sides, every trans, at grid and off-grid batches.
TYPED_TEST(TrsmCandidates, AutoReadsTheSm89TranscribedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_TRSM_ROUTE", nullptr);
    const auto tables = select::tables_in_borrow_order("trsm", select::dtype_name<T>(), select::device_of<B>(*this->ctx));
    if (tables.empty() || tables.front()->source != "transcribed:8b9adeb3")
        GTEST_SKIP() << "this device reads a measured trsm table";
    struct Row { int n, q, batch; const char* expect; };
    const Row rows[] = {{8, 4, 512, "cta"},     {32, 64, 128, "cta"},     {33, 8, 512, "blocked"},
                        {48, 8, 300, "blocked"}, {200, 3, 128, "blocked"}, {1, 1, 2048, "cta"}};
    for (const Row& r : rows)
        for (Side side : {Side::Left, Side::Right})
            for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                Spec s{side, Uplo::Lower, t, Diag::NonUnit, r.n, r.q, r.batch};
                s.seed = 41u + r.n;
                auto p = make_solve<T>(s);
                EXPECT_EQ(traced_choice([&] { this->run(p); }), r.expect) << label(s);
                expect_solved(p, "auto " + label(s));
            }
}

// key_of's every field reaches choose(): a synthetic table for this device whose winner
// changes with side, trans (ConjTrans folding to T), order, q on each side, and batch alone.
// Fixing a field in key_of, or taking q from the wrong extent of B, turns exactly its row red.
TYPED_TEST(TrsmCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_TRSM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("trsm.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("trsm." + dtype + "." + dev + ".txt",
                       "# op=trsm dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: side:exact trans:exact order:log:2 q:log batch:log\n"
                       "side=L trans=N order=8 q=8 batch=128 | sg_left 1 | blocked 2 | cta 3\n"
                       "side=L trans=N order=8 q=8 batch=32768 | cta 1 | blocked 2 | sg_left 3\n"
                       "side=L trans=N order=8 q=64 batch=128 | blocked 1 | cta 2 | sg_left 3\n"
                       "side=L trans=N order=16 q=8 batch=128 | cta 1 | sg_left 2 | blocked 3\n"
                       "side=L trans=T order=8 q=8 batch=128 | cta 1 | sg_left 2 | blocked 3\n"
                       "side=R trans=N order=8 q=8 batch=128 | cta 1 | blocked 2\n"
                       "side=R trans=N order=8 q=64 batch=128 | blocked 1 | cta 2\n"
                       "side=R trans=N order=16 q=8 batch=128 | blocked 1 | cta 2\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { Side side; Transpose t; int n, q, batch; const char* expect; const char* field; };
    const Probe probes[] = {{Side::Left, Transpose::NoTrans, 8, 8, 128, "sg_left", "base"},
                            {Side::Left, Transpose::NoTrans, 8, 8, 32768, "cta", "batch"},
                            {Side::Left, Transpose::NoTrans, 8, 64, 128, "blocked", "q (Left: B.cols)"},
                            {Side::Left, Transpose::NoTrans, 16, 8, 128, "cta", "order"},
                            {Side::Left, Transpose::Trans, 8, 8, 128, "cta", "trans"},
                            {Side::Left, Transpose::ConjTrans, 8, 8, 128, "cta", "trans (C folds to T)"},
                            {Side::Right, Transpose::NoTrans, 8, 8, 128, "cta", "side"},
                            {Side::Right, Transpose::NoTrans, 8, 64, 128, "blocked", "q (Right: B.rows)"},
                            {Side::Right, Transpose::NoTrans, 16, 8, 128, "blocked", "order (Right: A.rows)"}};
    for (const auto& k : probes) {
        Spec s{k.side, Uplo::Lower, k.t, Diag::NonUnit, k.n, k.q, k.batch};
        s.seed = 43u;
        auto p = make_solve<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row";
        expect_solved(p, std::string("the ") + k.field + " row");
    }
}

// RouteTrsm.RhsCountFollowsSide, ported: the trace key's q is B.cols on Side::Left and B.rows
// on Side::Right, and order is A's order on both.
TYPED_TEST(TrsmCandidates, TraceKeyQFollowsSide) {
    using T = typename TestFixture::T;
    for (Side side : {Side::Left, Side::Right}) {
        Spec s{side, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit, 12, 37, 5};
        auto p = make_solve<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        const std::string want = std::string("side=") + side_s(side) + " trans=T order=12 q=37 batch=5 ->";
        EXPECT_NE(line.find(want), std::string::npos) << line;
        expect_solved(p, line);
    }
}

// RouteTrsm.VendorFreeStillFindsANativeRouteAtEveryOrder, ported: with a table naming only the
// vendor, a vendor-free build takes the last resort, blocked, at order 32 and at order 4096;
// under the shipped tables Auto runs cta at 32 and blocked at 4096 in every build.
TYPED_TEST(TrsmCandidates, VendorFreeLastResortIsBlocked) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_TRSM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    const Spec s32{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 32, 128, 64};
    const Spec s4k{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 4096, 2, 1};
    {
        auto p = make_solve<T>(s32);
        const std::string got = traced_choice([&] { this->run(p); });
        const auto tables = select::tables_in_borrow_order("trsm", dtype, select::device_of<TestFixture::B>(*this->ctx));
        if (!tables.empty() && tables.front()->source == "transcribed:8b9adeb3") EXPECT_EQ(got, "cta");
        else EXPECT_NE(got, "vendor") << "nothing ranks the vendor at order 32 on " << dev;
        expect_solved(p, "auto order 32");
    }
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("trsm.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("trsm." + dtype + "." + dev + ".txt",
                       "# op=trsm dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: side:exact trans:exact order:log:2 q:log batch:log\n"
                       "side=L trans=N order=32 q=128 batch=128 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    for (const Spec& s : {s32, s4k}) {
        auto p = make_solve<T>(s);
        std::string all;
        const std::string got = traced_choice([&] { this->run(p); }, &all);
        EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : "blocked") << "order " << s.n;
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_solved(p, "vendor-only table, order " + std::to_string(s.n));
    }
}

// The coverage row (§5.6): the real backend (the old shape builder left it AUTO), the trsm key
// (m = B.rows, n = B.cols, k = order, side, uplo, transA, diag) and the native flags.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(TrsmCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "trsm_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 16, 4, 2};
    const Spec hi{Side::Right, Uplo::Upper, Transpose::ConjTrans, Diag::Unit, 48, 3, 2};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_TRSM_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("trsm", C{ts::Cta{}});
            auto p = make_solve<T>(lo);
            this->run(p);
        }
        const Pin pin("trsm", C{ts::Blocked{}});
        auto p = make_solve<T>(hi);
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,trsm,", 0) != 0) continue;
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
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    for (const auto& [key, s, algo] : {std::tuple{klo, lo, "cta"}, std::tuple{khi, hi, "blocked"}}) {
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key << ": the backend column";
        EXPECT_NE(f[3], "AUTO") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[15], as_int(s.uplo)) << key;
        EXPECT_EQ(f[16], as_int(s.side)) << key;
        EXPECT_EQ(f[17], as_int(s.diag)) << key;
        EXPECT_EQ(f[18], as_int(s.trans)) << key;
    }
}

// RouteTrsm.CorrectnessGatesAreNotSpeedGates' CPU case, ported: on a CPU queue no native
// family can run, so native pins throw and Auto is the vendor (or no route without one).
template <typename Config>
class TrsmCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(TrsmCandidatesCpu, Types);

TYPED_TEST(TrsmCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_TRSM_ROUTE", nullptr);
    Spec s{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 8, 3, 2};
    for (const C& c : ts::candidates<T>()) {
        if (std::holds_alternative<ts::Vendor>(c)) continue;
        auto p = make_solve<T>(s);
        const Pin pin("trsm", c);
        EXPECT_THROW(((void)trsm<B, T>(*this->ctx, p.A(), p.B(), p.alpha, s.side, s.uplo, s.trans, s.diag)),
                     std::invalid_argument)
            << select::to_string(c);
    }
    auto p = make_solve<T>(s);
    auto call = [&] {
        (void)trsm<B, T>(*this->ctx, p.A(), p.B(), p.alpha, s.side, s.uplo, s.trans, s.diag);
        this->ctx->wait();
    };
    if constexpr (batchlas::select::level3_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        expect_solved(p, "cpu vendor");
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

// The transcribed sm_89 rows, read with Table::nearest directly so every device checks them:
// cta | blocked | vendor up to order 32 and blocked | vendor above, on both sides and trans.
TEST(TrsmTranscribedTable, Sm89RowsHoldTheOldPreference) {
    struct Row { const char* dtype; const char* side; const char* trans; int n, q, batch; const char* first; std::size_t len; };
    const Row rows[] = {
        {"float", "L", "N", 32, 8, 512, "cta", 3},       {"float", "L", "N", 48, 8, 512, "blocked", 2},
        {"double", "R", "T", 24, 4096, 128, "cta", 3},   {"double", "R", "T", 1024, 1, 32768, "blocked", 2},
        {"cfloat", "L", "T", 1, 1, 128, "cta", 3},       {"cfloat", "R", "N", 64, 16, 8192, "blocked", 2},
        {"cdouble", "R", "N", 16, 2, 2048, "cta", 3},    {"cdouble", "L", "T", 33, 2, 2048, "cta", 3},
        {"cdouble", "L", "N", 40, 2, 2048, "blocked", 2}};
    for (const Row& r : rows) {
        const auto tables = select::tables_in_borrow_order("trsm", r.dtype, select::device_from_key("sm_89"));
        ASSERT_FALSE(tables.empty()) << r.dtype;
        const select::Table& t = *tables.front();
        ASSERT_EQ(t.device, "sm_89") << r.dtype;
        EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
        const select::Key key{{"side", r.side}, {"trans", r.trans}, {"order", r.n}, {"q", r.q}, {"batch", r.batch}};
        const select::TableRow* row = t.nearest(key);
        ASSERT_NE(row, nullptr) << t.file;
        const std::string what = t.file + ":" + std::to_string(row->line);
        ASSERT_EQ(row->ranked.size(), r.len) << what;
        EXPECT_EQ(row->ranked.front().spelling, r.first) << what;
        EXPECT_EQ(row->ranked.back().spelling, "vendor") << what;
        EXPECT_FALSE(row->timed) << what;
    }
}

}  // namespace
