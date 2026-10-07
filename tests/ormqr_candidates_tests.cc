// ormqr's flat-selection suite (docs/design/flat-kernel-selection.md §8): every candidate
// pinned on shapes straddling its limits, pinned-equals-direct bit for bit, a saturating batch,
// can_run equals launch, the exact workspace in a poisoned arena, the pin rules, key fields,
// coverage and the transcribed tables.

#include <gtest/gtest.h>

#include <batchlas/blas/functions/ormqr.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/tuning_params.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/ops/ormqr/choice.hh"

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

namespace om = batchlas::ops::ormqr;
using C = om::OrmqrChoice;
using Pin = select::ScopedPin<C>;
using cd = std::complex<double>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;

template <typename T>
T mk(double r, double i) {
    if constexpr (kCx<T>) return T(RealOf<T>(r), RealOf<T>(i));
    else return T(r);
}
template <typename T>
cd up(T v) {
    if constexpr (kCx<T>) return {double(v.real()), double(v.imag())};
    else return {double(v), 0.0};
}
template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

const char* side_s(Side s) { return s == Side::Left ? "L" : "R"; }
const char* trans_s(Transpose t) { return t == Transpose::NoTrans ? "N" : (t == Transpose::Trans ? "T" : "C"); }

struct Region {
    std::size_t off = 0;
    int rows = 0, cols = 0, ld = 0, stride = 0;
};

struct Spec {
    Side side = Side::Left;
    Transpose trans = Transpose::NoTrans;
    int m = 8, k = 5, q = 3, batch = 3;
    int hint = 0;
    int period = 0;  // > 0: item it repeats item it % period
    unsigned seed = 1;
};

std::string label(const Spec& s) {
    return std::string(side_s(s.side)) + trans_s(s.trans) + " m=" + std::to_string(s.m) + " k=" +
           std::to_string(s.k) + " q=" + std::to_string(s.q) + " batch=" + std::to_string(s.batch) +
           (s.hint ? " hint=" + std::to_string(s.hint) : "");
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// A (m x k reflectors) and C in one buffer of large finite poison at padded lds and strides.
// A's diagonal and upper triangle hold in-range poison: the reflectors are unit lower, so a
// kernel that reads either answers wrongly. tau makes each H_j unitary with a nonzero
// imaginary part for complex T, so op(Q) stays well conditioned and a conjugation bug shows.
template <typename T>
struct Apply {
    Spec s;
    Region a, c;
    UnifiedVector<T> mem, tau;
    std::vector<T> mem0;
    UnifiedVector<T*> aptr, cptr;

    MVof<T> A() { return {mem.data() + a.off, s.m, s.k, a.ld, a.stride, s.batch, aptr.data()}; }
    MVof<T> Cm() { return {mem.data() + c.off, c.rows, c.cols, c.ld, c.stride, s.batch, cptr.data()}; }
    Span<T> Tau() { return tau.to_span(); }
    std::size_t ai(int it, int i, int j) const { return a.off + std::size_t(it) * a.stride + std::size_t(j) * a.ld + i; }
    std::size_t ci(int it, int i, int j) const { return c.off + std::size_t(it) * c.stride + std::size_t(j) * c.ld + i; }
};

template <typename T>
Apply<T> make_apply(const Spec& s) {
    Apply<T> p;
    p.s = s;
    p.a = {0, s.m, s.k, s.m + 3, (s.m + 3) * s.k + 5};
    const int cr = s.side == Side::Left ? s.m : s.q, cc = s.side == Side::Left ? s.q : s.m;
    p.c = {std::size_t(p.a.stride) * s.batch + 11, cr, cc, cr + 2, (cr + 2) * cc + 7};
    p.mem = UnifiedVector<T>(p.c.off + std::size_t(p.c.stride) * s.batch + 13, mk<T>(-999, 777));
    p.tau = UnifiedVector<T>(std::size_t(std::max(1, s.k)) * s.batch, mk<T>(0, 0));
    p.aptr = UnifiedVector<T*>(s.batch, nullptr);
    p.cptr = UnifiedVector<T*>(s.batch, nullptr);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<double> u(-1, 1);
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    const double theta = kCx<T> ? 0.7 : 0.0;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        for (int j = 0; j < s.k; ++j) {
            double norm2 = 1.0;
            for (int i = 0; i < s.m; ++i) {
                if (it != r) {
                    p.mem[p.ai(it, i, j)] = p.mem[p.ai(r, i, j)];
                    continue;
                }
                const T v = i > j ? mk<T>(u(gen) / std::sqrt(double(s.m)), u(gen) / std::sqrt(double(s.m)))
                                  : mk<T>(300, -200);
                p.mem[p.ai(it, i, j)] = v;
                if (i > j) norm2 += std::norm(up(v));
            }
            if (it != r) {
                p.tau[std::size_t(it) * s.k + j] = p.tau[std::size_t(r) * s.k + j];
                continue;
            }
            const cd t = (1.0 + std::polar(1.0, theta)) / norm2;
            p.tau[std::size_t(it) * s.k + j] = mk<T>(t.real(), t.imag());
        }
        for (int j = 0; j < p.c.cols; ++j)
            for (int i = 0; i < p.c.rows; ++i)
                p.mem[p.ci(it, i, j)] = it != r ? p.mem[p.ci(r, i, j)] : mk<T>(u(gen), u(gen));
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

// op(Q) C or C op(Q) for one item, in double, from the stored reflectors and tau.
template <typename T>
std::vector<cd> reference(const Apply<T>& p, int it) {
    const Spec& s = p.s;
    const int m = s.m;
    std::vector<cd> Q(std::size_t(m) * m, 0.0);
    for (int i = 0; i < m; ++i) Q[i + std::size_t(i) * m] = 1.0;
    for (int j = 0; j < s.k; ++j) {  // Q <- Q H_j
        std::vector<cd> v(m, 0.0);
        v[j] = 1.0;
        for (int i = j + 1; i < m; ++i) v[i] = up(p.mem0[p.ai(it, i, j)]);
        const cd tj = up(p.tau[std::size_t(it) * s.k + j]);
        for (int r = 0; r < m; ++r) {
            cd w = 0.0;
            for (int i = 0; i < m; ++i) w += Q[r + std::size_t(i) * m] * v[i];
            for (int i = 0; i < m; ++i) Q[r + std::size_t(i) * m] -= tj * w * std::conj(v[i]);
        }
    }
    auto opq = [&](int i, int j) {
        if (s.trans == Transpose::NoTrans) return Q[i + std::size_t(j) * m];
        const cd x = Q[j + std::size_t(i) * m];
        return s.trans == Transpose::ConjTrans ? std::conj(x) : x;
    };
    const int rows = p.c.rows, cols = p.c.cols;
    std::vector<cd> X(std::size_t(rows) * cols, 0.0);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i) {
            cd acc = 0.0;
            for (int t = 0; t < m; ++t)
                acc += s.side == Side::Left ? opq(i, t) * up(p.mem0[p.ci(it, t, j)])
                                            : up(p.mem0[p.ci(it, i, t)]) * opq(t, j);
            X[i + std::size_t(j) * rows] = acc;
        }
    return X;
}

template <typename T>
double tol(const Spec& s) {
    return 64.0 * (s.m + s.k + 1) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

// The checked items against the reference, every element outside C's footprint bit for bit,
// and, for a repeating batch, every item bit-identical to its representative.
template <typename T>
void expect_applied(const Apply<T>& p, const std::string& what) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0) for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    else if (s.batch <= 8) for (int it = 0; it < s.batch; ++it) items.push_back(it);
    else items = {0, 1, s.batch / 2, s.batch - 1};
    for (int it : items) {
        const auto X = reference(p, it);
        double err = 0, scale = 1;
        for (int j = 0; j < p.c.cols; ++j)
            for (int i = 0; i < p.c.rows; ++i) {
                const cd x = X[i + std::size_t(j) * p.c.rows];
                err = std::max(err, std::abs(up(p.mem[p.ci(it, i, j)]) - x));
                scale = std::max(scale, std::abs(x));
            }
        ASSERT_TRUE(std::isfinite(err) && err / scale <= tol<T>(s)) << what << " item " << it << " error " << err / scale;
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

// The outermost ormqr trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("ormqr ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no ormqr trace line in: " + err + ">";
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

// A table set with this device's ormqr table for `dtype` replaced by `text`.
void install_table(const std::string& dtype, const std::string& dev, const std::string& rows) {
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("ormqr.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("ormqr." + dtype + "." + dev + ".txt",
                       "# op=ormqr dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: side:exact trans:exact m:log k:log q:log batch:log\n" + rows);
    select::testing::set_builtin_tables(std::move(files));
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class OrmqrCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MVof<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::factorization_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native family is GPU-only";
    }

    // The limit oracle: the drivers' own refusals, not ormqr.cc's can_run. Complex Trans is
    // refused by ormqr_blocked (`unsupported`) and by every vendor unmqr.
    static bool defined(Transpose t) { return !(kCx<T> && t == Transpose::Trans); }
    static bool expect_runs(const C& c, Transpose t) {
        if (std::holds_alternative<om::Blocked>(c)) return defined(t);
        return defined(t) && kVendor;
    }
    static bool is_undefined_error(const std::exception& e) {
        return std::string(e.what()).find("is not an ormqr operation") != std::string::npos;
    }
    // A Vendor choice spells the class word `vendor`, which falls back to Auto without a vendor.
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<om::Vendor>(c) && !kVendor; }

    static std::int32_t blocked_width(const Spec& s) {
        if (s.hint > 0) return std::max(1, std::min(s.hint, std::max(1, s.k)));
        return tuning::ormqr_block_size_for_n(s.m);
    }

    std::size_t size(Apply<T>& p, Queue* q = nullptr) {
        return ormqr_buffer_size<B, T>(q ? *q : *this->ctx, p.A(), p.Cm(), p.s.side, p.s.trans, p.Tau(), p.s.hint);
    }
    // Sized by the public sizing call, run in a poisoned arena; the guard bytes must survive.
    void run(Apply<T>& p, Queue* q = nullptr) {
        Queue& qq = q ? *q : *this->ctx;
        const std::size_t need = size(p, &qq);
        UnifiedVector<std::byte> ws(need + 4096, std::byte{0xA5});
        (void)ormqr<B, T>(qq, p.A(), p.Cm(), p.s.side, p.s.trans, p.Tau(), Span<std::byte>(ws.data(), need), p.s.hint);
        qq.wait();
        for (std::size_t i = need; i < ws.size(); ++i)
            ASSERT_EQ(ws[i], std::byte{0xA5}) << label(p.s) << ": wrote " << (i - need) << " bytes past the workspace";
    }
    void run_pinned(const C& c, Apply<T>& p) {
        const Pin pin("ormqr", c);
        run(p);
    }
    bool pin_accepted(const C& c, Apply<T>& p) {
        const Pin pin("ormqr", c);
        try {
            (void)size(p);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos && !is_undefined_error(e))
                throw;
            return false;
        }
    }
    std::string auto_choice(Apply<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_ORMQR_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }

    // The family's own driver with its own sizing call.
    bool direct(const C& c, Apply<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const Spec& s = p.s;
        try {
            if (std::holds_alternative<om::Blocked>(c)) {
                const auto w = blocked_width(s);
                UnifiedVector<std::byte> ws(ormqr_blocked_buffer_size<B, T>(q, p.A(), p.Cm(), s.side, s.trans, p.Tau(), w));
                (void)ormqr_blocked<B, T>(q, p.A(), p.Cm(), s.side, s.trans, p.Tau(), ws.to_span(), w);
            } else {
                if constexpr (kVendor) {
                    // The Vendor family spells a real ConjTrans as Trans (real unmqr takes N/T only).
                    const Transpose vt = !kCx<T> && s.trans == Transpose::ConjTrans ? Transpose::Trans : s.trans;
                    UnifiedVector<std::byte> ws(backend::ormqr_vendor_buffer_size<B, T>(q, p.A(), p.Cm(), s.side, vt, p.Tau()));
                    (void)backend::ormqr_vendor<B, T>(q, p.A(), p.Cm(), s.side, vt, p.Tau(), ws.to_span());
                } else {
                    if (why) *why = "no vendor solver in this build";
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

TYPED_TEST_SUITE(OrmqrCandidates, Types);

// Shapes across the WY widths (16/32/64 and the hinted 16): k below, at and past a block, k < m,
// k = m, one-column C, both sides.
std::vector<Spec> straddles() {
    std::vector<Spec> out;
    const std::tuple<int, int, int, int> shapes[] = {{1, 1, 1, 0},   {5, 3, 4, 0},    {16, 16, 7, 0}, {17, 17, 3, 0},
                                                     {40, 33, 1, 16}, {64, 15, 9, 16}, {70, 64, 5, 0}, {100, 37, 2, 0}};
    unsigned seed = 3;
    for (const auto& [m, k, q, hint] : shapes)
        for (Side side : {Side::Left, Side::Right})
            for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                Spec s{side, t, m, k, q, 2, hint};
                s.seed = seed++;
                out.push_back(s);
            }
    return out;
}

// §8.1: each candidate on every straddle; accepted shapes are correct, refused ones throw from
// the pin with "cannot run this shape".
TYPED_TEST(OrmqrCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    int ran = 0, refused = 0;
    for (const C& c : om::candidates<T>())
        for (const Spec& s : straddles()) {
            if (TestFixture::vendor_word_falls_back(c)) continue;
            auto p = make_apply<T>(s);
            if (!this->expect_runs(c, s.trans)) {
                EXPECT_FALSE(this->pin_accepted(c, p)) << name(c, s);
                ++refused;
                continue;
            }
            this->run_pinned(c, p);
            expect_applied(p, name(c, s));
            ++ran;
        }
    EXPECT_GT(ran, 0);
    if (kCx<T>) EXPECT_GT(refused, 0) << "complex Trans must be refused";
}

// The pinned public call is the family's own driver, bit for bit.
TYPED_TEST(OrmqrCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    for (const C& c : om::candidates<T>())
        for (const Spec& s : {Spec{Side::Left, Transpose::ConjTrans, 48, 20, 6, 3, 0, 0, 7},
                              Spec{Side::Right, Transpose::NoTrans, 33, 33, 4, 2, 16, 0, 8},
                              Spec{Side::Left, Transpose::NoTrans, 100, 37, 4, 2, 5, 0, 9}}) {  // hint 5 != ladder
            if (!this->expect_runs(c, s.trans)) continue;
            auto a = make_apply<T>(s);
            auto b = make_apply<T>(s);
            this->run_pinned(c, a);
            std::string why;
            ASSERT_TRUE(this->direct(c, b, &why)) << name(c, s) << ": " << why;
            for (std::size_t e = 0; e < a.mem.size(); ++e)
                ASSERT_TRUE(same_bits(a.mem[e], b.mem[e])) << name(c, s) << ": differs at element " << e;
        }
}

// §8.1 / docs/developer/agent-guide.md §8.5: batch 1024 of one matrix, every item bit-identical to item 0.
TYPED_TEST(OrmqrCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    for (const C& c : om::candidates<T>())
        for (Side side : {Side::Left, Side::Right}) {
            Spec s{side, kCx<T> ? Transpose::ConjTrans : Transpose::Trans, 24, 24, 8, 1024, 0, 1, 11};
            if (!this->expect_runs(c, s.trans)) continue;
            auto p = make_apply<T>(s);
            this->run_pinned(c, p);
            expect_applied(p, name(c, s) + " saturating");
        }
}

// §8.2 (R3): a pin is accepted exactly when the family's own driver launches.
TYPED_TEST(OrmqrCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    int disagreements = 0;
    for (const C& c : om::candidates<T>())
        for (Side side : {Side::Left, Side::Right})
            for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                if (TestFixture::vendor_word_falls_back(c)) continue;
                Spec s{side, t, 8, 5, 3, 2};
                auto a = make_apply<T>(s);
                auto b = make_apply<T>(s);
                const bool pin = this->pin_accepted(c, a);
                std::string why;
                const bool run = this->direct(c, b, &why);
                EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the driver "
                                    << (run ? "launches" : "refuses: " + why);
                EXPECT_EQ(pin, this->expect_runs(c, t)) << name(c, s) << ": can_run disagrees with the oracle";
                disagreements += pin != run;
            }
    EXPECT_EQ(disagreements, 0);
}

// R5 (ports BufferSizeAgreesWithDispatchUnderAnUnmatchedForcedRoute): under every pin the public
// size is exactly the chosen family's own, and the call runs in exactly that much (run() checks
// the guard bytes). Auto's size is the size of the family Auto names.
TYPED_TEST(OrmqrCandidates, BufferSizeIsTheChosenFamilysAndRuns) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const Spec s{Side::Left, Transpose::NoTrans, 40, 40, 6, 3};
    std::map<std::string, std::size_t> own;
    for (const C& c : om::candidates<T>()) {
        if (!this->expect_runs(c, s.trans)) continue;
        auto p = make_apply<T>(s);
        Queue& q = *this->ctx;
        std::size_t want = 0;
        if (std::holds_alternative<om::Blocked>(c)) {
            want = ormqr_blocked_buffer_size<B, T>(q, p.A(), p.Cm(), s.side, s.trans, p.Tau(), TestFixture::blocked_width(s));
        } else if constexpr (TestFixture::kVendor) {
            want = backend::ormqr_vendor_buffer_size<B, T>(q, p.A(), p.Cm(), s.side, s.trans, p.Tau());
        }
        own[select::to_string(c)] = want;
        {
            const Pin pin("ormqr", c);
            EXPECT_EQ(this->size(p), want) << name(c, s);
        }
        this->run_pinned(c, p);
        expect_applied(p, name(c, s) + " exact workspace");
    }
    auto p = make_apply<T>(s);
    const std::string pick = this->auto_choice(p);
    ASSERT_TRUE(own.count(pick)) << pick;
    const ScopedEnvVar clear("BATCHLAS_ORMQR_ROUTE", nullptr);
    EXPECT_EQ(this->size(p), own[pick]) << "auto picked " << pick;
}

// The old ormqr_dispatch ran an out-of-order caller's work on an in-order copy; so does this.
TYPED_TEST(OrmqrCandidates, OutOfOrderQueueIsSequenced) {
    using T = typename TestFixture::T;
    Queue ooo(*this->ctx, false);
    ASSERT_FALSE(ooo.in_order());
    for (const C& c : om::candidates<T>()) {
        const Spec s{Side::Right, Transpose::ConjTrans, 36, 30, 5, 4, 0, 0, 21};
        if (!this->expect_runs(c, s.trans)) continue;
        auto p = make_apply<T>(s);
        const Pin pin("ormqr", c);
        this->run(p, &ooo);
        expect_applied(p, name(c, s) + " out-of-order queue");
    }
}

// R6: spellings that name nothing compiled throw instead of meaning Auto.
TYPED_TEST(OrmqrCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "cta", "native:cta", "blocked:64", "blocked:nb=32", "two_stage", "tiny",
                             // removed aliases (phase 5): each must stay an error
                             "native:blocked", "vendor:auto"}) {
        auto p = make_apply<T>(Spec{});
        const Pin pin("ormqr", std::string_view(word));
        EXPECT_THROW((void)this->size(p), std::invalid_argument) << word;
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
}

// Ports ForcingCannotSelectAnUnsupportedRoute: complex Trans is refused under Auto and under
// every pin (sizing and running alike) as an argument error, and nothing is written; both
// drivers refuse it too, so no family could have served it. Every real Trans runs.
TYPED_TEST(OrmqrCandidates, ComplexTransIsRefusedUnderEveryPin) {
    using T = typename TestFixture::T;
    for (Side side : {Side::Left, Side::Right}) {
        const Spec s{side, Transpose::Trans, 8, 6, 3, 2};
        std::vector<std::optional<C>> pins{std::nullopt};
        for (const C& c : om::candidates<T>()) pins.push_back(c);
        for (const auto& c : pins) {
            auto p = make_apply<T>(s);
            std::optional<Pin> pin;
            if (c) pin.emplace("ormqr", *c);
            const std::string what = (c ? select::to_string(*c) : std::string("auto")) + " " + label(s);
            if (!kCx<T>) {
                if (c && !this->expect_runs(*c, s.trans)) continue;
                this->run(p);
                expect_applied(p, what);
                continue;
            }
            for (int call = 0; call < 2; ++call) {
                try {
                    if (call == 0) (void)this->size(p);
                    else this->run(p);
                    ADD_FAILURE() << what << " was accepted";
                } catch (const std::invalid_argument& e) {
                    EXPECT_TRUE(TestFixture::is_undefined_error(e)) << what << ": " << e.what();
                }
            }
            for (std::size_t e = 0; e < p.mem.size(); ++e)
                ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": a refused call wrote element " << e;
        }
        if constexpr (kCx<T>) {
            for (const C& c : om::candidates<T>()) {
                auto p = make_apply<T>(s);
                std::string why;
                EXPECT_FALSE(this->direct(c, p, &why)) << name(c, s) << ": the driver accepted complex Trans";
            }
        }
    }
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment.
TYPED_TEST(OrmqrCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_ORMQR_ROUTE", nullptr);
    const Spec s{Side::Left, Transpose::NoTrans, 12, 8, 4, 3};
    std::string auto_pick;
    {
        auto p = make_apply<T>(s);
        auto_pick = this->auto_choice(p);
    }
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    const std::pair<const char*, std::string> expect[] = {
        {"BLOCKED", "blocked"}, {"blocked", "blocked"}, {"native", "blocked"},
        {"vendor", vendor_pick},       {"auto", auto_pick}};
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = make_apply<T>(s);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_ORMQR_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("ormqr", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_ORMQR_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            expect_applied(p, what);
        }
    if constexpr (TestFixture::kVendor) {
        auto p = make_apply<T>(s);
        const Pin pin("ormqr", std::string_view("vendor"));
        EXPECT_EQ(traced_choice([&] { this->run(p); }), "vendor");
    }
}

// The retired BATCHLAS_ORMQR_PROVIDER is not read: neither a valid word nor a bad one changes
// the choice or throws.
TYPED_TEST(OrmqrCandidates, RetiredProviderVariableIsNotRead) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_ORMQR_ROUTE", nullptr);
    const Spec s{Side::Right, Transpose::NoTrans, 10, 7, 3, 2};
    std::string auto_pick;
    {
        auto p = make_apply<T>(s);
        auto_pick = this->auto_choice(p);
    }
    for (const char* word : {"vendor", "blocked", "cta"}) {
        const ScopedEnvVar retired("BATCHLAS_ORMQR_PROVIDER", word);
        auto p = make_apply<T>(s);
        EXPECT_NO_THROW((void)this->size(p)) << word;
        EXPECT_EQ(traced_choice([&] { this->run(p); }), auto_pick) << word;
    }
}

// Auto against the shipped transcribed tables on a device that reads them: blocked first
// everywhere. Their complex-Trans rows name the vendor (the old router's choice), but that call
// is refused before choose() (ComplexTransIsRefusedUnderEveryPin).
TYPED_TEST(OrmqrCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const auto tables = select::tables_in_borrow_order("ormqr", select::dtype_name<T>(), select::device_of<B>(*this->ctx));
    if (tables.empty() || tables.front()->source.rfind("transcribed:", 0) != 0)
        GTEST_SKIP() << "this device does not read a transcribed ormqr table";
    for (const Spec& s : {Spec{Side::Left, Transpose::NoTrans, 9, 9, 2, 300}, Spec{Side::Right, Transpose::ConjTrans, 70, 3, 40, 128},
                          Spec{Side::Left, Transpose::Trans, 33, 20, 1, 5}, Spec{Side::Right, Transpose::Trans, 2, 1, 700, 3000}}) {
        auto p = make_apply<T>(s);
        if (!TestFixture::defined(s.trans)) continue;
        EXPECT_EQ(this->auto_choice(p), "blocked") << label(s);
        expect_applied(p, "auto " + label(s));
    }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with side,
// trans (T and C kept apart), m, k, q and batch alone. Needs the vendor to tell winners apart.
TYPED_TEST(OrmqrCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!TestFixture::kVendor) GTEST_SKIP() << "only blocked can run, so no key can change the choice";
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    const TableGuard restore;
    install_table(dtype, dev,
                  "side=L trans=N m=16 k=8 q=8 batch=128 | blocked 1 | vendor 2\n"
                  "side=L trans=N m=16 k=8 q=8 batch=32768 | vendor 1 | blocked 2\n"
                  "side=L trans=N m=16 k=8 q=512 batch=128 | vendor 1 | blocked 2\n"
                  "side=L trans=N m=16 k=1 q=8 batch=128 | vendor 1 | blocked 2\n"
                  "side=L trans=N m=1024 k=8 q=8 batch=128 | vendor 1 | blocked 2\n"
                  "side=L trans=C m=16 k=8 q=8 batch=128 | vendor 1 | blocked 2\n"
                  "side=L trans=T m=16 k=8 q=8 batch=128 | blocked 1 | vendor 2\n"
                  "side=R trans=N m=16 k=8 q=8 batch=128 | vendor 1 | blocked 2\n"
                  "side=R trans=N m=16 k=8 q=512 batch=128 | blocked 1 | vendor 2\n");
    struct Probe { Side side; Transpose t; int m, k, q, batch; const char* expect; const char* field; };
    const char* t_pick = "blocked";  // complex Trans is refused before choose(): skipped below
    const Probe probes[] = {{Side::Left, Transpose::NoTrans, 16, 8, 8, 128, "blocked", "base"},
                            {Side::Left, Transpose::NoTrans, 16, 8, 8, 16384, "vendor", "batch"},
                            {Side::Left, Transpose::NoTrans, 16, 8, 400, 128, "vendor", "q (Left: C.cols)"},
                            {Side::Left, Transpose::NoTrans, 16, 1, 8, 128, "vendor", "k"},
                            {Side::Left, Transpose::NoTrans, 900, 8, 8, 128, "vendor", "m"},
                            {Side::Left, Transpose::ConjTrans, 16, 8, 8, 128, "vendor", "trans C"},
                            {Side::Left, Transpose::Trans, 16, 8, 8, 128, t_pick, "trans T (not folded into C)"},
                            {Side::Right, Transpose::NoTrans, 16, 8, 8, 128, "vendor", "side"},
                            {Side::Right, Transpose::NoTrans, 16, 8, 400, 128, "blocked", "q (Right: C.rows)"}};
    for (const auto& k : probes) {
        if (!TestFixture::defined(k.t)) continue;
        Spec s{k.side, k.t, k.m, k.k, k.q, k.batch};
        auto p = make_apply<T>(s);
        EXPECT_EQ(this->auto_choice(p), k.expect) << "the " << k.field << " row";
    }
}

// The trace key is key_of itself: q follows the side, k is min(A.rows, A.cols), T stays T.
TYPED_TEST(OrmqrCandidates, TraceKeyFollowsSideAndKeepsTrans) {
    using T = typename TestFixture::T;
    for (Side side : {Side::Left, Side::Right}) {
        const Spec s{side, Transpose::ConjTrans, 12, 5, 37, 5};
        auto p = make_apply<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        const std::string want = std::string("side=") + side_s(side) + " trans=C m=12 k=5 q=37 batch=5 ->";
        EXPECT_NE(line.find(want), std::string::npos) << line;
    }
    if (!kCx<T>) {
        const Spec s{Side::Left, Transpose::Trans, 6, 6, 2, 2};
        auto p = make_apply<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        EXPECT_NE(line.find(" trans=T m=6 k=6 q=2 batch=2 ->"), std::string::npos) << line;
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, blocked.
TYPED_TEST(OrmqrCandidates, VendorFreeLastResortIsBlocked) {
    using T = typename TestFixture::T;
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    const TableGuard restore;
    install_table(dtype, dev, "side=L trans=N m=16 k=8 q=8 batch=128 | vendor 1\n");
    const Spec s{Side::Left, Transpose::NoTrans, 20, 10, 3, 2};
    auto p = make_apply<T>(s);
    std::string all;
    const ScopedEnvVar clear("BATCHLAS_ORMQR_ROUTE", nullptr);
    const std::string got = traced_choice([&] { this->run(p); }, &all);
    EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : "blocked");
    if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
    expect_applied(p, "vendor-only table");
}

// The coverage row (§5.6): the real backend (the old builder left it AUTO), m/n/k from A, side,
// transA and the native flags. threadsafe: the child re-executes the binary.
TYPED_TEST(OrmqrCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "ormqr_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{Side::Left, Transpose::NoTrans, 16, 6, 4, 2};
    const Spec hi{Side::Right, Transpose::ConjTrans, 30, 9, 3, 2};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_ORMQR_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        for (const Spec& s : {lo, hi}) {
            const Pin pin("ormqr", C{om::Blocked{}});
            auto p = make_apply<T>(s);
            this->run(p);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,ormqr,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    for (const auto& [key, s] : {std::pair{std::string("16 6 6"), lo}, std::pair{std::string("30 9 9"), hi}}) {
        ASSERT_TRUE(rows.count(key)) << key;
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key << ": the backend column";
        EXPECT_EQ(f[10], "blocked") << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[16], as_int(s.side)) << key;
        EXPECT_EQ(f[18], as_int(s.trans)) << key;
    }
}

// On a CPU queue the native family cannot run: a blocked pin throws, Auto is the vendor (or no
// route without one) -- the old route table's `!is_gpu` gate.
template <typename Config>
class OrmqrCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(OrmqrCandidatesCpu, Types);

TYPED_TEST(OrmqrCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_ORMQR_ROUTE", nullptr);
    const Spec s{Side::Left, Transpose::NoTrans, 8, 5, 3, 2};
    auto p = make_apply<T>(s);
    {
        const Pin pin("ormqr", C{om::Blocked{}});
        EXPECT_THROW(((void)ormqr_buffer_size<B, T>(*this->ctx, p.A(), p.Cm(), s.side, s.trans, p.Tau())),
                     std::invalid_argument);
    }
    auto call = [&] {
        UnifiedVector<std::byte> ws(ormqr_buffer_size<B, T>(*this->ctx, p.A(), p.Cm(), s.side, s.trans, p.Tau()));
        (void)ormqr<B, T>(*this->ctx, p.A(), p.Cm(), s.side, s.trans, p.Tau(), ws.to_span());
        this->ctx->wait();
    };
    if constexpr (batchlas::select::factorization_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        expect_applied(p, "cpu vendor");
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

// The transcribed rows, read with Table::nearest directly so every device checks them: both
// devices, blocked | vendor except complex Trans (vendor alone), on and off the grid.
TEST(OrmqrTranscribedTable, RowsHoldTheOldPreferenceOnBothDevices) {
    struct Row { const char* dtype; const char* side; const char* trans; int m, k, q, batch; const char* first; std::size_t len; };
    const Row rows[] = {{"float", "L", "T", 8, 8, 32, 128, "blocked", 2},    {"double", "R", "C", 3000, 7, 1, 99, "blocked", 2},
                        {"cfloat", "L", "T", 64, 1, 1024, 2048, "vendor", 1}, {"cfloat", "R", "C", 5, 2, 3, 4, "blocked", 2},
                        {"cdouble", "R", "T", 512, 512, 2, 32768, "vendor", 1}, {"cdouble", "L", "N", 1, 1, 1, 1, "blocked", 2}};
    for (const char* dev : {"sm_89", "sm_120"})
        for (const Row& r : rows) {
            const auto tables = select::tables_in_borrow_order("ormqr", r.dtype, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << r.dtype;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << r.dtype;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            const select::Key key{{"side", r.side}, {"trans", r.trans}, {"m", r.m}, {"k", r.k}, {"q", r.q}, {"batch", r.batch}};
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
