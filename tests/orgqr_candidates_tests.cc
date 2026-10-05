// Every orgqr candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-select-p5/orgqr.md. The limit oracle reads the driver's own checks; which
// kernel ran is read back from the select trace or a bit-for-bit comparison with the direct
// driver, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/orgqr_native.hh"
#include "../src/ops/orgqr/choice.hh"

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

namespace oq = batchlas::ops::orgqr;
using C = oq::OrgqrChoice;
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
std::complex<double> up(T v) {
    if constexpr (kCx<T>) return {double(v.real()), double(v.imag())};
    else return {double(v), 0.0};
}
template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
template <typename T>
T poison() {
    return mk<T>(RealOf<T>(-999), RealOf<T>(777));
}

// One factored A at ld = m + 3 and stride = ld * n + 5 inside a buffer of large finite poison,
// and tau with poison past k * batch. `period > 0`: item it repeats item it % period bit for bit.
template <typename T>
struct Prob {
    int m = 0, n = 0, batch = 0, ld = 0, stride = 0, period = 0;
    UnifiedVector<T> mem, tau;
    UnifiedVector<T*> ptrs;  // the vendor geqrf/orgqr are pointer-array APIs
    std::vector<T> a0, f, tau0;
    MVof<T> A() { return {mem.data(), m, n, ld, stride, batch, ptrs.data()}; }
    int k() const { return std::min(m, n); }
    std::size_t at(int it, int i, int j) const { return std::size_t(it) * stride + std::size_t(j) * ld + i; }
};

template <Backend B, typename T>
Prob<T> make_prob(Queue& q, int m, int n, int batch, unsigned seed, int period = 0) {
    using R = RealOf<T>;
    Prob<T> p;
    p.m = m, p.n = n, p.batch = batch, p.period = period;
    p.ld = std::max(1, m + 3);
    p.stride = p.ld * std::max(1, n) + 5;
    p.mem = UnifiedVector<T>(std::size_t(p.stride) * std::max(1, batch) + 7, poison<T>());
    p.tau = UnifiedVector<T>(std::size_t(std::max(1, p.k())) * std::max(1, batch) + 4, poison<T>());
    p.ptrs = UnifiedVector<T*>(std::max(1, batch), nullptr);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> u(R(-1), R(1));
    for (int it = 0; it < batch; ++it)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) p.mem[p.at(it, i, j)] = mk<T>(u(gen), u(gen));
    p.a0.assign(p.mem.begin(), p.mem.end());
    // A wide A (n > m) has no vendor-free geqrf; only a refused pin or the vendor ever sees one.
    if (m >= 1 && n >= 1 && batch >= 1 && (n <= m || batchlas::select::factorization_vendor_available<B>)) {
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, geqrf_buffer_size<B, T>(q, p.A(), p.tau.to_span())));
        (void)geqrf<B, T>(q, p.A(), p.tau.to_span(), ws.to_span());
        q.wait();
    }
    if (period > 0)
        for (int it = period; it < batch; ++it) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) {
                    p.mem[p.at(it, i, j)] = p.mem[p.at(it % period, i, j)];
                    p.a0[p.at(it, i, j)] = p.a0[p.at(it % period, i, j)];
                }
            for (int r = 0; r < p.k(); ++r) p.tau[std::size_t(it) * p.k() + r] = p.tau[std::size_t(it % period) * p.k() + r];
        }
    p.f.assign(p.mem.begin(), p.mem.end());
    p.tau0.assign(p.tau.begin(), p.tau.end());
    return p;
}

template <typename T>
double tol(int m) {
    return 64.0 * std::max(m, 1) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

// ||Q^H Q - I||_max and ||Q R - A0||_max / ||A0||_max for one item, in double (R = triu(F)).
template <typename T>
std::pair<double, double> q_errors(const Prob<T>& p, int it) {
    const int m = p.m, n = p.n;
    double orth = 0, rec = 0, an = 0;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            std::complex<double> s = 0;
            for (int i = 0; i < m; ++i) s += std::conj(up(p.mem[p.at(it, i, a)])) * up(p.mem[p.at(it, i, b)]);
            orth = std::max(orth, std::abs(s - (a == b ? 1.0 : 0.0)));
        }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            std::complex<double> s = 0;
            for (int l = 0; l <= j; ++l) s += up(p.mem[p.at(it, i, l)]) * up(p.f[p.at(it, l, j)]);
            rec = std::max(rec, std::abs(s - up(p.a0[p.at(it, i, j)])));
            an = std::max(an, std::abs(up(p.a0[p.at(it, i, j)])));
        }
    return {orth, rec / std::max(an, 1e-300)};
}

// Q's checked items, every element outside A's footprint and all of tau bit for bit, and for a
// repeating batch every item bit-identical to its representative.
template <typename T>
void expect_q(const Prob<T>& p, const std::string& what) {
    std::vector<int> items;
    if (p.period > 0) for (int it = 0; it < std::min(p.period, p.batch); ++it) items.push_back(it);
    else if (p.batch <= 4) for (int it = 0; it < p.batch; ++it) items.push_back(it);
    else items = {0, 1, p.batch / 2, p.batch - 1};
    for (int it : items) {
        const auto [orth, rec] = q_errors(p, it);
        ASSERT_TRUE(std::isfinite(orth) && orth <= tol<T>(p.m)) << what << " item " << it << " |Q^H Q - I| " << orth;
        ASSERT_TRUE(std::isfinite(rec) && rec <= tol<T>(p.m)) << what << " item " << it << " |QR - A| " << rec;
    }
    std::vector<char> in(p.mem.size(), 0);
    for (int it = 0; it < p.batch; ++it)
        for (int j = 0; j < p.n; ++j)
            for (int i = 0; i < p.m; ++i) in[p.at(it, i, j)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!in[e]) ASSERT_TRUE(same_bits(p.mem[e], p.f[e])) << what << ": wrote outside A at element " << e;
    for (std::size_t e = 0; e < p.tau.size(); ++e)
        ASSERT_TRUE(same_bits(p.tau[e], p.tau0[e])) << what << ": wrote tau element " << e;
    if (p.period > 0)
        for (int it = p.period; it < p.batch; ++it)
            for (int j = 0; j < p.n; ++j)
                for (int i = 0; i < p.m; ++i)
                    ASSERT_TRUE(same_bits(p.mem[p.at(it, i, j)], p.mem[p.at(it % p.period, i, j)]))
                        << what << ": item " << it << " differs from its representative at (" << i << "," << j << ")";
}

std::string label(int m, int n, int batch) {
    return "m=" + std::to_string(m) + " n=" + std::to_string(n) + " batch=" + std::to_string(batch);
}
std::string name(const C& c, int m, int n, int batch) { return select::to_string(c) + " " + label(m, n, batch); }

// The outermost orgqr trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("orgqr ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no orgqr trace line in: " + err + ">";
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

// The old predicate (route_orgqr.hh at 424a45bc): native to 512 on both extents where the
// native driver runs, the vendor elsewhere; vendor-free, native wherever it runs.
std::string old_first(int m, int n, bool vendor) {
    const bool runs = n <= m && m >= 1 && n >= 1;
    if (!vendor) return runs ? "blocked" : "none";
    return runs && m <= 512 && n <= 512 ? "blocked" : "vendor";
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class OrgqrCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MVof<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::factorization_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native driver is a GPU kernel";
    }

    Prob<T> prob(int m, int n, int batch, unsigned seed, int period = 0) {
        return make_prob<B, T>(*this->ctx, m, n, batch, seed, period);
    }

    // ---- the limit oracle: orgqr_blocked_dispatch's own checks, not orgqr.cc's can_run ----
    static bool expect_runs(const C& c, int m, int n, int batch, bool het = false) {
        if (std::holds_alternative<oq::Blocked>(c))
            return sycl_orgqr::orgqr_blocked_available<T>() && m >= 1 && n >= 1 && batch >= 1 && n <= m && !het;
        return kVendor;
    }
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<oq::Vendor>(c) && !kVendor; }

    void run(Prob<T>& p) {
        const std::size_t bytes = orgqr_buffer_size<B, T>(*this->ctx, p.A(), p.tau.to_span());
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, bytes));
        (void)orgqr<B, T>(*this->ctx, p.A(), p.tau.to_span(), ws.to_span());
        this->ctx->wait();
    }
    void run_pinned(const C& c, Prob<T>& p) {
        const Pin pin("orgqr", c);
        run(p);
    }
    std::string auto_choice(Prob<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_ORGQR_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }
    // Acceptance through the sizing call, which runs the same choose(); nothing is launched.
    bool pin_accepted(const C& c, const MV& A, Span<T> tau) {
        const Pin pin("orgqr", c);
        try {
            (void)orgqr_buffer_size<B, T>(*this->ctx, A, tau);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }

    static Event apply(Queue& c, const MV& oa, const MV& oc, Side os, Transpose ot, Span<T> otau,
                       Span<std::byte> ows, int32_t obs) {
        return ormqr<B, T>(c, oa, oc, os, ot, otau, ows, obs);
    }
    static std::size_t apply_size(Queue& c, const MV& oa, const MV& oc, Side os, Transpose ot, Span<T> otau,
                                  int32_t obs) {
        return ormqr_buffer_size<B, T>(c, oa, oc, os, ot, otau, obs);
    }

    // The family's own driver: Blocked = orgqr_blocked_dispatch with the public ormqr, Vendor =
    // the library loop.
    bool direct(const C& c, const MV& A, Span<T> tau, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        try {
            if (std::holds_alternative<oq::Blocked>(c)) {
                const std::size_t bytes = sycl_orgqr::orgqr_blocked_buffer_size<T>(q, A, tau, apply_size);
                UnifiedVector<std::byte> ws(std::max<std::size_t>(1, bytes));
                (void)sycl_orgqr::orgqr_blocked_dispatch<T>(q, A, tau, ws.to_span(), apply, apply_size);
            } else {
                if constexpr (kVendor) {
                    UnifiedVector<std::byte> ws(
                        std::max<std::size_t>(1, backend::orgqr_vendor_buffer_size<B, T>(q, A, tau)));
                    (void)backend::orgqr_vendor<B, T>(q, A, tau, ws.to_span());
                } else {
                    if (why) *why = "no vendor factorization library in this build";
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

TYPED_TEST_SUITE(OrgqrCandidates, Types);

// §8.1: each candidate across the n <= m diagonal (24/24, 24/25), the native block sizes
// (16/32), the old 512 window on both extents, and a degenerate extent. Accepted shapes run and
// produce Q; refused ones throw from the pin. The vendor is never launched above the diagonal.
TYPED_TEST(OrgqrCandidates, PinnedCandidatesStraddleTheirLimits) {
    int ran = 0, refused = 0;
    const std::tuple<int, int, int> shapes[] = {{1, 1, 3},   {16, 16, 3},  {33, 17, 2}, {24, 24, 3}, {24, 25, 2},
                                                {40, 24, 3}, {7, 40, 2},   {512, 64, 1}, {513, 33, 1},
                                                {300, 300, 1}, {16, 0, 2}, {16, 8, 0}};
    for (const C& c : oq::candidates<typename TestFixture::T>())
        for (auto [m, n, batch] : shapes) {
            const std::string what = name(c, m, n, batch);
            if (TestFixture::vendor_word_falls_back(c)) continue;
            const bool runs = this->expect_runs(c, m, n, batch);
            auto p = this->prob(m, n, batch, 11u + m + n);
            if (!runs) {
                EXPECT_FALSE(this->pin_accepted(c, p.A(), p.tau.to_span())) << what;
                ++refused;
                continue;
            }
            if (n > m || n < 1 || batch < 1) continue;  // the vendor's own business, untested here
            ASSERT_NO_THROW(this->run_pinned(c, p)) << what;
            expect_q(p, what);
            ++ran;
        }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// Each pin is its family's own driver, bit for bit, on a non-natural ld and stride.
TYPED_TEST(OrgqrCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    for (const C& c : oq::candidates<T>())
        for (auto [m, n] : {std::pair{40, 24}, std::pair{65, 65}, std::pair{200, 70}}) {
            if (!this->expect_runs(c, m, n, 3)) continue;
            auto a = this->prob(m, n, 3, 23u + m);
            auto b = this->prob(m, n, 3, 23u + m);
            this->run_pinned(c, a);
            std::string why;
            ASSERT_TRUE(this->direct(c, b.A(), b.tau.to_span(), &why)) << name(c, m, n, 3) << ": " << why;
            for (std::size_t e = 0; e < a.mem.size(); ++e)
                ASSERT_TRUE(same_bits(a.mem[e], b.mem[e])) << name(c, m, n, 3) << ": differs at element " << e;
            expect_q(a, name(c, m, n, 3));
        }
}

// A saturating batch of 1024 identical items: every item bit-identical to item 0, pinned and Auto.
TYPED_TEST(OrgqrCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    std::vector<std::optional<C>> pins{std::nullopt, C{oq::Blocked{}}};
    for (const auto& c : pins) {
        auto p = this->prob(48, 32, 1024, 31u, /*period=*/1);
        const std::string what = (c ? select::to_string(*c) : std::string("auto")) + " " + label(48, 32, 1024);
        if (c) this->run_pinned(*c, p);
        else EXPECT_EQ(this->auto_choice(p), "blocked") << what;
        expect_q(p, what);
    }
}

// R3: can_run equals launch. For every candidate and shape, the pin is accepted exactly when the
// direct driver launches, and both agree with the oracle (the diagonal, degenerate extents and a
// heterogeneous batch). The vendor is compared below the diagonal only.
TYPED_TEST(OrgqrCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    int disagreements = 0;
    for (const C& c : oq::candidates<T>())
        for (auto [m, n, batch, het] : {std::tuple{8, 8, 2, false}, std::tuple{8, 9, 2, false}, std::tuple{30, 1, 2, false},
                                        std::tuple{1, 30, 2, false}, std::tuple{24, 16, 4, true}, std::tuple{8, 0, 2, false},
                                        std::tuple{0, 0, 2, false}, std::tuple{8, 8, 0, false}}) {
            if (TestFixture::vendor_word_falls_back(c)) continue;
            if (std::holds_alternative<oq::Vendor>(c) && (n > m || n < 1 || batch < 1 || het)) continue;
            auto a = this->prob(m, n, batch, 61u + m);
            auto b = this->prob(m, n, batch, 61u + m);
            UnifiedVector<int> rows(std::max(1, batch)), cols(std::max(1, batch));
            for (int i = 0; i < batch; ++i) rows[i] = m - (i % 2), cols[i] = n - (i % 2);
            const auto view = [&](Prob<T>& p) { return het ? p.A().with_active_dims(rows.to_span(), cols.to_span()) : p.A(); };
            const std::string what = name(c, m, n, batch) + (het ? " heterogeneous" : "");
            const bool pin = this->pin_accepted(c, view(a), a.tau.to_span());
            std::string why;
            const bool run = this->direct(c, view(b), b.tau.to_span(), &why);
            EXPECT_EQ(pin, run) << what << ": can_run says " << pin << ", the driver "
                                << (run ? "launches" : "refuses: " + why);
            EXPECT_EQ(pin, this->expect_runs(c, m, n, batch, het)) << what << ": can_run disagrees with the oracle";
            disagreements += pin != run;
        }
    EXPECT_EQ(disagreements, 0);
}

// R5: exactly orgqr_buffer_size bytes inside a poisoned arena; nothing past the end is written.
// A native call is never sized by the batch-linear vendor loop (the orgqr_buffer_size defect).
TYPED_TEST(OrgqrCandidates, ExactWorkspaceInAPoisonedArena) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    constexpr std::size_t kGuard = 4096;
    for (const C& c : oq::candidates<T>())
        for (auto [m, n] : {std::pair{1, 1}, std::pair{33, 20}, std::pair{130, 130}}) {
            if (!this->expect_runs(c, m, n, 5)) continue;
            auto p = this->prob(m, n, 5, 99u + m);
            const Pin pin("orgqr", c);
            const std::size_t bytes = orgqr_buffer_size<B, T>(*this->ctx, p.A(), p.tau.to_span());
            if (std::holds_alternative<oq::Blocked>(c)) {
                const std::size_t own = sycl_orgqr::orgqr_blocked_buffer_size<T>(
                    *this->ctx, p.A(), p.tau.to_span(), TestFixture::apply_size);
                EXPECT_EQ(bytes, own) << name(c, m, n, 5) << ": not the native driver's own size";
            }
            UnifiedVector<std::byte> arena(bytes + kGuard);
            std::memset(arena.data(), 0xFF, bytes);
            std::memset(arena.data() + bytes, 0xA5, kGuard);
            const std::string what = name(c, m, n, 5) + " bytes=" + std::to_string(bytes);
            ASSERT_NO_THROW(((void)orgqr<B, T>(*this->ctx, p.A(), p.tau.to_span(), Span<std::byte>(arena.data(), bytes)),
                             this->ctx->wait()))
                << what;
            for (std::size_t i = 0; i < kGuard; ++i)
                ASSERT_EQ(static_cast<unsigned>(arena[bytes + i]), 0xA5u) << what << ": wrote " << i << " bytes past";
            expect_q(p, what);
        }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(OrgqrCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "blocked:nb=32", "blocked:16", "native:vendor", "cta", "native:cta",
                             "vendor:blocked", "native:tiny",
                             // removed aliases (phase 5): each must stay an error
                             "native:blocked", "vendor:auto"}) {
        auto p = this->prob(16, 8, 2, 5u);
        const Pin pin("orgqr", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
    for (const C& c : oq::candidates<T>()) {
        if (!this->expect_runs(c, 16, 8, 2)) continue;
        auto p = this->prob(16, 8, 2, 5u);
        EXPECT_NO_THROW(this->run_pinned(c, p)) << select::to_string(c);
    }
}

// The named can_run-false cases (RouteOrgqr.CorrectnessGatesIncludeTheOnesInheritedFromOrmqr,
// ported): n > m, an empty extent, batch 0 and a heterogeneous batch refuse a blocked pin from
// both orgqr and orgqr_buffer_size; batch 1 and 8192 x 8192 are not correctness gates.
TYPED_TEST(OrgqrCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const C blocked{oq::Blocked{}};
    struct Case { const char* what; int m, n, batch; bool het; };
    for (const Case& k : {Case{"n > m", 128, 256, 2, false}, Case{"n = 0", 16, 0, 2, false},
                          Case{"m = n = 0", 0, 0, 2, false}, Case{"batch 0", 16, 8, 0, false},
                          Case{"heterogeneous", 32, 16, 4, true}}) {
        UnifiedVector<T> a(std::size_t(std::max(1, k.m)) * std::max(1, k.n) * std::max(1, k.batch), T(1));
        UnifiedVector<T> tau(std::size_t(std::max(1, std::min(k.m, k.n))) * std::max(1, k.batch), T(0));
        MVof<T> A(a.data(), k.m, k.n, std::max(1, k.m), std::max(1, k.m * k.n), k.batch);
        UnifiedVector<int> rows(std::max(1, k.batch)), cols(std::max(1, k.batch));
        for (int i = 0; i < k.batch; ++i) rows[i] = k.m - i, cols[i] = k.n - i;
        if (k.het) A = A.with_active_dims(rows.to_span(), cols.to_span());
        const Pin pin("orgqr", blocked);
        UnifiedVector<std::byte> ws(1 << 20);
        for (bool sizing : {true, false}) {
            try {
                if (sizing) (void)orgqr_buffer_size<B, T>(*this->ctx, A, tau.to_span());
                else (void)orgqr<B, T>(*this->ctx, A, tau.to_span(), ws.to_span());
                this->ctx->wait();
                ADD_FAILURE() << k.what << (sizing ? " (sizing)" : " (call)") << " was accepted";
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << k.what << ": " << e.what();
            }
        }
    }
    for (auto [m, n, batch] : {std::tuple{64, 64, 1}, std::tuple{8192, 8192, 1}, std::tuple{8, 8, 8192}}) {
        UnifiedVector<T> tau(std::size_t(n) * batch, T(0));
        const MVof<T> A(nullptr, m, n, m, m * n, batch);
        EXPECT_TRUE(this->pin_accepted(blocked, A, tau.to_span())) << label(m, n, batch);
    }
}

// A heterogeneous batch: no native family can run it (one identity, one apply), so Auto takes the
// vendor at the full storage order, as before; vendor-free there is no route.
TYPED_TEST(OrgqrCandidates, HeterogeneousBatchHasNoNativeRoute) {
    using T = typename TestFixture::T;
    auto p = this->prob(32, 16, 4, 71u);
    UnifiedVector<int> rows(4), cols(4);
    for (int i = 0; i < 4; ++i) rows[i] = 32 - i, cols[i] = 16 - i;
    const auto het = p.A().with_active_dims(rows.to_span(), cols.to_span());
    ASSERT_TRUE(het.is_heterogeneous());
    EXPECT_FALSE(this->pin_accepted(C{oq::Blocked{}}, het, p.tau.to_span()));
    const ScopedEnvVar clear("BATCHLAS_ORGQR_ROUTE", nullptr);
    auto call = [&] {
        const std::size_t bytes = orgqr_buffer_size<TestFixture::B, T>(*this->ctx, het, p.tau.to_span());
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, bytes));
        (void)orgqr<TestFixture::B, T>(*this->ctx, het, p.tau.to_span(), ws.to_span());
        this->ctx->wait();
    };
    if constexpr (TestFixture::kVendor) EXPECT_EQ(traced_choice(call), "vendor");
    else EXPECT_THROW(call(), batchlas::NoRouteError);
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment.
// `vendor:auto` (an old alias) throws; bare `vendor` falls back vendor-free.
TYPED_TEST(OrgqrCandidates, ClassWordsAndSpellings) {
    const ScopedEnvVar clear("BATCHLAS_ORGQR_ROUTE", nullptr);
    const bool v = TestFixture::kVendor;
    const std::pair<const char*, const char*> expect[] = {
        {"BLOCKED", "blocked"}, {"blocked", "blocked"}, {"Blocked", "blocked"},
        {"native", "blocked"},         {"auto", "blocked"},           {"vendor", v ? "vendor" : "blocked"},
        {"vendor:auto", nullptr}};
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = this->prob(40, 24, 3, 81u);
            select::testing::reset_warnings();
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_ORGQR_ROUTE" : " via ScopedPin");
            auto call = [&] {
                const ScopedEnvVar env("BATCHLAS_ORGQR_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("orgqr", std::string_view(word));
                this->run(p);
            };
            if (!spelling) {
                EXPECT_THROW(call(), std::invalid_argument) << what;
                continue;
            }
            std::string err;
            EXPECT_EQ(traced_choice(call, &err), spelling) << what;
            const bool warns = std::string(word) == "vendor" && !v;
            EXPECT_EQ(err.find("orgqr pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                << what << ": " << err;
            expect_q(p, what);
        }
}

// §5.3: a ScopedPin wins over BATCHLAS_ORGQR_ROUTE, and nested pins restore the outer one.
TYPED_TEST(OrgqrCandidates, ScopedPinBeatsTheEnvironment) {
    const ScopedEnvVar env("BATCHLAS_ORGQR_ROUTE", "blocked");
    {
        auto p = this->prob(24, 25, 2, 91u);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = this->prob(40, 24, 2, 92u);
    if constexpr (TestFixture::kVendor) {
        EXPECT_EQ(traced_choice([&] {
                      const Pin pin("orgqr", C{oq::Vendor{}});
                      this->run(p);
                  }),
                  "vendor");
        expect_q(p, "vendor over env blocked");
        const Pin outer("orgqr", C{oq::Vendor{}});
        {
            const Pin inner("orgqr", C{oq::Blocked{}});
            auto r = this->prob(40, 24, 2, 93u);
            EXPECT_EQ(traced_choice([&] { this->run(r); }), "blocked");
        }
        auto r = this->prob(40, 24, 2, 94u);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "vendor") << "the inner pin did not restore the outer";
        expect_q(r, "outer vendor");
    } else {
        const Pin over("orgqr", std::string_view("auto"));
        EXPECT_EQ(traced_choice([&] { this->run(p); }), "blocked");
        expect_q(p, "auto over env blocked");
    }
}

// Auto against the transcribed table this device reads (sm_89 or sm_120, the same rows): the old
// 512 ceiling straddled on each extent, on and off the grid (RouteOrgqr.PreferredIsNative-
// UpToTheMeasuredCeiling and VendorFreeFallbackHandsOverTheNativeRoute, ported, live).
TYPED_TEST(OrgqrCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    const auto tables = select::tables_in_borrow_order("orgqr", select::dtype_name<T>(),
                                                       select::device_of<TestFixture::B>(*this->ctx));
    if (tables.empty() || (tables.front()->device != "sm_89" && tables.front()->device != "sm_120"))
        GTEST_SKIP() << "this device reads no transcribed orgqr table";
    struct Row { int m, n, batch; };
    const Row rows[] = {{1, 1, 3},     {96, 96, 2},  {200, 150, 2}, {512, 512, 1}, {512, 40, 1},
                        {513, 40, 1},  {520, 8, 1},  {700, 33, 1},  {513, 513, 1}, {530, 500, 1}};
    for (const Row& r : rows) {
        auto p = this->prob(r.m, r.n, r.batch, 41u + r.m);
        const std::string want = old_first(r.m, r.n, TestFixture::kVendor);
        EXPECT_EQ(this->auto_choice(p), want) << label(r.m, r.n, r.batch);
        expect_q(p, "auto " + label(r.m, r.n, r.batch));
    }
}

// key_of's both fields reach choose(): a synthetic table whose winner changes with m alone and
// with n alone. Fixing a field in key_of turns exactly its row red. Needs both candidates runnable.
TYPED_TEST(OrgqrCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!TestFixture::kVendor) GTEST_SKIP() << "vendor-free: one runnable candidate cannot show a key";
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("orgqr.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("orgqr." + dtype + "." + dev + ".txt",
                       "# op=orgqr dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: m:log n:log:2\n"
                       "m=64 n=64 | blocked 1 | vendor 2\n"
                       "m=256 n=64 | vendor 1 | blocked 2\n"
                       "m=64 n=16 | vendor 1 | blocked 2\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { int m, n; const char* expect; const char* field; };
    for (const Probe& k : {Probe{64, 64, "blocked", "base"}, Probe{256, 64, "vendor", "m"},
                           Probe{64, 16, "vendor", "n"}}) {
        auto p = this->prob(k.m, k.n, 2, 43u);
        EXPECT_EQ(this->auto_choice(p), k.expect) << "the " << k.field << " row";
        expect_q(p, std::string("the ") + k.field + " row");
    }
}

// The trace line carries the key the table lookup used, plus batch.
TYPED_TEST(OrgqrCandidates, TraceKeyIsMAndN) {
    auto p = this->prob(37, 12, 5, 47u);
    const std::string line = traced_line([&] { this->run(p); });
    EXPECT_NE(line.find("m=37 n=12 batch=5 ->"), std::string::npos) << line;
    expect_q(p, line);
}

// RouteOrgqr.VendorFreeFallbackHandsOverTheNativeRoute's other half: with a table naming only
// the vendor, a vendor-free build takes the last resort, blocked, at n = 64 and above the old
// ceiling; a vendor build takes the vendor.
TYPED_TEST(OrgqrCandidates, VendorFreeLastResortIsBlocked) {
    using T = typename TestFixture::T;
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("orgqr.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("orgqr." + dtype + "." + dev + ".txt",
                       "# op=orgqr dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: m:log n:log:2\nm=64 n=64 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    for (auto [m, n] : {std::pair{64, 64}, std::pair{600, 520}}) {
        auto p = this->prob(m, n, 1, 53u);
        std::string all;
        const ScopedEnvVar clear("BATCHLAS_ORGQR_ROUTE", nullptr);
        const std::string got = traced_choice([&] { this->run(p); }, &all);
        EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : "blocked") << label(m, n, 1);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_q(p, "vendor-only table, " + label(m, n, 1));
    }
}

// The coverage row (§5.6): the backend, the old builder's key (m, n, k = min, Left, NoTrans) and
// the native flags. threadsafe: the child re-executes the binary, so CUDA is initialised fresh.
TYPED_TEST(OrgqrCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "orgqr_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_ORGQR_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            auto p = this->prob(40, 24, 2, 57u);
            const Pin pin("orgqr", C{oq::Blocked{}});
            this->run(p);
        }
        if constexpr (TestFixture::kVendor) {
            auto p = this->prob(48, 16, 2, 58u);
            const Pin pin("orgqr", C{oq::Vendor{}});
            this->run(p);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,orgqr,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), TestFixture::kVendor ? 2u : 1u);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    std::vector<std::tuple<std::string, const char*, const char*>> want{{"40 24 24", "native", "blocked"}};
    if (TestFixture::kVendor) want.emplace_back("48 16 16", "vendor", "vendor");
    for (const auto& [key, origin, algo] : want) {
        ASSERT_TRUE(rows.count(key)) << "no " << key << " row";
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key << ": the backend column";
        EXPECT_EQ(f[9], origin) << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[16], as_int(Side::Left)) << key;
        EXPECT_EQ(f[18], as_int(Transpose::NoTrans)) << key;
    }
}

// RouteOrgqr's CPU case, ported: on a CPU queue the native driver cannot run, so a blocked pin
// throws and Auto is the vendor (or no route without one).
template <typename Config>
class OrgqrCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(OrgqrCandidatesCpu, Types);

TYPED_TEST(OrgqrCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    UnifiedVector<T> a(16 * 8 * 2, T(1)), tau(8 * 2, T(0));
    const MVof<T> A(a.data(), 16, 8, 16, 128, 2);
    {
        const Pin pin("orgqr", C{oq::Blocked{}});
        EXPECT_THROW(((void)orgqr_buffer_size<B, T>(*this->ctx, A, tau.to_span())), std::invalid_argument);
    }
    const ScopedEnvVar clear("BATCHLAS_ORGQR_ROUTE", nullptr);
    if constexpr (batchlas::select::factorization_vendor_available<B>)
        EXPECT_NO_THROW(((void)orgqr_buffer_size<B, T>(*this->ctx, A, tau.to_span())));
    else
        EXPECT_THROW(((void)orgqr_buffer_size<B, T>(*this->ctx, A, tau.to_span())), batchlas::NoRouteError);
}

// The transcribed rows, read with Table::nearest directly so every device checks them: at grid
// and off-grid (m, n) below the diagonal, the first entry is the old predicate's choice on both
// devices, and every row ranks both candidates.
TEST(OrgqrTranscribedTable, RowsHoldTheOldWindowOnBothDevices) {
    const std::pair<int, int> points[] = {{1, 1},     {96, 96},    {300, 299},  {511, 511}, {512, 512},
                                          {513, 1},   {513, 513},  {514, 300},  {600, 512}, {640, 513},
                                          {1000, 10}, {2000, 600}, {9000, 9000}, {450, 449}, {512, 384}};
    for (const char* dev : {"sm_89", "sm_120"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const auto tables = select::tables_in_borrow_order("orgqr", dt, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << dt;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << dt;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            for (auto [m, n] : points) {
                const select::TableRow* row = t.nearest(select::Key{{"m", m}, {"n", n}});
                ASSERT_NE(row, nullptr) << t.file;
                const std::string what = t.file + ":" + std::to_string(row->line) + " for " + label(m, n, 0);
                ASSERT_EQ(row->ranked.size(), 2u) << what;
                EXPECT_EQ(row->ranked.front().spelling, old_first(m, n, true)) << what;
                EXPECT_FALSE(row->timed) << what;
            }
        }
}

}  // namespace
