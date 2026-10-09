// Every getrf candidate, pinned: docs/design/flat-kernel-selection.md §8 (phase 5). The limit
// oracle reads the drivers' own capacity queries; which kernel ran is read back from the select
// trace or a bit-for-bit comparison with the direct driver, never assumed from the pin.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/trsm.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include <batchlas/verify/residuals.hh>

#include "../src/extensions/getrf_native.hh"
#include "../src/ops/getrf/choice.hh"
#include "../src/util/resident_capacity.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace gf = batchlas::ops::getrf;
using C = gf::GetrfChoice;
using Pin = select::ScopedPin<C>;

template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
constexpr bool kCx = !std::is_same_v<T, RealOf<T>>;

template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// A batch with a padded ld, a stride that is not ld*n and poisoned padding: a large finite value
// any kernel would accept, so a kernel that reads or writes the pad shows it.
template <typename T>
struct Lu {
    int n = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> buf;
    std::vector<T> a0;
    UnifiedVector<T*> ptrs;
    UnifiedVector<int64_t> piv;
    UnifiedVector<int32_t> info;
    MVof<T> A() { return MVof<T>(buf.data(), n, n, ld, stride, batch, ptrs.data()); }
    const int* ip(int b) const { return reinterpret_cast<const int*>(piv.data()) + std::size_t(b) * n; }
};

// Column-dominant rows, cyclically shifted so every column pivots off the diagonal; complex
// entries carry an imaginary part. `period` > 0 repeats item it % period (saturating batches).
template <typename T>
Lu<T> make_lu(int n, int batch, unsigned seed, int period = 0) {
    Lu<T> p;
    p.n = n, p.batch = batch, p.ld = n + 3, p.stride = p.ld * std::max(n, 1) + 5;
    p.buf = UnifiedVector<T>(std::size_t(p.stride) * std::max(batch, 1), verify::make<T>(-9.75e3, 4.5e3));
    p.ptrs = UnifiedVector<T*>(std::size_t(std::max(batch, 1)), nullptr);
    p.piv = UnifiedVector<int64_t>(std::max<std::size_t>(std::size_t(n) * batch, 1), int64_t(0x0BADBEEF0BADBEEFLL));
    p.info = UnifiedVector<int32_t>(std::size_t(std::max(batch, 1)), int32_t(-12345));
    for (int b = 0; b < batch; ++b) {
        verify::Rng rng(seed + 977u * std::uint64_t(period > 0 ? b % period : b));
        auto next = [&] { return rng.next(); };
        for (int j = 0; j < n; ++j)
            for (int r = 0; r < n; ++r) {
                const double re = next(), im = next();
                const double d = r == j ? 4.0 * n : 0.0;
                p.buf[std::size_t(b) * p.stride + std::size_t(j) * p.ld + (r + 1) % n] = verify::make<T>(re + d, im);
            }
    }
    p.a0.assign(p.buf.begin(), p.buf.end());
    return p;
}

// ||P A0 - L U||_F / ||A0||_F for one item, P rebuilt from the 1-based interchange list.
template <typename T>
double residual(const Lu<T>& p, int b) {
    const std::size_t off = std::size_t(b) * p.stride;
    return verify::getrf_residual(verify::view(p.a0.data() + off, p.n, p.n, p.ld), verify::view(p.buf.data() + off, p.n, p.n, p.ld),
                                  VectorView<int32_t>(const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.ip(b))), p.n, 1));
}

// info, pivots and the residual of the first and last item, and the padding bit for bit.
template <typename T>
void expect_factored(const Lu<T>& p, const std::string& what, bool info_passed = true) {
    for (int b = 0; info_passed && b < p.batch; ++b) ASSERT_EQ(p.info[b], 0) << what << " item " << b;
    for (int b : {0, p.batch - 1}) {
        for (int k = 0; k < p.n; ++k)
            ASSERT_TRUE(p.ip(b)[k] >= k + 1 && p.ip(b)[k] <= p.n) << what << " item " << b << " pivot " << k << " = " << p.ip(b)[k];
        const double r = residual(p, b);
        EXPECT_VERIFY(T, verify::Check::factorization, p.n, r) << what << " item " << b << " residual " << r;
    }
    for (std::size_t e = 0; e < p.a0.size(); ++e) {
        const int r = int(e % p.stride), i = r % p.ld, j = r / p.ld;
        if (!(i < p.n && j < p.n) || int(e / p.stride) >= p.batch)
            ASSERT_TRUE(same_bits(p.buf[e], p.a0[e])) << what << ": padding written at element " << e;
    }
}

// The choice the outermost getrf trace line names for whatever `run` calls.
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
        if (line.rfind("getrf ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no getrf trace line in: " + err + ">";
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
class GetrfCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MVof<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::factorization_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native tiers are GPU kernels";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    // ---- the limit oracle: each driver's own capacity query, not getrf.cc's can_run ----
    std::size_t budget() const {
        return resident::device_slm_budget(this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    }
    int max_wg() const {
        return static_cast<int>(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    }
    int tiny_n() const {
        return max_wg() >= sycl_getrf::kGetrfTinyWgSize ? sycl_getrf::getrf_tiny_max_n<T>() : 0;
    }
    int cta_n() const { return sycl_getrf::getrf_cta_max_n_for_slm<T>(budget()); }
    bool expect_runs(const C& c, int n) const {
        if (n < 1) return false;
        if (std::holds_alternative<gf::Tiny>(c)) return n <= tiny_n();
        if (std::holds_alternative<gf::Cta>(c)) return sycl_getrf::getrf_cta_fits<T>(n, budget());
        if (std::holds_alternative<gf::Blocked>(c)) return sycl_getrf::getrf_blocked_available<T>();
        return kVendor;
    }
    // The Vendor choice spells the class word `vendor`, which falls back to Auto where no vendor exists.
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<gf::Vendor>(c) && !kVendor; }

    void run(Lu<T>& p, bool pass_info = true) {
        const std::size_t bytes = getrf_buffer_size<B, T>(*this->ctx, p.A());
        UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
        (void)getrf<B, T>(*this->ctx, p.A(), p.piv.to_span(), Span<std::byte>(ws.data(), bytes),
                          pass_info ? p.info.to_span() : Span<int32_t>{});
        this->ctx->wait();
    }
    void run_pinned(const C& c, Lu<T>& p) {
        const Pin pin("getrf", c);
        run(p);
    }
    // getrf_buffer_size runs choose() and launches nothing.
    bool pin_accepted(const C& c, const MV& A) {
        const Pin pin("getrf", c);
        try {
            (void)getrf_buffer_size<B, T>(*this->ctx, A);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }
    std::string auto_choice(Lu<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_GETRF_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }

    // The family's own driver, with the public gemm and trsm for Blocked.
    bool direct(const C& c, Lu<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const MV A = p.A();
        try {
            std::size_t bytes = 0;
            if (std::holds_alternative<gf::Tiny>(c)) bytes = sycl_getrf::getrf_tiny_buffer_size<T>(q, A);
            if (std::holds_alternative<gf::Cta>(c)) bytes = sycl_getrf::getrf_cta_buffer_size<T>(q, A);
            if (std::holds_alternative<gf::Blocked>(c)) bytes = sycl_getrf::getrf_blocked_buffer_size<T>(q, A);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
            const Span<std::byte> w(ws.data(), bytes);
            if (std::holds_alternative<gf::Tiny>(c)) {
                (void)sycl_getrf::getrf_tiny_dispatch<T>(q, A, p.piv.to_span(), w, p.info.to_span());
            } else if (std::holds_alternative<gf::Cta>(c)) {
                (void)sycl_getrf::getrf_cta_dispatch<T>(q, A, p.piv.to_span(), w, p.info.to_span());
            } else if (std::holds_alternative<gf::Blocked>(c)) {
                (void)sycl_getrf::getrf_blocked_dispatch<T>(
                    q, A, p.piv.to_span(), w, p.info.to_span(),
                    [](Queue& c2, const MV& ga, const MV& gb, const MV& gc, T al, T be, Transpose ta, Transpose tb,
                       ComputePrecision pr) { return gemm<B, T>(c2, ga, gb, gc, al, be, ta, tb, pr); },
                    [](Queue& c2, const MV& ta, const MV& tb, T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
                        return trsm<B, T>(c2, ta, tb, al, sd, ul, tr, dg);
                    });
            } else {
                if (why) *why = "the vendor has no direct driver here";
                return false;
            }
            q.wait();
            return true;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return false;
        }
    }

    static std::string name(const C& c, int n, int batch) {
        return select::to_string(c) + " n=" + std::to_string(n) + " batch=" + std::to_string(batch);
    }
    std::vector<int> straddle() const {
        std::vector<int> ns{1, 4, 5, 8, 9, 16, 17, 24, 25, 32, 33, cta_n(), cta_n() + 1, 130};
        std::sort(ns.begin(), ns.end());
        ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
        return ns;
    }
};

TYPED_TEST_SUITE(GetrfCandidates, Types);

// §8.1: each candidate on orders straddling the tiny buckets (4/5, 8/9, 16/17), both tiny
// ceilings, the CTA ceiling and beyond. Accepted shapes run and are correct; refused ones throw.
TYPED_TEST(GetrfCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    for (const C& c : gf::candidates<T>()) {
        if (TestFixture::vendor_word_falls_back(c)) continue;
        for (int n : this->straddle()) {
            auto p = make_lu<T>(n, 3, 11u + n);
            const std::string what = TestFixture::name(c, n, 3);
            if (this->expect_runs(c, n)) {
                ASSERT_NO_THROW(this->run_pinned(c, p)) << what;
                expect_factored(p, what);
            } else {
                EXPECT_FALSE(this->pin_accepted(c, p.A())) << what;
                const Pin pin("getrf", c);
                UnifiedVector<std::byte> ws(1 << 16);
                EXPECT_THROW(((void)getrf<B, T>(*this->ctx, p.A(), p.piv.to_span(), ws.to_span(), p.info.to_span())),
                             std::invalid_argument)
                    << what;
            }
        }
    }
}

// The pinned public call runs exactly the family's driver: factor and pivots bit for bit.
TYPED_TEST(GetrfCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    for (const C& c : gf::candidates<T>()) {
        if (std::holds_alternative<gf::Vendor>(c)) continue;
        for (int n : {std::min(13, this->tiny_n()), std::min(40, this->cta_n()), 70}) {
            if (!this->expect_runs(c, n)) continue;
            auto a = make_lu<T>(n, 5, 23u + n);
            auto b = make_lu<T>(n, 5, 23u + n);
            this->run_pinned(c, a);
            std::string why;
            ASSERT_TRUE(this->direct(c, b, &why)) << TestFixture::name(c, n, 5) << ": " << why;
            for (std::size_t e = 0; e < a.buf.size(); ++e)
                ASSERT_TRUE(same_bits(a.buf[e], b.buf[e])) << TestFixture::name(c, n, 5) << " element " << e;
            for (std::size_t k = 0; k < std::size_t(n) * 5; ++k)
                ASSERT_EQ(a.ip(0)[k], b.ip(0)[k]) << TestFixture::name(c, n, 5) << " pivot " << k;
        }
    }
}

// A saturating batch of 1024 identical items: every item's factor and pivots equal item 0's.
TYPED_TEST(GetrfCandidates, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::T;
    for (const C& c : gf::candidates<T>()) {
        if (std::holds_alternative<gf::Vendor>(c)) continue;
        const int n = std::holds_alternative<gf::Tiny>(c) ? std::min(16, this->tiny_n())
                    : std::holds_alternative<gf::Cta>(c) ? std::min(24, this->cta_n()) : 48;
        if (!this->expect_runs(c, n)) continue;
        auto p = make_lu<T>(n, 1024, 71u, /*period=*/1);
        this->run_pinned(c, p);
        const std::string what = TestFixture::name(c, n, 1024);
        expect_factored(p, what);
        for (int it = 1; it < p.batch; ++it) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    ASSERT_TRUE(same_bits(p.buf[std::size_t(it) * p.stride + std::size_t(j) * p.ld + i],
                                          p.buf[std::size_t(j) * p.ld + i]))
                        << what << ": item " << it << " differs at (" << i << "," << j << ")";
            for (int k = 0; k < n; ++k) ASSERT_EQ(p.ip(it)[k], p.ip(0)[k]) << what << " item " << it;
        }
    }
}

// §8.2 (R3): on every straddling shape a native pin is accepted exactly when the driver launches.
TYPED_TEST(GetrfCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    int disagreements = 0;
    for (const C& c : gf::candidates<T>()) {
        if (std::holds_alternative<gf::Vendor>(c)) continue;
        std::vector<int> ns = this->straddle();
        ns.insert(ns.end(), {this->tiny_n(), this->tiny_n() + 1, this->cta_n() + 2, 0});
        for (int n : ns) {
            auto a = make_lu<T>(n, 2, 61u + n);
            auto b = make_lu<T>(n, 2, 61u + n);
            const bool pin = this->pin_accepted(c, a.A());
            std::string why;
            const bool run = this->direct(c, b, &why);
            EXPECT_EQ(pin, run) << TestFixture::name(c, n, 2) << ": can_run says " << pin << ", the driver "
                                << (run ? "launches" : "refuses: " + why);
            EXPECT_EQ(pin, this->expect_runs(c, n)) << TestFixture::name(c, n, 2) << ": can_run disagrees with the oracle";
            disagreements += pin != run;
        }
        // An empty batch: every native driver refuses it as degenerate extents.
        {
            auto a = make_lu<T>(8, 0, 67u);
            auto b = make_lu<T>(8, 0, 67u);
            const bool pin = this->pin_accepted(c, a.A());
            std::string why;
            const bool run = this->direct(c, b, &why);
            EXPECT_EQ(pin, run) << TestFixture::name(c, 8, 0) << ": the driver " << (run ? "launches" : why);
            EXPECT_FALSE(pin) << TestFixture::name(c, 8, 0);
            disagreements += pin != run;
        }
        // A non-square view: refused by the pin and by the driver alike.
        UnifiedVector<T> w(std::size_t(24) * 32, verify::make<T>(1.0, 0.5));
        UnifiedVector<T*> wp(1, nullptr);
        const MVof<T> W(w.data(), 24, 32, 24, 24 * 32, 1, wp.data());
        EXPECT_FALSE(this->pin_accepted(c, W)) << select::to_string(c) << " non-square";
    }
    EXPECT_EQ(disagreements, 0);
}

// One launch covers the batch with a single order: no native family takes a heterogeneous batch.
// Auto takes the vendor; vendor-free there is no route.
TYPED_TEST(GetrfCandidates, HeterogeneousBatchHasNoNativeRoute) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int n = 16, batch = 4;
    auto p = make_lu<T>(n, batch, 81u);
    UnifiedVector<int> act(batch);
    for (int b = 0; b < batch; ++b) act[b] = n - b;
    const MVof<T> het = p.A().with_active_dims(act.to_span(), act.to_span());
    ASSERT_TRUE(het.is_heterogeneous());
    for (const C& c : gf::candidates<T>()) {
        if (std::holds_alternative<gf::Vendor>(c)) continue;
        EXPECT_FALSE(this->pin_accepted(c, het)) << select::to_string(c);
    }
    const ScopedEnvVar clear("BATCHLAS_GETRF_ROUTE", nullptr);
    if constexpr (TestFixture::kVendor) {
        EXPECT_EQ(traced_choice([&] {
                      const std::size_t bytes = getrf_buffer_size<B, T>(*this->ctx, het);
                      UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
                      (void)getrf<B, T>(*this->ctx, het, p.piv.to_span(), Span<std::byte>(ws.data(), bytes),
                                        p.info.to_span());
                      this->ctx->wait();
                  }),
                  "vendor");
    } else {
        EXPECT_THROW(((void)getrf_buffer_size<B, T>(*this->ctx, het)), batchlas::NoRouteError);
    }
}

// The squareness gate (route_vocabulary's RouteGetrf.Squareness..., ported): a non-square getrf
// is the vendor's (an ordinary positional call), refused by every native pin; vendor-free, no route.
TYPED_TEST(GetrfCandidates, NonSquareIsTheVendorsOnly) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    UnifiedVector<T> w(std::size_t(24) * 32 * 2, verify::make<T>(1.0, 0.5));
    UnifiedVector<T*> wp(2, nullptr);
    const MVof<T> W(w.data(), 24, 32, 24, 24 * 32, 2, wp.data());
    const ScopedEnvVar clear("BATCHLAS_GETRF_ROUTE", nullptr);
    if constexpr (TestFixture::kVendor)
        EXPECT_EQ((getrf_buffer_size<B, T>(*this->ctx, W)), (backend::getrf_vendor_buffer_size<B, T>(*this->ctx, W)));
    else
        EXPECT_THROW(((void)getrf_buffer_size<B, T>(*this->ctx, W)), batchlas::NoRouteError);
}

// §8.3 (R5): exactly getrf_buffer_size bytes inside a larger arena whose tail is a guard. The
// workspace starts all-ones, and with an empty info span every family draws scratch from it.
TYPED_TEST(GetrfCandidates, ExactWorkspaceInAPoisonedArena) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    constexpr std::size_t kGuard = 4096;
    for (const C& c : gf::candidates<T>()) {
        if (TestFixture::vendor_word_falls_back(c)) continue;
        for (int n : {1, this->tiny_n(), this->cta_n(), 100}) {
            if (!this->expect_runs(c, n)) continue;
            for (bool pass_info : {true, false}) {
                auto p = make_lu<T>(n, 5, 99u + n);
                const Pin pin("getrf", c);
                const std::size_t bytes = getrf_buffer_size<B, T>(*this->ctx, p.A());
                UnifiedVector<std::byte> arena(bytes + kGuard);
                std::memset(arena.data(), 0xFF, bytes);
                std::memset(arena.data() + bytes, 0xA5, kGuard);
                const std::string what = TestFixture::name(c, n, 5) + " bytes=" + std::to_string(bytes) +
                                         (pass_info ? " info" : " no-info");
                ASSERT_NO_THROW(((void)getrf<B, T>(*this->ctx, p.A(), p.piv.to_span(),
                                                   Span<std::byte>(arena.data(), bytes),
                                                   pass_info ? p.info.to_span() : Span<int32_t>{}),
                                 this->ctx->wait()))
                    << what;
                for (std::size_t i = 0; i < kGuard; ++i)
                    ASSERT_EQ(static_cast<unsigned>(arena[bytes + i]), 0xA5u)
                        << what << ": wrote " << i << " bytes past the sized workspace";
                expect_factored(p, what, pass_info);
            }
        }
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(GetrfCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "tiny:1", "cta:nb=8", "native:vendor", "native:tiny:8", "lu", "native:auto",
                             // removed aliases (phase 5): each must stay an error
                             "native:tiny", "native:cta", "native:blocked"}) {
        auto p = make_lu<T>(8, 2, 5u);
        const Pin pin("getrf", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
    for (const C& c : gf::candidates<T>()) {
        if (!this->expect_runs(c, 8)) continue;
        auto p = make_lu<T>(8, 2, 5u);
        EXPECT_NO_THROW(this->run_pinned(c, p)) << select::to_string(c);
    }
}

// The named can_run-false cases, each with its message: one past each ceiling, n = 0 and batch = 0.
TYPED_TEST(GetrfCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    struct Case { C c; int n, batch; };
    const std::vector<Case> cases{{gf::Tiny{}, this->tiny_n() + 1, 1}, {gf::Cta{}, this->cta_n() + 1, 1},
                                  {gf::Tiny{}, 0, 1},  {gf::Cta{}, 0, 1},  {gf::Blocked{}, 0, 1},
                                  {gf::Tiny{}, 8, 0},  {gf::Cta{}, 8, 0},  {gf::Blocked{}, 8, 0}};
    for (const auto& [c, n, batch] : cases) {
        auto p = make_lu<T>(n, batch, 7u);
        const Pin pin("getrf", c);
        try {
            (void)getrf_buffer_size<B, T>(*this->ctx, p.A());
            ADD_FAILURE() << TestFixture::name(c, n, batch) << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                << TestFixture::name(c, n, batch) << ": " << e.what();
        }
    }
}

// The pivot-format gate (route_vocabulary's RouteLuPivotFormat getrf half, ported): the native
// tiers write packed int32 where netlib writes int64, so a NETLIB getrf runs no native family,
// even on a GPU queue.
TYPED_TEST(GetrfCandidates, NetlibBackendRunsNoNativeFamily) {
#if BATCHLAS_HAS_HOST_BACKEND
    using T = typename TestFixture::T;
    for (const C& c : gf::candidates<T>()) {
        if (std::holds_alternative<gf::Vendor>(c)) continue;
        auto p = make_lu<T>(16, 2, 9u);
        const Pin pin("getrf", c);
        EXPECT_THROW(((void)getrf_buffer_size<Backend::NETLIB, T>(*this->ctx, p.A())), std::invalid_argument)
            << select::to_string(c);
    }
#else
    GTEST_SKIP() << "no NETLIB backend in this build";
#endif
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment.
// Bare `native` is the row's best runnable non-vendor; bare `vendor` falls back to Auto
// (with the warning) where there is no vendor.
TYPED_TEST(GetrfCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const int n = std::min(16, this->tiny_n());
    std::string auto_pick;
    {
        auto p = make_lu<T>(n, 4, 3u);
        auto_pick = this->auto_choice(p);
    }
    ASSERT_TRUE(auto_pick == "tiny" || auto_pick == "cta" || auto_pick == "blocked" || auto_pick == "vendor")
        << auto_pick;
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    std::string native_pick = auto_pick;
    if (auto_pick == "vendor") {
        auto p = make_lu<T>(n, 4, 3u);
        const Pin pin("getrf", std::string_view("native"));
        native_pick = traced_choice([&] { this->run(p); });
        EXPECT_NE(native_pick, "vendor");
    }
    const std::pair<const char*, std::string> expect[] = {
        {"CTA", "cta"},
        {"Tiny", "tiny"},        {"vendor", vendor_pick}, {"native", native_pick},     {"auto", auto_pick}};
    for (const auto& [word, spelling] : expect) {
        for (bool via_env : {false, true}) {
            auto p = make_lu<T>(n, 4, 3u);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_GETRF_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("getrf", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_GETRF_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
            EXPECT_EQ(err.find("getrf pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                << what << ": " << err;
            expect_factored(p, what);
        }
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_GETRF_ROUTE, and nested pins restore the outer one.
TYPED_TEST(GetrfCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const int big = std::max(this->tiny_n(), this->cta_n()) + 1;
    const ScopedEnvVar env("BATCHLAS_GETRF_ROUTE", "tiny");
    {
        auto p = make_lu<T>(big, 2, 1u);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_lu<T>(big, 2, 1u);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("getrf", C{gf::Blocked{}});
                  this->run(p);
              }),
              "blocked");
    expect_factored(p, "blocked over env tiny");
    const Pin outer("getrf", C{gf::Cta{}});
    {
        const Pin inner("getrf", C{gf::Blocked{}});
        auto r = make_lu<T>(8, 2, 2u);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "blocked");
    }
    auto r = make_lu<T>(8, 2, 2u);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "cta") << "the inner pin did not restore the outer";
    expect_factored(r, "outer cta");
}

// Auto against the transcribed tables on the real device (getrf_tests' L12/T9 and the RouteGetrf
// windows, ported): the old tiny window, the old blocked windows on both of cfloat's axes, and
// the old vendor-free walk (tiny for the single types up to 32, cta for double up to 32, blocked
// above; cta wherever it fits otherwise). Off-grid orders and batches on purpose.
TYPED_TEST(GetrfCandidates, AutoReproducesTheOldRouterOnTheRealDevice) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const auto tables = select::tables_in_borrow_order("getrf", select::dtype_name<T>(), select::device_of<B>(*this->ctx));
    if (tables.empty() || (tables.front()->device != "sm_89" && tables.front()->device != "sm_120"))
        GTEST_SKIP() << "this device reads no transcribed getrf table";
    constexpr bool kF = std::is_same_v<T, float>, kCF = std::is_same_v<T, std::complex<float>>;
    constexpr bool kD = std::is_same_v<T, double>;
    const int cta = this->cta_n();
    auto old_free = [&](int n, int batch) -> std::string {  // route_getrf.hh's vendor-free walk
        if (kF && n >= 256) return "blocked";
        if (kCF && (n >= 512 || (n >= 256 && batch >= 256))) return "blocked";
        if ((kF || kCF) && n <= 32) return "tiny";
        if (kD && n > 32) return "blocked";
        if (n <= cta) return "cta";
        return n <= this->tiny_n() ? "tiny" : "blocked";
    };
    auto old_vendor = [&](int n, int batch) -> std::string {  // preferred(), else the vendor
        if (kF && n >= 5 && n <= 32) return "tiny";
        if (kCF && ((n >= 5 && n <= 7) || (n >= 9 && n <= 24))) return "tiny";
        if (kF && n >= 256) return "blocked";
        if (kCF && (n >= 512 || (n >= 256 && batch >= 256))) return "blocked";
        return "vendor";
    };
    const std::pair<int, int> cells[] = {{3, 3},    {4, 7},     {5, 3},     {7, 300},   {8, 2},    {9, 3},
                                         {13, 600}, {24, 5},    {25, 5},    {31, 1000}, {33, 3},   {40, 2},
                                         {200, 2},  {255, 300}, {256, 2},   {256, 256}, {300, 255}, {300, 257},
                                         {511, 2},  {512, 2},   {600, 3}};
    for (auto [n, batch] : cells) {
        auto p = make_lu<T>(n, batch, 41u + n);
        const std::string want = TestFixture::kVendor ? old_vendor(n, batch) : old_free(n, batch);
        EXPECT_EQ(this->auto_choice(p), want) << "n=" << n << " batch=" << batch;
        expect_factored(p, "auto n=" + std::to_string(n) + " batch=" + std::to_string(batch));
    }
}

// key_of's every field reaches choose(): a synthetic table for this device whose winner changes
// with n alone and with batch alone. Fixing a field in key_of turns exactly its row red.
TYPED_TEST(GetrfCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("getrf.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("getrf." + dtype + "." + dev + ".txt",
                       "# op=getrf dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: n:log:3 batch:log\n"
                       "n=8 batch=128 | tiny 1 | cta 2 | blocked 3\n"
                       "n=8 batch=32768 | blocked 1 | cta 2 | tiny 3\n"
                       "n=12 batch=128 | cta 1 | blocked 2 | tiny 3\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { int n, batch; const char* expect; const char* field; };
    const Probe probes[] = {{8, 128, "tiny", "base"}, {8, 30000, "blocked", "batch"}, {12, 128, "cta", "n"}};
    for (const auto& k : probes) {
        if (std::string(k.expect) == "tiny" && this->tiny_n() < 8) continue;
        auto p = make_lu<T>(k.n, k.batch, 43u);
        EXPECT_EQ(this->auto_choice(p), k.expect) << "the " << k.field << " row";
        expect_factored(p, std::string("the ") + k.field + " row");
    }
}

// A table naming only the vendor: a vendor-free build takes the last resort, blocked.
TYPED_TEST(GetrfCandidates, VendorFreeLastResortIsBlocked) {
    using T = typename TestFixture::T;
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("getrf.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("getrf." + dtype + "." + dev + ".txt",
                       "# op=getrf dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: n:log:3 batch:log\n"
                       "n=32 batch=128 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    for (int n : {8, 200}) {
        auto p = make_lu<T>(n, 3, 17u);
        std::string all;
        const ScopedEnvVar clear("BATCHLAS_GETRF_ROUTE", nullptr);
        const std::string got = traced_choice([&] { this->run(p); }, &all);
        EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : "blocked") << "n=" << n;
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_factored(p, "vendor-only table, n=" + std::to_string(n));
    }
}

// The coverage row (§5.6): the real backend, the key (m = n = k = order) and the native flags.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(GetrfCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "getrf_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_GETRF_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("getrf", C{gf::Cta{}});
            auto p = make_lu<T>(16, 2, 1u);
            this->run(p);
        }
        const Pin pin("getrf", C{gf::Blocked{}});
        auto p = make_lu<T>(48, 2, 2u);
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,getrf,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 14u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    for (const auto& [key, algo] : {std::pair{std::string("16 16 16"), "cta"}, std::pair{std::string("48 48 48"), "blocked"}}) {
        ASSERT_TRUE(rows.count(key)) << "no row " << key;
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key;
        EXPECT_EQ(f[9], "native") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
    }
}

// On a CPU queue no native family can run: native pins throw and Auto is the vendor (netlib),
// or there is no route without one.
template <typename Config>
class GetrfCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(GetrfCandidatesCpu, Types);

TYPED_TEST(GetrfCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_GETRF_ROUTE", nullptr);
    for (const C& c : gf::candidates<T>()) {
        if (std::holds_alternative<gf::Vendor>(c)) continue;
        auto p = make_lu<T>(8, 2, 3u);
        const Pin pin("getrf", c);
        EXPECT_THROW(((void)getrf_buffer_size<B, T>(*this->ctx, p.A())), std::invalid_argument) << select::to_string(c);
    }
    auto p = make_lu<T>(8, 2, 3u);
    if constexpr (batchlas::select::factorization_vendor_available<B>) {
        const std::size_t bytes = getrf_buffer_size<B, T>(*this->ctx, p.A());
        EXPECT_EQ(bytes, (backend::getrf_vendor_buffer_size<B, T>(*this->ctx, p.A())));
    } else {
        EXPECT_THROW(((void)getrf_buffer_size<B, T>(*this->ctx, p.A())), batchlas::NoRouteError);
    }
}

// The transcribed rows, read with Table::nearest directly so every device checks them, on both
// transcribed devices: each threshold of the old predicates from both sides.
TEST(GetrfTranscribedTable, RowsHoldTheOldPreferenceOnBothDevices) {
    struct Row { const char* dtype; int n, batch; const char* ranked; };
    const Row rows[] = {
        {"float", 4, 128, "vendor tiny cta blocked"},      {"float", 5, 128, "tiny vendor cta blocked"},
        {"float", 32, 8192, "tiny vendor cta blocked"},    {"float", 33, 8192, "vendor cta blocked"},
        {"float", 255, 2, "vendor cta blocked"},           {"float", 256, 2, "blocked vendor cta"},
        {"cfloat", 7, 512, "tiny vendor cta blocked"},     {"cfloat", 8, 512, "vendor tiny cta blocked"},
        {"cfloat", 9, 512, "tiny vendor cta blocked"},     {"cfloat", 24, 512, "tiny vendor cta blocked"},
        {"cfloat", 25, 512, "vendor tiny cta blocked"},    {"cfloat", 33, 512, "vendor cta blocked"},
        {"cfloat", 256, 255, "vendor cta blocked"},        {"cfloat", 256, 256, "blocked vendor cta"},
        {"cfloat", 511, 64, "vendor cta blocked"},         {"cfloat", 512, 64, "blocked vendor cta"},
        {"double", 32, 2048, "vendor cta tiny blocked"},   {"double", 33, 2048, "vendor blocked cta"},
        {"double", 1024, 32768, "vendor blocked cta"},     {"cdouble", 16, 128, "vendor cta tiny blocked"},
        {"cdouble", 17, 128, "vendor cta blocked"},        {"cdouble", 1024, 32768, "vendor cta blocked"},
    };
    for (const char* dev : {"sm_89", "sm_120"})
        for (const Row& r : rows) {
            const auto tables = select::tables_in_borrow_order("getrf", r.dtype, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << r.dtype;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << r.dtype;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            const select::TableRow* row = t.nearest(select::Key{{"n", r.n}, {"batch", r.batch}});
            ASSERT_NE(row, nullptr) << t.file;
            std::string got;
            for (const auto& e : row->ranked) got += (got.empty() ? "" : " ") + e.spelling;
            EXPECT_EQ(got, r.ranked) << t.file << ":" << row->line << " n=" << r.n << " batch=" << r.batch;
            EXPECT_FALSE(row->timed) << t.file;
        }
}

}  // namespace
