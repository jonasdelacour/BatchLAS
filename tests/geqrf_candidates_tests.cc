// Every geqrf candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-select-p5/geqrf.md. The limit oracle reads the drivers' own capacity
// queries; which kernel ran is read back from the select trace or a bit-for-bit comparison
// with the direct driver, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/geqrf_native.hh"
#include "../src/ops/geqrf/choice.hh"
#include "../src/ops/geqrf/geqrf.hh"

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
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace gq = batchlas::ops::geqrf;
using C = gq::GeqrfChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;
using D = std::complex<double>;

template <typename T>
T mk(double r, double i) {
    if constexpr (kCx<T>) return T(RealOf<T>(r), RealOf<T>(i));
    else return T(r);
}
template <typename T>
D up(T v) {
    if constexpr (kCx<T>) return {double(v.real()), double(v.imag())};
    else return {double(v), 0.0};
}
template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

struct Spec {
    int m = 8, n = 8, batch = 3;
    int period = 0;  // > 0: item it repeats item it % period
    unsigned seed = 1;
};

// A at a padded ld and a stride that is not ld * n, inside a buffer of large finite poison: a
// kernel that reads the pad gets a wrong answer, one that writes it is caught bit for bit.
template <typename T>
struct Qr {
    Spec s;
    int ld = 0, stride = 0;
    UnifiedVector<T> mem, tau;
    std::vector<T> mem0;
    UnifiedVector<T*> ptrs;  // the vendor geqrf is a pointer-array API

    MVof<T> A() { return {mem.data(), s.m, s.n, ld, stride, s.batch, ptrs.data()}; }
    std::size_t at(int it, int i, int j) const { return std::size_t(it) * stride + std::size_t(j) * ld + i; }
    int k() const { return std::min(s.m, s.n); }
};

template <typename T>
Qr<T> make_qr(const Spec& s) {
    Qr<T> p;
    p.s = s;
    p.ld = s.m + 3;
    p.stride = p.ld * s.n + 7;
    p.mem = UnifiedVector<T>(std::size_t(p.stride) * s.batch + 5, mk<T>(-9.75e3, 4.5e3));
    p.tau = UnifiedVector<T>(std::size_t(std::max(1, p.k())) * s.batch, mk<T>(-12345.0, -12345.0));
    p.ptrs = UnifiedVector<T*>(s.batch, nullptr);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i)
                p.mem[p.at(it, i, j)] = it < reps ? mk<T>(u(gen), u(gen)) : p.mem[p.at(it % reps, i, j)];
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

// ||Q R - A|| / ||A|| for one item, Q = H_0 ... H_{k-1} from the packed reflectors, in double.
template <typename T>
double qr_residual(const Qr<T>& p, int it) {
    const int m = p.s.m, n = p.s.n, k = p.k();
    std::vector<D> Q(std::size_t(m) * k, 0.0);
    for (int j = 0; j < k; ++j) Q[std::size_t(j) * m + j] = 1.0;
    for (int i = k - 1; i >= 0; --i) {
        const D t = up(p.tau[std::size_t(it) * k + i]);
        for (int c = 0; c < k; ++c) {
            D w = Q[std::size_t(c) * m + i];
            for (int r = i + 1; r < m; ++r) w += std::conj(up(p.mem[p.at(it, r, i)])) * Q[std::size_t(c) * m + r];
            const D f = t * w;
            Q[std::size_t(c) * m + i] -= f;
            for (int r = i + 1; r < m; ++r) Q[std::size_t(c) * m + r] -= f * up(p.mem[p.at(it, r, i)]);
        }
    }
    double num = 0, den = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            D acc = 0.0;
            for (int q = 0; q <= std::min(k - 1, j); ++q) acc += Q[std::size_t(q) * m + i] * up(p.mem[p.at(it, q, j)]);
            const D a = up(p.mem0[p.at(it, i, j)]);
            num += std::norm(acc - a);
            den += std::norm(a);
        }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

template <typename T>
double tol(int m, int n) {
    return std::max(0.5 * (m + n), 8.0) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

// The checked items' residuals (all of a small batch, the representatives of a repeating one),
// every element outside the m x n windows bit for bit, every tau finite, and for a repeating
// batch every item's factor and tau bit-identical to its representative's.
template <typename T>
void expect_factored(const Qr<T>& p, const std::string& what) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0) for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    else if (s.batch <= 8) for (int it = 0; it < s.batch; ++it) items.push_back(it);
    else items = {0, 1, s.batch / 2, s.batch - 1};
    for (int it : items) {
        const double r = qr_residual(p, it);
        ASSERT_TRUE(std::isfinite(r) && r <= tol<T>(s.m, s.n)) << what << " item " << it << " residual " << r;
    }
    for (std::size_t e = 0; e < p.mem.size(); ++e) {
        const int it = int(e / p.stride), off = int(e % p.stride), j = off / p.ld, i = off % p.ld;
        if (it < s.batch && j < s.n && i < s.m) continue;
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": wrote outside A's window at element " << e;
    }
    for (std::size_t e = 0; e < std::size_t(p.k()) * s.batch; ++e)
        ASSERT_TRUE(std::isfinite(std::abs(up(p.tau[e])))) << what << ": tau[" << e << "] not written";
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it) {
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i)
                    ASSERT_TRUE(same_bits(p.mem[p.at(it, i, j)], p.mem[p.at(it % s.period, i, j)]))
                        << what << ": item " << it << " differs from its representative at (" << i << "," << j << ")";
            for (int i = 0; i < p.k(); ++i)
                ASSERT_TRUE(same_bits(p.tau[std::size_t(it) * p.k() + i], p.tau[std::size_t(it % s.period) * p.k() + i]))
                    << what << ": tau of item " << it << " differs from its representative";
        }
}

std::string label(const Spec& s) {
    return std::to_string(s.m) + "x" + std::to_string(s.n) + " batch=" + std::to_string(s.batch);
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// The outermost geqrf trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("geqrf ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no geqrf trace line in: " + err + ">";
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
class GeqrfCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MVof<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = dispatch::factorization_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native tiers are GPU kernels";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    std::size_t budget() const {
        const auto lm = std::size_t(this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
        return lm > 4096 ? lm - 4096 : 0;
    }
    int tiny_max() const { return sycl_geqrf::geqrf_tiny_max_n_for_slm<T>(budget()); }
    bool cta_fits(int m, int n) const { return sycl_geqrf::geqrf_cta_fits<T>(m, n, budget()); }
    int cta_max_square() const {
        int best = 0;
        for (int n = 1; n <= 4096 && cta_fits(n, n); ++n) best = n;
        return best;
    }

    // ---- the limit oracle: the drivers' own capacity queries, not geqrf.cc's can_run ----
    bool expect_runs(const C& c, int m, int n) const {
        if (std::holds_alternative<gq::Vendor>(c)) return kVendor;
        if (m < n || n < 1) return false;
        if (std::holds_alternative<gq::Tiny>(c)) return m == n && n <= tiny_max();
        if (std::holds_alternative<gq::Cta>(c)) return cta_fits(m, n);
        return sycl_geqrf::geqrf_blocked_available<T>();
    }
    // The Vendor choice spells the class word `vendor`, which falls back to Auto where no vendor
    // can run, instead of throwing (§5.3).
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<gq::Vendor>(c) && !kVendor; }

    sycl_geqrf::GeqrfTrailingGemm<T> seam() const {
        return [](Queue& c, const MV& ga, const MV& gb, const MV& gc, T a, T b, Transpose ta, Transpose tb,
                  ComputePrecision pr) { return gemm<B, T>(c, ga, gb, gc, a, b, ta, tb, pr); };
    }

    void run(Qr<T>& p) {
        const std::size_t ws = geqrf_buffer_size<B, T>(*this->ctx, p.A(), p.tau.to_span());
        UnifiedVector<std::byte> w(std::max<std::size_t>(1, ws));
        (void)geqrf<B, T>(*this->ctx, p.A(), p.tau.to_span(), w.to_span());
        this->ctx->wait();
    }
    void run_pinned(const C& c, Qr<T>& p) {
        const Pin pin("geqrf", c);
        run(p);
    }
    // Acceptance is asked of the sizing call, which runs the same choose() (R5).
    bool pin_accepted(const C& c, Qr<T>& p) {
        const Pin pin("geqrf", c);
        try {
            (void)geqrf_buffer_size<B, T>(*this->ctx, p.A(), p.tau.to_span());
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }
    std::string auto_choice(const Spec& s) {
        const ScopedEnvVar clear("BATCHLAS_GEQRF_ROUTE", nullptr);
        auto p = make_qr<T>(s);
        return traced_choice([&] { run(p); });
    }

    // The family's own driver: Tiny, Cta, Blocked with the public gemm, or the vendor.
    bool direct(const C& c, Qr<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const MV A = p.A();
        try {
            std::size_t ws = 0;
            if (std::holds_alternative<gq::Tiny>(c)) ws = sycl_geqrf::geqrf_tiny_buffer_size<T>(q, A);
            else if (std::holds_alternative<gq::Cta>(c)) ws = sycl_geqrf::geqrf_cta_buffer_size<T>(q, A);
            else if (std::holds_alternative<gq::Blocked>(c)) ws = sycl_geqrf::geqrf_blocked_buffer_size<T>(q, A);
            UnifiedVector<std::byte> w(std::max<std::size_t>(1, ws));
            if (std::holds_alternative<gq::Tiny>(c)) {
                (void)sycl_geqrf::geqrf_tiny_dispatch<T>(q, A, p.tau.to_span(), w.to_span());
            } else if (std::holds_alternative<gq::Cta>(c)) {
                (void)sycl_geqrf::geqrf_cta_dispatch<T>(q, A, p.tau.to_span(), w.to_span());
            } else if (std::holds_alternative<gq::Blocked>(c)) {
                (void)sycl_geqrf::geqrf_blocked_dispatch<T>(q, A, p.tau.to_span(), w.to_span(), seam());
            } else {
                if constexpr (kVendor) {
                    UnifiedVector<std::byte> vw(std::max<std::size_t>(
                        1, backend::geqrf_vendor_buffer_size<B, T>(q, A, p.tau.to_span())));
                    (void)backend::geqrf_vendor<B, T>(q, A, p.tau.to_span(), vw.to_span());
                } else {
                    if (why) *why = "no vendor in this build";
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

    // Square orders straddling tiny's buckets and ceiling and Cta's square fit, tall shapes
    // straddling Cta's area, and a wide one.
    std::vector<Spec> straddles() const {
        const int t = tiny_max(), c = cta_max_square();
        std::vector<Spec> v;
        for (int n : {1, 2, 8, 9, 16, 17, t, t + 1, c, c + 1}) v.push_back({n, n, 3});
        int tall_m = 1;
        while (cta_fits(tall_m + 1, 24)) ++tall_m;
        for (auto [m, n] : {std::pair{tall_m, 24}, std::pair{tall_m + 1, 24}, std::pair{40, 7}, std::pair{300, 70},
                            std::pair{24, 40}})
            v.push_back({m, n, 3});
        for (std::size_t i = 0; i < v.size(); ++i) v[i].seed = 900u + unsigned(i);
        return v;
    }
};

TYPED_TEST_SUITE(GeqrfCandidates, Types);

// §8.1: each candidate on shapes straddling its limits. Accepted shapes factor correctly;
// refused ones throw from the pin.
TYPED_TEST(GeqrfCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    ASSERT_GE(this->tiny_max(), 8);
    ASSERT_GT(this->cta_max_square(), this->tiny_max());
    int ran = 0, refused = 0;
    for (const C& c : gq::candidates<T>())
        for (const Spec& s : this->straddles()) {
            auto p = make_qr<T>(s);
            const std::string what = name(c, s);
            if (TestFixture::vendor_word_falls_back(c)) {
                if (s.m < s.n) continue;  // vendor-free wide: no route at all
                EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), this->auto_choice(s)) << what;
                expect_factored(p, what + " (vendor-free: Auto)");
                continue;
            }
            if (!this->expect_runs(c, s.m, s.n)) {
                EXPECT_FALSE(this->pin_accepted(c, p)) << what << " was accepted";
                ++refused;
                continue;
            }
            ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
            this->run_pinned(c, p);
            if (s.m >= s.n) expect_factored(p, what);
            ++ran;
        }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// The pinned facade runs exactly that family's driver: bit-identical to the direct call.
TYPED_TEST(GeqrfCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : gq::candidates<T>())
        for (auto [m, n] : {std::pair{13, 13}, std::pair{29, 29}, std::pair{64, 24}, std::pair{100, 70}}) {
            if (!this->expect_runs(c, m, n) || TestFixture::vendor_word_falls_back(c)) continue;
            const Spec s{m, n, 3, 0, 4242u + unsigned(m)};
            auto pinned = make_qr<T>(s);
            auto direct = make_qr<T>(s);
            const std::string what = name(c, s);
            this->run_pinned(c, pinned);
            std::string why;
            ASSERT_TRUE(this->direct(c, direct, &why)) << what << ": " << why;
            expect_factored(pinned, what);
            for (std::size_t e = 0; e < pinned.mem.size(); ++e)
                ASSERT_TRUE(same_bits(pinned.mem[e], direct.mem[e])) << what << ": not this family's driver; element " << e;
            for (std::size_t e = 0; e < pinned.tau.size(); ++e)
                ASSERT_TRUE(same_bits(pinned.tau[e], direct.tau[e])) << what << ": tau differs at " << e;
            ++compared;
        }
    EXPECT_GT(compared, 0);
}

// The SLM tiers at a saturating batch: 1024 items repeating 7 distinct matrices; each
// representative is correct and every other item is bit-identical to its representative.
TYPED_TEST(GeqrfCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    struct Case { C c; int m, n; };
    const int t = this->tiny_max();
    const Case cases[] = {{gq::Tiny{}, t, t},   {gq::Tiny{}, 9, 9},   {gq::Cta{}, 40, 24},
                          {gq::Cta{}, 17, 17},  {gq::Blocked{}, 100, 70}, {gq::Blocked{}, 64, 64}};
    for (const auto& k : cases) {
        if (!this->expect_runs(k.c, k.m, k.n)) continue;
        Spec s{k.m, k.n, 1024, 7, 777u + unsigned(k.m)};
        auto p = make_qr<T>(s);
        this->run_pinned(k.c, p);
        expect_factored(p, name(k.c, s));
        if (::testing::Test::HasFatalFailure()) return;
    }
}

// §8.2 (R3): on every straddling shape a pin is accepted exactly when the family's driver launches.
TYPED_TEST(GeqrfCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    int disagreements = 0;
    for (const C& c : gq::candidates<T>())
        for (const Spec& s : this->straddles()) {
            if (TestFixture::vendor_word_falls_back(c)) continue;  // Auto, not the vendor
            auto a = make_qr<T>(s);
            auto b = make_qr<T>(s);
            const bool pin = this->pin_accepted(c, a);
            std::string why;
            // A wide view would corrupt the vendor's own input contract nowhere; it is a legal call.
            const bool run = this->direct(c, b, &why);
            EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the driver "
                                << (run ? "launches" : "refuses: " + why);
            EXPECT_EQ(pin, this->expect_runs(c, s.m, s.n)) << name(c, s) << ": can_run disagrees with the oracle";
            disagreements += pin != run;
        }
    EXPECT_EQ(disagreements, 0);
}

// §8.3 / R5: under each pin, a workspace of exactly geqrf_buffer_size bytes, inside a poisoned
// arena, is enough: nothing past its end is written.
TYPED_TEST(GeqrfCandidates, ExactWorkspaceInAPoisonedArena) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    constexpr std::size_t kGuard = 4096;
    for (const C& c : gq::candidates<T>())
        for (auto [m, n, batch] : {std::tuple{16, 16, 5}, std::tuple{40, 24, 3}, std::tuple{100, 70, 3},
                                   std::tuple{1, 1, 2}, std::tuple{300, 33, 2}}) {
            if (!this->expect_runs(c, m, n) || TestFixture::vendor_word_falls_back(c)) continue;
            const Spec s{m, n, batch, 0, 31u + unsigned(m)};
            auto p = make_qr<T>(s);
            const Pin pin("geqrf", c);
            const std::size_t need = geqrf_buffer_size<B, T>(*this->ctx, p.A(), p.tau.to_span());
            UnifiedVector<std::byte> arena(need + kGuard, std::byte{0xA5});
            (void)geqrf<B, T>(*this->ctx, p.A(), p.tau.to_span(), Span<std::byte>(arena.data(), need));
            this->ctx->wait();
            expect_factored(p, name(c, s) + " in an exact workspace");
            for (std::size_t e = need; e < arena.size(); ++e)
                ASSERT_EQ(arena[e], std::byte{0xA5}) << name(c, s) << ": wrote " << (e - need)
                                                     << " bytes past a workspace of " << need;
        }
}

// geqrf_buffer_size_bound, the sizing of callers that size once and factor sub-views
// (band_reduction, sytrd_sy2sb): at least every sub-view's own geqrf_buffer_size, under Auto
// and under every pin that can run the sub-view, and it never dereferences A or tau.
TYPED_TEST(GeqrfCandidates, BoundCoversEverySubViewChoice) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    for (auto [bm, bn, batch] : {std::tuple{64, 31, 4}, std::tuple{257, 96, 3}, std::tuple{48, 48, 1}}) {
        const MVof<T> bound_view(nullptr, bm, bn, bm, bm * bn, batch);
        const std::size_t bound =
            geqrf_buffer_size_bound<B, T>(*this->ctx, bound_view, Span<T>(nullptr, std::size_t(bn) * batch));
        for (int m = 1; m <= bm; m += std::max(1, bm / 9))
            for (int n = 1; n <= bn; n += std::max(1, bn / 7)) {
                const MVof<T> sub(nullptr, m, n, m, m * n, batch);
                const Span<T> tau(nullptr, std::size_t(std::min(m, n)) * batch);
                EXPECT_LE((geqrf_buffer_size<B, T>(*this->ctx, sub, tau)), bound) << m << "x" << n << " under Auto";
                for (const C& c : gq::candidates<T>()) {
                    if (!this->expect_runs(c, m, n) || TestFixture::vendor_word_falls_back(c)) continue;
                    const Pin pin("geqrf", c);
                    EXPECT_LE((geqrf_buffer_size<B, T>(*this->ctx, sub, tau)), bound)
                        << m << "x" << n << " pinned " << select::to_string(c) << " in a " << bm << "x" << bn << " bound";
                }
            }
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(GeqrfCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "cta:1", "blocked:nb=32", "native:vendor", "native:lpanel", "lpanel",
                             "fused", "native:cta:8"}) {
        auto p = make_qr<T>(Spec{});
        const Pin pin("geqrf", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
}

// The named can_run-false cases, each with its message (absorbs route_vocabulary_tests'
// RouteGeqrf.WideIsUnsupportedByEveryNativeArm, CorrectnessGatesAreNotSpeedGates and
// CtaCapacityIsAnAreaAndAHeightNotTwoExtentBounds): wide, empty, tiny on a tall panel and one
// order past its ceiling, cta one row past its area.
TYPED_TEST(GeqrfCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    const int t = this->tiny_max();
    int tall_m = 1;
    while (this->cta_fits(tall_m + 1, 24)) ++tall_m;
    struct Case { C c; int m, n; };
    const Case cases[] = {{gq::Tiny{}, 24, 40},       {gq::Cta{}, 24, 40},        {gq::Blocked{}, 24, 40},
                          {gq::Tiny{}, 2 * t, t},     {gq::Tiny{}, t + 1, t + 1}, {gq::Cta{}, tall_m + 1, 24},
                          {gq::Blocked{}, 8, 0},      {gq::Cta{}, 0, 0},          {gq::Tiny{}, 0, 0}};
    for (const auto& k : cases) {
        const Spec s{k.m, k.n, 2};
        auto p = make_qr<T>(s);
        EXPECT_FALSE(this->pin_accepted(k.c, p)) << name(k.c, s) << " was accepted";
    }
}

// A heterogeneous batch: one launch has one (m, n), so every native pin throws. Auto takes the
// vendor; vendor-free there is no route.
TYPED_TEST(GeqrfCandidates, HeterogeneousBatchHasNoNativeRoute) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    auto p = make_qr<T>(Spec{32, 16, 4});
    UnifiedVector<int> ar(4), ac(4);
    for (int b = 0; b < 4; ++b) ar[b] = 32 - b, ac[b] = 16 - b;
    const MVof<T> H = p.A().with_active_dims(ar.to_span(), ac.to_span());
    ASSERT_TRUE(H.is_heterogeneous());
    for (const C& c : gq::candidates<T>()) {
        if (std::holds_alternative<gq::Vendor>(c)) continue;
        const Pin pin("geqrf", c);
        try {
            (void)geqrf_buffer_size<B, T>(*this->ctx, H, p.tau.to_span());
            ADD_FAILURE() << select::to_string(c) << " accepted a heterogeneous batch";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
        }
    }
    const ScopedEnvVar clear("BATCHLAS_GEQRF_ROUTE", nullptr);
    auto size = [&] { (void)geqrf_buffer_size<B, T>(*this->ctx, H, p.tau.to_span()); };
    if constexpr (TestFixture::kVendor) EXPECT_NO_THROW(size());
    else EXPECT_THROW(size(), dispatch::NoRouteError);
}

// §5.3, and RouteGeqrf.BatchlasGeqrfRouteIsActuallyRead ported: the legacy spellings and the
// class words, via ScopedPin and via BATCHLAS_GEQRF_ROUTE. Bare `native` is the row's best
// runnable non-vendor; bare `vendor` falls back to Auto where there is no vendor.
TYPED_TEST(GeqrfCandidates, LegacyAliasesAndClassWords) {
    using T = typename TestFixture::T;
    const Spec s{40, 24, 3};
    const std::string auto_pick = this->auto_choice(s);
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    const std::pair<const char*, std::string> expect[] = {
        {"native:cta", "cta"}, {"native:blocked", "blocked"}, {"NATIVE:CTA", "cta"}, {"cta", "cta"},
        {"Blocked", "blocked"}, {"vendor", vendor_pick},       {"native", "cta"},    {"auto", auto_pick}};
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = make_qr<T>(s);
            select::testing::reset_warnings();
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_GEQRF_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("geqrf", std::string_view(word));
                this->run(p);
            });
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_GEQRF_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            expect_factored(p, what);
        }
    {   // native:tiny on a square order inside the ceiling.
        auto p = make_qr<T>(Spec{9, 9, 2});
        EXPECT_EQ(traced_choice([&] {
                      const Pin pin("geqrf", std::string_view("native:tiny"));
                      this->run(p);
                  }),
                  "tiny");
        expect_factored(p, "native:tiny");
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_GEQRF_ROUTE, and nested pins restore the outer one.
TYPED_TEST(GeqrfCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const ScopedEnvVar env("BATCHLAS_GEQRF_ROUTE", "tiny");
    {
        auto p = make_qr<T>(Spec{40, 24, 2});
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_qr<T>(Spec{40, 24, 2});
    EXPECT_EQ(traced_choice([&] { this->run_pinned(C{gq::Blocked{}}, p); }), "blocked");
    expect_factored(p, "blocked over env tiny");
    const Pin outer("geqrf", C{gq::Cta{}});
    {
        const Pin inner("geqrf", C{gq::Blocked{}});
        auto r = make_qr<T>(Spec{16, 16, 2});
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "blocked");
    }
    auto r = make_qr<T>(Spec{16, 16, 2});
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "cta") << "the inner pin did not restore the outer";
    expect_factored(r, "outer cta");
}

// The shipped (transcribed) table on this device reproduces the old router at the window
// edges the deleted RouteGeqrf tests pinned: per-type order floors, the tall-panel clause, the
// tiny window and its gaps, the CTA/Blocked crossover (vendor-present = Auto; vendor-free = the
// class word `native`). Hand-read from the deleted predicates at the measured 97,280 B budget.
TYPED_TEST(GeqrfCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEQRF_ROUTE", nullptr);
    if (this->budget() != 97280) GTEST_SKIP() << "the expectations are read at a 97,280 B budget";
    const auto tables = select::tables_in_borrow_order("geqrf", select::dtype_name<T>(),
                                                       select::device_of<TestFixture::B>(*this->ctx));
    if (tables.empty() || tables.front()->source.rfind("transcribed:", 0) != 0)
        GTEST_SKIP() << "this device's first geqrf table is not the transcription";
    struct Row { int m, n; const char* vendor; const char* native; };
    std::vector<Row> rows;
    if constexpr (std::is_same_v<T, float>)
        rows = {{3, 3, "vendor", "tiny"},     {4, 4, "tiny", "tiny"},     {17, 17, "vendor", "tiny"},
                {21, 21, "tiny", "tiny"},     {33, 33, "vendor", "cta"},  {63, 63, "vendor", "cta"},
                {64, 64, "cta", "cta"},       {96, 96, "cta", "cta"},     {97, 97, "blocked", "blocked"},
                {160, 40, "cta", "cta"},      {159, 40, "vendor", "cta"}, {124, 31, "vendor", "cta"},
                {400, 100, "blocked", "blocked"}, {130, 70, "cta", "cta"},
                {400, 90, "blocked", "blocked"}};  // 400x90 is past Cta's area: the fit, not the crossover
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        rows = {{4, 4, "vendor", "tiny"},  {5, 5, "tiny", "tiny"},    {9, 9, "vendor", "tiny"},
                {17, 17, "vendor", "cta"}, {20, 20, "vendor", "cta"}, {22, 22, "vendor", "tiny"},
                {24, 24, "tiny", "tiny"},  {47, 47, "vendor", "cta"}, {48, 48, "cta", "cta"},
                {77, 77, "blocked", "blocked"}, {128, 32, "cta", "cta"}, {127, 32, "vendor", "cta"}};
    else if constexpr (std::is_same_v<T, double>)
        rows = {{8, 8, "vendor", "cta"},   {48, 48, "vendor", "cta"},     {49, 49, "vendor", "blocked"},
                {75, 75, "vendor", "blocked"}, {76, 76, "blocked", "blocked"}, {320, 40, "blocked", "blocked"},
                {319, 40, "vendor", "blocked"}};
    else
        rows = {{8, 8, "vendor", "cta"},       {54, 54, "vendor", "cta"},       {96, 96, "vendor", "blocked"},
                {255, 255, "vendor", "blocked"}, {256, 256, "blocked", "blocked"}, {256, 32, "blocked", "blocked"},
                {255, 32, "vendor", "blocked"}};
    for (const Row& r : rows) {
        const Spec s{r.m, r.n, 2, 0, 5u + unsigned(r.n)};
        auto p = make_qr<T>(s);
        const std::string want = TestFixture::kVendor ? r.vendor : r.native;
        EXPECT_EQ(traced_choice([&] { this->run(p); }), want) << "Auto " << label(s);
        expect_factored(p, "auto " + label(s));
        auto q = make_qr<T>(s);
        EXPECT_EQ(traced_choice([&] {
                      const Pin pin("geqrf", std::string_view("native"));
                      this->run(q);
                  }),
                  r.native)
            << "native (the vendor-free walk) " << label(s);
    }
}

// key_of's every field reaches choose(): a synthetic table for this device whose winner changes
// with form, n and aspect alone. Fixing a field in key_of turns exactly its row red.
TYPED_TEST(GeqrfCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEQRF_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("geqrf.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("geqrf." + dtype + "." + dev + ".txt",
                       "# op=geqrf dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: form:exact n:log:3 aspect:log\n"
                       "form=sq n=8 aspect=1 | tiny 1 | cta 2 | blocked 3\n"
                       "form=sq n=16 aspect=1 | blocked 1 | cta 2\n"
                       "form=tall n=8 aspect=2 | cta 1 | blocked 2\n"
                       "form=tall n=8 aspect=8 | blocked 1 | cta 2\n"
                       "form=tall n=16 aspect=2 | blocked 1 | cta 2\n"
                       "form=wide n=8 aspect=2 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { int m, n; const char* expect; const char* field; };
    const Probe probes[] = {{8, 8, "tiny", "base"},           {16, 16, "blocked", "n (square)"},
                            {17, 8, "cta", "form"},           {64, 8, "blocked", "aspect"},
                            {33, 16, "blocked", "n (tall)"},  {23, 8, "cta", "aspect is floor(m/n)"}};
    for (const auto& k : probes) {
        auto p = make_qr<T>(Spec{k.m, k.n, 2, 0, 43u});
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row";
        expect_factored(p, std::string("the ") + k.field + " row");
    }
    if constexpr (TestFixture::kVendor) {
        auto p = make_qr<T>(Spec{8, 17, 2, 0, 44u});
        EXPECT_EQ(traced_choice([&] { this->run(p); }), "vendor") << "the wide row";
    }
}

// The trace prints the key the lookup used: form, n = cols, aspect = max/min.
TYPED_TEST(GeqrfCandidates, TraceShowsTheLookupKey) {
    using T = typename TestFixture::T;
    for (auto [m, n, want] : {std::tuple{70, 20, "form=tall n=20 aspect=3 ->"}, std::tuple{12, 12, "form=sq n=12 aspect=1 ->"}}) {
        auto p = make_qr<T>(Spec{m, n, 5});
        const std::string line = traced_line([&] { this->run_pinned(C{gq::Blocked{}}, p); });
        EXPECT_NE(line.find(want), std::string::npos) << line;
        expect_factored(p, line);
    }
}

// RouteGeqrf.VendorFreeFallbackHandsOverTheNativeRoute, ported: with a table naming only the
// vendor, a vendor-free build takes the last resort, blocked; a vendor build takes the vendor.
TYPED_TEST(GeqrfCandidates, VendorFreeWalkIsTheOldTieBreak) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEQRF_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("geqrf.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("geqrf." + dtype + "." + dev + ".txt",
                       "# op=geqrf dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: form:exact n:log:3 aspect:log\n"
                       "form=tall n=16 aspect=4 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    for (auto [m, n] : {std::pair{64, 16}, std::pair{20, 20}}) {
        auto p = make_qr<T>(Spec{m, n, 2});
        std::string all;
        const std::string got = traced_choice([&] { this->run(p); }, &all);
        EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : "blocked") << m << "x" << n;
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_factored(p, "vendor-only table");
    }
}

// The coverage row (§5.6): the backend, m, n, k = min(m, n) and the native flags.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(GeqrfCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "geqrf_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_GEQRF_ROUTE", nullptr);
        dispatch::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("geqrf", C{gq::Cta{}});
            auto p = make_qr<T>(Spec{40, 24, 2});
            this->run(p);
        }
        if constexpr (TestFixture::kVendor) {
            auto p = make_qr<T>(Spec{8, 20, 2});  // wide: vendor, no native candidate runs
            this->run(p);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,geqrf,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 14u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    ASSERT_TRUE(rows.count("40 24 24")) << "no 40x24 row";
    const auto& f = rows["40 24 24"];
    EXPECT_EQ(f[3], backend);
    EXPECT_EQ(f[9], "native");
    EXPECT_EQ(f[10], "cta");
    EXPECT_EQ(f[12], "1");
    EXPECT_EQ(f[13], "1");
    if constexpr (TestFixture::kVendor) {
        ASSERT_TRUE(rows.count("8 20 8")) << "no wide row";
        const auto& w = rows["8 20 8"];
        EXPECT_EQ(w[9], "vendor");
        EXPECT_EQ(w[10], "vendor");
        EXPECT_EQ(w[12], "1");
        EXPECT_EQ(w[13], "0") << "no native candidate runs a wide view";
    }
}

// RouteGeqrf.CorrectnessGatesAreNotSpeedGates' CPU case, ported: on a CPU queue no native
// family can run, so native pins throw and Auto is the vendor (or no route without one).
template <typename Config>
class GeqrfCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(GeqrfCandidatesCpu, Types);

TYPED_TEST(GeqrfCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_GEQRF_ROUTE", nullptr);
    auto p = make_qr<T>(Spec{12, 8, 2});
    for (const C& c : gq::candidates<T>()) {
        if (std::holds_alternative<gq::Vendor>(c)) continue;
        const Pin pin("geqrf", c);
        EXPECT_THROW(((void)geqrf_buffer_size<B, T>(*this->ctx, p.A(), p.tau.to_span())), std::invalid_argument)
            << select::to_string(c);
    }
    auto size = [&] { (void)geqrf_buffer_size<B, T>(*this->ctx, p.A(), p.tau.to_span()); };
    if constexpr (dispatch::factorization_vendor_available<B>) {
        EXPECT_EQ(traced_choice([&] {
                      const std::size_t ws = geqrf_buffer_size<B, T>(*this->ctx, p.A(), p.tau.to_span());
                      UnifiedVector<std::byte> w(std::max<std::size_t>(1, ws));
                      (void)geqrf<B, T>(*this->ctx, p.A(), p.tau.to_span(), w.to_span());
                      this->ctx->wait();
                  }),
                  "vendor");
    } else {
        EXPECT_THROW(size(), dispatch::NoRouteError);
    }
}

// The transcribed rows, read with Table::nearest directly so every device checks them, on
// both devices: they are the same old preference (the old predicates read no architecture).
TEST(GeqrfTranscribedTable, RowsHoldTheOldPreference) {
    struct Row { const char* dtype; const char* form; int n, aspect; const char* ranked; };
    const Row rows[] = {
        {"float", "sq", 21, 1, "tiny|vendor|cta|blocked"},      {"float", "sq", 17, 1, "vendor|tiny|cta|blocked"},
        {"float", "sq", 128, 1, "tiny|blocked|cta|vendor"},     {"float", "tall", 40, 4, "cta|blocked|vendor"},
        {"float", "tall", 40, 3, "vendor|cta|blocked"},         {"float", "tall", 100, 2, "blocked|cta|vendor"},
        {"double", "sq", 76, 1, "blocked|tiny|cta|vendor"},     {"double", "tall", 40, 7, "vendor|cta|blocked"},
        {"double", "tall", 40, 8, "cta|blocked|vendor"},        {"double", "tall", 60, 2, "vendor|blocked|cta"},
        {"cfloat", "sq", 20, 1, "vendor|cta|tiny|blocked"},     {"cfloat", "sq", 22, 1, "vendor|tiny|cta|blocked"},
        {"cdouble", "sq", 255, 1, "vendor|cta|tiny|blocked"},   {"cdouble", "tall", 32, 8, "cta|blocked|vendor"},
        {"cdouble", "wide", 64, 4, "vendor"}};
    for (const char* device : {"sm_89", "sm_120"})
        for (const Row& r : rows) {
            const auto tables = select::tables_in_borrow_order("geqrf", r.dtype, select::device_from_key(device));
            ASSERT_FALSE(tables.empty()) << r.dtype;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, device) << r.dtype;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            const select::TableRow* row = t.nearest({{"form", r.form}, {"n", r.n}, {"aspect", r.aspect}});
            ASSERT_NE(row, nullptr) << t.file;
            std::string got;
            for (const auto& e : row->ranked) got += (got.empty() ? "" : "|") + e.spelling;
            EXPECT_EQ(got, r.ranked) << t.file << ":" << row->line;
            EXPECT_FALSE(row->timed) << t.file;
        }
}

}  // namespace
