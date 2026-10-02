// The rules-as-data engine against today's resolver, offline (no GPU): sm_120 facts, the
// committed RuleSets, and resolve_route over the shape the library would build.

#include "../src/routing/potrf_tiers.hh"
#include "../src/routing/getrs_tiers.hh"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace batchlas;
namespace rp = batchlas::routing::potrf;
using dispatch::Route;

namespace {

constexpr Backend kB = Backend::CUDA;

launch_plan::DeviceFacts sm120_facts() {
    launch_plan::DeviceFacts d;
    d.local_mem_bytes = 101376;
    d.max_wg_size = 1024;
    d.compute_units = 188;
    return d;
}

std::string name_of(Route r) {
    if (dispatch::is_vendor(r)) return "vendor";
    switch (r.algo) {
        case dispatch::Algorithm::Tiny: return "native:tiny";
        case dispatch::Algorithm::CTA: return "native:cta";
        case dispatch::Algorithm::LPanel: return "native:lpanel";
        case dispatch::Algorithm::Blocked: return "native:blocked";
        default: return "?";
    }
}

// Today's library decision for the same described device: the shape builder, the model
// pricing when the profile's gate passed, and the generic resolver.
template <class T>
std::string oracle(int64_t n, int64_t b, Uplo u, arch::RoutingProfile prof, bool vendor) {
    const auto d = sm120_facts();
    auto s = backend::potrf_op_shape_from_facts<kB, T>(d, n, n, b, u, true, true, false,
                                                       sycl_potrf::potrf_blocked_available<T>());
    s.profile = prof;
    backend::potrf_price_routes<T>(s, d, sycl_trsm::trsm_cta_max_n<T>(), 0, 0);
    return name_of(dispatch::resolve_route_uninstrumented<dispatch::Op::potrf, T>(Route{}, s,
                                                                                 vendor));
}

template <class T>
std::string engine(int64_t n, int64_t b, Uplo u, arch::RoutingProfile prof, bool vendor,
                   const routing::RuleSet* rs = nullptr) {
    const auto c = rp::ctx_from_facts<kB, T>(sm120_facts(), n, b, u, prof, vendor);
    try {
        const auto& set = rs ? *rs : routing::potrf_rules::rules_for_profile(prof);
        return std::string(routing::select<rp::Tiers<kB, T>>(set, rp::kBound<kB, T>, c,
                                                             routing::Pin{}).id());
    } catch (const routing::no_route_error&) {
        return "vendor";   // today's resolver returns Vendor when nothing native serves
    }
}

std::vector<int64_t> grid_batches() {
    std::vector<int64_t> v;
    for (int e = 0; e <= 18; ++e) v.push_back(int64_t(1) << e);
    return v;
}

template <class T>
void check_grid(const char* dt, arch::RoutingProfile prof, int nmax, int& cells, int& bad) {
    for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
        for (int n = 1; n <= nmax; ++n) {
            for (int64_t b : grid_batches()) {
                for (bool vendor : {true, false}) {
                    ++cells;
                    const auto want = oracle<T>(n, b, u, prof, vendor);
                    const auto got = engine<T>(n, b, u, prof, vendor);
                    if (want != got && ++bad <= 10) {
                        std::printf("MISMATCH %s %c n=%d b=%lld vendor=%d: today %s, rules %s\n", dt,
                                    u == Uplo::Upper ? 'U' : 'L', n, static_cast<long long>(b),
                                    vendor, want.c_str(), got.c_str());
                    }
                }
            }
        }
    }
}

int env_nmax() {
    const char* v = std::getenv("ROUTING_RULES_NMAX");
    return v ? std::atoi(v) : 2048;
}

}  // namespace

// (i) Every grid cell, both rule sources, with and without the vendor.
TEST(RoutingRules, GridMatchesResolver) {
    for (auto prof : {arch::RoutingProfile::sm_120, arch::RoutingProfile::sm_89}) {
        int cells = 0, bad = 0;
        const int nmax = env_nmax();
        check_grid<float>("float", prof, nmax, cells, bad);
        check_grid<double>("double", prof, nmax, cells, bad);
        check_grid<std::complex<float>>("cfloat", prof, nmax, cells, bad);
        check_grid<std::complex<double>>("cdouble", prof, nmax, cells, bad);
        std::printf("profile %s: %d cells, %d mismatches\n",
                    std::string(arch::to_string(prof)).c_str(), cells, bad);
        EXPECT_EQ(bad, 0);
    }
}

// (ii) Off-grid: log-uniform batch, uniform n. 1/octave bands are a known approximation;
// the measured disagreement is printed and bounded, not zero.
TEST(RoutingRules, OffGridFidelity) {
    std::mt19937_64 rng(11);
    std::uniform_int_distribution<int> nd(1, 2048), dt(0, 3), ud(0, 1);
    std::uniform_real_distribution<double> ld(0, 18);
    int bad = 0, bad_hand = 0;
    const int N = 10000;
    for (int i = 0; i < N; ++i) {
        const int n = nd(rng);
        const int64_t b = std::max<int64_t>(1, static_cast<int64_t>(std::pow(2.0, ld(rng))));
        const Uplo u = ud(rng) ? Uplo::Upper : Uplo::Lower;
        const int t = dt(rng);
        for (auto prof : {arch::RoutingProfile::sm_120, arch::RoutingProfile::sm_89}) {
            std::string w, g;
            switch (t) {
                case 0: w = oracle<float>(n, b, u, prof, true); g = engine<float>(n, b, u, prof, true); break;
                case 1: w = oracle<double>(n, b, u, prof, true); g = engine<double>(n, b, u, prof, true); break;
                case 2: w = oracle<std::complex<float>>(n, b, u, prof, true);
                        g = engine<std::complex<float>>(n, b, u, prof, true); break;
                default: w = oracle<std::complex<double>>(n, b, u, prof, true);
                         g = engine<std::complex<double>>(n, b, u, prof, true); break;
            }
            if (w != g) (prof == arch::RoutingProfile::sm_120 ? bad : bad_hand)++;
        }
    }
    std::printf("off-grid disagreement: model %d/%d (%.3f%%), hand %d/%d\n", bad, N,
                100.0 * bad / N, bad_hand, N);
    EXPECT_EQ(bad_hand, 0);
    EXPECT_LE(bad, N * 15 / 1000) << "design target is <= 0.1% after batch bisection";
}

// (iv) Flip ONE rule's first two candidates: exactly that box's grid cells turn red.
TEST(RoutingRules, GuardFlipOneRule) {
    using T = float;
    const auto& base = routing::potrf_rules::k_sm_120;
    std::vector<routing::Rule> copy(base.begin, base.end);
    const auto c0 = rp::ctx_from_facts<kB, T>(sm120_facts(), 128, 8192, Uplo::Lower,
                                              arch::RoutingProfile::sm_120, true);
    const routing::Rule* hit = base.match(c0.key(), c0.features());
    ASSERT_NE(hit, nullptr);
    const std::size_t idx = static_cast<std::size_t>(hit - base.begin);
    ASSERT_GE(copy[idx].nrank, 2);
    std::swap(copy[idx].rank[0], copy[idx].rank[1]);
    routing::RuleSet mutated = base;
    mutated.begin = copy.data();
    mutated.end = copy.data() + copy.size();
    std::printf("flipped R%04u: n [%lld,%lld] batch [%lld,%lld]\n", hit->id,
                static_cast<long long>(hit->lo[0]), static_cast<long long>(hit->hi[0]),
                static_cast<long long>(hit->lo[1]), static_cast<long long>(hit->hi[1]));
    int red = 0, red_outside = 0, inside = 0;
    for (int n = 1; n <= 1024; ++n) {
        for (int64_t b : grid_batches()) {
            const bool in = n >= hit->lo[0] && n <= hit->hi[0] && b >= hit->lo[1] && b <= hit->hi[1];
            inside += in;
            const auto want = oracle<T>(n, b, Uplo::Lower, arch::RoutingProfile::sm_120, true);
            const auto got = engine<T>(n, b, Uplo::Lower, arch::RoutingProfile::sm_120, true, &mutated);
            if (want != got) {
                ++red;
                if (!in) ++red_outside;
            }
        }
    }
    std::printf("red %d of %d cells inside the box, %d outside\n", red, inside, red_outside);
    EXPECT_EQ(red, inside);
    EXPECT_EQ(red_outside, 0);
}

// Every registered tier is named by some RuleSet, or it is a finished tier nothing can select.
TEST(RoutingRules, NoUnrankedTier) {
    using Ts = rp::Tiers<kB, float>;
    std::set<std::string> named;
    for (const auto* rs : {&routing::potrf_rules::k_sm_120, &routing::potrf_rules::k_hand}) {
        for (const auto* r = rs->begin; r != rs->end; ++r) {
            for (int i = 0; i < r->nrank; ++i) {
                named.insert(std::string(routing::potrf_rules::kNames[r->rank[i].name]));
            }
        }
    }
    for (auto id : routing::potrf_rules::kPinnableOnly) named.insert(std::string(id));
    for (auto id : Ts::ids) EXPECT_TRUE(named.count(std::string(id))) << id << " is never ranked";
}

// ---- getrs: the second op, (n, nrhs, batch) x (dtype, trans), hand rules from route_oracle --

namespace {

template <class T>
dispatch::GetrsShape getrs_shape(int64_t n, int64_t nrhs, int64_t batch, Transpose tr) {
    dispatch::GetrsShape s;
    s.op = dispatch::Op::getrs;
    s.scalar = dispatch::scalar_kind_of<T>;
    s.backend = kB;
    s.m = n;
    s.n = nrhs;
    s.k = n;
    s.batch = batch;
    s.transA = tr;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.blocked_available = sycl_getrs::getrs_blocked_available<T>();
    if (sycl_getrs::getrs_fused_available<T>()) {
        s.fused_max_elems =
            static_cast<int64_t>(sycl_getrs::getrs_fused_max_rhs_elems<T>(101376 - 4096));
        s.fused_max_nrhs = sycl_getrs::kGetrsFusedMaxRhs;
    }
    return s;
}

std::string getrs_name(Route r) {
    if (dispatch::is_vendor(r)) return "vendor";
    return r.algo == dispatch::Algorithm::CTA ? "native:fused" : "native:blocked";
}

template <class T>
int getrs_compare(int64_t n, int64_t r, int64_t b, Transpose tr, bool verbose) {
    namespace rg = batchlas::routing::getrs;
    int bad = 0;
    for (bool vendor : {true, false}) {
        rg::Ctx<T> c;
        c.s = getrs_shape<T>(n, r, b, tr);
        c.vendor_legal = vendor;
        const auto want = getrs_name(
            dispatch::resolve_route_uninstrumented<dispatch::Op::getrs, T>(Route{}, c.s, vendor));
        std::string got;
        try {
            got = std::string(rg::select_ctx<kB, T>(c).id());
        } catch (const routing::no_route_error&) {
            got = "vendor";
        }
        if (want != got) {
            ++bad;
            if (verbose) {
                std::printf("GETRS MISMATCH n=%lld nrhs=%lld b=%lld vendor=%d: today %s rules %s\n",
                            static_cast<long long>(n), static_cast<long long>(r),
                            static_cast<long long>(b), vendor, want.c_str(), got.c_str());
            }
        }
    }
    return bad;
}

}  // namespace

TEST(RoutingRulesGetrs, MatchesResolver) {
    const int64_t ns[] = {1, 2, 31, 32, 33, 64, 100, 127, 128, 129, 512, 1000, 2048, 3000, 4096, 9000};
    const int64_t rs[] = {1, 2, 3, 4, 5, 8, 9, 16, 63, 64, 65, 127, 128, 129, 1000};
    const int64_t bs[] = {1, 64, 127, 128, 129, 1000, 65536, 2000000};
    int cells = 0, bad = 0;
    for (Transpose tr : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        for (int64_t n : ns) {
            for (int64_t r : rs) {
                for (int64_t b : bs) {
                    cells += 8;
                    bad += getrs_compare<float>(n, r, b, tr, bad < 10);
                    bad += getrs_compare<double>(n, r, b, tr, bad < 10);
                    bad += getrs_compare<std::complex<float>>(n, r, b, tr, bad < 10);
                    bad += getrs_compare<std::complex<double>>(n, r, b, tr, bad < 10);
                }
            }
        }
    }
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> lg(0, 13), lb(0, 21);
    for (int i = 0; i < 20000; ++i) {
        const int64_t n = std::max<int64_t>(1, static_cast<int64_t>(std::pow(2.0, lg(rng))));
        const int64_t r = std::max<int64_t>(1, static_cast<int64_t>(std::pow(2.0, lg(rng))));
        const int64_t b = std::max<int64_t>(1, static_cast<int64_t>(std::pow(2.0, lb(rng))));
        const Transpose tr = (i % 3 == 0) ? Transpose::NoTrans : (i % 3 == 1 ? Transpose::Trans
                                                                            : Transpose::ConjTrans);
        cells += 2;
        switch (i % 4) {
            case 0: bad += getrs_compare<float>(n, r, b, tr, bad < 10); break;
            case 1: bad += getrs_compare<double>(n, r, b, tr, bad < 10); break;
            case 2: bad += getrs_compare<std::complex<float>>(n, r, b, tr, bad < 10); break;
            default: bad += getrs_compare<std::complex<double>>(n, r, b, tr, bad < 10); break;
        }
    }
    std::printf("getrs: %d decisions, %d mismatches; %zu rules\n", cells, bad,
                static_cast<std::size_t>(routing::getrs_rules::k_hand.end -
                                         routing::getrs_rules::k_hand.begin));
    EXPECT_EQ(bad, 0);
}

// Pins: an illegal pin is refused loudly (strict throws) and the rules decide; a legal pin wins.
TEST(RoutingRules, PinsAreHonouredOrRefused) {
    using T = float;
    using Ts = rp::Tiers<kB, T>;
    const auto up = rp::ctx_from_facts<kB, T>(sm120_facts(), 128, 8192, Uplo::Upper,
                                              arch::RoutingProfile::sm_120, true);
    const auto& rs = routing::potrf_rules::k_sm_120;
    auto lp = routing::parse_pin<Ts>("native:lpanel", false);
    const auto s1 = routing::select<Ts>(rs, rp::kBound<kB, T>, up, lp);
    EXPECT_EQ(s1.reason, routing::Reason::PinRefused);
    EXPECT_EQ(s1.id(), "vendor");
    EXPECT_EQ(s1.forced, "native:lpanel");
    auto strict = routing::parse_pin<Ts>("native:lpanel", true);
    EXPECT_THROW((void)routing::select<Ts>(rs, rp::kBound<kB, T>, up, strict), std::invalid_argument);
    const auto lo = rp::ctx_from_facts<kB, T>(sm120_facts(), 128, 8192, Uplo::Lower,
                                              arch::RoutingProfile::sm_120, true);
    auto cta = routing::parse_pin<Ts>("native:blocked", true);
    const auto s2 = routing::select<Ts>(rs, rp::kBound<kB, T>, lo, cta);
    EXPECT_EQ(s2.reason, routing::Reason::Pinned);
    EXPECT_EQ(s2.id(), "native:blocked");
    // A bare origin restricts the ranking: the rule's vendor-free entry, not the vendor.
    const auto up64 = rp::ctx_from_facts<kB, T>(sm120_facts(), 64, 8192, Uplo::Upper,
                                                arch::RoutingProfile::sm_120, true);
    auto nat = routing::parse_pin<Ts>("native", true);
    const auto s3 = routing::select<Ts>(rs, rp::kBound<kB, T>, up64, nat);
    EXPECT_EQ(s3.id(), "native:cta");
    // Nothing native serves Upper 128: the origin pin has no legal tier at all.
    EXPECT_THROW((void)routing::select<Ts>(rs, rp::kBound<kB, T>, up, nat), routing::no_route_error);
}

// Selection latency: shape build + match + plan, offline (no workspace query).
TEST(RoutingRules, SelectLatency) {
    using T = float;
    const auto d = sm120_facts();
    const int N = 200000;
    std::size_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        const auto c = rp::ctx_from_facts<kB, T>(d, 1 + (i % 1024), 8192, Uplo::Lower,
                                                 arch::RoutingProfile::sm_120, true);
        sink += routing::select<rp::Tiers<kB, T>>(routing::potrf_rules::k_sm_120,
                                                  rp::kBound<kB, T>, c, routing::Pin{}).tier;
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        auto s = backend::potrf_op_shape_from_facts<kB, T>(d, 1 + (i % 1024), 1 + (i % 1024), 8192,
                                                          Uplo::Lower, true, true, false, true);
        s.profile = arch::RoutingProfile::sm_120;
        backend::potrf_price_routes<T>(s, d, sycl_trsm::trsm_cta_max_n<T>(), 0, 0);
        sink += static_cast<std::size_t>(
            dispatch::resolve_route_uninstrumented<dispatch::Op::potrf, T>(Route{}, s, true).algo);
    }
    auto t2 = std::chrono::steady_clock::now();
    const double a = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    const double b = std::chrono::duration<double, std::micro>(t2 - t1).count() / N;
    std::printf("select (shape + match + plan): %.3f us/call; today (shape + price + resolve): "
                "%.3f us/call [sink %zu]\n", a, b, sink);
    EXPECT_LT(a, 5.0);
}
