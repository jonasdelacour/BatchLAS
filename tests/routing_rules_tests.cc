// The rules-as-data engine against today's resolver, offline (no GPU): described device facts,
// the committed RuleSets, and resolve_route over the shape the library would build. Runs in a
// vendor build and in a -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF build (the vendor is then illegal
// even when a cell asks for it, which is exactly the vendor-free walk).

#include "../src/routing/potrf_tiers.hh"
#include "../src/routing/getrs_tiers.hh"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace batchlas;
namespace rp = batchlas::routing::potrf;
using dispatch::Route;
using launch_plan::DeviceFacts;
using routing::LiveCause;
using routing::Reason;

namespace {

constexpr Backend kB = Backend::CUDA;
constexpr bool kVendorBuilt = dispatch::solver_vendor_available<kB>;

DeviceFacts facts(int cus, std::size_t local_mem, int max_wg) {
    DeviceFacts d;
    d.compute_units = cus;
    d.local_mem_bytes = local_mem;
    d.max_wg_size = max_wg;
    return d;
}
const DeviceFacts kSm120 = facts(188, 101376, 1024);   // the profile's own SKU
const DeviceFacts kSm89 = facts(128, 101376, 1024);

// Independent of rp::legacy_tier_id, which is under test.
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

class ScopedEnv {
public:
    ScopedEnv(const char* k, const char* v) : k_(k) {
        if (const char* o = std::getenv(k)) old_ = o, had_ = true;
        if (v) setenv(k, v, 1); else unsetenv(k);
    }
    ~ScopedEnv() {
        if (had_) setenv(k_, old_.c_str(), 1); else unsetenv(k_);
    }
private:
    const char* k_;
    std::string old_;
    bool had_ = false;
};

// Today's library decision for the same described device, Auto or a forced route.
template <class T>
std::string oracle(const DeviceFacts& d, int64_t n, int64_t b, Uplo u, arch::RoutingProfile prof,
                   bool vendor, Route forced = Route{}) {
    auto s = backend::potrf_op_shape_from_facts<kB, T>(d, n, n, b, u, true, true, false,
                                                       sycl_potrf::potrf_blocked_available<T>());
    s.profile = prof;
    backend::potrf_price_routes<T>(s, d, sycl_trsm::trsm_cta_max_n<T>(), 0, 0);
    return name_of(dispatch::resolve_route_uninstrumented<dispatch::Op::potrf, T>(
        forced, s, vendor && kVendorBuilt));
}

struct Got {
    std::string id;
    Reason reason = Reason::Rule;
    LiveCause live = LiveCause::None;
};

template <class T, class Ts = rp::Tiers<kB, T>>
Got engine(const rp::Ctx<T>& c, const routing::Pin& pin = {},
           const routing::RuleSet* rs = nullptr, bool use_rs = false) {
    static constexpr auto bound = routing::bind<Ts>(routing::potrf_rules::kNames);
    try {
        const auto s = routing::select<Ts>(use_rs ? rs : rp::rules_for<T>(c), "potrf", bound, c,
                                           pin, [](const rp::Ctx<T>& x) {
                                               return rp::live_rule<kB, T>(x);
                                           });
        return {std::string(s.id()), s.reason, s.live};
    } catch (const routing::no_route_error&) {
        return {"vendor", Reason::NoRule, LiveCause::None};   // the resolver's "nothing serves"
    }
}

template <class T>
Got engine(const DeviceFacts& d, int64_t n, int64_t b, Uplo u, arch::RoutingProfile prof,
           bool vendor, std::string_view arch = {}) {
    return engine<T>(rp::ctx_from_facts<kB, T>(d, n, b, u, prof, vendor, arch));
}

std::vector<int64_t> grid_batches() {
    std::vector<int64_t> v;
    for (int e = 0; e <= 18; ++e) v.push_back(int64_t(1) << e);
    return v;
}

int env_nmax() {
    const char* v = std::getenv("ROUTING_RULES_NMAX");
    return v ? std::atoi(v) : 2048;
}

struct Tally {
    int cells = 0, bad = 0, no_rule = 0, contested = 0, no_ruleset = 0, hole = 0;
    void add(const Got& g) {
        ++cells;
        no_rule += g.reason == Reason::NoRule && g.live == LiveCause::Hole;
        contested += g.live == LiveCause::Contested;
        no_ruleset += g.live == LiveCause::NoRuleSet;
        hole += g.live == LiveCause::Hole;
    }
};

template <class T>
void check_grid(const char* dt, const DeviceFacts& d, arch::RoutingProfile prof,
                std::string_view arch, int nmax, Tally& t) {
    for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
        for (int n = 1; n <= nmax; ++n) {
            for (int64_t b : grid_batches()) {
                for (bool vendor : {true, false}) {
                    const auto want = oracle<T>(d, n, b, u, prof, vendor);
                    const auto got = engine<T>(d, n, b, u, prof, vendor, arch);
                    t.add(got);
                    if (want != got.id && ++t.bad <= 10) {
                        std::printf("MISMATCH %s %c n=%d b=%lld vendor=%d: today %s, rules %s\n", dt,
                                    u == Uplo::Upper ? 'U' : 'L', n, static_cast<long long>(b),
                                    vendor, want.c_str(), got.id.c_str());
                    }
                }
            }
        }
    }
}

Tally grid_all(const DeviceFacts& d, arch::RoutingProfile prof, std::string_view arch, int nmax) {
    Tally t;
    check_grid<float>("float", d, prof, arch, nmax, t);
    check_grid<double>("double", d, prof, arch, nmax, t);
    check_grid<std::complex<float>>("cfloat", d, prof, arch, nmax, t);
    check_grid<std::complex<double>>("cdouble", d, prof, arch, nmax, t);
    return t;
}

}  // namespace

TEST(RoutingRules, BuildFlavour) {
    std::printf("vendor library linked: %s\n", kVendorBuilt ? "yes" : "NO (vendor-free build)");
}

// (i) Every grid cell, on each device the rules were compiled for, both vendor modes. The
// rules must DECIDE there: a live fallback everywhere would also match, vacuously.
TEST(RoutingRules, GridMatchesResolver) {
    struct Cfg { const char* what; DeviceFacts d; arch::RoutingProfile p; const char* arch; };
    const Cfg cfgs[] = {
        {"sm_120 model on its SKU", kSm120, arch::RoutingProfile::sm_120, "sm_120"},
        {"sm_89 profile forced on the sm_120 SKU (hand)", kSm120, arch::RoutingProfile::sm_89, "sm_89"},
        {"sm_89 hand on its SKU", kSm89, arch::RoutingProfile::sm_89, "sm_89"},
    };
    for (const auto& c : cfgs) {
        const Tally t = grid_all(c.d, c.p, c.arch, env_nmax());
        std::printf("%s: %d cells, %d mismatches, %d contested (live), %d holes, %d no-ruleset\n",
                    c.what, t.cells, t.bad, t.contested, t.hole, t.no_ruleset);
        EXPECT_EQ(t.bad, 0) << c.what;
        EXPECT_EQ(t.hole, 0) << c.what << ": a rule set with a hole";
        EXPECT_EQ(t.no_ruleset, 0) << c.what << ": no RuleSet decided on its own device";
        EXPECT_LT(t.contested, t.cells / 20) << c.what << ": almost everything priced live";
    }
}

// (ii) Off-grid: log-uniform batch, uniform n. Mismatches are counted only where a RULE
// decided; live-priced shapes agree by construction and are reported as a fraction.
TEST(RoutingRules, OffGridFidelity) {
    std::mt19937_64 rng(11);
    std::uniform_int_distribution<int> nd(1, 2048), dt(0, 3), ud(0, 1);
    std::uniform_real_distribution<double> ld(0, 18);
    int bad[2] = {0, 0}, live[2] = {0, 0};
    const int N = 10000;
    for (int i = 0; i < N; ++i) {
        const int n = nd(rng);
        const int64_t b = std::max<int64_t>(1, static_cast<int64_t>(std::pow(2.0, ld(rng))));
        const Uplo u = ud(rng) ? Uplo::Upper : Uplo::Lower;
        const int t = dt(rng);
        int k = 0;
        for (auto prof : {arch::RoutingProfile::sm_120, arch::RoutingProfile::sm_89}) {
            std::string w;
            Got g;
            switch (t) {
                case 0: w = oracle<float>(kSm120, n, b, u, prof, true);
                        g = engine<float>(kSm120, n, b, u, prof, true); break;
                case 1: w = oracle<double>(kSm120, n, b, u, prof, true);
                        g = engine<double>(kSm120, n, b, u, prof, true); break;
                case 2: w = oracle<std::complex<float>>(kSm120, n, b, u, prof, true);
                        g = engine<std::complex<float>>(kSm120, n, b, u, prof, true); break;
                default: w = oracle<std::complex<double>>(kSm120, n, b, u, prof, true);
                         g = engine<std::complex<double>>(kSm120, n, b, u, prof, true); break;
            }
            if (g.live != LiveCause::None) ++live[k];
            else if (w != g.id) ++bad[k];
            ++k;
        }
    }
    std::printf("off-grid, rule-decided disagreement: model %d/%d (%.3f%%), %.2f%% priced live; "
                "hand %d/%d, %.2f%% live\n", bad[0], N, 100.0 * bad[0] / N, 100.0 * live[0] / N,
                bad[1], N, 100.0 * live[1] / N);
    EXPECT_EQ(bad[1], 0);
    EXPECT_LE(bad[0], N / 1000) << "design target: <= 0.1% of off-grid shapes";
    EXPECT_LT(live[0], N / 10) << "the contested-band probe made most shapes live";
}

// (iii) Rules are a cache of ONE device's policy. Other facts of the same family (or another
// arch) must not read them: everything is priced live, and so matches today. The counted
// divergence shows what reading the sm_120 rules there would have cost.
TEST(RoutingRules, OtherDeviceFactsPriceLive) {
    struct Cfg { const char* what; DeviceFacts d; arch::RoutingProfile p; const char* arch; };
    const Cfg cfgs[] = {
        {"sm_120 family, 170 CUs", facts(170, 101376, 1024), arch::RoutingProfile::sm_120, "sm_120"},
        {"sm_120 family, 84 CUs", facts(84, 101376, 1024), arch::RoutingProfile::sm_120, "sm_120"},
        {"sm_120 family, 48 CUs", facts(48, 101376, 1024), arch::RoutingProfile::sm_120, "sm_120"},
        {"cc 100 (B200-like), nearest sm_120", facts(148, 232448, 1024), arch::RoutingProfile::sm_120, "sm_100"},
        {"cc 90 (H100-like), nearest sm_89", facts(132, 232448, 1024), arch::RoutingProfile::sm_89, "sm_90"},
    };
    const int nmax = std::min(env_nmax(), 1024);
    for (const auto& c : cfgs) {
        Tally t;
        int would_differ = 0, rules_decided = 0;
        auto one = [&](auto z, int n, int64_t b, Uplo u, bool vendor) {
            using T = decltype(z);
            const auto ctx = rp::ctx_from_facts<kB, T>(c.d, n, b, u, c.p, vendor, c.arch);
            const auto want = oracle<T>(c.d, n, b, u, c.p, vendor);
            const auto got = engine<T>(ctx);
            t.add(got);
            rules_decided += rp::rules_for<T>(ctx) != nullptr;
            if (want != got.id) ++t.bad;
            const auto frozen = engine<T>(ctx, {}, &routing::potrf_rules::k_sm_120, true);
            if (frozen.live == LiveCause::None && frozen.id != want) ++would_differ;
        };
        for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
            for (int n = 1; n <= nmax; ++n) {
                for (int64_t b : grid_batches()) {
                    for (bool vendor : {true, false}) {
                        one(float{}, n, b, u, vendor);
                        one(double{}, n, b, u, vendor);
                        one(std::complex<float>{}, n, b, u, vendor);
                        one(std::complex<double>{}, n, b, u, vendor);
                    }
                }
            }
        }
        std::printf("%s: %d cells, %d decided by a RuleSet, mismatches %d; reading the sm_120 "
                    "rules would differ on %d (%.2f%%)\n", c.what, t.cells, rules_decided, t.bad,
                    would_differ, 100.0 * would_differ / t.cells);
        EXPECT_EQ(rules_decided, 0) << c.what << ": a RuleSet compiled for other facts decided";
        EXPECT_GT(t.no_ruleset, t.cells / 2) << c.what;
        EXPECT_EQ(t.bad, 0) << c.what;
        if (std::string(c.arch) == "sm_120") EXPECT_GT(would_differ, 0) << "vacuous facts check";
    }
}

// The blocked NB/W environment overrides change the policy the rules were compiled under.
TEST(RoutingRules, BlockingOverridesPriceLive) {
    auto c = rp::ctx_from_facts<kB, float>(kSm120, 600, 1024, Uplo::Lower,
                                           arch::RoutingProfile::sm_120, true);
    EXPECT_NE(rp::rules_for<float>(c), nullptr);
    c.nb_env = 32;
    EXPECT_EQ(rp::rules_for<float>(c), nullptr);
    EXPECT_EQ(engine<float>(c).live, LiveCause::NoRuleSet);
}

// (iv) Flip ONE rule's first two candidates: exactly that box's grid cells turn red. In a
// vendor-free build the vendor entry is inert, so the rule's rank becomes `native:blocked`
// instead, and the red set is the box's cells where today's walk is not blocked.
TEST(RoutingRules, GuardFlipOneRule) {
    using T = float;
    const auto& base = routing::potrf_rules::k_sm_120;
    std::vector<routing::Rule> copy(base.begin, base.end);
    const auto c0 = rp::ctx_from_facts<kB, T>(kSm120, 128, 8192, Uplo::Lower,
                                              arch::RoutingProfile::sm_120, true);
    const routing::Rule* hit = base.match(c0.key(), c0.features());
    ASSERT_NE(hit, nullptr);
    const std::size_t idx = static_cast<std::size_t>(hit - base.begin);
    ASSERT_GE(copy[idx].nrank, 2);
    if (kVendorBuilt) {
        std::swap(copy[idx].rank[0], copy[idx].rank[1]);
    } else {
        copy[idx].rank[0] = routing::Candidate{rp::name_index("native:blocked"), {}};
        copy[idx].nrank = 1;
    }
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
            const auto c = rp::ctx_from_facts<kB, T>(kSm120, n, b, Uplo::Lower,
                                                     arch::RoutingProfile::sm_120, true);
            const auto want = oracle<T>(kSm120, n, b, Uplo::Lower, arch::RoutingProfile::sm_120, true);
            inside += in && (kVendorBuilt || want != "native:blocked");
            if (want != engine<T>(c, {}, &mutated, true).id) {
                ++red;
                if (!in) ++red_outside;
            }
        }
    }
    std::printf("red %d of %d cells inside the box, %d outside\n", red, inside, red_outside);
    EXPECT_GT(inside, 0);
    EXPECT_EQ(red, inside);
    EXPECT_EQ(red_outside, 0);
}

// (v) A hole is not masked: delete the rule holding float L n=256 b=8192 and the cells of
// that box report reason no_rule / live cause hole (the answer stays today's, priced live).
// GridMatchesResolver's `hole == 0` is the assertion this break turns red.
TEST(RoutingRules, HoleIsVisible) {
    using T = float;
    const auto& base = routing::potrf_rules::k_sm_120;
    const auto c0 = rp::ctx_from_facts<kB, T>(kSm120, 256, 8192, Uplo::Lower,
                                              arch::RoutingProfile::sm_120, true);
    const routing::Rule* hit = base.match(c0.key(), c0.features());
    ASSERT_NE(hit, nullptr);
    std::vector<routing::Rule> copy;
    for (const auto* r = base.begin; r != base.end; ++r) if (r != hit) copy.push_back(*r);
    std::vector<std::uint32_t> off(base.key_off, base.key_off + base.nkeys + 1);
    for (std::size_t k = c0.key() + 1; k < off.size(); ++k) --off[k];
    routing::RuleSet holed = base;
    holed.begin = copy.data();
    holed.end = copy.data() + copy.size();
    holed.key_off = off.data();
    int holes = 0, cells = 0, wrong = 0;
    for (int n = static_cast<int>(hit->lo[0]); n <= std::min<int64_t>(hit->hi[0], 2048); ++n) {
        for (int64_t b : grid_batches()) {
            if (b < hit->lo[1] || b > hit->hi[1]) continue;
            const auto c = rp::ctx_from_facts<kB, T>(kSm120, n, b, Uplo::Lower,
                                                     arch::RoutingProfile::sm_120, true);
            const auto g = engine<T>(c, {}, &holed, true);
            ++cells;
            holes += g.live == LiveCause::Hole && g.reason == Reason::NoRule;
            wrong += g.id != oracle<T>(kSm120, n, b, Uplo::Lower, arch::RoutingProfile::sm_120, true);
        }
    }
    std::printf("deleted R%04u: %d cells, %d reported as holes, %d wrong\n", hit->id, cells, holes,
                wrong);
    EXPECT_GT(cells, 0);
    EXPECT_EQ(holes, cells);
    EXPECT_EQ(wrong, 0);
}

// Every registered tier is named by some RuleSet, or it is a finished tier nothing can select.
TEST(RoutingRules, NoUnrankedTier) {
    using Ts = rp::Tiers<kB, float>;
    std::set<std::string> named;
    for (const auto* rs : routing::potrf_rules::kSets) {
        for (const auto* r = rs->begin; r != rs->end; ++r) {
            for (int i = 0; i < r->nrank; ++i) {
                named.insert(std::string(routing::potrf_rules::kNames[r->rank[i].name]));
            }
        }
    }
    for (auto id : routing::potrf_rules::kPinnableOnly) named.insert(std::string(id));
    for (auto id : Ts::ids) EXPECT_TRUE(named.count(std::string(id))) << id << " is never ranked";
}

// (vi) legal() off the compile facts. Each tier's legal() must equal its RouteTable gate (the
// independently tested correctness predicate; cta_wg: CTA's gate AND its own geometry), and a
// legal tier's plan must fit the device. Ranks only name tiers legal on the compile facts, so
// these devices are where a loosened legal() would launch an unfit kernel.
namespace {
template <class T>
int legality_sweep(const DeviceFacts& d, const char* what) {
    using Ts = rp::Tiers<kB, T>;
    int bad = 0;
    std::vector<int> ns;
    for (int n = 1; n <= 600; ++n) ns.push_back(n);
    for (int n : {700, 744, 745, 1024, 1500, 2048, 4096}) ns.push_back(n);
    for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
        for (int n : ns) {
            for (int64_t b : {1, 7, 1024, 65536}) {
                for (bool vendor : {true, false}) {
                    const auto c = rp::ctx_from_facts<kB, T>(d, n, b, u, arch::RoutingProfile::sm_120,
                                                             vendor);
                    for (std::size_t t = 0; t < Ts::size; ++t) {
                        const bool legal = Ts::legal(t, c);
                        bool want;
                        if (dispatch::is_vendor(Ts::routes[t])) {
                            want = kVendorBuilt && vendor;
                        } else if (Ts::ids[t] == "native:cta_wg") {
                            want = rp::Tbl<T>::supports(Ts::routes[t], c.s) &&
                                   potrf_plan::cta_wg_geometry<T>(n, b, d).fits;
                        } else {
                            want = rp::Tbl<T>::supports(Ts::routes[t], c.s);
                        }
                        const bool fits = Ts::fits(Ts::plan(t, c, {}));
                        if (legal != want || (legal && !fits)) {
                            if (++bad <= 6) {
                                std::printf("LEGALITY %s %s %c n=%d b=%lld: legal=%d gate=%d fits=%d\n",
                                            what, std::string(Ts::ids[t]).c_str(),
                                            u == Uplo::Upper ? 'U' : 'L', n,
                                            static_cast<long long>(b), legal, want, fits);
                            }
                        }
                    }
                }
            }
        }
    }
    return bad;
}
}  // namespace

TEST(RoutingRules, LegalMatchesGateAndFitsOffCompileFacts) {
    struct Cfg { const char* what; DeviceFacts d; };
    const Cfg cfgs[] = {
        {"sm_120", kSm120}, {"sm_89", kSm89}, {"170 CUs", facts(170, 101376, 1024)},
        {"48 KB local", facts(188, 49152, 1024)}, {"64 KB local", facts(132, 65536, 1024)},
        {"228 KB local", facts(148, 232448, 1024)}, {"max_wg 512", facts(188, 101376, 512)},
    };
    for (const auto& c : cfgs) {
        int bad = legality_sweep<float>(c.d, c.what) + legality_sweep<double>(c.d, c.what) +
                  legality_sweep<std::complex<float>>(c.d, c.what) +
                  legality_sweep<std::complex<double>>(c.d, c.what);
        EXPECT_EQ(bad, 0) << c.what;
    }
}

// (vii) Pin words. Each spelling today's BATCHLAS_POTRF_ROUTE accepts names the tier it named
// before -- "cta" is the sub-group CTA kernel, never cta_wg, whatever the TierList order --
// unknown words are refused (not Auto), and BATCHLAS_ROUTING_STRICT turns refusal into a throw.
TEST(RoutingRules, PinWordsNameTheirTier) {
    using Ts = rp::Tiers<kB, float>;
    struct W { const char* word; const char* tier; bool origin_only; bool active; };
    const W words[] = {
        {"cta", "native:cta", false, true},          {"CTA", "native:cta", false, true},
        {"native:cta", "native:cta", false, true},   {"batchlas_cta", "native:cta", false, true},
        {"Native:CTA", "native:cta", false, true},   {"tiny", "native:tiny", false, true},
        {"lpanel", "native:lpanel", false, true},    {"native:lpanel", "native:lpanel", false, true},
        {"blocked", "native:blocked", false, true},  {"batchlas-blocked", "native:blocked", false, true},
        {"vendor", "vendor", false, true},           {"netlib", "vendor", false, true},
        {"vendor:auto", "vendor", false, true},      {"native:cta_wg", "native:cta_wg", false, true},
        {"native", "", true, true},                  {"NATIVE", "", true, true},
        {"batchlas", "", true, true},                {"auto", "", false, false},
        {"bogus", "", false, true},
    };
    for (const auto& w : words) {
        ScopedEnv e("BATCHLAS_POTRF_ROUTE", w.word);
        const auto p = rp::pin_from_env<Ts>();
        EXPECT_EQ(p.active, w.active) << w.word;
        EXPECT_EQ(p.origin_only, w.origin_only) << w.word;
        const std::string got = p.tier >= 0 ? std::string(Ts::ids[p.tier]) : std::string();
        EXPECT_EQ(got, w.tier) << w.word;
    }
    for (Route r : dispatch::kPotrfOrder) {
        EXPECT_GE(Ts::index_of(rp::legacy_tier_id(r)), 0) << "legacy route maps to no tier";
    }
    EXPECT_EQ(rp::legacy_tier_id({dispatch::Origin::Native, dispatch::Algorithm::CTA}), "native:cta");
    {
        ScopedEnv e("BATCHLAS_POTRF_ROUTE", "lpanel");
        ScopedEnv s("BATCHLAS_ROUTING_STRICT", "1");
        const auto p = rp::pin_from_env<Ts>();
        EXPECT_TRUE(p.strict);
        const auto up = rp::ctx_from_facts<kB, float>(kSm120, 128, 8192, Uplo::Upper,
                                                      arch::RoutingProfile::sm_120, true);
        EXPECT_THROW(((void)rp::select_ctx<kB, float>(up, p)), std::invalid_argument);
    }
    {
        ScopedEnv e("BATCHLAS_POTRF_ROUTE", "lpanel");
        ScopedEnv s("BATCHLAS_ROUTING_STRICT", "0");
        EXPECT_FALSE(rp::pin_from_env<Ts>().strict);
    }
}

// (viii) Pinned calls against today's resolver with the same forced route: a pin the shape
// cannot take falls back to Auto in both (here loudly), a bare origin takes the vendor-free walk.
namespace {
template <class T, class Ts = rp::Tiers<kB, T>>
int pin_sweep(const char* word, const char* dt, int& cells) {
    ScopedEnv e("BATCHLAS_POTRF_ROUTE", word);
    const auto pin = rp::pin_from_env<Ts>();
    const auto parsed = dispatch::parse_route_value(word);
    const Route forced = parsed ? *parsed : Route{};
    std::vector<int> ns;
    for (int n = 1; n <= 80; ++n) ns.push_back(n);
    for (int n : {100, 128, 200, 256, 257, 300, 368, 369, 500, 744, 745, 1024, 2048}) ns.push_back(n);
    int bad = 0;
    for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
        for (int n : ns) {
            for (int64_t b : {1, 64, 4096, 65536}) {
                for (bool vendor : {true, false}) {
                    for (auto prof : {arch::RoutingProfile::sm_120, arch::RoutingProfile::sm_89}) {
                        ++cells;
                        const auto c = rp::ctx_from_facts<kB, T>(kSm120, n, b, u, prof, vendor);
                        const auto want = oracle<T>(kSm120, n, b, u, prof, vendor, forced);
                        const auto got = engine<T, Ts>(c, pin);
                        if (want != got.id && ++bad <= 5) {
                            std::printf("PIN %s %s %c n=%d b=%lld vendor=%d: today %s, rules %s\n",
                                        word, dt, u == Uplo::Upper ? 'U' : 'L', n,
                                        static_cast<long long>(b), vendor, want.c_str(),
                                        got.id.c_str());
                        }
                    }
                }
            }
        }
    }
    return bad;
}
}  // namespace

TEST(RoutingRules, PinsMatchResolver) {
    int cells = 0, bad = 0;
    for (const char* w : {"cta", "native:cta", "lpanel", "blocked", "tiny", "vendor", "native",
                          "bogus"}) {
        bad += pin_sweep<float>(w, "float", cells);
        bad += pin_sweep<double>(w, "double", cells);
        bad += pin_sweep<std::complex<float>>(w, "cfloat", cells);
        bad += pin_sweep<std::complex<double>>(w, "cdouble", cells);
    }
    std::printf("pinned: %d decisions, %d differ from today's forced resolve\n", cells, bad);
    EXPECT_EQ(bad, 0);
}

// (ix) TierList order is not policy: the same tiers registered in reverse decide identically,
// Auto and pinned.
TEST(RoutingRules, TierOrderIsNotPolicy) {
    using T = float;
    using Rev = routing::TierList<rp::CtaWgTier<T>, rp::VendorTier<kB, T>, rp::BlockedTier<kB, T>,
                                  rp::LpanelTier<T>, rp::CtaTier<T>, rp::TinyTier<T>>;
    int cells = 0, bad = 0;
    for (const char* w : {"", "cta", "lpanel", "native", "bogus"}) {
        ScopedEnv e("BATCHLAS_POTRF_ROUTE", *w ? w : nullptr);
        const auto pf = rp::pin_from_env<rp::Tiers<kB, T>>();
        const auto pr = rp::pin_from_env<Rev>();
        for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
            for (int n = 1; n <= 400; ++n) {
                for (int64_t b : {1, 300, 8192}) {
                    for (bool vendor : {true, false}) {
                        const auto c = rp::ctx_from_facts<kB, T>(kSm120, n, b, u,
                                                                 arch::RoutingProfile::sm_120, vendor);
                        ++cells;
                        bad += engine<T>(c, pf).id != engine<T, Rev>(c, pr).id;
                    }
                }
            }
        }
    }
    std::printf("reversed TierList: %d decisions, %d differ\n", cells, bad);
    EXPECT_EQ(bad, 0);
}

// ---- getrs: the second op, (n, nrhs, batch) x (dtype, trans), hand rules from route_oracle --

namespace {

constexpr bool kGetrsVendorBuilt = dispatch::factorization_vendor_available<kB>;

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
        const auto want = getrs_name(dispatch::resolve_route_uninstrumented<dispatch::Op::getrs, T>(
            Route{}, c.s, vendor && kGetrsVendorBuilt));
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
    const auto up = rp::ctx_from_facts<kB, T>(kSm120, 128, 8192, Uplo::Upper,
                                              arch::RoutingProfile::sm_120, true);
    auto lp = routing::parse_pin<Ts>("native:lpanel", false);
    if (kVendorBuilt) {
        const auto s1 = rp::select_ctx<kB, T>(up, lp);
        EXPECT_EQ(s1.reason, Reason::PinRefused);
        EXPECT_EQ(s1.id(), "vendor");
        EXPECT_EQ(s1.forced, "native:lpanel");
    } else {
        // Vendor-free: nothing serves Upper 128 at all, pin or no pin.
        EXPECT_THROW(((void)rp::select_ctx<kB, T>(up, lp)), routing::no_route_error);
    }
    auto strict = routing::parse_pin<Ts>("native:lpanel", true);
    EXPECT_THROW(((void)rp::select_ctx<kB, T>(up, strict)), std::invalid_argument);
    const auto lo = rp::ctx_from_facts<kB, T>(kSm120, 128, 8192, Uplo::Lower,
                                              arch::RoutingProfile::sm_120, true);
    const auto s2 = rp::select_ctx<kB, T>(lo, routing::parse_pin<Ts>("native:blocked", true));
    EXPECT_EQ(s2.reason, Reason::Pinned);
    EXPECT_EQ(s2.id(), "native:blocked");
    // A bare origin restricts the ranking: the rule's vendor-free entry, not the vendor.
    const auto up64 = rp::ctx_from_facts<kB, T>(kSm120, 64, 8192, Uplo::Upper,
                                                arch::RoutingProfile::sm_120, true);
    EXPECT_EQ((rp::select_ctx<kB, T>(up64, routing::parse_pin<Ts>("native", true)).id()),
              "native:cta");
    // Nothing native serves Upper 128 (n > the CTA ceiling): refused, strict throws.
    EXPECT_THROW(((void)rp::select_ctx<kB, T>(up, routing::parse_pin<Ts>("native", true))),
                 std::invalid_argument);
}

// Selection latency: shape build + match + plan, offline (no workspace query).
TEST(RoutingRules, SelectLatency) {
    using T = float;
    const int N = 200000;
    std::size_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        const auto c = rp::ctx_from_facts<kB, T>(kSm120, 1 + (i % 1024), 8192, Uplo::Lower,
                                                 arch::RoutingProfile::sm_120, true);
        sink += rp::select_ctx<kB, T>(c, routing::Pin{}).tier;
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        auto s = backend::potrf_op_shape_from_facts<kB, T>(kSm120, 1 + (i % 1024), 1 + (i % 1024),
                                                          8192, Uplo::Lower, true, true, false, true);
        s.profile = arch::RoutingProfile::sm_120;
        backend::potrf_price_routes<T>(s, kSm120, sycl_trsm::trsm_cta_max_n<T>(), 0, 0);
        sink += static_cast<std::size_t>(
            dispatch::resolve_route_uninstrumented<dispatch::Op::potrf, T>(Route{}, s, true).algo);
    }
    auto t2 = std::chrono::steady_clock::now();
    const double a = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    const double b = std::chrono::duration<double, std::micro>(t2 - t1).count() / N;
    std::printf("select (shape + match + plan, live where contested): %.3f us/call; today "
                "(shape + price + resolve): %.3f us/call [sink %zu]\n", a, b, sink);
    EXPECT_LT(a, 5.0);
}
