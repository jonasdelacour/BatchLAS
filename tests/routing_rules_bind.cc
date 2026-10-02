// Compile-only probe for routing::bind<>: a rules name table naming a tier the TierList does
// not register must not compile (ROUTING_BIND_BAD), and the same TU without it must.

#include <batchlas/routing/rules.hh>

namespace probe {
struct Ctx {};
struct A {
    static constexpr std::string_view id = "native:a";
    static constexpr batchlas::dispatch::Route route{batchlas::dispatch::Origin::Native,
                                                     batchlas::dispatch::Algorithm::CTA};
    struct Plan {};
};
struct V {
    static constexpr std::string_view id = "vendor";
    static constexpr batchlas::dispatch::Route route{batchlas::dispatch::Origin::Vendor,
                                                     batchlas::dispatch::Algorithm::Auto};
    struct Plan {};
};
using Tiers = batchlas::routing::TierList<A, V>;

#ifdef ROUTING_BIND_BAD
inline constexpr std::array<std::string_view, 3> kNames{"native:a", "native:removed", "vendor"};
#else
inline constexpr std::array<std::string_view, 2> kNames{"native:a", "vendor"};
#endif
inline constexpr auto kBound = batchlas::routing::bind<Tiers>(kNames);
}  // namespace probe

int routing_rules_bind_probe() { return probe::kBound[0]; }
