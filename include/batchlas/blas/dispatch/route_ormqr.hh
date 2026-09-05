#pragma once

// ORMQR's routing table.
//
// The op it replaces is the smallest of the three Provider-based choosers, and
// it is where the cost of conflating "forced" with "supported" is easiest to
// see. `choose_ormqr_provider` opened with
//
//     Provider chosen = normalize_ormqr_vendor_like(policy.forced);
//     if (chosen != Provider::Auto) return chosen;
//
// -- a forced provider was returned WITHOUT ever being checked against
// ormqr_supports_blocked. Two things followed from that, both fixed here by
// construction rather than by remembering to add a check:
//
//   1. FORCING COULD RUN AN UNSUPPORTED KERNEL. ormqr_supports_blocked is false
//      for complex with Transpose::Trans, and on any non-GPU queue. But
//      ormqr_dispatch's tail is `if (chosen == Vendor) vendor else blocked`, so
//      BATCHLAS_ORMQR_PROVIDER=blocked ran the blocked path on exactly the
//      inputs the predicate exists to exclude.
//
//   2. THE BUFFER SIZE AND THE CALL COULD DISAGREE. For a forced value that is
//      neither Vendor nor Blocked -- cta, two_stage, jacobi, all of which
//      parse -- ormqr_dispatch fell into its `else` arm and reset chosen to
//      Vendor, while ormqr_buffer_size_dispatch's tail is `if (chosen ==
//      Vendor) vendor_size; return blocked_size`, and so returned the BLOCKED
//      size. A caller that sized its workspace with ormqr_buffer_size then hit
//      "ormqr: insufficient workspace for chosen provider" from the very call
//      it had just sized for.
//
// Both come from the same root as the Provider enum itself: a value that means
// "the user asked for this" and a value that means "this can serve the shape"
// were the same value. Splitting `supports` from the forced request makes the
// first impossible, and resolving once through a pure table makes the second
// impossible.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

// Blocked, then the vendor. This is what the shared std::array<Provider, 6>
// order came to for ormqr: BatchLAS_CTA, _TwoStage and _Jacobi were listed but
// matched by no branch in the chooser, so they were inert padding.
inline constexpr Route kOrmqrOrder[] = {
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::ormqr, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // Verbatim ormqr_supports_blocked, and nothing else.
    static bool supports(Route r, const OpShape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;
        if (r.algo != Algorithm::Blocked && r.algo != Algorithm::Auto) return false;

        if (!s.is_gpu) return false;
        if constexpr (is_std_complex_v<T>) {
            // Complex with a plain Trans (as opposed to ConjTrans) is excluded.
            // Transcribed as-is: ormqr_supports_blocked gives no reason for it,
            // and a refactor is not the place to guess at one.
            if (s.transA == Transpose::Trans) return false;
        }
        return true;
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // ormqr has none: no shape ever sent a supported blocked call to the vendor.
    // The old chooser expressed that by putting BatchLAS_Blocked ahead of Vendor
    // in the order and testing only its support predicate, so "preferred" and
    // "supported" coincide. Kept as a distinct function rather than collapsed,
    // because the day someone measures a crossover this is where it goes -- and
    // putting it in `supports` instead would make that crossover a correctness
    // claim, which is the trap this split exists to prevent.
    static bool preferred(Route r, const OpShape& s) {
        return is_native(r) && supports(r, s);
    }

    static constexpr const Route* order_begin() { return kOrmqrOrder; }
    static constexpr const Route* order_end() {
        return kOrmqrOrder + (sizeof(kOrmqrOrder) / sizeof(kOrmqrOrder[0]));
    }
};

template <typename T>
inline Route resolve_ormqr_route(Route forced, const OpShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::ormqr, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
