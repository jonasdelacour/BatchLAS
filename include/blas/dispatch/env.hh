#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <string>
#include <string_view>

#include <blas/dispatch/provider.hh>

namespace batchlas::blas::dispatch {

inline std::string uppercase_ascii(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return s;
}

inline std::string lowercase_ascii(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

inline Provider parse_provider_value(const char* v) {
    if (!v || !*v) return Provider::Auto;
    std::string s = lowercase_ascii(std::string(v));

    if (s == "auto") return Provider::Auto;
    if (s == "vendor") return Provider::Vendor;

    if (s == "cta" || s == "batchlas_cta" || s == "batchlas-cta") return Provider::BatchLAS_CTA;
    if (s == "blocked" || s == "batchlas_blocked" || s == "batchlas-blocked") return Provider::BatchLAS_Blocked;
    if (s == "two_stage" || s == "two-stage" || s == "batchlas_two_stage" || s == "batchlas-two-stage") {
        return Provider::BatchLAS_TwoStage;
    }

    if (s == "jacobi" || s == "batchlas_jacobi" || s == "batchlas-jacobi") {
        return Provider::BatchLAS_Jacobi;
    }

    if (s == "netlib") return Provider::Netlib;

    // Unknown value: keep behavior conservative (Auto).
    return Provider::Auto;
}

// Reads BATCHLAS_<OP>_PROVIDER, where <OP> is uppercased.
inline Provider parse_provider_env(const char* opname) {
    if (!opname || !*opname) return Provider::Auto;
    const std::string key = "BATCHLAS_" + uppercase_ascii(std::string(opname)) + "_PROVIDER";
    return parse_provider_value(std::getenv(key.c_str()));
}

inline constexpr std::array<Provider, 6> default_order_cta_blocked_vendor_netlib = {
    Provider::BatchLAS_CTA,
    Provider::BatchLAS_Blocked,
    Provider::BatchLAS_TwoStage,
    Provider::BatchLAS_Jacobi,
    Provider::Vendor,
    Provider::Netlib,
};

// GESVD gets its own order because the one-sided Jacobi kernel (gesvdj_cta)
// dominates the older CTA path on the axis that matters, and the shared order
// buried it behind exactly the path it replaces.
//
// The two paths were measured head-to-head on this tree (RTX 4090, float,
// n = 32, 256 samples, uncontended). Singular-value relative error against the
// known spectrum, `max_relerr` from benchmarks/gesvd_relacc:
//
//   log10(kappa) |     1      2       3       4      5      6
//   CTA          | 1.4e-6 4.7e-5  3.1e-3   0.235  1.046  1.857
//   gesvdj_cta   | 4.8e-6 5.9e-6  1.2e-5  7.1e-5 6.3e-4 5.6e-3
//
// The CTA path forms the normal equations, so it squares the condition number
// and loses the singular VALUES, not merely the vectors -- its orthogonality
// over the same sweep runs 2.8e-6 -> 0.879 -> 4.371 while Jacobi stays flat at
// ~1.3e-5. It is ahead only at kappa = 1e1, where both are ~1e-6 and the
// difference cannot matter.
//
// Speed points the same way wherever U is requested, because the CTA path
// builds U through an assembly + back-transform chain while Jacobi gets it for
// free (U *is* the rotated, normalised A). Time per matrix, float, saturated
// batch, jobu = All: n=8 23x, n=16 4.1x, n=32 parity. Adding U costs CTA 17x at
// n=8 (0.065 -> 1.126 us) and costs Jacobi nothing (0.0448 -> 0.0452 us).
//
// The one regime where CTA is genuinely faster is values-only at n >= 16
// (2.2x at n=32), and that is precisely where it has no correct digits past
// kappa = 1e3. A 2.2x that returns 1.857 relative error is not a default worth
// keeping; it stays reachable via BATCHLAS_GESVD_PROVIDER=cta for callers who
// know their input is well conditioned.
//
// Hermitian input is unaffected: gesvd_supports_jacobi returns false when
// hermitian_uplo is set, so those calls still fall through to the CTA path.
inline constexpr std::array<Provider, 6> default_order_gesvd = {
    Provider::BatchLAS_Jacobi,
    Provider::BatchLAS_CTA,
    Provider::BatchLAS_Blocked,
    Provider::BatchLAS_TwoStage,
    Provider::Vendor,
    Provider::Netlib,
};

// Per-op default order. Ops that do not name themselves here keep the shared
// order; note that Provider::BatchLAS_Jacobi is matched by no chooser other
// than gesvd's, so moving it only affects gesvd.
inline constexpr std::array<Provider, 6> default_order_for_op(const char* opname) {
    if (opname && std::string_view(opname) == "GESVD") {
        return default_order_gesvd;
    }
    return default_order_cta_blocked_vendor_netlib;
}

inline DispatchPolicy policy_from_env(const char* op) {
    DispatchPolicy p;
    p.forced = parse_provider_env(op);
    p.order = default_order_for_op(op);
    // Phase 0: no additional env knobs.
    p.log = false;
    p.require_in_order = false;
    return p;
}

} // namespace batchlas::blas::dispatch
