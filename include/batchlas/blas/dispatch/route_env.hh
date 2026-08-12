#pragma once

// One environment vocabulary for route selection, and the legacy spellings that
// map onto it.
//
// Today there are five non-communicating mechanisms:
//
//   BATCHLAS_<OP>_PROVIDER          syev, gesvd, ormqr only   (dispatch/env.hh)
//   BATCHLAS_GEMM_VARIANT           gemm                      (gemm_variant.hh)
//   BATCHLAS_{SYMM,SYRK,SYR2K,TRMM}_VARIANT                   (*_custom_dispatch.cc)
//   ad-hoc per-op knobs             BATCHLAS_ORTHO_GRAM, BATCHLAS_ORMQR_IMPL, ...
//   (and Backend itself, chosen once per Queue)
//
// The canonical spelling becomes BATCHLAS_<OP>_ROUTE, taking either an origin
// ("vendor", "native"), an algorithm ("cta", "expand_gemm", ...), or both
// joined by a colon ("native:register_tiled").
//
// THE LEGACY SPELLINGS MUST KEEP WORKING. They appear in committed benchmark
// scripts and in the provenance of recorded results under output/; silently
// changing what they mean would invalidate measurements that are still being
// compared against. Each therefore maps onto a Route here, and
// parse_route_env() reports which variable it honoured so a diagnostic can
// quote it back.
//
// STATUS: additive. Nothing reads this yet; see WP0_DISPATCH_SPEC.md step S4.

#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch {

inline std::string route_lowercase(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Origin words. Note "netlib" maps to Vendor, not to an algorithm: netlib
// LAPACK is somebody else's code. Provider::Netlib had to be normalised to
// Provider::Vendor by hand in all three consumers for exactly this reason.
inline std::optional<Origin> parse_origin_word(std::string_view w) {
    if (w == "auto") return Origin::Auto;
    if (w == "vendor" || w == "netlib") return Origin::Vendor;
    if (w == "native" || w == "batchlas") return Origin::Native;
    return std::nullopt;
}

inline std::optional<Algorithm> parse_algorithm_word(std::string_view w) {
    if (w == "auto") return Algorithm::Auto;
    if (w == "direct") return Algorithm::Direct;
    if (w == "cta" || w == "batchlas_cta" || w == "batchlas-cta") return Algorithm::CTA;
    if (w == "blocked" || w == "batchlas_blocked" || w == "batchlas-blocked") return Algorithm::Blocked;
    if (w == "two_stage" || w == "two-stage" || w == "batchlas_two_stage" ||
        w == "batchlas-two-stage") return Algorithm::TwoStage;
    if (w == "jacobi" || w == "batchlas_jacobi" || w == "batchlas-jacobi") return Algorithm::Jacobi;
    if (w == "register_tiled" || w == "sycl" || w == "custom") return Algorithm::RegisterTiled;
    if (w == "split_k" || w == "splitk") return Algorithm::SplitK;
    if (w == "expand_gemm" || w == "expand") return Algorithm::ExpandGemm;
    if (w == "triangular_tiles" || w == "triangular") return Algorithm::TriangularTiles;
    if (w == "gram_tiles" || w == "gram") return Algorithm::GramTiles;
    if (w == "fused_device" || w == "cublasdx" || w == "dx" || w == "fused") return Algorithm::FusedDevice;
    if (w == "diag_full_gemm" || w == "gemm") return Algorithm::DiagFullGemm;
    return std::nullopt;
}

// "native:cta" / "vendor" / "cta" -> Route. Unknown text yields nullopt so the
// caller can decide between ignoring it and throwing; the legacy parsers
// silently fell back to Auto, which hid typos.
inline std::optional<Route> parse_route_value(std::string_view raw) {
    const std::string v = route_lowercase(std::string(raw));
    if (v.empty()) return std::nullopt;

    const auto colon = v.find(':');
    if (colon != std::string::npos) {
        const auto o = parse_origin_word(v.substr(0, colon));
        const auto a = parse_algorithm_word(v.substr(colon + 1));
        if (!o || !a) return std::nullopt;
        return Route{*o, *a};
    }

    if (const auto o = parse_origin_word(v)) {
        // A bare origin leaves the algorithm free.
        return Route{*o, Algorithm::Auto};
    }
    if (const auto a = parse_algorithm_word(v)) {
        // A bare algorithm implies Native, EXCEPT the device-library ones,
        // which are vendor code by definition.
        const Origin o = (*a == Algorithm::FusedDevice) ? Origin::Vendor : Origin::Native;
        return Route{o, *a};
    }
    return std::nullopt;
}

// The legacy variable for an op, if it had one, and how its values map.
//
// The mappings are NOT guesses -- each reproduces what the old parser did:
//   * gemm_variant_request() (gemm_variant.hh) returns Vendor when UNSET, and
//     recognises sycl|custom, native|cuda-native|direct-cuda, cublasdx|dx, auto.
//   * parse_cublasdx_variant_request() (route_common.hh) returns AUTO when
//     unset, and recognises vendor, cublasdx|dx|custom, auto.
//   * syrk_route_request() additionally recognises triangular, gram and gemm.
inline std::string_view legacy_variable_for(Op op) {
    switch (op) {
        case Op::gemm:  return "BATCHLAS_GEMM_VARIANT";
        case Op::symm:  return "BATCHLAS_SYMM_VARIANT";
        case Op::syrk:  return "BATCHLAS_SYRK_VARIANT";
        case Op::syr2k: return "BATCHLAS_SYR2K_VARIANT";
        case Op::trmm:  return "BATCHLAS_TRMM_VARIANT";
        case Op::syev:  return "BATCHLAS_SYEV_PROVIDER";
        case Op::gesvd: return "BATCHLAS_GESVD_PROVIDER";
        case Op::ormqr: return "BATCHLAS_ORMQR_PROVIDER";
        default: return {};
    }
}

// What an UNSET legacy variable meant. This asymmetry is real and load-bearing:
// GEMM defaulted to Vendor while the four level-3 ops defaulted to Auto, so the
// level-3 native tile kernels have been running by default and GEMM's has not.
inline Route legacy_unset_default(Op op) {
    switch (op) {
        case Op::gemm: return Route{Origin::Vendor, Algorithm::Auto};
        default:       return Route{Origin::Auto, Algorithm::Auto};
    }
}

// Legacy values whose meaning does NOT match the canonical vocabulary.
//
// THE TRAP: `BATCHLAS_GEMM_VARIANT=native` does not mean "BatchLAS's own
// kernel". It is gemm_variant.hh's alias for `cuda-native` / `direct-cuda` --
// the raw CUDA path -- and GemmVariantRequest::Native is consumed only as an
// EXCLUSION: both gemm_use_sycl_custom and gemm_use_cublasdx_custom return
// false for it, so the call falls through to gemm_vendor_impl. In the canonical
// vocabulary that is Origin::Vendor.
//
// So the same word means opposite things in the two vocabularies. Mapping it
// through the generic parser would flip GEMM from vendor to native for anyone
// who had set it -- silently, and only for that one spelling. Caught by
// tests/route_gemm_equivalence_tests.cc; do not "simplify" this away.
inline std::optional<Route> parse_legacy_route_value(Op op, std::string_view raw) {
    const std::string v = route_lowercase(std::string(raw));

    if (op == Op::gemm) {
        if (v == "native" || v == "cuda-native" || v == "direct-cuda") {
            return Route{Origin::Vendor, Algorithm::Direct};
        }
    }
    return parse_route_value(v);
}

struct ParsedRouteEnv {
    Route route{};
    RouteRequestSource source{};
    bool found = false;      // a variable was set (and parsed)
    bool unparsed = false;   // a variable was set but its value was not understood
};

// Canonical variable first, then the legacy one. Returns found=false when
// neither is set -- the CALLER supplies the default, because that default
// differs per op (see legacy_unset_default).
inline ParsedRouteEnv parse_route_env(Op op) {
    ParsedRouteEnv out;

    const std::string canonical = "BATCHLAS_" + op_env_stem(op) + "_ROUTE";
    if (const char* raw = std::getenv(canonical.c_str()); raw && *raw) {
        out.source = {canonical, raw, false};
        if (const auto r = parse_route_value(raw)) {
            out.route = *r;
            out.found = true;
        } else {
            out.unparsed = true;
        }
        return out;
    }

    const std::string_view legacy = legacy_variable_for(op);
    if (!legacy.empty()) {
        const std::string key(legacy);
        if (const char* raw = std::getenv(key.c_str()); raw && *raw) {
            out.source = {key, raw, true};
            // The LEGACY parser, not the canonical one -- see the note on
            // parse_legacy_route_value: "native" means opposite things.
            if (const auto r = parse_legacy_route_value(op, raw)) {
                out.route = *r;
                out.found = true;
            } else {
                out.unparsed = true;
            }
        }
    }
    return out;
}

} // namespace batchlas::dispatch
