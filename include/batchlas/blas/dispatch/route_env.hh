#pragma once

// One environment vocabulary for route selection: BATCHLAS_<OP>_ROUTE, taking an
// origin, an algorithm, or both joined by a colon. The legacy per-op spellings map
// onto it and must keep working: benchmark scripts and recorded results use them.
// evidence: docs/perf/dispatch.md#the-environment-vocabulary

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

// "netlib" is an origin, not an algorithm: netlib LAPACK is somebody else's code.
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

// Unrecognised text yields nullopt rather than Auto, so a typo is visible.
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
        return Route{*o, Algorithm::Auto};
    }
    if (const auto a = parse_algorithm_word(v)) {
        const Origin o = (*a == Algorithm::FusedDevice) ? Origin::Vendor : Origin::Native;
        return Route{o, *a};
    }
    return std::nullopt;
}

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

// Every op's unset default is Auto, i.e. whatever preferred() says; GEMM's former
// Vendor default is gone. evidence: docs/perf/gemm.md#the-auto-flip
inline Route legacy_unset_default(Op op) {
    static_cast<void>(op);
    return Route{Origin::Auto, Algorithm::Auto};
}

// The legacy vocabulary collides with the canonical one, load-bearingly: legacy
// `native` is the raw CUDA VENDOR path (the opposite of canonical "native"), and
// legacy `custom` is the fused cuBLASDx kernel, not the register-tiled GEMM family.
// Do not "simplify" these away; pinned by tests/route_vocabulary_tests.cc.
inline bool is_level3_tile_op(Op op) {
    return op == Op::symm || op == Op::syrk || op == Op::syr2k || op == Op::trmm;
}

inline std::optional<Route> parse_legacy_route_value(Op op, std::string_view raw) {
    const std::string v = route_lowercase(std::string(raw));

    if (op == Op::gemm) {
        if (v == "native" || v == "cuda-native" || v == "direct-cuda") {
            return Route{Origin::Vendor, Algorithm::Direct};
        }
    }

    if (is_level3_tile_op(op)) {
        if (v == "custom") return Route{Origin::Vendor, Algorithm::FusedDevice};
        if (v == "tiles")  return Route{Origin::Native, Algorithm::TriangularTiles};
        if (op == Op::syrk && v == "narrow") {
            return Route{Origin::Native, Algorithm::GramTiles};
        }
        if ((op == Op::syrk || op == Op::syr2k) && v == "gemm") {
            // Deliberately WRONG: it stores BOTH triangles, and exists only as a
            // measurement baseline. Vendor: it runs through the cuBLASDx entry point.
            return Route{Origin::Vendor, Algorithm::DiagFullGemm};
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

// Canonical variable first, then the legacy one; on found=false the CALLER supplies
// the default.
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
