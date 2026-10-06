#pragma once

/// @file
/// @brief The route-selection environment vocabulary: `BATCHLAS_<OP>_ROUTE=origin[:algorithm]`.
///
/// A value is an origin (`auto`, `native`/`batchlas`, `vendor`/`netlib`), an
/// algorithm (implying Native, except `fused_device`, which is Vendor), or
/// both joined by a colon. The legacy per-op spellings (`BATCHLAS_GEMM_VARIANT`,
/// `BATCHLAS_SYEV_PROVIDER`, ...) are read only when the canonical variable is
/// unset, and must keep working: benchmark scripts and recorded results use them.
/// @ingroup dispatch
// evidence: docs/perf/dispatch.md#the-environment-vocabulary

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <string_view>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/settings.hh>

namespace batchlas::dispatch {

/// @addtogroup dispatch
/// @{

/// @brief ASCII lower-casing, as applied to every route value before parsing.
inline std::string route_lowercase(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

/// @brief Parses an origin word; `nullopt` when it is not one.
/// @note "netlib" is an origin, not an algorithm: netlib LAPACK is somebody else's code.
inline std::optional<Origin> parse_origin_word(std::string_view w) {
    if (w == "auto") return Origin::Auto;
    if (w == "vendor" || w == "netlib") return Origin::Vendor;
    if (w == "native" || w == "batchlas") return Origin::Native;
    return std::nullopt;
}

/// @brief Parses an algorithm word, including its aliases (`sycl` and `custom` for RegisterTiled,
///        `gemm` for DiagFullGemm, ...); `nullopt` when it is not one.
inline std::optional<Algorithm> parse_algorithm_word(std::string_view w) {
    if (w == "auto") return Algorithm::Auto;
    if (w == "direct") return Algorithm::Direct;
    if (w == "cta" || w == "batchlas_cta" || w == "batchlas-cta") return Algorithm::CTA;
    if (w == "tiny" || w == "batchlas_tiny" || w == "batchlas-tiny") return Algorithm::Tiny;
    if (w == "lpanel" || w == "batchlas_lpanel" || w == "batchlas-lpanel") return Algorithm::LPanel;
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

/// @brief Parses a canonical `BATCHLAS_<OP>_ROUTE` value.
/// @param raw  `origin`, `algorithm` or `origin:algorithm`, case-insensitive
/// @return the route; `nullopt` for empty or unrecognised text (never Auto, so a typo is visible)
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

/// @brief The pre-Route variable an op still honours, or empty when it never had one.
/// @trap Do not add a case for an op that never had a legacy spelling: that invents one.
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

/// @brief The route an op resolves from when no variable is set: `{Auto, Auto}` for every op.
// GEMM's former Vendor default is gone. evidence: docs/perf/gemm.md#the-auto-flip
inline Route legacy_unset_default(Op op) {
    static_cast<void>(op);
    return Route{Origin::Auto, Algorithm::Auto};
}

/// @brief True for symm, syrk, syr2k and trmm: the ops whose legacy parser knows `custom`, `tiles`, `narrow`, `gemm`.
inline bool is_level3_tile_op(Op op) {
    return op == Op::symm || op == Op::syrk || op == Op::syr2k || op == Op::trmm;
}

/// @brief Parses a legacy variable's value, falling back to the canonical parser.
/// @trap The legacy vocabulary collides with the canonical one, load-bearingly: legacy
///       `native` for gemm is the raw CUDA VENDOR path (the opposite of canonical
///       `native`), and legacy `custom` is the fused cuBLASDx kernel, not the
///       register-tiled GEMM family. Do not "simplify" these away; pinned by
///       tests/route_vocabulary_tests.cc.
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

/// @brief Result of reading an op's route variables.
struct ParsedRouteEnv {
    Route route{};                 ///< the parsed route; meaningful only when `found`
    RouteRequestSource source{};   ///< which variable, and its raw value
    bool found = false;      ///< a variable was set (and parsed)
    bool unparsed = false;   ///< a variable was set but its value was not understood
};

/// @brief Prints a one-time stderr warning for a set-but-unrecognised route variable.
///
/// Reported here rather than by each adapter: every call site spells the
/// fallback `parsed.found ? parsed.route : legacy_unset_default(op)`, which
/// collapses "set but not understood" into "unset", so without this a typo
/// silently resolves to Auto and an A/B driven by it measures the same route
/// twice. Once per variable, because these are read per call.
inline void warn_unparsed_route_env(const RouteRequestSource& src) {
    static std::set<std::string> warned;
    if (!warned.insert(src.variable).second) return;
    std::fprintf(stderr,
                 "batchlas: %s=\"%s\" is not a recognised route; ignoring it and "
                 "using the default. Expected an origin (auto|native|vendor), an "
                 "algorithm, or origin:algorithm.\n",
                 src.variable.c_str(), src.value.c_str());
}

/// @brief Reads `op`'s route request: the canonical variable first, then the legacy one.
///
/// Values come from the batchlas::settings() snapshot, not from `getenv`; the
/// variable NAMES are composed here because RouteRequestSource carries them into
/// diagnostics and tests assert on the literals. An unrecognised value warns once
/// (warn_unparsed_route_env()) and sets `unparsed`.
/// @return `found == false` when neither variable is set or parsed; the CALLER then
///         supplies the default (legacy_unset_default()).
inline ParsedRouteEnv parse_route_env(Op op) {
    ParsedRouteEnv out;

    const std::string canonical = "BATCHLAS_" + op_env_stem(op) + "_ROUTE";
    if (const char* raw = batchlas::settings().routing.canonical_route(op).get();
        raw && *raw) {
        out.source = {canonical, raw, false};
        if (const auto r = parse_route_value(raw)) {
            out.route = *r;
            out.found = true;
        } else {
            out.unparsed = true;
            warn_unparsed_route_env(out.source);
        }
        return out;
    }

    const std::string_view legacy = legacy_variable_for(op);
    if (!legacy.empty()) {
        const std::string key(legacy);
        if (const char* raw = batchlas::settings().routing.legacy_route(op).get();
            raw && *raw) {
            out.source = {key, raw, true};
            if (const auto r = parse_legacy_route_value(op, raw)) {
                out.route = *r;
                out.found = true;
            } else {
                out.unparsed = true;
                warn_unparsed_route_env(out.source);
            }
        }
    }
    return out;
}

/// @}

} // namespace batchlas::dispatch
