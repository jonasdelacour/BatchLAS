#pragma once

/// @file
/// @brief Tuning constants, bucketed by problem order n, with runtime `BATCHLAS_TUNE_*`
/// overrides.
///
/// Each knob has five constants (`_TINY` n <= 64, `_SMALL` <= 128, `_MEDIUM` <= 256, `_LARGE`
/// <= 512, `_XLARGE` above), a constexpr table `<knob>_default_for_n(n)`, and a runtime accessor
/// `<knob>_for_n(n)` that consults its `BATCHLAS_TUNE_*` override first. The constants were
/// measured on sm_89 in float only.
/// @warning Do not change a `BATCHLAS_TUNE_*` variable between a `*_buffer_size()` query and its
///          call: the accessors feed both, and the workspace would be sized for another width.
/// @see @ref perf_tuning
/// @ingroup api_config
// GENERATED constants (evaluation/tuning/generate_tuning_header.py); regenerate, do not hand-edit.
// The accessors, this include and these doc comments are mirrored in the generator's template:
// change both or a retune reverts it. evidence: docs/perf/tuning.md#tuning-regenerating-the-header

#include <cstdint>
#include <cstdlib>

#include <batchlas/settings.hh>

namespace batchlas::tuning {

namespace detail {

/// @brief The captured override if it is a clean positive 32-bit integer, else `fallback`.
///
/// Unset, empty, unparseable, trailing garbage (`"16x"`), non-positive or out-of-range values all
/// yield `fallback`, the compiled constant.
/// @param captured the captured `BATCHLAS_TUNE_*` value from settings()
/// @param fallback the compiled constant for this n
/// @ingroup api_config_lowlevel
// strtol stays HERE, not in settings.cc: env_int_or reads "16x" as 16, exactly what a retune
// harness types. evidence: docs/perf/tuning.md#tuning-runtime-overrides-of-the-constants
inline int32_t tuning_env_override(const EnvValue& captured, int32_t fallback) {
    const char* v = captured.get();
    if (v == nullptr || *v == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0') return fallback;   // not a clean integer
    if (parsed <= 0 || parsed > 2147483647L) return fallback;  // non-positive / out of range
    return static_cast<int32_t>(parsed);
}

} // namespace detail

inline constexpr int32_t ORMQR_BLOCK_SIZE_TINY = 16;
inline constexpr int32_t ORMQR_BLOCK_SIZE_SMALL = 16;
inline constexpr int32_t ORMQR_BLOCK_SIZE_MEDIUM = 24;
inline constexpr int32_t ORMQR_BLOCK_SIZE_LARGE = 48;
inline constexpr int32_t ORMQR_BLOCK_SIZE_XLARGE = 56;

// gebrd's panel width: deliberately NOT the ormqr constant (opposite gradients).
// evidence: docs/perf/gesvd.md#gesvd-the-gebrd-panel-width-split-from-ormqr
inline constexpr int32_t GEBRD_BLOCK_SIZE_TINY = 16;
inline constexpr int32_t GEBRD_BLOCK_SIZE_SMALL = 8;
inline constexpr int32_t GEBRD_BLOCK_SIZE_MEDIUM = 8;
inline constexpr int32_t GEBRD_BLOCK_SIZE_LARGE = 16;
inline constexpr int32_t GEBRD_BLOCK_SIZE_XLARGE = 16;

// sb2st wave back-transform geometry. 0 = keep the shape-adaptive heuristic in
// sytrd_sb2st_hh.cc, and ALL ship 0 on purpose: the heuristic already matches the tuned winner.
// Only tile {1,2,4,8} x subs {4,8,16} are instantiated; any other pair silently falls through to
// the slower tiled kernel. evidence: docs/perf/syev.md#syev-the-2026-08-07-constant-retune
inline constexpr int32_t SB2ST_BACK_TILE_TINY = 0;
inline constexpr int32_t SB2ST_BACK_TILE_SMALL = 0;
inline constexpr int32_t SB2ST_BACK_TILE_MEDIUM = 0;
inline constexpr int32_t SB2ST_BACK_TILE_LARGE = 0;
inline constexpr int32_t SB2ST_BACK_TILE_XLARGE = 0;

inline constexpr int32_t SB2ST_BACK_SUBS_TINY = 0;
inline constexpr int32_t SB2ST_BACK_SUBS_SMALL = 0;
inline constexpr int32_t SB2ST_BACK_SUBS_MEDIUM = 0;
inline constexpr int32_t SB2ST_BACK_SUBS_LARGE = 0;
inline constexpr int32_t SB2ST_BACK_SUBS_XLARGE = 0;

// sy2sb panel back-transform WY width. 0 = keep the shape gate in sytrd_sy2sb.cc; positive is
// clamped to kd. This SHADOWS ORMQR_BLOCK_SIZE_* on syev's hot path.
// evidence: docs/perf/tuning.md#tuning-knobs-that-shadow-each-other
inline constexpr int32_t SY2SB_ORMQR_NB_TINY = 0;
inline constexpr int32_t SY2SB_ORMQR_NB_SMALL = 0;
inline constexpr int32_t SY2SB_ORMQR_NB_MEDIUM = 0;
inline constexpr int32_t SY2SB_ORMQR_NB_LARGE = 32;
inline constexpr int32_t SY2SB_ORMQR_NB_XLARGE = 0;

inline constexpr int32_t SYTRD_BLOCK_SIZE_TINY = 8;
inline constexpr int32_t SYTRD_BLOCK_SIZE_SMALL = 8;
inline constexpr int32_t SYTRD_BLOCK_SIZE_MEDIUM = 16;
inline constexpr int32_t SYTRD_BLOCK_SIZE_LARGE = 8;
inline constexpr int32_t SYTRD_BLOCK_SIZE_XLARGE = 48;

inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_TINY = 0;
inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_SMALL = 0;
inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_MEDIUM = 0;
inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_LARGE = 128;
inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_XLARGE = 0;

inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_TINY = 0;
inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_SMALL = 0;
inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_MEDIUM = 0;
inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_LARGE = 0;
inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_XLARGE = 0;

inline constexpr int32_t STEDC_RECURSION_THRESHOLD_TINY = 32;
inline constexpr int32_t STEDC_RECURSION_THRESHOLD_SMALL = 32;
inline constexpr int32_t STEDC_RECURSION_THRESHOLD_MEDIUM = 32;
inline constexpr int32_t STEDC_RECURSION_THRESHOLD_LARGE = 32;
inline constexpr int32_t STEDC_RECURSION_THRESHOLD_XLARGE = 32;

// StedcMergeVariant: 1 = Fused (one work-group per merge), 2 = FusedCta (a sub-group partition
// per secular root). Trap: FusedCtaConditionedHeavyDeflation asserts only finite-and-sorted, so
// its passing is weak evidence for variant 2. evidence: docs/perf/stedc.md#stedc-current-tuning-values
inline constexpr int32_t STEDC_MERGE_VARIANT_TINY = 2;
inline constexpr int32_t STEDC_MERGE_VARIANT_SMALL = 2;
inline constexpr int32_t STEDC_MERGE_VARIANT_MEDIUM = 2;
inline constexpr int32_t STEDC_MERGE_VARIANT_LARGE = 2;
inline constexpr int32_t STEDC_MERGE_VARIANT_XLARGE = 2;

// threads_per_root and wg_multiplier are tuned through syev, NOT the stedc bench (which
// disagrees with its consumer). evidence: docs/perf/stedc.md#stedc-current-tuning-values
inline constexpr int32_t STEDC_THREADS_PER_ROOT_TINY = 8;
inline constexpr int32_t STEDC_THREADS_PER_ROOT_SMALL = 8;
inline constexpr int32_t STEDC_THREADS_PER_ROOT_MEDIUM = 8;
inline constexpr int32_t STEDC_THREADS_PER_ROOT_LARGE = 8;
inline constexpr int32_t STEDC_THREADS_PER_ROOT_XLARGE = 8;

inline constexpr int32_t STEDC_WG_MULTIPLIER_TINY = 8;
inline constexpr int32_t STEDC_WG_MULTIPLIER_SMALL = 8;
inline constexpr int32_t STEDC_WG_MULTIPLIER_MEDIUM = 8;
inline constexpr int32_t STEDC_WG_MULTIPLIER_LARGE = 8;
inline constexpr int32_t STEDC_WG_MULTIPLIER_XLARGE = 8;

// Compile-time tables (constexpr): the bucket constant for n, no override.

/// @brief Compiled `ORMQR_BLOCK_SIZE_*` bucket for n (no override).
/// @ingroup api_config_lowlevel
inline constexpr int32_t ormqr_block_size_default_for_n(int32_t n) {
    if (n <= 64) return ORMQR_BLOCK_SIZE_TINY;
    if (n <= 128) return ORMQR_BLOCK_SIZE_SMALL;
    if (n <= 256) return ORMQR_BLOCK_SIZE_MEDIUM;
    if (n <= 512) return ORMQR_BLOCK_SIZE_LARGE;
    return ORMQR_BLOCK_SIZE_XLARGE;
}

inline constexpr int32_t gebrd_block_size_default_for_n(int32_t n) {
    if (n <= 64) return GEBRD_BLOCK_SIZE_TINY;
    if (n <= 128) return GEBRD_BLOCK_SIZE_SMALL;
    if (n <= 256) return GEBRD_BLOCK_SIZE_MEDIUM;
    if (n <= 512) return GEBRD_BLOCK_SIZE_LARGE;
    return GEBRD_BLOCK_SIZE_XLARGE;
}

inline constexpr int32_t sb2st_back_tile_default_for_n(int32_t n) {
    if (n <= 64) return SB2ST_BACK_TILE_TINY;
    if (n <= 128) return SB2ST_BACK_TILE_SMALL;
    if (n <= 256) return SB2ST_BACK_TILE_MEDIUM;
    if (n <= 512) return SB2ST_BACK_TILE_LARGE;
    return SB2ST_BACK_TILE_XLARGE;
}

inline constexpr int32_t sb2st_back_subs_default_for_n(int32_t n) {
    if (n <= 64) return SB2ST_BACK_SUBS_TINY;
    if (n <= 128) return SB2ST_BACK_SUBS_SMALL;
    if (n <= 256) return SB2ST_BACK_SUBS_MEDIUM;
    if (n <= 512) return SB2ST_BACK_SUBS_LARGE;
    return SB2ST_BACK_SUBS_XLARGE;
}

inline constexpr int32_t sy2sb_ormqr_nb_default_for_n(int32_t n) {
    if (n <= 64) return SY2SB_ORMQR_NB_TINY;
    if (n <= 128) return SY2SB_ORMQR_NB_SMALL;
    if (n <= 256) return SY2SB_ORMQR_NB_MEDIUM;
    if (n <= 512) return SY2SB_ORMQR_NB_LARGE;
    return SY2SB_ORMQR_NB_XLARGE;
}

inline constexpr int32_t sytrd_block_size_default_for_n(int32_t n) {
    if (n <= 64) return SYTRD_BLOCK_SIZE_TINY;
    if (n <= 128) return SYTRD_BLOCK_SIZE_SMALL;
    if (n <= 256) return SYTRD_BLOCK_SIZE_MEDIUM;
    if (n <= 512) return SYTRD_BLOCK_SIZE_LARGE;
    return SYTRD_BLOCK_SIZE_XLARGE;
}

inline constexpr int32_t latrd_lower_panel_wg_hint_default_for_n(int32_t n) {
    if (n <= 64) return LATRD_LOWER_PANEL_WG_HINT_TINY;
    if (n <= 128) return LATRD_LOWER_PANEL_WG_HINT_SMALL;
    if (n <= 256) return LATRD_LOWER_PANEL_WG_HINT_MEDIUM;
    if (n <= 512) return LATRD_LOWER_PANEL_WG_HINT_LARGE;
    return LATRD_LOWER_PANEL_WG_HINT_XLARGE;
}

inline constexpr int32_t stedc_recursion_threshold_default_for_n(int32_t n) {
    if (n <= 64) return STEDC_RECURSION_THRESHOLD_TINY;
    if (n <= 128) return STEDC_RECURSION_THRESHOLD_SMALL;
    if (n <= 256) return STEDC_RECURSION_THRESHOLD_MEDIUM;
    if (n <= 512) return STEDC_RECURSION_THRESHOLD_LARGE;
    return STEDC_RECURSION_THRESHOLD_XLARGE;
}

inline constexpr int32_t stedc_merge_variant_default_for_n(int32_t n) {
    if (n <= 64) return STEDC_MERGE_VARIANT_TINY;
    if (n <= 128) return STEDC_MERGE_VARIANT_SMALL;
    if (n <= 256) return STEDC_MERGE_VARIANT_MEDIUM;
    if (n <= 512) return STEDC_MERGE_VARIANT_LARGE;
    return STEDC_MERGE_VARIANT_XLARGE;
}

inline constexpr int32_t stedc_threads_per_root_default_for_n(int32_t n) {
    if (n <= 64) return STEDC_THREADS_PER_ROOT_TINY;
    if (n <= 128) return STEDC_THREADS_PER_ROOT_SMALL;
    if (n <= 256) return STEDC_THREADS_PER_ROOT_MEDIUM;
    if (n <= 512) return STEDC_THREADS_PER_ROOT_LARGE;
    return STEDC_THREADS_PER_ROOT_XLARGE;
}

inline constexpr int32_t stedc_wg_multiplier_default_for_n(int32_t n) {
    if (n <= 64) return STEDC_WG_MULTIPLIER_TINY;
    if (n <= 128) return STEDC_WG_MULTIPLIER_SMALL;
    if (n <= 256) return STEDC_WG_MULTIPLIER_MEDIUM;
    if (n <= 512) return STEDC_WG_MULTIPLIER_LARGE;
    return STEDC_WG_MULTIPLIER_XLARGE;
}

// Runtime accessors: what production code calls. Each consults its BATCHLAS_TUNE_* override
// through settings() first and falls back to the table above.

/// @brief ormqr WY block width for order n; override `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE`.
/// @ingroup api_config_lowlevel
inline int32_t ormqr_block_size_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.ormqr_block_size,  // BATCHLAS_TUNE_ORMQR_BLOCK_SIZE
                                       ormqr_block_size_default_for_n(n));
}

/// @brief gebrd panel width for order n; override `BATCHLAS_TUNE_GEBRD_BLOCK_SIZE`.
/// @ingroup api_config_lowlevel
inline int32_t gebrd_block_size_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.gebrd_block_size,  // BATCHLAS_TUNE_GEBRD_BLOCK_SIZE
                                       gebrd_block_size_default_for_n(n));
}

/// @brief sb2st wave back-transform tile for order n, or 0 for "keep the call site's heuristic";
/// override `BATCHLAS_TUNE_SB2ST_BACK_TILE`.
/// @note 0 is reachable only from the compiled constant: the override forces a geometry, never
///       auto. The same holds for sb2st_back_subs_for_n() and sy2sb_ormqr_nb_for_n().
/// @ingroup api_config_lowlevel
inline int32_t sb2st_back_tile_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.sb2st_back_tile,  // BATCHLAS_TUNE_SB2ST_BACK_TILE
                                       sb2st_back_tile_default_for_n(n));
}

/// @brief sb2st wave back-transform sub-group count for order n, or 0 for the heuristic;
/// override `BATCHLAS_TUNE_SB2ST_BACK_SUBS`.
/// @ingroup api_config_lowlevel
inline int32_t sb2st_back_subs_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.sb2st_back_subs,  // BATCHLAS_TUNE_SB2ST_BACK_SUBS
                                       sb2st_back_subs_default_for_n(n));
}

/// @brief sy2sb back-transform WY width for order n, or 0 for the shape gate; override
/// `BATCHLAS_TUNE_SY2SB_ORMQR_NB`.
/// @ingroup api_config_lowlevel
inline int32_t sy2sb_ormqr_nb_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.sy2sb_ormqr_nb,  // BATCHLAS_TUNE_SY2SB_ORMQR_NB
                                       sy2sb_ormqr_nb_default_for_n(n));
}

/// @brief sytrd panel width for order n; override `BATCHLAS_TUNE_SYTRD_BLOCK_SIZE`.
/// @ingroup api_config_lowlevel
inline int32_t sytrd_block_size_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.sytrd_block_size,  // BATCHLAS_TUNE_SYTRD_BLOCK_SIZE
                                       sytrd_block_size_default_for_n(n));
}

/// @brief latrd lower-panel work-group size hint for order n, 0 for none; override
/// `BATCHLAS_TUNE_LATRD_WG_HINT`.
/// @ingroup api_config_lowlevel
inline int32_t latrd_lower_panel_wg_hint_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.latrd_wg_hint,  // BATCHLAS_TUNE_LATRD_WG_HINT
                                       latrd_lower_panel_wg_hint_default_for_n(n));
}

/// @brief latrd_lower_panel_wg_hint_for_n() at n = 256, for the legacy path.
/// @ingroup api_config_lowlevel
inline int32_t latrd_lower_panel_wg_hint() { return latrd_lower_panel_wg_hint_for_n(256); }

/// @brief Whether sytrd fuses the panel update at order n. No `BATCHLAS_TUNE_*` override; see
/// SelectionSettings::sytrd_fuse_panel_update.
/// @ingroup api_config_lowlevel
inline constexpr bool sytrd_fuse_panel_update_for_n(int32_t n) {
    if (n <= 64) return SYTRD_FUSE_PANEL_UPDATE_TINY != 0;
    if (n <= 128) return SYTRD_FUSE_PANEL_UPDATE_SMALL != 0;
    if (n <= 256) return SYTRD_FUSE_PANEL_UPDATE_MEDIUM != 0;
    if (n <= 512) return SYTRD_FUSE_PANEL_UPDATE_LARGE != 0;
    return SYTRD_FUSE_PANEL_UPDATE_XLARGE != 0;
}

/// @brief stedc leaf size for order n; override `BATCHLAS_TUNE_STEDC_RECURSION_THRESHOLD`.
/// @note 32 is the CTA invariant (sub-group width), not a tuning result.
/// @ingroup api_config_lowlevel
inline int32_t stedc_recursion_threshold_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.stedc_recursion_threshold,  // BATCHLAS_TUNE_STEDC_RECURSION_THRESHOLD
                                       stedc_recursion_threshold_default_for_n(n));
}

/// @brief stedc merge variant for order n, as a StedcMergeVariant value (1 Fused, 2 FusedCta);
/// override `BATCHLAS_TUNE_STEDC_MERGE_VARIANT`.
/// @ingroup api_config_lowlevel
// Callers cast to StedcMergeVariant unchecked; 0 (Auto) would re-enter tuning resolution, so it
// must stay unreachable (it is: non-positive overrides fall back).
inline int32_t stedc_merge_variant_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.stedc_merge_variant,  // BATCHLAS_TUNE_STEDC_MERGE_VARIANT
                                       stedc_merge_variant_default_for_n(n));
}

/// @brief stedc secular-solve threads per root for order n; override
/// `BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT`.
/// @ingroup api_config_lowlevel
inline int32_t stedc_threads_per_root_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.stedc_threads_per_root,  // BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT
                                       stedc_threads_per_root_default_for_n(n));
}

/// @brief stedc merge work-group multiplier for order n; override
/// `BATCHLAS_TUNE_STEDC_WG_MULTIPLIER`.
/// @ingroup api_config_lowlevel
inline int32_t stedc_wg_multiplier_for_n(int32_t n) {
    return detail::tuning_env_override(settings().geometry.tune.stedc_wg_multiplier,  // BATCHLAS_TUNE_STEDC_WG_MULTIPLIER
                                       stedc_wg_multiplier_default_for_n(n));
}

} // namespace batchlas::tuning
