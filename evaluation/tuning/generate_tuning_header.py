#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _find_best_params(profile: Dict[str, Any], bench: str) -> Dict[str, int]:
    results = profile.get("results")
    if not isinstance(results, list):
        return {}

    for entry in results:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("bench")) != bench:
            continue

        best = entry.get("best")
        if not isinstance(best, dict):
            return {}
        params = best.get("params")
        if not isinstance(params, dict):
            return {}

        parsed: Dict[str, int] = {}
        for key, value in params.items():
            try:
                parsed[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return parsed

    return {}


def _find_bench_entry(profile: Dict[str, Any], bench: str) -> Optional[Dict[str, Any]]:
    results = profile.get("results")
    if not isinstance(results, list):
        return None
    for entry in results:
        if isinstance(entry, dict) and str(entry.get("bench")) == bench:
            return entry
    return None


def _bucket_name_from_n(n: int) -> str:
    if n <= 64:
        return "tiny"
    if n <= 128:
        return "small"
    if n <= 256:
        return "medium"
    if n <= 512:
        return "large"
    return "xlarge"


def _derive_param_buckets(entry: Optional[Dict[str, Any]],
                          param_name: str,
                          fallback_tiny: int,
                          fallback_small: int,
                          fallback_medium: int,
                          fallback_large: int,
                          fallback_xlarge: int,
                          direction: str) -> Dict[str, int]:
    buckets = {
        "tiny": int(fallback_tiny),
        "small": int(fallback_small),
        "medium": int(fallback_medium),
        "large": int(fallback_large),
        "xlarge": int(fallback_xlarge),
    }

    if not isinstance(entry, dict):
        return buckets

    per_case = entry.get("per_case_best")
    if not isinstance(per_case, list) or not per_case:
        best_params = entry.get("best", {}).get("params", {})
        if isinstance(best_params, dict) and param_name in best_params:
            v = int(best_params[param_name])
            buckets = {
                "tiny": v,
                "small": v,
                "medium": v,
                "large": v,
                "xlarge": v,
            }
        return buckets

    score_best: Dict[str, float] = {}
    for case_entry in per_case:
        if not isinstance(case_entry, dict):
            continue
        fixed = case_entry.get("fixed")
        params = case_entry.get("params")
        if not isinstance(fixed, dict) or not isinstance(params, dict):
            continue
        if "n" not in fixed or param_name not in params:
            continue
        try:
            n = int(fixed["n"])
            pval = int(params[param_name])
            value = float(case_entry.get("value"))
        except (TypeError, ValueError):
            continue

        bucket = _bucket_name_from_n(n)
        prev_score = score_best.get(bucket)
        is_better = False
        if prev_score is None:
            is_better = True
        elif direction == "min":
            is_better = value < prev_score
        else:
            is_better = value > prev_score

        if is_better:
            score_best[bucket] = value
            buckets[bucket] = pval

    return buckets


def _emit_header(out_path: Path,
                 source_profile: Path,
                 ormqr_tiny: int,
                 ormqr_small: int,
                 ormqr_medium: int,
                 ormqr_large: int,
                 ormqr_xlarge: int,
                 gebrd: Dict[str, int],
                 sb2st_tile: Dict[str, int],
                 sb2st_subs: Dict[str, int],
                 sy2sb_nb: Dict[str, int],
                 sytrd_tiny: int,
                 sytrd_small: int,
                 sytrd_medium: int,
                 sytrd_large: int,
                 sytrd_xlarge: int,
                 latrd_wg: Dict[str, int],
                 sytrd_fuse: Dict[str, int],
                 stedc_rt: Dict[str, int],
                 stedc_mv: Dict[str, int],
                 stedc_tpr: Dict[str, int],
                 stedc_wgm: Dict[str, int]) -> None:
    header = f"""#pragma once

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
/// @ingroup config
// GENERATED constants (evaluation/tuning/generate_tuning_header.py); regenerate, do not hand-edit.
// The accessors, this include and these doc comments are mirrored in the generator's template:
// change both or a retune reverts it. evidence: docs/perf/tuning.md#tuning-regenerating-the-header
// Source profile: {source_profile}

#include <cstdint>
#include <cstdlib>

#include <batchlas/settings.hh>

namespace batchlas::tuning {{

namespace detail {{

/// @brief The captured override if it is a clean positive 32-bit integer, else `fallback`.
///
/// Unset, empty, unparseable, trailing garbage (`"16x"`), non-positive or out-of-range values all
/// yield `fallback`, the compiled constant.
/// @param captured the captured `BATCHLAS_TUNE_*` value from settings()
/// @param fallback the compiled constant for this n
/// @ingroup config
// strtol stays HERE, not in settings.cc: env_int_or reads "16x" as 16, exactly what a retune
// harness types. evidence: docs/perf/tuning.md#tuning-runtime-overrides-of-the-constants
inline int32_t tuning_env_override(const EnvValue& captured, int32_t fallback) {{
    const char* v = captured.get();
    if (v == nullptr || *v == '\\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\\0') return fallback;   // not a clean integer
    if (parsed <= 0 || parsed > 2147483647L) return fallback;  // non-positive / out of range
    return static_cast<int32_t>(parsed);
}}

}} // namespace detail

inline constexpr int32_t ORMQR_BLOCK_SIZE_TINY = {ormqr_tiny};
inline constexpr int32_t ORMQR_BLOCK_SIZE_SMALL = {ormqr_small};
inline constexpr int32_t ORMQR_BLOCK_SIZE_MEDIUM = {ormqr_medium};
inline constexpr int32_t ORMQR_BLOCK_SIZE_LARGE = {ormqr_large};
inline constexpr int32_t ORMQR_BLOCK_SIZE_XLARGE = {ormqr_xlarge};

// gebrd's panel width: deliberately NOT the ormqr constant (opposite gradients).
// evidence: docs/perf/gesvd.md#gesvd-the-gebrd-panel-width-split-from-ormqr
inline constexpr int32_t GEBRD_BLOCK_SIZE_TINY = {gebrd["tiny"]};
inline constexpr int32_t GEBRD_BLOCK_SIZE_SMALL = {gebrd["small"]};
inline constexpr int32_t GEBRD_BLOCK_SIZE_MEDIUM = {gebrd["medium"]};
inline constexpr int32_t GEBRD_BLOCK_SIZE_LARGE = {gebrd["large"]};
inline constexpr int32_t GEBRD_BLOCK_SIZE_XLARGE = {gebrd["xlarge"]};

// sb2st wave back-transform geometry. 0 = keep the shape-adaptive heuristic in
// sytrd_sb2st_hh.cc, and ALL ship 0 on purpose: the heuristic already matches the tuned winner.
// Only tile {{1,2,4,8}} x subs {{4,8,16}} are instantiated; any other pair silently falls through to
// the slower tiled kernel. evidence: docs/perf/syev.md#syev-the-2026-08-07-constant-retune
inline constexpr int32_t SB2ST_BACK_TILE_TINY = {sb2st_tile["tiny"]};
inline constexpr int32_t SB2ST_BACK_TILE_SMALL = {sb2st_tile["small"]};
inline constexpr int32_t SB2ST_BACK_TILE_MEDIUM = {sb2st_tile["medium"]};
inline constexpr int32_t SB2ST_BACK_TILE_LARGE = {sb2st_tile["large"]};
inline constexpr int32_t SB2ST_BACK_TILE_XLARGE = {sb2st_tile["xlarge"]};

inline constexpr int32_t SB2ST_BACK_SUBS_TINY = {sb2st_subs["tiny"]};
inline constexpr int32_t SB2ST_BACK_SUBS_SMALL = {sb2st_subs["small"]};
inline constexpr int32_t SB2ST_BACK_SUBS_MEDIUM = {sb2st_subs["medium"]};
inline constexpr int32_t SB2ST_BACK_SUBS_LARGE = {sb2st_subs["large"]};
inline constexpr int32_t SB2ST_BACK_SUBS_XLARGE = {sb2st_subs["xlarge"]};

// sy2sb panel back-transform WY width. 0 = keep the shape gate in sytrd_sy2sb.cc; positive is
// clamped to kd. This SHADOWS ORMQR_BLOCK_SIZE_* on syev's hot path.
// evidence: docs/perf/tuning.md#tuning-knobs-that-shadow-each-other
inline constexpr int32_t SY2SB_ORMQR_NB_TINY = {sy2sb_nb["tiny"]};
inline constexpr int32_t SY2SB_ORMQR_NB_SMALL = {sy2sb_nb["small"]};
inline constexpr int32_t SY2SB_ORMQR_NB_MEDIUM = {sy2sb_nb["medium"]};
inline constexpr int32_t SY2SB_ORMQR_NB_LARGE = {sy2sb_nb["large"]};
inline constexpr int32_t SY2SB_ORMQR_NB_XLARGE = {sy2sb_nb["xlarge"]};

inline constexpr int32_t SYTRD_BLOCK_SIZE_TINY = {sytrd_tiny};
inline constexpr int32_t SYTRD_BLOCK_SIZE_SMALL = {sytrd_small};
inline constexpr int32_t SYTRD_BLOCK_SIZE_MEDIUM = {sytrd_medium};
inline constexpr int32_t SYTRD_BLOCK_SIZE_LARGE = {sytrd_large};
inline constexpr int32_t SYTRD_BLOCK_SIZE_XLARGE = {sytrd_xlarge};

inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_TINY = {latrd_wg["tiny"]};
inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_SMALL = {latrd_wg["small"]};
inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_MEDIUM = {latrd_wg["medium"]};
inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_LARGE = {latrd_wg["large"]};
inline constexpr int32_t LATRD_LOWER_PANEL_WG_HINT_XLARGE = {latrd_wg["xlarge"]};

inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_TINY = {sytrd_fuse["tiny"]};
inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_SMALL = {sytrd_fuse["small"]};
inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_MEDIUM = {sytrd_fuse["medium"]};
inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_LARGE = {sytrd_fuse["large"]};
inline constexpr int32_t SYTRD_FUSE_PANEL_UPDATE_XLARGE = {sytrd_fuse["xlarge"]};

inline constexpr int32_t STEDC_RECURSION_THRESHOLD_TINY = {stedc_rt["tiny"]};
inline constexpr int32_t STEDC_RECURSION_THRESHOLD_SMALL = {stedc_rt["small"]};
inline constexpr int32_t STEDC_RECURSION_THRESHOLD_MEDIUM = {stedc_rt["medium"]};
inline constexpr int32_t STEDC_RECURSION_THRESHOLD_LARGE = {stedc_rt["large"]};
inline constexpr int32_t STEDC_RECURSION_THRESHOLD_XLARGE = {stedc_rt["xlarge"]};

// StedcMergeVariant: 1 = Fused (one work-group per merge), 2 = FusedCta (a sub-group partition
// per secular root). Trap: FusedCtaConditionedHeavyDeflation asserts only finite-and-sorted, so
// its passing is weak evidence for variant 2. evidence: docs/perf/stedc.md#stedc-current-tuning-values
inline constexpr int32_t STEDC_MERGE_VARIANT_TINY = {stedc_mv["tiny"]};
inline constexpr int32_t STEDC_MERGE_VARIANT_SMALL = {stedc_mv["small"]};
inline constexpr int32_t STEDC_MERGE_VARIANT_MEDIUM = {stedc_mv["medium"]};
inline constexpr int32_t STEDC_MERGE_VARIANT_LARGE = {stedc_mv["large"]};
inline constexpr int32_t STEDC_MERGE_VARIANT_XLARGE = {stedc_mv["xlarge"]};

// threads_per_root and wg_multiplier are tuned through syev, NOT the stedc bench (which
// disagrees with its consumer). evidence: docs/perf/stedc.md#stedc-current-tuning-values
inline constexpr int32_t STEDC_THREADS_PER_ROOT_TINY = {stedc_tpr["tiny"]};
inline constexpr int32_t STEDC_THREADS_PER_ROOT_SMALL = {stedc_tpr["small"]};
inline constexpr int32_t STEDC_THREADS_PER_ROOT_MEDIUM = {stedc_tpr["medium"]};
inline constexpr int32_t STEDC_THREADS_PER_ROOT_LARGE = {stedc_tpr["large"]};
inline constexpr int32_t STEDC_THREADS_PER_ROOT_XLARGE = {stedc_tpr["xlarge"]};

inline constexpr int32_t STEDC_WG_MULTIPLIER_TINY = {stedc_wgm["tiny"]};
inline constexpr int32_t STEDC_WG_MULTIPLIER_SMALL = {stedc_wgm["small"]};
inline constexpr int32_t STEDC_WG_MULTIPLIER_MEDIUM = {stedc_wgm["medium"]};
inline constexpr int32_t STEDC_WG_MULTIPLIER_LARGE = {stedc_wgm["large"]};
inline constexpr int32_t STEDC_WG_MULTIPLIER_XLARGE = {stedc_wgm["xlarge"]};

// Compile-time tables (constexpr): the bucket constant for n, no override.

/// @brief Compiled `ORMQR_BLOCK_SIZE_*` bucket for n (no override).
/// @ingroup config
inline constexpr int32_t ormqr_block_size_default_for_n(int32_t n) {{
    if (n <= 64) return ORMQR_BLOCK_SIZE_TINY;
    if (n <= 128) return ORMQR_BLOCK_SIZE_SMALL;
    if (n <= 256) return ORMQR_BLOCK_SIZE_MEDIUM;
    if (n <= 512) return ORMQR_BLOCK_SIZE_LARGE;
    return ORMQR_BLOCK_SIZE_XLARGE;
}}

inline constexpr int32_t gebrd_block_size_default_for_n(int32_t n) {{
    if (n <= 64) return GEBRD_BLOCK_SIZE_TINY;
    if (n <= 128) return GEBRD_BLOCK_SIZE_SMALL;
    if (n <= 256) return GEBRD_BLOCK_SIZE_MEDIUM;
    if (n <= 512) return GEBRD_BLOCK_SIZE_LARGE;
    return GEBRD_BLOCK_SIZE_XLARGE;
}}

inline constexpr int32_t sb2st_back_tile_default_for_n(int32_t n) {{
    if (n <= 64) return SB2ST_BACK_TILE_TINY;
    if (n <= 128) return SB2ST_BACK_TILE_SMALL;
    if (n <= 256) return SB2ST_BACK_TILE_MEDIUM;
    if (n <= 512) return SB2ST_BACK_TILE_LARGE;
    return SB2ST_BACK_TILE_XLARGE;
}}

inline constexpr int32_t sb2st_back_subs_default_for_n(int32_t n) {{
    if (n <= 64) return SB2ST_BACK_SUBS_TINY;
    if (n <= 128) return SB2ST_BACK_SUBS_SMALL;
    if (n <= 256) return SB2ST_BACK_SUBS_MEDIUM;
    if (n <= 512) return SB2ST_BACK_SUBS_LARGE;
    return SB2ST_BACK_SUBS_XLARGE;
}}

inline constexpr int32_t sy2sb_ormqr_nb_default_for_n(int32_t n) {{
    if (n <= 64) return SY2SB_ORMQR_NB_TINY;
    if (n <= 128) return SY2SB_ORMQR_NB_SMALL;
    if (n <= 256) return SY2SB_ORMQR_NB_MEDIUM;
    if (n <= 512) return SY2SB_ORMQR_NB_LARGE;
    return SY2SB_ORMQR_NB_XLARGE;
}}

inline constexpr int32_t sytrd_block_size_default_for_n(int32_t n) {{
    if (n <= 64) return SYTRD_BLOCK_SIZE_TINY;
    if (n <= 128) return SYTRD_BLOCK_SIZE_SMALL;
    if (n <= 256) return SYTRD_BLOCK_SIZE_MEDIUM;
    if (n <= 512) return SYTRD_BLOCK_SIZE_LARGE;
    return SYTRD_BLOCK_SIZE_XLARGE;
}}

inline constexpr int32_t latrd_lower_panel_wg_hint_default_for_n(int32_t n) {{
    if (n <= 64) return LATRD_LOWER_PANEL_WG_HINT_TINY;
    if (n <= 128) return LATRD_LOWER_PANEL_WG_HINT_SMALL;
    if (n <= 256) return LATRD_LOWER_PANEL_WG_HINT_MEDIUM;
    if (n <= 512) return LATRD_LOWER_PANEL_WG_HINT_LARGE;
    return LATRD_LOWER_PANEL_WG_HINT_XLARGE;
}}

inline constexpr int32_t stedc_recursion_threshold_default_for_n(int32_t n) {{
    if (n <= 64) return STEDC_RECURSION_THRESHOLD_TINY;
    if (n <= 128) return STEDC_RECURSION_THRESHOLD_SMALL;
    if (n <= 256) return STEDC_RECURSION_THRESHOLD_MEDIUM;
    if (n <= 512) return STEDC_RECURSION_THRESHOLD_LARGE;
    return STEDC_RECURSION_THRESHOLD_XLARGE;
}}

inline constexpr int32_t stedc_merge_variant_default_for_n(int32_t n) {{
    if (n <= 64) return STEDC_MERGE_VARIANT_TINY;
    if (n <= 128) return STEDC_MERGE_VARIANT_SMALL;
    if (n <= 256) return STEDC_MERGE_VARIANT_MEDIUM;
    if (n <= 512) return STEDC_MERGE_VARIANT_LARGE;
    return STEDC_MERGE_VARIANT_XLARGE;
}}

inline constexpr int32_t stedc_threads_per_root_default_for_n(int32_t n) {{
    if (n <= 64) return STEDC_THREADS_PER_ROOT_TINY;
    if (n <= 128) return STEDC_THREADS_PER_ROOT_SMALL;
    if (n <= 256) return STEDC_THREADS_PER_ROOT_MEDIUM;
    if (n <= 512) return STEDC_THREADS_PER_ROOT_LARGE;
    return STEDC_THREADS_PER_ROOT_XLARGE;
}}

inline constexpr int32_t stedc_wg_multiplier_default_for_n(int32_t n) {{
    if (n <= 64) return STEDC_WG_MULTIPLIER_TINY;
    if (n <= 128) return STEDC_WG_MULTIPLIER_SMALL;
    if (n <= 256) return STEDC_WG_MULTIPLIER_MEDIUM;
    if (n <= 512) return STEDC_WG_MULTIPLIER_LARGE;
    return STEDC_WG_MULTIPLIER_XLARGE;
}}

// Runtime accessors: what production code calls. Each consults its BATCHLAS_TUNE_* override
// through settings() first and falls back to the table above.

/// @brief ormqr WY block width for order n; override `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE`.
/// @ingroup config
inline int32_t ormqr_block_size_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.ormqr_block_size,  // BATCHLAS_TUNE_ORMQR_BLOCK_SIZE
                                       ormqr_block_size_default_for_n(n));
}}

/// @brief gebrd panel width for order n; override `BATCHLAS_TUNE_GEBRD_BLOCK_SIZE`.
/// @ingroup config
inline int32_t gebrd_block_size_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.gebrd_block_size,  // BATCHLAS_TUNE_GEBRD_BLOCK_SIZE
                                       gebrd_block_size_default_for_n(n));
}}

/// @brief sb2st wave back-transform tile for order n, or 0 for "keep the call site's heuristic";
/// override `BATCHLAS_TUNE_SB2ST_BACK_TILE`.
/// @note 0 is reachable only from the compiled constant: the override forces a geometry, never
///       auto. The same holds for sb2st_back_subs_for_n() and sy2sb_ormqr_nb_for_n().
/// @ingroup config
inline int32_t sb2st_back_tile_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.sb2st_back_tile,  // BATCHLAS_TUNE_SB2ST_BACK_TILE
                                       sb2st_back_tile_default_for_n(n));
}}

/// @brief sb2st wave back-transform sub-group count for order n, or 0 for the heuristic;
/// override `BATCHLAS_TUNE_SB2ST_BACK_SUBS`.
/// @ingroup config
inline int32_t sb2st_back_subs_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.sb2st_back_subs,  // BATCHLAS_TUNE_SB2ST_BACK_SUBS
                                       sb2st_back_subs_default_for_n(n));
}}

/// @brief sy2sb back-transform WY width for order n, or 0 for the shape gate; override
/// `BATCHLAS_TUNE_SY2SB_ORMQR_NB`.
/// @ingroup config
inline int32_t sy2sb_ormqr_nb_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.sy2sb_ormqr_nb,  // BATCHLAS_TUNE_SY2SB_ORMQR_NB
                                       sy2sb_ormqr_nb_default_for_n(n));
}}

/// @brief sytrd panel width for order n; override `BATCHLAS_TUNE_SYTRD_BLOCK_SIZE`.
/// @ingroup config
inline int32_t sytrd_block_size_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.sytrd_block_size,  // BATCHLAS_TUNE_SYTRD_BLOCK_SIZE
                                       sytrd_block_size_default_for_n(n));
}}

/// @brief latrd lower-panel work-group size hint for order n, 0 for none; override
/// `BATCHLAS_TUNE_LATRD_WG_HINT`.
/// @ingroup config
inline int32_t latrd_lower_panel_wg_hint_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.latrd_wg_hint,  // BATCHLAS_TUNE_LATRD_WG_HINT
                                       latrd_lower_panel_wg_hint_default_for_n(n));
}}

/// @brief latrd_lower_panel_wg_hint_for_n() at n = 256, for the legacy path.
/// @ingroup config
inline int32_t latrd_lower_panel_wg_hint() {{ return latrd_lower_panel_wg_hint_for_n(256); }}

/// @brief Whether sytrd fuses the panel update at order n. No `BATCHLAS_TUNE_*` override; see
/// SelectionSettings::sytrd_fuse_panel_update.
/// @ingroup config
inline constexpr bool sytrd_fuse_panel_update_for_n(int32_t n) {{
    if (n <= 64) return SYTRD_FUSE_PANEL_UPDATE_TINY != 0;
    if (n <= 128) return SYTRD_FUSE_PANEL_UPDATE_SMALL != 0;
    if (n <= 256) return SYTRD_FUSE_PANEL_UPDATE_MEDIUM != 0;
    if (n <= 512) return SYTRD_FUSE_PANEL_UPDATE_LARGE != 0;
    return SYTRD_FUSE_PANEL_UPDATE_XLARGE != 0;
}}

/// @brief stedc leaf size for order n; override `BATCHLAS_TUNE_STEDC_RECURSION_THRESHOLD`.
/// @note 32 is the CTA invariant (sub-group width), not a tuning result.
/// @ingroup config
inline int32_t stedc_recursion_threshold_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.stedc_recursion_threshold,  // BATCHLAS_TUNE_STEDC_RECURSION_THRESHOLD
                                       stedc_recursion_threshold_default_for_n(n));
}}

/// @brief stedc merge variant for order n, as a StedcMergeVariant value (1 Fused, 2 FusedCta);
/// override `BATCHLAS_TUNE_STEDC_MERGE_VARIANT`.
/// @ingroup config
// Callers cast to StedcMergeVariant unchecked; 0 (Auto) would re-enter tuning resolution, so it
// must stay unreachable (it is: non-positive overrides fall back).
inline int32_t stedc_merge_variant_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.stedc_merge_variant,  // BATCHLAS_TUNE_STEDC_MERGE_VARIANT
                                       stedc_merge_variant_default_for_n(n));
}}

/// @brief stedc secular-solve threads per root for order n; override
/// `BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT`.
/// @ingroup config
inline int32_t stedc_threads_per_root_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.stedc_threads_per_root,  // BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT
                                       stedc_threads_per_root_default_for_n(n));
}}

/// @brief stedc merge work-group multiplier for order n; override
/// `BATCHLAS_TUNE_STEDC_WG_MULTIPLIER`.
/// @ingroup config
inline int32_t stedc_wg_multiplier_for_n(int32_t n) {{
    return detail::tuning_env_override(settings().geometry.tune.stedc_wg_multiplier,  // BATCHLAS_TUNE_STEDC_WG_MULTIPLIER
                                       stedc_wg_multiplier_default_for_n(n));
}}

}} // namespace batchlas::tuning
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BatchLAS tuning constants header from tuning profile JSON")
    parser.add_argument("--profile", type=Path, required=True, help="Path to tuning profile JSON")
    parser.add_argument("--out", type=Path, required=True, help="Output tuning header path")
    parser.add_argument("--fallback-ormqr-block-size-tiny", type=int, default=-1)
    parser.add_argument("--fallback-ormqr-block-size-small", type=int, default=-1)
    parser.add_argument("--fallback-ormqr-block-size-medium", type=int, default=-1)
    parser.add_argument("--fallback-ormqr-block-size-large", type=int, default=-1)
    parser.add_argument("--fallback-ormqr-block-size-xlarge", type=int, default=-1)
    parser.add_argument("--fallback-gebrd-block-size-tiny", type=int, default=-1)
    parser.add_argument("--fallback-gebrd-block-size-small", type=int, default=-1)
    parser.add_argument("--fallback-gebrd-block-size-medium", type=int, default=-1)
    parser.add_argument("--fallback-gebrd-block-size-large", type=int, default=-1)
    parser.add_argument("--fallback-gebrd-block-size-xlarge", type=int, default=-1)
    parser.add_argument("--fallback-sytrd-block-size-tiny", type=int, default=-1)
    parser.add_argument("--fallback-sytrd-block-size-small", type=int, default=-1)
    parser.add_argument("--fallback-sytrd-block-size-medium", type=int, default=-1)
    parser.add_argument("--fallback-sytrd-block-size-large", type=int, default=-1)
    parser.add_argument("--fallback-sytrd-block-size-xlarge", type=int, default=-1)
    parser.add_argument("--fallback-latrd-lower-panel-wg-hint", type=int, default=0)
    parser.add_argument("--fallback-sytrd-fuse-panel-update-tiny", type=int, default=-1)
    parser.add_argument("--fallback-sytrd-fuse-panel-update-small", type=int, default=-1)
    parser.add_argument("--fallback-sytrd-fuse-panel-update-medium", type=int, default=-1)
    parser.add_argument("--fallback-sytrd-fuse-panel-update-large", type=int, default=-1)
    parser.add_argument("--fallback-sytrd-fuse-panel-update-xlarge", type=int, default=-1)
    for param in ["recursion-threshold", "merge-variant", "threads-per-root", "wg-multiplier"]:
        for bucket in ["tiny", "small", "medium", "large", "xlarge"]:
            parser.add_argument(f"--fallback-stedc-{param}-{bucket}", type=int, default=-1)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())

    ormqr_params = _find_best_params(profile, "ormqr_blocked")
    sytrd_params = _find_best_params(profile, "sytrd_blocked")
    latrd_params = _find_best_params(profile, "latrd_lower_panel")
    syev_params = _find_best_params(profile, "syev")

    ormqr_entry = _find_bench_entry(profile, "ormqr_blocked")
    sytrd_entry = _find_bench_entry(profile, "sytrd_blocked")
    syev_entry = _find_bench_entry(profile, "syev")

    ormqr_fallback = int(ormqr_params.get("block_size", 64))
    sytrd_fallback = int(sytrd_params.get("nb", 32))

    ormqr_tiny_fallback = args.fallback_ormqr_block_size_tiny if args.fallback_ormqr_block_size_tiny > 0 else ormqr_fallback
    ormqr_small_fallback = args.fallback_ormqr_block_size_small if args.fallback_ormqr_block_size_small > 0 else ormqr_fallback
    ormqr_medium_fallback = args.fallback_ormqr_block_size_medium if args.fallback_ormqr_block_size_medium > 0 else ormqr_fallback
    ormqr_large_fallback = args.fallback_ormqr_block_size_large if args.fallback_ormqr_block_size_large > 0 else ormqr_fallback
    ormqr_xlarge_fallback = args.fallback_ormqr_block_size_xlarge if args.fallback_ormqr_block_size_xlarge > 0 else ormqr_fallback

    sytrd_tiny_fallback = args.fallback_sytrd_block_size_tiny if args.fallback_sytrd_block_size_tiny > 0 else sytrd_fallback
    sytrd_small_fallback = args.fallback_sytrd_block_size_small if args.fallback_sytrd_block_size_small > 0 else sytrd_fallback
    sytrd_medium_fallback = args.fallback_sytrd_block_size_medium if args.fallback_sytrd_block_size_medium > 0 else sytrd_fallback
    sytrd_large_fallback = args.fallback_sytrd_block_size_large if args.fallback_sytrd_block_size_large > 0 else sytrd_fallback
    sytrd_xlarge_fallback = args.fallback_sytrd_block_size_xlarge if args.fallback_sytrd_block_size_xlarge > 0 else sytrd_fallback

    ormqr_buckets = _derive_param_buckets(ormqr_entry,
                                          "block_size",
                                          ormqr_tiny_fallback,
                                          ormqr_small_fallback,
                                          ormqr_medium_fallback,
                                          ormqr_large_fallback,
                                          ormqr_xlarge_fallback,
                                          direction="min")
    sytrd_buckets = _derive_param_buckets(sytrd_entry,
                                          "nb",
                                          sytrd_tiny_fallback,
                                          sytrd_small_fallback,
                                          sytrd_medium_fallback,
                                          sytrd_large_fallback,
                                          sytrd_xlarge_fallback,
                                          direction="min")

    # Prefer end-to-end syev-coupled tuning when available.
    if isinstance(syev_entry, dict):
        sytrd_buckets = _derive_param_buckets(syev_entry,
                                              "nb",
                                              sytrd_buckets["tiny"],
                                              sytrd_buckets["small"],
                                              sytrd_buckets["medium"],
                                              sytrd_buckets["large"],
                                              sytrd_buckets["xlarge"],
                                              direction="min")

    # gebrd is tuned through gesvd, whose gebrd stage is 79-95% of the call.
    # Falls back to 16 -- the value gesvd shipped while gebrd and ormqr shared a
    # constant -- so a profile without a gesvd bench leaves behaviour unchanged.
    gesvd_entry = _find_bench_entry(profile, "gesvd")
    gesvd_params = _find_best_params(profile, "gesvd")
    gebrd_default = int(gesvd_params.get("gebrd_nb", 16))

    def _gebrd_fallback(bucket: str) -> int:
        v = getattr(args, f"fallback_gebrd_block_size_{bucket}")
        return v if v > 0 else gebrd_default

    gebrd_buckets = _derive_param_buckets(
        gesvd_entry, "gebrd_nb",
        *[_gebrd_fallback(b) for b in ["tiny", "small", "medium", "large", "xlarge"]],
        direction="min")

    # sb2st and sy2sb: 0 means "no opinion, keep the call site's heuristic",
    # which is what ships. A profile without these benches therefore leaves
    # behaviour unchanged rather than inventing a geometry.
    sb2st_entry = _find_bench_entry(profile, "sb2st")
    sy2sb_entry = _find_bench_entry(profile, "sy2sb")

    sb2st_tile_buckets = _derive_param_buckets(
        sb2st_entry, "back_tile", 0, 0, 0, 0, 0, direction="min")
    sb2st_subs_buckets = _derive_param_buckets(
        sb2st_entry, "back_subs", 0, 0, 0, 0, 0, direction="min")
    sy2sb_nb_buckets = _derive_param_buckets(
        sy2sb_entry, "sy2sb_nb", 0, 0, 0, 0, 0, direction="min")

    latrd_wg_fallback = int(latrd_params.get("wg", args.fallback_latrd_lower_panel_wg_hint))
    latrd_wg_buckets = _derive_param_buckets(syev_entry,
                                             "wg",
                                             latrd_wg_fallback,
                                             latrd_wg_fallback,
                                             latrd_wg_fallback,
                                             latrd_wg_fallback,
                                             latrd_wg_fallback,
                                             direction="min")

    sytrd_fuse_default = int(syev_params.get("fuse", 0))
    sytrd_fuse_buckets = _derive_param_buckets(
        syev_entry,
        "fuse",
        args.fallback_sytrd_fuse_panel_update_tiny if args.fallback_sytrd_fuse_panel_update_tiny >= 0 else sytrd_fuse_default,
        args.fallback_sytrd_fuse_panel_update_small if args.fallback_sytrd_fuse_panel_update_small >= 0 else sytrd_fuse_default,
        args.fallback_sytrd_fuse_panel_update_medium if args.fallback_sytrd_fuse_panel_update_medium >= 0 else sytrd_fuse_default,
        args.fallback_sytrd_fuse_panel_update_large if args.fallback_sytrd_fuse_panel_update_large >= 0 else sytrd_fuse_default,
        args.fallback_sytrd_fuse_panel_update_xlarge if args.fallback_sytrd_fuse_panel_update_xlarge >= 0 else sytrd_fuse_default,
        direction="min")

    stedc_entry = _find_bench_entry(profile, "stedc")
    stedc_params = _find_best_params(profile, "stedc")

    def _stedc_fallback(param_cli: str, bucket: str) -> int:
        v = getattr(args, f"fallback_stedc_{param_cli.replace('-', '_')}_{bucket}")
        return v if v > 0 else -1

    stedc_rt_buckets = _derive_param_buckets(
        stedc_entry, "recursion_threshold",
        *[_stedc_fallback("recursion-threshold", b) if _stedc_fallback("recursion-threshold", b) > 0
          else int(stedc_params.get("recursion_threshold", 32)) for b in ["tiny", "small", "medium", "large", "xlarge"]],
        direction="min")
    stedc_mv_buckets = _derive_param_buckets(
        stedc_entry, "flat",
        *[_stedc_fallback("merge-variant", b) if _stedc_fallback("merge-variant", b) > 0
          else int(stedc_params.get("flat", 2)) for b in ["tiny", "small", "medium", "large", "xlarge"]],
        direction="min")
    stedc_tpr_buckets = _derive_param_buckets(
        stedc_entry, "threads_per_root",
        *[_stedc_fallback("threads-per-root", b) if _stedc_fallback("threads-per-root", b) > 0
          else int(stedc_params.get("threads_per_root", 32)) for b in ["tiny", "small", "medium", "large", "xlarge"]],
        direction="min")
    stedc_wgm_buckets = _derive_param_buckets(
        stedc_entry, "wg_multiplier",
        *[_stedc_fallback("wg-multiplier", b) if _stedc_fallback("wg-multiplier", b) > 0
          else int(stedc_params.get("wg_multiplier", 1)) for b in ["tiny", "small", "medium", "large", "xlarge"]],
        direction="min")

    _emit_header(args.out,
                 args.profile,
                 ormqr_buckets["tiny"],
                 ormqr_buckets["small"],
                 ormqr_buckets["medium"],
                 ormqr_buckets["large"],
                 ormqr_buckets["xlarge"],
                 gebrd_buckets,
                 sb2st_tile_buckets,
                 sb2st_subs_buckets,
                 sy2sb_nb_buckets,
                 sytrd_buckets["tiny"],
                 sytrd_buckets["small"],
                 sytrd_buckets["medium"],
                 sytrd_buckets["large"],
                 sytrd_buckets["xlarge"],
                 latrd_wg_buckets,
                 sytrd_fuse_buckets,
                 stedc_rt_buckets,
                 stedc_mv_buckets,
                 stedc_tpr_buckets,
                 stedc_wgm_buckets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
