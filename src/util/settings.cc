// The one place BatchLAS calls std::getenv. RULE: each field uses the SAME parser its call site
// used (bespoke parsers stay at the site; the field is an EnvValue). Field order follows the header.
// evidence: docs/design/runtime-internals.md#runtime-internals-the-settings-loader

#include <batchlas/settings.hh>
// Direct: this throwing TU never reaches sycl-device-queue.hh, where others get error.hh.
#include <batchlas/error.hh>

#include <batchlas/backend_config.h>
#include <batchlas/util/env.hh>

#include <atomic>
#include <cctype>
#include <cstdio>
#include <optional>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

// From backend_config.h.in; defaulted to the SAFE reading for an older generated header.
#ifndef BATCHLAS_ALLOW_UNSAFE_ENV
#define BATCHLAS_ALLOW_UNSAFE_ENV 0
#endif

namespace batchlas {

namespace {

// Function-local static: dispatch coverage reads settings from a static initialiser.
Settings& mutable_settings() {
    static Settings s;
    return s;
}

// What configure() last set, kept apart so reload_settings() can re-apply it.
std::optional<Settings>& configured_settings() {
    static std::optional<Settings> s;
    return s;
}

std::once_flag g_load_once;

// Closes configure(). Atomic: a Queue may be built on another thread.
std::atomic<bool> g_queue_constructed{false};

// The unsafe gate: one warning per VARIABLE per process (reloads are frequent); never reset.
#if !BATCHLAS_ALLOW_UNSAFE_ENV
std::once_flag g_warn_skip_pointer_checks;
std::once_flag g_warn_latrd_force_unsafe;
std::once_flag g_warn_blas_health_off;

void warn_unsafe_ignored(const char* variable, const char* consequence) {
    std::fprintf(stderr,
                 "batchlas: %s is set in the environment, but this build was configured "
                 "with BATCHLAS_ALLOW_UNSAFE_ENV=OFF, so it is being IGNORED (%s). "
                 "Reconfigure with -DBATCHLAS_ALLOW_UNSAFE_ENV=ON to allow it.\n",
                 variable, consequence);
}
#endif

// ---------------------------------------------------------------------------

EnvValue raw(const char* name) { return EnvValue(std::getenv(name)); }

// Not env_string_or(): two knobs below must NOT map set-but-empty to the fallback.
const char* raw_or_null(const char* name) { return std::getenv(name); }

// Assigns only an EXPORTED variable, so a reload keeps configure()'s value for every unset one.
// `parse` takes the variable's name. evidence: docs/design/environment.md#environment-configure-sets-the-base-not-a-lock
template <class T, class Parse>
void set_if_present(T& field, const char* name, Parse parse) {
    if (std::getenv(name) != nullptr) field = parse(name);
}
void set_if_present(EnvValue& field, const char* name) { set_if_present(field, name, raw); }

bool truthy(const char* name) { return env_truthy(std::getenv(name)); }
auto positive_or(int fallback) { return [fallback](const char* n) { return env_positive_int_or(n, fallback); }; }
auto int_or(int fallback) { return [fallback](const char* n) { return env_int_or(n, fallback); }; }
bool any_present(const char* a, const char* b) { return std::getenv(a) != nullptr || std::getenv(b) != nullptr; }

void load_routing(RoutingSettings& r) {
    // The names are synthesised ("BATCHLAS_" + upper-cased op + "_ROUTE"), so a grep
    // for BATCHLAS_* string literals misses them.
    for (std::size_t i = 0; i < RoutingSettings::ops.size(); ++i) {
        std::string name = "BATCHLAS_";
        for (const char c : RoutingSettings::ops[i])
            name += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        name += "_ROUTE";
        set_if_present(r.values[i], name.c_str());
    }
}

void load_selection(SelectionSettings& s) {
    set_if_present(s.expand_route, "BATCHLAS_EXPAND_ROUTE");
    set_if_present(s.gemv_segt, "BATCHLAS_GEMV_SEGT");
    set_if_present(s.gesvd_bidiag, "BATCHLAS_GESVD_BIDIAG");
    set_if_present(s.geqrf_leaf, "BATCHLAS_GEQRF_LEAF");
    set_if_present(s.getrf_laswp, "BATCHLAS_GETRF_LASWP");
    set_if_present(s.getrf_right_laswp, "BATCHLAS_GETRF_RIGHT_LASWP");
    set_if_present(s.getrf_leaf, "BATCHLAS_GETRF_LEAF");
    set_if_present(s.getrs_laswp, "BATCHLAS_GETRS_LASWP");
    set_if_present(s.iluk_device, "BATCHLAS_ILUK_DEVICE");
    set_if_present(s.latrd_impl, "BATCHLAS_LATRD_IMPL");
    set_if_present(s.ormqr_impl, "BATCHLAS_ORMQR_IMPL");
    set_if_present(s.ormqr_wy, "BATCHLAS_ORMQR_WY");
    set_if_present(s.ortho_gram, "BATCHLAS_ORTHO_GRAM");
    set_if_present(s.sb2st_back_wave, "BATCHLAS_SB2ST_BACK_WAVE");
    set_if_present(s.sb2st_subgroup, "BATCHLAS_SB2ST_SUBGROUP");
    set_if_present(s.syev_two_stage_chase, "BATCHLAS_SYEV_TWO_STAGE_CHASE");
    set_if_present(s.syevx_algorithm, "BATCHLAS_SYEVX_ALGORITHM");
    set_if_present(s.syevx_preconditioner, "BATCHLAS_SYEVX_PRECONDITIONER");
    set_if_present(s.syevx_filter_degree_auto, "BATCHLAS_SYEVX_FILTER_DEGREE_AUTO");
    set_if_present(s.syevx_instr_host, "BATCHLAS_SYEVX_INSTR_HOST");
    set_if_present(s.syevx_projected_vendor, "BATCHLAS_SYEVX_PROJECTED_VENDOR");
    set_if_present(s.syevx_soft_lock, "BATCHLAS_SYEVX_SOFT_LOCK");
    set_if_present(s.sytrd_impl, "BATCHLAS_SYTRD_IMPL");
    set_if_present(s.sytrd_trailing_update, "BATCHLAS_SYTRD_TRAILING_UPDATE");
    set_if_present(s.tuned_dir, "BATCHLAS_TUNED_DIR");

    set_if_present(s.sytrd_force_local_small, "BATCHLAS_SYTRD_FORCE_LOCAL_SMALL", truthy);

    // Tri-state from ONE read of the value, as the call site does.
    {
        const char* v = std::getenv("BATCHLAS_SYTRD_FUSE_PANEL_UPDATE");
        if (v) {
            if (env_truthy(v)) {
                s.sytrd_fuse_panel_update = true;
            } else if (env_falsy(v)) {
                s.sytrd_fuse_panel_update = false;
            } else {
                s.sytrd_fuse_panel_update.reset();  // a spelling that is neither
            }
        }
    }
}

void load_geometry(GeometrySettings& g) {
    // env_positive_int_or == the old bare-atoi form ("<= 0 is unset") except on int overflow (safe side).
    set_if_present(g.latrd_grid_groups, "BATCHLAS_LATRD_GRID_GROUPS", positive_or(0));
    set_if_present(g.latrd_grid_min_n, "BATCHLAS_LATRD_GRID_MIN_N", positive_or(768));
    set_if_present(g.latrd_grid_wg, "BATCHLAS_LATRD_GRID_WG", positive_or(0));
    set_if_present(g.latrd_lower_panel_wg_hint, "BATCHLAS_LATRD_LOWER_PANEL_WG_HINT", positive_or(0));

    set_if_present(g.sb2st_back_subs, "BATCHLAS_SB2ST_BACK_SUBS", positive_or(0));
    set_if_present(g.sb2st_back_tile_w, "BATCHLAS_SB2ST_BACK_TILE_W", positive_or(0));

    // RAW: 0 selects the streaming kernel here, so "<= 0 means unset" is wrong.
    set_if_present(g.sb2st_back_tile, "BATCHLAS_SB2ST_BACK_TILE");

    set_if_present(g.potrf_nb, "BATCHLAS_POTRF_NB", positive_or(0));
    set_if_present(g.potrf_w, "BATCHLAS_POTRF_W", positive_or(0));

    set_if_present(g.syev_two_stage_kd, "BATCHLAS_SYEV_TWO_STAGE_KD", positive_or(32));
    set_if_present(g.syev_two_stage_sb2st_block, "BATCHLAS_SYEV_TWO_STAGE_SB2ST_BLOCK", positive_or(32));

    set_if_present(g.sy2sb_ormqr_nb, "BATCHLAS_SY2SB_ORMQR_NB");   // three-valued: unset / off / n

    set_if_present(g.sytrd_block_size, "BATCHLAS_SYTRD_BLOCK_SIZE", positive_or(0));
    set_if_present(g.trmm_tile_m, "BATCHLAS_TRMM_TILE_M", positive_or(0));
    set_if_present(g.trsm_outer_nb, "BATCHLAS_TRSM_OUTER_NB", positive_or(0));

    set_if_present(g.expand_max_bytes, "BATCHLAS_EXPAND_MAX_BYTES");            // strtoull, size_t
    set_if_present(g.gesvd_blocked_gebrd_min, "BATCHLAS_GESVD_BLOCKED_GEBRD_MIN");  // atoi, 0 < default

    // syevx / LOBPCG keep their call sites' atoi forms: some accept 0 as a MEANING.
    {
        if (const char* v = std::getenv("BATCHLAS_SYEVX_CHECK_EVERY")) {
            const int parsed = std::atoi(v);
            g.syevx_check_every = parsed > 0 ? parsed : 4;
        }
    }
    {
        if (const char* v = std::getenv("BATCHLAS_SYEVX_EXTRA_DIRECTIONS")) {
            g.syevx_extra_directions.reset();
            const int parsed = std::atoi(v);
            if (parsed >= 0) g.syevx_extra_directions = parsed;  // 0 == "no guard block"
        }
    }
    {
        if (const char* v = std::getenv("BATCHLAS_SYEVX_FILTER_DEGREE")) {
            const int parsed = std::atoi(v);
            g.syevx_filter_degree = parsed > 0 ? parsed : 0;
        }
    }
    {
        if (const char* v = std::getenv("BATCHLAS_SYEVX_INIT_POWER")) {
            g.syevx_init_power.reset();
            const int parsed = std::atoi(v);
            if (parsed >= 0) g.syevx_init_power = parsed;  // 0 == "no power iterations"
        }
    }
    {
        if (const char* v = std::getenv("BATCHLAS_SYEVX_LOCK_FACTOR")) {
            const double parsed = std::atof(v);
            g.syevx_lock_factor = parsed > 0.0 ? parsed : 0.1;
        }
    }

    set_if_present(g.tune.ormqr_block_size, "BATCHLAS_TUNE_ORMQR_BLOCK_SIZE");
    set_if_present(g.tune.gebrd_block_size, "BATCHLAS_TUNE_GEBRD_BLOCK_SIZE");
    set_if_present(g.tune.sb2st_back_tile, "BATCHLAS_TUNE_SB2ST_BACK_TILE");
    set_if_present(g.tune.sb2st_back_subs, "BATCHLAS_TUNE_SB2ST_BACK_SUBS");
    set_if_present(g.tune.sy2sb_ormqr_nb, "BATCHLAS_TUNE_SY2SB_ORMQR_NB");
    set_if_present(g.tune.sytrd_block_size, "BATCHLAS_TUNE_SYTRD_BLOCK_SIZE");
    set_if_present(g.tune.latrd_wg_hint, "BATCHLAS_TUNE_LATRD_WG_HINT");
    set_if_present(g.tune.stedc_recursion_threshold, "BATCHLAS_TUNE_STEDC_RECURSION_THRESHOLD");
    set_if_present(g.tune.stedc_merge_variant, "BATCHLAS_TUNE_STEDC_MERGE_VARIANT");
    set_if_present(g.tune.stedc_threads_per_root, "BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT");
    set_if_present(g.tune.stedc_wg_multiplier, "BATCHLAS_TUNE_STEDC_WG_MULTIPLIER");
}

void load_diagnostics(DiagnosticsSettings& d) {
    if (any_present("BATCHLAS_QUEUE_PROFILING", "BATCHLAS_BENCH_PROFILING"))
        d.profiling = truthy("BATCHLAS_QUEUE_PROFILING") || truthy("BATCHLAS_BENCH_PROFILING");

    if (any_present("BATCHLAS_KERNEL_TRACE", "BATCHLAS_TRACE_KERNELS"))
        d.kernel_trace = truthy("BATCHLAS_KERNEL_TRACE") || truthy("BATCHLAS_TRACE_KERNELS");
    set_if_present(d.metal_flush_each_op, "BATCHLAS_METAL_FLUSH_EACH_OP", truthy);

    // First NON-EMPTY wins: a set-but-empty variable falls through, never an empty filename.
    if (any_present("BATCHLAS_KERNEL_TRACE_PATH", "BATCHLAS_TRACE_PATH"))
        d.kernel_trace_path = "batchlas_kernels.trace.json";
    if (const char* p = raw_or_null("BATCHLAS_KERNEL_TRACE_PATH"); p && *p) {
        d.kernel_trace_path = p;
    } else if (const char* q = raw_or_null("BATCHLAS_TRACE_PATH"); q && *q) {
        d.kernel_trace_path = q;
    }

    set_if_present(d.coverage_out, "BATCHLAS_COVERAGE_OUT");
    set_if_present(d.select_trace, "BATCHLAS_SELECT_TRACE", truthy);

    // PRESENCE ONLY, empty string included: the call site tests the pointer.
    set_if_present(d.debug_filter_degree, "BATCHLAS_DEBUG_FILTER_DEGREE", [](const char*) { return true; });

    set_if_present(d.debug_sytrd_small, "BATCHLAS_DEBUG_SYTRD_SMALL", truthy);
    set_if_present(d.gesvd_profile, "BATCHLAS_GESVD_PROFILE", truthy);
    set_if_present(d.cta_debug_sync, "BATCHLAS_CTA_DEBUG_SYNC", truthy);

    set_if_present(d.syevx_trace, "BATCHLAS_SYEVX_TRACE", truthy);

    set_if_present(d.steqr_cta_check, "BATCHLAS_STEQR_CTA_CHECK");

    // Deliberate asymmetry with the trace path: ANY set value, empty included, is taken.
    if (const char* v = raw_or_null("BATCHLAS_DUMP_BANDR1_DIR")) {
        d.dump_bandr1.dir = v;
    }

    set_if_present(d.dump_bandr1.step, "BATCHLAS_DUMP_BANDR1_STEP", truthy);
    set_if_present(d.dump_bandr1.abw_only, "BATCHLAS_DUMP_BANDR1_ABW_ONLY", truthy);
    set_if_present(d.dump_bandr1.step_index, "BATCHLAS_DUMP_BANDR1_STEP_INDEX", int_or(-1));
    set_if_present(d.dump_bandr1.sweep_index, "BATCHLAS_DUMP_BANDR1_SWEEP_INDEX", int_or(-1));
    set_if_present(d.dump_bandr1.step_in_sweep, "BATCHLAS_DUMP_BANDR1_STEP_IN_SWEEP", int_or(-1));
    set_if_present(d.dump_bandr1.batch, "BATCHLAS_DUMP_BANDR1_BATCH", int_or(-1));
}

void load_unsafe(UnsafeSettings& u) {
    // Read exactly as the call site did, THEN gate, so a warning names a variable actually set.
    // Only a move the ENVIRONMENT made is gated: configure() is not, and is the base here.
    [[maybe_unused]] const UnsafeSettings base = u;

    // ANY non-empty value not starting with '0' skips, "false"/"off"/"no" included. Not env_truthy.
    {
        const char* v = std::getenv("BATCHLAS_SKIP_POINTER_CHECKS");
        if (v) u.skip_pointer_checks = (*v && *v != '0');
    }

    set_if_present(u.latrd_grid_force_unsafe, "BATCHLAS_LATRD_GRID_FORCE_UNSAFE", truthy);

    if (const char* v = std::getenv("BATCHLAS_BLAS_HEALTH")) {
        // off | warn (default) | error, compared exactly as the call site does.
        const std::string mode = v;
        if (mode == "off") {
            u.blas_health = UnsafeSettings::BlasHealth::Off;
        } else if (mode == "error") {
            u.blas_health = UnsafeSettings::BlasHealth::Error;
        } else {
            u.blas_health = UnsafeSettings::BlasHealth::Warn;
        }
    }

#if !BATCHLAS_ALLOW_UNSAFE_ENV
    // The gate refuses the UNSAFE DIRECTION, not every non-default value:
    // BlasHealth::Error is stricter than the default and is let through.
    if (u.skip_pointer_checks && !base.skip_pointer_checks) {
        u.skip_pointer_checks = base.skip_pointer_checks;
        std::call_once(g_warn_skip_pointer_checks, [] {
            warn_unsafe_ignored("BATCHLAS_SKIP_POINTER_CHECKS",
                                "device-pointer argument checks stay ENABLED");
        });
    }
    if (u.latrd_grid_force_unsafe && !base.latrd_grid_force_unsafe) {
        u.latrd_grid_force_unsafe = false;
        std::call_once(g_warn_latrd_force_unsafe, [] {
            warn_unsafe_ignored("BATCHLAS_LATRD_GRID_FORCE_UNSAFE",
                                "the grid-latrd work-group count stays clamped to the "
                                "co-residency cap, which is what keeps its spin barrier "
                                "from deadlocking");
        });
    }
    if (u.blas_health == UnsafeSettings::BlasHealth::Off && base.blas_health != u.blas_health) {
        u.blas_health = base.blas_health;
        std::call_once(g_warn_blas_health_off, [] {
            warn_unsafe_ignored("BATCHLAS_BLAS_HEALTH=off",
                                "the host-dgemm correctness probe still runs; it is the only "
                                "detection of a host BLAS that computes dgemm incorrectly");
        });
    }
#endif
}

void load_from_env(Settings& s) {
    load_routing(s.routing);
    load_selection(s.selection);
    load_geometry(s.geometry);
    load_diagnostics(s.diagnostics);
    load_unsafe(s.unsafe);
}

}  // namespace

const Settings& settings() {
    std::call_once(g_load_once, [] { load_from_env(mutable_settings()); });
    return mutable_settings();
}

void configure(const Settings& s) {
    if (detail::queue_constructed()) {
        throw batchlas::api_misuse(
            "BatchLAS: batchlas::configure() was called after a Queue had already been "
            "constructed. Routing and geometry settings are read by *_buffer_size() queries "
            "as well as by the matching solve, and several of them change how much scratch a "
            "solve needs, so changing them mid-run would let two calls in one process "
            "disagree. Call configure() before constructing any Queue.");
    }
    // Burn the one-time load first, or a later settings() would overwrite this.
    (void)settings();
    mutable_settings() = s;
    configured_settings() = s;
}

namespace detail {

void reload_settings() {
    // settings() first, as in configure(), or call_once would later discard this reload.
    (void)settings();
    // Environment ON TOP OF configure() (or the defaults), never ignored; the loaders assign only
    // exported variables. evidence: docs/design/runtime-internals.md#runtime-internals-the-settings-loader
    mutable_settings() = configured_settings().value_or(Settings{});
    load_from_env(mutable_settings());
}

void note_queue_constructed() noexcept {
    g_queue_constructed.store(true, std::memory_order_relaxed);
}

bool queue_constructed() noexcept {
    return g_queue_constructed.load(std::memory_order_relaxed);
}

}  // namespace detail

}  // namespace batchlas
