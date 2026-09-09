// The one place BatchLAS calls std::getenv.
//
// Everything about WHY this file exists is on <batchlas/settings.hh>. What is
// here is the load itself, and the one rule that governs it: each field is
// produced by the SAME parser its call site used before, so that migrating a
// call site is a change of where the string comes from and never a change of
// what the string means. Where a site had a bespoke parser the field is an
// EnvValue and the parser stays at the site; this file only captures the string.
//
// The order of the assignments below follows the order of the fields in the
// header, so the two read side by side.

#include <batchlas/settings.hh>

#include <batchlas/backend_config.h>
#include <batchlas/blas/dispatch/route_env.hh>  // legacy_variable_for
#include <batchlas/util/env.hh>

#include <atomic>
#include <cstdio>
#include <optional>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

// Set by cmake/BatchLASOptions.cmake through cmake/backend_config.h.in
// (#cmakedefine01, so it is 0 or 1 whenever that header has been regenerated).
// Defaulted here so this file still compiles against an older generated header,
// and so the SAFE reading is the one you get when nobody has said otherwise.
#ifndef BATCHLAS_ALLOW_UNSAFE_ENV
#define BATCHLAS_ALLOW_UNSAFE_ENV 0
#endif

namespace batchlas {

namespace {

// The single mutable copy. A function-local static, so it is initialised on
// first use rather than in static-initialisation order: dispatch coverage reads
// its variable from a namespace-scope dynamic initialiser, and that has to work.
Settings& mutable_settings() {
    static Settings s;
    return s;
}

// What configure() was last given, if anything. Held separately from
// mutable_settings() so that reload_settings() can tell "the environment is the
// authority here" from "the caller took the wheel", and re-apply the latter.
// Same function-local-static reasoning as above.
std::optional<Settings>& configured_settings() {
    static std::optional<Settings> s;
    return s;
}

std::once_flag g_load_once;

// Closes configure(). std::atomic rather than a plain bool because a Queue may
// legitimately be constructed on a thread other than the one that would call
// configure(), and this is the one flag in this file that can race.
std::atomic<bool> g_queue_constructed{false};

// ---------------------------------------------------------------------------
// The unsafe gate.
//
// One warning per VARIABLE per process, not per load: reload_settings() runs
// once per ScopedEnvVar construction and destruction, so a per-load warning
// would print thousands of lines in the test suite. These flags are deliberately
// never reset.
//
// Compiled out entirely when the option is ON: nothing is refused then, so an
// unused warning helper would only earn a -Wunused-function diagnostic.
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

// env_string_or() maps a set-but-empty variable to the fallback. Two of the
// string knobs here must NOT do that -- see the call sites in the loader -- so
// the plain read is spelled out rather than reached through a helper whose
// empty-string rule is the opposite of what those sites want.
const char* raw_or_null(const char* name) { return std::getenv(name); }

void load_routing(RoutingSettings& r) {
    // Both arrays are indexed by Op and both are rebuilt from scratch, so a
    // reload cannot leave a stale entry behind for a variable that has since
    // been unset.
    for (std::size_t i = 0; i < static_cast<std::size_t>(dispatch::Op::COUNT); ++i) {
        const auto op = static_cast<dispatch::Op>(i);

        // The canonical name is SYNTHESISED, exactly as parse_route_env
        // synthesised it: "BATCHLAS_" + the upper-cased op name + "_ROUTE".
        // This is why a grep for BATCHLAS_* string literals misses thirteen live
        // routing variables -- no literal for them exists anywhere in the tree.
        r.canonical[i] = raw(("BATCHLAS_" + dispatch::op_env_stem(op) + "_ROUTE").c_str());

        // legacy_variable_for() returns an empty view for the ops that never had
        // a legacy spelling; the entry stays unset for those.
        const std::string_view legacy = dispatch::legacy_variable_for(op);
        r.legacy[i] = legacy.empty() ? EnvValue::unset() : raw(std::string(legacy).c_str());
    }
}

void load_selection(SelectionSettings& s) {
    s.expand_route = raw("BATCHLAS_EXPAND_ROUTE");
    s.gemm_cublasdx_kernel = raw("BATCHLAS_GEMM_CUBLASDX_KERNEL");
    s.gemm_experimental = raw("BATCHLAS_GEMM_EXPERIMENTAL");
    s.gemm_sycl_kernel = raw("BATCHLAS_GEMM_SYCL_KERNEL");
    s.gemv_segt = raw("BATCHLAS_GEMV_SEGT");
    s.gesvd_bidiag = raw("BATCHLAS_GESVD_BIDIAG");
    s.getrf_laswp = raw("BATCHLAS_GETRF_LASWP");
    s.getrs_laswp = raw("BATCHLAS_GETRS_LASWP");
    s.iluk_device = raw("BATCHLAS_ILUK_DEVICE");
    s.latrd_impl = raw("BATCHLAS_LATRD_IMPL");
    s.ormqr_impl = raw("BATCHLAS_ORMQR_IMPL");
    s.ormqr_wy = raw("BATCHLAS_ORMQR_WY");
    s.ortho_gram = raw("BATCHLAS_ORTHO_GRAM");
    s.sb2st_back_wave = raw("BATCHLAS_SB2ST_BACK_WAVE");
    s.sb2st_subgroup = raw("BATCHLAS_SB2ST_SUBGROUP");
    s.syev_small_kernel = raw("BATCHLAS_SYEV_SMALL_KERNEL");
    s.syev_two_stage_chase = raw("BATCHLAS_SYEV_TWO_STAGE_CHASE");
    s.syevx_algorithm = raw("BATCHLAS_SYEVX_ALGORITHM");
    s.syevx_preconditioner = raw("BATCHLAS_SYEVX_PRECONDITIONER");
    s.syevx_bounds_legacy = raw("BATCHLAS_SYEVX_BOUNDS_LEGACY");
    s.syevx_filter_degree_auto = raw("BATCHLAS_SYEVX_FILTER_DEGREE_AUTO");
    s.syevx_instr_host = raw("BATCHLAS_SYEVX_INSTR_HOST");
    s.syevx_projected_vendor = raw("BATCHLAS_SYEVX_PROJECTED_VENDOR");
    s.syevx_soft_lock = raw("BATCHLAS_SYEVX_SOFT_LOCK");
    s.sytrd_impl = raw("BATCHLAS_SYTRD_IMPL");
    s.sytrd_trailing_update = raw("BATCHLAS_SYTRD_TRAILING_UPDATE");

    s.sytrd_force_local_small = env_truthy(std::getenv("BATCHLAS_SYTRD_FORCE_LOCAL_SMALL"));

    // The tri-state. Both questions are asked of ONE read of the value, which is
    // what the call site does; asking getenv twice would be a different program
    // if the environment changed in between.
    {
        const char* v = std::getenv("BATCHLAS_SYTRD_FUSE_PANEL_UPDATE");
        if (env_truthy(v)) {
            s.sytrd_fuse_panel_update = true;
        } else if (env_falsy(v)) {
            s.sytrd_fuse_panel_update = false;
        } else {
            s.sytrd_fuse_panel_update.reset();  // unset, or a spelling that is neither
        }
    }
}

void load_geometry(GeometrySettings& g) {
    // env_positive_int_or is the call sites' own reader for the first three, and
    // is byte-for-byte equivalent to the bare-atoi form the rest of this group
    // used: both mean "unparseable or <= 0 is unset". The one input class that
    // differs is an integer literal too large for int, where atoi is undefined
    // and stoi-in-a-try lands on the fallback -- a change in the safe direction.
    g.latrd_grid_groups = env_positive_int_or("BATCHLAS_LATRD_GRID_GROUPS", 0);
    g.latrd_grid_min_n = env_positive_int_or("BATCHLAS_LATRD_GRID_MIN_N", 768);
    g.latrd_grid_wg = env_positive_int_or("BATCHLAS_LATRD_GRID_WG", 0);
    g.latrd_lower_panel_wg_hint = env_positive_int_or("BATCHLAS_LATRD_LOWER_PANEL_WG_HINT", 0);

    g.sb2st_back_subs = env_positive_int_or("BATCHLAS_SB2ST_BACK_SUBS", 0);
    g.sb2st_back_tile_w = env_positive_int_or("BATCHLAS_SB2ST_BACK_TILE_W", 0);

    // RAW: 0 selects the streaming kernel here, so "<= 0 means unset" is wrong.
    g.sb2st_back_tile = raw("BATCHLAS_SB2ST_BACK_TILE");

    g.potrf_nb = env_positive_int_or("BATCHLAS_POTRF_NB", 0);
    g.potrf_w = env_positive_int_or("BATCHLAS_POTRF_W", 0);

    g.syev_two_stage_kd = env_positive_int_or("BATCHLAS_SYEV_TWO_STAGE_KD", 32);
    g.syev_two_stage_sb2st_block =
        env_positive_int_or("BATCHLAS_SYEV_TWO_STAGE_SB2ST_BLOCK", 32);

    g.sy2sb_ormqr_nb = raw("BATCHLAS_SY2SB_ORMQR_NB");   // three-valued: unset / off / n
    g.syev_cta_max_n = raw("BATCHLAS_SYEV_CTA_MAX_N");   // strtol, range-checked, typed default

    g.sytrd_block_size = env_positive_int_or("BATCHLAS_SYTRD_BLOCK_SIZE", 0);
    g.trmm_tile_m = env_positive_int_or("BATCHLAS_TRMM_TILE_M", 0);
    g.trsm_outer_nb = env_positive_int_or("BATCHLAS_TRSM_OUTER_NB", 0);

    g.expand_max_bytes = raw("BATCHLAS_EXPAND_MAX_BYTES");            // strtoull, size_t
    g.gesvd_blocked_gebrd_min = raw("BATCHLAS_GESVD_BLOCKED_GEBRD_MIN");  // atoi, 0 < default

    // syevx / LOBPCG. Each of these reproduces its call site's atoi form rather
    // than env.hh's, because two of them accept 0 as a MEANING and one accepts a
    // value below its own default.
    {
        const char* v = std::getenv("BATCHLAS_SYEVX_CHECK_EVERY");
        const int parsed = v ? std::atoi(v) : 0;
        g.syevx_check_every = parsed > 0 ? parsed : 4;
    }
    {
        const char* v = std::getenv("BATCHLAS_SYEVX_EXTRA_DIRECTIONS");
        g.syevx_extra_directions.reset();
        if (v) {
            const int parsed = std::atoi(v);
            if (parsed >= 0) g.syevx_extra_directions = parsed;  // 0 == "no guard block"
        }
    }
    {
        const char* v = std::getenv("BATCHLAS_SYEVX_FILTER_DEGREE");
        const int parsed = v ? std::atoi(v) : 0;
        g.syevx_filter_degree = parsed > 0 ? parsed : 0;
    }
    {
        const char* v = std::getenv("BATCHLAS_SYEVX_INIT_POWER");
        g.syevx_init_power.reset();
        if (v) {
            const int parsed = std::atoi(v);
            if (parsed >= 0) g.syevx_init_power = parsed;  // 0 == "no power iterations"
        }
    }
    {
        const char* v = std::getenv("BATCHLAS_SYEVX_LOCK_FACTOR");
        const double parsed = v ? std::atof(v) : 0.0;
        g.syevx_lock_factor = parsed > 0.0 ? parsed : 0.1;
    }

    g.tune.ormqr_block_size = raw("BATCHLAS_TUNE_ORMQR_BLOCK_SIZE");
    g.tune.gebrd_block_size = raw("BATCHLAS_TUNE_GEBRD_BLOCK_SIZE");
    g.tune.sb2st_back_tile = raw("BATCHLAS_TUNE_SB2ST_BACK_TILE");
    g.tune.sb2st_back_subs = raw("BATCHLAS_TUNE_SB2ST_BACK_SUBS");
    g.tune.sy2sb_ormqr_nb = raw("BATCHLAS_TUNE_SY2SB_ORMQR_NB");
    g.tune.sytrd_block_size = raw("BATCHLAS_TUNE_SYTRD_BLOCK_SIZE");
    g.tune.latrd_wg_hint = raw("BATCHLAS_TUNE_LATRD_WG_HINT");
    g.tune.stedc_recursion_threshold = raw("BATCHLAS_TUNE_STEDC_RECURSION_THRESHOLD");
    g.tune.stedc_merge_variant = raw("BATCHLAS_TUNE_STEDC_MERGE_VARIANT");
    g.tune.stedc_threads_per_root = raw("BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT");
    g.tune.stedc_wg_multiplier = raw("BATCHLAS_TUNE_STEDC_WG_MULTIPLIER");
}

void load_diagnostics(DiagnosticsSettings& d) {
    d.profiling = env_truthy(std::getenv("BATCHLAS_QUEUE_PROFILING")) ||
                  env_truthy(std::getenv("BATCHLAS_BENCH_PROFILING"));

    d.kernel_trace = env_truthy(std::getenv("BATCHLAS_KERNEL_TRACE")) ||
                     env_truthy(std::getenv("BATCHLAS_TRACE_KERNELS"));

    // First NON-EMPTY wins, then the built-in default. The emptiness test is the
    // point: BATCHLAS_KERNEL_TRACE_PATH= (set, empty) falls through to
    // BATCHLAS_TRACE_PATH rather than producing an empty filename.
    d.kernel_trace_path = "batchlas_kernels.trace.json";
    if (const char* p = raw_or_null("BATCHLAS_KERNEL_TRACE_PATH"); p && *p) {
        d.kernel_trace_path = p;
    } else if (const char* q = raw_or_null("BATCHLAS_TRACE_PATH"); q && *q) {
        d.kernel_trace_path = q;
    }

    d.coverage_out = raw("BATCHLAS_COVERAGE_OUT");

    // PRESENCE ONLY, empty string included: the call site tests the pointer.
    d.debug_filter_degree = raw_or_null("BATCHLAS_DEBUG_FILTER_DEGREE") != nullptr;

    d.debug_sytrd_small = env_truthy(std::getenv("BATCHLAS_DEBUG_SYTRD_SMALL"));
    d.gesvd_profile = env_truthy(std::getenv("BATCHLAS_GESVD_PROFILE"));
    d.cta_debug_sync = env_truthy(std::getenv("BATCHLAS_CTA_DEBUG_SYNC"));

    // The call site hand-rolls env_truthy's exact spelling set; this is the same
    // function, not a widening.
    d.syevx_trace = env_truthy(std::getenv("BATCHLAS_SYEVX_TRACE"));

    d.steqr_cta_check = raw("BATCHLAS_STEQR_CTA_CHECK");

    // Note the asymmetry with kernel_trace_path above: bandr1_dump_root() returns
    // std::string(v) for ANY set value, so BATCHLAS_DUMP_BANDR1_DIR= (set, empty)
    // yields an empty root, not the default. Preserved.
    d.dump_bandr1.dir = "output/bandr1_dumps";
    if (const char* v = raw_or_null("BATCHLAS_DUMP_BANDR1_DIR")) {
        d.dump_bandr1.dir = v;
    }

    d.dump_bandr1.step = env_truthy(std::getenv("BATCHLAS_DUMP_BANDR1_STEP"));
    d.dump_bandr1.abw_only = env_truthy(std::getenv("BATCHLAS_DUMP_BANDR1_ABW_ONLY"));
    d.dump_bandr1.step_index = env_int_or("BATCHLAS_DUMP_BANDR1_STEP_INDEX", -1);
    d.dump_bandr1.sweep_index = env_int_or("BATCHLAS_DUMP_BANDR1_SWEEP_INDEX", -1);
    d.dump_bandr1.step_in_sweep = env_int_or("BATCHLAS_DUMP_BANDR1_STEP_IN_SWEEP", -1);
    d.dump_bandr1.batch = env_int_or("BATCHLAS_DUMP_BANDR1_BATCH", -1);
}

void load_unsafe(UnsafeSettings& u) {
    // Every field is first read EXACTLY as its call site read it, and only then
    // gated. Reading first is what lets the warning name a variable that was
    // actually set, rather than one that merely exists.

    // !(v && *v && *v != '0') is "checks enabled", so the negation is "skip".
    // ANY non-empty value not starting with '0' skips -- including "false",
    // "off" and "no". Not env_truthy; see the header.
    {
        const char* v = std::getenv("BATCHLAS_SKIP_POINTER_CHECKS");
        u.skip_pointer_checks = (v && *v && *v != '0');
    }

    u.latrd_grid_force_unsafe = env_truthy(std::getenv("BATCHLAS_LATRD_GRID_FORCE_UNSAFE"));

    {
        // off | warn (default) | error, compared exactly as the call site does.
        const char* v = std::getenv("BATCHLAS_BLAS_HEALTH");
        const std::string mode = v ? v : "warn";
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
    if (u.skip_pointer_checks) {
        u.skip_pointer_checks = false;
        std::call_once(g_warn_skip_pointer_checks, [] {
            warn_unsafe_ignored("BATCHLAS_SKIP_POINTER_CHECKS",
                                "device-pointer argument checks stay ENABLED");
        });
    }
    if (u.latrd_grid_force_unsafe) {
        u.latrd_grid_force_unsafe = false;
        std::call_once(g_warn_latrd_force_unsafe, [] {
            warn_unsafe_ignored("BATCHLAS_LATRD_GRID_FORCE_UNSAFE",
                                "the grid-latrd work-group count stays clamped to the "
                                "co-residency cap, which is what keeps its spin barrier "
                                "from deadlocking");
        });
    }
    if (u.blas_health == UnsafeSettings::BlasHealth::Off) {
        u.blas_health = UnsafeSettings::BlasHealth::Warn;
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
        throw std::runtime_error(
            "BatchLAS: batchlas::configure() was called after a Queue had already been "
            "constructed. Routing and geometry settings are read by *_buffer_size() queries "
            "as well as by the matching solve, and several of them change how much scratch a "
            "solve needs, so changing them mid-run would let two calls in one process "
            "disagree. Call configure() before constructing any Queue.");
    }
    // Burn the one-time load first, so a later settings() cannot overwrite what
    // was just configured.
    (void)settings();
    mutable_settings() = s;
    configured_settings() = s;
}

namespace detail {

void reload_settings() {
    // settings() first, for the same reason configure() does it: the load must
    // have happened once before this overwrite, or std::call_once would run it
    // afterwards and discard the reload.
    (void)settings();
    // A reload re-reads the environment ON TOP OF what configure() last set,
    // rather than on top of the defaults. Both halves of that matter, and each
    // one is a defect the other way round:
    //
    //  * Starting from the DEFAULTS discarded configure() entirely, and a
    //    ScopedEnvVar on a COMPLETELY UNRELATED variable -- built anywhere in an
    //    embedding application's own harness -- was enough to do it, silently.
    //    That is the ambient-state defect this work package exists to remove.
    //  * Ignoring the environment once configure() had been called made every
    //    ScopedEnvVar in the process a no-op, which turns an A/B test into two
    //    runs of the same arm that agree by construction. Twelve guards in this
    //    tree have already failed that way; tests/settings_tests.cc's own
    //    regression case (c) is exactly it, and it calls configure() in case (a)
    //    first, so the whole binary would have gone quietly green.
    //
    // So configure() is the base and an explicitly set variable still wins over
    // it. A caller who wants a knob pinned against the environment sets it in
    // the Settings AND does not export it; a test that pins one gets its pin.
    if (configured_settings().has_value()) {
        mutable_settings() = *configured_settings();
    }
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
