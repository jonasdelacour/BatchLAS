#pragma once

/// @file
/// @brief Every `BATCHLAS_*` environment variable the library reads, captured once into
/// batchlas::Settings, plus the programmatic override configure().
///
/// The environment is read once, on the first call to settings(). A knob whose call site uses
/// one of the shared parsers in `<batchlas/util/env.hh>` is a typed field parsed by that same
/// parser; a knob with a bespoke parser is an EnvValue (the raw string) and its parser stays at
/// the call site, so every variable means exactly what it meant before this header existed.
/// The user guide, with a field/variable/default table per group, is the "Configuration"
/// section of @ref md_docs_2cpp-api.
/// @see @ref design_environment
/// @ingroup config
// Move where the STRING comes from, never how it is parsed (seven boolean dialects coexist).
// evidence: docs/design/environment.md#environment-move-the-string-not-the-parser

#include <batchlas/export.hh>
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace batchlas {

/// @brief A raw captured environment value, shaped like the `std::getenv` call it replaces.
///
/// Distinguishes "unset" from "set to the empty string", and hands back a `const char*` so a
/// migrated call site keeps the parser it already has.
/// @ingroup config
// Unset vs empty is load-bearing: the TRACE_PATH fall-through, and the route pin reader treats
// a set-but-empty BATCHLAS_<OP>_ROUTE as unset.
class EnvValue {
public:
    /// @brief An unset value.
    EnvValue() = default;

    /// @brief Construct from what `std::getenv` returned.
    /// @param raw the variable's value, or `nullptr` for an unset variable
    explicit EnvValue(const char* raw) {
        if (raw) {
            set_ = true;
            value_ = raw;
        }
    }

    /// @brief An unset value.
    static EnvValue unset() { return EnvValue(); }

    /// @brief A value that is set to `v` (which may be empty).
    static EnvValue of(std::string v) {
        EnvValue e;
        e.set_ = true;
        e.value_ = std::move(v);
        return e;
    }

    /// @brief True when the variable was set, including set to the empty string.
    bool is_set() const noexcept { return set_; }

    /// @brief The value, or `nullptr` when unset: the drop-in for `std::getenv("...")`.
    const char* get() const noexcept { return set_ ? value_.c_str() : nullptr; }

    /// @brief The value as text; empty when unset.
    const std::string& value() const noexcept { return value_; }

private:
    bool set_ = false;
    std::string value_;
};

/// @brief The route pins: `BATCHLAS_<OP>_ROUTE`, one raw string per op that selects a kernel.
///
/// Raw strings on purpose: `src/select` parses them for every op (`auto` | `native` | `vendor` |
/// a choice spelling such as `lpanel:panel=8`). Values are trimmed and case-folded; a set-but-empty
/// value means `auto`. A value the op does not understand throws `std::invalid_argument`; only
/// `native` and `vendor` warn and fall back to `auto` when nothing in that class can run the call.
/// The selection layer is described in docs/design/flat-kernel-selection.md.
/// @ingroup config
struct RoutingSettings {
    /// @brief The ops that read a route variable, by their `<op>` spelling.
    static constexpr std::array<std::string_view, 19> ops{
        "gemm", "gemv", "trsm", "trmm", "symm", "syrk", "syr2k", "potrf", "posv", "getrf",
        "getrs", "getri", "gesv", "geqrf", "orgqr", "ormqr", "syev", "gesvd", "spmm"};

    /// @brief `BATCHLAS_<OP>_ROUTE`, in `ops` order.
    std::array<EnvValue, ops.size()> values{};

    /// @brief Position of `op` in `ops`, or `ops.size()` when `op` reads no route variable.
    static constexpr std::size_t index_of(std::string_view op) {
        for (std::size_t i = 0; i < ops.size(); ++i)
            if (ops[i] == op) return i;
        return ops.size();
    }

    /// @brief The captured `BATCHLAS_<OP>_ROUTE` for `op`.
    /// @throws std::invalid_argument for an op not in `ops`
    const EnvValue& route(std::string_view op) const { return values[checked(op)]; }
    /// @brief Mutable access to the captured `BATCHLAS_<OP>_ROUTE` for `op`, as configure() needs.
    EnvValue& route(std::string_view op) { return values[checked(op)]; }

private:
    static std::size_t checked(std::string_view op) {
        const std::size_t i = index_of(op);
        if (i == ops.size())
            throw std::invalid_argument("batchlas: no BATCHLAS_<OP>_ROUTE for op '" + std::string(op) + "'");
        return i;
    }
};

/// @brief Which kernel or algorithm runs, for the knobs outside the route pins.
///
/// Every field changes which code path executes. `syevx_algorithm` and `syevx_preconditioner`
/// override an explicit SyevxParams field, and `gesvd_bidiag` changes numerics.
/// @ingroup config
// evidence: docs/design/environment.md#environment-knobs-that-override-an-explicit-argument
struct SelectionSettings {
    /// `BATCHLAS_EXPAND_ROUTE` = `expand` | `loop`: pins the scratch-expansion route of hemm, herk
    /// and her2k, so a test can reach the arm the shape would not have picked (symm and trmm pin
    /// `expand` through their `BATCHLAS_<OP>_ROUTE`). Not op-keyed, so RoutingSettings does not
    /// hold it.
    // Read at two sites (src/expansion_budget.hh, src/backends/triangular_expand.hh) by two
    // independent parsers that currently agree; both read this one field.
    EnvValue expand_route{};

    /// `BATCHLAS_GEMV_SEGT` = `off` | `auto` | `2` | `4` | `8`: segmented-tail width of the
    /// native gemv.
    // Never latch it in a static: a later setenv goes unseen and tests pass on the default arm.
    // That prohibition is why detail::reload_settings() exists.
    EnvValue gemv_segt{};

    /// `BATCHLAS_GESVD_BIDIAG` = `bdsdc` | `normal` | `bdsqr`.
    /// @warning Changes numerics: `normal` squares the condition number. The thin-tall-U rule at
    ///          the call site overrides it so a buffer-size query and its solve agree.
    // evidence: docs/perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver
    EnvValue gesvd_bidiag{};

    /// `BATCHLAS_GETRF_LEAF` = `slm` | `reg` (default: the register-resident panel leaf wherever
    /// the panel fits). Re-read per call.
    EnvValue getrf_leaf{};

    /// `BATCHLAS_GEQRF_LEAF` = `auto` (default) | `reg` (the register panel leaf wherever it
    /// fits). Re-read per call.
    // evidence: docs/perf/qr.md#qr-the-register-leaf-ab
    EnvValue geqrf_leaf{};

    /// `BATCHLAS_GETRF_LASWP` = `inloop` | `defer_walk` | `defer_gather` (default). Presence is
    /// latched at the call site, the value re-read per call.
    EnvValue getrf_laswp{};

    /// `BATCHLAS_GETRF_RIGHT_LASWP` = `walk` | `gather`; unset takes the measured crossover.
    EnvValue getrf_right_laswp{};

    /// `BATCHLAS_GETRS_LASWP` = `walk` | `gather`. Deliberately not latched (as gemv_segt).
    EnvValue getrs_laswp{};

    /// `BATCHLAS_ILUK_DEVICE`: only a first character of `0` or `1` is inspected; `true`, `on`
    /// and `yes` fall through to the shape default (batch_size >= 32).
    EnvValue iluk_device{};

    /// `BATCHLAS_LATRD_IMPL` = `legacy` | `device` | `grid`. Re-read per call.
    EnvValue latrd_impl{};

    /// `BATCHLAS_ORMQR_IMPL`: only the exact value `device` has an effect.
    EnvValue ormqr_impl{};

    /// `BATCHLAS_ORMQR_WY` = `gemm` | `trmm` | `measured`: overrides the measured gate.
    EnvValue ormqr_wy{};

    /// `BATCHLAS_ORTHO_GRAM`: only the exact value `gemm` has an effect (forces the GEMM gram
    /// route in place of the syrk one).
    EnvValue ortho_gram{};

    /// `BATCHLAS_SB2ST_BACK_WAVE`: the wave-parallel back-transform is on unless disabled with one
    /// of `0`, `false`, `off`, `no`, `n`, `disable`, `disabled` (case-insensitive).
    // Fails OPEN, with a disable set deliberately WIDER than env_falsy. Must stay raw: never
    // route it through env_falsy. evidence: docs/design/environment.md#environment-sb2st-back-wave-fails-open
    EnvValue sb2st_back_wave{};

    /// `BATCHLAS_SB2ST_SUBGROUP` = `auto` | `on` | `off` (case-folded). `on` throws when kd > 32
    /// or the device lacks sub-group size 32.
    EnvValue sb2st_subgroup{};

    /// `BATCHLAS_SYEV_TWO_STAGE_CHASE`: only the exact value `givens` has an effect.
    /// @warning Read by both the solve and its `*_buffer_size` query; do not let a ScopedEnvVar
    ///          straddle a sizing/solve pair.
    EnvValue syev_two_stage_chase{};

    /// `BATCHLAS_SYEVX_ALGORITHM`: overrides an explicit `SyevxParams::method`. An env-supplied
    /// algorithm degrades where an explicit request throws, and an unrecognised value still wins
    /// (it parses to Auto).
    EnvValue syevx_algorithm{};

    /// `BATCHLAS_SYEVX_PRECONDITIONER`: overrides the SyevxParams default; loses to an explicit
    /// request and to a configured ILU(k) factor.
    EnvValue syevx_preconditioner{};

    /// `BATCHLAS_SYEVX_BOUNDS_LEGACY`: reverts to the previous spectral-bounds computation.
    /// Parsed with `atoi != 0`, so `true`, `on` and `yes` are all false.
    EnvValue syevx_bounds_legacy{};

    /// `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO`: derive the Chebyshev filter degree automatically.
    /// Off by default because it wins only at batch 1. Parsed with `atoi != 0`.
    // evidence: docs/perf/syevx.md#syevx-automatic-chebyshev-filter-degree
    EnvValue syevx_filter_degree_auto{};

    /// `BATCHLAS_SYEVX_INSTR_HOST`: forces the host instrumentation path. First-character parser
    /// `{1,t,T,y,Y}`: `true` works, `on` does not.
    EnvValue syevx_instr_host{};

    /// `BATCHLAS_SYEVX_PROJECTED_VENDOR`: same first-character dialect.
    /// @warning Changes the workspace size: do not change it between a buffer-size query and its
    ///          call.
    EnvValue syevx_projected_vendor{};

    /// `BATCHLAS_SYEVX_SOFT_LOCK`.
    /// @warning The parser is inverted: it enables unless the first character is one of
    ///          `{0,n,N,f,F}`, so `=off` and an empty export both read as on.
    // Recorded, not fixed. evidence: docs/design/environment.md#environment-parser-defects-recorded-not-fixed
    EnvValue syevx_soft_lock{};

    /// `BATCHLAS_SYTRD_FORCE_LOCAL_SMALL` (env_truthy): forces the local-memory small-n path past
    /// the tuning heuristic. It cannot force an unschedulable launch; the device gate still applies.
    bool sytrd_force_local_small = false;

    /// `BATCHLAS_SYTRD_FUSE_PANEL_UPDATE`: forced on (env_truthy), forced off (env_falsy), or
    /// `std::nullopt` to let the tuned default decide. A value that is neither reads as nullopt.
    std::optional<bool> sytrd_fuse_panel_update{};

    /// `BATCHLAS_SYTRD_IMPL`: only the exact value `device` has an effect.
    EnvValue sytrd_impl{};

    /// `BATCHLAS_SYTRD_TRAILING_UPDATE` = `gemm` | `syr2k` | `her2k` | `rank2k`, either case;
    /// `syr2k` and `her2k` select one route.
    EnvValue sytrd_trailing_update{};

    /// `BATCHLAS_TUNED_DIR`: a directory of select tables (`src/select/select.hh`); a
    /// file there replaces the built-in table of the same name.
    EnvValue tuned_dir{};
};

/// @brief Launch geometry, block widths, iteration counts and tuning thresholds.
///
/// Where the call site's default is a function of n, the scalar type, an argument or a device
/// property, the field is a sentinel (0, `std::nullopt` or an unset EnvValue) and the default
/// stays at the call site. Fields with a constant default carry it.
/// @ingroup config
// Never materialise a function-valued default here: it pins a tuned curve at one point.
// evidence: docs/design/environment.md#environment-geometry-defaults-stay-at-the-call-site
struct GeometrySettings {
    /// `BATCHLAS_LATRD_GRID_GROUPS` (env_positive_int_or). 0 = auto,
    /// `min(residency cap, ceil((n-1)/32))`. A forced value is still clamped by the residency cap
    /// unless UnsafeSettings::latrd_grid_force_unsafe lifts it.
    int latrd_grid_groups = 0;

    /// `BATCHLAS_LATRD_GRID_MIN_N` (env_positive_int_or): the n at which the grid latrd path
    /// becomes eligible.
    // evidence: docs/perf/syev.md#syev-latrd-grid-gate-confirmed-in-eigenvector-mode
    int latrd_grid_min_n = 768;

    /// `BATCHLAS_LATRD_GRID_WG` (env_positive_int_or). 0 = computed. Only 32, 64, 128 and 256 are
    /// honoured; any other value is silently ignored.
    int latrd_grid_wg = 0;

    /// `BATCHLAS_LATRD_LOWER_PANEL_WG_HINT`. 0 = unset; only 64, 128 and 256 take effect, and only
    /// on the device path. Wins over `BATCHLAS_TUNE_LATRD_WG_HINT`.
    // Deliberately not unified. evidence: docs/design/environment.md#environment-two-variables-for-one-quantity
    int latrd_lower_panel_wg_hint = 0;

    /// `BATCHLAS_SB2ST_BACK_SUBS` (env_positive_int_or). 0 = the call site's default chain.
    /// @note Force together with sb2st_back_tile_w; only tile {1,2,4,8} x subs {4,8,16} are
    ///       instantiated.
    int sb2st_back_subs = 0;

    /// `BATCHLAS_SB2ST_BACK_TILE_W` (env_positive_int_or): column tile of the wave kernel.
    /// 0 = the tuned heuristic.
    int sb2st_back_tile_w = 0;

    /// `BATCHLAS_SB2ST_BACK_TILE`: tile of the tiled back-transform kernel (a different kernel
    /// from sb2st_back_tile_w). `0` selects the streaming kernel.
    // RAW: env_positive_int_or would silently turn =0 (streaming kernel) into "unset".
    EnvValue sb2st_back_tile{};

    /// `BATCHLAS_POTRF_NB`. 0 = unset; the default is type-dependent
    /// (128/96/96/64 for float/double/cfloat/cdouble).
    int potrf_nb = 0;
    /// `BATCHLAS_POTRF_W`. 0 = unset; the default is type-dependent (128/32/32/16).
    int potrf_w = 0;

    /// `BATCHLAS_SYEV_TWO_STAGE_KD` (env_positive_int_or): the band width, clamped to `[1, n-1]`
    /// at the call site.
    int syev_two_stage_kd = 32;

    /// `BATCHLAS_SYEV_TWO_STAGE_SB2ST_BLOCK` (env_positive_int_or). Read by four solve/sizing
    /// pairs.
    int syev_two_stage_sb2st_block = 32;

    /// `BATCHLAS_SY2SB_ORMQR_NB`: unset, `off` or `0` (never hint), or a positive forced value
    /// (clamped to kd at the call site). A value that does not parse, is negative or is above 1024
    /// reads as unset. Wins over `BATCHLAS_TUNE_SY2SB_ORMQR_NB`.
    EnvValue sy2sb_ormqr_nb{};

    /// `BATCHLAS_SYTRD_BLOCK_SIZE`. 0 = unset; the default is n-bucketed and type-dependent.
    /// Wins over `BATCHLAS_TUNE_SYTRD_BLOCK_SIZE`.
    int sytrd_block_size = 0;

    /// `BATCHLAS_TRMM_TILE_M`. 0 = unset (a function of m); bucketed to 16, 32, 64 or 128.
    int trmm_tile_m = 0;

    /// `BATCHLAS_TRSM_OUTER_NB`. 0 = unset: 128 for Side::Left, the CTA nb for Side::Right.
    int trsm_outer_nb = 0;

    /// `BATCHLAS_EXPAND_MAX_BYTES` (`strtoull`): only ever lowers the expansion ceiling, whose
    /// default is `GLOBAL_MEM_SIZE / 4`.
    EnvValue expand_max_bytes{};

    /// `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` (bare `atoi`, default 1).
    /// @warning An unparseable value reads as 0, which widens the blocked gebrd path to every n.
    // Recorded, not changed. evidence: docs/design/environment.md#environment-parser-defects-recorded-not-fixed
    EnvValue gesvd_blocked_gebrd_min{};

    /// `BATCHLAS_SYEVX_CHECK_EVERY`: iterations between convergence checks; each check is a full
    /// pipeline drain.
    int syevx_check_every = 4;

    /// `BATCHLAS_SYEVX_EXTRA_DIRECTIONS`: 0 means "no guard block"; nullopt means the default
    /// `max(2, k/4)`. A negative value reads as unset.
    std::optional<int> syevx_extra_directions{};

    /// `BATCHLAS_SYEVX_FILTER_DEGREE`. 0 = unset (only > 0 is accepted). Top of a four-level
    /// precedence chain; setting it also disables the automatic degree.
    int syevx_filter_degree = 0;

    /// `BATCHLAS_SYEVX_INIT_POWER`: 0 means "no power iterations"; nullopt means 4 unless
    /// SyevxParams supplied a value.
    std::optional<int> syevx_init_power{};

    /// `BATCHLAS_SYEVX_LOCK_FACTOR` (`atof`, > 0 only): a column is masked once its residual is
    /// this multiple of the tolerance. An unparseable value yields the default.
    // Locking exactly at tol oscillates: a masked column is not frozen and drifts back above tol.
    double syevx_lock_factor = 0.1;

    /// @brief The runtime overrides of include/batchlas/tuning_params.hh (`BATCHLAS_TUNE_*`).
    ///
    /// Parsed by `tuning::detail::tuning_env_override`, which rejects trailing garbage and
    /// values <= 0 or above INT32_MAX; each default is the n-bucketed compiled constant.
    /// @see @ref perf_tuning
    // A settings() read added to tuning_params.hh must be mirrored in the generator's template.
    // evidence: docs/perf/tuning.md#tuning-regenerating-the-header
    struct TuneSettings {
        EnvValue ormqr_block_size{};           ///< `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE`
        EnvValue gebrd_block_size{};           ///< `BATCHLAS_TUNE_GEBRD_BLOCK_SIZE`
        EnvValue sb2st_back_tile{};            ///< `BATCHLAS_TUNE_SB2ST_BACK_TILE`
        EnvValue sb2st_back_subs{};            ///< `BATCHLAS_TUNE_SB2ST_BACK_SUBS`
        EnvValue sy2sb_ormqr_nb{};             ///< `BATCHLAS_TUNE_SY2SB_ORMQR_NB`
        EnvValue sytrd_block_size{};           ///< `BATCHLAS_TUNE_SYTRD_BLOCK_SIZE`
        EnvValue latrd_wg_hint{};              ///< `BATCHLAS_TUNE_LATRD_WG_HINT`
        EnvValue stedc_recursion_threshold{};  ///< `BATCHLAS_TUNE_STEDC_RECURSION_THRESHOLD`

        /// `BATCHLAS_TUNE_STEDC_MERGE_VARIANT`: a StedcMergeVariant value; 0 (Auto) is unreachable.
        // An enum smuggled through an int, cast with no range check; one variant is known to
        // deadlock on some hardware. evidence: docs/design/environment.md#environment-parser-defects-recorded-not-fixed
        EnvValue stedc_merge_variant{};

        EnvValue stedc_threads_per_root{};     ///< `BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT`
        EnvValue stedc_wg_multiplier{};        ///< `BATCHLAS_TUNE_STEDC_WG_MULTIPLIER`
    } tune{};  ///< The eleven `BATCHLAS_TUNE_*` overrides.
};

/// @brief Tracing, dumping, profiling and the opt-in checks. None changes a numeric result.
///
/// Three of these name a filesystem path the library opens for writing (the kernel trace and
/// the coverage output from `atexit` handlers, the band-reduction dump via
/// `create_directories()`); an embedding application can clear them with configure().
/// @ingroup config
// evidence: docs/design/environment.md#environment-files-written-from-the-environment
struct DiagnosticsSettings {
    /// `BATCHLAS_QUEUE_PROFILING` or `BATCHLAS_BENCH_PROFILING` (env_truthy, ORed): enable SYCL
    /// queue profiling. Opt-in because it costs on every submit; the kernel trace implies it.
    bool profiling = false;

    /// `BATCHLAS_KERNEL_TRACE` or `BATCHLAS_TRACE_KERNELS` (env_truthy, ORed).
    bool kernel_trace = false;

    /// The first non-empty of `BATCHLAS_KERNEL_TRACE_PATH` and `BATCHLAS_TRACE_PATH`. Written
    /// from an `atexit` handler.
    std::string kernel_trace_path = "batchlas_kernels.trace.json";

    /// `BATCHLAS_COVERAGE_OUT`: dispatch coverage is written to `<value>.<pid>` at exit.
    EnvValue coverage_out{};

    /// `BATCHLAS_SELECT_TRACE` (env_truthy): one stderr line per `select::choose` decision
    /// (`src/select/select.hh`).
    bool select_trace = false;

    /// `BATCHLAS_DEBUG_FILTER_DEGREE`: presence only, so any value including the empty string
    /// enables it. Prints one line per batch item per iteration.
    bool debug_filter_degree = false;

    /// `BATCHLAS_DEBUG_SYTRD_SMALL` (env_truthy). Prints once.
    bool debug_sytrd_small = false;

    /// `BATCHLAS_GESVD_PROFILE` (env_truthy): forces a `wait_and_throw()` per stage, serialising
    /// the pipeline, and prints CSV rows to stderr.
    bool gesvd_profile = false;

    /// `BATCHLAS_SYEVX_TRACE` (env_truthy).
    bool syevx_trace = false;

    /// `BATCHLAS_CTA_DEBUG_SYNC` (env_truthy): prints `[cta-debug] <stage>` and drains the
    /// pipeline at each stage, so an async error surfaces at a named stage.
    // Not "unsafe": it removes no check. evidence: docs/design/environment.md#environment-two-knobs-that-look-unsafe-and-are-not
    bool cta_debug_sync = false;

    /// `BATCHLAS_STEQR_CTA_CHECK` (first character `{1,t,T,y,Y}`): steqr_cta scans its per-item
    /// status and throws if any item did not converge.
    /// @warning Unset, non-convergence in steqr_cta is silent.
    // ADDS a check, so it is not "unsafe"; forcing it to its default would entrench the silence.
    EnvValue steqr_cta_check{};

    /// @brief The `BATCHLAS_DUMP_BANDR1_*` band-reduction dump family. The four index selectors
    /// (env_int_or) default to -1, "no filter"; a non-negative value dumps only that index.
    struct DumpBandR1Settings {
        /// `BATCHLAS_DUMP_BANDR1_DIR`: created with `create_directories()`; files are written
        /// under it while `step` is on.
        std::string dir = "output/bandr1_dumps";

        /// `BATCHLAS_DUMP_BANDR1_STEP` (env_truthy): the master enable.
        bool step = false;

        /// `BATCHLAS_DUMP_BANDR1_ABW_ONLY` (env_truthy): suppress every dump except ABw.
        bool abw_only = false;

        int step_index = -1;     ///< `BATCHLAS_DUMP_BANDR1_STEP_INDEX`
        int sweep_index = -1;    ///< `BATCHLAS_DUMP_BANDR1_SWEEP_INDEX`
        int step_in_sweep = -1;  ///< `BATCHLAS_DUMP_BANDR1_STEP_IN_SWEEP`
        int batch = -1;          ///< `BATCHLAS_DUMP_BANDR1_BATCH`; -1 dumps every item
    } dump_bandr1{};  ///< The band-reduction dump family.
};

/// @brief The overrides that remove a guarantee: setting one can make a correct program crash,
/// hang or silently compute wrong numbers.
///
/// Unless the library was built with the CMake option `BATCHLAS_ALLOW_UNSAFE_ENV=ON` (default
/// OFF), the environment cannot move these fields in their unsafe direction: they keep their safe
/// values and each such variable produces one warning on stderr. configure() is not gated.
/// @ingroup config
// evidence: docs/design/environment.md#environment-the-unsafe-group-and-its-gate
struct UnsafeSettings {
    /// `BATCHLAS_SKIP_POINTER_CHECKS`: disables the per-argument USM reachability check that turns
    /// host memory passed to a device call into an invalid_argument. Without it such a call dies
    /// in the CUDA runtime with no catchable error.
    /// @warning Any non-empty value whose first character is not `0` disables the checks, so
    ///          `=false`, `=off` and `=no` all turn them off.
    // Parser !(v && *v && *v != '0') preserved character for character on purpose.
    bool skip_pointer_checks = false;

    /// `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` (env_truthy): lets `BATCHLAS_LATRD_GRID_GROUPS` exceed
    /// the co-residency cap.
    /// @warning The grid barrier is a spin barrier that relies on co-residency: the kernel can
    ///          hang (it looks like slow JIT). Run forced measurements under `timeout`.
    bool latrd_grid_force_unsafe = false;

    /// @brief Behaviour of the host-dgemm health probe (`BATCHLAS_BLAS_HEALTH`).
    enum class BlasHealth {
        Off,   ///< `off`: skip the probe; a broken host BLAS then gives silently wrong double results. Gated.
        Warn,  ///< `warn` (default): warn when the probe fails.
        Error  ///< `error`: throw when the probe fails. Stricter than the default, so not gated.
    };
    /// `BATCHLAS_BLAS_HEALTH` = `off` | `warn` | `error`.
    BlasHealth blas_health = BlasHealth::Warn;
};

/// @brief Everything BatchLAS reads from its environment, in one value.
///
/// Copyable on purpose; to change one field programmatically:
/// @code
/// auto s = batchlas::settings();
/// s.diagnostics.dump_bandr1.step = false;
/// batchlas::configure(s);
/// @endcode
/// @ingroup config
struct Settings {
    RoutingSettings routing{};          ///< Route pins (`BATCHLAS_<OP>_ROUTE`).
    SelectionSettings selection{};      ///< Kernel and algorithm selection.
    GeometrySettings geometry{};        ///< Launch geometry and tuning thresholds.
    DiagnosticsSettings diagnostics{};  ///< Tracing, dumping, profiling, opt-in checks.
    UnsafeSettings unsafe{};            ///< Overrides that remove a guarantee.
};

/// @brief The process-wide settings.
///
/// The environment is read once, under `std::call_once`, on the first call. Thread-safe, and
/// safe to call from a static initialiser.
/// @return a reference valid for the life of the process. Its contents change under configure()
///         and detail::reload_settings(), so do not cache fields across a call that could do
///         either, and never hoist a field into a function-local static.
/// @ingroup config
BATCHLAS_API const Settings& settings();

/// @brief Install `s` as the settings, for an embedding application that wants to stop
/// inheriting ambient process state.
///
/// What is installed is the base, not a lock: a later detail::reload_settings() (every
/// ScopedEnvVar triggers one) re-reads the environment on top of it, so a knob set here and not
/// exported holds for the life of the process, and a knob something explicitly exports still
/// wins. The unsafe group is not gated here.
/// @param s the complete settings to install
/// @pre No batchlas::Queue has been constructed yet in this process.
/// @throws batchlas::api_misuse (a `std::runtime_error`) if a Queue already exists; nothing is
///         changed.
/// @ingroup config
// evidence: docs/design/environment.md#environment-configure-sets-the-base-not-a-lock
BATCHLAS_API void configure(const Settings& s);

namespace detail {

/// @brief Re-read the environment on top of the last configure() (or the defaults).
///
/// ScopedEnvVar calls this from its constructor and destructor, which is how a test's pinned
/// knob takes effect and then stops leaking into later tests.
/// @warning A raw `::setenv` does not reach it. A reload between a `*_buffer_size()` query and
///          its solve can under-size the workspace. Not thread-safe with respect to concurrent
///          settings() readers: call it from a test body or benchmark setup only.
/// @ingroup config
// evidence: docs/design/environment.md#environment-what-a-settings-reload-does-not-cover
BATCHLAS_API void reload_settings();

/// @brief Latch recording that a Queue exists, which closes configure(). Called only by Queue's
/// constructors.
/// @ingroup config
BATCHLAS_API void note_queue_constructed() noexcept;

/// @brief True once note_queue_constructed() has been called.
/// @ingroup config
BATCHLAS_API bool queue_constructed() noexcept;

}  // namespace detail

}  // namespace batchlas
