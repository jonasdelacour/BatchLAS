#pragma once

// One place where BatchLAS reads its environment, and one place an embedding
// application can override or lock down what it read.
//
// THE DEFECT THIS CLOSES (audit finding A-3). Before this header, ~106 distinct
// BATCHLAS_* environment variables were read by ~77 std::getenv calls scattered
// through src/ and include/. They steer routing, kernel selection, launch
// geometry, diagnostics and -- in three cases -- whether arguments are validated
// at all. Ambient process state silently decided which kernel ran. There was no
// programmatic equivalent, no allow-list, and no way for a host application to
// stop inheriting a variable somebody exported for a benchmark last week. The
// debugging value of those knobs is real and none of them is removed here; what
// changes is that the READ happens once, in one place, and is overridable.
//
// THE ONE RULE THAT SHAPED THIS FILE: move where the STRING comes from, not how
// it is parsed. The tree contains SEVEN mutually incompatible boolean dialects
// (env_truthy's exact {1,true,TRUE,on,ON}; a case-folding variant that also
// accepts "yes"; two first-character forms, {1,t,T,y,Y} and one that is
// INVERTED; an atoi!=0 form for which "true" is false; a form that only looks at
// '0'/'1'; and sytrd_sb2st_hh's deliberately wider disable set) and three
// integer readers that disagree about trailing garbage. Normalising them would
// change the reading of real spellings at real call sites, which is exactly the
// class of silent behaviour change this repository has been bitten by before.
// So: a knob whose call site uses one of the shared parsers in
// <batchlas/util/env.hh> gets a TYPED field here (settings.cc calls that same
// parser, so the value is bit-for-bit what the site computed before); a knob
// with a bespoke parser gets an EnvValue, which is the raw captured string, and
// its parser stays where it is. EnvValue::get() is deliberately getenv-shaped --
// it returns nullptr for an unset variable -- so migrating such a site is
// replacing std::getenv("BATCHLAS_FOO") with settings().group.foo.get() and
// nothing else.
//
// KEEPING TESTS GREEN. settings() reads the environment once under
// std::call_once. Fifteen test files mutate the environment through
// batchlas::ScopedEnvVar and expect the library to notice, so ScopedEnvVar's
// constructor AND destructor call detail::reload_settings(), which re-runs the
// load. See the note on reload_settings() for what that does not cover.

#include <batchlas/export.hh>
#include <array>
#include <cstddef>
#include <optional>
#include <string>

// For batchlas::dispatch::Op, which keys the per-op route arrays. route.hh is
// enums plus <string>, no SYCL: this header is INSTALLED and must stay out of
// <sycl/sycl.hpp>'s way. See the note at the top of blas/linalg.hh -- pulling
// SYCL into a public header costs ~4.1 s per consumer translation unit, and
// route_env.hh (which will include this one) is reached by every device TU.
#include <batchlas/blas/dispatch/route.hh>

namespace batchlas {

// A raw captured environment value, shaped like the std::getenv call it replaces.
//
// The distinction between "unset" and "set to the empty string" is load-bearing
// at several sites: kernel-trace.hh falls through from BATCHLAS_KERNEL_TRACE_PATH
// to BATCHLAS_TRACE_PATH only when the first is set-but-empty, and
// parse_route_env treats a set-but-empty canonical route variable as absent so
// the legacy spelling still gets a turn. A std::optional<std::string> would
// carry that too; this type exists to also hand back a const char* so the
// migrated call site keeps the parser it already has.
class EnvValue {
public:
    EnvValue() = default;

    // Construct from what std::getenv returned: nullptr means unset.
    explicit EnvValue(const char* raw) {
        if (raw) {
            set_ = true;
            value_ = raw;
        }
    }

    static EnvValue unset() { return EnvValue(); }

    static EnvValue of(std::string v) {
        EnvValue e;
        e.set_ = true;
        e.value_ = std::move(v);
        return e;
    }

    bool is_set() const noexcept { return set_; }

    // nullptr when unset -- the drop-in for std::getenv("...") at a call site
    // that keeps its own parser.
    const char* get() const noexcept { return set_ ? value_.c_str() : nullptr; }

    // Empty when unset. Callers that only care about the text use this.
    const std::string& value() const noexcept { return value_; }

private:
    bool set_ = false;
    std::string value_;
};

// ---------------------------------------------------------------------------
// routing -- the route vocabulary, BATCHLAS_<OP>_ROUTE and its legacy spellings.
//
// These are RAW STRINGS on purpose. dispatch::parse_route_env(Op) is the single
// route parser: it handles three documented word collisions (legacy
// BATCHLAS_GEMM_VARIANT=native selects the VENDOR path, the opposite of
// canonical "native"; legacy `custom` means the fused cuBLASDx kernel for the
// level-3 tile ops and the register-tiled family for gemm; syrk/syr2k legacy
// `gemm` selects a deliberately WRONG both-triangles baseline), and
// tests/route_vocabulary_tests.cc pins every one of them. Nothing here
// reimplements any of that -- parse_route_env keeps its parsers and loses only
// its two std::getenv calls.
struct RoutingSettings {
    // BATCHLAS_<OP>_ROUTE, indexed by dispatch::Op.
    //
    // FOUR SLOTS ARE INERT and will stay so until somebody wires them: hemm,
    // herk, her2k and iluk have no parse_route_env call site anywhere in src/,
    // include/, tests/ or benchmarks/ -- Op::hemm/herk/her2k appear only as
    // coverage labels. op_env_stem() can spell BATCHLAS_HEMM_ROUTE, and this
    // array will faithfully capture it, but no adapter reads it. The array is
    // indexed by Op rather than listing 17 named fields precisely so that
    // wiring one later is a one-line change at the adapter and not here; do not
    // read "there is a slot" as "the variable works".
    std::array<EnvValue, static_cast<std::size_t>(dispatch::Op::COUNT)> canonical{};

    // The legacy per-op spellings: BATCHLAS_{GEMM,SYMM,SYRK,SYR2K,TRMM}_VARIANT
    // and BATCHLAS_{SYEV,GESVD,ORMQR}_PROVIDER. Benchmark scripts and recorded
    // results use them, so they must keep working. The canonical spelling wins
    // when both are set (pinned: RouteVocabulary.CanonicalSpellingWinsOverLegacy).
    //
    // NOTE: legacy[Op::gemm] is BATCHLAS_GEMM_VARIANT, which has TWO readers
    // with two vocabularies and two different unset defaults --
    // parse_route_env(Op::gemm) defaults to {Auto,Auto}, and
    // gemm_variant_request() in src/backends/gemm_variant.hh defaults to
    // GemmVariantRequest::Vendor. Both now read this one field, so they can no
    // longer disagree about what the user typed; they still keep their own
    // defaults for the unset case, which is deliberate -- unifying those is a
    // behaviour change (it moves which kernel a bare gemm() call runs) and is
    // not part of this work.
    std::array<EnvValue, static_cast<std::size_t>(dispatch::Op::COUNT)> legacy{};

    const EnvValue& canonical_route(dispatch::Op op) const {
        return canonical[static_cast<std::size_t>(op)];
    }

    const EnvValue& legacy_route(dispatch::Op op) const {
        return legacy[static_cast<std::size_t>(op)];
    }
};

// ---------------------------------------------------------------------------
// selection -- which kernel or algorithm runs, for the knobs that are NOT part
// of the route vocabulary.
//
// A fifth group, added because the four in the design have no home for these and
// dropping them would leave the audit finding open. Every field here changes
// which code path executes; three of them override an explicit API argument,
// which is the strongest form of the A-3 defect (syevx_algorithm and
// syevx_preconditioner override SyevxParams; gesvd_bidiag changes NUMERICS, not
// just speed -- its "normal" arm squares the condition number).
struct SelectionSettings {
    // BATCHLAS_EXPAND_ROUTE = expand | loop. Pins the scratch-expansion route so
    // a test can reach the arm the shape would not have picked. Read at two
    // sites (src/expansion_budget.hh and src/backends/triangular_expand.hh) with
    // two independent parsers that currently agree; both now read this one
    // field. Not op-keyed, so parse_route_env never sees it.
    EnvValue expand_route{};

    // BATCHLAS_GEMM_CUBLASDX_KERNEL. Kernel selection INSIDE the vendor route;
    // ~20 accepted spellings, unset means CuBLASDxGemmVariant::VendorFallback.
    // Two reads in one file, one asking "is it set" and one asking "what does it
    // say"; both now read this field, so they cannot see different answers.
    EnvValue gemm_cublasdx_kernel{};

    // BATCHLAS_GEMM_EXPERIMENTAL. Unlocks five GEMM kernel variants marked
    // experimental. Its parser case-folds and also accepts "yes", which
    // env_truthy does not -- hence raw.
    EnvValue gemm_experimental{};

    // BATCHLAS_GEMM_SYCL_KERNEL. Forces one named register-tiled GEMM kernel
    // (~38 accepted spellings); unset means KernelVariant::Direct. Two reads,
    // presence and value.
    EnvValue gemm_sycl_kernel{};

    // BATCHLAS_GEMV_SEGT = off | auto | 2 | 4 | 8. Segmented-tail width for the
    // native gemv. Its call site carries an explicit prohibition on latching the
    // getenv in a function-local static, because a cached read makes a later
    // setenv invisible and the test then passes green on the DEFAULT arm. That
    // prohibition is why detail::reload_settings() exists.
    EnvValue gemv_segt{};

    // BATCHLAS_GESVD_BIDIAG = bdsdc | normal | bdsqr. Changes NUMERICS: the
    // "normal" arm squares the condition number (measured relative error 8.5e-1
    // against 5.0e-1 at kappa 1e6). Deliberately overridden by the thin-tall-U
    // rule at the call site so a buffer-size query and its solve cannot disagree.
    EnvValue gesvd_bidiag{};

    // BATCHLAS_GETRF_LASWP = inloop | defer_walk | defer_gather (default
    // DeferGather). Its call site is deliberately hybrid -- presence latched in
    // a static, value re-read per call, so a harness can swap arms mid-run.
    EnvValue getrf_laswp{};

    // BATCHLAS_GETRS_LASWP = walk | gather. Deliberately NOT latched at its call
    // site, for the same reason as gemv_segt.
    EnvValue getrs_laswp{};

    // BATCHLAS_ILUK_DEVICE. Only the first character is inspected, and only
    // against '0' and '1': "true", "on" and "yes" all fall through to the shape
    // default (batch_size >= 32). Narrower than a reader would guess, which is
    // exactly why the parser stays at the call site.
    EnvValue iluk_device{};

    // BATCHLAS_LATRD_IMPL = legacy | device | grid. Deliberately re-read on
    // every call so one process can A/B the implementations.
    EnvValue latrd_impl{};

    // BATCHLAS_ORMQR_IMPL. Only the exact value "device" has an effect.
    EnvValue ormqr_impl{};

    // BATCHLAS_ORMQR_WY = gemm | trmm | measured. Overrides the measured gate.
    EnvValue ormqr_wy{};

    // BATCHLAS_ORTHO_GRAM. Only the exact value "gemm" has an effect; it forces
    // the GEMM route in place of the syrk gram-tile one where that exists.
    EnvValue ortho_gram{};

    // BATCHLAS_SB2ST_BACK_WAVE. THE knob env.hh warns about: it FAILS OPEN (the
    // wave-parallel back-transform is on unless explicitly disabled) and it
    // accepts a deliberately WIDER disable set than env_falsy --
    // {0,false,off,no,n,disable,disabled}, case-insensitively. env_falsy's exact
    // spellings silently turned the wave path ON for "False"/"Off"/"no", which
    // is the regression that made the local helper exist. Raw, and it must stay
    // raw: do not route this through env_falsy.
    EnvValue sb2st_back_wave{};

    // BATCHLAS_SB2ST_SUBGROUP = auto | on | off (case-folded). ForceOn THROWS
    // when kd > 32 or the device lacks sub_group_size 32, so the three states
    // are genuinely distinct and a bool would lose one.
    EnvValue sb2st_subgroup{};

    // BATCHLAS_SYEV_SMALL_KERNEL = cta | fused | cta_fused | jacobi. Its parser
    // returns a `forced` out-parameter so the caller can tell "set to cta" from
    // "unset"; is_set() is what preserves that distinction.
    EnvValue syev_small_kernel{};

    // BATCHLAS_SYEV_TWO_STAGE_CHASE. Only the exact value "givens" has an
    // effect. Read by BOTH the solve and its *_buffer_size query in two callers,
    // so the two must see the same answer -- one read here guarantees that
    // within a run; do not let a ScopedEnvVar straddle a sizing/solve pair.
    EnvValue syev_two_stage_chase{};

    // BATCHLAS_SYEVX_ALGORITHM. Overrides an explicit SyevxParams::method. The
    // asymmetry at the call site is load-bearing: an env-supplied algorithm
    // DEGRADES where an explicit request THROWS, and an unrecognised value still
    // wins (it parses to Auto).
    EnvValue syevx_algorithm{};

    // BATCHLAS_SYEVX_PRECONDITIONER. Overrides an explicit SyevxParams field;
    // loses to an explicit request and to a configured ILU(k) factor. Documented
    // in the PUBLIC headers (blas/enums.hh, blas/extensions.hh).
    EnvValue syevx_preconditioner{};

    // BATCHLAS_SYEVX_BOUNDS_LEGACY. Reverts to the previous spectral-bounds
    // computation. Parsed with atoi != 0, so "true"/"on"/"yes" are all FALSE.
    EnvValue syevx_bounds_legacy{};

    // BATCHLAS_SYEVX_FILTER_DEGREE_AUTO. Cannot ship on: measured 2.2x faster at
    // batch 1 and 0.48x at batch >= 8. atoi != 0 again.
    EnvValue syevx_filter_degree_auto{};

    // BATCHLAS_SYEVX_INSTR_HOST. A/B hatch forcing the host instrumentation
    // path. First-character parser {1,t,T,y,Y}: "true" works, "on" does NOT.
    EnvValue syevx_instr_host{};

    // BATCHLAS_SYEVX_PROJECTED_VENDOR. Same first-character dialect. It changes
    // the WORKSPACE SIZE, so it must not change between a buffer-size query and
    // its call.
    EnvValue syevx_projected_vendor{};

    // BATCHLAS_SYEVX_SOFT_LOCK. Its parser is INVERTED -- it enables unless the
    // first character is one of {0,n,N,f,F} -- so `=off` evaluates TRUE, and so
    // does an empty export. That is a latent defect, recorded rather than fixed
    // here: fixing it is a behaviour change and belongs in its own commit, and
    // silently normalising it while moving the read is exactly what this file
    // refuses to do.
    EnvValue syevx_soft_lock{};

    // BATCHLAS_SYTRD_FORCE_LOCAL_SMALL, via env_truthy. Cannot force an
    // unschedulable launch: the local-memory small-n path is still gated on the
    // device properties, so what this forces past is the tuning heuristic.
    bool sytrd_force_local_small = false;

    // BATCHLAS_SYTRD_FUSE_PANEL_UPDATE -- genuinely TRI-STATE, and the one knob
    // that makes env.hh's "an unset variable is neither truthy nor falsy"
    // contract load-bearing. Its call site reads the same value through
    // env_truthy AND env_falsy and needs all three answers: forced on, forced
    // off, and "let tuning::sytrd_fuse_panel_update_for_n(n) (device path) or
    // the legacy (CUDA && n == 256) rule decide". std::nullopt is that third
    // state; a plain bool would silently pin the tuned default.
    //
    // A value that is neither truthy nor falsy ("banana") reads as nullopt,
    // which is exactly what the call site computed before.
    std::optional<bool> sytrd_fuse_panel_update{};

    // BATCHLAS_SYTRD_IMPL. Only the exact value "device" has an effect. Latched
    // at its call site today; its near-twin latrd_impl deliberately is not.
    EnvValue sytrd_impl{};

    // BATCHLAS_SYTRD_TRAILING_UPDATE = gemm | syr2k | her2k | rank2k, each
    // accepted in upper and lower case. syr2k and her2k map to one route on
    // purpose.
    EnvValue sytrd_trailing_update{};
};

// ---------------------------------------------------------------------------
// geometry -- launch geometry and tuning thresholds.
//
// TWENTY OF THESE CANNOT CARRY A SCALAR DEFAULT, because their default is a
// FUNCTION: n-bucketed (every BATCHLAS_TUNE_*, and the sb2st/sytrd/latrd block
// widths), type-dependent (potrf's nb and w, syev's cta max n), argument-
// dependent (trsm's outer nb depends on Side, trmm's tile on m, syevx's extra
// directions on the eigenvalue count) or device-dependent (the expansion byte
// budget is GLOBAL_MEM_SIZE/4). For those the field is a SENTINEL -- 0, or an
// unset EnvValue -- and the default STAYS AT THE CALL SITE. Materialising a
// scalar here would pin a tuned curve at one point, which is precisely the shape
// of the STEDC leaf-cliff and the float-only-tuning defects already in this
// repository's history. Fields whose call-site default really is a constant
// carry it.
struct GeometrySettings {
    // --- latrd grid path -------------------------------------------------

    // BATCHLAS_LATRD_GRID_GROUPS, via env_positive_int_or. 0 = auto, i.e.
    // min(residency cap, ceil((n-1)/32)). A forced value is still CLAMPED by the
    // residency cap unless unsafe.latrd_grid_force_unsafe lifts it.
    int latrd_grid_groups = 0;

    // BATCHLAS_LATRD_GRID_MIN_N, via env_positive_int_or. The n at which the
    // grid latrd path becomes eligible. The ~50 lines of measured evidence for
    // 768 live at the call site and stay there.
    int latrd_grid_min_n = 768;

    // BATCHLAS_LATRD_GRID_WG, via env_positive_int_or. 0 = the computed
    // work-group size. Only {32,64,128,256} are honoured; any other positive
    // value is read and then silently ignored by the call site.
    int latrd_grid_wg = 0;

    // BATCHLAS_LATRD_LOWER_PANEL_WG_HINT. 0 = unset. Bare atoi at the call site,
    // where only {64,128,256} take effect; env_positive_int_or agrees with atoi
    // on every input that reaches a non-zero result here.
    //
    // TWO VARIABLES FOR ONE QUANTITY: this one stacks above
    // BATCHLAS_TUNE_LATRD_WG_HINT (see tune.latrd_wg_hint), which accepts any
    // positive int. They are NOT unified, because the call site honours this one
    // only on the device path while the legacy path asks tuning:: with n pinned
    // to 256 -- unifying would make the outer knob live on a path it has never
    // affected.
    int latrd_lower_panel_wg_hint = 0;

    // --- sb2st back-transform --------------------------------------------

    // BATCHLAS_SB2ST_BACK_SUBS, via env_positive_int_or. 0 = the four-level
    // default chain at the call site. Paired with sb2st_back_tile_w: forcing
    // only one of the pair produces a geometry that was never measured, and only
    // tile in {1,2,4,8} x subs in {4,8,16} are actually instantiated.
    int sb2st_back_subs = 0;

    // BATCHLAS_SB2ST_BACK_TILE_W, via env_positive_int_or. 0 = the tuned
    // heuristic. This is the column tile for the WAVE kernel.
    int sb2st_back_tile_w = 0;

    // BATCHLAS_SB2ST_BACK_TILE -- a DIFFERENT kernel (the tiled one), not an
    // abbreviation of the field above, and RAW because 0 IS A MEANINGFUL VALUE:
    // the call site jumps to the streaming kernel on 0. env_positive_int_or maps
    // 0 to "unset", so giving this the same reader as its sibling one screen
    // above would silently stop BATCHLAS_SB2ST_BACK_TILE=0 selecting the
    // streaming kernel -- a route change with no diagnostic.
    EnvValue sb2st_back_tile{};

    // --- potrf -----------------------------------------------------------

    // BATCHLAS_POTRF_NB and BATCHLAS_POTRF_W. 0 = unset; the defaults are
    // TYPE-DEPENDENT (nb 128/96/96/64 and w 128/32/32/16 for
    // float/double/cfloat/cdouble) and stay at the call site. Deliberately
    // latched there today, with the reason written down: "read once so the
    // sizing query and the call agree". settings() makes that property
    // structural rather than per-site.
    int potrf_nb = 0;
    int potrf_w = 0;

    // --- two-stage syev --------------------------------------------------

    // BATCHLAS_SYEV_TWO_STAGE_KD, via env_positive_int_or. The band width; 32 is
    // the call site's constant default and is carried here. The call site still
    // clamps the result to [1, n-1].
    int syev_two_stage_kd = 32;

    // BATCHLAS_SYEV_TWO_STAGE_SB2ST_BLOCK, via env_positive_int_or. Read at four
    // call sites, each pairing a solve with its buffer-size query.
    int syev_two_stage_sb2st_block = 32;

    // BATCHLAS_SY2SB_ORMQR_NB -- RAW, because it is THREE-VALUED: unset, the
    // spelling "off" (or 0) meaning "never hint", and a positive forced value
    // clamped to 0..1024. A plain int cannot represent that. Stacks above
    // tune.sy2sb_ormqr_nb.
    EnvValue sy2sb_ormqr_nb{};

    // BATCHLAS_SYEV_CTA_MAX_N -- RAW. Its parser is strtol with an explicit
    // reject outside 0..32, and its default is TYPE-DEPENDENT (24 for
    // complex<double>, 32 otherwise). 32 means "off"; lowering it speeds up
    // LOBPCG's projected solve but flips a marginal case in
    // ILUKTests.SyevxInstrumentationAndPreconditioner, which is why it is opt-in.
    EnvValue syev_cta_max_n{};

    // BATCHLAS_SYTRD_BLOCK_SIZE. 0 = unset. The default is both n-bucketed and
    // type-dependent (with a complex override for 256 < n <= 512 that
    // deliberately lives at the consumer rather than in the generated header),
    // so it stays at the call site. Stacks above tune.sytrd_block_size.
    int sytrd_block_size = 0;

    // --- level-3 tiles ---------------------------------------------------

    // BATCHLAS_TRMM_TILE_M. 0 = unset; the default is a function of m. The call
    // site buckets the value to {16,32,64,128}, so any other value is rounded.
    int trmm_tile_m = 0;

    // BATCHLAS_TRSM_OUTER_NB. 0 = unset; the default is SIDE-dependent (128 for
    // Side::Left, the CTA nb for Side::Right). Deliberately not latched at its
    // call site -- see detail::reload_settings().
    int trsm_outer_nb = 0;

    // BATCHLAS_EXPAND_MAX_BYTES -- RAW. Parsed with strtoull at the call site,
    // and it only ever LOWERS a ceiling whose default is a DEVICE property
    // (GLOBAL_MEM_SIZE/4), so there is no scalar default to carry.
    EnvValue expand_max_bytes{};

    // BATCHLAS_GESVD_BLOCKED_GEBRD_MIN -- RAW, and this one inverts. The call
    // site's default is 1 and it parses with bare atoi, so an UNPARSEABLE value
    // yields 0, which is LOWER than the default and therefore silently WIDENS
    // the blocked-gebrd path to every n instead of falling back. Routing it
    // through env_int_or would change that to 1. Recorded, not changed.
    EnvValue gesvd_blocked_gebrd_min{};

    // --- syevx / LOBPCG --------------------------------------------------

    // BATCHLAS_SYEVX_CHECK_EVERY. Iterations between convergence checks; each
    // check is a full pipeline drain. 4 is the call site's constant default.
    int syevx_check_every = 4;

    // BATCHLAS_SYEVX_EXTRA_DIRECTIONS -- optional, because 0 IS MEANINGFUL ("no
    // guard block") and distinct from unset, and the default depends on the
    // requested eigenvalue count (max(2, k/4)). A negative value is rejected by
    // the call site's parser and reads as unset here, matching it.
    std::optional<int> syevx_extra_directions{};

    // BATCHLAS_SYEVX_FILTER_DEGREE. 0 = unset (the call site's parser accepts
    // only > 0). Top of a four-level precedence chain; setting it also disables
    // the auto-derivation of the degree.
    int syevx_filter_degree = 0;

    // BATCHLAS_SYEVX_INIT_POWER -- optional, because 0 IS MEANINGFUL ("no power
    // iterations") and the default of 4 applies only when SyevxParams did not
    // supply one.
    std::optional<int> syevx_init_power{};

    // BATCHLAS_SYEVX_LOCK_FACTOR. The only non-integer numeric knob in the tree.
    // A column is masked once its residual is this multiple of the requested
    // tolerance; locking exactly at tol oscillates, because a masked column is
    // not frozen and drifts back above tol. Parsed with atof at the call site,
    // which accepts only > 0.0, so 0.1 is both the default and what an
    // unparseable value yields.
    double syevx_lock_factor = 0.1;

    // --- the generated tuning header's runtime overrides -----------------
    //
    // include/batchlas/tuning_params.hh is GENERATED, and its own comment says
    // the accessors at the bottom are hand-maintained AND MIRRORED IN THE
    // GENERATOR'S TEMPLATE. Any settings() read introduced into those accessors
    // must also land in evaluation/tuning/generate_tuning_header.py, or the next
    // retune (~12 minutes, so it will happen) silently reverts this family.
    //
    // Raw, because tuning_env_override is stricter than env.hh's readers: strtol
    // with an explicit reject of trailing garbage and of anything <= 0 or above
    // INT32_MAX, where env_int_or's stoi-in-a-try accepts "16x" as 16. Every
    // field's default is the n-bucketed compiled constant at the call site.
    struct TuneSettings {
        EnvValue ormqr_block_size{};           // BATCHLAS_TUNE_ORMQR_BLOCK_SIZE
        EnvValue gebrd_block_size{};           // BATCHLAS_TUNE_GEBRD_BLOCK_SIZE
        EnvValue sb2st_back_tile{};            // BATCHLAS_TUNE_SB2ST_BACK_TILE
        EnvValue sb2st_back_subs{};            // BATCHLAS_TUNE_SB2ST_BACK_SUBS
        EnvValue sy2sb_ormqr_nb{};             // BATCHLAS_TUNE_SY2SB_ORMQR_NB
        EnvValue sytrd_block_size{};           // BATCHLAS_TUNE_SYTRD_BLOCK_SIZE
        EnvValue latrd_wg_hint{};              // BATCHLAS_TUNE_LATRD_WG_HINT
        EnvValue stedc_recursion_threshold{};  // BATCHLAS_TUNE_STEDC_RECURSION_THRESHOLD

        // An ENUM smuggled through an int knob: the call site static_casts it to
        // StedcMergeVariant with no range check, and tuning_env_override rejects
        // <= 0, so variant 0 is unreachable from the environment. One merge
        // variant is known to deadlock on some hardware.
        EnvValue stedc_merge_variant{};        // BATCHLAS_TUNE_STEDC_MERGE_VARIANT

        EnvValue stedc_threads_per_root{};     // BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT
        EnvValue stedc_wg_multiplier{};        // BATCHLAS_TUNE_STEDC_WG_MULTIPLIER
    } tune{};
};

// ---------------------------------------------------------------------------
// diagnostics -- tracing, dumping, profiling and the opt-in checks.
//
// THREE OF THESE OPEN A FILESYSTEM PATH FOR WRITING inside library code, two of
// them from an atexit handler: the kernel trace writes trace_path(), the
// dispatch coverage writes $BATCHLAS_COVERAGE_OUT + "." + getpid(), and the
// band-reduction dump calls create_directories() under $BATCHLAS_DUMP_BANDR1_DIR.
// An embedding application that inherits a hostile environment gets directories
// created and files written at a path it never chose. configure() is what lets
// such an application clear these before it starts work.
struct DiagnosticsSettings {
    // BATCHLAS_QUEUE_PROFILING or BATCHLAS_BENCH_PROFILING, via env_truthy.
    // TWO NAMES, ONE FIELD: the call site ORs them, so the field is the OR.
    // Profiling is opt-in because it costs on every submit in a non-benchmark
    // run; the kernel trace implies it.
    bool profiling = false;

    // BATCHLAS_KERNEL_TRACE or BATCHLAS_TRACE_KERNELS, via env_truthy. Again two
    // names ORed into one field.
    bool kernel_trace = false;

    // The first NON-EMPTY of BATCHLAS_KERNEL_TRACE_PATH and BATCHLAS_TRACE_PATH,
    // else "batchlas_kernels.trace.json". The precedence -- and the fall-through
    // when the first variable is set but empty -- is resolved here so the call
    // site does not have to re-derive it. Opened for writing from an atexit
    // handler.
    std::string kernel_trace_path = "batchlas_kernels.trace.json";

    // BATCHLAS_COVERAGE_OUT -- raw, and read at two lifetimes today (static
    // initialisation decides whether coverage is enabled; an atexit handler
    // re-reads it to build the filename), which is a pair that could already
    // disagree. One capture removes that.
    EnvValue coverage_out{};

    // BATCHLAS_DEBUG_FILTER_DEGREE. PRESENCE ONLY: any value, INCLUDING THE
    // EMPTY STRING, enables it, because the call site only tests the getenv
    // pointer. It prints one line per batch item per iteration.
    bool debug_filter_degree = false;

    // BATCHLAS_DEBUG_SYTRD_SMALL, via env_truthy. Prints once.
    bool debug_sytrd_small = false;

    // BATCHLAS_GESVD_PROFILE, via env_truthy. Forces a wait_and_throw() per
    // stage -- it serialises the pipeline -- and prints CSV rows to stderr.
    bool gesvd_profile = false;

    // BATCHLAS_SYEVX_TRACE. Its call site hand-rolls a parser that accepts
    // exactly env_truthy's spelling set, so routing it through env_truthy is
    // byte-for-byte identical; it was simply never consolidated.
    bool syevx_trace = false;

    // BATCHLAS_CTA_DEBUG_SYNC, via env_truthy.
    //
    // The brief lists this under "safety", and it is NOT: it prints
    // "[cta-debug] <stage>" and calls wait_and_throw() at each stage. It removes
    // no check, changes no numerics, and cannot select a different kernel -- the
    // only behavioural difference is that an async SYCL exception surfaces
    // earlier and at a named stage, which is strictly SAFER. Its accidental cost
    // is a full pipeline drain per stage. Padding the unsafe group with it would
    // misrepresent what that group gates.
    bool cta_debug_sync = false;

    // BATCHLAS_STEQR_CTA_CHECK -- raw ({1,t,T,y,Y} first character, a fourth
    // boolean dialect).
    //
    // Also listed under "safety" in the brief, and it is the OPPOSITE of unsafe:
    // it ADDS a check. Set, steqr_cta waits and scans the per-item status array
    // and throws if any item did not converge; unset, non-convergence is SILENT.
    // The unsafe condition here is the DEFAULT, not the variable, so forcing it
    // to its default under a locked-down build would ENTRENCH silent
    // non-convergence. It stays in diagnostics for that reason.
    EnvValue steqr_cta_check{};

    // BATCHLAS_DUMP_BANDR1_* -- one coherent subsystem: a root path, a master
    // enable, a content filter and four selector indices, six of them read in
    // one function. Grouped so that clearing the family is one assignment.
    struct DumpBandR1Settings {
        // BATCHLAS_DUMP_BANDR1_DIR. create_directories() is called on this, and
        // files are written under it.
        std::string dir = "output/bandr1_dumps";

        // BATCHLAS_DUMP_BANDR1_STEP, via env_truthy. The master enable; the root
        // path above is only consulted while this is on.
        bool step = false;

        // BATCHLAS_DUMP_BANDR1_ABW_ONLY, via env_truthy. Suppresses every dump
        // except ABw.
        bool abw_only = false;

        // The four selectors, via env_int_or with a -1 sentinel meaning "no
        // filter". A non-negative value disables dumping for every other index.
        int step_index = -1;     // BATCHLAS_DUMP_BANDR1_STEP_INDEX
        int sweep_index = -1;    // BATCHLAS_DUMP_BANDR1_SWEEP_INDEX
        int step_in_sweep = -1;  // BATCHLAS_DUMP_BANDR1_STEP_IN_SWEEP
        int batch = -1;          // BATCHLAS_DUMP_BANDR1_BATCH: -1 dumps every item
    } dump_bandr1{};
};

// ---------------------------------------------------------------------------
// unsafe -- the overrides that REMOVE a guarantee.
//
// Membership is not "the knob is scary"; it is "setting this can make a correct
// program crash, hang, or silently compute wrong numbers, and nothing else in
// the process will say so". Three variables qualify. Two more that the brief
// grouped here do not, and the reasons are written on them in `diagnostics`.
//
// THE GATE. When the library is built WITHOUT the CMake option
// BATCHLAS_ALLOW_UNSAFE_ENV (which defaults to OFF), these fields keep their
// safe values no matter what the environment says, and each variable that was
// set produces ONE warning on stderr naming the variable and the option.
//
// The gate refuses the UNSAFE DIRECTION of a field, not every value that differs
// from the default. That distinction matters for blas_health: forcing it to its
// default would also block "error", which is STRICTER than the default and
// therefore not something a lockdown has any reason to refuse. Only "off" is
// gated.
//
// configure() is NOT gated: an embedding application that deliberately sets one
// of these in code has made a choice, which is exactly the affordance A-3 says
// is missing. The gate is about AMBIENT process state.
struct UnsafeSettings {
    // BATCHLAS_SKIP_POINTER_CHECKS. Safe value: false (checks ENABLED).
    //
    // What it disables: the one-USM-query-per-pointer-argument check (~70ns,
    // noise against a kernel launch) that turns "you passed host memory to a
    // device call" into a thrown std::invalid_argument naming the argument.
    // Without it, ordinary host memory reaches the device as a wild address,
    // giving CUDA_ERROR_ILLEGAL_ADDRESS and then a SIGABRT from inside the CUDA
    // runtime during teardown that no catch block can stop -- and the identical
    // code is correct on the host backend, so a CPU prototype passes and the GPU
    // run dies.
    //
    // PARSER, PRESERVED EXACTLY: the call site is !(v && *v && *v != '0'), so
    // ANY non-empty value whose first character is not '0' DISABLES the checks --
    // `=false`, `=off` and `=no` all turn them OFF. That is surprising, and it is
    // not env_truthy; settings.cc reproduces it character for character rather
    // than tightening it, because tightening it would silently change the
    // meaning of three real spellings at the one site in the tree that uses this
    // dialect.
    bool skip_pointer_checks = false;

    // BATCHLAS_LATRD_GRID_FORCE_UNSAFE, via env_truthy. Safe value: false.
    //
    // A DEADLOCK, not a wrong answer. It replaces the residency cap
    // (compute units / batch) with the forced group count from
    // BATCHLAS_LATRD_GRID_GROUPS, letting the launch put more work-groups on a
    // matrix than are guaranteed co-resident -- and the grid path's barrier is a
    // sense-reversing SPIN barrier whose termination argument is exactly that
    // co-residency. The kernel HANGS rather than fails, and the in-tree note at
    // the call site records that a hang there looks exactly like slow JIT: run
    // forced-unsafe measurements under `timeout`.
    //
    // Note that the escape hatch is largely self-defeating anyway: at batch >=
    // 128 the cap it exists to escape had already clamped the count to 1.
    bool latrd_grid_force_unsafe = false;

    // BATCHLAS_BLAS_HEALTH = off | warn | error. Safe value: Warn.
    //
    // THE FIFTH GENUINELY UNSAFE VARIABLE, and it is not in the brief's list of
    // four. `off` SKIPS the host-dgemm probe entirely, and that probe is the only
    // thing standing between the user and a host BLAS that computes dgemm
    // INCORRECTLY -- the known-broken Cooperlake OpenBLAS kernel. With the probe
    // off, every double and complex<double> result from the host/NETLIB backend
    // is silently wrong by O(1), with no diagnostic at all.
    //
    // `error` turns the probe's warning into a throw. It is STRICTER than the
    // default, so the gate lets it through; only `off` is refused.
    enum class BlasHealth { Off, Warn, Error };
    BlasHealth blas_health = BlasHealth::Warn;
};

// ---------------------------------------------------------------------------

// Everything BatchLAS reads from its environment, in one value.
//
// Copyable and assignable on purpose: the way to change one field
// programmatically is
//
//     auto s = batchlas::settings();
//     s.diagnostics.dump_bandr1.step = false;
//     batchlas::configure(s);
struct Settings {
    RoutingSettings routing{};
    SelectionSettings selection{};
    GeometrySettings geometry{};
    DiagnosticsSettings diagnostics{};
    UnsafeSettings unsafe{};
};

// The process-wide settings. The environment is read ONCE, under
// std::call_once, on the first call; thread-safe, and safe to call from a static
// initialiser (dispatch coverage does).
//
// The returned reference stays valid for the life of the process, but its
// CONTENTS can change under configure() and under detail::reload_settings(), so
// do not cache the fields across a call that could do either. In practice that
// means: read settings().group.field where you used to call std::getenv, and do
// NOT hoist it into a function-local static -- that latch is the defect three
// call sites in this tree already carry explicit written prohibitions against.
BATCHLAS_API const Settings& settings();

// Programmatic override, for an embedding application that wants to stop
// inheriting ambient process state.
//
// Permitted until the first Queue is constructed; afterwards it THROWS
// std::runtime_error. The restriction is not bureaucracy: routing and geometry
// knobs are read by *_buffer_size() queries as well as by the matching solve,
// and several of them change the workspace size, so a change taken mid-run would
// let two calls in one process disagree about how much scratch a solve needs.
// Configure before you build a Queue.
//
// (std::runtime_error rather than a batchlas-specific exception type: a real
// exception hierarchy is a separate piece of work, and inventing half of one
// here would have to be undone.)
//
// WHAT YOU CONFIGURE IS THE BASE, NOT A LOCK. detail::reload_settings() re-reads
// the environment on top of whatever configure() last set, rather than on top of
// the defaults. A knob you set here and do not export therefore holds for the
// life of the process; a knob something explicitly exports still wins.
//
// Both halves are deliberate, and each is a defect the other way round. Rebuilding
// from the defaults discarded this call entirely -- a ScopedEnvVar on an unrelated
// variable, anywhere in an embedding application's own harness, was enough to do
// it silently. Ignoring the environment afterwards made every ScopedEnvVar in the
// process a no-op, which turns an A/B test into two runs of the same arm that
// agree by construction.
BATCHLAS_API void configure(const Settings& s);

namespace detail {

// Re-read the environment, discarding whatever settings() currently holds.
//
// THIS IS WHAT KEEPS THE TEST SUITE GREEN. Fifteen test files pin a route or a
// geometry knob with batchlas::ScopedEnvVar and expect the very next library
// call to see it, so ScopedEnvVar calls this from its constructor AND its
// destructor. Without the destructor call, a test's knob would leak into every
// test that runs after it in the same process.
//
// It is also what preserves three call sites' explicit written prohibitions on
// latching a getenv in a function-local static ("a getenv cached there makes a
// later setenv invisible, and the test then passes green on the default arm").
// settings() IS that latch; reload_settings() is the release.
//
// TWO THINGS IT DOES NOT COVER, both recorded rather than papered over:
//
//   * A test or benchmark that calls ::setenv directly instead of using
//     ScopedEnvVar does not reach this, and will now read the pre-existing
//     value. Several do today. Those call sites have to move to ScopedEnvVar.
//
//   * A reload that lands BETWEEN a *_buffer_size() query and its solve can
//     change a block width and therefore under-size the workspace the caller
//     already allocated. That hazard is written on tuning_params.hh's accessors
//     and on two-stage's chase knob; it existed before this header (those knobs
//     were re-read per call) and is unchanged by it. Do not let a ScopedEnvVar
//     straddle a sizing/solve pair.
//
// Not thread-safe with respect to concurrent settings() readers, for the same
// reason ScopedEnvVar is not: the process environment is not thread-safe either.
// Call it from a test body or a benchmark setup, never from a parallel region.
BATCHLAS_API void reload_settings();

// Latch: records that a Queue has been constructed, which is what closes
// configure(). Called from Queue's constructors; nothing else should call it.
BATCHLAS_API void note_queue_constructed() noexcept;

// True once note_queue_constructed() has been called.
BATCHLAS_API bool queue_constructed() noexcept;

}  // namespace detail

}  // namespace batchlas
