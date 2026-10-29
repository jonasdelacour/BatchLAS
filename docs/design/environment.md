# Environment variables {#design_environment}

> **Status:** current · checked 2026-10-29

Every `BATCHLAS_*` variable the library reads is captured once into `batchlas::Settings`
(`include/batchlas/settings.hh`, which documents each field; `src/util/settings.cc` is the only
reader; loader design: [settings loader](runtime-internals.md#runtime-internals-the-settings-loader);
user guide: [Configuration](../cpp-api.md#configuration)). Test and benchmark-only variables are not
listed, except [`BATCHLAS_VERIFY_RECORD`](#environment-verify-record-is-read-raw).

`settings()` reads the environment once under `std::call_once` and is safe from a static
initialiser. The reference lives for the process, but its contents change under `configure()` and
`detail::reload_settings()`: read `settings().group.field` at the point of use, never latch it in a
function-local static.

## Move the string, not the parser {#environment-move-the-string-not-the-parser}

Knobs keep their historical boolean dialects: `env_truthy` (exactly `1`, `true`, `TRUE`, `on`,
`ON`), a case-folding variant that adds `yes`, first character `1tTyY` (`on` fails), inverted first
character (on unless `0nNfF`), `atoi != 0` (`true` is false), first character `0`/`1` only, and the
wider disable set of `SB2ST_BACK_WAVE`. A knob with a shared parser gets a typed field; one with a
bespoke parser gets an `EnvValue` holding the raw string, whose `get()` is `nullptr` when unset, so
`std::getenv("X")` becomes `settings().group.x.get()`. Unset and set-but-empty stay distinct.

## Routing and selection variables

"Re-read" means read on every call; never latch those.

| Variable | Values (default) | Note |
| --- | --- | --- |
| `BATCHLAS_<OP>_ROUTE` | `auto`, `native`, `vendor`, or a choice spelling (`auto`) | [Route pins](#environment-route-pins-are-raw-strings). |
| `BATCHLAS_TUNED_DIR` | directory (built-in tables) | A same-named file (`gemm.sm_89.txt`) replaces the built-in rows dtype by dtype; other names warn. |
| `BATCHLAS_GEMV_SEGT` | `off`, `auto`, `2`, `4`, `8` (unset) | Re-read. |
| `BATCHLAS_GESVD_BIDIAG` | `bdsdc`, `normal`, `bdsqr` (`bdsdc`) | Changes numerics; [overrides](#environment-knobs-that-override-an-explicit-argument). |
| `BATCHLAS_GETRF_LEAF`, `BATCHLAS_GEQRF_LEAF` | `slm`, `reg` (`reg`); `auto`, `reg` (`auto`) | Re-read. |
| `BATCHLAS_GETRF_LASWP` | `inloop`, `defer_walk`, `defer_gather` (`defer_gather`) | Presence latched, value re-read. |
| `BATCHLAS_GETRF_RIGHT_LASWP`, `BATCHLAS_GETRS_LASWP` | `walk`, `gather` (measured crossover; unset) | getrs re-read. |
| `BATCHLAS_ILUK_DEVICE` | first character `0`/`1` (batch >= 32) | `true`, `on`, `yes` fall through to the default. |
| `BATCHLAS_LATRD_IMPL` | `legacy`, `device`, `grid` (unset) | Re-read. |
| `BATCHLAS_ORMQR_IMPL`, `BATCHLAS_SYTRD_IMPL` / `BATCHLAS_ORTHO_GRAM` | `device` / `gemm` (unset) | Only that value has an effect. |
| `BATCHLAS_ORMQR_WY` | `gemm`, `trmm`, `measured` (measured gate) | |
| `BATCHLAS_SB2ST_BACK_WAVE` | on | [Fails open](#environment-sb2st-back-wave-fails-open). |
| `BATCHLAS_SB2ST_SUBGROUP` | `auto`, `on`, `off` (`auto`) | `on` throws when kd > 32 or no sub-group of 32. |
| `BATCHLAS_SYEV_TWO_STAGE_CHASE` | `givens` (unset) | Read by the solve and the sizing query. |
| `BATCHLAS_SYEVX_ALGORITHM` | method name (`SyevxParams::method`) | Overrides the argument; unrecognised parses to `Auto`. |
| `BATCHLAS_SYEVX_PRECONDITIONER` | name (unset) | Loses to an explicit request and to a configured ILU(k). |
| `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO` | `atoi` nonzero (off) | [evidence](../perf/syevx.md#syevx-automatic-chebyshev-filter-degree) |
| `BATCHLAS_SYEVX_INSTR_HOST`, `BATCHLAS_SYEVX_PROJECTED_VENDOR` | first character `1tTyY` (off) | The latter changes the workspace size. |
| `BATCHLAS_SYEVX_SOFT_LOCK` | inverted first character (on) | `=off` and empty read as on. |
| `BATCHLAS_SYTRD_FORCE_LOCAL_SMALL` | truthy (off) | Cannot force an unschedulable launch. |
| `BATCHLAS_SYTRD_FUSE_PANEL_UPDATE` | truthy / falsy (tuning rule) | Tri-state; unrecognised counts as unset. |
| `BATCHLAS_SYTRD_TRAILING_UPDATE` | `gemm`, `syr2k`, `her2k`, `rank2k`, either case (unset) | `syr2k` and `her2k` are one route. |

## Geometry variables

Group `geometry`. A default that depends on n, type, argument or device stays at the
[call site](#environment-geometry-defaults-stay-at-the-call-site).

| Variable | Values (default) | Note |
| --- | --- | --- |
| `BATCHLAS_LATRD_GRID_GROUPS` | positive int (0 = auto) | Clamped by the residency cap. |
| `BATCHLAS_LATRD_GRID_MIN_N` | positive int (768) | [evidence](../perf/syev.md#syev-latrd-grid-gate-confirmed-in-eigenvector-mode) |
| `BATCHLAS_LATRD_GRID_WG` | `32`, `64`, `128`, `256` (computed) | Other values ignored silently. |
| `BATCHLAS_LATRD_LOWER_PANEL_WG_HINT` | `64`, `128`, `256` (unset) | Device path only. |
| `BATCHLAS_SB2ST_BACK_SUBS`, `BATCHLAS_SB2ST_BACK_TILE_W` | positive int (call site) | Force together; instantiated: tile {1, 2, 4, 8}, subs {4, 8, 16}. |
| `BATCHLAS_SB2ST_BACK_TILE` | int (call site) | `0` selects the streaming kernel. |
| `BATCHLAS_POTRF_NB`, `BATCHLAS_POTRF_W` | int (type default) | nb 128/96/96/64, w 128/32/32/16 for s/d/c/z. |
| `BATCHLAS_SYEV_TWO_STAGE_KD`, `BATCHLAS_SYEV_TWO_STAGE_SB2ST_BLOCK` | positive int (32) | KD clamped to [1, n-1]. |
| `BATCHLAS_SY2SB_ORMQR_NB` | int or `off` (unset) | `off`/`0` disable; positive forced, clamped to kd; bad, negative or > 1024 is unset. |
| `BATCHLAS_SYTRD_BLOCK_SIZE` | int (bucketed by n and type) | |
| `BATCHLAS_TRMM_TILE_M` | int (function of m) | Bucketed to 16, 32, 64, 128. |
| `BATCHLAS_GEMM_WIDE_STAGES` | `2`, `3` (unset) | `complex<float>` wide tiles: `2` register prefetch, `3` cp.async (sm_80+, default sm_120+). |
| `BATCHLAS_TRSM_OUTER_NB` | int (128 `Left`, CTA nb `Right`) | Re-read. |
| `BATCHLAS_EXPAND_MAX_BYTES` | bytes (`GLOBAL_MEM_SIZE / 4`) | Only lowers the ceiling. |
| `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` | bare `atoi` (1) | [Parser defect](#environment-parser-defects-recorded-not-fixed). |
| `BATCHLAS_SYEVX_CHECK_EVERY` | int (4) | Each check drains the pipeline. |
| `BATCHLAS_SYEVX_EXTRA_DIRECTIONS` | int (`max(2, k/4)`) | `0` = no guard block; negative is unset. |
| `BATCHLAS_SYEVX_FILTER_DEGREE`, `_INIT_POWER`, `_LOCK_FACTOR` | int > 0 (auto); int (4); float > 0 (0.1) | `INIT_POWER=0` = no power iterations. |
| `BATCHLAS_TUNE_*` (13: `ORMQR_BLOCK_SIZE`, `GEBRD_BLOCK_SIZE`, `SB2ST_BACK_TILE`, `SB2ST_BACK_SUBS`, `SY2SB_ORMQR_NB`, `SYTRD_BLOCK_SIZE`, `LATRD_WG_HINT`, `STEDC_RECURSION_THRESHOLD`, `STEDC_MERGE_VARIANT`, `STEDC_THREADS_PER_ROOT`, `STEDC_WG_MULTIPLIER`, `L2_CHUNK_PERCENT`, `L2_CHUNK_ITEMS`) | int (tuned) | `tuning_env_override`, @ref perf_tuning. `L2_CHUNK_ITEMS` forces the blocked drivers' batch chunk. |

## Diagnostic variables

Group `diagnostics`; truthy is `env_truthy`.

| Variable | Values (default) | Note |
| --- | --- | --- |
| `BATCHLAS_QUEUE_PROFILING`, `BATCHLAS_BENCH_PROFILING` | truthy (off) | ORed; kernel trace implies profiling. |
| `BATCHLAS_KERNEL_TRACE`, `BATCHLAS_TRACE_KERNELS` | truthy (off) | ORed. |
| `BATCHLAS_KERNEL_TRACE_PATH`, `BATCHLAS_TRACE_PATH` | path (`batchlas_kernels.trace.json`) | First non-empty wins; written at exit. |
| `BATCHLAS_METAL_FLUSH_EACH_OP` | truthy (off) | One command buffer per operation, to attribute a GPU fault; profiling implies it. |
| `BATCHLAS_COVERAGE_OUT` | path prefix (off) | Written at exit as `<value>.<pid>`. |
| `BATCHLAS_SELECT_TRACE` | truthy (off) | One stderr line per selection. |
| `BATCHLAS_DEBUG_FILTER_DEGREE` | presence (off) | Empty enables it. |
| `BATCHLAS_DEBUG_SYTRD_SMALL`, `BATCHLAS_SYEVX_TRACE` | truthy (off) | |
| `BATCHLAS_GESVD_PROFILE`, `BATCHLAS_CTA_DEBUG_SYNC` | truthy (off) | Drain per stage. |
| `BATCHLAS_STEQR_CTA_CHECK` | first character `1tTyY` (off) | Throws if an item did not converge. |
| `BATCHLAS_DUMP_BANDR1_STEP` (enable), `_ABW_ONLY`, `_DIR` (`output/bandr1_dumps`), `_STEP_INDEX`, `_SWEEP_INDEX`, `_STEP_IN_SWEEP`, `_BATCH` (`-1` = no filter) | | Band-reduction dumps. |

## Unsafe variables

Group `unsafe`, [gated](#environment-the-unsafe-group-and-its-gate).

| Variable | Values (default) | Note |
| --- | --- | --- |
| `BATCHLAS_SKIP_POINTER_CHECKS` | non-empty, first character not `0` (checks on) | `false`, `off`, `no` also disable the checks. |
| `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` | truthy (off) | Forces `LATRD_GRID_GROUPS`; can deadlock. |
| `BATCHLAS_BLAS_HEALTH` | `off`, `warn`, `error` (`warn`) | Only `off` is gated. |

## Route pins are raw strings {#environment-route-pins-are-raw-strings}

`Settings::routing` holds one raw `EnvValue` per op in `RoutingSettings::ops` (29 ops). Names are
synthesised (`BATCHLAS_` + upper-cased op + `_ROUTE`), so a grep for literals misses them; to add a
pinned op, add it to that array. `select::detail::pin_text` / `resolve_pin` trim and case-fold the
value (empty is `auto`). A spelling that does not parse, is not compiled, or cannot run the shape
throws `batchlas::invalid_argument`; `native` and `vendor` warn once and fall back to `auto`.
`select::ScopedPin` wins over the environment on its thread and nests; a strict pin makes a class
word throw instead. Python: `batchlas.set_route`, the `batchlas.route` context manager and
`batchlas.route_pin` (`python/batchlas/_api.py`) set the variable and reload. Retired, no longer
read: `BATCHLAS_<OP>_VARIANT`, `BATCHLAS_<OP>_PROVIDER`, `BATCHLAS_GEMM_SYCL_KERNEL`,
`BATCHLAS_GEMM_EXPERIMENTAL`, `BATCHLAS_GEMM_CUBLASDX_KERNEL`, `BATCHLAS_SYEV_SMALL_KERNEL`,
`BATCHLAS_SYEV_CTA_MAX_N` (@ref design_flat_selection).

## Knobs that override an explicit argument {#environment-knobs-that-override-an-explicit-argument}

- `BATCHLAS_SYEVX_ALGORITHM` overrides `SyevxParams::method`; an environment algorithm degrades
  where an explicit request throws.
- `BATCHLAS_SYEVX_PRECONDITIONER` overrides a `SyevxParams` field.
- `BATCHLAS_GESVD_BIDIAG=normal` squares the condition number (relative error 8.5e-1 against 5.0e-1
  at kappa 1e6, [gesvd Tier 3](../perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver)); a
  thin-tall U overrides it.

## Geometry defaults stay at the call site {#environment-geometry-defaults-stay-at-the-call-site}

A knob whose default depends on n, type, argument or device stores a sentinel (0 or unset) and the
call site applies the default, so a tuned curve is not pinned at one point; constant defaults live in
the struct. Where 0 is meaningful the field bypasses `env_positive_int_or` (which maps 0 to unset):
`SB2ST_BACK_TILE` is raw because 0 selects the streaming kernel.

## Two variables for one quantity {#environment-two-variables-for-one-quantity}

The outer variable wins over its `BATCHLAS_TUNE_*` twin: `SYTRD_BLOCK_SIZE`,
`LATRD_LOWER_PANEL_WG_HINT`, `SY2SB_ORMQR_NB` and the two sb2st back-transform knobs. The latrd pair
stays separate because the outer knob applies only on the device path; the sb2st pair must be forced
together, since forcing one gives an unmeasured geometry.

## sb2st back wave fails open {#environment-sb2st-back-wave-fails-open}

`BATCHLAS_SB2ST_BACK_WAVE` is on unless set to one of `{0, false, off, no, n, disable, disabled}`,
case-insensitive. Do not route it through `env_falsy`, which would turn the wave on for `False`,
`Off` and `no`.

## Parser defects, recorded not fixed {#environment-parser-defects-recorded-not-fixed}

Each fix changes behaviour and belongs in its own commit:

- `BATCHLAS_SYEVX_SOFT_LOCK` is inverted: `=off` and an empty value enable it.
- `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` uses bare `atoi`: garbage gives 0, which widens the blocked
  gebrd path to every n (`env_int_or` would give 1).
- `BATCHLAS_SKIP_POINTER_CHECKS` disables the checks for any non-empty value not starting with `0`.
- `BATCHLAS_TUNE_STEDC_MERGE_VARIANT` is cast to `StedcMergeVariant` unchecked: variant 0 is
  unreachable from the environment, and one variant deadlocks on some hardware.

## Files written from the environment {#environment-files-written-from-the-environment}

The kernel trace, the selection coverage (`$BATCHLAS_COVERAGE_OUT.<pid>`) and the band-reduction
dump open paths for writing, two of them from an `atexit` handler. `configure()` lets an embedding
application clear them first.

## BATCHLAS_VERIFY_RECORD is read raw {#environment-verify-record-is-read-raw}

`BATCHLAS_VERIFY_RECORD=<path>` makes every `verify::pass` append one calibration row
(@ref design_verification). It is the one `BATCHLAS_*` variable read outside `Settings`, because
`verify/tolerance.hh` belongs to the uninstalled `batchlas::verify` target and tests set it with a
bare `setenv` mid-process. It is read with `std::getenv` on every `pass`; the file stays open while
the path is unchanged, a new path reopens it, unset or empty closes it, and an unopenable path
records nothing and is retried.

## The unsafe group and its gate {#environment-the-unsafe-group-and-its-gate}

`Settings::unsafe` holds variables that can make a correct program crash, hang or silently compute
wrong numbers: `SKIP_POINTER_CHECKS` lets host memory reach a device call
(`CUDA_ERROR_ILLEGAL_ADDRESS` on CUDA), `LATRD_GRID_FORCE_UNSAFE` deadlocks (a hang looks like slow
JIT: run under `timeout`), and `BLAS_HEALTH=off` skips the host dgemm probe that catches a broken
host BLAS. Without the CMake option `BATCHLAS_ALLOW_UNSAFE_ENV` (default `OFF`) these keep their
safe values and each set variable warns once; only the unsafe direction is refused, and
`configure()` is not gated. The `dev` preset turns it `ON`; `cuda` and `release` set it back `OFF`
so the pre-push gate runs what a release build runs (`CMakePresets.json`).

## Two knobs that look unsafe {#environment-two-knobs-that-look-unsafe-and-are-not}

Both stay in `diagnostics`: `BATCHLAS_CTA_DEBUG_SYNC` removes no check and only surfaces async
exceptions earlier; `BATCHLAS_STEQR_CTA_CHECK` adds a check, so its unsafe state is the default.

## configure() sets the base {#environment-configure-sets-the-base-not-a-lock}

`configure()` is accepted until the first `Queue` is constructed, then throws
`batchlas::api_misuse` (@ref design_error_model), because sizing queries and solves must read the
same settings. It sets a base, not a lock: `detail::reload_settings()` re-reads the environment on
top of it, so an exported variable still wins.

## What a reload does not cover {#environment-what-a-settings-reload-does-not-cover}

`ScopedEnvVar` reloads in its constructor and destructor; a direct `::setenv` does not. A reload
between a `*_buffer_size()` query and its solve can change a block width and under-size the
caller's workspace. The reload is not thread-safe against concurrent `settings()` readers.

## Shared parsers in env.hh {#environment-the-shared-parsers-in-envhh}

`<batchlas/util/env.hh>` holds `env_truthy`, `env_falsy`, `env_int_or` and `env_positive_int_or`.
Their accepted spellings are the contract: do not widen them; a knob that wants `Off` or `No`
case-folds its own value. `env_positive_int_or` is the reader for kernel-geometry knobs; zero and
negative fall back to the default.

## ScopedEnvVar {#environment-scopedenvvar}

`ScopedEnvVar` (`<batchlas/util/env.hh>`) sets a variable for one scope with `setenv`/`unsetenv`
(not `putenv`) and restores it on exit. The name is borrowed and must outlive the object; a null
value unsets for the scope. Not copyable; nested instances restore innermost first.
