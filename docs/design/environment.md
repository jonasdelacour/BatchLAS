# Environment variables {#design_environment}

> **Status:** current · updated 2026-10-06

Every `BATCHLAS_*` variable the library reads is captured once into `batchlas::Settings`
(`include/batchlas/settings.hh`, `src/util/settings.cc`). The user guide is
[Configuration](../cpp-api.md#configuration). Test and benchmark-only variables are not listed.

## Snapshot and reload

- `settings()` reads the environment once, under `std::call_once`. It is thread-safe and may be
  called from a static initialiser.
- The reference stays valid for the life of the process, but its contents change under
  `configure()` and `detail::reload_settings()`.
- Read `settings().group.field` at the point of use. Copying a field into a function-local static
  latches it across a later `setenv`.

## Move the string, not the parser {#environment-move-the-string-not-the-parser}

The tree uses seven incompatible boolean dialects, and the parsers stay as they were:

| Dialect | Accepts |
| --- | --- |
| `env_truthy` | exactly `1`, `true`, `TRUE`, `on`, `ON` |
| case-folding variant | the above, case-insensitive, plus `yes` |
| first character | `1`, `t`, `T`, `y`, `Y` (`true` works, `on` does not) |
| inverted first character | on unless the first character is `0`, `n`, `N`, `f`, `F` |
| `atoi != 0` | `true` is false |
| `0` or `1` only | first character `0` or `1` |
| wider disable set | `SB2ST_BACK_WAVE` only, see below |

A knob with a shared parser gets a typed field. A knob with a bespoke parser gets an `EnvValue`
holding the raw string. `EnvValue::get()` returns `nullptr` when the variable is unset, so
`std::getenv("BATCHLAS_FOO")` becomes `settings().group.foo.get()`. Unset and set-but-empty stay
distinct: `BATCHLAS_KERNEL_TRACE_PATH` falls through to `BATCHLAS_TRACE_PATH` only when set and empty.

## Routing and selection variables

"Re-read" means the value is read on every call.

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `BATCHLAS_<OP>_ROUTE` | `auto`, `native`, `vendor`, or a choice spelling (`lpanel:panel=8`) | `auto` | Pins the route of one op. See [route pins](#environment-route-pins-are-raw-strings). |
| `BATCHLAS_TUNED_DIR` | directory | built-in tables | Replaces the table of the same name (`gemm.float.sm_89.txt`). Other files are skipped with a warning. |
| `BATCHLAS_EXPAND_ROUTE` | `expand`, `loop` | unset | Chooses expansion or the per-item loop for hemm/herk/her2k. |
| `BATCHLAS_GEMV_SEGT` | `off`, `auto`, `2`, `4`, `8` | unset | Re-read; must not be latched. |
| `BATCHLAS_GESVD_BIDIAG` | `bdsdc`, `normal`, `bdsqr` | unset | Bidiagonal solver. Changes numerics; see [knobs that override an argument](#environment-knobs-that-override-an-explicit-argument). |
| `BATCHLAS_GETRF_LEAF` | `slm`, `reg` | `reg` | Re-read per call. |
| `BATCHLAS_GEQRF_LEAF` | `auto`, `reg` | `auto` | Re-read per call. |
| `BATCHLAS_GETRF_LASWP` | `inloop`, `defer_walk`, `defer_gather` | `defer_gather` | Presence latched, value re-read. |
| `BATCHLAS_GETRF_RIGHT_LASWP` | `walk`, `gather` | measured crossover | Right-side row swaps. |
| `BATCHLAS_GETRS_LASWP` | `walk`, `gather` | unset | Re-read; must not be latched. |
| `BATCHLAS_ILUK_DEVICE` | first character `0` or `1` | shape default | `true`, `on`, `yes` fall through to the default. |
| `BATCHLAS_LATRD_IMPL` | `legacy`, `device`, `grid` | unset | Re-read per call. |
| `BATCHLAS_ORMQR_IMPL` | `device` | unset | Only `device` has an effect. |
| `BATCHLAS_ORMQR_WY` | `gemm`, `trmm`, `measured` | measured gate | Overrides the measured gate. |
| `BATCHLAS_ORTHO_GRAM` | `gemm` | unset | Only `gemm` has an effect. |
| `BATCHLAS_SB2ST_BACK_WAVE` | `0`, `false`, `off`, `no`, `n`, `disable`, `disabled` disable (case-insensitive) | on | Wave back-transform. Fails open. |
| `BATCHLAS_SB2ST_SUBGROUP` | `auto`, `on`, `off` | `auto` | Forced `on` throws when kd > 32 or there is no sub-group of 32. |
| `BATCHLAS_SYEV_TWO_STAGE_CHASE` | `givens` | unset | Read by the solve and by the sizing query. |
| `BATCHLAS_SYEVX_ALGORITHM` | method name | `SyevxParams::method` | Overrides the explicit method. An unrecognised value parses to `Auto`. |
| `BATCHLAS_SYEVX_PRECONDITIONER` | preconditioner name | unset | Overrides a `SyevxParams` field. Loses to an explicit request and to a configured ILU(k). |
| `BATCHLAS_SYEVX_BOUNDS_LEGACY` | `atoi` nonzero | off | `true`, `on`, `yes` are false. |
| `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO` | `atoi` nonzero | off | [evidence](../perf/syevx.md#syevx-automatic-chebyshev-filter-degree) |
| `BATCHLAS_SYEVX_INSTR_HOST` | first character `1`, `t`, `T`, `y`, `Y` | unset | `on` does not work. |
| `BATCHLAS_SYEVX_PROJECTED_VENDOR` | first character `1`, `t`, `T`, `y`, `Y` | unset | Changes the workspace size. |
| `BATCHLAS_SYEVX_SOFT_LOCK` | first character not in `0`, `n`, `N`, `f`, `F` | on | Inverted; `=off` and an empty value read as on. |
| `BATCHLAS_SYTRD_FORCE_LOCAL_SMALL` | truthy | off | Cannot force an unschedulable launch. |
| `BATCHLAS_SYTRD_FUSE_PANEL_UPDATE` | truthy or falsy | tuning rule | Tri-state: forced on, forced off, or the tuning rule. Unrecognised values count as unset. |
| `BATCHLAS_SYTRD_IMPL` | `device` | unset | Latched at its call site. |
| `BATCHLAS_SYTRD_TRAILING_UPDATE` | `gemm`, `syr2k`, `her2k`, `rank2k` (either case) | unset | `syr2k` and `her2k` are one route. |

## Geometry variables

Fields are in the `geometry` group. A default that depends on n, type, argument or device stays at
the call site; see [geometry defaults](#environment-geometry-defaults-stay-at-the-call-site).

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `BATCHLAS_LATRD_GRID_GROUPS` | positive int | `0` (auto) | Still clamped by the residency cap. |
| `BATCHLAS_LATRD_GRID_MIN_N` | positive int | `768` | [evidence](../perf/syev.md#syev-latrd-grid-gate-confirmed-in-eigenvector-mode) |
| `BATCHLAS_LATRD_GRID_WG` | `32`, `64`, `128`, `256` | call site | Other values are ignored silently. |
| `BATCHLAS_LATRD_LOWER_PANEL_WG_HINT` | `64`, `128`, `256` | call site | Device path only. |
| `BATCHLAS_SB2ST_BACK_SUBS` | positive int | call site | Force together with `BATCHLAS_SB2ST_BACK_TILE_W`. Only tile in {1, 2, 4, 8} and subs in {4, 8, 16} are instantiated. |
| `BATCHLAS_SB2ST_BACK_TILE_W` | positive int | call site | Column tile of the wave kernel. |
| `BATCHLAS_SB2ST_BACK_TILE` | int | call site | Tiled kernel. `0` selects the streaming kernel. |
| `BATCHLAS_POTRF_NB`, `BATCHLAS_POTRF_W` | int | `0` (type default) | Defaults: nb 128/96/96/64 and w 128/32/32/16 for float, double, cfloat, cdouble. |
| `BATCHLAS_SYEV_TWO_STAGE_KD` | positive int | `32` | Clamped to [1, n-1]. |
| `BATCHLAS_SYEV_TWO_STAGE_SB2ST_BLOCK` | positive int | `32` | Read by four solve and sizing pairs. |
| `BATCHLAS_SY2SB_ORMQR_NB` | int, or `off` | unset | `off` or `0` disables the hint. Positive is forced, clamped to kd. Unparseable, negative or above 1024 is unset. |
| `BATCHLAS_SYTRD_BLOCK_SIZE` | int | `0` (bucketed default) | Bucketed by n and type. |
| `BATCHLAS_TRMM_TILE_M` | int | `0` (unset) | Bucketed to 16, 32, 64, 128. |
| `BATCHLAS_TRSM_OUTER_NB` | int | 128 (`Left`), CTA nb (`Right`) | Re-read per call. |
| `BATCHLAS_EXPAND_MAX_BYTES` | bytes (`strtoull`) | `GLOBAL_MEM_SIZE / 4` | Only lowers the ceiling. |
| `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` | int (bare `atoi`) | `1` | Unparseable gives 0 and widens the path; see [parser defects](#environment-parser-defects-recorded-not-fixed). |
| `BATCHLAS_SYEVX_CHECK_EVERY` | int | `4` | Each check drains the pipeline. |
| `BATCHLAS_SYEVX_EXTRA_DIRECTIONS` | int | `max(2, k/4)` | `0` means no guard block. Negative is unset. |
| `BATCHLAS_SYEVX_FILTER_DEGREE` | int > 0 | auto | Disables the automatic degree. |
| `BATCHLAS_SYEVX_INIT_POWER` | int | `4` | `0` means no power iterations. |
| `BATCHLAS_SYEVX_LOCK_FACTOR` | float > 0 | `0.1` | Only non-integer knob. |
| `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE`, `BATCHLAS_TUNE_GEBRD_BLOCK_SIZE`, `BATCHLAS_TUNE_SB2ST_BACK_TILE`, `BATCHLAS_TUNE_SB2ST_BACK_SUBS`, `BATCHLAS_TUNE_SY2SB_ORMQR_NB`, `BATCHLAS_TUNE_SYTRD_BLOCK_SIZE`, `BATCHLAS_TUNE_LATRD_WG_HINT`, `BATCHLAS_TUNE_STEDC_RECURSION_THRESHOLD`, `BATCHLAS_TUNE_STEDC_MERGE_VARIANT`, `BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT`, `BATCHLAS_TUNE_STEDC_WG_MULTIPLIER` | int | tuned | Read through `tuning_env_override`. See @ref perf_tuning. |

## Diagnostic variables

Fields are in the `diagnostics` group. Truthy values are `1`, `true`, `TRUE`, `on`, `ON`.

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `BATCHLAS_QUEUE_PROFILING`, `BATCHLAS_BENCH_PROFILING` | truthy | off | ORed. Kernel trace implies profiling. |
| `BATCHLAS_KERNEL_TRACE`, `BATCHLAS_TRACE_KERNELS` | truthy | off | ORed. |
| `BATCHLAS_KERNEL_TRACE_PATH`, `BATCHLAS_TRACE_PATH` | path | `batchlas_kernels.trace.json` | First non-empty value wins. Written at exit. |
| `BATCHLAS_COVERAGE_OUT` | path prefix | off | Written at exit as `<value>.<pid>`. |
| `BATCHLAS_SELECT_TRACE` | truthy | off | One stderr line per selection decision. |
| `BATCHLAS_DEBUG_FILTER_DEGREE` | presence | off | The empty string enables it. |
| `BATCHLAS_DEBUG_SYTRD_SMALL` | truthy | off | Prints once. |
| `BATCHLAS_GESVD_PROFILE` | truthy | off | Drains per stage. |
| `BATCHLAS_SYEVX_TRACE` | truthy | off | |
| `BATCHLAS_CTA_DEBUG_SYNC` | truthy | off | Prints each stage and waits; a full drain per stage. |
| `BATCHLAS_STEQR_CTA_CHECK` | first character `1`, `t`, `T`, `y`, `Y` | off | Throws if any item did not converge. |
| `BATCHLAS_DUMP_BANDR1_DIR` | path | `output/bandr1_dumps` | Created with `create_directories()`. |
| `BATCHLAS_DUMP_BANDR1_STEP` | truthy | off | Master enable. |
| `BATCHLAS_DUMP_BANDR1_ABW_ONLY` | truthy | off | |
| `BATCHLAS_DUMP_BANDR1_STEP_INDEX`, `BATCHLAS_DUMP_BANDR1_SWEEP_INDEX`, `BATCHLAS_DUMP_BANDR1_STEP_IN_SWEEP`, `BATCHLAS_DUMP_BANDR1_BATCH` | int | `-1` | `-1` means no filter. |

## Unsafe variables

Fields are in the `unsafe` group. See [the unsafe group and its gate](#environment-the-unsafe-group-and-its-gate).

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `BATCHLAS_SKIP_POINTER_CHECKS` | any non-empty value whose first character is not `0` | checks on | Disables the USM pointer check. `false`, `off`, `no` also disable it. |
| `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` | truthy | off | Forces the group count from `BATCHLAS_LATRD_GRID_GROUPS`. Can deadlock. |
| `BATCHLAS_BLAS_HEALTH` | `off`, `warn`, `error` | `warn` | Only `off` is gated. |

## Route pins are raw strings {#environment-route-pins-are-raw-strings}

`Settings::routing` holds one raw `EnvValue` per op, in the order of `RoutingSettings::ops`.
The names are built from the op (`BATCHLAS_` + op + `_ROUTE`), so a grep for string literals
misses them. `select::detail::pin_text` and `resolve_pin` (`src/select`) parse the value:

- The value is trimmed and case-folded. An empty value is `auto`.
- A spelling that does not parse, is not a compiled candidate, or cannot run the shape throws
  `std::invalid_argument`.
- `native` and `vendor` warn once and fall back to `auto` when nothing in their class can run.

Tests pin a route with `ScopedPin` (`src/select/select.hh`), which wins over the environment.
To add a pinned op, add its name to `RoutingSettings::ops`. `BATCHLAS_<OP>_VARIANT`,
`BATCHLAS_<OP>_PROVIDER`, `BATCHLAS_GEMM_SYCL_KERNEL`, `BATCHLAS_GEMM_EXPERIMENTAL`,
`BATCHLAS_GEMM_CUBLASDX_KERNEL`, `BATCHLAS_SYEV_SMALL_KERNEL` and `BATCHLAS_SYEV_CTA_MAX_N` are
retired and no longer read. See `docs/design/flat-kernel-selection.md` (sections 5.3 and 12).

## Knobs that override an explicit argument {#environment-knobs-that-override-an-explicit-argument}

Three knobs change behaviour that an API argument set explicitly:

- `BATCHLAS_SYEVX_ALGORITHM` overrides `SyevxParams::method`. An environment algorithm degrades
  where an explicit request throws.
- `BATCHLAS_SYEVX_PRECONDITIONER` overrides a `SyevxParams` field (`blas/enums.hh`,
  `blas/extensions.hh`).
- `BATCHLAS_GESVD_BIDIAG` changes numerics. `normal` squares the condition number: relative error
  8.5e-1 against 5.0e-1 at kappa 1e6
  ([gesvd Tier 3](../perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver)). The call site
  overrides it for thin-tall U.

## Geometry defaults stay at the call site {#environment-geometry-defaults-stay-at-the-call-site}

Twenty geometry knobs have a default that depends on n, type, argument or device. Their field is a
sentinel (0 or unset), and the default stays at the call site, so a tuned curve is not pinned at
one point. Constant defaults are stored in the struct.

Where 0 is meaningful, the field is not read by `env_positive_int_or`, which maps 0 to unset.
`SB2ST_BACK_TILE` is raw because 0 selects the streaming kernel.

## Two variables for one quantity {#environment-two-variables-for-one-quantity}

Five `BATCHLAS_TUNE_*` names sit under a newer variable for the same quantity. The outer variable
wins: `BATCHLAS_SYTRD_BLOCK_SIZE` over `TUNE_SYTRD_BLOCK_SIZE`, `BATCHLAS_LATRD_LOWER_PANEL_WG_HINT`
over `TUNE_LATRD_WG_HINT`, `BATCHLAS_SY2SB_ORMQR_NB` over `TUNE_SY2SB_ORMQR_NB`, and the two sb2st
back-transform knobs over their `TUNE_` twins.

The latrd pair is not unified, because the outer knob applies only on the device path. The sb2st
pair must be forced together, because forcing one gives an unmeasured geometry.

## sb2st back wave fails open {#environment-sb2st-back-wave-fails-open}

`BATCHLAS_SB2ST_BACK_WAVE` is on unless disabled. Its disable set is wider than `env_falsy`:
`{0, false, off, no, n, disable, disabled}`, case-insensitive. Do not route it through `env_falsy`,
which would turn the wave path on for `False`, `Off` and `no`.

## Parser defects, recorded not fixed {#environment-parser-defects-recorded-not-fixed}

Fixing these changes behaviour, so each fix belongs in its own commit:

- `BATCHLAS_SYEVX_SOFT_LOCK` is inverted. `=off` and an empty value enable it.
- `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` uses bare `atoi`. An unparseable value gives 0, which widens
  the blocked gebrd path to every n. `env_int_or` would give 1 instead.
- `BATCHLAS_SKIP_POINTER_CHECKS` disables the checks for any non-empty value not starting with `0`.
  `settings.cc` reproduces this exactly.
- `BATCHLAS_TUNE_STEDC_MERGE_VARIANT` is `static_cast` to `StedcMergeVariant` without a range check.
  Variant 0 cannot be reached from the environment, and one variant deadlocks on some hardware.

## Files written from the environment {#environment-files-written-from-the-environment}

Three diagnostics knobs open a path for writing: the kernel trace (`kernel_trace_path`), the
selection coverage (`$BATCHLAS_COVERAGE_OUT.<pid>`) and the band-reduction dump
(`$BATCHLAS_DUMP_BANDR1_DIR`). Two write from an `atexit` handler. `configure()` lets an embedding application clear them first.

## The unsafe group and its gate {#environment-the-unsafe-group-and-its-gate}

A variable is in `Settings::unsafe` when it can make a correct program crash, hang or compute wrong
numbers silently:

- `BATCHLAS_SKIP_POINTER_CHECKS` lets host memory reach a device call. On CUDA the run dies with
  `CUDA_ERROR_ILLEGAL_ADDRESS`. The host backend passes.
- `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` deadlocks. A hang looks like slow JIT, so run forced
  measurements under `timeout`.
- `BATCHLAS_BLAS_HEALTH=off` skips the host dgemm probe. A broken host BLAS then gives double and
  `complex<double>` results that are silently wrong.

**Gate.** Without the CMake option `BATCHLAS_ALLOW_UNSAFE_ENV` (default `OFF`), these fields keep
their safe values, and each variable that is set prints one warning. Only the unsafe direction is
refused. `configure()` is not gated.

## Two knobs that look unsafe {#environment-two-knobs-that-look-unsafe-and-are-not}

Both stay in `diagnostics`:

- `BATCHLAS_CTA_DEBUG_SYNC` removes no check and changes no numerics. It only surfaces async
  exceptions earlier.
- `BATCHLAS_STEQR_CTA_CHECK` adds a check. Its unsafe condition is the default, so forcing the
  default would hide non-convergence.

## configure() sets the base {#environment-configure-sets-the-base-not-a-lock}

`configure()` is accepted until the first `Queue` is constructed. After that it throws
`batchlas::api_misuse` (see @ref design_error_model) and changes nothing, because sizing queries
and solves must read the same settings.

`configure()` sets a base, not a lock. `detail::reload_settings()` re-reads the environment on top
of the last `configure()` values, so an exported variable still wins.

## What a reload does not cover {#environment-what-a-settings-reload-does-not-cover}

`ScopedEnvVar` calls `detail::reload_settings()` in its constructor and destructor. Not covered:

- A direct `::setenv` call does not reach the reload. Use `ScopedEnvVar`.
- A reload between a `*_buffer_size()` query and its solve can change a block width and under-size
  the workspace the caller allocated.

The reload is not thread-safe against concurrent `settings()` readers.

## Shared parsers in env.hh {#environment-the-shared-parsers-in-envhh}

`<batchlas/util/env.hh>` holds the shared readers: `env_truthy`, `env_falsy`, `env_int_or` and
`env_positive_int_or`. The accepted spellings are the contract, so do not widen them. A knob that
wants `Off` or `No` must case-fold its own value. `env_positive_int_or` is the reader for
kernel-geometry knobs in `src/extensions`; zero and negative values fall back to the default.

## ScopedEnvVar {#environment-scopedenvvar}

`ScopedEnvVar` (`<batchlas/util/env.hh>`) sets a variable for one scope and restores it on exit.

- It uses `setenv` and `unsetenv`, not `putenv`, which keeps the caller's buffer alive.
- The name is borrowed and must outlive the object. A null value means unset for the scope.
- Copying is deleted. Nested instances restore innermost first.
