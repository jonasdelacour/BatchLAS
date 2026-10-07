# Environment: settings captured from BATCHLAS_* variables {#design_environment}

> **Covers:** why every `BATCHLAS_*` variable the library reads is captured once into
> `batchlas::Settings` (`include/batchlas/settings.hh`, `src/util/settings.cc`), the rule that
> kept every parser as it was, the table of variables with their parser dialects and traps, the
> `unsafe` group and its CMake gate, `configure()` and `reload_settings()` semantics, and the
> shared parsers and `ScopedEnvVar` in `include/batchlas/util/env.hh`.
> **Status:** current. Introduced in `b52e07b0` (2026-09-09); the routing group and the variable
> table were updated on 2026-10-06 for flat kernel selection, which retired the legacy route
> variables and several selection knobs. The user-facing guide, with a
> field/variable/default table per group, is [Configuration](../cpp-api.md#configuration); this
> page is the design record and the parser reference. API reference: the `config` group.

## environment: audit finding A-3

Before `settings.hh`, about 106 distinct `BATCHLAS_*` variables were read by about 77
`std::getenv` calls scattered through `src/` and `include/`. They steer routing, kernel
selection, launch geometry, diagnostics and, in three cases, whether arguments are validated at
all. Ambient process state silently decided which kernel ran. There was no programmatic
equivalent, no allow-list, and no way for a host application to stop inheriting a variable
somebody exported for a benchmark last week. The debugging value of the knobs is real and none
was removed: what changed is that the read happens once, in one place, and is overridable.

`settings()` reads the environment once under `std::call_once`. It is thread-safe and safe to
call from a static initialiser (the selection coverage in `src/select/coverage.cc` does). The reference it returns is valid for
the life of the process, but its contents change under `configure()` and
`detail::reload_settings()`, so a field must not be cached across a call that could do either.
Read `settings().group.field` where the code used to call `std::getenv`, and do not hoist it into
a function-local static: that latch is what three call sites carry explicit written prohibitions
against ("a getenv cached there makes a later setenv invisible, and the test then passes green on
the default arm"). `settings()` is that latch; `reload_settings()` is its release.

## environment: move the string, not the parser

The one rule that shaped `settings.hh`: move where the string comes from, not how it is parsed.
The tree contains seven mutually incompatible boolean dialects:

1. `env_truthy`: exactly `{1, true, TRUE, on, ON}`;
2. a case-folding variant that also accepts `yes` (`BATCHLAS_GEMM_EXPERIMENTAL`, no longer read
   since flat kernel selection deleted the gemm variant machinery);
3. first character in `{1, t, T, y, Y}` (`SYEVX_INSTR_HOST`, `SYEVX_PROJECTED_VENDOR`,
   `STEQR_CTA_CHECK`): `true` works, `on` does not;
4. an inverted first-character form: on unless the first character is in `{0, n, N, f, F}`
   (`SYEVX_SOFT_LOCK`);
5. `atoi != 0`, for which `true` is false (`SYEVX_BOUNDS_LEGACY`, `SYEVX_FILTER_DEGREE_AUTO`);
6. a form that looks only at `0` and `1` (`ILUK_DEVICE`);
7. `sytrd_sb2st_hh.cc`'s deliberately wider disable set (`SB2ST_BACK_WAVE`).

There are also three integer readers that disagree about trailing garbage (`env_int_or`'s
`stoi` accepts `"16x"` as 16, `tuning_env_override`'s `strtol` rejects it, bare `atoi` returns 0
for anything unparseable). Normalising them would change the reading of real spellings at real
call sites, which is exactly the class of silent behaviour change this repository has been bitten
by before.

So: a knob whose call site uses one of the shared parsers in `<batchlas/util/env.hh>` gets a
typed field, and `settings.cc` calls that same parser, so the value is bit-for-bit what the site
computed before. A knob with a bespoke parser gets an `EnvValue`, the raw captured string, and its
parser stays at the call site. `EnvValue::get()` is getenv-shaped (it returns `nullptr` for an
unset variable), so migrating such a site is replacing `std::getenv("BATCHLAS_FOO")` with
`settings().group.foo.get()` and nothing else. `EnvValue` also keeps "unset" apart from "set to
the empty string", which is load-bearing: `kernel-trace.hh` falls through from
`BATCHLAS_KERNEL_TRACE_PATH` to `BATCHLAS_TRACE_PATH` only when the first is set but empty, and
the route-pin reader (`select::detail::pin_text`) treats a set-but-empty `BATCHLAS_<OP>_ROUTE`
as `auto`. (Before flat kernel selection, `parse_route_env` treated a set-but-empty canonical
route variable as absent so that the legacy spelling still got a turn.)

## environment: the variables and their parser dialects

Every variable the library reads, its field, and how its value is parsed. "raw" means an
`EnvValue` whose parser stays at the call site. Variables read only by tests and benchmark
harnesses (`BATCHLAS_TEST_BACKEND`, `BATCHLAS_BENCH_*`, ...) are deliberately absent.

| variable | field | parser | notes |
| --- | --- | --- | --- |
| `BATCHLAS_<OP>_ROUTE` | `routing.route(op)`, one slot per entry of `RoutingSettings::ops` | raw; trimmed and case-folded by the op's pin reader | `auto`, `native`, `vendor` or a choice spelling; see [route pins](#environment-route-pins-are-raw-strings) |
| `BATCHLAS_TUNED_DIR` | `selection.tuned_dir` | raw, a directory path | a table file there (named op, type, device key, as `gemm.float.sm_89.txt`) replaces the built-in table of that name; other files are skipped with a warning |
| `BATCHLAS_EXPAND_ROUTE` | `selection.expand_route` | raw, two agreeing parsers | `expand` or `loop` |
| `BATCHLAS_GEMV_SEGT` | `selection.gemv_segt` | raw | `off`, `auto`, 2, 4, 8; must not be latched |
| `BATCHLAS_GESVD_BIDIAG` | `selection.gesvd_bidiag` | raw | changes numerics, see [below](#environment-knobs-that-override-an-explicit-argument) |
| `BATCHLAS_GETRF_LEAF` | `selection.getrf_leaf` | raw, re-read per call | `slm` or `reg` (default) |
| `BATCHLAS_GEQRF_LEAF` | `selection.geqrf_leaf` | raw, re-read per call | `auto` (default) or `reg` |
| `BATCHLAS_GETRF_LASWP` | `selection.getrf_laswp` | raw, presence latched, value re-read | `inloop`, `defer_walk`, `defer_gather` (default) |
| `BATCHLAS_GETRF_RIGHT_LASWP` | `selection.getrf_right_laswp` | raw | `walk` or `gather`; unset takes the measured crossover |
| `BATCHLAS_GETRS_LASWP` | `selection.getrs_laswp` | raw, must not be latched | `walk` or `gather` |
| `BATCHLAS_ILUK_DEVICE` | `selection.iluk_device` | raw, first char `0`/`1` only | `true`, `on`, `yes` fall through to the shape default |
| `BATCHLAS_LATRD_IMPL` | `selection.latrd_impl` | raw, re-read per call | `legacy`, `device`, `grid` |
| `BATCHLAS_ORMQR_IMPL` | `selection.ormqr_impl` | raw | only `device` has an effect |
| `BATCHLAS_ORMQR_WY` | `selection.ormqr_wy` | raw | `gemm`, `trmm`, `measured`; overrides the measured gate |
| `BATCHLAS_ORTHO_GRAM` | `selection.ortho_gram` | raw | only `gemm` has an effect |
| `BATCHLAS_SB2ST_BACK_WAVE` | `selection.sb2st_back_wave` | raw, wider disable set | fails open, see [below](#environment-sb2st-back-wave-fails-open) |
| `BATCHLAS_SB2ST_SUBGROUP` | `selection.sb2st_subgroup` | raw, case-folded | `auto`, `on`, `off`; forced on throws when kd > 32 or no sub-group 32 |
| `BATCHLAS_SYEV_TWO_STAGE_CHASE` | `selection.syev_two_stage_chase` | raw | only `givens`; read by solve and sizing query |
| `BATCHLAS_SYEVX_ALGORITHM` | `selection.syevx_algorithm` | raw | overrides `SyevxParams::method` |
| `BATCHLAS_SYEVX_PRECONDITIONER` | `selection.syevx_preconditioner` | raw | overrides a `SyevxParams` field |
| `BATCHLAS_SYEVX_BOUNDS_LEGACY` | `selection.syevx_bounds_legacy` | raw, `atoi != 0` | `true`, `on`, `yes` are all false |
| `BATCHLAS_SYEVX_FILTER_DEGREE_AUTO` | `selection.syevx_filter_degree_auto` | raw, `atoi != 0` | off by default; [evidence](../perf/syevx.md#syevx-automatic-chebyshev-filter-degree) |
| `BATCHLAS_SYEVX_INSTR_HOST` | `selection.syevx_instr_host` | raw, first char `{1,t,T,y,Y}` | `on` does not work |
| `BATCHLAS_SYEVX_PROJECTED_VENDOR` | `selection.syevx_projected_vendor` | raw, first char `{1,t,T,y,Y}` | changes the workspace size |
| `BATCHLAS_SYEVX_SOFT_LOCK` | `selection.syevx_soft_lock` | raw, inverted | `=off` and an empty export read as on |
| `BATCHLAS_SYTRD_FORCE_LOCAL_SMALL` | `selection.sytrd_force_local_small` | `env_truthy` | cannot force an unschedulable launch |
| `BATCHLAS_SYTRD_FUSE_PANEL_UPDATE` | `selection.sytrd_fuse_panel_update` | `env_truthy` and `env_falsy` | tri-state `std::optional<bool>` |
| `BATCHLAS_SYTRD_IMPL` | `selection.sytrd_impl` | raw, latched at its call site | only `device` has an effect |
| `BATCHLAS_SYTRD_TRAILING_UPDATE` | `selection.sytrd_trailing_update` | raw | `gemm`, `syr2k`, `her2k`, `rank2k`, either case; syr2k and her2k are one route |
| `BATCHLAS_LATRD_GRID_GROUPS` | `geometry.latrd_grid_groups` | `env_positive_int_or` | 0 = auto; still clamped by the residency cap |
| `BATCHLAS_LATRD_GRID_MIN_N` | `geometry.latrd_grid_min_n` | `env_positive_int_or` | 768; [evidence](../perf/syev.md#syev-latrd-grid-gate-confirmed-in-eigenvector-mode) |
| `BATCHLAS_LATRD_GRID_WG` | `geometry.latrd_grid_wg` | `env_positive_int_or` | only 32, 64, 128, 256 honoured; others silently ignored |
| `BATCHLAS_LATRD_LOWER_PANEL_WG_HINT` | `geometry.latrd_lower_panel_wg_hint` | `env_positive_int_or` (agrees with the old `atoi`) | only 64, 128, 256 take effect; device path only |
| `BATCHLAS_SB2ST_BACK_SUBS` | `geometry.sb2st_back_subs` | `env_positive_int_or` | paired with `SB2ST_BACK_TILE_W` |
| `BATCHLAS_SB2ST_BACK_TILE_W` | `geometry.sb2st_back_tile_w` | `env_positive_int_or` | column tile of the wave kernel |
| `BATCHLAS_SB2ST_BACK_TILE` | `geometry.sb2st_back_tile` | raw | the tiled kernel; 0 selects the streaming kernel |
| `BATCHLAS_POTRF_NB`, `BATCHLAS_POTRF_W` | `geometry.potrf_nb`, `potrf_w` | int, 0 = unset | type-dependent defaults at the call site |
| `BATCHLAS_SYEV_TWO_STAGE_KD` | `geometry.syev_two_stage_kd` | `env_positive_int_or` | 32, clamped to `[1, n-1]` at the call site |
| `BATCHLAS_SYEV_TWO_STAGE_SB2ST_BLOCK` | `geometry.syev_two_stage_sb2st_block` | `env_positive_int_or` | 32; read by four solve/sizing pairs |
| `BATCHLAS_SY2SB_ORMQR_NB` | `geometry.sy2sb_ormqr_nb` | raw, three-valued | unset, `off`/0 = never hint, positive forced (clamped to kd); unparseable, negative or above 1024 reads as unset |
| `BATCHLAS_SYTRD_BLOCK_SIZE` | `geometry.sytrd_block_size` | int, 0 = unset | n-bucketed and type-dependent default |
| `BATCHLAS_TRMM_TILE_M` | `geometry.trmm_tile_m` | int, 0 = unset | bucketed to 16, 32, 64, 128 |
| `BATCHLAS_TRSM_OUTER_NB` | `geometry.trsm_outer_nb` | int, 0 = unset, not latched | 128 for `Side::Left`, the CTA nb for `Side::Right` |
| `BATCHLAS_EXPAND_MAX_BYTES` | `geometry.expand_max_bytes` | raw, `strtoull` | only lowers a device-derived ceiling (`GLOBAL_MEM_SIZE / 4`) |
| `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` | `geometry.gesvd_blocked_gebrd_min` | raw, bare `atoi` | unparseable widens the path, see [below](#environment-parser-defects-recorded-not-fixed) |
| `BATCHLAS_SYEVX_CHECK_EVERY` | `geometry.syevx_check_every` | int | 4; each check is a full pipeline drain |
| `BATCHLAS_SYEVX_EXTRA_DIRECTIONS` | `geometry.syevx_extra_directions` | `std::optional<int>` | 0 = no guard block; negative reads as unset |
| `BATCHLAS_SYEVX_FILTER_DEGREE` | `geometry.syevx_filter_degree` | int, > 0 only | top of a four-level chain; disables the auto degree |
| `BATCHLAS_SYEVX_INIT_POWER` | `geometry.syevx_init_power` | `std::optional<int>` | 0 = no power iterations |
| `BATCHLAS_SYEVX_LOCK_FACTOR` | `geometry.syevx_lock_factor` | `atof`, > 0 only | 0.1; the only non-integer knob |
| `BATCHLAS_TUNE_{ORMQR_BLOCK_SIZE, GEBRD_BLOCK_SIZE, SB2ST_BACK_TILE, SB2ST_BACK_SUBS, SY2SB_ORMQR_NB, SYTRD_BLOCK_SIZE, LATRD_WG_HINT, STEDC_RECURSION_THRESHOLD, STEDC_MERGE_VARIANT, STEDC_THREADS_PER_ROOT, STEDC_WG_MULTIPLIER}` | `geometry.tune.*` | raw, `tuning_env_override` | see @ref perf_tuning |
| `BATCHLAS_QUEUE_PROFILING`, `BATCHLAS_BENCH_PROFILING` | `diagnostics.profiling` | `env_truthy`, ORed | kernel trace implies it |
| `BATCHLAS_KERNEL_TRACE`, `BATCHLAS_TRACE_KERNELS` | `diagnostics.kernel_trace` | `env_truthy`, ORed | |
| `BATCHLAS_KERNEL_TRACE_PATH`, `BATCHLAS_TRACE_PATH` | `diagnostics.kernel_trace_path` | first non-empty | default `batchlas_kernels.trace.json`; written at exit |
| `BATCHLAS_COVERAGE_OUT` | `diagnostics.coverage_out` | raw | written at exit as `<value>.<pid>` |
| `BATCHLAS_SELECT_TRACE` | `diagnostics.select_trace` | `env_truthy` | one stderr line per kernel-selection decision |
| `BATCHLAS_DEBUG_FILTER_DEGREE` | `diagnostics.debug_filter_degree` | presence only | the empty string enables it |
| `BATCHLAS_DEBUG_SYTRD_SMALL` | `diagnostics.debug_sytrd_small` | `env_truthy` | prints once |
| `BATCHLAS_GESVD_PROFILE` | `diagnostics.gesvd_profile` | `env_truthy` | drains per stage |
| `BATCHLAS_SYEVX_TRACE` | `diagnostics.syevx_trace` | `env_truthy` (identical hand-rolled set) | |
| `BATCHLAS_CTA_DEBUG_SYNC` | `diagnostics.cta_debug_sync` | `env_truthy` | not unsafe, see [below](#environment-two-knobs-that-look-unsafe-and-are-not) |
| `BATCHLAS_STEQR_CTA_CHECK` | `diagnostics.steqr_cta_check` | raw, first char `{1,t,T,y,Y}` | adds a check |
| `BATCHLAS_DUMP_BANDR1_DIR` | `diagnostics.dump_bandr1.dir` | string | `output/bandr1_dumps`; `create_directories()` is called on it |
| `BATCHLAS_DUMP_BANDR1_STEP` | `diagnostics.dump_bandr1.step` | `env_truthy` | master enable |
| `BATCHLAS_DUMP_BANDR1_ABW_ONLY` | `diagnostics.dump_bandr1.abw_only` | `env_truthy` | |
| `BATCHLAS_DUMP_BANDR1_{STEP_INDEX,SWEEP_INDEX,STEP_IN_SWEEP,BATCH}` | `diagnostics.dump_bandr1.*` | `env_int_or`, -1 = no filter | |
| `BATCHLAS_SKIP_POINTER_CHECKS` | `unsafe.skip_pointer_checks` | `!(v && *v && *v != '0')` | `=false`, `=off`, `=no` all disable the checks |
| `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` | `unsafe.latrd_grid_force_unsafe` | `env_truthy` | can deadlock |
| `BATCHLAS_BLAS_HEALTH` | `unsafe.blas_health` | `off`, `warn`, `error` | only `off` is gated |

## environment: route pins are raw strings

`Settings::routing` holds one raw `EnvValue` per op that reads a `BATCHLAS_<OP>_ROUTE` variable,
in the order of `RoutingSettings::ops`. `settings.cc` synthesises the names (`"BATCHLAS_"` +
upper-cased op + `"_ROUTE"`), so a grep for `BATCHLAS_*` string literals misses them. The strings
stay raw because the parser belongs to the selection layer, not to the settings: `src/select`
(`select::detail::pin_text` and `resolve_pin`) trims and case-folds the value, reads an empty one as
`auto`, and accepts `auto`, `native` (the best runnable non-vendor candidate), `vendor`, or a
choice spelling of that op (`lpanel:panel=8`). A spelling that does not parse, is not a compiled
candidate for the op and scalar type, or cannot run the shape throws `std::invalid_argument`;
`native` and `vendor` instead warn once and fall back to `auto` when nothing in their class can run
the call. A test pins with `ScopedPin` (`src/select/select.hh`), a thread-local slot that wins over
the environment. An op whose selection is still hand-written reads the same slot and parses its own
word list. The design and the pin rules are in `docs/design/flat-kernel-selection.md` (sections
5.3 and 12).

`RoutingSettings::route(op)` throws `std::invalid_argument` for an op that has no slot; adding an
op that reads a pin means adding its spelling to `RoutingSettings::ops`, nothing else in
`settings.cc`.

**Historical: the route vocabulary before flat kernel selection.** Until phase 5 of flat kernel
selection (`0bd26dfe`, "delete the old dispatch layer, the legacy route vocabulary and dead code"),
the routing group held two arrays indexed by `Op`: `canonical` for `BATCHLAS_<OP>_ROUTE` and
`legacy` for `BATCHLAS_{GEMM,SYMM,SYRK,SYR2K,TRMM}_VARIANT` and `BATCHLAS_{SYEV,GESVD,ORMQR}_PROVIDER`,
with the canonical spelling winning when both were set. `dispatch::parse_route_env(Op)` was the
single parser and handled three word collisions: legacy `BATCHLAS_GEMM_VARIANT=native` selected the
*vendor* path, the opposite of canonical `native`; legacy `custom` meant the fused cuBLASDx kernel
for the level-3 tile ops and the register-tiled family for gemm; syrk/syr2k legacy `gemm` selected a
deliberately wrong both-triangles baseline (`tests/route_vocabulary_tests.cc` pinned every one).
Four slots (`hemm`, `herk`, `her2k`, `iluk`) were captured but read by nothing.
`BATCHLAS_GEMM_VARIANT` had two readers with different unset defaults (`parse_route_env` defaulted
to `{Auto, Auto}`, `gemm_variant_request()` to `GemmVariantRequest::Vendor`), kept apart because
unifying them would have moved which kernel a bare `gemm()` ran. The legacy variables,
`BATCHLAS_GEMM_SYCL_KERNEL` (~38 spellings, unset was `KernelVariant::Direct`) and
`BATCHLAS_GEMM_EXPERIMENTAL` (unlocked five experimental GEMM variants) are no longer read; nor,
since the level-3 flat-selection wave deleted cuBLASDx, is `BATCHLAS_GEMM_CUBLASDX_KERNEL`
(`selection.gemm_cublasdx_kernel`, ~20 spellings); and neither are `BATCHLAS_SYEV_SMALL_KERNEL` (`cta`, `fused`, `cta_fused`, `jacobi`; its `is_set()`
separated "forced cta" from unset) or `BATCHLAS_SYEV_CTA_MAX_N` (`strtol`, rejected outside 0..32;
24 for `complex<double>`, else 32 = off; [evidence](../perf/syev.md#syev-the-lobpcg-projected-solve-knob)),
whose `Settings` fields were retired with them; see
[gemm](../perf/gemm.md) for that layer's record.

## environment: knobs that override an explicit argument

The `selection` group was added beside the four groups of the original design because these knobs
had no other home and dropping them would have left A-3 open. Every field changes which code path
runs, and three override an explicit API argument, the strongest form of the defect:

- `BATCHLAS_SYEVX_ALGORITHM` overrides `SyevxParams::method`. The asymmetry at the call site is
  load-bearing: an env-supplied algorithm degrades where an explicit request throws, and an
  unrecognised value still wins (it parses to `Auto`).
- `BATCHLAS_SYEVX_PRECONDITIONER` overrides a `SyevxParams` field; it loses to an explicit request
  and to a configured ILU(k) factor. It is documented in the public headers (`blas/enums.hh`,
  `blas/extensions.hh`).
- `BATCHLAS_GESVD_BIDIAG = bdsdc | normal | bdsqr` changes numerics, not just speed: the `normal`
  arm squares the condition number (relative error 8.5e-1 against 5.0e-1 at kappa 1e6; see
  [gesvd Tier 3](../perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver)). The call site
  deliberately overrides it with the thin-tall-U rule so a buffer-size query and its solve cannot
  disagree.

## environment: knobs read by a sizing query and its solve

Several knobs are read by a `*_buffer_size()` query as well as by the matching solve, and some
change the workspace size: `BATCHLAS_SYEV_TWO_STAGE_CHASE` (two callers),
`BATCHLAS_SYEV_TWO_STAGE_SB2ST_BLOCK` (four call sites), `BATCHLAS_SYEVX_PROJECTED_VENDOR`,
`BATCHLAS_POTRF_NB`/`_W`, and every `BATCHLAS_TUNE_*` accessor. One capture per run guarantees
they agree within a run. A `ScopedEnvVar` that straddles a sizing/solve pair, or a
`configure()` taken mid-run, would not; that is why `configure()` closes at the first `Queue`.
`potrf`'s call site was latched for exactly this reason ("read once so the sizing query and the
call agree"); `settings()` makes the property structural.

Others are deliberately re-read per call so one process can A/B two arms: `GEMV_SEGT` and
`GETRS_LASWP` (the call site forbids a latch, because a cached read makes a later `setenv`
invisible and the test passes green on the default arm), `TRSM_OUTER_NB`, `LATRD_IMPL`,
`GETRF_LEAF`, `GEQRF_LEAF`. `GETRF_LASWP` is hybrid: presence latched in a static, value re-read
per call. `SYTRD_IMPL` is latched today while its near-twin `LATRD_IMPL` deliberately is not.

## environment: geometry defaults stay at the call site

Twenty geometry knobs cannot carry a scalar default because their default is a function:
n-bucketed (every `BATCHLAS_TUNE_*`, and the sb2st/sytrd/latrd block widths), type-dependent
(`potrf`'s nb 128/96/96/64 and w 128/32/32/16 for float/double/cfloat/cdouble; syev's CTA max n
before it was retired),
argument-dependent (trsm's outer nb depends on `Side`, trmm's tile on m, syevx's extra directions
on the eigenvalue count), or device-dependent (the expansion byte budget is
`GLOBAL_MEM_SIZE / 4`). For those the field is a sentinel (0, or an unset `EnvValue`) and the
default stays at the call site. Materialising a scalar in the struct would pin a tuned curve at
one point, which is the shape of the STEDC leaf-cliff and the float-only-tuning defects already
in this repository's history. Fields whose call-site default really is a constant carry it
(`latrd_grid_min_n = 768`, `syev_two_stage_kd = 32`, `syevx_check_every = 4`,
`syevx_lock_factor = 0.1`).

`BATCHLAS_SYTRD_BLOCK_SIZE`'s complex override for 256 < n <= 512 lives at the consumer rather
than in the generated tuning header, so the next retune cannot overwrite it.

**Sentinels where 0 is meaningful.** `SYEVX_EXTRA_DIRECTIONS` (0 = no guard block, default
`max(2, k/4)`) and `SYEVX_INIT_POWER` (0 = no power iterations, default 4 when `SyevxParams` did
not supply one) are `std::optional<int>`. `SB2ST_BACK_TILE` is raw because the call site jumps to
the streaming kernel on 0, and `env_positive_int_or` maps 0 to unset: giving it the same reader as
its sibling `SB2ST_BACK_TILE_W` would silently stop `BATCHLAS_SB2ST_BACK_TILE=0` selecting the
streaming kernel, a route change with no diagnostic. `SY2SB_ORMQR_NB` is raw because it is
three-valued.

**Lock factor.** A LOBPCG column is masked once its residual is `syevx_lock_factor` times the
requested tolerance. Locking exactly at tol oscillates, because a masked column is not frozen and
drifts back above tol.

## environment: two variables for one quantity

Five `BATCHLAS_TUNE_*` names sit under a second, older variable for the same quantity, and the
outer one wins: `BATCHLAS_SYTRD_BLOCK_SIZE` over `TUNE_SYTRD_BLOCK_SIZE`,
`BATCHLAS_LATRD_LOWER_PANEL_WG_HINT` over `TUNE_LATRD_WG_HINT`, `BATCHLAS_SY2SB_ORMQR_NB` over
`TUNE_SY2SB_ORMQR_NB`, and the two sb2st back-transform knobs over their `TUNE_` twins.

The latrd pair is not unified on purpose. The call site honours `LATRD_LOWER_PANEL_WG_HINT` only
on the device path, while the legacy path asks `tuning::` with n pinned to 256; unifying would make
the outer knob live on a path it has never affected.

The sb2st pair (`SB2ST_BACK_SUBS`, `SB2ST_BACK_TILE_W`) must be forced together: forcing only one
produces a geometry that was never measured, and only tile in {1, 2, 4, 8} x subs in {4, 8, 16}
are instantiated.

## environment: sb2st back wave fails open

`BATCHLAS_SB2ST_BACK_WAVE` is the knob `env.hh` warns about. It fails open (the wave-parallel
back-transform is on unless explicitly disabled) and accepts a deliberately wider disable set than
`env_falsy`: `{0, false, off, no, n, disable, disabled}`, case-insensitively. `env_falsy`'s exact
spellings silently turned the wave path on for `False`, `Off` and `no`, which is the regression
that made the local helper exist. It must stay raw; do not route it through `env_falsy`.

`BATCHLAS_SYTRD_FUSE_PANEL_UPDATE` is the one knob that makes `env.hh`'s "an unset variable is
neither truthy nor falsy" contract load-bearing. Its call site reads the value through
`env_truthy` and `env_falsy` and needs all three answers: forced on, forced off, and "let
`tuning::sytrd_fuse_panel_update_for_n(n)` (device path) or the legacy (CUDA and n == 256) rule
decide". A value that is neither (`banana`) reads as `std::nullopt`, which is what the call site
computed before.

## environment: parser defects recorded, not fixed

Moving a read without changing its meaning means keeping known-bad parsers. Each is a behaviour
change that belongs in its own commit:

- `BATCHLAS_SYEVX_SOFT_LOCK` is inverted: it enables unless the first character is one of
  `{0, n, N, f, F}`, so `=off` evaluates true, and so does an empty export.
- `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` parses with bare `atoi` against a default of 1, so an
  unparseable value yields 0, which is lower than the default and silently widens the blocked
  gebrd path to every n instead of falling back. Routing it through `env_int_or` would change
  that to 1.
- `BATCHLAS_SKIP_POINTER_CHECKS` disables the checks for any non-empty value whose first
  character is not `0`, so `=false`, `=off` and `=no` all turn them off. `settings.cc` reproduces
  it character for character, because tightening it would silently change the meaning of three
  real spellings at the one site that uses this dialect. It is in the `unsafe` group, so the
  default build refuses it anyway.
- `BATCHLAS_TUNE_STEDC_MERGE_VARIANT` smuggles an enum through an int knob: the call site
  `static_cast`s it to `StedcMergeVariant` with no range check, and `tuning_env_override`
  rejects values <= 0, so variant 0 is unreachable from the environment. One merge variant is
  known to deadlock on some hardware.

## environment: files written from the environment

Three diagnostics knobs open a filesystem path for writing inside library code, two of them from
an `atexit` handler: the kernel trace writes `kernel_trace_path`, the selection coverage writes
`$BATCHLAS_COVERAGE_OUT.<pid>`, and the band-reduction dump calls `create_directories()` under
`$BATCHLAS_DUMP_BANDR1_DIR`. An embedding application that inherits a hostile environment gets
directories created and files written at a path it never chose; `configure()` is what lets it
clear these before it starts work. `BATCHLAS_COVERAGE_OUT` used to be read at two lifetimes
(static initialisation decides whether coverage is on, the `atexit` handler re-read it to build the
file name), a pair that could disagree; one capture removed that. The six `DUMP_BANDR1` knobs read
in one function are grouped into one struct so clearing the family is one assignment.

## environment: the unsafe group and its gate

Membership in `Settings::unsafe` is not "the knob is scary". It is "setting this can make a
correct program crash, hang, or silently compute wrong numbers, and nothing else in the process
will say so". Three variables qualify:

- `BATCHLAS_SKIP_POINTER_CHECKS` disables the one-USM-query-per-pointer-argument check (about
  70 ns, noise against a kernel launch) that turns "you passed host memory to a device call" into
  a thrown `invalid_argument` naming the argument. Without it, host memory reaches the device as a
  wild address: `CUDA_ERROR_ILLEGAL_ADDRESS`, then a `SIGABRT` from inside the CUDA runtime during
  teardown that no catch block can stop. The identical code is correct on the host backend, so a
  CPU prototype passes and the GPU run dies.
- `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` is a deadlock, not a wrong answer. It replaces the residency
  cap (compute units / batch) with the forced group count from `BATCHLAS_LATRD_GRID_GROUPS`,
  putting more work-groups on a matrix than are guaranteed co-resident, and the grid path's
  barrier is a sense-reversing spin barrier whose termination argument is exactly that
  co-residency. The kernel hangs, and a hang there looks exactly like slow JIT: run forced-unsafe
  measurements under `timeout`. The escape hatch is largely self-defeating anyway: at batch >= 128
  the cap had already clamped the count to 1.
- `BATCHLAS_BLAS_HEALTH=off` skips the host-dgemm probe, the only thing standing between the user
  and a host BLAS that computes dgemm incorrectly (the known-broken Cooperlake OpenBLAS kernel).
  With the probe off, every double and `complex<double>` result from the host/NETLIB backend is
  silently wrong by O(1). This one was not in the original brief's list; it was found during the
  migration.

**The gate.** When the library is built without the CMake option `BATCHLAS_ALLOW_UNSAFE_ENV`
(default OFF), these fields keep their safe values whatever the environment says, and each
variable that was set produces one warning on stderr naming the variable and the option. The gate
refuses the unsafe *direction* of a field, not every value that differs from the default:
`BATCHLAS_BLAS_HEALTH=error` is stricter than the default `warn` and is let through; only `off` is
refused.

`configure()` is not gated: an embedding application that sets one of these in code has made a
choice, which is exactly the affordance A-3 says is missing. The gate is about ambient process
state.

## environment: two knobs that look unsafe and are not

The original brief listed two more variables under "safety". Both stay in `diagnostics`:

- `BATCHLAS_CTA_DEBUG_SYNC` prints `[cta-debug] <stage>` and calls `wait_and_throw()` at each
  stage. It removes no check, changes no numerics and cannot select a different kernel; the only
  behavioural difference is that an async SYCL exception surfaces earlier and at a named stage,
  which is strictly safer. Its cost is a full pipeline drain per stage. Padding the unsafe group
  with it would misrepresent what that group gates.
- `BATCHLAS_STEQR_CTA_CHECK` is the opposite of unsafe: it adds a check. Set, `steqr_cta` waits,
  scans the per-item status array and throws if any item did not converge; unset,
  non-convergence is silent. The unsafe condition is the default, not the variable, so forcing it
  to its default under a locked-down build would entrench silent non-convergence.

## environment: configure sets the base, not a lock

`configure()` is permitted until the first `Queue` is constructed (`Queue`'s constructors call
`detail::note_queue_constructed()`); afterwards it throws `batchlas::api_misuse`, a
`std::runtime_error`, and changes nothing. The deadline exists because routing and geometry knobs
are read by `*_buffer_size()` queries as well as by the matching solve, and several change the
workspace size, so a change taken mid-run would let two calls in one process disagree about how
much scratch a solve needs. The header's earlier note that it throws a plain `std::runtime_error`
"because a real exception hierarchy is separate work" was superseded when the hierarchy landed
(see @ref design_error_model).

What `configure()` installs is the base, not a lock: `detail::reload_settings()` re-reads the
environment on top of whatever `configure()` last set, rather than on top of the defaults. A knob
set in code and not exported therefore holds for the life of the process; a knob something
explicitly exports still wins. Both halves are deliberate, and each is a defect the other way
round. Rebuilding from the defaults discarded the `configure()` call entirely: a `ScopedEnvVar` on
an unrelated variable anywhere in an embedding application's own harness was enough to do it,
silently. Ignoring the environment afterwards made every `ScopedEnvVar` in the process a no-op,
which turns an A/B test into two runs of the same arm that agree by construction.

## environment: what a settings reload does not cover

`detail::reload_settings()` is what keeps the test suite green: fifteen test files pin a route or
a geometry knob with `ScopedEnvVar` and expect the next library call to see it, so `ScopedEnvVar`
calls it from its constructor and its destructor. Without the destructor call a test's knob would
leak into every later test in the same process. Two cases it does not cover, both recorded rather
than papered over:

- A test or benchmark that calls `::setenv` directly instead of using `ScopedEnvVar` does not
  reach it and reads the pre-existing value. Several did when this landed; those call sites have
  to move to `ScopedEnvVar`.
- A reload between a `*_buffer_size()` query and its solve can change a block width and under-size
  the workspace the caller already allocated. The hazard predates `settings.hh` (those knobs were
  re-read per call) and is unchanged by it.

It is not thread-safe with respect to concurrent `settings()` readers, for the same reason
`ScopedEnvVar` is not: the process environment is not thread-safe either.

## environment: the shared parsers in env.hh

`<batchlas/util/env.hh>` holds the shared readers. Six byte-identical truthiness parsers had
accumulated across `src/` (`queue.hh`, `kernel-trace.hh`, `syev_cta.cc`, `sytrd_blocked.cc`,
`gesvd_blocked.cc`, `band_reduction.cc`), plus four copy-pasted `stoi`/try-catch blocks in
`band_reduction.cc`. The accepted spellings are exactly what all of them accepted, so
consolidating was semantics-preserving, and the exact-spelling lists are the contract: widening
them would silently change how every existing call site reads the same strings. A knob that wants
`Off`/`No` has to case-fold its own value (as `sytrd_sb2st_hh.cc` does); do not fold that helper
back into `env_falsy`.

`env_truthy` and `env_falsy` take the *value* of a variable. A name-taking overload
`env_truthy(const char* name, bool fallback)` was once also declared and has been removed: it gave
one function name two incompatible readings of the same `const char*` parameter, so
`env_truthy("BATCHLAS_DEBUG")` silently selected the value form and evaluated the name as a value,
which is always false. It had zero call sites. Renaming the value forms (`env_value_truthy`) was
rejected because it churns every call site to defend against an overload nothing uses.

`env_positive_int_or` is the shape every kernel-geometry knob in `src/extensions` wants: a forced
work-group count, tile width or sub-group count is meaningless at zero or below, and the call sites
all fall back to the computed default there. Those sites had each grown their own `atoi` plus
`> 0` parser; naming the pattern keeps the clamp visible instead of re-derived, or dropped, when a
site is routed through the plain `env_int_or`.

`env.hh` declares `detail::reload_settings()` instead of including `settings.hh`, because
`env.hh` is reached by nearly every device translation unit and a declaration is all
`ScopedEnvVar` needs. (The original reason, that `settings.hh` pulled in the route vocabulary, went
away with the legacy routing arrays; `settings.hh` now includes only standard headers and
`export.hh`.)

## environment: ScopedEnvVar

Sixteen near-identical copies of this RAII class had accumulated across `tests/` and
`benchmarks/` (one per op whose route is pinned by an env knob), plus a single-key variant in
`syevx_tests.cc` and two hand-rolled save/setenv/restore blocks in `syev_blocked_tests.cc`. They
agreed on everything except one point: three treated a null value as "unset for the duration" and
the other thirteen passed it straight to `setenv`, which is undefined behaviour. The null-aware
reading is a strict superset, so it is the one kept.

- `setenv`/`unsetenv` rather than `putenv`: `putenv` keeps the caller's buffer alive inside the
  environment, a lifetime trap in a scope that restores on exit.
- The name is borrowed, not copied, and must outlive the object (in practice a string literal).
  Two of the gemm-family benchmarks construct these inside the timed lambda, and a `std::string`
  member would put a heap allocation per construction into a measured region.
- Copying is deleted: it would restore the same variable twice, the second time from a snapshot
  the first restore has invalidated.
- Nested instances restore innermost first, so each destructor's reload leaves the settings
  agreeing with the environment the enclosing scope still has pinned.
- Not thread-safe, because the process environment is not.
