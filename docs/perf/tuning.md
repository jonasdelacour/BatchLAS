# Tuning constants {#perf_tuning}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda

`include/batchlas/tuning_params.hh` holds the kernel parameters that depend on problem order n.
The shipped values come from the 2026-08-07 retune (`924b3a59`). Every constant was measured on
sm_89, in `float`, through `evaluation/tuning/`: treat them as hypotheses on other GPUs and for
other scalar types. Per-op sweep numbers live on the op pages linked in
[where the per-op sweeps live](#tuning-where-the-per-op-sweeps-live).

## tuning: what the constants are

Each tunable quantity has five constants, one per size bucket of n, and a constexpr table function
`<knob>_default_for_n(n)` that picks one:

| bucket | n |
| --- | --- |
| `TINY` | n <= 64 |
| `SMALL` | 64 < n <= 128 |
| `MEDIUM` | 128 < n <= 256 |
| `LARGE` | 256 < n <= 512 |
| `XLARGE` | n > 512 |

| family | consumer | meaning of 0 |
| --- | --- | --- |
| `ORMQR_BLOCK_SIZE_*` | `ormqr` WY block width | n/a |
| `GEBRD_BLOCK_SIZE_*` | `gebrd` panel width in `gesvd` | n/a |
| `SB2ST_BACK_TILE_*`, `SB2ST_BACK_SUBS_*` | sb2st wave back-transform geometry | keep the shape-adaptive heuristic in `sytrd_sb2st_hh.cc` (all ship 0) |
| `SY2SB_ORMQR_NB_*` | WY width of the sy2sb panel back-transform | keep the shape gate in `sytrd_sy2sb.cc` |
| `SYTRD_BLOCK_SIZE_*` | `sytrd` panel width | n/a |
| `LATRD_LOWER_PANEL_WG_HINT_*` | latrd lower-panel work-group size | no hint |
| `SYTRD_FUSE_PANEL_UPDATE_*` | fused panel update on/off | off |
| `STEDC_RECURSION_THRESHOLD_*` | stedc leaf size | n/a; 32 is the CTA invariant, not a tuning result |
| `STEDC_MERGE_VARIANT_*` | `StedcMergeVariant` (1 Fused, 2 FusedCta) | `Auto`, never returned (see below) |
| `STEDC_THREADS_PER_ROOT_*`, `STEDC_WG_MULTIPLIER_*` | stedc secular-solve geometry | n/a |

Production code calls the runtime accessors `<knob>_for_n(n)`. Each consults its `BATCHLAS_TUNE_*`
override first and falls back to the table. `latrd_lower_panel_wg_hint()` pins n to 256 for the
legacy path. `sytrd_fuse_panel_update_for_n` has no override; its control is
`BATCHLAS_SYTRD_FUSE_PANEL_UPDATE`, which is tri-state (@ref design_environment).

## tuning: how the constants were measured

The harness (@ref tuning_harness) grid-searches benchmark executables per bucket and writes a JSON
profile. `evaluation/tuning/generate_tuning_header.py` turns per-case winners into bucket constants.
A retune takes about 12 minutes. Every group is A/B'd end to end at its consumers before adoption;
see [the 2026-08-07 constant retune](syev.md#syev-the-2026-08-07-constant-retune).

- **Tune through the consumer.** stedc's threads-per-root and work-group multiplier are chosen
  through `syev`. stedc's own benchmark measures the merge in isolation and disagrees with its
  consumer.
- **A constant that the heuristic already matches stays 0.** The sb2st back-transform geometry
  spans 3.6x across its grid at n = 1024, but the shape-adaptive heuristic picks the winner to
  within 0.6%. Freezing it into buckets would only lose adaptivity. The sb2st bench stays in the
  default space as a tripwire.
- **A grid must contain the shipped value.** Otherwise a retune "wins" with the best of a grid that
  never contained the incumbent. stedc's `wg_multiplier` once swept `[1,2,4]` while 8 shipped.

## tuning: the float-only caveat

Tuning was float-only. The constants apply unchanged to `double`, `complex<float>` and
`complex<double>`. When another type looks bad, sweep that type first. Per-type corrections live at
the consumer, where a retune cannot overwrite them:

- `sytrd_block_size_default<T>` returns 32 for complex at 256 < n <= 512. The float value 8 costs
  complex 1.16x to 1.20x ([evidence](syev.md#syev-complex-panel-width-in-the-256-to-512-bucket)).
- The sb2st back-transform geometry is selected per type at n = 512
  ([evidence](syev.md#syev-per-type-sb2st-back-transform-geometry-wp4)).

Before trusting "no change" from a knob, check for a second consumer (aliasing) or a bypass on the
measured path (shadowing). See [knobs that shadow each other](#tuning-knobs-that-shadow-each-other).

## tuning: runtime overrides of the constants

Each runtime accessor passes its captured `settings().geometry.tune.<field>` and the table value to
`detail::tuning_env_override`. One process can therefore A/B a retune without a rebuild. The
eleven variables are `BATCHLAS_TUNE_{ORMQR_BLOCK_SIZE, GEBRD_BLOCK_SIZE, SB2ST_BACK_TILE,
SB2ST_BACK_SUBS, SY2SB_ORMQR_NB, SYTRD_BLOCK_SIZE, LATRD_WG_HINT, STEDC_RECURSION_THRESHOLD,
STEDC_MERGE_VARIANT, STEDC_THREADS_PER_ROOT, STEDC_WG_MULTIPLIER}`.

`BATCHLAS_SY2SB_ORMQR_NB` is not a `BATCHLAS_TUNE_*` variable. It is three-valued: `0` or `off`
means never hint, and values above 1024 are ignored (`sy2sb_ormqr_nb_env` in
`src/extensions/sytrd_sy2sb.cc`). `BATCHLAS_SYEV_TWO_STAGE_KD` goes through `env_positive_int_or`
in `src/util/settings.cc`, which accepts `"16x"`.

- **Parser.** `strtol` with an explicit reject: unset, empty, unparseable, trailing garbage,
  non-positive or above `INT32_MAX` all return the compiled constant. This is stricter than
  `env_int_or`, which reads `"16x"` as 16. A retune harness types exactly such inputs, so the
  parser stays in the tuning header.
- **Freshness.** The value is read from the captured settings on every call and never cached.
  `ScopedEnvVar` re-reads the snapshot (`detail::reload_settings()`). A raw `setenv` is not seen.
- **Hazard.** `*_buffer_size()` queries consult the accessors as well as the solve. Do not change a
  variable between a buffer-size query and its call, or the workspace desynchronises from the block
  width. Flip variables between runs.
- **0 means no opinion.** `SB2ST_BACK_TILE`, `SB2ST_BACK_SUBS` and `SY2SB_ORMQR_NB` return 0 to keep
  the call-site heuristic. A non-positive environment value counts as unset, so 0 is reachable only
  from the compiled constant. The variables force a geometry; they cannot force auto.
- **Merge variant.** Callers cast the result to `StedcMergeVariant` with no range check. 0 is
  `Auto`, which would re-enter tuning. Non-positive values fall back to the constant, so variant 0
  is unreachable from the environment.

Five of these sit under an older, outer variable for the same quantity, which wins. See
[two variables for one quantity](../design/environment.md#environment-two-variables-for-one-quantity).

## tuning: regenerating the header

The constants are generated. The accessors are hand-maintained and mirrored in the template
`_emit_header` in `evaluation/tuning/generate_tuning_header.py`. Any change to an accessor, to
`tuning_env_override`, or to the `#include <batchlas/settings.hh>` that brings in `EnvValue` must
land in the template too. The Doxygen text and evidence pointers are carried verbatim by the
template, so header comment edits must be copied into it.

The template is the committed header with each constant replaced by its placeholder, plus one line,
`// Source profile: <path>`, after the "GENERATED" banner. Regenerating with the committed
constants reproduces the header exactly apart from that line. Check a template edit by regenerating
with the committed constants and diffing.

> **Warning:** the template is a Python f-string. Write C++ character escapes with doubled
> backslashes: a single `'\0'` becomes a raw NUL, which makes git treat the header as binary.

**The committed header wins.** CMake no longer generates `tuning_params.hh`. `cmake/tuning_params.h.in`
and the `batchlas_tuning_header` target were removed on 2026-10-07, because `include/` precedes the
build tree on the include path and the committed file shadowed the generated one. A retune takes
effect only when its constants are ported into the committed header by hand
([evidence](syev.md#syev-the-shadowed-tuning-header)).

## tuning: knobs that shadow each other

- `SY2SB_ORMQR_NB_*` shadows `ORMQR_BLOCK_SIZE_*` on `syev`'s hot path. With the sy2sb shape gate
  active (n >= 1024 and batch >= 32 returns kd), changing the ormqr constant leaves the kernel trace
  identical. A positive `SY2SB_ORMQR_NB` is clamped to kd at the call site.
- `GEBRD_BLOCK_SIZE_*` was split from `ORMQR_BLOCK_SIZE_*` on 2026-08-06. `gesvd` used the ormqr
  constant for its bidiagonal reduction, which has the opposite gradient: flat for gebrd, 2.16x
  between 16 and 56 for ormqr.
- Only the instantiated sb2st back-transform pairs exist: tile in {1, 2, 4, 8} x subs in {4, 8, 16}.
  Any other pair silently falls through to the slower tiled kernel, so a grid must stay in that set.

## tuning: where the per-op sweeps live

| constants | evidence |
| --- | --- |
| `ORMQR_BLOCK_SIZE_*`, `SYTRD_BLOCK_SIZE_*`, `SY2SB_ORMQR_NB_*`, `SB2ST_BACK_*` | [syev: the 2026-08-07 constant retune](syev.md#syev-the-2026-08-07-constant-retune), [the 2026-08-03 block-size sweep](syev.md#syev-the-2026-08-03-block-size-sweep-in-the-syev-context) |
| `GEBRD_BLOCK_SIZE_*` | [gesvd: the gebrd panel-width split from ORMQR](gesvd.md#gesvd-the-gebrd-panel-width-split-from-ormqr) |
| `STEDC_MERGE_VARIANT_*`, `STEDC_THREADS_PER_ROOT_*`, `STEDC_WG_MULTIPLIER_*`, `STEDC_RECURSION_THRESHOLD_*` | [stedc: current tuning values](stedc.md#stedc-current-tuning-values) |
| `LATRD_LOWER_PANEL_WG_HINT_*`, `SYTRD_FUSE_PANEL_UPDATE_*` | none recorded; see [tuning: open debts](#tuning-open-debts) |

## tuning: open debts

- `LATRD_LOWER_PANEL_WG_HINT_LARGE = 128` (the only non-zero hint) and the all-zero
  `SYTRD_FUSE_PANEL_UPDATE_*` have no recorded sweep. The hint came in with `924b3a59`, whose
  message does not quantify it, and its harness profile is not in `benchmarks/results/`. Re-measure
  through `syev` before relying on either.
- No double or complex sweep exists for any bucket, apart from the per-type corrections in
  [the float-only caveat](#tuning-the-float-only-caveat).
