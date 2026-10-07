# Tuning constants: tuning_params.hh and its overrides {#perf_tuning}

> **Covers:** what the constants in `include/batchlas/tuning_params.hh` are, how they were
> measured, the float-only caveat, the `BATCHLAS_TUNE_*` runtime overrides and their parser, and
> how to regenerate the header without losing the hand-maintained parts. Per-op sweep numbers
> live on the op pages linked from [where the sweeps live](#tuning-where-the-per-op-sweeps-live).
> **Status:** current.
> **Machine:** RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda.
> **Measured:** the shipped constants come from the 2026-08-07 retune (`924b3a59`), with the
> gebrd split of 2026-08-06.

Every constant in the header was measured on sm_89, in `float`, through the harness in
`evaluation/tuning/` (@ref tuning_harness). Treat them as hypotheses on any other GPU and for any
other scalar type.

## tuning: what the constants are

Each tunable quantity has five constants, one per size bucket of the problem order n, and a
constexpr table function `<knob>_default_for_n(n)` that picks one:

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

Production code calls the runtime accessors `<knob>_for_n(n)` (and `latrd_lower_panel_wg_hint()`,
which pins n to 256 for the legacy path); each consults its `BATCHLAS_TUNE_*` override first and
falls back to the table. `sytrd_fuse_panel_update_for_n` has no override: its runtime control is
`BATCHLAS_SYTRD_FUSE_PANEL_UPDATE`, which is tri-state (@ref design_environment).

## tuning: how the constants were measured

The harness grid-searches benchmark executables per bucket and writes a JSON profile;
`evaluation/tuning/generate_tuning_header.py` turns per-case winners into bucket constants. A
retune takes about 12 minutes. The 2026-08-07 retune ran on a trimmed space and every group was
A/B'd end to end at its consumers before adoption; see
[the 2026-08-07 constant retune](syev.md#syev-the-2026-08-07-constant-retune) for the before and
after table.

Two rules that came out of that retune and bind the next one:

- **Tune through the consumer.** stedc's threads-per-root and work-group multiplier are chosen
  through `syev`, overruling stedc's own benchmark, which measures the merge in isolation and
  disagrees with its consumer.
- **A constant that the heuristic already matches stays 0.** The sb2st back-transform geometry
  (the wave-parallel path, about 53% of `syev` at n = 1024, as recorded in the header before
  2026-09-30) spans 3.6x across its grid at n = 1024, but the shape-adaptive heuristic already picks the
  winner to within 0.6%, so freezing it into buckets would only lose adaptivity on unmeasured
  shapes. The sb2st bench stays in the default space as a tripwire.

## tuning: the float-only caveat

Tuning was float-only. The header is regenerated from float benchmarks, and the constants are
applied to `double`, `complex<float>` and `complex<double>` unchanged. When another type looks
bad, sweep that type first. Per-type corrections live at the consumer, where the next retune
cannot overwrite them: for example `sytrd_block_size_default<T>` returns 32 for complex at
256 < n <= 512, where the float value of 8 costs complex 1.16x to 1.20x
([evidence](syev.md#syev-complex-panel-width-in-the-256-to-512-bucket)), and the sb2st
back-transform geometry is selected per type at n = 512
([evidence](syev.md#syev-per-type-sb2st-back-transform-geometry-wp4)).

Before trusting "no change" from a knob, check whether it has a second consumer (aliasing) or is
bypassed on the path you measured (shadowing); see [shadowing](#tuning-knobs-that-shadow-each-other).

## tuning: runtime overrides of the constants

Each runtime accessor passes its captured `settings().geometry.tune.<field>` and the table value to
`detail::tuning_env_override`, so one process can A/B a retune without a rebuild. The eleven
variables are `BATCHLAS_TUNE_{ORMQR_BLOCK_SIZE, GEBRD_BLOCK_SIZE, SB2ST_BACK_TILE, SB2ST_BACK_SUBS,
SY2SB_ORMQR_NB, SYTRD_BLOCK_SIZE, LATRD_WG_HINT, STEDC_RECURSION_THRESHOLD, STEDC_MERGE_VARIANT,
STEDC_THREADS_PER_ROOT, STEDC_WG_MULTIPLIER}`.

The contract below was originally described as matching `BATCHLAS_SY2SB_ORMQR_NB` and
`BATCHLAS_SYEV_TWO_STAGE_KD`. As of 2026-10-06 neither does: `BATCHLAS_SY2SB_ORMQR_NB` is
three-valued (`0` or `off` means "never hint", values above 1024 are ignored;
`sy2sb_ormqr_nb_env` in `src/extensions/sytrd_sy2sb.cc`), and `BATCHLAS_SYEV_TWO_STAGE_KD` goes
through `env_positive_int_or` in `src/util/settings.cc`, which accepts `"16x"`. The
`BATCHLAS_TUNE_*` contract is:

- **Parser.** `strtol` with an explicit reject: unset, empty, unparseable, trailing garbage,
  non-positive or above `INT32_MAX` all return the compiled constant, bit-for-bit the
  no-override behaviour. This is stricter than `env_int_or`'s `stoi`-in-a-try, which reads
  `"16x"` as 16, and those are exactly the inputs a retune harness types, which is why the parser
  stays in the tuning header rather than moving into `settings.cc`.
- **Freshness.** The value is taken from the captured settings on every call, never cached in the
  accessor. `ScopedEnvVar` re-reads the snapshot on construction and destruction
  (`detail::reload_settings()`), so a harness that flips a knob the supported way sees it. A raw
  `setenv` is not seen.
- **Hazard.** Every accessor is consulted by `*_buffer_size()` queries as well as by the matching
  solve. The variable must not change between a buffer-size query and its call, or the allocated
  workspace desynchronises from the block width actually used. Flip it between runs, not inside
  one.
- **0 means "no opinion".** `SB2ST_BACK_TILE`, `SB2ST_BACK_SUBS` and `SY2SB_ORMQR_NB` return 0 to
  keep the call site's heuristic. Because a non-positive environment value is treated as unset, 0
  is reachable only from the compiled constant: the variables exist to force a geometry, not to
  force auto.
- **The merge variant is an enum through an int.** Callers cast the result to
  `StedcMergeVariant` with no range check. 0 is `StedcMergeVariant::Auto`, which would re-enter
  tuning resolution, so 0 (like every non-positive value) falls back to the constant and variant 0
  is unreachable from the environment.

Five of these sit under an older, outer variable for the same quantity, which wins; see
[two variables for one quantity](../design/environment.md#environment-two-variables-for-one-quantity).

## tuning: regenerating the header

The constants are generated; the accessors are hand-maintained and mirrored in the generator's
template (`_emit_header` in `evaluation/tuning/generate_tuning_header.py`). Any change to an
accessor, to `tuning_env_override`, or to the `#include <batchlas/settings.hh>` that brings in
`EnvValue` must land in the template too, or the next retune emits a header that no longer
compiles or silently reverts the change. The same holds for the Doxygen documentation and the
evidence pointers in the header: the template carries them verbatim, so a comment edit in the
header must be copied into the template too.

As of 2026-10-06 the template is the committed header with each constant replaced by its
placeholder, plus one extra line, `// Source profile: <path>`, after the "GENERATED" banner.
Regenerating with the committed constants reproduces the header exactly apart from that line.
The rebuild also fixed a latent defect: the template is a Python f-string, and it spelled the
C++ character literal `'\0'` with a single backslash, which Python turns into a raw NUL byte. Two
NULs reached every regenerated header (inside `tuning_env_override`). DPC++ clang still compiles
them, with a `-Wnull-character` warning per literal ("null character(s) preserved in char
literal"), but the file stops being plain text: git treats it as binary and diffs nothing. The
committed header never showed it because retunes were ported by hand (see "The committed header wins", below). Check a template edit by
regenerating with the committed constants and diffing.

**The committed header wins.** CMake used to generate a `tuning_params.hh` into the build tree from
`cmake/tuning_params.h.in`, but `include/` precedes the build tree on the include path, so the
committed file shadowed it on every build and the CMake `batchlas_tuning_header` target was a no-op;
both were removed on 2026-10-07.
A retune takes effect only when its constants are ported into the committed header by hand
([evidence](syev.md#syev-the-shadowed-tuning-header)).

**A grid must contain the shipped value**, or a retune "wins" with the best of a grid that never
contained the incumbent (stedc's `wg_multiplier` once swept `[1,2,4]` while 8 shipped). See
@ref tuning_harness for this and the other harness traps.

## tuning: knobs that shadow each other

- `SY2SB_ORMQR_NB_*` shadows `ORMQR_BLOCK_SIZE_*` on `syev`'s hot path: with the sy2sb shape gate
  active (n >= 1024 and batch >= 32 returns kd), changing the ormqr constant leaves the kernel
  trace identical. A positive `SY2SB_ORMQR_NB` is clamped to kd at the call site.
- `GEBRD_BLOCK_SIZE_*` was split out of `ORMQR_BLOCK_SIZE_*` on 2026-08-06. `gesvd` had used the
  ormqr constant to size its bidiagonal reduction, an unrelated kernel with the opposite gradient
  (flat for gebrd, 2.16x between 16 and 56 for ormqr), so one knob pinned the steep parameter at
  the flat one's optimum.
- Only the instantiated sb2st back-transform pairs exist: tile in {1, 2, 4, 8} x subs in
  {4, 8, 16}. Any other pair silently falls through to the slower tiled kernel, so a tuning grid
  must not leave that set.

## tuning: where the per-op sweeps live

| constants | evidence |
| --- | --- |
| `ORMQR_BLOCK_SIZE_*`, `SYTRD_BLOCK_SIZE_*`, `SY2SB_ORMQR_NB_*`, `SB2ST_BACK_*` | [syev: the 2026-08-07 constant retune](syev.md#syev-the-2026-08-07-constant-retune), [the 2026-08-03 block-size sweep](syev.md#syev-the-2026-08-03-block-size-sweep-in-the-syev-context) |
| `GEBRD_BLOCK_SIZE_*` | [gesvd: the gebrd panel-width split from ORMQR](gesvd.md#gesvd-the-gebrd-panel-width-split-from-ormqr) |
| `STEDC_MERGE_VARIANT_*`, `STEDC_THREADS_PER_ROOT_*`, `STEDC_WG_MULTIPLIER_*`, `STEDC_RECURSION_THRESHOLD_*` | [stedc: current tuning values](stedc.md#stedc-current-tuning-values) |
| `LATRD_LOWER_PANEL_WG_HINT_*`, `SYTRD_FUSE_PANEL_UPDATE_*` | none recorded; see open debts |

## tuning: open debts

- `LATRD_LOWER_PANEL_WG_HINT_LARGE = 128` (the only non-zero hint) and the all-zero
  `SYTRD_FUSE_PANEL_UPDATE_*` have no sweep recorded on any evidence page. The hint went from 0 to
  128 in `924b3a59` (the 2026-08-07 retune), whose message does not quantify it; the harness
  profile it came from is not in `benchmarks/results/`. Re-measure through `syev` before relying
  on either.
- Every constant is float-only and sm_89-only. No double or complex sweep exists for any bucket
  except the per-type corrections named in [the float-only caveat](#tuning-the-float-only-caveat).
