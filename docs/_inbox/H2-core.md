# Inbox from shard H2-core

## -> docs/perf/gesvd.md: gesvd: the gebrd panel-width split from ORMQR

This heading already exists (added by the gesvd shard in this pass). It lacks the nb sweep and
the per-n retune winners below, which were in the header comment; please add them to that
section rather than creating a new heading.

`GEBRD_BLOCK_SIZE_*` (gebrd's panel width in `gesvd`) was split out of `ORMQR_BLOCK_SIZE_*` on
2026-08-06. `gesvd` had used the ormqr constant to size its bidiagonal reduction, an unrelated
kernel, and the two have opposite gradients. Measured on `gesvd_blocked` n = 512, batch = 256,
through the `gesvd.gebrd` stage timer (RTX 4090, CUDA, float), nb against time:

| nb | 8 | 12 | 16 | 24 | 32 | 48 |
| --- | --- | --- | --- | --- | --- | --- |
| gesvd.gebrd (ms) | 234.9 | 232.1 | 230.7 | 235.5 | 240.9 | 256.6 |

gebrd's optimum is 16 at n = 512 and its curve is flat there (within 2%); ormqr's is steep
(2.16x between 16 and 56). Sharing one knob pinned the steep parameter at the flat one's optimum.

Retuned 2026-08-07 (`924b3a59`) with the knobs separate: the small sizes want 8, not 16.
Per-case winners at jobu = jobvh = 0, where `gesvd.gebrd` is about 95% of the call: n = 128: 8,
n = 256: 8, n = 512: 16, n = 1024: 16. The axis is worth 3.5-48% depending on n, so it is not a
flat parameter everywhere. Shipped: `GEBRD_BLOCK_SIZE_{TINY,SMALL,MEDIUM,LARGE,XLARGE}` =
16, 8, 8, 16, 16.

(Source: the comment above `GEBRD_BLOCK_SIZE_*` in `include/batchlas/tuning_params.hh` before
this pass.)

Code sites that now point here:
- `include/batchlas/tuning_params.hh` (comment above `GEBRD_BLOCK_SIZE_TINY`):
  `evidence: docs/perf/gesvd.md#gesvd-the-gebrd-panel-width-split-from-ormqr`

## For whoever owns evaluation/tuning/generate_tuning_header.py (no page)

`include/batchlas/tuning_params.hh` now carries Doxygen API docs (`@file`, per-accessor
`@brief`/`@ingroup config`) and short trap comments with `evidence:` pointers; its lab-notebook
blocks (gebrd sweep, sb2st heuristic-vs-tuned table, stedc merge-variant history, the syev-side
threads-per-root/wg-multiplier table) are gone. The generator's `_emit_header` template carries
only short comments, so a regeneration would drop the API docs and pointers. Mirror the header's
comments into the template (comments only; the accessors and constants already match). This
completes item 1 of `A4-tridiag.md` on the generator side.

## For whoever owns docs/cpp-api.md (no code pointer)

- "Setting it programmatically": `configure()` after the first `Queue` throws
  `batchlas::api_misuse` (`src/util/settings.cc:375`), which is a `std::runtime_error`. The
  page says `std::runtime_error`; true but less precise than the rest of the page's error model.
- "Devices and queues": the string `Device` constructor throws `batchlas::device_error` when no
  device of the type exists and `batchlas::invalid_argument` for an unknown type string; the page
  says `std::runtime_error`.
- "Synchronisation and threading": the wrong-thread guard throws `batchlas::api_misuse`
  (`src/queue.hh:47`); the page says `std::runtime_error`.

## For whoever owns docs/design/known-defects.md (no code pointer)

Three latent environment-parser defects are recorded (not fixed) in
`docs/design/environment.md#environment-parser-defects-recorded-not-fixed`:
`BATCHLAS_SYEVX_SOFT_LOCK` is inverted (`=off` and an empty export read as on),
`BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` reads an unparseable value as 0 and widens the blocked-gebrd
path to every n, and `BATCHLAS_TUNE_STEDC_MERGE_VARIANT` is cast to the enum with no range check.
Consider an "At a glance" row linking there.

## For whoever owns include/batchlas/blas/linalg.hh / docs/design/build-performance.md

`include/batchlas/sycl_interop.hh` and `include/batchlas/settings.hh` used to repeat the
"~4.1 s per consumer TU" cost of pulling `<sycl/sycl.hpp>` into a public header; they now say
"see the note at the top of blas/linalg.hh" without the number (the number stays in
`blas/linalg.hh`). Once `docs/design/build-performance.md#build-performance-the-umbrella-header-excludes-device-code`
exists, those two one-line comments can cite it instead.

## For whoever owns the section index pages (docs/pages/architecture.md, docs/perf/README.md)

New pages from this shard, to be linked with `@subpage`:
- `@ref design_error_model` "Error model" (architecture; suggested section "Core runtime")
- `@ref design_environment` "Environment and settings" (architecture)
- `@ref design_symbol_visibility` "Symbol visibility" (architecture; or "Build")
- `@ref design_workspace` "Workspace arena and sizing" (architecture)
- `@ref perf_tuning` "Tuning constants" (perf evidence index)
