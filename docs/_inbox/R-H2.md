# Inbox from shard R-H2 (review of H2-core)

## -> docs/design/flat-kernel-selection.md: (no new heading; correction to section 5.3)

Section 5.3 says "`BATCHLAS_<OP>_ROUTE` is read on every call (factor_bench's ArmEnv sets it per
arm with setenv)". The value is read from `settings().routing`, the snapshot captured once by
`src/util/settings.cc`; `select::detail::pin_text` (`src/select/select.cc`) reads that snapshot
on every call, but a raw `::setenv` is not seen until `detail::reload_settings()` runs
(`ScopedEnvVar` triggers it). A harness that sets the variable with plain `setenv` between arms
inside one process gets the first arm's value for every arm. Worth checking `factor_bench`'s
ArmEnv. Recorded on `docs/design/environment.md#environment-route-pins-are-raw-strings` and
`#environment-what-a-settings-reload-does-not-cover`.

No code site points here.

## -> docs/cpp-api.md: (no new heading; "What gets thrown")

The kernel-selection layer throws outside the `batchlas::exception` hierarchy: an invalid
`BATCHLAS_<OP>_ROUTE` pin is a plain `std::invalid_argument`, "no runnable kernel" and a bad tuned
table are plain `std::runtime_error`, and `batchlas::NoRouteError` derives from
`std::runtime_error` only. A user who follows the page and catches `batchlas::exception` misses
all of them. Full table: `docs/design/error-model.md#error-model-kernel-selection-throws-outside-the-hierarchy`.
The H2-core items for this page (api_misuse / device_error / invalid_argument wording) are still
open too.

No code site points here.

## -> docs/design/known-defects.md: candidate row "kernel selection throws outside the error hierarchy"

Same finding as above, as a defect: reclassifying the `std::` throws in `src/select/select.hh`,
`src/select/select.cc` and `include/batchlas/no_route.hh` into `batchlas::invalid_argument` /
`batchlas::unsupported` is a behaviour change for callers that catch the `std::` type. Evidence:
`docs/design/error-model.md#error-model-kernel-selection-throws-outside-the-hierarchy`. Also:
`gesv`/`posv` throw `batchlas::internal_error` for an empty or heterogeneous batch, which by
meaning is `invalid_argument` or `unsupported`.

## For whoever owns AGENTS.md (section 9)

"Unrecognised values silently mean Auto" is stale since flat kernel selection: under `src/select`
an unrecognised `BATCHLAS_<OP>_ROUTE` throws `std::invalid_argument`; only `native` and `vendor`
fall back to Auto (with a warning) when nothing in their class can run. "legacy
`_VARIANT`/`_PROVIDER` spellings are still parsed" is also stale: they are no longer read.

## For whoever owns .github/ci/comment_density_waivers.txt

`check_comment_density.py` reports eight waivers in this shard's headers as dead weight (each file
is now under 18%): lines 100, 103, 104, 105, 106, 108, 109, 111 (`error.hh`, `settings.hh`,
`sycl_interop.hh`, `tuning_params.hh`, `util/env.hh`, `util/mempool.hh`,
`util/sycl-device-queue.hh`, `util/workspace.hh`).

## For whoever owns docs/perf/gesvd.md (still open from H2-core)

The H2-core item "gesvd: the gebrd panel-width split from ORMQR" (the nb sweep 234.9 / 232.1 /
230.7 / 235.5 / 240.9 / 256.6 ms and the per-n retune winners) is not yet in the page; those
numbers currently live only in `docs/_inbox/H2-core.md`.
