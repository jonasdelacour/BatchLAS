# Dispatch and coverage {#perf_dispatch}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · checked against `src/select` and the level-3 `choice.hh` files 2026-10-29

The cross-op dispatch rules that are not owned elsewhere: the level-3 pin words, how a buffer-size query
sizes the same route as its call, the coverage instrument's past defects, and the dispatch designs and bugs
that shaped `src/select`. How ops choose: @ref design_flat_selection and @ref api_selection; per-op ranking:
@ref selection_tables. The vendor gate, the coverage row kinds and the verification scripts are owned by
@ref design_vendor_independence. Level-3 families, `can_run` terms, table edges and their measurements are on
@ref perf_level3.

## The level-3 pin words {#the-level-3-pin-words}

`BATCHLAS_<OP>_ROUTE` is resolved by `select::detail::resolve_pin` (`src/select/select.hh`) for every op:

- `auto`: the table. `native`: the row's first runnable non-vendor entry. `vendor`: the candidate spelled
  `vendor`. If nothing of that class can run, both class words warn and fall back to Auto, so `vendor` in a
  vendor-free build means Auto; a `ScopedPin` built with `StrictPin` throws instead.
- A spelling that does not parse, is not a compiled candidate for the dtype, or whose `can_run` refuses the
  call throws `batchlas::invalid_argument` before anything is written.
- The value comes from the settings snapshot; call `detail::reload_settings()` after a raw `setenv`.

Tests: `<Op>Candidates.UnknownPinsThrow`, `.CanRunFalsePinsThrow` and `.ClassWordsAndSpellings` in
`tests/{symm,syrk,syr2k,trmm}_candidates_tests.cc`.

| op | spellings | throws when pinned |
|---|---|---|
| `symm` | `mirror`, `expand`, `vendor` | `mirror`, `expand` on a heterogeneous batch; `expand` when one item's scratch does not fit (e.g. `BATCHLAS_EXPAND_MAX_BYTES=0`) |
| `syrk` | `gram`, `triangular`, `vendor` | `triangular` on complex<double> (not compiled); `gram` at n > 128 |
| `syr2k` | `triangular`, `vendor` | `triangular` on complex<double> (not compiled) |
| `trmm` | `triangular`, `expand`, `vendor` | `triangular` with `Side::Right`. `vendor` is the `cublas?trmm` loop; the old `vendor` (expand + gemm) is now `expand` |

Every other word throws: another op's spelling, `cublasdx`, the legacy `tiles`/`narrow`/`gemm`/`custom`/`dx`/
`fused` words and `diag_full_gemm` (the `UnknownPinsThrow` lists). hemm, herk and her2k are pinned the same way
through `BATCHLAS_HEMM_ROUTE`, `BATCHLAS_HERK_ROUTE` and `BATCHLAS_HER2K_ROUTE`; `BATCHLAS_EXPAND_ROUTE` and
herk's `BATCHLAS_SYRK_ROUTE=gram` opt-in are gone.

## Buffer-size queries {#dispatch-buffer-size-queries-and-the-route-they-size}

Every `<op>_buffer_size` runs the same `select::pick` as its op and returns exactly the chosen family's need,
through the op's `workspace()` visit in `src/ops/<op>/<op>.cc`. A nested op adds its children's sizes through
their public sizing functions. Query and call agree because they run the same choice, not because the query
over-allocates.

1. **The vendor path sizes itself from the vendor query.** `potrf_vendor` in `src/backends/cusolver.cc` calls
   `backend::potrf_vendor_buffer_size`. An unqualified `potrf_buffer_size` resolved to the facade query, which
   returned the max of native and vendor sizes and so handed cuSOLVER the wrong workspace size.
2. **Sizing is pure.** `band_reduction.cc` and `sytrd_sy2sb.cc` size `geqrf` once against a dummy view through
   `geqrf_buffer_size_bound` (the maximum over every candidate the device can run, `src/ops/geqrf/geqrf.cc`).
   No `geqrf` sizer may dereference `A.data_ptr()` or `tau.data()`, which are `nullptr` there.
   `getri_buffer_size` runs under `BumpAllocator::measuring()`, so everything reachable from it must not touch
   the workspace.

## Coverage instrument {#dispatch-the-coverage-instrument}

`src/select/coverage.hh` writes `linked` rows (is the kernel in the build), `reached` rows (did a call get
there, with the chosen family spelling) and `miss` rows (nothing could serve the call). Linked is not reached:
a present symbol is never evidence that it runs. Row kinds, keys and the tools that read them
(`scripts/route_diff.sh`, `scripts/coverage_merge.sh`) are described in
[the coverage instrument](../design/vendor-independence.md#vendor-independence-the-coverage-instrument).

### Instrument defects {#instrument-defects}

Each of these made the instrument look healthy while it reported almost nothing:

1. The gate-declined half was unrecorded, so a shape moving off a native kernel was invisible.
2. `uplo`, `side` and `diag` were missing from the key, so calls differing only in `uplo` shared one row.
3. `emit()` opened the file with `"w"`, so each of 53 test binaries truncated the last (now one file per pid).
4. A compile-time gate let weak-symbol interposition replace the library's resolver with an uninstrumented
   copy: a file with a correct header and zero `reached` rows. The gate is now a runtime bool in
   `src/select/coverage.cc`.
5. **Still open:** `route_diff.sh compare` applies no `backend != AUTO` filter, so pure-layer test shapes made
   a clean 65-decision move look like 240 lines of churn.

## Negative results {#dispatch-negative-results}

| Alternative | Result | Verdict |
|---|---|---|
| Split each level-3 TU into portable and CUDA halves and transcribe the gate thresholds into `RouteTable::preferred` | Silent route change: the transcribed `>= 3` rule rejected the syrk tile route for 129 <= n <= 383 at every batch, and n = 256 went to a route that wrote both triangles | Rejected. Flat selection transcribes the whole old gate per grid cell (`--fidelity` against `ff340fc6`) and was accepted at 100.00% off-grid agreement (@ref design_flat_selection) |
| Public entry point as the sideways vendor fallback | A public re-entry after a gate that had already returned true recursed without bound (reachable with a forced pin on a CPU queue) | Rejected. Each `vendor` family calls `backend::<op>_vendor<B, T>` directly |
| One flag that "becomes true for every backend" for tile substitution | Too wide in type and in backend | Replaced by `select::level3_tile_substitution_enabled<B, T>` ([why](../design/vendor-independence.md#the-vendor-gate-why-the-tile-route-predicate-is-per-backend-and-scalar)) |

The stub-TU layer for absent vendor libraries and the compile-time coverage gate are rejected on
@ref design_vendor_independence.

## Correctness findings {#dispatch-correctness-findings}

Wrong answers found, how they hid, and what guards them now.

* **`ormqr` buffer size and call disagreed by 108x.** The route-era `choose_ormqr_provider` returned a forced
  provider without checking `ormqr_supports_blocked`, so a forced `blocked` ran on complex `Trans` and on non-GPU
  queues, and a forced `cta`, `two_stage` or `jacobi` ran on the vendor while the query returned the blocked size
  (2560 bytes against 276480). Flat selection prevents both: a pin that fails `can_run` throws, and an op and its
  `*_buffer_size` run the same choice.
* **`vendor` must be one fieldless family.** Route-era `is_vendor()` let a forced cuBLASDx request count as "the
  caller asked for the vendor". cuBLASDx is deleted; `vendor` is the only family that calls a vendor library.
* **An order-walk fallback inverted GEMM's default.** Taking the first merely supported route moved an 8x8x8
  batch-1 GEMM from vendor to native. A last resort now applies only when no table entry can run.
* **Two ROCm defects invisible to the CUDA build** (a `trsm` instantiation in the old parameter order, four
  orphaned macro-continuation lines), caught by `scripts/rocm_syntax_check.sh`.
* **Signature divergence between vendor TUs.** `trsm`'s vendor form takes `alpha` last, the public form third;
  generated facade bodies would have passed `alpha` as `side`. Each `vendor` family spells its
  `backend::<op>_vendor<B, T>` call by hand, so the hazard is one call site per op.
* **An instantiation binds as hard as a definition.** `syev` and `ormqr` instantiations lived in
  `cusolver.cc`/`cublas.cc` and vanished from builds without those libraries. `scripts/facade_symbol_check.sh`
  guards it.

Level-3 correctness findings are on @ref perf_level3.

## Open debts

- `route_diff.sh compare` has no backend filter (instrument defect 5).
- A coverage row cannot confirm that a particular shape ran: rows are keyed on a power-of-two `shape_class` and
  are first-writer-wins. Prove a shape with a deliberate break that is red only for it.
- The vendor-free suite is not green; the gap is missing kernels (`NoRouteError`), not dispatch
  (@ref design_vendor_free_status).
- Level-3 debts (transcribed sm_89 tables, unverified windows) are on @ref perf_level3.

## Raw evidence

Raw data is kept at the git tag `perf-evidence/vendor-independence` (`git show <tag>:<path>`), with the
superseded `WP0_DISPATCH_SPEC.md`, `WP1_LEVEL3_SPEC.md`, `VENDOR_FREE_BASELINE.md` and
`VENDOR_INDEPENDENCE_PLAN.md`. Level-3 sweep drivers are listed on @ref perf_level3.
