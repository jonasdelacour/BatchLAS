# Inbox from shard C1-select

C1 owns src/select/, the selection pages and the `selection` / `selection_ops` groups in
docs/pages/api_groups.dox. Items below are for pages and files C1 does not own.

## -> docs/pages/architecture.md: rows for the selection pages

The vendor-independence row's description ("The Route vocabulary, the entry-point facade, the vendor
gate, the resolver and the coverage instrument") describes the deleted route layer. Suggested rows:

    | @subpage design_flat_selection "Flat kernel selection" | How every op chooses its kernel: families, can_run, per-device tuned tables, pins, trace, coverage; the tuner; how the RouteTable layer was replaced. |
    | @subpage design_flat_selection_phase3_plan "Flat selection: phase-3 plan (history)" | The 2026-10-04 execution plan for posv, the tuner, trsm and gemm; cited as "plan §n". |
    | @subpage md_docs_2design_2vendor-independence "Vendor independence" | The vendor-free build, the per-library vendor gate, entry-point contracts (validators, info spans), the coverage instrument; the RouteTable layer as history. |

`design_flat_selection` and `design_flat_selection_phase3_plan` are new page labels (H1 `{#...}`);
neither page is a `@subpage` of anything yet, so Doxygen lists them at top level.

## -> docs/Doxyfile (coordinator): INPUT for the selection group and the two READMEs

`src/select` (FILE_PATTERNS picks up the three `.hh`), `src/ops` (the `choice.hh`, `vendor.hh`,
`geqrf.hh`, `can_run.hh` headers, for `selection_ops`), `tuned/README.md` (label
`tuned_tables_readme`) and `tools/tune/README.md` (label `tune_tool_readme`). Without the two
READMEs, `@ref tuned_tables_readme` / `@ref tune_tool_readme` in docs/design/flat-kernel-selection.md,
docs/extending.md and docs/design/vendor-free-status.md warn. Verified with a local Doxyfile that
`@INCLUDE`s docs/Doxyfile and adds exactly these: zero warnings in C1's files. Optional:
`EXCLUDE_SYMBOLS += batchlas::select::testing` (test hooks).

benchmarks/results/{routing,tuning}/README.md were deliberately NOT labelled or proposed for INPUT:
they are under the `benchmarks/results/**` LFS filter, and the CI docs job builds from LFS pointers
(no `lfs: true`), so on CI they would render as pointer text and any label in them could never
resolve.

## -> docs/tools/gen_db_pages.py (owner of the generator): use the new label

`selection_page()` writes `@ref md_docs_2design_2flat-kernel-selection`. Doxygen 1.18 still resolves
it (to `design_flat_selection.html#md_docs_2design_2flat-kernel-selection`), but the page now has the
label `design_flat_selection`; `@ref design_flat_selection` is the stable spelling.

## -> per-op shards owning src/ops/<op>/choice.hh: join selection_ops

The group `selection_ops` (child of `selection`) is defined in api_groups.dox. Each `choice.hh`
should put its declarations in it, e.g. `/// @addtogroup selection_ops` + `/// @{` ... `/// @}` inside
`namespace batchlas::ops::<op>` (two-line form: a one-line `/** @addtogroup g @{ */` makes Doxygen read
`@{` as the group title and warn). In src/ these lines count toward the 18% density ceiling.

## -> docs/pages/api_groups.dox, group `dispatch` (owner of that group): stale brief

`@defgroup dispatch Dispatch and routing` still says "How a call picks a Route{Origin, Algorithm}:
RouteTable, supports(), preferred(), env overrides." All of that is deleted. Suggested brief: "The
public dispatch surface: queue-dispatch overloads, NoRouteError and the BATCHLAS_<OP>_ROUTE pin
variables. How a kernel is chosen is the developer group @ref selection."
