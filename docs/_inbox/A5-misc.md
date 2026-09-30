# Inbox from shard A5-misc

- **docs/ci.md** (`## Running all of it locally`): the parenthetical
  "(`tests/README.md`'s table quotes an older 38-of-45; the counts here are the
  measured current ones.)" is now obsolete; tests/README.md quotes 65 of 70 and
  links back to that section. Drop the parenthetical.
- **docs/perf/syev.md (syev shard)**: the A5 pointers now cite its anchors
  (reviewer, 2026-09-30): `src/extensions/syev_blocked.cc:258` ->
  `#syev-stebz-values-only-in-the-blocked-solver-wp1`,
  `src/extensions/syev_jacobi_cta.cc:49` -> `#syev-the-2026-08-03-small-n-bake-off`,
  README Performance -> `#syev-the-blocked-over-cusolver-headline-measurement`.
  Do not rename those three headings. That page's small-n bake-off section still
  cites `JACOBI_EIGENSOLVER_PLAN.md §13.1` as a live source; the file was never
  committed, so it should say so (or drop the name).
- The root SYEV_* plan files still mention `SYEV_PERF_RESEARCH.md` and
  `JACOBI_EIGENSOLVER_PLAN.md`; neither file exists in the repository, so their
  migrated text should not carry those names as live references.
- **docs/pages/index.md / section pages**: link the two new pages
  `@ref guide_ritz_values` (guide) and `@ref design_cpu_target_detection`
  (architecture/design).
