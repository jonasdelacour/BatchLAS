# Inbox from shard A6-tooling

- `.github/ci/comment_density_waivers.txt`: with the public-API doc-comment exemption, two waived
  headers are now under 18% and their waiver lines can be deleted (the checker lists them as
  "free to delete"): `include/batchlas/blas/extensions.hh` (line 77 before my header edit,
  10.81%) and `include/batchlas/util/sycl-local-accessor-helpers.hh` (line 111, 16.67%).
  A6 owns only the header comments of that file, so the lines were left in place.
- `docs/Doxyfile` near `USE_MATHJAX`: the comment says math in the pages is written GitHub-style
  (`$...$`). The binding convention is now `\f$...\f$` / `\f[...\f]`; the comment is stale.
- `docs/pages/index.md` (or whichever section page owns the developer area): link
  `@subpage documentation_conventions` (docs/developer/documentation.md) so it is not an
  orphan top-level page in the tree view, and link `docs/ci.md`.
- RESOLVED at A6 review (0 outside docs/): `check_markdown_locations.py` was red until every root plan file was migrated (8 left at the
  time of writing: SYEVX_PLAN, SYEVX_RANGE_PLAN, SYEV_PERF_IDEATION, SYEV_PERF_IMPLEMENTATION_PLAN,
  SYEV_PLAN_BASELINES, SYEV_PLAN_RESULTS, SYEV_RETUNE_RESULTS, SYEV_RETUNE_WORKFLOW_PLAN).
- (A6 review) `docs/Doxyfile` does not exclude `docs/_inbox/`: every inbox scratch file is published as a
  site page (e.g. `md_docs_2__inbox_2A6-tooling.html`) and its warnings count against
  `BATCHLAS_DOCS_STRICT=1` (A1-syev.md:18 and A4-tridiag.md:27 warn today). Add `docs/_inbox` to
  `EXCLUDE`, or delete the directory once merged.
- (A6 review) Policy question: `check_markdown_locations.py` allows `.github/**` but not `.claude/**`.
  Claude Code reads `.claude/commands|agents|skills/*.md` in place, the same reason `.github/skills` is
  exempt. Nothing is tracked there today, so the gate is correct now; add `.claude/` to
  `ALLOWED_TREES` (plus a self-test row and the four prose mentions) if such files are ever committed.
