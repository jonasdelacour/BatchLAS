# Inbox from shard S1-level3

## -> docs/design/known-defects.md: trmm generic recursion reads the whole square of A

Candidate only, found by reading `src/extensions/trmm.cc` during the 2026-09-30 comment pass, not
confirmed by a test. The MKL-instantiated generic `trmm` ends its recursion at n <= 256 in a plain
`gemm` on the diagonal block with `beta = 1` (a `triangularize` call there was commented out). As
written, the base case reads A's unreferenced triangle and its stored diagonal under `Diag::Unit`,
and accumulates into C instead of overwriting it. The full note and what would settle it are in
`docs/perf/level3.md#trmm-the-generic-recursion-reads-the-whole-square-of-a`; a known-defects row
could link there.

Code sites that point at the level3.md section (no pointer into known-defects.md needed):
- `src/extensions/trmm.cc:25-26`

## -> docs/perf/dispatch.md: note on duplicated level-3 material

Not a move, a consistency note. `docs/perf/dispatch.md` sections "Level-3 route arms", "Expansion
crossover", "`syrk` tile boundaries", "`syr2k` batch boundary", "`trmm`: no threshold" and
"`herk` and `her2k` crossovers" restate numbers that `docs/perf/level3.md` owns. level3.md's herk
section was renamed to "herk and her2k: the GEMM-plus-fold crossovers" (its old slug collided with
dispatch.md's). The dispatch owner may want to replace those restatements with links to level3.md so
the two copies cannot drift.
