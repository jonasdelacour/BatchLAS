# Inbox from shard S4a-syevx

No code pointer from this shard targets a page it does not own, so nothing here is needed to make
a pointer resolve. The items are index links, stale file:line citations and anchor fixes.

## -> docs/perf/README.md: index row for ortho

New page `docs/perf/ortho.md` (`@ref perf_ortho`, "ortho: the Gram product and the algorithm
rules"). Add an index row and a `\subpage perf_ortho` in whichever section page lists the perf
pages. Native by default: yes (the Gram product is syrk or GEMM; Householder on host devices).

## -> docs/design/known-defects.md: citable anchors for the numbered defects

Headings of the form `## 1. ...` get Doxygen ids with an `autotoc_md` prefix
(`autotoc_md1-orthos-transposed-arm-builds-a-view-that-does-not-describe-the-memory`), so the
GitHub slug is not live on the site and `check_doc_anchors.py` rejects any `evidence:` pointer to
them. Two code sites wanted such a pointer and now carry a plain mention instead:

- `src/extensions/ortho.cc:188` (the CGS `A_i` view, defect 1): `// See docs/design/known-defects.md, defect 1.`
- `src/extensions/lanczos.cc:55` (the padded two-column multiply, defect 3): `// See docs/design/known-defects.md, defect 3.`

Suggested fix: give the numbered headings explicit ids or unnumbered distinctive text (for example
`## Defect 1: ortho's transposed arm builds a view that does not describe the memory`), then turn
the two mentions into `evidence:` pointers. Also update that page's `file:line` citations:
the CGS view is now `src/extensions/ortho.cc:189-193` (was `:218-224`), and the lanczos multiply
is `src/extensions/lanczos.cc:112-117` (was `:107-111`); `padded_output` is at `:57`-ish.

## -> docs/perf/level3.md: link the full per-precision Gram table

`### syrk for the ortho gram matrix` cites `ortho.cc:176` for `gram_max_k`; it is now
`src/extensions/ortho.cc:143`. The full `k x precision` table (float 1.62/1.12/0.96x, double
1.02/1.20/1.34x at k = 32/64/128) that used to be the comment at that line is now at
`docs/perf/ortho.md#ortho-the-gram-matrix-through-syrk-per-precision`; a `@ref perf_ortho` link
from that section would join the two.

## -> docs/perf/gemv.md, docs/perf/trsm.md, docs/perf/dispatch.md: stale ortho.cc line citations

The comment moves in `src/extensions/ortho.cc` shifted every line after ~46 (most citations were
already stale before this pass). Current positions: `inv_trans` at `:123`, the two
`level3_tile_route_available` gates at `:146` and `:149`, the Cholesky `trsm` calls at `:171` and
`:260`, the CGS `A_i`/`C` views at `:189-193`.
