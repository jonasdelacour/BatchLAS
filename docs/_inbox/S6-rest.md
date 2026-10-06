# Inbox from shard S6-rest

## -> docs/perf/blackwell.md: known-defects #11 link renamed

`docs/design/known-defects.md`'s `## 11. Native gemm reads C at beta zero` is now
`## Defect 11: native gemm reads C at beta zero`, because a heading that starts with a number gets
an `autotoc_md`-prefixed Doxygen id and so cannot be cited by an `evidence:` pointer. The link at
`docs/perf/blackwell.md:1013`, `[known-defects #11](../design/known-defects.md#11-native-gemm-reads-c-at-beta-zero)`,
should become `../design/known-defects.md#defect-11-native-gemm-reads-c-at-beta-zero`.

Code sites now pointing at the new heading:

- `src/extensions/geqrf_blocked.cc:206`:
  `evidence: docs/design/known-defects.md#defect-11-native-gemm-reads-c-at-beta-zero`

## -> tests/syev_candidates_tests.cc (whoever owns tests): a citation of a numbered heading

`tests/syev_candidates_tests.cc:374` mentions
`docs/design/known-defects.md#14-the-hermitian-drivers-read-the-unreferenced-triangle`. The GitHub
slug resolves, but the Doxygen id is `autotoc_md14-...`, so `check_doc_anchors.py` will report it
as "Doxygen id differs" once it is collected. The heading was left numbered so this citation, and
the page links to `#1-...` / `#12-...` from `docs/perf/ortho.md`, `docs/design/runtime-internals.md`
and `docs/design/flat-kernel-selection.md`, keep resolving. To make #14 citable, rename it to
`## Defect 14: the Hermitian drivers read the unreferenced triangle` and update this test comment
in the same change. The same applies to `src/extensions/ortho.cc:188` (defect 1) and
`src/extensions/lanczos.cc:55` (defect 3), which carry plain "see defect N" mentions (S4a's item).
