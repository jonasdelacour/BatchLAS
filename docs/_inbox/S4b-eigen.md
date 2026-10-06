# Inbox from shard S4b-eigen

## -> docs/design/known-defects.md: known defects: the CTA SYTRD Lower path

Moved from `src/extensions/syev_cta.cc` (the `uplo_eff` note in the real/complex `syev_cta`
driver). The source said, verbatim: "The CTA SYTRD/SYEV pipeline currently exhibits severe
correctness issues specifically for the Uplo::Lower path. Until the lower-path kernel is fixed,
we run the (known-good) Uplo::Upper pipeline. To preserve the public API contract when callers
only initialize the lower triangle, we first explicitly symmetrize: A_upper := conj(A_lower)."
No test, commit or failure mode was recorded with it. The fused kernel (`syev_cta_fused.cc`)
also always runs the Upper reduction and symmetrises Lower input while loading its tile, so the
defect is worked around on both CTA paths; whether the Lower `sytrd_cta` kernel is still wrong,
and how, is unverified. Suggested row: "located, worked around, not reproduced".

Code sites that now point here:

- `src/extensions/syev_cta.cc` (the `uplo_eff` note):
  `evidence: docs/design/known-defects.md#known-defects-the-cta-sytrd-lower-path`

## -> docs/design/known-defects.md: known defects: stedc recursive driver forwards jobz to the leaf

A4's item 5, now recorded on the owned page `docs/perf/stedc.md#stedc-eigenvalues-only-still-builds-eigenvectors`
(and in a trap comment at the recursive leaf in `src/extensions/stedc.cc`, which cites that stedc
anchor, not this one). Candidate for the "unverified candidates" table once a direct
`stedc(..., NoEigenVectors, ...)` test with `StedcAlgorithm::Recursive` and n > the recursion
threshold has been run. No code site points at this heading.

## -> docs/perf/README.md: index row for the new sytrd page

Add a row: `[sytrd.md](sytrd.md)` | `sytrd_blocked` + `latrd`, `sytrd_sy2sb`, `sytrd_sb2st_hh`
and its Q2 back-transform (`@ref perf_sytrd`) | n/a: internal reductions under `syev`, no vendor
arm of their own. The steqr page now has the label `perf_steqr` (it had none). Both need a
`\subpage` entry wherever the perf section page lists its children. No code site points here.
