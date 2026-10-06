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

## -> .github/ci/comment_density_waivers.txt (coordinator): S4b-eigen waivers now free

Second pass (after the flat-selection merge). These waiver lines are dead weight; every file is
under 18% (`check_comment_density.py --all` reports them as "waived but now UNDER"):
stedc.cc (:136), stedc_internal.hh (:137), stedc_levels_plan.hh (:138), stedc_merge_kernels.hh
(:139), stedc_secular.hh (:140), steqr_cta_device.hh (:142), syev_blocked.cc (:143),
syev_cta_fused.cc (:144), syev_jacobi_cta.cc (:145), sytrd_cta_device.hh (:152, now 7.9%),
sytrd_sb2st_hh.cc (:153), sytrd_sb2st_hh.hh (:154, now 16.2%), sytrd_sy2sb.cc (:155),
two_stage_common.hh (:157), latrd_lower_panel.cc (:132). Still needed:
`src/extensions/uplo_mirror.hh` (:158), 4 comment lines over 7 code lines, all traps (in place,
imaginary diagonal, ODR declaration-only) plus the pointer; suggested reason: "short header whose
remaining lines are traps; narrative in docs/perf/syev.md#syev-the-upper-to-lower-mirror-for-lower-only-providers".
No code site points here.

## -> docs/design/known-defects.md: tridiagonal_solver addresses Q with stride n

Seen while checking H6's open items against `src/extensions/tridiag_solver.cc` (owned by
S4b-eigen; no code change made). H6 recorded it: the rotation update addresses Q as
`Q[k*m+l]` (stride n) while the identity fill uses `Q.ld()`, so a Q with `ld() != n` gets a wrong
answer; QR steps are capped at six per eigenvalue with no convergence report; nothing in `src/`
calls it. The public doc says `@pre Q.ld() == n`. Suggested row: "located, documented as a
precondition, unfixed; no caller". No code site points here.
