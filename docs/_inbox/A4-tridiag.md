# Inbox from shard A4-tridiag

Items found while migrating STEQR_MATHEMATICS.md -> docs/algorithms/steqr.md (algo_steqr) and
STEDC_MERGE_OPTIMIZATION.md -> docs/perf/stedc.md (perf_stedc). Not applied: files not owned.

1. include/batchlas/tuning_params.hh (about lines 155-198) and the generator template in
   evaluation/tuning/generate_tuning_header.py: the StedcMergeVariant history (3072ea6 flip,
   re-test at 0bb92fb, FusedCta 11-35% faster) and the syev-side threads-per-root /
   wg-multiplier sweep table (2026-08-07) are lab-notebook material. Both are now recorded in
   `docs/perf/stedc.md#stedc-current-tuning-values`. Suggest shrinking the header comments to the
   trap ("FusedCtaConditionedHeavyDeflation asserts only finite-and-sorted") plus
   `evidence: docs/perf/stedc.md#stedc-current-tuning-values`, in the generator too so a retune
   does not reintroduce them.
2. SYEV_RETUNE_RESULTS.md section 14 ("Defect A - the merge-variant flip was backwards") is
   summarised in `perf_stedc#stedc-current-tuning-values`. Whoever migrates that file can link
   there instead of duplicating it.
3. Stale comment: include/batchlas/blas/extensions.hh, enum SteqrUpdateScheme, says PG is the
   "current default"; SteqrParams::cta_update_scheme defaults to EXP. Noted in
   `docs/algorithms/steqr.md#steqr-the-cta-update-schemes-exp-and-pg`.
4. Stale comment: tests/stedc_tests.cc FusedCtaConditionedHeavyDeflation says the tuning
   tables select secular_threads_per_root = 4 for n <= 64; STEDC_THREADS_PER_ROOT_* is 8.
5. Possible defect worth a look (not verified): stedc_impl (recursive driver,
   src/extensions/stedc.cc) forwards the caller's jobz to the leaf steqr_dispatch, while the
   merges always consume leaf eigenvectors. A direct stedc(..., NoEigenVectors, ...) call with
   StedcAlgorithm::Recursive may merge from unset leaf vectors. The level driver ignores jobz.
   Candidate for docs/design/known-defects.md after a test.
6. docs/pages section pages: algo_steqr and perf_stedc need `\subpage` entries in whichever
   section pages list algorithms / perf pages.
