# Inbox from shard A2-syevx

## For whoever owns include/batchlas/blas/enums.hh (lines ~326-327)

The `SyevxAlgorithm` enumerator comments still say `DirectSubset` (Tier 2) and `Filtered`
(Tier 3) are "not yet implemented". Both have been implemented since 2026-08-02. See
`docs/perf/syevx.md#syevx-implementation-status-by-tier`. The "Tier N" wording now refers to the
tier table in `docs/design/syevx.md#syevx-the-original-tier-plan`.

## For whoever owns include/batchlas/settings.hh (BATCHLAS_SYEVX_FILTER_DEGREE_AUTO, ~line 279)

The one-line "measured 2.2x faster at batch 1 and 0.48x at batch >= 8" could carry
`evidence: docs/perf/syevx.md#syevx-automatic-chebyshev-filter-degree`, where the full grid now
lives.
