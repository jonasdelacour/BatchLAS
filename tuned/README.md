# Tuned selection tables {#tuned_tables_readme}

> **Status:** current · inventory as of 2026-10-11 · the generated @ref selection_tables page is always current.

One file per (op, device): `<op>.<device>.txt`, for example `potrf.sm_120.txt`. The first key is
`dtype`; each row is one shape of one dtype and the ranked list of every candidate timed there,
fastest first, ending in its ledger tier (`# deep`). The library embeds these files at build time
and splits each into one table per dtype. `select::choose` takes the first runnable entry of the
nearest row. The layout, generation and check are in
[the tuning data layout](../docs/design/tuning-data.md); lookup rules in
[the flat kernel selection design](../docs/design/flat-kernel-selection.md) §5.4-§5.5; the tuner in
[tools/tune/README.md](../tools/tune/README.md).

The files are plain git, not LFS, so table changes stay readable in diffs.

## Producing tables

Each table is generated from its ledger, never edited by hand; the rules and commands are in
[the tuning data layout](../docs/design/tuning-data.md#tuning-data-generation-and-check).

## Inventory

Every op ships a table for each dtype it instantiates (float, double, cfloat, cdouble; symm, syrk
and syr2k are real only) on sm_89 and sm_120. spmm also ships a CPU table. Any other device borrows
the nearest table and warns once. `tuned_tables_tests` (`EveryOpShipsATableForEveryDtypeOnEveryShippedDevice`)
checks this inventory.

Counts: 45 files, 22 per GPU device plus `spmm.cpu.txt`.

| device | ledger tiers | where from |
|---|---|---|
| sm_120 | `deep`, plus `transcribed` rows where the deep lattice measured nothing near; sytrf and sysv all `transcribed` | "The sm_120 deep tables" below; sytrf and sysv: @ref perf_sytrf |
| sm_89 | `transcribed` (the old routers' order, untimed); potrf `converted` | the old transcribed tables and the sm_89 potrf forced-route archive, migrated (`docs/design/tuning-data.md`) |
| cpu (spmm) | `transcribed` | likewise |

Transcribed rows reproduce deleted routers' windows at their grid points; they are not
measurements, and a measured row near them replaces them. Their provenance (transcriber, commit)
is the run line of the ledger, run id `transcribed-<sha>`: posv `7e71a6e0`, trsm `8b9adeb3`; gemm,
gemv, getri, gesv, spmm, syev, geqrf, orgqr, ormqr, getrf, getrs, gesvd `424a45bc`; symm, syrk,
syr2k, trmm `ff340fc6`; potri `628b77d5` (both devices, no timing: `fused` for n <= 8, `blocked` above);
sytrf, sysv `30cc8e06` (both devices, @ref perf_sytrf). The transcribers' sources are in git history, for example
`git show 0bd26dfe:tools/transcribe/posv_transcribe.cc`. The migration script that wrote the
ledgers is in the commit that added them (`git log -- scripts/migrate_tuning_data.py`).

## The sm_120 deep tables

All sm_120 tables come from one `--tier deep` campaign over all 19 ops and every dtype each op
instantiates, on threadripper02 (4x RTX PRO 6000 Blackwell, sm_120, CUDA 13.2, DPC++
`/opt/dpcpp-cuda`), 2026-10-08 06:23 to 2026-10-09 04:54 CEST, four starts of
`batchlas_tune all --tier deep --devices 0,1,2,3`, plus a re-measure of 250 cells the box's other
user had damaged (`--remeasure-keys`). Details are in
[tiered tuning](../docs/design/tiered-tuning.md), "Engine: the sm_120 deep run".

- **2026-10-11/12 updates without a new campaign.** Specs on `batchlas::verify` and two kernel fixes
  (defect 11: native gemm read C at beta = 0; the stedc one-pole merge's convergence flag) staled
  most families. Stratified `--tier custom` samples (about 128 cells per op, all 19 ops) found no
  verdict change from the new checks except the gesvd `cta` normal-equations cells (correctly bad)
  and the syev cells the stedc fix repaired. Then, per op:
  - re-measured at deep: gemv (its 2026-10-09 run was dirty), potrf, gesv, posv, syev, trmm
    (their samples disagreed systematically: the blocked paths got faster);
  - carried over by sample-proof rehash lines (docs/design/tuning-data.md, "Kernel hashes"): gemm
    (9 noise-level disagreements re-measured at deep with `--only-named`), geqrf, getrf, orgqr,
    ormqr, gesvd, symm.
- **Dimension cap 2048.** No cell was planned with a matrix dimension above 2048 (`--max-dim`). The
  first run measured some larger cells before the cap existed; they stay. Above the cap the nearest
  measured row decides.
- **Refinement coverage is uneven.** Every starting lattice is complete. geqrf, ormqr, getri, posv,
  gesv, orgqr, syev, gesvd and complex getrs are only lightly refined, so their crossovers sit on
  lattice spacing.
- **Accuracy is not timed.** LOBPCG pins its projected syev solves to the CTA solver at n <= 32 for
  that reason (`docs/perf/syevx.md`, "LOBPCG: the accuracy pin on the projected syev").

## Staleness

The hashes and the staleness warning are specified in
[the tuning data layout](../docs/design/tuning-data.md#tuning-data-kernel-hash).
