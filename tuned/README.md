# Tuned selection tables {#tuned_tables_readme}

> **Status:** current · inventory as of 2026-10-23 · the generated @ref selection_tables page is always current.

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

Every op ships a table for each dtype it instantiates (float, double, cfloat, cdouble; hemm, herk
and her2k are complex only) on sm_89 and sm_120. spmm also ships a CPU table. Any other device borrows
the nearest table and warns once. `tuned_tables_tests` (`EveryOpShipsATableForEveryDtypeOnEveryShippedDevice`)
checks this inventory.

Counts: 53 files, 26 per GPU device plus `spmm.cpu.txt`.

| device | ledger tiers | where from |
|---|---|---|
| sm_120 | `deep`, plus `transcribed` rows where the deep lattice measured nothing near and a few `custom` rows (gemv, posv, trmm); gemm double and cdouble are `transcribed` only; potri and potrs all `transcribed` (from sm_89's measured rows); sygst and sygvd all `transcribed` | "The sm_120 deep tables" below; potri: @ref perf_potri; potrs: @ref perf_potrs; sygst and sygvd: @ref perf_sygvd |
| sm_89 | `transcribed` (the old routers' order, untimed); potrf `converted` plus 12 `deep` float Upper rows; geqrf 16 `deep` wide rows (both re-measured 2026-10-24); potri and potrs float and cfloat `deep` (2026-10-26), double and cdouble `transcribed` from them; sygst float and cfloat `deep` (2026-10-27, every itype and uplo), its double and cdouble `transcribed` from them | the old transcribed tables and the sm_89 potrf forced-route archive, migrated (`docs/design/tuning-data.md`) |
| cpu (spmm) | `transcribed` | likewise |

Transcribed rows reproduce deleted routers' windows at their grid points; they are not
measurements, and a measured row near them replaces them. Their provenance (transcriber, commit)
is the run line of the ledger, run id `transcribed-<sha>`: posv `7e71a6e0`, trsm `8b9adeb3`; gemm,
gemv, getri, gesv, spmm, syev, geqrf, orgqr, ormqr, getrf, getrs, gesvd `424a45bc`; symm, syrk,
syr2k, trmm `ff340fc6`. Later rule changes are their own transcribed runs: gesvd `141d31db`
(complex general above 32 gains `blocked`, Hermitian Upper copies Lower) and `167a4eaf` (sm_89
cfloat 33..64 puts `blocked` first, docs/perf/gesvd.md); the complex rows of symm, syrk and syr2k
and every hemm, herk and her2k row `26168c21` (the deleted `cublas.cc` rules, docs/perf/level3.md).
potri `628b77d5` (both devices, no timing: `fused` for n <= 8, `blocked` above), superseded by
`c6aa0c2b` (cta and blocked:nb, provisional; `scripts/transcribe/potri_transcribe.py c6aa0c2b
provisional --date 2026-10-26`) and `f517330f` / `f517330f-gap` (2026-10-26: sm_89 double from the
measured float ranking, cdouble from cfloat's, the over-cap float/cfloat cells from the nearest measured
cell, every sm_120 row from sm_89's; `scripts/transcribe/potri_transcribe.py f517330f measured --date
2026-10-26`; @ref perf_potri).
potrs `136b6464`: the sm_89 deep float and cfloat best records' order copied, untimed, to sm_89 double
(from float) and cdouble (from cfloat) and to every sm_120 dtype (FP64 is not tuned on a 1:64 part; no
sm_120 run; `scripts/transcribe/potrs_transcribe.py 136b6464 2026-10-26`; @ref perf_potrs).
Each of those three commands, run with `--ledger-dir` on a copy of the ledgers, appends lines
byte-identical to its run's.
sygvd `ab235065` (both devices, no timing: `compose` first everywhere, @ref perf_sygvd); sygst
`fd79f23c` (keys n, batch) and `7d0b0f84` (keys itype, uplo, n, batch), both written by
`scripts/transcribe/sygst_transcribe.py` as of the commit the run id names (committed in `39a8ec07`; no
timing: sm_89 double and cdouble, every sm_120 dtype and the sm_89 cells past the memory cap copy the
nearest sm_89 `deep` float or cfloat row with the same itype and uplo, @ref perf_sygvd, "sygst:
measured"). `scripts/transcribe/ledger_add_key.py` (`7cb2217c`) gave the earlier cells `itype=1,uplo=L`.
The potrf sm_89 Upper rows are run `custom-5dc7f884`, `factor_bench` medians
([the blocked Upper driver](../docs/perf/potrf.md#the-blocked-upper-driver)). The potri (`c6aa0c2b`, `f517330f`) and potrs transcribers are
in `scripts/transcribe/` (committed in `a01aa47d`); the older ones' sources are in git history, for example
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
- **gemm, 2026-10-21.** float and cfloat re-measured at deep (refinement cap 2x) on the round-5
  kernels, the first rows that time `splitk` (`docs/perf/gemm.md`); double and cdouble keep their
  transcribed rows (FP64 is not tuned on 1:64 parts).
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
