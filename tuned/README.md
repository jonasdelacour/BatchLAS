# Tuned selection tables {#tuned_tables_readme}

> **Status:** current · the generated @ref selection_tables page is always current.

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

Counts: 59 files, 29 per GPU device plus `spmm.cpu.txt`.

| device | ledger tiers | where from |
|---|---|---|
| sm_120 | `deep`, plus `transcribed` rows where the deep lattice measured nothing near and a few `custom` rows (gemv, posv, trmm); gemm double and cdouble are `transcribed` only; potri and potrs all `transcribed` (from sm_89's measured rows); sytrf, sysv, sytrs, sygst and sygvd all `transcribed` | "The sm_120 deep tables" below; potri: @ref perf_potri; potrs: @ref perf_potrs; sytrf, sysv and sytrs: @ref perf_sytrf; sygst and sygvd: @ref perf_sygvd |
| sm_89 | `transcribed` (the old routers' order, untimed); potrf `converted` plus 12 `deep` float Upper rows; geqrf 16 `deep` wide rows (both re-measured 2026-10-24); potri and potrs float and cfloat `deep` (2026-10-26), double and cdouble `transcribed` from them; sygst float and cfloat `deep` (2026-10-27, every itype and uplo), its double and cdouble `transcribed` from them; sytrs float and cfloat `preview` (double, cdouble `transcribed` from them) and sysv float and cfloat `preview` (2026-10-26, @ref perf_sytrf) | the old transcribed tables and the sm_89 potrf forced-route archive, migrated (`docs/design/tuning-data.md`) |
| cpu (spmm) | `transcribed` | likewise |

Transcribed rows reproduce deleted routers' windows (or copy a measured ranking, untimed) at their
grid points; a measured row near them replaces them. Provenance is the ledger run line, run id
`transcribed-<sha>`. Transcribers live in `scripts/transcribe/` (their docstrings give the
invocation; re-running with `--ledger-dir` on a ledger copy appends byte-identical lines); older
ones are in git history (`git show 0bd26dfe:tools/transcribe/posv_transcribe.cc`), the migration
script in `git log -- scripts/migrate_tuning_data.py`.

| Run id | Ops | Rule |
|---|---|---|
| `7e71a6e0`, `8b9adeb3` | posv, trsm | old router windows |
| `424a45bc` | gemm, gemv, getri, gesv, spmm, syev, geqrf, orgqr, ormqr, getrf, getrs, gesvd | old router windows |
| `ff340fc6` | symm, syrk, syr2k, trmm | old router windows |
| `141d31db`, `167a4eaf` | gesvd | complex general > 32 gains `blocked`; sm_89 cfloat 33..64 `blocked` first (@ref perf_gesvd) |
| `26168c21` | complex symm/syrk/syr2k, all hemm/herk/her2k | the deleted `cublas.cc` rules (@ref perf_level3) |
| `c6aa0c2b`, `f517330f`, `f517330f-gap` | potri | sm_89 double/cdouble and all sm_120 from the measured float/cfloat ranking (`potri_transcribe.py`, @ref perf_potri) |
| `136b6464` | potrs | same copy rule (`potrs_transcribe.py`, @ref perf_potrs) |
| `ab235065` | sygvd | `compose` first everywhere (@ref perf_sygvd) |
| `fd79f23c`, `7d0b0f84` | sygst | nearest sm_89 `deep` row with the same itype, uplo (`sygst_transcribe.py`; earlier cells keyed `itype=1,uplo=L` by `ledger_add_key.py`) |
| `30cc8e06`, `transcribed-sytrf-blocked` | sytrf, sysv | `blocked` first from n = 256, second 33..128, third n <= 32 (@ref perf_sytrf) |
| `transcribed-sytrs-d67ad754` | sytrs | measured sm_89 float/cfloat order (`sytrs_transcribe.py`) |

The potrf sm_89 Upper rows are run `custom-5dc7f884`, `factor_bench` medians
([the blocked Upper driver](../docs/perf/potrf.md#the-blocked-upper-driver)).

## The sm_120 deep tables

All sm_120 tables come from one `--tier deep` campaign over all 19 ops and every dtype each op
instantiates, on threadripper02 (4x RTX PRO 6000 Blackwell, sm_120, CUDA 13.2, DPC++
`/opt/dpcpp-cuda`), 2026-10-08 06:23 to 2026-10-09 04:54 CEST, four starts of
`batchlas_tune all --tier deep --devices 0,1,2,3`, plus a re-measure of 250 cells the box's other
user had damaged (`--remeasure-keys`). Details are in
[tiered tuning](../docs/design/tiered-tuning.md), "Engine: the sm_120 deep run".

- **Later re-measures at deep:** gemv, potrf, gesv, posv, syev, trmm (2026-10-11/12, after the
  `batchlas::verify` specs and defect-11 / stedc fixes; the other families carried over by
  sample-proof rehash lines, docs/design/tuning-data.md "Kernel hashes"); gemm float and cfloat
  2026-10-21 (first rows timing `splitk`). gemm double and cdouble stay transcribed (FP64 is not
  tuned on 1:64 parts).
- **Dimension cap 2048** (`--max-dim`): above it the nearest measured row decides.
- **Refinement coverage is uneven.** Every starting lattice is complete. geqrf, ormqr, getri, posv,
  gesv, orgqr, syev, gesvd and complex getrs are only lightly refined, so their crossovers sit on
  lattice spacing.
- **Accuracy is not timed.** LOBPCG pins its projected syev solves to the CTA solver at n <= 32 for
  that reason (`docs/perf/syevx.md`, "LOBPCG: the accuracy pin on the projected syev").

## Staleness

The hashes and the staleness warning are specified in
[the tuning data layout](../docs/design/tuning-data.md#tuning-data-kernel-hash).
