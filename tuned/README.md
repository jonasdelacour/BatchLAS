# Tuned selection tables {#tuned_tables_readme}

> **Status:** current · inventory as of 2026-10-09 · the generated @ref selection_tables page is always current.

One file per (op, dtype, device): `<op>.<dtype>.<device>.txt`, for example `potrf.float.sm_120.txt`.
Each row is a measured shape and the ranked list of every candidate timed there, fastest first
after the 3% tie rule. The library embeds these files at build time. `select::choose` takes the
first runnable entry of the nearest row. Format and lookup rules are in
[the flat kernel selection design](../docs/design/flat-kernel-selection.md) §5.4-§5.5. The tuner that
writes them is in [tools/tune/README.md](../tools/tune/README.md).

The files are plain git, not LFS, so table changes stay readable in diffs.

## Inventory

Every op ships a table for each dtype it instantiates (float, double, cfloat, cdouble; symm, syrk
and syr2k are real only) on sm_89 and sm_120. spmm also ships a CPU table. Any other device borrows
the nearest table and warns once. `tuned_tables_tests` (`EveryOpShipsATableForEveryDtypeOnEveryShippedDevice`)
checks this inventory.

Each table has one of four sources:

- **deep**: a tiered `--tier deep` run's ledger, `source=ledger:<dir>`, raw data in
  `benchmarks/results/tuning/ledger/`. See "The sm_120 deep tables" below.
- **measured**: timed by the old tuner, `source=tuner:<raw jsonl>`, raw data in `benchmarks/results/tuning/`.
- **converted**: timed forced-route sweeps, `source=benchmarks/results/routing/<sweep>.jsonl`.
- **transcribed**: an old router's preference order, untimed, `source=transcribed:<sha>`, CSV in
  `transcribed/`. "Relabelled" means the sm_89 CSV was written for sm_120 with `--device sm_120`.

| op | dtype | sm_89 | sm_120 |
|---|---|---|---|
| potrf | all | converted (`sm89_potrf_archive.jsonl`) | **deep** (`ledger/<op>.<dtype>.sm_120`) |
| posv | all | transcribed `7e71a6e0` (`posv.sm_89.csv`) | **deep** |
| trsm | all | transcribed `8b9adeb3` (`trsm.sm_89.csv`) | **deep** |
| gemm | all | transcribed `424a45bc` (`gemm.sm_89.csv`) | **deep** |
| gemv, getri, gesv, spmm, syev | all | transcribed `424a45bc` (`<op>.sm_89.csv`) | **deep** |
| geqrf, orgqr, ormqr, getrf, getrs, gesvd | all | transcribed `424a45bc` (one CSV per op, both devices) | **deep** |
| symm, syrk, syr2k | float, double | transcribed `ff340fc6` (one CSV per op, both devices) | **deep** |
| trmm | all | transcribed `ff340fc6` (`trmm.csv`, both devices) | **deep** |

spmm also has a CPU table, transcribed (`spmm.cpu.csv`).

Counts: 144 files, 70 per GPU device (16 ops x 4 dtypes, plus symm, syrk and syr2k x 2 dtypes) and
4 CPU. All 70 sm_120 tables are timed (deep ledger tables), plus the 4 sm_89 potrf tables. The other
70 (sm_89 and CPU) replay an old router. A deep table keeps a transcribed row only where no measured
row lies within the lattice spacing (`tiers=...,transcribed:<n>` in its header counts them).

## The sm_120 deep tables

All 70 sm_120 tables come from one `--tier deep` campaign over all 19 ops and every dtype each op
instantiates, on threadripper02 (4x RTX PRO 6000 Blackwell, sm_120, CUDA 13.2, DPC++
`/opt/dpcpp-cuda`), 2026-10-08 06:23 to 2026-10-09 04:54 CEST, four starts of
`batchlas_tune all --tier deep --devices 0,1,2,3`. Two follow-up runs on 2026-10-09 went into the same
ledger: a re-measure of 250 cells the box's other user had damaged (`--remeasure-keys`), and a fresh
deep gemv run, because the defect-13 fix edited `src/ops/gemv/gemv.cc`, which every gemv family
hashes. The ledger is `benchmarks/results/tuning/ledger/` (LFS). Details are in
[tiered tuning](../docs/design/tiered-tuning.md), "Engine: the sm_120 deep run". Regenerate with:

    python3 scripts/sweep_to_table.py --ledger benchmarks/results/tuning/ledger --out tuned --replace-timed

- **Dimension cap 2048.** No cell was planned with a matrix dimension above 2048 (`--max-dim`). The
  first run measured some larger cells before the cap existed; they stay. Above the cap the nearest
  measured row decides.
- **Refinement coverage is uneven.** Every starting lattice is complete. geqrf, ormqr, getri, posv,
  gesv, orgqr, syev, gesvd and complex getrs are only lightly refined, so their crossovers sit on
  lattice spacing.
- **Accuracy is not timed.** LOBPCG pins its projected syev solves to the CTA solver at n <= 32 for
  that reason (`docs/perf/syevx.md`, "LOBPCG: the accuracy pin on the projected syev").
- **sm_120 only.** The sm_89 tables are untouched.

## Producing tables

**Do not edit the converted, tuner or ledger tables by hand.** Regenerate them:

    python3 scripts/sweep_to_table.py --ledger benchmarks/results/tuning/ledger   # rewrite every sm_120 table
    python3 scripts/sweep_to_table.py --date 2026-10-04   # rewrite tuned/potrf.*.sm_89.txt
    python3 scripts/sweep_to_table.py --check             # verify tuned/ against ledgers, sweeps and CSVs

Every sm_120 table is written from its ledger. The old sm_120 potrf and posv sweeps are still
validated but no longer write a table.

`--check` fails on any difference, the header date included, hence `--date`. The posv sweep
resumed after guard refusals. Its OpSpec sets `dedupe_latest`, so a later kept row replaces an
earlier one for the same (cell, arm, pass). Details are in `benchmarks/results/routing/README.md`.

### Transcribed tables

A transcribed table has the header `source=transcribed:<sha>` and entries `<spelling> -` (untimed).
Each row is an old router's preference order for one grid cell. Transcribers are deleted; their
sources are in the parent of the deletion commit, for example `git show 0bd26dfe:tools/transcribe/posv_transcribe.cc`.
The CSVs in `transcribed/` remain, so `--check` still re-derives every table.

To regenerate one, build its transcriber against the old commit and convert the CSV:

    python3 scripts/sweep_to_table.py --transcribe tuned/transcribed/<op>.<device>.csv --sha <sha>

The `--sha` must be hex; it is resolved with `git rev-parse`. Each row is all-timed or all-untimed.
An untimed row needs a `source=transcribed:<sha>` header. A table may hold both kinds.

| table | transcriber | built against | notes |
|---|---|---|---|
| posv sm_89 | `tools/transcribe/posv_transcribe.cc` | `7e71a6e0` | evaluates `route_posv.hh` at every `choice.hh` grid cell |
| trsm sm_89 | `tools/transcribe/trsm_transcribe.cc` | `8b9adeb3` | batch floor and float Side::Right rule sit below the grid and are not transcribed |
| gemm sm_89 | `tools/transcribe/gemm_transcribe.cc` | `424a45bc` (built tree) | adds edge rows below the grid (batch 1, 63, 64; double k 1, 2; float NN squares 1, 2, 4, 40, 49, 56) |
| symm, syrk, syr2k, trmm | `tools/transcribe/<op>_transcribe.cc` | `ff340fc6` | one CSV for both devices; `<op>_gate.py --data` shows 100% agreement |

Notes:

- A transcribed row reproduces a deleted window. It is not a measurement, and a timed row replaces it
  when the tuner sweeps that device.
- A transcribed row matches the old decision only at its grid point. Between points the gemm tables
  differ from the old `select_kernel_variant`: the first native entry differs on ~6.7% of random
  off-grid double shapes and ~31.7% of float shapes. Both are untimed and pending a retune.
- `wide` and `reg` rows transcribed at aligned (multiple-of-64) `packed` points also serve packed
  non-multiple shapes, which run the predicated leg the old router never chose.

### Tuner tables

A tuned table has the header `source=tuner:<raw jsonl>` and a real `kernels=<hash>`. The tuner
writes raw JSONL, then calls `python3 scripts/sweep_to_table.py --tuner <jsonl> --out tuned`.
`--check` re-derives a tuned table from its raw file when that file is present. A tuned table
replaces the converted one for its (op, dtype, device).

- No tuner table ships any more. The old trsm float/double sm_120 tables
  (`benchmarks/results/tuning/trsm.{float,double}.sm_120.jsonl`) and the converted sm_120 potrf and
  posv tables were replaced by the deep ledger tables with `--replace-timed`; their raw files stay.

### Ledger tables

A ledger table has the header `source=ledger:<dir>`. Tiered runs (`batchlas_tune --tier preview|coarse|deep`,
see [tiered tuning](../docs/design/tiered-tuning.md)) append per-cell records to
`<op>.<dtype>.<device>/` ledger directories. Generate the table with:

    python3 scripts/sweep_to_table.py --ledger <dir or root> --out <scratch dir>

- Per cell the converter keeps the best current record (deep > coarse > preview > custom). Each row
  ends in `# <tier>`.
- The header carries `tiers=` (row counts) and `family_kernels=<family>:<hash>,...`. A changed hash on
  the winner's family drops the record. A changed hash on a losing candidate marks the cell partly
  stale, and the next run re-races that candidate.
- The converter refuses to overwrite a converted or `tuner:` table without `--replace-timed`.
  Transcribed and ledger tables are replaced without it.
- Every sm_120 table is ledger-sourced since 2026-10-09 ("The sm_120 deep tables").

## Staleness

- Converted and transcribed tables say `kernels=unknown`, so they are reported stale.
- The sm_120 deep tables carry real `kernels=` and `family_kernels=` hashes from the tree they were
  generated in, so they read current until a hashed source changes.
- Retuning an op on a device stamps its kernel-source hash and clears the stale report.
- `python3 .github/ci/check_tuned_tables.py` and the CMake configure step recompute each op's hash
  from `tools/tune/<op>_spec.cc` and warn on a `kernels=` mismatch. They never fail.
- The potrf sm_89 tables come from an archive spanning several kernel versions. Only `kernel_current`
  rows are kept, and they have no `lpanel` timings. `Lpanel{16}` has never been timed on any device.
