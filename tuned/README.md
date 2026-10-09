# Tuned selection tables {#tuned_tables_readme}

One file per (op, dtype, device): `<op>.<dtype>.<device>.txt`, e.g. `potrf.float.sm_120.txt`.
Each row is a measured shape and the ranked list of every candidate timed there, fastest first
after the 3% tie rule. The library embeds these files at build time and `select::choose` picks
the first runnable entry of the nearest row. Format, lookup and borrowing rules:
[the flat kernel selection design](../docs/design/flat-kernel-selection.md), §5.4-§5.5.
The docs site regenerates a per-op summary of every file here on each build (families, keys,
provenance and which family ranks first where): the "Kernel selection tables" page,
@ref selection_tables. The tuner that writes them is described in
[tools/tune/README.md](../tools/tune/README.md).

These files are plain git, not LFS, so that table changes stay readable in diffs.

## What is here

The inventory below is as of 2026-10-09; the generated selection-tables page is always current.
Every op ships a table for every dtype it instantiates (float, double, cfloat, cdouble; symm, syrk
and syr2k are real-only) on sm_89 and sm_120, and spmm also on the CPU, so none of these devices
borrows (R8); any other device borrows and warns once. `tuned_tables_tests` (`EveryOpShipsATableForEveryDtypeOnEveryShippedDevice`) holds this
inventory. Three kinds of source:

- **deep**: a tiered `--tier deep` run's ledger (`source=ledger:<dir>`, raw in
  `benchmarks/results/tuning/ledger/`), see "The sm_120 deep tables" below;
- **measured**: timed by the old tuner (`source=tuner:<raw jsonl>`, raw in `benchmarks/results/tuning/`);
- **converted**: timed forced-route sweeps (`source=benchmarks/results/routing/<sweep>.jsonl`);
- **transcribed**: the old router's preference order, untimed (`source=transcribed:<sha>`, CSV in
  `transcribed/`). A transcription marked "relabelled" is the sm_89 CSV written for sm_120 with
  `--device sm_120` (header `transcriber_device=sm_89`): the old router read no architecture.

| op | dtype | sm_89 | sm_120 | cpu |
|---|---|---|---|---|
| potrf | all | converted (`sm89_potrf_archive.jsonl`) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| posv | all | transcribed `7e71a6e0` (`posv.sm_89.csv`) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| trsm | all | transcribed `8b9adeb3` (`trsm.sm_89.csv`) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| gemm | all | transcribed `424a45bc` (`gemm.sm_89.csv`) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| gemv | all | transcribed `424a45bc` (`gemv.sm_89.csv`) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| geqrf | all | transcribed `424a45bc` (`geqrf.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| orgqr | all | transcribed `424a45bc` (`orgqr.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| ormqr | all | transcribed `424a45bc` (`ormqr.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| getrf | all | transcribed `424a45bc` (`getrf.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| getrs | all | transcribed `424a45bc` (`getrs.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| getri | all | transcribed `424a45bc` (`getri.sm_89.csv`) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| gesv | all | transcribed `424a45bc` (`gesv.sm_89.csv`) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| gesvd | all | transcribed `424a45bc` (`gesvd.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| spmm | all | transcribed `424a45bc` (`spmm.sm_89.csv`) | **deep** (`ledger/<op>.<dtype>.sm_120`) | transcribed (`spmm.cpu.csv`) |
| syev | all | transcribed `424a45bc` (`syev.sm_89.csv`) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| symm | float, double | transcribed `ff340fc6` (`symm.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| syrk | float, double | transcribed `ff340fc6` (`syrk.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| syr2k | float, double | transcribed `ff340fc6` (`syr2k.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |
| trmm | all | transcribed `ff340fc6` (`trmm.csv`, both devices) | **deep** (`ledger/<op>.<dtype>.sm_120`) | — |

144 files: 70 per GPU device (16 ops x 4 dtypes, plus symm, syrk and syr2k x 2) and 4 cpu. Timed:
all 70 on sm_120 (deep ledger tables) and the 4 sm_89 potrf tables; the other 70 (sm_89 and cpu)
replay an old router. A deep table keeps a transcribed row only where no measured row lies within
the lattice spacing (`tiers=...,transcribed:<n>` in its header counts them).

## The sm_120 deep tables

All 70 sm_120 tables come from one `--tier deep` campaign over all 19 ops and every dtype each
op instantiates, on threadripper02 (4x RTX PRO 6000 Blackwell, sm_120, CUDA 13.2, DPC++
`/opt/dpcpp-cuda`), 2026-10-08 06:23 to 2026-10-09 04:54 CEST, four restarts of
`batchlas_tune all --tier deep --devices 0,1,2,3`. On 2026-10-09 two follow-up runs went into the
same ledger: a re-measure of 250 cells the box's other user had damaged (`--remeasure-keys`,
design page "Engine: re-measuring named cells"), and a fresh deep gemv run, because the defect-13
fix edited `src/ops/gemv/gemv.cc`, which every gemv family hashes. The ledger is
`benchmarks/results/tuning/ledger/` (LFS). Regenerate with

    python3 scripts/sweep_to_table.py --ledger benchmarks/results/tuning/ledger --out tuned --replace-timed

Caveats:

- **Dimension cap 2048.** No cell was planned with a matrix dimension above 2048 (`--max-dim`). The
  first run measured some larger cells before the cap existed; they stay. Above the cap the
  nearest measured row decides.
- **Refinement coverage is uneven.** Every op's starting lattice is complete, but refinement was
  cut short when the campaign was stopped. geqrf, ormqr, getri, posv, gesv, orgqr, syev, gesvd and
  complex getrs are only lightly refined, so their crossovers sit on lattice spacing, not on a
  bisected edge.
- **Accuracy is not timed.** A table ranks by time only. LOBPCG pins its projected syev solves to
  the CTA solver at n <= 32 for that reason (`docs/perf/syevx.md`, "LOBPCG: the accuracy pin on
  the projected syev").
- **sm_120 only.** The sm_89 tables are untouched; a device with no table of its own still borrows
  the nearest one.

## How they are produced

The `potrf.*` tables and the sm_120 `posv.*` tables are seed tables, converted from the forced-route
sweeps in `benchmarks/results/routing/` (provenance in the README there):

    python3 scripts/sweep_to_table.py --date 2026-10-04   # rewrite tuned/potrf.*.txt, posv.*.sm_120.txt
    python3 scripts/sweep_to_table.py --check             # verify tuned/ matches the sweeps (§10.2)

Do not edit them by hand: `--check` fails on any difference from the sweeps (the header date
included, hence `--date`). The posv sweep was resumed after guard refusals, and its resume re-ran
224 complete pass-2 cells; its OpSpec source sets `dedupe_latest`, so a later kept row replaces
an earlier one for the same (cell, arm, pass) instead of failing as a duplicate (details in the
routing README).

Transcribed tables (header `source=transcribed:<sha>`, entries `<spelling> -`) hold an old
router's preference order per grid cell, untimed. A per-op C++ transcriber writes a CSV and
`python3 scripts/sweep_to_table.py --transcribe <csv> --sha <sha>` turns it into tables;
`--check` re-derives them from the CSV named in their `transcriber_csv=` header line. Each row
is all-timed or all-untimed, an untimed row needs a `source=transcribed:<sha>` header, and the
sha must be hex (`--sha` is resolved with `git rev-parse`); a table may hold both row kinds. The
C++ loader and `--check` both apply exactly these rules. The converter's module docstring documents both input schemas and
how to add an op.

**The transcribers are deleted** (phase 5 rip). Each one compiled against a tree that still had
the old router, so none could build here. The tables carry the old-router commit each was compiled
against: `source=transcribed:424a45bc` (100 tables: gemm, gemv, geqrf, gesv, gesvd, getrf, getri,
getrs, orgqr, ormqr, spmm, syev), `7e71a6e0` (posv sm_89, 4), `8b9adeb3` (trsm sm_89, 6) and
`ff340fc6` (symm, syrk, syr2k, trmm: 20). Their sources, and the off-grid data-gate scripts that
came with them, are retrievable from the parent of the deletion commit, e.g.
`git show 0bd26dfe:tools/transcribe/posv_transcribe.cc`; the level-3 four's from the last merge of
their wave, `git show eeacaaa9:tools/transcribe/<op>_transcribe.cc` and `<op>_gate.py`. A
`tools/transcribe/...` path below, in `docs/perf/`, or in the per-op phase 5 notes (folded into the spec §12; originals at `git show 94cefb3a:docs/design/flat-select-p5/<op>.md`), means
that git path. The CSVs in `transcribed/` stay, so `--check` still re-derives every table.

The posv sm_89 tables are the first transcribed ones. `tools/transcribe/posv_transcribe.cc` is
built host-only against a checkout that still has `route_posv.hh` (its header gives the g++ line)
and evaluates that router's own predicates at every cell of the `src/ops/posv/choice.hh` grid,
so a row reads `tiny - | cta - | blocked -` inside the old tiny window and `cta - | blocked -`
elsewhere. Transcriber CSVs live in `transcribed/` (plain git; the embed only reads `*.txt` at
this level). Regenerate with

    g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/posv_transcribe.cc -o /tmp/pt
    /tmp/pt sm_89 > tuned/transcribed/posv.sm_89.csv
    python3 scripts/sweep_to_table.py --transcribe tuned/transcribed/posv.sm_89.csv --sha 7e71a6e0

The trsm sm_89 tables come from `tools/transcribe/trsm_transcribe.cc` the same way (built against
8b9adeb3, which still has `route_trsm.hh`; the g++ line is in its header, then `--transcribe
tuned/transcribed/trsm.sm_89.csv --sha 8b9adeb3`). On the grid (batch >= 128) the old router
preferred every native route, so a row is `cta - | blocked - | vendor -` at order <= 32 and
`blocked - | vendor -` above. Its batch floor (batch < 8 went to the vendor) and the float
Side::Right rule (batch < 128 above order 32) sit below the grid and are not transcribed.

The gemm sm_89 tables come from `tools/transcribe/gemm_transcribe.cc`, which links against a
BUILT 424a45bc tree (the old `select_kernel_variant` lives in libbatchlas_sycl; the build line is
in its header), then `--transcribe tuned/transcribed/gemm.sm_89.csv --sha 424a45bc`. A row is the
old vendor-vs-native decision (`route_gemm.hh` preferred(), the predicate the cuBLAS TU's
re-route consulted) with the old native kernel (`select_kernel_variant`) mapped to its family
spelling, then the old fallbacks (tiled/direct, plus `small` for a real max(m, n, k) <= 64: the
one native launch that survives batch > 65535); `vendor` leads where the old route was the
vendor. `layout=packed` cells were evaluated on contiguous 16-byte-aligned views, `strided` on
ld = rows + 1. Beyond the tuner grid the CSV carries edge rows that bracket the old predicate's
below-grid edges: real types at batch {1, 63, 64}, double at k {1, 2}, float NN squares 1, 2, 4,
40, 49, 56 and one-axis-off neighbours of the small squares (the transcriber's header lists them).

The sm_120 gemm tables and the sm_120 trsm cfloat/cdouble tables are those same sm_89 CSVs written
for sm_120 (neither transcriber reads a device fact; its device argument only labels the rows):

    python3 scripts/sweep_to_table.py --transcribe tuned/transcribed/gemm.sm_89.csv --sha 424a45bc --device sm_120
    python3 scripts/sweep_to_table.py --transcribe tuned/transcribed/trsm.sm_89.csv --sha 8b9adeb3 --device sm_120

The second command leaves `trsm.{float,double}.sm_120.txt` alone: `--transcribe` never overwrites
a tuner table.

The level-3 four (symm, syrk, syr2k, trmm) were transcribed at `ff340fc6`, the last commit with
their hand-written rules. Each `<op>_transcribe.cc` is host-only C++ holding byte-for-byte copies
of the old predicate bodies (`<op>_gate.py --fidelity` diffs them against `git show ff340fc6:`;
build lines in the file headers), evaluated at every cell of `src/ops/<op>/choice.hh`'s grid with
capacities unlimited (`can_run` re-applies them). It writes one CSV for both devices, then
`python3 scripts/sweep_to_table.py --transcribe tuned/transcribed/<op>.csv --sha ff340fc6`. A row
is the old vendor-present Auto choice, then every other structurally runnable candidate of that
dtype, `vendor` last unless it was the Auto choice. symm and syrk key on `form` (sq|tall|wide); a
cell whose extents contradict its form holds the decision at the form's representative shape.
`<op>_gate.py --data` replays random off-grid points through the old rules and through
`sweep_to_table.nearest` plus a `can_run` model: 100% agreement for every op, dtype and device
(details: `docs/design/flat-kernel-selection.md` §12 "Level-3 four").

A transcribed row reproduces a deleted window; it is not a measurement. It is replaced by a timed
row when the tuner sweeps that device.

A transcribed row reproduces the old decision only AT its grid point. Between grid points the
gemm tables differ from the old `select_kernel_variant` (phase-5 review, spec §12 Phase 3.4):
the first native entry differs on ~6.7% of random off-grid double shapes and ~31.7% of float
ones, including the factorizations' panel updates (e.g. geqrf double NN 32x16x16 now `direct`,
was Tiled16), because the double Direct/Tiled16 and float register-tile edges are bracketed only
on squares. And `wide`/`reg` rows transcribed at aligned (multiple-of-64) `packed` points now also
serve packed NON-multiple shapes, which run the predicated leg the old router never chose (e.g.
double 304^3 b64: `wide:m=64:n=64:k=16`, was Tiled16). Both are untimed and pending the retune.

Tuned tables (header `source=tuner:<raw jsonl>` and a real `kernels=<hash>`) come from
`tools/tune/batchlas_tune` (usage, protocol and raw schema: [tools/tune/README.md](../tools/tune/README.md)). The tuner writes
raw JSONL and calls `python3 scripts/sweep_to_table.py --tuner <jsonl> --out tuned`, so rows are
formatted by the same code as the converted tables; `--check` re-derives a tuned table from its raw
file when that file is present, and a tuned table replaces the converted one for its
(op, dtype, device). Staleness (§6.5): `python3 .github/ci/check_tuned_tables.py` and the CMake
configure step recompute each op's kernel hash from the source list in `tools/tune/<op>_spec.cc`
and warn, never fail, on a table whose `kernels=` differs or says `unknown`.

Ledger tables (header `source=ledger:<dir>`) come from tiered runs (`batchlas_tune --tier
preview|coarse|deep`, design: [tiered tuning](../docs/design/tiered-tuning.md)). Each run appends
per-cell records to a ledger directory `<op>.<dtype>.<device>/`, and
`python3 scripts/sweep_to_table.py --ledger <dir or root> --out <scratch dir>` writes the table: per cell
the best current record (deep > coarse > preview > custom), each row ending in `# <tier>`, the
header carrying `tiers=` row counts and `family_kernels=<family>:<hash>,...` (the per-family
kernel hashes that decide which records are stale). A changed hash on the winner's family drops
the record. A changed hash on a candidate that ran but lost makes the cell partly stale: the row
stays, and the next run re-races that candidate. A candidate that could not run there is not
compared. `--check` re-derives the table from the ledger its `source=` names. The converter
refuses to overwrite a converted or `tuner:` table with a ledger table unless `--replace-timed` is
given (transcribed and ledger tables are replaced without it), so compare a scratch `--out` with
`tuned/` before switching a table. Every sm_120 table is ledger-sourced since 2026-10-09 (above).

The old tuner's trsm float/double sm_120 tables
(`benchmarks/results/tuning/trsm.{float,double}.sm_120.jsonl`) and the converted potrf and posv
sm_120 tables were replaced by the deep ledger tables with `--replace-timed`; their raw files stay.

## Staleness

Every converted or transcribed table says `kernels=unknown` and is therefore reported stale. That
is intended: they stay stale until phase 4, when `tools/tune` retunes each op on each device and
stamps the kernel-source hash. The sm_120 deep tables carry real `kernels=` and `family_kernels=`
hashes from the tree they were generated in, so they read current until a hashed source changes. The potrf sm_89 tables come from an archive across several kernel eras (only
`kernel_current` rows are kept) and have no `lpanel` timings at all; `Lpanel{16}` has never been
timed on any device.
