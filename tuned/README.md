# Tuned selection tables

One file per (op, dtype, device): `<op>.<dtype>.<device>.txt`, e.g. `potrf.float.sm_120.txt`.
Each row is a measured shape and the ranked list of every candidate timed there, fastest first
after the 3% tie rule. The library embeds these files at build time and `select::choose` picks
the first runnable entry of the nearest row. Format, lookup and borrowing rules:
`docs/design/flat-kernel-selection.md` §5.4-§5.5.

These files are plain git, not LFS, so that table changes stay readable in diffs.

## What is here

Every op ships a table for every dtype (float, double, cfloat, cdouble) on sm_89 and sm_120, and
spmm also on the CPU, so none of these devices borrows (R8); any other device borrows and warns
once. `tuned_tables_tests` (`EveryOpShipsATableForEveryDtypeOnEveryShippedDevice`) holds this
inventory. Three kinds of source:

- **measured**: timed by the tuner (`source=tuner:<raw jsonl>`, raw in `benchmarks/results/tuning/`);
- **converted**: timed forced-route sweeps (`source=benchmarks/results/routing/<sweep>.jsonl`);
- **transcribed**: the old router's preference order, untimed (`source=transcribed:<sha>`, CSV in
  `transcribed/`). A transcription marked "relabelled" is the sm_89 CSV written for sm_120 with
  `--device sm_120` (header `transcriber_device=sm_89`): the old router read no architecture.

| op | dtype | sm_89 | sm_120 | cpu |
|---|---|---|---|---|
| potrf | all | converted (`sm89_potrf_archive.jsonl`) | converted (`sm120_potrf_sweep{,_edges}.jsonl`) | — |
| posv | all | transcribed `7e71a6e0` (`posv.sm_89.csv`) | converted (`sm120_posv_sweep.jsonl`) | — |
| trsm | float, double | transcribed `8b9adeb3` (`trsm.sm_89.csv`) | **measured** (`tuning/trsm.<dtype>.sm_120.jsonl`) | — |
| trsm | cfloat, cdouble | transcribed `8b9adeb3` (`trsm.sm_89.csv`) | transcribed, relabelled from `trsm.sm_89.csv` | — |
| gemm | all | transcribed `424a45bc` (`gemm.sm_89.csv`) | transcribed, relabelled from `gemm.sm_89.csv` | — |
| gemv | all | transcribed `424a45bc` (`gemv.sm_89.csv`) | transcribed `424a45bc` (`gemv.sm_120.csv`) | — |
| geqrf | all | transcribed `424a45bc` (`geqrf.csv`, both devices) | transcribed (same CSV) | — |
| orgqr | all | transcribed `424a45bc` (`orgqr.csv`, both devices) | transcribed (same CSV) | — |
| ormqr | all | transcribed `424a45bc` (`ormqr.csv`, both devices) | transcribed (same CSV) | — |
| getrf | all | transcribed `424a45bc` (`getrf.csv`, both devices) | transcribed (same CSV) | — |
| getrs | all | transcribed `424a45bc` (`getrs.csv`, both devices) | transcribed (same CSV) | — |
| getri | all | transcribed `424a45bc` (`getri.sm_89.csv`) | transcribed `424a45bc` (`getri.sm_120.csv`) | — |
| gesv | all | transcribed `424a45bc` (`gesv.sm_89.csv`) | transcribed `424a45bc` (`gesv.sm_120.csv`) | — |
| gesvd | all | transcribed `424a45bc` (`gesvd.csv`, both devices) | transcribed (same CSV) | — |
| spmm | all | transcribed `424a45bc` (`spmm.sm_89.csv`) | transcribed `424a45bc` (`spmm.sm_120.csv`) | transcribed (`spmm.cpu.csv`) |
| syev | all | transcribed `424a45bc` (`syev.sm_89.csv`) | transcribed `424a45bc` (`syev.sm_120.csv`) | — |

124 files: 60 per GPU device (15 ops x 4 dtypes) and 4 cpu. Timed: 10 on sm_120 (potrf x 4 and posv
x 4 converted, trsm float and double measured) and the 4 sm_89 potrf tables; the other 110 replay an
old router.

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
getrs, orgqr, ormqr, spmm, syev), `7e71a6e0` (posv sm_89, 4) and `8b9adeb3` (trsm sm_89, 6). Their
sources, and the off-grid data-gate scripts that came with them, are retrievable from the parent
of the deletion commit, e.g. `git show 0bd26dfe:tools/transcribe/posv_transcribe.cc`. A
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
`tools/tune/batchlas_tune` (usage, protocol and raw schema: `tools/tune/README.md`). The tuner writes
raw JSONL and calls `python3 scripts/sweep_to_table.py --tuner <jsonl> --out tuned`, so rows are
formatted by the same code as the converted tables; `--check` re-derives a tuned table from its raw
file when that file is present, and a tuned table replaces the converted one for its
(op, dtype, device). Staleness (§6.5): `python3 .github/ci/check_tuned_tables.py` and the CMake
configure step recompute each op's kernel hash from the source list in `tools/tune/<op>_spec.cc`
and warn, never fail, on a table whose `kernels=` differs or says `unknown`.

The two tuner tables are `trsm.{float,double}.sm_120.txt`, from
`benchmarks/results/tuning/trsm.{float,double}.sm_120.jsonl` (provenance in the README there). The
tuner was stopped during cfloat, so cfloat and cdouble stay transcribed.

## Staleness

Every converted or transcribed table says `kernels=unknown` and is therefore reported stale. That
is intended: they stay stale until phase 4, when `tools/tune` retunes each op on each device and
stamps the kernel-source hash. The trsm tuner tables say `kernels=c923160f` and are reported stale
against today's `1dbe529b`. Two changes to the hashed sources sit between the sweep and HEAD, and
neither touches a kernel or a candidate: the removal of an unused `#include "gemm_kernels.hh"` from
`src/sycl/trsm_native.cc`, and the phase-5 rip's edit of `src/ops/trsm/choice.hh` (the `aliases`
array deleted, `Rules{aliases, last_resort}` -> `Rules{last_resort}`, two comments reworded). So
the timings still describe the shipped kernels. The potrf sm_89 tables come from an archive across several kernel eras (only
`kernel_current` rows are kept) and have no `lpanel` timings at all; `Lpanel{16}` has never been
timed on any device.
