# Tuned selection tables

One file per (op, dtype, device): `<op>.<dtype>.<device>.txt`, e.g. `potrf.float.sm_120.txt`.
Each row is a measured shape and the ranked list of every candidate timed there, fastest first
after the 3% tie rule. The library embeds these files at build time and `select::choose` picks
the first runnable entry of the nearest row. Format, lookup and borrowing rules:
`docs/design/flat-kernel-selection.md` §5.4-§5.5.

These files are plain git, not LFS, so that table changes stay readable in diffs.

## What is here

| op | device | source | file |
|---|---|---|---|
| potrf | sm_120, sm_89 | converted route sweeps (timed) | `potrf.<dtype>.<device>.txt` |
| posv | sm_89 | **transcribed** old router, untimed (`source=transcribed:7e71a6e0`) | `posv.<dtype>.sm_89.txt`, from `transcribed/posv.sm_89.csv` |
| posv | sm_120 | none yet: the sweep `benchmarks/results/routing/sm120_posv_sweep.jsonl` is converted later; until then sm_120 borrows the sm_89 posv tables and warns once | — |
| trsm | sm_89 | **transcribed** old router, untimed (`source=transcribed:8b9adeb3`) | `trsm.<dtype>.sm_89.txt`, from `transcribed/trsm.sm_89.csv` |
| trsm | sm_120 | none yet: a `tools/tune` sweep; until then sm_120 borrows the sm_89 trsm tables and warns once | — |
| gemm | sm_89 | **transcribed** old routing, untimed (`source=transcribed:424a45bc`) | `gemm.<dtype>.sm_89.txt`, from `transcribed/gemm.sm_89.csv` |
| gemm | sm_120 | none yet: a `tools/tune` sweep; until then sm_120 borrows the sm_89 gemm tables and warns once | — |

## How they are produced

The current `potrf.*` tables are seed tables, converted from the forced-route sweeps in
`benchmarks/results/routing/` (provenance in the README there):

    python3 scripts/sweep_to_table.py           # rewrite tuned/potrf.*.txt
    python3 scripts/sweep_to_table.py --check   # verify tuned/ matches the sweeps (CI gate §10.2)

Do not edit them by hand: `--check` fails on any difference from the sweeps.

Transcribed tables (header `source=transcribed:<sha>`, entries `<spelling> -`) hold an old
router's preference order per grid cell, untimed. A per-op C++ transcriber writes a CSV and
`python3 scripts/sweep_to_table.py --transcribe <csv> --sha <sha>` turns it into tables;
`--check` re-derives them from the CSV named in their `transcriber_csv=` header line. Each row
is all-timed or all-untimed, an untimed row needs a `source=transcribed:<sha>` header, and the
sha must be hex (`--sha` is resolved with `git rev-parse`); a table may hold both row kinds. The
C++ loader and `--check` both apply exactly these rules. The converter's module docstring documents both input schemas and
how to add an op.

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

A transcribed row reproduces a deleted window; it is not a measurement. It is replaced by a timed
row when the tuner sweeps that device.

Tuned tables (header `source=tuner:<raw jsonl>` and a real `kernels=<hash>`) come from
`tools/tune/batchlas_tune` (usage, protocol and raw schema: `tools/tune/README.md`). The tuner writes
raw JSONL and calls `python3 scripts/sweep_to_table.py --tuner <jsonl> --out tuned`, so rows are
formatted by the same code as the converted tables; `--check` re-derives a tuned table from its raw
file when that file is present, and a tuned table replaces the converted one for its
(op, dtype, device). Staleness (§6.5): `python3 .github/ci/check_tuned_tables.py` and the CMake
configure step recompute each op's kernel hash from the source list in `tools/tune/<op>_spec.cc`
and warn, never fail, on a table whose `kernels=` differs or says `unknown`. Every table here is
still `unknown` until phase 4 retunes it.

## Staleness

Every converted or transcribed table says `kernels=unknown` and is therefore reported stale. That
is intended: they stay stale until phase 4, when `tools/tune` retunes each op on each device and
stamps the kernel-source hash. The potrf sm_89 tables come from an archive across several kernel eras (only
`kernel_current` rows are kept) and have no `lpanel` timings at all; `Lpanel{16}` has never been
timed on any device.
