# Tuned selection tables

One file per (op, dtype, device): `<op>.<dtype>.<device>.txt`, e.g. `potrf.float.sm_120.txt`.
Each row is a measured shape and the ranked list of every candidate timed there, fastest first
after the 3% tie rule. The library embeds these files at build time and `select::choose` picks
the first runnable entry of the nearest row. Format, lookup and borrowing rules:
`docs/design/flat-kernel-selection.md` §5.4-§5.5.

These files are plain git, not LFS, so that table changes stay readable in diffs.

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

## Staleness

Every converted table says `kernels=unknown` and is therefore reported stale. That is intended:
they stay stale until phase 4, when `tools/tune` retunes potrf on each device and stamps the
kernel-source hash. sm_89 tables come from an archive across several kernel eras (only
`kernel_current` rows are kept) and have no `lpanel` timings at all; `Lpanel{16}` has never been
timed on any device.
