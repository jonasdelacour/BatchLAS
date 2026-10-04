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

## Staleness

Every converted table says `kernels=unknown` and is therefore reported stale. That is intended:
they stay stale until phase 4, when `tools/tune` retunes potrf on each device and stamps the
kernel-source hash. Both devices' tables come from a forced-route sweep of the current kernels
(`benchmarks/results/routing/README.md`); `Lpanel{16}` has never been timed on any device.
