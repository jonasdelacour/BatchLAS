# Tiered tuning {#design_tiered_tuning}

> **Covers:** the redesign of how BatchLAS fills its tuned tables and tuning constants: user-chosen
> tiers (ultra / coarse / deep), measurements that skip work which cannot change a table, a per-cell
> result ledger in which a cheaper run never overwrites a better one, and a tuning tab in benchviz.
> **Status:** design, approved in conversation on 2026-10-07; not implemented. Sub-project 1 (the
> engine) is specified in full here; sub-projects 2 and 3 are specified at the interface level and
> get their own design pass. Machine facts refer to the RTX 4090 box (sm_89) and threadripper02
> (sm_120). The current tuner is described in @ref tune_tool_readme and
> @ref design_flat_selection §6.

## Why tiered tuning

Filling the tables is the job that matters most: replacing the transcribed (untimed) tables with
measurements once per machine. Today that job does not fit on one GPU.

- `batchlas_tune` times every runnable candidate at every cell of a full lattice. Each cell costs at
  least three cold processes (a JIT pass and two timed passes), each with 1.5 s of warm-up per
  candidate, plus 16 reps. A cell is repeated whenever its two passes disagree by more than 10%.
- trsm float on sm_120 took about 16 GPU-hours: 3480 cells, 5 h 20 min on 3 GPUs
  (`benchmarks/results/tuning/README.md`). gemm works out to about 100 GPU-hours for float alone
  and has never been run. sm_89 was transcribed rather than measured to avoid "60-100 h on a
  shared box" (@ref design_flat_selection_phase3_plan).
- Only 4 of the 19 routed ops (gemm, potrf, posv, trsm) have a tuner spec. Only 16 of the 144
  shipped tables are measured. The other 128 are `source=transcribed`, and the only way to find
  that out is to read table headers.
- The `tuning_params.hh` constants come from a second system (`evaluation/tuning/`). It was run on
  float only, and its results were ported into the header by hand.

## Tiered tuning: decisions

These were decided with the maintainer on 2026-10-07.

| Question | Decision |
| --- | --- |
| Main purpose | Initial fill of every table for a machine; incremental retune and exploration are secondary |
| Speed versus rigor | The user picks a tier per run: **ultra** (under 30 min), **coarse** (2-4 h), **deep** (8-12 h), each a budget for one GPU. More GPUs shard the work. Every tier avoids measurements that cannot change a table. |
| A cheaper run must not degrade a better one | **Higher fidelity wins per cell.** A table row comes from the highest-tier result whose kernel hashes are still current. A lower tier fills gaps and replaces stale results, never current ones. |
| Op scope | All 19 routed ops, plus the `tuning_params.hh` constants under the same tiers, ledger and UI. Their header is generated, no longer hand-ported. |
| Result store | In the repository, under Git LFS (`benchmarks/results/tuning/`), so precedence holds across machines and clones |
| Per-rep raw timings | Kept for deep runs only |
| UI | A Tuning tab in `benchviz serve`, with `batchlas_tune` as its measuring engine. Views: status matrix, launch with live progress, winner map per op, and a diff against the shipped table. |
| Persistent workers versus the sticky carve-out | A persistent worker that processes cells in ascending size order, plus a fresh-process audit. A mismatch drops that op and dtype back to one process per cell. |

## Tiered tuning: sub-projects

| # | Sub-project | Depends on | Spec |
| --- | --- | --- | --- |
| 1 | **Engine**: tiers, racing, adaptive grid, persistent worker, ledger, table generation; shown on the 4 existing specs | none | this page, in full |
| 2 | **Coverage**: specs for the other 15 ops; the constants tuner moved onto the engine and `tuning_params.hh` generated | 1 (spec interface, ledger) | own design pass; interface below |
| 3 | **benchviz Tuning tab**: the four views | 1 (ledger format, driver progress protocol) | own design pass; interface below |

Sub-projects 2 and 3 can proceed in parallel once the engine's spec interface, ledger schema and
progress protocol are frozen.

## Engine: tiers and the per-cell algorithm

A tier is a named set of parameters. The algorithm is the same in every tier, so ultra results are
valid low-fidelity data, not a different kind of number.

| Parameter | ultra | coarse | deep |
| --- | --- | --- | --- |
| Starting lattice | every 4th point of each `choice.hh` axis (ends kept) | every 2nd point | the full lattice |
| Bisection between disagreeing neighbours | off | until hi/lo < 1.25 | until hi/lo < 1.1 (today's rule) |
| Race: min / max reps per candidate | 3 / 6 | 4 / 12 | 6 / 16 |
| Elimination confidence | loose | medium | today's level, plus a confirmation round in reversed order |
| Fresh-process audit sample | 2% of cells | 2% | 10% |

The 3% tie margin is the same in every tier, because it is the ranking rule the converter and
`rank()` share. The lattices nest (ultra ⊂ coarse ⊂ deep), so results from different tiers land on
the same cells. An op whose grid is not a lattice (gemm's demand-driven shapes, `grid()` override)
is subsampled in the same proportions by a stable hash of the cell key, so the subsets also nest.

Each cell goes through three steps, in every tier:

1. **Probe.** Each candidate's pin is checked with the existing probe (`*_buffer_size` under
   `select::ScopedPin`, or one untimed run for trsm). With 0 or 1 runnable candidates the cell is
   recorded and nothing is timed: one candidate cannot lose.
2. **Skip.** If the ledger already holds a current result at the same or a higher tier, the cell is
   skipped. If the result is *partly stale* (some candidate hashes changed), only the changed
   candidates are raced, against the stored winner and runner-up, at the stored record's tier (see
   [the ledger](#engine-the-ledger-and-table-generation)).
3. **Race.** The surviving candidates are timed in rounds. Each round runs every survivor once,
   interleaved, with the order rotated per round, and the inputs restored untimed. After each round
   a candidate is eliminated when its slowdown against the current leader is confidently above the
   tie margin. The race stops when one candidate remains, when the survivors are confidently within
   the tie margin of each other, or at the tier's maximum reps. The race starts with the winner of
   the nearest finished neighbour cell, which tightens elimination early. Before its times count,
   every candidate is verified on items 0 and batch-1, exactly as today (`residuals.hh`).

The elimination statistic (for example a ratio-of-medians bound from paired rounds) is chosen in
the implementation plan. It is validated offline first; see
[testing](#engine-testing-and-acceptance).

Across cells:

- **Breadth-first.** For every op and dtype in a run, the starting lattice is measured before any
  refinement. A run interrupted at any point leaves complete, usable tables.
- **Bisection on every key axis,** not only `n`: batch, q and nrhs too, using each key's own
  distance: `log` axes are bisected geometrically, `exact` axes (uplo, side, trans) never. Nothing is inferred or interpolated: a
  table simply has fewer rows where neighbours agree, and `select::choose` already takes the
  nearest row.
- **Time estimate and budget.** Before starting, the driver predicts the run's cost from earlier
  ledger timings for the same op, dtype and device, or from a flop/byte model of the op against the
  device's ceilings when there are none. The estimate is printed (and shown in the UI). `--budget
  <h>` stops refinement when the budget is spent; the starting lattice always completes.
- **Op order follows dependencies.** potrf and trsm are tuned before posv, whose cta and blocked
  times depend on their choices (@ref tune_tool_readme, section Specs). Other compositions found in
  sub-project 2 are added to the same ordering.

## Engine: the ledger and table generation

**Ledger.** There is one directory per (op, dtype, device),
`benchmarks/results/tuning/ledger/<op>.<dtype>.<device>/`, with one file per run, `<run id>.jsonl`
(Git LFS). A run only ever writes its own file, and readers take the union of the files. An
append-only file in LFS cannot be merged, so two boxes tuning the same op would always conflict;
one file per run cannot.

| Record | Fields |
| --- | --- |
| `run` | run id, tier, host, device key and name, batchlas git sha (`-dirty`), argv, guard settings, tolerated foreign pids, worker mode per op/dtype (persistent or fresh-process), date |
| `cell` | run id, tier, the key fields, `round` (0 = starting lattice), for each candidate: family hash, status (`ok`/`skipped`/`bad`/`error`), reason, median ms, interval, reps; the ranked tie set |
| `audit` | run id, the cell key, persistent versus fresh-process medians and winners, verdict |

Per-rep timings go to `<run id>.reps.jsonl` in the same directory, written by deep runs only. A
truncated last line (a killed run) is ignored with a warning, so a cell is either fully recorded or
absent.

**Per-candidate hashes.** A spec's `kernel-sources` block becomes `family -> files`, plus a common
set (dispatch, `src/select/`) that every family depends on. A family's hash covers the common set
plus its own files. `cmake/BatchLASTunedStaleness.cmake`, `.github/ci/check_tuned_tables.py` and
the driver compute it the same way. The posv coupling becomes explicit: posv's families list the
potrf and trsm families they call.

**Which record counts.** Precedence is deep > coarse > ultra > transcribed.

- A `cell` record is *current* when every candidate's hash matches the source tree.
- It is *partly stale* when only some changed, or when the candidate list gained a family the
  record never raced. The next run of any tier re-races just those
  candidates against the stored winner and runner-up, at the stored record's tier, and appends a
  merged record. Editing one kernel therefore keeps every deep cell deep, and costs a fraction of a
  retune.
- It is *stale* when the stored winner's hash changed or the winner's family was removed. The cell is re-raced at the record's tier.

**Table generation.** `scripts/sweep_to_table.py --ledger <file>` replaces `--tuner`. Its output
stays byte-reproducible, and `--check` keeps re-deriving every table.

- For each cell it takes the best current record.
- A lower-tier row is emitted only to fill a gap: when no higher-tier row lies within the lower
  tier's lattice spacing under `select::choose`'s own distance. A coarse point can therefore never
  sit inside a region a deep run already resolved.
- Transcribed rows survive only where nothing measured covers them.
- The header reports the tier mix and the hashes:
  `source=ledger:<dir> tiers=deep:812,coarse:120,ultra:0,transcribed:0 family_kernels=<family>:<hash>,...`.
- Each row ends in a tier comment (`# deep`). Both table parsers (`src/select/select.cc` and
  `sweep_to_table.py`) already strip a trailing `#` comment from a row, so neither changes.
  Provenance is then readable from `tuned/` without the UI.
- The op-level `kernels=<hash>` header stays as it is, for the configure-time staleness check. The
  per-family hashes go in a new `family_kernels=` header word.

Results merge by device key (sm_89, sm_120), and the host is recorded. A deep sm_89 run on one 4090
box counts for every 4090.

The existing raw files (`trsm.{float,double}.sm_120.jsonl`, schema 1) are imported into ledgers as
deep records, so measured tables are not lost in the migration.

## Engine: persistent workers and the carve-out audit

**Worker.** Each GPU gets one long-lived worker (`batchlas_tune_impl --worker`), started by the
driver under the existing launcher, `CUDA_VISIBLE_DEVICES` fence and per-GPU flock. Cells arrive on
a pipe, and results come back as `cell` records.

- The worker warms the clocks for 3 s at start; each cell then gets a short top-up of about 0.2 s
  per candidate.
- Each kernel is JIT-compiled once per worker (the on-disk JIT cache stays shared).
- The idle guard runs between cells instead of around each child.
- If the worker dies, it restarts and the cell is retried. A candidate that crashes when run alone
  is recorded as `error`, as today.

**The sticky carve-out.** The SLM carve-out attribute is sticky per CUfunction
(`benchmarks/factor_bench.cc` header). In one process, an earlier, larger launch can make a later
launch succeed that would fail in a fresh process. That (a) hides a feasibility bug and (b) may
shift the L1/SLM split, which changes timing. Two defences:

1. **Ascending order.** Within an op and dtype, a worker processes its cells in ascending order of
   input size. The first launch of a kernel that needs more than 48 KB therefore happens exactly as
   in a fresh process: no larger launch has run before it.
2. **Fresh-process audit.** Every run re-measures a sample of its cells in fresh one-cell processes
   (today's `--cell` path; the tier table gives the fraction). It compares feasibility per candidate
   and the winner. A mismatch marks that op and dtype *fresh-process only* in the ledger, and its
   remaining cells run one process per cell. The verdict is shown in the status matrix.

The first task of the implementation plan measures the real per-child overhead (process start,
SYCL and CUDA init, libbatchlas static init, JIT cache load) on both boxes. If it is small next to a
raced cell, the worker is dropped and the engine keeps one process per cell, which removes the
carve-out question.

## Engine: driver interface

- `batchlas_tune <op>[,<op>...|all] --tier ultra|coarse|deep --dtype ... --devices ...` replaces the
  protocol flags (`--reps`, `--warm`, `--passes`, `--remeasure`, `--refine-ratio`). Those stay as
  expert overrides, and a run that uses them is recorded with tier `custom`, ranked below ultra.
- `--plan` prints the cells, the skipped cells with their reasons, and the time estimate, then exits
  without measuring. benchviz calls it for the estimate before launching.
- `--budget <h>` caps refinement time.
- `--progress-fd <n>` writes one JSON line per event (cell started or finished, candidate
  eliminated, audit verdict, worker restart). This is the protocol benchviz consumes over SSE.
- `--status` prints the op × dtype × device matrix (tier mix, coverage, stale cells, audit verdicts,
  age) from the ledgers and tables, without a GPU.
- `--gate` is unchanged.

## Coverage: interface for sub-project 2

- Each of the 15 remaining ops gets an `<op>_spec.cc` implementing `OpSpec` (`tools/tune/spec.hh`):
  keys, candidates and axes from its `choice.hh`, a problem builder, host verification, a size cap,
  and per-family kernel sources. Ops whose public entry point composes other ops (gesv, getri, syev,
  gesvd, posv) declare those dependencies for the op order and the hashes.
- The constants tuner becomes a second kind of spec. Its "candidates" are values of one
  `tuning_params.hh` knob, its cells are the knob's n-buckets and dtypes (not only float), and it
  measures through the existing `BATCHLAS_TUNE_*` overrides. Its ledger yields a generated header,
  replacing the hand port. A knob that feeds two consumers (aliasing) is tuned on the consumer the
  maintainer names, and the other consumer is reported, not silently retuned.
- `evaluation/tuning/` is retired once its knobs are covered, with its `sensitivity.py` axis pruning
  carried over as the knob's starting lattice.

## benchviz Tuning tab: interface for sub-project 3

The tab reads the ledgers and tables directly, and drives the engine through
`batchlas_tune --plan`, `--progress-fd` and `--status`. It keeps benchviz's server, SSE stream,
`gpu_guard.sh`, campaign store and figure style (`benchmarks/benchviz/style.py`).

- **Status matrix.** op × dtype × device: tier mix, coverage, stale cells, audit verdict, age.
- **Launch and live progress.** Choose ops, dtypes, devices and tier; see the estimate before
  starting; watch cells finish and candidates get eliminated; stop and resume. Resuming is the skip
  rule applied to a partial run.
- **Winner map per op.** A heatmap over two chosen keys (for example n × batch), with the others
  fixed. Colour shows the winning family, intensity shows the margin to the runner-up, and a marker
  shows the tier. Clicking a cell shows every candidate's timing and interval.
- **Diff against the shipped table.** Cells whose winner changed, with the predicted per-cell
  speedup or slowdown against the table in `tuned/`, before the tables are written.

## Engine: testing and acceptance

The engine changes which kernel ships for every shape, so its tests follow the agent guide's
"guards that cannot fail" rules (@ref dev_agent_guide §8).

- **Offline replay before any GPU time.** The deep trsm raw files hold every rep of every candidate
  at 4452 cells. A replay harness runs the racing and bisection logic for each tier against those
  reps, with no GPU. It reports reps saved, cells saved, and the misranking rate: the fraction of
  cells where the chosen candidate is more than 3% slower than the exhaustive best. The elimination
  statistic and the tier numbers above are fixed only after this replay. Acceptance: deep misranks
  at most 0.2% of cells, coarse misranks at most 1% of cells, ultra at most 5%, all by more than 3%.
- **Unit tests without a GPU.** Precedence (a coarse record never displaces a current deep one; a
  stale deep one is displaced), the gap-fill rule (a coarse row inside a deep bracket is never
  emitted), partly-stale re-race selection, lattice nesting, and `--check` reproducibility. Each has
  a deliberate break that turns exactly its own test red.
- **Racing guards.** A synthetic spec with fixed timing distributions, including a true near-tie, a
  candidate 4% slower than the leader (must be eliminated in deep), and one 2% slower (must be kept
  as a tie), straddles the tie margin in both directions.
- **Audit guard.** A test pins a candidate whose feasibility differs between fresh and warm
  processes (an SLM-heavy launch ordered after a larger one) and checks that the audit flags it.
- **End-to-end.** An ultra run of potrf on one GPU finishes under its budget, and the tables it
  writes pass `tuned_tables_tests` and `--check`. A coarse run of trsm float on sm_120 agrees with
  the existing deep table within the replay's misranking bound.

## Engine: replay results on the trsm data

`tune_replay` (`tools/tune/replay.cc`, host-only) replays the race and the bisection of a tier against
the exhaustive trsm sweeps: 4452 float cells and 4098 double cells, each with every rep of every
candidate that was `ok` in both passes of the final attempt (threadripper02, sm_120, data from
2026-10-05). The two passes are interleaved (p1r0, p2r0, p1r1, ...). The starting lattice is the
tier's, cut down to cells the raw file holds; bisection proposes midpoints on every log axis, and a
midpoint the raw file lacks is counted (`refine_unavailable`) and dropped. `race_misrank` is over the
cells the replay measured, `table_misrank` over all cells: the nearest measured cell is picked by a
port of `nearest()` in `scripts/sweep_to_table.py`, and its pick counts as a misrank when it is more
than 3% slower than the exhaustive best at the cell. Like `select::choose`, the pick is the first entry of that row's ranking (survivors, then the eliminated) that can run at the cell; a cell where no entry can run counts as a misrank and as `unrunnable`. `reps_fraction` is the
candidate-reps the replay timed over those in the file. The tier values are the ones in the tier table
above, unchanged.

| Tier | dtype | cells measured / cells | reps_fraction | race_misrank | table_misrank | Bound |
| --- | --- | --- | --- | --- | --- | --- |
| deep | float | 4417 / 4452 | 0.2375 | 0.113% | 0.157% | 0.2%: met |
| deep | double | 4095 / 4098 | 0.2248 | 0.024% | 0.024% | 0.2%: met |
| coarse | float | 810 / 4452 | 0.0279 | 0.123% | 8.33% | 1%: race met, table missed |
| coarse | double | 711 / 4098 | 0.0272 | 0% | 9.83% | 1%: race met, table missed |
| ultra | float | 132 / 4452 | 0.0031 | 0% | 19.50% | 5%: race met, table missed |
| ultra | double | 124 / 4098 | 0.0030 | 0% | 14.20% | 5%: race met, table missed |

Racing is not the problem: every tier meets its race bound, deep saves 76-78% of the reps, and
changing confidence or reps moves `table_misrank` by under one point. The table bound is set by the
lattice and the bisection ratio. The trade-off is easier to read with the loss of the chosen candidate,
`exhaustive(chosen) / best - 1`, over all cells, where chosen is the `select::choose` pick: `mean_loss`
and the percentiles cover the cells where some entry of the nearest row can run, `unrunnable` counts
the rest, `time_weighted_loss` is the summed chosen time over the summed best time minus one, and
`table_misrank_lattice` repeats `table_misrank` over the round-0 cells only, which removes the raw
file's refinement points at winner flips. `est_gpu_h` is `reps_fraction` times the exhaustive run
(float 16.0 GPU-h, double 31.5 GPU-h, from `benchmarks/results/tuning/README.md`), without per-cell
process overhead. Each row uses the named tier's race parameters; the config column changes only the
lattice stride and the bisection ratio.

| config | dtype | cells_measured | reps_fraction | est_gpu_h | table_misrank % | table_misrank_lattice % | mean_loss | p99_loss | max_loss | time_weighted_loss | unrunnable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ultra default | float | 132 | 0.0031 | 0.05 | 19.50 | 17.76 | 0.0819 | 1.283 | 2.40 | 0.0592 | 0 |
| ultra default | double | 124 | 0.0030 | 0.10 | 14.20 | 11.06 | 0.0581 | 0.999 | 2.84 | 0.0190 | 0 |
| ultra stride 2 | float | 616 | 0.0150 | 0.24 | 11.48 | 8.91 | 0.0297 | 0.614 | 1.61 | 0.0205 | 0 |
| ultra stride 2 | double | 585 | 0.0150 | 0.47 | 10.49 | 5.62 | 0.0367 | 0.770 | 1.92 | 0.0110 | 0 |
| ultra stride 1, no refine | float | 3480 | 0.0867 | 1.39 | 6.36 | 0.17 | 0.0110 | 0.239 | 0.77 | 0.0031 | 0 |
| ultra stride 1, no refine | double | 3219 | 0.0853 | 2.69 | 6.05 | 0.00 | 0.0209 | 0.601 | 2.08 | 0.0080 | 0 |
| ultra stride 1, refine 1.25 | float | 4021 | 0.1005 | 1.61 | 2.52 | 0.17 | 0.0035 | 0.115 | 0.77 | 0.0025 | 0 |
| ultra stride 1, refine 1.25 | double | 3725 | 0.0991 | 3.12 | 1.42 | 0.00 | 0.0037 | 0.052 | 2.08 | 0.0027 | 0 |
| ultra stride 1, refine 1.1 | float | 4390 | 0.1074 | 1.72 | 0.29 | 0.17 | 0.0006 | 0.021 | 0.08 | 0.0004 | 0 |
| ultra stride 1, refine 1.1 | double | 4088 | 0.1062 | 3.35 | 0.02 | 0.00 | 0.0004 | 0.014 | 0.08 | 0.0002 | 0 |
| ultra stride 2, refine 1.25 | float | 820 | 0.0193 | 0.31 | 7.88 | 5.86 | 0.0192 | 0.542 | 1.86 | 0.0104 | 0 |
| ultra stride 2, refine 1.25 | double | 712 | 0.0189 | 0.60 | 9.66 | 4.32 | 0.0373 | 0.790 | 1.95 | 0.0083 | 0 |
| coarse default | float | 810 | 0.0279 | 0.45 | 8.33 | 6.38 | 0.0210 | 0.599 | 1.86 | 0.0119 | 0 |
| coarse default | double | 711 | 0.0272 | 0.86 | 9.83 | 4.38 | 0.0375 | 0.790 | 1.95 | 0.0083 | 0 |
| coarse stride 1, refine 1.25 | float | 4025 | 0.1476 | 2.36 | 2.54 | 0.06 | 0.0036 | 0.118 | 0.77 | 0.0024 | 0 |
| coarse stride 1, refine 1.25 | double | 3727 | 0.1416 | 4.46 | 1.49 | 0.00 | 0.0038 | 0.055 | 2.08 | 0.0027 | 0 |
| coarse stride 2, refine 1.1 | float | 813 | 0.0280 | 0.45 | 8.31 | 6.38 | 0.0210 | 0.599 | 1.86 | 0.0115 | 0 |
| coarse stride 2, refine 1.1 | double | 715 | 0.0273 | 0.86 | 9.83 | 4.38 | 0.0375 | 0.790 | 1.95 | 0.0083 | 0 |
| coarse stride 1, refine 1.1 | float | 4413 | 0.1583 | 2.53 | 0.11 | 0.06 | 0.0006 | 0.020 | 0.03 | 0.0003 | 0 |
| coarse stride 1, refine 1.1 | double | 4094 | 0.1518 | 4.78 | 0.05 | 0.00 | 0.0004 | 0.015 | 0.08 | 0.0002 | 0 |
| deep default | float | 4417 | 0.2375 | 3.80 | 0.16 | 0.11 | 0.0006 | 0.022 | 0.06 | 0.0004 | 0 |
| deep default | double | 4095 | 0.2248 | 7.08 | 0.02 | 0.00 | 0.0004 | 0.015 | 0.03 | 0.0001 | 0 |

Reading it: with the fallback, `unrunnable` is 0 in every row, so the earlier unrunnable choices were
cells where the first entry could not run and the next one could. The full lattice is what matters:
at stride 1 with the ultra race parameters, bisecting to 1.1 gives 0.29% and 0.02% for 1.72 and 3.35
GPU-h, against 0.11% and 0.05% for 2.53 and 4.78 GPU-h with the coarse race parameters; bisecting to
1.25 gives 2.5% and 1.4%, and no bisection 6.4% and 6.1%. The sparse rows (stride 2 and 4) stay at 8 to
20% whatever the ratio, because the replay drops the `q` and `batch` midpoints the raw file lacks; it
cannot say what bisection on those axes would buy. The mean and time-weighted losses are small in every
row (at most 8.2% and 5.9%, both ultra default float), but the p99 loss of the sparse rows is 0.5 to
1.3, so they are right on average and badly wrong at a few percent of shapes.

## Engine: replay of shrunken trsm grids

Instead of sparse tiers, the grids themselves can shrink. `tune_replay --axis-stride name=k` keeps every
k-th value and the last of one axis in the starting lattice, and `--axis-keep name=v1:v2` keeps only
the listed values; the other axes stay full. The raw files hold the same axes for both dtypes
(`--print-axes`): order 1 2 4 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024 (log, weight 2), q 1 2 4
8 16 32 64 128 256 512 1024 4096 (log), batch 128 512 2048 8192 32768 (log), plus side L R and trans N T.
Every row uses the ultra race parameters, stride 1 and bisection to 1.1. Bisection refills only where
the raw file has points, which is `order` (the raw sweep refined only along it), so shrinking `order`
is partly repaired while shrinking `q` or `batch` is not. The metrics are over all raw cells, so a cell
at a dropped value pays for the dropped measurement. Same machine and data as above; est_gpu_h as above.

| config | dtype | measured | est_gpu_h | table_misrank % | lattice % | mean_loss | p99_loss | max_loss | tw_loss |
|---|---|---|---|---|---|---|---|---|---|
| baseline (ultra s1 r1.1) | float | 4390 | 1.72 | 0.29 | 0.17 | 0.0006 | 0.021 | 0.08 | 0.0004 |
| baseline (ultra s1 r1.1) | double | 4088 | 3.35 | 0.02 | 0.00 | 0.0004 | 0.014 | 0.08 | 0.0002 |
| batch stride 2 | float | 2920 | 1.14 | 4.27 | 3.39 | 0.0162 | 0.615 | 1.86 | 0.0080 |
| batch stride 2 | double | 2546 | 2.10 | 4.76 | 3.76 | 0.0250 | 0.757 | 2.36 | 0.0069 |
| batch keep 8192:32768 | float | 1356 | 0.54 | 11.32 | 10.11 | 0.0263 | 0.540 | 0.93 | 0.0188 |
| batch keep 8192:32768 | double | 1226 | 0.98 | 6.83 | 5.03 | 0.0230 | 0.799 | 1.27 | 0.0041 |
| batch keep 128:32768 | float | 2118 | 0.84 | 7.86 | 7.30 | 0.0299 | 0.864 | 1.86 | 0.0193 |
| batch keep 128:32768 | double | 1789 | 1.50 | 6.86 | 5.28 | 0.0329 | 0.837 | 2.36 | 0.0062 |
| batch keep 2048 | float | 916 | 0.36 | 13.52 | 11.24 | 0.0314 | 0.549 | 0.93 | 0.0200 |
| batch keep 2048 | double | 814 | 0.67 | 7.22 | 4.16 | 0.0192 | 0.421 | 1.73 | 0.0036 |
| q stride 2 | float | 2964 | 1.14 | 1.50 | 0.95 | 0.0039 | 0.063 | 1.46 | 0.0007 |
| q stride 2 | double | 2435 | 2.02 | 2.98 | 1.03 | 0.0074 | 0.279 | 0.78 | 0.0021 |
| q stride 3 | float | 1728 | 0.68 | 6.42 | 5.89 | 0.0122 | 0.325 | 0.91 | 0.0103 |
| q stride 3 | double | 1696 | 1.39 | 6.12 | 5.65 | 0.0207 | 0.553 | 1.04 | 0.0137 |
| order stride 2 | float | 1911 | 0.74 | 4.65 | 2.41 | 0.0066 | 0.142 | 1.06 | 0.0026 |
| order stride 2 | double | 1837 | 1.51 | 6.93 | 2.05 | 0.0244 | 0.755 | 0.91 | 0.0045 |
| order stride 4 | float | 1348 | 0.53 | 6.45 | 3.85 | 0.0128 | 0.422 | 1.06 | 0.0040 |
| order stride 4 | double | 1269 | 1.06 | 6.34 | 4.57 | 0.0176 | 0.430 | 4.00 | 0.0045 |
| q2 + batch2 | float | 1978 | 0.75 | 6.04 | 5.09 | 0.0212 | 0.704 | 1.86 | 0.0086 |
| q2 + batch2 | double | 1530 | 1.28 | 5.66 | 3.42 | 0.0203 | 0.639 | 2.19 | 0.0050 |
| q2 + batch2 + order2 | float | 824 | 0.31 | 7.84 | 5.86 | 0.0191 | 0.542 | 1.86 | 0.0100 |
| q2 + batch2 + order2 | double | 716 | 0.60 | 9.66 | 4.32 | 0.0373 | 0.790 | 1.95 | 0.0083 |
| q2 + order2 | float | 1254 | 0.48 | 5.05 | 2.87 | 0.0085 | 0.201 | 1.06 | 0.0028 |
| q2 + order2 | double | 1120 | 0.93 | 8.17 | 2.67 | 0.0279 | 0.748 | 0.91 | 0.0051 |
| q3 + batch2 + order2 | float | 491 | 0.19 | 10.53 | 9.68 | 0.0277 | 0.561 | 2.17 | 0.0207 |
| q3 + batch2 + order2 | double | 447 | 0.37 | 12.71 | 8.45 | 0.0571 | 0.859 | 5.12 | 0.0164 |
| q2 + batch keep 2048 + order2 | float | 243 | 0.09 | 16.85 | 12.90 | 0.0399 | 0.594 | 1.13 | 0.0247 |
| q2 + batch keep 2048 + order2 | double | 212 | 0.18 | 11.05 | 5.25 | 0.0382 | 0.797 | 1.73 | 0.0066 |

Reading it: batch is the costliest axis to shrink, because the winner flips along it (stride 2 already
gives 4.3% and 4.8%; one batch value gives 13.5% and 7.2%). q stride 2 is the cheapest single cut
(1.5% float, 3.0% double, at 66% and 60% of the baseline GPU-hours), and order stride 2 gives 4.7% and
6.9%. Combinations add their misranks rather than their savings: q stride 2 with batch stride 2 and
order stride 2 costs 0.31 and 0.60 GPU-h and misranks 7.8% and 9.7%, the same as the sparse coarse
tier. No shrunken grid in this sweep meets the 1% table bound; the baseline does, at 1.72 and 3.35
GPU-h.

## Tiered tuning: open risks

- Racing assumes timing noise is roughly stationary within a cell. Clock ramps after idle gaps
  break that. The worker's warm start and between-cell top-up are the defence, and the replay
  cannot test it, because the raw files come from warm, back-to-back runs.
- If the per-child overhead turns out large and the audit fails often, the speedup has to come
  mostly from racing and the adaptive grid. The replay quantifies how much they provide.
- Budgets are for one GPU and assume a quiet box. On the shared 4090 box, the guard's waits come on
  top.
