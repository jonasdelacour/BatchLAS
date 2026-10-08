# Tiered tuning {#design_tiered_tuning}

> **Status:** current · sub-project 1 (engine) implemented, validated on threadripper02 (RTX PRO 6000,
> sm_120) 2026-10-07 · sub-projects 2 and 3 specified at interface level only

The tuner fills the selection tables and the tuning constants in tiers. The user picks a tier
(preview, coarse or deep) per run. Measurements that cannot change a table are skipped, and a
per-cell ledger ensures that a cheaper run never overwrites a better result. The current tuner is
described in @ref tune_tool_readme; the flat selection it feeds is in @ref design_flat_selection §6.
No shipped table comes from the engine yet.

## Why tiered tuning

- `batchlas_tune` times every runnable candidate at every cell of a full lattice. Each cell costs
  at least three cold processes, each with 1.5 s of warm-up per candidate.
- trsm float on sm_120 took about 16 GPU-hours (3480 cells, 5 h 20 min on 3 GPUs;
  `benchmarks/results/tuning/README.md`). gemm float is estimated at about 100 GPU-hours and has
  never been run.
- Only 4 of the 19 routed ops have a tuner spec. Of the 144 shipped tables, 16 are measured and 128
  are `source=transcribed`.
- The `tuning_params.hh` constants come from `evaluation/tuning/`, which ran on float only. Its
  results were ported into the header by hand.

## Decisions

| Question | Decision |
| --- | --- |
| Main purpose | Initial fill of every table for a machine. Incremental retune and exploration come second. |
| Speed versus rigor | The user picks a tier per run. Budgets come from a cost model printed by `--plan`. More GPUs shard the work. |
| Grid reduction | Only preview is sparse. Coarse and deep use the full lattice, because sparse grids misrank 4-13% of cells. |
| Precedence | Higher fidelity wins per cell. A lower tier fills gaps and replaces stale results, never current ones. |
| Op scope | All 19 routed ops, plus the `tuning_params.hh` constants, under the same tiers, ledger and UI. The header is generated. |
| Result store | `benchmarks/results/tuning/` under Git LFS, so precedence holds across machines and clones. |
| Raw per-rep timings | Kept for deep runs only (deferred, see [open risks](#open-risks)). |
| UI | A Tuning tab in `benchviz serve`, driving `batchlas_tune`. |
| Worker | One persistent worker per GPU, with a fresh-process audit. See [persistent workers](#engine-persistent-workers-and-the-carve-out-audit). |

## Sub-projects

| # | Sub-project | Depends on | State |
| --- | --- | --- | --- |
| 1 | Engine: tiers, racing, adaptive grid, persistent worker, ledger, table generation, on the 4 existing specs | none | implemented |
| 2 | Coverage: specs for the other 15 ops; constants tuner on the engine; `tuning_params.hh` generated | 1 | interface below |
| 3 | benchviz Tuning tab: four views | 1 | interface below |

Sub-projects 2 and 3 can proceed in parallel once the spec interface, ledger schema and progress
protocol are frozen.

## Engine: tiers and the per-cell algorithm

Every tier runs the same algorithm, so preview results are low-fidelity data of the same kind as
deep results.

| Parameter | preview | coarse | deep |
| --- | --- | --- | --- |
| Starting lattice | every 2nd point of each `choice.hh` axis, ends kept | full lattice | full lattice |
| Bisection | index space until adjacent, then geometric to hi/lo < 1.1; also between cells when a runner-up is within 10% | geometric to hi/lo < 1.1 | geometric to hi/lo < 1.1 |
| Refinement cap (x round-0 cells, per op and dtype) | 3.0 | 1.0 | 2.0 |
| Race reps, min / max per candidate | 3 / 6 | 4 / 12 | 6 / 16 |
| Elimination confidence | 0.80 | 0.90 | 0.98, plus a reversed confirmation round |
| Fresh-process audit sample | 2% of cells | 2% | 10% |

The tie margin is 3% in every tier. The lattices nest (preview ⊂ coarse ⊂ deep). An op whose grid
is not a lattice is subsampled by a stable hash of the cell key, so subsets still nest.

Each cell goes through three steps:

1. **Probe.** Each candidate's pin is checked with the `*_buffer_size` probe under
   `select::ScopedPin`, or one untimed run for trsm. With 0 or 1 runnable candidates the cell is
   recorded and nothing is timed.
2. **Skip.** A current ledger result at the same or higher tier skips the cell. A partly stale
   result re-races only the changed candidates, against the stored winner and runner-up.
3. **Race.** Survivors are timed in interleaved rounds, with the order rotated per round and inputs
   restored untimed. A candidate is eliminated when it is confidently slower than the leader by more
   than the tie margin. The race stops when one candidate remains, when the survivors are confidently
   within the margin, or at the max reps. The race starts from the winner of the nearest finished
   neighbour. Before its times count, each candidate is verified on items 0 and batch-1
   (`residuals.hh`).

Across cells:

- **Breadth first.** The starting lattice of every op and dtype is measured before any refinement,
  so an interrupted run leaves usable tables.
- **Bisection on every key axis** (n, batch, q, nrhs), using each key's own distance. `log` axes
  bisect geometrically, `exact` axes (uplo, side, trans) never, and `batch` refills only its own
  axis values. Nothing is interpolated. `select::choose` takes the nearest row.
- **Estimate and budget.** The driver predicts cost from earlier ledger timings for the same op,
  dtype and device, or from a flop/byte model against the device ceilings. `--budget <h>` stops
  refinement; the starting lattice always completes.
- **Op order follows dependencies.** potrf and trsm are tuned before posv, whose times depend on
  their choices (@ref tune_tool_readme, section Specs).

## Engine: the ledger and table generation

**Ledger.** One directory per (op, dtype, device): `benchmarks/results/tuning/ledger/<op>.<dtype>.<device>/`,
with one `<run id>.jsonl` per run (Git LFS). A run writes only its own file and readers take the
union. Appending to one shared LFS file would conflict between boxes.

| Record | Fields |
| --- | --- |
| `run` | run id, tier, host, device key and name, batchlas git sha (`-dirty`), argv, worker mode per op and dtype, date |
| `cell` | run id, tier, key fields, `round` (0 = starting lattice); per candidate: family hash, status (`ok`, `skipped`, `bad`, `error`, `eliminated`), reason, median ms, interval, reps; the ranked tie set |
| `audit` | run id, cell key, persistent and fresh-process medians and winners, verdict |

A truncated last line (a killed run) is ignored with a warning, so a cell is either fully recorded
or absent. Results merge by device key (sm_89, sm_120); the host is recorded.

**Hashes.** A spec's `kernel-sources` block maps each family to its files, plus a common set
(dispatch, `src/select/`) that every family depends on. Each spec lists its `src/ops/<op>/<op>.cc`,
where `can_run` lives, so an edit to what can run stales every cell. `kernel-deps` adds files
outside the source list (`// family: <name> "path" ...` or `// common "path" ...`). The posv
coupling is explicit: posv's families list the potrf and trsm families they call.

**Which record counts.** Precedence is deep > coarse > preview > custom > transcribed. A `cell`
record is:

- **Current** when every candidate's hash matches the source tree. A candidate stored as `skipped`
  is not compared, because a change to what can run goes through the hashed `<op>.cc`.
- **Partly stale** when only some hashes changed, or a family appeared that the record never raced.
  The next run re-races just those candidates against the stored winner and runner-up, at the stored
  tier, and appends a merged record. Editing one kernel keeps every deep cell deep.
- **Stale** when the stored winner's hash changed or its family was removed. The cell is re-measured
  at the running tier.

A record in which every candidate is `error` counts as no record. `best_records` (in `ledger.cc` and
`sweep_to_table.py`) applies this one rule, so planning, refinement, `--status` and table generation
agree.

**Table generation.** `scripts/sweep_to_table.py --ledger <file>` replaces `--tuner`. Output is
byte-reproducible, and `--check` re-derives every table. It refuses to overwrite a timed table that
is not from a ledger unless `--replace-timed` is given.

- Each cell takes its best current record.
- A lower-tier row is emitted only to fill a gap: no higher-tier row may lie within the lower tier's
  lattice spacing under `select::choose`'s distance. A coarse point never sits inside a region a deep
  run resolved.
- Transcribed rows survive only where nothing measured covers them.
- The header reports the tier mix and per-family hashes:
  `source=ledger:<dir> tiers=deep:812,coarse:120,preview:0,transcribed:0 family_kernels=<family>:<hash>,...`.
  The op-level `kernels=` hash is unchanged.
- Each row ends in a tier comment (`# deep`), which both parsers already strip.

**Import.** `batchlas_tune --import-raw` turns `trsm.{float,double}.sm_120.jsonl` (schema 1) into
deep runs. Their hashes are `legacy:c923160f`, so they are stale until a diagnostic
`--assume-current` judges them by stored hashes.

## Engine: persistent workers and the carve-out audit

**Worker.** Each GPU gets one long-lived `batchlas_tune_impl --worker`, started by the driver under
the existing `CUDA_VISIBLE_DEVICES` fence and per-GPU flock. Cells arrive on a pipe and come back as
`cell` records.

- The worker warms the clocks for 3 s at start, then tops up each cell by about 0.2 s per candidate.
- Each kernel is JIT-compiled once per worker. The on-disk JIT cache stays shared.
- The idle guard runs between cells. If the worker dies, it restarts and the cell is retried. A
  candidate that crashes alone is recorded as `error`.
- A worker `error` with a sticky CUDA error restarts the worker and re-races the cell in a fresh
  child. If the fresh child reproduces it in two consecutive cells, that op and dtype race the
  candidate alone, one process per cell.

**The sticky carve-out.** The SLM carve-out attribute is sticky per CUfunction
(`benchmarks/factor_bench.cc` header), so an earlier larger launch can make a later one succeed that
fails in a fresh process. Two defences:

1. **Ascending order.** A worker runs cells in ascending per-item footprint (input bytes / batch,
   then total bytes). The driver restarts the worker before any cell smaller than one it already ran.
2. **Fresh-process audit.** A sample of cells is re-measured in one-cell fresh processes. The sample
   is chosen by `fnv1a64(run_id + key) % 1000 < audit_fraction * 1000`, and each op and dtype gets one
   audit before its first unaudited worker cell. A mismatch is a candidate usable in one process and
   refused or failing in the other, or a winner more than 10% slower in the fresh run. A mismatch
   marks that op and dtype fresh-process only. Each audited cell gets an `audit` record.

**Result.** No carve-out mismatch was found (threadripper02, GPU 1, 2026-10-07, `tune_race_gpu_tests`).
For potrf float and double, posv float and trsm float, at batch 512, a worker raced the larger cell
then the smaller one at or just below the 48 KB hole, and a fresh child raced the smaller one. Every
candidate's feasibility agreed. potrf pads requests out of the hole and its kernels carry no static
shared ([the 48 KB launch hole](../perf/potrf.md)). The guard stays in the test suite. The mismatch
path is covered by injected verdicts in `tune_tests`.

## Engine: measured per-child overhead

The worker decision rests on this measurement (threadripper02, sm_120, GPU 1, other GPUs idle, one
measuring process at a time, 2026-10-07). The cell is potrf float `uplo=L,n=16,batch=8192`, run as 20
sequential `batchlas_tune_impl --cell` children after a warm-up child.

| Configuration | median s | p90 s |
| --- | --- | --- |
| `time`, 1 arm, reps 1, warm 0 | 0.488 | 0.494 |
| `probe`, 1 arm, no timing | 0.488 | 0.492 |
| `time`, 6 float arms, reps 3, warm 0.2 | 1.802 | 1.809 |

The 0.49 s is process start, SYCL and CUDA init, static init, JIT-cache load and verification
launches. It is derived as wall time minus in-child kernel time, and is a lower bound because the
JIT cache was warm. Against a raced preview cell that is 37-60% of the measuring time. The old driver
ran three children per cell, about 1.5 s of start-up per cell, so per-cell processes would add about
70% to the preview estimate.

**Decision:** one persistent worker per GPU.

## Engine: driver interface

- `batchlas_tune <op>[,<op>...|all] --tier preview|coarse|deep --dtype ... --devices ...`. The old
  protocol flags (`--reps`, `--warm`, `--passes`, `--remeasure`, `--refine-ratio`) remain as expert
  overrides. A run that uses them is tier `custom`, ranked below preview, and writes to the shared
  ledger only with an explicit `--ledger DIR`.
- `--plan` prints cells, skipped cells with reasons, and the estimate, then exits.
- `--budget <h>` caps refinement time.
- `--progress-fd <n>` writes one JSON line per event (cell started or finished, candidate eliminated,
  audit verdict, worker restart, refinement cap). benchviz consumes this over SSE.
- `--status` prints the op × dtype × device matrix (tier mix, coverage, stale cells, audit verdicts,
  age) from ledgers and tables, without a GPU.
- `--gate` is unchanged.

## Engine: refinement convergence rules

Implemented in `refine_all_axes` (`tools/tune/grid.cc`), shared by the driver and `tune_replay`. Each
cell's raced medians arrive as a `RefineCell`.

1. **Flip.** A bracket is a winner flip when the two winners differ and the flip is decisive at
   *either* end: there the other end's winner is more than the 3% tie slower than the local winner.
   A winner that is not runnable there, or was eliminated, counts as decisively slower. Near-tie
   alternation is not a flip.
2. **Margin trigger.** The preview margin (10%) applies only to brackets whose ends are both
   starting-lattice cells at the running tier. A refinement midpoint never re-triggers it.
3. **Batch.** `batch` is never bisected below its lattice spacing. It refills its own axis values.
4. **Cap.** Refinement cells per op and dtype are at most `refine_cap_factor` × round-0 cells
   (preview 3.0, coarse 1.0, deep 2.0). Both counts include the ledger's current records, so a run
   stopped by Ctrl-C or `--budget` resumes within the same total. Flips are refined before margin
   hedges. Past the cap, refinement stops and the run reports it.
5. **Estimate.** The refinement estimate is ratio × lattice cells measured now, where ratio is the
   ledger's refined / round-0 count at the tier (at most the cap factor), else half the cap factor.

The rule must be two-sided: requiring rule 1 at *both* ends dropped the trsm flips, whose crossover
sits near one end of the bracket, and failed every tier in replay.

**GPU run** (potrf float, preview, one GPU, persistent worker): 90 lattice cells and 90 refinement
cells in 7 rounds (90, 46, 23, 12, 6, 2, 1), 219 s wall. Refinement converged well inside the cap of
270. The estimate was about 2x high, because the no-history default assumes 1.5 refinement cells per
lattice cell.

## Engine: testing and acceptance

Tests follow the agent guide's "guards that cannot fail" rules (@ref dev_agent_guide §8).

- **Offline replay.** `tune_replay` (`tools/tune/replay.cc`) replays a tier's racing and bisection
  against the exhaustive trsm sweeps (4452 float and 4098 double cells, every rep of every candidate),
  without a GPU. The score is the table misrank: the fraction of cells where the nearest row's first
  runnable entry is more than 3% slower than the exhaustive best. Acceptance: deep at most 0.2%,
  coarse at most 1%, preview at most 5%, each measured as the excess over the noise floor.
- **Unit tests without a GPU.** Precedence (a coarse record never displaces a current deep one; a
  stale deep one is displaced), gap fill, partly-stale selection, lattice nesting and `--check`
  reproducibility. Each has a deliberate break that turns only its own test red.
- **Racing guard.** A synthetic spec with a true near-tie, a candidate 4% slower than the leader
  (eliminated in deep) and one 2% slower (kept as a tie).
- **Audit guard.** A candidate whose feasibility differs between fresh and warm processes is flagged.
- **End to end.** A preview potrf run on one GPU finishes within the `--plan` estimate, and its
  tables pass `tuned_tables_tests` and `--check`. A coarse trsm float run on sm_120 agrees with the
  deep table within the replay bound.

## Engine: replay results on trsm

Holdout mode (the default) races on pass-1 reps and scores against pass-2 medians. Pass 1 and pass
2 disagree by more than 3% on about 1.5% of float cells (1.15% double). An oracle that times every
cell and picks from all pass-1 reps scores that same floor, so the verdict is the excess over it.

| Tier | dtype | Bound | Measured (lattice + refined) | Excess, race / table | Verdict |
| --- | --- | --- | --- | --- | --- |
| preview | float | 5% | 2197 (616 + 1581) | 0.41% / 3.89% | PASS |
| preview | double | 5% | 1763 (585 + 1178) | 0.21% / 4.25% | PASS |
| coarse | float | 1% | 4427 (3480 + 947) | 0.03% / 0.04% | PASS |
| coarse | double | 1% | 4066 (3219 + 847) | 0.03% / 0.12% | PASS |
| deep | float | 0.2% | 4427 (3480 + 947) | 0.01% / 0.09% | PASS |
| deep | double | 0.2% | 4068 (3219 + 849) | -0.02% / 0.07% | PASS |

The estimated cost for trsm float on sm_120 is 0.57 GPU-h for preview, 0.86 for coarse and 0.86 for
deep. The estimate is `est_gpu_h = [measure_s + live_candidates × (warm_topup_s + verify_s) +
cell_overhead_s per cell] / 3600`, with `warm_topup_s` 0.2, `verify_s` 0.05 and `cell_overhead_s` 0
for a persistent worker. Per-candidate warm-up dominates; the summed rep time is 0.07 GPU-h.

**Alternatives** (same-sample replay, table misrank over all raw cells, float / double):

| Alternative | Result | Verdict |
| --- | --- | --- |
| Stride 2 start, geometric bisection to 1.25 | 7.9% / 9.7% on 820 / 712 cells | rejected |
| Stride 2 start, index bisection, no margin | 6.7% / 5.8% on 1412 / 1096 cells | rejected |
| Stride 2 start, index bisection, margin 0.10 | 3.6% / 4.3% on 2818 / 2363 cells | preview |
| Stride 4 start, any refinement | 7-13% on 501-1812 cells | rejected |
| Full lattice, geometric bisection to 1.25 | 2.5% / 1.4% on 4021 / 3725 cells | rejected |
| Full lattice, geometric bisection to 1.1, coarse race | 0.11% / 0.05% on 4413 / 4094 cells | coarse, deep |
| No bisection on the full lattice | 6.4% / 6.1% on 3480 / 3219 cells | rejected |
| Shrunk trsm grid: batch stride 2 | 4.3% / 4.8% on 2920 / 2546 cells | rejected |
| Shrunk trsm grid: one batch value | 13.5% / 7.2% on 916 / 814 cells | rejected |
| Shrunk trsm grid: q stride 2 | 1.5% / 3.0% on 2964 / 2435 cells | rejected |
| Shrunk trsm grid: order stride 2 | 4.7% / 6.9% on 1911 / 1837 cells | rejected |
| Shrunk trsm grid: q, batch and order stride 2 | 7.8% / 9.7% on 824 / 716 cells | rejected |

Batch is the costliest axis to shrink, because the winner flips along it. No shrunken grid meets the
1% coarse bound. The full lattice does.

## Coverage: interface for sub-project 2

- Each of the 15 remaining ops gets `<op>_spec.cc` implementing `OpSpec` (`tools/tune/spec.hh`):
  keys, candidates and axes from its `choice.hh`, a problem builder, host verification, a size cap,
  and per-family kernel sources. Ops that compose other ops (gesv, getri, syev, gesvd, posv) declare
  those dependencies for op order and hashes.
- The constants tuner becomes a second spec kind. Its candidates are values of one
  `tuning_params.hh` knob, its cells are the knob's n-buckets and dtypes, and it measures through the
  `BATCHLAS_TUNE_*` overrides. Its ledger yields a generated header. A knob with two consumers is
  tuned on the consumer the maintainer names; the other is reported, not retuned.
- `evaluation/tuning/` is retired once its knobs are covered. Its `sensitivity.py` axis pruning becomes
  the knob's starting lattice.

## benchviz Tuning tab: interface for sub-project 3

The tab reads ledgers and tables directly and drives `batchlas_tune --plan`, `--progress-fd` and
`--status`. It keeps benchviz's server, SSE stream, `gpu_guard.sh`, campaign store and figure style
(`benchmarks/benchviz/style.py`).

- **Status matrix.** op × dtype × device: tier mix, coverage, stale cells, audit verdict, age.
- **Launch and progress.** Choose ops, dtypes, devices and tier. The estimate is shown before start.
  Cells and eliminations stream live. Stop and resume use the skip rule.
- **Winner map.** A heatmap over two chosen keys, with the others fixed. Colour is the winning family,
  intensity the margin to the runner-up, and a marker the tier. Clicking a cell shows every
  candidate's timing and interval.
- **Diff against the shipped table.** Cells whose winner changed, with the predicted per-cell
  speedup or slowdown, before tables are written.

## Open risks

- Racing assumes timing noise is stationary within a cell. Clock ramps after idle gaps break that.
  The worker's warm start and top-up are the defence. The replay cannot test this, because the raw
  files come from warm, back-to-back runs.
- If the audit fails often, the speedup depends mostly on racing and the adaptive grid.
- Deferred, not implemented: per-rep `<run id>.reps.jsonl` files for deep runs, and guard settings
  and tolerated foreign pids in `run` records.
- Budgets assume one GPU on a quiet box. On the shared 4090 box, guard waits come on top.
