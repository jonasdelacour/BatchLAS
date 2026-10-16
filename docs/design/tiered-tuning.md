# Tiered tuning {#design_tiered_tuning}

> **Status:** current · sub-project 1 (engine) implemented, validated on threadripper02 (RTX PRO 6000,
> sm_120) 2026-10-07 · specs for all 19 ops; the sm_120 deep run ships `tuned/*.sm_120.txt` · Tuning
> tab built 2026-10-09 · the constants tuner (sub-project 2) is specified at interface level only

The tuner fills the selection tables and the tuning constants in tiers. The user picks a tier
(preview, coarse or deep) per run. Measurements that cannot change a table are skipped, and a
per-cell ledger ensures that a cheaper run never overwrites a better result. The current tuner is
described in @ref tune_tool_readme; the flat selection it feeds is in @ref design_flat_selection §6.
The sm_120 tables come from the engine ("Engine: the sm_120 deep run").

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
| 2 | Coverage: specs for the other 15 ops; constants tuner on the engine; `tuning_params.hh` generated | 1 | specs implemented; constants tuner: interface below |
| 3 | benchviz Tuning tab: four views | 1 | built 2026-10-09; interface and as-built notes below |

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
| Race reps, floor per candidate; cap 50 ms of the slowest arm, at most 16 ("Engine: the lean cell") | 3 | 4 | 6 |
| Elimination confidence | 0.80 | 0.90 | 0.98, plus a reversed confirmation round |
| Fresh-process audit sample | 2% of cells | 2% | 1% |

The tie margin is 3% in every tier. The lattices nest (preview ⊂ coarse ⊂ deep). An op whose grid
is not a lattice is subsampled by a stable hash of the cell key, so subsets still nest.

Each cell goes through three steps:

1. **Probe.** Each candidate's pin is checked with the `*_buffer_size` probe under
   `select::ScopedPin`, or one untimed run for trsm. With 0 or 1 runnable candidates the cell is
   recorded and nothing is timed.
2. **Skip.** A current ledger result at the same or higher tier skips the cell. A partly stale
   result re-races only the changed candidates, against the stored winner and runner-up.
3. **Race.** Gross losers are cut and the survivors warmed ("Engine: the lean cell"). Survivors are
   then timed in interleaved rounds, with the order rotated per round and inputs restored untimed. A
   candidate is eliminated when it is confidently slower than the leader by more than the tie
   margin. The race stops when one candidate remains, when the survivors are confidently within the
   margin, or at the rep cap. The race starts from the winner of the nearest finished neighbour.
   Each candidate that can rank first is verified on the default items (`batchlas::verify`,
   @ref design_verification), every candidate once per size decade ("Engine: tie-set verification").

Across cells:

- **Breadth first.** The starting lattice of every op and dtype is measured before any refinement,
  so an interrupted run leaves usable tables.
- **Bisection on every key axis** (n, batch, q, nrhs), using each key's own distance. `log` axes
  bisect geometrically, `exact` axes (uplo, side, trans) never, and `batch` refills only its own
  axis values. Nothing is interpolated. `select::choose` takes the nearest row.
- **Estimate and budget.** The driver predicts cost from earlier ledger timings for the same op,
  dtype and device, or from a byte model. A cell takes the recorded phases of the nearest cell of
  the same protocol, tier and mode (worker or fresh; exact keys equal, log distance), its per-arm
  phases (warm, timed, verify) scaled by the arm count over `ph_arms`; without such a cell, the lean
  model: `--cell-overhead-s` plus, per arm, (`lean_max_reps` of the slowest arm + 3) × its nearest
  median + min(0.2 s, 2 × median) + 0.05 s verify. Each history cell also carries
  the recorded worker-start rate × median start and the audit fraction × median audit; the run once
  carries the ld audits (`ph_ld` summed per run, median over runs). `--plan` says how many cells came from
  history. `--budget <h>` stops refinement; the starting lattice always completes.
- **Op order follows dependencies.** potrf and trsm are tuned before posv, whose times depend on
  their choices (@ref tune_tool_readme, section Specs). The map over all 19 ops is in "Engine: op
  order over all 19 ops".

## Engine: the ledger and table generation

**Ledger.** One file per (op, device): `benchmarks/results/tuning/ledger/<op>.<device>.jsonl` (Git
LFS), every line tagged with its dtype, appended under `flock` by every dtype job and process. The
layout, the line kinds and the dirty-run rule are in @ref design_tuning_data; it replaced one
directory per (op, dtype, device) with one file per run on 2026-10-11.

| Record | Fields |
| --- | --- |
| `run` | dtype, run id, tier, host, device key and name, batchlas git sha (`-dirty`), argv, worker mode per op and dtype, date, keys, candidates, protocol |
| `cell` | run id, tier, key fields, `round` (0 = starting lattice); per candidate: family hash, status (`ok`, `skipped`, `bad`, `error`, `eliminated`), reason, median ms, interval, reps; the ranked tie set; wall seconds per phase (`ph_*`) and `ph_mode` |
| `audit` | run id, cell key, fresh-process and persistent medians, verdict, `audit_s` |

Results merge by device key (sm_89, sm_120); the host is recorded.

**Hashes.** Kernel hash v2, which ignores comments and whitespace
([kernel hashes](tuning-data.md#tuning-data-kernel-hash)). A spec's `kernel-sources` block maps each family to its files, plus a common set
(the op's shared headers) that every family depends on; `src/select/` is not hashed. Each spec lists
its `src/ops/<op>/<op>.cc`, where `can_run` lives, as a common `kernel-deps` entry, so an edit to what
can run stales every cell. `kernel-deps` adds files
outside the source list (`// family: <name> "path" ...` or `// common "path" ...`). The posv
coupling is explicit: posv's families list the potrf and trsm families they call.

**Which record counts, table generation:** @ref design_tuning_data
([the ledger file](tuning-data.md#tuning-data-ledger-file),
[generation and the check](tuning-data.md#tuning-data-generation-and-check)).

**Import.** `batchlas_tune --import-raw` turns a schema-1 raw sweep into a deep run. Its hashes are
`legacy:<kernels>` unless the raw file's op hash equals today's, so an old sweep imports stale.

## Engine: the lean cell

The race of one cell (`run_race`, `tools/tune/cell_runner.hh`; constants in `race.hh`). Ledgers
record it as `protocol=lean`; `full` is the engine before it (a fixed 0.2 s warm-up per arm), and an
absent field is older still. `--plan` history reads only `lean` cells.

1. **One untimed run per arm**, so JIT and module loads stay out of the cut.
2. **Gross cut.** Two timed interleaved rounds, then `race_step`. Two pairs are below every tier's
   floor, so only the gross rule can fire: an arm above 4x the leader on both runs leaves, with the
   median of those two runs ("Engine: readiness for a deep run over all 19 ops"). It gets no
   warm-up share and no reps.
3. **Warm until stable.** The survivors' samples are dropped, then interleaved untimed rounds run
   until two consecutive round totals (run spans only) agree within 2%, after at least 2 rounds and
   at most 0.2 s per arm (`kWarmTol`, `kWarmMinRounds`, `kWarmCapPerArmS`; `warm_converged`).
4. **Rep cap from a time budget.** `lean_max_reps` = ceil(50 ms / the slowest survivor's median over
   the last two warm rounds), clamped to [`min_reps_for(confidence)`, 16] (`kRepCap`). The
   floor is the first round count at which the paired-ratio rule can decide (3 / 4 / 6), so the
   confidence rule stays satisfiable; a tier's `min_reps` equals it. Rounds are shared, so the
   slowest arm sets the count.
5. **Verification** of the tie set and `CellRequest::verify` only ("Engine: tie-set verification").

The cell runs on the worker's one `Queue` (`OpSpec::run_cell(req, q)`; the ld audit's re-run reuses
it). The bench's workspace is a lease from that Queue's arena, which keeps its blocks across cells,
and `~ArmBench` drains the Queue so an async error cannot surface in the next cell. Each spec's
`clear_info` poisons outputs that `reset` does not restore (syev `w`, gesvd `s`/U/Vh, getrf
pivots, geqrf `tau`), so an arm that writes nothing cannot pass on an earlier arm's output.

## Engine: tie-set verification

A cell verifies only the arms that can rank first, plus the arms its bucket has not verified yet.
Rules (`verify_next` in `race.hh`, `VerifyBook` and `verify_bucket` in `schedule.hh`):

- **Tie set.** After the race, the unverified arms within the 3% tie of the fastest survivor's median
  (with no survivor: of the fastest eliminated arm), the order `record_from_arms` ranks by. A failure
  leaves the set, and the next arms that can rank first are verified, until the set holds no
  unverified arm. Gross-cut and eliminated arms outside it are not verified.
- **Once per bucket.** Bucket = (op, dtype, exact keys, floor(log10 per-item bytes)). The driver
  sends each cell the arms that have no verdict in its bucket yet (`CellJob::verify`, worker request
  `verify`, child `--verify`). A pin refusal or an unverified arm stays wanted.
- **Unverified arms stay in the row** with ledger flag `v<i> = 0` (@ref design_tuning_data): a pin
  that falls through needs them.
- **A failure** (residual or info) is a verdict on the kernel in its bucket. Later cells of the bucket
  record the arm `bad` (`verify failed @ <key>`) without timing it. Each current record of the bucket
  that ranks it unverified, this run's or the ledger's, is appended again with it `bad`, through the
  ld audit's retract path. Records that verified it keep it. The driver prints `verify: <arm> marked
  bad in N recorded cells of bucket <b>` and emits `verify_retract`.
- **A verified winner.** A record's front arm is one it verified (`verified_front`). When a retraction
  (ld audit or bucket fail) or a partial re-race whose arms all failed would put an unverified arm
  first, unverified arms move below the first verified one (none: the ranking empties), and the cell
  is queued again once in the run (`remeasure`, all live arms), which verifies the new tie set.
- **Fresh audit.** The fresh child gets the cell's `verify` list. `audit_compare` treats an arm one
  side left unverified as feasible if it ran there (ok, eliminated or bad), so `bad` against
  unverified is no mismatch. Refused or `error` against ran still is. The fresh child's own verdicts go
  into the book like a cell's.

Evidence: `TuneVerify.*` in `tests/tune_tests.cc`. Deliberate breaks, each with its own red set:
`want()` asking nothing, retracting verified records, `audit_compare` ignoring the flag, a strict tie
(no 3%), no `v<i>` field, no retract, the audit's verdicts not booked, verifying every arm.

## Engine: persistent workers and the carve-out audit

**Worker.** Each GPU gets one long-lived `batchlas_tune_impl --worker`, started by the driver under
the existing `CUDA_VISIBLE_DEVICES` fence and per-GPU flock. Cells arrive on a pipe and come back as
`cell` records.

- The worker warms the clocks for 3 s at start; each cell then warms its survivors until stable.
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
   refused or failing in the other, or a winner more than 10% slower in the fresh run. An arm left
   unverified on either side only has to have run on both ("Engine: tie-set verification"). A mismatch
   marks that op and dtype fresh-process only. Each audited cell gets an `audit` record.

**Result.** No carve-out mismatch was found (threadripper02, GPU 1, 2026-10-07, `tune_race_gpu_tests`).
For potrf float and double, posv float and trsm float, at batch 512, a worker raced the larger cell
then the smaller one at or just below the 48 KB hole, and a fresh child raced the smaller one. Every
candidate's feasibility agreed. potrf pads requests out of the hole and its kernels carry no static
shared ([potrf: the 48 KB launch hole](../perf/potrf.md#potrf-the-48-kb-launch-hole)). The guard stays in the test suite. The mismatch
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

The estimated cost for trsm float on sm_120 was 0.57 GPU-h for preview, 0.86 for coarse and 0.86 for
deep under the `full` protocol's model (0.2 s warm-up per candidate). The estimate is now `est_gpu_h = [measure_s + Σ_candidates
(3 × median + min(0.2 s, 2 × median)) + (verify_s + cell_overhead_s) per cell] / 3600` (`lean_arm_fixed_s`,
the same per-arm term as `--plan`'s model; `verify_s` once a cell, for the tie set), with `verify_s` 0.05
and `cell_overhead_s` 0 for a persistent worker. The summed rep time is 0.07 GPU-h.

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

Built on 2026-10-09 as `/tuning` in `benchviz serve` (`benchmarks/benchviz/tuning.py`,
`tuning.html`; user guide in benchmarks/benchviz/README.md "Tuning tab"). Where it departs from the
interface above:

| Interface above | As built | Why |
| --- | --- | --- |
| reads `--status` | reads the ledgers through `scripts/sweep_to_table.py` (`read_ledger`, best record, staleness, `ledger_rows`, `parse_table`, `nearest`) | the views show exactly what `sweep_to_table.py` would write next and need no build |
| status matrix: staleness | per kernel family: the table header's hash against hash v2 of the sources now, the keys the ledger holds stale in it per dtype, and `stale_kernel_warnings`; per dtype, the table's rows and `behind` when its header does not name the newest run | one table per (op, device) carries the hashes; a row-exact check per dtype (the `--check` rule, header hashes) is in the op view, where it costs one `ledger_rows` |
| launch: tiers | preview, coarse, deep, and custom with listed cells (`--remeasure-keys`); `--allow-dirty` on request; `--device-key` when no GPU is visible | the tuner's flags since the tuning data layout (@ref design_tuning_data) |
| `gpu_guard.sh`, campaign store | the tuner's own guard (`--allow-idle-foreign`, `--guard-wait`); runs in `benchviz_runs/_tuning/<run>/` (`run.json`, `events.jsonl`, `run.log`, `exit`, `remeasure-keys.txt`) | the tuner already guards every GPU it uses; a run is not a campaign |
| figure style | interactive SVG | the map is clicked, not printed |

Since the tuning data layout the tab reads one ledger and one table per (op, device) and filters
each by dtype. Loading all 39 ledger files takes about 13 s (threadripper02, 2026-10-11); the
server warms this at start.

The diff's two picks are the same section 5.4 lookup: in the shipped table, and in the rows
`sweep_to_table.py` would write now. Each takes the first entry the cell's record did not
refuse (`skipped`, that is `can_run` false). The predicted speedup is the ratio of the two picks'
medians in that one record. A race calls candidates within 3% a tie and keeps the earlier one, so
a changed pick can read as low as 0.971×. Against main's tables (2026-10-09), gemm double
on sm_120 changes 2918 picks, geomean 1.08× where both are timed.

Two engine additions came with it: every progress event carries `t` (Unix seconds), and `--plan`
emits one `plan_job` event per (op, dtype) (tools/tune/README.md). `ledger_rows` now compares a row
only with keys in its own and the adjacent `floor(index / stride)` buckets. All 70 sm_120 ledger
tables came out byte-identical, in 8.4 s instead of 126.6 s.

## Engine: readiness for a deep run over all 19 ops

> **Status:** as built 2026-10-08, after every op got a spec. The problems come from the spec
> writers' preview runs on threadripper02 (sm_120).

**Strict pins.** `vendor` is both the vendor family's spelling and a routing class word, and a
class word that cannot serve the shape falls back to Auto with a warning. A tuner arm pinned to
`vendor` where `can_run` refuses it therefore timed Auto and credited the time to the vendor
(seen on spmm cdouble N/N nrhs=1 and on gesvd). The tuner's pin is now strict
(`select::ScopedPin(op, word, select::StrictPin{})`, used only in `cell_runner.hh`): a class word
that nothing in its class can serve throws `std::invalid_argument`, as a spelling does, and
`vendor` must resolve to the candidate spelled `vendor`. The arm is `skipped`, reason
`pin refused (strict): ...`. Library callers and `BATCHLAS_<OP>_ROUTE` keep the fallback. spmm's
spec-level workaround is gone; gesvd's stays, because there `can_run` admits a shape whose vendor
call faults the device (known defect 15), which no pin rule can see.

**Gross losers.** After every round from the second on, whatever `min_reps` says, the race
eliminates an arm whose lower median paired ratio to the leader exceeds 4.0 (`kGrossLoserRatio`,
`race.hh`; the lower of the two middle ratios, so at two rounds both must exceed it). One stalled
round never kills an arm. It is recorded `eliminated` with the median of the rounds it ran. The lean
cell applies it first, after two rounds and before the warm-up ("Engine: the lean cell"). geqrf's vendor
(`geqrfBatched`, one CTA per matrix) is tens to hundreds of times slower than blocked on large
cells, and before this rule each of its timed rounds cost minutes.

## Engine: matrix-dimension cap

`--max-dim N` (default 2048 for tiered runs, 0 = off) is a maintainer rule: no cell is measured with a matrix dimension above N, whatever its bytes. `OpSpec::dims` returns the extents a cell allocates (default: every integer key but `batch`; geqrf overrides it because its `m` is derived from `form`, `n` and `aspect`), and `plan_round` plans a larger cell as `skip:dim` before the byte cap, so refinement never measures one either.

## Engine: dominance carry-forward removed

Until 2026-10-08 an arm eliminated at more than 10x the winner's median was not timed at cells
*beyond* that cell in the same run, op and dtype (larger along `refine_key()`, or every other
integer key at least as large with more bytes), and was recorded `skipped`, reason
`dominated:<key of the losing cell>`. It was removed after it fired thousands of times in the
sm_120 deep run (`trsm cdouble vendor` 569 cells, `symm double vendor` 660; also gemm, syrk,
syr2k, trmm, geqrf):

- It assumed an arm that loses by 10x keeps losing as the cell grows. That is false at a
  vendor/native crossover: the vendor arm loses on launch overhead at small sizes and wins at large
  ones, which is exactly the edge the tables must place.
- "Beyond" crossed lines: a loss at one `aspect` or other key carried to cells with a different
  one, where the ranking need not be related.
- Its motivation, a geqrf vendor call of many minutes on a float tall n=1024 aspect=256 cell, is gone
  with the matrix-dimension cap above.

What stays is the in-race gross-loser rule (ratio > 4 over two rounds), which only uses data from
the same cell. A ledger record that still holds a `dominated:` skip is *partly stale*: the skipped
arm's family counts as stale whatever its hash, so the next run re-races exactly those arms against
the stored winner and runner-up at the stored tier and writes a merged record (`stale_candidates`,
`ledger.cc`, and `stale_families` in `sweep_to_table.py`; shared fixture
`tests/data/ledger_dominated`).

## Engine: the guard reads NVML, falls back to nvidia-smi {#engine-guard-retries-a-failed-nvidia-smi-query}

**Backend.** `--guard-backend auto` (default) loads `libnvidia-ml.so.1` with `dlopen` in the driver
(`nvml_guard.cc`; structs declared locally, no `nvml.h`). NVML creates no CUDA context and ignores
`CUDA_VISIBLE_DEVICES`. Its index is nvidia-smi's `--id`; preflight dies if the UUIDs differ. If NVML
does not load, `auto` warns and uses nvidia-smi; `nvml` refuses to start; `nvidia-smi` forces the
old path. A query that fails mid-run warns once and switches the run to nvidia-smi. The run line
records `guard` (`nvml`, `nvidia-smi`, `off`; `nvml->nvidia-smi` once a run fell back, rewritten at the
next cell).
The custom-run `meta` and the gate's `guard` record hold the backend at start. `batchlas_tune --guard-probe` prints what both
backends see on every GPU, read-only, and exits 1 on a disagreement.

**Activity.** NVML samples SM load per process (`nvmlDeviceGetProcessUtilization`). The driver drops
every live worker, and each exited child or worker for 10 s (NVML's sample lag; pids get reused). `foreign_active` (`tune_core.hh`) flags a tolerated pid or the
whole GPU above `--util-ceiling`, and any other pid at any SM load.

| check | NVML | nvidia-smi |
| --- | --- | --- |
| before a cell or child | compute pids, foreign activity in the last 1 s | compute pids; whole-GPU utilization at a full guard (after a 1 s idle on a worker, every 60 s) |
| after it | new pids, foreign activity since its start | new pids |
| `ph_guard` per worker cell (one potrf coarse smoke, GPU 2; an observation) | 0.002-0.003 s | 0.09-0.10 s, plus the 1 s idle per minute |

A tolerated pid that becomes active during a cell discards it (`rerun_discarded`, `tune_core.hh`).
The run, worker cell or fresh child, is re-run at once; after a second discard each re-run first
idles 1 s and waits for a quiet GPU, and after `--guard-wait` of that the run stops. A discard never
becomes an arm outcome, an arm error or a candidate drop. NVML reports a process's last sample
period only, so a burst that ended before the after-check can escape.

**Retries.** The sm_120 deep run died with `cannot query GPU 3 with nvidia-smi` while the maintainer
reset GPU 0: `nvidia-smi` fails for every GPU for a moment during a reset. A failed nvidia-smi query
(`--query-compute-apps` or the utilization sample) is retried after 5 s, doubling to 60 s, until
`--guard-wait` (default 300 s) of retrying has passed, with one line on stderr per retry; only then
does the run stop (`query_with_backoff`, `tune_core.hh`).

## Engine: repeat crashers are dropped

An arm benched for confirmed worker errors (two consecutive cells in which the worker and the fresh
child both reported `error`) races alone in a fresh child per cell. If it then errors in 5
consecutive such cells (`kDropAfterErrors`, `worker.hh`), it is dropped for the rest of that op and
dtype in this run: `error`, reason `dropped after 5 consecutive errors`, not run. One clean cell
resets the count. The case is the cdouble gemv vendor, which segfaults inside cuBLASLt on every
shape on sm_120 (known defect 13) and cost two worker crashes and a retry per cell, about +3 h on a
deep gemv cdouble run.

## Engine: op order over all 19 ops

A multi-op run measures op by op in a stable topological order of `op_dependencies()`
(`schedule.cc`): an op follows every op whose table its timings depend on through a public entry
point, and input order breaks ties. `all` starts from every registered spec in the canonical order,
which gives

gemm, trsm, syr2k, gemv, spmm, trmm, symm, syrk, potrf, getrf, getrs, geqrf, ormqr, getri, posv,
gesv, orgqr, syev, gesvd.

| op | after |
| --- | --- |
| trmm, symm | gemm (`expand` calls the public gemm) |
| getrf | gemm, trsm |
| getrs | trsm |
| geqrf | gemm |
| ormqr | gemm, trmm |
| getri | trsm, getrf |
| posv | potrf, trsm |
| gesv | getrf, getrs |
| orgqr | ormqr |
| syev | gemm, trmm, syr2k, geqrf, ormqr |
| gesvd | gemm, trmm, syr2k |

A dependency not in the run holds nothing back. The hashes still do not follow a dependency's
table: retuning gemm changes trmm's `expand` times without staling trmm's cells.

In a multi-op run, an (op, dtype) pair the spec refuses (symm, syrk and syr2k are real-only) is
skipped with one printed `== note: skipping <op> <dtype>: ...` line. A single-op run still refuses
it.

## Engine: pipelined cell queues across op and dtype jobs

The driver used to shard each round round-robin over the GPUs and join every GPU at the end of each
(op, dtype) round, so one GPU finishing gemm float's largest round-2 cells left the other three
idle; over 19 ops, 4 dtypes and several rounds those tails cost hours. Now each job's round is one
queue sorted by (per-item footprint, bytes), and one thread per GPU (`gpu_loop` in
`tiered_driver.cc`) pops the next cell of the earliest job, in op order, that has queued cells.
Every GPU therefore still runs ascending footprints within a job's round, and the carve-out restart
rule is unchanged; moving to another job usually costs a restart, which the summary line and the
`schedule` progress event count. A job's next round is planned by whichever thread records its last
cell, never earlier, because refinement reads that round's results; until then the job waits for
its stragglers while idle GPUs work on later jobs. Pipelining does not break the op order's
dependencies: during one run a composed op times against the library's embedded tables, not the
ledger, so an earlier op's unfinished rounds change nothing it measures. Host tests
(`TuneTieredPipeline.*`) check no wait behind a straggler, ascending footprints per GPU and round,
planning only after the round's last record, and records identical to a one-GPU run.

## Engine: the ld audit

Added 2026-10-09. Every cell times and verifies at one leading dimension (`ld_pad`, normally 0), so a
candidate that derives `ld` from `n` verifies, wins and ships. The deep sm_120 run was suspected of
shipping exactly that for the syev `vendor` family. That suspicion turned out to be wrong
(@ref design_known_defects "known defects", the LOBPCG entry), but nothing in the tuner
could have caught such a defect if it had been real.

**The rule.** For each (op, dtype) in a tiered run, the driver asks every candidate's *first verified
cell* (status `ok`, or `eliminated` after passing verification) for one extra verification at
`ld_pad + 3` (`kLdAuditPad`, `tools/tune/spec.hh`); an arm the lean race left unverified waits. The check runs in the same worker or child,
straight after the timed cell, as a `verify`-mode cell, which is a fresh problem built by the spec
at the padded `ld`. `LdAuditBook` (`schedule.hh`) records which arms have a verdict (pass or
fail). A candidate that the cell did not verify, or whose pin refused the padded shape there
(`skipped`), stays wanted for its next cell.

**On a failure.** If the padded verification comes back `bad` or `error`, or throws, the arm is
`bad` in that cell's record with the reason `ld audit: ld_pad +3: ...`. The failure condemns the
kernel, not the cell, so the arm is `bad` in every cell of that (op, dtype):

- later cells, and cells still running on other GPUs, record it `bad` (reason
  `ld audit failed @ <key>`) without timing it (`record()` applies `LdAuditBook::bad`);
- every current record that already ranks it, from this run or an earlier one in the ledger, is
  appended once more with the arm `bad` and out of `ranked` (`ld_retract`, `schedule.hh`). The
  newer line wins in `best_records` and `sweep_to_table.py`, so no table ranks it.

The driver prints `ld audit FAILED` and `ld audit: <arm> marked bad in N recorded cells`, emits
`ld_audit` and `ld_retract` progress events, and the run summary gives one line per (op, dtype)
with passed, skipped and failed counts and lists the failures (event `ld_audit_summary`).
`skipped` counts arms whose pin refused the padded shape at every cell that asked. The carve-out audit
(above) leaves out an arm that failed the ld audit, because its fresh child re-verifies at the
natural `ld`, where the arm passes. Counting that as a worker mismatch would send the rest of the
run to fresh children for nothing.

**Limits.** The batch stride is padded only where a spec derives it from `ld` (most use
`ld * cols`). gemm's `packed` layout ignores `ld_pad`. A spec that takes no `ld_pad` audits
nothing. The audit is a once-per-candidate check, not a sweep: a defect that appears only at some
shapes can still get through.

**Evidence.** Host tests `TuneLdAudit.*` cover the child-side fold, the book, the request and
outcome round trip, the driver's first-cell rule and reporting, and the carve-out interplay. A
deliberate break of each part turns exactly its own test red. GPU check (threadripper02, GPU 2): with
vendor syev's run calls passing `rows()` as `lda`, a preview `syev float jobz=V n=12:18
batch=2:256` sweep reported `ld audit FAILED: vendor ... bad residual` on its first cell and
`5 passed, 0 skipped, 1 failed`. With the break removed, the same sweep reported `6 passed, 0
skipped, 0 failed`.

## Engine: re-measuring named cells

Added 2026-10-09 for the cells of the sm_120 deep run that the box's other user damaged.
`--remeasure-keys FILE` takes one `<op> <dtype> <key>` line per cell, the key in the ledger's
own `name=value,...` form and field order. Each named cell counts as missing, whatever the
ledger holds: `plan_round` gives it the reason `remeasure`, every runnable candidate, the
running tier and a fresh record, not a merge. The stored record's `round` carries over, so a
re-measured refinement midpoint stays a refinement record for the cap and the margin hedge. The
driver appends the named cells to round 0 when they are off the lattice. Round 0 is never
budgeted, so the named cells always run, but so does every other cell round 0 plans: with stale
records (a kernel change) that is the whole stale lattice, whatever `--budget`. `--only-named`
makes round 0 exactly the named cells, recorded at the run's tier, with no lattice and no
refinement; tier `custom` always behaves so. A named cell
is measured past `--max-dim`, because the deep run measured some cells before the cap existed,
but not past `--cap-gib`. A key whose fields do not match the op's, or an op and dtype the run
does not tune, stops the run. The newer record wins in `best_records` (same tier, later date or
run id), so the tables pick up the re-measure with no other change. Host tests: `TuneRemeasure.*`
(a deliberate break of the planner, of the round-0 append or of `--only-named` turns them red).

## Engine: the sm_120 deep run

> **Status:** measured 2026-10-08/09 on threadripper02 (4x RTX PRO 6000 Blackwell, sm_120,
> CUDA 13.2, DPC++ `/opt/dpcpp-cuda`), GPUs 0-3, on a box shared with another user. The tables
> it produced ship as `tuned/*.sm_120.txt`; the ledger is `benchmarks/results/tuning/ledger/`.

**What ran.** `batchlas_tune all --tier deep --dtype float,double,cfloat,cdouble --devices 0,1,2,3`
covered all 19 ops and every dtype each op instantiates: 70 (op, dtype) jobs, matrix dimensions up
to 2048. It ran from 2026-10-08 06:23 to 2026-10-09 04:54 CEST, with four starts. The first
ran 11.5 h and wrote 56,144 cells, until a geqrf cell with an 8192-wide matrix timed out its
worker. The second died after 3 minutes and 345 cells when `nvidia-smi` failed during a GPU reset.
The third ran 4.2 h (32,912 cells) and the fourth 4.9 h (35,323 cells), when the maintainer
stopped it. That is about 20.6 h of measuring and 124,724 cell records. On 2026-10-09 two
follow-up runs went into the same ledger. One re-measured 250 cells that the other user's jobs
had damaged ("Engine: re-measuring named cells"): 2.5 min on 4 GPUs. The other re-ran gemv in
full (8,515 cells in 33.5 min on 4 GPUs), because the defect-13 fix below edited `src/ops/gemv/gemv.cc`
and so staled every gemv record; that run (`20261009T042647-threadripper02-422873`) was `-dirty`, so
every reader drops it, and gemv was re-measured at deep on 2026-10-11 (8,558 cells, run
`20261011T204301-threadripper02-39013`). The campaign wrote 133,489 cell records, the dirty run's
included. On 2026-10-12 the 19 sm_120 ledgers hold 210,505 cell records: 157,444 measured
(the 2026-10-11 samples and re-measures included) and 53,061 transcribed.

**Refinement coverage.** Every job's starting lattice is complete. Refinement is not: geqrf,
ormqr, getri, posv, gesv, orgqr, syev, gesvd and complex getrs got little of it before the stop
(a resumed run wanted 4,072 more refinement cells for the 15 ops of the re-measure run alone), so
their crossovers sit on lattice spacing.

**Interference.** Cells where every candidate stalled (about 2.3 ms or 7-11 ms, more than twice
both log-neighbours, which agree on the winner) were screened over each table's first log key: 83
strict outliers (over 2x both neighbours) and 252 soft ones (over 2x their log-interpolated
time). The 250 non-gemv soft cells were re-measured at deep; afterwards the screen found 16 strict and 154 soft outliers, 153 of them among the cells just re-measured.
The cells left reproduce, so they are real behaviour. Most are the gesvd `jacobi` cliff
at m = 33, where the sub-group solver stops fitting, and spmm and syrk cells within a few
microseconds of their neighbours.

**What the run found, and what was fixed.** Each fix has its own section or defect entry:

- *Dominance carry-forward* fired thousands of times and hid vendor/native crossovers. It was
  removed ("Engine: dominance carry-forward removed").
- *Unbounded matrix dimensions:* a geqrf vendor cell took many minutes. Fixed by the dimension
  cap ("Engine: matrix-dimension cap").
- *Idle tails:* GPUs waited at every (op, dtype) round boundary. Fixed by the pipelined queues
  ("Engine: pipelined cell queues across op and dtype jobs").
- *A transient `nvidia-smi` failure* killed the second start. The guard now retries
  ("Engine: the guard reads NVML, falls back to nvidia-smi").
- *Repeat crashers:* the complex<double> cuBLAS gemv segfaults on every shape. The tuner now drops
  such an arm ("Engine: repeat crashers are dropped"), and the library refuses it in `can_run`
  (known defects, defect 13).
- *LOBPCG:* with the new syev tables, LOBPCG's smallest-end solve returned 0. The cause was null P
  columns, not the vendor (known defects, the LOBPCG entry). The projected solves also need more
  accuracy than the speed tables can see, so they are now pinned to the CTA solver at n <= 32
  ([LOBPCG: the accuracy pin on the projected syev](../perf/syevx.md#lobpcg-the-accuracy-pin-on-the-projected-syev)).
- *Single-`ld` verification* could not catch a candidate that derives `ld` from `n`. Fixed by
  "Engine: the ld audit".

## Open risks

- Racing assumes timing noise is stationary within a cell. Clock ramps after idle gaps break that.
  The worker's warm start and each cell's warm-until-stable are the defence. The replay cannot test this, because the raw
  files come from warm, back-to-back runs.
- If the audit fails often, the speedup depends mostly on racing and the adaptive grid.
- Deferred, not implemented: per-rep `<run id>.reps.jsonl` files for deep runs, and guard settings
  and tolerated foreign pids in `run` records.
- Budgets assume one GPU on a quiet box. On the shared 4090 box, guard waits come on top.
