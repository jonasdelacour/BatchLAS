# Tiered tuning {#design_tiered_tuning}

> **Covers:** the redesign of how BatchLAS fills its tuned tables and tuning constants: user-chosen
> tiers (preview / coarse / deep), measurements that skip work which cannot change a table, a per-cell
> result ledger in which a cheaper run never overwrites a better one, and a tuning tab in benchviz.
> **Status:** design approved in conversation on 2026-10-07. Sub-project 1 (the engine) is
> implemented on branch `worktree-tiered-tuning-spec` and validated end to end on threadripper02
> (sm_120) on 2026-10-07: raw import round trip, preview potrf (4 dtypes), coarse trsm float on 4
> GPUs scored against the exhaustive sweep, resume and staleness ("Engine: end-to-end validation on
> sm_120"). No shipped table comes from it yet. Sub-projects 2 and 3 are specified at the interface
> level and get their own design pass. Machine facts refer to the RTX 4090 box (sm_89) and threadripper02
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
| Speed versus rigor | The user picks a tier per run: **preview** (sparse, fills gaps only), **coarse** (the full lattice, fewer reps) or **deep**. Budgets are per-op estimates that `--plan` prints from a cost model of warm-up, verification and rep time, not from rep counts ("Engine: where the old tuner's time went"). More GPUs shard the work. Every tier avoids measurements that cannot change a table. |
| Grid reduction | Sparse grids misrank 4-13% of cells with up to about 2x tail loss in the replay, so the lattice is not thinned in coarse or deep. **Preview** is the only sparse tier (every 2nd point, bisection in index space plus a runner-up margin) and fills gaps only. |
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

A tier is a named set of parameters. The algorithm is the same in every tier, so preview results are
valid low-fidelity data, not a different kind of number.

| Parameter | preview | coarse | deep |
| --- | --- | --- | --- |
| Starting lattice | every 2nd point of each `choice.hh` axis (ends kept) | the full lattice | the full lattice |
| Bisection between neighbours | in index space of the axis, until adjacent, then geometric to hi/lo < 1.1; also between two starting-lattice cells when a runner-up is within 10% | geometric, until hi/lo < 1.1 | geometric, until hi/lo < 1.1 |
| Refinement cap per op and dtype | 3.0 x round-0 cells | 1.0 x round-0 cells | 2.0 x round-0 cells |
| Race: min / max reps per candidate | 3 / 6 | 4 / 12 | 6 / 16 |
| Elimination confidence | 0.80 | 0.90 | 0.98, plus a confirmation round in reversed order |
| Fresh-process audit sample | 2% of cells | 2% | 10% |

The 3% tie margin is the same in every tier, because it is the ranking rule the converter and
`rank()` share. The lattices nest (preview ⊂ coarse ⊂ deep), so results from different tiers land on
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
  distance: `log` axes are bisected geometrically, `exact` axes (uplo, side, trans) never, and
  batch only ever refills its own axis values. A bracket refines on a decisive winner flip, not on
  near-tie alternation, and refinement is capped per op and dtype
  ([convergence rules](#engine-refinement-convergence-rules)). Nothing is inferred or interpolated: a
  table simply has fewer rows where neighbours agree, and `select::choose` already takes the
  nearest row.
- **Time estimate and budget.** Before starting, the driver predicts the run's cost from earlier
  ledger timings for the same op, dtype and device, or from a flop/byte model of the op against the
  device's ceilings when there are none. It covers the expected refinement cells too (this ledger's
  refined / round-0 ratio at the tier, else half the cap), and `--plan` prints it with and without
  them. The estimate is printed (and shown in the UI). `--budget
  <h>` stops refinement when the budget is spent; the starting lattice always completes.
- **Op order follows dependencies.** potrf and trsm are tuned before posv, whose cta and blocked
  times depend on their choices (@ref tune_tool_readme, section Specs). The full map over all 19
  ops is in "Engine: op order over all 19 ops".

## Engine: the ledger and table generation

**Ledger.** There is one directory per (op, dtype, device),
`benchmarks/results/tuning/ledger/<op>.<dtype>.<device>/`, with one file per run, `<run id>.jsonl`
(Git LFS). A run only ever writes its own file, and readers take the union of the files. An
append-only file in LFS cannot be merged, so two boxes tuning the same op would always conflict;
one file per run cannot.

| Record | Fields |
| --- | --- |
| `run` | run id, tier, host, device key and name, batchlas git sha (`-dirty`), argv, worker mode per op/dtype (persistent or fresh-process), date. Guard settings and tolerated foreign pids: a deferred follow-up, not written yet |
| `cell` | run id, tier, the key fields, `round` (0 = starting lattice), for each candidate: family hash, status (`ok`/`skipped`/`bad`/`error`/`eliminated`), reason, median ms, interval, reps; the ranked tie set |
| `audit` | run id, the cell key, persistent versus fresh-process medians and winners, verdict |

Per-rep timings in `<run id>.reps.jsonl` (same directory, deep runs only) are a deferred
follow-up: no run writes the file yet, and readers already skip the name. A
truncated last line (a killed run) is ignored with a warning, so a cell is either fully recorded or
absent.

**Per-candidate hashes.** A spec's `kernel-sources` block becomes `family -> files`, plus a common
set (dispatch, `src/select/`) that every family depends on. A family's hash covers the common set
plus its own files. The spec's `kernel-deps` comment block adds files outside the source list:
`// family: <name> "path" ...` for one family, and `// common "path" ...` for every family. Inside
the source block itself, a comment line `// family: <name>` opens that family's section (reopening
appends) and `// common` returns to the common set; paths before any marker are common, and the
op-level hash takes every path in block order. CMake and CI read only the source block. Each
spec lists its `src/ops/<op>/<op>.cc` there, where `can_run` lives, so an edit to what can run
stales every cell. Neither kind of line moves the op-level `kernels=` hash. `cmake/BatchLASTunedStaleness.cmake`, `.github/ci/check_tuned_tables.py` and
the driver compute it the same way. The posv coupling becomes explicit: posv's families list the
potrf and trsm families they call.

**Which record counts.** Precedence is deep > coarse > preview > custom > transcribed.

- A `cell` record is *current* when every candidate's hash matches the source tree. The hash of a
  candidate stored as `skipped` (it could not run at that cell) is not compared: an edit to its
  family cannot change the ranking, and a change to what can run goes through the hashed
  `<op>.cc`. A candidate skipped with reason `dominated:<key>` was never tried there, so
  its family is stale whatever its hash ("Engine: dominance carry-forward removed").
- It is *partly stale* when only some changed, or when the candidate list gained a family the
  record never raced. The next run of any tier re-races just those
  candidates against the stored winner and runner-up, at the stored record's tier, and appends a
  merged record. The re-raced candidates are ranked by their new times; the unchanged ones keep their
  stored times and order below them. Editing one kernel therefore keeps every deep cell deep, and
  costs a fraction of a retune.
- It is *stale* when the stored winner's hash changed or the winner's family was removed. The cell
  is re-measured at the running tier, like a cell with no record.
- A record in which every candidate is `error` (a failed child, not a result) counts as no record;
  the driver does not write one when a child fails without any `ok`, `bad` or `skipped` candidate.
  Both `best_records` (C++ `ledger.cc` and `sweep_to_table.py`) apply this one rule, so planning,
  refinement, `--status` and table generation agree.

**Table generation.** `scripts/sweep_to_table.py --ledger <file>` replaces `--tuner`. Its output
stays byte-reproducible, and `--check` keeps re-deriving every table. It refuses to overwrite an
existing timed table that is not a ledger's (a `converted` or `tuner:` source) unless
`--replace-timed` is given, so a preview run cannot silently replace measured tables; the
diagnostic `--assume-current` needs an explicit `--out` outside `tuned/`.

- For each cell it takes the best current record.
- A lower-tier row is emitted only to fill a gap: when no higher-tier row lies within the lower
  tier's lattice spacing under `select::choose`'s own distance. A coarse point can therefore never
  sit inside a region a deep run already resolved.
- Transcribed rows survive only where nothing measured covers them.
- The header reports the tier mix and the hashes:
  `source=ledger:<dir> tiers=deep:812,coarse:120,preview:0,transcribed:0 family_kernels=<family>:<hash>,...`.
- Each row ends in a tier comment (`# deep`). Both table parsers (`src/select/select.cc` and
  `sweep_to_table.py`) already strip a trailing `#` comment from a row, so neither changes.
  Provenance is then readable from `tuned/` without the UI.
- The op-level `kernels=<hash>` header stays as it is, for the configure-time staleness check. The
  per-family hashes go in a new `family_kernels=` header word.

**Record vocabulary.** A candidate's `status` is `ok`, `skipped`, `bad`, `error` or `eliminated`
(raced and dropped by the race). The race verifies every candidate it raced, eliminated ones
included: an eliminated candidate that fails verification is `bad` and is left out of its row; one
that passes carries the median of the reps it did time, which the table prints; one with no median
is left out of its row. A record is *stale* when
the winner's hash differs or its family is gone, *partly stale* when another candidate's hash
differs (a `skipped` one's is not compared, unless its reason is `dominated:`) or a current family was never timed; an empty ranking
has no winner to go stale.

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

1. **Ascending order.** A worker processes its cells in ascending order of *per-item footprint*
   (input bytes / batch, then total bytes): the SLM a launch asks for follows the matrix size, not
   the batch, so ascending total bytes is not enough. The driver restarts the worker before any
   cell whose per-item footprint is smaller than one it has already run. The first launch of a
   kernel that needs more than 48 KB therefore happens exactly as in a fresh process.
2. **Fresh-process audit.** Every run re-measures a sample of its cells in fresh one-cell processes
   (today's `--cell` path; the tier table gives the fraction). It compares feasibility per candidate
   and the winner. A mismatch marks that op and dtype *fresh-process only* in the ledger, and its
   remaining cells run one process per cell. The verdict is shown in the status matrix.

**As built** (Task 8). Each GPU's share of a round is sorted by per-item footprint; a refinement
round, or a smaller cell after a larger one, restarts the worker. The full guard (utilization too)
runs before a worker starts and, after a 1 s idle, every 60 s of worker time; between other cells
only foreign compute processes are checked. A worker cell with an `error` candidate may carry a
sticky CUDA error: the worker restarts and the cell is raced again in a fresh child, whose result
is recorded. When the fresh child reproduces the `error` in two consecutive cells, it is the
candidate's own error: that op and dtype stop sending the candidate to the worker and race it alone
in a fresh child per cell (errors can depend on the shape). The audit picks a cell when
`fnv1a64(run_id + key) % 1000 < audit_fraction * 1000`, and until an op and dtype has been audited
once, its next worker cell that did not fall back to a fresh child, whatever the hash picked. A
mismatch is a candidate usable in one process and refused or failing in the other (`eliminated`
against `bad` too: the race verifies eliminated candidates as well), or a different winner whose
worker winner is more than 10% slower in the fresh run (3 to 10% swaps are noise at preview's rep
counts). Every audited cell gets an `audit` ledger record; a mismatch appends a second `run` line
with `wm.<op>.<dtype>: fresh` (readers keep the last `run` line of a run id).

**No carve-out mismatch found** (threadripper02, GPU 1, 2026-10-07; `tune_race_gpu_tests`). For
each of potrf float (n=128, then n=110), potrf double (n=96, then n=78), posv float (n=128, then
n=110, nrhs=1) and trsm float (Left, NoTrans, order=q=128, then 104), all at batch 512, a worker
raced the larger cell, then the smaller one at or just below the 48 KB hole, and a fresh child
raced the smaller one. Every candidate's feasibility agreed. This is expected: potrf pads requests
out of the hole and its kernels carry no static shared (docs/perf/potrf.md#potrf-the-48-kb-launch-hole).
The test stays as the guard; the mismatch path itself is tested with injected verdicts in
`tune_tests`.

The first task of the implementation plan measures the real per-child overhead (process start,
SYCL and CUDA init, libbatchlas static init, JIT cache load) on both boxes. If it is small next to a
raced cell, the worker is dropped and the engine keeps one process per cell, which removes the
carve-out question.

## Engine: measured per-child overhead

Measured on threadripper02 (4x RTX PRO 6000 Blackwell, sm_120), GPU 1 only, the other three idle, one
measuring process at a time, 2026-10-07. The cell is potrf float `uplo=L,n=16,batch=8192`, run as
20 sequential `batchlas_tune_impl --cell` children with the driver's environment
(`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1`), after one throwaway child per configuration
to warm the JIT cache. Wall time is `time.perf_counter` around the child. Only this box was available;
the 4090 box was not measured.

| config | median s | p90 s |
| --- | --- | --- |
| (a) `time`, 1 arm (`tiny`), reps 1, warm 0 | 0.488 | 0.494 |
| (b) `probe`, 1 arm, no timing | 0.488 | 0.492 |
| (c) `time`, 6 float arms, reps 3, warm 0.2 | 1.802 | 1.809 |

(a) and (b) are equal to the millisecond, and the kernel time summed over (a)'s reps is 0.02 ms, so
the 0.49 s is process start, SYCL and CUDA init, library static init, JIT-cache load and the
verification launches. Time-to-first-kernel was not instrumented, so this start-up figure is derived
(wall time minus in-child kernel time), not read from a timestamp. It is also a lower bound: the JIT
cache was warm, so the old driver's cold `jit` child is not represented. The 20 children of (a) range
0.477-0.503 s (about 5%).
(c) adds 1.31 s of in-child work: 6 candidates x 0.2 s warm-up = 1.2 s, and the 18 timed reps sum to
0.7 ms. At this cell size the rep time is negligible and a raced cell is its warm-up top-up.

All ratios below were measured at this one small cell (n=16) only. The brief's rule compares the overhead with the measuring time of a raced preview cell. With the
preview race at 4 live candidates the work is about 4 x 0.2 s = 0.8 s plus verification, so the
child overhead of 0.49 s is about 60% of it; against (c)'s measured 1.31 s it is 37%. Both exceed
25%, and the cell is small only in rep time: a larger cell adds rep seconds, but the warm-up
dominates the old trsm data too (see "where the old tuner's time went"), so the ratio does not fall
below the threshold at median cell size.

Context from that section: the old driver ran three children per cell (one `jit` child, then two
`time` passes). That is 3 x 0.49 s = 1.5 s of pure start-up per cell, 6.4 ks (1.8 GPU-h) for the
~4400 cells of the full lattice (the coarse and deep measured-cell counts in the replay tables),
against 16 GPU-h in total for the old trsm float run. A preview run measures 2934 cells on trsm float,
so one child per cell pays 0.49 s x 2934 = 1.43 ks (0.40 GPU-h). The preview estimate of 0.570 GPU-h
assumes `--cell-overhead-s` = 0, so a per-cell process adds about 70% to it. A persistent worker
removes that.

Decision: WORKER=yes. The per-child overhead (0.49 s median) is 37-60% of a raced preview cell's
measuring time, above the 25% threshold. Task 8 runs.

## Engine: driver interface

- `batchlas_tune <op>[,<op>...|all] --tier preview|coarse|deep --dtype ... --devices ...` replaces the
  protocol flags (`--reps`, `--warm`, `--passes`, `--remeasure`, `--refine-ratio`). Those stay as
  expert overrides. A run that uses them is recorded with tier `custom`, ranked below preview,
  only when `--ledger DIR` is given explicitly, so a spot check never lands in the shared ledger.
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
  at most 0.2% of cells, coarse misranks at most 1% of cells, preview at most 5%, all by more than 3%.
- **Unit tests without a GPU.** Precedence (a coarse record never displaces a current deep one; a
  stale deep one is displaced), the gap-fill rule (a coarse row inside a deep bracket is never
  emitted), partly-stale re-race selection, lattice nesting, and `--check` reproducibility. Each has
  a deliberate break that turns exactly its own test red.
- **Racing guards.** A synthetic spec with fixed timing distributions, including a true near-tie, a
  candidate 4% slower than the leader (must be eliminated in deep), and one 2% slower (must be kept
  as a tie), straddles the tie margin in both directions.
- **Audit guard.** A test pins a candidate whose feasibility differs between fresh and warm
  processes (an SLM-heavy launch ordered after a larger one) and checks that the audit flags it.
- **End-to-end.** A preview run of potrf on one GPU finishes within the `--plan` estimate, and the tables it
  writes pass `tuned_tables_tests` and `--check`. A coarse run of trsm float on sm_120 agrees with
  the existing deep table within the replay's misranking bound.

## Engine: where the old tuner's time went

The replay's first cost estimate scaled the old run's GPU-hours by `reps_fraction`. That is wrong: the sum of every
`rep` record in `trsm.float.sm_120.jsonl` is 251 s (0.070 GPU-h), and in the double file 709 s (0.197 GPU-h),
against a 5 h 20 min run on 3 GPUs (about 16 GPU-h) for float (`benchmarks/results/tuning/README.md`). Over 99% of
the old tuner's time was per-arm warm-up (1.5 s per candidate per pass per cell) and process start-up, JIT and
verification, which do not shrink with fewer reps. The cost model of `tune_replay` therefore charges

`est_gpu_h = [measure_s + live_candidates x (warm_topup_s + verify_s) + cell_overhead_s per cell] / 3600`

with `measure_s` the rep time the race consumed, `warm_topup_s` = 0.2 from the tier parameters, `--verify-s` = 0.05
and `--cell-overhead-s` = 0 (a persistent worker, no per-cell process). With those assumptions, on trsm sm_120
(holdout replay of the tier defaults):

| tier | dtype | measured cells | measure_s | est_gpu_h |
| --- | --- | --- | --- | --- |
| preview | float | 2934 | 9.2 | 0.570 |
| preview | double | 2364 | 23.2 | 0.499 |
| coarse | float | 4427 | 32.0 | 0.856 |
| coarse | double | 4066 | 84.9 | 0.811 |
| deep | float | 4426 | 48.0 | 0.861 |
| deep | double | 4068 | 131.0 | 0.824 |

The estimate is dominated by the per-candidate term (about 0.25 s per candidate per measured cell), so a tier's
cost follows its measured cells and not its reps: coarse and deep cost about the same, and preview about two thirds
of them. A per-cell process (`--cell-overhead-s`) would add its start-up time to every measured cell and raise all
three. These numbers exclude that overhead and the audit processes, and are estimates for trsm sm_120 only.

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
candidate-reps the replay timed over those in the file. `measure_s` is the rep time the replay consumed and
`est_gpu_h` the cost model of "Engine: where the old tuner's time went".

The acceptance run scores the race on samples it did not see. In holdout mode (the default; `--no-holdout`
switches it off) the race uses only the pass-1 reps, in rep order, and every misrank and loss is scored
against the pass-2 median per candidate. Deep's reversed confirmation round cannot be modelled by this, so
deep is raced like coarse with its own reps and confidence.

Pass 1 and pass 2 disagree by more than 3% on about 1.5% of the cells (float; 1.15% double), so even an oracle that
times every cell of the raw file (no lattice, no bisection, whatever grid shrinking is asked for) and picks from all 16 pass-1 reps, with no elimination, scores 1.50% (float) and 1.15% (double) misrank
against the pass-2 reference. That is the noise floor of the reference, and no tuner can score below it. The
bounds measure what the tuner loses, so the verdict is the excess over the floor (`excess` = tier `table %` minus
`floor %`; the race excess is checked against the same bound and is in the JSON line): preview 5%, coarse 1%, deep 0.2%.

| tier | dtype | measured/cells | est_gpu_h | table % | floor % | excess | bound | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| preview | float | 2934/4452 | 0.570 | 4.47 | 1.50 | 2.96 | 5.0% | PASS |
| preview | double | 2364/4098 | 0.499 | 5.37 | 1.15 | 4.22 | 5.0% | PASS |
| coarse | float | 4427/4452 | 0.856 | 1.55 | 1.50 | 0.04 | 1.0% | PASS |
| coarse | double | 4066/4098 | 0.811 | 1.27 | 1.15 | 0.12 | 1.0% | PASS |
| deep | float | 4426/4452 | 0.861 | 1.59 | 1.50 | 0.09 | 0.2% | PASS |
| deep | double | 4068/4098 | 0.824 | 1.22 | 1.15 | 0.07 | 0.2% | PASS |

Preview loses 3.0 and 4.2 points over the oracle; coarse and deep lose 0.04 to 0.12 points, which is under their 1% and
0.2% bounds. Preview double is the one row close to its bound.

Without holdout, the race and the reference share the same reps, which flatters the result (same-sample
table, same tier values):

| tier | dtype | measured/cells | reps | race % | table % | lattice % | mean | p99 | max | tw | bound | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| preview | float | 2818/4452 | 0.0730 | 0.25 | 3.59 | 3.59 | 0.0190 | 0.800 | 2.11 | 0.0118 | 5.0% | PASS |
| preview | double | 2363/4098 | 0.0703 | 0.00 | 4.34 | 3.85 | 0.0186 | 0.604 | 3.27 | 0.0046 | 5.0% | PASS |
| coarse | float | 4413/4452 | 0.1583 | 0.11 | 0.11 | 0.06 | 0.0006 | 0.020 | 0.03 | 0.0003 | 1.0% | PASS |
| coarse | double | 4094/4098 | 0.1518 | 0.02 | 0.05 | 0.00 | 0.0004 | 0.015 | 0.08 | 0.0002 | 1.0% | PASS |
| deep | float | 4417/4452 | 0.2375 | 0.11 | 0.16 | 0.11 | 0.0006 | 0.022 | 0.06 | 0.0004 | 0.2% | PASS |
| deep | double | 4095/4098 | 0.2248 | 0.02 | 0.02 | 0.00 | 0.0004 | 0.015 | 0.03 | 0.0001 | 0.2% | PASS |


The sections below record how these values were chosen. Rows named preview in those experiment tables use
the preview race parameters (3/6 reps, confidence 0.80) with the stride and bisection stated in the row,
not the final preview tier.

The experiment tables below are all same-sample (no holdout), so they measure the lattice and the bisection,
not the race. Changing confidence or reps moves `table_misrank` by under one point; the table bound is set by the
lattice and the bisection ratio. The trade-off is easier to read with the loss of the chosen candidate,
`exhaustive(chosen) / best - 1`, over all cells, where chosen is the `select::choose` pick: `mean_loss`
and the percentiles cover the cells where some entry of the nearest row can run, `unrunnable` counts
the rest, `time_weighted_loss` is the summed chosen time over the summed best time minus one, and
`table_misrank_lattice` repeats `table_misrank` over the round-0 cells only, which removes the raw
file's refinement points at winner flips. The tables carry no GPU-hour column, because the measured
time of a tier is dominated by costs that do not scale with reps (next section). Each row uses the named
tier's race parameters; the config column changes only the
lattice stride and the bisection ratio.

| config | dtype | cells_measured | reps_fraction | table_misrank % | table_misrank_lattice % | mean_loss | p99_loss | max_loss | time_weighted_loss | unrunnable |
|---|---|---|---|---|---|---|---|---|---|---|
| preview default | float | 132 | 0.0031 | 19.50 | 17.76 | 0.0819 | 1.283 | 2.40 | 0.0592 | 0 |
| preview default | double | 124 | 0.0030 | 14.20 | 11.06 | 0.0581 | 0.999 | 2.84 | 0.0190 | 0 |
| preview stride 2 | float | 616 | 0.0150 | 11.48 | 8.91 | 0.0297 | 0.614 | 1.61 | 0.0205 | 0 |
| preview stride 2 | double | 585 | 0.0150 | 10.49 | 5.62 | 0.0367 | 0.770 | 1.92 | 0.0110 | 0 |
| preview stride 1, no refine | float | 3480 | 0.0867 | 6.36 | 0.17 | 0.0110 | 0.239 | 0.77 | 0.0031 | 0 |
| preview stride 1, no refine | double | 3219 | 0.0853 | 6.05 | 0.00 | 0.0209 | 0.601 | 2.08 | 0.0080 | 0 |
| preview stride 1, refine 1.25 | float | 4021 | 0.1005 | 2.52 | 0.17 | 0.0035 | 0.115 | 0.77 | 0.0025 | 0 |
| preview stride 1, refine 1.25 | double | 3725 | 0.0991 | 1.42 | 0.00 | 0.0037 | 0.052 | 2.08 | 0.0027 | 0 |
| preview stride 1, refine 1.1 | float | 4390 | 0.1074 | 0.29 | 0.17 | 0.0006 | 0.021 | 0.08 | 0.0004 | 0 |
| preview stride 1, refine 1.1 | double | 4088 | 0.1062 | 0.02 | 0.00 | 0.0004 | 0.014 | 0.08 | 0.0002 | 0 |
| preview stride 2, refine 1.25 | float | 820 | 0.0193 | 7.88 | 5.86 | 0.0192 | 0.542 | 1.86 | 0.0104 | 0 |
| preview stride 2, refine 1.25 | double | 712 | 0.0189 | 9.66 | 4.32 | 0.0373 | 0.790 | 1.95 | 0.0083 | 0 |
| coarse default | float | 810 | 0.0279 | 8.33 | 6.38 | 0.0210 | 0.599 | 1.86 | 0.0119 | 0 |
| coarse default | double | 711 | 0.0272 | 9.83 | 4.38 | 0.0375 | 0.790 | 1.95 | 0.0083 | 0 |
| coarse stride 1, refine 1.25 | float | 4025 | 0.1476 | 2.54 | 0.06 | 0.0036 | 0.118 | 0.77 | 0.0024 | 0 |
| coarse stride 1, refine 1.25 | double | 3727 | 0.1416 | 1.49 | 0.00 | 0.0038 | 0.055 | 2.08 | 0.0027 | 0 |
| coarse stride 2, refine 1.1 | float | 813 | 0.0280 | 8.31 | 6.38 | 0.0210 | 0.599 | 1.86 | 0.0115 | 0 |
| coarse stride 2, refine 1.1 | double | 715 | 0.0273 | 9.83 | 4.38 | 0.0375 | 0.790 | 1.95 | 0.0083 | 0 |
| coarse stride 1, refine 1.1 | float | 4413 | 0.1583 | 0.11 | 0.06 | 0.0006 | 0.020 | 0.03 | 0.0003 | 0 |
| coarse stride 1, refine 1.1 | double | 4094 | 0.1518 | 0.05 | 0.00 | 0.0004 | 0.015 | 0.08 | 0.0002 | 0 |
| deep default | float | 4417 | 0.2375 | 0.16 | 0.11 | 0.0006 | 0.022 | 0.06 | 0.0004 | 0 |
| deep default | double | 4095 | 0.2248 | 0.02 | 0.00 | 0.0004 | 0.015 | 0.03 | 0.0001 | 0 |

Reading it: with the fallback, `unrunnable` is 0 in every row, so the earlier unrunnable choices were
cells where the first entry could not run and the next one could. The full lattice is what matters:
at stride 1 with the preview race parameters, bisecting to 1.1 gives 0.29% and 0.02% on 4390 and 4088
measured cells, against 0.11% and 0.05% on 4413 and 4094 with the coarse race parameters; bisecting to
1.25 gives 2.5% and 1.4%, and no bisection 6.4% and 6.1%. The sparse rows (stride 2 and 4) stay at 8 to
20% whatever the ratio, because the replay drops the `q` and `batch` midpoints the raw file lacks; it
cannot say what bisection on those axes would buy. The mean and time-weighted losses are small in every
row (at most 8.2% and 5.9%, both preview default float), but the p99 loss of the sparse rows is 0.5 to
1.3, so they are right on average and badly wrong at a few percent of shapes.

## Engine: replay of shrunken trsm grids

Instead of sparse tiers, the grids themselves can shrink. `tune_replay --axis-stride name=k` keeps every
k-th value and the last of one axis in the starting lattice, and `--axis-keep name=v1:v2` keeps only
the listed values; the other axes stay full. The raw files hold the same axes for both dtypes
(`--print-axes`): order 1 2 4 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024 (log, weight 2), q 1 2 4
8 16 32 64 128 256 512 1024 4096 (log), batch 128 512 2048 8192 32768 (log), plus side L R and trans N T.
Every row uses the preview race parameters, stride 1 and bisection to 1.1. Bisection refills only where
the raw file has points, which is `order` (the raw sweep refined only along it), so shrinking `order`
is partly repaired while shrinking `q` or `batch` is not. The metrics are over all raw cells, so a cell
at a dropped value pays for the dropped measurement. Same machine and data as above.

| config | dtype | measured | table_misrank % | lattice % | mean_loss | p99_loss | max_loss | tw_loss |
|---|---|---|---|---|---|---|---|---|
| baseline (preview s1 r1.1) | float | 4390 | 0.29 | 0.17 | 0.0006 | 0.021 | 0.08 | 0.0004 |
| baseline (preview s1 r1.1) | double | 4088 | 0.02 | 0.00 | 0.0004 | 0.014 | 0.08 | 0.0002 |
| batch stride 2 | float | 2920 | 4.27 | 3.39 | 0.0162 | 0.615 | 1.86 | 0.0080 |
| batch stride 2 | double | 2546 | 4.76 | 3.76 | 0.0250 | 0.757 | 2.36 | 0.0069 |
| batch keep 8192:32768 | float | 1356 | 11.32 | 10.11 | 0.0263 | 0.540 | 0.93 | 0.0188 |
| batch keep 8192:32768 | double | 1226 | 6.83 | 5.03 | 0.0230 | 0.799 | 1.27 | 0.0041 |
| batch keep 128:32768 | float | 2118 | 7.86 | 7.30 | 0.0299 | 0.864 | 1.86 | 0.0193 |
| batch keep 128:32768 | double | 1789 | 6.86 | 5.28 | 0.0329 | 0.837 | 2.36 | 0.0062 |
| batch keep 2048 | float | 916 | 13.52 | 11.24 | 0.0314 | 0.549 | 0.93 | 0.0200 |
| batch keep 2048 | double | 814 | 7.22 | 4.16 | 0.0192 | 0.421 | 1.73 | 0.0036 |
| q stride 2 | float | 2964 | 1.50 | 0.95 | 0.0039 | 0.063 | 1.46 | 0.0007 |
| q stride 2 | double | 2435 | 2.98 | 1.03 | 0.0074 | 0.279 | 0.78 | 0.0021 |
| q stride 3 | float | 1728 | 6.42 | 5.89 | 0.0122 | 0.325 | 0.91 | 0.0103 |
| q stride 3 | double | 1696 | 6.12 | 5.65 | 0.0207 | 0.553 | 1.04 | 0.0137 |
| order stride 2 | float | 1911 | 4.65 | 2.41 | 0.0066 | 0.142 | 1.06 | 0.0026 |
| order stride 2 | double | 1837 | 6.93 | 2.05 | 0.0244 | 0.755 | 0.91 | 0.0045 |
| order stride 4 | float | 1348 | 6.45 | 3.85 | 0.0128 | 0.422 | 1.06 | 0.0040 |
| order stride 4 | double | 1269 | 6.34 | 4.57 | 0.0176 | 0.430 | 4.00 | 0.0045 |
| q2 + batch2 | float | 1978 | 6.04 | 5.09 | 0.0212 | 0.704 | 1.86 | 0.0086 |
| q2 + batch2 | double | 1530 | 5.66 | 3.42 | 0.0203 | 0.639 | 2.19 | 0.0050 |
| q2 + batch2 + order2 | float | 824 | 7.84 | 5.86 | 0.0191 | 0.542 | 1.86 | 0.0100 |
| q2 + batch2 + order2 | double | 716 | 9.66 | 4.32 | 0.0373 | 0.790 | 1.95 | 0.0083 |
| q2 + order2 | float | 1254 | 5.05 | 2.87 | 0.0085 | 0.201 | 1.06 | 0.0028 |
| q2 + order2 | double | 1120 | 8.17 | 2.67 | 0.0279 | 0.748 | 0.91 | 0.0051 |
| q3 + batch2 + order2 | float | 491 | 10.53 | 9.68 | 0.0277 | 0.561 | 2.17 | 0.0207 |
| q3 + batch2 + order2 | double | 447 | 12.71 | 8.45 | 0.0571 | 0.859 | 5.12 | 0.0164 |
| q2 + batch keep 2048 + order2 | float | 243 | 16.85 | 12.90 | 0.0399 | 0.594 | 1.13 | 0.0247 |
| q2 + batch keep 2048 + order2 | double | 212 | 11.05 | 5.25 | 0.0382 | 0.797 | 1.73 | 0.0066 |

Reading it: batch is the costliest axis to shrink, because the winner flips along it (stride 2 already
gives 4.3% and 4.8%; one batch value gives 13.5% and 7.2%). q stride 2 is the cheapest single cut
(1.5% float, 3.0% double, with 68% and 60% of the baseline's measured cells), and order stride 2 gives 4.7% and
6.9%. Combinations add their misranks rather than their savings: q stride 2 with batch stride 2 and
order stride 2 measures 824 and 716 cells and misranks 7.8% and 9.7%, the same as the sparse coarse
tier. No shrunken grid in this sweep meets the 1% table bound; the baseline does, on 4390 and 4088 measured cells.

## Engine: replay of index-space refinement

`refine_all_axes` has a second mode (`RefineOpts`, `tune_replay --refine-mode index`). Between two
measured neighbours at positions i < j of the axis's full value list with j - i > 1 it proposes the value
at (i + j) / 2, so every proposal is on the `choice.hh` lattice; adjacent neighbours fall back to the
geometric midpoint at the tier's ratio. `--refine-margin m` also refines a bracket whose two winners
agree when, at either end, the runner-up's raced median is within m of the winner's (the replay passes
the per-cell medians in `RefineOpts::cells`). The tables below predate the
[convergence rules](#engine-refinement-convergence-rules). Exact axes are never bisected. Every row uses the preview race
parameters, bisection ratio 1.1 and the full axes; `s2` and `s4` are the starting lattice stride on every
log axis. Same machine, data and metrics as above.

| config | dtype | measured | table_misrank % | lattice % | mean_loss | p99_loss | max_loss | tw_loss |
|---|---|---|---|---|---|---|---|---|
| baseline (s1 geometric 1.1) | float | 4390 | 0.29 | 0.17 | 0.0006 | 0.021 | 0.08 | 0.0004 |
| baseline (s1 geometric 1.1) | double | 4088 | 0.02 | 0.00 | 0.0004 | 0.014 | 0.08 | 0.0002 |
| s2 index | float | 1412 | 6.72 | 6.38 | 0.0281 | 0.938 | 2.18 | 0.0112 |
| s2 index | double | 1096 | 5.78 | 3.60 | 0.0202 | 0.639 | 1.95 | 0.0062 |
| s2 index margin 0.05 | float | 2542 | 4.25 | 4.45 | 0.0219 | 0.785 | 2.17 | 0.0126 |
| s2 index margin 0.05 | double | 2214 | 5.22 | 4.78 | 0.0222 | 0.733 | 3.27 | 0.0059 |
| s2 index margin 0.10 | float | 2818 | 3.59 | 3.59 | 0.0190 | 0.800 | 2.11 | 0.0118 |
| s2 index margin 0.10 | double | 2363 | 4.34 | 3.85 | 0.0186 | 0.604 | 3.27 | 0.0046 |
| s4 index | float | 612 | 13.32 | 12.67 | 0.0579 | 1.242 | 2.99 | 0.0526 |
| s4 index | double | 501 | 10.42 | 7.58 | 0.0487 | 1.037 | 3.10 | 0.0075 |
| s4 index margin 0.05 | float | 1566 | 11.10 | 10.89 | 0.0598 | 1.404 | 2.77 | 0.0354 |
| s4 index margin 0.05 | double | 1402 | 6.76 | 6.18 | 0.0242 | 0.716 | 1.41 | 0.0059 |
| s4 index margin 0.10 | float | 1812 | 9.07 | 9.17 | 0.0477 | 1.304 | 2.76 | 0.0348 |
| s4 index margin 0.10 | double | 1434 | 6.81 | 6.31 | 0.0247 | 0.716 | 1.31 | 0.0070 |

Reading it: index refinement at stride 2 refills the lattice with 1412 and 1096 measured cells and misranks 6.7%
and 5.8%, against 7.9% and 9.7% for the geometric stride-2 rows above. The margin trigger buys a few
points more at about twice the cells (margin 0.10: 3.6% and 4.3% on 2818 and 2363 cells). Stride 4 stays
at 7 to 13%. None of these comes near the 1% table bound that the full lattice meets
with 4390 and 4088 cells, and p99 loss stays near 0.6 to 1.4 in every sparse row: the replay can bisect only along `order`
in the raw files, so the q and batch refills here are limited to cells the raw file holds, and a flip
between two lattice points on those axes is still left to the nearest-cell rule.

## Engine: refinement convergence rules

> **Status:** as built 2026-10-07 (Task 8b). Replay on the sm_120 trsm raw files of threadripper02;
> GPU run on threadripper02 GPU 1 (RTX PRO 6000, sm_120).

**The problem.** A preview potrf float run planned 108 cells and measured 2029 over 16 rounds
(47, 76, 107, 186, 345, 353, 383, 212, 119, 62, then 49 over rounds 11 to 15). Of the 1939
refinement cells, 908 came from winner flips and 1031 from the 0.10 margin trigger. Many flips were
near-tie winners alternating between neighbours (`lpanel:panel=8` against `lpanel:panel=16` within
3%), batch was bisected down to values such as 431, and a 4x2 starting grid grew to 129 cells. The
plan and the estimate counted the starting lattice only.

**The rules** (`refine_all_axes` in `tools/tune/grid.cc`, the same code in the driver and in
`tune_replay`; each cell's raced medians arrive as a `RefineCell`):

1. A bracket is a winner flip when the two winners differ and the flip is decisive at *either*
   end: there, the other end's winner is more than the 3% tie slower than the local winner. A
   winner with no median at an end counts as decisively slower there only when it is not runnable
   there or was eliminated. Within the tie at both ends (near-tie alternation) is no flip.
2. The margin trigger (preview, 10%) applies only to brackets whose two ends are both starting
   lattice (round 0) cells at the running tier, measured now or stored by an earlier run, so a
   refinement midpoint never re-triggers it and a resumed run still hedges.
3. `batch` is never bisected below its lattice spacing: it refills its own axis values (index
   mode in every tier) and never takes a geometric midpoint.
4. Refinement cells per op and dtype are at most `refine_cap_factor` x round-0 cells (preview 3.0,
   coarse 1.0, deep 2.0; `TierParams`). Both counts are the ledger's current records at the
   running tier, overlaid by this run's: round-0 records are the base, and refined (round > 0)
   records count against the cap. A run stopped by Ctrl-C or `--budget` after its lattice
   therefore resumes refinement on the next run, and the total over runs stays within the cap.
   Midpoints come out flips first, then margin hedges; past
   the cap the first ones run, refinement stops and the run reports it (`refine_cap` progress event,
   `refinement cap hit` line). `tune_replay` applies the same cap.
5. The estimate counts refinement: ratio x the lattice cells measured now, at their mean estimate,
   where ratio is this ledger's refined / round-0 records at the tier (at most the cap factor), else
   cap factor x 0.5. `--plan` prints the lattice-only and the with-refinement estimate, the `plan`
   event carries `refine_cells`, `est_refine_s` and `est_total_s`, and the run ends with a summary
   line per op and dtype.

**Replay re-validation** (holdout acceptance, `tune_replay --tier T` with the tier defaults; excess
over the noise floor; measured = lattice + refinement cells the raw file holds). "Before" is the
tuner before these rules; "as built" is rules 1 to 5 above with preview cap 3.0:

| tier | dtype | bound | before: measured | before: excess race / table | as built: measured (lattice + refined) | as built: excess race / table | verdict |
|---|---|---|---|---|---|---|---|
| preview | float | 5% | 2934 | 0.51% / 2.96% | 2197 (616 + 1581) | 0.41% / 3.89% | PASS |
| coarse | float | 1% | 4427 | 0.03% / 0.04% | 4427 (3480 + 947) | 0.03% / 0.04% | PASS |
| deep | float | 0.2% | 4426 | 0.01% / 0.09% | 4427 (3480 + 947) | 0.01% / 0.09% | PASS |
| preview | double | 5% | 2364 | 0.59% / 4.22% | 1763 (585 + 1178) | 0.21% / 4.25% | PASS |
| coarse | double | 1% | 4066 | 0.03% / 0.12% | 4066 (3219 + 847) | 0.03% / 0.12% | PASS |
| deep | double | 0.2% | 4068 | -0.02% / 0.07% | 4068 (3219 + 849) | -0.02% / 0.07% | PASS |

No cap is hit in the replay. Coarse and deep measure exactly what the unrestricted flip rule measured:
the trsm files hold no bracket whose two winners are within the tie at both ends. Preview measures 25%
fewer cells than before.

Why "either end" and cap 3.0: a first version required rule 1 at *both* ends with a preview cap
of 1.0, and every tier failed:

| rule 1 / preview cap | tier | dtype | measured (lattice + refined, capped) | excess table | verdict |
|---|---|---|---|---|---|
| both ends / 1.0 | preview | float / double | 1232 (135 dropped) / 1170 (62 dropped) | 7.52% / 7.32% | FAIL / FAIL |
| both ends | coarse | float / double | 3859 / 3619 | 3.75% / 1.73% | FAIL / FAIL |
| both ends | deep | float / double | 3849 / 3621 | 3.91% / 1.68% | FAIL / FAIL |
| both ends / 3.0 | preview | float / double | 1702 / 1495 | 6.49% / 4.73% | FAIL / PASS |
| either end / 1.0 | preview | float / double | 1232 (135 dropped) / 1170 (62 dropped) | 7.84% / 7.32% | FAIL / FAIL |
| either end / 2.0 | preview | float / double | 1848 (78 dropped) / 1755 (2 dropped) | 5.75% / 4.32% | FAIL / PASS |

The trsm flips are one-sided: their crossover sits near one end of the bracket, where the other
end's winner is within the tie, and far behind at the far end. Requiring both ends dropped those
brackets, and the cells between read the wrong neighbour. The replay cannot show the growth these
rules exist to stop: it measures only midpoints the raw file holds (3915 and 3534 preview midpoints
were unavailable before), so its refinement counts are lower bounds of a GPU run's.

**GPU run** (`batchlas_tune potrf --tier preview --dtype float --devices 1`, persistent worker):

| | planned | measured | rounds (cells per round) | wall |
|---|---|---|---|---|
| before the rules (Task 8) | 108 lattice | 2029 | 16 (47, 76, 107, ... 49 over rounds 11 to 15) | 2388 s |
| as built | 90 lattice + ~135 refinement (est 0.05 h lattice, 0.13 h with refinement) | 90 + 90 = 180 | 7 (90, 46, 23, 12, 6, 2, 1) | 219 s |

The 108 planned cells include 18 `skip:cap` cells, so the cap base is 90 and the preview cap 270
refinement cells. Refinement converged at 90 without hitting it, well within (1 + 3.0) x 90.
There were no worker restarts and 2 audits. The estimate is high by about 2x (453 s planned, 219 s
wall), mostly from the no-history default (cap x 0.5 = 1.5 refinement cells per lattice cell against
the 1.0 measured). The next run's history ratio replaces it. The first version (both ends, cap 1.0)
measured 90 + 79 = 169 cells in 205 s.

## Engine: end-to-end validation on sm_120

> **Status:** measured 2026-10-07 on threadripper02 (4x RTX PRO 6000 Blackwell, sm_120), all GPUs idle,
> one measuring process at a time. Ledgers and tables went to a scratch directory, not to `tuned/`
> or `benchmarks/results/`: the imported trsm data predates kernel edits, and a preview run must not
> replace the converted potrf tables. Shipping ledger tables needs coarse or deep runs and a
> maintainer decision.

**Import round trip.** `batchlas_tune --import-raw` of `benchmarks/results/tuning/trsm.{float,double}.sm_120.jsonl`
writes deep runs whose hashes are `legacy:c923160f` (the kernels changed since), so an ordinary
`sweep_to_table.py --ledger` writes an empty table: every record is stale. The diagnostic
`--assume-current` judges records by their stored hashes instead. With it, the generated tables equal
the shipped `tuned/trsm.{float,double}.sm_120.txt` in every row: 4452 and 4098 rows, 0 rows missing
or extra, **0 ranking differences**, and also 0 time differences. Two importer/generator bugs
turned up and are fixed (`e3ca37f9`). A tiered trsm cell's key carries the hidden grid axes `uplo`
and `diag`, so the generator refused every coarse record. And an import keyed by the table keys
alone could never match a tiered cell. The importer now keys by the spec's grid axes, and the
generator projects through the op's `tuner_key`, which refuses a varied hidden axis.

**Preview, potrf, 4 dtypes, GPU 1** (`--tier preview --dtype float,double,cfloat,cdouble`):

| dtype | planned cells (measure + skip:cap) | lattice measured | refinement planned / measured | audits (all ok) |
|---|---|---|---|---|
| float | 108 (90 + 18) | 90 | ~135 / 83 | 7 |
| double | 108 (88 + 20) | 88 | ~132 / 152 | 11 |
| cfloat | 108 (88 + 20) | 88 | ~132 / 78 | 8 |
| cdouble | 108 (82 + 26) | 82 | ~123 / 93 | 8 |

That is 754 cells in 906 s wall for the whole run, against a plan estimate of 1593 s (1.76x high).
There were 0 worker restarts, 0 audit mismatches and no refinement cap hit. The four tables (173,
240, 166 and 175 rows, all `# preview`) pass the converter's ports of the loader rules (keys,
candidates, timed/untimed rows) and re-derive byte for byte from their ledgers.

**Coarse, trsm float, GPUs 0-3, scored against the exhaustive sweep.** The plan put 3480 lattice cells
(840 `skip:cap`) at 1.48 h, or 2.22 h with refinement. That estimate is summed over GPUs, so about 33 min
of wall time on 4 GPUs. The run measured 3480 lattice and 2661 refinement cells (22 rounds; the cap was
3480) in **1529 s wall**. That is 1.3x under the per-GPU estimate, and 1.7 GPU-hours at most,
against the 16 GPU-hours of the exhaustive sweep. All 127 audits were ok, and there were 0 restarts.
Scoring: for each of the 4452 raw cells, take the coarse table's nearest row (`nearest()`), then its first
entry the raw file ran at that cell, and count a misrank when that entry is more than 3% slower than
the raw best. The reference is the raw file's mean of pass medians per candidate, the replay's
`exhaustive`.

| table | misrank | mean loss | p99 loss | max loss | misranked at a cell the run measured |
|---|---|---|---|---|---|
| coarse ledger (6141 rows) | 26 / 4452 = 0.58% | 0.08% | 2.5% | 12.6% | 21 of 26 |
| shipped table (control) | 0 | 0.07% | 2.5% | 3.0% | — |

That is below the replay's 1.50% noise floor. The floor scores pass 1 against pass 2, while this
reference averages both, so the two are not the same measure. The kernels changed between the sweep
and this run, and the run is a new sample. So the 0.58% is data, not a pass/fail verdict. The worst
cells are `blocked` against `vendor` near order 100-180 (up to 12.6%), and 21 of the 26 are cells the
coarse run timed itself: the race and the two-day-old sweep disagree there, rather than the gap fill
being wrong.

**Resume.** Rerunning the preview potrf float command: `--plan` shows 90 `skip:current` and 18
`skip:cap` (0 to measure), and the run exits in 1.3 s without writing a run file.

**Staleness.** One comment line appended to `src/sycl/trsm_sg_left.cc` (family `sg_left`), then
`--tier coarse --plan`. This runs against the coarse ledger with its stored hashes rewritten to
today's values: adding `<op>.cc` to every family changed every hash, and no hashed source file
changed since the run. Every edit was reverted, with an empty `git diff`.

| cells | count | why |
|---|---|---|
| measure (full) | 711 | `sg_left` was the stored winner, so the record is stale |
| `partial:sg_left` | 223 | side L, order <= 32: `sg_left` ran but lost; re-race it plus the stored winner and runner-up |
| `skip:current` | 2546 | `sg_left` was `skipped` there (side R, or order > 32), so its hash is not compared |
| `skip:cap` | 840 | |

A first version compared every candidate's hash, `skipped` ones included. It planned 711 measure,
2769 partial and 0 current: 1612 of those cells were above order 32, where `sg_left` cannot run,
which is about 33 single-GPU minutes of useless re-races. Control edits:

- `src/ops/trsm/trsm.cc` (`can_run`): all 3480 cells measure.
- `src/extensions/potrf_lpanel.cc` on the potrf float preview ledger: 21 measure, 22
  `partial:lpanel` and 47 current, every uplo=U cell among them (`lpanel` cannot run on Upper).
  Before the rule it was 21 measure, 69 partial and 0 current.

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

**Gross losers.** After every round, from the first on and whatever `min_reps` says, the race
eliminates an arm whose median paired ratio to the leader exceeds 4.0 (`kGrossLoserRatio`,
`race.hh`). It is recorded `eliminated` with the median of the rounds it ran. geqrf's vendor
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

What stays is the in-race gross-loser rule (ratio > 4 after one round), which only uses data from
the same cell. A ledger record that still holds a `dominated:` skip is *partly stale*: the skipped
arm's family counts as stale whatever its hash, so the next run re-races exactly those arms against
the stored winner and runner-up at the stored tier and writes a merged record (`stale_candidates`,
`ledger.cc`, and `stale_families` in `sweep_to_table.py`; shared fixture
`tests/data/ledger_dominated`).

## Engine: guard retries a failed nvidia-smi query

The sm_120 deep run died with `cannot query GPU 3 with nvidia-smi` while the maintainer reset
GPU 0: `nvidia-smi` fails for every GPU for a moment during a reset. A failed guard query
(`--query-compute-apps` or the utilization sample) is now retried after 5 s, doubling to 60 s,
until `--guard-wait` (default 300 s) of retrying has passed, with one line on stderr per retry;
only then does the run stop (`query_with_backoff`, `tune_core.hh`).

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
(@ref md_docs_2design_2known-defects "known defects", the LOBPCG entry), but nothing in the tuner
could have caught such a defect if it had been real.

**The rule.** For each (op, dtype) in a tiered run, the driver asks every candidate's *first verified
cell* (status `ok`, or `eliminated` after passing verification) for one extra verification at
`ld_pad + 3` (`kLdAuditPad`, `tools/tune/spec.hh`). The check runs in the same worker or child,
straight after the timed cell, as a `verify`-mode cell, which is a fresh problem built by the spec
at the padded `ld`. `LdAuditBook` (`schedule.hh`) records which arms have a verdict. A candidate
that the cell did not verify stays wanted for its next cell.

**On a failure.** If the padded verification comes back `bad` or `error`, or throws, the arm is
`bad` in that cell's record with the reason `ld audit: ld_pad +3: ...`, so it is never ranked
there. The driver prints `ld audit FAILED`, emits an `ld_audit` progress event, and the run summary
gives one line per (op, dtype) with passed, skipped and failed counts and lists the failures (event
`ld_audit_summary`). A pin that refuses the padded shape counts as `skipped`. The carve-out audit
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
budgeted, so `--budget 1e-9` runs exactly the named cells and starts no refinement. A named cell
is measured past `--max-dim`, because the deep run measured some cells before the cap existed,
but not past `--cap-gib`. A key whose fields do not match the op's, or an op and dtype the run
does not tune, stops the run. The newer record wins in `best_records` (same tier, later date or
run id), so the tables pick up the re-measure with no other change. Host tests: `TuneRemeasure.*`
(a deliberate break of the planner or of the round-0 append turns them red).

## Tiered tuning: open risks

- Racing assumes timing noise is roughly stationary within a cell. Clock ramps after idle gaps
  break that. The worker's warm start and between-cell top-up are the defence, and the replay
  cannot test it, because the raw files come from warm, back-to-back runs.
- If the per-child overhead turns out large and the audit fails often, the speedup has to come
  mostly from racing and the adaptive grid. The replay quantifies how much they provide.
- Deferred follow-ups, not implemented: per-rep `<run id>.reps.jsonl` files for deep runs, and
  guard settings and tolerated foreign pids in `run` records.
- Budgets are for one GPU and assume a quiet box. On the shared 4090 box, the guard's waits come on
  top.
