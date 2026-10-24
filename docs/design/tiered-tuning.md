# Tiered tuning {#design_tiered_tuning}

> **Status:** current · checked 2026-10-21

`batchlas_tune` fills the selection tables in tiers (preview, calibrate, coarse, deep) with one
algorithm per cell, a per-(op, device) ledger in which a cheaper run never displaces a better
current result, and one persistent worker per GPU. Usage and flags: @ref tune_tool_readme. Ledger,
table, tier and staleness rules: @ref design_tuning_data (normative). The tables feed
@ref design_flat_selection §6.

## Tiered tuning: components and state

| # | Component | State |
| --- | --- | --- |
| 1 | Engine: tiers, lean race, flip-frontier planning, calibrate, persistent worker, audits, ledger, table generation | implemented (`tools/tune/`) |
| 2 | Coverage: one spec per routed op | implemented, all 19 ops (`tools/tune/<op>_spec.cc`) |
| 2 | Constants tuner on the engine (`tuning_params.hh`) | **open** ([below](#coverage-interface-for-sub-project-2)) |
| 3 | benchviz Tuning tab | built (`benchmarks/benchviz/tuning.py`) |

- Shipped measured tables: `tuned/*.sm_120.txt` from the sm_120 deep run and later deep runs into
  the same ledgers. sm_89 and cpu tables are untimed (@ref design_tuning_data, "Shipped set").
- Cost before the engine (2026-10-07): trsm float on sm_120 took about 16 GPU-hours (3480 cells,
  5 h 20 min on 3 GPUs, three cold processes per cell with 1.5 s warm-up per candidate); gemm
  float was estimated at about 100 GPU-hours. 4 of 19 ops had a spec; 16 of 144 tables were
  measured.

## Tiered tuning: decisions

| Question | Decision |
| --- | --- |
| Main purpose | Initial fill of every table for a machine; incremental retune and exploration second. |
| Speed versus rigor | The user picks a tier per run; `--plan` prices it; more GPUs shard the work. |
| Grid reduction | Stride-2 start plus flip-frontier rounds; gemm, geqrf, getri, orgqr and spmm keep the full coarse and deep lattice ("Engine: round planning by flip frontier"). |
| Precedence | Higher fidelity wins per cell. A lower tier fills gaps and replaces stale results, never current ones. |
| Op scope | All 19 routed ops. The `tuning_params.hh` constants under the same tiers and ledger: open. |
| Result store | `benchmarks/results/tuning/ledger/` under Git LFS, so precedence holds across machines and clones. |
| Raw per-rep timings | Not stored (deferred, [open risks](#open-risks)). |
| UI | A Tuning tab in `benchviz serve`, driving `batchlas_tune`. |
| Worker | One persistent worker per GPU plus a fresh-process audit ([persistent workers](#engine-persistent-workers-and-the-carve-out-audit)). |
| New machine | `--tier calibrate` probes the nearest shipped table and measures where the device disagrees ("Engine: calibrate tier from a borrowed table"). |

## Engine: tiers and the per-cell algorithm

Every tier runs the same algorithm, so preview results are low-fidelity data of the same kind as
deep results. Parameters: `params()` in `tools/tune/tier.hh`.

| Parameter | preview | coarse | deep |
| --- | --- | --- | --- |
| Starting lattice | every 2nd point of each `choice.hh` axis (batch: the CU-aligned ladder), ends kept, then flip-frontier rounds | as preview; stride-1 ops (gemm, geqrf, getri, orgqr, spmm): full lattice | as coarse |
| Bisection | geometric to hi/lo < 1.1 between adjacent grid values (grid midpoints are the frontier's); stride-1 ops: index space until adjacent, then geometric, plus a 10% runner-up margin | as preview; stride-1 ops: geometric to hi/lo < 1.1 | as coarse |
| Refinement cap (x round-0 cells, per op and dtype; stride-1 ops only) | 3.0 | 1.0 | 2.0 |
| Race reps floor per candidate (cap: 50 ms of the slowest arm, at most 16) | 3 | 4 | 6 |
| Elimination confidence | 0.80 | 0.90 | 0.98, plus a reversed confirmation round |
| Fresh-process audit sample | 2% | 2% | 1% |

- `calibrate` races at coarse quality (reps floor 4, confidence 0.90) on a stride-4 probe
  lattice against a prior table ("Engine: calibrate tier from a borrowed table").
- The tie margin is 3% in every tier. Lattices nest (preview ⊂ coarse ⊂ deep); a grid that is not a
  lattice is subsampled by a stable hash of the cell key, so subsets still nest.

Per cell:

1. **Probe.** Each candidate's pin is checked with the `*_buffer_size` probe under
   `select::ScopedPin`, or one untimed run for ops with no workspace. With 0 or 1 runnable
   candidates the cell is recorded untimed.
2. **Skip.** A current ledger record at the same or higher tier skips the cell. A partly stale
   record re-races only the changed candidates against the stored winner and runner-up.
3. **Race** ("Engine: the lean cell"), starting from the winner of the nearest finished neighbour.
   A candidate is eliminated when it is confidently slower than the leader by more than the tie
   margin; the race stops at one survivor, a confident tie, or the rep cap. Arms that can rank
   first are verified with `batchlas::verify` (@ref design_verification; "Engine: tie-set
   verification").

Across cells:

- **Breadth first.** Round 0 of every op and dtype completes before refinement, so an interrupted
  run leaves usable tables.
- **Every key axis** (n, batch, q, nrhs) refines with its own distance: `log` axes bisect
  geometrically, `exact` axes (uplo, side, trans) never, `batch` refills only its own axis values.
  Nothing is interpolated; `select::choose` takes the nearest row.
- **Estimate.** A cell takes the recorded phases (`ph_*`) of the nearest ledger cell of the same
  protocol, tier and mode (worker or fresh; exact keys equal, log distance), per-arm phases scaled by
  arm count over `ph_arms`, plus that history's worker-start rate and audit share; ld audits once
  per run. Without history: `--cell-overhead-s` plus, per arm, (`lean_max_reps` of the slowest arm
  + 3) × its nearest median + min(0.2 s, 2 × median), plus 0.05 s verify per cell. `--plan` says
  how many cells came from history. `--budget <h>` stops refinement; round 0 always completes.
- **Op order** follows dependencies ("Engine: op order over all 19 ops").

## Engine: the ledger and table generation

- One ledger per (op, device), dtype-tagged lines, appended under `flock`; line kinds, best-record
  rule, dirty runs and kernel hashes: @ref design_tuning_data
  ([the ledger file](tuning-data.md#tuning-data-ledger-file),
  [kernel hashes](tuning-data.md#tuning-data-kernel-hash),
  [generation and the check](tuning-data.md#tuning-data-generation-and-check)).
- Results merge by device key (sm_89, sm_120); the host is recorded.
- A spec's `kernel-sources` block maps each family to its files plus a common set; `src/select/` is
  not hashed. Every spec lists its `src/ops/<op>/<op>.cc` (where `can_run` lives) as a common
  `kernel-deps` entry, so an edit to what can run stales every cell. `kernel-deps` lines
  (`// family: <name> "path" ...`, `// common "path" ...`) add files outside the source list:
  posv's `cta` and `blocked` list potrf's and trsm's kernels, trmm's and symm's `expand` list gemm's.
- `batchlas_tune --import-raw` turns a schema-1 raw sweep into a deep run. Its hashes are
  `legacy:<kernels>` unless the raw file's op hash equals today's (`tools/tune/ledger.cc:468`), so
  an old sweep imports stale.

## Engine: the lean cell

The race of one cell (`run_race`, `tools/tune/cell_runner.hh`; constants in `race.hh`). Ledgers
record it as `protocol=lean-cold`: every run (warm-up, race, verification) restores the inputs, scrubs
L2 with a read of min(4 x L2, memory / 16) bytes leased from the worker Queue's arena, waits, then
starts the clock ([the cold protocol](../perf/batch-sensitivity.md#batch-sensitivity-the-cold-protocol)).
`lean` is the same race on L2-warm operands, `full` the engine before it (a fixed 0.2 s warm-up per
arm), and an absent field is older still. A measured record of any other protocol ranks below every
`lean-cold` record of its cell whatever its tier (C++ `best_records`, `sweep_to_table.py`), and the
planner re-measures its cell in full (`protocol`), on and off round 0; until then the table keeps it.
`--plan` history reads only `lean-cold` cells, so the first cold run plans from the model.

After each timed run the worker samples NVML (SM clock, clock-event reason 0x4, SW power cap). A cell
capped in more than 25% of its samples is raced once more; the record carries `clk_n`, `clk_capped`,
`clk_mhz`, `throttled` and `throttle_reraces` (@ref design_tuning_data).

| Step | Rule | Constant |
| --- | --- | --- |
| 1. Untimed run per arm | keeps JIT and module loads out of the cut | |
| 2. Gross cut | two timed interleaved rounds, then `race_step`: an arm whose lower median paired ratio to the leader exceeds 4 leaves with that median, no warm-up, no reps | `kGrossLoserRatio` = 4.0 |
| 3. Warm until stable | survivors' samples dropped; untimed interleaved rounds until two consecutive round totals agree within 2%, at least 2 rounds, at most 0.2 s per arm (`warm_converged`) | `kWarmTol`, `kWarmMinRounds`, `kWarmCapPerArmS` |
| 4. Rep cap | `lean_max_reps` = ceil(50 ms / slowest survivor's median over the last two warm rounds), clamped to [`min_reps_for(confidence)`, 16]; the floor (3 / 4 / 6) is the first round count at which the paired-ratio rule can decide | `kRepBudgetS` = 0.05, `kRepCap` = 16 |
| 5. Verify | the tie set and `CellRequest::verify` only | "Engine: tie-set verification" |

- The cell runs on the worker's one `Queue` (`OpSpec::run_cell(req, q)`; the ld audit's re-run
  reuses it). The workspace is a lease from that Queue's arena; `~ArmBench` drains the Queue so an
  async error cannot surface in the next cell.
- Each spec's `clear_info` poisons outputs `reset` does not restore (syev `w`, gesvd `s`/U/Vh,
  getrf pivots, geqrf `tau`), so an arm that writes nothing cannot pass on an earlier arm's output.

## Engine: tie-set verification

A cell verifies only the arms that can rank first, plus the arms its bucket has not verified yet
(`verify_next` in `race.hh`; `VerifyBook`, `verify_bucket`, `record_from_arms`, `verified_front`
in `schedule.hh`).

- **Tie set.** The unverified arms within 3% of the fastest survivor's median (no survivor: of the
  fastest eliminated arm). A failure leaves the set and the next arms that can rank first are
  verified, until the set holds no unverified arm. Gross-cut and eliminated arms outside it are not
  verified.
- **Once per bucket.** Bucket = (op, dtype, exact keys, floor(log10 per-item bytes)). Each cell is
  sent the arms with no verdict in its bucket (`CellJob::verify`, worker request `verify`, child
  `--verify`). A pin refusal or an unverified arm stays wanted.
- **Unverified arms stay in the row** with ledger flag `v<i> = 0`: a pin that falls through needs
  them.
- **A failure** (residual or info) condemns the kernel in its bucket: later cells record it `bad`
  (`verify failed @ <key>`) untimed, and each current record of the bucket that ranks it unverified,
  this run's or the ledger's, is appended again with it `bad` (the ld audit's retract path). The
  driver prints `verify: <arm> marked bad in N recorded cells of bucket <b>` and emits
  `verify_retract`.
- **A verified winner.** A record's front arm is one it verified. When a retraction or an all-failed
  partial re-race would put an unverified arm first, unverified arms move below the first verified
  one (none: the ranking empties) and the cell is queued again once (`remeasure`, all live arms).
- **Fresh audit.** The fresh child gets the cell's `verify` list. `audit_compare` treats an arm one
  side left unverified as feasible if it ran there (ok, eliminated or bad); refused or `error`
  against ran is still a mismatch. The child's verdicts go into the book.

Evidence: `TuneVerify.*` in `tests/tune_tests.cc`. Deliberate breaks, each with its own red set:
`want()` asking nothing, retracting verified records, `audit_compare` ignoring the flag, a strict
tie (no 3%), no `v<i>` field, no retract, the audit's verdicts not booked, verifying every arm.

## Engine: persistent workers and the carve-out audit

**Worker.** Each GPU runs one `batchlas_tune_impl --worker` under its `CUDA_VISIBLE_DEVICES`
fence and per-GPU flock; cells arrive on a pipe and return as `cell` records.

- Warms the clocks for 3 s at start (`worker_warm`, `batchlas_tune.cc`); each cell then warms its
  survivors until stable. Each kernel JITs once per worker; the on-disk JIT cache stays shared.
- The idle guard runs between cells. A worker that dies is restarted and the cell retried; a
  candidate that crashes alone is `error`.
- A worker `error` with a sticky CUDA error restarts the worker and re-races the cell in a fresh
  child. If the fresh child reproduces it in two consecutive cells, that op and dtype race the
  candidate alone, one process per cell ("Engine: repeat crashers are dropped").

**Sticky carve-out.** The SLM carve-out attribute is sticky per CUfunction
(`benchmarks/factor_bench.cc` header): an earlier larger launch can make a later one succeed that
fails in a fresh process. Defences:

1. **Ascending order.** A worker runs cells in ascending per-item footprint (input bytes / batch,
   then total bytes) and is restarted before any cell smaller than one it already ran.
2. **Fresh-process audit.** Cells with `fnv1a64(run_id + key) % 1000 < audit_fraction * 1000`, plus
   the first worker cell of each op and dtype, are re-raced in a one-cell fresh child. A mismatch
   (usable in one and refused or failing in the other, or a winner more than 10% slower fresh; an
   arm unverified on either side only has to have run on both) sends the rest of that op and dtype
   to fresh children. Each audited cell gets an `audit` record.

**Result** (threadripper02, GPU 1, 2026-10-07, `tune_race_gpu_tests`): no carve-out mismatch. For
potrf float and double, posv float and trsm float at batch 512, a worker raced the larger cell then
the smaller one at or just below the 48 KB hole, and a fresh child raced the smaller one; every
candidate's feasibility agreed. potrf pads requests out of the hole and its kernels carry no static
shared ([potrf: the 48 KB launch hole](../perf/potrf.md#potrf-the-48-kb-launch-hole)). The guard
stays in the test suite; the mismatch path is covered by injected verdicts in `tune_tests`.

## Engine: measured per-child overhead

threadripper02, sm_120, GPU 1, other GPUs idle, one measuring process, 2026-10-07. Cell: potrf
float `uplo=L,n=16,batch=8192`, 20 sequential `batchlas_tune_impl --cell` children after a warm-up
child.

| Configuration | median s | p90 s |
| --- | --- | --- |
| `time`, 1 arm, reps 1, warm 0 | 0.488 | 0.494 |
| `probe`, 1 arm, no timing | 0.488 | 0.492 |
| `time`, 6 float arms, reps 3, warm 0.2 | 1.802 | 1.809 |

- 0.49 s = process start, SYCL and CUDA init, static init, JIT-cache load and verification
  launches (wall minus in-child kernel time; a lower bound, JIT cache warm). That is 37-60% of a
  raced preview cell's measuring time; three children per cell (about 1.5 s) would add about 70% to
  a preview estimate.
- **Decision:** one persistent worker per GPU. `--cell-overhead-s` defaults to 0.49.

## Engine: driver interface

Driver: `tiered_driver.cc`; planning: `schedule.cc`. Full flag table: @ref tune_tool_readme.

| Flag | Effect |
| --- | --- |
| `<op>[,<op>...\|all] --tier preview\|coarse\|deep\|calibrate --dtype ... --devices ...` | a tiered run (calibrate: `--prior <device>`, `--probe-stride 4`) |
| `--reps`, `--warm`, `--passes`, `--remeasure`, `--refine-ratio` (and other custom flags) | expert overrides: tier `custom`, ranked below preview, written to a ledger only with an explicit `--ledger DIR` |
| `--plan` | cells, skips with reasons, the estimate; no GPU |
| `--budget <h>` | caps refinement time |
| `--progress-fd <n>` | one JSON event per line (cell start/done, elimination, audit, worker restart, refinement cap, `plan_job`, `schedule`; every event carries `t`); benchviz reads it over SSE |
| `--status` | op × dtype × device matrix (tier mix, coverage, stale cells, audits, age) from ledgers and tables; no GPU |
| `--gate` | old choice against Auto per cell (@ref tune_tool_readme, "Gate mode") |

## Engine: refinement convergence rules

`refine_all_axes` (`tools/tune/grid.cc`), shared by the driver and `tune_replay`; each cell's raced
medians arrive as a `RefineCell`.

1. **Flip.** A bracket is a flip when its two winners differ and the flip is decisive at *either*
   end: there the other end's winner is more than the 3% tie slower than the local winner, or is
   not runnable or was eliminated. Near-tie alternation is not a flip.
2. **Margin.** The preview margin (10%) applies only to brackets whose ends are both round-0 cells
   at the running tier; a refinement midpoint never re-triggers it.
3. **Batch** is never bisected below its lattice spacing; it refills its own axis values (the
   CU-aligned ladder: "Engine: the CU-aligned batch ladder").
4. **Cap.** Without a frontier, refinement cells per op and dtype ≤ `refine_cap_factor` × round-0
   cells (preview 3.0, coarse 1.0, deep 2.0), counting the ledger's current records, so a stopped
   run resumes within the same total. Flips refine before margin hedges. Past the cap the run prints
   `refinement cap hit`. A frontier run has no cap.
5. **Estimate.** Ratio × lattice cells measured now, ratio = the ledger's refined / round-0 count
   at the tier (at most the cap factor), else half the cap factor. A frontier run prices
   `kFrontierShare` = 0.86 × the grid cells round 0 leaves.

> **Note:** the flip rule must be one-sided per end. Requiring it at *both* ends dropped the trsm
> flips, whose crossover sits near one end of the bracket, and failed every tier in replay.

GPU run (potrf float, preview, one GPU, persistent worker, before the frontier): 90 lattice and 90
refinement cells in 7 rounds (90, 46, 23, 12, 6, 2, 1), 219 s wall, well inside the cap of 270. The
no-history estimate was about 2x high (it assumes 1.5 refinement cells per lattice cell).

## Engine: testing and acceptance

Tests follow the agent guide's "guards that cannot fail" rules (@ref dev_agent_guide §8).

| Guard | What it checks |
| --- | --- |
| `tune_replay --raw <sweep> --tier <t>` (`tools/tune/replay.cc`) | replays a tier's racing and bisection as planned before the frontier (stride-1 lattice, refinement cap, `TierParams` rep cap; not the lean race) against the exhaustive trsm sweeps (4452 float, 4098 double cells, every rep), no GPU. Score: table misrank, the share of cells where the nearest row's first runnable entry is more than 3% slower than the exhaustive best. Bounds (excess over the noise floor): deep 0.2%, coarse 1%, preview 5% (`replay.cc:290`) |
| `tune_replay --slowdown` | planning schemes on shipped tables, ledgers and raw sweeps ("Engine: flip frontier and the slowdown replay") |
| Host unit tests (`tests/tune_tests.cc`) | precedence (coarse never displaces a current deep record; a stale deep one is displaced), gap fill, partly-stale selection, lattice nesting, `--check` reproducibility; each with a deliberate break that turns only its own test red |
| Racing guard | a synthetic spec with a true near-tie, an arm 4% slower (eliminated in deep) and one 2% slower (kept as a tie) |
| Audit guard | an arm whose feasibility differs between fresh and warm processes is flagged |
| End to end | a preview potrf run on one GPU finishes within the `--plan` estimate and its tables pass `tuned_tables_tests` and `--check`; a coarse trsm float run on sm_120 agrees with the deep table within the replay bound |

## Engine: replay results on trsm

Holdout mode (default): race on pass-1 reps, score against pass-2 medians. Pass 1 and pass 2
disagree by more than 3% on about 1.5% of float cells (1.15% double); an oracle that picks from all
pass-1 reps scores the same floor, so the verdict is the excess over it. Pre-frontier plan:

| Tier | dtype | Bound | Measured (lattice + refined) | Excess, race / table | Verdict |
| --- | --- | --- | --- | --- | --- |
| preview | float | 5% | 2197 (616 + 1581) | 0.41% / 3.89% | PASS |
| preview | double | 5% | 1763 (585 + 1178) | 0.21% / 4.25% | PASS |
| coarse | float | 1% | 4427 (3480 + 947) | 0.03% / 0.04% | PASS |
| coarse | double | 1% | 4066 (3219 + 847) | 0.03% / 0.12% | PASS |
| deep | float | 0.2% | 4427 (3480 + 947) | 0.01% / 0.09% | PASS |
| deep | double | 0.2% | 4068 (3219 + 849) | -0.02% / 0.07% | PASS |

Estimated cost, trsm float on sm_120: 0.57 GPU-h preview, 0.86 coarse, 0.86 deep under the `full`
protocol's model (0.2 s warm-up per candidate); summed rep time 0.07 GPU-h. The model now:
`est_gpu_h = [measure_s + Σ_candidates (3 × median + min(0.2 s, 2 × median)) + (verify_s +
cell_overhead_s) per cell] / 3600` (`lean_arm_fixed_s`, as `--plan`), `verify_s` 0.05,
`cell_overhead_s` 0 for a persistent worker.

Same-sample replay, table misrank over all raw cells, float / double:

| Alternative | Result | Verdict |
| --- | --- | --- |
| Stride 2 start, geometric bisection to 1.25 | 7.9% / 9.7% on 820 / 712 cells | rejected |
| Stride 2 start, index bisection, no margin | 6.7% / 5.8% on 1412 / 1096 cells | rejected |
| Stride 2 start, index bisection, margin 0.10 | 3.6% / 4.3% on 2818 / 2363 cells | preview (stride-1 ops) |
| Stride 4 start, any refinement | 7-13% on 501-1812 cells | rejected |
| Full lattice, geometric bisection to 1.25 | 2.5% / 1.4% on 4021 / 3725 cells | rejected |
| Full lattice, geometric bisection to 1.1, coarse race | 0.11% / 0.05% on 4413 / 4094 cells | coarse, deep (stride-1 ops) |
| No bisection on the full lattice | 6.4% / 6.1% on 3480 / 3219 cells | rejected |
| Shrunk trsm grid: batch stride 2 | 4.3% / 4.8% on 2920 / 2546 cells | rejected |
| Shrunk trsm grid: one batch value | 13.5% / 7.2% on 916 / 814 cells | rejected |
| Shrunk trsm grid: q stride 2 | 1.5% / 3.0% on 2964 / 2435 cells | rejected |
| Shrunk trsm grid: order stride 2 | 4.7% / 6.9% on 1911 / 1837 cells | rejected |
| Shrunk trsm grid: q, batch and order stride 2 | 7.8% / 9.7% on 824 / 716 cells | rejected |

Batch is the costliest axis to shrink, because the winner flips along it. No shrunken grid meets
the 1% coarse bound; the full lattice does. The frontier (next sections) superseded these plans for
every op but the stride-1 ones.

## Engine: flip frontier and the slowdown replay

**Rule** (`flip_frontier`, `tools/tune/grid.cc`). After a stride-2 start, an unmeasured cell is
measured when its k = 6 nearest measured cells hold a decisive flip: one neighbour's winner
(fastest median) is more than 5% slower than the best at another neighbour, or not runnable there.
Neighbours come only from the cell's own exact-key group. Distance is `select::choose`'s: sum of
weight × |log2 a − log2 b| over the log keys (`NearestIndex`, the one host port). Rounds repeat until
none is left.

**Slowdown replay** (`tune_replay --slowdown`, `slowdown_replay` in `tools/tune/replay_core.cc`):

| Part | Definition |
| --- | --- |
| Universe | every timed row of a shipped table (`--table`, deep and custom rows), every current best record of a ledger (`--ledger`, family hashes from the table header), or every ok cell of a raw sweep (`--raw`) |
| Start lattice `ledger` | the ledger's round-0 values per log axis (values under 1% of the axis's commonest dropped as re-measure strays); a table adopts its ledger's lattice. Deep runs race at stride 1, so round 0 is the spec lattice: this models the driver |
| Start lattice `raw` | the sweep's round-0 values |
| Start lattice `inferred` | a table with no ledger: each exact-key group's own distinct row values; denser at old crossovers than the driver's start |
| Group with no start cell | takes the stride of its own values (the driver's round 0 covers every group) |
| Score | t(pick) / t(fastest arm) per universe cell. A measured cell picks its fastest raced arm; any other picks the nearest measured row's first entry that runs there (`NearestIndex::find`, never crossing `dtype`). No runnable entry or no row of its dtype is a failure, listed, not scored. Truth is the shipped medians: race noise and the 3% tie order are not modelled. Quantiles: nearest rank, sorted[⌈f·n⌉ − 1] (below 1000 cells p99.9 is the worst cell) |
| Gate (per line) | no failures, something measured, geomean ≤ 0.05%, p99.9 ≤ 1.10 |
| Schemes | `full`, `lattice` (round 0 at stride 1), `stride2`, `stride4`, `stride2+frontier`, `stride4+frontier`, `stride2+frontier+prune` (a frontier cell races only arms within 1.5x of the best at a neighbour), `lattice+bisect` and `stride2+frontier+bisect` (driver models, next section) |
| Measured cells | scored separately (`measured_*`); exactly 1.0 by construction except under prune |

**Gate run** (2026-10-16, threadripper02, one core, 44 s): shipped `tuned/*.sm_120.txt` with their
ledgers, plus the two raw trsm sweeps:

```
tune_replay $(for t in tuned/*.sm_120.txt; do echo --table $t; done) \
    $(for l in benchmarks/results/tuning/ledger/*.sm_120.jsonl; do echo --ledger $l; done) \
    --raw benchmarks/results/tuning/trsm.float.sm_120.jsonl --raw benchmarks/results/tuning/trsm.double.sm_120.jsonl
```

All 19 tables, 115,595 cells, `ledger` lattice:

| Scheme | Cells kept | Arm runs | Geomean slowdown | p99 | p99.9 | Cells >10% slow | >3% (misrank) | Worst | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 100% | 100% | 0.000% | 1.000 | 1.000 | 0.00% | 0.00% | 1.00x | 0 |
| lattice | 59% | 47% | 3.079% | 1.747 | 4.241 | 8.74% | 13.86% | 16.92x | 0 |
| stride2 | 12% | 10% | 5.129% | 2.055 | 5.237 | 14.03% | 21.20% | 42.69x | 6 |
| stride4 | 3% | 2% | 8.905% | 3.516 | 40.767 | 17.40% | 24.52% | 578.78x | 32 |
| stride2+frontier | 76% | 82% | 0.044% | 1.000 | 1.072 | 0.08% | 0.15% | 5.28x | 0 |
| stride4+frontier | 82% | 88% | 0.073% | 1.000 | 1.234 | 0.17% | 0.24% | 5.28x | 0 |
| stride2+frontier+prune | 76% | 63% | 0.074% | 1.000 | 1.255 | 0.16% | 0.29% | 5.28x | 0 |

- Prune's measured cells alone: geomean 0.037%, p99.9 1.104, 0.10% over 10% slow, worst 2.01x.
- Ledgers as the universe give the same `stride2+frontier` row (0.044%, p99.9 1.072).
- Raw trsm (8550 cells): `stride2+frontier` keeps 78% for 0.006% geomean, p99.9 1.014, misrank
  0.00% / 0.02% (float / double), against 3.6% / 4.3% for the preview scheme in the race replay;
  `lattice` misranks 4.99% / 5.59% here and 6.4% / 6.1% there.

**Against the tuner-overhaul plan's replay** (2026-10, `inferred` lattice on deep rows only, 115,547
cells): geomean / p99.9 / worst.

| Scheme | `inferred`, deep rows | Plan | `inferred`, deep + custom | `ledger` |
| --- | --- | --- | --- | --- |
| stride2 | 4.324% / 5.952 / 204.28x | 4.32% / 5.928 / 204.28x | 4.345% / 6.108 / 204.28x | 5.129% / 5.237 / 42.69x |
| stride4 | 9.124% / 35.041 / 578.78x | 9.12% / 35.023 / 578.78x | 9.064% / 33.822 / 578.78x | 8.905% / 40.767 / 578.78x |
| stride2+frontier | 0.018% / 1.054 / 1.78x | 0.02% / 1.052 / 1.78x | 0.018% / 1.052 / 1.78x | 0.044% / 1.072 / 5.28x |
| stride4+frontier | 0.013% / 1.027 / 1.77x | 0.01% / 1.016 / 1.77x | 0.013% / 1.021 / 1.77x | 0.073% / 1.234 / 5.28x |
| stride2+frontier+prune | 0.044% / 1.166 / 2.27x | 0.04% / 1.171 / 2.27x | 0.045% / 1.174 / 2.27x | 0.074% / 1.255 / 5.28x |

The p99.9 residue is the quantile rule. Custom rows move gemv (29 rows: p99.9 1.167 to 1.052) and
trmm (17 rows: 1.060 to 1.187). The `ledger` lattice is the whole gap to the plan: its start holds
only spec values, so the per-op failures below are cells the plan's start measured as midpoints.

`stride2+frontier` per op (`ledger` lattice; raw trsm last; right columns: p99.9 / worst with the
`inferred` lattice here and in the plan):

| Op | Cells | Kept | Arm runs | Geomean | p99.9 | Worst | Gate | Inferred p99.9 / worst | Plan p99.9 / worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemm | 32225 | 81% | 84% | 0.110% | 1.339 | 5.28x | FAIL | 1.100 / 1.31x | 1.100 / 1.31x |
| gemv | 8587 | 89% | 91% | 0.017% | 1.063 | 1.29x | PASS | 1.052 / 1.78x | 1.163 / 1.78x |
| geqrf | 2186 | 58% | 60% | 0.037% | 1.140 | 1.32x | FAIL | 1.121 / 1.32x | 1.104 / 1.32x |
| gesvd | 2034 | 66% | 77% | 0.000% | 1.000 | 1.00x | PASS | 1.000 / 1.00x | 1.000 / 1.00x |
| gesv | 877 | 59% | 65% | 0.007% | 1.061 | 1.06x | PASS | 1.061 / 1.06x | 1.008 / 1.06x |
| getrf | 1141 | 70% | 68% | 0.001% | 1.000 | 1.01x | PASS | 1.000 / 1.01x | 1.000 / 1.01x |
| getri | 363 | 80% | 80% | 0.057% | 1.113 | 1.11x | FAIL | 1.113 / 1.11x | 1.111 / 1.11x |
| getrs | 10653 | 77% | 78% | 0.002% | 1.000 | 1.06x | PASS | 1.000 / 1.21x | 1.000 / 1.21x |
| orgqr | 497 | 40% | 40% | 0.062% | 1.262 | 1.26x | FAIL | 1.000 / 1.00x | 1.000 / 1.00x |
| ormqr | 1370 | 26% | 26% | 0.000% | 1.000 | 1.00x | PASS | 1.000 / 1.00x | 1.000 / 1.00x |
| posv | 8616 | 81% | 86% | 0.018% | 1.058 | 1.54x | PASS | 1.000 / 1.01x | 1.000 / 1.01x |
| potrf | 1374 | 80% | 85% | 0.001% | 1.000 | 1.01x | PASS | 1.000 / 1.01x | 1.000 / 1.01x |
| spmm | 2894 | 75% | 76% | 0.047% | 1.183 | 1.24x | FAIL | 1.224 / 1.39x | 1.187 / 1.39x |
| syev | 1226 | 93% | 95% | 0.000% | 1.000 | 1.00x | PASS | 1.000 / 1.01x | 1.000 / 1.01x |
| symm | 3604 | 65% | 66% | 0.112% | 1.851 | 1.87x | FAIL | 1.000 / 1.00x | 1.000 / 1.00x |
| syr2k | 941 | 60% | 69% | 0.000% | 1.000 | 1.00x | PASS | 1.000 / 1.00x | 1.000 / 1.00x |
| syrk | 12805 | 56% | 64% | 0.003% | 1.002 | 1.25x | PASS | 1.000 / 1.25x | 1.000 / 1.03x |
| trmm | 2566 | 94% | 95% | 0.071% | 1.299 | 1.40x | FAIL | 1.187 / 1.73x | 1.037 / 1.43x |
| trsm | 21636 | 80% | 83% | 0.014% | 1.022 | 1.24x | PASS | 1.013 / 1.23x | 1.012 / 1.23x |
| trsm float (raw) | 4452 | 79% | 77% | 0.004% | 1.011 | 1.02x | PASS | - | - |
| trsm double (raw) | 4098 | 76% | 77% | 0.008% | 1.015 | 1.04x | PASS | - | - |

**Verdict.**

- The overall gate passes for `stride2+frontier` with both lattices: 0.044% / 1.072 with `ledger`
  (the driver's start; the binding number), 0.018% / 1.052 with `inferred`.
- Per op (`ledger`): gemm fails both bounds (0.110%, 1.339; worst 5.28x at cdouble T/N 8×8×8, batch
  32768); symm fails both (0.112%, 1.851). geqrf, getri, orgqr, spmm and trmm meet the geomean but
  fail p99.9: getri (363 cells) and orgqr (497) on their worst cell, the others on their
  third-worst. With `inferred`, gemm passes the geomean (0.036%) and sits on the p99.9 bound
  (1.100); symm and orgqr pass.
- `stride4+frontier` measures more and scores worse with `ledger`. Prune cuts arm runs from 82% to
  63% but raises geomean to 0.074% and p99.9 to 1.255: opt-in only (replay scheme, not a driver
  mode).

Numbers are dated 2026-10-16: the sm_120 gemm table has since been regenerated (22,203 deep rows on
2026-10-21) and was not re-replayed.

## Engine: round planning by flip frontier

Preview, coarse and deep runs plan rounds by the flip frontier (`TieredRun::build` and `refine`,
`tools/tune/tiered_driver.cc`); in frontier mode preview's index refill, margin and cap are off.
Tests: `TuneTieredDriver.Frontier*`, `TuneTieredDriver.Preview*`, `TuneFrontier.Start*`,
`TuneReplay.BisectSchemes*`.

| Part | Rule |
| --- | --- |
| Round 0 (`frontier_start`, shared with the replay) | grid cells on `tier_lattice(axes, 2)`; an exact-key group with none (filtered by `grid()`, a `--grid` override) takes every second value of its own log values, ends kept, and with none of those starts whole; every grid cell whose best current record at the running tier or higher is partly stale joins, so its `partial:` re-race runs as on the full lattice |
| Frontier rounds | grid cells not yet planned whose k nearest measured cells hold a decisive flip (`flip_frontier`). Measured: this run's records and the ledger's current ones of the same or a higher tier, refinement midpoints and older stride-1 runs included. A `skip:single` record counts (its one arm wins, the others cannot run) |
| Bisection | 3% tie, `refine_all_axes` in index mode, same rounds. A midpoint on the grid is dropped (the frontier's to pick); geometric midpoints between adjacent grid values place a crossover inside a frontier bracket |
| Stop | when neither adds a cell. No refinement cap; `--budget` stops rounds after round 0 |
| `--plan` | counts round 0; prices `kFrontierShare` = 0.86 of the grid cells after round 0 that `plan_round` would measure; prints `frontier (stride 2, k 6, 5%): ~N cells, data-dependent` |

**Per op** (`OpSpec::frontier()` returns a `FrontierPlan`: stride, k, slower; `tools/tune/spec.hh:55`).
Rule: the cheapest variant that passes the per-op gate (geomean ≤ 0.05%, p99.9 ≤ 1.10) in the
replay above (`ledger` lattice), else the full lattice.

| Op | Plan | Why |
| --- | --- | --- |
| gemm | stride 1 (full lattice, demand-driven grid, bisection and cap) | no variant passes |
| geqrf, getri, orgqr, spmm | stride 1 | p99.9 fails at k 6, 8 and 10 alike, on the same worst cells |
| symm | stride 2, k 8 | cheapest passing variant |
| trmm | stride 2, k 8 | passes at k 8 (97.0% kept), fails at k 6 |
| every other op | stride 2, k 6, 5% (default) | passes |

gemm and symm searched in full (k 6, 8, 10; slower 5%, 3%, 2%; stride 2 and 4):

| Op | Variant | Kept | Geomean | p99.9 | Gate |
| --- | --- | --- | --- | --- | --- |
| gemm | stride 2, k 6, 5% | 81.0% | 0.110% | 1.339 | FAIL |
| gemm | stride 2, k 6, 3% | 82.2% | 0.106% | 1.339 | FAIL |
| gemm | stride 2, k 8, 5% | 82.8% | 0.104% | 1.320 | FAIL |
| gemm | stride 2, k 10, 5% (3%, 2% alike) | 85.1% | 0.089% | 1.090 | FAIL |
| gemm | stride 4, k 8 or 10 | 91.8-95.9% | 0.093% | 1.095 | FAIL |
| symm | stride 2, k 6 (3%, 2% alike) | 65.4% | 0.112% | 1.851 | FAIL |
| symm | stride 4, k 6 | 81.7% | 0.039% | 1.106 | FAIL |
| symm | stride 2, k 8, 5% | 70.6% | 0.000% | 1.000 | PASS |
| symm | stride 2, k 10, 5% | 74.0% | 0.000% | 1.000 | PASS |

**Driver-model replay** (2026-10-16, k 6, same inputs as the gate run). The driver picks only grid
cells and reaches the rest by bisection; `lattice+bisect` (pre-frontier coarse and deep plan) and
`stride2+frontier+bisect` (this driver) model that; a midpoint the universe lacks is not measured
(`bisect_missing`).

| Universe | Scheme | Measured | Geomean | p99.9 | Midpoints missing |
| --- | --- | --- | --- | --- | --- |
| raw trsm (8550 cells) | `lattice+bisect` | 95.4% | 0.009% | 1.030 | 687 |
| raw trsm | `stride2+frontier+bisect` | 71.7% | 0.023% | 1.036 | 675 |
| tables but gemm (83,370) | `lattice+bisect` | 73,217 | 0.412% | - | - |
| tables but gemm | `stride2+frontier+bisect` | 51,048 | 0.582% | - | - |
| gemm table (32,225) | `lattice+bisect` | 58.4% | 1.049% | 1.730 | 1488 |

- On the tables both schemes fail the gate: a table is the union of several runs and re-measures,
  so bisection from its medians stalls on midpoints those runs never measured. Read the table rows
  as a comparison.
- The frontier measures 30% fewer cells for +0.17% pooled geomean. Per op the excess is under 0.05%
  for geqrf, gesv, gesvd, getrf, ormqr, posv, potrf, syev, syrk and trmm, and largest for spmm
  (+1.32%), trsm (+0.25%), symm (+0.24% at k 6, +0.08% at k 8), gemv (+0.23%) and getrs (+0.09%).
  The raw trsm sweeps give +0.005% (float) and +0.022% (double).
- Cells a deep run measures, driver model over the 115,595 table cells: pre-frontier plan about
  92,000 (79.6%); frontier with only gemm on the full lattice (symm k 8) about 70,000, 24% fewer.
  geqrf, getri, orgqr and spmm (5,940 table cells) on the full lattice give back part of that.

## Engine: calibrate tier from a borrowed table

`--tier calibrate` fills a new device's tables from the nearest shipped table (the prior): it
probes the prior on the device and measures only where the device disagrees. Code: `CalibrateRound`
(`grid.cc`), `Prior`, `prior_table`, `calibrate_replay` (`replay_core.cc`), `TieredRun::build`,
`plan`, `refine`, `copy_prior` (`tiered_driver.cc`). Tests: `TuneCalibrate.*`, `TuneTier.*`,
`sweep_to_table.py --self-test` ("prior rows").

| Part | Rule |
| --- | --- |
| Prior | `--prior <device>` reads `tuned/<op>.<device>.txt`. Default (`prior_table`): among `tuned/<op>.<d>.txt`, `d` neither the device's own key nor `cpu`, the newest header `date=` among the fully timed (≥ 90% timed rows); else the larger timed-row share, then the newest date, then the device name. Today: sm_120 for every op. Read host-side (`load_table_oracle`, untimed rows in `OracleSet::rows`); a cell takes the prior's nearest row (`NearestIndex::find`), fields beyond the table keys ignored; the prior's pick is that row's first current candidate |
| Probe (round 0) | `frontier_start(grid, axes, 4)` (`--probe-stride 8`: cheaper). Probes race every arm at coarse quality. A probe *disagrees* when the device's winner is more than 5% faster than the prior's pick, or the pick cannot run |
| Refinement | open grid cells u whose prior pick is decisively beaten at one of u's 6 nearest measured cells (`OpSpec::frontier().k`, same exact keys) that disagrees with the prior; a cell with no prior row takes `flip_frontier`'s rule. Bisection (3% tie) adds off-grid midpoints only next to a disagreeing cell (one of its 2 nearest). Stops when neither adds a cell, or at `--budget` |
| Arms off the probes | the prior's top 2 at the nearest prior row (transcribed prior: order only), every candidate of a family the prior's rows for the dtype never name (trsm.sm_89 has no `sg_left`), and the winners of the 6 nearest measured cells. Probes and `--remeasure-keys` cells race every arm. Left-out arms are `skipped`, reason `unraced:calibrate`: never stale, never "not runnable" to the frontier. `--only-named` is refused |
| Prior rows | at job end, every grid cell with no measured record and no contradiction gets the prior's nearest row as a `prior` record (untimed, unhashed, never stale, below every measured tier). Without them the device's sparse own table would shadow the borrowed one everywhere (`select::detail::walk` reads the own table first). A contradicted cell left unmeasured gets no row. A rerun appends no row whose ranking is unchanged |
| Ranking | `calibrate` between preview and coarse; `prior` between `transcribed` and `converted`. Generation rules: @ref design_tuning_data |
| `--plan` | prices the probe; refinement is data-dependent. The summary prints probes, disagreeing probes, prior rows copied and contradicted cells |

**Replay** (`tune_replay --table tuned/*.sm_120.txt --ledger ...sm_120.jsonl --prior tuned/*.sm_89.txt`,
2026-10-16, CPU only). Universe: the shipped sm_120 tables with sm_120's own table hidden; prior:
the sm_89 tables, the only real cross-device data in the tree (a weak prior: all but potrf are
untimed router order). `calibrate_replay` emulates the driver without bisection and scores select's
pick on the generated table. Arm runs = timed arms raced / timed arms in the universe.

| Scheme | Probes (disagree) | Measured | Arm runs | Prior rows | Geomean | p99 | p99.9 | > 10% slow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prior` (borrow blindly) | - | 0% | 0% | 115,595 | 19.8% | 12.3 | 451 | 29.6% |
| `calibrate-probe` (no refinement) | 3,403 (1,322) | 2.9% | 2.1% | 27,245 | 8.46% | 3.20 | 37.6 | 16.8% |
| `calibrate8` | 1,406 (511) | 70.1% | 51.1% | 34,573 | 1.47% | 1.259 | 10.1 | 2.83% |
| `calibrate` (stride 4) | 3,403 (1,322) | 80.7% | 61.3% | 22,255 | 0.291% | 1.074 | 1.770 | 0.79% |
| `stride2+frontier` (from scratch, reference) | - | 76.1% | 81.8% | - | 0.044% | 1.000 | 1.072 | 0.08% |
| `calibrate`, prior = sm_120 itself (sanity) | 3,403 (0) | 2.9% | 2.1% | 112,192 | 0.132% | 1.027 | 1.030 | 0% |

- With the sm_89 prior 39% of probes disagree; calibrate measures more cells than a from-scratch
  frontier with fewer arms per cell (61% against 82%) and a worse table (0.29% against 0.044%).
  The loss sits where the prior is trusted between two agreeing probes: syr2k (2 of 32 probes
  disagree) 5.6% geomean, gesvd 2.0%, spmm 0.79%, geqrf 0.68%, gemm 0.48%.
- The sanity row is the floor of a perfect prior: select's tie rule costs 0.13%.

| Alternative | Result | Verdict |
| --- | --- | --- |
| Also require a device flip next to a contradicted cell | measured 76.2%, geomean 0.39%, p99.9 2.23 | rejected |

**Open gate:** calibrate pays off only with a prior close to the device. Whether a shipped table is
that close for a real second device is unmeasured: a lean deep run on the 2x4090 box, then this
replay with the sm_120 prior against it.

## Engine: strict pins and gross losers {#engine-readiness-for-a-deep-run-over-all-19-ops}

- **Strict pins.** `vendor` is both a family spelling and a routing class word, and a class word
  that cannot serve the shape falls back to Auto. The tuner pins strictly
  (`select::ScopedPin(op, word, select::StrictPin{})` in `cell_runner.hh`): a class word nothing in
  its class can serve throws `batchlas::invalid_argument`, and `vendor` must resolve to the candidate
  spelled `vendor`. The arm is `skipped`, reason `pin refused (strict): ...`. Library callers and
  `BATCHLAS_<OP>_ROUTE` keep the fallback.
  - Without it, an arm pinned to `vendor` where `can_run` refuses timed Auto under the vendor's name
    (seen on spmm cdouble N/N nrhs=1 and on gesvd).
  - gesvd keeps a spec-level refusal: `can_run` admits a shape whose vendor call faults the device
    (known defect 15), which no pin rule can see.
- **Gross losers.** From the second round on, whatever `min_reps` says, the race eliminates an arm
  whose lower median paired ratio to the leader exceeds 4.0 (`kGrossLoserRatio`, `race.hh`; the
  lower of the two middle ratios, so at two rounds both must exceed it). One stalled round never
  kills an arm; it is recorded `eliminated` with the median of the rounds it ran. The lean cell
  applies it first. Case: geqrf's vendor (`geqrfBatched`, one CTA per matrix) is tens to hundreds of
  times slower than blocked on large cells, minutes per timed round without the rule.

## Engine: the CU-aligned batch ladder

> **Status:** built 2026-10-21. Offline replay of the sm_120 ledgers plus a getri and potrf float coarse
> A/B on GPU 2 (warm operands, the tuner's protocol); no full re-tune run yet.

**Rule.** A tiered run (preview, coarse, deep, calibrate) can replace an op's `choice.hh` batch axis
with `batch_ladder(cus, classes, extras)` (`tools/tune/grid.cc`): max(1, round(c × CUs)) per class c,
plus the spec's `ladder_extra_batches()`. The default classes are 0.25, 2, 16 and 128: latency (C/4),
about one wave of a two-resident kernel (2C), and saturated (16C, 128C). On sm_120 (188 SMs) that is
47, 376, 3008, 24064; on an RTX 4090 (128 SMs), 32, 256, 2048, 16384. The key stays `batch:log`. The
table format, hashes, protocol and the meaning of existing rows are unchanged. The run's `grid` field
records the ladder as its lattice.

- **`--batch-ladder auto` (default).** The ladder applies to an (op, dtype) whose ledger has no run, or
  has a run whose `grid` is this ladder (`ladder_applies`). On a ledger of `choice.hh`-axis runs only it
  stays off and prints `batch ladder off`: no ladder cell is current there, so turning it on means
  re-measuring every cell. A host-only `--plan` of `gemm --tier deep --dtype float` on the committed
  sm_120 ledger plans 0 cells with the ladder off and 3380 cells, 2.61 h with refinement with it forced.
- **`--batch-ladder default`** forces the default classes, `c,c,...` other classes; `off`, `--batches`
  or `--grid batch=...` keep the `choice.hh` axis.
- **Samples keep the `choice.hh` axis.** `--tier custom` and `--only-named` measure named keys only.
- **Old-axis rows under a forced ladder.** The Region (refinement's domain) spans the declared grid as
  well as the ladder, so old rows (batch 32768; 1-27 for non-level-3 ops) stay refinement cells and a
  remeasure inside them requeues. A partly stale old row inside the Region joins round 0, as a grid
  cell does. Old rows are never retired: `sweep_to_table.py` writes every non-stale record, so the
  table keeps 128/512/2048/8192/32768 beside the ladder rows, and the L2-warm and co-tenant-contaminated
  old rows (getrf float b64-255, the trsm 2.27 ms spikes) still govern their neighbourhoods. A full
  ladder re-tune therefore starts from a fresh ledger root (see "Re-tune").
- **Compute units.** A measuring run reads them on the first `--devices` GPU (`MAX_COMPUTE_UNITS` from
  the `--info` child); another GPU of the run with a different count only warns. `--plan` has no CUDA
  context and nvidia-smi has no SM-count query: it takes `--cus N`, else a name table (RTX PRO 6000
  Blackwell 188, RTX 4090 128), else it refuses. A wrong table entry plans other batches than the
  measuring run uses; `--cus N` pins both.
- **Grids that are not lattices** (gemm, syrk, symm, ormqr): `rebatch` gives each line of the declared
  grid (the cell less `batch`) the ladder. A line declared at one batch only (syrk's `trans=C`, at
  batch 1) keeps it.
- **Batch is still never bisected** ("Engine: refinement convergence rules", rule 3): the frontier and
  bisection refill only ladder values.

| Op | Hook | Batches on sm_120 | Reason |
| --- | --- | --- | --- |
| gemm | `ladder_classes`: no latency class; extras 65536, 131072 | 376, 3008, 24064, 65536, 131072 | its axis starts at 128; gemm is half of all round-0 cells and a fourth class adds a third to it; sliced past 65535 |
| ormqr | `ladder_classes`: no latency class | 376, 3008, 24064 | as gemm's |
| syrk, symm | extras 1, 2, 4, 8, 65536, 131072 | 1, 2, 4, 8, 47, 376, 3008, 24064, 65536, 131072 | the vendor-loop/native crossovers lie at batch 1-8; sliced past 65535 |
| syr2k | extras 1, 2, 3, 65536, 131072 | 1, 2, 3, 47, ..., 24064, 65536, 131072 | the vendor loop wins batch 1 only; the ledger holds 1, 2, 3 |
| trmm | extras 1, 65536, 131072 | 1, 47, ..., 24064, 65536, 131072 | below 128 the ledger holds batch 1 only |
| the rest | none | 47, 376, 3008, 24064 | batch < 128 is out of scope (agent guide §10); the losses it costs are below |

`TuneSpecLadder.RealSpecHooksGiveTheDocumentedLadders` (`tune_race_gpu_tests`) pins this table, and
`TuneSpecLadder.Level3OldAxisGridsReachPastTheGridYCap` the `choice.hh`-axis grids (every multi-batch level-3
line holds 65536 and 131072; ormqr, potrf and trsm none).

**Cost.** Empty ledger (`batchlas_tune all --tier deep --dtype float,double,cfloat,cdouble --plan
--ledger <empty>`, sm_120): round-0 cells 27061 → 24120, frontier estimate 51963 → 39834 cells, 20.1
→ 19.9 GPU-h. More of the top row fits the 4 GiB cap at 24064 than at 32768, so the byte-model lattice
estimate rises (8.05 → 8.69 h). On the committed ledger the comparison is 0 against a full re-tune:
`all --tier deep --plan` on the committed ledgers (whose records are partly stale anyway since later
kernel changes) plans, float and cfloat, 6748 cells / 5.99 h with `auto` (the old axis, stale cells
only) and 12765 cells / 10.38 h with the ladder forced; all four dtypes 19415 / 22.2 h against
23877 / 23.8 h. A fresh ledger, float and cfloat: 14267 cells / 10.8 h on the old axis, 12883 / 11.0 h
on the ladder (fewer cells; more of them fit the cap at 24064). The ops whose round-0 count changes (empty ledger):

| Op | Round-0 cells to measure, `choice.hh` axis | Ladder |
| --- | ---: | ---: |
| syrk | 1410 | 656 |
| getrs | 1219 | 696 |
| gemv | 542 | 320 |
| getrf | 311 | 146 |
| spmm | 1188 | 940 |
| getri | 322 | 276 |
| gemm | 13582 | 12594 |
| ormqr, symm, potrf, trsm, posv, syev, syr2k, trmm | 5044 | 5049 |

**GPU A/B (the direct check).** One float coarse run per op on GPU 2 into a scratch ledger, batch
axis = the union of `choice.hh` and the ladder (`--grid batch=47:128:376:512:2048:3008:8192:24064:32768`),
table from that ledger. `tune_replay --slowdown` then keeps only the ladder batches (`--batch-ladder 188`)
or only the `choice.hh` batches (`--batch-ladder 1 --batch-classes 128,512,2048,8192,32768`) and scores
every measured cell of the union. getri is a stride-1 op (coarse = full lattice: every cell, every
arm); potrf ran its frontier (437 of 750 cells), so its oracle is the frontier's cells. Warm operands.

| Op | Scored | Kept batches | `ladder` (every kept cell): geomean / p99 / worst | `ladder+frontier+bisect`: geomean / worst |
| --- | --- | --- | --- | --- |
| getri | all 171 | ladder | 0.10% / 1.044 / 1.09 | 0.62% / 1.28 |
| getri | all 171 | `choice.hh` | 0.10% / 1.046 / 1.12 | 0.66% / 1.28 |
| getri | batch ≥ 128 (148) | ladder | 0.11% / 1.044 / 1.09 | 0.71% / 1.28 |
| getri | batch ≥ 128 (148) | `choice.hh` | 0.11% / 1.046 / 1.12 | 0.76% / 1.28 |
| potrf | all 437 | ladder | 0.84% / 1.317 / 1.38 | 0.93% / 1.38 |
| potrf | all 437 | `choice.hh` | 0.65% / 1.228 / 1.39 | 0.64% / 1.39 |
| potrf | batch ≥ 128 (375) | ladder | 0.80% / 1.317 / 1.36 | 0.91% / 1.36 |
| potrf | batch ≥ 128 (375) | `choice.hh` | 0.53% / 1.165 / 1.36 | 0.51% / 1.23 |

getri: no difference. potrf: the ladder is 0.3-0.4 points of geomean worse at the same worst cell
(float L n=175 b=512 picks `lpanel:panel=16`; n=1280 b=128 reads the b=47 row and picks vendor).

**Held-out routing loss on the committed ledgers (a lower bound).** `tune_replay --slowdown --ledger
<op>.sm_120.jsonl --batch-ladder 188 --batch-classes 0.25,2,16,128 [--batch-extra ...]
--drop-nonmonotone 5 --min-batch B`. The ledgers hold no ladder batch, so `nearest_batches` keeps each
exact-key group's measured batch nearest each ladder value; the rest are held out and scored:
t(first runnable entry of the nearest kept row) / t(fastest). `--drop-nonmonotone 5` first drops
co-tenant-contaminated cells (fastest or any arm > 5x itself at a larger batch of the line): potrf 1,
trsm 8, getrf 8, symm 8, spmm 72. Like for like: today = `stride2+frontier+bisect` on every batch,
ladder = `ladder+frontier+bisect` on the kept ones (what the driver runs), plus `ladder` (every kept
cell, no frontier). The proxy keeps four of the five old batches at or above 128, so for most ops the
held-out set is batch 8192 only; it is a lower bound on the real ladder's loss, not a test of it.

Scored batch ≥ 128 (agent guide §10):

| Op | Kept | Held out | Today: geomean / worst | Ladder+frontier: geomean / p99 / worst | `ladder`: geomean / worst |
| --- | --- | --- | --- | --- | --- |
| potrf | 128, 512, 2048, 32768 | 8192 | 0.06% / 1.19 | 0.31% / 1.135 / 1.50 | 0.25% / 1.50 |
| trsm | 128, 512, 2048, 32768 | 8192 | 0.60% / 3.57 | 1.05% / 1.359 / 3.57 | 0.52% / 2.59 |
| posv | 128, 512, 2048, 32768 | 8192 | 0.23% / 7.85 | 0.34% / 1.096 / 7.85 | 0.13% / 1.78 |
| getrf | 128, 512, 2048, 32768 | 255, 256, 8192 | 0.14% / 1.28 | 0.54% / 1.169 / 1.47 | 0.45% / 1.47 |
| getrs | 128, 512, 2048, 32768 | 8192 | 0.28% / 1.74 | 0.87% / 1.300 / 2.90 | 0.59% / 2.90 |
| getri | 128, 512, 2048, 32768 | 8192 | 0.06% / 1.11 | 0.62% / 1.224 / 1.57 | 0.56% / 1.57 |
| syev | 128, 512, 2048, 32768 | 8192 | 0.01% / 1.07 | 0.21% / 1.064 / 1.46 | 0.20% / 1.46 |
| gemv | 128, 320, 8192, 32768 | 319, 1024 | 1.20% / 3.09 | 2.06% / 1.550 / 3.09 | 1.19% / 2.57 |
| trmm | 128, 1024, 32768 | none | 3.64% / 8.46 | 2.95% / 1.488 / 1.78 | 0 |
| gemm, ormqr, spmm, syrk, symm, syr2k | all | none | as ladder | identical | 0 by construction |

Like for like the ladder costs 0.2-0.9 points of geomean (gemv 0.9, getrs 0.6, getri 0.6, trsm 0.45),
worst 1.5-7.85 (posv's 7.85 is today's worst too). Scored from batch 32 the table is the same except
getrf (kept 64 for 47; held out 128, 255, 256, 8192: today 0.20% / 1.36, ladder 0.64% / 1.47) and
getrs (kept 127; today 0.30% / 2.41, ladder 1.71% / 2.90).

**Batch < 32 without extras.** Scored from batch 1 (min-batch 0), ladder+frontier against today:
getrs 3.66% / worst 3.59 (0.33% / 2.68; double n=128 nrhs=6 b=1 picks cta), gemv 2.53% / 5.64 (0.98% /
3.09; double T b=1 picks vendor), getrf 0.87% / 2.82 (0.28% / 1.38; float n=32 b=8 picks cta), spmm
18.42% / 16.92 (10.89% / 16.92, already the worst op today). These ops get no extras: batch 1-16 is
outside the batches this project tunes for (§10), and their small-batch rows are kept today only
because the `choice.hh` axes start at 1. Add `ladder_extra_batches()` to an op that needs them. The
level-3 extras: syrk 0.51% / 3.16 (tall T n=1153 k=18 b=5: vendor), symm 0.53% / 2.81, syr2k and trmm
as today; extras 1, 2, 3 only cost syrk 2.23% and symm 1.15%, extra 1 only 3.44% and 9.35%.

**Rejected classes** (scheme `ladder`, geomean at batch ≥ 32; kept batches in brackets for potrf; a
ranking of class sets against each other, not against today):

| Classes | potrf | trsm | posv | getrf | getrs | getri | syev | gemv | Round-0 cells / est. GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `choice.hh` axis (today) | | | | | | | | | 27061 / 20.1 |
| 0.25, 1, 8, 64 [128, 2048, 8192] | 0.88% | 1.25% | 0.26% | 0.88% | 0.50% | 1.83% | 2.75% | 2.35% | 27375 / 21.9 |
| 0.25, 1, 8, 64, 256 [+32768] | 0.69% | 0.53% | 0.12% | 0.67% | 0.16% | 0.31% | 2.73% | 1.65% | 28049 / 23.5 |
| 0.25, 4, 64 [128, 512, 8192] | 0.84% | 1.19% | 0.42% | 0.96% | 1.83% | 2.01% | 0.76% | 2.35% | 17835 / 14.7 |
| 0.25, 2, 8, 32, 128 [all five] | 0 | 0 | 0 | 0.30% | 1.02% | 0 | 0 | 0.04% | 28789 / 25.8 |
| **0.25, 2, 16, 128** [128, 512, 2048, 32768] | 0.25% | 0.52% | 0.13% | 0.51% | 1.50% | 0.56% | 0.20% | 1.19% | 24120 / 19.9 |

Costs are empty-ledger `--plan` totals with the gemm and ormqr hook. C/4, C, 8C, 64C (the plain
reading of latency, one wave, saturated) has no top class near 32768, and the small-n CTA cells lose
up to 3.7x (getri double n=2 b=32768). The five-class ladder reproduces today's five-point axis but
plans more cells than today. The three-class ladder is the cheap option: 34% fewer round-0 cells and
27% fewer GPU-h than today, at 0.4-2.0% geomean on these ops (`--batch-ladder 0.25,4,64`).

**Batches past the grid-y cap.** Level-3 (gemm, syrk, symm, syr2k, trmm) runs a batch past
`max_groups_yz` (65535) in chunks ([gemm batch chunking](../perf/gemm.md#gemm-batch-chunking)),
so every native competes there and a table without a row past 32768 extends its 32768 (or lower) row
to any batch. `with_slice_batches` (`tools/tune/grid.cc`) adds 65536 and 131072 to these specs'
`ladder_extra_batches()` and to their `choice.hh` axes (the specs' `axes()` and `grid()`; `choice.hh`
itself is unchanged, so no kernel hash moves). herk has no spec. A cell still passes the 4 GiB byte cap
(`--cap-gib`): at 65536 that admits at most 64 KiB of operands per item (float gemm 64^3 NN packed, not
128^3), so the large shapes need `--cap-gib 32` (96 GB parts) to get their rows. Host-only `--plan`, sm_120
(`--cus 188`), all ops, float and cfloat, before → after: empty ledger 12883 → 13357 cells, 11.00 → 11.73 GPU-h
with refinement; the committed ledger on its old axis 14149 → 14627 cells, 13.52 → 13.92 h; at `--cap-gib 32`
(empty ledger) 17542 → 19614 cells, 39.3 → 52.1 h. gemm is +428 of the 474 cells at the default cap. The committed-ledger totals are above the 6748 cells
quoted below because F4b's `src/ops/gemm/choice.hh` edit moved gemm's op hash, so every gemm record is stale.

**Re-tune** (sm_120, GPUs 0, 2, 3; not run). A fresh ledger root, so no old-axis row survives into
the tables; the committed ledger stays in git history:

    L=$HOME/batchlas-tuning/ladder-sm120/ledger; mkdir -p $L
    batchlas_tune all --tier deep --dtype float,cfloat --devices 0,2,3 --ledger $L --plan
    batchlas_tune all --tier deep --dtype float,cfloat --devices 0,2,3 --ledger $L \
        --guard-wait 86400 --progress-fd 3 3>events.jsonl
    cp $L/*.sm_120.jsonl benchmarks/results/tuning/ledger/
    python3 scripts/sweep_to_table.py --device sm_120

The plan step prints the totals (empty ledger, float and cfloat: 12883 cells, 4.85 h lattice, 11.0 h with refinement). FP64 is skipped on 1:64
parts (agent guide §10); add `double,cdouble` for a full run (19.9 GPU-h with all four). The copied
ledgers keep the path `TunedTables.Sm120TablesAreDeepLedgerTablesNamingTheirLedger` checks, but lose
the double and cdouble rows unless those were re-tuned too. Then `tune_replay --slowdown --table
tuned/<op>.sm_120.txt --ledger benchmarks/results/tuning/ledger/<op>.sm_120.jsonl` checks the new rows
with the driver's own scheme.

A ladder re-tune on sm_89 turns these `tuned_tables_tests` checks red, because they require exactly the
`choice.hh` grid_batch rows: `PosvSm89TablesHoldExactlyTheChoiceGrid`,
`TrsmTranscribedTablesHoldExactlyTheChoiceGrid`, `GemmTranscribedTablesHoldExactlyTheChoiceGrid`,
`GemvTranscribedTablesHoldExactlyTheChoiceGrid`, `OrmqrTablesHoldExactlyTheChoiceGridOnBothDevices`,
`GetrfTablesHoldExactlyTheChoiceGridOnBothDevices`, `GetrsTranscribedTablesHoldExactlyTheChoiceGrid`,
`GetriTranscribedTablesHoldExactlyTheChoiceGrid`, `SpmmTranscribedTablesHoldExactlyTheChoiceGrid`
(its CPU table too, if re-tuned) and `SyevTablesHoldExactlyTheChoiceGridOnBothDevices`. Relax each to
"its keys cover the run's ledger grid" in the same change. Any sm_89 level-3 re-tune (either axis, now with the
65536 and 131072 rows) does the same to `SyrkTranscribedTable.RowsAreExactlyTheChoiceGridOnSm89`,
`Syr2kTranscribedTable.RowsAreTheGridAndHoldTheTranscribedRule`,
`SymmTranscribedTable.HoldsTheChoiceGridAndTheTranscribedRule` and
`TrmmTranscribedTable.RowsHoldTheChoiceGridAndTheTranscribedPreference`.

## Engine: matrix-dimension cap

`--max-dim N` (default 2048 for tiered runs, off for custom, 0 = off): no cell is measured with a
matrix dimension above N, whatever its bytes. `OpSpec::dims` returns the extents a cell allocates
(default: every integer key but `batch`; geqrf overrides it, its `m` derives from `form`, `n` and
`aspect`). `plan_round` plans a larger cell as `skip:dim` before the byte cap, so refinement never
measures one either.

## Engine: dominance carry-forward removed

Every arm is timed at every cell. A ledger record that still holds a `dominated:<key>` skip is
*partly stale*: the skipped arm's family counts as stale whatever its hash, so the next run re-races
exactly those arms against the stored winner and runner-up at the stored tier and writes a merged
record (`stale_candidates`, `ledger.cc`; `stale_families`, `sweep_to_table.py`; fixture
`tests/data/ledger_dominated`).

| Alternative | Result | Verdict |
| --- | --- | --- |
| Skip an arm eliminated at > 10x the winner at every larger cell of the run, op and dtype (`dominated:<key>`) | fired thousands of times in the sm_120 deep run (`trsm cdouble vendor` 569 cells, `symm double vendor` 660; also gemm, syrk, syr2k, trmm, geqrf); hid vendor/native crossovers, where the vendor loses on launch overhead small and wins large; "larger" crossed `aspect` and other keys | removed 2026-10-08; its motivation (a minutes-long geqrf vendor cell) is covered by the dimension cap |

## Engine: the guard reads NVML, falls back to nvidia-smi {#engine-guard-retries-a-failed-nvidia-smi-query}

**Backend** (`--guard-backend auto|nvml|nvidia-smi`). `auto` loads `libnvidia-ml.so.1` with
`dlopen` in the driver (`nvml_guard.cc`; structs declared locally, no `nvml.h`). NVML creates no CUDA
context and ignores `CUDA_VISIBLE_DEVICES`; its index is nvidia-smi's `--id`, and preflight dies if
the UUIDs differ. If NVML does not load, `auto` warns and uses nvidia-smi; `nvml` refuses to start.
A query that fails mid-run warns once and switches the run to nvidia-smi. The run line records
`guard` (`nvml`, `nvidia-smi`, `off`; `nvml->nvidia-smi` after a fallback). `batchlas_tune
--guard-probe` prints both backends' view of every GPU, read-only, and exits 1 on a disagreement.

**Activity.** NVML samples SM load per process (`nvmlDeviceGetProcessUtilization`). The driver drops
every live worker, and each exited child or worker for 10 s (NVML's sample lag; pids get reused).
`foreign_active` (`tune_core.hh`) flags a tolerated pid or the whole GPU above `--util-ceiling`, and
any other pid at any SM load.

| Check | NVML | nvidia-smi |
| --- | --- | --- |
| before a cell or child | compute pids, foreign activity in the last 1 s | compute pids; whole-GPU utilization at a full guard (after a 1 s idle on a worker, every 60 s) |
| after it | new pids, foreign activity since its start | new pids |
| `ph_guard` per worker cell (one potrf coarse smoke, GPU 2; an observation) | 0.002-0.003 s | 0.09-0.10 s, plus the 1 s idle per minute |

- **Discards.** A tolerated pid that becomes active during a cell discards it (`rerun_discarded`,
  `tune_core.hh`): re-run at once; after a second discard each re-run first idles 1 s and waits for
  a quiet GPU; after `--guard-wait` the run stops. A discard is never an arm outcome. NVML reports a
  process's last sample period only, so a burst that ended before the after-check can escape.
- **Retries.** `nvidia-smi` fails for every GPU for a moment during a GPU reset (it killed one start
  of the sm_120 deep run). A failed nvidia-smi query is retried after 5 s, doubling to 60 s, until
  `--guard-wait` (default 300 s) has passed, one stderr line per retry; then the run stops
  (`query_with_backoff`, `tune_core.hh`).

## Engine: repeat crashers are dropped

An arm benched for confirmed worker errors (worker and fresh child both `error` in two consecutive
cells) races alone in a fresh child per cell. After 5 consecutive errors there (`kDropAfterErrors`,
`worker.hh`) it is dropped for the rest of that op and dtype in the run: `error`, reason `dropped
after 5 consecutive errors`, not run. One clean cell resets the count. Case: the cdouble gemv
vendor segfaults inside cuBLASLt on every shape on sm_120 (known defect 13), costing two worker
crashes and a retry per cell, about +3 h on a deep gemv cdouble run.

## Engine: op order over all 19 ops

A multi-op run measures op by op in a stable topological order of `op_dependencies()`
(`schedule.cc:292`): an op follows every op whose table its timings depend on through a public entry
point; input order breaks ties. `all` starts from `canonical_op_order()`:

gemm, trsm, syr2k, gemv, spmm, trmm, symm, syrk, potrf, getrf, getrs, geqrf, ormqr, getri, posv,
gesv, orgqr, syev, gesvd.

| Op | After |
| --- | --- |
| trsm, trmm, symm | gemm |
| potrf, getrf | gemm, trsm |
| getrs | trsm |
| geqrf | gemm |
| ormqr | gemm, trmm |
| getri | trsm, getrf |
| posv | potrf, trsm |
| gesv | getrf, getrs |
| orgqr | ormqr |
| syev | gemm, trmm, syr2k, geqrf, ormqr |
| gesvd | gemm, trmm, syr2k |

- A dependency not in the run holds nothing back.
- Hashes do not follow a dependency's *table*: retuning gemm changes trmm's `expand` times without
  staling trmm's cells (open, [open risks](#open-risks)).
- In a multi-op run an (op, dtype) pair the spec refuses (symm, syrk, syr2k are real-only) is
  skipped with one `== note: skipping <op> <dtype>: ...` line; a single-op run refuses it.

## Engine: pipelined cell queues across op and dtype jobs

- Each (op, dtype) job's round is one queue sorted by (per-item footprint, bytes). One thread per
  GPU (`gpu_loop`, `tiered_driver.cc`) pops the next cell of the earliest job, in op order, that has
  queued cells. A GPU never waits behind another job's straggler.
- Each GPU still runs ascending footprints within a job's round; the carve-out restart rule is
  unchanged. Switching job usually costs a restart, counted in the summary and the `schedule`
  progress event.
- A job's next round is planned by the thread that records its last cell, never earlier, because
  refinement reads that round's results.
- Op-order dependencies hold: during a run a composed op times against the library's embedded
  tables, not the ledger.
- Rejected: per-round round-robin sharding joined at each (op, dtype) round; one GPU's tail of large
  cells left the others idle, hours over 19 ops × 4 dtypes.
- Tests: `TuneTieredPipeline.*` (no wait behind a straggler, ascending footprints per GPU and round,
  planning only after the round's last record, records identical to a one-GPU run).

## Engine: the ld audit

Every cell times and verifies at one leading dimension (`ld_pad`, normally 0), so a candidate that
derives `ld` from `n` would verify, win and ship. The ld audit catches it.

- **Rule.** Per (op, dtype) in a tiered run, every candidate's *first verified cell* (`ok`, or
  `eliminated` after passing verification) gets one extra verification at `ld_pad + 3`
  (`kLdAuditPad`, `tools/tune/spec.hh:41`), in the same worker or child, straight after the timed
  cell, as a `verify`-mode cell (a fresh problem at the padded `ld`). An arm the race left
  unverified waits. `LdAuditBook` (`schedule.hh`) records verdicts; a candidate not verified, or
  whose pin refused the padded shape (`skipped`), stays wanted for its next cell.
- **On a failure** (`bad`, `error` or a throw) the arm is `bad` in that cell, reason
  `ld audit: ld_pad +3: ...`, and in every cell of that (op, dtype): later cells and cells running on
  other GPUs record it `bad` (`ld audit failed @ <key>`) untimed (`LdAuditBook::bad`); every current
  record that ranks it, this run's or the ledger's, is appended again with it `bad` and out of
  `ranked` (`ld_retract`, `schedule.hh`). The newer line wins in `best_records` and
  `sweep_to_table.py`.
- **Reporting.** `ld audit FAILED`, `ld audit: <arm> marked bad in N recorded cells`; events
  `ld_audit`, `ld_retract`, `ld_audit_summary` (passed, skipped, failed per op and dtype; `skipped` =
  pin refused the padded shape at every cell that asked). The carve-out audit ignores an arm that
  failed the ld audit (its fresh child verifies at the natural `ld`).
- **Limits.** The batch stride is padded only where a spec derives it from `ld` (most use
  `ld * cols`). gemm's `packed` layout ignores `ld_pad`. A spec that takes no `ld_pad` audits
  nothing. Once per candidate, not a sweep: a defect at some shapes only can still get through.
- **Evidence.** Host tests `TuneLdAudit.*` (child-side fold, book, request/outcome round trip, the
  first-cell rule and reporting, carve-out interplay), each with its own deliberate break. GPU
  check (threadripper02, GPU 2): with vendor syev's run calls passing `rows()` as `lda`, a preview
  `syev float jobz=V n=12:18 batch=2:256` sweep reported `ld audit FAILED: vendor ... bad residual`
  on its first cell and `5 passed, 0 skipped, 1 failed`; without the break, `6 passed, 0 skipped,
  0 failed`. The suspected syev `vendor` ld defect in the sm_120 deep run was not real
  (@ref design_known_defects, the LOBPCG entry).

## Engine: re-measuring named cells

`--remeasure-keys FILE`: one `<op> <dtype> <key>` line per cell, the key in the ledger's
`name=value,...` form and field order.

- Each named cell counts as missing whatever the ledger holds: `plan_round` gives it reason
  `remeasure`, every runnable candidate, the running tier and a fresh record (not a merge). The
  stored `round` carries over, so a re-measured midpoint stays a refinement record for the cap and
  margin.
- Named cells off the lattice are appended to round 0, which is never budgeted: every other cell
  round 0 plans also runs (with stale records, the whole stale lattice, whatever `--budget`).
- `--only-named`: round 0 is exactly the named cells at the run's tier, no lattice, no refinement.
  Tier `custom` always behaves so.
- A named cell is measured past `--max-dim` but not past `--cap-gib`. A key whose fields do not
  match the op's, or an op and dtype the run does not tune, stops the run.
- The newer record wins in `best_records` (same tier, later date or run id), so tables pick up the
  re-measure with no other change.
- Tests: `TuneRemeasure.*` (breaks of the planner, the round-0 append and `--only-named`).

## Engine: the sm_120 deep run

> **Status:** measured 2026-10-08/09 on threadripper02 (4x RTX PRO 6000 Blackwell, sm_120,
> CUDA 13.2, DPC++ `/opt/dpcpp-cuda`), GPUs 0-3, on a box shared with another user. Ledgers:
> `benchmarks/results/tuning/ledger/*.sm_120.jsonl`.

`batchlas_tune all --tier deep --dtype float,double,cfloat,cdouble --devices 0,1,2,3`: all 19 ops,
70 (op, dtype) jobs, matrix dimensions up to 2048, 2026-10-08 06:23 to 2026-10-09 04:54 CEST.

| Start or follow-up | Duration | Cell records | Ended by |
| --- | --- | --- | --- |
| 1 | 11.5 h | 56,144 | a geqrf cell with an 8192-wide matrix timed out its worker |
| 2 | 3 min | 345 | `nvidia-smi` failed during a GPU reset |
| 3 | 4.2 h | 32,912 | |
| 4 | 4.9 h | 35,323 | stopped by the maintainer |
| re-measure (2026-10-09) | 2.5 min on 4 GPUs | 250 | cells damaged by the other user's jobs |
| gemv full re-run `20261009T042647-threadripper02-422873` | 33.5 min on 4 GPUs | 8,515 | `-dirty`: every reader drops it |
| gemv deep `20261011T204301-threadripper02-39013` | | 8,558 | |

- The four starts: about 20.6 h and 124,724 cell records; the campaign, the dirty run included,
  133,489. On 2026-10-12 the 19 sm_120 ledgers held 210,505 cell records: 157,444 measured
  (2026-10-11 samples and re-measures included) and 53,061 transcribed.
- **Refinement coverage.** Every job's starting lattice is complete. geqrf, ormqr, getri, posv,
  gesv, orgqr, syev, gesvd and complex getrs got little refinement before the stop (a resumed run
  wanted 4,072 more refinement cells for the 15 ops of the re-measure run alone), so their
  crossovers sit on lattice spacing (open).
- **Interference.** Cells where every candidate stalled (about 2.3 ms or 7-11 ms, more than twice
  both log-neighbours, which agree on the winner), screened over each table's first log key: 83
  strict outliers (over 2x both neighbours) and 252 soft (over 2x their log-interpolated time). The
  250 non-gemv soft cells were re-measured at deep; the screen then found 16 strict and 154 soft,
  153 of them among the re-measured cells. These reproduce, so they are real: mostly the gesvd
  `jacobi` cliff at m = 33, where the sub-group solver stops fitting, and spmm and syrk cells within
  a few microseconds of their neighbours.
- Later deep runs into the same ledgers replaced some tables (gemm sm_120: 22,203 deep rows,
  2026-10-21).

| Found by the run | Fix |
| --- | --- |
| dominance carry-forward hid vendor/native crossovers | removed ("Engine: dominance carry-forward removed") |
| unbounded matrix dimensions (a geqrf vendor cell took many minutes) | "Engine: matrix-dimension cap" |
| GPUs idle at every (op, dtype) round boundary | "Engine: pipelined cell queues across op and dtype jobs" |
| a transient `nvidia-smi` failure killed start 2 | guard retries ("Engine: the guard reads NVML, falls back to nvidia-smi") |
| complex<double> cuBLAS gemv segfaults on every shape | "Engine: repeat crashers are dropped"; `can_run` refuses it (known defect 13) |
| LOBPCG's smallest-end solve returned 0 with the new syev tables | null P columns, not the vendor (known defects, LOBPCG entry); projected solves pinned to the CTA solver at n ≤ 32 ([LOBPCG: the accuracy pin on the projected syev](../perf/syevx.md#lobpcg-the-accuracy-pin-on-the-projected-syev)) |
| single-`ld` verification cannot catch a candidate that derives `ld` from `n` | "Engine: the ld audit" |

## Open: the constants tuner (sub-project 2) {#coverage-interface-for-sub-project-2}

The per-op specs are done (`OpSpec`, `tools/tune/spec.hh`; one `<op>_spec.cc` per op, composing
ops declare their dependencies for op order and hashes). Not implemented:

- A second spec kind whose candidates are values of one `tuning_params.hh` knob, cells the knob's
  n-buckets and dtypes, measured through the `BATCHLAS_TUNE_*` overrides, with a ledger that yields
  the header. A knob with two consumers is tuned on the consumer the maintainer names; the other is
  reported, not retuned.
- Retiring `evaluation/tuning/` (today the source of `tuning_params.hh`, float only, via
  `generate_tuning_header.py`) once its knobs are covered; its `sensitivity.py` axis pruning becomes
  the knob's starting lattice.

## benchviz Tuning tab: interface for sub-project 3

`/tuning` in `benchviz serve` (`benchmarks/benchviz/tuning.py`, `tuning.html`; user guide:
`benchmarks/benchviz/README.md`, "Tuning tab"). It reads ledgers and tables through
`scripts/sweep_to_table.py` (`read_ledger`, best record, staleness, `ledger_rows`, `parse_table`,
`nearest`), so the views show exactly what `sweep_to_table.py` would write next and need no build.

| View | Content |
| --- | --- |
| Status matrix | op × dtype × device. Staleness per kernel family: the header's hash against hash v2 of the sources now, the keys the ledger holds stale per dtype, `stale_kernel_warnings`; per dtype, rows and `behind` when the header does not name the newest run. The row-exact `--check` rule per dtype is in the op view |
| Launch and progress | tiers preview, calibrate, coarse, deep, custom with listed cells (`--remeasure-keys`); `--allow-dirty` on request; `--device-key` when no GPU is visible. Estimate before start; cells and eliminations stream live (SSE); stop and resume use the skip rule. Runs live in `benchviz_runs/_tuning/<run>/` (`run.json`, `events.jsonl`, `run.log`, `exit`, `remeasure-keys.txt`); the tuner's own guard (`--guard-wait`; no foreign context on a measured GPU) replaces `gpu_guard.sh` |
| Decision map | small multiples: columns and rows each facet one key or a pair of exact keys (gemm: columns ta × tb, rows four k slices of the planned lattice). Axes: the run's planned lattice (`grid` of its run line); other keys fixed, by default to the densest *measured* slice, batch at the largest value holding a quarter of the commonest one's cells. An unmeasured cell shows the next table's pick, faded by the weighted log2 distance to its serving row; cells served by an untimed row are hatched. Colourings: winning family, margin (runner-up over winner, 1× to 2×), vs vendor (vendor over best native). Interactive SVG, clickable for every candidate's timing and interval |
| Ranked diff | cells whose pick changes between the shipped table and the rows `sweep_to_table.py` would write now, each pick the first entry the cell's record did not refuse (`skipped`, that is `can_run` false); predicted speedup = ratio of the two picks' medians in that record (a 3% tie keeps the earlier pick, so a change can read as low as 0.971×). Ranked by milliseconds saved; a change whose picks' intervals overlap is marked noise and hidden by default |

- Without the fading, gemm float TT on sm_89 reads as a clean rule when it rests on 10 measured
  cells.
- Observations: against main's tables (2026-10-09), gemm double on sm_120 changes 2918 picks,
  geomean 1.08× where both are timed. On the 2026-10-18 coarse gemm sm_89 run, 2549 float picks
  change, 281 within noise and 1709 not timed at the cell (transcribed); first in the ranking:
  batch-32768 8×8×8 TT moves from cuBLAS to `small` (43×).
- Load time: all 39 ledgers take about 13 s (threadripper02, 2026-10-11); the server warms this at
  start. `ledger_rows` compares a row only with keys in its own and the adjacent
  `floor(index / stride)` buckets: all 70 sm_120 ledger tables byte-identical, 8.4 s instead of
  126.6 s.

## Tiered tuning: open risks and debts {#open-risks}

- Racing assumes timing noise is stationary within a cell; clock ramps after idle gaps break that.
  The worker's warm start and each cell's warm-until-stable are the defence. The replay cannot test
  it: the raw files come from warm, back-to-back runs.
- If the audit fails often, the speedup rests on racing and the adaptive grid alone.
- Not implemented: per-rep `<run id>.reps.jsonl` files for deep runs, and the guard threshold
  (`--util-ceiling`) in `run` records (the guard backend is recorded).
- Hashes do not follow a dependency's table ("Engine: op order over all 19 ops").
- Budgets assume one GPU on a quiet box. On the shared 4090 box, guard waits come on top.
- sm_89 has no measured table: the 2026-10-18 coarse gemm sm_89 run (the benchviz observation
  above) is not in the committed ledger; `tuned/gemm.sm_89.txt` is all `transcribed` on 2026-10-21.
- The calibrate gate is unmeasured ("Engine: calibrate tier from a borrowed table").

## License

SPDX-License-Identifier: MIT
