# Tiered tuning {#design_tiered_tuning}

> **Status:** current · checked 2026-10-29

`batchlas_tune` fills the selection tables in tiers (preview, calibrate, coarse, deep) with one
algorithm per cell, a per-(op, device) ledger in which a cheaper run never displaces a better
current result, and one persistent worker per GPU. Usage and flags: @ref tune_tool_readme. Ledger,
table, tier and staleness rules: @ref design_tuning_data (normative). The tables feed
@ref design_flat_selection §6.

## Tiered tuning: components and state

| Component | State |
| --- | --- |
| Engine: tiers, lean race, flip-frontier planning, calibrate, persistent worker, audits, ledger, table generation | implemented (`tools/tune/`) |
| Coverage: one spec per routed op | implemented (`tools/tune/<op>_spec.cc`, 29 ops) |
| Constants tuner on the engine (`tuning_params.hh`) | **open** ([below](#coverage-interface-for-sub-project-2)) |
| benchviz Tuning tab | built (`benchmarks/benchviz/tuning.py`) |

Measured tables: `tuned/*.sm_120.txt` of the 19 ops in the sm_120 deep run, and 9 sm_89 tables.
The ten ops added since have untimed sm_120 tables (@ref design_tuning_data, "Shipped set").

## Tiered tuning: decisions

| Question | Decision |
| --- | --- |
| Main purpose | Initial fill of every table for a machine; incremental retune second. |
| Speed versus rigor | The user picks a tier per run; `--plan` prices it; more GPUs shard the work. |
| Grid reduction | Stride-2 start plus flip-frontier rounds; gemm, geqrf, getri, orgqr and spmm keep the full lattice ("Engine: round planning by flip frontier"). |
| Precedence | Higher fidelity wins per cell. A lower tier fills gaps and replaces stale results, never current ones. |
| Result store | `benchmarks/results/tuning/ledger/` under Git LFS, so precedence holds across machines. |
| Worker | One persistent worker per GPU plus a fresh-process audit. |
| New machine | `--tier calibrate` probes the nearest shipped table and measures where the device disagrees. |

## Engine: tiers and the per-cell algorithm

Every tier runs the same algorithm, so preview results are low-fidelity data of the same kind as
deep results. Parameters: `params()` in `tools/tune/tier.hh`.

| Parameter | preview | coarse | deep |
| --- | --- | --- | --- |
| Starting lattice | stride 2 of each `choice.hh` axis (batch: the CU-aligned ladder), ends kept, then flip-frontier rounds; stride-1 ops (gemm, geqrf, getri, orgqr, spmm): full lattice at coarse and deep | as preview | as coarse |
| Bisection (stride-1 ops) | index space plus a 10% runner-up margin | geometric to hi/lo < 1.1 | as coarse |
| Refinement cap (x round-0 cells; stride-1 ops only) | 3.0 | 1.0 | 2.0 |
| Race reps floor per candidate | 3 | 4 | 6 |
| Elimination confidence | 0.80 | 0.90 | 0.98, plus a reversed confirmation round |
| Fresh-process audit sample | 2% | 2% | 1% |

- `calibrate` races at coarse quality on a stride-4 probe lattice against a prior table.
- The tie margin is 3% in every tier. Lattices nest (preview ⊂ coarse ⊂ deep); a grid that is not a
  lattice is subsampled by a stable hash of the cell key.

Per cell: **probe** each candidate's pin (`*_buffer_size` under `select::ScopedPin`, or one untimed
run); 0 or 1 runnable candidates record the cell untimed. **Skip** it if a current record at the
same or higher tier exists; a partly stale record re-races only the changed candidates against the
stored winner and runner-up. **Race** ("Engine: the lean cell") and verify the arms that can rank
first ("Engine: tie-set verification").

Across cells: round 0 of every op and dtype completes before refinement, so an interrupted run
leaves usable tables. `log` axes bisect geometrically, `exact` axes (uplo, side, trans) never,
`batch` refills only its own values; nothing is interpolated (`select::choose` takes the nearest
row). The `--plan` estimate takes the recorded phases (`ph_*`) of the nearest ledger cell of the
same protocol, tier and mode, else a model (`--cell-overhead-s` plus per-arm reps × median, plus
0.05 s verify). `--budget <h>` stops refinement; round 0 always completes.

## Engine: the ledger and table generation

- One ledger per (op, device), dtype-tagged lines, appended under `flock`; line kinds, best-record
  rule, dirty runs and kernel hashes: @ref design_tuning_data
  ([the ledger file](tuning-data.md#tuning-data-ledger-file),
  [kernel hashes](tuning-data.md#tuning-data-kernel-hash),
  [generation and the check](tuning-data.md#tuning-data-generation-and-check)).
- A spec's `kernel-sources` block maps each family to its files plus a common set; `src/select/` is
  not hashed. Every spec lists its `src/ops/<op>/<op>.cc` (where `can_run` lives) as common, so an
  edit to what can run stales every cell. `kernel-deps` lines add files outside the source list
  (posv's `cta` and `blocked` list potrf's and trsm's kernels).
- `batchlas_tune --import-raw` turns a schema-1 raw sweep into a deep run; its hashes are
  `legacy:<kernels>` unless the raw op hash equals today's, so an old sweep imports stale.

## Engine: the lean cell

The race of one cell (`run_race`, `tools/tune/cell_runner.hh`; constants in `race.hh`), recorded as
`protocol=lean-cold`: every run restores the inputs and scrubs L2 before the clock starts
(@ref tune_tool_readme; [the cold protocol](../perf/batch-sensitivity.md#batch-sensitivity-the-cold-protocol)).
A measured record of any other protocol (`lean`, `full`, absent) ranks below every `lean-cold`
record of its cell whatever its tier, and the planner re-measures it; `--plan` history reads only
`lean-cold` cells.

| Step | Rule | Constant |
| --- | --- | --- |
| 1. Untimed run per arm | keeps JIT and module loads out of the cut | |
| 2. Gross cut | after two timed rounds, an arm whose lower median paired ratio to the leader exceeds 4 leaves | `kGrossLoserRatio` = 4.0 |
| 3. Warm until stable | untimed rounds until two consecutive totals agree within 2%, 2 to 0.2 s per arm | `kWarmTol`, `kWarmMinRounds`, `kWarmCapPerArmS` |
| 4. Rep cap | ceil(50 ms / slowest survivor's median), clamped to [reps floor, 16] | `kRepBudgetS` = 0.05, `kRepCap` = 16 |
| 5. Verify | the tie set only | "Engine: tie-set verification" |

- NVML is sampled after each timed run; a cell clock-capped in more than 25% of samples is raced
  once more (`kThrottledFraction`, `kThrottleReraces`, `tune_core.hh`).
- The cell runs on the worker's one `Queue`; `~ArmBench` drains it so an async error cannot surface
  in the next cell. Each spec's `clear_info` poisons outputs `reset` does not restore, so an arm
  that writes nothing cannot pass on an earlier arm's output.

## Engine: tie-set verification

A cell verifies only the arms that can rank first plus the arms its bucket has not verified yet
(`verify_next`, `race.hh`; `VerifyBook`, `verified_front`, `schedule.hh`).

- **Tie set:** the unverified arms within 3% of the fastest survivor's median. A failure leaves the
  set and the next arms that can rank first are verified.
- **Once per bucket** = (op, dtype, exact keys, floor(log10 per-item bytes)). Unverified arms stay in
  the row with ledger flag `v<i> = 0`: a pin that falls through needs them.
- **A failure** condemns the kernel in its bucket: later cells record it `bad` untimed, and every
  current record of the bucket ranking it unverified is appended again with it `bad`
  (`verify_retract`). A record's front arm is always a verified one.
- **Fresh audit:** `audit_compare` treats an arm one side left unverified as feasible if it ran there.

Evidence: `TuneVerify.*` in `tests/tune_tests.cc`, one deliberate break per rule.

## Engine: persistent workers and the carve-out audit

Each GPU runs one `batchlas_tune_impl --worker` under its `CUDA_VISIBLE_DEVICES` fence and per-GPU
flock; it warms clocks for 3 s at start (`worker_warm`), each kernel JITs once per worker. A dead
worker is restarted and the cell retried; a sticky CUDA error re-races the cell in a fresh child
("Engine: repeat crashers are dropped").

The SLM carve-out attribute is sticky per CUfunction (`benchmarks/factor_bench.cc` header): an
earlier larger launch can make a later one succeed that fails in a fresh process. Defences:

1. **Ascending order.** A worker runs cells in ascending per-item footprint and is restarted before
   any cell smaller than one it already ran.
2. **Fresh-process audit.** A hashed sample (`audit_fraction`) plus the first worker cell of each
   op and dtype is re-raced in a fresh child. A feasibility mismatch, or a winner more than 10%
   slower fresh, sends the rest of that op and dtype to fresh children.

Checked on threadripper02, 2026-10-07 (`tune_race_gpu_tests`, potrf, posv, trsm at the 48 KB hole):
no mismatch ([potrf: the 48 KB launch hole](../perf/potrf.md#potrf-the-48-kb-launch-hole)).

## Engine: measured per-child overhead

A fresh `batchlas_tune_impl --cell` child costs 0.49 s before any kernel (process, SYCL and CUDA
init, JIT-cache load; threadripper02, sm_120, potrf float n=16 batch=8192, median of 20,
2026-10-07), 37-60% of a raced preview cell. Hence one persistent worker per GPU;
`--cell-overhead-s` defaults to 0.49 (`kChildOverheadS`, `schedule.hh`).

## Engine: driver interface

Driver: `tiered_driver.cc`; planning: `schedule.cc`. Full flag table: @ref tune_tool_readme.
Expert overrides (`--reps`, `--warm`, `--passes`, `--remeasure`, `--refine-ratio`) make the run tier
`custom`, ranked below preview and written only with an explicit `--ledger DIR`. `--plan` and
`--status` need no GPU; `--progress-fd <n>` emits one JSON event per line (benchviz reads it over SSE).

## Engine: refinement convergence rules

`refine_all_axes` (`tools/tune/grid.cc`), shared by the driver and `tune_replay`.

1. **Flip.** A bracket is a flip when its two winners differ and the flip is decisive at *either*
   end: the other end's winner is more than the 3% tie slower there, or cannot run. Requiring it at
   both ends dropped the trsm flips (crossover near one end) and failed every tier in replay.
2. **Margin.** The preview margin (10%) applies only to brackets whose ends are both round-0 cells.
3. **Batch** is never bisected; it refills its own axis values ("Engine: the CU-aligned batch ladder").
4. **Cap.** Without a frontier, refinement cells per op and dtype ≤ `refine_cap_factor` × round-0
   cells, counting the ledger's current records, so a stopped run resumes within the same total.
   Flips refine before margin hedges. A frontier run has no cap.
5. **Estimate.** The ledger's refined / round-0 ratio at the tier, else half the cap factor; a
   frontier run prices `kFrontierShare` = 0.86 of the grid cells round 0 leaves.

## Engine: replay results on trsm

`tune_replay --raw <sweep> --tier <t>` (`tools/tune/replay.cc`) replays a stride-1 tier's racing and
bisection against the exhaustive trsm sweeps (4452 float, 4098 double cells), no GPU. Score: table
misrank (nearest row's pick > 3% slower than the exhaustive best), as excess over the pass-1/pass-2
noise floor (~1.5%). Bounds: deep 0.2%, coarse 1%, preview 5%; all six tier × dtype lines pass
(worst: preview double 4.25%). Same-sample table misrank, float / double:

| Alternative | Result | Verdict |
| --- | --- | --- |
| Full lattice, geometric bisection to 1.1 | 0.11% / 0.05% | coarse, deep (stride-1 ops) |
| Full lattice, geometric bisection to 1.25 | 2.5% / 1.4% | rejected |
| Full lattice, no bisection | 6.4% / 6.1% | rejected |
| Stride 2, index bisection, margin 0.10 | 3.6% / 4.3% | preview (stride-1 ops) |
| Stride 2, index bisection, no margin | 6.7% / 5.8% | rejected |
| Stride 4, any refinement | 7-13% | rejected |
| Shrunk trsm grids (batch, q or order at stride 2) | 1.5-13.5%, none under 1% | rejected; batch is the costliest axis to shrink |

## Engine: flip frontier and the slowdown replay

**Rule** (`flip_frontier`, `tools/tune/grid.cc`). After a stride-2 start, an unmeasured cell is
measured when its k = 6 nearest measured cells (own exact-key group, `select::choose`'s distance via
`NearestIndex`) hold a decisive flip: one neighbour's winner is more than 5% slower than the best at
another, or cannot run there. Rounds repeat until none is left.

**Slowdown replay** (`tune_replay --slowdown`, `slowdown_replay` in `replay_core.cc`). Universe:
every timed row of a shipped table (`--table`), current ledger record (`--ledger`) or raw sweep cell
(`--raw`). Score per cell: t(pick) / t(fastest arm), the pick being the nearest measured row's first
entry that runs there. Gate: no failures, geomean ≤ 0.05%, p99.9 ≤ 1.10. Truth is the shipped
medians; race noise and tie order are not modelled.

Gate run, 2026-10-16, threadripper02: the 19 `tuned/*.sm_120.txt` tables with their ledgers
(115,595 cells, `ledger` start lattice).

| Scheme | Cells kept | Arm runs | Geomean | p99.9 | Worst | Failures |
| --- | --- | --- | --- | --- | --- | --- |
| lattice (round 0, stride 1) | 59% | 47% | 3.079% | 4.241 | 16.92x | 0 |
| stride2 | 12% | 10% | 5.129% | 5.237 | 42.69x | 6 |
| **stride2+frontier** | 76% | 82% | 0.044% | 1.072 | 5.28x | 0 |
| stride4+frontier | 82% | 88% | 0.073% | 1.234 | 5.28x | 0 |
| stride2+frontier+prune (arms within 1.5x) | 76% | 63% | 0.074% | 1.255 | 5.28x | 0 |

`stride2+frontier` passes overall; per op it fails for gemm (0.110%, p99.9 1.339) and symm (0.112%,
1.851), and on p99.9 only for geqrf, getri, orgqr, spmm and trmm. Prune is a replay scheme only.
The sm_120 gemm table was regenerated on 2026-10-21 and not re-replayed.

## Engine: round planning by flip frontier

Preview, coarse and deep runs plan rounds by the flip frontier (`TieredRun::build` and `refine`,
`tiered_driver.cc`); preview's index refill, margin and cap are off in frontier mode. Tests:
`TuneTieredDriver.Frontier*`, `TuneTieredDriver.Preview*`, `TuneFrontier.Start*`,
`TuneReplay.BisectSchemes*`.

- **Round 0** (`frontier_start`, shared with the replay): grid cells on `tier_lattice(axes, 2)`;
  an exact-key group with none takes every second value of its own, ends kept. Partly stale grid
  cells join round 0.
- **Frontier rounds** pick grid cells whose k nearest measured cells (this run's and the ledger's
  current records at the same or a higher tier) hold a decisive flip. **Bisection** (3% tie, index
  mode) adds off-grid midpoints only. Stop when neither adds a cell; `--budget` stops rounds after
  round 0.

Per op (`OpSpec::frontier()` returns a `FrontierPlan`, `spec.hh`; default stride 2, k 6, 5%): the
cheapest variant that passes the per-op gate in the slowdown replay, else the full lattice.

| Op | Plan | Why |
| --- | --- | --- |
| gemm | stride 1 | no variant passes: best searched, stride 2 k 10, geomean 0.089% |
| geqrf, getri, orgqr, spmm | stride 1 | p99.9 fails at k 6, 8 and 10 on the same worst cells |
| symm | stride 2, k 8 | k 8: 0.000% / 1.000 (70.6% kept); k 6: 0.112% / 1.851 |
| trmm | stride 2, k 8 | passes at k 8, fails at k 6 |
| every other op | default | passes |

Driver-model replay (`stride2+frontier+bisect` against `lattice+bisect`, 2026-10-16): the frontier
measures about 30% fewer cells for +0.17% pooled geomean; raw trsm +0.005% float, +0.022% double.

## Engine: calibrate tier from a borrowed table

`--tier calibrate` fills a new device's tables from the nearest shipped table (the prior): it
probes the prior on the device and measures only where the device disagrees. Code: `CalibrateRound`
(`grid.cc`), `prior_table`, `calibrate_replay` (`replay_core.cc`), `copy_prior`
(`tiered_driver.cc`). Tests: `TuneCalibrate.*`, `sweep_to_table.py --self-test` ("prior rows").

| Part | Rule |
| --- | --- |
| Prior | `--prior <device>`, else `prior_table`: among the other non-cpu devices' tables, the newest `date=` among those ≥ 90% timed; else the larger timed share, then newest, then name. A cell's pick is the prior's nearest row's first current candidate |
| Probe (round 0) | `frontier_start(grid, axes, 4)` (`--probe-stride`), every arm at coarse quality. A probe disagrees when the device's winner is more than 5% faster than the prior's pick, or the pick cannot run |
| Refinement | open cells whose prior pick is decisively beaten at one of their 6 nearest measured cells that disagrees; bisection only next to a disagreeing cell |
| Arms off the probes | the prior's top 2, any family the prior never names, the winners of the 6 nearest measured cells; the rest `skipped`, reason `unraced:calibrate` (never stale). `--only-named` is refused |
| Prior rows | at job end every unmeasured, uncontradicted grid cell gets the prior's row as a `prior` record (untimed, never stale, below every measured tier); without them the device's sparse own table would shadow the borrowed one (`select::detail::walk` reads it first) |

Replay (2026-10-16, sm_120 tables as universe, sm_89 tables as prior; a weak prior, mostly untimed):

| Scheme | Measured | Arm runs | Geomean | p99.9 |
| --- | --- | --- | --- | --- |
| `prior` (borrow blindly) | 0% | 0% | 19.8% | 451 |
| `calibrate` (stride 4) | 80.7% | 61.3% | 0.291% | 1.770 |
| `calibrate8` | 70.1% | 51.1% | 1.47% | 10.1 |
| `stride2+frontier` from scratch | 76.1% | 81.8% | 0.044% | 1.072 |
| `calibrate`, prior = sm_120 itself | 2.9% | 2.1% | 0.132% | 1.030 |

Requiring a device flip next to a contradicted cell as well: 76.2% measured, 0.39%: rejected.
**Open gate:** calibrate pays off only with a prior close to the device; no real second-device
table exists to test that.

## Engine: strict pins and gross losers {#engine-readiness-for-a-deep-run-over-all-19-ops}

- **Strict pins.** The tuner pins with `select::ScopedPin(op, word, select::StrictPin{})`
  (`cell_runner.hh`): a class word nothing in its class can serve throws, and `vendor` must resolve
  to the candidate spelled `vendor`, else the arm is `skipped`, `pin refused (strict)`. Without it a
  `vendor` arm where `can_run` refuses timed Auto under the vendor's name (spmm, gesvd). Library
  callers and `BATCHLAS_<OP>_ROUTE` keep the fallback. gesvd also keeps a spec-level refusal (known
  defect 15).
- **Gross losers.** From round 2 the race drops an arm whose lower median paired ratio to the leader
  exceeds 4.0 (`kGrossLoserRatio`), recorded `eliminated`; one stalled round never kills an arm. Case:
  geqrf's vendor is tens to hundreds of times slower than blocked on large cells.

## Engine: the CU-aligned batch ladder

**Rule.** A tiered run can replace an op's `choice.hh` batch axis with `batch_ladder(cus, classes,
extras)` (`grid.cc`): max(1, round(c × CUs)) per class c, plus the spec's `ladder_extra_batches()`.
Default classes 0.25, 2, 16, 128 (`default_batch_classes`): latency, about one wave, saturated. sm_120
(188 SMs): 47, 376, 3008, 24064; RTX 4090 (128): 32, 256, 2048, 16384. The key stays `batch:log`;
the run's `grid` field records the ladder. Batch is still never bisected.

- **`--batch-ladder auto` (default)** applies it only to an (op, dtype) whose ledger has no run or a
  ladder run (`ladder_applies`); on a ledger of `choice.hh`-axis runs it prints `batch ladder off`.
  `default` or `c,c,...` force it; `off`, `--batches` or `--grid batch=...` keep the `choice.hh` axis.
  `--tier custom` and `--only-named` measure named keys only.
- **Forced on an old ledger**, old-axis rows are never retired (`sweep_to_table.py` writes every
  non-stale record), so the old rows keep governing their neighbourhoods: a ladder re-tune starts from
  a fresh ledger root.
- **Compute units**: a measuring run reads `MAX_COMPUTE_UNITS` on the first `--devices` GPU. `--plan`
  takes `--cus N`, else a name table (RTX PRO 6000 Blackwell 188, RTX 4090 128), else refuses.
- **Grids that are not lattices** (gemm, syrk, symm, ormqr): `rebatch` gives each declared grid line
  the ladder; a line declared at one batch keeps it.

| Op | Hook | Reason |
| --- | --- | --- |
| gemm, ormqr | no latency class (`without_latency_class`) | the axis starts at 128; gemm is half of all round-0 cells |
| syrk, symm | extras 1, 2, 4, 8 | vendor-loop/native crossovers at batch 1-8 |
| syr2k | extras 1, 2, 3 | the vendor loop wins batch 1 only |
| trmm | extra 1 | below 128 the ledger holds batch 1 only |
| the rest | none | batch < 128 is out of scope (agent guide §10) |

`TuneSpecLadder.RealSpecHooksGiveTheDocumentedLadders` (`tune_race_gpu_tests`) pins this table.

**Batches past the grid-y cap.** Level-3 runs a batch past 65535 in chunks
([gemm batch chunking](../perf/gemm.md#gemm-batch-chunking)), so a table without a row past 32768
extends that row to any batch. `with_slice_batches` (`grid.cc`) adds 65536 and 131072 to the gemm,
syrk, symm, syr2k and trmm extras and `choice.hh`-axis grids (`choice.hh` itself unchanged, no hash
moves); `TuneSpecLadder.Level3OldAxisGridsReachPastTheGridYCap` pins it. At the 4 GiB `--cap-gib`
65536 admits only 64 KiB per item, so the large shapes need `--cap-gib 32`.

**Evidence** (sm_120, 2026-10-21). Empty-ledger `all --tier deep --plan`, four dtypes: round-0 cells
27061 → 24120, 20.1 → 19.9 GPU-h. Held-out replay on the committed ledgers (a lower bound): the
ladder costs 0.2-0.9 points of geomean at batch ≥ 128. A getri and potrf float coarse A/B on GPU 2:
getri no difference, potrf 0.3-0.4 points worse. Class sets, geomean on 8 ops at batch ≥ 32:

| Classes | Geomean range | Round-0 cells / GPU-h | Verdict |
| --- | --- | ---: | --- |
| **0.25, 2, 16, 128** | 0.13-1.50% | 24120 / 19.9 | default |
| 0.25, 2, 8, 32, 128 | 0-1.02% | 28789 / 25.8 | rejected: more cells than the `choice.hh` axis (27061 / 20.1) |
| 0.25, 1, 8, 64 | 0.26-2.75% | 27375 / 21.9 | rejected: no top class near 32768, small-n CTA cells lose up to 3.7x |
| 0.25, 4, 64 | 0.42-2.35% | 17835 / 14.7 | cheap option, `--batch-ladder 0.25,4,64` |

**Re-tune** (sm_120; not run, no shipped table has ladder rows yet):

    L=$HOME/batchlas-tuning/ladder-sm120/ledger; mkdir -p $L
    batchlas_tune all --tier deep --dtype float,cfloat --devices 0,2,3 --ledger $L --plan
    batchlas_tune all --tier deep --dtype float,cfloat --devices 0,2,3 --ledger $L \
        --guard-wait 86400 --progress-fd 3 3>events.jsonl
    cp $L/*.sm_120.jsonl benchmarks/results/tuning/ledger/
    python3 scripts/sweep_to_table.py --device sm_120

FP64 is skipped on 1:64 parts (agent guide §10); the copied ledgers lose the double and cdouble rows
unless those are re-tuned too. Then check with `tune_replay --slowdown --table ... --ledger ...`. A
ladder re-tune on sm_89 (or any sm_89 level-3 re-tune) turns red the `tuned_tables_tests` that
require exactly the `choice.hh` grid (`*HoldExactlyTheChoiceGrid*`, `*RowsAreExactlyTheChoiceGrid*`
and the level-3 `*TranscribedTable*` grid checks): relax each to "its keys cover the run's ledger
grid" in the same change.

## Engine: matrix-dimension cap

`--max-dim N` (default 2048 for tiered runs, off for custom, 0 = off): no cell is measured with a
matrix dimension above N. `OpSpec::dims` returns the extents a cell allocates (geqrf overrides it);
`plan_round` plans a larger cell as `skip:dim` before the byte cap.

## Engine: dominance carry-forward removed

Every arm is timed at every cell. A ledger record that still holds a `dominated:<key>` skip is
partly stale, so the next run re-races exactly those arms (`stale_candidates`, `ledger.cc`;
`stale_families`, `sweep_to_table.py`; fixture `tests/data/ledger_dominated`).

| Alternative | Result | Verdict |
| --- | --- | --- |
| Skip an arm eliminated at > 10x the winner at every larger cell | hid vendor/native crossovers in the sm_120 deep run (vendor loses small on launch overhead, wins large) | removed 2026-10-08; the dimension cap covers its motivation |

## Engine: the guard reads NVML, falls back to nvidia-smi {#engine-guard-retries-a-failed-nvidia-smi-query}

`--guard-backend auto|nvml|nvidia-smi`. `auto` `dlopen`s `libnvidia-ml.so.1` (`nvml_guard.cc`), which
creates no CUDA context; if it does not load, `auto` warns and uses nvidia-smi, `nvml` refuses. A
query that fails mid-run switches the run to nvidia-smi. The run line records `guard`.
`--guard-probe` prints both backends' view of every GPU.

- **Activity** (`foreign_active`, `tune_core.hh`): NVML samples SM load per process; a tolerated pid
  or the whole GPU above `--util-ceiling` (default 5), or any other pid at any load, is foreign.
- **Discards** (`rerun_discarded`): foreign activity during a cell discards it; it re-runs, after a
  second discard waiting for a quiet GPU, and the run stops after `--guard-wait`.
- **Retries** (`query_with_backoff`): a failed nvidia-smi query (it fails for every GPU during a GPU
  reset, which killed one start of the sm_120 deep run) retries after 5 s, doubling to 60 s, until
  `--guard-wait` (default 300 s), then the run stops.

## Engine: repeat crashers are dropped

An arm benched for confirmed worker errors races alone in a fresh child per cell; after 5
consecutive errors there (`kDropAfterErrors`, `worker.hh`) it is dropped for the rest of that op and
dtype (`error`, `dropped after 5 consecutive errors`). One clean cell resets the count. Case: the
cdouble gemv vendor segfaults on every shape on sm_120 (known defect 13).

## Engine: op order {#engine-op-order-over-all-19-ops}

A multi-op run measures ops in a stable topological order of `op_dependencies()` (`schedule.cc`): an
op follows every op whose table its timings depend on. `all` starts from `canonical_op_order()`
(gemm, trsm, syr2k, gemv, spmm, trmm, symm, syrk, potrf, getrf, getrs, geqrf, ormqr, getri, posv,
gesv, orgqr, syev, gesvd, hemm, herk, her2k); ops not listed follow in registration order.

- A dependency not in the run holds nothing back.
- Hashes do not follow a dependency's table: retuning gemm changes trmm's `expand` times without
  staling trmm's cells ([open risks](#open-risks)).
- A multi-op run skips an (op, dtype) the spec refuses with `== note: skipping <op> <dtype>: ...`;
  a single-op run refuses it.

## Engine: pipelined cell queues across op and dtype jobs

Each (op, dtype) job's round is one queue sorted by per-item footprint. One thread per GPU
(`gpu_loop`, `tiered_driver.cc`) pops the next cell of the earliest job in op order that has queued
cells, so a GPU never waits behind another job's straggler. A job's next round is planned by the
thread that records its last cell. Rejected: round-robin sharding joined at each round (one GPU's
tail left the others idle for hours). Tests: `TuneTieredPipeline.*`.

## Engine: the ld audit

Every cell times and verifies at one leading dimension, so a candidate that derives `ld` from `n`
would verify, win and ship.

- **Rule.** Per (op, dtype), every candidate's first verified cell gets one extra verification at
  `ld_pad + 3` (`kLdAuditPad`, `spec.hh`) in the same worker; `LdAuditBook` (`schedule.hh`) records
  verdicts.
- **On a failure** the arm is `bad` in every cell of that (op, dtype), and every current record
  ranking it is appended again with it `bad` (`ld_retract`). Events `ld_audit`, `ld_retract`,
  `ld_audit_summary`.
- **Limits.** Once per candidate, not a sweep. The batch stride is padded only where a spec derives
  it from `ld`; gemm's `packed` layout ignores `ld_pad`; a spec without `ld_pad` audits nothing.
- **Evidence.** `TuneLdAudit.*`; on GPU a deliberately broken vendor syev `lda` was caught on its
  first cell (threadripper02).

## Engine: re-measuring named cells

`--remeasure-keys FILE`: one `<op> <dtype> <key>` line per cell, the key in the ledger's
`name=value,...` form and field order.

- Each named cell counts as missing: reason `remeasure`, every runnable candidate, the running tier,
  a fresh record. The stored `round` carries over.
- Named cells off the lattice join round 0, which is never budgeted. `--only-named` (always, for
  tier `custom`) makes round 0 exactly the named cells.
- A named cell is measured past `--max-dim` but not past `--cap-gib`. A malformed key, or an op and
  dtype the run does not tune, stops the run. The newer record wins in `best_records`.
- Tests: `TuneRemeasure.*`.

## Engine: the sm_120 deep run

`batchlas_tune all --tier deep --dtype float,double,cfloat,cdouble --devices 0,1,2,3` on
threadripper02 (4x RTX PRO 6000 Blackwell), 2026-10-08/09: the 19 ops then specified, 70 jobs, four
starts totalling about 20.6 h and 124,724 cell records, plus re-measures of 250 cells damaged by
another user's jobs. Ledgers: `benchmarks/results/tuning/ledger/*.sm_120.jsonl`.

- geqrf, ormqr, getri, posv, gesv, orgqr, syev, gesvd and complex getrs got little refinement
  before the stop, so their crossovers sit on lattice spacing (open).
- Remaining stall outliers reproduce (mostly the gesvd `jacobi` cliff at m = 33), so they are real.

| Found by the run | Fix |
| --- | --- |
| dominance carry-forward hid crossovers | "Engine: dominance carry-forward removed" |
| a geqrf cell with an 8192-wide matrix timed out its worker | "Engine: matrix-dimension cap" |
| GPUs idle at every (op, dtype) round boundary | "Engine: pipelined cell queues across op and dtype jobs" |
| a transient `nvidia-smi` failure killed a start | "Engine: the guard reads NVML, falls back to nvidia-smi" |
| cdouble cuBLAS gemv segfaults | "Engine: repeat crashers are dropped"; `can_run` refuses it (known defect 13) |
| LOBPCG's smallest-end solve failed with the new syev tables | null P columns ([LOBPCG: the accuracy pin](../perf/syevx.md#lobpcg-the-accuracy-pin-on-the-projected-syev)) |
| single-`ld` verification misses an `ld`-from-`n` candidate | "Engine: the ld audit" |

## Open: the constants tuner (sub-project 2) {#coverage-interface-for-sub-project-2}

Not implemented: a second spec kind whose candidates are values of one `tuning_params.hh` knob,
cells its n-buckets and dtypes, measured through the `BATCHLAS_TUNE_*` overrides, with a ledger that
yields the header (a knob with two consumers is tuned on the one the maintainer names). Then retire
`evaluation/tuning/` (today's float-only source of `tuning_params.hh`); its `sensitivity.py` axis
pruning becomes the knob's starting lattice.

## benchviz Tuning tab: interface for sub-project 3

`/tuning` in `benchviz serve` (`benchmarks/benchviz/tuning.py`, `tuning.html`; user guide:
`benchmarks/benchviz/README.md`, "Tuning tab"). It reads ledgers and tables through
`scripts/sweep_to_table.py`, so the views show exactly what `sweep_to_table.py` would write next.

| View | Content |
| --- | --- |
| Status matrix | op × dtype × device; staleness per kernel family, stale keys per dtype, `behind` when the header does not name the newest run |
| Launch and progress | any tier, listed cells (`--remeasure-keys`); estimate first, cells stream over SSE; runs in `benchviz_runs/_tuning/<run>/`; the tuner's own guard replaces `gpu_guard.sh` |
| Decision map | small multiples over the planned lattice; an unmeasured cell shows the next table's pick, faded by distance to its serving row (without it gemm float TT on sm_89 reads as a clean rule on 10 measured cells) |
| Ranked diff | cells whose pick changes between the shipped table and the next one, ranked by milliseconds saved; overlapping intervals are marked noise |

`ledger_rows` compares a row only with keys in its own and adjacent `floor(index / stride)` buckets:
byte-identical output, 8.4 s instead of 126.6 s over the 70 sm_120 ledger tables (2026-10-11).

## Tiered tuning: open risks and debts {#open-risks}

- Racing assumes timing noise is stationary within a cell; clock ramps after idle gaps break that.
  The worker's warm start and warm-until-stable are the defence; the replay cannot test it.
- If the audit fails often, the speedup rests on racing and the adaptive grid alone.
- Not implemented: per-rep `<run id>.reps.jsonl` files for deep runs, and `--util-ceiling` in `run`
  records.
- Hashes do not follow a dependency's table ("Engine: op order").
- Budgets assume one GPU on a quiet box; on the shared 4090 box guard waits come on top.
- No measured sm_89 table, and the ten newer ops have no measured table on any device.
- The calibrate gate and the ladder re-tune are unmeasured.

## License

SPDX-License-Identifier: MIT
