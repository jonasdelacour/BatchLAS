# batchlas_tune {#tune_tool_readme}

The tuner that produces `tuned/<op>.<dtype>.<device>.txt` (spec:
[flat kernel selection](../../docs/design/flat-kernel-selection.md) §6; plan:
[the phase-3 plan](../../docs/design/flat-kernel-selection-phase3-plan.md) §1 "P3.2", §2 option B, §4;
the tables themselves: [tuned/README.md](../../tuned/README.md)). It times every entry
of an op's `candidates<T>()` over the op's declared grid, pinned with `select::ScopedPin` through the
public entry point, and hands the raw timings to `scripts/sweep_to_table.py --tuner`, which writes
the table with the same formatter as the converted seed tables.

Built with the benchmarks (`-DBATCHLAS_BUILD_BENCHMARKS=ON`): `cmake --build build --target batchlas_tune`.
That builds two binaries: `batchlas_tune`, a SYCL-free launcher, and `batchlas_tune_impl`, which
does the work. Always run `batchlas_tune`. Linking libbatchlas enumerates devices during static
init, which retains a CUDA context (about 550 MiB) on every visible GPU, so the launcher starts
the driver with `CUDA_VISIBLE_DEVICES=""` and each child gets its own GPU explicitly. The driver
refuses to run without the launcher, and its guard dies if the driver's pid ever shows up on a GPU.

## Usage

The normal workflow is tiered: plan, run, inspect, then write tables from the ledger.

    batchlas_tune potrf --tier preview --dtype float,double --devices 1 --plan   # cells, skips, estimate; no GPU
    batchlas_tune potrf --tier preview --dtype float,double --devices 1          # measure into the ledger
    batchlas_tune --status                                                       # what each ledger holds
    python3 scripts/sweep_to_table.py --ledger benchmarks/results/tuning/ledger --out /tmp/tables   # source=ledger: tables

A rerun measures only what is missing, stale or below the requested tier: a repeated preview run
plans every cell `skip:current` and exits in seconds, and a kernel edit re-races only the cells
whose families changed (`partial:<family>`). The `--plan` estimate is the sum over GPUs, so divide
by the number of `--devices`; on threadripper02 it ran 1.3x to 1.8x high (docs/design/tiered-tuning.md,
"Engine: end-to-end validation on sm_120"). Write ledger tables to a scratch `--out` and compare
them with `tuned/` first: `--ledger` refuses to overwrite a timed table that is not a ledger's
(a converted or `tuner:` source) and names it; `--replace-timed` is the deliberate switch, for when
the ledger tables are meant to replace the measured ones. Transcribed and ledger tables are
overwritten without it. `sweep_to_table.py --ledger ... --assume-current --out DIR` is a
diagnostic that judges records by their stored hashes (an imported raw sweep round-trips to the
shipped table); it refuses a missing `--out` or one that resolves to `tuned/`.

Two modes. A **tiered** run (`--tier preview|coarse|deep`, docs/design/tiered-tuning.md) records
per-cell results in the ledger and skips cells the ledger already holds. A **custom** run is the
expert two-pass protocol below; any protocol flag (`--reps`, `--warm`, `--passes`, `--remeasure`,
`--refine-ratio`, `--no-refine`, `--no-jit`, `--ld-pad`, `--raw`) selects it. It is recorded in
the ledger as tier `custom`, ranked below preview, only when `--ledger DIR` is given explicitly;
otherwise it writes its raw JSONL (and tables with `--out`) and leaves every ledger alone. With
neither mode, the tuner stops and asks.

    # tiered: ledger records, then tables (omit --out to update the ledger only; a scratch --out,
    # since the converter refuses to overwrite converted or tuner tables without --replace-timed)
    batchlas_tune potrf,trsm,posv --tier preview --dtype float,double --devices 1 --out /tmp/tables
    batchlas_tune all --tier coarse --devices 1,2 --budget 6 --progress-fd 3 3>events.jsonl

    # what a run would do, from the ledger and nvidia-smi's compute capability; no GPU work
    batchlas_tune potrf --tier preview --devices 1 --dtype float --plan
    batchlas_tune --status                      # op x dtype x device matrix of ledgers and tables
    batchlas_tune --import-raw benchmarks/results/tuning/trsm.float.sm_120.jsonl   # schema 1 -> deep run

    # custom: raw JSONL per dtype, then tables (omit --out for raw only)
    batchlas_tune potrf --dtype float,double,cfloat,cdouble --devices 1 \
                  --raw benchmarks/results/tuning --out tuned

    # a smaller custom protocol for a smoke test or a spot check
    batchlas_tune posv --devices 1 --dtype float --n-list 24,32 --nrhs-list 2 --batches 8192 --uplo L \
                  --reps 8 --warm 0.5 --raw /tmp/raw --out /tmp/tables

    # §10.3 gate: old choice pinned vs Auto, two passes with the arm order reversed
    batchlas_tune potrf --devices 1 --gate --old-csv old.csv --gate-csv gate.csv   # old choices given
    # trsm: the old choices come from --old-csv (no parent tuner has a trsm spec)
    # uplo/diag invariance A/B (raw only; the converter refuses it as a table):
    batchlas_tune trsm --devices 1 --dtype float --grid uplo=L:U --grid diag=N:U --no-refine \
                  --grid order=16:64:256 --grid q=8:128 --batches 8192 --raw /tmp/ab
    batchlas_tune trsm --devices 1 --gate --old-csv trsm_old.csv --gate-csv gate.csv

    batchlas_tune --list        # ops, key names, candidates per dtype, current kernel hash

| flag | default | meaning |
|---|---|---|
| `<op>` | required (not for `--list`, `--status`, `--import-raw`) | one op, a comma list or `all` (every registered spec); tiered runs take the ops in dependency order ("Op order" below) and skip, with one note, a dtype a spec refuses. Gate and custom runs take one op |
| `--tier` | none | `preview`, `coarse` or `deep`; see "Tiered mode" |
| `--plan` | off | print the starting lattice's cells, skips with reasons and the time estimate, then exit; no GPU |
| `--budget H` | none | stop refinement after H hours of measuring; the starting lattice always completes (`--plan` warns when its estimate with refinement exceeds H) |
| `--progress-fd N` | none | one JSON event per line on fd N; see "Tiered mode" |
| `--ledger DIR` | `<repo>/benchmarks/results/tuning/ledger` | ledger root, one `<op>.<dtype>.<device>/` directory per table; a custom run records into it only when this flag is given |
| `--device-key sm_NN` | nvidia-smi compute capability of the first `--devices` GPU | the device `--plan` reads the ledger for |
| `--cell-overhead-s` | 0.49 | per-child start-up in the estimate (measured on threadripper02) |
| `--no-worker` | off | race every cell in a fresh `--cell --mode race` child instead of the per-GPU worker |
| `--audit-fraction F` | the tier's (preview, coarse 0.02; deep 0.10) | share of worker cells re-raced in a fresh child |
| `--status` | | the op x dtype x device matrix: runs, cells, tier mix of each cell's best record, stale and partly stale counts, newest run and its age, the table's source, tier mix and date. No GPU |
| `--import-raw F` | | a schema-1 raw sweep into the ledger as a deep run; op, dtype and device from its meta |
| `--dtype` | `float` (gate with `--old-csv`: every dtype in the CSV) | comma list; one raw file and one table per dtype |
| `--devices` | required (not for `--plan`) | GPU indices (nvidia-smi / PCI order); inside `CUDA_VISIBLE_DEVICES` when that is set; see "Multi-GPU" |
| `--raw DIR` | `<repo>/benchmarks/results/tuning` | custom: raw JSONL (Git LFS there) |
| `--out DIR` | none | table directory; runs the converter after each dtype (tiered: `--ledger`, custom: `--tuner`) |
| `--reps`, `--warm`, `--passes` | 16, 1.5 s, 2 | custom: §6.3 protocol |
| `--remeasure` | 0.10 | custom: re-measure a cell once when pass medians differ by more |
| `--refine-ratio`, `--no-refine` | 1.1 | custom: §6.2 bisection stops when hi/lo < ratio |
| `--no-jit` | off | custom: skip the throwaway JIT pass |
| `--cap-gib` | 4 | skip cells whose matrices exceed this |
| `--max-dim` | 2048 tiered, off custom | `skip:dim`: no cell with a matrix dimension (m, n, k, order, q, nrhs, derived rows; never `batch`) above this; 0 = off |
| `--ld-pad` | 0 | custom: ld = n + pad (non-natural leading dimension) |
| `--n-list`, `--batches`, `--nrhs-list`, `--uplo`, `--grid key=v1:v2` | the op's `choice.hh` grid | replace whole grid axes |
| `--no-guard`, `--guard-wait`, `--util-ceiling` | on, 300 s, 5 % | the idle guard; see "Guard" |
| `--allow-idle-foreign` | off | tolerate another user's idle CUDA contexts; see "Guard" |
| `--lock-dir` | `/tmp` | per-GPU flock files `batchlas_tune_gpu<N>.lock` |
| `--cell-timeout` | 1800 s | a child running longer is killed and counts as a failed child |
| `--gate`, `--old-csv`, `--parent-bin`, `--gate-csv`, `--gate-limit` | 1.05 | gate mode |

## Tiered mode

The tier parameters, the ledger and the record rules are in docs/design/tiered-tuning.md
("Engine: tiers and the per-cell algorithm", "Engine: the ledger and table generation"). Here,
what the driver (`tiered_driver.cc`, planning in `schedule.cc`) does with them.

- **Op order.** A stable topological sort over `op_dependencies()` (`schedule.cc`), input order
  breaking ties; `all` starts from the canonical order gemm, trsm, syr2k, gemv, spmm, trmm, symm,
  syrk, potrf, getrf, getrs, geqrf, ormqr, getri, posv, gesv, orgqr, syev, gesvd
  (docs/design/tiered-tuning.md, "Engine: op order over all 19 ops").
- **Rounds, breadth-first.** Round 0 is the tier's starting lattice (`tier_lattice` of the op's
  axes; gemm's demand-driven grid is subsampled by key hash) for every op and dtype, in op order.
  Then refinement rounds (`refine_all_axes`, the tier's mode and margin) over every cell the
  ledger and this run ranked inside the requested grid, until no midpoint is left, `--budget`
  is spent or the refinement cap is hit. Rounds are per (op, dtype) job, and the GPUs are
  pipelined across jobs: each round is one queue sorted by (per-item footprint, bytes), every GPU
  pops the next cell of the earliest job (op order) that has queued cells, and a job's next round
  is planned only once all of its current round's cells are recorded. A GPU never waits for
  another job's straggler; the summary counts the (op, dtype) switches and the carve-out restarts
  the pop order costs (progress `schedule`). See docs/design/tiered-tuning.md, "Engine: pipelined
  cell queues across op and dtype jobs".
- **Refinement rules** (docs/design/tiered-tuning.md, "Engine: refinement convergence rules").
  A bracket is a flip when its two winners differ and, at either end, the other end's winner is
  more than the 3% tie slower (or cannot win there: not runnable, or eliminated without a median);
  near-tie alternation (within the tie at both ends) refines nothing. The preview margin (runner-up within 10%) applies only
  between two round-0 cells at the running tier, stored or measured now. `batch` refills only its own axis values, never a
  geometric midpoint. Refinement cells per op and dtype are capped at `refine_cap_factor` x
  round-0 records at the running tier (preview 3.0, coarse 1.0, deep 2.0), counting the ledger's
  current records as well as this run's, and refined records already stored count against it: a
  run stopped by Ctrl-C or `--budget` after its lattice refines on the next run, within the same
  total; past it the first cells in `refine_all_axes` order (flips, then margin hedges)
  run, refinement stops, and the run prints `refinement cap hit` and emits `refine_cap`.
- **Per cell** (`plan_round`): a dimension over `--max-dim` is `skip:dim` (checked first, `OpSpec::dims`); over `--cap-gib` is `skip:cap`; a current record at the same or a
  higher tier is `skip:current`; a partly stale one at the same or a higher tier re-races only
  the changed or added families (a candidate stored `skipped`, which could not run there, is not
  compared, but one skipped `dominated:` by the removed carry-forward is stale; the spec's `// common` deps line puts `<op>.cc`, where `can_run` lives, in every family)
  plus its stored winner and runner-up, at the stored record's
  tier, and appends the merged record (`partial:<families>`; re-raced candidates ranked by their
  new times, the unchanged ones below them in stored order). A stale record, or one where every
  candidate is `error`, counts as missing: measured in full at the run's tier. A cell whose probe
  found at most one runnable candidate is `skip:single`: recorded with that candidate ranked,
  untimed, which the converter writes as an untimed row (`<spelling> - # <tier>`).
  A probe result comes from an earlier child of the same cell (its `skipped` arms).
- **Measuring.** A cell is raced (`--mode race`, `run_race` in `cell_runner.hh`): a
  `warm_topup_s` warm-up per live candidate, interleaved, then rounds of one timed run per live
  candidate (order rotated, every other round reversed in deep), `race_step` after each round, until
  a winner, a tie or `max_reps` rounds (a candidate raced alone, or whose rivals all failed, still
  runs `min_reps` rounds). Every raced candidate is then verified as in the custom protocol, the
  eliminated too: one that passes keeps the median of its rounds (status `eliminated`, ranked
  after the survivors), one that fails is `bad` and left out of the row. The nearest finished cell's winner is
  raced first. Each candidate's median, min and max go into a ledger `cell` record. From the first
  round on, whatever `min_reps` says, an arm whose median paired ratio to the leader exceeds 4.0 is
  eliminated (a gross loser).
- **Strict pins.** Every arm is pinned strictly (`select::StrictPin`): a `vendor` (or `native`)
  pin that its class cannot serve is `skipped`, reason `pin refused (strict): ...`, instead of
  timing Auto under the vendor's name.
- **No dominance carry-forward.** Every arm is timed at every cell. The removed rule (an arm
  eliminated at > 10x not timed at larger cells) skipped arms at vendor/native crossovers; a
  ledger record still holding a `dominated:<key>` skip is partly stale, and the next run re-races
  those arms against the stored winner and runner-up
  (docs/design/tiered-tuning.md, "Engine: dominance carry-forward removed").
- **Worker.** Each GPU gets one `batchlas_tune_impl --worker` (3 s clock warm-up kernel at start),
  fed its share of a round in ascending per-item footprint (bytes / batch) and restarted before a
  cell smaller than one it has run. Between cells the guard checks for foreign compute processes
  (the worker excepted); the full guard with utilization runs before a worker starts and every 60 s
  after a 1 s idle. A worker that exits or exceeds `--cell-timeout` is restarted and the cell retried
  once; a second failure, or any `error` candidate (a possible sticky CUDA error), restarts it and
  races the cell in a fresh child, whose result is recorded (progress `worker_restart`). A candidate
  whose `error` the fresh child reproduced in two consecutive cells is the candidate's own error: for
  the rest of the run that op and dtype race it alone in a fresh child per cell, the other candidates
  on the worker (all of them benched: the cell is a fallback). Arms raced alone are left out of the
  audit's worker side. A benched arm that errors in 5 consecutive fresh-child cells is dropped for
  the rest of that op and dtype: `error`, reason `dropped after 5 consecutive errors`, not run. In a fresh child a failed
  child is retried once, then every arm is run alone, as below; a failure that leaves no `ok`, `bad`
  or `skipped` candidate writes no record, so the next run measures the cell again.
- **Audit.** A worker cell with `fnv1a64(run_id + key) % 1000 < audit_fraction * 1000` is raced again
  in a fresh child; until an op and dtype has been audited once, its next worker cell that did not
  fall back is audited whatever the hash picked. A
  candidate usable (`ok`/`eliminated`) in one and refused or failing in the other (`eliminated`
  against `bad` is inconclusive), or a different winner whose worker winner is more than 10% slower
  in the fresh run, is a mismatch: the ledger gets
  an `audit` record either way, and on a mismatch the run line is rewritten with
  `wm.<op>.<dtype>: fresh` and the rest of that op and dtype runs in fresh children.
- **Ledger.** One run file per (op, dtype) under `--ledger`, opened at the first record. Its `run`
  record always carries the op's full key spec and candidate list, whatever `--grid` narrowed.
- **Estimate.** Per cell: 0.49 s child start-up (`--cell-overhead-s`) plus, per candidate,
  `max_reps` times the nearest ledger record's median (else bytes / 500 GB/s), the warm-up top-up
  and 0.05 s of verification. Refinement adds, per op and dtype, ratio x the lattice cells to
  measure at their mean estimate, where ratio is this ledger's refined / round-0 records at the tier
  (at most the cap factor), else cap factor x 0.5. `--plan` prints both the lattice-only and the
  with-refinement estimate; the run ends with a summary line per op and dtype (cells per round,
  lattice and refinement counts, the planned refinement, wall time).
- **Progress events** (`--progress-fd`): `{"ev":"plan","cells":N,"est_s":S,"refine_cells":R,"est_refine_s":E,"est_total_s":T}`
  (`est_s` is the lattice only), `{"ev":"refine_cap","op":o,"dtype":d,"lattice":L,"refined":n,"cap":c,"dropped":k}`,
  `{"ev":"cell_start",<op, dtype, key fields>,"gpu":g}`,
  `{"ev":"cell_done",<op, dtype, key fields>,"ranked":"a|b","tier":t}`,
  `{"ev":"eliminated",<op, dtype, key fields>,"cand":c,"round":r}`,
  `{"ev":"audit",<op, dtype, key fields>,"verdict":v,"fresh_ms":f,"warm_ms":w,"fresh":b}`,
  `{"ev":"worker_restart",<op, dtype, key fields>,"gpu":g,"restarts":n,"fallback":b}`,
  `{"ev":"schedule","switches":s,"restarts":r,"switch_restarts":k}`, `{"ev":"done"}`.

## Protocol (§6.3) and where it lives

This is the custom protocol; the tiered one is above.

- **One cell per process.** The driver forks and execs itself with `--cell` once per (cell, pass);
  the SLM carve-out is sticky per CUfunction (`benchmarks/factor_bench.cc` header).
- In the child (`cell_runner.hh`): each candidate is probed through the public `*_buffer_size`
  under its pin; a refused pin (`std::invalid_argument`, `can_run` false) is `skipped` and never
  timed. Warm-up `--warm` seconds per runnable candidate, interleaved in the timed order. Then
  `--reps` reps, each rep running every candidate once with the order rotated left by the rep
  index, inputs restored untimed before each call. Then one more untimed run per candidate,
  verified on the host on items 0 and batch-1 with factor_bench's residuals (`residuals.hh`);
  a failure is `bad` and the candidate is dropped from the cell.
- In the driver: a JIT pass (one untimed run of every candidate) over every cell, then pass 1
  over the whole grid, then pass 2 over the whole grid with the candidate order reversed. A cell's
  time per candidate is the mean of its pass medians. A cell where any candidate's pass medians
  differ by more than 10% gets both passes again, once, and that attempt replaces the first,
  unless no child of the repeat timed anything; then the first attempt stands.
- A failed child (nonzero exit, timeout, unreadable result, or a foreign process found after it)
  is retried once. If it fails again for any reason but the guard, every arm is run alone once;
  the arms that fail alone are recorded as `error` (`crashed alone: ...`) and the rest are timed
  together. One aborting candidate therefore costs that candidate, not the cell.
- Tie rule: within 3% of the best ties, ties in `candidates<T>()` order (`rank()` in
  `tune_core.cc`, `rank()` in the converter; `tune_tests` checks they agree).
- §6.2: after the grid, along each line of fixed (uplo, other keys, batch), neighbouring n points
  with different winners are bisected at round(sqrt(lo*hi)) until hi/lo < 1.1 or the bracket is
  two adjacent integers. Each round of midpoints gets its own JIT pass and two passes. Batch is
  never refined. Refined points are ordinary rows. A midpoint that was measured but has no winner
  is not measured again; its bracket is printed and written as a `stalled` record.

## Multi-GPU

`--devices 1,2,3` runs one child per GPU at a time, shards cells round-robin, and keeps both passes
of a cell on the same GPU. Each GPU is held under an exclusive flock for the whole run, so two
tuners never share a device. **This departs from docs/developer/agent-guide.md §10's one-measuring-process-per-box rule**
in the same way as the potrf and posv seed sweeps did (`benchmarks/results/routing/README.md`): use
one device when the box is shared or when a verdict hinges on a few percent. Children get
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<gpu>`, so indices match nvidia-smi.
If you exported `CUDA_VISIBLE_DEVICES` to fence the tuner in, `--devices` must name GPUs inside
that list (as nvidia-smi indices); UUID lists are refused. Every listed GPU must report the same
device key, since one run writes one device's tables.

## Guard

Before each child: no compute process on the GPU and utilization at most `--util-ceiling`, else
wait up to `--guard-wait` and then stop. After each child: a compute process on the GPU discards
that child's numbers and the child is retried. Any process with a context counts, including one
whose kernels run elsewhere, and an entry nvidia-smi cannot show (`[N/A]`) counts as foreign.
At start the tuner warns when a listed GPU drives a display (docs/developer/agent-guide.md §13) or when any other GPU
on the box has compute processes (docs/developer/agent-guide.md §10). Without nvidia-smi the tuner stops; `--no-guard`
measures without the guard.

`--allow-idle-foreign` is for a shared box where another user holds idle contexts on every GPU
(maintainer decision, 2026-10-04). Before each child, foreign compute processes are tolerated
while utilization is at most `--util-ceiling`, and that child remembers their pids. After the
child, only a pid that was not in that set discards the numbers. A busy GPU still refuses, and
so does any entry nvidia-smi cannot show as a pid (`[N/A]`), since a new one could not be told
apart from it. At start the tuner prints one warning per listed GPU naming the tolerated
`pid(user)` entries. The raw files record `allow_idle_foreign` and `tolerated_foreign`
(`gpu:pid(user),..;..` as seen at start) in `meta` (gate mode: a `guard` record), and each
`pass`/`rep` record carries the pids that child ran beside. Without the flag the guard is strict
as above.

## Raw JSONL (schema 1)

One file per (op, dtype, device): `<raw>/<op>.<dtype>.<device>.jsonl`, one flat JSON object per line.

| kind | fields |
|---|---|
| `meta` (first line) | `schema`, `op`, `dtype`, `device`, `device_name`, `batchlas` (8-hex HEAD, `-dirty` when tracked files differ), `kernels` (§6.4 hash), `kernel_sources` (`\|`-joined), `date`, `keys` (the `# keys:` line), `candidates` (`\|`-joined, list order), `reps`, `warm_s`, `passes`, `tie`, `remeasure`, `refine_ratio` (0 = off), `cap_gib`, `ld_pad`, `devices`, `allow_idle_foreign`, `tolerated_foreign` (see "Guard"), `argv` |
| `rep` | `op`, `dtype`, `device`, the key fields, `pass` (1-based), `attempt` (0, or 1 for a re-measure), `reverse`, `gpu`, `tolerated_foreign` (only with `--allow-idle-foreign`), `cand`, `rep`, `slot` (position in that rep's rotated order), `ms` |
| `pass` | as `rep` without `rep`/`slot`/`ms`, plus `status` (`ok`, `skipped`, `bad`, `error`), `reason`, `median_ms`, `residual`, `info_nonzero`, `reps` |
| `retry` | `op`, `dtype`, the key fields, `mode`, `error`: a child that failed and was run again |
| `cell` (last lines) | the key fields, `round` (0 = coarse grid), `refined`, `status` (`ok`, `none` = no candidate timed in every pass of any attempt, `skipped` = over the cap), `reason`, `final_attempt` (the attempt used, -1 when none), `ranked` (`\|`-joined) |
| `stalled` | `op`, `dtype`, `note`: a refinement bracket left wider than the ratio because its midpoint has no winner |

Key fields are written under the op's key names; integer values are JSON integers. Times are
milliseconds per call for the whole batch, as in the sweeps. The converter reads `meta` and the
`pass` records of each cell's highest attempt that timed anything; a candidate enters the table
when it is `ok` in all `passes` passes of that attempt. Gate mode writes `gate.<op>.<device>.jsonl` with `gate` records
(`pass`, `arm`, `status`, `reason`, `median_ms`) when `--raw` is given.

## Gate mode (§10.3, plan §4)

Cells come from `--old-csv` (columns `dtype`, every key name, `old`; an old binary's `native:<algo>` and
`vendor:auto` readbacks are normalised; empty fields and `"..."` quoting are read, so a previous gate CSV
works; a row of the wrong width, an empty `old`, or a dtype outside an explicit `--dtype` stops
the gate) or from the grid flags with `--parent-bin`, a `batchlas_tune` built
from the parent commit, whose Auto choice is read from its coverage `reached` row (`--mode probe`).
The branch's own Auto choice is read the same way. Cells where both agree are `same` and not timed.
Otherwise one child per pass times `auto` against the old spelling pinned, interleaved and
rotated, pass 2 in reversed base order. Verdict `FAIL` when new/old > `--gate-limit` in both
passes, `BAD_ROW` when an arm failed, else `pass`; `ERROR` when a probe failed (the error text is
in the `old` or `new` column, CSV-quoted); `cap` for a cell over `--cap-gib`, which is neither
probed nor timed. The CSV has the columns of
`benchmarks/results/routing/sm120_potrf_phase2_gate.csv` plus every key name. Exit status: 1 on
any FAIL; else 3 when any row is `ERROR` or `BAD_ROW`, or no cell was gated (`same` or `pass`);
else 0. The last line counts every verdict.

## Specs

One spec per op, registered with `BATCHLAS_TUNE_REGISTER` and listed in `CMakeLists.txt`; `--list`
prints the registered ops. Adding one is step 4 of
[adding an op to kernel selection](../../docs/extending.md#adding-an-op-to-kernel-selection).
`<op>_spec.cc` implements `OpSpec` (`spec.hh`; not `select::OpSpec`): key names, candidates and grid axes from the op's
`choice.hh`, a problem builder (potrf, posv: SPD A; posv, trsm: random B; trsm: a diagonally
dominant triangular A, the other triangle poisoned), host verification, the 4 GiB
cap, how an old coverage route maps to a spelling, and the kernel-source list for the §6.4 hash
between the `kernel-sources-begin` / `kernel-sources-end` markers. `check_tuned_tables.py` and
`cmake/BatchLASTunedStaleness.cmake` parse that block from the source, and the driver refuses to
run when the block and its compiled list differ (rebuild). The hash is
`sha256sum <files> | sha256sum`, first 8 hex digits, paths relative to the repository.
posv's list holds only its own kernels: its `cta` and `blocked` times also depend on potrf's and
trsm's choices (plan §2 "Coupling"), which the hash does not follow yet. trsm's list is its native
kernels (not the vendor TU); its `uplo` and `diag` are hidden grid axes fixed at L and N (not table
keys, plan §1.2), its `trans=T` times ConjTrans for a complex scalar, and trsm has no workspace,
so a pin is probed by one untimed run instead of a sizing call.

gemv, trmm, symm, syrk and syr2k take no workspace either and are probed the same way. Their
non-key arguments are fixed: trmm uplo L, trans N, diag N; symm side L, uplo L; syrk uplo L; syr2k
uplo L, trans N; gemv's `trans=T` times ConjTrans for a complex scalar. Inputs carry large finite
poison in the triangle an op must not read (A for trmm and symm) or write (C for syrk and syr2k: a
changed element is a nonzero info), and trmm's output is re-poisoned before every run. symm, syrk
and syr2k are real-only: `--dtype cfloat` is refused and `--list` prints `-`. symm and syrk key on
`form`, which `form_of` derives from the extents, so their `grid()` holds only the consistent cells
of the lattice (`--grid` filters it, as gemm's does). trmm's and symm's `expand` call the public
gemm, so their deps list gemm's kernels and `gemm.cc`, and their tables should be tuned after gemm's.

syev and gesvd check values-only results against host LAPACKE (`host_reference.hh`; the impl links
it when the host backend is built, and without it those arms are `bad`). gesvd has no batch key:
each cell runs at a batch derived from m and n (a power of two in [128, 16384] with
m n batch <= 2^24), and its grid drops the cells no call keys (Hermitian non-square, square thin).
One vendor refusal lives in a spec because the library would not report it as a refused pin:
gesvd's values-only non-square vendor call faults the device (known defect 15) although `can_run`
admits it. spmm's `vendor` arm needs none: where `can_run` refuses it (known defect 17), the strict
pin refuses it too.
spmm tunes the GPU tables only; its cpu tables stay transcribed.
