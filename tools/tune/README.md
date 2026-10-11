# batchlas_tune {#tune_tool_readme}

> **Status:** current.

`batchlas_tune` times every entry of an op's `candidates<T>()` over the op's grid, appends the
results to the ledger `benchmarks/results/tuning/ledger/<op>.<device>.jsonl`, and writes
`tuned/<op>.<device>.txt` from it through `scripts/sweep_to_table.py`
([the tuning data layout](../../docs/design/tuning-data.md)). Candidates are pinned with
`select::ScopedPin` through the public entry point. The output tables are described in
[tuned/README.md](../../tuned/README.md). The selection design is in
[flat kernel selection](../../docs/design/flat-kernel-selection.md) §6.

## Build and launch

    cmake --build build --target batchlas_tune     # needs -DBATCHLAS_BUILD_BENCHMARKS=ON

The build produces two binaries:

- `batchlas_tune`: a SYCL-free launcher. Always run this one.
- `batchlas_tune_impl`: does the work. The launcher starts it with `CUDA_VISIBLE_DEVICES=""`
  (linking libbatchlas retains about 550 MiB of CUDA context per visible GPU). Each child gets its
  GPU explicitly. The driver refuses to run without the launcher.

## Usage

Tiered runs are the normal workflow: plan, measure, inspect, write tables.

    # plan: cells, skips and estimate; no GPU
    batchlas_tune potrf --tier preview --dtype float,double --devices 1 --plan
    # measure into the ledger
    batchlas_tune potrf --tier preview --dtype float,double --devices 1
    # what each ledger holds
    batchlas_tune --status
    # ledger tables into a scratch directory
    python3 scripts/sweep_to_table.py --op potrf --device sm_89 --out /tmp/tables

    # tiered, several ops, tables in a scratch directory
    batchlas_tune potrf,trsm,posv --tier preview --dtype float,double --devices 1 --out /tmp/tables
    batchlas_tune all --tier coarse --devices 1,2 --budget 6 --progress-fd 3 3>events.jsonl

    # import a schema-1 raw sweep as a deep run
    batchlas_tune --import-raw /tmp/raw/trsm.float.sm_120.jsonl

    # custom protocol: raw JSONL per dtype; with --ledger also a custom run there and, with --out, tables
    batchlas_tune potrf --dtype float,double,cfloat,cdouble --devices 1 \
                  --raw /tmp/raw --ledger benchmarks/results/tuning/ledger --out tuned

    # smoke test
    batchlas_tune posv --devices 1 --dtype float --n-list 24,32 --nrhs-list 2 --batches 8192 --uplo L \
                  --reps 8 --warm 0.5 --raw /tmp/raw

    # uplo/diag invariance A/B (raw only; sweep_to_table refuses a ledger that varies a hidden axis)
    batchlas_tune trsm --devices 1 --dtype float --grid uplo=L:U --grid diag=N:U --no-refine \
                  --grid order=16:64:256 --grid q=8:128 --batches 8192 --raw /tmp/ab

    # list ops, key names, candidates per dtype and the current kernel hash
    batchlas_tune --list

Modes:

- **Tiered** (`--tier preview|coarse|deep`): records per-cell results in the ledger and skips cells
  it already holds. A run with no `--out` updates the ledger only.
- **Custom** (any of `--reps`, `--warm`, `--passes`, `--remeasure`, `--refine-ratio`, `--no-refine`,
  `--no-jit`, `--ld-pad`, `--raw`): the two-pass protocol below. Recorded in the ledger as tier
  `custom` (ranked below preview) only when `--ledger DIR` is given explicitly.
- **Gate** (`--gate`): compares the old choice with Auto. See "Gate mode".
- With no mode flag the tuner stops and asks.

A rerun measures only what is missing, stale or below the requested tier. A repeated preview run
skips every cell (`skip:current`) and exits in seconds. After a kernel edit only the cells whose
families changed are re-raced (`partial:<family>`).

## Flags

| flag | default | meaning |
|---|---|---|
| `<op>` | required (not for `--list`, `--status`, `--import-raw`) | one op, a comma list or `all` (every registered spec). Tiered runs take the ops in dependency order ("Op order" below) and skip, with one note, a dtype a spec refuses. Gate and custom runs take one op |
| `--tier` | none | `preview`, `coarse` or `deep`; `custom` with `--remeasure-keys`: exactly the named cells, every candidate, no lattice and no refinement (a rehash sample, [kernel hashes](../../docs/design/tuning-data.md#tuning-data-kernel-hash)) |
| `--plan` | off | print cells, skips with reasons and the estimate, then exit. No GPU |
| `--budget H` | none | stop refinement after H hours; the starting lattice always completes. `--plan` warns if its estimate exceeds H |
| `--only-named` | off | with `--remeasure-keys` and a tiered `--tier`: round 0 is exactly the named cells at the run's tier, no lattice and no refinement (how the cells a rehash sample disagrees on are re-measured at deep) |
| `--remeasure-keys FILE` | none | lines `<op> <dtype> <name=v,...>` (the ledger key, in its field order): those cells are measured again in round 0 at the run's tier whatever the ledger holds, refinement midpoints and cells past `--max-dim` included (not past `--cap-gib`). Every op and dtype named must be in the run. Other cells the tier plans (stale or missing lattice cells, refinement) are measured too. See "Engine: re-measuring named cells" in [tiered tuning](../../docs/design/tiered-tuning.md) |
| `--progress-fd N` | none | one JSON event per line on fd N |
| `--ledger DIR` | `<repo>/benchmarks/results/tuning/ledger` | ledger root, one `<op>.<device>.jsonl` file per table, every dtype in it. Custom runs record here only when given |
| `--allow-dirty` | off | measure into a ledger from a `--repo` tree with uncommitted changes to tracked files. Without it such a run is refused; with it the run is stamped `<sha>-dirty` and every reader leaves it out of the tables |
| `--device-key sm_NN` | compute capability of the first `--devices` GPU | device whose ledger `--plan` reads |
| `--cell-overhead-s` | 0.49 | per-child start-up in the estimate |
| `--no-worker` | off | race every cell in a fresh `--cell --mode race` child instead of the per-GPU worker |
| `--audit-fraction F` | preview and coarse 0.02; deep 0.10 | share of worker cells re-raced in a fresh child |
| `--status` | | op x dtype x device matrix: runs, cells, tier mix, stale and partly stale counts, newest run and age, table source, tier mix and date. No GPU |
| `--import-raw F` | | schema-1 raw sweep into the ledger as a deep run; op, dtype and device come from its meta |
| `--dtype` | `float` (gate with `--old-csv`: every dtype in the CSV) | comma list; one raw file and one table per dtype |
| `--devices` | required (not for `--plan`) | GPU indices in nvidia-smi / PCI order. Inside `CUDA_VISIBLE_DEVICES` when that is set; UUID lists are refused |
| `--raw DIR` | `<repo>/benchmarks/results/tuning` | custom: raw JSONL (Git LFS) |
| `--out DIR` | none | table directory: after the run, `sweep_to_table.py --op <op> --device <dev>` regenerates each measured op's table from the ledger (custom runs: only with `--ledger`) |
| `--reps`, `--warm`, `--passes` | 16, 1.5 s, 2 | custom: §6.3 protocol |
| `--remeasure` | 0.10 | custom: re-measure a cell once when pass medians differ by more |
| `--refine-ratio`, `--no-refine` | 1.1 | custom: §6.2 bisection stops when hi/lo < ratio |
| `--no-jit` | off | custom: skip the throwaway JIT pass |
| `--cap-gib` | 4 | skip cells whose matrices exceed this |
| `--max-dim` | 2048 tiered, off custom | `skip:dim`: no cell with a matrix dimension (m, n, k, order, q, nrhs, derived rows; never `batch`) above this; 0 = off |
| `--ld-pad` | 0 | custom: ld = n + pad (non-natural leading dimension) |
| `--n-list`, `--batches`, `--nrhs-list`, `--uplo`, `--grid key=v1:v2` | the op's `choice.hh` grid | replace whole grid axes |
| `--no-guard`, `--guard-wait`, `--util-ceiling` | on, 300 s, 5 % | idle guard (see "Guard") |
| `--allow-idle-foreign` | off | tolerate another user's idle CUDA contexts (see "Guard") |
| `--lock-dir` | `/tmp` | per-GPU flock files `batchlas_tune_gpu<N>.lock` |
| `--cell-timeout` | 1800 s | a child running longer is killed and counts as a failed child |
| `--gate`, `--old-csv`, `--parent-bin`, `--gate-csv`, `--gate-limit` | `--gate-limit` 1.05 | gate mode |

## Tiered mode

Parameters and record rules: [tiered tuning](../../docs/design/tiered-tuning.md) ("Engine: tiers and
the per-cell algorithm"; "Engine: the ledger and table generation"). The driver is `tiered_driver.cc`
and planning is in `schedule.cc`.

- **Op order.** A stable topological sort over `op_dependencies()` (`schedule.cc`), input order
  breaking ties. `all` starts from the canonical order gemm, trsm, syr2k, gemv, spmm, trmm, symm,
  syrk, potrf, getrf, getrs, geqrf, ormqr, getri, posv, gesv, orgqr, syev, gesvd ("Engine: op order
  over all 19 ops" in the design page).
- **Rounds.** Round 0 is the tier's starting lattice for every op and dtype, in op order. Refinement
  rounds follow over each cell ranked inside the requested grid, until no midpoint is left, `--budget`
  is spent or the refinement cap is hit.
- **Pipelined queues.** Rounds are per (op, dtype) job. Each round is one queue sorted by (per-item
  footprint, bytes); every GPU pops the next cell of the earliest job (op order) that has queued
  cells, and a job's next round is planned only once all of its current round's cells are recorded.
  A GPU never waits for another job's straggler. The summary counts the (op, dtype) switches and the
  carve-out restarts they cost (progress `schedule`). See "Engine: pipelined cell queues across op
  and dtype jobs".
- **Refinement.** A bracket is a flip when its two winners differ and the other end's winner is more
  than the 3% tie slower there (or cannot win there). Near-tie alternation refines nothing. The preview
  margin (runner-up within 10%) applies only between round-0 cells at the running tier. Refinement per
  op and dtype is capped at `refine_cap_factor` x round-0 records (preview 3.0, coarse 1.0, deep 2.0).
  Past the cap the run prints `refinement cap hit` and emits `refine_cap`.
- **Per cell.** A dimension over `--max-dim` (`OpSpec::dims`, checked first): `skip:dim`. Over `--cap-gib`: `skip:cap`. A current record at the same or higher tier: `skip:current`.
  A partly stale record re-races only the changed or added families, plus its stored winner and
  runner-up (`partial:<families>`). A candidate stored `skipped` is not compared, but one skipped
  `dominated:` by the removed carry-forward is stale. `<op>.cc` (where `can_run` lives) is a dependency
  of every family.
  A stale record, or one where every candidate is `error`, counts as missing. A cell whose probe found
  at most one runnable candidate is `skip:single` and is written as an untimed row.
- **Measuring.** Each cell is raced (`run_race` in `cell_runner.hh`): a warm-up per live candidate,
  then interleaved rounds of one timed run per candidate until a winner, a tie or `max_reps` rounds.
  Eliminated candidates are still verified; a passing one keeps its median with status `eliminated`
  (ranked after the survivors). A failing one is `bad` and left out. From the first round on, whatever
  `min_reps` says, an arm whose median paired ratio to the leader exceeds 4.0 is eliminated (a gross
  loser).
- **Strict pins.** Every arm is pinned strictly (`select::StrictPin`): a `vendor` (or `native`) pin
  that its class cannot serve is `skipped`, reason `pin refused (strict): ...`, instead of timing Auto
  under the vendor's name.
- **No dominance carry-forward.** Every arm is timed at every cell. A ledger record still holding a
  `dominated:<key>` skip is partly stale; the next run re-races those arms against the stored winner
  and runner-up ("Engine: dominance carry-forward removed").
- **Worker.** Each GPU runs one `batchlas_tune_impl --worker`. A worker that exits or exceeds
  `--cell-timeout` is restarted and the cell retried once. A second failure, or any `error` candidate,
  restarts the worker and races the cell in a fresh child. An arm whose `error` the fresh child
  reproduced in two consecutive cells races alone in a fresh child per cell; if it errors in 5
  consecutive such cells it is dropped for the rest of that op and dtype (`error`, reason
  `dropped after 5 consecutive errors`, not run).
- **Audit.** A worker cell is re-raced in a fresh child when `fnv1a64(run_id + key) % 1000` falls
  below `audit_fraction * 1000`. The first worker cell of each op and dtype is always audited. A
  mismatch (usable in one run and refused or failing in the other, or a different winner more than
  10% slower in the fresh run) is logged as an `audit` record. The rest of that op and dtype then runs
  in fresh children.
- **ld audit.** Each candidate's first verified cell per op and dtype is verified once more, in the
  same worker or child, at `ld_pad + 3`. A failure makes the arm `bad` there (reason `ld audit: ...`),
  prints `ld audit FAILED`, and the summary gives passed/skipped/failed per op and dtype. The audit
  above ignores an ld-audit failure (its fresh child verifies at the natural ld). Rationale:
  [tiered tuning](../../docs/design/tiered-tuning.md#engine-the-ld-audit).
- **Ledger.** One file per (op, device) under `--ledger`, shared by every dtype job and process:
  lines are appended under `flock`, one `write()` each, and carry `dtype`. A run writes one `run`
  line per dtype with the full key spec and candidate list, whatever `--grid` narrowed. A dirty
  `--repo` tree is refused without `--allow-dirty`.
- **Estimate.** Per cell: `--cell-overhead-s` plus, per candidate, `max_reps` times the nearest ledger
  median (else bytes / 500 GB/s), the warm-up top-up and 0.05 s of verification. `--plan` prints the
  lattice-only and with-refinement estimates. The estimate is the sum over GPUs, so divide by
  `--devices`; it ran 1.3x to 1.8x high on threadripper02.

Progress events (`--progress-fd`), one JSON object per line; every event also carries `t`, Unix time in seconds:

| event | fields |
|---|---|
| `plan` | `cells`, `est_s` (lattice only), `refine_cells`, `est_refine_s`, `est_total_s` |
| `plan_job` | one per (op, dtype), before `plan`: `op`, `dtype`, `device`, `cells`, `measure`, `mix` (`12 current, 300 measure`), `est_s`, `refine_cells`, `est_refine_s` |
| `refine_cap` | `op`, `dtype`, `lattice`, `refined`, `cap`, `dropped` |
| `cell_start` | op, dtype, key fields, `gpu` |
| `cell_done` | op, dtype, key fields, `ranked` (`a\|b`), `tier`, `round` (absent on a single-candidate cell) |
| `eliminated` | op, dtype, key fields, `cand`, `round` |
| `audit` | op, dtype, key fields, `verdict`, `fresh_ms`, `warm_ms`, `fresh` |
| `worker_restart` | op, dtype, key fields, `gpu`, `restarts`, `fallback` |
| `ld_audit` | op, dtype, key fields, `cand`, `verdict` (`fail`), `reason` |
| `ld_audit_summary` | `op`, `dtype`, `passed`, `skipped`, `failed`, `failures` |
| `schedule` | `switches`, `restarts`, `switch_restarts` |
| `done` | none |

## Custom protocol (§6.3)

- **One cell per process.** The driver execs itself with `--cell` once per (cell, pass). The SLM
  carve-out is sticky per CUfunction (`benchmarks/factor_bench.cc`).
- **In the child** (`cell_runner.hh`): each candidate is probed through the public `*_buffer_size`
  under its pin. A refused pin (`std::invalid_argument`, or `can_run` false) is `skipped` and never
  timed. Then `--warm` seconds of warm-up per runnable candidate, `--reps` reps with the order rotated
  by rep index (inputs restored untimed before each call), and one verified run checked by
  `batchlas::verify` ([verification](../../docs/design/verification.md)) on items 0, batch/2 and
  batch-1 against the cell's bound `c * f(n) * eps` (potrf factorization and posv solve: n; trsm and
  gemm, componentwise under the blas kind: order + 2 and k + 2). A failed verification is `bad`.
- **In the driver:** a JIT pass, pass 1 over the grid, then pass 2 with the candidate order reversed.
  A cell's time per candidate is the mean of its pass medians. If any candidate's pass medians differ
  by more than 10%, both passes run again once; that attempt replaces the first unless nothing in it
  timed.
- **Failures.** A failed child (nonzero exit, timeout, unreadable result, or a foreign process after
  it) is retried once. If it fails again, every candidate is run alone. Those that fail alone are
  `error` (`crashed alone: ...`), and the rest are timed together.
- **Tie rule.** Within 3% of the best is a tie, broken by `candidates<T>()` order (`rank()` in
  `tune_core.cc`; the cell record's `ranked` list carries it into the ledger).
- **Refinement (§6.2).** Along each line of fixed (uplo, other keys, batch), neighbouring n points with
  different winners are bisected at `round(sqrt(lo*hi))` until hi/lo < 1.1 or the bracket is two
  adjacent integers. Batch is never refined. A midpoint with no winner is written as a `stalled` record.

## Multi-GPU

`--devices 1,2,3` runs one child per GPU at a time, shards cells round-robin, and keeps both passes of
a cell on one GPU. Each GPU is held under an exclusive flock for the run. Children get
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<gpu>`. Every listed GPU must report the same
device key, because one run writes one device's tables.

> **Note:** Running several GPUs at once departs from the one-measuring-process rule in
> [agent guide §10](../../docs/developer/agent-guide.md). Use one device on a shared box, or when a
> verdict hinges on a few percent.

## Guard

Before each child: no compute process on the GPU and utilization at most `--util-ceiling`. Otherwise
the tuner waits up to `--guard-wait` and then stops. After each child: a compute process on the GPU
discards that child's numbers and the child is retried. Any process with a context counts, and an
entry nvidia-smi shows as `[N/A]` counts as foreign. The tuner warns at start when a listed GPU drives
a display, or when another GPU has compute processes. Without nvidia-smi the tuner stops; `--no-guard`
skips the guard.

`--allow-idle-foreign` tolerates other users' idle contexts while utilization is at most
`--util-ceiling`. Each child remembers their pids, and only a new pid discards its numbers. A busy GPU,
or an `[N/A]` entry, still refuses. The raw files record `allow_idle_foreign` and `tolerated_foreign`
(`gpu:pid(user),..;..`) in `meta`. Each `pass` and `rep` record carries the pids the child ran beside.

## Raw JSONL (schema 1)

One file per (op, dtype, device): `<raw>/<op>.<dtype>.<device>.jsonl`, one flat JSON object per line.
Times are milliseconds per call for the whole batch.

| kind | fields |
|---|---|
| `meta` (first line) | `schema`, `op`, `dtype`, `device`, `device_name`, `batchlas` (8-hex HEAD, `-dirty` if tracked files differ), `kernels` (§6.4 hash), `kernel_sources` (`\|`-joined), `date`, `keys`, `candidates` (`\|`-joined, list order), `reps`, `warm_s`, `passes`, `tie`, `remeasure`, `refine_ratio` (0 = off), `cap_gib`, `ld_pad`, `devices`, `allow_idle_foreign`, `tolerated_foreign`, `argv` |
| `rep` | `op`, `dtype`, `device`, key fields, `pass` (1-based), `attempt` (0, or 1 for a re-measure), `reverse`, `gpu`, `tolerated_foreign`, `cand`, `rep`, `slot` (position in the rotated order), `ms` |
| `pass` | as `rep` without `rep`, `slot`, `ms`; plus `status` (`ok`, `skipped`, `bad`, `error`), `reason`, `median_ms`, `residual`, `info_nonzero`, `reps` |
| `retry` | `op`, `dtype`, key fields, `mode`, `error` (a failed child that was run again) |
| `cell` (last lines) | key fields, `round` (0 = coarse grid), `refined`, `status` (`ok`, `none`, `skipped` = over the cap), `reason`, `final_attempt` (-1 when none), `ranked` (`\|`-joined) |
| `stalled` | `op`, `dtype`, `note` (a bracket wider than the ratio because its midpoint has no winner) |

Key fields use the op's key names. Integer values are JSON integers. With `--ledger`, `import_schema1` reads `meta` and the
`pass` records of each cell's highest attempt that timed anything. A candidate enters the table when it
is `ok` in every pass of that attempt, and the run becomes a `custom` run of the ledger.

## Gate mode (§10.3)

Gate mode compares the old choice with Auto for each cell. Cells come from `--old-csv` (columns
`dtype`, every key name, `old`; a previous gate CSV works) or from the grid flags with `--parent-bin`,
a `batchlas_tune` built from the parent commit. Its choice is read from the coverage `reached` row.

- Cells where both agree are `same` and are not timed. Otherwise each pass times `auto` against the old
  spelling pinned, interleaved, with pass 2 in reversed order.
- Verdicts: `FAIL` when new/old > `--gate-limit` in both passes; `BAD_ROW` when an arm failed; `ERROR`
  when a probe failed; `cap` for a cell over `--cap-gib` (neither probed nor timed); otherwise `pass`.
- Exit status: 1 on any `FAIL`; 3 when any row is `ERROR` or `BAD_ROW`, or no cell was gated; else 0.
- With `--raw`, `gate.<op>.<device>.jsonl` holds `gate` records (`pass`, `arm`, `status`, `reason`, `median_ms`).

## Specs

One spec per op, registered with `BATCHLAS_TUNE_REGISTER` and listed in `CMakeLists.txt`. `--list`
prints the registered ops. Adding one is step 4 of
[adding an op to kernel selection](../../docs/extending.md#adding-an-op-to-kernel-selection).

Each `<op>_spec.cc` implements `OpSpec` (`spec.hh`, not `select::OpSpec`) and provides:

- key names, candidates and grid axes from the op's `choice.hh`;
- a problem builder: potrf and posv use SPD A; posv and trsm use random B; trsm uses a diagonally
  dominant triangular A with the other triangle poisoned;
- verification through `batchlas::verify`, the 4 GiB cap, and how an old coverage route maps to a spelling;
- the kernel-source list for the kernel hash, between the `kernel-sources-begin` and
  `kernel-sources-end` markers: hash v2, sha256 over `"<sha256(file without comments and
  whitespace)>  <path>\n"` lines, first 8 hex digits, paths relative to the repository
  ([kernel hashes](../../docs/design/tuning-data.md#tuning-data-kernel-hash)).
  `check_tuned_tables.py` and `sweep_to_table.py` parse this block, and the driver refuses to run
  when it differs from the compiled list (rebuild).

Limits of the hash: posv's list holds only its own kernels, but its `cta` and `blocked` times depend on
potrf's and trsm's choices (not yet followed). trsm's list covers its native kernels, not the vendor
TU. Its `uplo` and `diag` are hidden axes fixed at L and N. Its `trans=T` times ConjTrans for complex
scalars, and it has no workspace, so a pin is probed by one untimed run.

gemv, trmm, symm, syrk and syr2k take no workspace either and are probed the same way. Their
non-key arguments are fixed: trmm uplo L, trans N, diag N; symm side L, uplo L; syrk uplo L; syr2k
uplo L, trans N; gemv's `trans=T` times ConjTrans for a complex scalar. Inputs carry large finite
poison in the triangle an op must not read (A for trmm and symm) or write (C for syrk and syr2k: a
changed element is a nonzero info), and trmm's output is re-poisoned before every run. symm, syrk
and syr2k are real-only: `--dtype cfloat` is refused and `--list` prints `-`. symm and syrk key on
`form`, which `form_of` derives from the extents, so their `grid()` holds only the consistent cells
of the lattice (`--grid` filters it, as gemm's does). trmm's and symm's `expand` call the public
gemm, so their deps list gemm's kernels and `gemm.cc`, and their tables should be tuned after gemm's.

syev and gesvd check eigen- and singular values against host LAPACKE (`<batchlas/verify/reference.hh>`;
without LAPACKE those arms are `bad`). gesvd has no batch key: each cell runs at a batch derived from m and n
(a power of two in [128, 16384] with m n batch <= 2^24), and its grid drops the cells no call keys
(Hermitian non-square, square thin). One vendor refusal lives in a spec because the library would
not report it as a refused pin: gesvd's values-only non-square vendor call faults the device (known
defect 15) although `can_run` admits it. spmm's `vendor` arm needs none: where `can_run` refuses it
(known defect 17), the strict pin refuses it too. spmm tunes the GPU tables only; its cpu tables stay
transcribed.

Every spec verifies through `batchlas::verify` ([verification library](../../docs/design/verification.md)):
inputs from `MatrixView` fills, the `MatrixView` checks on the default items, the bound of the check's
kind. A spec with one check reports its raw value against `verify::bound`; a spec with several
(getrf, geqrf, orgqr, syev, gesvd) reports the worst value divided by its own bound, against 1.
