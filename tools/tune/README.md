# batchlas_tune {#tune_tool_readme}

> **Status:** current. Reference for the tuner's commands, flags and output formats.

`batchlas_tune` times every `candidates<T>()` entry of an op over its grid (each strictly pinned
through the public entry point), appends the results to the ledger
`benchmarks/results/tuning/ledger/<op>.<device>.jsonl` and writes `tuned/<op>.<device>.txt` from it
with `scripts/sweep_to_table.py`. Layout: @ref design_tuning_data; tables:
[tuned/README.md](../../tuned/README.md); algorithm: @ref design_tiered_tuning; selection:
[flat kernel selection](../../docs/design/flat-kernel-selection.md) §6.

Build with `scripts/b batchlas_tune` (`-DBATCHLAS_BUILD_BENCHMARKS=ON`, CUDA or ROCm backend). Run
`batchlas_tune`, the SYCL-free launcher; it starts `batchlas_tune_impl` with `CUDA_VISIBLE_DEVICES=""`
and gives each child its GPU. `tune_replay` is the host-only replay ("Offline replay").

## Usage

    batchlas_tune potrf --tier preview --dtype float,double --devices 1 --plan   # plan + estimate, no GPU
    batchlas_tune potrf,trsm --tier preview --devices 1 --out /tmp/tables        # measure, then tables
    batchlas_tune all --tier coarse --devices 1,2 --budget 6 --progress-fd 3 3>events.jsonl
    batchlas_tune core --tier preview --dtype float,cfloat --devices 1             # a quick first table
    batchlas_tune --status | --list | --guard-probe          # ledger matrix / ops+candidates+hash / guard view
    batchlas_tune --import-raw /tmp/raw/trsm.float.sm_120.jsonl                  # schema-1 raw -> deep run
    # custom protocol, raw only (an A/B over hidden axes: sweep_to_table refuses such a ledger)
    batchlas_tune trsm --devices 1 --grid uplo=L:U --grid diag=N:U --no-refine --batches 8192 --raw /tmp/ab

Modes (none given: the tuner stops and asks):

- **Tiered** (`--tier`): measures only cells missing, stale or below the tier into the ledger;
  tables only with `--out`.
- **Custom** (any of `--reps`, `--warm`, `--passes`, `--remeasure`, `--refine-ratio`, `--no-refine`,
  `--no-jit`, `--ld-pad`, `--raw`; not with `--tier`): the two-pass protocol, one op; a ledger run
  of tier `custom` (below preview) only with an explicit `--ledger DIR`.
- **Gate** (`--gate`): old choice against Auto, one op ("Gate mode").

## Flags

| flag | default | meaning |
|---|---|---|
| `<op>` | required (not for `--list`, `--status`, `--import-raw`) | one op, a comma list, `all`, or `core` (gemm, trsm, syrk, herk, potrf, getrf, geqrf, syev). Tiered runs order ops by dependency (`op_order()`, `schedule.cc`) and skip, with a note, a dtype a spec refuses |
| `--tier` | none | `preview` (round 0 only: no frontier or bisection rounds; a demand-driven grid at key hash % 12), `coarse`, `deep`, `calibrate`; `custom` only with `--remeasure-keys` (exactly the named cells, every candidate, no lattice or refinement) |
| `--prior DEVICE` | newest fully timed `tuned/<op>.*.txt` | calibrate only: the borrowed table |
| `--probe-stride N` | 4 | calibrate only: probe every N-th value of each log axis |
| `--plan` | off | print cells, skips with reasons and the estimate (summed over GPUs), then exit. No GPU |
| `--budget H` | none | stop refinement after H hours; round 0 always completes |
| `--remeasure-keys FILE` | none | lines `<op> <dtype> <name=v,...>` (ledger key order): measured again in round 0 at the run's tier, past `--max-dim` included (not past `--cap-gib`) |
| `--only-named` | off | with `--remeasure-keys`: measure only those cells (refused with calibrate) |
| `--progress-fd N` | none | JSON events on fd N ("Progress events") |
| `--ledger DIR` | `<repo>/benchmarks/results/tuning/ledger` | ledger root, one `<op>.<device>.jsonl` per table |
| `--out DIR` | none | after the run, regenerate each measured op's table into DIR via `sweep_to_table.py` (custom: needs `--ledger`) |
| `--repo DIR` | the configured source tree | repository the tuner reads tables, specs and the ledger default from |
| `--allow-dirty` | off | measure from a `--repo` with uncommitted tracked changes; the run is stamped `<sha>-dirty` and no table reader uses it |
| `--device-key sm_NN` | first `--devices` GPU | device whose ledger `--plan` reads |
| `--devices` | required (not for `--plan`) | GPU indices in PCI order, inside `CUDA_VISIBLE_DEVICES` if set; UUID lists refused |
| `--dtype` | `float` (gate with `--old-csv`: every dtype in the CSV) | comma list of `float,double,cfloat,cdouble` |
| `--cell-overhead-s` | 0.49 | per-child start-up in the estimate |
| `--no-worker` | off | race every cell in a fresh child instead of the per-GPU worker |
| `--audit-fraction F` | preview, coarse 0.02; deep 0.01 | share of worker cells re-raced in a fresh child |
| `--cap-gib` | 4 | `skip:cap` for cells whose matrices exceed this |
| `--max-dim` | 2048 tiered, off custom | `skip:dim` above this matrix dimension (never `batch`); 0 = off |
| `--n-list`, `--batches`, `--nrhs-list`, `--uplo`, `--grid key=v1:v2` | the op's `choice.hh` grid | replace whole axes; a batch axis turns the ladder off |
| `--batch-ladder` | `auto` | `auto` (only on a ledger with no run or a ladder run), `default`, `off`, or classes `c,c,...` ([the batch ladder](../../docs/design/tiered-tuning.md#engine-the-cu-aligned-batch-ladder)) |
| `--cus N` | the GPU's compute units | ladder unit; `--plan` without it reads the GPU name (RTX PRO 6000 Blackwell 188, RTX 4090 128) or refuses |
| `--raw DIR` | `<repo>/benchmarks/results/tuning` | custom: schema-1 raw JSONL (Git LFS) |
| `--reps`, `--warm`, `--passes`, `--remeasure` | 16, 1.5 s, 2, 0.10 | custom: reps, warm-up per candidate, passes; re-measure when pass medians differ by more |
| `--refine-ratio`, `--no-refine`, `--no-jit`, `--ld-pad` | 1.1, -, -, 0 | custom: bisection stops at hi/lo < ratio; no refinement; no JIT pass; ld = n + pad |
| `--no-guard`, `--guard-wait`, `--util-ceiling` | on, 300 s, 5 % | idle guard ("Guard") |
| `--guard-backend` | `auto` | `auto` (NVML, else nvidia-smi with a warning), `nvml`, `nvidia-smi` |
| `--lock-dir` | `/tmp` | per-GPU flock files `batchlas_tune_gpu<N>.lock` |
| `--cell-timeout` | 1800 s | a longer child is killed and counts as failed |
| `--gate`, `--old-csv`, `--parent-bin`, `--gate-csv`, `--gate-limit` | `--gate-limit` 1.05 | gate mode |

`--allow-idle-foreign` was removed and is refused. `--cell`, `--worker`, `--info` and the other
flags not listed are the driver's internal child interface.

## Tiered mode

Rounds, frontier, race, verification, audits, workers and the estimate: the "Engine:" sections of
@ref design_tiered_tuning. Driver `tiered_driver.cc`, planning `schedule.cc`. Cell outcomes printed:

| outcome | meaning |
|---|---|
| `skip:dim` / `skip:cap` | over `--max-dim` / `--cap-gib` |
| `skip:current` | a current record at the same or higher tier exists |
| `partial:<families>` | partly stale: only changed families, the stored winner and runner-up re-race |
| `protocol` | a record from another protocol: re-measured in full |
| `skip:single` | at most one runnable candidate: written as an untimed row |

Candidate statuses: `ok`, `eliminated` (keeps its median, ranked after survivors), `skipped`
(`pin refused (strict): ...`), `bad` (failed verification), `error`.

### Progress events

One JSON object per line on `--progress-fd`; every event also carries `t` (Unix seconds).

| event | fields |
|---|---|
| `plan` | `cells`, `est_s` (lattice only), `refine_cells`, `est_refine_s`, `est_total_s` |
| `plan_job` | per (op, dtype), before `plan`: `op`, `dtype`, `device`, `cells`, `measure`, `mix`, `est_s`, `refine_cells`, `est_refine_s` |
| `refine_cap` | `op`, `dtype`, `lattice`, `refined`, `cap`, `dropped` |
| `cell_start` | op, dtype, key fields, `gpu` |
| `cell_done` | op, dtype, key fields, `ranked` (`a\|b`), `tier`, `round` (absent on a single-candidate cell) |
| `eliminated` | op, dtype, key fields, `cand`, `round` |
| `audit` | op, dtype, key fields, `verdict`, `fresh_ms`, `warm_ms`, `fresh`, `audit_s` |
| `worker_restart` | op, dtype, key fields, `gpu`, `restarts`, `fallback` |
| `throttled` | op, dtype, key fields, `gpu`, `capped`, `samples` |
| `ld_audit` | op, dtype, key fields, `cand`, `verdict` (`fail`), `reason` |
| `ld_audit_summary` | `op`, `dtype`, `passed`, `skipped`, `failed`, `failures` |
| `schedule` | `switches`, `restarts`, `switch_restarts` |
| `done` | none |

## Custom protocol (§6.3)

Specified in [flat kernel selection](../../docs/design/flat-kernel-selection.md) §6.2-§6.3. One
`--cell` child per (cell, pass), as the SLM carve-out is sticky per CUfunction; it verifies items 0,
batch/2 and batch-1. A failed child is retried once, then each candidate runs alone (failures:
`error`, `crashed alone: ...`). Ties (3%, `candidates<T>()` order): `rank()` in `tune_core.cc`.

## Multi-GPU

`--devices 1,2,3` runs one child per GPU under an exclusive flock, shards cells round-robin and
keeps both passes of a cell on one GPU; all GPUs must share one device key. This departs from the
one-measuring-process rule in [agent guide §10](../../docs/developer/agent-guide.md): use one
device on a shared box, or when a verdict hinges on a few percent.

## Guard

Before each cell or child a GPU in `--devices` must hold no other compute context, idle or not
(`[N/A]` counts as foreign), and, via nvidia-smi, utilisation at most `--util-ceiling`; else the
tuner waits up to `--guard-wait`, then stops. A run that saw foreign activity is discarded and
re-run, never an arm error. Details: [the guard](../../docs/design/tiered-tuning.md#engine-guard-retries-a-failed-nvidia-smi-query);
no idle tolerance: [C5](../../docs/perf/batch-sensitivity.md#batch-sensitivity-c5-co-tenant).

## Raw JSONL (schema 1)

One file per (op, dtype, device): `<raw>/<op>.<dtype>.<device>.jsonl`, one flat JSON object per line.
Times are milliseconds per call for the whole batch. Key fields use the op's key names.

| kind | fields |
|---|---|
| `meta` (first line) | `schema`, `op`, `dtype`, `device`, `device_name`, `batchlas` (8-hex HEAD, `-dirty`), `kernels`, `kernel_sources` (`\|`-joined), `date`, `keys`, `candidates` (`\|`-joined), `reps`, `warm_s`, `passes`, `tie`, `remeasure`, `refine_ratio` (0 = off), `cap_gib`, `ld_pad`, `devices`, `guard_backend`, `argv` |
| `rep` | `op`, `dtype`, `device`, key fields, `pass` (1-based), `attempt` (0, or 1 for a re-measure), `reverse`, `gpu`, `cand`, `rep`, `slot`, `ms` |
| `pass` | as `rep` without `rep`, `slot`, `ms`; plus `status` (`ok`, `skipped`, `bad`, `error`), `reason`, `median_ms`, `residual`, `info_nonzero`, `reps` |
| `retry` | `op`, `dtype`, key fields, `mode`, `error` |
| `cell` (last lines) | key fields, `round` (0 = coarse grid), `refined`, `status` (`ok`, `none`, `skipped`), `reason`, `final_attempt` (-1 when none), `ranked` (`\|`-joined) |
| `stalled` | `op`, `dtype`, `note` (a bracket whose midpoint has no winner) |

`--import-raw` and `--ledger` read `meta` and the `pass` records of each cell's highest attempt that
timed anything; a candidate `ok` in every pass of it enters the table.

## Offline replay (`tune_replay`)

Host-only (`build/tools/tune/tune_replay`), one JSON line per result. Definitions:
[the slowdown replay](../../docs/design/tiered-tuning.md#engine-flip-frontier-and-the-slowdown-replay).

- `--raw <jsonl> --tier preview|coarse|deep`: re-race a raw exhaustive sweep and score table
  misrank (pre-frontier model). Options: `--confidence`, `--min-reps`, `--max-reps`, `--stride`,
  `--refine`, `--refine-mode geometric|index`, `--refine-margin`, `--refine-cap`, `--verify-s`,
  `--cell-overhead-s`, `--no-holdout`, `--axis-keep name=v:v`, `--axis-stride name=k`, `--print-axes`.
- `--slowdown` (implied by `--table`/`--ledger`): sources `--table tuned/<op>.<dev>.txt`,
  `--ledger <op>.<dev>.jsonl` (needs its `--table`) and `--raw`, each repeatable. `--scheme a,b` from
  `full`, `lattice`, `stride2`, `stride4`, `stride2+frontier`, `stride4+frontier`,
  `stride2+frontier+prune`, `lattice+bisect`, `stride2+frontier+bisect` (models today's driver;
  default: all); `--k` (6) and `--slower` (0.05) set the frontier rule. Output per (scheme, source)
  and per (scheme, `overall/<kind>`): `lattice`, `kept`, `arm_runs`, `geomean`, `p99`, `p999`,
  `slow10`, `misrank`, `worst`, `worst_key`, `measured_*`, `failures`, `failure_list`, `gate`
  (no failures, geomean <= 0.05% and p99.9 <= 1.10).
- `--batch-ladder CUS` (`--batch-classes c,c`, `--batch-extra b,b`) adds `ladder` and
  `ladder+frontier+bisect` and prints `kept_batches`; `--drop-nonmonotone F` and `--min-batch B`
  filter cells first.
- `--prior tuned/<op>.<dev>.txt` (repeatable) adds `calibrate`, `calibrate8`, `calibrate-probe` and
  `prior`, with `probes`, `disagreed` and `prior_rows` in the output.

## Gate mode (§10.3)

Compares the old choice with Auto per cell. Cells come from `--old-csv` (columns `dtype`, every key
name, `old`; a previous gate CSV works) or from the grid flags with `--parent-bin`, a
`batchlas_tune` built from the parent commit (its choice is read from the coverage `reached` row).
The old choice is pinned verbatim, so it must be a current spelling.

- Cells where both agree are `same` and untimed. Otherwise each pass times `auto` against the pinned
  old spelling, interleaved, pass 2 reversed.
- Verdicts: `FAIL` when new/old > `--gate-limit` in both passes; `BAD_ROW` when an arm failed;
  `ERROR` when a probe failed; `cap` over `--cap-gib`; otherwise `pass`. Written to `--gate-csv`.
- Exit status: 1 on any `FAIL`; 3 on any `ERROR` or `BAD_ROW`, or when no cell was gated; else 0.
- With `--raw`, `gate.<op>.<device>.jsonl` holds `gate` records (`pass`, `arm`, `status`, `reason`,
  `median_ms`).

## Specs

One `<op>_spec.cc` per op implements `OpSpec` (`spec.hh`), registers with `BATCHLAS_TUNE_REGISTER`
and is listed in `CMakeLists.txt` (step 4 of
[adding an op to kernel selection](../../docs/extending.md#adding-an-op-to-kernel-selection)). It
gives keys, candidates and grid from `choice.hh`, a problem builder, verification through
`batchlas::verify` (@ref design_verification), the coverage-route-to-spelling map, and the
`kernel-sources-begin`/`-end` list that `check_tuned_tables.py` and `sweep_to_table.py` parse and the
driver checks against the compiled one ([kernel hashes](../../docs/design/tuning-data.md#tuning-data-kernel-hash)).

- Fixed non-key arguments: trsm `uplo` L, `diag` N (hidden axes); trmm uplo L, trans N, diag N;
  symm side L, uplo L; syrk uplo L; syr2k uplo L, trans N. `trans=T` is ConjTrans for complex.
- symm, syrk, syr2k are real-only (`--list` prints `-`); hemm, herk, her2k complex-only.
- syev and gesvd check values against host LAPACKE (absent: those arms are `bad`). gesvd has no
  batch key (a power of two in [128, 16384], m n batch <= 2^24) and refuses the values-only
  non-square vendor call (known defect 15). spmm tunes GPU tables only.
