---
name: tune
description: Run, monitor or debug a batchlas_tune / benchviz tuning campaign, A/B a tuning constant, or find where the vendor still beats native ("run the tuner", "the tune run is doing nothing", "where do we still lose to cuBLAS").
---

# Tuning campaigns

The authoritative reference is `tools/tune/README.md` (flags, tiers, ledgers). This skill covers the operational traps that cost past sessions hours.

## Before starting
1. Run `scripts/agent/gpu_health.sh`. A FAIL stops GPU work: tell the user.
2. Run `scripts/agent/tune_procs.sh`. A held `/tmp/batchlas_tune_gpuN.lock` means that GPU is queued behind another run, possibly a days-old one from another worktree or one owned by another user. Report it; never kill a run you did not start without asking. Use `tune_procs.sh --kill PID` for your own: it takes the worker children with it.
3. The tuner needs `-DBATCHLAS_BUILD_BENCHMARKS=ON` in the tree and a clean checkout, and the ledgers present (`git lfs pull --include 'benchmarks/results/tuning/ledger/*'`).
4. Use `--plan` first and tell the user the estimate. Policy:
   - Saturating batch only (>= 128).
   - float and cfloat only on 1:64-FP64 GPUs (`--dtype float,cfloat`).
   - One measuring process per box.

## Running
- Launch with `run_in_background` and `--progress-fd 3 3>events.jsonl`.
- Then, also in the background, run `scripts/agent/wait_for.sh --pid <tuner pid> --log events.jsonl --stall-min 20`. Exit 3 means no event for 20 minutes: a stuck cell. Inspect it with `tune_procs.sh` and the run log, and tell the user. Do not wait for the user to ask for status; one hung cell went unnoticed for 5.5 h.
- Keep `--cell-timeout` (default 1800 s) unless a cell is known to be long.

## A/B of a tuning constant
Use the `BATCHLAS_TUNE_*` environment overrides (`include/batchlas/tuning_params.hh`) on one build. Each rebuild of `tuning_params.hh` costs a device relink, about 25 min. Port the winning value by hand once at the end. A/B protocol: `docs/developer/agent-guide.md` section 10 (alternate the order, at least 14 reps for an effect under 5%).

## Where the vendor still wins
`python3 scripts/agent/vendor_gaps.py <op> --device sm_89 [--dtype float]` reads the tuned tables at batch >= 128 and prints per-group win counts, geomean and worst ratios, and the worst rows. Do not write a new `losses.py`.

## After
- Run `python3 scripts/sweep_to_table.py --check`, then the tuned-table tests (`scripts/b -B <tree> tuned_tables_tests` and `ctest -R '^tuned_tables_tests$'`).
- Run `/prepush`. Ledgers are append-only LFS data; any other raw campaign output stays out of the repo unless the user asks.
