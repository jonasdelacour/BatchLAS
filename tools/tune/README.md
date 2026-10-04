# batchlas_tune

The tuner that produces `tuned/<op>.<dtype>.<device>.txt` (spec: `docs/design/flat-kernel-selection.md`
§6; plan: `flat-kernel-selection-phase3-plan.md` §1 "P3.2", §2 option B, §4). It times every entry
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

    # tune: raw JSONL per dtype, then tables (omit --out for raw only)
    batchlas_tune potrf --dtype float,double,cfloat,cdouble --devices 1 \
                  --raw benchmarks/results/tuning --out tuned

    # a smaller protocol for a smoke test or a spot check
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
| `--dtype` | `float` (gate with `--old-csv`: every dtype in the CSV) | comma list; one raw file and one table per dtype |
| `--devices` | required | GPU indices (nvidia-smi / PCI order); inside `CUDA_VISIBLE_DEVICES` when that is set; see "Multi-GPU" |
| `--raw DIR` | `<repo>/benchmarks/results/tuning` | raw JSONL (Git LFS there) |
| `--out DIR` | none | table directory; runs the converter after each dtype |
| `--reps`, `--warm`, `--passes` | 16, 1.5 s, 2 | §6.3 protocol |
| `--remeasure` | 0.10 | re-measure a cell once when pass medians differ by more |
| `--refine-ratio`, `--no-refine` | 1.1 | §6.2 bisection stops when hi/lo < ratio |
| `--no-jit` | off | skip the throwaway JIT pass |
| `--cap-gib` | 4 | skip cells whose matrices exceed this |
| `--ld-pad` | 0 | ld = n + pad (non-natural leading dimension) |
| `--n-list`, `--batches`, `--nrhs-list`, `--uplo`, `--grid key=v1:v2` | the op's `choice.hh` grid | replace whole grid axes |
| `--no-guard`, `--guard-wait`, `--util-ceiling` | on, 300 s, 5 % | the idle guard; see "Guard" |
| `--allow-idle-foreign` | off | tolerate another user's idle CUDA contexts; see "Guard" |
| `--lock-dir` | `/tmp` | per-GPU flock files `batchlas_tune_gpu<N>.lock` |
| `--cell-timeout` | 1800 s | a child running longer is killed and counts as a failed child |
| `--gate`, `--old-csv`, `--parent-bin`, `--gate-csv`, `--gate-limit` | 1.05 | gate mode |

## Protocol (§6.3) and where it lives

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
tuners never share a device. **This departs from AGENTS.md §10's one-measuring-process-per-box rule**
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
At start the tuner warns when a listed GPU drives a display (AGENTS.md §13) or when any other GPU
on the box has compute processes (AGENTS.md §10). Without nvidia-smi the tuner stops; `--no-guard`
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

Cells come from `--old-csv` (columns `dtype`, every key name, `old`; legacy aliases such as
`native:lpanel` are normalised; empty fields and `"..."` quoting are read, so a previous gate CSV
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

`<op>_spec.cc` implements `OpSpec` (`spec.hh`): key names, candidates and grid axes from the op's
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
