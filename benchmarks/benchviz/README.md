# benchviz: BatchLAS vs vendor LAPACK and BLAS

This tool measures and plots BatchLAS against the vendor LAPACK library the build links:
cuSOLVER/cuBLAS on CUDA, rocSOLVER/rocBLAS on ROCm. It covers every LAPACK op that has a
vendor route. The output is publication-ready vector figures, and a local dashboard shows them
updating while the campaign runs. It needs Python 3 with matplotlib, numpy and pandas, plus a
TeX installation with `latex` and `dvipng` for the Computer Modern text. It installs nothing else.

```sh
# 1. Build the harnesses (the library plus four targets).
cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ -DBATCHLAS_BUILD_BENCHMARKS=ON
cmake --build build --target factor_bench syev_benchmark ormqr_benchmark gesvd_vendor_benchmark \
    gemm_benchmark gemv_benchmark trsm_benchmark trmm_benchmark syrk_benchmark syr2k_benchmark spmm_benchmark \
    stedc_benchmark steqr_benchmark sytrd_blocked_benchmark sytrd_cta_benchmark sytrd_sy2sb_benchmark \
    sytrd_sb2st_benchmark -j"$(nproc)"

# 2a. Start the dashboard and launch runs from the browser.
python3 benchmarks/benchviz serve            # http://127.0.0.1:8765
#     From a laptop: ssh -L 8765:localhost:8765 <gpu-box>

# 2b. Or run from the terminal. The dashboard can watch this too.
python3 benchmarks/benchviz run --ops potrf,getrf,syev --types float,double --preset quick
python3 benchmarks/benchviz run --ops all --types all --preset full --campaign paper-4090

# 3. Re-render every figure (e.g. after a style change).
python3 benchmarks/benchviz plot paper-4090

# 4. Compare any two logs, e.g. one run per build (see "Comparing logs").
python3 benchmarks/benchviz run --campaign before --build-dir ../old-checkout/build --ops syev,stedc --types float
python3 benchmarks/benchviz run --campaign after  --build-dir build                 --ops syev,stedc --types float
python3 benchmarks/benchviz compare before after
```

## Which build is measured

benchviz does not compile anything. It runs whatever harness binaries it finds, searching in this order:
`build/` of the checkout it runs from, then `build/presets/{benchmarks,dev-tests}`, then, when that
checkout is a worktree, the main checkout's `build/`. `run --build-dir <dir>` overrides the search.
So the code you measure is the code **last compiled** into that directory. It is not the checkout's HEAD.
A worktree whose `build/` was compiled days ago measures days-old kernels.

```sh
python3 benchmarks/benchviz info     # the build a run would use now, and every campaign's state
```

`info`, the `serve` banner and the dashboard header all show that build: its estimated source
commit (`~sha`, the last commit before the library was linked), its build time, and a warning when
it is behind. "Behind" means source files changed after it was built, or main has source commits it
lacks. Each campaign records the builds it used in `campaign.json` under `provenance.builds`. A
resume on a different build adds an entry there, and every result row records its `binary` path.
Campaigns created before this was added show "not recorded".

For a read-only copy you can open on a phone or send to someone, `export` writes a static page plus
downscaled figures: `python3 benchmarks/benchviz export <campaign>... --out <dir>`. Pass `--fragment`
when the host wraps the page in its own `<html>` skeleton.

Campaigns are stored in `benchviz_runs/<name>/` (git-ignored):

| File | Contents |
|---|---|
| `results.jsonl` | Append-only, one row per (cell, arm) |
| `campaign.json` | The request and its provenance: GPU, driver, the builds measured, and the benchviz checkout's SHA and dirty flag |
| `figures/<op>/{speedup_n,throughput_n,heatmap}.{pdf,png}` | The per-op figures |
| `figures/_summary/summary_<precision>.{pdf,png}` | The cross-op summaries |

A log comparison's `campaign.json` names its two source campaigns (`kind: compare`), and it has no
`results.jsonl`.

`--gpu 0,1` (or both GPU chips in the run panel) splits a campaign's cells across the cards, one
worker per GPU. Both arms of a cell always run on the same card, so every ratio compares like with
like. The two workers share the host CPU, so for figures you will publish, prefer one card. Stop kills
the cell in flight within about half a second, and a stopped campaign resumes where it left off.

Re-running with the same `--campaign` resumes it and only measures the missing cells. Passing
different `--ops` or `--types` extends it.

## What is compared

| Op | Harness | Vendor arm | BatchLAS arm |
|---|---|---|---|
| potrf, getrf, getrs, geqrf, orgqr | `factor_bench` (verified residuals) | `--arms=vendor` | `--arms=native` |
| gesv, posv | `factor_bench` | sub-ops pinned vendor | sub-ops pinned native |
| ormqr | `ormqr_benchmark` | `BATCHLAS_ORMQR_ROUTE=vendor` | `=native` |
| syev (with vectors) | `syev_benchmark` | `BATCHLAS_SYEV_ROUTE=vendor` | `=native`, tuned nb and fuse |
| gesvd (n ≤ 32) | `gesvd_vendor_benchmark` | direct `gesvdjBatched` call | `gesvd_cta` |
| gemm, gemv, trsm, spmm | `<op>_benchmark` | `BATCHLAS_<OP>_ROUTE=vendor` (cuBLAS / cuSPARSE) | `=native` |
| trmm, syrk, syr2k (float) | `<op>_benchmark` | `=vendor` | `=triangular` (triangular-tile kernels) |
| stedc, steqr, sytrd, sytrd_cta, sy2sb, sb2st | their own harnesses | none | the harness's direct call |

The n-figures use square cells (m = n = k). spmm uses a random CSR pattern with 16 nonzeros per row,
16 right-hand sides, and n rows. The rectangular ops are also swept over a second dimension; see
"Rectangular ops".

Some ops are left out, each for a stated reason:
- **syevx:** no vendor route. (stedc, steqr and the sytrd family have none either; they are measured
  alone. See "Sub-operations".)
- **symm:** its native arm cannot be pinned; it is expand-then-gemm through the routed gemm.
- **hemm, herk, her2k:** no route variable and no coverage rows.
- **syrk and syr2k in double, and trmm in double:** no native kernel. For syrk double, nothing is recorded.
- **Plain `native` for syrk and syr2k:** it selects a wrong-answer route or throws, which is why those
  arms pin `triangular`.

**The BatchLAS arm is pinned `native`, never `auto`.** Where `auto` already prefers the vendor,
an `auto` arm would time the vendor twice and report 1.0×.

**Timing rules.**
- One process per (cell, arm). The reasons are in `benchmarks/run_factor_grid.sh`.
- Every run goes through `gpu_guard.sh`, which requires an exclusive GPU and discards
  contaminated cells.
- Ratios are only quoted at saturation. The n-figures use the largest batch at which both arms
  were measured; the heatmap shows the full grid.

**Correctness.**
- `factor_bench` rows are verified on the host. It re-measures a row flagged `bad` because it
  was noisy, and drops a row flagged `bad` for any other reason.
- The minibench ops (ormqr, syev, gesvd) are timed but not verified.

## Routes

Each process runs with `BATCHLAS_COVERAGE_OUT` set, and each row records the route that
actually ran (`route`) plus every sub-op route (`subroutes`). A pin is never taken on trust:

- A BatchLAS arm that resolved to a vendor route is dropped from the plots. The reason is shown
  in the dashboard's failure table.
- A vendor arm that resolved to a native route is dropped the same way.
- For the composed ops (gesv, posv), both arms take the outer `blocked` route, so their sub-op
  routes decide which arm they belong to.
- An **open marker** in the speedup figure means the BatchLAS arm is native at the top but calls
  a vendor sub-op (for example, blocked syev still uses cuBLAS gemm).
- For getrs, orgqr and ormqr this cannot be determined. Their harness runs an untimed
  setup factorization in the same process, and its routes land in the same coverage file.

### Forced-route sweeps (`--sweep routes`)

A routing cost model needs every route timed on the same cell, not only the one the native
walk picks. `--sweep routes` replaces the `batchlas` arm with one arm per native route,
named `route:<origin>:<algorithm>` (e.g. `route:native:lpanel`), plus the usual `vendor` arm.
The routes come from the op's `k<Op>Order` in `include/batchlas/blas/dispatch/route_<op>.hh`
(`ops.native_routes`), and each arm pins `BATCHLAS_<OP>_ROUTE` to its route.

A forced route skips `preferred()` but not `supports()`, so an unsupported pin falls back
to the automatic walk. Such a row is kept with `ok=false` and a reason starting
`unsupported/fallback: pinned X, reached Y`: it records `supports()`, and it is not retried
on resume. Every row also carries `cc` (the card's compute capability).

`--pass N` makes each pass its own rows (field `pass`), and odd passes run a cell's arms in
reverse order, so two passes give cross-pass reproduction with alternated A/B order. Run a
throwaway JIT pass into a separate campaign first. `potrf_upper` is potrf with
`--uplo=upper`; it is left out of `--ops all`.

```sh
python3 benchmarks/benchviz run --campaign potrf-routes --sweep routes --ops potrf --types all \
    --batches 128,2048,32768 --no-rect --no-plot --gpu 1,2,3 --pass 1
```

## Figures

| Figure | Content |
|---|---|
| `speedup_n` | Vendor time / BatchLAS time against n, at the saturated batch. One series per precision (S/D/C/Z) with a 2σ band, and a red dashed parity line at 1×. Log2 y-axis only when the range exceeds 8×. |
| `throughput_n` | GFLOP/s against n (LAWN 41 counts, complex = 4× real), or matrices/s for syev and gesvd. BatchLAS (○) vs vendor (★), one panel per precision, with 2σ bands. |
| `heatmap` | Speedup over the whole n × batch grid, in viridis with log2 colour steps. A red boundary follows the 1× crossing. Empty cells were not measured. |
| `summary_<t>` | Op × n speedup table at saturation, annotated with the value in each cell. |
| `speedup_2d` | Rectangular ops only. Speedup over the op's two shape axes (below), one panel per precision, annotated, with the same colour scale and 1× boundary as `heatmap`. Hatched cells are shapes the op does not define. |
| `throughput_2d` | Rectangular ops only. GFLOP/s over the same axes: one row per precision, BatchLAS and vendor side by side on a shared log colour scale. |

An op with no vendor arm draws `throughput_n` with BatchLAS alone, and `heatmap` becomes its
throughput over n × batch, annotated. It has no speedup figures and is left out of `summary_<t>`,
except in a log comparison, where every op gets them.

## Sub-operations

syev's stages are ops of their own, in the dashboard group "Eigensolver stages". No vendor library
ships them batched, so each has one arm, BatchLAS. A campaign gives their throughput, and a build
comparison ("Comparing logs") gives their speedup.

| Op | Harness | What it runs | Arguments |
|---|---|---|---|
| stedc | `stedc_benchmark` | divide and conquer, with vectors | threshold 0, merge and driver `Auto`: the tuning tables syev gets |
| steqr | `steqr_benchmark` | QR iteration, with vectors; n ≤ 32 is `steqr_cta`, above it `steqr_wg` | 50 sweeps, interleaved working vectors; the harness pins the Wilkinson shift |
| sytrd | `sytrd_blocked_benchmark` | blocked reduction to tridiagonal, syev_blocked's stage 1 | nb as syev_blocked chooses it, including its complex override at 256 < n ≤ 512 |
| sytrd_cta | `sytrd_cta_benchmark` | one-CTA reduction, n ≤ 32 | defaults |
| sy2sb | `sytrd_sy2sb_benchmark` | two-stage stage 1, dense to band | kd = 32, as `choose_two_stage_kd` |
| sb2st | `sytrd_sb2st_benchmark` | two-stage stage 2, the bulge chase | kd = 32 |

- **Spelled out, not defaulted.** Every argument is passed explicitly, through harness entry points that
  have existed for months, so an older build's binary runs exactly the same cell. A harness that
  mapped "0" to the library's default would make an old build silently measure something else.
- **The copies go stale.** `sytrd_nb` and `two_stage_kd` in `ops.py` copy the library's choices. A
  retune that changes them is not measured until those copies are updated too.
- **steqr stops at n = 128.** Above n = 32, `steqr_wg` is slow: 210 ms at n = 128, batch 256, against
  0.6 ms for stedc. n = 256 would take minutes per cell.
- **Routes.** None of these ops is dispatched, so the route is the function the harness calls. Coverage
  still records the dispatched sub-ops underneath (stedc's merge gemm, for example). An open marker
  means one of them resolved to a vendor library.
- **Flop counts.** sytrd, sytrd_cta and sy2sb use 4n³/3 (LAWN 41's sytrd; sy2sb has the same leading
  order). stedc, steqr and sb2st have no canonical count and are plotted in matrices/s.

## Comparing logs

`compare <baseline> <candidate>` pairs any two logs cell by cell: two builds, two runs of one build, a
campaign from last week against one from today, or runs made in two different checkouts. Neither log has
to have been made for the comparison.

```sh
python3 benchmarks/benchviz logs                     # every campaign in every checkout's benchviz_runs/
python3 benchmarks/benchviz compare baseline main-3df4e99-square
python3 benchmarks/benchviz compare ../other/benchviz_runs/x/results.jsonl y --exact
python3 benchmarks/benchviz compare before after --base-arm vendor --new-arm vendor   # did cuSOLVER move?
```

A source is a campaign name (looked up in `--root`, then in every checkout; an ambiguous name lists
the paths to choose from), a campaign directory, or its `results.jsonl`. The comparison is written to
`benchviz_runs/<name>/` (default `cmp-<baseline>-vs-<candidate>`), and the dashboard, `plot` and
`export` treat it like any other campaign. The dashboard's **Compare logs** button offers every log on
the box, grouped by checkout.

- **What the speedup is.** The candidate's chosen arm goes in the BatchLAS slot and the baseline's in
  the vendor slot, so every figure works unchanged; above 1× the candidate is faster. The arm is BatchLAS
  on both sides by default. `--base-arm` and `--new-arm` pick either arm of either log: vendor against
  vendor shows a driver or CUDA update, and BatchLAS against vendor within one log is that log's own speedup.
- **How cells are matched.**
  - First, the same cell (op, precision, shape, batch) in both logs.
  - Then, for a shape the two measured but never at a common batch (two grids, two memory budgets), each
    log's largest batch, compared by **time per matrix**. That assumes both points are saturated, which is
    what the top of a ladder is for. The overview counts these rescaled pairs, and `--exact` drops them.
  - A shape only one log measured has no ratio. The overview counts those too.
- **Labels.** `Build <sha>` when the log recorded its binaries. Older logs did not record them, so they
  are labelled `HEAD <sha>`: the checkout benchviz ran from, which is not necessarily what was measured.
- **The vendor control.** With BatchLAS on both sides, both logs also ran the vendor arm, and the vendor
  library did not change. So its ratio measures the machine: clocks, contention, the driver. The overview
  reports it, and flags it beyond 5 %. A control of 1.08× means every speedup in the comparison is
  inflated by roughly that much.
- **Live.** A comparison stores no rows; it re-reads both logs on every load. Comparing against a
  campaign that is still running therefore fills in as it runs, and the dashboard re-renders the figures
  when a source gains rows.
- **Warnings.** It warns about different devices, different batch grids, and two logs of the same
  binaries. The last makes every ratio run-to-run noise, which is a useful A/A test in its own right.

For figures you will quote, measure the two campaigns back to back on the same card, with the same grid.

## Rectangular ops

A square sweep cannot show where a tall-skinny QR or a small-k GEMM wins, so an op whose shape has a second
dimension also gets a 2-D sweep (its `Plane` in `ops.py`). Each shape runs once, at its saturated batch:
the largest power of two that fits the memory budget for that shape. The map's cells can therefore sit at
different batches, and every cell is a saturated ratio.

| Op | y axis | x axis | Constraint |
|---|---|---|---|
| geqrf, orgqr | rows m | columns n | n ≤ m (factor_bench) |
| ormqr | order of Q, m | reflectors n | n ≤ m |
| getrs, gesv, posv | order n | right-hand sides (1, 4, 16, 64, 256) | |
| gemm | output order m = n | inner dimension k | |
| gemv | rows m | columns n | |
| trsm | triangle order n | right-hand sides q | |
| syrk, syr2k | order n | rank k | |
| spmm | rows n | right-hand sides (4 … 128) | |

The square point of each plane is the square sweep's cell, not a second measurement. Left square:
potrf and getrf (factor_bench only takes m = n for them), syev, trmm (`trmm_benchmark`'s operands only
agree at m = n = k), and gesvd (its harness takes n only).

Every preset turns the sweep on; `--no-rect` (or the dashboard's "Rectangular shapes" box) turns it off.
It adds about 90% to a `quick` campaign of the vendor-compared ops in every precision (1,667 cells become 3,157;
the six sub-operations add another 328, one arm each). Campaigns created before this existed keep their
plan: a grid saved without the field loads with the sweep off.

The trsm map needs a `trsm_benchmark` built after `BM_TRSM` started reading its second argument as the
right-hand-side count. An older binary ignores it and times n × n, so `benchviz info` must not report
the build as behind.

The style is the house style of `plotting/stylesheet.py`:
- LaTeX Computer Modern, drawn at 20 × 10 in with 30 pt text and scaled down by `\includegraphics`.
- A full box around every panel, ticks pointing in, and a light full grid.
- Dotted connectors with markers in the order ○ △ □ ★ ◇, and the stylesheet's colour order.
- A frameless legend with enlarged markers.
- Units in brackets on every axis.
- Bold column titles on multi-panel figures, and no in-figure title (the caption carries it).

PDFs embed TrueType fonts (`pdf.fonttype 42`).

## Grids

A grid is a preset plus any overrides. The dashboard's run panel edits one field by field and previews
the n × batch cells it covers; the CLI takes the same fields as flags.

| Field | Flag | Meaning |
|---|---|---|
| orders | `--orders` | n values. `16,32,64` lists them, `4:512` doubles from 4 to 512, and `8:128:8` steps by 8. Blank uses each op's own ladder. Values outside an op's supported range are dropped. |
| batch mode | `--batch-mode` | `ladder` (default), `list`, or `saturated`, which takes one batch per n: the largest that fits the memory budget. |
| ladder | `--batch-min --batch-max --batch-step` | batch-min × step^k, up to min(batch-max, memory cap) |
| list | `--batches` | an explicit batch list, same syntax as orders |
| reps | `--reps` | timed repetitions per arm-cell |
| memory | `--mem-gib` | per-arm device-memory budget, which caps batch for each (op, n) |
| rectangular | `--rect` / `--no-rect` | also sweep the rectangular ops' shape planes, one saturated batch per shape |

| Preset | Grid |
|---|---|
| `smoke` | every 3rd order, batch ×16 from 256 |
| `saturation` | every order, only the saturated batch; enough for speedup vs n and throughput |
| `quick` | every order, batch ×4 from 64 |
| `full` | every order, batch ×2 from 32 |

On a 4090 an arm-cell takes about 3–8 s, mostly process start, warm-up and the exclusive-GPU check.

## ROCm

`factor_bench` selects its backend at compile time, so a ROCm build compares against
rocSOLVER/rocBLAS. Run with `--backend rocm`. `gpu_guard.sh` is NVIDIA-only, so on ROCm the
device is pinned with `ROCR_VISIBLE_DEVICES` and exclusivity is not checked. gesvd has no ROCm
arm, because `gesvd_vendor_benchmark` calls cuSOLVER directly.
