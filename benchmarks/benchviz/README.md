# benchviz: BatchLAS vs vendor LAPACK and BLAS {#benchviz}

`benchviz` times BatchLAS against the vendor library the build links (cuSOLVER/cuBLAS/cuSPARSE on
CUDA, rocSOLVER/rocBLAS on ROCm), writes vector figures, and serves a live dashboard. It compiles
nothing: build the `*_benchmark` / `factor_bench` targets first (`scripts/b <target>`). Needs Python 3
with matplotlib, numpy, pandas, and `latex` + `dvipng` (or pass `--no-tex`).

```sh
python3 benchmarks/benchviz serve                    # dashboard at http://127.0.0.1:8765 (ssh -L 8765:localhost:8765 <box>)
python3 benchmarks/benchviz info                     # which build a run would use, and every campaign's state
python3 benchmarks/benchviz list-ops
python3 benchmarks/benchviz run --ops potrf,getrf,syev --types float,double --preset quick
python3 benchmarks/benchviz run --ops all --types all --preset full --campaign paper-4090 --dry-run
python3 benchmarks/benchviz plot paper-4090 [--ops gemm] [--no-tex]
python3 benchmarks/benchviz logs [--all]              # every campaign in every checkout's benchviz_runs/
python3 benchmarks/benchviz compare before after [--exact] [--base-arm vendor --new-arm vendor] [--name X]
python3 benchmarks/benchviz import <campaign> <factor_bench or run_factor_grid.sh CSVs...>
python3 benchmarks/benchviz export <campaign>... --out <dir> [--fragment]   # static read-only copy
```

Every subcommand takes `--root` (campaign directory, default `benchviz_runs/`).

## Which build is measured

Search order: the checkout's `build/`, `build-benchmarks/`, `build-dev-tests/`, then the main
checkout's `build/` (from `git --git-common-dir`, or `BATCHLAS_MAIN_CHECKOUT`). `--build-dir`
(repeatable) overrides it. The measured code is what was **last compiled** into that tree, not
HEAD: run `info` first; it, the `serve` banner and the dashboard header show the build's estimated
commit (`~sha`) and warn when the build is behind.

### Toolchain and SYCL implementation

Each campaign's `campaign.json` `provenance.builds[]` records `sycl_impl` (`DPCPP`/`ACPP`),
`sycl_impl_version`, compiler, `build_type`, `fp_model` / `fp_contract` / `fp_contract_effective`,
`sycl_targets`, and for acpp `acpp_env`; every result row records its `binary`. `info` and run logs
warn on an icpx build without `-ffp-model=precise`, an acpp build whose effective fp contract is not
`on`, and targets that cannot reach the box's NVIDIA GPU.

Processes from an acpp tree get `ACPP_ADAPTIVITY_LEVEL=1`, `ACPP_RT_SCHEDULER=direct` and
`ACPP_APPDB_DIR=<campaign>/acpp-appdb` unless the environment sets them (environment wins and is
recorded). Rows record `acpp_jit` when the process JIT-compiled. For cache-warming passes, export one
`ACPP_APPDB_DIR` for all of them.

## Campaign files

`benchviz_runs/<name>/` (git-ignored):

| File | Contents |
|---|---|
| `results.jsonl` | append-only, one row per (cell, arm), with `route` and `subroutes` from `BATCHLAS_COVERAGE_OUT` |
| `campaign.json` | request and provenance (GPU, driver, builds, benchviz SHA); `kind: compare` for a comparison, which has no `results.jsonl` |
| `figures/<op>/{speedup_n,throughput_n,heatmap,speedup_2d,throughput_2d}.{pdf,png}` | per-op figures |
| `figures/_summary/summary_<precision>.{pdf,png}` | op x n speedup at saturation |

Re-running with the same `--campaign` resumes (measures missing cells only); new `--ops`/`--types`
extend it. `--gpu 0,1` splits cells across cards (both arms of a cell on one card; default `--gpu 1`).

## What is compared

| Op | Harness | Vendor arm | BatchLAS arm |
|---|---|---|---|
| potrf, getrf, getrs, geqrf, orgqr | `factor_bench` (verified) | `--arms=vendor` | `--arms=native` |
| gesv, posv | `factor_bench` | sub-ops pinned vendor | sub-ops pinned native |
| ormqr, syev, gemm, gemv, trsm, spmm | `<op>_benchmark` | `BATCHLAS_<OP>_ROUTE=vendor` | `=native` |
| trmm, syrk, syr2k (float) | `<op>_benchmark` | `=vendor` | `=triangular` |
| gesvd (n <= 32) | `gesvd_vendor_benchmark` | `gesvdjBatched` | `gesvd_cta` |
| stedc, steqr, sytrd, sytrd_cta, sy2sb, sb2st | own harnesses | none | direct call |

- The BatchLAS arm is pinned `native`, never `auto` (an `auto` arm can time the vendor twice).
- A pin is not trusted: an arm whose recorded route belongs to the other side is dropped from the
  plots (reason in the dashboard's failure table). An open marker in `speedup_n` means a native
  top-level route called a vendor sub-op. getrs, orgqr and ormqr sub-routes are not attributable
  (the untimed setup factorization shares the coverage file).
- Not covered: syevx, symm, hemm, herk, her2k; syrk, syr2k and trmm in double.
- Every run goes through `gpu_guard.sh` (exclusive GPU; `--no-guard` skips). Ratios are quoted at
  the largest batch both arms measured. Only `factor_bench` rows are verified.
- Sub-operation arguments (`sytrd_nb`, `two_stage_kd` in `ops.py`) copy the library's choices; a
  retune that changes them is not measured until those copies are updated. steqr stops at n = 128.

## Grids

| Flag | Meaning |
|---|---|
| `--preset` | `smoke` (every 3rd order, batch x16 from 256), `saturation` (saturated batch only), `quick` (default; x4 from 64), `full` (x2 from 32) |
| `--orders` | `16,32,64`, `4:512` (doubling), `8:128:8` (step 8); default each op's ladder |
| `--batch-mode` | `ladder`, `list`, `saturated` (largest batch fitting `--mem-gib`) |
| `--batch-min --batch-max --batch-step` | ladder batch-min x step^k up to min(batch-max, memory cap) |
| `--batches` | explicit list (implies `list`) |
| `--reps`, `--mem-gib` | timed reps per arm-cell; per-arm device-memory budget |
| `--rect` / `--no-rect` | rectangular shape planes (`Plane` in `ops.py`), one saturated batch per shape |
| `--grid-json` | a whole grid as JSON (what the dashboard sends) |
| `--backend cuda\|rocm`, `--no-plot`, `--replot-s`, `--dry-run` | |

An arm-cell takes about 3-8 s on an RTX 4090. Figure style lives in `style.py`.

## Comparing logs

A source is a campaign name (looked up in `--root`, then every checkout), a campaign directory, or
a `results.jsonl`. Output: `benchviz_runs/cmp-<baseline>-vs-<candidate>/` (or `--name`); it re-reads
both logs on load. Above 1x the candidate is faster. Cells match on (op, precision, shape, batch),
then by time per matrix at each log's largest batch unless `--exact`. With BatchLAS on both sides
the vendor arm is a machine control, flagged beyond 5 %. `compare` prints each side's toolchain
(`provenance.base` / `provenance.new`) and warns on differences besides the one under test.

### Implementation A/B (DPC++ vs AdaptiveCpp)

```sh
python3 benchmarks/benchviz ab --build-dir build-dpcpp --build-dir build-acpp-rel \
    --ops gemm,potrf --types float,double --preset saturation --gpu 3 --campaign impl-ab
```

Two `--build-dir` (baseline first) plus `run`'s grid flags; cells run on both builds back to back,
alternating order. Rows land in `<campaign>-dpcpp` / `<campaign>-acpp` (`-base` / `-new` for one
implementation) and `<campaign>` is their `--exact` comparison. Use Release trees, at saturation.

## Tuning tab

`serve` also hosts `/tuning` over the ledgers `benchmarks/results/tuning/ledger/<op>.<device>.jsonl`
and tables `tuned/<op>.<device>.txt` (see docs/design/tuning-data.md, docs/design/tiered-tuning.md).

```sh
python3 benchmarks/benchviz serve --tuned ../main-checkout/tuned --build-dir ../other/build
python3 benchmarks/benchviz tune --tier coarse --ops potrf,getrs --devices 0,1 --plan     # estimate only
python3 benchmarks/benchviz tune --tier coarse --ops potrf --devices '' --device-key sm_120 --plan
python3 benchmarks/benchviz tune --tier custom --ops potrf --dtype float --devices 1 --remeasure-keys keys.txt
```

`tune` flags: `--tier {preview,calibrate,coarse,deep,custom}` (required), `--ops`, `--dtype`,
`--devices`, `--device-key`, `--remeasure-keys`, `--allow-dirty`, `--max-dim` (2048), `--budget`
[h], `--guard-wait` (3600 s), `--ledger`, `--name`, `--build-dir`, `--plan`.
`serve` flags: `--host`, `--port` (8765), `--read-only`, `--token`, `--ledger`, `--tuned`,
`--build-dir`.

The read-only views (Status, Op decision map, Runs) need no GPU or build; they read ledgers through
`scripts/sweep_to_table.py`. A run needs a build whose `batchlas_tune --help` lists every flag
passed. One tuning run at a time; a dirty tree runs only with `--allow-dirty` and never reaches
`tuned/`; "Run again" resumes from the ledger. Runs live in `benchviz_runs/_tuning/<run>/`; a run
started from the terminal is watched the same way and outlives the server.

## HTTP endpoints

GET: `/`, `/tuning`, `/api/meta`, `/api/logs`, `/api/state?c=<campaign>`,
`/api/op?c=<campaign>&op=<op>`, `/api/events?c=<campaign>` (SSE), `/fig/<campaign>/<path under
figures/>`, `/api/tune/{meta,status,op,cell,run,events}`. POST: `/api/run`, `/api/plan`,
`/api/stop`, `/api/compare`, `/api/replot`, `/api/tune/{plan,run,stop}`. `--read-only` allows only
the plan and compare POSTs; `--token` requires `?t=<token>`.

## ROCm

Run a ROCm build with `--backend rocm`. `gpu_guard.sh` is NVIDIA-only: the device is pinned with
`ROCR_VISIBLE_DEVICES` and exclusivity is not checked. gesvd has no ROCm arm.
