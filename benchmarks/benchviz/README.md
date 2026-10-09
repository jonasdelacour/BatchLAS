# benchviz: BatchLAS vs vendor LAPACK and BLAS {#benchviz}

`benchviz` measures BatchLAS against the vendor library the build links (cuSOLVER/cuBLAS on
CUDA, rocSOLVER/rocBLAS on ROCm) for every LAPACK op with a vendor route. It writes vector
figures and a local dashboard that updates while a campaign runs.

Requirements: Python 3 with matplotlib, numpy and pandas; a TeX installation with `latex` and
`dvipng` for Computer Modern text. `benchviz` compiles nothing; build the harnesses first.

```sh
# 1. Build the harnesses (the library plus the benchmark targets).
cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ -DBATCHLAS_BUILD_BENCHMARKS=ON
cmake --build build --target factor_bench syev_benchmark ormqr_benchmark gesvd_vendor_benchmark \
    gemm_benchmark gemv_benchmark trsm_benchmark trmm_benchmark syrk_benchmark syr2k_benchmark spmm_benchmark \
    stedc_benchmark steqr_benchmark sytrd_blocked_benchmark sytrd_cta_benchmark sytrd_sy2sb_benchmark \
    sytrd_sb2st_benchmark -j"$(nproc)"

# 2a. Dashboard (launch runs from the browser). From a laptop: ssh -L 8765:localhost:8765 <gpu-box>
python3 benchmarks/benchviz serve            # http://127.0.0.1:8765

# 2b. Terminal runs (the dashboard can watch these).
python3 benchmarks/benchviz run --ops potrf,getrf,syev --types float,double --preset quick
python3 benchmarks/benchviz run --ops all --types all --preset full --campaign paper-4090

# 3. Re-render every figure (e.g. after a style change).
python3 benchmarks/benchviz plot paper-4090

# 4. Compare two logs, e.g. one run per build.
python3 benchmarks/benchviz run --campaign before --build-dir ../old-checkout/build --ops syev,stedc --types float
python3 benchmarks/benchviz run --campaign after  --build-dir build                 --ops syev,stedc --types float
python3 benchmarks/benchviz compare before after
```

## Which build is measured

`benchviz` runs the harness binaries it finds, searched in this order:

1. `build/` of the checkout it runs from;
2. `build/presets/{benchmarks,dev-tests}`;
3. when the checkout is a worktree, the main checkout's `build/`.

`run --build-dir <dir>` overrides the search. The measured code is the code **last compiled**
into that directory, not the checkout's HEAD.

```sh
python3 benchmarks/benchviz info     # the build a run would use now, and every campaign's state
```

`info`, the `serve` banner and the dashboard header show the build's estimated source commit
(`~sha`), its build time, and a warning when it is behind (source changed after the build, or
main has commits it lacks). Each campaign records its builds in `campaign.json` under
`provenance.builds`; every result row records its `binary` path. Older campaigns show
"not recorded". `info --build-dir <dir>` (repeatable) describes other trees.

### Toolchain and SYCL implementation

Each `provenance.builds[]` entry records how its binaries were compiled:

| Field | Source |
|---|---|
| `sycl_impl` | `BATCHLAS_SYCL_IMPL_RESOLVED` in `CMakeCache.txt` (`DPCPP` or `ACPP`); trees configured before it existed are inferred from `-fsycl-targets` / `--acpp-targets` (`sycl_impl_inferred`) |
| `sycl_impl_version` | acpp: the `AdaptiveCpp version:` line of `acpp --acpp-version`; DPC++: release and intel/llvm commit from `clang++ --version` |
| `compiler`, `compiler_id`, `compiler_version` | `CMAKE_CXX_COMPILER`, CMake's compiler id (`Clang` for both), the `--version` head |
| `build_type` | `CMAKE_BUILD_TYPE` |
| `fp_model`, `fp_contract`, `fp_flags` | the last `-ffp-model=` / `-ffp-contract=` and every result-changing fp flag in `batchlas_sycl_obj`'s `flags.make` |
| `fp_contract_effective` | the flag, else the driver default at -O2+: `on` for DPC++, `fast` for acpp |
| `sycl_targets` | `-fsycl-targets=` or `--acpp-targets=` |
| `acpp_env` (acpp only) | the `ACPP_*` variables its runs got, which came from the environment, and whether the JIT database is per campaign |
| `acpp_coarse_grained_events` (acpp only) | whether `src/` or `include/` uses the queue property; it is not an environment knob |

Warnings (in `info`, run logs and `warnings`):

- an icpx build without `-ffp-model=precise`;
- an acpp build whose effective fp contract is not `on`: `cmake/BatchLASSyclAcpp.cmake` passes
  `-ffp-contract=on` to match DPC++ (design page R11);
- an NVIDIA box with targets that cannot reach it (acpp `generic` can).

**acpp runtime knobs.** Every process run from an acpp tree gets `ACPP_ADAPTIVITY_LEVEL=1`,
`ACPP_RT_SCHEDULER=direct` and `ACPP_APPDB_DIR=<campaign>/acpp-appdb` (per-campaign JIT and
adaptivity database), unless the environment already sets them; any `ACPP_*` variable in the
environment wins and is recorded. Each acpp row records `sycl_impl`, its `acpp_env`, and
`acpp_jit`: true when the process printed acpp's "new binaries being JIT-compiled" warning. The
harness warm-up keeps a first-launch JIT out of the timed iterations. For throwaway passes that
fill the JIT cache before the measured campaign (design page section 7), export one
`ACPP_APPDB_DIR` for all of them, since resuming a campaign skips the cells it has.

`export` writes a static, read-only copy with downscaled figures:

```sh
python3 benchmarks/benchviz export <campaign>... --out <dir> [--fragment]
```

`--fragment` omits the page skeleton for hosts that supply their own `<html>`.

## Campaign files

Campaigns live in `benchviz_runs/<name>/` (git-ignored).

| File | Contents |
|---|---|
| `results.jsonl` | Append-only, one row per (cell, arm) |
| `campaign.json` | Request and provenance: GPU, driver, builds measured, benchviz checkout SHA and dirty flag |
| `figures/<op>/{speedup_n,throughput_n,heatmap}.{pdf,png}` | Per-op figures |
| `figures/_summary/summary_<precision>.{pdf,png}` | Cross-op summaries |

A comparison has `kind: compare` in `campaign.json` and no `results.jsonl`.

- `--gpu 0,1` splits cells across GPUs, one worker per GPU. Both arms of a cell run on the same
  card. Two workers share the host CPU, so publish from one card.
- Stop kills the cell in flight within about 0.5 s. Re-running with the same `--campaign` resumes
  and measures only missing cells. Changing `--ops` or `--types` extends the campaign.

## What is compared

| Op | Harness | Vendor arm | BatchLAS arm |
|---|---|---|---|
| potrf, getrf, getrs, geqrf, orgqr | `factor_bench` (verified residuals) | `--arms=vendor` | `--arms=native` |
| gesv, posv | `factor_bench` | sub-ops pinned vendor | sub-ops pinned native |
| ormqr | `ormqr_benchmark` | `BATCHLAS_ORMQR_ROUTE=vendor` | `=native` |
| syev (with vectors) | `syev_benchmark` | `BATCHLAS_SYEV_ROUTE=vendor` | `=native`, tuned nb and fuse |
| gesvd (n ≤ 32) | `gesvd_vendor_benchmark` | direct `gesvdjBatched` call | `gesvd_cta` |
| gemm, gemv, trsm, spmm | `<op>_benchmark` | `BATCHLAS_<OP>_ROUTE=vendor` (cuBLAS / cuSPARSE) | `=native` |
| trmm, syrk, syr2k (float) | `<op>_benchmark` | `=vendor` | `=triangular` |
| stedc, steqr, sytrd, sytrd_cta, sy2sb, sb2st | own harnesses | none | the harness's direct call |

- The n-figures use square cells (m = n = k). spmm uses a random CSR pattern with 16 nonzeros
  per row and 16 right-hand sides.
- **The BatchLAS arm is pinned `native`, never `auto`.** Where `auto` prefers the vendor, an
  `auto` arm times the vendor twice and reports 1.0×.

Left out:

- **syevx:** no vendor route. stedc, steqr and the sytrd family have none either (see
  "Sub-operations").
- **symm:** the native arm cannot be pinned; it is expand-then-gemm through the routed gemm.
- **hemm, herk, her2k:** no route variable and no coverage rows.
- **syrk and syr2k in double, and trmm in double:** no native kernel (nothing is recorded for
  syrk double).
- **Plain `native` for syrk and syr2k:** selects a wrong-answer route or throws, so those arms
  pin `triangular`.

### Timing and correctness

- One process per (cell, arm). See `benchmarks/run_factor_grid.sh` for the reasons.
- Every run goes through `gpu_guard.sh`, which requires an exclusive GPU and discards
  contaminated cells.
- Ratios are quoted only at saturation: the n-figures use the largest batch both arms were
  measured at; the heatmap shows the full grid.
- `factor_bench` rows are verified on the host. A row flagged `bad` for noise is re-measured;
  any other `bad` row is dropped.
- ormqr, syev and gesvd are timed but not verified.

## Routes

Each process sets `BATCHLAS_COVERAGE_OUT`. Each row records the route that ran (`route`) and every
sub-op route (`subroutes`). A pin is not trusted:

- A BatchLAS arm that resolved to a vendor route is dropped from the plots; the dashboard's
  failure table gives the reason.
- A vendor arm that resolved to a native route is dropped the same way.
- For gesv and posv, both arms take the outer `blocked` route, so their sub-op routes decide
  which arm they belong to.
- An **open marker** in the speedup figure means the BatchLAS arm is native at the top but calls
  a vendor sub-op (for example, blocked syev uses cuBLAS gemm).
- For getrs, orgqr and ormqr the sub-op routes cannot be attributed: the harness runs an untimed
  setup factorization in the same process, and its routes land in the same coverage file.

## Figures

| Figure | Content |
|---|---|
| `speedup_n` | Vendor time / BatchLAS time against n at the saturated batch. One series per precision (S/D/C/Z) with a 2σ band; red dashed parity line at 1×. Log2 y-axis only when the range exceeds 8×. |
| `throughput_n` | GFLOP/s against n (LAWN 41 counts, complex = 4× real), or matrices/s for syev and gesvd. BatchLAS (○) vs vendor (★), one panel per precision, 2σ bands. |
| `heatmap` | Speedup over the n × batch grid, viridis with log2 steps. A red boundary follows the 1× crossing. Empty cells were not measured. |
| `summary_<t>` | Op × n speedup table at saturation, value annotated in each cell. |
| `speedup_2d` | Rectangular ops only: speedup over both shape axes, annotated, same colour scale and 1× boundary as `heatmap`. Hatched cells are undefined shapes. |
| `throughput_2d` | Rectangular ops only: GFLOP/s over the same axes, BatchLAS and vendor side by side on a shared log scale. |

An op with no vendor arm gets `throughput_n` with BatchLAS alone, and `heatmap` shows its
throughput over n × batch. It has no speedup figures and is left out of `summary_<t>`, except in
a log comparison.

## Sub-operations

syev's stages are ops in the dashboard group "Eigensolver stages". No vendor library ships them
batched, so each has one BatchLAS arm.

| Op | Harness | What it runs | Arguments |
|---|---|---|---|
| stedc | `stedc_benchmark` | divide and conquer, with vectors | threshold 0, merge and driver `Auto` (the tuning tables syev uses) |
| steqr | `steqr_benchmark` | QR iteration, with vectors; n ≤ 32 is `steqr_cta`, above it `steqr_wg` | 50 sweeps, interleaved working vectors; Wilkinson shift pinned |
| sytrd | `sytrd_blocked_benchmark` | blocked reduction to tridiagonal (syev_blocked stage 1) | nb as syev_blocked chooses it, including the complex override at 256 < n ≤ 512 |
| sytrd_cta | `sytrd_cta_benchmark` | one-CTA reduction, n ≤ 32 | defaults |
| sy2sb | `sytrd_sy2sb_benchmark` | two-stage stage 1, dense to band | kd = 32, as `choose_two_stage_kd` |
| sb2st | `sytrd_sb2st_benchmark` | two-stage stage 2, bulge chase | kd = 32 |

- Every argument is passed explicitly, so an older build runs the same cell. A harness that maps
  "0" to the library default would silently measure something else.
- `sytrd_nb` and `two_stage_kd` in `ops.py` copy the library's choices. A retune that changes
  them is not measured until those copies are updated.
- steqr stops at n = 128: `steqr_wg` takes 210 ms at n = 128, batch 256 (stedc: 0.6 ms);
  n = 256 takes minutes per cell.
- None of these ops is dispatched, so the route is the function the harness calls. Coverage still
  records the dispatched sub-ops underneath (for example, stedc's merge gemm).
- Flop counts: sytrd, sytrd_cta and sy2sb use 4n³/3. stedc, steqr and sb2st have no canonical
  count and are plotted in matrices/s.

## Comparing logs

```sh
python3 benchmarks/benchviz logs                     # every campaign in every checkout's benchviz_runs/
python3 benchmarks/benchviz compare baseline main-3df4e99-square
python3 benchmarks/benchviz compare ../other/benchviz_runs/x/results.jsonl y --exact
python3 benchmarks/benchviz compare before after --base-arm vendor --new-arm vendor   # did cuSOLVER move?
```

A source is a campaign name (looked up in `--root`, then every checkout; an ambiguous name lists
the candidate paths), a campaign directory, or a `results.jsonl`. The output goes to
`benchviz_runs/<name>/` (default `cmp-<baseline>-vs-<candidate>`). `compare` stores no rows and
re-reads both logs on each load, so it fills in while a campaign runs.

- **Speedup.** The candidate's chosen arm goes in the BatchLAS slot and the baseline's in the
  vendor slot; above 1× the candidate is faster. The default arm is BatchLAS on both sides.
  `--base-arm` / `--new-arm` select either arm of either log.
- **Cell matching.** First the same (op, precision, shape, batch). Then, for a shape measured at
  no common batch, each log's largest batch compared by time per matrix (assumes both are
  saturated). `--exact` drops these rescaled pairs. A shape measured in one log only has no ratio.
- **Labels.** `Build <sha>` when the log recorded its binaries; older logs show `HEAD <sha>`, the
  checkout benchviz ran from, which may differ from what was measured.
- **Vendor control.** With BatchLAS on both sides, the vendor arm also runs in both logs. Its
  ratio measures the machine (clocks, contention, driver). The overview flags it beyond 5 %; a
  control of 1.08× means every speedup is inflated by about that much.
- **Warnings.** Different devices, different batch grids, and two logs of the same binaries. The
  last gives ratios that are pure run-to-run noise (a useful A/A test).

For figures you will quote, measure both campaigns back to back on the same card with the same
grid.

- **Toolchains.** A comparison carries each side's `toolchain` (implementation and version,
  compiler, build type, fp model and contract, targets, commit, acpp knobs) in
  `provenance.base` / `provenance.new`, and `compare` prints them. It warns on any difference
  besides the one under test: a different build type, fp model or effective fp contract always;
  a different compiler version or targets unless the implementations differ.

### Implementation A/B (DPC++ vs AdaptiveCpp)

```sh
python3 benchmarks/benchviz ab --build-dir build-dpcpp --build-dir build-acpp-rel \
    --ops gemm,potrf --types float,double --preset saturation --gpu 3 --campaign impl-ab
```

`ab` takes exactly two `--build-dir` (baseline first) and the grid flags of `run`. Every cell runs
on both builds back to back on one card; the build that goes first alternates from cell to cell.
Rows land in `<campaign>-dpcpp` and `<campaign>-acpp` (`-base` / `-new` when both builds are one
implementation), and `<campaign>` is their comparison, paired with `--exact`. When the two logs
differ in implementation, figures and reports name the arms `DPC++` and `AdaptiveCpp`, adding
the commit only when the commits differ (which is also a warning). The vendor control then
includes each implementation's vendor interop path, not only the machine state. Compare
implementations at saturation only, from Release trees.

## Rectangular ops

Ops with a second shape dimension also get a 2-D sweep (their `Plane` in `ops.py`). Each shape
runs once at its saturated batch: the largest power of two that fits the memory budget for that
shape.

| Op | y axis | x axis | Constraint |
|---|---|---|---|
| geqrf, orgqr | rows m | columns n | n ≤ m |
| ormqr | order of Q, m | reflectors n | n ≤ m |
| getrs, gesv, posv | order n | right-hand sides (1, 4, 16, 64, 256) | |
| gemm | output order m = n | inner dimension k | |
| gemv | rows m | columns n | |
| trsm | triangle order n | right-hand sides q | |
| syrk, syr2k | order n | rank k | |
| spmm | rows n | right-hand sides (4 … 128) | |

The square point of each plane is the square sweep's cell. Left square: potrf and getrf
(`factor_bench` takes m = n only), syev, trmm, and gesvd.

- Presets enable the sweep; `--no-rect` (or the dashboard's "Rectangular shapes" box) disables it.
  It adds about 90 % to a `quick` campaign of the vendor-compared ops (1,667 cells become 3,157;
  the six sub-operations add 328, one arm each).
- The trsm map needs a `trsm_benchmark` that reads its second argument as the right-hand-side
  count. An older binary times n × n and ignores it; `benchviz info` reports such a build as
  behind.

### Style

Figures follow `style.py`, the source of truth for the house style:

- LaTeX Computer Modern, drawn at 20 × 10 in with 30 pt text, scaled by `\includegraphics`.
- Full box around each panel, ticks pointing in, light full grid.
- Dotted connectors with markers in the order ○ △ □ ★ ◇; frameless legend with enlarged markers.
- Units in brackets on every axis; bold column titles on multi-panel figures; no in-figure title.
- PDFs embed TrueType fonts (`pdf.fonttype 42`).

## Grids

A grid is a preset plus overrides. The dashboard's run panel edits the same fields and previews
the n × batch cells; the CLI takes them as flags.

| Field | Flag | Meaning |
|---|---|---|
| orders | `--orders` | n values. `16,32,64` lists them, `4:512` doubles from 4 to 512, `8:128:8` steps by 8. Blank uses each op's ladder. Out-of-range values are dropped. |
| batch mode | `--batch-mode` | `ladder` (default), `list`, or `saturated` (one batch per n: the largest that fits the memory budget) |
| ladder | `--batch-min --batch-max --batch-step` | batch-min × step^k, up to min(batch-max, memory cap) |
| list | `--batches` | explicit batch list, same syntax as orders |
| reps | `--reps` | timed repetitions per arm-cell |
| memory | `--mem-gib` | per-arm device-memory budget; caps batch per (op, n) |
| rectangular | `--rect` / `--no-rect` | sweep the rectangular shape planes, one saturated batch per shape |

| Preset | Grid |
|---|---|
| `smoke` | every 3rd order, batch ×16 from 256 |
| `saturation` | every order, saturated batch only (speedup vs n, throughput) |
| `quick` | every order, batch ×4 from 64 |
| `full` | every order, batch ×2 from 32 |

On an RTX 4090 an arm-cell takes about 3–8 s, mostly process start, warm-up and the
exclusive-GPU check.

## ROCm

`factor_bench` selects its backend at compile time. Run a ROCm build with `--backend rocm`; it
compares against rocSOLVER/rocBLAS. `gpu_guard.sh` is NVIDIA-only, so on ROCm the device is
pinned with `ROCR_VISIBLE_DEVICES` and exclusivity is not checked. gesvd has no ROCm arm, because
`gesvd_vendor_benchmark` calls cuSOLVER directly.
