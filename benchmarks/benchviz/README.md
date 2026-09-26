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
    gemm_benchmark gemv_benchmark trsm_benchmark trmm_benchmark syrk_benchmark syr2k_benchmark spmm_benchmark -j"$(nproc)"

# 2a. Start the dashboard and launch runs from the browser.
python3 benchmarks/benchviz serve            # http://127.0.0.1:8765
#     From a laptop: ssh -L 8765:localhost:8765 <gpu-box>

# 2b. Or run from the terminal. The dashboard can watch this too.
python3 benchmarks/benchviz run --ops potrf,getrf,syev --types float,double --preset quick
python3 benchmarks/benchviz run --ops all --types all --preset full --campaign paper-4090

# 3. Re-render every figure (e.g. after a style change).
python3 benchmarks/benchviz plot paper-4090
```

Campaigns are stored in `benchviz_runs/<name>/` (git-ignored):

| File | Contents |
|---|---|
| `results.jsonl` | Append-only, one row per (cell, arm) |
| `campaign.json` | The request and its provenance: GPU, driver, git SHA, dirty flag |
| `figures/<op>/{speedup_n,throughput_n,heatmap}.{pdf,png}` | The per-op figures |
| `figures/_summary/summary_<precision>.{pdf,png}` | The cross-op summaries |

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

BLAS cells are square (m = n = k). spmm uses a random CSR pattern with 16 nonzeros per row, 16 right-hand
sides, and n rows.

Some ops are left out, each for a stated reason:
- **syevx, sytrd, stedc, steqr:** no vendor route.
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

## Figures

| Figure | Content |
|---|---|
| `speedup_n` | Vendor time / BatchLAS time against n, at the saturated batch. One series per precision (S/D/C/Z) with a 2σ band, and a red dashed parity line at 1×. Log2 y-axis only when the range exceeds 8×. |
| `throughput_n` | GFLOP/s against n (LAWN 41 counts, complex = 4× real), or matrices/s for syev and gesvd. BatchLAS (○) vs vendor (★), one panel per precision, with 2σ bands. |
| `heatmap` | Speedup over the whole n × batch grid, in viridis with log2 colour steps. A red boundary follows the 1× crossing. Empty cells were not measured. |
| `summary_<t>` | Op × n speedup table at saturation, annotated with the value in each cell. |

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
