# benchviz: BatchLAS vs vendor LAPACK

This tool measures and plots BatchLAS against the vendor LAPACK library the build links:
cuSOLVER/cuBLAS on CUDA, rocSOLVER/rocBLAS on ROCm. It covers every LAPACK op that has a
vendor route. The output is publication-ready vector figures, and a local dashboard shows them
updating while the campaign runs. It needs Python 3 with matplotlib, numpy and pandas, plus a
TeX installation with `latex` and `dvipng` for the Computer Modern text. It installs nothing else.

```sh
# 1. Build the harnesses (the library plus four targets).
cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ -DBATCHLAS_BUILD_BENCHMARKS=ON
cmake --build build --target factor_bench syev_benchmark ormqr_benchmark gesvd_vendor_benchmark -j"$(nproc)"

# 2a. Start the dashboard and launch runs from the browser.
python3 benchmarks/benchviz serve            # http://127.0.0.1:8765
#     From a laptop: ssh -L 8765:localhost:8765 <gpu-box>

# 2b. Or run from the terminal. The dashboard can watch this too.
python3 benchmarks/benchviz run --ops potrf,getrf,syev --types float,double --preset quick
python3 benchmarks/benchviz run --ops all --types all --preset full --campaign paper-4090

# 3. Re-render. Use --no-titles for a paper, where the caption carries the title.
python3 benchmarks/benchviz plot paper-4090 --no-titles
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

syevx, sytrd, stedc and steqr have no vendor route in dispatch, so a comparison would put
BatchLAS against itself.

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

| Figure | Content | Size |
|---|---|---|
| `speedup_n` | vendor time / BatchLAS time against n, one line per precision (S/D/C/Z), log2 axes | 3.5 in (single column) |
| `throughput_n` | GFLOP/s against n (LAWN 41 counts, complex = 4× real), or matrices/s for syev and gesvd. BatchLAS solid blue, vendor dashed gray, one panel per precision | 7.16 in (double column) |
| `heatmap` | speedup over the whole n × batch grid. Diverging blue/red around a neutral 1×, clipped at 8×; hatched cells were not measured | 7.16 in |
| `summary_<t>` | op × n speedup table at saturation | single column |

All text is 7–8 pt LaTeX Computer Modern, so a figure set at `\columnwidth` is typeset 1:1. The
colours pass the dataviz palette validator, including colour-vision-deficiency separation. Every
precision line also has its own marker and a direct label, so colour never carries identity
alone. PDFs embed TrueType fonts (`pdf.fonttype 42`), as most publishers require.

## Grids

| Preset | Orders | Batch ladder | Arm-cells (all ops, all precisions) |
|---|---|---|---|
| `smoke` | every 3rd order | ×16 steps from 256 | ~360 |
| `quick` | every order | ×4 steps from 64 | ~2,300 |
| `full` | every order | ×2 steps from 32 | ~5,000 |

On a 4090 an arm-cell takes about 5–8 s, mostly process start and warm-up. `--mem-gib` (default
3) caps the batch size for each order. `--orders 16,32,64` overrides the order ladder.

## ROCm

`factor_bench` selects its backend at compile time, so a ROCm build compares against
rocSOLVER/rocBLAS. Run with `--backend rocm`. `gpu_guard.sh` is NVIDIA-only, so on ROCm the
device is pinned with `ROCR_VISIBLE_DEVICES` and exclusivity is not checked. gesvd has no ROCm
arm, because `gesvd_vendor_benchmark` calls cuSOLVER directly.
