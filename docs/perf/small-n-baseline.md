# Small-n factorization baseline {#perf_small_n_baseline}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, cuSOLVER 12.2.0.11, driver 595.84 · measured 2026-09-10

Native versus vendor timings for potrf, getrf, geqrf, orgqr and getrs at \f$4 \le n \le 512\f$, all four
types. The sm_89 routing windows in `tuned/` for these ops and the gates in
@ref design_small_n_plan rest on these numbers.

## Run setup

- GPU 1 of the 2x RTX 4090 box, one harness at a time via `benchmarks/gpu_guard.sh 1`;
  `RelWithDebInfo`, DPC++ `/opt/dpcpp-cuda`, `-fsycl-targets=nvidia_gpu_sm_89`.
- Harness `benchmarks/factor_bench.cc`, driven by `benchmarks/run_factor_grid.sh`: 20 (op, type)
  grids, 1,392 paired cells. Raw rows: `benchmarks/results/factor_baseline_<op>_<type>.csv`.
- Ratios are `vendor_ms / native_ms`; **> 1 means native wins**. Each row carries the route
  read back from the coverage instrument. CSVs use route-era spellings (`vendor:auto`,
  `native:<tier>`); today's are `vendor`, `cta`, `blocked`.
- Three batches per order (half, nominal, double the `SAT_LADDER` of `docs/perf/lu.md`); tables
  quote the **top** batch. `~` marks a cell that moved > 5% on the last doubling: not saturated
  (see [Saturation](#saturation)). 61 rows with `rel_sd >= 10%` (all \f$n \le 8\f$, 52 of them orgqr) and
  three cdouble cells that oversubscribe 24 GB and measure UVM paging are excluded.

## potrf

- **float** wins at \f$n \le 33\f$, edge bracketed by 0.95 at \f$n = 64\f$; minimum 0.43 at \f$n = 128\f$; back above
  1 on the blocked driver at 512 (1.19, still rising). All four types are below 1.00 for
  \f$96 \le n \le 384\f$. double and cdouble win nowhere above \f$n = 33\f$ except 65.
- **The \f$2^k\f$ rungs read high.** Every type reads higher at 17 than 16, 33 than 32 and 65 than 64
  (up to 2.1x). Take boundaries from the `+1` neighbours, never from powers of two.

## getrf

- **float** wins at \f$33 \le n \le 64\f$ (1.13 to 1.63) and \f$n \ge 256\f$ (1.19 at 256, 2.17 at 512); edges 0.55 at
  32, 0.83 at 65, 0.91 at 192. **cfloat** wins at \f$33 \le n \le 48\f$ and \f$n \ge 256\f$. **double** and **cdouble**
  never win (0.06 to 0.92).
- **Vendor cliff at 32/33:** cuBLAS float b16384 goes 0.316 -> 1.363 ms (4.3x for a 6% larger
  matrix). At \f$n \le 32\f$ native is 0.21-0.67x of the vendor (32-bit) and 0.06-0.27x (64-bit); at \f$n = 4\f$
  native takes 67.9 us against cuBLAS 14.6 us and a 1.1 us DRAM roof.

## geqrf

| shape (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 8 (b32768) | 0.08~ | *dropped* | 0.02~ | 0.04~ |
| 33 (b16384) | 0.76~ | 0.69~ | 0.22 | 0.34 |
| 48 (b16384) | 1.02~ | 1.74~ | 0.32~ | 0.45 |
| 64 (b16384) | 1.71~ | 2.72 | 0.58 | 0.52 |
| 65 (b8192) | 2.46~ | 2.31 | 0.66~ | 0.57 |
| 96 (b8192) | 2.69~ | 2.28 | 1.16~ | 0.49 |
| 128 (b8192) | 3.64~ | 3.79 | 1.76~ | 0.68 |
| 192 (b4096) | 5.38~ | 5.71~ | 2.47 | 1.06 |
| 512 (b1024) | 13.38~ | 10.52~ | 4.96~ | 1.89~ |
| \f$128 \times 32\f$ (b16384) | 2.23~ | 3.79~ | 0.68~ | 0.68 |
| \f$512 \times 32\f$ (b16384) | 2.68 | 3.39~ | 1.58 | 2.16 |
| \f$512 \times 64\f$ (b16384) | 3.18 | 4.10~ | 2.37 | 1.73 |

- Square crossovers: float and cfloat \f$n \ge 48\f$, double \f$n \ge 96\f$, cdouble \f$n \ge 192\f$. The float \f$n = 48\f$
  cell is 1.02 and moving (0.94 at b4096), so \f$n \ge 64\f$ is the defensible float edge. Shipped
  per-type floors: [QR: route arms](qr.md#qr-route-arms).
- Below the crossover native is far behind (0.02x at double \f$n = 8\f$): the CTA kernel reduces a short
  column over a work-group sized for the trailing update.
- Tall panels cross earlier: \f$512 \times 32\f$ and \f$512 \times 64\f$ win for every type, but \f$128 \times 32\f$ loses for double and
  cdouble (0.68), so a `rows >= 128` tall clause is not a bracketed edge.

## orgqr

The vendor arm is a per-item `cusolverDnXorgqr` loop on an out-of-order sub-queue (the
`batch > 1` branch of `orgqr_vendor`, `src/backends/cublas.cc`). Every ratio means "beats the
per-item loop", not "beats cuSOLVER".

| shape (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 16 (b32768) | 209.57 | 172.40 | 97.72 | 42.25 |
| 64 (b16384) | 27.10~ | 15.09~ | 27.09 | 11.31 |
| 256 (b4096) | 5.30 | 4.04 | 8.46 | 4.75 |
| 512 (b1024) | 3.64 | 2.54 | 4.61 | 2.77 |
| \f$512 \times 64\f$ (b16384) | 2.71 | 1.95 | 4.93 | *excluded* (2.71 at b8192) |
| \f$1024 \times 128\f$ (b8192) | 1.73 | 1.63 | 2.91 | *excluded* (2.72 at b4096) |

- Native (`blocked`) wins every one of 210 kept cells; minimum 1.58 (cfloat \f$1024 \times 128\f$ b2048). This
  backs `tuned/orgqr.sm_89.txt` ranking `blocked` first for `m <= 512`, every type. The `m >= 513`
  rows stay vendor-first although \f$1024 \times 128\f$ wins 1.63-2.91x: an unflipped win, not a measured loss.
- All 52 orgqr discards are on the native arm: each follows a vendor rep whose `batch` cuSOLVER
  calls on the sub-queue are not bounded by the harness's `q->wait()`. Re-measure single-arm before
  quoting an orgqr number in a gate.

## getrs

`tuned/getrs.sm_89.txt` (transcribed at `424a45bc`) ranks `cta` first for `n >= 32` with
`nrhs <= 2` (every type) and `nrhs` 3-4 (float only); `vendor` first everywhere else.

**The floor is 32.** Worst rung of each batch ladder, `nrhs = 1`:

| T | \f$n=4\f$ | \f$n=8\f$ | \f$n=16\f$ | \f$n=17\f$ | \f$n=24\f$ | \f$n=32\f$ | \f$n=48\f$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| float | 0.71 | 1.12 | 1.04 | 0.76 | 0.95 | 2.29 | 1.94 |
| cfloat | 0.59 | 0.69 | 3.76 | 1.38 | 1.27 | 1.40 | 1.60 |
| double | 0.52 | 0.98 | 0.99 | 0.87 | 3.66 | 3.77 | 3.74 |
| cdouble | 0.23 | 0.41 | 2.00 | 2.02 | 1.93 | 2.13 | 2.46 |

- 32 is the first order where all four types clear 1.0 and stay clear; below it the band is
  non-monotone. At \f$32 \le n \le 512\f$, `nrhs = 1`, `cta` wins every type (1.25x to 3.99x). At
  `nrhs = 4` double and cdouble sit below 1.00 for \f$32 \le n \le 65\f$, hence float-only.
- The fused kernel spends one work-group on a solve of a few dozen flops; `cublas?getrsBatched`
  has no such floor. Write-up: [getrs order floor evidence](lu.md#getrs-order-floor-evidence).

## Saturation

The `SAT_LADDER` does not saturate these ops: 201 of 442 clean cells (45%) moved > 5% on the last
doubling. At potrf float \f$n \ge 96\f$ native is linear in batch and cuSOLVER superlinear (x2.24-2.36
per doubling), so the ratio has no fixed point inside 24 GB. Quote potrf float 512 as
**"\f$\ge 1.19\f$ and rising"** (1.503 at b4096) and geqrf float 512 as **"\f$\le 13.4\f$ and falling"** (10.98
at b2048). potrf float 128 (0.466 at b32768) and geqrf float 128 (3.618 at b16384) do saturate.

## Open: disagreements with the record

- **potrf \f$n = 8\f$.** `docs/perf/potrf.md` (CTA kernel measured against cuSOLVER) records 2.75x at
  float \f$n = 8\f$, batch 4096; every instrument here (the archived `realpotrf.cpp`, `factor_bench`,
  both GPUs) reads 2.07-2.18x, and no raw data backs the prior. Its \f$n = 8\f$ and \f$n = 16\f$ rows should be
  replaced; the cell is > 80% launch latency.
- **geqrf float \f$128 \le n \le 512\f$** reads 22-77% above `docs/perf/qr.md`. At \f$n = 128\f$ this is exact tier
  mismatch (the record quotes `cta`, 15.4 ms; the default is `blocked`, 9.35 ms). The rest is the
  blocked driver itself (vendor reproduces, GEMM route ruled out); not isolated.
- **orgqr's vendor arm** is ~2.3x cheaper per item at \f$n \le 256\f$ than recorded and unchanged at 512:
  a per-item launch cost, possibly driver 595.84; untested.

## Warm-up order and the variance gate

The harness warms up on a time budget (`WARM_S` per arm), discards the warm-up reps, and runs the
warm-up in the same arm order as the timed loop. A cold first run fabricated a 3.7x result; with a
per-arm warm-up the first timed rep of arm 0 was 2.2x slow every time (0.0567 against 0.0261 ms,
potrf float \f$n = 8\f$) and tripped the `rel_sd` gate on a stable cell.

## The arms list

`--arms=` splits on commas exactly and rejects an empty list (substring matching once let
`--arms=vendor` drop the native arm). Each arm name is its own route pin, recorded in the CSV
`pin_parsed` column (parsed, not proven reached), and the residual and pivot gate runs per arm. An
arm whose pin cannot run the shape (`std::invalid_argument` from `choose()`) is refused and flagged
`pin refused` in its CSV row; the other arms of the cell still run.
