# Small-n factorization baseline

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · measured 2026-09-10

Native versus vendor timings for potrf, getrf, geqrf, orgqr and getrs at n = 4..512, all four
types. The routing windows in `tuned/` for these ops and the gates in
`docs/design/small-n-factorization-plan.md` rest on these numbers.

## Run setup

- **Box:** 2x RTX 4090 (sm_89, 128 SM, 24 GB, 72 MB L2), driver 595.84. GPU 1 throughout, one
  harness on the card at a time, each process run through `benchmarks/gpu_guard.sh 1`.
- **Build:** `build/presets/dev-tests`, `RelWithDebInfo`, DPC++ at `/opt/dpcpp-cuda`,
  `-fsycl-targets=nvidia_gpu_sm_89`, CUDA 13.2 (V13.2.86), cuSOLVER 12.2.0.11.
- **Harness:** `benchmarks/factor_bench.cc`, driven by `benchmarks/run_factor_grid.sh`. Scope: 20
  (op, type) grids, 2,784 timed rows, 1,392 paired cells. Every grid completed and no route threw.

Ratios are `vendor_ms / native_ms`. **> 1 means native wins.** Each row carries a
`resolved_route` read back per arm from the coverage instrument in a separate process.
`pin_parsed = 1` means the pin parsed, not that it ran. The CSVs use route-era spellings
(`vendor:auto`, `native:<tier>`); current coverage records `vendor`, `cta` and `blocked`.

## Discards

61 rows in 60 cells are flagged `bad=1` (relative sd >= 10% over 7 interleaved reps). They stay
in the CSVs and are excluded from every ratio here: 52 orgqr (see [orgqr](#orgqr)), 5 getrs,
2 getrf and 1 geqrf, all at n <= 8 where a call takes 20-60 us.

Three more cells are excluded by hand: `orgqr` cdouble 1024x128 b8192 (48 GB), `geqrf` cdouble
1024x128 b8192 (32 GB) and `orgqr` cdouble 512x64 b16384 (24 GB). They oversubscribe the card and
measure UVM paging, and their `rel_sd` (0.001-0.008) passes the variance gate.

## How to read the grid

Three batches per order: half, nominal and double the `SAT_LADDER` value of `docs/perf/lu.md`.
The tables quote the **top** batch of each ladder. A `~` marks a cell whose time moved more than
5% between the nominal and top batch. Cells read `ratio (native route)`.

> **Warning:** a `~` cell is not saturated. Its number is the best reading available, not a
> converged one. `~` is common here; see [Saturation](#saturation).

## potrf

| n (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 4 (b32768) | 1.94 (cta) | 1.14~ (cta) | 0.86 (cta) | 0.97 (cta) |
| 8 (b32768) | 2.10 (cta) | 1.21~ (cta) | 1.07 (cta) | 1.36 (cta) |
| 16 (b32768) | 1.64 (cta) | 0.66~ (cta) | 0.63 (cta) | 0.72 (cta) |
| 17 (b32768) | 2.70 (cta) | 1.47~ (cta) | 0.94 (cta) | 1.53 (cta) |
| 24 (b32768) | 2.07~ (cta) | 1.38~ (cta) | 0.96 (cta) | 1.43 (cta) |
| 32 (b16384) | 1.30~ (cta) | 0.81 (cta) | 0.81 (cta) | 1.01 (cta) |
| 33 (b16384) | 1.84~ (cta) | 1.30~ (cta) | 1.08 (cta) | 1.55 (cta) |
| 48 (b16384) | 1.08~ (cta) | 0.79~ (cta) | 0.76~ (cta) | 0.77 (cta) |
| 64 (b16384) | 0.95~ (cta) | 0.52~ (cta) | 0.54 (cta) | 0.59 (cta) |
| 65 (b8192) | 1.21~ (cta) | 0.72~ (cta) | 1.06 (cta) | 1.29 (cta) |
| 96 (b8192) | 0.63~ (cta) | 0.45~ (cta) | 0.52 (cta) | 0.51 (blocked) |
| 128 (b8192) | 0.43~ (cta) | 0.53~ (blocked) | 0.46 (blocked) | 0.58 (blocked) |
| 256 (b4096) | 0.85~ (blocked) | 0.92~ (blocked) | 0.83 (blocked) | 0.82 (blocked) |
| 512 (b1024) | 1.19~ (blocked) | 1.08~ (blocked) | 0.99 (blocked) | 0.90 (blocked) |

- **float** wins at n <= 33 (33 is the last unambiguous win; 0.95 at 64 brackets the edge). Minimum
  **0.43 at n = 128**; it recovers on the blocked driver: **1.19 at 512**, still rising (see
  [Saturation](#saturation)).
- **cfloat** wins at 4, 8, 17, 24, 33 and 512. **double** and **cdouble** win nowhere above
  n = 33 except 65. All four types are below 1.00 for 96 <= n <= 384.
- **The 2^k rungs read high.** Every type reads higher at 17 than at 16, 33 than at 32 and 65 than
  at 64 (up to 2.1x). Read the `+1` neighbours; a boundary taken from powers of two is wrong.

## getrf

| n (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 4 (b32768) | 0.21~ (cta) | *dropped* | 0.07~ (cta) | 0.06~ (cta) |
| 32 (b16384) | 0.55 (cta) | 0.55~ (cta) | 0.29~ (cta) | 0.41 (cta) |
| 33 (b16384) | 1.13 (cta) | 1.06 (cta) | 0.20 (blocked) | 0.50 (cta) |
| 64 (b16384) | 1.63 (cta) | 0.82 (cta) | 0.31 (blocked) | 0.43 (cta) |
| 65 (b8192) | 0.83 (cta) | 0.62 (cta) | 0.18~ (blocked) | 0.32 (cta) |
| 128 (b8192) | 0.72 (cta) | 0.46~ (blocked) | 0.28~ (blocked) | 0.40 (blocked) |
| 192 (b4096) | 0.91~ (blocked) | 0.79~ (blocked) | 0.52~ (blocked) | 0.50 (blocked) |
| 256 (b4096) | 1.19~ (blocked) | 1.06 (blocked) | 0.68~ (blocked) | 0.57 (blocked) |
| 512 (b1024) | 2.17~ (blocked) | 1.60~ (blocked) | 0.75~ (blocked) | 0.74 (blocked) |

Full grid: `benchmarks/results/factor_baseline_getrf_<type>.csv`.

- **float** wins at 33..64 (1.13 to 1.63) and n >= 256 (1.19 at 256, 2.17 at 512). Edges: 0.55 at
  32, 0.83 at 65, 0.91 at 192. Nothing above 512 was measured.
- **cfloat** wins at 33..48 (1.06 / 1.09; 0.82 at 64 is inside the dip) and at n >= 256.
- **double** and **cdouble** never win (0.06 to 0.92).
- **Vendor cliff at 32/33.** At float b16384 cuBLAS costs 0.3159 ms at n = 32 and 1.3634 ms at
  n = 33: **4.3x for a 6% larger matrix**. Native goes 0.5730 -> 1.2069 ms. At n <= 32 native is
  0.21-0.67x of the vendor for the 32-bit types and 0.06-0.27x for the 64-bit types.
- **Small end.** At n = 4, batch 32768, cuBLAS costs 14.6 us and native 67.9 us, against a DRAM
  roof of 1.1 us. Target for the small-n plan: **0.21x at float n = 4 and 0.55x at float n = 32.**

## geqrf

| shape (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 8 (b32768) | 0.08~ (cta) | *dropped* | 0.02~ (cta) | 0.04~ (cta) |
| 32 (b16384) | 0.73~ (cta) | 0.62~ (cta) | 0.21 (cta) | 0.33 (cta) |
| 33 (b16384) | 0.76~ (cta) | 0.69~ (cta) | 0.22 (cta) | 0.34 (cta) |
| 48 (b16384) | 1.02~ (cta) | 1.74~ (cta) | 0.32~ (cta) | 0.45 (cta) |
| 64 (b16384) | 1.71~ (cta) | 2.72 (cta) | 0.58 (blocked) | 0.52 (cta) |
| 65 (b8192) | 2.46~ (cta) | 2.31 (cta) | 0.66~ (blocked) | 0.57 (cta) |
| 96 (b8192) | 2.69~ (cta) | 2.28 (cta) | 1.16~ (blocked) | 0.49 (blocked) |
| 128 (b8192) | 3.64~ (blocked) | 3.79 (blocked) | 1.76~ (blocked) | 0.68 (blocked) |
| 192 (b4096) | 5.38~ (blocked) | 5.71~ (blocked) | 2.47 (blocked) | 1.06 (blocked) |
| 512 (b1024) | 13.38~ (blocked) | 10.52~ (blocked) | 4.96~ (blocked) | 1.89~ (blocked) |
| 128x32 (b16384) | 2.23~ (cta) | 3.79~ (cta) | 0.68~ (cta) | 0.68 (cta) |
| 512x32 (b16384) | 2.68 (cta) | 3.39~ (blocked) | 1.58 (blocked) | 2.16 (blocked) |
| 512x64 (b16384) | 3.18 (blocked) | 4.10~ (blocked) | 2.37 (blocked) | 1.73 (blocked) |

`1024x128` (b8192) is excluded: cdouble reads 1.691 there and is paging (2.956 at b4096 and
2.130 at b2048 are kept). Full grid: `benchmarks/results/factor_baseline_geqrf_<type>.csv`.

- **Spread:** 0.02x to 13.4x, the widest of any op. Below the crossover native is not close
  (0.02x at double n = 8): the CTA kernel reduces a length-8 column over a work-group sized for the
  trailing update.
- **Square crossovers:** float n >= 48 (1.02; 0.76 at 33), cfloat n >= 48 (1.74; 0.69 at 33),
  double n >= 96 (1.16; 0.66 at 65), cdouble n >= 192 (1.06; 0.58 at 129 in the CSV).
- **Tall panels cross earlier.** 512x32 and 512x64 win for every type, cdouble included (2.16 /
  1.73). Exception: 128x32 is float 2.23 and cfloat 3.79, against double 0.68 and cdouble 0.68.
- **Flip gates supported:** float and cfloat `n >= 48`, double `n >= 96`, cdouble `n >= 192`. The
  float n = 48 cell is 1.02 and moving (0.94 at b4096), so `n >= 64` is the defensible float edge
  if only one number may be quoted.

## orgqr

The vendor arm is a per-item `cusolverDnXorgqr` loop on an out-of-order sub-queue (the
`batch > 1` branch of `orgqr_vendor` in `src/backends/cublas.cc`), not a batched routine.

> **Note:** every orgqr ratio means "beats the per-item loop". None means "beats cuSOLVER".

| shape (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 16 (b32768) | 209.57 | 172.40 | 97.72 | 42.25 |
| 33 (b16384) | 60.53 | 33.93 | 59.45~ | 20.27 |
| 64 (b16384) | 27.10~ | 15.09~ | 27.09 | 11.31 |
| 128 (b8192) | 9.47 | 5.29 | 12.27 | 6.57~ |
| 256 (b4096) | 5.30 | 4.04 | 8.46 | 4.75 |
| 512 (b1024) | 3.64 | 2.54 | 4.61 | 2.77 |
| 512x64 (b16384) | 2.71 | 1.95 | 4.93 | *excluded* (2.71 at b8192) |
| 1024x128 (b8192) | 1.73 | 1.63 | 2.91 | *excluded* (2.72 at b4096) |

Full grid, including the `*dropped*` small-n cells: `benchmarks/results/factor_baseline_orgqr_<type>.csv`.

- Every native route is `blocked`. Native wins every kept cell, all four types, falling with n:
  from 2.54x (cfloat n = 512) to 209.6x (float n = 16, b32768). This supports the orgqr flip
  (native at every `n <= 512`, every type) with no exception.
- **No kept cell loses.** The minimum over 210 kept cells is **1.58** (cfloat `1024x128` b2048).
  The excluded paging cells read below 1.00 (cdouble `1024x128` b8192 at 0.892) and are the same
  artefact as the excluded `512x64` cell.
- **Shipped table:** `tuned/orgqr.<dtype>.sm_89.txt` ranks `blocked` first for `m <= 512` and
  `vendor` first for `m >= 513`. The `1024x128` cells, where native wins 1.63-2.91x, are
  vendor-first: an unflipped win, not a measured loss.
- **Discards.** All 52 orgqr discards are on the native arm (`rel_sd` 0.10-0.12, vendor 0.02-0.07).
  Each native rep follows a vendor rep that dispatched `batch` cuSOLVER calls on an out-of-order
  sub-queue, which the harness's `q->wait()` does not bound. Re-measure single-arm before quoting
  an orgqr number in a gate.

## getrs

Two right-hand-side widths: `nrhs <= 2` (route-era clause A, every type) and `nrhs = 4` (clause B,
float only). Both carried live traffic when this grid was measured.

Today getrs chooses from `tuned/getrs.<dtype>.sm_89.txt` (transcribed at `424a45bc`, untimed).
`cta` ranks first for `n >= 32` with `nrhs <= 2` (every type) and `nrhs` 3-4 (float). `vendor`
ranks first everywhere else: `n=31 nrhs=1` is vendor-first, `n=32 nrhs=1` is cta-first, and
`n=32 nrhs=3` is cta-first for float and vendor-first for the other three types.

| n (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| **nrhs = 1** | | | | |
| 4 (b32768) | **0.71**~ | **0.59**~ | **0.52**~ | **0.23**~ |
| 8 (b32768) | 1.12 | **0.69**~ | **0.98**~ | **0.41**~ |
| 16 (b32768) | 1.04~ | 3.76 | **0.99** | 2.00 |
| 17 (b32768) | **0.76**~ | 1.38 | **0.92** | 2.04 |
| 32 (b16384) | 2.29 | 1.40~ | 3.86 | 2.17 |
| 64 (b16384) | 1.65~ | 1.25 | 3.95~ | 2.66 |
| 512 (b1024) | 1.73~ | 1.35~ | 1.73~ | 1.60~ |
| **nrhs = 4** | | | | |
| 4 (b32768) | **0.45**~ | 0.35~ | 0.12~ | 0.07~ |
| 32 (b16384) | 1.30 | 1.03~ | 0.85 | 0.58 |
| 512 (b1024) | 1.53~ | 1.27~ | 2.75~ | 2.14 |

Bold marks a cell inside a shipped clause that loses. Every bold cell is at `n <= 24`, below the
floor. Full grid: `benchmarks/results/factor_baseline_getrs_<type>.csv`.

- **Above the floor.** At `nrhs = 1` and 32 <= n <= 512, `cta` wins every type with no losses
  (1.25x to 3.99x). The `nrhs = 4` cells for double and cdouble sit below 1.00 from n = 32 to 65;
  clause B is float-only, so those cells are measured, not routed.
- **Below the floor.** Clauses A and B cover 266 measured cells. **32 lose**, all at n <= 24, worst
  **0.234** (cdouble n = 4, nrhs = 1, batch 32768). `docs/perf/lu.md` records clauses A and B as
  zero-loss; this grid contradicts that below n = 32. The losses grow with batch: float n = 4,
  nrhs = 1 reads 1.504 at b8192 and 0.521 at b65536.

**The floor is 32.** This table reads each order at the **worst** rung of its batch ladder, the
statistic that shows whether a clause has a loss anywhere inside it (`nrhs = 1`):

| T | n=4 | n=8 | n=16 | n=17 | n=24 | n=32 | n=48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| float | 0.71 | 1.12 | 1.04 | 0.76 | 0.95 | 2.29 | 1.94 |
| cfloat | 0.59 | 0.69 | 3.76 | 1.38 | 1.27 | 1.40 | 1.60 |
| double | 0.52 | 0.98 | 0.99 | 0.87 | 3.66 | 3.77 | 3.74 |
| cdouble | 0.23 | 0.41 | 2.00 | 2.02 | 1.93 | 2.13 | 2.46 |

- 32 is the first order where all four types clear the flip gate and stay clear above it. Below
  it the band is non-monotone, so no lower floor is defensible. Clause B has the same floor
  (float `nrhs = 4` clears from 32).

**Mechanism.** The fused kernel gives one work-group to a matrix whose solve is a few dozen flops,
so the work-group is the cost; `cublas?getrsBatched` has no such floor. Full write-up:
[getrs order floor evidence](lu.md#getrs-order-floor-evidence) in `docs/perf/lu.md`.

## Saturation

The `SAT_LADDER` inherited from `docs/perf/lu.md` does not saturate these ops. Of 442 cells with a
clean nominal and top rung, **201 (45%) moved more than 5% on the last doubling**. The cause is
not statistical. At potrf float n >= 96 the native arm is linear in batch and the vendor arm is
superlinear (n = 128: native x1.99 per doubling, cuSOLVER x2.24-2.35; n = 512: x2.08-2.10 against
x2.32-2.36), so the ratio has no fixed point inside the ladder.

Extensions past the ladder, one process per cell:

| cell | ladder | extension | verdict |
|---|---|---|---|
| potrf float 128 | 0.322 / 0.363 / 0.429 | 0.457 (b16384), **0.466** (b32768) | saturated at b32768, +2.0% on the last doubling |
| potrf float 256 | 0.620 / 0.687 / 0.851 | 0.916 (b8192) | still +7.6%, **not saturated** |
| potrf float 512 | 0.948 / 1.045 / 1.187 | 1.330 (b2048), **1.503** (b4096) | still +13%, **not saturated**, rising |
| geqrf float 128 | 3.175 / 3.322 / 3.643 | **3.618** (b16384) | saturated at b16384, -0.7% |
| geqrf float 512 | 26.279 / 17.364 / 13.376 | **10.98** (b2048) | still -18%, **not saturated**, falling |

Quote potrf float 512 as **">= 1.19 and rising"** and geqrf float 512 as **"<= 13.4 and falling"**,
not as point estimates. Neither converges inside 24 GB.

Two small-order effects also affect `~` cells: the potrf CTA arm steps superlinearly when the
batch's working set crosses the 72 MB L2 (float n = 32 at b16384 reads x2.51 against x1.77 one
rung below), and below n ~ 16 both arms are launch-bound.

## Where this baseline disagrees with the record

Of 66 cells with a per-cell number in `docs/perf/{potrf,lu,qr}.md`, **43 reproduce within 15% and
23 do not**. Every disagreement is at the prior's own batch. The vendor arm reproduces every vendor
time the record states to three significant figures, so no disagreement is the vendor moving.

| op | type | n | batch | prior | now | delta | explanation |
|---|---|---:|---:|---:|---:|---:|---|
| geqrf | float | 128 | 4096 | 2.01 | 3.32 | +65% | **tier.** The prior is the CTA time; the shipped default is Blocked |
| geqrf | float | 256 | 2048 | 5.62 | 7.39 | +32% | unexplained (blocked driver) |
| orgqr | float | 32 | 8192 | 123.2 | 52.9 | -57% | vendor per-item loop ~2.3x cheaper per item |
| potrf | float | 8 | 8192* | 2.75 | 2.08 | -24% | see [The potrf n = 8 finding](#the-potrf-n--8-finding) |

`*` The prior's batch (4096) is not on this ladder; the nearest rung (8192) is used.

The other geqrf rows (+22% to +77%, n = 128..512) are unexplained; the orgqr cdouble rows at 256
and 512 move the other way (+37%, +62%).

**geqrf n = 128 is a tier mismatch, and it is exact.** Pinned: `cta` 15.419 ms, `blocked` 9.349 ms,
vendor 31.024 ms (float). `docs/perf/qr.md` records "30.9 -> 15.4 ms (2.01x)"; its native number is
the CTA time, while its own tier table gives the default at that cell as blocked, 9.97 ms. The
`form=sq n=128` row of `tuned/geqrf.float.sm_89.txt` ranks `tiny` (which cannot run at this
order), then `blocked`, then `cta`, then `vendor`.

**The other geqrf cells (+22% to +77%) are not explained.** Ruled out: the tier, the vendor arm
(it reproduces exactly), and the trailing GEMM route (forcing `BATCHLAS_GEMM_ROUTE=vendor` moves
the native arm by 0.1% at n = 128 / 256 / 512). Left: the blocked driver itself, which runs 1.31x
faster at n = 256 and 1.43x faster at n = 512 than `order.csv` records. Not isolated further.

**orgqr's vendor arm halved at n <= 256 and did not move at n = 512.** That is the signature of a
per-item launch cost: about 54 us per item at n = 32, against 486 us at n = 512, where each item's
own kernel hides it. Driver 595.84 is a candidate, but this run cannot test it.

## The potrf n = 8 finding

`docs/perf/potrf.md` records **2.75x** at float n = 8, batch 4096, from the CTA-vs-cuSOLVER table
in `experiments/wp4_potrf/README.md` §4. This grid reads **2.08-2.18x**. The number does not
reproduce on any instrument, card or pin.

| instrument | card | vendor ms | native ms | ratio |
|---|---|---:|---:|---:|
| archived `realpotrf.cpp`, `BATCHLAS_POTRF_ROUTE` pinned | GPU 1 | 0.0284 | 0.0130 | **2.18** |
| `factor_bench`, interleaved | GPU 1 | 0.0255 | 0.0121 | 2.10 |
| `factor_bench`, the grid, b8192 / 16384 / 32768 | GPU 1 | 0.0395 / 0.0588 / 0.1024 | 0.0190 / 0.0284 / 0.0488 | 2.08 / 2.07 / 2.10 |

Ruled out:

- **The harness.** The archived instrument, rebuilt from `git show perf-evidence/vendor-independence:experiments/wp4_potrf/phase2_ab/realpotrf.cpp`, reproduces the §4 controls, and `factor_bench` agrees with it to 0.3%.
- **The route, kernel, card and vendor library.** Coverage shows `cta` and `vendor` on every row. `src/extensions/potrf_cta_device.hh` matches the tag. GPU 0 gives 2.18. `libcusolver.so.12.2.0.11` predates both campaigns.
- **Saturation.** The ratio is flat at 2.07-2.10 over batch 8192-32768. No batch reads 2.75.

The cell is dispatch-bound: at n = 8, batch 4096, more than 80% of both arms is per-call launch
latency (DRAM roof 2.2 us, native 13 us, vendor 28 us).

**No raw data backs the prior.** `experiments/wp4_potrf/` at the evidence tag has no
CTA-vs-cuSOLVER grid. The n = 8 and n = 16 rows of that table should be replaced by the numbers on
this page; the rest reproduces.

## Warm-up order and the variance gate

The harness warms up on a time budget (`WARM_S` per arm) and discards the warm-up reps. A cold
first run (SYCL JIT plus cold clocks) fabricated a **3.7x** result in this repository.

The warm-up runs in the same arm order as the timed loop. With a per-arm warm-up, the first timed
rep of arm 0 came in **2.2x slow every time** (0.0567 ms against a steady 0.0261 at potrf float
n = 8) and tripped the `rel_sd` gate on a stable cell.

## The arms list

`--arms=` splits on commas exactly and rejects an empty list. (It once matched by substring, so
`--arms=vendor` silently dropped the native arm.) Each arm name is its own route pin, recorded in
the CSV `pin_parsed` column, and the residual and pivot gate runs per arm.

An arm whose pin cannot run the shape throws `std::invalid_argument` before anything is timed. For
example, `BATCHLAS_GETRF_ROUTE=tiny` at cdouble n = 32 throws (`src/ops/getrf/getrf.cc:48`).

## Raw data

| what | path |
|---|---|
| the 20 grids, every row including the discarded ones | `benchmarks/results/factor_baseline_<op>_<type>.csv` |
| the harness, ladder driver and GPU guard | `benchmarks/factor_bench.cc`, `benchmarks/run_factor_grid.sh`, `benchmarks/gpu_guard.sh` |
| the archived potrf n = 8 instrument | `git show perf-evidence/vendor-independence:experiments/wp4_potrf/phase2_ab/realpotrf.cpp` |
| the prior tables this page compares against | `docs/perf/potrf.md` §"CTA kernel measured against cuSOLVER", `docs/perf/qr.md` §"geqrf order and batch grid" and §"orgqr grid", `docs/perf/lu.md` §"Negative results" 4 |

The saturation extensions, tier pins and GEMM-route probe are not in the CSVs; their times are
quoted inline.
