# Native factorizations at n = 4..512: the committed baseline grid

The P0 baseline of `docs/design/small-n-factorization-plan.md`. Every later plan (P1-P7) is
gated on a number from this page, so it is the one grid that has to be reproducible rather
than merely quotable — and the last section of this page is the list of places where the
number already in `docs/perf` no longer is.

**Box.** 2x RTX 4090 (sm_89, 128 SM, 24 GB, 72 MB L2), driver 595.84. GPU 1 throughout
(GPU 1 does not drive the display), one harness on the card at a time, every process through
`benchmarks/gpu_guard.sh 1`. No run exited 5. **Date.** 2026-09-10, 01:21-11:20 CEST.
**Build.** `build/presets/dev-tests`, `CMAKE_BUILD_TYPE=RelWithDebInfo`, DPC++ at
`/opt/dpcpp-cuda`, `-fsycl-targets=nvidia_gpu_sm_89`, CUDA 13.2 (V13.2.86), cuSOLVER
12.2.0.11 and cuBLAS from that toolkit; the harness TU itself compiles `-O2 -DNDEBUG`
against those shipped `.so`s. **Harness.** `benchmarks/factor_bench.cc` driven by
`benchmarks/run_factor_grid.sh`; raw rows in `benchmarks/results/factor_baseline_<op>_<type>.csv`.

Ratios are `vendor_ms / native_ms`; **> 1 means native wins**. Every row carries a
`resolved_route` read back per arm from the dispatch-coverage instrument in a **separate
process**, never inferred from the pin: `pin_parsed = 1` says the pin was understood, not
that it took.

**Size of the run.** 20 (op, type) grids, **2,784 timed rows / 1,392 paired cells**, all five
ops on all four types. Every grid completed; nothing crashed, no route threw, and every op has
a vendor arm for every type.

**Discards.** **61 rows in 60 cells** were flagged `bad=1` by the harness, all for one reason,
`relsd` (relative sd >= 10% over 7 interleaved reps). They stay in the CSVs and are excluded
from every ratio here. **52 of the 60 are `orgqr`** — see that section; the rest are 5 `getrs`,
2 `getrf`, 1 `geqrf`, all at n <= 8 where the call is 20-60 us. **Three further cells are
excluded by hand**, because the harness has no gate that can see them:

| cell | arrays | native ms, previous rung -> this rung |
|---|---|---|
| `orgqr` cdouble 1024x128 b8192 | 48 GB | 1,822.6 -> 15,553.2 (**x8.5** for a x2 batch) |
| `geqrf` cdouble 1024x128 b8192 | 32 GB | 951.6 -> 3,581.0 (x3.8) |
| `orgqr` cdouble 512x64 b16384 | 24 GB | 858.7 -> 3,203.6 (x3.7) |

These oversubscribe 24 GB of device memory and measure UVM paging, not the kernel. `rel_sd` is
0.001-0.008 on all three — thrashing is *consistent*, so a variance gate cannot catch it. The
matching cfloat and double cells at 24 GB are clean (x1.97-1.99) and are kept.

## How to read the grid

Three batches per order — half, nominal, double the `SAT_LADDER` value of `docs/perf/lu.md`.
The tables quote the **top** batch of each ladder, the most saturated reading taken, and mark a
cell `~` when the ratio moved more than 5% between the nominal and the top batch. **A `~` cell
is not saturated**: the number is the best reading available, not a converged one. `~` is the
common case here, and the reason is mechanical — see [Saturation](#saturation).

Cells read `ratio (native route)`; the vendor arm resolved `vendor:auto` on every row of all
20 grids.

## potrf

| n (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 4 (b32768) | 1.94 (cta) | 1.14~ (cta) | 0.86 (cta) | 0.97 (cta) |
| 8 (b32768) | 2.10 (cta) | 1.21~ (cta) | 1.07 (cta) | 1.36 (cta) |
| 9 (b32768) | 1.39 (cta) | 0.68~ (cta) | 0.46 (cta) | 0.51 (cta) |
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
| 129 (b4096) | 0.56~ (cta) | 0.58~ (blocked) | 0.75 (blocked) | 0.75 (blocked) |
| 192 (b4096) | 0.68~ (blocked) | 0.71~ (blocked) | 0.69 (blocked) | 0.73 (blocked) |
| 256 (b4096) | 0.85~ (blocked) | 0.92~ (blocked) | 0.83 (blocked) | 0.82 (blocked) |
| 384 (b1024) | 0.94~ (blocked) | 0.91~ (blocked) | 0.94 (blocked) | 0.88 (blocked) |
| 512 (b1024) | 1.19~ (blocked) | 1.08~ (blocked) | 0.99 (blocked) | 0.90 (blocked) |

**float** wins n <= 48 (1.08 at 48 — but that cell is `~` and reads 0.94 one rung lower, so 33
is the last unambiguous win; bracketed by 0.95 at 64), loses through the CTA band with a
minimum of **0.43 at n = 128**, and recovers on the blocked driver to 0.85 at 256, 0.94 at 384
and **1.19 at 512** — the first potrf cell above 1.00 that the record does not contain, and it
is still rising (1.33 at b2048, 1.50 at b4096). **cfloat** wins only at 4, 8, 17, 24, 33 and
512. **double** and **cdouble** win nowhere above n = 33 except 65 (1.06 / 1.29) and lose least
at 384 (0.94 / 0.88). All four types are below 1.00 across 96 <= n <= 384.

**The order ladder is not smooth and the 2^k rungs are the worst ones.** Every type reads
higher at 17 than at 16 and higher at 33 than at 32 — float 1.64 -> 2.70 and 1.30 -> 1.84,
cdouble 0.72 -> 1.53 and 1.01 -> 1.55 — and higher at 65 than at 64 (float 0.95 -> 1.21, double
0.54 -> 1.06). n = 9 is the exception in the other direction (float 2.10 -> 1.39). The `+1`
neighbours the ladder exists to probe are worth up to 2.1x, and a boundary read off powers of
two alone would be wrong at every type.

## getrf

| n (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 4 (b32768) | 0.21~ (cta) | *dropped* | 0.07~ (cta) | 0.06~ (cta) |
| 8 (b32768) | 0.26~ (cta) | 0.22~ (cta) | 0.11 (cta) | 0.12~ (cta) |
| 9 (b32768) | 0.46~ (cta) | 0.50 (cta) | 0.23 (cta) | 0.56 (cta) |
| 16 (b32768) | 0.47~ (cta) | 0.67~ (cta) | 0.27~ (cta) | 0.61 (cta) |
| 17 (b32768) | 0.49 (cta) | 0.38 (cta) | 0.24 (cta) | 0.33 (cta) |
| 24 (b32768) | 0.52 (cta) | 0.46~ (cta) | 0.26 (cta) | 0.36 (cta) |
| 32 (b16384) | 0.55 (cta) | 0.55~ (cta) | 0.29~ (cta) | 0.41 (cta) |
| 33 (b16384) | 1.13 (cta) | 1.06 (cta) | 0.20 (blocked) | 0.50 (cta) |
| 48 (b16384) | 1.27 (cta) | 1.09 (cta) | 0.25~ (blocked) | 0.41 (cta) |
| 64 (b16384) | 1.63 (cta) | 0.82 (cta) | 0.31 (blocked) | 0.43 (cta) |
| 65 (b8192) | 0.83 (cta) | 0.62 (cta) | 0.18~ (blocked) | 0.32 (cta) |
| 96 (b8192) | 0.92 (cta) | 0.50 (cta) | 0.25~ (blocked) | 0.36 (blocked) |
| 128 (b8192) | 0.72 (cta) | 0.46~ (blocked) | 0.28~ (blocked) | 0.40 (blocked) |
| 129 (b4096) | 0.87~ (cta) | 0.54 (blocked) | 0.30~ (blocked) | 0.40 (blocked) |
| 192 (b4096) | 0.91~ (blocked) | 0.79~ (blocked) | 0.52~ (blocked) | 0.50 (blocked) |
| 256 (b4096) | 1.19~ (blocked) | 1.06 (blocked) | 0.68~ (blocked) | 0.57 (blocked) |
| 384 (b1024) | 1.91~ (blocked) | 1.41~ (blocked) | 0.92 (blocked) | 0.69~ (blocked) |
| 512 (b1024) | 2.17~ (blocked) | 1.60~ (blocked) | 0.75~ (blocked) | 0.74 (blocked) |

**float** wins in two disjoint windows, 33 <= n <= 64 (1.13 / 1.27 / 1.63) and n >= 256
(1.19 / 1.91 / 2.17), bracketed on all four edges by 0.55 at 32, 0.83 at 65, 0.91 at 192 and
nothing above 512. **cfloat** repeats the shape one third lower: 33..48 (1.06 / 1.09, with
0.82 at 64 already inside the dip) and n >= 256. **double never wins at any order**, from 0.07
at n = 4 to 0.92 at 384; **cdouble** likewise, 0.06 to 0.74.

**The n = 32/33 step is a vendor cliff, not a native one.** At float b16384 cuBLAS costs
0.3159 ms at n = 32 and 1.3634 ms at n = 33 — **4.3x for a 6% larger matrix** — while native
goes 0.5730 -> 1.2069. cuBLAS `getrfBatched` has a small-square special path that ends at 32,
and it is the strongest vendor code in this whole page: at n <= 32 native is 0.21-0.67x of it
for the 32-bit types and 0.06-0.27x for the 64-bit ones. The other end of the range is the
mirror image: at n = 4, batch 32768, cuBLAS costs 14.6 us against native's 67.9 us for a cell
whose DRAM roof is 1.1 us. **This is the P1 target, stated as a number: 0.21x at float n = 4
and 0.55x at float n = 32.**

## geqrf

| shape (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 4 (b32768) | 0.17~ (cta) | 0.14~ (cta) | 0.03~ (cta) | 0.03~ (cta) |
| 8 (b32768) | 0.08~ (cta) | *dropped* | 0.02~ (cta) | 0.04~ (cta) |
| 9 (b32768) | 0.12 (cta) | 0.11~ (cta) | 0.03~ (cta) | 0.04 (cta) |
| 16 (b32768) | 0.22~ (cta) | 0.22 (cta) | 0.07 (cta) | 0.11 (cta) |
| 17 (b32768) | 0.23~ (cta) | 0.26~ (cta) | 0.07 (cta) | 0.11 (cta) |
| 24 (b32768) | 0.36 (cta) | 0.43 (cta) | 0.10 (cta) | 0.16 (cta) |
| 32 (b16384) | 0.73~ (cta) | 0.62~ (cta) | 0.21 (cta) | 0.33 (cta) |
| 33 (b16384) | 0.76~ (cta) | 0.69~ (cta) | 0.22 (cta) | 0.34 (cta) |
| 48 (b16384) | 1.02~ (cta) | 1.74~ (cta) | 0.32~ (cta) | 0.45 (cta) |
| 64 (b16384) | 1.71~ (cta) | 2.72 (cta) | 0.58 (blocked) | 0.52 (cta) |
| 65 (b8192) | 2.46~ (cta) | 2.31 (cta) | 0.66~ (blocked) | 0.57 (cta) |
| 96 (b8192) | 2.69~ (cta) | 2.28 (cta) | 1.16~ (blocked) | 0.49 (blocked) |
| 128 (b8192) | 3.64~ (blocked) | 3.79 (blocked) | 1.76~ (blocked) | 0.68 (blocked) |
| 129 (b4096) | 3.20 (blocked) | 3.32~ (blocked) | 1.61 (blocked) | 0.58 (blocked) |
| 192 (b4096) | 5.38~ (blocked) | 5.71~ (blocked) | 2.47 (blocked) | 1.06 (blocked) |
| 256 (b4096) | 7.57 (blocked) | 7.51~ (blocked) | 3.23 (blocked) | 1.50~ (blocked) |
| 384 (b1024) | 11.67~ (blocked) | 9.42~ (blocked) | 4.46~ (blocked) | 1.67~ (blocked) |
| 512 (b1024) | 13.38~ (blocked) | 10.52~ (blocked) | 4.96~ (blocked) | 1.89~ (blocked) |
| 128x32 (b16384) | 2.23~ (cta) | 3.79~ (cta) | 0.68~ (cta) | 0.68 (cta) |
| 512x32 (b16384) | 2.68 (cta) | 3.39~ (blocked) | 1.58 (blocked) | 2.16 (blocked) |
| 512x64 (b16384) | 3.18 (blocked) | 4.10~ (blocked) | 2.37 (blocked) | 1.73 (blocked) |
| 1024x128 (b8192) | 7.16~ (blocked) | 5.09 (blocked) | 6.42~ (blocked) | *excluded* (2.96 at b4096) |

The widest spread of any op: **0.02x to 13.4x**. Native wins from n >= 48 for float (1.02,
bracketed by 0.76 at 33) and cfloat (1.74, bracketed by 0.69), from **n >= 96 for double**
(1.16, bracketed by 0.66 at 65) and from **n >= 192 for cdouble** (1.06, bracketed by 0.58 at
129), and the margin grows monotonically with n from there. Below the crossover it is not close
— 0.08x at float n = 8, 0.02x at double n = 8 — because the CTA kernel reduces a length-8
column over a work-group sized for the trailing update.

**Tall panels are the shape the callers issue, and they cross over earlier.** At 512x32 and
512x64 every type wins, cdouble included (2.16 / 1.73), where square cdouble at those n loses
0.33 / 0.52. 128x32 is the exception: float 2.23 and cfloat 3.79 against double 0.68 and
cdouble 0.68 — the only tall cells either 64-bit type loses. `1024x128` for cdouble is the
oversubscribed cell excluded above (its b8192 reading, 1.691, is paging); the two rungs below it
read 2.130 and 2.956 and are kept.

**The P0 flip this supports**, restated against these numbers rather than the plan's: float and
cfloat `n >= 48`, double `n >= 96`, cdouble `n >= 192` — each with the bracketing non-winner
named above. That is one rung earlier than the plan proposed for float/cfloat/double and two
for cdouble. The float n = 48 cell is 1.02 and moving (0.94 at b4096), so `n >= 64` is the
defensible float edge if only one number may be quoted.

## orgqr

The vendor arm is a per-item `cusolverDnXorgqr` loop on an out-of-order sub-queue
(`cublas.cc:1414-1419`), not a batched routine. Every ratio here means "beats the per-item
loop"; none of them means "beats cuSOLVER".

| shape (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 4 (b32768) | *dropped* | *dropped* | *dropped* | 197.48~ |
| 8 (b32768) | *dropped* | *dropped* | *dropped* | 126.90~ |
| 9 (b32768) | *dropped* | *dropped* | *dropped* | 95.18 |
| 16 (b32768) | 209.57 | 172.40 | 97.72 | 42.25 |
| 17 (b32768) | 180.94~ | 131.24 | 90.94~ | 33.99 |
| 24 (b32768) | 112.98~ | 71.90~ | 75.59 | 20.86 |
| 32 (b16384) | *dropped* | 36.98 | 49.80~ | 13.39 |
| 33 (b16384) | 60.53 | 33.93 | 59.45~ | 20.27 |
| 48 (b16384) | 43.53~ | 26.18~ | 37.41 | 17.10 |
| 64 (b16384) | 27.10~ | 15.09~ | 27.09 | 11.31 |
| 65 (b8192) | 24.93 | 16.38 | 33.34 | 12.73 |
| 96 (b8192) | 18.87~ | 10.67 | 22.09~ | 9.69 |
| 128 (b8192) | 9.47 | 5.29 | 12.27 | 6.57~ |
| 129 (b4096) | 12.80~ | 8.69~ | 19.76~ | 6.49 |
| 192 (b4096) | 10.17 | 5.78 | 11.41~ | 6.41 |
| 256 (b4096) | 5.30 | 4.04 | 8.46 | 4.75 |
| 384 (b1024) | 4.29 | 3.08 | 5.67 | 3.46 |
| 512 (b1024) | 3.64 | 2.54 | 4.61 | 2.77 |
| 128x32 (b16384) | 25.91~ | 14.74 | 23.35~ | 11.23 |
| 512x32 (b16384) | 7.95 | 4.91 | 8.62 | 3.31 |
| 512x64 (b16384) | 2.71 | 1.95 | 4.93 | *excluded* (2.71 at b8192) |
| 1024x128 (b8192) | 1.73 | 1.63 | 2.91 | *excluded* (2.72 at b4096) |

Every native route is `native:blocked`. **Native wins every square cell that was kept**, all
four types, from 2.54x (cfloat n = 512) to 209.6x (float n = 16), falling monotonically with n.

**No kept cell loses, anywhere in the op.** Over 210 kept cells the minimum is **1.58** (cfloat
`1024x128` b2048) and the maximum 501.2 (cdouble n = 4 b8192). The single sub-1.00 reading in
the raw data, cdouble `1024x128` b8192 at 0.892, is the 48 GB oversubscribed cell excluded
above — its own two lower rungs read 2.687 and 2.719, so it is paging, not a crossover. The
other excluded cdouble cell, `512x64` b16384, reads 1.643 against 2.674 / 2.709 below it: same
artefact, and it does not cross either. That is
what the plan's orgqr flip (native everywhere at `n <= 512`, every type) needs, and this grid
gives it with no exception.

**52 of the run's 60 discarded cells are here, and they are one-sided**: **all 52 are the
native arm**, 49 of them at n <= 48, with `rel_sd` 0.10-0.12, while the vendor arm on the same
cell reads 0.02-0.07. The pattern is that each native rep is preceded in the interleave by a vendor rep
that dispatched `batch` separate cuSOLVER calls across an out-of-order sub-queue; the harness's
`q->wait()` bounds the *main* queue. Treat every orgqr number on this page as a median with a
wide arm, and re-measure orgqr single-arm before quoting one in a gate.

## getrs

Two right-hand-side widths. `nrhs = 1` is inside shipped clause A of
`route_getrs.hh:97` (`s.nrhs() <= 2`, **every type**, no order or batch bound); `nrhs = 4` is
inside clause B (float only). Everything below therefore describes **live routed traffic in a
vendor-present build**, not a hypothetical.

| n (batch) | float | cfloat | double | cdouble |
|---|---|---|---|---|
| **nrhs = 1** | | | | |
| 4 (b32768) | **0.71**~ | **0.59**~ | **0.52**~ | **0.23**~ |
| 8 (b32768) | 1.12 | **0.69**~ | **0.98**~ | **0.41**~ |
| 9 (b32768) | 1.02~ | **0.83**~ | **0.97**~ | **0.82** |
| 16 (b32768) | 1.04~ | 3.76 | **0.99** | 2.00 |
| 17 (b32768) | **0.76**~ | 1.38 | **0.92** | 2.04 |
| 24 (b32768) | 1.11~ | 1.27~ | 3.69 | 1.96 |
| 32 (b16384) | 2.29 | 1.40~ | 3.86 | 2.17 |
| 48 (b16384) | 1.94~ | 1.60 | 3.99 | 2.48 |
| 64 (b16384) | 1.65~ | 1.25 | 3.95~ | 2.66 |
| 96 (b8192) | 2.04 | 1.63~ | 3.31 | 2.49 |
| 128 (b8192) | 1.75~ | 1.56 | 2.95 | 2.53~ |
| 192 (b4096) | 2.15 | 1.47 | 2.46~ | 2.15 |
| 256 (b4096) | 1.93 | 1.36 | 2.16~ | 1.99 |
| 384 (b1024) | 1.90~ | 1.45~ | 1.92~ | 1.73~ |
| 512 (b1024) | 1.73~ | 1.35~ | 1.73~ | 1.60~ |
| **nrhs = 4** | | | | |
| 4 (b32768) | **0.45**~ | 0.35~ | 0.12~ | 0.07~ |
| 8 (b32768) | **0.46**~ | 0.38~ | 0.26~ | 0.18~ |
| 16 (b32768) | 1.64 | 1.84 | 0.56~ | 0.53 |
| 32 (b16384) | 1.30 | 1.03~ | 0.85 | 0.58 |
| 64 (b16384) | 1.51~ | 1.27~ | 0.94 | 0.71 |
| 128 (b8192) | 1.74 | 1.60~ | 1.11 | 1.23 |
| 256 (b4096) | 1.72 | 1.39 | 1.20 | 1.88 |
| 512 (b1024) | 1.53~ | 1.27~ | 2.75~ | 2.14 |

(Bold marks a cell inside a shipped `preferred()` clause that loses. The `nrhs = 4` table omits
the +1 rungs for width; they are in the CSV and change nothing.)

At `nrhs = 1` and 32 <= n <= 512 the fused tier is the strongest native code on this page —
**1.25x to 3.99x, every type, every order, no losses** — and `docs/perf/lu.md` is right about
that band. (The `nrhs = 4` columns for double and cdouble sit below 1.00 from n = 32 to 65;
clause B is float-only, so those cells are measured, not routed.)

**It is wrong below it.** Clause A and clause B together cover 266 measured cells here, and
**32 of them lose**, all at n <= 24, worst **0.234 at cdouble n = 4 nrhs = 1 batch 32768**:
float 9 losses (0.448 min), cfloat 4 (0.593), double 10 (0.521), cdouble 9 (0.234). `lu.md`
records clause A as "286 cells, geomean 2.261, **min 1.116, zero losses**" and clause B as
"min 1.133, zero losses". **The losses are one-directional in batch**, which is what makes them
a defect rather than noise: float n = 4 nrhs = 1 reads 1.504 / 1.073 / 0.709 / **0.521** at
batch 8192 / 16384 / 32768 / 65536, and float n = 4 nrhs = 4 reads 0.658 / 0.448 / **0.302**
over the last three. The clause has no order bound and no batch bound; the grid behind it
evidently had no n < 32 rung and no batch above ~16384. **A cheap P0 repair is an order floor
on clause A and B** — the bracketing winner is n = 24 nrhs = 1 (1.11 float, 1.27 cfloat, 3.69
double, 1.96 cdouble) — but the shape of the loss (deepening with batch at n = 4 while n = 17
bounces 0.878 / 0.755 / 0.922) says the floor should be measured, not guessed.

**The floor was measured, and it is 32.** The paragraph above asks for a measured floor rather
than a guessed one; this is the table that answers it. Same 20 grids, `nrhs = 1`, but read at the
**worst rung of each order's batch ladder** instead of at the top rung — the statistic that decides
whether a clause has a loss anywhere inside it:

| T | n=4 | n=8 | n=16 | n=17 | n=24 | n=32 | n=48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| float | 0.71 | 1.12 | 1.04 | 0.76 | 0.95 | 2.29 | 1.94 |
| cfloat | 0.59 | 0.69 | 3.76 | 1.38 | 1.27 | 1.40 | 1.60 |
| double | 0.52 | 0.98 | 0.99 | 0.87 | 3.66 | 3.77 | 3.74 |
| cdouble | 0.23 | 0.41 | 2.00 | 2.02 | 1.93 | 2.13 | 2.46 |

Read this way the bracketing winner named above dissolves: **n = 24 is 0.95 for float** at its worst
rung, against the 1.11 the top-batch table quotes, so it cannot bracket the floor. 32 is the first
order where all four types clear the flip gate *and* stay clear above it, and the band below it is
non-monotone — float passes at 8, fails at 9, 16, 17 and 24 — so no lower floor is defensible.
Clause B takes the same floor on the same grid: float `nrhs = 4` reads 0.45 / 0.46 at n = 4 / 8 and
1.07 / 1.10 at 17 / 24, clearing only from 32 (1.30).

The mechanism, and why the losses deepen with batch: the fused kernel gives one work-group to a
matrix whose whole solve is a few dozen flops, so the work-group **is** the cost, while
`cublas?getrsBatched` keeps its per-item work inside one kernel and pays no such floor. P2 of the
small-n plan is what reclaims the band — a fused factor-and-solve kernel holding the matrix in
registers has no work-group floor to pay. The full write-up, with the correction it forces on the
"min 1.116, zero losses" reading, is `docs/perf/lu.md` §[`getrs` order floor
evidence](lu.md#getrs-order-floor-evidence).

## Saturation

**The `SAT_LADDER` inherited from `docs/perf/lu.md` does not saturate these ops.** Of the 442
cells with a clean nominal and top rung, **201 (45%) moved more than 5% on the last doubling**. The cause is not statistical: at potrf
float n >= 96 the **native arm is exactly linear in batch and the vendor arm is superlinear**,
so the ratio has no fixed point inside the ladder at all.

| potrf float | native, per doubling | cuSOLVER, per doubling |
|---|---|---|
| n = 128, b 2048 -> 4096 -> 8192 | x1.99, x1.99 | x2.24, x2.35 |
| n = 256, b 1024 -> 2048 -> 4096 | x2.13, x2.03 | x2.37, x2.52 |
| n = 512, b 256 -> 512 -> 1024 | x2.10, x2.08 | x2.32, x2.36 |

Four cells were carried past the ladder to find the fixed point (same guard, same binary,
one process per cell):

| cell | ladder | extension | verdict |
|---|---|---|---|
| potrf float 128 | 0.322 / 0.363 / 0.429 | 0.457 (b16384), **0.466** (b32768) | saturated at b32768, +2.0% on the last doubling |
| potrf float 256 | 0.620 / 0.687 / 0.851 | 0.916 (b8192) | still +7.6%, **not saturated** |
| potrf float 512 | 0.948 / 1.045 / 1.187 | 1.330 (b2048), **1.503** (b4096) | still +13%, **not saturated**, rising |
| geqrf float 128 | 3.175 / 3.322 / 3.643 | **3.618** (b16384) | saturated at b16384, -0.7% |
| geqrf float 256 | 8.735 / 7.393 / 7.573 | 7.775 (b8192) | +2.7%, effectively flat |
| geqrf float 512 | 26.279 / 17.364 / 13.376 | **10.98** (b2048) | still -18%, **not saturated**, falling |

The two n = 512 cells move in opposite directions and neither converges inside 24 GB. Quote
potrf float 512 as **">= 1.19 and rising"** and geqrf float 512 as **"<= 13.4 and falling"**,
not as point estimates. Two further mechanisms show up on the small orders and are worth
knowing before reading a `~` there: the potrf CTA arm takes a one-off superlinear step exactly
where the batch's working set crosses the 72 MB L2 (float n = 32 at b16384 = 67 MB reads x2.51
against x1.77 on the rung below; n = 48 at b8192 = 75 MB reads x2.56), and below n ~ 16 both
arms are launch-bound, so the ratio there is a ratio of dispatch overheads (see the next
section).

## Where this baseline disagrees with the record

66 cells on this page have a per-cell number already recorded in `docs/perf/{potrf,lu,qr}.md`.
**43 reproduce within 15%. 23 do not.** Every disagreement is below, at the prior's own batch; `*` marks the two rows where the
prior's batch (4096) is not on this ladder for that order and the nearest rung, 8192, is used —
the ratio there is flat to 1% across 4096..32768, so the substitution changes nothing.
The vendor arm reproduces the record to three significant figures wherever the record states a
vendor time — cuBLAS `getrfBatched` float n = 512 b512 49.67 ms against a recorded 49.73;
`geqrfBatched` float n = 256 b2048 121.61 against 121.4 and n = 512 b512 370.83 against 370.9;
cuSOLVER `potrfBatched` float n = 128 b4096 1.0503 ms against a table whose ratio implies 1.10
— so **no disagreement here is the vendor moving.**

| op | type | n | batch | prior | now | delta | explanation |
|---|---|---:|---:|---:|---:|---:|---|
| geqrf | float | 128 | 4096 | 2.01 | 3.32 | +65% | **tier.** The prior is the CTA time; the shipped default is Blocked |
| geqrf | cfloat | 128 | 4096 | 3.04 | 3.71 | +22% | unexplained (CTA cannot serve this order for cfloat) |
| geqrf | cdouble | 128 | 4096 | 0.52 | 0.69 | +32% | unexplained (CTA cannot serve this order for cdouble) |
| geqrf | float | 256 | 2048 | 5.62 | 7.39 | +32% | unexplained (blocked driver; three causes ruled out) |
| geqrf | cfloat | 256 | 2048 | 5.11 | 6.85 | +34% | unexplained, same |
| geqrf | cdouble | 256 | 2048 | 0.84 | 1.32 | +57% | unexplained, same |
| geqrf | float | 512 | 512 | 12.18 | 17.36 | +43% | unexplained, same |
| geqrf | cfloat | 512 | 512 | 8.87 | 13.08 | +47% | unexplained, same |
| geqrf | cdouble | 512 | 512 | 1.41 | 2.49 | +77% | unexplained, same |
| orgqr | float | 32 | 8192 | 123.2 | 52.9 | -57% | vendor per-item loop ~2.3x cheaper per item |
| orgqr | double | 32 | 8192 | 114.1 | 45.1 | -60% | same |
| orgqr | cfloat | 32 | 8192 | 65.1 | 38.2 | -41% | same |
| orgqr | float | 128 | 4096 | 17.1 | 9.35 | -45% | same |
| orgqr | double | 128 | 4096 | 30.8 | 12.0 | -61% | same |
| orgqr | cfloat | 128 | 4096 | 10.6 | 5.23 | -51% | same |
| orgqr | float | 256 | 2048 | 9.5 | 5.32 | -44% | same |
| orgqr | double | 256 | 2048 | 15.3 | 8.21 | -46% | same |
| orgqr | cfloat | 256 | 2048 | 5.8 | 3.96 | -32% | same |
| orgqr | double | 512 | 512 | 6.0 | 4.52 | -25% | same |
| orgqr | cdouble | 256 | 2048 | 3.4 | 4.66 | +37% | unexplained; cdouble moves the *other* way |
| orgqr | cdouble | 512 | 512 | 1.7 | 2.75 | +62% | unexplained, same |
| potrf | float | 8 | 8192* | 2.75 | 2.08 | -24% | see below — does not reproduce on any instrument |
| potrf | cfloat | 8 | 8192* | 1.79 | 1.43 | -20% | same |

**geqrf at n = 128 is a tier mismatch, and it is exact.** Pinning the arm at that cell:
`cta` 15.419 ms, `blocked` 9.349 ms, vendor 31.024 ms. `qr.md`'s order grid records
"30.9 -> 15.4 ms (2.01x)" — the native number is the **CTA** time to four figures, while its own
tier table two sections earlier records the *default* at that cell as blocked, 9.97 ms. The
order grid's float `n = 128` row describes a route that has not been the default since
`native_tier_preferred` shipped. **It explains that one row and no other**: `qr.md`'s own tier
table records the `cta` pin as "n/a" for cfloat at n >= 112 and for cdouble at n >= 80, so at
those cells the prior's native arm was already the blocked driver, as ours is.

**geqrf everywhere else is not that**, because CTA cannot serve those orders at all
(`m*n > cta_max_elems`) and both arms resolved as they do today. Ruled out: the tier (above);
the vendor arm (reproduces exactly); and the **trailing GEMM route** — forcing
`BATCHLAS_GEMM_ROUTE=vendor` under the native arm changes it by 0.1% at all three of n = 128 /
256 / 512, so the default already is the vendor-GEMM configuration and a GEMM change cannot be
the story. What is left is the blocked driver itself (panel, `larft`, `pack_v`, schedule),
which is 1.31x faster at n = 256 and 1.43x at n = 512 than `order.csv` records. Not isolated
further; this run had no bisection budget.

**orgqr's vendor arm halved at n <= 256 and did not move at n = 512.** That is the signature of
a per-item *launch* cost, not a per-item work cost: at n = 32 the loop is nearly pure launch
overhead (54 us/item today), at n = 512 each item's own kernel hides it (486 us/item), and only
the small end moved. Driver 595.84 is the obvious candidate and this run cannot test it —
`qr.md` records no absolute vendor time at these cells. The cdouble rows moving the *opposite*
way (+37%, +62%) are unexplained and are the one place where the pattern breaks.

## The potrf n = 8 finding

`docs/perf/potrf.md` records **2.75x** at float n = 8, batch 4096, and
`experiments/wp4_potrf/README.md` §4 is where it comes from. This run reads **2.08-2.18x**, and
the number does not reproduce on any instrument, on either card, at any pin.

| instrument | card | vendor ms | native ms | ratio |
|---|---|---|---|---|
| archived `realpotrf.cpp`, `BATCHLAS_POTRF_ROUTE` pinned | GPU 1 | 0.0284 | 0.0130 | **2.18** |
| archived `realpotrf.cpp`, same | GPU 0 (the card §4 used) | 0.0292 | 0.0134 | **2.18** |
| archived `realpotrf.cpp`, unpinned (build resolves) | GPU 1 | 0.0260 | — | — |
| `factor_bench`, one arm per process | GPU 1 | 0.0261 | 0.0143 | 1.83 |
| `factor_bench`, interleaved | GPU 1 | 0.0255 | 0.0121 | 2.10 |
| `factor_bench`, the grid, b8192 / 16384 / 32768 | GPU 1 | 0.0395 / 0.0588 / 0.1024 | 0.0190 / 0.0284 / 0.0488 | 2.08 / 2.07 / 2.10 |

The archived harness is the one whose source the P0 harness was ported from; it was rebuilt
verbatim (`git show perf-evidence/vendor-independence:experiments/wp4_potrf/phase2_ab/realpotrf.cpp`)
against today's `.so`s. **Its controls reproduce §4 exactly** — n = 48 reads 1.227 against a
recorded 1.26, n = 128 reads 0.364 against 0.36, n = 32 reads 1.820 against 1.85 — and
`factor_bench` agrees with it to within 0.3% at both control orders. The disagreement is
confined to the two smallest rows of that table (n = 8, and n = 16 at 1.63 against 1.88).

Ruled out, in the order they were tested:

* **The new harness.** The archived instrument reads the same thing.
* **The route.** Coverage readback gives `native:cta` and `vendor:auto` on every row, which is
  what §4's table assumes, and the archived harness's unpinned run resolves to the vendor time.
* **The kernel.** `src/extensions/potrf_cta_device.hh` is character-identical to the tag once
  comments are stripped; `potrf_cta.cc` differs only in which exception types five `throw`s
  construct.
* **The card.** GPU 0, which §4 used (`experiments/gpu_guard.sh 0`), gives 2.18 — the same as
  GPU 1.
* **The vendor library.** `libcusolver.so.12.2.0.11`, installed 2026-05-08, predates both
  campaigns.
* **Saturation.** The ratio is flat at 2.07-2.10 over batch 8192-32768, and a linear fit of the
  ladder gives vendor `18.5 us + 2.56 ns/matrix` against native `9.07 us + 1.21 ns/matrix`:
  **marginal ratio 2.11, intercept ratio 2.04.** There is no batch at which this cell reads
  2.75.

What that leaves is the character of the cell rather than a defect. At n = 8, batch 4096 the
DRAM roof is 2.1 MB / 950 GB/s = **2.2 us**, against 13 us native and 28 us vendor: **more than
80% of both arms is per-call dispatch and launch latency**, which is also why this is the most
context-sensitive number in the whole probe — the same cell reads 1.83 / 2.10 / 2.18 / 2.18
across four process configurations, a 19% spread, while n = 48 reads 1.227 / 1.228 / 1.230, a
0.3% spread. `docs/perf/README.md`'s own rule ("an unsaturated ratio measures overhead, not the
algorithm") applies to it.

**And there is no raw data behind the prior.** `experiments/wp4_potrf/` at the evidence tag
holds 25 CSVs; every one belongs to the SLM capacity study, the register probes, or the Phase 2
blocked campaign. **§4's table — the CTA-vs-cuSOLVER grid, all four types, n = 8..155 — has no
archived CSV, and neither §4 nor `potrf.md` names the harness that produced it.** It is the
only per-cell table in the potrf record that cannot be re-derived from the archive. The
conclusion this page records is therefore: *the n = 8 and n = 16 rows of that table are not
reproducible and should be replaced by the numbers here*; the rest of it reproduces and stands.

## Raw data

| what | path |
|---|---|
| the 20 grids, every row including the discarded ones | `benchmarks/results/factor_baseline_<op>_<type>.csv` |
| the harness | `benchmarks/factor_bench.cc` |
| the ladder driver and the GPU guard | `benchmarks/run_factor_grid.sh`, `benchmarks/gpu_guard.sh` |
| the archived instrument used to re-test the potrf n = 8 prior | `git show perf-evidence/vendor-independence:experiments/wp4_potrf/phase2_ab/realpotrf.cpp` |
| the prior tables this page compares against | `docs/perf/potrf.md` §"CTA kernel measured against cuSOLVER", `docs/perf/qr.md` §"geqrf order and batch grid" and §"orgqr grid", `docs/perf/lu.md` §"Negative results" 4 |

The saturation extensions, the tier pins and the GEMM-route probe in this page were run one
process per cell through the same guard and are not in the CSVs; they are quoted inline with
both arms' absolute times so they can be re-run from the text.

## Warm-up order and the variance gate

Method detail behind the `relsd` discards of [How to read the grid](#how-to-read-the-grid), moved
out of `benchmarks/factor_bench.cc` so the gate can be audited from this page. The harness warms up
**time-based** (a `WARM_S` budget per arm, so the warm-up loop runs for `WARM_S x arms`) and
**discards** the warm-up reps: a cold first run — SYCL JIT plus cold clocks — has fabricated a
**3.7x** result in this repository.

The warm-up is also **interleaved**, in the same arm order the timed loop uses, and that is measured
rather than stylistic. With a per-arm warm-up (all of arm 0, then all of arm 1) the first *timed* rep
of arm 0 is the only one in the run not preceded by an arm-1 call, and it came in **2.2x slow every
time** — **0.0567 ms against a steady 0.0261** at `potrf` float n = 8 — which alone pushed the vendor
arm's `rel_sd` to **0.27** and tripped the gate on a cell whose median was perfectly stable. Warming
in the timed loop's own order removes it.

The consequence for reading the grid: a `relsd` discard on this page is a property of the *route*, not
an artefact of the first rep, because no arm in the timed loop has an unwarmed neighbour position. The
0.0261 ms steady reading is the same `factor_bench`, one-arm-per-process vendor time quoted in
[The potrf n = 8 finding](#the-potrf-n--8-finding).

## The arms list

`benchmarks/factor_bench.cc` originally supported exactly two arms, `vendor` and
`native`, with `--route=` applied to the second, and matched `--arms=` by
`std::string::find`. Two defects followed from that, and the register-resident tier's
A/B could not be run through it at all.

**The substring match is the recorded `--name` trap in another costume.**
`--arms=native` asked for one arm and got one, but only because `find("vendor")` missed;
`--arms=vendor` silently dropped the native arm. Neither spelling is an error and
neither prints a warning -- the CSV simply comes back with one row where the caller
expected two, and a script that ratios row 0 against row 1 reads whatever came next.
`--arms=` now splits on commas EXACTLY and rejects an empty list.

**Two arms cannot express a three-tier comparison.** Comparing `vendor`, `cta` and
`tiny` needed either two processes ratioing through a shared vendor arm -- which
reintroduces precisely the clock drift the in-process interleave exists to remove -- or
this change. `Cfg` now carries a `std::vector<std::string>` of arm names, each name is
its own route pin (`--route=` still overrides the pin of the arm literally named
`native`, which is the two-arm form every existing script uses), and the warm-up loop,
the timed loop, the per-arm workspace sizing and the per-arm untimed correctness re-run
all already iterated over `arms`, so they needed no change.

The guards that make an added arm safe were also already there and are worth naming,
because an arm whose pin does not parse measures whatever `automatic()` picks and says
nothing about it: `pin_parsed_now()` is recorded per arm in the CSV's `pin_parsed`
column, and the untimed residual/pivot gate runs per arm, so a fast wrong answer cannot
be reported as a win. What is NOT guarded is an arm whose pin parses but whose route
`supports()` refuses -- `BATCHLAS_GETRF_ROUTE=tiny` at cdouble n=32 parses fine and then
falls through to the vendor, so that cell must be excluded by the caller.
