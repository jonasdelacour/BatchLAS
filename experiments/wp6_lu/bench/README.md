# WP6 — LU measured: `getrf`, `getrs`, `getri` against cuBLAS

This is the MEASURE phase. It re-derives nothing from
`experiments/wp6_lu/kernels/`: every number here was produced by its own runs, on
its own binaries, and where it overlaps the implementer's grid and the
orchestrator's baseline it **reproduces both**, which is stated in §1 as a check
and not as a courtesy.

**Everything is the PUBLIC API.** One program (`lubench6.cpp`) linked twice —
once against `build/`, once against `build-novendor/` — so "vendor-free" is the
BUILD, never a forced route. Correctness is checked IN PROCESS against a HOST
oracle on every timed row and the RESOLVED ROUTE is printed on every row.

---

## 0. The one-paragraph answer

**cuBLAS `getrf`/`getri`Batched are genuinely batched, and at small `n` they are
also genuinely better — but at large `n` they are BATCH-PARALLEL ONLY, and that
is where every large native "win" comes from.** Read at the batch schedule an A/B
grid would use, native's geomeans are `getrf` 0.885×, `getri` 1.463×,
`getrs` 0.617×. Read at each arm's own best batch — the honest reading, because
cuBLAS's µs/item is still falling at the grid's batch for every `n ≥ 512` cell —
they are **0.805×, 1.284×** and the `n = 2048` row collapses from **7.10× to
2.95×**. Underneath that, three things are true and none of them is a routing
question: for FP64 cuBLAS is already at **46–91 % of this card's FP64 peak** and
there is nearly nothing left to win; for FP32 **both arms are at 1–10 % of peak**
and the whole op is bound by a row-interchange kernel that is 13–63 % of native
`getrf`; and `getrs` is not a loss but a **crossover on `nrhs`**, losing 0.32×
at `nrhs = 1` and winning 1.36× at `nrhs = 128`.

---

## 1. The vendor arm reproduces two independent tables

Before any ratio: the vendor arm of this directory was compared, cell by cell,
against the orchestrator's baseline table and against the implementer's
`grid_vendor.csv`.

* **28/28 `getrf` cells match the orchestrator's baseline within 2.2 %**
  (float `0.156 / 2.606 / 5.760 / 31.74 / 49.56 / 94.42 / 518.6` ms against its
  `0.159 / 2.629 / 5.770 / 31.64 / 49.73 / 94.53 / 519.2`; cdouble `n = 2048`
  2594.60 against 2594). `getri` float `n = 2048` 1475.16 against 1476.7.
* **The A/B ratios at the implementer's batch schedule match to within 1 %**
  (`getrf` float 0.539/1.599/0.733/0.972/1.225/2.018/7.314 against their
  0.545/1.614/0.734/0.966/1.235/2.029/7.291; `getri` float 33.842 against
  33.942).

So the disagreements that follow are about WHICH BATCH to read at, not about
what the two arms cost.

---

## 2. Saturation, established before any ratio exists

`sat_summary.txt` — µs **per batch item** against batch, at fixed `n`, both arms,
all four types, `getrf` and `getri`. The rule, fixed before it was applied: a rung
SATURATES at the first batch within 5 % of the best point on the ladder, provided
the top of the ladder is flat to within 5 %.

**cuBLAS does not saturate at `n ≥ 1024` on any ladder this box can hold**, and
it is unsaturated at `n = 256` and `512` for float too. The extreme case:

```
cuBLAS getrf cdouble n=2048   WALL TIME, not per item
   batch    4      8     16     32     64    128
   ms    2587   2589   2591   2595   2657   2801
```

Sixteen times the work for 0.3 % more time. That call is a **latency chain**, not
a throughput limit: batch is free to cuBLAS up to ~64 there. The same shape,
milder, everywhere above `n = 512`. Native is the opposite — it saturates at
batch 4–32 at `n = 2048`, because it parallelises WITHIN the item.

Several rungs are also **interior-optimal**, i.e. the grid's batch schedule is
pessimistic to the arm named — to cuBLAS on two cells and to native on two:

| cell | best batch | grid batch | penalty at the grid batch |
|---|---|---|---|
| cuBLAS `getri` float n=256 | 256 (14.05 µs/item) | 2048 (20.30) | 1.45× |
| cuBLAS `getrf` float n=256 | 256 (8.19) | 2048 (15.50) | 1.89× |
| native `getrf` float n=512 | 128 (48.27) | 1024 (87.67) | 1.82× |
| native `getri` float n=512 | 64 (24.63) | 1024 (38.39) | 1.56× |

Both arms have them; neither is silently corrected. They are why §3 reports two
readings side by side.

---

## 3. The A/B, both readings, and the gap between them IS the finding

`tail_summary.txt`, `geomeans.txt`. `ratio_at_common_batch` is the reading an A/B
grid produces at the schedule 32:8192 … 2048:32. `ratio_at_own_ceilings` puts each
arm at its own best measured batch.

```
                        geo_common   geo_ceiling   wins_common  wins_ceiling
   getrf  (28 cells)       0.885        0.805          9/28        10/28
   getri  (28 cells)       1.463        1.284         16/28        16/28

   per order (both ops, all types)
   n=  32                  0.358        0.352          0/8          0/8
   n=  64                  0.581        0.594          1/8          1/8
   n= 128                  0.610        0.667          2/8          2/8
   n= 256                  1.042        1.041          3/8          4/8
   n= 512                  1.351        1.502          5/8          5/8
   n=1024                  1.952        1.746          6/8          6/8
   n=2048                  7.098        2.954          8/8          8/8
```

**The `n = 2048` row is a factor of 2.4 apart.** Every headline number at that
order must carry which reading it is. The individual collapses:

| cell | at the grid's batch | at both ceilings |
|---|---|---|
| `getrf` float n=2048 | 7.31× | **2.33×** |
| `getri` float n=2048 | 33.84× | **8.09×** |
| `getri` cfloat n=2048 | 25.64× | **13.40×** |
| `getrf` cdouble n=2048 | 3.74× | **1.05×** |
| `getri` cdouble n=2048 | 4.59× | **2.33×** |
| `getrf` double n=2048 | 2.76× | **1.52×** |

`getrf` cdouble at `n = 2048` is the one to quote when someone quotes 3.73×: at
batch 128 — the largest 24 GB holds — it is **1.053×, and cuBLAS is still
unsaturated there**, so the true figure is at most that and probably below 1.

### Per type, at the ceilings

```
                getrf                                  getri
        32    64   128   256   512  1024  2048     32    64   128   256   512  1024  2048
float  0.55  1.61  0.74  1.03  1.80  2.00  2.33   0.55  0.97  1.43  3.41  5.48  6.63  8.09
double 0.28  0.32  0.33  0.64  0.73  0.82  1.52   0.23  0.54  1.07  1.11  1.30  1.16  2.15
cfloat 0.55  0.82  0.54  0.68  1.27  1.53  2.87   0.23  0.34  0.71  1.75  3.19  5.14 13.40
cdoubl 0.41  0.44  0.41  0.55  0.71  0.85  1.05   0.23  0.47  0.68  0.85  0.96  1.03  2.33
```

geomeans at the ceilings: `getrf` float **1.28×**, cfloat 0.99×, double 0.56×,
cdouble 0.59×; `getri` float **2.58×**, cfloat 1.55×, double 0.90×, cdouble 0.76×.

---

## 4. Why: the roofline says the FP64 case is nearly closed and the FP32 case is wide open

`tail_summary.txt` carries achieved REAL flop rates (the harness's GFLOP/s column
is the LAPACK convention, `2/3 n³` for `getrf` regardless of type; real hardware
flops are 4× that for the complex types). References for one RTX 4090: FP32
82.6 TFLOP/s theoretical, ~47 TFLOP/s measured for GEMM on this box; FP64
1.29 TFLOP/s.

```
getrf, % of the type's roofline, each arm at its own best batch
        n=128        n=512         n=2048
        cuBLAS nat   cuBLAS nat    cuBLAS nat
double   53%   18%    63%   46%     46%   70%
cdouble  73%   30%    90%   64%     81%   85%
float   1.2%  0.9%   1.2%  2.2%    1.4%  3.2%
cfloat  4.1%  2.2%   3.5%  4.5%    2.9%  8.3%
```

Two conclusions the ratios alone cannot give:

1. **For FP64, cuBLAS is close to the machine.** At cdouble `n = 512..1024` it
   runs at 90 % and 91 % of the card's FP64 peak. There is no 2× to find there,
   only the last 10 %, and native's 0.41–0.85× band is a real efficiency gap
   (30–78 % of peak) rather than a routing artefact. At cdouble `n = 2048` BOTH
   arms are at 81 % and 85 % — parity at the roofline, which is exactly what
   §3's corrected 1.053× says.
2. **For FP32, neither arm is close to anything.** cuBLAS's float `getrf` sits at
   1.0–1.4 TFLOP/s at EVERY order — a hard cap, 1.2–1.7 % of peak. Native reaches
   2.6 TFLOP/s. Both leave 15–40× on the table against the measured GEMM ceiling.
   The FP32 LU on this box is not compute-bound at all, and §5 says what it is
   bound by instead.

---

## 5. nsys: where the time goes, on winners AND losers

`nsys_splits.txt`, `kernsum/*.txt`. Captures are of the vendor-free binary at
`WARM_S=0.2`, 2 reps; **no time in this section is a performance number**.
`*.nsys-rep` and `*.sqlite` are gitignored.

### The winner (`getrf` float n=2048 b=32, 7.31× / 2.33×)

```
  44.2%  378 launches  LuLaswpKernel<GetrfBlockedLaswpTag, float>
  29.0%  135           GemmRegister128x128Kernel<float, false>
  14.7%  123           GetrfPanelGlobalKernel<float>
   8.1%   45           GemmRegister128x128Kernel<float, true>
   2.1%   69           GetrfPanelResidentKernel<float>
   1.8%  189           TrsmCtaKernel<float, 32, Left>
```

**44 % of the best `getrf` cell is the row interchange.** At `n = 1024` it is
**63.2 %**. That is an independent confirmation, by profile, of the implementer's
44 %/74 % figures, which were priced by a timing-only break — two different
methods, same answer. It is also the mechanism behind native's interior optima in
§2: the interchange's p-side touches `ib` scattered rows per column, one 128 B
line each, so it is L2-resident at small batch and DRAM-bound past it.

### The loser (`getrf` double n=128 b=4096, 0.27× / 0.33×) — and it is NOT a slow kernel

```
NATIVE                                     VENDOR, same cell
  48.5%  16  GetrfPanelResidentKernel<double>     99.9%  12  getrf_panel<double,double,4,4,3,3,4,true>
  33.6%  24  LuLaswpKernel<...,double>             0.1%  12  init_data_ptr_array
   9.2%  12  GemmTiledGeneralKernel<double,16,..>
   8.8%  12  TrsmCtaKernel<double,32,Left>
```

**cuBLAS factorises the whole `n = 128` problem in ONE fused kernel.** Native
runs four kernels per block step — panel, interchange, trsm, gemm — with a global
round-trip between each. The loss at small `n` is a DECOMPOSITION, not a kernel
quality problem, and no amount of tuning the four kernels closes it; only a
CTA-resident arm that keeps the whole factorization in local memory does. Native
HAS such an arm, and `double`'s tier hook routes `n > 32` away from it, so at
`n = 128` double pays the four-kernel price. That is where a re-measurement is
owed (§8).

### The complex prediction: **REFUTED for the LU trailing update**

The standing prediction on record is that every complex trailing gemm
short-circuits to `Tiled16`. Observed:

```
cx_getrf_cdouble_1024   42.3% GemmRegister64x64K16WideKernel<complex<double>, false>
                        35.9% GemmRegister64x64K16WideKernel<complex<double>, true>
lose_getrf_cdouble_128  22.2% + 4.6%  same wide-scalar kernel
```

Both complex trailing updates reach the **wide-scalar register kernel**, not
Tiled16, at both a small and a large order. `nb = 32` is why: `gemm_kernels.cc`
gates the complex wide-scalar path on `min_dim ≥ 32`, and the trailing NN update's
`k` is exactly `nb`. The prediction survives for **double**, which has no register
kernel on this path and lands on `GemmTiledGeneralKernel<double,16,…>` — observed
in the loser above. So the deficit the campaign records as "complex loses on
Tiled16" is, for LU, **a `double` deficit, not a complex one**.

### `getrs`, both sides of its own crossover

```
nrhs = 1  (0.57x)                            nrhs = 128  (2.15x)
  34.1% 1694  GemmTiledGeneralKernel<float,16>   35.6%   58  LuLaswpKernel<GetrsLaswpTag>
  24.8% 2479  TrsmCtaKernel<float,32,Left>       22.0%  351  GemmRegister128x128Kernel<float,true>
  21.5%   77  LuLaswpKernel<GetrsLaswpTag>       17.5% 1394  GemmRegisterTiledKernel<float,32,32,8>
  12.6%   30  LuLaswpKernel<GetrfBlockedLaswp>   17.3% 1871  TrsmCtaKernel<float,32,Left>
```

(The `GetrfBlockedLaswp` rows are the harness's untimed factorisation, run once
per process; they are listed rather than removed.) At `nrhs = 1` the routed
trsm's own trailing update degenerates to `m×1×k` and falls to `Tiled16` and
`GemmDirect` — a GEMV in all but name, and therefore **WP7's problem, arriving
inside WP6**. The permutation is the other half: `lu_laswp.hh` launches
`range<2>(batch, ncols)`, so at `nrhs = 1` it is `batch` work-items TOTAL — 32 of
them at `n = 2048, batch = 32`, on 128 SMs, each walking 2048 sequential
interchanges. That is this repository's recurring batch-only-parallelism defect,
in a shipped kernel, and it is visible as the 0.083× worst cell.

`getri` cfloat `n = 32` (0.23×, the worst `getri` cell) is **86.2 % one kernel**:
`TrsmCtaKernel<complex<float>,32,Left>`. Nothing in WP6 is on that critical path;
it is WP3's trsm.

---

## 6. `getrs` is not a loss — it is a crossover on `nrhs`

The implementer's grid measured `getrs` at `nrhs = 1` only and reported 0.257×,
0 wins in 28. That reading is correct and it is one point on a three-axis surface.
Sweeping `nrhs` with `n = 512` and `batch = 256` BOTH FIXED (`order_tables.txt`):

```
          nrhs=1  nrhs=2  nrhs=8  nrhs=32  nrhs=128  nrhs=512
 float     0.737   0.601   0.570    1.095     2.151     1.577
 double    0.709   1.100   1.025    1.135     1.334     1.286
 cfloat    0.575   0.465   0.465    0.751     1.148     1.185
 cdouble   0.158   0.383   0.383    0.553     1.043     1.053
```

and the geomean over every `getrs` cell measured, grouped by `nrhs`:

```
 nrhs      1      2      8     32     64    128    512
 geomean 0.323  0.586  0.484  0.848  1.088  1.362  1.261
 wins     0/24   1/4    1/12   2/4   13/20   4/4    4/4
```

**Monotone, and it crosses 1.0 between `nrhs = 32` and `nrhs = 64`** — later for
the complex types (between 32 and 128), immediately for `double`. The all-cell
geomean is 0.617× over 72 cells with 25 wins, which is the number to quote, not
0.257×.

The other two axes, each with the other two held fixed:

```
 BATCH axis, n=512 nrhs=8       b=64   b=128  b=256  b=512  b=1024
 float                          0.542  0.553  0.570  0.572  0.558   FLAT
 cdouble                        0.507  0.443  0.383  0.272  0.268   degrades

 ORDER axis at nrhs=1           n=64   n=256  n=512  n=1024 n=2048
 float                          0.214  0.515  0.568  0.647  0.606
 cdouble                        0.110  0.086  0.083  0.150  0.218
```

So for `getrs` the deciding axis is `nrhs`; batch is flat for float and adverse
for cdouble; order improves the ratio at `nrhs = 1` but never crosses.

---

## 7. Order and batch, separated — and the dominant axis is BATCH

The whole point of `order32_*` and `order1024_*` is that ONE fixed-batch order
sweep cannot distinguish an order crossover from a batch crossover wearing its
clothes. Two of them can. Same orders, same types, same ops, batch the only
difference:

```
                         geomean getrf   geomean getri
 batch FIXED = 32            1.438           3.571      (15/28 and 22/28 wins)
 batch FIXED = 1024          0.668           0.907      ( 4/20 and  8/20 wins)
```

**A 2.2× and a 3.9× swing from the batch axis alone.** The order crossover moves
with it:

```
 getrf, batch=32     n=32   n=64   n=128  n=256  n=512  n=1024 n=2048
 float               0.907  1.060  0.969  1.121  2.642  6.020  7.338
 double              0.988  0.518  0.500  0.933  1.609  2.233  2.767
 cfloat              0.931  0.909  0.363  0.689  1.906  3.696  5.056
 cdouble             0.931  0.665  0.602  1.201  2.258  3.159  3.730

 getrf, batch=1024   n=32   n=64   n=128  n=256  n=512
 float               0.701  1.757  0.873  1.221  1.470
 double              0.365  0.430  0.378  0.604  0.602
 cfloat              0.607  0.872  0.567  0.903  1.157
 cdouble             0.415  0.452  0.373  0.514  0.676

 getri, batch=32     0.797  1.000  2.079  5.573 15.098 30.193 33.781  (float)
 getri, batch=1024   0.696  1.081  1.556  3.078  3.970    --     --   (float)
```

At batch 32 every type's `getrf` crosses 1.0 by `n = 512`. At batch 1024 only
float crosses at all within `n ≤ 512`, and `double` never does. **A `preferred()`
window written from an order sweep alone would be wrong at both ends**; the
routing step needs `(order, batch)` and, for `getrs`, `nrhs`.

---

## 8. Workspace — native is `O(batch)` and independent of `n`

From the `ws` column, which is `getrf_buffer_size` / `getri_buffer_size` /
`getrs_buffer_size` on the exact view being timed:

| path | bytes | at n=2048, batch=32 | at n=64, batch=16384 |
|---|---|---|---|
| `getrf` vendor | 4 B/item | 512 B (allocator floor) | 65,536 B |
| `getrf` native | **36 B/item** | 2,560 B (floor) | 589,824 B |
| `getri` vendor | 4 B/item | 512 B | 65,536 B |
| `getri` native | **0** | 0 | 0 |
| `getrs` vendor | 0 | 0 | 0 |
| `getrs` native | **0** | 0 | 0 |

Native `getrf` is 9× the vendor's per item — `info` plus **four** 8-byte pointer
arrays, one per sub-view role — and it is still 576 KB at the largest batch this
directory measures. Neither path allocates matrix scratch, so nothing scales with
`n²`: the blocked driver factorises in place. `getri` native is the only path in
either arm that needs **zero** bytes, because it writes `P` straight into `C`
instead of writing `I` and permuting.

For scale, the `getrs` gather the implementer costed and did not build is
67 MB at `n = 2048, nrhs = 64, batch = 32` — four orders of magnitude more than
anything above. §6 is the argument for revisiting it; this is its price.

---

## 9. Losses, stated plainly

* `getrf` **loses overall**: 0.805× geomean at the ceilings, 10 wins in 28. It
  loses every `n ≤ 128` cell for every type except one, and it loses `double` and
  `cdouble` at every order below 2048.
* `getrf` **double is the worst family**: 0.560× geomean, one win in seven. §5
  says why — a four-kernel decomposition against a single fused vendor kernel,
  plus the only type on the trailing path with no register GEMM.
* `getrs` **loses at `nrhs ≤ 8`**, 0.32–0.59× geomean, and the worst single cell
  in the directory is `getrs` cdouble `n = 512, nrhs = 1` at **0.083×**.
* `getri` loses `n ≤ 64` for every type and loses `cdouble` up to `n = 512`.
* The `n = 2048` headline numbers **shrink by 2.4× on average** when the vendor is
  given the batch it needs, and `getrf` cdouble's shrinks by 3.5× to parity.
* Native has its own unsaturated and interior-optimal cells (§2); they are listed
  with the vendor's.

Against that: `getri` float and cfloat are large, robust wins from `n = 256` up
(3.4–13.4× at the ceilings), `getrf` float wins from `n = 256` up, and **at
`n = 2048` native wins all eight (op, type) cells even on the corrected reading**.
`preferred()` is unchanged, as instructed.

---

## 10. Hygiene, and what was checked rather than assumed

* **GPU 1 pinned** (`CUDA_VISIBLE_DEVICES=1`) for every run; the two arms never
  ran concurrently; the machine has two RTX 4090s and co-running fabricates
  results.
* **JIT and clocks warmed** for `WARM_S` seconds per cell, untimed and discarded.
* **Medians of 3–5 reps**, with mean and relative sd on every row.
* **DISCARD RULE, fixed before use**: a cell is dropped and NAMED when either arm
  is flagged `BAD`, when either arm's relative sd exceeds 10 %, or when an arm is
  missing. **Nothing was discarded in any table here.** 982 timed rows were
  produced (491 per arm); all 982 are flagged `ok`; the largest relative sd
  observed is **7.2 %** (cuBLAS `getri` double `n = 64`, batch 1024), one row is
  above 5 %, five are above 2 %, and the remaining 977 are below it.
* **The discard rule was BROKEN and confirmed to fire.** A copy of three real
  cells with one row's flag set to `BAD` and one row's relsd set to 0.50 was fed
  to `analyse_ab.py`; it dropped both, named both with their reason, and computed
  the geomean over the one survivor. A discard rule that has never rejected
  anything is not evidence that nothing needed rejecting.
* **Correctness in process on every timed row**, against a host oracle: residual
  under `Tol<T>`, `info == 0`, a non-zero count of non-diagonal pivots, and the
  pivot sequence compared ELEMENTWISE against `LAPACKE_?getrf`. A fast wrong
  answer cannot enter a table here.
* **The resolved route is printed on every row.** All 491 native rows resolved to
  a native arm — 357 `native:blocked`, 134 `native:cta` — and none silently
  resolved to the vendor. All 491 vendor rows are `vendor:auto`. That check is not
  ceremonial: a bare `BATCHLAS_GETRF_ROUTE=native` names neither tier, so
  `supports()` refuses it and `route_resolve.hh` falls through to `automatic()`,
  i.e. silently to cuBLAS — which is why the native arm here is a BUILD and
  carries no pin at all.
* **Nothing was timed under `BATCHLAS_KERNEL_TRACE`** (~60 % inflation), and the
  nsys captures are labelled as splits, never quoted as times.
* **One harness change from `kernels/luverify.cpp`**: the `getrs` row printed a
  hard `size_t(0)` in the workspace column, so no `getrs` workspace figure existed
  anywhere. It now prints `getrs_buffer_size`. Nothing else was touched — WP4's
  measurement was 2× off precisely because it re-derived the driver.

---

## 11. Files and command lines

**Programs.** `lubench6.cpp`; `build_v.sh` / `build_nv.sh` build it against
`build/` and `build-novendor/` respectively.

```bash
bash build_v.sh && bash build_nv.sh
GPU=1 bash run_sat.sh      # batch ladders, float + cdouble, getrf + getri
GPU=1 bash run_tail.sh     # the deeper batches at n >= 1024
GPU=1 bash run_rest.sh     # sat2 (double + cfloat), getrs, both order sweeps
GPU=1 bash run_nsys.sh     # the splits; captures are gitignored
python3 analyse_sat.py   > sat_summary.txt
python3 analyse_tail.py  > tail_summary.txt
python3 geomeans.py      > geomeans.txt
python3 analyse_order.py > order_tables.txt
python3 analyse_ab.py getrs_vendor.csv getrs_native.csv "getrs A/B" > getrs_summary.txt
bash split.sh <tag> ...  > nsys_splits.txt
```

**Cell lists** are generated by `gen_cells.py` (`sat`, `sat2`, `order32`,
`order1024`) or written out (`tail_cells.txt`, `getrs_cells.txt`), so every
sweep's batch schedule is a recorded decision rather than a side effect.

**Data.** `sat_{vendor,native}.csv`, `sat2_*`, `tail_*`, `getrs_*`, `order32_*`,
`order1024_*`; derived `sat_summary.txt`, `tail_summary.txt`, `geomeans.txt`,
`order_tables.txt`, `getrs_summary.txt`, `nsys_splits.txt`, `kernsum/*_kern.txt`.

**Not committed** (`.gitignore`): the two binaries, `nsys/` (44 MB of
`*.nsys-rep` and `*.sqlite`), and any trace JSON.

---

## 12. What this directory does NOT settle

1. **No `preferred()` window is proposed and none was changed.** §7 is the reason
   it cannot be written from an order sweep: the crossover order moves by more
   than a factor of two between batch 32 and batch 1024, and `getrs` needs `nrhs`
   as well. The routing step needs a 2-D (and for `getrs`, 3-D) window.
2. **`getrf` double at `n = 64..128` was not re-measured against its own CTA
   arm.** §5 shows the loss there is the four-kernel decomposition, and the tier
   hook currently sends double away from the one-kernel arm above `n = 32` on the
   strength of a native-vs-native sweep that never asked what the vendor does.
   That is the highest-value open measurement in WP6.
3. **The row interchange was not re-engineered, only priced** — 44 % at the best
   `getrf` cell, 63 % at `n = 1024`, by profile. The two candidate fixes and their
   crossovers are in `lu_laswp.hh`; neither is measured here.
4. **`getrs`'s `nrhs = 1` deficit has two causes and they were not separated
   quantitatively** — the batch-only-parallel permutation launch and the trsm's
   trailing update degenerating to a GEMV. Both are named by profile; splitting
   them needs the gather built, which is a routing-step decision (§6, §8).
5. **cdouble `n = 2048` is memory-limited, not measurement-limited.** cuBLAS is
   still unsaturated at batch 128 and 24 GB will not hold 256, so 1.053× is an
   upper bound on native's advantage there, not a measurement of it.
6. **`nrhs` was swept at one order and one batch** (`n = 512`, batch 256) plus two
   points on the order ladder. The crossover is monotone and consistent across all
   four types, but its position at other orders is interpolated, not measured.
