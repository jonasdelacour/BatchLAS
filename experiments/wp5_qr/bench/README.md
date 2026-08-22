# WP5 — the measured grid: native `geqrf` and `orgqr` against cuSOLVER/cuBLAS

This is the measurement phase for WP5. The kernels are already written, verified
and **route-neutral** (`preferred()` is false for both ops), so nothing here
changes a route. It exists to answer three questions:

1. **Is native QR ever faster than the vendor, and where?**
2. **Where does the time go?**
3. **Where does it lose, and is the loss real or an artefact of the cell?**

Everything below is `build/` (vendor present, `Origin::Auto` → cuSOLVER/cuBLAS)
against `build-novendor/` (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`, the same
`Origin::Auto` → the native kernel through `route_resolve.hh:60-63`). Same
program, linked twice.

---

## 0. Headline

| | geomean vendor_ms/native_ms | cells won |
|---|---|---|
| `geqrf`, 9 orders × 4 types | **3.24x** | 25/36 |
| `orgqr`, 9 orders × 4 types | **7.85x** | 31/36 |
| both | **5.04x** | 56/72 |

But the geomean is the least useful number in this file, because **the two ops
move in OPPOSITE directions with order**:

* `geqrf` native starts behind and pulls away — 0.21–0.78x at n = 32, **15–181x
  at n = 2048**.
* `orgqr` native starts far ahead and falls behind — 12–123x at n = 32, **0.31–
  0.46x at n = 2048** for three of the four types.

They have opposite shapes because the vendor arms are different KINDS of thing.
`cublas?geqrfBatched` is a genuine batched routine that is latency-bound and
saturates at ~380 GFLOP/s (float) no matter the order; cuSOLVER `orgqr` is not
batched at all — `cublas.cc:1414-1419` loops `cusolverDnXorgqr` once per batch
item — so it is a very good single-matrix kernel paying a linear serialisation
penalty. Native beats the first at large n and the second at large batch.

---

## 1. Method, and the four ways this measurement could have lied

**Reproduce.** Run everything from the worktree root, GPU pinned to card 1.

```bash
bash experiments/wp5_qr/bench/build_v.sh      # -> qrbench_v   (build/)
bash experiments/wp5_qr/bench/build_nv.sh     # -> qrbench_nv  (build-novendor/)

bash experiments/wp5_qr/bench/run_order.sh       > order.csv        2> order_err.txt
bash experiments/wp5_qr/bench/run_batch.sh       > batch.csv        2> batch_err.txt
bash experiments/wp5_qr/bench/run_tier.sh        > tier.csv         2> tier_err.txt
bash experiments/wp5_qr/bench/run_tall.sh        > tall.csv         2> tall_err.txt
bash experiments/wp5_qr/bench/run_orgqr_batch.sh > orgqr_batch.csv  2> orgqr_batch_err.txt

python3 experiments/wp5_qr/bench/analyse.py      order.csv       > order_summary.txt
python3 experiments/wp5_qr/bench/analyse.py      batch.csv       > batch_summary.txt
python3 experiments/wp5_qr/bench/analyse.py      tall.csv        > tall_summary.txt
python3 experiments/wp5_qr/bench/analyse.py      orgqr_batch.csv > orgqr_batch_summary.txt
python3 experiments/wp5_qr/bench/analyse_tier.py tier.csv        > tier_summary.txt

bash experiments/wp5_qr/bench/run_nsys.sh          # the split; see nsys_split.md
bash experiments/wp5_qr/bench/run_nsys_orgqr.sh    # the CLEAN orgqr split
```

**A/B is the BUILD, not an environment variable.** `qrbench_v` links
`build/src/*.so` and compiles against `build/include`, where
`BATCHLAS_HAS_CUBLAS` is 1; `qrbench_nv` links `build-novendor/src/*.so` against
`build-novendor/include`, where it is 0. A forced route inside a build that still
links cuSOLVER is not the same experiment — `route_resolve.hh:101` falls through
to `automatic()` when a forced route is unsupported, and `automatic()` returns
`{Vendor, Auto}`.

**It times the PUBLIC API and nothing else.** `geqrf<Backend::CUDA,T>` and
`orgqr<Backend::CUDA,T>`, through the facade, with the workspace the facade's own
`*_buffer_size` asked for. No harness-local driver: WP4's `phase2.cpp` defined its
own `Blocked<T>` class, was 2x slower than the shipped code and contradicted the
real numbers by a factor of two.

**Every row carries its RESOLVED ROUTE, from `backend::geqrf_route` /
`backend::orgqr_route` — the same functions the facade calls.** An unrecognised
`BATCHLAS_*_ROUTE` value silently means `{Auto,Auto}`, which with `preferred()`
all-false is the VENDOR, so a "native" run that looks like the vendor probably
IS the vendor. This is not a hypothetical: in `tier.csv` **20 of the 44 (type, n) cells had a
`cta` pin that resolved to `native:blocked`**, because `m*n` exceeded
`cta_max_elems`. `analyse_tier.py` prints each one as `PIN-DID-NOT-TAKE` and
excludes it. Tabulating them would have produced a CTA/Blocked table in which the
two arms were the same code.

**Correctness is checked in the same process, on the timed array.** Three probes
per orgqr row (‖QRx−Ax‖ from the packed factor; ‖QᴴQx−x‖ on the explicit Q;
‖QRx−Ax‖ with the explicit Q), one per geqrf row. Five apparent wins entered the
WP4 record because a racing kernel was fast and wrong.

**The probes propagate NaN.** `wp5qr.cpp`'s probes use `std::max(0.0, x)`, which
returns the FIRST argument when the comparison is false — so a NaN residual reads
as a PERFECT one. That is a recorded defect
(`experiments/wp5_qr/kernels/README.md` §4b, break K5) and it is still present in
`wp5qr.cpp`. `qrbench.cpp` uses a NaN-propagating `nanmax` in all four probes and
prints an `ok`/`BAD` flag; **0 of 456 timed rows came back `BAD`**.

**Warm, interleaved, and repeated.** `WARM_S=1.5` inside the harness (a cold
first run once fabricated a 3.7x regression here), 7 reps at n ≤ 512, 5 at 1024,
3 at 2048, median reported. A/B is interleaved per cell — `qrbench_v` then
`qrbench_nv` at the same n and batch, back to back — not all-of-A-then-all-of-B,
so drift lands on both arms of every ratio. **Nothing was timed under
`BATCHLAS_KERNEL_TRACE`.**

### The discard rule, and the three cells it discarded

A cell is dropped if EITHER arm has relative sd > 10%, if either arm's flag is
not `ok`, or if either arm's residual exceeds 1e-4 (float/cfloat) or 1e-11
(double/cdouble). Applied in `analyse.py` only, so it cannot be applied
inconsistently between tables.

**456 timed rows: 368 of them form 184 vendor/vendor-free A/B cells
(`order.csv` 72, `batch.csv` 56, `tall.csv` 32, `orgqr_batch.csv` 24), and the
remaining 88 are the single-arm tier sweep. Exactly three A/B cells were
discarded, all of them the VENDOR arm at tiny batch where its absolute time is
~1.2 ms:**

| cell | discarded because |
|---|---|
| `batch.csv` geqrf float n=64 b=128 | vendor rel sd = 13.7% |
| `batch.csv` geqrf cfloat n=64 b=128 | vendor rel sd = 12.1% |
| `batch.csv` geqrf cfloat n=64 b=32 | vendor rel sd = 12.0% |

Every other cell is under 6.1%, and 448 of 456 rows are under 2%.

### One methodological caveat that the tables carry rather than hide

In the ORDER sweep the batch schedule varies with n (memory-bounded: three
n²·batch arrays at cdouble n=2048 is already 6.4 GB). So an "order crossover"
read off that table is confounded with a batch change. `analyse.py` therefore
reports crossovers on the two axes SEPARATELY, each holding the other fixed, and
only three order-crossovers in `order.csv` are clean (same batch on both sides).
The clean statements about order come from `tier.csv` and from the fixed-n blocks
of `batch.csv`.

---

## 2. `geqrf` — the order sweep

`order.csv` / `order_summary.txt`. vendor ms → vendor-free ms (ratio; **bold** =
native ahead).

| n, batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 32, 8192 | 0.8 → 1.0 (0.78x) | 2.3 → 10.8 (0.21x) | 1.0 → 1.4 (0.71x) | 5.8 → 17.6 (0.33x) |
| 64, 8192 | 6.3 → 2.9 (**2.14x**) | 15.7 → 29.6 (0.53x) | 13.1 → 5.0 (**2.64x**) | 31.5 → 58.9 (0.54x) |
| 96, 4096 | 12.5 → 5.0 (**2.50x**) | 25.7 → 32.6 (0.79x) | 24.0 → 10.6 (**2.26x**) | 51.2 → 125.7 (0.41x) |
| 128, 4096 | 30.9 → 15.4 (**2.01x**) | 60.8 → 37.1 (**1.64x**) | 59.2 → 19.5 (**3.04x**) | 115.8 → 224.0 (0.52x) |
| 160, 2048 | 30.9 → 8.5 (**3.65x**) | 56.6 → 28.1 (**2.01x**) | 54.4 → 16.4 (**3.33x**) | 110.0 → 178.9 (0.61x) |
| 256, 2048 | 121.4 → 21.6 (**5.62x**) | 228.7 → 72.3 (**3.17x**) | 227.1 → 44.4 (**5.11x**) | 434.9 → 520.6 (0.84x) |
| 512, 512 | 370.9 → 30.5 (**12.18x**) | 685.8 → 101.5 (**6.76x**) | 560.8 → 63.2 (**8.87x**) | 1112.1 → 789.5 (**1.41x**) |
| 1024, 128 | 2112.0 → 51.4 (**41.08x**) | 4290.9 → 169.9 (**25.26x**) | 3428.6 → 108.9 (**31.49x**) | 5993.9 → 1392.2 (**4.31x**) |
| 2048, 32 | 21283.2 → 117.6 (**181.02x**) | 30529.4 → 359.3 (**84.98x**) | 24888.3 → 242.5 (**102.65x**) | 41947.2 → 2815.9 (**14.90x**) |

**Do not quote the right-hand column as "181x faster than cuBLAS".** At n ≥ 512
`cublas?geqrfBatched` is nowhere near saturated — its wall time is nearly
independent of batch (`batch.csv`, float n = 1024: 1204 ms at b = 8, 2276 ms at
b = 256, a 32x increase in work for 1.9x the time). The ms column IS a valid
absolute target at each stated cell, and every cell here is one a caller could
actually issue. The honest ceiling-to-ceiling statement is the throughput one:

| type | native geqrf, saturated | cuBLAS geqrfBatched ceiling | ratio at equal saturation |
|---|---|---|---|
| float | 3564 GFLOP/s (n=1024) | ~380–390 | **~9.2x** |
| double | 1079 GFLOP/s (n=1024) | ~200 | **~5.4x** |
| cfloat | 1683 GFLOP/s (n=1024) | ~205 | **~8.2x** |
| cdouble | 132 GFLOP/s (n=1024) | ~105–110 | **~1.2x** |

(cuBLAS ceilings from `experiments/wp5_qr/baseline/README.md` §3; native from the
`GFLOPs` column of `order.csv`, against 2mn² − 2n³/3.)

### The losses, stated plainly

`geqrf` native LOSES in 11 of 36 order cells:

* **cdouble at n ≤ 256** — 0.33x, 0.54x, 0.41x, 0.52x, 0.61x, 0.84x. The worst
  block in the grid.
* **double at n ≤ 96** — 0.21x, 0.53x, 0.79x.
* **float and cfloat at n = 32** — 0.78x, 0.71x.

Three separate mechanisms, and §4 and §6 separate them: the n = 32 cells are a
work-per-launch floor, the double/cdouble small-n cells are FP64 rate plus a
mis-set tier boundary, and cdouble everywhere is the transposed GEMM.

---

## 3. `geqrf` — the batch axis (the crossover is in BOTH, and they differ per type)

`batch.csv` / `batch_summary.txt`. WP4 found its potrf crossover was in ORDER and
not in BATCH, which is not the intuition; for geqrf **both axes carry a real
crossover, and for double and cdouble the batch axis is the DOMINANT one.**

### n = 64 (the CTA tier)

| batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 32 | 1.24 → 0.14 (8.67x) | 2.26 → 0.52 (4.37x) | discarded (sd 12%) | 4.66 → 0.92 (5.05x) |
| 128 | discarded (sd 14%) | 2.27 → 0.52 (4.38x) | discarded (sd 12%) | 4.63 → 0.92 (5.01x) |
| 512 | 1.24 → 0.21 (5.85x) | 2.20 → 1.90 (1.16x) | 1.40 → 0.39 (3.59x) | 4.56 → 3.67 (1.24x) |
| 2048 | 1.40 → 0.78 (1.79x) | 3.42 → 7.47 (**0.46x**) | 2.11 → 1.31 (1.61x) | 8.13 → 14.81 (**0.55x**) |
| 8192 | 6.24 → 2.95 (2.12x) | 15.87 → 29.71 (**0.53x**) | 13.15 → 4.98 (2.64x) | 31.53 → 59.17 (**0.53x**) |
| 16384 | 10.16 → 5.83 (1.74x) | 31.71 → 59.38 (**0.53x**) | 26.75 → 9.87 (2.71x) | 61.41 → 118.32 (**0.52x**) |

**double and cdouble at n = 64 cross from a 4–5x WIN to a 0.53x LOSS between
b = 128 and b = 2048, and then sit flat.** Both arms are launch-bound below
b ≈ 512 and linear above it, so the flat 0.53x from b = 2048 to b = 16384 is the
SATURATED ratio and the 4–5x at b = 32 is overhead being compared to overhead.
Per this repository's standing policy — batch = 1 is not a regime we optimise
for, and algorithms are compared only at saturation — **0.53x is the number, and
the 5x is not.**

### n = 256 and n = 1024 (the blocked tier)

| batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| n=256, 32 | 14.13x | 15.61x | 8.01x | 6.11x |
| n=256, 128 | 11.47x | 11.88x | 7.68x | 3.66x |
| n=256, 512 | 8.36x | 4.96x | 5.89x | 1.25x |
| n=256, 2048 | 5.62x | 3.17x | 5.14x | **0.84x** |
| n=1024, 8 | 79.63x | 100.84x | 72.46x | 27.31x |
| n=1024, 32 | 80.11x | 67.28x | 64.36x | 13.24x |
| n=1024, 128 | 41.26x | 25.14x | 31.51x | 4.31x |
| n=1024, 256 | 23.89x | 13.46x | 18.24x | — |

The ratio falls monotonically with batch at every (type, n) here, because the
vendor is still on its flat launch-bound plateau while native is already linear.
**Only cdouble actually crosses**, at n = 256 between b = 512 and b = 2048.

---

## 4. The native-internal tier boundary is a CAPACITY, and it is measurably wrong

`tier.csv` / `tier_summary.txt`, vendor-free build, `BATCHLAS_GEQRF_ROUTE=cta`
against `=blocked`. Ratio is blocked_ms / cta_ms, so **>1 means CTA is ahead**.
Cells whose `cta` pin did not take are excluded (20 of 44; see §1).

| type | n=48 | n=64 | n=80 | n=96 | n=112 | n=128 |
|---|---|---|---|---|---|---|
| float | 2.69 | 2.03 | 2.04 | 1.29 | **0.82** | **0.70** |
| double | 0.98 | **0.92** | **0.77** | **0.73** | pin n/a | pin n/a |
| cfloat | 3.17 | 2.09 | 1.25 | 1.08 | pin n/a | pin n/a |
| cdouble | 2.59 | 1.93 | pin n/a | pin n/a | pin n/a | pin n/a |

(n = 16 and n = 32 are NULL CELLS: at n ≤ nb the blocked driver is one panel on
the resident leaf, i.e. literally the CTA code. `analyse_tier.py` labels them
rather than reporting a 1.00x "tie".)

**Two findings, both actionable and neither currently encoded anywhere:**

1. **For float the tier ladder is inverted above n ≈ 104.** `supports()` admits
   CTA up to `m*n <= 24320`, i.e. up to n = 155 square, and with `preferred()`
   all-false a vendor-free `Origin::Auto` takes CTA first. At n = 128 that is
   **1.43x slower** than the blocked driver the same build already has
   (15.42 ms against 10.77 ms at b = 4096). The 2.01x vendor ratio in §2's
   n = 128 float row would be **2.87x** on the blocked tier.
2. **For double the CTA tier never wins at all once the blocked driver has more
   than one panel** — 0.92x at n = 64, 0.73x at n = 96. Its `nb` is 16, so it
   reaches multi-panel blocking sooner than the other three types.

Neither of these is a `supports()` question. Both are exactly the "fit judgement
between two native routes" that `route_potrf.hh:284-296` says belongs in
`preferred()`. **This report does not change `preferred()`.**

A secondary observation, reported as an observation and not as a mechanism: for
float the CTA tier has a **cliff between n = 96 and n = 112** — 5.01 ms → 12.00 ms
at b = 4096, a 2.4x time increase for 1.6x the flops. The resident tile grows
from 36,864 B to 50,176 B across that step, which is the region where two blocks
per SM stop fitting in the 100 KB shared-memory carveout. That arithmetic is
consistent with the cliff but **was not verified with an occupancy counter**.

---

## 5. The shape the library actually asks for: tall panels

`tall.csv` / `tall_summary.txt`. Every cell in §2 is square; the two in-tree
callers of `geqrf` are not (`band_reduction.cc:595`, `sytrd_sy2sb.cc:504` both
pass an m × r panel with r ≪ m). Ratios, vendor/native:

| m × n, batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 128×32, 4096 | **1.57** | 0.62 | **3.31** | 0.71 |
| 512×32, 2048 | **2.11** | **1.16** | **2.27** | **1.30** |
| 1024×32, 1024 | **2.26** | **1.73** | **1.25** | 0.85 |
| 2048×32, 512 | **1.99** | **2.02** | **1.05** | 0.73 |
| 512×64, 1024 | **2.80** | **2.09** | **2.45** | 0.73 |
| 1024×64, 512 | **4.52** | **3.68** | **2.21** | 0.82 |
| 2048×64, 256 | **7.58** | **6.32** | **3.28** | **1.28** |
| 1024×128, 256 | **10.85** | **8.30** | **7.27** | **1.67** |

geomean: float 3.34x, double 2.37x, cfloat 2.44x, **cdouble 0.96x**. Native wins
26 of 32 tall cells and is essentially at parity for cdouble. The margin grows
with m at fixed n, which is the direction the library's own callers move in.

---

## 6. `orgqr` — and why it is a different problem

### Order sweep (`order.csv`; n = 96 and n = 160 omitted here for width, both in
`order_summary.txt` and both consistent with their neighbours)

| n, batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 32, 8192 | **123.2x** | **114.1x** | **65.1x** | **12.4x** |
| 64, 8192 | **50.1x** | **60.2x** | **25.7x** | **10.5x** |
| 128, 4096 | **17.1x** | **30.8x** | **10.6x** | **6.0x** |
| 256, 2048 | **9.5x** | **15.3x** | **5.8x** | **3.4x** |
| 512, 512 | **3.8x** | **6.0x** | **2.3x** | **1.7x** |
| 1024, 128 | **1.26x** | **2.44x** | 0.82x | 0.78x |
| 2048, 32 | 0.41x | **1.34x** | 0.31x | 0.46x |

### Is the large-n loss a batch artefact? Partly — and only for float at n = 1024

`orgqr_batch.csv`. The vendor arm is a per-item loop, so its time is linear in
batch **by construction**; at b = 32 on a 128-SM card it has not yet paid for
serialisation. The baseline recorded that the analogous vendor-build loss WAS a
batch artefact, so the losing cells had to be re-measured before being called
losses:

| type, n | b=16 | b=32 | b=64 | b=128 | b=256 |
|---|---|---|---|---|---|
| float, 1024 | — | 0.84x | **1.11x** | **1.27x** | **1.33x** |
| double, 1024 | — | **2.10x** | **2.32x** | **2.43x** | **2.47x** |
| cfloat, 1024 | — | 0.55x | 0.70x | 0.82x | 0.88x |
| cdouble, 1024 | — | 0.69x | 0.75x | 0.78x | — |
| float, 2048 | 0.33x | 0.40x | 0.45x | 0.47x | — |
| cfloat, 2048 | 0.26x | 0.31x | 0.35x | — | — |
| cdouble, 2048 | 0.44x | 0.46x | — | — | — |

* **float n = 1024 flips at b ≥ 64 and is still rising at b = 256.** The order
  sweep's 1.26x at b = 128 reproduces here as 1.275x in an independent process.
* **cfloat n = 1024 is still climbing (0.55 → 0.88) and would plausibly cross
  somewhere past b = 512** — which does not fit in 24 GB at this order, so it is
  NOT claimed.
* **n = 2048 is a genuine loss for float, cfloat and cdouble** — climbing, but
  from 0.26–0.44x, and still 0.35–0.47x at the largest batch that fits.
* **double never loses at any (n, batch) measured.**

Verification that the vendor arm really is the per-item loop: float n = 1024
vendor times are 20.4 / 45.4 / 93.9 / 188.5 ms at b = 32 / 64 / 128 / 256 —
linear to within 4%. Native is 24.3 / 40.8 / 73.7 / 141.4, i.e. still saturating.

### `orgqr`'s other axis: memory

The vendor `orgqr` sizes its workspace as `single_ws * batch`
(`cublas.cc:1447`). Measured, at the cells in §6's order table:

| cell | vendor orgqr ws | native orgqr ws |
|---|---|---|
| cdouble n=64 b=8192 | **4870 MB** | 1476 MB |
| cdouble n=32 b=8192 | **4467 MB** | 671 MB |
| float n=32 b=8192 | 1120 MB | 168 MB |
| float n=2048 b=32 | 103 MB | 562 MB |

So the memory advantage reverses exactly where the speed advantage does. Native
orgqr is 3.3–6.7x cheaper in workspace at small n and large batch (where it is
also 12–123x faster) and 2.7–5.5x more expensive at n ≥ 1024 (where it loses).

---

## 7. Where the time goes — and the standing prediction is CONFIRMED

Full tables in `nsys_split.md`; captures are gitignored. `native:blocked` cells,
vendor-free build.

| | float n=1024 (**41x win**) | cdouble n=1024 (**4.3x win**) | cdouble n=256 (**0.84x LOSS**) | double n=64 (**0.53x LOSS**) |
|---|---|---|---|---|
| **transposed GEMM (Tiled16)** | **46.6%** | **69.7%** | **51.3%** | — |
| NN GEMM (Register128x128 / 64x64-wide) | 24.3% | 21.4% | 15.6% | — |
| WY construction (larft + pack_v) | 19.7% | 6.5% | 22.3% | — |
| **panel factorisation** | **9.3%** | **2.5%** | **10.6%** | **100.0%** |

The two losing cells are profiled deliberately: a split taken at a winning cell
does not explain a loss. `double n=64` is `native:cta`, which has no trailing
update at all — that loss is the panel kernel at FP64 rate and no GEMM change can
touch it. `cdouble n=256` is `native:blocked` and shows the SAME mechanism as the
winning n=1024 cell, with `larft` grown to 20.3%.

**The standing prediction (baseline finding G1) is confirmed by direct
observation, for float as well as for complex.** `route_gemm.hh` and
`gemm_kernels.cc:470-482` short-circuit every transposed form to
`max_dim <= 32 ? Direct : Tiled16` before the register ladder, and the panel
update's `W1 = Vᴴ A22` is exactly such a form. It is the single largest kernel in
a vendor-free blocked `geqrf` in **all three blocked cells profiled** — float
n=1024, cdouble n=1024 and cdouble n=256 — and for cdouble n=1024 it is more than
two thirds of the whole call. The NN update G3 does reach the good kernel
for both — `GemmRegister128x128Kernel` for float, `GemmRegister64x64K16WideKernel`
for cdouble — so G3 is not the problem, exactly as predicted.

Three consequences worth carrying:

1. **The "budget the effort on the panel" advice this phase inherited is wrong
   for the shipped driver.** The panel is 9.3% (float) / 2.5% (cdouble) of the
   call. The prediction that the trailing update was cheap was made against a
   *routed* trailing update; the shipped driver's trailing update lands on
   Tiled16 for its biggest GEMM and is 71% of the call.
2. **The cdouble deficit is not a WP5 problem and cannot be fixed inside WP5.**
   Closing it needs a transposed wide-scalar/register GEMM — WP2 territory. cdouble
   is the only type whose vendor-free `geqrf` is not comfortably ahead, and 69.7%
   of it is one kernel that WP5 does not own.
3. **`orgqr` has a different bottleneck at each end of the range**: `larft` is
   49.0% at float n = 64 b = 8192, and the transposed GEMM is 41.9% at
   float n = 1024. The identity fill plus the copy-back — the two kernels that
   exist only because orgqr is ormqr-on-an-identity — cost ~11% at n = 1024 and
   ~17-22% at n = 64 (the range spans the median-corrected and the raw
   figures; the FIRST launch of the identity fill carries a unified-memory
   first-touch cost of 32 ms against 3.5 ms thereafter, and is excluded from the
   median-corrected end). That is the entire measurable price of not specialising, and
   it is smaller than the 1.5x flop ratio the design note predicted.

---

## 8. Answers

**Is native QR ever faster than cuSOLVER?** Yes, and by large margins over most
of the useful range. `geqrf`: native wins 25 of 36 order cells, everything at
n ≥ 128 except cdouble below n = 512, up to 181x at the largest cell and ~9.2x
ceiling-to-ceiling for float. `orgqr`: native wins 31 of 36, everything at
n ≤ 512 for all four types, up to 123x.

**Where does it lose?**

| loss | scale | cause (measured) |
|---|---|---|
| `geqrf` cdouble, n ≤ 256 | 0.33–0.84x | at n=256, profiled: 51.3% transposed GEMM on Tiled16 + 20.3% `larft`, panel only 10.6% (§7). Not a WP5 defect. Below n=80 the route is `native:cta` and the split is 100% panel, i.e. FP64 rate. |
| `geqrf` double/cdouble, n = 64, saturated batch | 0.52–0.55x, flat | CTA tier at FP64 rate — profiled at double n=64 as **100.0% one kernel**, `GeqrfPanelResidentKernel<double>`. For double the blocked tier is already 1.09x faster at this cell (§4), so part of it is a tier-boundary error and not a kernel limit. |
| `geqrf` float n = 112 and n = 128 (CTA stays eligible to n = 155) | 1.22x and 1.43x below its OWN blocked tier | the capacity-based tier ladder (§4). Costs nothing against the vendor — n=128 is still 2.0x ahead — but it leaves 1.43x on the table. |
| `geqrf` all types, n = 32 | 0.21–0.78x | a work-per-launch floor; the vendor's batched kernel is at 463 GFLOP/s here and native at 362. |
| `orgqr` n = 2048, float/cfloat/cdouble | 0.31–0.47x | cuSOLVER's per-item `orgqr` is an excellent single-matrix kernel (6.4 TFLOP/s float) and at b ≤ 128 the loop has not yet serialised; native additionally does 1.5x the flops and pays Tiled16 for the biggest of them. |
| `orgqr` cfloat/cdouble n = 1024 | 0.78–0.88x at the largest batch that fits | same, thinner; cfloat is still climbing with batch. |

**Where does the time go?** §7. For `geqrf`, 71% trailing GEMM / 20% WY / 9%
panel at float n = 1024, 91% / 6.5% / 2.5% at cdouble n = 1024, and 67% / 22% / 11% at the losing cdouble
n = 256 cell. In the CTA tier it is 100% one kernel. For `orgqr`, 66% GEMM
/ 12% WY / 11% identity-and-copy-back (median-corrected) at float n = 1024, but 49% `larft` at
float n = 64.

**Anything a router should be told?** Three things, none of them applied here:
the float CTA/blocked boundary is at n ≈ 104 and not at the capacity (155); the
double CTA tier should probably never be preferred above n = 48; and the
saturated batch matters for double/cdouble at n = 64, where the sign of the
result depends on it.

---

## 9. Diff hygiene

`.gitignore` in this directory covers `qrbench_v`, `qrbench_nv`, `*.nsys-rep`,
`*.sqlite` and `*.qdstrm`; the whole `nsys/` subdirectory is additionally ignored
by the repository-root `.gitignore:112` (`experiments/**/nsys/`), which is why
the derived kernel tables are written to `kernsum/` and not left beside the
captures. Only `.cpp`, `.sh`, `.py`, `.md`, the CSVs and the
derived `*_summary.txt` and `kernsum/*_kern.txt` tables are tracked. No profiler
capture and no `BATCHLAS_KERNEL_TRACE` JSON is in this directory.
