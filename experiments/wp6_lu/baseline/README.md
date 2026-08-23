# WP6 LU — the vendor baseline, and the three design questions settled by measurement

Everything here was produced on **GPU 1 of this box (2× RTX 4090, 128 SMs, sm_89)** except the
pivot and hole probes, which ran on **GPU 0** while GPU 1 was busy with the grid; every run is
pinned with `CUDA_VISIBLE_DEVICES` and no two timed runs shared a card. Nothing was ever timed
under `BATCHLAS_KERNEL_TRACE`.

**No file under `src/`, `include/`, `tests/` or `benchmarks/` was modified.** Libraries under
test: `build/` (vendor present) and `build-novendor/` (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`),
both at `95c58a8`.

---

## The headline, in six sentences

1. **cuBLAS `getrfBatched` is deeply unsaturated at n ≥ 1024, and the effect is far larger than
   WP5's.** float n=1024 costs **48.9 ms at batch 1 and 170.3 ms at batch 256** — 256× the work
   for 3.5× the time — and cdouble n=2048 costs **2575.9 ms at batch 1 and 2659.3 ms at batch
   64**, i.e. 64× the work for **1.03×** the time. Any wall-clock ratio taken against it at
   large n is mostly the vendor's launch behaviour and must be caveated.
2. **`getri` on the routed `trsm` is the right build, and it needs no permutation kernel at
   all.** Writing `P` straight into `C` instead of writing `I` and permuting it gives geomean
   **1.60×** over `cublas<t>getriBatched` (18 of 28 cells), rising to **74.9×** at float
   n=2048, with **zero** extra workspace beyond the caller's own `C` (the vendor needs 512 B–32 KB
   of info array, so neither side has a workspace hazard here — unlike WP5's `orgqr`).
3. **`getrs` on the routed `trsm` is the right build only for many right-hand sides.** At
   nrhs = 64 it is geomean **1.55×** (25/28 cells, worst 0.95×); at **nrhs = 1 it is geomean
   0.36× — a 2.8× LOSS, in 25 of 28 cells, as bad as 0.09×**. That is the single most important
   negative result in this directory.
4. **The row interchange is not free and it is the wrong shape.** A LAPACK-faithful `laswp`
   (one work-item per column, walking the interchange list) is **half the cost of a composed
   `getrs`** at n=128. Collapsing the list to a permutation and gathering once turns
   `getri`'s geomean from 0.97× to **1.60×** and `getrs`'s (nrhs=64) from 1.17× to **1.55×**.
5. **Partial pivoting costs 1.25–2.65× over the unpivoted lower bound, it is almost entirely the
   SEARCH and not the swap (swap alone is 1.00–1.20×), and it gets CHEAPER as n grows** — so the
   plan's fear that it "serialises the whole factorization at small n" is correct in direction
   and the number is 2.65× at n=16, 1.27× at n=152. It is flat in batch.
6. **The 48 KB launch hole is real, is deterministic, and fires ONLY on the
   `sycl::reduce_over_group` arm** — at exactly 49 152 declared bytes, 5/5 runs, at every
   work-group width from 32 to 512, while the identical kernel with an explicit SLM tree
   argmax at the identical byte count launches fine. WP6's pivot search must use the explicit
   tree, or apply `potrf_cta.cc:258-296`'s band-and-pad in all three places.

---

## 1. What was built

| file | what it is |
|---|---|
| `lubench.cpp` | times the **public** `getrf` / `getrs` / `getri`, and a routed-`trsm` composition of the last two, correctness checked in the same process against a host reference on the factorization identity. Modes `getrf`, `getrs`, `getrs_trsm`, `getri`, `getri_trsm`; `LASWP=gather` switches the composition's permutation strategy; `BREAK=…` corrupts one guarded property at a time. |
| `pivotcost.cpp` | standalone SYCL (links **no** BatchLAS library — there is no native `getrf` to call). One CTA-resident LU written four ways: `nopiv`, `swaponly`, `pivman`, `pivgrp`. `WG=` sets the work-group width, `PAD=` sets the total local-memory request to an exact byte count. |
| `routeq_lu.cpp` | asks the resolver **and** `select_kernel_variant` what a blocked LU's trailing GEMM and panel TRSM resolve to, per type, vendor-present and vendor-free. |

`lubench.cpp` and `routeq_lu.cpp` are each built twice — `build_v.sh` / `build_nv.sh` — against
`build/` and `build-novendor/`, so "vendor-free" means the **build** and not an environment
variable inside a build that still links cuBLAS. `build_pivot.sh` builds the standalone probe.

Command lines, verbatim:

```
bash build_v.sh                                   # -> lubench_v
bash build_v.sh  routeq_lu.cpp routeq_lu_v        # -> routeq_lu_v
bash build_nv.sh routeq_lu.cpp routeq_lu_nv       # -> routeq_lu_nv
bash build_pivot.sh                               # -> pivotcost

bash run_grid.sh  > grid.csv  2> grid_err.txt     # section 2, 3
bash run_sat.sh   > sat.csv   2> sat_err.txt      # section 2
GPU=0 bash run_pivot.sh > pivot.csv 2> pivot_err.txt   # section 4
GPU=0 bash run_wg.sh    > wg.csv    2> wg_err.txt      # section 4
GPU=0 bash run_hole.sh  > hole.csv  2> hole_err.txt    # section 5
GPU=0 bash run_hole2.sh > hole2.csv 2> hole2_err.txt   # section 5
bash run_break.sh > break.csv 2> break_err.txt    # section 6
CUDA_VISIBLE_DEVICES=0 ./routeq_lu_v  > routeq_lu_v.csv
CUDA_VISIBLE_DEVICES=0 ./routeq_lu_nv > routeq_lu_nv.csv

python3 analyse.py grid_norm.csv > summary.txt
python3 analyse_sat.py sat.csv   > summary_sat.txt
python3 analyse_pivot.py pivot.csv > summary_pivot.txt
```

**Discard rule:** a cell with relative sd > 10 % over its repetitions is discarded and named.
`analyse.py` discarded **none** of the 280 grid cells and flagged **none** BAD.
`analyse_pivot.py` discarded 5 of 174 pivot cells, all at n ≤ 32 where the kernel runs for tens
of microseconds; they are listed at the top of `summary_pivot.txt`. Warm-up is `WARM_S`
seconds of untimed repetitions (1.5 s for the grid, 1.0 s for the sweeps) before the first
timed one; medians are reported, means and relative sd are in every row.

**One harness bug is baked into `grid.csv` and is worth knowing.** `printf '-,'` is parsed by
bash as an *option*, so the vendor rows in `grid.csv` are missing their leading `laswp` column
and `grid_err.txt` is 224 lines of `printf: -,: invalid option`. The timings are unaffected;
`grid_norm.csv` is `grid.csv` with the column restored, and `run_grid.sh` now says
`printf '%s,' '-'`.

---

## 2. The vendor target

Public API, vendor-present build, saturating batch, `2n³/3` flops per item for `getrf`.
Residuals are `‖(P A) x − L(U x)‖∞ / ‖A x‖∞` against a host reference, on item 0 and the last
item. Full table in `summary.txt`; every cell also in `grid_norm.csv`.

| n | batch | float ms / GFLOP/s | double ms / GFLOP/s | cfloat ms / GFLOP/s | cdouble ms / GFLOP/s |
|---:|---:|---|---|---|---|
| 32 | 8192 | 0.159 / **1129** | 0.561 / 319 | 0.245 / 732 | 1.488 / 120 |
| 64 | 8192 | 2.629 / 545 | 3.284 / 436 | 2.787 / 514 | 9.418 / 152 |
| 128 | 4096 | 5.770 / 992 | 8.803 / 651 | 6.988 / 820 | 24.66 / 232 |
| 256 | 2048 | 31.64 / 724 | 46.21 / 496 | 39.60 / 578 | 84.51 / 271 |
| 512 | 512 | 49.73 / 921 | 59.53 / 770 | 76.82 / 596 | 163.0 / 281 |
| 1024 | 128 | 94.53 / 969 | 111.8 / 820 | 122.7 / 747 | 351.5 / 261 |
| 2048 | 32 | 519.2 / 353 | 587.9 / 312 | 564.0 / 325 | **2594 / 70.6** |

Two things to read off it. The **GFLOP/s is not monotone in n** — float peaks at 1129 (n=32),
collapses to 545 at n=64, recovers to 992 at n=128 and collapses again to 353 at n=2048 — which
is what a routine with a small-n special case and no large-n blocking looks like. And **cdouble
n=2048 at 70.6 GFLOP/s is the softest target in the whole table**, 5× below its own n=512.

### Saturation, established before any comparison exists

`run_sat.sh` sweeps batch at fixed n. The number that matters is **µs per batch item**; a
saturated routine's curve is flat, an unsaturated one's keeps falling. Full tables in
`summary_sat.txt`.

| | batch 1 | … | largest tested | verdict |
|---|---|---|---|---|
| `getrf` float n=64 | 81.2 µs/item | | 0.317 µs/item @ 16384 | **saturates** — flat within 5 % from batch 512 |
| `getrf` float n=1024 | 48 920 µs/item (48.9 ms wall) | | 665 µs/item @ 256 (170.3 ms wall) | **NOT saturated**: still falling 10 % from batch 128→256 |
| `getrf` float n=2048 | 363 045 µs/item (363.0 ms wall) | | 8 601 µs/item @ 64 (550.5 ms wall) | **NOT saturated**: 64× the work for 1.52× the time |
| `getrf` cdouble n=2048 | 2 575 885 µs/item (2575.9 ms wall) | | 41 551 µs/item @ 64 (2659.3 ms wall) | **NOT saturated at all**: 64× the work for **1.03×** the time |
| `getri` float n=64 | 91.4 µs/item | | 0.207 µs/item @ 16384 | saturates from batch ≈ 4096 |
| `getri` float n=256 | 2418.9 µs/item | | **13.85 µs/item @ 256**, then *worse*: 20.38 @ 2048 | saturates at 256 and **degrades above it** |

Two consequences the later tables must carry:

* **Every ratio at n ≥ 1024 flatters the composition**, exactly as WP5's `geqrf` ratios did.
  The 74.9× `getri` win at float n=2048 is not "74.9× faster than cuBLAS"; it is a comparison
  against a routine that is barely using the GPU at that batch.
* **The grid penalises the vendor at two cells.** `getri` float n=256 is best for cuBLAS at
  batch 256 (13.85 µs/item) and the grid measured it at batch 2048 (20.38 µs/item), so the
  vendor is 1.47× pessimistic there. Likewise `getrf` float n=1024 reaches 1076 GFLOP/s at
  batch 256 against the grid's 969 at batch 128.

---

## 3. Should `getrs` and `getri` be built on the routed `trsm`?

**`getri`: yes, and unconditionally above n ≈ 128. `getrs`: yes for many right-hand sides, no
for one.**

Both compositions are built from the **public, routed** `trsm` and a permutation kernel written
in the harness. Every row prints the route each `trsm` resolved to; in the vendor-present build
they are `native:cta` at n=32 and `native:blocked` at every larger n, for all four types — so
the composition is already entirely native even with cuBLAS linked, and the vendor-free build
resolves identically (`routeq_lu_v.csv` vs `routeq_lu_nv.csv`, `vendor_free` columns diff to
zero lines).

Composition, in full:

```
getrs:  laswp(X)                         then trsm(L, Left/Lower/Unit) then trsm(U, Left/Upper/NonUnit)
getri:  fill C with I, laswp(C)          then the same two trsms          ("list" strategy)
getri:  write P straight into C          then the same two trsms          ("gather" strategy)
getrs:  build perm once, gather X into S then the same two trsms on S     ("gather" strategy)
```

### Geomeans (composition ÷ cuBLAS, > 1 means the composition wins)

| op | permutation | nrhs | cells | geomean | wins | worst | best |
|---|---|---:|---:|---:|---:|---:|---:|
| `getri` | list | 1 | 28 | 0.97× | 10 | 0.15× | 27.4× |
| `getri` | **gather** | 1 | 28 | **1.60×** | 18 | 0.23× | 74.9× |
| `getrs` | list | 1 | 28 | **0.36×** | 3 | 0.09× | 1.15× |
| `getrs` | gather | 1 | 28 | **0.36×** | 3 | 0.09× | 1.15× |
| `getrs` | list | 64 | 28 | 1.17× | 20 | 0.87× | 2.22× |
| `getrs` | **gather** | 64 | 28 | **1.55×** | 25 | 0.95× | 3.74× |

### `getri`, gather strategy, cell by cell

| n (batch) | float | double | cfloat | cdouble |
|---:|---:|---:|---:|---:|
| 32 (8192) | 0.54× | 0.23× | 0.23× | 0.23× |
| 64 (8192) | 0.83× | 0.53× | 0.35× | 0.54× |
| 128 (4096) | **1.32×** | 0.90× | 1.06× | 0.89× |
| 256 (2048) | 3.89× | 1.16× | 2.05× | 1.04× |
| 512 (512) | 5.75× | 1.28× | 3.01× | 1.02× |
| 1024 (128) | 15.7× | 1.16× | 6.05× | 1.11× |
| 2048 (32) | 74.9× | 3.93× | 25.9× | 4.30× |

The crossover is n ≈ 128 for float and cfloat and n ≈ 256 for double and cdouble; below it the
vendor's `getriBatched` small-n path wins by up to 4×. **Every n ≥ 512 number is against an
unsaturated vendor** (section 2) and must not be quoted alone.

### `getrs` at nrhs = 1 — the loss, stated plainly

| n (batch) | float | double | cfloat | cdouble |
|---:|---:|---:|---:|---:|
| 32 (8192) | 0.20× | 0.19× | **0.10×** | **0.09×** |
| 128 (4096) | 0.41× | 0.23× | 0.34× | 0.14× |
| 512 (512) | 0.66× | 0.32× | 0.59× | 0.26× |
| 2048 (32) | 0.94× | 1.14× | 0.87× | 1.07× |

cuBLAS `getrsBatched` with one RHS is 3–11× faster than two routed `trsm` calls, and the gather
strategy does not help (0.36× either way) because at nrhs = 1 the permutation is a rounding
error and the loss is entirely in the triangular solves. The reason is structural: `trsm`'s
native blocked driver is built to amortise a panel over many columns, and one column gives it
nothing to amortise. **A native `getrs` must therefore either keep a separate narrow-RHS path
or ship route-neutral at small nrhs.** `inv.cc`, the only internal consumer, does not use
`getrs` at all, so nothing internal is exposed to this.

### Workspace

The WP5 hazard (vendor `orgqr` needing 4644 MB against the composition's 416 MB) **does not
recur here**, and the direction is mildly reversed:

| | vendor | composition (list) | composition (gather) |
|---|---|---|---|
| `getri`, n=2048, batch 32 | 512 B | **0 B** | 262 144 B (`int32[n]` per item) |
| `getrs`, n=2048, nrhs 64, batch 32 | 0 B | **0 B** | 67 371 008 B (an out-of-place RHS + `int32[n]` per item) |
| `getri`, n=32, batch 8192 | 32 768 B | 0 B | 1 048 576 B |

`getrf_vendor_buffer_size` and `getri_vendor_buffer_size` are just `allocation_size<int>(batch)`
— a per-item info array and nothing else (`cublas.cc:1516`, `:1552`) — so the vendor side is
never the hazard. The interesting asymmetry is inside the composition: **`getri`'s gather needs
only the `int32[n]` permutation because the permuted identity can be written directly, while
`getrs`'s gather needs a whole second RHS** (it cannot permute in place without a cycle
decomposition). A native `getrs` that wants the gather must either allocate that or do the
in-place cycle walk.

### The permutation kernel is half the op, and that is a design finding

`BREAK=laswp` removes the row interchange and produces a *wrong* answer, but its timing is a
clean cost decomposition:

| float, n=128, nrhs=128, batch=256 | with laswp | without | laswp share |
|---|---:|---:|---:|
| `getrs_trsm` | 0.4456 ms | 0.2252 ms | **49 %** |
| `getri_trsm` | 0.4580 ms | 0.2251 ms | **51 %** |

The cause is structural, not a bad kernel. LAPACK's `ipiv` is a **sequence of interchanges**,
so it must be walked in order: one work-item per column, walking k. In column-major that gives
each work-item a contiguous walk but puts consecutive work-items `ldb` apart, so every warp
access is 32 separate transactions. The list cannot be parallelised — but it can be
**collapsed**: applying the interchanges to an identity index array once yields a permutation,
and a gather under it puts consecutive work-items on consecutive *rows*, which in column-major
is contiguous. That is the whole difference between the "list" and "gather" rows above.

---

## 4. What does pivoting cost?

`pivotcost.cpp`, one CTA-resident LU per work-group, whole matrix in local memory at `ld = n|1`,
work-group width 256 unless stated. Four arms:

* `nopiv` — Doolittle, no search, no swap. **The lower bound, and not a usable algorithm**: on
  the row-permuted test matrix its residual is 1.5e-03 (float n=64) against the pivoted arms'
  4.6e-07. `swaponly`'s residual is meaningless by construction (a random pivot list); both are
  timing references only.
* `swaponly` — no search, swaps against a precomputed pivot list of the same length.
* `pivman` — work-group argmax by an explicit SLM tree.
* `pivgrp` — the same, with **two** `sycl::reduce_over_group` per column.

### By n, batch 4096, wg 256 (ratios are × the unpivoted lower bound)

| type | n | nopiv ms | swap/np | **pivman/np** | pivgrp/np | SLM nopiv | SLM pivman |
|---|---:|---:|---:|---:|---:|---:|---:|
| float | 16 | 0.0394 | 1.20 | **2.65** | 3.30 | 1 100 | 3 140 |
| float | 24 | 0.0674 | 1.17 | **2.42** | 3.05 | 2 412 | 4 452 |
| float | 48 | 0.2539 | 1.08 | **1.67** | 2.07 | 9 420 | 11 460 |
| float | 64 | 0.4839 | 1.08 | **1.52** | 1.72 | 16 652 | 18 692 |
| float | 96 | 1.9153 | 1.05 | **1.45** | 1.44 | 37 260 | 39 300 |
| float | 128 | 6.5424 | 1.02 | **1.35** | 1.38 | 66 060 | 68 100 |
| float | 152 | 10.2384 | 1.01 | **1.27** | 1.25 | 93 036 | 95 076 |
| double | 32 | 0.2329 | 1.04 | **1.69** | 4.25 | 8 464 | 11 524 |
| double | 64 | 1.6403 | 1.02 | **1.44** | 2.03 | 33 296 | 36 356 |
| double | 110 | 8.3023 | 1.03 | **1.35** | 1.49 | 97 696 | 100 756 |
| cfloat | 16 | 0.0424 | 1.18 | **2.65** | 3.29 | 2 188 | 4 228 |
| cfloat | 64 | 0.9096 | 1.09 | **1.66** | 1.62 | 33 292 | 35 332 |
| cfloat | 110 | 5.6251 | 1.02 | **1.31** | 1.29 | 97 692 | 99 732 |
| cdouble | 16 | 0.1534 | 1.03 | **1.52** | 3.47 | 4 368 | 7 428 |
| cdouble | 64 | 5.1024 | 1.04 | **1.32** | 1.46 | 66 576 | 69 636 |
| cdouble | 78 | 8.4937 | 1.03 | **LAUNCH_FAIL** | 1.34 | 98 608 | 101 668 |

Every cell, including `double n=16` and `cfloat n=32` (discarded, relative sd 0.30 and 0.15), is
in `pivot.csv` and `summary_pivot.txt`.

**Three readings.**

1. **The swap is nearly free (1.00–1.20×); the search is the entire cost.** Dividing the two
   columns, the argmax alone is 1.25–2.2×. Any effort spent making the row exchange clever is
   spent on the 3 % end of the problem.
2. **The cost falls with n and is flat in batch.** float n=64, wg 256, batch 128 → 16384:
   pivman/nopiv is 1.85, 1.72, 1.53, 1.57, 1.53, 1.52, 1.52, 1.52. So this is a per-matrix
   property, not an occupancy artefact, and it does not get worse at the batch sizes this
   library cares about.
3. **`reduce_over_group` is the wrong tool here, on speed as well as on the hole.** For float
   and cfloat it is a wash (0.87–1.25× of the explicit tree). For **double and cdouble it is
   1.5–4.7× worse** — `double n=16` is 7.07× the unpivoted bound against pivman's 2.00×.

**The argmax scratch is not free in capacity, and it cost one launch outright.** `pivman`'s
tree needs `wg × (sizeof(real) + sizeof(int))` bytes on top of the tile — 2 040 B at wg 256 for
float, 3 060 B for cdouble. At **cdouble n=78 that pushed the request to 101 668 B against this
device's 101 376 B hard cap and the launch failed** ("Excessive allocation of local memory on
the device"), where the identical shape without the scratch fits at 98 608 B. Whatever WP6
ships, the scratch must be counted in the `fits` predicate, in the `local_accessor` allocation
**and** in the capacity query — the same three places `geqrf_cta.cc` applies its pad.

**A second, separate cliff: blocks per SM.** `run_wg.sh` and the `hole` section show a hard step
whenever `slm + 1024` crosses 102 400/2 ≈ 50 688 B, i.e. when residency drops from 2 blocks per
SM to 1:

| float, batch 1024, wg 256 | n=109 | n=110 | n=111 | n=112 |
|---|---:|---:|---:|---:|
| `nopiv` SLM / ms | 47 536 / 0.643 | 48 852 / 0.660 | 49 296 / 0.674 | 50 636 / **1.091** |
| `pivman` SLM / ms | 49 576 / 0.891 | 50 892 / **1.543** | 51 336 / 1.565 | 52 676 / 1.595 |
| `pivgrp` SLM / ms | 47 536 / 0.887 | 48 852 / 0.905 | 49 296 / 0.922 | 50 636 / **1.547** |

`pivman`'s 2 040 B of scratch **moves the cliff down by two orders of n** (it falls at 110, not
112) and costs 1.73× exactly there. The work-group width matters more than anything else in
this probe — at n=128, `nopiv` is 39.7 ms at wg 32 and 4.77 ms at wg 512 (`wg.csv`) — so WP6
must tune wg per (type, n) and not inherit potrf's.

---

## 5. The 48 KB launch hole — reproduced, and attributed

The `n` ladder in `run_pivot.sh` found **nothing**: every point from 43 692 to 56 340 B launched.
That is not evidence the class is gone, because **the hole WP4 recorded is not a range but
specific byte counts** (48 896 passes, 49 152 FAILS, 49 664 passes), and an `n` ladder steps over
49 152 rather than landing on it. `PAD=` holds the kernel, shape and work-group fixed and moves
only the declared byte count. One process per point, ascending, because the SLM attribute is
sticky per `CUfunction`.

`hole.csv`, float n=64, batch 1024, wg 256:

| declared bytes | `nopiv` | `pivman` (explicit tree) | `pivgrp` (`reduce_over_group`) |
|---:|---|---|---|
| 48 640 | ok | ok | ok |
| 48 896 | ok | ok | ok |
| **49 152** | ok | ok | **LAUNCH_FAIL: unknown internal error** |
| 49 408 | ok | ok | ok |
| 49 664 | ok | ok | ok |

`hole2.csv` brackets it:

* **Deterministic**: 5 separate processes at 49 152, 5 failures.
* **128 B either side is fine**: 49 024 ok, 49 152 FAIL, 49 280 ok.
* **Independent of work-group width**: wg 32, 64, 128, 256, 512 all fail at 49 152.
* **The control passes**: `pivman` at the identical 49 152 bytes launches at all five widths.
* **The band is wider for wide scalars**: `pivgrp` at 48 896 is *fine* for float and cfloat but
  **fails for double and cdouble**; 49 664 is fine for all four.

That is the WP4 signature reproduced exactly, at byte granularity, and now **attributed**: it is
the group collective, not the tile. The mechanism is that `reduce_over_group` allocates local
memory the `local_accessor` accounting cannot see, so the effective footprint is the declared
one plus an invisible, `sizeof(T)`-dependent amount.

**Recommendation for WP6.** Use the explicit SLM tree argmax. It is faster for double and
cdouble, it is a wash for float and cfloat, and it sidesteps the hole entirely — at the cost of
`wg × (sizeof(real) + sizeof(int))` bytes that must be counted in all three places. If a future
kernel does want a group collective, `potrf_cta.cc:258-296`'s band-and-pad is mandatory, and
the band must be widened for 8- and 16-byte scalars on this evidence.

---

## 6. What routes where, for LU shapes

`routeq_lu.cpp` asks both halves: the `Route` (a `RouteTable` decision on plain metadata) and
the `KernelVariant` (the selector inside the native GEMM, which reads pointers, leading
dimensions, stride **and batch**). A right-looking blocked `getrf` at panel start `j0`, width
`nb`, on an N×N parent issues `G: A22 -= L21·U12` (NN, m = N−j0−nb, n = N−j0−nb, **k = nb**) and
`T: U12 = L11⁻¹A12` (`trsm` Left/Lower/Unit, order nb, nrhs N−j0−nb).

`routeq_lu_v.csv` and `routeq_lu_nv.csv` are the two builds; their `vendor_free` rows **diff to
zero lines**, which is the check that the resolver is not reading anything build-dependent.

### The trailing GEMM, vendor-free (what a vendor-free `getrf` would actually run)

| type | verdict |
|---|---|
| float | `Native:RegisterTiled` → `Tiled128x128RegisterK8` at m,n ≥ 128; `Tiled32x32Register` on the k=nb=64 tail; `Direct` at N=128 |
| **double** | `Native:RegisterTiled` → **`Tiled16` at EVERY cell, all 13 shapes** |
| cfloat | `Tiled64x64RegisterK16Wide` at every cell except the tail panel N=2048/j0=1920/batch=32, where m=n=k=64 gives only 32 CTAs and the gate refuses (`Direct`) |
| cdouble | `Tiled64x64RegisterK16Wide` at every cell except the tail panel N=2048/j0=1920/batch=32, where m=n=k=64 gives only 32 CTAs and the gate refuses (`Direct`) |

**This inverts the prediction in the brief.** Complex does *not* land on `Tiled16`: WP4's
CTA-count relaxation (`gemm_kernels.cc:695-707`) admits it, because at these batches
`ceil(m/64)·ceil(n/64)·batch` is in the thousands. **`double` is the type stuck on `Tiled16`**,
for a structural reason: that relaxation is `if constexpr (is_std_complex_v<T>)` — complex only
— and the only other wide-scalar door, at `:642`, needs `min_dim >= 256`, which `k = nb = 32`
or `64` can never satisfy. So the LU trailing update is the *exact* population the
"GEMM demand is panel updates" note describes, and for `double` there is no register kernel on
that path at any problem size.

The deficit is bounded rather than catastrophic: the wide-scalar measurement in
`gemm_kernels.cc:606-616` puts `double` at only 1.01–1.08× of `Tiled16` and `Tiled16` at ~92 %
of the 4090's 1.44 TFLOP/s FP64 ceiling. **Record it as a prediction: a vendor-free blocked
`double` `getrf` will be bound by `Tiled16` in its trailing update and there is no WP6-local
fix.** The fix is a transposed/predicated wide-scalar kernel and belongs to GEMM, not here.

### An artefact worth recording, because it produced the predicted answer

The first version of `routeq_lu.cpp` used **batch = 1** parent matrices, on the reasoning that
"the variant selector reads m, n, k and the transposes". It reported **`Tiled16` for every
complex trailing update** — exactly the answer the brief predicted, and wrong. The CTA-count
gate multiplies by `A.batch_size()`, and `can_use_64x64_k16_wide_fast_path` also reads
`stride()`. A harness that saves memory by shrinking the batch cannot ask this question.

### Vendor-present, and the `trsm`

With cuBLAS linked, the trailing GEMM takes **`Native:RegisterTiled` for `double` at every N ≤
1024** (and `Vendor` at N=2048/batch=32) and `Vendor` for float, cfloat and cdouble at almost
every cell — consistent with the recorded finding that cuBLAS is weak at small batched DGEMM.
The panel `trsm` and both solve `trsm`s resolve **`Native` in both builds, for every type, every
N, and both nrhs = 1 and nrhs = N**. The algorithm splits on the TRIANGULAR ORDER, not on the
op: the panel `trsm` is `Native:CTA` wherever `nb = 32` (N = 128 and 256) and `Native:Blocked`
at `nb = 64`, and the two full-order solves are `Native:Blocked` at every N in this table
(N >= 128) — the grid's own n=32 rows are where `Native:CTA` appears for the solves. So a
blocked `getrf` built on the routed `trsm` gets the native kernel whether or not a vendor is
present, and WP3's window is the one WP6 inherits.

---

## 7. Every deliberate break, and its outcome

`run_break.sh`, float and cdouble, n=128, nrhs=128, batch=256. Two of these turned nothing red
on the first attempt, and both were real findings.

| break | what it corrupts | outcome |
|---|---|---|
| `BREAK=piv` | the `getrf` probe ignores the pivot list (P := I) | **first attempt: NOTHING RED** — residual bit-identical at 2.446e-07 / 1.055e-15. See below. After the fix: 2.4e-07 → **1.903e+00** (float), 1.485e-15 → **1.674e+00** (cdouble). |
| `BREAK=laswp` | the composition drops its row interchange | **first attempt: NOTHING RED** — same cause. After the fix: 3.4e-07 → **1.989e+00** (`getrs_trsm`), 3.1e-07 → **1.945e+00** (`getri_trsm`). |
| `BREAK=factor` | zeroes the strict lower triangle of the returned factor | RED: 2.446e-07 → **7.415e-02** (float), 1.485e-15 → **7.812e-02** (cdouble). |
| `BREAK=sol` | scales the solution by 1.01 | RED on all three solve probes: → **1.000e-02** exactly, for `getrs`, `getrs_trsm`, `getri` and `getri_trsm`. |

### The two breaks that turned nothing red — the finding

The harness originally generated a **diagonally dominant** test matrix (`A = rand + n·I`), the
obvious choice for "well conditioned so the residual measures the kernel". On such a matrix
**partial pivoting selects the diagonal at every step**, so `ipiv` is the identity, no row is
ever exchanged, and the entire pivot path — the vendor's, the probe's, and the composition's
`laswp` — is unexercised. Both breaks were therefore no-ops, and *every* residual in the
directory would have been blind to pivoting.

The fix is one line of setup, not one line of assertion: keep the dominance (it is what makes
the residual measure the kernel) and then **row-permute each item by a per-item random
permutation**. Conditioning is unchanged; partial pivoting must now undo the permutation, so
`ipiv` is non-trivial by construction. Both breaks go red immediately.

An **anti-vacuity assertion on the configuration** was added alongside: every row prints
`ntpiv`, the number of item-0 pivots that are not the diagonal, and the row is flagged BAD if it
is zero. It reads 29/32, 119/128, 1019/1024, 2041/2048 across the grid. That assertion is
*necessary and not sufficient* — it would have caught this particular case, but it says nothing
about whether the probe *uses* the pivots, which is what `BREAK=piv` is for.

**This is a warning for WP6's tests.** `tests/inverse_tests.cc` is one test, float, n=40,
batch=2, on `Matrix<float>::Random()`; if that generator is diagonally dominant or otherwise
pivot-free, the only `getri` test in the tree cannot see a pivot bug either.

---

## 8. What this directory does NOT settle

* **Nothing here is a native `getrf`.** `pivotcost.cpp` is a probe: it has no blocked tier, no
  `info` output, no route table, and its unpivoted arm is numerically unusable. The 1.25–2.65×
  pivot figure is a property of *that* CTA kernel, and a blocked driver's panel would pay it
  only on the panel.
* **The `getrs`/`getri` compositions use a harness-local permutation kernel**, which is exactly
  the "harness re-implements the driver" hazard the campaign warns about — unavoidable here
  because no native `getrs` exists to time. When one does, re-time it; the numbers above are a
  *bound on what a composition can achieve*, not a measurement of a shipped driver.
* **The vendor-free build cannot run `getrf`/`getrs`/`getri` at all**, so every timing in
  sections 2–3 is from `build/`. `build_nv.sh` exists and `routeq_lu_nv` was built and run with
  it, which is what establishes that the route columns agree.
* **`getrs` with `transA = Trans/ConjTrans` was not measured.** The composition would need the
  two solves in the opposite order with the permutation applied last, in reverse, and that is a
  different kernel.
* **The n=2048 and cdouble cells are against an unsaturated vendor.** Stated in section 2 and
  repeated here because it is the single easiest number in this directory to quote wrongly.
