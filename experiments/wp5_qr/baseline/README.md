# WP5 QR — the vendor baseline, and the two design questions settled by measurement

Everything here was produced on **GPU 1 of this box (2× RTX 4090, 128 SMs, sm_89)**, claimed
exclusively for every timed run through `gpu_guard.sh` (`/home/jonaslacour/BatchLAS/experiments/gpu_guard.sh`
— the path `scripts/gpu_guard.sh` does not exist in this worktree). Every timed run printed
`gpu_guard: GPU N exclusive for the whole run`; no run was kept that did not.

**No file under `src/`, `include/`, `tests/` or `benchmarks/` was modified.** Libraries under
test: `build/` (vendor present) and `build-novendor/` (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`),
both at `95c58a8`.

---

## The headline, in five sentences

1. **The vendor target is soft and it gets softer with `n`.** cuBLAS `geqrfBatched` saturates
   at ~380 GFLOP/s (float) and ~110 GFLOP/s (cdouble) *regardless of n*, and at n ≥ 512 its
   wall time is nearly independent of batch — 32 float QRs of 2048² take **21.4 s**, and 256
   of them take 23.2 s.
2. **The trailing update is not the problem — the panel is.** Measured over all 18 panels of a
   real N=1024 blocked geqrf, the two WY GEMMs cost **33.4 ms vendor-free** against cuSOLVER's
   **2109.8 ms** for the whole factorization. Even cdouble, the worst type, has 4.3× of
   headroom in the BLAS-3 core alone. Whatever WP5 builds, its **panel factorization** will
   decide the outcome.
3. **orgqr-via-ormqr is VIABLE and mostly a large win**, because vendor `orgqr` is not batched
   at all — `cublas.cc:1414` loops `cusolverDnXorgqr` once per batch item. Routed
   ormqr-on-identity is **111× faster** at n=64/batch 8192 and still 1.2–2.3× faster at
   n=1024; it loses only at n=2048, and even there float flips back to a **1.12× win** once
   batch reaches 128. The two Qs agree to 1.6e-06 (float) / 6.2e-15 (cdouble).
4. **The shipped `ORMQR_BLOCK_SIZE_*` table is wrong for three of four scalar types**, in the
   vendor build as well as the vendor-free one — it was tuned on CUDA/float only. The measured
   best widths are **16 or 32**, not the shipped 24/48/56, and the shipped value costs
   1.11–1.55×.
5. **The complex deficit is real, is concentrated in ONE gemm, and is now quantified.** The
   transposed panel gemm `W = Vᴴ A₂₂` lands on `Tiled16` for **every** scalar type in a
   vendor-free build and runs 3.1–5.0× slower than cuBLAS. The NN update `A₂₂ -= V W` is at
   parity — *provided the block width is ≥ 32*, below which complex loses its register kernel
   too.

---

## 1. What was built, and why not the existing benchmarks

| file | what it is |
|---|---|
| `wp5qr.cpp` | times the **public** `geqrf` / `orgqr` / `ormqr`, no forced routes, correctness checked in the same process. Modes `geqrf`, `orgqr`, `ormqrI`, `qcheck` (orgqr *and* ormqr-on-identity back to back, plus their elementwise agreement), `synthI` (ormqr-on-identity from synthetic reflectors, so it runs where there is no geqrf). |
| `gemmtrail.cpp` | times geqrf's two WY trailing GEMMs on **real sub-views** of an N×N parent at the parent `ld`, stride and batch. |
| `routeq_qr.cpp` | pure-host: asks `resolve_gemm_route` what those shapes resolve to, vendor-present and vendor-free. |

`benchmarks/geqrf_benchmark.cc` could not answer any of this: it is registered with
`BATCHLAS_REGISTER_BENCHMARK` (float/double only — **no complex**), it never checks the answer,
and its GFLOP count is `2mn² + (2/3)n³`, which has the **wrong sign** on the second term
(LAPACK's geqrf is `2mn² − (2/3)n³`). This directory uses the correct count throughout.
`benchmarks/gemm_benchmark` allocates operands at `ld == rows`, which is structurally
incapable of seeing the sub-view question.

Each program is built **twice** — `build_v.sh`/`build_nv.sh`, `build_gt_v.sh`/`build_gt_nv.sh` —
against `build/` and `build-novendor/`, so "vendor-free" means the **build** and not an
environment variable inside a build that still has cuSOLVER linked in.

### Command lines

```
bash build_v.sh ; bash build_nv.sh ; bash build_gt_v.sh ; bash build_gt_nv.sh
g++ -O1 -std=c++20 -I ../../../include -I ../../../build/include routeq_qr.cpp -o routeq_qr

bash run_break.sh  > break.csv        # anti-vacuity matrix (see §2)
bash sweep.sh      > sweep_raw.txt    # sections A-D, the main sweep
bash run_gemmtrail.sh > gemmtrail.csv # trailing GEMM pair, both builds, interleaved
bash run_panelsum.sh  > panelsum.csv  # every panel of an N=1024 geqrf
bash run_sat.sh    > sat.csv          # saturation ladder
bash run_sat2.sh   > sat2.csv         # extended saturation for the vendor geqrf
bash run_nb.sh     > nb.csv           # block-width probe, parts 1 and 2
bash run_nb2.sh    > nb2.csv          # wide nb + the vendor-build control
bash run_cross.sh  > cross.csv        # the one cell where ormqr-on-identity lost
bash run_variant_trace.sh > variants.csv   # KernelVariant, observed not reasoned
./routeq_qr        > routeq_qr.csv

python3 analyse.py > summary.txt ; python3 analyse_gt.py > summary_gt.txt
python3 analyse_ps.py > summary_ps.txt ; python3 analyse_nb2.py > summary_nb2.txt
```

### Measurement rules applied

* Warm-up runs until `WARM_S` seconds have elapsed (default 1.5 s) and is discarded — a cold
  run has fabricated a 3.7× result in this repository before.
* Median of `reps` (5 for n ≤ 512, 3 above), with mean and **relative sd printed in every row**.
* **Discard rule: any cell with rel_sd > 10% is dropped.** No cell in any file here exceeded
  it; the worst in the main sweep is 1.6% (`geqrf,float,64,8192`).
* A/B pairs are **interleaved** per cell (`run_gemmtrail.sh`, `run_panelsum.sh` run `_v` then
  `_nv` back to back; `qcheck` runs orgqr and ormqr-on-identity in one process).
* Nothing was timed under `BATCHLAS_KERNEL_TRACE`. `run_variant_trace.sh` runs untimed, writes
  its JSON to the scratch dir and deletes it.
* **Cold clocks, and why they did not leak in.** `gpu_guard` reports the SM clock it saw before
  each section started, and three of the four main sections began at **210 MHz** (`sweep_err.txt`)
  — the card was idle when the guard sampled. That is exactly the trap that once fabricated a
  3.7x result here. It is absorbed by the per-cell warm-up, which runs the op in a loop for
  `WARM_S` seconds and discards all of it before the first timed iteration; the evidence is the
  spread, which is under 2% in every cell of every file and under 0.2% in most. The single
  noisiest cell in the whole sweep (1.6%) is the very first one measured.
* Every route claim is **verified from the run itself**: `wp5qr.cpp` prints the resolved
  `Route` (`route=1:3` = `Origin::Native` / `Algorithm::Blocked`) and the block width actually
  used, so a silently-ignored pin cannot pass as a native run.

---

## 2. THE CHECKS CAN FAIL — and two of the breaks prove something else

`BREAK=<n>` damages the *checker's reference*. Control (`BREAK=0`) is green in every cell.
Full matrix in `break.csv`; worst residual per column, at n=96 batch=8:

| break | what it damages | float | double | cfloat | cdouble |
|---|---|---|---|---|---|
| 0 | control | 4.1e-07 | 1.6e-15 | 3.8e-07 | 1.5e-15 |
| **1** | drop the **last** reflector | **4.1e-07 (GREEN)** | **1.6e-15 (GREEN)** | 2.1e-02 RED | 2.1e-02 RED |
| 2 | reversed reflector order | 2.081 RED | 2.081 RED | 1.814 RED | 1.814 RED |
| 3 | drop Q's last column | 0.994 / 2.4e-02 RED | 0.994 / 2.4e-02 RED | 0.935 / 1.3e-02 RED | 0.935 / 1.3e-02 RED |
| **4** | conjugate `tau` | **4.1e-07 (GREEN)** | **1.6e-15 (GREEN)** | 0.758 RED | 0.758 RED |
| 5 | drop a **middle** reflector | 0.839 RED | 0.839 RED | 0.466 RED | 0.466 RED |

**BREAK 4 green for real types is the correct null result** — conjugation is the identity on a
real scalar.

**BREAK 1 green for real types is a finding, and it matters directly to WP5's guarding test.**
`SHOW_TAU` (in `break2.txt`) explains it:

```
SHOW_TAU float   n=96: |tau[k-1]|=0.000000e+00  |tau[k-2]|=1.098851e+00
SHOW_TAU cdouble n=96: |tau[k-1]|=1.553246e+00  |tau[k-2]|=1.587534e+00
```

On a **square** matrix the final reflector acts on a 1×1 trailing block. LAPACK's `larfg`
returns `tau = 0` there for a real scalar (`H_k = I`, nothing to do), but a *non-zero* `tau`
for a complex one, because it still has to rotate the diagonal of R onto the real axis.

> **Consequence for WP5.** A short-final-panel regression test written on a **square, real**
> matrix guards nothing — dropping the last reflector is a no-op by construction. That is
> precisely the shape class that produced the silent `sy2sb` stage-1 failure. The guarding test
> **must** use `m > n` (so the final reflector is non-trivial), or a middle panel, or complex.
> Add BREAK 5 as the standing check; it is red for all four types.

---

## 3. Q1 — THE VENDOR BASELINE (`summary.txt` §A, `sweep_raw.txt` section A)

`geqrf` through the public API, vendor build, no forced routes. Residual is
‖QRx − Ax‖∞/‖Ax‖∞ over 3 random probes × items {0, batch−1}. GFLOP/s uses `2mn² − 2n³/3`.

| type | n | batch | med ms | GFLOP/s | rel_sd | residual |
|---|---|---|---|---|---|---|
| float | 64 | 8192 | 6.185 | 462.9 | 0.016 | 2.50e-07 |
| float | 128 | 4096 | 30.894 | 370.7 | 0.005 | 2.98e-07 |
| float | 256 | 2048 | 121.638 | 376.6 | 0.001 | 3.86e-07 |
| float | 512 | 512 | 370.829 | 247.1 | 0.000 | 5.03e-07 |
| float | 1024 | 128 | 2109.813 | 86.9 | 0.000 | 6.39e-07 |
| float | 2048 | 32 | 21311.621 | 17.2 | 0.000 | 8.44e-07 |
| double | 64 | 8192 | 15.757 | 181.7 | 0.001 | 9.36e-16 |
| double | 128 | 4096 | 60.954 | 187.9 | 0.002 | 1.89e-15 |
| double | 256 | 2048 | 228.761 | 200.3 | 0.001 | 1.97e-15 |
| double | 512 | 512 | 685.625 | 133.6 | 0.000 | 4.15e-15 |
| double | 1024 | 128 | 4274.255 | 42.9 | 0.000 | 3.91e-15 |
| double | 2048 | 32 | 30496.316 | 12.0 | 0.000 | 6.68e-15 |
| cfloat | 64 | 8192 | 13.172 | 217.4 | 0.004 | 3.84e-07 |
| cfloat | 128 | 4096 | 59.182 | 193.5 | 0.002 | 5.08e-07 |
| cfloat | 256 | 2048 | 226.819 | 202.0 | 0.001 | 4.59e-07 |
| cfloat | 512 | 512 | 561.116 | 163.3 | 0.000 | 6.43e-07 |
| cfloat | 1024 | 128 | 3428.596 | 53.5 | 0.000 | 8.69e-07 |
| cfloat | 2048 | 32 | 24914.510 | 14.7 | 0.000 | 1.09e-06 |
| cdouble | 64 | 8192 | 31.514 | 90.9 | 0.001 | 1.13e-15 |
| cdouble | 128 | 4096 | 115.846 | 98.9 | 0.001 | 2.32e-15 |
| cdouble | 256 | 2048 | 435.714 | 105.1 | 0.001 | 2.06e-15 |
| cdouble | 512 | 512 | 1111.516 | 82.4 | 0.000 | 4.26e-15 |
| cdouble | 1024 | 128 | 5993.803 | 30.6 | 0.000 | 5.02e-15 |
| cdouble | 2048 | 32 | 41982.781 | 8.7 | 0.001 | 5.66e-15 |

### 3a. SATURATION — read the GFLOP/s column with this caveat, or misread it (`sat.csv`, `sat2.csv`)

The batch schedule above is **memory-bounded at large n**, not saturation-chosen, and cuBLAS
`geqrfBatched` is nowhere near saturated there. Its wall time is nearly **independent of batch**:

| type | n | batch | med ms | GFLOP/s |
|---|---|---|---|---|
| float | 2048 | 32 | 21361 | 17.2 |
| float | 2048 | 64 | 21715 | 33.8 |
| float | 2048 | 128 | 22255 | 65.9 |
| float | 2048 | 256 | 23151 | **126.7** |
| float | 512 | 128 | 277 | 82.7 |
| float | 512 | 2048 | 937 | **391.1** |
| cdouble | 512 | 128 | 885 | 25.9 |
| cdouble | 512 | 1024 | 1675 | **109.4** |

**cuBLAS `geqrfBatched`'s saturated ceiling is ~380–390 GFLOP/s for float and ~105–110 for
cdouble, and it is essentially the same at every n** — it is a small-matrix routine that
processes one column at a time, so it is latency-bound and the ceiling does not move.
`n=256` is the only n in the main table that reached that ceiling (376.9 at b=2048 vs 378.4
at b=4096 — plateaued).

> **How to use the table.** The **ms column is a valid absolute target** at each stated cell:
> WP5's geqrf will be run at the same (n, batch) and can be compared directly. The **GFLOP/s
> column at n ≥ 512 is not a statement about cuBLAS's ceiling**, and quoting "cuBLAS geqrf gets
> 17 GFLOP/s" would be wrong. Both readings matter and they say the same thing for WP5: at
> n ≥ 512 the vendor leaves an enormous amount of the card idle.

### 3b. cuSOLVER `orgqr` — and a workspace defect worth recording

`orgqr` for batch > 1 is **not batched**: `cublas.cc:1414-1419` opens an out-of-order
sub-queue and calls `cusolverDnXorgqr` once per batch member. Two consequences:

* **Time is linear in batch** with per-call latency, which is why it loses so heavily at large
  batch (§4).
* **Workspace is `single_ws × batch`** (`cublas.cc:1447`). Measured, at the cells in §4:
  **1164 MB** for float n=64 b=8192 and **4644 MB** for cdouble n=64 b=8192 — for a problem
  whose *data* is 268 MB. Routed ormqr-on-identity needs 104 MB and 416 MB for the same cells.
  A native `orgqr` closes a memory hazard, not only a speed gap.

---

## 4. Q2 — IS ORGQR-VIA-ORMQR VIABLE? **YES.** (`summary.txt` §B, `cross.csv`)

`ormqr` works vendor-free — measured here directly, not inherited: `synthI` runs in
`build-novendor/` for all four types with orthonormality residuals 1.6e-06 / 4.1e-15, while
`geqrf` and `orgqr` in the same build throw
`no route for geqrf<T> ... built without cuBLAS` (`smoke.txt`).

**Correctness first.** `qcheck` computes both Qs in one process and compares them elementwise
(`dQ`, worst over items 0 and batch−1). Q from `ormqr(F, I, Left, NoTrans)` **is** Q from
cuSOLVER `orgqr`:

| type | dQ range over all six n |
|---|---|
| float / cfloat | 6.9e-07 … 3.2e-06 |
| double / cdouble | 1.4e-15 … 6.2e-15 |

Both also pass an independent `‖QᴴQx − x‖` and `‖QRx − Ax‖` probe.

**Speed**, vendor build, one process per cell, orgqr and ormqr-on-identity back to back:

| type | n | batch | cuSOLVER orgqr ms | ormqr-on-I ms | **ratio** | orgqr ws MB | ormqr ws MB |
|---|---|---|---|---|---|---|---|
| float | 64 | 8192 | 450.19 | 4.03 | **111.8×** | 1164 | 104 |
| float | 128 | 4096 | 256.11 | 7.67 | **33.4×** | 1030 | 100 |
| float | 256 | 2048 | 307.73 | 19.51 | **15.8×** | 899 | 149 |
| float | 512 | 512 | 177.04 | 24.97 | **7.1×** | 417 | 149 |
| float | 1024 | 128 | 93.94 | 40.16 | **2.3×** | 200 | 86 |
| float | 2048 | 32 | 56.82 | 84.36 | 0.67× | 98 | 42 |
| double | 64 | 8192 | 787.47 | 12.16 | **64.8×** | 2324 | 208 |
| double | 256 | 2048 | 1236.48 | 96.33 | **12.8×** | 1797 | 297 |
| double | 1024 | 128 | 603.73 | 294.94 | **2.1×** | 400 | 171 |
| double | 2048 | 32 | 643.80 | 661.99 | 0.97× | 196 | 85 |
| cfloat | 64 | 8192 | 457.34 | 6.62 | **69.1×** | 2324 | 208 |
| cfloat | 256 | 2048 | 391.71 | 37.44 | **10.5×** | 1797 | 297 |
| cfloat | 1024 | 128 | 128.81 | 101.94 | **1.3×** | 400 | 171 |
| cfloat | 2048 | 32 | 93.12 | 216.42 | 0.43× | 196 | 85 |
| cdouble | 64 | 8192 | 1726.66 | 51.03 | **33.8×** | 4644 | 416 |
| cdouble | 256 | 2048 | 2417.15 | 469.03 | **5.2×** | 3593 | 594 |
| cdouble | 1024 | 128 | 1600.65 | 1378.66 | **1.2×** | 801 | 342 |
| cdouble | 2048 | 32 | 1845.57 | 2681.60 | 0.69× | 392 | 170 |

(Full grid, all 24 cells, in `summary.txt`.)

**Ratios above 1 are not a compliment to `ormqr`; they are the cost of the per-item loop.**
The ratio falls monotonically with n *and* with falling batch, exactly as a per-call-latency
model predicts.

### 4a. The one cell where it lost, re-measured (`cross.csv`)

n=2048/batch=32 has two confounds — an unsaturated batch and a block width the §5 ladder shows
is 1.24–1.39× off. Raising batch as far as memory allows:

| type | n | batch | orgqr ms | ormqr-on-I ms | ratio |
|---|---|---|---|---|---|
| float | 2048 | 32 | 56.67 | 84.21 | 0.67× |
| float | 2048 | 64 | 118.59 | 124.59 | 0.95× |
| float | 2048 | 128 | 233.37 | 207.51 | **1.12×** |
| cdouble | 2048 | 32, nb=56 | 1845.70 | 2681.45 | 0.69× |
| cdouble | 2048 | 32, **nb=32** | 1845.23 | 2122.13 | 0.87× |
| cfloat | 2048 | 64, nb=56 | 187.84 | 294.85 | 0.64× |

So the n=2048 loss **is partly a batch artefact for float** (it flips to a win at batch 128),
and is partly a block-width artefact for cdouble (nb=32 recovers 1.26× of it). It is a real
loss for complex at the batches that fit in 24 GB.

### 4b. VERDICT and the caveat WP5 must carry

> **Implement `orgqr` as `ormqr` on an identity.** It is correct to 1e-06/1e-15 against
> cuSOLVER, it is 1.2–111× faster over the whole small-and-medium range this library actually
> operates in, it needs 4–11× less workspace, and the code already exists and already routes
> to `Native:Blocked` by default. Specialise later, if at all.

Two things to record honestly:

1. **The identity does 1.5× the nominal work of a specialised `orgqr`** at m=n=k (`4mnk − 2nk²`
   = 2n³ against `4mnk − 2(m+n)k² + 4k³/3` = 4n³/3). That is the theoretical price of not
   exploiting the leading identity block, and it is what a specialised routine would win back —
   at most 1.5×, against a 2.3–111× margin over the vendor across most of the range.
2. **These ratios are from the VENDOR build.** In the vendor-free build ormqr-on-identity is
   0.88–2.39× slower (§6), so at n ≥ 1024 the vendor-free margin over cuSOLVER `orgqr` is
   thinner or negative. The fix for that is not in `orgqr` — it is the `Tiled16` transposed
   gemm of §6.

---

## 5. Q3 — THE PANEL BLOCK WIDTH (`summary_nb2.txt`, `nb.csv`, `nb2.csv`)

### 5a. What is shipped, and the header that is dead

`resolve_ormqr_block_size` (`ormqr.hh:220-228`) returns `tuning::ormqr_block_size_for_n(A.rows())`
when no hint is passed, which buckets to:

```
n <= 64 -> 16   n <= 128 -> 16   n <= 256 -> 24   n <= 512 -> 48   else 56
```

Note: those are the constants in **`include/batchlas/tuning_params.hh`** (16/16/24/48/56). The
`configure_file` copy in `build/include/batchlas/tuning_params.hh` says **16/32/64/128/128** and
is **never compiled** — `src/CMakeLists.txt:89-90` puts `${PROJECT_SOURCE_DIR}/include` *ahead
of* `${PROJECT_BINARY_DIR}/include`. This is deliberate and documented at
`cmake/BatchLASPackaging.cmake:78-87`; it is flagged here only so nobody reads the wrong file.
The harness prints the width it actually used, and it prints 16/24/48/56 — confirmed at runtime.

### 5b. Did the tuning grid cover all four scalar types? **No — float only.**

`evaluation/tuning/tune.py` takes a **single** `--type` for an entire run (`--type` is a required
scalar argument, `tune.py:494`), and every example in `evaluation/tuning/README.md` is
`--type float` (lines 18-20, 165, 176, 312-318, 362). The `ormqr_blocked` space
(`evaluation/tuning/spaces/default.json`) has no type axis at all. So the shipped buckets are a
CUDA/**float** optimum applied to all four types — the same defect already recorded for `syev`.

### 5c. Measured, not reasoned — and the shipped width is wrong for three of four types

End-to-end WY apply (`ormqr` on an identity), `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE` forced, median
ms, lower better. `]` = shipped, `*` = best.

**Vendor-free build (what WP5 must be fast in), n=1024 batch=64:**

| nb | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 8 | 84.99 | 219.11 | 193.69 | 2302.60 |
| 16 | 45.36 | **123.65*** | 101.53 | 1547.54 |
| 24 | 43.31 | 150.67 | 120.80 | 1818.16 |
| **32** | **36.82*** | 135.65 | **81.54*** | **1061.24*** |
| 48 | 40.09 | 155.28 | 96.67 | 1144.85 |
| 56 `]` | 45.48 | 174.82 | 113.18 | 1333.81 |
| 64 | 46.63 | 181.25 | 122.95 | 1243.43 |
| 96 | 62.46 | 231.67 | 192.70 | 1544.61 |
| 128 | 83.04 | 303.45 | 296.93 | 1936.75 |
| 192 | 132.13 | 481.79 | 583.70 | 2948.22 |

**cost of the shipped width: float 1.24× · double 1.41× · cfloat 1.39× · cdouble 1.26×**

At n=256 batch=512 (shipped width 24): best is 16/16/16/32, shipped costs
**float 1.11× · double 1.32× · cfloat 1.26× · cdouble 1.55×**.

**It is a TYPE problem, not merely a build problem.** The same ladder in the **vendor** build —
the configuration the table was tuned in — still shows the shipped width costing
**double 1.32–1.41× and cdouble 1.46–1.47×**, while float at n=256 is exactly 1.00× (that is
the cell it was tuned at). Full ladder in `summary_nb2.txt`.

### 5d. Recommendation for geqrf, and the mechanism behind it

> **Use nb = 32 for float / cfloat / cdouble and nb = 16 for double.** Do **not** inherit
> `ormqr_block_size_for_n`. Keep it a multiple of 16 and never below 32 for complex.

Three independent reasons, each measured:

* **Multiples of 16.** `m` of the transposed panel gemm `W = Vᴴ A₂₂` **is the block width**, and
  that gemm is `Tiled16` for every type (§6). 24 and 56 are not multiples of 16 and both sit in
  local minima-free troughs; 16 and 32 win almost everywhere.
* **≥ 32 for complex, mechanically.** `select_kernel_variant` gates the wide-scalar complex
  kernel on `min_dim >= 32` (`gemm_kernels.cc:700`). `min_dim` of the NN update `A₂₂ -= V W` is
  the block width. At nb=24 complex G3 falls to `Tiled16` and costs **1.72–2.30×**; at nb ≥ 32
  it reaches `Tiled64x64RegisterK16Wide` and is at **parity with cuBLAS**. Both halves observed
  in `variants.csv` (§6b) and timed in `summary_gt.txt`.
* **Not wider than 32, despite what the GEMMs alone say.** This is a *negative* result worth
  recording. Measuring the trailing GEMM pair *alone* (`nb.csv` PART 1) says wider is always
  better and shows a spectacular float cliff at nb=128 — effective throughput jumps
  6896 → **18906** GFLOP/s, because `m >= 128` finally admits the float TN register kernel
  `Tiled128x32RegisterK32TN` (`gemm_kernels.cc:472`). **End to end that cliff does not survive:**
  nb=128 is the *worst* width tested in both builds (83.0 ms vs 36.8 at nb=32). The panel
  factorization and `larft` cost, which PART 1 cannot see, dominate. **PART 1 is the wrong
  instrument on its own; do not tune a block width on the trailing update alone.**

---

## 6. Q4 — WHAT THE TRAILING UPDATE ROUTES TO (`routeq_qr.csv`, `variants.csv`, `summary_gt.txt`)

A blocked right-looking geqrf at panel `j0`, width `ib`, on an N×N parent (`m1 = N−j0`,
`n2 = N−j0−ib`) issues:

```
G1   W    = Vᴴ A₂₂      m=ib,  n=n2, k=m1   transA = Trans (real) / ConjTrans (complex)
G3   A₂₂ -= V W         m=m1,  n=n2, k=ib   NN, alpha=-1, beta=1
```

G1 and G3 have **identical flop counts**. Every operand of both is a sub-view at the parent
`ld`; the harness builds each one explicitly with the parent `ld`, stride and batch, never via
`operator()(Slice,Slice)`.

### 6a. Resolver Route (`routeq_qr.csv`, 96 rows)

| type | vendor-present | vendor-free |
|---|---|---|
| float | `Vendor:Auto` (all 24) | `Native:RegisterTiled` (all 24) |
| double | `Native:RegisterTiled` (18) / `Vendor:Auto` (6) | `Native:RegisterTiled` (all 24) |
| cfloat | `Vendor:Auto` (all 24) | `Native:RegisterTiled` (all 24) |
| cdouble | `Vendor:Auto` (all 24) | `Native:RegisterTiled` (all 24) |

The six `double` vendor rows are **exactly** the N=2048 cells, where batch=32 fails
`route_gemm.hh`'s `s.batch < 64` gate. Measured, that gate **costs** here: at N=2048 b=32 the
vendor build is **1.14–1.25× SLOWER** than the vendor-free build on the same shapes
(`summary_gt.txt`, double rows, free/vend 0.79–0.88). Small, but it is a real preferred()
window edge on a shape geqrf will issue.

### 6b. KernelVariant — OBSERVED, not reasoned (`variants.csv`)

`NOTRACE` = no SYCL kernel was emitted at all, i.e. the call went to cuBLAS.

| build | type | G1 (transposed) | G3 (NN) |
|---|---|---|---|
| vendor | float / cfloat / cdouble | NOTRACE (cuBLAS) | NOTRACE (cuBLAS) |
| vendor | double | `tiled16` | `tiled16` |
| **vendor-free** | **float** | **`tiled16`** | `register_128x128_k8` |
| **vendor-free** | **double** | **`tiled16`** | `tiled16` |
| **vendor-free** | **cfloat** | **`tiled16`** | nb=24 → `tiled16`; nb=56 → `register_64x64_k16_wide` |
| **vendor-free** | **cdouble** | **`tiled16`** | nb=24 → `tiled16`; nb=56 → `register_64x64_k16_wide` |

> **G1 is `Tiled16` for EVERY scalar type in a vendor-free build**, float included. The
> transposed short-circuit at `gemm_kernels.cc:470-482` returns `max_dim <= 32 ? Direct : Tiled16`
> for everything except the three **float** register-TN forms, and those need `m >= 128` — i.e.
> a block width of 128, which §5d measured to be the worst width end-to-end. Complex cannot
> reach them at any width: the gate at `:472` tests `transA == Transpose::Trans` and a complex
> panel update is `ConjTrans`. **There is no block width at which geqrf's transposed panel gemm
> reaches a register kernel and is also a good block width.**

### 6c. What that costs, measured (`summary_gt.txt`, 64 cells)

Vendor-free ÷ vendor, same shape, interleaved:

| type | G1 (transposed) | G3 (NN), nb ≥ 32 | G3 (NN), nb = 24 |
|---|---|---|---|
| float | **2.18 – 4.73×** | 0.98 – 1.08× | 0.98 – 0.99× |
| double | 1.00× (0.86× at b=32) | 1.00× (0.79× at b=32) | 1.00× |
| cfloat | **2.85 – 4.77×** | 0.98 – 1.01× | **1.72 – 1.87×** |
| cdouble | **2.35 – 3.14×** | 0.91 – 1.09× | **2.30×** |

`double` is 1.00× because **both builds run the identical native kernel** — the resolver hands
double to `Native:RegisterTiled` even with cuBLAS present. Separate processes, separate `.so`s,
agreeing to 0.03% (`panelsum.csv`, e.g. 11.8008 vs 11.8012 ms). That is the strongest internal
control in this directory: the harness is demonstrably not running one library twice.

### 6d. THE PREDICTION ON RECORD (`summary_ps.txt`)

Summing **both trailing GEMMs over all 18 panels** of a real N=1024, nb=56, batch=128 blocked
geqrf — the BLAS-3 lower bound for the driver, excluding the panel factorization:

| type | build | G1 ms | G3 ms | **sum ms** | eff GFLOP/s | cuSOLVER geqrf ms | headroom |
|---|---|---|---|---|---|---|---|
| float | vendor | 5.21 | 7.91 | 13.12 | 13928 | 2109.8 | 160.8× |
| float | **vendor-free** | 25.05 | 8.35 | **33.40** | 5471 | 2109.8 | **63.2×** |
| double | vendor | 75.90 | 70.33 | 146.23 | 1250 | 4274.3 | 29.2× |
| double | **vendor-free** | 75.90 | 70.33 | **146.23** | 1250 | 4274.3 | **29.2×** |
| cfloat | vendor | 10.66 | 15.89 | 26.55 | 27527 | 3428.6 | 129.1× |
| cfloat | **vendor-free** | 53.19 | 16.16 | **69.35** | 10538 | 3428.6 | **49.4×** |
| cdouble | vendor | 339.29 | 356.27 | 695.56 | 1051 | 5993.8 | 8.6× |
| cdouble | **vendor-free** | 1059.92 | 339.58 | **1399.50** | 522 | 5993.8 | **4.3×** |

**The complex deficit, stated as a prediction rather than discovered later:**

> A vendor-free blocked `geqrf` will pay **2.55× (float), 1.00× (double), 2.61× (cfloat),
> 2.01× (cdouble)** on its BLAS-3 core relative to the same driver in a vendor-present build.
> **Essentially all of it is G1**, the transposed panel gemm: G1 alone is 4.81× / 1.00× / 4.99×
> / 3.12×, while G3 is 1.06× / 1.00× / 1.02× / 0.95×.
>
> **This is not fixable inside WP5.** `route_gemm.hh:113-114` refuses complex outright, and
> `gemm_kernels.cc:470-482` returns `Tiled16` for every transposed form of every type before the
> register ladder is reached. Closing it needs a **transposed** wide-scalar / register kernel in
> WP2 territory. Record it and move on, exactly as WP4 did with the `ConjTrans` short-circuit.

**And the reason it does not block WP5:** even the worst cell — cdouble, vendor-free — leaves
**4.3× of headroom against the whole cuSOLVER call**, and float leaves 63×. The trailing update
is already far cheaper than the target. **WP5 will be decided by the panel factorization, not by
the trailing update, and effort should be budgeted accordingly.**

---

## 7. Every cell that disagrees with a conclusion above

* **cuSOLVER `orgqr` beats routed ormqr-on-identity at n=2048**: 0.67× (float b32), 0.97×
  (double b32), 0.43× (cfloat b32), 0.69× (cdouble b32). Partly a batch artefact for float
  (1.12× win at b=128) and partly a width artefact for cdouble (0.87× at nb=32), but the
  complex loss at n=2048 is real at every batch that fits.
* **The vendor build is SLOWER than the vendor-free build in 6 cells**: all double at N=2048
  batch=32 (free/vend 0.79–0.88), because `route_gemm.hh`'s `batch >= 64` gate hands those to
  cuBLAS. And ormqr-on-identity is *faster* vendor-free in exactly three cells: float n=64
  (0.96×), double n=64 (0.94×) and double n=2048 (0.88×).
* **BREAK 1 and BREAK 4 are green for the real types** (§2). Both are correct null results,
  but BREAK 1's is a trap for WP5's regression test and is called out as such.
* **PART 1 of the block-width probe contradicts PART 2** and is wrong (§5d): nb=128 is the best
  width for the trailing GEMMs alone and the worst end to end.
* `nb=16` beats `nb=32` for **double** at both n (1.10–1.32×) while `nb=32` wins for the other
  three — so the recommended width is genuinely type-dependent and a single bucket table cannot
  express it.
* `cfloat` at n=256 vendor-free is a near-tie between nb=16 (14.69 ms) and nb=32 (14.97 ms),
  2% apart; the "use 32" recommendation costs it that 2% and buys the `min_dim >= 32` gate.

## 8. Files

`sweep_raw.txt` · `sat.csv` · `sat2.csv` · `gemmtrail.csv` · `panelsum.csv` · `nb.csv` ·
`nb2.csv` · `cross.csv` · `break.csv` · `break2.txt` · `smoke.txt` · `variants.csv` ·
`routeq_qr.csv` — raw. `summary*.txt` — derived, regenerate with `analyse*.py`.
Binaries are behind `.gitignore`; no trace JSON is kept.
