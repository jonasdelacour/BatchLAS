# WP7 R4 -- the vendor `gemv` baseline

What this measures: the **shipped public `batchlas::gemv`** entry point
(`src/dispatch/entry_points/level3.cc:143`), linked against the vendor-present
build in `build/`. That entry point has no route resolution at all -- it is
`backend::gemv_vendor` or a throw -- so every number here is
`cublasXgemvStridedBatched` (batch > 1) reached through the real API, argument
checks, view plumbing and all.

## FINDING

**cuBLAS batched `gemv` is already at the DRAM roof for 102 of the 104 cells in
the main sweep. WP7 is a parity exercise -- with exactly one exception.**

The best cells in this sweep reach **950-1000 GB/s** of useful traffic on an
RTX 4090 (theoretical 1008 GB/s). That is the roof; treat 950 GB/s as
"achievable", not the 900 the task assumed. Across the whole DRAM-resident
ladder -- all four scalar types, both `transA`, square and non-square, `n` from
32 to 2048, `batch` from 64 to 65536 -- cuBLAS sits between **890 and 1000 GB/s**,
i.e. **94-105% of that roof**. There is nothing to win at those shapes, and
`NoTrans` (the structurally hard direction) is not measurably worse than `Trans`.
Two independent passes agree to `spread <= 1.01` on 98 of 104 cells.

### The one exception: `complex<double>` + `transA = Trans`

| | |
|---|---|
| region | `64 <= m <= 320`, `n >= 128`, working set above L2 |
| measured | **310-380 GB/s** |
| roof | 950 GB/s |
| **headroom** | **2.5-3.0x** |

Worst cells, both reproduced across two independent passes to four significant
figures (`vendor_baseline.csv` vs `vendor_baseline_p2.csv`):

| type | m | n | batch | transA | pass 1 | pass 2 | % of roof |
|---|---|---|---|---|---|---|---|
| cdouble | 256 | 256 | 1024 | Trans | 322.7 GB/s | 322.8 GB/s | 34% |
| cdouble | 64 | 2048 | 1024 | Trans | 374.3 GB/s | 379.6 GB/s | 39% |

It is exclusively this type and this `transA`. At **identical bytes and
identical (m, n)** (`close.csv`, tag `typecheck`):

| m x n, A = 1024 MB, Trans | float | double | cfloat | **cdouble** |
|---|---|---|---|---|
| 256 x 256 | 936 | 957 | 937 | **325** |
| 64 x 256 | 966 | 966 | 967 | **376** |

It is **not** an `ld` alignment effect (`refine.csv`, tag `ldtest`): padding
`ld` to 257, 264 or 320 at `256x256` leaves it at 307-325 GB/s. The control
(`ldctrl`) shows the `LD` knob works -- padding a *healthy* cell to `ld = 513`
costs it 922 -> 718 GB/s -- so "padding did nothing" is a result, not a broken
instrument.

It appears as soon as `A` leaves the L2 (`refine.csv` tag `batchdep`,
`close.csv` tag `batchdep2`), i.e. at every genuinely DRAM-bound size. One cell
on the boundary (`256x256`, batch 256, 256 MB) is bimodal: 914 GB/s at
`rel_sd = 0.21`.

**So: WP7's only measurable performance prize is a native
`complex<double>` `Trans` gemv for `64 <= m <= 320`, `n >= 128`. Everywhere
else, parity with cuBLAS is the entire goal, and the ceiling is DRAM.**

Caveat on the region's edges: the `m` boundary is not clean. At `n = 128` only
`m in {64, 80}` is slow (`m = 48` is 747, `m >= 96` is at the roof); at
`n = 256` the slow band runs `m = 64..320` and `m >= 384` is at the roof. See
`refine.csv` tag `mboundary` and the `cdouble transA=Trans` panel of `grid.txt`.

## Files

| file | what |
|---|---|
| `gemvbase.cpp` | the harness: one cell per process, prints one CSV row |
| `build.sh` | links it against the already-built `build/src/*.so` (no library rebuild) |
| `run.sh` | the main cell sweep |
| `vendor_baseline.csv`, `vendor_baseline_p2.csv` | the two independent passes |
| `analyse.py` -> `summary.txt` | per-transA tables + the headroom count |
| `crosspass.py` -> `crosspass.txt` | cross-pass medians; separates real slow paths from noise |
| `probe_slow.sh` -> `slowpath_probe.csv`, `grid.py` -> `grid.txt` | the (m, n) grid at a fixed 1 GB footprint |
| `probe_refine.sh` -> `refine.csv` | m boundary, `ld` sensitivity, batch dependence |
| `probe_close.sh` -> `close.csv` | type-specificity, and the batch cliff at a second shape |

`gemvbase_v` is a build artefact and is in `.gitignore`.

Column note: `vendor_baseline*.csv` and `slowpath_probe.csv` were written before
the `LD` knob existed and have 11 columns (no trailing `ld`); `refine.csv` and
`close.csv` have a leading `tag` column and a trailing `ld`. Every reader here
parses by header name, so the shapes mix safely -- but a hand-written `awk $11`
does not.

Across all 468 measured rows in all five CSVs, `relerr` is exactly 0 and no cell
failed. `Transpose::Trans` on a complex type is confirmed empirically to be a
plain transpose, not a conjugate one: the host reference does not conjugate and
matches to the last bit.

## Method

* `CUDA_VISIBLE_DEVICES=0`, one dedicated RTX 4090 (this box has two).
* Every cell: 1.0 s of untimed warm-up (JIT, clocks, and the first-touch
  migration of a multi-GB shared allocation), then 15 timed reps (11 in the
  probes), **median** reported. The main sweep was run twice, in two separate
  sessions; `crosspass.py` quotes both.
* `alpha = 1`, `beta = 0`, `inc = 1`, `ld = m` (overridable with `LD=`),
  `stride = ld*n`.
* **A correctness check runs in the same process**, against a host reference over
  items 0 and `batch-1`, so a fast wrong answer cannot enter the record. Every
  row of every file here reports `relerr = 0` exactly. Item 0 alone would be
  blind to a wrong per-item stride, which is why `batch-1` is also checked.
* Bytes counted `= (m*n + len_x + len_y) * sizeof(T) * batch`; `A` dominates.
  Only the `m*n` *live* elements count, so an `LD`-padded run is not credited
  with traffic it never made.

### The column `analyse.py` adds and the CSV cannot: footprint

`A` is re-read on every rep. On a 4090 the L2 is 72 MB, so a cell whose `A` fits
in L2 measures **L2 bandwidth, not DRAM**, and its "% of roof" goes over 100%
and means nothing -- `float 64x64 x4096` reads 3119 GB/s. `analyse.py` marks
those rows `L2` and excludes them from the headroom verdict. This is why the
sweep carries two extra "L2-escape" cells (`32x32 x 65536`, `64x64 x 16384`):
without them the entire small-`n` end of the ladder is float-L2-resident and
unreadable.

### Layout, and which transA is structurally hard

Column major, `A(i,j)` at `i + j*ld`.

* `Trans`: `y_j = sum_i A_ij x_i` -- contiguous **down a column**, coalesced, a
  natural sub-group reduction. The easy case.
* `NoTrans`: `y_i = sum_j A_ij x_j` -- for a fixed row the stride is `ld`. The
  hard case; it has to be tiled through shared memory to coalesce.

Note that this predicts the opposite of what was measured: the one slow region
cuBLAS has is in `Trans`, the *easy* direction, and `NoTrans` is at the roof
everywhere. Whatever the hard direction costs a hand-written kernel, cuBLAS has
already paid it.
