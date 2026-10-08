# TRSM {#perf_trsm}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · step-9 to step-16 grids 2026-08-19/20 · tables 2026-10-04 (sm_89), 2026-10-05 (sm_120)

Native batched `trsm` on the GPU: the `cta` (V1), `blocked` (V2) and `sg_left` kernels, the routing
tables that choose among them, and the measurements behind those tables. The page gives the
decisions, the numbers they rest on, and the rejected alternatives.

> **Warning:** the grids below were measured on a kernel that returned wrong answers until the group
> barrier was added on 2026-08-21. Read [open debts](#trsm-open-debts) before trusting any ratio here.

## Choices and routing {#choices-flat-selection-p33}

Routing is in `src/ops/trsm/trsm.cc`. The choice vocabulary is `src/ops/trsm/choice.hh:14-19`, in
tie-break order. `can_run` (`trsm.cc:45-61`) is correctness only. Every native family also requires
`d.is_gpu && !A.is_heterogeneous() && !B.is_heterogeneous() && order >= 1 && q >= 1 && A.batch_size() >= 1`
(`trsm.cc:47-48`).

| spelling | implementation | `can_run` beyond the common term |
|---|---|---|
| `cta` | `sycl_trsm::trsm_native_v1_dispatch` (V1) | `max_wg >= 32`, order <= `trsm_cta_max_n<T>()` (32) (`trsm.cc:49-53`) |
| `sg_left` | `sycl_trsm::trsm_native_sg_left_dispatch` ([sub-group Left kernel](blackwell.md#trsm-sub-group-left-kernel)) | `Side::Left`, `has_sg32`, order <= `kTrsmSgLeftMaxN` (32), `max_wg >= kTrsmSgLeftWgSize` (128) (`trsm.cc:54-57`) |
| `blocked` | `sycl_trsm::trsm_native_blocked` (V2) with the public `gemm` injected | `max_wg >= 32`, `trsm_blocked_available<T>()`, `trsm_cta_max_n<T>() >= 1` (`trsm.cc:58`) |
| `vendor` | `backend::trsm_vendor` | a level-3 vendor library is linked (`d.has_vendor`, `trsm.cc:59`) |

Selection takes the first runnable entry of the nearest row of `tuned/trsm.<dtype>.<device>.txt`
(@ref selection_tables). The key is `side:exact trans:exact order:log:2 q:log batch:log`
(`choice.hh:26-27`). `q` is `B.cols()` for `Side::Left` and `B.rows()` for `Side::Right`. ConjTrans
folds to `T`. uplo and diag are not keys. The last resort is `blocked`, then `vendor`. The CPU has no
native family. trsm takes no workspace. `A.batch_size() != B.batch_size()` throws `invalid_argument`
in `trsm_validate_params`.

### What the tables say

Grid: batch from 128 up (`choice.hh:31-34`). Provenance is in `tuned/README.md`.

| table | source | first entry |
|---|---|---|
| `trsm.<all four>.sm_89.txt` | transcribed, untimed (`8b9adeb3`) | `cta \| blocked \| vendor` on all 1920 rows at order <= 32; `blocked \| vendor` on all 2400 rows above. `sg_left` in no row |
| `trsm.{cfloat,cdouble}.sm_120.txt` | the sm_89 transcription, relabelled | the same |
| `trsm.float.sm_120.txt` | measured (`benchmarks/results/tuning/trsm.float.sm_120.jsonl`) | order <= 32: Left `sg_left` 876 / `cta` 310 rows, Right `cta` 921 / `vendor` 92; above 32: Left `blocked` 601 / `vendor` 538, Right `blocked` 645 / `vendor` 469 |
| `trsm.double.sm_120.txt` | measured (`trsm.double.sm_120.jsonl`) | order <= 32: Left `sg_left` 587 / `cta` 523 / `vendor` 15 / `blocked` 1, Right `cta` 898 / `vendor` 95; above 32: Left `blocked` 874 / `vendor` 144, Right `blocked` 905 / `vendor` 56 |

On sm_89, `sg_left` runs only when pinned. The vendor runs only where no native family can: a
heterogeneous batch, or a CPU queue. The sm_120 result is on the
[Blackwell page](blackwell.md#trsm-on-sm_120-result).

The removed router (`route_trsm.hh`, deleted in flat selection P3.3) chose native for every type,
side and order at batch >= 8, except `float` with `Side::Right` at batch 8-127 above order 32. Those
calls lie below the table's smallest batch. See [the batch floor](#the-batch-floor).

### The shape builder and the field mapping {#the-shape-builder-and-the-field-mapping}

* **Key.** `key_of` (`trsm.cc:36-39`) builds `side`, `trans` (ConjTrans folds to `T`, since the
  kernels differ only in a conjugation), `order` = `A.rows()`, `q` = `rhs_count` and `batch` =
  `A.batch_size()`.
* **`can_run`** reads the views directly and never dereferences `data_ptr()`. Device facts (`is_gpu`,
  `has_sg32`, `max_wg`, `has_vendor`) come from the memoized `select::describe`.
* **Coverage row.** `trsm.cc:96-97` maps `m = B.rows()`, `n = B.cols()`, `k` = triangular order, plus
  transA, uplo, side and diag. `select::run` fills in the backend that actually ran.

### Tuning knobs and environment {#tuning-knobs-and-environment}

* `BATCHLAS_TRSM_ROUTE` takes `auto`, `native`, `vendor`, or a spelling (`cta`, `sg_left`,
  `blocked`). `native` takes the first non-vendor runnable entry (sm_89: `cta` at order <= 32,
  `blocked` above). A spelling the shape cannot run (`cta` or `sg_left` above order 32, `sg_left` on
  `Side::Right`) and an unknown word throw `invalid_argument`. `native` and `vendor` fall back to
  Auto with a warning when nothing of their class can run. The old `native:cta` and `native:blocked`
  aliases are gone.
* `BATCHLAS_TRSM_VARIANT` is read by nothing. Setting it pins nothing.
* `BATCHLAS_TRSM_OUTER_NB` is V2's outer block width, a tuning knob and not a selection one
  (`trsm_outer_block`, `trsm_native.cc`). The default is 128 for `Side::Left` and `cta_max_n` (32) for
  `Side::Right`, rounded down to whole CTA blocks. A value of 0 means unset. It is read per call and
  never latched. A function-local static would hide a later change, so an A/B harness or a
  `ScopedEnvVar` test would silently measure one arm twice.

---

## Design: V1, V2 and the canonical fold {#design-v1-v2-and-the-canonical-fold}

**V1** (`cta`): one work-group per matrix and rhs slice, one work-item per independent solve. The
solution vector lives in registers as `T x[N]`. The canonical triangle is staged once into SLM and
broadcast. The 24 canonical `(side, uplo, transA, diag)` cases fold into one recurrence through
`canonicalise()` (`src/sycl/trsm_canonical.hh:19`, shared with `sg_left`). The index map is
`rho(s) = fwd ? s : order-1-s`.

**V2** (`blocked`): a host-side two-level driver. The outer level blocks at `OUTER_NB` and issues a
trailing GEMM. Each outer panel is solved by the inner `nb = cta_max_n = 32` loop, which is V1, against
its own shorter prefix. The crossover is a capacity limit, not a tuned choice.

The grid is `batch * ceil(q/WG)`, never batch alone. The work-group ladder walks `{256,128,64,32}` and
takes the first `cand` with `bs*ceil(q/cand) >= 4*CU` (`trsm_v1_ladder_wg`, `src/sycl/trsm_native.hh:36-46`).
On sm_120 it also skips a rung that leaves over half its lanes without an rhs column
([ladder cap](blackwell.md#trsm-v1-ladder-cap)).

The work-group size cannot exceed 256. The worst instantiation, `complex<double>` at N=32, uses 226
registers: `226*256 = 57,856` of 65,536 per block, the whole-block count. The enforced check is the
per-sub-partition bound `ceil(warps/4)*32*regs <= 16384`, `resident::sm89_fits` (2 x 32 x 232 = 14,848 of 16,384;
[the LU page](lu.md#the-register-cap-that-binds-is-per-sub-partition)). It is a `static_assert` at
`trsm_native.cc:94`.

Diagonal-block inversion is rejected at every tier; see [Negative results](#trsm-negative-results).

### The register gate and the CTA capacity {#the-register-gate-and-the-cta-capacity}

The gate is: stack frame == 0, 0 spill bytes, and `registers * WG <= 65536` as a first screen (the table below). The binding rule is the per-sub-partition
bound `ceil(warps/4)*32*regs <= 16384`, which is the same at WG = 256 but not in general
([agent guide, GPU kernel design facts](../developer/agent-guide.md#11-gpu-kernel-design-facts-sm_89-mostly-general)). Measure it with
`scripts/register_probe.sh`, which replays the shared library's `link.txt`. A per-TU
`-Xcuda-ptxas -v` log is not a valid measurement. Device code is compiled at the shared-library device
link, so such a log reports "argument unused" and no spill whatever the kernel does.

| type | N=8 | N=16 | N=32 | regs*256 at N=32 |
|---|---|---|---|---|
| float | 44 | 76 | 114 | 29,184 |
| double | 59 | 101 | 153 | 39,168 |
| complex\<float\> | 50 | 86 | 148 | 37,888 |
| complex\<double\> | 74 | 138 | 226 | **57,856** |

All 24 kernels (4 types x 3 buckets x 2 sides) have zero frame and zero spill. **Read the frame
column, not the spill column.** N=64 fails: float 119 registers with a **256 B frame**, double 145
registers with a **512 B frame**, both with zero spill. A 256 B frame is 64 floats, so `x[]` itself is
in local memory, which defeats V1's design.

### The `Side::Left` staging tile {#the-sideleft-staging-tile}

For `Side::Left`, thread `u` owns column `u`. At step `s` the lanes of a warp read
`B(rho(s), u0+lane)`, addresses `ldb` apart. ncu, float, n=32, q=1024, batch=512:

| | load sec/req | store sec/req | DRAM vs analytic floor | kernel time |
|---|---|---|---|---|
| Left, before | **31.39** (7.85x) | **32.00** (8.00x) | 0.85x | 0.517 ms |
| Left, after | **5.13** | **4.00** | — | **0.145 ms** |
| Right (never staged) | 5.13 (1.28x) | 4.00 | 0.75x | 0.141 ms |

The defect is LSU/L1 transaction count, not DRAM. DRAM traffic is 0.75-0.85x of the analytic floor
`2*q*n*sizeof(T)*batch`, because the bytes a lane skips at step `s` are read at steps `s+1..s+7` from
cache. Vectorised loads and wider tiles aim at the wrong resource.

Staging is gated to float (`trsm_stage_left`, `trsm_native.cc:58`). For complex it costs registers:
`complex<float>` N=32 gains a 464 B frame and `complex<double>` a 232 B one. Complex also cannot gain
more than 2x, since over-fetch is `32/sizeof(T)` lanes per sector. Float loses only because 32/4 = 8.
With the gate, complex measures 1.00-1.01x. Tile height is 16 for `sizeof(T) <= 4` and 8 otherwise.
The row stride is `NB_STAGE + 1`, which keeps the read-out conflict-free.

### The two-level blocked driver {#the-two-level-blocked-driver}

With the old `nb = 32` outer block, every trailing GEMM at n=512 had one dimension pinned at 32. GEMM
intensity then tops out near `2w/sizeof(T)` = 16 flop/byte for float, against a machine balance of
about 42 on the RTX 4090. The traffic model below counts B elements per batch item, in units of q at
n=512 (ideal 1024):

| OUTER_NB, nb | traffic | |
|---|---|---|
| 32 | 5824 (5.7x) | |
| **128, 32** | **4096 (4.0x)** | **shipped** |
| 128, 64 | 3328 | |
| 128, 128 | 2560 | |

The width is side-dependent, and the sweep is measured. Float, worst cell per order, `vendor_ms/native_ms`:

| order | Left nb32 | nb64 | nb128 | nb256 | Right nb32 | nb64 | nb128 | nb256 |
|---|---|---|---|---|---|---|---|---|
| 128 | 1.18 | 1.10 | 1.20 | 1.17 | 1.00 | 0.96 | 1.00 | 0.98 |
| 256 | 0.75 | 0.78 | **0.87** | 0.75 | 1.01 | 0.94 | **0.83** | 1.01 |
| 512 | 0.58 | 0.74 | **0.76** | 0.75 | 1.07 | 0.92 | **0.82** | 0.91 |

Widening helps Left at every large order and hurts Right at every large order. Left's update is
`C(nb x q)` and Right's is `C(q x nb)`. The two land in different clauses of the gemm kernel selector
of the time, and wider `nb` shortens the inner `k` below the `k >= 128` gate. Whether today's gemm
tables reproduce this mechanism has not been re-measured. `nb256` matching `nb32` at order 256 (0.75
vs 0.75; 1.01 vs 1.01) confirms the knob does what it says.

The side split is a float-only measurement. `nb_sweep.csv` also holds `complex<float>`, which is nearly
insensitive to the knob: 6.91-6.97x at order 128, 7.82-8.10x at 256, 17.67-18.59x at 512, with one
27.41x outlier at nb256. `double` and `complex<double>` were never swept. The shipped side-dependent
default applies to all four types on float evidence alone.

---

## The measured grid {#the-measured-grid}

All ratios are `vendor_ms / native_ms`; **>1 means native is faster**. Runs were on RTX 4090, one card
held exclusive by `experiments/gpu_guard.sh`, with warmup before every timed iteration.
`bench::pristine(B)` is restored between iterations, because trsm is in place. The two legs of each pair
differ only in `BATCHLAS_TRSM_ROUTE`. A ratio is quoted only at saturation: the top batch's GFLOP/s is
within 1.15x of the batch below it, per `(type, side, n, q)`.

### Step-9 grid

The grid is `TrsmOrthoSizes` in `benchmarks/trsm_benchmark.cc`: n in {8,16,32,64,128,256} x q in
{256,1024,4096} x batch in {128,512,2048}, all four types, both sides. The library never issues a square
RHS. Real calls pass a k x k factor as A and an m x k basis as B, so the triangular order is small. The
suite's calls are all `Side::Right, Lower, Trans, NonUnit` inside V1's capacity of 32.

Nine of the 54 cells are dropped by a 6 GB per-cell cap, computed for `complex<double>` and applied to
all types. The grid prints each dropped cell. Read byte counts from `trsm_grid_bytes()`.

Min-max ratio over all saturated cells at each order:

| type | side | order 8 | 32 | 128 | 256 |
|---|---|---|---|---|---|
| double | Right | 4.46-9.62 | 3.65-4.30 | 1.97-2.04 | 1.58-1.62 |
| double | Left | 3.38-8.46 | 2.39-3.52 | 1.62-1.86 | **1.39**-1.51 |
| complex\<double\> | Right | 3.96-9.87 | 3.76-14.35 | 1.33-5.30 | **1.20**-4.74 |
| complex\<double\> | Left | 2.27-6.07 | 2.86-10.39 | 3.52-5.12 | 3.49-4.66 |
| complex\<float\> | Right | **1.01**-1.52 | 1.42-3.84 | 4.99-6.30 | 5.51-8.01 |
| complex\<float\> | Left | 1.06-1.40 | 2.71-3.50 | 6.54-15.85 | 8.20-**21.91** |
| float | Right | 1.62-4.59 | 2.21-3.61 | **0.97**-1.58 | 1.02-1.63 |
| float | Left | 1.61-3.58 | **0.70**-0.87 | 0.71-0.79 | **0.57**-0.63 |

Saturated subsets as ranked in the notes (narrower than the min-max table above, which spans all
saturated cells): double 32/32 wins (1.39-9.62x), complex<double> 30/30 (1.20-4.66x), complex<float> 30/30
(1.01-21.91x), float/Right 18/18 (1.54-4.59x). Float/Left wins 6 of 16. The double path wins 1.39-6.37x
at the same float/Left shapes, because cuBLAS's double triangular path is weak enough that the
over-fetch never decides the race. That is why the decision is per type, and why the tables are per
dtype. The step-16 grid later removed the float/Left loss.

### The batch floor {#the-batch-floor}

The removed router had `if (s.batch < 8) return false;`, which sent batch < 8 to the vendor. It came from
`starved.sh`, a profile over batch {1,8,32} x q {32,128} x n {8,32,128}. The script's own header says
the numbers are launch-overhead dominated and must not be ranked. The floor was deleted in P3.3. The
tables start at batch 128, and smaller batches take the batch-128 rows. Ratios at q = 32 (the q = 128
leg tells the same story unless noted):

| type, side, order | batch=1 | batch=8 | batch=32 |
|---|---|---|---|
| float Right n=32 | **0.462** | 2.108 | 2.063 |
| float Right n=128 | **0.401** | **0.779** | **0.778** |
| float Left n=32 | **0.415** | 1.223 | 1.232 |
| float Left n=128 | **0.229** | **0.756** | **0.756** |
| double Right n=32 | **0.803** | 1.325 | 1.329 |
| double Right n=128 | **0.852** | 1.215 | 1.213 |
| double Right n=8 | 1.131 | 2.932 | 2.915 |
| complex\<double\> Right n=32 | 3.536 | 9.975 | 39.767 |

The old floor sat at the first measured win. Batch=1 loses at every order >= 32 for both real types.
Batch=8 wins at every order except float at 128, which stays a loss (0.740-0.810x on both sides at batch
8 and 32). The floor was type-blind and over-broad. Over both q legs, `double` at order 8 wins
1.09-1.15x at batch=1, `complex<float>` wins 1.29-12.8x at batch=1, and `complex<double>` wins
2.1-11.1x at batch=1.

Below batch 8, sm_89 now runs `cta` (order <= 32) or `blocked`. That recovers the complex and
double-order-8 wins. It also sends the real-type batch=1 losses to native, and nothing has re-measured
those; see [debt 5](#trsm-open-debts). On sm_120 the float and double tables are measured at batch 128,
and their nearest rows decide. For float at q = 32 they rank `vendor` first at orders 64 and 128 on both
sides, and native (`sg_left` Left, `cta` Right) at order 32.

### float, Side::Right: the order clause

The removed router's rule was `return s.batch >= 128 || order <= 32;`, which sent batch 8-127 above
order 32 to the vendor.

* Bracketed: at batch 8 and 32, order 128 measures 0.740-0.810x (a loss) and order 32 measures
  1.157-2.108x (a win). The clause kept the winner and dropped the loser.
* Unbracketed: order 64 was never measured below batch 128. The 32/64 cut point is an interpolation.
* Today sm_89 runs `blocked` there. The sm_120 measured table ranks `vendor` first at order 64 and 128
  for small q.

### The final grid after the routed trailing GEMM {#the-final-grid-after-the-routed-trailing-gemm}

Step 16's grid (`experiments/wp3_s16/baseline.csv`) covers orders 8 to 512, both sides, **float and
`complex<float>` only**. Worst clean cell per order (relative sd <= 10%):

| | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| float Left | 1.60 | 1.72 | 1.69 | 1.65 | 1.42 | 1.25 | 1.21 |
| float Right | 1.62 | 2.37 | 2.23 | 2.01 | 1.60 | 1.23 | **1.00** |
| complex\<float\> Left | 1.05 | 1.41 | 2.64 | 4.69 | 11.21 | 16.62 | 51.70 |
| complex\<float\> Right | 1.01 | 1.03 | 1.35 | 1.90 | 8.49 | 15.31 | 19.64 |

The file holds 224 clean vendor/native pairs, and 223 win. The single loser is float / Right / order 512 /
q=256 / batch=128, at 0.9787, 0.9776 and 0.9832 over three longer repeats. It is the smallest-work cell
at that order (~1.0 ms), and its neighbours win 1.30-1.38x. No router clause was fitted to it, so the
sm_89 row is `blocked | vendor` like its neighbours. The deleted notes said "167 of 168"; that count does
not reproduce from committed data.

The float/Left losing region, by grid:

| grid | clean pairs | losing cells | worst |
|---|---|---|---|
| step 13, before two-level blocking | 192 | 14 (float/Left, orders 256 and 512) | 0.583 |
| step 13, after | 179 | 8 (float/Left, orders 256 and 512) | 0.760 |
| step 16, after routing the trailing GEMM | 224 | 1 (float/Right 512/256/128) | 0.995 |

All 8 step-13 losers satisfy `q*batch >= 524288`, and no cell below that loses. That is why step 13 used a
work threshold and not an order cap. Neither side is DRAM-bound there (11-26% of peak), so the cause was
re-read amplification escaping L2. The cause was not in trsm. V2 called `sycl_gemm::gemm_custom`, the
native entry point, which bypassed gemm's routing. Step 16 injects the **routed** gemm instead.
`trsm_native_blocked` takes a mandatory `TrsmTrailingGemm<T>` (`src/sycl/trsm_native.hh:64-70`), and the
`blocked` arm of `launch` (`src/ops/trsm/trsm.cc:69-79`) passes the public `gemm`. At n=512, q=1024,
batch=512, float, the solve went from 18.8 ms to 11.19 ms, against the vendor's 14.28 ms. The injected
GEMM has no per-call cost: cuBLAS uses `cublasGemmStridedBatchedEx`, and the solve issues 15 GEMM calls.

**Leading dimension.** Every operand trsm hands GEMM is a sub-view that carries its parent's `ld`. The six
shapes V2 issues at order 512 (float, q=1024, batch=512):

| shape | native, ld==rows | native, real ld | vendor, real ld | vendor/native |
|---|---|---|---|---|
| outer m=128 n=1024 k=128 | 0.98 ms | 1.53 | 0.96 | **0.62x** |
| outer k=256 | 2.35 | 2.73 | 1.31 | **0.48x** |
| outer k=384 | 3.49 | 3.78 | 1.63 | **0.43x** |
| inner m=32 n=1024 k=32 | 0.248 | 0.406 | 0.235 | **0.58x** |
| inner k=64 | 0.356 | 0.680 | 0.335 | **0.49x** |
| inner k=96 | 0.487 | 0.887 | 0.426 | **0.48x** |

Strided operands are the only case trsm issues, so a square-matrix GEMM benchmark cannot find this. The
padded and unpadded operands were first initialised differently. After that was fixed, the reference cell
moved 0.34%, so the effect is not a data artefact. The mechanism is in [Refuted mechanisms](#trsm-negative-results).

### End-to-end through `ortho`

Both A/Bs ran with the route unset, so the default selection is what ran.

* **`Side::Right`** (step 9): `ortho` at m in {1024,4096}, k in {16..256}, batch in {128,512}, Chol2 and
  ShiftChol3. 80 cells, all at or above parity, 1.147x-2.719x. Within 4.4% of forced native.
* **`Side::Left`** (step 12): 80 cells, best 2.385x, worst 0.986x, with 7 cells fractionally below parity.
  Forced native at order 256 loses 0.783x. The default tracks the vendor there (4.08 ms default, 4.07 ms
  vendor, 4.35 ms native), so the float/Left predicate of step 12 declined native where it should. The
  sm_89 table now ranks native first at order 256. That choice has not been through this A/B.

Neither A/B has been re-run since step 12 ([debt 8](#trsm-open-debts)).

---

## Negative results {#trsm-negative-results}

| Alternative | Result | Verdict |
|---|---|---|
| Cooperative CTA solve, V3 (`experiments/wp3_s14/v3_cooperative_kernel.patch`, not in tree) | Registers fit (float 106 at N=128, zero spill). Float `Side::Left` worst cell at order 128 goes 1.18x to 0.80x; order 64 goes to 0.39x; orders 256 and 512 move 0.02x despite a predicted 1.6x traffic cut. The traffic model counts bytes, not the serial recurrence's critical path | Rejected |
| Diagonal-block inversion | The "free for ortho" argument compares a trsm residual bound with a CholQR orthogonality bound. Any constant above ~2 flips Chol2 from recovering to not (spec §2.4) | Rejected before building |

Float `Side::Left` worst cell, V3 against V1 (81 cells clean in both runs; per-cell CSVs were deleted
before aggregation):

| order | 8 | 16 | 32 | **64** | **128** | 256 | 512 |
|---|---|---|---|---|---|---|---|
| step 13 (V1 + two-level blocking) | 1.59 | 1.72 | 1.77 | **1.48** | **1.18** | 0.86 | 0.76 |
| step 14 (V3 cooperative) | 1.59 | 1.72 | 1.80 | **0.39** | **0.80** | 0.84 | 0.77 |

`Side::Right` is unaffected. V3 was `Side::Left`-only, and its lane map is wrong for `Side::Right`.

### Rejected: the N=64 CTA bucket {#rejected-the-n64-cta-bucket}

Re-tested after the staging tile, because the tile had cut float `Side::Left` from 114 to 53 registers.
It still fails, and by more: float N=64 Left is 72 registers with a **456 B frame**, and N=64 Right is
119 registers with a **256 B frame**, both zero spill. There is deliberately no N=64 bucket.
`smallest_bucket_ge` returns **0** above 32.

### Rejected: `OUTER_NB` of 128 for `Side::Right` {#rejected-outer_nb-of-128-for-sideright}

`OUTER_NB = 128` on `Side::Right` regresses orders 256 and 512 from 1.01 and 1.07 to 0.83 and 0.82.
Right keeps the single-level schedule. See the sweep in [the two-level blocked driver](#the-two-level-blocked-driver).

### Rejected: the starvation guard {#rejected-the-starvation-guard}

Spec §10's `batch*q < 8*CU*32 -> vendor` is refuted by measurement. At batch=8, q=32 the product is 256
against a threshold of 32,768, and native wins those cells 2.2-2.4x. The guard would have handed back
every one of them. The criterion stated in advance (real trsm above 1.10x vendor at the saturated ortho
shape keeps real on the vendor) did not fire. Double won every cell on both sides, and float won every
`Side::Right` cell.

### Refuted mechanisms

* **The strided-`ld` collapse is not in `register_tiled_common.hh`.** The shapes never execute that file.
  The selector sent them to the 128x128 register kernel, and the dispatcher re-evaluated the same predicate
  to pick the aligned leg. ncu shows every transaction counter identical between packed and strided
  operands. The loss is exposed global-load latency on operand **B alone** (barrier stall 1.552 to 7.703).
  It is beta-independent, which refutes the epilogue explanation. Double-buffering the k-loop and packing
  B were both measured and gained nothing. What worked was routing: a leg predicate used as a route gate
  sent the call to a slower kernel. Routing gained a geomean 1.74x/1.75x (native 0.58x to 0.99x of cuBLAS
  packed; 0.54x to 0.93x strided), but changed no sm_89 runtime. The old float NN window required `m==n==k`.
  See [the strided-ld defect](gemm.md#the-strided-ld-defect-and-the-routing-fix).
* **"The inner blocking level does not matter"** (step 14) was wrong. It was masked by V3's slowness. nsys
  puts the inner GEMMs at 7.83 ms, 42% of the solve for 20% of the flops.

The lesson, twice over: confirm which kernel runs before explaining why it is slow.

---

## Correctness findings

### The missing group barrier {#the-missing-group-barrier}

V1 staged the canonical triangle into SLM with a loop strided by `lane`. Lane `s` then read
`sLc[tri_idx(s,s)]`, a value written by a different lane, with no barrier in between. `sDiv[0]` had the same
problem. The fix is one `sycl::group_barrier` (`src/sycl/trsm_native.cc:184`), commit `c7d6d91`, 2026-08-21.

| | max relative diff vs vendor | items wrong | native residual |
|---|---|---|---|
| barrier deleted | **6.05e+16** | 127 / 128 | 8.0e+05 |
| barrier restored | 4.27e-07 | 0 / 128 | 2.38e-07 (= vendor) |

The blocked potrf driver was also affected. Vendor-free potrf at n=1024, batch=256 returned `info != 0`
for 61-75 of 256 items before the fix, always at a panel's first column. After it, 0/256 fail over every
rep up to batch 1024.

**How it hid.** The ladder picks wg=32 for every trsm test, which uses `bs <= 3, q <= 257`. That is a
single sub-group, which executes both loops in lock step. The race needs more than one sub-group. The
blocked potrf panel solve at n=1024, batch=256 gives wg=256, and the race fires there.

**The shipped regression test was vacuous.** It called V1 directly at n=16 and passed with the barrier
deleted. The reproducer goes through V2: order **48** (final V1 block of order 16), q=976, batch=128.
Orders whose final block lands in the N=16 bucket (48, 77, 80, 109) fail 90-128 of 128 items, while 32,
33, 64, 65, 96 and 155 pass. The guard is `TrsmNativeBlocked.MultiSubGroupWorkGroupStagesItsTriangleCorrectly`
(`tests/trsm_tests.cc:561`). It was verified red with the barrier deleted.

### The bucket ladder that truncated {#the-bucket-ladder-that-truncated}

`smallest_bucket_ge` returned 64 for any `n > 32`, and the dispatch `default:` label mapped 64 onto the
N=32 instantiation. An order-33 solve therefore solved the leading 32x32 system and left the last row of B
untouched. Nothing caught it. Now `smallest_bucket_ge` returns 0 and V1 throws
(`TrsmNativeCta.OverCapacityThrowsRatherThanTruncating`). The facade caps order through `cta`'s `can_run`,
but V2 calls the direct entry on its diagonal blocks.

### Testing traps

* **Every blocked test stopped at order 100**, which is a single panel with `OUTER_NB = 128`. The outer level
  never ran. `TwoLevelPanelStructure` uses 129, 256, 300 and 384 instead.
* **Alpha is applied exactly once.** For blocks `i > 0`, alpha arrives through the trailing GEMM's beta. The
  natural `beta = 1` is wrong at every later block and invisible to any `alpha == 1` test. Mutation tests:
  outer beta breaks 4 tests, inner beta breaks 9.
* **Complex exposes ConjTrans and the reciprocal.** For real scalars ConjTrans equals Trans, so real cells
  cannot see it. `tri_fill` gives every element a nonzero imaginary part. The complex reciprocal uses Smith's
  form, since the textbook `conj(d)/|d|^2` returns 0 for representable results.
* **The oracle is an independent multiply-back.** The netlib and cuBLAS backends fold the 24 cases the same
  way, so checking against either validates the fold against itself.
* **`ctest -L blas -L ortho` runs zero tests and exits 0.** Repeated `-L` is an AND. Use
  `ctest -L 'blas|ortho'`, which runs 20 tests.

---

## Open debts {#trsm-open-debts}

1. **Every grid predates the barrier.** The WP3 grids (steps 9 to 16), the batch-floor profile, the ortho A/Bs
   and the strided-ld re-diagnosis were all added 2026-08-19/20. The barrier landed 2026-08-21. Every ratio on
   this page may reflect the racing kernel. The 2026-09-15 dating pass cleared nothing. The smallest grid cell
   (q=256, batch=128) runs wg=64 under the ladder rule. No measured cell is confirmed at single-sub-group width,
   and that is an inference from the ladder arithmetic, not a re-run. The sm_89 tables transcribe these grids,
   so the taint carries into them. The only post-barrier trsm timings are the sm_120 float and double tables
   (2026-10-05; [Blackwell page](blackwell.md#trsm-on-sm_120-result)). No trsm CSVs exist under
   `benchmarks/results/`, and the batch-floor profile has never been re-measured.
2. **`double` and `complex<double>` were last measured at step 9** (orders 8 to 256), before the staging tile,
   two-level blocking and the routed GEMM. Those change V2 for every type. The sm_89 native-first rows for both
   rest on that evidence and the transcribed old `preferred()`. `complex<double>` has no post-step-9
   measurement on either device.
3. **No order above 512 has been measured on sm_89, for any type.** `blocked`'s `can_run` has no order term.
   The tables run to order 1024, and larger orders take the order-1024 row (blocked first on sm_89).
4. **float at batch 8-127 is barely measured above order 32.** The tables start at batch 128, so these calls
   take the batch-128 rows. The only data is the starved profile (order 128 at 0.740-0.810x on both sides,
   `Side::Left` at 0.756-0.780x), which cannot rank. Order 64 was never run below batch 128, so the old
   `order <= 32` cut was an interpolation.
5. **Below batch 8 the decision flipped, and neither side is measured at saturation.** The old floor of 8 was
   type-blind. Since P3.3 those calls run native on sm_89, including the real-type batch=1 cells that the profile
   puts at 0.229-0.852x.
6. **Every complex "vendor" ratio is native against another BatchLAS kernel.** `trsm_vendor` in
   `src/backends/cublas.cc` (the complex branch from `:654`) diverts both complex types to a hand-written
   sequential per-RHS SYCL substitution. `cublasCtrsmBatched` and `cublasZtrsmBatched` (`:764`) are unreachable.
   The diversion rests on an uncited comment about NaNs under SYCL/USM interop. The 8-52x complex ratios at
   large order are not a vendor-independence result. Whether the diversion is still warranted is open.
7. **complex<float> `Side::Right` at orders 8-16 (1.01-1.05x) is a roofline tie, not a defect.** Native runs at
   88.5-90.5% of the 1008 GB/s DRAM peak. The ceiling is 1.12-1.18x. At small `q*batch` the data fits in L2,
   and those cells win 1.25-2.04x.
8. **The end-to-end `ortho` A/B has not been re-run since step 12**, before two-level blocking and the routed GEMM.
9. **The step-16 cell count does not reconcile.** The committed `baseline.csv` holds 224 clean pairs with one
   loser, not "167 of 168". The conclusion is unchanged.
10. **`experiments/wp3_s14`'s per-cell CSVs were deleted before aggregation.** The V3 table is a record, not a
    derivation from data.
11. **The residual strided-`ld` slope is unexplained.** Routing reached 0.93x of cuBLAS on strided operands. The
    rest is exposed load latency on operand B, monotonic in stride. Ranking native above vendor in gemm's
    strided rows is arguable, not a win.
12. ~~**Heterogeneous-batch gate.**~~ Withdrawn 2026-09-15: the gate is armed. The native `can_run` has the term
    (`trsm.cc:47`), and `TrsmCandidates.HeterogeneousBatchHasNoNativeRoute` (`tests/trsm_candidates_tests.cc:686`)
    pins it. The vendor's `can_run` lacks the term ([known-defects 12](../design/known-defects.md#defect-12-vendor-potrf-and-trsm-accept-a-heterogeneous-batch)).
13. ~~**`OpShape::compute_units` is dead.**~~ Closed: `OpShape` was deleted with the route layer. A speed rule
    belongs in the tables, never in `can_run`.
14. **`MatrixView::operator()(Slice,Slice)` passes the parent pointer array** (`matrix.hh:1008`). Reported and
    left alone. V2 passes the parent's `ld` and `stride` at every sub-view, because the constructor defaults
    `stride` to `ld*cols` when 0 is passed.
15. **No extension is vendor-free through WP3.** `ortho_tests` is blocked by `potrf`, `geqrf`, `orgqr` and `syev`.
    `cond_tests` and `inverse_tests` are blocked by `syev`, `getrf` and `getri`. There is no CPU trsm.
16. **The suite exercises native trsm only because the batch floor is gone.** Every trsm call the suite issues has
    batch <= 5. A change to the top rows of a table must be validated by an A/B through a real caller.
17. **The composed `posv` fallback is 1.5-5.6x slower under Auto legs than under vendor-pinned legs.** The
    non-`potrf` part is about 20x. `posv` and `gesv` have no vendor family of their own. Their `blocked`
    family composes sub-ops that select for themselves. The arm named `blocked` pins only the outer route, so
    both legs stay on Auto. The pins for the other arms are `{POTRF=vendor, TRSM=vendor}` and
    `{POTRF=tiny, TRSM=cta}` (`benchmarks/factor_bench.cc:458-486`).

    Data: `benchmarks/results/p2_posv_float.csv` (float, n=17, batch=16384, 7 reps, all `bad = 0`, residuals
    3.4e-08 to 5.0e-08). `rel_sd` is at most 0.0035, and the rows the argument rests on are all at or below
    0.002. Times are in ms:

    | `nrhs` | `tiny` (fused) | `blocked` (Auto legs) | vendor-pinned legs | native-pinned legs |
    |---:|---:|---:|---:|---:|
    | 1 | 0.172186 | **1.323346** | 0.256094 | 1.230026 |
    | 4 | 0.239649 | **1.325243** | 0.316300 | 1.230784 |

    Across n in {4,8,9,16,17,24,32}, batch in {8192,16384,32768} and nrhs in {1,4}, all 42 cells are above
    parity. The range is 1.50x (n=8, nrhs=1, batch=8192) to 5.56x (n=17, nrhs=1, batch=32768). The gap is not
    monotonic in n, nrhs or batch.

    Standalone float `potrf` at n=17, batch=16384 takes 0.200091 ms vendor and 0.106518 ms `tiny`
    (`benchmarks/results/p9_potrf_tiny.csv`). So the non-`potrf` remainder is 0.056 ms (22%) of the vendor-pinned
    arm and 1.124 ms (91%) of the native-pinned arm, about 20x. Native `potrf` is 1.9-2.6x faster than vendor
    `potrf` at this shape. The composed arm barely moves with nrhs: `blocked` rises 0.14% and native-pinned
    0.06% for 4x the right-hand sides. Vendor-pinned rises 23.5%. The signature is clean at n <= 17 and fades
    by order 32. `gesv` shows no gap at the same cell (`benchmarks/results/p2_gesv_float.csv`), so this is
    specific to `posv`.

    **Hypothesis.** The composed cost is the two triangular solves. The native trsm at order ~17 costs roughly a
    fixed amount per batch item, independent of q, because one work-group per matrix carries q for free. If so,
    the sm_89 row `cta | blocked | vendor` at order <= 32 is the thing to revisit, since it has no q term.
    **Falsifier, not yet run:** a standalone trsm A/B at order 17, in the side and transpose that the `posv`
    fallback issues, with q in {1,4,16,64} and batch=16384, `BATCHLAS_TRSM_ROUTE=cta` against `=vendor`,
    interleaved in one process. Flat in q, and about 20x the vendor at q=1, confirms the hypothesis. Tracking the
    vendor refutes it and moves the cost into the composition itself.
18. ~~**`trsm_op_shape` never set `s.backend`.**~~ Resolved by flat selection: the coverage row carries the backend
    (`TrsmCandidates.CoverageRowCarriesBackendKeyAndNativeFlags`, `tests/trsm_candidates_tests.cc:917`).

---

## Raw evidence {#trsm-raw-evidence}

Raw data is at the git tag `perf-evidence/vendor-independence`, readable with
`git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| step-9 routing grid, all four types, both sides | `experiments/wp3_s9/{left,right}-{vendor,native}.csv`, `sweep.sh`, `analyse.py` |
| batch-floor profile (never ranked) | `experiments/wp3_s9/starved-*.csv`, `starved.sh` |
| end-to-end `ortho` A/B, `Side::Right` | `experiments/wp3_s9/ortho-{vendor,native,default}.csv`, `ortho_ab.sh` |
| `Side::Left` staging tile: ncu profile, grid | `experiments/wp3_s12/` (`profile.sh`, `left_sweep.sh`, `left-*.csv`) |
| end-to-end `ortho` A/B, `Side::Left` | `experiments/wp3_s12/ortho-left-{vendor,native,default}.csv`, `ortho_left_ab.sh` |
| two-level blocking, `OUTER_NB` sweep, orders to 512 | `experiments/wp3_s13/{baseline-before,baseline,nb_sweep}.csv` |
| rejected cooperative CTA solve (V3) | `experiments/wp3_s14/` (`v3_cooperative_kernel.patch`, `measure.py`) |
| routed trailing GEMM; isolated shapes at `ld==rows` vs real `ld` | `experiments/wp3_s16/baseline.csv`, `{outer,inner}{,pad}-{vendor,native}.csv`, `recheck-*` |
| strided-`ld` re-diagnosis and routing fix | `experiments/wp4_gemm_ld/` |
| GPU exclusivity guard | `experiments/gpu_guard.sh` |
| verification pass behind the spec corrections | `experiments/wp3/verification_pass_raw.md` |

Sweep CSVs have the columns `name,arg0,arg1,arg2,iterations,avg_ms,stddev_ms,GFLOPS,Time (us) / matrix`, with
`arg0 = n`, `arg1 = q` and `arg2 = batch`. GFLOPS uses `n^2 q` for all four types, so complex is understated
4x by construction. Compare `avg_ms` across types, never GFLOPS. The step-13/14/16 `baseline.csv` files have the
columns `type,side,route,n,q,batch,ms,sd_pct`, and `sd_pct <= 10` is the cleanliness gate.

**Measurement notes.**

* `gpu_guard.sh` samples foreign processes only at the start and end of a run. Two copies of your own sweep
  queued behind a co-tenant are invisible to it, and produced 22 of 180 cells at 10-103% sd. The drivers now
  take a `flock` and drop any leg with a cell above 10% sd.
* Harness rows are registered as `BM_TRSM_StarvedRight`, not `BM_TRSM_OrthoRightStarved`, because `--name`
  matches by substring and the latter would pull profile-only rows into the saturated grid
  (`benchmarks/trsm_benchmark.cc:152-155`).
* `trsm_announce_route_env()` (`benchmarks/trsm_benchmark.cc:94-107`) prints the pin once per process, so a log
  records which leg it measured. An unknown pin throws at the first call.
* `ncu --kernel-name` matches the mangled name. `regex:TrsmCtaKernel` matches nothing, and ncu then emits no
  metrics. Use `--kernel-name-base demangled` (`experiments/wp3_s12/profile.sh`).
