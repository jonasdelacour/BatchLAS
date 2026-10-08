# GEMV {#perf_gemv}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · measured 2026-08/09, tables transcribed 2026-10-05

Native batched `gemv` for all four scalar types and all three `transA`. This page gives the
families, the five kernel bodies, the gates that pick a body, the `complex<double>` transposed
window, and the measurements behind them.

Box: RTX 4090, 128 SMs, 72 MB L2, 1008 GB/s theoretical DRAM peak. **~950 GB/s is the achievable
roof.** Ratios are `vendor_ms / native_ms` (above 1 means native is faster), except the body-5
tables, which use `body3_ms / body5_ms`.

## Families and kernel bodies

### The families and their `can_run` terms

`gemv` decides in `src/ops/gemv/gemv.cc` by flat selection (@ref design_flat_selection).
`select::run` takes the first entry of the nearest row of `tuned/gemv.<dtype>.<device>.txt` that
`can_run` admits. The last resort is `vendor`, then `direct` (`src/ops/gemv/choice.hh:30`). The
tables are transcribed, not timed (`source=transcribed:424a45bc`, `tuned/README.md`).

| spelling | driver | bodies | `can_run` (`src/ops/gemv/gemv.cc:53-64`) |
|---|---|---|---|
| `cta` | `sycl_gemv::gemv_native_cta` | 3, 5 | native term, `gemv_cta_available<T>()`, `d.is_gpu && d.has_sg32 && transA != NoTrans` |
| `direct` | `sycl_gemv::gemv_native_direct` | 1, 2, 4 | native term and `gemv_direct_available<T>()`; no `is_gpu` term |
| `vendor` | `backend::gemv_vendor` | — | `d.has_vendor` |

`can_run` is correctness only, never a speed cutoff. The native term (`gemv.cc:55-57`) refuses a
heterogeneous `A`, negative extents and `batch < 1`, and requires x and y to match A in batch and
length ([shape builder contract](#gemv-the-shape-builder-contract)). `m == 0 || n == 0` is not
refused; it quick-returns.

`has_sg32` comes from `sycl::info::device::sub_group_sizes`, through `Device::supports_sub_group_size`.
Do not use `get_property(MAX_SUB_GROUP_SIZE)`: it returns `sub_group_sizes()[0]`, which is wrong in
both directions. A `reqd_sub_group_size(32)` launch on a device without 32 aborts.

> **Note:** `direct` has no GPU gate on purpose. `src/sycl/gemv_native.cc` also builds for
> `native_cpu`, and bodies 1 and 2 need no collective, local memory or required sub-group size.

Layout is column-major: `A(i,j)` sits at `i + j*ld`, one work-item per output element. `NoTrans` is
coalesced. `Trans` and `ConjTrans` need the sub-group reduction.

### The five kernel bodies

| body | kernel | family | shape | decomposition |
|---|---|---|---|---|
| 1 | `GemvDirectNKernel<T>` | `direct` | `NoTrans` | one work-item per output row |
| 2 | `GemvDirectTKernel<T>` | `direct` | `Trans`/`ConjTrans` | one work-item per output column (the portable arm) |
| 3 | `GemvCtaTKernel<T>` | `cta` | transposed, GPU, sg 32 | one 32-lane sub-group per output, `shift_group_left` ladder |
| 4 | `GemvSegNKernel<T,W>` | `direct` | `NoTrans`, `out_len <= 16`, sg 32 | `W = 32/out_len` lanes per output, one sub-group per batch item |
| 5 | `GemvSegTKernel<T,W>` | `cta` | transposed, short reduction, sg 32 | `W` outputs per sub-group, `L = 32/W` lanes each |

A spelling does not name the body, and the trace and coverage `chosen_algo` carry only the spelling.
Body choice is derived in the driver, not made a family or `can_run` term, because that would put a
speed cutoff in the correctness predicate. To tell bodies apart, use `gemv_seg_trans_width_debug`
(test-only) or a break that is red for one body only.

All five use zero local memory. The reduction is a hand-rolled shuffle ladder;
`sycl::reduce_over_group` measured 1.5–4.7× slower for `double` and `complex<double>`. Body 5 uses
the same registers as body 3 at every `W` (float 28, double 36, cfloat 38, cdouble 40). Its 12 extra
entry points add +0.45% to link time, inside a 0.84 s spread.

### The cdouble window in the tables

Keys are `trans:exact out:log red:log batch:log` (`choice.hh:34`). `out` and `red` are the y and x
lengths and swap with `transA` (`gemv.cc:27-35`); ConjTrans folds to `T` (`gemv.cc:37-44`). The
grid (`choice.hh:37-39`) straddles every edge: `out` 255/256, `red` 63/64 and 352/353, `batch` 319/320.

| rows | ranking |
|---|---|
| 48, `complex<double>` only (`trans=T`, `red` 64–352, `out` ≥ 256, `batch` ≥ 320) | `cta \| vendor \| direct` |
| every other `trans=T` row | `vendor \| cta \| direct` |
| every `trans=N` row | `vendor \| direct` (`cta` cannot run NoTrans) |

With the vendor present, `cta` ranks first only for `complex<double>`, `transA != NoTrans`,
`64 <= red_len <= 352`, `out_len >= 256`, `batch >= 320`. `direct` never ranks first while the vendor
can run. Without a vendor, `cta` takes every transposed GPU call.

The band is on `red_len` (`n` under NoTrans, `m` otherwise). A window written on `out_len` tests the
wrong extent and inverts. `tests/gemv_candidates_tests.cc` (`GemvTranscribedTable.RowsHoldTheOldPreference`)
checks the rows against the old preference.

### The sub-route gates

These are kernel-internal, not keys, families or `can_run` terms.

**Body 4** (`gemv_seg_width`, `src/sycl/gemv_native.cc:76-81`): `W` is the largest power of two with
`W*out_len <= 32`. `W == 1` means no segmentation, so body 4 serves `out_len <= 16` and body 1 takes
17 and up. It also requires `Device::supports_sub_group_size(32)`, false on `native_cpu`.

**Body 5** (`gemv_seg_trans_width`, `src/sycl/gemv_native.cc:83-107`) has three gates, all on
`red_len()` and transcribed cell by cell from a CSV:

| gate | float | `complex<float>` | double | `complex<double>` |
|---|---|---|---|---|
| 1: body 5 runs at `red_len <=` | 32 | 16 | 48 | 64 |
| 2: `W = 8` up to `red_len <=`, then `W = 4` | 24 | 16 | 32 | 32 |
| 3: floor on `out_len*batch` | `16*CU` in the `W = 8` band, `64*CU` in the `W = 4` band (2048 / 8192 here) | | | |

`W ∈ {2, 4, 8}` is instantiated; `W = 2` keeps `BATCHLAS_GEMV_SEGT=2` meaning `W = 2`.

* **`BATCHLAS_GEMV_ROUTE`** takes `auto`, `native`, `vendor` or a spelling (`cta`, `direct`). A
  spelling the shape cannot take throws `invalid_argument` (`src/select/select.hh:402-407`); `cta` on a
  NoTrans call is an error, not a reroute. Only `native` or `vendor` with nothing of their class
  runnable falls back to Auto, with a warning.
* **`BATCHLAS_GEMV_SEGT`** (`off` / `auto` / `2|4|8`) picks the body-5 spelling inside `cta` and
  bypasses all three gates. It is read on every launch, never latched (`gemv_segt_mode`,
  `src/sycl/gemv_native.cc:116-124`).

> **Note:** The trace or coverage spelling is the only record of which family ran. It does not name the body.

## Evidence for each boundary

### The vendor baseline

cuBLAS `gemvStridedBatched` measures 94–105% of the roof on 90 of 92 reproducing cells (all four
types, both `transA`, `n` 32…2048, `batch` 64…65536). A batched gemv reads A once for two flops, so
parity is the achievable result; a large speedup on a DRAM-resident cell is a measurement error.

The exception is `complex<double>` transposed. At matched bytes (A = 1024 MB, `Trans`), cdouble
reads 325 GB/s at 256×256 and 376 GB/s at 64×256, where float, double and cfloat read 936–967 GB/s.
Controls rule out the alternatives. Padding `ld` to 257/264/320 leaves 307–325 GB/s, and `ld = 513`
on a healthy cell still drops 922 → 718 GB/s, so the instrument responds. Over 241 cdouble
transposed cells the vendor is bimodal (median 334 GB/s on the 68 cells we win, 892 on the other 173).
The native CTA body reads 936–941 GB/s for all four types.

### The cdouble window boundaries

The dip is a discrete kernel-selection switch inside cuBLAS, not a gradient. At `out 512, red 128`,
cdouble `Trans` reads 894.9 / 919.4 / 930.1 GB/s at batch 128 / 192 / 256, then 358–363 GB/s at
batch 320–512. No function of `n*batch` fits it, so the window needs its own batch edge. The finer
grid (`g6_fit2_p1.csv`) places the switch between batch 256 and 288. See [open debts](#gemv-open-debts).

| boundary | inside | bracketing non-winner |
|---|---|---|
| `red_len >= 64` | 2.41 at `red_len 64` | **0.9515** at 32; 1.10 at 40 |
| `red_len <= 352` | 2.84 at `red_len 352` | **1.0304** at 384, where cuBLAS is back at the roof |
| `out_len >= 256` | 2.32 at `out_len 256` | **0.9988** at 192; 0.546 at 64 (cuBLAS L2-resident, 1777 GB/s) |
| `batch >= 320` | 63 cells, geomean 2.572, min 2.2261 | **0.9628** at `out 512, red 128, batch 256`; **0.9562** at batch 128 |
| `scalar == cdouble` | | float **0.9340**, double **0.9722**, cfloat **0.6644** |

Below batch 128 the region is refuted outright. 27 of 46 cells in the `(red_len, out_len)` band at
batch 1…96 lose; the worst is **0.5417** at `out 512, red 128, batch 64`. Shipped clause on grid B
(`Trans` alone): 63 cells, geomean 2.572, zero losses. ConjTrans is a peer (median 0.07% apart).

> **Warning:** The weakest admitted cell is an L2-boundary corner. The lowest-footprint admitted cell
> (cdouble, `out_len 256`, `red_len 64`, `batch 320`) is 84 MB against a 72 MB L2. It reads 2.313 on
> device 0 and 1.87–2.07 on the idle card. Only the vendor moves (~20%). **Read the clause floor as
> ~1.87, not 2.23.** The `g6_fit` grids were measured on device 0 only.

Device hygiene: `nvidia-smi --query-compute-apps` is per device and cannot see Xorg or gnome-shell.
That is invisible in DRAM-resident numbers, not in L2-resident ones (`out 64, red 64, batch 512`:
vendor 1366 GB/s on device 1, 761 on device 0). Native-vs-native A/B runs on device 0; vendor tables
run on device 1. `rel_sd` does not catch contention, so parity rows carry a per-device foreign-process
count. Rebuild `ab/build.sh` after any decision change: it prints the route in its own TU.

### The body-5 gates

**Gate 1** is where body 3 stops being materially short of the roof. GB/s at `out_len 2048, batch 512,
Trans`, DRAM-resident; bold is the last rung admitted.

| `red_len` | 1 | 8 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| float | 59.3 | 261.5 | 485.5 | 655.9 | **832.6** | 912.9 | 922.7 | 931.0 |
| cfloat | 104.8 | 450.2 | **779.8** | 911.1 | 921.7 | 929.4 | 925.2 | 932.7 |
| double | 33.3 | 149.8 | 281.9 | 415.9 | 548.1 | **817.0** | 928.7 | 932.2 |
| cdouble | 26.6 | 118.8 | 224.2 | 329.2 | 434.3 | 456.2 | **707.9** | 932.5 |

Forcing `W ∈ {4,8}` past each gate measures 0.983–0.996 at DRAM footprint, so the gate is a real boundary.

**Body 5 against body 3** (11 reps, median, worse of two passes):

| grid | cells | geomean | min | max | below 1.00 |
|---|---|---|---|---|---|
| `Trans`, admitted | 83 | **3.286** | 1.070 | 10.47 | 0 |
| `ConjTrans`, admitted | 36 | **2.746** | 1.074 | 10.44 | 0 |
| skinny (`out_len` 1…64), admitted | 30 | **1.566** | 1.037 | 3.04 | 0 |

Declined cells all stay within 0.977–1.009.

**Gate 3 (floor on `out_len*batch`)** is needed. Without it, 16 cells at `out_len` 1…4 lose, worst
0.891×, all at `out_len*batch <= 4096`. A single `8*CU` floor leaves five cells at 0.976–0.998; the
two-row floor (`16*CU` / `64*CU`) costs seven wins of 1.02–1.36×.

**Odd `ld`** costs body 5 something but never flips the sign: every admitted odd-`ld` cell stays at or
above 1.06×.

### The body-4 gate

Before the repair, the NoTrans `out_len < 32` family was 0.08×–0.38× of cuBLAS on 13 cells. The worst,
`cfloat out=1 red=2048 batch=512`, reads 99.2 GB/s against the vendor's 1206.9.

`ncu` (cdouble, `red_len 2048`, `batch 512`) shows two effects that both stop at exactly 32 lanes:
sectors per load fall from 32.0 at `out_len 1` to 8.5 at 32, and DRAM throughput rises from 7% to
85%. It is a warp effect, not a bytes effect: float turns at the same `out_len 32` (sectors per load
32.00 → 2.50), and padding `ld` changes nothing.

| | before | after |
|---|---|---|
| cells below 0.50× (of 192) | **15** | **2** |
| worst cell | **0.08×** | **0.44×** |
| `NoTrans`, `out_len < 32` (24 cells) | min 0.08, median 0.38, 13 blockers | min 0.60, median **1.16**, 0 blockers |

Body 4's `out_len <= 16` edge is bracketed on one side by [kernel hypotheses](#kernel-hypotheses-refuted).
On the other it rests on arithmetic. **`17 <= out_len <= 31` is unmeasured**; a test pins the boundary,
not a timing. The last blockers, `cdouble out=64 red=64 batch=512` (0.450 / 0.472 after the repair,
0.862 / 0.861 after body 5), do not reach parity: the vendor runs above DRAM peak there, from L2 residency.

## Negative results

### Routing hypotheses refuted

| Alternative | Result | Verdict |
|---|---|---|
| `64 <= m <= 320 && n >= 256` | Fitted at a fixed ~1 GB footprint, where batch is invisible. Admits 17 cells below 1.00×, worst 0.36× | Refuted |
| `n*batch >= 131072` | cuBLAS at the roof (924.7 GB/s) at `m=128, n=512, batch=256`; 0.97 / 0.97 | Refuted twice |
| `A >= 256 MB` in place of a batch term | 0.9628 at 256 MB. The dip switches on at 537 MB for one shape and 134 MB for another | Refuted by a cell |
| `64 <= m <= 320 && A >= 512 MB` | Worst cell 1.01×; never clears the ≥ 1.15× bar | Fails the gate |
| WP7 clause search (`clause_search.py`) | No batch term; the two passing clauses captured at most 22 of 68 wins | Superseded |
| `red_len >= 48` | 88 cells, geomean 2.135, min 1.1605, zero losses; the vendor is still on the slope at red 48 | Declined |
| `out_len >= 768 && batch >= 128` | ~18 cells at 2.26×–2.91×, no measured loss | Not shipped ([open debts](#gemv-open-debts)) |

### Kernel hypotheses refuted

| Alternative | Result | Verdict |
|---|---|---|
| The shuffle ladder is a fixed cost per output | At `out_len 2048, batch 512, Trans, red 32`: float 833.8, **double 547.5**, **cfloat 921.2**, cdouble 434.5 GB/s. Double and cfloat differ 1.68× at identical bytes and shuffle count; `sm__pipe_fp64_cycles_active` is 58–85% for double, 0% for float and cfloat | Refuted. The cost is FP64 work on a 1/64-rate part (160 double-adds per output in `sg_sum`, 320 for `complex<double>`). Body 5 reduces it to `L*log2(L)` |
| Runtime `W` | Compile-time `W` gives 90–98% of the vendor on the same shapes; runtime `W` is slower by up to 2× for `out_len >= 8` (`ncu`: the loop does not unroll) | Refuted |
| Sector floor (`L*sizeof(T) >= 32`) | Below `red_len ≈ 32`, float at `out_len 2048` runs 5.16–5.91× at `W = 8` against 3.34–3.40× at `W = 4` | Refuted below `red_len ≈ 32` |
| Only `double` and `complex<double>` need body 5 | At `red_len 8`, body 3 reads 261 GB/s (float) and 450 (cfloat). Body 5 is worth 5.91× and 3.98× | Refuted; all four types emit it |
| Per-call route resolution is a regression | 0.164 µs per call, 2–3% of a minimal batched launch | Not a regression |

`out_len = 12` is the one case where body 1 leads by more than noise: `W = 2` leaves 8 of 32 lanes
idle because 12 does not divide 32.

## GEMV: the shape builder contract

The rules below apply to the native term of `can_run` in `src/ops/gemv/gemv.cc`.

* **The native term is the only agreement check.** It requires homogeneous `A`, `batch >= 1`, x and y
  matching A in batch, `X.size() == red_len` and `Y.size() == out_len`. The public entry validates
  nothing, and no throw was added, because it would turn silent misbehaviour into a crash on live
  paths. A non-conforming call goes to `vendor`; in a vendor-free build `select::pick` throws `NoRouteError`.
* **Batch agreement.** The vendor's strided-batched call reads `A.batch_size()` items from all three
  views with each view's stride. A disagreement is a buffer overrun in the vendor too.
* **`can_run` reads metadata only** (`rows()`, `cols()`, `size()`, `batch_size()`, `is_heterogeneous()`),
  never `data_ptr()`. A data read in a sizing path segfaults.
* **Lengths swap with `transA`.** `red_len` is `n` under NoTrans and `m` otherwise. Table keys use these
  lengths. Coverage `m` and `n` are A's extents as stored; `k` repeats `m`.
* **`transA` separates the kernels.** It is a table key and part of `coverage.cc`'s `variant_key`.
  Dropping it merges body 1 with bodies 2/3 into one first-writer-wins row.
* **Only `A` can be heterogeneous.** `VectorView` has no active-size concept.
* **Capabilities are asked of the kernel TU** (`gemv_direct_available`, `gemv_cta_available`).
* **Environment.** `BATCHLAS_GEMV_ROUTE` is parsed by `select`; `BATCHLAS_GEMV_SEGT` by the driver.

## GEMV: correctness findings

Every timed sweep reports `relerr` exactly 0. **That is not evidence of numerical quality.** The A/B
data are multiples of 1/256 and every sum stays below 2^24, so any reduction order is bit-exact. The
correctness evidence is `tests/gemv_tests.cc` and its breaks. Two contracts are silently wrong if misstated:

1. **Quick return.** The reference `?GEMV` returns when `m == 0 || n == 0 || (alpha == 0 && beta == 1)` and
   leaves `y` untouched. It does not compute `y = beta*y`. `A` is never read when `alpha == 0`. Tested on
   opposite arms: `n == 0` under NoTrans, `m == 0` under Trans.
2. **No `__restrict__`.** `ortho.cc:198-203` passes `A_i` and `A_next` as views into one allocation.
   They are element-disjoint but alias at the object level.

### Blind guards found and closed

1. **The pre-WP7 fixture (40 cases)** used a fixed 10×10 matrix, batch 5, `ld == rows`, `inc == 1`, square
   only, no `ConjTrans`, and real data only. Breaks `cross`, `conj`, `ld`, `xinc`, `yinc`, `segld`,
   `segxinc` and `segyinc` all stayed green. `ConjTrans` is the live production path (`ortho.cc:123-124`).
2. **The natural batch stride.** All 232 cases used `a_stride == ld*n`. `ortho.cc:189-191` passes
   `A.stride() == m*A.cols()` against a view whose `ld*cols` is `m*i`. The `stride_pad` cases, one per
   body, now cover it (break `padstride` turns 32 cases red).
3. **No guard band past `y`.** Body 5's tail sub-group can run past the last output. Three breaks of its
   mask and clamp stayed green over 376 cases. `run_case` now allocates and poisons 64 guard elements.
   `segTtailwrite` and `segTclampoff2` turn exactly the three partial-tail cases red.
4. **No `Op::gemv` route assertions.** Vendor-free, pinning `native:direct` removed the CTA kernel from every
   decision and stayed green. `tests/gemv_candidates_tests.cc` now covers it
   (`GemvDeviceAllows.CtaNeedsAGpuWithSubGroup32AndATransposedCall`, `AutoReadsEveryKeyField`,
   `TraceKeyOutRedFollowTrans`, `CanRunFalsePinsThrow`, among others).

Anti-vacuity: `SegTransCasesAreReachable` skips, with the numbers in its message, when `16*CU` or `64*CU`
exceed the body-5 cases' `out_len*batch` (2385 and 8288). `gemv_seg_trans_width_debug` resolves through the
same gate function as the launcher, because two copies of one boundary had drifted apart.

Breaks that stayed green are correct, not gaps:

| break | why green is correct |
|---|---|
| `flatten` (mis-pairs `b` and `i`) | Still a bijection onto the same `(b, i)` set. Flattening is a performance property |
| `segactive` (drops `jsub < W` in body 4) | The silenced lanes were never read. A work saving |
| `segTtail` (`return` instead of mask) | The mask is a spec requirement (a shuffle reached by part of a sub-group is UB), not an observable value |
| `segTclampoff` (clamp removed, mask kept) | The clamp affects only inactive lane groups; with `segTclampoff2` the pair is red |
| `segTlaunch` (sub-groups not divided by `W`) | Extra sub-groups return at once. A performance defect |

Armed breaks (14 against body 4, 24 against body 5; `breaks_kernel.py`, `breaks_body5.py`) turn 13 and 21
cases red respectively; the rest are the green ones above. The earlier `preferred()` breaks
(`gemv_axisswap`, `gemv_nobatch`, `gemv_alltypes`) guard the window now in the tables.

### The known bad caller

`src/extensions/ortho.cc:189-194`, in the `transA = Trans` branch, builds `A_i` as `i × m` with `ld = m` and
passes `A(Slice(), i)` as `x`. That column has length `A.rows()`, so the lengths agree only when
`A.rows() == m`. The call is structurally wrong under the vendor. It is not fixed, and no throw was added
because it would crash a live path. The length checks make both native families unrunnable for it, so it
goes to `vendor` and behaves as before. A fix needs the correct `A_i` and an `ortho(..., Transpose::Trans)`
test that checks row orthogonality; `ortho_tests` has none. Other `ortho` calls are fine: over 56 cells the
worst is 0.75×, the median 1.14×.

## GEMV: open debts

* **Upper corners are unmeasured.** Nothing above `batch 1024` or `out_len 2048` was measured, yet the
  tables rank `cta` first above batch 320 and `out_len` 256, with rows to `out=32768`, `batch=32768`.
  Grid J (`out_len` 256/512 × `red_len` 64/128/256 × batch 2048/4096/8192) was not run.
* **The batch floor of 320 is the lowest enumerated threshold, not the lowest winning one.** Batch 288 wins all
  six cells in the band (one pass, `Trans`). Bracketing needs batch 272…320 in two passes, both spellings.
* **`out_len >= 768 && batch >= 128`** is measured and unrouted (about 18 cells at 2.26×–2.91×; grids H and I).
* **The `red_len` upper edge of 352 holds only for the smallest admitted `out_len`.** At `out_len 2048` the
  vendor is still dipped at `red_len` 384 and 448. A two-variable boundary needs a fit on four `out_len` levels.
* **The clause was fitted on device 0.** Only the lowest-footprint cell was re-measured on another device.
* **The L2-resident window above the body-5 gates is measured and not taken.** It needs a footprint term the
  table keys cannot express.
* **`17 <= out_len <= 31` on the NoTrans arm is unmeasured.** Body 4 declines it by arithmetic.
* **`complex<double>` transposed at short reduction is cleared, not solved.** 0.862 / 0.861 is not parity.
* **Coverage cannot confirm a shape.** `coverage.cc` keys on a power-of-two `shape_class` and is
  first-writer-wins. Coverage confirms reachability only.
* **The live capture is not verified.** The intended move is `vendor` → `cta` for cdouble, `transA != NoTrans`,
  `red_len` 64..352, `out_len >= 256`, `batch >= 320`. The body-5 pass was diffed (0 decisions removed,
  60 added, all gemv). The transcription reproduces the old decision at 100.00% of random off-grid points
  per (device, dtype, scenario) ([Phase 5, gemv](../design/flat-kernel-selection.md#phase-5-gemv)).
* **The tables are untimed.** No tuner sweep of gemv exists on either device. The sm_120 rankings are the
  sm_89 window carried over.

## Raw evidence

Raw data is kept at tag `perf-evidence/vendor-independence`. Retrieve any path with
`git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| vendor baseline, DRAM roof, cdouble dip and controls | `experiments/wp7_gemv/baseline/README.md`, `vendor_baseline{,_p2}.csv`, `refine.csv`, `close.csv`, `grid.txt` |
| parity grid, `m × n` map, batch ladder, refuted `n*batch` rule | `experiments/wp7_gemv/ab/README.md`, `ab_p{1,2}.csv`, `refine_c_p{1,2}.csv`, `batchdep_p{1,2}.csv`, `outelems_p{1,2}.csv` |
| `(out_len, red_len)` parity audit, blockers, `ncu` geometry, prize grid, clause search | `experiments/wp7_gemv/audit/README.md`, `parity_p{1,2}.csv`, `blockers_p3.csv`, `geometry.csv`, `mechanism.csv`, `prize_p{1,2}.csv`, `oos_p{1,2}.csv`, `clause_search.py`, `clause_report.txt`, `typecheck.csv`, `ortho_shapes.csv` |
| body 4: runtime vs `constexpr` `W`, parity ladder, break campaigns | `experiments/wp7_gemv/repair/README.md`, `parity_r{1,2}.csv`, `outlen_body{1,4}.csv`, `mechanism_body4.csv`, `breaks_kernel.py`, `breaks_route.py` |
| body 5: FP64-pipe mechanism, `W` tables, gates, skinny defect, odd `ld`, registers, link time | `experiments/wp8_gemv/README.md`, `ncu_precheck.csv`, `typecurve.csv`, `wchoice_p1.csv`, `wfine_p1.csv`, `plane_p{1,2}.csv`, `planeF_p{1,2}.csv`, `conjF_p{1,2}.csv`, `plane_cells.txt`, `skinny_p{1,2}.csv`, `skinny2_p{1,2}.csv`, `skinny3_p{1,2}.csv`, `resid_p{3,4}.csv`, `above_p{1,2}.csv`, `oddld_p1.csv`, `regs.csv`, `linktime.sh`, `breaks_body5.py` |
| cdouble window: fitting grids, clause scorer, unbracketed corners | `experiments/wp8_gemv/g6_fit_p{1,2}.csv`, `g6_fit2_p1.csv`, `g6_clauses.py`, `g6_score.py`, `g6_summary_p1.txt`, `g6_summary_2pass.txt`, `g6_summary_fit2.txt`, `g6_cells3.py` |
| routing gates: clause probe, admitted set, `preferred()` breaks, `route_diff` health | `experiments/wp8_route/clause_probe.cc`, `admitted.csv`, `breaks.py`, `after.sh`, `health.sh`, `gate_a.sh` |
| defects found and deliberately not fixed (ortho `Trans`, `cond`'s vendor bypass, lanczos's discarded gemm column) | `WP7_FILED_DEFECTS.md` |
| campaign narrative, gates re-run after the repair, the four shipped windows | `VENDOR_INDEPENDENCE_PLAN.md` |
