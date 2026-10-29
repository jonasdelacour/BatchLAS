# GEMV {#perf_gemv}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · measured 2026-08/09; sm_89 table transcribed, sm_120 table deep 2026-10-11

Native batched `gemv` for all four scalar types and all three `transA`: the families, the five
kernel bodies, the gates that pick a body, and the `complex<double>` transposed window. Box: RTX
4090, 72 MB L2, **~950 GB/s achievable DRAM roof**. Ratios are `vendor_ms / native_ms` (above 1:
native faster), except body-5 ratios, which are `body3_ms / body5_ms`.

## Families and kernel bodies

### The families and their `can_run` terms

Flat selection (@ref design_flat_selection) in `src/ops/gemv/gemv.cc`: first entry of the nearest
row of `tuned/gemv.<device>.txt` that `can_run` admits; last resort `vendor`, then `direct`
(`src/ops/gemv/choice.hh`).

| spelling | driver | bodies | `can_run` |
|---|---|---|---|
| `cta` | `sycl_gemv::gemv_native_cta` | 3, 5 | native term, `gemv_cta_available<T>()`, `d.is_gpu && d.has_sg32 && transA != NoTrans` |
| `direct` | `sycl_gemv::gemv_native_direct` | 1, 2, 4 | native term and `gemv_direct_available<T>()`; no GPU term (builds for `native_cpu`) |
| `vendor` | `backend::gemv_vendor` | — | `!A.is_heterogeneous()` |

The native term refuses a heterogeneous `A`, negative extents and `batch < 1`, and requires x and y
to match A in batch and length ([shape builder contract](#gemv-the-shape-builder-contract));
`m == 0 || n == 0` quick-returns. `has_sg32` comes from `sycl::info::device::sub_group_sizes` via
`Device::supports_sub_group_size`; `get_property(MAX_SUB_GROUP_SIZE)` returns `sub_group_sizes()[0]`
and is wrong both ways, and a `reqd_sub_group_size(32)` launch on a device without 32 aborts.

### complex<double>: Auto may now pick cuBLAS {#gemv-complex-double-vendor-route-change}

`can_run(Vendor)` no longer refuses `complex<double>` (known-defects #13, fixed), so every table row
that ranks `vendor` first runs cuBLAS. None of those rows is timed: `tuned/gemv.sm_120.txt` has 188
such cdouble rows and `tuned/gemv.sm_89.txt` 1360, all `transcribed`. A route diff of `gemv_tests`
and `spmm_tests` on sm_120 moved one decision (`m=64 n=32768 batch=2` NoTrans, `direct` to
`vendor`; passes). sm_89 was not re-checked. A cdouble retune decides these rows.

### The five kernel bodies

| body | kernel | family | shape | decomposition |
|---|---|---|---|---|
| 1 | `GemvDirectNKernel<T>` | `direct` | `NoTrans` | one work-item per output row |
| 2 | `GemvDirectTKernel<T>` | `direct` | `Trans`/`ConjTrans` | one work-item per output column (portable arm) |
| 3 | `GemvCtaTKernel<T>` | `cta` | transposed, GPU, sg 32 | one 32-lane sub-group per output, `shift_group_left` ladder |
| 4 | `GemvSegNKernel<T,W>` | `direct` | `NoTrans`, `out_len <= 16`, sg 32 | `W = 32/out_len` lanes per output |
| 5 | `GemvSegTKernel<T,W>` | `cta` | transposed, short reduction, sg 32 | `W` outputs per sub-group, `L = 32/W` lanes each |

The body is derived in the driver, not a family or `can_run` term (that would put a speed cutoff
in the correctness predicate), so trace and coverage carry only the spelling. To tell bodies apart
use `gemv_seg_trans_width_debug` (test-only) or a break that is red for one body only. All five use
zero local memory; the hand-rolled shuffle ladder beats `sycl::reduce_over_group` by 1.5–4.7× for
`double` and `complex<double>`.

### The cdouble window in the tables

Keys are `trans:exact out:log red:log batch:log`; `out`/`red` are the y/x lengths and swap with
`transA`; ConjTrans folds to `T`. The grid straddles every edge (`out` 255/256, `red` 63/64 and
352/353, `batch` 319/320). On sm_89 (transcribed), `cta` ranks first only for `complex<double>`,
`trans=T`, `64 <= red <= 352`, `out >= 256`, `batch >= 320`; every other row ranks `vendor` first.
The band is on `red_len`: a window written on `out_len` inverts. The deep sm_120 table ranks native
first in most rows, in every type. `GemvTranscribedTable.RowsHoldTheTranscribedPreference`
(`tests/gemv_candidates_tests.cc`) checks the sm_89 rows.

### The sub-route gates

Kernel-internal, not keys, families or `can_run` terms (`src/ops/gemv/kernels/gemv_native.cc`).

**Body 4** (`gemv_seg_width`): `W` is the largest power of two with `W*out_len <= 32`; `W == 1`
means body 1, so body 4 serves `out_len <= 16`. Needs sub-group 32 (false on `native_cpu`).

**Body 5** (`gemv_seg_trans_width`), three gates on `red_len()`:

| gate | float | `complex<float>` | double | `complex<double>` |
|---|---|---|---|---|
| 1: body 5 runs at `red_len <=` | 32 | 16 | 48 | 64 |
| 2: `W = 8` up to `red_len <=`, then `W = 4` | 24 | 16 | 32 | 32 |
| 3: floor on `out_len*batch` | `16*CU` (`W = 8`), `64*CU` (`W = 4`) | | | |

* **`BATCHLAS_GEMV_ROUTE`** takes `auto`, `native`, `vendor` or a spelling. A spelling the shape
  cannot take throws `invalid_argument` (`cta` on NoTrans is an error, not a reroute).
* **`BATCHLAS_GEMV_SEGT`** (`off` / `auto` / `2|4|8`) forces the body-5 width and bypasses all three
  gates. Read per launch, never latched (`gemv_segt_mode`).

## Evidence for each boundary

### The vendor baseline

cuBLAS `gemvStridedBatched` runs at 94–105% of the roof on 90 of 92 cells (all types, both
`transA`, `n` 32…2048, `batch` 64…65536): parity is the achievable result. The exception is
`complex<double>` transposed: 325 GB/s at 256×256 against 936–967 for the other types at matched
bytes, unchanged by `ld` padding; the native CTA body reads 936–941 GB/s for all four types.

### The cdouble window boundaries

The dip is a discrete kernel switch inside cuBLAS (at `out 512, red 128`: 930 GB/s at batch 256,
358–363 at 320–512), so the window needs its own batch edge.

| boundary | inside | bracketing non-winner |
|---|---|---|
| `red_len >= 64` | 2.41 at 64 | **0.9515** at 32 |
| `red_len <= 352` | 2.84 at 352 | **1.0304** at 384 |
| `out_len >= 256` | 2.32 at 256 | **0.9988** at 192 |
| `batch >= 320` | 63 cells, geomean 2.572, min 2.2261 | **0.9628** at batch 256 |
| `scalar == cdouble` | | float 0.9340, double 0.9722, cfloat 0.6644 |

The lowest-footprint admitted cell (84 MB against 72 MB L2) reads 1.87–2.07 on an idle card: read
the floor as ~1.87. Data: `experiments/wp8_gemv/g6_fit*`.

### The body-5 gates

**Gate 1** is where body 3 stops being short of the roof (GB/s, `out_len 2048, batch 512, Trans`;
bold is the last admitted rung):

| `red_len` | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|
| float | 485.5 | 655.9 | **832.6** | 912.9 | 922.7 | 931.0 |
| cfloat | **779.8** | 911.1 | 921.7 | 929.4 | 925.2 | 932.7 |
| double | 281.9 | 415.9 | 548.1 | **817.0** | 928.7 | 932.2 |
| cdouble | 224.2 | 329.2 | 434.3 | 456.2 | **707.9** | 932.5 |

Forcing `W ∈ {4,8}` past each gate measures 0.983–0.996. Admitted cells: `Trans` 83 cells geomean
3.286 (min 1.070), `ConjTrans` 36 cells 2.746, skinny `out_len` 1…64 30 cells 1.566; declined cells
0.977–1.009. **Gate 3** is needed: without it 16 cells at `out_len*batch <= 4096` lose (worst
0.891); a single `8*CU` floor leaves five at 0.976–0.998. Data: `experiments/wp8_gemv/`.

### The body-4 gate

Before body 4 the NoTrans `out_len < 32` family was 0.08–0.38× of cuBLAS (`cfloat out=1 red=2048
batch=512`: 99.2 against 1206.9 GB/s). `ncu`: sectors per load fall from 32.0 at `out_len 1` to 8.5
at 32; a warp effect, not a bytes effect. After: 24 cells min 0.60, median 1.16. **`17 <= out_len <=
31` is unmeasured**; the edge rests on arithmetic and a test pins it. Data: `experiments/wp7_gemv/repair/`.

## Negative results

### Kernel hypotheses refuted

| Alternative | Result | Verdict |
|---|---|---|
| Shuffle ladder is a fixed cost per output | double 547.5 vs cfloat 921.2 GB/s at identical bytes; `sm__pipe_fp64_cycles_active` 58–85% for double | Refuted: FP64 work on a 1:64 part; body 5 cuts it to `L*log2(L)` |
| Runtime `W` | Up to 2× slower than compile-time `W` for `out_len >= 8` (loop does not unroll) | Refuted |
| Sector floor `L*sizeof(T) >= 32` | float `red_len < 32`: 5.16–5.91× at `W = 8` vs 3.34–3.40× at `W = 4` | Refuted |
| Only FP64 types need body 5 | float/cfloat at `red_len 8`: body 5 worth 5.91× / 3.98× | Refuted; all four types emit it |
| `64 <= m <= 320 && n >= 256`, `n*batch >= 131072`, `A >= 256 MB` as the cdouble window | Admit cells at 0.36–0.97 | Refuted |

## GEMV: the shape builder contract

* **The native term of `can_run` is the only agreement check:** homogeneous `A`, `batch >= 1`, x and y
  matching A in batch, `X.size() == red_len`, `Y.size() == out_len`. The public entry validates
  nothing; a non-conforming call goes to `vendor`, or throws `NoRouteError` vendor-free.
* **`can_run` reads metadata only**, never `data_ptr()`.
* **Lengths swap with `transA`**; table keys use `red_len`/`out_len`, coverage `m`/`n` are A's stored extents.
* **`transA` is part of `coverage.cc`'s `variant_key`**; dropping it merges bodies 1 and 2/3.
* **Only `A` can be heterogeneous.**

## GEMV: correctness findings

Timed sweeps report `relerr` 0 because the A/B data are exact in any summation order; the evidence
is `tests/gemv_tests.cc`. Two contracts: **quick return** when `m == 0 || n == 0 || (alpha == 0 &&
beta == 1)` leaves `y` untouched and `A` is never read when `alpha == 0`; **no `__restrict__`**
(`ortho.cc` passes views into one allocation).

### Blind guards found and closed

The test fixture covers non-natural `ld`, `inc` and batch stride (`stride_pad`, as `ortho.cc`
passes), `ConjTrans`, complex data, 64 poisoned guard elements past `y` (body 5's tail), and
`Op::gemv` route assertions in `tests/gemv_candidates_tests.cc`. `SegTransCasesAreReachable` skips
with its numbers when `16*CU`/`64*CU` exceed the body-5 cases' `out_len*batch`.
`gemv_seg_trans_width_debug` uses the launcher's gate function. Breaks that stay green by design:
`flatten`, `segactive`, `segTtail` (mask is a UB requirement), `segTclampoff` alone, `segTlaunch`.

## GEMV: open debts

* Nothing above `batch 1024` or `out_len 2048` was measured, yet sm_89 rows rank `cta` first up to 32768.
* The batch floor 320 is the lowest enumerated threshold: batch 288 wins all six band cells (one pass).
* `out_len >= 768 && batch >= 128` is measured (~18 cells at 2.26–2.91×) and unrouted.
* The `red_len <= 352` edge holds only at the smallest `out_len`; at 2048 the vendor is still dipped at 384–448.
* `complex<double>` transposed at short reduction is 0.86×, cleared but not parity.
* The sm_89 table is untimed; the sm_120 cdouble `vendor` rows are transcribed.

## Raw evidence

Raw data is at tag `perf-evidence/vendor-independence` (`git show perf-evidence/vendor-independence:<path>`):
`experiments/wp7_gemv/{baseline,ab,audit,repair}/` (vendor baseline, parity grids, body 4),
`experiments/wp8_gemv/` (body 5, cdouble window fits), `experiments/wp8_route/` (routing gates).
