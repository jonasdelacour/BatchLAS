# WP3 step 16 — the trailing-update GEMM, and the sub-view leading dimension

## Result

Worst clean cell per order (relative sd ≤ 10%), vendor_ms / native_ms, >1 = native wins:

| | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| float Left | 1.60 | 1.72 | 1.69 | 1.65 | 1.42 | **1.25** | **1.21** |
| float Right | 1.62 | 2.37 | 2.23 | 2.01 | 1.60 | 1.23 | 1.00 |
| complex\<float\> Left | 1.05 | 1.41 | 2.64 | 4.69 | 11.21 | 16.62 | 51.70 |
| complex\<float\> Right | 1.01 | 1.03 | 1.35 | 1.90 | 8.49 | 15.31 | 19.64 |

**167 of the 168 measured cells win.** The last float/`Side::Left` losses are
gone — 0.86 → 1.25 at order 256 and 0.76 → 1.21 at order 512 — so
`preferred()`'s work threshold is removed and Side::Left is preferred at every
order and size. `Side::Right` improved as a side effect (order 128: 1.49 → 2.42
on cells clean in both runs), since it uses the same blocked driver above
order 32.

The one cell that does not clear parity is float / `Side::Right` / order 512 /
q=256 / batch=128, at **0.978–0.983×** reproduced over three repeats with a
longer measurement window. It is the smallest-work cell at that order (1.0 ms
total) and its neighbours win 1.30–1.38×. A 2% deficit on one cell is not worth
a fitted special case in the router.

## The cause

V2's trailing updates called `sycl_gemm::gemm_custom` directly. That is the
NATIVE kernel entry point and it **bypasses `RouteTable<Op::gemm>` entirely**, so
the updates always got the native GEMM whether or not it was the better choice.

Why that matters here and not in a GEMM benchmark: **every operand trsm hands
GEMM is a sub-view carrying its parent's leading dimension** — a 128-row `C`
with `ld = 512`. Measured on the six shapes V2 issues at order 512 (float,
q=1024, batch=512), `BATCHLAS_BENCH_LD_PAD` set to reproduce the real `ld`:

| shape | native (ld=rows) | native (real ld) | vendor (real ld) | vendor/native |
|---|---|---|---|---|
| outer m=128 n=1024 k=128 | 0.98 ms | 1.53 | 0.96 | **0.62×** |
| outer k=256 | 2.35 | 2.73 | 1.31 | **0.48×** |
| outer k=384 | 3.49 | 3.78 | 1.63 | **0.43×** |
| inner m=32 n=1024 k=32 | 0.248 | 0.406 | 0.235 | **0.58×** |
| inner k=64 | 0.356 | 0.680 | 0.335 | **0.49×** |
| inner k=96 | 0.487 | 0.887 | 0.426 | **0.48×** |

With `ld == rows` the native kernel is at parity on the inner shapes
(0.86–0.98×); with the real `ld` it is 2× off. **cuBLAS barely moves.** Strided
is the only case trsm ever issues.

The mechanism, from reading `src/sycl/gemm/register_tiled_common.hh`:

* `TileAStride = TileM+1` and `TileBStride = TileK+1` (:68-69) are both **odd**,
  so 16-byte alignment can never be proven and every shared fragment load
  degrades to scalar `ld.shared.b32`. `register_128x128.hh:20-26` names exactly
  this as the defect it was written to avoid.
* B is staged `[n][k]`, so a thread's values stride by `TileK+1` and cannot
  vectorize.
* The epilogue has `local_col` fastest-varying (:289-291), so adjacent lanes
  write columns 2 apart — **4096 B apart at ldc=512** — in 8-byte chunks, and
  the inner GEMM always runs `beta != 0`, making every one a read-modify-write.
* `can_use_*_fast_path` requires `is_contiguous_dense_matrix`, which every trsm
  sub-view fails by construction, so none of the aligned NN fast paths were
  ever reachable from this driver.

## The fix

`trsm_native_blocked` takes a `TrsmTrailingGemm<T>` — a callable with a
signature **identical** to both `gemm_custom` and the routed `batchlas::gemm`,
so neither side adapts. Empty means `gemm_custom`.

`src/dispatch/entry_points/level3.cc` passes the routed `gemm`. Injection rather
than an include keeps the kernel TU free of the dispatch layer, and:

* **the vendor-free build is unaffected** — `resolve_route`'s vendor-off
  fallback returns the native GEMM there anyway
  (`route_resolve.hh:60-63`). Confirmed: vendor-free `trsm_tests` 54 → 59
  passing (the +5 are step 14's new cases), failing set byte-identical.
* **no per-call cost** — cuBLAS GEMM uses `cublasGemmStridedBatchedEx`
  (`cublas.cc:118`), so unlike the trsm vendor path there are no pointer arrays
  to build and no device drain, which matters at 15 GEMM calls per solve.

Effect at n=512, q=1024, batch=512: native 18.8 ms → **11.19 ms**, against the
vendor trsm's 14.28 ms.

## What this corrected

**Step 14 concluded the inner blocking level did not matter**, because replacing
it wholesale with a cooperative solve changed nothing. That was wrong, and it
was masked: the cooperative solve was slow in its own way, so removing the inner
level and adding a slower diagonal solve cancelled out. The nsys profile shows
the inner GEMMs at **7.83 ms, 42% of the solve for 20% of the flops**.

**And my own first measurement here was misleading.** Run with `ld == rows` it
showed the inner shapes at 0.86–0.98× and I wrote that they were "not a routing
problem". At the real `ld` they are 0.48–0.58×. The `ld` is not a detail of the
benchmark; it is the whole effect.

## Data

`baseline.csv` — the full 8..512 grid after the change (`measure.py`, the same
driver and protocol as wp3_s13/wp3_s14). `outer-*.csv` / `inner-*.csv` are the
isolated GEMM shapes at `ld == rows`; `outerpad-*.csv` / `innerpad-*.csv` the
same at the real `ld` (`trailing_shapes.sh`, `outer_ld.sh`). `quick-*.csv` is
the first check of the formerly-losing cells; `recheck-*` the three repeats of
the one cell that still ties.

---

## ✱✱ CORRECTED by the step-17 pass (commit `3f0afbd`)

**The measurements on this page stand. The mechanism section above is wrong.**

The four bullets blaming `src/sycl/gemm/register_tiled_common.hh` describe a file
these shapes never execute. `select_kernel_variant` (`gemm_kernels.cc:509-511`)
routes them to `Tiled128x128RegisterK8` with `AlignedFastPath = **true**` in BOTH
columns of the table above; `can_use_128x128_fast_path` never tests contiguity,
only `ld%4` and a 16-byte base, which a strided sub-view satisfies.

ncu says every transaction counter is byte-identical between the two configs —
16.00 load sectors/request (the ideal), same DRAM sectors, same instructions,
119 registers, 0 spill. The loss is entirely exposed global-load latency at the
k-loop barrier, it belongs to operand **B** alone (A 1.003x, C 1.056x, B 1.552x),
it is a slope rather than a cliff, and it is beta-independent — so the
read-modify-write epilogue argument above is refuted directly.

Double-buffering the k-loop and packing B were both BUILT and measured to recover
nothing. What worked was routing. See the step 17 section of
`WP3_TRSM_SPEC_CORRECTIONS.md` and `experiments/wp4_gemm_ld/`.

One caveat on the numbers here: the padded operands were allocated uninitialized
while unpadded ones used `::Random`, so cross-pad ratios were confounded. Fixed
in `3f0afbd`; the reference cell moved 0.34%, so the effect is real.
