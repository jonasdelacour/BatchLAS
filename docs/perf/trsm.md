# TRSM {#perf_trsm}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · grids 2026-08-19/20 · tables 2026-10-04 (sm_89), 2026-10-11 (sm_120, deep)

Native batched `trsm` on the GPU: the `cta` (V1), `blocked` (V2) and `sg_left` kernels, the tables that
choose among them, and the numbers behind the shipped windows.

> **Warning:** the grids below were measured on a kernel that returned wrong answers until the group
> barrier was added on 2026-08-21. Read [open debts](#trsm-open-debts) before trusting any ratio here.

## Choices and routing {#choices-flat-selection-p33}

Vocabulary and key: `src/ops/trsm/choice.hh`, in tie-break order. `can_run` (`src/ops/trsm/trsm.cc`) is
correctness only. A masked batch is refused by every family. Every native family also requires
`d.is_gpu && !A.is_heterogeneous() && !B.is_heterogeneous() && order >= 1 && q >= 1 && A.batch_size() >= 1`.

| spelling | implementation | `can_run` beyond the common term |
|---|---|---|
| `cta` | `sycl_trsm::trsm_native_v1_dispatch` (V1) | `max_wg >= 32`, order <= `trsm_cta_max_n<T>()` (32) |
| `sg_left` | `sycl_trsm::trsm_native_sg_left_dispatch` ([sub-group Left kernel](blackwell.md#trsm-sub-group-left-kernel)) | `Side::Left`, `has_sg32`, order <= `kTrsmSgLeftMaxN` (32), `max_wg >= kTrsmSgLeftWgSize` (128) |
| `blocked` | `sycl_trsm::trsm_native_blocked` (V2) with the public `gemm` injected | `max_wg >= 32`, `trsm_blocked_available<T>()`, `trsm_cta_max_n<T>() >= 1`; no order term |
| `vendor` | `backend::trsm_vendor` | `d.has_vendor`, homogeneous A and B |

The key is `side:exact trans:exact order:log:2 q:log batch:log`. `q` is `B.cols()` for `Side::Left`
and `B.rows()` for `Side::Right`. ConjTrans folds to `T` (the kernels differ only in a conjugation).
uplo and diag are not keys. The last resort is `blocked`, then `vendor`. The CPU has no native
family. trsm takes no workspace. `A.batch_size() != B.batch_size()` throws `invalid_argument` in
`trsm_validate_params`. The coverage row maps `m = B.rows()`, `n = B.cols()`, `k` = triangular order.

### What the tables say

The grid starts at batch 128; smaller batches take the batch-128 rows. Provenance: `tuned/README.md`.

| table | source | first entry |
|---|---|---|
| `trsm.sm_89.txt`, every dtype | transcribed, untimed (`8b9adeb3`) | `cta \| blocked \| vendor` at order <= 32, `blocked \| vendor` above. `sg_left` in no row |
| `trsm.sm_120.txt` | deep (complex rows partly transcribed) | order <= 32: Left mostly `sg_left` / `cta`, Right `cta`; above 32: `blocked` first in most rows, `vendor` in the rest |

On sm_89, `sg_left` runs only when pinned, and the vendor only where no native family can (a
heterogeneous batch, a CPU queue). The sm_120 result is on the
[Blackwell page](blackwell.md#trsm-on-sm_120-result).

### Tuning knobs and environment {#tuning-knobs-and-environment}

* `BATCHLAS_TRSM_ROUTE` takes `auto`, `native`, `vendor`, or a spelling. `native` takes the first
  non-vendor runnable entry. A spelling the shape cannot run (`cta`/`sg_left` above order 32,
  `sg_left` on `Side::Right`) or an unknown word throws `invalid_argument`. `native` and `vendor` fall
  back to Auto with a warning when nothing of their class can run.
* `BATCHLAS_TRSM_OUTER_NB` is V2's outer block width (`trsm_outer_block`, `trsm_native.cc`), a tuning
  knob. Default 128 for `Side::Left` and `cta_max_n` (32) for `Side::Right`, rounded down to whole CTA
  blocks; 0 means unset. It is read per call from `settings().geometry.trsm_outer_nb` and never
  latched, so an A/B harness or a `ScopedEnvVar` test measures both arms.
* `BATCHLAS_TRSM_VARIANT` is read by nothing.

---

## Design: V1, V2 and the canonical fold {#design-v1-v2-and-the-canonical-fold}

**V1** (`cta`): one work-group per matrix and rhs slice, one work-item per independent solve. The
solution lives in registers as `T x[N]`; the canonical triangle is staged once into SLM and broadcast.
The 24 `(side, uplo, transA, diag)` cases fold into one recurrence through `canonicalise()`
(`src/ops/trsm/trsm_canonical.hh`, shared with `sg_left`), index map `rho(s) = fwd ? s : order-1-s`.
Buckets are N = 8, 16, 32.

**V2** (`blocked`): a host-side two-level driver. The outer level blocks at `OUTER_NB` and issues a
trailing GEMM; each outer panel is solved by an inner `nb = 32` loop of V1 calls. The V1/V2 crossover
at 32 is a capacity limit, not a tuned choice.

The V1 grid is `batch * ceil(q/WG)`. The work-group ladder walks `{256,128,64,32}` and takes the first
`cand` with `bs*ceil(q/cand) >= 4*CU` (`trsm_v1_ladder_wg`, `src/ops/trsm/trsm_native.hh`). On sm_120 it
also skips a rung that leaves over half its lanes idle ([ladder cap](blackwell.md#trsm-v1-ladder-cap)).

### The register gate and the CTA capacity {#the-register-gate-and-the-cta-capacity}

The gate: stack frame == 0, 0 spill bytes, and the per-sub-partition bound
`ceil(warps/4)*32*regs <= 16384` (`resident::sm89_fits`, a `static_assert` in `trsm_native.cc`;
[agent guide §11](../developer/agent-guide.md#11-gpu-kernel-design-facts-sm_89-mostly-general),
[LU page](lu.md#the-register-cap-that-binds-is-per-sub-partition)). Measure with
`scripts/register_probe.sh out.log '' batchlas_extensions_level3`, which replays the shared library's
device link. A per-TU `-Xcuda-ptxas -v` log reports "argument unused" and no spill whatever the kernel does.

| type | N=8 | N=16 | N=32 | regs*256 at N=32 |
|---|---|---|---|---|
| float | 44 | 76 | 114 | 29,184 |
| double | 59 | 101 | 153 | 39,168 |
| complex\<float\> | 50 | 86 | 148 | 37,888 |
| complex\<double\> | 74 | 138 | 226 | **57,856** |

All 24 kernels (4 types x 3 buckets x 2 sides) have zero frame and zero spill, so WG = 256 is the cap.
**Read the frame column, not the spill column:** N=64 has zero spill but a 256 B (float) / 512 B
(double) frame, i.e. `x[]` in local memory ([rejected](#rejected-the-n64-cta-bucket)).

### The `Side::Left` staging tile {#the-sideleft-staging-tile}

For `Side::Left`, thread `u` owns column `u`, so a warp's reads at step `s` are `ldb` apart. ncu,
float, n=32, q=1024, batch=512: load sectors/request 31.39 -> 5.13, store 32.00 -> 4.00, kernel
0.517 -> 0.145 ms (`Side::Right`, never staged: 0.141 ms). DRAM was already 0.85x of the analytic floor;
the defect is LSU transaction count, so wider loads aim at the wrong resource.

Staging is gated to the real types (`trsm_stage_left`; the profile above is float only). For complex
it moves `x[]` into local memory (`complex<float>` N=32 gains a 464 B frame, `complex<double>` 232 B)
and over-fetch caps the gain at 2x; with the gate, complex measures 1.00-1.01x. Tile height is 16 rows for `sizeof(T) <= 4`, else 8; row stride `NB_STAGE + 1` keeps the
read-out conflict-free.

### The two-level blocked driver {#the-two-level-blocked-driver}

With a single `nb = 32` level, every trailing GEMM has one dimension pinned at 32, capping intensity
near 16 flop/byte (float) against a 4090 balance of ~42. `OUTER_NB = 128, nb = 32` cuts B traffic at
n=512 from 5.7x to 4.0x of ideal. Float, worst cell per order, `vendor_ms/native_ms`
(`experiments/wp3_s13/nb_sweep.csv`):

| order | Left nb32 | nb64 | nb128 | nb256 | Right nb32 | nb64 | nb128 | nb256 |
|---|---|---|---|---|---|---|---|---|
| 128 | 1.18 | 1.10 | 1.20 | 1.17 | 1.00 | 0.96 | 1.00 | 0.98 |
| 256 | 0.75 | 0.78 | **0.87** | 0.75 | **1.01** | 0.94 | 0.83 | 1.01 |
| 512 | 0.58 | 0.74 | **0.76** | 0.75 | **1.07** | 0.92 | 0.82 | 0.91 |

Shipped: 128 for Left, 32 for Right. The side split is float-only evidence applied to all four types;
`complex<float>` is insensitive to the knob, `double` and `complex<double>` were never swept.

---

## The measured grid {#the-measured-grid}

Ratios are `vendor_ms / native_ms`; **>1 means native is faster**. RTX 4090, one card held exclusive,
warm-up before every timed iteration, `bench::pristine(B)` restored between iterations (trsm is in
place), legs differing only in `BATCHLAS_TRSM_ROUTE`, quoted only at saturation.

**Step 9** (`TrsmOrthoSizes` in `benchmarks/trsm_benchmark.cc`: n {8..256} x q {256,1024,4096} x batch
{128,512,2048}, all four types, both sides). Native won every saturated `double` (1.39-9.62x),
`complex<double>` (1.20-4.66x), `complex<float>` (1.01-21.91x) and float/Right (1.54-4.59x) cell.
Float/Left lost 10 of 16 (0.57-0.87x), removed by the staging tile and step 16. The per-type split
is why the tables are per dtype. `double` and
`complex<double>` rest on this grid alone ([debt 2](#trsm-open-debts)).

**Below batch 128** only an unrankable launch-overhead profile exists (`starved.sh`, batch {1,8,32}):
batch=1 loses for both real types at order >= 32 (0.229-0.852x), batch 8-32 wins except float at order
128 (0.740-0.810x), complex wins at every batch. sm_89 now runs native there ([debt 4](#trsm-open-debts)).

### The final grid after the routed trailing GEMM {#the-final-grid-after-the-routed-trailing-gemm}

Step 16 (`experiments/wp3_s16/baseline.csv`), orders 8-512, both sides, float and `complex<float>`
only. Worst clean cell (relative sd <= 10%) per order:

| | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| float Left | 1.60 | 1.72 | 1.69 | 1.65 | 1.42 | 1.25 | 1.21 |
| float Right | 1.62 | 2.37 | 2.23 | 2.01 | 1.60 | 1.23 | **1.00** |
| complex\<float\> Left | 1.05 | 1.41 | 2.64 | 4.69 | 11.21 | 16.62 | 51.70 |
| complex\<float\> Right | 1.01 | 1.03 | 1.35 | 1.90 | 8.49 | 15.31 | 19.64 |

223 of 224 clean pairs win; the loser (float/Right, order 512, q=256, batch=128, 0.978-0.983x) is the
smallest-work cell at that order and got no clause. Before step 16 V2 called `sycl_gemm::gemm_custom`,
bypassing gemm's routing, and float/Left lost 8 cells at orders 256-512 (worst 0.760x).
`trsm_native_blocked` now takes a mandatory `TrsmTrailingGemm<T>` and the `blocked` arm passes the
public `gemm`: n=512, q=1024, batch=512 float went 18.8 -> 11.19 ms against the vendor's 14.28 ms.
Every operand V2 hands GEMM is a sub-view carrying the parent's `ld`; native GEMM on those shapes ran
0.43-0.62x of cuBLAS ([the strided-ld defect](gemm.md#the-strided-ld-defect-and-the-routing-fix)).

**End-to-end through `ortho`** (route unset): `Side::Right` (step 9) 80 cells at 1.147-2.719x;
`Side::Left` (step 12) 80 cells at 0.986-2.385x. Neither has been re-run since ([debt 7](#trsm-open-debts)).

---

## Negative results {#trsm-negative-results}

| Alternative | Result | Verdict |
|---|---|---|
| Cooperative CTA solve, V3 (`experiments/wp3_s14/v3_cooperative_kernel.patch`) | Float Left worst cell at order 64 1.48 -> 0.39x, order 128 1.18 -> 0.80x, 256-512 unchanged; the traffic model ignored the serial critical path. Per-cell CSVs deleted before aggregation | Rejected |
| Diagonal-block inversion | Any residual constant above ~2 flips Chol2 from recovering to not | Rejected before building |
| Starvation guard `batch*q < 8*CU*32 -> vendor` | At batch=8, q=32 native wins 2.2-2.4x; the guard would hand every such cell back | Rejected |
| Fixing the strided-`ld` loss in `register_tiled_common.hh`, double-buffering, packing B | Those shapes never ran that file; the loss was a route gate sending them to a slower kernel. Fixed by gemm routing | Refuted |

### Rejected: the N=64 CTA bucket {#rejected-the-n64-cta-bucket}

Re-tested after the staging tile: float N=64 Left is 72 registers with a 456 B frame, Right 119 with a
256 B frame, both zero spill. There is deliberately no N=64 bucket; `smallest_bucket_ge` returns 0 above 32.

### Rejected: `OUTER_NB` of 128 for `Side::Right` {#rejected-outer_nb-of-128-for-sideright}

It regresses orders 256 and 512 from 1.01 and 1.07 to 0.83 and 0.82
([sweep](#the-two-level-blocked-driver)). Right's update is `C(q x nb)` and a wider `nb` shortens the
inner `k` below gemm's `k >= 128` gate of the time; not re-measured against today's gemm tables.

---

## Correctness findings

### The missing group barrier {#the-missing-group-barrier}

V1 staged the triangle into SLM with a lane-strided loop, then lane `s` read `sLc[tri_idx(s,s)]` (and
`sDiv[0]`), written by another lane, with no barrier. Fixed by one `sycl::group_barrier`
(`src/ops/trsm/kernels/trsm_native.cc`, commit `c7d6d91`, 2026-08-21). Deleted: max relative diff vs
vendor 6.05e+16, 127/128 items wrong; restored: 4.27e-07, 0/128. Vendor-free blocked potrf at n=1024,
batch=256 had returned `info != 0` for 61-75 items.

It hid because every trsm test ran wg=32, a single sub-group in lock step. The guard
`TrsmNativeBlocked.MultiSubGroupWorkGroupStagesItsTriangleCorrectly` goes through V2 at order 48 (final
V1 block in the N=16 bucket), q=976, batch=128, and was verified red with the barrier deleted.

### The bucket ladder that truncated {#the-bucket-ladder-that-truncated}

`smallest_bucket_ge` returned 64 for `n > 32` and the dispatch `default:` mapped it onto N=32, so an
order-33 solve solved the leading 32x32 system and left B's last row untouched. It now returns 0 and V1
throws (`TrsmNativeCta.OverCapacityThrowsRatherThanTruncating`). The facade caps order through `cta`'s
`can_run`, but V2 calls the direct entry on its diagonal blocks.

### Testing traps

* Blocked tests must exceed `OUTER_NB = 128` or the outer level never runs (`TwoLevelPanelStructure`
  uses 129, 256, 300, 384).
* Alpha is applied once: for blocks `i > 0` it arrives through the trailing GEMM's beta. The natural
  `beta = 1` is invisible to any `alpha == 1` test.
* Complex data with nonzero imaginary parts (`tri_fill`) is needed to see ConjTrans; the complex
  reciprocal uses Smith's form.
* The oracle is an independent multiply-back: netlib and cuBLAS fold the 24 cases the same way.
* `ctest -L blas -L ortho` is an AND and runs zero tests; use `ctest -L 'blas|ortho'`.

---

## Open debts {#trsm-open-debts}

1. **Every grid predates the barrier** (2026-08-21), and the sm_89 tables transcribe them. The only
   post-barrier trsm timings are the sm_120 tables ([Blackwell page](blackwell.md#trsm-on-sm_120-result)).
2. **`double` and `complex<double>` were last measured at step 9**, before the staging tile, two-level
   blocking and the routed GEMM. `complex<double>` has no later measurement on either device.
3. **No order above 512 is measured on sm_89.** The tables run to order 1024; larger orders take that row.
4. **Below batch 128 nothing is measured at saturation.** Float above order 32 at batch 8-127 and the
   real-type batch=1 cells (0.229-0.852x in the profile) now run native on sm_89.
5. **Every complex "vendor" ratio is native against another BatchLAS kernel.** `trsm_vendor` in
   `src/backends/cublas.cc` diverts both complex types to a sequential per-RHS SYCL substitution, citing
   unreproduced cuBLAS NaNs under USM interop; `cublasCtrsmBatched`/`cublasZtrsmBatched` are unreachable.
6. **complex<float> `Side::Right` at orders 8-16 (1.01-1.05x) is a roofline tie**: native runs at
   88.5-90.5% of DRAM peak.
7. **The end-to-end `ortho` A/B has not been re-run since step 12.**
8. **The residual strided-`ld` slope is unexplained**: routed gemm reaches 0.93x of cuBLAS on strided
   operands; the rest is exposed load latency on operand B.
9. **`MatrixView::operator()(Slice,Slice)` passes the parent pointer array.** V2 passes the parent's
   `ld` and `stride` at every sub-view.
10. **The suite reaches native trsm only with batch <= 5.** A change to a table's top rows needs an A/B
    through a real caller.
11. **The composed `posv` fallback is 1.5-5.6x slower under Auto legs than vendor-pinned legs**
    (`benchmarks/results/p2_posv_float.csv`, float, n 4-32, batch 8192-32768; `gesv` shows no gap). The
    non-`potrf` part is ~20x the vendor's and flat in nrhs. Hypothesis: native trsm at order ~17 costs a
    fixed amount per batch item independent of q, so the sm_89 `cta | blocked | vendor` rows at order
    <= 32 need a q term. Falsifier, not yet run: order 17, q in {1,4,16,64}, batch=16384, `cta` vs
    `vendor`, interleaved in one process.

---

## Raw evidence {#trsm-raw-evidence}

At git tag `perf-evidence/vendor-independence` (`git show perf-evidence/vendor-independence:<path>`):

| topic | path |
|---|---|
| step-9 grid, batch-floor profile, `Side::Right` ortho A/B | `experiments/wp3_s9/` |
| staging tile ncu profile and grid, `Side::Left` ortho A/B | `experiments/wp3_s12/` |
| two-level blocking, `OUTER_NB` sweep | `experiments/wp3_s13/{baseline-before,baseline,nb_sweep}.csv` |
| rejected V3 | `experiments/wp3_s14/` |
| routed trailing GEMM, isolated `ld` shapes | `experiments/wp3_s16/` |
| strided-`ld` re-diagnosis | `experiments/wp4_gemm_ld/` |

Sweep CSV GFLOPS uses `n^2 q` for all types (complex understated 4x); compare `avg_ms`. `baseline.csv`
columns are `type,side,route,n,q,batch,ms,sd_pct`, `sd_pct <= 10` is the cleanliness gate. Bench rows
are named `BM_TRSM_StarvedRight` so `--name` substring matching keeps them out of the saturated grid.
`ncu --kernel-name` matches the mangled name; use `--kernel-name-base demangled`.
