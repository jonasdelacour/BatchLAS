# GEMM {#perf_gemm}

> **Status:** current · RTX 4090 (sm_89) and RTX PRO 6000 Blackwell Max-Q (sm_120, threadripper02), CUDA 13.2, /opt/dpcpp-cuda

gemm selects through flat kernel selection (@ref design_flat_selection). This page lists the choices, what each
`tuned/gemm.*` table ranks first, the measurements behind each window, and per kernel the facts the code comments
cite. `tuned/gemm.sm_89.txt` is transcribed from the old router (`424a45bc`) and untimed. `tuned/gemm.sm_120.txt`
is deep-measured for float and complex<float> (2026-10-21); double and complex<double> keep transcribed rows (FP64
is not tuned on 1:64 parts, @ref dev_agent_guide §10).

## GEMM: the sm_120 cuBLAS gap campaign (2026-10)

Native gemm against cuBLAS on sm_120, every cell where both were timed. *Losing*: native > 1.05× vendor; *excess*:
summed native − vendor ms over losing cells; *geomean*: native / vendor over all timed cells (< 1 is faster). Every
row but the last is a coarse pass; the last is the deep shipped table (2.2-2.7× the cells), so compare its losing
*share*.

| stage | float losing / excess / geomean | complex<float> | complex<double> |
|---|---|---|---|
| before PR 188 (deep table) | 2685 / 554 ms / 1.074 | 3270 / 1467 ms / 1.061 | 1652 / 6386 ms / 1.000 |
| PR 188 | 1424 / 235 ms / 1.006 | 2312 / 337 ms / 0.970 | — |
| + WP-J, K, L, N2, Q | 688 / 111 ms / 0.960 | 623 / 209 ms / 0.924 | 44 / 40 ms / 0.794 |
| + WP-O, T, F | 208 / 27 ms / 0.869 | 615 / 66 ms / 0.914 | 3 / 3 ms / 0.777 |
| + WP-V, U | 210 / 24 ms / 0.861 | 209 / 11 ms / 0.862 | deferred |
| + WP-W, X, Y (deep, shipped table) | 242 of 5909 (4.1%) / 15 ms / 0.837 | 354 of 11471 (3.1%) / 13 ms / 0.858 | deferred |

Left in the shipped table (median loss 1.07-1.08×): float strided skinny shapes on the wide tiles (TT/NN/TN/NT, up
to 1.22×) and complex<float> NT/NC strided mid shapes on the 128×128 and 64×64 wide tiles.

**Protocol** for every sm_120 number below unless a section says otherwise: threadripper02, one GPU,
`batchlas_tune --mode time`, beta = 1, every arm verified, 4-16 reps × 2 passes in alternated arm or build order;
ratios are native / cuBLAS in the same run. Passes agree within 3% unless noted; under 5% a difference is a screen,
not a win. Sanity anchors: vendor SGEMM 45-48 TFLOP/s at 512³ (near 80 means TF32), DGEMM below ~1.45 TFLOP/s. The
table's own measurements are `benchmarks/results/tuning/ledger/gemm.sm_120.jsonl`; per-work-package A/B output was
not committed. Work packages with no per-cell A/B (WP-A, G, B, I, K, L, J, the skinny and small-complex rows) were
built while no GPU was free; the deep table times them as candidates.

## GEMM: what ships

### Choices (flat selection, P3.4)

The vocabulary is `src/ops/gemm/choice.hh`, in tie-break order. `src/ops/gemm/gemm.cc` validates the views, calls
`select::run` and launches one kernel from `src/ops/gemm/kernels/gemm_kernels.cc`. `can_run` is correctness only:
every register and wide tile hard-wires its transpose form, so a form it does not instantiate is refused there (a
silent wrong answer otherwise).

| spelling | launcher | scalars | `can_run` beyond the common term |
|---|---|---|---|
| `direct` | `gemm_direct`, one work-item per C element | all | `max_wg >= 64` |
| `tiled` | `gemm_tiled`, 16×16 shared tile | all | `max_wg >= 256` |
| `small` | `gemm_small`, `small_batched.hh` | all | `small_fits`: `max(m,n,k) <= small_max_dim<T>()` (64; 32 for 16-byte scalars), `max_wg >= small_wg` (128; 144 or 196 for float NN 33..56), sub-group 32 except on the float NN tiled leg, `small_slm_bytes` within `slm_budget_for(wg)` |
| `reg:m=..:n=..:k=..:u=..` | `gemm_reg`, register-tiled | float | a `reg_configs` entry, `reg_form` (the form is instantiated), `max_wg >= threads()`, `slm_bytes()` within `slm_budget_for(threads)` |
| `wide:m=..:n=..:k=..` | `gemm_wide`, 16-byte-granule tiles | all (`wide:m=128:n=128:k=8` complex<float> only) | a `wide_configs` entry, `wide_serves<T>` (a complex form listed exactly; a real `T` served by the C build), `max_wg >= threads()`, `wide_slm_bytes<T>` within `slm_budget_for(threads)` |
| `splitk:m=..:n=..:k=..:s=..` | `gemm_splitk`, a wide host tile over s k-slices plus a reduction ([WP-L](#gemm-wp-l-global-split-k-over-the-transposed-wide-tiles)) | all | s ∈ {2, 4, 8}, `s <= max_groups_yz`, a `splitk_host` row serving the form, the host's work-group and local-memory terms, ldw × n ≤ INT_MAX, partials ≤ `scratch_budget / batch` |
| `vendor` | `backend::gemm_vendor` | all | `has_vendor` and no batch mask |

The common native term: the device is a GPU or has no vendor (`natives_run_on`: a CPU queue keeps the host BLAS,
maintainer decision), `native_serves_precision`, m, n, k > 0, batch ≥ 1, no heterogeneous operand. Every native
honours a batch mask. Candidate counts (`candidates<T>()`): float 43, double 30, complex<float> 31, complex<double>
30, each with 18 `splitk` (six hosts × three s).

#### Batch above the grid-z cap {#gemm-batch-chunking}

`direct`, `tiled`, `reg` and `wide` put the batch in SYCL dimension 0 = CUDA grid z (65535, `kMaxGridBatch`).
`launch()` runs them on consecutive chunks of at most `max_groups_yz` items, and `splitk` on chunks of
`max_groups_yz / s` (its grid is batch × s); `small` is 1-D. Chunks are disjoint strided sub-batches, and an
out-of-order Queue is drained before the last Event returns. Tests: `GridCeilingIsServedByBatchChunking` and
`SplitKGridCeilingIsServedByBatchChunking` (at, one past and several chunks past the cap).

**Precision** (`native_serves_precision<T>`). `Default` and the scalar's own precision (F32 for float and
complex<float>, F64 for the double scalars) are exact on every native. F16, BF16 and TF32 on a single-precision
scalar only permit a less accurate product: a vendor build sends them to the vendor, a vendor-free build answers at
full precision. Any other pair is refused everywhere. Test: `ComputePrecisionContract`.

**Forms**, after the real-scalar C→T fold:

| choice | forms instantiated |
|---|---|
| `direct`, `tiled`, `small` | any |
| `reg:m=32:n=32:k=8:u=1`, `reg:m=64:n=64:k=8:u=1`, `reg:m=128:n=64:k=32:u={2,4}` | NN |
| `reg:m=64:n=64:k=16:u=1`, `reg:m=128:n=32:k={16,32}:u=1`, `reg:m=128:n=128:k=8:u={1,4}`, `reg:m=256:n=128:k=8:u=4`, `reg:m=128:n=256:k=8:u=4` | NN NT TN TT |
| `reg:m=128:n=64:k=16:u=1` | NT TN TT |
| `reg:m=32:n=128:k=16:u=1` | NN TN TT |
| `wide:m=64:n=64:k={8,16}`, `wide:m=32:n=32:k=16`, `wide:m=16:n=16:k=16` | all nine complex; real NN NT TN TT |
| `wide:m=128:n=128:k=8` | complex<float> only, all nine |
| `wide:m=128:n=32:k=16`, `wide:m=32:n=128:k=16` | complex NC NT TN CN; real NT TN TT |
| `wide:m=64:n=32:k=16`, `wide:m=32:n=64:k=16` | complex NN NT NC TN CN CT; real NN NT TN TT |

`WideCfg::forms` lists the complex instantiations, `WideCfg::real` the N/C builds a real scalar compiles, and
`WideCfg::scalars` the scalars ([WP-T](#gemm-wp-t-tt-tc-and-cc-on-the-small-transposed-tiles),
[WP-O](#gemm-wp-o-a-128x128-complex-float-tile-on-wide-v2)). Split-k hosts inherit their row's forms.

Derived inside the launcher, never a field or `can_run` term: the transpose instance, the aligned vs predicated
leg ([the strided-ld fix](#the-strided-ld-defect-and-the-routing-fix)), the `small` bucket, TR/TC/stages, and the
copy-async legs. A launcher throws on a config or form it does not instantiate;
`GemmCandidates.TransposedPinOnAMissingInstantiationThrows` and `CanRunEqualsLaunch` hold the two together.

**Selection** is the first runnable entry of the nearest row of `tuned/gemm.<device>.txt`, keyed `ta:exact tb:exact
layout:exact m:log n:log k:log batch:log` (C folds to T for a real scalar; `packed` = A, B and C contiguous with
16-byte bases, else `strided`; every log key weight 1). The last resort is `direct`, then `vendor`. gemm takes no
caller workspace (split-k leases its partials from the Queue). A heterogeneous batch is split into homogeneous items
before selection, and each item chooses. Every library gemm (the blocked drivers' trailing updates, the level-3
`expand` and `fold` families) calls the public `gemm`, so no caller reaches a kernel the table did not choose.

### Pins and environment

* `BATCHLAS_GEMM_ROUTE`: `auto`, `native`, `vendor`, or a spelling (`reg:m=128:n=128:k=8:u=1`, `wide:m=64:n=64:k=16`,
  `small`). A bad pin throws; old kernel names (`128x128x8`, `tiled16`) and router words (`register_tiled`, `sycl`)
  throw (`GemmCandidates.UnknownAndDeletedPinsThrow`).
* `BATCHLAS_GEMM_WIDE_STAGES`: `2` forces the register-prefetch leg of the wide kernels, `3` admits their copy-async
  legs from sm_80 (default: sm_120 only). A test and A/B switch.
* `BATCHLAS_GEMM_VARIANT` and `BATCHLAS_GEMM_SYCL_KERNEL` are read by nothing.
* `BATCHLAS_SELECT_TRACE=1` prints key, choice and runner-up; the coverage `reached` row's `chosen_algo` is the
  spelling; `BATCHLAS_KERNEL_TRACE=1` is the only place the derived leg shows (`_aligned`, `_s3`, `_async[_vec]`).

## GEMM: what the transcribed sm_89 rows rank first

Counts from `tuned/gemm.sm_89.txt` (every row `transcribed`, run `transcribed-424a45bc`): the first entry, and the
first non-`vendor` entry (what `native` and a vendor-free build run). The real tables have edge rows at batch 1, 63
and 64 and, for double, k = 1 and 2; the complex tables have batch 128, 2048 and 32768 only.

| dtype | rows | ranked first | first native entry |
|---|---|---|---|
| float | 11226 | `vendor` 11154; `small` 72 | `tiled` 5244, `reg:m=128:n=128:k=8:u=1` 2172, `small` 1392, `reg:m=128:n=32:k=32:u=1` 1068, `direct` 792, others 558 |
| double | 12672 | `tiled` 6956, `vendor` 5440, `direct` 256, `wide:m=64:n=64:k=16` 20 | `tiled` 12078, `direct` 564, `wide:m=64:n=64:k=16` 30 |
| cfloat, cdouble | 7464 each | `vendor` 7464 | `tiled` 3972, `wide:m=32:n=128:k=16` 1146, `wide:m=128:n=32:k=16` 1146, `wide:m=64:n=64:k=16` 1014, `direct` 186 |

| rows | first entry | window |
|---|---|---|
| double, batch ≥ 64 and k ≥ 2 (7232) | native: `tiled`, `direct` (NN max ≤ 24, transposed ≤ 32), `wide` (NN packed squares 256-1024) | [Double](#double-the-only-fully-native-window), [the wide scalar kernel](#the-wide-scalar-kernel) |
| double, batch 1 or 63, or k = 1 (5440) | `vendor` | the `batch >= 64` floor and the `k >= 2` boundary |
| float NN squares ≤ 48, batch ≥ 64 (72) | `small` | [The small tiled kernel](#the-small-tiled-kernel) |
| every other float row | `vendor` | [Float NN at max_dim 32](#float-nn-at-max_dim-32) |
| float NN, max ≥ 128, k ≥ 8, min(m,n) ≥ 64, and (min(m,n) ≥ 128 or k < 128) (2172) | native `reg:m=128:n=128:k=8:u=1` | [the strided-ld fix](#the-strided-ld-defect-and-the-routing-fix) |
| float transposed, m ≥ 128, n ≥ 32, k ≥ 128 (1068) | native `reg:m=128:n=32:k=32:u=1` | 0.34-0.55× of cuBLAS, so vendor first |
| complex NN, min(m,n,k) ≥ 32 (1014 per type) | native `wide:m=64:n=64:k=16` | [the CTA count gate](#the-cta-count-gate-for-complex) |
| complex NC m ≥ 128, n ≥ 32, k ≥ 8; CN m ≥ 32, n ≥ 128, k ≥ 8 (1146 each) | native `wide:m=128:n=32:k=16`; `wide:m=32:n=128:k=16` | [Wide-scalar transposed tiles](#wide-scalar-transposed-tiles) |
| every complex row | `vendor` | [Complex is refused](#complex-is-refused) |

The rows are exact on the grid and approximate off it: against the old selector the first native entry differs on
~6.7% of random off-grid double shapes and ~31.7% of float ones. The complex grid has no batch below 128, so a
batch-8 complex NN call takes the batch-128 row, whose native walk ranks `wide` where the CTA ladder measured losses.

## GEMM: what the sm_120 deep rows rank first

Counts from `tuned/gemm.sm_120.txt` (2026-10-21; 22203 deep rows, 14314 transcribed). Double (6396 rows) and
complex<double> (2480) are all transcribed; double ranks `vendor` 4044, `tiled` 2228, `direct` 120, `wide` 4;
complex<double> ranks `vendor` everywhere.

| dtype, deep rows | ranked first |
|---|---|
| float, 10368 | `wide` 5133 (64×64×16 2285, 64×32 759, 32×32 722), `reg` 4272 (128×128 u1 2097, u4 1014, 256×128 799, 128×256 154), `vendor` 609, `splitk` 285, `direct` 51 |
| complex<float>, 11835 | `wide` 10714 (64×64×16 4181, 128×128×8 1503, 32×32 1138, 64×32 944, 16×16 712), `vendor` 694, `splitk` 364, `direct` 37 |

## GEMM: evidence for each boundary

The sm_89 windows, RTX 4090, 2026-09 (raw data under [raw evidence](#gemm-raw-evidence)). A timed table replaces a
row only after a `tools/tune` sweep re-brackets these edges.

### Double, the only fully native window

All double rows with batch ≥ 64 and k ≥ 2 rank native first; no squareness, form or size term. Square NN, batch
512 (4096 for n ≤ 32), native / cuBLAS: n = 4-24 `direct` 2.5-4.5×; n = 32-200 `tiled` 1.04-1.90× (minimum at 64);
n = 256-512 `wide` 64×64 1.13×. Transposed squares 1.01-1.12×; non-square demand shapes 36 of 36 win
(1.04-1.46×); the ratio is flat or rising with batch (64 → 8192). Upper bound: none measured above 2048³; cuBLAS
DGEMM sits at 78-87% and native at 88-98% of the FP64 ceiling at every size from 4 to 2048.

| k (512×512×k) | 1 | 2 | 3 | 8 | 16 |
|---|---|---|---|---|---|
| native / cuBLAS | **0.49×** | 1.64× | 1.62× | 1.49× | 1.09× |

k = 1 is the only losing double shape (cuBLAS has a rank-1 path at β = 0). The `direct`/`tiled` boundary at 24:
n = 24 b4096 `direct` 1.64× of `tiled`, n = 25 a wash (0.99-1.02), n = 28 `tiled` 1.04×, n = 32 `tiled` 1.29×; 24
rather than 25 because 25 is inside the spread.

### Float NN at max_dim 32

Square NN, batch 512, native / cuBLAS (β = 0, 1): n = 32 1.03×, 1.08× (the last win); **n = 33 0.92×, 0.96×** (the
bracketing loss); 48 0.58×; 127 0.36-0.44×. The small tiled kernel later took 33..48 to 1.15-2.3×
([The small tiled kernel](#the-small-tiled-kernel)). NN 128..512 measured 0.39-0.98× and every transposed cell over
n = 64..768 0.23-0.55×, so every other float row ranks `vendor` first.

### Complex is refused

All complex rows of `tuned/gemm.sm_89.txt` rank `vendor` first: the native complex kernels ran 3.2-7.1× slower than
cuBLAS when the window was set (cdouble 0.39 vs 1.24 TFLOP/s; cfloat 6.6-6.9 vs 45-49), and the transposed wide tiles
do not beat cuBLAS on the drivers' ragged shapes. A timed sm_89 complex table would reopen it.

### The CTA count gate for complex

The old selector admitted complex NN to `wide:m=64:n=64:k=16` on `min_dim >= 32 && ctas >= kMinCtas` (64 cfloat, 128
cdouble; `ctas = ceil(m/64)·ceil(n/64)·batch`). The crossover against `tiled` is a work-group count, not a batch:
cfloat `ctas >= 64` admits 26 cells (worst 1.08×, no loss); cdouble `ctas >= 128` admits 24 (worst 1.08×). Bracketing
losses: cfloat 33×61×33 b32 (32 CTAs) 0.75×; cdouble 33×61×33 b64 (64 CTAs) 0.93×; 16³ 0.71× / 0.28× (the
`min_dim >= 32` floor). In the tables it is the 1014 NN rows per type with min(m,n,k) ≥ 32; the batch half is not
encoded ([open debts](#gemm-open-debts)).

### The strided ld defect and the routing fix

`trsm`, `getrf` and `geqrf` hand gemm sub-views carrying the parent's leading dimension (a 128-row C at `ld = 512`).
On the six shapes `trsm` issues at order 512 (float, batch 512) the old native path ran 0.43-0.62× of cuBLAS at the
real `ld` and 0.86-0.98× at `ld == rows`; cuBLAS barely moves. ncu (128×1024×128 b512, pad 0 vs 384): every
transaction counter identical, time 917 → 1493 µs, DRAM 89% → 55%; 96% of the penalty is operand B. No mechanism
below L2 is established.

The fix was routing, not a kernel: the 128×128 fast-path predicate had doubled as a routing gate. A shape-only gate
gives geomean 1.74× (pad 0) and 1.75× (pad 384) over 12 shapes, native 0.58× → 0.99× of cuBLAS at `ld == rows` and
0.54× → 0.93× strided, and removes the `ld % 4 != 0` cliff. Each bound has a counterexample: `mn_min >= 64`
(32×1024×32 a wash), `mn_min >= 128` when k ≥ 128 (64×64×512 0.77/0.69), `max_dim >= 128` (64³ 1.02×), `k >= 8`. The
gate survives as data: `reg:m=128:n=128:k=8:u=1` is the first native entry of exactly the 2172 sm_89 float NN rows
that satisfy it; `RegCfg::aligned_leg` names the unpredicated configs and the launcher picks the leg per call.
Callers also pack where they can (geqrf packs V at `ld = mp`).

## GEMM: register-tiled float kernels (`reg`)

### The 128x128 float kernel

`register_128x128.hh`: 128×128×8 tile, 8×8 accumulators, 256 threads. Its SASS inner loop matches nvcc's for the
same body (512 FFMA, 32 `LDS.128`, 2 `BAR.SYNC`, zero spill), and SYCL runs at 98.7-100.1% of the CUDA build. Four
levers, each measured alone on sm_89 (512³ b1024 unless noted):

1. **Prefetch registers are `ext_vector_type(4)`**, not a 16-byte struct (LLVM sinks a struct's load to its store):
   0.91× → 1.04× of cuBLAS.
2. **Lane swizzle** (m on lane bit 0, n on bit 1): shared-load wavefronts 805M → 537M (cuBLAS 549M).
3. **`L2::128B` on the k-contiguous loads**: 256³ b4096 0.80× → 1.00×, neutral elsewhere.
4. **[The staged epilogue](#the-staged-epilogue)** for k ≤ 64 on the aligned leg, every k on the predicated leg.

Launch bound two groups per CU (128-register cap) is a guard: one group 127 registers, same speed; three groups 80
registers + 1.4 KB spill, 8.8× slower. Paired against the previous kernel the rework gained 1.09-1.30× (128³ to
2048³); the predicated leg gains 1.67-2.21× over the generic one (n = 160-1056), up to 4.39× at unaligned `ld`, but
stays 0.72-0.84× of cuBLAS. The box is power-limited (455-475 W), so compute-bound cells move ±5% between runs. On
sm_89 the float rows above 48 keep `vendor` first: 384 β=1 (0.99-1.02) and the predicated leg at 512 ld+1
(0.89-0.94) do not clear the bar.

### GEMM: the 128x128 kernel in every real transpose form

`Gemm128x128Body` takes `OpA, OpB` (previously NN only; NT/TN/TT ran 19-27 TFLOP/s at 1024³ b128 against NN's 57).
Only the loaders differ: an m/n-contiguous operand (A, Bᵀ) is one 16-byte shared store per 128 contiguous elements;
a k-contiguous one (B, Aᵀ) is two lanes per column with four transposing scalar stores and `L2::128B`. Two details
hold the 128 cap on the Bᵀ loader: its pointer advances by `TileK·ldb` (64-bit) per slab instead of `k0·ldb` (an
`int` overflows past 2³¹ elements; a 64-bit product spilled 20 B), and its predicated bound is one live `int`. The
optional 128×64 config row was rejected: on the generic kernel it got a 288-byte stack frame in all forms.

### GEMM: WP-G transpose-aware scalar staging

`register_tiled_common.hh` (every `reg` row but 128×128 and the large tiles). With an odd `ld` or base each operand
is staged one element per lane, lane-fastest along the operand's contiguous index (k for Aᵀ, n for Bᵀ; previously m
and k whatever the form, one sector per lane). The B fragment read `sb[(4c + j)·s + t]` puts columns c and c + 8 on
one bank for any stride (4·s·8 ≡ 0 mod 32), so no stride pad fixes it; `b_slot` XORs the k row with the column's
32-column group, inside the row's aligned 4-block, no memory. Test: `RegScalarStagingEveryFormAtSaturatingBatch`.

### GEMM: WP-I, an unpredicated interior loop on the 128x128 predicated leg

A work-group whose tile lies inside C (uniform over the group) runs a second copy of the mainloop with no row or
column predicates; edge groups and the k-tail slab keep the bounds. At the 128 cap every single-loop variant (per-slab
branch, hoisted flag, `kfull ||` term) spilled 4-48 B; the shipped form launches through `Gemm128x128Inlined`
(call-site `always_inline`) with the test written edge-first, zero spill. ptxas is chaotic here: re-probe
(`scripts/register_probe.sh`) after any edit. Test: `Reg128x128PredicatedInteriorAndEdgeGroupsAtSaturatingBatch`
(NaN in the k tails: a finite poison times a correctly zeroed operand is a bit-exact −0 and stays green). An interior
test that admits an edge group only over-reads, so no numeric test sees it.

### GEMM: WP-N2, the u4 stack frame and zero-filled predicated copies

`reg:m=128:n=128:k=8:u=4`: the aligned leg is u1's kernel; the predicated leg stages 4 local slabs by zero-filled
4-byte `cp.async` (`copy_async.hh`), 3 in flight; any `ld`, any base, the k tail needs no branch. Measured 2026-10-19,
sm_120, 8 reps × 2 passes.

* **The 23-33× defect** (WP-N: NN 1024³ b128 377 ms against u1's 16.6): the body ran out of line on a stack copy of
  the functor, and each `asm volatile(... : "memory")` reloaded every member through `this`. The always-inline
  wrapper removed it; `Stages > 2` always inlines.
* **The aligned cp.async leg is deleted**: it never beat the register prefetch (NN 1024³ 1.145× vs u1's 1.043×).
* **The cap needs `.reqntid`**: the launch bound alone left staged NN/TN/TT at 155-159 registers (2.2-2.4× vendor);
  `reqd_work_group_size(1, 1, 256)` holds 128 (it lives in `register_128x128.hh`, not `kernel_attrs.hh`, whose hash
  is in every op's table).
* **Pointer bump per slab**, not `ptr + k0·ld` per copy (that spilled 52-72 B); staged NN still spilled 12 B, so u4
  NN takes the direct epilogue.

Predicated cells (`ld = m + 1`): u4 wins 8% on NN 256×512×128 b2048 (1.316× vs u1 1.425×) and 4% on NT 1218×128×96,
ties or loses elsewhere (NN 384×384×48: wide 1.136×, u4 1.437×). A zero-filled copy names the operand's batch base,
not its computed address (+0.0-0.7%, [WP-W](#gemm-wp-w-a-256x128-float-tile-with-four-copy-async-stages)). Test:
`Reg128x128CopyAsyncPipelineEverySlabCountAtSaturatingBatch` (slab counts 1-17, NaN k tails 5/37/69).

### GEMM: WP-F, float mid-size parity on sm_120

Target: ~600 float cells at m, n ≥ 128 losing 1.08-1.38×, mostly strided at k = 64-128 (2026-10-20, 188 SMs). ncu:
the cells are DRAM-bound and both sides move the same bytes; two defects of ours, neither the tile size:

* the predicated leg's direct epilogue stored scalars four rows apart, 3.7× the store sectors of a whole-column store
  (151M vs 41M). Shipped: the predicated leg (u1 and u4) stages C at every k; the 16 priors of a pass load after the
  stage write (its accumulators are dead) and before the barrier, all in flight. Staging with inline priors fixed
  the sectors but not the time (2.614 ms, DRAM 59%); batched priors 1.923 ms, DRAM 91.8% (vendor 1.819). The aligned
  leg keeps `kStagedEpilogueMaxK = 64`.
* u4's k-contiguous operand read 16 rows' sectors per 128 useful bytes (343M vs 112M). Shipped: k fastest, 8 lanes
  per 32-byte row segment, shared stride padded by 4 (132) so the transposed writes are conflict-free;
  `RegCfg::slm_bytes` 33,792 B.

| cell (b128 unless noted) | parent best / vendor | new best / vendor |
|---|---|---|
| NN strided 256×512×78 b2048 | u4 1.345 | u4 1.081 |
| TN strided 2048×1024×96 | 1.361 | u4 1.067 |
| NT strided 362×1024×96 | u4 1.171 | u4 0.994 |
| control NN packed 512³ b2048 (aligned, untouched) | u1 1.076 | u1 1.063 |

Screen of 28 losing cells: median 1.110 → 0.954. No k ceiling: staging wins or ties at k = 256-2048 except u1 TN at
k ≥ 1024 (1.6-3.8% slower, a route nothing picks). beta = 0 stores `alpha · acc` alone (`alpha · acc + 0 · 0` turned
−0 into +0). On sm_89 the hoisted `32·ld` copy stride spilled 16 B in staged u4, so `issue` rebuilds it behind an
empty `asm volatile`. Rejected: u4's pipeline on aligned data (NN 512³ 1.146 vs u1 1.070). Test:
`Reg128x128PredicatedStagedEpilogueAtEveryK`; the u4 copy's row bound is not covered by values.

### GEMM: WP-U, the u1 128x128 launch bound enforced

`Gemm128x128Body` and `Gemm128x128Inlined` carry `reqd_work_group_size(1, 1, 256)`, so ptxas holds the two-group cap
([known defect 20](../design/known-defects.md#defect-20-batchlas_launch_bounds-emits-no-maxntid-so-ptxas-ignores-the-bound);
0 ignored bounds in `libbatchlas_sycl`). u1 sm_120 registers 130-139 → 128. The aligned staged epilogue re-reads
`%tid.x` behind `asm volatile` (else TT spilled 32 B). Timed 8 reps × 2 passes × 3 runs, pinned u1, new / parent: mid
k (64-256) 0.85-0.96; NN at k ≥ 1024 1.02-1.03 (the 139-register mainloop ran better at one group). The k edge, NN
packed 512×512 b128: k = 512 0.963-0.996, k = 768 1.006-1.018, k = 1024 1.020-1.035. Shipped.

### GEMM: WP-W, a 256x128 float tile with four copy-async stages

`reg:m=256:n=128:k=8:u=4` and `reg:m=128:n=256:k=8:u=4` (`register_256x128.hh`, NN NT TN TT), 2026-10-20. Target: 95
cells on u4 at 1.05-1.17× vendor, where cuBLAS runs `cutlass_80_simt_sgemm_256x128_8x4` / `128x256_8x4_tt`.

* 256 lanes, one group per SM, 16×8 (8×16) accumulators. 4 slabs of k = 8, 3 in flight, one zero-filled 4-byte
  `cp.async` per element with u4's lane maps, strides padded by 4: 50,176 B (`can_run` refuses a smaller budget).
  A slab past k is issued anyway (`src-size` 0), so no branch guards the refill.
* A 16-byte leg, derived in `launch_register_k8_s4_large`, when every packet operand has a 16-byte base and
  `ld`, stride ≡ 0 mod 4.
* Epilogue staged at every k, 4 passes; pass p + 1's priors load while pass p stores; none at beta = 0. Stage stride
  TileM + 16: at + 4 the two column slots of an `STS.128` phase overlapped in 12 banks (conflicts 4.19M → 0, time flat).
* ncu (TT strided 1024²×362): the tile moves exactly cuBLAS's global sectors, shared wavefronts and instructions;
  the remaining gap is issue (`wait`, `lg_throttle` on the 12 back-to-back `LDGSTS`).

| cell (b128 unless noted) | best before / vendor | best new / vendor |
|---|---|---|
| TT strided 1024×1024×362 | u4 1.159 | 128×256 1.035 |
| TN packed 1024×1024×362 | u4 1.089 | 256×128 0.947 |
| NN strided 512×512×1024 b256 | u4 1.184 | 256×128 1.028 |
| NT strided 362×1024×96 | u4 0.984 | 128×256 0.987 (flat) |
| NN packed 1024×2048×96 (aligned) | u1 0.704 | 256×128 0.720 (u1 stays) |

15 of 16 strided cells gain 5-13%. 128×256 serves TT best, 256×128 NN and TN; at n = 128 the 128×256 tile is half
empty (1.44-1.48×). Registers 255 (256×128) is the one-group limit with zero margin. **Accepted assumption:**
`src-size` 0 never translates its address, so the large tiles keep the computed address of a zero-filled copy
(clamping it to the base cost 0.1-1.5%; on u4 it cost ≤ 0.7% and shipped). Tests:
`RegLargeTilesEveryFormSlabCountAndLegAtSaturatingBatch`, `RegLargeTilesBetaZeroAndTheDerivedLeg`.

### The staged epilogue

At k ≤ 64 the 128×128 kernel is store-bound (512²×32 b4096 writes 4.3 GB of C). The direct epilogue stores four
128-byte column pieces per warp instruction (DRAM 87.6% of peak against 89.6%); the staged one routes C through local
memory, 4 passes of 32 columns, so each warp stores whole 128-row columns. sm_89, vs cuBLAS, 512², β = 0: k = 32 direct
0.982 / staged 1.002; k = 64 0.994 / 1.000; k = 128 1.003 / 1.004; k = 256 1.18 / 1.12, hence
`kStagedEpilogueMaxK = 64` on the aligned leg. The predicated leg stages at every k
([WP-F](#gemm-wp-f-float-mid-size-parity-on-sm_120)).

### GEMM: the register-tiled launcher table

`register_launchers.hh`: `RegTile` holds `launch_register_tiled<>`'s template parameters as a structural type, so
`launch_reg<T, RegTile{...}>` serves every `reg_configs` row (`launch_reg_cfg` in `gemm_kernels.cc`). **Trap:** there
`ThreadTileCols` defaults to `ThreadTileRows`; here `TR` and `TC` default independently to 4, so every row states both.

## GEMM: wide kernels (`wide`, `splitk`)

### The wide scalar kernel

`register_64x64_k16_wide.hh`: 64×64×16 tile, 4×4 thread tile, the first register-tiled kernel for a non-float scalar.
Today it holds two kernels: the NN kernel (double only) and [wide v2](#gemm-wide-v2-one-double-buffered-64x64-kernel-for-every-form)
(every form for float and the complex scalars). sm_89 at 256³-1024³, both betas:

| scalar | vs `tiled` | vs cuBLAS |
|---|---|---|
| complex<float> | 7.0-7.7× | 0.98-1.08× |
| complex<double> | 3.56-3.60× | 1.12× |
| double | 1.01-1.08× | 1.07-1.15× |
| float | — | **0.85-0.93×**, so no sm_89 float row ranks `wide` |

On sm_89 it ranks first only in the 20 double NN packed squares 256-1024 (batch ≥ 64); FP64 caps at ~1.44 TFLOP/s and
`tiled` already reaches 92% of it (a 4090-specific conclusion). Load-bearing: the 16-byte access granule, `may_alias`
on the punning types, a native vector type for staging, the POD complex scalar, m fastest in the epilogue.

### Wide-scalar transposed tiles

`register_wide_transposed.hh` (`launch_wide_transposed`): the non-64×64 wide tiles in every form, and double's 64×64
tile in every form but NN at k = 16. A tile is matched to a panel width, not square: potrf's trailing update is NC at (nb, W) = (128, 128)
float, (96, 32) double and cfloat, (64, 16) cdouble; geqrf's `V^H A22` is CN at nb = 32. `register_tiled_common.hh`
would go through the Annex-G complex multiply, so these use the POD scalar; conjugation is applied once per element,
not per FMA. A `ConjTrans` instance serves a real `Trans`; for complex, `wide_serves` refuses the substitution.

sm_89, batch 1024, ld + 8, exact tile multiples: cdouble NC 1.12-1.14× of cuBLAS and 3.35-3.53× of `tiled`; cfloat
0.86-1.07× of cuBLAS; double 0.92-1.00× of `tiled` (so no double row ranks them). On the drivers' ragged sizes the
cdouble window collapses (CN 32×n×256: n = 136 0.64×, 224 0.99×, 384 1.13×), so `vendor` stays first and the tiles
are the rows' first native entry (1.68-3.52× of `tiled`). **A macro-tiled kernel must be swept at sizes that are not
multiples of its tile.** Bracketing non-winners (cdouble vs cuBLAS): m below the 128-row NC tile 32×32×96 0.28; k = 1
0.35; batch 64 0.57. Raw: `benchmarks/results/p6_gemm_*.csv`.

### GEMM: wide instantiations in every transpose form

`WideCfg::forms` is a 3×3 bit mask over (op(A), op(B)) ∈ {N, T, C}² (`wide_bit(a, b) = 1 << (3a + b)`), so complex NT,
TN, CT, TC, CC and TT no longer fall to `tiled` (about 8 TFLOP/s against cuBLAS's 45-51 on sm_120). `wide_serves<T>`
is the one rule (complex exact; real folds T to C and compiles only N/C builds). Rows are only appended, and a form a
row does not list is a `static_assert` in `launch_wide_cfg`, so `can_run`, the launcher and the trace name cannot
drift. The tuner grid gained complex TC, CC, TT squares and real TT panels.

### GEMM: skinny 64x32 and 32x64 wide tiles

`wide:m=64:n=32:k=16` and `wide:m=32:n=64:k=16`: 128 lanes, 4×4 thread tile, configs of `launch_wide_transposed`.
At a skinny output the 64×64 tile left half or more of its lanes on padding (cfloat CN strided 256×32×128 b128 lost
2.19× on sm_120). The deep table ranks them first in 1703 (64×32) and 711 (32×64) rows. Sliced K inside a group is
not built.

### GEMM: WP-B coalesced C epilogue on the predicated legs

At small k a cell is one read-modify-write pass over C. The band store (whole 16-byte bands) is a per-group decision
about C alone in the wide kernels (tile inside C, `ldc` whole granules, 16-byte item base), so a predicated leg takes
it on interior groups. Every other group stages C through the mainloop's local memory and stores with lane =
consecutive row, one contiguous run per warp store at any `ldc` (`wide_scalar::staged_c_store`; generic `reg` inline,
column stride TileM + 1 so the TC-strided writes spread over the banks). Barriers: every mainloop already ends on one
after its last read of the stage; each chunk adds one between the stage writes and the read-back, and one more before
the next chunk. No local memory is added. complex<double> skips staging (its element store is the band store). Test:
`CoalescedCEpilogueEveryTileAtSaturatingBatch`.

### GEMM: wide v2, one double-buffered 64x64 kernel for every form

`GemmWide64Body` (`register_64x64_k16_wide.hh`) runs `wide:m=64:n=64` for float, complex<float> and (since
[WP-J](#gemm-wp-j-complex-double-on-wide-v2)) complex<double> in all nine forms; double keeps the old kernels.

* **Staging is a pure copy.** Conjugation is folded into compile-time FMA signs (`wide_v2::fma_signed`).
* **One shared layout**, `s[kk][i]` in 16-byte granules, for both operands; the granule index is XORed with
  `swz(kk) = ((kk / VecN) & 3) << 1` on store and read, so the transposing store is conflict-free (the old one was
  16-way). Four values, not eight: eight left complex<float> TN and CN 24 B of spill at the cap.
* **Two buffers, one barrier per slab, the next slab prefetched into `ext_vector` registers**; an odd slab count
  ends on buffer 0 with no barrier, so the staged epilogue adds one.
* **The leg** (16-byte vs element loads) is per launch (`wide64_v2_vec_loads`: bases, ld, batch stride); predication
  is per work-group and slab, so one ragged edge no longer sends interior groups down the predicated path.
* **Launch bound** on `operator()`: 3 groups per CU float, 2 complex<float>, 1 complex<double> and 128×128. The
  bound is ignored on a lambda, and needs `.reqntid` ([WP-Q](#gemm-wp-q-the-launch-bound-was-ignored)).
  `operator()` is `always_inline`: out of line, complex<float> spilled 64-104 B behind the call while the entry
  reported zero.

complex<float> at 3 groups (80-register cap) spills 76-240 B, so 2 is the bound. Local memory: complex<float> 32 KiB
(16-byte leg) or 33,792 B (element leg, [WP-Y](#gemm-wp-y-coalesced-element-leg-lane-maps)), over an Apple GPU's
32 KiB, so `can_run`'s `wide_slm_bytes` term refuses it there. Tests: `WideV2EveryFormAtSaturatingBatch`,
`WideV2TileAndSlabEdgesUnderNaNPoison`, and the leg layouts in `PinnedChoiceLaunchesItsOwnKernel`.

### GEMM: WP-F, a k8 slab for the 64x64 wide tile

`wide:m=64:n=64:k=8`: at k ≤ 8 the 16-deep slab was half zero-fill. Wide v2 at `TileK = 8` for float and the complex
scalars; double runs `launch_wide_transposed` at K = 8 in every form. Each lane still stages CF 16-byte granules, so
warps 0-3 stage A and 4-7 B (warp-uniform), with their own `KStep` and k-contiguous lane map; float's swizzle flips
granule bit 2 to keep its transposing store on 32 banks. Correctness never depends on the swizzle (store and read
apply the same bijection). Local memory halves. Test: `WideK8SlabStraddlesTheSlabDepthAtSaturatingBatch`.

### GEMM: WP-K, pipelined staging for the transposed wide tiles

`launch_wide_transposed` tiles: `og::wide_stages<T>` gives 2 stages when 2·k·(m+n)·sizeof(T) ≤ 40 KiB, else 1 (cdouble
128×32, 32×128, 64×32, 32×64); two stages take one barrier per slab. The next slab is fetched into registers before
the current FMAs, except on the 64×64 double tile and every complex<double> tile (prefetch spilled all 85 cdouble
instances at 255 registers). 16-byte granules along the contiguous dimension, XOR swizzle on the k-contiguous store,
none for complex<double>: at VecN = 1 the per-kk XORed fragment addresses are slab-invariant, were hoisted one
register each and spilled, so it uses a conflict-free sector-pair map instead, and one guarded 16-byte load per
granule (the full/predicated diamond is not if-convertible). Metal (about 26 KiB) loses double-buffered cfloat
128×32 and 32×128 through `can_run`. Test: `WideTransposedPipelineEveryTileFormAndLegAtSaturatingBatch`.

### GEMM: WP-L, global split-k over the transposed wide tiles

Target: few tiles with a large k (tiles × batch below ~2 × SMs, k ≥ 256), where each group walks k serially (pre-188
sm_120: float NT 32×64×1024 b128 3.36× vendor, cfloat NC 3.51×). The deep table ranks `splitk` first in 649 rows; the
WP-X A/B: float NT strided 32×64×1024 b128 `splitk 32x64 s4` 0.710× vs `wide:m=32:n=64` 1.115×.

* Hosts: `splitk_host` rows (the six non-64×64, non-v2 rows), s ∈ {2, 4, 8}. Group z computes item z / s over its k
  slice (kc = ⌈k / s⌉ rounded up to TileK) into partials at α = 1, β = 0; a slice past k stores zeros.
* Partials `[batch][s][ldw × n]`, ldw = m rounded to 4 (`splitk_ld`), leased from the Queue's arena (sized with
  `BumpAllocator::measuring()`). The arena keeps its peak until `trim_workspace()`
  ([workspace](../design/workspace.md#workspace-the-arena-keeps-its-peak)).
* Reduction (`gemm_splitk_reduce`) sums slices 0..s-1 in order (deterministic, no atomics), then C = α Σ + β C; β = 0
  never reads C. An out-of-order Queue waits for the slices first.
* `can_run` holds the partials to `scratch_budget` = min(MAX_MEM_ALLOC_SIZE, GLOBAL_MEM_SIZE / 4) / batch (past it the
  lease throws `std::bad_alloc`); no k or tile-count term, the table draws the window. The tuner's large-k grid
  (`grid_gram_*`, `grid_large_skinny_k`) is not in the sm_89 transcription.

Tests: `SplitKEveryHostEveryFormAtSaturatingBatchAndSliceEdges` (the arena is poisoned before each call and must not
grow), `SplitKOnAnOutOfOrderQueueReducesAfterTheSlices`, `SplitKGridCeilingIsServedByBatchChunking`. The
`scratch_budget` term has no test.

### GEMM: WP-J, complex double on wide v2

`wide:m=64:n=64:k={16,8}` for complex<double> run `GemmWide64Body` in all nine forms (CF = 4: one element per
granule). The k-contiguous lane map puts 16 lanes down one column (256 contiguous bytes); the swizzle is `kk & 7`,
because an 8-lane phase holds 8 consecutive kk of one column 1 KiB apart. Fragments are the granule bit-cast
(`__builtin_bit_cast`), not a `memcpy` through a private array (acpp leaves that on the stack). Launch bound 1 group
per CU: FP64 is compute-bound at 1/64 rate, and 2 groups would cap registers at 128 against 134-224 used. Local
memory 64 KiB at k = 16, so a budget under 64 KiB (acpp without `BATCHLAS_ACPP_SLM_OPTIN`, Intel Arc) refuses that
tile (`WideTileFitsOrIsRefused`); k = 8 (32 KiB) still runs. Only an 8-byte base reaches the element leg
(`WideV2ComplexDoubleElementLegAtAnEightByteBase`). No complex<double> A/B: FP64 is not tuned on these parts.

### GEMM: WP-Q, a one-round staged epilogue for wide v2

On the element leg the loss was mostly C, not the loads (float TN strided 2048×1024×32: staged epilogue 2.26 ms at
56% DRAM, band store on the packed twin 1.45 ms at 86%). Wide v2 passes `staged_c_store` a whole-chunk Batch
(`EpiBatch`: 16 at k = 16, 8 at k = 8): a thread keeps one row and every 4th column, all priors of the chunk in flight,
one round trip; priors load only at beta ≠ 0. Other callers keep Batch 2 and the element map. 2026-10-19, sm_120:

| cell (b128) | parent / vendor | new / vendor |
|---|---|---|
| cfloat NC strided 64×1448×8 (k8) | 1.39 | 0.94 |
| float TN strided 2048×1024×32 (k16) | 1.51 | 1.16 |
| float TN packed 1024×1024×362 (large k, unchanged) | 1.49 | 1.48 |
| 16-byte leg: cfloat NC packed 512×512×64 (k8) | 0.96 | 0.97 |

Timed apart, the epilogue alone is the win (0.70-0.85 new / old); the bound alone helps float k8 5-6% and costs
complex<float> 1-2%. complex<float> at one group per CU was mixed (0.97-1.16), so 2 groups stays; float at 4 groups
measured 0.99-1.03 of 3, so 3 stays. Breaks run on the GPU: store guard `< cw` aborts; priors read at beta = 0 turns
the NaN-C cases red.

#### GEMM: WP-Q, the launch bound was ignored

ptxas printed `.minnctapersm is ignored when neither .maxntid nor .reqntid is specified` 112 times for
`libbatchlas_sycl` (sm_120): `intel::max_work_group_size(1, 1, N)` emits no `.maxntid` on these functors, so
`BATCHLAS_LAUNCH_BOUNDS` never capped a register count. `reqd_work_group_size(1, 1, 256)` on `GemmWide64Body::operator()`
applies it (float 64-72 → 80 registers at 3 groups; complex<float> k8 112-128 → 96). Every gemm functor now declares
it; the tiny solvers do not
([known defect 20](../design/known-defects.md#defect-20-batchlas_launch_bounds-emits-no-maxntid-so-ptxas-ignores-the-bound)).

### GEMM: WP-O, a 128x128 complex float tile on wide v2

`wide:m=128:n=128:k=8` (complex<float> only, `WideCfg::scalars`, `v2`): `GemmWide64Body` at 128×128, 8×8 complex
thread tile, split staging, each operand staged as two 64-wide halves with the 64×64 lane map, one group per CU,
254-255 registers, zero spill. The halves unroll through `static_for`: a one-trip loop let LICM hoist edge tests and
spilled 4 B in float 64×64 NC on sm_89. A larger tile at k = 16 is a `static_assert` (64 KiB).

Why 128×128 (ncu, cfloat NN packed 1024³ b128, 17.2G FFMA in every arm): the arms are issue-bound, and the thread
tile sets the LDS share. 64×64×16 18.85G instructions, 22.21 ms; 128×64×8 19.08G (its swizzled fragment bases added
0.47G of address arithmetic), 21.51 ms; 128×128×8 18.46G, 21.03 ms; cuBLAS `cutlass_100_simt_cgemm_128x128_8x4` 18.26G,
20.00 ms. Against 64×64×16 (2026-10-20): 0.90-0.94 on the large cells (NN packed 1024³ 1.039 vs 1.137 × vendor);
64×64 stays best at m = 64 (NN 64×2048×32: 1.489 vs 1.013) and on 192³ (1.290 vs 0.952). The deep table ranks it
first in 1503 rows. Test: `Wide128x128EveryFormAtSaturatingBatchAndTileEdges`; a half's i bound only stops reads past
m or n and is not value-testable.

### GEMM: WP-V, three copy-async stages for the 128x128 complex float tile

`wide:m=128:n=128:k=8` has a second leg, derived in `launch_wide64_v2`: 3 slabs by 8- and 16-byte `cp.async`
(`wide_v2::acopy`, zero-filled by `src-size` 0), 2 in flight, no register prefetch, 48 KiB, trace `…_s3`. Gate: DPC++,
`cuda_compute_capability() >= 120` (the only architecture timed; `=3` admits sm_80), a 48 KiB budget at launch, and
both bases 8-byte aligned (`alignof(std::complex<float>)` is 4; a 4-byte base faulted with
`CUDA_ERROR_MISALIGNED_ADDRESS`, sticky). `can_run` keeps the two-slab term; otherwise the WP-O register prefetch runs.
Lanes 0-127 copy the first halves, 128-255 the second, so all eight warps issue copies.

2026-10-20, 10-12 reps: 3 stages spread 0.926-0.982 of the WP-O leg; 4 stages with split staging 0.940-1.003
(`mio_throttle` 0.20: every copy on four warps). Against the parent build every one of 21 cells gains 1.6-7% (NN
strided 512³ 1.075 → 1.004 × vendor), none loses; cuBLAS still leads on short-k NN/NT (k = 23-45, 1.06-1.07×). The
remaining gap is `long_scoreboard` (0.14 vs cuBLAS 0.07). Tests run twice, derived leg and `BATCHLAS_GEMM_WIDE_STAGES=2`,
plus a 4-byte base per form.

### GEMM: WP-X, copy-async stages for the transposed wide tiles

Target (2026-10-20/21): strided skinny and panel cells of `launch_wide_transposed` at 1.08-1.25× vendor. ncu: DRAM-bound;
cuBLAS (`cutlass_100_simt_cgemm_64x32_16x4`, element `cp.async`, 4-5 stages) at 87-93% of DRAM, ours at 73-81%, from
the staged C epilogue (8 serial round trips for a 64×32 cfloat tile) and element loads touching twice the sectors
they use (2.1-3.2× cuBLAS's count).

* Epilogue (float, complex<float>): `staged_c_store` in one pass and one round trip, as wide v2's. Double precision
  keeps two per trip.
* Copy leg (complex<float> only): 2 stages, one barrier per slab; one 16-byte copy per granule along i on the 16-byte
  leg, else one 8-byte element per copy on consecutive addresses; conjugation at the fragment read. Gate as WP-V's,
  plus ld ≤ INT_MAX / 128 (int offsets). Traces `_async`, `_async_vec`, `_splitk_async[_vec]`.
* 3 and 4 stages cut groups per SM and lost 3-10%; float's 4-byte copies cost 16-45 registers and ran slower, so
  float takes only the epilogue.

| cell (strided) | parent / vendor | new / vendor | `=2` |
|---|---|---|---|
| cfloat NC 2048×32×32 b128, 128×32 | 1.174 | 0.958 | 0.983 |
| cfloat NC 32×2048×128 b128, 32×128 | 1.040 | 0.935 | 0.987 |
| float TN 2048×32×45 b2048, 128×32 | 1.202 | 1.051 | 1.052 |
| float NN 2048×32×64 b2048, 64×32 | 1.188 | 1.101 | 1.103 |

Every target cell gains 5.6-19%; no control loses beyond vendor drift. The copy leg's sector count equals cuBLAS's
(NC 32×2048×128: 16.1M). The vendor still leads on float strided (1.05-1.13×). Not done: the same leg on wide v2's
64×64 tile.

### GEMM: WP-X review, beta zero keeps negative zero

`wide_scalar::lin_epi` (wide v2, the transposed tiles, the split-k reduction) and `LinearEpilogue::apply` return
`alpha · acc` exactly at beta = 0: `alpha · acc + 0 · 0` stored −0 as +0. Callers pass prior = +0 and the helper takes
beta as −0 (complex: beta.re → −0, beta.im → +0 in the real part and −0 in the imaginary one), a loop-invariant select.
A per-element select cost 2-3% on complex<float> transposed cells (NT strided 2048×32×32 b128: 0.985-1.001 vs 0.965
× vendor); the hoisted form is level. Test: `BetaZeroKeepsNegativeZeroOnEveryNativeChoice` (A = 0, alpha = −1,
beta = 0, C NaN): with the old helpers red on every `wide` and `splitk` choice, green on the rest.

### GEMM: WP-T, TT, TC and CC on the small transposed tiles

`WideCfg::real` lists the N/C forms a real scalar compiles: real CC (so its TT, TC, CT) on all six transposed tiles;
complex TT, TC and CC on 32×32 and 16×16 only (the complex grid has no TT/TC/CC panels). Previously those forms fell
to `tiled` or `wide:m=64:n=64`. 2026-10-20, 10 reps:

| cell | parent / vendor | new / vendor |
|---|---|---|
| double TT strided 128×32×32 b32768 | `tiled` 1.19 | 128×32 0.94 |
| cdouble TT packed 32³ b128 | `tiled` 1.69 | 16×16 0.75 |
| float TT strided 32×91×1024 b128 | 64×64×16 1.45 | `splitk 32x128 s4` 0.74 |
| bracket: double TT packed 256²×8 b128 | 64×64×8 0.53 | 64×32 0.89 |
| bracket: float TT strided 256³ b2048 | 64×64×16 1.43 | 64×32 1.71 |

The brackets keep 64×64 at a large output; the windows between are the table's. Every form on the panel tiles cost
+28% device link; the shipped set +12-14%. Test: `SmallWideTilesRunTheirTransposedForms` and the every-form suites.

### GEMM: WP-Y, coalesced element-leg lane maps

Wide v2's element leg (odd ld, `k % 4 ≠ 0` packed, unaligned base) moved 4.4× the 16-byte leg's global sectors at
large k (float TN packed 1024²×362: 969M vs 218M). Shipped, float and complex<float>, k-contiguous operand only:
lane `st` takes row `st % TileK` of columns `st / TileK + j · OpThreads / TileK` (`wide_v2::eload`), so an instruction
is 2 columns × 64 B (k16); that operand's shared row is padded by 4 floats and unswizzled, and at k = 16 rows 8-15 flip
the element's slot in its granule (`ekx`) so the transposing store hits 32 banks. complex<float> loads 8 bytes per
element at an 8-byte base. The column step is 64-bit (`16·ld·CF` passes INT_MAX at ld ≈ 67M), rebuilt per call behind
an empty `asm volatile` (hoisted, it spilled 8 B). `wide_slm_bytes` adds 2 × 16 B per k row.

2026-10-21, 12 reps × 4 runs: forms with both operands k-contiguous (TN, CN) gain 1.6-7.4% (float TN packed
1024²×362 1.489 → 1.377 × vendor; cfloat TN strided 512×512×64 k8 1.083 → 1.025); one k-contiguous operand 0-2.3%; no
k-contiguous operand and the 16-byte leg flat. Sectors at 362: 969M → 279M. The 128×128 three-stage leg is unchanged
(DRAM-bound; the pad would take it past 48 KiB). Test: `WideV2ElementLegKContiguousMapAtSaturatingBatch`; the pad and
flip are bank layout only, invisible to value tests.

## GEMM: small kernels (`small`)

### The small batched kernel

`small_batched.hh`: a 128-lane group takes 128 / (2·NB) matrices, NB ∈ {8, 16, 32, 64} from max(m,n,k). op(B) is
staged in local memory (odd stride, so either transpose is coalesced); each lane holds one row of op(A) and NB/2
columns of C, lanes running down the rows (`direct` runs them across columns, ld apart); β = 0 skips the C read.
sm_89, batch 32768, square NN, β = 0, vs cuBLAS: n = 8 8.5×, 16 2.69×, 32 1.91× (`direct` was 0.64×), 40 0.86×, 48
0.88×, 64 1.02×. The NB = 64 bucket is shared-load bound, which is why NN 33..56 has its own kernel. double measured at
parity with `direct`/`tiled`.

### The small tiled kernel

Float NN with 32 < max(m,n,k) ≤ 56: one matrix per group of (NB/4)² lanes (NB = 48 or 56), A staged `[k][m]`, B as
stored, a 4×4 register tile of C per lane (an FMA costs 1/8 of a shared load). At β ≠ 0 each lane reads its C tile
before the barrier (48³ β=1: 0.91× → 1.09×); at β = 0 that prefetch is compiled out (its 27 registers cost 12% at 56³).
sm_89, 2026-09-27, batch 32768, vs cuBLAS (β = 0 / 1): 33 2.30 / 1.94; 48 1.36 / 1.15; **56 1.23 / 1.08**; at 48 the
win holds from batch 64 (1.33 / 1.43). 57..64 keeps NB = 64 (a tiled NB = 64 measured 0.89-0.91×). So `small` ranks
first in exactly the 72 float NN square rows up to 48 at batch ≥ 64; 49-64 and batch 1/63 rank `vendor`
(`GemmTranscribedTable.Sm89BracketsTheTranscribedVendorEdges`: 48³ `small`, 49³ and 52³ `vendor`).

### GEMM: complex small and the staged transposed-A row

`small` compiles for both complex scalars on the POD `Cx` (`fma_acc`, `dev_mul`; no `__mulsc3`), conjugating at
staging. complex<double> stops at 32 (`small_max_dim`): its 64 bucket would need 66.5 KB of local memory and 384
registers. Local memory is a `can_run` term (`small_slm_bytes` against `slm_budget_for`). A transposed A goes through
the slot's `sB` region in storage order (`sB[r + l·(NB+1)]`, conflict-free since NB + 1 is odd), a barrier, each lane
reads its row, a second barrier before op(B) is staged over it; this also serves the real T forms. complex<float> 64
writes its inner product as four 16-iteration loops in a file-local macro: as one loop the unroll was declined and the
row went to a 512 B stack frame. Tests: `SmallEveryBucketEveryFormAtSaturatingBatch`,
`GemmCanRun.SmallCeilingAndLocalMemoryPerScalar`.

## GEMM: shared device code

### GEMM: the POD device scalar

`src/sycl/device_scalar.hh`. `std::complex` must not reach device code: its `operator*` is Annex-G (an `isnan` branch
and a `__mulsc3`/`__muldc3` call). Launchers re-type operands and scalars to the layout-compatible aggregate `Cx<R>`
at the pointer boundary; the PTX of the GEMM instances has zero `__mulsc3`, `__muldc3` and `call.uni`. Division is
Smith's algorithm: \f$(c-di)/(c^2+d^2)\f$ overflows above about \f$10^{19}\f$ (float) or \f$10^{154}\f$ (double) and
returns 0. POTRF uses real-component helpers (`dev_real`, `dev_div_real`); the division/reciprocal asymmetry is
deliberate (reference `?trsm` divides, `?potf2` multiplies by a reciprocal; unifying them moves one off its LAPACK
rounding).

### GEMM: the heterogeneous-batch loop

`src/backends/gemm_heterogeneous.hh` (`detail::gemm_heterogeneous_loop`), shared by the cuBLAS, rocBLAS and
vendor-free paths: one single-matrix GEMM per member, the per-item terminal the only parameter. `m == 0` or `n == 0`
members are skipped, a `k == 0` member is `C := beta·C`, and an all-skipped batch returns a valid Event
(`create_event_after_external_work()`, right for cuBLAS and rocBLAS; an MKL backend would need the queue's own event).
The public `gemm` splits a heterogeneous batch before choosing; the backends' loops serve direct `gemm_vendor` callers.

### The subgroup workspace budget

`kSubgroupWorkspaceBudgetBytes` (`group_blas_subgroup_common.hh`) is `device_limits::subgroup_workspace_budget_bytes()`,
generated from `cmake/device_limits.h.in` (`batchlas_subgroup_workspace_budget_bytes`, `cmake/BatchLASDetectSYCL.cmake`):
the architecture's table local memory less a 4 KiB reserve, 16 KiB floor. NVIDIA 45,056; AMD and Intel 61,440;
unrecognised GPU 28,672. It is deliberately not the device's real local memory (101,376 B on sm_89): five `if
constexpr` workspace predicates read it, and `group_blas_rankk.hh` is shared by symm, herk, syrk and syr2k, so raising
it retunes five ops. Run-time capacities query `DeviceProperty::LOCAL_MEM_SIZE` (`src/util/resident_capacity.hh`).
No budget but the shipped one was ever timed; there is no override.

## GEMM: negative results

| Alternative | Result | Verdict |
|---|---|---|
| Double-buffering the 128×128 k-loop (2024) | 0 time recovered: the prefetch was sunk to its store | Refuted; use `ext_vector` prefetch registers |
| 128×128 lane maps with both lane bits in one operand; B stride 132; `st.global.cs`; 16-deep slab; packing B | ±1% to −2%, or loses as m grows (sm_89) | Dead |
| Wide-scalar tile for float (sm_89) | 0.85-0.93× of cuBLAS SGEMM | Dead on sm_89 (wide v2 serves float on sm_120) |
| 128×128 8×8 tile for wide scalars (pre-v2) | cdouble spills 3.4 KB and cannot launch (208 × 512 > 65,536 registers) | Not launchable for cdouble |
| Bare `min_dim >= 32` complex floor | 12 of 12 losses at batch 1-8 | Dead; the CTA gate replaced it |
| Float native windows on sm_89 (NN 128..512, transposed, non-square) | 0.23-0.55× | Dead |
| u4 aligned cp.async leg (WP-N2) | NN 1024³ 1.145× vs u1 1.043×; TT 1.117× vs 1.052× | Deleted |
| u4 pipeline on aligned data (WP-F) | NN 512³ 1.146 vs u1 1.070 | Reverted |
| `reg:m=128:n=64:k=8:u=1` row | 288-byte stack frame in all forms | Not built |
| Wide v2 128×64 complex<float> tile (WP-O) | 0.95-0.99 of 64×64 on large cells, first in none of 15 cells; spilled at 2 groups | Not built |
| Wide v2 4 stages, split staging (WP-V) | 0.940-1.003 of WP-O leg vs 0.926-0.982 for 3 spread | 3 stages ship |
| Wide v2 element leg via `cp.async` (WP-Q) | complex<float> 110 → 185 registers, 1.26× slower; with the old map 0.98-1.04 | Dead |
| Wide v2 element leg, register staging of 32 consecutive floats | 1.02-1.04× slower; at a 64 cap spilled, 1.14-1.28× | Dead |
| WP-Y map with the i-contiguous operand i-fastest too | float NT strided 2048×32×23 k8 1.169 (4× the shared stores) | Dead |
| Funnel loads (two aligned 16-byte loads + select) | requests 49M → 26M, same time | Dead |
| complex<float> wide v2 at 1 group per CU | 0.97-1.16 of 2 groups, mixed | 2 groups stay |
| Float wide v2 at 4 groups per CU | 0.99-1.03 of 3 | 3 groups stay |
| 3 or 4 copy-async stages on the transposed tiles (WP-X); float 4-byte copies | 3-10% slower (occupancy); float +16-45 registers | 2 stages, complex<float> only |
| 256×128: copies spread over the 8 k steps (CUTLASS order) | moves the stall to `mio_throttle`; loses but on TN 1448×256×128 | Dead |
| 256×128: priors one pass ahead in one buffer | 0.1-1.5% slower | Rejected |
| Clamp zero-size copies to the base on 256×128/128×256; skip slabs past k | +0.1-1.7% | Rejected (shipped on u4: ≤ 0.7%) |
| Direct u1 TN instance kept for k ≥ 1024 | 1.6-3.8% faster there, but u1 is 1.5× vendor and no table ranks it | Not kept |

## GEMM: correctness findings

* **Hard-wired forms.** Nine transposed launchers dropped the conjugation on `ConjTrans` (fixed `f236575`), and 18
  NN-only register variants computed NN on a transposed call (fixed in P3.4). `can_run` lists the instantiated forms
  per config (`reg_form`, `wide_serves`), so such a pin throws; guard
  `ForcedTransposedLauncherRejectsMismatchedTransposeForm` (`tests/gemm_tests.cc`).
* **The β = 0 microbenchmark is blind to epilogue defects**: an epilogue with m slow-varying ran 26.0 TFLOP/s at
  β = 0 vs 41 standalone. Measure both betas, and beta = 0 with A = 0, alpha = −1 for the −0 rule.
* **Event-profiling sums over queued submissions do not measure kernel time** (30 submissions summed to 19.8 ms for
  a 3.15 ms kernel).
* **An unbounded over-read is invisible to value tests**: rows past m or n feed only discarded accumulators, so a
  dropped i bound, interior test or edge clip is correct until it leaves the allocation. The tests use one
  poison-padded buffer; `src-size` 0 copies are assumed never to translate their address.
* **Register counts at a cap are chaotic**: a logically equivalent rewrite moves spill between instances; re-run
  `scripts/register_probe.sh` (zero spill library-wide is the invariant) after any kernel edit.

## GEMM: open debts

* **sm_89 is untimed.** Every sm_89 row is transcribed; wide v2, the copy-async legs, WP-U and the large tiles were
  probed there, not timed. FP64 rows on both devices are transcribed, so complex<double> wide v2 (WP-J) runs in
  windows drawn for the pre-v2 kernels (`wide` is the first native entry of 1014 sm_89 NN rows and of 1264 sm_120
  cdouble rows), reached by vendor-free builds and `native`.
* **The complex batch edge of the CTA gate is not in the sm_89 table**: a complex NN call below batch 128 takes a
  batch-128 row whose native walk ranks `wide` where the ladder measured losses (0.17-0.96× of `tiled`). Only a
  vendor-free build pays it; the fix is a grid edge row.
* **The sm_120 table misses k > 512 at 512×512 b128**: k = 768 and 1024 borrow the k = 512 row and run u1 at 1.01-1.06×
  vendor (WP-U).
* **Vendor-free heterogeneous GEMM is ~7 GFLOP/s**, one launch per member; a single-launch variant is deferred.
* **The `ld` slope has no mechanism below L2**; ncu exposes no row-activate counter.
* **The demand tables are `ctest` coverage captures**, not user workloads.
* **TF32 is reachable but unmeasured**: the natives answer a TF32 request only vendor-free, at full precision
  ([precision contract](#gemm-batch-chunking)).
* **`gemm_direct`'s batch offsets are `int`**: overflow at large batch × stride; not a `can_run` term.
* **AdaptiveCpp drops `reqd_work_group_size`**, so an acpp CUDA build presumably runs the u4 leg at one group per SM
  (not probed).

## GEMM: raw evidence

sm_89 window data is at the git tag `perf-evidence/vendor-independence` (`git show
perf-evidence/vendor-independence:<path>`):

| topic | path |
|---|---|
| Double window, saturation, Direct/Tiled16 boundary | `experiments/wp2_e3/` |
| Float NN and transposed windows; predicated 128×128 | `experiments/wp2_e4/` |
| Non-square double, k = 1 boundary | `experiments/wp2_e5/` |
| Wide-scalar bake-off, PTX evidence, cuBLAS baselines | `experiments/wide_scalar_gemm/` |
| SYCL-vs-CUDA parity, SASS counts, TF32 probe | `experiments/sycl_vs_cuda/FINDINGS.md` |
| Strided-`ld` ncu campaign and routing fix | `experiments/wp4_gemm_ld/` |
| Complex routing, CTA ladder | `experiments/wp4_complex/` |
| Trailing-update GEMM inside `trsm` | `experiments/wp3_s16/` |

Transposed-tile CSVs in tree: `benchmarks/results/p6_gemm_*.csv`, `p6_e2e_before.csv`,
`p6_e2e_with_refused_window.csv`. sm_120: `benchmarks/results/tuning/ledger/gemm.sm_120.jsonl` (the deep table).
