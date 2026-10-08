# GEMM {#perf_gemm}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda

gemm selects through flat kernel selection (@ref design_flat_selection). This page lists the choices, the
`tuned/gemm.*` rows that Auto ranks first on sm_89, and the measurements behind each window. The tables are
transcribed from the old router (`source=transcribed:424a45bc`) and are untimed, so these windows are what Auto
runs on sm_89.

Measurement setup: one dedicated GPU (`experiments/gpu_guard.sh`), warm SYCL JIT (`--warmup=5`), median of 3,
both β=0 and β=1 unless a row says otherwise. Sanity anchors: vendor SGEMM reaches 45–48 TFLOP/s at 512³ (a
number near 80 is TF32); vendor DGEMM stays below ~1.45 TFLOP/s (FP64 is 1/64 of FP32 on this part).

Auto on sm_89:

* **double:** native first at batch ≥ 64 and k ≥ 2 (`tiled`; `direct` for small squares; `wide:m=64:n=64:k=16`
  on large packed squares).
* **float:** `small` first on NN squares up to 48 at batch ≥ 64; vendor everywhere else.
* **complex:** vendor first everywhere.

## GEMM: what ships

### Choices (flat selection, P3.4)

The vocabulary is `src/ops/gemm/choice.hh`, listed in tie-break order. `src/ops/gemm/gemm.cc` validates the
views, calls `select::run` (`gemm.cc:172-180`), and launches one kernel from `src/sycl/gemm_kernels.cc`.

| spelling | launcher | scalars | `can_run` beyond the common term (`gemm.cc`) |
|---|---|---|---|
| `direct` | `gemm_direct`, one work-item per C element | all | `grid && max_wg >= 64` (`:103`) |
| `tiled` | `gemm_tiled`, 16×16 shared tile | all | `grid && max_wg >= 256` (`:104`) |
| `small` | `gemm_small`, `small_batched.hh` | float, double | real, `max(m,n,k) <= 64`, `max_wg >= 128` (144 or 196 for float NN 33..56), sub-group 32 except on the float NN tiled leg (`:105-108`, `choice.hh:115-120`) |
| `reg:m=..:n=..:k=..:u=..` | `gemm_reg`, register-tiled family | float | `grid`, a `reg_configs` entry, `reg_form(*cfg, ta, tb)`, `max_wg >= cfg->threads()` (`:109-115`) |
| `wide:m=..:n=..:k=..` | `gemm_wide`, 16-byte-granule tiles | all | `grid`, a `wide_configs` entry, `wide_form<T>` (a real `Trans` uses a `ConjTrans` instance; a complex `Trans` is refused), `max_wg >= cfg->threads()` (`:116-120`) |
| `vendor` | `backend::gemm_vendor` | all | `d.has_vendor` (`:121`) |

The common native term (`gemm.cc:97-101`):

```cpp
const bool device = d.is_gpu || !d.has_vendor;
const bool native = device && precision == ComputePrecision::Default && s.m > 0 && s.n > 0 && s.k > 0 &&
                    A.batch_size() >= 1 && !A.is_heterogeneous() && !B.is_heterogeneous() && !C.is_heterogeneous();
const bool grid = native && A.batch_size() <= kMaxGridBatch;  // every 3-D launch but small's
```

`device` keeps a CPU queue on the host BLAS vendor (maintainer decision). `grid` is the CUDA grid-z limit: the
batch goes in SYCL dimension 0, so a batch above 65535 (`kMaxGridBatch`, `choice.hh:104`) is native only through
`small`. Candidate counts: float 19, double 9 (no `reg`), complex 8 (no `small`, no `reg`).

Forms are folded after the real-scalar C→T fold:

| choice | forms instantiated |
|---|---|
| `direct`, `tiled`, `small` | any |
| `reg:m=32:n=32:k=8:u=1`, `reg:m=64:n=64:k=8:u=1`, `reg:m=128:n=128:k=8:u=1` | NN |
| `reg:m=64:n=64:k=16:u=1` | NN NT TN TT |
| `reg:m=128:n=32:k=16:u=1`, `reg:m=128:n=32:k=32:u=1` | NN NT TN TT |
| `reg:m=128:n=64:k=16:u=1` | NT TN TT |
| `reg:m=32:n=128:k=16:u=1` | NN TN TT |
| `reg:m=128:n=64:k=32:u=4`, `reg:m=128:n=64:k=32:u=2` | NN |
| `wide:m=64:n=64:k=16` | NN, CN, NC |
| `wide:m=128:n=32:k=16` | NC |
| `wide:m=32:n=128:k=16` | CN |
| `wide:m=32:n=32:k=16`, `wide:m=16:n=16:k=16` | NN (see [blackwell.md](blackwell.md)) |

Removed with no alias: the pin-only `Tiled128x32RegisterK32{S1U1,S2U2,S2U2TT8x4,S2U2TT4x8}`, the experimental
`Tiled128x32RegisterK32{Persistent,SplitK4,S1U4}` and `Tiled128x64RegisterK32LargeTT4x8{,U2}`, and the
`BATCHLAS_GEMM_EXPERIMENTAL` gate. A pin on any of these names throws.

Derived inside the launcher, never a field or a `can_run` term: the transpose instance, the aligned vs
predicated leg (see [the strided-ld fix](#the-strided-ld-defect-and-the-routing-fix)), the `small` bucket, and
TR/TC/stages (`reg_configs`). A launcher throws when handed a config or form it does not instantiate;
`can_run` lists exactly the instantiated forms, so `GemmCandidates.TransposedPinOnAMissingInstantiationThrows`
and `CanRunEqualsLaunch` (`tests/gemm_candidates_tests.cc`) hold.

Selection is the first runnable entry of the nearest row of `tuned/gemm.<dtype>.<device>.txt`, keyed
`ta:exact tb:exact layout:exact m:log n:log k:log batch:log`. ConjTrans folds to `T` for real scalars.
`layout=packed` when A, B and C are contiguous with 16-byte-aligned bases, else `strided`. Every log key has
weight 1 (work ~ m·n·k·batch). The last resort is `direct`, then `vendor` (CPU and `precision != Default`).
gemm takes no workspace. A heterogeneous batch is split into homogeneous items before `select::run` in every
build (`gemm.cc:156-164`), and each item chooses separately.

sm_89 and sm_120 tables are both transcribed; `tuned/gemm.*.sm_120.txt` is the same CSV written for sm_120, so
sm_120 runs the 4090 decision until a `tools/tune` sweep (`tools/tune/gemm_spec.cc`) replaces it. No gemm sweep
has been run (maintainer, 2026-10-05).

**Callers.** Every library gemm goes through the public `gemm` and its table: the blocked drivers' trailing
updates (`src/ops/potrf/potrf.cc:75`, `getrf.cc:74`, `geqrf.cc:46`, `trsm.cc:77`), the level-3 `expand` families
(`src/ops/symm/symm.cc:81-83`, `src/ops/trmm/trmm.cc:90-91`), and the hemm, herk and her2k expansions in
`src/backends/cublas.cc` (`:221`, `:225`, `:381`, `:447`). No caller reaches a kernel the table did not choose.

### Pins and environment

* `BATCHLAS_GEMM_ROUTE`: `auto`, `native`, `vendor`, or a spelling (`reg:m=128:n=128:k=8:u=1`,
  `wide:m=64:n=64:k=16`, `small`). `native` walks the nearest row skipping `vendor`; `vendor` takes the vendor.
  A bad pin throws. Only `native` and `vendor` with nothing runnable in their class fall back to Auto, with a
  warning. Old kernel names (`128x128x8`, `tiled16`, `64x64x16tn`) and router words (`register_tiled`, `sycl`,
  `vendor:auto`) throw (`GemmCandidates.UnknownAndDeletedPinsThrow`).
* `BATCHLAS_GEMM_VARIANT`, `BATCHLAS_GEMM_SYCL_KERNEL` and `BATCHLAS_GEMM_EXPERIMENTAL` are read by nothing.
* Observing a choice: `BATCHLAS_SELECT_TRACE=1` prints the key, choice and runner-up. The coverage `reached`
  row's `chosen_algo` is the spelling. `BATCHLAS_KERNEL_TRACE=1` is the only place the aligned/predicated leg
  shows.

## GEMM: what the transcribed sm_89 rows rank first

Counts read from `tuned/gemm.<dtype>.sm_89.txt` (all four headed `source=transcribed:424a45bc`): the first
entry of every row, and the first entry other than `vendor` (what the native walk and a vendor-free build run).
The real tables have edge rows at batch 1, 63 and 64 and, for double, k = 1 and 2. The complex tables have
batch 128, 2048 and 32768 only.

| dtype | rows | ranked first | first native entry |
|---|---|---|---|
| float | 11226 | `vendor` 11154; `small` 72 (NN squares 1–48 at batch 64, 128, 2048, 32768, both layouts) | `tiled` 5244, `reg:m=128:n=128:k=8:u=1` 2172, `small` 1392, `reg:m=128:n=32:k=32:u=1` 1068, `direct` 792, `reg:m=32:n=32:k=8:u=1` 312, `reg:m=32:n=128:k=16:u=1` 120, `reg:m=128:n=32:k=16:u=1` 120, `reg:m=64:n=64:k=16:u=1` 6 |
| double | 12672 | `tiled` 6956, `vendor` 5440, `direct` 256, `wide:m=64:n=64:k=16` 20 | `tiled` 12078, `direct` 564, `wide:m=64:n=64:k=16` 30 |
| cfloat, cdouble | 7464 each | `vendor` 7464 | `tiled` 3972, `wide:m=32:n=128:k=16` 1146, `wide:m=128:n=32:k=16` 1146, `wide:m=64:n=64:k=16` 1014, `direct` 186 |

What each group of rows encodes (checked row by row against the old predicate, 0 exceptions):

| rows | first entry | measured window |
|---|---|---|
| double, batch ≥ 64 and k ≥ 2 (7232) | native: `tiled` 6956, `direct` 256, `wide` 20 | [Double, the only fully native window](#double-the-only-fully-native-window) |
| double, batch 1 or 63, or k = 1 (5440) | `vendor` | the `batch >= 64` floor and the `k >= 2` boundary |
| double NN, max(m,n,k) ≤ 24; transposed, ≤ 32 | native `direct` | the Direct/Tiled16 boundary |
| double NN packed squares 256–1024 (20) | `wide:m=64:n=64:k=16` | the `min_dim >= 256` arm of [the wide scalar kernel](#the-wide-scalar-kernel); never `strided` |
| float NN squares ≤ 48, batch ≥ 64 (72) | `small` | [The small tiled kernel](#the-small-tiled-kernel) |
| every other float row | `vendor` | [Float NN at max_dim 32](#float-nn-at-max_dim-32) |
| float NN, max(m,n,k) ≥ 128, k ≥ 8, min(m,n) ≥ 64, and (min(m,n) ≥ 128 or k < 128) (2172) | native `reg:m=128:n=128:k=8:u=1`, both layouts | [the strided-ld fix](#the-strided-ld-defect-and-the-routing-fix) |
| float transposed, m ≥ 128, n ≥ 32, k ≥ 128 (1068) | native `reg:m=128:n=32:k=32:u=1`; other transposed `tiled`, or `small` at max ≤ 32 | transposed register exit, 0.34–0.55× of cuBLAS, so vendor first |
| complex NN, min(m,n,k) ≥ 32 (1014 per type) | native `wide:m=64:n=64:k=16` | [the CTA count gate](#the-cta-count-gate-for-complex) |
| complex NC with m ≥ 128, n ≥ 32, k ≥ 8; CN with m ≥ 32, n ≥ 128, k ≥ 8 (1146 each) | native `wide:m=128:n=32:k=16`; `wide:m=32:n=128:k=16` | [Wide-scalar transposed tiles](#wide-scalar-transposed-tiles) |
| every complex row | `vendor` | [Complex is refused](#complex-is-refused) |

**The rows are exact on the grid and approximate off it.** Against the old selector, the first native entry
differs on ~6.7% of random off-grid double shapes and ~31.7% of float ones. Double edges are bracketed only on
squares, and a `wide`/`reg` row transcribed at an aligned packed point also serves packed non-multiple shapes,
which run the predicated leg. The complex grid has no batch below 128, so a batch-8 complex NN call takes the
batch-128 row, and its native walk ranks `wide` first where the CTA ladder measured losses (cfloat 33×61×33 b8
0.56×, cdouble 129×96×129 b8 0.79×, cdouble 33×61×33 b64 0.93×). In a cuBLAS build the vendor leads those rows.

## GEMM: evidence for each boundary

Each subsection names the rows it decides. A timed table replaces a row only after a `tools/tune` sweep of that
device, and the sweep must re-bracket the edges below.

### Double, the only fully native window

**Decides:** all 7232 double rows with batch ≥ 64 and k ≥ 2 rank native first; rows at batch 1 or 63, or k = 1,
rank `vendor` first. There is no squareness, form or size term.

`double`, square NN, batch 512 (4096 for n ≤ 32), GFLOP/s, spread 0.0–0.3%:

| n | native kernel | cuBLAS | native | ratio |
|---|---|---|---|---|
| 4 | Direct | 16.5–16.6 | 58.4–59.0 | 3.55× |
| 8 | Direct | 86.1–86.3 | 387–388 | 4.49–4.51× |
| 16 | Direct | 227–234 | 888 | 3.80–3.92× |
| 24 | Direct | 437–448 | 1130 | 2.52–2.59× |
| 32 | Tiled16 | 982–1018 | 1213 | 1.19–1.23× |
| 48 | Tiled16 | 633–647 | 1204 | 1.86–1.90× |
| 64 | Tiled16 | 1151–1168 | 1214 | 1.04–1.05× |
| 96 | Tiled16 | 903–913 | 1329–1330 | 1.46–1.47× |
| 136 | Tiled16 | 698–703 | 1214–1215 | 1.73–1.74× |
| 200 | Tiled16 | 829–833 | 1270 | 1.53× |
| 256–512 | wide 64×64 | 1239–1246 | 1399–1411 | 1.13× |

**Saturation.** Batch 64 → 128 → 512 → 2048 → 8192 at the two largest margins: n=48 1.58 → 1.76 → 1.90 → 1.95 →
1.96×; n=136 1.82 → 1.73 → 1.74 → 1.75 → 1.75×. The ratio is flat or rising, not pulled toward 1.

**Non-square (E5)**, batch 128, NN/NT/TN: 992×992×32 1.10–1.14×, 480×480×32 1.14–1.17×, 288×288×32 1.21–1.22×,
224×224×32 1.24–1.25×, 248×248×8 1.34–1.41×, 312×312×8 1.21–1.25×. 36 of 36 cells win. Edges: 1024³ 1.13×,
2048³ 1.14×, 4096×64×64 1.04–1.06×, 64×4096×64 1.04–1.05×, 992×992×8 1.39–1.46×. The same shapes measure
0.22–0.51× for float (0 of 36 win), so the widening is double-only.

**The `k >= 2` boundary**, at 512×512×k:

| k | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| ratio | **0.49×** | 1.64× | 1.62× | 1.58× | 1.53× | 1.49× | 1.34× | 1.09× |

k=1 is the only losing double shape measured. Its advantage is β=0 only (cuBLAS 230 → 114 GFLOP/s at β=1; native
112 either way), so cuBLAS has a rank-1 path. k=1 is 761 calls in the demand table, which puts the boundary at 2.

**Transposed double**, square, batch 512, n=32..512, both betas: TN 1.01–1.12×, NT 1.06–1.11×, TT 1.04–1.12×,
CN 1.01–1.11×. The minimum is at n=32. No losses, spreads 0.0–4.2%.

**Why no upper bound.** cuBLAS DGEMM sits at 78–87% of the ~1.44 TFLOP/s FP64 ceiling and native at 88–98%, at
every size from 4 to 2048. This is a ceiling argument, not a measurement above 2048.

**The Direct/Tiled16 boundary at 24** (double, GFLOP/s, batch shown):

| n | batch | Direct | Tiled16 | winner |
|---|---|---|---|---|
| 24 | 512 | 708 | 518 | Direct 1.37× |
| 24 | 4096 | 1126 | 687 | Direct 1.64× |
| 25 | 4096 | 750 | 746 | wash (0.99–1.02× at b512/b4096) |
| 28 | 4096 | 903 | 937 | Tiled16 1.04× |
| 32 | 512 | 903 | 973 | Tiled16 1.08× |
| 32 | 4096 | 938 | 1211 | Tiled16 1.29× |

The grid has 24 and 32 but no 25–31. 24 rather than 25 because 25 is inside the run-to-run spread.

### Float NN at max_dim 32

Square NN, batch 512, both betas:

| n | ratio (β=0, β=1) | verdict |
|---|---|---|
| 8 | 1.43×, 1.46× | keep |
| 16 | 1.22×, 1.31× | keep |
| 32 | 1.03×, 1.08× | keep, the last win |
| **33** | **0.92×, 0.96×** | **the bracketing loss** |
| 48 | 0.58×, 0.60× | |
| 64 | 0.79×, 0.83× | |
| 96 | 0.49×, 0.43× | |
| 127 | 0.36×, 0.44× | |

**Superseded for 33..48 on 2026-09-27**: the 4×4-tiled small kernel takes those squares to 1.15–2.3×, and the
window ends at 48 (see [The small tiled kernel](#the-small-tiled-kernel)).

**Removed windows.** NN 128..512: n=128 0.97–0.98×, n=192 0.39–0.47×, n=256 0.80–0.87×, n=384 0.77–0.79×, n=512
0.91× (flat across batch 128/512/1024). Transposed 128..512: all 30 claimed cells lose, 0.34–0.55×; over
n=64..768 all 48 cells lose, 0.23–0.55×. The transposed register family plateaus near 15–18 TFLOP/s against
cuBLAS SGEMM at 45+.

**Decides:** `tuned/gemm.float.sm_89.txt` ranks `vendor` first in 11154 of 11226 rows. The other 72 are the NN
squares up to 48 at batch ≥ 64, which rank `small`.

### Complex is refused

**Decides:** all 7464 rows of each of `tuned/gemm.cfloat.sm_89.txt` and `tuned/gemm.cdouble.sm_89.txt` rank
`vendor` first.

The native complex kernels ran 3.2–7.1× slower than cuBLAS when the window was set (cdouble 0.386–0.392 vs
1.238–1.246 TFLOP/s; cfloat 6.6–6.9 vs 45–49 TFLOP/s). The transposed wide tiles added later
([Wide-scalar transposed tiles](#wide-scalar-transposed-tiles)) do not beat cuBLAS on the shapes the drivers issue.
A timed complex table is what would reopen the question.

### The 128x128 float kernel

`src/sycl/gemm/register_128x128.hh`: 128×128×8 macro tile, 8×8 accumulators per thread, 256 threads. Its SASS
inner loop matches nvcc's for the same SGEMM body (512 FFMA, 32 `LDS.128`, 2 `BAR.SYNC`, zero spill), and SYCL
runs at 98.7–100.1% of the CUDA build (512³b512, 1024³b64, 256³b1024).

The 2026-09-27 rework had four levers, each measured alone:

1. **Prefetch.** The prefetch registers must be a `ext_vector_type(4)`, not a 16-byte struct. With a struct,
   LLVM sinks the global load to its shared store and there is no prefetch. Paired vs cuBLAS at 512³ b1024,
   β=0: 0.91× → 1.04×.
2. **Lane swizzle.** An `LDS.128` costs 4 wavefronts when its address depends on both lane bits 0 and 1, and 2
   otherwise. Giving m lane bit 0 and n lane bit 1 cuts shared-load wavefronts 805M → 537M at 512³ b1024 (cuBLAS
   549M).
3. **`L2::128B` hint on B.** `ld.global.L2::128B` takes 256³ b4096 from 0.80× to 1.00× and is neutral elsewhere.
4. **Staged epilogue** for k ≤ 64, see [The staged epilogue](#the-staged-epilogue).

The launch bound `min_work_groups_per_cu(2)` (128-register cap) is a guard. 1 → 127 registers, same speed; 2 →
128 registers (shipped); 3 → 80 registers plus 1.4 KB spill, 8.8× slower.

In-tree, `gemm_benchmark` (`BM_GEMM<`), paired `t_vendor / t_native`, new/old in the last column:

| shape | β | new | old | new/old |
|---|---|---|---|---|
| 128³ b16384 | 0 / 1 | 1.01 / 1.01 | 0.93 / 0.99 | 1.09 / 1.02 |
| 256³ b4096 | 0 / 1 | 1.01 / 1.00 | 0.82 / 0.90 | 1.23 / 1.11 |
| 384³ b2048 | 0 / 1 | 1.05–1.14 / 0.99–1.02 | 0.80–0.84 / 0.81–0.83 | 1.30 / 1.22 |
| 512³ b1024 | 0 / 1 | 1.11–1.13 / 1.06–1.11 | 0.87–0.91 / 0.88 | 1.25 / 1.21–1.25 |
| 768³ b256 | 0 / 1 | 1.11–1.15 / 1.07–1.14 | 0.93–0.98 / 0.91–0.94 | 1.13–1.20 / 1.16–1.21 |
| 1024³ b256 | 0 / 1 | 1.12 / 1.12–1.14 | 0.95 / 0.95–0.96 | 1.18 / 1.17–1.20 |
| 2048³ b32 | 0 | 1.12–1.13 | 0.96 | 1.17 |
| 512×256×512 b1024 | 0 | 1.10–1.12 | 0.90 | 1.22–1.24 |
| 2048²×128 b64 | 0 | 1.06 | 0.85 | 1.25 |
| 512²×128 b2048 | 0 | 1.06 | 0.97 | 1.09 |
| 1024²×64, 512²×32, 512²×16, 512²×8 | 0 | 1.00–1.01 | 1.00 | 1.00 |
| 256³ b4096 ld+1 (predicated) | 0 | 0.95 | 0.79 | 1.20 |
| 512³ b1024 ld+1 (predicated) | 1 | 0.89–0.94 | 0.75 | 1.17–1.24 |
| 1024³ b256 ld+1 (predicated) | 0 | 1.03 | 0.82 | 1.26 |
| 320³ b2048 (predicated, ragged) | 0 | 0.93–1.01 | 0.77 | 1.19–1.31 |
| 544³ b512 (predicated, ragged) | 0 | 1.00–1.01 | 0.79 | 1.26–1.28 |

**Hazard.** The box is power-limited (455–475 W, SM clock 1.8–2.8 GHz within one run), so compute-bound cells
move ±5% between runs, and arm order matters (the harness rotates order every rep). A cross-process ratio drifts
with GPU temperature; an interleaved in-process ratio is steadier.

**The float rows above 48 keep `vendor` first**, deliberately. The aligned leg clears 1.11× at 512–2048 on both
betas, but 384 β=1 (0.99–1.02), 768 β=1 (1.07 in one run) and the predicated leg at 512 ld+1 (0.89–0.94) do not.
A table can rank it with `layout` as the key (`packed` = A, B, C contiguous, 16-byte bases) and keep `vendor` on
`strided`, with no leg term in `can_run`. `packed` does not test divisibility by the tile, so ragged packed shapes
run the predicated leg (320³, 544³ above).

**Predicated leg (E4).** Square NN, batch 512 (96 for n ≥ 544), GFLOP/s:

| n | 160 | 192 | 224 | 320 | 544 | 672 | 800 | 1056 |
|---|---|---|---|---|---|---|---|---|
| generic | 7 892 | 9 781 | 11 611 | 12 188 | 13 372 | 14 107 | 14 654 | 15 065 |
| predicated | 13 170 | 18 000 | 22 467 | 25 288 | 27 101 | 29 715 | 31 354 | 33 314 |
| gain | 1.67× | 1.84× | 1.93× | 2.07× | 2.03× | 2.11× | 2.14× | 2.21× |

The unaligned-`ld` cases gain most: n=256 ld+2 7 237 → 23 966 (3.31×), n=512 ld+2 8 399 → 36 862 (4.39×). This
moves the bucket from 0.36–0.51× to 0.72–0.84× of cuBLAS, still a loss, so the rows keep `vendor` first there.
Only the generic leg changed.

**Event-timed, β=1, GFLOP/s:**

| shape | vendor | 128×64×32 (old) | 128×128×8 (new) | vs vendor |
|---|---|---|---|---|
| 128³ b4096 | 14480 | 7223 | 14254 | 98.4% |
| 256³ b1024 | 29187 | 14065 | 25596 | 87.7% |
| 512³ b512 | 40755 | 22672 | 41545 | 101.9% |
| 512×256×512 b512 | 41066 | 21974 | 37044 | 90.2% |
| 512×64×512 b512 | 20298 | 16208 | 17822 | 87.8% |
| 1024³ b64 | 45870 | 24038 | 44062 | 96.1% |

An older claim of parity with cuBLAS SGEMM (43.6 vs 43.9 TFLOP/s at 512³b512) no longer reproduces: the cuBLAS
half now measures 47.3. A ratio against a vendor is only as durable as that vendor's version.

### The wide scalar kernel

`src/sycl/gemm/register_64x64_k16_wide.hh`: 64×64×16 macro tile, 4×4 thread tile, the only register-tiled
variant serving a non-float scalar. Ratios at 256³b512, 512³b128, 1024³b32, both betas:

| scalar | vs `Tiled16` | vs cuBLAS |
|---|---|---|
| `complex<float>` | 7.0–7.7× | 0.98–1.08× CGEMM |
| `complex<double>` | 3.56–3.60× | 1.12× ZGEMM |
| `double` | 1.01–1.08× | 1.07–1.15× DGEMM |
| `float` | — | **0.85–0.93× SGEMM**, so no float row ranks `wide` |

Registers / spill, sm_89: 55/56 (float), 72/76 (double), 72/80 (cfloat), 132/134 (cdouble), zero spill in all 16
entries (`scripts/register_probe.sh`). The `double` row is small on purpose: FP64 on this part caps at ~1.44
TFLOP/s, and `Tiled16` already reaches 92% of it. This conclusion is 4090-specific and inverts on a 1:2-FP64
part.

Load-bearing details (each reverts a measured property if dropped): 16-byte access granule (so an 8-lane LDS
phase covers all 32 banks); `may_alias` on the punning types; a native vector type for the staging copy;
`std::complex` never reaching device code (POD `Cx<R>`, multiply as four `fma`s, no `__mulsc3`); shared strides
exactly `TileM`/`TileN`, with m fastest-varying in the epilogue.

The kernel is `wide:m=64:n=64:k=16` in the tables. It ranks first only in the 20 double NN packed squares 256–1024
(batch ≥ 64). The old `min_dim >= 256` arm fired on 0.64% of real non-float calls (46 of 7223). Structurally,
internal panel updates have a small k, so `min_dim` cannot rise with problem size. No single relaxation rescues
it: zero calls are blocked by the k floor alone.

### The CTA count gate for complex

The old selector admitted complex NN to the wide kernel on `min_dim >= 32 && ctas >= kMinCtas`, with `kMinCtas`
64 for `complex<float>` and 128 for `complex<double>`, where `ctas = ceil(m/64)*ceil(n/64)*batch`. In the tables
it is `wide:m=64:n=64:k=16` in the 1014 NN rows per type with min(m,n,k) ≥ 32. The complex grid's smallest batch
is 128, so the batch half of the gate is not encoded (see the open debts).

Wide vs the route it replaces, saturated, both betas: geomean 3.98× (cfloat) and 2.90× (cdouble) over 116 refused
cells. That number is not the gate: the newly captured call sites run at batch 1–8, where wide loses 12 of 12
cells (cfloat 0.60–0.80×, cdouble down to 0.174×). The crossover is roughly a constant number of work-groups, not
a constant batch: the wide kernel launches up to 16× fewer CTAs than `Tiled16`.

`tiled16_ms / wide_ms`, >1 means wide wins:

| type | shape | b1 | b8 | b16 | b32 | b64 | b128 | b256 |
|---|---|---|---|---|---|---|---|---|
| cfloat | 129×96×129 | 0.58 | 1.02 | 1.60 | 1.93 | 2.65 | 2.75 | 2.73 |
| cfloat | 96×64×96 | 0.64 | 0.77 | 0.96 | 1.64 | 2.75 | 3.68 | 3.88 |
| cfloat | 33×61×33 | 0.60 | 0.56 | 0.65 | 0.75 | 1.08 | 1.55 | 1.98 |
| cdouble | 129×96×129 | 0.21 | 0.79 | 1.37 | 1.38 | 1.79 | 1.77 | 1.76 |
| cdouble | 96×64×96 | 0.24 | 0.45 | 0.67 | 1.31 | 2.59 | 2.60 | 2.58 |
| cdouble | 33×61×33 | 0.17 | 0.18 | 0.33 | 0.48 | 0.93 | 1.82 | 1.82 |

Re-indexed by CTA count: cfloat `ctas >= 64` admits 26 clean cells (worst 1.08×, no losses); cdouble
`ctas >= 128` admits 24 cells (worst 1.08×, no losses). Counterexamples on the other side:

* cfloat below 64: 33×61×33 b32 (32 CTAs) 0.75×. (The source comment cites 129×96×129 b8 at 48 CTAs as 0.79×; the
  ladder reads that cell 0.79× only for cdouble, so the source comment's type is in doubt.)
* cdouble below 128: 33×61×33 b64 (64 CTAs) 0.93×. 64 CTAs is ambiguous for cdouble (0.93× loss and 1.31× win), so 128 is
  chosen to admit no loss, at the cost of a real 1.37× (129×96×129 b16).
* `min_dim >= 32` is needed independently: 16×16×16 loses 0.71× (cfloat) and 0.28× (cdouble); 32³ wins 2.28× and 1.05×.

The old `min_dim >= 256` arm is still checked first, so nothing that routed to the kernel stopped doing so.

### Wide-scalar transposed tiles

`src/sycl/gemm/register_wide_transposed.hh`: four variants, four types, 16 kernels. They are the
`wide:m=64:n=64:k=16` CN/NC instances, `wide:m=128:n=32:k=16` (NC) and `wide:m=32:n=128:k=16` (CN). Work-group
256 for all four.

Why a new kernel: `register_tiled_common.hh` uses `accum += a * b` on `T`, which for `std::complex` is an Annex-G
multiply (an `isnan` branch and a `__mulsc3`/`__muldc3` call). `register_64x64_k16_wide.hh` is NN only. The new
header takes the wide kernel's POD scalar and granule, stages the transposed operand through shared memory with a
transposed store, and keeps global reads coalesced in all four forms. Conjugation is applied once per element at
the staging store, costing nothing per FMA.

The macro tile follows the drivers' shapes, not a square grid: potrf's trailing update and W×W fold use
`NoTrans/ConjTrans` at (nb, W) = (128, 128) for float, (96, 32) for double and cfloat, (64, 16) for cdouble; geqrf's
`W1 = V^H A22` and `W2 = T^H W1` use `ConjTrans/NoTrans` at nb = 32 (16 for double). Every complex transposed
shape in the tree has a dimension of 16 or 32, so the tiles match a panel width: `64x64x16wide_cn` and
`64x64x16wide_nc` (general), `128x32x16wide_nc` (potrf at W = 32), `32x128x16wide_cn` (geqrf at nb = 32).

One instance serves real and complex transposes. `wide_trans_matches<T>` allows a `ConjTrans` instance to serve a
real `Trans`, since conj is the identity for real scalars. For complex, `can_run` (`wide_form` in `gemm.cc`)
refuses the substitution, so Auto skips the entry and a pin throws.

**NC, potrf trailing shape (m × 32 × k, `NoTrans/ConjTrans`), `128x32x16wide_nc`.** Ratios are `arm_ms / native_ms`
at batch 1024, `ld` padded (+8):

| type | m=128 k=32 | 128,96 | 256,32 | 256,96 | 512,32 | 512,96 | 1024,32 | 1024,96 | 256×128×96 |
|---|---|---|---|---|---|---|---|---|---|
| cdouble vs cuBLAS | 1.123 | 1.120 | 1.135 | 1.130 | 1.142 | 1.134 | 1.144 | 1.137 | 1.136 |
| cdouble vs Tiled16 | 3.350 | 3.461 | 3.424 | 3.503 | 3.461 | 3.522 | 3.478 | 3.529 | 3.530 |
| cfloat vs cuBLAS | 0.955 | 1.069 | 1.013 | 1.011 | 1.010 | 1.012 | 1.006 | 1.006 | 0.863 |
| cfloat vs Tiled16 | 1.603 | 1.815 | 1.428 | 1.882 | 1.589 | 2.056 | 1.484 | 2.108 | 2.948 |
| double vs cuBLAS | 1.032 | 1.084 | 1.049 | 1.098 | 1.062 | 1.105 | 1.069 | 1.110 | 1.111 |
| **double vs Tiled16** | **0.919** | **0.970** | **0.954** | **0.990** | **0.968** | **0.995** | **0.972** | **0.997** | **0.998** |
| float vs cuBLAS | 0.806 | 1.062 | 0.712 | 0.917 | 0.827 | 0.941 | 0.844 | 0.942 | 0.773 |
| float vs Tiled16 | 1.373 | 1.975 | 1.252 | 1.834 | 1.556 | 1.999 | 1.622 | 2.098 | 2.713 |

**CN, geqrf panel shape (32 × n × k, `ConjTrans/NoTrans`), `32x128x16wide_cn`:**

| type | n=64 k=128 | 64,512 | 256,32 | 256,128 | 256,512 | 512,128 | 512,512 | 128×256×512 |
|---|---|---|---|---|---|---|---|---|
| cdouble vs cuBLAS | 0.570 | 0.563 | 1.130 | 1.124 | 1.126 | 1.127 | 1.128 | 1.129 |
| cdouble vs Tiled16 | 1.759 | 1.762 | 3.422 | 3.501 | 3.531 | 3.515 | 3.536 | 3.536 |
| cfloat vs cuBLAS | 0.909 | 0.847 | 1.066 | 0.952 | 0.915 | 0.974 | 0.932 | 0.639 |
| cfloat vs Tiled16 | 1.567 | 1.739 | 1.291 | 1.920 | 2.507 | 2.110 | 2.755 | 3.718 |

Against `Tiled16` the tiles win for both complex types. Against cuBLAS only complex<double> clears the R8 bar
(≥ 1.11×). cfloat tops out at 1.069×, and double loses to `Tiled16` (0.92–1.00×), so no double row ranks these
tiles.

**Saturation.** Ratios rise with batch and do not level off inside the memory ceiling. cdouble NC m=512 k=96:
1.099 → 1.120 → 1.134 at batch 64 → 256 → 1024. cdouble CN 32×256×512: 1.052 → 1.112 → 1.126. cfloat NC m=512 k=96
vs Tiled16: 2.881 → 1.984 → 2.056 (not monotone at batch 64). Quoted ratios are at batch 1024.

**Bracketing non-winners** (vs cuBLAS / vs Tiled16):

| edge | cell | vs cuBLAS | vs Tiled16 |
|---|---|---|---|
| m below the 128-row NC tile | cdouble 32×32×96 (potrf W×W fold) | 0.283 | 0.857 |
| n below the 32-col NC tile | cdouble 512×16×64 | 0.581 | 1.789 |
| n below the 128-col CN tile | cdouble 32×64×512 | 0.563 | 1.762 |
| m below the 32-row CN tile | cdouble 16×512×512 | 0.567 | 1.773 |
| k → 1 | cdouble 512×32×1 | 0.346 | 0.563 |
| k = 8 | cdouble 512×32×8 | 1.163 | 1.884 |
| batch below saturation | cdouble 128×32×96 at batch 64 | 0.569 | 1.707 |

There is no measured high-side bracket: cdouble at 512³ batch 256 and 1024×1024×512 batch 128 both read 1.13× of
cuBLAS.

**Ragged sizes.** The first grid used only exact multiples of the tile (64 … 1024 against a 128-wide tile). The
drivers issue `n2 = m - j2` (224, 192, 160, 128, …), which leaves a mostly empty trailing tile. Re-measured:

| cdouble CN, 32 × n × 256, batch 1024 | n=136 | 160 | 192 | 224 | 288 | 384 | 480 |
|---|---|---|---|---|---|---|---|
| vs cuBLAS | 0.637 | 0.707 | 0.846 | 0.985 | 0.847 | **1.125** | 1.057 |
| vs Tiled16 | 1.990 | 2.209 | 2.646 | 3.080 | 2.650 | 3.521 | 3.308 |

| cdouble NC, m × 32 × 96, batch 1024 | m=136 | 160 | 200 | 224 | 288 | 384 | 480 |
|---|---|---|---|---|---|---|---|
| vs cuBLAS | 0.718 | 0.718 | 0.996 | 0.995 | 0.855 | **1.133** | 1.065 |
| vs Tiled16 | 1.988 | 2.209 | 2.859 | 3.078 | 2.649 | 3.515 | 3.308 |

The 1.13× window is visible only at exact tile multiples, so `t_native <= 0.90 t_vendor` is not met on the drivers'
population and `vendor` stays first. Against `Tiled16`, raggedness costs little (1.68–3.52× cdouble, 1.68–2.42×
cfloat), which is why the tiles are the rows' first native entry. **A macro-tiled kernel must be swept at sizes
that are not multiples of its tile**, or the sweep reports its best case as the average.

**Rows.** In `tuned/gemm.cfloat.sm_89.txt` and `tuned/gemm.cdouble.sm_89.txt` (0 exceptions):

| rows (per type) | first entry | first native entry |
|---|---|---|
| NC with m ≥ 128, n ≥ 32, k ≥ 8: 1146 | `vendor` | `wide:m=128:n=32:k=16` |
| CN with m ≥ 32, n ≥ 128, k ≥ 8: 1146 | `vendor` | `wide:m=32:n=128:k=16` |
| other NC and CN: 330 each | `vendor` | `tiled` (300) or `direct` (30) |
| NT, TN, CT (complex `Trans`): 3036 | `vendor` | `tiled` or `direct` (`wide_form` refuses a complex `Trans`) |

The tiles therefore run in a vendor-free or ROCm build and under `BATCHLAS_GEMM_ROUTE=native`. A cuBLAS build runs
the vendor. The gain is 1.68–3.52× in a vendor-free build and nothing in a vendor build.

**Register residency** (`scripts/register_probe.sh`, zero spill in all 576 entries):

| tile / form | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 64×64 CN | 43 | 66 | 72 | 128 |
| 64×64 NC | 45 | 64 | 72 | 132 |
| 128×32 NC | 47 | 64 | 72 | 124 |
| 32×128 CN | 53 | 60 | 72 | 128 |

The cdouble column (124–132 registers) caps a work-group at 512 work-items, i.e. 33% occupancy.

**Armed breaks** (`gemm_tests --gtest_filter='GemmTest/*.WideTransposed*'`, 36 live cases):

| break | observed red |
|---|---|
| transposed A or B staging forms the `NoTrans` address | 14 each (the CN or NC tests plus the real-widening test) |
| drop `dev_conj` on A or on B | 6 each, exactly the complex tests |
| drop the epilogue `col >= n` or `row >= m` guard | 18 each |
| swap the transposed-store index decomposition | 0: a performance break, visible only in an `ncu` sector count |
| drop the complex refusal in `wide_trans_matches<T>` | 4, the complex widening tests |
| drop the staging bounds test on A | 0: the B-side test zeroes the product, so the A-side test guards only out-of-allocation reads |

The dispatch breaks (selector row, NC gate, route CTA floor, route window per type, the `k >= 8` exclusion) each
turn exactly the test that names them red, plus the route-equivalence invariant.

**Not established:**

* The A/B harness behind these numbers is not in the tree. `tools/tune/batchlas_tune` with `gemm_spec.cc` now
  pins, interleaves and verifies every candidate, but none of these numbers was re-taken with it.
* No `ncu` reading was taken for coalescing or the 40 KB cdouble tiles. All timings are wall-clock.
* The 16-wide tiles (potrf complex<double> at W = 16, geqrf double at nb = 16) are not built. The n = 16 cell
  measured 1.789× of `Tiled16`.
* No upper bound on m, n or k is measured (largest 1024×1024×512). k = 2..7 is unmeasured and refused.
* No float row ranks these tiles, though float beats `Tiled16` by 1.25–2.71× on the NC shapes. They were not A/B'd
  against `reg:m=128:n=32:k=32:u=1`.
* The plan's premise "float transposed is 0.23–0.55× across 48 of 48 cells" describes a square grid. On the panel
  shapes the drivers issue, float native reads 0.71–1.24× of cuBLAS.

### The strided ld defect and the routing fix

Every operand `trsm` hands GEMM is a sub-view carrying its parent's leading dimension (a 128-row `C` with
`ld = 512`). On the six shapes `trsm` issues at order 512 (float, q=1024, batch 512), native/vendor at the real `ld`
and at `ld == rows`:

| shape (m × n × k) | real `ld`: native / vendor (ms) | ratio | `ld == rows`: native (ms) |
|---|---|---|---|
| 128 × 1024 × 128 | 1.53 / 0.96 | 0.62× | 0.98 |
| 128 × 1024 × 256 | 2.73 / 1.31 | 0.48× | 2.35 |
| 128 × 1024 × 384 | 3.78 / 1.63 | 0.43× | 3.49 |
| 32 × 1024 × 32 | 0.406 / 0.235 | 0.58× | 0.248 (0.86–0.98× of vendor) |
| 32 × 1024 × 64 | 0.680 / 0.335 | 0.49× | 0.356 |
| 32 × 1024 × 96 | 0.887 / 0.426 | 0.48× | 0.487 |

cuBLAS barely moves, and no square benchmark sees this. Routing the trailing updates through the gemm router
instead of the native `gemm_custom` took the n=512 solve from 18.8 ms to 11.19 ms (vendor `trsm` 14.28 ms), with no
kernel change.

**ncu, m=128 n=1024 k=128 b512 β=1, pad 0 vs pad 384.** Every transaction counter is identical (2,097,152 load
requests, 16.00 sectors per request, identical DRAM sectors, identical instructions, 119 registers). Only time
moves: 917.3 → 1493.1 µs (1.63×), DRAM throughput 89% → 55%. cuBLAS pays 1.05× on the same shape. The regression is
exposed global-load latency at the k-loop barrier (barrier 68% of the added stall). It belongs to operand B: pad
on B alone costs 96% of the penalty, A 0.7%, C 9.8%. B is read as 32 bytes from each of 16 columns per warp, and
those streams are `ldb*4` bytes apart. It is a slope, not a cliff (monotone in pad, beta-independent). Footprint is
ruled out: a 4× allocation at `ld == rows` costs nothing. No mechanism below L2 is established.

**The fix was routing.** `can_use_128x128_fast_path` was a leg predicate, but the old selector also used it as a
routing gate, so failing it handed the call to a much slower kernel. Routing by a shape-only gate (`gemm_kernels.cc:575`
in the parent tree) gives geomean 1.74× (pad 0) and 1.75× (pad 384) over 12 shapes, moving native from 0.58× to
0.99× of cuBLAS at `ld == rows` and 0.54× to 0.93× strided. Examples: 1024×1024×64 b128 ld1408 3.187 → 1.337 ms,
1000×1024×128 b128 ld1384 2.954 → 1.569 ms, 128×128×8 b512 0.074 → 0.030 ms. It also removes the `ld % 4 != 0`
cliff (pad 1: 1.874 → 1.003 ms). The gate's bounds each have a counterexample: `mn_min >= 64` (32×1024×32 is a
wash), `mn_min >= 128` when k >= 128 (64×64×512 b512 0.77/0.69), `max_dim >= 128` (64×64×64 is 1.02×), `k >= 8`
(TileK; 1024×1024×8 wins 2.00×).

**Reach.** With cuBLAS present this changes no runtime, because the float rows the gate covers rank `vendor` first.
It matters for vendor-free and ROCm builds and the native walk. Since P3.4 the gate is not code: `RegCfg::aligned_leg`
(`src/ops/gemm/choice.hh`) names the unpredicated configs, the launcher picks the leg per call, and `can_run` never
reads alignment. The shape gate survives as data: `reg:m=128:n=128:k=8:u=1` is the first native entry of the 2172
float NN rows that satisfy it (936 packed, 1236 strided), and of no other row.

## GEMM: negative results

| Alternative | Result | Verdict |
|---|---|---|
| Double-buffering the 128×128 k-loop (2024) | 127 registers, zero spill, halved barriers, same sector count; 0 time recovered | Refuted 2026-09-27: the prefetch was sunk to its store (see [lever 1](#the-128x128-float-kernel)) |
| Lane maps that keep both lane bits in one operand (8×4, 4×8, 2×16) | 3.0 wavefronts per LDS, ±1% | Dead |
| Padding the B tile stride to 132 | −1–2% | Dead |
| Skipping the last-slab reload; `st.global.cs` C stores; 16-deep k slab | neutral; 0.982 → 0.982; 0.95× at 256³ before the L2 hint, dominated after it | Dead |
| Packing B into contiguous scratch | pays at the same roofline, loses as m grows | Dead |
| Mechanism for the `ld` defect (epilogue, odd tile strides) | The shapes never ran that file; they ran `Tiled128x128RegisterK8` with the aligned leg in both columns | Refuted; confirm the running kernel first |
| Wide-scalar tile for float | 0.85–0.93× of cuBLAS SGEMM | Dead |
| 128×128 8×8 tile for wide scalars | double 208 registers, cfloat 247, zero spill; cdouble spills 3.4 KB and fails to launch (208 × 512 > 65 536 registers) | Not launchable for cdouble |
| `complex-split` candidate | matches the 64×64 tile at 247 registers, 1 block/SM | Not landed; 64×64 is the only candidate with no unlaunchable or spilling config |
| FFMA:shared-load ratio as a wide-scalar lever | complex within 5% across 16:1–32:1; double within 4% across 4:1–16:1 | Not a lever for complex or double; it is for float |
| Bare `min_dim >= 32` floor for the complex relaxation | 12 of 12 losses at batch 1–8 | Dead |
| Float native windows (NN 128..512, transposed 128..512, non-square demand shapes) | 0.23–0.55× in all measured cells | Dead |

## GEMM: correctness findings

* **Nine transposed launchers computed the wrong answer for `ConjTrans`** (fixed in `f236575`). They hard-wire
  OpA/OpB, so a `<Trans, NoTrans>` instance dropped the conjugation silently. Forceable only by name, which is how
  a benchmark compares variants. The guard is `ForcedTransposedLauncherRejectsMismatchedTransposeForm`
  (`tests/gemm_tests.cc`), which checks against `Tiled16`, not the vendor (inert in a vendor-free build).
* **Eighteen NN-only register variants computed NN on a transposed call** (fixed in P3.4). `can_run` lists the
  instantiated forms per config (`reg_form`, `wide_form`, `src/ops/gemm/gemm.cc:76-86`), so such a pin throws.
* **Heterogeneous batches**: members with `m == 0` or `n == 0` are skipped, a `k == 0` member is `C := beta*C`, and
  an all-skipped batch returns a valid `Event`. Vendor-free these did not exist, which caused all 17 remaining
  vendor-free `gemm_tests` failures. Vendor-free `gemm_tests` now pass 184/184.
* **The β=0 microbenchmark is blind to epilogue defects.** The first 128×128 epilogue made m the slow-varying thread
  index: 26.0 TFLOP/s at β=0 vs 41 for the standalone kernel. Making m fastest-varying gave 41.1. Measure both betas.
* **Event-profiling sums over queued submissions do not measure kernel time.** 30 submissions summed to 19.836 ms
  for a 3.15 ms kernel (queue wait included). Any SYCL-vs-CUDA-event comparison timed this way is suspect.
* **An equivalence test must assert its own exception list.** The deleted `route_gemm_equivalence_tests.cc`
  counted divergences separately, and asserted that unset and `"auto"` agree. The transcribed tables plus the
  34-cell live cross-check replace it.

## The subgroup workspace budget

`kSubgroupWorkspaceBudgetBytes` (`include/batchlas/blas/device/detail/group_blas_subgroup_common.hh:58`) is
`device_limits::subgroup_workspace_budget_bytes()`, generated from `cmake/device_limits.h.in`. CMake sets it per
architecture as that architecture's table local memory less a 4 KiB reserve, with a 16 KiB floor:

| architecture | table local memory | budget (bytes) |
|---|---|---|
| `nvidia_gpu_sm_*` | 49,152 | **45,056** |
| `amd_gpu_gfx*` | 65,536 | **61,440** |
| `intel*` | 65,536 | **61,440** |
| unrecognised GPU | 32,768 | **28,672** |

It is deliberately not the device's real local memory (101,376 bytes on sm_89). Raising it would retune five ops
as a side effect. Every run-time capacity queries `DeviceProperty::LOCAL_MEM_SIZE` instead
(`src/util/resident_capacity.hh`).

Five `if constexpr` predicates compare a workspace struct with the budget: `register_matrix_workspace_supported_v`,
`complex_rank2k_workspace_supported_v`, `complex_rank2k_in_kernel_workspace_supported_v`,
`optimized_gemm_workspace_supported_v`, `gemm_workspace_supported_v`. `group_blas_rankk.hh` is shared by symm,
herk, syrk and syr2k, so a change here moves five ops.

A budget of 45,056 on a 61,440 part keeps every variant but cuts staging width: 6 instead of 8 subgroups for double
and cfloat, and 1 instead of 3 (register-matrix path) for cdouble. The unrecognised-GPU fallback (28,672) removes
the `optimized_gemm` and `gemm` paths for float, and the register-matrix and complex rank-2k paths for cdouble.

No benchmark or test has timed any budget except the shipped one; the table is a compile-time census, not a cost.
A cache entry (`BATCHLAS_DEVICE_GEMM_TILE_CAP_BYTES`, default `0` = derive per architecture) replaced the old
`BATCHLAS_DEVICE_GEMM_WORKSPACE_CAP_BYTES`, whose cached default of 45056 overrode the derivation on every
architecture in older build trees. Configure migrates a cached legacy `45056` and warns on other cached values.

## GEMM: the heterogeneous-batch loop

A heterogeneous batch cannot use a strided or pointer-batched vendor call, so every backend walks the batch and
issues one single-matrix GEMM per member. The loop is `src/backends/gemm_heterogeneous.hh`
(`detail::gemm_heterogeneous_loop`), shared by the cuBLAS and rocBLAS backends and the vendor-free facade. The
per-item terminal is the only parameter. The public `gemm` splits a heterogeneous batch before choosing, so each
member selects through the table; the backends' loops serve direct `gemm_vendor` callers only. The empty-batch
Event is `create_event_after_external_work()`, which is right for cuBLAS and rocBLAS because their work leaves the
SYCL queue. An MKL backend, whose work is submitted to the queue, would need the queue's own event instead.

## GEMM: the POD device scalar

`src/sycl/device_scalar.hh`. `std::complex` must not reach device code: its `operator*` is Annex-G conformant
(an `isnan` branch and a call to `__mulsc3`/`__muldc3`). Launchers re-type operands and scalars to the aggregate
`Cx<R>` at the pointer boundary, which is layout-compatible with `std::complex`. The PTX of the GEMM instances
using it has zero `__mulsc3`, `__muldc3` and `call.uni`.

TRSM added division (Smith's algorithm, not \f$1/(c+di) = (c-di)/(c^2+d^2)\f$, which overflows above about
\f$10^{19}\f$ in float or \f$10^{154}\f$ in double and then returns 0 silently), conjugation and a finiteness test.
POTRF uses real-component helpers (`dev_real`, `dev_div_real`, ...) because a Cholesky diagonal is real. The
division/reciprocal asymmetry is deliberate: reference `?trsm` divides and reference `?potf2` multiplies by a
reciprocal, and unifying them would move one off its LAPACK rounding.

## GEMM: the register-tiled launcher table

`src/sycl/gemm/register_launchers.hh`. `RegTile` holds the template parameters of `launch_register_tiled<>` as a
structural type, so one launcher `launch_reg<T, RegTile{...}>` serves every row of `reg_configs`
(`src/ops/gemm/choice.hh`, turned into a `RegTile` by `launch_reg_cfg` in `src/sycl/gemm_kernels.cc`).
**Trap:** the defaults differ from `launch_register_tiled<>`. There `ThreadTileCols` defaults to `ThreadTileRows`;
here `TR` and `TC` default independently to 4, so every row states both.

## The small batched kernel

`src/sycl/gemm/small_batched.hh` (`small`). Float NN at max(m,n,k) ≤ 48 previously ran `Direct`, whose adjacent
lanes read B and write C `ld` apart. The kernel gives a 128-lane group 128 / (2·NB) matrices, with NB in {8, 16,
32, 64} from max(m,n,k). op(B) is staged in local memory (odd `ld`, so either transpose is coalesced), each lane
holds one row of op(A) and NB/2 columns of C, and β = 0 skips the C read. Real scalars only: `std::complex` is the
Annex-G trap, so `small` is not a complex candidate.

Batch 32768 unless noted, `BM_GEMM` square NN, β = 0, ms:

| n | cuBLAS | before | small | vs cuBLAS |
|---:|---:|---:|---:|---:|
| 8 | 0.175 | 0.024 (Direct) | 0.021 | 8.5× |
| 16 | 0.358 | 0.235 (Direct) | 0.133 | 2.69× |
| 24 | 1.008 | 0.842 (Direct) | 0.361 | 2.79× |
| 32 | 1.161 | 1.747 (Direct) | 0.608 | **1.91×** (was 0.64×) |
| 40 | 1.267 | 3.670 (Tiled16) | 1.474 | 0.86× (was 0.35×) |
| 48 | 1.524 | 6.087 (Tiled16) | 1.741 | 0.88× (was 0.25×) |
| 64 | 2.419 | 2.612 (32×32 reg) | 2.365 | **1.02×** (was 0.93×) |

`small` is the first native entry of every float row with max(m,n,k) ≤ 32 (NN 948, transposed 168) and of the NN
rows with `min_dim > 32` up to 64 (276). Non-square shapes above 32 keep `direct` or `tiled` first. double measured
at parity with Direct/Tiled16, so `small` is a double candidate that no double row ranks first. The NB = 64 bucket
is shared-load bound; 40..48 still lose to cuBLAS, which is why NN 33..56 has its own kernel below.

## The staged epilogue

At k ≤ 64 the 128×128 kernel is store-bound (512²×32 b4096 writes 4.3 GB of C). The direct epilogue stores four
128-byte column pieces per warp instruction, and DRAM reaches 87.6% of peak against 89.6% for two 256-byte pieces,
a 2–3% loss at k ≤ 32. The staged epilogue routes C through local memory (4 passes of 32 columns × 128 rows) so each
warp stores whole 128-row columns (512 B per `STG.128`). `kStagedEpilogueMaxK = 64`, aligned leg only.

| k (512², β=0) | direct | staged |
|---|---|---|
| 32 (b4096) | 0.982 | 1.002 |
| 64 (b4096) | 0.994 | 1.000 |
| 128 (b2048) | 1.003 | 1.004 |
| 256 (b2048) | 1.18 | 1.12 |

Result after the gate: 1.004–1.011 vs the old kernel at k ≤ 64, at or above cuBLAS. The first version spilled 36 bytes
at the 128-register cap; decoding the slot from `nb` alone fixed it.

## The small tiled kernel

`small_batched.hh`, float NN with 32 < max(m,n,k) ≤ 56, measured 2026-09-27. One matrix per work-group of (NB/4)²
lanes (NB = 48 or 56). A is staged `[k][m]` and B as stored, both in local memory, and each lane holds a 4×4 register
tile of C: one `LDS.128` of A and a quarter of four `LDS.128` of B per 16 FMAs. At β ≠ 0 each lane reads its C tile
before the barrier (48³ β=1: 0.91× → 1.09×). At β = 0 that prefetch is compiled out, because its 27 registers cost
12% at 56³. 48³ b32768 moves 906 MB and the kernel sustains ~860 GB/s.

`gemm_benchmark`, paired `t_vendor / t_native`, square NN (new, old in parentheses):

| n, batch | β=0 | β=1 |
|---|---|---|
| 33, 32768 | 2.30 (1.25) | 1.94 (0.99) |
| 40, 32768 | 1.62 (1.06) | 1.37 (0.80) |
| 44, 32768 | 1.56 (1.11) | 1.32 (0.81) |
| 48, 32768 | 1.36 (1.06) | 1.15 (0.82) |
| 48, 131072 | 1.36 (1.06) | 1.16 (0.83) |
| 56, 32768 | 1.23 (1.12) | 1.08 (0.79) |
| 64, 16384 | 1.05 (1.05), unchanged kernel | |
| 48, 64 / 256 / 1024 | 1.33 / 1.44 / 1.90 | 1.43 / — / 2.64 |
| 40, 64 / 1024 | 1.34 / 1.81 | b256: 1.75 |

57..64 keeps the NB = 64 kernel (the tiled NB = 64 variant measured 0.89–0.91× at 64³ β=0). The window is
float, square NN, batch ≥ 64, through 48. 49..56 runs the tiled kernel natively but stays vendor first, because 56³
β=1 is 1.08×. Non-square NN with min_dim ≤ 32 and max 33..56 is unmeasured.

**In the table:** `small` ranks first in exactly 72 rows of `tuned/gemm.float.sm_89.txt`: the NN squares 1, 2, 4, 8,
16, 24, 32, 40 and 48 at batch 64, 128, 2048 and 32768, both layouts. The squares 49, 56, 64 and every batch-1 or
batch-63 square rank `vendor` first. The edges are checked by `GemmTranscribedTable.Sm89BracketsTheOldVendorEdges`
(48³ `small`, 49³ and 52³ `vendor`, 32³ b16 `vendor`, b64 `small`). The (NB/4)² work-group and sub-group-32 need
outside the tiled leg are `can_run` terms (`small_wg`, `small_fits` in `choice.hh`).

## GEMM: open debts

* **Complex is vendor-dependent in a cuBLAS build**, and measured: the transposed tiles beat `Tiled16` but lose to
  cuBLAS on ragged `n`. See [Wide-scalar transposed tiles](#wide-scalar-transposed-tiles).
* **The complex batch edge of the CTA gate is not in the table.** A complex NN call below batch 128 takes a
  batch-128 row, and its native walk ranks `wide` where the gate measured losses (0.17–0.96× of `Tiled16`). Only a
  vendor-free build pays it. The fix is a grid edge row.
* **The wide kernel's predicated leg has never been timed against `Tiled16`.** It is correct (round-off on 70×53×37).
  Auto reaches it on packed non-multiple shapes near a transcribed `wide` row (double 304³ b64, syev double n=300's
  update), where the pre-P3.4 selector ran `Tiled16`.
* **Vendor-free heterogeneous GEMM is ~7 GFLOP/s** against a ~47 TFLOP/s FP32 peak. It is launch-bound (one launch per
  batch member); a single-launch variant is deferred.
* **The `ld` slope has no established mechanism below L2.** Sector counts, hit rates and DRAM sector counts are
  unchanged. ncu exposes no row-activate counter, so row-buffer locality was not measured.
* **The routing geomean's 12-cell subset is not identified.** `experiments/wp4_gemm_ld/routing/summary.csv` holds 15
  cells (geomean 1.51 / 1.53); the four quoted cells reproduce.
* **The demand tables are `ctest` coverage captures, not user workloads.** No capture of user workloads exists, so
  the small-batch population the CTA gate protects is test-suite evidence.
* **The double window reaches past its measurements** (largest measured 2048³).
* **sm_120 has no measured gemm table.** Its windows in [blackwell.md](blackwell.md) (gemm section)
  are not reachable by Auto until a `tools/tune` sweep.
* **TF32 is reachable but unmeasured.** `experiments/sycl_vs_cuda/tf32_smoke.cpp` emits real `mma.sync` TF32
  instructions with correct results. No staging, so no throughput number. Every native `can_run` rejects
  `ComputePrecision != Default`.
* **The direct kernel's batch offsets are `int`.** `gemm_direct` can overflow at large batch × stride. Not a
  `can_run` term.

## GEMM: raw evidence

Raw data is kept at the git tag `perf-evidence/vendor-independence`. Retrieve a path with
`git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| Double window n=4..512, saturation, Direct/Tiled16 boundary | `experiments/wp2_e3/` |
| Float NN and transposed windows; predicated 128×128 fix | `experiments/wp2_e4/` |
| Non-square double, demand shapes, k=1 boundary | `experiments/wp2_e5/` |
| Auto flip: prediction, route diff, transposed double | `experiments/wp2_e6/` |
| Wide-scalar bake-off, PTX/ptxas evidence, cuBLAS baselines | `experiments/wide_scalar_gemm/` |
| SYCL-vs-CUDA parity, SASS counts, TF32 probe | `experiments/sycl_vs_cuda/FINDINGS.md` |
| strided-`ld` ncu campaign and routing fix | `experiments/wp4_gemm_ld/` |
| Complex routing defect, merge gate, CTA ladder | `experiments/wp4_complex/` |
| Trailing-update GEMM inside `trsm`, sub-view `ld` | `experiments/wp3_s16/` |

Transposed-tile CSVs (in tree, under `benchmarks/results/`): `p6_gemm_{nc_potrf_shapes_complex,nc_potrf_shapes_real,cn_geqrf_shapes_complex,bracket_cells,ragged_tile_sweep}.csv`,
`p6_e2e_before.csv` and `p6_e2e_with_refused_window.csv`.
