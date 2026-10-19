# GEMM {#perf_gemm}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda

gemm selects through flat kernel selection (@ref design_flat_selection). This page lists the choices, the
`tuned/gemm.*` rows that Auto ranks first on sm_89, and the measurements behind each window. The sm_89 table is
transcribed from the old router (`424a45bc`) and is untimed, so these windows are what Auto runs on sm_89. The
sm_120 table is deep-measured (`tuned/README.md`).

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
views, calls `select::run` (`gemm.cc:172-180`), and launches one kernel from `src/ops/gemm/kernels/gemm_kernels.cc`.

| spelling | launcher | scalars | `can_run` beyond the common term (`gemm.cc`) |
|---|---|---|---|
| `direct` | `gemm_direct`, one work-item per C element | all | `grid && max_wg >= 64` (`:103`) |
| `tiled` | `gemm_tiled`, 16×16 shared tile | all | `grid && max_wg >= 256` (`:104`) |
| `small` | `gemm_small`, `small_batched.hh` | all | `max(m,n,k) <= small_max_dim<T>()` (64; 32 for complex<double>), `max_wg >= 128` (144 or 196 for float NN 33..56), sub-group 32 except on the float NN tiled leg, `small_slm_bytes` within the budget (`small_fits`, `src/ops/gemm/choice.hh`) |
| `reg:m=..:n=..:k=..:u=..` | `gemm_reg`, register-tiled family | float | `grid`, a `reg_configs` entry, `reg_form(*cfg, ta, tb)`, `max_wg >= cfg->threads()` (`:109-115`) |
| `wide:m=..:n=..:k=..` | `gemm_wide`, 16-byte-granule tiles | all | `grid`, a `wide_configs` entry, `wide_form<T>` (`wide_serves<T>` in `choice.hh`: a complex form must be listed exactly; a real `Trans` uses the listed `ConjTrans` instance), `max_wg >= cfg->threads()` (`:116-120`) |
| `splitk:m=..:n=..:k=..:s=..` | `gemm_splitk`, a wide host tile over s k-slices plus a reduction (WP-L, *unmeasured*) | all | `grid`, s ∈ {2, 4, 8}, batch × s ≤ min(`max_groups_yz`, INT_MAX), a `splitk_host` row serving the form, the host's `max_wg` and local-memory terms, ldw × n ≤ INT_MAX, partials ≤ `scratch_budget` |
| `vendor` | `backend::gemm_vendor` | all | `d.has_vendor` (`:121`) |

The common native term (`gemm.cc:97-101`):

```cpp
const bool device = d.is_gpu || !d.has_vendor;
const bool native = device && precision == ComputePrecision::Default && s.m > 0 && s.n > 0 && s.k > 0 &&
                    A.batch_size() >= 1 && !A.is_heterogeneous() && !B.is_heterogeneous() && !C.is_heterogeneous();
const bool grid = native && A.batch_size() <= kMaxGridBatch;  // every 3-D launch but small's
```

`device` keeps a CPU queue on the host BLAS vendor (maintainer decision). `grid` is the CUDA grid-z limit: the
batch goes in SYCL dimension 0, so a batch above 65535 (`kMaxGridBatch`, `src/ops/gemm/choice.hh:139`) is native only through
`small`. Candidate counts: float 23, double 12 (no `reg`), complex 12 (no `reg`).

Forms are folded after the real-scalar C→T fold:

| choice | forms instantiated |
|---|---|
| `direct`, `tiled`, `small` | any |
| `reg:m=32:n=32:k=8:u=1`, `reg:m=64:n=64:k=8:u=1` | NN |
| `reg:m=64:n=64:k=16:u=1`, `reg:m=128:n=128:k=8:u=1`, `reg:m=128:n=128:k=8:u=4` | NN NT TN TT |
| `reg:m=128:n=32:k=16:u=1`, `reg:m=128:n=32:k=32:u=1` | NN NT TN TT |
| `reg:m=128:n=64:k=16:u=1` | NT TN TT |
| `reg:m=32:n=128:k=16:u=1` | NN TN TT |
| `reg:m=128:n=64:k=32:u=4`, `reg:m=128:n=64:k=32:u=2` | NN |
| `wide:m=64:n=64:k=16`, `wide:m=64:n=64:k=8` | all nine complex forms; real NN, NT, TN, TT (see [the k=8 slab](#gemm-wp-f-a-k8-slab-for-the-64x64-wide-tile)) |
| `wide:m=128:n=32:k=16` | complex NC, NT, TN, CN; real NT, TN |
| `wide:m=32:n=128:k=16` | complex CN, TN, NT, NC; real NT, TN |
| `wide:m=32:n=32:k=16`, `wide:m=16:n=16:k=16` | complex NN, NT, NC, TN, CN, CT; real NN, NT, TN (see [blackwell.md](blackwell.md)) |
| `wide:m=64:n=32:k=16`, `wide:m=32:n=64:k=16` | complex NN, NT, NC, TN, CN, CT; real NN, NT, TN (see [the skinny tiles](#gemm-skinny-64x32-and-32x64-wide-tiles)) |

The wide forms are the complex instantiations; a real scalar compiles only their N/C builds, so a real form is
served when its C-folded form is listed. See [GEMM: wide instantiations in every transpose
form](#gemm-wide-instantiations-in-every-transpose-form).

Removed with no alias: the pin-only `Tiled128x32RegisterK32{S1U1,S2U2,S2U2TT8x4,S2U2TT4x8}`, the experimental
`Tiled128x32RegisterK32{Persistent,SplitK4,S1U4}` and `Tiled128x64RegisterK32LargeTT4x8{,U2}`, and the
`BATCHLAS_GEMM_EXPERIMENTAL` gate. A pin on any of these names throws.

Derived inside the launcher, never a field or a `can_run` term: the transpose instance, the aligned vs
predicated leg (see [the strided-ld fix](#the-strided-ld-defect-and-the-routing-fix)), the `small` bucket, and
TR/TC/stages (`reg_configs`). A launcher throws when handed a config or form it does not instantiate;
`can_run` lists exactly the instantiated forms, so `GemmCandidates.TransposedPinOnAMissingInstantiationThrows`
and `CanRunEqualsLaunch` (`tests/gemm_candidates_tests.cc`) hold.

Selection is the first runnable entry of the nearest row of `tuned/gemm.<device>.txt`, keyed
`ta:exact tb:exact layout:exact m:log n:log k:log batch:log`. ConjTrans folds to `T` for real scalars.
`layout=packed` when A, B and C are contiguous with 16-byte-aligned bases, else `strided`. Every log key has
weight 1 (work ~ m·n·k·batch). The last resort is `direct`, then `vendor` (CPU and `precision != Default`).
gemm takes no workspace. A heterogeneous batch is split into homogeneous items before `select::run` in every
build (`gemm.cc:156-164`), and each item chooses separately.

The sm_89 table is transcribed. `tuned/gemm.sm_120.txt` comes from the `--tier deep` campaign
(`tools/tune/gemm_spec.cc`): 32225 deep rows, plus 14733 `transcribed` rows where the lattice measured nothing near.

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

Counts read from `tuned/gemm.sm_89.txt` (every row `transcribed`, run `transcribed-424a45bc`): the first
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

**Decides:** the float rows of `tuned/gemm.sm_89.txt` rank `vendor` first in 11154 of 11226 rows. The other 72 are the NN
squares up to 48 at batch ≥ 64, which rank `small`.

### Complex is refused

**Decides:** all 7464 cfloat and all 7464 cdouble rows of `tuned/gemm.sm_89.txt` rank
`vendor` first.

The native complex kernels ran 3.2–7.1× slower than cuBLAS when the window was set (cdouble 0.386–0.392 vs
1.238–1.246 TFLOP/s; cfloat 6.6–6.9 vs 45–49 TFLOP/s). The transposed wide tiles added later
([Wide-scalar transposed tiles](#wide-scalar-transposed-tiles)) do not beat cuBLAS on the shapes the drivers issue.
A timed complex table is what would reopen the question.

### The 128x128 float kernel

`src/ops/gemm/kernels/register_128x128.hh`: 128×128×8 macro tile, 8×8 accumulators per thread, 256 threads. Its SASS
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

`src/ops/gemm/kernels/register_64x64_k16_wide.hh`: 64×64×16 macro tile, 4×4 thread tile, the only register-tiled
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

`src/ops/gemm/kernels/register_wide_transposed.hh`: four variants, four types, 16 kernels. They are the
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

**Rows.** In the cfloat and cdouble rows of `tuned/gemm.sm_89.txt` (0 exceptions):

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
routing gate, so failing it handed the call to a much slower kernel. Routing by a shape-only gate (`gemm_kernels.cc` line 575
at `08d07619^`) gives geomean 1.74× (pad 0) and 1.75× (pad 384) over 12 shapes, moving native from 0.58× to
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
The budget has no override: it is always derived from the architecture table
(`batchlas_subgroup_workspace_budget_bytes`, `cmake/BatchLASDetectSYCL.cmake`).

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

## GEMM: wide instantiations in every transpose form

**Unmeasured: CUDA was down at implementation time** (2026-10-19, `cuInit` 999 on the RTX 4090 box). Nothing here
is a timing; the A/B and the regenerated sm_89 table are pending.

Before this change `WideCfg` carried only `nn, cn, nc`, so every complex NT, TN, CT, TC, CC and TT call fell to
`tiled` (one output per work-item, Annex-G complex multiply), about 8 TFLOP/s against cuBLAS's 45-51 on the
2026-10-18 coarse run: cfloat NT/TN/CT was 55.8% of the summed log-loss against the vendor. The kernel in
`register_wide_transposed.hh` already takes `OpA` and `OpB` apart (`TransA`, `ConjA`, `TransB`, `ConjB`), so the
change is new instantiations only:

* `WideCfg::forms` is a 3×3 bit mask over (op(A), op(B)) ∈ {N, T, C}² (`wide_bit(a, b) = 1 << (3a + b)`).
  `wide_serves<T>` is the one rule: a complex form must be listed exactly; a real scalar folds T to C and compiles
  only the N/C builds (conj is the identity), so a real TT runs the CC build of the 64×64 tile.
* Masks: 64×64 all nine; the panel tiles 128×32 and 32×128 their issued form, its mirror and the matching `Trans`
  forms (NC, NT, TN, CN); 32×32 and 16×16 every form but TT, TC and CC.
* Rows are only appended (`wide_variant` reads a table indexed by row and form, `static_assert`ed to one row per
  `wide_configs` row), and a form a row does not list is a `static_assert` in `launch_wide_cfg`, so can_run, the
  launcher and the trace name cannot drift apart.
* The grid gains complex TC, CC, TT squares and real TT panels, so a table can rank the new forms.

**Build cost**, `libbatchlas_sycl` device link, GPU-only tree (`BATCHLAS_CPU_TARGET=none`, sm_89), link alone (the
`.so` removed and relinked, objects current), two runs each: before 36.4-36.9 s; all nine forms on 32×32 and 16×16
too, 49.9-50.2 s (+37%); TT, TC and CC kept off the two small tiles, 47.0 s (+28%). The plan's ceiling was +30%.

**Registers** (`scripts/register_probe.sh`, ptxas only, no GPU needed): zero spill on all 604 entries of
`libbatchlas_sycl`, and every `GemmWideTransposedKernel` passes the sm_89 four-partition gate. The new forms sit
where the old CN/NC instances do: cfloat 64-72, cdouble 124-136 (32×32 CT the highest), real CC 53 (float) and 68
(double) on 64×64.

**Pending:** the per-cell A/B (cfloat TN 768³, NT 512³, CT 512³ strided, NT 256²×64; float TT 512³ against
`reg:m=64:n=64:k=16:u=1`) and a coarse re-race. The 16-way
shared-store bank conflict of a transposed A operand (the CN tile today) applies to every new T/C A form; that is
WP-D's work, not this change's.

## GEMM: skinny 64x32 and 32x64 wide tiles

**Unmeasured: CUDA was down at implementation time** (2026-10-19). No timing below; the A/B and the re-race are
pending.

Before this change only `wide:m=64:n=64:k=16` served complex CN and NC at a skinny output (min(m, n) ≤ 64),
with half or more of its 256 lanes computing padding: cfloat CN strided 256×32×128 b128 lost 2.19x to the vendor
and NC 32×256×256 b128 2.31x on the 2026-10-18 coarse run. NN 64×32×256 b128 launches 128 work-groups of 256
lanes, half of them padding; the 64×32 tile launches the same 128 with none idle (only sliced K, below, raises
the count). Two rows are appended to `wide_configs`, `{64, 32, 16, 4, 4}` and `{32, 64, 16, 4, 4}` (128 lanes, a
4×4 thread tile), as configs of the existing `launch_wide_transposed` kernel; no kernel code changed. They take
the small-tile mask (`kWideSmallTileForms`: every form but TT, TC and CC), the rule WP-A settled on for build
cost.

* Spellings `wide:m=64:n=32:k=16` and `wide:m=32:n=64:k=16`; candidates float 21, double 11, complex 10. Both are
  appended after `wide:m=16:n=16:k=16`, so no existing tie-break moves.
* **Build cost**, measured as above (link alone, two runs each): 53.8-54.4 s before, 62.6 s after (+16%).
* **Registers** (`scripts/register_probe.sh`): zero spill on all 702 entries of `libbatchlas_sycl`; every new
  instance passes the four-partition gate at 128 lanes. float 48, double 71-76, cfloat 72, cdouble 124-136.
* **Not done: sliced K** inside a work-group (a 64×32 output on 256 lanes as two k-slices reduced through shared
  memory). It needs a new field or family and is worth building only if the A/B shows the 128-lane tiles still
  short of the vendor at the skinny shapes; bracket it by work-group count, not batch.

**Pending:** the A/B (pass ≤ 1.10x vendor) on cfloat NC strided 32×256×256 b128, CN strided 256×32×128 b128, CN
packed 32³ b2048 and NN 64×32×256 b128.

## GEMM: the 128x128 kernel in every real transpose form

**Unmeasured: CUDA was down at implementation time** (2026-10-19). No timing below; the A/B and the re-race are
pending.

Before this change `reg:m=128:n=128:k=8:u=1` instantiated NN only, so float NT, TN and TT fell to the 4×4 register
tiles or `wide`, 19-27 TFLOP/s at 1024³ b128 against NN's 57. `Gemm128x128Body` now takes `OpA, OpB`. The shared
layout (`sa` [k][m], `sb` [k][n]), the lane swizzle, the compute loop and both epilogues are unchanged; only the
loaders differ, and both already existed:

| operand | contiguous in | loader |
|---|---|---|
| A (N), B^T | m or n | a warp reads 128 contiguous elements of one k row, one 16-byte shared store |
| B (N), A^T | k | two lanes per column read 4 k, four transposing scalar stores (2-way bank conflict), `L2::128B` |

So NT uses two m/n loaders, TN two k loaders, TT a k loader for A and an m/n loader for B. The predicated leg
uses the same lane maps with per-element zero fill, so an odd `ld` stays coalesced. `can_use_128x128_fast_path`
reads the effective m, n, k (the alignment terms are the same: every packet starts at a multiple of 4 elements
along its contiguous direction). Trace names gain `_nt`, `_tn`, `_tt`, each with an `_aligned` twin.

**Registers** (`scripts/register_probe.sh`, ptxas only): all twelve legs (4 forms × predicated, aligned, aligned
staged) at 126-128 registers, zero spill, library-wide zero spill. Two source details are load-bearing at the
128 cap, both on the B^T loader only (NN's code is untouched): its pointer advances by `TileK * ldb` (64-bit)
once per slab instead of indexing `k0 * ldb` (an `int` product overflows past \f$2^{31}\f$ elements in one item;
a 64-bit product spilled the staged NT leg, 20 B), and its predicated bound is one live `int` (four hoisted row
predicates beside A's four spilled the predicated NT leg, 20 B).

**Build cost**, `libbatchlas_sycl` link alone, GPU-only tree, two runs each: NN only 46.5-46.7 s, all four forms
50.9-51.3 s (+9%).

**Rejected: the plan's optional `reg:m=128:n=64:k=8:u=1` row.** As a config row it lands on the generic
`register_tiled_common.hh` kernel, not this body; at an 8×8 thread tile ptxas gives it a 288-byte stack frame
(48 registers, accumulators in local memory) in all four forms. A 128×64 variant of this body needs new loader
code (128 threads), so it was left out.

## GEMM: WP-G transpose-aware scalar staging

**Unmeasured: CUDA was down at implementation time** (2026-10-19). Design rationale only; the A/B (float TN
strided 128x32x1024 b2048 against `reg:m=128:n=32`, `reg:m=32:n=128`, `wide:m=64:n=64:k=16` and the vendor; float TT
strided 256³ b128) is pending.

`register_tiled_common.hh`, the generic register tiles: 2×2 (32x32x8), 4×4, and 8×4 (both 128x64x32 rows), every
`reg_configs` row except 128×128. Two changes:

- **Scalar staging lane map.** With an odd `ld` or base the packet loaders are skipped and every operand is
  staged one element per lane. The lane-fastest index used to be m for A and k for B whatever the form, so a
  transposed operand had a lane stride of `ld` (one sector per lane). Now the fastest index is the operand's
  contiguous one: k for Aᵀ, n for Bᵀ, as `register_wide_transposed.hh` does. The shared layout is unchanged, so the
  compute loop and epilogue are untouched. Costs moved to shared stores: Aᵀ at TileK 16 is 2-way (TileK 32: none);
  Bᵀ is conflict-free (odd stride).
- **B fragment swizzle, not a stride pad.** The fragment read `sb[(4c + j) * s + t]` puts lanes with local
  column c and c + 8 on one bank for any stride s, because 4·s·8 ≡ 0 (mod 32): 2-way on the 64-column tiles
  (16 local columns per warp), 4-way on 32×128 (32). Padding the stride cannot remove it and would change
  `slm_bytes` (a `choice.hh` edit, which stales the lattice). `b_slot(col, row)` XORs the k row with the column's
  32-column group (`(col >> 5) & 3`). That stays inside the row's aligned 4-block, so it needs no memory, and
  every bank is distinct at every t for TileN ≤ 128. One extra LOP per fragment row; the reads were scalar
  already (the odd stride rules out vector loads), so no vectorised read is lost. The 32-column tiles (`TileN <= 32`)
  skip the XOR at compile time (ptxas could not see `q == 0`: 32x32x8 went 40 to 55 registers with it).

**Registers** (`scripts/register_probe.sh`, ptxas only, against the parent): every `GemmRegisterTiledKernel`
instantiation within -8..+3 registers of the parent (e.g. 64x64x16 NN 48 to 51, 32x128x16 TN 56 to 59), except
the aligned 128x64x32 8×4 legs, 64 to 72 (u2) and 68 to 72 (u4): 256 threads, still 3 blocks/SM by registers
against 2 by their 49,920 B of local memory, so not binding. Zero spill library-wide; the 128x64x32 rows keep the
128-byte stack frame they already had.

Test: `GemmCandidates.RegScalarStagingEveryFormAtSaturatingBatch` (strided: the scalar helpers; padded: the packet
stores; packed 128x128x64 NN: the aligned leg of 128x32x32 and both 128x64x32 rows; each at batch 1024, plus a
β = 0, NaN-C pass). A deliberate break of the transposed-B scalar split should turn NT and TT red on 64x64x16,
128x32x16, 128x32x32 and 128x64x16, and TT on 32x128x16.

## GEMM: WP-B coalesced C epilogue on the predicated legs

**Unmeasured: CUDA was down at implementation time** (2026-10-19). Design rationale only; the A/B (float NT/TN/NN
strided 256x256x16 b128, TN strided 256x512x8 b128, NN packed 256x256x16 b128 as the regression guard, cfloat NN
packed 128x256x8 and CN strided 1024x32x8 b128) and the C read-modify-write rate check are pending.

At small k a cell is one read-modify-write pass over C, and the predicated and transposed tiles stored C one
element per lane with lanes 16 bytes (four rows) apart, or `ldc` apart in the generic register tiles. Only the
aligned NN legs stored whole 16-byte bands, because the C alignment test was tied to A's and B's. Four kernels
change, and only their epilogues:

- **The band store is a per-group decision about C alone** in the wide kernels: the tile lies inside C, `ldc`
  is a multiple of the 16-byte granule, and C's item base is 16-byte aligned. The test is uniform over the
  work-group, so a predicated leg takes the band store on its interior groups. It is an epilogue choice inside the
  kernel, never a routing term or a `can_run` term.
- **Every other group stages C through the mainloop's local memory**, then stores it with lane = consecutive row,
  predicated on row and column. One warp store is then one contiguous run of a column at any `ldc`.
  - 128x128: the staged epilogue now runs on the predicated leg too (k ≤ `kStagedEpilogueMaxK`, both legs). Its
    stage layout and the barrier before it (the odd-slab tail reads `sa`) are unchanged. The predicated leg
    stores scalars at rows `lane + 32 i`, one load-then-store per element. The aligned leg's code is untouched,
    and so is the direct epilogue at k > 64.
  - 64x64 NN wide and `register_wide_transposed.hh`: `wide_scalar::staged_c_store`. `sa` then `sb` form one
    column-major stage of stride TileM, so a chunk is (TileK·(TileM+TileN))/TileM columns: 2 chunks for
    64x64, 128x32 and 32x128, 1 for 32x32 and 16x16. A 16-byte scalar (complex<double>) skips staging, because
    its one-element store is already the band store.
  - Generic `reg` (`register_tiled_common.hh`), both legs: the same chunked stage inline (a reg header cannot
    include a wide one, or the family hashes would miss the dependency). The column stride is TileM + 1:
    lanes run along n there, and an even stride would put the TC-strided writes on one bank. No band store.
- **Barriers.** Every mainloop already ends on a barrier after its last read of the stage buffers (the 128x128
  odd-slab tail is the exception, and it already had one), so no barrier is added before the first chunk. Each
  chunk adds one barrier between the stage writes and the read-back. Every chunk but the last adds one more
  between the read-back and the next chunk's writes. Prior C is read two elements per thread ahead of their
  stores, and never at β = 0.
- **Budgets.** No local memory is added and `slm_bytes` is unchanged.

**Registers** (`scripts/register_probe.sh`, ptxas only): zero spill across all 315 entry functions of
`libbatchlas_sycl`. The 128x128 legs are at 126-128 registers, the new predicated staged legs at 128. Getting
there took four register-cap concessions, each found by a probe:

- A per-group 16-byte branch in 128x128 (interior groups, or the direct epilogue at k > 64) spilled 20-36 B,
  so 128x128 has no band store off the aligned leg.
- Prefetching all 16 prior values before the pass barrier also spilled.
- NN's predicated A loader now bounds its four rows with one live int, as B^T already did. With its four
  hoisted predicates, the staged NN leg spilled 28-40 B even with no read-back at all. NT keeps the four,
  because two ints spilled NT 48 B.
- In the wide and generic kernels, the chunk loop is not unrolled and the read-back batch is 2. A batch of 4
  with every chunk unrolled cost +8 to +24 registers.

Against the WP-G probe, the ones that cost occupancy (256 threads):

| kernel | registers | blocks/SM (registers) |
|---|---|---|
| wide double 128x32, 32x128 | 60-64 → 71-76 | 4 → 3 |
| wide double 64x64 NC | 64 → 72 | 4 → 3 |
| wide double 32x32 (64 threads) | 60 → 76 | 16 → 12 |
| wide double 64x64 NN predicated | 72 → 78 | 3 → 3 |
| wide float 64x64 NC | 45 → 55 | 5 → 4 |
| reg 64x64x16 NT, reg 64x64x8 NN | 48 → 55 | 5 → 4 |
| reg 128x64x16 TN | 48 → 56 | 5 → 4 |

cfloat wide is unchanged at 72. Every other float wide or reg instantiation keeps its block count or gains one.
The baseline is the WP-G probe log, taken before WP-G's two commits. Its kernel code is assumed to match the parent
of this change; that was not re-checked.

**Build cost**, `libbatchlas_sycl` link alone, GPU-only tree, two runs each: parent 50.5-50.7 s, WP-B 54.2-54.4 s
(+7%). The four predicated staged 128x128 legs are the new instantiations.

Tests:

- `GemmCandidates.CoalescedCEpilogueEveryTileAtSaturatingBatch` covers every reg and wide candidate on 256x256x16
  strided at batch 1024 (complex<double> skips it), then 200x136x8 at ld = m + 1 and padded, then β = 0 with NaN
  in C.
- `Reg128x128EveryFormEveryLegAtSaturatingBatch` gains a strided 200x150x72 leg. Its strided 200x150x37 leg now
  runs the predicated staged epilogue, so the new leg keeps the predicated direct one covered.

Deliberate breaks, unrun:

- Dropping the barrier after the stage writes should turn the staged groups red: every reg row, and every wide
  row except complex<double>. In 128x128 (the barrier after the pass's stage writes) it should turn the strided
  and ld = m + 1 cases red, while the packed aligned legs stay green.
- Dropping the barrier between chunks should turn red only the multi-chunk tiles (wide 64x64, 128x32 and 32x128;
  reg 32x32x8, 64x64x8, 64x64x16, 128x64x16 and 32x128x16), as a bit-identity failure.
- Dropping the `n0 + TileN <= n` term of the wide band-store test should turn only the padded wide cases red, by
  writing outside C.

## GEMM: wide v2, one double-buffered 64x64 kernel for every form

**Unmeasured: CUDA was down at implementation time** (2026-10-19). Design rationale, build cost and registers
only; the A/B is pending.

`wide:m=64:n=64:k=16` for float and complex<float> now runs `GemmWide64Body` (`register_64x64_k16_wide.hh`), one
functor templated over (op(A), op(B)) ∈ {N, T, C}² and the global-load leg. It replaces both the NN kernel and
`launch_wide_transposed` at 64×64 for those two scalars. double and complex<double> keep the old kernels:
complex<double> is already at 124-136 registers without prefetch, and the plan's link budget was +25%. The spelling
is unchanged; only its timing moves, so the retune absorbs it. What changed, against the root causes in the
2026-10-18 loss map (cfloat CN/NC m,n > 64 at 1.19x, NN at 1.13x):

- **Staging is a pure copy.** Conjugation is folded into compile-time FMA signs (`wide_v2::fma_signed`, still 4
  FMAs, bit-identical to conjugating at staging), so a C operand moves in 16-byte granules like an N one.
- **One shared layout, `s[kk][i]` in 16-byte granules, for both operands in every form.**
  - An operand contiguous along i (A, Bᵀ) is one granule per load and store.
  - An operand contiguous along k (Aᵀ, B, including NN's B) loads 16 bytes down k.
    - complex<float> loads two columns (a 2×2 block) and swaps them into one granule per row.
    - float stores four scalars, one per row.
  - The granule index is XORed with `swz(kk) = ((kk / VecN) & 3) << 1`, the same on store and fragment read.
    The lane maps (`GemmWide64Body::first_k`, `first_i`) put 4 row groups × 2 adjacent granules in each
    bank-conflict set: an 8-lane phase of a 16-byte store, or a warp of 4-byte ones. By that construction the
    transposing store is conflict-free; the old transposed store was 16-way, and NN's B store 4-way.
  - Four swizzle values, not eight, so the fragment read needs four offsets. The eight-value form
    (`(kk >> 1) & 7`) left complex<float> TN and CN 24 B of spill at the 128 cap.
- **Two buffers, one barrier per slab, the next slab prefetched into `ext_vector` registers**: the 128×128
  kernel's schedule, unrolled by two so the buffer index is constant. An odd slab count ends with a read of buffer
  0 and no barrier after it, so the staged epilogue adds one.
- **The leg.** The global-load leg (16-byte moves vs element loads) is per launch, from A's and B's bases, ld and
  batch stride (`wide64_v2_vec_loads`); the parity of k, m and n does not enter it. Predication is per work-group
  and per slab: an operand is predicated only in an edge group (in its own extent) or the k tail. So one ragged
  edge no longer sends every interior group down the predicated path.
- **Epilogue**: the WP-B rule unchanged, the band store per group and otherwise `staged_c_store`. Both A buffers
  are the stage's lo half, so a chunk is the whole 64-column tile (one pass).
- **Launch bound** on the functor's `operator()`: 3 groups per CU for float, 2 for complex<float>.
  - The bound is ignored on a lambda.
  - `operator()` is `always_inline`. Out of line, complex<float>'s body spilled 64-104 B behind the call, while the
    entry function reported zero, so only the non-inlined function line of the probe showed it.

**Registers** (`scripts/register_probe.sh`, ptxas only): zero spill on all 724 entry functions of
`libbatchlas_sycl`.

- float: 79-80 registers, 3 groups per SM.
- complex<float>: 126-128 registers at the 128 cap, 2 groups per SM (16 warps; the old NN kernel ran 80 registers
  at 3).
- A 3-group bound for complex<float> caps it at 80 and spills 76-240 B in all 18 instances, so 2 is the bound.
- Whether 16 warps with prefetch beat 24 without is the open question the A/B answers.

**Build cost**, `libbatchlas_sycl` link alone (`.so` removed, objects current), GPU-only tree, two runs each:
parent 63.0-63.1 s, v2 74.6-74.9 s (+18%). That is under the plan's 25% ceiling at which v2 would have been
restricted further.

Tests: `GemmCandidates.WideV2EveryFormAtSaturatingBatch` runs all 9 forms at batch 1024 (7 distinct problems,
each item bit-identical to its representative) on four shapes:

- 100×70×37 at ld = m + 1: element loads, edge groups, a k tail, 3 slabs.
- 200×136×40 padded: the 16-byte leg with interior and edge groups.
- 128×128×64 packed: all interior, 4 slabs, the band store.
- 130×70×64 at ld = m + 1: an even slab count into the staged epilogue.

Then β = 0 with NaN in C per form, and every leg layout (`wide_v2_leg_layouts`) at 160×136×48 batch 4: four
interior groups over three full slabs, the only place the 16-byte load runs. The layouts take each refusing term
alone on A and on B (odd ld, item stride = 1 mod 4, odd base), plus Ld2 (ld = 2 mod 4), which complex<float>
takes on the 16-byte leg (odd columns 16- but not 32-byte aligned) and float refuses. `PinnedChoiceLaunchesItsOwnKernel`
expects a `_aligned` trace name in every form on packed data, and the same leg layouts at 160×136×48 (ragged
in m and n) with the leg each must take, so a tile-multiple term copied into the leg predicate goes red.

**Local memory.** complex<float> v2 asks for 32 KiB (two buffers per operand), over an Apple GPU's 32 KiB limit
once the runtime's 8 B per item is added. can_run's wide arm checks `wide_slm_bytes<T>` (choice.hh, static_asserted
against the kernel) against `slm_budget_for(threads)`, so Metal refuses it there instead of failing at launch.

Deliberate breaks, unrun:

- Dropping the A conjugation sign should turn red only CN, CT and CC (complex<float>); the B sign only NC, TC
  and CC.
- Dropping the swizzle on the read side only should turn every v2 form red for both scalars, and nothing else.
- A k-contiguous lane map that puts two lanes on one granule (complex<float>: drop the `(tid >> 1) & 4` term of
  `first_k`, so rows 8-15 are never staged) should turn red every complex<float> form except NT and NC, the forms
  with no k-contiguous operand. Swapping two lane-map bits is not such a break: `first_k`/`first_i` feed both the
  global address and the shared store, so a bijective remap stays right and changes only bank conflicts, which no
  test observes (the conflict-free claim waits for an `l1tex` bank-conflict count once CUDA is back).
- Dropping a term of `wide64_v2_vec_loads` (A's or B's base, ld or batch stride, or the whole B conjunct) should
  turn red the matching `wide_v2_leg_layouts` rows in the trace test, and fault or go wrong in the numeric one.
- `<=` for `<` in the k-tail predicate should turn red the three ragged-k shapes, and leave 128×128×64 and
  130×70×64 green.
- Dropping the per-slab barrier should show as bit-identity failures across items.

**Pending:** the plan's A/B (pass: every cell ≤ 1.05x vendor, and NN packed 512³ not slower than 2.78 ms):

- cfloat CN strided 1024³ b128;
- NC strided 256×128×96 b128, also against `wide:m=128:n=32:k=16`;
- CN packed 256×128×96 b128;
- NN strided 362×90×96 b128;
- TN packed 768³ b128.

Then a coarse re-race. Not done:

- the C prefetch before the last slab (the plan's P2.5);
- an 8-byte vector load for complex<float> on the element leg (it needs an 8-byte-aligned base, a third leg).

## GEMM: WP-F, a k8 slab for the 64x64 wide tile

**Unmeasured: CUDA was down at implementation time** (2026-10-19). Design rationale only; the A/B is pending.

Every wide tile had a 16-deep slab, so at k ≤ 8 half of each slab was zero-filled and its FMAs wasted. The
2026-10-18 loss map left 1778 cfloat cells at k ≤ 16 after WP-B (635 losing, gmL 1.21). One row is appended to
`wide_configs`, `{64, 64, 8, 4, 4, kWideAll}`, spelled `wide:m=64:n=64:k=8`. It comes after every existing row, so
no tie-break moves. Candidates: float 22, double 12, complex 11 (12 since WP-H).

- **float and complex<float>: wide v2 at `TileK = 8`.** `GemmWide64Body` takes the slab depth as a template
  parameter; the kernel class gains a defaulted fifth argument, so the `device_calls_tests` pattern still matches.
- **Each lane still stages CF 16-byte granules**, so at k = 8 one operand's slab needs only 128 lanes. Warps 0-3
  stage A and warps 4-7 stage B. The branch is warp-uniform, a lane holds one prefetch set instead of two, and
  `KStep` (the rows one staging pass covers along i) is `128 / GPR`.
- **The k-contiguous lane map at k = 8** has 4 row groups (complex<float>) or 2 (float) in the low tid bits and
  the column above them.
  - For complex<float>, an 8-lane phase is still 4 row groups × 2 column pairs, so `swz` is unchanged.
  - For float, a warp holds 2 row groups × 4 granules, so the swizzle flips granule bit 2 (`(kk / 4 & 1) << 2`)
    to keep the transposing scalar store on 32 distinct banks.
  - Correctness does not depend on the swizzle: store and fragment read apply the same bijection.
- **Epilogue unchanged.** The stage (both A buffers + both B buffers) is now 32 columns, so `staged_c_store`
  takes two passes instead of one.
- **double and complex<double> run `launch_wide_transposed` with `K = 8` in every form, NN included.** The NN
  64x64x16 kernel keeps `K = 16`, so the NN aligned-leg trace name exists only at k = 16.
- **Local memory** halves: float 8 KiB, complex<float> 16 KiB, double 8 KiB, complex<double> 16 KiB
  (`wide_slm_bytes`, static_asserted against the kernel).

The launch bound stays 3 groups per CU for float and 2 for complex<float>.

**Registers** (`scripts/register_probe.sh`, ptxas only): zero spill on all 802 entry functions of
`libbatchlas_sycl`. Every new instance passes the four-partition gate at 256 lanes:

| instance | k = 8 | k = 16 |
|---|---|---|
| float v2 | 78-80 | 79-80 |
| complex<float> v2 | 121-126 | 126-128 |
| double transposed | 66-70 | (NN kernel and transposed) |
| complex<double> transposed | 128-132 | (NN kernel and transposed) |

complex<float> at k = 8 stays above the 85-register cap of 3 groups, so its bound stays 2.

**Build cost**, `libbatchlas_sycl` link alone (`.so` removed, objects current), GPU-only tree: 87.0 s and 87.8 s.
WP-D recorded 74.6-74.9 s by the same method, so the link grows about 17%. The two figures were not taken
back to back.

Tests: `GemmCandidates.WideK8SlabStraddlesTheSlabDepthAtSaturatingBatch` runs 64x64x8 and 64x64x16 in all 9 forms:

- k = 8, 9, 16 and 17 on 100×70 at ld = m + 1 (batch 1024, 7 distinct problems, bit-identical items);
- the same k on 200×136 padded (the 16-byte leg, interior and edge groups);
- packed 128×128×24 (three full k8 slabs, the band store);
- β = 0 with NaN in C.

The oracle row, the literal local-memory bytes and the spelling are added. The epilogue sweep, the trace test (every
leg layout) and every pinned-candidate test pick the row up from `candidates<T>()`.

Deliberate breaks, unrun:

- Every lane staging A (`stage_a = true`) leaves B unstaged. That turns every float and complex<float> form red at
  64x64x8 and nothing at 64x64x16.
- The k = 16 k-contiguous map at k = 8 puts lanes outside the slab. That turns red every v2 form except NT and NC.
- The k = 16 `KStep` for complex<float> writes row kb + 8 past the slab, into the other buffer. That turns red every
  complex<float> form except TN and CN.
- A float swizzle that store and read disagree on turns every float form red at k = 8.
- `<=` in the k-tail predicate turns red the k = 9 and 17 shapes.
- Routing the k8 row to the k16 kernel gives the same bits, so only `PinnedChoiceLaunchesItsOwnKernel`'s
  `_k8_` trace names turn red.

**Pending:** the A/B (pass ≤ 1.05x vendor). cfloat NN packed 128×256×8 b128, arms `wide:m=64:n=64:k=8`,
`wide:m=64:n=64:k=16`, `vendor`; cfloat CN strided 512×64×8 b128. Large-m,n k ≤ 16 cells are DRAM-bound at parity
on both sides and are not the target.

## GEMM: complex small and the staged transposed-A row

**Unmeasured: CUDA was down at implementation time** (2026-10-19, WP-H). Design rationale only; the A/B is pending.

`small` now compiles for complex<float> and complex<double>, and `candidates<complex>()` lists it third, as the
real lists do (complex 12). The tables do not rank it yet, so Auto reaches it only where nothing ranked can run:
a vendor-free complex call past the grid-z ceiling.

- **Arithmetic is the POD `Cx`** (`device_scalar.hh`). Operands and α, β are re-typed at the launcher.
  - The inner product is `fma_acc`, 4 FMAs and no `__mulsc3`.
  - The epilogue is `dev_mul(α, acc)`, then `fma_acc(out, β, C)`, which is skipped at β = 0.
  - The real path keeps its expression.
- **Conjugation happens at staging.** `ConjTrans` B is conjugated as it enters local memory, and `ConjTrans` A
  as it enters the transposed-A stage below. For a real scalar, `ca` and `cb` are constant false.
- **Ceiling.** complex<double> stops at 32 (`small_max_dim`, mirrored by `kSmallMaxDimFor`).
  - Its 64 bucket would need 64 × 65 × 16 B = 66.5 KB of local memory.
  - It would also hold 64 row elements plus 32 accumulators, 384 registers.
  - At 32 the bucket is 2 slots × 32 × 33 × 16 B = 33.8 KB, with a 32-entry row and 16 accumulators.
  - complex<float> 64 is 33.3 KB with the same register count as double 64.
- **Local memory is a `can_run` term** (`small_slm_bytes` against `slm_budget_for`), static_asserted in
  `gemm_kernels.cc` against the kernel's element counts. It refuses, for example, complex<float> and double at
  64 on the M5 Max's selector budget.
- **Transposed A goes through local memory.** A transposed A is stored k × m, so a lane's row of op(A) is a
  stored column, `lda` away from its neighbours' rows: every load was a separate sector.
  - The slot's `sB` region first takes A in storage order, so the loads are coalesced. The order is
    `sB[r + l·(NB+1)]`, which is conflict-free for 4-, 8- and 16-byte scalars, because NB + 1 is odd.
  - After a barrier, each lane reads its row.
  - A second barrier keeps the row in place until every lane of the slot has read it, and only then is op(B)
    staged over it.
  - The two barriers cost no extra local memory and run only on a transposed A. `ta` is uniform per launch.
  - This also changes the real T forms (the float TT/TN target).
- **complex<float> 64 writes its inner product as four sequential 16-l loops.**
  - As one 64-l loop it is past LLVM's pragma unroll threshold. The unroll was declined and a[] went to a
    512 B stack frame: 96 registers, and the stack-frame column was the only sign.
  - A lambda per chunk fixed that but put 160-184 B of stack on other buckets, double 64 and float 64
    included. The chunks are therefore a file-local macro.
  - Every other bucket keeps its single loop.

**Registers** (`scripts/register_probe.sh`, ptxas only): zero spill and zero stack on every small kernel. The
four-partition gate at 128 lanes allows up to 512 registers.

| NB | float | double | complex<float> | complex<double> |
|---|---|---|---|---|
| 8 | 39 | 40 | 40 | 64 |
| 16 | 40 | 64 | 56 | 122 |
| 32 | 64 | 125 | 127 | 226 |
| 64 | 128 | 253 | 255 | (refused) |

**Build cost:** the `libbatchlas_sycl` link replayed by the probe took 87.7 s. WP-F recorded 87.0-87.8 s, so the
16 new kernels are not visible in the link time. The two figures were not taken back to back.

Tests:

- `GemmCandidates.SmallEveryBucketEveryFormAtSaturatingBatch` covers NB 8/16/32/64 (ragged, k = NB on the
  first) in all 9 forms:
  - strided and aligned-odd-ld layouts;
  - batch 1024 repeating 7 problems, items bit-identical;
  - complex α and β ≠ 0;
  - β = 0 with NaN in C;
  - complex<double> above 32 refused, leaving C untouched.
- `GemmCanRun.SmallCeilingAndLocalMemoryPerScalar` checks the literal bytes and a budget one byte short on each
  side.
- The oracle (`small_runs`) admits complex, so every pinned-candidate test picks the family up.
- `gemm_tests` `SmallBatched*` run complex with ConjTrans.

Deliberate breaks, unrun:

- Dropping the A conj turns red only the CN/CT/CC forms of complex.
- Dropping the B conj turns red only NC/TC/CC.
- Swapping the stage index (`sB[l + r·(NB+1)]`) turns red every T/C-A form of every scalar.
- Removing the second barrier is a race between the row read and the op(B) stage. It shows on T/C-A forms
  as item differences at batch 1024.
- `α.re · acc` in the epilogue turns every complex case red. It is not caught by β = 0 alone, because α is
  complex in every case.

**Pending:** the A/B. cfloat CN packed 32³ b2048, arms `small`, `wide:m=16:n=16:k=16`, `vendor`; float TT
strided 64³ b128 against the parent commit.

## GEMM: WP-I, an unpredicated interior loop on the 128x128 predicated leg

**Unmeasured: CUDA was down at implementation time** (2026-10-19). Design rationale, registers and build cost
only.

Target: float NN strided, k ≥ 48, at 256-384 (384³ b128 at 30.8 against cuBLAS's 37.7 TFLOP/s in the
2026-10-18 loss map). An odd `ld` sends every group of such a call down the predicated leg, which bounds
every element of every slab. Now a work-group whose 128×128 tile lies inside C (`m0 + 128 <= m`,
`n0 + 128 <= n`, uniform over the group) runs a second copy of the mainloop. That copy has no row or column
predicates, and only the k-tail slab is bounded. Loads stay scalar, with the same lane maps. Edge groups
run the old loop unchanged, and the aligned leg is untouched: its register counts match the parent in all
eight instances.

Three source details are load-bearing at the 128-register cap (`scripts/register_probe.sh`, ptxas only):

| variant | result |
|---|---|
| per-slab branch inside the one loop, per operand (`kfull && m0 + 128 <= m`) | NT direct 8 B, NN staged 20 B, NT staged 16 B spill |
| the same, one branch for both operands | NN 4 B, NT 20-24 B |
| the same, interior flag hoisted out of the loop | NT direct 20 B, NN staged 4 B |
| two loop copies, `Body::operator()` inlined by heuristic | the two copies push `operator()` out of line: 32 B spill behind the call, 160-200 B stack frame in the entry |
| two copies, `always_inline` on `Body::operator()` | zero spill, but the aligned leg's registers move (124-128, against the parent's 127-128) |
| two copies, the predicated leg launched through `Gemm128x128Inlined` (call-site `[[clang::always_inline]]`) | zero spill except 4 B in the never-launched `_with_offset` twin of staged NT |
| interior copy's k tail as a `kfull ||` term, or unswitched by hand | 8-48 B |
| **shipped**: the wrapper, the test written edge-first (`m0 + 128 > m` or the n term) | zero spill on all 816 entry functions, no stack frame; predicated 127-128 registers, aligned identical to the parent |

ptxas is chaotic at this cap: a logically equivalent rewrite moves spill between instances, so any edit
here needs a fresh probe. Build cost: the probe's `libbatchlas_sycl` link, one run each, 88.6 s (parent)
and 88.8 s.

Test: `GemmCandidates.Reg128x128PredicatedInteriorAndEdgeGroupsAtSaturatingBatch` covers 5 forms × 5 legs
at batch 1024, each item bit-identical to its representative. Every leg has odd operand bases, which is
what keeps it on the predicated leg (several of the lds are multiples of 4):

- 256×256×72 strided: all four groups interior, nine full slabs, the direct epilogue. 256 has the
  `m0 + 128 == m` boundary on the interior path; 384 added only a middle group at 2.25× the host work.
- 256×256×37: the interior copy's k tail after full slabs, the staged epilogue.
- 256×256×5: one partial slab, so the prologue's load is the only one bounded and the loop body never runs.
- 300×200×45 at ld = m + 1, and 300×200×72 strided: interior, m-edge, n-edge and corner groups in one
  launch.

The k-tail legs (37, 5, 45) fill every gap with NaN instead of the finite poison. With a finite poison,
dropping only one operand's k bound multiplies the poison by the other operand's correctly zeroed value,
a bit-exact -0, and the result stays green; NaN × 0 is NaN.

Deliberate breaks, unrun:

- A wrong element index in the interior copy should turn the 256 legs red in that form.
- Treating every interior slab as full, or dropping either operand's k bound alone, should turn red the
  k = 37, 5 and 45 legs, and leave the k = 72 legs green. A break confined to the prologue (`kfull`
  forced true at `k0 == 0`) should turn red only k = 5.
- An interior test that admits an edge group is invisible to every numeric test. It only over-reads rows
  and columns past m and n, which feed accumulators the epilogue discards. Such a bug faults only when
  the over-read leaves the allocation.

**Pending:** the A/B (pass ≤ 1.05× vendor) on float NN strided 384³ b128 and 256×256×64 b128.

## GEMM: WP-K, pipelined staging for the transposed wide tiles

*Unmeasured* (2026-10-19, sm_120 build only; no GPU was free). Design, budgets and test coverage.

Target: the skinny and panel tiles `launch_wide_transposed` serves, 128x32, 32x128, 64x32, 32x64,
32x32 and 16x16 in every scalar. The 2026-10-18 sm_120 loss map: cdouble with one dim ≤ 64 at median
1.17 (504 ms), cfloat skinny 132 ms, float with one dim ≤ 64 161 ms. The old mainloop did load,
barrier, compute, barrier with element loads, so at low tile counts no other group hid the latency.

| piece | rule |
|---|---|
| stages | `og::wide_stages<T>`: 2 when 2·k·(m+n)·sizeof(T) ≤ 40 KiB, else 1. 1 for cdouble 128x32, 32x128, 64x32 and 32x64 (80 and 48 KiB double-buffered). 64x64 double stays at 1 (cdouble 64x64 is wide v2 since WP-J). `wide_slm_bytes` reads the same function |
| barriers | two stages: one per slab, slab s+1 stored into the stage slab s−1 was read from. One stage: two per slab, as before |
| prefetch | the next slab's granules go into registers before the current FMAs: (m+n)·k / threads elements per thread (128x32: 10, 64x32: 12, 32x32: 16, 16x16: 8). Off on 64x64 double |
| granules | 16 bytes along the contiguous dimension: k for A^T and B, i for A and B^T. Along k a warp takes 32/KG columns of KG granules, i fastest across lanes |
| swizzle | the k-contiguous store XORs its granule column by (kk/VecN)·(32/TileK) mod granules-per-row, the fragment read applies the same XOR. Conflict-free except float 16x16 (2-way) |
| leg | `wide64_v2_vec_loads` per launch (bases 16-byte aligned, ld and stride whole granules). A group's slab is predicated (element loads, zero fill) when the group is an edge or the slab is the k tail |
| conj | applied in registers at the shared store |
| epilogue | unchanged. The staged path takes both stages as its space, behind one barrier after the last compute |

No new spelling, row or key: the tiles keep their `wide:m=..:n=..:k=16` spellings, so a
`partial:wide` re-race picks the change up. Metal (about 26 KiB at 256 lanes) loses double-buffered
cfloat 128x32 and 32x128 through `can_run`'s local-memory term.

Test: `GemmCandidates.WideTransposedPipelineEveryTileFormAndLegAtSaturatingBatch` runs every form of
the six tiles on six shapes and the wide v2 leg layouts:

- 2TM×2TN×48 padded (16-byte leg, three slabs) and ×64 at ld = m+1 (element leg, four slabs, staged
  epilogue), each at batch 1024 (256 for 16-byte scalars), 7 distinct problems;
- TM+1×TN+1 at k = 16, 17 and 37, NaN in the gaps, on both legs.

`GemmWideSlm.SlmBytesMatchWhatTheKernelsAllocate` holds literal bytes per tile and scalar.

Deliberate breaks, unrun:

- `store(cur)` in place of `store(cur ^ 1)`: red at every k > 16 on the two-stage tiles. Green at
  k = 16, and for cdouble 128x32, 32x128, 64x32 and 32x64.
- `soff<false, …>` for A in `compute`: red only in the forms with A transposed (TN, CN, CT).
- dropping `gk + q < k` in `gload`: red only in the k = 17 and 37 NaN cases.
- dropping the barrier before `staged_c_store`: possible item mismatches in the ld = m+1 batch cases
  only (a race, so not guaranteed).

**Pending:** registers from the probe (the table below), and the A/B on
`ta=C,tb=N,layout=strided,m=1024,n=32,k=1024,batch=128` (cdouble) and
`ta=N,tb=T,layout=strided,m=32,n=1024,k=256,batch=2048` (float, 1.73× before).

## GEMM: WP-L, global split-k over the transposed wide tiles

*unmeasured* (2026-10-19; no GPU time). Design, registers and tests only.

Target: few tiles with a large k (tiles × batch below about 2 × SMs, k ≥ 256), where the work-group count is
the batch and each group walks k serially. Pre-188 sm_120 cells: float NT 32×64×1024 b128 3.36×, cfloat NC
32×64×1024 3.51×, cdouble CN 1024×32×1024 b128 1.77-2.4×.

- **Family** `splitk:m=..:n=..:k=..:s=..`, every scalar. Hosts: the `wide_configs` rows other than 64×64
  (`splitk_host`: 128×32, 32×128, 32×32, 16×16, 64×32, 32×64, all at k = 16), so every host runs
  `launch_wide_transposed` in each form it instantiates. s ∈ {2, 4, 8}. 18 candidates per scalar: float 40,
  double, cfloat and cdouble 30.
- **Slices.** `launch_wide_transposed<T, P, Split = true>` launches batch × s groups in grid z. Group z computes
  item z / s over k ∈ [slice · kc, min(k, slice · kc + kc)), kc = ⌈k / s⌉ rounded up to TileK, and stores item z of
  the partials at α = 1, β = 0. A slice past k stores zeros. `Split = false` compiles the old kernel unchanged.
- **Partials** `[batch][s][ldw × n]`, ldw = m rounded up to 4 (`splitk_ld`), so interior tiles take the 16-byte band
  store. Leased from the Queue's arena for the call and sized with `BumpAllocator::measuring()`; no public
  `gemm_buffer_size`. Bytes: s · ldw · n · batch · sizeof(T) (cdouble 32×64, s = 8, b128: 33.6 MB). The arena
  keeps its peak until `trim_workspace()` ([the arena keeps its peak](../design/workspace.md#workspace-the-arena-keeps-its-peak)),
  so one large split-k call pins its partials in the Queue.
- **Reduction** (`splitk_reduce`, trace `gemm_splitk_reduce`): one work-item per element, m fastest, sums slices
  0..s-1 in that order (deterministic, no atomics), then C = α Σ + β C; β = 0 never reads C. An out-of-order
  Queue waits for the slices first.
- **`can_run`** (correctness only): the native terms, s ∈ kSplitKSlices, batch × s ≤ `max_groups_yz` and ≤ INT_MAX
  (the int group id), a host row that instantiates the form, the host's work-group and local-memory terms,
  ldw × n ≤ INT_MAX, and the partials within `select::Device::scratch_budget` = min(MAX_MEM_ALLOC_SIZE,
  GLOBAL_MEM_SIZE / 4); past it the lease throws `std::bad_alloc` after the choice (cdouble 32×2048, s = 8,
  batch 2048 asks 17 GB). The launcher also refuses the INT_MAX, ldw × n and budget terms. No k or tile-count term: the table decides
  the window.
- **Tuning grid** (`choice.hh`, tuner only, not in the sm_89 transcription): Gram squares m = n ∈ {16, 32, 64} ×
  k ∈ {1024, 2048, 4096}, and 32 × {64..2048} both ways at k = 2048; panel forms, strided, every grid batch.
  At batch 32768 batch × s ≥ 65536 refuses every s on CUDA, so split-k is timed at batch 128 and 2048 only, and a
  call nearer the 32768 row (log distance: batch above about 8192) never picks it. Accepted: at that batch the
  unsplit grid already fills the device.

**Registers** (`scripts/register_probe.sh`, sm_120 GPU-only tree, ptxas only): zero spill on all 544 entry
functions of `libbatchlas_sycl`; every split body passes the four-partition gate at its host's work-group.

| split body | registers | against the unsplit instance |
|---|---|---|
| float | 39-61 | -9 to +5 |
| double | 38-72 | 0 to +4 |
| complex<float> | 40-80 | -8 to +8 |
| complex<double> | 42-118 | -18 to +2 |
| `GemmSplitKReduceKernel` | 30 (single precision), 32 (double precision) | |

Tests:

- `GemmCandidates.SplitKEveryHostEveryFormAtSaturatingBatchAndSliceEdges`: every host, s and served form. Shapes
  (m + 1) × (n + 1) × 300 (136 for 16-byte scalars) at ld = m + 1, batch 1024 repeating 7 problems; m × n × 40
  padded; (m + 1) × (n - 1) × 1. Then β = 0 with NaN in C, and a second identical call compared bit for bit.
  Before each pinned call the arena bytes the lease gets are filled with 1e6, and the call must not grow the
  arena (same address), so a skipped store shows as poison rather than fresh zeros or the last call's partials.
- `GemmCandidates.SplitKOnAnOutOfOrderQueueReducesAfterTheSlices`: `splitk:m=32:n=32:k=16:s=8`, 33×33×300 (136),
  batch 1024 on an out-of-order Queue with poisoned partials, three repetitions.
- `GemmCandidates.SplitKGridCeilingIsBatchTimesSlices`: s = 2 and 8 at batch ⌊65535 / s⌋ (a launch) and one past
  (a refusal, C untouched).
- Every generic per-candidate test (pins, `CanRunEqualsLaunch`, the launcher bit for bit, the kernel trace name
  `<host trace>_splitk`, NaN C, the saturating batch, missing instantiations, the candidate set) picks the family
  up from `candidates<T>()`.

Deliberate breaks, unrun:

- Drop the last slice (global dim 0 = batch × (s − 1)): the reduction reads stale arena bytes; red in the split-k
  tests and the generic splitk cases only.
- `k_hi = k` (each slice runs to the end of k): later slices double-count; red at every k > kc (300, 136, 40 at
  s ≥ 2 where k > kc), green at k = 1.
- β applied per slice in the reduction (`acc = lin_epi(1, beta, acc, prior)` inside the loop): red only in β ≠ 0
  cases; the NaN-C β = 0 cases stay green.
- `bid = z % batch` instead of `z / s`: items mix; red in the batch-1024 bit-identity checks.
- Dropping `batch * s` from `can_run`: `SplitKGridCeilingIsBatchTimesSlices` turns red (the launch fails past
  the grid).
- An empty slice returns before its store (`if (k_lo >= k_hi) return;` after the decode): red at s = 8, k = 300
  (float, double, cfloat), s = 4 and 8 at k = 40, and s = 4 and 8 at k = 136 (16-byte scalars), all through the
  poison, and the out-of-order test (s = 8 at k = 300 / 136); green at k = 1 (slice 0 only stores) and at s = 2.
  Unpoisoned splitk calls elsewhere (generic cases, the grid ceiling at k = 20) go red only where the arena holds
  nonzero stale bytes.
- Deleting `if (!ctx.in_order()) out->wait();`: likely red in
  `SplitKOnAnOutOfOrderQueueReducesAfterTheSlices` only (a race, not certain).
- No test for the `scratch_budget` term: refusing needs partials past a quarter of device memory.

**Pending:** the A/B (pass ≤ 1.05× vendor). float `ta=N,tb=T,layout=strided,m=32,n=64,k=1024,batch=128` and cfloat
`ta=N,tb=C,…`, arms `splitk:m=32:n=64:k=16:s=4`, `splitk:m=32:n=64:k=16:s=8`, `wide:m=32:n=64:k=16`, `vendor`;
cdouble `ta=C,tb=N,layout=strided,m=1024,n=32,k=1024,batch=128`, arms `splitk:m=128:n=32:k=16:s={2,4,8}`,
`wide:m=128:n=32:k=16`, `vendor`.

## GEMM: WP-J, complex double on wide v2

*Unmeasured* (2026-10-19): no GPU was available at implementation time. Design, registers and tests only;
the A/B is pending.

`wide:m=64:n=64:k=16` and `wide:m=64:n=64:k=8` for complex<double> now run `GemmWide64Body` in all 9 forms, NN
included. Before this they ran the NN 64x64x16 kernel (NN at k = 16) and `launch_wide_transposed` (every other
form, and every form at k = 8): one buffer, element loads, two barriers per slab. The spellings are unchanged;
double keeps the old kernels.

- **CF = 4, VecN = 1, NB = 4.** One element is one 16-byte granule. The staging identity
  `GPR * TileK == CF * OpThreads` holds as it is (64 · 16 = 4 · 256; 64 · 8 = 4 · 128 for the split slab).
- **k-contiguous operand** (Aᵀ, B): lane map `kk = tid % TileK`, `i = (tid / TileK) * 4`, so 16 consecutive
  lanes read 256 contiguous bytes down one column. `sstore` writes the four columns of row kk as four
  granules.
- **Swizzle `swz = kk & 7`.** An 8-lane store phase holds 8 consecutive kk of one column; rows are 1 KiB apart,
  so without the XOR all 8 lanes hit the same 4 banks. The fragment read and the i-contiguous store stay
  conflict-free under any per-row XOR.
- **`fma_signed` and `elem`** are templated over the real type; a complex<double> fragment is the 16-byte
  granule bit-cast (`__builtin_bit_cast`), a register move. A `memcpy` through a private array is the
  pattern acpp leaves on the stack for 16-byte complex<double> (`check_acpp_jit.py` STACK_KNOWN); not yet
  checked on an acpp tree.
- **Launch bound** 1 group per CU (8 warps): FP64 runs at 1/64 rate, so the kernel is compute-bound and needs no
  more warps. 2 groups would cap registers at 128 against 134-224 used.
- **Local memory:** 64 KiB at k = 16 and 32 KiB at k = 8 (`wide_slm_bytes`, static_asserted against the
  kernel). **Capability loss:** the k = 16 tile no longer fits a budget under 64 KiB (local memory less the
  4 KiB reserve): acpp without `BATCHLAS_ACPP_SLM_OPTIN` (48 KiB) and any 64 KiB device such as Intel Arc
  (60 KiB budget), where the old 32 KiB kernels ran it. can_run refuses `wide:m=64:n=64:k=16` there and Auto
  falls through the table; `gemm_tests` checks the refusal wherever the tile is over budget
  (`WideTileFitsOrIsRefused`). k = 8 (32 KiB) still runs.

**Registers** (`scripts/register_probe.sh`, ptxas, sm_120): zero spill and no stack frame on all 457 entry
functions of `libbatchlas_sycl`. The 36 complex<double> v2 instances:

| slab | registers |
|---|---|
| k = 8 | 134-168 (NN 136-156, TN/CN 146-168) |
| k = 16 | 196-224 (TN/CN 224) |

The gate at 256 lanes: ceil(8/4) · 32 · 224 = 14336 ≤ 16384.

Tests:

- `GemmCandidates.WideV2EveryFormAtSaturatingBatch` now covers complex<double>.
- `WideK8SlabStraddlesTheSlabDepthAtSaturatingBatch` runs it at batch 1024 (it was 256 for 16-byte scalars).
- `WideV2TileAndSlabEdgesUnderNaNPoison` covers both slabs and all 9 forms. Its shapes are m, n ∈ {63..65,
  127..129} and k ∈ {7, 8, 9, 15, 16, 17, 33}, with NaN in every gap, then 129×65×33 at batch 1024 with
  bit-identical items.
- complex<double> takes the 16-byte leg on every element-offset layout: ld and stride are always whole
  granules and bases stay 16-byte aligned. Only an 8-byte base reaches the element leg:
  `WideV2ComplexDoubleElementLegAtAnEightByteBase` copies each problem one double in (both slabs, 9 forms,
  NaN gaps, beta ≠ 0, 160×136×48 padded and 100×70×37 at ld = m + 1, then batch 1024), and
  `PinnedChoiceLaunchesItsOwnKernel` checks that view launches the element-leg kernel.

Deliberate breaks, unrun:

- Dropping `+ c` in the CF = 4 `sstore` column should turn red every complex<double> form except NT and NC,
  the forms with no k-contiguous operand, and no float or complex<float> case.
- `swz` returning 0 on the fragment read only, for CF = 4, should turn red every complex<double> v2 form and
  nothing else.
- Using the complex<float> lane map (`CF == 2` branch of `first_k`/`first_i`) for CF = 4 should turn red
  complex<double> forms except NT and NC.
- Flipping the A sign in `fma_signed` should turn red CN, CT and CC for both complex scalars. The template is
  shared, so the break is not cdouble-only.

**Pending:** the A/B (pass ≤ 1.05x vendor), arms `wide:m=64:n=64:k=16`, `wide:m=64:n=64:k=8`, `vendor`:

- cdouble TN packed 512×1024×128 b128;
- NC strided 1024×32×1024 b128;
- CT packed 512³ b128;
- NN packed 512³ b128 against the parent's NN kernel.

**Stale table rows:** the cdouble rows of `tuned/gemm.sm_120.txt` and `tuned/gemm.sm_89.txt` that rank a
64x64 wide spelling first (e.g. m = n = k = 64 b2048, 128×128×8) were timed on the old NN and
`launch_wide_transposed` kernels. The wide family hash marks them stale but Auto still follows them, so those
cells now run an unmeasured kernel. Before merge: a saturation A/B (batch ≥ 128, beta = 1, strided ld) against
the parent, or a retune of the cdouble wide rows.

## GEMM: WP-N, a four-stage cp.async 128x128 aligned leg

*Unmeasured* (2026-10-19; built and probed for sm_120, no GPU run).

Target: float NN at m, n ≥ 256 and k 65-256 on `reg` 128x128 (sm_120 median 1.16× vendor, max 1.81×). The
k = 8 slab gives 8-32 barriers per tile there, and the register prefetch hides one slab of latency.

| | `reg:m=128:n=128:k=8:u=1` | `reg:m=128:n=128:k=8:u=4` |
|---|---|---|
| aligned leg | 2 local slabs, register prefetch | 4 local slabs filled by `cp.async`, 3 in flight |
| predicated leg | two-stage body | the same two-stage body (same instantiation) |
| local memory (`RegCfg::slm_bytes`) | 25,088 B | 33,280 B (two groups per SM: 65 KB) |
| trace names, aligned | `gemm_sycl_register_128x128_k8[_nt\|_tn\|_tt]_aligned` | `..._k8_s4[_nt\|_tn\|_tt]_aligned` |

On the 128x128 row `u` names the stage count (1 = the register double buffer); the field list is unchanged.
`copy_async` (`register_128x128.hh`) is `cp.async.cg.shared.global.L2::128B` (16 B) for the packet operand
(A, B^T) and `cp.async.ca ... 4` per element for a k-contiguous one (B, A^T), whose `[k][m|n]` tile transposes
the copy. Inline PTX under `__SYCL_DEVICE_ONLY__ && __NVPTX__ && __SYCL_CUDA_ARCH__ >= 800`; other targets
(AdaptiveCpp, Metal, older NVIDIA) copy synchronously through the same pipeline. Step s: `wait_group 2`,
barrier, refill buffer (s + 3) mod 4, commit (empty past the end), compute buffer s mod 4. Unrolled by 4 so
the buffer index is a constant. `wait_group 0` before the staged epilogue reuses the A tile.

Probe (`scripts/register_probe.sh`, sm_120, `libbatchlas_sycl`): all 8 u4 entries 128 registers, 0 spill,
no stack frame; the gate `ceil(8/4)·32·128 = 8192 ≤ 16384` holds at two groups per SM. The same probe shows
the two-stage body at 125-139 registers on sm_120 (identical before WP-N), over the two-group cap of 128 in
aligned NN direct 139, NN staged 135, TN staged 130, TT staged 139 and predicated NT staged 133, TN staged
129: those run one group per SM.

Risk: the k-contiguous operand issues four 4-byte copies per slab where u1 issues one 16-byte load (more
L1 wavefronts; NN's B is that operand).

Test: `GemmCandidates.Reg128x128CopyAsyncPipelineEverySlabCountAtSaturatingBatch`, 5 forms (CC folds to TT)
× slab counts 1, 2, 3, 4, 5, 8, 9, 17 at batch 1024 (7 problems, items bit-identical), packed and padded,
128 × 256 and 256 × 128; NaN C at beta = 0; a ragged and an odd-ld view on the predicated body.

Deliberate breaks, unrun:

- `copy_async_wait<Stages - 2>` → `<Stages - 1>`: slab s is read before it lands; red on the u4 aligned cases
  of that test (every k), green for u1.
- refill `(b + Stages - 1) % Stages` → `(b + 1) % Stages`: overwrites a buffer still to be read; red from
  3 slabs (k ≥ 24), k = 8 and 16 green.
- drop the B^T `gb +=` in `issue`: NT and TT red from 2 slabs; NN and TN green.
- `(b_k + i)` → `b_k` in the 4-byte copies: NN and TN red; NT and TT green.

A/B keys: `ta=N,tb=N,layout=strided,m=256,n=512,k=128,batch=2048` and
`ta=N,tb=N,layout=packed,m=1512,n=1024,k=78,batch=128` (both odd or ragged: the predicated body), and a
packed multiple, `m=n=1024,k=128,batch=128`, which the u4 aligned leg serves.

## GEMM: the register-tiled launcher table

`src/ops/gemm/kernels/register_launchers.hh`. `RegTile` holds the template parameters of `launch_register_tiled<>` as a
structural type, so one launcher `launch_reg<T, RegTile{...}>` serves every row of `reg_configs`
(`src/ops/gemm/choice.hh`, turned into a `RegTile` by `launch_reg_cfg` in `src/ops/gemm/kernels/gemm_kernels.cc`).
**Trap:** the defaults differ from `launch_register_tiled<>`. There `ThreadTileCols` defaults to `ThreadTileRows`;
here `TR` and `TC` default independently to 4, so every row states both.

## The small batched kernel

`src/ops/gemm/kernels/small_batched.hh` (`small`). Float NN at max(m,n,k) ≤ 48 previously ran `Direct`, whose adjacent
lanes read B and write C `ld` apart. The kernel gives a 128-lane group 128 / (2·NB) matrices, with NB in {8, 16,
32, 64} from max(m,n,k). op(B) is staged in local memory (odd `ld`, so either transpose is coalesced), each lane
holds one row of op(A) and NB/2 columns of C, and β = 0 skips the C read. Complex was added later on the POD
scalar (see [the complex small section](#gemm-complex-small-and-the-staged-transposed-a-row)).

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
warp stores whole 128-row columns (512 B per `STG.128`). `kStagedEpilogueMaxK = 64`. This was the aligned
leg only until WP-B extended it (see the section on the coalesced C epilogue above).

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

**In the table:** `small` ranks first in exactly 72 float rows of `tuned/gemm.sm_89.txt`: the NN squares 1, 2, 4, 8,
16, 24, 32, 40 and 48 at batch 64, 128, 2048 and 32768, both layouts. The squares 49, 56, 64 and every batch-1 or
batch-63 square rank `vendor` first. The edges are checked by `GemmTranscribedTable.Sm89BracketsTheTranscribedVendorEdges`
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
