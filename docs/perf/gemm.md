# GEMM {#perf_gemm}

> **Status:** current · RTX 4090 (sm_89) and RTX PRO 6000 Blackwell Max-Q (sm_120), CUDA 13.2, /opt/dpcpp-cuda

gemm selects through flat kernel selection (@ref design_flat_selection). This page lists the choices, the
`tuned/gemm.*` rows that Auto ranks first on sm_89, and the measurements behind each window. The sm_89 table is
transcribed from the old router (`424a45bc`) and is untimed, so these windows are what Auto runs on sm_89. The
sm_120 table is deep-measured for float and complex<float> on the kernels of the 2026-10 campaign below;
double and complex<double> keep transcribed rows (FP64 tuning is deferred on 1:64 parts, @ref dev_agent_guide).

## GEMM: the sm_120 cuBLAS gap campaign (2026-10)

Native gemm against cuBLAS on sm_120 over the coarse grid, every cell where both were timed. *Losing*: native
> 1.05× vendor; *excess*: summed native − vendor ms over losing cells; *geomean*: native / vendor over all
timed cells (< 1 is faster than cuBLAS). Each row is a full pass of the kernels at that point: coarse, except
the last, which is deep (2.2-2.7× the cells, refinement midpoints included), so compare its losing *share*.

| stage | float losing / excess / geomean | complex<float> | complex<double> |
|---|---|---|---|
| before PR 188 (deep table) | 2685 / 554 ms / 1.074 | 3270 / 1467 ms / 1.061 | 1652 / 6386 ms / 1.000 |
| PR 188 | 1424 / 235 ms / 1.006 | 2312 / 337 ms / 0.970 | — |
| + WP-J, K, L, N2, Q | 688 / 111 ms / 0.960 | 623 / 209 ms / 0.924 | 44 / 40 ms / 0.794 |
| + WP-O, T, F | 208 / 27 ms / 0.869 | 615 / 66 ms / 0.914 | 3 / 3 ms / 0.777 |
| + WP-V, U | 210 / 24 ms / 0.861 | 209 / 11 ms / 0.862 | deferred |
| + WP-W, X, Y (deep, shipped table) | 242 of 5909 (4.1%) / 15 ms / 0.837 | 354 of 11471 (3.1%) / 13 ms / 0.858 | deferred |

What is left in the shipped table (median loss 1.07-1.08×): float strided skinny shapes on the wide tiles
(TT/NN/TN/NT, up to 1.22×) and complex<float> NT/NC strided mid shapes on the 128×128 and 64×64 wide tiles.
Float mid shapes are closed by WP-W's 256×128 and 128×256 tiles (every remaining float mid bucket is under
0.5 ms of excess); a 128×64 float tile for ragged n (181) was considered and not built, those cells now
summing to under 1 ms. Every work package has its own section below.

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
views, calls `select::run` (`gemm.cc:209-217`), and launches one kernel from `src/ops/gemm/kernels/gemm_kernels.cc`.

| spelling | launcher | scalars | `can_run` beyond the common term (`gemm.cc`) |
|---|---|---|---|
| `direct` | `gemm_direct`, one work-item per C element | all | `grid && max_wg >= 64` (`:103`) |
| `tiled` | `gemm_tiled`, 16×16 shared tile | all | `grid && max_wg >= 256` (`:104`) |
| `small` | `gemm_small`, `small_batched.hh` | all | `max(m,n,k) <= small_max_dim<T>()` (64; 32 for complex<double>), `max_wg >= 128` (144 or 196 for float NN 33..56), sub-group 32 except on the float NN tiled leg, `small_slm_bytes` within the budget (`small_fits`, `src/ops/gemm/choice.hh`) |
| `reg:m=..:n=..:k=..:u=..` | `gemm_reg`, register-tiled family | float | `grid`, a `reg_configs` entry, `reg_form(*cfg, ta, tb)`, `max_wg >= cfg->threads()` (`:109-115`) |
| `wide:m=..:n=..:k=..` | `gemm_wide`, 16-byte-granule tiles | all (`wide:m=128:n=128:k=8` complex<float> only) | `grid`, a `wide_configs` entry, `wide_form<T>` (`wide_serves<T>` in `choice.hh`: a complex form must be listed exactly; a real `Trans` uses the listed `ConjTrans` instance), `max_wg >= cfg->threads()` (`:116-120`) |
| `splitk:m=..:n=..:k=..:s=..` | `gemm_splitk`, a wide host tile over s k-slices plus a reduction (WP-L, *unmeasured*) | all | `grid`, s ∈ {2, 4, 8}, batch × s ≤ min(`max_groups_yz`, INT_MAX), a `splitk_host` row serving the form, the host's `max_wg` and local-memory terms, ldw × n ≤ INT_MAX, partials ≤ `scratch_budget` |
| `vendor` | `backend::gemm_vendor` | all | `d.has_vendor` (`:121`) |

The common native term (`gemm.cc:98-100`):

```cpp
const bool device = d.is_gpu || !d.has_vendor;
const bool native = device && precision == ComputePrecision::Default && s.m > 0 && s.n > 0 && s.k > 0 &&
                    A.batch_size() >= 1 && !A.is_heterogeneous() && !B.is_heterogeneous() && !C.is_heterogeneous();
```

`device` keeps a CPU queue on the host BLAS vendor (maintainer decision). The batch is no term: `direct`, `tiled`,
`reg`, `wide` and `splitk` put it in SYCL dimension 0 (CUDA grid z, 65535; split-k batch × s), and `launch`
(`src/ops/gemm/gemm.cc:165-180`) runs a larger batch as slices of at most `max_groups_yz` (split-k: `max_groups_yz / s`)
items, each slice re-deriving its own leg. `small` is 1-D and takes the whole batch
([the batch slices](level3.md#level-3-the-65535-batch-slices)). Candidate counts: float 25, double 12 (no `reg`), complex<float> 13, complex<double> 12 (no `reg`).

Forms are folded after the real-scalar C→T fold:

| choice | forms instantiated |
|---|---|
| `direct`, `tiled`, `small` | any |
| `reg:m=32:n=32:k=8:u=1`, `reg:m=64:n=64:k=8:u=1` | NN |
| `reg:m=64:n=64:k=16:u=1`, `reg:m=128:n=128:k=8:u=1`, `reg:m=128:n=128:k=8:u=4` | NN NT TN TT |
| `reg:m=256:n=128:k=8:u=4`, `reg:m=128:n=256:k=8:u=4` | NN NT TN TT (see [WP-W](#gemm-wp-w-a-256x128-float-tile-with-four-copy-async-stages)) |
| `reg:m=128:n=32:k=16:u=1`, `reg:m=128:n=32:k=32:u=1` | NN NT TN TT |
| `reg:m=128:n=64:k=16:u=1` | NT TN TT |
| `reg:m=32:n=128:k=16:u=1` | NN TN TT |
| `reg:m=128:n=64:k=32:u=4`, `reg:m=128:n=64:k=32:u=2` | NN |
| `wide:m=64:n=64:k=16`, `wide:m=64:n=64:k=8` | all nine complex forms; real NN, NT, TN, TT (see [the k=8 slab](#gemm-wp-f-a-k8-slab-for-the-64x64-wide-tile)) |
| `wide:m=128:n=128:k=8` | complex<float> only, all nine forms (see [WP-O](#gemm-wp-o-a-128x128-complex-float-tile-on-wide-v2)) |
| `wide:m=128:n=32:k=16` | complex NC, NT, TN, CN; real NT, TN, TT |
| `wide:m=32:n=128:k=16` | complex CN, TN, NT, NC; real NT, TN, TT |
| `wide:m=32:n=32:k=16`, `wide:m=16:n=16:k=16` | all nine complex forms; real NN, NT, TN, TT (see [blackwell.md](blackwell.md)) |
| `wide:m=64:n=32:k=16`, `wide:m=32:n=64:k=16` | complex NN, NT, NC, TN, CN, CT; real NN, NT, TN, TT (see [the skinny tiles](#gemm-skinny-64x32-and-32x64-wide-tiles)) |

`WideCfg::forms` lists the complex instantiations and `WideCfg::real` the N/C builds a real scalar compiles;
a real form is served when its C-folded form is in `real`, and any form only for a scalar in `WideCfg::scalars`
(`wide_compiles`, `wide_serves`). Split-k hosts
inherit their row's forms. Real TT on the six non-64×64 tiles and complex TT, TC, CC on 32×32 and 16×16:
[WP-T](#gemm-wp-t-tt-tc-and-cc-on-the-small-transposed-tiles). See [GEMM: wide instantiations in every transpose
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
build (`gemm.cc:189-197`), and each item chooses separately.

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
4. **Staged epilogue** for k ≤ 64 on the aligned leg and at every k on the predicated leg (since WP-F), see
   [The staged epilogue](#the-staged-epilogue).

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
  forms (NC, NT, TN, CN); 32×32 and 16×16 every form but TT, TC and CC
  ([WP-T](#gemm-wp-t-tt-tc-and-cc-on-the-small-transposed-tiles) later added them).
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
the small-tile mask (every form but TT, TC and CC), the rule WP-A settled on for build cost
([WP-T](#gemm-wp-t-tt-tc-and-cc-on-the-small-transposed-tiles) later added real TT).

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
    and so is the direct epilogue at k > 64 (on the aligned leg only since WP-F, which stages the predicated leg
    at every k).
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
- Prefetching all 16 prior values before the pass barrier also spilled. WP-F later ships exactly that batch,
  spill-free, on the predicated leg (loaded after the stage write, when the pass's accumulators are dead; see
  [WP-F](#gemm-wp-f-float-mid-size-parity-on-sm_120)).
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

**Local memory.** complex<float> v2 asks for 32 KiB on the 16-byte leg and 33,792 B on the element leg (two buffers
per operand; the element leg pads its k-contiguous rows, [WP-Y](#gemm-wp-y-coalesced-element-leg-lane-maps)), over
an Apple GPU's 32 KiB limit once the runtime's 8 B per item is added. can_run's wide arm checks `wide_slm_bytes<T>`
(choice.hh, static_asserted against the kernel) against `slm_budget_for(threads)`, so Metal refuses it there instead of failing at launch.

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
- an 8-byte vector load for complex<float> on the element leg (WP-Q measured an 8-byte `cp.async` instead: no gain,
  [negative](#gemm-wp-q-element-leg-loads-negative)).

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
- **Local memory** halves: float 8 KiB, complex<float> 16 KiB, double 8 KiB, complex<double> 16 KiB (since WP-Y
  the float and complex<float> element leg: 8,704 B and 16,896 B)
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
| prefetch | the next slab's granules go into registers before the current FMAs: (m+n)·k / threads elements per thread (128x32: 10, 64x32: 12, 32x32: 16, 16x16: 8). Off on 64x64 double and on every complex<double> tile (prefetch spilled all 85 cdouble instantiations, 255 registers, sm_120 probe) |
| granules | 16 bytes along the contiguous dimension: k for A^T and B, i for A and B^T. Along k a warp takes 32/KG columns of KG granules, i fastest across lanes |
| swizzle | the k-contiguous store XORs its granule column by (kk/VecN)·(32/TileK) mod granules-per-row, the fragment read applies the same XOR. Conflict-free except float 16x16 (2-way). None for complex<double>: a warp stages 16 i × 2 kk instead, lanes l and l+8 one 32-byte sector, each 8-lane phase 8 consecutive i of one row |
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
- `soff<false, …>` for A in `compute`: red only in the forms with A transposed (TN, CN, CT), and
  never for cdouble (no XOR there).
- dropping `gk + q < k` in `gload`: red only in the k = 17 and 37 NaN cases.
- dropping the barrier before `staged_c_store`: possible item mismatches in the ld = m+1 batch cases
  only (a race, so not guaranteed).

**Registers** (`scripts/register_probe.sh`, sm_120 GPU-only tree): zero spill on every entry function of
`libbatchlas_sycl`. Before the two fixes below, 50 complex<double> instantiations of
`GemmWideTransposedKernel` spilled (255 registers, 2-4.7 KB), every 16-byte × split one among them;
before WP-K every cdouble instance sat at 124-136 with zero spill.

- **Hoisted swizzle addresses.** At VecN = 1 the XOR differs per kk, and the per-kk × band fragment
  addresses are slab-invariant, so they were hoisted out of the slab loop, one register each. Fix: no XOR
  and the sector-pair staging map above at VecN = 1 (50 → 32).
- **The full/predicated diamond.** In the PTX the slab's FMAs sit in the join block after
  `if (more) stage`, not before it, and ptxas pulls them back only across if-convertible code. The
  16-byte leg's per-granule `full ? 16-byte : predicated loop` diamond is not if-convertible. Fix: at VecN = 1 one
  guarded 16-byte load per granule (32 → 0). Peeling the last slab to drop `if (more)` instead spilled 54.

| scalar | element leg | element, split | 16-byte leg | 16-byte, split |
|---|---|---|---|---|
| float | 40-96 | 38-80 | 48-85 | 40-87 |
| double | 68-153 | 52-172 | 62-164 | 52-164 |
| complex<float> | 72-168 | 64-168 | 64-168 | 64-168 |
| complex<double> | 64-168 | 80-182 | 66-168 | 80-186 |

**Pending:** the A/B on
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
  At batch 32768 batch × s ≥ 65536 refused every s on CUDA when this grid was timed (split-k now runs such a batch
  as slices), so split-k is timed at batch 128 and 2048 only, and a call nearer the 32768 row (log distance: batch
  above about 8192) never picks it. Accepted: at that batch the unsplit grid already fills the device.

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
- `GemmCandidates.SplitKPastTheGridRunsAsSlices`: s = 2 and 8 at batch ⌊65535 / s⌋ (one slice), one past it and
  2 ⌊65535 / s⌋ + 1 (two and three slices), every item bit-identical to item 0.
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
- (Before the batch slices) dropping `batch * s` from `can_run`: the split-k grid-ceiling test turned red (the
  launch fails past the grid). The term is now the per-slice `slice * s <= INT_MAX`.
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

## GEMM: WP-N2, the u4 stack frame and zero-filled predicated copies

Measured 2026-10-19, RTX PRO 6000 Blackwell Max-Q (sm_120), one GPU, `batchlas_tune` A/B (beta = 1, every arm
verified), 8 reps × 2 passes in alternating order; ratios are the mean of both passes against cuBLAS.

`reg:m=128:n=128:k=8:u=4` (WP-N) shipped a 16-byte `cp.async` aligned leg that ran 23-33× slower than u1 (NN packed
1024³ b128: 377 ms against 16.6). Shipped now:

| leg | u1 | u4 |
|---|---|---|
| aligned | register double buffer | the same instantiation (`..._k8[_nt\|_tn\|_tt]_aligned`) |
| predicated | register double buffer, scalar loads | 4 local slabs filled by zero-filled 4-byte `cp.async`, 3 in flight (`..._k8_s4[_nt\|_tn\|_tt]`) |
| staged epilogue (k ≤ 64) | every form | NT, TN, TT; NN stays direct |

**The 23-33× defect.** ncu: 99% of L1 sectors were local memory, IPC 0.17. The aligned body ran out of line on a
stack copy of the functor (probe: 520-680 B stack frame, 0 spill; WP-N's "no stack frame" was wrong), and each
`asm volatile(... : "memory")` made LLVM reload every member through `this` after every copy. Launching through an
always-inline wrapper removed it (stack 0); the aligned u4 then ran at 1.10× u1's time (NN 1024³).

**The aligned cp.async leg is deleted** (negative result): it never beat the register prefetch.

| cell (packed, aligned leg) | u1 | u4 aligned cp.async | vendor ms |
|---|---|---|---|
| NN 1024³ b128 | 1.043× | 1.145× | 6.76 |
| NT 1024³ b128 | 1.002× | 0.999× | 6.88 |
| NN 512×512×128 b2048 | 1.032× | 1.017× | 5.67 |
| TT 512³ b128 | 1.052× | 1.117× | 1.02 |

NN and TT lose on the k-contiguous operand (four 4-byte copies per lane per slab where u1 issues one 16-byte load).
NT ties, and NN 512×512×128 is under the 5% noise floor.

**Predicated pipeline.** Each lane copies one element per copy, zero-filled through `src-size` 0 when its row,
column or k index is out of range, so any ld and any base alignment work and the k tail needs no branch. A packet
operand (A, B^T) takes consecutive m (n) per lane, 32 apart per copy, so each copy is one 128-byte warp access at
an odd ld. A k-contiguous one (B, A^T) keeps the transposing map. Step s: `wait_group 2`, barrier, refill buffer
(s + 3) mod 4, commit, compute. Register-cap details (`scripts/register_probe.sh`, sm_120; the probe only relinks,
so rebuild `batchlas_sycl` first):

- The launch bound alone does not cap registers: staged predicated NN, TN and TT ran 155-159 (one group per SM),
  measured 2.2-2.4× vendor. The pipeline's wrapper adds `sycl::reqd_work_group_size(1, 1, 256)` (`.reqntid`), and
  ptxas then holds 128. Like `BATCHLAS_LAUNCH_BOUNDS` it is NVPTX device code only (the spir64 CPU AOT never sees
  it), but defined in `register_128x128.hh`, not `src/sycl/kernel_attrs.hh`: that header is in every op's tuning
  hash, so moving it there staled all 19 tables (measured: gemm, potrf, posv, trsm pins all moved). AdaptiveCpp
  drops both attributes, so an acpp CUDA build presumably runs this leg at one group per SM (speed only; not
  probed).
- At 128, staged NN and NT spilled 72 and 52 B while each copy computed `ptr + k0 * ld`. With every operand pointer
  bumped one slab per issue, NT fits and NN spills 12 B, so the NN instance takes the direct epilogue (derived in
  `launch_register_128x128_k8`, never a field).
- Shipped: all 7 u4 instances (predicated only; NN has no staged one) 117-127 registers, 0 spill, 0 stack.
  u1 is unchanged (125-139). Rebased on the spill-free complex<double> wide-transposed tiles: total gemm spilling 0.

| cell (layout=strided: ld = m + 1, odd) | u1 | wide 64×64 | **u4** | vendor ms |
|---|---|---|---|---|
| NN 256×512×128 b2048 | 1.425× | 1.457× | **1.316×** | 2.96 |
| NT 1218×128×96 b128 | 1.350× | 1.389× | **1.298×** | 0.194 |
| NN 256×256×64 b1024 | 1.094× | 1.106× | **1.071×** | 0.675 |
| TT 512×512×64 b128 (two runs) | 1.557×, 1.519× | 1.535×, 1.507× | 1.529×, 1.494× | 0.229, 0.237 |
| NN 1448×128×32 b128 (two runs) | 1.663×, 1.660× | **1.507×, 1.483×** | 1.591×, 1.568× | 0.133 |
| NN 384×384×48 b128 | 1.276× | **1.136×** | 1.437× | 0.129 |

u4 wins 8% and 4% on two cells (NN 256×512×128, NT 1218×128×96, the latter borderline at 16 samples); the rest tie
(under the 5% floor) or lose to u1 or wide (k = 32-48 NN, direct epilogue). **u4 is unranked:** no row of
`tuned/gemm.*.txt` names `reg:m=128:n=128:k=8:u=4`, so Auto never runs it until the float `layout=strided` rows are
retuned. On an aligned view u4 is an alias of u1's kernel: a packed cell times the same code twice, and a u4 row
there would report `u=4` for u1's kernel. ncu on NN 256×512×128 (u4): 979 GB/s, 67% DRAM; the top stalls are
`beta * C` waiting on the predicated leg's scattered C loads (direct epilogue), not the mainloop.

Test: `GemmCandidates.Reg128x128CopyAsyncPipelineEverySlabCountAtSaturatingBatch` (5 forms, odd-base or odd-ld
views, slab counts 1-17 with k tails 5, 37, 69 under NaN poison, 300×200 edge groups, NaN C at beta = 0) and the
WP-I test run for u4 as well as u1. Deliberate breaks, run on sm_120 (first red case; both tests stop there):

| break | first red | green before it |
|---|---|---|
| `copy_async_wait<Stages - 1>` | NN k = 5 (one slab, read before it lands) | WP-I u4 NN k = 72, 37 (the race hides) |
| refill buffer `(b + 1) % Stages` | NN 256×128×32 (4 slabs) | NN k = 5, 8, 16, 24 (≤ 3 slabs never refill) |
| drop the B^T pointer bump | NT 256×128×16 (2 slabs) | every NN leg, NT k = 5, 8 |
| B's k bound dropped (`lim` = 4) | NN k = 37, 5 (NaN) | NN k = 72 |
| A's row bound dropped (`lim` = 128) | none: rows past m feed only discarded accumulators | all |
| drop the A pointer bump (packet operand, NN/NT) | NN 256×128×16 (2 slabs); WP-I test NN 256² k = 72 | NN k = 5, 8 |
| A's k bound dropped (packet operand, `lim = m - (m0 + a_m)`) | NN k = 5 (NaN); WP-I test NN k = 37 (NaN) | WP-I NN k = 72 |

A 16-byte-aligned ragged view (Padded 300×200×40) is in the test too, so an aligned-base shortcut in the pipeline
is caught. A zero-filled copy (`src-size` 0) once named its computed address, up to ~127·ld past the operand at a
ragged edge or 7 k rows past k. Since the WP-W review it names the operand's batch base instead (`a_ptr`, `b_ptr`):
+0.0-0.7% on u4 strided TT 1024²×362, TN packed 1024²×362, NN strided 512×256×128 b2048 and NT packed 1024³ (10
reps × 2 passes; see [WP-W](#gemm-wp-w-a-256x128-float-tile-with-four-copy-async-stages)).

## GEMM: WP-Q, a one-round staged epilogue for wide v2

Measured 2026-10-19 on threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120), one GPU, `batchlas_tune
--mode time` (beta = 1, every arm verified), 8 reps (4 at 1024³ and 512³ b2048), two passes in alternated
build order per cell. Parent: the WP-J/K/L/N tree.

Target: the element leg (odd ld). ncu shows the loss is mostly not the loads. float TN strided
2048×1024×32 b128 spends its time in C. The parent staged epilogue (`staged_c_store`, 2 prior loads in
flight per thread) ran 2.26 ms at 56% DRAM throughput; the 16-byte band store on the packed cell ran 1.45 ms
at 86%.

| piece | rule |
|---|---|
| batch | `staged_c_store<…, Batch>`; wide v2 passes `EpiBatch = 4·TileK·64/256` (16 at k = 16, 8 at k = 8): one round trip per chunk. The default 2 keeps the old loop for `launch_wide_transposed` and the NN kernel |
| addressing | only a whole-chunk Batch (`Batch > 2` and `Batch·Threads/64 = CW`; any other Batch keeps the element map): a thread keeps one row (`tid % 64`) and every 4th column, so a load is `p + b·4·ldc`. `nc` (columns inside C, 0 on a row outside m) is the only guard. Priors load only when beta ≠ 0 (a NaN C at beta = 0 is never read) |
| launch bound | `[[sycl::reqd_work_group_size(1, 1, 256)]]` on `GemmWide64Body::operator()`. Without it the bound was ignored ([below](#gemm-wp-q-the-launch-bound-was-ignored)) |

| cell (b128 unless noted) | arm | parent / vendor | new / vendor | new / parent |
|---|---|---|---|---|
| cfloat NN strided 256×512×32 | k16 | 1.37 | 1.07 | 0.80 |
| cfloat TN strided 724×128×32 | k16 | 1.25 | 0.98 | 0.80 |
| cfloat CN strided 1448×64×16 | k16 | 1.28 | 0.96 | 0.76 |
| cfloat NC strided 256×512×32 | k16 | 1.37 | 1.06 | 0.80 |
| cfloat CN strided 304×256×8 | k8 | 1.31 | 1.02 | 0.79 |
| cfloat NC strided 64×1448×8 | k8 | 1.39 | 0.94 | 0.74 |
| float TN strided 2048×1024×32 | k16 | 1.51 | 1.16 | 0.77 |
| float NN strided 1218×128×8 | k8 | 1.53 | 1.19 | 0.79 |
| float NN strided 256×512×96 b2048 | k16 | 1.46 | 1.39 | 0.95 |
| float TT strided 2048×256×32 | k8 | 1.49 | 1.22 | 0.82 |
| float TT packed 362×512×8 | k8 | 1.47 | 1.14 | 0.82 |
| float NT packed 181×1024×64 | k8 | 1.26 | 1.14 | 0.91 |
| float TT packed 1218×256×64 | k16 | 1.26 | 1.00 | 0.81 |
| float TN packed 1024×1024×362 | k16 | 1.49 | 1.48 | 1.00 |
| cfloat NN strided 1024³ | k16 | 1.19 | 1.21 | 1.01 |
| 16-byte leg: float TN packed 2048×1024×32 | k16 | 1.03 | 1.03 | 1.00 |
| 16-byte leg: float TT packed 1024×1024×32 | k8 | 0.97 | 0.97 | 0.99 |
| 16-byte leg: float NN packed 512³ b2048 | k16 | 1.40 | 1.39 | 1.00 |
| 16-byte leg: cfloat NC packed 512×512×64 (16 reps) | k8 | 0.96 | 0.97 | 1.013 |
| 16-byte leg: cfloat NN packed 1024³ | k16 | 1.15 | 1.15 | 1.00 |

Both passes of every cell agree within 3%. The win is the small-k cells, where C traffic (2mn) dwarfs A and B
((m+n)k). Large k (float TN packed k = 362, cfloat 1024³) is unchanged: the gap there is the mainloop.

The complex<float> 16-byte leg lost 0.6-1.3% in every rerun: NC packed 512×512×64 k8 1.013 (16 reps),
NN packed 1024³ k16 1.006, CT packed 512³ k8 1.006. Timed apart, half of it is the now-applied launch
bound, which moves its register allocation (k8: 124-135 → 96), and half the epilogue
([split](#gemm-wp-q-epilogue-and-bound-timed-apart)). Not resolved.

Tests: the existing wide v2 suites (`WideV2EveryFormAtSaturatingBatch`, `WideK8SlabStraddles…`,
`WideV2TileAndSlabEdgesUnderNaNPoison`, `CoalescedCEpilogueEveryTileAtSaturatingBatch`) already run the
staged path: odd and even ld, ragged m and n, beta ≠ 0, NaN C at beta = 0, batch 1024 bit-identical.
Deliberate breaks, run on the GPU (filter: those four plus `PinnedCandidatesEveryFormShapeAndLayout`):

- store guard `< nc` → `< cw` (writes past n and m): the binary aborts in the first float
  `PinnedCandidatesEveryFormShapeAndLayout`.
- row term only, `nc = lim` (rows past m written into the ld gap; review follow-up, full binaries): no
  abort. `gemm_candidates_tests` red 33 (the wide v2 and C-epilogue suites, `BetaZeroDoesNotReadC`,
  `ParentLdPanelUpdate`, `GridCeilingIsACanRunTerm`, `SaturatingBatchIsBitIdentical…`, `PinnedRun…`, and
  complex `AutoReadsEveryKeyField`: every case with a ragged m on the staged group), `gemm_tests` red 21 (the
  forced wide ragged and sub-view cases), float, complex<float> and complex<double> alike.
- priors read at beta = 0: red exactly `WideV2EveryForm`, `WideK8Slab` and `CoalescedCEpilogue` for float,
  complex<float> and complex<double> (their NaN-C cases); 9 tests.
- column step `4·ldc` → `4·(ldc | 1)`: red in the 16 tests with an even-ld staged group (float,
  complex<float>, complex<double>); odd-ld cases stay green.

### GEMM: WP-Q, the launch bound was ignored

ptxas printed `.minnctapersm is ignored when neither .maxntid nor .reqntid is specified` 112 times for
`libbatchlas_sycl` (sm_120): `intel::max_work_group_size(1, 1, N)` on these functors emits no `.maxntid`, so
`BATCHLAS_LAUNCH_BOUNDS` never capped a register count. Adding `reqd_work_group_size` to wide v2 drops the
count to 24 (the 128x128 functor, untouched here). Registers (`scripts/register_probe.sh`):

| instances | parent (no bound) | new (bound applied) |
|---|---|---|
| float, all forms, k16 / k8 | 64-72 / 55-61 | 80 / 79-80 (3 groups) |
| complex<float> k16 | 106-128 | 122-128 (2 groups) |
| complex<float> k8 | 112-128 | 96 |
| complex<double> | 196-224 / 134-168 | 222-230 / 152-154 (1 group) |

Without the bound the new epilogue pushed complex<float> k8 NT/NC/TT to 130-135 registers (one group per
SM). With it: zero spill on every gemm entry except the 50 known complex<double> `GemmWideTransposedKernel`
ones (0 after the rebase onto the WP-K spill fix). float at 4 groups per SM (`kMinGroupsPerCu = 4`, 64 registers, no spill) measured 0.99-1.03 of 3 groups
on 8 cells, so the bound stays 3.

The other users of `BATCHLAS_LAUNCH_BOUNDS` (128x128 gemm, getrf/gesv/posv tiny) still ignore it:
[known-defects #20](../design/known-defects.md#defect-20-batchlas_launch_bounds-emits-no-maxntid-so-ptxas-ignores-the-bound).
sm_89: nothing was probed or timed there. Every wide v2 row in the sm_89 tables was raced with no bound in
effect, so treat them as a hypothesis until the retune. The gemm kernel hash moved, so the sm_120 and sm_89
gemm tables record a stale hash until then.

### GEMM: WP-Q, epilogue and bound timed apart

Review follow-up, same machine and protocol (8 reps, 16 for the 16-byte cell, two passes in alternated
order). Three builds: the parent; the new epilogue with the bound still ignored (`reqd_work_group_size`
removed); WP-Q. new / old per step:

| cell (b128) | arm | epilogue alone | bound alone |
|---|---|---|---|
| float TN strided 2048×1024×32 | k16 | 0.753 | 1.013 |
| float NN strided 1218×128×8 | k8 | 0.848 | 0.944 |
| float TT packed 362×512×8 | k8 | 0.801 | 0.948 |
| cfloat NN strided 256×512×32 | k16 | 0.799 | 1.013 |
| cfloat NC strided 64×1448×8 | k8 | 0.695 | 1.022 |
| 16-byte leg: cfloat NC packed 512×512×64 | k8 | 1.006 | 1.006 |

The epilogue is the win. The bound alone helps float k8 (56-69 → 79-80 registers, 5-6%) and costs
complex<float> 1-2%. complex<float> at one group per SM (`kMinGroupsPerCu = 1`, k16 138-150 / k8 126-140
registers) was tried instead: new / 2-group bound 1.16 on CN strided 1448×64×16, 1.03 on NN strided
256×512×32, 0.97 on NC packed 512×512×64 k8 and on NN packed 1024³ (12 reps; 4 at 1024³). Mixed, so 2 groups
stays.

### GEMM: WP-Q, element-leg loads (negative)

On float TN packed 1024×1024×362 b128 (element leg, 3.79 ms) against 360 (16-byte leg, 3.13 ms), both run at
89% L1/TEX throughput, mostly fragment reads (633M shared wavefronts each). The element leg adds 4.4× the global
sectors (969M vs 218M) and 3.4× the requests. Three rewrites of the element leg each lost to the parent's
4-scalar loads, so the leg is unchanged:

| variant | result |
|---|---|
| `cp.async` 4 B (float) / 8 B (complex<float>, 8-byte base), coalesced lane map, swizzle `kk & 7` | complex<float> 110 → 185 registers (8 swizzle offsets in the fragment read): NN strided 256×512×32 1.26× slower |
| `cp.async` with the parent lane map and swizzle | float 1.02-1.04× slower, complex<float> 0.98-1.01 |
| register staging, float i-contiguous operand read 32 consecutive floats per load (scalar shared stores) | 1.02-1.04× slower at the 85-register cap. At a 64 cap it spilled and ran 1.14-1.28× slower |

WP-Y's k-contiguous lane map is the fix that held ([WP-Y](#gemm-wp-y-coalesced-element-leg-lane-maps)).

## GEMM: WP-O, a 128x128 complex float tile on wide v2

Measured 2026-10-20 on threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120), one GPU, `batchlas_tune --mode
time` (beta = 1, every arm verified), 4-8 reps, two passes in alternated arm order per cell. Both passes of every
cell agree within 3%, except TN packed 512×256×64 (0.3 ms, up to 9% on every arm).

`wide:m=128:n=128:k=8` (complex<float> only, all 9 forms) runs `GemmWide64Body` at `TileM = TileN = 128`:

- 256 lanes as 16 × 16, an 8 × 8 complex thread tile (128 accumulator registers).
- k = 8 slab, split staging (warps 0-3 A, 4-7 B), two buffers: 32 KiB (16-byte leg), 33,280 B (element leg, WP-Y).
- Each operand is staged as two 64-wide halves. Each half uses the 64x64 lane map, `KStep` and swizzle, the second
  one 64 elements down i (64 ld for a k-contiguous operand). `sstore` takes the row width as a parameter.
- Epilogue: band store, or `staged_c_store` with `EpiBatch = 2·TileK·(TileM + TileN)/256` = 16 (four passes).
- Launch bound: one group per CU (`TileM·TileN > 4096`). Zero spill, no stack frame (`scripts/register_probe.sh`):
  254-255 registers on sm_120, 248-255 on sm_89 (a `-DBATCHLAS_NVIDIA_ARCH=sm_89` tree, probed only).
- The halves unroll through `static_for`. A one-trip `for` loop per half let LICM hoist the edge tests out of
  the k loop and spilled 4 B in float `wide:m=64:n=64:k=16` NC (both legs) on sm_89 at its 80-register cap,
  while sm_120 stayed clean. With `static_for`, that kernel's PTX has the parent's opcode counts (1680
  instructions) and both probes print `total spilling 0`.

`WideCfg` gains a scalar set (`scalars`, default all) and a `v2` flag. `wide_serves`, `candidates<T>()` and
`launch_wide_form` read the scalar set, so the row exists for complex<float> only (13 candidates; complex<double>
stays at 12). `splitk_host` excludes v2 rows. `wide_v2_row<T>` is 64x64 (every scalar but double) or a `v2`
row for a scalar it compiles. A larger tile at k = 16 is a `static_assert`: it would need 64 KiB. It is a new candidate: no table row
ranks it yet, so Auto is unchanged until the retune.

**Why 128x128 and not the planned 128x64.** ncu on the parent (`wide:m=64:n=64:k=16`) at cfloat NN packed 1024³
b128 (durations at ncu's locked clock): every arm is issue-bound, not starved for intensity. DRAM is at 13-14%
and L2 hits 84-91%. Issue slots are 73-78% busy and FFMA is 90-94% of the instructions. The gap is the
non-FFMA instructions per FFMA (17.205G FFMA in every arm), and the thread tile sets the LDS share:

| arm | thread tile | warps/SM | regs | instructions | LDS.128 | other non-FFMA | ms |
|---|---|---|---|---|---|---|---|
| 64x64x16 (parent) | 4×4 | 16 | 128 | 18.854G | 1.074G | 0.575G | 22.21 |
| 128x64x8 | 8×4 | 8 | 180 | 19.084G | 0.805G | 1.074G | 21.51 |
| 128x128x8 | 8×8 | 8 | 254 | 18.455G | 0.537G | 0.713G | 21.03 |
| cuBLAS `cutlass_100_simt_cgemm_128x128_8x4_nn` | 8×8 | 8 | 220 | 18.263G | 0.537G (+0.05G LDS) | 0.47G | 20.00 |

128x64 removed a quarter of the LDS but added 0.47G of address arithmetic (LEA, LOP3, SHF), where ptxas
rematerializes the swizzled fragment bases. 128x128 matches cuBLAS's instruction count to 1%. What remains is
stall: `dispatch_stall` 0.44 vs 0.31 per issue and `long_scoreboard` 0.09 vs 0.02. cuBLAS stages through
`LDGSTS` with four stages; v2 uses one register prefetch.

**128x64 (negative), measured and dropped.** 8×4 complex, one group per CU (180 registers). Against
`wide:m=64:n=64:k=16` it was 0.95-0.99 on the large cells (NN, NT, CN, NC, CC). It was slower on TN strided 768³
(1.037) and on the skinny cells. In the 15 cells timed beside 128x128 (two passes each) it ranked first among the native arms in none: 128x128 or 64x64 beat it in each, so its
18 kernels are not built. With 2 groups per CU (128-register cap) it spilled 256-3016 B. Moving the next slab's
loads after the FMAs (no prefetch) made it 4.3-4.5 KB.

Re-measured after the review fixes (`static_for` halves change the 128x128 code; sm_120 registers stay 254-255),
same protocol, 2026-10-20. Rows marked * are from the first run and were not re-timed.

| cell (cfloat) | 128x128 / vendor | 64x64x16 / vendor | 64x64x8 / vendor | 128x128 / 64x64x16 |
|---|---|---|---|---|
| NN packed 1024³ b128 | 1.039 | 1.137 | 1.193 | 0.91 |
| CN packed 1024³ b128 | 0.926 | 1.002 | 1.078 | 0.92 |
| NN strided 1024³ b128 | 1.079 | 1.196 | 1.284 | 0.90 |
| NT packed 512³ b2048 | 1.020 | 1.140 | 1.182 | 0.90 |
| TN strided 768³ b128 | 1.066 | 1.171 | 1.341 | 0.91 |
| NN packed 512×512×128 b2048 | 1.030 | 1.105 | 1.142 | 0.93 |
| NC strided 256×1024×64 b2048 | 0.944 | 1.024 | 1.069 | 0.92 |
| CC packed 1024³ b128 | 0.963 | 1.046 | 1.118 | 0.92 |
| NN packed 256³ b128 | 0.892 | 0.947 | 0.995 | 0.94 |
| CN strided 384³ b128 | 0.975 | 1.017 | 1.156 | 0.96 |
| NN strided 128³ b2048 * | 0.929 | 0.994 | 1.064 | 0.93 |
| NN packed 128×512×64 b2048 | 1.057 | 1.070 | 1.102 | 0.99 |
| NT packed 192³ b2048 * | 1.290 | 0.952 | 0.982 | 1.35 |
| TN packed 512×256×64 b128 * | 1.090 | 0.894 | 0.952 | 1.22 |
| NN packed 64×2048×32 b128 | 1.489 | 1.013 | 1.001 | 1.47 |

Against the parent, the 0.90-0.94 cells clear 5% in both passes and both runs. CN strided 384³ (0.96-0.97) and
NN 128×512×64 (0.99) are under 5% at 4-8 reps × 2 passes: screens, not wins (§10). Against cuBLAS, 128x128 is
ahead only on CN and CC packed 1024³, NC strided, NN 256³, CN strided 384³ and NN strided 128³; NN, NT and TN at
the large cells stay 2-8% behind.

The parent stays best where 128-wide tiles waste work (m = 64), cover the batch in too few groups, or lose the 64x64
tile's 16 warps (192³, k = 64). Those windows are the retune's to draw.

**Build cost**, `libbatchlas_sycl` link alone (`.so` removed, objects current), GPU-only tree, two runs each: parent
79.1-79.3 s, with 128x64 alone 81.8 s, with 128x64 and 128x128 94.1-95.1 s, with 128x128 alone (the shipped
tree, a later session on the same box) 86.2-87.2 s, +9-10%. That is under the 25% ceiling, so all 9 forms stay.

Tests: `GemmCandidates.Wide128x128EveryFormAtSaturatingBatchAndTileEdges` runs all 9 forms with NaN in every gap and
beta ≠ 0, each form to its own first failure (so a break's red set is per form). It skips where
`wide_slm_bytes` (33,280 B) is over the device's local memory budget (the M5 Max):

- (m, n) ∈ {(63, 65), (64, 300), (100, 128), (127, 257), (128, 100), (129, 129), (255, 191), (257, 64)}: m
  straddles 64, 128 and 256, n straddles 64, 128 and 256 (n = 300 runs a third group at n0 = 256);
  k ∈ {7, 8, 9, 16, 17, 33}, at ld = m + 1 and padded;
- 257×136×33 at ld = m + 1 and 256×256×24 padded, at batch 1024 (7 distinct problems, bit-identical items);
- every `wide_v2_leg_layouts` row at 288×136×48;
- beta = 0 with NaN in C.

The oracle row (`cf_only`), the literal local-memory size (32 KiB then, 33,280 B since WP-Y) and the trace names are
added, and `PinnedChoiceLaunchesItsOwnKernel` runs the leg layouts on the new row.

Deliberate breaks, run on the GPU. The per-form red sets come from the new test alone. In
`PinnedCandidatesEveryFormShapeAndLayout`, `PinnedRunIsTheLauncherBitForBit` and `CanRunEqualsLaunch` each break was
red for complex<float> and `wide:m=128:n=128:k=8` only, but those tests stop at their first failure.

- B's second half stored over its first (`b_ib + 64·h` → `b_ib`): red in all 9 forms.
- A's global half offset not scaled by lda (`a_half` = 64): red in the 6 forms where A is k-contiguous (TN, TT, TC,
  CN, CT, CC), green in NN, NT and NC.
- B's global half offset not scaled by ldb (`b_half` = 64): red in the 3 forms where B is k-contiguous (NN, TN, CN),
  green in the other 6.
- The 64x64 band stride `GPR/NB` in the B fragment read (equal to `NSep/VecN` at 64x64 for every scalar): red in
  all 9 forms.
- Green, recorded: a half's i bound without the `- 64·h`. It only stops reads past m or n, whose products land in
  rows and columns that are never stored, so no value test can see it. The 64x64 i bound has the same property.

`LDGSTS` staging: [WP-V](#gemm-wp-v-three-copy-async-stages-for-the-128x128-complex-float-tile). The 64x128 tile
was not built: 128x128 makes it redundant on these cells.

## GEMM: WP-V, three copy-async stages for the 128x128 complex float tile

Measured 2026-10-20 on threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120), one GPU, `batchlas_tune --mode
time` (beta = 1, every arm verified), 2 passes in alternated order (parent, new, new, parent): 10 reps for the
variants table, 12 for the parent table. Ratios are each build's mean over both passes against its own cuBLAS arm.
Nothing here was timed on another architecture.

`wide:m=128:n=128:k=8` (complex<float>) gets a second leg, derived in `launch_wide64_v2`, never a field or key:

| leg | when | staging | local | registers (sm_120 / sm_89) |
|---|---|---|---|---|
| cp.async (trace `…_s3`) | DPC++, `cuda_compute_capability() >= 120`, slm budget ≥ 48 KiB, A and B bases 8-byte aligned | 3 slabs, 2 in flight, no register prefetch | 48 KiB | 229-232 / ≤ 249 |
| register prefetch (WP-O) | otherwise (acpp, below sm_120, Metal, a 4-byte base) | 2 slabs | 32 KiB (16-byte leg), 33,280 B (element leg) | 254-255 / 248-255 |

`BATCHLAS_GEMM_WIDE_STAGES=2` forces the register prefetch; `=3` admits the cp.async leg from sm_80 (the A/B
switch for an unmeasured architecture, sm_89 included).

- Copies (`wide_v2::acopy`): 16 bytes per granule along i on the 16-byte leg (`src-size` 8 or 0 at the i edge), two
  8-byte copies on the element leg, 8 bytes per element along k (a granule pairs two columns, which land in
  different rows). Out-of-range elements are zero-filled (`src-size` 0), so the k tail needs no branch. The
  swizzle and fragment reads are unchanged.
- Step s: `wait_group 1`, barrier, refill the buffer slab s - 1 was read from, commit (empty past the end), compute.
- Lanes 0-127 copy the first 64-wide half of A and of B, lanes 128-255 the second, so all eight warps issue copies.
- can_run keeps the two-slab term (`wide_slm_bytes`, 33,280 B): the cp.async leg only runs where the launcher sees
  48 KiB, else the two-slab leg.
- The cp.async legs trace as the register-prefetch name plus `_s3`.
- Every 8-byte copy needs an 8-byte-aligned element. `alignof(std::complex<float>)` is 4, so a view at a 4-byte
  base is legal; the launcher sends it to the register prefetch.
- `scripts/register_probe.sh`: 0 spill, 0 stack on both probes (sm_120 build and a `-DBATCHLAS_NVIDIA_ARCH=sm_89`
  tree, probe only).

**Variants**, one build, each timed against the WP-O leg in the same binary (same protocol, vendor-normalised):

| variant | cells | vs WP-O leg |
|---|---|---|
| 4 stages, split staging (warps 0-3 A, 4-7 B), 64 KiB | 21 | 0.940-1.003 (TN strided 1024×1024×32 1.003) |
| 4 stages, every lane both operands, 64 KiB | 9 | 0.943-0.982 |
| **3 stages, every lane both operands, 48 KiB** | 21 | **0.926-0.982**, best or within 1% of best in every cell |

ncu, cfloat NN strided 512³ b128 (stalls per issue; durations at ncu's locked clock):

| arm | ms | issue busy | dispatch | long_scoreboard | mio_throttle | lg_throttle |
|---|---|---|---|---|---|---|
| WP-O register prefetch | 3.00 | 67% | 0.48 | 0.11 | 0.04 | 0.04 |
| 4 stages, split | 2.97 | 66% | 0.36 | 0.13 | 0.20 | 0.08 |
| **3 stages, spread (shipped)** | 2.78 | 71% | 0.35 | 0.14 | 0.06 | 0.05 |
| cuBLAS `cutlass_100_simt_cgemm_128x128_8x4` | 2.74 | 73% | 0.29 | 0.07 | 0.08 | — |

Split staging put every copy on four warps (`mio_throttle` 0.20); spreading them removed it. The remaining gap to
cuBLAS is `long_scoreboard` (0.14 vs 0.07).

**Shipped leg against the parent build** (two builds, the same cells; vendor-normalised = new/vendor over
parent/vendor):

| cell (cfloat, b128 unless noted) | parent / vendor | new / vendor | vendor-normalised |
|---|---|---|---|
| NN strided 256×1024×23 | 1.099 | 1.060 | 0.965 |
| TN strided 1024×256×78 | 1.086 | 1.037 | 0.955 |
| NN strided 512³ | 1.075 | 1.004 | 0.934 |
| TN strided 768³ | 1.069 | 0.994 | 0.930 |
| NC strided 256×512×78 | 0.971 | 0.936 | 0.964 |
| NN packed 2048×362×64 | 1.076 | 1.057 | 0.982 |
| NN strided 512×512×23 | 1.117 | 1.071 | 0.959 |
| NT strided 256×256×181 | 0.894 | 0.850 | 0.951 |
| NC strided 128×1024×78 | 0.962 | 0.935 | 0.972 |
| NC strided 76×2048×32 | 1.005 | 0.960 | 0.956 |
| NT strided 128×128×45 b2048 | 1.079 | 1.057 | 0.979 |
| TN packed 2048×128×111 | 1.085 | 1.035 | 0.954 |
| NN strided 128×256×78 b2048 | 1.091 | 1.040 | 0.954 |
| TN strided 1024×1024×32 | 1.087 | 1.061 | 0.976 |
| NN strided 362×2048×128 | 1.084 | 1.026 | 0.946 |
| control: NN packed 1024³ | 1.030 | 1.010 | 0.981 |
| control: CN packed 1024³ | 0.962 | 0.933 | 0.969 |
| control: CC packed 1024³ | 1.001 | 0.971 | 0.970 |
| control: NN packed 256³ | 0.881 | 0.846 | 0.961 |
| control: NT packed 512³ b2048 | 1.015 | 0.999 | 0.984 |
| control: NN packed 512×512×128 b2048 | 1.031 | 1.010 | 0.980 |

Every cell gains 1.6-7%; none loses. In-binary variant runs agree within 1-2% per cell. cuBLAS still leads on short-k
NN/NT (k = 23-45: 1.06-1.07×), NN packed 2048×362×64 (1.06×), TN 1024×1024×32 (1.06×). The tables do not rank
the row yet, so Auto changes only after the retune.

Tests: `GemmCandidates.Wide128x128EveryFormAtSaturatingBatchAndTileEdges` (WP-O's) runs the cp.async leg on
sm_120. Deliberate breaks, run on sm_120, one temporary build switching them at run time (first red per form):

| break | first red | green before it |
|---|---|---|
| `copy_async_wait<Stages - 1>` | all 9 forms, k = 7 (one slab); TT and CT at 64×300 | TT, CT at 63×65 k = 7 |
| refill buffer `(buf + 1) % Stages` | all 9 forms, k = 17 (3 slabs) | k = 7, 8, 9, 16 (≤ 2 slabs never refill) |
| drop B's pointer bump | all 9 forms, k = 9 (2 slabs) | k = 7, 8 |
| B's half offset dropped (`pb + 0`) | all 9 forms, 63×65 k = 7 | none |
| B's k bound dropped (`k` → 2³⁰) | all 9 forms, k = 7 (NaN) | none |
| A's i bound dropped (`a_lim` → 2³⁰) | none: rows past m feed only discarded accumulators (as WP-O) | all |

The i-edge `src-size` (8 rather than 16) on the 16-byte leg is as untestable as A's i bound: it only stops reads
past m. The accepted `src-size` 0 assumption of WP-N2 holds here too (zero-filled copies may name an address past
the allocation).

### WP-V review: the sm_120 gate, 8-byte bases and the _s3 traces

Changes after review (same machine and protocol):

- **Gate:** cc ≥ 120, the only architecture timed. sm_80-90 keep WP-O's leg unless `BATCHLAS_GEMM_WIDE_STAGES=3`.
- **8-byte bases:** a cfloat view at a 4-byte base faulted on the cp.async leg (`CUDA_ERROR_MISALIGNED_ADDRESS`,
  sticky). The launcher now checks both bases.
- **Traces and knob:** the cp.async legs trace as `_s3`. `BATCHLAS_GEMM_WIDE_STAGES=2` is a test and A/B switch,
  so the register-prefetch leg runs on sm_120 again.
- **Shared header:** the cp.async helpers moved to `src/ops/gemm/kernels/copy_async.hh`, shared with the float
  128x128 pipeline. Registers, stack and spill are identical in all 732 gemm entries on both probes (sm_120 and
  sm_89).
- **Not done:** caching the leg test per device. The other gemm launchers already call `local_mem_bytes` per
  launch, and `cuda_compute_capability` is memoised.

Tests: `Wide128x128EveryFormAtSaturatingBatchAndTileEdges` runs everything twice, once on the derived leg and once
with `BATCHLAS_GEMM_WIDE_STAGES=2`. Each form adds 257×136×33 at a 4-byte base. `PinnedChoiceLaunchesItsOwnKernel`
adds 128x128 at a 4-byte base and padded and Gap1 views under `=2`.

Deliberate breaks, on sm_120, through a temporary run-time switch:

| break | red |
|---|---|
| base check dropped | `CUDA_ERROR_MISALIGNED_ADDRESS` on the first 4-byte-base case, which aborts the process |
| `=2` ignored | PinnedChoiceLaunchesItsOwnKernel: 18 cases (9 forms × the 2 `=2` views) |
| the 16-byte cp.async leg traced without `_s3` | PinnedChoiceLaunchesItsOwnKernel: 45 cases (9 forms × 5 aligned-leg views) |

A's k bound and A's pointer bump have no break of their own. They are the same `acopy` and `issue` code as B's,
whose breaks are above.

Re-measured, the old WP-V build against the fixed one (12 reps × 2 passes, vendor-normalised): NN strided 512³
1.001, TN strided 1024×256×78 1.003, NC strided 256×512×78 1.003, NN packed 2048×362×64 1.003, NN packed 1024³
1.001. In the same binary on NN strided 512³, `=2` gives 1.075× vendor and the default gives 1.005× (0.935, which
matches the 0.934 above).

## GEMM: WP-X, copy-async stages for the transposed wide tiles

Measured 2026-10-20/21 on threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120), one GPU, `batchlas_tune --mode
time` (beta = 1, every arm verified), 10 reps × 2 passes in alternated order (parent, new, `=2`, then reversed).
Ratios are each process's mean over both passes against its own cuBLAS arm. Nothing was timed on another
architecture.

Target: the strided skinny and panel cells `launch_wide_transposed` serves (128x32, 32x128, 64x32, 32x64, 32x32,
16x16) that the shipped sm_120 table still loses at 1.08-1.25× (cfloat NC/NT, float NN/TN/TT).

**What limits them** (ncu, durations at ncu's locked clock; sectors are L1 global-load sectors):

| cell | arm | µs | DRAM % | load sectors | long_scoreboard / issue |
|---|---|---|---|---|---|
| cfloat NC strided 2048×32×32 b128 | `wide:m=128:n=32:k=16` (parent) | 123 | 72.7 | 15.4M | 5.62 |
| | cuBLAS `cutlass_100_simt_cgemm_64x32_16x4_nh_align1` | 98 | 90.1 | 5.7M | 1.79 |
| cfloat NT strided 2048×32×32 b128 | `wide:m=64:n=32:k=16` (parent) | 122 | 73.2 | 17.6M | 6.49 |
| | cuBLAS `cutlass_100_simt_cgemm_64x32_16x4_nt_align1` | 102 | 87.0 | 5.7M | 1.74 |
| cfloat NC strided 256×32×128 b2048 | `wide:m=128:n=32:k=16` (parent) | 631 | 81.1 | 95.9M | 2.91 |
| | cuBLAS `cutlass_100_simt_cgemm_64x32_16x4_nh_align1` | 558 | 92.6 | 32.1M | 0.64 |
| float NN strided 2048×32×64 b2048 | `wide:m=64:n=32:k=16` (parent) | 1682 | 73.9 | 247M | 15.99 |
| | cuBLAS `cutlass_80_simt_sgemm_128x32_8x5_nn_align1` | 1364 | 91.6 | 77M | 1.90 |

Every cell is DRAM-bound: cuBLAS (element-wise `cp.async`, 4-5 stages, 16% occupancy) sits at 87-93% of DRAM, the
parent at 73-81%. Two causes, in order of size:

- **The staged C epilogue.** At ld = m + 1 every group stores C through local memory, two prior loads per thread
  per round trip: 8 serial round trips for a 64x32 complex<float> tile. At k = 32 the C traffic equals the A
  traffic, so half the bytes moved at the latency of one load per lane.
- **Element loads.** The element leg reads a 16-byte granule as VecN loads at a 16-byte lane stride, so each load
  touches twice the sectors it uses (2.1-3.2× cuBLAS's sector count).

**Design.**

| piece | rule |
|---|---|
| epilogue (float, complex<float>, both legs) | `staged_c_store` in one pass and one round trip, as wide v2's (WP-Q): the stage space holds the whole tile, each thread one row and TileN / (Threads / TileM) columns (16; 4 on 16x16), every prior load in flight. Double precision keeps two per trip |
| cp.async leg (complex<float> only) | 2 stages, no register prefetch, one barrier per slab (wide v2's 128x128 ring at Stages = 2). Along i on the 16-byte leg one 16-byte copy per granule (`src-size` cut at the i edge); every other copy one 8-byte element, consecutive lanes on consecutive global addresses (along i, or along k when the operand is k-contiguous). Zero fill by `src-size` 0, so no predicated diamond. Conjugation at the fragment read. Swizzle and fragment reads unchanged |
| gate (launcher, never a key) | DPC++, `cuda_compute_capability() >= 120`, both bases 8-byte aligned, ld ≤ INT_MAX / 128 (int offsets), the 2-stage local memory (can_run's term). `BATCHLAS_GEMM_WIDE_STAGES=2` forces the register prefetch, `=3` admits sm_80+ |
| traces | `<trace>_async` (element copies) or `<trace>_async_vec` (16-byte granules along i), and `<host>_splitk_async[_vec]` for the split-k bodies, which take the same leg |

**Variants** (one build, the leg chosen at run time; ratio to the vendor):

| cell | parent | register prefetch + epilogue | cp.async 2 | 3 stages | 4 stages |
|---|---|---|---|---|---|
| cfloat NC strided 2048×32×32 b128, 128x32 | 1.162 | 0.979 | 0.942 | 1.017 | 1.035 |
| cfloat NT strided 2048×32×32 b128, 64x32 | 1.129 | 0.980 | 0.981 | 1.036 | 1.026 |
| cfloat NC strided 256×32×128 b2048, 64x32 | 1.087 | 1.051 | 1.004 | 1.029 | 1.034 |
| cfloat NC strided 32×2048×128 b128, 32x128 | 1.037 | 0.988 | 0.924 | 0.965 | 0.976 |
| cfloat NC packed 2048×32×64 b128, 128x32 | 0.952 | 0.959 | 0.941 | 1.055 | 1.101 |
| float NT strided 2048×32×128 b2048, 128x32 | 1.107 | 1.040 | 1.068 | 1.074 | 1.068 |
| float TN strided 2048×32×45 b2048, 128x32 | 1.196 | 1.047 | 1.067 | 1.064 | 1.070 |

Three and four stages (60 and 80 KiB on 128x32 complex<float>, 36-48 KiB on 64x32) cut the groups per SM from two
to one, or four to two, and lose 3-10%. Float's 4-byte copies raise the element-leg registers from 72-80 to 96
(128x32, 64x32) and 125-128 (32x32 NN, TT), and the 256-lane tiles drop from three groups per SM to two; float
keeps the register prefetch and takes only the epilogue.

**A/B against the parent build** (ratio to the vendor, mean of both passes; `=2` is the new build with the
register prefetch forced, so new / `=2` is the copy leg alone; float has no copy leg):

| cell (b = batch) | arm | parent | new | `=2` |
|---|---|---|---|---|
| cfloat NC strided 2048×32×32 b128 | 128x32 | 1.174 | **0.958** | 0.983 |
| cfloat NT strided 2048×32×32 b128 | 64x32 | 1.116 | **0.974** | 0.978 |
| cfloat NC strided 2048×32×64 b128 | 64x32 | 1.156 | **0.989** | 1.069 |
| cfloat NC strided 128×32×96 b2048 | 128x32 | 1.110 | **0.965** | 0.963 |
| cfloat NC strided 256×32×32 b2048 | 128x32 | 1.170 | **1.010** | 0.998 |
| cfloat NC strided 32×2048×128 b128 | 32x128 | 1.040 | **0.935** | 0.987 |
| cfloat NC strided 45×32×128 b2048 | 32x32 | 1.076 | **1.005** | 1.045 |
| cfloat NT strided 256×32×128 b2048 | 128x32 | 1.056 | **0.984** | 1.002 |
| cfloat NC strided 64×32×128 b32768 | 64x32 | 1.065 | **0.994** | 1.009 |
| cfloat NC strided 32×256×96 b2048 | 32x64 | 1.058 | **0.991** | 1.004 |
| cfloat CN strided 1024×32×1024 b128 | 128x32 | 0.935 | 0.863 | 0.942 |
| | splitk 128x32 s4 | 1.005 | 0.959 | 1.003 |
| float NN strided 2048×32×64 b2048 | 64x32 | 1.188 | **1.101** | 1.103 |
| float TT strided 362×32×32 b2048 | 64x32 | 1.290 | **1.126** | 1.122 |
| float NT strided 2048×32×128 b2048 | 128x32 | 1.104 | **1.045** | 1.043 |
| float TN strided 2048×32×45 b2048 | 128x32 | 1.202 | **1.051** | 1.052 |
| float NN strided 1448×32×96 b2048 | 64x32 | 1.167 | **1.100** | 1.097 |
| float NT strided 32×1024×256 b2048 | 32x128 | 1.062 | 0.990 | 0.990 |
| control: cfloat NC packed 2048×32×64 b128 | 128x32 | 0.961 | 0.963 | 0.952 |
| control: cfloat TN packed 512×32×512 b128 | 128x32 | 0.777 | 0.784 | 0.773 |
| control: cfloat CC packed 32³ b128 | 16x16 | 0.874 | 0.813 | 0.836 |
| control: cfloat NN packed 48³ b2048 (12 reps) | 16x16 | 0.834 | 0.802 | 0.802 |
| control: cfloat NN packed 48³ b2048 | 32x32 | 1.368 | 1.053 | 1.306 |
| control: cfloat NT packed 32³ b32768 | 16x16 | 0.945 | 0.929 | 0.943 |
| control: cfloat NN strided 48³ b2048 | 16x16 | 1.012 | 0.954 | 1.004 |
| control: cfloat TN packed 64³ b2048 | 32x32 | 1.042 | 1.034 | 1.020 |
| control: cfloat NN packed 96³ b2048 | 32x32 | 0.739 | 0.721 | 0.733 |
| control: cfloat CN packed 32×128×256 b2048 | 32x128 | 0.839 | 0.840 | 0.819 |
| control: float TN packed 1024×32×512 b128 | 128x32 | 1.023 | 1.020 | 1.026 |
| control: float NT strided 32×64×1024 b128 (20 reps) | 32x64 | 1.101 | 1.115 | — |
| | splitk 32x64 s4 | 0.722 | 0.710 | 0.709 |

Passes agree within 2% except the 48³ and 32×64×1024 cells (the vendor arm moves 4-8% between processes; the
32×64×1024 native arm is 2% faster in absolute time, 0.0454 vs 0.0464 ms). Every target cell gains 5.6-19% over the
parent (float NT strided 2048×32×128 the least, 1.104 → 1.045); no control loses beyond vendor drift (float NT
strided 32×64×1024, 1.101 → 1.115, is 2% faster in absolute time) except the noisy 48³ first run (0.816 → 0.853,
not reproduced at 12 reps).
The copy leg alone (new against `=2`) is worth 0-8% on the strided cells and costs up to 2.5% on two packed
compute-heavy controls (CN packed 32×128×256, TN packed 64³), still level with the parent there. The vendor now
leads only on float strided (1.05-1.13×; float NN/TT on 64x32 still issue element loads) and cfloat NC 256×32×32
(1.01).

**After** (ncu, same cells):

| cell | arm | µs | DRAM % | load sectors | registers |
|---|---|---|---|---|---|
| cfloat NC strided 2048×32×32 b128 | new (copy leg) | 101.5 | 87.9 | 7.5M | 102 |
| | `=2` (register prefetch + epilogue) | 103.3 | 86.5 | 15.4M | 128 |
| cfloat NC strided 32×2048×128 b128 | new (copy leg) | 240.8 | 90.3 | 16.1M | 105 |
| | `=2` | 261.5 | 85.7 | 48.0M | 128 |
| | cuBLAS `cutlass_80_simt_cgemm_forwardCompat_32x64_16x4_nh_align1` | 269.7 | 83.6 | 16.1M | 96 |

The copy leg's sector count equals cuBLAS's.

**Registers** (`scripts/register_probe.sh`): zero spill on all 1132 entries of `libbatchlas_sycl` (sm_120) and all
1044 of a `-DBATCHLAS_NVIDIA_ARCH=sm_89` probe tree; no gemm entry has a stack frame. complex<float> copy-leg
bodies: 56-72 (16x16), 88-168 (the others); complex<double> unchanged (up to 186).

**Tests.** `WideTransposedPipelineEveryTileFormAndLegAtSaturatingBatch` runs everything under the derived leg and
under `=2`, and adds per form beta = 0 with NaN in C (m + 1 × 2n − 1 × 33, ld = m + 1) and, for complex, a 4-byte
base (2m + 1 × n + 3 × 37). `SplitKEveryHostEveryFormAtSaturatingBatchAndSliceEdges` runs complex<float> under
both legs. `PinnedChoiceLaunchesItsOwnKernel` checks the `_async[_vec]` and `_splitk_async[_vec]` traces and adds per
tile and split host the `=2` and `=3` views (padded, ld = m + 1) and a 4-byte base. Both pipeline suites
also run under `=3` on sm_80+ (the copy leg's only test below sm_120; on sm_120 it repeats the derived leg). Deliberate breaks, run on sm_120 through a
temporary run-time switch (first red per test; the suites stop at the first fatal failure):

| break | red | green |
|---|---|---|
| `copy_async_wait<Stages - 1>` | 128x32 NT at 256×64×48 b1024 (three slabs), split s2 at k = 300 | — |
| refill the buffer being read (`nb = buf`) | 128x32 NT at k = 48 (pipeline), k = 64 (pinned), split k = 300 | — |
| ConjA dropped at the fragment read | the first CN form of each suite (128x32 CN) | NT, NC, TN before it |
| k bound dropped (`k_rem` → 2³⁰) | the k-tail cases: k = 37 packed, k = 17 NaN-poisoned, split k = 300 | k = 48, 64, 16 |
| i bound dropped (`i_rem` → 2³⁰) | none: rows and columns past m, n feed discarded accumulators | all |
| beta = 0 epilogue reads C | the new NaN-C case, float and complex<float> (128x32 NT) | every beta ≠ 0 case |
| base check dropped | `CUDA_ERROR_MISALIGNED_ADDRESS` on the first 4-byte-base case (process abort) | — |
| `=2` ignored | PinnedChoiceLaunchesItsOwnKernel: the 304 `=2` cases only | the 4-byte-base and derived-leg cases |

Full suites on sm_120 (`BATCHLAS_TEST_BACKEND=CUDA`): `gemm_tests` 158 passed; `tune_tests` 221 passed, 1 skipped
(an LFS pointer in a fresh worktree); `gemm_candidates_tests` 132 passed, 2 failed: `GridCeilingIsACanRunTerm`
for complex<float> and complex<double>, which expects the vendor first at 2×2×2 b65536 while the shipped sm_120
coarse table (7dc92ec3) ranks `small` first. No kernel code is involved.

**Not done.** wide v2's 64x64 tile (the largest remaining complex<float> bucket, 89 coarse cells at 1.06-1.16×:
NT/NC strided 64×2048×128, 512×256×64, NN strided 64×2048×64) has the same element-leg loads and was not
touched. The tables do not rank the change until the retune; the gemm kernel hash moved.

## GEMM: WP-X review, beta zero keeps negative zero

Measured 2026-10-21 on threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120), one GPU, `batchlas_tune --mode
time` (beta = 1, every arm verified), against a snapshot of the 6b4918a5 build, both processes in alternated order.

**beta = 0.** The shared linear epilogues returned alpha · acc + 0 · 0 at beta = 0, so an alpha · acc of -0 was
stored as +0 on every path but the 128x128 predicated staged store (WP-F's own fix). `wide_scalar::lin_epi` (wide
v2, the transposed tiles, the split-k reduction) and `LinearEpilogue::apply` now return alpha · acc exactly.

| helper | rule at beta = 0 (callers pass prior = +0, C unread) |
|---|---|
| `lin_epi`, real | beta → -0, a select on beta alone (hoisted): alpha · acc + (-0 · +0) is alpha · acc |
| `lin_epi`, complex | beta.re → -0, beta.im → +0 in the real part and -0 in the imaginary one: each part adds -0 or subtracts +0 |
| `LinearEpilogue::apply` | `beta == 0 ? alpha * acc : …` (its hot callers already branch on beta, so the select folds) |

At beta ≠ 0 the arithmetic is unchanged. A per-element select in the complex `lin_epi` (and a uniform branch)
cost 2-3% on the complex<float> transposed cells (NT strided 2048×32×32 b128, 64x32: 0.965 → 0.985-1.001 of the
vendor, snapshot against itself 0.963 / 0.964); the hoisted form is level (0.965 / 0.972).

**Test.** `BetaZeroKeepsNegativeZeroOnEveryNativeChoice`: every native choice, scalar and form, A = 0, alpha = -1,
beta = 0, C full of NaN, packed, ld = m + 1 and padded; every element's real part must be -0. Break (both helpers
restored): red on every `wide` and `splitk` choice of every scalar, in every case they run; green on `direct`,
`tiled`, `small` and every `reg`.

**Registers** (`scripts/register_probe.sh`, against the same build with the old helpers): zero spill on both
architectures. sm_120: 830 of 884 gemm entries unchanged, the largest rises complex<double> 64x32 CT element leg
132 → 153 and complex<float> 16x16 CN split body 72 → 80, neither below its previous occupancy step. sm_89: 833
unchanged; wide v2 complex<float> 128x128 249 → 253 (one group per SM either way), double 32x64 and 64x32 TN split
bodies 136-138 → 156 (three groups per SM either way).

**A/B** (ratio to the vendor, mean of two passes, 10 reps; snapshot → this build):

| cell | arm | snapshot | new |
|---|---|---|---|
| cfloat NC strided 2048×32×32 b128 | `wide:m=128:n=32:k=16` | 0.962 | 0.943 |
| cfloat NT strided 2048×32×32 b128 (16 reps) | `wide:m=64:n=32:k=16` | 0.967 / 0.968 | 0.965 / 0.972 |
| float NT strided 2048×32×128 b2048 | `wide:m=128:n=32:k=16` | 1.066 | 1.065 |
| cfloat NT strided 64×2048×128 b128 (16 reps) | `wide:m=64:n=64:k=16` | 1.002 / 1.010 | 0.992 / 0.995 |
| float NN strided 256×256×64 b512 | `wide:m=64:n=64:k=16` | 0.977 | 0.986 |
| double NT strided 2048×32×64 b512 | `wide:m=64:n=32:k=16` | 0.898 | 0.898 |

(Two figures: two processes of each build.) The 256² float cell's passes are 0.980 / 0.973 against 0.987 / 0.986.

**Other review items.** The transposed tiles' 16-byte copy leg traces `_async_vec` (the element leg `_async`). The
one-round epilogue asserts that it takes `staged_c_store`'s row map. The pipeline and split-k suites run under
`BATCHLAS_GEMM_WIDE_STAGES=3` on sm_80+, and `PinnedChoiceLaunchesItsOwnKernel` checks `=3` and the split hosts'
4-byte base.

## GEMM: WP-T, TT, TC and CC on the small transposed tiles

Measured 2026-10-20 on threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120), one GPU, `batchlas_tune
--mode time` (beta = 1, every arm verified), 10 reps, two passes in alternated arm order per cell. Parent: the
WP-N2/WP-Q tree.

Before: the four small tiles took every form but TT, TC and CC, and 128×32 / 32×128 four forms, to save link
time. A real TT (C build CC) and a complex TT, TC or CC therefore fell to `tiled` or `wide:m=64:n=64` (coarse
run it1: double TT strided at batch 32768 1.11-1.23× vendor, cdouble TT/TC/CC 32³ b128 1.65×, 96³ b2048 1.54×,
float TT strided skinny 1.38-1.45×). No kernel code changed: `launch_wide_transposed` already takes any
`OpA, OpB`.

| piece | rule |
|---|---|
| `WideCfg::real` | new mask: the N/C forms a real scalar compiles. `forms` stays the complex set. `wide_compiles<T>` reads one or the other (after WP-O's scalar set); `launch_wide_form`, `launch_wide_cfg`'s `static_assert` and `wide_serves` all go through it |
| real | CC (its TT, TC, CT) on 128×32, 32×128, 32×32, 16×16, 64×32, 32×64 |
| complex | TT, TC, CC on 32×32 and 16×16 only |
| split-k | hosts inherit the row's forms, so `splitk` gains the same forms |

| cell | parent arm | parent / vendor | new arm | new / vendor | new / parent |
|---|---|---|---|---|---|
| double TT strided 128×32×32 b32768 | tiled | 1.19 | wide 128x32 | 0.94 | 0.79 |
| double TT strided 64×32×128 b32768 | tiled | 1.07 | wide 64x32 | 0.89 | 0.83 |
| double TT strided 54×32×64 b32768 | tiled | 1.17 | wide 64x32 | 0.95 | 0.81 |
| double TT strided 38×64×64 b32768 | tiled | 0.83 | wide 16x16 | 0.75 | 0.90 |
| cdouble CC packed 32³ b128 | tiled | 1.70 | wide 16x16 | 0.76 | 0.45 |
| cdouble TT packed 32³ b128 | tiled | 1.69 | wide 16x16 | 0.75 | 0.44 |
| cdouble TC packed 32³ b128 | tiled | 1.69 | wide 16x16 | 0.76 | 0.45 |
| cdouble CC packed 48³ b128 | wide 64x64x16 | 1.14 | wide 16x16 | 0.60 | 0.52 |
| cdouble CC strided 96³ b2048 | wide 64x64x8 | 1.55 | wide 32x32 | 0.94 | 0.61 |
| cdouble TT strided 96³ b2048 | wide 64x64x8 | 1.55 | wide 32x32 | 0.96 | 0.62 |
| cdouble TC packed 96³ b2048 | wide 64x64x8 | 1.55 | wide 32x32 | 0.93 | 0.60 |
| float TT strided 32×256×1024 b2048 | wide 64x64x16 | 1.42 | wide 32x128 | 1.00 | 0.70 |
| float TT strided 32×91×1024 b128 | wide 64x64x16 | 1.45 | splitk 32x128 s4 | 0.74 | 0.51 |
| float TT strided 2048×32×2048 b128 | wide 64x64x16 | 1.27 | wide 64x32, splitk 128x32 s2 | 1.00 | 0.79 |
| float TT strided 128×32×96 b32768 | wide 64x64x16 | 1.18 | wide 64x32 | 1.00 | 0.85 |
| bracket: double TT packed 256²×8 b128 | wide 64x64x8 | 0.53 | wide 64x32 | 0.89 | 1.68 |
| cfloat CC packed 32³ b128 | wide 64x64x16 | 1.01 | wide 16x16 | 0.84 | 0.83 |
| cfloat TT strided 48³ b128 | wide 64x64x16 | 0.76 | wide 16x16 | 0.73 | 0.96 |
| cfloat CC strided 96³ b2048 | wide 64x64x16 | 0.91 | wide 32x32 | 0.81 | 0.88 |
| cfloat TC packed 48³ b32768 | wide 64x64x16 | 0.96 | wide 16x16 | 0.96 | 1.00 |
| saturating: cdouble TT packed 32³ b32768 | small | 0.90 | wide 32x32 | 0.84 | 0.94 |
| cdouble CC strided 160³ b2048 | wide 64x64x16 | 1.28 | wide 32x32 | 0.92 | 0.72 |
| bracket: cdouble CC strided 256³ b2048 | wide 64x64x8 | 0.88 | wide 32x32 | 0.89 | 1.02 |
| bracket: cfloat TT strided 256³ b2048 | wide 64x64x16 | 1.17 | wide 32x32 | 1.44 | 1.23 |
| bracket: float TT strided 256³ b2048 | wide 64x64x16 | 1.43 | wide 64x32 | 1.71 | 1.20 |

Passes agree within 3% except the cdouble 96³ cells (up to 12% between passes; every pass of the new arm is
below the vendor and the parent). The bracket keeps 64×64×8 at a large output. float 32×91×1024 without split-k:
wide 32x128 0.92. The cfloat, saturating and bracket rows below the first bracket are the review re-run
(same box and protocol; passes differ up to 7% on the cdouble 32³ b32768 and 160³ cells, never reordering
the winner against the parent): the cdouble 32³ b128 ratios against `tiled` hold at batch 128 only; at
batch 32768 the parent is `small` and the gain is 6%. cdouble 512³ TT b128: all four tiles within 1% (0.877-0.886).
Each type has one losing cell; the windows between are left to the retune.

**Link** (`scripts/register_probe.sh`, the `libbatchlas_sycl` device link alone, box shared at load 5-6):
parent 81.9 s (767 entries); every form on the four small tiles plus CC on the panel tiles, all scalars,
104.5 s (927, +28%); shipped 91.9-93.6 s (863, +12-14%). The complex grid has no TT/TC/CC panels, so complex
64×32, 32×64, 128×32 and 32×128 stay without them. cfloat keeps its 32×32 / 16×16 TT, TC, CC (about 24 of the
96 new entries): its coarse-grid losses are at 256³ and up, but the cfloat rows above win 4-17% over the parent
below that. The figure is the nvptx64 device link alone: the full-target CPU AOT of the new instances and the
vendor-free tree are not measured here (pre-push gate). Review rebuild (no instance change): 94.3 s at load 8, 863 entries, zero spill.

**Registers:** zero spill on all 863 entries. New instances (both operands transposed): float 38-85, double
68-163, cfloat 16×16 64-80 and 32×32 168, cdouble 16×16 72-86 and 32×32 136-190; the 32×32 and 16×16 tiles
are 64 lanes, two warps, so the four-partition gate (`ceil(2/4)·32·190 = 6080 ≤ 16384`) holds.

**Tests:** the oracle rows in `tests/gemm_candidates_tests.cc` gain the forms (`WideRow::real_only` for the CC a
real scalar alone compiles), so the existing every-form suites run them: saturating batch 1024 bit-identical,
ld = m + 1, NaN gaps and NaN C at beta = 0, complex alpha, beta ≠ 0, split-k slice edges, tile edges ±1.
`gemm_tests` `SmallWideTilesRunTheirTransposedForms` runs all eight transposed pairs and checks the 12 complex
refusals; the panel-tile own-build tests run a real TT. Deliberate breaks, run on the GPU:

- complex TT on wide 32×32 / 16×16 launched as TC: red exactly `PinnedCandidatesEveryFormShapeAndLayout` and
  `WideTransposedPipelineEveryTileFormAndLegAtSaturatingBatch` for complex<float> and complex<double>.
- real CC dropped from 64×32 (oracle unchanged): red 6 tests each for float and double (pinned every-form,
  pinned bit-for-bit, `CanRunEqualsLaunch`, skinny, split-k, pipeline), all on 64×32 TT, TC, CT, CC.

The sm_120 and sm_89 tables do not rank the new forms until the retune; the gemm kernel hash moved.
## GEMM: WP-F, float mid-size parity on sm_120

Measured 2026-10-20 on threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs), one GPU, `batchlas_tune
--mode time` (beta = 1, every arm verified), two passes per build in alternated order (parent, new, new reversed,
parent reversed); 6 reps on the target cells, 4 on the screen. Ratios are against cuBLAS, both builds' vendor
arms pooled. Parent: `f4d63663` (WP-N2 and WP-Q).

Target: the coarse race left ~600 float cells at m, n ≥ 128 losing 1.08-1.38× (median 1.13), served by
`wide:m=64:n=64:k=16` and `reg:m=128:n=128:k=8:u=1/u=4`, mostly `layout=strided` (odd ld) at k = 64-128.

**ncu first** (`--set full`, one launch each, parent build):

| cell (b128 unless noted) | arm | kernel | ms | DRAM % | IPC | regs, occ | global ld / st sectors (M) | top stall |
|---|---|---|---|---|---|---|---|---|
| NN strided 256×512×78 b2048 | vendor | `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` | 1.819 | 92.3 | 1.63 | 210, 16.7% | 85 / 41 | barrier |
| | u4 | `..._k8_s4` (direct epilogue) | 2.415 | 70.6 | 1.33 | 127, 33% | 300 / 151 | long scoreboard |
| TN strided 2048×1024×96 | vendor | `..._256x128_8x4_tn_align1` | 2.093 | 70.0 | 1.65 | 212, 16.7% | 112 / 41 | MIO throttle |
| | wide 64×64 | wide v2 element leg | 2.862 | 53.5 | 1.48 | 80, 49% | 544 / 41 | MIO throttle |
| TT strided 1024×2048×96 | vendor | `..._128x256_8x4_tt_align1` | 1.834 | 82.9 | 2.07 | 212, 16.7% | 95 / 41 | |
| | u4 | `..._k8_s4_tt` (direct, k > 64) | 2.596 | 58.5 | 1.47 | 127, 32% | 333 / 151 | long scoreboard |
| NN packed 215×2048×96 | vendor | `..._128x64_8x5_nn_align1` (128 lanes) | 0.422 | 78.7 | 2.41 | 128, 25% | 28 / 8.7 | |
| | wide 64×64 | wide v2 16-byte leg | 0.550 | 59.9 | 1.84 | 80, 48% | 104 / 8.7 | short scoreboard |
| TN packed 512×256×78 b2048 | vendor | `..._128x256_8x4_tn_align1` | 1.827 | 93.6 | 1.67 | 210, 16.7% | 88 / 34 | |
| | wide 64×64 | wide v2 (k % 16 ≠ 0) | 2.214 | 81.7 | 1.75 | 80, 49% | 445 / 34 | MIO throttle |
| NN packed 512³ b2048 | vendor | `..._256x128_8x4_nn_align1` | 13.16 | 45.8 | 3.00 | 210, 16.7% | | selected |
| | u1 | aligned leg | 14.39 | 41.5 | 2.45 | 139, 16.6% | | long scoreboard |

cuBLAS runs CUTLASS SIMT kernels (`align1`: 4-byte `cp.async` at any ld, 4 or 5 stages, k = 8, one 256-lane
group per SM). Both sides move the same DRAM bytes (e.g. 2.60 GB at 256×512×78); the cells are DRAM-bound and
the loss is how much of the run DRAM is busy. Two defects of ours, neither the tile size:

- **The predicated 128×128 leg's direct epilogue** (k > 64, and u4 NN at every k): scalar stores with lanes four
  rows apart, 3.7× the store sectors and L2 write traffic of a whole-column store (151M against 41M).
- **u4's k-contiguous operand** (B in NN and TN, Aᵀ in TN and TT): two lanes per column, four 4-byte copies
  each, so a warp copy touched 16 rows' sectors for 128 useful bytes (343M load sectors against 112M, LSU/MIO
  throttle).

**Shipped** (`register_128x128.hh`; no new spelling, row or key, so the next race picks it up):

| piece | rule |
|---|---|
| epilogue | the predicated leg (u1 and u4, every form) stages C at every k; the aligned leg keeps `kStagedEpilogueMaxK = 64`. The predicated direct instances are no longer compiled |
| priors | per pass, the 16 C priors are loaded after the stage write (its accumulators are dead) and before the barrier, all in flight; none at beta = 0 |
| u4 lane map | a k-contiguous operand takes k fastest: 8 lanes per 32-byte row segment, 4 rows per copy, 32 rows apart (`lda32`); out of range iff the k index or the row is |
| u4 shared stride | 132 for both operands (4 pad), so the k-fastest transposed writes are conflict-free; local memory 33,792 B (`RegCfg::slm_bytes`) |
| u4 staged NN | fits at 128 registers once `tid` is re-read after the copies (`mov %tid.x` in `asm volatile`); with the epilogue's addresses hoisted into the mainloop it spilled 20 B |

Staging alone (priors read inline) fixed the sectors (41M) but not the time: 2.614 ms, DRAM 59%, long
scoreboard on each prior. Batching the priors: 1.923 ms, DRAM 91.8% (vendor 1.819, 92.3%).

| cell (b128 unless noted) | parent best | / vendor | new best | / vendor |
|---|---|---|---|---|
| NN strided 256×512×78 b2048 | u4 | 1.345 | u4 | 1.081 |
| TN strided 2048×1024×96 | wide 64×64 / u4 | 1.361 | u4 | 1.067 |
| TT strided 1024×2048×96 | u4 | 1.347 | u4 | 1.130 |
| NT strided 362×1024×96 | u4 | 1.171 | u4 | 0.994 |
| NN packed 215×2048×96 | u4 | 1.182 | u4 | 0.984 |
| TN packed 512×256×78 b2048 | u4 / wide 64×64 | 1.266 | u4 | 1.054 |
| control, NN packed 512³ b2048 (aligned, untouched) | u1 | 1.076 | u1 | 1.063 |
| control, NT packed 1024³ (aligned, untouched) | u1 | 1.000 | u1 | 0.972 |

Each step on its own (6 reps): the epilogue took NN strided 256×512×78 u4 from 1.342 to 1.122 and u1 from
1.461 to 1.207; the lane map then took TN strided 2048×1024×96 from 1.179 to 1.067 and TN packed 512×256×78 from
1.124 to 1.054 (NN 1.088 → 1.081, TT 1.136 → 1.130: one k-contiguous operand). The controls run unchanged code
and move within the ~3% noise of this shared GPU.

Screen, 28 losing cells drawn from the coarse race (m, n ≥ 128, k ≥ 32, at most 3 per form × layout × arm;
`sample.txt` in the job dir): median best native / vendor 1.110 → 0.954, geomean new / parent 0.854; 5 cells
still above 1.05: NN strided 1024×128×78 b2048 1.081, NT packed 2048×181×128 1.119 and NN strided 512×181×96
1.105 (n = 181 leaves the second 128 tile 41% full), TT/TN strided k = 128 1.09, TN packed 1024×2048×64 1.057
(aligned leg). In 25 of 28 the new best is u4, so the float strided and k % 8 ≠ 0 rows need the race.

**ncu after** (new build): TT strided 1024×2048×96 u4 2.018 ms, DRAM 77% (vendor 1.834, 83%), load sectors 119M
(vendor 95M), IPC 2.11; TN strided 2048×1024×96 u4 2.177 ms, DRAM 69.5% (vendor 2.093, 70.0%), load sectors 135M (342M before the lane map, vendor 112M), shared bank conflicts 38M (146M before; vendor 195M), IPC 1.76.

**Negative:** u4's pipeline on aligned (packed) data too, instead of u1's register prefetch: NN 512³ b2048 1.146
against u1's 1.070, NT 1024³ 1.121 against 1.028, TT 512³ 1.192 against 1.103; a win only on TN packed
1024×2048×64 (1.023 against 1.056). Reverted: the aligned leg stays u1's kernel.

**Registers** (`scripts/register_probe.sh`, sm_120): zero spill on every entry of `libbatchlas_sycl`; the 4 u4
instances (all staged) 123, 127, 127, 128; u1 predicated staged 126-133. NT at 133 matches the parent's staged NT
(k ≤ 64), but at k > 64 the parent ran direct at 128: one group per SM instead of two there (its wrapper has no
`.reqntid`, known defect #20; measured below). Link of `libbatchlas_sycl`: 81 s, parent 81 s.

Tests: `GemmCandidates.Reg128x128PredicatedStagedEpilogueAtEveryK` (u1 and u4, 5 forms, k = 64, 65, 96, 137 on
300×200 and 256×256 at Gap1, Strided and OddLd, NaN poison on the one-element k tails, batch 1024 repeating 7
problems at beta ≠ 0, then NaN C at beta = 0). The WP-I, WP-N2 and WP-B suites now run the staged path at k > 64.
Deliberate breaks, run (filter: float `Reg128x128*` and `CoalescedCEpilogue*`):

| break | red | green |
|---|---|---|
| store uses the neighbour pass's prior (`prior[q ^ 1]`) | every predicated beta ≠ 0 case, first at u1 NN 200×150×37 | the padded aligned legs |
| priors read at beta = 0 | the three tests with NaN C on a predicated leg | the other three |
| u4 k-fastest copy steps 31 rows (`lda32 - lda`) | four tests, first at u4 TN in each (NN, NT pass first) | the u1-only and aligned tests |
| u4 Aᵀ copy drops its k bound | the three NaN-poison k-tail tests, first at u4 TN 256×256×37, ×5 and 300×200×65 | the other three |
| beta = 0 stores `alpha * acc + 0 * 0` (review fix reverted) | `Reg128x128BetaZeroKeepsNegativeZero`, 11 u4 cases (+0.0 where reference BLAS gives -0.0) | u1 (its codegen kept -0.0 either way), every other test |

The u4 copy's row bound is not covered by values: rows past m (n) feed only C rows that are never stored, so
dropping it reads out of bounds with right answers.

Every ratio above is a pinned arm. Auto still ranks arms by the parent's timings (`tuned/gemm.sm_120.txt`, whose
gemm kernel hash is now stale) and keeps choosing `wide:m=64:n=64:k=16` or `vendor` on most of these cells until
the float race is re-run and gated; none of these numbers is an Auto result yet.

### WP-F review: long k and the u1 NT occupancy

The predicated leg stages at every k, but the first measurements stopped at k = 137. Strided 512×512, batch 256,
12 reps where a 6-rep pass was ambiguous; best of parent / new over vendor:

| form, k | u4 parent → new | u1 parent → new |
|---|---|---|
| NN 256 | 1.053 → 0.926 | 1.160 → 1.028 |
| NN 512 | 1.241 → 1.158 | 1.355 → 1.270 |
| NN 1024 (12 reps) | 1.230 → 1.177 | 1.347 → 1.294 |
| NN 2048 | 1.201 → 1.167 | 1.311 → 1.285 |
| NT 256 | 1.019 → 0.952 | 1.033 → 0.954 |
| NT 512 | 0.983 → 0.954 | 0.985 → 0.932 |
| NT 1024 | 1.087 → 1.090 | 1.084 → 1.053 |
| NT 2048 (12 reps) | 1.079 → 1.084 | 1.067 → 1.044 |
| TN 256 | 1.284 → 1.089 | 1.597 → 1.473 |
| TN 512 | 1.235 → 1.094 | 1.518 → 1.477 |
| TN 1024 (12 reps) | 1.230 → 1.116 | 1.507 → 1.531 |
| TN 2048 (12 reps) | 1.206 → 1.111 | 1.471 → 1.527 |

No k ceiling for the predicated leg: staging wins or ties (u4 NT, within 0.5%) everywhere except u1 TN at
k ≥ 1024 (1.6% and 3.8% slower, reproduced). There u1 is 1.5× vendor and 1.37× u4 in the same build, so no race
ranks it; keeping a direct u1 TN instance for it would add a compiled kernel for a route nothing picks.

u1 NT predicated at k > 64 went from the direct epilogue at 128 registers (two groups per SM) to the staged one at
133 (one group per SM: 2 warps × 32 × 133 > 16,384 / 2 per sub-partition; no `.reqntid`, known defect #20).
It is faster anyway: 362×1024×96 b128 1.235 → 1.048, 1024×1024×128 b128 1.114 → 0.976, and every NT row above.

beta = 0 on the predicated staged store is `alpha * acc` alone (the aligned leg already was): `alpha * acc + 0 * 0`
turned a -0.0 into +0.0 (alpha = -1, A = 0). Registers unchanged (same counts per instance, zero spill).

sm_89 (found at the WP-O/WP-T/WP-F merge): the k-fastest copies' `32 · ld` stride, hoisted out of the mainloop
and held in a register, spilled 16 B (12 B loads) in staged u4 NN, TN and TT at 128 registers; NT, which copies
neither operand k-fastest, sat at 125 with none. `issue` now rebuilds `32 · ld` per call behind an empty
`asm volatile`: sm_89 u4 125-127 registers, zero spill on both arches. Not re-timed.

## GEMM: WP-U, the u1 128x128 launch bound enforced

`Gemm128x128Body` and `Gemm128x128Inlined` (u1, every leg) now carry `[[sycl::reqd_work_group_size(1, 1, 256)]]`
like the u4 pipeline wrapper already did, so ptxas sees `.reqntid` and holds the two-group cap (known defect #20;
`minnctapersm is ignored` count in `libbatchlas_sycl`: 0). The aligned staged epilogue re-reads `%tid.x` behind
`asm volatile` as the u4 one does: without it u1 aligned staged TT spilled 32 B at the cap on sm_120.
Float only: the `reg` family has no complex<float> or double instance (all 16 `GemmRegister128x128Kernel` entries
in the probe are float; the complex<float> 128x128 at the cap is wide v2's WP-O tile, a different kernel).

**Registers** (`scripts/register_probe.sh`, `libbatchlas_sycl`, zero spill on every entry on sm_120 and sm_89):
u1 sm_120 parent → new, aligned direct NN 139 → 128, aligned staged NN 135 → 128, TN 130 → 128, TT 139 → 128,
predicated staged NT 133 → 128; the other u1 instances were 126-128 and are now 118-128 (aligned staged NT
128 → 118, aligned direct TN 126 → 128, predicated staged TT 126 → 127). u4 unchanged. sm_89 u1 127-128: untimed,
and unrouted (no `tuned/gemm.sm_89.txt` row ranks u1 128x128 first; only pins and vendor-free builds reach it).

**Timing** (RTX PRO 6000 Blackwell Max-Q, sm_120, float, pinned `reg:m=128:n=128:k=8:u=1`, parent and new builds
side by side, 8 reps × 2 passes in swapped order per build, three runs; new/parent is the range over the runs,
×vendor is the third run):

| cell (batch) | new / parent | parent → new ×vendor |
|---|---|---|
| NN packed 1024×2048×64 (128) | 0.853-0.874 | 0.862 → 0.756 |
| TT packed 512×256×64 (2048) | 0.902-0.909 | 0.873 → 0.787 |
| TN packed 2048×1024×64 (128) | 0.913 | 1.058 → 0.967 |
| NT strided 362×1024×96 (128) | 0.914-0.921 | 1.091 → 0.989 |
| NN packed 1024×2048×96 (128) | 0.928-0.931 | 0.781 → 0.725 |
| TN packed 1024×128×64 (2048) | 0.936-0.939 | 1.040 → 0.982 |
| NN packed 512×256×128 (2048) | 0.942-0.949 | 1.073 → 1.013 |
| NN packed 512×512×256 (128) | 0.934-0.948 | 0.910 → 0.866 |
| NT packed 2048×181×128 (128) | 0.961-0.965 | 1.184 → 1.125 |
| NT packed 1024³, TT packed 512³, NN packed 512³ (2048), NN strided 256×512×78 (2048), TN strided 2048×1024×96, NN packed 215×2048×96, TT packed 1024×2048×128 | 0.993-1.011 | within 1% |
| NN packed 512×512×78 (128) | 1.018-1.024 | 0.861 → 0.876 |
| NN strided 512×512×1024 (256) | 1.019-1.024 | 1.292 → 1.325 |
| NN packed 1024³ (128) | 1.025-1.030 | 1.059 → 1.087 |
| NN packed 2048³ (16) | 1.026-1.030 | 1.066 → 1.100 |
| TT packed 512×362×8 (128) | 1.000-1.063 | 0.864 → 0.899 |

Mid k (64-256) gains 4-15%; NN at k ≥ 1024 loses 2-3%, where the 139-register aligned direct NN mainloop ran
better at one group per SM. TT 512×362×8 runs the predicated staged TT instance, whose registers did not move
(126 → 127): two 40-rep passes gave 1.000 and 1.063, noise at 63 µs. Shipped. Tables not re-raced.

**The k edge** (NN packed 512×512, batch 128, the size whose largest-k table row, k = 512, ranks u1 first; same
harness, 12 reps × 2 swapped passes, three runs, new / parent):

| k | 256 | 384 | 512 | 768 | 1024 |
|---|---|---|---|---|---|
| new / parent | 0.884-0.931 | 0.943-0.978 | 0.963-0.996 | 1.006-1.018 | 1.020-1.035 |

The win turns into a loss between k = 512 and 768. Auto (`BATCHLAS_SELECT_TRACE=1`) sends 512×512×{512, 768, 1024}
batch 128 to u1: no larger-k row exists at that size, so k = 768 and 1024 borrow the k = 512 row, and there u1
trails vendor in both builds (new 1.01-1.06× vendor). Auto sends NN packed 1024³ and 1024×1024×768 (128), 2048³ (16),
strided 512×512×1024 (256) and packed 512×512×78 (128) to vendor, so the other losses above reach only pins and
vendor-free builds. Parent build for every run: `libbatchlas_sycl.so` md5 `1a5b2f30…`, source `2c59144c`.

**Retune follow-up.** `register_128x128.hh` is in the kernel hash of gemm, geqrf, getrf, gesv, ormqr, orgqr,
gesvd, symm, trmm and syev (`tools/tune/*_spec.cc`); all ten tables are hash-stale, and the non-gemm ones time u1
128x128 gemm sub-calls with the parent kernel. Three rows the stale gemm table sends to vendor now favour u1: TN
packed 2048×1024×64 (128) 0.967× vendor, NT strided 362×1024×96 (128) 0.989×, TN packed 1024×128×64 (2048) 0.982×.
A re-race should also add a k > 512 row at 512×512 batch 128.

## GEMM: WP-W, a 256x128 float tile with four copy-async stages

Measured 2026-10-20 on threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs), one GPU, `batchlas_tune
--mode time` (beta = 1, every arm verified), 8 reps × 2 passes in alternated arm order, all arms in one binary;
ratios are against that run's cuBLAS arm. Parent: `7dc92ec3` (the coarse sm_120 table).

Target: the largest float loss bucket of the shipped table, 95 cells at m, n ≥ 128 on `reg:m=128:n=128:k=8:u=4`
at 1.05-1.17× vendor (13.3 ms excess of 23.9), mostly TT/TN/NN strided at k = 64-362. On these shapes cuBLAS
runs `cutlass_80_simt_sgemm_256x128_8x4_{nn,tn}_align1` or `..._128x256_8x4_tt_align1` (WP-F's ncu).

New choices `reg:m=256:n=128:k=8:u=4` and `reg:m=128:n=256:k=8:u=4` (float, NN NT TN TT; `register_256x128.hh`):

| piece | rule |
|---|---|
| tile | 256 lanes, one group per SM (`reqd_work_group_size` + bound 1), 16×8 (8×16) accumulators as 4-wide bands, the 128x128 kernel's lane swizzle on a 2 × 4 warp grid |
| staging | 4 slabs of k = 8, 3 in flight; one zero-filled 4-byte `cp.async` per element (any ld, any base), u4's lane maps (packet operand m/n-fastest, k-contiguous operand k-fastest), strides padded by 4: 50,176 B (`RegCfg::slm_bytes`, the >48 KiB opt-in; `can_run` refuses a smaller budget) |
| loop | step s: `wait_group 2`, barrier, issue slab s + 3, commit, compute. A slab past k is issued anyway (every copy `src-size` 0), so no branch guards the refill |
| 16-byte leg | derived in `launch_register_k8_s4_large`, never a field: when every packet operand (A in NN/NT, Bᵀ in NT/TT) has a 16-byte base and ld, stride ≡ 0 mod 4, it copies 16-byte granules (edge granule clipped by `src-size`); TN has no packet operand. Traces `…_s4[_nt\|_tn\|_tt]` and `…_aligned` |
| epilogue | staged at every k: 4 passes of TileN / 4 columns, whole-column stores; pass p + 1's 32 priors load while pass p stores; none at beta = 0, where the store is `alpha * acc`. Stage stride TileM + 16 (≡ 16 mod 32): the two column slots of an 8-lane `STS.128` phase take distinct banks |

**ncu** (`--set full`, TT strided 1024×1024×362 b128, durations at ncu's locked clock):

| arm | ms | IPC | issue busy | regs, occ | global ld sectors | shared ld wavefronts | top stalls per issue (beyond selected) |
|---|---|---|---|---|---|---|---|
| cuBLAS `128x256_8x4_tt_align1` | 2.24 | 2.80 | 68% | 212, 16.7% | 123M | 149M | not_selected 0.79, dispatch 0.21, mio 0.19, barrier 0.18, long_sb 0.15, wait 0.06 |
| u4 (parent best) | 2.71 | 2.39 | 59% | 127, 33% | 168M | 198M | not_selected 2.18, dispatch 0.80, barrier 0.66, mio 0.63, long_sb 0.47, lg 0.37 |
| 128×256, first version | 2.43 | 2.55 | 62% | 253, 16.7% | 123M | 149M | not_selected 0.83, wait 0.28, dispatch 0.26, long_sb 0.24, lg 0.10 |
| 128×256, shipped | 2.41 | 2.53 | | 214, 16.7% | 123M | 149M | not_selected 0.86, dispatch 0.32, lg 0.21, mio 0.17, wait 0.17, long_sb 0.12 |

The new tile moves exactly cuBLAS's global sectors, shared wavefronts and instruction count (1.76-1.79 G
each). Both spend ~13% of their samples after the mainloop. The gap left is issue: `wait` and `lg_throttle` on
the 12 back-to-back `LDGSTS` per step, the address and `src-size` (`ISETP`/`SEL`) chain feeding them.

**Variants** (same harness, run-time switch in one binary, 6 reps × 2 passes, ×vendor):

| variant | TT str 1024²×362 | NN str 512×256×128 b2048 | TN str 1448×256×128 | NN packed 1024³ | TT packed 512³ |
|---|---|---|---|---|---|
| copies at step start, priors after the stage write | 1.048 | 1.019 | 1.029 | 1.033 | 1.052 |
| copies spread over the 8 k steps (CUTLASS's order) | 1.060 | 1.043 | 1.005 | 1.087 | 1.073 |
| copies at step start, priors one pass ahead (**shipped**) | 1.043 | 1.015 | 1.007 | 1.034 | 1.052 |
| copies spread, priors ahead | 1.053 | 1.028 | 0.989 | 1.083 | 1.072 |
| priors ahead, copies in 2 / 4 / 8 groups | 1.058 / 1.054 / 1.055 | 1.020 / 1.020 / 1.029 | 0.988 / 0.984 / 0.981 | 1.044 / 1.052 / 1.083 | 1.063 / 1.068 / 1.070 |

Spreading the copies moves the stall from `lg_throttle` to `mio_throttle` (ncu: 0.36, the copies queue behind
the fragment reads) and loses everywhere but TN 1448×256×128. The 16-byte leg against the element leg on the
same aligned views (8 reps): NT packed 1024³ 0.971 vs 0.993, TT packed 512³ 1.045 vs 1.057, NN packed 1024³
1.033 vs 1.041, NN packed 1024×2048×96 0.725 vs 0.727, TT packed 1024²×362 1.036 vs 1.039: shipped.

Stage stride (review): at TileM + 4 (≡ 4 mod 32) the second column slot of each 8-lane `STS.128` phase overlapped
the first in 12 banks. TileM + 16 (32 × 272 and 64 × 144 floats, within the 12,544 of the slabs), ncu on 256×128
TT strided 1024²×362: shared-store bank conflicts 4,194,304 → 0, store wavefronts 8,388,608 → 4,194,304; time
flat (±0.2% on the four review cells, 10 reps × 2 passes): shipped.

**A/B, shipped build** (best of u1/u4 against best of the two new tiles, each ×vendor in the same run):

| cell (b128 unless noted) | vendor ms | best before | ×vendor | best new | ×vendor (pass 1, 2) | new / before |
|---|---|---|---|---|---|---|
| TT strided 1024×1024×362 | 2.815 | u4 | 1.159 | 128×256 | 1.035 (1.034, 1.036) | 0.893 |
| TT packed 1024×1024×362 | 2.786 | u4 | 1.160 | 128×256 | 1.024 (1.021, 1.026) | 0.883 |
| TN packed 1024×1024×362 | 3.050 | u4 | 1.089 | 256×128 | 0.947 (0.946, 0.949) | 0.870 |
| TT strided 1024×2048×128 | 2.457 | u4 | 1.129 | 128×256 | 1.037 (1.036, 1.038) | 0.919 |
| TT strided 512×256×128 b2048 | 2.613 | u4 | 1.139 | 128×256 | 1.037 (1.036, 1.037) | 0.910 |
| NN strided 1024×128×128 b2048 | 2.795 | u4 | 1.123 | 256×128 | 1.017 (1.017, 1.016) | 0.906 |
| NN strided 512×256×128 b2048 | 2.711 | u4 | 1.119 | 256×128 | 1.014 (1.012, 1.015) | 0.906 |
| TN strided 1448×256×128 | 0.513 | u4 | 1.074 | 256×128 | 0.999 (1.007, 0.991) | 0.930 |
| TN strided 1024×2048×128 | 2.820 | u4 | 1.083 | 256×128 | 0.973 (0.971, 0.975) | 0.898 |
| NT strided 2048×1024×128 | 2.594 | u4 | 1.070 | 256×128 | 1.019 (1.018, 1.020) | 0.952 |
| NN strided 256×256×181 b2048 | 1.814 | u4 | 1.133 | 256×128 | 1.027 (1.028, 1.027) | 0.906 |
| NN strided 1024×128×96 b2048 | 2.446 | u4 | 1.102 | 256×128 | 1.012 (1.013, 1.011) | 0.918 |
| TN strided 181×512×128 b2048 | 2.610 | u4 | 1.089 | 256×128 | 0.963 (0.965, 0.962) | 0.884 |
| NN strided 512×181×128 b2048 | 2.167 | u4 | 1.084 | 256×128 | 1.004 (1.005, 1.003) | 0.926 |
| TT strided 256×512×128 b2048 | 2.881 | u4 | 1.072 | 128×256 | 0.982 (0.983, 0.981) | 0.916 |
| NN strided 256×512×78 b2048 | 2.181 | u4 | 1.056 | 256×128 | 0.978 (1.002, 0.955) | 0.926 |
| NN packed 1024×1024×1024 | 6.400 | u1 | 1.068 | 256×128 | 1.024 (1.024, 1.024) | 0.959 |
| NT packed 1024×1024×1024 | 6.505 | u1 | 0.983 | 256×128 | 0.967 (0.969, 0.966) | 0.984 |
| TT packed 512×512×512 | 1.013 | u4 | 0.932 | 128×256 | 0.958 (0.898, 1.025) | 1.028 |
| NN packed 1024×2048×96 | 2.853 | u1 | 0.704 | 256×128 | 0.720 (0.718, 0.723) | 1.023 |
| NN packed 512×512×512 b2048 | 15.310 | u1 | 1.065 | 256×128 | 1.028 (1.030, 1.027) | 0.965 |
| TN packed 2048×1024×64 | 1.762 | u1 | 0.960 | 256×128 | 0.983 (0.981, 0.985) | 1.024 |
| NT strided 362×1024×96 | 0.413 | u4 | 0.984 | 128×256 | 0.987 (0.982, 0.991) | 1.003 |
| NN strided 512×512×1024 b256 | 3.713 | u4 | 1.184 | 256×128 | 1.028 (1.031, 1.026) | 0.868 |

15 of the 16 strided cells gain 5-13% (median 0.91 of the parent's best); NT strided 362×1024×96 is flat (1.003).
4 strided cells beat cuBLAS in both passes: TN 1024×2048×128, TN 181×512×128, TT 256×512×128 and NT 362×1024×96
(where u4 already did); TN 1448×256×128 (1.007, 0.991) and NN 256×512×78 (1.002, 0.955) straddle 1. NT strided
2048×1024×128 gains least among the gainers (0.952): NT has no k-contiguous operand, where u4 lost least. On aligned
cells the new tiles win at k ≥ 512 (NN 1024³ 0.959, 512³ b2048 0.965) and lose 2-3% at k ≤ 96 and on TT 512³
(noisy, the vendor arm moved 0.90 ↔ 1.03 between passes) to u1, which stays a candidate: nothing about u1 or
u4 changed, so those cells keep their arm once raced. 128×256 serves TT best, 256×128 NN and TN; on n = 128
the 128×256 tile is half empty (1.44-1.48×), on m = 362 the 256×128 one (1.14×).

**Registers** (`scripts/register_probe.sh`, `libbatchlas_sycl`, zero spill and zero stack on all 14 entries,
both arches): sm_120 256×128 255 (all 7), 128×256 210-214; sm_89 255 (7) and 212-218 (7). One group per
SM: ⌈8 warps / 4⌉ × 32 × 256 = 16,384 ≤ 16,384 per sub-partition, zero margin (ptxas allocates in units of 8, so
255 counts as 256). Holding the priors one pass ahead in a single buffer (load after the store, `prior[CW][RW]`)
left 256×128 at 255 (128×256 203-209) and was 0.1-1.5% slower on the four review cells: rejected.

Tests: `GemmCandidates.RegLargeTilesEveryFormSlabCountAndLegAtSaturatingBatch` (both tiles, NN NT TN TT CC,
m/n straddling 256/128 at 255-300, k from 1 to 17 slabs including 3 = the prologue alone, Strided / Gap1 /
OddLd / ItemSkew element legs and Padded / Packed 16-byte legs, NaN-poisoned k tails, batch 1024 repeating 7
problems, then NaN C at beta = 0; 160 cases) and `RegLargeTilesBetaZeroAndTheDerivedLeg` (-0.0 at beta = 0 on
both legs; the leg taken per (A, B) layout pair, and at base + 1 element with whole-granule ld and stride, by
kernel trace, then its values). The generic suites
(`PinnedCandidatesEveryFormShapeAndLayout`, `PinnedChoiceLaunchesItsOwnKernel`, `BetaZeroDoesNotReadC`,
`CoalescedCEpilogue…`, `CanRunEqualsLaunch`) pick the rows up from `reg_rows`.

Deliberate breaks, run on sm_120 through a temporary run-time switch (filter: the two new tests plus the four
generic suites above; first red in the new slab test):

| break | red tests | first red in the slab test |
|---|---|---|
| `copy_async_wait<Stages - 1>` (compute before the slab lands) | slab test, `PinnedCandidates…`, `CoalescedCEpilogue…` | NN 256×128×5 (one slab) |
| refill buffer `(b + 1) % Stages` | all six | NN 257×129×9 (2 slabs; k = 5, 8 green) |
| k-contiguous operand's k bound dropped (`left = lim`) | slab test, `PinnedCandidates…`, `BetaZeroDoesNotReadC` | NN 256×128×5 under NaN poison |
| A's pointer bump dropped | all six | NN 257×129×9 (2 slabs) |
| store uses the other pass's priors (`prior[(p + 1) & 1]`) | slab test, leg test, `PinnedCandidates…`, `CoalescedCEpilogue…` | NN 256×128×5 |
| priors read at beta = 0 | none (the beta = 0 store never reads them) | — |
| beta = 0 stores `alpha * acc + 0 * 0` | leg test only (180,000 of 180,000 elements +0.0) | — |
| 16-byte edge clip dropped (`src-size` 16) | none (rows past m, n are discarded) | — |
| leg derivation ignores Bᵀ's alignment | all six; the leg test's trace part names `…_nt_aligned` where `…_nt` was expected | TT 256×128×5 |
| 16-byte leg's k bound dropped | slab test only (after the fix below) | NN Padded 264×136×45 under NaN poison |
| store guard `col >= n` dropped (writes past n) | `PinnedCandidates…`, `BetaZeroDoesNotReadC` (wrote outside C), then the process aborts (CUDA out of resources) in the slab test | — |
| leg derivation ignores the base (review) | leg test only: `…_aligned` for the 6 base + 1 cases with a packet operand (NN, NT, TT of both tiles), then NN 256×128 base + 1 wrong (backward error 960: the misaligned 16-byte copy reads other elements, no fault) | — |

The 16-byte leg's k bound first stayed green: no 16-byte leg in the slab test had a k tail under NaN poison
(Padded 264×136 ran at k = 48). It now runs at k = 45, NaN-poisoned.

**Not covered by values:** the packet copy's row bound and the 16-byte edge clip (rows past m, n feed only
discarded accumulators), and priors read at beta = 0 (the beta = 0 store never uses them).

**Accepted assumption: `src-size` 0 never translates its address.** A zero-filled copy keeps its computed
address. Worst case past the operand: a k-contiguous operand (B, Aᵀ) at a ragged n (m) edge names up to
(TileN − 1)·ld (TileM − 1) beyond the last column of the last item, and 3 slabs (+23 elements) past k; a
packet operand (A, Bᵀ) up to TileM − 1 (TileN − 1) elements past a column and (k + 23)·ld into the slabs
past k. PTX does not promise that such an address is never translated (CUTLASS relies on the same); the tests
use one poison-padded buffer, so an unmapped address is never hit. Measured fixes (10 reps × 2 passes, ×vendor,
399a5a4e against the fix in the same session):

| fix | TT str 1024²×362 | TN packed 1024²×362 | NN str 512×256×128 b2048 | NT packed 1024³ | verdict |
|---|---|---|---|---|---|
| clamp every zero-size copy to `a_ptr` / `b_ptr`, 256×128 | 1.043 → 1.049, 1.051 → 1.059 | 0.948 → 0.961 | 1.020 → 1.028 | 0.972 → 0.975 | +0.3-1.4%: rejected |
| same, 128×256 | 1.033 → 1.049, 1.043 → 1.058 | 0.961 → 0.974 | 1.044 → 1.054 | 0.992 → 0.992 | +0.1-1.5%: rejected |
| skip the slabs past k (uniform branch), 256×128 | 1.045 → 1.048 | 0.948 → 0.964 | 1.019 → 1.019 | 0.973 → 0.971 | +1.7% on TN: rejected |
| same, 128×256 (210 → 247-255 registers) | 1.037 → 1.052 | 0.961 → 0.958 | 1.042 → 1.037 | 0.992 → 0.974 | +1.4% on TT: rejected |
| clamp, u4 128×128 (predicated leg) | 1.164 → 1.164, 1.162 → 1.162 | 1.090 → 1.097 | 1.126 → 1.130 | 0.996 → 0.995 | ≤ 0.7%: **shipped** |

The u4 128×128 pipeline (WP-N2) now names the operand's batch base; the large tiles keep the assumption. Wide v2
(`wide_v2::acopy`) has the same pattern and was not measured.

**Unranked:** no row of `tuned/gemm.sm_120.txt` names the new spellings, so Auto runs them only after the float
race is re-run; every ratio above is a pinned arm. sm_89: probed, not timed. The new header is in the kernel
hash of the ten tables that list `register_128x128.hh`.

## GEMM: WP-Y, coalesced element-leg lane maps

Measured 2026-10-21 on threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120), one GPU, `batchlas_tune --mode
time` (beta = 1, every arm verified), 12 reps. Each cell ran four times, new / parent / parent / new, each run two
passes in alternated arm order. Parent: `7dc92ec3` (a `git archive` tree, so it shares no source with the new one).

Target: wide v2's element leg (odd ld, `k % 4 ≠ 0` packed, unaligned base) at large k, where WP-Q measured
4.4× the 16-byte leg's global sectors ([negative](#gemm-wp-q-element-leg-loads-negative)).

**ncu first** (parent, one launch, `--clock-control none`):

| cell (b128 unless noted) | arm | ms | load sectors (M) | L1/TEX % | top stall |
|---|---|---|---|---|---|
| float TN packed 1024×1024×362 | k16, element leg | 3.79 | 969 (4.4×) | 86 | MIO throttle |
| float TN packed 1024×1024×360 | k16, 16-byte leg | 3.12 | 218 | 88 | MIO throttle |
| float TN strided 724×64×128 b2048 | k16 | 1.37 | 259 | 60 | MIO throttle |
| cfloat TN strided 512×512×64 | k8 | 0.614 | 102 | 59 | long scoreboard |
| cfloat NT strided 512×256×64 | k16 | 0.233 | 44 (5.2× ideal) | 67 | long scoreboard; DRAM 68% |

The k-contiguous operand (Aᵀ, B) is the loss. Its old map gave a lane one 16-byte granule along k (4 k of one
column, 4 scalar loads): each load instruction touched 8 columns × 2-3 sectors, and the next three instructions
touched the same sectors again.

**Shipped** (`register_64x64_k16_wide.hh`; no spelling, row or key moved; the leg stays derived in the launcher):

| piece | rule |
|---|---|
| lane map | element leg, float and complex<float>, k-contiguous operand only: lane `st` takes row `st % TileK` of columns `st / TileK + j · OpThreads / TileK` (`wide_v2::eload`). TileK lanes walk one column run, so an instruction is 2 columns × 64 B (k16) or 4 × 32 B (float k8) |
| shared row | that operand's row padded by 4 floats and unswizzled. At k = 16, rows 8-15 also flip the element's slot inside its granule (`ekx`; the fragment read un-flips by a compile-time index), so the 16 × 2 transposing store hits 32 banks |
| complex<float> | one 8-byte load per element where the lane's base is 8-byte aligned (`eload`, and the old map's `gload`), else two 4-byte loads |
| column step | 64-bit (`16 · ld · CF` passes INT_MAX at ld ≈ 67M for complex<float>), rebuilt per call behind an empty `asm volatile`: hoisted, its multiples spilled 8 B at the 80 and 128 caps (float and complex<float> k16, TN and CN) |
| local memory | `wide_slm_bytes` bounds both legs: + 2 × 16 B per k row for float and complex<float> v2 rows (float 64×64×16 17,408 B, complex<float> 33,792 B; 128×128 33,280 B) |

The i-contiguous operand keeps the granule map (16-byte shared stores). complex<double> and the 128×128 three-stage
cp.async leg are unchanged.

| cell (b128 unless noted) | arm | parent / vendor | new / vendor | new / parent |
|---|---|---|---|---|
| float TN packed 1024×1024×362 (two rounds) | k16 | 1.489, 1.479 | 1.377, 1.383 | 0.926, 0.931 |
| float TT strided 1024×1024×362 (two rounds) | k16 | 1.571, 1.568 | 1.533, 1.531 | 1.025, 0.978 |
| float TN strided 362×64×96 b2048 | k16 | 1.143 | 1.081 | 0.945 |
| float TN strided 724×64×128 b2048 | k16 | 1.106 | 1.054 | 0.951 |
| float NN strided 2048×64×96 b2048 | k16 | 1.185 | 1.158 | 0.977 |
| float TT strided 1024×45×128 b2048 | k16 | 1.073 | 1.061 | 0.988 |
| float TN strided 1024×1024×256 | k8 | 1.533 | 1.515 | 0.984 |
| cfloat TN strided 1024×256×78 | k16 / k8 | 1.144 / 1.260 | 1.116 / 1.199 | 0.975 / 0.952 |
| cfloat TN strided 512×512×64 | k8 | 1.083 | 1.025 | 0.949 |
| cfloat CN strided 512×512×256 | k16 / k8 | 1.099 / 1.240 | 1.073 / 1.175 | 0.975 / 0.945 |
| cfloat NN strided 1024×1024×64 | k16 / k8 | 1.172 / 1.240 | 1.177 / 1.228 | 1.003 / 0.989 |
| cfloat CC strided 256×256×96 b2048 | k16 | 0.972 | 0.963 | 0.990 |
| no k-contiguous operand: cfloat NT strided 512×256×64 | k16 | 1.018 | 1.019 | 1.002 |
| no k-contiguous operand: cfloat NC strided 128×1024×64 | k16 | 1.022 | 1.018 | 0.994 |
| no k-contiguous operand: float NT strided 2048×32×23 b2048 | k8 | 1.112 | 1.111 | 0.999 |
| 16-byte leg: float NN packed 512³ b2048 | k16 | 1.394 | 1.395 | 1.000 |
| 16-byte leg: cfloat NN packed 1024³ | k16 | 1.151 | 1.151 | 1.000 |

Every new / parent is the mean of the four runs; the per-run spread is under 2% except float TT strided
1024×1024×362, whose first round had one new run at 4.36 ms against 4.01-4.19 for the other seven (the second
round: 0.978). Forms with both operands k-contiguous (TN, CN) gain 1.6-7.4% (new / parent 0.926-0.984); forms
with one: NN 0-2.3%, CC 1%, TT flat (1024×1024×362 1.025 then 0.978, 1024×45×128 0.988).

**ncu after** (same protocol): float TN packed 362 3.49 ms (parent 3.79, 16-byte leg 3.12), load sectors 279M (969M),
L1/TEX 85%, top stall short scoreboard; float TN strided 724×64×128 1.29 ms (1.37), sectors 85M (259M), DRAM 79%
(74%); cfloat TN strided 512×512×64 k8 0.551 ms (0.614), sectors 64M (102M), DRAM 63% (56%). The fixed-clock ncu
gain (7.5% at 362) is larger than the timed one: under repeated launches this Max-Q part is power-limited
(`nvidia-smi dmon`: 1350-2325 MHz at 226-274 W).

**Negative** (each timed against the same parent, 6-10 reps):

| variant | result |
|---|---|
| WP-F's map, 8 lanes per 32-byte segment, at k = 16 (two instructions per 64-byte column run), with the i-contiguous operand i-fastest | 351M sectors at 362; 1-3.5% on the k16 cells |
| TileK lanes per column, the i-contiguous operand i-fastest too (64 lanes per row, scalar shared stores) | float TN packed 362 0.962, cfloat TN 512²×64 k8 0.953, but float NT strided 2048×32×23 k8 1.169: 4× the shared store instructions on the i-contiguous operand, so it went back to the granule map |
| funnel loads: two aligned 16-byte loads per granule and a runtime select (old map) | requests 49M → 26M, sectors 507M, L1 86%; same time as the parent |
| 8-byte complex<float> loads on the i-contiguous operand | NT, NC within ±0.5% (kept: it is the same branch as the k-contiguous load) |

The 128×128 complex<float> three-stage leg (`acopy`) is unchanged: cfloat NN strided 1024×1024×64 is DRAM-bound
there (2.09 ms at DRAM 68%, L1 32%; vendor 2.00 ms at 70%), and the padded row would take its three stages past
48 KiB.

**Registers** (`scripts/register_probe.sh`): `total spilling 0` on sm_120 and on a `-DBATCHLAS_NVIDIA_ARCH=sm_89`
tree. Element-leg instances on sm_120: float 79-80; complex<float> k16 128 (122 for NT, NC), k8 96-103; the
128×128 register path 254-255.

Tests: `GemmCandidates.WideV2ElementLegKContiguousMapAtSaturatingBatch` (float and complex<float>, 64×64×8 and
64×64×16, all 9 forms at ld = m + 1 with NaN in every gap and beta ≠ 0: 200×136×48, k = 15, 16, 17, 33 on 130×70,
200×136×48 at batch 1024 with 7 distinct problems, each item bit-identical to its representative; complex<float>
again at a 4-byte base). Deliberate breaks, run (filter: the wide v2 suites, `WideK8*`, `Wide128x128*`):

| break | red | green |
|---|---|---|
| fragment read without the k = 16 flip | float and complex<float> wide v2 suites, every first failure 64×64×16 with a k-contiguous operand (NN first); every 64×64×8 form passed first | 128×128 (k = 8), complex<double> |
| 8-byte load halves swapped (`eload`) | complex<float> only: 64×64 (k16, k8) and 128×128 under `WIDE_STAGES=2` in the 7 forms with a k-contiguous operand | float; 128×128 NT, NC; the cp.async stages |
| 8-byte load halves swapped (`gload`) | complex<float> only: every form at some layout, including the 16-byte leg's predicated edge slabs | float |
| `eload` drops its k bound | the k-tail cases (k = 7, 9, 15, 37) of float and complex<float>, and 128×128 at a 4-byte base and under `WIDE_STAGES=2` | full-slab cases |
| column step one element long | float and complex<float>; 128×128 in the 7 forms with a k-contiguous operand | 128×128 NT, NC |

The padding and the flip are bank-conflict layout only: a store and read that agree on them are right without them,
so no value test sees a dropped pad.

**64-bit column step** (review fix): registers within ±2 of the 32-bit step on sm_120 (complex<float> TT/TC/CT/CC
k16 128 → 126, float NN/CN k8 ±1), `total spilling 0` on sm_120 and sm_89. Against the 32-bit build (12 and 10
reps, four runs per order, both orders, vs each build's vendor): float TN packed 1024×1024×362 1.008 / 1.003,
TN strided 362×64×96 b2048 1.004 / 1.003, cfloat TN strided 512×512×64 k8 1.000 / 0.997.

**Table staleness.** WP-Y edits `register_64x64_k16_wide.hh`, in the gemm kernel hash, so `tuned/gemm.sm_120.txt`
is kernel-hash-stale (a warning at load). Until the retune it ranks the wide v2 arms by pre-WP-Y timings: the
element-leg TN/CN cells above are 1.6-7.4% faster than the times it holds.

Not done: the same map on the 128×128 cp.async leg's k-contiguous operand (needs 48+ KiB or an 8-value swizzle,
which cost WP-Q 75 registers); the sm_120 retune.

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
warp stores whole 128-row columns (512 B per `STG.128`). `kStagedEpilogueMaxK = 64` bounds the aligned leg. This
was the aligned leg only until WP-B extended it (see the section on the coalesced C epilogue above); since WP-F
the predicated leg stages at every k
([WP-F](#gemm-wp-f-float-mid-size-parity-on-sm_120)).

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
