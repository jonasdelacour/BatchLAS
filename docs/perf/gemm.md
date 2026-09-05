# GEMM: the native tiers, the Auto flip, and the strided-ld routing defect (WP2, WP3 S16)

All measurements: RTX 4090 / sm_89, one dedicated GPU via `experiments/gpu_guard.sh`, warm SYCL JIT (`--warmup=5`), median of 3, **both β=0 and
β=1**, unless a row says otherwise. Sanity anchors: vendor SGEMM must reach 45–48 TFLOP/s at 512³ (a number near 80 is TF32, not FP32); vendor
DGEMM must never exceed ~1.45 TFLOP/s, since a 4090 is 1/64 FP64.

## What ships

### The route arms

`kGemmOrder` (`include/batchlas/blas/dispatch/route_gemm.hh:12-15`) has exactly two arms:

| Origin | Algorithm | notes |
|---|---|---|
| `Native` | `RegisterTiled` | the SYCL kernel family in `src/sycl/gemm/`; `Algorithm::Auto` is also accepted by `supports()` |
| `Vendor` | `Auto` | cuBLAS / rocBLAS / MKL; `supports()` is unconditionally true |

`supports()` (`route_gemm.hh:19-32`) is **correctness only**: `precision == Default`, `m,n,k > 0`. Since WP2 C2 a heterogeneous batch is
*supported* natively — the facade walks the batch (`src/backends/gemm_heterogeneous.hh`), each member homogeneous by construction — and is
refused by `preferred()` instead. Conflating correctness with speed is the trap the split at `route_gemm.hh:3-5` prevents: a window inside
`supports()` leaves a 1024³ float GEMM at batch 256 with no route at all vendor-free.

The unset default is `{Origin::Auto, Algorithm::Auto}` for every op (`route_env.hh:88-91`). GEMM used to be the one op defaulting to a *forced*
Vendor; WP2 E6 removed that asymmetry.

### The preferred window as implemented

Quoted from `include/batchlas/blas/dispatch/route_gemm.hh:34-67`, in order of evaluation:

```
r.origin == Native, supports(r,s), s.is_gpu, !s.heterogeneous_batch      :35-40
complex<float>, complex<double>              -> false                    :43-44
s.batch < 64                                 -> false                    :48
float :  s.m == s.n && s.n == s.k                                        :51
         transA == NoTrans && transB == NoTrans                          :54
         max_dim <= 32                       -> true, else false         :57,58
double:  s.k >= 2                            -> the whole predicate      :62
anything else                                -> false                    :64
```

Read plainly: **native is preferred only for `double` (any shape, any transpose form, `k >= 2`, `batch >= 64`, GPU, homogeneous) and for `float`
NN squares with `max_dim <= 32`.** Complex is `false` at every shape. `preferred()` returning false never makes a route ineligible — vendor-free,
`resolve_route` still falls back to any *supported* native route (`route_resolve.hh:18-62`), so every narrowing below costs a vendor-free build
nothing.

Two sources disagree with the shipped predicate, and the code wins: `experiments/wp2_e6/README.md` and `route_env.hh:88` describe the flip's
double half as "square, n=4..512", and `scripts/gemm_demand.py:50-68` transcribes `m == n == k && max_dim <= 512` with no `k >= 2`. Both are the
pre-E5 predicate; E5 landed after E6 and removed squareness and the bound. The `gemm_demand.py` copy is a live defect — see [Open
debts](#open-debts).

### The kernel selector is the second gate

`preferred()` picks an *Origin*; `select_kernel_variant` (`src/sycl/gemm_kernels.cc:450-557`) picks the kernel, and it is the gate that decides
throughput. Its whole register ladder for float sits inside `if constexpr (is_same_v<T,float>)`. Reachable exits:

| condition | variant | line |
|---|---|---|
| transposed, float, `m>=128 && n>=32 && k>=128` | `Tiled128x32RegisterK32{TN,NT,TT}` | :471-482 |
| transposed, anything else | `max_dim <= 32 ? Direct : Tiled16` | :483 |
| float NN, `m,n,k >= 128` and 128×128 fast path | `Tiled128x128RegisterK8` | :520 |
| float NN, `m,n,k >= 128`, squareish, aligned | `Tiled128x32RegisterK32S2U1Aligned` / `Tiled128x64RegisterK32Large{,U2}` | :523-533 |
| float NN, `max_dim>=128 && k>=8 && mn_min>=64 && (mn_min>=128 \|\| k<128)` | `Tiled128x128RegisterK8` | :578 |
| float NN, remaining ladder | `128x32K16`, `32x128K16`, `64x64K16`, `64x64`, `32x32`, then `max_dim<=48 ? Direct : Tiled16` | :582-597 |
| non-float NN, `min_dim >= 256` and wide fast path | `Tiled64x64RegisterK16Wide` | :641-645 |
| complex NN, `min_dim >= 32 && ctas >= 64` (cfloat) / `>= 128` (cdouble) | `Tiled64x64RegisterK16Wide` | :694-703 |
| double, otherwise | `max_dim <= 24 ? Direct : Tiled16` | :725 |
| cfloat / cdouble, otherwise | `max_dim <= 64 ? Direct : Tiled16` | :728 |

Consequence, and the reason `preferred()` refuses complex: **a widened `preferred()` for complex does not route complex to a register kernel, it
routes it to `Tiled16`** — 3.2–7.1× slower than cuBLAS. Required order: port the kernel → widen the selector → widen the predicate.

## Evidence for each boundary

### Double, the only fully native window

`double`, square NN, batch 512 (batch 4096 for n ≤ 32), spreads 0.0–0.3%, GFLOP/s:

| n | native kernel | cuBLAS | native | ratio |
|---|---|---|---|---|
| 4 | Direct | 16.5–16.6 | 58.4–59.0 | 3.55× |
| 8 | Direct | 86.1–86.3 | 387–388 | 4.49–4.51× |
| 16 | Direct | 227–234 | 888 | 3.80–3.92× |
| 24 | Direct | 437–448 | 1130 | 2.52–2.59× |
| 32 | Tiled16 *(was Direct)* | 982–1018 | 1213 | 1.19–1.23× |
| 48 | Tiled16 | 633–647 | 1204 | 1.86–1.90× |
| 64 | Tiled16 | 1151–1168 | 1214 | 1.04–1.05× |
| 96 | Tiled16 | 903–913 | 1329–1330 | 1.46–1.47× |
| 136 | Tiled16 | 698–703 | 1214–1215 | 1.73–1.74× |
| 200 | Tiled16 | 829–833 | 1270 | 1.53× |
| 256–512 | wide 64×64 | 1239–1246 | 1399–1411 | 1.13× |

**Saturation verified, not asserted.** Sweeping batch 64 → 128 → 512 → 2048 → 8192 at the two largest margins, the ratio is stable or *rising*:
n=48 1.58 → 1.76 → 1.90 → 1.95 → 1.96×; n=136 1.82 → 1.73 → 1.74 → 1.75 → 1.75×. Launch overhead would pull a ratio toward 1 instead.

**Non-square (E5), on the shapes the demand table says the library issues**, batch 128, NN/NT/TN: 992×992×32 1.10–1.14×, 480×480×32 1.14–1.17×,
288×288×32 1.21–1.22×, 224×224×32 1.24–1.25×, 248×248×8 1.34–1.41×, 312×312×8 1.21–1.25×. **36 of 36 cells win.** Edges: 1024³ 1.13×, 2048³
1.14×, 4096×64×64 1.04–1.06×, 64×4096×64 1.04–1.05×, 992×992×8 1.39–1.46×. The same 36 shapes measure **0.22–0.51× for float**, 0 of 36 — which
is why E5 is a double-only widening.

**The `k >= 2` boundary is bracketed by its own counterexample.** At 512×512×k:

| k | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| ratio | **0.49×** | 1.64× | 1.62× | 1.58× | 1.53× | 1.49× | 1.34× | 1.09× |

k=1 is the only losing double shape in the whole work package. Its advantage is β=0 only — cuBLAS 230 → 114 GFLOP/s when β=1, native 112 either
way — i.e. cuBLAS has a rank-1 path. k=1 is 761 calls in the demand table, so the boundary sits at 2 rather than at a rounder number.

**Transposed double**, square, batch 512, n=32..512, both betas — measured only because the E6 prediction step noticed the double branch had no
transpose test at all. TN 1.01–1.12×, NT 1.06–1.11×, TT 1.04–1.12×, **CN (ConjTrans) 1.01–1.11×**; the minimum in every form is at n=32, and
every n≥64 cell is 1.09–1.12×. **No losses anywhere**, spreads 0.0–4.2%. Had double behaved like float here, the flip would have shipped a 2–3×
regression on every transposed double GEMM.

The contrast with float is a **ceiling argument, not luck**: cuBLAS DGEMM sits at 78–87% of the 4090's ~1.44 TFLOP/s FP64 ceiling and native at
88–98%, at every size from 4 to 2048. That is also why the window has no upper size bound — a gap from a size-independent mechanism needs no size
cutoff, and a cliff at 2048 would itself be the unjustified number.

**The `max_dim <= 24` Direct/Tiled16 boundary** (`gemm_kernels.cc:553`), double, GFLOP/s:

| n | batch | Direct | Tiled16 | winner |
|---|---|---|---|---|
| 24 | 512 | 708 | 518 | Direct 1.37× |
| 24 | 4096 | 1126 | 687 | Direct 1.64× |
| 25 | 4096 | 750 | 746 | wash (0.99–1.02× across b512/b4096) |
| 28 | 4096 | 903 | 937 | Tiled16 1.04× |
| 32 | 512 | 903 | 973 | Tiled16 1.08× |
| 32 | 4096 | 938 | 1211 | Tiled16 1.29× |

n=32 was the *only* losing cell in the accepted double window (0.92–0.96× at batch 4096) and it was a misplaced kernel boundary, not a routing
problem. Moving it to 24 made n=32 a 1.14–1.23× win. 24 rather than 25 because 25 is inside the run-to-run spread and a boundary belongs where
the evidence is unambiguous.

### Float NN at max_dim 32

The window kept, and the cell that brackets it — square NN, batch 512, both betas:

| n | ratio (β=0, β=1) | verdict |
|---|---|---|
| 8 | 1.43×, 1.46× | keep |
| 16 | 1.22×, 1.31× | keep |
| 32 | 1.03×, 1.08× | keep — the last win |
| **33** | **0.92×, 0.96×** | **the bracketing loss** |
| 48 | 0.58×, 0.60× | |
| 64 | 0.79×, 0.83× | |
| 96 | 0.49×, 0.43× | |
| 127 | 0.36×, 0.44× | |

**Two float windows were removed on measurement**; 40 cells argued to narrow and none to widen.

*NN 128..512*: n=128 0.97–0.98×, n=192 0.39–0.47×, n=256 0.80–0.87×, n=384 0.77–0.79×, n=512 0.91×. Not an unsaturated artefact — flat across
batch 128 / 512 / 1024 (n=192 measures 0.42/0.41, 0.40/0.49, 0.39/0.48 at the three batches). Above the removed window it is the same story:
n=640 0.90–0.94×, n=768 0.91–0.94×, n=1024 0.97×.

*Transposed 128..512*: the claimed window is 30 cells (5 sizes × TN/NT/TT × 2 betas) and **all 30 lose, 0.34–0.55×**. Over the wider grid
n=64..768 it is 48 of 48 losses, 0.23–0.55×, worst at n=96 (0.23×). This is not a fallback effect: TN runs its dedicated `register_128x32_k32_tn`
kernel, traced and confirmed. The transposed register family plateaus near 15–18 TFLOP/s while cuBLAS SGEMM reaches 45+.

### Complex is refused

`preferred()` returns false for complex at every shape (`route_gemm.hh:43-45`). The reason is the selector: the only complex register arm is
`Tiled64x64RegisterK16Wide`, reachable at `min_dim >= 256` + aligned NN or through the CTA gate below; everything else falls to `Tiled16`,
measured 3.2–7.1× slower than cuBLAS (cdouble 0.386–0.392 vs 1.238–1.246 TFLOP/s; cfloat 6.6–6.9 vs 45–49 TFLOP/s). The route equivalence test
asserts complex never moves under the flip.

### The Auto flip

WP2 E6 changed `legacy_unset_default(Op::gemm)` from a forced `{Vendor, Auto}` to `{Auto, Auto}`. Route diff, checked field by field: **262
decisions moved, 0 regressions** — added decisions that are not native: 0; native decisions lost: 0; complex decisions moved: 0; double moved
27–28 per transpose form across all 9 forms; float moved 11, **NN only**. That float moved in NN only is direct confirmation the float narrowing
was load-bearing: without it, float would have moved in all nine forms at 0.34–0.55×. E5's own diff was a further 81 decisions, all double, all
vendor→native, zero regressions, zero complex.

The flip changes nothing in a vendor-free build (the Vendor default was never reached there; failing set verified byte-identical) and nothing for
explicit requests — `BATCHLAS_GEMM_VARIANT=vendor` still means vendor, which is the escape hatch if a future cuBLAS turns a cell around. That is
not hypothetical; see the aged-out parity claim below.

Vocabulary trap (`route_env.hh:97-160`): `BATCHLAS_GEMM_VARIANT=native` does **not** mean BatchLAS's own kernel — it aliases
`cuda-native`/`direct-cuda`, is consumed only as an exclusion, and is `Origin::Vendor` in the canonical vocabulary.

### The 128x128 float kernel

`src/sycl/gemm/register_128x128.hh` — 128×128×8 macro tile, 8×8 accumulators (64 per thread), 256 threads. Ported from
`experiments/sycl_vs_cuda/`, which settled the premise directly: the same SGEMM body compiled by nvcc and by DPC++ produces **the same SASS inner
loop** (512 FFMA, 32 `LDS.128`, 16 FFMA per `LDS.128`, 2 `BAR.SYNC`, 115 vs 113 registers, zero spill) and the same runtime — SYCL at 99.3% /
100.1% / 98.7% of the CUDA build at 512³b512, 1024³b64, 256³b1024. The in-tree gap was kernel design, not language: BatchLAS's best SYCL GEMM was
21 TFLOP/s at 512³b512 where this kernel reaches 43.6. The "80 TFLOP/s peak" is a TF32 number (cuBLAS TF32 78.04 / 84.11); strict-FP32 SGEMM tops
out at 43.9–47.5 here, and the hand-written kernel already reaches it. Its 8×8 tile issues 4 vectorized shared loads per 64 FFMAs — a 16:1 ratio
against the older family's 2.0–2.7:1.

In-tree, event-timed, β=1, GFLOP/s:

| shape | vendor | 128×64×32 (was) | 128×128×8 (now) | vs vendor |
|---|---|---|---|---|
| 128³ b4096 | 14480 | 7223 | 14254 | 98.4% |
| 256³ b1024 | 29187 | 14065 | 25596 | 87.7% |
| 512³ b512 | 40755 | 22672 | 41545 | 101.9% |
| 512×256×512 b512 | 41066 | 21974 | 37044 | 90.2% |
| 512×64×512 b512 | 20298 | 16208 | 17822 | 87.8% |
| 1024³ b64 | 45870 | 24038 | 44062 | 96.1% |

**The predicated leg was unlocked by E4.** The selector used to hand squareish float that failed the 128×128 fast path to the generic 128×32×32
route, behind a comment saying the predicated path had never been benchmarked against it. Measured (square NN, batch 512, 96 for n ≥ 544, both
betas, GFLOP/s):

| n | 160 | 192 | 224 | 320 | 544 | 672 | 800 | 1056 |
|---|---|---|---|---|---|---|---|---|
| generic | 7 892 | 9 781 | 11 611 | 12 188 | 13 372 | 14 107 | 14 654 | 15 065 |
| predicated | 13 170 | 18 000 | 22 467 | 25 288 | 27 101 | 29 715 | 31 354 | 33 314 |
| gain | 1.67× | 1.84× | 1.93× | 2.07× | 2.03× | 2.11× | 2.14× | 2.21× |

The gain **grows with n**, which is what a per-tile predication cost looks like against a route whose throughput has plateaued. It moves the
bucket from 0.36–0.51× to 0.72–0.84× of cuBLAS — still a loss, which is why `preferred()` does not claim it, but it halves the damage
vendor-free. The unaligned-`ld` cases gain most: n=256 ld+2 7 237 → 23 966 (**3.31×**), n=512 ld+2 8 399 → 36 862 (**4.39×**) — the shape class
BatchLAS's own factorisations hand to `gemm`, since a panel is a sub-view carrying its parent's `ld`. Only the *generic* leg changed; the aligned
leg is a different tuned route and was never in the measurement.

**One in-tree claim aged out.** `register_128x128.hh:33-35` still records "43.6 TFLOP/s against cuBLAS SGEMM's 43.9" at 512³b512, i.e. parity.
Re-measured, the native half reproduces exactly (43.5); the cuBLAS half does not — it now measures **47.3**. A ratio recorded against a vendor is
only as durable as that vendor's version.

### The wide scalar kernel

`src/sycl/gemm/register_64x64_k16_wide.hh` — 64×64×16 macro tile, 4×4 thread tile, the only register-tiled variant serving a non-float scalar.
Against cuBLAS and against a faithful standalone replica of the in-tree `Tiled16`, at 256³b512, 512³b128, 1024³b32, both betas:

| scalar | vs `Tiled16` | vs cuBLAS |
|---|---|---|
| `complex<float>` | 7.0–7.7× | 0.98–1.08× CGEMM |
| `complex<double>` | 3.56–3.60× | 1.12× ZGEMM |
| `double` | 1.01–1.08× | 1.07–1.15× DGEMM |
| `float` | — | **0.85–0.93× SGEMM**, so float never routes here |

Registers / spill, sm_89: 55/56 (float), 72/76 (double), 72/80 (cfloat), 132/134 (cdouble), **zero spill in all 16 entries**; after the
device-scalar types were lifted to `src/sycl/device_scalar.hh`, `scripts/register_probe.sh` still reports 56 / 76 / 80 / 132. Caveat from the
source: cuBLAS CGEMM has ~±5% spread (44.76–45.69 over 5 repeats at 512³b128 β=1), so every cfloat ratio is ±5%; the 5-run means give 48.41 vs
45.35 = 1.068×, which exceeds the combined spread. ZGEMM/DGEMM spread is 0.5–0.7%.

Read the `double` row as small **on purpose**: FP64 on a 4090 is 1/64 of FP32, ceiling ~1.44 TFLOP/s; this kernel reaches 1.415 (99%) but the
naive `Tiled16` already reaches 1.33 (92%). There was never 3× on the table for double on this part. **That conclusion is 4090-specific and
inverts on a 1:2-FP64 datacenter part**, where `Tiled16` would not be near the ceiling.

Five load-bearing details, each found in PTX, each reverting a measured property if dropped: a **16-byte** (not 4-element) access granule, so an
8-lane LDS phase lands on exactly the 32 banks at every scalar width; `may_alias` on the punning types, or -O3 reorders the shared stores against
the fragment loads across the barrier; a native LLVM vector type for the whole-granule staging copy, because SROA splits a struct copy back into
element accesses; `std::complex` never reaching device code (POD `Cx<R>`, multiply as four `fma`s — kills `__mulsc3`/`__muldc3` and the Annex-G
isnan branch); shared strides exactly `TileM`/`TileN`, m fastest-varying in the epilogue.

**How often the `min_dim >= 256` arm fires, measured rather than assumed:** 46 of 7223 real non-float gemm calls, **0.64%** — after removing the
2312 synthetic probe rows that `route_gemm_equivalence_tests.cc` feeds straight to the resolver. With probes left in it looks like 3.56%, and
every probe hit is a large square aligned shape, i.e. exactly the cells a new tile wants credit for. Restricted to `max(m,n) >= 128`, 91.6% are
blocked by `k < 256` and 69% by a transpose. Structural, not test sizing: the dominant internal GEMM is a panel update (large m, large n, small
k) and k is a blocking constant clustered at 1/8/32/48/96/136, so `min_dim` — a min over k — cannot rise with problem size. **No single
relaxation rescues it: zero calls are blocked by the k floor alone.**

### The CTA count gate for complex

`gemm_kernels.cc:539-548` admits complex NN to the wide kernel on `min_dim >= 32 && ctas >= kMinCtas`, with `kMinCtas` 64 for `complex<float>`
and 128 for `complex<double>`, where `ctas = ceil(m/64)*ceil(n/64)*batch`.

Forced-wide vs the route it replaces, at saturation, both betas, geomean over 116 refused cells: **cfloat 3.98×, cdouble 2.90×**, null controls
1.000. That number is *not* the gate: it is measured in a regime the newly-captured call sites never enter — every demand shape the relaxation
captures runs at batch 1–8, and there wide loses in 12 of 12 cells (cfloat 0.60–0.80×, cdouble as bad as 0.174×) while winning at b256 in 12 of
12. A 180-cell ladder (batch 1..256 × 5 shapes × 2 types) shows the crossover is **not a constant batch** — it moves 8 → 128 by shape — but is
very nearly a constant number of work-groups: the wide kernel launches up to 16× fewer CTAs than `Tiled16` and cannot fill a 128-SM part at small
batch, and cdouble needs twice the CTAs because its 32 KB of shared memory caps it at 3 blocks/SM.

`tiled16_ms / wide_ms`, >1 means wide wins:

| type | shape | b1 | b8 | b16 | b32 | b64 | b128 | b256 |
|---|---|---|---|---|---|---|---|---|
| cfloat | 129×96×129 | 0.58 | 1.02 | 1.60 | 1.93 | 2.65 | 2.75 | 2.73 |
| cfloat | 96×64×96 | 0.64 | 0.77 | 0.96 | 1.64 | 2.75 | 3.68 | 3.88 |
| cfloat | 33×61×33 | 0.60 | 0.56 | 0.65 | 0.75 | 1.08 | 1.55 | 1.98 |
| cdouble | 129×96×129 | 0.21 | 0.79 | 1.37 | 1.38 | 1.79 | 1.77 | 1.76 |
| cdouble | 96×64×96 | 0.24 | 0.45 | 0.67 | 1.31 | 2.59 | 2.60 | 2.58 |
| cdouble | 33×61×33 | 0.17 | 0.18 | 0.33 | 0.48 | 0.93 | 1.82 | 1.82 |

Re-indexed by CTA count instead of batch, the eight crossovers collapse onto two: cfloat `ctas >= 64` admits 26 clean cells, **worst 1.08×, zero
losses**; cdouble `ctas >= 128` admits 24 cells, **worst 1.08×, zero losses**. Every bound has a measured counterexample on the other side:

* cfloat just below 64 CTAs: 129×96×129 b8 = 48 CTAs, **0.79× loss**.
* cdouble just below 128: 33×61×33 b64 = 64 CTAs, **0.93× loss**.
* 64 CTAs is genuinely ambiguous for cdouble — it holds that 0.93× loss *and* a 1.31× win (96×64×96 b32). 128 is chosen to admit no loss and
  knowingly gives up a real 1.37× (129×96×129 b16). Conservative on purpose.
* `min_dim >= 32` is needed independently: the CTA gate alone would admit tiny shapes at huge batch, and 16×16×16 loses 0.71× (cfloat) / 0.28×
  (cdouble), while 32³ wins 2.28× / 1.05×.

The `min_dim >= 256` arm is kept ahead of this one so nothing routing to the kernel today stops doing so; 256³b4 and 512³b1 were verified
unchanged by trace.

### The strided ld defect and the routing fix

Every operand `trsm` hands GEMM is a sub-view carrying its parent's leading dimension — a 128-row `C` with `ld = 512`. On the six shapes `trsm`
V2 issues at order 512 (float, q=1024, batch 512), native/vendor ms at the real `ld`: the three outer shapes m=128 n=1024 k={128,256,384} measure
1.53/0.96, 2.73/1.31 and 3.78/1.63 (**0.62×, 0.48×, 0.43×**), the three inner shapes m=32 n=1024 k={32,64,96} measure 0.406/0.235, 0.680/0.335
and 0.887/0.426 (**0.58×, 0.49×, 0.48×**). The same native shapes at `ld == rows` take 0.98, 2.35, 3.49 and 0.248, 0.356, 0.487 ms — 0.86–0.98×
of the vendor on the inner shapes. **cuBLAS barely moves, and no square benchmark can see this.** (Routing those trailing updates through `RouteTable<Op::gemm>` instead of calling `gemm_custom`
directly took the n=512 solve from 18.8 ms to **11.19 ms** against a 14.28 ms vendor `trsm`, with no kernel change.)

**ncu, on m=128 n=1024 k=128 b512 β=1, pad 0 vs pad 384:** every transaction counter is byte-identical — 2,097,152 load requests, 33,554,432
sectors, **16.00 sectors/request** in both, identical DRAM sectors, identical instructions, 119 registers, zero spill. A per-SASS-instruction
check across all 1000 instructions × 7 traffic counters found **0 differences**. Only the time moves: 917.3 → 1493.1 µs (1.63×), DRAM throughput
89.28% → 55.02% of peak. cuBLAS (`ampere_sgemm_128x128_nn`, same tile, same 4096-block grid, 118 registers) pays 1.05× on the same shape (869.4 →
912.4 µs) and its long_scoreboard stall is flat.

The whole regression is exposed global-load latency at the k-loop barrier. Warp cycles per issued instruction 13.80 → 22.89; of the +9.09,
**barrier accounts for 68%** (1.552 → 7.703) and long_scoreboard for 33% (8.755 → 11.740); eligible warps per scheduler 0.55 → 0.29 with active
warps unchanged. It belongs to **one operand, B**: pad applied one operand at a time gives none 0.9775 ms, A only 0.9816 (0.7% of the penalty), C
only 1.0327 (9.8%), **B only 1.5173 (96.0%)**. B is read as 32 B from each of 16 different columns per warp — 16 L1 tag requests against 4 for a
coalesced load — and those 16 streams are `ldb*4` bytes apart. It is a **slope, not a cliff** (padB 0/4/8/32/64/128/256/384/896 →
0.977/1.000/1.005/1.068/1.204/1.425/1.510/1.518/1.628 ms, monotone; the power-of-two byte strides add only 5–7%) and it is **beta-independent**.
Footprint is ruled out: 4× the allocation at `ld == rows` costs nothing (18 001 vs 17 571 GFLOPS) while the same footprint with a stride costs
the full 1.57×.

**The fix that worked was routing, not a kernel change.** `can_use_128x128_fast_path` is a *leg* predicate — the dispatcher re-evaluates it and
picks `<true>`/`<false>` itself — but the selector used it as a *routing* gate, so failing it did not demote a call to the predicated leg, it
handed the call to an entirely different, much slower kernel. Routing by what the kernel can run (`gemm_kernels.cc:501`) is worth **geomean 1.74×
/ 1.75×** (pad 0 / pad 384) over 12 shapes, moving native from 0.58× → 0.99× of cuBLAS at `ld == rows` and 0.54× → 0.93× strided:

1024×1024×64 b128 ld1408 3.187 → 1.337 ms (2.38×), 1000×1024×128 b128 ld1384 2.954 → 1.569 ms (1.88×), 1024×1024×16 b128 2.622 → 1.232 ms
(2.13×), 128×128×8 b512 0.074 → 0.030 ms (2.43×). It also subsumes the `ld % 4 != 0` cliff (pad 1: 1.874 → 1.003 ms), because that branch does
not consult the alignment predicate at all. **Every bound in that gate has a measured counterexample:** `mn_min >= 64` — 32×1024×32 is 0.97× (a
wash-to-loss); `mn_min >= 128 when k >= 128` — the tuned routes it would displace win, 64×64×512 b512 0.77/0.69, 64×64×1024 b256 0.58/0.62,
64×1024×512 b256 0.64/0.62, while at `k < 128` 128×128 wins even at `mn_min = 64` (1024×64×64 1.80×, 64×1024×64 1.59–1.81×); `max_dim >= 128` —
64×64×64 is a wash (1.02×); `k >= 8` — it is the kernel's TileK, and 1024×1024×8 wins 2.00×.

**Reach, stated honestly in the code:** with cuBLAS present this changes no runtime at all, because float's `preferred()` window requires `m == n
== k`, so every shape the gate captures resolves to the vendor (coverage: 79 native float gemm calls against 102,791 vendor). The deliverable is
the vendor-free and ROCm builds, and making a future `preferred()` flip *arguable*. At 0.93× it is not yet arguable.

## Negative results

* **Double-buffering the 128×128 k-loop.** 127 registers, zero spill, barriers halved, and it incidentally fixed the split-`LDG` defect to
  *exactly* cuBLAS's sector count — for **zero time recovered**. cuBLAS uses 17.664 KB shared per block against our 9.216 KB and is
  occupancy-limited by registers anyway, so the extra shared memory is free for it; that asymmetry is why copying its structure did not copy its
  result.
* **Packing B into contiguous scratch.** Pays at the same roofline the kernel already achieves; loses harder as m grows.
* **The WP3 mechanism for the `ld` defect.** WP3 blamed `register_tiled_common.hh` — odd tile strides `TileM+1`/`TileK+1`, `[n][k]` B staging, a
  read-modify-write epilogue, a contiguity predicate every sub-view fails. **Those shapes never execute that file**: they route to
  `Tiled128x128RegisterK8` with `AlignedFastPath = true` in *both* columns, and `can_use_128x128_fast_path` never tests contiguity. The effect is
  beta-independent and B-only, which refutes the epilogue story directly. Confirm which kernel runs before theorising about why it is slow — the
  second time in this campaign a named mechanism belonged to code that was not executing.
* **The wide-scalar tile for float.** 0.85–0.93× of cuBLAS SGEMM. Halving the thread tile to fit wide scalars costs float exactly what the
  64-accumulator tile bought it.
* **A 128×128 8×8 tile for wide scalars.** Not a spilling problem but a **launchability** one: double at 8×8 compiles to 208 registers and cfloat
  to 247, both with *zero* spill; only cdouble spills (3.4 KB, costing 3.5%). What fails is 208 × 512 threads > the 65,536 registers-per-block
  limit — cdouble throws at launch. The "128 accumulator registers cannot fit and it spills" belief in the original brief is measured false.
* **The `complex-split` candidate.** Matches the 64×64 tile's throughput at 247 registers and 1 block/SM. Not landed: the 64×64 tile uses one
  shape for all four scalars and is the only candidate with no unlaunchable and no spilling configuration.
* **The FFMA:shared-load ratio as a design lever for wide scalars.** A tile-vs-occupancy scan at 32:1 / 21.3:1 / 16:1 lands within 5% across the
  board for complex, and 4:1 / 8:1 / 16:1 within 4% for double — the shared pipe is over-provisioned by an order of magnitude for FP64 on a
  consumer part. It *is* the discriminator for float (the 128×128 kernel's whole thesis), which is why the two kernels have different shapes.
* **A bare `min_dim >= 32` floor for the complex relaxation** (`routing_proposal/`, kept unapplied). Exactly what the 2.90–3.98× geomean argues
  for, and it would have regressed every shape it newly captured — see the 12-of-12 losses above.
* **Two float `preferred()` windows** — NN 128..512 and the entire transposed window; see [Float NN at max_dim 32](#float-nn-at-max_dim-32). And
  `preferred()` for float on the non-square demand shapes: 0 of 36 cells win (0.22–0.51×), which is why E5 is a double-only widening.

## Correctness findings

* **Nine transposed launchers computed the wrong answer for `ConjTrans`** (fixed in `f236575`). They hard-wire OpA/OpB, and `ConjTrans` is a
  distinct enum value (NoTrans=0, Trans=1, ConjTrans=2), so a launcher instantiated `<Trans, NoTrans>` silently dropped the conjugation and
  returned a plausible matrix. **How it hid:** unreachable from `select_kernel_variant` but forceable by name via `BATCHLAS_GEMM_SYCL_KERNEL` —
  exactly how a benchmark compares variants, so it produced a valid-looking timing for an incorrect result. **Why the existing test could not
  fail:** the pre-existing ConjTrans case is 18×14×12 and cannot reach a 64×64 macro tile at all — blind by construction. **The guard now:**
  `ForcedTransposedLauncherRejectsMismatchedTransposeForm` (`tests/gemm_tests.cc:2534`) forces `64x64x16tn` on a 96×96×80 CN shape at α=2, β=−1,
  referencing `Tiled16` and *not* the vendor — a vendor reference is inert in a vendor-free build, where the fallback would be the kernel under
  test. CN is 789 of 2245 `complex<float>` calls in the demand capture.
* **Three heterogeneous-batch semantics existed only inside cuBLAS-gated code**: `m == 0` / `n == 0` members are *skipped*, a `k == 0` member is
  not a GEMM but `C := beta*C`, and an all-skipped batch must still return a valid `Event`. Vendor-free they did not exist — all 17 remaining
  vendor-free `gemm_tests` failures were heterogeneous batch. The loop is now `src/backends/gemm_heterogeneous.hh` with the per-item terminal as
  a parameter, so both backends share one copy. Vendor-free `gemm_tests` 167/184 → **184/184**.
* **A benchmark's own hygiene is part of the measurement.** In the first `ld` campaign the padded operands were allocated *uninitialized* while
  the unpadded ones used `::Random`, so every cross-`ld` ratio compared data content as well as leading dimension. Fixed; the reference cell
  moved 0.34% — the effect was real, but nobody knew that until it was checked.
* **Summing SYCL event-profiling intervals over queued submissions does not measure kernel time.** With 30 submissions in flight the summed
  `command_start..command_end` interval reported **19.836 ms** for a kernel whose true time is **3.15 ms** — 6.3×, since the interval includes
  queue wait. Any in-tree SYCL-vs-CUDA-event comparison timed this way is suspect.
* **A β=0 microbenchmark is structurally blind to an epilogue defect.** The first in-tree 128×128 version scored 26.0 TFLOP/s against the
  standalone kernel's 41: its epilogue had m as the *slow*-varying thread index, so with `beta != 0` the read of C became one scattered
  transaction per lane. Making m fastest-varying took it to 41.1. Measure both betas on both arms, always.
* **The route equivalence test asserts its own exception list.** `tests/route_gemm_equivalence_tests.cc` pins the decision against a transcribed
  replica of the legacy behaviour, with `ReplicaIsFaithful` so the replica cannot drift and pass vacuously. The four intended divergences (C2
  heterogeneous widening, E4 float narrowing, E6 default flip, E5 double widening) are classified and **counted separately** — one boolean would
  let a divergence vanish from the grid while another kept the count non-zero. E6's exception is paired with `UnsetNowMeansAuto*`, asserting
  *positively* that unset and `"auto"` agree on every shape in the grid.

## Open debts

* **Complex is what is still vendor-dependent, and that is the honest headline.** The panel-update population that dominates real demand needs a
  *transposed and predicated* wide-scalar kernel — a new kernel, not a routing change. `ConjTrans` is supported by **no** register-tiled variant
  in the tree (every TN/NT/TT launcher passes `Transpose::Trans`; no instantiation in the built `.so` carries `ConjTrans`), and for complex that
  is the transpose that matters: herk, her2k and hemm all issue it.
* **`scripts/gemm_demand.py`'s `preferred()` replica has drifted and is now wrong.** Its `double` branch (`:50-68`) is the pre-E5 `m == n == k &&
  max_dim <= 512` with no `k >= 2`. Every demand figure it produces under-counts double, and `--check` will report genuine native rows as
  disagreements. Its own docstring says a drifted replica is worse than no replica.
* **The double window deliberately reaches past its measurements.** No upper size bound; largest measured 2048³. The FP64-ceiling argument is an
  argument, not a measurement, above 2048.
* **Nothing in `ctest` asserts on kernel choice or throughput**, and `scripts/route_diff.sh` records resolver `Route`s, not `KernelVariant`s, so
  it is structurally blind to every selector change on this page. A regression in the CTA gate or the `mn_min` gate would be **completely
  silent**, visible only as the suite getting slower.
* **`SelectSyclKernelVariantForTest` (`tests/gemm_tests.cc:213`) hard-codes `Matrix<float>` and `select_kernel_variant<float>`,** so every
  `GemmDispatchPolicyTest` is float-only: the double `max_dim <= 24` boundary, the complex CTA gate and the wide kernel's `min_dim >= 256` arm
  have **no dispatch-policy test at all**. Same blind-by-construction shape as the 18×14×12 ConjTrans case.
* **The wide kernel's predicated leg has never been timed against `Tiled16`.** It is correct (round-off on 70×53×37) and reachable, but both
  routing arms gate on the aligned fast path or on a CTA count, so no timing of the predicated leg exists.
* **The 12-cell subset behind the 1.74× / 1.75× routing geomean is not identified in the preserved data.**
  `experiments/wp4_gemm_ld/routing/summary.csv` holds 15 cells (geomean 1.51 / 1.53 over all of them); the four quoted cells reproduce from
  `routing/raw/e4-*`, but the aggregate is not re-derivable without knowing which 12 were used.
* **Vendor-free heterogeneous GEMM is ~7 GFLOP/s** against a ~47 TFLOP/s FP32 peak — ~6000× off, and launch-bound, not kernel-bound: one launch
  per batch member, and vendor-present versus vendor-free measure identical within a 2–13% spread (6.96/6.99, 7.25/7.39, 7.58/8.14 at 64³b4096 /
  128³b1024 / 256³b256). The single-launch alternative is buildable without new infrastructure (`KernelMatrixView` already carries
  `active_rows_`/`active_cols_`) and is deferred.
* **The `ld` slope has no established mechanism below L2.** Sector counts, L1/L2 hit rates, L2 slice distribution, DRAM channel distribution and
  DRAM sector counts are all unchanged; the DRAM is idle 45% of the time delivering identical sectors. The remaining candidate is row-buffer
  locality, and **ncu exposes no row-activate counter**, so it was not measured.
* **The demand tables are `ctest` coverage captures, not user workloads.** The batch 1–8 distribution behind the CTA gate is evidence about
  *test-suite runtime*; **no capture of user workloads exists**. Batch=1 is not an optimisation target here, so the honest reading is: the gate
  must not regress the small-batch population, and stands to help a large-batch population whose size is unknown.
* **Three smaller items.** `split_k` is compiled but triple-gated (name-only selection, `BATCHLAS_GEMM_EXPERIMENTAL`, and a predicate requiring
  float/NN/`m,n,k>=256`/`m%128`/`n%32`/`k%128`) and has never been measured; `gemm_benchmark` is NN-only and structurally cannot measure a
  transposed shape, which is why the complex campaign needed the standalone `experiments/wp4_complex/gpu1/cx_gemm_bench.cpp`; and
  `Tiled128x32RegisterK32` is unreachable from the selector *and* rejected by name while `launch_register_128x32_k32_variant` has no caller — two
  dead enum entries worth deleting so the enum count matches the reachable count.
* **TF32 is reachable but unmeasured.** `experiments/sycl_vs_cuda/tf32_smoke.cpp` compiles `joint_matrix` with `precision::tf32` for sm_89 and
  its PTX carries 64 real `mma.sync...m16n16k8.f32.tf32.tf32.f32` instructions with correct results — reachability only, no staging and no reuse,
  so no throughput number. Whether a *tuned* SYCL `joint_matrix` GEMM reaches cuBLAS's ~78 TFLOP/s is not measured, and `supports()` rejects
  `ComputePrecision != Default` regardless.

## Raw evidence

Raw data is preserved at the git tag `perf-evidence/vendor-independence`.
Retrieve any path below with `git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| Double window n=4..512, saturation, the Direct/Tiled16 boundary | `experiments/wp2_e3/` (`e3_double.csv`, `e3_small.csv`, `e3_sat.csv`, `e3_bound.csv`, `e3_after.csv`) |
| Float NN and transposed windows; the predicated-128×128 selector fix | `experiments/wp2_e4/` (`e4_nn.csv`, `e4_batch.csv`, `e4_trans.csv`, `e4_n192.csv`, `e4_large.csv`, `e4_ld.csv`) |
| Non-square double, the demand shapes, the k=1 boundary | `experiments/wp2_e5/` (`e5_double.csv`, `e5_float.csv`, `e5_edges.csv`, `e5_k.csv`) |
| The Auto flip: prediction, route diff, transposed double | `experiments/wp2_e6/` (`e6_predict.py`, `e6_dtrans.csv`) |
| Wide-scalar tile bake-off, PTX/ptxas evidence, cuBLAS baselines | `experiments/wide_scalar_gemm/`, `experiments/wide_scalar_gemm/measure/` |
| SYCL-vs-CUDA parity, SASS counts, the 128×128 design, TF32 probe | `experiments/sycl_vs_cuda/FINDINGS.md` |
| The strided-`ld` ncu campaign and the routing fix | `experiments/wp4_gemm_ld/gpu1/README.md`, `experiments/wp4_gemm_ld/routing/` |
| Complex routing defect, merge gate, CTA ladder | `experiments/wp4_complex/README.md` + `smallbatch/`, `batchsweep/`, `routing_proposal/` |
| The trailing-update GEMM inside `trsm`, the sub-view `ld` | `experiments/wp3_s16/README.md` |
| Design narrative and per-step verdicts | `WP2_GEMM_SPEC.md`, `WP2_WIDE_SCALAR_GEMM_VERDICT.md`, `VENDOR_INDEPENDENCE_PLAN.md` |
