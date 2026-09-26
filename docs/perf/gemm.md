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

`preferred()` returns false for complex at every shape. The reason was the selector: the only complex register arm was
`Tiled64x64RegisterK16Wide`, reachable at `min_dim >= 256` + aligned NN or through the CTA gate below; everything else fell to `Tiled16`,
measured 3.2–7.1× slower than cuBLAS (cdouble 0.386–0.392 vs 1.238–1.246 TFLOP/s; cfloat 6.6–6.9 vs 45–49 TFLOP/s). The route equivalence test
asserts complex never moves under the flip.

P6 added two more complex register arms (the transposed panel tiles) and **the refusal still stands** — this time on its own measurement, not
for want of a kernel: see [Wide-scalar transposed tiles](#wide-scalar-transposed-tiles).

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

### Wide-scalar transposed tiles

`src/sycl/gemm/register_wide_transposed.hh` — the P6 family, now **measured**. Four variants, 4 types, 16 kernels. Two of them are
**selected** for complex transposed shapes in `select_kernel_variant`; `preferred()` is **unchanged**, so in a build with cuBLAS present
complex still routes to the vendor. That split is the result, not a staging step: see "the window that was refused" below.

Raw csvs: `benchmarks/results/p6_gemm_{nc_potrf_shapes_complex,nc_potrf_shapes_real,cn_geqrf_shapes_complex,bracket_cells,ragged_tile_sweep}.csv`
plus `benchmarks/results/p6_e2e_before.csv` (the shipping state) and
`benchmarks/results/p6_e2e_with_refused_window.csv` (the withdrawn route window). 220 kernel cells (880 timed arms) plus 48 end-to-end cells; 1 discarded (named below). Harness: an out-of-tree standalone A/B built on
`factor_bench.cc`'s rules — one cell per process under `gpu_guard.sh 1`, warm-up interleaved in the timed loop's **arm order**, 7 reps,
per-arm medians, `rel_sd` gate, and an untimed host-checked run of every arm in double promotion on items 0 and batch-1.

**Why a new kernel and not a flag.** Neither existing family serves these shapes. `register_tiled_common.hh` carries `Transpose OpA/OpB`
but its inner loop is `accum += a * b` on `T`; at `std::complex` that `operator*` is Annex-G conformant — an `isnan` branch and a
`__mulsc3`/`__muldc3` call per multiply. `register_64x64_k16_wide.hh` keeps `std::complex` out of device code but reads A as m×k and B as
k×n and is NN only. The new header is the intersection: the wide kernel's POD scalar, 16-byte fragment granule and unpadded shared stride,
with the transposed operand staged through shared by a transposed **store** rather than a transposed global read, so the global reads stay
coalesced in all four forms. Conjugation is applied once per element at the staging store and costs **nothing per FMA**.

**The macro tile is a parameter because the demand is not square.** Read off the drivers, not off a square benchmark grid:

| driver | call | logical shape | form | (nb, W) by scalar |
|---|---|---|---|---|
| `potrf_blocked.cc:370` | `A22 -= L21 L21^H` | m_trailing × W × nb | `NoTrans/ConjTrans` | float (128,128), double & cfloat (96,32), cdouble (64,16) |
| `potrf_blocked.cc:357` | the W×W diagonal fold | W × W × nb | `NoTrans/ConjTrans` | same |
| `geqrf_blocked.cc:277` | `W1 = V^H A22` | nb × n2 × m_panel | `ConjTrans/NoTrans` | nb = 32, or 16 for double |
| `geqrf_blocked.cc:282` | `W2 = T^H W1` | nb × n2 × nb | `ConjTrans/NoTrans` | same |
| `geqrf_blocked.cc:286` | `A22 -= V W2` | m × n2 × nb | `NoTrans/NoTrans` | already served by the NN wide kernel |

Every complex transposed shape in the tree has a dimension of 16 or 32. The tiles match a panel width instead:
`64x64x16wide_cn` and `64x64x16wide_nc` (general), `128x32x16wide_nc` (potrf trailing + fold at W = 32), `32x128x16wide_cn`
(geqrf `W1` and `T^H W1` at nb = 32). Work-group 256 for all four; shared per work-group is `(M·K + K·N)·sizeof(T)`, 40 KB at cdouble for
the rectangular tiles, under the 48 KB hole in every case.

**One instantiation serves both real and complex transposes.** `wide_trans_matches<T>` licenses a single widening: for a *real* scalar
conj is the identity, so a `ConjTrans` instantiation is a correct `Trans` — which is what lets one variant serve `potrf_blocked.cc`'s
`kTrailingTransB<T>` (ConjTrans for complex, Trans for real) and holds the count to 4 × 4 = 16. For complex the substitution is refused and
the call falls back to `Tiled16`.

#### The grid

Ratios are **in time**, `arm_ms / native_ms`; > 1 means the tile wins. Batch 1024 unless stated. `ld` is padded on every operand
(`+8`) except where the driver packs it (geqrf's V and W1), matching what each caller actually hands `gemm`.

**NC, the potrf trailing shape (m × 32 × k, `NoTrans/ConjTrans`), `128x32x16wide_nc`:**

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

**CN, the geqrf panel shape (32 × n × k, `ConjTrans/NoTrans`), `32x128x16wide_cn`:**

| type | n=64 k=128 | 64,512 | 256,32 | 256,128 | 256,512 | 512,128 | 512,512 | 128×256×512 |
|---|---|---|---|---|---|---|---|---|
| cdouble vs cuBLAS | 0.570 | 0.563 | 1.130 | 1.124 | 1.126 | 1.127 | 1.128 | 1.129 |
| cdouble vs Tiled16 | 1.759 | 1.762 | 3.422 | 3.501 | 3.531 | 3.515 | 3.536 | 3.536 |
| cfloat vs cuBLAS | 0.909 | 0.847 | 1.066 | 0.952 | 0.915 | 0.974 | 0.932 | 0.639 |
| cfloat vs Tiled16 | 1.567 | 1.739 | 1.291 | 1.920 | 2.507 | 2.110 | 2.755 | 3.718 |

**Read the two rows against each other and the whole result is there.** Against `Tiled16` the tile wins broadly, for both complex types.
Against cuBLAS only complex<double> ever clears R8's 1.11× bar, complex<float> tops out at 1.069×, and `double` **loses to `Tiled16`**
(0.92–1.00×) — which is already 1.03–1.11× of cuBLAS on these shapes, so for `double` there is nothing to win and the selector leaves it
alone. That double row is the clearest single negative result in the grid.

#### Saturation, and where it is not reached (R8a)

The ratio rises with batch and does not reach a fixed point inside the memory ceiling, so it is quoted with its direction and the batch it
was read at. cdouble NC m=512 k=96: **1.099 → 1.120 → 1.134** at batch 64 → 256 → 1024 (still rising, +1.2% on the last doubling).
cdouble CN 32×256×512: **1.052 → 1.112 → 1.126**. cfloat NC m=512 k=96 vs Tiled16: **2.881 → 1.984 → 2.056** — not monotone, and the
batch-64 cell is a small-work cell, not a saturated one. Every ratio quoted above is the batch-1024 reading.

#### Bracketing non-winners, all measured

| edge | cell | vs cuBLAS | vs Tiled16 |
|---|---|---|---|
| m below the 128-row NC tile | cdouble 32×32×96 (the potrf W×W fold) | 0.283 | 0.857 |
| n below the 32-col NC tile | cdouble 512×16×64 (the shape potrf cdouble issues) | 0.581 | 1.789 |
| n below the 128-col CN tile | cdouble 32×64×512 | 0.563 | 1.762 |
| m below the 32-row CN tile | cdouble 16×512×512 | 0.567 | 1.773 |
| k → 1 | cdouble 512×32×1 | 0.346 | 0.563 |
| k = 8 (the other side of that edge) | cdouble 512×32×8 | 1.163 | 1.884 |
| batch below saturation | cdouble 128×32×96 at batch 64 | 0.569 | 1.707 |

**No high-side bracket exists.** cdouble at 512³ batch 256 and 1024×1024×512 batch 128 both still read 1.13× of cuBLAS and 3.5× of
Tiled16, in both transposed forms. The window has no measured upper bound in m, n or k — the same debt the `double` window already carries.

#### The window that was refused, and the grid defect that produced it

A `preferred()` window for complex<double> — the transposed forms, filled tiles, `k >= 8`, `ctas >= 1024` — **was written, built, tested and
then withdrawn.** Its kernel evidence was the table above (every admitted cell ≥ 1.11×, every refused cell a measured loser). It failed the
end-to-end gate and then failed its own re-measurement:

| op | type | n | batch | route unchanged (ms) | window open (ms) | ratio |
|---|---|---|---|---|---|---|
| geqrf | cdouble | 256 | 1024 | 163.42 | 168.24 | **0.9714** |
| geqrf | cdouble | 512 | 1024 | 889.72 | 892.56 | 0.9968 |
| geqrf | cdouble | 512 | 256 | 227.68 | 228.27 | 0.9975 |
| geqrf | cdouble | 256 | 256 | 43.96 | 43.94 | 1.0005 |

Every other cell of the before/after (potrf and geqrf, cfloat / cdouble / double, n ∈ {256, 512}, batch ∈ {256, 1024}) moved by less than
0.7%. A `BATCHLAS_KERNEL_TRACE` run confirms the kernel really did run — 24 launches of `gemm_sycl_register_32x128_k16_wide_cn` inside one
`geqrf cdouble 256` call — so this is not a window that failed to fire.

**The cause is a defect in the grid, not in the kernel: every `n` in the CN sweep and every `m` in the NC sweep was an exact multiple of the
tile's own macro dimension.** 64, 128, 256, 512, 1024 against a 128-wide tile. The drivers issue `n2 = m - j2`, i.e. 224, 192, 160, 128, …,
which leave a mostly-empty trailing column tile. Re-measured at those:

| cdouble CN, 32 × n × 256, batch 1024 | n=136 | 160 | 192 | 224 | 288 | 384 | 480 |
|---|---|---|---|---|---|---|---|
| vs cuBLAS | 0.637 | 0.707 | 0.846 | 0.985 | 0.847 | **1.125** | 1.057 |
| vs Tiled16 | 1.990 | 2.209 | 2.646 | 3.080 | 2.650 | 3.521 | 3.308 |

| cdouble NC, m × 32 × 96, batch 1024 | m=136 | 160 | 200 | 224 | 288 | 384 | 480 |
|---|---|---|---|---|---|---|---|
| vs cuBLAS | 0.718 | 0.718 | 0.996 | 0.995 | 0.855 | **1.133** | 1.065 |
| vs Tiled16 | 1.988 | 2.209 | 2.859 | 3.078 | 2.649 | 3.515 | 3.308 |

The "1.13× window" is visible **only at exact tile multiples**. cuBLAS's ratio is flat in raggedness because it tiles the output
differently; this kernel pays the full cost of a quarter-full trailing tile. `t_native <= 0.90 t_vendor` is therefore not met on the
population the drivers actually generate, and the route is left alone. cfloat is the same story one notch lower (0.824–0.958 ragged).

**Against `Tiled16` raggedness costs almost nothing** (1.68–3.52× cdouble, 1.68–2.42× cfloat across the same ragged sweep), which is why the
**selector** row ships and the **route** row does not. `Tiled16` is one accumulator per thread; a partly-empty 128-wide tile still does far
more work per load than that.

The general lesson, and it generalises past this kernel: **a macro-tiled kernel must be swept at sizes that are NOT multiples of its own
tile**, or the sweep measures the kernel's best case and calls it the average. Bracketing on size alone does not catch it — every cell in the
first grid was bracketed, and every bracket was itself tile-aligned.

#### What ships

`select_kernel_variant` (`src/sycl/gemm_kernels.cc`), complex only, via `wide_transposed_tile_for` in `gemm_kernels.hh`:
`NoTrans/ConjTrans` with `m >= 128 && n >= 32` → `128x32x16wide_nc`; `ConjTrans/NoTrans` with `m >= 32 && n >= 128` →
`32x128x16wide_cn`; both need `k >= 8` and `ctas >= 64` for the tile in question. Everything else is `Tiled16`, unchanged.

This is reachable in a **vendor-free or ROCm build**, and under a forced `BATCHLAS_GEMM_VARIANT=sycl` — `preferred()` refuses complex, so a
complex shape in a cuBLAS build never reaches it. That is the honest scope: the deliverable is the vendor-independence build, worth
**1.68–3.52×** there, and nothing at all in the vendor build.

#### Register residency

`scripts/register_probe.sh` (`BATCHLAS_BUILD_DIR=build/presets/dev-tests`, target `batchlas_sycl`): 576 entry functions, **0 with non-zero
spill**, and all 16 `GemmWideTransposedKernel` instantiations present in the cubin — which is the check that the instantiation budget was
actually spent, not merely written. Per instantiation:

| tile / form | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 64×64 CN | 43 | 66 | 72 | 128 |
| 64×64 NC | 45 | 64 | 72 | 132 |
| 128×32 NC | 47 | 64 | 72 | 124 |
| 32×128 CN | 53 | 60 | 72 | 128 |

Zero spill everywhere, and the profile tracks the NN wide kernel it was ported from (55/72/72/132). The cdouble column at 124–132 caps a
work-group at 512 work-items, i.e. 33% occupancy — the same ceiling the NN kernel runs at.

#### R9: the armed breaks

The kernel's own breaks, run against `gemm_tests --gtest_filter='GemmTest/*.WideTransposed*'` (36 live cases: 9 tests × 4 types on CUDA):

| break | expected red | observed red |
|---|---|---|
| 1. transposed A staging forms the `NoTrans` address | the CN tests, all 4 types (12) | **14** — the 12, plus `RealTransWideningOnALeg` for float and double, which also reaches the CN tile |
| 2. transposed B staging forms the `NoTrans` address | the NC tests (12) | **14** — same +2, on the NC widening test |
| 3. drop `dev_conj` on A | the CN tests, complex only (6) | **6**, exactly |
| 4. drop `dev_conj` on B | the NC tests, complex only (6) | **6**, exactly |
| 5. drop the epilogue `col >= n` guard | 15 | **18** — the two n = 32 potrf shapes stay green (n = TileN, no edge), correctly; the replica's count was low by 3 |
| 6. drop the epilogue `row >= m` guard | 16 | **18**, and `NC128x32PotrfTrailingShape` stays green at **beta = 1** exactly as predicted: out-of-range rows accumulate zero, so the epilogue writes `prior` |
| 7. swap the transposed-store index decomposition (**the break the plan names**) | 0 — it cannot go red | **0**, confirmed on hardware |
| 8. drop the complex refusal in `wide_trans_matches<T>` | the 2 widening tests, complex only (4) | **4**, exactly |
| 9. drop the staging bounds test on A | *(new; not predicted)* | **0** |

Break 7 is a **performance** defect that no correctness test in this design can see: the staging loop is a full cover of the tile, so
swapping the decomposition writes the same (i, kk) set and only changes which lane fetches which element. The answer is bit-identical; the
global reads stop being coalesced. It has to be caught by an `ncu` sector count, not by `ctest`.

Break 9 is the one finding from arming that was not predicted at all. Dropping `if (gm < m && gk < k)` on the A staging leg changes nothing,
because the **B** staging leg's own bounds test still writes a zero for every out-of-range `kk`, and the product `af * 0` kills the garbage;
out-of-range **rows** are dropped by the epilogue guard. So the A-side test is redundant for correctness whenever the B-side one holds — it
earns its place only as an out-of-allocation read guard, which a test whose operands are sub-views of a wider parent can never exercise.
Two tests defending one property is a guard you cannot arm; recorded rather than removed.

The dispatch breaks, against `gemm_tests --gtest_filter='GemmDispatchPolicyTest.*'` plus `route_gemm_equivalence_tests` (these were run
against the version that still carried the refused route window, which is why two of them name route tests that no longer exist):

| break | expected red | observed red |
|---|---|---|
| A. the selector row never fires | `ComplexTransposedTakesTheWideTransposedTile` | that, **plus** the route-inside-selector invariant — correctly: a routed shape landing on `Tiled16` was the 3.5× regression that test existed for |
| B. the NC gate stops requiring a filled 128-row tile | `WideTransposedSelectorRefusesEveryMeasuredLoser` | exactly that |
| C. the route CTA floor drops below the selector's | the subset invariant | that, plus `ComplexTransposedTakesTheWideTransposedTile` |
| D. the route window opens for complex<float> | `...IsComplexDoubleOnly` | that, plus `RouteGemmEquivalence.ComplexFloat` |
| E. the route window narrows until the equivalence grid misses it | `RouteGemmEquivalence.ComplexDouble` (the non-vacuity guard) | that, plus `...IsComplexDoubleOnly` |
| F. the `k >= 8` rank-1 exclusion is dropped | `WideTransposedSelectorRefusesEveryMeasuredLoser` | exactly that |

#### Discarded cells, and what could not be established

One cell of 244 tripped the `rel_sd < 10%` gate and is excluded: **float 512×32×96 batch 256, arm `64x64x16wide_nc`, rel_sd 0.172.** Its
`128x32x16wide_nc` and vendor arms in the same process were at 0.002 and 0.004, so it is that arm on that cell, not the cell.

Not established:

* **The A/B harness these numbers came from is NOT in the tree.** `benchmarks/` has no gemm binary that can pin a `KernelVariant` per arm
  *and* interleave the arms in one process, and `factor_bench.cc` covers factorizations only. The harness was a standalone `.cc` compiled
  against `include/` + `build/presets/dev-tests/include/` and linked against the 14 component `.so`s (the recipe `docs/perf/level3.md` uses),
  structured on `factor_bench.cc`'s rules. It was deliberately not added to `benchmarks/CMakeLists.txt`, because the `dev-tests` preset has
  `BATCHLAS_BUILD_BENCHMARKS=OFF` and adding a source no build in the loop compiles is how this repo acquired 321 never-compiled lines. Until
  it is in-tree and built, **every cell on this page is reproducible only by rebuilding that harness**. It should become
  `benchmarks/gemm_ab_bench.cc` in a change that also runs the `benchmarks` preset.
* **No `ncu` reading was taken.** The occupancy check for the 40 KB cdouble tiles is outstanding, and break 7 has no coalescing
  measurement behind it. Every timing on this page is wall-clock.
* **The 16-wide tiles are still not built**, so potrf complex<double> (W = 16) and geqrf double (nb = 16) reach no register kernel. The
  n = 16 cell measured **1.789× of Tiled16** — a win the vendor-free build is leaving on the table — but on one cell, and potrf's `(nb, W)`
  and a 16-wide tile are one coupled piece of work.
* **No upper bound on m, n or k is measured** for the selector window; the largest cells are 1024×1024×512.
* **The `k` edge is bracketed at 1 and 8 only**; `k` in 2..7 is unmeasured and refused.
* **`float` transposed is not routed to these tiles**, although it beats `Tiled16` by 1.25–2.71× on the NC shapes. It was not A/B'd against
  the existing `Tiled128x32RegisterK32{TN,NT,TT}` family, which owns `m >= 128 && n >= 32 && k >= 128` for float, and a selector row that
  cannot say which of the two is better is not a row worth writing.
* **The plan's premise that "float transposed is 0.23–0.55× across 48 of 48 cells" does not describe this demand.** On the panel shapes the
  drivers issue, float native reads **0.71–1.24×** of cuBLAS. That figure came from a square grid.
* **The plan's acceptance bar ("≥ 0.9× cuBLAS") is on the wrong side of R8 by 1.23×.** R8 needs `t_native <= 0.90 t_vendor`, i.e. ≥ 1.11×.
  0.9× cuBLAS is 1.11× the vendor *in time*. The bar as written would have passed this kernel's cfloat cells; R8 does not.

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

## The subgroup workspace budget

This is the one constant on this page that is fixed at **configure** time, by CMake, and it is not a measured tuning result — it is a
compatibility constraint with a measured cost. It is recorded here because `cmake/BatchLASDetectSYCL.cmake` cites this section, and because the
whole of the claim it cites is negative: **nothing in this tree has ever timed any budget but the one each architecture already ships.**

### What it is and where the number comes from

`kSubgroupWorkspaceBudgetBytes` (`include/batchlas/blas/device/detail/group_blas_subgroup_common.hh:58`) is
`device_limits::subgroup_workspace_budget_bytes()`, generated from `cmake/device_limits.h.in`. CMake derives it per architecture in
`batchlas_subgroup_workspace_budget_bytes()` as **that architecture's table local memory less a 4 KiB reserve**, with a 16 KiB floor:

| architecture | table local mem | budget |
|---|---|---|
| `nvidia_gpu_sm_*` | 49,152 | **45,056** |
| `amd_gpu_gfx*` | 65,536 | **61,440** |
| `intel*` | 65,536 | **61,440** |
| unrecognised GPU | 32,768 | **28,672** |

It is deliberately **not** the device's real local-memory capacity. On sm_89 that is 101,376 bytes, and a probe reads it; letting the probe reach
this constant would raise the budget by 2.25x and retune five ops as a side effect of fixing a capacity table. Every run-time capacity asks
`DeviceProperty::LOCAL_MEM_SIZE` instead (`src/util/resident_capacity.hh`).

### The five gates it drives

Five `if constexpr` predicates compare a workspace struct against the budget, and between them decide which tile variants are compiled in at all:

| predicate | consumed by |
|---|---|
| `register_matrix_workspace_supported_v` | `group_blas_gemm.hh:399`, `group_blas_rankk.hh:371,387,409,445` |
| `complex_rank2k_workspace_supported_v` | `group_blas_rankk.hh:367,383,403,439` |
| `complex_rank2k_in_kernel_workspace_supported_v` | staged rank-2k path |
| `optimized_gemm_workspace_supported_v` | `group_blas_gemm.hh:390` |
| `gemm_workspace_supported_v` | `group_blas_gemm.hh:361,369` |

`group_blas_rankk.hh` is the shared body behind **symm, herk, syrk and syr2k**, so a change here moves five ops, not one.

### What moving it actually changes (measured)

Compiled with `/opt/dpcpp-cuda/bin/clang++ -fsycl -std=c++20` against the real header, with only the generated
`MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES` substituted. `sizeof` in bytes; `1`/`0` is the gate.

| budget | type | regmat | c2k | c2kik | optgemm | gemm | gates (regmat/c2k/c2kik/optgemm/gemm) |
|---|---|---|---|---|---|---|---|
| 28,672 | float | 25,216 | 16,768 | 12,640 | 41,472 | 41,472 | 1 1 1 **0 0** |
| 28,672 | double | 25,088 | 27,200 | 25,280 | 82,944 | 82,944 | 1 1 1 0 0 |
| 28,672 | cfloat | 25,088 | 27,200 | 25,280 | 82,944 | 82,944 | 1 1 1 0 0 |
| 28,672 | cdouble | 41,728 | 37,504 | 28,160 | 165,888 | 165,888 | **0 0** 1 0 0 |
| 45,056 | float | 25,216 | 16,768 | 12,640 | 41,472 | 41,472 | 1 1 1 **1 1** |
| 45,056 | double | 41,984 | 33,536 | 25,280 | 82,944 | 82,944 | 1 1 1 0 0 |
| 45,056 | cfloat | 41,984 | 33,536 | 25,280 | 82,944 | 82,944 | 1 1 1 0 0 |
| 45,056 | cdouble | 41,728 | 41,728 | 44,160 | 165,888 | 165,888 | 1 1 1 0 0 |
| 61,440 | float | 25,216 | 16,768 | 12,640 | 41,472 | 41,472 | 1 1 1 1 1 |
| 61,440 | double | 50,432 | 33,536 | 25,280 | 82,944 | 82,944 | 1 1 1 0 0 |
| 61,440 | cfloat | 50,432 | 33,536 | 25,280 | 82,944 | 82,944 | 1 1 1 0 0 |
| 61,440 | cdouble | 58,624 | 58,624 | 50,560 | 165,888 | 165,888 | 1 1 1 0 0 |

Two things to read off it.

**The gates are not the whole story.** The struct sizes move with the budget, because `subgroup_limit_for_workspace_v` spends the budget on
staging depth. Between 45,056 and 61,440 no gate flips at all — and yet:

| budget | float regmat/c2k/c2kik | double | cfloat | cdouble |
|---|---|---|---|---|
| 28,672 | 8 / 8 / 8 | 2 / 5 / 8 | 2 / 5 / 8 | **0 / 0 / 1** |
| 45,056 | 8 / 8 / 8 | **6** / 8 / 8 | **6** / 8 / 8 | **1 / 2 / 6** |
| 61,440 | 8 / 8 / 8 | **8** / 8 / 8 | **8** / 8 / 8 | **3 / 6 / 8** |

(subgroups staged per work-group; the hard cap is `kMaxSubgroupsPerWorkGroup = 8`.)

So a build pinned to NVIDIA's 45,056 on an AMD or Intel part keeps every variant compiled but loses **25% of the staging width for double and
cfloat** (6 subgroups instead of 8) and **two thirds of it for cdouble** (1 instead of 3 on the register-matrix path, 2 instead of 6 on
complex rank-2k). That is the concrete content of "silently retunes five BLAS-3 ops".

**28,672 is a different kernel set, not a slower one.** The unrecognised-GPU fallback drops `optimized_gemm` and `gemm` for float, and drops the
register-matrix and complex-rank-2k paths for cdouble entirely.

### Why this is not a tuning result

**No benchmark in this tree has ever timed any budget but the shipped one.** `grep -rn 'WORKSPACE_CAP\|workspace_budget\|SUBGROUP_WORKSPACE'
benchmarks tests python examples` returns nothing: the budget is not a benchmark parameter, not a test parameter, and not reachable from the
environment. The table above is a **compile-time** census of which variants exist at each budget — it says what changes, not what it costs. Every
number on the rest of this page was measured at one budget only: 45,056, on sm_89.

Widening or narrowing the budget is therefore a separate, measured change, and it needs a harness that does not exist yet: the cap is a CMake
cache entry, so an A/B is a reconfigure and a full rebuild per arm, not a runtime flag.

### The stale-cache defect

`BATCHLAS_DEVICE_GEMM_WORKSPACE_CAP_BYTES` was a `CACHE STRING` whose historical default was the literal `45056`. When the derivation above
replaced that default, the new code was placed behind `if(... GREATER 0)` — so **any already-configured build tree kept overriding it**, and every
architecture got NVIDIA's 45,056. On NVIDIA the override and the derivation agree, which is why the first verification pass did not see it; on AMD
and Intel the cost is the 6-vs-8 and 1-vs-3 rows above.

The cache entry is now named `BATCHLAS_DEVICE_GEMM_TILE_CAP_BYTES` and defaults to `0` ("derive per architecture"). The old name is migrated at
configure time: a cached legacy `45056` is dropped with a status line, and any other cached value is carried across to the new name with a
deprecation warning, so a deliberate override is never silently lost.

## Open debts

* **Complex is still vendor-dependent in a cuBLAS build, and that is now a measured result rather than a gap.** The transposed wide-scalar
  kernel exists (`register_wide_transposed.hh`), is tested, and is **selected** for complex transposed shapes — but only against `Tiled16`,
  because a `preferred()` window for complex<double> was measured and **refused**: it cleared R8 only at tile-aligned sizes and ran at
  0.64–0.99× of cuBLAS on the ragged `n` the blocked drivers actually issue. The vendor-free and ROCm builds gain 1.68–3.52×; the vendor
  build gains nothing. The 16-wide tiles that potrf complex<double> and geqrf double need are still not built, and the n = 16 cell measured
  1.79× of `Tiled16` unexploited. See above.
* ~~`scripts/gemm_demand.py`'s `preferred()` replica has drifted~~ — **paid**. `scripts/gemm_demand.py:50-77` now transcribes the shipped
  predicate exactly (float NN square `max_dim <= 32`; double `k >= 2`, no squareness, no upper bound). Verified against
  `route_gemm.hh` rather than re-asserted.
* **The double window deliberately reaches past its measurements.** No upper size bound; largest measured 2048³. The FP64-ceiling argument is an
  argument, not a measurement, above 2048.
* **`scripts/route_diff.sh` records resolver `Route`s, not `KernelVariant`s**, so it is structurally blind to every selector change on this
  page. `GemmDispatchPolicyTest` now covers the complex transposed selector window and both its edges, but a regression in the float CTA gate
  or the `mn_min` gate is still **completely silent**, visible only as the suite getting slower.
* **`SelectSyclKernelVariantForTest` hard-codes `Matrix<float>` and `select_kernel_variant<float>`,** so every pre-existing
  `GemmDispatchPolicyTest` is float-only: the double `max_dim <= 24` boundary, the complex CTA gate and the wide kernel's `min_dim >= 256` arm
  still have **no dispatch-policy test**. Same blind-by-construction shape as the 18×14×12 ConjTrans case. A templated
  `SelectSyclKernelVariantForTestT<T>` now sits beside it and is used by one assertion (that complex transposed shapes still fall to `Tiled16`);
  porting the existing float assertions onto it is the rest of the debt.
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

## The small batched kernel

**2026-09-26.** Float NN at `max(m, n, k) <= 48` ran `Direct`: one work-item per output,
an 8 x 8 group whose FAST index is the column, so adjacent lanes read B and write C
`ld` apart; the transpose switch sat in the k loop and C was read even at beta = 0. At
n = 32, batch 32768 that is 1.75 ms -- 0.22 TB/s of a ~400 MB problem, 0.61x cuBLAS.
33..48 fell to `Tiled16`, worse still (n = 48: 6.09 ms, 0.25x).

`src/sycl/gemm/small_batched.hh` (`KernelVariant::SmallBatched`, forced as `small`): a
128-lane group holds 128 / (2 * NB) matrices, NB in {8, 16, 32, 64} from max(m, n, k);
op(B) is staged in local memory (odd ld, storage-order walk, so either transpose reads
coalesced); each lane holds one row of op(A) in registers and NB / 2 columns of C;
lanes run down the rows, so A loads and C stores are coalesced; beta = 0 skips the C
read. Real scalars only -- `std::complex`'s `operator*` is the Annex G trap -- so a
complex call forced to `small` runs `Direct`.

Batch 32768 unless noted, `BM_GEMM` square NN, beta = 0, ms:

| n | cuBLAS | before | small | vs cuBLAS |
|---:|---:|---:|---:|---:|
| 8 | 0.175 | 0.024 (Direct) | 0.021 | 8.5x |
| 16 | 0.358 | 0.235 (Direct) | 0.133 | 2.69x |
| 24 | 1.008 | 0.842 (Direct) | 0.361 | 2.79x |
| 32 | 1.161 | 1.747 (Direct) | 0.608 | **1.91x** (was 0.64x) |
| 40 | 1.267 | 3.670 (Tiled16) | 1.474 | 0.86x (was 0.35x) |
| 48 | 1.524 | 6.087 (Tiled16) | 1.741 | 0.88x (was 0.25x) |
| 64 | 2.419 | 2.612 (32x32 reg) | 2.365 | **1.02x** (was 0.93x) |

Routed for float: every `max_dim <= 32` shape (NN and transposed), and NN squares-ish
with `min_dim > 32` up to 64. Non-square shapes above 32 keep their previous kernels:
the panel-update shapes (large m, n, small k) are unmeasured here, and the kernel pads
m and n to the bucket. double measured at parity with Direct / Tiled16 (fp64 is
compute-bound at 1/64 rate) and is not routed. `preferred()` is untouched: only float
NN `max_dim <= 32` was already native.

The NB = 64 bucket is shared-load bound (one broadcast `ld.shared` per FMA); 40..48
still lose to cuBLAS.
