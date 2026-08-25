# WP7 — native vs vendor `gemv`, measured

What this measures: the **shipped public `batchlas::gemv`** (`src/dispatch/entry_points/level3.cc`)
in the **vendor-present** build, with the two arms selected by `BATCHLAS_GEMV_ROUTE`
and the **resolved route printed as a column**. A kernel being linked is not
evidence it ran; the route column is.

Ratio convention throughout: **`vendor_ms / native_ms`**, so `1.00` is parity and
`> 1` means the native kernel is faster.

The "default native route" is the one `RouteTable<Op::gemv, T>` actually picks in
a vendor-free build:

| `transA` | default native route | why |
|---|---|---|
| `NoTrans` | `native:direct` | there is no `NoTrans` CTA body — one work-item per output row is already fully coalesced |
| `Trans`, `ConjTrans` | `native:cta` | one 32-lane sub-group per output element; `Direct` is the fallback for devices with no enumerated sub-group size 32 |

---

## HEADLINE

**Vendor-freedom at parity, with one large measured exception and two named weaknesses.**

* **84 default-route cells** (4 scalar types × 3 `transA` × 7 shapes, saturating
  batch, DRAM-resident), **two independent passes**: `relerr` is **exactly 0** on
  every one, and the worst is **0.75×**.

> **CORRECTION, from the WP7 performance audit and the repair pass that followed.**
> This section originally read “**no cell is below 0.50×**”. That was true OF THIS
> GRID and false of the operation: **the minimum `out_len` anywhere in the seven
> shapes here is 64**, and the sub-0.50× family lives BELOW the 32-lane warp
> width. `experiments/wp7_gemv/audit/parity_p{1,2}.csv` resolves the same
> question on an (`out_len`, `red_len`) ladder that reaches `out_len = 1` and
> finds **15 cells below 0.50×**, worst **0.08×** — 13 of them one family
> (`Direct`, `NoTrans`, `out_len < 32`), which is **structurally absent from this
> grid**. That family has since been FIXED (body 4, the segmented NoTrans body —
> see `../repair/README.md`); the remaining two are the `complex<double>`
> short-reduction weakness already named below. A grid that cannot reach a regime
> is not evidence about it.
* 71 of the 84 sit between **0.95× and 1.05×** — i.e. at the DRAM roof, where
  the recon phase already established there is nothing to win.
* **`preferred()` ships ALL-FALSE.** A vendor-present build sends `gemv`
  nothing. This is a measured result, not a placeholder: see
  [Why no `preferred()` clause](#why-no-preferred-clause), which includes a
  hypothesis that was **built, tested out-of-sample and refuted**.
* The **one prize the recon phase identified is real and is claimed**:
  `complex<double>` + transposed, at large footprint, is **1.88×–3.07×** faster
  natively, reproducing across two passes to within 0.03×. It is reachable today
  with `BATCHLAS_GEMV_ROUTE=native:cta` — **for transposed GPU shapes only.** On
  a `NoTrans` shape, on a CPU device, or on a GPU that does not enumerate a
  sub-group size of 32, that same pin **silently measures the vendor** in a
  vendor-present build (`supports(CTA)` is false, so `resolve_route` falls
  through to `automatic()`, which with `preferred()` all-false IS the vendor) and
  silently measures `native:direct` in a vendor-free one. Nothing is printed
  either way, and a misspelled value behaves identically. **The resolved-route
  column is the only way to know which arm ran.**

## The 84-cell parity table (both passes)

Produced by `analyse.py ab_p1.csv ab_p2.csv`. Worst cell per `(type, transA)`
over the seven shapes:

| `transA` | float | double | complex&lt;float&gt; | complex&lt;double&gt; |
|---|---|---|---|---|
| `NoTrans` (Direct) | 0.99 | 0.96 | 0.96 | 0.95 |
| `Trans` (CTA) | 0.98 | 0.97 | 0.97 | **0.75** |
| `ConjTrans` (CTA) | 0.98 | 0.97 | 0.96 | **0.75** |

and the best cells, all `complex<double>` transposed: **3.01× / 2.87×** at
256×256×1024 and **2.02× / 1.92×** at 64×2048×1024.

`ConjTrans` had **zero test coverage and zero measurement** in this tree before
WP7 and is the live production path — `ortho.cc` selects it for all four complex
types. It measures within 0.01× of `Trans` everywhere, which is what a single
sign flip should cost.

## The `complex<double>` transposed region, mapped

`refine.sh` sweeps m × n at a **fixed ~1 GB footprint** so only SHAPE varies
(a cell that fits in the 72 MB L2 measures L2 bandwidth and means nothing).
`refine_analyse.py refine_c_p1.csv refine_c_p2.csv`, cross-pass median,
`transA = ConjTrans`:

```
   m\n       64     128     256     512    1024    2048
    48     0.48    0.58    0.63    0.62    0.66    1.00
    64     0.74    1.90    1.90    1.89    1.89    2.01
    96     1.01    0.99    2.69    2.69    2.69    2.71
   128     1.10    1.05    2.82    2.82    2.84    2.84
   192     0.99    1.11    2.95    2.97    2.99    2.98
   256     0.97    1.07    3.03    3.03    3.03    3.02
   320     1.09    1.14    3.04    3.06    3.05    3.06
   384     1.16    1.15    1.03    1.15    1.16    3.07
```

`transA = Trans` (`refine_t_p1.csv`) reproduces the same surface to within 0.2×.

**The axis is `m`, and `m` is `red_len()` under a transposed `transA`, NOT
`out_len()`.** A predicate written on `out_len()` would test `n`, never touch
`m`, and invert the window. On THIS SLICE the clean rectangle is

> `64 <= m <= 320` **and** `n >= 256` — **24 cells, every one ≥ 1.89×**,
> cross-pass spread ≤ 0.06×, `relerr` exactly 0.

`m = 48` is outside it in the other direction and **loses** (0.47–0.66×), and
`n = 128` is mixed (1.90× at `m=64`, 0.98× at `m=96`).

> **CORRECTION.** That rectangle is a **fixed-footprint slice, not a shippable
> window**, and the next section is why. Because `refine.sh` holds A at ~1 GB,
> `batch` moves inversely with shape and the batch axis is invisible here.
> Resolved on an (m, n, **batch**) grid — `../audit/prize_p{1,2}.csv`, 396 rows
> × 2 passes — the same rectangle **admits 17 cells below 1.00×, worst 0.36×**:
> at `m=128, n=256, transA=Trans` the ratio is 0.52 at batch 128, 0.99 at batch
> 256 and 2.35 at batch 512. Quote the rectangle as a description of this slice,
> never as a candidate predicate.

## Why no `preferred()` clause

The 24-cell rectangle above satisfies the acceptance gate — ≥ 1.15× median,
reproduced across two independent passes. It is still not shippable, and the
reason is a **third axis the rectangle does not see**.

`batchdep.sh` walks the batch down at two in-rectangle shapes:

| shape | batch | A | vendor GB/s | native GB/s | ratio p1 / p2 |
|---|---|---|---|---|---|
| 256×256 | 256 | 268 MB | — | — | 0.99 / 0.98 |
| 256×256 | **512** | 537 MB | — | — | **2.85 / 2.83** |
| 64×2048 | 32 | 67 MB | — | — | **0.36 / 0.36** |
| 64×2048 | **64** | 134 MB | — | — | **1.97 / 1.98** |

So cuBLAS's dip switches on **inside the rectangle**, at 537 MB for one shape and
134 MB for the other. It is therefore neither a footprint threshold nor an L2
boundary (the 72 MB L2 is far below both, and 268 MB shows no dip at all) —
which is exactly why **B4 forbids an L2-residency gate**, and the data agrees.

### The hypothesis that was built, tested and refuted

All four transition points straddle the same value of **`n * batch`** — the number
of output elements — between 65,536 and 131,072. That is a clean, checkable rule
over fields the shape actually carries, so it was **predicted and then tested on
two shapes it was not fitted on** (`outelems.sh`):

| m | n | batch | `n*batch` | A | vendor GB/s | native GB/s | p1 | p2 | prediction |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 512 | 128 | 65,536 | 134 MB | 894.8 | 859.7 | 0.96 | 0.96 | lose/tie ✔ |
| 128 | 512 | **256** | **131,072** | 268 MB | 924.7 | 899.7 | **0.97** | **0.97** | WIN ✘ |
| 128 | 512 | 512 | 262,144 | 537 MB | 331.8 | 918.0 | 2.77 | 2.78 | WIN ✔ |
| 64 | 256 | 256 | 65,536 | 67 MB | 774.4 | 671.3 | 0.87 | 1.02 | lose/tie ✔ |
| 64 | 256 | **512** | **131,072** | 134 MB | 405.0 | 686.0 | **1.69** | **1.64** | WIN ✔ |
| 64 | 256 | 1024 | 262,144 | 268 MB | 371.8 | 699.8 | 1.88 | 1.87 | WIN ✔ |

**Refuted.** At `m=128, n=512, batch=256` the rule predicts a win and cuBLAS is at
924.7 GB/s — at the roof, no dip at all. Fitting the threshold on two transitions
and shipping it would have moved that cell, and every cell like it, onto a route
that is 0.97× — and, one batch lower, onto cells measuring 0.54×.

**Conclusion: no predicate over `(scalar, transA, m, n, batch)` separates win from
loss with the data in hand, so `preferred()` ships all-false.** The win is not
lost — it is one environment variable away, documented above — and mapping the
third axis is a bounded follow-up, not a redesign.

## The two known weaknesses (B6 requires these be stated, not passed silently)

Neither is reached by default in a vendor-present build, because `preferred()` is
all-false. Both are reached in the **vendor-free** build, where the alternative is
not a slower kernel but **no kernel at all**.

### 1. `complex<double>` transposed with a SHORT reduction: 0.43–0.76× of cuBLAS

> **WIDENED by the audit.** The table below is the ~1 GB slice and lists only
> `m ∈ {48, 64}`. The measured boundary is larger: `../audit/prize_p{1,2}.csv`
> shows that at **batch 128 the whole `m <= 128` column loses**, 0.27–0.60×
> across `n ∈ {128, 256, 512}` (`m=32`: 0.50/0.39/0.31; `m=48`: 0.43/0.34/0.28;
> `m=64`: 0.60/0.45/0.36; `m=128`: 0.59/0.52/0.99), and that `red_len = 64` at
> batch 512 is **0.43–0.46×** — two cells below the 0.50× blocker line. State it
> as: **`complex<double>` transposed loses below ~1.0× for `red_len() <= 64` at
> any `n`, and for `red_len() <= 128` whenever `batch <= 128`.**
>
> The audit also checked the mechanism story below rather than accepting it. It
> is directionally right and it is NOT occupancy or coalescing: sweeping
> `red_len` at m=256, batch=512, ncu holds occupancy at 82–95% and the grid
> constant while DRAM throughput climbs 38.5% (`red_len`=32) → 42.1% (48) →
> 64.8% (64) → 81.3% (96) → 93.4% (128) → 94.0/94.4/95.6% (192/256/512). But the
> exact `r/(r+c)` fit does NOT hold — the implied constant falls from 1.6 to 0.28
> across the ladder — so quote it as a **measured curve, fully amortised by
> `red_len = 128`**, not as a model.

DRAM-resident, ~1 GB, `transA = ConjTrans`:

| m | n | batch | vendor | native | ratio |
|---|---|---|---|---|---|
| 48 | 64 | 20345 | 978.7 GB/s | 464.9 GB/s | **0.48×** |
| 64 | 64 | 15258 | 975.6 GB/s | 720.6 GB/s | 0.76× |
| ≥128 | any | — | ~950 GB/s | 900–946 GB/s | ~1.0× |

**Mechanism, and it is arithmetic rather than a mystery.** The CTA body spends one
32-lane sub-group per output element and folds the lanes with a
`shift_group_left` ladder. The ladder is a **fixed cost per output**: 5 shift
steps, doubled to 10 for a complex scalar because the real and imaginary halves
fold separately. The useful work is `ceil(red_len/32)` rounds of loads. At
`red_len = 64` that is 2 rounds against 10 shuffles; at `red_len = 48` it is 2
rounds of which the second has only 16 of 32 lanes live (75% lane utilisation) —
and 0.48× is below even that 0.75, because the ladder does not shrink with the
round count. At `red_len >= 128` the ladder amortises over 4+ rounds and the
kernel reaches the roof.

**The named fix, still not attempted here:** serve several outputs per sub-group
when `red_len` is small — W lanes per output with `W in {8, 16}` — which cuts the
ladder to `log2(W)` steps and raises the loads in flight per warp.

> That is exactly the shape of the fix the repair pass DID land on the other
> side: `src/sycl/gemv_native.cc`'s **body 4** puts `W = 32/out_len` lanes on
> each output for `NoTrans` with a short OUTPUT, and it removed that whole
> family. The transposed short-REDUCTION case is the same idea transposed and is
> a bounded follow-up. It was not done in the repair pass because it is a fifth
> kernel body on a different arm, and body 4 had to be measured alone to be
> attributable — which is how the runtime-vs-`constexpr` `W` result (a **2×**
> difference, see `../repair/README.md`) was found at all.

### 2. L2-RESIDENT `complex<double>` transposed: 0.45–0.56× of cuBLAS

| m | n | batch | A | vendor GB/s | native GB/s | ratio |
|---|---|---|---|---|---|---|
| 64 | 256 | 128 | 34 MB | 1398.2 | 634.8 | **0.45×** |
| 128 | 512 | 64 | 67 MB | 1721.3 | 960.6 | **0.56×** |

**Mechanism.** Both cells fit in the RTX 4090's 72 MB L2, and cuBLAS's 1398 and
1721 GB/s are **above the ~1008 GB/s DRAM peak** — it is converting L2 residency
into bandwidth the native kernel never sees. The native kernel is written for the
DRAM-streaming regime: one pass over A, one sub-group per output, no blocking
that would give a second pass anything to hit in L2. Its 635–961 GB/s is the same
number it produces when A is in DRAM, which is the honest reading: the kernel is
not slower here, it is **unchanged** here while the vendor gets faster. The
recon phase's own README excludes L2-resident cells from every roof verdict for
this reason.

## Method

* `CUDA_VISIBLE_DEVICES=0`, one dedicated RTX 4090 (this box has two).
* **Saturation only**: batch ≥ 128 and A DRAM-resident on every cell of the main
  grid. An unsaturated ratio measures overhead, not the kernel.
* Arms **interleaved within one session** — vendor, `native:cta` and
  `native:direct` for the same cell run back to back — so a clock or contention
  drift would have to hit all three identically to survive.
* One process per cell; each re-warms for 1.0 s (JIT, clocks, and the first-touch
  migration of a multi-GB shared allocation — a cold first run has fabricated a
  3.7× result in this tree before), then 9 timed reps, **median** reported.
* **A correctness check runs in the same process**, against a host reference over
  items 0 and `batch-1`, so a fast wrong answer cannot enter the record. Item 0
  alone would be blind to a wrong per-item stride. Every row of every file here
  reports `relerr = 0` exactly — **840 rows across nine CSVs**.
* The harness resolves the printed route in **its own translation unit**, so it
  must be rebuilt after any `preferred()` change or the route column lies
  (campaign trap 2).

## Files

| file | what |
|---|---|
| `gemvab.cpp`, `build.sh` | the harness; links against the already-built `build/src/*.so` |
| `run.sh` → `ab_p1.csv`, `ab_p2.csv` | the 7-shape × 4-type × 3-`transA` × 3-arm parity grid, twice |
| `analyse.py` | renders it and applies the acceptance gate |
| `refine.sh` → `refine_c_p{1,2}.csv`, `refine_t_p1.csv` | the m × n map of the `complex<double>` transposed region at a fixed ~1 GB |
| `refine_analyse.py` | renders that map and computes the winning rectangle |
| `batchdep.sh` → `batchdep_p{1,2}.csv` | the batch ladder at two in-rectangle shapes — the third axis |
| `outelems.sh` → `outelems_p{1,2}.csv` | the out-of-sample test that **refuted** the `n*batch` threshold |

`gemvab_v` is a build artefact and is in `.gitignore`.
