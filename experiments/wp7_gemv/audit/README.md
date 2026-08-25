# WP7 — the performance AUDIT of the native `gemv`

Independent re-measurement of `src/sycl/gemv_native.cc` against cuBLAS, through
the shipped public API. This directory is the auditor's; `../ab/` is the
implementer's and `../baseline/` is the recon phase's. Nothing in `src/` or
`tests/` was edited to produce any number here.

Ratio convention throughout: **`vendor_ms / native_ms`**. `1.00` is parity,
`> 1` means the native kernel is faster. The "default native route" is the one a
vendor-free build actually resolves: `native:direct` for `NoTrans` (there is no
`NoTrans` CTA body) and `native:cta` for `Trans`/`ConjTrans`.

---

## Verdict in five lines

| audit item | verdict |
|---|---|
| **1. Parity gate (B6)** | **FAILS as shipped** — 15 of 192 default-route cells below 0.50×, reproduced on three passes, 13 of them one shape family. **But no shape `ortho` issues is affected**: on the 56 real production cells the worst is 0.75× and the median is 1.14×. |
| **2. B5, the occupancy fix** | **Implemented correctly and verified by the profiler** — but it does not reach the cells that matter, and the flattening has a coalescing side-effect below the warp width. |
| **3. The one prize** | **Real, and reproduces to 3 s.f.** 68 of 241 `cdouble` transposed cells win by ≥ 1.15×, the bulk of them 2.5–3.08×; the native kernel is at the DRAM roof (921–941 GB/s) and cuBLAS is at a third of it (304–370). |
| **4. D2, ConjTrans** | **Measured for the first time, as a full peer.** The native kernel treats the two spellings identically (median 0.07 %, max 3.1 % apart); what differs is **cuBLAS**. A clause may say `transA != NoTrans`, with one boundary caveat. |
| **5. `preferred()`** | **RECOMMEND ALL-FALSE.** No clause captures the win without a threshold pinned to the edge of the sampled range. One "do no harm" alternative is named below with its exact risk. |

---

## 0. Method, and one thing that went wrong

* **The GPU moved.** This box has two RTX 4090s and the campaign default is
  device 0. Partway through the first pass another agent's `syr2k_tests` was on
  device 0 (470 MB, 31% util) and produced sustained 2.2 ms rows at a 0.5 MB
  shape. That pass was **discarded**, not corrected. Everything reported here
  ran on **device 1** — identical part, identical max SM and memory clocks,
  identical power limit — and every row carries a `foreign` column counting
  compute processes on the target device at the time of the run.
  **All 1 814 rows that carry that column report `foreign = 0`,** and the parity
  sweep contains **zero rows** with a relative standard deviation above 0.3 (the
  contaminated rows had 1.4–2.7).
* **The library changed underneath the sweep.** Two rows of `prize_p1.csv` died
  with `invalid ELF header` when another agent rebuilt
  `build/src/libbatchlas_sycl.so` at 17:27; `src/sycl/gemv_native.cc` carries a
  17:24 mtime. This is disclosed rather than hidden, and it was checked rather
  than assumed: the **native arm's own timings agree across that boundary to a
  median of 1.0010 and a worst of 1.054 over 197 paired cells**, the kernel is
  still 570 lines with the same three bodies and the same work-group ladder, and
  the 15 blocker cells were **re-run a third time against the post-rebuild
  binary** (`blockers_p3.csv`) and all 15 reproduce to ±0.02×.
* Otherwise the campaign rules: arms interleaved within one cell, every arm
  pinned **explicitly** (`vendor:auto` / `native:direct` / `native:cta` — never a
  bare `native`, campaign trap 3), the **resolved route printed on every row**
  and machine-checked against the pin (**2 052 of 2 052 agree**, campaign trap 8),
  11 reps, median, and a host reference check over items 0 and `batch-1` in the
  same process. **`relerr` is exactly 0 on all 2 052 rows.** Two rows failed
  outright, both at the moment of the foreign rebuild described below; there are
  no other gaps.
* Cross-pass reproducibility: **median ratio spread 1.0042** over the 192 parity
  cells and **1.0054** over the 197 prize cells.

---

## 1. The parity gate — 15 blockers, and they are one shape family

### Why this sweep found what `../ab/run.sh` could not

The implementer's grid defines cells in `(m, n)` and is square plus two aspect
extremes. Re-deriving their gate from their own CSVs reproduces their headline
exactly — 84 cells, worst 0.75×, zero below 0.50× — and also shows why:

```
out_len values present in the implementer grid, by transA:
  transA=C : [64, 128, 256, 512, 1024, 2048]
  transA=N : [64, 128, 256, 512, 1024, 2048]
  transA=T : [64, 128, 256, 512, 1024, 2048]
```

**The smallest output length anywhere in that grid is 64.** The failure is at
`out_len < 32`. It is not that their arithmetic is wrong; it is that the region
is structurally absent from their sample — the same blindness the lead
documented in `tests/gemv_tests.cc`, in the benchmark instead of the test.

This sweep defines every cell in **`(out_len, red_len)`** and maps it to `(m, n)`
per `transA`, so a skinny cell stays skinny when the operation is transposed,
and it covers the regime `ortho.cc` issues: output length 1 to 2048 against a
reduction length of 64 to 2048, batch 128–512.

### The result, and it splits exactly at the warp width

Worst of two passes, 192 default-route cells:

| `transA` | bucket | cells | min | median | max | below 0.50× |
|---|---|---|---|---|---|---|
| `NoTrans` | `out_len < 32` | 24 | **0.08** | **0.38** | 0.91 | **13** |
| `NoTrans` | `out_len >= 32` | 40 | 0.98 | 1.00 | 1.33 | 0 |
| `Trans` | `out_len < 32` | 24 | 0.74 | 1.17 | 3.79 | 0 |
| `Trans` | `out_len >= 32` | 40 | 0.45 | 1.00 | 2.70 | 1 |
| `ConjTrans` | `out_len < 32` | 24 | 0.75 | 1.17 | 2.50 | 0 |
| `ConjTrans` | `out_len >= 32` | 40 | 0.45 | 1.00 | 2.83 | 1 |

The transposed arms are **fine** at short output length — the CTA body puts a
32-lane sub-group on each output, so `out_len` does not bound its parallelism.
`NoTrans` has no CTA body, and one work-item per output row is the whole design.

### The 15 blockers (pass 1 / pass 2, and the post-rebuild third pass)

| type | `transA` | out | red | batch | A | vendor GB/s | native GB/s | p1 | p2 | p3 |
|---|---|---|---|---|---|---|---|---|---|---|
| `cfloat` | N | 1 | 2048 | 512 | 8 MB | 1206.9 | 99.2 | **0.08** | 0.08 | 0.09 |
| `float` | N | 1 | 2048 | 512 | 4 MB | 691.6 | 96.8 | 0.14 | 0.14 | 0.15 |
| `double` | N | 1 | 2048 | 512 | 8 MB | 776.3 | 139.0 | 0.18 | 0.18 | 0.19 |
| `cfloat` | N | 1 | 512 | 512 | 2 MB | 466.2 | 90.5 | 0.19 | 0.19 | 0.19 |
| `cfloat` | N | 4 | 1024 | 512 | 16 MB | 1700.8 | 422.6 | 0.25 | 0.21 | 0.27 |
| `cdouble` | N | 1 | 2048 | 512 | 16 MB | 563.8 | 135.8 | 0.24 | 0.24 | 0.24 |
| `double` | N | 4 | 1024 | 512 | 16 MB | 1372.7 | 385.0 | 0.28 | 0.28 | 0.28 |
| `cdouble` | N | 4 | 1024 | 512 | 32 MB | 1216.1 | 347.8 | 0.29 | 0.29 | 0.29 |
| `float` | N | 16 | 2048 | 512 | 64 MB | 2869.3 | 925.5 | 0.32 | 0.57 | 0.38 |
| `cdouble` | N | 1 | 512 | 512 | 4 MB | 392.1 | 127.4 | 0.32 | 0.33 | 0.32 |
| `double` | N | 1 | 512 | 512 | 2 MB | 375.2 | 122.8 | 0.33 | 0.33 | 0.33 |
| `float` | N | 1 | 512 | 512 | 1 MB | 240.5 | 81.8 | 0.34 | 0.34 | 0.32 |
| `float` | N | 4 | 1024 | 512 | 8 MB | 1012.9 | 380.0 | 0.38 | 0.38 | 0.38 |
| `cdouble` | T | 64 | 64 | 512 | 32 MB | 1419.7 | 639.1 | 0.45 | 0.45 | 0.45 |
| `cdouble` | C | 64 | 64 | 512 | 32 MB | 1411.0 | 639.8 | 0.45 | 0.46 | 0.43 |

A further 16 cells sit in `[0.50, 0.85)` — see `parity_report.txt`.

### Is this reached in production today?

**No, and that is the only reason it is not a stop-ship.** `gemv` has exactly one
internal caller, `ortho.cc:227-232`. In its `NoTrans` branch the two calls are
`(out_len = i, red_len = m)` transposed — which takes the **CTA** body and
measures 1.17–3.79× — and `(out_len = m, red_len = i)` un-transposed, which has a
**large** output length and measures 0.98–1.33×. In its `Trans` branch,
`gemv_op_shape` returns `nullopt` (the length disagreement the lead already
verified) and the call takes the vendor. `larft_wy.hh:210` uses the device-level
`gemv`, not this entry point.

So the blockers are reachable **only through the public API** — and in a
vendor-free build there is no other route, which is exactly where a 0.08× lands
on a user with no way out.

`ortho_shapes.sh` closes the last gap by measuring the production shapes
directly, including the corner neither this audit's main ladder nor `../ab/`
reached: `i` in `ortho.cc` **starts at 1**, so call 2 runs a large output against
a reduction of 1–32, which is one to thirty-two flops per output. 56 cells,
`float` and `cdouble`, `m` in {512, 2048}, `i` in {1, 2, 4, 8, 16, 32, 64},
batch 512:

| | worst | median | best |
|---|---|---|---|
| call 1 — `c(i) = A(m,i)^H y(m)`, CTA arm | **0.75×** | 1.09× | 2.52× |
| call 2 — `y(m) = A(m,i) c(i)`, Direct arm | **1.00×** | 1.23× | 2.97× |

**Not one production cell is below 0.75×, and call 2 — the `NoTrans` Direct body,
the one that fails so badly at short *output* — never drops below 1.00× here,
because in `ortho` its output length is `m`, not `i`.** The `NoTrans` failure
needs a short OUTPUT, and `ortho` only ever gives that body a short REDUCTION.
The best cells are the smallest ones: `float`, `m = 2048`, `i = 1` is **2.97×**
(326.9 → 972.4 GB/s), and `cdouble`, `i = 1`, `m = 2048` transposed is 2.52×.

This is the strongest single argument that WP7 should land: on the shapes the
library itself issues, the native kernel **is at or above cuBLAS on 49 of 56
measured points**, its median is 1.14×, and its worst — the only two cells below
0.85× — is 0.75×.

---

## 2. B5 — implemented, verified, and not sufficient

### The flattening is real (`geometry.csv`, read from `ncu`, not from the source)

| body | shape | grid | block | static SMEM | dynamic SMEM |
|---|---|---|---|---|---|
| `GemvDirectNKernel<cdouble>` | **m=64, batch=128** | **256** | 32 | 0 | 0 |
| `GemvDirectNKernel<float>` | m=64, batch=128 | 256 | 32 | 0 | 0 |
| `GemvDirectNKernel<cdouble>` | m=64, batch=**32** | 64 | 32 | 0 | 0 |
| `GemvDirectNKernel<cdouble>` | m=64, batch=**16** | 32 | 32 | 0 | 0 |
| `GemvDirectNKernel<cdouble>` | m=4096, batch=128 | 2048 | 256 | 0 | 0 |
| `GemvCtaTKernel<cdouble>` | m=128,n=64,batch=128 | 1024 | 256 | 0 | 0 |
| `GemvCtaTKernel<cdouble>` | m=256,n=256,batch=1024 | 32768 | 256 | 0 | 0 |

* **The required proof holds.** Body 1 at `m = 64, batch = 128` launches
  **256 work-groups** on a 128-SM box — twice the SM count, and twice what the
  `nd_range<2>` draft would have produced. The grid tracks `out_len * batch`, not
  `batch`: it falls to 64 at batch 32 and 32 at batch 16, i.e. it is never pinned
  to the batch count.
* **Zero local memory, confirmed by the hardware profiler** rather than by
  `grep`. Static *and* dynamic shared memory are `0.00` bytes on **all three
  bodies at every shape probed**. The recorded 48 KB launch hole is structurally
  unreachable. (`launch__occupancy_limit_shared_mem` reports 32 blocks, i.e. SMEM
  is not the occupancy limiter.)

### But the small-output cells ARE still starved, and there is a second effect

`mechanism.csv`, `cdouble`, `red_len = 2048`, `batch = 512`, sweeping `out_len`
straight through the 32-lane warp width:

| `out_len` | grid | block | sectors / global load | achieved occupancy | DRAM throughput |
|---|---|---|---|---|---|
| 1 | 16 | 32 | **32.00** | **2.08 %** | **7.03 %** |
| 2 | 32 | 32 | 16.00 | 2.08 % | 8.53 % |
| 4 | 64 | 32 | 12.00 | 2.08 % | 15.63 % |
| 8 | 128 | 32 | 10.00 | 2.08 % | 28.87 % |
| 16 | 256 | 32 | 9.00 | 4.15 % | 53.92 % |
| 24 | 384 | 32 | 9.00 | 6.19 % | 63.66 % |
| 31 | 496 | 32 | 9.50 | 8.01 % | 74.94 % |
| **32** | 512 | 32 | **8.50** | 8.27 % | 85.35 % |
| 48 | 768 | 32 | 8.67 | 12.38 % | 90.78 % |
| 64 | 512 | 64 | 8.50 | 16.58 % | 95.89 % |
| 128 | 512 | 128 | 8.50 | 33.23 % | 97.22 % |
| 256 | 512 | 256 | 8.50 | 66.33 % | 97.05 % |

**Two independent things go wrong at once, and both stop at `out_len = 32`.**

1. **Coalescing.** Sectors per global load is `32 / out_len` below the warp
   width and floors at 8.5 above it. The flattening maps
   `b = gid / out_len, i = gid % out_len`, so consecutive work-items hold
   adjacent elements of a column **only while they stay inside one batch item**.
   Below `out_len = 32` a warp straddles batch items, whose rows are `stride_a`
   apart, and one 32-lane load touches up to 32 separate sectors.
2. **Parallelism.** The launch is `out_len * batch` work-items in total. At
   `out_len = 1, batch = 512` that is 512 items = 16 warps on 128 SMs, and
   achieved occupancy is 2.08 %. Flattening changed *how* those items are
   grouped; it cannot manufacture items that do not exist.

**The transition is in LANES, not in bytes** — the discriminating control. The
same ladder for `float` (4-byte scalar) turns at the same `out_len = 32`:

| `out_len` | 1 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| `float` sectors / load | 32.00 | 4.00 | 3.00 | **2.50** | 2.50 | 2.50 |
| `cdouble` sectors / load | 32.00 | 10.00 | 9.00 | **8.50** | 8.50 | 8.50 |

A byte-alignment story predicts different turning points for a 4-byte and a
16-byte scalar. Both turn at 32.

**And it is not an `ld` artefact.** Padding the leading dimension away from `m` —
the cheap fix if it were alignment — moves nothing:

| shape | `ld` | sectors / load | DRAM % |
|---|---|---|---|
| out=16, red=2048 | 16 (= m) | 9.00 | 53.92 |
| out=16, red=2048 | 17 | 9.50 | 51.63 |
| out=16, red=2048 | 24 | 9.00 | 51.32 |
| out=64, red=2048 | 64 (= m) | 8.50 | 95.89 |
| out=64, red=2048 | 65 | 8.75 | 95.62 |
| out=64, red=2048 | 72 | 8.50 | 93.24 |

**Fairness to the implementer:** B5 asked for the batch-only extent to be
removed and it was, correctly and verifiably. The residual starvation belongs to
the *one-work-item-per-output* design of Body 1, not to the flattening, and the
counterfactual `nd_range<2>` launch was not measured (that would require editing
`src/`, which the audit may not do). What is measured is that the flattened
mapping additionally loses coalescing below the warp width.

### The named fix

Body 1's own header comment states the premise unconditionally —

> *"work-items `i` and `i+1` read `A[i + j*ld]` and `A[i+1 + j*ld]`: adjacent, so
> a 32-lane group covers 32 consecutive elements of one column and the access is
> fully coalesced."*

— and that is true **only for `out_len >= 32`**. The fix is the one the
implementer already named for a different weakness, applied to Body 1: when
`out_len < 32`, give each output **`W = 32 / out_len` lanes** and let the warp
cover `out_len` outputs × `W` reduction steps, with lane index
`i + out_len * jsub`. Consecutive lanes then hold consecutive `i` within one
batch item — 32 contiguous elements per warp again — the reduction gains a
factor `W` of parallelism, and the fold is a `log2(W)`-step shuffle, not the full
5-step ladder. This is a **fourth kernel body** or a templated variant of Body 1,
and it should be landed and measured on its own, not alongside anything else.

### Body 3's short-reduction weakness is NOT occupancy

The implementer's "known weakness 1" blames a fixed ladder cost. The profiler
agrees it is not a launch problem — occupancy stays 82–95 % and the grid stays
16384 — and shows the shortfall is straightforwardly in DRAM utilisation:

| `red_len` | 32 | 48 | 64 | 96 | 128 | 192 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|
| rounds = `ceil(red/32)` | 1 | 2 | 2 | 3 | 4 | 6 | 8 | 16 |
| DRAM throughput | 38.5 % | 42.1 % | 64.8 % | 81.3 % | **93.4 %** | 94.0 % | 94.4 % | 95.6 % |
| occupancy | 91.8 % | 94.7 % | 90.1 % | 92.4 % | 86.4 % | 82.7 % | 82.4 % | 85.9 % |

Consistent with a per-output cost amortising over the load rounds, and fully
amortised by `red_len = 128`. The direction of the implementer's story checks
out; the exact `r/(r+c)` fit does not (the implied `c` falls from 1.6 to 0.28
across the ladder), so it is reported as a measured curve rather than as a model.

---

## 3. The prize — real, bigger than claimed, and type-exclusive

`prize_p{1,2}.csv`: `cdouble`, `m` across 11 values × `n` in {128, 256, 512} ×
`batch` in {128, 256, 512}, for **both** transposed spellings, two passes.
394 rows, `relerr` exactly 0, route column 100 % correct, `foreign = 0`.

`transA = Trans`, `batch = 512` (p1 / p2):

|  m \ n | 128 | 256 | 512 |
|---|---|---|---|
| 32 | 0.29/0.29 | 0.24/0.24 | 0.50/0.49 |
| 48 | 0.27/0.27 | 0.78/0.78 | 0.69/0.67 |
| 64 | 0.37/0.37 | 1.86/1.87 | 1.91/1.90 |
| 80 | 1.79/1.80 | 2.02 (p2) | 2.08/1.99 |
| 128 | 0.99/0.99 | 2.35/2.35 | 2.51/2.56 |
| 192 | 1.02/1.02 | 2.64/2.72 | 2.87/2.86 |
| 256 | 1.00/1.00 | 2.71/2.70 | 2.72/2.88 |
| 320 | 1.04/1.04 | 2.87/2.88 | 2.93/3.02 |
| 384 | 1.02/1.02 | 1.03/1.03 | 1.03/1.04 |
| 448 | 1.02/1.02 | 1.03/1.03 | 1.02/1.03 |
| 512 | 1.01/1.01 | 1.03/1.04 | 1.01/1.01 |

The `m` band closes sharply at 320→384 in every batch and both spellings, which
confirms the recon phase's band and confirms **the axis is `m`, which under a
transposed `transA` is `red_len()`, not `out_len()`** (B3's trap, avoided).

**The whole effect is a cuBLAS dip, and it is `complex<double>`-exclusive.**
`typecheck.csv` holds the byte count constant by scaling batch:

| A ≈ 1 GB, `m=256, n=256`, `Trans` | batch | vendor GB/s | native GB/s | ratio |
|---|---|---|---|---|
| `float` | 8192 | 939.6 | 940.0 | 1.00 |
| `double` | 4096 | 945.9 | 940.8 | 0.99 |
| `complex<float>` | 4096 | 941.4 | 940.7 | 1.00 |
| **`complex<double>`** | 2048 | **323.0** | **936.4** | **2.90** |

The native CTA body reads **936–941 GB/s for all four scalars** — the same number
— and cuBLAS alone falls to a third of the roof for `complex<double>`. Pooled
over all 241 measured `cdouble` transposed cells, the vendor is **bimodal**:
median **334 GB/s** in the 68 cells the native kernel wins, median **892 GB/s** in
the other 173.

---

## 4. D2 — `ConjTrans`, measured for the first time

`ConjTrans` was run as a **full peer** of `Trans` in every sweep here, not as a
spot check: 98 paired prize shapes over two passes, plus 64 parity cells, plus
half of the out-of-sample grid.

**The native kernel does not care which spelling it gets**, which is what a
runtime `conj` flag inside an `if constexpr` should cost:

relative gap in GB/s between the two spellings, over 98 shapes × 2 passes:

| arm | median | 90th percentile | max |
|---|---|---|---|
| `native:cta` | **0.07 %** | 0.52 % | **3.1 %** |
| `vendor` | 0.99 % | 7.8 % | **13.0 %** |

So the whole `T`/`C` divergence in the *ratio* — up to 0.216 on a ≈ 2.8 base — is
**cuBLAS**, not us: it is slower for `ConjTrans` at `m = 384, 448` (804–820 GB/s
vs 907–917) and faster at `m = 80, n = 512, batch = 128`.

The loss regions, the win regions and the `m = 320 → 384` cliff sit in the same
places for both. **A `preferred()` clause may legitimately say
`transA != NoTrans`; it does not have to say `transA == Trans` only.** The one
caveat: at `m = 384, n = 512, batch = 512` the two straddle the 1.15× gate
(`T` 1.03, `C` 1.17) — which is an argument for keeping the `m <= 320` band, not
for splitting the clause by spelling.

---

## 5. `preferred()` — RECOMMEND ALL-FALSE

**Do not edit `route_gemv.hh`. It is already correct.**

The win is real (68 of 241 cells at ≥ 1.15×, up to 3.08×), but the lead's rule is
that **every cell a clause admits must measure ≥ 1.15× in both passes**. Scoring
the clause family over the fitted grid pooled with a purpose-built out-of-sample
grid (`oos_p{1,2}.csv`: 22 shapes whose `m`, `n` and `batch` values appear
nowhere in the fitted grid):

| candidate | fitted grid | out-of-sample | verdict |
|---|---|---|---|
| `64 <= m <= 320 and n >= 256` (the rectangle in `../ab/README.md`) | admits 71, **17 below 1.00×, worst 0.36×** | admits 30, all ≥ 1.87× | **REFUTED** — the OOS grid happens to contain no `n = 256` cell, so it does not probe where this fails. The fitted grid does. |
| `64 <= m <= 320 and n*batch >= 131072` | admits 35, worst 0.99× | admits 36, **2 below 1.00×** (`m=96, n=192, batch=1024` → 0.97/0.97) | **REFUTED out-of-sample** — independently of, and agreeing with, the implementer's own refutation of this rule |
| `64 <= m <= 320 and A >= 512 MB` | admits 34, worst **1.01×**, none below 1.00× | admits 18, worst **1.01×**, none below 1.00× | **survives "do no harm", fails the ≥ 1.15× gate** (4 admitted cells sit at 1.01–1.08×) |
| `64 <= m <= 320 and n >= 768` / `... and A >= 1024 MB` | passes strictly | passes strictly | **passes the letter of the gate and should still not ship** — see below |

The clauses that pass strictly (`clause_report.txt`) all rest on a threshold
sitting **on the edge of the sampled range**: `n >= 768` is the smallest `n` in
the out-of-sample grid and nothing between `n = 512` and `n = 768` was measured;
`A >= 1024 MB` sits at the top of the footprint ladder. Each captures at most 22
of the 68 measured wins. This is the same shape of fit as the `n*batch` rule that
was built, tested out-of-sample and refuted — one grid step from being wrong in
the direction that moves live traffic onto a 0.36× route.

**Recommendation: `preferred()` ships all-false, exactly as it stands.** WP7's
honest headline is vendor-freedom at parity plus a large, documented,
opt-in win — `BATCHLAS_GEMV_ROUTE=native:cta` — which is the headline WP6 shipped
and is the right call again.

**If the lead wants the 3× by default**, the only defensible clause is

```
scalar == complex<double>            // type-exclusive, proven at matched bytes
  && transA != NoTrans               // Trans and ConjTrans agree to 0.06x
  && 64 <= red_len() && red_len() <= 320     // NOT out_len(); the axis is m
  && (int64) m * n * batch * 16 >= 512 << 20 // footprint, from CSV, not from L2
```

with these caveats stated in the same commit: it is a **footprint** threshold
fitted to this data (it is *not* an L2 gate — 512 MB is 7× the 4090's 72 MB L2, so
B4's specific prohibition is not engaged, but D3's "must come from a measured
CSV" is the only thing carrying it); **nothing between 256 MB and 512 MB was
measured** at these shapes; and it admits four cells at 1.01–1.08× where the win
is nil. It does no measured harm on 52 cells across four passes and two
independently designed grids. The `red_len()` spelling is mandatory — writing it
on `out_len()` tests `n` and inverts the window.

**The unmapped axis, stated plainly:** the win frontier moves with `batch` at
`n = 256` and `n = 512` (at `n=256`: 0.45× at batch 128, 0.99× at batch 256,
2.52× at batch 512, all at `m = 128`) and does not move with `batch` at all at
`n = 192` (no dip at batch 1024) or at `n >= 384` (dip already present at batch
192). Whatever governs cuBLAS's dip is a property of `n` with a batch-dependent
floor, and neither footprint, nor L2 residency, nor `n*batch` reproduces it.
Mapping it is a bounded follow-up: sweep `n` in 32-wide steps from 192 to 768 at
`m = 128, 256` across batch 64…1024.

---

## Files

| file | what |
|---|---|
| `parity.sh` → `parity_p{1,2}.csv` | the `(out_len, red_len)` parity ladder, 16 shapes × 4 types × 3 `transA` × 3 arms, twice |
| `analyse_parity.py` → `parity_report.txt` | renders it, audits the route column and `relerr`, applies the B6 gate |
| `blockers_recheck.sh` → `blockers_p3.csv` | the 15 blockers, re-run against the post-rebuild binary |
| `geometry.sh` → `geometry.csv` | `ncu` launch geometry and shared-memory bytes for all three bodies (B5) |
| `mechanism.sh` → `mechanism.csv` | `ncu` sectors-per-request, occupancy and DRAM throughput across the warp width, with the `ld` control |
| `prize.sh` → `prize_p{1,2}.csv` | the `cdouble` transposed region, 11 `m` × 3 `n` × 3 `batch` × 2 `transA`, twice |
| `analyse_prize.py` → `prize_report.txt` | renders the grids and scores candidate predicates |
| `oos.sh` → `oos_p{1,2}.csv` | 22 shapes sharing no `m`, `n` or `batch` value with the fitted grid |
| `analyse_oos.py` → `oos_report.txt` | the out-of-sample verdict on each candidate |
| `clause_search.py` → `clause_report.txt` | the whole clause family scored over the pooled 241 cells |
| `typecheck.sh` → `typecheck.csv` | the dip is `complex<double>`-only, at matched bytes |
| `ortho_shapes.sh` → `ortho_shapes.csv` | the shapes `ortho.cc` actually issues, including `i = 1..32` |

The harness is `../ab/gemvab_v`, rebuilt from `../ab/build.sh` at the start of
this audit (campaign trap 2). It is covered by `../ab/.gitignore`; `.gitignore`
here covers the profiler reports and stderr spools (campaign trap 7).
