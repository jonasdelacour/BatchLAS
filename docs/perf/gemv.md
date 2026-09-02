# GEMV: five kernel bodies, the only native tier that is not GPU-gated (WP7 and its repair pass)

Native batched `gemv` for all four scalar types and all three `transA`, plus the routing decisions from WP7, its
repair pass, and the WP6/WP7 closure pass (whose experiment directories are named `wp8_*` — a misnomer; WP8 proper is
sparse `spmm`). Box: RTX 4090, 128 SMs, 72 MB L2, ~1008 GB/s theoretical DRAM peak, **~950 GB/s treated as the
achievable roof**; two cards in the chassis, device 0 drives the display. Ratio convention `vendor_ms / native_ms`
(1.00 is parity, > 1 means native is faster) — except in the body-5 A/B tables, where it is `body3_ms / body5_ms`.

## What ships

### The route arms

`kGemvOrder` (`include/batchlas/blas/dispatch/route_gemv.hh:147-151`) is a **capability ladder, tighter first**, not a
preference list:

| order | Route | `supports()` — correctness only, never a speed cutoff |
|---|---|---|
| 1 | `{Native, CTA}` | `cta_available && is_gpu && has_sg32 && transA != NoTrans` (`route_gemv.hh:217-218`) |
| 2 | `{Native, Direct}` | `direct_available` — **and no `is_gpu` clause at all** (`route_gemv.hh:203`) |
| 3 | `{Vendor, Auto}` | everything |

Both native arms are additionally refused for `heterogeneous_batch` (one launch covers the batch with a single
`(m, n, ld, stride)` tuple, and `VectorView` has no active-size concept, so gemv cannot get gemm's heterogeneous
walker) and for `m < 0 || n < 0 || batch < 1`. `m == 0 || n == 0` is **not** refused: it is a legal call that
quick-returns. `has_sg32` is **enumerated** from `sycl::info::device::sub_group_sizes`, never
`get_property(MAX_SUB_GROUP_SIZE)` (which returns `sub_group_sizes()[0]` and is wrong in both directions — and for a
kernel carrying `[[sycl::reqd_sub_group_size(32)]]` the "accepted although it has no 32" direction aborts the launch).

**The missing `is_gpu` gate is the work package.** `tests/gemv_tests.cc` instantiates
`GemvMatrixViewTest` over eight configurations — `/0..3` are `Backend::NETLIB` on a
`native_cpu Device("cpu")` queue, `/4..7` are `Backend::CUDA`. (The file as it now stands runs
**264 tests from 16 suites**, once `GemvCoverageTest`'s eight are counted.) A GPU-gated native `gemv`
closes 20 of the 40 vendor-free failures, leaves the suite red, and moves the vendor-free burn-down by **zero**.
`src/sycl/gemv_native.cc` is therefore compiled for the `native_cpu` target too, and bodies 1 and 2 use no collective,
no local memory and no required sub-group size. Vendor-free `gemv_tests`: **40 FAILED → 0** (264/264 after the repair
pass, in both builds). The layout fact that inverts the usual intuition: column-major `A(i,j)` at `i + j*ld`, one
work-item per output element ⇒ **`NoTrans` is already fully coalesced and needs no collective; it is
`Trans`/`ConjTrans` that wants the sub-group reduction.**

### The five kernel bodies

| body | kernel | route | shape | decomposition |
|---|---|---|---|---|
| 1 | `GemvDirectNKernel<T>` | `{Native, Direct}` | `NoTrans` | one work-item per output row |
| 2 | `GemvDirectTKernel<T>` | `{Native, Direct}` | `Trans`/`ConjTrans` | one work-item per output column — **the portable arm**. Bodies 1 *and* 2 are what run on `native_cpu`: body 1 for `NoTrans`, body 2 for the transposed spellings |
| 3 | `GemvCtaTKernel<T>` | `{Native, CTA}` | transposed, GPU, enumerated sg 32 | one 32-lane sub-group per output, `shift_group_left` ladder |
| 4 | `GemvSegNKernel<T,W>` | `{Native, Direct}` | `NoTrans`, `out_len <= 16`, sg 32 | `W = 32/out_len` lanes per output, one sub-group per batch item, fold at stride `out_len` |
| 5 | `GemvSegTKernel<T,W>` | `{Native, CTA}` | transposed, short reduction, sg 32 | `W` outputs per sub-group, `L = 32/W` lanes each, fold at stride 1 |

**A route column cannot tell you which body ran.** `{Native, Direct}` names bodies 1/2/4 and `{Native, CTA}` names
bodies 3/5; those choices are decompositions, not algorithms, and are deliberately kept below the routing vocabulary
(putting them in `supports()` would be a speed cutoff in the predicate that carries correctness only, and would
re-introduce the GPU gate). What *can* separate them: `gemv_seg_trans_width_debug` (test-only, resolving through the
**same** gate function the launcher calls) and breaks that are red for one body and green for the other.

All five declare **zero bytes of local memory**, static and dynamic: no `local_accessor` is created anywhere in the
TU, and `ncu` confirms it directly on bodies 1, 2 and 3 (`audit/geometry.csv`) and on body 5 (`wp8_gemv/regs.csv`) —
**body 4 is the one body no profiler row covers for shared memory**. That keeps the recorded 48 KB launch hole
structurally unreachable in this TU. The reduction is a hand-rolled shuffle
ladder, explicitly not `sycl::reduce_over_group` (which allocates static shared, and which WP6 measured 1.5–4.7×
slower for `double`/`complex<double>`). `GemvSegTKernel<T,W>` uses **exactly** the registers `GemvCtaTKernel<T>` uses
at every `W` — float 28, double 36, cfloat 38, cdouble 40. Device link: body 4's **20** extra entry functions (5 `W` values × 4 types)
measured **58.03 s absent vs 57.59 s present**, body 5's 12 measured **+0.26 s / +0.45%** (inside the arm's 0.84 s
spread). *(`repair/README.md` writes "5 instantiations per scalar type — 40 extra entry functions" in one sentence;
`linktime.sh` and `wp8_gemv/README.md` §12 both say "twenty", and 5 × 4 = 20. Twenty is the number.)*

### The shipped `preferred()` window

**The WP7 exploration notes say `preferred()` ships all-false. That is no longer what ships.**
`experiments/wp7_gemv/ab/README.md`, `experiments/wp7_gemv/audit/README.md` §5 and the WP7 and "WP7 REPAIR PASS"
sections of `VENDOR_INDEPENDENCE_PLAN.md` all recommend and record all-false — and so does `route_gemv.hh`'s own
`---- MEASURED WINDOW ----` preamble, which still opens **"ALL-FALSE, AND THAT IS A RESULT, NOT AN OMISSION"** two
hundred lines above the `WP8 ROUTING PASS` block that supersedes it. What is *not* stale: the plan's summary row
(`VENDOR_INDEPENDENCE_PLAN.md:25`) and its "WP6/WP7 performance-closure pass" section both state the shipped clause
correctly. The closure pass re-searched the clause family with `batch` as a first-class term and landed a window.
The code is the authority:

```cpp
// include/batchlas/blas/dispatch/route_gemv.hh:469-486
static bool preferred(Route r, const GemvShape& s) {
    if (!is_native(r) || r.algo != Algorithm::CTA) return false;
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        if (s.transA == Transpose::NoTrans) return false;
        const int64_t red = s.red_len();   // == A.rows() under Trans
        const int64_t out = s.out_len();   // == A.cols() under Trans
        return red >= 64 && red <= 352 && out >= 256 && s.batch >= 320;   // :483
    }
    return false;
}
```

**`complex<double>` only, `{Native, CTA}` only, `transA != NoTrans`, `64 <= red_len() <= 352`, `out_len() >= 256`,
`batch >= 320`.** `float`, `double` and `complex<float>` are all-false at every shape, and so is the `Direct` tier.
`red_len()` is `n` under `NoTrans` and **`m` under `Trans`/`ConjTrans`**; the measured band is on `m`, so a predicate
written on `out_len()` tests the wrong extent and *inverts* the window — an error caught twice during WP7.

### The sub-route gates

Body 4 (`src/sycl/gemv_native.cc:175-180`): `W = gemv_seg_width(out_len)`, the largest power of two with
`W*out_len <= 32`; `W == 1` means "no segmentation available", so **body 4 serves `out_len <= 16`** and body 1 takes
17 and above. It also requires `Device::supports_sub_group_size(32)` — false on `native_cpu`, which is why the 20
NETLIB rows keep body 1. Body 5 (`src/sycl/gemv_native.cc:281-387`) has three gates, all on `red_len()` and never on
`out_len()`, all transcribed cell by cell from a CSV rather than derived from an inequality:

| gate | float | complex&lt;float&gt; | double | complex&lt;double&gt; |
|---|---|---|---|---|
| 1 — body 5 runs at `red_len <=` | 32 | 16 | 48 | 64 |
| 2 — `W = 8` up to `red_len <=`, then `W = 4` | 24 | 16 | 32 | 32 |
| 3 — floor on `out_len*batch` | `16*CU` in the `W = 8` band, `64*CU` in the `W = 4` band (2048 / 8192 here) | | | |

`W ∈ {2, 4, 8}` is instantiated; `W = 2` exists only so `BATCHLAS_GEMV_SEGT=2` keeps meaning `W = 2` rather than
silently resolving elsewhere. Environment: **`BATCHLAS_GEMV_ROUTE`** (`direct` / `cta` / `native` / `vendor`) selects
the route; **`BATCHLAS_GEMV_SEGT`** (`off` / `auto` / `2|4|8`) selects the body-5 spelling and bypasses all three
gates. The latter is re-read on **every launch and never latched** — a latched presence flag has been a blind guard
eleven times in this campaign. Two silent traps, both measured:

* A bare `BATCHLAS_GEMV_ROUTE=native` resolves to the first **supported** native route — `Direct` for `NoTrans`, for a
  CPU device, and for a GPU without an enumerated 32. **76 of 104 decisions in `gemv_tests` land on Direct.** Pin
  `native:cta` / `native:direct` explicitly.
* Pinning a route the shape cannot take does **not** fail and does **not** warn: `resolve_route` falls through to
  `automatic()`, which is the **vendor** in a vendor-present build and `native:direct` in a vendor-free one. Measured:
  `native:cta` on `NoTrans` shapes sends 76 of 136 decisions to cuBLAS/OpenBLAS while the operator believes CTA is
  pinned; a misspelled value behaves identically, because `ParsedRouteEnv::unparsed` is discarded (campaign-wide, not
  a gemv invention). **The resolved-route column is the only way to know which arm ran.**

## Evidence for each boundary

### The vendor baseline

cuBLAS `gemvStridedBatched` measures **94–105% of the ~950 GB/s roof on 90 of 92 reproducing cells** (102 of 104 in
the main sweep before cross-pass filtering), over all four types, both `transA`, `n` 32…2048, `batch` 64…65536; two
independent passes agree to `spread <= 1.01` on 98 of 104 cells. A batched gemv reads A once for two flops: nothing to
hide behind, nothing to reuse, so **parity is the achievable outcome** and a large speedup on a DRAM-resident cell is
a measurement error. The one exception is type-exclusive — at matched bytes and matched `(m, n)` (A = 1024 MB,
`Trans`):

| m × n | float | double | cfloat | **cdouble** |
|---|---|---|---|---|
| 256 × 256 | 936 | 957 | 937 | **325** GB/s |
| 64 × 256 | 966 | 966 | 967 | **376** GB/s |

Ruled out with controls. **Not `ld` alignment**: padding `ld` to 257/264/320 at 256×256 leaves it at 307–325 GB/s,
while the control — padding a *healthy* cell to `ld = 513` — correctly costs 922 → 718 GB/s, so "padding did nothing"
is a result and not a broken instrument. **Not our kernel**: pooled over 241 measured `cdouble` transposed cells the
vendor is bimodal, median 334 GB/s on the 68 cells we win and 892 GB/s on the other 173, while the native CTA body
reads 936–941 GB/s for *all four* types at matched bytes.

### The `cdouble` window boundaries

The dip is a **discrete kernel-selection switch inside cuBLAS**, not a gradient: at `out_len 512, red_len 128,
cdouble, Trans` its throughput is 894.9 / 919.4 / 930.1 GB/s at batch 128 / 192 / 256, then 360.4 / 359.7 / 363.1 /
358.4 at batch 320 / 384 / 448 / 512. One batch rung, a 2.6× fall, and it stays fallen. That is why no function of
`n*batch` and no power law `n^a*batch` can describe it, and why the shipped predicate names `batch` outright.

**The finer grid puts the rung one step lower than the shipped floor.** `g6_fit2_p1.csv` walks the same shape through
`batch 288` and reads the vendor at **359.3 GB/s** there against 931.3 at batch 256 — so the switch is between 256 and
288, and 320 is one measured rung *above* it. See [open-debts](#open-debts): 288 is not a bracketed boundary, it is a
threshold the search never enumerated.

Every boundary, with the measured non-winner that brackets it:

| boundary | inside | the bracketing non-winner |
|---|---|---|
| `red_len >= 64` | 2.41 at `red_len 64` | **0.9515** at 32; 1.1028 at 40, 1.31 at 48, 1.16 at 56 — ragged, and 48 sits on cuBLAS's *slope* (456 GB/s at batch 128 rising to 765 at batch 1024) |
| `red_len <= 352` | 2.84 at `red_len 352` | **1.0304 / 1.0314** at 384, where cuBLAS is back at the roof (901–906 GB/s) |
| `out_len >= 256` | 2.32 at `out_len 256` | **0.9988** at 192, 0.994 at 128, 0.989 at 96; and 0.546 / 0.559 at 64 / 32, where cuBLAS reads 1777 / 1597 GB/s — **above** the DRAM peak, i.e. L2-resident, the family this table declines to chase |
| `batch >= 320` | 63 cells, geomean 2.572, min 2.2261 | **0.9628** at `out 512, red 128, batch 256` (cuBLAS 930.1); 0.9616 at batch 192; **0.9562** at batch 128 |
| `scalar == cdouble` | — | float **0.9340** (`out 256, red 128, b 512`, cuBLAS 2742 GB/s — L2); double **0.9722** (`out 512, red 128, b 1024`, with 0.9746/0.9749 beside it); cfloat **0.6644** (`out 256, red 48, b 512`, cuBLAS 2637 GB/s — L2) |

**Below batch 128 the region is not marginal, it is refuted outright.** 46 cells inside the (`red_len`, `out_len`)
band at batch 1…96 hold **27 losses**, worst **0.5417** at `out 512, red 128, batch 64` — cuBLAS 1764.2 GB/s, above
the DRAM peak, L2 again. A clause with no batch floor routes every one of them.

Shipped-clause scores over two passes (grid B, `red_len` walked 8..512 at `out_len` 256/512): `Trans` alone **63
cells, geomean 2.572, min 2.2261, zero losses**; both spellings pooled, **68 cells, geomean 2.821, min 2.3741**.
`ConjTrans` is a full peer — the native kernel treats the two spellings identically (median 0.07%, max 3.1% apart in
GB/s over 98 paired shapes × 2 passes) and it is *cuBLAS* that diverges (median 0.99%, max 13.0%).

**The weakest admitted cell is an L2-boundary corner, and is recorded as such.** The `g6_fit` grids carry `gpu=0` on
every row: the clause was fitted on the display GPU, and its lowest-footprint admitted cell — `cdouble, out_len 256,
red_len 64, batch 320` — is 84 MB against a 72 MB L2, exactly where the cross-device control (taken at DRAM-resident
footprint) does not hold. Fitted on device 0 it reads 2.313; on the idle card 1.867 / 2.065 (adversarial review) and
2.394 / 2.091 (lead). The **native** arm agrees across devices to 3.5%; only the vendor arm moves, by ~20%, and only
here. **Read the clause's floor as ~1.87, not 2.23.** Every other admitted cell re-measured on the idle card sits at
2.25–2.49. Device hygiene generally: `nvidia-smi --query-compute-apps` is per-device and cannot see
Xorg/gnome-shell/firefox, invisible in DRAM-resident numbers and **not** invisible in L2-resident ones — at
`out 64, red 64, batch 512` the vendor reads 1366.2 GB/s on device 1 and 760.6 on device 0 while the native arm
reproduces to 1.6%. Native-vs-native A/B ran on device 0; every vendor-facing table on device 1.

Three more hygiene rules this campaign paid for, all encoded in the harnesses and all easy to skip:
**(1)** `rel_sd` does **not** catch contention — a contended row can have a *low* relative standard deviation, which
is why every parity and prize row carries a per-device foreign-process count instead. **(2) Campaign trap 2:** the A/B
harness resolves and prints the route *in its own TU*, so it must be rebuilt after any `preferred()` change or the
route column lies (`ab/build.sh`; the audit rebuilt it before starting). **(3)** A foreign rebuild of
`libbatchlas_sycl.so` landed mid-sweep during the audit — two `prize_p1.csv` rows died with `invalid ELF header` —
and that was disclosed and *checked* rather than assumed away: the native arm's own timings agree across the boundary
to a median of **1.0010** and a worst of 1.054 over 197 paired cells, and all 15 blockers were re-run against the
post-rebuild binary (`blockers_p3.csv`, reproducing to ±0.02×).

### The body-5 gates

**Gate 1** is where body 3 stops being materially short of the roof, per type — its own GB/s at `out_len 2048, batch 512, Trans`, DRAM-resident (bold = last rung admitted):

| `red_len` | 1 | 8 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| float | 59.3 | 261.5 | 485.5 | 655.9 | **832.6** | 912.9 | 922.7 | 931.0 |
| cfloat | 104.8 | 450.2 | **779.8** | 911.1 | 921.7 | 929.4 | 925.2 | 932.7 |
| double | 33.3 | 149.8 | 281.9 | 415.9 | 548.1 | **817.0** | 928.7 | 932.2 |
| cdouble | 26.6 | 118.8 | 224.2 | 329.2 | 434.3 | 456.2 | **707.9** | 932.5 |

The other side is measured, not assumed: forcing `W ∈ {4,8}` past each gate measures **0.983–0.996** just above every
type's gate at DRAM-resident footprint — a revert, which is what makes gate 1 load-bearing rather than decorative.

**What body 5 is worth**, gates as shipped, body 5 against body 3 interleaved rep by rep inside one process, 11 reps,
median, two passes, worse pass quoted:

| grid | cells | geomean | min | max | below 1.00 |
|---|---|---|---|---|---|
| `Trans`, admitted | 83 | **3.286** | 1.070 | 10.47 | 0 |
| `Trans`, declined | 53 | — | 0.993 | 1.003 | — |
| `ConjTrans`, admitted | 36 | **2.746** | 1.074 | 10.44 | 0 |
| `ConjTrans`, declined | 16 | — | 0.985 | 1.002 | — |
| skinny (`out_len` 1…64), admitted | 30 | **1.566** | 1.037 | 3.04 | 0 |
| skinny, declined | 74 | — | 0.977 | 1.009 | — |

**Gate 3 exists because this pass committed the campaign's trap 8 and then caught it.** The `(out_len, red_len)` plane
it fitted on starts at `out_len = 64` and reported 83 admitted cells with zero below 1.00×. Re-running the WP7 audit's
parity grid — which reaches `out_len = 1` — showed the native arm 3–6% *slower* at `out_len = 1` than before the
change. Walking the output axis down found **16 losing cells, worst 0.891×, every one at `out_len*batch <= 4096`**:

| `out_len*batch` | 128 | 512 | 1024 | 2048 | 4096 | ≥ 8192 |
|---|---|---|---|---|---|---|
| losing cells | 4 | 7 | 2 | 2 | 1 | 0 |
| worst | 0.891 | 0.957 | 0.978 | 0.983 | 0.998 | n/a |

A single floor of `8*CU` left five cells at 0.976–0.998; a third and fourth pass at **31 reps** separated them —
`double` recovered (0.986/1.002, noise) but **three float cells reproduced below 1.00 in both passes**
(`out_len 4 @ batch 512` 0.976/0.986, `out_len 16 @ batch 128` 0.985/0.983, `out_len 1 @ batch 4096` 0.998/0.990), all
in the thin-margin `W = 4` band. Hence the two-row floor, at the cost of seven small wins (1.02×–1.36×) on shapes
whose whole launch is a few thousand outputs; the alternative was shipping a reproduced 0.976×.

**Odd `ld`** costs body 5 something and **never inverts the sign**: at `out_len 2048, batch 512`, packed → odd `ld`,
cdouble `red_len 8` 7.32× → 6.65×, double `red_len 8` 9.92× → 5.59×, cfloat `red_len 8` 4.06× → 2.13×, float
`red_len 32` 1.076× → 1.062×. Every admitted odd-`ld` cell stays at or above 1.06×, and the suite exercises `ld = 79`
at `m = 70`, so this is a live layout, not a hypothetical.

### The body-4 gate

The family body 4 fixed: `{Native, Direct}`, `NoTrans`, `out_len < 32`, **0.08×–0.38× of cuBLAS on 13 cells**, worst
`cfloat out=1 red=2048 batch=512` at 0.08 (vendor 1206.9 GB/s vs native 99.2), reproduced on three passes to ±0.02×.
Two independent effects, **both stopping exactly at 32 lanes**, both from `ncu` (`cdouble`, `red_len 2048`,
`batch 512`):

| `out_len` | 1 | 2 | 4 | 8 | 16 | 24 | 31 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| sectors / global load | 32.0 | 16.0 | 12.0 | 10.0 | 9.0 | 9.0 | 9.5 | **8.5** | 8.67 | 8.5 |
| achieved occupancy | 2.08% | 2.08% | 2.08% | 2.08% | 4.15% | 6.19% | 8.01% | 8.27% | 12.38% | 16.58% |
| DRAM throughput | 7.03% | 8.53% | 15.63% | 28.87% | 53.92% | 63.66% | 74.94% | 85.35% | 90.78% | 95.89% |

Two controls make it a **warp** story and not a bytes story: `float` (a 4× narrower scalar) turns at the same
`out_len = 32` (sectors/load 32.00 → 2.50 across the same ladder), and padding `ld` moves nothing (at `out_len 16`,
`ld` 16/17/24 → 9.00/9.50/9.00). Result of the repair, re-measured on the audit's own harness, two fresh passes:

| | before | after |
|---|---|---|
| cells below 0.50× (of 192) | **15** | **2** |
| worst cell | **0.08×** | **0.44×** |
| the `NoTrans`, `out_len < 32` family (24 cells) | min 0.08, median 0.38, 13 blockers | min 0.60, median **1.16**, max 4.09, **0 blockers** |

Body 4's gate boundary (`out_len <= 16`) is bracketed on one side by the `out_len` 1..16 ladder in
[kernel-hypotheses-refuted](#kernel-hypotheses-refuted) and on the other only by arithmetic (`W == 1` for
`out_len >= 17`); **`17 <= out_len <= 31` is unmeasured** and the boundary is pinned by a test, not a timing.

Where the last two sub-0.50× blockers went: `cdouble out=64 red=64 batch=512` measured 0.450 (`T`) / 0.472 (`C`) after
the repair pass and **0.862 / 0.861** after body 5. Cells below 0.50× across the whole parity ladder: **0**. They do
not reach parity and cannot — the vendor is at ~1400 GB/s there, above the DRAM peak, converting L2 residency into
bandwidth that a streaming kernel never sees.

## Negative results

### Routing hypotheses refuted

* **`64 <= m <= 320 && n >= 256` (the "24-cell rectangle", every cell ≥ 1.89× on its own slice). REFUTED.** It was
  fitted at a *fixed ~1 GB footprint*, where `batch` moves inversely with shape and the batch axis is invisible.
  Resolved on an `(m, n, batch)` grid it admits **17 cells below 1.00×, worst 0.36×**: at `m=128, n=256, Trans` the
  ratio is 0.52 at batch 128, 0.99 at batch 256 and 2.35 at batch 512.
* **`n*batch >= 131072`. REFUTED twice, independently.** All four observed transitions straddle it, so it was
  predicted and then tested on shapes it was not fitted on: at `m=128, n=512, batch=256` it predicts a win and cuBLAS
  is at **924.7 GB/s** — the roof — for 0.97/0.97. The auditor's out-of-sample grid refuted it separately at
  `m=96, n=192, batch=1024` (0.97/0.97, cuBLAS 925 GB/s).
* **`A >= 256 MB` instead of a batch term. REFUTED by a cell** — 0.9628 at `out 512, red 128, batch 256`, which *is*
  256 MB; that is the answer to "isn't batch just a proxy for size". An **L2-residency gate is separately forbidden**
  (`route_gemv.hh:279-284`) and the data agrees: the dip switches on at 537 MB for one shape and 134 MB for another,
  while 268 MB shows none — all far above the 72 MB L2.
* **`64 <= m <= 320 && A >= 512 MB`: survives "do no harm" and fails the gate anyway.** On both the fitted grid
  (admits 34) and the auditor's out-of-sample grid (admits 18) its worst cell is **1.01×** and nothing is below
  1.00× — but four admitted cells sit at **1.01–1.08×**, where the win is nil, so it never clears the ≥ 1.15×
  bar. The audit named it as the *only* defensible clause of its day, with the caveats stated in the same breath:
  it is a **footprint** threshold from a CSV rather than an L2 gate (512 MB is 7× the 72 MB L2), and **nothing
  between 256 MB and 512 MB was measured** at those shapes.
* **Everything the WP7 clause search could produce.** `clause_search.py` enumerates `(m band) × (n threshold) ×
  (A threshold)` and **contains no `batch` term**, so every REFUTED verdict in `clause_report.txt` is a verdict about
  clauses that cannot express the boundary. The two candidates passing strictly (`n >= 768`, `A >= 1024 MB`) sat on
  the edge of the sampled range and captured at most 22 of the 68 measured wins. Re-searched with `batch` first-class,
  a clause survives — which is why the all-false verdict is superseded rather than wrong.
* **`red_len >= 48`: passes the letter of the gate and is still declined.** Over two passes it scores 88 cells,
  geomean 2.135, **min 1.1605, zero losses** — 0.9% above the bar. Declined on **mechanism, not margin**: at
  `red_len 64..352` the vendor sits on the *flat floor* of its dip (304–386 GB/s at every `out_len` and batch), so
  extrapolating into unmeasured corners is safe; at `red_len 48` it is on the *slope*, still rising at the top of the
  batch ladder, while the clause admits batches above 1024 and `out_len` above 2048.
* **`out_len >= 768 && batch >= 128` as a second disjunct: ~18 more cells at 2.26×–2.91× with no measured loss, NOT
  SHIPPED.** 128 is the lowest batch that grid reached at those `out_len`, so the floor is the edge of the sampled
  range wearing a boundary's clothes — the objection WP7's audit raised against its own `A >= 1024 MB`.

### Kernel hypotheses refuted

* **"The shuffle ladder is a fixed cost per output — 5 steps, doubled to 10 for a complex scalar." REFUTED by the
  counters.** That reading predicts `double` (5 shuffles of a 64-bit value) and `complex<float>` (10 shuffles of a
  32-bit value) are hurt equally. Measured on body 3 at `out_len 2048, batch 512, Trans`, `red_len 32`: float 833.8,
  **double 547.5**, **cfloat 921.2**, cdouble 434.5 GB/s — double and cfloat **1.68× apart at identical bytes and
  identical shuffle count**. `ncu`: `sm__pipe_fp64_cycles_active` is 85.6/86.1/84.7% for cdouble and 85.0/82.7/58.3%
  for double across `red_len` 32/64/128, and **exactly 0.00%** for float and cfloat, while occupancy holds at 79–93%
  and sectors-per-load is ideal. **The fold is FP64 work on a 1/64-rate GeForce part**: `sg_sum` runs 5 add steps on
  all 32 lanes — 160 double-adds per output for `double`, 320 for `complex<double>`, against only `red_len` useful
  FMAs. Body 5 cuts that to `L*log2(L)`, 8 at `L = 4`. A second, earlier model dies with it: the audit checked the
  implementer's `r/(r+c)` amortisation fit against its own ladder (DRAM throughput 38.5 / 42.1 / 64.8 / 81.3 /
  93.4 / 94.0 / 94.4 / 95.6% at `red_len` 32…512, occupancy pinned at 82–95%, grid constant) and the implied
  constant `c` **falls from 1.6 to 0.28 across the ladder** — so the curve is directionally right and the fit is
  not a model. Quote it as a measured curve, fully amortised by `red_len = 128`.
* **"`W` can be a runtime value." REFUTED, and it is worth 2×.** Body 4's first version carried `W` as a runtime
  `const int`: it fixed `out_len <= 4` and **regressed `out_len >= 8` below the body it replaced.** GB/s, float,
  ~128 MB, `NoTrans`:

  | `out_len` | 1 | 2 | 4 | 8 | 12 | 16 |
  |---|---|---|---|---|---|---|
  | body 1 | 235.1 | 335.4 | 517.3 | 730.5 | 692.9 | 827.2 |
  | body 4, `W` runtime | 906.5 | 707.9 | 576.1 | 607.5 | **373.8** | **461.1** |
  | body 4, `W` `constexpr` | **934.9** | **921.4** | **913.2** | **903.0** | 624.7 | **861.1** |
  | cuBLAS | 901.3 | 1281.0 | 1112.4 | 1012.8 | 897.6 | 962.0 |

  `ncu` says it was **not the memory system**: at `out_len 16` the runtime-`W` version already had sectors-per-load
  2.50 (ideal) and 8.27% occupancy against body 1's 3.00 and 4.12%, and still ran at 26% of DRAM where body 1 reached
  69%. Better coalescing *and* better occupancy, slower kernel — the loop, not the traffic. With `W` a compile-time
  constant the trip count and address stride are known, the loop unrolls, and the same shapes run at 90–98% of the
  vendor. `out_len = 12` remains the one place body 1 is ahead by more than noise (0.77× vs body 4's 0.70×): `W = 2`
  leaves 8 of 32 lanes idle because 12 does not divide 32.
* **The sector floor (`L*sizeof(T) >= 32`, giving `W <= 4` for float). REFUTED below `red_len ≈ 32`.** float at
  `out_len 2048`, `red_len <= 16` runs **5.16×–5.91× at `W = 8`** (`L = 4`, i.e. 16-byte runs — *half* a sector)
  against 3.34×–3.40× at `W = 4`. Below the warp width the kernel is not sector-bound: body 3 is idling `32 - red_len`
  lanes and recovering them dominates a wasted half-sector. The floor re-asserts itself at the long end, which is
  exactly where the shipped table turns to `W = 4`.
* **"Only `double` and `complex<double>` need body 5." REFUTED — all four types emit it.** The plan's budget rested on
  a grid whose minimum `red_len` was 64. Walking `red_len` to 1 shows every type collapsing below the warp width,
  because body 3 puts 32 lanes on the reduction whatever its length: at `red_len 8`, body 3 runs 261 GB/s for float
  and 450 for cfloat against a 950 roof, where body 5 is worth 5.91× and 3.98×.
* **The `latrd` `symv` opportunity is not this work package** (a different kernel), and **per-call route resolution is
  not a regression**: 0.164 µs total (0.077 µs `sub_group_sizes`, 0.067 µs `getenv` plus two `std::string`
  constructions), 2–3% of a minimal batched launch.

## Correctness findings

Across every timed sweep here `relerr` is exactly 0 (468 baseline rows, 840 A/B rows, 2052 audit rows, 1152 repair
rows) — **and that is not evidence of numerical quality.** The A/B harness generates `h * 0.0625` for `h ∈ [0,16]`, so
every product is a multiple of 1/256 and every sum out to `red_len = 2048` stays under 2^24: a float reduction in
*any* order is bit-exact and `relerr == 0` is guaranteed by the data, not earned by the kernel. It cannot detect a
precision regression at all, and it is armed against a *structural* error only **partly** — the imaginary part is
0.5× the real part, so a dropped complex cross-term does move it. This is the same blind-guard shape the test work
exists to close, one level up in the instrument. The correctness
evidence is `tests/gemv_tests.cc` and its breaks. Two contracts that are silently wrong if mis-stated, both matched and both tested. **(1)** Reference `?GEMV`
quick-returns on `m == 0 || n == 0 || (alpha == 0 && beta == 1)` and leaves `y` **completely untouched** — it does
*not* compute `y = beta*y`. Both vendors match, so a native path that scaled `y` would return a **route-dependent**
wrong answer. `A` is also never read when `alpha == 0`, so a NaN in `A` cannot leak into `y = beta*y`. The two halves
of the quick return are tested on *opposite arms*, each where the launch could actually write: `n == 0` under
`NoTrans`, `m == 0` under `Trans` (under `NoTrans`, `m == 0` gives an empty launch and the test would be vacuous).
**(2)** **No `__restrict__` on any pointer** — `ortho.cc:227-232` passes `A_i` and `A_next` as views into the *same*
allocation; they are element-disjoint but alias at the object level, and `__restrict__` promises about the object.

### Blind guards found and closed

1. **The pre-WP7 fixture (40 `GemvMatrixViewTest` cases).** Fixed 10×10, batch 5, `ld == rows`, `inc == 1`, square
   only, `ConjTrans` never used, `beta != 0` in exactly one test — and **the complex tests use purely real data**, so
   every imaginary cross-term is identically zero and two tests compare only `std::real`. A kernel that dropped every
   complex cross-term, got `ConjTrans` backwards, or ignored `ld`/`xinc`/`yinc` passed all forty. Measured: breaks
   `cross`, `conj`, `ld`, `xinc`, `yinc`, `segld`, `segxinc`, `segyinc` each leave all 40 **green** while turning
   coverage cases red (`cross` 84/0, `segld` 20/0, `segxinc` 16/0, `segyinc` 20/0). `ConjTrans` is the **live
   production path** — `ortho.cc:119-120` selects it for **both** complex types (`ab/README.md` says "all four", but
   the ternary is `std::is_same_v<T, std::complex<float_t>> ? ConjTrans : Trans`, and only on the `NoTrans` arm) —
   and it had no coverage and no measurement at all.
2. **The ninth blind guard — the natural batch stride.** All 232 cases in the suite as it then stood (40 pre-WP7 +
   192 new) used `a_stride == ld*n`,
   `x_stride == size*inc`, `y_stride == size*inc`, so a kernel that *derived* each stride rather than reading it from
   the view passed the whole suite — while `ortho.cc:218-222` hands the native path `A.stride() == m*A.cols()` against
   a view whose `ld*cols` is `m*i`, every CGS iteration. Four `stride_pad` cases, one per body; break `padstride`
   turns exactly 32 red, nothing else.
3. **The twelfth blind guard — no guard band past `y`.** Body 5's tail sub-group covers `W` outputs and can run past
   the last one; its correctness rests on a mask and a clamp. **Three separate breaks against that pair came back
   green over 376 cases**, because an out-of-range write landed past the end of the allocation where nothing was
   looking. `run_case` now allocates 64 elements of guard, poisons them before the call and asserts them untouched
   after; `segTtailwrite` and `segTclampoff2` then turn exactly the three partial-tail cases red
   (`out_len*batch mod W != 0`) and nothing else.
4. **`tests/route_vocabulary_tests.cc` had zero `Op::gemv` assertions** and did not include `route_gemv.hh`. Proved
   blind from two directions before being closed: vendor-present, **all 114 `gemv` decisions in a full `ctest` capture
   resolve to `vendor:auto`**, so no vendor-present test could observe the table at all; vendor-free, pinning
   `native:direct` removes the CTA kernel from every decision and all `gemv_tests` still pass. It now covers all three
   CTA gates, the Direct arm's absent `is_gpu` clause, `kGemvOrder`, the `out_len()`/`red_len()` swap, the
   all-false-for-three-types claim, and every boundary of the cdouble window from both sides — arming proved by 7
   breaks including `unarmed` (the V1 trap itself).

Two anti-vacuity devices are worth keeping. `SegTransCasesAreReachable` **skips with a message naming the numbers**
when `16*CU`/`64*CU` exceed the body-5 cases' own `out_len*batch` (2385 and 8288) — on a bigger device every body-5
case would silently become a body-3 case and the section would stay green while proving nothing. And
`gemv_seg_trans_width_debug` resolves through the *same* gate function the launcher calls: two copies of one boundary,
the driver's flipped by a break while the test-visible copy keeps the old sense, is a recorded campaign defect.

### Breaks that stayed green

Recorded rather than dropped, because each is a claim about what a test *cannot* see:

| break | what it does | why green is correct |
|---|---|---|
| `flatten` | mis-pairs `b` and `i` in the flattened index | the wrong mapping is still a **bijection** onto the same `(b, i)` set and each work-item derives its own addresses, so every pair is computed exactly once. The flattening is a *performance* property; no correctness test can observe it. 176/176 passed |
| `segactive` | drops the `jsub < W` half of body 4's accumulate guard | the fold is **closed**: lane `i`'s total draws only from lanes `i + m*t`, so the silenced lanes were already unread. A work saving, not a correctness condition. 232/232 passed |
| `segTtail` | `return` instead of masking body 5's partial tail | the same closure from the other side. The mask is a **spec-conformance** requirement (a shuffle reached by part of a sub-group is UB), not an observable-value one. No test in any suite can catch it |
| `segTclampoff` | removes the clamp, keeps the mask | the clamp only affects addresses an inactive lane group *forms*; its partner `segTclampoff2` (both removed) **is** red, showing the pair is load-bearing together |
| `segTlaunch` | leaves the sub-group count at `out_len*batch` instead of `/W` | the extra sub-groups all have `base >= total` and return: a W-fold over-launch, a performance defect, invisible to any value check |

The armed breaks were each applied, rebuilt in `build-novendor`, run and reverted: **14** against body 4 and **24**
against body 5 (`breaks_kernel.py`, `breaks_body5.py`); 13 and 21 of them turn cases red and the rest are the green
ones tabled above. Between them they cover `ld`, `xinc`, `yinc`, the three per-view batch strides, the lane map, both folds (real and complex
are separate code), the write mask, conjugation, `alpha`/`beta`, and every gate edge. Three route-clause breaks guard
the shipped window: `gemv_axisswap` (spells the band on `out_len`, inverting it), `gemv_nobatch` (drops the batch
floor, admitting 0.9562) and `gemv_alltypes` (drops the type gate, admitting 0.9340 / 0.9722 / 0.6644). Notable
single-case arming: `segwidth34` (an off-by-one in `w * 2 * out_len <= 32`, made 34) turns **only**
`SegmentGateBoundaryNoTranspose` red, `out_len == 17` being the only length with `32 < 2*out_len <= 34`; `segfold` is
an identity at `m = 1`, which is why the body-4 ladder walks `m ∈ {1, 4, 10, 16, 17}`; and direction is tested in both
senses — `conj` (ConjTrans stops conjugating) turns exactly the 12 complex ConjTrans cases red, `conjalways` (plain
Trans conjugates too) exactly the 20 complex plain-Trans cases, since one break can only move one of them.

### The known bad caller

`src/extensions/ortho.cc:216-224`'s `transA = Trans` branch builds `A_i` as `i × m` with `ld = m` and passes
`A(Slice(), i)` — a column of length `A.rows()` — as `x`, so the lengths agree only in the accidental case
`A.rows() == m`. **It is structurally wrong today, under the vendor**, and WP7 deliberately neither fixed it nor threw
on it (a new host-level validation throw would turn today's silent misbehaviour into a crash in a live path). The
length checks in `gemv_op_shape` (`src/backends/gemv_route.hh:73-76`) guarantee it returns `nullopt` → the vendor,
i.e. it keeps going exactly where it went before WP7 rather than becoming a native out-of-bounds read. Fixing it needs
the right `A_i`, an `A_next` that is the *i*-th vector rather than the *i*-th column, and an
`ortho(..., Transpose::Trans)` test that checks orthogonality of the **rows** — which `ortho_tests` does not have,
which is why it survived. Otherwise what the library issues is fine: over the 56 cells `ortho.cc` actually calls
`gemv` on (`i ∈ {1..64}`, `m ∈ {512, 2048}`, batch 512, float and cdouble) the worst is **0.75×**, the median
**1.14×**, and 49 of 56 are at or above cuBLAS — the 0.08× family needed a short *output*, and `ortho` only ever gives
the `NoTrans` body a short *reduction*.

## Open debts

* **The clause's upper batch and `out_len` corners are unbracketed.** Nothing in the fitting grids was measured above
  `batch 1024` or `out_len 2048` (confirmed by reading `g6_fit{,2}_p*.csv`: max batch 1024, max `out_len` 2048), yet
  the predicate admits every batch above 320 and every `out_len` above 256. A cell list exists for the **batch** half
  — grid J, `out_len` 256/512 × `red_len` 64/128/256 × batch 2048/4096/8192 — and was not run; **nothing at all
  brackets `out_len` above 2048**, not even as a cell list.
* **The batch floor of 320 is the lowest threshold the search *enumerated*, not the lowest that wins.**
  `g6_clauses.py`'s `BT` list is `[1, 64, 128, 192, 256, 320, 384, 448, 512, 640]` — it contains no 288. The
  two-spelling grid did measure `batch 288` inside the band, on one pass and `Trans` only, and **all six cells win**:
  1.2742 / 4.4478 / 1.6973 at `out_len 256` and 2.3382 / 2.5103 / 2.8884 at `out_len 512` (`red_len` 64/128/256).
  So the floor gives up a measured rung for the same reason `out_len >= 768` was declined — an unbracketed edge —
  except that here the edge is the **threshold list**, not the sampled range. Bracketing it needs batch 272…320 in
  two passes and both spellings.
* **`out_len >= 768 && batch >= 128` is measured and unrouted** — ~18 cells at 2.26×–2.91× left on the table because
  its batch floor is the edge of the sampled range. Its bracketing sweep is one cell list (grids H and I).
* **The upper `red_len` edge of 352 is the boundary for the *smallest* admitted `out_len`, not a universal one.** At
  `out_len 2048` the vendor is still dipped at `red_len` 384 and 448 (313.6 and 317.2 GB/s, ratios 2.998 and 2.973)
  where at `out_len` 768 and 1024 it is back at the roof at the same `red_len` and batch (900.9 / 899.8 GB/s, 1.0406 /
  1.0426); it closes for good at `red_len 512` even at `out_len 2048` (927.7 GB/s, 1.0166). Encoding the movement
  needs a two-variable boundary fitted on four `out_len` levels; the cells are captured, the fit is not attempted.
* **The clause was fitted on the display GPU** (`gpu=0` on every `g6_fit` row). Only its lowest-footprint cell was
  re-measured across devices; the rest of the band rests on a cross-device control taken at DRAM-resident footprint.
* **The L2-resident window above the body-5 gates is measured and not taken.** At `out_len 256, batch 512` (33–67 MB
  against a 72 MB L2) body 5 at `W = 4` measures **1.40×–2.09×** for cfloat at `red_len 24..64`, **2.62×** for double
  at `red_len 64` and **1.22×–1.71×** for float at `red_len 48..128` — all *above* their gates, while the same
  `red_len` at `out_len 2048` measures 0.986×–0.996×. Separating them needs a **footprint** term, which is the
  L2-residency reasoning `route_gemv.hh:279-284` forbids and which would be no better founded in a launcher.
* **`17 <= out_len <= 31` on the `NoTrans` arm is unmeasured** — body 4 declines it by arithmetic (`W == 1`), and no
  timing brackets that side of its gate. **`complex<double>` transposed at short reduction is cleared, not solved**:
  body 5 lifted the last two sub-0.50× cells to 0.862/0.861, which is not parity and cannot be.
* **Instrument limit.** A `gemv` coverage row **cannot** confirm a particular *shape* ran: `coverage.cc` keys rows on
  a power-of-two `shape_class`, first-writer-wins, so the m/n/batch columns can report another call's shape. What a
  coverage dump *can* settle is reachability, and it did once here: `lanczos_tests` fails identically in both builds
  and its dump holds only `linked,gemv` rows with **zero `reached` rows** — it never calls `gemv` at all, which is
  how that failure was excluded from WP7's ledger rather than assumed out of it.
* **Not verified in this record:** the count of route decisions the shipped clause actually moves in a live capture.
  The intended move is enumerated (`vendor:auto → native:cta` for cdouble, `transA != NoTrans`, `red_len` 64..352,
  `out_len >= 256`, `batch >= 320`) and a pure-layer probe admits **384 grid cells, every one cdouble, every one
  `native:cta`** — but no post-clause `route_diff` output was found in the sources read. The body-5 pass, which by
  construction moves nothing, *was* diffed: **0 removed decisions, 60 added, gemv-only, 0 non-gemv rows moved.**

## Raw evidence

Raw data is preserved at tag `perf-evidence/vendor-independence`; retrieve any path with `git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| the vendor baseline, the DRAM roof, the cdouble dip and its `ld`/type/batch controls | `experiments/wp7_gemv/baseline/README.md`, `vendor_baseline{,_p2}.csv`, `refine.csv`, `close.csv`, `grid.txt` |
| the 84-cell parity grid, the ~1 GB `m × n` map, the batch ladder, the refuted `n*batch` rule | `experiments/wp7_gemv/ab/README.md`, `ab_p{1,2}.csv`, `refine_c_p{1,2}.csv`, `batchdep_p{1,2}.csv`, `outelems_p{1,2}.csv` |
| the `(out_len, red_len)` parity audit, the 15 blockers, `ncu` geometry and mechanism, the prize grid, the clause search | `experiments/wp7_gemv/audit/README.md`, `parity_p{1,2}.csv`, `blockers_p3.csv`, `geometry.csv`, `mechanism.csv`, `prize_p{1,2}.csv`, `oos_p{1,2}.csv`, `clause_search.py`, `clause_report.txt`, `typecheck.csv`, `ortho_shapes.csv` |
| body 4, the runtime-vs-`constexpr` `W` result, the re-measured parity ladder, the break campaigns | `experiments/wp7_gemv/repair/README.md`, `parity_r{1,2}.csv`, `outlen_body{1,4}.csv`, `mechanism_body4.csv`, `breaks_kernel.py`, `breaks_route.py` |
| body 5: the FP64-pipe mechanism, the `W` tables, the three gates, the skinny defect, odd `ld`, registers, link time | `experiments/wp8_gemv/README.md`, `ncu_precheck.csv`, `typecurve.csv`, `wchoice_p1.csv`, `wfine_p1.csv` (there is no `wfine_p2.csv` in the tree, although `gemv_native.cc` and the README cite `wfine_p{1,2}`), `plane_p{1,2}.csv`, `planeF_p{1,2}.csv`, `conjF_p{1,2}.csv`, `plane_cells.txt`, `skinny_p{1,2}.csv`, `skinny2_p{1,2}.csv`, `skinny3_p{1,2}.csv`, `resid_p{3,4}.csv`, `above_p{1,2}.csv`, `oddld_p1.csv`, `regs.csv`, `linktime.sh`, `breaks_body5.py` |
| the cdouble window: the fitting grids, the clause scorer, the named candidates with their refuting cells, the unbracketed corners | `experiments/wp8_gemv/g6_fit_p{1,2}.csv`, `g6_fit2_p1.csv`, `g6_clauses.py`, `g6_score.py`, `g6_summary_p1.txt`, `g6_summary_2pass.txt`, `g6_summary_fit2.txt`, `g6_cells3.py` |
| the routing pass's gates: the pure-layer clause probe, the admitted set, the `preferred()` breaks, `route_diff` health | `experiments/wp8_route/clause_probe.cc`, `admitted.csv`, `breaks.py`, `after.sh`, `health.sh`, `gate_a.sh` |
| the three defects found and deliberately not fixed (ortho `Trans`, `cond`'s vendor bypass, lanczos's discarded gemm column) | `WP7_FILED_DEFECTS.md` |
| campaign narrative, the gates re-run after the repair, the four shipped windows | `VENDOR_INDEPENDENCE_PLAN.md` (WP7, "WP7 REPAIR PASS", "The WP6/WP7 performance-closure pass") |
