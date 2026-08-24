# WP6-PERF — the LU performance pass, both halves

WP6 landed native `getrf` / `getrs` / `getri`, correct and fully tested, and shipped
**route-neutral** because the family lost to cuBLAS overall. This pass is the
performance follow-up. It has two halves, and **one of them is a refutation**:

| half | what was tried | outcome |
|---|---|---|
| **A** — packed sub-group `getrf` | one sub-group per matrix, shuffle argmax, no work-group barriers | **REFUTED and not implemented.** Wins only for 32-bit types at `n ≤ 32`, and even the best cell leaves the shipped arm at 0.81× of cuBLAS. §1 |
| **B** — fused narrow-RHS `getrs` | one kernel: permutation + both substitutions, no GEMM, no separate laswp | **KEPT, and the routing default moved.** 2.08× of cuBLAS on 63 in-window cells with zero losses, against 0.32× before. §2–§6 |

Everything below is measured on this box (2× RTX 4090, 128 SMs, GPU 1 pinned,
1008 GB/s DRAM peak), through the **public API**, on WP6's own harness
(`experiments/wp6_lu/bench/lubench6.cpp`) with WP6's own build scripts and cell
format, against an in-process host oracle on every timed row, with the resolved
route printed and read on every row, and nothing under `BATCHLAS_KERNEL_TRACE`.

---

## 0. The one-paragraph answer

`getrs` was WP6's worst op: **0.32× of `cublas?getrsBatched` at `nrhs = 1`**, because
`trsm`'s blocked driver was being asked to amortise a panel over a single column and
the permutation was a second launch costing 26.4% of the call on its own. The fused
tier replaces the whole composition with **one kernel per matrix** and is **2.08×**
of cuBLAS across the 63 grid cells that now route to it, **zero losses, minimum
1.12×**, and 8.26× faster than the composition it replaces. `preferred()` now
carries that window — `nrhs ≤ 2` for every type, plus `nrhs ≤ 4` for `float` — so
**the default moved in every build**: `scripts/route_diff.sh` reports **27 decisions
changed, all of them `getrs`, all `vendor:auto → native:cta`, and no other op
touched.** `getrf` and `getri` are unchanged.

---

## 1. HALF A — the packed sub-group `getrf` arm, PROTOTYPED AND REFUTED

`proto/pivsg.cpp`, `proto/ab.csv`, `proto/settle.csv`. Built with `proto/build.sh`,
run with `proto/run_ab.sh` and `proto/run_settle.sh`.

**The idea.** The shipped small-`n` `getrf` CTA arm uses one *work-group* per matrix
and pays a work-group barrier at every panel step. A *sub-group* arm — one sub-group
per matrix, G matrices per work-group, argmax by shuffle — pays no work-group
barriers at all. Three passes × 61 reps; medians reproduce to 3–4 significant
figures across passes.

`pivman_ms / pivsg_ms`, so **> 1 means the sub-group arm is ahead**:

| | n=16 | n=24 | n=32 | n=48 | n=64 | n=96 | n=128 |
|---|---|---|---|---|---|---|---|
| float | **1.38** | **1.25** | **1.13** | 0.79 | 0.64 | 0.39 | 0.31 |
| cfloat | **1.39** | **1.19** | **1.03** | 0.78 | 0.66 | 0.36 | — |
| double | 0.81 | 0.83 | 0.88 | 0.88 | 0.60 | 0.34 | — |
| cdouble | 0.91 | 0.89 | 0.92 | 0.77 | 0.64 | — | — |

**Why it was refuted, and it is not the ratios.** It wins only for 32-bit types at
`n ≤ 32`, and **it changes no routing decision even there**: the shipped native
`getrf` is 0.551× of cuBLAS at float `n = 32` (0.2883 vs 0.1589 ms) and 0.584× at
`n = 16` (0.0752 vs 0.0439). The best cell in the table, 1.38×, therefore lands at
**0.81× of cuBLAS** — still a loss. A new tier plus a load-bearing type-and-order
gate, to move a route that stays on the vendor either way, is cost with no decision
attached. The regression above `n = 32` has a mechanism worth keeping: 32 lanes per
matrix costs more trailing-update parallelism than the saved barriers buy.

**Also established, and it closes the obvious next idea:** the shipped small-`n`
`getrf` is **already one fused kernel** — nsys shows a single
`GetrfPanelResidentKernel<float>` and no other kernel in the profile — so there is
no launch-count win available at small `n`. The remaining gap lives *inside* that
kernel and closing it is kernel engineering, not routing.

---

## 2. HALF B — the fused narrow-RHS `getrs` tier

`src/extensions/getrs_fused.cc`, routed as `{Native, CTA}` in `route_getrs.hh`.

### The mechanism it removes

nsys, vendor-free, float `n = 512` `nrhs = 1` batch 512, **one public-API call**,
the composed arm:

```
  29.3%  TrsmCtaKernel<float,32,Side=Left>          37,967 instances
  26.4%  LuLaswpKernel<GetrsLaswpTag>                1,186 instances, 659 us each
  39.7%  GemmTiledGeneralKernel<float,16> x3       ~26,000 instances
   3.1%  GemmDirectKernel<float>                     9,489 instances
```

The 39.7% is **matrix–vector products run through a tile-16 GEMM kernel**. The fused
tier issues **one kernel**: the interchange walk, the forward substitution against
unit-lower `L`, and the back substitution against `U`, one work-group per matrix,
parallel over rows, with the RHS block and one `nb × nb` diagonal block resident and
`L`/`U` streamed. At `n = 512` float the matrix is 1 MB per item, so a
CTA-resident-matrix design was never available.

### What was KEPT, with its number

| kept | number | against |
|---|---|---|
| the fused tier itself | **8.26×** at `nrhs=1`, 6.56× at 2, 4.40× at 4, 2.67× at 8; 107 pooled cells, **zero losses**, worst 1.419 | the composition |
| … the same, vs the vendor | **2.084×** geomean, 63 shipped-default cells, min 1.122, max 4.922, **zero losses** | cuBLAS |
| a resident diagonal block rather than pure streaming | 0.6506 vs **1.0102** ms (float n=512 b=512) | the streaming variant, which is a reverted arm with its number |
| `nb = 16` below `n=1024`, `nb = 32` at and above | n=2048 b=32: nb 8 1.5513 / nb 16 1.3772 / **nb 32 1.2838** | its own sweep |
| the folded permutation | 26.4% of the composed call removed | a separate laswp launch |
| the exact per-(type, body, width) register cap | **1.027–1.062×** at `nrhs=8` (§7a) | the max-over-everything cap it replaces |

### What was REVERTED or REFUSED, with its number and mechanism

| rejected | number | mechanism |
|---|---|---|
| pure streaming (no resident block) | 1.0102 vs 0.6506 ms, **0.64×** | a work-group barrier per *column* instead of per *block* |
| window `C4` (float `nrhs ≤ 8`) | 3 losses, worst **0.686×** | float n=2048 nrhs=8 at batch 4–16: the CTA count *is* the batch |
| window `C6` (`nrhs ≤ 4`, every type) | 20 losses, worst **0.577×** | cdouble n=32: the block solve is 16 lanes of one sub-group |
| window `C7` (the whole capability) | 55 losses, worst **0.294×** | being *able* to run a shape is not evidence you should |
| windows `C8`/`C9` (`nrhs ≤ 4, n ≥ 128` + a batch or order bound) | 4 losses each, **0.940–0.987×** | the dips are **mid-ladder**, so no boundary in `n` or batch reaches them. These were the leading proposal until the third flatness pass |
| the `+1` bank-conflict pad on the block's leading dimension | **kept anyway**, but see §7 — its own A/B is 2.17% *against* it at the one cell with the largest signal | portability, not measurement |

---

## 3. BEFORE / AFTER

### 3a. Both arms reproduce WP6 before any ratio is quoted

`check_wp6.py` → `bench/wp6_reproduction.txt`.

| arm | shared cells with `wp6_lu/bench` | worst disagreement | cells above 5% |
|---|---|---|---|
| cuBLAS | 60 | **2.90%** | 0 |
| the composition (BEFORE) | 42 | **0.76%** | 0 |

WP6 published float `nrhs=1` `0.214 / 0.515 / 0.568 / 0.647 / 0.606` at `n = 64…2048`;
this directory measures `0.213 / 0.508 / 0.566 / 0.648 / 0.604`. cdouble reproduces
to three decimals at all five orders.

### 3b. WP6's own grid, 168 cells, geomean `cuBLAS_ms / native_ms`

```
  nrhs             1        2        4       16       64      128
  BEFORE       0.256    0.331    0.326    0.457    1.094    1.478
    wins        0/28     2/28     2/28     3/28    19/28    24/28
  AFTER        2.117    2.173    1.421       = BEFORE (tier not instantiated)
    wins       28/28    28/28    22/27
```

**WP6's "0.32× at `nrhs=1`, rising to 1.36× at 128" is inverted: 2.12× at `nrhs=1`,
28 wins in 28.** WP6's worst cell in the whole LU family — `getrs` cdouble `n=512`
`nrhs=1` at **0.083×** — is now **1.765×, a factor of 21**.

### 3c. What a **vendor-free** build ships across the whole grid

Fused where the capacity gate admits, composition elsewhere (83 / 85 cells):
**0.523× → 1.291×** of cuBLAS, 50 → 124 wins, the op itself **2.468×** faster.

### 3d. What an **unpinned, vendor-present** user gets — the shipped default

`bench/run_default.sh`, no route variable set at all, scored against the same cells
with the vendor pinned (`bench/default_summary.txt`):

```
  route column agrees with the window predicate on ALL 84 cells
  INSIDE  the window   63 cells  geomean 2.084  min 1.122  max 4.922  losses 0
  OUTSIDE it           21 cells  geomean 0.998  min 0.982  max 1.005
```

The outside rows compare the vendor to itself across two sessions and land within
±1.8%, which is this harness's session-to-session floor.

---

## 4. THE ROUTING DECISION

### The window

```cpp
static bool preferred(Route r, const GetrsShape& s) {
    if (!is_native(r)) return false;
    if (r.algo != Algorithm::CTA) return false;   // the composition, never
    if (s.nrhs() <= 2) return true;               // clause A
    if constexpr (std::is_same_v<T, float>) {     // clause B
        if (s.nrhs() <= 4) return true;
    }
    return false;
}
```

Scored over **461 pooled cells** from seven sweeps, deduplicated on
`(type, n, nrhs, batch)`, both arms' routes read from the printed route column
(`bench/analyse_window.py`, `bench/window_summary.txt`):

| clause | cells | geomean | min | losses |
|---|---|---|---|---|
| A — `nrhs ≤ 2`, every type | 286 | **2.261** | 1.116 | **0** |
| … `nrhs = 1` | 142 | 2.290 | 1.242 | 0 |
| … `nrhs = 2` | 144 | 2.232 | 1.116 | 0 |
| B — `float`, `nrhs = 3…4` | 36 | **1.611** | 1.133 | **0** |
| **both** | **322** | **2.177** | **1.116** | **0** |

**It is flat in batch, and that took four passes.** Full `SAT_LADDER` ladders at
`n = 32, 64, 128, 256, 512, 1024, 2048`, all four types, at `nrhs = 1`, `nrhs = 2`
and `float nrhs = 4`: **zero of the 63 ladder rows crosses 1.0 anywhere inside the
window**, over **322 laddered in-window cells** across `flat`, `flat2`, `flat3` and
`flat4`, lowest rung **1.116×**.

**`flat4` exists because a review caught that it did not.** Before it, there was
**no order-32 ladder in this directory at any width**, so the small-`n` end of
clause A — including the window's own stated minimum — rested on a single
saturating batch point. That is the one-cell over-fit this campaign keeps paying
for, and it was one review away from shipping again. `flat4` laddered `nrhs = 2` at
`n = 128` and `n = 1024` (the two interior orders clause A turns on), and `nrhs = 1`,
`2` and `float 4` at `n = 32` and `n = 256`. **Result: every one FLAT-WIN, zero
crossings.** The minimum moved from a single point at 1.123× to a full ladder whose
lowest rung is **1.116×**.

### Why clause B is float-only

The other three types **cross 1.0 mid-ladder** at `nrhs = 4`:

```
  double  n=128   0.940x at batch 2048   (1.363x at 256, 1.111x at 8192)
  cfloat  n=1024  0.976x at batch 16
  cdouble n=128   0.980x at batch 1024
  cdouble n=1024  0.987x at batch 16
```

— and cdouble is **0.577×** at `n = 32` outright. A dip in the *middle* of a ladder
cannot be closed by any boundary in `n` or in batch.

### What the window costs, stated

**84 measured winning cells are handed to the vendor**, the largest at **3.944×**
(`double n=1024 nrhs=4 batch 256`), then 3.144×, 3.097×, 2.880×, 2.745×. They are
given up because the clause that captures them dips below 1.0 elsewhere on its own
ladder. Recovering them is new work — a per-`(type, order)` predicate measured at
more orders, or a kernel fix for the dip — not a constant.

### The thinnest margin in the window

`cdouble n = 32 nrhs = 2`, whose ladder runs **1.257 / 1.162 / 1.132 / 1.120 /
1.116** at batch 1024 → 16384. It declines with batch and *flattens* rather than
falling, so it is a flat win by the rule — but it is the only cell in 322 under
1.12×, and it is the first place a re-measurement on another box should look.

### `route_diff.sh` — exactly which decisions moved

Captured before and after with `preferred()` the only difference, `ctest -LE slow`
in both builds:

| build | rows | decisions | ops touched | change |
|---|---|---|---|---|
| `build` (cuBLAS present) | 4012 | 3600 | **`getrs` only** | **27** decisions `vendor:auto → native:cta` — float ×18, double ×3, cfloat ×3, cdouble ×3 |
| `build-novendor` | 3731 | 3437 | **`getrs` only** | **14**, all `Backend::AUTO` |

The vendor-free build's 14 are the *synthetic* shapes `route_vocabulary_tests`
resolves at the pure layer with `vendor_available = true`; **every real
(`Backend::CUDA`) vendor-free decision is byte-identical**, which is what it should
be — a vendor-free build already reached the fused tier through
`native_tier_preferred`. `getrf`, `getri`, `gemm`, `trsm`, `orgqr`, `ormqr`,
`potrf`, `geqrf` and everything else: **zero rows changed.**

### What was NOT landed, and why

The **composition** wins against the vendor at `nrhs = 64` (geomean 1.094×, 19 wins
of 28) and `nrhs = 128` (1.478×, 24 of 28). It is **not** in `preferred()`: those
columns carry 9 and 4 losses respectively and **no batch ladder exists anywhere on
that axis**. That is a window-shaped opportunity, not a window. It is open work.

---

## 5. EVERY BREAK, AND WHETHER IT TURNED RED

Each break: patch the source, **rebuild the `.so`**, re-run the whole binary.

### 5a. The fused tier (16 breaks, recorded in `tests/getrf_tests.cc`)

15 of 16 RED. `piv_base` 28, `rhs_ld` 24, `unit_u` 24, `trans_perm_forward` 20,
`perm_wrong_side` 20, `swap_solves` 19, `last_row` 13, `conj` 6 (cfloat + cdouble
only, correctly), `cap_inversion` 4, `hole_pad` 4, `reg_cap` 1, `facade_arm` 4,
`tier_pref` 4, `supports_gates` 8, `dispatch_gates` a process abort. **`cap_band`
turned nothing red**, correctly — after the capacity repair the clamp and the
re-check close the same hole from opposite ends. A **17th** break, `B5` (the `+1`
pad removed), is recorded in `src/extensions/getrs_fused.cc` and **also turned
nothing red**, correctly: it is a performance choice.

> The break record's own prose said *"fourteen … thirteen of the fourteen"* over a
> sixteen-row table with fifteen REDs. Corrected in this pass. Counting the
> campaign's own evidence wrong is the WP5 *"zero suites closed"* failure again.

### 5b. The window (5 more, run in this pass)

| break | what was corrupted | `getrf_tests` | `route_vocabulary_tests` |
|---|---|---|---|
| **W1** | clause A switched off | **RED, 6** | **RED, 1** |
| **W2** | clause B widened to every type | **RED, 3** | **RED, 2** |
| **W3** | the composition also made preferred | green | **RED, 2** |
| **V1** | the `getrs_shape()` fused capacities back to 0 | n/a | **RED, 3** |
| **R1** | the register cap removed | **RED, 1** (launch abort) | n/a |

**Four of the five outcomes are themselves findings:**

* **W1 leaves float green, and that is correct.** With clause A off, float `nrhs = 1`
  is still inside clause B. Only double, cfloat and cdouble move.
* **W3 turns nothing red in `getrf_tests`, and that is what the pure suite is for.**
  Making the composition preferred changes no *resolved* route, because CTA is
  first in `kGetrsOrder`. Only a direct assertion on `preferred()` sees it.
* **V1 is the blind guard made visible** — see §6.
* **R1 reproduces a hard launch abort**, not a wrong answer:
  `"Exceeded the number of registers available on the hardware. The kernel uses 68
  registers per work-item for a total of 1024 work-items per work-group."`

---

## 6. THE REVIEW FINDINGS, TRIAGED

Twelve findings were handed to this pass. **Nine confirmed, one refuted, two
accepted as verification.**

| # | severity | verdict | disposition |
|---|---|---|---|
| 1 | NIT | **verification, no defect** | An adversarial sweep found no wrong answer in the fused tier. Accepted; nothing to fix. |
| 2 | PERF | **CONFIRMED** | The register cap charged every kernel the widest kernel's registers. **Fixed** — §7. |
| 3 | HYGIENE | **CONFIRMED** | Break record miscounted itself (14/13 vs 16/15). **Fixed** in `tests/getrf_tests.cc`. |
| 4 | HYGIENE | **CONFIRMED** | The implementer report quoted 63/55 (pre-change). The real figures are **95 / 87** and are used throughout this document. |
| 5 | HYGIENE | **CONFIRMED** | `BATCHLAS_GETRS_ROUTE=native` changed meaning when CTA joined `kGetrsOrder` first. **Documented** in `route_getrs.hh`'s header and **asserted** in `BareOriginResolvesToASpecificAlgorithm`. Pin `native:blocked` to mean what `native` used to. |
| 6 | NIT | **CONFIRMED** | The report's closing gitignore warning was already stale: `pivsg` *is* ignored (`proto/.gitignore:5`). No action needed; see §8. |
| 7 | CRASH | **REFUTED** | *"No test reaches a shape where the register cap can bite."* It does. `FusedGetrsLaunchHoleAt48KiB`'s top rung is `n = 1428` at `nrhs = 8`, picks `wg = 1024`, and runs `transA = Trans`. **Break R1 reproduced the abort and the test went RED.** The review looked only at the two tests whose names mention widths and boundaries. |
| 8 | HYGIENE | **CONFIRMED, and it was the worst of the twelve** | `getrs_shape()` never set the fused capacities, so `supports({Native, CTA})` was false on every shape in the pure suite and **every getrs routing assertion in it held regardless of the table**. **Fixed**; break V1 proves the repair load-bearing. |
| 9 | HYGIENE | **CONFIRMED** | The 2.1× shipped in no default build. **Fixed** — the window is §4, and `route_diff` shows the 27 decisions that moved. |
| 10 | PERF | **CONFIRMED, arithmetic reproduced exactly** | *"82% of DRAM peak, ceiling reached"* holds only in an `n = 256…512` band. Per-cell table now in `getrs_fused.cc`'s header and in §7. |
| 11 | PERF | **CONFIRMED** | The folded permutation is 8.0% of the call at `n = 2048` and is the one fully serial part. Recorded as the next lever; **not** fixed in this pass. |
| 12 | NIT | **CONFIRMED** | The `+1` pad's own A/B is 2.17% *against* the spelling that was kept, at the cell with the largest signal. Comment **corrected** to say so; the pad stays for portability, which is now stated as the actual reason. |

### The one that mattered most

Finding 8. `route_vocabulary_tests` reported **78/78 through the window flip and
through its inverse** — the grid agent noticed the helper gap but reported it as
*"stays 78/78, and that is not evidence"*. It was worse than uncovered: two
assertions in it asserted the **opposite** of live behaviour and passed only because
the helper described a device on which the tier under test cannot run. Both are
rewritten around the window, both sides of it.

---

## 7. THE REPAIRS THIS PASS MADE, EACH WITH ITS NUMBER

### 7a. The exact register cap — **KEPT**, +2.7% to +6.2%

`getrs_fused_wg` capped the work-group at `65536 / (regs + 8)` using **one register
number per width, the max over both kernel bodies and all four scalar types**. At
`NR = 8` that number is 86 (`GetrsFusedTKernel<complex<double>,8>`), giving
`wg = 672` — while `GetrsFusedNKernel<float,8>` uses **48** registers and fits 1024.
Both coordinates are compile-time known at the call site, so the cap is now per
`(type, body, width)`. Measured with `scripts/register_probe.sh` on
`batchlas_extensions_factorization`, **528 entry functions, 0 with spill**:

```
             NR=1        NR=2        NR=4        NR=8
  type    NoTr  Tr    NoTr  Tr    NoTr  Tr    NoTr  Tr
  float    39   39     48   40     48   48     48   68
  double   39   46     52   44     44   51     61   72
  cfloat   40   42     40   43     40   48     48   56
  cdouble  54   56     56   58     56   58     72   86
```

**The A/B** (`regcap/`, three passes per arm — the two arms are two *builds*, so the
cross-pass median spread stands in for interleaving; every pass verified
`native:cta` from the route column):

| cell | before | after | b/a | cross-pass spread |
|---|---|---|---|---|
| float n=2048 nrhs=8 b=8…64 | 3.041–3.237 | 2.914–3.110 | **1.041–1.048** | ≤1.1% |
| float n=2048 nrhs=8 b=4 | 2.881–2.901 | 2.809–2.839 | **1.027** | ≤1.1% |
| cfloat n=1344 nrhs=8 b=8…64 | 3.209–3.305 | 3.032–3.155 | **1.048–1.062** | ≤1.1% |
| double n=1344 nrhs=8, all b | 6.232–6.301 | 6.228–6.299 | **1.000** | ≤0.4% |
| float n=2048 nrhs=1 b=8…64 | 1.220–1.529 | 1.220–1.534 | **1.000** | ≤0.5% |

**Kept, but the honest framing is that it is a 4–6% win, not the 1.4× the `wg`
tuning table suggested.** `wg 672 → 1024` is not `512 → 1024`, and `double` shows
**exactly 1.000 at every batch** because at `n = 1344` the kernel is bound by the
dependent recurrence rather than by thread count — more threads buy nothing there.
That is a measured negative inside a kept change and it is why the change is scoped
as small.

**One cell discarded**: `float n=2048 nrhs=1 batch=4` re-ran at 0.854 / 1.012 /
1.071 ms — a **25% cross-pass spread**, on 4 work-groups over 128 SMs. Not
reportable in either direction.

**The routing decision is unaffected**: `C4`'s three losses (float `n=2048`
`nrhs=8` at batch 4–16, 0.686 / 0.778 / 0.877) improve by ~3–5% and remain losses.

### 7b. The DRAM-peak claim — **narrowed**

Achieved fraction of 1008 GB/s at `nrhs = 1`, recomputed per cell from
`bench/grid_cta.csv`:

```
             float   double   cfloat  cdouble
   n=32       72%      38%      95%      24%
   n=64       78%      61%      81%      42%
   n=256      78%      79%      85%      75%
   n=512      82%      86%      88%      83%    <- the band
   n=1024     70%      74%      80%      71%
   n=2048     41%      50%      60%      41%
```

Two named mechanisms, both **open work rather than a ceiling**: at large `n` the CTA
count *is* the batch (32 work-groups on 128 SMs at `n=2048`); at small `n`, `nb=16`
leaves the block solve to 16 lanes of one sub-group.

### 7c. Everything else repaired

* The break record's self-count (§5a).
* `route_vocabulary_tests`' `getrs_shape()` helper and the three assertions built on
  it (§6, finding 8).
* `getrf_tests`' vendor-present `is_vendor` assertions, **replaced** by both sides of
  the window on the real device, plus a new check that the *builder* reports a
  non-zero `fused_max_elems` — the half `route_vocabulary_tests` structurally cannot
  see.
* The `+1` pad comment, the header's *"preferred() is all-false"* paragraph, and the
  `BATCHLAS_GETRS_ROUTE=native` meaning change.

---

## 8. HYGIENE, AND A NEW TRAP WORTH RECORDING

**A stale harness reports a stale route, and it looks exactly like a failed flip.**
The first unpinned run of §3d reported **`vendor:auto` on all 63 in-window cells** —
i.e. "the window did not land". It had. `lubench6.cpp` includes
`src/backends/getrs_route.hh` and resolves the *printed* route in its **own**
translation unit, while the *actual* dispatch happens inside the `.so`. Rebuilding
the `.so` alone therefore leaves the harness printing the **old** table's answer.
Re-running `build_v.sh` / `build_nv.sh` fixed it. **Any `preferred()` change requires
rebuilding every bench binary before its route column can be believed.** This is the
brief's *"verify every route pin took effect"* rule with the instrument, rather than
the environment, as the thing that lied.

**Discards, and the rule applied to them.** `flat4` dropped one cell of 134 for
vendor relative sd > 10% (`float n=32 nrhs=1 b=4096`, relsd 0.268). Re-run at three
passes × nine reps (`bench/run_noisy4.sh`): vendor medians **0.0835 / 0.0811 /
0.0839** (spread 1.035), CTA **0.0310 / 0.0311 / 0.0319** (spread 1.029), ratios
**2.694 / 2.608 / 2.630** — a stable win, and the discard was one noisy pass rather
than an unstable cell. 162 rows were dropped earlier for the `cta` pin falling
through the **capacity** ceiling, which is the gate working; `analyse_shipped.py`
reads those as "shipped route = composition" rather than discarding them.

**Diff hygiene.** No `*.nsys-rep`, no `*.sqlite`, no kernel-trace JSON, no compiled
binary is stageable anywhere under `experiments/`. `git check-ignore -v` confirms
`proto/pivsg`, `bench/lubench6_v`, `bench/lubench6_nv`, `bench/blindguard` and
`wp6_getrs/proto/fusedrs_nv` are all covered by existing `.gitignore` entries; every
untracked file under `experiments/` is text or CSV.

---

## 9. THE GATES

| gate | result |
|---|---|
| `build/tests/getrf_tests` | 200 ran, **95 PASSED, 0 FAILED** |
| `build-novendor/tests/getrf_tests` | 200 ran, **87 PASSED, 0 FAILED** |
| `build/tests/route_vocabulary_tests` | **78 / 78** |
| `build-novendor/tests/route_vocabulary_tests` | **78 / 78** |
| `build`: `ctest -L "blas\|ortho"` | **100% passed, 0 failed out of 23** |
| `build-novendor`: `ctest -LE slow` | **33 PASSED / 23 FAILURES of 56** — the recorded set, **failing-set diff EMPTY** |
| `scripts/register_probe.sh … batchlas_extensions_factorization` | **528 entry functions, 0 with spill**; `regs × wg ≤ 65536` verified for all 32 fused kernels |

The vendor-free 23 are the pre-existing set — `options_api`, `syevx`, `lanczos`,
`gemv`, `trsm`, `ortho`, `cond`, `ormqr`, `ormqr_cta`, `ormqr_blocked`, `orgqr`,
`iluk`, `symm`, `hemm`, `herk`, `her2k`, `syrk`, `syr2k`, `syev`, `trmm`,
`sytrd_blocked`, `syev_cta`, `syev_blocked` — **no LU suite among them**;
`getrf_tests`, `potrf_tests`, `geqrf_tests`, `inverse_tests`, `linalg_layer_tests`
and `route_vocabulary_tests` all Passed.

---

## 10. WHAT IS STILL OPEN

1. **The serial permutation walk — 8.0% at `n = 2048`.** `if (tid < nrhs)` over `n`
   dependent local-memory swaps: at `nrhs = 1` one work-item of up to 1024 does `n`
   round-trips while the rest wait at a barrier. Priced by rebuilding without it
   (`wp6_getrs/proto/noperm.csv`, residual column confirms the break took):
   float n=2048 b=32 **1.2802 → 1.1776 ms (8.0%)**, float n=512 b=512 3.5%, cdouble
   n=2048 2.4%. The largest named residual, and the only fully serial part.
2. **The CTA count is the batch.** One work-group per matrix caps occupancy at the
   batch size; at `n = 2048` the saturating batch is 32 on 128 SMs, which is 41–60%
   of DRAM peak. Splitting a matrix across work-groups is a redesign, not a tune.
3. **The 84 winning cells handed to the vendor** (§4). Recovering them needs a
   per-`(type, order)` predicate measured at more orders, or a kernel fix for the
   mid-ladder dip at `double n=128` / `cfloat n=1024` / `cdouble n=128, 1024`.
4. **The composition's wide-`nrhs` window** (§4) — 1.09× at `nrhs=64`, 1.48× at 128,
   but 9 and 4 losses and no batch ladder. Needs a flatness pass before it can ship.
5. **`nrhs > 8`.** The tier is instantiated to 8. Raising `kGetrsFusedMaxRhs` needs
   `native_tier_preferred` to gain a window at the same time — at `nrhs = 16` the
   composition is already ahead for double (0.55×) and cfloat (0.58×) at `n = 512`.
6. **The latent vendor gate defect** recorded in `route_getrs.hh`: `cublas.cc`'s
   `getrs` calls `cusolverDnXgetrs` for `batch ≤ 1` from a TU gated on
   `BATCHLAS_HAS_CUBLAS`. A cuBLAS-present / cuSOLVER-absent configure claims a
   vendor it cannot link. Untouched here; the fix belongs in `vendor_available.hh`.
