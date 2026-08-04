# SYEV retune: measured results

**Provenance for every number below.** Device: NVIDIA RTX 4090 **#1 only** (128 SMs, driver
595.84, CUDA 13.2, cuSOLVER 12.2.0.1). Device 0 was excluded — another user's job occupied it
for part of the session, and no cross-device comparison was made. Build: `build/presets/cuda`
at commit `7911847`. Each figure is the **median of 5 process-level repeats** with the IQR
recorded; a margin below **1.10×**, or with overlapping IQRs, is reported as *neutral*, not as
a win. Every comparison ran at that cell's measured knee batch, listed in the tables.
Measured 2026-08-03.

---

## 0. Headline

1. **The regression hypothesis is dead.** The committed tuning constants are optimal or
   statistically indistinguishable from optimal at every n measured. There is no lost tuning
   to restore.
2. **The real win is routing, and it is large.** `syev_cta` — the kernel `Auto` sends *every*
   n ≤ 32 to — does not win a single cell in either precision. Routing small n correctly is
   worth **1.1×–3.9×**, and it makes BatchLAS beat cuSOLVER across the whole n = 4..32 range.
3. **A new defect surfaced: workspace footprint.** Two independent SYEV paths allocate
   workspace wildly disproportionate to problem size. This, not compute, is the binding limit
   on usable batch — and it is what caused an out-of-memory event on this shared machine
   during the work.

---

## 1. The regression: refuted, three independent ways

The starting hypothesis was that `include/batchlas/tuning_params.hh` (committed, ORMQR = 16 in
every bucket) shadows the CMake-generated copy (ORMQR 16/32/64/128/128), and that the loss of
the larger ORMQR block sizes is why BatchLAS stopped beating cuSOLVER for n = 32..512.

**The shadowing is real.** Confirmed empirically, not from reading CMake: a probe TU compiled
with the real `-I` flags from `flags.make` prints `ormqr_block_size_for_n(512) = 16` and
`sytrd_block_size_for_n(512) = 24`. `PROJECT_SOURCE_DIR` precedes `PROJECT_BINARY_DIR`, so the
committed header wins and the generated one is dead code. That part stands, and it matters
independently: while it holds, **no retune can take effect at all**.

**But the values are right.** Refutation 1 — history: ORMQR has been 16 in every bucket since
the header was created (`7363746`, 2026-03-02) and has never been modified. Nothing regressed;
there was never a larger value in the compiled path. Refutation 2 — a direct A/B: forcing
ORMQR = 128 at n=512 / eigenvectors / batch 128 gave 452.32 µs/matrix against 301.95 unset,
i.e. **1.50× slower**, IQRs disjoint. Refutation 3 — the full sweep in §3: 16 is the measured
optimum wherever the knob has any effect at all.

**And there is no baseline to restore to.** An exhaustive search of git history — every CSV,
plot, notebook and markdown table across 400+ revisions — found **no artifact anywhere**
recording BatchLAS beating cuSOLVER at n = 32..512, or at any n. The only in-repo statement of
the claim is the goal line of `SYEV_RETUNE_WORKFLOW_PLAN.md`. The claim may well be true of an
uncommitted run; it simply has no recorded evidence, so nothing here calibrates against it.

**What the values actually are, though, is unjustified rather than wrong.** They came from a
standalone `ormqr_blocked` microbenchmark at batch 8192–512, not from `syev`; at n ≤ 64 the
search space only offered `block_size ∈ [4,8,12,16]`, so 16 was the *ceiling of the search*
rather than a measured optimum; and the generator's `_derive_param_buckets` keeps the
parameters of whichever *case* had the lower absolute time rather than the best block size per
case. They happen to be right. §3 is the first measurement that establishes that.

---

## 2. Small-n bake-off (n = 4..32) — the actionable result

Five in-tree contenders plus cuSOLVER. `BM_SYEV_CTA_PIPELINED` and `BM_SYEV_CTA_TRIDIAG_REF`
were discovered during the work: `syev_cta_fused_benchmark` and `syev_jacobi_cta_benchmark`
each register **two** benchmarks, so results must be keyed on the CSV `name` column.

| type | n | mode | winner | wg | µs/matrix | runner-up | ratio | verdict | vs `syev_cta` | vs cuSOLVER | batch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| double | 4 | values | **jacobi** | 4 | 0.00718 | cta_fused | 3.59× | clean | 3.87× | — | 16384 |
| double | 4 | vectors | **jacobi** | 2 | 0.01072 | cta_fused | 3.05× | clean | 3.75× | 15.17× | 4096 |
| double | 8 | values | **jacobi** | 4 | 0.04809 | tridiag_ref | 3.52× | clean | 3.75× | — | 16384 |
| double | 8 | vectors | **jacobi** | 2 | 0.11924 | cta_fused | 2.83× | clean | 2.91× | 3.70× | 1024 |
| double | 16 | values | **jacobi** | 1 | 0.33064 | cta_fused | 2.55× | clean | 2.56× | — | 16384 |
| double | 16 | vectors | **jacobi** | 1 | 0.49690 | cta_fused | 2.49× | clean | 2.50× | 3.29× | 1024 |
| double | 32 | values | **jacobi** | 4 | 2.35063 | cta_fused | 1.58× | clean | 1.70× | — | 4096 |
| double | 32 | vectors | **jacobi** | 2 | 3.31753 | vendor | 1.37× | clean | — | 1.37× | 1024 |
| float | 4 | values | **jacobi** | 2 | 0.00102 | cta_fused | 1.74× | clean | 3.84× | — | 16384 |
| float | 4 | vectors | **jacobi** | 1 | 0.00104 | cta_fused | 1.71× | clean | 4.55× | 18.57× | 16384 |
| float | 8 | values | **jacobi** | 2 | 0.00391 | cta_fused | 1.56× | clean | 2.23× | — | 16384 |
| float | 8 | vectors | **jacobi** | 2 | 0.00478 | cta_fused | 1.37× | clean | 2.55× | 8.42× | 16384 |
| float | 16 | values | **cta_fused** | 2 | 0.02740 | jacobi | 1.10× | *neutral* | 1.17× | — | 16384 |
| float | 16 | vectors | **cta_fused** | 1 | 0.02933 | jacobi | 1.22× | clean | 1.25× | 4.10× | 16384 |
| float | 32 | values | **cta_fused** | 2 | 0.11934 | tridiag_ref | 1.03× | *neutral* | 1.03× | — | 16384 |
| float | 32 | vectors | **cta_fused** | 2 | 0.28426 | tridiag_ref | 1.11× | clean | 1.12× | 1.79× | 1024 |

### 2.1 Proposed routing rule

```
double, n <= 32          -> syev_jacobi_cta      (1.37x - 3.87x over syev_cta)
float,  n <= 8           -> syev_jacobi_cta      (2.23x - 4.55x over syev_cta)
float,  16 <= n <= 32    -> syev_cta_fused       (1.03x - 1.25x over syev_cta)
```

Both winners are **currently unreachable from `Auto`**. `syev_cta` wins nothing, anywhere.

Two cells are *neutral* and are not claimed: float n=16 values (1.10× over jacobi) and float
n=32 values (1.03× over tridiag_ref). The float n≥16 rule rests on the vector-mode cells.

**FP64 caveat, load-bearing.** This card runs FP64 at 1/64 rate, which inflates Jacobi's margin
against tridiagonalisation-based paths. The float column is the better predictor for a 1:2 FP64
datacenter GPU. The `double` rule should be gated on *measured FP64 throughput*, not on the
scalar type. Do not ship the double rule to datacenter hardware without re-measuring.

**Accuracy is a bonus here, not the justification.** `JACOBI_EIGENSOLVER_PLAN.md` §13.1 records
Jacobi resolving graded-SPD eigenvalues to 4.5e-07 relative error where `syev_cta` returns
2.7e+28. That argues for Jacobi on ties; the table above shows it also wins outright on speed.

### 2.2 `syev_cta_fused`: route it

It was implemented, tested, and unreachable, with no head-to-head measurement in tree. This is
the first one. It is the float winner at n = 16..32 and second-best at n = 4..8. **Route it,
do not retire it.**

---

## 3. Per-n block-size sweep, measured in the `syev` context (float)

The measurement that had never been taken. `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE` /
`BATCHLAS_TUNE_SYTRD_BLOCK_SIZE` (added in `7911847`), blocked provider, at the knee batch.

| n | mode | knob | best | committed | verdict | sweep (µs/matrix) |
|---|---|---|---|---|---|---|
| 32 | vectors | ormqr | 16 | 16 | **= committed** | 8:0.747 **16:0.708** 32:0.851 64:0.844 128:0.847 |
| 64 | vectors | ormqr | 16 | 16 | **= committed** | 8:2.420 **16:2.297** 32:2.661 64:4.869 128:5.024 |
| 128 | vectors | ormqr | 16 | 16 | **= committed** | 8:7.861 **16:7.446** 32:8.142 64:12.742 128:22.805 |
| 256 | vectors | ormqr | 32 | 16 | 1.016× *neutral* | 8:50.756 16:44.615 32:43.932 64:51.573 128:72.105 |
| 512 | vectors | ormqr | 32 | 16 | 1.059× *neutral* | 8:395.5 16:340.1 32:321.0 64:337.5 128:420.7 |
| 64 | vectors | sytrd | 8 | 8 | **= committed** | **8:2.298** 16:2.439 24:2.564 32:2.785 48:3.001 64:3.567 |
| 128 | vectors | sytrd | 8 | 8 | **= committed** | **8:7.456** 16:7.629 24:8.431 32:8.773 48:10.534 64:11.123 |
| 256 | vectors | sytrd | 16 | 16 | **= committed** | 8:47.055 **16:44.592** 24:45.887 32:48.041 48:53.468 |
| 512 | vectors | sytrd | 24 | 24 | **= committed** | 8:380.8 16:343.4 **24:339.9** 32:342.6 48:359.3 |
| 512 | values | sytrd | 24 | 24 | **= committed** | 8:299.2 16:262.5 **24:259.7** 32:261.2 48:277.3 |

**ORMQR = 16 is optimal wherever it has any effect**, and the alternative the hypothesis
favoured is far worse: 128 costs **2.1× at n=64** and **3.1× at n=128**. The only candidate
improvements are ORMQR = 32 at n = 256 and 512 in vector mode, worth 1.016× and 1.059× — both
below the 1.10× threshold, so neither is claimed.

**Internal validation.** In *eigenvalues-only* mode the ORMQR sweep is flat to three decimals
(n=128: 8:7.355 16:7.355 32:7.355 64:7.360 128:7.352). That is exactly right — with no
eigenvectors there is no back-transform for ORMQR to affect — and it confirms the override
plumbing is wired to the code path it claims to control.

**SYTRD matches the committed value at every n measured**, and its sweep is *not* flat, so the
knob is live and the committed choice is genuinely the optimum.

### 3.1 Double confirms float independently

| n | mode | knob | best | committed | verdict | sweep (µs/matrix) |
|---|---|---|---|---|---|---|
| 32 | vectors | ormqr | 16 | 16 | **= committed** | 8:4.877 **16:4.842** 32:5.924 64:5.926 128:5.924 |
| 64 | vectors | ormqr | 16 | 16 | **= committed** | 8:13.574 **16:12.821** 32:14.809 64:26.680 128:26.567 |
| 128 | vectors | ormqr | 16 | 16 | **= committed** | 8:50.395 **16:43.639** 32:46.494 64:72.031 128:117.373 |
| 512 | vectors | ormqr | 32 | 16 | 1.053× *neutral* | 8:2117.0 16:1732.3 **32:1645.1** 64:1814.6 128:2312.2 |
| 64 | vectors | sytrd | 8 | 8 | **= committed** | **8:12.833** 16:13.328 24:13.726 32:14.402 48:15.046 |
| 128 | vectors | sytrd | 8 | 8 | **= committed** | **8:43.648** 16:45.102 24:46.910 32:48.398 48:52.887 |
| 512 | values | sytrd | 16 | 24 | 1.066× *neutral* | 8:1355.2 **16:1202.6** 24:1281.3 32:1255.5 48:1331.7 |
| 512 | vectors | sytrd | 16 | 24 | 1.048× *neutral* | 8:1805.4 **16:1652.4** 24:1732.0 32:1705.4 48:1782.3 |

Same conclusion, reached independently: ORMQR = 16 is optimal wherever the knob bites, and 64
or 128 is catastrophic at n = 128 (2.7× worse). The values-mode ORMQR sweep is flat again, as
it must be.

**One candidate worth a second look, still not claimed.** At n = 512 in double, SYTRD = 16
beats the committed 24 by 1.066× (values) and 1.048× (vectors), and ORMQR = 32 beats 16 by
1.053× (vectors, and 1.059× in float). Each is individually below the 1.10× threshold, but the
SYTRD result is *consistent across both modes* and the ORMQR result is *consistent across both
precisions* — which is more than noise usually manages. If anyone retunes the XLARGE bucket
specifically, that is where to look. It is not evidence of a regression: the committed value is
not beaten by the value the original hypothesis favoured, and the margin is small either way.

---

## 4. New defect: workspace footprint is the binding constraint on batch

Measured, `syev_cta`, n = 32, **eigenvectors**:

| batch | peak device memory | µs/matrix |
|---|---|---|
| 512 | 445 MiB | — |
| 1 024 | 2 237 MiB | 0.320 |
| 4 096 | 7 505 MiB | 0.192 |
| 16 384 | **24 083 MiB** (whole card) | 0.592 ← *regresses* |

That is **~1.8 MB of workspace per 32×32 matrix — roughly 450× the 4 KB of data** — scaling
linearly above batch 512. Eigenvalues-only is unaffected (~343 MiB at batch 16384), so it is
specific to the eigenvector workspace. The timing at 16384 going *backwards* is memory
pressure, not compute.

`syev_blocked` has the same problem independently: eigenvalues-only at batch 16384 (n=32),
15258 (n=64), 3814 (n=128) and 953 (n=256) each exceeded 3 GB.

Consequences: usable batch is capped at ~4096 for n=32 with vectors on a 24 GB card, well
below where these kernels would otherwise saturate; and this caused a genuine OOM on a
**shared** machine during this work. Worth a dedicated investigation — the fix would raise the
usable batch ceiling for every small-n eigenvector solve.

---

## 5. Coverage, gaps, and what is NOT established

- **973 successful measurements.** Phase A (small-n) complete for both precisions. Phase B
  complete for both precisions except n = 256, below.
- **n = 256 is missing from Phase B entirely** (both modes, both precisions). `syev_blocked`
  exceeded the 3 GB per-invocation abort even at batch 476 — the §4 workspace defect again.
  This is a real gap: the MEDIUM bucket (n ≤ 256) is the one bucket whose block sizes were not
  re-measured in double. No conclusion in this document rests on it, because §3 and §3.1 agree
  at every other n in both precisions.
- **No cuSOLVER route exists for eigenvalues-only.** `benchmarks/syev_benchmark.cc` hardcodes
  `JobType::EigenVectors` (lines 53, 59) and takes `(n, batch, nb, fuse)` — no jobz argument.
  So `values`-mode cells rank our five kernels against each other but **cannot** be ranked
  against the vendor. Closing this needs a small benchmark change.
- **Phase B batch is capped at 512** (blocked provider) because of §4. Some Phase B cells are
  therefore below their natural knee; the block-size *ranking* is what is being read, and it is
  consistent across n, but absolute µs/matrix at small n is not a saturated figure.
- **Device 0 was never used** and no cross-device comparison was made.
- **Not established:** the n = 32..512 routing switch points against cuSOLVER (Phase B measured
  block sizes, not provider crossovers); the eigenvector routing grid re-measurement; the
  two-stage `kd` retune; anything in double for Phase B.

---

## 6. Recommended next actions

1. **Route the small-n winners** (§2.1). Largest measured win available, no new kernels needed.
2. **Fix the include-order shadowing** so a retune can ever take effect. Recommendation from
   the forensics pass: delete the CMake generator path and keep the committed header as the
   single source of truth, retargeting `generate_tuning_header.py` at it — the tuning pipeline
   is demonstrably not in use (no `build/tuning/profile.json` exists anywhere, and
   `BATCHLAS_TUNING_PROFILE` is empty in the cache).
3. **Investigate the eigenvector workspace** (§4). It is the binding constraint on batch.
4. **Leave the block sizes alone.** §3 says they are already right.
5. Then the outstanding ideation items: re-measure the eigenvector routing grid (stale since
   grid-`latrd` landed), the two-stage `kd` sweep with the `nb` hint active, and the
   `stedc` grid-barrier port.

---

# Part 2 — eigenvector routing grid and two-stage `kd` (2026-08-04)

Same provenance rules as Part 1. RTX 4090 **device 1 only**, build `12963a8`, float,
eigenvectors, median of 3, **one measuring process at a time** (see the contention note).

## 7. The eigenvector routing grid: re-measured, and the routing survives

The grid in `syev.hh` was measured at `27851a6`, before the grid-barrier `latrd` path existed.
Re-measured, blocked/vendor, `> 1` = vendor wins:

| n \ batch | 1 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 2.08 | 4.43 | 4.48 | 4.32 | 3.93 | 3.22 | 2.29 | 1.55 | 1.11 |
| 128 | 2.00 | 4.16 | 4.09 | 3.79 | 3.39 | 2.71 | 1.72 | 1.22 | 0.98 |
| 256 | 3.96 | 2.67 | 2.59 | 2.44 | 2.19 | 1.87 | 1.26 | 1.21 | 1.08 |
| **320** | 5.00 | 1.23 | 1.35 | 1.24 | 1.03 | **0.79** | **0.68** | **0.47** | **0.36** |
| **512** | 6.42 | 2.07 | 1.93 | 1.62 | 1.27 | **0.86** | **0.73** | **0.74** | — |
| **640** | 8.78 | 2.27 | 2.03 | 1.67 | 1.18 | **0.85** | **0.86** | **0.81** | — |
| 768 | 9.99 | 2.56 | 2.21 | 1.73 | 1.10 | 1.05 | 1.07 | — | — |
| 896 | 10.05 | 2.27 | 1.93 | 1.52 | 1.06 | 1.19 | 1.24 | — | — |
| 1024 | 10.14 | 2.29 | 1.97 | 1.50 | 1.15 | 1.44 | — | — | — |

> **SUPERSEDED — see §11.** This section concluded the routing needed no edit. That was wrong,
> and wrong for the reason the whole document warns about elsewhere: the batch ladder here
> stopped at 1024, which is not saturation for small n. Measured at saturation (§11), the
> routing is wrong at five of nine sizes. The table below is kept as a record of the
> batch-dependent behaviour, not as a routing basis.

**The carve-out looked confirmed at the batches measured here.** What moved:

- **n = 1024 improved substantially** — 15.33 → 10.14 at batch 1, 3.33 → 2.29 at batch 8 —
  which is grid-`latrd` doing its job. The vendor still wins, so the decision does not flip.
- **The carve-out is wider than recorded.** The original grid stopped at batch 256; the
  blocked win keeps *growing* with batch, reaching **0.36 at n=320 / batch 1024** — blocked is
  **2.8× faster than cuSOLVER** there. The existing `batch >= 128` predicate already covers it.
- n = 768 and 896 are vendor-win at every batch, confirming the upper edge.
- n = 128 at batch 1024 is 0.98 — nominally ours, inside the noise band, not carved out.

### 7.1 A contention artifact that nearly moved the carve-out

A first pass had n=768 at **0.28** and **0.38** for batch 32 and 64 — an apparent 3.6× blocked
win that would have extended the carve-out's upper edge. It was false. Two measuring processes
had overlapped on the device, and the inflated arm was the **vendor** one: 6885 µs/matrix
against 1110 when re-run alone. Clean, the row is 1.73 / 1.10 / 1.05 — vendor throughout.
Recorded because the artifact was large, plausible, and pointed the *wrong way*.

## 8. `latrd_grid_min_n = 768` confirmed for eigenvector mode

The gate was derived eigenvalues-only and applied in both modes. legacy/grid, `> 1` = grid wins:

| n | batch 1 | batch 8 | batch 64 |
|---|---|---|---|
| 256 | 0.734 | 0.734 | 0.788 |
| 384 | 0.885 | 0.894 | 0.881 |
| 512 | 1.012 | 1.018 | 0.956 |
| **768** | **1.379** | **1.375** | 1.090 |
| **1024** | **1.843** | **1.781** | — |

Same crossover. **The constant does not need splitting per mode.** The win shrinks as batch
grows, which is the mechanism behaving: once the batch alone saturates the SMs there is no
starvation left to absorb.

## 9. Two-stage `kd`: the prediction is disproved, and a second stale claim found

Total ms, eigenvectors, median of 3:

| n / batch | kd=16 | **kd=32** | kd=48 | kd=64 | kd=96 | kd=128 | blocked |
|---|---|---|---|---|---|---|---|
| 128 / 2048 | 27.6 | 22.0 | 22.0 | 21.0 | 18.5 | **16.9** | **14.6** |
| 256 / 1024 | 69.5 | **61.3** | 64.2 | 65.7 | 93.5 | 75.7 | **40.9** |
| 512 / 512 | 217.7 | **194.2** | 216.8 | 236.1 | 294.9 | — | **191.4** |
| 1024 / 128 | 445.3 | **369.9** | 394.9 | 433.2 | 541.3 | 715.3 | 475.5 |
| 2048 / 32 | 1171 | **1066** | 1160 | 1286 | 1609 | 1943 | **876** |

**`kd = 32` remains optimal at n >= 256.** The argument that removing the split-WY penalty
(`f7f3c57`) should push the optimum up to 96–128 holds only at n = 128 — where two-stage loses
to blocked anyway, so a better kd only narrows a loss.

The nb-hint A/B at n = 1024, the only shape where `f7f3c57`'s gate fires, explains why:

| kd | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|
| hint gives | 1.019× | 1.057× | 1.064× | 1.034× | 0.981× | **0.926×** |

The hint **helps narrow bands and hurts wide ones**, exactly as `sytrd_sy2sb.cc:44` predicts
(LARFT work is O(m·k·nb) and doubles with nb). Removing the split-WY penalty did not free wide
kd — it made wide kd relatively worse.

### 9.1 "Two-stage wins at n >= 1024" is now shape-dependent

Grid-`latrd` sped the *blocked* baseline up underneath that comparison:

| n=2048 / batch=32 | blocked | two-stage (kd=32) | winner |
|---|---|---|---|
| latrd **legacy** (as originally measured) | 1235.1 | 1066 | two-stage 1.16× |
| latrd **grid** (today's default) | **875.8** | 1066 | **blocked 1.22×** |

The legacy figure reproduces the committed table's 1265.7, so the comparison is like-for-like.
At n=1024/batch=128 the `latrd` impl makes no difference (475.5 vs 475.7 — batch 128 already
saturates 128 SMs) and two-stage wins by **1.29×**, better than the recorded 1.13×.

**So: two-stage wins where the batch saturates the device and loses where it does not**, because
that is exactly where grid-`latrd` rescues blocked. This is the second claim in this codebase
invalidated by a later optimisation that nothing re-checked — the same failure mode as §7.

## 10. Net effect on the routing

**No routing change is warranted from Part 2.** The eigenvector carve-out is confirmed, the
`latrd` gate is confirmed, `kd = 32` is confirmed, and two-stage should still not be routed for
eigenvectors. What changed is the *documentation*: three tables now carry re-measured numbers,
a date, a build SHA, and the shape-dependence that the plain claims elided.

---

# Part 3 — saturated routing (2026-08-04), which supersedes §7

## 11. Routing decided at saturation, keyed on n alone

§7 concluded the eigenvector routing needed no change. **That was wrong.** Its batch ladder
stopped at 1024, and for small n that is not saturation — at n = 64 the crossover needs batch
~16384. Small batch flatters the vendor, whose fixed launch cost is lowest, so a ladder that
never reaches saturation systematically over-credits it.

Re-measured at, for each n, the largest batch where **all three providers fit** (blocked and
two-stage carry far larger workspaces than the vendor path). Float, eigenvectors, µs/matrix,
median of 3, RTX 4090 device 1, one process on the device, build `12963a8`:

| n | batch | blocked | vendor | two_stage | winner | margin | old Auto | new Auto |
|---|---|---|---|---|---|---|---|---|
| 64 | 16384 | **1.64** | 2.02 | 2.21 | blocked | 1.23× | vendor ✗ | blocked |
| 128 | 4069 | **6.65** | 7.85 | 10.35 | blocked | 1.18× | vendor ✗ | blocked |
| 256 | 2034 | **36.92** | 37.14 | 57.56 | blocked | 1.01× | vendor | blocked |
| 320 | 1302 | **74.53** | 215.23 | 111.20 | blocked | 1.49× | blocked ✓ | blocked |
| 512 | 508 | 384.27 | 504.69 | **380.02** | two_stage | 1.01× | blocked | two_stage |
| 640 | 651 | 836.30 | 1008.06 | **727.48** | two_stage | 1.15× | blocked ✗ | two_stage |
| 768 | 452 | 1612.98 | 1414.99 | **1184.11** | two_stage | 1.19× | vendor ✗ | two_stage |
| 1024 | 254 | 4089.02 | 2706.84 | **2441.93** | two_stage | 1.11× | vendor ✗ | two_stage |
| 2048 | 64 | 33842.6 | **15019.1** | 24782.3 | vendor | 1.65× | vendor ✓ | vendor |

**The old routing was wrong at five of nine sizes**, by 1.11×–1.23×.

### 11.1 The rule

```
n <= 32          -> CTA family (syev_choose_small_kernel; unchanged)
64 <= n <= 320   -> BatchLAS_Blocked
512 <= n <= 1024 -> BatchLAS_TwoStage
n >= 2048        -> Vendor
```

`syev_prefer_vendor` is no longer consulted for eigenvectors. It was removed from that path
rather than re-tuned, because its whole structure is batch-keyed and every grid feeding it was
built from unsaturated cells.

### 11.2 Verified end to end

Auto dispatches to the measured winner at every shape (idle GPU, ratio Auto/expected):

| n | 64 | 128 | 320 | 640 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| ratio | 1.000 | 0.999 | 1.001 | 1.000 | 1.000 | 1.002 |

Tests reproduce the baseline exactly: 16 pass, 3 fail (`lanczos`, `steqr`, `stedc`, all
pre-existing).

### 11.3 What is NOT established

- **n = 2048 was measured at batch 64**, below the 128 SMs — memory-limited, not saturated,
  because blocked and two-stage cannot fit a larger batch at that size. It is the weakest row
  in the table and the 2048 boundary rests on it.
- **Float only.** The mechanism is type-independent but the crossovers may not be.
- **Eigenvectors only.** `syev_benchmark` hardcodes `JobType::EigenVectors`, so eigenvalues-only
  keeps `syev_prefer_two_stage_values` and the historical ordering, unmeasured.
- **`Uplo::Upper`** falls back to the vendor — neither BatchLAS path supports it.

### 11.4 The recurring lesson

This is the third table in this repo invalidated the same way, and the second one *I* produced.
A measurement is only as good as the regime it samples: §7 was methodologically careful about
medians, IQRs and name-matching, and still reached the wrong conclusion because the batch
ladder stopped short. **Routing must be decided at saturation** — the regime where the kernel,
not the launch overhead, is what is being compared.
