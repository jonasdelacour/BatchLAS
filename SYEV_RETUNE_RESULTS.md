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
