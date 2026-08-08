# SYEV performance: where the remaining time is, and how to get it

**Scope.** `syev` for `float` and `complex<float>` on CUDA / RTX 4090, at batch sizes
large enough to saturate the device. Supersedes `SYEV_PERF_IDEATION.md`, which was
written against `e2ff635` (PR #51) and whose top three items have since been done.

**Status of the numbers below.** Everything in §2–§4 was measured for this document on
2026-08-08 against `ad0fae7` (post PR #67), device 1, one process on the GPU, with
`nsys` for kernel attribution and `ncu` for hardware counters. §5 mixes measured results,
arithmetic from those measurements, and clearly-labelled hypotheses. Anything marked
*unmeasured* is a hypothesis and nothing else.

---

## 1. Where syev stands today

Routing is now decided per scalar type and per job mode, from grids measured at
saturation (`include/blas/functions/syev.hh`):

| n | float, vectors | cfloat, vectors | float, values | cfloat, values |
|---|---|---|---|---|
| ≤ 8 | jacobi_cta | cta_fused | jacobi_cta | cta_fused |
| 9–32 | cta_fused | cta | cta_fused | cta |
| 33–320 | blocked | blocked | blocked | blocked |
| 384–448 | blocked | blocked | two-stage | two-stage |
| 512 | two-stage | blocked | two-stage | two-stage |
| 640–1024 | two-stage | **vendor** | two-stage | two-stage |
| 2048+ | vendor | vendor | two-stage | two-stage |

The two places we still lose to cuSOLVER:

* **cfloat eigenvectors above n = 512** — vendor by 1.07× (640), 1.19× (768), 1.29× (1024).
* **float eigenvectors at n = 2048** — vendor by 1.65× (and that row was measured at
  batch 64, i.e. unsaturated; it is the weakest row in the table).

Everywhere else we are ahead of cuSOLVER, by up to 2.6× (cfloat n=192) and 2.9×
(float n=320).

### The number that frames everything else

At n = 512, batch = 512, float, eigenvectors, our fastest provider takes **≈326 µs/matrix**
(blocked 325.9, two-stage 332.5 — two-stage is the routed one).
Counting the eigensolve at a conservative 4n³ flops, that is **1.65 TFLOP/s** — about
**3.5 % of the ~47 TFLOP/s that cuBLAS SGEMM sustains on this card**. At n = 256 it is
2.1 TFLOP/s, ~4.4 %.

We are not flop-limited. We are structure-limited. That is the single most important
fact for judging the ideas in §5: an algorithm that spends **ten times the flops** in a
shape the GPU likes still has a factor of ~4 in hand.

---

## 2. Where the time actually goes

nsys GPU-kernel time, benchmark setup excluded, batch at saturation, eigenvectors unless
stated. Percentages are of solve GPU time.

### 2.1 n = 256, batch = 1024 — blocked (routed for both types)

| phase | float | | cfloat | | cfloat/float |
|---|---|---|---|---|---|
| latrd panel (symv) | 46.4 ms | 35.5 % | 114.9 ms | 41.7 % | 2.48× |
| vendor GEMM (trailing + ormqr) | 33.2 ms | 25.3 % | 95.2 ms | 34.6 % | 2.87× |
| stedc / steqr (real arithmetic) | 27.1 ms | 20.7 % | 28.5 ms | 10.3 % | 1.05× |
| ormqr larft | 9.3 ms | 7.1 % | 17.9 ms | 6.5 % | 1.93× |
| syr2k trailing update | 6.9 ms | 5.3 % | — | — | **float-only path** |
| trailing update (small) | 3.3 ms | 2.5 % | 7.5 ms | 2.7 % | 2.30× |
| **total** | **130.8 ms** | | **275.5 ms** | | **2.11×** |

### 2.2 n = 512, batch = 512

Blocked (routed for cfloat):

| phase | cfloat | |
|---|---|---|
| latrd panel (symv) | 1016.3 ms | **71.5 %** |
| vendor GEMM | 204.8 ms | 14.4 % |
| ormqr larft | 102.7 ms | 7.2 % |
| stedc / steqr | 60.1 ms | 4.2 % |

Two-stage (routed for float):

| phase | float | | cfloat (forced) | | cfloat/float |
|---|---|---|---|---|---|
| sb2st back-transform (Q2) | 206.6 ms | 30.4 % | 805.0 ms | 40.6 % | **3.90×** |
| sb2st chase (stage 2) | 113.1 ms | 16.6 % | 548.4 ms | 27.7 % | **4.85×** |
| vendor GEMM | 148.2 ms | 21.8 % | 258.2 ms | 13.0 % | 1.74× |
| ormqr larft | 84.0 ms | 12.4 % | 197.7 ms | 10.0 % | 2.35× |
| stedc / steqr | 50.4 ms | 7.4 % | 49.0 ms | 2.5 % | 0.97× |
| geqr2 (sy2sb panel QR) | 39.0 ms | 5.7 % | 52.4 ms | 2.6 % | 1.34× |
| **total** | **679.1 ms** | | **1981.2 ms** | | **2.92×** |

### 2.3 n = 256, batch = 1024, **eigenvalues only**, float — blocked

| phase | ms | % |
|---|---|---|
| latrd panel (symv) | 46.8 | 52.7 % |
| **stedc / steqr** | **25.1** | **28.3 %** |
| syr2k trailing | 6.9 | 7.8 % |
| vendor GEMM | 6.3 | 7.1 % |

### Four things these tables say

1. **The latrd panel symv is the dominant term everywhere the blocked path is routed** —
   35–53 % at n = 256, **71.5 %** for cfloat at n = 512.
2. **stedc is the second-largest float term and is untouched.** 20.7 % of the float
   eigenvector solve at n = 256, and **28.3 % of the eigenvalues-only solve** — where it
   is computing eigenvectors that are then thrown away (`syev_blocked.cc` hardcodes
   `internal_jobz = JobType::EigenVectors` because the merge needs them). It runs in real
   arithmetic for both types, so its absolute cost is identical for float and cfloat.
3. **The complex trailing update never reaches a level-3 triangular kernel.**
   `sytrd_blocked.cc:783` gates it on `(B == CUDA) && is_same_v<T, float>`, so complex
   falls back to *two full n₂×n₂ GEMMs* where float does one triangle-only syr2k. That
   gate was written before batched `herk`/`her2k` existed in the tree (e61eca1, 10b1c8b)
   and was never revisited.
4. **Two-stage stage 2 is where complex falls apart** — 4.85× and 3.90× float, against
   1.74× for the GEMMs in the same solve. Together those two kernels are 68 % of the
   complex two-stage solve, and they are the whole reason two-stage is never routed for
   cfloat eigenvectors.

---

## 3. What the hardware counters say the limits are

`ncu` on the latrd panel kernel, one panel (ib = 32, j₀ = 0). "×once" compares measured
traffic to the ideal of reading the trailing triangle exactly once.

| | float n=256 b=1024 | float n=512 b=256 | cfloat n=512 b=256 |
|---|---|---|---|
| duration | 3.96 ms | 11.37 ms | 22.72 ms |
| DRAM bytes | 605 MB (**0.14× once**) | 4850 MB (1.13× once) | 12291 MB (1.43× once) |
| DRAM throughput | 15.5 % of peak | 43.4 % | 55.0 % |
| L2 bytes | 8368 MB (**1.95× once**) | 10069 MB (**2.34× once**) | 21450 MB (**2.50× once**) |
| L1TEX bytes | 51292 MB (**11.9× once**) | 52499 MB (**12.2× once**) | 135680 MB (**15.8× once**) |
| SM throughput | 52.7 % | 10.8 % | 10.7 % |
| achieved occupancy | 46.7 % | 33.2 % | 33.2 % |

This **refines, and partly corrects, the note currently in `syev.hh`** ("the kernel only
achieves ~330 GB/s of this card's ~1000 GB/s … so it is latency/occupancy bound"). The
sharper statement is:

* **DRAM traffic is already near optimal** (1.1–1.4× the ideal; at n = 256 the matrices
  stay resident in the 72 MB L2 and DRAM sees only 0.14×). We are not wasting DRAM.
* **L2 traffic is ~2.3×** the ideal, which is exactly the symv reading the lower triangle
  **twice** — thread *r* reads `A(r,c)` on its row walk and thread *c* reads the same
  element on its column walk (`latrd_lower_panel.cc:522` and `:530`).
* **L1 traffic is 12–16× the ideal.** This is the real bottleneck. The column walk has
  lane *r* reading `A(c,r)`, i.e. a stride of `lda` across the warp, so one warp request
  touches 32 separate sectors and returns 4 useful bytes from each.
* **The SMs are idle**: 10.8 % SM throughput at n = 512 with 33 % occupancy. This kernel
  is not arithmetic-bound in either precision, and at n = 512 it is not DRAM-bound either.

**Headroom:** at n = 512 the useful data must cross DRAM once (4295 MB). At 100 % of the
1008 GB/s peak that is 4.26 ms against 11.37 ms measured — **2.7× on the dominant kernel**,
if the L1 over-fetch and the double L2 read are removed.

And on two-stage stage 2, at n = 512 batch 512:

| | float | cfloat |
|---|---|---|
| chase: SM / occupancy | 45.0 % / 65.5 % | 29.8 % / 45.1 % |
| back-transform: SM / occupancy | **93.7 % / 82.8 %** | **35.3 % / 49.8 %** |

The float back-transform is genuinely saturated — 93.7 % of peak SM throughput. **The
complex one is not: 35.3 %, with occupancy halved from 82.8 % to 49.8 %.** That is a
register-pressure signature, not an arithmetic one, and it means the complex kernel's
3.90× cost is *not* the price of complex arithmetic. It is recoverable.

---

## 4. A measured result from this investigation

The back-transform's `tile` × `subs` constants were tuned on float only (the table in
`sytrd_sb2st_hh.cc:800` says so). Sweeping them at n = 512, batch = 512, eigenvectors,
two-stage, µs/matrix:

| | subs=4 | subs=8 | subs=16 |
|---|---|---|---|
| **float** tile=8 | 359.95 | **332.65** | 349.52 |
| **float** tile=4 | 362.36 | 369.89 | 398.62 |
| **float** tile=2 | 385.01 | 417.33 | 470.78 |
| shipped default (float) | | **332.46** | |
| **cfloat** tile=8 | 1166.8 | 972.53 | 1009.0 |
| **cfloat** tile=4 | 989.84 | 864.22 | 1075.2 |
| **cfloat** tile=2 | **855.75** | 882.69 | 1218.7 |
| shipped default (cfloat) | | 972.41 | |

The shipped constants are **exactly optimal for float** and cost **complex\<float\> 1.14×**.
The complex optimum sits at a *smaller* tile and fewer sub-groups — precisely what the
occupancy collapse in §3 predicts. This is the same defect class as the routing and `nb`
bugs fixed in PR #65: a constant measured on float and applied to every type.

*Caveat, stated because it matters:* this 1.14× lands on a path cfloat is **not currently
routed to** in eigenvector mode, and 855.75 is still behind blocked (698) and the vendor
(707) at that shape. It does not flip any routing decision on its own. It is worth taking
because it is free, and because it is a prerequisite for §5.3 mattering.

---

## 5. Opportunities, ranked

### Tier A — measured defects, known fixes, small changes

**A1. Give the complex trailing update a level-3 path.** (§2, finding 3)
`sytrd_blocked.cc:783` restricts the syr2k trailing update to `float`, so cfloat issues
two full GEMMs where float issues one triangle. Batched `her2k` (one GEMM + Hermitian
fold) already exists in the tree and is used by `ortho`. Vendor GEMM is 34.6 % of the
cfloat solve at n = 256 and 14.4 % at n = 512; the trailing update is roughly half of it.
*Estimated 1.05–1.12× for cfloat, larger at n = 256 than at 512. Low risk, small diff.*

**A2. Stop computing eigenvectors in `stedc` when the caller asked for values.** (§2.3)
28.3 % of the float eigenvalues-only solve at n = 256 is a full eigenvector D&C whose
output is discarded. The two-stage path already does the right thing (it uses `stebz`);
the blocked path, which owns all of n ≤ 320 in values mode, does not. LAPACK's answer is
`dsterf` (O(n²)); `stebz` is already in the tree. *Estimated up to 1.35× for
eigenvalues-only at n ≤ 320, identically for float and cfloat since stedc is real.*

**A3. Adopt the complex `tile`/`subs` constants from §4.** *Measured 1.14×* on complex
two-stage eigenvectors. Free; the knobs exist.

**A5. The grid latrd path cannot engage at the batch sizes this library targets.**
`choose_grid_launch` computes `cap = MAX_COMPUTE_UNITS / batch` by integer division and
returns `G = 1` (= use the legacy kernel) when `cap < 1`. On a 128-SM card that means
**the grid path is dead for every batch ≥ 128**, and with it the whole
`latrd_grid_min_n = 768` gate and its tuning. Measured, float n = 1024, blocked,
eigenvalues-only, µs/matrix — legacy vs grid:

| batch | cap | legacy | grid | |
|---|---|---|---|---|
| 32 | 4 | 3466.5 | 2018.2 | **1.72×** |
| 64 | 2 | 2706.5 | 2110.5 | **1.28×** |
| 128 | 1 | 2953.3 | 2955.8 | identical — grid path not taken |

The cap exists because the software grid barrier deadlocks unless all participating
work-groups are co-resident, and "total work-groups ≤ SM count" is the conservative way to
guarantee that. It is *too* conservative: these work-groups are ≤ 256 threads with a
modest local-memory footprint, so several are resident per SM, and the cap could be
`SMs × achievable_blocks_per_SM` instead. That would let G ≥ 2 up to batch 256–512.

*Honest expectation: this probably does not pay by itself.* The grid path's known
mechanism is curing batch starvation, and there is no starvation at batch ≥ 128. Its value
is that raising the cap is what makes the L2-residency question in §6 answerable at all.

**A4. Re-measure the n = 2048 row.** It is the one place float loses badly (1.65×) and
the only row in the routing table taken at an unsaturated batch (64). With `blocked` and
`two_stage` carrying larger workspaces than the vendor, this may be a memory-capacity
artifact rather than a real verdict.

### Tier B — kernel redesign, where the measured headroom is

**B1. A single-read, shared-memory-staged panel symv.** *The single biggest item in this
document.* The panel is 35–71 % of every blocked solve, and §3 shows it moving **12–16×**
its useful bytes through L1 and **2.3×** through L2, at 10.8 % SM throughput and 33 %
occupancy. The standard remedy (MAGMA's `symv`) stages a tile of A through shared memory
once and uses each loaded element for *both* the row and the column contribution,
accumulating per-block partial `y` vectors and reducing them in a small second pass. That
removes the double read and the uncoalesced column walk in one change.

Note the kernel's own comment records that a partial attempt at this was tried and
rejected ("one sub-group per column … the extra barrier destroys reuse"). That experiment
changed the *access pattern* without changing the *number of reads*; the point of the
shared-memory version is that one load serves two updates, so the barrier is paid once for
twice the work.

*Ceiling from the counters: 2.7× on the panel at n = 512.* Applied to the measured phase
shares, end-to-end:

| | at the 2.7× ceiling | at a more realistic 2.0× |
|---|---|---|
| cfloat n=512, vectors (panel 71.5 %) | 1.82× | 1.56× |
| cfloat n=256, vectors (panel 41.7 %) | 1.36× | 1.26× |
| float n=256, vectors (panel 35.5 %) | 1.29× | 1.22× |
| float n=256, values (panel 52.7 %) | 1.50× | 1.36× |

Even the conservative column is the largest single win available, and it applies to both
types, both job modes, and every n where blocked is routed. It would also move the cfloat
blocked/vendor crossover well past 512.

**B2. Fix the complex stage-2 occupancy.** (§3) The complex back-transform runs at 35.3 %
SM throughput where float runs at 93.7 %, with occupancy halved. Shrink the per-thread
working set for complex — the repo has already learned this lesson once
(`register-residency-traps`: the thread tile must shrink as the scalar widens). A3 is the
cheap version of this; the real version is a type-aware tiling in `unmqr_hb2st_wave`.
Then re-check whether the C99 Annex G complex multiply (the `isnan` branch + `__mulsc3`
call that cost the latrd panel 1.22–1.29×) is present in the chase — the chase's 4.85×,
above the ~4× arithmetic ratio, and its latency-bound profile make it the better candidate
than the back-transform. `__mulsc3` is still present in `libbatchlas_extensions_sytrd.so`.
*If stage 2 came down to the ~2.2× complex/float ratio the panel achieved after its own
fix, the two stage-2 kernels would fall from 1353 ms to 704 ms, i.e. 1.49× on the complex
two-stage solve — about 650 µs/matrix against the 973.8 baseline. That is ahead of both
blocked (698) and the vendor (707), which would open n ≥ 512 cfloat eigenvectors to
two-stage and attack the one region where we still lose. Note A3 and B2 are not additive:
A3 is the cheap fraction of the same occupancy problem.*

**B3. Extend the CTA-resident solve to n = 64–128.** There is a **complete, tested,
unmerged implementation** on branch `cta-large-n` (04101dc, based on the much older
27851a6): it replaces the sub-group-partition collectives with a work-group-partition
equivalent and lifts the n ≤ 32 cap to whatever shared memory affords — measured as
**n = 128 for float, 64 for cfloat** on this card. It carries 36/36 + 6/6 + 5/5 passing
tests and **no performance measurement at all**. A CTA-resident solve at n = 64–128 skips
~15 kernel launches and all global round-trips, which is exactly the regime (`blocked`,
2.2–11 µs/matrix) where fixed overheads dominate. *Rebasing and benchmarking it is the
highest value-per-hour item in Tier B, because the code already exists.*

**B4. stedc.** Every merge kernel is still `nd_range(batch*128, 128)` — one work-group per
matrix (`stedc.cc:148,184,318,349,395,422,712`). At the large batches that matter here
that is not starvation, so the grid-barrier treatment that fixed `latrd` is *not*
obviously the right fix. But stedc is 20.7 % of the float eigenvector solve at n = 256 and
has never been profiled internally. Profile before designing.

### Tier C — novel and algorithmic, high ceiling, speculative

These are ordered by expected value, not by novelty. All are *unmeasured*. The 3.5 %
of-peak figure in §1 is what makes any of them plausible: they all trade flops for shape.

**C1. Block Jacobi at n = 64–256 — the repo's own strongest evidence points here.**
The SVD analogue landed three commits ago: `gesvdj_cta` (one-sided Jacobi) beats the
tridiagonalizing CTA path by **4.1× at n = 16 and 23× at n = 8 with vectors**, at *better*
accuracy, and was just extended to n = 64. The symmetric analogue has the same structural
advantages: no back-transform (V accumulates as you rotate), no tridiagonal solve, no
stedc, no larft, everything in shared memory, and high relative accuracy on graded input.

The cost is ~8–10 sweeps × ~4n³ ≈ 30–40n³ against our ~4n³ — a 10× flop premium. From §1
we currently run at 1.65–2.1 TFLOP/s, so a block-Jacobi implementation reaching 17–21
TFLOP/s breaks even, and a good GEMM-based block rotation on this card should reach
30–40. *That is the arithmetic case for a 1.5–2.5× win at n = 128–256, and considerably
more below that.* It also composes with B3: the same shared-memory residency budget.

**C2. Iterative refinement of the eigendecomposition (Ogita–Aishima).** Compute a cheap,
low-accuracy (Λ̂, X̂) — TF32 tensor cores, or a truncated Jacobi — then refine: form
R = X̂ᴴAX̂ − Λ̂, solve the diagonal Sylvester correction elementwise
(E_ij = R_ij/(λ̂_j − λ̂_i)), and update X ← X̂(I + E). Each iteration is **two GEMMs and an
elementwise pass**, converges quadratically, and is 100 % tensor-core-friendly.

Why this is attractive *here specifically*: our eigensolve runs at 3.5 % of SGEMM rate, so
converting O(n³) of latency-bound reduction into O(n³) of GEMM is a strictly good trade
even at several times the flops. `joint_matrix` + `precision::tf32` is already verified to
emit real `mma.sync` on sm_89 in this codebase. The known caveat is clustered eigenvalues,
where the elementwise divide breaks down and the correction must be solved blockwise —
which is a real limitation, not a detail, and would need a cluster detection pass.
*Highest ceiling on this list; also the most work and the most numerical risk.*

**C3. Tensor cores for the two-stage GEMMs.** sy2sb's trailing updates plus ormqr are
21.8 % (float) / 13.0 % (cfloat) of the two-stage solve and run on cuBLAS S/C GEMM.
TF32 is ~1.7–1.9× SGEMM on this card. The accuracy question is real — TF32 has a 10-bit
mantissa and this is a similarity transformation, not a residual computation — so the
honest form is either 3×TF32 splitting (≈fp32 accuracy, ~1/3 the tensor rate, i.e. roughly
a wash) or plain TF32 *with* C2's refinement on top. Alone: *~1.1× at best.* As the
first stage of C2: the enabling step.

**C4. 3M (Karatsuba) complex GEMM.** Three real GEMMs instead of four for every complex
level-3 op — a 25 % flop cut on 13–35 % of the cfloat solve, so *~1.03–1.09× end to end*.
Componentwise accuracy is weaker than 4M (normwise is fine; Higham analysed this). Our own
GEMM is in-tree so it is implementable, but the payoff is small and it competes with A1,
which attacks the same term more cheaply.

**C5. Spectral divide-and-conquer (QDWH-eig / ZOLO-EIG).** All BLAS-3, splits into
independent subproblems, but 5–10× the flops of tridiagonal reduction *and* the flops are
QR-heavy rather than GEMM-heavy. At our batch sizes the device is already saturated, so
the parallelism argument that makes QDWH attractive on clusters does not apply. *Listed
for completeness; I would not start here.*

**C6. Real embedding of the Hermitian problem — considered and rejected.** A Hermitian
n×n maps to a real symmetric 2n×2n with doubled eigenvalues. That is 8× the n³ of a real
n-solve against 4× for a complex n-solve, so it is 2× worse in flops. Our float path is
~2.4× more efficient per flop than our complex path, which nearly cancels it — but not
quite, and it doubles the memory. Measured proxy: float blocked at n = 1024 would need to
beat cfloat blocked at n = 512, and it does not (§2.2 scaling). *Do not pursue.*

---

## 6. Negative results from this investigation

Recorded so they are not re-derived.

* **The L2-residency hypothesis for the panel is untested, and currently untestable.**
  I expected the grid latrd path to win at large batch by shrinking the concurrent working
  set into the 72 MB L2. Forcing it changed nothing (float n=256/b=1024 31.06 vs 32.19,
  float n=512/b=512 327.3 vs 325.9, cfloat n=512/b=512 698.8 vs 698.3) — but that A/B was
  **vacuous**, for the reason in A5 below: the grid path silently falls back to the legacy
  kernel at batch ≥ 128. It was legacy against legacy. The hypothesis is neither confirmed
  nor refuted; §3 shows the panel's *primary* problem is L1 request efficiency regardless.
* **`sb2st_hh_benchmark` does not register a complex type.** `--type=cfloat` silently
  produces zero rows. Any complex stage-2 tuning has to go through `syev_benchmark` with
  `BATCHLAS_SYEV_PROVIDER=two_stage`, as §4 does.
* **stedc is already run in real arithmetic for complex input** (`syev_blocked.cc:244`
  instantiates `stedc<B, Real>` and lifts Z with a phase afterwards). The obvious
  "run the tridiagonal solve in real arithmetic" idea is already implemented; §2 confirms
  it, with the stedc row costing the same absolute ms for both types.

---

## 7. Suggested order

1. **B3** — rebase `cta-large-n` and benchmark it. The code exists and is tested; this is
   the only item where the implementation cost is already sunk.
2. **A2**, then **A1**, then **A3** — three small, independently-shippable fixes worth an
   estimated 1.35× (values-only, n ≤ 320), 1.05–1.12× (cfloat blocked) and a measured
   1.14× (cfloat two-stage) respectively.
3. **B1** — the shared-memory single-read symv. Largest measured headroom (2.7× on a
   kernel that is 35–71 % of the solve) and it helps float and cfloat, both job modes,
   at every n where blocked is routed. Prototype against
   `latrd_lower_panel_benchmark` before touching the solver.
4. **B2** — complex stage-2 occupancy and the Annex G escape in the chase. This is the
   item that could open n ≥ 512 cfloat eigenvectors, the one region where we still lose.
5. **A4**, **A5** and **B4** — re-measure n = 2048; raise the grid residency cap so the
   L2 question becomes answerable; profile stedc internally.
6. **C1** — block Jacobi at n = 64–256, on the strength of the `gesvdj_cta` result.
   Start with a cost model and a single-shape prototype before committing.
7. **C2** — iterative refinement. Highest ceiling, most risk; only worth starting once
   1–5 have bounded what the conventional path can still give.

## 8. Reproducing any of this

```bash
# routing / end-to-end, one shape (n, batch, nb=0 → shipped default, fuse, jobz, uplo)
CUDA_VISIBLE_DEVICES=1 build/benchmarks/syev_benchmark --backend=CUDA --type=float,cfloat \
    --warmup=2 --min_iters=5 512 512 0 0 1 0

# force a provider / an implementation
BATCHLAS_SYEV_PROVIDER=blocked|two_stage|vendor|cta
BATCHLAS_LATRD_IMPL=legacy|grid          BATCHLAS_LATRD_GRID_MIN_N=<n>
BATCHLAS_SB2ST_BACK_TILE_W=<1,2,4,8>     BATCHLAS_SB2ST_BACK_SUBS=<4,8,16>

# kernel attribution (sees cuBLAS; the SYCL trace does not)
nsys profile -t cuda -s none -o out build/benchmarks/syev_benchmark ...
nsys stats --report cuda_gpu_kern_sum --format csv out.nsys-rep

# hardware counters on one kernel
ncu -k regex:LatrdLowerPanel -c 2 --metrics dram__bytes_read.sum,lts__t_bytes.sum,\
l1tex__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active --csv \
    build/benchmarks/latrd_lower_panel_benchmark --backend=CUDA --type=float 512 256 32 0 0
```

Measure on an idle device (`nvidia-smi` first — this box has two 4090s and contention has
produced spurious 3.6× "wins" before), warm the clocks, and match `--name` exactly: it is
a substring filter.
