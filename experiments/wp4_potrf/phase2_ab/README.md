# WP4 Phase 2 — settling corrections Open Questions 5 and 6 by measurement

Everything here was produced on **GPU 1 of this box (2× RTX 4090, 128 SMs, sm_89)**, one
card claimed exclusively for every run through `gpu_guard.sh` (a copy of
`/home/jonaslacour/BatchLAS/experiments/gpu_guard.sh`; the path the task named,
`scripts/gpu_guard.sh`, does not exist in this worktree). Every run printed
`gpu_guard: GPU 1 exclusive for the whole run`; no run was kept that did not.

Library under test: `build/` at `eb02c10` (vendor present — cuBLAS/cuSOLVER linked). Nothing
under `src/`, `include/` or `tests/` was modified.

---

## The headline, before any timing

**Open Question 5 is settled, but not in the way it was asked.** The routed
`trsm(Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit)` is not merely "within
15 % of the vendor" on potrf's panel shapes — it is **1.9× to 42× FASTER**, and for complex
there is no vendor `trsm` on this backend at all. A bespoke `PotrfPanelSolveKernel` would be
wasted work.

**But the routed trsm RETURNS A WRONG ANSWER on exactly those shapes**, at the batch sizes
this campaign is required to measure at, on the DEFAULT route with no environment variable
set. That is a shipped defect, it is not caused by anything potrf does, and it blocks the
blocked driver until it is fixed or worked around. See §4.

**Open Question 6 is settled and one of its premises is refuted.** The trailing update lands
on `Tiled16` for every scalar type (the `ConjTrans` short-circuit at
`gemm_kernels.cc:470` is intact after `f236575`/`7d84208`). But the strided-`ld`
collapse WP3 measured at 0.43–0.62× **does not reproduce** for these shapes: sub-view and
`ld == rows` measure the same to within 1 %. The reason is that the collapse is a property
of the 128×128 register kernel, and the `ConjTrans` short-circuit means potrf never reaches
it. See §3.

---

## 1. What was built, and why not the existing benchmarks

`phase2.cpp` (built by `build.sh` against the already-built `build/src/*.so`, flags copied
from `build/benchmarks/CMakeFiles/gemm_benchmark.dir/{flags.make,link.txt}` via
`experiments/wp4_complex/gpu1/build_bench.sh`).

`benchmarks/trsm_benchmark` and `benchmarks/gemm_benchmark` allocate their own operands at
`ld == rows`. Every operand a blocked Cholesky driver hands `trsm`/`gemm` is a **sub-view of
the parent** carrying the parent `ld` AND the parent batch stride. A benchmark that
allocates its own operands is structurally incapable of seeing that, which is the whole
subject of Open Question 6. And the split of a whole blocked potrf across leaf / panel /
trailing cannot be obtained from any per-op benchmark at all.

Modes:

| mode | what it measures |
|---|---|
| `panel <type> <n> <nb> <batch> <reps>` | the panel trsm at every `j` the driver would issue, operands as real sub-views of one `n × n` parent |
| `panelflat <type> <m2> <ib> <batch> <reps>` | same shape, freshly allocated at `ld == rows` — the control |
| `trail <type> <m> <n> <k> <batch> <sub\|flat> <T\|C> <reps>` | one trailing-update gemm, `alpha=-1, beta=1, transA=N` |
| `blocked <type> <n> <nb> <W> <batch> <reps>` | the whole right-looking driver, per-stage split, residual and `info` |
| `vendorpotrf <type> <n> <batch> <reps>` | the routed `potrf` (cuSOLVER here) on the same parents |
| `trsmdiff <type> <n> <ib> <j> <batch> <sub\|flat> <slack> <rep>` | native trsm vs vendor trsm on one input, plus a host residual for each |

Every sub-view is built by the explicit 6-arg `MatrixView` constructor with the parent `ld`
**and** stride **and** batch ([FIX-B-trap]; `matrix.hh:1140` propagates the parent pointer
array against its own comment, and `matrix.cc:1839` resolves a 0 stride to `ld*cols` of the
CHILD).

### Measurement hygiene actually applied

* **Warm-up.** The first job of each process runs the call in a loop for `BENCH_WARM_S`
  (default 1.5 s) before the first timed rep; later jobs in the same process re-warm for
  0.25 s (the card has been busy throughout). An idle 4090 sits at 210 MHz and a first-run
  SYCL JIT once fabricated a 3.7× loss in this tree.
* **Interleaving.** `run_panel.sh` and `run_trail.sh` run `default`, `vendor`, `native` back
  to back for each cell rather than all of one route then all of the next.
* **Discard rule.** Any cell with relative sd > 10 % is discarded. **No cell in `panel.csv`
  or `trail.csv` came close**: the worst relative sd anywhere in either file is 1.4 %
  (`panel.csv`, cdouble n=512 j=0, 0.013). Nothing was discarded, and the full files are
  here including every cell that disagrees with a conclusion below.
* **Timing never runs under `BATCHLAS_KERNEL_TRACE`** (~60 % inflation). The trace runs are
  a separate script, `run_routes.sh`, and its timings are not quoted anywhere.
* **Route pins verified.** `run_routes.sh` shows the pins taking effect: `gemm_sycl_tiled16`
  appears in the trace only under `BATCHLAS_GEMM_ROUTE=native`, and vanishes under
  `=vendor` and under the default. (The SYCL trace cannot see cuBLAS, so a vendor call is
  an *absence*.) The route table is separately interrogated directly by `routeq`, which does
  not depend on the trace at all.
* **Batch 128 throughout.** 128 SMs; and trsm's float/`Side::Right` clause is
  `s.batch >= 128 || order <= 32` (`route_trsm.hh:304`), so a smaller batch would measure a
  *different route* and call it the same thing.

---

## 2. Open Question 5 — the panel solve

`panel.csv`, produced by `run_panel.sh`. Rows are `route,mode,type,n,nb,j,m2,ib,batch,
med_ms,min_ms,rel_sd,gflops`, `nb = potrf_cta_max_n<T>()` = {float 155, double 109,
cfloat 109, cdouble 77}, batch 128, 7 reps of 3 calls each, `j` subsampled to 4 values evenly
across the loop. Summary via `analyse.py panel`.

**vendor / native, worst and best cell per type** (over n ∈ {512, 1024, 2048}):

| type | worst cell | best cell | cells where the vendor wins |
|---|---|---|---|
| float | 0.92× (n=2048, m2=33) | 2.28× (n=2048, m2=1273) | 2 of 12, both the final short panel (m2 = 33 and 47), each ≈0.11 ms |
| double | 2.12× | 3.90× | 0 of 12 |
| cfloat | 4.61× | 27.79× | 0 of 12 |
| cdouble | 4.02× | 42.16× | 0 of 12 |

`default` equals `native` to within 1 % in 41 of the 47 cells (worst 2.5 %, float n=1024
j=310; all six outliers are within the cells' own spread) — i.e. **the resolver already picks native for these shapes**, confirmed
independently by `routeq.txt`.

### The panel-solve control: is the trsm sensitive to the parent `ld`?

`panelflat.csv` (`run_panelflat.sh`) repeats the `j = 0` cells with the operands freshly
allocated at `ld == rows` instead of sliced out of the `n × n` parent. `sub / flat`, so > 1
means the sub-view is slower:

| type | m2 × ib | native | vendor |
|---|---|---|---|
| float | 869×155 | 1.01 | 1.00 |
| float | 1893×155 | 0.98 | 0.93 |
| double | 915×109 | 1.00 | 1.00 |
| double | 1939×109 | 1.01 | 1.00 |
| cfloat | 915×109 | 1.00 | **1.11** |
| cdouble | 947×77 | 1.03 | **1.44** |

**The native trsm is `ld`-insensitive on these shapes** (1.00–1.03), so §2's ratios are not
an artefact of the storage layout in either direction. The *vendor* complex fallback is not:
it re-reads the whole triangle from global memory per work-item, so the parent `ld` costs it
1.11–1.44×. That makes the complex ratios in the table above, if anything, *understated* for
the shapes potrf actually issues.

**The complex "vendor" is not cuBLAS.** `cublas.cc:1111-1218` intercepts
`std::complex<float>` and `std::complex<double>` before any cuBLAS call and runs a
hand-written SYCL kernel — one work-item per (batch, row), serial substitution, re-reading
the whole triangle from global memory each step. `route_trsm.hh:288-291` already records
this. So the complex ratios above are native-vs-fallback, not native-vs-vendor, and the
"within 15 % of vendor" test is vacuous for complex on the CUDA backend.

### VERDICT: **USE_ROUTED_TRSM.** Do not write `PotrfPanelSolveKernel`.

The routed trsm is faster than every alternative that exists, by 1.9–3.9× for the real types
and by 4.6–42× for complex, on the real shapes at the real parent `ld`. A bespoke kernel
would have to beat *that*, not the vendor.

### How much of a whole potrf is the panel solve?

From `blocked.csv` (per-stage split, W=128, batch 128). The share depends strongly on `nb`
and on which build you are in, so here is the range rather than one number, taken over the
best-performing `nb` for each (type, n) in each mode:

| stage | vendor-free (`native`) | vendor-present (`default`) |
|---|---|---|
| leaf (CTA potrf) | 0.3 – 14 % | 1 – 30 % |
| **panel solve (trsm)** | **5 – 22 %** | **8 – 33 %** |
| trailing update (gemm + fold) | 65 – 95 % | 38 – 91 % |

Worked examples at the best correct `nb`: float n=512 nb=128 `native` — leaf 13.8 %, panel
21.6 %, trailing 64.6 %. double n=512 nb=96 — 11.1 / 19.9 / 69.0. cdouble n=1024 nb=64
`native` — 0.4 / 5.4 / 94.2.

So a hypothetical 2× panel kernel is worth **3–11 % end to end**, and the same 2× in the
trailing update is worth **33–48 %**. Combined with §2's ratio — the routed trsm is already
1.9–42× faster than the only alternative that exists — this is the second, independent reason
not to write `PotrfPanelSolveKernel`. **Every unit of Phase 2's effort belongs in the trailing
update**, where a vendor-free float build is at 0.18× of cuBLAS.

(The per-stage split is measured with a `q->wait()` between stages, which inflates the total
by 1–5 % against the un-instrumented `med_ms` in the same row — `staged_ms` and `med_ms` are
both in the CSV so the inflation can be read off. Proportions are quoted from the split;
ratios are quoted from `med_ms`.)

---

## 3. Open Question 6 — the trailing update

### 3a. What it routes to

`routeq.txt` (`routeq.cpp`, host-only, asks `resolve_gemm_route<T>` directly).

**Why not `scripts/route_diff.sh`, which the question names.** That script runs the whole
ctest suite and diffs the routes the *tests* reach. No test in the tree issues potrf's
trailing-update shape — Phase 2 has not been written — so the capture would contain zero
rows for the question being asked, and a zero-row capture is exactly the failure mode that
script's own header warns about. It is also blind to `KernelVariant` by construction (it
records resolver `Route`s only), and `KernelVariant` is precisely where the `ConjTrans`
short-circuit lives. `routeq` answers the `Route` half exactly and `run_routes.sh`'s trace
answers the `KernelVariant` half.

| type | vendor present | vendor free |
|---|---|---|
| float | `Vendor:Auto` | `Native:RegisterTiled` |
| **double** | **`Native:RegisterTiled`** | `Native:RegisterTiled` |
| cfloat | `Vendor:Auto` | `Native:RegisterTiled` |
| cdouble | `Vendor:Auto` | `Native:RegisterTiled` |

Identical for `transB = Trans` and for `transB = ConjTrans`, and for every (m, n, k) the
driver issues. **Double's trailing update leaves cuBLAS today**, with the vendor present and
no environment variable set: `RouteTable<Op::gemm,double>::preferred` is `batch >= 64 &&
k >= 2` with no transpose test (`route_gemm.hh:122, :206`).

**And `Native:RegisterTiled` means `Tiled16` here, for every type.**
`select_kernel_variant` (`gemm_kernels.cc:470-482`) short-circuits any transposed operand to
`max_dim <= 32 ? Direct : Tiled16` (`:482`) **before** the float register ladder and before the
wide-scalar ladder. Confirmed still true after this session's ConjTrans commits: `f236575`
added guards inside nine transposed *launchers* and `7d84208` relaxed the *wide* kernel's
routing gate, and neither touches `:470`. The only exceptions are the three float cases at
`:472-480`, which require `transB == Transpose::Trans` **exactly** — `ConjTrans` is a
distinct enum value (`NoTrans=0, Trans=1, ConjTrans=2`), so potrf's `ConjTrans` misses them.
The trace in `routes.txt` shows `gemm_sycl_tiled16` and nothing else under
`BATCHLAS_GEMM_ROUTE=native`, for all four types. So **the 64×64 wide tile is structurally
unreachable for potrf's trailing update, and the complex trailing update lands on Tiled16** —
which is what the question asked to have verified.

### 3b. What it costs, at `ld == rows` and at the parent `ld`

`trail.csv`, produced by `run_trail.sh`; summary via `analyse.py trail`. `sub` = operands are
sub-views of one `(max(m,n)+k)`-square parent; `flat` = freshly allocated at `ld == rows`.
Batch 128, 7 reps of 3 calls, worst relative sd in the file 1.2 %.

**The strided-`ld` collapse does not reproduce.** `sub/flat` for the native kernel:

| type | `sub/flat` (native), ConjTrans | `sub/flat` (vendor) |
|---|---|---|
| float | 0.89 – 1.07 | 0.98 – 1.06 |
| double | 0.99 – 1.01 | 1.00 |
| cfloat | 0.93 – 1.09 | 0.99 – 1.01 |
| cdouble | 1.00 | 1.00 |

WP3's 0.43–0.62× is **not** a general property of native GEMM on sub-views. It is a property
of `Tiled128x128RegisterK8`, whose loss is exposed B-load latency
([[native-gemm-strided-ld-collapse]]) — and the `ConjTrans` short-circuit means potrf's
trailing update never reaches that kernel. The one cell here that *does* reach a register
kernel shows the effect: float `128×128×155` with `transB = Trans` routes to
`Tiled128x32RegisterK32NT` and measures **`sub/flat` = 1.43**, and `869×128×155` `Trans`
measures 1.09. So the effect is real, it is register-kernel-specific, and potrf is immune to
it only because it is stuck on the slowest kernel in the ladder.

**What the trailing update actually costs.** `native / vendor` (>1 = native faster):

| type | rectangle `m2×128×ib` | diagonal `128×128×ib` | whole-A22 `m2×m2×ib` |
|---|---|---|---|
| float (C) | 0.18× (n=1024), 0.17× (n=2048) | 0.15× | 0.13× |
| float (T) | 0.31× | 0.28× | — |
| double (C or T) | **1.18×** | **1.15×** | **1.18×** |
| cfloat (C) | 0.23× | 0.21× | 0.22× |
| cdouble (C) | 0.33× | 0.34× | 0.33× |

So the cost of vendor freedom in potrf's trailing update is **5.5× for float, 4.3× for
cfloat, 3.0× for cdouble, and nothing at all for double** (double gains 1.18×).

**A free 1.8× for float, from an enum.** For a REAL type `ConjTrans` and `Trans` are the same
operation. Passing `Transpose::Trans` moves float's trailing update off `Tiled16` and onto
`Tiled128x32RegisterK32NT`:

| shape | ConjTrans (native) | Trans (native) | gain |
|---|---|---|---|
| float 869×128×155 sub | 1.1858 ms | 0.6710 ms | **1.77×** |
| float 128×128×155 sub | 0.1452 ms | 0.0780 ms | **1.86×** |
| double 915×128×109 sub | 2.4610 ms | 2.4555 ms | 1.00× (no double register ladder) |

It does not close the gap to cuBLAS (0.31× rather than 0.18×) and it does nothing for
double, but it is one enum value in the driver and it is worth 1.77–1.86× of the dominant
stage in a vendor-free float build. It must NOT be done for complex, where the two are
different operations — `PHASE2_BREAK=conj` in this harness is exactly that substitution, and
the residual goes from 4.0e-07 to 1.9e-02 for cfloat and cdouble while staying bit-identical
for float and double (`breakcheck.csv`).

**The W-decomposition is worth taking.** The whole-A22 gemm `869×869×155` costs 7.35 ms
(native, flat) against 1.12 ms for one `869×128×155` rectangle; the decomposition issues
`ceil(m2/W)` rectangles of shrinking height plus `ceil(m2/W)` `W×W` diagonal blocks, i.e.
about half the arithmetic of the square form. Both the arithmetic and the measurement agree
with §2.6 of the spec, on the corrected 25 % waste figure.

---

## 4. THE BLOCKING FINDING: the routed trsm returns a wrong answer on potrf's panel shapes

This was found by the residual check in `blocked`, not looked for.

### 4a. What is wrong

`sycl_trsm::trsm_native_blocked` (V2, the blocked driver, reached for every triangular order
above `trsm_cta_max_n<T>() == 32`) computes a **wrong answer** for
`Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit`. `trsmdiff.csv` and
`trsmbug.csv` compare it against `backend::trsm_vendor` on the same input in one process,
and against a host forward-substitution residual so that neither side is assumed correct:

* the **vendor** residual is 1.2e-07 … 4.8e-07 (float/cfloat) and 4.4e-16 … 8.9e-16
  (double/cdouble) in **every** cell of both files;
* the **native** residual reaches 1e+04 … 1e+20 in the failing cells.

All four scalar types. It is **not** a sub-view or strided-`ld` artefact: it reproduces
identically with `flat` operands at `ld == rows` (`trsmbug.csv`). It is **not** the
over-long `MatrixView` span from `matrix.cc:1839`: padding the parent by `n*n` elements
(`slack`) changes nothing.

### 4b. The two conditions, measured

`trsmthresh.csv`, double, `flat` operands, `q = m2` the non-triangular extent of B:

**(i) `q * batch` must be large.** At order 48, `q = 976`, the batch sweep is exact:

| batch | 16 | 32 | 48 | 64 | 66 | **67** | 68 | 70 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|
| `q*batch` | 15616 | 31232 | 46848 | 62464 | 64416 | **65392** | 66368 | 68320 | 93696 | 124928 |
| items wrong | 0 | 0 | 0 | 0 | 0 | **1–2** | 1–2 | 1–2 | 1–2 | 123–128 |

Clean at `q*batch = 64416`, broken at `65392`. The boundary sits between them; **65536 =
2^16** is inside that interval once the global size is rounded up to a work-group multiple.
I did not confirm the mechanism — this is where the evidence stops.

**(ii) the triangular order matters, and the pattern is the V1 bucket of the FINAL block.**
At batch 128, `q ≈ 870–990` (so condition (i) holds everywhere):

| order | 32 | 33 | 48 | 64 | 65 | 77 | 96 | 109 | 128 | 155 | 160 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| final block `order mod 32` | 0 | 1 | 16 | 0 | 1 | 13 | 0 | 13 | 0 | 27 | 0 |
| V1 bucket for it | – | 8 | **16** | – | 8 | **16** | – | **16** | – | 32 | – |
| items wrong (2 reps) | 0,0 | 0,0 | **125,126** | 0,0 | 0,0 | **97,98** | 0,0 | **90,97** | 1,0 | 0,0 | 5,5 |

Every order whose final V1 block lands in the **N=16 bucket** fails almost the whole batch.

**Independently confirmed by a different experiment.** The `nb` sweep in `blocked.csv` never
mentions buckets, and its cfloat n=1024 `native` rows fall out the same way: `nb` = 48, 80 and
109 give residual `inf` with 127, 128 and 91 of 128 items bad, while 32, 64 and 96 give
5.06e-07 with 1, 6 and 6. `80 mod 32 = 16` — the bucket rule predicts 80 and nothing in that
sweep was chosen to test it.
Orders 128 and 160 fail 0–5 items sporadically, which is a second and probably different
(racy) failure: repeated runs of the same configuration give different counts (`trsmbug.csv`,
float order 64: 7, 7, 4, 4, 5 items over five reps; double order 128 with slack: 1, 0, 0, 0,
1). The N=16-bucket failure is by contrast deterministic and total.

At batch 32 (`q*batch ≈ 30k`, below the threshold) orders 48, 77, 109 and 155 are all clean.
**Both conditions are needed.**

### 4c. Why this matters more than it looks

* It is on the **default** path. `routeq.txt`: the panel shape resolves to `Native:Blocked`
  for double, cfloat and cdouble at every batch ≥ 8, and for float at batch ≥ 128 — with a
  vendor present and no environment variable set.
* The three orders that fail totally, **48, 77 and 109**, include `potrf_cta_max_n<T>()` for
  **double, cfloat (109) and cdouble (77)** — i.e. the natural `nb` for three of the four
  types is exactly a broken one. Float's 155 happens to be clean.
* The whole blocked potrf inherits it: `blocked` at `float n=1024 nb=48 batch=128`, default
  routing, no env, gives residual `inf` and 68 of 128 items with `info != 0`
  (`fillcheck.txt`, `wrongans.csv`). With `BATCHLAS_TRSM_ROUTE=vendor` every cell of
  `wrongans.csv` is clean.
* **Unverified but likely reach beyond potrf:** `ortho.cc:202` and `:289` are trsm's other
  in-tree callers, with a `k × k` Cholesky factor as A (`k` up to 256, so many orders in the
  broken buckets) and an `m × k` basis as B with `m` up to 4096 and batch up to 512, i.e.
  `q*batch` far above the threshold. I did NOT test `Side::Left` or the ortho path; this is
  flagged as a hypothesis, not a result.

### 4d. What Phase 2 should do about it

1. Report it as a trsm defect, not a potrf one. It is reproducible in six lines with
   `trsm_native_blocked` and no potrf at all (`mode_trsmdiff`).
2. Until it is fixed, the blocked driver's correctness tests must run at
   `q * batch > 65536` and at an `nb` whose final V1 block lands in the N=16 bucket, or they
   will be green over the defect. `nb = 64` at `n = 512, batch = 8` — the shape a small test
   would naturally pick — is clean in every cell measured here.
3. The `nb` recommendation in §5 lists correctness alongside every timing.

---

## 5. `nb` and `W` — MEASURED, not reasoned

`blocked.csv` (`run_blocked.sh`) and `blocked2.csv` (`run_blocked2.sh`), summarised by
`analyse.py blocked` and `analyse.py blocked2`. W=128 for the `nb` sweep, batch 128, 4 reps,
worst relative sd 1.3 % in `blocked.csv` and 1.2 % in `blocked2.csv`; the rest under 0.9 %.
Nothing approached the 10 % discard rule and nothing was discarded.

**A note on which rows are usable for tuning.** Many `native`-trsm rows are marked
`<-- WRONG ANSWER` (§4). A wrong answer does not change the *arithmetic issued*: the same
kernels run on the same shapes with the same launch geometry, only the values differ, and
FP32/FP64 throughput on this part does not depend on operand values. So those rows are used
for **`nb` selection** and never quoted as an end-to-end result. Every conclusion below is
cross-checked against a `vtrsm`/`vgemm` row (vendor trsm — correct at every `nb` measured).

### `nb`

| type | recommended `nb` | `cta_max_n<T>()` | why |
|---|---|---|---|
| **float** | **128** | 155 | The trailing gemm's `k` **is** `nb`, and float's only transposed register kernel needs `k >= 128` (`gemm_kernels.cc:476`). Below 128 it is unreachable; above 128 the leaf gets slower for nothing. |
| **double** | **96** | 109 | Flat minimum of the total at 96 at both n; 109 is slower *and* in the broken trsm bucket. |
| **cfloat** | **96** | 109 | Minimum of leaf+panel+trailing at both n; 109 is slower and broken. |
| **cdouble** | **64** | 77 | Minimum at both n in `native` and in `default`; 77 is slower and broken. |

All four are multiples of 32, which is also — by luck, not design — the set that avoids §4b's
N=16 bucket.

**float, n=512, W=128, total ms** (`nativeT` = the `Transpose::Trans` variant):

| nb | 32 | 48 | 64 | 96 | **128** | 155 |
|---|---|---|---|---|---|---|
| default | 2.272 | 1.845 | 1.507 | 1.306 | **1.114** | 1.234 |
| native (vendor-free) | 3.754 | 3.194 | 2.929 | 2.634 | **2.462** | 2.524 |
| nativeT | 3.744 | 3.203 | 2.932 | 2.634 | **1.587** | 2.100 |

**float, n=1024, W=128, correct rows only (`vtrsm*` = vendor trsm + native gemm):**

| nb | 64 | 96 | **128** | 155 |
|---|---|---|---|---|
| vtrsm (ConjTrans) | 19.983 | 17.707 | 16.629 | **16.406** |
| vtrsmT (Trans) | 19.993 | 17.699 | **9.180** | 11.604 |
| vgemm (both vendor) | 9.562 | 7.883 | 7.080 | **7.022** |

The `Trans` substitution is worth **1.00× at nb ≤ 96 and 1.81× at nb = 128** — exactly the
`k >= 128` gate, confirmed end to end on correct answers. It is the single largest lever
found in this campaign, and it costs one enum value.

**double, n=512 / n=1024** (`native` and `vtrsm` agree on the shape of the curve):

| nb | 32 | 48 | 64 | 80 | **96** | 109 |
|---|---|---|---|---|---|---|
| n=512 native | 8.594 | 7.814 | 7.399 | 7.190 | **7.019** | 7.258 (wrong) |
| n=1024 vtrsm | 59.343 | 55.456 | 53.338 | 53.538 | **50.980** | 53.682 |

**cfloat, `native`, leaf+panel+trailing (ms):** n=512 — 7.38 / 6.32 / 5.75 / 5.47 /
**5.22** / 5.40 for nb 32/48/64/80/96/109; n=1024 total — 43.4 / 42.8 / 36.7 / 38.1 /
**34.1** / 37.9.

**cdouble, total ms:** n=512 `native` — 81.3 / 78.6 / **76.2** / 79.5 for nb 32/48/64/77;
n=1024 `native` — 576 / 749 / **550** / 579; n=1024 `default` — 205 / 202 / **190** / 203.

### `W` (the trailing-update column-panel width)

**`W` is not one number: its optimum depends on which gemm serves the trailing update**, and
that is the one knob whose answer differs between the vendor-present and vendor-free builds.
Two brackets were measured, and they are at different `n`, so read them as two curves rather
than one table.

**Coarse bracket, n=1024, nb=96 (64 for cdouble), from `blocked.csv`:**

| W | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| float `native` | **14.914** | 16.347 | 20.022 | 25.907 |
| float `default` (vendor gemm) | 6.190 | **5.941** | 7.859 | 10.889 |
| double `native` | **41.234** | 44.628 | 51.668 | 63.756 |
| cfloat `native` | **31.099** | 34.103 | 40.388 | 50.659 |
| cdouble `native` | **507.7** | 550.2 | 628.8 | 761.3 |

**Fine bracket, n=512, TRAILING STAGE ONLY (the only stage `W` touches), from
`blocked2.csv`.** `vtrsm` = native gemm; `vgemm` = vendor gemm. The W=128 column is the
matching row of the main sweep.

| trailing ms | W=16 | W=32 | W=64 | W=96 | W=128 |
|---|---|---|---|---|---|
| float, native gemm | 1.693 | **1.491** | 1.569 | 1.706 | 1.860 |
| float, vendor gemm | 1.995 | 1.026 | 0.691 | **0.632** | 0.641 |
| double, native gemm | 3.942 | **3.885** | 4.197 | 4.497 | 4.908 |
| double, vendor gemm | 9.469 | **4.440** | 4.855 | 5.710 | 4.900 |
| cfloat, native gemm | 3.127 | **2.839** | 3.109 | 3.377 | 3.819 |
| cfloat, vendor gemm | 1.746 | **1.025** | 1.074 | 1.309 | 1.336 |
| cdouble, native gemm | **52.145** | 54.191 | 59.159 | 63.510 | 68.270 |
| cdouble, vendor gemm | 20.924 | **17.959** | 19.673 | 21.295 | 23.099 |

**Recommendation: `W = 32`**, and `W = 16` for cdouble on a native gemm. For a
vendor-present float build the optimum moves out to `W = 96–128`; if Phase 2 wants one
constant, 32 costs float-with-vendor about 2 % of the *trailing stage* at n=512 (1.026 vs
0.632 is a 1.6× loss on that stage — so this is one place where a per-route constant is
actually justified, and the route is already known at the call site through the injected
seam).

**Why smaller `W` wins, and why the spec's 128 was chosen against the wrong number.** The
discarded upper triangle of the diagonal block is `W²/2` per panel over `ceil(m2/W)` panels
= `m2·W/2` against a useful `m2²/2`, i.e. waste `= W/m2` — **linear in `W`**. That is the
25 % the corrections document computed (`W/m2 = 128/512`), not the spec's 12.5 %, and the
measurement follows it: the curve is monotonic in `W` above 32 in every native-gemm row here.
The turn-around below 32 is launch count and tile efficiency, and it is now bracketed rather
than assumed.

### Where this leaves potrf overall

At the recommended `nb`, W=128, batch 128, n=512, against cuSOLVER's batched potrf on the
same matrices (`vendorpotrf` rows, correct in every cell):

| type | cuSOLVER | blocked, vendor present | blocked, vendor free | vendor-free penalty |
|---|---|---|---|---|
| float (nb 128) | 0.902 | 1.114 (1.23×) | 2.462 (2.73×), **1.587 (1.76×) with `Trans`** | 2.21× → **1.42×** |
| double (nb 96) | 5.955 | 7.021 (1.18×) | 7.019 (1.18×) | **1.00×** |
| cfloat (nb 96) | 1.560 | 2.267 (1.45×) † | 5.146 (3.30×) † | 2.27× |
| cdouble (nb 64) | 22.021 | 27.776 (1.26×) † | 76.241 (3.46×) † | 2.75× |

† These four cells carry `info != 0` on 1–5 of 128 items from the §4 trsm defect. The
*timing* is unaffected (same kernels, same shapes, same launch geometry) but they are not
end-to-end correctness results. The nearest fully-correct rows are cfloat n=512 nb=48
`native` 6.131 ms and cdouble n=512 nb=64 `vtrsm` 115.263 ms, the latter with the slow vendor
panel solve.

Those rows are at `W = 128`, i.e. **before** the `W` retune above; at `W = 32` the trailing
stage is 1.25× (float), 1.26× (double), 1.35× (cfloat) and 1.26× (cdouble) faster on a native
gemm, so the vendor-free column improves further and none of the ratios above is a floor.

And at n=1024, correct rows only: float `vgemm` 7.022 ms against cuSOLVER 7.302 — **the
blocked driver with both primitives routed to the vendor already beats cuSOLVER's own batched
potrf by 1.04×.** That is the ceiling this design is working against, and it is a good one.

---

## 6. What my own instrument could NOT see — reported because a negative is the point

The task asked what I deliberately broke and whether it turned red. `run_breakcheck.sh`
injects three defects into the blocked driver (`PHASE2_BREAK`) and reruns the residual;
results in `breakcheck.csv`, n=256, nb=64, W=128, batch=4, all four types.

| break | what it does | float | double | cfloat | cdouble |
|---|---|---|---|---|---|
| `conj` | trailing gemm uses `Trans` instead of `ConjTrans` | 3.26e-07 (no-op, correct) | 8.10e-16 (no-op, correct) | **1.90e-02 RED** | **1.90e-02 RED** |
| `stride` | every sub-view built with the child's `ld*cols` stride — the `matrix.cc:1839` default | **inf RED**, info 3 | **1.99e+266 RED**, info 3 | **9.39e+25 RED**, info 3 | **6.75e+234 RED**, info 3 |
| `nofold` | the `W×W` diagonal block gemm written straight into `A` instead of into scratch + fold | 3.26e-07 **GREEN** | 8.10e-16 **GREEN** | 4.04e-07 **GREEN** | 5.64e-16 **GREEN** |

`conj` and `stride` behave exactly as they should — `conj` is a no-op for real types *by
definition*, so its staying green there is the check working, not failing.

**`nofold` did not turn red, and that is a finding.** Writing the symmetric product straight
into `A` clobbers the **upper** triangle of every `W×W` diagonal block, which LAPACK
`potrf(Lower)` must leave untouched. My residual is computed over the lower triangle only, so
it is **blind to it by construction** — this repository's recurring blind-guard shape, found
here in my own instrument. Two consequences:

* Phase 2's tests must **poison the opposite triangle and assert it survives**, or the fold
  is unguarded. The same gap already exists in the Phase 1 test
  `PaddedLeadingDimensionAndNonDefaultStride`, which never asserts that rows `n..ld-1` (the
  driver's `L21` region) are untouched.
* `nofold` is also **measurably cheaper** — trailing stage 0.0650 vs 0.0737 ms (float),
  0.0629 vs 0.0745 (cfloat), ~11 % of the trailing update — so the scratch-plus-fold is a
  real tax paid purely for the storage contract, and nothing in a residual-only test will
  ever notice if someone "optimises" it away.

A fourth thing the instrument initially could not see: the residual was first computed on
**batch item 0 only**, which sits at offset 0 and therefore cannot move when the batch stride
is wrong — the `stride` break stayed green until the check was extended to item `batch-1`.
It is now `max(item 0, item batch-1)`.

---

## 7. Two contract facts Phase 2 needs, found the hard way

**(a) The vendor batched `trsm` and `potrf` REQUIRE a pointer array on the view.**
`cublas.cc:1220` calls `A.data_ptrs(ctx)`, and a `MatrixView` built by the 6-arg constructor
has an empty `data_ptrs_` span, so the call throws `"data_ptrs target is null"`
(`src/matrix.cc:2369`) — not a wrong answer, an abort. The blocked driver's `A11` and `A21`
are exactly such views. This harness gives each ROLE its own scratch array (never the
parent's, which is the [FIX-B-trap]); `init_data_ptr_array` recomputes it from that view's
own `data_ptr()`/`stride` on every call. Note the cost: `data_ptrs(ctx)` launches a kernel
**and waits** on every single call (`matrix.cc:2377-2383`), so every vendor batched trsm in
this library already pays a host round-trip per call.

**(b) `MatrixView`'s span runs off the end of the parent.** The 6-arg constructor sizes
`data_` as `stride * batch_size` from the OFFSET pointer (`matrix.cc:1839-1840`), so a
sub-view at `(j,j)` of an `n × n × batch` parent claims `j*ld + j` elements past the
allocation. `potrf_cta_dispatch` never reads `A.data().size()` so it is inert there, and
measured here it is inert for `trsm` and `gemm` too (the `slack` column of `trsmbug.csv`
changes no result). Recorded so that the next person does not have to re-establish it.

---

## 8. Files

| file | what |
|---|---|
| `phase2.cpp`, `build.sh` | the harness |
| `gpu_guard.sh` | copy of `experiments/gpu_guard.sh` from the main checkout |
| `panel.csv`, `run_panel.sh` | OQ5 — panel trsm, 3 routes × 4 types × 3 n × 4 j |
| `panelflat.csv`, `run_panelflat.sh` | the `ld == rows` control for the panel trsm |
| `trail.csv`, `run_trail.sh` | OQ6 — trailing gemm, sub vs flat, 3 routes, C vs T |
| `blocked.csv`, `run_blocked.sh` | nb and W sweeps, per-stage split, end-to-end, `vendorpotrf` reference |
| `blocked2.csv`, `run_blocked2.sh` | the same on CORRECT answers only (vendor trsm), plus the `Trans` variant and W down to 16 |
| `run_diag.sh` | per-item info/residual dump used while isolating §4 |
| `routeq.cpp`, `build_routeq.sh`, `routeq.txt` | what the resolver returns, host-only |
| `routes.txt`, `run_routes.sh` | kernel-name trace (never timed) |
| `breakcheck.csv`, `run_breakcheck.sh` | the deliberate-break validation of §6 |
| `trsmdiff.csv`, `run_trsmdiff.sh` | native vs vendor trsm, first pass |
| `trsmbug.csv`, `run_trsmbug.sh` | sub vs flat, slack, determinism |
| `trsmthresh.csv`, `run_trsmthresh.sh` | the two thresholds of §4b |
| `wrongans.csv`, `run_wrongans.sh` | the route bisection that isolated it to trsm |
| `fillcheck.txt`, `run_fillcheck.sh` | ruling the input in/out as the cause |
| `analyse.py` | all the summary tables above |


---

## 9. What Phase 2 should carry forward, in one list

1. **Do not write `PotrfPanelSolveKernel`.** Use the routed `trsm`. (§2)
2. **Fix, or route around, the native trsm defect first.** It is on the default path, it hits
   three of the four types' natural `nb`, and it makes every correctness test of the blocked
   driver meaningless unless the test runs at `q*batch > 65536` with an `nb` in the N=16
   bucket. (§4)
3. **Trailing update: `alpha = -1, beta = 1`, transB = `ConjTrans` for complex and
   `Transpose::Trans` for REAL types.** The `Trans` spelling is worth 1.81× end to end for
   float at `nb = 128` and 1.00× at `nb <= 96`, and 1.00× for double. (§3b, §5)
4. **`nb` = float 128, double 96, cfloat 96, cdouble 64.** None is `cta_max_n<T>()`. (§5)
5. **`W` = 32 for a native trailing gemm (16 for cdouble), 96-128 for a vendor one — not the spec's flat 128.** The waste is `W/m2`, linear in W. (§5)
6. **The injected-gemm seam is required and is where the money is**, 65–95 % of the driver.
   `routeq` shows why: double already routes native with the vendor present, the other three
   route to the vendor, and a hardcoded `sycl_gemm::gemm_custom` would take double's 1.18×
   win away from float/cfloat/cdouble as a 3–5.5× loss. (§3)
7. **Test the fold by poisoning the opposite triangle**, not by a residual. (§6)
8. **Give every sub-view its own pointer array** if it can reach the vendor. (§7a)
