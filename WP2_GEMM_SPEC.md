# WP2 — GEMM: close the envelope

Output of a 12-agent design pass (5 mapping, 3 designs, 3 adversarial judges, 1 synthesis).
Supersedes §WP2 of `VENDOR_INDEPENDENCE_PLAN.md`, which it contradicts in three places.

## What the plan — and my own first reading — got wrong

**1. There is a SECOND gate, and it is the real one.** `supports()` is correctness-only and
`preferred()` holds every restriction; that much is right, and it makes WP2 look like a pure
routing problem. It is not. `select_kernel_variant` (`src/sycl/gemm_kernels.cc:466`) puts its
*entire* register ladder inside `if constexpr (std::is_same_v<T, float>)`. Double falls to
`max_dim <= 32 ? Direct : Tiled16` (`:510-512`); both complex types to
`max_dim <= 64 ? Direct : Tiled16` (`:514`).

So **"native/register_tiled" for double and complex is Tiled16** — the one-accumulator 16×16
kernel. Widening `preferred()` for complex without touching the selector does not route
complex to a register kernel; it routes it to Tiled16, which
`WP2_WIDE_SCALAR_GEMM_VERDICT.md` measures at **3.2–7.1× slower than cuBLAS**. That is a
straight regression, and the comment at `route_gemm.hh:84-88` currently invites it.

**2. `preferred()` is dead in a default build.** `legacy_unset_default(Op::gemm)` returns a
*forced* `{Vendor, Auto}`, which `route_resolve.hh:78-80` answers before `automatic()` ever
consults `preferred()`. Every widening below is invisible until the default flips — which is
why the flip is last, not first.

**3. The measurement the plan asks for largely already exists.**
`WP2_WIDE_SCALAR_GEMM_VERDICT.md` carries a guarded, warm-clock, serialized table at β=0 **and
β=1** for double / cfloat / cdouble across 256³b512, 512³b128, 1024³b32. Its verdict: land
exactly one kernel — the **64×64×16 macro tile with a 4×4 thread tile** — for double, cfloat
and cdouble, and land **nothing for float** (float measures 0.85–0.93× vs cuBLAS, a loss).
Versus cuBLAS the wide-scalar tile is 1.10–1.15× on double, 1.12× on cdouble, 1.01–1.08× on
cfloat. Versus the in-tree Tiled16 it is 3.6× / 7.4×.

So the complex and double cells need an **integration**, not a sweep.

## What the library actually issues

From the WP1 S7 capture (`scripts/gemm_demand.py`):

```
gemm coverage rows: 2795   (calls: 163438)
routed native today:  30 / 2795 (1.1%)
preferred() would accept: 307 rows / 10571 calls
  -> flipping the default moves ~277 rows off cuBLAS
     call-weighted:  double 9022   float 1549   complex 0
```

Two facts that reframe the work:

- **The native GEMM kernel is essentially never chosen today.** 30 of 2795 rows, and 28 of
  those 30 are forced-variant tests bypassing `preferred()` entirely. The plan's headline
  ("the float-NN-large-square cell is at 88–102% of cuBLAS") describes a cell the library
  almost never asks for.
- **Double is 85% of everything the flip moves**, and nobody expects it to be. Its window is
  the loosest in `preferred()` — `return max_dim <= 512;` with *no transpose test at all*,
  reached after the square + `batch >= 64` gate — and `select_kernel_variant` serves that
  entire window with Tiled16. The double half of the flip is a bet on Tiled16 against cuBLAS
  DGEMM.

(The synthesis agent independently computed 305 rows / 10567 calls against my 307 / 10571.
The two-row discrepancy is unresolved and immaterial to every conclusion here; it is recorded
rather than silently reconciled.)

## Two tracks, deliberately separate

They have different risk profiles and different acceptance criteria, so they do not share
commits.

### Correctness track — vendor-free completeness

| # | Step | Risk |
|---|---|---|
| C1 | Hoist the heterogeneous per-item loop into a portable header (pure move) | none — route diff must be EMPTY |
| C2 | Give the vendor-free facade a heterogeneous arm; stop `supports()` rejecting it | vendor-free only |
| C3 | Record the cost of vendor-free heterogeneous; do **not** gate on it | none |

**Status: C1–C3 landed.** Vendor-free `gemm_tests` went **167/184 → 184/184**, and the
vendor-free suite **24/53 → 25/53** with `gemm_tests` the only suite to leave the failing set
and nothing newly failing. Vendor-present stayed at its documented 52/53, with a route diff of
4 lines that are all `native_route_supported` 0→1 and **zero decisions moved**.

### C3 result — what vendor-freedom costs for heterogeneous batch

`gemm_heterogeneous_benchmark`, float, GPU-guarded, JIT warmed, 10 iterations:

| shape | batch | vendor-present GFLOP/s | vendor-free GFLOP/s |
|---|---|---|---|
| 64×64×32 | 4096 | 6.96 | 6.99 |
| 128×128×32 | 1024 | 7.25 | 7.39 |
| 256×256×32 | 256 | 7.58 | 8.14 |

**Vendor-freedom costs nothing measurable here** — every difference is inside the 2–13%
run-to-run spread (Std/Avg on these rows). That is the expected result and the reason C3 does
not gate: both paths pay the same dominant cost, one kernel launch per batch member.

The number worth acting on is the absolute one. ~7 GFLOP/s against a ~47 TFLOP/s FP32 peak is
roughly 6000× off, and it is launch-bound, not kernel-bound. The single-launch alternative is
buildable without new infrastructure — `KernelMatrixView` already carries `active_rows_` /
`active_cols_` — and is **deferred**: it is a performance project with its own measurement,
not part of closing a correctness gap.

This closed the **17 remaining vendor-free `gemm_tests` failures**, which were all
heterogeneous batch. `gemm_heterogeneous_vendor_impl` (`cublas.cc:60-104`) also carries the
`m==0`/`n==0` skip and the `k==0 → scale(beta)` substitution, so those semantics — currently
vendor-only — come with it.

### Envelope track — performance

| # | Step | Risk |
|---|---|---|
| E1 | Correct the three in-tree claims that would cause a wrong edit | none (comments) |
| E2 | Port the 64×64×16 t4×4 wide-scalar tile into `src/`; wire `select_kernel_variant` | native-vs-native only |
| E3 | Settle **double** first — 85% of the flip | the largest live risk |
| E4 | Float NN and float transposed — expect a **narrowing** as much as a widening | medium |
| E5 | Non-square: unlock the predicated *kernel* first (route diff EMPTY), then the *route* | medium |
| E6 | **The flip**: `legacy_unset_default(Op::gemm)` → `{Auto, Auto}` | highest |

### E2 result — the kernel landed, and the demand measurement that reframes E3–E6

`WP2_WIDE_SCALAR_GEMM_VERDICT.md` §5 named one follow-up as more valuable than any faster
tile: *"A 7.5× win behind a gate that never fires is worth exactly zero"* — so measure
whether BatchLAS's own call sites hit the gate before believing the win. That measurement is
now done, and it changes the plan.

**What landed.** `src/sycl/gemm/register_64x64_k16_wide.hh`, wired into the enum,
`kernel_trace_name`, force-by-name, `select_kernel_variant` and `gemm_custom`. It is the only
register-tiled variant serving a non-float scalar. Verified:

- 16 new forced-variant tests, **all four scalar types**, aligned and ragged, `α=2, β=-1`.
- The kernel provably *runs*: `BATCHLAS_KERNEL_TRACE` shows 8 launches of
  `gemm_sycl_register_64x64_k16_wide` against 8 of `gemm_sycl_tiled16`.
- All five load-bearing PTX properties survived the port, read out of the device image the
  build actually embeds: **zero** `__mulsc3`/`__muldc3`/`call.uni`; **zero** scalar
  `ld.shared` in any fragment path (float `ld.shared.v4`, double and complex `ld.shared.v2` —
  the 16-byte granule holding across scalar widths); vector global staging on the fast path;
  vector epilogue stores; and **zero** `.local` traffic, i.e. no spill.
- Route diff vs `wp2-c2`: **0 decisions removed, 3 added**, all `native,register_tiled` NN for
  `double`/`complex<float>`/`complex<double>` at a shape class that had none, all attributable
  to the new tests. No existing decision moved — which is the acceptance criterion, since the
  kernel lands behind `gemm_use_sycl_custom`'s `Vendor` default.

**What the demand measurement found.** Against the whole suite's coverage capture:

| population | non-float gemm calls | gate fires | rate |
|---|---|---|---|
| full-suite capture | 23 134 | 823 | 3.56% |
| **real demand** (probe rows removed) | **7 223** | **46** | **0.64%** |

The full-suite number is inflated: **2 312 of the 2 795 `gemm` rows come from
`tests/route_gemm_equivalence_tests.cc`'s synthetic `dims[] × batches[]` cross-product**
(lines 119–120), which feeds shapes straight to the resolver and never executes a GEMM. Those
are probes, not demand. Restricted to calls where the problem is genuinely not small
(`max(m,n) >= 128`): **91.6% are blocked by `k < 256`** and **69% by a transpose**.

**That is structural, not test sizing.** The dominant internal GEMM here is a *panel update* —
large m, large n, small k — and k is the blocking factor, clustered at 1/8/32/48/96/136, a
tuning constant that does not grow with the problem. `min_dim` takes the min over k, so for
that population the gate cannot fire at *any* problem size.

**And no single relaxation rescues it: zero calls are blocked by the k floor alone.** Every
large-m,n small-k call is also transposed or ragged. So:

- **E3/E4/E5 cannot deliver an internal win by widening a predicate.** They can only move the
  large-square-aligned-NN cells, which real demand barely occupies. They are still worth doing
  for the **public API** — a user calling `batchlas::gemm` with a large square complex matrix
  is a normal thing to do, and vendor-free that call is 3.6–7.7× better than it was — but the
  claim "this closes the vendor gap for BatchLAS's complex work" would be false.
- **The deferred item is now the main one.** ConjTrans plus a predicated path for wide scalars
  is what the panel-update population actually needs, and the verdict doc's own numbers say
  the complex fallback is 3.6–7.7× off the vendor there. That is a new kernel, not a routing
  change, and it should be ranked ahead of E4/E5.

### E3 result — double is not the flip's risk, it is the flip's win

E3 was written as *"settle **double** first — 85% of the flip, the largest live risk."* Both
halves were wrong, in opposite directions.

**The premise was probe rows.** The "85% of the flip / 9022 double calls" figure came from a
table that was ~83% `route_gemm_equivalence_tests` probes. `scripts/gemm_demand.py` now takes
`--minus=<probe-capture>`; on real demand `preferred()` accepts **1 row / 4 calls**. Correcting
only for the `batch < 64` blocker — which is *test-scale*, since batch is the user's parameter
and this suite runs 1–8 — the flip would move ~666 double and ~690 float calls, and **zero
complex**.

**And it is not a risk.** Of those 666 double calls, **585 land on `Tiled16`, 57 on `Direct`,
and only 24 on the wide tile WP2 measured** — so the real question was never about the new
kernel. It was *"does `Tiled16` beat cuBLAS DGEMM at n = 34..200?"*, which nothing in the tree
had measured. Measured now: RTX 4090, batch 512, median of 3, **both betas**, warm JIT,
`gpu_guard`, spreads **0.0–0.3%**.

| n | native kernel | cuBLAS | native | ratio |
|---|---|---|---|---|
| 4 | Direct | 16.5 | 58.4 | **3.55×** |
| 8 | Direct | 86.1 | 388 | **4.51×** |
| 16 | Direct | 227 | 888 | **3.92×** |
| 24 | Direct | 440 | 1127 | **2.56×** |
| 32 | Tiled16 *(was Direct)* | 982 | 1213 | **1.23×** |
| 48 | Tiled16 | 633 | 1204 | **1.90×** |
| 64 | Tiled16 | 1151 | 1214 | 1.05× |
| 96 | Tiled16 | 903 | 1330 | **1.47×** |
| 136 | Tiled16 | 698 | 1215 | **1.74×** |
| 200 | Tiled16 | 829 | 1270 | **1.53×** |
| 256–512 | wide-64x64 | 1239–1246 | 1399–1411 | 1.13× |

**Native wins every cell in the accepted window.** Corroboration, since margins this size
deserve it: the endpoints reproduce an independent prior measurement — vendor 1244 GFLOP/s at
512³ against `WP2_WIDE_SCALAR_GEMM_VERDICT.md`'s 1232–1247 for cuBLAS DGEMM, and native 1411
against its 1413–1415 for the same tile. Peak vendor DGEMM observed is 1246 GFLOP/s, inside
the ~1450 FP64 ceiling a 4090 can reach, so these are FP64 numbers and not something else.

**Saturation, checked rather than asserted.** Sweeping batch 64 → 8192 at the two largest
margins, both arms plateau by 2048 and the ratio is *stable or rising* — n=48 goes
1.58× → 1.96×, n=136 1.82× → 1.75×. Launch overhead would shrink the ratio toward 1 as batch
grows; it does not, so this is an algorithm difference.

**One routing defect found and fixed.** n=32 was the *only* losing cell (0.92–0.96× at batch
4096), and it sat exactly on `select_kernel_variant`'s `max_dim <= 32 ? Direct : Tiled16`
boundary for double. Tiled16 beats Direct there by 1.08× at batch 512 and 1.29× at 4096, both
betas, outside spread; Direct still wins clearly at n=24 (1.37–1.64×) and n=25 is a wash. The
boundary moved to `max_dim <= 24`, chosen where the evidence is unambiguous rather than at the
first sign of crossover, and n=32 became a 1.14–1.23× win. That is a native-vs-native kernel
choice, so it moves no `Route` and the route diff stays clean.

**What this does not say.** Every number here is square, NN, aligned, real-typed. It says the
double *window as `preferred()` currently defines it* is safe to route natively — not that
GEMM generally is. Complex is still refused outright by `preferred()`, and the panel-update
population (non-square, transposed) is untouched by any of it.

### E4 result — float narrows, and the bigger win was in the selector

E4 was written as *"expect a **narrowing** as much as a widening"*. That was right. Measured
across all four regions of float's window — RTX 4090, square, median of 3, both betas,
`gpu_guard`, warm JIT — **40 cells argue to narrow and none argue to widen**.

| window | verdict | measured |
|---|---|---|
| NN `max_dim <= 32` | **keep** | n=8 **1.46×**, n=16 **1.31×**, n=32 **1.08×** |
| NN `128..512` | **removed** | n=128 0.97×, n=192 0.40×, n=256 0.87×, n=384 0.79×, n=512 0.91× |
| transposed `128..512` | **removed** | all 30 cells 0.34–0.55×, across TN, NT and TT |

The NN result is not an unsaturated artefact — the ratios are flat across batch 128 / 512 /
1024. The transposed result is not a fallback effect either: TN runs its own
`register_128x32_k32_tn` kernel, traced and confirmed. That family simply plateaus near 15–18
TFLOP/s while cuBLAS SGEMM reaches 45+.

**Narrowing costs the vendor-free build nothing.** `preferred()` only orders routes that both
exist; `resolve_route` falls back to any *supported* native route when the vendor is absent
(`route_resolve.hh:60-62`). This only stops a vendor-**present** build choosing a slower kernel.

**The selector fix was worth more than the window change.** `select_kernel_variant` sent
squareish float that could not use the 128×128 fast path to the *generic* 128×32×32 route,
behind a comment saying the 128×128 **predicated** path "has not been benchmarked against the
generic route below, so misaligned work keeps its existing kernel until that measurement
exists." Benchmarked now, the predicated path wins everywhere tried:

| case | generic | predicated | gain |
|---|---|---|---|
| n=192 | 9 781 | 18 000 | 1.84× |
| n=320 | 12 188 | 25 288 | 2.07× |
| n=1056 | 15 065 | 33 314 | 2.21× |
| n=256, ld+2 | 7 237 | 23 966 | **3.31×** |
| n=512, ld+2 | 8 399 | 36 862 | **4.39×** |

The gain grows with n, which is what a per-tile predication cost looks like against a route
whose throughput has plateaued. **The unaligned-leading-dimension case gains most**, and that
is the one that matters for real demand: a panel is a sub-view carrying its parent's ld, so
unaligned ld is exactly what the factorisations hand to `gemm`. This is E5's prerequisite
("unlock the predicated *kernel* first") answered for float.

**One in-tree claim aged out.** `register_128x128.hh` records "43.6 TFLOP/s against cuBLAS
SGEMM's 43.9" at 512³ b512 — parity. Re-measured: native **43.5**, which reproduces exactly;
cuBLAS **47.3**, which does not. The vendor moved, presumably a cuBLAS upgrade, and the parity
claim did not survive it. Worth noting as a class of hazard: a ratio recorded against a vendor
is only as durable as that vendor's version.

**Where this leaves E6.** With E3 and E4 applied, the flip's remaining scope is: `double` from
n=4 to 512 (a 1.05–4.51× win) and `float` NN at `max_dim <= 32` (1.03–1.46×). Everything else
now prefers the vendor on measurement rather than on assumption.

## Non-negotiable measurement rules

Every one of these is here because it has already cost this project real time:

- **β=0 AND β=1 on both arms.** A prior kernel scored 26 instead of 41 TFLOP/s with an
  identical inner loop, purely because the epilogue made the `beta != 0` read of C one
  scattered transaction per lane.
- **Compare only at saturation**, and verify saturation rather than asserting it. An
  unsaturated ratio is overhead, not algorithm.
- **Warm the JIT.** A cold first run once fabricated an entire 3.7× regression.
- **Absolute sanity anchors before any ratio is believed:** vendor float must reach ~45–47
  TFLOP/s at 512³b512; vendor double must never exceed ~1.45 TFLOP/s (a 4090 is 1/64 FP64); a
  float number near 80 TFLOP/s means TF32, not FP32.
- **A win inside run-to-run noise is not a win.** WP1 nearly reported a 10.9% `syr2k`
  improvement that was noise — the spread at that shape was 13%. Report the spread.
- **Never compare against the in-tree fallback and call it a win over the vendor.** The
  complex 7× figure is exactly that error; against cuBLAS the same kernel is 1.01–1.08×.
- **`experiments/gpu_guard.sh` for every number.** It refuses to start when the card is busy,
  which it has already done during this work package.

## Route-diff discipline, with one correction

Every routing step enumerates its intended moves in advance and `scripts/route_diff.sh` must
match them line for line.

**A pre-flip `preferred()` edit does NOT produce an empty diff**, contrary to the obvious
assumption: `tests/gemm_tests.cc` (`RouteAdapterAutoHonoursTheMeasuredWindow`) sets
`BATCHLAS_GEMM_VARIANT=auto` and drives `resolve_gemm_route` through the instrumented path,
so `preferred()` is exercised and recorded even while the production default is Vendor.

`tests/route_gemm_equivalence_tests.cc` pins the current decision against a transcribed
replica of the *legacy* behaviour, with a `ReplicaIsFaithful` test so the replica cannot drift
and pass vacuously. Any widening **will** fail it. That is the mechanism working: it forces
each widening to be declared rather than slipped in.

## Deferred, with reasons

- **ConjTrans for complex.** No register-tiled variant of any type supports ConjTrans — every
  TN/NT/TT launcher passes `Transpose::Trans`, and no instantiation in the built `.so` carries
  `ConjTrans`. For complex this is the transpose that matters (herk/her2k/hemm all issue it),
  so this cell needs a **new kernel**, not a routing change.
- **TF32 / `ComputePrecision != Default`.** `supports()` rejects it; a separate track.
- **`split_k`.** Compiled but behind three simultaneous gates (name-only selection,
  `BATCHLAS_GEMM_EXPERIMENTAL`, and a predicate requiring float/NN/`m,n,k>=256`/`m%128`/
  `n%32`/`k%128`). Ungating is its own measurement.
- **Two dead enum entries.** `Tiled128x32RegisterK32` is unreachable by the selector *and* by
  name (`gemm_kernels.cc:217-218` returns false for it), and
  `launch_register_128x32_k32_variant` has no caller. Worth deleting so the enum count matches
  the reachable count.
