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

This closes the **17 remaining vendor-free `gemm_tests` failures**, which are all
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
