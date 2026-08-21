# WP4 step 0.2 — the real per-work-group SLM ceiling, measured

Everything here was run on **GPU 0** (RTX 4090, sm_89) under `experiments/gpu_guard.sh`.
No library file was modified and no library rebuild was needed: every probe is a
standalone SYCL program built with `build_probe.sh`, whose flags are copied from
`experiments/wp4_complex/gpu1/build_bench.sh` (which copied them from
`build/benchmarks/CMakeFiles/gemm_benchmark.dir/flags.make`).

## Verdict, in four numbers

| | value | how it was established |
|---|---|---|
| `sycl::info::device::local_mem_size` | **101,376** | `slm_probe`, direct query |
| Hard cap on **static + dynamic** shared per block | **101,376** | `slm_probe_nostatic`, bisected, zero static shared |
| Largest `local_accessor` byte count that launches when the kernel also carries 256 B of static shared | **101,120** | `slm_probe`, bisected at wg = 32/64/128/256/512/1024 — identical at all six |
| blocks/SM limited by shared memory | `floor(102400 / (dynamic + static + 1024))` | `ncu`, matches on all 11 sizes in `occ_sweep.csv` |

**The spec's 49,152 / 45,056 is refuted by direct measurement**, confirming W1.
`cmake/BatchLASDetectSYCL.cmake:44-45` hardcodes 49152 for any `^nvidia_gpu_sm_[0-9]+$`
and `build/include/batchlas/device_limits.hh:23` carries it forward; the detection
routine never queries `local_mem_size`. 97,280 B launches and computes correctly
(`maxn_fitcheck.csv`), so the spec's "48 KB stays the hard per-work-group ceiling"
(spec:267) is false on this device.

## The finding nobody was looking for: a launch HOLE at 48 KB

`slm_probe` found 49,152 B **failing** at wg = 32 while 45,056 and 65,536 both passed.
It reproduces, at every work-group size, and it is not noise.

* Measured hole, for a kernel with 256 B of static shared: dynamic request in
  **(48,896, 49,152]** — `scan_hole_boundary.csv`, boundary located to 8 B at wg =
  32/64/128/256/1024, identical at all five.
* Mechanism: CUDA's non-opt-in per-block limit is 49,152 B for **static + dynamic**.
  The UR CUDA adapter raises `MaxDynamicSharedMemorySize` only when the *dynamic*
  request alone exceeds 49,152. In the band where `static + dynamic > 49152 ≥ dynamic`
  it neither fits nor opts in → `CUDA_ERROR_INVALID_VALUE` at `enqueueKernelLaunch`
  (`unified-runtime/source/adapters/cuda/enqueue.cpp:439`).
* Static shared measured at exactly 256 B by `ncu`
  (`launch__shared_mem_per_block_static`), and the hole is exactly 256 B wide.
  A kernel with no group collectives has zero static shared, no hole, and a ceiling
  of 101,376 — `slm_probe_nostatic`, which is the control that confirms the model.

**The attribute is sticky per kernel function, so the failure is order-dependent.**
`slm_hole_order` demonstrates it on one kernel name:

```
--- COLD (no warm-up launch) ---
float n=110 49064: FAIL unknown internal error
padded      49408: OK
float n=110 49064 (after pad): OK
--- WARM (65536 launched first, same kernel function) ---
warm 65536      : OK
float n=110 49064: OK
```

This matters for potrf specifically: the CTA kernel is templated on
`<T, NB, TS, Scope>` and takes `n` as a **runtime** argument, so one CUfunction serves
every `n`. A process that happens to run a large `n` first will silently work; a
process that starts at the bad `n` will fail. That is an intermittent failure, not a
deterministic one, and the residual test would only see it on a cold run.

**Required mitigation (cheap):** when the computed request lands in `(48128, 49152]`,
round the `local_accessor` size up to **49,408**. Verified: 49,064 fails cold, 49,408
passes cold (`maxn_fitcheck.csv`, last two rows). The lower bound 48,128 is
`49152 − 1024`, conservative against a potrf static-shared figure that is not known
until the kernel exists. Occupancy is unaffected — both give 2 blocks/SM.

Which `n` are at risk, under the spec's own sizing formula:

| type | at-risk `n` (request bytes) |
|---|---|
| `float` | **110** (49,064) — measured to fail cold |
| `double` | none |
| `complex<float>` | none |
| `complex<double>` | 55 (48,624) — only if potrf's static shared ≥ 528 B |

Note also that **49,152 exactly cannot be launched cold at all** on this stack, which
is worth knowing before anyone writes it as a budget constant.

## Occupancy — measured, not assumed

`occ_sweep.csv`, one `ncu` run per (bytes, wg). The spec's `shared_per_SM ≈ 102400`
(spec:284) is **correct** — `cudaDeviceProp.sharedMemPerMultiprocessor = 102400`
(`devprop.log`) — but the spec's formula omits `reservedSharedMemPerBlock = 1024`
and the kernel's static shared, both of which are inside the divisor.

| local_accessor bytes | blocks/SM (shared-limited) | max warps/SM at wg=128 |
|---|---|---|
| 8,192 | 10 | 83.3 % |
| 12,800 | 7 | 58.3 % |
| 17,066 | 5 | 41.7 % |
| 25,600 | 3 | 25.0 % |
| 32,768 | 3 | 25.0 % |
| 45,056 | **2** | 16.7 % |
| 49,408 | 2 | 16.7 % |
| 65,536 | **1** | 8.3 % |
| 81,920 | 1 | 8.3 % |
| 97,280 | **1** | 8.3 % |
| 101,120 | 1 | 8.3 % |

The register limit in that CSV is the *probe's* (38 regs/thread), not potrf's, and is
not predictive. `launch__occupancy_limit_shared_mem` is, because it depends only on
the shared request and the carveout the driver picks — which was 102,400 in every row
but one (8,192 B at wg = 256 dropped to a 65,536 carveout).

So `gesvdj_cta.cc:1011-1016` is right on the substance: **occupancy, not the hard cap,
is the binding constraint.** Raising the budget from 45,056 to 97,280 buys ~1.47× in
`n` and costs exactly half the resident blocks.

## Re-derived `potrf_cta_max_n<T>()`

Formula: spec §4.1 **plus the W9 fix**, since the spec's formula has no `off[]` term:

```
LDA            = n | 1
slm_per_matrix = LDA*n*sizeof(T) + NB*sizeof(real_t) + 64
off_bytes      = 4 * ceil_div(n - nb, TS)          # W9, corrections doc:515
slm_per_wg     = G * slm_per_matrix + off_bytes
```

**`off[]` placement: one copy per WORK-GROUP, not per matrix.** It is a function of
`(m2, TS)` only, and `m2 = n − j − ib` is work-group-uniform, so all `G` matrices
decode the same table (corrections doc, Open question 7). At `G = 1` — which is where
`potrf_cta_max_n<T>()` is evaluated — the two placements are numerically identical;
the choice only saves `G−1` copies in the packed small-`n` case. It still needs a
writer at the top of every panel and a barrier before the first read; that is step 1.3,
not this step.

Constants from spec §3.1 (spec:177). `complex<double>` has `NB = 8`, not 16 —
spec:278's own spot-check used 16 and is wrong.

### At the recommended budget 97,280 (= runtime 101,376 − 4,096)

| `T` | `max_n` | fits | first miss |
|---|---|---|---|
| `float` | **155** | `155*155*4 + 16*4 + 64 + 140 = 96,368` ≤ 97,280 | `n=156`: `157*156*4 + 64 + 64 + 140 = 98,236` |
| `double` | **109** | `109*109*8 + 16*8 + 64 + 96 = 95,336` ≤ 97,280 | `n=110`: `111*110*8 + 128 + 64 + 96 = 97,968` |
| `complex<float>` | **109** | `109*109*8 + 16*4 + 64 + 96 = 95,272` ≤ 97,280 | `n=110`: `111*110*8 + 64 + 64 + 96 = 97,904` |
| `complex<double>` | **77** | `77*77*16 + 8*8 + 64 + 140 = 95,132` ≤ 97,280 | `n=78`: `79*78*16 + 64 + 64 + 140 = 98,860` |

All four `fits` rows were **launched cold, in their own process, and returned the
correct answer** (`maxn_fitcheck.csv`). These are the corrections doc's 155/109/109/77,
now confirmed by execution rather than by arithmetic.

The `off[]` term costs **0** in `n` at this budget for every type — but it must still
be in the formula, or the OOB write W9 describes is real.

Full ladder at every candidate budget, with the boundary arithmetic on both sides:
`max_n_derivation.txt`.

| budget | blocks/SM | float | double | cfloat | cdouble |
|---|---|---|---|---|---|
| 11,520 | 8 | 53 | 37 | 37 | 26 |
| 15,786 | 6 | 61 | 43 | 43 | 31 |
| 19,200 | 5 | 68 | 48 | 48 | 33 |
| 24,320 | 4 | 77 | 54 | 54 | 38 |
| 32,853 | 3 | 89 | 63 | 63 | 45 |
| 45,056 (spec) | 2 | 105 | 74 | 74 | 52 |
| 49,920 | 2 | 111 | 78 | 78 | 55 |
| **97,280 (recommended fit)** | **1** | **155** | **109** | **109** | **77** |
| 100,352 | 1 | 157 | 111 | 111 | 79 |
| 101,120 (hard) | 1 | 158 | 111 | 111 | 79 |

## Recommendation

**1. `potrf_cta_max_n<T>()` / `supports()` uses `runtime LOCAL_MEM_SIZE − 4096 = 97,280`
→ 155 / 109 / 109 / 77.**

Reasoning. `supports()` is a correctness question — can this kernel produce the right
answer for this shape — and the answer is measured yes at all four of those `n`.
Shipping 105/74/74/52 would make `supports()` return false for a band the kernel can
hold, and per `route_resolve.hh:60-63` that leaves **float `n` in 106..155 with no
route at all in a vendor-free build**, which is the entire point of WP4. The 4,096-byte
reserve is not superstition: it is the repo's own convention
(`batchlas_subgroup_workspace_budget_bytes`, `cmake/BatchLASDetectSYCL.cmake:57-67`) and
it covers the potrf kernel's static shared, which is unknown until the kernel is
written. It must read the **runtime** query, never `device_limits.hh` — that constant is
a hardcoded 49,152 and is wrong by 2.06×.

I do **not** recommend 101,120 (the measured hard ceiling) even though it launches:
it buys 3 more `n` for `float` and 0 for `double`/`cfloat`/`cdouble` over 100,352, and
it leaves zero headroom for whatever static shared the real kernel ends up carrying —
at which point the ceiling moves and `supports()` starts lying.

**2. Add the hole-avoidance pad to whatever allocates the tile.** If the computed
request is in `(48128, 49152]`, allocate 49,408 instead. Without it `float n = 110`
fails on a cold process and passes on a warm one. This is a correctness fix, so it
belongs with `supports()`, not with the tuning.

**3. `preferred()` is a different and much smaller window, and it must not be
guessed here.** At the fit ceiling the kernel runs at **1 block/SM and 8.3 % of peak
warp occupancy at wg = 128** — 4 warps against a 48-warp SM. The spec already says as
much for a different reason ("The fit ceiling is not the useful ceiling", spec:290) and
`gesvdj_cta.cc:1011-1016` reached the same conclusion for the same hardware reason.

The occupancy ladder above is the menu the §10.3 grid should be cut against. As a
*starting hypothesis only* — not a shipped claim — the ≥4 blocks/SM budget of 24,320
(float 77 / double 54 / cfloat 54 / cdouble 38) is where the SM stops being starved of
blocks; ≥2 blocks/SM (49,920) is the loosest defensible line. Per W2 `preferred()`
returns false everywhere at merge anyway, exactly as trsm shipped
(`route_trsm.hh:53-55`), so the honest answer is: **fit ceiling large (97,280),
preferred window measured later and expected to be much smaller.**

**4. Re-cut §10.3's `n` list.** spec:593's "plus 52/74 exactly at the per-type
ceilings" now measures the wrong boundary. The boundaries are 155/109/109/77, and the
grid should straddle the occupancy cliffs at 24,320 / 32,853 / 49,920 as well, because
that is where the answer will actually change.

## What is in this directory

| file | what it is |
|---|---|
| `build_probe.sh` | standalone SYCL build; `build_probe.sh <src> <out>` |
| `slm_probe.cpp` / `slm_probe_gpu0.log` | device-info dump + bisection at six work-group sizes |
| `slm_probe_nostatic.cpp` / `nostatic_gpu0.log` | control with zero static shared — ceiling 101,376, no hole |
| `slm_scan.cpp` / `scan_wg32_48k.csv` / `scan_hole_boundary.csv` | dense scans that located the hole to 8 B |
| `slm_hole_order.cpp` | the order-dependence demonstration (cold FAIL, warm OK) |
| `slm_occ.cpp` / `run_occ_sweep.sh` / `occ_sweep.csv` | one launch per size, profiled by `ncu` |
| `devprop.cu` / `devprop.log` | `cudaDeviceProp` — the numbers SYCL does not expose |
| `derive_max_n.py` / `max_n_derivation.txt` | the arithmetic, both boundaries, all ten budgets |
| `run_maxn_fitcheck.sh` / `maxn_fitcheck.csv` | every candidate `max_n` launched **cold**, one process each |

### Falsifiability

The correctness assertion in these probes was checked by breaking it. `slm_scan.cpp`'s
SLM store was edited to write `0` at one byte, rebuilt, and run at 45,056 B: it
reported `WRONG ANSWER` (red). The edit was then reverted, rebuilt, and re-run at the
same size: `ok=1` (green). The probe is therefore capable of distinguishing "launched"
from "launched and computed the right thing", which is the distinction the whole
bisection rests on.
